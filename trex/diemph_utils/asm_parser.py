import copy

ADDR_IDX = 0
ASM_IDX = 1
RAW_IDX = 2
CFG_IDX = 3


def skip(operand, chars):
    while operand[0] in chars:
        operand = operand[1:]
        if len(operand) == 0:
            break
    return operand


def parse_operand(operator, location, operand1, ori, diemph_ctx, stack_vars, instr_idx):
    operand1 = operand1.strip(' ')
    ret = ''

    if ('offset' in operand1 or 'off_' in operand1
            or 'loc_' in operand1 or 'unk_' in operand1
            or ('sub_' in operand1 and 
            ('lea' in operator or 'mov' in operator))):
        # operand1 = operand1.replace('offset ', '')
        return ' num '
    if 'xmmword ptr' in operand1:
        ret += 'xmmword ptr '
        operand1 = operand1.replace('xmmword ptr ', '')
    if 'dword ptr' in operand1:
        ret += 'dword ptr '
        operand1 = operand1.replace('dword ptr ', '')
    if 'qword ptr' in operand1:
        ret += 'qword ptr '
        operand1 = operand1.replace('qword ptr ', '')
    if 'word ptr' in operand1:
        ret += 'word ptr '
        operand1 = operand1.replace('word ptr ', '')
    if 'byte ptr' in operand1:
        ret += 'byte ptr '
        operand1 = operand1.replace('byte ptr ', '')
    if 'short ptr' in operand1:
        ret += 'short ptr '
        operand1 = operand1.replace('short ptr ', '')
    if 'ptr' in operand1:
        ret += 'ptr '
        operand1 = operand1.replace('ptr ', '')

    operand1 = operand1.replace('-', '+')

    operand1 = operand1.strip()

    if operand1[0:3] == 'cs:':
        ret += 'cs : '
        operand1 = operand1[3:]
    if operand1[0:3] == 'ss:':
        ret += 'ss : '
        operand1 = operand1[3:]
    if operand1[0:3] == 'fs:':
        ret += 'fs : '
        operand1 = operand1[3:]
    if operand1[0:3] == 'ds:':
        ret += 'ds : '
        operand1 = operand1[3:]
    if operand1[0:3] == 'es:':
        ret += 'es : '
        operand1 = operand1[3:]
    if operand1[0:3] == 'gs:':
        ret += 'gs : '
        operand1 = operand1[3:]

    operand1 = operand1.strip()

    if (operator[0] == 'j') and not isregister(operand1):
        return 'hexvar '

    if (operand1[0:4] == 'loc_' or operand1[0:4] == 'off_'
        or operand1[0:4] == 'unk_' or operand1[0:4] == 'sub_'
        or operand1[0:4] == 'arg_' or operand1[0:4] == 'def_'
            or operand1[0:4] == 'var_'):
        ret += 'hexvar '
        operand1 = operand1[4:]
        # skip characters 0-9
        operand1 = skip(operand1, '0123456789ABCDEF')
    operand1 = operand1.strip()

    if len(operand1) == 0:
        return ret

    if operand1[0:6] == 'locret':
        ret += 'hexvar '
        operand1 = operand1[6:]

    if operand1[0] == '(' and operand1[-1] == ')':
        ret += '( num ) '
        return ret
    if operator == 'lea' and location == 2:
        # handle some address constants
        if not ishexnumber(operand1) and not isaddr(operand1):
            return 'num '
    if operator == 'call' and location == 1:
        if len(operand1) > 3:
            return 'hexvar '
    if operator == 'extrn':
        return 'hexvar '
    operand1 = operand1.strip()

    if ishexnumber(operand1):
        ret += 'num '
        return ret
    elif ispurenumber(operand1):
        ret += 'num '
        return ret
    if isaddr(operand1):
        params = operand1[1:-1].split('+')
        if stack_vars is not None:
            if len(params) == 2 and 'rbp' in params[0]:
                var_id = operand1[1:-1].strip()
                if var_id in stack_vars:
                    stack_vars[var_id].append((instr_idx, location))
                else:
                    stack_vars[var_id] = [(instr_idx, location)]

        for i in range(len(params)):
            params[i] = params[i].strip()
            if ishexnumber(params[i]):
                params[i] = 'num '
            elif ispurenumber(params[i]):
                params[i] = 'num '
            elif params[i][0:4] == 'var_':
                params[i] = 'hexvar '
            elif params[i][0:4] == 'arg_':
                params[i] = 'hexvar '
            elif not isregister(params[i]):
                if params[i].find('*') == -1:
                    params[i] = 'hexvar '
                else:
                    # split by *
                    sub_params = params[i].split('*')
                    for j in range(len(sub_params)):
                        sub_params[j] = sub_params[j].strip()
                        if ispurenumber(sub_params[j]):
                            sub_params[j] = 'num '
                        elif ishexnumber(sub_params[j]):
                            sub_params[j] = 'num '
                    params[i] = ' * '.join(sub_params)
            elif isregister(params[i]):
                params[i] = norm_register(params[i])
        s1 = ' + '
        ret += '[ '+s1.join(params)+' ] '
        return ret

    if not isregister(operand1) and len(operand1) > 4:
        ret += 'hexvar '
    elif not isregister(operand1):
        ret += operand1+' '

    if isregister(operand1):
        ret += norm_register(operand1) + ' '

    return ret


# handle ida code to better quality code for NLP model
def parse_asm(code, ori, diemph_ctx=[]):
    annotation = None
    operator, operand = None, None
    operand1, operand2, operand3 = None, None, None
    if code.find(';') != -1:
        id = code.find(';')
        annotation = code[id+1:]
        code = code[0:id]
    if code.find(' ') != -1:
        id = code.find(' ')
        operand = code[id+1:]
        operator = code[0:id]
    else:
        operator = code
    if operand != None:
        if operand.find(',') != -1:
            strs = operand.split(',')
            if len(strs) == 2:
                operand1, operand2 = strs[0], strs[1]
            else:
                operand1, operand2, operand3 = strs[0], strs[1], strs[2]
        else:
            operand1 = operand
            operand2 = None
    # if not ori and operand1 != None:
    #     if 'sub_' in operand1 and 'jmp' in operator:
    #         operator = 'call'

    if operand1 != None:
        operand1 = parse_operand(
            operator, 1, operand1, ori=ori, diemph_ctx=diemph_ctx, stack_vars=None, instr_idx=0)
    if operand2 != None:
        operand2 = parse_operand(
            operator, 2, operand2, ori=ori, diemph_ctx=diemph_ctx, stack_vars=None, instr_idx=0)
    if operand3 != None:
        operand3 = parse_operand(
            operator, 3, operand3, ori=ori, diemph_ctx=diemph_ctx, stack_vars=None, instr_idx=0)

    # if not ori:
    #     if 'endbr' in code:
    #         return 'SKIP', None, None, None, None
    #     if ';' == code.strip()[0]:
    #         return 'SKIP', None, None, None, None
    #     if operand1 is not None:
    #         if 'push' in operator and 'bp' in operand1:
    #             return 'SKIP', None, None, None, None
    #         if operand2 is not None:
    #             if 'mov' in operator and 'ax' in operand1 and 'fs:xxx' in operand2:
    #                 return 'SKIP', None, None, None, None

    return operator, operand1, operand2, operand3, annotation


normalize = {    
    # 'rsi': 'esi',
    # 'rdi': 'edi',
}


def isregister(x):
    if x in normalize:
        x = normalize[x]
    registers = ['rax', 'rbx', 'rcx', 'rdx', 'rsi', 'rdi', 'rbp',
                 'rsp', 'r8', 'r9', 'r10', 'r11', 'r12', 'r13', 'r14', 'r15']
    # registers=['rax','rbx','rcx','rdx','rsi','rdi','rbp','rsp','r8','r9','r10','r11','r12','r13','r14','r15']
    return x in registers


def norm_register(x):
    if x in normalize:
        x = normalize[x]
    return x


def ispurenumber(number):
    if number[0] == '+' or number[0] == '-':
        number = number[1:]
    # whether every char is digit
    for i in range(len(number)):
        if str.isdigit(number[i]):
            continue
        else:
            return False
    return True


def isaddr(number):
    return number[0] == '[' and number[-1] == ']'


def ishexnumber(number):
    if number[0] == '+' or number[0] == '-':
        number = number[1:]
    if number[-1] == 'h':
        for i in range(len(number)-1):
            if str.isdigit(number[i]) or (number[i] >= 'A' and number[i] <= 'F'):
                continue
            else:
                return False
    else:
        return False
    return True


class FunctionParser:

    def __init__(self, ori, instr_list):
        self.ori = ori
        self.instr_list = instr_list
        self.stack_vars = {}
        self.untouched_stack_vars = {}
        self.instr_parsed = self._parse()
        self.largest_register = self._find_largest_register()
        self.stack_vars = {k: v for k, v in self.stack_vars.items() 
                                if k not in self.untouched_stack_vars}
        for k, v in self.stack_vars.items():
            self.stack_vars[k] = [(idx, loc) for idx, loc in v if idx < 150]
        self.stack_var_list = self._sort_stack_var_list()
        # print()

    def _sort_stack_var_list(self):
        stack_var_list = [(k,sorted(v)) for k, v in self.stack_vars.items()]
        stack_var_list = sorted(stack_var_list, key=lambda x: x[0][1])
        return stack_var_list


    def get_stack_var_list(self):
        return self.stack_var_list

    def rewrite(self, to_rewrite, new_var, instr_parsed=None):
        if instr_parsed == None:
            instr_parsed = copy.deepcopy(self.instr_parsed)
        if to_rewrite not in self.stack_vars:
            print('ERROR: rewrite: %s not in stack_vars' % to_rewrite)
            return instr_parsed
        locations = self.stack_vars[to_rewrite]
        for loc in locations:
            instr_parsed[loc[0]][loc[1]] = new_var
        return instr_parsed

    def finalize(self, instr_parsed=None):
        if instr_parsed == None:
            instr_parsed = self.instr_parsed
        ret = []
        for instr in instr_parsed:
            ret.append((instr[0], instr[1], instr[2], instr[3], instr[4]))
        return ret

    def _find_largest_register(self):
        instr_str = ' '.join(
            [i for i in self.instr_list if ';' != i.strip()[0]])
        registers = ['r15', 'r14', 'r13', 'r12', 'r11', 'r10', 'r9', 'r8']
        for reg in registers:
            if reg in instr_str:
                return reg
        return 'r8'

    def _parse(self):
        ret = []
        for idx, instr in enumerate(self.instr_list):
            instr = instr.strip()
            annotation = ""
            operator, operand = None, None
            operand1, operand2, operand3 = None, None, None
            if instr.find(';') != -1:
                id = instr.find(';')
                code = instr[0:id]
                annotation = instr[id:]
            else:
                code = instr
            if code.find(' ') != -1:
                id = code.find(' ')
                operand = code[id+1:]
                operator = code[0:id]
            else:
                operator = code

            if operand != None:
                if operand.find(',') != -1:
                    strs = operand.split(',')
                    if len(strs) == 2:
                        operand1, operand2 = strs[0], strs[1]
                    else:
                        operand1, operand2, operand3 = strs[0], strs[1], strs[2]
                else:
                    operand1 = operand
                    operand2 = None
            if 'lea' in operator:
                # not saving stack vars
                current_stack_vars = self.untouched_stack_vars
            else:
                current_stack_vars = self.stack_vars
            if operand1 != None:
                operand1 = parse_operand(
                    operator, 1, operand1, ori=self.ori, diemph_ctx=[],
                    stack_vars=current_stack_vars, instr_idx=idx)
            if operand2 != None:
                operand2 = parse_operand(
                    operator, 2, operand2, ori=self.ori, diemph_ctx=[],
                    stack_vars=current_stack_vars, instr_idx=idx)
            if operand3 != None:
                operand3 = parse_operand(
                    operator, 3, operand3, ori=self.ori, diemph_ctx=[],
                    stack_vars=current_stack_vars, instr_idx=idx)
            ret.append([operator, operand1, operand2, operand3, annotation])
        return ret

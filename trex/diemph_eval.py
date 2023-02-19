import random
import torch
import argparse
from fairseq.models.trex import TrexModel
import os
import pickle
from diemph_utils import data_loader
from diemph_utils import asm_parser
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
import copy

torch.manual_seed(456)
np.random.seed(456)
random.seed(456)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        required=True, help='checkpoint path')
    parser.add_argument('--stack-model', type=str,
                        default='')
    parser.add_argument('--data', type=str,
                        required=True, help='data path')
    parser.add_argument(
        '--sample', type=str, required=True, help='sample')
    parser.add_argument(
        '--bf-removal', action='store_true', help='use original tokens')
    parser.add_argument(
        '--neg-num', type=int, default=500, help='number of negative samples')
    parser.add_argument(
        '--emb-in', type=str, default='', help='emb input path')
    parser.add_argument(
        '--emb-out', type=str, default='', help='emb output path')
    parser.add_argument(
        '--baseline-emb', type=str, default='', help='baseline embedding path')
    parser.add_argument(
        '--dbg-out', type=str, default='', help='debug output path')
    parser.add_argument(
        '--rewrite-strategy', type=str, default='jtrans', help='rewrite strategy')
    args = parser.parse_args()
    return args

# Note: this function is used in other files


def build_input_tokens(instr_list, max_token=510, bos=True):
    if bos:
        static = ['<s>']
        inst_pos_emb = ['<s>']
        op_pos_emb = ['<s>']
        arch_emb = ['<s>']
        byte1 = ['<s>']
    else:
        static = []
        inst_pos_emb = []
        op_pos_emb = []
        arch_emb = []
        byte1 = []
    instr_cnt = 0
    token_cnt = 0
    for i in instr_list:
        ops = i.split()
        length = len(ops)
        token_cnt += length
        if token_cnt >= max_token:
            break
        static.extend(ops)
        inst_pos_emb.extend([str(instr_cnt)] * length)
        op_pos_emb.extend([str(i) for i in range(length)])
        arch_emb.extend(['x64'] * length)
        byte1.extend(['##'] * length)
        instr_cnt += 1
    return {
        'static': ' '.join(static),
        'inst_pos_emb': ' '.join(inst_pos_emb),
        'op_pos_emb': ' '.join(op_pos_emb),
        'arch_emb': ' '.join(arch_emb),
        'byte1': ' '.join(byte1),
        'byte2': ' '.join(byte1),
        'byte3': ' '.join(byte1),
        'byte4': ' '.join(byte1),
    }

# Note: this function is used in other files


def get_instr_list(func, ori, rewrite_strategy):
    instrs = data_loader.get_instr_list(func)
    instr_list = build_trex_instr_list(instrs, ori, rewrite_strategy)
    return instr_list


def get_reordered_instr_list(func, ori, rewrite_strategy):
    instrs = data_loader.get_instr_list(func, ori=False)
    instr_list = build_trex_instr_list(instrs, ori, rewrite_strategy)
    return instr_list


def build_trex_instr_list(instrs, ori, rewrite_strategy):
    op_list = []
    for i in instrs:
        ops = asm_parser.parse_asm(i, ori=ori)
        op_list.append(ops)
    return build_trex_instr_list_from_op_list(op_list, ori, rewrite_strategy)


def jtrans_should_skip(ops):
    if 'nop' in ops[0]:
        return 1
    if 'xchg' in ops[0] and ops[1] == ops[2]:
        return 1

    if 'endbr' in ops[0] or 'bnd' in ops[0]:
        return 1
    if 'push' in ops[0] and ops[1] != None and 'rbp' in ops[1]:
        return 1
    if ('mov' in ops[0]
        and ops[1] != None and 'rbp' in ops[1]
            and ops[2] != None and 'rsp' in ops[2]):
        return 1
    if ('mov' in ops[0]
        and ops[1] != None and 'rax' in ops[1]
            and ops[2] != None and 'fs' in ops[2]):
        return 2
    if ('or' in ops[0]
        and ops[1] != None and '[ rsp + num  + hexvar  ]' in ops[1]
            and ops[2] != None and 'num' in ops[2]):
        return 1

    return 0
                
def binkit_should_skip(ops):
    if 'nop' in ops[0]:
        return 1
    if 'xchg' in ops[0] and ops[1] == ops[2]:
        return 1

    if 'push' in ops[0] and ops[1] != None and 'rbp' in ops[1]:
        return 1
    if 'push' in ops[0] and ops[1] != None and 'r15' in ops[1]:
        return 1
    if 'push' in ops[0] and ops[1] != None and 'r14' in ops[1]:
        return 1
    if 'push' in ops[0] and ops[1] != None and 'r13' in ops[1]:
        return 1
    if ('mov' in ops[0]
        and ops[1] != None and 'rbp' in ops[1]
            and ops[2] != None and 'rsp' in ops[2]):
        return 1
    return 0


def binkit_stack_should_skip(ops):
    if 'nop' in ops[0]:
        return 1
    if 'xchg' in ops[0] and ops[1] == ops[2]:
        return 1

    if ('sub' in ops[0]
        and ops[1] != None and 'rsp' in ops[1]
            and ops[2] != None and 'num' in ops[2]):
        return 1
    # mov [ rsp + num  + hexvar  ] , r9 
    if ('mov' in ops[0]
        and ops[1] != None and 'rsp + num  + hexvar' in ops[1]
            and ops[2] != None and 'r9' in ops[2]):
        return 1
    # mov [ rdi + num ] , al 
    if ('mov' in ops[0]
        and ops[1] != None and 'rdi + num' in ops[1]
            and ops[2] != None and 'al' in ops[2]):
        return 1
    # mov edi , cs: hexvar 
    if ('mov' in ops[0]
        and ops[1] != None and 'edi' in ops[1]
            and ops[2] != None and 'cs' in ops[2] and 'hexvar' in ops[2]):
        return 1
    # mov [ rbp + hexvar ] , r9
    if ('mov' in ops[0]
        and ops[1] != None and 'rbp + hexvar' in ops[1]
            and ops[2] != None and 'r9' in ops[2]):
        return 1
    return 0    

def how_should_skip(ops):
    if 'nop' in ops[0]:
        return 1
    if 'xchg' in ops[0] and ops[1] == ops[2]:
        return 1
    
    # push rbp
    if 'push' in ops[0] and ops[1] != None and 'rbp' in ops[1]:
        return 1
    # push r15
    if 'push' in ops[0] and ops[1] != None and 'r15' in ops[1]:
        return 1
    # model2: push r14
    if 'push' in ops[0] and ops[1] != None and 'r14' in ops[1]:
        return 1
    # new-dataset: push r13
    if 'push' in ops[0] and ops[1] != None and 'r13' in ops[1]:
        return 1
    # new-dataset: push r12
    if 'push' in ops[0] and ops[1] != None and 'r12' in ops[1]:
        return 1
    return 0

def binkit_no_analysis_should_skip(ops):
    if 'nop' in ops[0]:
        return 1
    if 'xchg' in ops[0] and ops[1] == ops[2]:
        return 1

    if 'push' in ops[0] and ops[1] != None and 'rbp' in ops[1]:
        return 1
    if 'push' in ops[0] and ops[1] != None and 'r15' in ops[1]:
        return 1
    if 'push' in ops[0] and ops[1] != None and 'r14' in ops[1]:
        return 1
    if 'push' in ops[0] and ops[1] != None and 'r13' in ops[1]:
        return 1
    if 'call' in ops[0] and ops[1] != None and 'hexvar' in ops[1]:
        return 1
    return 0


def binkit_no_class_importance_should_skip(ops):
    if 'nop' in ops[0]:
        return 1
    if 'xchg' in ops[0] and ops[1] == ops[2]:
        return 1

    # push rbp
    if 'push' in ops[0] and ops[1] != None and 'rbp' in ops[1]:
        return 1
    #  add rsp num
    if ('add' in ops[0]
        and ops[1] != None and 'rsp' in ops[1]
            and ops[2] != None and 'num' in ops[2]):
        return 1
    #  pop rbp
    if 'pop' in ops[0] and ops[1] != None and 'rbp' in ops[1]:
        return 1
    #  ret
    if 'ret' in ops[0]:
        return 1
    # pop rbx
    if 'pop' in ops[0] and ops[1] != None and 'rbx' in ops[1]:
        return 1

    return 0

def build_trex_instr_list_from_op_list(op_list, ori, rewrite_strategy):
    instr_list = []
    to_skip = 0
    first_endbr = True
    for ops in op_list:
        if to_skip > 0:
            to_skip -= 1
            continue
        ret = ''
        # if not ori:
        #     if 'nop' in ops[0]:
        #         continue
        #     if 'xchg' in ops[0] and 'ax' in ops[1] and 'ax' in ops[2]:
        #         continue            
        
        # rewrite for all (give it a try)
        if 'jz' in ops[0]:
            new_op0 = ops[0].replace('jz', 'je')
            # construct a new tuple
            ops = (new_op0, ops[1], ops[2], ops[3])
        elif 'jnz' in ops[0]:
            new_op0 = ops[0].replace('jnz', 'jne')
            # construct a new tuple
            ops = (new_op0, ops[1], ops[2], ops[3])
        elif 'retn' in ops[0]:
            ops = ('ret', ops[1], ops[2], ops[3])
        elif 'jnb' in ops[0]:
            new_op0 = ops[0].replace('jnb', 'jae')
            # construct a new tuple
            ops = (new_op0, ops[1], ops[2], ops[3])
        elif 'setnz' in ops[0]:
            new_op0 = ops[0].replace('setnz', 'setne')
            # construct a new tuple
            ops = (new_op0, ops[1], ops[2], ops[3])
        elif 'setz' in ops[0]:
            new_op0 = ops[0].replace('setz', 'sete')
            # construct a new tuple
            ops = (new_op0, ops[1], ops[2], ops[3])
        elif 'setnb' in ops[0]:
            new_op0 = ops[0].replace('setnb', 'setae')
            # construct a new tuple
            ops = (new_op0, ops[1], ops[2], ops[3])
        elif 'setnl' in ops[0]:
            new_op0 = ops[0].replace('setnl', 'setge')
            # construct a new tuple
            ops = (new_op0, ops[1], ops[2], ops[3])
        elif 'setnbe' in ops[0]:
            new_op0 = ops[0].replace('setnbe', 'seta')
            # construct a new tuple
            ops = (new_op0, ops[1], ops[2], ops[3])
        elif 'setnle' in ops[0]:
            new_op0 = ops[0].replace('setnle', 'setg')
            # construct a new tuple
            ops = (new_op0, ops[1], ops[2], ops[3])

        # XXX: swap op1 and op2
        # if ops[1] is not None and ops[2] is not None:
        #     ops = (ops[0], ops[2], ops[1], ops[3])
        ret += ops[0] + ' '
        if ops[1] != None:
            op1 = ops[1].strip()
            # if not ori and len(op_list) < 8:
            #     op1 = op1.replace('cs :', '')            
            ret +=  op1.strip() + ' '
        if ops[2] != None:
            op2 = ops[2].strip()
            # if not ori and len(op_list) < 8:
            #     op2 = op2.replace('cs :', '')
            ret += ', ' + op2.strip() + ' '
        if ops[3] != None:
            op3 = ops[3].strip()
            # if not ori and len(op_list) < 8:
            #     op3 = op3.replace('cs :', '')
            ret += ', ' + op3.strip() + ' '
        if not ori:
            if 'endbr64' in ops[0]:
                if first_endbr:
                    first_endbr = False
                else:
                    break
            if 'binarycorp' in rewrite_strategy:
                to_skip = jtrans_should_skip(ops)
            elif 'binkit' in rewrite_strategy:
                to_skip = binkit_should_skip(ops)
            elif 'how' in rewrite_strategy:
                to_skip = how_should_skip(ops)
            elif 'stackbink' in rewrite_strategy:
                to_skip = binkit_stack_should_skip(ops)
            elif 'noanalbink' in rewrite_strategy:
                to_skip = binkit_no_analysis_should_skip(ops)
            elif 'noclassbink' in rewrite_strategy:
                to_skip = binkit_no_class_importance_should_skip(ops)

            if to_skip > 0:
                to_skip = to_skip-1
                continue

        instr_list.append(ret)
    return instr_list


def encode_one(model, instr_list):
    tokens = build_input_tokens(instr_list)
    input = model.process_token_dict(model.encode(tokens))
    out = model.model(input, classification_head_name='similarity')[
        0]['features']
    return out.detach()


usable_regs = {
    'r15': ['r13', 'r14', 'r15'],
    'r14': ['r13', 'r14', 'r15'],
    'r13': ['r13', 'r14', 'r15'],
    'r12': ['r13', 'r14', 'r15'],
    'r11': ['r12', 'r13', 'r14', 'r15'],
    'r10': ['r11', 'r12', 'r13', 'r14', 'r15'],
    'r9': ['r10', 'r11', 'r12', 'r13', 'r14', 'r15'],
    'r8': ['r10', 'r11', 'r12', 'r13', 'r14', 'r15'],
}


def rewrite_one_func(func: asm_parser.FunctionParser, ori):
    ret = []
    stack_vars = func.get_stack_var_list()
    current_usable_regs = usable_regs[func.largest_register]
    # len_current_usable_regs = len(current_usable_regs)
    current_usable_regs = ['r10', 'r11']
    current_instr_list = copy.deepcopy(func.instr_parsed)
    reg_idx = 0
    for varname, _ in stack_vars[:2]:
        current_instr_list = func.rewrite(to_rewrite=varname,
                                new_var=current_usable_regs[reg_idx %
                                                    len(current_usable_regs)],
                                    instr_parsed=current_instr_list)
        reg_idx += 1
    ret.append(build_trex_instr_list_from_op_list(
            func.finalize(current_instr_list), ori=ori, rewrite_strategy=''))
    return ret
        

def main():
    args = parse_args()
    print(args)
    ori = not args.bf_removal
    query_funcs, candi_funcs = data_loader.load_sample(args.sample)
    # encode all funcs
    if args.emb_in != '':
        with open(args.emb_in, 'rb') as f:
            embs = pickle.load(f)
            query_embs_in = embs['query_embs']
            candi_embs_in = embs['candi_embs']

            query_name2emb_in = {}
            for i in query_embs_in:
                query_name2emb_in[(i[0], i[1])] = (i[2], i[3], i[4])
            candi_name2emb_in = {}
            for i in candi_embs_in:
                candi_name2emb_in[(i[0], i[1])] = (i[2], i[3], i[4])

            for func in query_funcs:
                func['emb'] = query_name2emb_in[(
                    func['project_name'], func['funcname'])][0]
                func['rewritten_instr_list'] = query_name2emb_in[(
                    func['project_name'], func['funcname'])][1]
                func['rewritten_instr_list_emb'] = query_name2emb_in[(
                    func['project_name'], func['funcname'])][2]
            for func in candi_funcs:
                func['emb'] = candi_name2emb_in[(
                    func['project_name'], func['funcname'])][0]
                func['rewritten_instr_list'] = candi_name2emb_in[(
                    func['project_name'], func['funcname'])][1]
                func['rewritten_instr_list_emb'] = candi_name2emb_in[(
                    func['project_name'], func['funcname'])][2]

            print('load emb from %s' % args.emb_in)
    else:
        model = TrexModel.from_pretrained(
            os.path.dirname(args.model), os.path.basename(args.model),
            data_name_or_path=args.data)

        model.cuda()
        model.eval()
        for func in tqdm(query_funcs):
            if ori:
                instr_list = get_instr_list(func, ori=ori, rewrite_strategy=args.rewrite_strategy)[:300]
            else:
                instr_list = get_reordered_instr_list(func, ori=False, rewrite_strategy=args.rewrite_strategy)[:300]
            func['emb'] = encode_one(model, instr_list)
            if not ori:
                # XXX: legacy code
                func['rewritten_instr_list'] = []
                func['rewritten_instr_list_emb'] = []
            else:
                func['rewritten_instr_list'] = []
                func['rewritten_instr_list_emb'] = []

        for func in tqdm(candi_funcs):
            instr_list = get_instr_list(func, ori=ori, rewrite_strategy=args.rewrite_strategy)[:300]
            func['emb'] = encode_one(model, instr_list)
            if not ori:
                func['rewritten_instr_list'] = []
                func['rewritten_instr_list_emb'] = []
            else:
                func['rewritten_instr_list'] = []
                func['rewritten_instr_list_emb'] = []


        if args.emb_out != '':
            query_embs = [(i['project_name'], i['funcname'], i['emb'],
                           i['rewritten_instr_list'], i['rewritten_instr_list_emb'])
                          for i in query_funcs]
            candi_embs = [(i['project_name'], i['funcname'], i['emb'],
                           i['rewritten_instr_list'], i['rewritten_instr_list_emb'])
                          for i in candi_funcs]
            to_save = {
                'query_embs': query_embs,
                'candi_embs': candi_embs,
            }
            with open(args.emb_out, 'wb') as f:
                pickle.dump(to_save, f)
            print('save embs to', args.emb_out)

    if args.baseline_emb != '':
        baseline_query_name2emb = {}
        baseline_candi_name2emb = {}
        with open(args.baseline_emb, 'rb') as f:
            to_load = pickle.load(f)
            for i in to_load['query_embs']:
                baseline_query_name2emb[(i[0], i[1])] = i[2]
            for i in to_load['candi_embs']:
                baseline_candi_name2emb[(i[0], i[1])] = i[2]

    query_name2func = {(i['project_name'], i['funcname'])
                        : i for i in query_funcs}
    candi_name2func = {(i['project_name'], i['funcname'])
                        : i for i in candi_funcs}

    # random seed 233
    torch.manual_seed(233)
    np.random.seed(233)
    NEG_SAMPLE = args.neg_num
    neg_sample_num = min(NEG_SAMPLE + 20, len(candi_funcs))
    results = []
    # ret, candi_names, sorted_idx
    dbg_results = []
    for func in tqdm(query_funcs):
        query_name = (func['project_name'], func['funcname'])
        query_emb = func['emb']
        if query_name not in candi_name2func:
            continue
        gt_func = candi_name2func[query_name]
        current_candi_funcs = np.random.choice(
            candi_funcs, neg_sample_num, replace=False).tolist()
        candi_names = [(f['project_name'], f['funcname']) for f in current_candi_funcs
                       if f['funcname'] != gt_func['funcname']]
        candi_names = [query_name] + candi_names[:NEG_SAMPLE]
        candi_embs = [candi_name2func[n]['emb'] for n in candi_names]
        candi_embs_tensor = torch.stack(candi_embs).squeeze(1)
        # cosine similarity
        sim = torch.nn.functional.cosine_similarity(
            query_emb, candi_embs_tensor, dim=-1).squeeze(-1).cpu()
        
        rewritten_sim = sim

        if args.baseline_emb != '':
            ori_len = len(get_instr_list(func, ori=True, rewrite_strategy=''))
            new_len = len(get_instr_list(func, ori=False, rewrite_strategy=args.rewrite_strategy))
            if (ori_len - new_len) / ori_len > 0.2 and len(func['cfg'].nodes) < 3:
                use_baseline = True
            else:
                use_baseline = False
            
            baseline_query_emb = baseline_query_name2emb[query_name]
            baseline_embs = [baseline_candi_name2emb[n] for n in candi_names]
            baseline_embs_tensor = torch.stack(baseline_embs).squeeze(1)
            baseline_sim = torch.nn.functional.cosine_similarity(
                baseline_query_emb, baseline_embs_tensor, dim=-1).squeeze(-1).cpu()
            sort_my = torch.argsort(sim, descending=True)
            baseline_mask = torch.zeros_like(rewritten_sim)
            baseline_mask[sort_my[:3]] = 1

            if use_baseline:
                final_sim = 0.5 * sim + 0.5 * baseline_sim * baseline_mask
            else:
                final_sim = sim
        else:
            final_sim = sim
        

        # argsort by similarity
        sorted_idx = torch.argsort(final_sim, descending=True)
        ret = torch.where(sorted_idx == 0)[0].item()
        if (ret > 0 and
            data_loader.get_instr_list(candi_name2func[candi_names[0]], ori=True)
                == data_loader.get_instr_list(candi_name2func[candi_names[sorted_idx[0]]], ori=True)):
            ret = 0.1

        results.append(ret)
        dbg_results.append(
            (ret, candi_names, sorted_idx.cpu().numpy(), final_sim.cpu().numpy()))


    if args.dbg_out != '':
        with open(args.dbg_out, 'wb') as f:
            pickle.dump(dbg_results, f)
        print('save dbg results to', args.dbg_out)
    print("Mean rank: ", np.mean(results))
    print("Median rank: ", np.median(results))
    print("Top 1: %f=%d/%d" %
          (np.mean([i < 1 for i in results]), np.sum([i < 1 for i in results]), len(results)))
    print("Top 5: %f=%d/%d" %
          (np.mean([i < 5 for i in results]), np.sum([i < 5 for i in results]), len(results)))
    print("Top 10: %f=%d/%d" %
          (np.mean([i < 10 for i in results]), np.sum([i < 10 for i in results]), len(results)))
    print("Done")


# main
if __name__ == '__main__':
    main()

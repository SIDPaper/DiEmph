from diemph_eval import build_trex_instr_list, build_input_tokens, build_trex_instr_list_from_op_list
from tqdm import tqdm
from diemph_utils import data_loader, asm_parser
import pickle
import os
import glob
import argparse
import torch
import numpy as np
import random
import copy
# set random seed
random.seed(2333)
np.random.seed(2333)
torch.manual_seed(2333)


rule_out_funcs = set([
    '_start',
    'deregister_tm_clones',
    'register_tm_clones',
    '__do_global_dtors_aux',
    'frame_dummy',
    '__libc_csu_init',
    '__libc_csu_fini',
    '_fini',
    '__do_global_dtors_aux_fini_array_entry',
    '__libc_start_main',
])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jtrans-dataset', type=str, default='jtrans-dataset')
    parser.add_argument('--sample-in', type=str,
                        default='')
                        # default='data-src/jtrans/raw_sample_results.pkl')
    parser.add_argument('--out-dir', type=str, default='data-src/jtrans-dbg')
    parser.add_argument('--sample-size', type=int, default=1250)    
    parser.add_argument('--normal', action='store_true')
    # parser.add_argument('--rewrite-stack-vars', action='store_true')
    parser.add_argument('--rewrite-bad-ops', action='store_true')
    # parser.add_argument('--rewrite-layout', action='store_true')
    parser.add_argument('--rewrite-strategy', type=str, default='jtrans')
    args = parser.parse_args()
    return args


def _find_files(project):
    global opt_list
    ret_files = {}
    for opt in opt_list:
        file = glob.glob(os.path.join(project, '*' + opt + '*extract.pkl'))
        if len(file) == 0:
            continue
        file = file[0]
        ret_files[opt] = file
    return ret_files



def collect_func_from_one_project(project):
    global opt_list
    # find all files
    files = _find_files(project)
    if len(files) == 0:
        return []        
    # load all files
    funcs = []
    loaded_files = {}
    for opt in opt_list:
        if opt not in files:
            continue
        loaded_files[opt] = pickle.load(open(files[opt], 'rb'))

    all_valid_func_names = set(loaded_files[opt_list[0]].keys())
    for opt in opt_list[1:]:
        all_valid_func_names = all_valid_func_names & set(loaded_files[opt].keys())
    all_valid_func_names = all_valid_func_names - rule_out_funcs

    for func_name in all_valid_func_names:
        ret = {}
        ret['func_name'] = func_name
        ret['project_name'] = os.path.basename(project)
        for opt in opt_list:
            if opt not in files:
                continue
            ret[opt] = {}
            ret[opt]['addr'] = loaded_files[opt][func_name][0]
            ret[opt]['asm'] = loaded_files[opt][func_name][1]
        funcs.append(ret)
    return funcs


class TrexPairwiseFileWriter:
    def __init__(self, out_dir, split_name):
        self.static0 = open(os.path.join(
            out_dir, "%s.static.input0" % split_name), 'w')
        self.static1 = open(os.path.join(
            out_dir, "%s.static.input1" % split_name), 'w')
        self.inst_pos_emb0 = open(os.path.join(
            out_dir, "%s.inst_pos_emb.input0" % split_name), 'w')
        self.inst_pos_emb1 = open(os.path.join(
            out_dir, "%s.inst_pos_emb.input1" % split_name), 'w')
        self.op_pos_emb0 = open(os.path.join(
            out_dir, "%s.op_pos_emb.input0" % split_name), 'w')
        self.op_pos_emb1 = open(os.path.join(
            out_dir, "%s.op_pos_emb.input1" % split_name), 'w')
        self.arch_emb0 = open(os.path.join(
            out_dir, "%s.arch_emb.input0" % split_name), 'w')
        self.arch_emb1 = open(os.path.join(
            out_dir, "%s.arch_emb.input1" % split_name), 'w')
        self.byte10 = open(os.path.join(
            out_dir, "%s.byte1.input0" % split_name), 'w')
        self.byte11 = open(os.path.join(
            out_dir, "%s.byte1.input1" % split_name), 'w')
        self.byte20 = open(os.path.join(
            out_dir, "%s.byte2.input0" % split_name), 'w')
        self.byte21 = open(os.path.join(
            out_dir, "%s.byte2.input1" % split_name), 'w')
        self.byte30 = open(os.path.join(
            out_dir, "%s.byte3.input0" % split_name), 'w')
        self.byte31 = open(os.path.join(
            out_dir, "%s.byte3.input1" % split_name), 'w')
        self.byte40 = open(os.path.join(
            out_dir, "%s.byte4.input0" % split_name), 'w')
        self.byte41 = open(os.path.join(
            out_dir, "%s.byte4.input1" % split_name), 'w')
        self.label = open(os.path.join(out_dir, "%s.label" % split_name), 'w')

    def write(self, input0, input1, label):
        self.static0.write(input0['static'] + '\n')
        self.static1.write(input1['static'] + '\n')
        self.inst_pos_emb0.write(input0['inst_pos_emb'] + '\n')
        self.inst_pos_emb1.write(input1['inst_pos_emb'] + '\n')
        self.op_pos_emb0.write(input0['op_pos_emb'] + '\n')
        self.op_pos_emb1.write(input1['op_pos_emb'] + '\n')
        self.arch_emb0.write(input0['arch_emb'] + '\n')
        self.arch_emb1.write(input1['arch_emb'] + '\n')
        self.byte10.write(input0['byte1'] + '\n')
        self.byte11.write(input1['byte1'] + '\n')
        self.byte20.write(input0['byte2'] + '\n')
        self.byte21.write(input1['byte2'] + '\n')
        self.byte30.write(input0['byte3'] + '\n')
        self.byte31.write(input1['byte3'] + '\n')
        self.byte40.write(input0['byte4'] + '\n')
        self.byte41.write(input1['byte4'] + '\n')
        self.label.write(label + '\n')
        # flush all
        self.static0.flush()
        self.static1.flush()
        self.inst_pos_emb0.flush()
        self.inst_pos_emb1.flush()
        self.op_pos_emb0.flush()
        self.op_pos_emb1.flush()
        self.arch_emb0.flush()
        self.arch_emb1.flush()
        self.byte10.flush()
        self.byte11.flush()
        self.byte20.flush()
        self.byte21.flush()
        self.byte30.flush()
        self.byte31.flush()
        self.byte40.flush()
        self.byte41.flush()
        self.label.flush()



def _construct_normal_pairs(args, funcs, split, writer):
    global O0_str, O1_str, O2_str, O3_str
    for current_func in tqdm(funcs):
        # positive pair1 - O0 vs O3
        instr_list = current_func[O0_str]['asm']
        trex_instr_list = build_trex_instr_list(instr_list, ori=True, rewrite_strategy='')
        input0 = build_input_tokens(trex_instr_list, bos=False)
        instr_list = current_func[O3_str]['asm']
        trex_instr_list = build_trex_instr_list(instr_list, ori=True, rewrite_strategy='')
        input1 = build_input_tokens(trex_instr_list, bos=False)
        writer.write(input0, input1, '1')
        # positive pair2 - O1 vs O3
        instr_list = current_func[O1_str]['asm']
        trex_instr_list = build_trex_instr_list(instr_list, ori=True, rewrite_strategy='')
        input0 = build_input_tokens(trex_instr_list, bos=False)
        instr_list = current_func[O3_str]['asm']
        trex_instr_list = build_trex_instr_list(instr_list, ori=True, rewrite_strategy='')
        input1 = build_input_tokens(trex_instr_list, bos=False)
        writer.write(input0, input1, '1')
        # positive pair3 - O0 vs O1
        instr_list = current_func[O0_str]['asm']
        trex_instr_list = build_trex_instr_list(instr_list, ori=True, rewrite_strategy='')
        input0 = build_input_tokens(trex_instr_list, bos=False)
        instr_list = current_func[O1_str]['asm']
        trex_instr_list = build_trex_instr_list(instr_list, ori=True, rewrite_strategy='')
        input1 = build_input_tokens(trex_instr_list, bos=False)
        writer.write(input0, input1, '1')

        # random select either 2*3 or 3*3
        if random.random() > 0.5:
          neg_funcs = random.sample(funcs, 6)
        #   neg_funcs = random.sample(funcs, 2)
        else:
          neg_funcs = random.sample(funcs, 9)
        #   neg_funcs = random.sample(funcs, 3)
        
        for neg_func in neg_funcs:
          if neg_func['func_name'] == current_func['func_name']:
            continue
          neg_opt = random.sample(opt_list, 1)[0]
          while neg_opt not in neg_func:
            neg_opt = random.sample(opt_list, 1)[0]          
          instr_list = neg_func[neg_opt]['asm']
          trex_instr_list = build_trex_instr_list(instr_list, ori=True, rewrite_strategy='')
          input0 = build_input_tokens(trex_instr_list, bos=False)
          writer.write(input0, input1, '-1')

def _rewrite_stack_var_for_one_func(func_parser):
        stack_vars = func_parser.get_stack_var_list()
        current_instr_list = copy.deepcopy(func_parser.instr_parsed)
        if len(stack_vars) > 0:
            idx = 0
            reg_list = ['r10', 'r11']
            for stack_var_name, _ in stack_vars[:2]:
                current_instr_list = func_parser.rewrite(to_rewrite=stack_var_name,
                new_var=reg_list[idx % len(reg_list)],
                instr_parsed=current_instr_list)
                idx += 1
        return build_trex_instr_list_from_op_list(
            func_parser.finalize(current_instr_list), ori=True, rewrite_strategy='')


def _construct_rewrite_stack_pairs(args, funcs, split, writer):
    global O0_str, O1_str, O2_str, O3_str
    # whether to combine two modifications
    STACK_ORI = False
    for current_func in tqdm(funcs):
        # O0 vs O3
        instr_list_0 = current_func[O0_str]['asm']
        trex_instr_list = build_trex_instr_list(instr_list_0, ori=STACK_ORI, rewrite_strategy='')
        input0 = build_input_tokens(trex_instr_list, bos=False)
        instr_list_3 = current_func[O3_str]['asm']
        trex_instr_list = build_trex_instr_list(instr_list_3, ori=STACK_ORI, rewrite_strategy='')
        input1 = build_input_tokens(trex_instr_list, bos=False)
        writer.write(input0, input1, '1')
        # O0 rewritten vs O3 rewritten
        func_parser = asm_parser.FunctionParser(ori=STACK_ORI, instr_list=instr_list_0)
        rewritten_list_0 =_rewrite_stack_var_for_one_func(func_parser)
        rewritten_input_0 = build_input_tokens(rewritten_list_0, bos=False)
        func_parser3 = asm_parser.FunctionParser(ori=STACK_ORI, instr_list=instr_list_3)
        rewritten_list_3 = _rewrite_stack_var_for_one_func(func_parser3)
        rewritten_input_3 = build_input_tokens(rewritten_list_3, bos=False)
        writer.write(rewritten_input_0, rewritten_input_3, '1')
        # O0 rewritten vs O1 rewritten
        instr_list_1 = current_func[O1_str]['asm']
        func_parser1 = asm_parser.FunctionParser(ori=STACK_ORI, instr_list=instr_list_1)
        rewritten_list_1 = _rewrite_stack_var_for_one_func(func_parser1)
        rewritten_input_1 = build_input_tokens(rewritten_list_1, bos=False)
        writer.write(rewritten_input_0, rewritten_input_1, '1')
        # O0 vs O0 rewritten
        writer.write(input0, rewritten_input_0, '1')        

        neg_funcs = random.sample(funcs, 3)
        # random select either 2*3 or 3*3
        if random.random() > 0.5:
          neg_funcs = random.sample(funcs, 2)        
        else:
          neg_funcs = random.sample(funcs, 3)
        
        for neg_func in neg_funcs:
            if neg_func['func_name'] == current_func['func_name']:
                continue
            neg_opt = random.sample(opt_list, 1)[0]
            while neg_opt not in neg_func:
                neg_opt = random.sample(opt_list, 1)[0]
            # if 'Os' not in neg_func and neg_opt == 'Os':
            #     neg_opt = 'O3'
            neg_instr_list = neg_func[neg_opt]['asm']
            neg_trex_instr_list = build_trex_instr_list(neg_instr_list, ori=STACK_ORI, rewrite_strategy='')
            neg_input = build_input_tokens(neg_trex_instr_list, bos=False)
            writer.write(input0, neg_input, '-1')
            neg_func_parser = asm_parser.FunctionParser(ori=STACK_ORI, instr_list=neg_instr_list)
            neg_rewritten_list = _rewrite_stack_var_for_one_func(neg_func_parser)
            neg_rewritten_input = build_input_tokens(neg_rewritten_list, bos=False)
            writer.write(rewritten_input_0, neg_rewritten_input, '-1')            



def _construct_rewrite_bad_ops_pairs(args, funcs, split, writer, rewrite_strategy):
    global O0_str, O1_str, O2_str, O3_str
    for current_func in tqdm(funcs):
        # positive pair1 - O0 vs O3
        instr_list = current_func[O0_str]['asm']
        trex_instr_list = build_trex_instr_list(instr_list, ori=False, rewrite_strategy=rewrite_strategy)
        input0 = build_input_tokens(trex_instr_list, bos=False)
        instr_list = current_func[O3_str]['asm']
        trex_instr_list = build_trex_instr_list(instr_list, ori=False, rewrite_strategy=rewrite_strategy)
        input1 = build_input_tokens(trex_instr_list, bos=False)
        writer.write(input0, input1, '1')
        # positive pair2 - O1 vs O3
        instr_list = current_func[O1_str]['asm']
        trex_instr_list = build_trex_instr_list(instr_list, ori=False, rewrite_strategy=rewrite_strategy)
        input0 = build_input_tokens(trex_instr_list, bos=False)
        instr_list = current_func[O3_str]['asm']
        trex_instr_list = build_trex_instr_list(instr_list, ori=False, rewrite_strategy=rewrite_strategy)
        input1 = build_input_tokens(trex_instr_list, bos=False)
        writer.write(input0, input1, '1')
        # positive pair3 - O0 vs O1
        instr_list = current_func[O0_str]['asm']
        trex_instr_list = build_trex_instr_list(instr_list, ori=False, rewrite_strategy=rewrite_strategy)
        input0 = build_input_tokens(trex_instr_list, bos=False)
        instr_list = current_func[O1_str]['asm']
        trex_instr_list = build_trex_instr_list(instr_list, ori=False, rewrite_strategy=rewrite_strategy)
        input1 = build_input_tokens(trex_instr_list, bos=False)
        writer.write(input0, input1, '1')

        # random select either 2*3 or 3*3
        if random.random() > 0.5:
          neg_funcs = random.sample(funcs, 6)
        #   neg_funcs = random.sample(funcs, 2)
        else:
          neg_funcs = random.sample(funcs, 9)
        #   neg_funcs = random.sample(funcs, 3)
        
        for neg_func in neg_funcs:
          if neg_func['func_name'] == current_func['func_name']:
            continue
          neg_opt = random.sample(opt_list, 1)[0]
          while neg_opt not in neg_func:
            neg_opt = random.sample(opt_list, 1)[0]
          instr_list = neg_func[neg_opt]['asm']
          trex_instr_list = build_trex_instr_list(instr_list, ori=False, rewrite_strategy=rewrite_strategy)
          input0 = build_input_tokens(trex_instr_list, bos=False)
          writer.write(input0, input1, '-1')




def construct_pairs(args, funcs, split, rewrite_strategy):    
    writer = TrexPairwiseFileWriter(args.out_dir, split)
    if args.normal:
        _construct_normal_pairs(args, funcs, split, writer)
    # if args.rewrite_stack_vars:
    #     _construct_rewrite_stack_pairs(args, funcs, split, writer)
    if args.rewrite_bad_ops:
        _construct_rewrite_bad_ops_pairs(args, funcs, split, writer, rewrite_strategy=rewrite_strategy)



opt_list = ['O0', 'O1', 'O2', 'O3', 'Os']
def main():
    global opt_list, O0_str, O1_str, O2_str, O3_str
    args = parse_args()
    print(args)

    if ((args.sample_in != '' and 'binkit' in args.sample_in)
        or ('binkit' in args.jtrans_dataset)):
        opt_list = ['clang4O0', 'clang4O1', 'clang4O2', 'clang4O3',
                    'clang5O0', 'clang5O1', 'clang5O2', 'clang5O3',
                    'clang6O0', 'clang6O1', 'clang6O2', 'clang6O3',
                    'clang7O0', 'clang7O1', 'clang7O2', 'clang7O3',
                    'gcc494O0', 'gcc494O1', 'gcc494O2', 'gcc494O3',
                    'gcc550O0', 'gcc550O1', 'gcc550O2', 'gcc550O3',
                    'gcc640O0', 'gcc640O1', 'gcc640O2', 'gcc640O3',
                    'gcc730O0', 'gcc730O1', 'gcc730O2', 'gcc730O3',
                    'gcc820O0', 'gcc820O1', 'gcc820O2', 'gcc820O3',]
    elif ((args.sample_in != '' and 'how' in args.sample_in)
        or ('how' in args.jtrans_dataset)):
        opt_list = [
            'gcc7O0', 'gcc7O1', 'gcc7O2', 'gcc7O3', 'gcc7Os',
            'clang5O0', 'clang5O1', 'clang5O2', 'clang5O3', 'clang5Os',
            'clang3O0', 'clang3O1', 'clang3O2', 'clang3O3', 'clang3Os',
            # 'clang7O0', 'clang7O1', 'clang7O2', 'clang7O3', 'clang7Os',
            # 'clang9O0', 'clang9O1', 'clang9O2', 'clang9O3', 'clang9Os',
            'gcc48O0', 'gcc48O1', 'gcc48O2', 'gcc48O3', 'gcc48Os',
            'gcc5O0', 'gcc5O1', 'gcc5O2', 'gcc5O3', 'gcc5Os',
            # 'gcc9O0', 'gcc9O1', 'gcc9O2', 'gcc9O3', 'gcc9Os',
        ]
    # opt_list = ['O0', 'O1']
    O0_str = opt_list[0]
    O1_str = opt_list[1]
    O2_str = opt_list[2]
    O3_str = opt_list[3]
    
    print("Opt list: ", opt_list)
    print("O0_str:%s, O1_str:%s, O2_str:%s, O3_str:%s" %(O0_str, O1_str, O2_str, O3_str))
    if args.sample_in == '':
        all_projects = [os.path.join(args.jtrans_dataset, f) for f in os.listdir(args.jtrans_dataset) if os.path.isdir(
            os.path.join(args.jtrans_dataset, f))]

        # shuffle projects
        random.shuffle(all_projects)
        if args.sample_size == -1:
            TOTAL = 9999999999
            pbar = tqdm(total=len(all_projects))
        else:
            TOTAL = args.sample_size
            if len(all_projects) * 10 < TOTAL:
                sample_per_proj = int(TOTAL / len(all_projects)) + 3
            else:
                sample_per_proj = 10
            print("Sample %d functions from %d projects" %(TOTAL, len(all_projects)))        
            pbar = tqdm(total=TOTAL)
        all_funcs = []
        current_idx = 0
        for project in all_projects:
            if args.sample_size != -1:
                sample_per_proj = int((TOTAL - len(all_funcs))/ (len(all_projects) - current_idx)) + 3
            else:
                sample_per_proj = 999999999
            current_idx += 1
            funcs = collect_func_from_one_project(project)
            # randomly select 10 functions
            funcs = random.sample(funcs, min(len(funcs), sample_per_proj))
            all_funcs.extend(funcs)
            if args.sample_size != -1:
                pbar.update(len(funcs))
                if pbar.n >= TOTAL:
                    break
            else:
                pbar.update(1)

    # save all_funcs
        pickle.dump(all_funcs, open(os.path.join(
            args.out_dir, 'raw_sample_results.pkl'), 'wb'))
        print("Save to %s" % os.path.join(
            args.out_dir, 'raw_sample_results.pkl'))
    else:        
        all_funcs = pickle.load(open(args.sample_in, 'rb'))
        print("Load from %s" % args.sample_in)

    random.seed(5678)
    # split train and valid
    random.shuffle(all_funcs)
    train_funcs = all_funcs[:int(len(all_funcs) * 0.9)]
    valid_funcs = all_funcs[int(len(all_funcs) * 0.9):][:100]
    # construct pairs
    construct_pairs(args, train_funcs, 'train', rewrite_strategy=args.rewrite_strategy)
    construct_pairs(args, valid_funcs, 'valid', rewrite_strategy=args.rewrite_strategy)


if __name__ == '__main__':
    from diemph_utils import asm_parser
    asm_parser.parse_asm('imul    rdx, rsi, -28h', ori=True)
    main()



if 0==1:
    for i, f in tqdm(enumerate(train_funcs)):
        # prob = [i for i in f['O3']['asm'] if 'imul' in i[:5] 
        # and '+8' in i and 'r8' not in i
        # and 'rsi' in i
        # and 'h' not in i[-3:]]        
        prob = [i for i in f['O3']['asm'] if '28h' in i
         and 'imul' in i and 'rsi' in i and 'rdx' in i]
        if len(prob) > 0:
            print(prob)
            print(f['project_name'], f['func_name'])
            print(i)
            break


from diemph_utils import asm_parser
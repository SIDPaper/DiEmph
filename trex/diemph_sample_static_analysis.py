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
    parser.add_argument('--dataset', type=str, default='jtrans-dataset')
    parser.add_argument('--sample-in', type=str,
                        default='')
                        # default='data-src/jtrans/raw_sample_results.pkl')
    parser.add_argument('--fout', type=str, default='data-static-analysis/sample_results.static.pkl')
    parser.add_argument('--sample-size', type=int, default=-1)    
    args = parser.parse_args()
    return args


opt_list = ['O0', 'O1', 'O2', 'O3', 'Os']
# opt_list = ['clang4O0', 'clang4O1', 'clang4O3', 'gcc494O3']
# opt_list = ['clang5O0', 'clang5O1', 'clang5O3', 'gcc7O3']
# opt_list = ['O0', 'O1']
O0_str = opt_list[0]
O1_str = opt_list[1]
O2_str = opt_list[2]
O3_str = opt_list[3]



def _find_files(project):
    ret_files = {}
    for opt in opt_list:
        file = glob.glob(os.path.join(project, '*' + opt + '*extract.pkl'))
        if len(file) == 0:
            continue
        file = file[0]
        ret_files[opt] = file
    return ret_files

def collect_func_from_one_project(project):
    # find all files
    files = _find_files(project)
    if len(files) == 0:
        return []
    # load all files
    funcs = []
    loaded_files = {}
    for opt, file in files.items():
        loaded_files[opt] = pickle.load(open(file, 'rb'))

    all_valid_func_names = set()
    first = True
    for loaded_file in loaded_files.values():
        if first:
            all_valid_func_names = set(loaded_file.keys())
            first = False
        all_valid_func_names = all_valid_func_names & set(loaded_file.keys())
    
    
    all_valid_func_names = all_valid_func_names - rule_out_funcs
    for func_name in all_valid_func_names:
        ret = {}
        ret['func_name'] = func_name
        ret['project_name'] = os.path.basename(project)
        for opt, loaded_file in loaded_files.items():
            ret[opt] = {}
            ret[opt]['cfg'] = loaded_file[func_name][3]

        funcs.append(ret)
    return funcs



def main():
    args = parse_args()
    print(args)

    if args.sample_in == '':
        all_projects = [os.path.join(args.dataset, f) for f in os.listdir(args.dataset) if os.path.isdir(
            os.path.join(args.dataset, f))]

        # shuffle projects
        random.shuffle(all_projects)
        if args.sample_size > 0:
            all_projects = all_projects[:args.sample_size]
        pbar = tqdm(total=len(all_projects))
        all_funcs = []
        current_idx = 0
        for project in all_projects:
            current_idx += 1
            funcs = collect_func_from_one_project(project)
            all_funcs.extend(funcs)
            pbar.update(1)

    # save all_funcs
    pickle.dump(all_funcs, open(args.fout, 'wb'))
    print("Save to %s" % args.fout)



if __name__ == '__main__':
    from diemph_utils import asm_parser
    asm_parser.parse_asm('imul    rdx, rsi, -28h', ori=True)
    main()



if 0==1:
    pass

from diemph_utils import asm_parser
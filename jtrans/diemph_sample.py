import pickle
import sys
from datautils.playdata import DatasetBase as DatasetBase
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import argparse
from data import FunctionDataset_CL
from transformers import BertTokenizer
from matplotlib import pyplot as plt
import os
from diemph_data_util import load_data, dump_cfg, parse_cfg
import random

# set random seeds
# torch.manual_seed(233)
# np.random.seed(233)
# random.seed(233)
SEED=1024
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


class FunctionDataset_Fast(torch.utils.data.Dataset):
    def __init__(self, arr1, arr2):
        self.arr1 = arr1
        self.arr2 = arr2
        assert(len(arr1) == len(arr2))

    def __getitem__(self, idx):
        return self.arr1[idx].squeeze(0), self.arr2[idx].squeeze(0), torch.tensor([idx])

    def __len__(self):
        return len(self.arr1)


ruleout_names = set(
    {'frame_dummy', 'register_tm_clones', 'libc_csu_init', 'libc_csu_fini', 'global_dtors_aux', 'global_dtors_aux_fini_array_entry',
     '_start', '_init_proc', '_fini_proc',
     '__do_global_dtors_aux',
        '__libc_csu_init',
        '__libc_csu_fini',
        '__libc_start_main',
        'deregister_tm_clones', 
     })

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="jTrans-FastEval")
    # parser.add_argument("--experiment_path", type=str, default='./experiments/BinaryCorp-3M/jTrans.pkl', help="experiment to be evaluated")
    parser.add_argument("--query_path", type=str,
                        default='hard-experiments/findh/jtrans-O0-no-inline.prep.pkl', help="experiment to be evaluated")
    parser.add_argument("--candi_path", type=str,
                        default='hard-experiments/findh/jtrans-O3-no-inline.prep.pkl', help="experiment to be evaluated")
    parser.add_argument("--sample_size", type=int,
                        default=1000, help="size of the function pool")
    parser.add_argument("--additional_pool_size", type=int,
                        default=0, help="size of the function pool")
    parser.add_argument("--data_record_path", type=str,
                        default='foobar.pkl', help="data record")    


    additional_rule_out = pickle.load(open('tmp-binkit-func-names.pkl', 'rb'))
    ruleout_names = ruleout_names.union(additional_rule_out)
    additional_rule_out = pickle.load(open('tmp-hows-func-names.pkl', 'rb'))
    ruleout_names = ruleout_names.union(additional_rule_out)
    print("ruleout_names", len(ruleout_names))
    
    args = parser.parse_args()

    query_bin_file = open(args.query_path, 'rb')
    query_bin = pickle.load(query_bin_file)
    query_bin_file.close()
    candi_bin_file = open(args.candi_path, 'rb')
    candi_bin = pickle.load(candi_bin_file)
    candi_bin_file.close()

    MINIMAL_LENGTH = 5

    all_query_func_names = set()
    all_query_func_names_no_proj = set()
    for item in query_bin:
        if item['funcname'] in all_query_func_names_no_proj:
            continue
        if item['funcname'] in ruleout_names:
            continue
        cfg = item['ori_cfg']
        nodes = [cfg.nodes[node] for node in cfg.nodes]
        all_len = 0
        for node in nodes:
            all_len += len(node['asm'])
            if all_len > MINIMAL_LENGTH + 3:
                break
        if all_len < MINIMAL_LENGTH:
            continue
        all_query_func_names.add((item['project_name'], item['funcname']))
        all_query_func_names_no_proj.add(item['funcname'])

    raw_candi_func_names = set([(item['project_name'], item['funcname']) for item in candi_bin])    
    all_candi_func_names = raw_candi_func_names & all_query_func_names
    all_candi_func_names_no_proj = set([item[1] for item in all_candi_func_names])
    for item in candi_bin:
        if item['funcname'] in all_candi_func_names_no_proj:
            continue
        if item['funcname'] in ruleout_names:
            continue
        cfg = item['ori_cfg']
        nodes = [cfg.nodes[node] for node in cfg.nodes]
        all_len = 0
        for node in nodes:
            all_len += len(node['asm'])
            if all_len > MINIMAL_LENGTH + 3:
                break
        if all_len < MINIMAL_LENGTH:
            continue
        all_candi_func_names.add((item['project_name'], item['funcname']))
        all_candi_func_names_no_proj.add(item['funcname'])

    print('query size: ', len(all_query_func_names))
    print('candi size: ', len(all_candi_func_names))
    print('raw query size: ', len(query_bin))
    print('raw candi size: ', len(candi_bin))

    possible_query = all_query_func_names & all_candi_func_names
    print('possible query size: ', len(possible_query))

    print("Now I attempt to sample %d functions from the query set" %
          args.sample_size)

    assert(args.sample_size <= len(possible_query))
    sampled_query = random.sample(possible_query, args.sample_size)
    sampled_candidates = set(sampled_query)
    other_candidates = random.sample((all_candi_func_names - sampled_candidates),
                                        args.additional_pool_size)
    sampled_candidates = list(sampled_candidates) + other_candidates
    print("Now I have %d functions in the sampled candidate set" %
            len(sampled_candidates))
    rcd = {
        'selected_query_func_names': sampled_query,
        'candidate_func_names': sampled_candidates
    }
    fout = open(args.data_record_path, 'wb')
    pickle.dump(rcd, fout)
    fout.close()
    print("data record saved to %s" % args.data_record_path)
    
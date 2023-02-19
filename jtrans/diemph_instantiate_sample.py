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
from eval_save import BinBertModel
import copy

# set random seeds
torch.manual_seed(233)
np.random.seed(233)

def get_func_ebd(ebds, opt='O3'):
    funcarr=[]
    dataset2ebd_idx={}
    ebd2dataset_idx={}
    idx = 0
    all_len = len(ebds)
    for i in tqdm(range(len(ebds))):        
        if ebds[i].get(opt) is not None and type(ebds[i][opt]) is not int:
            ebd=ebds[i][opt]
            funcarr.append(ebd / ebd.norm())
            dataset2ebd_idx[idx]=i
            ebd2dataset_idx[i]=idx
            idx += 1
        else:
            continue
    print()
    return funcarr, dataset2ebd_idx, ebd2dataset_idx




class FunctionDataset_Fast(torch.utils.data.Dataset): 
    def __init__(self,arr1,arr2): 
        self.arr1=arr1
        self.arr2=arr2
        assert(len(arr1)==len(arr2))
    def __getitem__(self, idx):            
        return self.arr1[idx].squeeze(0),self.arr2[idx].squeeze(0), torch.tensor([idx])
    def __len__(self):
        return len(self.arr1)


def load_binfolder_func_dataset(functions, candidates_names, baseline=False):        
    already_in = set()    
    ret = []    
    for i in tqdm(range(len(functions))):              
        if (functions[i]['project_name'], functions[i]['funcname']) not in candidates_names:
            continue
        if (functions[i]['project_name'], functions[i]['funcname']) in already_in:
            continue
        already_in.add((functions[i]['project_name'], functions[i]['funcname']))
        ret.append(functions[i])
    return ret
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="jTrans-FastEval")
    # parser.add_argument("--experiment_path", type=str, default='./experiments/BinaryCorp-3M/jTrans.pkl', help="experiment to be evaluated")
    # parser.add_argument("--O0", type=str, default='experiments/BinaryCorp-3M/jTrans-inline-O0-all.pkl', help="experiment to be evaluated")
    parser.add_argument("--O0", type=str, default='experiments/curl/jtrans-O0-no-inline.prep.pkl', help="experiment to be evaluated")
    parser.add_argument("--O3", type=str, default = 'experiments/curl/jtrans-O3-no-inline.prep.pkl', help="experiment to be evaluated")
    parser.add_argument("--sample", type=str, default = 'samples-nothard/libcurl-sample-500-500.pkl')
    # parser.add_argument("--out", type=str, default = 'samples/opensslh-sample-500-500.instance.pkl')
    parser.add_argument("--out", type=str, default = 'tmp.pkl')
    args = parser.parse_args()



    sample_info = pickle.load(open(args.sample, 'rb'))
    rcd_candidates_function_names = sample_info['candidate_func_names']
    rcd_query_function_names = sample_info['selected_query_func_names']    
        
    fin=open(args.O3,'rb')
    functions_candidates=pickle.load(fin)
    fin.close()

    fin=open(args.O0,'rb')
    functions_query=pickle.load(fin)
    fin.close()

    print("Load candidate functions...")
    candidates_func = load_binfolder_func_dataset(functions_candidates, rcd_candidates_function_names, baseline=True)
    print("Load query functions...")
    query_func = load_binfolder_func_dataset(functions_query, rcd_query_function_names, baseline=True)
    
    for f in candidates_func:
        if 'emb' in f:
            del f['emb']
        if 'baseline_emb' in f:
            del f['baseline_emb']
        if 'cfg' in f and 'ori_cfg' in f:
            del f['cfg']                    
        f['cfg'] = f['ori_cfg']
        del f['ori_cfg']

    for f in query_func:
        if 'emb' in f:
            del f['emb']
        if 'baseline_emb' in f:
            del f['baseline_emb']
        if 'cfg' in f and 'ori_cfg' in f:
            del f['cfg']                    
        f['cfg'] = f['ori_cfg']
        del f['ori_cfg']

    to_save = {
        'query_func': query_func,
        'candidates_func': candidates_func
    }
    print("Successfully loaded %d query functions and %d candidate functions" % (len(query_func), len(candidates_func)))
    pickle.dump(to_save, open(args.out, 'wb'))
    print("Saved to ", args.out)    
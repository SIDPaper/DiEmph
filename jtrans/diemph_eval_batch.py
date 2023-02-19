import pickle
import sys
from datautils.playdata import DatasetBase as DatasetBase
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import argparse
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
    funcarr=[]
    funcarr_baseline=[]
    dataset2ebd_idx={}
    ebd2dataset_idx={}
    name2dataset_idx={}
    already_in = set()
    idx = 0
    all_len = len(functions)
    for i in tqdm(range(len(functions))):              
        if (functions[i]['project_name'], functions[i]['funcname']) not in candidates_names:
            continue
        if (functions[i]['project_name'], functions[i]['funcname']) in already_in:
            continue
        already_in.add((functions[i]['project_name'], functions[i]['funcname']))
        if baseline:
            # backword compatibility
            if 'baseline_emb' in functions[i]:
                ebd=functions[i]['baseline_emb']
            else:
                ebd=functions[i]['emb']
            funcarr_baseline.append(ebd / ebd.norm())
        
        ebd=functions[i]['emb']
        funcarr.append(ebd / ebd.norm())
        dataset2ebd_idx[idx]=i
        ebd2dataset_idx[i]=idx
        name2dataset_idx[(functions[i]['project_name'], functions[i]['funcname'])]=idx
        idx += 1
        
    return funcarr, funcarr_baseline, dataset2ebd_idx, ebd2dataset_idx, name2dataset_idx
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="jTrans-FastEval")
    parser.add_argument("--O0", type=str, default='motivation/moti.O0.ori.pkl', help="experiment to be evaluated")
    parser.add_argument("--O3", type=str, default = 'motivation/moti.O3.ori.pkl', help="experiment to be evaluated")
    parser.add_argument("--sample", type=str, default = 'motivation/moti-ex.pkl')
    parser.add_argument("--dbg_out", type=str, default = '')
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
    candidates_func_arr, candidates_func_arr_baseline, candidates_dataset2ebd_idx, candidates_ebd2dataset_idx, candidates_name2dataset_idx = load_binfolder_func_dataset(functions_candidates, rcd_candidates_function_names, baseline=True)
    print("Load query functions...")
    query_func_arr, query_func_arr_baseline, query_dataset2ebd_idx, query_ebd2dataset_idx, _ = load_binfolder_func_dataset(functions_query, rcd_query_function_names, baseline=True)
    
    candidates_func_tensors = torch.stack(candidates_func_arr).squeeze(1)
    query_func_tensors = torch.stack(query_func_arr).squeeze(1)

    candidates_func_tensors_baseline = torch.stack(candidates_func_arr_baseline).squeeze(1)
    query_func_tensors_baseline = torch.stack(query_func_arr_baseline).squeeze(1)

    simlarities= torch.mm(query_func_tensors, candidates_func_tensors.t())
    simlarities = simlarities.cpu().numpy()
    simlarities_baseline= torch.mm(query_func_tensors_baseline, candidates_func_tensors_baseline.t())
    simlarities_baseline = simlarities_baseline.cpu().numpy()

    results = []
    dbg_results = []    
    for i in range(len(query_func_arr)):        
        query_ds_idx = query_dataset2ebd_idx[i]

        func_cfg = functions_query[query_ds_idx]['cfg']
        use_baseline = len(func_cfg.nodes) < 2
        # use_baseline = False
        query_func_name = (functions_query[query_ds_idx]['project_name'], functions_query[query_ds_idx]['funcname'])
        gt_idx = candidates_name2dataset_idx[query_func_name]
        
        current_similarity = simlarities[i]
        rankings = np.argsort(-current_similarity)        
        rank = np.where(rankings == gt_idx)[0][0]
        
        current_baseline_similarity = simlarities_baseline[i]
        rankings_baseline = np.argsort(-current_baseline_similarity)
        rank_baseline = np.where(rankings_baseline == gt_idx)[0][0]
        
        if use_baseline:
            all_similarity = 0.3*current_similarity + 0.7*current_baseline_similarity
        else:
            all_similarity = current_similarity + current_baseline_similarity
        
        # all_similarity = current_similarity

        rankings_all = np.argsort(-all_similarity)
        rank_all = np.where(rankings_all == gt_idx)[0][0]

        # deal with tie
        if rank_all > 9 and all_similarity[gt_idx] >= all_similarity[rankings_all[9]]:
            rank_all = 9.5
        if rank_all > 4 and all_similarity[gt_idx] >= all_similarity[rankings_all[4]]:
            rank_all = 4.5
        if rank_all > 0 and all_similarity[gt_idx] >= all_similarity[rankings_all[0]]:
            rank_all = 0.5

        
        ret_rank = rank_all
        
        results.append(ret_rank)
        
        if use_baseline:
            dbg_rankings = rankings_baseline
        else:
            dbg_rankings = rankings
        # top 20 name
        top20 = []
        for idx in dbg_rankings[:20]:
            candidate_ebd_idx = candidates_dataset2ebd_idx[idx]
            dbg_func = functions_candidates[candidate_ebd_idx]
            top20.append((dbg_func['project_name'], dbg_func['funcname']))

        dbg_results.append((query_func_name, ret_rank, rank, rank_baseline, top20, use_baseline))

    if args.dbg_out != '':
        dbg_fout = open(args.dbg_out, 'wb')
        pickle.dump(dbg_results, dbg_fout)
        dbg_fout.close()
        print("Dumped debug results to {}".format(args.dbg_out))

    print("Total results: %d" % (len(results)))
    print("mean rank: %f" % (np.mean(results)))
    print("median rank: %f" % (np.median(results)))
    # pr1
    print("pr1: %f" % (np.mean(np.array(results) < 1)))
    # pr5
    print("pr5: %f" % (np.mean(np.array(results) < 5)))
    # pr10
    print("pr10: %f" % (np.mean(np.array(results) < 10)))
    print("Len of query: %d" % (len(query_func_arr)))
    print("Len of candidates: %d" % (len(candidates_func_arr)))


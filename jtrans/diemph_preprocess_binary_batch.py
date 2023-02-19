from transformers import BertTokenizer, BertForMaskedLM, BertModel
from tokenizer import *
import pickle
from torch.utils.data import DataLoader
import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from data import diemph_gen_funcstr_wraper, help_tokenize, load_paired_data,FunctionDataset_CL
from transformers import AdamW
import torch.nn.functional as F
import argparse
import wandb
import logging
import sys
import time
import data
from eval_save import BinBertModel
import readidadata
from readidadata import ADDR_IDX,ASM_IDX,RAW_IDX,CFG_IDX
import networkx as nx
from diemph_process_one_binary import extract_call_targets
from diemph_inline_helper import InlineHelper

def get_args():
  parser = argparse.ArgumentParser(description="jTrans-EvalSave")
  # parser.add_argument("--model_path", type=str, default='./models/jTrans-finetune', help="Path to the model")
  # parser.add_argument("--dataset_path", type=str, default='./BinaryCorp/small_test', help="Path to the dataset")
  # parser.add_argument("--binary_list_file", type=str, default='bins-O3.txt', help="Path to the dataset")
  parser.add_argument("--binary_list_file", type=str, default='coreutils-bin-O0.txt', help="Path to the dataset")
  # parser.add_argument("--experiment_path", type=str, default='./experiments/BinaryCorp-3M/jTrans-trial.pkl', help="Path to the experiment")
  # parser.add_argument("--tokenizer", type=str, default='./jtrans_tokenizer/')
  # parser.add_argument("--fout", type=str, default='./experiments/BinaryCorp-3M/jTrans-inline-O3.pkl')
  parser.add_argument("--fout", type=str, default='tmp.pkl')
  parser.add_argument("--inline-num", type=int, default=20)
  parser.add_argument("--ori", type=bool, default=False)
  args = parser.parse_args()
  return args



if __name__ == '__main__':
  MAX_LEN = 512
  args = get_args()
  print(args)
  # load picked binary
  fin = open(args.binary_list_file, 'r')
  binary_list = fin.readlines()
  fin.close()
  binfolder_binary_entries = []
  all_binary_len = len(binary_list)
  print("Loading binaries ...")
  
  for b in tqdm(binary_list):    
    project_name = os.path.basename(os.path.dirname(b))
    bin_fin = open(b.strip(), 'rb')
    binary = pickle.load(bin_fin)
    bin_fin.close()
    addr2function = {}
    for name, entry in binary.items():
      addr2function[entry[0]] = entry

    inline_helper = InlineHelper(addr2function)

    for name, entry in tqdm(binary.items()):
      my_cfg = entry[CFG_IDX]
      if len(my_cfg.nodes()) < args.inline_num and (not args.ori):
        new_cfg, logical_order = inline_helper.inline_func(entry[ADDR_IDX])        
        blk_size = [len(n[1]['asm']) for n in logical_order]
        acc = 0
        acc_blk_size = {}
        for i in range(len(blk_size)):
          acc += blk_size[i]
          node_addr = logical_order[i][0]
          acc_blk_size[node_addr] = acc
          
        for n in new_cfg.nodes():
          node = new_cfg.nodes[n]
          if node['num'] == 0 and acc_blk_size[n] > (MAX_LEN*1.2):
            node['num'] = 9999
      else:
        new_cfg = my_cfg
        func_addr = entry[ADDR_IDX]
        new_cfg.nodes[func_addr]['num'] = -1
        logical_order = [(n, my_cfg.nodes[n]) for n in my_cfg.nodes()]
      
      binfolder_binary_entries.append({
        'project_name': project_name,
        'funcname': name,
        'funcaddr': entry[ADDR_IDX],
        'cfg': new_cfg,
        'ori_cfg': my_cfg,
        'dbg_logical_order': logical_order,
      })

  fout = open(args.fout, 'wb')
  pickle.dump(binfolder_binary_entries, fout)
  fout.close()
  exit(0)

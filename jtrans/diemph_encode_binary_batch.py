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


def get_args():
  parser = argparse.ArgumentParser(description="jTrans-EvalSave")
  parser.add_argument("--model-path", type=str, default='', help="Path to the model")
  parser.add_argument("--binary-entry-path", type=str, default='', help="Path to the dataset")
  parser.add_argument("--tokenizer", type=str, default='./jtrans_tokenizer/')
  parser.add_argument("--fout", type=str, default='tmp.pkl')  
  parser.add_argument("--ori", type=bool, default=False)
  parser.add_argument("--baseline-model", type=str, default='')
  parser.add_argument("--sample", type=str, default='')
  parser.add_argument("--rewrite-strategy", type=str, default='')
  parser.add_argument("--mix-baseline", type=bool, default=False)
  args = parser.parse_args()
  return args



if __name__ == '__main__':
  MAX_LEN = 512
  args = get_args()
  # args.mix_baseline = False
  print(args)
  print("Loading binaries ...")
  # load picked binary
  fin = open(args.binary_entry_path, 'rb')
  binfolder_binary_entries = pickle.load(fin)
  fin.close()
  print("Loaded %d entries" % len(binfolder_binary_entries))
  
  model = BinBertModel.from_pretrained(args.model_path)
  model.eval()
  device = torch.device("cuda")
  model.to(device)
  
  if args.baseline_model != '':
    baseline_model = BinBertModel.from_pretrained(args.baseline_model)
    baseline_model.eval()
    baseline_model.to(device)
  else:
    baseline_model = model

  tokenizer = BertTokenizer.from_pretrained(args.tokenizer)
  ret = []
  rewrite_strategy = args.rewrite_strategy
  if not args.ori and args.rewrite_strategy == '': 
    if 'how' in args.model_path:
      rewrite_strategy = 'how'
    elif 'binkit' in args.model_path:
      rewrite_strategy = 'binkit'
    elif 'binarycorp' in args.model_path:
      rewrite_strategy = 'binarycorp'
    else:
      rewrite_strategy = 'jtrans'
      
  print("Encoding with strategy %s..."%rewrite_strategy)    
  selected_names = set()
  if args.sample != '':
    samples = pickle.load(open(args.sample, 'rb'))
    query_names = samples['selected_query_func_names']
    for name in query_names:
      selected_names.add(name)
    candi_names = samples['candidate_func_names']
    for name in candi_names:
      selected_names.add(name)

  for entry in tqdm(binfolder_binary_entries):
    current_name = (entry['project_name'],entry['funcname'])
    if args.sample != '' and current_name not in selected_names:
      continue
    # inlined
    cfg = entry['cfg']
    funcstr = diemph_gen_funcstr_wraper(cfg, ori=args.ori, rewrite_strategy=rewrite_strategy)
    tokenizer_ret = tokenizer(funcstr, add_special_tokens=True,max_length=512,padding='max_length',truncation=True,return_tensors='pt')
    input_ids = tokenizer_ret['input_ids'].to('cuda')
    attention_mask = tokenizer_ret['attention_mask'].to('cuda')
    output = model(input_ids, attention_mask=attention_mask)
    pooler_output = output.pooler_output
    if args.mix_baseline:
      funcstr = diemph_gen_funcstr_wraper(cfg, ori=True)
      tokenizer_ret = tokenizer(funcstr, add_special_tokens=True,max_length=512,padding='max_length',truncation=True,return_tensors='pt')
      input_ids = tokenizer_ret['input_ids'].to('cuda')
      attention_mask = tokenizer_ret['attention_mask'].to('cuda')
      baseline_output = baseline_model(input_ids, attention_mask=attention_mask)
      baseline_pooler_output = baseline_output.pooler_output          
    else:
      baseline_pooler_output = pooler_output

    
    
    ret.append(
      {
        'project_name': entry['project_name'],
        'funcname': entry['funcname'],
        'funcaddr': entry['funcaddr'],
        'cfg': entry['cfg'],
        'ori_cfg': entry['ori_cfg'],
        'emb': pooler_output.cpu().detach(),
        'baseline_emb': baseline_pooler_output.cpu().detach(),
      }
    )
  fout = open(args.fout, 'wb')
  pickle.dump(ret, fout)
  fout.close()


  #   current


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
import networkx as nx

def get_args():
  parser = argparse.ArgumentParser(description="jTrans-EvalSave")
  parser.add_argument("--model_path", type=str, default='./models/jTrans-finetune', help="Path to the model")
  # parser.add_argument("--dataset_path", type=str, default='./BinaryCorp/small_test', help="Path to the dataset")
  parser.add_argument("--binary_path", type=str, default='small_test/vnstat-vnstatd/vnstat-vnstatd-O0-33f0082f67ff37b9850f68f86ebc6b82_extract.pkl', help="Path to the dataset")
  # parser.add_argument("--experiment_path", type=str, default='./experiments/BinaryCorp-3M/jTrans-trial.pkl', help="Path to the experiment")
  parser.add_argument("--tokenizer", type=str, default='./jtrans_tokenizer/')
  parser.add_argument("--encode", action='store_true', help="Encode the dataset")
  parser.add_argument("--inline-num", type=int, default=3)
  args = parser.parse_args()
  return args


def extract_call_targets(cfg):
  bb_ls=[]
  targets = set()
  for bb in cfg.nodes:
      bb_ls.append(bb)
  bb_ls.sort()
  for bx in range(len(bb_ls)):
    bb=bb_ls[bx]
    asm=cfg.nodes[bb]['asm']
    for code in asm:
      operator,operand1,operand2,operand3,annotation = readidadata.parse_asm(code, ori=False)
      if operator == 'SKIP':
        continue
      if 'call' in operator or 'jmp' in operator:
        if operand1.startswith('hex_'):
          targets.add(int(operand1[4:], 16))
  return targets

def extract_call_targets_one_block(bb):
  asm = bb['asm']
  targets = set()
  for code in asm:
    operator,operand1,operand2,operand3,annotation = readidadata.parse_asm(code, ori=False)
    if operator == 'SKIP':
      continue
    if 'call' in operator or 'jmp' in operator:
      if operand1.startswith('hex_'):
        targets.add(int(operand1[4:], 16))
  return targets




if __name__ == '__main__':
  args = get_args()
  
  # load picked binary
  fin = open(args.binary_path, 'rb')
  binary = pickle.load(fin)
  fin.close()

  addr2function = {}
  for name, entry in binary.items():
    addr2function[entry[0]] = entry

  binfolder_binary_entries = {}

  for name, entry in binary.items():
    my_cfg = entry[3]
    targets = extract_call_targets(my_cfg)
    if len(my_cfg.nodes()) < args.inline_num:
      callee_cfgs = []
      for target in targets:
        if target in addr2function:
          callee_cfgs.append(addr2function[target][3])
        
      # merge my_cfg and callee_cfgs
      merged_cfg = my_cfg.copy()
      i = 0
      for callee_cfg in callee_cfgs:
        for node in callee_cfg.nodes():
          node['num'] = i
        i += 1
        merged_cfg = nx.compose(merged_cfg, callee_cfg)
      binfolder_binary_entries[name] = (entry[0], entry[1], entry[2], merged_cfg)
    else:
      binfolder_binary_entries[name] = entry

  # write to file, append .binfolder.pkl
  fout = open(args.binary_path + '.binfolder.pkl', 'wb')
  pickle.dump(binfolder_binary_entries, fout)
  fout.close()  

  print("Done, saved to {}".format(args.binary_path + '.binfolder.pkl'))

  if args.encode:
    model = BinBertModel.from_pretrained(args.model_path)
    model.eval()
    device = torch.device("cuda")
    model.to(device)
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)
    all_data = [term for term in binfolder_binary_entries.items()]
    BATCH_SIZE = 8
    begin = 0
    end = -1
    ret = {}
    while begin < len(all_data):
      end = begin+BATCH_SIZE
      current_batch = all_data[begin:end]
      batch_input_ids = []
      batch_attention_mask = []
      for name, entry in current_batch:
        cfg = entry[3]
        funcstr = diemph_gen_funcstr_wraper(cfg)
        tokenizer_ret = tokenizer(funcstr, add_special_tokens=True,max_length=512,padding='max_length',truncation=True,return_tensors='pt')
        input_ids = tokenizer_ret['input_ids'].to('cuda')
        attention_mask = tokenizer_ret['attention_mask'].to('cuda')
        batch_input_ids.append(input_ids)
        batch_attention_mask.append(attention_mask)
      batch_input_ids = torch.cat(batch_input_ids, dim=0)
      batch_attention_mask = torch.cat(batch_attention_mask, dim=0)
      output=model(input_ids=batch_input_ids,attention_mask=batch_attention_mask)
      output = output.pooler_output
      for i in range(len(current_batch)):
        name, entry = current_batch[i]
        ret[name] = output[i].detach().cpu()
      print("\r%d/%d"%(begin, len(all_data)), end='')
      begin = end
      
    fout = open(args.binary_path + '.binfolder.encoded.pkl', 'wb')
    pickle.dump(ret, fout)
    fout.close()
    print("Done, saved to {}".format(args.binary_path + '.binfolder.encoded.pkl'))




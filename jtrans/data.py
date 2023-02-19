import sys
from datautils.playdata import DatasetBase as DatasetBase
import networkx
import os
import networkx as nx
from collections import defaultdict
from tqdm import tqdm
import pickle
import argparse
import re
import readidadata
import torch
import random
import time
MAXLEN=512

vocab_data = open("./jtrans_tokenizer/vocab.txt").read().strip().split("\n") + ["[SEP]", "[PAD]", "[CLS]", "[MASK]"]
my_vocab = defaultdict(lambda: 512, {vocab_data[i] : i for i in range(len(vocab_data))})

def help_tokenize(line):
    global my_vocab
    ret = {}
    split_line = line.strip().split(' ')
    split_line_len = len(split_line)
    if split_line_len <= 509:
        split_line = ['[CLS]']+split_line+['[SEP]']
        attention_mask = [1] * len(split_line) + [0] * (512 - len(split_line))
        split_line = split_line + (512-len(split_line))*['[PAD]']
    else:
        split_line = ['[CLS]'] + split_line[:510] + ['[SEP]']
        attention_mask = [1]*512
    input_ids = [my_vocab[e] for e in split_line]
    ret['input_ids'] = torch.tensor(input_ids, dtype=torch.long)
    ret['attention_mask'] = torch.tensor(attention_mask, dtype=torch.long)
    return ret

def gen_funcstr(f,convert_jump, ori, rewrite_strategy):
    cfg=f[3]
    #print(hex(f[0]))
    bb_ls,code_lst,map_id=[],[],{}
    for bb in cfg.nodes:
        if 'num' in cfg.nodes[bb]:
            bb_ls.append((bb,cfg.nodes[bb]['num']))
        else:
            bb_ls.append((bb,0))
    bb_ls.sort(key=lambda x:(x[1],x[0]))    
    context_addrs = set(cfg.nodes)
    for bx in range(len(bb_ls)):
        bb, _=bb_ls[bx]
        asm=cfg.nodes[bb]['asm']
        map_id[bb]=len(code_lst)        
        # code_lst.append('%d-%x'%(bb, bb))
        for code in asm:
            operator,operand1,operand2,operand3,annotation=readidadata.parse_asm(code, ori=ori, rewrite_strategy=rewrite_strategy, binfolder_ctx=context_addrs)
            if operator == 'SKIP':
                continue
            code_lst.append(operator)
            if operand1!=None:
                code_lst.append(operand1)
            if operand2!=None:
                code_lst.append(operand2)
            if operand3!=None:
                code_lst.append(operand3)
    # print(map_id)
    for c in range(len(code_lst)):
        op=code_lst[c]
        if op.startswith('hex_'):
            jumpaddr=int(op[4:],base=16)
            if jumpaddr in map_id:
                jumpid=map_id[jumpaddr]
                if jumpid < MAXLEN:
                    code_lst[c]='JUMP_ADDR_{}'.format(jumpid)
                else:
                    code_lst[c]='JUMP_ADDR_EXCEEDED'
            else:
                # print("ERROR: jumpaddr not found",hex(jumpaddr))
                code_lst[c]='UNK_JUMP_ADDR'
            if not convert_jump:
                code_lst[c]='CONST'
    func_str=' '.join(code_lst)
    return func_str


def diemph_gen_funcstr_wraper(cfg, ori=False, rewrite_strategy=''):
    return gen_funcstr([0,0,0,cfg], True, ori=ori, rewrite_strategy=rewrite_strategy)

def load_unpair_data(datapath,filt=None,alldata=True,convert_jump=True,opt=None, fp=None, ori=False):
    dataset = DatasetBase(datapath,filt, alldata)
    dataset.load_unpair_data()
    functions=[]
    for i in dataset.get_unpaird_data():  #proj, func_name, func_addr, asm_list, rawbytes_list, cfg, bai_featrue
        f = (i[2], i[3], i[4], i[5], i[6])
        func_str=gen_funcstr(f,convert_jump, ori=ori)
        if len(func_str) > 0:
            fp.write(func_str+"\n")

def load_paired_data(datapath, ori, rewrite_strategy, filt=None,alldata=True,convert_jump=True,opt=None,add_ebd=False):
   
    dataset = DatasetBase(datapath,filt,alldata, opt=opt)
    functions=[]
    func_emb_data=[]
    SUM=0
    for i in tqdm(dataset.get_paired_data_iter()):  #proj, func_name, func_addr, asm_list, rawbytes_list, cfg, bai_featrue
        functions.append([])
        if add_ebd:
            func_emb_data.append({'proj':i[0],'funcname':i[1]})
        for o in opt:
            if i[2].get(o):                                  
                f=i[2][o]
                func_str=gen_funcstr(f,convert_jump, ori=ori, rewrite_strategy=rewrite_strategy)
                if len(func_str)>0:                    
                    if add_ebd:
                        func_emb_data[-1][o]=len(functions[-1])
                    functions[-1].append(func_str)
                    SUM+=1        
    print('TOTAL ',SUM)
    return functions,func_emb_data

class FunctionDataset_CL(torch.utils.data.Dataset): #binary version dataset
    def __init__(self,tokenizer,path='../BinaryCorp/extract',filt=None,alldata=True,convert_jump_addr=True,opt=None,add_ebd=True, ori=False):  #random visit
        functions,ebds=load_paired_data(datapath=path,filt=filt,alldata=alldata,convert_jump=convert_jump_addr,opt=opt,add_ebd=add_ebd, ori=ori)
        self.datas=functions
        self.ebds=ebds
        self.tokenizer=tokenizer
        self.opt=opt
        self.convert_jump_addr=True
    def __getitem__(self, idx):             #also return bad pair

        pairs=self.datas[idx]
        while len(pairs)<2:
            idx=random.randint(0,len(self.datas)-1)
            pairs=self.datas[idx]
        if self.opt==None:
            pos=random.randint(0,len(pairs)-1)
            pos2=random.randint(0,len(pairs)-1)
            while pos2==pos:
                pos2=random.randint(0,len(pairs)-1)
            f1=pairs[pos]   #give three pairs
            f2=pairs[pos2]
        else:
            pos=random.randint(0,len(pairs)-1)
            pos2=random.randint(0,len(pairs)-1)
            while pos2==pos:
                pos2=random.randint(0,len(pairs)-1)
            # pos=0
            # pos2=1
            f1=pairs[pos]
            f2=pairs[pos2]
        ftype=random.randint(0,len(self.datas)-1)
        while ftype==idx:
            ftype=random.randint(0,len(self.datas)-1)
        pair_opp=self.datas[ftype]
        pos3=random.randint(0,len(pair_opp)-1)
        f3=pair_opp[pos3]
        ret1 = help_tokenize(f1)
        token_seq1=ret1['input_ids']
        mask1=ret1['attention_mask']

        ret2 = help_tokenize(f2)
        token_seq2=ret2['input_ids']
        mask2=ret2['attention_mask']

        ret3 = help_tokenize(f3)
        token_seq3=ret3['input_ids']
        mask3=ret3['attention_mask']

        return token_seq1,token_seq2,token_seq3,mask1,mask2,mask3
    def __len__(self):
        return len(self.datas)

class FunctionDataset_CL_Load(torch.utils.data.Dataset): #binary version dataset
    def __init__(self,tokenizer,path='../BinaryCorp/extract',filt=None,alldata=True,convert_jump_addr=True,opt=None,add_ebd=True, load=None, ori=True, rewrite_strategy=''):  #random visit
        if load:
            start = time.time()
            self.datas = pickle.load(open(load, 'rb'))
            print('load time:', time.time() - start)
            print("load data size:", len(self.datas))
            self.tokenizer=tokenizer
            self.opt=opt
            self.convert_jump_addr=True
        else:
            print("rewrite strategy:", rewrite_strategy)
            functions,ebds=load_paired_data(datapath=path,filt=filt,alldata=alldata,convert_jump=convert_jump_addr,opt=opt,add_ebd=add_ebd, ori=ori, rewrite_strategy=rewrite_strategy)
            self.datas=[]
            for func_list in functions:
                tmp = []
                for f in func_list:
                    tmp.append(help_tokenize(f))
                self.datas.append(tmp)
            self.ebds=ebds
            self.tokenizer=tokenizer
            self.opt=opt
            self.convert_jump_addr=True
    def __getitem__(self, idx):             #also return bad pair

        pairs=self.datas[idx]
        while len(pairs)<2:
            idx=random.randint(0,len(self.datas)-1)
            pairs=self.datas[idx]
        if self.opt==None:
            pos=random.randint(0,len(pairs)-1)
            pos2=random.randint(0,len(pairs)-1)
            while pos2==pos:
                pos2=random.randint(0,len(pairs)-1)
            f1=pairs[pos]   #give three pairs
            f2=pairs[pos2]
        else:
            # pos=random.randint(0,len(pairs)-1)
            # pos2=random.randint(0,len(pairs)-1)
            # while pos2==pos:
            #     pos2=random.randint(0,len(pairs)-1)
            pos=0
            pos2=1
            f1=pairs[pos]
            f2=pairs[pos2]
        ftype=random.randint(0,len(self.datas)-1)
        while ftype==idx or len(self.datas[ftype])<2:
            ftype=random.randint(0,len(self.datas)-1)
        pair_opp=self.datas[ftype]
        pos3=random.randint(0,len(pair_opp)-1)
        f3=pair_opp[pos3]

        token_seq1=f1['input_ids']
        mask1=f1['attention_mask']

        token_seq2=f2['input_ids']
        mask2=f2['attention_mask']

        token_seq3=f3['input_ids']
        mask3=f3['attention_mask']

        return token_seq1,token_seq2,token_seq3,mask1,mask2,mask3
    def __len__(self):
        return len(self.datas)

def load_filter_list(name):
    import csv
    f=csv.reader(open(name,'r'))
    S=set()
    for i in f:
        S.add(i[1])
    return list(S)
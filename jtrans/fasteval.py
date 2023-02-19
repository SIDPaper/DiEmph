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

# set random seeds
torch.manual_seed(233)
np.random.seed(233)


def eval_O(ebds,TYPE1,TYPE2, record=False):
    funcarr1=[]
    funcarr2=[]
    dataset2ebd_idx={}
    idx = 0
    for i in range(len(ebds)):
        if ebds[i].get(TYPE1) is not None and type(ebds[i][TYPE1]) is not int:
            if ebds[i].get(TYPE2) is not None and type(ebds[i][TYPE2]) is not int:
                ebd1,ebd2=ebds[i][TYPE1],ebds[i][TYPE2]
                funcarr1.append(ebd1 / ebd1.norm())
                funcarr2.append(ebd2 / ebd2.norm())
                dataset2ebd_idx[idx]=i
                idx += 1
        else:
            continue

    ft_valid_dataset=FunctionDataset_Fast(funcarr1,funcarr2)
    dataloader = DataLoader(ft_valid_dataset, batch_size=POOLSIZE, num_workers=24, shuffle=True)
    SIMS=[]
    Recall_AT_1=[]
    all_indices=[]
    correct_posi=[]
    for idx, (anchor,pos, indices) in enumerate(tqdm(dataloader)):
        anchor = anchor.cuda()
        pos =pos.cuda()
        if anchor.shape[0]==POOLSIZE:
            for i in range(len(anchor)):    # check every vector of (vA,vB)
                vA=anchor[i:i+1]  #pos[i]
                sim = np.array(torch.mm(vA, pos.T).cpu().squeeze())
                y=np.argsort(-sim)
                posi=0
                for j in range(len(pos)):
                    if y[j]==i:
                        posi=j+1
                        break 
                correct_posi.append(posi)
                if posi==1:
                    Recall_AT_1.append(1)
                else:
                    Recall_AT_1.append(0)
                SIMS.append(1.0/posi)
                all_indices.append(indices[i])
    dbg = []
    if record:
        indices_array = torch.stack(all_indices).squeeze(-1).numpy()
        for i, idx in enumerate(indices_array):
            ebd_idx = dataset2ebd_idx[idx]
            proj_name = ebds[ebd_idx]['proj']
            func_name = ebds[ebd_idx]['funcname']
            recall = Recall_AT_1[i]
            posi = correct_posi[i]
            dbg.append((proj_name, func_name, recall, posi))
    dbg_sorted = sorted(dbg, key=lambda x:x[-1])

    print(TYPE1,TYPE2,'MRR{}: '.format(POOLSIZE),np.array(SIMS).mean())
    print(TYPE1,TYPE2,'Recall@1: ', np.array(Recall_AT_1).mean())
    return np.array(Recall_AT_1).mean()

class FunctionDataset_Fast(torch.utils.data.Dataset): 
    def __init__(self,arr1,arr2): 
        self.arr1=arr1
        self.arr2=arr2
        assert(len(arr1)==len(arr2))
    def __getitem__(self, idx):            
        return self.arr1[idx].squeeze(0),self.arr2[idx].squeeze(0), torch.tensor([idx])
    def __len__(self):
        return len(self.arr1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="jTrans-FastEval")
    # parser.add_argument("--experiment_path", type=str, default='./experiments/BinaryCorp-3M/jTrans.pkl', help="experiment to be evaluated")
    parser.add_argument("--experiment_path", type=str, default='./experiments/BinaryCorp-3M/jTrans-trial.pkl', help="experiment to be evaluated")
    parser.add_argument("--poolsize", type=int, default=500, help="size of the function pool")
    parser.add_argument("--dbg_dataset_path", type=str, default='./small_test', help="Path to the dataset")
    parser.add_argument("--dbg_tokenizer", type=str, default='./jtrans_tokenizer/')
    args = parser.parse_args()

    POOLSIZE=args.poolsize
    ff=open(args.experiment_path,'rb')
    ebds=pickle.load(ff)
    ff.close()

    print(f'evaluating...poolsize={POOLSIZE}')

    eval_O(ebds,'O0','O3', record=True)    
    # eval_O(ebds,'O0','Os')
    # eval_O(ebds,'O1','Os')
    # eval_O(ebds,'O1','O3')
    # eval_O(ebds,'O2','Os')
    eval_O(ebds,'O2','O3')

if False:
    tokenizer = BertTokenizer.from_pretrained(args.dbg_tokenizer)
    dbg_dataset= FunctionDataset_CL(tokenizer,args.dbg_dataset_path,None,True,opt=['O0', 'O3'], add_ebd=True, convert_jump_addr=True)
    all_ebds = dbg_dataset.ebds
    datas = dbg_dataset.datas
    proj_func_to_ebd = {}
    for ebd, data in zip(all_ebds, datas):
        proj_func_to_ebd[(ebd['proj'], ebd['funcname'])] = (ebd, data)

    length_2_recall = []
    missed = []
    for proj, func, recall, posi in dbg:
        proj_func = (proj, func)
        if proj_func not in proj_func_to_ebd:
            print(f'proj_func {proj_func} not in proj_func_to_ebd')            
            continue
        ebd, data = proj_func_to_ebd[proj_func]
        if 'O0' not in ebd or 'O3' not in ebd:
            missed.append(proj_func)
            continue
        idx0 = ebd['O0']
        idx3 = ebd['O3']
        data0 = data[idx0]
        data3 = data[idx3]
        len0 = len(data0)
        len3 = len(data3)
        length_2_recall.append((len0, len3, posi))

    # histogram of recall for different lengths
    # 1. histogram of recall for length less than 500
    shorter_than_500 = [x for x in length_2_recall if x[0] < 500 and x[1] < 500]
    plt.close()
    plt.hist([x[2] for x in shorter_than_500], bins=20)
    plt.title('recall for length less than 500')
    plt.xlabel('recall')
    plt.ylabel('count')
    plt.savefig('tmp_recall_less_than_500.png')
    # 2. histogram of recall for length less than 1000
    shorter_than_1000 = [x for x in length_2_recall if (500 < x[0] < 1000) or (500 < x[1] < 1000)]
    plt.close()
    plt.hist([x[2] for x in shorter_than_1000], bins=20)
    plt.title('recall for length less than 1000')
    plt.xlabel('recall')
    plt.ylabel('count')
    plt.savefig('tmp_recall_less_than_1000.png')
    # 3. histogram of recall for length larger than 1000
    longer_than_1000 = [x for x in length_2_recall if (x[0] > 1000) or (x[1] > 1000)]
    plt.close()
    plt.hist([x[2] for x in longer_than_1000], bins=20)
    plt.title('recall for length larger than 1000')
    plt.xlabel('recall')
    plt.ylabel('count')
    plt.savefig('tmp_recall_larger_than_1000.png')


    plt.close()
    plt.figure(figsize=(10, 10))
    plt.title('Length and Recall')
    plt.xlabel('Length of O0')
    plt.ylabel('Length of O3')
    plt.scatter([x[0] for x in length_2_recall], [x[1] for x in length_2_recall], c=[x[2] for x in length_2_recall], cmap='bwr', alpha=0.5)
    # set x lim
    plt.xlim(0, 1000)
    plt.ylim(0, 1000)
    plt.colorbar()
    plt.savefig('tmp-length_2_recall.png')


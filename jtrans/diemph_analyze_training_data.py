from binfolder_utils import prog_model
import random
import torch
import argparse
import pickle
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
from transformers import BertTokenizer
from eval_save import BinBertModel
from data import diemph_gen_funcstr_wraper
import copy
import readidadata

torch.manual_seed(456)
np.random.seed(456)
random.seed(456)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        default='../jtrans-ori/models/finetune-ori/finetune_epoch_10', help='checkpoint path')
    parser.add_argument('--sample-in', type=str,
                        # default='')
                        default='data-static-analysis/jtrans-sample-20.pkl')
                        # default='data-static-analysis/jtrans-sample-20.pkl')
    parser.add_argument('--fout', type=str, default='data-static-analysis-ret/jtrans-static-results')
    parser.add_argument('--post', action='store_true', default=False)
    args = parser.parse_args()
    return args

def encode(model, tokenizer, cfg):
  funcstr = diemph_gen_funcstr_wraper(cfg, ori=True, rewrite_strategy='')
  tokenizer_ret = tokenizer(funcstr, add_special_tokens=True,max_length=512,padding='max_length',truncation=True,return_tensors='pt')
  input_ids = tokenizer_ret['input_ids'].cuda()
  attention_mask = tokenizer_ret['attention_mask'].cuda()
  output = model(input_ids, attention_mask=attention_mask)
  dbg_result = output.pooler_output
  dbg_tensor = dbg_result.detach()
  dbg_tensor = dbg_tensor/dbg_tensor.norm()
  dbg_tensor = dbg_tensor.cuda()
  return dbg_tensor

def analyze_one_cfg(model, tokenizer, cfg):
  ori_emb = encode(model, tokenizer, cfg)
  importance_analysis = prog_model.InstructionImportanceAnalysis(cfg)
  instr_list = importance_analysis.get_instr_list()
  cnt = 0
  mutated_tensors = []

  for n in sorted(cfg.nodes):
    for i in range(len(cfg.nodes[n]['asm'])):
      cnt += 1
      new_cfg = copy.deepcopy(cfg)
      new_cfg.nodes[n]['asm'][i] = ";"+new_cfg.nodes[n]['asm'][i]
      mutated_tensors.append(encode(model, tokenizer, new_cfg).squeeze(0))
      if cnt > 200:        
        break
    if cnt > 200:
      break
  
  mutate_emb_tensors = torch.stack(mutated_tensors)
  self_cos_similarity = torch.cosine_similarity(
      ori_emb, mutate_emb_tensors, dim=1)
  self_cos_similarity_array = self_cos_similarity.cpu().numpy()
  mean = self_cos_similarity_array.mean()
  std = self_cos_similarity_array.std()
  results_current_func = []

  for i in range(len(mutated_tensors)):
    marker = '*' if self_cos_similarity_array[i] < mean - std else ' '
    results_current_func.append(
          (i, marker, instr_list[i].importance,
           instr_list[i].code, self_cos_similarity_array[i]))

  return results_current_func

def main():
  args = parse_args()
  print(args)
  train_data = pickle.load(open(args.sample_in, 'rb'))
  print("Loaded %d samples"%len(train_data))
  # load model
  model = BinBertModel.from_pretrained(args.model)
  model.eval()
  model.cuda()
  tokenizer = BertTokenizer.from_pretrained('./jtrans_tokenizer/')
  # analyze
  results_all = []
  shuffled_train_data = random.sample(train_data, len(train_data))
  opt_list = ['O0', 'O3']
  if 'binkit' in args.sample_in:
    opt_list = ['clang4O0', 'clang4O1', 'clang4O3', 'gcc494O3']
  elif 'how' in args.sample_in:
    # opt_list = ['clang5O3', 'gcc7O3']
    opt_list = ['clang5O0', 'clang5O1', 'clang5O3', 'gcc7O3']
  for i, func in tqdm(enumerate(shuffled_train_data[:100])):
    for opt in opt_list:
      if opt not in func:
        continue
      if len(func[opt]['cfg']) < 3:
        continue
      cfg = func[opt]['cfg']
      results_current_func = analyze_one_cfg(model, tokenizer, cfg)
      results_all.append(results_current_func)
  print(opt_list)
  #  save results
  pickle.dump(results_all, open(args.fout+".raw.pkl", 'wb'))
  print("Saved %d results to %s"%(len(results_all), args.fout+".raw.pkl"))

  if not args.post:
    exit(0)

  # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
  #
  # post analysis
  #
  # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
  def normalize_instr(instr):
    ops = readidadata.parse_asm(instr, ori=True, rewrite_strategy='')
    ops_str = ['' if op is None else op for op in ops[:4]]
    return ' '.join(ops_str)
  print("Post analysis")
  
  if args.post:
    analsis_results = results_all
  else:
    exit(0)
  print("Post Analysis %d results"%len(results_all))
  # try kde
  sensitive_instrs = []
  non_sensitive_instrs = []
  from scipy import stats
  from scipy import integrate
  for variance in tqdm(results_all):
    all_variances = [i[4] for i in variance]
    kde = stats.gaussian_kde(all_variances)
    for entry in variance:
      new_entry = (entry[0], entry[1], entry[2], normalize_instr(entry[3]), entry[4])
      if integrate.quad(kde, a=-np.inf, b=entry[4])[0] < 0.1:
        sensitive_instrs.append(new_entry)
      else:
        non_sensitive_instrs.append(new_entry)


  # sensitive_instrs = []
  # non_sensitive_instrs = []
  # for variances in tqdm(results_all):
  #   for entry in variances:
  #     new_entry = (entry[0], entry[1], entry[2], normalize_instr(entry[3]), entry[4])
  #     if entry[1] == '*':
  #       sensitive_instrs.append(new_entry)
  #     else:
  #       non_sensitive_instrs.append(new_entry)
  
  instr_to_importance = {}
  for entry in sensitive_instrs:
    if entry[3] not in instr_to_importance:
      instr_to_importance[entry[3]] = []
    instr_to_importance[entry[3]].append(entry[2])
  # for entry in non_sensitive_instrs:
  #   if entry[3] not in instr_to_importance:
  #     instr_to_importance[entry[3]] = []
  #   instr_to_importance[entry[3]].append(entry[2])
  

  problematic = [i for i in sensitive_instrs if i[2] < 0.2]
  counter = {}
  for entry in problematic:
    if entry[3] not in counter:
      counter[entry[3]] = 0
    counter[entry[3]] += 1
  

  sorted_problematic = sorted(counter.items(), key=lambda x: x[1], reverse=True)

  sensitive_not_problematic = [i for i in sensitive_instrs if i[2] >= 0.2]
  counter = {}
  for entry in sensitive_not_problematic:
    if entry[3] not in counter:
      counter[entry[3]] = 0
    counter[entry[3]] += 1

  sorted_sensitive_not_problematic = sorted(counter.items(), key=lambda x: x[1], reverse=True)
  sorted_sensitive_not_problematic = [i for i in sorted_sensitive_not_problematic if i[1] > 1]
  
  only_in_problematic = []
  sensitive_not_problematic_set = set([i[0] for i in sorted_sensitive_not_problematic])
  for entry in sorted_problematic:
    if entry[0] not in sensitive_not_problematic_set:
      only_in_problematic.append(entry)

  # print top 10
  print("Problematic instructions")
  for i in only_in_problematic[:10]:
    print("%s: %d"%(i[0], i[1]))
  

if __name__ == '__main__':
  main()
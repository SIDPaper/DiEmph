import random
import torch
import argparse
from fairseq.models.trex import TrexModel
import os
import pickle
from diemph_utils import data_loader
from diemph_utils import asm_parser
from diemph_utils import prog_model, prog_model_stack, prog_model_no_global
from diemph_utils.prog_model import ReachDefinitionAnalysis
import diemph_eval
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
import copy

torch.manual_seed(456)
np.random.seed(456)
random.seed(456)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        default='', help='checkpoint path')
    parser.add_argument('--data', type=str,
                      default='', help='data path')    
    parser.add_argument('--sample-in', type=str,
                        default='')
    parser.add_argument('--fout', type=str, default='')
    parser.add_argument('--post', type=bool, default=False)
    args = parser.parse_args()
    return args


def analyze_one_cfg(model, current_func):  
  importance_analysis = prog_model.InstructionImportanceAnalysis(current_func)
  current_instr_list = importance_analysis.get_instr_list()[:120]
  asm_current_instr_list = [instr.code for instr in current_instr_list]
  trex_instr_list = diemph_eval.build_trex_instr_list(asm_current_instr_list, ori=True, rewrite_strategy='')
  ori_emb = diemph_eval.encode_one(model, trex_instr_list)
  mutate_embs = []
  for i in range(0, len(trex_instr_list)):
      mutate_instr_list = trex_instr_list.copy()
      mutate_instr_list = mutate_instr_list[:i] + mutate_instr_list[i+1:]
      mutate_embs.append(diemph_eval.encode_one(
          model, mutate_instr_list).squeeze(0))
  mutate_emb_tensors = torch.stack(mutate_embs)
  self_cos_similarity = torch.cosine_similarity(
      ori_emb, mutate_emb_tensors, dim=1)
  self_cos_similarity_array = self_cos_similarity.cpu().numpy()
  mean = self_cos_similarity_array.mean()
  std = self_cos_similarity_array.std()
  results_current_func = []
  for i, current_instr in enumerate(trex_instr_list):
      marker = '*' if self_cos_similarity_array[i] < mean - std else ' '
      results_current_func.append(
          (i, marker, current_instr_list[i].importance,
           current_instr, self_cos_similarity_array[i]))
  return results_current_func


def main():
  args = parse_args()
  print(args)
  train_data = pickle.load(open(args.sample_in, 'rb'))
  print("Loaded %d samples"%len(train_data))
  # load model
  model = TrexModel.from_pretrained(
            os.path.dirname(args.model), os.path.basename(args.model),
            data_name_or_path=args.data)
  model.cuda()
  model.eval()

  results_all = []
  shuffled_train_data = random.sample(train_data, len(train_data))
  opt_list = ['O0', 'O3']
  if 'binkit' in args.sample_in:
    opt_list = ['clang4O0', 'clang4O1', 'clang4O3', 'gcc494O3']
  elif 'how' in args.sample_in:
    opt_list = ['clang5O0', 'clang5O1', 'clang5O3', 'gcc7O3']
  for i, func in tqdm(enumerate(shuffled_train_data[:100])):
    for opt in opt_list:
      if opt not in func:
        continue
      if len(diemph_eval.get_instr_list(func[opt], ori=True, rewrite_strategy='')) < 20:
        continue
      cfg = func[opt]['cfg']
      results_current_func = analyze_one_cfg(model, cfg)
      results_all.append(results_current_func)

  #  save results
  pickle.dump(results_all, open(args.fout+".raw.pkl", 'wb'))
  print("Saved %d results to %s"%(len(results_all), args.fout+".raw.pkl"))
  
  if not args.post:
    exit(0)

  print("We are doing post analysis with %d results"%len(results_all))

  # kde
  sensitive_instrs = []
  non_sensitive_instrs = []
  from scipy import stats
  from scipy import integrate
  for variance in tqdm(results_all):
    all_variances = [i[4] for i in variance]
    kde = stats.gaussian_kde(all_variances)
    for entry in variance:
      if integrate.quad(kde, a=-np.inf, b=entry[4])[0] < 0.1:
        sensitive_instrs.append(entry)
      else:
        non_sensitive_instrs.append(entry)

      
  instr_to_importance = {}
  for entry in sensitive_instrs:
    if entry[3] not in instr_to_importance:
      instr_to_importance[entry[3]] = []
    instr_to_importance[entry[3]].append(entry[2])
  

  problematic = [i for i in sensitive_instrs if i[2] < 0.2]
  counter = {}
  for entry in problematic:
    if entry[3] not in counter:
      counter[entry[3]] = 0
    counter[entry[3]] += 1
  

  sorted_problematic = sorted(counter.items(), key=lambda x: x[1], reverse=True)

  sensitive_not_problematic = [i for i in sensitive_instrs if i[2] >= 0.5]
  
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
  

  print()


# main
if __name__ == '__main__':
    main()



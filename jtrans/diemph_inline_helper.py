import copy
import sys
from typing import Dict, Tuple
from datautils.playdata import DatasetBase as DatasetBase
import networkx
import os
import networkx as nx
from collections import defaultdict
from tqdm import tqdm
import pickle
import argparse
import re
from readidadata import ADDR_IDX,ASM_IDX,RAW_IDX,CFG_IDX
import readidadata
import torch
import random
import time
from diemph_process_one_binary import extract_call_targets,extract_call_targets_one_block

callee_saved_regs = set(
  [
    'rbp', 'rbx', 'r12', 'r13', 'r14', 'r15', 'rsp'    
  ]
)

param_regs = set(
  [
    'rdi', 'rsi', 'rdx', 'rcx', 'r8', 'r9',
    'edi', 'esi', 'edx', 'ecx', 'r8d', 'r9d',
    'di', 'si', 'dx', 'cx', 'r8w', 'r9w',
    'dil', 'sil', 'dl', 'cl', 'r8b', 'r9b'    
  ]
)

class InlineConfigure:
  def __init__(self):
    self.callee_size_factor = 3
    self.caller_size_factor = 3
    self.callee_in_degree_factor = 5
    self.level_factor = 20
    self.overall_budget = 100

  # to_string, print every field
  def to_string(self):
    out_str = ''
    for key, value in self.__dict__.items():
      out_str += '%s: %s' % (key, value)
    return out_str



class InlineHelper:

  def __init__(self, addr2function:Dict[int, Tuple]):
    self.addr2function = addr2function
    self.call_graph = self._build_call_graph()
    # TODO: inline configuration
    self.inline_config = InlineConfigure()

  def _should_inline(self, caller:int, callee:int, level:int):
    # TODO: TAIL CALL    
    if callee not in self.addr2function:
      return False
    cost = 0
    callee_size = len(self.addr2function[callee][CFG_IDX].nodes)
    caller_size = len(self.addr2function[caller][CFG_IDX].nodes)
    callee_in_degree = self.call_graph.in_degree[callee]
    
    cost += caller_size * self.inline_config.callee_size_factor
    cost += callee_size * self.inline_config.caller_size_factor
    cost += callee_in_degree * self.inline_config.callee_in_degree_factor
    if callee_in_degree == 1:
      cost -= 50
    cost += level * self.inline_config.level_factor
    return cost < self.inline_config.overall_budget



  def _perform_inline(self, func_addr:int, level:int):
    func = self.addr2function[func_addr]
    func_cfg = func[CFG_IDX]
    cfgs = [func_cfg]
    logical_order = []
    # nodes id sorted
    node_ids = [n for n in func_cfg.nodes]
    node_ids.sort()    
    for nid in node_ids:
      current_block = func_cfg.nodes[nid]      
      logical_order.append((nid, current_block))
      targets_for_current_block = extract_call_targets_one_block(current_block)
      for target in targets_for_current_block:
        if self._should_inline(func_addr, target, level):
          inlined_cfgs, inlined_logical_order = self._perform_inline(target, level+1)
          cfgs.extend(inlined_cfgs)
          logical_order.extend(inlined_logical_order)
    return cfgs, logical_order
 
  def _remove_prolog(self, node):
    code = node['asm']    
    for i in range(0, len(code)):
      opcode, op1, op2, _, _ = readidadata.parse_asm(code[i], ori=False)
      if opcode == 'push' and op1 == 'rbp':
        code[i] = ';'+ code[i]
      elif opcode == 'push' and op1 in callee_saved_regs:
        code[i] = ';'+ code[i]
      elif opcode == 'mov' and op1 == 'rbp' and op2 == 'rsp':
        code[i] = ';'+ code[i]
      elif opcode == 'sub' and op1 == 'rsp':
        code[i] = ';'+ code[i]
      elif opcode == 'mov' and op2 in param_regs:
        code[i] = ';'+ code[i]
      elif 'endbr' in opcode:
        code[i] = ';'+ code[i]
      elif opcode == 'SKIP':
        continue      
      else:
        break

  def _remove_epilog(self, node):
    code = node['asm']
    for i in range(len(code)-1, -1, -1):
      opcode, op1, op2, _, _ = readidadata.parse_asm(code[i], ori=False)
      if opcode == 'mov' and op1 == 'rsp' and op2 == 'rbp':
        code[i] = ';'+ code[i]
      elif opcode == 'pop' and op1 == 'rbp':
        code[i] = ';'+ code[i]
      elif opcode == 'pop' and op1 in callee_saved_regs:
        code[i] = ';'+ code[i]
      elif opcode == 'add' and op1 == 'rsp':
        code[i] = ';'+ code[i]
      elif 'ret' in code[i]:
        code[i] = ';'+ code[i]
      else:
        break    

  def inline_func(self, func_addr:int):
    func = self.addr2function[func_addr]
    func_cfg = func[CFG_IDX]
    cfgs, logical_order = self._perform_inline(func_addr, 0)

    new_cfg = networkx.compose_all(cfgs)
    # make a deep copy
    new_cfg = copy.deepcopy(new_cfg)
    for i, cfg in enumerate(cfgs):
      if i > 0:
        nodes_sorted_by_indeg = sorted(cfg.in_degree, key=lambda x:x[1])
        self._remove_prolog(new_cfg.nodes[nodes_sorted_by_indeg[0][0]])
        nodes_sorted_by_outdeg = sorted(cfg.out_degree, key=lambda x:x[1])
        for nid, outdeg in nodes_sorted_by_outdeg:
          if outdeg == 0:
            self._remove_epilog(new_cfg.nodes[nid])

      for n in cfg.nodes:
        if n == func_addr:
          new_cfg.nodes[n]['num'] = -1
        else:
          new_cfg.nodes[n]['num'] = i
        # save some space
        new_cfg.nodes[n]['raw'] = []
    return new_cfg, logical_order


  


  def _build_call_graph(self):
    call_graph = nx.DiGraph()
    for addr, func in self.addr2function.items():
      call_graph.add_node(addr)
      for target in extract_call_targets(func[CFG_IDX]):
        call_graph.add_edge(addr, target)
    return call_graph
    




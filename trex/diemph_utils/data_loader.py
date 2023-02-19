
import pickle

def load_sample(path):
    with open(path, 'rb') as f:
        sample = pickle.load(f)
    return sample['query_func'], sample['candidates_func']


def get_instr_list(func, ori=True):
    instr_list = []
    cfg = func['cfg']
    nodes = [n for n in cfg.nodes]
    nodes.sort()
    if not ori:
        exceptional_nodes = []
        for nid in nodes:
            if cfg.out_degree(nid) > 0:
                continue
            instr = ' '.join(cfg.nodes[nid]['asm'])
            if 'assert' in instr and 'fail' in instr:
                exceptional_nodes.append(nid)
            elif 'exit' in instr or 'abort' in instr:
                exceptional_nodes.append(nid)                            
        normal_nodes = [n for n in nodes if n not in exceptional_nodes]
        nodes = normal_nodes + exceptional_nodes

    for nid in nodes:
        current_node = cfg.nodes[nid]
        instr_list.extend(current_node['asm'])
    return instr_list
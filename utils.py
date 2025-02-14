import pickle
import torch
import numpy as np
from fvcore.common.config import CfgNode
from ete3 import Tree

def empty_config():
    cfgs = CfgNode()
    cfgs.num_epoch = 1
    cfgs.num_episodes = 1
    cfgs.num_episodes_baseline = 1
    cfgs.lr = 0.01
    cfgs.clip_value = 0.1
    cfgs.entropy_reg_strength = 1.0
    cfgs.risk_epsilon = 0.1

    cfgs.replay_buffer_size = 128
    cfgs.replay_buffer_sample_size = 32
    cfgs.replay_buffer_score_bound = 10

    cfgs.loss = CfgNode()
    cfgs.loss.BALANCED_ELU_LOSS = False  # 默认值
    cfgs.loss.ELU_LOSS = False  # 默认值

    cfgs.summary_name = "Try"
    cfgs.summary_path = "tb_summary"
    cfgs.checkpoint_path = "checkpoints"
    cfgs.reload_checkpoint_path = ""

    cfgs.dataset_path = ""
    cfgs.val_dataset_path = ""
    cfgs.instance_path = ""
    cfgs.sequences_file = ""
    cfgs.raw_tree_file = ""
    cfgs.c_best_tree_file = ""
    
    cfgs.dataset_taxa_list = []
    cfgs.dataset_len_list = []

    cfgs.env = CfgNode()
    cfgs.env.batch_size = 8
    cfgs.env.sequence_type = "DNA_WITH_GAP"


    cfgs.model = CfgNode()
    cfgs.model.vocab_size = 4
    cfgs.model.patch_size = 4
    cfgs.model.fixed_length = 1024
    cfgs.model.embed_dim = 32
    cfgs.model.encoder_attn_layers = 2
    cfgs.model.num_enc_heads = 4
    cfgs.model.num_enc_layers = 3


    cfgs.ratio_factor = 1.0

    return cfgs

evolution_model = "GTR+I+G"

def set_evolution_model(model_name):
    global evolution_model
    evolution_model = model_name

import random

class ReplayBuffer:
    def __init__(self, replay_buffer_size):
        self.replay_buffer_size = replay_buffer_size
        self.trees = []
        self.scores = []

    def add(self, trees, scores):
        for tree, score in zip(trees, scores):

            if any(existing_tree.topo_repr == tree.topo_repr for existing_tree in self.trees):
                continue

            if len(self.trees) < self.replay_buffer_size:
                self.trees.append(tree)
                self.scores.append(score)
            else:
                min_score_index = self.scores.index(min(self.scores))
                if score > self.scores[min_score_index]:
                    self.trees[min_score_index] = tree
                    self.scores[min_score_index] = score

    def sample(self, sample_num):
        if sample_num > len(self.trees):
            sample_num = len(self.trees)
        sampled_indices = random.sample(range(len(self.trees)), sample_num)
        sampled_trees = [self.trees[i] for i in sampled_indices]

        sampled_actions_list = [tree.sample_trajectory() for tree in sampled_trees]

        # if sampled_actions_list:
        #     import pdb; pdb.set_trace()

        return None
        # return sampled_actions_list

    def get_size(self):
        return len(self.trees)

    def check_duplicate(self, hash_function):
        return len(set([hash_function(tree) for tree in self.trees]))



class ReplayBufferDyn:
    def __init__(self, score_bound):
        self.score_bound = score_bound
        self.trees = []
        self.scores = []
        self.max_score = -np.inf

    def add(self, trees, scores):
        for tree, score in zip(trees, scores):

            if any(existing_tree.topo_repr == tree.topo_repr for existing_tree in self.trees):
                continue

            if self.max_score - self.score_bound < score:
                self.trees.append(tree)
                self.scores.append(score)
                self.max_score = max(self.max_score, score)

        drop_indices = []
        for i in range(len(self.trees)):
            if self.max_score - self.score_bound >= self.scores[i]:
                drop_indices.append(i)

        self.trees = [tree for idx, tree in enumerate(self.trees) if idx not in drop_indices]
        self.scores = [score for idx, score in enumerate(self.scores) if idx not in drop_indices]


    def sample(self, sample_num):
        # 随机选择 sample_num 个树
        if sample_num > len(self.trees):
            sample_num = len(self.trees)
        sampled_indices = random.sample(range(len(self.trees)), sample_num)
        sampled_trees = [self.trees[i] for i in sampled_indices]

        sampled_actions_list = [tree.sample_trajectory() for tree in sampled_trees]

        # if sampled_actions_list:
        #     import pdb; pdb.set_trace()

        return sampled_actions_list

    def get_size(self):
        return len(self.trees)

    def check_duplicate(self, hash_function):
        return len(set([hash_function(tree) for tree in self.trees]))


def read_fasta(filepath):
    all_seqs_dict = {}
    with open(filepath, 'r') as file:
        seq_id = None
        all_seqs = []
        for line in file:
            line = line.rstrip()
            if line.startswith('>'):
                if len(all_seqs) > 0 and seq_id is not None:
                    all_seqs_dict[seq_id] = all_seqs
                seq_id = line.lstrip('>')
                # all_seqs = []
                all_seqs = ""
            elif len(line) > 0:
                all_seqs = all_seqs + line

        if len(all_seqs) > 0 and seq_id is not None:
            all_seqs_dict[seq_id] = all_seqs

    return all_seqs_dict


def load_sequences(sequences_path):
    # load sequences
    if sequences_path.endswith('.fasta'):
        key_to_seqs_dict = read_fasta(sequences_path)
        # for now, only selecting the first set of sequences
        all_seq_keys = list(key_to_seqs_dict.keys())
        all_seqs = list(key_to_seqs_dict.values())

        all_seqs = [seq.upper().replace('?', 'N').replace('.', 'N') for seq in all_seqs]

        return all_seq_keys, all_seqs
    
    else:
        assert False
        return None, None



def pad_array_mask(L):

    L = [np.array(actions, dtype=np.int64) for actions in L]
    max_size = max(array.size for array in L)
    padded_arrays = [np.pad(array, (0, max_size - array.size), 'constant') for array in L]
    padded_arrays_mask = [np.pad(np.ones_like(array, np.bool_), (0, max_size - array.size), 'constant') for array in L]
    L = np.stack(padded_arrays)
    L_mask = np.stack(padded_arrays_mask)

    L, L_mask = torch.from_numpy(L), torch.from_numpy(L_mask)

    return L, L_mask



def get_score_indices_to_prev(actions_ij_prev, env, nb_seq, batch_size):

    score_indices_to_prev = []

    for batch_id in range(batch_size):
        a_prev = actions_ij_prev[batch_id]
        ii_prev, jj_prev = a_prev[0].item(), a_prev[1].item()

        indices = []

        len_tree_pairs = len(env.tree_pairs_dict[nb_seq+1])
        for idx, pair in enumerate(env.tree_pairs_dict[nb_seq]):
            ii, jj = pair

            if ii < ii_prev:
                if jj < ii_prev:
                    indices.append( env.action_indices_dict[nb_seq+1][pair] )
                elif jj == ii_prev:
                    indices.append( len_tree_pairs+ii )
                elif jj < jj_prev:
                    indices.append( env.action_indices_dict[nb_seq+1][pair] )
                else:
                    indices.append( env.action_indices_dict[nb_seq+1][(ii, jj+1)] )
            elif ii == ii_prev:
                if jj < jj_prev:
                    indices.append( len_tree_pairs+jj )
                else:
                    indices.append( len_tree_pairs+jj )
            elif ii < jj_prev:
                if jj < jj_prev:
                    indices.append( env.action_indices_dict[nb_seq+1][pair] )
                else:
                    indices.append( env.action_indices_dict[nb_seq+1][(ii, jj+1)] )
            else:
                indices.append( env.action_indices_dict[nb_seq+1][(ii+1, jj+1)] )

        score_indices_to_prev.append(indices)

    return score_indices_to_prev



def calculate_rf_distance(file1, file2):
    t1=Tree(file1)
    # print(os.path.join(preds, tree.split('.tre')[0]+'.pf.nwk'))
    t2=Tree(file2)
    norm_rf_dist = t1.compare(t2,unrooted=True)['norm_rf']
    rf = t1.compare(t2,unrooted=True)['rf']
    return rf, norm_rf_dist

def get_tree_str_from_newick(newick_file):
    with open(newick_file, "r") as file:
        tree_str = file.readline().strip()
    return tree_str
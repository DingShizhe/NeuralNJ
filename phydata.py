import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import Sampler
import itertools
import numpy as np
from Bio import Phylo
from Bio.Phylo.BaseTree import Clade
from environment import UnrootedPhyloTree,PhyloTree
import time
import io
import json
import random
import copy
import re
import raxmlpy
from utils import evolution_model

# evolution_model = 'GTR'
# evolution_model = 'GTR+I+G'
# evolution_model = 'JC'

CHARACTERS_MAPS = {
    'DNA': {
        'A': [1., 0., 0., 0.],
        'C': [0., 1., 0., 0.],
        'G': [0., 0., 1., 0.],
        'T': [0., 0., 0., 1.],
        'N': [1., 1., 1., 1.]
    },
    'RNA': {
        'A': [1., 0., 0., 0.],
        'C': [0., 1., 0., 0.],
        'G': [0., 0., 1., 0.],
        'U': [0., 0., 0., 1.],
        'N': [1., 1., 1., 1.]
    },
    'DNA_WITH_GAP': {
        'A': [1., 0., 0., 0.],
        'C': [0., 1., 0., 0.],
        'G': [0., 0., 1., 0.],
        'T': [0., 0., 0., 1.],
        '-': [1., 1., 1., 1.],
        'N': [1., 1., 1., 1.],
        '*': [0., 0., 0., 0.]
    },
    'RNA_WITH_GAP': {
        'A': [1., 0., 0., 0.],
        'C': [0., 1., 0., 0.],
        'G': [0., 0., 1., 0.],
        'U': [0., 0., 0., 1.],
        '-': [1., 1., 1., 1.],
        'N': [1., 1., 1., 1.]
    }
}

CHARS_DICT = CHARACTERS_MAPS['DNA_WITH_GAP']


# CHARS_DICT = {
#     'A': 1,
#     'C': 2,
#     'G': 3,
#     'T': 4,
#     'N': 0,
#     '-': 0,
#     '*': -1,
# }

char_to_index = {char: idx for idx, char in enumerate(CHARS_DICT.keys())}
HARS_DICT = list(CHARS_DICT.keys())
lookup_table = np.array(list(CHARS_DICT.values()), dtype=np.int8)

def _seq2array(seq):
    seq_indices = np.array([char_to_index[char] for char in seq], dtype=np.int8)

    return lookup_table[seq_indices]


def optimize_branch_length(phy_path, tre_path, iters=3):
    if phy_path.endswith('.phy'):
        seqs, seq_keys, num_sequences, sequence_length = load_phy_file_multirow(phy_path)
    elif phy_path.endswith('.aln') or phy_path.endswith('.fasta'):
        seqs, seq_keys, num_sequences, sequence_length = load_alignment_file(phy_path)

    with open(tre_path, "r") as file:
        tree_str = file.readline().strip()

    msa = {
        "labels": seq_keys,
        "sequences": seqs
    }

    tree_op, logllint, logllop = raxmlpy.optimize_brlen(tree_str, msa, is_root=False, iters=iters, model=evolution_model, opt_model=True)
    return logllop, tree_str


def only_padding_sample(original_seqs, target_length=-1):
    transposed_seqs = list(zip(*original_seqs))

    unique_columns_list = list(''.join(column) for column in transposed_seqs)
    weights = [1] * len(unique_columns_list)

    length = len(unique_columns_list)

    target_length = max(len(l) for l in original_seqs)
    # if target_length % 1 == 1:
    #     target_length += 1

    if length <= target_length:
        # Padding
        padding_length = target_length - length
        padded_columns = ['*' * len(original_seqs)] * padding_length
        padded_weights = [0] * padding_length

        msa = unique_columns_list + padded_columns
        weights += padded_weights
    elif length > target_length:
        # Sampling
        sampled_columns = random.choices(unique_columns_list, weights=weights, k=target_length)
        msa = sampled_columns
        weights = [1] * len(sampled_columns)
    return length, msa, weights


def compress_pad_seq(original_seqs, target_length=1024):
    transposed_seqs = list(zip(*original_seqs))
    unique_columns = {}
    for column in transposed_seqs:
        column_str = ''.join(column)
        if column_str not in unique_columns:
            unique_columns[column_str] = 1
        else:
            unique_columns[column_str] += 1

    unique_columns_list = list(unique_columns.keys())
    weights = list(unique_columns.values())
    compressed_length = len(unique_columns_list)

    if compressed_length <= target_length:
        # Padding
        padding_length = target_length - compressed_length
        padded_columns = ['*' * len(original_seqs)] * padding_length
        padded_weights = [0] * padding_length

        compressed_msa = unique_columns_list + padded_columns
        weights += padded_weights
    elif compressed_length > target_length:
        # Sampling
        sampled_columns = random.choices(unique_columns_list, weights=weights, k=target_length)
        compressed_msa = sampled_columns
        weights = [unique_columns[col] for col in sampled_columns]

    compressed_msa = [''.join(seq) for seq in zip(*compressed_msa)]

    return compressed_length, compressed_msa, weights


def hamming_distance(str1, str2):
    if len(str1) != len(str2):
        raise ValueError("str1 and str2 must have the same length")  
    
    return sum(ch1 != ch2 for ch1, ch2 in zip(str1, str2))

def pairwise_hamming_distance(strings):
    distances = []
    n = len(strings)

    for i in range(n):
        ___d = []
        for j in range(n):
            distance = hamming_distance(strings[i], strings[j])
            ___d.append(distance)
        distances.append(___d)

    return distances


class PhyDataset(Dataset):
    def __init__(self, root_dir, trajectory_num, device, pos=5, compress=False, only_pad_sample=True, num_per_file=None, len_list=None, taxa_list=None, C_solved=False):
        self.data = []
        self.len_to_taxa = {}  # Mapping from len to list of taxa
        self.trajectory_num = trajectory_num
        self.device = device
        self.compress = compress
        self.only_pad_sample = only_pad_sample
        self.pos = pos

        for len_folder in os.listdir(root_dir):
            len_path = os.path.join(root_dir, len_folder)
            if os.path.isdir(len_path):
                len_value = int(len_folder.replace("len", ""))
                if len_list is not None and len(len_list) > 0:
                    if len_value not in len_list:
                        continue
                self.len_to_taxa[len_value] = []

                for taxa_folder in os.listdir(len_path):
                    taxa_path = os.path.join(len_path, taxa_folder)
                    if os.path.isdir(taxa_path):
                        taxa_value = int(taxa_folder.replace("taxa", ""))
                        # if taxa_value != 40 and taxa_value!=50:
                        if taxa_list is not None and len(taxa_list) > 0:
                            if taxa_value not in taxa_list:
                                continue

                        self.len_to_taxa[len_value].append(taxa_value)
                        num_files = 0
                        for file in os.listdir(taxa_path):
                            if num_per_file is not None and num_files >= num_per_file:
                                    break
                            if file.endswith('.phy'):
                                file_path = os.path.join(taxa_path, file)
                                tre_path = file_path.replace('.phy', '.tre')
                                if not os.path.exists(tre_path):
                                    break

                                if C_solved:
                                    raw_tre_path = file_path.replace('.phy', '.tre')
                                else:
                                    raw_tre_path = ""

                                self.data.append((
                                    len_value,
                                    taxa_value,
                                    file_path,
                                    tre_path,
                                    raw_tre_path
                                ))
                                num_files += 1


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        len_value, taxa_value, file_path, tre_path, raw_tre_path = self.data[idx]
        seqs, seq_keys, num_sequences, sequence_length = load_phy_file_multirow(file_path)

        key_to_index = {key: int(key[self.pos:])-1 for i, key in enumerate(seq_keys)}
        idx_min = min(list(key_to_index.values()))
        key_to_index = {key:idx-idx_min for key,idx in key_to_index.items()}
        sorted_seqs = [None] * len(seqs)  
        sorted_keys = [None] * len(seq_keys)
        for seq, key in zip(seqs, seq_keys):
            sorted_seqs[key_to_index[key]] = seq
            sorted_keys[key_to_index[key]] = key

        if self.compress:
            compressed_length, compressed_seqs, weights = compress_pad_seq(sorted_seqs)
            seqs = compressed_seqs
            seqs = [''.join(s) for s in zip(*seqs)]
            weights = np.array(weights, dtype=np.float32)
        elif self.only_pad_sample:
            length, padded_seqs, weights = only_padding_sample(sorted_seqs)
            seqs = padded_seqs
            seqs = [''.join(s) for s in zip(*seqs)]
            weights = np.array(weights, dtype=np.float32)
        else:
            seqs = sorted_seqs
            weights = np.ones(len(seqs[0]), dtype=np.float32)

        distances = pairwise_hamming_distance(seqs)

        distances = np.array(distances, dtype=np.int32)

        # get data
        data = np.stack([_seq2array(seq) for seq in seqs])
        end_time2 = time.time()

        tree = load_tree_file(tre_path, self.device, pos=self.pos)
        if raw_tre_path != "":
            raw_tree = read_tree_data(raw_tre_path)
        else:
            raw_tree = None

        end_time3 = time.time()

        action = []
        action_set = []
        for i in range(self.trajectory_num):
            action_per_traj, action_set_per_traj = sample_trajectory_set_bottom_top(tree,pos=self.pos)
            action.append(action_per_traj)
            action_set.append(action_set_per_traj)
        
        action_np = np.array(action, dtype=np.int32)

        return {
            'data': data,
            'seqs': sorted_seqs,
            'weights': weights,
            'seq_keys': sorted_keys,
            'tree': tree,
            'raw_tree': raw_tree,
            'action': action_np,
            'action_set': action_set,
            'file_path': file_path,
            'tre_path': tre_path,
            'distances': distances,
        }

class PhyDataset_real(Dataset):
    def __init__(self, root_dir, trajectory_num, device, pos=2, compress=False, only_pad_sample=True, num_per_file=None, len_list=None, taxa_list=None, C_solved=False, csv_file=None, load_Csolver=False):
        self.data = []
        self.len_to_taxa = {}  # Mapping from len to list of taxa
        self.trajectory_num = trajectory_num
        self.device = device
        self.compress = compress
        self.only_pad_sample = only_pad_sample
        self.pos = pos
        self.load_Csolver = load_Csolver

        # Traverse through the directory structure
        # for len_folder in list(os.listdir(root_dir))[:20]:
        num_files = 0
        for file in os.listdir(root_dir):
            if num_per_file is not None and num_files >= num_per_file:
                break
            if csv_file is not None:
                if file not in sampled_file_names:
                    continue
            if file.endswith('.phy'):
                file_path = os.path.join(root_dir, file)
                tre_path = file_path.replace('.phy', '.tre')
                if not os.path.exists(tre_path):
                    break
                self.data.append((
                    file_path,
                    tre_path,
                ))
                num_files += 1


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path, tre_path = self.data[idx]
        # print(file_path)
        start_time = time.time()
        # seqs, seq_keys, num_sequences, sequence_length = load_phy_file(file_path)
        seqs, seq_keys, num_sequences, sequence_length = load_phy_file_multirow(file_path)
        end_time1 = time.time()


        key_to_index = {key: int(key[self.pos:])-1 for i, key in enumerate(seq_keys)}
        idx_min = min(list(key_to_index.values()))
        key_to_index = {key:idx-idx_min for key,idx in key_to_index.items()}
        sorted_seqs = [None] * len(seqs)  
        sorted_keys = [None] * len(seq_keys)
        for seq, key in zip(seqs, seq_keys):
            sorted_seqs[key_to_index[key]] = seq
            sorted_keys[key_to_index[key]] = key

        # import pdb; pdb.set_trace()
        # compressed_seqs
        if self.compress:
            compressed_length, compressed_seqs, weights = compress_pad_seq(sorted_seqs)
            seqs = compressed_seqs
            seqs = [''.join(s) for s in zip(*seqs)]
            weights = np.array(weights, dtype=np.float32)
        elif self.only_pad_sample:
            length, padded_seqs, weights = only_padding_sample(sorted_seqs)
            seqs = padded_seqs
            seqs = [''.join(s) for s in zip(*seqs)]
            weights = np.array(weights, dtype=np.float32)
        else:
            seqs = sorted_seqs
            weights = np.ones(len(seqs[0]), dtype=np.float32)

        distances = pairwise_hamming_distance(seqs)

        # import pdb; pdb.set_trace()

        distances = np.array(distances, dtype=np.int32)

        # get data
        data = np.stack([_seq2array(seq) for seq in seqs])
        end_time2 = time.time()

        tree = load_tree_file(tre_path, self.device, pos=self.pos)
        raw_tree = None

        end_time3 = time.time()

        # import pdb; pdb.set_trace()
        action = []
        action_set = []
        # evo_dist_mat = []
        for i in range(self.trajectory_num):
            # try:  
                # action_per_traj, action_set_per_traj = sample_trajectory_set(tree)
            action_per_traj, action_set_per_traj = sample_trajectory_set_bottom_top(tree,pos=self.pos)
            action.append(action_per_traj)
            action_set.append(action_set_per_traj)
            # evo_dist_mat.append(evo_dist_mat_list_per_traj)
        
        action_np = np.array(action, dtype=np.int32)
        # evo_dist_mat_np = np.array(evo_dist_mat)

        if self.load_Csolver:
            C_logllop, tree_str = optimize_branch_length(file_path, tre_path)
            return {
                'data': data,
                'seqs': sorted_seqs,
                'weights': weights,
                'seq_keys': sorted_keys,
                'tree': tree,
                'raw_tree': raw_tree,
                'action': action_np,
                'action_set': action_set,
                'file_path': file_path,
                'tre_path': tre_path,
                'distances': distances,
                'C_logllop': C_logllop,
                'tree_str': tree_str,
                'taxa_num': num_sequences,
                'seq_len': sequence_length,
            }

        return {
            'data': data,
            'seqs': sorted_seqs,
            'weights': weights,
            'seq_keys': sorted_keys,
            'tree': tree,
            'raw_tree': raw_tree,
            'action': action_np,
            'action_set': action_set,
            # 'action_set_mask': action_set_mask,
            'file_path': file_path,
            'tre_path': tre_path,
            'distances': distances,
            'taxa_num': num_sequences,
            'seq_len': sequence_length,
            # 'evo_dist_mat': evo_dist_mat
            # 'evo_dist_mat': None
        }



class PhySampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.size = len(self.dataset)

    def __iter__(self):
        # Create a list of indices for each len and taxa
        indices_dict = {(len_val, taxa_val): [] for len_val in self.dataset.len_to_taxa.keys() for taxa_val in self.dataset.len_to_taxa[len_val]}
        for idx, val in enumerate(self.dataset.data):
            len_val = val[0]
            taxa_val = val[1]
            indices_dict[(len_val, taxa_val)].append(idx)

        # Shuffle each list of indices if shuffle is True
        if self.shuffle:
            for idx_list in indices_dict.values():
                np.random.shuffle(idx_list)

        # Generate batches
        batched_indices = []
        for _, idx_list in indices_dict.items():
            batched_indices.extend([idx_list[i:i + self.batch_size] for i in range(0, len(idx_list), self.batch_size)])

        # Shuffle the batches if shuffle is True
        if self.shuffle:
            np.random.shuffle(batched_indices)
        
        return iter(batched_indices)


    def __len__(self):
        return len(self.dataset)
    

def load_phy_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    num_sequences, sequence_length = map(int, lines[0].split())

    sequences = {}
    for line in lines[1:]:
        parts = line.split()
        taxa, sequence = parts[0], ''.join(parts[1:])
        sequences[taxa] = sequence.upper().replace('?', 'N').replace('.', 'N')
    
    assert num_sequences == len(sequences)
    assert sequence_length == len(next(iter(sequences.values())))

    # split dict sequences to list sequences and list seq_keys
    seq_keys = list(sequences.keys())
    all_seqs = list(sequences.values())
    return all_seqs, seq_keys, num_sequences, sequence_length


def load_phy_file_multirow(file_path):
    with open(file_path, 'r') as file:
        first_line = file.readline().strip()
        species_count, sites_count = map(int, first_line.split())
        
        sequences = {}
        
        identifiers = []
        
        for line in file:
            line = line.strip()
            if line:
                if len(line.split()) > 1:
                    identifier, sequence = line.split(maxsplit=1)
                    sequences[identifier] = sequence.replace(" ", "").upper()
                    identifiers.append(identifier)

            else:
                break
        
        idx = 0
        for line in file:
            line = line.strip()
            if line:
                sequence = line
                sequences[identifiers[idx]] += sequence.replace(" ", "").upper()
                idx += 1
            else:
                idx = 0

    for identifier in identifiers:
        assert len(sequences[identifier]) == sites_count, f"sequence {identifier} length is not equal to sites_count"
    
    # assert species_count
    assert len(identifiers) == species_count, "species_count is not equal to identifiers"

    # change other symbols "k,^ ,..." to "-"
    for identifier, seq in sequences.items():
        updated_seq = ''
        for char in seq:
            if char not in HARS_DICT:
                updated_seq += '-'
            else:
                updated_seq += char
        sequences[identifier] = updated_seq

    seq_keys = list(sequences.keys())
    all_seqs = list(sequences.values())

    return all_seqs, seq_keys, species_count, sites_count


def read_phy_file_multirow(file_path):
    with open(file_path, 'r') as file:
        first_line = file.readline().strip()
        species_count, sites_count = map(int, first_line.split())
        
        sequences = {}
        identifiers = []
        
        for line in file:
            line = line.strip()
            if line:
                if len(line.split()) > 1:
                    identifier, sequence = line.split(maxsplit=1)
                    sequences[identifier] = sequence.replace(" ", "")
                    identifiers.append(identifier)

            else:
                break

        idx = 0
        for line in file:
            line = line.strip()
            if line:
                sequence = line
                sequences[identifiers[idx]] += sequence.replace(" ", "")
                idx += 1
            else:
                idx = 0

    # 验证每个序列的长度
    for identifier in identifiers:
        assert len(sequences[identifier]) == sites_count, f"sequence {identifier} length is not equal to sites_count"
    
    # assert species_count
    assert len(identifiers) == species_count, "species_count is not equal to identifiers"

    seq_keys = list(sequences.keys())
    all_seqs = list(sequences.values())

    return all_seqs, seq_keys, species_count, sites_count


def load_alignment_file(file_path):
    sequences = {}
    identifiers = []
    
    with open(file_path, 'r') as file:
        identifier = None
        current_sequence = []
        
        for line in file:
            line = line.strip()
            if not line:
                continue 
            
            if line.startswith('>'):
                if identifier:
                    sequences[identifier] = ''.join(current_sequence).upper().replace(" ", "")
                identifier = line[1:].split()[0]  
                identifiers.append(identifier)
                current_sequence = []
            else:
                current_sequence.append(line)
        
        if identifier:
            sequences[identifier] = ''.join(current_sequence).upper().replace(" ", "")
    
    species_count = len(sequences)
    sites_count = len(next(iter(sequences.values()))) if species_count > 0 else 0
    
    for identifier, sequence in sequences.items():
        assert len(sequence) == sites_count, f"sequence {identifier} length is not equal to sites_count"
    
    for identifier, sequence in sequences.items():
        sequences[identifier] = ''.join([char if char in HARS_DICT else '-' for char in sequence])
    
    seq_keys = list(sequences.keys())
    all_seqs = list(sequences.values())

    return all_seqs, seq_keys, species_count, sites_count


def load_tree_file(file_path,device, pos=5):
    with open(file_path, 'r') as file:
        tree = Phylo.read(file, 'newick')
    
    # clades3to2
    if len(tree.root.clades) == 3:
        new_node = Clade(branch_length=0.0)
        new_node.clades.append(tree.root.clades.pop())
        new_node.clades.append(tree.root.clades.pop())
        tree.root.clades.append(new_node)
    #unrooted_phlytree=tree_to_phlytree(tree.root, device=device, is_root=True,all_seqs=all_seqs,seq_keys=seq_keys)
    # innernode name
    leaf_num = len(tree.root.get_terminals())
    inner_node_id = leaf_num + 1
    for node in tree.root.get_nonterminals():
        if node.name is None:
            node.name = "taxon"+str(inner_node_id)
            inner_node_id += 1
    sorted_tree = sorted_bio_tree(tree.root, pos=pos)

    return sorted_tree


def read_tree_data(raw_tre_path):
    with open(raw_tre_path, 'r') as file:
        raw_tree = file.read()
    return raw_tree


def sorted_bio_tree(node, pos=5):
    if node.is_terminal():
        node.id = node.name
        return node
    else:
        left = sorted_bio_tree(node.clades[0],pos=pos)
        right = sorted_bio_tree(node.clades[1],pos=pos)
        left_idx = get_min_seq_index(left, pos=pos)
        right_idx = get_min_seq_index(right, pos=pos)
        if left_idx > right_idx:
            node.clades = [right,left]
            node.id = node.name
            node.name = right.name
        else:
            node.id = node.name
            node.name = left.name
        return node


def sample_trajectory(node, pos):
    # node clades number == 2
    assert len(node.clades)==2
    current_subtrees = [node.clades[0], node.clades[1]]
    actions = [[0, 1]]
    # actions = []

    for idx in range(len(node.get_terminals())-2):
        unleaf_subtree_indices = [i for i, st in enumerate(current_subtrees) if not st.is_terminal()]
        selected_subtree_idx = random.choice(unleaf_subtree_indices)
        selected_subtree = current_subtrees[selected_subtree_idx]

        current_subtrees.pop(selected_subtree_idx)
        assert len(selected_subtree.clades)==2
        current_subtrees.insert(0, selected_subtree.clades[1])
        current_subtrees.insert(0, selected_subtree.clades[0])

        indexed_subtrees = list(enumerate(current_subtrees))
        indexed_subtrees.sort(key=lambda x: get_min_seq_index(x[1], pos=pos))

        indexed_subtrees_indices = [index for index, _ in indexed_subtrees]
        sorted_index_of_left = indexed_subtrees_indices.index(0)
        sorted_index_of_right = indexed_subtrees_indices.index(1)

        current_subtrees = [x[1] for x in indexed_subtrees]

        a0 = min(sorted_index_of_left, sorted_index_of_right)
        a1 = max(sorted_index_of_left, sorted_index_of_right)

        actions.append([a0, a1])

    actions.reverse()
    return actions


def sample_trajectory_set(node,pos=5):
    assert len(node.clades)==2
    current_subtrees = [node.clades[0], node.clades[1]]

    actions = [[0, 1]]
    parent_sets = {(get_min_seq_index(node.clades[0],pos=pos),get_min_seq_index(node.clades[1],pos=pos)):node}
    parents = {node.clades[0]:node,node.clades[1]:node}
    actions_set = [[[0,1]]]

    for idx in range(len(node.get_terminals())-2):
        unleaf_subtree_indices = [i for i, st in enumerate(current_subtrees) if not st.is_terminal()]
        selected_subtree_idx = random.choice(unleaf_subtree_indices)
        selected_subtree = current_subtrees[selected_subtree_idx]

        current_subtrees.pop(selected_subtree_idx)
        assert len(selected_subtree.clades)==2
        current_subtrees.insert(0, selected_subtree.clades[1])
        current_subtrees.insert(0, selected_subtree.clades[0])

        parent_tree = parents[selected_subtree]
        parent_key = (get_min_seq_index(parent_tree.clades[0],pos=pos),get_min_seq_index(parent_tree.clades[1],pos=pos))
        if parent_key in parent_sets:
            parent_sets.pop(parent_key)
        selected_subtree_key = (get_min_seq_index(selected_subtree.clades[0],pos=pos),get_min_seq_index(selected_subtree.clades[1],pos=pos))
        parent_sets[selected_subtree_key] = selected_subtree
        parents[selected_subtree.clades[0]] = selected_subtree
        parents[selected_subtree.clades[1]] = selected_subtree
        if selected_subtree in parents:
            parents.pop(selected_subtree)


        indexed_subtrees = list(enumerate(current_subtrees))
        indexed_subtrees.sort(key=lambda x: get_min_seq_index(x[1], pos=pos))

        indexed_subtrees_indices_key = {get_min_seq_index(tree, pos=pos):idx for idx, (_, tree) in enumerate(indexed_subtrees)}
        
        action_set_per_step = []
        for _, node in parent_sets.items():
            a0 = indexed_subtrees_indices_key[get_min_seq_index(node.clades[0],pos=pos)]
            a1 = indexed_subtrees_indices_key[get_min_seq_index(node.clades[1],pos=pos)]
            assert a0 < a1
            action_set_per_step.append([a0,a1])
        actions_set.append(action_set_per_step)
        

        indexed_subtrees_indices = [index for index, _ in indexed_subtrees]
        sorted_index_of_left = indexed_subtrees_indices.index(0)
        sorted_index_of_right = indexed_subtrees_indices.index(1)

        current_subtrees = [x[1] for x in indexed_subtrees]

        a0 = min(sorted_index_of_left, sorted_index_of_right)
        a1 = max(sorted_index_of_left, sorted_index_of_right)
        assert indexed_subtrees_indices_key[get_min_seq_index(selected_subtree.clades[0],pos=pos)] == a0
        assert indexed_subtrees_indices_key[get_min_seq_index(selected_subtree.clades[1],pos=pos)] == a1

        actions.append([a0, a1])
    
    actions.reverse()
    actions_set.reverse()
    return actions, actions_set


def sample_trajectory_set_bottom_top(node, pos=5):
    assert len(node.clades)==2
    current_subtrees = [node.clades[0], node.clades[1]]
    
    mom_map = dict()

    def build_mom_map(node, mom_map, parent=None):
        if parent is not None:
            mom_map[node] = parent

        if not node.is_terminal():
            for child in node.clades:
                build_mom_map(child, mom_map, node)

    build_mom_map(node, mom_map)

    all_subtrees = node.get_terminals()
    all_leaves_num = len(all_subtrees)

    action_sets = []
    actions = []

    for step_id in range(all_leaves_num-1):

        all_subtrees_num = len(all_subtrees)
        all_subtrees = sorted(all_subtrees, key=lambda x: int(x.name[pos:]))
        all_subtree_idx_map = {subtree: i for i, subtree in enumerate(all_subtrees)}

        all_subtrees_actions = []
        subtree_mom_count = dict()
        action_set = []
        for subtree_id, subtree in enumerate(all_subtrees):
            subtree_mom = mom_map[subtree]
            subtree_mom_count[subtree_mom] = subtree_mom_count.get(subtree_mom, 0) + 1
            if subtree_mom_count[subtree_mom] == 2:
                all_subtrees_actions.append(subtree_mom)
                _i = all_subtree_idx_map[subtree_mom.clades[0]]
                _j = all_subtree_idx_map[subtree_mom.clades[1]]
                action_set.append(
                    # ACTION_INDICES_DICT[all_subtrees_num][(_i, _j)]
                    [_i, _j]
                )
        

        action = random.choice(action_set)
        # action_i, action_j = TREE_PAIRS_DICT[all_subtrees_num][action]
        action_i, action_j = action
        # subtree_i, subtree_j = all_subtrees[action_i], all_subtrees[action_j]

        all_subtrees.pop(action_j)
        all_subtrees[action_i] = mom_map[all_subtrees[action_i]]
        
        action_sets.append(action_set)
        actions.append(action)

    return actions, action_sets


def sample_trajectory_set_w_dist_mat(node,pos=5):
    # node clades number == 2

    EVO_DIST_TYPE_SMOOTH = False


    assert len(node.clades)==2
    current_subtrees = [node.clades[0], node.clades[1]]
    
    actions = [[0, 1]]
    actions_set = [[[0,1]]]
    if EVO_DIST_TYPE_SMOOTH:
        _d01 = node.clades[0].branch_length + node.clades[1].branch_length
        evo_dist_mat = [[0, _d01], [_d01, 0]]
    else:
        evo_dist_mat = [[0, 2], [2, 0]]


    parent_sets = {(get_min_seq_index(node.clades[0],pos=pos),get_min_seq_index(node.clades[1],pos=pos)):node}
    parents = {node.clades[0]:node,node.clades[1]:node}

    evo_dist_mat_list = [np.array(evo_dist_mat)]


    for idx in range(len(node.get_terminals())-2):
        
        new_evo_dist_mat = [
            [
                0 for _ in range(len(evo_dist_mat)+1)
            ] for _ in range(len(evo_dist_mat)+1)
        ]

        unleaf_subtree_indices = [i for i, st in enumerate(current_subtrees) if not st.is_terminal()]
        selected_subtree_idx = random.choice(unleaf_subtree_indices)
        selected_subtree = current_subtrees[selected_subtree_idx]

        current_subtrees.pop(selected_subtree_idx)
        assert len(selected_subtree.clades)==2
        current_subtrees.insert(0, selected_subtree.clades[1])
        current_subtrees.insert(0, selected_subtree.clades[0])

        parent_tree = parents[selected_subtree]
        parent_key = (get_min_seq_index(parent_tree.clades[0],pos=pos),get_min_seq_index(parent_tree.clades[1],pos=pos))
        if parent_key in parent_sets:
            parent_sets.pop(parent_key)
        selected_subtree_key = (get_min_seq_index(selected_subtree.clades[0],pos=pos),get_min_seq_index(selected_subtree.clades[1],pos=pos))
        parent_sets[selected_subtree_key] = selected_subtree
        parents[selected_subtree.clades[0]] = selected_subtree
        parents[selected_subtree.clades[1]] = selected_subtree
        if selected_subtree in parents:
            parents.pop(selected_subtree)


        indexed_subtrees = list(enumerate(current_subtrees))
        indexed_subtrees.sort(key=lambda x: get_min_seq_index(x[1],pos=pos))

        indexed_subtrees_indices_key = {get_min_seq_index(tree,pos=pos):idx for idx, (_, tree) in enumerate(indexed_subtrees)}
        
        action_set_per_step = []
        for _, node in parent_sets.items():
            a0 = indexed_subtrees_indices_key[get_min_seq_index(node.clades[0],pos=pos)]
            a1 = indexed_subtrees_indices_key[get_min_seq_index(node.clades[1],pos=pos)]
            assert a0 < a1
            action_set_per_step.append([a0,a1])
        actions_set.append(action_set_per_step)
        

        indexed_subtrees_indices = [index for index, _ in indexed_subtrees]
        sorted_index_of_left = indexed_subtrees_indices.index(0)
        sorted_index_of_right = indexed_subtrees_indices.index(1)

        current_subtrees = [x[1] for x in indexed_subtrees]

        # 保存动作
        a0 = min(sorted_index_of_left, sorted_index_of_right)
        a1 = max(sorted_index_of_left, sorted_index_of_right)
        assert indexed_subtrees_indices_key[get_min_seq_index(selected_subtree.clades[0],pos=pos)] == a0
        assert indexed_subtrees_indices_key[get_min_seq_index(selected_subtree.clades[1],pos=pos)] == a1

        actions.append([a0, a1])

        assert selected_subtree_idx == a0

        if EVO_DIST_TYPE_SMOOTH:
            left_d = selected_subtree.clades[0].branch_length
            right_d = selected_subtree.clades[1].branch_length
        else:
            left_d = 1
            right_d = 1

        for ii in range(a0):
            for jj in range(a0):
                new_evo_dist_mat[ii][jj] = evo_dist_mat[ii][jj]
        
        for ii in range(a0):
            new_evo_dist_mat[ii][a0] = evo_dist_mat[ii][a0] + left_d
            new_evo_dist_mat[a0][ii] = new_evo_dist_mat[ii][a0]

        for ii in range(a0+1,a1):
            for jj in range(a0):
                new_evo_dist_mat[ii][jj] = evo_dist_mat[ii][jj]
                new_evo_dist_mat[jj][ii] = evo_dist_mat[jj][ii]
        
        for ii in range(a0+1,a1):
            new_evo_dist_mat[ii][a0] = evo_dist_mat[ii][a0] + left_d
            new_evo_dist_mat[a0][ii] = new_evo_dist_mat[ii][a0]

        for ii in range(a0+1,a1):
            for jj in range(a0+1,a1):
                new_evo_dist_mat[ii][jj] = evo_dist_mat[ii][jj]
        
        for ii in range(a1):
            if ii == a0:
                new_evo_dist_mat[ii][a1] = left_d + right_d
                new_evo_dist_mat[a1][ii] = new_evo_dist_mat[ii][a1]
            else:
                new_evo_dist_mat[ii][a1] = evo_dist_mat[ii][a0] + right_d
                new_evo_dist_mat[a1][ii] = new_evo_dist_mat[ii][a1]

        for ii in range(a1+1, len(evo_dist_mat)+1):
            for jj in range(a0):
                new_evo_dist_mat[ii][jj] = evo_dist_mat[ii-1][jj]
                new_evo_dist_mat[jj][ii] = evo_dist_mat[jj][ii-1]
        
        for ii in range(a1+1, len(evo_dist_mat)+1):
            new_evo_dist_mat[ii][a0] = evo_dist_mat[ii-1][a0] + left_d
            new_evo_dist_mat[a0][ii] = new_evo_dist_mat[ii][a0]
        
        for ii in range(a1+1, len(evo_dist_mat)+1):
            for jj in range(a0+1, a1):
                new_evo_dist_mat[ii][jj] = evo_dist_mat[ii-1][jj]
                new_evo_dist_mat[jj][ii] = evo_dist_mat[jj][ii-1]
        
        for ii in range(a1+1, len(evo_dist_mat)+1):
            new_evo_dist_mat[ii][a1] = evo_dist_mat[ii-1][a0] + right_d
            new_evo_dist_mat[a1][ii] = new_evo_dist_mat[ii][a1]
        
        for ii in range(a1+1, len(evo_dist_mat)+1):
            for jj in range(a1+1, len(evo_dist_mat)+1):
                new_evo_dist_mat[ii][jj] = evo_dist_mat[ii-1][jj-1]


    actions.reverse()
    actions_set.reverse()
    evo_dist_mat_list.reverse()

    return actions, actions_set, evo_dist_mat_list


def get_min_seq_index(clade, pos=5):
    return int(clade.name[pos:])-1



def tree_to_phlytree(node, device, is_root=False,all_seqs=None,seq_keys=None, pos=5):
    if not is_root:
        if node.is_terminal():
            if all_seqs is None or seq_keys is None:
                seq = np.array([0])
            else:
                seq = all_seqs[seq_keys.index(node.name)]
            idx = [int(node.name[pos:])-1]
            curtree = PhyloTree(at_root=False, root_seq_data=[seq, [idx]], device=device)
            return {
                "tree": curtree,
                "branch_length": node.branch_length
            }
        else:
            left = tree_to_phlytree(node.clades[0], device, all_seqs=all_seqs,seq_keys=seq_keys, pos=pos)
            right = tree_to_phlytree(node.clades[1], device, all_seqs=all_seqs,seq_keys=seq_keys, pos=pos)

            curtree = PhyloTree(at_root=False, left_tree_data=left, right_tree_data=right, device=device)
            return {
                "tree": curtree,
                "branch_length": node.branch_length
            }
    else:
        left = tree_to_phlytree(node.clades[0], device, is_root=False,all_seqs=all_seqs,seq_keys=seq_keys, pos=pos)
        right = tree_to_phlytree(node.clades[1], device, is_root=False,all_seqs=all_seqs,seq_keys=seq_keys, pos=pos)
        seq_indices = sorted(left["tree"].seq_indices + right["tree"].seq_indices)
        curtree = UnrootedPhyloTree(log_score=None, left_tree_data=left, right_tree_data=right, branch_length=node.branch_length, seq_indices=seq_indices)
        return curtree


import itertools

ACTION_INDICES_DICT = {}
TREE_PAIRS_DICT = {}
for n in range(2, 200 + 1):
    tree_pairs = list(itertools.combinations(list(np.arange(n)), 2))
    TREE_PAIRS_DICT[n] = tree_pairs
    ACTION_INDICES_DICT[n] = {pair: idx for idx, pair in enumerate(tree_pairs)}


def custom_collate_fn(batch):
    #return None
    # import pdb; pdb.set_trace()
    batch_data = np.stack([item['data'] for item in batch])
    batch_actions = np.stack([item['action'] for item in batch])

    batch_actions_set = [item['action_set'] for item in batch]
    batch_actions_set_np = [item[0] for item in copy.deepcopy(batch_actions_set)]
    batch_actions_set_np = list(zip(*batch_actions_set_np))
    # padding
    for _step_id in range(len(batch_actions_set_np)):
        max_len = max(len(x) for x in batch_actions_set_np[_step_id])
        for x in batch_actions_set_np[_step_id]:
            x.extend([[0,0]] * (max_len - len(x)))

    batch_actions_set_tensor = [torch.from_numpy(np.stack(item)) for item in batch_actions_set_np]

    # action set complement
    batch_actions_set_complement = []
    for _step_id in range(len(batch_actions_set_np)):
        step_id = len(batch_actions_set_np) - _step_id + 1

        # print(step_id)

        batch_actions_set_complement_cur_step = []

        for batch_id in range(len(batch_actions_set_np[_step_id])):

            batch_actions_set_complement_cur_step.append(
                [
                    [p[0], p[1]] for p in ACTION_INDICES_DICT[step_id].keys()
                    if [p[0], p[1]] not in batch_actions_set[batch_id][0][_step_id]
                ]
            )
        batch_actions_set_complement.append(batch_actions_set_complement_cur_step)

    # import pdb; pdb.set_trace()

    batch_actions_set_complement_np = batch_actions_set_complement
    # padding
    for _step_id in range(len(batch_actions_set_complement_np)):
        max_len = max(len(x) for x in batch_actions_set_complement_np[_step_id])
        # print(max_len)
        for x in batch_actions_set_complement_np[_step_id]:
            # print("   ", max_len, len(x))
            # import pdb; pdb.set_trace()
            x.extend([[0,0]] * (max_len - len(x)))

    # import pdb; pdb.set_trace()
    batch_actions_set_complement_tensor = [torch.from_numpy(np.stack(item)) for item in batch_actions_set_complement_np]
    # import pdb; pdb.set_trace()


    # batch_actions_set_mask = np.stack([np.concatenate((item['action_set_mask'], np.full((t,s, max_n-item['action_set_mask'].shape[2]),False)),axis=2) for item in batch])

    batch_data = torch.from_numpy(batch_data)
    batch_actions = torch.from_numpy(batch_actions)
    # batch_actions_set = torch.from_numpy(batch_actions_set)
    # batch_actions_set_mask = torch.from_numpy(batch_actions_set_mask)

    batch_weights = np.stack([item['weights'] for item in batch])
    batch_weights = torch.from_numpy(batch_weights)

    batch_distances = np.stack([item['distances'] for item in batch])
    batch_distances = torch.from_numpy(batch_distances)

    data_seq = [item['seqs'] for item in batch]
    batch_all_seq_keys = [item['seq_keys'] for item in batch]
    batch_trees = [item['tree'] for item in batch]
    batch_raw_trees = [item['raw_tree'] for item in batch]
    batch_file_paths = [item['file_path'] for item in batch]

    return {
        'data': batch_data,
        'seqs': data_seq,
        'seq_keys': batch_all_seq_keys,
        'seq_weights': batch_weights,
        'trees': batch_trees,
        "raw_trees": batch_raw_trees,
        'actions': batch_actions,
        'actions_set': batch_actions_set,
        'actions_set_tensor': batch_actions_set_tensor,
        'actions_set_bar_tensor': batch_actions_set_complement_tensor,
        'batch_distances': batch_distances,
        'file_paths': batch_file_paths,
    }


    
def custom_infer_collate_fn(batch):
    #return None
    # import pdb; pdb.set_trace()
    batch_data = np.stack([item['data'] for item in batch])
    batch_actions = np.stack([item['action'] for item in batch])

    batch_actions_set = [item['action_set'] for item in batch]
    batch_actions_set_np = [item[0] for item in copy.deepcopy(batch_actions_set)]
    batch_actions_set_np = list(zip(*batch_actions_set_np))
    # padding
    for _step_id in range(len(batch_actions_set_np)):
        max_len = max(len(x) for x in batch_actions_set_np[_step_id])
        for x in batch_actions_set_np[_step_id]:
            x.extend([[0,0]] * (max_len - len(x)))

    batch_actions_set_tensor = [torch.from_numpy(np.stack(item)) for item in batch_actions_set_np]

    # action set complement
    batch_actions_set_complement = []
    for _step_id in range(len(batch_actions_set_np)):
        step_id = len(batch_actions_set_np) - _step_id + 1
        batch_actions_set_complement_cur_step = []

        for batch_id in range(len(batch_actions_set_np[_step_id])):

            batch_actions_set_complement_cur_step.append(
                [
                    [p[0], p[1]] for p in ACTION_INDICES_DICT[step_id].keys()
                    if [p[0], p[1]] not in batch_actions_set[batch_id][0][_step_id]
                ]
            )
        batch_actions_set_complement.append(batch_actions_set_complement_cur_step)

    # import pdb; pdb.set_trace()

    batch_actions_set_complement_np = batch_actions_set_complement
    # padding
    for _step_id in range(len(batch_actions_set_complement_np)):
        max_len = max(len(x) for x in batch_actions_set_complement_np[_step_id])
        # print(max_len)
        for x in batch_actions_set_complement_np[_step_id]:
            # print("   ", max_len, len(x))
            # import pdb; pdb.set_trace()
            x.extend([[0,0]] * (max_len - len(x)))

    # import pdb; pdb.set_trace()
    batch_actions_set_complement_tensor = [torch.from_numpy(np.stack(item)) for item in batch_actions_set_complement_np]
    # import pdb; pdb.set_trace()

    batch_data = torch.from_numpy(batch_data)
    batch_actions = torch.from_numpy(batch_actions
)

    batch_weights = np.stack([item['weights'] for item in batch])
    batch_weights = torch.from_numpy(batch_weights)

    batch_distances = np.stack([item['distances'] for item in batch])
    batch_distances = torch.from_numpy(batch_distances)

    data_seq = [item['seqs'] for item in batch]
    batch_all_seq_keys = [item['seq_keys'] for item in batch]
    batch_trees = [item['tree'] for item in batch]
    batch_file_paths = [item['file_path'] for item in batch]
    batch_taxa_num = [item['taxa_num'] for item in batch]
    batch_seq_len = [item['seq_len'] for item in batch]

    if 'C_logllop' in batch[0]:
        batch_C_logllops = [item['C_logllop'] for item in batch]
        batch_tree_strs = [item['tree_str'] for item in batch]
        return {
            'data': batch_data,
            'seqs': data_seq,
            'seq_keys': batch_all_seq_keys,
            'seq_weights': batch_weights,
            'trees': batch_trees,
            'actions': batch_actions,
            'actions_set': batch_actions_set,
            'actions_set_tensor': batch_actions_set_tensor,
            'actions_set_bar_tensor': batch_actions_set_complement_tensor,
            'file_paths': batch_file_paths,
            'C_logllops': batch_C_logllops,
            'tree_strs': batch_tree_strs,
            'taxa_nums': batch_taxa_num,
            'seq_lens': batch_seq_len,
        }
    else:
        return {
        'data': batch_data,
        'seqs': data_seq,
        'seq_keys': batch_all_seq_keys,
        'seq_weights': batch_weights,
        'trees': batch_trees,
        'actions': batch_actions,
        'actions_set': batch_actions_set,
        'actions_set_tensor': batch_actions_set_tensor,
        'actions_set_bar_tensor': batch_actions_set_complement_tensor,
        # 'batch_distances': batch_distances,
        'file_paths': batch_file_paths,
        'taxa_nums': batch_taxa_num,
        'seq_lens': batch_seq_len,
    }

    
def infer_custom_collate_fn(batch):
    #return None
    batch_data = np.stack([item['data'] for item in batch])

    batch_data = torch.from_numpy(batch_data)

    batch_weights = np.stack([item['weights'] for item in batch])
    batch_weights = torch.from_numpy(batch_weights)

    data_seq = [item['seqs'] for item in batch]
    batch_all_seq_keys = [item['seq_keys'] for item in batch]
    batch_file_paths = [item['file_path'] for item in batch]
    batch_taxa_num = [item['taxa_num'] for item in batch]
    batch_seq_len = [item['seq_len'] for item in batch]

    return {
        'data': batch_data,
        'seqs': data_seq,
        'seq_keys': batch_all_seq_keys,
        'seq_weights': batch_weights,
        'file_paths': batch_file_paths,
        'taxa_nums': batch_taxa_num,
        'seq_lens': batch_seq_len,
    }


def load_pi_instance(file_path):
    if file_path.endswith('.phy'):
        seqs, seq_keys, num_sequences, sequence_length = load_phy_file_multirow(file_path)
        pattern = r'^([a-zA-Z]+)([0-9]+)$'
        match = re.match(pattern, seq_keys[0])
        prefix = match.group(1)
        prefix_len = len(prefix)

        key_to_index = {key: int(key[prefix_len:])-1 for i, key in enumerate(seq_keys)}
        sorted_seqs = [None] * len(seqs) 
        sorted_keys = [None] * len(seq_keys)
        for seq, key in zip(seqs, seq_keys):
            sorted_seqs[key_to_index[key]] = seq
            sorted_keys[key_to_index[key]] = key
    elif file_path.endswith('.fasta') or file_path.endswith('.aln'):
        seqs, seq_keys, num_sequences, sequence_length = load_alignment_file(file_path)
        sorted_seqs = seqs
        sorted_keys = seq_keys

    length, padded_seqs, weights = only_padding_sample(sorted_seqs)
    seqs = padded_seqs
    seqs = [''.join(s) for s in zip(*seqs)]
    weights = np.array(weights, dtype=np.float32)

    distances = pairwise_hamming_distance(seqs)
    distances = np.array(distances, dtype=np.int32)

    # get data
    data = np.stack([_seq2array(seq) for seq in seqs])

    batch = [{
        'data': data,
        'seqs': sorted_seqs,
        'weights': weights,
        'seq_keys': sorted_keys,
        'file_path': file_path,
        'taxa_num': num_sequences,
        'seq_len': sequence_length,
    }]

    return infer_custom_collate_fn(batch)



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
    # 将序列字符转换为对应的索引
    seq_indices = np.array([char_to_index[char] for char in seq], dtype=np.int8)

    # 使用查找表得到最终的数组
    return lookup_table[seq_indices]



def only_padding_sample(original_seqs, target_length=-1):

    transposed_seqs = list(zip(*original_seqs))

    # 从字典中提取独特的列和相应的权重
    unique_columns_list = list(''.join(column) for column in transposed_seqs)
    weights = [1] * len(unique_columns_list)

    # 计算压缩后的MSA序列长度
    length = len(unique_columns_list)

    target_length = max(len(l) for l in original_seqs)
    if target_length % 2 == 1:
        target_length += 1

    if length <= target_length:
        # Padding
        padding_length = target_length - length
        padded_columns = ['*' * len(original_seqs)] * padding_length
        padded_weights = [0] * padding_length

        msa = unique_columns_list + padded_columns
        weights += padded_weights
    elif length > target_length:
        # Sampling
        # 使用权重进行随机采样，以得到目标长度的序列
        sampled_columns = random.choices(unique_columns_list, weights=weights, k=target_length)
        msa = sampled_columns
        weights = [1] * len(sampled_columns)
    return length, msa, weights


def compress_pad_seq(original_seqs, target_length=1024):
    # 存储新的MSA序列和权重矩阵
    # 转置序列以便操作列
    transposed_seqs = list(zip(*original_seqs))
   # 创建一个字典来存储每个独特列的出现次数
    unique_columns = {}
    for column in transposed_seqs:
        column_str = ''.join(column)
        if column_str not in unique_columns:
            unique_columns[column_str] = 1
        else:
            unique_columns[column_str] += 1

    # 从字典中提取独特的列和相应的权重
    unique_columns_list = list(unique_columns.keys())
    weights = list(unique_columns.values())

    # 计算压缩后的MSA序列长度
    compressed_length = len(unique_columns_list)


       # 根据目标长度处理序列
    if compressed_length <= target_length:
        # Padding
        padding_length = target_length - compressed_length
        padded_columns = ['*' * len(original_seqs)] * padding_length
        padded_weights = [0] * padding_length

        compressed_msa = unique_columns_list + padded_columns
        weights += padded_weights
    elif compressed_length > target_length:
        # Sampling
        # 使用权重进行随机采样，以得到目标长度的序列
        sampled_columns = random.choices(unique_columns_list, weights=weights, k=target_length)
        compressed_msa = sampled_columns
        weights = [unique_columns[col] for col in sampled_columns]

    # 转置回来得到压缩后的MSA序列
    compressed_msa = [''.join(seq) for seq in zip(*compressed_msa)]

    return compressed_length, compressed_msa, weights


def hamming_distance(str1, str2):
    """计算两个字符串之间的汉明距离"""
    if len(str1) != len(str2):
        raise ValueError("两个字符串必须有相同的长度")
    
    return sum(ch1 != ch2 for ch1, ch2 in zip(str1, str2))

def pairwise_hamming_distance(strings):
    """计算字符串列表中所有字符串对的汉明距离"""
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
    def __init__(self, root_dir, trajectory_num, device, compress=False, only_pad_sample=True, num_per_file=None, len_list=None, taxa_list=None, C_solved=False):
        self.data = []
        self.len_to_taxa = {}  # Mapping from len to list of taxa
        self.trajectory_num = trajectory_num
        self.device = device
        self.compress = compress
        self.only_pad_sample = only_pad_sample

        # Traverse through the directory structure
        # for len_folder in list(os.listdir(root_dir))[:20]:
        for len_folder in os.listdir(root_dir):
            len_path = os.path.join(root_dir, len_folder)
            if os.path.isdir(len_path):
                len_value = int(len_folder.replace("len", ""))
                if len_list is not None and len(len_list) > 0:
                    if len_value not in len_list:
                        continue
                # if len_value > 1000 or len_value < 950:
                #     continue

                self.len_to_taxa[len_value] = []

                for taxa_folder in os.listdir(len_path):
                    taxa_path = os.path.join(len_path, taxa_folder)
                    if os.path.isdir(taxa_path):
                        taxa_value = int(taxa_folder.replace("taxa", ""))
                        # if taxa_value != 40 and taxa_value!=50:
                        if taxa_list is not None and len(taxa_list) > 0:
                            if taxa_value not in taxa_list:
                                continue
                        
                        # if taxa_value > 50 or taxa_value <30:
                        #     continue
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
                                    raw_tre_path = file_path.replace('.phy', '_raw.tre')
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

                #         break
                # break
        # import pdb; pdb.set_trace()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        len_value, taxa_value, file_path, tre_path, raw_tre_path = self.data[idx]
        # print(file_path)
        start_time = time.time()
        seqs, seq_keys, num_sequences, sequence_length = load_phy_file(file_path)
        end_time1 = time.time()


        # 创建一个映射，将 seq_keys 中的键映射到其在列表中的位置
        # seq_keys_sorted = sorted(seq_keys, key=lambda x: int(x[5:]))  # 假设键格式为 "taxon" 后跟一个数字
        key_to_index = {key: int(key[5:])-1 for i, key in enumerate(seq_keys)}
        key_to_index = {key: int(key[self.pos:]) for i, key in enumerate(seq_keys)}
        idx_min = min(list(key_to_index.values()))
        key_to_index = {key:idx-idx_min for key,idx in enumerate(key_to_index)}
        # 重新排列 seqs 以匹配 seq_keys 的顺序
        sorted_seqs = [None] * len(seqs)  # 创建一个与 seqs 同样长度的列表
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

        tree = load_tree_file(tre_path, self.device)
        if raw_tre_path != "":
            # raw_tree = load_tree_file(raw_tre_path, self.device)
            raw_tree = read_tree_data(raw_tre_path)
        else:
            raw_tree = None

        end_time3 = time.time()

        # import pdb; pdb.set_trace()
        # 采样轨迹
        action = []
        action_set = []
        # evo_dist_mat = []
        for i in range(self.trajectory_num):
            # try:  
                # action_per_traj, action_set_per_traj = sample_trajectory_set(tree)
            action_per_traj, action_set_per_traj = sample_trajectory_set_bottom_top(tree)
            action.append(action_per_traj)
            action_set.append(action_set_per_traj)
            # evo_dist_mat.append(evo_dist_mat_list_per_traj)
        
        action_np = np.array(action, dtype=np.int32)
        # evo_dist_mat_np = np.array(evo_dist_mat)

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
    
    # 解析第一行以获取序列数量和长度
    num_sequences, sequence_length = map(int, lines[0].split())

    # 读取每个序列数据
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
        # 读取第一行并解析序列数和序列长度
        first_line = file.readline().strip()
        species_count, sites_count = map(int, first_line.split())
        
        # 初始化字典来存储序列数据
        sequences = {}
        
        # 初始化一个临时列表来按顺序存储序列标识符
        identifiers = []
        
        # 逐行读取文件
        for line in file:
            line = line.strip()
            # 如果行不为空，则处理该行
            if line:
                # 检测这是序列标识符行还是序列数据行
                if len(line.split()) > 1:
                    # 新序列，分割标识符和数据
                    identifier, sequence = line.split(maxsplit=1)
                    sequences[identifier] = sequence.replace(" ", "").upper()
                    identifiers.append(identifier)

            else:
                # 空行表示开头序列结束
                break
        
        # 如果空行后面还有内容，则继续读取; 考虑一个idx统计行数; 每遇到一个空行，idx置零; 当前行不空, 则sequences[identifiers[idx]] += sequence.replace(" ", ""), idx++
        idx = 0
        for line in file:
            line = line.strip()
            if line:
                sequence = line
                sequences[identifiers[idx]] += sequence.replace(" ", "").upper()
                idx += 1
            else:
                idx = 0

    # 验证每个序列的长度
    for identifier in identifiers:
        assert len(sequences[identifier]) == sites_count, f"序列 {identifier} 的长度不符合预期。"
    
    # change other symbols "k,^ ,..." to "-"
    for identifier, seq in sequences.items():
        updated_seq = ''
        for char in seq:
            if char not in HARS_DICT:
                updated_seq += '-'
            else:
                updated_seq += char
        sequences[identifier] = updated_seq

    # assert species_count
    assert len(identifiers) == species_count, "序列数与声明的数量不匹配。"

    seq_keys = list(sequences.keys())
    all_seqs = list(sequences.values())

    return all_seqs, seq_keys, species_count, sites_count


def read_phy_file_multirow(file_path):
    with open(file_path, 'r') as file:
        # 读取第一行并解析序列数和序列长度
        first_line = file.readline().strip()
        species_count, sites_count = map(int, first_line.split())
        
        # 初始化字典来存储序列数据
        sequences = {}
        
        # 初始化一个临时列表来按顺序存储序列标识符
        identifiers = []
        
        # 逐行读取文件
        for line in file:
            line = line.strip()
            # 如果行不为空，则处理该行
            if line:
                # 检测这是序列标识符行还是序列数据行
                if len(line.split()) > 1:
                    # 新序列，分割标识符和数据
                    identifier, sequence = line.split(maxsplit=1)
                    sequences[identifier] = sequence.replace(" ", "")
                    identifiers.append(identifier)

            else:
                # 空行表示开头序列结束
                break
        
        # 如果空行后面还有内容，则继续读取; 考虑一个idx统计行数; 每遇到一个空行，idx置零; 当前行不空, 则sequences[identifiers[idx]] += sequence.replace(" ", ""), idx++
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
        assert len(sequences[identifier]) == sites_count, f"序列 {identifier} 的长度不符合预期。"
    
    # assert species_count
    assert len(identifiers) == species_count, "序列数与声明的数量不匹配。"

    seq_keys = list(sequences.keys())
    all_seqs = list(sequences.values())

    return all_seqs, seq_keys, species_count, sites_count


def load_tree_file(file_path,device):
    # 读取 Phylogenetic tree
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
    sorted_tree = sorted_bio_tree(tree.root)

    return sorted_tree


def read_tree_data(raw_tre_path):
    with open(raw_tre_path, 'r') as file:
        raw_tree = file.read()
    return raw_tree


def sorted_bio_tree(node):
    #对bio phylo tree进行排序,保证每个结点左子树包含叶子标签严格小于右子树
    if node.is_terminal():
        node.id = node.name
        return node
    else:
        left = sorted_bio_tree(node.clades[0])
        right = sorted_bio_tree(node.clades[1])
        left_idx = get_min_seq_index(left)
        right_idx = get_min_seq_index(right)
        if left_idx > right_idx:
            node.clades = [right,left]
            node.id = node.name
            node.name = right.name
        else:
            node.id = node.name
            node.name = left.name
        return node


def sample_trajectory(node):
    # 将树分解为子树列表
    # node clades number == 2
    assert len(node.clades)==2
    current_subtrees = [node.clades[0], node.clades[1]]

    # 存储每个子树的动作
    actions = [[0, 1]]
    # actions = []

    # while 循环，直到所有叶节点被处理
    for idx in range(len(node.get_terminals())-2):
        # 选择一个非叶子节点的子树
        unleaf_subtree_indices = [i for i, st in enumerate(current_subtrees) if not st.is_terminal()]
        selected_subtree_idx = random.choice(unleaf_subtree_indices)
        selected_subtree = current_subtrees[selected_subtree_idx]

        # 移除选定的子树，并添加其子节点
        current_subtrees.pop(selected_subtree_idx)
        assert len(selected_subtree.clades)==2
        current_subtrees.insert(0, selected_subtree.clades[1])
        current_subtrees.insert(0, selected_subtree.clades[0])

        # 对子树进行排序
        indexed_subtrees = list(enumerate(current_subtrees))
        indexed_subtrees.sort(key=lambda x: get_min_seq_index(x[1]))

        indexed_subtrees_indices = [index for index, _ in indexed_subtrees]
        sorted_index_of_left = indexed_subtrees_indices.index(0)
        sorted_index_of_right = indexed_subtrees_indices.index(1)

        current_subtrees = [x[1] for x in indexed_subtrees]

        # 保存动作
        a0 = min(sorted_index_of_left, sorted_index_of_right)
        a1 = max(sorted_index_of_left, sorted_index_of_right)

        actions.append([a0, a1])

    actions.reverse()
    return actions


def sample_trajectory_set(node):
    # 将树分解为子树列表
    # node clades number == 2
    assert len(node.clades)==2
    current_subtrees = [node.clades[0], node.clades[1]]

    # 存储每个子树的动作
    actions = [[0, 1]]
    # 存储选择子树动作的直接父结点, 使用get_min_seq_index来索引对应结点
    parent_sets = {(get_min_seq_index(node.clades[0]),get_min_seq_index(node.clades[1])):node}
    parents = {node.clades[0]:node,node.clades[1]:node}
    # 存储当前步所有可能的动作
    actions_set = [[[0,1]]]

    # while 循环，直到所有叶节点被处理
    for idx in range(len(node.get_terminals())-2):
        # 选择一个非叶子节点的子树
        unleaf_subtree_indices = [i for i, st in enumerate(current_subtrees) if not st.is_terminal()]
        selected_subtree_idx = random.choice(unleaf_subtree_indices)
        selected_subtree = current_subtrees[selected_subtree_idx]

        # 移除选定的子树，并添加其子节点
        current_subtrees.pop(selected_subtree_idx)
        assert len(selected_subtree.clades)==2
        current_subtrees.insert(0, selected_subtree.clades[1])
        current_subtrees.insert(0, selected_subtree.clades[0])

        # 从parent_sets中移除选定的子树的父结点，并添加选定子树结点
        parent_tree = parents[selected_subtree]
        parent_key = (get_min_seq_index(parent_tree.clades[0]),get_min_seq_index(parent_tree.clades[1]))
        if parent_key in parent_sets:
            parent_sets.pop(parent_key)
        selected_subtree_key = (get_min_seq_index(selected_subtree.clades[0]),get_min_seq_index(selected_subtree.clades[1]))
        parent_sets[selected_subtree_key] = selected_subtree
        parents[selected_subtree.clades[0]] = selected_subtree
        parents[selected_subtree.clades[1]] = selected_subtree
        if selected_subtree in parents:
            parents.pop(selected_subtree)


        # 对子树进行排序
        indexed_subtrees = list(enumerate(current_subtrees))
        indexed_subtrees.sort(key=lambda x: get_min_seq_index(x[1]))

        # 获得子树的名字name对应的idx ：taxon<num> -> num
        indexed_subtrees_indices_key = {get_min_seq_index(tree):idx for idx, (_, tree) in enumerate(indexed_subtrees)}
        
        # 得到parent_set中每一个结点的子结点在indexed_subtrees_indices中的idx，构造(a0,a1)的集合
        # 保存动作
        action_set_per_step = []
        for _, node in parent_sets.items():
            a0 = indexed_subtrees_indices_key[get_min_seq_index(node.clades[0])]
            a1 = indexed_subtrees_indices_key[get_min_seq_index(node.clades[1])]
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
        assert indexed_subtrees_indices_key[get_min_seq_index(selected_subtree.clades[0])] == a0
        assert indexed_subtrees_indices_key[get_min_seq_index(selected_subtree.clades[1])] == a1

        actions.append([a0, a1])
    
    actions.reverse()
    actions_set.reverse()
    return actions, actions_set


def sample_trajectory_set_bottom_top(node):
    # 将树分解为子树列表
    # nde clades number == 2
    assert len(node.clades)==2
    current_subtrees = [node.clades[0], node.clades[1]]
    
    mom_map = dict()

    def build_mom_map(node, mom_map, parent=None):
        # 如果当前节点不是根节点，则将其添加到映射中
        if parent is not None:
            mom_map[node] = parent

        # 如果当前节点不是叶子节点，则递归遍历其孩子节点
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
        all_subtrees = sorted(all_subtrees, key=lambda x: int(x.name[5:]))
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
        
        # all_subtrees_actions = sorted(all_subtrees_actions, key=lambda x: int(x.name[5:]))

        action = random.choice(action_set)
        # action_i, action_j = TREE_PAIRS_DICT[all_subtrees_num][action]
        action_i, action_j = action
        # subtree_i, subtree_j = all_subtrees[action_i], all_subtrees[action_j]

        all_subtrees.pop(action_j)
        all_subtrees[action_i] = mom_map[all_subtrees[action_i]]
        
        action_sets.append(action_set)
        actions.append(action)

    return actions, action_sets


def sample_trajectory_set_w_dist_mat(node):
    # 将树分解为子树列表
    # node clades number == 2

    EVO_DIST_TYPE_SMOOTH = False


    assert len(node.clades)==2
    current_subtrees = [node.clades[0], node.clades[1]]
    
    # 存储sample的动作，第一步拆只能选择 [0,1]
    actions = [[0, 1]]
    # 存储当前步所有可能的动作
    actions_set = [[[0,1]]]
    # 存储当前步所有子树的进化距离矩阵
    if EVO_DIST_TYPE_SMOOTH:
        _d01 = node.clades[0].branch_length + node.clades[1].branch_length
        evo_dist_mat = [[0, _d01], [_d01, 0]]
    else:
        evo_dist_mat = [[0, 2], [2, 0]]


    # 存储选择子树动作的直接父结点, 使用get_min_seq_index来索引对应结点
    parent_sets = {(get_min_seq_index(node.clades[0]),get_min_seq_index(node.clades[1])):node}
    parents = {node.clades[0]:node,node.clades[1]:node}

    evo_dist_mat_list = [np.array(evo_dist_mat)]


    # while 循环，直到所有叶节点被处理
    for idx in range(len(node.get_terminals())-2):
        
        new_evo_dist_mat = [
            [
                0 for _ in range(len(evo_dist_mat)+1)
            ] for _ in range(len(evo_dist_mat)+1)
        ]

        # 选择一个非叶子节点的子树
        unleaf_subtree_indices = [i for i, st in enumerate(current_subtrees) if not st.is_terminal()]
        selected_subtree_idx = random.choice(unleaf_subtree_indices)
        selected_subtree = current_subtrees[selected_subtree_idx]

        # 移除选定的子树，并添加其子节点
        current_subtrees.pop(selected_subtree_idx)
        assert len(selected_subtree.clades)==2
        current_subtrees.insert(0, selected_subtree.clades[1])
        current_subtrees.insert(0, selected_subtree.clades[0])

        # 从parent_sets中移除选定的子树的父结点，并添加选定子树结点
        parent_tree = parents[selected_subtree]
        parent_key = (get_min_seq_index(parent_tree.clades[0]),get_min_seq_index(parent_tree.clades[1]))
        if parent_key in parent_sets:
            parent_sets.pop(parent_key)
        selected_subtree_key = (get_min_seq_index(selected_subtree.clades[0]),get_min_seq_index(selected_subtree.clades[1]))
        parent_sets[selected_subtree_key] = selected_subtree
        parents[selected_subtree.clades[0]] = selected_subtree
        parents[selected_subtree.clades[1]] = selected_subtree
        if selected_subtree in parents:
            parents.pop(selected_subtree)


        # 对子树进行排序
        indexed_subtrees = list(enumerate(current_subtrees))
        indexed_subtrees.sort(key=lambda x: get_min_seq_index(x[1]))

        # 获得子树的名字name对应的idx ：taxon<num> -> num
        indexed_subtrees_indices_key = {get_min_seq_index(tree):idx for idx, (_, tree) in enumerate(indexed_subtrees)}
        
        # 得到parent_set中每一个结点的子结点在indexed_subtrees_indices中的idx，构造(a0,a1)的集合
        # 保存动作
        action_set_per_step = []
        for _, node in parent_sets.items():
            a0 = indexed_subtrees_indices_key[get_min_seq_index(node.clades[0])]
            a1 = indexed_subtrees_indices_key[get_min_seq_index(node.clades[1])]
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
        assert indexed_subtrees_indices_key[get_min_seq_index(selected_subtree.clades[0])] == a0
        assert indexed_subtrees_indices_key[get_min_seq_index(selected_subtree.clades[1])] == a1

        actions.append([a0, a1])

        assert selected_subtree_idx == a0

        if EVO_DIST_TYPE_SMOOTH:
            left_d = selected_subtree.clades[0].branch_length
            right_d = selected_subtree.clades[1].branch_length
        else:
            left_d = 1
            right_d = 1

        # 更新进化距离矩阵
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


        # def print_block_matrix(matrix, a0, a1=None):
        #     n = len(matrix)

        #     if not a1:
        #         a1 = -1
        #         pad = 3
        #     else:
        #         pad = 7
            
        #     if a0 == 0:
        #         print('\n' + '-' * (5 * n + pad))

        #     for i in range(n):
        #         for j in range(n):
        #             if j == 0 and a0 == 0:
        #                 print('|', end=' ')
        #             print('{:.2f}'.format(matrix[i][j]), end=' ')
        #             if j == a0 - 1 or j == a0 or j == a1-1 or j == a1:
        #                 print('|', end=' ')
        #         if i == a0 - 1 or i == a0 or i == a1 - 1 or i == a1:
        #             print('\n' + '-' * (5 * n + pad))
        #         else:
        #             print()


        # print(a0, selected_subtree.clades[0].branch_length)
        # print(a1, selected_subtree.clades[1].branch_length)
        # print()
        # print()

        # print_block_matrix(evo_dist_mat, a0)
        # print()
        # print()
        # print_block_matrix(new_evo_dist_mat, a0, a1)

        evo_dist_mat = new_evo_dist_mat

        evo_dist_mat_list.append(np.array(evo_dist_mat))


        # import pdb; pdb.set_trace()

    actions.reverse()
    actions_set.reverse()
    evo_dist_mat_list.reverse()

    return actions, actions_set, evo_dist_mat_list


def get_min_seq_index(clade):
    # 此函数用于获取 clade 中最小的序列索引
    # 假设每个叶节点的名称形式为 'taxon<num>'
    # 对于非叶节点，递归地找到所有叶子节点并返回最小的索引
    # if clade.is_terminal():
    return int(clade.name[5:])-1
    # else:
    #     return min(get_min_seq_index(c) for c in clade.clades)


def tree_to_phlytree(node, device, is_root=False,all_seqs=None,seq_keys=None):
    if not is_root:
        if node.is_terminal():
            if all_seqs is None or seq_keys is None:
                seq = np.array([0])
            else:
                seq = all_seqs[seq_keys.index(node.name)]
            idx = [int(node.name[5:])-1]
            curtree = PhyloTree(at_root=False, root_seq_data=[seq, [idx]], device=device)
            return {
                "tree": curtree,
                "branch_length": node.branch_length
            }
        else:
            left = tree_to_phlytree(node.clades[0], device, all_seqs=all_seqs,seq_keys=seq_keys)
            right = tree_to_phlytree(node.clades[1], device, all_seqs=all_seqs,seq_keys=seq_keys)

            curtree = PhyloTree(at_root=False, left_tree_data=left, right_tree_data=right, device=device)
            return {
                "tree": curtree,
                "branch_length": node.branch_length
            }
    else:
        left = tree_to_phlytree(node.clades[0], device, is_root=False,all_seqs=all_seqs,seq_keys=seq_keys)
        right = tree_to_phlytree(node.clades[1], device, is_root=False,all_seqs=all_seqs,seq_keys=seq_keys)
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
    # batch 是一个列表，其中包含了来自 Dataset 的多个返回值
    # 每个元素是 __getitem__ 的输出
    # import pdb; pdb.set_trace()
    batch_data = np.stack([item['data'] for item in batch])
    batch_actions = np.stack([item['action'] for item in batch])

    # max_n = max([item['action_set'].shape[2] for item in batch])
    # t , s = batch[0]['action_set'].shape[:2]
    # batch_actions_set = np.stack([np.concatenate((item['action_set'], np.full((t,s, max_n-item['action_set'].shape[2], 2),-1)),axis=2) for item in batch])
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

    # batch_evo_dist_mat = [item['evo_dist_mat'] for item in batch]
    # batch_evo_dist_mat = [item[0] for item in batch_evo_dist_mat]
    # batch_evo_dist_mat = list(zip(*batch_evo_dist_mat))
    # batch_evo_dist_mat = [torch.from_numpy(np.stack(item)) for item in batch_evo_dist_mat]

    # import pdb; pdb.set_trace()

    # # 将batch_data转换为torch.Tensor并移动到GPU上（如果有GPU可用）
    # batch_data = [torch.tensor(data).to(batch['device']) for data in batch_data]
    # 重新组织 batch_data 以匹配期望的结构
    # batch_data 应该是一个列表，其中包含了 batch_size 个 taxa 长度的列表
    # 每个这样的列表包含了 len_value 长度的字符串

    # 返回重新组织后的字典
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
        # 'actions_set_mask': batch_actions_set_mask,
        'batch_distances': batch_distances,
        'file_paths': batch_file_paths,
        # 'evo_dist_mat': batch_evo_dist_mat
        # 'evo_dist_mat': None
    }


def infer_custom_collate_fn(batch):
    #return None
    # batch 是一个列表，其中包含了来自 Dataset 的多个返回值
    # 每个元素是 __getitem__ 的输出
    # import pdb; pdb.set_trace()
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

    # try:
    #     seqs, seq_keys, num_sequences, sequence_length = load_phy_file(file_path)
    # except:
    seqs, seq_keys, num_sequences, sequence_length = load_phy_file_multirow(file_path)

    pattern = r'^([a-zA-Z]+)([0-9]+)$'
    match = re.match(pattern, seq_keys[0])
    prefix = match.group(1)
    prefix_len = len(prefix)

    # 创建一个映射，将 seq_keys 中的键映射到其在列表中的位置
    # seq_keys_sorted = sorted(seq_keys, key=lambda x: int(x[self.self.pos:]))  # 假设键格式为 "Sp" 后跟一个数字
    key_to_index = {key: int(key[prefix_len:])-1 for i, key in enumerate(seq_keys)}
    # 重新排列 seqs 以匹配 seq_keys 的顺序
    sorted_seqs = [None] * len(seqs)  # 创建一个与 seqs 同样长度的列表
    sorted_keys = [None] * len(seq_keys)
    for seq, key in zip(seqs, seq_keys):
        sorted_seqs[key_to_index[key]] = seq
        sorted_keys[key_to_index[key]] = key

    # import pdb; pdb.set_trace()
    # compressed_seqs
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


if __name__ == "__main__":

    # preprocess_data(root_dir,output_dir)

    batch_size = 32  # 根据需要设置批处理大小
    device=torch.device("cuda:2")
    val_path = "data_gen/fixed_len_data"
    #dataset = PhyDataset('data_gen/data', trajectory_num=2, device="cpu")
    dataset = PhyDataset(val_path, trajectory_num=1, device=device, num_per_file=1024, taxa_list=[100], C_solved=False)
    sampler = PhySampler(dataset, batch_size)
    data_loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler, collate_fn=custom_collate_fn, num_workers=0)

    print("Data loaded")

    import time

    # 使用 DataLoader
    last_time = time.time()
    for batch in data_loader:
        # 处理 batch 数据
        # import pdb; pdb.set_trace()
        print(f"data: {len(batch['data'][0][0])}\n seq_keys: {len(batch['seq_keys'][0])}\n")
        # print(len(batch['evo_dist_mat']))
        # import pdb; pdb.set_trace()
        cost_time = time.time() - last_time
        last_time = time.time()
        print(f"Cost time: {cost_time}\n")
        #break

    # # 创建数据集实例
    # phy_dataset = PhylogeneticDataset(output_dir,device='cpu')

    # # 创建 DataLoader
    # phy_dataloader = torch.utils.data.DataLoader(phy_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn, num_workers=0)
    # last_time = time.time()
    # for batch in phy_dataloader:
    #     print(f"len: {batch['data'][0].shape}\n")
    #     print(f"len: {batch['keys'][0]}\n")
    #     print(f"len: {batch['trees'][0].topo_repr}\n")
    #     cost_time = time.time() - last_time
    #     last_time = time.time()
    #     print(f"Cost time: {cost_time}\n")
    #     # import pdb; pdb.set_trace()



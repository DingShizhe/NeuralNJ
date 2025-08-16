import itertools
import random
from copy import deepcopy
from multiprocessing import Pool
from typing import List, Union

import numpy as np
import raxmlpy as pllpy
import torch

CHARACTERS_MAPS = {
    "DNA": {
        "A": [1.0, 0.0, 0.0, 0.0],
        "C": [0.0, 1.0, 0.0, 0.0],
        "G": [0.0, 0.0, 1.0, 0.0],
        "T": [0.0, 0.0, 0.0, 1.0],
        "N": [1.0, 1.0, 1.0, 1.0],
    },
    "RNA": {
        "A": [1.0, 0.0, 0.0, 0.0],
        "C": [0.0, 1.0, 0.0, 0.0],
        "G": [0.0, 0.0, 1.0, 0.0],
        "U": [0.0, 0.0, 0.0, 1.0],
        "N": [1.0, 1.0, 1.0, 1.0],
    },
    "DNA_WITH_GAP": {
        "A": [1.0, 0.0, 0.0, 0.0],
        "C": [0.0, 1.0, 0.0, 0.0],
        "G": [0.0, 0.0, 1.0, 0.0],
        "T": [0.0, 0.0, 0.0, 1.0],
        "-": [1.0, 1.0, 1.0, 1.0],
        "N": [1.0, 1.0, 1.0, 1.0],
    },
    "RNA_WITH_GAP": {
        "A": [1.0, 0.0, 0.0, 0.0],
        "C": [0.0, 1.0, 0.0, 0.0],
        "G": [0.0, 0.0, 1.0, 0.0],
        "U": [0.0, 0.0, 0.0, 1.0],
        "-": [1.0, 1.0, 1.0, 1.0],
        "N": [1.0, 1.0, 1.0, 1.0],
    },
}

evolution_model = 'GTR+I+G'
# evolution_model = 'JC'


class PhyloTree(object):

    def __init__(
        self,
        at_root,
        left_tree_data=None,
        right_tree_data=None,
        root_seq_data=None,
        name="",
        device="cpu",
    ):
        """
        BINARY TREE REPRESENTATION OF THE PHYLOGENETIC TREE
        """
        assert (left_tree_data is not None and right_tree_data is not None) or (
            root_seq_data is not None
        )

        self.device = device

        self.left_tree_data = left_tree_data
        self.right_tree_data = right_tree_data
        self.at_root = at_root
        if root_seq_data is not None:
            seq_indices = root_seq_data
            self.seq_indices = seq_indices
            self.total_mutations = 0
            self.seq_indices_str = str(self.seq_indices)
            self.root_rep_str = self.seq_indices_str
            self.log_score = 0
        else:
            left_idx = left_tree_data["tree"].seq_indices[0]
            right_idx = right_tree_data["tree"].seq_indices[0]
            if left_idx > right_idx:
                self.left_tree_data = right_tree_data
                self.right_tree_data = left_tree_data

            self.root_seq, self.log_score = None, None
            seq_indices = (
                self.left_tree_data["tree"].seq_indices
                + self.right_tree_data["tree"].seq_indices
            )
            self.seq_indices = sorted(seq_indices)
            self.seq_indices_str = str(self.seq_indices)
        self.min_seq_index = min(self.seq_indices)
        self.name = name

    @property
    def is_leaf(self):
        return self.left_tree_data is None and self.right_tree_data is None

    @property
    def is_internal(self):
        return self.left_tree_data is not None or self.right_tree_data is not None

    def build_tree_str(self):
        if self.left_tree_data is None:
            return self.root_rep_str + "\n"

        l = self.left_tree_data["tree"]
        r = self.right_tree_data["tree"]
        left_str = str(l)
        right_str = str(r)
        left_str_parts = left_str.split("\n")[:-1]
        right_str_parts = right_str.split("\n")[:-1]

        data = [self.root_rep_str]
        data.append("├── " + left_str_parts[0])
        for p in left_str_parts[1:]:
            data.append("│   " + p)
        data.append("└── " + right_str_parts[0])
        for p in right_str_parts[1:]:
            data.append("    " + p)

        new_tree_str = "\n".join(data) + "\n"
        return new_tree_str

    def postorder_traversal_internal(self):
        """
        perform post order traversal over the tree, return the internal nodes
        In principle, this function will not be used or called because binary rooted tree is not the final tree
        :return: list of internal nodes,  dictionary of edge lengths between the node and its direct parent
        """
        # have a dictionary recording pairwise distances
        tree_edge_lengths = {}

        # leaves not included in the traversal
        stack = [self]
        stack_rev_postorder = []
        while len(stack) > 0:
            node = stack.pop()
            node_str = node.seq_indices_str
            if node.is_internal:
                # only keep internal nodes
                stack_rev_postorder.append(node)

            if node.left_tree_data is not None:
                for t in [node.left_tree_data, node.right_tree_data]:
                    t_str = t["tree"].seq_indices_str
                    stack.append(t["tree"])
                    tree_edge_lengths[(t_str, node_str)] = t["branch_length"]
                    tree_edge_lengths[(node_str, t_str)] = t["branch_length"]

        traversed_list = stack_rev_postorder[::-1]
        return traversed_list, tree_edge_lengths

    def to_unrooted_tree(self):

        traversed_list, tree_edge_lengths = (
            self.postorder_traversal_internal()
        )  # this list only contains internal nodes
        nb_internal_nodes = len(traversed_list)

        # nx_graph = nx.Graph()
        # ALL NODES HERE HAVE CHILDREN
        for idx, node in enumerate(traversed_list):

            left_tree = node.left_tree_data["tree"]
            right_tree = node.right_tree_data["tree"]

            if idx == nb_internal_nodes - 1:
                # add the distance between the left and right node to the edge length dict
                node_str = node.seq_indices_str
                l_str = left_tree.seq_indices_str
                r_str = right_tree.seq_indices_str
                total_edge_length = (
                    tree_edge_lengths[(node_str, l_str)]
                    + tree_edge_lengths[(node_str, r_str)]
                    if tree_edge_lengths[(node_str, l_str)] is not None
                    and tree_edge_lengths[(node_str, r_str)] is not None
                    else None
                )
                tree_edge_lengths[(l_str, r_str)] = total_edge_length
                tree_edge_lengths[(r_str, l_str)] = total_edge_length
                del tree_edge_lengths[(l_str, node_str)]
                del tree_edge_lengths[(node_str, l_str)]
                del tree_edge_lengths[(r_str, node_str)]
                del tree_edge_lengths[(node_str, r_str)]
            else:
                pass

        seq_indices = sorted(
            self.left_tree_data["tree"].seq_indices
            + self.right_tree_data["tree"].seq_indices
        )
        assert seq_indices == self.seq_indices
        return UnrootedPhyloTree(
            self.log_score,
            self.left_tree_data,
            self.right_tree_data,
            tree_edge_lengths,
            self.seq_indices,
        )


class UnrootedPhyloTree(object):
    def __init__(
        self,
        log_score,
        left_tree_data,
        right_tree_data,
        branch_length,
        seq_indices,
        name="",
    ):
        self.left_tree_data = left_tree_data
        self.right_tree_data = right_tree_data
        self.branch_length = branch_length
        self.log_score = log_score
        self.seq_indices = seq_indices

        self.topo_repr = format_rtree_topology(self, True, None)
        import pdb; pdb.set_trace
        self.name = name

    def sample_trajectory(self):
        current_subtrees = [self.left_tree_data["tree"], self.right_tree_data["tree"]]
        current_subtrees = sorted(current_subtrees, key=lambda x: x.min_seq_index)
        actions = [[0, 1]]

        for idx in range(len(self.seq_indices) - 2):

            unleaf_subtree_indices = [
                idx for idx, x in enumerate(current_subtrees) if not x.is_leaf
            ]

            # print(len(unleaf_subtree_indices), len(current_subtrees))

            selected_subtree_idx = random.choice(unleaf_subtree_indices)
            selected_subtree = current_subtrees[selected_subtree_idx]

            current_subtrees.pop(selected_subtree_idx)
            current_subtrees.insert(0, selected_subtree.right_tree_data["tree"])
            current_subtrees.insert(0, selected_subtree.left_tree_data["tree"])

            indexed_subtrees = list(enumerate(current_subtrees))
            indexed_subtrees.sort(key=lambda x: x[1].min_seq_index)

            indexed_subtrees_indices = [index for index, _ in indexed_subtrees]

            sorted_index_of_left = indexed_subtrees_indices.index(0)
            sorted_index_of_right = indexed_subtrees_indices.index(1)

            current_subtrees = [x[1] for x in indexed_subtrees]

            a0 = min(sorted_index_of_left, sorted_index_of_right)
            a1 = max(sorted_index_of_left, sorted_index_of_right)

            actions.append([a0, a1])

        actions.reverse()

        return actions


def format_rtree_topology(tree, at_root=False, sequence_keys=None):
    if tree.left_tree_data is None:
        assert len(tree.seq_indices) == 1
        k = sequence_keys[tree.seq_indices[0]] if sequence_keys else tree.seq_indices[0]
        return f"{k}"
    left_tree = tree.left_tree_data["tree"]
    right_tree = tree.right_tree_data["tree"]

    left = format_rtree_topology(left_tree, False, sequence_keys)
    right = format_rtree_topology(right_tree, False, sequence_keys)

    if at_root:
        return f"({left}, {right});"
    else:
        return f"({left}, {right})"


def format_rtree(tree, at_root=False, branch_length=None, sequence_keys=None):

    assert sequence_keys is not None

    b = branch_length if branch_length is not None else 0.12345

    if tree.left_tree_data is None:
        assert len(tree.seq_indices) == 1
        k = sequence_keys[tree.seq_indices[0]]
        return f"{k}:{b}"

    left_tree = tree.left_tree_data["tree"]
    right_tree = tree.right_tree_data["tree"]

    left = format_rtree(
        left_tree, False, tree.left_tree_data["branch_length"], sequence_keys
    )
    right = format_rtree(
        right_tree, False, tree.right_tree_data["branch_length"], sequence_keys
    )

    if at_root:
        return f"({left}, {right});"
    else:
        return f"({left}, {right}):{b}"


def assign_branch_length_rtree(tree, tree_tuple):

    if tree.left_tree_data is None:
        return

    left_tree = tree.left_tree_data["tree"]
    right_tree = tree.right_tree_data["tree"]

    left_tree_tuple, left_tree_bl, right_tree_tuple, right_tree_bl = tree_tuple

    tree.left_tree_data["branch_length"] = left_tree_bl
    tree.right_tree_data["branch_length"] = right_tree_bl

    assign_branch_length_rtree(left_tree, left_tree_tuple)
    assign_branch_length_rtree(right_tree, right_tree_tuple)


class PhylogeneticTreeState(object):

    def __init__(self, subtrees: List[Union[PhyloTree, UnrootedPhyloTree]]):

        self.subtrees = subtrees
        self.is_done = len(self.subtrees) == 1
        self.num_trees = len(self.subtrees)

        if type(subtrees[0]) == PhyloTree:
            trees = [x for x in self.subtrees if x.left_tree_data is not None]
            self.is_initial = len(trees) == 0
            self.last_state = False
            self.log_score = None
        else:
            self.is_initial = False
            self.last_state = True
            self.log_score = subtrees[0].log_score

    def display(self):
        if self.last_state:
            self.subtrees[0].display()
        else:
            tree_reps = [str(x) for x in self.subtrees]
            tree_reps_splitted = [x.split("\n") for x in tree_reps]
            num_lines = max([len(x) for x in tree_reps_splitted])
            reps = [""] * num_lines

            for splitted_rep in tree_reps_splitted:
                max_width = max([len(x) for x in splitted_rep])
                for idx in range(num_lines):
                    if idx < len(splitted_rep):
                        line = splitted_rep[idx]
                    else:
                        line = ""
                    reps[idx] = reps[idx] + line + " " * (max_width - len(line))
                    reps[idx] = reps[idx] + "\t"

            reps = "\n".join(reps)
            print(reps)


def process_(args):
    rtree_str, sequence_keys, sequences, seq_indices = args
    rtree = pllpy.treestr_to_tuples(rtree_str)
    tree_msa = {
        "labels": [sequence_keys[seq_ii] for seq_ii in seq_indices],
        "sequences": [sequences[seq_ii] for seq_ii in seq_indices],
    }

    utree_op_str, log_score_pre, log_score_op = pllpy.optimize_brlen(
        rtree_str, tree_msa, is_root=True, iters=3, model=evolution_model, opt_model=True
    )
    utree_op_tuple = pllpy.treestr_to_tuples(utree_op_str)
    rtree_op_tuple = pllpy.utree2rtree_guided(utree_op_tuple, rtree)

    return rtree_op_tuple, utree_op_tuple, utree_op_str, log_score_op


def get_logscore_from_tree_str(args):
    rtree_str, sequence_keys, sequences, seq_indices = args
    tree_msa = {
        "labels": [sequence_keys[seq_ii] for seq_ii in seq_indices],
        "sequences": [sequences[seq_ii] for seq_ii in seq_indices],
    }
    utree_op_str, log_score_pre, log_score_op = pllpy.optimize_brlen(
        rtree_str, tree_msa, is_root=True, iters=3, model=evolution_model, opt_model=True
    )
    return log_score_op


def compute_raw_tree_log_score(env, rtree_str_batch, parallel=False):
    batch_size = len(rtree_str_batch)
    if parallel:
        args_list = []
        for idx in range(batch_size):
            # new_tree = new_trees_constructed[idx]
            sequence_keys = env.seq_keys[idx]
            sequences = env.batch_seqs[idx]
            seq_idx = [i for i in range(len(sequence_keys))]
            rtree_str = rtree_str_batch[idx]
            args = (rtree_str, sequence_keys, sequences, seq_idx)
            args_list.append(args)

        with Pool(8) as pool:
            results = pool.map(get_logscore_from_tree_str, args_list)

        log_scores_op = []

        for idx in range(batch_size):
            rtree_op_tuple, utree_op_tuple, utree_op_str, log_score_op = results[idx]
            log_scores_op.append(log_score_op)

    else:

        log_scores_op = []

        for idx in range(batch_size):

            rtree_str = rtree_str_batch[idx]

            sequence_keys = env.seq_keys[idx]
            sequences = env.batch_seqs[idx]
            seq_idx = [i for i in range(len(sequence_keys))]

            # Optimize using C solver
            # rtree = pllpy.treestr_to_tuples(rtree_str)

            tree_msa = {
                "labels": [sequence_keys[seq_ii] for seq_ii in seq_idx],
                "sequences": [sequences[seq_ii] for seq_ii in seq_idx],
            }

            utree_op_str, log_score_pre, log_score_op = pllpy.optimize_brlen(
                rtree_str, tree_msa, is_root=True, iters=3, model=evolution_model, opt_model=True
            )
            log_scores_op.append(log_score_op)

        return log_scores_op


class PhyInferEnv(object):
    def __init__(self, cfg, device):
        self.device = device
        self.chars_dict = CHARACTERS_MAPS[cfg.env.sequence_type]
        self.states = None
        self.state_tensor = None

    def init_states(self, batch_seqs, seq_keys, seq_arrays, label_trees=None, step_action=False):
        self.batch_seqs = batch_seqs
        self.seq_keys = seq_keys

        self.action_indices_dict = {}
        self.tree_pairs_dict = {}
        for n in range(2, len(self.batch_seqs[0]) + 1):
            tree_pairs = list(itertools.combinations(list(np.arange(n)), 2))
            self.tree_pairs_dict[n] = tree_pairs
            self.action_indices_dict[n] = {
                pair: idx for idx, pair in enumerate(tree_pairs)
            }

        self.batch_size = len(self.batch_seqs)

        self.states = []

        for batch_id in range(self.batch_size):
            phylo_trees = [
                PhyloTree(
                    at_root=False,
                    root_seq_data=[idx],
                    device=self.device,
                    name=self.seq_keys[batch_id][idx],
                )
                for idx in range(len(self.batch_seqs[batch_id]))
            ]
            new_state = PhylogeneticTreeState(phylo_trees)
            self.states.append(new_state)

        self.init_state_tensor = seq_arrays
        self.state_tensor = None

        if label_trees is not None:
            self.label_trees = label_trees
            self.mom_maps = [dict() for _ in self.label_trees]
            for batch_id, mp in enumerate(self.mom_maps):
                self.build_mom_map(self.label_trees[batch_id], mp)
            self.count_match_label_steps = [0 for _ in range(self.batch_size)]

            if step_action:
                self.batch_action_set_step = [
                    self.get_step_action(self.mom_maps[idx], self.states[idx].subtrees)
                    for idx in range(self.batch_size)
                ]
            else:
                self.batch_action_set_step = []

        else:
            self.label_trees = None
            self.mom_maps = dict()
            self.batch_action_set_step = []

    def build_mom_map(self, node, mom_map, parent=None):
        if parent is not None:
            mom_map[node.id] = parent.id

        if not node.is_terminal():
            for child in node.clades:
                self.build_mom_map(child, mom_map, node)

    def get_step_action(self, mom_map, all_subtrees):
        # how to sort subtrees
        all_subtrees_num = len(all_subtrees)
        # all_subtrees = sorted(all_subtrees, key=lambda x: int(x.name[5:]))
        # assert
        all_subtree_idx_map = {
            subtree.name: i for i, subtree in enumerate(all_subtrees)
        }

        subtree_mom_count = dict()
        subtree_mon_child = dict()
        action_set = []
        for subtree_id, subtree in enumerate(all_subtrees):
            subtree_mom_name = mom_map[subtree.name]
            try:
                subtree_mom_count[subtree_mom_name] = (
                    subtree_mom_count.get(subtree_mom_name, 0) + 1
                )
                subtree_mon_child[subtree_mom_name] = subtree_mon_child.get(
                    subtree_mom_name, []
                ) + [subtree]
            except:
                import pdb

                pdb.set_trace()
            if subtree_mom_count[subtree_mom_name] == 2:
                _i = all_subtree_idx_map[subtree_mon_child[subtree_mom_name][0].name]
                _j = all_subtree_idx_map[subtree_mon_child[subtree_mom_name][1].name]
                assert _i < _j
                action_set.append(
                    # ACTION_INDICES_DICT[all_subtrees_num][(_i, _j)]
                    [_i, _j]
                )

        return action_set

    @staticmethod
    def remove_conserved_sites(sequences):
        ret_sequences = deepcopy(sequences)

        seq_len = len(ret_sequences[0])
        all_same_indices = []

        for idx in range(seq_len):
            ref_char = ret_sequences[0][idx]
            all_same = True
            for seq_idx in range(1, len(ret_sequences)):
                if ret_sequences[seq_idx][idx] != ref_char:
                    all_same = False
                    break
            if all_same:
                all_same_indices.append(idx)

        for seq_idx in range(len(ret_sequences)):
            seq_list = list(ret_sequences[seq_idx])
            for idx in all_same_indices[::-1]:
                seq_list[idx] = ""
            ret_sequences[seq_idx] = "".join(seq_list)

        return ret_sequences

    def action_to_indices(self, actions):
        indices = []
        for a in actions:
            tree_pairs = self.tree_pairs_dict[self.states[0].num_trees]
            i, j = tree_pairs[a]
            indices.append((i, j))

        return torch.Tensor(indices).long()


    def construct_new_tree(self, state, tree_pair_action, edge_pair_action, idx):
        state = self.states[idx]
        assert not state.is_done

        # tree_pair_action = actions[idx]
        # edge_pair_action = edge_actions[idx]
        l_length, r_length = edge_pair_action
        # l_length, r_length = 0.12345, 0.12345

        tree_pairs = self.tree_pairs_dict[state.num_trees]
        i, j = tree_pairs[tree_pair_action]
        # assert i < j
        # subtrees_indices_action.append((i, j))

        left_tree_data = {"tree": state.subtrees[i], "branch_length": l_length}

        right_tree_data = {"tree": state.subtrees[j], "branch_length": r_length}

        unrooted = len(state.subtrees) == 2

        # count match index of constructed and tree label tree
        if self.mom_maps:
            lmom = self.mom_maps[idx].get(state.subtrees[i].name, "")
            rmom = self.mom_maps[idx].get(state.subtrees[j].name, "")
        else:
            lmom, rmom = "", ""

        if lmom == rmom and lmom != "":
            new_tree_name = lmom
            self.count_match_label_steps[idx] += 1
        else:
            new_tree_name = ""

        new_tree = PhyloTree(
            at_root=unrooted,
            left_tree_data=left_tree_data,
            right_tree_data=right_tree_data,
            name=new_tree_name,
        )
        return new_tree, (i, j), unrooted


    def optimize_branch_length_parallel(self, new_trees):
        """并行优化树"""
        args_list = [
            (
                format_rtree(new_tree, True, None, self.seq_keys[idx]),
                self.seq_keys[idx],
                self.batch_seqs[idx],
                new_tree.seq_indices,
            )
            for idx, new_tree in enumerate(new_trees)
        ]
        with Pool(8) as pool:
            results = pool.map(process_, args_list)

        for idx, new_tree in enumerate(new_trees):
            rtree_op_tuple, utree_op_tuple, utree_op_str, log_score_op = results[idx]
            assign_branch_length_rtree(new_tree, rtree_op_tuple)
            new_tree.rtree_op_tuple = rtree_op_tuple
            new_tree.utree_op_tuple = utree_op_tuple
            new_tree.utree_op_str = utree_op_str
            new_tree.log_score = log_score_op


    def optimize_branch_length_sequential(self, new_trees):
        """顺序优化树"""
        for idx, new_tree in enumerate(new_trees):
            sequence_keys = self.seq_keys[idx]
            sequences = self.batch_seqs[idx]

            rtree_str = format_rtree(new_tree, True, None, sequence_keys)
            rtree = pllpy.treestr_to_tuples(rtree_str)

            tree_msa = {
                "labels": [sequence_keys[seq_ii] for seq_ii in new_tree.seq_indices],
                "sequences": [sequences[seq_ii] for seq_ii in new_tree.seq_indices],
            }

            utree_op_str, _, log_score_op = pllpy.optimize_brlen(
                rtree_str, tree_msa, is_root=True, iters=3, model=evolution_model, opt_model=True
            )
            utree_op_tuple = pllpy.treestr_to_tuples(utree_op_str)
            rtree_op_tuple = pllpy.utree2rtree_guided(utree_op_tuple, rtree)

            assign_branch_length_rtree(new_tree, rtree_op_tuple)
            new_tree.rtree_op_tuple = rtree_op_tuple
            new_tree.utree_op_tuple = utree_op_tuple
            new_tree.utree_op_str = utree_op_str
            new_tree.log_score = log_score_op

    def optimize_branch_length_no_br(self, new_trees):
        for idx, new_tree in enumerate(new_trees):
            sequence_keys = self.seq_keys[idx]
            # sequences = self.batch_seqs[idx]

            rtree_str = format_rtree(new_tree, True, None, sequence_keys)
            rtree = pllpy.treestr_to_tuples(rtree_str)

            assign_branch_length_rtree(new_tree, rtree)
            new_tree.rtree_op_tuple = rtree
            new_tree.utree_op_tuple = rtree
            new_tree.utree_op_str = rtree_str
            new_tree.log_score = -111111

    def step(
        self,
        actions,
        edge_actions,
        parallel=True,
        branch_optimize=True,
        agent=None,
        step_action=False,
    ):
        batch_size = len(actions)
        subtrees_indices_action = []
        new_trees_constructed = []

        for idx in range(batch_size):
            state = self.states[idx]
            tree_pair_action = actions[idx]
            edge_pair_action = edge_actions[idx]
            new_tree, (i, j), unrooted = self.construct_new_tree(state, tree_pair_action, edge_pair_action, idx)
            new_trees_constructed.append(new_tree)
            subtrees_indices_action.append((i, j))


        if unrooted and branch_optimize:
            if parallel:
                self.optimize_branch_length_parallel(new_trees_constructed)
            else:
                self.optimize_branch_length_sequential(new_trees_constructed)
        elif unrooted:
            self.optimize_branch_length_no_br(new_trees_constructed)


        new_states = []
        log_rewards = []
        batch_action_set_step = []

        for idx in range(batch_size):

            new_tree = new_trees_constructed[idx]

            if unrooted:
                unrooted_tree = new_tree.to_unrooted_tree()
                unrooted_tree.rtree_op_tuple = new_tree.rtree_op_tuple
                unrooted_tree.utree_op_tuple = new_tree.utree_op_tuple
                unrooted_tree.utree_op_str = new_tree.utree_op_str
                new_state = PhylogeneticTreeState([unrooted_tree])
            else:

                i, j = subtrees_indices_action[idx]
                state_trees_list = self.states[idx].subtrees
                state_trees_list[i] = new_tree
                state_trees_list.pop(j)

                new_state = PhylogeneticTreeState(state_trees_list)

                # zxr add
                subtrees = new_state.subtrees
                if self.label_trees is not None and step_action:
                    action_set_step = self.get_step_action(self.mom_maps[idx], subtrees)
                else:
                    action_set_step = None
                # zxr add
                batch_action_set_step.append(action_set_step)

            new_states.append(new_state)
            log_rewards.append(new_tree.log_score)

            done = new_state.is_done

        self.states = new_states
        # zxr add
        self.batch_action_set_step = batch_action_set_step

        if not done:
            subtree_num = self.state_tensor.size(1)

            # prepare base indices
            base_indices = [list(range(subtree_num)) for _ in range(batch_size)]
            for idx in range(batch_size):
                _ii, _jj = subtrees_indices_action[idx]
                base_indices[idx][_ii] = subtree_num
                base_indices[idx].pop(_jj)
            base_indices = torch.Tensor(base_indices).long()
            base_indices = base_indices.to(self.device)

            # prepare representation
            subtrees_indices_action = torch.Tensor(subtrees_indices_action).long()
            subtrees_indices_action = subtrees_indices_action.to(self.device)

            ii = subtrees_indices_action[:, 0]
            jj = subtrees_indices_action[:, 1]

            ij_indices = (ii, jj)

            if len(self.state_tensor.shape) == 4:
                ii = (
                    ii.unsqueeze(1)
                    .unsqueeze(2)
                    .unsqueeze(3)
                    .expand(
                        -1, -1, self.state_tensor.size(2), self.state_tensor.size(3)
                    )
                )
                jj = (
                    jj.unsqueeze(1)
                    .unsqueeze(2)
                    .unsqueeze(3)
                    .expand(
                        -1, -1, self.state_tensor.size(2), self.state_tensor.size(3)
                    )
                )
                # index to get new representation
                expanded_indices = (
                    base_indices.unsqueeze(2)
                    .unsqueeze(3)
                    .expand(
                        -1, -1, self.state_tensor.size(2), self.state_tensor.size(3)
                    )
                )
            elif len(self.state_tensor.shape) == 3:
                ii = (
                    ii.unsqueeze(1)
                    .unsqueeze(2)
                    .expand(-1, -1, self.state_tensor.size(2))
                )
                jj = (
                    jj.unsqueeze(1)
                    .unsqueeze(2)
                    .expand(-1, -1, self.state_tensor.size(2))
                )
                # index to get new representation
                expanded_indices = base_indices.unsqueeze(2).expand(
                    -1, -1, self.state_tensor.size(2)
                )

            subtree_i = torch.gather(self.state_tensor, 1, ii)
            subtree_j = torch.gather(self.state_tensor, 1, jj)


            if agent is None:
                new_subtree = (subtree_i + subtree_j) / 2
            else:
                new_subtree = agent.aggregate(
                    subtree_i, subtree_j, ij_indices, batchwise_ij_indices=True
                )

            new_state = torch.cat((self.state_tensor, new_subtree), dim=1)

            self.state_tensor = torch.gather(new_state, 1, expanded_indices)

        else:
            base_indices = None

        return done

    def dump_end_trees(
        self,
    ):
        trees = [x.subtrees[0] for x in self.states]
        scores = [x.log_score for x in trees]
        return trees, scores

    def evaluate_loglikelihood(self, get_all_tree=False):
        scores = []
        for state in self.states:
            assert state.is_done

            scores.append(state.log_score)

        best_idx = scores.index(max(scores))
        # import pdb; pdb.set_trace()

        if get_all_tree:
            rtree_op_tuple_list = [x.subtrees[0].rtree_op_tuple for x in self.states]
            utree_op_tuple_list = [x.subtrees[0].utree_op_tuple for x in self.states]
            utree_op_str_list = [x.subtrees[0].utree_op_str for x in self.states]
            return (
                torch.from_numpy(np.array(scores)).to(self.device),
                rtree_op_tuple_list,
                utree_op_tuple_list,
                utree_op_str_list,
            )

        else:
            try:
                return (
                    torch.from_numpy(np.array(scores)).to(self.device),
                    self.states[best_idx].subtrees[0].rtree_op_tuple,
                    self.states[best_idx].subtrees[0].utree_op_tuple,
                    self.states[best_idx].subtrees[0].utree_op_str,
                )
            except:
                import pdb; pdb.set_trace()

    def _seq2array(self, seq):
        seq = [self.chars_dict[x] for x in seq]
        data = np.array(seq)
        return data

    def get_current_trees(self):
        current_trees = []
        for state in self.states:
            subtrees = state.subtrees
            
            tree_strings = []
            for tree in subtrees:
                tree_str = format_rtree_topology(tree, at_root=True, sequence_keys=None)
                tree_strings.append(tree_str)
                
            current_trees.append(tree_strings)
        
        return current_trees

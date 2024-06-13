import argparse
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_value_
from torch.distributions import Categorical
from torch.distributions.normal import Normal
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter

from environment import PhyInferEnv
from environment import compute_raw_tree_log_score
import environment
import dendropy
import sys
import shutil
# from model import PhyloTreeModelOneStep as PGPI
# from model import PhyloFormer as PGPI
# from model_mlp_baseline import PhyloMLP as PGPI
# from model_attn_baseline import PhyloATTN as PGPI
# from model_attn_global_local_baseline import PhyloATTN as PGPI
# from model_attn_highway_baseline import PhyloATTN as PGPI
# from model_attn_xjforget_baseline import PhyloATTN as PGPI
# from model_attn_xrestforget_baseline import PhyloATTN as PGPI
# from model_attn_xrestforget_baseline_zxr import PhyloATTN as PGPI
# from model_attn_FUSExrestAttn_baseline import PhyloATTN as PGPI
# from models.model_attn_FUSExrestAttnNorm_baseline import PhyloATTN as PGPI
from model import PhyloATTN as PGPI
import utils
import copy
from ete3 import Tree
from tqdm import tqdm
import time
import einops
import raxmlpy

from phydata import PhyDataset, PhySampler, custom_collate_fn, load_pi_instance, load_tree_file, load_phy_file_multirow, load_phy_file
from phydata_infer import PhyDataset_infer, custom_infer_collate_fn

torch.autograd.set_detect_anomaly(True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


EXPLORE_TEMPERTURE = lambda step: 1.0

# EXPLORE_TEMPERTURE = lambda step: (2 - 1) / (32 - 0) * (step - 0) + 1
STOP_TIME = 100


def prepare_action_sets(batch_action_set, actions, nb_seq, env, step, device):
    actions_set_cur_step = [x[0][step] for x in batch_action_set]
    actions_set_cur_step = [[env.action_indices_dict[nb_seq][(ai, aj)] for ai, aj in actions] for actions in actions_set_cur_step]
    actions_set_list_cur_step = copy.deepcopy(actions_set_cur_step)
    actions_complement_set_cur_step = [[v for a, v in env.action_indices_dict[nb_seq].items() if v not in actions] for _b, actions in enumerate(actions_set_cur_step)]
    actions_set_cur_step, actions_set_cur_step_mask = utils.pad_array_mask(actions_set_cur_step)
    actions_complement_set_cur_step, actions_complement_set_cur_step_mask = utils.pad_array_mask(actions_complement_set_cur_step)
    actions_set_cur_step, actions_set_cur_step_mask = actions_set_cur_step.to(device), actions_set_cur_step_mask.to(device)
    actions_complement_set_cur_step, actions_complement_set_cur_step_mask = actions_complement_set_cur_step.to(device), actions_complement_set_cur_step_mask.to(device)
    return actions_set_cur_step, actions_complement_set_cur_step, actions_set_cur_step_mask, actions_complement_set_cur_step_mask, actions_set_list_cur_step



def compute_likelihood(phy_path, tre_path):
    try:
        seqs, seq_keys, num_sequences, sequence_length = load_phy_file(phy_path)
    except:
        seqs, seq_keys, num_sequences, sequence_length = load_phy_file_multirow(phy_path)

    with open(tre_path, "r") as file:
        tree_str = file.readline().strip()

    msa = {
        "labels": seq_keys,
        "sequences": seqs
    }

    logll = raxmlpy.compute_llh(tree_str, msa, is_root=False)
    return logll, tree_str


def optimize_branch_length(phy_path, tre_path, iters=3):
    try:
        seqs, seq_keys, num_sequences, sequence_length = load_phy_file(phy_path)
    except:
        seqs, seq_keys, num_sequences, sequence_length = load_phy_file_multirow(phy_path)

    with open(tre_path, "r") as file:
        tree_str = file.readline().strip()

    # import pdb; pdb.set_trace()

    # # cut
    # for iii in range(seqs):
    #     seqs[iii] = seq[iii][:256]

    msa = {
        "labels": seq_keys,
        "sequences": seqs
    }

    tree_op, logllint, logllop = raxmlpy.optimize_brlen(tree_str, msa, is_root=False, iters=iters)
    return logllop, tree_str

# def calculate_rf_distance(newick1, newick2):
#     taxa = dendropy.TaxonNamespace()
    
#     # import pdb; pdb.set_trace()
#     tree1 = dendropy.Tree.get(data=newick1, schema="newick", taxon_namespace=taxa)
#     tree2 = dendropy.Tree.get(data=newick2, schema="newick", taxon_namespace=taxa)
    
#     # 计算并返回RF距离
#     rf_distance = dendropy.calculate.treecompare.symmetric_difference(tree1, tree2)

#     # 获取两棵树的总分支数
#     total_branches_tree1 = len(list(tree1.preorder_edge_iter()))
#     total_branches_tree2 = len(list(tree2.preorder_edge_iter()))
#     total_branches = total_branches_tree1 + total_branches_tree2
    
#     # 计算相对的 RF 距离
#     relative_rf_distance = rf_distance / total_branches
#     return rf_distance, relative_rf_distance

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


def reinforce_rollout(batch, agent, env, cfgs, replay_buffer=None, eval=False, argmax=False, get_all_tree=False):

    if not eval:
        replay_buffer_sample_size = min(cfgs.replay_buffer_sample_size, replay_buffer.get_size())
        if replay_buffer_sample_size < 8:
            replay_buffer_sample_size = 2
        replay_actions_trajectories = replay_buffer.sample(replay_buffer_sample_size)

    batch_seqs, batch_seq_keys, batch_array = batch['seqs'], batch['seq_keys'], batch['data']

    batch_weights = batch['seq_weights']

    batch_array = batch_array.to(device)

    batch_weights = batch_weights.to(device)
    batch_seq_mask = batch_weights == 0
    
    batch_array_one_hot = batch_array #torch.zeros((*batch_array.shape, 4), device=device)


    # # # cut 
    # batch_array_one_hot = batch_array_one_hot[:,:,:256,:]
    # batch_seq_mask = batch_seq_mask[:, :256]

    # for x in batch_seqs:
    #     for iii in range(len(x)):
    #         x[iii] = x[iii][:256]
    # # import pdb; pdb.set_trace()

    env.init_states(batch_seqs, batch_seq_keys, batch_array_one_hot)

    selected_log_ps = []
    # logitss = []
    log_ps = []

    step = 0

    actions_ij_prev = None
    logits_prev = None

    while True:
        if step == 0:

            if eval:
                agent.eval()
                with torch.no_grad():
                    env.state_tensor = agent.encode_zxr(env.init_state_tensor)
            else:
                env.state_tensor = agent.encode_zxr(env.init_state_tensor)

        batch_size, nb_seq = env.state_tensor.shape[:2]

        if eval:
            agent.eval()

            with torch.no_grad():

                if actions_ij_prev is not None:
                    score_indices_to_prev = utils.get_score_indices_to_prev(actions_ij_prev, env, nb_seq, batch_size)
                    score_indices_to_prev = torch.from_numpy(np.array(score_indices_to_prev)).to(device)
                else:
                    score_indices_to_prev = None

                ret = agent.decode_zxr(env.state_tensor, batch_seq_mask, (actions_ij_prev, score_indices_to_prev, logits_prev))
        else:

            if actions_ij_prev is not None:
                score_indices_to_prev = utils.get_score_indices_to_prev(actions_ij_prev, env, nb_seq, batch_size)
                score_indices_to_prev = torch.from_numpy(np.array(score_indices_to_prev)).to(device)
            else:
                score_indices_to_prev = None

            ret = agent.decode_zxr(env.state_tensor, batch_seq_mask, (actions_ij_prev, score_indices_to_prev, logits_prev))


        logits = ret['logits']

        # EXPLORE_TEMPERTURE = 0.5 if step < 90 else 1.5

        log_p = torch.log_softmax(logits / EXPLORE_TEMPERTURE(step), dim=-1)
        # actions = Categorical(logits=log_p).sample()

        if eval:
            if argmax:
                actions = torch.argmax(logits, dim=1)
            else:
                actions = Categorical(logits=logits / EXPLORE_TEMPERTURE(step)).sample()

            # TODO
            # Beam Search ???

        else:
            actions = Categorical(logits=logits / EXPLORE_TEMPERTURE(step)).sample()

            if replay_actions_trajectories:
                replay_actions_cur_step = [x[step] for x in replay_actions_trajectories]
                replay_actions_cur_step = [env.action_indices_dict[nb_seq][(ai,aj)] for ai,aj in replay_actions_cur_step]
                replay_actions_cur_step = torch.from_numpy(np.array(replay_actions_cur_step)).to(actions.device)
                actions[-replay_actions_cur_step.size(0):] = replay_actions_cur_step


        actions_ij_prev = [env.tree_pairs_dict[nb_seq][a.item()] for a in actions]
        actions_ij_prev = torch.from_numpy(np.array(actions_ij_prev)).to(device).to(torch.int32)

        edge_actions = [(None, None) for _ in range(batch_size)]

        done = env.step(actions, edge_actions, branch_optimize=True, agent=agent)

        if done:
            break

        step += 1

        selected_log_p = torch.gather(log_p, 1, actions.unsqueeze(1))
        selected_log_ps.append(selected_log_p)
        log_ps.append(log_p)
    
        # logitss.append(logits)
        logits_prev = logits

    selected_log_ps = torch.cat(selected_log_ps, dim=1)

    if get_all_tree:
        scores, best_rtree_tuple, best_tree_tupe, best_tree = env.evaluate_loglikelihood(get_all_tree=True)
    else:
        scores, best_rtree_tuple, best_tree_tupe, best_tree = env.evaluate_loglikelihood()
    # import pdb; pdb.set_trace()
    # replace ")," with "):" , and "'," with ":", and remove all ',' in best_tree
    # best_tree = best_tree.replace("),", "):").replace("',", ":").replace(",", "")

    # import pdb; pdb.set_trace()

    if not eval:
        assert replay_buffer is not None
        _trees, _scores = env.dump_end_trees()
        replay_buffer.add(_trees, _scores)


    return selected_log_ps, log_ps, scores, best_tree


def RL_Search(cfgs, MSA_file, c_best_tree_file=None, raw_tree_file=None):
    env = PhyInferEnv(cfgs, device)

    # replay_buffer = utils.ReplayBuffer(cfgs.replay_buffer_size)

    trajectory_num = 1 #cfgs.ENV.TRAJECTORY_NUM
    # val_path = "/home/dingshizh/PGPI/data_gen/CSolver/fixed_len_data_n_30_100_test"

    # dataset_val = PhyDataset(val_path, trajectory_num=trajectory_num, device=device, num_per_file=16, taxa_list=[100], C_solved=True)
    batch_size = cfgs.env.batch_size
    # data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, collate_fn=custom_collate_fn, num_workers=2)

    print("Data loaded")

    if cfgs.reload_checkpoint_path:
        PGPI_pretrained = PGPI(cfgs).to(device)
        checkpoint = torch.load(cfgs.reload_checkpoint_path)
        PGPI_pretrained.load_state_dict(checkpoint['model_state_dict'], strict=False)

        accumulated_steps = checkpoint['accumulated_steps']
        epoch_now = checkpoint['epoch']
        print(f"Reloaded checkpoint at epoch {epoch_now}, accumulated_steps {accumulated_steps}")
    else:
        PGPI_pretrained = PGPI(cfgs).to(device)
        accumulated_steps = 0
        epoch_now = 0

    accumulated_steps = 0
    epoch_now = 0

    policy_network = PGPI_pretrained

    the_best_tree = None
    the_best_score = -np.inf


    MSA_file = MSA_file


    batch = load_pi_instance(
        MSA_file
    )

    taxa_num_list = batch["taxa_nums"]
    seq_len_list = batch["seq_lens"]

    def expand_one_instance(x, L):
        if isinstance(x, list):
            return x * L
        elif isinstance(x, torch.Tensor):
            return x.expand(L, *x.shape[1:])
        else:
            assert False

    fintune_instance_id = 0
    
    # batch size estimate
    # batch_size = 16 / ( (taxa_num_list[0]/50)^2 * (seq_len_list[0]/1024))
    batch_size = int(48 / ((taxa_num_list[0]/50)**2 * (seq_len_list[0]/1024)))
    print(f"batch_size : {batch_size} for taxa {taxa_num_list[0]} seq len {seq_len_list[0]}")

    batch = {k:expand_one_instance(batch[k][fintune_instance_id:fintune_instance_id+1], batch_size) for k in batch.keys()}

    # logitss, actions_sets_list, actions_set_masks, actions_sets, actions_set_complement_masks, actions_sets_complement, selected_log_ps, scores, best_tree = supervise_rollout(batch, PGPI_pretrained, env, eval=True, pretrained=True, branch_optimize=True)

    print(batch["data"][0].shape)
    print(batch["data"][0].shape)
    print(batch["data"][0].shape)


    if raw_tree_file:
        raw_tree_score, raw_tree_str = optimize_branch_length(
            MSA_file,
            raw_tree_file
        )
    else:
        raw_tree_score = None

    if c_best_tree_file:
        c_best_tree_score, c_best_tree_str = optimize_branch_length(
            MSA_file,
            c_best_tree_file
        )
    else:
        c_best_tree_score = None

    start_time = time.time()
    time_when_best_score_max = None
    first_time_eq_c_best = True
    times_when_condition_met = None
    first_time_eq_c_score = None


    for epoch in range(epoch_now+1, cfgs.num_epoch):

        the_best_tree_update = False

        print(f"Epoch {epoch + 1}/{cfgs.num_epoch}")



        # progress_bar = tqdm(range(cfgs.num_episodes), total=cfgs.num_episodes)

        for episode in range(cfgs.num_episodes):

            _, _, scores, best_tree = reinforce_rollout(batch, policy_network, env, cfgs, eval=True)


            if the_best_tree is None:
                the_best_tree = best_tree
            else:
                best_tree_log_score = max(scores).detach().cpu().item()
                if best_tree_log_score > the_best_score:
                    the_best_tree = best_tree
                    the_best_score = best_tree_log_score
                    the_best_tree_update = True
                    time_when_best_score_max = time.time() - start_time
        
            if c_best_tree_score - the_best_score < 2 and first_time_eq_c_best is True:
                rf_distance, relative_rf_distance = calculate_rf_distance(c_best_tree_str, the_best_tree)
                if rf_distance == 0:
                    times_when_condition_met = time.time() - start_time
                    first_time_eq_c_best = False
                    first_time_eq_c_score = the_best_score
                    # print info
                    print(f" Epoch {epoch + 1}/{cfgs.num_epoch}, Episode {episode + 1}/{cfgs.num_episodes}, Best Tree Score: {the_best_score}, RF Distance: {relative_rf_distance}, Time: {time.time() - start_time}")
            
            # print(the_best_tree)
        # 检查是否超过一分钟
        if time.time() - start_time > STOP_TIME:
            rf_distance, relative_rf_distance = calculate_rf_distance(c_best_tree_str, the_best_tree)
            break


    return the_best_tree, the_best_score, raw_tree_score, c_best_tree_score, time_when_best_score_max, times_when_condition_met, first_time_eq_c_score, relative_rf_distance, taxa_num_list, seq_len_list



def RL_argmax(instance_dir, cfgs, write_dir, write_file_name, csv_file_list=None):
    env = PhyInferEnv(cfgs, device)
    dataset_val = PhyDataset_infer(instance_dir,device=device, load_Csolver=True, pos=2, csv_file=csv_file_list)
    batch_size = 1 #cfgs.env.batch_size
    data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, collate_fn=custom_infer_collate_fn, num_workers=0)
    # import pdb; pdb.set_trace()

    print("Data loaded")

    if cfgs.reload_checkpoint_path:
        PGPI_pretrained = PGPI(cfgs).to(device)
        checkpoint = torch.load(cfgs.reload_checkpoint_path)
        PGPI_pretrained.load_state_dict(checkpoint['model_state_dict'], strict=False)

        accumulated_steps = checkpoint['accumulated_steps']
        epoch_now = checkpoint['epoch']
        print(f"Reloaded checkpoint at epoch {epoch_now}, accumulated_steps {accumulated_steps}")
    else:
        PGPI_pretrained = PGPI(cfgs).to(device)
        accumulated_steps = 0
        epoch_now = 0

    accumulated_steps = 0
    epoch_now = 0

    policy_network = PGPI_pretrained

    the_best_tree = None
    the_best_score = -np.inf

    import pandas as pd
    results_df = pd.DataFrame(columns=["File", "Time(avg)", "score", "C Best Tree Score", "RF distance between the best tree and the C best tree", "Taxa number", "Seq Length", "Tree"])
    for batch in data_loader_val:
        start_time = time.time()
        file_list = batch["file_paths"]
        c_best_tree_str_list = batch["tree_strs"]
        c_best_tree_score_list = batch["C_logllops"]
        taxa_num_list = batch["taxa_nums"]
        seq_len_list = batch["seq_lens"]


        _, _, scores, best_tree = reinforce_rollout(batch, policy_network, env, cfgs, eval=True, argmax=True, get_all_tree=True)

        # get RF distance between the best tree and the C best tree
        rf_distance_list = []
        for c_best_tree_str, best_tree_str in zip(c_best_tree_str_list, best_tree):
            # import pdb; pdb.set_trace()
            rf_distance, relative_rf_distance = calculate_rf_distance(c_best_tree_str, best_tree_str)
            rf_distance_list.append(relative_rf_distance)
        

        # score tensor to list
        scores = [score.detach().cpu().item() for score in scores]
        for file, score, c_best_tree_score, rf_distance, tree, taxa_num, seq_len in zip(file_list, scores, c_best_tree_score_list, rf_distance_list, best_tree, taxa_num_list, seq_len_list):
            new_row_df = pd.DataFrame({
                "File": [file],
                "Time(avg)": [(time.time() - start_time)/len(batch["file_paths"])],
                "score": [score],
                "C Best Tree Score": [c_best_tree_score],
                "RF distance between the best tree and the C best tree": [rf_distance],
                "Taxa number": [taxa_num],
                "Seq Length": [seq_len],
                "Tree": [tree]
            })
            results_df = pd.concat([results_df, new_row_df], ignore_index=True)
            print(f"File: {file}, The Best Score: {score}, RF distance between the best tree and the C best tree: {rf_distance}")
        # Save the DataFrame to a CSV file
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)
    results_df.to_csv(os.path.join(write_dir, write_file_name), index=False)
    return results_df


def RL_finetuning(cfgs, MSA_file, c_best_tree_file=None, raw_tree_file=None):
    env = PhyInferEnv(cfgs, device)
    trajectory_num = 1 #cfgs.ENV.TRAJECTORY_NUM
    batch_size = cfgs.env.batch_size

    print("Data loaded")

    if cfgs.reload_checkpoint_path:
        PGPI_pretrained = PGPI(cfgs).to(device)
        checkpoint = torch.load(cfgs.reload_checkpoint_path)
        PGPI_pretrained.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer = optim.Adam(PGPI_pretrained.parameters(), lr=cfgs.lr)
        accumulated_steps = checkpoint['accumulated_steps']
        epoch_now = checkpoint['epoch']
        print(f"Reloaded checkpoint at epoch {epoch_now}, accumulated_steps {accumulated_steps}")
    else:
        PGPI_pretrained = PGPI(cfgs).to(device)
        optimizer = optim.Adam(PGPI_pretrained.parameters(), lr=cfgs.lr)
        accumulated_steps = 0
        epoch_now = 0

    accumulated_steps = 0
    epoch_now = 0

    policy_network = PGPI_pretrained

    baseline_val = -np.inf
    baseline_val_batch = []

    the_best_tree = None
    the_best_score = -np.inf

    MSA_file = MSA_file
    batch = load_pi_instance(
        MSA_file
    )
    taxa_num_list = batch["taxa_nums"]
    seq_len_list = batch["seq_lens"]

    def expand_one_instance(x, L):
        if isinstance(x, list):
            return x * L
        elif isinstance(x, torch.Tensor):
            return x.expand(L, *x.shape[1:])
        else:
            assert False

    fintune_instance_id = 0

    # batch size estimate
    # batch_size = 16 / ( (taxa_num_list[0]/50)^2 * (seq_len_list[0] /1024))
    batch_size = int(24 / ((taxa_num_list[0]/50)**2 * (seq_len_list[0]/1024)))
    print(f"batch_size : {batch_size} for taxa {taxa_num_list[0]} seq len {seq_len_list[0]}")

    cfgs.replay_buffer_size = batch_size
    cfgs.replay_buffer_sample_size = int(batch_size/2)
    replay_buffer = utils.ReplayBuffer(cfgs.replay_buffer_size)

    batch = {k:expand_one_instance(batch[k][fintune_instance_id:fintune_instance_id+1], batch_size) for k in batch.keys()}

    print(batch["data"][0].shape)
    print(batch["data"][0].shape)
    print(batch["data"][0].shape)


    if raw_tree_file:
        raw_tree_score, raw_tree_str = optimize_branch_length(
            MSA_file,
            raw_tree_file
        )
    else:
        raw_tree_score = None

    if c_best_tree_file:
        c_best_tree_score, c_best_tree_str = optimize_branch_length(
            MSA_file,
            c_best_tree_file
        )
    else:
        c_best_tree_score = None

    start_time = time.time()
    time_when_best_score_max = None
    first_time_eq_c_best = True
    times_when_condition_met = None
    first_time_eq_c_score = None
    first_20s = True 
    first_50s = True


    for epoch in range(epoch_now+1, cfgs.num_epoch):
        optimizer.zero_grad()

        the_best_tree_update = False

        # if epoch == 0:
        if False:
            baseline = "RANDOM"
            cfgs.num_episodes_baseline = 50
        else:
            baseline = policy_network
            cfgs.num_episodes_baseline = 1

        if epoch == 1:
            baseline_scores_collect = []
            for episode in range(cfgs.num_episodes_baseline):
                _, _, baseline_scores, _ = reinforce_rollout(batch, baseline, env, cfgs, eval=True)
                baseline_scores_collect.append(baseline_scores)
            baseline_scores_collect = torch.cat(baseline_scores_collect, dim=0)
            baseline_val = max(baseline_scores_collect.min(), baseline_val)
        else:
            new_baseline_val = sum(baseline_val_batch) / len(baseline_val_batch)
            baseline_val = max(baseline_val, new_baseline_val)
            # baseline_val = np.min(replay_buffer.scores)
            baseline_val_batch = []


        print(f"Epoch {epoch + 1}/{cfgs.num_epoch}")

        # progress_bar = tqdm(range(cfgs.num_episodes), total=cfgs.num_episodes)

        for episode in range(cfgs.num_episodes):
            # print(f"episode {episode + 1}/{cfgs.num_episodes}")
            # _, _, scores, best_tree = reinforce_rollout(batch, policy_network, env, cfgs, eval=True)
            selected_log_ps, log_ps, scores, best_tree = reinforce_rollout(batch, policy_network, env, cfgs, replay_buffer)
            selected_log_p = selected_log_ps.sum(dim=1)

            # try policy loss
            policy_loss = (
                - (selected_log_p) * (scores - baseline_val)
            ).mean()

            entropy_reg_loss = - sum(
                [
                    - torch.sum(torch.exp(log_p) * log_p, dim=1).mean()
                    for log_p in log_ps
                ]
            )
            baseline_val_batch.append(scores.mean().item())

            loss = policy_loss + entropy_reg_loss * cfgs.entropy_reg_strength

            loss.backward()

            if episode == cfgs.num_episodes - 1:
                clip_grad_value_(policy_network.parameters(), clip_value=cfgs.clip_value)
                optimizer.step()


            baseline_val_batch.append(scores.mean().detach().item())

            if the_best_tree is None:
                the_best_tree = best_tree
            else:
                best_tree_log_score = max(scores).detach().cpu().item()
                if best_tree_log_score > the_best_score:
                    the_best_tree = best_tree
                    the_best_score = best_tree_log_score
                    the_best_tree_update = True
                    time_when_best_score_max = time.time() - start_time

            if c_best_tree_score - the_best_score < 1 and first_time_eq_c_best is True:
                rf_distance, relative_rf_distance = calculate_rf_distance(c_best_tree_str, the_best_tree)
                if rf_distance == 0:
                    times_when_condition_met = time.time() - start_time
                    first_time_eq_c_best = False
                    first_time_eq_c_score = the_best_score
                    # print info
                    print(f" Epoch {epoch + 1}/{cfgs.num_epoch}, Episode {episode + 1}/{cfgs.num_episodes}, Best Tree Score: {the_best_score}, RF Distance: {relative_rf_distance}, Time: {time.time() - start_time}")



            if time.time() - start_time > 20 and first_20s is True:
                the_best_score_20s = the_best_score
                first_20s = False

            if time.time() - start_time > 50 and first_50s is True:
                the_best_score_50s = the_best_score
                first_50s = False

        if time.time() - start_time > STOP_TIME:
            rf_distance, relative_rf_distance = calculate_rf_distance(c_best_tree_str, the_best_tree)
            break

    return the_best_tree, the_best_score, raw_tree_score, c_best_tree_score, time_when_best_score_max, times_when_condition_met, first_time_eq_c_score, relative_rf_distance, the_best_score_20s, the_best_score_50s, taxa_num_list, seq_len_list


def Argmax_inference(test_file_path, write_dir,write_file_name,csv_file_list=None):
    # test_file_name = test_file_path.split('/')[-3:]
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)
    # write_file_name = f"argmax_bs{cfgs.env.batch_size}_temp1_train100_len256_dim{cfgs.model.embed_dim}_patch{cfgs.model.patch_size}_{test_file_name[0]}_{test_file_name[1]}_{test_file_name[2]}.csv"
    RL_argmax(test_file_path, cfgs, write_dir, write_file_name,csv_file_list=csv_file_list)  


def Search_inference(test_file_path, write_dir,write_file_name):
    # ***************************************Search***************************************
    #list all files in the directory and filter out the files with the extension .phy
    # test_file_name = test_file_path.split('/')[-3:]
    files = os.listdir(test_file_path)
    files = [file for file in files if file.endswith(".phy")]
    # 0-31, 32-63, 64-95, 96-127
    # low_num = 0
    # high_num = 128
    # write_dir = f"/home/dingshizhe/iclr_2024_phylogfn_suppl/PhyloNet/search/"
    # write_file_name = f"onlysearch{STOP_TIME}_bs{cfgs.env.batch_size}_temp1_train100_len256_dim{cfgs.model.embed_dim}_patch{cfgs.model.patch_size}_{test_file_name[0]}_{test_file_name[1]}_{test_file_name[2]}.csv"
    # files = [file for file in files if int(file[:-4].split('_')[-1]) <= high_num and int(file[:-4].split('_')[-1]) >= low_num]
    # files = [file for file in files]
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)
    print(len(files))
    print(len(files))
    print(write_dir)
    print(write_file_name)
    import pandas as pd
    # Initialize a DataFrame to hold all the results
    results_df = pd.DataFrame(columns=["File", "Time When Best Score Max", "Times When Condition Met",
                                    "The Best Score", "C Best Tree Score", "RF distance", "First Time Eq C Score", "Taxa number", "Seq Length", "The Best Tree"])

    for file in files:
        MSA_file = os.path.join(test_file_path, file)
        c_best_tree_file = os.path.join(cfgs.instance_path, file[:-4] + ".tre")
        # raw_tree_file = os.path.join(cfgs.instance_path, file[:-4] + "_raw.tre")
        if all(os.path.exists(f) for f in [MSA_file, c_best_tree_file]):
            # Call the placeholder RL_Search function
            the_best_tree, the_best_score, _, c_best_tree_score, time_when_best_score_max, times_when_condition_met, first_time_eq_c_score, rf_distance, taxa_num_list, seq_len_list = RL_Search(cfgs, MSA_file, c_best_tree_file)
            # Append the results to the DataFrame
            new_row_df = pd.DataFrame({
                "File": [file],
                "The Best Score": [the_best_score],
                "Time When Best Score Max": [time_when_best_score_max],
                "Times When Condition Met": [str(times_when_condition_met)],  # Converting list to string for storage
                "First Time Eq C Score": [first_time_eq_c_score],
                # "Raw Tree Score": [raw_tree_score],
                "C Best Tree Score": [c_best_tree_score],
                "The Best Tree": [the_best_tree],
                "RF distance": [rf_distance],
                "Taxa number": [taxa_num_list],
                "Seq Length": [seq_len_list],
            })
            results_df = pd.concat([results_df, new_row_df], ignore_index=True)
            print(f"File: {file}, The Best Score: {the_best_score}, Time When Best Score Max: {time_when_best_score_max}, Times When Condition Met: {times_when_condition_met}, RF distance: {rf_distance}, C Best Tree Score: {c_best_tree_score}, First Time Eq C Score: {first_time_eq_c_score}")
            new_row_df.to_csv(os.path.join(write_dir,  file[:-4]+write_file_name), index=False)


    # Save the DataFrame to a CSV file
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)
    results_df.to_csv(os.path.join(write_dir, write_file_name), index=False)



def finetune_inference(test_file_path, write_fir, write_file_name):
     # ***************************************Search***************************************
    files = os.listdir(test_file_path)
    files = [file for file in files if file.endswith(".phy")]
    print(len(files))
    print(len(files))
    print(write_dir)
    print(write_file_name)
    import pandas as pd
    # Initialize a DataFrame to hold all the results
    results_df = pd.DataFrame(columns=["File", "Time When Best Score Max", "Times When Condition Met",
                                    "The Best Score", "C Best Tree Score", "RF distance", "Raw Tree Score", "First Time Eq C Score", "Taxa number", "Seq Length","The Best Tree", "20s score", "50s score"])
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)
    for file in files:
        MSA_file = os.path.join(test_file_path, file)
        c_best_tree_file = os.path.join(cfgs.instance_path, file[:-4] + ".tre")
        raw_tree_file = os.path.join(cfgs.instance_path, file[:-4] + ".tre")
        if all(os.path.exists(f) for f in [MSA_file, c_best_tree_file, raw_tree_file]):
            # Call the placeholder RL_Search function
            the_best_tree, the_best_score, raw_tree_score, c_best_tree_score, time_when_best_score_max, times_when_condition_met, first_time_eq_c_score, rf_distance, the_best_score_20s, the_best_score_50s, taxa_num_list, seq_len_list = RL_finetuning(cfgs, MSA_file, c_best_tree_file, raw_tree_file)
            # Append the results to the DataFrame
            new_row_df = pd.DataFrame({
                "File": [file],
                "The Best Score": [the_best_score],
                "Time When Best Score Max": [time_when_best_score_max],
                "Times When Condition Met": [str(times_when_condition_met)],  # Converting list to string for storage
                "First Time Eq C Score": [first_time_eq_c_score],
                "Raw Tree Score": [raw_tree_score],
                "C Best Tree Score": [c_best_tree_score],
                "The Best Tree": [the_best_tree],
                "RF distance": [rf_distance],
                "20s score": [the_best_score_20s],
                "50s score": [the_best_score_50s],
                "Taxa number": [taxa_num_list],
                "Seq Length": [seq_len_list],
            })
            results_df = pd.concat([results_df, new_row_df], ignore_index=True)
            print(f"File: {file}, The Best Score: {the_best_score}, the_best_score_20s: {the_best_score_20s}, the_best_score_50s: {the_best_score_50s} ,Time When Best Score Max: {time_when_best_score_max}, Times When Condition Met: {times_when_condition_met}, RF distance: {rf_distance}, C Best Tree Score: {c_best_tree_score}, Raw Tree Score: {raw_tree_score}, First Time Eq C Score: {first_time_eq_c_score}")
            new_row_df.to_csv(os.path.join(write_dir,  file[:-4]+write_file_name), index=False)
    # Save the DataFrame to a CSV file
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)
    results_df.to_csv(os.path.join(write_dir, write_file_name), index=False)




if __name__ == "__main__":

    # print(sys.argv)
    # config_file_path = sys.argv[2]
    # config_file_name = os.path.basename(config_file_path)
    # config_file_name = config_file_name.split('.')[0]
    config_file_name = "20240509-103334train_realMix"

    parser = argparse.ArgumentParser(description='Policy Gradient Training')
    parser.add_argument('--config_path', type=str, default="", help='Configuration file path')
    args = parser.parse_args()

    cfgs = utils.empty_config()
    cfgs.merge_from_file(args.config_path)

    test_file_dir = cfgs.instance_path
    csv_file_list = None
    # cfgs.infer_opt = "Reinforced"
    # cfgs.infer_opt = "Search" 
    # cfgs.infer_opt = "Argmax"

    # custom_tag = "20240509-103334train_realMix"
    custom_tag = "20240509-103334train_realMix"


    test_file_path = test_file_dir
    test_file_name = test_file_path.split('/')[-3:]
    write_dir =  f"/home/dingshizhe/PhyloNet/search/realdata/{cfgs.infer_opt}_realdata_{custom_tag}_dim{cfgs.model.embed_dim}_patch{cfgs.model.patch_size}/"
    write_file_name = f"{cfgs.infer_opt}_bs{cfgs.env.batch_size}_{test_file_name[0]}_{test_file_name[1]}_{test_file_name[2]}.csv"

    if cfgs.infer_opt == "Argmax":
        # ***************************************Argmax***************************************
        Argmax_inference(test_file_path, write_dir, write_file_name, csv_file_list)

    elif cfgs.infer_opt == "Search":
        Search_inference(test_file_path, write_dir, write_file_name)

    elif cfgs.infer_opt == "Reinforced":
        finetune_inference(test_file_path, write_dir, write_file_name)


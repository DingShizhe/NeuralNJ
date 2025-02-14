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
from model import PhyloATTN as PGPI
import utils
import copy
from ete3 import Tree
from tqdm import tqdm
import time
import einops
import raxmlpy

from phydata import PhyDataset, PhySampler, custom_collate_fn, load_pi_instance, load_tree_file, load_phy_file_multirow, load_phy_file

evolution_model = 'GTR+I+G'
# evolution_model = 'JC'

torch.autograd.set_detect_anomaly(True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


EXPLORE_TEMPERTURE = lambda step: 1.0
# EXPLORE_TEMPERTURE = lambda step: (2 - 1) / (32 - 0) * (step - 0) + 1

# STOP_TIME = 60
# STOP_STEP = 100


def optimize_branch_length(phy_path, tre_path, iters=3):
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

    tree_op, logllint, logllop = raxmlpy.optimize_brlen(tree_str, msa, is_root=False, iters=iters, model=evolution_model, opt_model=False)
    return logllop, tree_str


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
    
    batch_array_one_hot = batch_array 

    env.init_states(batch_seqs, batch_seq_keys, batch_array_one_hot)

    selected_log_ps = []
    log_ps = []

    step = 0

    actions_ij_prev = None
    logits_prev = None

    while True:
        if step == 0:
            if eval:
                agent.eval()
                with torch.no_grad():
                    env.state_tensor = agent.encode_zxr(env.init_state_tensor,batch_seq_mask)
            else:
                env.state_tensor = agent.encode_zxr(env.init_state_tensor,batch_seq_mask)

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
                actions = torch.argmax(logits, dim=-1)
            else:
                actions = Categorical(logits=logits / EXPLORE_TEMPERTURE(step)).sample()

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
    
        logits_prev = logits

    selected_log_ps = torch.cat(selected_log_ps, dim=1)

    if get_all_tree:
        scores, best_rtree_tuple, best_tree_tupe, best_tree = env.evaluate_loglikelihood(get_all_tree=True)
    else:
        scores, best_rtree_tuple, best_tree_tupe, best_tree = env.evaluate_loglikelihood()

    if not eval:
        assert replay_buffer is not None
        _trees, _scores = env.dump_end_trees()
        replay_buffer.add(_trees, _scores)

    return selected_log_ps, log_ps, scores, best_tree


def RL_finetuning(cfgs, MSA_file, c_best_tree_file=None, raw_tree_file=None):
    env = PhyInferEnv(cfgs, device)
    replay_buffer = utils.ReplayBuffer(cfgs.replay_buffer_size)
    trajectory_num = 1 #cfgs.ENV.TRAJECTORY_NUM
    batch_size = 1 #cfgs.env.batch_size

    print("Data loaded")

    if cfgs.reload_checkpoint_path:
        PGPI_pretrained = PGPI(cfgs).to(device)
        checkpoint = torch.load(cfgs.reload_checkpoint_path)
        PGPI_pretrained.load_state_dict(checkpoint['model_state_dict'])
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

    print(f"batch_size : {batch_size} for taxa {taxa_num_list[0]} seq len {seq_len_list[0]}")

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
        raw_tree_str = None

    if c_best_tree_file:
        c_best_tree_score, c_best_tree_str = optimize_branch_length(
            MSA_file,
            c_best_tree_file
        )
    else:
        c_best_tree_score = None
        c_best_tree_str = None
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
            baseline_val_batch = []


        print(f"Epoch {epoch + 1}/{cfgs.num_epoch}")

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

            step_cur = episode+(epoch-1)*cfgs.num_episodes

        # if time.time() - start_time > STOP_TIME: 
        if step_cur >= STOP_STEP:
            if raw_tree_str:
                rf_distance, relative_rf_distance = calculate_rf_distance(raw_tree_str, the_best_tree)
            else:
                rf_distance = None
                relative_rf_distance = None
            break

    return dict(the_best_tree=the_best_tree, the_best_score=the_best_score, raw_tree_score=raw_tree_score, c_best_tree_score=c_best_tree_score, time_when_best_score_max=time_when_best_score_max, relative_rf_distance=relative_rf_distance, step_cur=step_cur)


def RL_Search(cfgs, MSA_file, c_best_tree_file=None, raw_tree_file=None):
    env = PhyInferEnv(cfgs, device)

    trajectory_num = 1 
    batch_size = cfgs.env.batch_size

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

    def expand_one_instance(x, L):
        if isinstance(x, list):
            return x * L
        elif isinstance(x, torch.Tensor):
            return x.expand(L, *x.shape[1:])
        else:
            assert False

    fintune_instance_id = 0

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
        raw_tree_str = None

    if c_best_tree_file:
        c_best_tree_score, c_best_tree_str = optimize_branch_length(
            MSA_file,
            c_best_tree_file
        )
    else:
        c_best_tree_score = None
        c_best_tree_str = None

    start_time = time.time()
    time_when_best_score_max = None
    first_time_eq_c_best = True
    times_when_condition_met = None
    first_time_eq_c_score = None


    for epoch in range(epoch_now+1, cfgs.num_epoch):

        the_best_tree_update = False

        print(f"Epoch {epoch + 1}/{cfgs.num_epoch}")

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
            
            # print(the_best_tree)
        # 检查是否超过一分钟
            step_cur = episode+(epoch-1)*cfgs.num_episodes

        # if time.time() - start_time > STOP_TIME: 
        if step_cur >= STOP_STEP:
            if raw_tree_str:
                rf_distance, relative_rf_distance = calculate_rf_distance(raw_tree_str, the_best_tree)
            else:
                rf_distance = None
                relative_rf_distance = None
            break

    return dict(the_best_tree=the_best_tree, the_best_score=the_best_score, raw_tree_score=raw_tree_score, c_best_tree_score=c_best_tree_score, time_when_best_score_max=time_when_best_score_max, relative_rf_distance=relative_rf_distance, step_cur=step_cur)


def Agmax_one_instance(cfgs, MSA_file, c_best_tree_file=None, raw_tree_file=None):
    env = PhyInferEnv(cfgs, device)
    batch_size = cfgs.env.batch_size

    if cfgs.reload_checkpoint_path:
        PGPI_pretrained = PGPI(cfgs).to(device)
        checkpoint = torch.load(cfgs.reload_checkpoint_path)
        PGPI_pretrained.load_state_dict(checkpoint['model_state_dict'])
        accumulated_steps = checkpoint['accumulated_steps']
        epoch_now = checkpoint['epoch']
        print(f"Reloaded checkpoint at epoch {epoch_now}, accumulated_steps {accumulated_steps}")
    else:
        PGPI_pretrained = PGPI(cfgs).to(device)
        accumulated_steps = 0
        epoch_now = 0

    policy_network = PGPI_pretrained

    batch = load_pi_instance(MSA_file)

    def expand_one_instance(x, L):
        if isinstance(x, list):
            return x * L
        elif isinstance(x, torch.Tensor):
            return x.expand(L, *x.shape[1:])
        else:
            assert False

    fintune_instance_id = 0
    batch = {k:expand_one_instance(batch[k][fintune_instance_id:fintune_instance_id+1], batch_size) for k in batch.keys()}

    if raw_tree_file:
        raw_tree_score, raw_tree_str = optimize_branch_length(MSA_file, raw_tree_file)
    else:
        raw_tree_score = None
        raw_tree_str = None

    if c_best_tree_file:
        c_best_tree_score, c_best_tree_str = optimize_branch_length(MSA_file, c_best_tree_file)
    else:
        c_best_tree_score = None 
        c_best_tree_str = None

    _, _, scores, best_tree = reinforce_rollout(batch, policy_network, env, cfgs, eval=True, argmax=True)

    # import pdb; pdb.set_trace()
    score = scores[0].detach().cpu().item()
    best_tree_str = best_tree

    # import pdb; pdb.set_trace()
    
    rf_distance = None
    if c_best_tree_str:
        _, rf_distance = calculate_rf_distance(c_best_tree_str, best_tree_str)

    rf_distance_raw = None 
    if raw_tree_str:
        _, rf_distance_raw = calculate_rf_distance(raw_tree_str, best_tree_str)

    rf_distance_c_raw = None
    if raw_tree_str and c_best_tree_str:
        _, rf_distance_c_raw = calculate_rf_distance(raw_tree_str, c_best_tree_str)


    return dict(best_tree_str=best_tree_str, score=score, raw_tree_score=raw_tree_score, c_best_tree_score=c_best_tree_score, rf_distance=rf_distance, rf_distance_raw=rf_distance_raw, rf_distance_c_raw=rf_distance_c_raw)


def Argmax_inference(test_file_path, write_dir,write_file_name):
    files = os.listdir(test_file_path)
    files = [file for file in files if file.endswith(".phy")]

    if not os.path.exists(write_dir):
        os.makedirs(write_dir)

    # RL_argmax(test_file_path, cfgs, write_dir, write_file_name) 
    for file in files:
        MSA_file = os.path.join(test_file_path, file)
        if all(os.path.exists(f) for f in [MSA_file]):
            result_dict = Agmax_one_instance(cfgs, MSA_file)
            # save the best tree to write_dir as "MSA_file" + ".tre"
            with open(os.path.join(write_dir, file[:-4] + ".tre"), "w") as f:
                f.write(result_dict["best_tree_str"])


def Search_inference(test_file_path, write_dir,write_file_name):
    files = os.listdir(test_file_path)
    files = [file for file in files if file.endswith(".phy")]
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)

    for file in files:
        MSA_file = os.path.join(test_file_path, file)
        if all(os.path.exists(f) for f in [MSA_file]):
            # Call the placeholder RL_Search function
            result_dict = RL_Search(cfgs, MSA_file)
            with open(os.path.join(write_dir, file[:-4] + ".tre"), "w") as f:
                f.write(result_dict["the_best_tree"])


def finetune_inference(test_file_path, write_dir, write_file_name):
     # ***************************************Search***************************************
    files = os.listdir(test_file_path)
    files = [file for file in files if file.endswith(".phy")]

    if not os.path.exists(write_dir):
        os.makedirs(write_dir)
    for file in files:
        MSA_file = os.path.join(test_file_path, file)
        if all(os.path.exists(f) for f in [MSA_file]):
            # Call the placeholder RL_Search function
            result_dict = RL_finetuning(cfgs, MSA_file)
            
            with open(os.path.join(write_dir, file[:-4] + ".tre"), "w") as f:
                f.write(result_dict["the_best_tree"])




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Policy Gradient Training')
    parser.add_argument('--config_path', type=str, default="", help='Configuration file path')
    parser.add_argument('--infer_opt', type=str, default="Argmax", help='Options of inference: "Argmax" for NeuralNJ, "Search" for NeuralNJ-MC and "Finetune" for NeuralNJ-RL')
    parser.add_argument('--stop_step', type=int, default=100, help='Stop step')
    parser.add_argument('--evolution_model', type=str, default="GTR+I+G", help='Evolution model')
    args = parser.parse_args()

    cfgs = utils.empty_config()
    cfgs.merge_from_file(args.config_path)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    cfgs.instance_path = f"{current_dir}/{cfgs.instance_path}"
    cfgs.reload_checkpoint_path = f"{current_dir}/{cfgs.reload_checkpoint_path}"
    test_file_dir = cfgs.instance_path

    # label = "Argmax" 
    # label = "Search"
    label = args.infer_opt
    STOP_STEP = args.stop_step
    evolution_model = args.evolution_model
    utils.set_evolution_model(args.evolution_model)

    test_file_path = test_file_dir
    test_file_name = test_file_path.split('/')[-3:]
    write_dir =  f"examples/{label}_dim{cfgs.model.embed_dim}_patch{cfgs.model.patch_size}/"
    write_file_name = f"{label}_bs{cfgs.env.batch_size}.csv"

    if label == "Argmax":
        Argmax_inference(test_file_path, write_dir, write_file_name)

    elif label == "Search":
        Search_inference(test_file_path, write_dir, write_file_name)

    elif label == "Finetune":
        finetune_inference(test_file_path,write_dir, write_file_name)

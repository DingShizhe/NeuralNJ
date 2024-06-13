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
from tqdm import tqdm
import time
import einops
import raxmlpy

from phydata import PhyDataset, PhySampler, custom_collate_fn, load_pi_instance, load_tree_file, load_phy_file_multirow, load_phy_file


torch.autograd.set_detect_anomaly(True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# EXPLORE_TEMPERTURE = lambda step: 1.0

EXPLORE_TEMPERTURE = lambda step: (2 - 1) / (32 - 0) * (step - 0) + 1



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
    return logll


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

    tree_op, logllint, logllop = raxmlpy.optimize_brlen(tree_str, msa, is_root=False, iters=iters)
    return logllop


def reinforce_rollout(batch, agent, env, cfgs, replay_buffer=None, eval=False):

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

        log_p = torch.log_softmax(logits / EXPLORE_TEMPERTURE(step), dim=1)
        # actions = Categorical(logits=log_p).sample()

        if eval:
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

    scores, best_rtree_tuple, best_tree_tupe, best_tree = env.evaluate_loglikelihood()

    if not eval:
        assert replay_buffer is not None
        _trees, _scores = env.dump_end_trees()
        replay_buffer.add(_trees, _scores)


    return selected_log_ps, log_ps, scores, best_tree


def train(cfgs):

    time_str = time.strftime("%Y%m%d-%H%M%S")

    summary_writer_log_dir = os.path.join("infer_tb_logs/" + cfgs.summary_path, time_str+cfgs.summary_name)
    summary_writer = SummaryWriter(log_dir=summary_writer_log_dir)

    os.makedirs(
        os.path.join(cfgs.checkpoint_path, time_str+cfgs.summary_name),
        exist_ok=True
    )

    env = PhyInferEnv(cfgs, device)

    # replay_buffer = utils.ReplayBuffer(cfgs.replay_buffer_size)

    trajectory_num = 1 #cfgs.ENV.TRAJECTORY_NUM
    # val_path = "/home/dingshizhe/mnt/iclr_2024_phylogfn_suppl/PGPI/data_gen/CSolver/fixed_len_data_n_30_100_test"

    # dataset_val = PhyDataset(val_path, trajectory_num=trajectory_num, device=device, num_per_file=16, taxa_list=[100], C_solved=True)
    batch_size = cfgs.env.batch_size
    # data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, collate_fn=custom_collate_fn, num_workers=2)

    print("Data loaded")

    if cfgs.reload_checkpoint_path:
        PGPI_pretrained = PGPI(cfgs).to(device)
        checkpoint = torch.load(cfgs.reload_checkpoint_path)
        PGPI_pretrained.load_state_dict(checkpoint['model_state_dict'], strict=False)

        # only finetune the s_out layer
        # if True:
        if False:
            # 首先，冻结模型中的所有参数
            for param in PGPI_pretrained.parameters():
                param.requires_grad = False

            # 然后，解冻你想要优化的特定层的参数
            for param in PGPI_pretrained.s_out.parameters():
                param.requires_grad = True
            for param in PGPI_pretrained.h_linear_last.parameters():
                param.requires_grad = True
            for param in PGPI_pretrained.g_linear_last.parameters():
                param.requires_grad = True
            for param in PGPI_pretrained.g_attn_q.parameters():
                param.requires_grad = True
            for param in PGPI_pretrained.g_attn_k.parameters():
                param.requires_grad = True

            # 最后，更新优化器，只传递解冻的参数
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, PGPI_pretrained.parameters()), lr=cfgs.lr)
        else:
            optimizer = optim.Adam(PGPI_pretrained.parameters(), lr=cfgs.lr)
            # optimizer = optim.SGD(PGPI_pretrained.parameters(), lr=cfgs.lr)
            # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.93)
        accumulated_steps = checkpoint['accumulated_steps']
        epoch_now = checkpoint['epoch']
        print(f"Reloaded checkpoint at epoch {epoch_now}, accumulated_steps {accumulated_steps}")
    else:
        PGPI_pretrained = PGPI(cfgs).to(device)
        optimizer = optim.Adam(PGPI_pretrained.parameters(), lr=cfgs.lr)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.93)
        accumulated_steps = 0
        epoch_now = 0

    accumulated_steps = 0
    epoch_now = 0

    policy_network = PGPI_pretrained

    baseline_val = -np.inf
    baseline_val_batch = []

    the_best_tree = None
    the_best_score = -np.inf


    batch = load_pi_instance(
        os.path.join(cfgs.instance_path, cfgs.sequences_file)
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

    ### dynamic_batch_size ####
    batch_size = int(24 / ((taxa_num_list[0]/50)**2 * (seq_len_list[0]/1024)))
    print(f"batch_size : {batch_size} for taxa {taxa_num_list[0]} seq len {seq_len_list[0]}")

    cfgs.replay_buffer_size = batch_size
    cfgs.replay_buffer_sample_size = int(batch_size/2)
    replay_buffer = utils.ReplayBuffer(cfgs.replay_buffer_size)
    ### dynamic_batch_size ####

    batch = {k:expand_one_instance(batch[k][fintune_instance_id:fintune_instance_id+1], batch_size) for k in batch.keys()}

    # logitss, actions_sets_list, actions_set_masks, actions_sets, actions_set_complement_masks, actions_sets_complement, selected_log_ps, scores, best_tree = supervise_rollout(batch, PGPI_pretrained, env, eval=True, pretrained=True, branch_optimize=True)

    print(batch["data"][0].shape)
    print(batch["data"][0].shape)
    print(batch["data"][0].shape)


    if cfgs.raw_tree_file:
        raw_tree_score = optimize_branch_length(
            os.path.join(cfgs.instance_path, cfgs.sequences_file),
            os.path.join(cfgs.instance_path, cfgs.raw_tree_file)
        )
    else:
        raw_tree_score = None

    if cfgs.c_best_tree_file:
        c_best_tree_score = optimize_branch_length(
            os.path.join(cfgs.instance_path, cfgs.sequences_file),
            os.path.join(cfgs.instance_path, cfgs.c_best_tree_file)
        )
    else:
        c_best_tree_score = None

    # _, _, _, _, _, _, _, scores_ref, best_tree_ref = supervise_rollout(batch, PGPI_pretrained, env, eval=True, pretrained=True, branch_optimize=True)

    # import pdb; pdb.set_trace()
    # scores_ref = scores_ref.mean()
    # raw_scores_ref = raw_tree_scores.mean()
    # import pdb; pdb.set_trace()

    for epoch in range(epoch_now+1, cfgs.num_epoch):

        optimizer.zero_grad()

        the_best_tree_update = False

        print(f"Epoch {epoch + 1}/{cfgs.num_epoch}")

        # if epoch == 0:
        if False:
            baseline = "RANDOM"
            cfgs.num_episodes_baseline = 50
        else:
            # baseline = copy.deepcopy(policy_network)
            baseline = policy_network
            cfgs.num_episodes_baseline = 1
            # baseline = policy_network


        if epoch == 1:

            baseline_scores_collect = []
            for episode in tqdm(range(cfgs.num_episodes_baseline)):
                _, _, baseline_scores, _ = reinforce_rollout(batch, baseline, env, cfgs, eval=True)
                baseline_scores_collect.append(baseline_scores)
            baseline_scores_collect = torch.cat(baseline_scores_collect, dim=0)
            baseline_val = max(baseline_scores_collect.min(), baseline_val)

        else:
            new_baseline_val = sum(baseline_val_batch) / len(baseline_val_batch)
            baseline_val = max(baseline_val, new_baseline_val)
            # baseline_val = np.min(replay_buffer.scores)
            baseline_val_batch = []


        progress_bar = tqdm(range(cfgs.num_episodes), total=cfgs.num_episodes)

        for episode in progress_bar:

            selected_log_ps, log_ps, scores, best_tree = reinforce_rollout(batch, policy_network, env, cfgs, replay_buffer)
            selected_log_p = selected_log_ps.sum(dim=1)

            # try policy losse
            policy_loss = (
                - (selected_log_p) * (scores - baseline_val)
            ).mean()

            # # # try policy loss
            # policy_loss = (
            #     - (selected_log_p) * (scores - baseline_val) * (scores > baseline_val)
            # ).sum() / max((scores > baseline_val).sum(), 1.0)


            # epsilon = 0.5

            # def calculate_score_epsilon(scores, epsilon):
            #     sorted_scores, _ = torch.sort(scores)
            #     epsilon_abs = int(len(sorted_scores) * (1 - epsilon))
            #     score_epsilon = sorted_scores[epsilon_abs]
            #     return score_epsilon.item()

            # score_epsilon = calculate_score_epsilon(scores, epsilon)

            # # try policy loss
            # policy_loss = (
            #     - (selected_log_p) * (scores - score_epsilon) * (scores > score_epsilon)
            # ).sum() / (scores > score_epsilon).sum()

            # import pdb; pdb.set_trace()

            # advantage = scores - baseline_val
            # # advantage normalization
            # advantage_norm = advantage / (advantage.std() + 1e-8)
            # policy_loss = (
            #     - (selected_log_p) * advantage_norm
            # ).mean()

            # risk seeking policy loss
            # policy_loss = (
            #     - (selected_log_p) * (scores.detach() - baseline_val) * mask
            # ).sum() / K

            # branch_length_loss = - (scores * mask).sum() / K
            # try maximize entropy
            entropy_reg_loss = - sum(
                [
                    - torch.sum(torch.exp(log_p) * log_p, dim=1).mean()
                    for log_p in log_ps
                ]
            )

            baseline_val_batch.append(scores.mean().item())

            loss = policy_loss + entropy_reg_loss * cfgs.entropy_reg_strength

            # loss = policy_loss + entropy_reg_loss * cfgs.entropy_reg_strength
            # loss = branch_length_loss

            loss.backward()

            # for name, param in policy_network.named_parameters():
            #     if param.grad is not None and torch.isnan(param.grad).any():
            #         print(name)

            if episode == cfgs.num_episodes - 1:
                clip_grad_value_(policy_network.parameters(), clip_value=cfgs.clip_value)
                optimizer.step()


            baseline_val_batch.append(scores.mean().detach().item())

            g_episode = episode + cfgs.num_episodes * epoch

            summary_writer.add_scalar('Log likelihood (mean)', scores.mean().item(), g_episode)
            summary_writer.add_scalar('Log likelihood (std)', scores.std().item(), g_episode)
            summary_writer.add_scalar('Log likelihood (best)', scores.max().item(), g_episode)
            if c_best_tree_score is not None:
                summary_writer.add_scalar('Log likelihood Diff (best)', c_best_tree_score - scores.max().item(), g_episode)
                summary_writer.add_scalar('Log likelihood Diff (global best)', c_best_tree_score - the_best_score, g_episode)
                summary_writer.add_scalar('Log likelihood Ref', c_best_tree_score, g_episode)
            if raw_tree_score is not None:
                summary_writer.add_scalar('Log likelihood RAW Diff (best)', raw_tree_score - scores.max().item(), g_episode)
                summary_writer.add_scalar('Log likelihood RAW Ref', raw_tree_score, g_episode)
            if raw_tree_score is not None and raw_tree_score is not None:
                summary_writer.add_scalar('Log likelihood C-RAW Diff', c_best_tree_score - raw_tree_score, g_episode)
            summary_writer.add_scalar('Selected_Log_P (mean)', selected_log_p.mean().item(), g_episode)

            summary_writer.add_scalar('Policy Loss', policy_loss.item(), g_episode)
            summary_writer.add_scalar('Entropy Loss', entropy_reg_loss.item(), g_episode)
            summary_writer.add_scalar('Baseline Log likelihood', baseline_val, g_episode)
            summary_writer.add_scalar('Epoch', epoch, g_episode)

            # duplicate_count = replay_buffer.check_duplicate(lambda x: environment.format_rtree_topology(x, at_root=True, sequence_keys=env.sequence_keys))
            # summary_writer.add_scalar('Duplicate count', duplicate_count, g_episode)
            summary_writer.add_scalar('ReplayBuffer Size', replay_buffer.get_size(), g_episode)
            summary_writer.add_scalar('ReplayBuffer Mean', np.mean(replay_buffer.scores), g_episode)
            summary_writer.add_scalar('ReplayBuffer Std', np.std(replay_buffer.scores), g_episode)
            # summary_writer.add_scalar('Advantage Mean', advantage.mean().item(), g_episode)
            # summary_writer.add_scalar('Advantage Std', advantage.std().item(), g_episode)

            progress_bar.set_description(f"Episode {episode+1}")
            progress_bar.set_postfix(loss=policy_loss.item())

            if the_best_tree is None:
                the_best_tree = best_tree
            else:
                best_tree_log_score = max(scores).detach().cpu().item()
                if best_tree_log_score > the_best_score:
                    the_best_tree = best_tree
                    the_best_score = best_tree_log_score
                    the_best_tree_update = True
            
            # print(the_best_tree)

        if the_best_tree_update:
            torch.save(
                policy_network.state_dict(),
                os.path.join(cfgs.checkpoint_path, time_str+cfgs.summary_name, "model_best_%02d.pt" % epoch)
            )

    summary_writer.close()

    import pickle
    with open(os.path.join(
        cfgs.checkpoint_path, time_str+cfgs.summary_name, "best_tree.pkl"),
        "wb"
    ) as f:
        pickle.dump(the_best_tree, f)

    import pdb; pdb.set_trace()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Policy Gradient Training')
    parser.add_argument('--config_path', type=str, default="", help='Configuration file path')
    args = parser.parse_args()

    cfgs = utils.empty_config()
    cfgs.merge_from_file(args.config_path)

    os.makedirs("infer_tb_logs/" + cfgs.summary_path, exist_ok=True)
    os.makedirs(cfgs.checkpoint_path, exist_ok=True)

    train(cfgs)

import argparse
import copy
import os
import time

import numpy as np
import torch
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.nn.utils import clip_grad_value_
from torch.utils.tensorboard import SummaryWriter

import utils
from environment import PhyInferEnv
from model import PhyloATTN as PGPI
from phydata import PhyDataset, PhySampler, custom_collate_fn



torch.autograd.set_detect_anomaly(True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


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



def supervise_rollout(batch, agent, env, eval=False, pretrained=True, action_set=True, branch_optimize=False):
    batch_seqs, batch_seq_keys, batch_array, batch_action, batch_action_set = batch['seqs'], batch['seq_keys'], batch['data'], batch['actions'], batch['actions_set']

    batch_weights = batch['seq_weights']
    # batch_distances = batch['batch_distances']
    trees = batch['trees']

    # import pdb; pdb.set_trace()

    b = len(batch_action_set)
    # k = len(batch_action_set[0])
    n = len(batch_action_set[0][0])
    d = max([len(x) for z in batch_action_set for y in z for x in y])

    # b, _, n, d, _ = batch_action_set.shape

    batch_array = batch_array.to(device)
    batch_action = batch_action.to(device)

    batch_weights = batch_weights.to(device)
    batch_seq_mask = batch_weights == 0
    
    # batch_distances = batch_distances.to(device)
    # batch_distances = batch_distances / 1024.0


    batch_array_one_hot = batch_array 
    batch_trees = batch['trees']
    env.init_states(batch_seqs, batch_seq_keys, batch_array_one_hot, label_trees=batch_trees)
    
    selected_log_ps = []
    logitss = []

    actions_sets_list = []

    actions_set_masks = []
    actions_sets = []

    actions_set_complement_masks = []
    actions_sets_complement = []

    step = 0

    actions_ij_prev = None
    logits_prev = None

    while True:
        if step == 0:
            if eval:
                agent.eval()
                with torch.no_grad():
                    env.state_tensor = agent.encode_zxr(env.init_state_tensor, batch_seq_mask)
            else:
                # import pdb; pdb.set_trace()
                env.state_tensor = agent.encode_zxr(env.init_state_tensor, batch_seq_mask)

        # import pdb; pdb.set_trace()
        batch_size, nb_seq = env.state_tensor.shape[:2]

        if eval:
            agent.eval()
            torch.set_grad_enabled(False)
        else:
            torch.set_grad_enabled(True)

        if actions_ij_prev is not None:
            score_indices_to_prev = utils.get_score_indices_to_prev(actions_ij_prev, env, nb_seq, batch_size)
            score_indices_to_prev = torch.from_numpy(np.array(score_indices_to_prev)).to(device)
        else:
            score_indices_to_prev = None

        ret = agent.decode_zxr(env.state_tensor, batch_seq_mask, (actions_ij_prev, score_indices_to_prev, logits_prev))
        logits = ret['logits']
        log_p = torch.log_softmax(logits, dim=1)
        edge_actions = [(None, None) for _ in range(batch_size)]

        if pretrained:
            try:
                actions_ij_cur_step = batch_action[:, 0, step]
            except Exception as e:
                print(f"Error: {e}")
            actions_cur_step = [env.action_indices_dict[nb_seq][(ai.item(), aj.item())] for ai, aj in actions_ij_cur_step]
            actions_cur_step = torch.from_numpy(np.array(actions_cur_step)).to(device)
            actions = actions_cur_step

            if action_set:
                actions_set_cur_step, actions_complement_set_cur_step, actions_set_cur_step_mask, actions_complement_set_cur_step_mask, actions_set_list_cur_step = prepare_action_sets(batch_action_set, actions, nb_seq, env, step, device)

        torch.set_grad_enabled(True)

        edge_actions = [(None, None) for _ in range(batch_size)]

        done = env.step(actions, edge_actions, branch_optimize=branch_optimize, agent=agent)

        actions_ij_prev = actions_ij_cur_step
        logits_prev = logits

        if done:
            break

        # import pdb; pdb.set_trace()
        step += 1

        selected_log_p = torch.gather(log_p, 1, actions.unsqueeze(1))
        selected_log_ps.append(selected_log_p)
        logitss.append(logits)
        actions_set_masks.append(actions_set_cur_step_mask)
        actions_sets.append(actions_set_cur_step)
        actions_sets_list.append(actions_set_list_cur_step)

        actions_set_complement_masks.append(actions_complement_set_cur_step_mask)
        actions_sets_complement.append(actions_complement_set_cur_step)

        # Only first step
        # break

    selected_log_ps = torch.cat(selected_log_ps, dim=1)
    if branch_optimize:
        scores, best_rtree_tuple, best_tree_tupe, best_tree = env.evaluate_loglikelihood()
        return logitss, actions_sets_list, actions_set_masks, actions_sets, actions_set_complement_masks, actions_sets_complement, selected_log_ps, scores, best_tree
    
    # return selected_logitss, logitss, scores, best_tree
    return logitss, actions_sets_list, actions_set_masks, actions_sets, actions_set_complement_masks, actions_sets_complement, selected_log_ps


def reinforce_rollout(batch, agent, env, eval=False, get_all_tree=False):
    batch_seqs, batch_seq_keys, batch_array, batch_action, batch_action_set = batch['seqs'], batch['seq_keys'], batch['data'], batch['actions'], batch['actions_set']

    batch_weights = batch['seq_weights']

    b = len(batch_action_set)
    n = len(batch_action_set[0][0])
    d = max([len(x) for z in batch_action_set for y in z for x in y])


    batch_array = batch_array.to(device)
    batch_action = batch_action.to(device)

    batch_weights = batch_weights.to(device)
    batch_seq_mask = batch_weights == 0
    
    batch_array_one_hot = batch_array #torch.zeros((*batch_array.shape, 4), device=device)

    batch_trees = batch['trees']
    env.init_states(batch_seqs, batch_seq_keys, batch_array_one_hot, label_trees=batch_trees)

    selected_log_ps = []
    logitss = []
    actionss = []

    step = 0

    actions_ij_prev = None
    logits_prev = None

    while True:
        if step == 0:

            if eval:
                agent.eval()
                with torch.no_grad():
                    env.state_tensor = agent.encode_zxr(env.init_state_tensor, batch_seq_mask)
            else:
                env.state_tensor = agent.encode_zxr(env.init_state_tensor, batch_seq_mask)

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
        log_p = torch.log_softmax(logits, dim=1)
        # actions = Categorical(logits=log_p).sample()

        if eval:
            actions = torch.argmax(log_p, dim=-1)
        else:
            actions = Categorical(logits=log_p).sample()

        actionss.append(actions)

        actions_ij_prev = [env.tree_pairs_dict[nb_seq][a.item()] for a in actions]
        actions_ij_prev = torch.from_numpy(np.array(actions_ij_prev)).to(device).to(torch.int32)

        # actions_set_cur_step = torch.from_numpy(np.array(actions_set_cur_step)).to(device)
        edge_actions = [(None, None) for _ in range(batch_size)]

        done = env.step(actions, edge_actions, branch_optimize=True, agent=agent)

        if done:
            break

        step += 1

        selected_log_p = torch.gather(log_p, 1, actions.unsqueeze(1))
        selected_log_ps.append(selected_log_p)
    
        logitss.append(logits)
        logits_prev = logits

    selected_log_ps = torch.cat(selected_log_ps, dim=1)

    if get_all_tree:
        scores, best_rtree_tuple, best_tree_tupe, best_tree = env.evaluate_loglikelihood(get_all_tree=True)
    else:
        scores, best_rtree_tuple, best_tree_tupe, best_tree = env.evaluate_loglikelihood()


    return logitss, actionss, selected_log_ps, scores, best_tree, env.count_match_label_steps



def train(cfgs):

    time_str = time.strftime("%Y%m%d-%H%M%S")

    summary_writer_log_dir = os.path.join("tb_logs/" + cfgs.summary_path, time_str+cfgs.summary_name)
    summary_writer = SummaryWriter(log_dir=summary_writer_log_dir)

    os.makedirs(
        os.path.join(cfgs.checkpoint_path, time_str+cfgs.summary_name),
        exist_ok=True
    )

    env = PhyInferEnv(cfgs, device)

    train_path = cfgs.dataset_path
    val_path = cfgs.val_dataset_path
    trajectory_num = 1 #cfgs.ENV.TRAJECTORY_NUM
    taxa_list = cfgs.dataset_taxa_list
    len_list = cfgs.dataset_len_list
    dataset = PhyDataset(train_path, trajectory_num=trajectory_num, device=device,len_list=len_list, taxa_list=taxa_list)
    dataset_train_val = PhyDataset(train_path, trajectory_num=trajectory_num, device=device,num_per_file=32,len_list=[256,512,1024], taxa_list=taxa_list)
    dataset_val2 = PhyDataset(val_path, trajectory_num=trajectory_num, device=device, num_per_file=32, len_list=[1024], taxa_list=[50])

    batch_size = cfgs.env.batch_size

    sampler = PhySampler(dataset, batch_size)

    data_loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler, collate_fn=custom_collate_fn, num_workers=2)

    train_val_sample_val = PhySampler(dataset_train_val, batch_size, shuffle=False)
    data_loader_train_val = torch.utils.data.DataLoader(dataset_train_val, batch_sampler=train_val_sample_val, collate_fn=custom_collate_fn, num_workers=2)
    val_sample_val2 = PhySampler(dataset_val2, batch_size , shuffle=False)
    data_loader_val2 = torch.utils.data.DataLoader(dataset_val2, batch_sampler=val_sample_val2, collate_fn=custom_collate_fn, num_workers=2)
    

    BALANCED_ELU_LOSS = cfgs.loss.BALANCED_ELU_LOSS
    ELU_LOSS = cfgs.loss.ELU_LOSS


    print("Data loaded")

    if cfgs.reload_checkpoint_path:
        PGPI_pretrained = PGPI(cfgs).to(device)
        checkpoint = torch.load(cfgs.reload_checkpoint_path)
        PGPI_pretrained.load_state_dict(checkpoint['model_state_dict'])
        optimizer = optim.Adam(PGPI_pretrained.parameters(), lr=cfgs.lr)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.93)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        accumulated_steps = checkpoint['accumulated_steps']
        epoch_now = checkpoint['epoch']
        print(f"Reloaded checkpoint at epoch {epoch_now}, accumulated_steps {accumulated_steps}")
    else:
        PGPI_pretrained = PGPI(cfgs).to(device)
        optimizer = optim.Adam(PGPI_pretrained.parameters(), lr=cfgs.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        accumulated_steps = 0
        epoch_now = 0

    ratio = 1.0


    first_val_flag = True
    ref_scores_dict = dict()


    for epoch in range(epoch_now+1, cfgs.num_epoch):

        optimizer.zero_grad()
        print(f"Epoch {epoch + 1}/{cfgs.num_epoch}")
        print(f"Number of batches: {len(data_loader)}")
        # for batch in tqdm(data_loader):

        for _, batch in enumerate(data_loader):
            PGPI_pretrained.train()
            print(f"data: {len(batch['data'][0][0])}\t seq_keys: {len(batch['seq_keys'][0])}\t Tag: {time_str}")

            logitss, _, actions_set_masks, actions_sets, actions_set_complement_masks, actions_sets_complement, _ = supervise_rollout(batch, PGPI_pretrained, env, eval=False, pretrained=True)

            Policy_loss = 0

            precision = 0
            all_loss_pair_num = 0
            selected_scores_mean = []
            unselected_scores_mean = []
            selected_scores_max = []
            unselected_scores_max = []
            step_batch_num = 0
            for step_id in range(len(logitss)):
                if True:
                    selected_scores = torch.gather(logitss[step_id], 1, actions_sets[step_id])
                    unselected_scores = torch.gather(logitss[step_id], 1, actions_sets_complement[step_id])

                    selected_scores.masked_fill_(actions_set_masks[step_id] == False, 0)
                    unselected_scores.masked_fill_(actions_set_complement_masks[step_id] == False, 0)

                    selected_scores_mean.append((selected_scores.sum(axis=-1) / actions_set_masks[step_id].sum(axis=-1)).sum().item())
                    unselected_scores_mean.append((unselected_scores.sum(axis=-1) / actions_set_complement_masks[step_id].sum(axis=-1)).sum().item())


                    selected_scores.masked_fill_(actions_set_masks[step_id] == False, -1e9)
                    unselected_scores.masked_fill_(actions_set_complement_masks[step_id] == False, -1e9)
                    selected_scores_max.append(selected_scores.max(axis=-1)[0].sum().item())
                    unselected_scores_max.append(unselected_scores.max(axis=-1)[0].sum().item())
                    step_batch_num += selected_scores.shape[0]
                    selected_scores.masked_fill_(actions_set_masks[step_id] == False, 0)
                    unselected_scores.masked_fill_(actions_set_complement_masks[step_id] == False, 0)
                    if step_id == 0:
                        selected_scores.masked_fill_(actions_set_masks[step_id] == False, 0)
                        unselected_scores.masked_fill_(actions_set_complement_masks[step_id] == False, 0)

                        selected_scores.masked_fill_(actions_set_masks[step_id] == False, -1e9)
                        unselected_scores.masked_fill_(actions_set_complement_masks[step_id] == False, -1e9)
                        selected_scores.masked_fill_(actions_set_masks[step_id] == False, 0)
                        unselected_scores.masked_fill_(actions_set_complement_masks[step_id] == False, 0)

                    # margin = 0.1
                    margin = 0.5
                    # margin = 1
                    # margin = 2
                    # margin = 5

                    scores_diff = (selected_scores[:, :, None] - margin - unselected_scores[:, None, :])
                    scores_diff.masked_fill_(actions_set_masks[step_id][:, :, None] == False, 0)
                    scores_diff.masked_fill_(actions_set_complement_masks[step_id][:, None, :] == False, 0)


                    mask_sum_this_step = (actions_set_masks[step_id][:, :, None] * actions_set_complement_masks[step_id][:, None, :]).sum()

                    if BALANCED_ELU_LOSS:

                        unselected_scores.masked_fill_(actions_set_complement_masks[step_id] == False, -1e9)


                        # ratio = max(1 - 3 / (4 * 20) * epoch, 1/4) * cfgs.ratio_factor
                        ratio = max(1 - 3 / (4 * 20) * epoch /2, 1/4) * cfgs.ratio_factor
                        # ratio = max(1 - 3 / (4 * 20) * epoch /2 * 3, 1/4) * cfgs.ratio_factor

                        K = min(unselected_scores.size(1), max(int(unselected_scores.size(1) * ratio), 8))
                        unselected_scores_K, unselected_scores_K_indices = torch.topk(unselected_scores, K, dim=1)
                        
                        unselected_scores_K_mask = torch.gather(actions_set_complement_masks[step_id], 1, unselected_scores_K_indices)

                        # assert torch.all(unselected_scores_K_mask)

                        scores_diff = (selected_scores[:, :, None] - margin - unselected_scores_K[:, None, :])
                        scores_diff.masked_fill_(actions_set_masks[step_id][:, :, None] == False, 0)
                        scores_diff.masked_fill_(unselected_scores_K_mask[:, None, :] == False, 0)

                        mask_sum_this_step = (actions_set_masks[step_id][:, :, None] * unselected_scores_K_mask[:, None, :]).sum()

                        loss_this_step = F.elu( - scores_diff ).sum()

                        loss_this_step = loss_this_step / mask_sum_this_step

                        Policy_loss += loss_this_step
                    elif ELU_LOSS:
                        loss_this_step = F.elu( - scores_diff ).sum() / mask_sum_this_step
                        Policy_loss += loss_this_step


                if BALANCED_ELU_LOSS:
                    precision_step = (selected_scores[:, :, None] > unselected_scores_K[:, None, :])
                    precision_step.masked_fill_(actions_set_masks[step_id][:, :, None] == False, False)
                    precision_step.masked_fill_(unselected_scores_K_mask[:, None, :] == False, False)
                else:
                    precision_step = (selected_scores[:, :, None] > unselected_scores[:, None, :])
                    precision_step.masked_fill_(actions_set_masks[step_id][:, :, None] == False, False)
                    precision_step.masked_fill_(actions_set_complement_masks[step_id][:, None, :] == False, False)

                precision += precision_step.sum()
                all_loss_pair_num += mask_sum_this_step

            # import pdb; pdb.set_trace()

            precision =  precision / all_loss_pair_num

            # num_rows = batch_distances.size(1)
            # row, col = torch.triu_indices(num_rows, num_rows, offset=1)
            # batch_distances = batch_distances[:, row, col]

            debug = os.path.exists('debug')
            if debug:
                import pdb; pdb.set_trace()


            policy_loss = Policy_loss/len(logitss)

            loss = policy_loss

            loss.backward()

            if cfgs.clip_value != -1.0:
                clip_grad_value_(PGPI_pretrained.parameters(), cfgs.clip_value)
            optimizer.step()
            optimizer.zero_grad()

            # import pdb; pdb.set_trace()
            summary_writer.add_scalar('Loss/train', loss.item(), accumulated_steps*batch_size)
            summary_writer.add_scalar('Policy_Loss/train', policy_loss.item(), accumulated_steps*batch_size)
            summary_writer.add_scalar('Precision/train', precision, accumulated_steps*batch_size)
            summary_writer.add_scalar('Epoch', epoch, accumulated_steps*batch_size)
            summary_writer.add_scalar('Unaction set loss ratio', ratio, accumulated_steps*batch_size)

            current_lr = optimizer.param_groups[0]['lr']
            summary_writer.add_scalar('Learning_rate', current_lr, accumulated_steps*batch_size)
            # summary_writer.add_scalar('Loss/Policy_loss', Policy_loss.item(), accumulated_steps*batch_size)


            if accumulated_steps % 1280 == 0:
                PGPI_pretrained.eval()

                def eval_on_dataset(dataset_tag="test", ref_scores_dict=None):

                    scores_list = []
                    scores_ref_list = []
                    rfdist_list = []

                    argmax_action_in_action_set_count_list = []

                    if dataset_tag == 'taxa50':
                        eval_data_loader = data_loader_val2
                        summary_tag = dataset_tag
                    else:
                        eval_data_loader = data_loader_train_val
                        summary_tag = 'val_train'

                    with torch.no_grad():
                        print("Starting validation")
                        batch_idx = 0
                        for _, batch in enumerate(eval_data_loader):

                            print(len(eval_data_loader))
                            print(f"Val data: {len(batch['data'][0][0])}\t seq_keys: {len(batch['seq_keys'][0])}")

                            if first_val_flag:
                                _, _, _, _, _, _, _, _scores_ref, best_tree = supervise_rollout(batch, PGPI_pretrained, env, eval=True, pretrained=True, branch_optimize=True)

                                ref_scores_dict[dataset_tag] = ref_scores_dict.get(dataset_tag, []) + [_scores_ref]

                            scores_ref = ref_scores_dict[dataset_tag][batch_idx]

                            logitss_argmax, _, selected_log_ps_argmax, scores, best_tree_pred, count_match_label_steps = reinforce_rollout(batch, PGPI_pretrained, env, eval=True, get_all_tree=True)


                            for file,tree_pred_str in zip(batch['file_paths'], best_tree_pred):
                                tree_str = utils.get_tree_str_from_newick(file[:-4]+'.tre')
                                _, rfdist = utils.calculate_rf_distance(tree_str, tree_pred_str)
                                rfdist_list.append(rfdist)

                            count_match_label_steps = [x / (len(logitss)) for x in count_match_label_steps]
                            argmax_action_in_action_set_count_list.extend(count_match_label_steps)

                            scores_list.append(scores)
                            scores_ref_list.append(scores_ref)

                            batch_idx += 1


                    # summary_tag = "val" if dataset_tag == "test" else "val_train"

                    scores_tensor = torch.cat(scores_list, dim=0)
                    summary_writer.add_scalar('Tree Log likelihood (mean)/%s' % summary_tag, scores_tensor.mean().item(), accumulated_steps*batch_size)
                    summary_writer.add_scalar('Tree Log likelihood (std)/%s' % summary_tag, scores_tensor.std().item(), accumulated_steps*batch_size)

                    scores_ref_tensor = torch.cat(scores_ref_list, dim=0)
                    summary_writer.add_scalar('Tree Log likelihood (mean) ref/%s' % summary_tag, scores_ref_tensor.mean().item(), accumulated_steps*batch_size)
                    summary_writer.add_scalar('Tree Log likelihood (std) ref/%s' % summary_tag, scores_ref_tensor.std().item(), accumulated_steps*batch_size)

                    scores_diff = (scores_ref_tensor - scores_tensor).mean()
                    summary_writer.add_scalar('Tree Log likelihood Diff (mean) diff/%s' % summary_tag, scores_diff, accumulated_steps*batch_size)

                    summary_writer.add_scalar('Argmax_action_in_action_set_count (Correct)/%s' % summary_tag, sum(argmax_action_in_action_set_count_list) / len(argmax_action_in_action_set_count_list), accumulated_steps*batch_size)

                    # Convert the float values in rfdist_list to tensors
                    rfdist_list = [torch.tensor([x]) for x in rfdist_list]
                    # Now you can concatenate them
                    rfdistance = torch.cat(rfdist_list, dim=0)
                    summary_writer.add_scalar('Normalised RF distance in tree and label tree (mean)/%s' % summary_tag, rfdistance.mean().item(), accumulated_steps*batch_size)
                    summary_writer.add_scalar('Normalised RF distance in tree and label tree (std)/%s' % summary_tag, rfdistance.std().item(), accumulated_steps*batch_size)

                eval_on_dataset("taxa50", ref_scores_dict)
                eval_on_dataset("train", ref_scores_dict)

                first_val_flag = False

            accumulated_steps += 1


        scheduler.step()

        if epoch % 2 == 0:

            torch.save({
                'epoch': epoch,
                'model_state_dict': PGPI_pretrained.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accumulated_steps': accumulated_steps
            }, os.path.join(cfgs.checkpoint_path, time_str+cfgs.summary_name, f"checkpoint_{epoch}.pt"))

            print(f"Saved checkpoint at epoch {epoch}, accumulated_steps {accumulated_steps}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Policy Gradient Training')
    parser.add_argument('--config_path', type=str, default="", help='Configuration file path')
    args = parser.parse_args()

    cfgs = utils.empty_config()
    cfgs.merge_from_file(args.config_path)

    os.makedirs("tb_logs/" + cfgs.summary_path, exist_ok=True)
    os.makedirs(cfgs.checkpoint_path, exist_ok=True)

    train(cfgs)

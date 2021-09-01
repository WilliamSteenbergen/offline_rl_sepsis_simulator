import numpy as np
import torch
from sepsisSimDiabetes.State import State

feature_names = ["hr_state", "sysbp_state", "glucose_state",
                                                   "antibiotic_state", "vaso_state", "vent_state"]
# 'grade_norm' and 'pre' are relatively categorical/discrete
#unused_features = ['input_message_kid', 'time_stored', 'grade']
# categorical_features = ['stage']  # postive_feedback

# feature_names = ['grade_norm', 'pre-score_norm', 'stage_norm', 'failed_attempts_norm',
#                 'pos_norm', 'neg_norm', 'hel_norm', 'anxiety_norm']

target_names = ["beh_p_0","beh_p_1","beh_p_2","beh_p_3","beh_p_4","beh_p_5","beh_p_6","beh_p_7"]

# feature_names = feature_names + categorical_features

MAX_TIME = 28


def compute_is_weights_for_mdp_policy(behavior_df, eval_policy, eps=0.05, temp=0.1,
                                     reward_column='adjusted_score', no_grad=True,
                                     gr_safety_thresh=0.0, is_train=True, normalize_reward=False, use_knn=False,
                                     model_round_as_feature=False):
    # is_weights with batch processing
    df = behavior_df
    user_ids = df['Trajectory'].unique()
    n = len(user_ids)

    # now max_time is dynamic, finally!!
    MAX_TIME = max(behavior_df.groupby('Trajectory').size())

    # assert reward_column in ['adjusted_score', 'reward']

    pies = torch.zeros((n, MAX_TIME))  # originally 20
    pibs = torch.zeros((n, MAX_TIME))
    rewards = torch.zeros((n, MAX_TIME))
    lengths = np.zeros((n))  # we are not doing anything to this

    # compute train split reward mean and std
    # (n,): and not average
    user_rewards = df.groupby("Trajectory")[reward_column].mean()
    train_reward_mu = user_rewards.mean()
    train_reward_std = user_rewards.std()

    for idx, user_id in enumerate(user_ids):
        data = df[df['Trajectory'] == user_id]
        # get features, targets
        if not model_round_as_feature:
            features = np.asarray(data[feature_names]).astype(float)

            features_idx_list = []
            for feature_idx in data['State_id']:
                this_state = State(state_idx=feature_idx, idx_type='full',
                                   diabetic_idx=1)  # Diab a req argument, no difference
                # assert this_state == State(state_idx = feature_idx, idx_type = 'full', diabetic_idx = 0)
                features_idx_list.append(this_state.get_state_idx('proj_obs'))

            features_idxs = np.array(features_idx_list).astype(int)
        else:
            features = np.asarray(data[feature_names + ['model_round']]).astype(float)
        targets = np.asarray(data[target_names]).astype(float)
        actions = np.asarray(data['Action_taken']).astype(int)

        length = features.shape[0]
        lengths[idx] = length

        T = targets.shape[0]
        # shape: (1, T)
        beh_probs = torch.from_numpy(np.array([targets[i, a] for i, a in enumerate(actions)])).float()
        pibs[idx, :T] = beh_probs

        gr_mask = None
        if gr_safety_thresh > 0:
            if not is_train:
                # we only use KNN during validation
                if not use_knn:
                    gr_mask = None
                # else:
                #     knn_targets = np.asarray(data[knn_target_names]).astype(float)
                #     assert knn_targets.shape[0] == targets.shape[0]
                #     beh_action_probs = torch.from_numpy(knn_targets)
                #
                #     gr_mask = beh_action_probs >= gr_safety_thresh
            else:
                beh_action_probs = torch.from_numpy(targets)

                # gr_mask = beh_probs >= gr_safety_thresh
                gr_mask = beh_action_probs >= gr_safety_thresh

            # need to renormalize behavior policy as well?

        # assign rewards (note adjusted_score we only assign to last step)
        # reward, we assign to all
        if reward_column == 'Reward':
            reward = max(np.asarray(data[reward_column]))
            if reward == 0:
                reward = min(np.asarray(data[reward_column]))
            # only normalize reward during training

            # if reward != 0:
            #     print("reward is {}".format(reward))
            if normalize_reward and is_train:
                # might not be the best -- we could just do a plain shift instead
                # like -1 shift
                reward = (reward - train_reward_mu) / train_reward_std
                # print(reward)
                # if reward != 0:
                #     print("reward is not zero")

            rewards[idx, T - 1] = reward
        else:
            # normal reward
            # rewards[idx, :T] = torch.from_numpy(np.asarray(data[reward_column])).float()
            raise Exception("We currrently do not offer training in this mode")

        # last thing: model prediction
        # eval_action_probs = eval_policy.get_action_probability(torch.from_numpy(features).float(), no_grad,
        #                                                        action_mask=gr_mask)
        eval_action_probs = torch.from_numpy(eval_policy[features_idxs, :])
        pies[idx, :T] = torch.hstack([eval_action_probs[i, a] for i, a in enumerate(actions)])

    return pibs, pies, rewards, lengths.astype(int), MAX_TIME

def wis_ope(pibs, pies, rewards, length, no_weight_norm=False, max_time=MAX_TIME, per_sample=False, clip_lower=1e-16,
            clip_upper=1e3):
    # even for batch setting, this function should still work fine
    # but for WIS, batch setting won't give accurate normalization
    n = pibs.shape[0]
    weights = torch.ones((n, MAX_TIME))

    for i in range(n):
        last = 1
        for t in range(int(length[i])):
            assert pibs[i, t] != 0
            last = last * (pies[i, t] / pibs[i, t])
            weights[i, t] = last
        weights[i, length[i]:] = weights[i, length[i] - 1]
    # weights = torch.clip(weights, 1e-16, 1e3)
    weights = torch.clip(weights, clip_lower, clip_upper)
    if not no_weight_norm:
        weights_norm = weights.sum(dim=0)
        weights /= weights_norm  # per-step weights (it's cumulative)
    else:
        weights /= n

    # this accumulates
    if not per_sample:
        return (weights[:, -1] * rewards.sum(dim=-1)).sum(dim=0), weights[:, -1]
    else:
        # return w_i associated with each N
        return weights[:, -1] * rewards.sum(dim=-1), weights[:, -1]
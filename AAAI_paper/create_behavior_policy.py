import numpy as np
import cf.counterfactual as cf
import pandas as pd
import pickle
import os

# Sepsis Simulator code
from sepsisSimDiabetes.State import State
from sepsisSimDiabetes.Action import Action
from sepsisSimDiabetes.DataGenerator import DataGenerator
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from scipy.linalg import block_diag

from AAAI_paper.smartprimer_method_functions.wis import compute_is_weights_for_mdp_policy, wis_ope

SEED = 2  # Note this is not the only random seed, see the loop

np.random.seed(SEED)
NSIMSAMPS = 1000  # Samples to draw from the simulator
NSTEPS = 20  # Max length of each trajectory
NCFSAMPS = 5  # Counterfactual Samples per observed sample
DISCOUNT_Pol = 0.99  # Used for computing optimal policies
DISCOUNT = 1  # Used for computing actual reward
PHYS_EPSILON = 0.05  # Used for sampling using physician pol as eps greedy

PROB_DIAB = 0.2

# Option 1: Use bootstrapping w/replacement on the original NSIMSAMPS to estimate errors
USE_BOOSTRAP = True
N_BOOTSTRAP = 100

# Option 2: Use repeated sampling (i.e., NSIMSAMPS fresh simulations each time) to get error bars;
# This is done in the appendix of the paper, but not in the main paper
N_REPEAT_SAMPLING = 1

n_actions = Action.NUM_ACTIONS_TOTAL
n_components = 2

# These are added as absorbing states
n_states_abs = State.NUM_FULL_STATES

# These are added as absorbing states
n_states_obs = State.NUM_OBS_STATES + 2
discStateIdx = n_states_obs - 1
deadStateIdx = n_states_obs - 2

# Get the transition and reward matrix from file
with open("data/diab_txr_mats-replication.pkl", "rb") as f:
    mdict = pickle.load(f)

tx_mat = mdict["tx_mat"]
r_mat = mdict["r_mat"]

def check_rl_policy(rl_policy, obs_samps, proj_lookup):
    passes = True
    # Check the observed actions for each state
    obs_pol = np.zeros_like(rl_policy)
    for eps_idx in range(NSIMSAMPS):
        for time_idx in range(NSTEPS):
            this_obs_action = int(obs_samps[eps_idx, time_idx, 1])
            # Need to get projected state
            if this_obs_action == -1:
                continue
            this_obs_state = proj_lookup[int(obs_samps[eps_idx, time_idx, 2])]
            obs_pol[this_obs_state, this_obs_action] += 1

    # Check if each RL action conforms to an observed action
    for eps_idx in range(NSIMSAMPS):
        for time_idx in range(NSTEPS):
            this_full_state_unobserved = int(obs_samps[eps_idx, time_idx, 1])
            this_obs_state = proj_lookup[this_full_state_unobserved]
            this_obs_action = int(obs_samps[eps_idx, time_idx, 1])

            if this_obs_action == -1:
                continue
            # This is key: In some of these trajectories, you die or get discharge.
            # In this case, no action is taken because the sequence has terminated, so there's nothing to compare the RL action to
            true_death_states = r_mat[0, 0, 0, :] == -1
            true_disch_states = r_mat[0, 0, 0, :] == 1
            if np.logical_or(true_death_states, true_disch_states)[this_full_state_unobserved]:
                continue

            this_rl_action = rl_policy[proj_lookup[this_obs_state]].argmax()
            if obs_pol[this_obs_state, this_rl_action] == 0:
                print("Eps: {} \t RL Action {} in State {} never observed".format(
                    int(time_idx / NSTEPS), this_rl_action, this_obs_state))
                passes = False
    return passes

def get_mdp():
    # These are properties of the simulator, do not change
    n_actions = Action.NUM_ACTIONS_TOTAL

    tx_mat_full = np.zeros((n_actions, State.NUM_FULL_STATES, State.NUM_FULL_STATES))
    r_mat_full = np.zeros((n_actions, State.NUM_FULL_STATES, State.NUM_FULL_STATES))

    for a in range(n_actions):
        tx_mat_full[a, ...] = block_diag(tx_mat[0, a, ...], tx_mat[1, a,...])
        r_mat_full[a, ...] = block_diag(r_mat[0, a, ...], r_mat[1, a, ...])


    fullMDP = cf.MatrixMDP(tx_mat_full, r_mat_full)
    fullPol = fullMDP.policyIteration(discount=DISCOUNT_Pol, eval_type=1)

    return fullPol, fullMDP, r_mat_full


def marginalize_policy(policy, EPS):
    # Find states that end up being the same
    new_states = {}
    for i in range(n_states_abs - 2):
        this_state = State(state_idx=i, idx_type='full',
                           diabetic_idx=1)  # Diab a req argument, no difference
        # assert this_state == State(state_idx = i, idx_type = 'obs', diabetic_idx = 0)
        new_state_vars = tuple([this_state.hr_state, this_state.sysbp_state, this_state.glucose_state,
                          this_state.antibiotic_state, this_state.vaso_state, this_state.vent_state])
        # new_state_vars = tuple([this_state.hr_state,
        #                         this_state.antibiotic_state, this_state.vaso_state, this_state.vent_state])
        if new_state_vars not in new_states.keys():
            new_states[new_state_vars] = [i]
        else:
            new_states[new_state_vars].append(i)

    # Marginalize policy
    same=0
    different=0
    marginalized_policy = np.copy(policy)
    for marginalized_state in new_states.keys():
        rows = new_states[marginalized_state]
        avg_policy = np.mean(policy[rows,:],axis=0)
        if sum(sum(policy[rows,:] == np.repeat(avg_policy[np.newaxis,:], repeats=len(rows), axis=0))) == 32:
            same +=1
        else:
            different+=1
        marginalized_policy[rows,:] = np.repeat(avg_policy[np.newaxis,:], repeats=len(rows), axis=0)

    # Add the 5% randomness
    marginalized_policy = marginalized_policy * (1-EPS) + EPS * (1/n_actions)
    return marginalized_policy

def create_data_set(num_trajectories, beh_policy, high_vol=False):
    np.random.seed(1)
    dgen = DataGenerator()

    states, actions, lengths, rewards, diab, emp_tx_totals, emp_r_totals, difficulties = dgen.simulate(
        num_trajectories, 20, policy=beh_policy, policy_idx_type='full', output_state_idx_type='full',
        p_diabetes=PROB_DIAB, use_tqdm=False, high_vol=high_vol)

    output = {
        'states': states,
        'actions': actions,
        'lengths': lengths,
        'rewards': rewards,
        'diab': diab,
        'emp_tx_totals': emp_tx_totals,
        'emp_r_totals': emp_r_totals
    }

    data_set_list = []
    for i in range(num_trajectories):
        for j in range(21):
            row = []
            # Append trajectory number
            row.append(i)

            # Append state id
            row.append(states[i,j,0])

            # Append state variables
            this_state = State(state_idx=states[i,j,0], idx_type='full',
                               diabetic_idx=1)  # Diab a req argument, no difference
            new_state_vars = [this_state.hr_state, this_state.sysbp_state, this_state.glucose_state,
                                    this_state.antibiotic_state, this_state.vaso_state, this_state.vent_state]
            row += new_state_vars

            # Append reward and action probs
            if j == 20:
                row.append(0)
                row.append(-1)
                row += [-1] * n_actions
            else:
                row.append(rewards[i, j,0])
                row.append(actions[i, j,0])
                row += list(beh_policy[states[i, j,0], :])
            row.append(difficulties[i])

            data_set_list.append(row)

    dataset = pd.DataFrame(data_set_list, columns=["Trajectory", "State_id", "hr_state", "sysbp_state", "glucose_state",
                                                   "antibiotic_state", "vaso_state", "vent_state", "Reward", "Action_taken", "beh_p_0",
                                                   "beh_p_1", "beh_p_2", "beh_p_3", "beh_p_4",
                                                   "beh_p_5","beh_p_6","beh_p_7", 'difficulty'])

    return dataset, output


def evaluate_policy(policy, policy_type='full'):
    np.random.seed(1998)
    N_TRAJECTORIES = 5000
    dgen = DataGenerator()

    if policy_type == "proj_obs" or policy_type == 'obs':
        policy = policy[:-2,:]

    states, actions, lengths, rewards, diab, emp_tx_totals, emp_r_totals, difficulties = dgen.simulate(
        N_TRAJECTORIES, 20, policy=policy, policy_idx_type=policy_type, output_state_idx_type='full',
        p_diabetes=PROB_DIAB, use_tqdm=False)

    std = rewards.flatten()
    std = np.std(std)
    return sum(sum(rewards))[0]/N_TRAJECTORIES, std

def get_all_states():
    all_states = []
    for i in range(n_states_abs):
        this_state = State(state_idx=i, idx_type='full',
                           diabetic_idx=1)  # Diab a req argument, no difference

        state = [this_state.hr_state, this_state.sysbp_state, this_state.glucose_state,this_state.percoxyg_state, this_state.diabetic_idx,
                                this_state.antibiotic_state, this_state.vaso_state, this_state.vent_state]
        all_states.append(state)

    all_states_pd = pd.DataFrame(all_states, columns=["hr_state", "sysbp_state", "glucose_state", "oxygen_state",
                                                      "diabetes_state", "antibiotic_state", "vaso_state", "vent_state"])
    return all_states_pd

def get_bcpg_policy(filename):
    bcpg_policy_pd = pd.read_csv(filename)
    bcpg_policy = np.array(bcpg_policy_pd.iloc[:, 10:])

    smartprimer_policy = np.copy(bcpg_policy)
    for i in range(bcpg_policy.shape[0]):
        smartprimer_policy[i,-1] += 1 - np.sum(bcpg_policy[i,:])
    return smartprimer_policy


def get_empirical_mdp(beh_policy, n_trajectories, do_splits=False, output=None, split=None):
    # # Construct the projection matrix for obs->proj states
    # n_proj_states = int((n_states_obs - 2) / 5) + 2
    # proj_matrix = np.zeros((n_states_obs, n_proj_states))
    # for i in range(n_states_obs - 2):
    #     this_state = State(state_idx=i, idx_type='obs',
    #                        diabetic_idx=1)  # Diab a req argument, no difference
    #     # assert this_state == State(state_idx = i, idx_type = 'obs', diabetic_idx = 0)
    #     j = this_state.get_state_idx('proj_obs')
    #     proj_matrix[i, j] = 1
    #
    # # Add the projection to death and discharge
    # proj_matrix[deadStateIdx, -2] = 1
    # proj_matrix[discStateIdx, -1] = 1

    # proj_matrix = proj_matrix.astype(int)

    if do_splits:
        np.random.seed(1)
        dgen = DataGenerator()

        states, actions, lengths, rewards, diab, emp_tx_totals, emp_r_totals, difficulties = dgen.simulate(
            n_trajectories, 20, policy=beh_policy, policy_idx_type='full',
            p_diabetes=PROB_DIAB, use_tqdm=False, splits=True, split_df=split)

        # states, actions, lengths, rewards, diab, emp_tx_totals, emp_r_totals = dgen.simulate(
        #     n_trajectories, NSTEPS, policy=beh_policy, policy_idx_type='full',
        #     p_diabetes=PROB_DIAB, use_tqdm=False, splits=True, split_df=split)
        trajectories = np.array(list(set(split['Trajectory']))).astype(int)
        states = states[trajectories, :, :]
    else:
        np.random.seed(1)
        dgen = DataGenerator()
        states, actions, lengths, rewards, diab, emp_tx_totals, emp_r_totals, difficulties = dgen.simulate(
            n_trajectories, NSTEPS, policy=beh_policy, policy_idx_type='full',
            p_diabetes=PROB_DIAB, use_tqdm=False) #True, tqdm_desc='Behaviour Policy Simulation')

    emp_r_mat_output = np.copy(emp_r_totals)
    emp_tx_mat_output = np.copy(emp_tx_totals)

    ############## Construct the Transition Matrix w/proj states ##############
    # proj_tx_cts = np.zeros((n_actions, n_proj_states, n_proj_states))
    # proj_tx_mat = np.zeros_like(proj_tx_cts)

    # (1) NOTE: Previous code marginalized here, but now we are just getting observed quantities out, no components
    # est_tx_cts = np.copy(emp_tx_totals)
    # assert est_tx_cts.ndim == 3

    # (2) Add new aborbing states, and a new est_tx_mat with Absorbing states
    death_states = (emp_r_mat_output.sum(axis=0).sum(axis=0) < 0)
    disch_states = (emp_r_mat_output.sum(axis=0).sum(axis=0) > 0)

    est_tx_mat = np.zeros((n_actions, n_states_obs, n_states_obs))
    est_tx_mat[:, :-2, :-2] = np.copy(emp_tx_mat_output)

    death_states = np.concatenate([death_states, np.array([True, False])])
    disch_states = np.concatenate([disch_states, np.array([False, True])])
    assert est_tx_mat[:, death_states, :].sum() == 0
    assert est_tx_mat[:, disch_states, :].sum() == 0

    est_tx_mat[:, death_states, deadStateIdx] = 1
    est_tx_mat[:, disch_states, discStateIdx] = 1


    # Normalize
    nonzero_idx = est_tx_mat.sum(axis=-1) != 0
    est_tx_mat[nonzero_idx] /= est_tx_mat[nonzero_idx].sum(axis=-1, keepdims=True)

    ############ Construct the reward matrix, which is known ##################
    est_r_mat = np.zeros((n_actions, n_states_obs, n_states_obs))
    est_r_mat[:, :-2, :-2] = np.copy(emp_r_mat_output)
    est_r_mat[:, :, -2] = -1
    est_r_mat[:, :, -1] = 1


    ############ Construct the empirical prior on the initial state ##################
    initial_state_arr = np.copy(states[:, 0, 0])
    initial_state_counts = np.zeros((n_states_obs,1))
    for i in range(initial_state_arr.shape[0]):
        initial_state_counts[initial_state_arr[i]] += 1

    emp_p_initial_state = (initial_state_counts / initial_state_counts.sum()).T


    # Because some SA pairs are never observed, assume they cause instant death
    zero_sa_pairs = est_tx_mat.sum(axis=-1) == 0
    est_tx_mat[zero_sa_pairs, -2] = 1  # Always insta-death if you take a never-taken action


    # Construct an extra axis for the mixture component, of which there is only one
    emperical_MDP = cf.MatrixMDP(est_tx_mat, est_r_mat,
                           p_initial_state=emp_p_initial_state)
    # try:
    MDP_pol = emperical_MDP.policyIteration(discount=DISCOUNT_Pol)
    # except:
    #     assert np.allclose(proj_tx_mat.sum(axis=-1), 1)
    #     MDP_pol = projMDP.policyIteration(discount=DISCOUNT_Pol, skip_check=True)

    return MDP_pol

def evaluate_all_policies(foldername=None, filename=None):
    if foldername != None:
        for filename in os.listdir(foldername):
            if filename[-3:] == 'csv':
                policy2evaluate = get_bcpg_policy(foldername + filename)
                print("{} average reward and std: {}".format(filename, evaluate_policy(policy2evaluate)))

    elif filename != None:
        policy2evaluate = get_bcpg_policy(filename)
        print("{} average reward and std: {}".format(filename, evaluate_policy(policy2evaluate)))

    return 0

def smarprimer_wis(num_trajectories):
    policy, mdp, r_mat = get_mdp()
    beh_policy = marginalize_policy(policy, EPS)
    dataset, output = create_data_set(num_trajectories, beh_policy)

    # Add Allen's splits in a loop here
    opes=[]
    for split_i in range(10):
        if split_i % 3 == 0:
            print("Starting split {}".format(split_i))

        split_train = pd.read_csv('sontag_sepsis_folder_v2/train_df_sepsis_{}_split{}.csv'.format(num_trajectories, split_i))
        split_test = pd.read_csv('sontag_sepsis_folder_v2/test_df_sepsis_{}_split{}.csv'.format(num_trajectories, split_i))

        eval_policy = get_empirical_mdp(beh_policy, num_trajectories, True, output, split_train)
        pibs, pies, rewards, lengths, MAX_TIME = compute_is_weights_for_mdp_policy(split_test, eval_policy, reward_column='Reward')
        ope_scores, ess = wis_ope(pibs, pies, rewards, lengths, max_time=MAX_TIME)
        opes.append(ope_scores)

    print("For num trajectories: {}".format(num_trajectories))
    print("mean ope's: {}.     Std of ope's: {}".format(np.mean(opes), np.std(opes)))

if __name__ == "__main__":
    EPS = 0.15

    # df = pd.read_csv('sepsis_5000.csv')
    # smarprimer_wis(200)
    # smarprimer_wis(1000)
    # smarprimer_wis(5000)
    policy, mdp, r_mat = get_mdp()
    marginalized_policy = marginalize_policy(policy, EPS)
    # evaluate_all_policies()
    # random_policy = np.ones((1440, 8)) * (1/8)
    # print("Optimal policy average reward and std: {}".format(evaluate_policy(policy)))
    print("Marginalized policy average reward with {}% random actions and std: {}".format(EPS*100, evaluate_policy(marginalized_policy),))
    # print("Random policy average rewar d and std: {}".format(evaluate_policy(random_policy)))
    # # bcpg_policy = get_bcpg_policy("all_states_with_prob_from_bcpg_model.csv")
    # # print("BCPG policy total reward: {}".format(evaluate_policy(bcpg_policy)))
    # # # all_states = get_all_states()
    # # a=1
    # # all_states.to_csv("all_states.csv")
    #
    #
    dataset, _ = create_data_set(200, marginalized_policy, high_vol=True)
    dataset.to_csv("high_vol_sepsis_200_w_difficulty_15_random_extreme.csv")
    print("200 was done")

    dataset, _ = create_data_set(1000, marginalized_policy, high_vol=True)
    dataset.to_csv("high_vol_sepsis_1000_w_difficulty_15_random_extreme.csv")
    print("1000 was done")

    dataset, _ = create_data_set(5000, marginalized_policy, high_vol=True)
    dataset.to_csv("high_vol_sepsis_5000_w_difficulty_15_random_extreme.csv")
    print("5000 was done")
    # empirical_mdp_policy = get_empirical_mdp(marginalized_policy, 200)
    # print("Empirical MDP policy average reward on 200 and std: {}".format(
    #     evaluate_policy(empirical_mdp_policy, policy_type='obs')))
    #
    # empirical_mdp_policy = get_empirical_mdp(marginalized_policy, 1000)
    # print("Empirical MDP policy average reward on 1000 and std: {}".format(
    #     evaluate_policy(empirical_mdp_policy, policy_type='obs')))
    #
    # empirical_mdp_policy = get_empirical_mdp(marginalized_policy, 5000)
    # print("Empirical MDP policy average reward on 5000 and std: {}".format(
    #     evaluate_policy(empirical_mdp_policy, policy_type='obs')))
    #
    # empirical_mdp_policy = get_empirical_mdp(marginalized_policy, 10000)
    # print("Empirical MDP policy average reward on 5000 and std: {}".format(
    #     evaluate_policy(empirical_mdp_policy, policy_type='obs')))

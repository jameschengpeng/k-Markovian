import numpy as np
import math
import statistics
import random
import scipy.stats as st
import pandas as pd

# given a probability distribution for the next step
# return the decision for the next step
def next_step_decision(prob_distribution):
    decision = np.random.uniform(0,1)
    summation = 0
    for i,p in enumerate(prob_distribution):
        summation += p
        if summation >= decision:
            return i

# extract the transition probability from a dataframe of transition rules
# in trajectory, count from 0
def get_prob_from_df(df, trajectory, k_step):
    reindex_trajectory = [i+1 for i in trajectory]
    n_explored = len(set(reindex_trajectory))
    if len(reindex_trajectory) > k_step:
        partial = True
        traj = tuple(reindex_trajectory[-k_step:])
    else:
        partial = False
        traj = tuple(reindex_trajectory)
    try:
        return df[(df['n_explored']==n_explored) & (df['partial']==partial) & (df['trajectory']==str(traj))].iloc[0,3:].to_numpy()
    except:
        print(traj)

# given a table of transition rules, compute a sample rendezvous time
def sample_rendezvous(df, n_cafe, k_step):
    # the labellings for each agent. {agent's idx: real idx}
    agent1 = {i: None for i in range(n_cafe)}
    agent2 = {i: None for i in range(n_cafe)}
    # first day for each agent
    init1 = np.random.randint(n_cafe)
    init2 = np.random.randint(n_cafe)
    agent1[0] = init1
    agent2[0] = init2
    actual_visited1 = [init1]
    actual_visited2 = [init2]
    if init1 == init2:
        return 1
    else:
        all_cafes = set(list([i for i in range(n_cafe)]))
        # the trajectory is indexed by their OWN labelling
        trajectory1 = [0]
        trajectory2 = [0]
        timer = 1
        while agent1[trajectory1[-1]] != agent2[trajectory2[-1]]:
            prob1 = get_prob_from_df(df, trajectory1, k_step)
            next1 = next_step_decision(prob1)
            # to go to an unvisited one
            label1 = len(set(trajectory1))
            if next1 >= label1:
                unvisited1 = list(all_cafes - set(actual_visited1))
                # randomly pick a cafe from those that are unvisited (actual cafe)
                next1_actual = random.choice(unvisited1)
                # update agent1's labelling
                agent1[label1] = next1_actual
                trajectory1.append(label1)
                actual_visited1.append(next1_actual)
            # go to a cafe that is already visited
            else:
                trajectory1.append(next1)

            prob2 = get_prob_from_df(df, trajectory2, k_step)
            next2 = next_step_decision(prob2)
            # to go to an unvisited one
            label2 = len(set(trajectory2))
            if next2 >= label2:
                unvisited2 = list(all_cafes - set(actual_visited2))
                # randomly pick a cafe from those that are unvisited (actual cafe)
                next2_actual = random.choice(unvisited2)
                # update agent1's labelling
                agent2[label2] = next2_actual
                trajectory2.append(label2)
                actual_visited2.append(next2_actual)
            # go to a cafe that is already visited
            else:
                trajectory2.append(next2)
            timer += 1
    return timer

# check the expected rendezvous time based on a dataframe of transition rules
def rendezvous_exp_check(df, n_cafe, k_step, ci_width = 0.05):
    sample_rendezvous_time = list()
    counter = 0
    conf_interval = (-np.infty, np.infty)
    while conf_interval[1]-conf_interval[0] > ci_width:
        sample_rendezvous_time.append(sample_rendezvous(df, n_cafe, k_step))
        if counter % 100 == 0 and counter > 0:
            conf_interval = st.t.interval(alpha = 0.95, df = len(sample_rendezvous_time)-1, 
                                          loc = np.mean(sample_rendezvous_time), 
                                          scale = st.sem(sample_rendezvous_time))
        counter += 1
    # print(counter)
    sample_rendezvous_time = np.array(sample_rendezvous_time)
    return np.mean(sample_rendezvous_time), conf_interval, counter

# get the probability distribution given the extracted parameters
# first do tanh to each element, then compute by softmax
def get_prob(param):
    processed = [math.exp(math.tanh(p)) for p in param]
    denominator = sum(processed)
    return [p/denominator for p in processed]

# generate all possible trajectories by recursion
# count from 0
def generate_trajectory(n_explored, len_trajectory):
    # base case
    if len_trajectory == 1:
        return [[i] for i in range(n_explored)]
    # recursive relation
    else:
        last_result = generate_trajectory(n_explored, len_trajectory-1)
        new_result = list()
        for t in last_result:
            for i in range(n_explored):
                new_t = (t + [i]).copy()
                new_result.append(new_t)
        return new_result

# check if a trajectory is valid
def check_trajectory(trajectory, n_explored):
    # visited n_explored number of distinct cafes
    if len(set(trajectory)) != n_explored:
        return False
    # check cases like: you visit cafe 2 before cafe 1
    current_max = -1
    for c in trajectory:
        if c > current_max + 1:
            return False
        if c == current_max + 1:
            current_max = c
    return True
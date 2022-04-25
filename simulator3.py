import numpy as np
import math
import statistics
import random
import scipy.stats as st
import pandas as pd
import tool_box
from tool_box import get_prob, next_step_decision, generate_trajectory, check_trajectory

# in this version, the cafes are labelled by the order of visiting
# the first one you visit is labelled as one, the second distinct one you visited is labelled as two, and so on


# to store the model parameters
class k_markov_model:
    # initialize the parameters in kmm 
    def __init__(self, k_step, n_cafe, random_init = False, ci_width = 0.05):
        self.k_step = k_step
        self.n_cafe = n_cafe
        self.random_init = random_init
        self.ci_width = ci_width
        # store all parameters in a list
        self.params = list()
        # i represents the level of exploration (number of distinct cafes you have visited)
        for i in range(self.n_cafe):
            # your visiting history includes no more than k distinct cafes, the length of visiting history might be <= k
            if i+1 <= self.k_step:
                dimensions = [tuple([i+1 for t in range(j)]+[min(i+2, self.n_cafe)]) for j in range(i+1,self.k_step+1)]
            # your visiting history includes more than k distinct cafes, so the length of visiting history > k
            else:
                dimensions = [tuple([i+1 for t in range(self.k_step)]+[self.n_cafe])]
            # the parameters in this level of exploration
            params_level = list()
            for dim in dimensions:
                if self.random_init:
                    params_level.append(np.random.uniform(-1,1,dim))
                else:
                    params_level.append(np.ones(dim))
            self.params.append(params_level.copy())
        self.init_params = self.params.copy()
        self.best_param = None
        self.best_CI = None

    # given a trajectory, extract the probability distribution over all cafes
    # give equal probabilities to unvisited cafes
    # long_traj means that this trajectory is a sample long_trajectory, level is specified
    def extract_prob(self, trajectory, use_best = False, long_traj = False, level_long_traj = 1):
        distinct_cafes = set(trajectory)
        # level of exploration
        if not long_traj:
            level = len(distinct_cafes)
        else:
            level = level_long_traj
        # use full trajectory to determine
        if len(trajectory) <= self.k_step:
            if not use_best:
                use_param = self.params[level-1][len(trajectory)-level].copy()
            else:
                use_param = self.best_param[level-1][len(trajectory)-level].copy()
            for t in trajectory:
                use_param = use_param[t]
        # use partial trajectory to determine
        else:
            if not use_best:
                use_param = self.params[level-1][-1].copy()
            else:
                use_param = self.best_param[level-1][-1].copy()
            for t in trajectory[-self.k_step:]:
                use_param = use_param[t]
        if level <= self.n_cafe - 2:
            prob = get_prob(use_param)
            n_unvisit = self.n_cafe - level
            unvisited_prob = [prob[-1]/n_unvisit for i in range(n_unvisit)]
            return prob[:-1] + unvisited_prob
        else:
            prob = get_prob(use_param)
            return prob

    # to compute a sample rendezvous time
    def sample_run(self, use_best = False):
        # the labellings of each agent. {agent's idx: real idx}
        agent1 = {i: None for i in range(self.n_cafe)}
        agent2 = {i: None for i in range(self.n_cafe)}
        # first day for each agent
        init1 = np.random.randint(self.n_cafe)
        init2 = np.random.randint(self.n_cafe)
        agent1[0] = init1
        agent2[0] = init2
        actual_visited1 = [init1]
        actual_visited2 = [init2]
        if init1 == init2:
            return 1
        all_cafes = set(list([i for i in range(self.n_cafe)]))
        # the trajectory is indexed by their OWN labelling
        trajectory1 = [0]
        trajectory2 = [0]
        timer = 1
        while agent1[trajectory1[-1]] != agent2[trajectory2[-1]]:
            prob1 = self.extract_prob(trajectory1, use_best)
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

            # now, do the same for agent2
            prob2 = self.extract_prob(trajectory2, use_best)
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
            # update timer
            timer += 1
        return timer

    # repeat the sample run and obtain the average rendezvous time
    def loss(self, use_best = False):
        sample_rendezvous = list()
        counter = 0
        conf_interval = (-np.infty, np.infty)
        while conf_interval[1]-conf_interval[0] > self.ci_width:
            sample_rendezvous.append(self.sample_run(use_best))
            if counter % 100 == 0 and counter > 0:
                conf_interval = st.t.interval(alpha=0.95, df=len(sample_rendezvous)-1, 
                                              loc=np.mean(sample_rendezvous), 
                                              scale=st.sem(sample_rendezvous))
            counter += 1
        # print(counter)
        sample_rendezvous = np.array(sample_rendezvous)
        return np.mean(sample_rendezvous), conf_interval

    # reset the kmm simulator
    def reset(self):
        self.params = self.init_params.copy()
    
    # compute the transition probability based on the best param
    def transition_prob(self):
        df = pd.DataFrame(columns = ['n_explored', 'partial', 'trajectory'] + [str(i+1) for i in range(self.n_cafe)])
        # iterate all level of exploration
        for n in range(self.n_cafe):
            n_explored = n+1
            # iterate all trajectory length
            for t in range(self.k_step):
                len_trajectory = t+1
                # length of trajectory is at least the number of visited cafes
                if len_trajectory >= n_explored:
                    # generate all trajectories
                    all_traj = generate_trajectory(n_explored, len_trajectory)
                    for traj in all_traj:
                        # if n_explored <= n_cafe, need to check if this trajectory is valid
                        if n_explored <= self.n_cafe and check_trajectory(traj, n_explored):
                            transition_prob = self.extract_prob(traj, use_best = True)
                            new_row = dict()
                            new_row['n_explored'] = n_explored
                            new_row['partial'] = False
                            new_row['trajectory'] = tuple([c + 1 for c in traj])
                            for i in range(self.n_cafe):
                                new_row[str(i+1)] = transition_prob[i]
                            df = df.append(new_row, ignore_index=True)
        long_trajectories = generate_trajectory(self.n_cafe, self.k_step)
        for traj in long_trajectories:
            # no need to check if this is a valid trajectory. Beyond k_step, everything is possible
            # in traj, count from 0, but for level, count from 1, so add 1
            level = max(traj) + 1
            if level == self.n_cafe:
                # add one more element to traj so that it uses the correct parameter
                new_row = dict()
                new_row['n_explored'] = level
                new_row['trajectory'] = tuple([c+1 for c in traj])
                augmented_traj = ([0] + traj).copy()
                transition_prob = self.extract_prob(traj, use_best = True, long_traj=True, level_long_traj = level)
                new_row['partial'] = True
                for i in range(self.n_cafe):
                    new_row[str(i+1)] = transition_prob[i]
                df = df.append(new_row, ignore_index=True)
            else:
                last_k_step = tuple([c+1 for c in traj])
                original_traj = traj.copy()
                for l in range(level, self.n_cafe+1):
                    new_row = dict()
                    new_row['n_explored'] = l
                    new_row['trajectory'] = last_k_step
                    augmented_traj = ([i for i in range(l)] + original_traj).copy()
                    transition_prob = self.extract_prob(augmented_traj, use_best = True, long_traj=True, level_long_traj=l)
                    new_row['partial'] = True
                    for i in range(self.n_cafe):
                        new_row[str(i+1)] = transition_prob[i]
                    df = df.append(new_row, ignore_index=True)

        df = df.drop_duplicates()
        return df
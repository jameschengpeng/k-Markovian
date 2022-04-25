import numpy as np
import simulator3
from simulator3 import k_markov_model
import math
import random
import statistics

# general spsa optimizer for k_markov_model
class spsa_optimizer:
    def __init__(self, max_itr = 200, perturbation_scale = 1, alpha = 0.602, gamma = 0.101):
        self.max_itr = max_itr
        self.perturbation_scale = perturbation_scale
        self.alpha = alpha
        self.gamma = gamma
        self.A = 0.1 * max_itr


    # do a perturbation and evaluate the loss
    # pass in the kmm_model
    def perturbation_eval(self, kmm_model):
        # params is a list of list of numpy arrays
        # create two new_params, one add perturbation, one minus perturbation
        new_params_1 = list()
        new_params_2 = list()
        c_k = self.c/((self.itr+1)**self.gamma)
        perturb = list()
        for i in range(len(kmm_model.params)):
            # params for a certain level of exploration (number of distinct cafes that have been visited)
            new_params_1.append(list())
            new_params_2.append(list())
            perturb.append(list())
            for j in range(len(kmm_model.params[i])):
                # generate perturbation array
                dim = kmm_model.params[i][j].shape
                perturbation = np.random.uniform(0, 1, dim)
                perturbation[perturbation<=0.5] = -self.perturbation_scale
                perturbation[perturbation>0.5] = self.perturbation_scale
                new_params_1[i].append(kmm_model.params[i][j].copy() + c_k * perturbation.copy())
                new_params_2[i].append(kmm_model.params[i][j].copy() - c_k * perturbation.copy())
                perturb[i].append(perturbation.copy())
        new_model_1 = k_markov_model(kmm_model.k_step, kmm_model.n_cafe)
        new_model_2 = k_markov_model(kmm_model.k_step, kmm_model.n_cafe)

        new_model_1.params = new_params_1.copy()
        new_model_2.params = new_params_2.copy()

        new_loss_1, ci1 = new_model_1.loss()
        new_loss_2, ci2 = new_model_2.loss()
        del new_model_1, new_model_2
        return new_loss_1, new_loss_2, ci1, ci2, perturb, c_k, new_params_1, new_params_2

    # pass in the model.params into this function
    # set additional as True if you want additional iterations
    def optimize(self, kmm_model, a = 0.1, c = 0.07, additional=False, additional_itr=100):
        self.a = 0.1
        self.c = 0.07
        if additional == False:
            self.itr = 0
            current_loss, conf_interval = kmm_model.loss()
            self.least_loss = current_loss
            kmm_model.best_param = kmm_model.params.copy()
            self.loss_history = [(current_loss, conf_interval)]
            n_itr = self.max_itr
            self.itr_up_to_now = 0
        else:
            n_itr = additional_itr
        itr = 0
        while itr <= n_itr:
            self.itr_up_to_now += 1
            if additional == False:
                self.itr = itr
            else:
                self.itr = self.itr_up_to_now + itr
            if itr % (int(n_itr/10)) == 0:
                print(str(itr/(int(n_itr/10))*10) + " percent done")

            new_loss_1, new_loss_2, ci1, ci2, perturb, c_k, new_params_1, new_params_2 = self.perturbation_eval(kmm_model)
            if new_loss_1 < self.least_loss:
                self.least_loss = new_loss_1
                kmm_model.best_CI = ci1
                kmm_model.best_param = new_params_1.copy()

            if new_loss_2 < self.least_loss:
                self.least_loss = new_loss_2
                kmm_model.best_CI = ci2
                kmm_model.best_param = new_params_2.copy()

            del new_params_1, new_params_2

            multiple = (new_loss_1 - new_loss_2)/(2*c_k)
            gradient_est = list()

            for i in range(len(perturb)):
                gradient_est.append(list())
                for j in range(len(perturb[i])):
                    grad = multiple * np.reciprocal(perturb[i][j].copy())
                    gradient_est[i].append(grad.copy())

            a_k = self.a/((self.A+self.itr+1)**self.alpha)
            self.update_params(kmm_model, a_k, gradient_est)
            current_loss, conf_interval = kmm_model.loss()
            if current_loss < self.least_loss:
                self.least_loss = current_loss
                kmm_model.best_CI = conf_interval
                kmm_model.best_param = kmm_model.params
            self.loss_history.append(kmm_model.loss())
            
            itr += 1

    def update_params(self, model, multiplier, adder):
        for i in range(len(adder)):
            for j in range(len(adder[i])):
                model.params[i][j] = model.params[i][j] - multiplier * adder[i][j].copy()
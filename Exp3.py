# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 13:46:22 2021

@author: Xiaotong
"""

import numpy as np 
import math
import Gaussian_noise as gn

class Exp3:
    def __init__(self, setting, rho, kappa, T):
        """
        :param setting: 0-no combinatorial setting; 1-naive combination of arms
        :param rho: the matrix of perfomrance index
        :param kappa: the matrix of basic cost of tasks
        :param T: number of trial rounds
        """
        self.rho = rho
        self.kappa = kappa
        self.T = T
        self.K = rho.shape[0]       #the number of players
        self.M = rho.shape[1]       #the number of tasks
        self.comb = setting
    
    def draw(self,probability_distribution):
        varm = np.zeros((self.K,1))
        for i in range (0,self.K):
            varm[i] = np.random.choice(range(0,probability_distribution.shape[1]), size=1, p=probability_distribution[i])
        return varm
    
    def tact(self,varm):
        b = np.zeros((self.K,self.M))
        for i in range(0,self.K):
            v = varm[i]
            for j in range (0,self.M):
                b[i][j] = v%2
                v = int(v/2)
        return b
        
    def obtain_utility(self,arm):
        alpha = 1.1;
        beta = 0.03;
    
        u = np.zeros((self.K,1))
    
        arm_sum = np.sum(arm,axis = 0)
        act = np.zeros((self.K,self.M))
        for i in range(0,self.K):
            for j in range (0,self.M):
                if abs(arm_sum[j] - 0) < 0.01 :
                    u[i] += 0
                else:
                    act[i][j] = float(arm[i][j]/arm_sum[j])
                    u[i] += (alpha*self.rho[i][j]*(1-math.exp(-act[i][j]/self.rho[i][j])) -self.kappa[i][j]*arm[i][j])
        
            noise = gn.Gaussian_noise(1,0,1,[-1,1])
            u[i] += noise.sample_trunc()
        
        u -= beta*(1-np.sum(act)/self.M)
        return u

    def distr(self,weights, gamma=0.0):
        weight_sum = np.sum(weights,axis = 1)
        p = np.zeros(weights.shape)
        for i in range(0,p.shape[0]):
            for j in range (0, p.shape[1]):
                p[i][j] = (1.0-gamma)*(weights[i][j] /  weight_sum[i]) + gamma/weights.shape[1]
            
        return p

    def update_weights(self, weights, probability_distribution, utility, varm, gamma):
        update_rewards = np.zeros(probability_distribution.shape)
        for i in range (probability_distribution.shape[0]):
            idx = int(varm[i])
            update_rewards[i][idx] = utility[i]/probability_distribution[i][idx]
            weights[i][idx] = weights[i][idx]*math.exp(gamma*update_rewards[i][idx]/probability_distribution.shape[1])
        return weights
    
    def run(self):
        if self.comb > 0:
            omega = np.ones((self.K,2**self.M))
        else:
            omega = np.ones((self.K, self.M))
        r = []
        sumr = 0
        for i in range(1,self.T+1):
            gamma = min(1/omega.shape[1],np.sqrt(np.log(omega.shape[1])/(omega.shape[1]*i)))
            probability_distribution = self.distr(omega,gamma)
            varm = self.draw(probability_distribution)
            if self.comb > 0:
                arm = self.tact(varm)
            else:
                arm = varm
            utility = self.obtain_utility(arm)
            #print('u', utility)
            omega = self.update_weights(omega,probability_distribution,utility,varm,gamma)
            #print('omega',omega[0])
            sumr += np.sum(utility)
            avgr = sumr/i
            r.append(avgr)
        return r


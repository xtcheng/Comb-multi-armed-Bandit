# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 16:30:39 2021

@author: Xiaotong
"""

import itertools
import numpy as np 
import math
import Gaussian_noise as gn

class ComBand:
    def __init__(self,ncom, rho, kappa, T):
        """
        :param ncom: combinatorial setting with maximum number n
        :param rho: the matrix of perfomrance index
        :param kappa: the matrix of basic cost of tasks
        :param T: number of trial rounds
        """
        self.rho = rho
        self.kappa = kappa
        self.T = T
        self.K = rho.shape[0]       #the number of players
        self.M = rho.shape[1]       #the number of tasks
        self.n = ncom               #the maximum number of combination
    
    def action_list(self):
        action_list = list(itertools.combinations(range(0,self.M+self.n-1),self.n))
        return action_list
    
    def distr(self, weights, gamma):
        weight_sum = np.sum(weights,axis = 1)
        p = np.zeros(weights.shape)
        for i in range(0,p.shape[0]):
            for j in range (0, p.shape[1]):
                p[i][j] = (1.0-gamma[i])*(weights[i][j] /  weight_sum[i]) + gamma[i]/weights.shape[1]
            
        return p
    
    def draw(self, probability_distribution):
        varm = np.zeros((self.K,1))
        for i in range (0,self.K):
            varm[i] = np.random.choice(range(0,probability_distribution.shape[1]), size=1, p=probability_distribution[i])
        return varm
    
    def find_action(self,ract,action):
        true_action = np.zeros((self.K,self.M))
        for i in range(0,self.K):
            rank = int(ract[i])
            #print('rank',rank)
            for j in range(0,self.n):
                if action[rank][j] < self.M:
                    true_action[i][action[rank][j]] = 1
        return true_action  
    
    def obtain_utility(self,tact):
        alpha = 1.0
        u = np.zeros((self.K,1))
    
        arm_sum = np.sum(tact,axis = 0)
        act = np.zeros((self.K,self.M))
        for i in range(0,self.K):
            for j in range (0,self.M):
                act[i][j] = float(tact[i][j]/(arm_sum[j]-tact[i][j]+1))
                u[i] += alpha*math.log(1+act[i][j]*(10*self.rho[i][j])) -self.kappa[i][j]*tact[i][j]
        
            noise = gn.Gaussian_noise(1,0,0.01,[-0.1,0.1])
            u[i] += noise.sample_trunc()
        return u
    
    def update_weights(self,weights, probability_distribution,action,ract,rew,eta):
        P = np.zeros((self.K,weights.shape[1],weights.shape[1]))
        l = np.zeros((self.K,weights.shape[1],weights.shape[1]))
        hat_l = np.zeros((self.K,weights.shape[1]))
        ls=np.zeros((self.K,weights.shape[1]))
        sum_l = np.zeros((self.K,weights.shape[1]))
        for i in range(0,self.K):
            for j in range(0,weights.shape[1]):
                for q in range(0,self.n):
                    l[i][j][action[j][q]] = 1
                    ls[i][action[int(ract[i])][q]] = 1
                P[i] += probability_distribution[i][j]*np.outer(l[i][j],l[i][j])
            lam, v = np.linalg.eig(P[i])
            hat_l[i] = (self.n-rew[i])*np.linalg.pinv(P[i]).dot(ls[i])
        #print('hat_l',hat_l[i])
        for i in range(0,self.K):
            for j in range(0,weights.shape[1]):
                for q in range(0,self.n):
                    #print('a',action[j][q])
                    sum_l[i][j] += hat_l[i][action[j][q]]
                    #print('suml',i,j,sum_l[i][j])
                    #omega[i][j] = omega[i][j]*math.exp(-gamma*(weights.shape[1]-n)*sum_l[i][j]/(n*weights.shape[1]*(weights.shape[1]-1)))
                weights[i][j] = weights[i][j]*math.exp(-eta[i]*sum_l[i][j])
            
        return weights

    def cal_mineig(self,probability_distribution,action):
        P = np.zeros((self.K,len(action),len(action)))
        l = np.zeros((self.K,len(action),len(action)))
        mineig = np.zeros((self.K,1))
        for i in range(0,self.K):
            for j in range(0,len(action)):
                for q in range(0,self.n):
                    l[i][j][action[j][q]] = 1
                P[i] += probability_distribution[i][j]*np.outer(l[i][j],l[i][j])
            lam, v = np.linalg.eig(P[i])
            mineig[i] = np.amin(lam)
        return mineig
    
    def run(self):
        action = self.action_list()
        #print('action',action)
        omega = np.ones((self.K,len(action)))
        r = [] 
        #a = []
        sumr = 0
        eta = np.zeros((self.K,1))
        gamma = 0.193*np.ones((self.K,1))
        for i in range(1,self.T):
            #gamma = min(1/len(action),np.sqrt(np.log(len(action))/(len(action)*i)))
            probability_distribution = self.distr(omega,gamma)
            #print('p',probability_distribution)
            ract = self.draw(probability_distribution)
            #print('rank of action',ract)
            tact = self.find_action(ract,action)
            #print('true action',tact)
            rew = self.obtain_utility(tact)
            #print('rew',rew)
            #pdb.set_trace()
            mineig = self.cal_mineig(probability_distribution,action)
            for j in range(0,self.K):
                eta[j] = 1./math.sqrt(self.n)*math.sqrt(math.log(len(action))/(i*(self.M/self.n+2./mineig[j])))
                gamma[j] = math.sqrt(self.n)/mineig[j]*math.sqrt(math.log(len(action))/(i*(self.M/self.n+2./mineig[j])))
                #print('gamma',gamma)
            omega = self.update_weights(omega,probability_distribution,action,ract,rew,eta)
            sumr += np.sum(rew)
            avgr = sumr/i
            r.append(avgr)
        return r
    
    
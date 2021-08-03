# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 10:10:14 2021

@author: Xiaotong
"""
import itertools
import numpy as np 
import math
import random

class Exp3M:
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
        
    def obtain_alpha(self,gamma,omega):
        omega_sum = np.sum(omega,axis = 1)
        rhs = (float(1/self.n) - gamma/(self.M+self.n-1))/(1-gamma)
        alpha = np.zeros((self.K,1))
        omega_sorted =  -np.sort(-omega,axis = 1)
        for i in range(0,self.K):
            if np.max(omega[i])>=rhs*omega_sum[i]:
                cur = 0
                for j in range(0,self.M+self.n-1):
                    #pdb.set_trace()
                    for k in range(j+1,self.M+self.n-1):
                        cur += omega_sorted[i][k]
                    alpha[i] = rhs*cur/(1.-rhs*(j+1))
                    idx = np.argmax(omega_sorted[i] > alpha[i])
                    #if omega_sum[i] > 30000:
                    #pdb.set_trace()
                    if abs(idx -j) < 0.01:
                        break
        return alpha
    
    def modif_omeg(self, alpha,weights,count):
        if count > 2400:
            print('alp',alpha)
        omega_n = weights.copy()
        for k in range(0,weights.shape[0]):
            if alpha[k] > 0.95:
                for j in range(weights.shape[1]):
                    if weights[k][j] >= alpha[k]:
                        omega_n[k][j] = alpha[k]
                    else:
                        omega_n[k][j] = weights[k][j]      
        return omega_n
        
    def distr(self,weights, gamma):
        weight_sum = np.sum(weights,axis = 1)
        p = np.zeros(weights.shape)
        for i in range(0,p.shape[0]):
            for j in range (0, p.shape[1]):
                p[i][j] = self.n*((1.0-gamma)*(weights[i][j] /  weight_sum[i]) + gamma/weights.shape[1])
        return p
    
    def depround(self, probability_distribution,count):
        p = probability_distribution.copy()
        for j in range(p.shape[0]):
            a = np.any(np.logical_and(p[j]>0.01 , p[j]<0.99))
            while a:
                #pdb.set_trace()
                iset = np.where(np.logical_and(p[j]>0.01,p[j]<0.99))[0].tolist()
                #print('iset', j, iset ,len(iset))
                #ni = np.random.choice(iset[0],k=2)
                # ni = random.choices(iset[0],k=1)
                # nj = random.choices(iset[0].remove)
                ni = []
                while(len(ni)<2 and len(iset)>1):
                    ni1 = random.choices(iset,k=1)
                    #print(ni1)
                    if ni1 not in ni:
                        ni.append(ni1)
                
                alp = min(1-p[j][ni[0]],p[j][ni[1]])[0]
                beta = min(1-p[j][ni[1]],p[j][ni[0]])[0]
                if count > 2400:
                    ('p before update',p)
                    print('slct',count,ni)
                    #pdb.set_trace()
                upd = np.random.choice([0,1], size=1, p=[beta/(alp+beta),alp/(alp+beta)])
                # if alp > beta:
                    #     upd = -0.1
                # else:
                    #     upd = 1
                if upd < 1:
                    p[j][ni[0]] += alp
                    p[j][ni[1]] -= alp
                else:
                    p[j][ni[0]] -= beta
                    p[j][ni[1]] += beta
                if count > 2400:
                    print('p_in_dep',p)
                    #pdb.set_trace()
                a = np.any(np.logical_and(p[j]>0.01 , p[j]<0.99))
                if count > 2400:
                    print('a',a)
                #seti.append(np.where(abs(p[i] -1)< 0.01))
            p[j] = np.around(p[j])
            if count > 2400:
                print('p_in_dep_new',p)        
        return p
    
    def obtain_utility(self,tact):
        u = np.zeros((self.K,self.M))
    
        arm_sum = np.sum(tact,axis = 0)
        act = np.zeros((self.K,self.M))
        for i in range(0,self.K):
            for j in range (0,self.M):
                act[i][j] = float(tact[i][j]/(arm_sum[j]-tact[i][j]+1))
                u[i][j] = 1.0*math.log(1+act[i][j]*(10*self.rho[i][j])) -self.kappa[i][j]*tact[i][j]
        return u

    def update_weights(self,weights,pidx, probability_distribution,utility,gamma):
        update_rewards = np.zeros(probability_distribution.shape)
        for i in range (probability_distribution.shape[0]):
            for j in range(probability_distribution.shape[1]):
                if pidx[i][j] > 0.001 and j < self.M:
                    update_rewards[i][j] = utility[i][j]/probability_distribution[i][j]
                weights[i][j] = weights[i][j]*math.exp(self.n*gamma*update_rewards[i][j]/(self.M+self.n-1))
        return weights
    
    def run(self):
        omega = np.ones((self.K,self.M+self.n-1));
        omega_n = omega.copy()
        r = []
        sumr = 0
        for i in range(1,self.T):
            #gamma = min(1,np.sqrt((M+n-1)*np.log((M+n-1)/n)/((math.exp(0)-1)*(M+n-1)*3000)))
            gamma = 0.0108621188
            alpha = self.obtain_alpha(gamma,omega)
            omega_n = self.modif_omeg(alpha,omega,i)   
            probability_distribution = self.distr(omega_n,gamma)
            if i > 2400:
                print('omega_n',omega_n)
                print('p1',probability_distribution)    
            arm = self.depround(probability_distribution,i)
            #print('arm',arm)
            tact = arm[:,0:self.M]
            #print('tact',tact)
            utility = self.obtain_utility(tact)
            #print('utility',utility)
            omega = self.update_weights(omega,arm,probability_distribution,utility,gamma)
            if i > 2400:
                print('omega_n',omega_n)
                print('p2',probability_distribution)
                print('alpha',alpha)
                print('omega',omega)
            sumr += np.sum(utility)
            avgr = sumr/i
            r.append(avgr)
        return r
    
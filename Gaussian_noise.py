# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 12:00:00 2021

@author: Xiaotong
"""
import numpy as np 

class Gaussian_noise:
    def __init__(self, num, avg , var, bounds):
        """
        :param num: the size of generated noise.
        :param avg: the average of the gaussian distribution
        :param var: the variance of the gaussian distribution
        :param bounds: the bound of the truncated gaussian noise
        """
        self.n =  num
        self.mu = avg
        self.sig = var
        self.bounds = bounds

    def normal(self,x):
        return 1. / (np.sqrt(2 * np.pi) * self.sig) * np.exp(-0.5 * np.square(x - self.mu) / np.square(self.sig))


    def trunc_normal(self,x):
        if self.bounds is None: 
            self.bounds = (-np.inf, np.inf)

        norm = self.normal(x)
        norm[x < self.bounds[0]] = 0
        norm[x > self.bounds[1]] = 0

        return norm

    def sample_trunc(self):
        """ Sample `n` points from truncated normal distribution """
        x = np.linspace(self.mu - 5. * self.sig, self.mu + 5. * self.sig, 10000)
        y = self.trunc_normal(x)
        y_cum = np.cumsum(y) / y.sum()

        yrand = np.random.rand(self.n)
        sample = np.interp(yrand, y_cum, x)

        return sample
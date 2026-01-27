# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 12:27:34 2025

@author: mthoma
"""
import numpy as np
import pandas as pd

class BernoulliBandit(object):
    def __init__(self, p):
        self.p = p
        
    def display_add(self):
        reward = np.random.binomial(n=1, p=self.p)
        return reward
    
adA = BernoulliBandit(0.004)
adB = BernoulliBandit(0.016)
adC = BernoulliBandit(0.02)
adD = BernoulliBandit(0.028)
adE = BernoulliBandit(0.031)

def main():
    
    ads = [adA, adB, adC, adD, adE]
    
    n_prod = 100000
    n_ads = len(ads)
    alphas = np.ones(n_ads)
    betas = np.ones(n_ads)
    total_reward = 0
    avg_rewards = []
    
    for i in range(n_prod):
        theta_samples = [np.random.beta(alphas[k], betas[k]) for k in range(n_ads)]
        ad_chosen = np.argmax(theta_samples)
        R = ads[ad_chosen].display_add()
        alphas[ad_chosen] += R
        betas[ad_chosen] += 1 - R
        total_reward += R
        avg_reward_so_far = total_reward / (i + 1)
        avg_rewards.append(avg_reward_so_far)
    
    df_reward_comparison = pd.DataFrame(avg_rewards, columns=['Thomson Sampling'])
    df_reward_comparison.plot() 
    

if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 18:29:20 2025

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
    
    
    
def main():
    
    adA = BernoulliBandit(0.004)
    adB = BernoulliBandit(0.016)
    adC = BernoulliBandit(0.02)
    adD = BernoulliBandit(0.028)
    adE = BernoulliBandit(0.031)
    
    ads = [adA, adB, adC, adD, adE]
    
    ### Training Phase

    
    epsilons = [0.5, .2, .1, .05, .01]
    
    df_reward_comparison = pd.DataFrame()
    
    for epsilon in epsilons:
        
        n_test = 10000
        #n_prod = 90000
        n_ads = len(ads)
        Q = np.zeros(n_ads)
        N = np.zeros(n_ads)
        total_reward = 0
        avg_rewards = []
    
        ad_chosen = np.random.randint(n_ads)
        
        for i in range(n_test):
            R = ads[ad_chosen].display_add()
            
            N[ad_chosen] += 1
            Q[ad_chosen] += (1 / N[ad_chosen]) * (R - Q[ad_chosen])
            total_reward += R
            avg_reward_so_far = total_reward / (i + 1)
            avg_rewards.append(avg_reward_so_far)
        
            ad_chosen = np.random.randint(n_ads) if np.random.uniform() <= epsilon else np.argmax(Q)
            
        
    
        #best_add_idx = np.argmax(Q)
        
        df_reward_comparison[f"epsilon={epsilon}"] = avg_rewards
    
    #print(f"The best performing add is: {chr(ord('A') + best_add_idx)}")
    
    '''
    ### PRODUCTION PERIOD
    
    ad_chosen = best_add_idx
    for i in range(n_prod):
        R = ads[ad_chosen].display_add()
        total_reward += R
        avg_reward_so_far = total_reward / (n_test + i + 1)
        avg_rewards.append(avg_reward_so_far)
        
    '''
    

    print(df_reward_comparison)
    
    df_reward_comparison.plot()

                   

if __name__ == "__main__":
    main()
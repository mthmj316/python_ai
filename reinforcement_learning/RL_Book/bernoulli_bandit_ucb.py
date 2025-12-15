# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 18:29:20 2025

@author: mthoma

UCB -> Upper Confidence Bound

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

    
    ucb_list = [0.01, 0.03, 0.6, 0.09]
    
    df_reward_comparison = pd.DataFrame()
    
    for ucb in ucb_list:
        
        n_test = 100000
        #n_prod = 90000
        n_ads = len(ads)
        ad_indices = np.array(range(n_ads))
        Q = np.zeros(n_ads)
        N = np.zeros(n_ads)
        total_reward = 0
        avg_rewards = []
    
        for t in range(n_test):
            
            if any(N==0):
                ad_chosen = np.random.choice(ad_indices[N==0])
            else:
                uncertainty = np.sqrt(np.log(t) / N)
                
                ad_chosen = np.argmax(Q + ucb * uncertainty)
            
            R = ads[ad_chosen].display_add()
            
            N[ad_chosen] += 1
            Q[ad_chosen] += (1 / N[ad_chosen]) * (R - Q[ad_chosen])
            total_reward += R
            avg_reward_so_far = total_reward / (t + 1)
            avg_rewards.append(avg_reward_so_far)            
        
    
        #best_add_idx = np.argmax(Q)
        
        df_reward_comparison[f"ucb={ucb}"] = avg_rewards
    
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
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
    
    n_test = 10000
    n_ads = len(ads)
    Q = np.zeros(n_ads)
    N = np.zeros(n_ads)
    total_reward = 0
    avg_rewards = []

    for i in range(n_test):
        
        # get randomly the index of the ad ("the chosen action a")
        ad_chosen = np.random.randint(n_ads)
        
        # get the reward for the chosen ad (action)
        R = ads[ad_chosen].display_add()
        
        # increment the selection counter of the chosen add by one 
        N[ad_chosen] += 1
        
        # Lernrate 
        alpha = (1 / N[ad_chosen])
        
        # estimated action value of the chosen ad (action) after
        # the last selection
        Qn =  Q[ad_chosen]        
        
        # new etsimate action value for the chosen ad (action)
        Qnp1 = Qn + alpha * (R - Qn)
        
        # update the action value estimation for the chosen ad (action) in Q[] 
        Q[ad_chosen] = Qnp1
        
        # short form of the Q handling the lines before
        # Q[ad_chosen] += (1 / N[ad_chosen]) * (R - Q[ad_chosen])
        
        # rewards over all chosen adds (actions)
        total_reward += R
        
        # 
        avg_reward_so_far = total_reward / (i + 1)
        avg_rewards.append(avg_reward_so_far)
        
    
    best_add_idx = np.argmax(Q)
    
    print(f"The best performing add is: {chr(ord('A') + best_add_idx)}")
    
    '''
    ### PRODUCTION PERIOD
    
    ad_chosen = best_add_idx
    for i in range(n_prod):
        R = ads[ad_chosen].display_add()
        total_reward += R
        avg_reward_so_far = total_reward / (n_test + i + 1)
        avg_rewards.append(avg_reward_so_far)
        
    '''
    
    df_reward_comparison = pd.DataFrame(avg_rewards, columns=['A/B/n'])
    df_reward_comparison.plot()                   

if __name__ == "__main__":
    main()
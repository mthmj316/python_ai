# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 20:29:27 2026

@author: mthoma
"""

import numpy as np

class LineasThompsonSampling:
    def __init__(self, n_arms, n_features, alpha=1.0):
        self.n_arms = n_arms
        self.n_features = n_features
        self.alpha = alpha
        
        # One Bayesian linear model per arm
        self.A = [np.eye(n_features) for _ in range(n_arms)]
        print('A=', self.A)
        self.b = [np.zeros(n_features) for _ in range(n_arms)]
        print('b=', self.b)
        
    def select_arm(self, x):
        samples = []
        
        for a in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[a])            
            print('A_inv=',A_inv)
            
            mu = A_inv @ self.b[a]         
            print('mu=',mu)
            
            sigma = self.alpha * A_inv
            
            # Thompson sampling: sample weights
            w = np.random.multivariate_normal(mu, sigma)                   
            print('w=',w)
            
            samples.append(x @ w)                
            print('samples=',samples)
            
        return np.argmax(samples)
            
    def update(self, arm, x, reward):
        self.A[arm] += np.outer(x, x)
        print('A[arm]=',self.A[arm])
        self.b[arm] += reward * x
        print('b[arm]=',self.b[arm])
        
np.random.seed(0)

n_arms = 5
n_features = 5

true_weights = np.random.randn(n_arms, n_features)

def get_reward(arm, x):
    return x @ true_weights[arm] + np.random.randn() * .1

agent = LineasThompsonSampling(n_arms, n_features)

# Training loop 
for t in range(20):
    x = np.random.randn(n_features)       
    print('x=',x)
    
    arm = agent.select_arm(x)       
    print('arm=',arm)
    
    reward = get_reward(arm, x)       
    print('reward=',reward)
    
    agent.update(arm, x, reward)


print('True weights:')
print(true_weights)

print("\Learned weights:")
for a in range(n_arms):
    print(np.linalg.inv(agent.A[a]) @ agent.b[a])
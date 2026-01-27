# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 14:11:01 2026

@author: mthoma
"""
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
import scipy
pio.renderers.default='browser'


ADDS = ['A', 'B', 'C', 'D', 'E']
DASHMAP = {'A': 'solid', 'B': 'dot', 'C': 'dash', 'D': 'dashdot', 'E': 'longdash'}

class RegularizedLR(object):
    def __init__(self, name, alpha, rlambda, n_dim):
        self.name = name
        self.alpha = alpha
        self.rlambda = rlambda
        self.n_dim = n_dim
        self.m = np.zeros(n_dim)
        self.q = np.ones(n_dim) * rlambda
        self.w = self.get_sampled_weights()
        
    def get_sampled_weights(self):
        w = np.random.normal(self.m, self.alpha * self.q**(-1/2))
        return w
    
    def loss(self, w, *args):
        X, y = args
        n = len(y)
        regularizer = 0.5 * np.dot(self.q, (w - self.m)**2)
        pred_loss = sum([np.log(1+np.exp(np.dot(w, X[j]))) - y[j] * np.dot(w, X[j]) for j in range(n)])
        return regularizer + pred_loss
                         
    def fit(self, X, y):
        if y:
            X = np.array(X)
            y = np.array(y)
            minimization = scipy.minimize(self.loss, 
                                    self.w, args=(X,y), 
                                    metod="L-BFGS-B", 
                                    bounds = [(-10,10)]*3 + [(-1,1)], 
                                    options = {'maxiter':50})
            self.w = minimization.x
            self.m = self.w
            p = (1 + np.exp(-np.matmul(self.w, X.T)))**(-1)
            self.q = self.q + np.matmul(p * (1 - p), X**2)
            
    def calc_sigmoid(self, w, context):
        return 1 / (1 + np.exp(-np.dot(w, context)))
    
    def get_ucb(self, context):
        pred = self.calc_sigmoid(self.m, context)
        confidence = self.alpha * np.sqrt(np.sum(np.devide(np.array(context)**2, self.q)))
        ucb = pred + confidence
        return ucb
    
    def get_prediction(self, context):
        return self.calc_sigmoid(self.m, context)
    
    def sample_prediction(self, context):
        w = self.get_sampled_weights()
        return self.calc_sigmoid(w, context)
    
    def calculate_regret(ug, context, ad_options, ad):
        action_values = {a:ug.logistics(ug.beta[a], context) for a in ad_options}
        best_action = max(action_values, key=action_values.get)
        regret = action_values[best_action] - action_values[ad]
        return regret, best_action

class UserGenerator(object):
    def __init__(self):
        self.beta = {}
        self.beta['A'] = np.array([-4, -0.1, -3, 0.1])
        self.beta['B'] = np.array([-6, -0.1, 1, 0.1])
        self.beta['C'] = np.array([2, 0.1, 1, -0.1])
        self.beta['D'] = np.array([4, 0.1, -3, -0.2])
        self.beta['E'] = np.array([-0.1, 0, 0.5, -0.01])
        self.context = None
        
    def logistic(self, beta, context):
        f = np.dot(beta, context)
        p = 1 / (1 + np.exp(-f))
        return p
    
    def display_ad(self, ad):
        if ad in ADDS:
            p = self.logistic(self.beta[ad], self.context)
            reward = np.random.binomial(n=1, p=p)
            return reward
        else:
            raise Exception(f"Unknown add: {ad}!")
            
    def generate_user_with_context(self):
        # 0: Internationl, 1: U.S.
        location = np.random.binomial(n=1, p=0.6)
        # 0: Desktop, 1: Mobile.
        device = np.random.binomial(n=1, p=0.8)
        # User age changes between 10 and 70
        # with mean age 34
        age = 10 + int(np.random.beta((2,3)*60))
        # Add 1 to the concept for the intercept
        self.context = [1, device, location, age]
        return self.context
       
def get_scatter(x,y,name,showlegend):

    s = go.Scatter(x=x,
                   y=y,
                   legendgroup=name,
                   showlegend=showlegend,
                   name=name,
                   line=dict(color='blue',
                             dash=DASHMAP[name]))
    return s

def visualize_bandits(ug):
    ad_list = 'ABCDE'
    ages = np.linspace(10, 70)
    fig = make_subplots(rows=2, 
                        cols=2, 
                        subplot_titles=('Desktop, International',
                                        'Desktop, U.S',
                                        'Mobile, International',
                                        'Mobile, U.S.'))
    for device in [0,1]:
        for location in [0,1]:
            showlegend = (device == 0) and (location == 0)
            for ad in ad_list:
                probs = [ug.logistic(ug.beta[ad],
                         [1, device,location,age]) for age in ages]
                
                fig.add_trace(get_scatter(ages, probs, ad, showlegend),row=device+1, col=location+1)
    fig.update_layout(template='presentation')
    fig.show()


def select_ad_eps_greedy(ad_models, context, eps):
    if np.random.uniform() < eps:
        return np.random.choice(list(ad_models.keys()))
    else:
        prediction = {ad: ad_models[ad].get_prediction(context) for ad in ad_models}
        max_value = max(prediction.values())
        max_keys = [key for key, value in prediction.items() if value == max_value]
        return np.random.choice(max_keys)
               
if __name__ == "__main__":
    ug = UserGenerator()
    visualize_bandits(ug)
                   
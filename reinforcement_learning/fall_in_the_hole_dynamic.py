# -*- coding: utf-8 -*-
"""
Created on Sat Dec  6 13:49:02 2025

@author: mthoma
"""
import enum
import random
import numpy as np
import reinforcement_learning as rl

class Action(enum.IntEnum):
    FORWARD = 0
    BACKWARD = 1
    LEFT = 2
    RIGHT = 3

class StateAction(object):
    
    def __init__(self, dim):
        self.dim = dim
        self.state_action_map = []
        
        self.__init__state_action_map()
        
        # print("\n".join((",".join(str(i) for i in sublist)) for sublist in self.state_action_map))
        
    def __init__state_action_map(self):
        
        state_ctn = self.dim * self.dim
        for i in range(state_ctn):
            
            state_action = np.zeros(4)
            self.state_action_map.append(state_action)
            
            # Forward action: only in the last row you cannot move forward
            if(i < (state_ctn - self.dim)):
                state_action[0] = 1
            
            # Backward action: only in the first row you cannot move backward
            if(i > (self.dim - 1)):
                state_action[1] = 1
            
            # Left action: you cannot move left if i % self.dim == 0
            if not ((i % self.dim) == 0):
                state_action[2] = 1
                
            # Right action: not possible when you are at the far right
            if not(((i + 1) % self.dim) == 0):
                state_action[3] = 1
   
    def get_action(self, state):
        
        possible_actions = [i for i, val in enumerate(self.state_action_map[state]) if val == 1]        
        action = Action(random.choice(possible_actions))
        
        return action
    
    def get_state_count(self):
        return len(self.state_action_map)
                
    def __str__(self):
        
        to_str = "\n".join(["".join(str(i)) for i in self.state_action_map])
        
        return to_str
    
    def get_next_state(self, state, action):
        
        if action == Action.FORWARD:
            return state + self.dim
        elif action == Action.BACKWARD:
            return state - self.dim
        elif action == Action.LEFT:
            return state - 1
        else:
            return state + 1
    
class FallInTheHoleABnTesting(rl.MABandit):
    
    def __init__(self, dimension, n_traps):
        self.state_action = StateAction(dimension)
        self.trap_states = random.sample(list((num for num in range(self.state_action.get_state_count()) if num not in {0,24})), n_traps)
        print(self.trap_states)
        self.state = None
        self.action = None       
        self.N = []        
        for _ in range(self.state_action.get_state_count()):
            self.N.append(np.zeros(4))
        
        self.Q = []        
        for _ in range(self.state_action.get_state_count()):
            self.Q.append(np.zeros(4))
        
    def is_trap(self):        
        state = self.state_action.get_next_state(self.state, self.action)        
        return (state in self.trap_states)
    
    def is_exit(self):        
        state = self.state_action.get_next_state(self.state, self.action)   
        return (state in [0, self.state_action.get_state_count() - 1])
    
    def reward(self):
        if self.is_trap():
            return -100
        elif self.is_exit():
            return 100
        else:
            return 0
        
    def pull(self):        
        if self.state is None or self.is_round_over():
            # in case of training start or new training round
            self.state = np.random.randint(self.state_action.get_state_count())
        else:
            # in case a training round is ongoing
            self.state = self.state_action.get_next_state(self.state, self.action)
        
        self.action = self.state_action.get_action(self.state)
        
    def increment_N(self):
        self.N[self.state][self.action] += 1        
        return self.N[self.state][self.action]
        
    def last_Q(self):
        return self.Q[self.state][self.action]
        
    def update_Q(self, Qnp1):
        self.Q[self.state][self.action] = Qnp1
            
    def best_ad(self):
        # Not relevant
        pass
        
    def is_round_over(self):        
        return (self.is_trap() or self.is_exit()) 
    
    def __str__(self):
        N_as_str = "\n".join((",".join(str(i) for i in sublist)) for sublist in self.N)
        Q_as_str = "\n".join((",".join(str(i) for i in sublist)) for sublist in self.Q)
        return Q_as_str #+ "\n" + N_as_str

def main():
    
    dimension = 5
    n_traps = 3
    n_tests = 10000
    
    bandit = FallInTheHoleABnTesting(dimension, n_traps)
    
    abn_model = rl.ABNModel(n_tests, bandit)
    abn_model.apply()
    
    print(bandit)

if __name__ == "__main__":
    main()
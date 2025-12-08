# -*- coding: utf-8 -*-
"""
Created on Sat Dec  6 13:49:02 2025

@author: mthoma
"""
import enum
import random

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
        
    def __init__state_action_map(self):
        
        state_ctn = self.dim * self.dim
        for i in range(state_ctn):
            
            state_action = [0,0,0,0]
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
    
class FallInTheHole(object):
    
    def __init__(self, dimension, n_traps):
        self.state_action = StateAction(dimension)
        self.trap_states = random.sample(range(self.state_action.get_state_count()), n_traps)
        
    def is_trap(self, state):
        return (state in self.trap_states)
        
    def train(self, rounds, rl_model):
        self.rounds = rounds
        

def main():
    
    stateAction = StateAction(5)
    print(stateAction)
    print(stateAction.get_action(0))

if __name__ == "__main__":
    main()
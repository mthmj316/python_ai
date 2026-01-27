# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 17:56:14 2025

@author: mthoma
"""
import random
import os

clear = lambda: os.system('cls')

random.seed()

ALPHA = .1
GAMMA = .9

ACTION_RIGHT = 1
ACTION_LEFT = 0

TRAINING_ROUNDS = 1000

REWARDS = [
    [0,0],      # 0
    [100,0],    # 1
    [0,-100],   # 2
    [0,0],      # 3
    [-100,0],   # 4
    [0,0],      # 5   
    [0,0],      # 6
    [0,0],      # 7
    [0,100],    # 8
    [0,0]       # 9
    ]

quality = [
    [0,0],      # 0
    [0,0],      # 1
    [0,0],      # 2
    [0,0],      # 3
    [0,0],      # 4
    [0,0],      # 5   
    [0,0],      # 6
    [0,0],      # 7
    [0,0],       # 8
    [0,0]       # 9
    ]

# 0 and 9 victory and 3 death
game_over_states = [0, 3, 9]

###############################################################################
## Coding for AI training  ####################################################

def get_qnew(qold, qmax, reward, alpha=.1, gama=.9):
    
    qnew = qold + alpha * (reward + gama*qmax - qold)
    
    return qnew


def get_next_action(state):
    
    if state == 0:
        # The game character is on the far left side.
        # Hence it can move rigth only
        return ACTION_RIGHT
    elif state == len(quality) -1:
        # The game character is on the far right side.
        # Hence it can move left only
        return ACTION_LEFT
    else:
        return random.choice([ACTION_LEFT, ACTION_RIGHT])
    
    
def get_next_state(state, action):
    return (state -1) if action == ACTION_LEFT else state + 1


def is_training_round_over(state):
    return state in game_over_states
        

def run_training():
    
    for i in range(TRAINING_ROUNDS):
        
        #Take up staring position
        state = random.choice(range(len(quality)))
        
        while not is_training_round_over(state):
            action = get_next_action(state)
            next_state = get_next_state(state,action)
            
            qold = quality[state][action]
            reward = REWARDS[state][action]
            qmax = max(quality[next_state])
            
            quality[state][action] = get_qnew(qold, qmax, reward)
            
            state = next_state
            
    #print(quality)

## END: Coding for AI training ################################################
###############################################################################

###############################################################################
### Coding for playing the game ###############################################

ALLOWED_USER_INPUT = ['1', '2', '4', '5', '6', '7', '8', 'q', 'Q']

def is_game_over(state):
    
    if state == 3:
        print("AI is dead!")
        return True
    elif state in [0, 9]:
        print("AI wins!")
        return True
    else:
        return False

def get_next_play_action (state):
    
    action = None
    
    if state == 0:
        print("Far left is reached")
        action = ACTION_RIGHT
    elif state == len(quality)-1:
        print("Far right is reached")
        action = ACTION_LEFT
    else:
        quality_for_state = quality[state]
        
        if quality_for_state[0] > quality_for_state[1]:
            action = ACTION_LEFT
        elif quality_for_state[0] < quality_for_state[1]:
            action = ACTION_RIGHT
        else:
            action = random.choice([ACTION_LEFT, ACTION_RIGHT])
        
   # print(f"Next action {action}")
    
    return action
    
def play():
    
    clear()
    
    print()
    
    user_in = -1
    
    while True:
        
        print(f"Enter a number: {ALLOWED_USER_INPUT}  (To quit press q)")
        
        while user_in not in ALLOWED_USER_INPUT:
            user_in = input()
            
        if user_in in ['Q', 'q']:
            break
        
        state = int(user_in)
        user_in = -1
               
        print(f"Starting postion: {state}")
        
        while not is_game_over(state):
            action = get_next_play_action(state)
            state = get_next_state(state, action)
            
            print(f"Move to {state}")
            
    
## END: Coding for playing the game ###########################################
###############################################################################

def main():
    run_training()    
    play()


if __name__ == "__main__":
    main()
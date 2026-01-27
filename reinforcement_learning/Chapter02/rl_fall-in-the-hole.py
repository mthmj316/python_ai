# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 13:55:15 2025

@author: mthoma
"""
import random
# import file_access as fa
import time

random.seed()

ALPHA = .1
GAMMA = .9

ACTION_FORWARD = 0
ACTION_BACKWARD = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3

TRAINING_ROUNDS = 200

REWARDS = [
    [0,0,0,0],          # State = 0
    [0,0,100,0],        # State = 1
    [0,0,0,0],          # State = 2
    [-100,0,0,0],       # State = 3
    [0,0,0,0],          # State = 4
    [0,100,0,0],        # State = 5
    [-100,0,0,0],       # State = 6
    [0,0,0,0],          # State = 7
    [0,0,0,0],          # State = 8    
    [0,0,-100,0],       # State = 9
    [0,0,0,-100],       # State = 10
    [0,0,0,0],          # State = 11
    [0,0,-100,0],       # State = 12
    [0,-100,0,0],       # State = 13
    [0,0,0,0],          # State = 14
    [0,0,0,0],          # State = 15
    [0,-100,0,0],       # State = 16
    [0,0,0,0],          # State = 17
    [0,0,0,0],          # State = 18
    [100,0,0,0],        # State = 19
    [0,0,0,-100],       # State = 20
    [0,0,0,0],          # State = 21
    [0,0,0,0],          # State = 22
    [0,0,0,-100],       # State = 23
    [0,0,0,0]           # State = 24
    ]     

quality = [
    [0,0,0,0],#1a
    [0,0,0,0],#2a
    [0,0,0,0],#3a
    [0,0,0,0],#4a
    [0,0,0,0],#5a
    [0,0,0,0],#1b
    [0,0,0,0],#2b
    [0,0,0,0],#3b
    [0,0,0,0],#4b
    [0,0,0,0],#5b
    [0,0,0,0],#1c
    [0,0,0,0],#2c
    [0,0,0,0],#3c
    [0,0,0,0],#4c
    [0,0,0,0],#5c
    [0,0,0,0],#1d
    [0,0,0,0],#2d
    [0,0,0,0],#3d
    [0,0,0,0],#4d
    [0,0,0,0],#5d
    [0,0,0,0],#1e
    [0,0,0,0],#2e
    [0,0,0,0],#3e
    [0,0,0,0],#4e
    [0,0,0,0],#5e
    ]

###############################################################################
## Coding for AI training  ####################################################

game_over_states = [0,8,11,21,24]

def get_qnew(qold, qmax, reward, alpha=.1, gama=.9):
    
    qnew = qold + alpha * (reward + gama*qmax - qold)
    
    return qnew

def get_next_action(state): 
    if state in [0,5,10,15,20]:
        if state == 0:
            return random.choice([ACTION_FORWARD, ACTION_RIGHT])
        elif state == 20:
            return random.choice([ACTION_BACKWARD, ACTION_RIGHT])
        else:
            return random.choice([ACTION_FORWARD, ACTION_BACKWARD, ACTION_RIGHT])
    elif state in [4,9,14,19,24]:
        if state == 4:
            return random.choice([ACTION_FORWARD, ACTION_LEFT])
        elif state == 24:
            return random.choice([ACTION_BACKWARD, ACTION_LEFT])
        else:
            return random.choice([ACTION_FORWARD, ACTION_BACKWARD, ACTION_LEFT])
    elif state in [1,2,3]:
        return random.choice([ACTION_FORWARD, ACTION_LEFT, ACTION_RIGHT])
    elif state in [21,22,23]:
        return random.choice([ACTION_BACKWARD, ACTION_LEFT, ACTION_RIGHT])
    else:
        return random.choice([ACTION_FORWARD, ACTION_BACKWARD, ACTION_LEFT, ACTION_RIGHT])


def get_next_state(state, action):
    
    if action == ACTION_FORWARD:
        return state + 5
    elif action == ACTION_BACKWARD:
        return state - 5
    elif action == ACTION_LEFT:
        return state - 1
    else:
        return state + 1
    
def is_training_round_over(state):
    return state in game_over_states

def get_rewards(state, action):
    
    state_actions = REWARDS[state]
    
    rewards = state_actions[action]
    
    return rewards

def run_training():
    
    path_export = []
    
    for i in range(TRAINING_ROUNDS):
        
        #Take up staring position
        state = random.choice(range(len(quality)))
        
        path = [state]
        
        while not is_training_round_over(state):
            action = get_next_action(state)
            next_state = get_next_state(state,action)
            
            qold = quality[state][action]
            reward = get_rewards(state, action)
            qmax = max(quality[next_state])
            
            quality[state][action] = get_qnew(qold, qmax, reward)
            
            state = next_state
            
            path.append(state)
            
        path_export.append(path)
        
    
    return path_export
        
    
## END: Coding for AI training ################################################
###############################################################################

###############################################################################
### Coding for playing the game ###############################################

ALLOWED_USER_INPUT = ['1', '2', '3', '4', '5', '6', '7', '9', '10',
                      '12', '13', '14', '15', '16', '17', '18', 
                      '19', '20', '22', '23', 'q', 'Q']

def is_game_over(state):
    
    if state in [8, 11, 21]:
        print("AI is dead!")
        return True
    elif state in [0, 24]:
        print("AI wins!")
        return True
    else:
        return False

def get_next_play_action (state):
    
    action = None
    
    state_action = quality[state]
    
    action = state_action.index(max(state_action))
    
    return action


def play():
    
    
    user_in = -1
    
    path_export = []
    
    while True:
        
        print("Enter a number between 1 and 23 (not 8, 11 and 21). (To quit press q)")
        
        while user_in not in ALLOWED_USER_INPUT:
            user_in = input()
            
        if user_in in ['Q', 'q']:
            break
        
        state = int(user_in)
        user_in = -1
               
        print(f"Starting postion: {state}")
        
        path = [state]
        
        while not is_game_over(state):
            action = get_next_play_action(state)
            state = get_next_state(state, action)
            
            if state < 0 or state > 24:
                raise ValueError(f"Illigal state: {state}")
            
            print(f"Move to {state}")
            
            path.append(state)
            
        path_export.append(path)
    
    print(path_export)
    
    return path_export

## END: Coding for playing the game ###########################################
###############################################################################

def trainings_data_to_file(data_as_list):
    
    data = "\n".join((",".join(str(i) for i in sublist)) for sublist in data_as_list)
    
    file_name = f"./fall_in_the_hole_training_{time.time_ns()}.txt"
    
    # fa.write(file_name, data)    
    
    
            
def main():
    
    file_content = []
    file_content.append("### Trainings Data")
    file_content.extend(run_training())
    file_content.append("### Play Data")
    file_content.extend(play())
    
    trainings_data_to_file(file_content)
    
    print("\n".join((",".join(str(i) for i in sublist)) for sublist in quality))
   


if __name__ == "__main__":
    main()
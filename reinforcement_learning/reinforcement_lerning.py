# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 17:56:14 2025

@author: mthoma
"""
class MABandit(object):
    
    def reward(self):
        """
        Is called to get the reward for the randomly selected action

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        None.

        """
        raise Exception("must be overwritten by subclass")
        
    def pull(self):
        """
        Is called select the next action randomly

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        raise Exception("must be overwritten by subclass")

class RLModel(object):
    
    def __init__(self, n_tests):
        self.n_tests = n_tests
    
    def apply(self):
        pass
    
    
class ABNModel(RLModel):
    
    def __init__(self, n_tests, mabandit):
        super().__init__(n_tests)
        self.mabandit = mabandit
        
    
    def apply(self):
        
        print(self.n_tests)
        
        pass
    
    
m = ABNModel(4)

m.apply()
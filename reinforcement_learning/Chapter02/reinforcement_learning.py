# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 17:56:14 2025

@author: mthoma
"""
import abc

class MABandit(abc.ABC):    
    def reward(self):
        """
        Is called to get the reward for the randomly selected action

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
            the reward for the seletced action

        """
        pass
        
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
        pass
        
    def increment_N(self):
        """
        Is called to increment the selection counter of the selected action

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
            the new value of selection counter of the selected action

        """
        pass

    def last_Q(self):
        '''
        Returns the Q of the selected action before the current round.

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
            Q before this selection

        '''
        pass
        
        
    def update_Q(self, Qnp1):
        """
        Updates Q of the selected action

        Parameters
        ----------
        Qnp1 : float
            The new quality for the selected action.

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        None.

        """
        pass
        
    def is_round_over(self):
        """
        Returns True if a trainings round is over.
        By default it returned True.
        A subclass need to overwrite this method,
        if a training round has more than one steps.

        Returns
        -------
        bool
            DESCRIPTION.

        """
        pass

class RLModel():
    
    def __init__(self, n_tests):
        self.n_tests = n_tests
    
    def apply(self):
        pass
    
    
class ABNModel(RLModel):    
    def __init__(self, n_tests, mabandit):
        super().__init__(n_tests)
        self.mabandit = mabandit
    
    def apply(self):        
        for r in range(self.n_tests):    
            while True:
                # triggers the random selction of the next action
                self.mabandit.pull()
                
                # get the reward for the selected action
                R = self.mabandit.reward()
                
                # Note: within the mabandit the selection counter of the selected
                # action must be incremeneted
                N = self.mabandit.increment_N()
                
                # the significance level, representing the maximum probability
                #  (usually 5% or 0.05) of committing a Type I (false positive) error
                alpha = 1 / N
                
                #  Qn = the quality of the action after the last selection
                Qn = self.mabandit.last_Q()
                
                # Qn+1 = the new quality of the selected action
                Qnp1 = Qn + alpha  * (R - Qn)
                
                # update Q in the MAB (Multi-Armed-Bandit)
                self.mabandit.update_Q(Qnp1)
                
                if self.mabandit.is_round_over():
                    break
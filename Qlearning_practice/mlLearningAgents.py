# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# The agent here was written by Simon Parsons, based on the code in
# pacmanAgents.py
# learningAgents.py

from pacman import Directions
from game import Agent
import random
import game
import util
import numpy as np

# QLearnAgent
#
class QLearnAgent(Agent):

    # Constructor, called when we start running the
    def __init__(self, alpha=0.5, epsilon=0.05, gamma=0.8, numTraining = 10):
        # alpha       - learning rate
        # epsilon     - exploration rate
        # gamma       - discount factor
        # numTraining - number of training episodes
        #
        # These values are either passed from the command line or are
        # set to the default values above. We need to create and set
        # variables for them
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.numTraining = int(numTraining)
        # Count the number of games we have played
        self.episodesSoFar = 0

        ## Define my own attribute
        self.decode_state = None
        self.pick = None
        self.Qfuns = util.Counter()
    
    # Accessor functions for the variable episodesSoFars controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar +=1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
            return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value):
        self.epsilon = value

    def getAlpha(self):
        return self.alpha

    def setAlpha(self, value):
        self.alpha = value
        
    def getGamma(self):
        return self.gamma

    def getMaxAttempts(self):
        return self.maxAttempts

    
    # getAction
    #
    # The main method required by the game. Called every time that
    # Pacman is expected to move
    def getAction(self, state):
        
        # The data we have about the state of the game
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)
        
        ## change state into integer sequence
        decode_state = self.decodeState(state)
        
        ##
        # Pick an action but there is probability (epsilon)
        if legal:
            if np.random.random() > self.epsilon:
                pick = self.getPolicy(decode_state, legal)
        
            else:
                pick = random.choice(legal)

        ## update the Q-table
        if self.pick is not None:  # if it is the first entry, self.pick will be None
            self.update(self.decode_state, self.pick, decode_state, state.getScore(), legal)

        ## Record the decode state and action
        self.decode_state = decode_state
        self.pick = pick

        return pick

    def decodeState(self, state):
        decode_state = ''
        
        pacmanP = state.getPacmanPosition()
        decode_state += str(int(pacmanP[0])) + str(int(pacmanP[1]))
        
        ghost= state.getGhostPositions()
        decode_state += str(int(ghost[0][0])) + str(int(ghost[0][1]))
        
        ff = state.getFood()
        for index, ii in enumerate(ff):
            for index2, jj in enumerate(ii):
                if jj:
                    decode_state += str(index) + str(index2)
    
        return decode_state

    def getPolicy(self, decode_state, legal_actions): 
        ## Get the action and Q value of each action

        if legal_actions:
            maxv = float("-inf")
            best_move = None
            
            for action in legal_actions:
                q = self.Qfuns[(decode_state, action)]
                print action, ': ', q ,

                if q >= maxv:
                    maxv = q
                    best_move = action
                
            
            print '\n'
            print '=============='
            
            return best_move

    def update(self, state, action, new_state, reward, legal_actions):
        # update the Qfuns
        if reward > -500 and reward < 0:
            reward = 0

        if legal_actions:
            Q = []
            for next_action in legal_actions:
                Q.append(self.Qfuns[(new_state, next_action)])

            R = reward + self.gamma * max(Q)
        else:
            R = reward

        self.Qfuns[(state, action)] = self.Qfuns[(state, action)] + self.alpha * (R - self.Qfuns[(state, action)])





    # Handle the end of episodes
    #
    # This is called by the game after a win or a loss.
    def final(self, state):
        
        ## Before the game ended, update!
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        decode_state = self.decodeState(state)
        
        if self.pick is not None:  # if it is the first entry, self.pick will be None
            self.update(self.decode_state, self.pick, decode_state, state.getScore(), legal)
        
        ###

        print "A game just ended!"
        
        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print '%s\n%s' % (msg,'-' * len(msg))
            self.setAlpha(0)
            self.setEpsilon(0)



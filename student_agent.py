# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
import copy
import random
import math
from PolicyApproximator import *


approximator = None
policy_approximator = None

def NTupleApproximator_get_action(board, score):
    global approximator
    if approximator is None:
        approximator = load_approximator(NTupleApproximator.__new__(NTupleApproximator), "approximator.pkl")
    return approximator.get_action(board, score)

def PUCTMCTS_get_action(board, score):
    global approximator
    global policy_approximator
    if approximator is None:
        approximator = load_approximator(NTupleApproximator.__new__(NTupleApproximator))
    if policy_approximator is None:
        policy_approximator = load_policy_approximator(NTupleApproximator.__new__(NTupleApproximator))
    
    env = Game2048Env(board, score)
    mcts_puct = MCTS_PUCT( env, value_approximator=approximator, policy_approximator=policy_approximator, iterations= 50, c_puct=1.41, rollout_depth=6, gamma=0.99)
    root = PUCTNode(env)
    for _ in range(mcts_puct.iterations):
        mcts_puct.run_simulation(root)
    best_action, distribution = mcts_puct.best_action_distribution(root)
    return best_action

def get_action(state, score):
    return NTupleApproximator_get_action(state, score)
    
    # You can submit this random agent to evaluate the performance of a purely random strategy.



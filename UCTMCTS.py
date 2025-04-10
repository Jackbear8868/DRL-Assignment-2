import copy
import random
import math
import numpy as np

# UCT Node for MCTS
class UCTNode:
    def __init__(self, env, parent=None, action=None):
        """
        state: current board state (numpy array)
        score: cumulative score at this node
        parent: parent node (None for root)
        action: action taken from parent to reach this node
        """
        self.state = env.board.copy()
        self.score = env.score
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.total_reward = 0.0
        self.untried_actions = [a for a in range(4) if env.is_move_legal(a)]

    def fully_expanded(self):
		# A node is fully expanded if no legal actions remain untried.
        return len(self.untried_actions) == 0

class UCTMCTS:
    def __init__(self, env, iterations=500, exploration_constant=1.41, rollout_depth=10):
        self.env = env
        self.iterations = iterations
        self.c = exploration_constant  # Balances exploration and exploitation
        self.rollout_depth = rollout_depth

    def create_env_from_state(self, state, score):
        """
        Creates a deep copy of the environment with a given board state and score.
        """
        new_env = copy.deepcopy(self.env)
        new_env.board = state.copy()
        new_env.score = score
        return new_env

    def select_child(self, node):
        # Use the UCT formula: Q + c * sqrt(log(parent_visits)/child_visits) to select the child
        return max(
            node.children.values(),
            key=lambda child: child.total_reward + self.c * math.sqrt(math.log(node.visits) / child.visits)
        )

    def rollout(self, sim_env, depth):
        # Perform a random rollout from the current state up to the specified depth.
        for _ in range(depth):
            legal_actions = [a for a in range(4) if sim_env.is_move_legal(a)]
            if legal_actions == []:
                break
            action = random.choice(legal_actions)
            sim_env.step(action)
        return sim_env.score

    def backpropagate(self, node, reward):
        # Propagate the reward up the tree, updating visit counts and total rewards.
        while node is not None:
            node.visits += 1
            node.total_reward += (reward - node.total_reward) / node.visits
            node = node.parent

    def run_simulation(self, root):
        node = root
        sim_env = self.create_env_from_state(node.state, node.score)

        # Selection: Traverse the tree until reaching a non-fully expanded node.
        while node.fully_expanded() and node.children:
            node = self.select_child(node)
            sim_env.step(node.action)

        # Expansion: if the node has untried actions, expand one.
        if node.untried_actions != []:
            action = node.untried_actions.pop()
            sim_env.step(action)
            node.children[action] = UCTNode(sim_env, parent=node, action=action)
            node = node.children[action]

        # Rollout: Simulate a random game from the expanded node.
        rollout_reward = self.rollout(sim_env, self.rollout_depth)
        # Backpropagation: Update the tree with the rollout reward.
        self.backpropagate(node, rollout_reward)

    def best_action_distribution(self, root):
        '''
        Computes the visit count distribution for each action at the root node.
        '''
        total_visits = sum(child.visits for child in root.children.values())
        distribution = np.zeros(4)
        for action, child in root.children.items():
            distribution[action] += child.visits / total_visits if total_visits > 0 else 0
        return np.argmax(distribution), distribution

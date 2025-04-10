import copy
import random
import math
import numpy as np

# Node for TD-MCTS using the TD-trained value approximator
class TDMCTSNode:
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

# TD-MCTS class utilizing a trained approximator for leaf evaluation
class TDMCTS:
    def __init__(self, env, approximator, iterations=500, exploration_constant=1.41, rollout_depth=10, gamma=0.99):
        self.env = env
        self.approximator = approximator
        self.iterations = iterations
        self.c = exploration_constant
        self.rollout_depth = rollout_depth
        self.gamma = gamma

    def create_env_from_state(self, state, score):
        # Create a deep copy of the environment with the given state and score.
        new_env = copy.deepcopy(self.env)
        new_env.board = state.copy()
        new_env.score = score
        return new_env

    def select_child(self, node):
        # Use the UCT formula: Q + c * sqrt(log(parent.visits)/child.visits) to select the best child.
        return max(
            node.children.values(),
            key=lambda child: self.approximator.value(child.state) + self.c * math.sqrt(math.log(node.visits) / child.visits)
        )

    def rollout(self, sim_env, depth):
        # Perform a random rollout until reaching the maximum depth or a terminal state.
        # Use the approximator to evaluate the final state.
        for _ in range(depth):
            legal_actions = [a for a in range(4) if sim_env.is_move_legal(a)]
            if legal_actions == []:
                break
            action = random.choice(legal_actions)
            sim_env.step(action)
        return sim_env.score #self.approximator.value(sim_env.board)
        
    def backpropagate(self, node, reward):
        # Propagate the obtained reward back up the tree.
        while node is not None:
            node.visits += 1
            # node.total_reward += (reward - node.total_reward) / node.visits
            node = node.parent
            # reward *= self.gamma

    def run_simulation(self, root):
        node = root
        sim_env = self.create_env_from_state(node.state, node.score)

        # Selection: Traverse the tree until reaching an unexpanded node.
        while node.fully_expanded() and node.children:
            node = self.select_child(node)
            sim_env.step(node.action)

        # Expansion: If the node is not terminal, expand an untried action.
        if node.untried_actions != []:
            action = node.untried_actions.pop()
            sim_env.step(action, add_random_tile=False)
            node.children[action] = TDMCTSNode(sim_env, parent=node, action=action)
            node = node.children[action]

        # Rollout: Simulate a random game from the expanded node.
        rollout_reward = self.rollout(sim_env, self.rollout_depth)
        # Backpropagate the obtained reward.
        self.backpropagate(node, rollout_reward)

    def best_action_distribution(self, root):
        # Compute the normalized visit count distribution for each child of the root.
        total_visits = sum(child.visits for child in root.children.values())
        distribution = np.zeros(4)
        for action, child in root.children.items():
            distribution[action] += child.visits / total_visits if total_visits > 0 else 0
        return np.argmax(distribution), distribution

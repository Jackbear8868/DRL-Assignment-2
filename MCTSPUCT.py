import copy
import math
import random
import numpy as np

class PUCTNode:
    def __init__(self, env, parent=None, action=None, prior=0.0):
        self.state = env.board.copy()
        self.score = env.score
        self.parent = parent
        self.action = action
        self.prior = prior
        self.children = {}
        self.visits = 0
        self.total_reward = 0.0
        self.untried_actions = [a for a in range(4) if env.is_move_legal(a)]

    def fully_expanded(self):
        return len(self.untried_actions) == 0


class MCTS_PUCT:
    def __init__(self, env, value_approximator, policy_approximator, iterations=500, c_puct=1.41, rollout_depth=10, gamma=0.99):
        self.env = env
        self.value_approximator = value_approximator
        self.policy_approximator = policy_approximator
        self.iterations = iterations
        self.c_puct = c_puct
        self.rollout_depth = rollout_depth
        self.gamma = gamma

    def create_env_from_state(self, state, score):
        """Creates a deep copy of the environment to simulate a given state."""
        new_env = copy.deepcopy(self.env)
        new_env.board = state.copy()
        new_env.score = score
        return new_env

    def select_child(self, node):
        # TODO: Select the best child using the PUCT formula:
        # PUCT(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        # where Q(s,a) = child.total_reward / child.visits.
        best_child = None
        best_score = -float('inf')
        for child in node.children.values():
            score = child.total_reward / child.visits + self.c_puct * child.prior * math.sqrt(node.visits) / (1 + child.visits)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child


    def rollout(self, sim_env, depth):
        # TODO: Perform a random rollout until reaching the maximum depth or a terminal state.
        # TODO: Use the approximator to evaluate the final state.
        # Note: It's not necessary to perform rollouts if the value approximator is accurate.
        for _ in range(depth):
            legal_moves = [a for a in range(4) if sim_env.is_move_legal(a)]
            if not legal_moves:
                break
            action = random.choice(legal_moves)
            sim_env.step(action, add_random_tile=False)

        return self.value_approximator.value(sim_env.board)

    def backpropagate(self, node, reward):
        # TODO: Propagate the reward up the tree, updating visit counts and total rewards.
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent
            reward *= self.gamma

    def run_simulation(self, root):
        node = root
        sim_env = self.create_env_from_state(node.state, node.score)

        # TODO: Selection phase: traverse the tree until reaching an expandable node.
        while node.fully_expanded() and node.children:
            node = self.select_child(node)
            sim_env.step(node.action)

        # TODO: Expansion phase: if the node is not terminal, expand one untried action.
        if node.untried_actions:
            policy = self.policy_approximator.predict(sim_env.board)
            action = random.choice(node.untried_actions)
            node.untried_actions.remove(action)
            sim_env.step(action, add_random_tile=False)

            node.children[action] = PUCTNode(sim_env.board.copy(), sim_env.score, parent=node, action=action, prior=policy[action])
            node = node.children[action]

        # Rollout phase: simulate random moves from the expanded node.
        rollout_reward = self.rollout(sim_env, self.rollout_depth)
        # Backpropagation phase: update the tree with the obtained reward.
        self.backpropagate(node, rollout_reward)

    def best_action_distribution(self, root):
        total_visits = sum(child.visits for child in root.children.values())
        distribution = np.zeros(4)
        best_visits = -1
        best_action = None
        for action, child in root.children.items():
            distribution[action] = child.visits / total_visits if total_visits > 0 else 0
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = action
        return best_action, distribution



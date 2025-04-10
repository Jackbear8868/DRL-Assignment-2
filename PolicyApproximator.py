import copy
import math
import random
import numpy as np
from collections import defaultdict

from NTupleApproximator import *
from MCTSPUCT import *
from utils import *

class PolicyApproximator:
    def __init__(self, board_size, patterns):
        """
        Initializes the N-Tuple approximator.
        Hint: you can adjust these if you want.
        """
        self.board_size = board_size
        self.patterns = patterns
        self.actions = [0, 1, 2, 3]
        # Weight structure: [pattern_idx][feature_key][action]
        self.weights = [defaultdict(lambda: defaultdict(float)) for _ in range(len(patterns))]
        # Generate the 8 symmetrical transformations for each pattern and store their types.
        self.transforms = [lambda x: x, rot90, rot180, rot270, reflect_x, reflect_y, reflect_diag1, reflect_diag2]

        self.symmetry_patterns = []
        self.symmetry_types = []  # Store the type of symmetry transformation (rotation or reflection)
        for pattern in self.patterns:
            syms, types = self.generate_symmetries(pattern)
            self.symmetry_patterns.extend(syms)
            self.symmetry_types.extend(types)

        # TODO: Define corresponding action transformation functions for each symmetry.
        self.action_transform_map = {
            "identity": lambda a: a,
            "rot90": rot90_action,
            "rot180": rot180_action,
            "rot270": rot270_action,
            "reflect_x": reflect_x_action,
            "reflect_y": reflect_y_action,
            "reflect_diag1": reflect_diag1_action,
            "reflect_diag2": reflect_diag2_action,
        }


    def generate_symmetries(self, pattern):
        # TODO: Generate 8 symmetrical transformations of the given pattern.
        sysm = [trans(pattern) for trans in self.transforms]
        types = ["identity", "rot90", "rot180", "rot270", "reflect_x", "reflect_y", "reflect_diag1", "reflect_diag2"]
        return sysm, types


    def tile_to_index(self, tile):
        return 0 if tile == 0 else int(math.log(tile, 2))

    def get_feature(self, board, coords):
        # TODO: Extract tile values from the board based on the given coordinates and convert them into a feature tuple.
        return tuple(self.tile_to_index(board[x][y]) for (x, y) in coords)


    def predict(self, board):
        # TODO: Predict the policy (probability distribution over actions) given the board state.
        values = np.zeros(len(self.actions))
        for pattern_idx, pattern in enumerate(self.symmetry_patterns):
            feature = self.get_feature(board, pattern)
            symmetry_type = self.symmetry_types[pattern_idx]
            for action in self.actions:
                transformed_action = self.action_transform_map[symmetry_type](action)
                values[action] += self.weights[pattern_idx // len(self.transforms)][feature][transformed_action]

        exp_values = np.exp(values - np.max(values))
        return exp_values / np.sum(exp_values)


    def update(self, board, target_distribution, alpha=0.1):
        # TODO: Update policy based on the target distribution.
        # Update weights for each pattern and its symmetries.
        pred_distribution = self.predict(board)
        for sym_pattern_idx, sym_pattern in enumerate(self.symmetry_patterns):
            sym_type = self.symmetry_types[sym_pattern_idx]
            feature = self.get_feature(board, sym_pattern)
            for action in self.actions:
                transformed_action = self.action_transform_map[sym_type](action)
                self.weights[sym_pattern_idx // len(self.transforms)][feature][transformed_action] += alpha * (target_distribution[action] - pred_distribution[action])

def store_policy_approximator(policy_approximator, pickle_filename="policy_approximator.pkl"):
    with open(pickle_filename, "wb") as f:
        pickle.dump(policy_approximator.__dict__, f)

def load_policy_approximator(policy_approximator, pickle_filename="policy_approximator.pkl"):
    with open(pickle_filename, "rb") as f:
        state = pickle.load(f)
    policy_approximator.__dict__.update(state)
    return policy_approximator


def self_play_training_policy_value(env, mcts_puct, policy_approximator, value_approximator, num_episodes=50, value_lr=0.01):
    final_scores = []
    success_flags = []
    for episode in tqdm(range(num_episodes)):
        state = env.reset()
        done = False
        max_tile = np.max(state)

        while not done:
            # Create the root node for the MCTS-PUCT tree
            root = PUCTNode(state, env.score)

            # Run multiple simulations to build the MCTS search tree
            for _ in range(mcts_puct.iterations):
                mcts_puct.run_simulation(root)

            # TODO: Update the NTuple Policy Approximator using the MCTS action distribution
            best_action, action_distribution = mcts_puct.best_action_distribution(root)
            policy_approximator.update(state, action_distribution)

            # TODO: Calculate the TD error for the value approximator and update your approximator
            next_state, next_score, done, _ = Game2048Env(state, env.score).step(best_action, add_random_tile = False)
            delta = (next_score - env.score) + value_approximator.value(next_state) - value_approximator.value(state)
            value_approximator.update(state, delta, value_lr)

            # Execute the selected action in the real environment
            state, reward, done, _ = env.step(best_action)

        final_scores.append(env.score)
        success_flags.append(1 if max_tile >= 2048 else 0)
        if (episode + 1) % 100 == 0:
            avg_score = np.mean(final_scores[-100:])
            success_rate = np.sum(success_flags[-100:]) / 100
            print(f"Episode {episode+1}/{num_episodes} | Avg Score: {avg_score:.2f} | Success Rate: {success_rate:.2f}")
        
        if (episode + 1) % 1000 == 0:
            store_approximator(value_approximator)
            store_policy_approximator(policy_approximator)
    
    store_approximator(value_approximator)
    store_policy_approximator(policy_approximator)

if __name__ == "__main__":
    
    patterns = [
        ids2coords([0, 1, 2, 4, 5, 6]),
        ids2coords([1, 2, 5, 6, 9, 13]),
        ids2coords([0, 1, 2, 3, 4, 5]),
        ids2coords([0, 1, 5, 6, 7, 10]),
        ids2coords([0, 1, 2, 5, 9, 10]),
        ids2coords([0, 1, 5, 9, 13, 14]),
        ids2coords([0, 1, 5, 8, 9, 13]),
        ids2coords([0, 1, 2, 4, 6, 10]),
    ]

    approximator = None
    if approximator is None:
        approximator = load_approximator(NTupleApproximator.__new__(NTupleApproximator))
    
    policy_approximator = PolicyApproximator(board_size=4, patterns=patterns)

    env = Game2048Env()
    mcts_puct = MCTS_PUCT( env, value_approximator=approximator, policy_approximator=policy_approximator, iterations= 50, c_puct=1.41, rollout_depth=6, gamma=0.99)

    self_play_training_policy_value(env, mcts_puct, policy_approximator, approximator)
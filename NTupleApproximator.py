import pickle
import math
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from Game2048Env import Game2048Env
from utils import *

class NTupleApproximator:
    def __init__(self, board_size, patterns):
        """
        Initializes the N-Tuple approximator.
        Hint: you can adjust these if you want
        """
        self.board_size = board_size
        self.patterns = patterns
        # Create a weight dictionary for each pattern (shared within a pattern group)
        self.weights = [defaultdict(float) for _ in patterns]
        # Generate symmetrical transformations for each pattern
        self.trans_funcs = [rot90, rot180, rot270, reflect_y, reflect_x, reflect_diag1, reflect_diag2]
        self.symmetry_patterns = []
        for pattern in self.patterns:
            syms = self.generate_symmetries(pattern)
            for syms_ in syms:
                self.symmetry_patterns.append(syms_)

    def generate_symmetries(self, pattern):
        return [pattern] + [trans(pattern, self.board_size) for trans in self.trans_funcs]

    def tile_to_index(self, tile):
        return 0 if tile == 0 else int(math.log(tile, 2))

    def get_feature(self, board, coords):
        return tuple(self.tile_to_index(board[x, y]) for x, y in coords)

    def value(self, board):
        return sum(self.weights[i // (len(self.trans_funcs) + 1)][self.get_feature(board, pattern)] for i, pattern in enumerate(self.symmetry_patterns))

    def update(self, board, delta, alpha):
        for i, pattern in enumerate(self.symmetry_patterns):
            self.weights[i // (len(self.trans_funcs) + 1)][self.get_feature(board, pattern)] += alpha * delta / len(self.symmetry_patterns)
    
    def get_action(self, board, score):
        env = Game2048Env(board, score)
        legal_moves = [a for a in range(4) if env.is_move_legal(a)]
        best_action, best_value = None, float('-inf')
        for action in legal_moves:
            next_afterstate, next_score, _, _ = Game2048Env(board, score).step(action, add_random_tile=False)
            total_value = next_score - score + self.value(next_afterstate)
            if total_value > best_value:
                best_action, best_value = action, total_value
        return best_action

def store_approximator(approximator, pickle_filename="approximator.pkl"):
    with open(pickle_filename, "wb") as f:
        pickle.dump(approximator.__dict__, f)

def load_approximator(approximator, pickle_filename="approximator.pkl"):
    with open(pickle_filename, "rb") as f:
        state = pickle.load(f)
    approximator.__dict__.update(state)
    return approximator

def td_learning(env, approximator, num_episodes=50000, alpha=0.1, pickle_filename="approximator.pkl"):
    """
    Trains the 2048 agent using TD-Learning.

    Args:
        env: The 2048 game environment.
        approximator: NTupleApproximator instance.
        num_episodes: Number of training episodes.
        alpha: Learning rate.
        gamma: Discount factor.
    """
    final_scores = []
    success_flags = []
    for episode in tqdm(range(num_episodes)):
        afterstate = env.reset()
        previous_score = 0
        max_tile = np.max(afterstate)

        done = False
        while not done:
            legal_moves = [a for a in range(4) if env.is_move_legal(a)]
            if not legal_moves:
                break
            
            action = approximator.get_action(env.board, env.score,)
            next_afterstate, next_score, done, _ = env.step(action, add_random_tile=False)
            env.add_random_tile()

            delta = next_score - previous_score + approximator.value(next_afterstate) - approximator.value(afterstate)
            approximator.update(afterstate, delta, alpha)

            afterstate = next_afterstate
            previous_score = next_score
            max_tile = max(max_tile, np.max(afterstate))

        approximator.update(afterstate, -approximator.value(afterstate), alpha)

        final_scores.append(env.score)
        success_flags.append(1 if max_tile >= 2048 else 0)
        if (episode + 1) % 100 == 0:
            avg_score = np.mean(final_scores[-100:])
            success_rate = np.sum(success_flags[-100:]) / 100
            print(f"Episode {episode+1}/{num_episodes} | Avg Score: {avg_score:.2f} | Success Rate: {success_rate:.2f}")
        
        if (episode + 1) % 1000 == 0:
            store_approximator(approximator, pickle_filename)

    return final_scores

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
        
    pickle_filename = "approximator.pkl"
    
    env = Game2048Env()
    approximator = NTupleApproximator(board_size=4, patterns=patterns)
    final_scores = td_learning(env, approximator, num_episodes=100000, alpha=0.5, pickle_filename=pickle_filename)
    store_approximator(approximator, pickle_filename)

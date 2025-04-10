def rot90(coords, board_size=4):
    return [(y, board_size - 1 - x) for x, y in coords]

def rot180(coords, board_size=4):
    return [(board_size - 1 - x, board_size - 1 - y) for x, y in coords]

def rot270(coords, board_size=4):
    return [(board_size - 1 - y, x) for x, y in coords]

def reflect_y(coords, board_size=4):
    return [(x, board_size - 1 - y) for x, y in coords]

def reflect_x(coords, board_size=4):
    return [(board_size - 1 - x, y) for x, y in coords]

def reflect_diag1(coords, board_size=4):
    return [(y, x) for x, y in coords]

def reflect_diag2(coords, board_size=4):
    return [(board_size - 1 - y, board_size - 1 - x) for x, y in coords]

def rot90_action(action):
    # 90° clockwise: up->right, down->left, left->up, right->down
    mapping = {0: 3, 1: 2, 2: 0, 3: 1}
    return mapping[action]

def rot180_action(action):
    mapping = {0: 1, 1: 0, 2: 3, 3: 2}
    return mapping[action]

def rot270_action(action):
    mapping = {0: 2, 1: 3, 2: 1, 3: 0}
    return mapping[action]

def reflect_x_action(action):
    # Horizontal flip: up and down swap; left/right unchanged
    mapping = {0: 1, 1: 0, 2: 2, 3: 3}
    return mapping[action]

def reflect_y_action(action):
    # Vertical flip: left and right swap; up/down unchanged
    mapping = {0: 0, 1: 1, 2: 3, 3: 2}
    return mapping[action]

def reflect_diag1_action(action):
    # Main-diagonal flip: up->left, down->right, left->up, right->down
    mapping = {0: 2, 1: 3, 2: 0, 3: 1}
    return mapping[action]

def reflect_diag2_action(action):
    # Anti-diagonal flip: up->right, down->left, left->down, right->up
    mapping = {0: 3, 1: 2, 2: 1, 3: 0}
    return mapping[action]

def ids2coords(ids):
    id2coord = [
        (0, 0), (0, 1), (0, 2), (0, 3),
        (1, 0), (1, 1), (1, 2), (1, 3),
        (2, 0), (2, 1), (2, 2), (2, 3),
        (3, 0), (3, 1), (3, 2), (3, 3),
    ]
    return tuple(id2coord[id] for id in ids)

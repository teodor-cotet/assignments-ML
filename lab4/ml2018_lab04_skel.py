import sys
import os.path
from argparse import ArgumentParser
from copy import copy
from random import choice


class Maze:

    NORTH, EAST, SOUTH, WEST = 0, 1, 2, 3  # actions

    DYNAMICS = {  # the stochastic effects of actions
        NORTH: {(0, -1): 0.1, (-1, 0): .8, (0, 1): .1},
        EAST: {(-1, 0): 0.1, (0, 1): .8, (1, 0): .1},
        SOUTH: {(0, 1): 0.1, (1, 0): .8, (0, -1): .1},
        WEST: {(1, 0): 0.1, (0, -1): .8, (-1, 0): .1},
    }

    WALL, EMPTY = "x", " "

    VISUALS = {
        (0, 0, 1, 1): "\N{BOX DRAWINGS HEAVY DOWN AND RIGHT}",
        (1, 0, 0, 1): "\N{BOX DRAWINGS HEAVY DOWN AND LEFT}",
        (1, 0, 1, 0): "\N{BOX DRAWINGS HEAVY HORIZONTAL}",
        (0, 1, 1, 0): "\N{BOX DRAWINGS HEAVY UP AND RIGHT}",
        (1, 1, 0, 0): "\N{BOX DRAWINGS HEAVY UP AND LEFT}",
        (0, 1, 0, 1): "\N{BOX DRAWINGS HEAVY VERTICAL}",
        (1, 1, 1, 1): "\N{BOX DRAWINGS HEAVY VERTICAL AND HORIZONTAL}",
        (1, 1, 1, 0): "\N{BOX DRAWINGS HEAVY UP AND HORIZONTAL}",
        (1, 1, 0, 1): "\N{BOX DRAWINGS HEAVY VERTICAL AND LEFT}",
        (1, 0, 1, 1): "\N{BOX DRAWINGS HEAVY DOWN AND HORIZONTAL}",
        (0, 1, 1, 1): "\N{BOX DRAWINGS HEAVY VERTICAL AND RIGHT}",
        (1, 0, 0, 0): "\N{BOX DRAWINGS HEAVY LEFT}",
        (0, 1, 0, 0): "\N{BOX DRAWINGS HEAVY UP}",
        (0, 0, 1, 0): "\N{BOX DRAWINGS HEAVY RIGHT}",
        (0, 0, 0, 1): "\N{BOX DRAWINGS HEAVY DOWN}",
        (0, 0, 0, 0): "\N{BOX DRAWINGS HEAVY VERTICAL AND HORIZONTAL}",
        WEST: "\N{LEFTWARDS ARROW}",
        NORTH: "\N{UPWARDS ARROW}",
        EAST: "\N{RIGHTWARDS ARROW}",
        SOUTH: "\N{DOWNWARDS ARROW}",
    }

    def __init__(self, map_name):
        self._rewards, self._cells = {}, []
        with open(os.path.join("maps", map_name), "r") as map_file:
            for line in map_file.readlines():
                if ":" in line:
                    name, value = line.strip().split(":")
                    self._rewards[name] = float(value)
                else:
                    self._cells.append(list(line.strip()))
        self._states = [(i, j) for i, row in enumerate(self._cells)
                        for j, cell in enumerate(row) if cell != Maze.WALL]
        print(self._rewards)

    @property
    def actions(self):
        return [Maze.NORTH, Maze.EAST, Maze.SOUTH, Maze.WEST]

    @property
    def states(self):
        return copy(self._states)

    def is_final(self, state):
        row, col = state
        return self._cells[row][col] != Maze.EMPTY

    def effects(self, state, action):
        if self.is_final(state):
            return []
        row, col = state
        next_states = {}
        for (d_row, d_col), prob in Maze.DYNAMICS[action].items():
            next_row, next_col = row + d_row, col + d_col
            if self._cells[next_row][next_col] == Maze.WALL:
                next_row, next_col = row, col
            if (next_row, next_col) in next_states:
                prev_prob, _ = next_states[(next_row, next_col)]
                prob += prev_prob
            cell = self._cells[next_row][next_col]
            reward = self._rewards["default" if cell == Maze.EMPTY else cell]
            next_states[(next_row, next_col)] = (prob, reward)
        return [(s, p, r) for s, (p, r) in next_states.items()]

    def print_policy(self, policy):
        last_row = []
        height = len(self._cells)

        for row, row_cells in enumerate(self._cells):
            width = len(row_cells)
            for col, cell in enumerate(row_cells):
                if cell == Maze.WALL:
                    north, south, west, east = 0, 0, 0, 0
                    if last_row and len(last_row) > col:
                        north = last_row[col] == Maze.WALL
                    if row + 1 < height:
                        south = self._cells[row + 1][col] == Maze.WALL
                    if col > 0:
                        west = row_cells[col - 1] == Maze.WALL
                    if col + 1 < width:
                        east = row_cells[col + 1] == Maze.WALL
                    sys.stdout.write(Maze.VISUALS[(west, north, east, south)])
                elif self.is_final((row, col)):
                    sys.stdout.write(cell)
                else:
                    action = policy[(row, col)]
                    sys.stdout.write(Maze.VISUALS[action])
            sys.stdout.write("\n")
            last_row = row_cells
        sys.stdout.flush()

    def print_values(self, v):
        #x = 7
        #print("{0:3.8f} {1:5f}".format(23, 44))
        #print(f"{x:2.2f}")

        for r, row_cells in enumerate(self._cells):
            print(" | ".join([f"{v[(r, c)]:5.2f}" if cell == Maze.EMPTY else "     "
                              for c, cell in enumerate(row_cells)]))

def compute_value(s, game, gamma, policy, v):
    #p(s', r | s, pi(s)) * (r + gamma * v(s'))

    effs = game.effects(s, policy[s])
    val = 0.0
    for e in effs:
        (sp, p, r) = e
        val += p * (r + gamma * v[sp])

    return val

def compute_policy(s, game, gamma, v):

    max_a = game.actions[0]
    max_v = -1000000.0

    for a in game.actions:
        effs = game.effects(s, a)
        vv = 0
        for e in effs:
            (sp, p, r) = e
            vv += p * (r + gamma * v[sp])
        if max_v < vv:
            max_a = a
            max_v = vv

    return max_a


def policy_iteration(game, args):
    gamma = args.gamma
    max_delta = args.max_delta
    v = {s: 0 for s in game.states}
    policy = {s: choice(game.actions)
              for s in game.states if not game.is_final(s)}

    done = False
    
    while done == False:

        value_convergence = False

        #compute values till they converge for the current policy
        while value_convergence == False:
            
            delta = 0

            for s in game.states:
                if not game.is_final(s):
                    v_old = v[s]
                    v[s] = compute_value(s, game, gamma, policy, v)
                    delta = max(delta, v[s] - v_old)

            if delta < max_delta:
                value_convergence = True

        done = True
        #compute new policy with actualized values
        for s in game.states:
            if not game.is_final(s):
                policy_old = policy[s]
                policy[s] = compute_policy(s, game, gamma, v)

                if policy[s] != policy_old:
                    done = False
    return policy, v


def value_iteration(game, args):
    gamma = args.gamma
    max_delta = args.max_delta
    v = {s: 0 for s in game.states}
    policy = {s: choice(game.actions)
              for s in game.states if not game.is_final(s)}

    done = False
    while not done:

        #compute policy with current values
        for s in game.states:  
            policy[s] = compute_policy(s, game, gamma, v)

        delta = 0
        value_convergence = False
        # actualize values, but not till the converge
        for s in game.states:
            if not game.is_final(s):
                v_old = v[s]
                v[s] = compute_value(s, game, gamma, policy, v)
                delta = max(delta, v[s] - v_old)

        if delta < max_delta:
            value_convergence = True
        if value_convergence:
            done = True
            
    return policy, v


def main():
    argparser = ArgumentParser()
    argparser.add_argument("--map-name", type=str, default="suffer")
    argparser.add_argument("--gamma", type=float, default=.9)
    argparser.add_argument("--max-delta", type=float, default=1e-8)

    args = argparser.parse_args()
    game = Maze(args.map_name)
    print("Policy iteration:")
    policy, v = policy_iteration(game, args)
    game.print_values(v)
    game.print_policy(policy)

    print("Value iteration:")
    policy, v = value_iteration(game, args)
    game.print_values(v)
    game.print_policy(policy)


if __name__ == "__main__":
    main()

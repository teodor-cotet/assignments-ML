
import random
import pygame, sys
from argparse import ArgumentParser
from pygame.locals import *

WHITE = (255,255,255)
GRAY = (128, 128, 128)
BLACK = (0, 0 , 0)
square_size = 40
# define event handlers
BALL = 0
PADDLE1 = 1
PADDLE2 = 2
W = 3
H = 4
NR_ATTR = 5

BALL_POS = 0
BALL_VEL = 1

PLAYER1 = 0
PLAYER2 = 1
POINTS = 1
GAME_SPEED = 15
MAX_MOVES = 1800

MOVE_UP = -1
MOVE_DOWN = 1
MOVE_NOT = 0

TRAINING_ADV = 0.1
EVAL_ADV = 0.1
PLAY_ADV = 0.1

RAND = 0
GREEDY = 1
E_GREEDY = 2
BEST_Q_ACTION = 3
BEST_ALL_ACTION = 4


def init(args):

    h = args.h
    w = args.w
    p = args.p

    mh = h // 2

    add = 1 if p % 2 == 0 else 0
    paddle1_pos = (mh - p//2 , mh + p//2 - add)
    paddle2_pos = (mh - p//2 , mh + p//2 - add)
    state = (ball_init(args), paddle1_pos, paddle2_pos, args.w, args.h)

    return state

def ball_init(args):
    h = args.h
    w = args.w
    p = args.p
    ball_pos = (w // 2, h //2)
    horz = random.choice([-1, 1])
    vert = random.choice([-1, 1])
        
    ball_vel = (horz, vert)

    return (ball_pos, ball_vel)


def draw_square(canvas, col,  x, y):
    global square_size
    pygame.draw.polygon(canvas, col, [(x * square_size, y * square_size), (x * square_size, (y + 1) * square_size) , \
     ((x + 1) * square_size, (y + 1) * square_size) , ((x + 1) * square_size, y * square_size)])


def draw(canvas, state):

    global square_size
    canvas.fill(WHITE)

    w = state[W]
    h = state[H]

    ww = square_size * w
    hh = square_size * h

    for i in range(0, h):
        pygame.draw.line(canvas, GRAY, (0, i * square_size), (ww - 1, i * square_size), 1)

    for i in range(0, w):
        pygame.draw.line(canvas, GRAY, (i * square_size, 0), (i * square_size, hh - 1), 1)              

    paddle1_pos = state[PADDLE1]
    paddle2_pos = state[PADDLE2]

    for i in range(paddle1_pos[0], paddle1_pos[1] + 1):
        draw_square(canvas, BLACK, 0, i)

    for i in range(paddle2_pos[0], paddle2_pos[1] + 1):
        draw_square(canvas, BLACK, w - 1, i)
    draw_square(canvas, BLACK, state[BALL][BALL_POS][0], state[BALL][BALL_POS][1])


def get_next_ball(p, n, w, h, paddle1, paddle2):

    np = [p[0] + n[0], p[1] + n[1]]

    center1 = (paddle1[0] + paddle1[1]) // 2
    center_exist1 = False if (paddle1[1] - paddle1[0] - 1) % 2 == 0 else True
    center2 = (paddle2[0] + paddle2[1]) // 2
    center_exist2 = False if (paddle2[1] - paddle2[0] - 1) % 2 == 0 else True
    nn = None
    # paddle hit (or miss), change velocity or loose the game
    if np[0] == 0:
        if (np[1] == -1 and paddle1[0] <= 1) or (np[1] == h and paddle1[1] >= h - 2): # corner case
            nn = (-n[0], -n[1])
        elif paddle1[0] <= np[1] and np[1] <= paddle1[1]: # normal paddle hit
            if np[1] <= center1:
                nn = (-n[0], -1)
            else:
                nn = (-n[0], 1)
            if center_exist1 and np[1] == center1:
                nn = (-n[0], 0)
        else: # missed the paddle
            return -1
    elif np[0] == w - 1:
        if (np[1] == -1 and paddle2[0] <= 1) or (np[1] == h and paddle2[1] >= h - 2): # corner case
            nn = (-n[0], n[1])
        elif paddle2[0] <= np[1] and np[1] <= paddle2[1]: # normal paddle hit
            if np[1] <= center2:
                nn = (-n[0], -1)
            else:
                nn = (-n[0], 1)
            if center_exist2 and np[1] == center2:
                nn = (-n[0], 0)
        else: # missed the paddle
            return 1
    elif np[1] == -1 or np[1] == h: # wall hit
        nn = (n[0], -n[1])
    else: # nothing hit
        nn = (n[0], n[1])
    # compute next position
    np = (p[0] + nn[0] , p[1] + nn[1])
    return (np, nn)

def move_paddle(m, paddle, state):
    if m == MOVE_DOWN:
        if paddle[0] > 0:
            return (paddle[0] - 1, paddle[1] - 1)
    elif m == MOVE_UP:
        if paddle[1] < state[H] - 1:
            return (paddle[0] + 1, paddle[1] + 1)
    return paddle

def apply_action(state, a):
    new_state = [0] * NR_ATTR
    new_state[H] = state[H]
    new_state[W] = state[W]

    new_state[PADDLE1] = move_paddle(a[PLAYER1], state[PADDLE1], state)
    new_state[PADDLE2] = move_paddle(a[PLAYER2], state[PADDLE2], state)

    r = get_next_ball(state[BALL][BALL_POS], state[BALL][BALL_VEL], state[W], state[H], new_state[PADDLE1], new_state[PADDLE2])

    if r == 1 or r == -1:
        return r, r * POINTS
    else:
        new_state[BALL] = r

    return tuple(new_state), 0

def is_final_state(state, turns):
    if state == -1 or state == 1:
        return True
    if turns > MAX_MOVES:
        return True
    return False

def get_legal_actions(state):
    acts = [MOVE_DOWN, MOVE_NOT, MOVE_UP]
    return acts

def eliminateUselessInfo(state):
    if state == 1 or state == -1:
        return state
    return (state[BALL], state[PADDLE1])

def mirror_state(state):
    # mirror in regard to h/2 axis
    b = state[BALL]
    w = state[W]

    new_ball = ( (w - b[BALL_POS][0] - 1, b[BALL_POS][1]),  (-b[BALL_VEL][0], b[BALL_VEL][1]) )
    return (new_ball, state[PADDLE2], state[PADDLE1], state[W], state[H])

def epsilon_greedy(Q, state, legal_actions, epsilon):
    unexplored_actions = []

    state = eliminateUselessInfo(state)
    for a in legal_actions:
        if (state, a) not in Q:
            unexplored_actions.append(a)
    # if is an action which is not explored go for that
    if len(unexplored_actions) > 0:
        return random.choice(unexplored_actions)

    x = random.uniform(0.0, 1.0)

    if x <= epsilon:
        return random.choice(legal_actions)
    else:
        return best_action(Q, state, legal_actions)


def best_action(Q, state, legal_actions):

    # choose just from explored actions
    explored = []

    state = eliminateUselessInfo(state)
    for a in legal_actions:
        if (state, a) in Q:
            explored.append(a)
    # nothing explored, then choose an action randomly
    if len(explored) == 0:
        return random.choice(legal_actions)

    maxAction = explored[0]

    for a in explored:
        if Q[(state, a)] > Q[(state, maxAction)]:
            maxAction = a
    
    return maxAction

def best_game_action(state, PADDLE, chance):

    x = random.uniform(0.0, 1.0)
    if x < chance:
        return random.choice([MOVE_DOWN, MOVE_UP, MOVE_NOT])

    center = (state[PADDLE][0] + state[PADDLE][1]) // 2
    if center < state[BALL][BALL_POS][1]:
        return MOVE_UP
    elif center > state[BALL][BALL_POS][1]:
        return MOVE_DOWN
    else:
        return MOVE_NOT

def playOneGame(args, Q, clusters, split, strategy1, strategy2):

    score = 0
    state = init(args)
    turns = 0
   
    while not is_final_state(state, turns):

        actions = get_legal_actions(state)

        if strategy1 == E_GREEDY:
            action1 = epsilon_greedy(Q, state, actions, args.epsilon)
        elif strategy1 == RAND:
            action1 = random.choice(actions)
        elif strategy1 == GREEDY:
            action1 = epsilon_greedy(Q, state, actions, 0.0)
        elif strategy1 == BEST_Q_ACTION:
            action1 = best_action(Q, state, actions)
        elif strategy1 == BEST_ALL_ACTION:
            action1 = best_game_action(state, PADDLE1, TRAINING_ADV)


        if strategy2 == E_GREEDY:
            action2 = epsilon_greedy(Q, mirror_state(state), actions, args.epsilon) # use the same Q but from player2 perspective
        elif strategy2 == RAND:
            action2 = random.choice(actions)
        elif strategy2 == GREEDY:
            action2 = epsilon_greedy(Q, mirror_state(state), actions, 0.0)
        elif strategy2 == BEST_Q_ACTION:
            action2 = best_action(Q, mirror_state(state), actions)
        elif strategy2 == BEST_ALL_ACTION:
            action2 = best_game_action(state, PADDLE2, TRAINING_ADV)

        state_next, reward = apply_action(state, [action1, action2])
        score += reward

        maxDif = 0
        fst = False
        actions_all_next = get_legal_actions(state_next)
        
        # estimate future value
        state = eliminateUselessInfo(state)
        stateC = find_closest(state, clusters, split, args)
        if (stateC, action1) not in Q:
            Q[(stateC, action1)] = 0
        state_nexte = eliminateUselessInfo(state_next)
        state_nexteC = find_closest(state_nexte, clusters, split, args)

        for a in actions_all_next:

            if (state_nexteC, a) not in Q:
                continue
            # if there is a path explored from here we take it into account and we don't assign 0
            elif fst == False:
                fst = True
                maxDif = -POINTS

            if maxDif < Q[(state_nexteC, a)]:
                maxDif = Q[(state_nexteC, a)]
        Q[(stateC, action1)] = (1 - args.learning_rate) * Q[(stateC, action1)] + args.learning_rate \
        * (reward + args.discount * maxDif)

        state = state_next
        turns += 1
    return score

def find_closest(state, clusters, split, args):
    if state == 1 or state == -1:
        return state
    h = args.h
    w = args.w
    lw = w // split

    pos = state[BALL][BALL_POS]

    for i in range(split):
        st = i * lw
        dr = (i + 1) * lw
        if pos[0] >= st and pos[0] <= dr:
            for j in range(-i, i + 1):
                for k in range(-i, i + 1):
                    if (pos[0] + j, pos[1] + k) in clusters:
                        return ( ( (pos[0] + j, pos[1] + k) , state[BALL][BALL_VEL]), state[PADDLE1])
    return state

def q_learning(args, clusters, split):

    Q = {}
    train_scores = []
    eval_scores = []
    avg_score = 0.0

    for train_ep in range(args.train_episodes):
        score = playOneGame(args, Q, clusters, split,  E_GREEDY, BEST_ALL_ACTION)
        train_scores.append(score)

        if train_ep % args.eval_every == 0:
            avg_score = .0
            
            for i in range(args.eval_episodes):
                score = playOneGame(args, Q, clusters, split, BEST_Q_ACTION, BEST_ALL_ACTION)
                avg_score += score
            
            eval_scores.append(avg_score / (args.eval_episodes))
            print(avg_score / (args.eval_episodes))

    # play one game
    if args.final_show:

        window = pygame.display.set_mode((args.w * square_size, args.h * square_size), 0, 32)
        pygame.display.set_caption('Ping Pong')
        fps = pygame.time.Clock()
        state = init(args)

        final_score = 0
        turns = 0
        while not is_final_state(state, turns):
            draw(window, state)
            #action1 = best_action(Q, state, legal_actions)
            action1 = best_game_action(state, PADDLE1, PLAY_ADV)
            action2 = best_game_action(state, PADDLE2, PLAY_ADV) 
            state, reward = apply_action(state, [action1, action2])
            final_score += reward
            turns += 1

            pygame.display.update()
            fps.tick(GAME_SPEED)
    
    if args.plot_scores:
        from matplotlib import pyplot as plt
        import numpy as np
        plt.xlabel("Episode")
        plt.ylabel("Average score")
        plt.plot(
            np.linspace(1, args.train_episodes, args.train_episodes),
            np.convolve(train_scores, [0.2,0.2,0.2,0.2,0.2], "same"),
            linewidth = 1.0, color = "blue"
        )
        plt.plot(
            np.linspace(args.eval_every, args.train_episodes, len(eval_scores)),
            eval_scores, linewidth = 2.0, color = "red"
        )
        plt.show()

def put(clusters, dim, start, last, h):

    bucket = dim * 2 + 1
    for j in range(start, last + 1, bucket):
        for k in range(0, h + 1, bucket):
            if j + dim <= last and k + dim <= h:
                clusters[(j +  dim, k + dim)] = 0

def init_clusters(clusters, args, split):
    h = args.h
    w = args.w
    lw = w // split
    last = -1
    for i in range(split):
        start = last + 1
        last = last + lw
        if i == split - 1:
            last = w - 1
        put(clusters, i, start, last, h - 1)


if __name__ == "__main__":
    parser = ArgumentParser()
    # game parameters
    parser.add_argument("--w", type = int, default = 18, help = "Width")
    parser.add_argument("--h", type = int, default = 9, help = "Height")
    parser.add_argument("--p", type = int, default = 3, help = "Palette")
    
    # Q-learning params    
    parser.add_argument("--learning_rate", type = float, default = 0.4, help = "Learning rate")
    parser.add_argument("--discount", type = float, default = 0.999, help = "Value for the discount factor")
    parser.add_argument("--epsilon", type = float, default = 0.06, help = "Probability to choose a random action.")

    parser.add_argument("--train_episodes", type = int, default = 500, help = "Number of episodes")
    parser.add_argument("--eval_every", type = int, default = 50, help = "Evaluate policy every ... games.")
    parser.add_argument("--eval_episodes", type = float, default = 120, help = "Number of games to play for evaluation.")

    parser.add_argument("--final_show", dest="final_show", action="store_true", help = "Demonstrate final strategy.")
    parser.add_argument("--plot", dest="plot_scores", action="store_true", help = "Plot scores in the end")
    args = parser.parse_args()
    clusters = {}
    split = 3
    #init_clusters(clusters, args, split)
    
    state = init(args)
    q_learning(args, clusters, split)

# Tudor Berariu, 2016

# General imports
from copy import copy
from random import choice, random, uniform
from argparse import ArgumentParser
from time import sleep

# Game functions
from mini_pacman import ( get_initial_state,       # get initial state from file
                          get_legal_actions,  # get the legal actions in a state
                          is_final_state,         # check if a state is terminal
                          apply_action,       # apply an action in a given state
                          display_state )            # display the current state

def epsilon_greedy(Q, state, legal_actions, epsilon):
    # TODO (2) : Epsilon greedy
    unexplored_actions = []

    for a in legal_actions:
        if (state, a) not in Q:
            unexplored_actions.append(a)
    # if is an action which is not explored go for that
    if len(unexplored_actions) > 0:
        return choice(unexplored_actions)

    x = uniform(0.0, 1.0)

    if x <= epsilon:
        return choice(legal_actions)
    else:
        return best_action(Q, state, legal_actions)


def best_action(Q, state, legal_actions):

    # choose just from explored actions

    explored = []

    for a in legal_actions:
        if (state, a) in Q:
            explored.append(a)
    # nothing explored
    if len(explored) == 0:
        return choice(legal_actions)

    maxAction = explored[0]

    for a in explored:
        if Q[(state, a)] > Q[(state, maxAction)]:
            maxAction = a
    
    return maxAction

def q_learning(args):
    Q = {}
    train_scores = []
    eval_scores = []
    avg_score = 0.0
                                                          # for each episode ...
    for train_ep in range(0, args.train_episodes):
        score = 0
        state = get_initial_state(args.map_file)
        if args.verbose:
            display_state(state); sleep(args.sleep)

                                           # while current state is not terminal
        while not is_final_state(state, score):
                                               # choose one of the legal actions
            actions = get_legal_actions(state)
            action = epsilon_greedy(Q, state, actions, args.epsilon)

                            # apply action and get the next state and the reward
            state2, reward, msg = apply_action(state, action)
            score += reward

            # TODO (1) : Q-Learning
            maxDif = 0
            fst = False
            actions2 = get_legal_actions(state2)
            
            # print(state)
            #print(score)
            # estimate future value
            if (state, action) not in Q:
                Q[(state, action)] = 0

            for a in actions2:

                if (state2, a) not in Q:
                    continue
                elif fst == False:
                    fst = True
                    maxDif = -20

                if maxDif < Q[(state2, a)]:
                    maxDif = Q[(state2, a)]
            
            Q[(state, action)] = (1 - args.learning_rate) * Q[(state, action)] + args.learning_rate * \
            (reward + args.discount * maxDif)
            # display current state and sleep
            state = state2
            if args.verbose:
                print(msg); display_state(state); sleep(args.sleep)
       
        #print("Episode %6d / %6d" % (train_ep, args.train_episodes))
        train_scores.append(score)
                                                    # evaluate the greedy policy
        # we want to see how it has been improved after  eval_every iterations
        # construct policy
        if train_ep % args.eval_every == 0:
            avg_score = 0.0
            for i in range(0, args.eval_episodes):
                score = 0
                state = get_initial_state(args.map_file)
                # now istead of letting the algorithm a chance to explore we go greedy
                while not is_final_state(state, score):
                    legal_actions = get_legal_actions(state)
                    bestAct = best_action(Q, state, legal_actions)
                    state, reward, msg = apply_action(state, bestAct)
                    score += reward
                avg_score += score
                # TODO (4) : Evaluate
            eval_scores.append(avg_score / args.eval_episodes)
            print(avg_score / args.eval_episodes)

        if train_ep >= args.train_episodes:
            score = 0
            state = get_initial_state(args.map_file)
            # now istead of letting the algorithm a chance to explore we go greedy
            while not is_final_state(state, score):
                legal_actions = get_legal_actions(state)
                bestAct = best_action(Q, state, legal_actions)
                state, reward, msg = apply_action(state, bestAct)
                score += reward
            avg_score += score

    print(avg_score / args.eval_episodes)
    # --------------------------------------------------------------------------
    if args.final_show:
        state = get_initial_state(args.map_file)
        final_score = 0
        while not is_final_state(state, final_score):
            action = best_action(Q, state, get_legal_actions(state))
            state, reward, msg = apply_action(state, action)
            final_score += reward
            print(msg); display_state(state); sleep(args.sleep)

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

if __name__ == "__main__":
    parser = ArgumentParser()
    # Input file
    parser.add_argument("--map_file", type = str, default = "mini_map",
                        help = "File to read map from.")
    # Meta-parameters
    parser.add_argument("--learning_rate", type = float, default = 0.25,
                        help = "Learning rate")
    parser.add_argument("--discount", type = float, default = 0.9999999999999,
                        help = "Value for the discount factor")
    parser.add_argument("--epsilon", type = float, default = 0.1,
                        help = "Probability to choose a random action.")
    # Training and evaluation episodes
    parser.add_argument("--train_episodes", type = int, default = 4000,
                        help = "Number of episodes")
    parser.add_argument("--eval_every", type = int, default = 10,
                        help = "Evaluate policy every ... games.")
    parser.add_argument("--eval_episodes", type = int, default = 10,
                        help = "Number of games to play for evaluation.")
    # Display
    parser.add_argument("--verbose", dest="verbose",
                        action = "store_true", help = "Print each state")
    parser.add_argument("--plot", dest="plot_scores", action="store_true",
                        help = "Plot scores in the end")
    parser.add_argument("--sleep", type = float, default = 0.1,
                        help = "Seconds to 'sleep' between moves.")
    parser.add_argument("--final_show", dest = "final_show",
                        action = "store_true",
                        help = "Demonstrate final strategy.")
    args = parser.parse_args()
    q_learning(args)
#small 
# discount = 0.99
# epsilon = 0.05
# learning_rate = 0.25

# big 
# 
# 
# discount = 0.99
# epsilon = 0.002
# learning_rate =  0.47
# 1400 converge, dar cu scor 5-6
# epsilin mare (0.1), solutii mai bune, learning rate mic => solutii mai buend ar mai greud e gasit


# huge
# discount = 0.99
# epsilon = 0.004
# learning_Rate = 0.6
# 1700 converge, scor -1.0
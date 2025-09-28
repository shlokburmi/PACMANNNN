# ghost_agents.py
# ---------------
# Licensing Information: You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# and Dan Klein. Student side autograding was added by Brad Miller, Nick Hay,
# and Pieter Abbeel.

from game import Agent, Actions, Directions
import random
from util import manhattanDistance
import util


class GhostAgent(Agent):
    def __init__(self, index):
        self.index = index

    def get_action(self, state):
        dist = self.get_distribution(state)
        if not dist or len(dist) == 0:
            return Directions.STOP
        
        # Robust weighted random choice
        total = float(sum(dist.values()))
        if total == 0:
            # If all probabilities are zero, choose a random legal action
            legal_actions = state.get_legal_actions(self.index)
            if not legal_actions:
                return Directions.STOP
            return random.choice(legal_actions)

        r = random.uniform(0, total)
        upto = 0
        for action, prob in dist.items():
            if upto + prob >= r:
                return action
            upto += prob
        # Fallback to a random choice from distribution keys if something goes wrong
        return random.choice(list(dist.keys()))

    def get_distribution(self, state):
        """
        Returns a Counter encoding a distribution over actions from the provided state.
        """
        util.raise_not_defined()


class RandomGhost(GhostAgent):
    "A ghost that chooses a legal action uniformly at random."
    def get_distribution(self, state):
        dist = util.Counter()
        for a in state.get_legal_actions(self.index):
            dist[a] = 1.0
        dist.normalize()
        return dist


class DirectionalGhost(GhostAgent):
    "A ghost that prefers to rush Pacman, or flee when scared."
    def __init__(self, index, prob_attack=0.8, prob_scaredFlee=0.8):
        self.index = index
        self.prob_attack = prob_attack
        self.prob_scaredFlee = prob_scaredFlee

    def get_distribution(self, state):
        # Read variables from the state
        ghost_state = state.get_ghost_state(self.index)
        legal_actions = state.get_legal_actions(self.index)
        pos = state.get_ghost_position(self.index)
        is_scared = ghost_state.scared_timer > 0

        speed = 1
        if is_scared:
            speed = 0.5

        action_vectors = [Actions.direction_to_vector(a, speed) for a in legal_actions]
        new_positions = [(pos[0] + v[0], pos[1] + v[1]) for v in action_vectors]
        pacman_position = state.get_pacman_position()

        # Select best actions given the state
        distances_to_pacman = [manhattanDistance(p, pacman_position) for p in new_positions]
        if is_scared:
            best_score = max(distances_to_pacman)
            best_prob = self.prob_scaredFlee
        else:
            best_score = min(distances_to_pacman)
            best_prob = self.prob_attack
            
        best_actions = [action for action, distance in zip(legal_actions, distances_to_pacman) if distance == best_score]

        # Construct distribution
        dist = util.Counter()
        for a in best_actions:
            dist[a] = best_prob / len(best_actions)
        for a in legal_actions:
            dist[a] += (1 - best_prob) / len(legal_actions)

        if len(dist) == 0:
            return None
        dist.normalize()
        return dist


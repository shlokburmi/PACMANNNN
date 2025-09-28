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

import random
import util
from game import Agent, Actions, Directions
from util import manhattanDistance  # keep name as exported by util in this codebase


class GhostAgent(Agent):
    """
    Base ghost agent with robust distribution-based selection.
    """

    def __init__(self, index):
        self.index = index

    def get_action(self, state):
        dist = self.get_distribution(state)
        if not dist or len(dist) == 0:
            return Directions.STOP

        total = sum(dist.values())
        if total == 0:
            keys = list(dist.keys())
            return random.choice(keys) if keys else Directions.STOP

        r = random.uniform(0, total)
        upto = 0.0
        for action, prob in dist.items():
            if upto + prob >= r:
                return action
            upto += prob
        keys = list(dist.keys())
        return random.choice(keys) if keys else Directions.STOP

    def get_distribution(self, state):
        util.raise_not_defined()


class RandomGhost(GhostAgent):
    """
    Chooses legal actions uniformly at random.
    """

    def get_distribution(self, state):
        dist = util.Counter()
        for a in state.get_legal_actions(self.index):
            dist[a] = 1.0
        dist.normalize()
        return dist


class DirectionalGhost(GhostAgent):
    """
    Rushes Pacman when brave, flees when scared.
    """

    def __init__(self, index, prob_attack=0.8, prob_scared_flee=0.8):
        self.index = index
        self.prob_attack = prob_attack
        self.prob_scared_flee = prob_scared_flee

    def get_distribution(self, state):
        ghost_state = state.get_ghost_state(self.index)
        legal = state.get_legal_actions(self.index)
        pos = state.get_ghost_position(self.index)
        scared = ghost_state.scared_timer > 0
        speed = 0.5 if scared else 1.0

        vectors = [Actions.direction_to_vector(a, speed) for a in legal]
        next_positions = [(pos[0] + vx, pos[1] + vy) for (vx, vy) in vectors]
        pac = state.get_pacman_position()
        dists = [manhattanDistance(p, pac) for p in next_positions]

        if scared:
            best_score = max(dists)
            best_prob = self.prob_scared_flee
        else:
            best_score = min(dists)
            best_prob = self.prob_attack

        best_actions = [a for a, d in zip(legal, dists) if d == best_score]

        dist = util.Counter()
        if best_actions:
            for a in best_actions:
                dist[a] = best_prob / len(best_actions)
        if legal:
            for a in legal:
                dist[a] += (1.0 - best_prob) / len(legal)
        dist.normalize()
        return dist


class MinimaxGhost(GhostAgent):
    """
    A ghost using minimax with a simple evaluation.
    Note: agent index 0 is Pacman; ghosts are >= 1.
    """

    def __init__(self, index, depth=2):
        self.index = index
        self.depth = depth

    def get_action(self, game_state):
        _, action = self._minimax(game_state, self.depth * game_state.get_num_agents(), self.index)
        return action if action is not None else Directions.STOP

    def _minimax(self, state, depth, agent_index):
        if state.is_win() or state.is_lose() or depth == 0:
            return self._evaluate(state), None

        num_agents = state.get_num_agents()
        next_agent = (agent_index + 1) % num_agents
        actions = state.get_legal_actions(agent_index)
        if not actions:
            return self._evaluate(state), None

        results = []
        for a in actions:
            succ = state.generate_successor(agent_index, a)
            score, _ = self._minimax(succ, depth - 1, next_agent)
            results.append((score, a))

        if agent_index == 0:
            return max(results, key=lambda x: x[0])
        return min(results, key=lambda x: x[0])

    def _evaluate(self, state):
        score = state.get_score()
        pac = state.get_pacman_position()
        min_ghost = float('inf')
        for g in state.get_ghost_states():
            if g.scared_timer == 0:
                d = manhattanDistance(pac, g.get_position())
                if d < min_ghost:
                    min_ghost = d
        if min_ghost != float('inf') and min_ghost <= 1:
            return -float('inf')
        if min_ghost != float('inf'):
            score -= 1.5 / min_ghost
        score -= 2 * state.get_num_food()
        score -= 20 * len(state.get_capsules())
        return score

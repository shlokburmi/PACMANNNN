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
    """
    Base ghost agent class, with a robust action selection method.
    """
    def __init__(self, index):
        self.index = index

    def get_action(self, state):
        """
        Selects an action robustly from the distribution.
        """
        dist = self.get_distribution(state)
        if not dist or len(dist) == 0:
            return Directions.STOP
        
        # Manually implement weighted random choice for maximum compatibility
        total = sum(dist.values())
        if total == 0:
            return random.choice(list(dist.keys())) if list(dist.keys()) else Directions.STOP
        choice = random.uniform(0, total)
        upto = 0
        for action, prob in dist.items():
            if upto + prob >= choice:
                return action
            upto += prob
        return random.choice(list(dist.keys()))


    def get_distribution(self, state):
        util.raise_not_defined()


class RandomGhost(GhostAgent):
    """
    A ghost that chooses a legal action uniformly at random.
    """
    def get_distribution(self, state):
        dist = util.Counter()
        for a in state.get_legal_actions(self.index):
            dist[a] = 1.0
        dist.normalize()
        return dist


class DirectionalGhost(GhostAgent):
    """
    A ghost that prefers to rush Pacman, or flee when scared.
    """
    def __init__(self, index, prob_attack=0.8, prob_scaredFlee=0.8):
        self.index = index
        self.prob_attack = prob_attack
        self.prob_scaredFlee = prob_scaredFlee

    def get_distribution(self, state):
        ghost_state = state.get_ghost_state(self.index)
        legal_actions = state.get_legal_actions(self.index)
        pos = state.get_ghost_position(self.index)
        is_scared = ghost_state.scared_timer > 0
        speed = 0.5 if is_scared else 1.0

        action_vectors = [Actions.direction_to_vector(a, speed) for a in legal_actions]
        new_positions = [(pos[0] + v[0], pos[1] + v[1]) for v in action_vectors]
        pacman_position = state.get_pacman_position()

        distances_to_pacman = [manhattanDistance(p, pacman_position) for p in new_positions]
        if is_scared:
            best_score = max(distances_to_pacman)
            best_prob = self.prob_scaredFlee
        else:
            best_score = min(distances_to_pacman)
            best_prob = self.prob_attack

        best_actions = [action for action, distance in zip(legal_actions, distances_to_pacman)
                       if distance == best_score]

        dist = util.Counter()
        if best_actions:
            for a in best_actions:
                dist[a] = best_prob / len(best_actions)
        if legal_actions:
            for a in legal_actions:
                dist[a] += (1 - best_prob) / len(legal_actions)
        dist.normalize()
        return dist

class MinimaxGhost(GhostAgent):
    """
    An intelligent ghost using minimax with an improved evaluation function.
    """
    def __init__(self, index, depth=2):
        self.index = index
        self.depth = depth

    def get_action(self, gameState):
        _, bestAction = self._minimax(gameState, self.depth * gameState.get_num_agents(), self.index)
        return bestAction if bestAction is not None else Directions.STOP

    def _minimax(self, gameState, depth, agentIndex):
        if gameState.is_win() or gameState.is_lose() or depth == 0:
            return (self._evaluate(gameState), None)

        numAgents = gameState.get_num_agents()
        nextAgent = (agentIndex + 1) % numAgents
        
        legalActions = gameState.get_legal_actions(agentIndex)
        if not legalActions:
            return (self._evaluate(gameState), None)

        results = []
        for action in legalActions:
            successorState = gameState.generate_successor(agentIndex, action)
            score, _ = self._minimax(successorState, depth - 1, nextAgent)
            results.append((score, action))

        if agentIndex == 0: # Pacman is agent 0 (max player)
            return max(results, key=lambda x: x[0])
        else: # Ghosts are other agents (min players)
            return min(results, key=lambda x: x[0])

    def _evaluate(self, state):
        score = state.get_score()
        pacmanPos = state.get_pacman_position()
        
        min_dist_to_ghost = float('inf')
        for ghost_state in state.get_ghost_states():
            if ghost_state.scared_timer == 0:
                dist = manhattanDistance(pacmanPos, ghost_state.get_position())
                if dist < min_dist_to_ghost:
                    min_dist_to_ghost = dist
        
        if min_dist_to_ghost <= 1:
            return -float('inf')
        
        # Penalize being close to ghosts and reward being far away
        score -= 1.5 / min_dist_to_ghost
        
        # Penalize for remaining food and capsules
        score -= 2 * state.get_num_food()
        score -= 20 * len(state.get_capsules())
        return score


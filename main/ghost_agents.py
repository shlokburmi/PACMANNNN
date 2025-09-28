# ghost_agents.py
# ---------------
import random
import util
from game import Agent, Actions, Directions
from util import manhattanDistance

class GhostAgent(Agent):
    def __init__(self, index): self.index = index
    def get_action(self, state):
        dist = self.get_distribution(state)
        if not dist or len(dist) == 0: return Directions.STOP
        total = float(sum(dist.values()))
        if total <= 0:
            keys = list(dist.keys()); return random.choice(keys) if keys else Directions.STOP
        r = random.uniform(0.0, total); upto = 0.0
        for action, prob in dist.items():
            upto += prob
            if r <= upto: return action
        keys = list(dist.keys()); return random.choice(keys) if keys else Directions.STOP
    def get_distribution(self, state): util.raise_not_defined()

class RandomGhost(GhostAgent):
    def get_distribution(self, state):
        dist = util.Counter()
        for a in state.get_legal_actions(self.index): dist[a] = 1.0
        dist.normalize(); return dist

class DirectionalGhost(GhostAgent):
    def __init__(self, index, prob_attack=0.8, prob_scared_flee=0.8):
        self.index = index; self.prob_attack = prob_attack; self.prob_scared_flee = prob_scared_flee
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
        if not dists: return util.Counter()
        if scared:
            best_score = max(dists); best_prob = self.prob_scared_flee
        else:
            best_score = min(dists); best_prob = self.prob_attack
        best_actions = [a for a, d in zip(legal, dists) if d == best_score]
        dist = util.Counter()
        if best_actions:
            for a in best_actions: dist[a] = best_prob / len(best_actions)
        if legal:
            for a in legal: dist[a] += (1.0 - best_prob) / len(legal)
        dist.normalize(); return dist

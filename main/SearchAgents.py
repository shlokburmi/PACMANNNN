from search import SearchProblem
from game import Agent, Actions, Directions
import random
import util
import time


class FoodSearchProblem(SearchProblem):
    """
    A search problem associated with finding a path that collects all the food
    (dots) in a Pacman game.
    """

    def __init__(self, startingGameState):
        self.start = (startingGameState.get_pacman_position(), startingGameState.get_food())
        self.walls = startingGameState.get_walls()
        self.startingGameState = startingGameState

    def get_start_state(self):
        return self.start

    def is_goal_state(self, state):
        return state[1].count() == 0  # All food eaten

    def get_successors(self, state):
        successors = []
        x, y = state[0]
        food = state[1]
        for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
            nextx, nexty = x + dx, y + dy
            if not self.walls[nextx][nexty]:
                nextFood = food.copy()
                nextFood[nextx][nexty] = False
                action = Actions.vector_to_direction((dx, dy))  # convert vector to action string
                successors.append((((nextx, nexty), nextFood), action, 1))
        return successors

    def get_cost_of_actions(self, actions):
        if actions is None:
            return 999999
        x, y = self.start[0]
        cost = 0
        for action in actions:
            dx, dy = Actions.direction_to_vector(action)
            x, y = x + dx, y + dy
            if self.walls[x][y]:
                return 999999
            cost += 1
        return cost


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
        distances_to_pacman = [util.manhattan_distance(p, pacman_position) for p in new_positions]
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

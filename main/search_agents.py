# search_agents.py
# ----------------
from __future__ import print_function
from builtins import str, object
import time
import util
import search
from game import Directions, Agent, Actions
from search import manhattan_heuristic, euclidean_heuristic

class GoWestAgent(Agent):
    def get_action(self, state):
        if Directions.WEST in state.get_legal_pacman_actions():
            return Directions.WEST
        return Directions.STOP

class SearchAgent(Agent):
    def __init__(self, fn='depth_first_search', prob='PositionSearchProblem', heuristic='null_heuristic'):
        if fn not in dir(search): raise AttributeError(fn + ' is not a search function in search.py.')
        func = getattr(search, fn)
        if 'heuristics' in func.__code__.co_varnames:
            self.search_function = lambda x: func(x, heuristics=[manhattan_heuristic, euclidean_heuristic])
        elif 'heuristic' in func.__code__.co_varnames:
            if hasattr(search, heuristic):
                heur = getattr(search, heuristic)
            else:
                raise AttributeError(heuristic + ' is not in search.py.')
            self.search_function = lambda x: func(x, heuristic=heur)
        else:
            self.search_function = func
        if prob not in globals() or not prob.endswith('Problem'): raise AttributeError(prob + ' is not a search problem type.')
        self.search_type = globals()[prob]

    def register_initial_state(self, state):
        t0 = time.time()
        problem = self.search_type(state)
        self.actions = self.search_function(problem)
        cost = problem.get_cost_of_actions(self.actions)
        print('Path found with total cost of %d in %.2f seconds' % (cost, time.time()-t0))
        if '_expanded' in dir(problem): print('Search nodes expanded: %d' % problem._expanded)

    def get_action(self, state):
        if 'action_index' not in dir(self): self.action_index = 0
        if self.action_index < len(self.actions):
            a = self.actions[self.action_index]; self.action_index += 1; return a
        return Directions.STOP

class PositionSearchProblem(search.SearchProblem):
    def __init__(self, game_state, cost_fn=lambda x: 1, goal=(1,1), start=None, warn=True, visualize=True):
        self.walls = game_state.get_walls()
        self.start_state = game_state.get_pacman_position() if start is None else start
        self.goal = goal; self.cost_fn = cost_fn; self.visualize = visualize
        self._visited, self._visitedlist, self._expanded = {}, [], 0

    def get_start_state(self): return self.start_state

    def is_goal_state(self, state):
        is_goal = (state == self.goal)
        if is_goal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'draw_expanded_cells' in dir(__main__._display):
                    __main__._display.draw_expanded_cells(self._visitedlist)
        return is_goal

    def get_successors(self, state):
        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y = state
            dx, dy = Actions.direction_to_vector(action)
            nx, ny = int(x + dx), int(y + dy)
            if not self.walls[nx][ny]:
                successors.append(((nx, ny), action, self.cost_fn((nx, ny))))
        self._expanded += 1
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)
        return successors

    def get_cost_of_actions(self, actions):
        if actions is None: return 999999
        x, y = self.get_start_state(); cost = 0
        for a in actions:
            dx, dy = Actions.direction_to_vector(a)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.cost_fn((x, y))
        return cost

def maze_distance(p1, p2, game_state):
    x1, y1 = p1; x2, y2 = p2
    walls = game_state.get_walls()
    assert not walls[x1][y1], 'point1 is a wall: ' + str(p1)
    assert not walls[x2][y2], 'point2 is a wall: ' + str(p2)
    prob = PositionSearchProblem(game_state, start=p1, goal=p2, warn=False, visualize=False)
    return len(search.breadth_first_search(prob))

class FoodSearchProblem(search.SearchProblem):
    def __init__(self, starting_game_state):
        self.start = (starting_game_state.get_pacman_position(), starting_game_state.get_food())
        self.walls = starting_game_state.get_walls()
        self.starting_game_state = starting_game_state
        self._expanded = 0
        self.heuristic_info = {}

    def get_start_state(self): return self.start
    def is_goal_state(self, state): return state[1].count() == 0

    def get_successors(self, state):
        successors = []; self._expanded += 1
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y = state[0]
            dx, dy = Actions.direction_to_vector(direction)
            nx, ny = int(x + dx), int(y + dy)
            if not self.walls[nx][ny]:
                next_food = state[1].copy()
                next_food[nx][ny] = False
                successors.append((((nx, ny), next_food), direction, 1))
        return successors

    def get_cost_of_actions(self, actions):
        x, y = self.get_start_state()[0]; cost = 0
        for a in actions:
            dx, dy = Actions.direction_to_vector(a)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += 1
        return cost

def food_heuristic(state, problem):
    position, food_grid = state
    food_locs = food_grid.as_list()
    if not food_locs: return 0
    max_d = 0
    for f in food_locs:
        d = maze_distance(position, f, problem.starting_game_state)
        if d > max_d: max_d = d
    return max_d

def mst_food_heuristic(state, problem):
    position, food_grid = state
    food_locs = food_grid.as_list()
    if not food_locs: return 0
    nodes = [position] + food_locs
    if len(nodes) == 2:
        return maze_distance(nodes[0], nodes[1], problem.starting_game_state)
    if 'pair_d' not in problem.heuristic_info:
        problem.heuristic_info['pair_d'] = {}
    edges = []
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            a, b = nodes[i], nodes[j]
            if (a, b) not in problem.heuristic_info['pair_d']:
                d = maze_distance(a, b, problem.starting_game_state)
                problem.heuristic_info['pair_d'][(a,b)] = d; problem.heuristic_info['pair_d'][(b,a)] = d
            d = problem.heuristic_info['pair_d'][(a,b)]
            edges.append((d, a, b))
    parent = {v: v for v in nodes}
    def find(v):
        if parent[v] != v: parent[v] = find(parent[v]); return parent[v]
        return v
    def unite(a,b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[rb] = ra
    edges.sort(); cost = 0
    for d, u, v in edges:
        if find(u) != find(v):
            cost += d; unite(u, v)
    return cost

class SMHAFoodSearchAgent(Agent):
    def __init__(self):
        self.search_function = lambda prob: search.smha_search(prob, heuristics=[food_heuristic, mst_food_heuristic])
        self.search_type = FoodSearchProblem
    def register_initial_state(self, state):
        t0 = time.time(); problem = self.search_type(state)
        self.actions = self.search_function(problem)
        cost = problem.get_cost_of_actions(self.actions)
        print('Path found with total cost of %d in %.2f seconds' % (cost, time.time()-t0))
        if '_expanded' in dir(problem): print('Search nodes expanded: %d' % problem._expanded)
    def get_action(self, state):
        if 'action_index' not in dir(self): self.action_index = 0
        if self.action_index < len(self.actions):
            a = self.actions[self.action_index]; self.action_index += 1; return a
        return Directions.STOP

class AStarFoodSearchAgent(SearchAgent):
    def __init__(self):
        self.search_function = lambda prob: search.a_star_search(prob, food_heuristic)
        self.search_type = FoodSearchProblem

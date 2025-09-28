# search_agents.py
# ---------------
# Licensing Information: You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

from __future__ import print_function

from builtins import str
from builtins import object

import time
import util
import search
from game import Directions, Agent, Actions
from search import manhattan_heuristic, euclidean_heuristic


class GoWestAgent(Agent):
    "An agent that goes West until it can't."

    def get_action(self, state):
        if Directions.WEST in state.get_legal_pacman_actions():
            return Directions.WEST
        return Directions.STOP


class SearchAgent(Agent):
    """
    General-purpose search agent:
      - fn: name of search function in search.py (snake_case)
      - prob: name of problem class in this file (snake_case)
      - heuristic: name of heuristic in search.py (optional)
    """

    def __init__(self, fn='depth_first_search', prob='PositionSearchProblem', heuristic='null_heuristic'):
        if fn not in dir(search):
            raise AttributeError(fn + ' is not a search function in search.py.')
        func = getattr(search, fn)

        # Bind search_function with optional heuristic(s)
        if 'heuristics' in func.__code__.co_varnames:
            my_heuristics = [manhattan_heuristic, euclidean_heuristic]
            self.search_function = lambda x: func(x, heuristics=my_heuristics)
        elif 'heuristic' in func.__code__.co_varnames:
            if hasattr(search, heuristic):
                heur = getattr(search, heuristic)
            else:
                raise AttributeError(f"{heuristic} is not a function in search.py.")
            self.search_function = lambda x: func(x, heuristic=heur)
        else:
            self.search_function = func

        if prob not in globals() or not prob.endswith('Problem'):
            raise AttributeError(prob + ' is not a search problem type in this file.')
        self.search_type = globals()[prob]

    def register_initial_state(self, state):
        if self.search_function is None:
            raise Exception("No search function provided for SearchAgent")
        start_time = time.time()
        problem = self.search_type(state)
        self.actions = self.search_function(problem)
        total_cost = problem.get_cost_of_actions(self.actions)
        print('Path found with total cost of %d in %.2f seconds' % (total_cost, time.time() - start_time))
        if '_expanded' in dir(problem):
            print('Search nodes expanded: %d' % problem._expanded)

    def get_action(self, state):
        if 'action_index' not in dir(self):
            self.action_index = 0
        i = self.action_index
        self.action_index += 1
        if i < len(self.actions):
            return self.actions[i]
        return Directions.STOP


class PositionSearchProblem(search.SearchProblem):
    """
    A simple position search for reaching a (goal_x, goal_y) coordinate.
    """

    def __init__(self, game_state, cost_fn=lambda x: 1, goal=(1, 1), start=None, warn=True, visualize=True):
        self.walls = game_state.get_walls()
        self.start_state = game_state.get_pacman_position()
        if start is not None:
            self.start_state = start
        self.goal = goal
        self.cost_fn = cost_fn
        self.visualize = visualize
        self._visited, self._visitedlist, self._expanded = {}, [], 0

    def get_start_state(self):
        return self.start_state

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
                next_state = (nx, ny)
                cost = self.cost_fn(next_state)
                successors.append((next_state, action, cost))
        self._expanded += 1
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)
        return successors

    def get_cost_of_actions(self, actions):
        if actions is None:
            return 999999
        x, y = self.get_start_state()
        cost = 0
        for action in actions:
            dx, dy = Actions.direction_to_vector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999
            cost += self.cost_fn((x, y))
        return cost


def maze_distance(point1, point2, game_state):
    x1, y1 = point1
    x2, y2 = point2
    walls = game_state.get_walls()
    assert not walls[x1][y1], 'point1 is a wall: ' + str(point1)
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    prob = PositionSearchProblem(game_state, start=point1, goal=point2, warn=False, visualize=False)
    return len(search.breadth_first_search(prob))


class FoodSearchProblem(search.SearchProblem):
    """
    A search problem associated with finding a path that collects all food.
    State is a (pacman_position, food_grid).
    """

    def __init__(self, starting_game_state):
        self.start = (starting_game_state.get_pacman_position(), starting_game_state.get_food())
        self.walls = starting_game_state.get_walls()
        self.starting_game_state = starting_game_state
        self._expanded = 0
        self.heuristic_info = {}

    def get_start_state(self):
        return self.start

    def is_goal_state(self, state):
        return state[1].count() == 0

    def get_successors(self, state):
        successors = []
        self._expanded += 1
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
        x, y = self.get_start_state()[0]
        cost = 0
        for action in actions:
            dx, dy = Actions.direction_to_vector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999
            cost += 1
        return cost


def food_heuristic(state, problem):
    position, food_grid = state
    food_locs = food_grid.as_list()
    if not food_locs:
        return 0
    # Use farthest-maze-distance to a remaining food (admissible but expensive)
    max_d = 0
    for f in food_locs:
        d = maze_distance(position, f, problem.starting_game_state)
        if d > max_d:
            max_d = d
    return max_d


def mst_food_heuristic(state, problem):
    position, food_grid = state
    food_locs = food_grid.as_list()
    if not food_locs:
        return 0
    nodes = [position] + food_locs
    if len(nodes) == 1:
        return 0
    if len(nodes) == 2:
        return maze_distance(nodes[0], nodes[1], problem.starting_game_state)

    if 'mst_distances' not in problem.heuristic_info:
        problem.heuristic_info['mst_distances'] = {}

    # Precompute pairwise shortest (maze) distances and build edges
    edges = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            a, b = nodes[i], nodes[j]
            if (a, b) not in problem.heuristic_info['mst_distances']:
                d = maze_distance(a, b, problem.starting_game_state)
                problem.heuristic_info['mst_distances'][(a, b)] = d
                problem.heuristic_info['mst_distances'][(b, a)] = d
            d = problem.heuristic_info['mst_distances'][(a, b)]
            edges.append((d, a, b))

    # Kruskal MST
    parent = {v: v for v in nodes}

    def find_set(v):
        if parent[v] != v:
            parent[v] = find_set(parent[v])
        return parent[v]

    def union(a, b):
        ra, rb = find_set(a), find_set(b)
        if ra != rb:
            parent[rb] = ra

    edges.sort()
    mst_cost = 0
    for cost, u, v in edges:
        if find_set(u) != find_set(v):
            mst_cost += cost
            union(u, v)
    return mst_cost


class SMHAFoodSearchAgent(Agent):
    """
    Runs SMHA* on FoodSearchProblem with two heuristics:
      - food_heuristic
      - mst_food_heuristic
    """

    def __init__(self):
        my_heuristics = [food_heuristic, mst_food_heuristic]
        self.search_function = lambda prob: search.smha_search(prob, heuristics=my_heuristics)
        self.search_type = FoodSearchProblem

    def register_initial_state(self, state):
        start_time = time.time()
        problem = self.search_type(state)
        self.actions = self.search_function(problem)
        total_cost = problem.get_cost_of_actions(self.actions)
        print('Path found with total cost of %d in %.2f seconds' % (total_cost, time.time() - start_time))
        if '_expanded' in dir(problem):
            print('Search nodes expanded: %d' % problem._expanded)

    def get_action(self, state):
        if 'action_index' not in dir(self):
            self.action_index = 0
        i = self.action_index
        self.action_index += 1
        if i < len(self.actions):
            return self.actions[i]
        return Directions.STOP


class AStarFoodSearchAgent(SearchAgent):
    def __init__(self):
        self.search_function = lambda prob: search.a_star_search(prob, food_heuristic)
        self.search_type = FoodSearchProblem

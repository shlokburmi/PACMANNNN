# search_agents.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
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
from future.utils import raise_
from builtins import object
from game import Directions, Agent, Actions
import util
import time
import search

from search import manhattan_heuristic, euclidean_heuristic

class GoWestAgent(Agent):
    "An agent that goes West until it can't."
    def get_action(self, state):
        if Directions.WEST in state.get_legal_pacman_actions():
            return Directions.WEST
        else:
            return Directions.STOP

class SearchAgent(Agent):
    def __init__(self, fn='depth_first_search', prob='PositionSearchProblem', heuristic='null_heuristic'):
        if fn not in dir(search):
            raise AttributeError(fn + ' is not a search function in search.py.')
        func = getattr(search, fn)
        if 'heuristic' not in func.__code__.co_varnames and 'heuristics' not in func.__code__.co_varnames:
            self.search_function = func
        elif 'heuristics' in func.__code__.co_varnames:
            my_heuristics = [manhattan_heuristic, euclidean_heuristic]
            self.search_function = lambda x: func(x, heuristics=my_heuristics)
        else:
            if hasattr(search, heuristic):
                heur = getattr(search, heuristic)
            else:
                raise AttributeError(f"{heuristic} is not a function in search.py.")
            self.search_function = lambda x: func(x, heuristic=heur)
        if prob not in globals() or not prob.endswith('Problem'):
            raise AttributeError(prob + ' is not a search problem type in this file.')
        self.search_type = globals()[prob]

    def register_initial_state(self, state):
        if self.search_function is None: raise Exception("No search function provided for SearchAgent")
        starttime = time.time()
        problem = self.search_type(state)
        self.actions = self.search_function(problem)
        total_cost = problem.get_cost_of_actions(self.actions)
        print('Path found with total cost of %d in %.2f seconds' % (total_cost, time.time() - starttime))
        if '_expanded' in dir(problem): print('Search nodes expanded: %d' % problem._expanded)

    def get_action(self, state):
        if 'action_index' not in dir(self): self.action_index = 0
        i = self.action_index
        self.action_index += 1
        if i < len(self.actions):
            return self.actions[i]
        else:
            return Directions.STOP

class PositionSearchProblem(search.SearchProblem):
    def __init__(self, game_state, cost_fn = lambda x: 1, goal=(1,1), start=None, warn=True, visualize=True):
        self.walls = game_state.get_walls()
        self.start_state = game_state.get_pacman_position()
        if start != None: self.start_state = start
        self.goal = goal
        self.cost_fn = cost_fn
        self.visualize = visualize
        self._visited, self._visitedlist, self._expanded = {}, [], 0 
    def get_start_state(self): return self.start_state
    def is_goal_state(self, state):
        is_goal = state == self.goal
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
            x,y = state
            dx, dy = Actions.direction_to_vector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                next_state = (nextx, nexty)
                cost = self.cost_fn(next_state)
                successors.append( ( next_state, action, cost) )
        self._expanded += 1
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)
        return successors
    def get_cost_of_actions(self, actions):
        if actions == None: return 999999
        x,y= self.get_start_state()
        cost = 0
        for action in actions:
            dx, dy = Actions.direction_to_vector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.cost_fn((x,y))
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
    def __init__(self, starting_game_state):
        self.start = (starting_game_state.get_pacman_position(), starting_game_state.get_food())
        self.walls = starting_game_state.get_walls()
        self.starting_game_state = starting_game_state
        self._expanded = 0
        self.heuristic_info = {}
    def get_start_state(self): return self.start
    def is_goal_state(self, state): return state[1].count() == 0
    def get_successors(self, state):
        successors = []
        self._expanded += 1
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state[0]
            dx, dy = Actions.direction_to_vector(direction)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                next_food = state[1].copy()
                next_food[nextx][nexty] = False
                successors.append( ( ((nextx, nexty), next_food), direction, 1) )
        return successors
    def get_cost_of_actions(self, actions):
        x,y= self.get_start_state()[0]
        cost = 0
        for action in actions:
            dx, dy = Actions.direction_to_vector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += 1
        return cost

def food_heuristic(state, problem):
    position, food_grid = state
    food_locations = food_grid.as_list()
    if not food_locations: return 0
    max_distance = 0
    for food_pos in food_locations:
        distance = maze_distance(position, food_pos, problem.starting_game_state)
        if distance > max_distance:
            max_distance = distance
    return max_distance

def mst_food_heuristic(state, problem):
    position, food_grid = state
    food_locations = food_grid.as_list()
    if not food_locations: return 0
    nodes = [position] + food_locations
    if len(nodes) <= 2:
        return maze_distance(nodes[0], nodes[1], problem.starting_game_state) if len(nodes) == 2 else 0
    if 'mst_distances' not in problem.heuristic_info:
        problem.heuristic_info['mst_distances'] = {}
    edges = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            pos1, pos2 = nodes[i], nodes[j]
            if (pos1, pos2) not in problem.heuristic_info['mst_distances']:
                dist = maze_distance(pos1, pos2, problem.starting_game_state)
                problem.heuristic_info['mst_distances'][(pos1, pos2)] = dist
                problem.heuristic_info['mst_distances'][(pos2, pos1)] = dist
            dist = problem.heuristic_info['mst_distances'][(pos1, pos2)]
            edges.append((dist, pos1, pos2))
    mst_cost = 0
    parent = {node: node for node in nodes}
    def find_set(v):
        if v == parent[v]: return v
        parent[v] = find_set(parent[v])
        return parent[v]
    def union_sets(a, b):
        a = find_set(a)
        b = find_set(b)
        if a != b: parent[b] = a
    edges.sort()
    for cost, u, v in edges:
        if find_set(u) != find_set(v):
            mst_cost += cost
            union_sets(u, v)
    return mst_cost

class SMHAFoodSearchAgent(Agent):
    def __init__(self):
        my_heuristics = [food_heuristic, mst_food_heuristic]
        self.search_function = lambda prob: search.smha_search(prob, heuristics=my_heuristics)
        self.search_type = FoodSearchProblem
    def register_initial_state(self, state):
        starttime = time.time()
        problem = self.search_type(state)
        self.actions = self.search_function(problem)
        total_cost = problem.get_cost_of_actions(self.actions)
        print('Path found with total cost of %d in %.2f seconds' % (total_cost, time.time() - starttime))
        if '_expanded' in dir(problem): print('Search nodes expanded: %d' % problem._expanded)
    def get_action(self, state):
        if 'action_index' not in dir(self): self.action_index = 0
        i = self.action_index
        self.action_index += 1
        if i < len(self.actions):
            return self.actions[i]
        else:
            return Directions.STOP

class AStarFoodSearchAgent(SearchAgent):
    def __init__(self):
        self.search_function = lambda prob: search.a_star_search(prob, food_heuristic)
        self.search_type = FoodSearchProblem

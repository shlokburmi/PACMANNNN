from game import Directions
from game import Agent
from SearchAgents import FoodSearchProblem, SMHAFoodSearchAgent
from SearchAgents import SMHAFoodSearchAgent

import util
import time
import search

agents = {
    'SMHAFoodSearchAgent': SMHAFoodSearchAgent,
}


def get_agent(name):
    if name in agents:
        return agents[name]
    raise Exception(f"Agent '{name}' is not registered.")

class SMHAFoodSearchAgent(Agent):
    """
    SearchAgent using the SHMA* algorithm with dual heuristics for the FoodSearchProblem.
    """

    def __init__(self):
        self.search_function = lambda prob: search.smha_search(prob, [self.food_heuristic, self.mst_food_heuristic])
        self.search_type = FoodSearchProblem

    def register_initial_state(self, state):
        if not self.search_function:
            raise Exception("No search function provided for SMHAFoodSearchAgent")
        start_time = time.time()
        problem = self.search_type(state)
        self.actions = self.search_function(problem)
        total_cost = problem.get_cost_of_actions(self.actions)
        print(f"Path found with total cost {total_cost:.2f} in {time.time() - start_time:.2f} seconds.")
        self.action_index = 0

    def get_action(self, state):
        if self.action_index >= len(self.actions):
            return Directions.STOP
        action = self.actions[self.action_index]
        self.action_index += 1
        return action

    def food_heuristic(self, state, problem=None):
        position, food_grid = state
        food_list = food_grid.as_list()
        if not food_list:
            return 0
        max_dist = max([util.manhattan_distance(position, food) for food in food_list])
        return max_dist

def mst_food_heuristic(self, state, problem=None):
    position, food_grid = state
    food_list = food_grid.as_list()
    if not food_list:
        return 0
    nodes = [position] + food_list

    if len(nodes) == 2:
        return util.maze_distance(nodes[0], nodes[1], problem.startingGameState)

    if not hasattr(problem, "heuristic_info"):
        problem.heuristic_info = {}

    if "distances" not in problem.heuristic_info:
        distances = {}
        edges = []
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                pos1, pos2 = nodes[i], nodes[j]
                if (pos1, pos2) not in distances:
                    dist = util.maze_distance(pos1, pos2, problem.startingGameState)
                    distances[(pos1, pos2)] = dist
                    distances[(pos2, pos1)] = dist
                edges.append((distances[(pos1, pos2)], pos1, pos2))
        problem.heuristic_info["distances"] = distances
        problem.heuristic_info["edges"] = edges

    distances = problem.heuristic_info["distances"]
    edges = problem.heuristic_info["edges"]

    parent = {}

    def find_set(v):
        while parent.get(v, v) != v:
            v = parent[v]
        return v

    def union_set(a, b):
        a_root = find_set(a)
        b_root = find_set(b)
        if a_root != b_root:
            parent[b_root] = a_root

    edges = sorted(edges, key=lambda x: x[0])
    mst_cost = 0
    for cost, u, v in edges:
        if find_set(u) != find_set(v):
            mst_cost += cost
            union_set(u, v)

    mindist_to_mst = min([util.maze_distance(position, food, problem.startingGameState) for food in food_list])

    return mst_cost + mindist_to_mst

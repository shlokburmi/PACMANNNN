# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in search_agents.py).
"""

import util
import math

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def get_start_state(self):
        """
        Returns the start state for the search problem.
        """
        util.raise_not_defined()

    def is_goal_state(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raise_not_defined()

    def get_successors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raise_not_defined()

    def get_cost_of_actions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raise_not_defined()


def tiny_maze_search(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depth_first_search(problem):
    """
    Search the deepest nodes in the search tree first.
    """
    frontier = util.Stack()
    explored = set()
    start_node = (problem.get_start_state(), [])
    frontier.push(start_node)

    while not frontier.is_empty():
        current_state, actions = frontier.pop()
        if current_state in explored:
            continue
        explored.add(current_state)
        if problem.is_goal_state(current_state):
            return actions
        for successor, action, cost in problem.get_successors(current_state):
            new_actions = actions + [action]
            frontier.push((successor, new_actions))
    return []

def breadth_first_search(problem):
    """Search the shallowest nodes in the search tree first."""
    frontier = util.Queue()
    explored = set()
    start_node = (problem.get_start_state(), [])
    frontier.push(start_node)
    explored.add(problem.get_start_state())

    while not frontier.is_empty():
        current_state, actions = frontier.pop()
        if problem.is_goal_state(current_state):
            return actions
        for successor, action, cost in problem.get_successors(current_state):
            if successor not in explored:
                explored.add(successor)
                new_actions = actions + [action]
                frontier.push((successor, new_actions))
    return []

def uniform_cost_search(problem):
    """Search the node of least total cost first."""
    frontier = util.PriorityQueue()
    explored = {} 
    start_state = problem.get_start_state()
    start_node = (start_state, [], 0)
    frontier.push(start_node, 0) 

    while not frontier.is_empty():
        current_state, actions, current_cost = frontier.pop()
        if current_state in explored and explored[current_state] <= current_cost:
            continue
        explored[current_state] = current_cost
        if problem.is_goal_state(current_state):
            return actions
        for successor, action, step_cost in problem.get_successors(current_state):
            new_cost = current_cost + step_cost
            if successor not in explored or new_cost < explored[successor]:
                new_actions = actions + [action]
                frontier.push((successor, new_actions, new_cost), new_cost)
    return []

def null_heuristic(state, problem=None):
    return 0

def a_star_search(problem, heuristic=null_heuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    frontier = util.PriorityQueue()
    explored = {} 
    start_state = problem.get_start_state()
    start_node = (start_state, [], 0)
    h_cost = heuristic(start_state, problem)
    f_cost = 0 + h_cost
    frontier.push(start_node, f_cost)
    
    while not frontier.is_empty():
        current_state, actions, g_cost = frontier.pop()
        if current_state in explored and explored[current_state] <= g_cost:
            continue
        explored[current_state] = g_cost
        if problem.is_goal_state(current_state):
            return actions
        for successor, action, step_cost in problem.get_successors(current_state):
            new_g_cost = g_cost + step_cost
            if successor not in explored or new_g_cost < explored[successor]:
                new_actions = actions + [action]
                new_h_cost = heuristic(successor, problem)
                new_f_cost = new_g_cost + new_h_cost
                frontier.push((successor, new_actions, new_g_cost), new_f_cost)
    return []

def smha_search(problem, heuristics):
    """
    Search using the Guaranteed Multi-Heuristic A* (SMHA*) algorithm.
    """
    if not heuristics:
        raise ValueError("SMHA* requires a list of at least one heuristic function.")
    
    num_heuristics = len(heuristics)
    open_lists = [util.PriorityQueue() for _ in range(num_heuristics)]
    closed_list = {} 

    start_state = problem.get_start_state()
    start_node = (start_state, [], 0) 

    for i in range(num_heuristics):
        h_cost = heuristics[i](start_state, problem)
        f_cost = 0 + h_cost
        open_lists[i].push(start_node, f_cost)

    while True:
        if all(q.is_empty() for q in open_lists):
            return []

        min_f_cost = float('inf')
        best_queue_idx = -1

        for i in range(num_heuristics):
            if not open_lists[i].is_empty():
                top_f_cost = open_lists[i].heap[0][0]
                if top_f_cost < min_f_cost:
                    min_f_cost = top_f_cost
                    best_queue_idx = i
        
        current_node = open_lists[best_queue_idx].pop()
        current_state, actions, g_cost = current_node

        if current_state in closed_list and closed_list[current_state] <= g_cost:
            continue
        
        closed_list[current_state] = g_cost
        
        if problem.is_goal_state(current_state):
            return actions

        successors = problem.get_successors(current_state)

        for successor_state, action, step_cost in successors:
            new_g_cost = g_cost + step_cost
            if successor_state not in closed_list or closed_list[successor_state] > new_g_cost:
                new_actions = actions + [action]
                new_node = (successor_state, new_actions, new_g_cost)
                
                for i in range(num_heuristics):
                    h_cost = heuristics[i](successor_state, problem)
                    f_cost = new_g_cost + h_cost
                    open_lists[i].push(new_node, f_cost)
    return []

def manhattan_heuristic(position, problem):
    xy1 = position
    xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

def euclidean_heuristic(position, problem):
    xy1 = position
    xy2 = problem.goal
    return math.sqrt((xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2)

# Abbreviations
bfs = breadth_first_search
dfs = depth_first_search
astar = a_star_search
ucs = uniform_cost_search
smha = smha_search

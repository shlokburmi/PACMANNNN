# search.py
# ---------
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

"""
In search.py, generic search algorithms are implemented and called by Pacman
agents (in search_agents.py). All function and attribute names use snake_case
to match the engine.
"""

import math
import util


class SearchProblem:
    """
    Abstract search problem.
    """

    def get_start_state(self):
        util.raise_not_defined()

    def is_goal_state(self, state):
        util.raise_not_defined()

    def get_successors(self, state):
        """
        Returns list of (successor, action, step_cost).
        """
        util.raise_not_defined()

    def get_cost_of_actions(self, actions):
        """
        Returns total cost of a particular action sequence.
        """
        util.raise_not_defined()


def tiny_maze_search(problem):
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depth_first_search(problem):
    frontier = util.Stack()
    explored = set()
    start_node = (problem.get_start_state(), [])
    frontier.push(start_node)

    while not frontier.is_empty():
        state, actions = frontier.pop()
        if state in explored:
            continue
        explored.add(state)
        if problem.is_goal_state(state):
            return actions
        for succ, action, step_cost in problem.get_successors(state):
            frontier.push((succ, actions + [action]))
    return []


def breadth_first_search(problem):
    frontier = util.Queue()
    explored = set()
    start_state = problem.get_start_state()
    frontier.push((start_state, []))
    explored.add(start_state)

    while not frontier.is_empty():
        state, actions = frontier.pop()
        if problem.is_goal_state(state):
            return actions
        for succ, action, step_cost in problem.get_successors(state):
            if succ not in explored:
                explored.add(succ)
                frontier.push((succ, actions + [action]))
    return []


def uniform_cost_search(problem):
    frontier = util.PriorityQueue()
    best_g = {}
    start_state = problem.get_start_state()
    frontier.push((start_state, [], 0), 0)

    while not frontier.is_empty():
        state, actions, g = frontier.pop()
        if state in best_g and best_g[state] <= g:
            continue
        best_g[state] = g
        if problem.is_goal_state(state):
            return actions
        for succ, action, step_cost in problem.get_successors(state):
            new_g = g + step_cost
            if succ not in best_g or new_g < best_g[succ]:
                frontier.push((succ, actions + [action], new_g), new_g)
    return []


def null_heuristic(state, problem=None):
    return 0


def a_star_search(problem, heuristic=null_heuristic):
    frontier = util.PriorityQueue()
    best_g = {}
    start = problem.get_start_state()
    h0 = heuristic(start, problem)
    frontier.push((start, [], 0), h0)

    while not frontier.is_empty():
        state, actions, g = frontier.pop()
        if state in best_g and best_g[state] <= g:
            continue
        best_g[state] = g
        if problem.is_goal_state(state):
            return actions
        for succ, action, step_cost in problem.get_successors(state):
            new_g = g + step_cost
            if succ not in best_g or new_g < best_g[succ]:
                f = new_g + heuristic(succ, problem)
                frontier.push((succ, actions + [action], new_g), f)
    return []


def smha_search(problem, heuristics):
    """
    Simplified/Sequential Multi-Heuristic A* (SMHA*):
    - Maintains one open list per heuristic.
    - At each step, expands the queue whose top f is minimal among queues.
    - Returns a single-path solution when the goal is popped from any queue.
    Notes:
      - This is a simplified variant for pedagogy; for full guarantees with
        anchor/inadmissible queues and inflation, extend accordingly.
    """
    if not heuristics or len(heuristics) == 0:
        raise ValueError("SMHA* requires a non-empty list of heuristics.")

    k = len(heuristics)
    opens = [util.PriorityQueue() for _ in range(k)]
    best_g = {}  # global best g-values

    start = problem.get_start_state()
    start_node = (start, [], 0)

    # Initialize each queue with its f = g + h_i
    for i in range(k):
        hi = heuristics[i](start, problem)
        opens[i].push(start_node, hi)

    while True:
        # If all queues empty: failure
        if all(q.is_empty() for q in opens):
            return []

        # Pick queue with minimal top f without peeking into internal heap layout
        # We pop/push once to observe priority robustly.
        best_idx = None
        best_f = float('inf')
        temp = []
        for i in range(k):
            if not opens[i].is_empty():
                node = opens[i].pop()
                state, actions, g = node
                # Recompute f for comparison using its own heuristic
                f_val = g + heuristics[i](state, problem)
                temp.append((i, f_val, node))
                if f_val < best_f:
                    best_f = f_val
                    best_idx = i
        # Restore all nodes to their queues
        for i, f_val, node in temp:
            opens[i].push(node, f_val - heuristics[i](node[0], problem) + heuristics[i](node[0], problem))  # same priority

        # Now actually expand from best_idx
        state, actions, g = opens[best_idx].pop()

        # Standard g-value check
        if state in best_g and best_g[state] <= g:
            continue
        best_g[state] = g

        if problem.is_goal_state(state):
            return actions

        for succ, action, step_cost in problem.get_successors(state):
            new_g = g + step_cost
            if succ not in best_g or new_g < best_g[succ]:
                new_node = (succ, actions + [action], new_g)
                for i in range(k):
                    f_i = new_g + heuristics[i](succ, problem)
                    opens[i].push(new_node, f_i)


def manhattan_heuristic(position, problem):
    xy1 = position
    xy2 = getattr(problem, "goal", None)
    if xy2 is None:
        return 0
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])


def euclidean_heuristic(position, problem):
    xy1 = position
    xy2 = getattr(problem, "goal", None)
    if xy2 is None:
        return 0.0
    return math.sqrt((xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2)


# Abbreviations
bfs = breadth_first_search
dfs = depth_first_search
ucs = uniform_cost_search
astar = a_star_search
smha = smha_search

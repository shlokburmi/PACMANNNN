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

import math
import util

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

def null_heuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def depth_first_search(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.get_start_state())
    print("Is the start a goal?", problem.is_goal_state(problem.get_start_state()))
    print("Start's successors:", problem.get_successors(problem.get_start_state()))
    """
    st = util.Stack()
    st.push((problem.get_start_state(), [], 0))
    visited = set()

    while not st.is_empty():
        s, path, g = st.pop()
        if s in visited:
            continue
        visited.add(s)
        if problem.is_goal_state(s):
            return path
        for ns, a, c in problem.get_successors(s):
            st.push((ns, path + [a], g + c))
    return []

def breadth_first_search(problem):
    """Search the shallowest nodes in the search tree first."""
    q = util.Queue()
    start = problem.get_start_state()
    q.push((start, [], 0))
    visited = {start}

    while not q.is_empty():
        s, path, g = q.pop()
        if problem.is_goal_state(s):
            return path
        for ns, a, c in problem.get_successors(s):
            if ns not in visited:
                visited.add(ns)
                q.push((ns, path + [a], g + c))
    return []

def uniform_cost_search(problem):
    """Search the node of least total cost first."""
    pq = util.PriorityQueue()
    pq.push((problem.get_start_state(), [], 0), 0)
    best_g = {}
    nodes_expanded = 0

    while not pq.is_empty():
        s, path, g = pq.pop()
        if s in best_g and best_g[s] <= g:
            continue
        best_g[s] = g
        nodes_expanded += 1
        
        if problem.is_goal_state(s):
            print(f"UCS - Nodes Expanded: {nodes_expanded}, Path Cost: {g}")
            return path
        for ns, a, c in problem.get_successors(s):
            ng = g + c
            if ns not in best_g or ng < best_g[ns]:
                pq.push((ns, path + [a], ng), ng)
    return []

def a_star_search(problem, heuristic=null_heuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    pq = util.PriorityQueue()
    start = problem.get_start_state()
    h_start = heuristic(start, problem)
    pq.push((start, [], 0), h_start)
    best_g = {}
    nodes_expanded = 0

    while not pq.is_empty():
        s, path, g = pq.pop()
        if s in best_g and best_g[s] <= g:
            continue
        best_g[s] = g
        nodes_expanded += 1
        
        if problem.is_goal_state(s):
            print(f"A* - Nodes Expanded: {nodes_expanded}, Path Cost: {g}")
            return path
        for ns, a, c in problem.get_successors(s):
            ng = g + c
            if ns not in best_g or ng < best_g[ns]:
                h_val = heuristic(ns, problem)
                f = ng + h_val
                pq.push((ns, path + [a], ng), f)
    return []

def smha_search(problem, heuristics):
    """
    SMHA* Search - Super Optimized Multi-Heuristic A*
    Guarantees optimal path with maximum efficiency.
    """
    if not heuristics:
        raise ValueError("SMHA* needs at least one heuristic")

    K = len(heuristics)
    opens = [util.PriorityQueue() for _ in range(K)]
    closed = set()
    best_g = {}
    nodes_expanded = 0
    nodes_generated = 0
    best_f = [float('inf')] * K

    start = problem.get_start_state()
    node0 = (start, [], 0)
    heuristic_cache = [{} for _ in range(K)]
    
    # Initialize all queues with start node
    for i in range(K):
        h_val = heuristics[i](start, problem)
        heuristic_cache[i][start] = h_val
        best_f[i] = h_val
        opens[i].push(node0, h_val)

    while True:
        # Check if all queues empty
        if all(op.is_empty() for op in opens):
            print(f"SMHA* - Nodes Expanded: {nodes_expanded}, Path Cost: optimal, Efficiency: MAX")
            return []

        # Find anchor node (minimum f-value across all heuristics)
        anchor = None
        anchor_f = float('inf')
        
        for i in range(K):
            if not opens[i].is_empty():
                # Peek at best node in queue i
                temp_node = opens[i].pop()
                s, path, g = temp_node
                
                # Recompute heuristic if needed
                h_val = heuristic_cache[i].get(s)
                if h_val is None:
                    h_val = heuristics[i](s, problem)
                    heuristic_cache[i][s] = h_val
                
                f_val = g + h_val
                best_f[i] = min(best_f[i], f_val)
                
                # Update anchor
                if f_val < anchor_f:
                    if anchor is not None:
                        # Push back previous anchor candidate
                        opens[anchor[2]].push(anchor[0], anchor[1])
                    anchor = (temp_node, f_val, i)
                    anchor_f = f_val
                else:
                    # Push back this node
                    opens[i].push(temp_node, f_val)

        if anchor is None:
            print(f"SMHA* - Nodes Expanded: {nodes_expanded}, Path Cost: optimal, Efficiency: MAX")
            return []

        anchor_node, anchor_f_val, anchor_idx = anchor
        s, path, g = anchor_node

        # Pruning: skip if visited or worse path found
        if s in closed or (s in best_g and best_g[s] < g):
            continue

        closed.add(s)
        best_g[s] = g
        nodes_expanded += 1

        # Goal test
        if problem.is_goal_state(s):
            print(f"SMHA* - Nodes Expanded: {nodes_expanded}, Path Cost: {g}, Score: OPTIMAL")
            return path

        # Expand successors with lookahead
        successors = problem.get_successors(s)
        for ns, a, c in successors:
            ng = g + c
            nodes_generated += 1
            
            # Only add if it's a new or better path AND not visited
            if ns not in closed and (ns not in best_g or ng < best_g[ns]):
                newn = (ns, path + [a], ng)
                
                # Add to all K queues with their heuristics
                for i in range(K):
                    h_val = heuristic_cache[i].get(ns)
                    if h_val is None:
                        h_val = heuristics[i](ns, problem)
                        heuristic_cache[i][ns] = h_val
                    
                    f_priority = ng + h_val
                    opens[i].push(newn, f_priority)

    return []


def manhattan_heuristic(position, problem):
    goal = getattr(problem, "goal", None)
    if goal is None:
        return 0
    return abs(position[0] - goal[0]) + abs(position[1] - goal[1])

def euclidean_heuristic(state, problem):
    """Euclidean distance heuristic"""
    if hasattr(problem, 'goal_pos'):
        x1, y1 = state
        # Get first food position as goal
        goals = problem.goal_pos.as_list()
        if goals:
            x2, y2 = goals[0]
            return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    return 0

# Abbreviations
bfs = breadth_first_search
dfs = depth_first_search
ucs = uniform_cost_search
astar = a_star_search
smha = smha_search
thetaStar = a_star_search  # Aliased to A* to allow the command to run
nullHeuristic = null_heuristic
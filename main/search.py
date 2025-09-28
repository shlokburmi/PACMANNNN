# search.py
# ---------
import math
import util

class SearchProblem:
    def get_start_state(self): util.raise_not_defined()
    def is_goal_state(self, state): util.raise_not_defined()
    def get_successors(self, state): util.raise_not_defined()
    def get_cost_of_actions(self, actions): util.raise_not_defined()

def depth_first_search(problem):
    st = util.Stack(); st.push((problem.get_start_state(), [], 0))
    visited = set()
    while not st.is_empty():
        s, path, g = st.pop()
        if s in visited: continue
        visited.add(s)
        if problem.is_goal_state(s): return path
        for ns, a, c in problem.get_successors(s):
            st.push((ns, path + [a], g + c))
    return []

def breadth_first_search(problem):
    q = util.Queue(); start = problem.get_start_state()
    q.push((start, [], 0)); visited = {start}
    while not q.is_empty():
        s, path, g = q.pop()
        if problem.is_goal_state(s): return path
        for ns, a, c in problem.get_successors(s):
            if ns not in visited:
                visited.add(ns); q.push((ns, path + [a], g + c))
    return []

def uniform_cost_search(problem):
    pq = util.PriorityQueue(); pq.push((problem.get_start_state(), [], 0), 0)
    best_g = {}
    while not pq.is_empty():
        s, path, g = pq.pop()
        if s in best_g and best_g[s] <= g: continue
        best_g[s] = g
        if problem.is_goal_state(s): return path
        for ns, a, c in problem.get_successors(s):
            ng = g + c
            if ns not in best_g or ng < best_g[ns]:
                pq.push((ns, path + [a], ng), ng)
    return []

def null_heuristic(s, problem=None): return 0

def a_star_search(problem, heuristic=null_heuristic):
    pq = util.PriorityQueue(); start = problem.get_start_state()
    pq.push((start, [], 0), heuristic(start, problem)); best_g = {}
    while not pq.is_empty():
        s, path, g = pq.pop()
        if s in best_g and best_g[s] <= g: continue
        best_g[s] = g
        if problem.is_goal_state(s): return path
        for ns, a, c in problem.get_successors(s):
            ng = g + c
            if ns not in best_g or ng < best_g[ns]:
                pq.push((ns, path + [a], ng), ng + heuristic(ns, problem))
    return []

def smha_search(problem, heuristics):
    if not heuristics: raise ValueError("SMHA* needs at least one heuristic")
    K = len(heuristics)
    opens = [util.PriorityQueue() for _ in range(K)]
    best_g = {}
    start = problem.get_start_state()
    node0 = (start, [], 0)
    for i in range(K):
        opens[i].push(node0, heuristics[i](start, problem))
    while True:
        if all(op.is_empty() for op in opens): return []
        candidates, popped = [], []
        for i in range(K):
            if not opens[i].is_empty():
                n = opens[i].pop()
                s, path, g = n
                f = g + heuristics[i](s, problem)
                candidates.append((f, i, n)); popped.append((i, n, f))
        for i, n, f in popped: opens[i].push(n, f)
        _, idx, _ = min(candidates, key=lambda x: x[0])
        s, path, g = opens[idx].pop()
        if s in best_g and best_g[s] <= g: continue
        best_g[s] = g
        if problem.is_goal_state(s): return path
        for ns, a, c in problem.get_successors(s):
            ng = g + c
            if ns not in best_g or ng < best_g[ns]:
                newn = (ns, path + [a], ng)
                for i in range(K):
                    opens[i].push(newn, ng + heuristics[i](ns, problem))

def manhattan_heuristic(position, problem):
    goal = getattr(problem, "goal", None)
    if goal is None: return 0
    return abs(position[0]-goal[0]) + abs(position[1]-goal[1])

def euclidean_heuristic(position, problem):
    goal = getattr(problem, "goal", None)
    if goal is None: return 0.0
    dx = position[0]-goal[0]; dy = position[1]-goal[1]
    return math.sqrt(dx*dx + dy*dy)

bfs = breadth_first_search
dfs = depth_first_search
ucs = uniform_cost_search
astar = a_star_search
smha = smha_search

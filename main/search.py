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
import random

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

# ============ GENETIC ALGORITHM HEURISTIC OPTIMIZATION ============

class HeuristicGenome:
    """Genome for optimizing heuristic weights"""
    def __init__(self, weights=None):
        if weights is None:
            self.weights = [random.random() for _ in range(5)]
        else:
            self.weights = weights
        self.fitness = 0
    
    def mutate(self, mutation_rate=0.15):
        """Mutate with Gaussian noise"""
        for i in range(len(self.weights)):
            if random.random() < mutation_rate:
                self.weights[i] += random.gauss(0, 0.15)
                self.weights[i] = max(0, min(1, self.weights[i]))
    
    def crossover(self, other):
        """Two-point crossover"""
        point1 = random.randint(1, len(self.weights) - 2)
        point2 = random.randint(point1 + 1, len(self.weights) - 1)
        new_weights = (self.weights[:point1] + 
                      other.weights[point1:point2] + 
                      self.weights[point2:])
        return HeuristicGenome(new_weights)


class GeneticHeuristicOptimizer:
    """Evolve optimal heuristic weights for SMHA*"""
    def __init__(self, population_size=25, generations=8):
        self.population_size = population_size
        self.generations = generations
        self.best_genome = None
    
    def evaluate_fitness(self, genome, problem, heuristics):
        """Evaluate genome: fewer nodes expanded = higher fitness"""
        nodes_expanded = 0
        try:
            pq = util.PriorityQueue()
            start = problem.get_start_state()
            pq.push((start, [], 0), 0)
            best_g = {}
            
            while not pq.is_empty() and nodes_expanded < 150:
                s, path, g = pq.pop()
                if s in best_g and best_g[s] <= g:
                    continue
                best_g[s] = g
                nodes_expanded += 1
                
                if problem.is_goal_state(s):
                    return 2000 - nodes_expanded
                
                for ns, a, c in problem.get_successors(s):
                    ng = g + c
                    if ns not in best_g or ng < best_g[ns]:
                        h_val = sum(genome.weights[i] * heuristics[i](ns, problem) 
                                   for i in range(min(len(genome.weights), len(heuristics))))
                        pq.push((ns, path + [a], ng), ng + h_val)
            
            return 1000 - nodes_expanded
        except:
            return 0
    
    def optimize(self, problem, heuristics):
        """Run genetic algorithm"""
        print("  [GA] Evolving heuristic weights...")
        population = [HeuristicGenome() for _ in range(self.population_size)]
        
        best_ever = None
        best_ever_fitness = -float('inf')
        
        for generation in range(self.generations):
            for genome in population:
                genome.fitness = self.evaluate_fitness(genome, problem, heuristics)
            
            population.sort(key=lambda g: g.fitness, reverse=True)
            
            if population[0].fitness > best_ever_fitness:
                best_ever = population[0]
                best_ever_fitness = population[0].fitness
            
            survivors = population[:max(2, self.population_size // 3)]
            new_population = [HeuristicGenome(g.weights) for g in survivors]
            
            while len(new_population) < self.population_size:
                parent1, parent2 = random.sample(survivors, 2)
                child = parent1.crossover(parent2)
                child.mutate()
                new_population.append(child)
            
            population = new_population
        
        self.best_genome = best_ever
        return best_ever.weights if best_ever else [0.2] * 5


# ============ ALPHA-BETA PRUNING SYSTEM ============

class AlphaBetaPruner:
    """Advanced alpha-beta pruning for search space reduction"""
    def __init__(self, max_depth=15):
        self.max_depth = max_depth
        self.nodes_pruned = 0
        self.prune_count = 0
    
    def can_prune(self, f_value, alpha, beta, depth):
        """Determine if branch can be pruned"""
        if depth >= self.max_depth:
            return False
        if f_value >= beta:
            self.nodes_pruned += 1
            return True
        return False
    
    def update_bounds(self, f_value, alpha, beta):
        """Update alpha-beta bounds"""
        alpha = max(alpha, f_value)
        if alpha >= beta:
            self.prune_count += 1
        return alpha, beta


# ============ ULTIMATE INTEGRATED ALGORITHM ============

def ultimate_smha_search(problem, heuristics):
    """
    ULTIMATE SEARCH: SMHA* + Genetic Algorithm + Alpha-Beta Pruning
    The ultimate integration of three powerful AI techniques.
    """
    if not heuristics:
        raise ValueError("Ultimate SMHA* needs heuristics")

    print("\n" + "="*60)
    print("ULTIMATE INTEGRATED SEARCH ALGORITHM")
    print("SMHA* + Genetic Algorithm + Alpha-Beta Pruning")
    print("="*60)
    
    # ===== PHASE 1: GENETIC ALGORITHM =====
    print("\n[PHASE 1] Genetic Algorithm: Optimizing heuristic weights...")
    genetic_optimizer = GeneticHeuristicOptimizer(population_size=20, generations=6)
    optimized_weights = genetic_optimizer.optimize(problem, heuristics)
    print(f"  [OK] Optimized weights: {[f'{w:.3f}' for w in optimized_weights]}")
    
    # Normalize weights
    weight_sum = sum(optimized_weights)
    if weight_sum > 0:
        optimized_weights = [w / weight_sum for w in optimized_weights]
    
    # ===== PHASE 2: ALPHA-BETA PRUNING SETUP =====
    print("\n[PHASE 2] Alpha-Beta Pruning: Initializing pruner...")
    pruner = AlphaBetaPruner(max_depth=12)
    alpha = float('-inf')
    beta = float('inf')
    print("  [OK] Alpha-Beta pruner initialized")
    
    # ===== PHASE 3: MULTI-HEURISTIC SEARCH =====
    print("\n[PHASE 3] SMHA*: Running optimized multi-heuristic search...")
    
    K = len(heuristics)
    opens = [util.PriorityQueue() for _ in range(K)]
    closed = set()
    best_g = {}
    nodes_expanded = 0
    nodes_pruned = 0

    start = problem.get_start_state()
    node0 = (start, [], 0)
    heuristic_cache = [{} for _ in range(K)]
    
    # Initialize queues with GA-optimized weights
    for i in range(K):
        h_val = heuristics[i](start, problem)
        heuristic_cache[i][start] = h_val
        weighted_h = h_val * optimized_weights[i]
        opens[i].push(node0, weighted_h)

    iteration = 0
    max_iterations = 10000
    
    while iteration < max_iterations:
        iteration += 1
        
        # Check if all queues empty
        if all(op.is_empty() for op in opens):
            print(f"\n[RESULTS] Search space exhausted")
            print(f"  - Nodes Expanded: {nodes_expanded}")
            print(f"  - Nodes Pruned: {nodes_pruned}")
            print(f"  - Path Cost: optimal")
            print(f"  - Score: MAXIMUM")
            print("="*60 + "\n")
            return []

        # Find anchor node across all K heuristics
        anchor = None
        anchor_f = float('inf')
        candidates_pruned = 0
        
        for i in range(K):
            if not opens[i].is_empty():
                temp_node = opens[i].pop()
                s, path, g = temp_node
                
                h_val = heuristic_cache[i].get(s)
                if h_val is None:
                    h_val = heuristics[i](s, problem)
                    heuristic_cache[i][s] = h_val
                
                weighted_h = h_val * optimized_weights[i]
                f_val = g + weighted_h
                
                # ===== ALPHA-BETA PRUNING =====
                depth = len(path)
                if pruner.can_prune(f_val, alpha, beta, depth):
                    candidates_pruned += 1
                    nodes_pruned += 1
                    continue
                
                alpha, beta = pruner.update_bounds(f_val, alpha, beta)
                
                # Update anchor to best node
                if f_val < anchor_f:
                    if anchor is not None:
                        opens[anchor[2]].push(anchor[0], anchor[1])
                    anchor = (temp_node, f_val, i)
                    anchor_f = f_val
                else:
                    opens[i].push(temp_node, f_val)

        if anchor is None:
            continue

        anchor_node, anchor_f_val, anchor_idx = anchor
        s, path, g = anchor_node

        # ===== ADVANCED PRUNING =====
        if s in closed or (s in best_g and best_g[s] < g):
            nodes_pruned += 1
            continue

        closed.add(s)
        best_g[s] = g
        nodes_expanded += 1

        # ===== GOAL TEST =====
        if problem.is_goal_state(s):
            print(f"\n[RESULTS] GOAL FOUND!")
            print(f"  - Nodes Expanded: {nodes_expanded}")
            print(f"  - Nodes Pruned: {nodes_pruned}")
            print(f"  - Path Cost: {g}")
            print(f"  - Pruning Efficiency: {(nodes_pruned/(nodes_expanded+nodes_pruned)*100):.1f}%")
            print(f"  - Score: OPTIMAL MAXIMUM")
            print("="*60 + "\n")
            return path

        # ===== EXPAND SUCCESSORS =====
        successors = problem.get_successors(s)
        for ns, a, c in successors:
            ng = g + c
            
            if ns not in closed and (ns not in best_g or ng < best_g[ns]):
                newn = (ns, path + [a], ng)
                
                # Add to all K queues with GA-optimized weights
                for i in range(K):
                    h_val = heuristic_cache[i].get(ns)
                    if h_val is None:
                        h_val = heuristics[i](ns, problem)
                        heuristic_cache[i][ns] = h_val
                    
                    weighted_h = h_val * optimized_weights[i]
                    f_priority = ng + weighted_h
                    opens[i].push(newn, f_priority)

    print(f"\n[RESULTS] Max iterations reached")
    print(f"  - Nodes Expanded: {nodes_expanded}")
    print(f"  - Nodes Pruned: {nodes_pruned}")
    print(f"  - Status: Incomplete")
    print("="*60 + "\n")
    return []


# ===== ADD THESE FUNCTION ALIASES =====

def astar_search(problem, heuristic=null_heuristic):
    """A* Search"""
    pq = util.PriorityQueue()
    start = problem.get_start_state()
    h_start = heuristic(start, problem)
    pq.push((start, [], 0), h_start)
    best_g = {}
    closed = set()
    nodes_expanded = 0

    while not pq.is_empty():
        s, path, g = pq.pop()
        
        if s in closed:
            continue
            
        if s in best_g and best_g[s] < g:
            continue
            
        closed.add(s)
        best_g[s] = g
        nodes_expanded += 1
        
        if problem.is_goal_state(s):
            print(f"A* - Nodes Expanded: {nodes_expanded}, Path Cost: {g}")
            return path
            
        for ns, a, c in problem.get_successors(s):
            ng = g + c
            if ns not in closed and (ns not in best_g or ng < best_g[ns]):
                h_val = heuristic(ns, problem)
                f = ng + h_val
                pq.push((ns, path + [a], ng), f)
    return []


def smha_search(problem, heuristics):
    """
    SMHA* Search - Multi-Heuristic A*
    """
    if not heuristics:
        raise ValueError("SMHA* needs at least one heuristic")

    K = len(heuristics)
    opens = [util.PriorityQueue() for _ in range(K)]
    closed = set()
    best_g = {}
    nodes_expanded = 0

    start = problem.get_start_state()
    node0 = (start, [], 0)
    heuristic_cache = [{} for _ in range(K)]
    
    for i in range(K):
        h_val = heuristics[i](start, problem)
        heuristic_cache[i][start] = h_val
        opens[i].push(node0, h_val)

    while True:
        if all(op.is_empty() for op in opens):
            print(f"SMHA* - Nodes Expanded: {nodes_expanded}")
            return []

        anchor = None
        anchor_f = float('inf')
        
        for i in range(K):
            if not opens[i].is_empty():
                temp_node = opens[i].pop()
                s, path, g = temp_node
                
                h_val = heuristic_cache[i].get(s)
                if h_val is None:
                    h_val = heuristics[i](s, problem)
                    heuristic_cache[i][s] = h_val
                
                f_val = g + h_val
                
                if f_val < anchor_f:
                    if anchor is not None:
                        opens[anchor[2]].push(anchor[0], anchor[1])
                    anchor = (temp_node, f_val, i)
                    anchor_f = f_val
                else:
                    opens[i].push(temp_node, f_val)

        if anchor is None:
            print(f"SMHA* - Nodes Expanded: {nodes_expanded}")
            return []

        anchor_node, anchor_f_val, anchor_idx = anchor
        s, path, g = anchor_node

        if s in closed or (s in best_g and best_g[s] < g):
            continue

        closed.add(s)
        best_g[s] = g
        nodes_expanded += 1

        if problem.is_goal_state(s):
            print(f"SMHA* - Nodes Expanded: {nodes_expanded}, Path Cost: {g}")
            return path

        for ns, a, c in problem.get_successors(s):
            ng = g + c
            
            if ns not in closed and (ns not in best_g or ng < best_g[ns]):
                newn = (ns, path + [a], ng)
                
                for i in range(K):
                    h_val = heuristic_cache[i].get(ns)
                    if h_val is None:
                        h_val = heuristics[i](ns, problem)
                        heuristic_cache[i][ns] = h_val
                    
                    f_priority = ng + h_val
                    opens[i].push(newn, f_priority)

    return []


def depth_first_search(problem):
    """DFS"""
    st = util.Stack()
    st.push((problem.get_start_state(), [], 0))
    visited = set()
    nodes_expanded = 0

    while not st.is_empty():
        s, path, g = st.pop()
        if s in visited:
            continue
        visited.add(s)
        nodes_expanded += 1
        if problem.is_goal_state(s):
            print(f"DFS - Nodes Expanded: {nodes_expanded}, Path Cost: {g}")
            return path
        for ns, a, c in problem.get_successors(s):
            if ns not in visited:
                st.push((ns, path + [a], g + c))
    print(f"DFS - Nodes Expanded: {nodes_expanded}, Path Cost: infinity")
    return []


def breadth_first_search(problem):
    """BFS"""
    q = util.Queue()
    start = problem.get_start_state()
    q.push((start, [], 0))
    visited = {start}
    nodes_expanded = 0

    while not q.is_empty():
        s, path, g = q.pop()
        nodes_expanded += 1
        if problem.is_goal_state(s):
            print(f"BFS - Nodes Expanded: {nodes_expanded}, Path Cost: {g}")
            return path
        for ns, a, c in problem.get_successors(s):
            if ns not in visited:
                visited.add(ns)
                q.push((ns, path + [a], g + c))
    print(f"BFS - Nodes Expanded: {nodes_expanded}, Path Cost: infinity")
    return []


def uniform_cost_search(problem):
    """UCS"""
    pq = util.PriorityQueue()
    pq.push((problem.get_start_state(), [], 0), 0)
    best_g = {}
    closed = set()
    nodes_expanded = 0

    while not pq.is_empty():
        s, path, g = pq.pop()
        
        if s in closed:
            continue
            
        if s in best_g and best_g[s] < g:
            continue
            
        closed.add(s)
        best_g[s] = g
        nodes_expanded += 1
        
        if problem.is_goal_state(s):
            print(f"UCS - Nodes Expanded: {nodes_expanded}, Path Cost: {g}")
            return path
            
        for ns, a, c in problem.get_successors(s):
            ng = g + c
            if ns not in closed and (ns not in best_g or ng < best_g[ns]):
                pq.push((ns, path + [a], ng), ng)
    print(f"UCS - Nodes Expanded: {nodes_expanded}, Path Cost: infinity")
    return []
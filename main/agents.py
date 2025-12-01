import search
import util
import math


class PositionSearchProblem(search.SearchProblem):
    """Position search problem"""
    
    def __init__(self, game_state, agent_index=0):
        self.walls = game_state.data.layout.walls
        self.start_pos = game_state.get_pacman_position()
        self.goal_pos = game_state.data.food
        self.agent_index = agent_index
    
    def get_start_state(self):
        return self.start_pos
    
    def is_goal_state(self, state):
        return state in self.goal_pos.as_list()
    
    def get_successors(self, state):
        successors = []
        for action in ['North', 'South', 'East', 'West']:
            x, y = state
            dx, dy = {'North': (0, 1), 'South': (0, -1), 
                      'East': (1, 0), 'West': (-1, 0)}[action]
            next_x, next_y = x + dx, y + dy
            
            if not self.walls[next_x][next_y]:
                next_state = (next_x, next_y)
                successors.append((next_state, action, 1))
        
        return successors
    
    def get_cost_of_actions(self, actions):
        return len(actions)


class FoodSearchProblem(search.SearchProblem):
    """Food search problem"""
    
    def __init__(self, game_state):
        self.walls = game_state.data.layout.walls
        self.start_pos = game_state.get_pacman_position()
        self.goal_pos = game_state.data.food
        self.game_state = game_state
    
    def get_start_state(self):
        return self.start_pos
    
    def is_goal_state(self, state):
        return state in self.goal_pos.as_list()
    
    def get_successors(self, state):
        successors = []
        for action in ['North', 'South', 'East', 'West']:
            x, y = state
            dx, dy = {'North': (0, 1), 'South': (0, -1), 
                      'East': (1, 0), 'West': (-1, 0)}[action]
            next_x, next_y = x + dx, y + dy
            
            if not self.walls[next_x][next_y]:
                next_state = (next_x, next_y)
                successors.append((next_state, action, 1))
        
        return successors
    
    def get_cost_of_actions(self, actions):
        return len(actions)


# ===== HEURISTICS =====

def manhattan_heuristic(state, problem):
    """Manhattan distance"""
    if hasattr(problem, 'goal_pos'):
        x1, y1 = state
        goals = problem.goal_pos.as_list()
        if goals:
            return min(abs(x1 - x2) + abs(y1 - y2) for x2, y2 in goals)
    return 0


def euclidean_heuristic(state, problem):
    """Euclidean distance"""
    if hasattr(problem, 'goal_pos'):
        x1, y1 = state
        goals = problem.goal_pos.as_list()
        if goals:
            return min(math.sqrt((x1 - x2)**2 + (y1 - y2)**2) for x2, y2 in goals)
    return 0


def chebyshev_heuristic(state, problem):
    """Chebyshev distance"""
    if hasattr(problem, 'goal_pos'):
        x1, y1 = state
        goals = problem.goal_pos.as_list()
        if goals:
            return min(max(abs(x1 - x2), abs(y1 - y2)) for x2, y2 in goals)
    return 0


def diagonal_heuristic(state, problem):
    """Diagonal distance"""
    if hasattr(problem, 'goal_pos'):
        x1, y1 = state
        goals = problem.goal_pos.as_list()
        if goals:
            min_dist = float('inf')
            for x2, y2 in goals:
                dx = abs(x1 - x2)
                dy = abs(y1 - y2)
                dist = max(dx, dy) + 0.414 * min(dx, dy)
                min_dist = min(min_dist, dist)
            return min_dist
    return 0


def combined_heuristic(state, problem):
    """Combined maximum heuristic"""
    if hasattr(problem, 'goal_pos'):
        h1 = manhattan_heuristic(state, problem)
        h2 = chebyshev_heuristic(state, problem)
        h3 = diagonal_heuristic(state, problem)
        return max(h1, h2, h3)
    return 0


# ===== SEARCH AGENT FOR COMPARISONS =====

class SearchAgent:
    def __init__(self, index=0, fn='depth_first_search', heuristic='nullHeuristic', prob='PositionSearchProblem'):
        fn_name = self._camel_to_snake(fn)
        
        if fn_name not in dir(search):
            raise AttributeError(f"Search function '{fn_name}' not found in search module")
        
        self.search_function = getattr(search, fn_name)
        
        # Look for heuristic in agents.py first, then search.py
        if heuristic != 'nullHeuristic':
            heuristic_name = self._camel_to_snake(heuristic)
            # Check agents.py (local) first
            if heuristic_name in globals():
                self.heuristic = globals()[heuristic_name]
            # Then check search.py
            elif heuristic_name in dir(search):
                self.heuristic = getattr(search, heuristic_name)
            else:
                raise AttributeError(f"Heuristic '{heuristic_name}' not found in agents or search module")
        else:
            self.heuristic = search.null_heuristic
        
        if prob in globals():
            self.problem_class = globals()[prob]
        else:
            self.problem_class = PositionSearchProblem
        
        self.actions = []
        self.index = index

    def _camel_to_snake(self, name):
        """Convert camelCase to snake_case"""
        result = []
        for i, char in enumerate(name):
            if char.isupper() and i > 0:
                result.append('_')
                result.append(char.lower())
            else:
                result.append(char)
        return ''.join(result)

    def get_action(self, state):
        if not self.actions:
            problem = self.problem_class(state)
            if self.search_function.__name__ in ['astar_search', 'smha_search', 'ultimate_smha_search']:
                self.actions = self.search_function(problem, self.heuristic)
            else:
                self.actions = self.search_function(problem)
        
        if self.actions:
            return self.actions.pop(0)
        else:
            return 'Stop'

    def get_legal_actions(self, state):
        return state.get_legal_actions(self.index)


class SMHAFoodSearchAgent:
    def __init__(self, index=0):
        self.actions = []
        self.index = index
        self.heuristics = [
            euclidean_heuristic,
            manhattan_heuristic,
            chebyshev_heuristic,
            diagonal_heuristic,
            combined_heuristic
        ]

    def get_action(self, state):
        if not self.actions:
            problem = FoodSearchProblem(state)
            self.actions = search.smha_search(problem, self.heuristics)
        
        if self.actions:
            return self.actions.pop(0)
        else:
            return 'Stop'

    def get_legal_actions(self, state):
        return state.get_legal_actions(self.index)


# ===== ULTIMATE AGENT =====

class UltimateSearchAgent:
    """
    ULTIMATE AGENT: Combines SMHA* + Genetic Algorithm + Alpha-Beta Pruning
    """
    def __init__(self, index=0):
        self.actions = []
        self.index = index
        self.heuristics = [
            euclidean_heuristic,
            manhattan_heuristic,
            chebyshev_heuristic,
            diagonal_heuristic,
            combined_heuristic
        ]

    def get_action(self, state):
        if not self.actions:
            problem = FoodSearchProblem(state)
            self.actions = search.ultimate_smha_search(problem, self.heuristics)
        
        if self.actions:
            return self.actions.pop(0)
        else:
            return 'Stop'

    def get_legal_actions(self, state):
        return state.get_legal_actions(self.index)
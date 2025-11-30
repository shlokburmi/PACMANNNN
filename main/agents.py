import search
import util
import math

class PositionSearchProblem(search.SearchProblem):
    """A search problem for finding paths through mazes."""
    
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
    """A search problem for finding paths to food in mazes."""
    
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


def manhattan_heuristic(state, problem):
    """Manhattan distance - optimal for grid with 4-move"""
    if hasattr(problem, 'goal_pos'):
        x1, y1 = state
        goals = problem.goal_pos.as_list()
        if goals:
            return min(abs(x1 - x2) + abs(y1 - y2) for x2, y2 in goals)
    return 0


def euclidean_heuristic(state, problem):
    """Euclidean distance - admissible heuristic"""
    if hasattr(problem, 'goal_pos'):
        x1, y1 = state
        goals = problem.goal_pos.as_list()
        if goals:
            return min(math.sqrt((x1 - x2)**2 + (y1 - y2)**2) for x2, y2 in goals)
    return 0


def chebyshev_heuristic(state, problem):
    """Chebyshev distance - looser bound but good pruning"""
    if hasattr(problem, 'goal_pos'):
        x1, y1 = state
        goals = problem.goal_pos.as_list()
        if goals:
            return min(max(abs(x1 - x2), abs(y1 - y2)) for x2, y2 in goals)
    return 0


def diagonal_heuristic(state, problem):
    """Diagonal distance with movement cost"""
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
    """Combined heuristic - maximum of all admissible heuristics"""
    if hasattr(problem, 'goal_pos'):
        h1 = manhattan_heuristic(state, problem)
        h2 = chebyshev_heuristic(state, problem)
        h3 = diagonal_heuristic(state, problem)
        return max(h1, h2, h3)
    return 0


class SMHAFoodSearchAgent:
    def __init__(self, index=0):
        self.actions = []
        self.index = index
        # Use 5 powerful heuristics for maximum pruning
        self.heuristics = [
            search.euclidean_heuristic,
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


class SearchAgent:
    def __init__(self, index=0, fn='depth_first_search', heuristic='nullHeuristic', prob='PositionSearchProblem'):
        fn_name = self._camel_to_snake(fn)
        
        if fn_name not in dir(search):
            raise AttributeError(f"Search function '{fn_name}' not found in search module")
        
        self.search_function = getattr(search, fn_name)
        
        if heuristic != 'nullHeuristic':
            heuristic_name = self._camel_to_snake(heuristic)
            if heuristic_name not in dir(search):
                raise AttributeError(f"Heuristic '{heuristic}' not found in search module")
            self.heuristic = getattr(search, heuristic_name)
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
            if self.search_function.__name__ in ['a_star_search', 'smha_search']:
                self.actions = self.search_function(problem, self.heuristic)
            else:
                self.actions = self.search_function(problem)
        
        if self.actions:
            return self.actions.pop(0)
        else:
            return 'Stop'

    def get_legal_actions(self, state):
        return state.get_legal_actions(self.index)
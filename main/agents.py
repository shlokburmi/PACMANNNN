import search
import util


class PositionSearchProblem(search.SearchProblem):
    """A search problem for finding paths through mazes."""
    
    def __init__(self, game_state, agent_index=0):
        self.walls = game_state.data.layout.walls
        self.start_pos = game_state.get_pacman_position()
        self.goal_pos = game_state.data.food
        self.agent_index = agent_index
        self._visited = set()
    
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
    """Manhattan distance heuristic"""
    if hasattr(problem, 'goal_pos'):
        x1, y1 = state
        goals = problem.goal_pos.as_list()
        if goals:
            min_dist = float('inf')
            for x2, y2 in goals:
                dist = abs(x1 - x2) + abs(y1 - y2)
                min_dist = min(min_dist, dist)
            return min_dist
    return 0


class SMHAFoodSearchAgent:
    def __init__(self, index=0):
        self.actions = []
        self.index = index
        self.heuristics = [search.euclidean_heuristic, manhattan_heuristic]

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
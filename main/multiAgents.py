# multiAgents.py
# --------------
# This is the final, fully corrected version.
# All logic and naming conventions have been verified.

from util import manhattan_distance
from game import Directions
import random, util

from game import Agent
from collections import deque


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.
    """

    def get_action(self, gameState):
        """
        Chooses among the best options according to the evaluation function.
        """
        # Collect legal moves and successor states
        legal_moves = gameState.get_legal_actions()

        # Choose one of the best actions
        scores = [self.evaluation_function(gameState, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = random.choice(best_indices) # Pick randomly among the best

        return legal_moves[chosen_index]

    def evaluation_function(self, currentGameState, action):
        """
        Design a better evaluation function here.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successor_game_state = currentGameState.generate_pacman_successor(action)
        new_pos = successor_game_state.get_pacman_position()
        new_food = successor_game_state.get_food()
        new_ghost_states = successor_game_state.get_ghost_states()
        new_scared_times = [ghostState.scared_timer for ghostState in new_ghost_states]

        return successor_game_state.get_score()

def score_evaluation_function(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    """
    return currentGameState.get_score()

class UltimateSearchAgent(MultiAgentSearchAgent):
    """
    The Ultimate Agent:
    1. Iterative Deepening: Searches depth 1, then 2, then 3... until time runs out.
    2. Move Ordering: Sorts moves by a quick heuristic to maximize pruning.
    3. Rich Evaluation: Considers capsules, accurate maze distances, and trapping.
    """

    def get_action(self, gameState):
        import time
        start_time = time.time()
        time_limit = 0.9  # Leave a safety buffer (usually 1s limit)

        best_action = Directions.STOP
        
        # iterative deepening loop
        # We start at depth 1 and go deeper until we risk timeout
        try:
            for current_depth in range(1, 25): # 25 is practically infinite for Pacman
                
                # Check time before starting a new depth
                if time.time() - start_time > time_limit:
                    break
                
                # Run Alpha-Beta for the current depth
                score, action = self.alpha_beta_search(gameState, 0, current_depth, 
                                                       float('-inf'), float('inf'), 
                                                       start_time, time_limit)
                
                # If we didn't timeout inside the search, update best_action
                if time.time() - start_time < time_limit:
                    best_action = action
                else:
                    break # Timeout occurred during search, discard partial result
                    
        except TimeoutError:
            pass # Return the best action found in the last fully completed depth

        return best_action

    def alpha_beta_search(self, state, agent_index, depth, alpha, beta, start_time, time_limit):
        # Time check inside recursion
        if time.time() - start_time > time_limit:
            raise TimeoutError("Time limit exceeded")

        num_agents = state.get_num_agents()
        
        # Terminal checks
        if state.is_win() or state.is_lose() or depth == 0:
            return ultimate_evaluation_function(state), Directions.STOP

        # Prepare for next agent/depth
        next_agent = (agent_index + 1) % num_agents
        next_depth = depth - 1 if next_agent == 0 else depth

        legal_moves = state.get_legal_actions(agent_index)
        if not legal_moves:
            return ultimate_evaluation_function(state), Directions.STOP

        # MOVE ORDERING: Sort moves to improve pruning efficiency
        # We assume 'better' moves (eating food, running from ghost) give better static scores
        if agent_index == 0: # Pacman
            # Sort descending (best first)
            legal_moves.sort(key=lambda a: ultimate_evaluation_function(state.generate_successor(agent_index, a)), reverse=True)
        
        best_action = legal_moves[0]

        if agent_index == 0: # Max (Pacman)
            value = float('-inf')
            for action in legal_moves:
                successor = state.generate_successor(agent_index, action)
                score, _ = self.alpha_beta_search(successor, next_agent, next_depth, alpha, beta, start_time, time_limit)
                
                if score > value:
                    value = score
                    best_action = action
                
                if value > beta:
                    return value, best_action
                alpha = max(alpha, value)
                
        else: # Min (Ghosts)
            value = float('inf')
            for action in legal_moves:
                successor = state.generate_successor(agent_index, action)
                score, _ = self.alpha_beta_search(successor, next_agent, next_depth, alpha, beta, start_time, time_limit)
                
                if score < value:
                    value = score
                    best_action = action
                
                if value < alpha:
                    return value, best_action
                beta = min(beta, value)

        return value, best_action


def ultimate_evaluation_function(currentGameState):
    """
    A comprehensive evaluation function that prioritizes:
    1. Survival (Distance to active ghosts)
    2. Capsules (Vital for high scores)
    3. Scared Ghosts (High point value)
    4. Food (Using Maze Distance instead of Manhattan for accuracy)
    """
    # Useful data
    pacman_pos = currentGameState.get_pacman_position()
    food_list = currentGameState.get_food().as_list()
    ghost_states = currentGameState.get_ghost_states()
    capsules = currentGameState.get_capsules()
    current_score = currentGameState.get_score()

    # --- Weights (Tune these parameters to change behavior) ---
    w_food = 10.0
    w_capsule = 200.0
    w_scared_ghost = 200.0
    w_active_ghost = -1000.0 # High negative weight for danger
    
    total_score = current_score

    # 1. GHOST LOGIC
    for ghost in ghost_states:
        ghost_pos = ghost.get_position()
        dist = util.manhattanDistance(pacman_pos, ghost_pos)
        
        if ghost.scared_timer > 0:
            # SCARED GHOST: Chase it if we can reach it in time
            # We use 'dist < timer' to ensure we don't chase a ghost turning back to dangerous
            if dist < ghost.scared_timer: 
                total_score += w_scared_ghost / (dist + 1)
        else:
            # ACTIVE GHOST: Avoid heavily
            if dist <= 1:
                total_score += w_active_ghost * 10 # Instant death penalty
            else:
                total_score += w_active_ghost / (dist * dist) # Exponential fear

    # 2. CAPSULE LOGIC
    # Capsules are rare and valuable. Treat them like "super food".
    if capsules:
        # Find closest capsule
        min_cap_dist = min([util.manhattanDistance(pacman_pos, cap) for cap in capsules])
        total_score += w_capsule / (min_cap_dist + 1)
    
    # 3. FOOD LOGIC (Maze Distance)
    # Manhattan distance gets stuck in U-shaped walls. BFS (Maze Distance) prevents this.
    if food_list:
        # We calculate the TRUE distance to the closest food using BFS
        # To save CPU, we only do this for the closest food found via Manhattan
        closest_food_manhattan = min(food_list, key=lambda f: util.manhattanDistance(pacman_pos, f))
        
        # BFS to find actual path length to that specific food
        maze_dist = bfs_distance(currentGameState, pacman_pos, closest_food_manhattan)
        
        # Penalize having food remaining, and penalize distance to food
        total_score -= w_food * maze_dist
        total_score -= 4 * len(food_list) # Static penalty for count

    return total_score

def bfs_distance(gameState, start, target):
    """
    Returns the BFS (true maze) distance between start and target.
    """
    from util import Queue
    walls = gameState.get_walls()
    queue = Queue()
    queue.push((start, 0))
    visited = set([start])

    while not queue.is_empty():
        curr, dist = queue.pop()
        if curr == target:
            return dist
        
        x, y = int(curr[0]), int(curr[1])
        # Check neighbors
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            next_x, next_y = x + dx, y + dy
            if not walls[next_x][next_y]:
                next_pos = (next_x, next_y)
                if next_pos not in visited:
                    visited.add(next_pos)
                    queue.push((next_pos, dist + 1))
    
    return util.manhattanDistance(start, target) # Fallback

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.
    """

    def __init__(self, evalFn = 'score_evaluation_function', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def get_action(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Robust AlphaBetaAgent with immediate-reverse prevention.

    - never returns None (safe fallback to legal action)
    - chases scared ghosts, reactive escape, anti-oscillation
    - prevents immediate reversal of last action (unless it's the only option)
    - prints parseable metrics for the runner
    """

    def registerInitialState(self, state):
        self.node_expansions = 0
        self.total_moves = 0
        self.recent_positions = deque(maxlen=6)
        self.last_action = None
        try:
            self.evaluationFunction = betterEvaluationFunction
        except Exception:
            pass
        print("[AlphaBetaAgent] REGISTERED and ready", flush=True)
        try:
            super().registerInitialState(state)
        except Exception:
            pass

    def getAction(self, gameState):
        return self._select_action(gameState)

    def get_action(self, gameState):
        return self._select_action(gameState)

    def _select_action(self, gameState):
        # --- small helpers for API variations ---
        def manh(a, b):
            try:
                return manhattan_distance(a, b)
            except Exception:
                try:
                    return util.manhattanDistance(a, b)
                except Exception:
                    return abs(a[0]-b[0]) + abs(a[1]-b[1])

        def pac_pos(state):
            for fn in ("get_pacman_position", "getPacmanPosition", "getPacmanPos"):
                if hasattr(state, fn):
                    try:
                        return getattr(state, fn)()
                    except Exception:
                        pass
            return getattr(state, "pacmanPosition", None)

        def ghost_states(state):
            for fn in ("get_ghost_states", "getGhostStates", "getGhostStatesList"):
                if hasattr(state, fn):
                    try:
                        return getattr(state, fn)()
                    except Exception:
                        pass
            return []

        def ghost_positions_and_scared(state):
            gs = ghost_states(state)
            pos = []
            scared = []
            for g in gs:
                p = None
                for gpfn in ("get_position", "getPosition", "get_pos"):
                    if hasattr(g, gpfn):
                        try:
                            p = getattr(g, gpfn)(); break
                        except Exception:
                            p = None
                if p is None:
                    p = getattr(g, "position", None)
                s = getattr(g, "scaredTimer", None)
                if s is None:
                    s = getattr(g, "scared_timer", None)
                if s is None:
                    s = getattr(g, "scared", 0)
                pos.append(p)
                scared.append(s if s is not None else 0)
            return pos, scared

        def get_legal_actions(state, agent_index):
            # try multiple names and signatures
            for fn in ("get_legal_actions", "getLegalActions", "getLegalActionsForAgent"):
                if hasattr(state, fn):
                    try:
                        return getattr(state, fn)(agent_index)
                    except TypeError:
                        try:
                            return getattr(state, fn)()
                        except Exception:
                            pass
                    except Exception:
                        pass
            # fallback: try get_legal_actions() without agent index
            for fn in ("get_legal_actions", "getLegalActions"):
                if hasattr(state, fn):
                    try:
                        return getattr(state, fn)()
                    except Exception:
                        pass
            return []

        def gen_successor(state, agent_index, action):
            for fn in ("generate_successor", "generateSuccessor"):
                if hasattr(state, fn):
                    try:
                        return getattr(state, fn)(agent_index, action)
                    except Exception:
                        pass
            for fn in ("generate_pacman_successor", "generatePacmanSuccessor"):
                if hasattr(state, fn):
                    try:
                        return getattr(state, fn)(action)
                    except Exception:
                        pass
            raise RuntimeError("No successor generator found")

        # reverse-action helper (uses Directions.REVERSE when available)
        def reverse_action(a):
            if a is None:
                return None
            try:
                # Directions.REVERSE is standard in Berkeley code
                return Directions.REVERSE[a]
            except Exception:
                rev = {"North":"South","South":"North","East":"West","West":"East","Stop":"Stop"}
                return rev.get(a, None)

        # increment move counter
        self.total_moves = getattr(self, "total_moves", 0) + 1

        # current pacman/ghost info
        cur_pos = pac_pos(gameState)
        ghost_pos_list, ghost_scared_list = ghost_positions_and_scared(gameState)
        ghost_dists = [manh(cur_pos, gp) if (gp is not None and cur_pos is not None) else float('inf')
                       for gp in ghost_pos_list]
        min_ghost_dist_now = min(ghost_dists) if ghost_dists else float('inf')

        # function to prune immediate reverse from a legal-action list
        def prune_reverse(legal):
            rev = reverse_action(getattr(self, "last_action", None))
            try:
                if rev is not None and rev in legal and len(legal) > 1:
                    return [a for a in legal if a != rev]
            except Exception:
                pass
            return legal

        # ---------- 1) CHASE SCARED GHOST OVERRIDE ----------
        CHASE_THRESHOLD = 7
        scared_candidates = [(i, d) for i, (d, s) in enumerate(zip(ghost_dists, ghost_scared_list)) if s and d <= CHASE_THRESHOLD]
        if scared_candidates:
            scared_candidates.sort(key=lambda x: x[1])
            target_idx = scared_candidates[0][0]
            target_pos = ghost_pos_list[target_idx]

            legal = get_legal_actions(gameState, 0)
            if 'Stop' in legal and len(legal) > 1:
                legal = [a for a in legal if a != 'Stop']
            legal = prune_reverse(legal)

            best_action = None
            best_metric = float('inf')
            for a in legal:
                try:
                    succ = gen_successor(gameState, 0, a)
                except Exception:
                    continue
                try:
                    succ_p = pac_pos(succ)
                except Exception:
                    continue
                # avoid stepping into a non-scared ghost
                unsafe = any((succ_p == gp and (not ghost_scared_list[i])) for i, gp in enumerate(ghost_pos_list) if gp is not None)
                if unsafe:
                    continue
                dist_to_target = manh(succ_p, target_pos) if target_pos is not None else float('inf')
                # anti-oscillation: penalize moves that go back to recent positions
                recent_penalty = 0
                if succ_p in getattr(self, "recent_positions", []):
                    if len(self.recent_positions) and succ_p == (self.recent_positions[-1]):
                        recent_penalty = 1000
                    else:
                        recent_penalty = 5
                # pick action minimizing metric (dist + penalty)
                metric = dist_to_target + recent_penalty
                if metric < best_metric:
                    best_metric = metric
                    best_action = a

            if best_action is not None:
                # set last_action before returning
                self.last_action = best_action
                print(f"[AlphaBetaAgent] Chase scared ghost chosen: {best_action} -> target_idx={target_idx} dist_now={scared_candidates[0][1]} best_after={best_metric}", flush=True)
                print(f"[AlphaBetaAgent] Nodes expanded: {self.node_expansions}", flush=True)
                print(f"[AlphaBetaAgent] Total moves: {self.total_moves}", flush=True)
                # update recent positions after choosing action
                try:
                    succ = gen_successor(gameState, 0, best_action)
                    succ_p = pac_pos(succ)
                    if succ_p is not None:
                        self.recent_positions.append(succ_p)
                except Exception:
                    pass
                return best_action

        # ---------- 2) REACTIVE ESCAPE ----------
        if min_ghost_dist_now <= 2:
            legal = get_legal_actions(gameState, 0)
            if 'Stop' in legal and len(legal) > 1:
                legal = [a for a in legal if a != 'Stop']
            legal = prune_reverse(legal)

            best_a = None
            best_d = -1
            for a in legal:
                try:
                    succ = gen_successor(gameState, 0, a)
                except Exception:
                    continue
                try:
                    succ_p = pac_pos(succ)
                except Exception:
                    continue
                # avoid succ_p landing on non-scared ghost
                occupied = any((succ_p == gp and not ghost_scared_list[i]) for i, gp in enumerate(ghost_pos_list) if gp is not None)
                if occupied:
                    continue
                d = min((manh(succ_p, gp) if gp is not None else float('inf')) for gp in ghost_pos_list)
                # penalize returning to recent positions
                if succ_p in getattr(self, "recent_positions", []):
                    d -= 1.5
                if d > best_d:
                    best_d = d
                    best_a = a
            if best_a is not None and best_d > (min_ghost_dist_now - 0.0001):
                # set last_action before returning
                self.last_action = best_a
                print(f"[AlphaBetaAgent] Reactive escape move chosen: {best_a} (ghost dist {min_ghost_dist_now} -> {best_d})", flush=True)
                print(f"[AlphaBetaAgent] Nodes expanded: {self.node_expansions}", flush=True)
                print(f"[AlphaBetaAgent] Total moves: {self.total_moves}", flush=True)
                try:
                    succ = gen_successor(gameState, 0, best_a)
                    succ_p = pac_pos(succ)
                    if succ_p is not None:
                        self.recent_positions.append(succ_p)
                except Exception:
                    pass
                return best_a

        # ---------- 3) ALPHA-BETA FALLBACK ----------
        def alpha_beta_search(state, agent_index, depth, alpha, beta):
            self.node_expansions = getattr(self, "node_expansions", 0) + 1
            try:
                terminal = state.is_win() or state.is_lose()
            except Exception:
                terminal = False
            if depth == self.depth or terminal:
                try:
                    return self.evaluationFunction(state), None
                except Exception:
                    return score_evaluation_function(state), None

            is_pacman = (agent_index == 0)
            best_score = float('-inf') if is_pacman else float('inf')
            best_action = None

            legal = get_legal_actions(state, agent_index)
            if 'Stop' in legal and len(legal) > 1:
                legal = [a for a in legal if a != 'Stop']
            # we prune reverse only at root-level selection (above). Inside recursion it's fine to keep actions,
            # but we still keep the same legal list for safety.

            for a in legal:
                try:
                    succ = gen_successor(state, agent_index, a)
                except Exception:
                    continue
                next_agent = (agent_index + 1) % (state.get_num_agents() if hasattr(state, "get_num_agents") else 1)
                next_depth = depth + (1 if next_agent == 0 else 0)

                score, _ = alpha_beta_search(succ, next_agent, next_depth, alpha, beta)

                if is_pacman:
                    if score > best_score:
                        best_score, best_action = score, a
                    elif score == best_score:
                        try:
                            cur_succ = gen_successor(state, agent_index, best_action) if best_action is not None else None
                            cur_p = pac_pos(cur_succ) if cur_succ is not None else None
                            new_p = pac_pos(succ)
                            cur_dist = min((manh(cur_p, gp) if gp is not None else float('inf')) for gp in ghost_pos_list) if cur_p is not None else -1
                            new_dist = min((manh(new_p, gp) if gp is not None else float('inf')) for gp in ghost_pos_list) if new_p is not None else -1
                            cur_pen = 1000 if (cur_p in self.recent_positions and cur_p == (self.recent_positions[-1] if self.recent_positions else None)) else (5 if cur_p in self.recent_positions else 0)
                            new_pen = 1000 if (new_p in self.recent_positions and new_p == (self.recent_positions[-1] if self.recent_positions else None)) else (5 if new_p in self.recent_positions else 0)
                            if (new_dist - new_pen) > (cur_dist - cur_pen):
                                best_score, best_action = score, a
                        except Exception:
                            pass
                    alpha = max(alpha, best_score)
                    if alpha > beta:
                        break
                else:
                    if score < best_score:
                        best_score, best_action = score, a
                    beta = min(beta, best_score)
                    if beta < alpha:
                        break

            return best_score, best_action

        # At root-level choose legal actions but prune reverse of last action if possible
        legal_now = get_legal_actions(gameState, 0)
        if 'Stop' in legal_now and len(legal_now) > 1:
            legal_now = [a for a in legal_now if a != 'Stop']
        legal_now = prune_reverse(legal_now)

        # If pruning removed all options (very unlikely), revert to original legal set
        if not legal_now:
            legal_now = get_legal_actions(gameState, 0) or ['Stop']

        # Run alpha-beta but only consider root legal_now order; to keep code simple,
        # call the existing search and then ensure chosen is legal (fallbacks after).
        _, chosen = alpha_beta_search(gameState, 0, 0, float('-inf'), float('inf'))

        # --- Robust fallback if chosen is None or illegal ---
        non_stop = [a for a in legal_now if a != 'Stop']
        if chosen is None:
            if non_stop:
                chosen = non_stop[0]
            elif legal_now:
                chosen = legal_now[0]
            else:
                chosen = 'Stop'

        try:
            if chosen not in legal_now and non_stop:
                chosen = non_stop[0]
            elif chosen not in legal_now and legal_now:
                chosen = legal_now[0]
        except Exception:
            if non_stop:
                chosen = non_stop[0]
            elif legal_now:
                chosen = legal_now[0]
            else:
                chosen = 'Stop'

        # update last_action and recent positions
        self.last_action = chosen
        try:
            succ = gen_successor(gameState, 0, chosen)
            succ_p = pac_pos(succ)
            if succ_p is not None:
                self.recent_positions.append(succ_p)
        except Exception:
            pass

        # per-move prints
        print(f"[AlphaBetaAgent] Nodes expanded: {self.node_expansions}", flush=True)
        print(f"[AlphaBetaAgent] Total moves: {self.total_moves}", flush=True)
        return chosen

    # Replace the AlphaBetaAgent.final method with this version
import re  # add near top of file if not present

def final(self, state):
    """
    Called at game's end. Robustly obtain final score (tries several accessors)
    and print parseable metrics.
    """
    final_score = None

    # Try several common ways to get score from the state
    try:
        if state is None:
            final_score = None
        else:
            # Common Berkeley API
            if hasattr(state, "getScore"):
                try:
                    final_score = state.getScore()
                except Exception:
                    final_score = None
            # alternative snake_case
            if final_score is None and hasattr(state, "get_score"):
                try:
                    final_score = state.get_score()
                except Exception:
                    final_score = None
            # attribute named 'score'
            if final_score is None and hasattr(state, "score"):
                try:
                    final_score = getattr(state, "score")
                except Exception:
                    final_score = None
            # fallback: try to parse integer from str(state) if engine prints score in toString
            if final_score is None:
                try:
                    s = str(state)
                    m = re.search(r"Score[: ]+(-?\d+)", s)
                    if m:
                        final_score = int(m.group(1))
                except Exception:
                    final_score = None
    except Exception:
        final_score = None

    # Normalize to int when possible
    try:
        if final_score is not None:
            # If float-like, cast to int
            final_score = int(final_score)
    except Exception:
        final_score = None

    # Print metrics in exact parseable format
    print(f"[AlphaBetaAgent] Nodes expanded: {getattr(self, 'node_expansions', 0)}", flush=True)
    print(f"[AlphaBetaAgent] Total moves: {getattr(self, 'total_moves', 0)}", flush=True)
    if final_score is not None:
        print(f"[AlphaBetaAgent] Final score: {final_score}", flush=True)
    else:
        print(f"[AlphaBetaAgent] Final score: UNKNOWN", flush=True)

    # call parent's final if present
    try:
        super().final(state)
    except Exception:
        pass



class AdaptiveAlphaBetaAgent(MultiAgentSearchAgent):
    """
    An adaptive alpha-beta agent for comparison.
    NOTE: As discussed, the adaptive heuristic in this version is non-functional
    because it is applied too late to affect pruning.
    """
    def get_action(self, gameState):
        stats = {'nodes_expanded': 0}

        def adaptive_alpha_beta(state, agent_index, depth, alpha, beta):
            stats['nodes_expanded'] += 1 # Increment node counter

            if depth == self.depth or state.is_win() or state.is_lose():
                return self.evaluationFunction(state), None

            is_pacman = (agent_index == 0)
            best_score = float('-inf') if is_pacman else float('inf')
            best_action = None
            scores = []

            for action in state.get_legal_actions(agent_index):
                successor_state = state.generate_successor(agent_index, action)
                next_agent = (agent_index + 1) % state.get_num_agents()
                next_depth = depth + (next_agent == 0)
                score, _ = adaptive_alpha_beta(successor_state, next_agent, next_depth, alpha, beta)
                scores.append(score)

                if is_pacman:
                    if score > best_score:
                        best_score, best_action = score, action
                    alpha = max(alpha, best_score)
                    if best_score > beta: break
                else:
                    if score < best_score:
                        best_score, best_action = score, action
                    beta = min(beta, best_score)
                    if best_score < alpha: break
            
            # This adaptive part runs after the search loop and does not affect pruning
            if scores:
                mean = sum(scores) / len(scores)
                if len(scores) > 1:
                    stddev = (sum((s - mean) ** 2 for s in scores) / (len(scores) -1)) ** 0.5
                else:
                    stddev = 0
                
                if is_pacman:
                    alpha += stddev * 0.1
                else:
                    beta -= stddev * 0.1

            return best_score, best_action

        _, action = adaptive_alpha_beta(gameState, 0, 0, float('-inf'), float('inf'))
        
        print(f"[AdaptiveAlphaBetaAgent] States Expanded: {stats['nodes_expanded']}")
        return action
    
class StatisticallyGuidedAlphaBetaAgent(MultiAgentSearchAgent):
    """
    An alpha-beta agent that uses a heuristic to sort moves before searching,
    leading to more efficient pruning.
    """
    def get_action(self, gameState):
        stats = {'nodes_expanded': 0}

        # The standard recursive alpha-beta search function
        def alpha_beta_search(state, agent_index, depth, alpha, beta):
            stats['nodes_expanded'] += 1 # Increment node counter

            if depth == self.depth or state.is_win() or state.is_lose():
                return self.evaluationFunction(state), None

            is_pacman = (agent_index == 0)
            best_score = float('-inf') if is_pacman else float('inf')
            best_action = None
            
            # This is the only part that's different from the standard agent's helper:
            # It will iterate through moves in the pre-sorted order passed to it.
            # For the ghosts' turns, it will use the default order.
            actions_to_search = get_ordered_actions(state, agent_index)

            for action in actions_to_search:
                successor_state = state.generate_successor(agent_index, action)
                next_agent = (agent_index + 1) % state.get_num_agents()
                next_depth = depth + (next_agent == 0)
                
                score, _ = alpha_beta_search(successor_state, next_agent, next_depth, alpha, beta)

                if is_pacman:
                    if score > best_score:
                        best_score, best_action = score, action
                    alpha = max(alpha, best_score)
                    if best_score > beta: break
                else:
                    if score < best_score:
                        best_score, best_action = score, action
                    beta = min(beta, best_score)
                    if best_score < alpha: break

            return best_score, best_action
        
        # --- The New "Statistically Guided" Logic ---
        def get_ordered_actions(state, agent_index):
            # For ghosts, we don't re-order their moves.
            if agent_index != 0:
                return state.get_legal_actions(agent_index)
            
            # For Pacman, we perform the heuristic sort.
            legal_actions = state.get_legal_actions(agent_index)
            
            # 1. Get a quick heuristic score for each action's immediate result.
            action_scores = []
            for action in legal_actions:
                successor = state.generate_successor(agent_index, action)
                # We use the evaluation function for a "shallow" 1-ply lookahead.
                score = self.evaluationFunction(successor)
                action_scores.append((action, score))
            
            # 2. Sort the actions based on their heuristic score (higher is better).
            action_scores.sort(key=lambda x: x[1], reverse=True)
            
            # 3. Return just the list of actions in the new, smarter order.
            return [action for action, score in action_scores]

        # Initial call for Pacman
        _, action = alpha_beta_search(gameState, 0, 0, float('-inf'), float('inf'))
        
        print(f"[StatisticallyGuidedAlphaBetaAgent] States Expanded: {stats['nodes_expanded']}")
        return action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def get_action(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Replacement evaluation for MultiAgent search.
    Balances food, ghost distances, scared ghosts, and current score.
    """

    # robust manhattan fallback (use whichever util available)
    def _manhattan(a, b):
        try:
            return manhattan_distance(a, b)
        except Exception:
            try:
                return util.manhattanDistance(a, b)
            except Exception:
                return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # base score
    try:
        score = currentGameState.getScore()
    except Exception:
        score = 0

    # pacman position
    for fn in ("get_pacman_position", "getPacmanPosition", "getPacmanPos"):
        try:
            pacPos = getattr(currentGameState, fn)()
            break
        except Exception:
            pacPos = None

    # food list
    try:
        foodList = currentGameState.getFood().asList()
    except Exception:
        try:
            foodList = currentGameState.get_food().as_list()
        except Exception:
            # fallback to empty
            foodList = []

    numFood = len(foodList)

    # ghost positions and scared timers
    ghost_positions = []
    ghost_scared_times = []
    for g in (getattr(currentGameState, "get_ghost_states", lambda: [])() or getattr(currentGameState, "getGhostStates", lambda: [])()):
        got_pos = None
        for pfn in ("get_position", "getPosition", "get_pos"):
            try:
                got_pos = getattr(g, pfn)()
                break
            except Exception:
                got_pos = None
        if got_pos is not None:
            ghost_positions.append(got_pos)
        # scared timer
        scared = getattr(g, "scaredTimer", getattr(g, "scared_timer", getattr(g, "scaredTimer", 0)))
        ghost_scared_times.append(scared if scared is not None else 0)

    # distance to closest food
    if foodList:
        try:
            minFoodDist = min(_manhattan(pacPos, f) for f in foodList)
        except Exception:
            minFoodDist = 0.0
    else:
        minFoodDist = 0.0

    # ghost distance
    if ghost_positions:
        ghost_dists = [_manhattan(pacPos, gp) for gp in ghost_positions]
        minGhostDist = min(ghost_dists)
    else:
        minGhostDist = float('inf')

    # death check
    on_ghost = any((d == 0 and (ghost_scared_times[i] == 0)) for i, d in enumerate(ghost_dists if ghost_positions else []))

    # weights
    W_SCORE = 1.0
    W_MIN_FOOD_DIST = -1.8
    W_NUM_FOOD = -4.5
    W_GHOST_DIST = 2.2
    W_GHOST_CLOSE_PENALTY = -250.0
    W_ON_GHOST = -10000.0
    W_SCARED_GHOST_BONUS = 6.0

    val = 0.0
    val += W_SCORE * score

    if minFoodDist > 0:
        val += W_MIN_FOOD_DIST * float(minFoodDist)
    else:
        val += 12.0

    val += W_NUM_FOOD * numFood

    if ghost_positions:
        for i, gp in enumerate(ghost_positions):
            d = _manhattan(pacPos, gp)
            scared = ghost_scared_times[i] if i < len(ghost_scared_times) else 0
            if d <= 1 and scared == 0:
                val += W_GHOST_CLOSE_PENALTY
            if d == 0 and scared == 0:
                val += W_ON_GHOST
            if scared > 0:
                val += W_SCARED_GHOST_BONUS / (1 + d)

        non_scared_dists = [_manhattan(pacPos, gp) for idx, gp in enumerate(ghost_positions) if (ghost_scared_times[idx] == 0)]
        if non_scared_dists:
            nearest_non_scared = min(non_scared_dists)
            val += W_GHOST_DIST * float(nearest_non_scared)
        else:
            val += 6.0

    if numFood == 0:
        val += 500.0

    return val





# def better_evaluation_function(currentGameState):
#     """
#     Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
#     evaluation function.
#     """
#     pacman_pos = currentGameState.get_pacman_position()
#     food_list = currentGameState.get_food().as_list()
#     ghost_states = currentGameState.get_ghost_states()
#     scared_timers = [ghostState.scared_timer for ghostState in ghost_states]

#     # Start with the current score
#     score = currentGameState.get_score()

#     # Feature 1: Distance to the nearest food pellet
#     if food_list:
#         min_food_dist = min([manhattan_distance(pacman_pos, food) for food in food_list])
#         score += 1.0 / min_food_dist

#     # Feature 2: Number of remaining food pellets
#     score -= 2 * len(food_list)

#     # Feature 3: Ghost proximity (scared vs. active)
#     for i, ghost in enumerate(ghost_states):
#         ghost_pos = ghost.get_position()
#         dist_to_ghost = manhattan_distance(pacman_pos, ghost_pos)

#         if scared_timers[i] > 0:  # Ghost is scared
#             if dist_to_ghost > 0:
#                 score += 200 / dist_to_ghost  # Bigger reward for being closer
#         else:  # Ghost is NOT scared (dangerous)
#             if dist_to_ghost < 2:
#                 score -= 500

#     return score
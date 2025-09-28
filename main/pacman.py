# pacman.py
# ---------
# Licensing Information: You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.

from __future__ import print_function
from future import standard_library
standard_library.install_aliases()

from builtins import str, range, object
import sys, time, random, os, importlib

from game import GameStateData, Game, Directions, Actions
from util import nearest_point, manhattan_distance
import util, layout

class GameState(object):
    explored = set()

    def get_and_reset_explored():
        tmp = GameState.explored.copy()
        GameState.explored = set()
        return tmp
    get_and_reset_explored = staticmethod(get_and_reset_explored)

    def get_legal_actions(self, agent_index=0):
        if self.is_win() or self.is_lose(): return []
        return PacmanRules.get_legal_actions(self) if agent_index == 0 else GhostRules.get_legal_actions(self, agent_index)

    def generate_successor(self, agent_index, action):
        if self.is_win() or self.is_lose(): raise Exception("Can't generate a successor of a terminal state.")
        state = GameState(self)
        if agent_index == 0:
            state.data._eaten = [False for _ in range(state.get_num_agents())]
            PacmanRules.apply_action(state, action)
        else:
            GhostRules.apply_action(state, action, agent_index)
        if agent_index == 0: state.data.score_change += -TIME_PENALTY
        else: GhostRules.decrement_timer(state.data.agent_states[agent_index])
        GhostRules.check_death(state, agent_index)
        state.data._agent_moved = agent_index
        state.data.score += state.data.score_change
        GameState.explored.add(self.__hash__()); GameState.explored.add(state.__hash__())
        return state

    def get_legal_pacman_actions(self): return self.get_legal_actions(0)
    def generate_pacman_successor(self, action): return self.generate_successor(0, action)

    def get_pacman_state(self): return self.data.agent_states[0].copy()
    def get_pacman_position(self): return self.data.agent_states[0].get_position()
    def get_ghost_states(self): return self.data.agent_states[1:]
    def get_ghost_state(self, agent_index):
        if agent_index == 0 or agent_index >= self.get_num_agents():
            raise Exception("Invalid index passed to get_ghost_state")
        return self.data.agent_states[agent_index]
    def get_ghost_position(self, agent_index):
        if agent_index == 0: raise Exception("Pacman's index passed to get_ghost_position")
        return self.data.agent_states[agent_index].get_position()
    def get_ghost_positions(self): return [s.get_position() for s in self.get_ghost_states()]
    def get_num_agents(self): return len(self.data.agent_states)
    def get_score(self): return float(self.data.score)
    def get_capsules(self): return self.data.capsules
    def get_num_food(self): return self.data.food.count()
    def get_food(self): return self.data.food
    def get_walls(self): return self.data.layout.walls
    def has_food(self, x, y): return self.data.food[x][y]
    def has_wall(self, x, y): return self.data.layout.walls[x][y]
    def is_lose(self): return self.data._lose
    def is_win(self): return self.data._win

    def __init__(self, prev_state=None):
        self.data = GameStateData(prev_state.data) if prev_state is not None else GameStateData()

    def deep_copy(self):
        state = GameState(self); state.data = self.data.deep_copy(); return state

    def __eq__(self, other): return hasattr(other, "data") and self.data == other.data

    def __hash__(self):
        sep = "|"
        base = 1; food_hash = 0
        for row in self.data.food.data:
            for cell in row:
                if cell: food_hash += base
                base *= 2
        food_hash = str(food_hash)
        agent_hash = sep + sep.join([
            str(e.scared_timer) + sep + str(e.configuration.pos) + sep + str(e.configuration.direction)
            for e in self.data.agent_states
        ]) + sep
        return hash(food_hash + sep + agent_hash + sep + str(self.data.capsules) + sep + str(self.data.score))

    def __str__(self): return str(self.data)

    def initialize(self, layout_obj, num_ghost_agents=1000):
        self.data.initialize(layout_obj, num_ghost_agents)

SCARED_TIME = 40
COLLISION_TOLERANCE = 0.7
TIME_PENALTY = 1

class ClassicGameRules(object):
    def __init__(self, timeout=30): self.timeout = timeout
    def new_game(self, layout, pacman_agent, ghost_agents, display, quiet=False, catch_exceptions=False):
        agents = [pacman_agent] + ghost_agents[: layout.get_num_ghosts()]
        init_state = GameState(); init_state.initialize(layout, len(ghost_agents))
        game = Game(agents, display, self, catch_exceptions=catch_exceptions); game.state = init_state
        self.initial_state = init_state.deep_copy(); self.quiet = quiet; return game
    def process(self, state, game):
        if state.is_win(): self.win(state, game)
        if state.is_lose(): self.lose(state, game)
    def win(self, state, game):
        if not self.quiet: print("Pacman emerges victorious! Score: %d" % state.data.score)
        game.game_over = True
    def lose(self, state, game):
        if not self.quiet: print("Pacman died! Score: %d" % state.data.score)
        game.game_over = True
    def get_progress(self, game): return float(game.state.get_num_food()) / self.initial_state.get_num_food()
    def agent_crash(self, game, agent_index): print("Pacman crashed" if agent_index == 0 else "A ghost crashed")
    def get_max_total_time(self, agent_index): return self.timeout
    def get_max_startup_time(self, agent_index): return self.timeout
    def get_move_warning_time(self, agent_index): return self.timeout
    def get_move_timeout(self, agent_index): return self.timeout
    def get_max_time_warnings(self, agent_index): return 0

class PacmanRules(object):
    PACMAN_SPEED = 1
    def get_legal_actions(state):
        return Actions.get_possible_actions(state.get_pacman_state().configuration, state.data.layout.walls)
    get_legal_actions = staticmethod(get_legal_actions)
    def apply_action(state, action):
        legal = PacmanRules.get_legal_actions(state)
        if action not in legal: raise Exception("Illegal action " + str(action))
        pac = state.data.agent_states[0]
        vector = Actions.direction_to_vector(action, PacmanRules.PACMAN_SPEED)
        pac.configuration = pac.configuration.generate_successor(vector)
        nxt = pac.configuration.get_position(); near = nearest_point(nxt)
        if manhattan_distance(near, nxt) <= 0.5: PacmanRules.consume(near, state)
    apply_action = staticmethod(apply_action)
    def consume(position, state):
        x, y = position
        if state.data.food[x][y]:
            state.data.score_change += 10
            state.data.food = state.data.food.copy()
            state.data.food[x][y] = False
            state.data._food_eaten = position
            if state.get_num_food() == 0 and not state.data._lose:
                state.data.score_change += 500; state.data._win = True
        if position in state.get_capsules():
            state.data.capsules.remove(position); state.data._capsule_eaten = position
            for i in range(1, len(state.data.agent_states)):
                state.data.agent_states[i].scared_timer = SCARED_TIME
    consume = staticmethod(consume)

class GhostRules(object):
    GHOST_SPEED = 1.0
    def get_legal_actions(state, ghost_index):
        conf = state.get_ghost_state(ghost_index).configuration
        poss = Actions.get_possible_actions(conf, state.data.layout.walls)
        reverse = Actions.reverse_direction(conf.direction)
        if Directions.STOP in poss: poss.remove(Directions.STOP)
        if reverse in poss and len(poss) > 1: poss.remove(reverse)
        return poss
    get_legal_actions = staticmethod(get_legal_actions)
    def apply_action(state, action, ghost_index):
        legal = GhostRules.get_legal_actions(state, ghost_index)
        if action not in legal: raise Exception("Illegal ghost action " + str(action))
        gs = state.data.agent_states[ghost_index]
        speed = GhostRules.GHOST_SPEED
        if gs.scared_timer > 0: speed /= 2.0
        vector = Actions.direction_to_vector(action, speed)
        gs.configuration = gs.configuration.generate_successor(vector)
    apply_action = staticmethod(apply_action)
    def decrement_timer(ghost_state):
        timer = ghost_state.scared_timer
        if timer == 1: ghost_state.configuration.pos = nearest_point(ghost_state.configuration.pos)
        ghost_state.scared_timer = max(0, timer - 1)
    decrement_timer = staticmethod(decrement_timer)
    def check_death(state, agent_index):
        pac = state.get_pacman_position()
        if agent_index == 0:
            for idx in range(1, len(state.data.agent_states)):
                gs = state.data.agent_states[idx]; gp = gs.configuration.get_position()
                if GhostRules.can_kill(pac, gp): GhostRules.collide(state, gs, idx)
        else:
            gs = state.data.agent_states[agent_index]; gp = gs.configuration.get_position()
            if GhostRules.can_kill(pac, gp): GhostRules.collide(state, gs, agent_index)
    check_death = staticmethod(check_death)
    def collide(state, ghost_state, agent_index):
        if ghost_state.scared_timer > 0:
            state.data.score_change += 200
            GhostRules.place_ghost(state, ghost_state)
            ghost_state.scared_timer = 0
            state.data._eaten[agent_index] = True
        else:
            if not state.data._win:
                state.data.score_change -= 500; state.data._lose = True
    collide = staticmethod(collide)
    def can_kill(pac, ghost): return manhattan_distance(ghost, pac) <= COLLISION_TOLERANCE
    can_kill = staticmethod(can_kill)
    def place_ghost(state, ghost_state): ghost_state.configuration = ghost_state.start
    place_ghost = staticmethod(place_ghost)

def default(s): return s + " [Default: %default]"

def parse_agent_args(s):
    if s is None: return {}
    pieces = s.split(","); opts = {}
    for p in pieces:
        if "=" in p: k, v = p.split("=")
        else: k, v = p, 1
        opts[k] = v
    return opts

def read_command(argv):
    from optparse import OptionParser
    usage = "USAGE: python pacman.py <options>"
    parser = OptionParser(usage)
    parser.add_option("-n","--num_games", dest="num_games", type="int", default=1)
    parser.add_option("-l","--layout", dest="layout", default="medium_classic")
    parser.add_option("-p","--pacman", dest="pacman", default="KeyboardAgent")
    parser.add_option("-t","--text_graphics", action="store_true", dest="text_graphics", default=False)
    parser.add_option("-q","--quiet_text_graphics", action="store_true", dest="quiet_graphics", default=False)
    parser.add_option("-g","--ghosts", dest="ghost", default="RandomGhost")
    parser.add_option("-k","--numghosts", type="int", dest="num_ghosts", default=4)
    parser.add_option("-z","--zoom", type="float", dest="zoom", default=None)
    parser.add_option("-f","--fix_random_seed", action="store_true", dest="fix_random_seed", default=False)
    parser.add_option("-r","--record_actions", action="store_true", dest="record", default=False)
    parser.add_option("--replay", dest="game_to_replay", default=None)
    parser.add_option("-a","--agent_args", dest="agent_args")
    parser.add_option("-x","--num_training", dest="num_training", type="int", default=0)
    parser.add_option("--frame_time", dest="frame_time", type="float", default=0.1)
    parser.add_option("-c","--catch_exceptions", action="store_true", dest="catch_exceptions", default=False)
    parser.add_option("--timeout", dest="timeout", type="int", default=30)

    options, other = parser.parse_args(argv)
    if other: raise Exception("Command line input not understood: " + str(other))
    args = {}

    if options.fix_random_seed: random.seed("cs188")

    args["layout"] = layout.get_layout(options.layout)
    if args["layout"] is None: raise Exception("The layout " + options.layout + " cannot be found")

    default_zoom = 1.0
    lines = len(f'''{args["layout"]}'''.split("\n"))
    if lines > 25: default_zoom = 25/float(lines)

    no_keyboard = options.game_to_replay is None and (options.text_graphics or options.quiet_graphics)
    pacman_type = load_agent(options.pacman, no_keyboard)
    agent_opts = parse_agent_args(options.agent_args)
    if options.num_training > 0:
        args["num_training"] = options.num_training
        if "num_training" not in agent_opts: agent_opts["num_training"] = options.num_training
    args["pacman"] = pacman_type(**agent_opts)

    ghost_type = load_agent(options.ghost, no_keyboard)
    args["ghosts"] = [ghost_type(i + 1) for i in range(options.num_ghosts)]

    if options.quiet_graphics:
        import text_display; args["display"] = text_display.NullGraphics()
    elif options.text_graphics:
        import text_display; text_display.SLEEP_TIME = options.frame_time; args["display"] = text_display.PacmanGraphics()
    else:
        import graphics_display; args["display"] = graphics_display.PacmanGraphics(options.zoom or default_zoom, frame_time=options.frame_time)

    args["num_games"] = options.num_games
    args["record"] = options.record
    args["catch_exceptions"] = options.catch_exceptions
    args["timeout"] = options.timeout

    if options.game_to_replay is not None:
        print("Replaying recorded game %s." % options.game_to_replay)
        import pickle
        with open(options.game_to_replay, "rb") as f:
            recorded = pickle.load(f)
        recorded["display"] = args["display"]
        replay_game(**recorded); sys.exit(0)
    return args

def _safe_import(modname):
    try: return importlib.import_module(modname)
    except Exception: return None

def _discover_agent_classes():
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path: sys.path.insert(0, here)
    candidates = []
    for fname in os.listdir(here):
        low = fname.lower()
        if low.endswith("agents.py") or low.endswith("_agents.py"):
            candidates.append(os.path.splitext(fname)[0])
    for extra in ("ghost_agents", "search_agents"):
        p = os.path.join(here, extra + ".py")
        if os.path.exists(p) and extra not in candidates: candidates.append(extra)
    classes = {}
    for modname in candidates:
        mod = _safe_import(modname)
        if not mod: continue
        for name in dir(mod):
            obj = getattr(mod, name)
            if isinstance(obj, type): classes[name] = obj
    return classes

def load_agent(pacman, nographics):
    classes = _discover_agent_classes()
    if pacman in classes:
        if nographics and pacman in ("KeyboardAgent", "KeyboardGhost"):
            raise Exception("Using the keyboard requires graphics (not text display)")
        return classes[pacman]
    raise Exception("The agent '" + pacman + "' is not specified in any *agents.py.")

def replay_game(layout, actions, display):
    import pacman_agents, ghost_agents
    rules = ClassicGameRules()
    agents = [pacman_agents.GreedyAgent()] + [ghost_agents.RandomGhost(i + 1) for i in range(layout.get_num_ghosts())]
    game = rules.new_game(layout, agents[0], agents[1:], display); state = game.state
    display.initialize(state.data)
    for action in actions:
        state = state.generate_successor(*action); display.update(state.data); rules.process(state, game)
    display.finish()

def run_games(layout, pacman, ghosts, display, num_games, record, num_training=0, catch_exceptions=False, timeout=30):
    import __main__; __main__.__dict__["_display"] = display
    rules = ClassicGameRules(timeout); games = []
    for i in range(num_games):
        be_quiet = i < num_training
        if be_quiet:
            import text_display; game_display = text_display.NullGraphics(); rules.quiet = True
        else:
            game_display = display; rules.quiet = False
        game = rules.new_game(layout, pacman, ghosts, game_display, be_quiet, catch_exceptions)
        game.run()
        if not be_quiet: games.append(game)
        if record:
            import pickle
            fname = ("recorded-game-%d" % (i + 1)) + "-".join([str(t) for t in time.localtime()[1:6]])
            with open(fname, "wb") as f:
                pickle.dump({"layout": layout, "actions": game.move_history}, f)
    if (num_games - num_training) > 0:
        scores = [g.state.get_score() for g in games]
        wins = [g.state.is_win() for g in games]
        win_rate = wins.count(True) / float(len(wins))
        print("Average Score:", sum(scores) / float(len(scores)))
        print("Scores:        ", ", ".join([str(s) for s in scores]))
        print("Win Rate:       %d/%d (%.2f)" % (wins.count(True), len(wins), win_rate))
        print("Record:        ", ", ".join([["Loss","Win"][int(w)] for w in wins]))
    return games

if __name__ == "__main__":
    args = read_command(sys.argv[1:])
    run_games(**args)

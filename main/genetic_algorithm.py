# genetic_algorithm.py
# --------------------
# This file contains the core logic for a genetic algorithm.

import random
import pacman
import game
from text_display import NullGraphics
# How many individuals are in the population
POPULATION_SIZE = 100


# How many games to run for each individual to evaluate its fitness
NUM_FITNESS_GAMES = 5

# How many of the top individuals to keep for the next generation (elitism)
NUM_ELITES = 10

# The probability that a gene will be mutated
MUTATION_RATE = 0.15

# The magnitude of the mutation
MUTATION_AMOUNT = 0.3


def run_games_for_fitness(layout, pacman_agent, ghost_agent, num_games):
    """
    Runs a number of Pacman games and returns the average score and win rate.
    """
    scores = []
    wins = 0
    for _ in range(num_games):
        # Create a new game with the provided agents and layout
        # NullGraphics is used to speed up the simulation as we don't need to see it.
        
        # --- THIS IS THE FIX ---
        # ClassicGameRules is in the 'pacman' module, not the 'game' module.
        rules = pacman.ClassicGameRules()
        # --- END OF FIX ---
        
        this_game = rules.new_game(layout, pacman_agent, ghost_agent, NullGraphics())
        this_game.run()
        scores.append(this_game.state.get_score())
        if this_game.state.is_win():
            wins += 1
    
    avg_score = sum(scores) / len(scores)
    win_rate = wins / num_games
    return avg_score, win_rate


class Individual:
    """
    Represents a single individual in the population.
    The 'chromosome' is the set of weights for the evaluation function.
    """
    def __init__(self, weights):
        self.weights = weights
        self.fitness = 0.0
        self.win_rate = 0.0
        self.avg_score = 0.0  # Track raw score separately from fitness


def initialize_population(size):
    """
    Creates the initial population with random weights for the FINAL evaluation function.
    """
    population = []
    for _ in range(size):
        weights = {
            'score':    random.uniform(0.5, 2.0),   # Positive: Higher score is good
            'survival': random.uniform(1.0, 3.0),   # Positive: survival_feature is negative, so positive weight = penalty
            'progress': random.uniform(0.5, 2.0),   # Positive: Higher progress feature (fewer/closer food) is good
            'scared_ghost': random.uniform(2.0, 5.0), # Positive: Higher is better (incentivize eating ghosts)
        }
        population.append(Individual(weights))
    return population


def select_parents(population):
    """
    Selects two parents from the population using tournament selection.
    """
    tournament_size = 5
    
    def tournament():
        # Select a random subset of the population for the tournament
        competitors = random.sample(population, tournament_size)
        # The winner is the one with the highest fitness
        winner = max(competitors, key=lambda ind: ind.fitness)
        return winner

    parent1 = tournament()
    parent2 = tournament()
    return parent1, parent2


def crossover(parent1, parent2):
    """
    Performs crossover by averaging the weights of the two parents.
    """
    child_weights = {}
    for key in parent1.weights:
        child_weights[key] = (parent1.weights[key] + parent2.weights[key]) / 2.0
    return Individual(child_weights)


def mutate(individual):
    """
    Mutates the weights of an individual based on the mutation rate.
    """
    mutated_weights = individual.weights.copy()
    for key in mutated_weights:
        if random.random() < MUTATION_RATE:
            # Add or subtract a small random value
            mutation = random.uniform(-MUTATION_AMOUNT, MUTATION_AMOUNT)
            mutated_weights[key] += mutation
    return Individual(mutated_weights)


def save_statistics(best_score_history, avg_score_history, win_rate_history):
    """
    Saves the fitness statistics to a file and attempts to plot them.
    NOTE: We plot the SCORE, not the internal fitness value.
    """
    try:
        import matplotlib.pyplot as plt
        
        generations = range(1, len(best_score_history) + 1)
        
        # Plot Fitness (Score)
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(generations, best_score_history, 'g-', label='Best Score')
        plt.plot(generations, avg_score_history, 'b--', label='Average Score')
        plt.title('Score Progress')
        plt.xlabel('Generation')
        plt.ylabel('Game Score')
        plt.legend()
        plt.grid(True)
        
        # Plot Win Rate
        plt.subplot(1, 2, 2)
        plt.plot(generations, win_rate_history, 'r-', label='Max Win Rate')
        plt.title('Win Rate Progress')
        plt.xlabel('Generation')
        plt.ylabel('Win Rate (0-1)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('ga_progress.png')
        print("[GA] Progress graph saved to 'ga_progress.png'")
        
    except ImportError:
        print("[GA] matplotlib not found. Skipping graph generation.")
    except Exception as e:
        print(f"[GA] Could not save graph: {e}")

    # Save raw data to text file
    with open('ga_stats.txt', 'w') as f:
        f.write("Generation,Best Score,Average Score,Max Win Rate\n")
        for gen, (best, avg, win) in enumerate(zip(best_score_history, avg_score_history, win_rate_history), 1):
            f.write(f"{gen},{best:.2f},{avg:.2f},{win:.2f}\n")
    print("[GA] Statistics saved to 'ga_stats.txt'")


def run_genetic_algorithm(num_generations, layout_name):
    """
    The main function to run the genetic algorithm.
    """
    from ghost_agents import DirectionalGhost, ChaseGhost
    from pacman_agents import GreedyAgent, genetic_evaluation_function
    from layout import get_layout
    
    layout = get_layout(layout_name)
    
    # You can remove the debug print/assert lines now if you wish
    assert layout is not None, f"ERROR: Layout '{layout_name}' not found."

    # --- THIS IS THE FIX ---
    # The game expects a LIST of ghost agents, not just one.
    # We create one ghost agent for each ghost specified in the layout file.
    num_ghosts = layout.get_num_ghosts()
    ghosts = [ChaseGhost(i) for i in range(1, num_ghosts + 1)]
    # --- END OF FIX ---

    print(f"[GA] Initializing population of size {POPULATION_SIZE}...")
    population = initialize_population(POPULATION_SIZE)

    best_score_history = []
    avg_score_history = []
    win_rate_history = []

    for gen in range(num_generations):
        print(f"\n--- Generation {gen + 1} of {num_generations} ---")

        print("[GA] Evaluating fitness...")
        generation_score_sum = 0
        max_win_rate = 0.0
        
        for i, individual in enumerate(population):
            pacman_agent = GreedyAgent(eval_fn=lambda state: genetic_evaluation_function(state, individual.weights))
            
            # Pass the LIST of ghosts to the fitness function
            avg_score, win_rate = run_games_for_fitness(layout, pacman_agent, ghosts, NUM_FITNESS_GAMES)
            
            individual.avg_score = avg_score
            individual.win_rate = win_rate
            
            # --- FITNESS FUNCTION MODIFICATION ---
            # Prioritize WINNING. A win is worth 5000 points.
            # This ensures that a low-scoring win is better than a high-scoring loss.
            individual.fitness = avg_score + (win_rate * 5000)
            
            generation_score_sum += avg_score
            if win_rate > max_win_rate:
                max_win_rate = win_rate
            # print(f"  - Individual {i}: Score={avg_score:.2f}, WinRate={win_rate:.2f}, Fitness={individual.fitness:.2f}")

        # Sort by FITNESS (which includes the win bonus)
        population.sort(key=lambda ind: ind.fitness, reverse=True)
        
        # Track statistics based on SCORE (for user visibility)
        best_individual = population[0]
        best_score = best_individual.avg_score
        avg_score = generation_score_sum / POPULATION_SIZE
        
        best_score_history.append(best_score)
        avg_score_history.append(avg_score)
        win_rate_history.append(max_win_rate)
        
        print(f"[GA] Best Score: {best_score:.2f}, Avg Score: {avg_score:.2f}, Max Win Rate: {max_win_rate:.2f}")

        next_generation = []
        elites = population[:NUM_ELITES]
        next_generation.extend(elites)
        
        while len(next_generation) < POPULATION_SIZE:
            parent1, parent2 = select_parents(population)
            child = crossover(parent1, parent2)
            mutated_child = mutate(child)
            next_generation.append(mutated_child)
        
        population = next_generation

    best_individual = max(population, key=lambda ind: ind.fitness)
    
    # Save statistics and graph
    save_statistics(best_score_history, avg_score_history, win_rate_history)
    
    print("\n[GA] Genetic Algorithm finished.")
    print("-" * 40)
    print(f"[GA] Best Score: {best_individual.avg_score:.2f}")
    print(f"[GA] Best Win Rate: {best_individual.win_rate:.2f}")
    print("[GA] COPY THE WEIGHTS BELOW:")
    # Print the weights in a copy-paste friendly format
    print(f"best_weights = {best_individual.weights}")
    print("-" * 40)
    
    return best_individual.weights

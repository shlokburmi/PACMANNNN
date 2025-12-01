import subprocess
import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

class AlgorithmComparison:
    def __init__(self):
        self.results = defaultdict(lambda: defaultdict(dict))
        self.mazes = ['tiny_maze', 'small_maze', 'medium_maze', 'tricky_search']
        self.algorithms = [
            ('DFS', 'SearchAgent', {'fn': 'depthFirstSearch'}),
            ('BFS', 'SearchAgent', {'fn': 'breadthFirstSearch'}),
            ('UCS', 'SearchAgent', {'fn': 'uniformCostSearch'}),
            ('A*', 'SearchAgent', {'fn': 'astar_search', 'heuristic': 'manhattan_heuristic'}),
            ('SMHA*', 'SMHAFoodSearchAgent', {}),
            ('Ultimate', 'UltimateSearchAgent', {}),
        ]

    def run_test(self, maze, algo_name, agent_class, params):
        """Run single test and capture output (robust to encoding / special chars)"""
        cmd = f'python pacman.py -l {maze} -p {agent_class} -n 1 -q'
        if params:
            param_str = ','.join(f'{k}={v}' for k, v in params.items())
            cmd += f' -a {param_str}'

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, errors='ignore', timeout=120)
            output = (proc.stdout or '') + (proc.stderr or '')
            # Save debug for Ultimate if needed
            if algo_name == 'Ultimate':
                with open(f'debug_Ultimate_{maze}.txt', 'w', encoding='utf-8') as f:
                    f.write(output)
            return self._parse_output(output, algo_name)
        except subprocess.TimeoutExpired:
            return {'nodes_expanded': 0, 'path_cost': 0, 'score': 0, 'pruning_efficiency': 0}
        except Exception:
            return {'nodes_expanded': 0, 'path_cost': 0, 'score': 0, 'pruning_efficiency': 0}

    def _parse_output(self, text, algo_name):
        """Extract metrics. For Ultimate, prefer the last RESULTS block (handles multiple goal prints)."""
        result = {'nodes_expanded': 0, 'path_cost': 0, 'score': 0, 'pruning_efficiency': 0}

        # Score extraction (Pacman summary)
        score_match = re.search(r'Pacman emerges victorious! Score:\s*(\d+)', text)
        if not score_match:
            score_match = re.search(r'Score:\s*(\d+)', text)
        if score_match:
            try:
                result['score'] = int(score_match.group(1))
            except:
                result['score'] = 0

        if algo_name == 'Ultimate':
            # findall and use the last occurrence (most recent/result of run)
            nodes_all = re.findall(r'Nodes Expanded:\s*(\d+)', text)
            costs_all = re.findall(r'Path Cost:\s*(\d+)', text)
            prunes_all = re.findall(r'Pruning Efficiency:\s*([\d.]+)%', text)

            if nodes_all:
                try:
                    result['nodes_expanded'] = int(nodes_all[-1])
                except:
                    result['nodes_expanded'] = 0

            if costs_all:
                try:
                    result['path_cost'] = int(costs_all[-1])
                except:
                    result['path_cost'] = 0

            if prunes_all:
                try:
                    result['pruning_efficiency'] = float(prunes_all[-1])
                except:
                    result['pruning_efficiency'] = 0.0

            # fallback: sometimes "Nodes Pruned" appears and pruning % absent
            if result['pruning_efficiency'] == 0.0:
                pruned_all = re.findall(r'Nodes Pruned:\s*(\d+)', text)
                if pruned_all and nodes_all:
                    try:
                        last_pruned = int(pruned_all[-1])
                        last_nodes = int(nodes_all[-1])
                        total = last_nodes + last_pruned
                        if total > 0:
                            result['pruning_efficiency'] = round(last_pruned / total * 100, 1)
                    except:
                        pass

            return result

        # Generic agents: single-line patterns
        nodes_match = re.search(r'Nodes Expanded[:\s]+(\d+)', text)
        if nodes_match:
            result['nodes_expanded'] = int(nodes_match.group(1))
        cost_match = re.search(r'Path Cost[:\s]+(\d+)', text)
        if cost_match:
            result['path_cost'] = int(cost_match.group(1))

        return result

    def run_all_tests(self):
        print("\n" + "="*80)
        print("ALGORITHM COMPARISON ACROSS ALL MAZES")
        print("="*80 + "\n")

        total = len(self.mazes) * len(self.algorithms)
        i = 0
        for maze in self.mazes:
            print(f"\n{'='*60}")
            print(f"Testing Maze: {maze.upper()}")
            print(f"{'='*60}")
            for algo_name, agent_class, params in self.algorithms:
                i += 1
                print(f"[{i}/{total}] Testing {algo_name:10} on {maze:15} ... ", end='', flush=True)
                metrics = self.run_test(maze, algo_name, agent_class, params)
                self.results[maze][algo_name] = metrics
                if metrics['nodes_expanded'] > 0:
                    print(f"[OK] Nodes: {int(metrics['nodes_expanded']):4} | Cost: {int(metrics['path_cost']):3} | Score: {int(metrics['score']):4} | Prune: {metrics['pruning_efficiency']:.1f}%")
                else:
                    print("[FAIL] No data extracted")

        self.print_summary()
        self.generate_graphs()

    def print_summary(self):
        print("\n" + "="*80)
        print("DETAILED RESULTS SUMMARY")
        print("="*80 + "\n")
        for maze in self.mazes:
            print(f"\n{maze.upper()}")
            print("-" * 120)
            print(f"{'Algorithm':<12} {'Nodes Expanded':<18} {'Path Cost':<12} {'Score':<10} {'Pruning %':<12}")
            print("-" * 120)
            for algo_name, _, _ in self.algorithms:
                m = self.results[maze].get(algo_name, {})
                nodes = int(m.get('nodes_expanded', 0))
                cost = int(m.get('path_cost', 0))
                score = int(m.get('score', 0))
                prune = float(m.get('pruning_efficiency', 0.0))
                status = "[OK]" if nodes > 0 else "[XX]"
                print(f"{status} {algo_name:<10} {nodes:<18} {cost:<12} {score:<10} {prune:<12.1f}")
            print()

    def generate_graphs(self):
        # minimal graphs to match previous behaviour
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        metrics = ['nodes_expanded', 'path_cost', 'score', 'pruning_efficiency']
        titles = ['Nodes Expanded', 'Path Cost', 'Score', 'Pruning Efficiency (%)']

        for ax, metric, title in zip(axes.flat, metrics, titles):
            x = np.arange(len(self.mazes))
            width = 0.13
            for idx, (algo_name, _, _) in enumerate(self.algorithms):
                vals = [self.results[m].get(algo_name, {}).get(metric, 0) for m in self.mazes]
                offset = (idx - len(self.algorithms)/2) * width + width/2
                ax.bar(x + offset, vals, width, label=algo_name)
            ax.set_title(title)
            ax.set_xticks(x)
            ax.set_xticklabels([m.replace('_',' ').title() for m in self.mazes], rotation=20)
            ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('algorithm_comparison.png', dpi=300, bbox_inches='tight')
        print("\n[OK] Graph saved as: algorithm_comparison.png")

def main():
    comparator = AlgorithmComparison()
    comparator.run_all_tests()

if __name__ == '__main__':
    main()
import mlrose_hiive as mlrose
import pandas as pd
import matplotlib.pyplot as plt
import time
import warnings
from mlrose_hiive import MaxKColorGenerator
warnings.filterwarnings("ignore")


def plot_curves(iterations, rhc_fitness_curve, sa_fitness_curve, ga_fitness_curve, mimic_fitness_curve, problem_name):
    plt.figure()
    plt.plot(iterations, [i[0] for i in rhc_fitness_curve], label='RHC', color='green')
    plt.plot(iterations, [i[0] for i in sa_fitness_curve], label='SA', color='red')
    plt.plot(iterations, [i[0] for i in ga_fitness_curve], label='GA', color='blue')
    plt.plot(iterations, [i[0] for i in mimic_fitness_curve], label='MIMIC', color='orange')
    plt.legend(loc="best")
    plt.xlabel("Iterations")
    plt.ylabel("Fitness")
    plt.savefig(f"results/{problem_name}_fitness_graph.png")
    plt.close()


def run_random_optimizer(fitness, problem_name, skip_mimic, skip_ga):
    print(f"Running Experiments for {problem_name} Optimization Problem")
    print()

    problem = mlrose.DiscreteOpt(length=100, fitness_fn=fitness, maximize=True)

    # K color problem generator, comment out for all 3 problem run. This is the "Complex" network in the report
    # problem = MaxKColorGenerator().generate(seed=random_seed, number_of_nodes=100, max_connections_per_node=4, max_colors=5)
    
    random_seed = 42
    max_attempts = 100
    # tested different iterations across experiments by varying this value
    max_iters = 100

    print("Random Hill Climb")
    start_time = time.time()
    rhc_best_state, rhc_best_fitness, rhc_fitness_curve = mlrose.random_hill_climb(problem,max_attempts=max_attempts,max_iters=max_iters,curve=True,random_state=random_seed,restarts=100)
    rhc_time = time.time() - start_time

    print("Simulated Annealing")
    start_time = time.time()
    sa_best_state, sa_best_fitness, sa_fitness_curve = mlrose.simulated_annealing(problem,max_attempts=max_attempts,max_iters=max_iters,curve=True,random_state=random_seed,schedule=mlrose.GeomDecay())
    sa_time = time.time() - start_time

    if not skip_ga:
        print("Genetic Algorithm")
        start_time = time.time()
        ga_best_state, ga_best_fitness, ga_fitness_curve = mlrose.genetic_alg(problem,max_attempts=max_attempts,max_iters=max_iters,curve=True,random_state=random_seed,pop_size=200,mutation_prob=0.2)
        ga_time = time.time() - start_time

    if not skip_mimic:
        print("MIMIC")
        start_time = time.time()
        mimic_best_state, mimic_best_fitness, mimic_fitness_curve = mlrose.mimic(problem,max_attempts=max_attempts,max_iters=max_iters,curve=True,random_state=random_seed,keep_pct=0.25)
        mimic_time = time.time() - start_time

    iterations = range(1, max_iters+1)
    if skip_mimic and not skip_ga:
        plt.figure()
        plt.plot(iterations, [i[0] for i in rhc_fitness_curve], label='RHC', color='green')
        plt.plot(iterations, [i[0] for i in sa_fitness_curve], label='SA', color='red')
        plt.plot(iterations, [i[0] for i in ga_fitness_curve], label='GA', color='blue')
        plt.legend(loc="best")
        plt.xlabel("Iterations")
        plt.ylabel("Fitness")
        plt.savefig(f"results/{problem_name}_fitness_sans_mimic.png")
        plt.close()

        data = [('RHC', round(rhc_time, 5)),
                ('SA', round(sa_time, 5)),
                ('GA', round(ga_time, 5))]

        df = pd.DataFrame(data, columns=['Algorithm', 'Time'])
        print(df)
    elif skip_ga and not skip_mimic:
        plt.figure()
        plt.plot(iterations, [i[0] for i in rhc_fitness_curve], label='RHC', color='green')
        plt.plot(iterations, [i[0] for i in sa_fitness_curve], label='SA', color='red')
        plt.plot(iterations, [i[0] for i in mimic_fitness_curve], label='GA', color='blue')
        plt.legend(loc="best")
        plt.xlabel("Iterations")
        plt.ylabel("Fitness")
        plt.savefig(f"results/{problem_name}_fitness_sans_ga.png")
        plt.close()

        data = [('RHC', round(rhc_time, 5)),
                ('SA', round(sa_time, 5)),
                ('MIMIC', round(mimic_time, 5))]

        df = pd.DataFrame(data, columns=['Algorithm', 'Time'])
        print(df)
    elif skip_mimic and skip_ga:
        plt.figure()
        plt.plot(iterations, [i[0] for i in rhc_fitness_curve], label='RHC', color='green')
        plt.plot(iterations, [i[0] for i in sa_fitness_curve], label='SA', color='red')
        plt.legend(loc="best")
        plt.xlabel("Iterations")
        plt.ylabel("Fitness")
        plt.savefig(f"results/{problem_name}_fitness_sans_mimic_ga.png")
        plt.close()

        data = [('RHC', round(rhc_time, 5)),
                ('SA', round(sa_time, 5))]

        df = pd.DataFrame(data, columns=['Algorithm', 'Time'])
        print(df)
    else:
        plot_curves(iterations, rhc_fitness_curve, sa_fitness_curve, ga_fitness_curve, mimic_fitness_curve, problem_name)

        data = [('RHC', round(rhc_time, 5)),
                ('SA', round(sa_time, 5)),
                ('GA', round(ga_time, 5)),
                ('MIMIC', round(mimic_time, 5))]

        df = pd.DataFrame(data, columns=['Algorithm', 'Time'])
        print(df)


fitness = mlrose.FourPeaks(t_pct=0.1)
run_random_optimizer(fitness, "Four Peaks", False, False)

edges = [(0, 1), (0, 2), (0, 4), (1, 3), (2, 0), (2, 3), (3, 4)]
fitness = mlrose.MaxKColor(edges)
run_random_optimizer(fitness, "Max K Color", False, False)

fitness = mlrose.OneMax()
run_random_optimizer(fitness, "OneMax", False, False)


import mlrose_hiive
import numpy as np
import matplotlib.pyplot as plt
import time
from statistics import mean

np.random.seed(625)

# Define alternative N-Cont_Peaks fitness function for maximization problem
def queens_max(state):

    # Initialize counter
    fitness_cnt = 0
    # For all pairs of queens
    for i in range(len(state) - 1):
        for j in range(i + 1, len(state)):
            if (state[j] != state[i]) and (state[j] != state[i] + (j - i)) and (state[j] != state[i] - (j - i)):
                fitness_cnt += 1

    return fitness_cnt

def plot_graph(curve, label,savename, type):
    fitness = curve[:,0]
    iterations = curve[:,1]

    plt.plot(iterations,fitness)
    plt.legend([label])
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.title(f'Optimized Hyperparams for Fitness vs Iterations - {label} - {type}', fontsize=10)
    plt.savefig(savename)
    plt.show()

def plot_curve_avg(curve, label,savename, type):
    fitness_mean = np.mean(curve, axis=0)
    iterations = np.arange(0, len(fitness_mean), dtype=int)  # Start from 1 to get integer iterations

    plt.plot(iterations,fitness_mean)

    fitness_std = np.std(curve, axis=0)
    plt.fill_between(iterations, fitness_mean - fitness_std, fitness_mean + fitness_std, alpha=0.4)

    plt.legend([label])
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.title(f'Optimized Hyperparams for Fitness vs Iterations - {label} - {type}', fontsize=10)
    plt.savefig(savename)
    plt.show()


def sa():
    fitness_cust = mlrose_hiive.ContinuousPeaks()
    size_bits = [10,25,50,100]
    max_attempts = [500, 1000, 5000, 10000]
    initial_temp = [1,10,50,100]
    seeds = [620, 625, 630, 640]

    best_fitness_sa_time = []
    best_time = float('inf')

    for n in size_bits:
        problem = mlrose_hiive.DiscreteOpt(length=n, fitness_fn=fitness_cust, maximize=True, max_val=2)

        for attempt in max_attempts:
            for t in initial_temp:
                for schedule_name, schedule_class in [('ArithDecay', mlrose_hiive.ArithDecay),
                                                      ('GeomDecay', mlrose_hiive.GeomDecay),
                                                      ('ExpDecay', mlrose_hiive.ExpDecay)]:

                    all_fitness_scores = []
                    all_fitness_curves = []
                    all_fitness_fevals = []

                    if t > 10 and schedule_name=='ArithDecay':
                        print("Arith Decay and t is ", t)
                        continue

                    for seed in seeds:
                        np.random.seed(seed)
                        start = time.time()
                        best_state, best_fitness, fitness_curve = mlrose_hiive.simulated_annealing(
                            problem=problem,
                            schedule=schedule_class(init_temp=t),
                            max_attempts=attempt,
                            curve=True)
                        end = time.time()
                        diff = end - start

                        all_fitness_scores.append(best_fitness)
                        all_fitness_curves.append(fitness_curve[:, 0])  # Only take the first column
                        all_fitness_fevals.append(fitness_curve[:, 1])  # Only take the first column

                    avg_fitness_score = mean(all_fitness_scores)
                    max_len = max(len(fc) for fc in all_fitness_curves)
                    padded_curves = [pad_fitness_curve(fc, max_len) for fc in all_fitness_curves]
                    padded_fevals = [pad_fitness_curve(fc, max_len) for fc in all_fitness_fevals]

                    if len(best_fitness_sa_time) == 0 or avg_fitness_score > test[0]:
                        test = [avg_fitness_score, schedule_name, attempt, t, diff, padded_fevals, padded_curves]
                        best_time = diff
                    elif avg_fitness_score == test[0] and diff < best_time:
                        test = [avg_fitness_score, schedule_name, attempt, t, diff, padded_fevals, padded_curves]
                        best_time = diff

                    print("Finished a Schedule")

        print(f"Finished n={n}")
        best_fitness_sa_time.append(test)


    counter = 0
    print("===")

    for result in best_fitness_sa_time:
        print(f"Best Fitness Score for N = {size_bits[counter]} - SA")
        print( f"Params = Fitness: {result[0]}, Schedule: {result[1]}, Attempt: {result[2]}, Temperature: {result[3]}, Time: {result[4]}, FeVals Avg: {mean(np.mean(result[5], axis=0))}")

        fitness_curves = result[-1]

        plot_curve_avg(fitness_curves, f"{size_bits[counter]} Cont Peaks", f"{size_bits[counter]}_Cont_Peaks_Fit_Iter_SA_new", "SA")
        counter += 1

def pad_fitness_curve(curve, length):
    """Pad the fitness curve to the specified length by extending the last value."""
    return np.pad(curve, (0, length - len(curve)), 'edge')


def rhc():
    fitness_cont_peaks = mlrose_hiive.ContinuousPeaks(t_pct=0.1)
    size_bits = [10,25,50,100]
    max_iterations = [1000,5000,10000]
    max_attempts = [5,10,20,30,50,100,500,1000]
    restarts = [0,1,5,10,20]
    best_time = float('inf')


    best_fitness_score = []
    seeds = [620, 625, 630, 640]

    best_fitness_rhc_time = []
    best_time = float('inf')
    for n in size_bits:
        test = []
        best_fitness_rhc = []
        problem = mlrose_hiive.DiscreteOpt(length=n, fitness_fn=fitness_cont_peaks, maximize=True, max_val=2)
        for iter in max_iterations:
            for attempt in max_attempts:
                for restart in restarts:
                    all_fitness_scores = []
                    all_fitness_curves = []
                    all_fitness_fevals = []

                    for seed in seeds:
                        np.random.seed(seed)
                        start = time.time()
                        best_state, best_fitness, fitness_curve = mlrose_hiive.random_hill_climb(
                            problem=problem, curve=True, max_attempts=attempt, max_iters=iter, restarts=restart)
                        end = time.time()
                        diff = end - start

                        all_fitness_scores.append(best_fitness)
                        all_fitness_curves.append(fitness_curve[:, 0])  # Only take the first column
                        all_fitness_fevals.append(fitness_curve[:, 1])  # Only take the first column

                    avg_fitness_score = mean(all_fitness_scores)

                    max_len = max(len(fc) for fc in all_fitness_curves)
                    padded_curves = [pad_fitness_curve(fc, max_len) for fc in all_fitness_curves]
                    padded_fevals = [pad_fitness_curve(fc, max_len) for fc in all_fitness_fevals]

                    if len(test) == 0 or avg_fitness_score > test[0]:
                        test = [avg_fitness_score, iter, restart, attempt, diff, padded_fevals, padded_curves]
                        best_time = diff
                    elif avg_fitness_score == test[0] and diff < best_time:
                        test = [avg_fitness_score, iter, restart, attempt, diff, padded_fevals, padded_curves]
                        best_time = diff

                    best_fitness_rhc.append([best_fitness, iter, attempt, restart])

        # print(f"The best param with time is {test}, n={n}")
        best_fitness_rhc_time.append(test)

    counter = 0
    print("===")

    for result in best_fitness_rhc_time:
        print(f"Best Fitness Score for N = {size_bits[counter]} - RHC")
        print(
            f"Params = Fitness: {result[0]}, Iteration: {result[1]}, Restart: {result[2]}, Attempt: {result[3]}, Diff: {result[4]}, FeVals Avg: {mean(np.mean(result[5], axis=0))}")

        fitness_curves = result[-1]

        plot_curve_avg(fitness_curves, f"{size_bits[counter]} Cont Peaks",
                       f"{size_bits[counter]}_Cont_Peaks_Fit_Iter_RHC_new", "RHC")
        counter += 1

def ga():
    size_bits = [10,25,50,100]
    population = [100, 200, 300, 500]
    mutation_prob = [0.1, 0.2, 0.3, 0.4]
    max_attempts = [5, 10, 20, 30]
    best_fitness_ga_time = []
    best_time = float('inf')
    seeds = [620, 625, 630, 640]

    fitness_cont_peaks = mlrose_hiive.ContinuousPeaks()

    for n in size_bits:
        test = []
        problem = mlrose_hiive.DiscreteOpt(length=n, fitness_fn=fitness_cont_peaks, maximize=True, max_val=2)
        for pop in population:
            for attempt in max_attempts:
                for mut in mutation_prob:
                    all_fitness_scores = []
                    all_fitness_curves = []
                    all_fitness_fevals = []

                    for seed in seeds:
                        np.random.seed(seed)
                        start = time.time()
                        best_state, best_fitness, fitness_curve = mlrose_hiive.genetic_alg(problem=problem,
                                                                                           max_attempts=attempt,
                                                                                           mutation_prob=mut,
                                                                                           pop_size=pop, curve=True)

                        end = time.time()
                        diff = end - start

                        all_fitness_scores.append(best_fitness)
                        all_fitness_curves.append(fitness_curve[:, 0])  # Only take the first column
                        all_fitness_fevals.append(fitness_curve[:, 1])  # Only take the first column

                    print("Finished a Seed")
                    avg_fitness_score = mean(all_fitness_scores)

                    max_len = max(len(fc) for fc in all_fitness_curves)
                    padded_curves = [pad_fitness_curve(fc, max_len) for fc in all_fitness_curves]
                    padded_fevals = [pad_fitness_curve(fc, max_len) for fc in all_fitness_fevals]

                    if len(test) == 0 or avg_fitness_score > test[0]:
                        test = [avg_fitness_score, mut, attempt, pop, diff, padded_fevals, padded_curves]
                        best_time = diff
                    elif avg_fitness_score == test[0] and diff < best_time:
                        test = [avg_fitness_score, mut, attempt, pop, diff, padded_fevals, padded_curves]
                        best_time = diff

        # print(f"The best param with time is {test}, n={n}")
        best_fitness_ga_time.append(test)
        print(f"Finished N={n}")

    counter = 0
    print("===")

    for result in best_fitness_ga_time:
        print(f"Best Fitness Score for N = {size_bits[counter]} - GA")
        print(
            f"Params = Fitness: {result[0]}, Mut: {result[1]}, Attempt: {result[2]}, Pop: {result[3]}, Time: {result[4]}, FeVals Avg: {mean(np.mean(result[5], axis=0))}")

        fitness_curves = result[-1]

        plot_curve_avg(fitness_curves, f"{size_bits[counter]} Cont Peaks",
                       f"{size_bits[counter]}_Cont_Peaks_Fit_Iter_GA_new",
                       "GA")
        counter += 1

def rhc_param_tune():
    fitness_cust = mlrose_hiive.ContinuousPeaks()
    bit_problem_size = 50
    max_attempts = [5,10,20,30,50,100,500,1000]
    restarts = [0, 1, 5, 10, 20]

    plt.figure()
    # first tune number of iterations
    for attempt in max_attempts:
        problem = mlrose_hiive.DiscreteOpt(length=bit_problem_size, fitness_fn=fitness_cust, maximize=True, max_val=2)
        best_state, best_fitness, fitness_curve = mlrose_hiive.random_hill_climb(problem=problem, curve=True, max_attempts=attempt)
        first_col = fitness_curve[:,0]
        plt.plot(first_col, label=f"max_attempts={attempt}")

    plt.title(f'RHC - {bit_problem_size} Cont_Peaks - Modifying Max Attempts', fontsize=10)
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Fitness Score')
    plt.savefig('Cont_Peaks_50_Max_Attempt_Tune_RHC.png')
    plt.show()
    plt.close()

    plt.figure()
    for restart in restarts:
        problem = mlrose_hiive.DiscreteOpt(length=bit_problem_size, fitness_fn=fitness_cust, maximize=True, max_val=2)
        best_state, best_fitness, fitness_curve = mlrose_hiive.random_hill_climb(problem=problem, curve=True, restarts=restart)
        first_col = fitness_curve[:,0]
        plt.plot(first_col, label=f"restart={restart}")

    plt.title(f'RHC - {bit_problem_size} Cont_Peaks - Modifying Restarts', fontsize=10)
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Fitness Score')
    plt.savefig('Cont_Peaks_50_Restarts_Tune_RHC.png')
    plt.show()

def sa_param_tune():
    fitness_cust = mlrose_hiive.ContinuousPeaks()
    bit_problem_size = 50
    max_attempts = [5, 10, 20, 30, 50]
    initial_temp = [1, 10, 100]

    problem = mlrose_hiive.DiscreteOpt(length=bit_problem_size, fitness_fn=fitness_cust, maximize=True,
                                       max_val=2)
    plt.figure()
    best_state, best_fitness, fitness_curve = mlrose_hiive.simulated_annealing(problem=problem,
                                                                                   schedule=mlrose_hiive.ArithDecay(),
                                                                                   curve=True)
    first_col = fitness_curve[:, 0]
    plt.plot(first_col, label=f"Schedule=ArithDecay")
    best_state, best_fitness, fitness_curve = mlrose_hiive.simulated_annealing(problem=problem,
                                                                                   schedule=mlrose_hiive.ExpDecay(),
                                                                                   curve=True)
    first_col = fitness_curve[:, 0]
    plt.plot(first_col, label=f"Schedule=ExpDecay")
    best_state, best_fitness, fitness_curve = mlrose_hiive.simulated_annealing(problem=problem,
                                                                                   schedule=mlrose_hiive.GeomDecay(),
                                                                                   curve=True)
    first_col = fitness_curve[:, 0]
    plt.plot(first_col, label=f"Schedule=ExpDecay")

    plt.title(f'SA - {bit_problem_size} Cont_Peaks - Modifying Decay Schedules', fontsize=10)
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Fitness Score')
    plt.savefig('Cont_Peaks_50_Schedule_Tune_SA.png')
    plt.show()
    plt.close()
    plt.figure()

    for attempt in max_attempts:
        problem = mlrose_hiive.DiscreteOpt(length=bit_problem_size, fitness_fn=fitness_cust, maximize=True, max_val=2)
        best_state, best_fitness, fitness_curve = mlrose_hiive.simulated_annealing(problem=problem,
                                                                                   schedule=mlrose_hiive.GeomDecay(),
                                                                                   max_attempts=attempt,
                                                                                   curve=True)
        first_col = fitness_curve[:,0]
        plt.plot(first_col, label=f"max_attempt={attempt}")

    plt.title(f'SA - {bit_problem_size} Cont_Peaks - Modifying Restarts', fontsize=10)
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Fitness Score')
    plt.savefig('Cont_Peaks_50_Restarts_Tune_SA.png')
    plt.show()

    for t in initial_temp:
        problem = mlrose_hiive.DiscreteOpt(length=bit_problem_size, fitness_fn=fitness_cust, maximize=True, max_val=2)
        best_state, best_fitness, fitness_curve = mlrose_hiive.simulated_annealing(problem=problem,
                                                                                   schedule=mlrose_hiive.GeomDecay(init_temp=t),
                                                                                   curve=True)
        first_col = fitness_curve[:,0]
        plt.plot(first_col, label=f"init_temp={t}")

    plt.title(f'SA - {bit_problem_size} Cont_Peaks - Modifying Initial Temp', fontsize=10)
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Fitness Score')
    plt.savefig('Cont_Peaks_50_Init_T_Tune_SA.png')
    plt.show()

def ga_param_tune():
    fitness_cust = mlrose_hiive.ContinuousPeaks()
    bit_problem_size = 50
    population = [100, 200, 300, 500]
    mutation_prob = [0.1, 0.2, 0.3, 0.4]
    max_attempts = [5, 10, 20, 30]

    problem = mlrose_hiive.DiscreteOpt(length=bit_problem_size, fitness_fn=fitness_cust, maximize=True, max_val=2)

    plt.figure()
    # first tune number of iterations
    for attempt in max_attempts:
        best_state, best_fitness, fitness_curve = mlrose_hiive.genetic_alg(problem=problem, curve=True, max_attempts=attempt)
        first_col = fitness_curve[:,0]
        plt.plot(first_col, label=f"max_attempts={attempt}")

    plt.title(f'GA - {bit_problem_size} Cont_Peaks - Modifying Max Attempts', fontsize=10)
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Fitness Score')
    plt.savefig('Cont_Peaks_50_Max_Attempt_Tune_GA.png')
    plt.show()
    plt.close()

    plt.figure()
    # first tune number of iterations
    for mut in mutation_prob:
        best_state, best_fitness, fitness_curve = mlrose_hiive.genetic_alg(problem=problem, curve=True, mutation_prob=mut)
        first_col = fitness_curve[:,0]
        plt.plot(first_col, label=f"mutation_prob={mut}")

    plt.title(f'GA - {bit_problem_size} Cont_Peaks - Modifying Mutation', fontsize=10)
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Fitness Score')
    plt.savefig('Cont_Peaks_50_Mutation_Prob_Tune_GA.png')
    plt.show()
    plt.close()

    plt.figure()
    # first tune number of iterations
    for pop in population:
        best_state, best_fitness, fitness_curve = mlrose_hiive.genetic_alg(problem=problem, curve=True, pop_size=pop)
        first_col = fitness_curve[:,0]
        plt.plot(first_col, label=f"pop_size={pop}")

    plt.title(f'GA - {bit_problem_size} Cont_Peaks - Modifying Population', fontsize=10)
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Fitness Score')
    plt.savefig('Cont_Peaks_50_Pop_Size_Tune_GA.png')
    plt.show()
    plt.close()

#DEPRECATED - Use the print statements to generate the results, found in result.txt
# def wall_clock_compare():
#     # RHC     Best Params = [Fitness : 189, Iterations : 5000, Max Attempts:   1000, Restarts :   10.]
#     bit_problem_size = 100
#     fitness_cust = mlrose_hiive.CustomFitness(queens_max)
#     problem = mlrose_hiive.DiscreteOpt(length=bit_problem_size, fitness_fn=fitness_cust, maximize=True, max_val=2)
#     start = time.time()
#     best_state, best_fitness, fitness_curve = mlrose_hiive.random_hill_climb(problem=problem, curve=True,
#                                                                              max_attempts=1000, max_iters=5000,
#                                                                              restarts=10)
#     end = time.time()
#     print(f'Time taken to simulate RHC with optimized Params for {bit_problem_size} Cont_Peaks: {end-start}, fitness score: {best_fitness}')
#
#     # SA  Best Params = [Fitness: 4938.0, Schedule: 'ExpDecay',  Attempt: 50, Initial_Temp: 1, tcp: 0.1]
#     start = time.time()
#     best_state, best_fitness, fitness_curve = mlrose_hiive.simulated_annealing(problem=problem, curve=True, schedule=mlrose_hiive.ExpDecay(init_temp=1), max_attempts=500)
#     end = time.time()
#     print(f'Time taken to simulate SA with optimized Params for {bit_problem_size} Cont_Peaks: {end-start}, fitness score: {best_fitness}')
#
#     # GA  Best Params = [Fitness: 189, Mutation :  0.4, Attempt : 3.00e+01, Population: 3.00e+02]
#     start = time.time()
#     best_state, best_fitness, fitness_curve = mlrose_hiive.genetic_alg(problem=problem, curve=True, pop_size=100, max_attempts=20, mutation_prob= 0.4)
#     end = time.time()
#     print(f'Time taken to simulate GA with optimized Params for {bit_problem_size} Cont_Peaks: {end-start}, fitness score: {best_fitness}')



def main():
    rhc()
    sa()
    ga()
    rhc_param_tune()
    sa_param_tune()
    ga_param_tune()
    # wall_clock_compare()

if __name__ == "__main__":
    main()


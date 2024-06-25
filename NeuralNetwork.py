import numpy as np
import mlrose_hiive
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import learning_curve

import time
from ucimlrepo import fetch_ucirepo

rice_cammeo_and_osmancik = fetch_ucirepo(id=545)
rice_data = rice_cammeo_and_osmancik.data

X = rice_data.features
y = rice_data.targets.Class

y[y == 'Cammeo'] = 1
y[y == 'Osmancik'] = 0
y = y.astype(int)

# Pre process data onto the same scale
X = preprocessing.scale(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=625)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=625)


schedules = [mlrose_hiive.GeomDecay(), mlrose_hiive.ExpDecay(), mlrose_hiive.ArithDecay()]

def plot_learning_curves(x_range, train_results, test_results, title, x_label,y_label, save_name, parameter_range):
    #Find mean
    test_results_mean = np.mean(test_results, axis=1)
    train_results_mean = np.mean(train_results, axis=1)

    plt.figure()
    #Plot MEAN
    plt.plot(parameter_range, test_results_mean)
    plt.plot(x_range, train_results_mean)

    #Plot STD
    test_results_std = np.std(test_results, axis=1)
    train_results_std = np.std(train_results, axis=1)
    plt.fill_between(parameter_range, test_results_mean - test_results_std, test_results_mean + test_results_std, alpha=0.4)
    plt.fill_between(parameter_range, train_results_mean - train_results_std, train_results_mean + train_results_std, alpha=0.4)

    plt.legend(['Test Results', 'Train Results'])
    plt.title(title, fontsize=10)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.ylim(0.7,1.1)
    plt.grid()
    plt.savefig(save_name)
    plt.show()
    return

def plot_loss_curve(model, xlabel, ylabel, title, label, savefig):
    plt.figure()
    plt.plot(model.fitness_curve[:, 0], label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.title(title, fontsize=10)
    plt.savefig(savefig)
    plt.show()
def plot_loss_curve_BP(model, xlabel, ylabel, title, label, savefig):
    plt.figure()
    plt.plot(-model.fitness_curve, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.title(title, fontsize=10)
    plt.savefig(savefig)
    plt.show()

# RHC

learning_rates = [0.01]
restarts = [2, 5, 10, 15]
populations = [100, 200, 300, 500]
nodes = [16]

new = True
time_rhc_current = 0
for learning_rate in learning_rates:
    for node in nodes:
        for restart in restarts:

            if new:
                best_nn_rhc = mlrose_hiive.NeuralNetwork(hidden_nodes=[node], activation='tanh',
                                                               algorithm='random_hill_climb',
                                                               restarts=restart,
                                                               max_iters=2000, bias=True, is_classifier=True,
                                                               learning_rate=learning_rate, early_stopping=True,
                                                               max_attempts=100, random_state=620, curve=True)
                start = time.time()
                best_nn_rhc.fit(X_train, y_train)
                end = time.time()
                time_rhc = end - start

                ytrain_pred = best_nn_rhc.predict(X_train)
                ytrain_accuracy_rhc = accuracy_score(y_train, ytrain_pred)

                ytest_pred = best_nn_rhc.predict(X_test)
                ytest_accuracy_rhc = accuracy_score(ytest_pred, y_test)

                optimized_rhc_NN_model = best_nn_rhc
                optimized_rhc_time = time_rhc
                optimized_node = node
                optimized_restart = restart
                optimized_lr = learning_rate

                best_accuracy_rhc = ytest_accuracy_rhc
                print(f"Learning Rate: {learning_rate}")
                print(f"Restarts: {restart}")
                print(f"Node: {node}")
                print(f"Time: {time_rhc}")
                print(f"Learning Rate: {learning_rate}")
                print(f"Accuracy Score: {best_accuracy_rhc}")

                new = False
            else:
                nn_model_rhc_instance = mlrose_hiive.NeuralNetwork(hidden_nodes=[node], activation='tanh',
                                                          algorithm='random_hill_climb',
                                                          max_iters=2000, bias=True, restarts=restart, is_classifier=True,
                                                          learning_rate=learning_rate, early_stopping=True,
                                                          max_attempts=100, random_state=620, curve=True)

                start = time.time()
                nn_model_rhc_instance.fit(X_train, y_train)
                end = time.time()
                time_rhc = end - start

                ytrain_pred = nn_model_rhc_instance.predict(X_train)
                ytrain_accuracy_rhc = accuracy_score(y_train, ytrain_pred)

                ytest_pred = nn_model_rhc_instance.predict(X_test)
                ytest_accuracy_rhc = accuracy_score(ytest_pred, y_test)
                print(f"Instance Test Results - {ytest_accuracy_rhc}")

                if ytest_accuracy_rhc > best_accuracy_rhc:
                    optimized_rhc_NN_model = nn_model_rhc_instance

                    optimized_rhc_NN_model = best_nn_rhc
                    optimized_rhc_time = time_rhc
                    optimized_node = node
                    optimized_restart = restart
                    optimized_lr = learning_rate

                    print(f"Learning Rate: {learning_rate}")
                    print(f"Restarts: {restart}")
                    print(f"Node: {node}")
                    print(f"Time: {time_rhc}")
                    best_accuracy_rhc = ytest_accuracy_rhc
                    print(f"Accuracy : {best_accuracy_rhc}")

            print("Completed One Iteration")

plot_loss_curve(model= optimized_rhc_NN_model,xlabel="Iterations", ylabel="Training Loss", title="Rice (NN) RHC - Training Loss vs. Iterations (Optimized Params)", savefig="nn_loss_curve_opt_RHC.png", label="RHC - Optimized Params")

y_test_pred_rhc = optimized_rhc_NN_model.predict(X_test)
print("-------")
print(f"RHC Complete!!! Optiomized Params and Scores below")
print(f"Time: {optimized_rhc_time}")
print(f"Node: {optimized_node}")
print(f"Learning rate: {optimized_lr}")
print(f"Restarts: {optimized_restart}")
print(f"Best Accuracy Score: {best_accuracy_rhc}")
print("-------")

train_sizes_abs, train_results, test_results = learning_curve(optimized_rhc_NN_model, X_train, y_train,
                                                              train_sizes=np.linspace(0.1, 1.0, 10), cv=5)

param_range = np.linspace(0.1, 1.0, 10) * 100
plot_learning_curves(x_range=param_range, train_results=train_results, test_results=test_results,
                     title="Rice - Neural Network (RHC) - Accuracy vs % Trained",
                     save_name="NN_rice_RHC_learning_curve_score.png", x_label="Percent Trained", y_label="Accuracy",
                     parameter_range=param_range)


# SA
learning_rates = [0.01]
restarts = [2, 5, 10, 15]
schedules = [mlrose_hiive.GeomDecay(), mlrose_hiive.ExpDecay(), mlrose_hiive.ArithDecay()]
nodes = [16]

new = True
time_sa_current = 0
for learning_rate in learning_rates:
    for node in nodes:
        for restart in restarts:
            for schedule in schedules:
                if new:
                    best_nn_sa = mlrose_hiive.NeuralNetwork(hidden_nodes=[node], activation='tanh',
                                                                   algorithm='simulated_annealing',
                                                                   restarts=restart,
                                                                   max_iters=2000, bias=True, is_classifier=True, schedule=schedule,
                                                                   learning_rate=learning_rate, early_stopping=True,
                                                                   max_attempts=100, random_state=620, curve=True)
                    start = time.time()
                    best_nn_sa.fit(X_train, y_train)
                    end = time.time()
                    time_sa = end - start

                    ytrain_pred = best_nn_sa.predict(X_train)
                    ytrain_accuracy_sa = accuracy_score(y_train, ytrain_pred)

                    ytest_pred = best_nn_sa.predict(X_test)
                    ytest_accuracy_sa = accuracy_score(ytest_pred, y_test)

                    optimized_sa_NN_model = best_nn_sa
                    optimized_sa_time = time_sa
                    optimized_node = node
                    optimized_restart = restart
                    optimized_lr = learning_rate

                    best_accuracy_sa = ytest_accuracy_sa
                    print(f"Learning Rate: {learning_rate}")
                    print(f"Restarts: {restart}")
                    print(f"Node: {node}")
                    print(f"Time: {time_sa}")
                    print(f"Learning Rate: {learning_rate}")
                    print(f"Accuracy Score: {best_accuracy_sa}")

                    new = False
                else:
                    nn_model_sa_instance = mlrose_hiive.NeuralNetwork(hidden_nodes=[node], activation='tanh',
                                                              algorithm='simulated_annealing',
                                                              max_iters=1000, bias=True, restarts=restart, is_classifier=True, schedule=schedule,
                                                              learning_rate=learning_rate, early_stopping=True,
                                                              max_attempts=100, random_state=620, curve=True)

                    start = time.time()
                    nn_model_sa_instance.fit(X_train, y_train)
                    end = time.time()
                    time_sa = end - start

                    ytrain_pred = nn_model_sa_instance.predict(X_train)
                    ytrain_accuracy_sa = accuracy_score(y_train, ytrain_pred)

                    ytest_pred = nn_model_sa_instance.predict(X_test)
                    ytest_accuracy_sa = accuracy_score(ytest_pred, y_test)
                    print(f"Instance Test Results - {ytest_accuracy_sa}")

                    if ytest_accuracy_sa > best_accuracy_sa:
                        optimized_sa_NN_model = nn_model_sa_instance

                        optimized_sa_NN_model = best_nn_sa
                        optimized_sa_time = time_sa
                        optimized_node = node
                        optimized_restart = restart
                        optimized_lr = learning_rate

                        print(f"Learning Rate: {learning_rate}")
                        print(f"Restarts: {restart}")
                        print(f"Node: {node}")
                        print(f"Time: {time_sa}")
                        best_accuracy_sa = ytest_accuracy_sa
                        print(f"Accuracy : {best_accuracy_sa}")

            print("Completed One Iteration")

plot_loss_curve(model= optimized_sa_NN_model,xlabel="Iterations", ylabel="Training Loss", title="Rice (NN) SA - Training Loss vs. Iterations (Optimized Params)", savefig="nn_loss_curve_opt_SA.png", label="SA - Optimized Params")

y_test_pred_sa = optimized_sa_NN_model.predict(X_test)
print("-------")
print(f"SA Complete!!! Optiomized Params and Scores below")
print(f"Time: {optimized_sa_time}")
print(f"Node: {optimized_node}")
print(f"Learning rate: {optimized_lr}")
print(f"Restarts: {optimized_restart}")
print(f"Best Accuracy Score: {best_accuracy_sa}")
print("-------")

train_sizes_abs, train_results, test_results = learning_curve(optimized_sa_NN_model, X_train, y_train,
                                                              train_sizes=np.linspace(0.1, 1.0, 10), cv=5)

param_range = np.linspace(0.1, 1.0, 10) * 100
plot_learning_curves(x_range=param_range, train_results=train_results, test_results=test_results,
                     title="Rice - Neural Network (SA) - Accuracy vs % Trained",
                     save_name="NN_rice_SA_learning_curve_score.png", x_label="Percent Trained", y_label="Accuracy",
                     parameter_range=param_range)

# GA
learning_rates = [0.01]
restarts = [2]
nodes = [16]
populations = [200]

new = True
time_ga_current = 0
for learning_rate in learning_rates:
    for node in nodes:
        for restart in restarts:
            for pop in populations:
                if new:
                    best_nn_ga = mlrose_hiive.NeuralNetwork(hidden_nodes=[node], activation='tanh',
                                                                   algorithm='genetic_alg',
                                                                   restarts=restart,
                                                                   max_iters=2000, bias=True, is_classifier=True, pop_size=pop,
                                                                   learning_rate=learning_rate, early_stopping=True,
                                                                   max_attempts=100, random_state=620, curve=True)
                    start = time.time()
                    best_nn_ga.fit(X_train, y_train)
                    end = time.time()
                    time_ga = end - start

                    ytrain_pred = best_nn_ga.predict(X_train)
                    ytrain_accuracy_ga = accuracy_score(y_train, ytrain_pred)

                    ytest_pred = best_nn_ga.predict(X_test)
                    ytest_accuracy_ga = accuracy_score(ytest_pred, y_test)

                    optimized_ga_NN_model = best_nn_ga
                    optimized_ga_time = time_ga
                    optimized_node = node
                    optimized_restart = restart
                    optimized_lr = learning_rate
                    optimized_pop = pop

                    best_accuracy_ga = ytest_accuracy_ga
                    print(f"Learning Rate: {learning_rate}")
                    print(f"Restarts: {restart}")
                    print(f"Node: {node}")
                    print(f"Time: {time_ga}")
                    print(f"Learning Rate: {learning_rate}")
                    print(f"Accuracy Score: {best_accuracy_ga}")
                    print(f"Population: {optimized_pop}")

                    new = False
                else:
                    nn_model_ga_instance = mlrose_hiive.NeuralNetwork(hidden_nodes=[node], activation='tanh',
                                                              algorithm='genetic_alg',
                                                              max_iters=1000, bias=True, restarts=restart, is_classifier=True, pop_size=pop,
                                                              learning_rate=learning_rate, early_stopping=True,
                                                              max_attempts=100, random_state=620, curve=True)

                    start = time.time()
                    nn_model_ga_instance.fit(X_train, y_train)
                    end = time.time()
                    time_ga = end - start

                    ytrain_pred = nn_model_ga_instance.predict(X_train)
                    ytrain_accuracy_ga = accuracy_score(y_train, ytrain_pred)

                    ytest_pred = nn_model_ga_instance.predict(X_test)
                    ytest_accuracy_ga = accuracy_score(ytest_pred, y_test)
                    print(f"Instance Test Results - {ytest_accuracy_ga}")

                    if ytest_accuracy_ga > best_accuracy_ga:
                        optimized_ga_NN_model = nn_model_ga_instance

                        optimized_ga_NN_model = best_nn_ga
                        optimized_ga_time = time_ga
                        optimized_node = node
                        optimized_restart = restart
                        optimized_lr = learning_rate
                        optimized_pop = pop

                        print(f"Learning Rate: {learning_rate}")
                        print(f"Restarts: {restart}")
                        print(f"Node: {node}")
                        print(f"Time: {time_ga}")
                        print(f"Population: {optimized_pop}")
                        best_accuracy_ga = ytest_accuracy_ga
                        print(f"Accuracy : {best_accuracy_ga}")

                print("Completed One Iteration")

plot_loss_curve(model= optimized_ga_NN_model,xlabel="Iterations", ylabel="Training Loss", title="Rice (NN) GA - Training Loss vs. Iterations (Optimized Params)", savefig="NN_rice_GA_loss_curve_score.png", label="GA - Optimized Params")

y_test_pred_ga = optimized_ga_NN_model.predict(X_test)
print("-------")
print(f"GA Complete!!! Optiomized Params and Scores below")
print(f"Time: {optimized_ga_time}")
print(f"Node: {optimized_node}")
print(f"Learning rate: {optimized_lr}")
print(f"Restarts: {optimized_restart}")
print(f"Population: {optimized_pop}")
print(f"Best Accuracy Score: {best_accuracy_ga}")
print("-------")

train_sizes_abs, train_results, test_results = learning_curve(optimized_ga_NN_model, X_train, y_train,
                                                              train_sizes=np.linspace(0.1, 1.0, 10), cv=5)

param_range = np.linspace(0.1, 1.0, 10) * 100
plot_learning_curves(x_range=param_range, train_results=train_results, test_results=test_results,
                     title="Rice - Neural Network (GA) - Accuracy vs % Trained",
                     save_name="NN_rice_GA_learning_curve_score.png", x_label="Percent Trained", y_label="Accuracy",
                     parameter_range=param_range)


# Back Prop
learning_rates = [0.01]
restarts = [2, 5, 10, 15]
nodes = [16]
populations = [100, 200, 300]

new = True
time_ga_current = 0
for learning_rate in learning_rates:
    for node in nodes:
        for restart in restarts:
            for pop in populations:
                if new:
                    best_nn_back = mlrose_hiive.NeuralNetwork(hidden_nodes=[node], activation='tanh',
                                                                   algorithm='gradient_descent',
                                                                   restarts=restart,
                                                                   max_iters=2000, bias=True, is_classifier=True, pop_size=pop,
                                                                   learning_rate=learning_rate, early_stopping=True,
                                                                   max_attempts=100, random_state=620, curve=True)
                    start = time.time()
                    best_nn_back.fit(X_train, y_train)
                    end = time.time()
                    time_ga = end - start

                    ytrain_pred = best_nn_back.predict(X_train)
                    ytrain_accuracy_bp = accuracy_score(y_train, ytrain_pred)

                    ytest_pred = best_nn_back.predict(X_test)
                    ytest_accuracy_bp = accuracy_score(ytest_pred, y_test)

                    optimized_bp_NN_model = best_nn_back
                    optimized_ga_time = time_ga
                    optimized_node = node
                    optimized_restart = restart
                    optimized_lr = learning_rate
                    optimized_pop = pop

                    best_accuracy_bp = ytest_accuracy_bp
                    print(f"Learning Rate: {learning_rate}")
                    print(f"Restarts: {restart}")
                    print(f"Node: {node}")
                    print(f"Time: {time_ga}")
                    print(f"Learning Rate: {learning_rate}")
                    print(f"Accuracy Score: {best_accuracy_bp}")
                    print(f"Population: {optimized_pop}")

                    new = False
                else:
                    nn_model_ga_instance = mlrose_hiive.NeuralNetwork(hidden_nodes=[node], activation='tanh',
                                                              algorithm='gradient_descent',
                                                              max_iters=1000, bias=True, restarts=restart, is_classifier=True, pop_size=pop,
                                                              learning_rate=learning_rate, early_stopping=True,
                                                              max_attempts=100, random_state=620, curve=True)

                    start = time.time()
                    nn_model_ga_instance.fit(X_train, y_train)
                    end = time.time()
                    time_ga = end - start

                    ytrain_pred = nn_model_ga_instance.predict(X_train)
                    ytrain_accuracy_bp = accuracy_score(y_train, ytrain_pred)

                    ytest_pred = nn_model_ga_instance.predict(X_test)
                    ytest_accuracy_bp = accuracy_score(ytest_pred, y_test)
                    print(f"Instance Test Results - {ytest_accuracy_bp}")

                    if ytest_accuracy_bp > best_accuracy_bp:
                        optimized_bp_NN_model = nn_model_ga_instance

                        # optimized_bp_NN_model = nn_model_ga_instance
                        optimized_ga_time = time_ga
                        optimized_node = node
                        optimized_restart = restart
                        optimized_lr = learning_rate
                        optimized_pop = pop

                        print(f"Learning Rate: {learning_rate}")
                        print(f"Restarts: {restart}")
                        print(f"Node: {node}")
                        print(f"Time: {time_ga}")
                        print(f"Population: {optimized_pop}")
                        best_accuracy_bp = ytest_accuracy_bp
                        print(f"Accuracy : {best_accuracy_bp}")

                print("Completed One Iteration")


plot_loss_curve_BP(model= optimized_bp_NN_model,xlabel="Iterations", ylabel="Training Loss", title="Rice (NN) BP - Training Loss vs. Iterations (Optimized Params)", savefig="NN_rice_BP_loss_curve_score.png", label="BP - Optimized Params")

y_test_pred_ga = optimized_bp_NN_model.predict(X_test)
print("-------")
print(f"BP Complete!!! Optiomized Params and Scores below")
print(f"Time: {optimized_ga_time}")
print(f"Node: {optimized_node}")
print(f"Learning rate: {optimized_lr}")
print(f"Restarts: {optimized_restart}")
print(f"Population: {optimized_pop}")
print(f"Best Accuracy Score: {best_accuracy_bp}")
print("-------")

train_sizes_abs, train_results, test_results = learning_curve(optimized_bp_NN_model, X_train, y_train,
                                                              train_sizes=np.linspace(0.1, 1.0, 10), cv=5)

param_range = np.linspace(0.1, 1.0, 10) * 100
plot_learning_curves(x_range=param_range, train_results=train_results, test_results=test_results,
                     title="Rice - Neural Network (BP) - Accuracy vs % Trained",
                     save_name="NN_rice_BP_learning_curve_score.png", x_label="Percent Trained", y_label="Accuracy",
                     parameter_range=param_range)


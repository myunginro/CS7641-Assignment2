from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import mlrose_hiive
import numpy as np
import time
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve, train_test_split
from sklearn import preprocessing


def plot_learning_curve(solver, X, y, title, fname, folder):
    full_name = folder + fname
    train_szs, train_scores, test_scores = learning_curve(solver, X, y, train_sizes=np.linspace(0.2, 1.0, 5))

    tr_means = np.mean(train_scores, axis=1)
    tr_stdev = np.std(train_scores, axis=1)
    tt_means = np.mean(test_scores, axis=1)
    tt_stdev = np.std(test_scores, axis=1)
    norm_train_szs = (train_szs / max(train_szs)) * 100.0

    plt.figure()
    plt.title(title)
    plt.xlabel('% of Training Data')
    plt.ylabel('Mean Accuracy Score')

    plt.plot(norm_train_szs, tr_means, 'o-', color='darkorange', label='train')
    plt.plot(norm_train_szs, tt_means, 'o-', color='navy', label='test')
    plt.fill_between(norm_train_szs, tr_means-tr_stdev, tr_means+tr_stdev, color='darkorange', alpha=0.1)
    plt.fill_between(norm_train_szs, tt_means-tt_stdev, tt_means+tt_stdev, color='navy', alpha=0.1)
    plt.legend(loc='best')

    plt.savefig(full_name, format='png')
    plt.close()

def train(algorithm, X_train, X_test, y_train, y_test, X, y):
    clf = mlrose_hiive.NeuralNetwork(hidden_nodes=[5, 2],activation='sigmoid',algorithm=algorithm, early_stopping=True,max_attempts=100, max_iters=5000,bias=True, learning_rate=.01,restarts=0, curve=True, random_state=42)
    start_time = time.time()
    clf.fit(X_train, y_train)
    runtime = time.time() - start_time

    y_pred = clf.predict(X_test)
    final_score = accuracy_score(y_test, y_pred)
    print(f"nn_{algorithm} accuracy: {final_score}")
    print(f"{algorithm} runtime is {runtime} s")

    fitness_curve = clf.fitness_curve
    if algorithm != "gradient_descent":
        fitness_curve = [i[0] for i in fitness_curve]

    plt.plot(fitness_curve)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.savefig(f"results/NN_{algorithm}_loss.png")
    plt.clf()

    # commented due to runtime, uncomment to generate validation curves
    # plot_learning_curve(clf, X, y, f"{algorithm} cross validation curve", f"{algorithm}_cv_curve.png", "results/")

np.random.seed(42)

# Load Data
print("Loading Breast Cancer Dataset")
X, y = load_breast_cancer(return_X_y=True)
scaler = preprocessing.StandardScaler().fit(X)
X_scaled = scaler.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)

algorithms = ['gradient_descent', 'random_hill_climb', 'simulated_annealing', 'genetic_alg']

for alg in algorithms:
    train(alg, X_train, X_test, y_train, y_test, X_scaled, y)
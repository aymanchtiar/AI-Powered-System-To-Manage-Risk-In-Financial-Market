import random
import pandas as pd
import numpy as np
from deap import base, creator, tools, algorithms
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import copy

# Load the preprocessed file
file_path = '/content/drive/MyDrive/data_stock/EURUSD_1d_processed.csv'
df = pd.read_csv(file_path)

# Define the label and drop irrelevant columns
label = 'Label_8'
X = df.drop(['Label_1', 'Label_2', 'Label_3', 'Label_4', 'Label_5', 'Label_6', 'Label_7', 'Label_8', 'Label_9', 'Label_10'], axis=1)
y = df[label]

# Manually split the data (90% for training, 10% for testing)
split_ratio = 0.9
split_idx = int(len(X) * split_ratio)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define hyperparameter space for SVM
param_spaces = {
    "SVM": {
        'C': [0.01, 0.1, 1, 10, 100, 1000], 
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  
        'degree': [2, 3, 4, 5],  
        'gamma': ['scale', 'auto', 0.01, 0.1, 1],  
        'coef0': [0.0, 0.5, 1.0], 
        'shrinking': [True, False], 
        'probability': [True, False], 
        'tol': [0.001, 0.0001, 0.01], 
        'cache_size': [200, 500, 1000], 
    }
}

# Create DEAP objects for genetic algorithms
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Function to create an individual for SVM
def create_individual():
    space = param_spaces["SVM"]
    individual = ["SVM"]
    for param in space:
        individual.append(random.choice(space[param]))
    return individual

toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Register crossover, mutation, and selection operations
toolbox.register("mate", tools.cxTwoPoint)

def custom_mutate(individual, indpb=0.2):
    for i, param in enumerate(individual[1:]):
        if random.random() < indpb:
            param_name = list(param_spaces["SVM"].keys())[i]
            space = param_spaces["SVM"][param_name]
            individual[1 + i] = random.choice(space)
    return individual,

toolbox.register("mutate", custom_mutate, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

best_individuals = {}

# Evaluation function for SVM
def evalModelModified(individual):
    params = dict(zip(param_spaces["SVM"].keys(), individual[1:]))
    model = SVC(**params)

    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    score = accuracy_score(y_test, predictions)

    classifier_name = "SVM"
    if classifier_name not in best_individuals or best_individuals[classifier_name].fitness.values[0] < score:
        best_individuals[classifier_name] = copy.deepcopy(individual)
        best_individuals[classifier_name].fitness.values = (score,)

    return (score,)

toolbox.register("evaluate", evalModelModified)

# Run the genetic algorithm
population = toolbox.population(n=50)
hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=30, stats=stats, halloffame=hof, verbose=True)

# Extract and save the best parameters and accuracies for SVM
if 'SVM' in best_individuals:
    best_svm = best_individuals['SVM']
    best_params = dict(zip(param_spaces['SVM'].keys(), best_svm[1:]))
    best_accuracy = best_svm.fitness.values[0]

    print(f"Best SVM Parameters: {best_params}")
    print(f"Best Accuracy: {best_accuracy}")
else:
    print("No best individual found for SVM.")

import random
import pandas as pd
import numpy as np
from deap import base, creator, tools, algorithms
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import copy

# Corrected param_spaces
param_spaces = {
    "Random Forest": {'n_estimators': [200, 400, 600],
                      'max_depth': [10, 120],
                      'min_samples_split': [2, 5, 10],
                      'min_samples_leaf': [1, 2, 4],
                     }
}

file_path = '/content/drive/MyDrive/PROJECT_STOCK/old_data2/AAPL_60m_processed.csv'
df = pd.read_csv(file_path)

label = 'Label_10'
X = df.drop(['Label_1', 'Label_2', 'Label_3', 'Label_4', 'Label_5', 'Label_6', 'Label_7', 'Label_8', 'Label_9', 'Label_10'], axis=1)
y = df[label]

split_ratio = 0.9
split_idx = int(len(X) * split_ratio)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train = y[:split_idx]
y_test = y[split_idx:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def custom_mutate(individual, indpb):
    for i in range(len(individual)):
        if random.random() < indpb:
            swap_idx = random.randint(0, len(individual) - 1)
            individual[i], individual[swap_idx] = individual[swap_idx], individual[i]
    return individual,

def create_individual():
    return [random.choice(values) for key, values in param_spaces["Random Forest"].items()]

if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", custom_mutate, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

best_individuals = {}

def evalModelModified(individual):
    params = dict(zip(param_spaces["Random Forest"].keys(), individual))
    model = RandomForestClassifier(**params)
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    score = accuracy_score(y_test, predictions)

    classifier_name = "Random Forest"
    if classifier_name not in best_individuals or best_individuals[classifier_name].fitness.values[0] < score:
        best_individuals[classifier_name] = copy.deepcopy(individual)
        best_individuals[classifier_name].fitness.values = (score,)

    return (score,)

toolbox.register("evaluate", evalModelModified)

population = toolbox.population(n=60)
hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.5, ngen=10, stats=stats, halloffame=hof, verbose=True)

# Extract and save the best parameters and accuracies for Random Forest
if 'Random Forest' in best_individuals:
    best_rf = best_individuals['Random Forest']
    best_params = dict(zip(param_spaces['Random Forest'].keys(), best_rf))
    best_accuracy = best_rf.fitness.values[0]

    print(f"Best Random Forest Parameters: {best_params}")
    print(f"Best Accuracy: {best_accuracy}")
else:
    print("No best individual found for Random Forest.")

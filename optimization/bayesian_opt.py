import numpy as np
from bayes_opt import BayesianOptimization

MIN_MARGIN = 0.20

def compute_margin(r, a, b, gamma):
    return MIN_MARGIN + a * np.log(r) + b * (r - 1) ** gamma

def fitness(a, b, gamma):
    targets = [(2.33, 1.7), (5.0, 2.5), (13.0, 5.0)]
    error = 0

    for r, target in targets:
        m = compute_margin(r, a, b, gamma)
        error += (1 + m - target) ** 2

    return -error

optimizer = BayesianOptimization(
    f=fitness,
    pbounds={"a": (0.05, 1.0), "b": (0.01, 1.0), "gamma": (0.3, 1.2)},
)

optimizer.maximize(init_points=20, n_iter=50)
print(optimizer.max)
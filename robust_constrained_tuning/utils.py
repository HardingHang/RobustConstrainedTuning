import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import spatial

def euclidean_distance(p, q):
    return np.linalg.norm(p - q)


def worst_case_workload_within_uncertainty(workload_composition, uncertainty_bound, q_cost, u_cost,
                                           q_request=False, u_request=False):
    q_cost = q_cost / np.max(q_cost)

    if np.max(u_cost) > 0:
        u_cost = u_cost / np.max(u_cost)
    else:
        u_cost = np.zeros_like(u_cost)
    size = len(workload_composition)

    q_worst_case = None
    u_worst_case = None

    if q_request:
        concat_cost = np.zeros_like(u_cost)
        cost = np.concatenate((q_cost, concat_cost))
        obj = lambda x: 0 - np.dot(x, cost)

        lower_bound = [0.0] * size
        upper_bound = [1.0] * size
        bound = tuple(zip(lower_bound, upper_bound))

        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
                       {'type': 'ineq',
                        'fun': lambda x: uncertainty_bound - euclidean_distance(x, workload_composition)}]

        res = minimize(obj, workload_composition, constraints=constraints, bounds=bound, method='SLSQP')
        q_worst_case = res.x

    if u_request:
        concat_cost = np.zeros_like(q_cost)
        cost = np.concatenate((concat_cost, u_cost))
        obj = lambda x: 0 - np.dot(x, cost)

        lower_bound = [0.0] * size
        upper_bound = [1.0] * size
        bound = tuple(zip(lower_bound, upper_bound))

        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
                       {'type': 'ineq',
                        'fun': lambda x: uncertainty_bound - euclidean_distance(x, workload_composition)}]

        res = minimize(obj, workload_composition, constraints=constraints, bounds=bound, method='SLSQP')
        u_worst_case = res.x

    return q_worst_case, u_worst_case


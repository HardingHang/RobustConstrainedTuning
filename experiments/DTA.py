import pickle
import unittest
import numpy as np

from robust_constrained_tuning import worst_case_workload_within_uncertainty
from selection.cost_evaluation import CostEvaluation
from selection.dbms.postgres_dbms import PostgresDatabaseConnector
from selection.index_selection_evaluation import IndexSelection
from selection.workload import Workload


def DTA_variant(config):
    with open("./tpcds/candidates", 'rb') as f:
        candidates = pickle.load(f)
    maintenance_cost = np.load('./tpcds/index_maintenance_cost.npy')
    nominal_workload = np.load("./tpcds/workload.npy")

    config['index_selection']['parameters']['candidate_indexes'] = candidates
    config['index_selection']['parameters']['maintenance_cost'] = maintenance_cost
    config['index_selection']['parameters']['workload_composition'] = nominal_workload
    db_connector = PostgresDatabaseConnector("tpcds_10")
    cost_evaluation = CostEvaluation(db_connector)
    index_selection = IndexSelection('tpcds_10', config=config['index_selection'])
    num_updates = 7

    with open("./tpcds/queries", 'rb') as f:
        queries = pickle.load(f)

    indexes, _, _, _ = index_selection.run(Workload(queries))
    with open("./tpcds/indexes", 'wb') as f:
        pickle.dump(indexes, f)
    print(len(indexes))
    print(indexes)

    queries_cost = np.asarray([cost_evaluation.calculate_cost(Workload([q]), indexes) for q in queries])
    queries_cost_without_indexes = np.asarray(
        [cost_evaluation.calculate_cost(Workload([q]), []) for q in queries])

    workload_cost_without_indexes = np.dot(nominal_workload[:len(queries)], queries_cost_without_indexes)
    print("workload cost without indexes", workload_cost_without_indexes)

    q_composition, _ = worst_case_workload_within_uncertainty(
        nominal_workload, config['index_selection']['parameters']['uncertainty_bound'],
        queries_cost_without_indexes, np.ones(num_updates),
        q_request=True)
    q_composition = q_composition[:len(queries)]
    workload_cost_without_indexes = np.dot(q_composition, queries_cost_without_indexes)
    print("worst workload cost without_indexes", workload_cost_without_indexes)

    nominal_workload_cost = np.dot(nominal_workload[:len(queries)], queries_cost)
    print("nominal workload cost", nominal_workload_cost)

    worst_case_workload, _ = worst_case_workload_within_uncertainty(nominal_workload,
                                                                    config['index_selection']['parameters'][
                                                                        'uncertainty_bound'],
                                                                    queries_cost,
                                                                    np.ones(num_updates), q_request=True)

    worst_workload_cost = np.dot(worst_case_workload[:len(queries)], queries_cost)
    print("worst-case workload cost", worst_workload_cost)

    sum_maintenance_cost = 0
    for index in indexes:
        cost = maintenance_cost[candidates.index(index)]
        print(index, cost)
        sum_maintenance_cost += cost
    print(sum_maintenance_cost)

    table_list = ["catalog_returns", "catalog_sales", "inventory", "store_returns",
                  "store_sales", "web_returns", "web_sales"]
    index_maintenance_cost_per_update = np.zeros(self.num_updates)
    for index in indexes:
        pos = candidates.index(index)
        indexed_table = index.table().name
        if indexed_table in table_list:
            affected_update_idx = table_list.index(indexed_table)
            index_maintenance_cost_per_update[affected_update_idx] += maintenance_cost[pos]
    index_maintenance_cost = np.dot(nominal_workload[-num_updates:], index_maintenance_cost_per_update)
    print('nominal index_maintenance_cost', index_maintenance_cost)

    _, u_composition = worst_case_workload_within_uncertainty(
        nominal_workload, config['index_selection']['parameters']['uncertainty_bound'], np.ones(len(queries)),
        index_maintenance_cost_per_update, u_request=True)
    u_composition = u_composition[0 - num_updates:]
    new_cost = np.dot(index_maintenance_cost_per_update, u_composition)
    print('worst-case index_maintenance_cost', new_cost)


if __name__ == '__main__':
    config = {
        "index_selection": {
            "name": "anytime_cost_constrained",
            "parameters": {
                "budget_MB": 10000,
                "max_index_width": 2,
                "is_cost_constrained": True,
                "is_robust": True,
                "uncertainty_bound": 0.05,
                "cost_limit": 2.01725178,
                "candidate_indexes": None,
                "maintenance_cost": None,
                "workload_composition": None
            },
        },
    }

    # storage budget 5000MB
    for i in [0.5, 0.6, 0.7, 0.8]:
        maintenance_cost = 2.4072370326072616
        config['index_selection']['parameters']['budget_MB'] = 5000
        config['index_selection']['parameters']['cost_limit'] = maintenance_cost * i
        DTA_variant(config)

    # storage budget 7500MB
    for i in [0.5, 0.6, 0.7, 0.8]:
        maintenance_cost = 3.347175139322912
        config['index_selection']['parameters']['budget_MB'] = 7500
        config['index_selection']['parameters']['cost_limit'] = maintenance_cost * i
        DTA_variant(config)

    # storage budget 7500MB
    for i in [0.5, 0.6, 0.7, 0.8]:
        maintenance_cost = 4.034503567096869
        config['index_selection']['parameters']['budget_MB'] = 10000
        config['index_selection']['parameters']['cost_limit'] = maintenance_cost * i
        DTA_variant(config)

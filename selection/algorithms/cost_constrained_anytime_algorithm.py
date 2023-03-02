import itertools
import logging
import math
import time
import numpy as np


from robust_constrained_tuning import worst_case_workload_within_uncertainty
from ..candidate_generation import candidates_per_query, syntactically_relevant_indexes
from ..index import Index, index_merge
from ..selection_algorithm import DEFAULT_PARAMETER_VALUES, SelectionAlgorithm
from ..utils import get_utilized_indexes, indexes_by_table, mb_to_b

# budget_MB: The algorithm can utilize the specified storage budget in MB.
# max_index_width: The number of columns an index can contain at maximum.
# max_runtime_minutes: The algorithm is stopped either if all seeds are evaluated or
#                      when max_runtime_minutes is exceeded. Whatever happens first.
#                      In case of the latter, the current best solution is returned.
from ..workload import Workload

DEFAULT_PARAMETERS = {
    "budget_MB": DEFAULT_PARAMETER_VALUES["budget_MB"],
    "max_index_width": DEFAULT_PARAMETER_VALUES["max_index_width"],
    "max_runtime_minutes": 30,
    "cost_limit": 10,
    "candidate_indexes": None,
    "maintenance_cost": None,
}


# This algorithm is related to the DTA Anytime algorithm employed in SQL server.
# Details of the current version of the original algorithm are not published yet.
# See the documentation for a general description:
# https://docs.microsoft.com/de-de/sql/tools/dta/dta-utility?view=sql-server-ver15
#
# Please note, that this implementation does not reflect the behavior and performance
# of the original algorithm, which might be continuously enhanced and optimized.
class CostConstrainedAnytimeAlgorithm(SelectionAlgorithm):
    def __init__(self, database_connector, parameters=None):
        if parameters is None:
            parameters = {}
        SelectionAlgorithm.__init__(
            self, database_connector, parameters, DEFAULT_PARAMETERS
        )
        self.disk_constraint = mb_to_b(self.parameters["budget_MB"])
        self.max_index_width = self.parameters["max_index_width"]
        self.max_runtime_minutes = self.parameters["max_runtime_minutes"]

        self.is_robust = self.parameters["is_robust"]
        self.is_cost_constrained = self.parameters["is_cost_constrained"]
        self.cost_limit = self.parameters["cost_limit"]

        self.maintenance_cost = self.parameters["maintenance_cost"]
        self.candidate_indexes = self.parameters["candidate_indexes"]
        self.workload_composition = self.parameters["workload_composition"]
        self.uncertainty_bound = self.parameters["uncertainty_bound"]

        self.table_list = ["catalog_returns", "catalog_sales", "inventory", "store_returns",
                           "store_sales", "web_returns", "web_sales"]
        self.num_updates = 7

    def _calculate_best_indexes(self, workload):
        logging.info("Calculating best indexes Anytime")

        # Generate syntactically relevant candidates
        candidates = candidates_per_query(
            workload,
            self.parameters["max_index_width"],
            candidate_generator=syntactically_relevant_indexes,
        )

        # Obtain best (utilized) indexes per query
        candidates, _ = get_utilized_indexes(workload, candidates, self.cost_evaluation)

        self._add_merged_indexes(candidates)

        # Remove candidates that cannot meet budget requirements
        seeds = []
        filtered_candidates = set()

        new_candidates = []
        for candidate in candidates:
            if candidate in self.candidate_indexes:
                new_candidates.append(candidate)

        for candidate in new_candidates:
            if candidate.estimated_size > self.disk_constraint:
                continue

            if self.is_cost_constrained:
                maintenance_cost = self.maintenance_cost[self.candidate_indexes.index(candidate)]
                index_maintenance_cost = \
                    np.full_like(np.ones(self.num_updates), maintenance_cost)
                if maintenance_cost > 0:
                    if self.is_robust:
                        _, u_composition = worst_case_workload_within_uncertainty(
                            self.workload_composition, self.uncertainty_bound,
                            np.ones(len(workload.queries)),
                            index_maintenance_cost,
                            u_request=True)
                        u_composition = u_composition[0 - self.num_updates:]
                        if np.dot(u_composition[0 - self.num_updates:], index_maintenance_cost) > self.cost_limit:
                            continue
                    else:
                        if np.dot(self.workload_composition[0 - self.num_updates:], index_maintenance_cost) > self.cost_limit:
                            continue

            seeds.append({candidate})
            filtered_candidates.add(candidate)

        # For reproducible results, we sort the seeds and candidates
        seeds = sorted(seeds, key=lambda candidate: candidate)
        filtered_candidates = set(
            sorted(filtered_candidates, key=lambda candidate: candidate)
        )

        seeds.append(set())
        candidates = filtered_candidates

        start_time = time.time()
        best_configuration = (None, None)
        start_time = time.time()
        best_configuration = (None, None)
        if self.is_robust:
            current_costs = self._simulate_and_evaluate_worst_case_cost(workload, [])
        else:
            current_costs = self._simulate_and_evaluate_cost(workload, [])
        indexes, costs = self.enumerate_greedy(
            workload, set(), current_costs, candidates, math.inf
        )
        if best_configuration[0] is None or costs < best_configuration[1]:
            best_configuration = (indexes, costs)

        indexes = best_configuration[0]
        return list(indexes)

    def _add_merged_indexes(self, indexes):
        index_table_dict = indexes_by_table(indexes)
        for table in index_table_dict:
            for index1, index2 in itertools.permutations(index_table_dict[table], 2):
                merged_index = index_merge(index1, index2)
                if len(merged_index.columns) > self.max_index_width:
                    new_columns = merged_index.columns[: self.max_index_width]
                    merged_index = Index(new_columns)
                if merged_index not in indexes:
                    self.cost_evaluation.estimate_size(merged_index)
                    indexes.add(merged_index)

    # based on AutoAdminAlgorithm
    def enumerate_greedy(
            self, workload, current_indexes, current_costs, candidate_indexes, number_indexes,
    ):
        assert (
                current_indexes & candidate_indexes == set()
        ), "Intersection of current and candidate indexes must be empty"
        if len(current_indexes) >= number_indexes:
            return current_indexes, current_costs

        # (index, cost)
        best_index = (None, None)

        logging.debug(f"Searching in {len(candidate_indexes)} indexes")
        removed_indexes = []

        index_maintenance_cost_per_update = np.zeros(self.num_updates)
        if self.is_cost_constrained:
            for index in current_indexes:
                indexed_table = index.table().name
                pos = self.candidate_indexes.index(index)
                if indexed_table in self.table_list:
                    affected_update_idx = self.table_list.index(indexed_table)
                    index_maintenance_cost_per_update[affected_update_idx] += self.maintenance_cost[pos]

        for index in candidate_indexes:
            if (
                    sum(idx.estimated_size for idx in current_indexes | {index})
                    > self.disk_constraint
            ):
                # index configuration is too large
                continue
            indexed_table = index.table().name
            if self.is_cost_constrained:
                pos = self.candidate_indexes.index(index)
                if indexed_table in self.table_list:
                    affected_update_idx = self.table_list.index(indexed_table)
                    index_maintenance_cost_per_update[affected_update_idx] += self.maintenance_cost[pos]

            if self.is_robust:
                if self.is_cost_constrained and indexed_table in self.table_list:
                    _, u_composition = worst_case_workload_within_uncertainty(
                        self.workload_composition, self.uncertainty_bound,
                        np.ones(len(workload.queries)),
                        index_maintenance_cost_per_update,
                        u_request=True)
                    u_composition = u_composition[0 - self.num_updates:]
                    if (
                            np.dot(u_composition, index_maintenance_cost_per_update) > self.cost_limit
                            # current_maintenance_cost + self.maintenance_cost[self.candidate_indexes.index(index)]
                            # > self.cost_limit
                    ):
                        removed_indexes.append(index)
                        continue

                #cost = self._simulate_and_evaluate_worst_case_cost(workload, current_indexes | {index})
                cost = self._simulate_and_evaluate_cost(workload, current_indexes|{index})

            else:
                if self.is_cost_constrained and indexed_table in self.table_list:
                    if (
                            np.dot(self.workload_composition[0 - self.num_updates:],
                                   index_maintenance_cost_per_update) > self.cost_limit
                            # current_maintenance_cost + self.maintenance_cost[self.candidate_indexes.index(index)]
                            # > self.cost_limit
                    ):
                        removed_indexes.append(index)
                        continue

                cost = self._simulate_and_evaluate_cost(workload, current_indexes | {index})

            if not best_index[0] or cost < best_index[1]:
                best_index = (index, cost)

        for index in removed_indexes:
            candidate_indexes.remove(index)

        if best_index[0] and best_index[1] < current_costs:
            current_indexes.add(best_index[0])
            candidate_indexes.remove(best_index[0])
            current_costs = best_index[1]

            logging.debug(f"Additional best index found: {best_index}")

            return self.enumerate_greedy(
                workload,
                current_indexes,
                current_costs,
                candidate_indexes,
                number_indexes,
            )
        return current_indexes, current_costs

    # copied from AutoAdminAlgorithm
    def _simulate_and_evaluate_cost(self, workload, indexes):
        queries_cost = np.asarray(
            [self.cost_evaluation.calculate_cost(Workload([q]), indexes) for q in workload.queries])
        cost = np.dot(self.workload_composition[:len(workload.queries)], queries_cost)
        return round(cost, 2)

    def _simulate_and_evaluate_worst_case_cost(self, workload, indexes):
        queries_cost = np.asarray(
            [self.cost_evaluation.calculate_cost(Workload([q]), indexes) for q in workload.queries])

        worst_case_workload, _ = worst_case_workload_within_uncertainty(self.workload_composition,
                                                                        self.uncertainty_bound, queries_cost,
                                                                        np.ones(self.num_updates), q_request=True)
        worst_workload_cost = np.dot(worst_case_workload[:len(workload.queries)], queries_cost)
        return round(worst_workload_cost, 2)

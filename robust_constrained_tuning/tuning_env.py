import copy

import gym
from gym import spaces
from selection.dbms.postgres_dbms import PostgresDatabaseConnector
from selection.cost_evaluation import CostEvaluation
from selection.workload import *
from selection.index import *
from selection.utils import *
from selection.candidate_generation import *
from .utils import *
import numpy as np
import time
import math
import pickle


class StorageConstrainedTuningEnv(gym.Env):
    def __init__(self, config=None, render_mode=None):
        self.robust = config["robust"]
        self.constrained = config["constrained"]
        self.sample_as_proxy = config['sample_as_proxy']

        self.benchmark = config["benchmark"]
        self.db_name = config["db_name"]

        self.db_connector = PostgresDatabaseConnector(self.db_name)
        self.cost_evaluation = CostEvaluation(self.db_connector)
        self.storage_constraint = mb_to_b(config["storage_constraint"])
        self.uncertainty_bound = config["uncertainty_bound"]
        self.sample_file = config["sample_file"]

        if self.benchmark == "tpch":
            queries_file = "./%s/queries" % self.benchmark
            candidates_file = "./%s/candidates" % self.benchmark
            workload_file = "./%s/workload.npy" % self.benchmark
            maintenance_cost_file = "./%s/index_maintenance_cost.npy" % self.benchmark
            self.num_updates = 2
        elif self.benchmark == "tpcds":
            queries_file = "./%s/queries" % self.benchmark
            candidates_file = "./%s/candidates" % self.benchmark
            workload_file = "./%s/workload.npy" % self.benchmark
            maintenance_cost_file = "./%s/index_maintenance_cost.npy" % self.benchmark
            sample_file = "./%s/%s" % (self.benchmark, self.sample_file)
            self.num_updates = 7

        with open(queries_file, 'rb') as f:
            self.queries = pickle.load(f)
        with open(candidates_file, 'rb') as f:
            self.candidate_indexes = pickle.load(f)
        self.num_queries = len(self.queries)
        self.indexes_size = []
        for index in self.candidate_indexes:
            self.indexes_size.append(index.estimated_size)
        self.indexes_size = np.asarray(self.indexes_size)

        # s = np.ones(self.num_queries)
        # self.nominal_workload = s / np.sum(s)
        self.nominal_workload = np.load(workload_file)
        self.maintenance_cost = np.load(maintenance_cost_file)
        if self.sample_as_proxy:
            self.samples = np.load(sample_file)

        self.num_candidates = len(self.candidate_indexes)

        self.action_space = spaces.Discrete(self.num_candidates)
        self.observation_space = spaces.MultiBinary(self.num_candidates)

        self.actual_reward = None
        self.cur_obs = None
        self.action_mask = None

        self.workload_cost_without_indexes = None
        self.cur_workload_cost = None
        self.action_mask = np.zeros(self.num_candidates, dtype=np.int8)

        self.cost_limit = config["cost_limit"]

        self.reset_count = 0
        self.cur_cost = 0
        self.cur_storage_consumption = 0

        self.action_mask_duration = 0
        self.reward_calculation_duration = 0

        self.table_list = ["catalog_returns", "catalog_sales", "inventory", "store_returns",
                           "store_sales", "web_returns", "web_sales"]

        self.init_action_mask = np.zeros(self.num_candidates, dtype=np.int8)
        for i in range(self.num_candidates):
            index = self.candidate_indexes[i]
            if index.is_single_column() and index.estimated_size < self.storage_constraint:
                self.init_action_mask[i] = 1

        if self.constrained:
            valid_actions = np.nonzero(self.action_mask)[0]
            for pos in valid_actions:
                indexed_table = self.candidate_indexes[pos].table().name
                if indexed_table in self.table_list:
                    affected_update_idx = self.table_list.index(indexed_table)
                    index_maintenance_cost_per_update = np.zeros(self.num_updates)
                    index_maintenance_cost_per_update[affected_update_idx] += self.maintenance_cost[pos]
                    if self.robust:
                        _, u_composition = worst_case_workload_within_uncertainty(
                            self.nominal_workload, self.uncertainty_bound, np.ones(self.num_queries),
                            index_maintenance_cost_per_update, u_request=True)
                        u_composition = u_composition[0 - self.num_updates:]
                        new_cost = np.dot(index_maintenance_cost_per_update, u_composition)
                    else:
                        new_cost = np.dot(index_maintenance_cost_per_update,
                                          self.nominal_workload[0 - self.num_updates:])
                    if new_cost > self.cost_limit:
                        self.init_action_mask[pos] = 0

    def reset(self, seed=None, options=None):
        self.actual_reward = 0
        self.cur_storage_consumption = 0
        self.cur_cost = 0
        # reset cost evaluation due to the request cache is too large
        self.db_connector.close()
        self.db_connector = PostgresDatabaseConnector(self.db_name)
        self.cost_evaluation = CostEvaluation(self.db_connector)
        self.reset_count += 1

        self.cur_obs = np.zeros(self.num_candidates, dtype=np.int8)
        self.action_mask = np.zeros(self.num_candidates, dtype=np.int8)
        for i in range(self.num_candidates):
            index = self.candidate_indexes[i]
            if index.estimated_size < self.storage_constraint:
                self.action_mask[i] = 1
        # self.action_mask = copy.deepcopy(self.init_action_mask)

        self.info = {"cost": 0, "action_mask": self.action_mask}

        queries_cost_without_indexes = []
        for q in self.queries:
            queries_cost_without_indexes.append(self.cost_evaluation.calculate_cost(Workload([q]), []))
        queries_cost_without_indexes = np.asarray(queries_cost_without_indexes)

        if self.robust:
            if self.sample_as_proxy:
                queries_cost_without_indexes = np.concatenate(
                    (queries_cost_without_indexes, np.zeros(self.num_updates)))
                samples_workload_cost_without_indexes = np.dot(self.samples, queries_cost_without_indexes)
                self.workload_cost_without_indexes = np.max(samples_workload_cost_without_indexes)
            else:
                q_composition, _ = worst_case_workload_within_uncertainty(
                    self.nominal_workload, self.uncertainty_bound, queries_cost_without_indexes,
                    np.ones(self.num_updates),
                    q_request=True)
                q_composition = q_composition[:self.num_queries]
                self.workload_cost_without_indexes = np.dot(queries_cost_without_indexes, q_composition)
        else:
            self.workload_cost_without_indexes = np.dot(queries_cost_without_indexes,
                                                        self.nominal_workload[:self.num_queries])

        # self.workload_cost_without_indexes = np.dot(queries_cost_without_indexes,
        #                                             self.nominal_workload[:self.num_queries])

        self.cur_workload_cost = self.workload_cost_without_indexes

        return self.cur_obs

    def step(self, action):
        start_time = time.time()
        self._action(action)
        end_time = time.time()
        duration = end_time - start_time
        self.action_mask_duration += duration

        start_time = time.time()
        reward = self._reward()
        end_time = time.time()
        duration = end_time - start_time
        self.reward_calculation_duration += duration
        if self.cur_cost > self.cost_limit:
            self.info['cost'] = 1
        else:
            self.info['cost'] = 0
        self.info['action_mask'] = self.action_mask

        if self._terminated():
            terminated = 1
            if self.sample_as_proxy:
                self._actual()
        else:
            terminated = 0

        return self.cur_obs, reward, terminated, self.info

    def _actual(self):
        indexes = [self.candidate_indexes[i] for i in np.where(self.cur_obs == 1)[0]]

        queries_cost = []
        for q in self.queries:
            queries_cost.append(self.cost_evaluation.calculate_cost(Workload([q]), indexes))
        queries_cost = np.asarray(queries_cost)
        q_composition, _ = worst_case_workload_within_uncertainty(
            self.nominal_workload, self.uncertainty_bound, queries_cost, np.ones(self.num_updates),
            q_request=True)
        q_composition = q_composition[:self.num_queries]
        workload_cost = np.dot(queries_cost, q_composition)

        queries_cost_without_indexes = []
        for q in self.queries:
            queries_cost_without_indexes.append(self.cost_evaluation.calculate_cost(Workload([q]), []))
        queries_cost_without_indexes = np.asarray(queries_cost_without_indexes)
        q_composition, _ = worst_case_workload_within_uncertainty(
            self.nominal_workload, self.uncertainty_bound, queries_cost_without_indexes, np.ones(self.num_updates),
            q_request=True)
        q_composition = q_composition[:self.num_queries]
        workload_cost_without_indexes = np.dot(queries_cost_without_indexes, q_composition)

        self.actual_reward = (workload_cost_without_indexes - workload_cost) * 1.0 / workload_cost_without_indexes

        u_cost = self._index_maintenance_cost()
        _, u_composition = worst_case_workload_within_uncertainty(
            self.nominal_workload, self.uncertainty_bound, np.ones(self.num_queries), u_cost,
            u_request=True)
        u_composition = u_composition[0 - self.num_updates:]
        self.cur_cost = np.dot(u_cost, u_composition)

    def _reward(self):
        indexes = [self.candidate_indexes[i] for i in np.where(self.cur_obs == 1)[0]]

        queries_cost = []
        for q in self.queries:
            queries_cost.append(self.cost_evaluation.calculate_cost(Workload([q]), indexes))
        queries_cost = np.asarray(queries_cost)
        if self.robust:
            if self.sample_as_proxy:
                workload_q_cost = np.concatenate((queries_cost, np.zeros(self.num_updates)))
                samples_workload_cost = np.dot(self.samples, workload_q_cost)
                workload_cost = np.max(samples_workload_cost)
            else:
                q_composition, _ = worst_case_workload_within_uncertainty(
                    self.nominal_workload, self.uncertainty_bound, queries_cost, np.ones(self.num_updates),
                    q_request=True)
                q_composition = q_composition[:self.num_queries]
                workload_cost = np.dot(queries_cost, q_composition)
        else:
            workload_cost = np.dot(queries_cost, self.nominal_workload[:self.num_queries])

        # workload_cost = np.dot(queries_cost, self.nominal_workload[:self.num_queries])

        reward = (self.cur_workload_cost - workload_cost) * 1.0 / self.workload_cost_without_indexes

        self.cur_workload_cost = workload_cost
        return reward

    def _action(self, action):
        index = self.candidate_indexes[action]
        self.cur_obs[action] = 1
        self.action_mask[action] = 0

        self.cur_storage_consumption = np.dot(self.indexes_size, self.cur_obs)

        valid_actions = np.nonzero(self.action_mask)[0]
        for pos in valid_actions:
            if self.cur_storage_consumption + self.indexes_size[pos] > self.storage_constraint:
                self.action_mask[pos] = 0

        u_cost = self._index_maintenance_cost()

        if self.robust:
            if self.sample_as_proxy:
                workload_u_cost = np.concatenate((np.zeros(self.num_queries), u_cost))
                samples_u_cost = np.dot(self.samples, workload_u_cost)
                self.cur_cost = np.max(samples_u_cost)
            else:
                _, u_composition = worst_case_workload_within_uncertainty(
                    self.nominal_workload, self.uncertainty_bound, np.ones(self.num_queries), u_cost,
                    u_request=True)
                u_composition = u_composition[0 - self.num_updates:]
                self.cur_cost = np.dot(u_cost, u_composition)
        else:
            self.cur_cost = np.dot(u_cost, self.nominal_workload[0 - self.num_updates:])

        if self.constrained:
            valid_actions = np.nonzero(self.action_mask)[0]
            for pos in valid_actions:
                indexed_table = self.candidate_indexes[pos].table().name
                if indexed_table in self.table_list:
                    affected_update_idx = self.table_list.index(indexed_table)
                    index_maintenance_cost_per_update = copy.deepcopy(u_cost)
                    index_maintenance_cost_per_update[affected_update_idx] += self.maintenance_cost[pos]
                    if self.robust:
                        if self.sample_as_proxy:
                            workload_u_cost = np.concatenate(
                                (np.zeros(self.num_queries), index_maintenance_cost_per_update))
                            samples_u_cost = np.dot(self.samples, workload_u_cost)
                            new_cost = np.max(samples_u_cost)
                        else:
                            _, u_composition = worst_case_workload_within_uncertainty(
                                self.nominal_workload, self.uncertainty_bound, np.ones(self.num_queries),
                                index_maintenance_cost_per_update, u_request=True)
                            u_composition = u_composition[0 - self.num_updates:]
                            new_cost = np.dot(index_maintenance_cost_per_update, u_composition)
                    else:
                        new_cost = np.dot(index_maintenance_cost_per_update,
                                          self.nominal_workload[0 - self.num_updates:])
                    if new_cost > self.cost_limit:
                        self.action_mask[pos] = 0

    def _terminated(self):
        return np.count_nonzero(self.action_mask) == 0 or np.count_nonzero(self.cur_obs) == 60

    def _index_maintenance_cost(self):
        index_maintenance_cost_per_update = np.zeros(self.num_updates)
        indexes_pos = np.nonzero(self.cur_obs)[0]
        for pos in indexes_pos:
            indexed_table = self.candidate_indexes[pos].table().name
            if indexed_table in self.table_list:
                affected_update_idx = self.table_list.index(indexed_table)
                index_maintenance_cost_per_update[affected_update_idx] += self.maintenance_cost[pos]

        return index_maintenance_cost_per_update

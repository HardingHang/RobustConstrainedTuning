import copy
import json
import logging
import pickle
import sys
import time


from .algorithms.cost_constrained_anytime_algorithm import CostConstrainedAnytimeAlgorithm
from .algorithms.anytime_algorithm import AnytimeAlgorithm
from .algorithms.auto_admin_algorithm import AutoAdminAlgorithm
from .algorithms.db2advis_algorithm import DB2AdvisAlgorithm
from .algorithms.dexter_algorithm import DexterAlgorithm
from .algorithms.drop_heuristic_algorithm import DropHeuristicAlgorithm
from .algorithms.extend_algorithm import ExtendAlgorithm
from .algorithms.extend_variant import ExtendVariant
from .algorithms.extend_variant import ExtendVariant
from .algorithms.relaxation_algorithm import RelaxationAlgorithm

from .benchmark import Benchmark
from .dbms.postgres_dbms import PostgresDatabaseConnector
from .query_generator import QueryGenerator
from .selection_algorithm import AllIndexesAlgorithm, NoIndexAlgorithm
from .table_generator import TableGenerator
from .workload import Workload

ALGORITHMS = {
    "anytime": AnytimeAlgorithm,
    "anytime_cost_constrained": CostConstrainedAnytimeAlgorithm,
    "auto_admin": AutoAdminAlgorithm,
    "db2advis": DB2AdvisAlgorithm,
    "dexter": DexterAlgorithm,
    "drop": DropHeuristicAlgorithm,
    "extend": ExtendAlgorithm,
    "relaxation": RelaxationAlgorithm,
    "no_index": NoIndexAlgorithm,
    "all_indexes": AllIndexesAlgorithm,
}

DEFAULT_CONFIG = {
    "name": "db2advis",
    "parameters": {
        "max_index_width": 2,
        "budget_MB": 50,
        "try_variations_seconds": 0
    },
    "timeout": 300
}


class IndexSelection:
    def __init__(self, db_name, config=DEFAULT_CONFIG):
        self.db_connector = None
        self.disable_output_files = False
        self.database_name = db_name
        self.setup_db_connector(self.database_name)
        self.db_connector.drop_indexes()
        self.db_connector.create_statistics()
        self.db_connector.commit()
        self.algorithm_name = config["name"]
        self.algorithm_params = config["parameters"]


    def run(self, workload):
        self.db_connector.drop_indexes()
        self.db_connector.commit()
        algorithm = self.create_algorithm_object(self.algorithm_name, self.algorithm_params)
        logging.info(f"Running algorithm {self.algorithm_name}")
        indexes = algorithm.calculate_best_indexes(workload)
        logging.info(f"Indexes found: {indexes}")
        what_if = algorithm.cost_evaluation.what_if

        cost_requests = (
            self.db_connector.cost_estimations
            if self.algorithm_name == "db2advis"
            else algorithm.cost_evaluation.cost_requests
        )
        cache_hits = (
            0 if self.algorithm_name == "db2advis" else algorithm.cost_evaluation.cache_hits
        )
        return indexes, what_if, cost_requests, cache_hits

    def create_algorithm_object(self, algorithm_name, parameters):
        algorithm = ALGORITHMS[algorithm_name](self.db_connector, parameters)
        return algorithm

    def setup_db_connector(self, database_name):
        if self.db_connector:
            logging.info("Create new database connector (closing old)")
            self.db_connector.close()
        self.db_connector = PostgresDatabaseConnector(database_name)

from selection.dbms.postgres_dbms import PostgresDatabaseConnector
from selection.cost_evaluation import CostEvaluation
from selection.workload import *
from selection.index import *
from selection.utils import *
from selection.candidate_generation import *


def candidate_indexes(queries, db_name):
    """
    :return: all potential indexes with width <= 2
    """
    db_connector = PostgresDatabaseConnector(db_name)
    cost_evaluation = CostEvaluation(db_connector)
    max_index_width = 2
    workload = Workload(queries)

    candidates = candidates_per_query(
        workload,
        max_index_width,
        candidate_generator=syntactically_relevant_indexes,
    )

    candidates, _ = get_utilized_indexes(workload, candidates, cost_evaluation)
    index_table_dict = indexes_by_table(candidates)
    for table in index_table_dict:
        for index1, index2 in itertools.permutations(index_table_dict[table], 2):
            merged_index = index_merge(index1, index2)
            if len(merged_index.columns) > max_index_width:
                new_columns = merged_index.columns[: max_index_width]
                merged_index = Index(new_columns)
            if merged_index not in candidates:
                cost_evaluation.estimate_size(merged_index)
                candidates.add(merged_index)

    return list(candidates)

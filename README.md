# RobustConstrainedTuning

## Folder Structure

~~~
.
├── selection                   # index selection algorithms and database connector
├── safeop                      # RL algorithms
├── robust_constrained_tuning   # Implementation of our approach
├── experiments                 # Experimental Evaluation
└── README.md 
~~~

## Requirements
- ```PostgreSQL with HypoPG extension ```
- ```psycopg2 ```
- ```Pytorch ```
- ```gym ```


## Usage
To run, you need database prepared, which is not included in this repo. You can use ```query_generator``` and ```table_generator``` in ```selection``` to generate database instance and the corresponding queries.
Other experimental data, including index maintenance cost, workload samples, etc, is inclued in ```experiments/tpcds```.

For easy usage, you can refer to the code in  ```experiments```.
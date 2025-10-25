# Instructions:
## Setup Requirements
### Environment setup
* We recommend using Conda or venv.

* Conda config with dependencies are listed in the gurobi_env_stl.yml

* With Conda is installed, create a new conda environment `gurobi_env_stl` by:

  * `conda env create -f gurobi_env_stl.yml`

### Obtain Gurobi academic license
* Our code uses Gurobi optimizer as the primary MIP solver.
* You do not have to install nor obtain the named academic license for our code to work (although it is free!)
  * Obtain Gurobi WLS academic license (also free!) - https://www.gurobi.com/features/academic-wls-license/
  * Follow the instructions until you can download the `gurobi.lic` license file locally on your computer.
* Tips for those running on Ubuntu:
  * You can set the enviroment variable: `export GRB_LICENSE_FILE=/PATH_TO_LICENSE_FILE/gurobi.lic`
    * e.g) `export GRB_LICENSE_FILE=~/Downloads/gurobi.lic`
  * We recommend adding this line to bashrc by: `echo 'export GRB_LICENSE_FILE=/PATH_TO_LICENSE_FILE/gurobi.lic' >> ~/.bashrc`  


## Run Experiments
* You can run experiment 2 with two scripts, both inside the /scripts directory.
* run_oracle_and_heuristics_experiment2.py
  * This runs the Oracle Baseline (B1 in the paper) and Baselines B2 and B3 (MILP hybrid, Closest Helper Heuristics).
* run_ours_experiment2.py
  * This runs Our decentralized MILP method.
* You can tweak the experiment parameters such as number of helpers, number of existing tasks and etc.
  * Modify `experiment_parameters.txt` file.  

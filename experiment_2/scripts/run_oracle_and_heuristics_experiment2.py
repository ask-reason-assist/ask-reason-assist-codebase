import os
import copy
from datetime import datetime
from utils import *
from world_generator import ExperimentWorldGenerator
from Oracle import *
from heuristics_baseline import run_heuristics_baseline
from detailed_logging_utils.oracle_detailed_saves_util import *
import numpy as np

def run_oracle_experiment_with_heuristics():
    """Run the Oracle-based experiment with heuristics comparison."""
    
    # Load parameters
    parameters = read_experiment_parameters()
    world_gen = ExperimentWorldGenerator(parameters)
    
    # Setup output
    timestamp = datetime.now().strftime("%Y-%m-%d")

    
    output_file = f'oracle_optimal_test_{parameters["SEED"]}.txt'
    # Uncomment if you want the .txt file that details each seed iteration results:
    
    # with open(output_file, 'w') as f:
    #     f.write(f"Oracle Optimal Test Results with Heuristics Comparison - Starting Seed: {parameters['SEED']}\n")
    #     f.write("=" * 80 + "\n\n")
    
    # Track metrics across all seeds
    oracle_makespan_diffs = []
    basic_makespan_diffs = []
    complex_makespan_diffs = []
    milp_makespan_diffs = []
    closest_helper_matches = []
    
    # Run experiments for each seed
    for seed_iter in range(parameters['SEED_ITERATION']):
        print(f"\n{'='*60}")
        print(f"ORACLE SEED ITERATION {seed_iter + 1}: SEED = {parameters['SEED'] + seed_iter}")
        print(f"{'='*60}")
        
        # Generate world and initial conditions
        exp_data = world_gen.generate_world_and_tasks(seed_iter)
        
        # Extract data
        world = exp_data['world']
        oracle = exp_data['oracle']
        helper_robots = exp_data['helper_robots']
        initial_tasks = exp_data['initial_tasks']
        help_task = exp_data['help_task']
        oracle_results = exp_data['oracle_results']
        
        print(f'Conflict Occurs at: {world.conflict_cell}')
        print(f"Initial Optimized Assignment: {oracle_results['assignment']}")
        print(f"Initial System Makespan (sum): {oracle_results['system_makespan']:.2f}")
        print(f"Initial System Cost (With Cost Function: {oracle.cost_function}): {oracle_results['system_cost']:.2f}")
        
        # Print robot categories and task assignments
        print_robot_and_task_categories(helper_robots, initial_tasks, help_task)
        
        # Re-optimize with help task (Oracle approach)
        updated_results = optimize_with_help_task(
            oracle, initial_tasks, help_task, helper_robots, oracle_results
        )
        
        # Calculate final metrics for Oracle
        # Calculate final metrics for Oracle
        oracle_metrics = calculate_oracle_metrics(oracle_results, updated_results, world, helper_robots, oracle, initial_tasks, help_task)        
        # Run heuristics baselines
        heuristics_results = run_heuristics_baseline(exp_data)
        
        # Compare closest helper with Oracle's choice
        closest_helper_is_optimal = (
            heuristics_results['closest_helper_id'] == oracle_metrics['help_agent_id']
        )
        closest_helper_matches.append(closest_helper_is_optimal)
        
        # Calculate makespan differences
        initial_makespan = oracle_results['system_makespan']
        oracle_diff = oracle_metrics['final_system_makespan'] - initial_makespan
        basic_diff = heuristics_results['basic_result']['final_system_makespan'] - initial_makespan
        complex_diff = heuristics_results['complex_result']['final_system_makespan'] - initial_makespan
        milp_diff = heuristics_results['milp_result']['final_system_makespan'] - initial_makespan
        
        oracle_makespan_diffs.append(oracle_diff)
        basic_makespan_diffs.append(basic_diff)
        complex_makespan_diffs.append(complex_diff)
        milp_makespan_diffs.append(milp_diff)
        
        # Write results to a txt file
        # write_oracle_results_with_heuristics(
        #     output_file, exp_data['seed'], seed_iter, oracle_results, updated_results, 
        #     oracle_metrics, heuristics_results, closest_helper_is_optimal, helper_robots, initial_tasks, help_task
        # )
        
        # Save results to CSV
        save_oracle_heuristics_csv(
            output_file, exp_data['seed'], seed_iter, oracle_results, oracle_metrics, 
            heuristics_results, closest_helper_is_optimal, helper_robots
        )
        #Uncomment if you want the detailed meterics:   
        # save_detailed_assignments_csv(
        #     output_file, exp_data['seed'], seed_iter, oracle_results, oracle_metrics, 
        #     heuristics_results, helper_robots, initial_tasks, help_task
        # )
        
        # Print summary
        print_oracle_summary_with_heuristics(
            exp_data['seed'], oracle_results, oracle_metrics, heuristics_results
        )
    
    # Write final summary statistics
    write_final_summary(
        output_file, oracle_makespan_diffs, basic_makespan_diffs, 
        complex_makespan_diffs, milp_makespan_diffs, closest_helper_matches,[]
    )
    
    print(f"\nOracle results with heuristics comparison saved to: {output_file}")

def print_robot_and_task_categories(helper_robots, initial_tasks, help_task):
    """Print robot categories and task distribution."""
    print("\n--- Robot Categories ---")
    robots_by_category = {}
    for robot in helper_robots:
        category = getattr(robot, 'category', 'Unknown')
        if category not in robots_by_category:
            robots_by_category[category] = []
        robots_by_category[category].append(robot.id)
    
    for category, robot_ids in robots_by_category.items():
        print(f"  {category}: Robots {robot_ids}")
    
    print("\n--- Task Categories ---")
    tasks_by_category = {}
    for task in initial_tasks:
        category = task.get('category', 'Unknown')
        if category not in tasks_by_category:
            tasks_by_category[category] = []
        tasks_by_category[category].append(task['name'])
    
    for category, task_names in tasks_by_category.items():
        print(f"  {category}: Tasks {task_names}")
    
    help_category = help_task.get('category', 'Any Category')
    print(f"  Help Task: {help_task['name']} (Category: {help_category})")

def optimize_with_help_task(oracle, initial_tasks, help_task, helper_robots, oracle_results):
    """Re-optimize Oracle assignment with the help task."""
    updated_task_list = initial_tasks + [help_task]
    
    # Try all approaches for the updated task list
    updated_reorder_assignment, updated_reorder_makespan_list = oracle.find_optimal_assignment_with_reordering(updated_task_list, helper_robots)
    updated_global_assignment, updated_global_makespan_list = oracle.find_optimal_assignment_global_reordering(updated_task_list, helper_robots)
    updated_simple_assignment, updated_simple_makespan_list = oracle.find_optimal_assignment_simple_reordering(updated_task_list, helper_robots)
    
    # NEW: Exact 2-b logic for all agents
    exact_2b_result = oracle.find_optimal_assignment_exact_2b_all_agents(updated_task_list, helper_robots, oracle_results)

    # CRITICAL FIX: Calculate system makespan consistently for ALL methods
    # For traditional methods: sum of helper makespans + requester makespan
    requester_base_makespan = np.mean(oracle_results['makespan_list'])
    
    # Method 1: Reorder
    updated_reorder_system_makespan = sum(updated_reorder_makespan_list)
    # Calculate help render time for reorder method
    help_render_time_reorder = 0
    help_task_name = help_task['name']
    for robot in helper_robots:
        if robot.id in updated_reorder_assignment and help_task_name in updated_reorder_assignment[robot.id]:
            # Calculate help render time for this robot
            assigned_tasks = [task for task in updated_task_list if task['name'] in updated_reorder_assignment[robot.id]]
            current_pos = robot.initial_pos
            for task in assigned_tasks:
                if task['name'] == help_task_name:
                    help_render_time_reorder = oracle.get_path_cost(current_pos, task['pickup'])
                    break
                help_render_time_reorder += oracle.get_path_cost(current_pos, task['pickup'])
                help_render_time_reorder += oracle.get_path_cost(task['pickup'], task['dropoff'])
                current_pos = task['dropoff']
            break
    final_reorder_makespan = updated_reorder_system_makespan + round(requester_base_makespan + help_render_time_reorder)
    
    # Method 2: Global
    updated_global_system_makespan = sum(updated_global_makespan_list)
    # Calculate help render time for global method
    help_render_time_global = 0
    for robot in helper_robots:
        if robot.id in updated_global_assignment and help_task_name in updated_global_assignment[robot.id]:
            # Calculate help render time for this robot
            assigned_tasks = [task for task in updated_task_list if task['name'] in updated_global_assignment[robot.id]]
            current_pos = robot.initial_pos
            for task in assigned_tasks:
                if task['name'] == help_task_name:
                    help_render_time_global = oracle.get_path_cost(current_pos, task['pickup'])
                    break
                help_render_time_global += oracle.get_path_cost(current_pos, task['pickup'])
                help_render_time_global += oracle.get_path_cost(task['pickup'], task['dropoff'])
                current_pos = task['dropoff']
            break
    final_global_makespan = updated_global_system_makespan + round(requester_base_makespan + help_render_time_global)
    
    # Method 3: Simple
    updated_simple_system_makespan = sum(updated_simple_makespan_list)
    # Calculate help render time for simple method
    help_render_time_simple = 0
    for robot in helper_robots:
        if robot.id in updated_simple_assignment and help_task_name in updated_simple_assignment[robot.id]:
            # Calculate help render time for this robot
            assigned_tasks = [task for task in updated_task_list if task['name'] in updated_simple_assignment[robot.id]]
            current_pos = robot.initial_pos
            for task in assigned_tasks:
                if task['name'] == help_task_name:
                    help_render_time_simple = oracle.get_path_cost(current_pos, task['pickup'])
                    break
                help_render_time_simple += oracle.get_path_cost(current_pos, task['pickup'])
                help_render_time_simple += oracle.get_path_cost(task['pickup'], task['dropoff'])
                current_pos = task['dropoff']
            break
    final_simple_makespan = updated_simple_system_makespan + round(requester_base_makespan + help_render_time_simple)
    
    # Method 4: Exact 2-b (already calculated consistently)
    if exact_2b_result and 'final_system_makespan' in exact_2b_result:
        final_exact_2b_makespan = exact_2b_result['final_system_makespan']
        updated_exact_2b_assignment = exact_2b_result['assignment']
        updated_exact_2b_makespan_list = exact_2b_result['makespans']
    else:
        final_exact_2b_makespan = float('inf')
        updated_exact_2b_assignment = {}
        updated_exact_2b_makespan_list = []

    print(f"Method comparison (final system makespans):")
    print(f"  Reorder: {final_reorder_makespan}")
    print(f"  Global: {final_global_makespan}")
    print(f"  Simple: {final_simple_makespan}")
    print(f"  Exact 2-b all agents: {final_exact_2b_makespan}")

    # FIXED: Compare apples to apples - all final system makespans
    costs = [
        (final_reorder_makespan, updated_reorder_assignment, updated_reorder_makespan_list, "reorder"),
        (final_global_makespan, updated_global_assignment, updated_global_makespan_list, "global"),
        (final_simple_makespan, updated_simple_assignment, updated_simple_makespan_list, "simple"),
        (final_exact_2b_makespan, updated_exact_2b_assignment, updated_exact_2b_makespan_list, "exact_2b_all_agents")
    ]
    
    best_cost, updated_assignment, updated_makespan_list, best_method = min(costs, key=lambda x: x[0])
    
    print(f"Best Oracle method: {best_method} with final system makespan {best_cost}")
    
    return {
        'assignment': updated_assignment,
        'makespan_list': updated_makespan_list,
        'system_cost': best_cost,  # This is now the final system makespan for all methods
        'system_makespan': best_cost,  # Consistent naming
        'best_method': best_method
    }
def calculate_oracle_metrics(oracle_results, updated_results, world, helper_robots, oracle, initial_tasks, help_task):
    """Calculate Oracle experiment metrics."""
    # Find help agent and render time
    help_agent_id = None
    help_render_time = 0
    
    for agent_id, tasks in updated_results['assignment'].items():
        if any('HelpTask-ClearObstruction' in t for t in tasks):
            help_agent_id = agent_id
            break
    
    if help_agent_id is not None:
        help_agent = next(a for a in helper_robots if a.id == help_agent_id)
        
        # Calculate help render time (th) - time to reach conflict cell
        # Get the agent's full path for the updated assignment
        updated_path = get_agent_full_path(help_agent, updated_results['assignment'][help_agent_id], initial_tasks + [help_task], oracle)        
        # Count transitions until reaching the conflict cell (help site)
        help_render_time = 0
        for i, position in enumerate(updated_path):
            if position == world.conflict_cell:
                help_render_time = i
                break
        
        print(f"Help agent {help_agent_id} reaches conflict at time {help_render_time}")
    
    # FIXED: Use the Oracle's chosen result directly - don't recalculate!
    # The updated_results already contains the final system makespan from the chosen method
    final_system_makespan = updated_results['system_makespan']  # This is already the complete system makespan
    
    # Calculate system cost using the Oracle's makespan list
    # For exact_2b method, this might need special handling
    if updated_results.get('best_method') == 'exact_2b_all_agents':
        # For exact_2b, the system_cost is the final_system_makespan
        final_system_cost = final_system_makespan
    else:
        # For other methods, calculate using standard cost function
        updated_makespan_list_with_requester = copy.deepcopy(updated_results['makespan_list'])
        requester_makespan = oracle_results['requester_makespan'] + help_render_time
        updated_makespan_list_with_requester.append(round(requester_makespan))
        final_system_cost = compute_cost(updated_makespan_list_with_requester, cost_function='mixed')
    
    help_cost = final_system_cost - oracle_results['system_cost']
    
    return {
        'help_agent_id': help_agent_id,
        'help_render_time': help_render_time,
        'final_system_makespan': final_system_makespan,  # Use Oracle's result directly
        'final_system_cost': final_system_cost,
        'help_cost': help_cost
    }



if __name__ == "__main__":
    run_oracle_experiment_with_heuristics()
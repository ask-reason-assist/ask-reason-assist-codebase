import os
import copy
import numpy as np
from datetime import datetime
from utils import (
    read_experiment_parameters, compute_cost, parse_formula_string,
    get_adjacent_free_cell, count_transitions_until_target, simple_th_and_tstar
)
from world_generator import ExperimentWorldGenerator

def run_milp_experiment():
    """Run the MILP-based experiment."""
    
    # Load parameters
    parameters = read_experiment_parameters()
    world_gen = ExperimentWorldGenerator(parameters)
    
  
    
    output_file = f'ours_optimal_test_{parameters["SEED"]}.txt'
    with open(output_file, 'w') as f:
        f.write(f"Ours Optimal Test Results - Starting Seed: {parameters['SEED']}\n")
        f.write("=" * 60 + "\n\n")
    
    # Run experiments for each seed
    for seed_iter in range(parameters['SEED_ITERATION']):
        print(f"\n{'='*60}")
        print(f"SEED ITERATION {seed_iter + 1}: SEED = {parameters['SEED'] + seed_iter}")
        print(f"{'='*60}")
        
        # Generate world and initial conditions
        exp_data = world_gen.generate_world_and_tasks(seed_iter)
        
        # Setup MILP agents
        setup_milp_agents(exp_data)
        
        # Run MILP optimization
        milp_results = run_milp_optimization(exp_data, parameters)
        
        # Calculate metrics and write results
        write_milp_results(output_file, exp_data, milp_results, seed_iter)
        save_milp_results_to_csv(output_file, exp_data, milp_results, seed_iter)
        # save_individual_helper_csv(output_file, exp_data, milp_results, seed_iter)
    
    print(f"\nResults saved to: {output_file}")

def setup_milp_agents(exp_data):
    """Setup MILP agents with LTL specifications."""
    world = exp_data['world']
    helper_robots = exp_data['helper_robots']
    initial_tasks = exp_data['initial_tasks']
    oracle_results = exp_data['oracle_results']
    
    for helper in helper_robots:
        ltl_locs = {}
        assigned_task_names = oracle_results['assignment'].get(helper.id, [])
        assigned_tasks = [task for task in initial_tasks if task['name'] in assigned_task_names]
        spec_clauses = []
        
        for idx, loc in enumerate(assigned_tasks):
            i, j = loc['pickup'][0], loc['pickup'][1]
            pickup_key = f'pickup_{idx}_{i}_{j}'
            ltl_locs[pickup_key] = loc['pickup']
            
            i, j = loc['dropoff'][0], loc['dropoff'][1]
            dropoff_key = f'dropoff_{idx}_{i}_{j}'
            ltl_locs[dropoff_key] = loc['dropoff']
            
            spec_clauses.append(f"F(IMPLIES_NEXT({pickup_key},{dropoff_key}))")
        
        helper.ltl_locs = ltl_locs
        helper.llm_stl_expression_help = "&".join(spec_clauses)
        helper.ltl_locs['help_site'] = world.conflict_cell
        helper.ltl_locs['help_site_drop'] = get_adjacent_free_cell(world, world.conflict_cell)
        helper.init_default_ltl_locations(helper.ltl_locs)

def run_milp_optimization(exp_data, parameters):
    """Run MILP optimization for all helpers."""
    helper_robots = exp_data['helper_robots']
    oracle_results = exp_data['oracle_results']
    world = exp_data['world']
    
    original_makespans = []
    help_costs = []
    updated_makespans = []
    helper_info = []
    
    def find_feasible_solution(helper, formula, max_attempts=3):
        for attempt in range(max_attempts):
            # Check T threshold before solving
            if helper.T > 45:
                print(f"Helper {helper.id}: T={helper.T} exceeds threshold of 45, marking as infeasible")
                return [3, None, float('inf')]  # Return infeasible status
                
            results = helper.solve_physical_visits(formula)
            if results[0] == 2:  # Feasible solution found
                return results
            else:
                helper.T = int(helper.T * 1.2)
                if hasattr(helper, '_task_states'):
                    helper._task_states.clear()
                if hasattr(helper, '_task_counter'):
                    helper._task_counter = 0
                helper._build_model()
                helper.init_default_ltl_locations(helper.ltl_locs)
                print(f"Infeasible solution for helper_{helper.id} with T={helper.T} (attempt {attempt + 1})")
        return results
    
    if parameters.get('TRUST_ORACLE_INIT_MILP', False):
        # Use Oracle results to set T and skip original solving
        for helper in helper_robots:
            original_formula_help = parse_formula_string(helper.llm_stl_expression_help)
            if original_formula_help is None:
                original_formula_help = ""
            
            if helper.id in oracle_results['makespan_dict']:
                original_makespan = oracle_results['makespan_dict'][helper.id]
                original_makespans.append(original_makespan)
                
                # Set appropriate T
                if original_makespan < 1.2 * np.mean(oracle_results['makespan_list']):
                    helper.T = int(1.2 * np.mean(oracle_results['makespan_list']))
                else:
                    helper.T = int(1.1 * original_makespan)
                
                # Check T threshold before proceeding
                if helper.T > 45:
                    print(f"Helper {helper.id}: Initial T={helper.T} exceeds threshold, marking as infeasible")
                    help_costs.append(float('inf'))
                    updated_makespans.append(float('inf'))
                    helper_info.append({
                        'helper_id': helper.id,
                        'original_makespan': original_makespan,
                        'updated_makespan': float('inf'),
                        'help_cost': float('inf'),
                        't_star': float('inf'),
                        'th': float('inf'),
                        'feasible': False
                    })
                    continue
                
                # Add help task to formula
                if original_formula_help == "":
                    helper.llm_stl_expression_help = "F(IMPLIES_NEXT(help_site,help_site_drop))"
                else:
                    helper.llm_stl_expression_help = f"{helper.llm_stl_expression_help}&F(IMPLIES_NEXT(help_site,help_site_drop))"
                
                updated_formula_help = parse_formula_string(helper.llm_stl_expression_help)
                
                # Reset and rebuild
                if hasattr(helper, '_task_counter'):
                    helper._task_counter = 0
                if hasattr(helper, '_task_states'):
                    helper._task_states.clear()
                helper._build_model()
                helper.init_default_ltl_locations(helper.ltl_locs)
                
                # Solve with help task
                updated_results_help = find_feasible_solution(helper, updated_formula_help)
                print(f"Helper {helper.id} - Original makespan: {original_makespan}, Setting T to {helper.T}")
                if updated_results_help[0] == 2:
                    updated_path = updated_results_help[1]
                    th = count_transitions_until_target(updated_path, world.conflict_cell)
                    tstar = updated_results_help[2] - original_makespan
                    cost = th + tstar
                    
                    help_costs.append(cost)
                    updated_makespan = updated_results_help[2]
                    updated_makespans.append(updated_makespan)
                    
                    helper_info.append({
                        'helper_id': helper.id,
                        'original_makespan': original_makespan,
                        'updated_makespan': updated_makespan,
                        'help_cost': cost,
                        't_star': tstar,
                        'th': th,
                        'feasible': True
                    })
                    print(f"Helper {helper.id} - Cost: {cost}, Updated makespan: {updated_makespan}")

                else:
                    help_costs.append(float('inf'))
                    updated_makespans.append(float('inf'))
                    helper_info.append({
                        'helper_id': helper.id,
                        'original_makespan': original_makespan,
                        'updated_makespan': float('inf'),
                        'help_cost': float('inf'),
                        't_star': float('inf'),
                        'th': float('inf'),
                        'feasible': False
                    })
    else:
        # Full MILP solving (original approach)
        for helper in helper_robots:
            original_formula_help = parse_formula_string(helper.llm_stl_expression_help)
            if original_formula_help is None:
                original_formula_help = ""
                original_results_help = [2, None, 0, None, None]  # manually set to feasible for empty formula
                
            else:
                original_results_help = find_feasible_solution(helper, original_formula_help)
        
            if original_results_help[0] == 2:  # GRB.OPTIMAL
                original_makespan = original_results_help[2]
                original_makespans.append(original_makespan)
                print(f"Helper {helper.id} - Original makespan: {original_makespan}")

                # ADD IMPLIES_NEXT for help_site -> help_site_drop
                if original_formula_help == "":
                    helper.llm_stl_expression_help = "F(IMPLIES_NEXT(help_site,help_site_drop))"
                else:
                    helper.llm_stl_expression_help = f"{helper.llm_stl_expression_help}&F(IMPLIES_NEXT(help_site,help_site_drop))"
                
                updated_formula_help = parse_formula_string(helper.llm_stl_expression_help)
                
                # CRITICAL: Reset task counter before solving with new formula
                if hasattr(helper, '_task_counter'):
                    helper._task_counter = 0
                if hasattr(helper, '_task_states'):
                    helper._task_states.clear()
                
                # Rebuild model to ensure clean state
                helper._build_model()
                helper.init_default_ltl_locations(helper.ltl_locs)
                
                updated_results_help = find_feasible_solution(helper, updated_formula_help)
                
                if updated_results_help[0] == 2:
                    th, tstar, cost = simple_th_and_tstar(world, original_results_help, updated_results_help)
                    help_costs.append(cost)
                    updated_makespan = updated_results_help[2]
                    updated_makespans.append(updated_makespan)
                    
                    # Store helper info for tracking
                    helper_info.append({
                        'helper_id': helper.id,
                        'original_makespan': original_makespan,
                        'updated_makespan': updated_makespan,
                        'help_cost': cost,
                        't_star': tstar,
                        'th': th,
                        'feasible': True
                    })
                    
                    print(f"Helper {helper.id} - Cost: {cost}, Updated makespan: {updated_makespan}")
                else:
                    print(f"Helper {helper.id} - No feasible solution after adding help task (T exceeded threshold or infeasible)")
                    help_costs.append(float('inf'))  # Mark as infeasible
                    updated_makespans.append(float('inf'))
                    
                    # Store helper info for infeasible case
                    helper_info.append({
                        'helper_id': helper.id,
                        'original_makespan': original_makespan,
                        'updated_makespan': float('inf'),
                        'help_cost': float('inf'),
                        't_star': float('inf'),
                        'th': float('inf'),
                        'feasible': False
                    })
            else:
                print(f"Helper {helper.id} - No feasible original solution (T exceeded threshold or infeasible)")
                original_makespans.append(float('inf'))
                help_costs.append(float('inf'))
                updated_makespans.append(float('inf'))
                
                # Store helper info for no original solution case
                helper_info.append({
                    'helper_id': helper.id,
                    'original_makespan': float('inf'),
                    'updated_makespan': float('inf'),
                    'help_cost': float('inf'),
                    't_star': float('inf'),
                    'th': float('inf'),
                    'feasible': False
                })
    
    return {
        'original_makespans': original_makespans,
        'help_costs': help_costs,
        'updated_makespans': updated_makespans,
        'helper_info': helper_info
    }

def write_milp_results(output_file, exp_data, milp_results, seed_iter):
    """Write MILP results to file."""
    # Calculate final metrics
    oracle_results = exp_data['oracle_results']
    original_makespans = milp_results['original_makespans']
    help_costs = milp_results['help_costs']
    helper_info = milp_results['helper_info']
    helper_robots = exp_data['helper_robots']
    initial_tasks = exp_data['initial_tasks']
    help_task = exp_data['help_task']
    
    # Calculate system costs and best helper
    if original_makespans and all(m != float('inf') for m in original_makespans):
        sum_original_makespan = sum(original_makespans)
        original_makespan_list_with_requester = copy.deepcopy(original_makespans)
        requester_makespan = oracle_results['requester_makespan']
        original_makespan_list_with_requester.append(round(requester_makespan))
        sum_original_makespan += round(requester_makespan)
        original_system_cost = compute_cost(original_makespan_list_with_requester, cost_function=exp_data['oracle'].cost_function)
        
        if help_costs and any(c != float('inf') for c in help_costs):
            min_help_cost = min([c for c in help_costs if c != float('inf')])
            best_helper_id = None
            true_t_h = 0
            best_helper_updated_makespan = 0
            for info in helper_info:
                if info['help_cost'] == min_help_cost and info['feasible']:
                    best_helper_id = info['helper_id']
                    true_t_h = info['th']
                    best_helper_updated_makespan = info['updated_makespan']
                    break
            
            updated_requester_makespan = round(requester_makespan) + true_t_h
            
            # Calculate final system makespan: only the best helper's makespan is updated
            # All other helpers keep their original makespans
            final_helper_makespans = []
            for i, original_makespan in enumerate(original_makespans):
                helper_id = helper_robots[i].id
                if helper_id == best_helper_id:
                    final_helper_makespans.append(best_helper_updated_makespan)
                else:
                    final_helper_makespans.append(original_makespan)
            
            final_system_makespan = sum(final_helper_makespans) + round(updated_requester_makespan)
            
            # For system cost calculation, use the final makespans
            updated_makespan_list_with_requester = final_helper_makespans + [round(updated_requester_makespan)]
            updated_system_cost = compute_cost(updated_makespan_list_with_requester, cost_function=exp_data['oracle'].cost_function)
        else:
            min_help_cost = float('inf')
            updated_system_cost = float('inf')
            system_updated_makespan = float('inf')
            best_helper_id = None
            final_system_makespan = float('inf')
    else:
        sum_original_makespan = float('inf')
        min_help_cost = float('inf')
        updated_system_cost = float('inf')
        system_updated_makespan = float('inf')
        best_helper_id = None
        original_system_cost = float('inf')
        final_system_makespan = float('inf')
    
    # Write to file
    
    with open(output_file, 'a') as f:
        f.write(f"Seed Iteration {seed_iter + 1} (Seed = {exp_data['seed']}):\n")
        f.write(f"=================SUMMARY SEED: {exp_data['seed']}=================\n\n")
        
        # Write robot and task category information (matching Oracle format)
        f.write(f"--- ROBOT CATEGORIES ---\n")
        robots_by_category = {}
        for robot in helper_robots:
            category = getattr(robot, 'category', 'Unknown')
            if category not in robots_by_category:
                robots_by_category[category] = []
            robots_by_category[category].append(robot.id)
        
        for category, robot_ids in robots_by_category.items():
            f.write(f"  {category}: Robots {robot_ids}\n")
        
        f.write(f"\n--- TASK CATEGORIES ---\n")
        tasks_by_category = {}
        for task in initial_tasks:
            category = task.get('category', 'Unknown')
            if category not in tasks_by_category:
                tasks_by_category[category] = []
            tasks_by_category[category].append(task['name'])
        
        for category, task_names in tasks_by_category.items():
            f.write(f"  {category}: Tasks {task_names}\n")
        
        help_category = help_task.get('category', 'Any Category')
        f.write(f"  Help Task: {help_task['name']} (Category: {help_category})\n\n")
        
        # Main metrics (matching Oracle format)
        f.write(f"  Original System Cost: {oracle_results['system_cost']}\n")
        f.write(f"  MILP Updated System Cost: {oracle_results['system_makespan']}\n")
        f.write(f"  MILP Final System Makespan: {final_system_makespan}\n")

        f.write(f"  MILP Help Cost (Delta): {oracle_results['system_makespan']-final_system_makespan}\n")
        f.write(f"  MILP Help Agent ID: {best_helper_id}")
        # Now with the best helper_id, extract the help render time (th) and tstar
        if best_helper_id is not None:
            help_info = next((info for info in helper_info if info['helper_id'] == best_helper_id), None)
            if help_info:
                true_t_h = help_info['th']
                t_star = help_info['t_star']
                f.write(f" (t_h={true_t_h}, t_star={t_star})")
            else:
                f.write(" (No feasible solution found)")
        else:
            f.write(" (No feasible solution found)")
        # Add category information for MILP's choice
        if best_helper_id is not None:
            help_robot = next((r for r in helper_robots if r.id == best_helper_id), None)
            if help_robot:
                help_robot_category = getattr(help_robot, 'category', 'Unknown')
                f.write(f" ({help_robot_category})")
        f.write(f"\n")
        
        f.write(f"=================END SUMMARY SEED:{exp_data['seed']}=================\n\n")
        f.write(f"=================Details SEED: {exp_data['seed']}=================\n\n")
        f.write(f"  Initial System Makespan: {oracle_results['system_makespan']}\n")
        f.write(f"  Individual Original Makespans (Oracle): {oracle_results['makespan_list']}\n")
        f.write(f"  Individual Original Makespans (MILP): {original_makespans}\n")
        f.write(f"  Individual Updated Makespans (MILP): {milp_results['updated_makespans']}\n")
        f.write(f"  MILP Help Render Time: {true_t_h if best_helper_id else 0}\n")
        f.write(f"  Initial Task Assignment (Oracle): {oracle_results['assignment']}\n")

        
        # Show MILP assignment details
        if best_helper_id is not None:
            milp_assignment = {}
            for info in helper_info:
                if info['feasible']:
                    assigned_task_names = oracle_results['assignment'].get(info['helper_id'], [])
                    if info['helper_id'] == best_helper_id:
                        assigned_task_names.append(help_task['name'])
                    milp_assignment[info['helper_id']] = assigned_task_names
            f.write(f"  MILP Assignment (with help): {milp_assignment}\n")
        else:
            f.write(f"  MILP Assignment: No feasible solution\n")
        
        # Individual helper details
        f.write(f"\n--- INDIVIDUAL HELPER DETAILS ---\n")
        for info in helper_info:
            f.write(f"  Helper {info['helper_id']}: ")
            f.write(f"Original={info['original_makespan']}, ")
            f.write(f"Updated={info['updated_makespan']}, ")
            f.write(f"Cost={info['help_cost']}, ")
            f.write(f"t_h={info['th']}, ")
            f.write(f"t_star={info['t_star']}, ")
            f.write(f"Feasible={info['feasible']}\n")
        
        f.write(f"=================END Details SEED:{exp_data['seed']}=================\n\n")

def save_milp_results_to_csv(output_file, exp_data, milp_results, seed_iter):
    """Save MILP results to CSV file for analysis."""
    import csv
    import os
    
    # Generate CSV filename based on the output_file
    csv_filename = output_file.replace('.txt', '_results.csv')
    
    # Check if this is the first seed iteration to write headers
    write_headers = not os.path.exists(csv_filename)
    
    oracle_results = exp_data['oracle_results']
    helper_info = milp_results['helper_info']
    help_costs = milp_results['help_costs']
    
    # Calculate best helper metrics
    if help_costs and any(c != float('inf') for c in help_costs):
        min_help_cost = min([c for c in help_costs if c != float('inf')])
        best_helper_info = next((info for info in helper_info if info['help_cost'] == min_help_cost and info['feasible']), None)
        best_helper_id = best_helper_info['helper_id'] if best_helper_info else None
        best_help_cost = min_help_cost if best_helper_info else float('inf')
        best_th = best_helper_info['th'] if best_helper_info else float('inf')
        best_tstar = best_helper_info['t_star'] if best_helper_info else float('inf')
        best_updated_makespan = best_helper_info['updated_makespan'] if best_helper_info else float('inf')
        
        # Calculate final system makespan
        if best_helper_info:
            final_helper_makespans = []
            for i, original_makespan in enumerate(milp_results['original_makespans']):
                helper_id = exp_data['helper_robots'][i].id
                if helper_id == best_helper_id:
                    final_helper_makespans.append(best_updated_makespan)
                else:
                    final_helper_makespans.append(original_makespan)
            
            requester_makespan = oracle_results['requester_makespan']
            updated_requester_makespan = round(requester_makespan) + best_th
            final_system_makespan = sum(final_helper_makespans) + updated_requester_makespan
        else:
            final_system_makespan = float('inf')
    else:
        best_helper_id = None
        best_help_cost = float('inf')
        best_th = float('inf')
        best_tstar = float('inf')
        best_updated_makespan = float('inf')
        final_system_makespan = float('inf')
    
    # Calculate help cost delta (improvement in system makespan)
    original_system_makespan = oracle_results['system_makespan']
    help_cost_delta = final_system_makespan - original_system_makespan if final_system_makespan != float('inf') else float('inf')
    
    with open(csv_filename, 'a', newline='') as csvfile:
        fieldnames = [
            'seed', 'seed_iteration',
            'original_system_makespan', 'final_system_makespan', 'help_cost_delta',
            'best_helper_id', 'best_help_cost', 'best_th', 'best_tstar',
            'best_original_makespan', 'best_updated_makespan',
            'oracle_assignment', 'milp_feasible',
            'num_helpers', 'num_feasible_helpers'
        ]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if write_headers:
            writer.writeheader()
        
        # Count feasible helpers
        num_feasible_helpers = sum(1 for info in helper_info if info['feasible'])
        
        # Create MILP assignment string for best helper
        milp_assignment_str = ""
        if best_helper_id is not None:
            milp_assignment = {}
            for info in helper_info:
                if info['feasible']:
                    assigned_task_names = oracle_results['assignment'].get(info['helper_id'], [])
                    if info['helper_id'] == best_helper_id:
                        assigned_task_names = assigned_task_names + [exp_data['help_task']['name']]
                    milp_assignment[info['helper_id']] = assigned_task_names
            milp_assignment_str = str(milp_assignment)
        
        writer.writerow({
            'seed': exp_data['seed'],
            'seed_iteration': seed_iter + 1,
            'original_system_makespan': original_system_makespan,
            'final_system_makespan': final_system_makespan if final_system_makespan != float('inf') else 'inf',
            'help_cost_delta': help_cost_delta if help_cost_delta != float('inf') else 'inf',
            'best_helper_id': best_helper_id if best_helper_id is not None else 'None',
            'best_help_cost': best_help_cost if best_help_cost != float('inf') else 'inf',
            'best_th': best_th if best_th != float('inf') else 'inf',
            'best_tstar': best_tstar if best_tstar != float('inf') else 'inf',
            'best_original_makespan': best_helper_info['original_makespan'] if best_helper_info and best_helper_info['original_makespan'] != float('inf') else 'inf',
            'best_updated_makespan': best_updated_makespan if best_updated_makespan != float('inf') else 'inf',
            'oracle_assignment': str(oracle_results['assignment']),
            'milp_feasible': best_helper_id is not None,
            'num_helpers': len(exp_data['helper_robots']),
            'num_feasible_helpers': num_feasible_helpers
        })
    
    print(f"Results saved to CSV: {csv_filename}")

def save_individual_helper_csv(output_file, exp_data, milp_results, seed_iter):
    """Save individual helper results to a separate CSV file."""
    import csv
    import os
    
    # Generate CSV filename for individual helpers
    csv_filename = output_file.replace('.txt', '_individual_helpers.csv')
    
    # Check if this is the first seed iteration to write headers
    write_headers = not os.path.exists(csv_filename)
    
    oracle_results = exp_data['oracle_results']
    helper_info = milp_results['helper_info']
    helper_robots = exp_data['helper_robots']
    
    with open(csv_filename, 'a', newline='') as csvfile:
        fieldnames = [
            'seed', 'seed_iteration', 'helper_id', 'helper_category',
            'original_makespan', 'updated_makespan', 'help_cost',
            'th', 'tstar', 'feasible', 'assigned_tasks', 'num_assigned_tasks'
        ]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if write_headers:
            writer.writeheader()
        
        # Write one row per helper
        for info in helper_info:
            helper = next((r for r in helper_robots if r.id == info['helper_id']), None)
            helper_category = getattr(helper, 'category', 'Unknown') if helper else 'Unknown'
            assigned_tasks = oracle_results['assignment'].get(info['helper_id'], [])
            
            writer.writerow({
                'seed': exp_data['seed'],
                'seed_iteration': seed_iter + 1,
                'helper_id': info['helper_id'],
                'helper_category': helper_category,
                'original_makespan': info['original_makespan'] if info['original_makespan'] != float('inf') else 'inf',
                'updated_makespan': info['updated_makespan'] if info['updated_makespan'] != float('inf') else 'inf',
                'help_cost': info['help_cost'] if info['help_cost'] != float('inf') else 'inf',
                'th': info['th'] if info['th'] != float('inf') else 'inf',
                'tstar': info['t_star'] if info['t_star'] != float('inf') else 'inf',
                'feasible': info['feasible'],
                'assigned_tasks': str(assigned_tasks),
                'num_assigned_tasks': len(assigned_tasks)
            })
    
    print(f"Individual helper results saved to CSV: {csv_filename}")

if __name__ == "__main__":
    run_milp_experiment()
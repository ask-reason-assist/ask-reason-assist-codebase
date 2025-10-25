


def write_oracle_results_with_heuristics(output_file, seed, seed_iter, oracle_results, updated_results, 
                                        oracle_metrics, heuristics_results, closest_helper_is_optimal, 
                                        helper_robots, initial_tasks, help_task):
    """Write Oracle results with heuristics comparison to file."""
    with open(output_file, 'a') as f:
        f.write(f"Seed Iteration {seed_iter + 1} (Seed = {seed}):\n")
        f.write(f"=================SUMMARY SEED: {seed}=================\n\n")
        
        # Write robot and task category information
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
        
        f.write(f"  Original System Cost: {oracle_results['system_cost']}\n")
        f.write(f"  Oracle Updated System Cost: {oracle_metrics['final_system_cost']}\n")
        f.write(f"  Oracle Help Cost (Delta): {oracle_metrics['help_cost']}\n")
        f.write(f"  Oracle Final System Makespan: {oracle_metrics['final_system_makespan']}\n")
        f.write(f"  Oracle Help Agent ID: {oracle_metrics['help_agent_id']}")
        
        # Add category information for Oracle's choice
        if oracle_metrics['help_agent_id'] is not None:
            help_robot = next((r for r in helper_robots if r.id == oracle_metrics['help_agent_id']), None)
            if help_robot:
                help_robot_category = getattr(help_robot, 'category', 'Unknown')
                f.write(f" ({help_robot_category})")
        f.write(f"\n")
        
        f.write(f"\n--- HEURISTICS COMPARISON ---\n")
        f.write(f"  Closest Helper ID: {heuristics_results['closest_helper_id']}")
        
        # Add category information for closest helper
        closest_robot = next((r for r in helper_robots if r.id == heuristics_results['closest_helper_id']), None)
        if closest_robot:
            closest_robot_category = getattr(closest_robot, 'category', 'Unknown')
            f.write(f" ({closest_robot_category})")
        f.write(f"\n")
        
        f.write(f"  Closest Helper Distance: {heuristics_results['closest_distance']}\n")
        f.write(f"  Closest Helper is Oracle Optimal: {closest_helper_is_optimal}\n")
        f.write(f"\n  Heuristic 2-a (Basic Insertion) Final Makespan: {heuristics_results['basic_result']['final_system_makespan']}\n")
        f.write(f"  Heuristic 2-b (Complex Shuffling) Final Makespan: {heuristics_results['complex_result']['final_system_makespan']}\n")
        f.write(f"  Heuristic 2-c (MILP Baseline) Final Makespan: {heuristics_results['milp_result']['final_system_makespan']}\n")
        f.write(f"  Heuristic 2-c Feasible: {heuristics_results['milp_result']['feasible']}\n")
        f.write(f"=================END SUMMARY SEED:{seed}=================\n\n")
        f.write(f"=================Details SEED: {seed}=================\n\n")
        f.write(f"  Initial System Makespan: {oracle_results['system_makespan']}\n")
        f.write(f"  Individual Original Makespans: {oracle_results['makespan_list']}\n")
        f.write(f"  Individual Updated Makespans (Oracle): {updated_results['makespan_list']}\n")
        f.write(f"  Oracle Help Render Time: {oracle_metrics['help_render_time']}\n")
        f.write(f"  Initial Task Assignment: {oracle_results['assignment']}\n")
        f.write(f"  Updated Task Assignment (Oracle): {updated_results['assignment']}\n")
        f.write(f"\n  Heuristic 2-a Assignment: {heuristics_results['basic_result']['assignment']}\n")
        f.write(f"  Heuristic 2-a Makespans: {heuristics_results['basic_result']['makespans']}\n")
        f.write(f"  Heuristic 2-a Help Render Time: {heuristics_results['basic_result']['help_render_time']}\n")
        f.write(f"\n  Heuristic 2-b Assignment: {heuristics_results['complex_result']['assignment']}\n")
        f.write(f"  Heuristic 2-b Makespans: {heuristics_results['complex_result']['makespans']}\n")
        f.write(f"  Heuristic 2-b Help Render Time: {heuristics_results['complex_result']['help_render_time']}\n")
        f.write(f"\n  Heuristic 2-c Makespans: {heuristics_results['milp_result']['makespans']}\n")
        f.write(f"  Heuristic 2-c Help Render Time: {heuristics_results['milp_result']['help_render_time']}\n")
        f.write(f"=================END Details SEED:{seed}=================\n\n")

def print_oracle_summary_with_heuristics(seed, oracle_results, oracle_metrics, heuristics_results):
    """Print Oracle experiment summary with heuristics."""
    print(f"\nSeed {seed} Results:")
    print(f"  Original System Makespan: {oracle_results['system_makespan']}")
    print(f"  Oracle Final Makespan: {oracle_metrics['final_system_makespan']}")
    print(f"  Heuristic 2-a Final Makespan: {heuristics_results['basic_result']['final_system_makespan']}")
    print(f"  Heuristic 2-b Final Makespan: {heuristics_results['complex_result']['final_system_makespan']}")
    print(f"  Heuristic 2-c Final Makespan: {heuristics_results['milp_result']['final_system_makespan']}")
    print(f"  Closest Helper: {heuristics_results['closest_helper_id']}, Oracle Choice: {oracle_metrics['help_agent_id']}")

def write_final_summary(output_file, oracle_diffs, basic_diffs, complex_diffs, milp_diffs, closest_matches, skipped_seeds):
    """Write final summary statistics to file."""
    import numpy as np
    
    with open(output_file, 'a') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"FINAL SUMMARY STATISTICS ACROSS ALL SEEDS\n")
        f.write(f"{'='*80}\n\n")
        
        # Report skipped seeds
        if skipped_seeds:
            f.write(f"SKIPPED SEED ITERATIONS: {skipped_seeds}\n")
            f.write(f"Reason: T exceeded threshold of 35 in 2-c approach after multiple regeneration attempts\n\n")
        
        total_attempted = len(oracle_diffs) + len(skipped_seeds)
        f.write(f"COMPLETED SEEDS: {len(oracle_diffs)}/{total_attempted}\n\n")
        
        # Calculate averages (filter out inf values)
        oracle_finite = [x for x in oracle_diffs if x != float('inf')]
        basic_finite = [x for x in basic_diffs if x != float('inf')]
        complex_finite = [x for x in complex_diffs if x != float('inf')]
        milp_finite = [x for x in milp_diffs if x != float('inf')]
        
        f.write(f"1) Average Makespan Difference (Oracle): {np.mean(oracle_finite):.2f}\n")
        f.write(f"   Oracle Feasible Rate: {len(oracle_finite)}/{len(oracle_diffs)} ({100*len(oracle_finite)/len(oracle_diffs):.1f}%)\n\n")
        
        f.write(f"2) Average Makespan Difference (Heuristic 2-a Basic): {np.mean(basic_finite):.2f}\n")
        f.write(f"   Heuristic 2-a Feasible Rate: {len(basic_finite)}/{len(basic_diffs)} ({100*len(basic_finite)/len(basic_diffs):.1f}%)\n\n")
        
        f.write(f"3) Average Makespan Difference (Heuristic 2-b Complex): {np.mean(complex_finite):.2f}\n")
        f.write(f"   Heuristic 2-b Feasible Rate: {len(complex_finite)}/{len(complex_diffs)} ({100*len(complex_finite)/len(complex_diffs):.1f}%)\n\n")
        
        f.write(f"4) Average Makespan Difference (Heuristic 2-c MILP): {np.mean(milp_finite):.2f}\n")
        f.write(f"   Heuristic 2-c Feasible Rate: {len(milp_finite)}/{len(milp_diffs)} ({100*len(milp_finite)/len(milp_diffs):.1f}%)\n\n")
        
        f.write(f"5) Closest Helper Matches Oracle Choice: {sum(closest_matches)}/{len(closest_matches)} ({100*sum(closest_matches)/len(closest_matches):.1f}%)\n\n")
        
        # Additional statistics
        if oracle_finite:
            f.write(f"Detailed Statistics:\n")
            f.write(f"  Oracle Makespan Diffs: Min={min(oracle_finite):.2f}, Max={max(oracle_finite):.2f}, Std={np.std(oracle_finite):.2f}\n")
        if basic_finite:
            f.write(f"  Basic Makespan Diffs: Min={min(basic_finite):.2f}, Max={max(basic_finite):.2f}, Std={np.std(basic_finite):.2f}\n")
        if complex_finite:
            f.write(f"  Complex Makespan Diffs: Min={min(complex_finite):.2f}, Max={max(complex_finite):.2f}, Std={np.std(complex_finite):.2f}\n")
        if milp_finite:
            f.write(f"  MILP Makespan Diffs: Min={min(milp_finite):.2f}, Max={max(milp_finite):.2f}, Std={np.std(milp_finite):.2f}\n")

def save_oracle_heuristics_csv(output_file, seed, seed_iter, oracle_results, oracle_metrics, heuristics_results, closest_helper_is_optimal, helper_robots):
    """Save Oracle vs Heuristics comparison results to CSV file."""
    import csv
    import os
    
    # Generate CSV filename based on the output_file
    csv_filename = output_file.replace('.txt', '_oracle_heuristics_comparison.csv')
    
    # Check if this is the first seed iteration to write headers
    write_headers = not os.path.exists(csv_filename)
    
    # Calculate makespan differences
    original_makespan = oracle_results['system_makespan']
    oracle_makespan_diff = oracle_metrics['final_system_makespan'] - original_makespan
    basic_makespan_diff = heuristics_results['basic_result']['final_system_makespan'] - original_makespan
    complex_makespan_diff = heuristics_results['complex_result']['final_system_makespan'] - original_makespan
    milp_makespan_diff = heuristics_results['milp_result']['final_system_makespan'] - original_makespan
    
    # Get helper robot categories
    oracle_helper_category = 'Unknown'
    closest_helper_category = 'Unknown'
    
    if oracle_metrics['help_agent_id'] is not None:
        oracle_help_robot = next((r for r in helper_robots if r.id == oracle_metrics['help_agent_id']), None)
        if oracle_help_robot:
            oracle_helper_category = getattr(oracle_help_robot, 'category', 'Unknown')
    
    closest_robot = next((r for r in helper_robots if r.id == heuristics_results['closest_helper_id']), None)
    if closest_robot:
        closest_helper_category = getattr(closest_robot, 'category', 'Unknown')
    
    with open(csv_filename, 'a', newline='') as csvfile:
        fieldnames = [
            'seed', 'seed_iteration',
            'original_makespan', 'original_system_cost',
            
            # Oracle results
            'oracle_final_makespan', 'oracle_makespan_diff', 'oracle_help_cost',
            'oracle_help_agent_id', 'oracle_helper_category', 'oracle_help_render_time',
            'oracle_best_method',
            
            # Heuristic 2-a (Basic) results
            'heuristic_2a_final_makespan', 'heuristic_2a_makespan_diff', 
            'heuristic_2a_help_render_time', 'heuristic_2a_feasible',
            
            # Heuristic 2-b (Complex) results
            'heuristic_2b_final_makespan', 'heuristic_2b_makespan_diff',
            'heuristic_2b_help_render_time', 'heuristic_2b_feasible',
            
            # Heuristic 2-c (MILP) results
            'heuristic_2c_final_makespan', 'heuristic_2c_makespan_diff',
            'heuristic_2c_help_render_time', 'heuristic_2c_feasible',
            
            # Comparison metrics
            'closest_helper_id', 'closest_helper_category', 'closest_distance',
            'closest_helper_is_optimal',
            
            # Performance rankings (1=best, 4=worst)
            'oracle_rank', 'heuristic_2a_rank', 'heuristic_2b_rank', 'heuristic_2c_rank',
            
            # Best performer
            'best_method', 'best_makespan'
        ]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if write_headers:
            writer.writeheader()
        
        # Calculate performance rankings
        results_for_ranking = [
            ('Oracle', oracle_metrics['final_system_makespan']),
            ('Heuristic_2a', heuristics_results['basic_result']['final_system_makespan']),
            ('Heuristic_2b', heuristics_results['complex_result']['final_system_makespan']),
            ('Heuristic_2c', heuristics_results['milp_result']['final_system_makespan'])
        ]
        
        # Sort by makespan (lower is better) and assign ranks
        results_for_ranking.sort(key=lambda x: x[1] if x[1] != float('inf') else float('inf'))
        
        ranks = {}
        for i, (method, makespan) in enumerate(results_for_ranking):
            if makespan == float('inf'):
                ranks[method] = 'inf'
            else:
                ranks[method] = i + 1
        
        best_method, best_makespan = results_for_ranking[0]
        
        writer.writerow({
            'seed': seed,
            'seed_iteration': seed_iter + 1,
            'original_makespan': original_makespan,
            'original_system_cost': oracle_results['system_cost'],
            
            # Oracle results
            'oracle_final_makespan': oracle_metrics['final_system_makespan'],
            'oracle_makespan_diff': oracle_makespan_diff,
            'oracle_help_cost': oracle_metrics['help_cost'],
            'oracle_help_agent_id': oracle_metrics['help_agent_id'],
            'oracle_helper_category': oracle_helper_category,
            'oracle_help_render_time': oracle_metrics['help_render_time'],
            'oracle_best_method': oracle_metrics.get('best_method', 'unknown'),
            
            # Heuristic 2-a results
            'heuristic_2a_final_makespan': heuristics_results['basic_result']['final_system_makespan'],
            'heuristic_2a_makespan_diff': basic_makespan_diff,
            'heuristic_2a_help_render_time': heuristics_results['basic_result']['help_render_time'],
            'heuristic_2a_feasible': True,  # Basic insertion is always feasible
            
            # Heuristic 2-b results
            'heuristic_2b_final_makespan': heuristics_results['complex_result']['final_system_makespan'],
            'heuristic_2b_makespan_diff': complex_makespan_diff,
            'heuristic_2b_help_render_time': heuristics_results['complex_result']['help_render_time'],
            'heuristic_2b_feasible': True,  # Complex shuffling is always feasible
            
            # Heuristic 2-c results
            'heuristic_2c_final_makespan': heuristics_results['milp_result']['final_system_makespan'],
            'heuristic_2c_makespan_diff': milp_makespan_diff,
            'heuristic_2c_help_render_time': heuristics_results['milp_result']['help_render_time'],
            'heuristic_2c_feasible': heuristics_results['milp_result']['feasible'],
            
            # Comparison metrics
            'closest_helper_id': heuristics_results['closest_helper_id'],
            'closest_helper_category': closest_helper_category,
            'closest_distance': heuristics_results['closest_distance'],
            'closest_helper_is_optimal': closest_helper_is_optimal,
            
            # Performance rankings
            'oracle_rank': ranks['Oracle'],
            'heuristic_2a_rank': ranks['Heuristic_2a'],
            'heuristic_2b_rank': ranks['Heuristic_2b'],
            'heuristic_2c_rank': ranks['Heuristic_2c'],
            
            # Best performer
            'best_method': best_method,
            'best_makespan': best_makespan if best_makespan != float('inf') else 'inf'
        })
    
    print(f"Oracle vs Heuristics comparison saved to CSV: {csv_filename}")

def save_detailed_assignments_csv(output_file, seed, seed_iter, oracle_results, oracle_metrics, heuristics_results, helper_robots, initial_tasks, help_task):
    """Save detailed task assignments for each method to CSV."""
    import csv
    import os
    
    # Generate CSV filename for detailed assignments
    csv_filename = output_file.replace('.txt', '_detailed_assignments.csv')
    
    # Check if this is the first seed iteration to write headers
    write_headers = not os.path.exists(csv_filename)
    
    with open(csv_filename, 'a', newline='') as csvfile:
        fieldnames = [
            'seed', 'seed_iteration', 'method', 'robot_id', 'robot_category',
            'assigned_tasks', 'num_tasks', 'has_help_task', 'robot_makespan',
            'is_help_agent', 'final_system_makespan'
        ]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if write_headers:
            writer.writeheader()
        
        # Define methods and their data
        methods_data = [
            ('Original', oracle_results['assignment'], oracle_results['makespan_list'], oracle_results['system_makespan'], None),
            ('Oracle', oracle_metrics.get('assignment', {}), oracle_metrics.get('makespan_list', []), oracle_metrics['final_system_makespan'], oracle_metrics['help_agent_id']),
            ('Heuristic_2a', heuristics_results['basic_result']['assignment'], heuristics_results['basic_result']['makespans'], heuristics_results['basic_result']['final_system_makespan'], heuristics_results['closest_helper_id']),
            ('Heuristic_2b', heuristics_results['complex_result']['assignment'], heuristics_results['complex_result']['makespans'], heuristics_results['complex_result']['final_system_makespan'], heuristics_results['closest_helper_id']),
            ('Heuristic_2c', heuristics_results['milp_result'].get('assignment', {}), heuristics_results['milp_result']['makespans'], heuristics_results['milp_result']['final_system_makespan'], heuristics_results['milp_result'].get('help_agent_id'))
        ]
        
        help_task_name = help_task['name']
        
        for method_name, assignment, makespan_list, final_makespan, help_agent_id in methods_data:
            for i, robot in enumerate(helper_robots):
                robot_id = robot.id
                robot_category = getattr(robot, 'category', 'Unknown')
                assigned_tasks = assignment.get(robot_id, [])
                num_tasks = len(assigned_tasks)
                has_help_task = help_task_name in assigned_tasks
                is_help_agent = (robot_id == help_agent_id)
                
                # Get robot makespan
                if i < len(makespan_list):
                    robot_makespan = makespan_list[i]
                else:
                    robot_makespan = 0
                
                writer.writerow({
                    'seed': seed,
                    'seed_iteration': seed_iter + 1,
                    'method': method_name,
                    'robot_id': robot_id,
                    'robot_category': robot_category,
                    'assigned_tasks': str(assigned_tasks),
                    'num_tasks': num_tasks,
                    'has_help_task': has_help_task,
                    'robot_makespan': robot_makespan if robot_makespan != float('inf') else 'inf',
                    'is_help_agent': is_help_agent,
                    'final_system_makespan': final_makespan if final_makespan != float('inf') else 'inf'
                })
    
    print(f"Detailed assignments saved to CSV: {csv_filename}")
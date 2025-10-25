import copy
import numpy as np
from utils import compute_cost, parse_formula_string, get_adjacent_free_cell, count_transitions_until_target
from Oracle import Oracle

def manhattan_distance(pos1, pos2):
    """Calculate Manhattan distance between two positions."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def find_closest_helper(helper_robots, help_site):
    """Find the helper robot closest to the help site by Manhattan distance."""
    closest_helper = None
    min_distance = float('inf')
    
    for helper in helper_robots:
        distance = manhattan_distance(helper.initial_pos, help_site)
        if distance < min_distance:
            min_distance = distance
            closest_helper = helper
    
    return closest_helper, min_distance

def run_heuristics_baseline(exp_data):
    """Run all heuristic baseline approaches."""
    world = exp_data['world']
    oracle = exp_data['oracle']
    helper_robots = exp_data['helper_robots']
    initial_tasks = exp_data['initial_tasks']
    help_task = exp_data['help_task']
    oracle_results = exp_data['oracle_results']
    
    # Find closest helper
    closest_helper, min_distance = find_closest_helper(helper_robots, world.conflict_cell)
    
    # 2-a) Basic approach: Insert help task at beginning of closest helper's queue
    basic_result = basic_insertion_approach(
        oracle, helper_robots, initial_tasks, help_task, oracle_results, closest_helper
    )
    
    # 2-b) Complex approach: Allow task shuffling but constrain help task to closest helper
    complex_result = complex_shuffling_approach(
        oracle, helper_robots, initial_tasks, help_task, oracle_results, closest_helper
    )
    
    # 2-c) MILP baseline: Only optimize closest helper with MILP
    milp_result = milp_baseline_approach(
        oracle, helper_robots, initial_tasks, help_task, oracle_results, closest_helper, world
    )
    
    return {
        'closest_helper_id': closest_helper.id,
        'closest_distance': min_distance,
        'basic_result': basic_result,
        'complex_result': complex_result,
        'milp_result': milp_result
    }

def basic_insertion_approach(oracle, helper_robots, initial_tasks, help_task, oracle_results, closest_helper):
    """2-a) Basic approach: Insert help task at beginning of closest helper's queue."""
    """This heuristics is too simple, where our method beats it greatly.... Thus this is not used or discussed in the paper."""
    
    # Create modified assignment
    modified_assignment = copy.deepcopy(oracle_results['assignment'])
    
    # Insert help task at the beginning of closest helper's task list
    if closest_helper.id in modified_assignment:
        modified_assignment[closest_helper.id].insert(0, help_task['name'])
    else:
        modified_assignment[closest_helper.id] = [help_task['name']]
    
    # Calculate new makespans with fixed assignment and order
    updated_makespans = []
    all_tasks = initial_tasks + [help_task]
    
    for helper in helper_robots:
        assigned_task_names = modified_assignment.get(helper.id, [])
        assigned_tasks = [task for task in all_tasks if task['name'] in assigned_task_names]
        
        # Calculate makespan with fixed order (no optimization)
        current_pos = helper.initial_pos
        total_time = 0
        
        for task in assigned_tasks:
            # Time to reach pickup
            total_time += oracle.get_path_cost(current_pos, task['pickup'])
            current_pos = task['pickup']
            
            # Time for pickup to dropoff
            total_time += oracle.get_path_cost(task['pickup'], task['dropoff'])
            current_pos = task['dropoff']
        
        updated_makespans.append(total_time)
    
    # Calculate final system makespan including requester
    requester_makespan = np.mean(oracle_results['makespan_list'])
    
    # Find when help is rendered (when closest helper reaches help site)
    help_render_time = 0
    if closest_helper.id in modified_assignment:
        assigned_tasks = [task for task in all_tasks if task['name'] in modified_assignment[closest_helper.id]]
        current_pos = closest_helper.initial_pos
        
        for task in assigned_tasks:
            if task['name'] == help_task['name']:
                help_render_time = oracle.get_path_cost(current_pos, task['pickup'])
                break
            # Move through previous tasks
            help_render_time += oracle.get_path_cost(current_pos, task['pickup'])
            help_render_time += oracle.get_path_cost(task['pickup'], task['dropoff'])
            current_pos = task['dropoff']
    
    final_requester_makespan = requester_makespan + help_render_time
    final_system_makespan = sum(updated_makespans) + round(final_requester_makespan)
    
    return {
        'assignment': modified_assignment,
        'makespans': updated_makespans,
        'help_render_time': help_render_time,
        'final_system_makespan': final_system_makespan
    }

def complex_shuffling_approach(oracle, helper_robots, initial_tasks, help_task, oracle_results, closest_helper):
    """2-b) Complex approach: Allow task shuffling but help task must go to closest helper.
        This corresponds to the baseline B2
    """
    
    # Start with original assignment
    modified_assignment = copy.deepcopy(oracle_results['assignment'])
    
    # Add help task to closest helper
    if closest_helper.id in modified_assignment:
        modified_assignment[closest_helper.id].append(help_task['name'])
    else:
        modified_assignment[closest_helper.id] = [help_task['name']]
    
    # Now optimize task order for each robot (including the closest one with help task)
    all_tasks = initial_tasks + [help_task]
    best_makespans = []
    
    for helper in helper_robots:
        assigned_task_names = modified_assignment.get(helper.id, [])
        assigned_tasks = [task for task in all_tasks if task['name'] in assigned_task_names]
        
        if not assigned_tasks:
            best_makespans.append(0)
            continue
        
        # Find optimal order for this helper's tasks
        best_makespan = float('inf')
        
        # Try different permutations (limit to reasonable number)
        import itertools
        max_permutations = min(100, len(list(itertools.permutations(assigned_tasks))))
        
        for perm in itertools.permutations(assigned_tasks):
            makespan = oracle._calculate_sequence_cost(helper.initial_pos, list(perm))
            if makespan < best_makespan:
                best_makespan = makespan
                # Update assignment with best order
                modified_assignment[helper.id] = [task['name'] for task in perm]
        
        best_makespans.append(best_makespan)
    
    # Calculate help render time
    help_render_time = 0
    if closest_helper.id in modified_assignment:
        assigned_tasks = [task for task in all_tasks if task['name'] in modified_assignment[closest_helper.id]]
        current_pos = closest_helper.initial_pos
        
        for task in assigned_tasks:
            if task['name'] == help_task['name']:
                help_render_time = oracle.get_path_cost(current_pos, task['pickup'])
                break
            help_render_time += oracle.get_path_cost(current_pos, task['pickup'])
            help_render_time += oracle.get_path_cost(task['pickup'], task['dropoff'])
            current_pos = task['dropoff']
    
    # Final calculations
    requester_makespan = np.mean(oracle_results['makespan_list'])
    final_requester_makespan = requester_makespan + help_render_time
    final_system_makespan = sum(best_makespans) + round(final_requester_makespan)
    
    return {
        'assignment': modified_assignment,
        'makespans': best_makespans,
        'help_render_time': help_render_time,
        'final_system_makespan': final_system_makespan
    }

def milp_baseline_approach(oracle, helper_robots, initial_tasks, help_task, oracle_results, closest_helper, world):
    """2-c) MILP baseline: Only optimize closest helper with MILP."""
    
    # Set up the closest helper for MILP optimization
    ltl_locs = {}
    assigned_task_names = oracle_results['assignment'].get(closest_helper.id, [])
    assigned_tasks = [task for task in initial_tasks if task['name'] in assigned_task_names]
    spec_clauses = []
    
    # Add original tasks
    for idx, loc in enumerate(assigned_tasks):
        i, j = loc['pickup'][0], loc['pickup'][1]
        pickup_key = f'pickup_{idx}_{i}_{j}'
        ltl_locs[pickup_key] = loc['pickup']
        
        i, j = loc['dropoff'][0], loc['dropoff'][1]
        dropoff_key = f'dropoff_{idx}_{i}_{j}'
        ltl_locs[dropoff_key] = loc['dropoff']
        
        spec_clauses.append(f"F(IMPLIES_NEXT({pickup_key},{dropoff_key}))")
    
    # Add help task
    ltl_locs['help_site'] = world.conflict_cell
    ltl_locs['help_site_drop'] = get_adjacent_free_cell(world, world.conflict_cell)
    spec_clauses.append("F(IMPLIES_NEXT(help_site,help_site_drop))")
    
    closest_helper.ltl_locs = ltl_locs
    closest_helper.llm_stl_expression_help = "&".join(spec_clauses)
    closest_helper.init_default_ltl_locations(closest_helper.ltl_locs)
    
    # Set appropriate T based on oracle result
    original_makespan = oracle_results['makespan_dict'].get(closest_helper.id, 0)
    closest_helper.T = min(max(30, int(1.2 * original_makespan)),50)

    
    # Solve MILP for closest helper only
    def find_feasible_solution(helper, formula, max_attempts=10):
        for attempt in range(max_attempts):
            # Check if T exceeds threshold
            if helper.T > 50:
                print(f"T={helper.T} exceeds threshold of 50 Skipping this configuration.")
                return (0, None, float('inf'))  # Return infeasible result
            
            if hasattr(helper, '_task_counter'):
                helper._task_counter = 0
            if hasattr(helper, '_task_states'):
                helper._task_states.clear()
            helper._build_model()
            helper.init_default_ltl_locations(helper.ltl_locs)
            
            results = helper.solve_physical_visits(formula)
            if results[0] == 2:  # Feasible solution found
                return results
            else:
                helper.T = int(helper.T * 1.2)
                print(f"Infeasible solution for closest helper {helper.id} with T={helper.T} (attempt {attempt + 1})")
        return results
    
    formula = parse_formula_string(closest_helper.llm_stl_expression_help)
    milp_results = find_feasible_solution(closest_helper, formula)
    
    # Calculate final makespans
    final_makespans = copy.deepcopy(oracle_results['makespan_list'])
    help_render_time = 0
    
    if milp_results[0] == 2:  # Feasible
        # Update closest helper's makespan
        for i, helper in enumerate(helper_robots):
            if helper.id == closest_helper.id:
                final_makespans[i] = milp_results[2]
                # Calculate help render time
                updated_path = milp_results[1]
                help_render_time = count_transitions_until_target(updated_path, world.conflict_cell)
                break
    else:
        # Infeasible or T too large - use infinity
        for i, helper in enumerate(helper_robots):
            if helper.id == closest_helper.id:
                final_makespans[i] = float('inf')
                break
        help_render_time = float('inf')
    
    # Final calculations
    requester_makespan = np.mean(oracle_results['makespan_list'])
    final_requester_makespan = requester_makespan + help_render_time
    # if final_requester_makespan is not float('inf'):
    import math
    if not math.isinf(final_requester_makespan):
        final_requester_makespan = round(final_requester_makespan)
        final_system_makespan = sum(final_makespans) + round(final_requester_makespan)
    else:
        final_system_makespan = float('inf')
    
    return {
        'makespans': final_makespans,
        'help_render_time': help_render_time,
        'final_system_makespan': final_system_makespan,
        'feasible': milp_results[0] == 2,
        't_exceeded_threshold': closest_helper.T > 50
    }


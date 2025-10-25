import heapq
import itertools
import random
from copy import deepcopy
import copy
from GridWorld import GridWorld
from utils import *
import numpy as np
class Oracle:
    """
    Enhanced Oracle that explores different task execution orders for optimal cost minimization.
    """
    def __init__(self, world: GridWorld, cost_function='mixed', max_time_horizon=float('inf')):
        self.world = world
        self.cost_function = cost_function  # Store cost function type
        self.max_time_horizon = max_time_horizon  # Maximum allowed makespan per agent
        self.costs = self._precompute_all_pairs_shortest_paths()

    def _precompute_all_pairs_shortest_paths(self):
        print("Oracle is precomputing all-pairs shortest paths... (this may take a moment)")
        costs = {}
        free_cells_tuple = tuple(self.world.free_cells)
        for start_node in free_cells_tuple:
            costs[start_node] = self._dijkstra(start_node)
        print("Oracle precomputation complete.")
        return costs

    def _dijkstra(self, start_pos):
        distances = {cell: float('inf') for cell in self.world.free_cells}
        distances[start_pos] = 0
        pq = [(0, start_pos)]
        while pq:
            dist, current_pos = heapq.heappop(pq)
            if dist > distances[current_pos]:
                continue
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                next_pos = (current_pos[0] + dx, current_pos[1] + dy)
                if next_pos in self.world.free_cells:
                    new_dist = distances[current_pos] + 1
                    if new_dist < distances[next_pos]:
                        distances[next_pos] = new_dist
                        heapq.heappush(pq, (new_dist, next_pos))
        return distances

    def get_path_cost(self, start_pos, end_pos):
        return self.costs.get(start_pos, {}).get(end_pos, float('inf'))

    def find_optimal_assignment(self, tasks, robots):
        """
        Assignment that respects sequential task execution constraints, robot-task category compatibility,
        and maximum time horizon constraints.
        """
        unassigned_tasks = list(tasks)
        robot_states = [
            {
                'id': robot.id,
                'current_pos': robot.initial_pos,
                'makespan': 0,
                'task_sequence': []  # Ordered list of tasks
            } for robot in robots
        ]
        
        # Assign tasks one by one, considering sequential execution and time horizon
        while unassigned_tasks:
            best_assignment = None
            best_cost = float('inf')
            
            for task in unassigned_tasks:
                for i, robot_state in enumerate(robot_states):
                    robot = robots[i]
                    
                    # Check robot-task category compatibility
                    if not self._is_robot_task_compatible(robot, task):
                        continue  # Skip incompatible robot-task pairs
                    
                    # Calculate INCREMENTAL cost for adding this task
                    current_pos = robot_state['current_pos']
                    pickup_cost = self.get_path_cost(current_pos, task['pickup'])
                    dropoff_cost = self.get_path_cost(task['pickup'], task['dropoff'])
                    incremental_cost = pickup_cost + dropoff_cost
                    new_makespan = robot_state['makespan'] + incremental_cost
                    
                    # Check time horizon constraint
                    if new_makespan > self.max_time_horizon:
                        continue  # Skip if exceeds time horizon
                    
                    # Calculate what cost would be with this assignment
                    temp_makespans = [rs['makespan'] for rs in robot_states]
                    temp_makespans[i] = new_makespan  # Update this robot's makespan
                    
                    current_metric = compute_cost(temp_makespans, self.cost_function)
                    
                    # CRITICAL: Verify this assignment doesn't create invalid paths
                    temp_sequence = robot_state['task_sequence'] + [task]
                    if self._validate_task_sequence_feasibility(robot_state['current_pos'], temp_sequence):
                        if current_metric < best_cost:
                            best_cost = current_metric
                            best_assignment = {
                                'robot_idx': i,
                                'task': task,
                                'incremental_cost': incremental_cost,
                                'new_makespan': new_makespan,
                                'new_pos': task['dropoff'],  # Robot will be at dropoff after this task
                                'new_cost': current_metric
                            }
            
            if best_assignment is None:
                print(f"WARNING: No feasible assignment found within time horizon {self.max_time_horizon}!")
                print(f"Remaining unassigned tasks: {[t['name'] for t in unassigned_tasks]}")
                break
                
            # Apply the best assignment and UPDATE robot state
            robot_idx = best_assignment['robot_idx']
            robot_state = robot_states[robot_idx]
            robot_state['task_sequence'].append(best_assignment['task'])
            robot_state['makespan'] = best_assignment['new_makespan']
            robot_state['current_pos'] = best_assignment['new_pos']  # CRITICAL: Update position
            
            unassigned_tasks.remove(best_assignment['task'])
            
            current_makespans = [rs['makespan'] for rs in robot_states]
            current_cost = compute_cost(current_makespans, self.cost_function)
            
            print(f"Assigned {best_assignment['task']['name']} to Robot {robot_state['id']}, "
                  f"new makespan: {robot_state['makespan']}, "
                  f"system cost ({self.cost_function}): {current_cost:.2f}")
        
        # Return results
        makespan_list = [rs['makespan'] for rs in robot_states]
        final_assignment = {
            rs['id']: [task['name'] for task in rs['task_sequence']] 
            for rs in robot_states
        }
        
        # Check if any agent exceeds time horizon
        if any(makespan > self.max_time_horizon for makespan in makespan_list):
            print(f"WARNING: Some agents exceed time horizon {self.max_time_horizon}")
            for i, makespan in enumerate(makespan_list):
                if makespan > self.max_time_horizon:
                    print(f"  Robot {robot_states[i]['id']}: makespan {makespan} > {self.max_time_horizon}")
        
        return final_assignment, makespan_list

    def _is_robot_task_compatible(self, robot, task):
        """Check if a robot can perform a specific task based on category compatibility."""
        # Get robot category
        robot_category = getattr(robot, 'category', 'Unknown')
        
        # Get task category
        task_category = task.get('category', 'Any Category')
        
        # Define compatibility rules
        # If task is "Any Category", any robot can do it
        if task_category == 'Any Category':
            return True
            
        # If task has a specific category, robot must match
        return robot_category == task_category

    def _validate_task_sequence_feasibility(self, start_pos, task_sequence):
        """
        Check if a task sequence can be executed without violating pickup-then-dropoff constraints.
        Validates that all positions are reachable and paths exist.
        """
        if not task_sequence:
            return True
        
        current_pos = start_pos
        
        # Validate each task in the sequence
        for i, task in enumerate(task_sequence):
            pickup_pos = task['pickup']
            dropoff_pos = task['dropoff']
            
            # Check if pickup position is reachable from current position
            pickup_cost = self.get_path_cost(current_pos, pickup_pos)
            if pickup_cost == float('inf'):
                print(f"Task {task['name']}: Cannot reach pickup {pickup_pos} from {current_pos}")
                return False
            
            # Check if dropoff position is reachable from pickup
            dropoff_cost = self.get_path_cost(pickup_pos, dropoff_pos)
            if dropoff_cost == float('inf'):
                print(f"Task {task['name']}: Cannot reach dropoff {dropoff_pos} from pickup {pickup_pos}")
                return False
            
            # Validate that pickup and dropoff positions are in free cells
            if pickup_pos not in self.world.free_cells:
                print(f"Task {task['name']}: Pickup position {pickup_pos} is not a free cell")
                return False
                
            if dropoff_pos not in self.world.free_cells:
                print(f"Task {task['name']}: Dropoff position {dropoff_pos} is not a free cell")
                return False
            
            # Update current position to dropoff for next iteration
            current_pos = dropoff_pos
        
        # Additional validation: Check for any specific task constraints
        # For example, ensure the help task can access the conflict cell
        for task in task_sequence:
            if 'HelpTask' in task['name']:
                # Help task should have pickup at conflict cell
                if hasattr(self.world, 'conflict_cell'):
                    if task['pickup'] != self.world.conflict_cell:
                        print(f"Help task {task['name']}: Pickup should be at conflict cell {self.world.conflict_cell}, but is at {task['pickup']}")
                        return False
        
        return True

    def _calculate_sequence_cost(self, start_pos, task_sequence):
        """
        Calculate the total cost of executing a sequence of tasks.
        Respects pickup-then-dropoff ordering.
        """
        if not task_sequence:
            return 0
            
        total_cost = 0
        current_pos = start_pos
        
        for task in task_sequence:
            # Cost to travel to pickup
            pickup_cost = self.get_path_cost(current_pos, task['pickup'])
            # Cost to travel from pickup to dropoff  
            dropoff_cost = self.get_path_cost(task['pickup'], task['dropoff'])
            
            total_cost += pickup_cost + dropoff_cost
            current_pos = task['dropoff']  # Robot ends at dropoff
        
        return total_cost

    def find_optimal_assignment_with_reordering(self, tasks, robots, max_permutations=100):
        """
        First assign tasks, then optimize execution order for each robot using the specified cost function.
        """
        # Step 1: Initial assignment using greedy approach
        initial_assignment, _ = self.find_optimal_assignment(tasks, robots)
        
        # Step 2: For each robot, try different execution orders of their assigned tasks
        best_assignment = initial_assignment
        best_makespan_list = None
        
        # Calculate initial cost
        initial_makespan_list = []
        for robot in robots:
            assigned_task_names = initial_assignment.get(robot.id, [])
            assigned_tasks = [task for task in tasks if task['name'] in assigned_task_names]
            robot_makespan = self._calculate_sequence_cost(robot.initial_pos, assigned_tasks)
            initial_makespan_list.append(robot_makespan)
        
        best_cost = compute_cost(initial_makespan_list, self.cost_function)
        best_makespan_list = initial_makespan_list
        
        print(f"\n--- Optimizing Task Execution Orders (using {self.cost_function}) ---")
        print(f"Initial cost ({self.cost_function}): {best_cost:.2f}")
        
        for robot in robots:
            assigned_task_names = initial_assignment.get(robot.id, [])
            if len(assigned_task_names) <= 1:
                continue  # No reordering needed for 0 or 1 tasks
                
            assigned_tasks = [task for task in tasks if task['name'] in assigned_task_names]
            
            print(f"Robot {robot.id} has {len(assigned_tasks)} tasks - trying different orders...")
            
            # Generate all permutations of this robot's tasks
            task_permutations = list(itertools.permutations(assigned_tasks))
            
            # Limit permutations to avoid exponential explosion
            if len(task_permutations) > max_permutations:
                print(f"  Too many permutations ({len(task_permutations)}), sampling {max_permutations}")
                task_permutations = random.sample(task_permutations, max_permutations)
            
            best_robot_order = assigned_tasks
            current_makespan_list = list(best_makespan_list)  # Copy current best
            robot_idx = next(i for i, r in enumerate(robots) if r.id == robot.id)
            
            for perm in task_permutations:
                perm_makespan = self._calculate_sequence_cost(robot.initial_pos, list(perm))
                
                # Calculate cost with this permutation
                test_makespan_list = list(current_makespan_list)
                test_makespan_list[robot_idx] = perm_makespan
                
                test_cost = compute_cost(test_makespan_list, self.cost_function)
                
                if test_cost < best_cost:
                    best_cost = test_cost
                    best_robot_order = list(perm)
                    best_makespan_list = test_makespan_list
            
            print(f"  Best order for Robot {robot.id}: {[t['name'] for t in best_robot_order]} "
                  f"(makespan: {best_makespan_list[robot_idx]:.2f})")
            
            # Update assignment with best order
            best_assignment[robot.id] = [task['name'] for task in best_robot_order]
        
        final_cost = compute_cost(best_makespan_list, self.cost_function)
        print(f"Final optimized cost ({self.cost_function}): {final_cost:.2f}")
        
        return best_assignment, best_makespan_list

    def find_optimal_assignment_global_reordering(self, tasks, robots, max_iterations=500):
        """
        More advanced: Try reassigning tasks between robots AND reordering using the specified cost function,
        while respecting time horizon constraints.
        """
        print(f"\n--- Global Task Reassignment and Reordering (using {self.cost_function}, max_horizon={self.max_time_horizon}) ---")
        
        # Start with initial assignment
        current_assignment, current_makespan_list = self.find_optimal_assignment(tasks, robots)
        best_assignment = current_assignment
        best_makespan_list = current_makespan_list
        
        best_metric = compute_cost(current_makespan_list, self.cost_function)
        
        print(f"Initial cost ({self.cost_function}): {best_metric:.2f}")
        
        # Add convergence tracking
        no_improvement_count = 0
        last_improvement_metric = best_metric
        
        for iteration in range(max_iterations):
            improved = False
            
            # Try moving each task to a different robot
            for task in tasks:
                current_robot_id = None
                for robot_id, task_names in current_assignment.items():
                    if task['name'] in task_names:
                        current_robot_id = robot_id
                        break
                
                if current_robot_id is None:
                    continue
                
                # Try assigning this task to each other robot
                for robot in robots:
                    if robot.id == current_robot_id:
                        continue
                    
                    # Check robot-task category compatibility before reassignment
                    if not self._is_robot_task_compatible(robot, task):
                        continue  # Skip incompatible robot-task pairs
                    
                    # Create test assignment
                    test_assignment = deepcopy(current_assignment)
                    test_assignment[current_robot_id].remove(task['name'])
                    test_assignment[robot.id].append(task['name'])
                    
                    # Calculate makespan for this reassignment with OPTIMAL ORDERING
                    test_makespan_list = []
                    valid_assignment = True
                    
                    for test_robot in robots:
                        assigned_task_names = test_assignment.get(test_robot.id, [])
                        assigned_tasks = [t for t in tasks if t['name'] in assigned_task_names]
                        
                        # Find the optimal order that minimizes SYSTEM COST, not individual makespan
                        if len(assigned_tasks) <= 1:
                            robot_makespan = self._calculate_sequence_cost(test_robot.initial_pos, assigned_tasks)
                        else:
                            # This is the key fix: optimize for system cost, not individual makespan
                            best_system_cost = float('inf')
                            best_robot_makespan = float('inf')
                            orders_to_try = list(itertools.permutations(assigned_tasks))[:24]  # Limit to 24 permutations
                            
                            for order in orders_to_try:
                                order_makespan = self._calculate_sequence_cost(test_robot.initial_pos, list(order))
                                
                                # Check time horizon constraint for this order
                                if order_makespan > self.max_time_horizon:
                                    continue
                                
                                # Calculate system cost with this order
                                temp_makespan_list = []
                                for temp_robot in robots:
                                    if temp_robot.id == test_robot.id:
                                        temp_makespan_list.append(order_makespan)
                                    else:
                                        # Use existing makespans for other robots (will be updated in outer loop)
                                        existing_tasks = test_assignment.get(temp_robot.id, [])
                                        existing_assigned = [t for t in tasks if t['name'] in existing_tasks]
                                        if len(existing_assigned) <= 1:
                                            temp_makespan_list.append(self._calculate_sequence_cost(temp_robot.initial_pos, existing_assigned))
                                        else:
                                            # For simplicity, use current makespan (this could be optimized further)
                                            current_idx = next(i for i, r in enumerate(robots) if r.id == temp_robot.id)
                                            if current_idx < len(current_makespan_list):
                                                temp_makespan_list.append(current_makespan_list[current_idx])
                                            else:
                                                temp_makespan_list.append(0)
                                
                                system_cost = compute_cost(temp_makespan_list, self.cost_function)
                                
                                # Choose order that minimizes SYSTEM cost, not individual makespan
                                if system_cost < best_system_cost:
                                    best_system_cost = system_cost
                                    best_robot_makespan = order_makespan
                            
                            robot_makespan = best_robot_makespan
                        
                        # Check time horizon constraint
                        if robot_makespan > self.max_time_horizon:
                            valid_assignment = False
                            break
                        
                        test_makespan_list.append(robot_makespan)
                    
                    if not valid_assignment:
                        continue  # Skip this assignment if it violates time horizon
                    
                    test_metric = compute_cost(test_makespan_list, self.cost_function)
                    
                    # Add minimum improvement threshold for sum-based costs
                    min_improvement = 0.001 
                    # If this is better by at least the minimum threshold, update current assignment
                    if test_metric < best_metric - min_improvement:
                        print(f"  Iteration {iteration}: Moving {task['name']} from Robot {current_robot_id} to Robot {robot.id}")
                        print(f"    Cost ({self.cost_function}) improved: {best_metric:.2f} -> {test_metric:.2f}")
                        print(f"    Makespans: {test_makespan_list} (all <= {self.max_time_horizon})")
                        
                        current_assignment = test_assignment
                        current_makespan_list = test_makespan_list
                        best_assignment = current_assignment
                        best_makespan_list = current_makespan_list
                        best_metric = test_metric
                        improved = True
                        no_improvement_count = 0
                        break
                
                if improved:
                    break
            
            if not improved:
                no_improvement_count += 1
                # Early termination if no improvement for several iterations
                if no_improvement_count >= 5:
                    print(f"  Early termination: No improvement for {no_improvement_count} iterations")
                    break
                print(f"  No improvement in iteration {iteration + 1}")
            
            # Additional safety check for infinite loops
            if iteration > 10 and abs(best_metric - last_improvement_metric) < 0.001:
                print(f"  Converged: minimal change in cost function")
                break
        
        print(f"  Final convergence after {iteration + 1} iterations")
        return best_assignment, best_makespan_list
    def find_optimal_assignment_simple_reordering(self, tasks, robots):
        """
        Simplified approach: Assign tasks first, then optimize order for each robot independently.
        This matches the approach used in Heuristic 2-b.
        """
        print(f"\n--- Simple Task Assignment with Order Optimization (using {self.cost_function}) ---")
        
        # Step 1: Initial assignment using greedy approach
        initial_assignment, _ = self.find_optimal_assignment(tasks, robots)
        
        # Step 2: For each robot, find the optimal order for their assigned tasks
        best_assignment = {}
        best_makespan_list = []
        
        for robot in robots:
            assigned_task_names = initial_assignment.get(robot.id, [])
            assigned_tasks = [task for task in tasks if task['name'] in assigned_task_names]
            
            if not assigned_tasks:
                best_makespan_list.append(0)
                best_assignment[robot.id] = []
                continue
            
            # Find optimal order for this robot's tasks
            best_makespan = float('inf')
            best_order = assigned_tasks
            
            # Try different permutations (limit to reasonable number)
            import itertools
            max_permutations = min(100, len(list(itertools.permutations(assigned_tasks))))
            
            for perm in itertools.permutations(assigned_tasks):
                makespan = self._calculate_sequence_cost(robot.initial_pos, list(perm))
                if makespan < best_makespan:
                    best_makespan = makespan
                    best_order = list(perm)
            
            best_makespan_list.append(best_makespan)
            best_assignment[robot.id] = [task['name'] for task in best_order]
            
            print(f"Robot {robot.id}: Optimal order gives makespan {best_makespan}")
        
        final_cost = compute_cost(best_makespan_list, self.cost_function)
        print(f"Final cost ({self.cost_function}): {final_cost:.2f}")
        
        return best_assignment, best_makespan_list
    

    def find_optimal_assignment_exact_2b_all_agents(self, tasks, robots, oracle_results):
        """
        Algorithm that implements Baseline B2 of the paper.
        It is a hybrid approach where the closest helper to the help site solves the MILP (and is a default helper),
        while the Oracle is allowed to shuffle and re-assign tasks to all other robots.
        """
        
        # Identify help task
        help_task = None
        initial_tasks = []
        for task in tasks:
            if 'HelpTask' in task['name']:
                help_task = task
            else:
                initial_tasks.append(task)
        
        if help_task is None:
            return self.find_optimal_assignment(tasks, robots)
        
        best_result = None
        best_system_makespan = float('inf')
        
        
        for robot in robots:

            # Check compatibility, not used yet, for CATL step
            if not self._is_robot_task_compatible(robot, help_task):
                print(f"    Robot {robot.id} incompatible with help task")
                continue
            
            # Start with original assignment
            modified_assignment = copy.deepcopy(oracle_results['assignment'])
            
            # Add help task to this robot (treating it as "closest helper")
            if robot.id in modified_assignment:
                modified_assignment[robot.id].append(help_task['name'])
            else:
                modified_assignment[robot.id] = [help_task['name']]
            
            # Now optimize task order for each robot (including the one with help task)
            all_tasks = initial_tasks + [help_task]
            best_makespans = []
            
            for helper in robots:
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
                    makespan = self._calculate_sequence_cost(helper.initial_pos, list(perm))
                    if makespan < best_makespan:
                        best_makespan = makespan
                        # Update assignment with best order
                        modified_assignment[helper.id] = [task['name'] for task in perm]
                
                best_makespans.append(best_makespan)
            
            # Calculate help render time (EXACT same logic as 2-b)
            help_render_time = 0
            if robot.id in modified_assignment:
                assigned_tasks = [task for task in all_tasks if task['name'] in modified_assignment[robot.id]]
                current_pos = robot.initial_pos
                
                for task in assigned_tasks:
                    if task['name'] == help_task['name']:
                        help_render_time = self.get_path_cost(current_pos, task['pickup'])
                        break
                    help_render_time += self.get_path_cost(current_pos, task['pickup'])
                    help_render_time += self.get_path_cost(task['pickup'], task['dropoff'])
                    current_pos = task['dropoff']
            
            # Final calculations (EXACT same as 2-b)
            requester_makespan = np.mean(oracle_results['makespan_list'])
            final_requester_makespan = requester_makespan + help_render_time
            final_system_makespan = sum(best_makespans) + round(final_requester_makespan)
            
            print(f"    Robot {robot.id} system makespan: {final_system_makespan}")
            
            # Check if this is the best assignment so far
            if final_system_makespan < best_system_makespan:
                best_system_makespan = final_system_makespan
                best_result = {
                    'assignment': modified_assignment,
                    'makespans': best_makespans,
                    'help_render_time': help_render_time,
                    'final_system_makespan': final_system_makespan,
                    'final_requester_makespan': final_requester_makespan,
                    'help_agent_id': robot.id
                }
                print(f"    New best! Robot {robot.id} with makespan {final_system_makespan}")
        
        if best_result is None:
            print("No feasible assignment found!")
            return {}, []
        
        print(f"Final best: Robot {best_result['help_agent_id']} with makespan {best_result['final_system_makespan']}")
        
        return best_result
    


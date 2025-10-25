import random
import numpy as np
from GridWorld import GridWorld
from robot import GridWorldAgent
from Oracle import Oracle
from utils import start_pos_init, get_adjacent_free_cell, compute_cost, read_experiment_parameters
import copy

class ExperimentWorldGenerator:
    """
    Generates consistent experimental worlds and initial conditions for both Oracle and MILP approaches.
    """
    
    def __init__(self, parameters):
        self.parameters = parameters
        self.seed_iterations = parameters['SEED_ITERATION']
        self.starting_seed = parameters['SEED']
        self.num_helper_robots = parameters['NUM_HELPER_ROBOTS']
        self.cost_function = parameters['COST_FUNCTION']
        self.num_initial_tasks = parameters['NUM_TASK_PER_HELPER'] * self.num_helper_robots
        self.use_unique_pick_up_dropoff = parameters['USE_UNIQUE_PICK_UP_DROPOFF']
        self.trust_oracle_init_milp = parameters.get('TRUST_ORACLE_INIT_MILP', False)
        self.use_categorized_task = parameters.get('USE_CATEGORIZED_TASK', False)
        self.max_time_horizon = parameters.get('MAX_TIME_HORIZON', float('inf'))  # Add max time horizon

        # Define robot categories
        self.robot_categories = ["Reach Forklift Truck", "Counter Balance", "Order Picker"]
    
    def generate_world_and_tasks(self, seed_iter):
        """
        Generate a complete experimental setup for a given seed iteration.
        Returns all necessary components for both Oracle and MILP experiments.
        """
        current_seed = self.starting_seed + seed_iter
        
        # Create world
        world = GridWorld(grid_size_rows=10, grid_size_cols=6, seed=current_seed)
        oracle = Oracle(world=world, cost_function=self.cost_function, max_time_horizon=self.max_time_horizon)
        
        # Create helper robots with categories
        helper_robots = self._create_categorized_robots(world, current_seed)
        
        # Generate initial tasks (category-specific)
        initial_tasks = self._generate_categorized_tasks(world, oracle, current_seed, helper_robots)
        
        # Get Oracle's optimal initial assignment (respecting categories and time horizon)
        oracle_results = self._get_oracle_optimal_assignment(oracle, initial_tasks, helper_robots)
        
        # Create help task (can be handled by any category)
        help_task = self._create_help_task(world, oracle)
        
        return {
            'seed': current_seed,
            'world': world,
            'oracle': oracle,
            'helper_robots': helper_robots,
            'initial_tasks': initial_tasks,
            'help_task': help_task,
            'oracle_results': oracle_results
        }
    def _create_categorized_robots(self, world, seed):
        """Create helper robots with assigned categories."""
        helper_robots = []
        random.seed(seed)
        
        # Distribute robots across categories as evenly as possible
        robots_per_category = self.num_helper_robots // len(self.robot_categories)
        remaining_robots = self.num_helper_robots % len(self.robot_categories)
        
        robot_id = 0
        for cat_idx, category in enumerate(self.robot_categories):
            # Calculate how many robots for this category
            num_robots_this_category = robots_per_category
            if cat_idx < remaining_robots:
                num_robots_this_category += 1
            
            # Create robots for this category
            for i in range(num_robots_this_category):
                helper = GridWorldAgent(
                    ltl_locs={},
                    capabilities=["lift pallet", "move"],
                    initial_pos=start_pos_init(world),
                    agent_type='forklift',
                    gridworld=world,
                    needs_help=False,
                    T=30
                )
                helper.id = robot_id  # Ensure unique ID
                helper.category = category  # Assign category
                helper_robots.append(helper)
                robot_id += 1
        
        return helper_robots
    
    def _generate_categorized_tasks(self, world, oracle, seed, helper_robots):
        """Generate initial tasks that are category-specific."""
        initial_tasks = []
        available_cells = list(world.free_cells)
        
        if len(available_cells) < self.num_initial_tasks * 2:
            raise ValueError("Not enough free cells to assign unique pickup and dropoff locations for all tasks.")
        
        random.seed(seed)  # Ensure reproducibility
        
        # Group robots by category
        robots_by_category = {}
        for robot in helper_robots:
            if robot.category not in robots_by_category:
                robots_by_category[robot.category] = []
            robots_by_category[robot.category].append(robot)
        
        # Generate tasks for each category
        task_id = 0
        for category, robots in robots_by_category.items():
            num_tasks_for_category = len(robots) * self.parameters['NUM_TASK_PER_HELPER']
            
            # Generate pickup and dropoff locations for this category's tasks
            if not self.use_unique_pick_up_dropoff:
                pickup_candidates = random.choices(available_cells, k=num_tasks_for_category)
                dropoff_candidates = random.choices(available_cells, k=num_tasks_for_category)
            else:
                # Sample unique locations for this category
                category_pickup_candidates = random.sample(available_cells, num_tasks_for_category)
                remaining_cells = list(set(available_cells) - set(category_pickup_candidates))
                category_dropoff_candidates = random.sample(remaining_cells, num_tasks_for_category)
                pickup_candidates = category_pickup_candidates
                dropoff_candidates = category_dropoff_candidates
            
            # Create tasks for this category

            for i in range(num_tasks_for_category):
                pickup_loc = pickup_candidates[i]
                dropoff_loc = dropoff_candidates[i]
                task_travel_time = oracle.get_path_cost(pickup_loc, dropoff_loc)
                if self.use_categorized_task:
                    task = {
                        'name': f'PNP-{task_id}',
                        'pickup': pickup_loc,
                        'dropoff': dropoff_loc,
                        'duration': task_travel_time,
                        'category': category  # Assign task to category
                    }
                    initial_tasks.append(task)
                    task_id += 1
                else:
                    task = {
                        'name': f'PNP-{task_id}',
                        'pickup': pickup_loc,
                        'dropoff': dropoff_loc,
                        'duration': task_travel_time,
                        'category': "Any Category"  # Assign task to category
                    }
                    initial_tasks.append(task)
                    task_id += 1
        
        return initial_tasks
    
    def _get_oracle_optimal_assignment(self, oracle, initial_tasks, helper_robots):
        """Get Oracle's optimal assignment using both approaches and pick the best."""
        # Try both approaches with category constraints
        reorder_assignment, reorder_makespan_list = oracle.find_optimal_assignment_with_reordering(initial_tasks, helper_robots)
        global_assignment, global_makespan_list = oracle.find_optimal_assignment_global_reordering(initial_tasks, helper_robots)
        
        reorder_system_total_cost = compute_cost(reorder_makespan_list, cost_function=oracle.cost_function)
        global_system_total_cost = compute_cost(global_makespan_list, cost_function=oracle.cost_function)
        
        # Pick the better approach
        if reorder_system_total_cost <= global_system_total_cost:
            initial_assignment = reorder_assignment
            initial_makespan_list = reorder_makespan_list
            initial_system_cost = reorder_system_total_cost
        else:
            initial_assignment = global_assignment
            initial_makespan_list = global_makespan_list
            initial_system_cost = global_system_total_cost
        
        # Calculate system metrics including requester
        initial_makespan_dict = {helper.id: makespan for helper, makespan in zip(helper_robots, initial_makespan_list)}
        initial_system_makespan = sum(initial_makespan_list)
        requester_makespan = np.mean(initial_makespan_list)
        initial_system_makespan += round(requester_makespan)
        
        initial_system_makespan_with_requester = copy.deepcopy(initial_makespan_list)
        initial_system_makespan_with_requester.append(round(requester_makespan))
        initial_system_cost = compute_cost(initial_system_makespan_with_requester, cost_function=oracle.cost_function)
        
        return {
            'assignment': initial_assignment,
            'makespan_list': initial_makespan_list,
            'makespan_dict': initial_makespan_dict,
            'system_makespan': initial_system_makespan,
            'system_cost': initial_system_cost,
            'requester_makespan': requester_makespan
        }
    
    def _create_help_task(self, world, oracle):
        """Create the help task (can be handled by any category)."""
        help_site_pickup = world.conflict_cell
        help_site_dropoff = get_adjacent_free_cell(world, help_site_pickup)
        help_task_travel_time = oracle.get_path_cost(help_site_pickup, help_site_dropoff)
        
        return {
            'name': 'HelpTask-ClearObstruction',
            'pickup': help_site_pickup,
            'dropoff': help_site_dropoff,
            'duration': help_task_travel_time,
            'category': 'Any Category'  # Help task can be handled by any category
        }
"""
@penglangye@foxmail.com
MDP (Markov Decision Process) Solver Library

A Python library for solving Markov Decision Processes with continuous state and action spaces,
supporting value iteration, policy iteration algorithms with penalty functions for constraints.

Main Features:
1. Grid-based discretization of state and action spaces
2. Stochastic perturbation sampling
3. Reward functions with penalty terms for constraints
4. Value iteration and policy iteration algorithms
5. Policy simulation and evaluation
6. Result visualization

Usage Example:
----------
```python
# Initialize solver
solver = MDPSolver(
    dim_x=2, dim_u=2, dim_epsilon=1, n_samples=100,
    transition_func=transition, reward_func=reward,
    constraint_func=constraint
)

# Set grids and PPF
solver.set_grids(
    x_limits=[[-1, 1], [-1, 1]],
    u_limits=[[-0.5, 0.5], [-0.5, 0.5]],
    x_grid_nums=[20, 20],
    u_grid_nums=[10, 10]
)
solver.set_ppf(norm.ppf, n_quantiles=1000)

# Run value iteration
policy_func, value_func_interp, info = solver.value_iteration(
    tol=1e-6, max_iter=100, verbose=True
)

# Use policy
optimal_control = solver.act([0.5, 0.3])
```
"""

import time
import numpy as np
from typing import Callable, List, Optional, Union, Tuple, Dict, Any
import warnings
from scipy import interpolate
import itertools
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus']=False

class MDPSolverCPU:
    """
    MDP solver class, responsible for managing grid generation, function embedding,
    expectation computation and other core functions
    
    Parameters:
        dim_x: State variable dimension
        dim_u: Control variable dimension
        dim_epsilon: Perturbation variable dimension
        n_samples: Number of perturbation samples
        transition_func: State transition function
        reward_func: Reward function
        constraint_func: Constraint function (optional)
        interpolation_method: Interpolation method for value and policy functions
                              Options: 'linear', 'nearest', 'slinear', 'cubic'
    """
    
    def __init__(
        self,
        dim_x: int,
        dim_u: int,
        dim_epsilon: int,
        n_samples: int,
        transition_func: Callable,
        reward_func: Callable,
        constraint_func: Optional[Callable] = None,
        beta: float = 0.95,
        lambda_pen: float = 10.,
        gamma_pen: float = 10.,
        inf_replace: float = 1e12,
        interpolation_method: str = 'linear'
    ):
        """
        Initialize MDP solver
    
        Parameters:
            dim_x: State variable dimension
            dim_u: Control variable dimension
            dim_epsilon: Perturbation variable dimension
            n_samples: Number of perturbation samples
            transition_func: State transition function
            reward_func: Reward function
            constraint_func: Constraint function (optional)
            beta: Discount factor
            lambda_pen: Penalty weight coefficient
            gamma_pen: Penalty scale parameter
            inf_replace: Value to replace infinity
            interpolation_method: Interpolation method for value and policy functions
                                  Options: 'linear', 'nearest', 'slinear', 'cubic'
        """
        # Store basic parameters
        self.dim_x = dim_x
        self.dim_u = dim_u
        self.dim_epsilon = dim_epsilon
        self.n_samples = n_samples
        self.beta = beta
        self.lambda_pen = lambda_pen
        self.gamma_pen = gamma_pen
        self.inf_replace = inf_replace
        self.interpolation_method = interpolation_method
    
        # Validate interpolation method
        valid_methods = ['linear', 'nearest', 'slinear', 'cubic']
        if interpolation_method not in valid_methods:
            raise ValueError(f"Invalid interpolation method: {interpolation_method}. "
                           f"Valid options are: {valid_methods}")
    
        # Store external functions
        self.transition_func = transition_func
        self.reward_func = reward_func
        self.constraint_func = constraint_func
    
        # Initialize internal variables (to be set later)
        self.x_grid = None
        self.u_grid = None
        self.x_grid_points = None  # Store grid points for each dimension
        self.u_grid_points = None  # Store grid points for each dimension
        self.x_grid_shape = None
        self.u_grid_shape = None
        self.x_grid_size = None
        self.u_grid_size = None
        self.value_func = None
        self.ppf_table = None
        self.penalty_reward_func = None
    
        # Initialize reward function with penalty
        self._initialize_penalty_reward_func()
    
        # Initialize interpolators (to be set later)
        self.value_interpolator = None
        self.policy_interpolator = None
    
        # Initialize policy function (to be set later)
        self.policy_func = None
    
        # Flag to track if problem has been solved
        self.is_solved = False
        self.solution_method = None
        self.convergence_info = None
    
        print(f"MDP Solver initialized:")
        print(f"  State dimension: {dim_x}, Control dimension: {dim_u}, Perturbation dimension: {dim_epsilon}")
        print(f"  Perturbation samples: {n_samples}, Discount factor: {beta}")
        print(f"  Interpolation method: {interpolation_method}")
        if constraint_func is not None:
            print(f"  Using constraint function, penalty parameters: λ={lambda_pen}, γ={gamma_pen}")
    
    def _initialize_penalty_reward_func(self):
        """Initialize reward function with penalty"""
        if self.constraint_func is None:
            # No constraints, use original reward function directly
            self.penalty_reward_func = self.reward_func
        else:
            # Create reward function with penalty
            def penalty_reward_func(x: np.ndarray, u: np.ndarray) -> np.ndarray:
                """
                Implementation of reward function with penalty (pure NumPy)
    
                Parameters:
                    x: State variables, shape (n, 2) or (2,)
                    u: Control variables, shape (n, 2) or (2,)
    
                Returns:
                    Reward values with penalty, shape (n,) or float
                """
                # Ensure at least 2D arrays
                x_2d = np.atleast_2d(x)
                u_2d = np.atleast_2d(u)
    
                # Get batch sizes
                n_x = x_2d.shape[0]
                n_u = u_2d.shape[0]
    
                # Handle broadcasting
                if n_x == 1 and n_u > 1:
                    x_2d = np.tile(x_2d, (n_u, 1))
                    n_x = n_u
                elif n_u == 1 and n_x > 1:
                    u_2d = np.tile(u_2d, (n_x, 1))
                    n_u = n_x
    
                # Calculate original reward
                original_reward = self.reward_func(x_2d, u_2d)
    
                # Calculate constraint values
                constraint_values = self.constraint_func(x_2d, u_2d)
    
                # Ensure constraint values are 2D arrays (n, dim_x)
                if constraint_values.ndim == 1:
                    # If constraint values are scalar or 1D, adjust according to dimension
                    if constraint_values.shape[0] == 1:
                        constraint_values = constraint_values.reshape(-1, 1)
                    else:
                        constraint_values = constraint_values.reshape(-1, self.dim_x)
                elif constraint_values.shape[0] == 1 and n_x > 1:
                    # If constraint function returns broadcasted results, need adjustment
                    constraint_values = np.tile(constraint_values, (n_x, 1))
    
                # Calculate max(H(x,u), 0)
                h_pos = np.maximum(constraint_values, 0.0)
    
                # Calculate L2 norm squared: ||h_pos||² = sum(h_pos², axis=1)
                if h_pos.ndim == 2:
                    norm_sq = np.sum(h_pos ** 2, axis=1)
                else:
                    norm_sq = h_pos ** 2
    
                # Calculate penalty term: exp(γ * ||max(H,0)||²) - 1
                penalty = np.exp(self.gamma_pen * norm_sq) - 1.
    
                # Reward with penalty
                result = original_reward - self.lambda_pen * penalty
    
                # If input is a single point, return scalar
                if x.ndim == 1 and u.ndim == 1:
                    return result[0]
                return result
    
            # Assign function to instance variable
            self.penalty_reward_func = penalty_reward_func
    
    def set_interpolation_method(self, method: str):
        """
        Set interpolation method for value and policy functions
        
        Parameters:
            method: Interpolation method
                    Options: 'linear', 'nearest', 'slinear', 'cubic'
                    
        Returns:
            self: For method chaining
        """
        valid_methods = ['linear', 'nearest', 'slinear', 'cubic']
        if method not in valid_methods:
            raise ValueError(f"Invalid interpolation method: {method}. "
                           f"Valid options are: {valid_methods}")
        
        old_method = self.interpolation_method
        self.interpolation_method = method
        
        print(f"Interpolation method changed from '{old_method}' to '{method}'")
        
        # Update interpolators if they exist
        if self.value_interpolator is not None:
            self._update_value_interpolator()
            print("Value function interpolator updated")
        
        if hasattr(self, 'policy_interpolators'):
            # Recreate policy interpolators if they exist
            # We need to get the policy grid from convergence_info
            if hasattr(self, 'convergence_info') and 'final_policy_grid' in self.convergence_info:
                self._create_policy_interpolators(self.convergence_info['final_policy_grid'])
                print("Policy interpolators updated")
            else:
                print("Warning: Policy grid not available, policy interpolators not updated")
        
        return self  # Return self for method chaining
    
    def set_ppf(
        self, 
        ppf_input: Union[List[Callable], Callable],
        n_quantiles: int = 1000
    ):
        """
        Set PPF function and generate lookup table
        
        Parameters:
            ppf_input: PPF function input, can be:
                      - List[callable]: Each perturbation dimension has its own PPF
                      - callable: All perturbation dimensions share the same PPF
            n_quantiles: Number of quantiles for generating dense table
            
        Returns:
            self: For method chaining
        """
        print(f"Generating PPF lookup table, number of quantiles: {n_quantiles}")
        
        # Generate quantile points (uniformly distributed between 0 and 1)
        quantiles = np.linspace(0, 1, n_quantiles + 2)[1:-1]  # Exclude 0 and 1
        
        if isinstance(ppf_input, list):
            # Each dimension has its own PPF
            assert len(ppf_input) == self.dim_epsilon, \
                f"PPF list length ({len(ppf_input)}) must equal perturbation dimension ({self.dim_epsilon})"
            
            # Generate table for each dimension
            ppf_table = np.zeros((n_quantiles, self.dim_epsilon))
            for i, ppf in enumerate(ppf_input):
                ppf_table[:, i] = ppf(quantiles)
                
        else:
            # All dimensions share the same PPF
            ppf_values = ppf_input(quantiles)
            
            # Ensure output shape is correct
            if ppf_values.ndim == 1:
                # 1D output, broadcast to all dimensions
                ppf_table = np.tile(ppf_values.reshape(-1, 1), (1, self.dim_epsilon))
            elif ppf_values.ndim == 2 and ppf_values.shape[1] == self.dim_epsilon:
                # Already correct shape
                ppf_table = ppf_values
            else:
                raise ValueError(f"PPF function output shape {ppf_values.shape} doesn't match perturbation dimension {self.dim_epsilon}")
        
        self.ppf_table = ppf_table
        self.n_quantiles = n_quantiles
        self.quantiles = quantiles
        
        print(f"PPF table generated, shape: {ppf_table.shape}")
        return self  # Return self for method chaining
    
    def set_grids(
        self,
        x_limits: Union[np.ndarray, List, Tuple],
        u_limits: Union[np.ndarray, List, Tuple],
        x_grid_nums: Union[int, List[int], np.ndarray],
        u_grid_nums: Union[int, List[int], np.ndarray]
    ):
        """
        Set state and control variable grids
        
        Parameters:
            x_limits: State variable bounds, shape (dim_x, 2) or (2,)
            u_limits: Control variable bounds, shape (dim_u, 2) or (2,)
            x_grid_nums: Number of grid points for each state variable dimension
            u_grid_nums: Number of grid points for each control variable dimension
            
        Returns:
            self: For method chaining
        """
        print("Generating state and control variable grids...")
        
        # Process state variable grid
        self._process_grid_limits(x_limits, self.dim_x, "State")
        self._process_grid_nums(x_grid_nums, self.dim_x, "State")
        
        # Generate state grid
        self.x_grid_points = [
            np.linspace(self.x_limits[i, 0], self.x_limits[i, 1], self.x_grid_nums[i])
            for i in range(self.dim_x)
        ]
        
        # Generate all combinations of state grid points
        self.x_grid = self._create_full_grid(self.x_grid_points)
        self.x_grid_shape = tuple(self.x_grid_nums)
        self.x_grid_size = np.prod(self.x_grid_nums)
        
        # Process control variable grid
        self._process_grid_limits(u_limits, self.dim_u, "Control")
        self._process_grid_nums(u_grid_nums, self.dim_u, "Control")
        
        # Generate control grid
        self.u_grid_points = [
            np.linspace(self.u_limits[i, 0], self.u_limits[i, 1], self.u_grid_nums[i])
            for i in range(self.dim_u)
        ]
        
        # Generate all combinations of control grid points
        self.u_grid = self._create_full_grid(self.u_grid_points)
        self.u_grid_shape = tuple(self.u_grid_nums)
        self.u_grid_size = np.prod(self.u_grid_nums)
        
        # Initialize value function (all zeros)
        self.value_func = np.zeros(self.x_grid_shape)
        
        # Initialize value function interpolator
        self._update_value_interpolator()
        
        print(f"State grid: {self.x_grid_shape} shape, {self.x_grid_size} points")
        print(f"Control grid: {self.u_grid_shape} shape, {self.u_grid_size} points")
        print(f"Value function initialized: {self.value_func.shape} shape")
        print(f"Interpolation method: {self.interpolation_method}")
        
        return self  # Return self for method chaining
    
    def _process_grid_limits(self, limits, dim, name):
        """Process grid bounds input"""
        limits = np.asarray(limits)
        
        if limits.shape == (2,):
            # All dimensions share the same bounds
            limits = np.tile(limits, (dim, 1))
        elif limits.shape == (dim, 2):
            # Each dimension has its own bounds
            pass
        else:
            raise ValueError(f"{name} variable bounds shape {limits.shape} incorrect, should be (2,) or ({dim}, 2)")
        
        # Store to corresponding attribute
        if name == "State":
            self.x_limits = limits
        else:
            self.u_limits = limits
    
    def _process_grid_nums(self, grid_nums, dim, name):
        """Process grid points number input"""
        if isinstance(grid_nums, (int, np.integer)):
            # All dimensions share the same number of grid points
            grid_nums = [grid_nums] * dim
        elif isinstance(grid_nums, (list, np.ndarray)):
            # Each dimension has its own number of grid points
            assert len(grid_nums) == dim, \
                f"{name} variable grid points length ({len(grid_nums)}) must equal dimension ({dim})"
        else:
            raise TypeError(f"{name} variable grid points must be int or list/array")
        
        # Store to corresponding attribute
        if name == "State":
            self.x_grid_nums = grid_nums
        else:
            self.u_grid_nums = grid_nums
    
    def _create_full_grid(self, grid_points):
        """Create all combination points of full grid"""
        # Use meshgrid to generate all combinations
        mesh = np.meshgrid(*grid_points, indexing='ij')
        
        # Stack meshgrid results and reshape to (n_points, dim) shape
        full_grid = np.stack(mesh, axis=-1).reshape(-1, len(grid_points))
        
        return full_grid
    
    def _update_value_interpolator(self):
        """Update value function interpolator"""
        if self.x_grid_points is not None and self.value_func is not None:
            # Create RegularGridInterpolator with specified method
            self.value_interpolator = interpolate.RegularGridInterpolator(
                self.x_grid_points,
                self.value_func,
                method=self.interpolation_method,
                bounds_error=False,
                fill_value=None  # We'll handle extrapolation ourselves
            )
    
    def _create_policy_interpolators(self, policy_grid):
        """
        Create policy interpolators for each control dimension
        
        Parameters:
            policy_grid: Policy grid with shape (x_grid_shape + (dim_u,))
        """
        self.policy_interpolators = []
        for d in range(self.dim_u):
            # Extract policy grid for this dimension
            policy_slice = policy_grid[..., d]
            # Create interpolator with specified method
            policy_interp = interpolate.RegularGridInterpolator(
                self.x_grid_points,
                policy_slice,
                method=self.interpolation_method,
                bounds_error=False,
                fill_value=None
            )
            self.policy_interpolators.append(policy_interp)
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get current solver parameters
        
        Returns:
            Dictionary of current solver parameters
        """
        params = {
            'dim_x': self.dim_x,
            'dim_u': self.dim_u,
            'dim_epsilon': self.dim_epsilon,
            'n_samples': self.n_samples,
            'beta': self.beta,
            'lambda_pen': self.lambda_pen,
            'gamma_pen': self.gamma_pen,
            'inf_replace': self.inf_replace,
            'interpolation_method': self.interpolation_method,
            'has_constraints': self.constraint_func is not None,
            'is_solved': self.is_solved,
            'solution_method': self.solution_method
        }
        
        # Add grid information if available
        if self.x_grid is not None:
            params['x_grid_shape'] = self.x_grid_shape
            params['x_grid_size'] = self.x_grid_size
            params['x_limits'] = self.x_limits.tolist() if hasattr(self, 'x_limits') else None
        
        if self.u_grid is not None:
            params['u_grid_shape'] = self.u_grid_shape
            params['u_grid_size'] = self.u_grid_size
            params['u_limits'] = self.u_limits.tolist() if hasattr(self, 'u_limits') else None
        
        # Add convergence info if available
        if self.convergence_info is not None:
            params['converged'] = self.convergence_info.get('converged', False)
            params['iterations'] = self.convergence_info.get('iterations', 0) or \
                                  self.convergence_info.get('policy_iterations', 0)
        
        return params
    
    def set_params(self, **kwargs):
        """
        Set solver parameters
        
        Parameters:
            **kwargs: Parameters to update
                     Supported parameters: beta, lambda_pen, gamma_pen, 
                     inf_replace, interpolation_method, n_samples
        
        Returns:
            self for method chaining
        """
        valid_params = ['beta', 'lambda_pen', 'gamma_pen', 'inf_replace', 
                       'interpolation_method', 'n_samples']
        
        changes = []
        
        for key, value in kwargs.items():
            if key not in valid_params:
                warnings.warn(f"Parameter '{key}' is not a valid solver parameter. "
                            f"Valid parameters are: {valid_params}")
                continue
            
            # Check if value is different
            old_value = getattr(self, key)
            if old_value != value:
                setattr(self, key, value)
                changes.append((key, old_value, value))
                
                # Special handling for certain parameters
                if key in ['lambda_pen', 'gamma_pen']:
                    # Reinitialize penalty reward function
                    self._initialize_penalty_reward_func()
                    self.is_solved = False  # Mark as not solved since parameters changed
        
        # Print changes if any
        if changes:
            print("Parameter changes:")
            for key, old_val, new_val in changes:
                print(f"  {key}: {old_val} -> {new_val}")
            
            # Extract changed keys for checking
            changed_keys = [key for key, _, _ in changes]  # Get just the keys
            
            # If interpolation method changed, update interpolators
            if 'interpolation_method' in changed_keys:  # Fixed: Check using changed_keys list
                if self.value_interpolator is not None:
                    self._update_value_interpolator()
                if hasattr(self, 'policy_interpolators'):
                    # Need to check if we have convergence_info to get policy grid
                    if hasattr(self, 'convergence_info') and 'final_policy_grid' in self.convergence_info:
                        self._create_policy_interpolators(self.convergence_info['final_policy_grid'])
            
            print(f"Total {len(changes)} parameter(s) updated.")
        
        return self
    
    def compute_penalty_rewards(self):
        """
        Compute penalty-augmented reward values for all points in the grid
        
        Returns:
            Reward matrix with shape (x_grid_size, u_grid_size)
        """
        if self.x_grid is None or self.u_grid is None:
            raise ValueError("Please set grids first (set_grids)")
        
        print(f"Computing penalty-augmented reward values...")
        print(f"  State points: {self.x_grid_size}, Control points: {self.u_grid_size}")
        
        # Pre-allocate reward matrix
        reward_matrix = np.zeros((self.x_grid_size, self.u_grid_size))
        
        # For memory efficiency, process in batches
        batch_size = min(1000, self.u_grid_size)
        
        for i in range(0, self.u_grid_size, batch_size):
            end_idx = min(i + batch_size, self.u_grid_size)
            u_batch = self.u_grid[i:end_idx]
            
            # Calculate reward for each state point
            # We need broadcast computation: for each state point, calculate reward for all control points
            # This can be achieved via vectorization
            
            # Expand state grid to match control batch
            # Shape: (x_grid_size * batch_size, dim_x)
            x_expanded = np.repeat(self.x_grid, len(u_batch), axis=0)
            u_expanded = np.tile(u_batch, (self.x_grid_size, 1))
            
            # Calculate reward
            batch_rewards = self.penalty_reward_func(x_expanded, u_expanded)
            
            # Reshape to matrix
            reward_matrix[:, i:end_idx] = batch_rewards.reshape(self.x_grid_size, -1)
            
            if i % (batch_size * 10) == 0 or i == 0:
                print(f"    Progress: {end_idx}/{self.u_grid_size} control points")
        
        # Replace infinite values
        reward_matrix = np.where(np.isinf(reward_matrix), 
                                -self.inf_replace if np.any(reward_matrix < 0) else self.inf_replace,
                                reward_matrix)
        
        print(f"Reward matrix computed, shape: {reward_matrix.shape}")
        return reward_matrix
    
    def sample_perturbations(self, n_samples: Optional[int] = None):
        """
        Sample perturbations from PPF table
        
        Parameters:
            n_samples: Number of samples, if None use n_samples from initialization
            
        Returns:
            Sampled perturbations, shape (n_samples, dim_epsilon)
        """
        if self.ppf_table is None:
            raise ValueError("Please set PPF function first (set_ppf)")
        
        n_samples = n_samples or self.n_samples
        
        # Randomly select quantile indices
        indices = np.random.randint(0, self.n_quantiles, size=(n_samples, self.dim_epsilon))
        
        # Sample from PPF table
        samples = np.zeros((n_samples, self.dim_epsilon))
        for d in range(self.dim_epsilon):
            samples[:, d] = self.ppf_table[indices[:, d], d]
        
        return samples
    
    def compute_next_state_values(
        self, 
        x_points: np.ndarray, 
        u_points: np.ndarray,
        epsilon_samples: Optional[np.ndarray] = None
    ):
        """
        Compute expected value function of next period states
    
        Parameters:
            x_points: State points, shape (n_x, dim_x)
            u_points: Control points, shape (n_u, dim_u) or (dim_u,)
            epsilon_samples: Perturbation samples, if None auto-sample
    
        Returns:
            Expected value function of next period states, shape (n_x, n_u)
        """
        if epsilon_samples is None:
            epsilon_samples = self.sample_perturbations()
    
        n_samples = epsilon_samples.shape[0]
    
        # Ensure u_points are 2D arrays
        u_points = np.atleast_2d(u_points)
    
        n_x = x_points.shape[0]
        n_u = u_points.shape[0]
    
        # We need to compute all combinations of (x_i, u_j, epsilon_k)
        # Total combinations: n_x * n_u * n_samples
    
        # 1. Expand x_points: each x_i repeated (n_u * n_samples) times
        # First, each x_i repeated n_u times
        x_repeat_u = np.repeat(x_points, n_u, axis=0)  # (n_x * n_u, dim_x)
        # Then, repeat the result n_samples times
        x_expanded = np.repeat(x_repeat_u, n_samples, axis=0)  # (n_x * n_u * n_samples, dim_x)
    
        # 2. Expand u_points: each u_j repeated n_samples times, then repeat whole result n_x times
        # First, each u_j repeated n_samples times
        u_repeat_eps = np.repeat(u_points, n_samples, axis=0)  # (n_u * n_samples, dim_u)
        # Then, repeat the result n_x times
        u_expanded = np.tile(u_repeat_eps, (n_x, 1))  # (n_x * n_u * n_samples, dim_u)
    
        # 3. Expand epsilon_samples: repeat whole result (n_x * n_u) times
        epsilon_expanded = np.tile(epsilon_samples, (n_x * n_u, 1))  # (n_x * n_u * n_samples, dim_epsilon)
    
        # Now all arrays have shape (n_x * n_u * n_samples, dim)
        # Compute next period states
        next_states = self.transition_func(x_expanded, u_expanded, epsilon_expanded)
    
        # Use interpolator to compute value function of next period states
        next_values = self.interpolate_values(next_states)
    
        # Compute expectation (average): for each (x_i, u_j) combination, average over n_samples perturbations
        # First reshape next_values to (n_x, n_u, n_samples)
        next_values_reshaped = next_values.reshape(n_x, n_u, n_samples)
        # Then average along the last dimension (perturbation dimension)
        expected_values = np.mean(next_values_reshaped, axis=2)  # shape (n_x, n_u)
    
        return expected_values
    
    def interpolate_values(self, points: np.ndarray):
        """
        Use interpolator to compute value function at points
        
        Parameters:
            points: Points to interpolate, shape (n, dim_x)
            
        Returns:
            Interpolated values, shape (n,)
        """
        if self.value_interpolator is None:
            raise ValueError("Value function interpolator not initialized, please set grids first")
        
        # Use interpolator to compute
        values = self.value_interpolator(points)
        
        # Handle extrapolation: for points outside grid, use nearest grid point value
        # First check which points are outside grid
        for d in range(self.dim_x):
            below_mask = points[:, d] < self.x_limits[d, 0]
            above_mask = points[:, d] > self.x_limits[d, 1]
            
            # For points below lower bound, use lower bound value
            if np.any(below_mask):
                # Create boundary points
                boundary_points = points.copy()
                boundary_points[below_mask, d] = self.x_limits[d, 0]
                values[below_mask] = self.value_interpolator(boundary_points[below_mask])
            
            # For points above upper bound, use upper bound value
            if np.any(above_mask):
                boundary_points = points.copy()
                boundary_points[above_mask, d] = self.x_limits[d, 1]
                values[above_mask] = self.value_interpolator(boundary_points[above_mask])
        
        return values
    
    def compute_current_values(
        self,
        reward_values: np.ndarray,
        next_state_values: np.ndarray
    ):
        """
        Compute current period value function
        
        Parameters:
            reward_values: Current period reward values, shape (n,)
            next_state_values: Expected value function of next period states, shape (n,)
            
        Returns:
            Current period value function, shape (n,)
        """
        # Bellman equation: V = R + β * E[V']
        current_values = reward_values + self.beta * next_state_values
        
        return current_values
    
    def update_value_function(self, new_values: np.ndarray):
        """
        Update value function
        
        Parameters:
            new_values: New value function values, shape same as x_grid
        """
        if new_values.shape != self.x_grid_shape:
            raise ValueError(f"New value function shape {new_values.shape} doesn't match grid shape {self.x_grid_shape}")
        
        self.value_func = new_values
        self._update_value_interpolator()
    
    def solve(
        self,
        method: str = 'value_iteration',
        tol: float = 1e-6,
        max_iter: int = 1000,
        max_value_iter: int = 100,  # Only for policy_iteration
        verbose: bool = True,
        plot_progress: bool = False,
        **kwargs
    ) -> 'MDPSolver':
        """
        Solve MDP problem, supporting multiple solution algorithms
    
        Parameters:
            method: Solution method, options:
                   - 'value_iteration': Value function iteration
                   - 'policy_iteration': Policy function iteration
                   - 'modified_policy_iteration': Modified policy iteration (if implemented)
            tol: Convergence tolerance
            max_iter: Maximum iterations (for value iteration) or maximum policy iterations (for policy iteration)
            max_value_iter: Maximum value function iterations per policy evaluation (only for policy_iteration)
            verbose: Whether to print iteration information
            plot_progress: Whether to plot convergence process
            **kwargs: Other algorithm-specific parameters
    
        Returns:
            self: Returns solver instance itself for method chaining
    
        Usage Example:
            ```
            solver = MDPSolver(...)
            solver.set_grids(...)
            solver.set_ppf(...)
            
            # Value function iteration solution
            result = solver.solve(
                method='value_iteration',
                tol=1e-6,
                max_iter=1000,
                verbose=True
            )
            
            # Policy iteration solution
            result = solver.solve(
                method='policy_iteration',
                tol=1e-6,
                max_policy_iter=50,
                max_value_iter=100,
                verbose=True
            )
            
            # Use solution results
            optimal_control = result.act([0.5, 0.3])
            ```
        """
        
        # Check if necessary settings are complete
        if self.x_grid is None or self.u_grid is None:
            raise ValueError("Please set grids first (set_grids)")
        
        if self.ppf_table is None:
            warnings.warn("PPF function not set, will use default perturbation sampling method")
        
        if verbose:
            print("\n" + "="*60)
            print(f"Starting MDP problem solution")
            print("="*60)
            print(f"Solution method: {method}")
            print(f"State grid size: {self.x_grid_size}")
            print(f"Control grid size: {self.u_grid_size}")
            print(f"Perturbation samples: {self.n_samples}")
            print(f"Discount factor: {self.beta}")
            print(f"Interpolation method: {self.interpolation_method}")
        
        start_time = time.time()
        
        # Call corresponding solution algorithm based on chosen method
        if method.lower() == 'value_iteration':
            # Value function iteration
            if verbose:
                print(f"Convergence tolerance: {tol}, Maximum iterations: {max_iter}")
            
            # Call value function iteration
            policy_func, value_func_interp, convergence_info = self.value_iteration(
                tol=tol,
                max_iter=max_iter,
                verbose=verbose,
                plot_progress=plot_progress
            )
            
            # Store convergence information
            self.convergence_info = convergence_info
            self.solution_method = 'value_iteration'
            
        elif method.lower() == 'policy_iteration':
            # Policy function iteration
            max_policy_iter = max_iter  # Rename to match parameter name
            if verbose:
                print(f"Policy iteration tolerance: {tol}")
                print(f"Maximum policy iterations: {max_policy_iter}")
                print(f"Maximum value iterations per policy evaluation: {max_value_iter}")
            
            # Call policy function iteration
            policy_func, value_func_interp, convergence_info = self.policy_iteration(
                tol=tol,
                max_policy_iter=max_policy_iter,
                max_value_iter=max_value_iter,
                verbose=verbose,
                plot_progress=plot_progress
            )
            
            # Store convergence information
            self.convergence_info = convergence_info
            self.solution_method = 'policy_iteration'
            
        elif method.lower() == 'modified_policy_iteration':
            # Modified policy iteration (if implemented)
            # Can add modified_policy_iteration method call here
            # If not implemented, give prompt
            raise NotImplementedError("modified_policy_iteration method not yet implemented")
            
        else:
            raise ValueError(f"Unsupported solution method: {method}. Options: 'value_iteration', 'policy_iteration'")
        
        # Calculate total time
        total_time = time.time() - start_time
        
        # Store solution results
        self.policy_func = policy_func
        self.value_func_interp = value_func_interp
        self.solution_time = total_time
        self.is_solved = True
        
        if verbose:
            print("\n" + "="*60)
            print(f"Solution completed")
            print("="*60)
            print(f"Solution method: {self.solution_method}")
            print(f"Solution time: {total_time:.3f} seconds")
            
            if hasattr(self, 'convergence_info'):
                if self.solution_method == 'value_iteration':
                    print(f"Iterations: {self.convergence_info['iterations']}")
                    print(f"Converged: {self.convergence_info['converged']}")
                    print(f"Final maximum change: {self.convergence_info['final_max_change']:.6e}")
                elif self.solution_method == 'policy_iteration':
                    print(f"Policy iterations: {self.convergence_info['policy_iterations']}")
                    print(f"Converged: {self.convergence_info['converged']}")
                    print(f"Final policy change states: {self.convergence_info['final_policy_change']}")
            
            print(f"Value function range: [{np.min(self.value_func):.6f}, {np.max(self.value_func):.6f}]")
            print("="*60)
        
        return self
    
    def value_iteration(
        self,
        tol: float = 1e-6,
        max_iter: int = 1000,
        verbose: bool = True,
        plot_progress: bool = False
    ):
        """
        Execute value function iteration algorithm
    
        Parameters:
            tol: Convergence tolerance
            max_iter: Maximum iterations
            verbose: Whether to print iteration information
            plot_progress: Whether to plot iteration process
    
        Returns:
            policy_func: Optimal policy function (interpolator)
            value_func_interp: Optimal value function interpolator
            convergence_info: Convergence information dictionary
        """
        if self.x_grid is None or self.u_grid is None:
            raise ValueError("Please set grids first (set_grids)")
    
        if verbose:
            print("\n" + "="*60)
            print("Starting value function iteration")
            print("="*60)
            print(f"State grid size: {self.x_grid_size}")
            print(f"Control grid size: {self.u_grid_size}")
            print(f"Perturbation samples: {self.n_samples}")
            print(f"Convergence tolerance: {tol}, Maximum iterations: {max_iter}")
            print(f"Interpolation method: {self.interpolation_method}")
    
        # 1. Compute penalty-augmented reward matrix
        if verbose:
            print("\n1. Computing penalty-augmented reward matrix...")
        reward_matrix = self.compute_penalty_rewards()  # shape (n_x, n_u)
    
        # 2. Initialize value function (if not initialized before)
        if self.value_func is None:
            self.value_func = np.zeros(self.x_grid_shape)
            self._update_value_interpolator()
    
        # For storing iteration process
        value_history = []
        max_change_history = []
        iteration_times = []
    
        # 3. Value function iteration main loop
        for iter_num in range(max_iter):
            iter_start_time = time.time()
    
            if verbose and (iter_num % 10 == 0 or iter_num < 5):
                print(f"\nIteration {iter_num}:")
    
            # 3a. Compute expected value of next period states
            # Note: Here we compute expected values for all state-control combinations
            next_state_values = self.compute_next_state_values(
                self.x_grid, self.u_grid
            )  # shape (n_x, n_u)
    
            # 3b. Compute Q-values: current reward + discounted future expected value
            Q_values = reward_matrix + self.beta * next_state_values
    
            # 3c. For each state, find control action that maximizes Q-value
            # New value function: maximum Q-value for each state
            new_value_func = np.max(Q_values, axis=1)  # shape (n_x,)
    
            # Policy: best control index corresponding to each state
            best_u_indices = np.argmax(Q_values, axis=1)  # shape (n_x,)
    
            # Reshape value function to grid shape
            new_value_func_grid = new_value_func.reshape(self.x_grid_shape)
    
            # Compute value function change
            value_change = np.abs(new_value_func_grid - self.value_func)
            max_change = np.max(value_change)
    
            # Record iteration information
            value_history.append(new_value_func_grid.copy())
            max_change_history.append(max_change)
            iteration_times.append(time.time() - iter_start_time)
    
            # Update value function
            old_value_func = self.value_func.copy()
            self.value_func = new_value_func_grid
            self._update_value_interpolator()
    
            if verbose and (iter_num % 10 == 0 or iter_num < 5):
                print(f"  Value function range: [{np.min(new_value_func_grid):.6f}, {np.max(new_value_func_grid):.6f}]")
                print(f"  Maximum change: {max_change:.6e}")
                print(f"  Iteration time: {iteration_times[-1]:.3f} seconds")
    
            # Check convergence
            if max_change < tol:
                if verbose:
                    print(f"\nConverged at iteration {iter_num}, maximum change: {max_change:.6e} < {tol}")
                break
    
        # If maximum iterations reached without convergence
        if iter_num == max_iter - 1 and max_change >= tol:
            if verbose:
                print(f"\nWarning: Reached maximum iterations {max_iter} without convergence")
                print(f"Final maximum change: {max_change:.6e} > {tol}")
    
        # 4. Construct optimal policy grid
        # Convert best control indices to actual control values
        best_u_values = np.zeros((self.x_grid_size, self.dim_u))
        for i, idx in enumerate(best_u_indices):
            best_u_values[i] = self.u_grid[idx]
    
        # Reshape to grid shape
        policy_grid = best_u_values.reshape(self.x_grid_shape + (self.dim_u,))
    
        # 5. Create interpolators
        if verbose:
            print("\nBuilding interpolators...")
    
        # Value function interpolator (already updated via _update_value_interpolator)
        value_func_interp = self.value_interpolator
    
        # Create policy interpolators
        self._create_policy_interpolators(policy_grid)
    
        # Wrap policy function
        def policy_func(x):
            """
            Optimal policy function
    
            Parameters:
                x: State points, shape (n, dim_x) or (dim_x,)
    
            Returns:
                Optimal control actions, shape (n, dim_u) or (dim_u,)
            """
            x_arr = np.asarray(x)
            was_1d = x_arr.ndim == 1
            if was_1d:
                x_arr = x_arr.reshape(1, -1)
    
            n = x_arr.shape[0]
            result = np.zeros((n, self.dim_u))
    
            for d in range(self.dim_u):
                result[:, d] = self.policy_interpolators[d](x_arr)
    
            # Ensure controls are within bounds
            for d in range(self.dim_u):
                result[:, d] = np.clip(result[:, d], 
                                       self.u_limits[d, 0], 
                                       self.u_limits[d, 1])
    
            if was_1d:
                return result[0]
            return result
    
        # 6. Record convergence information
        convergence_info = {
            'iterations': iter_num + 1,
            'max_change_history': np.array(max_change_history),
            'value_history': np.array(value_history),
            'iteration_times': np.array(iteration_times),
            'converged': max_change < tol,
            'final_max_change': max_change,
            'final_policy_grid': policy_grid
        }
    
        if verbose:
            print("\n" + "="*60)
            print("Value function iteration completed")
            print("="*60)
            print(f"Total iterations: {convergence_info['iterations']}")
            print(f"Converged: {convergence_info['converged']}")
            print(f"Final maximum change: {convergence_info['final_max_change']:.6e}")
            print(f"Total computation time: {np.sum(convergence_info['iteration_times']):.3f} seconds")
            print(f"Value function range: [{np.min(self.value_func):.6f}, {np.max(self.value_func):.6f}]")
    
        # Record tolerance in convergence info
        convergence_info['tol'] = tol
        
        # 7. Plot iteration process (if enabled)
        if plot_progress:
            self._plot_value_iteration_convergence(convergence_info, tol=tol)

        # 8. Return policy function
        self.policy_func = policy_func
        
        return policy_func, value_func_interp, convergence_info
    
    def _plot_value_iteration_convergence(self, convergence_info, tol=None):
        """
        Plot the convergence process of value iteration (log scale on y-axis)
        
        Parameters:
            convergence_info: Dictionary containing 'max_change_history'
            tol: Convergence tolerance
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Warning: matplotlib not installed, cannot plot convergence graph")
            return
        
        beta = self.beta
        iterations = np.arange(1, len(convergence_info['max_change_history']) + 1)
        max_changes = convergence_info['max_change_history']
        
        if len(max_changes) < 2:
            print("At least 2 iterations of data are required to plot convergence")
            return
        
        # Calculate theoretical convergence upper bound
        d1 = max_changes[0]  # d∞(V1, V0)
        theory_iter = np.arange(1, len(max_changes) + 1)
        theory_changes = d1 * (beta ** (theory_iter - 1))
        
        # Calculate theoretical maximum iterations
        t_b = int(np.ceil(np.log(tol / d1) / np.log(beta))) + 1
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot actual convergence
        plt.semilogy(iterations, max_changes, 'b-o', linewidth=2, markersize=8,
                     label='Actual convergence', alpha=0.8)
        
        # Plot theoretical convergence upper bound
        plt.semilogy(theory_iter, theory_changes, 'r--', linewidth=2,
                     label=f'Theoretical upper bound (β={beta:.3f})', alpha=0.7)
        
        # Mark theoretical maximum iterations
        if t_b <= len(max_changes):
            plt.axvline(x=t_b, color='g', linestyle=':', alpha=0.7,
                       label=f'Theoretical max iterations t={t_b}')
        
        # Mark convergence tolerance line
        plt.axhline(y=tol, color='gray', linestyle='--', alpha=0.5,
                   label=f'Tolerance ε={tol:.0e}')
        
        # Set labels and title
        plt.xlabel('Iterations', fontsize=12)
        plt.ylabel('Max Change (log scale)', fontsize=12)
        plt.title('Value Function Iteration Convergence Process\n(Linear Convergence Rate β)', fontsize=14, fontweight='bold')
        
        # Add grid
        plt.grid(True, which="both", ls="-", alpha=0.2)
        
        # Add legend
        plt.legend(fontsize=10)
        plt.tight_layout()
        
        # Analyze convergence characteristics
        if len(max_changes) > 2:
            # Calculate actual convergence rate
            actual_rates = max_changes[1:] / max_changes[:-1]
            avg_rate = np.mean(actual_rates[-min(5, len(actual_rates)):])
            
            # Calculate convergence speedup ratio (actual vs theoretical)
            if t_b > 0 and len(iterations) > 0:
                conv_ratio = len(iterations) / t_b
                print(f"Convergence Analysis:")
                print(f"  Theoretical convergence rate: β = {beta:.4f}")
                print(f"  Actual average convergence rate: {avg_rate:.4f}")
                print(f"  Theoretical maximum iterations: t_b = {t_b}")
                print(f"  Actual iterations: {len(iterations)}")
                print(f"  Speedup ratio (actual/theoretical): {conv_ratio:.2f}")
        
        plt.show()
    
    def policy_iteration(
            self,
            tol: float = 1e-6,
            max_policy_iter: int = 50,
            max_value_iter: int = 100,
            verbose: bool = True,
            plot_progress: bool = False
        ):
        """
        Execute policy iteration algorithm
        
        Parameters:
            tol: Convergence tolerance
            max_policy_iter: Maximum policy iterations
            max_value_iter: Maximum value iterations per policy evaluation
            verbose: Whether to print iteration information
            plot_progress: Whether to plot iteration progress
            
        Returns:
            policy_func: Optimal policy function (interpolator)
            value_func_interp: Optimal value function interpolator
            convergence_info: Convergence information dictionary
        """
        if self.x_grid is None or self.u_grid is None:
            raise ValueError("Please set grids first (set_grids)")
        
        import time
        
        if verbose:
            print("\n" + "="*60)
            print("Starting Policy Iteration")
            print("="*60)
            print(f"State grid size: {self.x_grid_size}")
            print(f"Control grid size: {self.u_grid_size}")
            print(f"Perturbation samples: {self.n_samples}")
            print(f"Policy iteration tolerance: {tol}, Max policy iterations: {max_policy_iter}")
            print(f"Max iterations per policy evaluation: {max_value_iter}")
        
        # 1. Calculate penalty-adjusted reward matrix
        if verbose:
            print("\n1. Calculating penalty-adjusted reward matrix...")
        reward_matrix = self.compute_penalty_rewards()  # Shape (n_x, n_u)
        
        # 2. Initialize policy function (choose control with max reward as initial policy)
        if verbose:
            print("\n2. Initializing policy function...")
        
        # Initialization: choose control with maximum reward for each state
        initial_policy_indices = np.argmax(reward_matrix, axis=1)  # Shape (n_x,)
        
        # Convert policy indices to policy value grid
        policy_indices = initial_policy_indices.copy()
        policy_grid = np.zeros((self.x_grid_size, self.dim_u))
        for i, idx in enumerate(policy_indices):
            policy_grid[i] = self.u_grid[idx]
        
        policy_grid = policy_grid.reshape(self.x_grid_shape + (self.dim_u,))
        
        # For storing iteration process
        policy_change_history = []
        value_change_history = []
        policy_iteration_times = []
        value_evaluation_times = []
        
        # 3. Main policy iteration loop
        for policy_iter in range(max_policy_iter):
            policy_iter_start_time = time.time()
            
            if verbose:
                print(f"\nPolicy iteration {policy_iter}:")
            
            # 3.1 Policy evaluation: compute value function under current policy
            if verbose:
                print(f"  Policy evaluation...")
            
            # Get rewards under current policy
            policy_rewards = np.zeros(self.x_grid_size)
            for i in range(self.x_grid_size):
                u_idx = policy_indices[i]
                policy_rewards[i] = reward_matrix[i, u_idx]
            
            policy_rewards = policy_rewards.reshape(self.x_grid_shape)
            
            # Initialize value function
            value_func = np.zeros(self.x_grid_shape)
            
            # Policy evaluation iterations (value iteration)
            value_eval_start_time = time.time()
            for value_iter in range(max_value_iter):
                # Update value function interpolator
                self.value_func = value_func
                self._update_value_interpolator()
                
                # Calculate expected next state values under current policy
                # Note: We only compute control points corresponding to current policy
                next_state_values = np.zeros(self.x_grid_size)
                
                # Batch computation for efficiency
                for i in range(self.x_grid_size):
                    # Get current state and corresponding control
                    x_point = self.x_grid[i:i+1]  # Keep 2D
                    u_idx = policy_indices[i]
                    u_point = self.u_grid[u_idx:u_idx+1]  # Keep 2D
                    
                    # Calculate expected next state value
                    expected_value = self.compute_next_state_values(
                        x_point, u_point
                    )
                    next_state_values[i] = expected_value[0]
                
                next_state_values = next_state_values.reshape(self.x_grid_shape)
                
                # Calculate new value function
                new_value_func = policy_rewards + self.beta * next_state_values
                
                # Calculate value function change
                value_change = np.abs(new_value_func - value_func)
                max_value_change = np.max(value_change)
                
                # Update value function
                value_func = new_value_func.copy()
                
                # Check convergence
                if max_value_change < tol:
                    if verbose and value_iter < 5:  # Only show details for first few iterations
                        print(f"    Value iteration {value_iter}: max change {max_value_change:.6e}")
                    break
                elif verbose and value_iter < 5:  # Only show details for first few iterations
                    print(f"    Value iteration {value_iter}: max change {max_value_change:.6e}")
            
            value_eval_time = time.time() - value_eval_start_time
            value_evaluation_times.append(value_eval_time)
            
            if verbose:
                print(f"  Policy evaluation completed: {value_iter+1} value iterations, time {value_eval_time:.3f}s")
                print(f"  Value function range: [{np.min(value_func):.6f}, {np.max(value_func):.6f}]")
            
            # 3.2 Policy improvement: improve policy based on current value function
            if verbose:
                print(f"  Policy improvement...")
            
            # Update value function interpolator
            self.value_func = value_func
            self._update_value_interpolator()
            
            # Calculate Q-value matrix
            # Note: We reuse compute_next_state_values, but it computes all control points
            # For efficiency, we can directly compute Q-value matrix
            next_state_values_all = self.compute_next_state_values(
                self.x_grid, self.u_grid
            )  # Shape (n_x, n_u)
            
            Q_values = reward_matrix + self.beta * next_state_values_all
            
            # Find optimal control for each state
            new_policy_indices = np.argmax(Q_values, axis=1)  # Shape (n_x,)
            
            # Calculate policy change
            policy_change = np.sum(new_policy_indices != policy_indices)
            policy_change_rate = policy_change / self.x_grid_size
            
            # Record history
            policy_change_history.append(policy_change)
            
            # Update policy
            policy_indices = new_policy_indices.copy()
            
            # Update policy value grid
            policy_grid = np.zeros((self.x_grid_size, self.dim_u))
            for i, idx in enumerate(policy_indices):
                policy_grid[i] = self.u_grid[idx]
            policy_grid = policy_grid.reshape(self.x_grid_shape + (self.dim_u,))
            
            policy_iter_time = time.time() - policy_iter_start_time
            policy_iteration_times.append(policy_iter_time)
            
            if verbose:
                print(f"  Policy improvement completed: changed {policy_change} state policies ({policy_change_rate*100:.2f}%)")
                print(f"  This policy iteration time: {policy_iter_time:.3f}s")
            
            # Check convergence: if policy doesn't change
            if policy_change == 0:
                if verbose:
                    print(f"\nConverged at policy iteration {policy_iter}, policy no longer changes")
                break
        
        # If reached max iterations without convergence
        if policy_iter == max_policy_iter - 1 and policy_change > 0:
            if verbose:
                print(f"\nWarning: Reached maximum policy iterations {max_policy_iter} without convergence")
                print(f"Final policy changes: {policy_change} states")
        
        # 4. Final policy evaluation (using converged value function)
        if verbose:
            print(f"\nPerforming final policy evaluation...")
        
        # Final value function is from last policy evaluation
        final_value_func = value_func
        self.value_func = final_value_func
        self._update_value_interpolator()
        
        # 5. Create interpolators
        if verbose:
            print("\nBuilding interpolators...")
        
        # Value function interpolator (already updated via _update_value_interpolator)
        value_func_interp = self.value_interpolator
        
        # Policy function interpolator
        # Create separate interpolator for each control dimension
        policy_interpolators = []
        for d in range(self.dim_u):
            # Extract policy grid for this dimension
            policy_slice = policy_grid[..., d]
            # Create interpolator
            policy_interp = interpolate.RegularGridInterpolator(
                self.x_grid_points,
                policy_slice,
                method='linear',
                bounds_error=False,
                fill_value=None
            )
            policy_interpolators.append(policy_interp)
        
        # Wrap policy function
        def policy_func(x):
            """
            Optimal policy function
            
            Parameters:
                x: State points, shape (n, dim_x) or (dim_x,)
                
            Returns:
                Optimal control actions, shape (n, dim_u) or (dim_u,)
            """
            x_arr = np.asarray(x)
            was_1d = x_arr.ndim == 1
            if was_1d:
                x_arr = x_arr.reshape(1, -1)
            
            n = x_arr.shape[0]
            result = np.zeros((n, self.dim_u))
            
            for d in range(self.dim_u):
                result[:, d] = policy_interpolators[d](x_arr)
            
            # Ensure controls are within bounds
            for d in range(self.dim_u):
                result[:, d] = np.clip(result[:, d], 
                                       self.u_limits[d, 0], 
                                       self.u_limits[d, 1])
            
            if was_1d:
                return result[0]
            return result
        
        # 6. Record convergence information
        convergence_info = {
            'policy_iterations': policy_iter + 1,
            'policy_change_history': np.array(policy_change_history),
            'value_change_history': np.array(value_change_history),
            'policy_iteration_times': np.array(policy_iteration_times),
            'value_evaluation_times': np.array(value_evaluation_times),
            'converged': policy_change == 0,
            'final_policy_change': policy_change,
            'final_value_func': final_value_func,
            'final_policy_grid': policy_grid
        }
        
        if verbose:
            print("\n" + "="*60)
            print("Policy Iteration Completed")
            print("="*60)
            print(f"Total policy iterations: {convergence_info['policy_iterations']}")
            print(f"Converged: {convergence_info['converged']}")
            print(f"Final policy change states: {convergence_info['final_policy_change']}")
            print(f"Total computation time: {np.sum(convergence_info['policy_iteration_times']):.3f}s")
            print(f"Value function range: [{np.min(final_value_func):.6f}, {np.max(final_value_func):.6f}]")
        
        # Record tolerance in convergence info
        convergence_info['tol'] = tol
        
        # 7. Plot iteration progress (if enabled)
        if plot_progress:
            self._plot_policy_iteration_convergence(convergence_info)
        
        # 8. Return policy function
        self.policy_func = policy_func
        
        return policy_func, value_func_interp, convergence_info
    
    def _plot_policy_iteration_convergence(self, convergence_info):
        """
        Plot the convergence process of policy iteration (log-log scale)
        
        Parameters:
            convergence_info: Dictionary containing 'policy_change_history'
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Warning: matplotlib not installed, cannot plot convergence graph")
            return
        
        iterations = np.arange(1, len(convergence_info['policy_change_history']) + 1)
        policy_changes = convergence_info['policy_change_history']
        
        if len(policy_changes) < 2:
            print("At least 2 iterations of data are required to plot convergence")
            return
        
        # Handle zero values (cannot be zero in log scale)
        plot_values = policy_changes.copy().astype(float)
        plot_values[plot_values == 0] = 0.5  # Replace 0 with 0.5 for display in log scale
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot actual convergence
        plt.loglog(iterations, plot_values, 'b-o', linewidth=2, markersize=8,
                   label='Policy changes', alpha=0.8)
        
        # Mark iteration where policy first stabilizes
        zero_changes = np.where(policy_changes == 0)[0]
        if len(zero_changes) > 0:
            first_stable = zero_changes[0] + 1
            plt.axvline(x=first_stable, color='r', linestyle=':', alpha=0.7,
                       label=f'Policy stabilized (t={first_stable})')
            
            # Mark the point
            plt.scatter([first_stable], [plot_values[first_stable-1]], 
                       color='red', s=100, zorder=5)
        
        # Set labels and title
        plt.xlabel('Iterations (log scale)', fontsize=12)
        plt.ylabel('Policy change states (log scale)', fontsize=12)
        plt.title('Policy Iteration Convergence Process\n(Superlinear Convergence)', fontsize=14, fontweight='bold')
        
        # Add grid
        plt.grid(True, which="both", ls="-", alpha=0.2)
        
        # Add legend
        plt.legend(fontsize=10)
        plt.tight_layout()
        
        # Analyze convergence characteristics
        if len(policy_changes) > 2:
            # Calculate convergence rate (ignore zeros)
            nonzero_changes = policy_changes[policy_changes > 0]
            if len(nonzero_changes) > 2:
                # Calculate change rate between adjacent iterations
                rates = nonzero_changes[1:] / nonzero_changes[:-1]
                
                # Fit quadratic convergence (if possible)
                if len(rates) >= 3:
                    # Check if convergence accelerates (rate decreases)
                    rate_decrease = np.mean(rates[:-1] - rates[1:])
                    
                    print(f"Convergence Analysis:")
                    print(f"  Initial policy changes: {policy_changes[0]} states")
                    if len(zero_changes) > 0:
                        print(f"  Policy stabilized at iteration: {first_stable}")
                    print(f"  Average convergence rate: {np.mean(rates):.4f}")
                    if rate_decrease > 0:
                        print(f"  Convergence characteristic: Superlinear convergence (rate decreasing)")
                    else:
                        print(f"  Convergence characteristic: Linear or sublinear convergence")
        
        plt.show()
    
    def get_state_grid_indices(self, points: np.ndarray):
        """
        Get indices of state points in the grid
        
        Parameters:
            points: State points, shape (n, dim_x)
            
        Returns:
            Linear indices of each point in grid, shape (n,)
        """
        if self.x_grid_points is None:
            raise ValueError("Grid not initialized")
        
        indices = []
        for i in range(points.shape[0]):
            idx = 0
            stride = 1
            for d in range(self.dim_x - 1, -1, -1):
                # Find nearest grid point
                grid_idx = np.argmin(np.abs(self.x_grid_points[d] - points[i, d]))
                idx += grid_idx * stride
                stride *= self.x_grid_nums[d]
            indices.append(idx)
        
        return np.array(indices, dtype=int)
    
    def get_control_grid_indices(self, points: np.ndarray):
        """
        Get indices of control points in the grid
        
        Parameters:
            points: Control points, shape (n, dim_u)
            
        Returns:
            Linear indices of each point in grid, shape (n,)
        """
        if self.u_grid_points is None:
            raise ValueError("Grid not initialized")
        
        indices = []
        for i in range(points.shape[0]):
            idx = 0
            stride = 1
            for d in range(self.dim_u - 1, -1, -1):
                # Find nearest grid point
                grid_idx = np.argmin(np.abs(self.u_grid_points[d] - points[i, d]))
                idx += grid_idx * stride
                stride *= self.u_grid_nums[d]
            indices.append(idx)
        
        return np.array(indices, dtype=int)
    
    def act(self, x: Union[np.ndarray, List]) -> np.ndarray:
        """
        Given current state, provide optimal control variable based on optimal policy function
        
        Parameters:
            x: State points, shape (n, dim_x) or (dim_x,)
            
        Returns:
            Optimal control actions, shape (n, dim_u) or (dim_u,)
            
        Note:
            Need to run value_iteration or policy_iteration first to get policy function
        """
        if not hasattr(self, 'policy_func') or self.policy_func is None:
            raise ValueError("Please run value_iteration or policy_iteration to get policy function first")
        
        return self.policy_func(x)
    
    def predict(
        self, 
        x0: Union[np.ndarray, List], 
        T: int, 
        epsilon_mean: Optional[Union[np.ndarray, List]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Given an initial state, provide deterministic optimal variable paths
        based on policy function and state transition function
        (ignoring randomness, i.e., all perturbations take their mean values)
        
        Parameters:
            x0: Initial state, shape (dim_x,) or (1, dim_x)
            T: Number of time steps to predict (including initial state, so total T time points)
            epsilon_mean: Mean of perturbations, if None then use mean from PPF table
            
        Returns:
            Dictionary with keys:
                - 'states': State path, shape (T, dim_x)
                - 'controls': Control path, shape (T-1, dim_u) or (T, dim_u) (if including initial control)
                - 'rewards': Reward path, shape (T-1,) or (T,) (if including initial reward)
        """
        # Check if policy function is defined
        if not hasattr(self, 'policy_func') or self.policy_func is None:
            raise ValueError("Please run value_iteration or policy_iteration to get policy function first")
        
        # Convert initial state to appropriate format
        x0_arr = np.asarray(x0)
        if x0_arr.ndim == 1:
            x0_arr = x0_arr.reshape(1, -1)
        
        # Get perturbation mean
        if epsilon_mean is None:
            # Calculate mean from PPF table
            if self.ppf_table is not None:
                epsilon_mean = np.mean(self.ppf_table, axis=0)
            else:
                # If no PPF table, assume mean is 0
                epsilon_mean = np.zeros(self.dim_epsilon)
        else:
            epsilon_mean = np.asarray(epsilon_mean)
            if epsilon_mean.ndim == 0 and self.dim_epsilon == 1:
                epsilon_mean = np.array([epsilon_mean])
            elif epsilon_mean.shape != (self.dim_epsilon,):
                raise ValueError(f"epsilon_mean shape should be ({self.dim_epsilon},), but got {epsilon_mean.shape}")
        
        # Initialize storage arrays
        states = np.zeros((T, self.dim_x))
        controls = np.zeros((T-1, self.dim_u))
        rewards = np.zeros(T-1)
        
        # Set initial state
        states[0] = x0_arr[0]
        
        # Generate deterministic path
        for t in range(T-1):
            # Current state
            current_state = states[t:t+1]  # Keep 2D
            
            # Calculate optimal control
            optimal_control = self.policy_func(current_state)
            if optimal_control.ndim == 1:
                optimal_control = optimal_control.reshape(1, -1)
            
            # Calculate reward (using penalty-adjusted reward function)
            current_reward = self.penalty_reward_func(current_state, optimal_control)
            if np.isscalar(current_reward):
                current_reward = np.array([current_reward])
            
            # Calculate next state (using perturbation mean)
            next_state = self.transition_func(current_state, optimal_control, epsilon_mean)
            if next_state.ndim == 1:
                next_state = next_state.reshape(1, -1)
            
            # Store results
            controls[t] = optimal_control[0]
            rewards[t] = current_reward[0]
            states[t+1] = next_state[0]
        
        return {
            'states': states,
            'controls': controls,
            'rewards': rewards,
            'epsilon_mean': epsilon_mean
        }
    
    def simulate(
        self,
        x0: Union[np.ndarray, List],
        T: int,
        n_simulations: int = 100,
        epsilon_generator: Optional[Callable] = None,
        seed: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Given an initial state, simulate state variable changes based on
        policy function and state transition function (considering randomness)
        
        Parameters:
            x0: Initial state, shape (dim_x,) or (1, dim_x)
            T: Number of time steps to simulate (including initial state, so total T time points)
            n_simulations: Number of simulations
            epsilon_generator: Perturbation generator function, if not provided then use sample_perturbations
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary with keys:
                - 'states': State path array, shape (n_simulations, T, dim_x)
                - 'controls': Control path array, shape (n_simulations, T-1, dim_u)
                - 'rewards': Reward path array, shape (n_simulations, T-1)
                - 'epsilon_samples': Perturbation sample array, shape (n_simulations, T-1, dim_epsilon)
        """
        # Check if policy function is defined
        if not hasattr(self, 'policy_func') or self.policy_func is None:
            raise ValueError("Please run value_iteration or policy_iteration to get policy function first")
        
        # Set random seed
        if seed is not None:
            np.random.seed(seed)
        
        # Convert initial state to appropriate format
        x0_arr = np.asarray(x0)
        if x0_arr.ndim == 1:
            x0_arr = x0_arr.reshape(1, -1)
        
        # Initialize storage arrays
        states = np.zeros((n_simulations, T, self.dim_x))
        controls = np.zeros((n_simulations, T-1, self.dim_u))
        rewards = np.zeros((n_simulations, T-1))
        epsilon_samples = np.zeros((n_simulations, T-1, self.dim_epsilon))
        
        # Set initial state for all simulations
        states[:, 0, :] = x0_arr[0]
        
        # Generate perturbations
        if epsilon_generator is None:
            # Use built-in sampling method
            epsilon_all = []
            for _ in range(n_simulations):
                # Sample perturbations for each time step
                epsilon_path = np.zeros((T-1, self.dim_epsilon))
                for t in range(T-1):
                    # Each call to sample_perturbations(1) gives 1 sample
                    eps_sample = self.sample_perturbations(1)
                    epsilon_path[t] = eps_sample[0]
                epsilon_all.append(epsilon_path)
        else:
            # Use user-provided generator
            epsilon_all = []
            for sim in range(n_simulations):
                epsilon_path = epsilon_generator(T-1, self.dim_epsilon)
                if epsilon_path.shape != (T-1, self.dim_epsilon):
                    raise ValueError(f"epsilon_generator should return array with shape ({T-1}, {self.dim_epsilon})")
                epsilon_all.append(epsilon_path)
        
        # Parallelize simulation (using simple for loop, for large-scale simulations consider parallelization)
        for sim in range(n_simulations):
            epsilon_path = epsilon_all[sim]
            
            for t in range(T-1):
                # Current state
                current_state = states[sim, t:t+1]  # Keep 2D
                
                # Calculate optimal control
                optimal_control = self.policy_func(current_state)
                if optimal_control.ndim == 1:
                    optimal_control = optimal_control.reshape(1, -1)
                
                # Calculate reward
                current_reward = self.penalty_reward_func(current_state, optimal_control)
                if np.isscalar(current_reward):
                    current_reward = np.array([current_reward])
                
                # Calculate next state (using perturbation for current time step)
                current_epsilon = epsilon_path[t:t+1]  # Keep 2D
                next_state = self.transition_func(current_state, optimal_control, current_epsilon)
                if next_state.ndim == 1:
                    next_state = next_state.reshape(1, -1)
                
                # Store results
                controls[sim, t] = optimal_control[0]
                rewards[sim, t] = current_reward[0]
                epsilon_samples[sim, t] = current_epsilon[0]
                states[sim, t+1] = next_state[0]
        
        return {
            'states': states,
            'controls': controls,
            'rewards': rewards,
            'epsilon_samples': epsilon_samples,
            'n_simulations': n_simulations,
            'T': T
        }
    
    def set_policy_func(self, policy_func: Callable):
        """
        Set policy function (for externally provided policy function)
        
        Parameters:
            policy_func: Policy function that accepts state and returns control
        """
        self.policy_func = policy_func
    
    def evaluate_policy(
        self,
        policy_func: Callable,
        x0: Union[np.ndarray, List],
        n_simulations: int = 100,
        T: int = 100,
        discount: bool = True,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Evaluate performance of a given policy
        
        Parameters:
            policy_func: Policy function to evaluate
            x0: Initial state
            n_simulations: Number of simulations
            T: Number of time steps per simulation
            discount: Whether to use discount factor
            seed: Random seed
            
        Returns:
            Dictionary containing evaluation results
        """
        # Set random seed
        if seed is not None:
            np.random.seed(seed)
        
        # Save current policy function
        current_policy_func = getattr(self, 'policy_func', None)
        
        try:
            # Use given policy function
            self.set_policy_func(policy_func)
            
            # Run simulation
            results = self.simulate(
                x0=x0,
                T=T,
                n_simulations=n_simulations,
                seed=seed
            )
            
            # Calculate performance metrics
            rewards = results['rewards']
            
            if discount:
                # Calculate discounted cumulative reward
                discount_factors = self.beta ** np.arange(T-1)
                discounted_rewards = rewards * discount_factors
                total_rewards = np.sum(discounted_rewards, axis=1)
            else:
                # Calculate undiscounted cumulative reward
                total_rewards = np.sum(rewards, axis=1)
            
            # Calculate statistics
            mean_total_reward = np.mean(total_rewards)
            std_total_reward = np.std(total_rewards)
            min_total_reward = np.min(total_rewards)
            max_total_reward = np.max(total_rewards)
            
            # Calculate average reward per time step
            mean_reward_per_step = np.mean(rewards, axis=0)
            
            return {
                'mean_total_reward': mean_total_reward,
                'std_total_reward': std_total_reward,
                'min_total_reward': min_total_reward,
                'max_total_reward': max_total_reward,
                'total_rewards': total_rewards,
                'mean_reward_per_step': mean_reward_per_step,
                'simulation_results': results
            }
            
        finally:
            # Restore original policy function
            if current_policy_func is not None:
                self.set_policy_func(current_policy_func)
    
    def plot_simulation_results(
        self,
        simulation_results: Dict[str, np.ndarray],
        figsize: Tuple[int, int] = (15, 10),
        n_samples_plot: int = 10
    ):
        """
        Plot charts of simulation results
        
        Parameters:
            simulation_results: Results dictionary returned by simulate method
            figsize: Figure size
            n_samples_plot: Number of samples to plot (randomly selected from simulations)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Warning: matplotlib not installed, cannot plot simulation results")
            return
        
        states = simulation_results['states']
        controls = simulation_results['controls']
        rewards = simulation_results['rewards']
        n_simulations = simulation_results['n_simulations']
        T = simulation_results['T']
        
        # Randomly select samples to plot
        if n_simulations > n_samples_plot:
            indices = np.random.choice(n_simulations, n_samples_plot, replace=False)
        else:
            indices = np.arange(n_simulations)
        
        # Time axis
        time_steps = np.arange(T)
        time_steps_control = np.arange(T-1)
        
        # Create figure
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        
        # 1. State variable paths (each dimension)
        ax = axes[0, 0]
        for sim_idx in indices:
            for d in range(self.dim_x):
                ax.plot(time_steps, states[sim_idx, :, d], alpha=0.5, linewidth=1)
        
        ax.set_xlabel('Time step')
        ax.set_ylabel('State value')
        ax.set_title(f'State variable paths ({n_samples_plot} samples)')
        ax.grid(True, alpha=0.3)
        
        # 2. Control variable paths (each dimension)
        ax = axes[0, 1]
        for sim_idx in indices:
            for d in range(self.dim_u):
                ax.plot(time_steps_control, controls[sim_idx, :, d], alpha=0.5, linewidth=1)
        
        ax.set_xlabel('Time step')
        ax.set_ylabel('Control value')
        ax.set_title(f'Control variable paths ({n_samples_plot} samples)')
        ax.grid(True, alpha=0.3)
        
        # 3. Reward paths
        ax = axes[1, 0]
        for sim_idx in indices:
            ax.plot(time_steps_control, rewards[sim_idx, :], alpha=0.5, linewidth=1)
        
        ax.set_xlabel('Time step')
        ax.set_ylabel('Reward')
        ax.set_title(f'Reward paths ({n_samples_plot} samples)')
        ax.grid(True, alpha=0.3)
        
        # 4. Cumulative reward distribution
        ax = axes[1, 1]
        cumulative_rewards = np.sum(rewards, axis=1)
        ax.hist(cumulative_rewards, bins=30, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Cumulative reward')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Cumulative reward distribution (n={n_simulations})')
        ax.grid(True, alpha=0.3)
        
        # Add statistical information
        mean_reward = np.mean(cumulative_rewards)
        std_reward = np.std(cumulative_rewards)
        ax.axvline(mean_reward, color='red', linestyle='--', label=f'Mean: {mean_reward:.2f}')
        ax.legend()
        
        # 5. Statistics of state variables at each time step
        ax = axes[2, 0]
        time_steps = np.arange(T)
        
        for d in range(min(3, self.dim_x)):  # Show at most 3 dimensions
            state_mean = np.mean(states[:, :, d], axis=0)
            state_std = np.std(states[:, :, d], axis=0)
            
            ax.plot(time_steps, state_mean, label=f'Dimension {d}', linewidth=2)
            ax.fill_between(time_steps, 
                           state_mean - state_std, 
                           state_mean + state_std, 
                           alpha=0.2)
        
        ax.set_xlabel('Time step')
        ax.set_ylabel('State value')
        ax.set_title('State variable statistics (mean ± std)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. Statistics of control variables at each time step
        ax = axes[2, 1]
        time_steps = np.arange(T-1)
        
        for d in range(min(3, self.dim_u)):  # Show at most 3 dimensions
            control_mean = np.mean(controls[:, :, d], axis=0)
            control_std = np.std(controls[:, :, d], axis=0)
            
            ax.plot(time_steps, control_mean, label=f'Dimension {d}', linewidth=2)
            ax.fill_between(time_steps, 
                           control_mean - control_std, 
                           control_mean + control_std, 
                           alpha=0.2)
        
        ax.set_xlabel('Time step')
        ax.set_ylabel('Control value')
        ax.set_title('Control variable statistics (mean ± std)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print statistical summary
        print("Simulation results statistical summary:")
        print(f"  Number of simulations: {n_simulations}")
        print(f"  Time steps: {T}")
        print(f"  Average cumulative reward: {np.mean(cumulative_rewards):.4f} ± {np.std(cumulative_rewards):.4f}")
        print(f"  Minimum cumulative reward: {np.min(cumulative_rewards):.4f}")
        print(f"  Maximum cumulative reward: {np.max(cumulative_rewards):.4f}")
        
        return fig
    
    def summary(self):
        """Print solver summary information"""
        print("\n" + "="*60)
        print("MDP Solver Summary")
        print("="*60)
        print(f"Dimensions:")
        print(f"  State: {self.dim_x}, Control: {self.dim_u}, Perturbation: {self.dim_epsilon}")
        print(f"Parameters:")
        print(f"  Discount factor: {self.beta}")
        print(f"  Penalty parameters: λ={self.lambda_pen}, γ={self.gamma_pen}")
        print(f"  Perturbation samples: {self.n_samples}")
        print(f"  Interpolation method: {self.interpolation_method}")
        
        if self.x_grid is not None:
            print(f"\nState grid:")
            print(f"  Shape: {self.x_grid_shape}")
            print(f"  Total points: {self.x_grid_size}")
            print(f"  Points per dimension: {self.x_grid_nums}")
            for i in range(self.dim_x):
                print(f"    Dimension {i}: [{self.x_limits[i, 0]:.3f}, {self.x_limits[i, 1]:.3f}]")
        
        if self.u_grid is not None:
            print(f"\nControl grid:")
            print(f"  Shape: {self.u_grid_shape}")
            print(f"  Total points: {self.u_grid_size}")
            print(f"  Points per dimension: {self.u_grid_nums}")
            for i in range(self.dim_u):
                print(f"    Dimension {i}: [{self.u_limits[i, 0]:.3f}, {self.u_limits[i, 1]:.3f}]")
        
        if self.ppf_table is not None:
            print(f"\nPPF table:")
            print(f"  Shape: {self.ppf_table.shape}")
            print(f"  Number of quantiles: {self.n_quantiles}")
        
        if self.value_func is not None:
            print(f"\nValue function:")
            print(f"  Shape: {self.value_func.shape}")
            print(f"  Range: [{np.min(self.value_func):.3f}, {np.max(self.value_func):.3f}]")
        
        if self.is_solved:
            print(f"\nSolution status:")
            print(f"  Solved: Yes")
            print(f"  Solution method: {self.solution_method}")
            print(f"  Solution time: {self.solution_time:.3f} seconds")
            if self.convergence_info:
                print(f"  Converged: {self.convergence_info.get('converged', 'N/A')}")
        else:
            print(f"\nSolution status: Not solved yet")
        
        print("="*60)

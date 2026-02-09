"""
@penglangye@foxmail.com
GPU-Accelerated MDP (Markov Decision Process) Solver

A GPU-accelerated version based on PyTorch, supporting efficient computation for large-scale state spaces.
Maintains the same API interface as the original MDPSolver but uses PyTorch internally for GPU acceleration.

Main Features:
1. Grid-based discretization of state and action spaces
2. Stochastic perturbation sampling
3. Reward functions with penalty terms for constraints
4. Value iteration and policy iteration algorithms
5. Policy simulation and evaluation
6. Result visualization

Main Optimizations:
1. GPU computation using PyTorch tensors
2. Batch matrix operations to reduce loops
3. Efficient interpolation computation
4. Intelligent memory management
5. Automatic device selection (CPU/GPU)

Usage Example:
----------
```python
# Initialize GPU solver
solver = MDPSolverGPU(
    dim_x=2, dim_u=2, dim_epsilon=1, n_samples=100,
    transition_func=transition, reward_func=reward,
    constraint_func=constraint,
    device='cuda'  # Optional: 'cuda', 'cpu' or auto-select
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
import warnings
from typing import Callable, List, Optional, Union, Tuple, Dict, Any

# Import PyTorch
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not installed, GPU acceleration unavailable")

class MDPSolverGPU:
    """
    GPU-Accelerated MDP Solver
    
    Parameters:
        dim_x: State variable dimension
        dim_u: Control variable dimension
        dim_epsilon: Perturbation variable dimension
        n_samples: Number of perturbation samples
        transition_func: State transition function
        reward_func: Reward function
        constraint_func: Constraint function (optional)
        device: Computation device ('cuda', 'cpu', or 'auto')
        beta: Discount factor
        lambda_pen: Penalty weight coefficient
        gamma_pen: Penalty scale parameter
        interpolation_method: Interpolation method (only supports 'linear' and 'nearest')
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
        device: str = 'auto',
        beta: float = 0.95,
        lambda_pen: float = 10.,
        gamma_pen: float = 10.,
        interpolation_method: str = 'linear'
    ):
        # Check if PyTorch is available
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not installed. Please install with: pip install torch")
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Store basic parameters
        self.dim_x = dim_x
        self.dim_u = dim_u
        self.dim_epsilon = dim_epsilon
        self.n_samples = n_samples
        self.beta = beta
        self.lambda_pen = lambda_pen
        self.gamma_pen = gamma_pen
        self.interpolation_method = interpolation_method
        
        # Validate interpolation method
        valid_methods = ['linear', 'nearest']
        if interpolation_method not in valid_methods:
            raise ValueError(f"GPU solver only supports: {valid_methods}")
        
        # Initialize PPF-related attributes
        self.ppf_table = None
        self.n_quantiles = None
        self.quantiles = None
        
        # Store user functions (keep original numpy versions)
        self.transition_func_np = transition_func
        self.reward_func_np = reward_func
        self.constraint_func_np = constraint_func
        
        # Create PyTorch version functions
        self.transition_func = self._create_torch_function(transition_func)
        self.reward_func = self._create_torch_function(reward_func)
        
        if constraint_func is not None:
            self.constraint_func = self._create_torch_function(constraint_func)
            # Create penalty-augmented reward function
            self.penalty_reward_func = self._create_penalty_reward_func()
        else:
            self.constraint_func = None
            self.penalty_reward_func = self.reward_func
        
        # Initialize internal variables
        self.x_grid = None
        self.u_grid = None
        self.x_grid_points = None
        self.u_grid_points = None
        self.x_grid_shape = None
        self.u_grid_shape = None
        self.x_grid_size = None
        self.u_grid_size = None
        self.value_func = None
        
        # Store interpolator
        self.value_interpolator = None
        
        # State flags
        self.is_solved = False
        self.solution_method = None
        self.convergence_info = None
        
        print(f"MDP GPU Solver Initialized:")
        print(f"  Device: {self.device}")
        print(f"  State Dimension: {dim_x}, Control Dimension: {dim_u}, Perturbation Dimension: {dim_epsilon}")
        print(f"  Perturbation Samples: {n_samples}, Discount Factor: {beta}")
        print(f"  Interpolation Method: {interpolation_method}")
        if constraint_func is not None:
            print(f"  Using Constraint Function, Penalty Parameters: λ={lambda_pen}, γ={gamma_pen}")
    
    def _create_torch_function(self, numpy_func: Callable) -> Callable:
        """Convert numpy function to PyTorch function"""
        def torch_func(*args):
            # Convert inputs to numpy arrays (if they are torch tensors)
            args_np = []
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    args_np.append(arg.detach().cpu().numpy())
                else:
                    args_np.append(arg)
            
            # Call original numpy function
            result = numpy_func(*args_np)
            
            # Convert result to torch tensor
            if isinstance(result, np.ndarray):
                return torch.from_numpy(result).float().to(self.device)
            else:
                return torch.tensor(result, dtype=torch.float32, device=self.device)
        
        return torch_func
    
    def _create_penalty_reward_func(self) -> Callable:
        """Create penalty-augmented reward function (PyTorch version)"""
        def penalty_reward_func(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
            # Ensure at least 2D tensors
            if x.dim() == 1:
                x = x.unsqueeze(0)
            if u.dim() == 1:
                u = u.unsqueeze(0)
            
            n_x = x.shape[0]
            n_u = u.shape[0]
            
            # Handle broadcasting
            if n_x == 1 and n_u > 1:
                x = x.repeat(n_u, 1)
                n_x = n_u
            elif n_u == 1 and n_x > 1:
                u = u.repeat(n_x, 1)
                n_u = n_x
            
            # Calculate original reward
            original_reward = self.reward_func(x, u)
            
            # If no constraint function, return original reward directly
            if self.constraint_func is None:
                return original_reward
            
            # Calculate constraint values
            constraint_values = self.constraint_func(x, u)
            
            # Ensure constraint values are 2D tensors
            if constraint_values.dim() == 1:
                constraint_values = constraint_values.unsqueeze(-1)
            
            # Calculate max(H(x,u), 0)
            h_pos = torch.clamp(constraint_values, min=0.0)
            
            # Calculate L2 norm squared
            if h_pos.dim() == 2:
                norm_sq = torch.sum(h_pos ** 2, dim=1)
            else:
                norm_sq = h_pos ** 2
            
            # Calculate penalty term: exp(γ * ||max(H,0)||²) - 1
            penalty = torch.exp(self.gamma_pen * norm_sq) - 1.0
            
            # Reward with penalty
            result = original_reward - self.lambda_pen * penalty
            
            return result
        
        return penalty_reward_func

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
            'interpolation_method': self.interpolation_method,
            'device': str(self.device),
            'has_constraints': self.constraint_func_np is not None,
            'is_solved': self.is_solved,
            'solution_method': self.solution_method
        }
        
        # Add grid information if available
        if self.x_grid is not None:
            params['x_grid_shape'] = self.x_grid_shape
            params['x_grid_size'] = self.x_grid_size
            params['x_limits'] = self.x_limits.tolist() if hasattr(self, 'x_limits') else None
            params['x_grid_nums'] = self.x_grid_nums
        
        if self.u_grid is not None:
            params['u_grid_shape'] = self.u_grid_shape
            params['u_grid_size'] = self.u_grid_size
            params['u_limits'] = self.u_limits.tolist() if hasattr(self, 'u_limits') else None
            params['u_grid_nums'] = self.u_grid_nums
        
        # Add PPF table information if available
        if self.ppf_table is not None:
            params['n_quantiles'] = self.n_quantiles
            params['ppf_table_shape'] = tuple(self.ppf_table.shape)
        
        # Add value function information if available
        if self.value_func is not None:
            params['value_func_shape'] = tuple(self.value_func.shape)
            if len(self.value_func.shape) > 0:
                params['value_func_range'] = [
                    torch.min(self.value_func).item() if hasattr(torch.min(self.value_func), 'item') else float(torch.min(self.value_func)),
                    torch.max(self.value_func).item() if hasattr(torch.max(self.value_func), 'item') else float(torch.max(self.value_func))
                ]
        
        # Add convergence info if available
        if self.convergence_info is not None:
            params['converged'] = self.convergence_info.get('converged', False)
            if 'iterations' in self.convergence_info:
                params['iterations'] = self.convergence_info.get('iterations', 0)
            elif 'policy_iterations' in self.convergence_info:
                params['iterations'] = self.convergence_info.get('policy_iterations', 0)
            
            # Add final change information
            if 'final_max_change' in self.convergence_info:
                params['final_max_change'] = self.convergence_info.get('final_max_change')
            if 'final_policy_change' in self.convergence_info:
                params['final_policy_change'] = self.convergence_info.get('final_policy_change')
        
        # Add solution time if available
        if hasattr(self, 'solution_time'):
            params['solution_time'] = self.solution_time
        
        # Add GPU specific information
        if torch.cuda.is_available() and self.device.type == 'cuda':
            params['gpu_memory_allocated'] = torch.cuda.memory_allocated(self.device)
            params['gpu_memory_reserved'] = torch.cuda.memory_reserved(self.device)
        
        return params
    
    def set_interpolation_method(self, method: str):
        """
        Set interpolation method for value and policy functions
        
        Parameters:
            method: Interpolation method
                    Options: 'linear', 'nearest', 'bilinear', 'trilinear'
                    
        Returns:
            self: For method chaining
        """
        valid_methods = ['linear', 'nearest', 'bilinear', 'trilinear']
        if method not in valid_methods:
            raise ValueError(f"Invalid interpolation method: {method}. "
                           f"Valid options are: {valid_methods}")
        
        old_method = self.interpolation_method
        self.interpolation_method = method
        
        print(f"Interpolation method changed from '{old_method}' to '{method}'")
        
        # Note: GPU solver doesn't maintain interpolators like CPU version
        # The interpolation method is used directly in interpolate_values method
        
        return self
    
    def set_params(self, **kwargs):
        """
        Set solver parameters
        
        Parameters:
            **kwargs: Parameters to update
                     Supported parameters: beta, lambda_pen, gamma_pen, 
                     interpolation_method, n_samples, device
        
        Returns:
            self for method chaining
        """
        valid_params = ['beta', 'lambda_pen', 'gamma_pen', 
                       'interpolation_method', 'n_samples', 'device']
        
        changes = []
        
        for key, value in kwargs.items():
            if key not in valid_params:
                warnings.warn(f"Parameter '{key}' is not a valid solver parameter. "
                            f"Valid parameters are: {valid_params}")
                continue
            
            # Check if value is different
            old_value = getattr(self, key)
            
            # Special handling for device parameter
            if key == 'device':
                if value != old_value:
                    new_device = torch.device(value)
                    if new_device.type == 'cuda' and not torch.cuda.is_available():
                        warnings.warn("CUDA not available, keeping current device")
                        continue
                    
                    # Move all tensors to new device
                    self._move_to_device(new_device)
                    setattr(self, key, new_device)
                    changes.append((key, str(old_value), str(new_device)))
                continue
            
            # For other parameters
            if old_value != value:
                setattr(self, key, value)
                changes.append((key, old_value, value))
                
                # Special handling for certain parameters
                if key in ['lambda_pen', 'gamma_pen']:
                    # Reinitialize penalty reward function
                    if self.constraint_func_np is not None:
                        self.penalty_reward_func = self._create_penalty_reward_func()
                    self.is_solved = False  # Mark as not solved since parameters changed
                
                elif key == 'interpolation_method':
                    # Validate interpolation method
                    valid_methods = ['linear', 'nearest', 'bilinear', 'trilinear']
                    if value not in valid_methods:
                        raise ValueError(f"Invalid interpolation method: {value}. "
                                       f"Valid options are: {valid_methods}")
                    self.is_solved = False  # Mark as not solved since interpolation changed
        
        # Print changes if any
        if changes:
            print("Parameter changes:")
            for key, old_val, new_val in changes:
                print(f"  {key}: {old_val} -> {new_val}")
            
            print(f"Total {len(changes)} parameter(s) updated.")
        
        return self
    
    def _move_to_device(self, new_device: torch.device):
        """
        Move all tensors to a new device
        
        Parameters:
            new_device: Target device
        """
        print(f"Moving solver to device: {new_device}")
        
        # List of tensor attributes to move
        tensor_attrs = [
            'x_grid', 'u_grid', 'value_func', 
            'ppf_table', 'quantiles'
        ]
        
        for attr_name in tensor_attrs:
            if hasattr(self, attr_name) and getattr(self, attr_name) is not None:
                tensor = getattr(self, attr_name)
                if isinstance(tensor, torch.Tensor):
                    setattr(self, attr_name, tensor.to(new_device))
        
        # Update device for future operations
        self.device = new_device
        print(f"All tensors moved to {new_device}")
            
    
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
            x_limits: State variable bounds
            u_limits: Control variable bounds
            x_grid_nums: Number of grid points for each state dimension
            u_grid_nums: Number of grid points for each control dimension
        """
        print("Generating state and control variable grids (GPU version)...")
        
        # Process state grid
        x_limits = np.asarray(x_limits)
        if x_limits.shape == (2,):
            x_limits = np.tile(x_limits, (self.dim_x, 1))
        self.x_limits = x_limits
        
        if isinstance(x_grid_nums, int):
            x_grid_nums = [x_grid_nums] * self.dim_x
        self.x_grid_nums = x_grid_nums
        
        # Generate state grid points
        self.x_grid_points = [
            np.linspace(x_limits[i, 0], x_limits[i, 1], x_grid_nums[i])
            for i in range(self.dim_x)
        ]
        
        # Create full grid (numpy)
        mesh_x = np.meshgrid(*self.x_grid_points, indexing='ij')
        x_grid_np = np.stack(mesh_x, axis=-1).reshape(-1, self.dim_x)
        
        # Convert to PyTorch tensor and move to device
        self.x_grid = torch.from_numpy(x_grid_np).float().to(self.device)
        self.x_grid_shape = tuple(x_grid_nums)
        self.x_grid_size = self.x_grid.shape[0]
        
        # Process control grid
        u_limits = np.asarray(u_limits)
        if u_limits.shape == (2,):
            u_limits = np.tile(u_limits, (self.dim_u, 1))
        self.u_limits = u_limits
        
        if isinstance(u_grid_nums, int):
            u_grid_nums = [u_grid_nums] * self.dim_u
        self.u_grid_nums = u_grid_nums
        
        # Generate control grid points
        self.u_grid_points = [
            np.linspace(u_limits[i, 0], u_limits[i, 1], u_grid_nums[i])
            for i in range(self.dim_u)
        ]
        
        # Create full grid (numpy)
        mesh_u = np.meshgrid(*self.u_grid_points, indexing='ij')
        u_grid_np = np.stack(mesh_u, axis=-1).reshape(-1, self.dim_u)
        
        # Convert to PyTorch tensor and move to device
        self.u_grid = torch.from_numpy(u_grid_np).float().to(self.device)
        self.u_grid_shape = tuple(u_grid_nums)
        self.u_grid_size = self.u_grid.shape[0]
        
        # Initialize value function
        self.value_func = torch.zeros(self.x_grid_shape, device=self.device)
        
        print(f"State Grid: {self.x_grid_shape} shape, {self.x_grid_size} points")
        print(f"Control Grid: {self.u_grid_shape} shape, {self.u_grid_size} points")
        print(f"Value Function Initialized: {self.value_func.shape} shape")
        
        return self
    
    def set_ppf(
        self, 
        ppf_input: Union[List[Callable], Callable],
        n_quantiles: int = 1000
    ):
        """
        Set PPF function and generate lookup table
        
        Parameters:
            ppf_input: PPF function
            n_quantiles: Number of quantiles
        """
        print(f"Generating PPF lookup table, number of quantiles: {n_quantiles}")
        
        # Generate quantile points
        quantiles = np.linspace(0, 1, n_quantiles + 2)[1:-1]
        
        if isinstance(ppf_input, list):
            # Each dimension has its own PPF
            ppf_table = np.zeros((n_quantiles, self.dim_epsilon))
            for i, ppf in enumerate(ppf_input):
                ppf_table[:, i] = ppf(quantiles)
        else:
            # All dimensions share the same PPF
            ppf_values = ppf_input(quantiles)
            if ppf_values.ndim == 1:
                ppf_table = np.tile(ppf_values.reshape(-1, 1), (1, self.dim_epsilon))
            else:
                ppf_table = ppf_values
        
        # Convert to PyTorch tensor
        self.ppf_table = torch.from_numpy(ppf_table).float().to(self.device)
        self.n_quantiles = n_quantiles
        self.quantiles = torch.from_numpy(quantiles).float().to(self.device)
        
        print(f"PPF Table Generated, Shape: {ppf_table.shape}")
        return self
    
    def sample_perturbations(self, n_samples: Optional[int] = None) -> torch.Tensor:
        """
        Sample perturbations from PPF table
        
        Parameters:
            n_samples: Number of samples
            
        Returns:
            Sampled perturbations, shape (n_samples, dim_epsilon)
        """
        if self.ppf_table is None:
            # If no PPF table, generate normal distribution samples
            n_samples = n_samples or self.n_samples
            return torch.randn(n_samples, self.dim_epsilon, device=self.device)
        
        n_samples = n_samples or self.n_samples
        
        # Randomly select quantile indices
        indices = torch.randint(0, self.n_quantiles, (n_samples, self.dim_epsilon), 
                               device=self.device)
        
        # Sample from PPF table
        samples = torch.zeros(n_samples, self.dim_epsilon, device=self.device)
        for d in range(self.dim_epsilon):
            samples[:, d] = self.ppf_table[indices[:, d], d]
        
        return samples
    
    # Interpolation method implementations
    def _interpolate_linear_1d(self, points: torch.Tensor) -> torch.Tensor:
        """1D Linear Interpolation (optimized version)"""
        # Get grid points
        x_points = self.x_grid_points[0]
        x_tensor = torch.tensor(x_points, device=self.device)
        
        # Flatten value function
        value_flat = self.value_func.flatten()
        
        # Find adjacent point indices
        idx_right = torch.searchsorted(x_tensor, points.squeeze(), side='right')
        idx_left = torch.clamp(idx_right - 1, 0, len(x_points) - 1)
        idx_right = torch.clamp(idx_right, 0, len(x_points) - 1)
        
        # Get x and y values of adjacent points
        x_left = x_tensor[idx_left]
        x_right = x_tensor[idx_right]
        y_left = value_flat[idx_left]
        y_right = value_flat[idx_right]
        
        # Avoid division by zero (when adjacent points are the same)
        mask = (x_right - x_left) > 1e-10
        t = torch.zeros_like(points.squeeze())
        t[mask] = (points.squeeze()[mask] - x_left[mask]) / (x_right[mask] - x_left[mask])
        
        # Linear interpolation
        result = y_left + t * (y_right - y_left)
        
        return result
    
    def _interpolate_linear_2d(self, points: torch.Tensor) -> torch.Tensor:
        """2D Bilinear Interpolation (using grid_sample)"""
        # Convert value function to format suitable for grid_sample
        value_tensor = self.value_func.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        
        # Normalize point coordinates to [-1, 1] range
        normalized_points = torch.zeros_like(points)
        
        for d in range(2):
            x_min, x_max = self.x_limits[d]
            # Linear mapping: [x_min, x_max] -> [-1, 1]
            normalized_points[:, d] = 2 * (points[:, d] - x_min) / (x_max - x_min) - 1
        
        # Adjust coordinate order: grid_sample expects (y, x) order
        grid = normalized_points[:, [1, 0]].unsqueeze(0).unsqueeze(2)  # (1, n, 1, 2)
        
        # Use grid_sample for bilinear interpolation
        interpolated = F.grid_sample(
            value_tensor, 
            grid, 
            mode='bilinear', 
            padding_mode='border',
            align_corners=True
        )
        
        # Extract result
        result = interpolated.squeeze()
        
        return result
    
    def _interpolate_linear_3d(self, points: torch.Tensor) -> torch.Tensor:
        """3D Trilinear Interpolation (using grid_sample)"""
        # Convert value function to format suitable for grid_sample
        value_tensor = self.value_func.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
        
        # Normalize point coordinates to [-1, 1] range
        normalized_points = torch.zeros_like(points)
        
        for d in range(3):
            x_min, x_max = self.x_limits[d]
            # Linear mapping: [x_min, x_max] -> [-1, 1]
            normalized_points[:, d] = 2 * (points[:, d] - x_min) / (x_max - x_min) - 1
        
        # Adjust coordinate order: grid_sample expects (z, y, x) order
        grid = normalized_points[:, [2, 1, 0]].unsqueeze(0).unsqueeze(2).unsqueeze(2)  # (1, n, 1, 1, 3)
        
        # Use grid_sample for trilinear interpolation
        interpolated = F.grid_sample(
            value_tensor, 
            grid, 
            mode='bilinear',  # For 3D, 'bilinear' is actually trilinear interpolation
            padding_mode='border',
            align_corners=True
        )
        
        # Extract result
        result = interpolated.squeeze()
        
        return result
    
    def _interpolate_nearest(self, points: torch.Tensor) -> torch.Tensor:
        """
        Nearest Neighbor Interpolation (GPU version)
        
        Parameters:
            points: Points to interpolate, shape (n, dim_x)
            
        Returns:
            Interpolated values, shape (n,)
        """
        # Flatten value function
        value_flat = self.value_func.flatten()
        
        # Calculate distance from each point to grid points
        # Since grid may be large, we use batch processing
        batch_size = min(10000, points.shape[0])
        n_points = points.shape[0]
        results = torch.zeros(n_points, device=self.device)
        
        for i in range(0, n_points, batch_size):
            end_idx = min(i + batch_size, n_points)
            batch_points = points[i:end_idx]
            
            # Calculate distances (using broadcasting)
            distances = torch.cdist(batch_points, self.x_grid)
            
            # Find nearest neighbor indices
            nearest_idx = torch.argmin(distances, dim=1)
            
            # Get values
            results[i:end_idx] = value_flat[nearest_idx]
        
        return results
    
    def _interpolate_multilinear_custom(self, points: torch.Tensor) -> torch.Tensor:
        """
        Custom Multi-Dimensional Linear Interpolation (for arbitrary dimensions)
        
        This method implements generic multi-dimensional linear interpolation,
        but may be slower than grid_sample.
        For high-dimensional cases where grid_sample is not supported, use this method.
        """
        n_points = points.shape[0]
        results = torch.zeros(n_points, device=self.device)
        
        # Interpolate each point
        for i in range(n_points):
            point = points[i]
            
            # Find adjacent grid point indices in each dimension
            lower_indices = []
            upper_indices = []
            weights = []
            
            for d in range(self.dim_x):
                # Find adjacent points in current dimension
                grid_points = torch.tensor(self.x_grid_points[d], device=self.device)
                point_d = point[d]
                
                # Find lower bound index
                idx_lower = torch.searchsorted(grid_points, point_d, side='right') - 1
                idx_lower = torch.clamp(idx_lower, 0, len(grid_points) - 2)
                idx_upper = idx_lower + 1
                
                lower_indices.append(idx_lower)
                upper_indices.append(idx_upper)
                
                # Calculate weight
                if grid_points[idx_upper] - grid_points[idx_lower] > 1e-10:
                    weight = (point_d - grid_points[idx_lower]) / (grid_points[idx_upper] - grid_points[idx_lower])
                else:
                    weight = 0.0
                weights.append(weight)
            
            # Calculate weighted sum of all corner points
            total_value = 0.0
            
            # Iterate over all 2^dim_x corner points
            for corner in range(2**self.dim_x):
                # Get corner point indices
                indices = []
                corner_weight = 1.0
                
                for d in range(self.dim_x):
                    # Decide whether to use lower or upper bound based on d-th bit of corner
                    use_upper = (corner >> d) & 1
                    if use_upper:
                        indices.append(upper_indices[d])
                        corner_weight *= weights[d]
                    else:
                        indices.append(lower_indices[d])
                        corner_weight *= (1 - weights[d])
                
                # Convert multi-dimensional index to linear index
                linear_idx = 0
                stride = 1
                for d in range(self.dim_x - 1, -1, -1):
                    linear_idx += indices[d] * stride
                    stride *= self.x_grid_nums[d]
                
                # Get value at this corner point
                corner_value = self.value_func.flatten()[linear_idx]
                
                # Accumulate
                total_value += corner_weight * corner_value
            
            results[i] = total_value
        
        return results
    
    def interpolate_values(self, points: torch.Tensor) -> torch.Tensor:
        """
        Use interpolator to compute value function at points
        
        Parameters:
            points: Points to interpolate
            
        Returns:
            Interpolated values
        """
        if self.interpolation_method == 'nearest':
            return self._interpolate_nearest(points)
        elif self.interpolation_method == 'linear':
            # Choose different linear interpolation method based on dimension
            if self.dim_x == 1:
                return self._interpolate_linear_1d(points)
            elif self.dim_x == 2:
                return self._interpolate_linear_2d(points)
            elif self.dim_x == 3:
                return self._interpolate_linear_3d(points)
            else:
                # Higher dimensions use custom linear interpolation
                return self._interpolate_multilinear_custom(points)
        elif self.interpolation_method == 'bilinear' and self.dim_x == 2:
            # Explicit bilinear interpolation
            return self._interpolate_linear_2d(points)
        elif self.interpolation_method == 'trilinear' and self.dim_x == 3:
            # Explicit trilinear interpolation
            return self._interpolate_linear_3d(points)
        else:
            raise ValueError(f"Unsupported interpolation method: {self.interpolation_method} for {self.dim_x} dimensions")
    
    def compute_next_state_values(
        self, 
        x_points: torch.Tensor, 
        u_points: torch.Tensor,
        epsilon_samples: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute expected value function of next period states (GPU-optimized version)
        
        Parameters:
            x_points: State points
            u_points: Control points
            epsilon_samples: Perturbation samples
            
        Returns:
            Expected value function of next period states
        """
        if epsilon_samples is None:
            epsilon_samples = self.sample_perturbations()
        
        n_samples = epsilon_samples.shape[0]
        n_x = x_points.shape[0]
        n_u = u_points.shape[0]
        
        # Optimization 1: Use tensor operations to avoid explicit loops
        # Expand dimensions for broadcasting
        x_expanded = x_points.unsqueeze(1).unsqueeze(2)  # [n_x, 1, 1, dim_x]
        u_expanded = u_points.unsqueeze(0).unsqueeze(2)  # [1, n_u, 1, dim_u]
        eps_expanded = epsilon_samples.unsqueeze(0).unsqueeze(1)  # [1, 1, n_samples, dim_epsilon]
        
        # Repeat to match dimensions
        x_expanded = x_expanded.expand(n_x, n_u, n_samples, self.dim_x)
        u_expanded = u_expanded.expand(n_x, n_u, n_samples, self.dim_u)
        eps_expanded = eps_expanded.expand(n_x, n_u, n_samples, self.dim_epsilon)
        
        # Merge dimensions for batch processing
        x_flat = x_expanded.reshape(-1, self.dim_x)
        u_flat = u_expanded.reshape(-1, self.dim_u)
        eps_flat = eps_expanded.reshape(-1, self.dim_epsilon)
        
        # Batch compute next states
        next_states = self.transition_func(x_flat, u_flat, eps_flat)
        
        # Interpolate to compute value function
        next_values = self.interpolate_values(next_states)
        
        # Reshape and compute expectation
        next_values = next_values.reshape(n_x, n_u, n_samples)
        expected_values = next_values.mean(dim=2)  # Average along sample dimension
        
        return expected_values
    
    def compute_penalty_rewards(self) -> torch.Tensor:
        """
        Compute penalty-augmented reward values for all grid points (GPU-optimized version)
        
        Returns:
            Reward matrix, shape (x_grid_size, u_grid_size)
        """
        print(f"Computing penalty-augmented reward values...")
        print(f"  State Points: {self.x_grid_size}, Control Points: {self.u_grid_size}")
        
        # Optimization: batch computation
        # Expand dimensions for all combinations
        x_expanded = self.x_grid.unsqueeze(1).expand(-1, self.u_grid_size, -1)
        u_expanded = self.u_grid.unsqueeze(0).expand(self.x_grid_size, -1, -1)
        
        # Merge for batch processing
        x_flat = x_expanded.reshape(-1, self.dim_x)
        u_flat = u_expanded.reshape(-1, self.dim_u)
        
        # Batch compute rewards
        batch_rewards = self.penalty_reward_func(x_flat, u_flat)
        
        # Reshape to matrix
        reward_matrix = batch_rewards.reshape(self.x_grid_size, self.u_grid_size)
        
        print(f"Reward Matrix Computed, Shape: {reward_matrix.shape}")
        return reward_matrix
    
    def value_iteration(
        self,
        tol: float = 1e-6,
        max_iter: int = 1000,
        verbose: bool = True,
        plot_progress: bool = False
    ):
        """
        Execute value function iteration algorithm (GPU-optimized version)
        
        Parameters:
            tol: Convergence tolerance
            max_iter: Maximum iterations
            verbose: Whether to print iteration information
            plot_progress: Whether to plot iteration progress
            
        Returns:
            policy_func: Optimal policy function
            value_func_interp: Optimal value function interpolator
            convergence_info: Convergence information dictionary
        """
        if self.x_grid is None or self.u_grid is None:
            raise ValueError("Please set grids first (set_grids)")
        
        if verbose:
            print("\n" + "="*60)
            print("Starting Value Function Iteration (GPU Version)")
            print("="*60)
            print(f"State Grid Size: {self.x_grid_size}")
            print(f"Control Grid Size: {self.u_grid_size}")
            print(f"Perturbation Samples: {self.n_samples}")
            print(f"Convergence Tolerance: {tol}, Maximum Iterations: {max_iter}")
        
        # 1. Compute penalty-augmented reward matrix
        if verbose:
            print("\n1. Computing penalty-augmented reward matrix...")
        reward_matrix = self.compute_penalty_rewards()  # shape (n_x, n_u)
        
        # 2. Initialize value function
        if self.value_func is None:
            self.value_func = torch.zeros(self.x_grid_shape, device=self.device)
        
        # Store iteration process
        value_history = []
        max_change_history = []
        iteration_times = []
        
        # 3. Value function iteration main loop
        for iter_num in range(max_iter):
            iter_start_time = time.time()
            
            if verbose and (iter_num % 10 == 0 or iter_num < 5):
                print(f"\nIteration {iter_num}:")
            
            # 3a. Compute expected value of next period states
            next_state_values = self.compute_next_state_values(
                self.x_grid, self.u_grid
            )  # shape (n_x, n_u)
            
            # 3b. Compute Q-values: current reward + discounted future expected value
            Q_values = reward_matrix + self.beta * next_state_values
            
            # 3c. For each state, find control action that maximizes Q-value
            # New value function: maximum Q-value for each state
            new_value_func, best_u_indices = torch.max(Q_values, dim=1)  # shape (n_x,)
            
            # Reshape value function to grid shape
            new_value_func_grid = new_value_func.reshape(self.x_grid_shape)
            
            # Compute value function change
            value_change = torch.abs(new_value_func_grid - self.value_func)
            max_change = torch.max(value_change).item()
            
            # Record iteration information
            if iter_num % 10 == 0:  # Reduce storage frequency to save memory
                value_history.append(new_value_func_grid.cpu().numpy().copy())
            max_change_history.append(max_change)
            iteration_times.append(time.time() - iter_start_time)
            
            # Update value function
            self.value_func = new_value_func_grid
            
            if verbose and (iter_num % 10 == 0 or iter_num < 5):
                print(f"  Value Function Range: [{torch.min(new_value_func_grid):.6f}, "
                      f"{torch.max(new_value_func_grid):.6f}]")
                print(f"  Maximum Change: {max_change:.6e}")
                print(f"  Iteration Time: {iteration_times[-1]:.3f} seconds")
            
            # Check convergence
            if max_change < tol:
                if verbose:
                    print(f"\nConverged at iteration {iter_num}, maximum change: {max_change:.6e} < {tol}")
                break
        
        # If maximum iterations reached without convergence
        if iter_num == max_iter - 1 and max_change >= tol:
            if verbose:
                print(f"\nWarning: Reached maximum iterations {max_iter} without convergence")
                print(f"Final Maximum Change: {max_change:.6e} > {tol}")
        
        # 4. Construct optimal policy grid
        # Convert best control indices to actual control values
        best_u_values = self.u_grid[best_u_indices]  # shape (n_x, dim_u)
        
        # Reshape to grid shape
        policy_grid = best_u_values.reshape(self.x_grid_shape + (self.dim_u,))
        
        # 5. Create policy function
        def policy_func(x):
            """
            Optimal policy function
            
            Parameters:
                x: State points
                
            Returns:
                Optimal control actions
            """
            x_arr = np.asarray(x)
            was_1d = x_arr.ndim == 1
            if was_1d:
                x_arr = x_arr.reshape(1, -1)
            
            n = x_arr.shape[0]
            result = np.zeros((n, self.dim_u))
            
            # Convert input to torch tensor
            x_tensor = torch.from_numpy(x_arr).float().to(self.device)
            
            # Find nearest grid point (simplified version)
            for i in range(n):
                # Calculate distance to all grid points
                distances = torch.norm(self.x_grid - x_tensor[i], dim=1)
                nearest_idx = torch.argmin(distances).item()
                
                # Get corresponding control value
                result[i] = best_u_values[nearest_idx].cpu().numpy()
            
            if was_1d:
                return result[0]
            return result
        
        # 6. Record convergence information
        convergence_info = {
            'iterations': iter_num + 1,
            'max_change_history': np.array(max_change_history),
            'value_history': np.array(value_history) if value_history else [],
            'iteration_times': np.array(iteration_times),
            'converged': max_change < tol,
            'final_max_change': max_change,
            'final_policy_grid': policy_grid.cpu().numpy()
        }
        
        if verbose:
            print("\n" + "="*60)
            print("Value Function Iteration Completed")
            print("="*60)
            print(f"Total Iterations: {convergence_info['iterations']}")
            print(f"Converged: {convergence_info['converged']}")
            print(f"Final Maximum Change: {convergence_info['final_max_change']:.6e}")
            print(f"Total Computation Time: {np.sum(convergence_info['iteration_times']):.3f} seconds")
            print(f"Value Function Range: [{torch.min(self.value_func):.6f}, "
                  f"{torch.max(self.value_func):.6f}]")
        
        # Record tolerance
        convergence_info['tol'] = tol
        
        # 7. Plot iteration process (if enabled)
        if plot_progress:
            self._plot_value_iteration_convergence(convergence_info, tol=tol)
        
        # 8. Store policy function
        self.policy_func = policy_func
        self.is_solved = True
        self.solution_method = 'value_iteration'
        
        return policy_func, None, convergence_info
 
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
        Execute policy iteration algorithm (GPU-optimized version)
        
        Parameters:
            tol: Convergence tolerance
            max_policy_iter: Maximum policy iterations
            max_value_iter: Maximum value iterations per policy evaluation
            verbose: Whether to print iteration information
            plot_progress: Whether to plot iteration progress
            
        Returns:
            policy_func: Optimal policy function
            value_func_interp: Optimal value function interpolator
            convergence_info: Convergence information dictionary
        """
        if self.x_grid is None or self.u_grid is None:
            raise ValueError("Please set grids first (set_grids)")
        
        if verbose:
            print("\n" + "="*60)
            print("Starting Policy Iteration (GPU Version)")
            print("="*60)
            print(f"State Grid Size: {self.x_grid_size}")
            print(f"Control Grid Size: {self.u_grid_size}")
            print(f"Perturbation Samples: {self.n_samples}")
            print(f"Policy Iteration Tolerance: {tol}, Max Policy Iterations: {max_policy_iter}")
            print(f"Max Iterations per Policy Evaluation: {max_value_iter}")
        
        # 1. Compute penalty-augmented reward matrix
        if verbose:
            print("\n1. Computing penalty-augmented reward matrix...")
        reward_matrix = self.compute_penalty_rewards()  # shape (n_x, n_u)
        
        # 2. Initialize policy (choose control with max reward as initial policy)
        if verbose:
            print("\n2. Initializing policy function...")
        
        # Initialization: choose control with maximum reward for each state
        initial_policy_indices = torch.argmax(reward_matrix, dim=1)  # shape (n_x,)
        
        # Convert policy indices to policy value grid
        policy_indices = initial_policy_indices.clone()
        
        # For storing iteration process
        policy_change_history = []
        value_change_history = []
        policy_iteration_times = []
        value_evaluation_times = []
        
        # 3. Main policy iteration loop
        for policy_iter in range(max_policy_iter):
            policy_iter_start_time = time.time()
            
            if verbose:
                print(f"\nPolicy Iteration {policy_iter}:")
            
            # 3.1 Policy evaluation: compute value function under current policy
            if verbose:
                print(f"  Policy evaluation...")
            
            # Get rewards under current policy
            policy_rewards = torch.zeros(self.x_grid_size, device=self.device)
            for i in range(self.x_grid_size):
                u_idx = policy_indices[i]
                policy_rewards[i] = reward_matrix[i, u_idx]
            
            policy_rewards = policy_rewards.reshape(self.x_grid_shape)
            
            # Initialize value function
            value_func = torch.zeros(self.x_grid_shape, device=self.device)
            
            # Policy evaluation iterations (value iteration)
            value_eval_start_time = time.time()
            for value_iter in range(max_value_iter):
                # Update value function for interpolation
                self.value_func = value_func
                
                # Calculate expected next state values under current policy
                # Note: We only compute control points corresponding to current policy
                next_state_values = torch.zeros(self.x_grid_size, device=self.device)
                
                # Batch computation for efficiency: group states with same control
                unique_controls = torch.unique(policy_indices)
                
                for u_idx in unique_controls:
                    # Find states using this control
                    state_mask = (policy_indices == u_idx)
                    if not torch.any(state_mask):
                        continue
                    
                    # Get states using this control
                    state_indices = torch.where(state_mask)[0]
                    x_batch = self.x_grid[state_indices]
                    u_point = self.u_grid[u_idx:u_idx+1]  # Keep 2D
                    
                    # Calculate expected next state value for this batch
                    expected_values = self.compute_next_state_values(x_batch, u_point)
                    next_state_values[state_indices] = expected_values.squeeze(1)
                
                next_state_values = next_state_values.reshape(self.x_grid_shape)
                
                # Calculate new value function
                new_value_func = policy_rewards + self.beta * next_state_values
                
                # Calculate value function change
                value_change = torch.abs(new_value_func - value_func)
                max_value_change = torch.max(value_change).item()
                
                # Update value function
                value_func = new_value_func.clone()
                
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
                print(f"  Value function range: [{torch.min(value_func):.6f}, {torch.max(value_func):.6f}]")
            
            # 3.2 Policy improvement: improve policy based on current value function
            if verbose:
                print(f"  Policy improvement...")
            
            # Update value function for interpolation
            self.value_func = value_func
            
            # Calculate Q-value matrix
            next_state_values_all = self.compute_next_state_values(
                self.x_grid, self.u_grid
            )  # Shape (n_x, n_u)
            
            Q_values = reward_matrix + self.beta * next_state_values_all
            
            # Find optimal control for each state
            new_policy_indices = torch.argmax(Q_values, dim=1)  # Shape (n_x,)
            
            # Calculate policy change
            policy_change = torch.sum(new_policy_indices != policy_indices).item()
            policy_change_rate = policy_change / self.x_grid_size
            
            # Record history
            policy_change_history.append(policy_change)
            
            # Update policy
            policy_indices = new_policy_indices.clone()
            
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
        
        # 5. Construct optimal policy grid
        # Convert best control indices to actual control values
        best_u_values = self.u_grid[policy_indices]  # shape (n_x, dim_u)
        
        # Reshape to grid shape
        policy_grid = best_u_values.reshape(self.x_grid_shape + (self.dim_u,))
        
        # 6. Create policy function
        def policy_func(x):
            """
            Optimal policy function
            
            Parameters:
                x: State points
                
            Returns:
                Optimal control actions
            """
            x_arr = np.asarray(x)
            was_1d = x_arr.ndim == 1
            if was_1d:
                x_arr = x_arr.reshape(1, -1)
            
            n = x_arr.shape[0]
            result = np.zeros((n, self.dim_u))
            
            # Convert input to torch tensor
            x_tensor = torch.from_numpy(x_arr).float().to(self.device)
            
            # Find nearest grid point for each input point
            for i in range(n):
                # Calculate distance to all grid points
                distances = torch.norm(self.x_grid - x_tensor[i], dim=1)
                nearest_idx = torch.argmin(distances).item()
                
                # Get corresponding control value
                result[i] = best_u_values[nearest_idx].cpu().numpy()
            
            if was_1d:
                return result[0]
            return result
        
        # 7. Record convergence information
        convergence_info = {
            'policy_iterations': policy_iter + 1,
            'policy_change_history': np.array(policy_change_history),
            'value_change_history': np.array(value_change_history),
            'policy_iteration_times': np.array(policy_iteration_times),
            'value_evaluation_times': np.array(value_evaluation_times),
            'converged': policy_change == 0,
            'final_policy_change': policy_change,
            'final_value_func': final_value_func.cpu().numpy(),
            'final_policy_grid': policy_grid.cpu().numpy()
        }
        
        if verbose:
            print("\n" + "="*60)
            print("Policy Iteration Completed (GPU Version)")
            print("="*60)
            print(f"Total Policy Iterations: {convergence_info['policy_iterations']}")
            print(f"Converged: {convergence_info['converged']}")
            print(f"Final Policy Change States: {convergence_info['final_policy_change']}")
            print(f"Total Computation Time: {np.sum(convergence_info['policy_iteration_times']):.3f}s")
            print(f"Value Function Range: [{torch.min(final_value_func):.6f}, "
                  f"{torch.max(final_value_func):.6f}]")
        
        # Record tolerance
        convergence_info['tol'] = tol
        
        # 8. Plot iteration progress (if enabled)
        if plot_progress:
            self._plot_policy_iteration_convergence(convergence_info)
        
        # 9. Store policy function
        self.policy_func = policy_func
        self.is_solved = True
        self.solution_method = 'policy_iteration'
        
        return policy_func, None, convergence_info
    
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
        plt.title('Policy Iteration Convergence Process (GPU)\n(Superlinear Convergence)', fontsize=14, fontweight='bold')
        
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

    
    def solve(
        self,
        method: str = 'value_iteration',
        tol: float = 1e-6,
        max_iter: int = 1000,
        max_value_iter: int = 100,  # For policy_iteration
        verbose: bool = True,
        **kwargs
    ):
        """
        Solve MDP problem (GPU version)
        
        Parameters:
            method: Solution method ('value_iteration' or 'policy_iteration')
            tol: Convergence tolerance
            max_iter: Maximum iterations
            max_value_iter: Maximum value iterations per policy evaluation (only for policy_iteration)
            verbose: Whether to print information
            **kwargs: Other parameters
            
        Returns:
            self: Solver instance
        """
        # Check necessary settings
        if self.x_grid is None or self.u_grid is None:
            raise ValueError("Please set grids first (set_grids)")
        
        if verbose:
            print("\n" + "="*60)
            print(f"Starting MDP Problem Solution (GPU Version)")
            print("="*60)
            print(f"Solution Method: {method}")
            print(f"Device: {self.device}")
            print(f"State Grid Size: {self.x_grid_size}")
            print(f"Control Grid Size: {self.u_grid_size}")
        
        start_time = time.time()
        
        # Call corresponding solution method
        if method.lower() == 'value_iteration':
            policy_func, value_func_interp, convergence_info = self.value_iteration(
                tol=tol,
                max_iter=max_iter,
                verbose=verbose,
                plot_progress=False
            )
        elif method.lower() == 'policy_iteration':
            max_policy_iter = max_iter  # Rename to match parameter name
            policy_func, value_func_interp, convergence_info = self.policy_iteration(
                tol=tol,
                max_policy_iter=max_policy_iter,
                max_value_iter=max_value_iter,
                verbose=verbose,
                plot_progress=False
            )
        else:
            raise ValueError(f"Unsupported solution method: {method}. Options: 'value_iteration', 'policy_iteration'")
        
        # Store convergence information
        self.convergence_info = convergence_info
        self.solution_time = time.time() - start_time
        self.solution_method = method
        self.is_solved = True
        
        if verbose:
            print(f"\nSolution Completed, Total Time: {self.solution_time:.3f} seconds")
            if hasattr(self, 'convergence_info'):
                if method.lower() == 'value_iteration':
                    print(f"Iterations: {self.convergence_info['iterations']}")
                    print(f"Converged: {self.convergence_info['converged']}")
                    print(f"Final maximum change: {self.convergence_info['final_max_change']:.6e}")
                elif method.lower() == 'policy_iteration':
                    print(f"Policy iterations: {self.convergence_info['policy_iterations']}")
                    print(f"Converged: {self.convergence_info['converged']}")
                    print(f"Final policy change states: {self.convergence_info['final_policy_change']}")
        
        return self
    
    def act(self, x: Union[np.ndarray, List]) -> np.ndarray:
        """
        Given current state, provide optimal control based on optimal policy function
        
        Parameters:
            x: State points
            
        Returns:
            Optimal control actions
            
        Note:
            Need to run solve first to obtain policy function
        """
        if not hasattr(self, 'policy_func') or self.policy_func is None:
            raise ValueError("Please run solve first to obtain policy function")
        
        return self.policy_func(x)
    
    def predict(
        self, 
        x0: Union[np.ndarray, List], 
        T: int, 
        epsilon_mean: Optional[Union[np.ndarray, List]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Given initial state, provide deterministic optimal variable paths
        
        Parameters:
            x0: Initial state
            T: Prediction time steps
            epsilon_mean: Perturbation mean
            
        Returns:
            Dictionary containing paths
        """
        if not hasattr(self, 'policy_func') or self.policy_func is None:
            raise ValueError("Please run solve first to obtain policy function")
        
        # Convert to numpy array
        x0_arr = np.asarray(x0)
        if x0_arr.ndim == 1:
            x0_arr = x0_arr.reshape(1, -1)
        
        # Get perturbation mean
        if epsilon_mean is None:
            if self.ppf_table is not None:
                epsilon_mean = torch.mean(self.ppf_table, dim=0).cpu().numpy()
            else:
                epsilon_mean = np.zeros(self.dim_epsilon)
        else:
            epsilon_mean = np.asarray(epsilon_mean)
        
        # Initialize storage arrays
        states = np.zeros((T, self.dim_x))
        controls = np.zeros((T-1, self.dim_u))
        rewards = np.zeros(T-1)
        
        # Set initial state
        states[0] = x0_arr[0]
        
        # Generate deterministic path
        for t in range(T-1):
            # Current state
            current_state = states[t:t+1]
            
            # Calculate optimal control
            optimal_control = self.policy_func(current_state)
            if optimal_control.ndim == 1:
                optimal_control = optimal_control.reshape(1, -1)
            
            # Calculate reward
            current_reward = self.penalty_reward_func_np(current_state, optimal_control)
            
            # Calculate next state
            next_state = self.transition_func_np(current_state, optimal_control, epsilon_mean)
            
            # Store results
            controls[t] = optimal_control[0]
            rewards[t] = current_reward[0] if np.isscalar(current_reward[0]) else current_reward[0, 0]
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
            raise ValueError("Please run solve first to obtain policy function")
        
        # Set random seed
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            if self.device.type == 'cuda':
                torch.cuda.manual_seed(seed)
        
        # Convert initial state to appropriate format
        x0_arr = np.asarray(x0)
        if x0_arr.ndim == 1:
            x0_arr = x0_arr.reshape(1, -1)
        
        # Initialize storage arrays (numpy for final output)
        states = np.zeros((n_simulations, T, self.dim_x))
        controls = np.zeros((n_simulations, T-1, self.dim_u))
        rewards = np.zeros((n_simulations, T-1))
        epsilon_samples = np.zeros((n_simulations, T-1, self.dim_epsilon))
        
        # Set initial state for all simulations
        states[:, 0, :] = x0_arr[0]
        
        # Generate perturbations
        if epsilon_generator is None:
            # Use built-in sampling method (GPU-optimized batch sampling)
            # Generate all perturbations at once for efficiency
            all_epsilons = torch.zeros((n_simulations, T-1, self.dim_epsilon), device=self.device)
            for sim in range(n_simulations):
                for t in range(T-1):
                    eps_sample = self.sample_perturbations(1)
                    all_epsilons[sim, t] = eps_sample[0]
            epsilon_all_numpy = all_epsilons.cpu().numpy()
        else:
            # Use user-provided generator
            epsilon_all_numpy = []
            for sim in range(n_simulations):
                epsilon_path = epsilon_generator(T-1, self.dim_epsilon)
                if epsilon_path.shape != (T-1, self.dim_epsilon):
                    raise ValueError(f"epsilon_generator should return array with shape ({T-1}, {self.dim_epsilon})")
                epsilon_all_numpy.append(epsilon_path)
            epsilon_all_numpy = np.array(epsilon_all_numpy)
        
        # GPU-optimized simulation (using PyTorch for computation)
        # Convert initial states to PyTorch tensor
        current_states_torch = torch.from_numpy(states[:, 0:1, :]).float().to(self.device)
        
        # Main simulation loop (time steps)
        for t in range(T-1):
            # Get current epsilon for all simulations
            current_epsilon_numpy = epsilon_all_numpy[:, t:t+1, :]
            current_epsilon_torch = torch.from_numpy(current_epsilon_numpy).float().to(self.device)
            
            # Calculate optimal control for all simulations (using numpy policy function)
            # Convert states to numpy for policy function
            current_states_numpy = current_states_torch.cpu().numpy()
            
            # Get optimal control from policy function
            optimal_control_numpy = self.policy_func(current_states_numpy)
            if optimal_control_numpy.ndim == 2 and optimal_control_numpy.shape[0] == n_simulations:
                optimal_control_torch = torch.from_numpy(optimal_control_numpy).float().to(self.device)
            else:
                # Handle broadcasting if needed
                optimal_control_torch = torch.from_numpy(optimal_control_numpy.reshape(n_simulations, -1)).float().to(self.device)
            
            # Calculate reward (using PyTorch for GPU acceleration)
            # Ensure tensors have correct shape for batch processing
            current_states_reshaped = current_states_torch.reshape(n_simulations, self.dim_x)
            optimal_control_reshaped = optimal_control_torch.reshape(n_simulations, self.dim_u)
            
            # Calculate reward using PyTorch function
            current_reward_torch = self.penalty_reward_func(current_states_reshaped, optimal_control_reshaped)
            
            # Calculate next state (using PyTorch for GPU acceleration)
            current_epsilon_reshaped = current_epsilon_torch.reshape(n_simulations, self.dim_epsilon)
            next_state_torch = self.transition_func(current_states_reshaped, 
                                                   optimal_control_reshaped, 
                                                   current_epsilon_reshaped)
            
            # Store results (convert to numpy for output)
            controls[:, t] = optimal_control_reshaped.cpu().numpy()
            rewards[:, t] = current_reward_torch.cpu().numpy()
            epsilon_samples[:, t] = current_epsilon_reshaped.cpu().numpy()
            
            # Update states for next iteration
            if t < T-2:
                # Reshape back to (n_simulations, 1, dim_x) for consistency
                current_states_torch = next_state_torch.unsqueeze(1)
                states[:, t+1] = next_state_torch.cpu().numpy()
            else:
                # Last iteration
                states[:, t+1] = next_state_torch.cpu().numpy()
        
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
            torch.manual_seed(seed)
            if self.device.type == 'cuda':
                torch.cuda.manual_seed(seed)
        
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
        ax.set_title(f'State Variable Paths ({n_samples_plot} Samples)')
        ax.grid(True, alpha=0.3)
        
        # 2. Control variable paths (each dimension)
        ax = axes[0, 1]
        for sim_idx in indices:
            for d in range(self.dim_u):
                ax.plot(time_steps_control, controls[sim_idx, :, d], alpha=0.5, linewidth=1)
        
        ax.set_xlabel('Time step')
        ax.set_ylabel('Control value')
        ax.set_title(f'Control Variable Paths ({n_samples_plot} Samples)')
        ax.grid(True, alpha=0.3)
        
        # 3. Reward paths
        ax = axes[1, 0]
        for sim_idx in indices:
            ax.plot(time_steps_control, rewards[sim_idx, :], alpha=0.5, linewidth=1)
        
        ax.set_xlabel('Time step')
        ax.set_ylabel('Reward')
        ax.set_title(f'Reward Paths ({n_samples_plot} Samples)')
        ax.grid(True, alpha=0.3)
        
        # 4. Cumulative reward distribution
        ax = axes[1, 1]
        cumulative_rewards = np.sum(rewards, axis=1)
        ax.hist(cumulative_rewards, bins=30, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Cumulative Reward')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Cumulative Reward Distribution (n={n_simulations})')
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
        ax.set_title('State Variable Statistics (Mean ± Std)')
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
        ax.set_title('Control Variable Statistics (Mean ± Std)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print statistical summary
        print("Simulation Results Statistical Summary:")
        print(f"  Number of simulations: {n_simulations}")
        print(f"  Time steps: {T}")
        print(f"  Average cumulative reward: {np.mean(cumulative_rewards):.4f} ± {np.std(cumulative_rewards):.4f}")
        print(f"  Minimum cumulative reward: {np.min(cumulative_rewards):.4f}")
        print(f"  Maximum cumulative reward: {np.max(cumulative_rewards):.4f}")
        
        return fig

    
    def penalty_reward_func_np(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Numpy version of penalty reward function"""
        # Ensure at least 2D arrays
        x_2d = np.atleast_2d(x)
        u_2d = np.atleast_2d(u)
        
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
        original_reward = self.reward_func_np(x_2d, u_2d)
        
        if self.constraint_func_np is None:
            return original_reward
        
        # Calculate constraint values
        constraint_values = self.constraint_func_np(x_2d, u_2d)
        
        # Ensure constraint values are 2D arrays
        if constraint_values.ndim == 1:
            constraint_values = constraint_values.reshape(-1, 1)
        
        # Calculate max(H(x,u), 0)
        h_pos = np.maximum(constraint_values, 0.0)
        
        # Calculate L2 norm squared
        if h_pos.ndim == 2:
            norm_sq = np.sum(h_pos ** 2, axis=1)
        else:
            norm_sq = h_pos ** 2
        
        # Calculate penalty term
        penalty = np.exp(self.gamma_pen * norm_sq) - 1.
        
        # Reward with penalty
        result = original_reward - self.lambda_pen * penalty
        
        return result
    
    def summary(self):
        """Print solver summary information"""
        print("\n" + "="*60)
        print("MDP GPU Solver Summary")
        print("="*60)
        print(f"Device: {self.device}")
        print(f"Dimensions:")
        print(f"  State: {self.dim_x}, Control: {self.dim_u}, Perturbation: {self.dim_epsilon}")
        print(f"Parameters:")
        print(f"  Discount Factor: {self.beta}")
        print(f"  Penalty Parameters: λ={self.lambda_pen}, γ={self.gamma_pen}")
        print(f"  Perturbation Samples: {self.n_samples}")
        print(f"  Interpolation Method: {self.interpolation_method}")
        
        if self.x_grid is not None:
            print(f"\nState Grid:")
            print(f"  Shape: {self.x_grid_shape}")
            print(f"  Total Points: {self.x_grid_size}")
            print(f"  Points per Dimension: {self.x_grid_nums}")
        
        if self.u_grid is not None:
            print(f"\nControl Grid:")
            print(f"  Shape: {self.u_grid_shape}")
            print(f"  Total Points: {self.u_grid_size}")
            print(f"  Points per Dimension: {self.u_grid_nums}")
        
        if self.is_solved:
            print(f"\nSolution Status:")
            print(f"  Solved: Yes")
            print(f"  Solution Method: {self.solution_method}")
            print(f"  Solution Time: {self.solution_time:.3f} seconds")
            if self.convergence_info:
                print(f"  Converged: {self.convergence_info.get('converged', 'N/A')}")
                print(f"  Iterations: {self.convergence_info.get('iterations', 'N/A')}")
        else:
            print(f"\nSolution Status: Not solved yet")
        
        print("="*60)


# Performance comparison test
def compare_performance():
    """Compare CPU and GPU version performance"""
    print("Performance Comparison Test")
    print("="*60)
    
    # Set up test problem
    dim_x = 2
    dim_u = 2
    dim_epsilon = 2
    n_samples = 100
    
    # Simple linear quadratic problem
    def transition(x, u, epsilon):
        # Ensure inputs are numpy arrays
        x_np = np.asarray(x)
        u_np = np.asarray(u)
        epsilon_np = np.asarray(epsilon)
        
        # Simple linear transition
        result = x_np + u_np + epsilon_np * 0.1
        return result
    
    def reward(x, u):
        # Ensure inputs are numpy arrays
        x_np = np.asarray(x)
        u_np = np.asarray(u)
        
        # Simple quadratic reward
        result = -(x_np**2).sum(axis=1) - (u_np**2).sum(axis=1)
        return result
    
    # Test different grid sizes
    grid_sizes = [
        (10, 5),    # Small grid
        (20, 10),   # Medium grid
        (40, 20),   # Large grid
    ]
    
    results = []
    
    for x_grid_num, u_grid_num in grid_sizes:
        print(f"\nTesting Grid Size: State {x_grid_num}x{x_grid_num}, Control {u_grid_num}x{u_grid_num}")
        
        # Create CPU solver
        cpu_solver = MDPSolverGPU(
            dim_x=dim_x, dim_u=dim_u, dim_epsilon=dim_epsilon,
            n_samples=n_samples, transition_func=transition,
            reward_func=reward, device='cpu'
        )
        
        cpu_solver.set_grids(
            x_limits=[[-2, 2], [-2, 2]],
            u_limits=[[-1, 1], [-1, 1]],
            x_grid_nums=[x_grid_num, x_grid_num],
            u_grid_nums=[u_grid_num, u_grid_num]
        )
        
        # CPU solution time
        cpu_start = time.time()
        #method = 'value_iteration'
        method = 'policy_iteration'
        cpu_solver.solve(method=method, tol=1e-4, max_iter=20, verbose=False)
        cpu_time = time.time() - cpu_start
        
        # Create GPU solver (if GPU available)
        if torch.cuda.is_available():
            gpu_solver = MDPSolverGPU(
                dim_x=dim_x, dim_u=dim_u, dim_epsilon=dim_epsilon,
                n_samples=n_samples, transition_func=transition,
                reward_func=reward, device='cuda'
            )
            
            gpu_solver.set_grids(
                x_limits=[[-2, 2], [-2, 2]],
                u_limits=[[-1, 1], [-1, 1]],
                x_grid_nums=[x_grid_num, x_grid_num],
                u_grid_nums=[u_grid_num, u_grid_num]
            )
            
            # GPU solution time
            gpu_start = time.time()
            gpu_solver.solve(method=method, tol=1e-4, max_iter=20, verbose=False)
            gpu_time = time.time() - gpu_start
            
            # Speedup
            speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        else:
            gpu_time = None
            speedup = None
        
        results.append({
            'grid_size': f"{x_grid_num}x{x_grid_num}/{u_grid_num}x{u_grid_num}",
            'cpu_time': cpu_time,
            'gpu_time': gpu_time,
            'speedup': speedup
        })
        
        print(f"  CPU Time: {cpu_time:.2f} seconds")
        if gpu_time:
            print(f"  GPU Time: {gpu_time:.2f} seconds")
            print(f"  Speedup: {speedup:.2f}x")
    
    # Print results summary
    print("\n" + "="*60)
    print("Performance Comparison Summary")
    print("="*60)
    print(f"{'Grid Size':<20} {'CPU Time(s)':<12} {'GPU Time(s)':<12} {'Speedup':<10}")
    print("-" * 60)
    
    for result in results:
        gpu_time_str = f"{result['gpu_time']:.2f}" if result['gpu_time'] else "N/A"
        speedup_str = f"{result['speedup']:.2f}x" if result['speedup'] else "N/A"
        print(f"{result['grid_size']:<20} {result['cpu_time']:<12.2f} {gpu_time_str:<12} {speedup_str:<10}")
    
    return results


if __name__ == "__main__":
    # If run directly, perform performance test
    compare_performance()

"""
MDP Solver Test Script (GPU/CPU Compatible)

This script demonstrates how to use the MDPSolver library to solve LQ (Linear Quadratic) problems,
including value iteration, policy iteration, prediction and simulation functions.

Usage:
    python test_mdp_solver.py
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Fix OpenMP conflict

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import time

# Import solver versions
print("Loading solver versions...")

# Import CPU solver
try:
    from mdp_solver_cpu import MDPSolverCPU
    CPU_AVAILABLE = True
    print("✓ CPU solver loaded successfully")
except ImportError as e:
    CPU_AVAILABLE = False
    print(f"✗ CPU solver not available: {e}")

# Import GPU solver
try:
    import torch
    if hasattr(torch, 'cuda') and torch.cuda.is_available():
        print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("✓ GPU solver will use CPU (CUDA not available)")
    
    from mdp_solver_gpu import MDPSolverGPU
    GPU_AVAILABLE = True
    print("✓ GPU solver loaded successfully")
except ImportError as e:
    GPU_AVAILABLE = False
    print(f"✗ GPU solver not available: {e}")
except Exception as e:
    GPU_AVAILABLE = False
    print(f"✗ Error loading GPU solver: {e}")


def create_lq_reward_function(
    Q: np.ndarray, 
    R: np.ndarray, 
    N: np.ndarray
) -> callable:
    """Create reward function for LQ problem"""
    assert Q.shape == (2, 2), f"Q should be (2,2) matrix, actual shape {Q.shape}"
    assert R.shape == (2, 2), f"R should be (2,2) matrix, actual shape {R.shape}"
    assert N.shape == (2, 2), f"N should be (2,2) matrix, actual shape {N.shape}"
    
    Q = (Q + Q.T) / 2
    R = (R + R.T) / 2
    
    def reward_func(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        x_2d = np.atleast_2d(x)
        u_2d = np.atleast_2d(u)
        
        n_x = x_2d.shape[0]
        n_u = u_2d.shape[0]
        
        if n_x > 1 and n_u == 1:
            u_2d = np.broadcast_to(u_2d, (n_x, 2))
        
        xQx = np.einsum('ni,ij,nj->n', x_2d, Q, x_2d)
        uRu = np.einsum('ni,ij,nj->n', u_2d, R, u_2d)
        xNu = 2 * np.einsum('ni,ij,nj->n', x_2d, N, u_2d)
        
        result = -(xQx + uRu + xNu)
        
        if x.ndim == 1 and u.ndim == 1:
            return result[0]
        return result
    
    return reward_func


def create_lq_transition_function(
    A: np.ndarray, 
    B: np.ndarray, 
    C: np.ndarray
) -> callable:
    """Create state transition function for LQ problem"""
    assert A.shape == (2, 2), f"A should be (2,2) matrix, actual shape {A.shape}"
    assert B.shape == (2, 2), f"B should be (2,2) matrix, actual shape {B.shape}"
    assert C.shape == (2, 2), f"C should be (2,2) matrix, actual shape {C.shape}"
    
    def transition_func(x: np.ndarray, u: np.ndarray, epsilon: np.ndarray) -> np.ndarray:
        x_2d = np.atleast_2d(x)
        u_2d = np.atleast_2d(u)
        epsilon_2d = np.atleast_2d(epsilon)
        
        n_x = x_2d.shape[0]
        n_u = u_2d.shape[0]
        n_eps = epsilon_2d.shape[0]
        
        n = max(n_x, n_u, n_eps)
        
        if n_x == 1 and n > 1:
            x_2d = np.tile(x_2d, (n, 1))
        elif n_x != n:
            raise ValueError(f"x dimension mismatch: {n_x} != {n}")
        
        if n_u == 1 and n > 1:
            u_2d = np.tile(u_2d, (n, 1))
        elif n_u != n:
            raise ValueError(f"u dimension mismatch: {n_u} != {n}")
        
        if n_eps == 1 and n > 1:
            epsilon_2d = np.tile(epsilon_2d, (n, 1))
        elif n_eps != n:
            raise ValueError(f"epsilon dimension mismatch: {n_eps} != {n}")
        
        result = (x_2d @ A.T) + (u_2d @ B.T) + (epsilon_2d @ C.T)
        
        if x.ndim == 1 and u.ndim == 1 and epsilon.ndim == 1:
            return result[0]
        return result
    
    return transition_func


def create_lq_constraint_function(
    H_mat: np.ndarray, 
    G_mat: np.ndarray, 
    b: np.ndarray
) -> callable:
    """Create constraint function for LQ problem"""
    assert H_mat.shape == (2, 2), f"H should be (2,2) matrix, actual shape {H_mat.shape}"
    assert G_mat.shape == (2, 2), f"G should be (2,2) matrix, actual shape {G_mat.shape}"
    assert b.shape == (2,), f"b should be (2,) vector, actual shape {b.shape}"
    
    def constraint_func(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        x_2d = np.atleast_2d(x)
        u_2d = np.atleast_2d(u)
        
        n_x = x_2d.shape[0]
        n_u = u_2d.shape[0]
        
        if n_x > 1 and n_u == 1:
            u_2d = np.broadcast_to(u_2d, (n_x, 2))
        
        result = (x_2d @ H_mat.T) + (u_2d @ G_mat.T) - b
        
        if x.ndim == 1 and u.ndim == 1:
            return result[0]
        return result
    
    return constraint_func


def setup_lq_problem():
    """Set up LQ problem parameters and functions"""
    print("="*60)
    print("Setting up LQ Problem Parameters")
    print("="*60)
    
    # Define system parameters
    Q = np.array([[2.0, 0.5], [0.5, 1.0]])
    R = np.array([[1.0, 0.2], [0.2, 0.8]])
    N = np.array([[0.3, 0.1], [0.1, 0.4]])
    
    A = np.array([[0.9, 0.1], [0.05, 0.85]])
    B = np.array([[0.8, 0.0], [0.0, 0.7]])
    C = np.array([[0.1, 0.0], [0.0, 0.1]])
    
    H_mat = np.array([[1.0, 0.0], [0.0, 1.0]])
    G_mat = np.array([[0.5, 0.0], [0.0, 0.5]])
    b = np.array([1.0, 1.0])
    
    # Create functions
    print("Creating LQ problem functions...")
    reward_func = create_lq_reward_function(Q, R, N)
    transition_func = create_lq_transition_function(A, B, C)
    constraint_func = create_lq_constraint_function(H_mat, G_mat, b)
    
    print("✓ Reward function created")
    print("✓ State transition function created")
    print("✓ Constraint function created")
    
    return reward_func, transition_func, constraint_func


def select_solver_version():
    """Let user select between GPU and CPU solver versions"""
    print("\n" + "="*60)
    print("Select Solver Version")
    print("="*60)
    
    if not GPU_AVAILABLE:
        print("GPU solver not available. Using CPU solver.")
        return MDPSolverCPU, 'cpu'
    
    print(f"Available options:")
    print(f"  1. CPU Solver (Numpy-based, stable)")
    print(f"  2. GPU Solver (PyTorch-based, accelerated)")
    
    while True:
        choice = input("\nSelect solver version (1-2): ").strip()
        
        if choice == '1':
            print("✓ Selected CPU solver")
            return MDPSolverCPU, 'cpu'
        elif choice == '2':
            print("✓ Selected GPU solver")
            
            # Let user choose GPU device
            print("\nSelect device:")
            print("  1. Auto-select (prefer GPU if available)")
            print("  2. CUDA (GPU)")
            print("  3. CPU")
            
            device_choice = input("Select device (1-3): ").strip()
            
            if device_choice == '1':
                device = 'auto'
            elif device_choice == '2':
                device = 'cuda'
            elif device_choice == '3':
                device = 'cpu'
            else:
                print("Invalid choice, using auto-select")
                device = 'auto'
            
            print(f"  Using device: {device}")
            
            # Return a factory function for GPU solver
            def gpu_solver_factory(*args, **kwargs):
                kwargs['device'] = device
                return MDPSolverGPU(*args, **kwargs)
            
            return gpu_solver_factory, device
        else:
            print("Invalid choice. Please enter 1 or 2.")

def compare_cpu_gpu_performance():
    """
    Compare performance between CPU and GPU solvers
    """
    if not GPU_AVAILABLE or not CPU_AVAILABLE:
        print("Both CPU and GPU solvers must be available for comparison.")
        return
    
    print("\n" + "="*60)
    print("CPU vs GPU Performance Comparison")
    print("="*60)
    
    # Set up LQ problem
    reward_func, transition_func, constraint_func = setup_lq_problem()
    
    # Test different grid sizes
    grid_configs = [
        {"name": "Small grid", "x_grid": [10, 10], "u_grid": [5, 5]},
        {"name": "Medium grid", "x_grid": [20, 20], "u_grid": [10, 10]},
        {"name": "Large grid", "x_grid": [40, 40], "u_grid": [20, 20]},
    ]
    
    results = []
    
    for config in grid_configs:
        print(f"\nTesting {config['name']}...")
        
        # CPU Solver
        print("  CPU Solver:")
        cpu_start = time.time()
        
        cpu_solver = MDPSolverCPU(
            dim_x=2, dim_u=2, dim_epsilon=2, n_samples=100,
            transition_func=transition_func, reward_func=reward_func,
            constraint_func=constraint_func, beta=0.95,
            interpolation_method='linear'
        )
        
        cpu_solver.set_grids(
            x_limits=[[-2, 2], [-2, 2]],
            u_limits=[[-1, 1], [-1, 1]],
            x_grid_nums=config['x_grid'],
            u_grid_nums=config['u_grid']
        )
        
        cpu_solver.set_ppf(stats.norm.ppf, n_quantiles=500)
        cpu_solver.solve(method='value_iteration', tol=1e-4, max_iter=20, verbose=False)
        
        cpu_time = time.time() - cpu_start
        
        # GPU Solver
        print("  GPU Solver:")
        gpu_start = time.time()
        
        gpu_solver = MDPSolverGPU(
            dim_x=2, dim_u=2, dim_epsilon=2, n_samples=100,
            transition_func=transition_func, reward_func=reward_func,
            constraint_func=constraint_func, beta=0.95,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            interpolation_method='linear'
        )
        
        gpu_solver.set_grids(
            x_limits=[[-2, 2], [-2, 2]],
            u_limits=[[-1, 1], [-1, 1]],
            x_grid_nums=config['x_grid'],
            u_grid_nums=config['u_grid']
        )
        
        gpu_solver.set_ppf(stats.norm.ppf, n_quantiles=500)
        gpu_solver.solve(method='value_iteration', tol=1e-4, max_iter=20, verbose=False)
        
        gpu_time = time.time() - gpu_start
        
        # Calculate speedup
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        
        results.append({
            'config': config['name'],
            'grid_size': f"{config['x_grid'][0]}x{config['x_grid'][1]}/{config['u_grid'][0]}x{config['u_grid'][1]}",
            'cpu_time': cpu_time,
            'gpu_time': gpu_time,
            'speedup': speedup
        })
        
        print(f"    CPU time: {cpu_time:.2f}s")
        print(f"    GPU time: {gpu_time:.2f}s")
        print(f"    Speedup: {speedup:.2f}x")
    
    # Display results
    print("\n" + "="*60)
    print("Performance Comparison Results")
    print("="*60)
    print(f"{'Configuration':<15} {'Grid Size':<20} {'CPU Time (s)':<12} {'GPU Time (s)':<12} {'Speedup':<10}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['config']:<15} {result['grid_size']:<20} {result['cpu_time']:<12.2f} {result['gpu_time']:<12.2f} {result['speedup']:<10.2f}")
    
    return results


def test_interpolation_methods(solver_factory):
    """Test different interpolation methods"""
    print("\n" + "="*60)
    print("Testing Different Interpolation Methods")
    print("="*60)
    
    # Set up LQ problem
    reward_func, transition_func, constraint_func = setup_lq_problem()
    
    # GPU solver only supports 'linear' and 'nearest'
    if solver_factory == MDPSolverCPU:
        interpolation_methods = ['linear', 'nearest', 'cubic']
    else:
        interpolation_methods = ['linear', 'nearest']
    
    results = {}
    
    for method in interpolation_methods:
        print(f"\nTesting {method} interpolation...")
        
        # Initialize solver with specific interpolation method
        solver = solver_factory(
            dim_x=2,
            dim_u=2,
            dim_epsilon=2,
            n_samples=50,
            transition_func=transition_func,
            reward_func=reward_func,
            constraint_func=constraint_func,
            beta=0.95,
            interpolation_method=method
        )
        
        # Set grids
        x_limits = [[-1.5, 1.5], [-1.5, 1.5]]
        u_limits = [[-0.8, 0.8], [-0.8, 0.8]]
        x_grid_nums = [15, 15]
        u_grid_nums = [8, 8]
        
        solver.set_grids(x_limits, u_limits, x_grid_nums, u_grid_nums)
        
        # Set PPF
        solver.set_ppf(stats.norm.ppf, n_quantiles=200)
        
        # Solve using value iteration
        start_time = time.time()
        
        policy_func, value_func_interp, convergence_info = solver.value_iteration(
            tol=1e-5,
            max_iter=50,
            verbose=False
        )
        
        elapsed_time = time.time() - start_time
        
        # Test policy at some points
        test_points = [
            [0.0, 0.0],
            [0.5, 0.3],
            [-0.5, -0.3]
        ]
        
        controls = []
        for point in test_points:
            control = solver.act(point)
            controls.append(control)
        
        # Get value function range (compatible with both PyTorch and numpy)
        if hasattr(solver.value_func, 'cpu'):  # PyTorch tensor
            value_min = solver.value_func.min().item()
            value_max = solver.value_func.max().item()
        else:  # numpy array
            value_min = np.min(solver.value_func)
            value_max = np.max(solver.value_func)
        
        results[method] = {
            'time': elapsed_time,
            'iterations': convergence_info['iterations'],
            'converged': convergence_info['converged'],
            'controls': controls,
            'value_range': [value_min, value_max]
        }
        
        print(f"  Solution time: {elapsed_time:.3f} seconds")
        print(f"  Iterations: {convergence_info['iterations']}")
        print(f"  Converged: {convergence_info['converged']}")
        print(f"  Value function range: [{value_min:.4f}, {value_max:.4f}]")
    
    # Compare results
    print("\n" + "="*60)
    print("Interpolation Methods Comparison")
    print("="*60)
    
    print("\nPerformance comparison:")
    print("-" * 60)
    print(f"{'Method':<10} {'Time (s)':<12} {'Iterations':<12} {'Converged':<12}")
    print("-" * 60)
    
    for method in interpolation_methods:
        res = results[method]
        print(f"{method:<10} {res['time']:<12.3f} {res['iterations']:<12} {str(res['converged']):<12}")
    
    # Compare control outputs
    test_point = [0.5, 0.3]
    print(f"\nControl outputs at test point {test_point}:")
    for method in interpolation_methods:
        controls = results[method]['controls']
        idx = test_points.index(test_point)
        print(f"  {method}: {controls[idx]}")
    
    return results


def test_parameter_management(solver_factory):
    """Test parameter management functions"""
    print("\n" + "="*60)
    print("Testing Parameter Management Functions")
    print("="*60)
    
    # Set up LQ problem
    reward_func, transition_func, constraint_func = setup_lq_problem()
    
    # Initialize solver
    print("\nInitializing solver...")
    solver = solver_factory(
        dim_x=2,
        dim_u=2,
        dim_epsilon=2,
        n_samples=100,
        transition_func=transition_func,
        reward_func=reward_func,
        constraint_func=constraint_func,
        beta=0.95,
        lambda_pen=10.,
        gamma_pen=10.,
        interpolation_method='linear'
    )
    
    # Get initial parameters
    print("\n1. Getting initial parameters...")
    initial_params = solver.get_params()
    print("Initial parameters:")
    for key, value in initial_params.items():
        print(f"  {key}: {value}")
    
    # Change parameters
    print("\n2. Changing parameters...")
    solver.set_params(
        beta=0.99,
        lambda_pen=20.,
        interpolation_method='cubic' if solver_factory == MDPSolverCPU else 'nearest'
    )
    
    # Get updated parameters
    print("\n3. Getting updated parameters...")
    updated_params = solver.get_params()
    print("Updated parameters:")
    for key, value in updated_params.items():
        print(f"  {key}: {value}")
    
    # Set grids and solve
    print("\n4. Setting grids and solving...")
    solver.set_grids(
        x_limits=[[-1.5, 1.5], [-1.5, 1.5]],
        u_limits=[[-0.8, 0.8], [-0.8, 0.8]],
        x_grid_nums=[15, 15],
        u_grid_nums=[8, 8]
    )
    solver.set_ppf(stats.norm.ppf, n_quantiles=200)
    
    # Solve with new parameters
    result = solver.solve(
        method='value_iteration',
        tol=1e-5,
        max_iter=50,
        verbose=False
    )
    
    print("\n5. Testing policy with new parameters...")
    test_point = [0.5, 0.3]
    optimal_control = solver.act(test_point)
    print(f"Optimal control at {test_point}: {optimal_control}")
    
    return solver


def test_value_iteration(solver_factory):
    """Test value function iteration"""
    print("\n" + "="*60)
    print("Testing Value Function Iteration")
    print("="*60)
    
    # Set up LQ problem
    reward_func, transition_func, constraint_func = setup_lq_problem()
    
    # Initialize solver
    print("\nInitializing MDP solver...")
    solver = solver_factory(
        dim_x=2,
        dim_u=2,
        dim_epsilon=2,
        n_samples=100,
        transition_func=transition_func,
        reward_func=reward_func,
        constraint_func=constraint_func,
        beta=0.95,
        lambda_pen=10.,
        gamma_pen=10.,
        interpolation_method='linear'
    )
    
    # Set grids
    print("\nSetting grids...")
    x_limits = [[-2, 2], [-2, 2]]
    u_limits = [[-1, 1], [-1, 1]]
    x_grid_nums = [20, 20]
    u_grid_nums = [10, 10]
    
    solver.set_grids(x_limits, u_limits, x_grid_nums, u_grid_nums)
    
    # Set PPF (normal distribution)
    print("\nSetting PPF function...")
    solver.set_ppf(stats.norm.ppf, n_quantiles=1000)
    
    # Run value function iteration
    print("\nRunning value function iteration...")
    start_time = time.time()
    
    policy_func, value_func_interp, convergence_info = solver.value_iteration(
        tol=1e-6,
        max_iter=200,
        verbose=True,
        plot_progress=True
    )
    
    elapsed_time = time.time() - start_time
    print(f"Value function iteration completed, time: {elapsed_time:.2f} seconds")
    
    # Test policy
    print("\nTesting policy function...")
    test_points = [
        [0.0, 0.0],
        [1.0, 0.5],
        [-1.0, -0.5],
        [0.5, -0.5]
    ]
    
    for i, point in enumerate(test_points):
        optimal_control = solver.act(point)
        print(f"State {point}: Optimal control = {optimal_control}")
    
    return solver, convergence_info

def test_solve_method(solver_factory):
    """
    Test solve method (unified solution interface)
    
    Parameters:
        solver_factory: Factory function for creating solver instances
    """
    print("\n" + "="*60)
    print("Testing Solve Method (Unified Solution Interface)")
    print("="*60)
    
    # Set up LQ problem
    reward_func, transition_func, constraint_func = setup_lq_problem()
    
    # Check if GPU solver - GPU only supports value iteration
    is_gpu_solver = solver_factory.__name__ != 'MDPSolverCPU' if hasattr(solver_factory, '__name__') else False
    
    if is_gpu_solver:
        print("\nGPU solver detected - only testing value iteration")
        methods = ['value_iteration']
    else:
        print("\nCPU solver detected - testing both value and policy iteration")
        methods = ['value_iteration', 'policy_iteration']
    
    results = {}
    
    for method in methods:
        print(f"\n--- Testing {method} ---")
        
        # Initialize solver
        solver = solver_factory(
            dim_x=2, dim_u=2, dim_epsilon=2, n_samples=100,
            transition_func=transition_func, reward_func=reward_func,
            constraint_func=constraint_func, beta=0.95,
            interpolation_method='linear'
        )
        
        # Use method chaining
        result = (solver
                  .set_grids(
                      x_limits=[[-2, 2], [-2, 2]],
                      u_limits=[[-1, 1], [-1, 1]],
                      x_grid_nums=[15, 15],
                      u_grid_nums=[8, 8]
                  )
                  .set_ppf(stats.norm.ppf, n_quantiles=500))
        
        # Solve with specified method
        if method == 'value_iteration':
            result = result.solve(
                method='value_iteration',
                tol=1e-5,
                max_iter=100,
                verbose=True
            )
        else:  # policy_iteration (CPU only)
            result = result.solve(
                method='policy_iteration',
                tol=1e-5,
                max_iter=10,
                max_value_iter=30,
                verbose=True
            )
        
        # Test policy
        test_point = [0.5, 0.3]
        optimal_control = result.act(test_point)
        
        # Get value function range
        if hasattr(result.value_func, 'cpu'):  # PyTorch tensor
            value_min = result.value_func.min().item()
            value_max = result.value_func.max().item()
        else:  # numpy array
            value_min = np.min(result.value_func)
            value_max = np.max(result.value_func)
        
        results[method] = {
            'solver': result,
            'control': optimal_control,
            'value_range': [value_min, value_max],
            'time': result.solution_time if hasattr(result, 'solution_time') else 0
        }
        
        print(f"  Solution completed in {results[method]['time']:.3f} seconds")
        print(f"  Optimal control at {test_point}: {optimal_control}")
        print(f"  Value function range: [{value_min:.4f}, {value_max:.4f}]")
    
    # Compare results if multiple methods were tested
    if len(methods) > 1:
        print("\n" + "="*60)
        print("Comparison of Solution Methods")
        print("="*60)
        
        # Get controls from both methods
        control1 = results['value_iteration']['control']
        control2 = results['policy_iteration']['control']
        
        print(f"\nTest state: {test_point}")
        print(f"Optimal control from value iteration: {control1}")
        print(f"Optimal control from policy iteration: {control2}")
        
        # Calculate difference
        diff_norm = np.linalg.norm(control1 - control2)
        print(f"Control difference (norm): {diff_norm:.6f}")
        
        if diff_norm < 0.01:
            print("✓ Both methods provide similar results (difference < 0.01)")
        else:
            print("⚠ Methods provide different results")
    
    # Return solvers for further testing
    solvers = [results[method]['solver'] for method in methods]
    
    if len(solvers) == 1:
        return solvers[0]
    else:
        return solvers[0], solvers[1]


def main():
    """Main function: run tests"""
    print("MDP Solver Test Script")
    print("="*60)
    
    # Select solver version
    solver_factory, device = select_solver_version()
    
    # User selects test mode
    print("\n" + "="*60)
    print("Select Test Mode")
    print("="*60)
    print("  1. Comprehensive test (takes longer, tests all functions)")
    print("  2. Quick test (suitable for development and debugging)")
    print("  3. Test interpolation methods")
    print("  4. Test parameter management")
    print("  5. Test value function iteration")
    print("  6. Test solve method (unified interface)")
    print("  7. Compare CPU vs GPU performance")
    
    choice = input("\nEnter choice (1-7): ").strip()
    
    if choice == '1':
        print("\nStarting comprehensive test...")
        # Run multiple tests
        test_interpolation_methods(solver_factory)
        test_parameter_management(solver_factory)
        test_value_iteration(solver_factory)
        test_solve_method(solver_factory)
        print("\n✓ Comprehensive test completed!")
    elif choice == '2':
        print("\nStarting quick test...")
        # Run quick test
        reward_func, transition_func, constraint_func = setup_lq_problem()
        
        solver = solver_factory(
            dim_x=2, dim_u=2, dim_epsilon=2, n_samples=50,
            transition_func=transition_func, reward_func=reward_func,
            constraint_func=constraint_func, beta=0.95
        )
        
        solver_result = (solver
                        .set_grids(
                            x_limits=[[-1.5, 1.5], [-1.5, 1.5]],
                            u_limits=[[-0.8, 0.8], [-0.8, 0.8]],
                            x_grid_nums=[10, 10],
                            u_grid_nums=[5, 5]
                        )
                        .set_ppf(stats.norm.ppf, n_quantiles=200)
                        .solve(
                            method='value_iteration',
                            tol=1e-4,
                            max_iter=50,
                            verbose=True
                        ))
        
        print("\n✓ Quick test completed!")
        return solver_result
    elif choice == '3':
        print("\nTesting interpolation methods...")
        test_interpolation_methods(solver_factory)
        print("\n✓ Interpolation methods test completed!")
    elif choice == '4':
        print("\nTesting parameter management...")
        test_parameter_management(solver_factory)
        print("\n✓ Parameter management test completed!")
    elif choice == '5':
        print("\nTesting value function iteration...")
        solver, conv_info = test_value_iteration(solver_factory)
        print("\n✓ Value function iteration test completed!")
        return solver
    elif choice == '6':
        print("\nTesting solve method...")
        result = test_solve_method(solver_factory)
        print("\n✓ Solve method test completed!")
        return result
    elif choice == '7':
        print("\nComparing CPU vs GPU performance...")
        if GPU_AVAILABLE and CPU_AVAILABLE:
            compare_cpu_gpu_performance()
        else:
            print("Both CPU and GPU solvers must be available for comparison.")
        print("\n✓ Performance comparison completed!")
    else:
        print("Invalid choice, using default quick test mode")
        reward_func, transition_func, constraint_func = setup_lq_problem()
        
        solver = solver_factory(
            dim_x=2, dim_u=2, dim_epsilon=2, n_samples=50,
            transition_func=transition_func, reward_func=reward_func,
            constraint_func=constraint_func, beta=0.95
        )
        
        solver_result = (solver
                        .set_grids(
                            x_limits=[[-1.5, 1.5], [-1.5, 1.5]],
                            u_limits=[[-0.8, 0.8], [-0.8, 0.8]],
                            x_grid_nums=[10, 10],
                            u_grid_nums=[5, 5]
                        )
                        .set_ppf(stats.norm.ppf, n_quantiles=200)
                        .solve(
                            method='value_iteration',
                            tol=1e-4,
                            max_iter=50,
                            verbose=True
                        ))
        if solver:
            print("\n✓ Test completed!")
        return solver_result
    
    print("\n" + "="*60)
    print("Test script completed")
    print("="*60)
    

if __name__ == "__main__":
    main()

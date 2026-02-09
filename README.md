# mdp-solver
MDP Solver for Economic Models

Usage Example:
----------
```python
# Initialize solver
solver = MDPSolverCPU(
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

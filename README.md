# mdp-solver
Markov Decision Process (MDP) solver specifically designed for economic model optimization.

$$
\begin{split}
J\left(x_{t}\right)=\sup_{u_{t}}\left\{ R\left(x_{t},u_{t}\right)+\beta\mathbb{E}_{\varepsilon_{t+1}} J\left(x_{t+1}\right) \right\} \\
\text{s.t. }\left\{ \begin{array}{ll}
x_{t+1} = F\left(x_{t},u_{t},\varepsilon_{t+1}\right)\\
H\left(x_{t},u_{t}\right)\leq 0
\end{array}\right.
\end{split}
$$

## ✨ Features

- 🚀 **GPU Acceleration Supported**: Leverages PyTorch for large-scale parallel computing (allows input functions to be CPU-based)
- 📊 **Various Economic Models**: LQR, RBC, consumption-saving, job search, and more
- ⚡ **Built-in Algorithms**: Value iteration and policy iteration
- 🎯 **Constraint Handling**: Interior-point penalty function method
- 📈 **Visualization**: Rich charting and analysis tools
- 🔧 **Easy Extension**: Modular design, easy to add new models


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

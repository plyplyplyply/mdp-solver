# Theoretical Foundations (English)

## 1 Infinite Horizon Discrete-Time MDP Stochastic Dynamic Programming

### 1.1 Model Setup

Let $\mathcal{X}\subseteq\mathbb{R}^{n_{x}}$ be the state space, where the state variable $\mathbf{x}_{t}\in\mathcal{X}$ is an $n_{x}$-dimensional real vector. Let $\mathcal{U}\subseteq\mathbb{R}^{n_{u}}$ be the control space, where the control variable $\mathbf{u}_{t}\in\mathcal{U}$ is an $n_{u}$-dimensional real vector. The random disturbance $\varepsilon_{t}\in\mathbb{R}^{n_{\varepsilon}}$ is an independent and identically distributed (i.i.d.) random vector whose components are mutually independent, with joint distribution $p\left(\varepsilon\right)$. The system dynamics are described by the state transition function $F:\mathcal{X}\times\mathcal{U}\times\mathbb{R}^{n_{\varepsilon}}\to\mathcal{X}$, i.e., $\mathbf{x}_{t+1}=F\left(\mathbf{x}_{t},\mathbf{u}_{t},\varepsilon_{t+1}\right)$. The reward function $R:\mathcal{X}\times\mathcal{U}\to\mathbb{R}$ gives the immediate reward for taking control $\mathbf{u}_{t}$ in state $\mathbf{x}_{t}$. The constraint function $H:\mathcal{X}\times\mathcal{U}\to\mathbb{R}^{n_{h}}$ defines the feasibility condition for the control variable, requiring $H\left(\mathbf{x}_{t},\mathbf{u}_{t}\right)\leq\mathbf{0}$ (component-wise inequality). The discount factor $\beta\in\left(0,1\right)$ is used to discount future rewards. The value function $J:\mathcal{X}\to\mathbb{R}$ represents the maximum expected cumulative discounted reward starting from a given state, satisfying the Bellman optimality equation:

\begin{split}
J\left(\mathbf{x}_{t}\right)=\sup_{\mathbf{u}_{t}}\left\{ R\left(\mathbf{x}_{t},\mathbf{u}_{t}\right)+\beta\mathbb{E}_{\varepsilon_{t+1}}J\left(\mathbf{x}_{t+1}\right)\right\} \\
\text{s.t. }\left\{ \begin{array}{ll}
\mathbf{x}_{t+1}=F\left(\mathbf{x}_{t},\mathbf{u}_{t},\varepsilon_{t+1}\right)\\
H\left(\mathbf{x}_{t},\mathbf{u}_{t}\right)\leq\mathbf{0}
\end{array}\right.
\end{split}

To facilitate numerical solution, a penalty function is introduced to handle constraints. Define the penalty term as $C\left(\mathbf{x}_{t},\mathbf{u}_{t}\right)=\exp\left(\gamma\cdot\|\max\left\{ H\left(\mathbf{x}_{t},\mathbf{u}_{t}\right),\mathbf{0}\right\} \|_{2}^{2}\right)-1$, where $\gamma>0$ is the penalty scale parameter, $\max\left\{ \cdot,\mathbf{0}\right\}$ denotes taking the non-negative part element-wise, and $\|\cdot\|_{2}^{2}$ is the square of the L2 norm. Construct the penalized reward function $R_{\text{pen}}\left(\mathbf{x}_{t},\mathbf{u}_{t}\right)=R\left(\mathbf{x}_{t},\mathbf{u}_{t}\right)-\lambda\cdot C\left(\mathbf{x}_{t},\mathbf{u}_{t}\right)$, where $\lambda>0$ is the penalty weight coefficient. Thus, the constrained optimization problem is transformed into an unconstrained one, i.e., maximizing the penalized expected cumulative discounted reward.

### 1.2 Solution Algorithms

Based on the above framework, we design two iterative algorithms: Value Function Iteration and Policy Function Iteration. Both algorithms require the following inputs: the state space $\mathcal{X}$ (discrete grid), the control space $\mathcal{U}$ (discrete grid), the state transition function $F$, the original reward function $R$, the constraint function $H$ (optional), the discount factor $\beta$, the penalty parameters $\lambda$ and $\gamma$, the random disturbance distribution $p\left(\varepsilon\right)$ (optional, an empty value indicates deterministic dynamic programming), and the convergence tolerance $\epsilon>0$. Both algorithms ultimately output the optimal policy $\pi^{*}:\mathcal{X}\to\mathcal{U}$ and the optimal value function $Q^{*}:\mathcal{X}\to\mathbb{R}$. For Value Function Iteration, an initial value function $Q_{0}:\mathcal{X}\to\mathbb{R}$ is also required as input; for Policy Function Iteration, an initial policy $\pi_{0}:\mathcal{X}\to\mathcal{U}$ is required.

#### 1.2.1 Value Function Iteration

After initializing iteration counter $k=0$, repeat the following process starting from $k=1$:

\begin{aligned}
& \text{For each state }\ \mathbf{x}\in\mathcal{X}:\\
& \quad\text{Define the unconstrained optimization problem:}\\
& \qquad Q_{k}\left(\mathbf{x},\mathbf{u}\right)=R_{\text{pen}}\left(\mathbf{x},\mathbf{u}\right)+\beta\mathbb{E}_{\varepsilon}\left\{ Q_{k}\left[F\left(\mathbf{x},\mathbf{u},\varepsilon\right)\right]\right\} \\
& \quad\text{Solve:}\\
& \qquad J_{k+1}\left(\mathbf{x}\right)=\sup_{\mathbf{u}\in\mathcal{U}}Q_{k}\left(\mathbf{x},\mathbf{u}\right)\\
& \qquad\pi_{k+1}\left(\mathbf{x}\right)=\arg\sup_{\mathbf{u}\in\mathcal{U}}Q_{k}\left(\mathbf{x},\mathbf{u}\right)
\end{aligned}

until $\Delta_{k}=\sup_{\mathbf{x}\in\mathcal{X}}|J_{k+1}\left(\mathbf{x}\right)-J_{k}\left(\mathbf{x}\right)|<\epsilon$. Given the property of $\Delta_{k}$ as a (supremum) norm, we have $\Delta_{k}\leq\beta^{k-1}\Delta_{1}$, which yields the maximum number of iterations for a given convergence tolerance $\epsilon>0$ as $\frac{\ln\left(\epsilon/\Delta_{1}\right)}{\ln\beta}+1$.

#### 1.2.2 Policy Iteration

After initializing iteration counter $k=0$, repeat the following process starting from $k=1$:

\begin{aligned}
& \textbf{Policy Evaluation}\\
& \text{Compute }J_{k}\left(\mathbf{x}\right)\leftarrow R_{\text{pen}}\left(\mathbf{x},\pi_{k}(\mathbf{x})\right)+\beta\mathbb{E}_{\varepsilon}\left\{ J_{k}\left[F\left(\mathbf{x},\mathbf{u},\varepsilon\right)\right]\right\} ,\quad\forall\mathbf{x}\in\mathcal{X} \text{ via iterative calculation.}\\
& \textbf{Policy Improvement:}\\
& \pi_{k+1}\left(\mathbf{x}\right)=\arg\sup_{\mathbf{u}\in\mathcal{U}}\left\{ R_{\text{pen}}\left(\mathbf{x},\mathbf{u}\right)+\beta\mathbb{E}\left\{ J_{k}\left[F\left(\mathbf{x},\mathbf{u},\varepsilon\right)\right]\right\} \right\} ,\quad\forall\mathbf{x}\in\mathcal{X}.
\end{aligned}

until $\Delta_{k}=\sup_{\mathbf{x}\in\mathcal{X}}|\pi_{k+1}\left(\mathbf{x}\right)-\pi_{k}\left(\mathbf{x}\right)|<\epsilon$.

#### 1.2.3 Numerical Expectation

In the standard MDP form defined above, we require the components of the random disturbance $\varepsilon_{t}$ to be mutually independent. This allows for the following probability integral transform and its inverse properties, facilitating Monte Carlo sampling for distribution calculations.

Let the components of the random vector $\mathbf{x}=\left(x_{1},\dots,x_{n}\right)^{\top}$ be mutually independent, and let the marginal cumulative distribution function of $x_{i}$ be $F_{i}\left(x\right)$ (strictly monotonic increasing), with its corresponding quantile function $F_{i}^{-1}\left(q\right)=\inf\left\{ x:F_{i}\left(x\right)\ge q\right\}$:

1. **Forward Transform (Uniformization):**
   Let $u_{i}:=F_{i}\left(x_{i}\right),\quad i=1,\dots,n$, then the random vector $\mathbf{u}=\left(u_{1},\dots,u_{n}\right)^{\top}$ follows an $n$-dimensional independent uniform distribution $U\left[0,1\right]^{n}$.

2. **Inverse Transform (Generating Arbitrary Distributions):**
   If the random vector $\mathbf{u}=\left(u_{1},\dots,u_{n}\right)^{\top}\sim U\left[0,1\right]^{n}$ with independent components, and let $x_{i}:=F_{i}^{-1}\left(u_{i}\right),\quad i=1,\dots,n$, then the components of $\mathbf{x}=\left(x_{1},\dots,x_{n}\right)^{\top}$ are mutually independent, $x_{i}\sim F_{i}$, and the joint distribution function is $F\left(\mathbf{x}\right)=\prod_{i=1}^{n}F_{i}\left(x_{i}\right)$.

### 1.3 Standard Model Examples

#### 1.3.1 Two-Dimensional Linear Quadratic Gaussian (LQG) Problem

Consider a linear quadratic Gaussian problem with two state variables and two control variables. This problem has a closed-form analytical solution (via the Riccati equation), making it suitable for verifying the accuracy of numerical solvers.

Minimize the expected discounted quadratic loss function:

$$\min_{\{\mathbf{u}_{s}\}_{s=t}^{\infty}}\mathbb{E}_{t}\sum_{s=t}^{\infty}\beta^{s-t}\left(\mathbf{x}_{s}^{\top}\mathbf{Q}\mathbf{x}_{s}+\mathbf{u}_{s}^{\top}\mathbf{R}\mathbf{u}_{s}+2\mathbf{x}_{s}^{\top}\mathbf{N}\mathbf{u}_{s}\right),$$

subject to the linear state transition equation:

$$\mathbf{x}_{t+1}=\mathbf{A}\mathbf{x}_{t}+\mathbf{B}\mathbf{u}_{t}+\mathbf{C}\varepsilon_{t+1},\quad\varepsilon_{t+1}\sim\mathcal{N}\left(\mathbf{0},\mathbf{I}_{n_{\varepsilon}}\right).$$

Where: $\mathbf{x}_{t}\in\mathbb{R}^{2}$ is the two-dimensional state variable, $\mathbf{u}_{t}\in\mathbb{R}^{2}$ is the two-dimensional control variable, $\mathbf{Q}\in\mathbb{R}^{2\times2}$ is a positive semi-definite state weight matrix, $\mathbf{R}\in\mathbb{R}^{2\times2}$ is a positive definite control weight matrix, $\mathbf{N}\in\mathbb{R}^{2\times2}$ is a cross-term weight matrix, $\mathbf{A}\in\mathbb{R}^{2\times2}$, $\mathbf{B}\in\mathbb{R}^{2\times2}$, $\mathbf{C}\in\mathbb{R}^{2\times2}$ are system matrices, and $\varepsilon_{t+1}\sim\mathcal{N}\left(\mathbf{0},\mathbf{I}_{2}\right)$ is an i.i.d. two-dimensional Gaussian random vector.

Following the standard MDP form defined above, we set:

* State variable $\mathbf{x}_{t}=\left(x_{1,t},x_{2,t}\right)^{\top}\in\mathbb{R}^{2}$;
* Control variable $\mathbf{u}_{t}=\left(u_{1,t},u_{2,t}\right)^{\top}\in\mathbb{R}^{2}$;
* Reward function $R\left(\mathbf{x}_{t},\mathbf{u}_{t}\right)=-\left(\mathbf{x}_{t}^{\top}\mathbf{Q}\mathbf{x}_{t}+\mathbf{u}_{t}^{\top}\mathbf{R}\mathbf{u}_{t}+2\mathbf{x}_{t}^{\top}\mathbf{N}\mathbf{u}_{t}\right)$;
* State transition function $F\left(\mathbf{x}_{t},\mathbf{u}_{t},\varepsilon_{t+1}\right)=\mathbf{A}\mathbf{x}_{t}+\mathbf{B}\mathbf{u}_{t}+\mathbf{C}\varepsilon_{t+1}$, where $\varepsilon_{t+1}\sim\mathcal{N}\left(\mathbf{0},\mathbf{I}_{2}\right)$.

#### 1.3.2 Planned Economy RBC (Real Business Cycle)

A social planner maximizes the expected lifetime utility of a representative household by choosing consumption and labor input. The optimization problem is:

$$\max_{\left\{ C_{s},N_{s},K_{s+1}\right\} _{s=t}^{\infty}}\mathbb{E}_{t}\sum_{s=t}^{\infty}\beta^{s-t}\left(\frac{C_{s}^{1-\gamma}}{1-\gamma}-\chi\frac{N_{s}^{1+\phi}}{1+\phi}\right),$$

subject to the capital accumulation equation $K_{t+1}=\left(1-\delta\right)K_{t}+A_{t}K_{t}^{\alpha}N_{t}^{1-\alpha}-C_{t}$ and the stochastic TFP process $\ln A_{t+1}=\rho\ln A_{t}+\sigma\varepsilon_{t+1}$.

Following the standard MDP form defined above, we set:

* State variable $\mathbf{x}_{t}=\left(\ln K_{t},\ln A_{t}\right)^{\top}$;
* Control variable $\mathbf{u}_{t}=\left(\ln C_{t},\ln N_{t}\right)^{\top}$;
* Reward function $R\left(\mathbf{x}_{t},\mathbf{u}_{t}\right)=\frac{e^{\left(1-\gamma\right)\ln C_{t}}}{1-\gamma}-\chi\frac{e^{\left(1+\phi\right)\ln N_{t}}}{1+\phi}$;
* State transition function:
  $$F\left(\mathbf{x}_{t},\mathbf{u}_{t},\varepsilon_{t+1}\right)=\begin{pmatrix}\ln\left(\left(1-\delta\right)e^{\ln K_{t}}+e^{\ln A_{t}+\alpha\ln K_{t}+\left(1-\alpha\right)\ln N_{t}}-e^{\ln C_{t}}\right)\\
\rho\ln A_{t}
\end{pmatrix}+\begin{pmatrix}0\\
\sigma
\end{pmatrix}\begin{pmatrix}\varepsilon_{t+1}\end{pmatrix},$$
  where $\varepsilon_{t+1}\sim\mathcal{N}\left(0,1\right)$;
* Constraint condition $H\left(\mathbf{x}_{t},\mathbf{u}_{t}\right)=e^{\ln C_{t}}-\left(1-\delta\right)e^{\ln K_{t}}-e^{\ln A_{t}+\alpha\ln K_{t}+\left(1-\alpha\right)\ln N_{t}}\leq0$.

#### 1.3.3 Consumption-Saving Model with Borrowing Constraint and Habit Formation

An individual chooses consumption to maximize expected lifetime utility under a borrowing constraint and habit-forming preferences. The optimization problem is:

$$\max_{\left\{ C_{s},A_{s+1},S_{s+1}\right\} _{s=t}^{\infty}}\mathbb{E}_{t}\sum_{s=t}^{\infty}\beta^{s-t}\frac{\left(C_{s}-\theta S_{s}\right)^{1-\gamma}}{1-\gamma},$$

subject to asset accumulation $A_{t+1}=\left(1+r\right)A_{t}+Y_{t}-C_{t}$, habit updating $S_{t+1}=\rho_{s}S_{t}+\left(1-\rho_{s}\right)C_{t}$, and the stochastic income process $\ln Y_{t+1}=\mu_{y}+\rho_{y}\ln Y_{t}+\sigma\varepsilon_{t+1}$, with the asset lower bound constraint $A_{t+1}\geq\underline{A}$ and the asset upper bound constraint $A_{t+1}\leq\bar{A}$.

Following the standard MDP form defined above, we set:

* State variable $\mathbf{x}_{t}=\left(A_{t},\ln S_{t},\ln Y_{t}\right)^{\top}$;
* Control variable $\mathbf{u}_{t}=\ln\left(C_{t}-\theta S_{t}\right)$;
* Reward function $R\left(\mathbf{x}_{t},\mathbf{u}_{t}\right)=\frac{e^{\left(1-\gamma\right)\ln\left(C_{t}-\theta S_{t}\right)}}{1-\gamma}$;
* State transition function:
  $$F\left(\mathbf{x}_{t},\mathbf{u}_{t},\varepsilon_{t+1}\right)=\begin{pmatrix}\left(1+r\right)A_{t}+e^{\ln Y_{t}}-e^{\ln\left(C_{t}-\theta S_{t}\right)}-\theta e^{\ln S_{t}}\\
\ln\left\{ \rho_{s}e^{\ln S_{t}}+\left(1-\rho_{s}\right)\left(e^{\ln\left(C_{t}-\theta S_{t}\right)}+\theta e^{\ln S_{t}}\right)\right\} \\
\mu_{y}+\rho_{y}\ln Y_{t}
\end{pmatrix}+\begin{pmatrix}0\\
0\\
\sigma
\end{pmatrix}\begin{pmatrix}\varepsilon_{t+1}\end{pmatrix},$$
  where $\varepsilon_{t+1}\sim\mathcal{N}\left(0,1\right)$;
* Constraint condition:
  $$H\left(\mathbf{x}_{t},\mathbf{u}_{t}\right)=\begin{pmatrix}-\left(1+r\right)A_{t}-e^{\ln Y_{t}}+e^{\ln\left(C_{t}-\theta S_{t}\right)}+\theta e^{\ln S_{t}}+\underline{A}\\
\left(1+r\right)A_{t}+e^{\ln Y_{t}}-e^{\ln\left(C_{t}-\theta S_{t}\right)}-\theta e^{\ln S_{t}}-\bar{A}
\end{pmatrix}\leq\mathbf{0}.$$

#### 1.3.4 McCall Model with Continuous Decision Smoothing

A job seeker receives a random wage offer $W_{t}\sim\text{i.i.d. }\text{lognormal}\left(\mu,\sigma\right)$ each period and decides whether to accept it. If accepted, they work permanently at that wage; if rejected, they receive unemployment benefit $b$ and continue searching. To facilitate optimization, we smooth the discrete accept/reject decision: the job seeker uses a continuous decision variable $d_{t}\in\mathbb{R}$, where the probability of accepting the current wage offer is given by the Sigmoid function $P\left(d_{t}\right)=1/\left(1+e^{-\lambda d_{t}}\right)$, and the probability of rejecting is $1-P\left(d_{t}\right)$. The limit $\lambda\rightarrow+\infty$ corresponds to the standard McCall model.

The goal is to maximize the expected discounted lifetime income:

$$\max_{\{d_{s}\}_{s=t}^{\infty}}\mathbb{E}_{t}\left[\sum_{s=t}^{\infty}\beta^{s-t}I_{s}\right],$$

where $I_{t}$ is the income in period $t$.

Following the standard MDP form defined above, we set:

* State variable $\mathbf{x}_{t}=\left(S_{t},\ln K_{t},\ln W_{t}\right)^{\top}$;
    * $S_{t}\in\left\{ 0,1\right\}$ is the employment status from the previous period (0 = unemployed and searching, 1 = employed, initialized to 0)
    * $\ln K_{t}$ is the logarithm of the wage accepted up to the previous period (meaningless if not yet employed, can be initialized to 0)
    * $\ln W_{t}$ is the logarithm of the wage offer received in the current period;
* Control variable $\mathbf{u}_{t}=d_{t}\in\mathbb{R}$;
* Reward function $R\left(\mathbf{x}_{t},\mathbf{u}_{t}\right)=S_{t}e^{\ln K_{t}}+\left(1-S_{t}\right)\left[P\left(d_{t}\right)e^{\ln W_{t}}+\left(1-P\left(d_{t}\right)\right)b\right]$;
* State transition function:
  $$F\left(\mathbf{x}_{t},\mathbf{u}_{t},\varepsilon_{t+1}\right)=S_{t}\begin{pmatrix}1\\
\ln K_{t}\\
\ln K_{t}
\end{pmatrix}+\left(1-S_{t}\right)\left[P\left(d_{t}\right)\begin{pmatrix}1\\
\ln W_{t}\\
\ln W_{t}
\end{pmatrix}+\left(1-P\left(d_{t}\right)\right)\begin{pmatrix}0\\
\ln K_{t}\\
\mu+\sigma\varepsilon_{t+1}
\end{pmatrix}\right],$$
  where $\varepsilon_{t+1}\sim\mathcal{N}\left(0,1\right)$.

Thus, the Bellman equation with an absorbing state can be uniformly written as:

$$J\left(\mathbf{x}_{t}\right)=\sup_{d_{t}}\left\{ R\left(\mathbf{x}_{t},d_{t}\right)+\beta\mathbb{E}_{t}J\left(\mathbf{x}_{t+1}\right)\right\},$$

where the next-period state $\mathbf{x}_{t+1}$ is given by the transition function $F$ above. This form automatically encompasses the absorbing nature of the employment state: when $S_{t}=1$, the first term of the transition function takes effect, the optimal decision no longer affects the future, and the value function reduces to $J\left(\mathbf{x}_{t}\right)=\frac{e^{\ln K_{t}}}{1-\beta}$.
# Boundedness of Linear Programs for Data-Driven Optimal Control via Moment-Matching

This repository contains the reference implementation for the paper "Boundedness of Linear Programs for Data-Driven Optimal Control via Moment-Matching" by Andrea Martinelli, Lucia Pezzetti, Niklas Schmid, and John Lygeros.

## Requirements
Use Python 3.10 or above and install requirements by running

```
pip install -r requirements.txt
```
The experiments in this repository use Mosek (https://www.mosek.com) as the primary linear programming (LP) solver for the one-shot LP formulation. While other solvers (e.g., HiGHS or Gurobi) may work in principle, all results reported in the paper were obtained using Mosek. For full reproducibility — especially regarding numerical stability and solve times — we therefore strongly recommend using the same solver. Mosek requires a valid license, researchers affiliated with a university can obtain a free academic license:
1. Create an account at https://www.mosek.com/products/academic-licenses
2. Register using your institutional email address.
3. Request an Academic License.
4. Download the license file (`mosek.lic`).
5. Place `mosec.lic` in
   - Linux/macOS: ```~/mosek/mosek.lic```
   - Windows: ```C:\Users\<username>\mosek\mosek.lic```

# LTI systems
The `data/dx_*_du_2_systems.json` files store controllable LTI systems with increasing state dimension and 2 inputs according to $x_{k+1} = Ax_k + Bu_k$, where $A \in \mathbb{R}^{n_x \times n_x}$, $B \in \mathbb{R}^{n_x \times 2}$, and $n_x \in \{2, \dots, 30\}$. The diagonal entries of $A$ are fixed to $A_{ii} = 0.5$, while all off-diagonal elements of $A$ and $B$ are drawn in an Erdős–Rényi fashion:

$$
A_{ij}, B_{ij} \sim
\begin{cases}
0, & \text{with prob. } 0.1, \\
\mathcal{U}([-0.1, 0.1]), & \text{with prob. } 0.9.
\end{cases}
$$

To reproduce the results in the paper run 

```
./run_bounded_ps_vs_dim_linear.sh
```

## Results
The obtained results are reported in the following plots

<p align="center">
  <img src="figures/boundedness_heatmap_dx_vs_N.png" height="250" style="vertical-align: middle;">
  <img src="figures/policy_value_comparison_N500_3000.png" height="250" style="vertical-align: middle;">
</p>

Left: Empirical boundedness rate of the linear programs as a function of the state dimension $dx$ (and the the size of the decision variable matrix in the LP $d$) and the number of samples $N$. Each entry corresponds to the percentage of random problem instances (out of 10 seeds) for which the corresponding LP admits a bounded solution. The left panel shows the proposed moment-matching formulation, while the right panel reports the baseline LP using as direction a fixed identity covariance structure. 
Right: Normalized closed-loop cost difference between the learned policy and the optimal LQR policy, and normalized difference between the learned value function and the optimal LQR value for $N=500$ (up) and $N=3000$ (down). For each state dimension, solid lines indicate the mean normalized error across seeds, and shaded regions denote the standard deviation range.

# Mechanical system with cubic damping

We further investigate the proposed method on a class of nonlinear control systems given by an $n$-dimensional point-mass model with modal spring coupling and cubic drag.
The continuous-time dynamics are

$$
\begin{aligned}
\dot{p} &= v, \\
m \dot{v} &= - K p - c \|v\|^2 v + B u.
\end{aligned}
$$

where $m$ is the mass, $c>0$ is the cubic-drag coefficient, $u \in \mathbb{R}$ is a scalar control input, and $p,v \in \mathbb{R}^n$ denote position and velocity, respectively.
The stiffness matrix $K \in \mathbb{R}^{n \times n}$ is a positive semidefinite stiffness matrix of the form $K = Q \Lambda Q^T$ with $Q$ orthogonal and diagonal spectrum $\Lambda = \mathrm{diag}(\lambda_1,\dots,\lambda_n)$.
The eigenvalues follow a power-law growth

$$
    \lambda_i = k_0 i^{\alpha}, \quad i = 1, \dots, n,
$$

which yields increasingly fast and stiff high-frequency modes as the dimension grows. The base stiffness is set so that the highest modal natural frequency equals a target $\omega_{\max}$:

$$
  k_0 = \frac{m\,\omega_{\max}^{2}}{n^{\alpha}},
  \qquad \omega_{\max} = 5\;\text{rad/s}.
$$

The input matrix $B \in \mathbb{R}^{n \times 1}$ is chosen as a normalized dense random vector, meaning that the single scalar actuator applies force in a generic direction that influences all coordinates, while keeping the overall input magnitude independent of the state dimension.

To study variability across problem instances, we randomise the system parameters across experimental seeds. The physical parameters are drawn from log-normal distributions centred at their nominal values:

$$
\begin{aligned}
   m &\sim \mathrm{LogNormal}(\bar{m},\, \sigma_m), \\
   c &\sim \mathrm{LogNormal}(\bar{c},\, \sigma_c).
\end{aligned}
$$

with nominal values $\bar m = 5.0$\,kg, $\bar c = 0.5$\,N\,s$^{2}$/m$^{2}$, and scale parameters $\sigma_m = 0.3$, $\sigma_c = 0.5$ (corresponding to coefficients of variation of approximately $30\%$ and $50\%$, respectively).
The modal parameters are also randomised per seed:

$$
\begin{aligned}
   Q \sim \mathrm{Haar}\bigl(O(n)\bigr) \quad \text{(uniform random orthogonal basis)}, \\
   \alpha \sim \mathrm{Uniform}(1.8, 2.2) \quad \text{(modal growth exponent)}.
\end{aligned}
$$

To reproduce the results in the paper run

```
./run_bounded_lp_vs_dim_nonlinear.sh
```

## Results
The obtained results are reported in the following plots

<p align="center">
  <img src="figures/boundedness_heatmap_nonlinear_dx_vs_N.png" height="250" style="vertical-align: middle;">
  <img src="figures/trajectories_dx4.png" height="250" style="vertical-align: middle;">
</p>

Left: Empirical boundedness rate of the linear programs as a function of the state dimension $dx$ (and the the size of the decision variable matrix in the LP $d$) and the number of samples $N$. Each entry corresponds to the percentage of random problem instances (out of 10 seeds) for which the corresponding LP admits a bounded solution. The left panel shows the proposed moment-matching formulation, while the right panel reports the baseline LP using as direction a fixed identity covariance structure. Polynomial features with degree 2 in $x$ have been used.
Right: $N = 5000, M =500, dx = 4$, $m=5kg, w_{max} = 5 rad/s, c = 0.5 Ns^2/m^2$. MM cost: $9675.50 \pm 5069.84$   |  Uncontrolled cost: $10104.59 \pm 5050.46$

# References and Contacts
Please reach out to lpezzetti@ethz.ch for any question about the code

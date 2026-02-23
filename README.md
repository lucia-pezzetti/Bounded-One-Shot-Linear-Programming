# Boundedness of Linear Programs for Data-Driven Optimal Control via Moment-Matching

This repository contains the reference implementation for the paper "Boundedness of Linear Programs for Data-Driven Optimal Control via Moment-Matching" by Andrea Martinelli, Lucia Pezzetti, Niklas Schmid, and John Lygeros.

##Requirements
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



# Mechanical system with cubic damping

# References and Contacts
Please reach out to lpezzetti@ethz.ch for any question about the code

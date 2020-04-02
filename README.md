# svm-dual-optimization
Using CVXopt to solve the SVM dual problem

<a href="http://cvxopt.org/">CVXOPT</a> is a free software package for convex optimization based on the Python programming 
language.

By solving for the [[Duality (optimization)|Lagrangian dual]] of the above problem, one obtains the simplified problem

: <math> \text{maximize}\,\, f(c_1 \ldots c_n) =  \sum_{i=1}^n c_i - \frac 1 2 \sum_{i=1}^n\sum_{j=1}^n y_ic_i(x_i \cdot x_j)y_jc_j,</math>
: <math> \text{subject to } \sum_{i=1}^n c_iy_i = 0,\,\text{and } 0 \leq c_i \leq \frac{1}{2n\lambda}\;\text{for all }i.</math>


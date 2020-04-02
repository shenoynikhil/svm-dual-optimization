# svm-dual-optimization
Implementation of Dual Support Vector Machine with a slack variable using quadratic programming with CVXOPT library

<a href="http://cvxopt.org/">CVXOPT</a> is a free software package for convex optimization based on the Python programming 
language.

By solving for the Duality (optimization)|Lagrangian dual]] of the above problem, one obtains the simplified problem

We first state the primal problem of hard margin linear SVM as

$$\min_{w,b} ||w||^2 $$ subject to $$y_i(w^Tx_i+b) \geq 1,$$ for $$i = 1,...,n$$

By Lagrangian function, coefficient $$\alpha$$ and fulfilling KKT(Karush-Kuhn-Tucker) conditions, we could solve for dual problem that optimizing $$\alpha$$ while minizing $$w, b$$ in the primal problem. That is written

$$\max_\alpha \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{ij} \alpha_i \alpha_j y_i y_j x_i^T x_j$$ subject to $$\alpha_i \geq 0$$ and $$\sum_i \alpha_i y_i = 0$$ for $$i = 1,...,n$$




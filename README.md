# svm-dual-optimization
<h3>Description:</h3>
<ul style="list-style-type:disc">
<li>A Python script to estimate from scratch Support Vector Machines for linear, polynomial and Gaussian kernels utilising the quadratic programming optimisation algorithm from library CVXOPT.</li>
<li>Support Vector Machines implemented from scratch and compared to scikit-learn's implementation.</li>

Given two classes of labelled examples, we are interested in finding a decision boundary resulting from an appropriate choice of support vectors.
 
#### Model
<li><p>Simulate labelled training dataset <img src="https://github.com/DrIanGregory/MachineLearning-SupportVectorMachines/blob/master/svgs/4388ea036963a2791929a7365e301c7a.svg" align=middle width=294.09701144999997pt height=27.91243950000002pt/> where there are N couples of <img src="https://github.com/DrIanGregory/MachineLearning-SupportVectorMachines/blob/master/svgs/81fe5e49971b8fdc94a28f66e9310309.svg" align=middle width=55.44161204999999pt height=24.65753399999998pt/> and k is the number (dimension) of x variables.</p></li>
<li>We are interested in SVM disrimitive analysis by finding the optimum decision boundary resulting from a choice of S support vectors.
This SVM optimization problem is a constrained convex quadratic optimization problem. 
Meaning it can be formulated in terms of Lagrange multipliers.
For the Lagrange multipliers to be applied under constraints, can use the Wolfe Dual Principle which 
is appropriate as long as the Karush Kuhn Tucker (KKT) conditions are satisfied.
Further, as this is a convex problem, Slater’s condition tells us that strong duality holds such that the dual and primal optimums give the solution.</li>
A benefit to this dual formulation is that the objective function only depends on the Lagrange multipliers (weights and intercept drop out).
Further, this formulation will be useful when requiring Kernals for entangled data sets.

<li>The Wolfe dual soft margin formula with kernel is given by

<p align="center"><img src="https://github.com/DrIanGregory/MachineLearning-SupportVectorMachines/blob/master/svgs/0acbd9783d20c53d1e9f750f2665520d.svg" align=middle width=333.89845664999996pt height=131.37932775pt/></p>

Where
<p><img src="https://github.com/DrIanGregory/MachineLearning-SupportVectorMachines/blob/master/svgs/c745b9b57c145ec5577b82542b2df546.svg" align=middle width=10.57650494999999pt height=14.15524440000002pt/> are the Lagrange multipliers, <img src="https://github.com/DrIanGregory/MachineLearning-SupportVectorMachines/blob/master/svgs/39ae080f4ae6ef7bda6a0ca0c44efc78.svg" align=middle width=32.48865674999999pt height=24.65753399999998pt/> is the kernel function, N are the number of training 
samples in the dataset, x is the matrix of training samples, y is the vector of target values, C is a supplied hyperparameter.</p>
</li>
<li>The non-zero Lagrange multipliers are the data points which contribute to the formation of the decision boundary.
<p>The hypothesis function <img src="https://github.com/DrIanGregory/MachineLearning-SupportVectorMachines/blob/master/svgs/4dd763dd7876885c2e5131a0b6d62d57.svg" align=middle width=133.02135495pt height=24.65753399999998pt/> is the decision boundary. The hypothesis formula in terms of the Kernel function is given by:</p></li>

<p align="center"><img src="https://github.com/DrIanGregory/MachineLearning-SupportVectorMachines/blob/master/svgs/554a33df7742aebf76ec7b81f6f3c17a.svg" align=middle width=283.76643075pt height=49.315569599999996pt/></p>
<p>Where S is the set of support vectors, <img src="https://github.com/DrIanGregory/MachineLearning-SupportVectorMachines/blob/master/svgs/c745b9b57c145ec5577b82542b2df546.svg" align=middle width=10.57650494999999pt height=14.15524440000002pt/> is the Lagrange multiplier, b is the bias term, y is the target from the examples, <img src="https://github.com/DrIanGregory/MachineLearning-SupportVectorMachines/blob/master/svgs/39ae080f4ae6ef7bda6a0ca0c44efc78.svg" align=middle width=32.48865674999999pt height=24.65753399999998pt/> is the Kernel and</p>

<p align="center"><img src="https://github.com/DrIanGregory/MachineLearning-SupportVectorMachines/blob/master/svgs/cb555672d4c84c369da09fd80f6811d8.svg" align=middle width=184.7945286pt height=69.0417981pt/></p>

<h3> Code for feeding data into CVXopt </h3>

The CVXOPT library solves the Wolfe dual soft margin constrained optimisation with the following API:
 
<p align="center"><img src="https://github.com/DrIanGregory/MachineLearning-SupportVectorMachines/blob/master/svgs/d815dd2e1e10d79a7162f6fe778314f4.svg" align=middle width=137.42467695pt height=78.26216475pt/></p>
<p>Note: <img src="https://github.com/DrIanGregory/MachineLearning-SupportVectorMachines/blob/master/svgs/ceddacf03a28d83100c38150c1076c1f.svg" align=middle width=12.785434199999989pt height=20.931464400000007pt/> indicates component-wise vector inequalities. It means that each row of the matrix <img src="https://github.com/DrIanGregory/MachineLearning-SupportVectorMachines/blob/master/svgs/b5087617bd5bed26b1da99fefb5353f1.svg" align=middle width=23.50114799999999pt height=22.465723500000017pt/> represents an inequality that must be satisfied.</p>
 
To use the CVXOPT convex solver API. The Wolfe dual soft margin formula is re-written as follows

<p align="center"><img src="https://github.com/DrIanGregory/MachineLearning-SupportVectorMachines/blob/master/svgs/a364906d0854671fe9b9718ce4ce1ec3.svg" align=middle width=212.12443724999997pt height=81.45851505pt/></p>

Where 
<br>
<p>G is a Gram matrix of all possible dot products of vectors <img src="https://github.com/DrIanGregory/MachineLearning-SupportVectorMachines/blob/master/svgs/d7084ce258ffe96f77e4f3647b250bbf.svg" align=middle width=17.521011749999992pt height=14.15524440000002pt/>.</p>

<p align="center"><img src="https://github.com/DrIanGregory/MachineLearning-SupportVectorMachines/blob/master/svgs/5ceca286e4d3c1cb407465d5db863df5.svg" align=middle width=357.85148685pt height=88.76800184999999pt/></p>

<p align="center"><img src="https://github.com/DrIanGregory/MachineLearning-SupportVectorMachines/blob/master/svgs/ceeaf43e7d8f6cde00a8a21441244b9f.svg" align=middle width=386.18483804999994pt height=144.88403325pt/></p>



```python
# In functions.py
def perform_cvx_opt(X, y, C = None, soft_threshold = 1e-4, kernel = 'linear', p = 3, gamma = 1e-1):
    n_samples, n_features = X.shape
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            if kernel == 'linear':
                K[i,j] = linear_kernel(X[i], X[j])
            elif kernel == 'poly':
                K[i,j] = polynomial_kernel(X[i], X[j], p)
            elif kernel == 'rbf' :
                K[i, j] = gaussian_kernel(X[i], X[j], gamma)
                
    P = cvxopt_matrix(np.outer(y, y) * K)
    q = cvxopt_matrix(np.ones(n_samples) * -1)
    A = cvxopt_matrix(y.reshape(1, -1))
    A = cvxopt_matrix(A, (1, n_samples), 'd')
    b = cvxopt_matrix(0.0)
    
    if C is None:
        G = cvxopt_matrix(np.diag(np.ones(n_samples) * -1))
        h = cvxopt_matrix(np.zeros(n_samples))
    else:
        tmp1 = np.diag(np.ones(n_samples) * -1)
        tmp2 = np.identity(n_samples)
        G = cvxopt_matrix(np.vstack((tmp1, tmp2)))
        tmp1 = np.zeros(n_samples)
        tmp2 = np.ones(n_samples) * C
        h = cvxopt_matrix(np.hstack((tmp1, tmp2)))
    
    # solve QP problem
    solution = cvxopt_solvers.qp(P, q, G, h, A, b)
    
    # Lagrange multipliers
    a = np.ravel(solution['x'])

    # Calculate weights
    w = np.matrix(np.zeros((1, n_features)))
    for i in range(n_samples):
        w += a[i] * y[i] * X[i]
        
    # Calculate Intercepts
    intercept = 0
    for i in range(n_samples):
        if a[i] > soft_threshold:
            intercept = y[i] - w.dot(np.transpose(X[i]))
            break
    
    intercept = float(intercept)
    
    return w.T, intercept, a
```
   


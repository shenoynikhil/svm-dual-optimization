# svm-dual-optimization
<h3>Description:</h3>
<ul style="list-style-type:disc">
<li>A Python script to estimate from scratch Support Vector Machines for linear, polynomial and Gaussian kernels utilising the quadratic programming optimisation algorithm from library CVXOPT.</li>
<li>Support Vector Machines implemented from scratch and compared to scikit-learn's implementation.</li>

<h3>Given two classes of labelled examples, we are interested in finding a decision boundary resulting from an appropriate choice of support vectors.</h3>
 
<ul style="list-style-type:disc"> 
<h3>Model</h3>
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
<p>Where S is the set of support vectors, <img src="https://github.com/DrIanGregory/MachineLearning-SupportVectorMachines/blob/master/svgs/c745b9b57c145ec5577b82542b2df546.svg" align=middle width=10.57650494999999pt height=14.15524440000002pt/> is the Lagrange multiplier, b is the bias term, y is the target from the examples, <img src="" align=middle width=32.48865674999999pt height=24.65753399999998pt/> is the Kernel and</p>

<p align="center"><img src="https://github.com/DrIanGregory/MachineLearning-SupportVectorMachines/blob/master/svgs/cb555672d4c84c369da09fd80f6811d8.svg" align=middle width=184.7945286pt height=69.0417981pt/></p>



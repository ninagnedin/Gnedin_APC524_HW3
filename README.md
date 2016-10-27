# Gnedin_APC524_HW3

Newton Solver

This package implements a Newton solver. The included files are:

newton.py: 
Implements the class newton, which returns a new object to find the roots of f(x) = 0 using Newton Raphson method. Required arguments are:

tol: tolerance, iterate until |f(x)| < tol
maxiter: maximum number of iterations
dx: step size to compute the approximate Jacobian

Optional arguments are:

Df: The analytic Jacobian. If not provided, the approximate Jacobian is used.

r: If the approximate root does not lie within a radius r of the initial guess x0, an exception is raised. i.e. if || x_k - x0 || > r. The default value if r is not provided is 20.

functions.py:
Contains a class Polynomial which is a callable polynomial object provided array of coefficients. 
The function ApproximateJacobian returns an approximation of the Jacobian Df(x)as a numpy matrix.
y0 is an optional parameter for functions of two variables - if provided the solver will treat it as a constant and find a root of x conditional on y0.

testNewton.py:

Contains several tests for newton.py

testLinear: To test linear polynomial

testQuad: To test quadratic with single root

testQuad2: To test a quadratic with no roots - should fail

testTwoParams: To test function with two parameters

testRadis: To test the radius exception - should fail

testNewtonStep: Testing that a single step of Newton performs correctly



testFunctions:

testApproxJacobian1: Testing that the approximate jacobian is correct for a 1d function.

testApproxJacobian2: Testing that the approximate jacobian is correct for a 2d function.

testApproxJacobian3: Testing that the approximate jacobian is correct for a 3d function.

testPolynomial: Test the Polynomial class.

testAnalytic: Test that analytic Jacobian gives the correct result by comparing with the approximate Jacobian.

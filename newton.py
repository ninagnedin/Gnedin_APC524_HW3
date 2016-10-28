# newton - Newton-Raphson solver
#
# For APC 524 Homework 3
# CWR, 18 Oct 2010

import numpy as N
import functions as F

class Newton(object):
    def __init__(self, f, tol=1.e-6, maxiter=20, dx=1.e-6,Df=None,r=None):
        """Return a new object to find roots of f(x) = 0 using Newton's method.
        tol:     tolerance for iteration (iterate until |f(x)| < tol)
        maxiter: maximum number of iterations to perform
        dx:      step size for computing approximate Jacobian
        Df:      analytical expression for Jacobian
        r:       approximated root must lie within radius r of initial guess x0
        y0:      for function of 2 parameters although solver will only solve along x dimension"""
        self._f = f
        self._tol = tol
        self._maxiter = maxiter
        self._dx = dx
        if(r is None):
        #Default value for r - since there's no alternative need a default
            self._r=20
        else:
            self._r=r
        #If it's not entered it will remain none and will use approx later
        self._Df=Df 

    def solve(self, x0):
        """Return a root of f(x) = 0, using Newton's method, starting from
        initial guess x0"""
        x = x0
        for i in xrange(self._maxiter):
            fx = self._f(x)
            if N.linalg.norm(fx) < self._tol:
                return x
            x = self.step(x,fx) 
            if N.linalg.norm(x-x0)>self._r:
                raise Exception("Root not within threshold radius of initial guess")
        #Check for convergence and raises exception if it does not converge
        if N.linalg.norm(fx)>self._tol:
            raise Exception("Did not converge in provided number of iterations")
        return x

    def step(self, x, fx=None):
        """Take a single step of a Newton method, starting from x
        If the argument fx is provided, assumes fx = f(x)"""
        if fx is None:
            fx = self._f(x)
        if self._Df is None:
            Df_x = F.ApproximateJacobian(self._f, x, self._dx)
        else:
            Df_x=F.AnalyticJacobian(self._Df,x)
        h = N.linalg.solve(N.matrix(Df_x), N.matrix(fx))
        return x - h #Chaged to float(h[0]) to avoid adding matrix to float, switched to -

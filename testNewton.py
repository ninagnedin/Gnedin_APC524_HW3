#!/usr/bin/env python

import newton
import unittest
import numpy as N
import math
import functions as F

class TestNewton(unittest.TestCase):
    def testLinear(self):
        f = lambda x : 3.0 * x + 6.0
        solver = newton.Newton(f, tol=1.e-15, maxiter=2)
        x = solver.solve(2.0)
        self.assertAlmostEqual(x, -2.0) #Change to assertAlmostEqual
        
    #Basic test for quadratic - should pass
    def testQuad(self):
        #P=(x-2)**2=x*2-4x+4
        f=F.Polynomial([1,-4,4])
        _Df=F.Polynomial([0,2,-4])
        solver=newton.Newton(f,tol=1.e-15,maxiter=100,Df=_Df)
        x=solver.solve(10) 
        self.assertAlmostEqual(x,2)
    
    #Basic test for quadratic with no roots - should fail
    def testQuad2(self):
        #P=x**2+2*x+6
        f=F.Polynomial([1,2,6])
        _Df=F.Polynomial([0,2,2])
        solver=newton.Newton(f,tol=1.e-15,maxiter=100,Df=_Df)
        x=solver.solve(0) 
        self.assertAlmostEqual(x,-1)
    
    #Test radius works - this test should fail
    def testRadius(self):
        #P=(x-2)**2=x*2-4x+4
        f=F.Polynomial([1,-4,4])
        _Df=F.Polynomial([0,2,-4])
        solver=newton.Newton(f,tol=1.e-15,maxiter=100,Df=_Df)
        x=solver.solve(45)
        print(x)
        self.assertAlmostEqual(x,2)
    
    #Function with two parameters - using analytic Jacobian
    def testTwoParamsAnalytic(self):
        f= lambda x: N.matrix([[2*x[0,0]-6*x[1,0]],[ x[0,0]+x[1,0]-8]])
        _Df= lambda x: N.matrix([[2, -6],[1, 1]])
        solver = newton.Newton(f, tol=1.e-6, maxiter=100, Df=_Df)
        x0=N.matrix([[7.],[0.]])
        x = solver.solve(x0)
        N.testing.assert_array_almost_equal(x, N.matrix([[6.], [2.]]))
    
    #Same function with two parameters as above - using approximate Jacobian
    def testTwoParamsApprox(self):
        f= lambda x: N.matrix([[2*x[0,0]-6*x[1,0]],[ x[0,0]+x[1,0]-8]])
        solver = newton.Newton(f, tol=1.e-6, maxiter=100)
        x0=N.matrix([[7.],[0.]])
        x = solver.solve(x0)
        N.testing.assert_array_almost_equal(x, N.matrix([[6.], [2.]]))
      
    #Function with three parameters - using analytic Jacobian     
    def testThreeParamsAnalytic(self):
        f= lambda x: N.matrix([[2*x[0,0]-6*x[1,0]+x[2,0]],[ x[0,0]+x[1,0]+x[2,0]-8],[x[0,0]-12*x[1,0]+12*x[2,0]+18]])
        _Df= lambda x: N.matrix([[2, -6,1],[1, 1,1],[1.,-12.,12.]])
        solver = newton.Newton(f, tol=1.e-6, maxiter=100, Df=_Df)
        x0=N.matrix([[7.],[0.],[2.]])
        x = solver.solve(x0)
        N.testing.assert_array_almost_equal(x, N.matrix([[6.], [2.],[0.]]))

    #Same function with two parameters as above - using approximate Jacobian        
    def testThreeParamsApprox(self):
        f= lambda x: N.matrix([[2*x[0,0]-6*x[1,0]+x[2,0]],[ x[0,0]+x[1,0]+x[2,0]-8],[x[0,0]-12*x[1,0]+12*x[2,0]+18]])
        solver = newton.Newton(f, tol=1.e-6, maxiter=100)
        x0=N.matrix([[7.],[0.],[2.]])
        x = solver.solve(x0)
        N.testing.assert_array_almost_equal(x, N.matrix([[6.], [2.],[0.]]))
    
    #Tests that step of Newton behaves as it should
    def testNewtonStep(self):
        f = lambda x : math.sin(x)
        x0=2
        solver = newton.Newton(f,tol=1.e-15, maxiter=1)
        x=solver.step(x0)
        self.assertTrue(abs(x-2*math.pi)<abs(2*math.pi-x0))
        
 
    
if __name__ == "__main__":
    unittest.main()

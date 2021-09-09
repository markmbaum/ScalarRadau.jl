using ScalarRadau
using Test

#stiff test function
F(x, y, _) = 50*(cos(x) - y)
#exact solution
S(x) = (50*(-50*exp(-50*x) + sin(x) + 50*cos(x)))/2501

#test with evenly spaced output
x, y = radau(F, 0.0, 0.0, 6.0, 100, atol=1e-9, rtol=1e-9)
@test maximum(@. abs(y - S(x))) < 1e-6

#test with end-point output only
yₙ = radau(F, 0.0, 0.0, 6.0, atol=1e-9, rtol=1e-9)
@test abs(yₙ - S(6)) < 1e-6

#test for convergence failure
@test_throws ErrorException radau((x,y,p)->rand(), 0.0, 0.0, 1.0, atol=1e-9, rtol=1e-9)

#test for max steps failure
@test_throws ErrorException radau(F, 0.0, 0.0, 1e6, atol=1e-9, rtol=1e-9, maxstep=1)
using ScalarRadau
using Test

#stiff test function
F(x, y, p) = 50*(cos(x) - y)
#exact solution
S(x) = (50*(-50*exp(-50*x) + sin(x) + 50*cos(x)))/2501

#test with evenly spaced output
x, y = radau(F, 0, 0, 6, 100, atol=1e-9, rtol=1e-9)
@test maximum(@. abs(y - S(x))) < 1e-6

#test with end-point output only
yₙ = radau(F, 0, 0, 6, atol=1e-9, rtol=1e-9)
@test abs(yₙ - S(6)) < 1e-6

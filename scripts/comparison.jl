using ScalarRadau
using DifferentialEquations
using BenchmarkTools
using Statistics
using PyPlot

##

#stiff, scalar ODE for testing 
f_sr(x, y, _) = 50*(cos(x) - y)
f_de(u, _, t) = 50*(cos(t) - u)
#known solution
s(x) = (50*(-50*exp(-50*x) + sin(x) + 50*cos(x)))/2501

##

#number of tolerances to sample
N = 10
#tolerance values
tols = 10 .^ LinRange(-6, -2, N)
#times and errors for ScalarRadau evaluations
t_sr = zeros(N)
e_sr = zeros(N)
#times and errors for DifferentialEquations evaluations
t_de = zeros(N)
e_de = zeros(N)
#starting point
x₀ = 0.0
y₀ = 0.0
#finishing coordinate
xₙ = 3.0
#exact solution end values
sₙ = s(xₙ)
#the problem for DifferentialEquations
prob = ODEProblem(f_de, x₀, (x₀,xₙ))

##

for (i,tol) ∈ enumerate(tols)

    #ScalarRadau error
    yₙ = radau!(y, x, f_sr, y₀, x₀, xₙ, rtol=tol, atol=tol)
    e_sr[i] = abs(yₙ - sₙ)/abs(sₙ)
    #ScalarRadau timing
    b = @benchmark radau($f_sr, $y₀, $x₀, $xₙ, rtol=$tol, atol=$tol)
    t_sr[i] = mean(b.times)

    #DifferentialEquations error
    sol = solve(prob, RadauIIA5(), reltol=tol, abstol=tol, save_start=false, saveat=xₙ)
    e_de[i] = abs(sol.u[end] - sₙ)/abs(sₙ)
    #DifferentialEquations timing
    b = @benchmark solve($prob, $RadauIIA5(), reltol=$tol, abstol=$tol, save_start=false, saveat=$xₙ)
    t_de[i] = mean(b.times)

end
# ScalarRadau

[![Build Status](https://github.com/wordsworthgroup/ScalarRadau.jl/workflows/CI/badge.svg)](https://github.com/wordsworthgroup/ScalarRadau.jl/actions)
[![Coverage](https://codecov.io/gh/wordsworthgroup/ScalarRadau.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/wordsworthgroup/ScalarRadau.jl)

This module implements the 5th order, Radau IIA method for solving a **scalar** ordinary differential equation (ODE).
* The method is stiffly accurate, B-stable, fully-implicit, and therefore very effective for stiff equations.
* Implementation mostly follows the description in chapter IV.8 in [Solving Ordinary Differential Equations II](https://www.springer.com/gp/book/9783540604525), by Ernst Hairer and Gerhard Wanner.
* Step size is adaptive.
* Functions implemented here expect to use `Float64` numbers.
* Dense output for continuous solutions is implemented using cubic Hermite interpolation.
* Approximate Jacobian evaluation is performed with a finite difference.
* Because the equation is scalar and the method has three stages, the size of the Jacobian is always 3 by 3 and [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl) is used for the Newton iterations.

The implementation here is designed for repeated solution of a scalar ODE using different parameters, initial conditions, or solution intervals. Julia's compiler will specialize with your particular ODE.

For a nice overview of Radau methods and their utility, check out: [Stiff differential equations solved by Radau methods](https://www.sciencedirect.com/science/article/pii/S037704279900134X).

### Quick Start

To solve an ODE, first define the derivative `dy/dx` in the form of a function, then call the `radau` function.
```julia
using ScalarRadau
F(x, y, param) = -y
x, y = radau(F, 1, 0, 2, 25)
```
The snippet above solves the equation `dy/dx = -y`, starting at `y=1`, between `x=0` and `x=2`, and returns 25 evenly spaced points in the solution interval.

### Details

##### In-place Solution

For maximum control over output points, the in-place function is

```julia
radau!(yout, xout, F, y₀, x₀, xₙ, param=nothing; rtol=1e-6, atol=1e-6, facmax=100.0, facmin=0.01, κ=1e-3, ϵ=0.25, maxnwt=7, maxstp=1000000)
```
The mandatory function arguments are
* `yout` - vector where output points will be written
* `xout` - sorted vector of `x` values where output points should be sampled
* `F` - scalar ODE in the form `F(x, y, param)`
* `y₀` - initial value for `y`
* `x₀` - starting point for `x`
* `xₙ` - end point of the integration

By default, the `param` argument is `nothing`, but it may be any type. It is passed to the `F` function whenever it's evaluated.

Keyword arguments are
* `rtol` - relative error tolerance
* `atol` - absolute error tolerance
* `facmax` - maximum fraction that the step size may increase, compared to the previous step
* `facmin` - minimum fraction that the step size may decrease, compared to the previous step
* `κ` (kappa) - stopping tolerance for Newton iterations
* `ϵ` (epsilon) - fraction of current step size used for finite difference Jacobian approximation
* `maxnwt` - maximum number of Newton iterations before step size reduction
* `maxstp` - maximum number of steps for the solver stops and throws an error

Two other functions are available for convenience, both of which use the in-place version internally.

##### Evenly Spaced Output

For evenly spaced output points (as in the quick start example) the function definition is

```julia
radau(F, y₀, x₀, xₙ, nout, param=nothing; kwargs...)
```

In this case, you must specify the number of output points with the `nout` argument. Keyword arguments and default values are the same as above. Solution vectors for `x` and `y` are returned.

##### End Point Output

To compute only the `y` value at the end of the integration interval (`xₙ`), the function is
```julia
radau(F, y₀, x₀, xₙ, param=nothing; kwargs...)
```
Again, keywords arguments and default values are identical to the in-place function.

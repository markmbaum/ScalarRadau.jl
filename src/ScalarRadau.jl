module ScalarRadau

using StaticArrays: SMatrix, SVector
using ForwardDiff: derivative

export radau, radau!

#-------------------------------------------------------------------------------
#RK tableau for RadauIIA 5th order

const c₁ = (4 - √6)/10
const c₂ = (4 + √6)/10
const c₃ = 1.0

const a₁₁ = (88 - 7*√6)/360
const a₁₂ = (296 - 169*√6)/1800
const a₁₃ = (-2 + 3*√6)/225

const a₂₁ = (296 + 169*√6)/1800
const a₂₂ = (88 + 7*√6)/360
const a₂₃ = (-2 - 3*√6)/225

const a₃₁ = (16 - √6)/36
const a₃₂ = (16 + √6)/36
const a₃₃ = 1/9

const e₁ = (-13 - 7*√6)/3
const e₂ = (-13 + 7*√6)/3
const e₃ = -1/3

#-------------------------------------------------------------------------------
#support functions

function ∂F∂y(F::T, x, y, param, f₀, h, ϵ)::Float64 where {T}
    #try a regular old finite difference
    ∂y = ϵ*h
    if ∂y > sqrt(eps(y))*1e-2
        return (F(x, y + ∂y, param) - f₀)/∂y
    else
        return derivative(y′ -> F(x, y′, param), y)
    end
end

function Jacobian(h::Float64, dfdy::Float64)::SMatrix{3,3,Float64,9}
    #temporary
    q::Float64 = h*dfdy
    #column-major storage   
    SMatrix{3,3,Float64,9}(
        1.0 - a₁₁*q,
        -a₂₁*q,
        -a₃₁*q,
        -a₁₂*q,
        1.0 - a₂₂*q,
        -a₃₂*q,
        -a₁₃*q,
        -a₂₃*q,
        1.0 - a₃₃*q
    )
end

function xinit(x::Float64, h::Float64)::NTuple{3,Float64}
    x₁ = x + h*c₁
    x₂ = x + h*c₂
    x₃ = x + h*c₃
    return x₁, x₂, x₃
end

function hinit(x₀::Float64,
               xₙ::Float64,
               f::Float64,
               atol::Float64,
               rtol::Float64)::Float64
    x = max(abs(x₀), abs(xₙ))
    d = (1/x)^6 + abs(f)^6
    h = ((atol + rtol)/d)^(1/6)
    return min(h, xₙ - x₀)
end

#-------------------------------------------------------------------------------
# wrappers, basically

"""
    radau(F, y₀, x₀, xₙ, param=nothing; rtol=1e-6, atol=1e-6, facmax=100, facmin=0.01, κ=1e-3, ϵ=0.25, maxnwt=7, maxstp=1000000)

Solve a stiff, scalar ODE, returning only the end value of `y`

* `F`: ordinary differential equation in the form F(x, y, param), returning dy/dx
* `y₀`: initial `y` value
* `x₀`: initial `x` value
* `xₙ`: end point of integration
* `param`: extra parameters passed through to `F`, can be `Any` type, `nothing` by default
* `rtol`: relative error tolerance
* `atol`: absolute error tolerance
* `facmax`: maximum fractional increase in step size from one step to the next
* `facmin`: minimum fractional decrease in step size from one step to the next
* `κ`: Newton stopping tolerance
* `ϵ`: finite difference width as fraction of step size
* `maxnwt`: maximum Newton iterations before step size reduction
* `maxstp`: maximum number of step attemps before error
"""
function radau(F::T,
               y₀::Real,
               x₀::Real,
               xₙ::Real,
               param=nothing;
               kwargs...
               )::Float64 where {T}
    radau!((), (), F, y₀, x₀, xₙ, param; kwargs...)
end

"""
    radau(F, y₀, x₀, xₙ, nout, param=nothing; rtol=1e-6, atol=1e-6, facmax=100, facmin=0.01, κ=1e-3, ϵ=0.25, maxnwt=7, maxstp=1000000)

Solve a stiff, scalar ODE, returning evenly spaced samples between `x₀` and `xₙ`

* `F`: ordinary differential equation in the form F(x, y, param), returning dy/dx
* `y₀`: initial `y` value
* `x₀`: initial `x` value
* `xₙ`: end point of integration
* `nout`: number of evenly spaced steps to return
* `param`: extra parameters passed through to `F`, can be `Any` type, `nothing` by default
* `rtol`: relative error tolerance
* `atol`: absolute error tolerance
* `facmax`: maximum fractional increase in step size from one step to the next
* `facmin`: minimum fractional decrease in step size from one step to the next
* `κ`: Newton stopping tolerance
* `ϵ`: finite difference width as fraction of step size
* `maxnwt`: maximum Newton iterations before step size reduction
* `maxstp`: maximum number of step attemps before error
"""
function radau(F::T,
               y₀::Real,
               x₀::Real,
               xₙ::Real,
               nout::Int,
               param=nothing;
               kwargs...
               )::NTuple{2,Vector{Float64}} where {T}
    @assert nout > 1 "number of output points should be greater than 1"
    #evenly spaced output points
    x = LinRange(x₀, xₙ, nout)
    y = zeros(Float64, nout)
    #integrate!
    radau!(y, x, F, y₀, x₀, xₙ, param; kwargs...)
    return x, y
end

#-------------------------------------------------------------------------------
#main function

"""
    radau!(yout, xout, F, y₀, x₀, xₙ, nout, param=nothing; rtol=1e-6, atol=1e-6, facmax=100, facmin=0.01, κ=1e-3, ϵ=0.25, maxnwt=7, maxstp=1000000)

Solve a stiff, scalar ODE, sampling the solution in-place at the coordinates determined by `xout`

* `yout`: vector for output solution values
* `xout`: vector of output solution coordinates
* `F`: ordinary differential equation in the form F(x, y, param), returning dy/dx
* `y₀`: initial `y` value
* `x₀`: initial `x` value
* `xₙ`: end point of integration
* `param`: extra parameters passed through to `F`, can be `Any` type, `nothing` by default
* `rtol`: relative error tolerance
* `atol`: absolute error tolerance
* `facmax`: maximum fractional increase in step size from one step to the next
* `facmin`: minimum fractional decrease in step size from one step to the next
* `κ`: Newton stopping tolerance
* `ϵ`: finite difference width as fraction of step size
* `maxnwt`: maximum Newton iterations before step size reduction
* `maxstp`: maximum number of step attemps before error
"""
function radau!(yout::Union{AbstractVector{<:Real},Tuple}, #output values to fill
                xout::Union{AbstractVector{<:Real},Tuple}, #output coordinates
                F::T, #differential equation dy/dx = F(x,y,param)
                y₀::Real, #initial value
                x₀::Real, #initial coordinate
                xₙ::Real, #stopping coordinate
                param=nothing; #extra parameter(s) of whatever type
                rtol::Float64=1e-6, #absolute component of error tolerance
                atol::Float64=1e-6, #relative component of error tolerance
                facmax::Float64=100.0, #maximum step size increase factor
                facmin::Float64=0.01, #minimum step size decrease factor
                κ::Float64=1e-3, #Newton stopping tuner
                ϵ::Float64=0.25, #finite diff fraction of step size
                maxnwt::Int=7, #max Newton iterations before h reduction
                maxstp::Int=1000000 #maximum number of steps before error
                ) where {T}
    #check direction
    @assert xₙ >= x₀
    #output points
    nout = length(xout)
    jout = 1 #tracking index
    #uniform types
    x = convert(Float64, x₀)
    y = convert(Float64, y₀)
    xₙ = convert(Float64, xₙ)
    #initial function eval at x0
    f₀ = F(x, y, param)
    #initial step size selection
    h₁ = hinit(x, xₙ, f₀, atol, rtol)
    h₂ = hinit(x, xₙ, F(x + h₁, y + h₁*f₀, param), atol, rtol)
    h = min(h₁, h₂)
    #allocation, essentially, to keep f₃ in scope
    f₃ = 0.0
    #counter
    nstp = 0
    while x < xₙ
        #don't overshoot the end of the integration interval
        h = min(h, xₙ - x)
        #finite diff ∂F/∂y, precision not necessary in practice, can also hurt
        ∂ = ∂F∂y(F, x, y, param, f₀, h, ϵ)
        #jacobian matrix
        J = Jacobian(h, ∂)
        #x coordinates for function evaluations inside interval
        x₁, x₂, x₃ = xinit(x, h)
        #initial newton guesses, extrapolation appears to make things slower
        z₁, z₂, z₃ = 0.0, 0.0, 0.0
        #newton iterations
        ΔZ = Inf # ∞ norm of changes to solution
        η = κ*(rtol*abs(y) + atol) #termination threshold
        nnwt::Int64 = 0
        nfail::Int64 = 0
        while ΔZ > η
            if nnwt == maxnwt
                #count the convergence failure
                nfail += 1
                #cut off unending disasters
                if nfail == 10
                    error("repeated Newton convergence failures ($nfail) in Radau")
                end
                #steeply reduce step size
                h /= 10.0
                #wipe the iteration counter
                nnwt = 0
                #reinitialize with the new step size
                J = Jacobian(h, ∂F∂y(F, x, y, param, f₀, h, ϵ))
                x₁, x₂, x₃ = xinit(x, h)
                z₁, z₂, z₃ = 0.0, 0.0, 0.0
            end
            #function evaluations
            f₁ = F(x₁, y + z₁, param)
            f₂ = F(x₂, y + z₂, param)
            f₃ = F(x₃, y + z₃, param)
            #newton system evaluation β = (h * Af) - z
            β = SVector{3,Float64}(
                h*(a₁₁*f₁ + a₁₂*f₂ + a₁₃*f₃) - z₁,
                h*(a₂₁*f₁ + a₂₂*f₂ + a₂₃*f₃) - z₂,
                h*(a₃₁*f₁ + a₃₂*f₂ + a₃₃*f₃) - z₃
            )
            #solve the linear system J*δ = β
            δ₁, δ₂, δ₃ = J\β
            #update
            z₁ += δ₁
            z₂ += δ₂
            z₃ += δ₃
            #norm of updates
            ΔZ = abs(δ₁) + abs(δ₂) + abs(δ₃)
            #count
            nnwt += 1
        end
        #scaled error estimate
        ze = (f₀*h + z₁*e₁ + z₂*e₂ + z₃*e₃)/5
        sc = rtol*abs(max(y, y + z₃)) + atol
        err = abs(ze)/sc
        #accept the step?
        if err < 1.0
            #reserve previous point for interpolation
            xₚ = x
            yₚ = y
            #advance the solution
            x += h
            y += z₃
            #dense output
            @inbounds begin 
                if (jout <= nout) && ((x > xout[jout]) | (x ≈ xout[jout]))
                    #set up cubic Hermite
                    u = h*f₀
                    v = h*f₃
                    H₃ =  2yₚ  + u - 2y + v
                    H₂ = -3yₚ - 2u + 3y - v
                    #interpolate at all points that have been passed or met
                    while (jout <= nout) && (x >= xout[jout])
                        ξ = (xout[jout] - xₚ)/h
                        yout[jout] = yₚ + ξ*(u + ξ*(H₂ + ξ*H₃))
                        jout += 1
                    end
                end
            end
            #f₃ is now at the beginning of the next interval
            f₀ = f₃
        end
        #count
        nstp += 1
        if nstp == maxstp
            error("maximum number of steps/attempts ($maxstp) reached in radau @ nx=$x y=$y h=$h")
        end
        #safety factor, loosely dependent on number of newton iterations
        facsaf = 0.9*(maxnwt + 1)/(maxnwt + nnwt)
        #step size selection, double sqrt faster than ^(1/4)
        h *= min(facmax, max(facmin, facsaf*sqrt(sqrt(1/err))))
    end
    return y
end

end

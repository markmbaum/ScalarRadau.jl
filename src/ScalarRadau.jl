module ScalarRadau

using StaticArrays: SMatrix, SVector

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

function ∂F∂y(F::T, x, y, param, f₀, h, ϵ) where {T}
    #don't use a step size that risks roundoff error
    ∂y = max(ϵ*h, sqrt(eps(y)))
    #compute a regular old forward diff
    (F(x, y + ∂y, param) - f₀)/∂y
end

function Jacobian(h, ∂)::SMatrix{3,3}
    #temporary
    q = h*∂
    #column-major storage   
    SMatrix{3,3}(
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

function xinit(x, h)
    x₁ = x + h*c₁
    x₂ = x + h*c₂
    x₃ = x + h*c₃
    return x₁, x₂, x₃
end

function hinit(x₀, xₙ, f, atol, rtol)
    x = max(abs(x₀), abs(xₙ))
    d = (1/x)^6 + abs(f)^6
    h = ((atol + rtol)/d)^(1/6)
    return min(h, xₙ - x₀)
end

#-------------------------------------------------------------------------------
# wrappers

function radau(F::U,
               y₀::T,
               x₀::Real,
               xₙ::Real,
               param=nothing;
               kwargs...
               ) where {U,T<:AbstractFloat}
    radau!((), (), F, y₀, x₀, xₙ, param; kwargs...)
end

function radau(F::U,
               y₀::T,
               x₀::Real,
               xₙ::Real,
               nout::Int,
               param=nothing;
               kwargs...
               ) where {U,T<:AbstractFloat}
    @assert nout > 1 "number of output points should be greater than 1"
    #evenly spaced output points
    x = range(x₀, xₙ, length=nout)
    #space for results
    y = zeros(T, nout)
    #integrate!
    radau!(y, x, F, y₀, x₀, xₙ, param; kwargs...)
    return x, y
end

#-------------------------------------------------------------------------------
#main function

function radau!(yout::Union{AbstractVector{<:T},Tuple{}}, #output values to fill
                xout::Union{AbstractVector{<:Real},Tuple{}}, #output coordinates
                F::U, #differential equation dy/dx = F(x,y,param)
                y₀::T, #initial value
                x₀::Real, #initial coordinate
                xₙ::Real, #stopping coordinate
                param=nothing; #extra parameter(s) of whatever type
                rtol::Real=1e-6, #relative component of error tolerance
                atol::Real=1e-6, #absolute component of error tolerance
                facmax::Real=100.0, #maximum step size increase factor
                facmin::Real=0.01, #minimum step size decrease factor
                κ::Real=1e-3, #Newton stopping tuner
                ϵ::Real=0.25, #finite diff fraction of step size
                maxnewt::Real=7, #max Newton iterations before h reduction
                maxstep::Real=1000000, #maximum number of steps before error
                maxfail::Real=10 #maximum number of step failures before error
                ) where {T<:AbstractFloat,U}
    #basic checks
    @assert xₙ >= x₀
    @assert rtol < 1
    @assert facmax > 1
    @assert 0 < facmin < 1
    @assert 0 < κ < 1
    @assert 0 < ϵ < 1
    #initial function eval at x0
    f₀ = F(x₀, y₀, param)
    #set initial coordinates
    x, y, _ = promote(x₀, y₀, f₀)
    #output points
    nout = length(xout)
    jout = 1 #tracking index
    #initial step size selection
    h₁ = hinit(x, xₙ, f₀, atol, rtol)
    h₂ = hinit(x, xₙ, F(x + h₁, y + h₁*f₀, param), atol, rtol)
    h = min(h₁, h₂)
    #allocation, essentially, to keep f₃ in scope
    f₃ = 0.0
    #counter
    nstep = 0
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
        nnewt = 0
        nfail = 0
        while ΔZ > η
            if nnewt == maxnewt
                #count the convergence failure
                nfail += 1
                #cut off unending disasters
                (nfail == maxfail) && error("repeated Newton convergence failures ($nfail) in Radau")
                #steeply reduce step size
                h /= 10.0
                #wipe the iteration counter
                nnewt = 0
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
            β = SVector{3}(
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
            nnewt += 1
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
            @inbounds if (jout <= nout) && ((x > xout[jout]) | (x ≈ xout[jout]))
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
            #f₃ is now at the beginning of the next interval
            f₀ = f₃
        end
        #count
        nstep += 1
        if nstep == maxstep
            error("maximum number of steps/attempts ($maxstep) reached in radau @ nx=$x y=$y h=$h")
        end
        #safety factor, loosely dependent on number of newton iterations
        facsaf = 0.9*(maxnewt + 1)/(maxnewt + nnewt)
        #step size selection, double sqrt faster than ^(1/4)
        h *= min(facmax, max(facmin, facsaf*sqrt(sqrt(1/err))))
    end
    return y
end

end

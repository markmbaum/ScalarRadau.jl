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

function ∂f∂y(𝒇::T, x, y, param, f₀, h, ϵ) where {T}
    #don't use a step size that risks roundoff error
    ∂y = max(ϵ*h, sqrt(eps(y)))
    #compute a regular old forward diff
    (𝒇(x, y + ∂y, param) - f₀)/∂y
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

function radau(𝒇::F,
               y₀::Real,
               x₀::Real,
               xₙ::Real,
               param=nothing;
               kwargs...
               ) where {F}
    radau!((), (), 𝒇, y₀, x₀, xₙ, param; kwargs...)
end

function radau(𝒇::F,
               y₀::Real,
               x₀::Real,
               xₙ::Real,
               nout::Int,
               param=nothing;
               kwargs...
               ) where {F}
    @assert nout > 1 "number of output points should be greater than 1"
    #make y float
    y₀ = float(y₀)
    #evenly spaced output points
    x = LinRange(x₀, xₙ, nout)
    #space for results
    y = zeros(typeof(y₀), nout)
    #integrate!
    radau!(y, x, 𝒇, y₀, x₀, xₙ, param; kwargs...)
    return x, y
end

#-------------------------------------------------------------------------------
#main function

function radau!(yout::Union{AbstractVector{<:Real},Tuple{}},
                xout::Union{AbstractVector{<:Real},Tuple{}},
                𝒇::F,
                y₀::Real,
                x₀::Real,
                xₙ::Real,
                param=nothing;
                rtol::Real=1e-6,
                atol::Real=1e-6,
                facmax::Real=100.0,
                facmin::Real=0.01,
                κ::Real=1e-3,
                ϵ::Real=0.25,
                maxnewt::Int=7,
                maxstep::Int=1000000,
                maxfail::Int=10) where {F}
    #basic checks
    @assert xₙ >= x₀
    @assert rtol < 1
    @assert facmax > 1
    @assert 0 < facmin < 1
    @assert 0 < κ < 1
    @assert 0 < ϵ < 1
    #set initial coordinates
    x, y = float(x₀), float(y₀)
    #initial function eval at x0
    f₀ = 𝒇(x, y, param)
    #output points
    nout = length(xout)
    jout = 1 #tracking index
    #initial step size selection
    h₁ = hinit(x, xₙ, f₀, atol, rtol)
    h₂ = hinit(x, xₙ, 𝒇(x + h₁, y + h₁*f₀, param), atol, rtol)
    h = min(h₁, h₂)
    #allocation, essentially, to keep f₃ in scope
    f₃ = zero(f₀)
    #counter
    nstep = 0
    while x < xₙ
        #don't overshoot the end of the integration interval
        h = min(h, xₙ - x)
        #finite diff ∂f/∂y, precision not necessary in practice, can also hurt
        ∂ = ∂f∂y(𝒇, x, y, param, f₀, h, ϵ)
        #jacobian matrix
        J = Jacobian(h, ∂)
        #x coordinates for function evaluations inside interval
        x₁, x₂, x₃ = xinit(x, h)
        #initial newton guesses, extrapolation appears to make things slower
        z₁, z₂, z₃ = zero(y), zero(y), zero(y)
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
                J = Jacobian(h, ∂f∂y(𝒇, x, y, param, f₀, h, ϵ))
                x₁, x₂, x₃ = xinit(x, h)
                z₁, z₂, z₃ = zero(y), zero(y), zero(y)
            end
            #function evaluations
            f₁ = 𝒇(x₁, y + z₁, param)
            f₂ = 𝒇(x₂, y + z₂, param)
            f₃ = 𝒇(x₃, y + z₃, param)
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
                    yout[jout] += yₚ + ξ*(u + ξ*(H₂ + ξ*H₃))
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

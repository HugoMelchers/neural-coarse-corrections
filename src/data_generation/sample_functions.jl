import ProgressBars: ProgressBar
import Random: Xoshiro, rand, randn

function sample_functions(settings, description = nothing)
    N = settings["num"]
    seed = settings["seed"]
    rng = if seed !== nothing
        Xoshiro(seed)
    else
        Xoshiro()
    end

    is_helmholtz = settings["problem"] == "helmholtz"
    T = ifelse(is_helmholtz, ComplexF64, Float64)

    # Configuration for theta
    theta_min = settings["theta-min"]
    theta_max = settings["theta-max"]
    theta_degree = settings["theta-degree"]
    theta_segments = settings["theta-segments"]
    theta_basis = BSpline((0, 1), theta_segments, theta_degree)
    theta_dofs = length(theta_basis)

    # Configuration for f, g, eta
    rhs_basis = if settings["rhs-type"] == "grf"
        # GRF basis
        rhs_len = settings["rhs-grf-length"]
        GRF1D((0, 1), rhs_len)
    else
        # Bspline basis
        rhs_degree = settings["rhs-degree"]
        rhs_segments = settings["rhs-segments"]
        BSpline((0, 1), rhs_segments, rhs_degree)
    end
    # If we normalise the RHS vectors, compute the matrices needed for that
    Gi, r = if settings["rhs-type"] == "bspline-orthonormal"
        get_assembled_parts(settings["rhs-segments"], settings["rhs-degree"])
    else
        I, 1
    end
    rhs_dofs = length(rhs_basis)

    theta_coeffs = Array{Float64, 3}(undef, theta_dofs, theta_dofs, N)
    f_coeffs = Array{T, 3}(undef, rhs_dofs, rhs_dofs, N)
    g1_coeffs = Array{T, 2}(undef, rhs_dofs, N)
    g2_coeffs = Array{T, 2}(undef, rhs_dofs, N)
    eta1_coeffs = Array{T, 2}(undef, rhs_dofs, N)
    eta2_coeffs = Array{T, 2}(undef, rhs_dofs, N)
    f_scale = settings["f-scale"]
    g_scale = settings["g-scale"]
    eta_scale = settings["eta-scale"]

    vx_coeffs = nothing
    vy_coeffs = nothing
    with_v = settings["problem"] == "advection-diffusion"
    if with_v
        vx_coeffs = Array{Float64, 3}(undef, theta_dofs, theta_dofs, N)
        vy_coeffs = Array{Float64, 3}(undef, theta_dofs, theta_dofs, N)
    end

    #=
    Note: instead of generating all the coefficients as randn(a, N) or randn(a, b, N), I do this
    problem by problem in a for loop. While this is slower, it means that when generating e.g. 100
    problems with seed=1234, the first 10 problems will be the same as when generating 10 problems
    with the same seed.
    =#
    iter = if description === nothing
        1:N
    else
        ProgressBar(1:N)
    end
    for i in iter
        if is_helmholtz
            # For Helmholtz:
            # - theta is interpreted as the wave number k
            # - theta_max is the maximal wave number
            # - theta_min is the *relative* minimum wave number
            # So for example, theta_min = 0.5 and theta_max = 10 means that k <= 10, and within each
            # problem, k varies by at most a factor 2
            scale_i = theta_max * rand(rng)
            theta_coeffs[:, :, i] .=
                scale_i .* (theta_min .+ (1 - theta_min) .* rand(rng, theta_dofs, theta_dofs))
        else
            theta_coeffs[:, :, i] .=
                theta_min .+ (theta_max - theta_min) .* rand(rng, theta_dofs, theta_dofs)
        end
        f_coeffs[:, :, i] .= f_scale * Gi * randn(rng, T, rhs_dofs, rhs_dofs) * Gi
        g1_coeffs[:, i] .= (r * g_scale) * Gi * randn(rng, T, rhs_dofs)
        g2_coeffs[:, i] .= (r * g_scale) * Gi * randn(rng, T, rhs_dofs)
        eta1_coeffs[:, i] .= eta_scale * Gi * randn(rng, T, rhs_dofs)
        eta2_coeffs[:, i] .= eta_scale * Gi * randn(rng, T, rhs_dofs)

        if with_v
            # Velocity fields are always constant in space
            vx_i, vy_i = if settings["v-type"] == "randn"
                v_scale = settings["v-scale"]
                (v_scale * randn(rng), v_scale * randn(rng))
            elseif settings["v-type"] == "radial"
                v_min = settings["v-min"]
                v_max = settings["v-max"]
                dist = settings["v-dist"]
                v_norm = if dist == "log"
                    v_min_log = log(v_min)
                    v_max_log = log(v_max)
                    exp(v_min_log + rand(rng) * (v_max_log - v_min_log))
                elseif dist == "linear"
                    v_min + rand(rng) * (v_max - v_min)
                elseif dist == "sqrt"
                    # v_min is ignored
                    v_max * sqrt(rand(rng))
                end
                angle = rand(rng) * 2π
                v_norm .* sincos(angle)
            end
            vx_coeffs[:, :, i] .= vx_i
            vy_coeffs[:, :, i] .= vy_i
        end
    end

    fields = Dict(
        "theta" => TensorFunctions2D(theta_basis, theta_basis, theta_coeffs),
        "f" => TensorFunctions2D(rhs_basis, rhs_basis, f_coeffs),
        "left:g" => Functions1D(rhs_basis, g1_coeffs),
        "right:g" => Functions1D(rhs_basis, g2_coeffs),
        "bottom:eta" => Functions1D(rhs_basis, eta1_coeffs),
        "top:eta" => Functions1D(rhs_basis, eta2_coeffs),
    )
    if with_v
        fields["v_x"] = TensorFunctions2D(theta_basis, theta_basis, vx_coeffs)
        fields["v_y"] = TensorFunctions2D(theta_basis, theta_basis, vy_coeffs)
    end
    return fields
end

function get_assembled_parts(n_segments, degree)
    # To generate data so that the right-hand side vectors assembled by the NGO have a
    # well-conditioned distribution, I compute the assembled vector for each of the basis functions
    # used to express f, g, and eta. Then, I compute three matrices so that doing this change of
    # basis produces nice distributions.
    knots = range(0, 1, n_segments + 1)
    basis = BSpline((0, 1), n_segments, degree; sparse = true)
    nodes, weights = repeated_gauss_legendre(knots, degree + 2)
    W = spdiagm(weights)
    vals = basis(nodes)

    # For f: K is the kronecker product kron(G², G²)
    # For eta: both bottom and top have covariance G²
    G = Array(vals' * W * vals)
    # For g, we integrate (∇φ⋅n⃗, g). In a tensor product setting, this is (∇φᵢⱼ(0, y)⋅n⃗, g(0, y)) =
    # (-φᵢ'(x)φⱼ(y), g(0, y)) for the left boundary and similar for the right.
    # So the covariance matrix will be of the form kron(G², vv^T) where v is the derivative of the
    # basis at the endpoint
    derivs = derivative(basis, [0.0])

    return inv(G), 1 / sqrt(sum(abs2, derivs))
end

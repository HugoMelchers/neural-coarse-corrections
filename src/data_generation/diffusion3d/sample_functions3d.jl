import ProgressBars: ProgressBar
import Random: Xoshiro, rand, randn

#=
3D equivalent of `sample_functions` (see `../sample_functions.jl`), for the diffusion equation only.

θ and f are tensor products of three 1D bases, and the boundary data g and η are tensor products of
two 1D bases on each face. The settings are interpreted in the same way as in 2D.
=#
function sample_functions_3d(settings, description = nothing)
    @assert settings["problem"] == "diffusion3d"
    N = settings["num"]
    seed = settings["seed"]
    rng = if seed !== nothing
        Xoshiro(seed)
    else
        Xoshiro()
    end

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

    theta_coeffs = Array{Float64, 4}(undef, theta_dofs, theta_dofs, theta_dofs, N)
    f_coeffs = Array{Float64, 4}(undef, rhs_dofs, rhs_dofs, rhs_dofs, N)
    g_coeffs =
        Dict(part => Array{Float64, 3}(undef, rhs_dofs, rhs_dofs, N) for part in DIRICHLET_FACES_3D)
    eta_coeffs =
        Dict(part => Array{Float64, 3}(undef, rhs_dofs, rhs_dofs, N) for part in NEUMANN_FACES_3D)
    f_scale = settings["f-scale"]
    g_scale = settings["g-scale"]
    eta_scale = settings["eta-scale"]

    # As in 2D, sample problem by problem so that a fixed seed gives the same first problems
    # regardless of N
    iter = if description === nothing
        1:N
    else
        ProgressBar(1:N)
    end
    for i in iter
        theta_coeffs[:, :, :, i] .=
            theta_min .+ (theta_max - theta_min) .* rand(rng, theta_dofs, theta_dofs, theta_dofs)
        f_coeffs[:, :, :, i] .=
            f_scale .* _mode_products(Gi, Gi, Gi, randn(rng, rhs_dofs, rhs_dofs, rhs_dofs))
        for part in DIRICHLET_FACES_3D
            g_coeffs[part][:, :, i] .= (r * g_scale) * Gi * randn(rng, rhs_dofs, rhs_dofs) * Gi
        end
        for part in NEUMANN_FACES_3D
            eta_coeffs[part][:, :, i] .= eta_scale * Gi * randn(rng, rhs_dofs, rhs_dofs) * Gi
        end
    end

    fields = Dict{String, Any}(
        "theta" => TensorFunctions3D(theta_basis, theta_basis, theta_basis, theta_coeffs),
        "f" => TensorFunctions3D(rhs_basis, rhs_basis, rhs_basis, f_coeffs),
    )
    for part in DIRICHLET_FACES_3D
        fields["$part:g"] = TensorFunctions2D(rhs_basis, rhs_basis, g_coeffs[part])
    end
    for part in NEUMANN_FACES_3D
        fields["$part:eta"] = TensorFunctions2D(rhs_basis, rhs_basis, eta_coeffs[part])
    end
    return fields
end

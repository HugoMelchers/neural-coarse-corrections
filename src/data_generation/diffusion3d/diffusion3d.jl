import LinearAlgebra: isposdef, Symmetric, cholesky
import ProgressBars: ProgressBar, set_description

#=
3D equivalent of `../diffusion.jl`, on the unit cube [0, 1]³. The boundary conditions are

- Dirichlet (imposed weakly with Nitsche's method) on the faces x = 0 and x = 1 (:left, :right)
- Neumann on the faces y = 0 and y = 1 (:front, :back) and z = 0 and z = 1 (:bottom, :top)

which reduces to the 2D setting when the solution does not depend on z.

The fields used are
- "theta", "f": TensorFunctions3D
- "left:g", "right:g": TensorFunctions2D in (y, z)
- "front:eta", "back:eta": TensorFunctions2D in (x, z)
- "bottom:eta", "top:eta": TensorFunctions2D in (x, y)
=#

const DIRICHLET_FACES_3D = (:left, :right)
const NEUMANN_FACES_3D = (:front, :back, :bottom, :top)

function solve_diffusion_3d(settings, fields)
    # As in `solve`, parallelise over problems rather than inside BLAS
    if Threads.nthreads() == 1
        @warn "Running on 1 thread. Launch julia with `--threads=<N>` to run on multiple threads"
    end
    LinearAlgebra.BLAS.set_num_threads(1)

    N = settings["num"]
    u_degree = settings["u-degree"]
    u_segments = settings["u-segments"]
    theta_degree = settings["theta-degree"]
    rhs_degree = if settings["rhs-type"] == "grf"
        u_degree
    else
        settings["rhs-degree"]
    end
    split = get(settings, "split", false)
    #=
    Tensor-product Gauss quadrature is exact if it is exact in each direction separately. In each
    direction, the stiffness integrand θ ∇w⋅∇u has degree at most 2p + deg(θ), and the load
    integrand w f has degree at most p + deg(f). (The 2D code uses 2p + deg(θ) + deg(f), which is
    more than necessary. In 3D, the number of quadrature nodes per element matters a lot more.)
    =#
    max_degree = 2 * u_degree + max(theta_degree, rhs_degree)

    u_basis_1d = BSpline((0, 1), u_segments, u_degree; sparse = true)
    n = length(u_basis_1d)
    coeffs_u = Array{Float64, 4}(undef, n, n, n, N)
    coeffs_u_f, coeffs_u_g, coeffs_u_eta = if split
        similar(coeffs_u), similar(coeffs_u), similar(coeffs_u)
    else
        nothing, nothing, nothing
    end

    K_stencil = sparsity_pattern_3d(u_basis_1d)

    progressbar = ProgressBar(1:N)
    set_description(progressbar, "Solving")

    Threads.@threads for i in progressbar
        fields_i = Dict([k => view(v, i) for (k, v) in pairs(fields)])
        K, b_f, b_g, b_eta =
            assemble_diffusion_3d(settings, fields_i, u_basis_1d, max_degree, K_stencil)

        if settings["check"]
            # Check that K is symmetric and positive definite
            relerr = maximum(abs, K .- K') / maximum(abs, K)
            @assert relerr < 1e-12
            @assert isposdef(Symmetric(K + K'))
        end

        # K is symmetric positive definite, so use a (sparse) Cholesky factorisation instead of LU
        F = cholesky(Symmetric(K))
        if split
            u_f = reshape(F \ b_f, n, n, n)
            u_g = reshape(F \ b_g, n, n, n)
            u_eta = reshape(F \ b_eta, n, n, n)
            coeffs_u_f[:, :, :, i] .= u_f
            coeffs_u_g[:, :, :, i] .= u_g
            coeffs_u_eta[:, :, :, i] .= u_eta
            coeffs_u[:, :, :, i] .= u_f .+ u_g .+ u_eta
        else
            b = b_f .+ b_g .+ b_eta
            coeffs_u[:, :, :, i] .= reshape(F \ b, n, n, n)
        end
    end
    if split
        fields["u_f"] = TensorFunctions3D(u_basis_1d, u_basis_1d, u_basis_1d, coeffs_u_f)
        fields["u_g"] = TensorFunctions3D(u_basis_1d, u_basis_1d, u_basis_1d, coeffs_u_g)
        fields["u_eta"] = TensorFunctions3D(u_basis_1d, u_basis_1d, u_basis_1d, coeffs_u_eta)
    end
    fields["u"] = TensorFunctions3D(u_basis_1d, u_basis_1d, u_basis_1d, coeffs_u)
    nothing
end

function assemble_diffusion_3d(settings, fields, basis, max_degree, stencil)
    #=
    Same weak form as in 2D:

        ⟨θ∇w, ∇u⟩_Ω - ⟨θ dw/dn, u⟩_ΓD - ⟨θw, du/dn⟩_ΓD + γ⟨θ w, u⟩_ΓD
        =
        ⟨w, f⟩_Ω - ⟨dw/dn, g⟩_ΓD + γ⟨w, g⟩_ΓD + ⟨w, η⟩_ΓN

    Note that θ appears in the Dirichlet terms on the left but not on the right, so this
    (intentionally) imposes the Dirichlet condition θu = g rather than u = g.

    See `assembly3d.jl` for how the element integrals are computed.
    =#
    K = similar(stencil)
    K .= 0.0
    b_f = zeros(size(K, 1))
    b_g = zeros(size(K, 1))
    b_eta = zeros(size(K, 1))

    p = basis.degree
    m = p + 1
    quad = ceil(Int, (max_degree + 1) / 2)

    K_el = zeros(m, m, m, m, m, m)
    K_el_mat = reshape(K_el, m^3, m^3)
    b_el = zeros(m, m, m)
    b_el_vec = view(b_el, :)

    # Interior: ∫ θ ∇w⋅∇u dx and ∫ w f dx
    global_quad, elements = interior_3d(basis, quad)
    nodes = global_quad.nodes
    grid = nodes ⊗ nodes ⊗ nodes
    theta_global = fields["theta"](grid)
    f_global = fields["f"](grid)
    buffers = element_buffers(p, (quad, quad, quad))
    theta_W = zeros(quad, quad, quad)
    f_W = zeros(quad, quad, quad)
    for element in elements
        (; dof_ids, node_ids, weights, u, du) = element
        theta_W .= view(theta_global, node_ids...) .* weights
        f_W .= view(f_global, node_ids...) .* weights

        # ∫ ∇w⋅θ∇u dx = ∫ ∂₁w θ ∂₁u dx + ∫ ∂₂w θ ∂₂u dx + ∫ ∂₃w θ ∂₃u dx
        K_el .= 0.0
        add_element_matrix!(K_el, buffers, 1.0, du[1], du[1], u[2], u[2], u[3], u[3], theta_W)
        add_element_matrix!(K_el, buffers, 1.0, u[1], u[1], du[2], du[2], u[3], u[3], theta_W)
        add_element_matrix!(K_el, buffers, 1.0, u[1], u[1], u[2], u[2], du[3], du[3], theta_W)
        add_to_sparse!(K, dof_ids, K_el_mat)

        # ∫ w f dx
        b_el .= 0.0
        add_element_vector!(b_el, buffers, 1.0, u[1], u[2], u[3], f_W)
        b_f[dof_ids] .+= b_el_vec
    end

    # Dirichlet faces
    gamma = settings["gamma"]
    for part in DIRICHLET_FACES_3D
        face_data, faces = boundary_faces_3d(basis, quad, part)
        theta_face = fields["theta"](face_data.grid)
        g_face = reshape(fields["$part:g"](face_data.face_grid), face_data.size_3d)
        axis, _ = face_axis(part)
        nq = ntuple(d -> d == axis ? 1 : quad, 3)
        buffers = element_buffers(p, nq)
        theta_W = zeros(nq)
        g_W = zeros(nq)
        for face in faces
            (; dof_ids, node_ids, weights, u, u_dn) = face
            theta_W .= view(theta_face, node_ids...) .* weights
            g_W .= view(g_face, node_ids...) .* weights
            # Test/trial function factors with the normal derivative in direction `axis`
            u_n = Base.setindex(u, u_dn, axis)

            # - ∫ θ ∇w⋅n u dx - ∫ w θ ∇u⋅n dx (these two terms are each other's adjoint)
            # γ∫ w θ u dx (I divide this by 2 and then add the adjoint, as in 2D)
            K_el .= 0.0
            add_element_matrix!(
                K_el,
                buffers,
                -1.0,
                u_n[1],
                u[1],
                u_n[2],
                u[2],
                u_n[3],
                u[3],
                theta_W,
            )
            add_element_matrix!(
                K_el,
                buffers,
                gamma / 2,
                u[1],
                u[1],
                u[2],
                u[2],
                u[3],
                u[3],
                theta_W,
            )
            add_to_sparse!(K, dof_ids, K_el_mat)
            add_to_sparse!(K, dof_ids, K_el_mat')

            # γ∫ w g dx - ∫ ∇w⋅n g dx
            b_el .= 0.0
            add_element_vector!(b_el, buffers, gamma, u[1], u[2], u[3], g_W)
            add_element_vector!(b_el, buffers, -1.0, u_n[1], u_n[2], u_n[3], g_W)
            b_g[dof_ids] .+= b_el_vec
        end
    end

    # Neumann faces: ∫ w η dx
    for part in NEUMANN_FACES_3D
        face_data, faces = boundary_faces_3d(basis, quad, part)
        eta_face = reshape(fields["$part:eta"](face_data.face_grid), face_data.size_3d)
        axis, _ = face_axis(part)
        nq = ntuple(d -> d == axis ? 1 : quad, 3)
        buffers = element_buffers(p, nq)
        eta_W = zeros(nq)
        for face in faces
            (; dof_ids, node_ids, weights, u) = face
            eta_W .= view(eta_face, node_ids...) .* weights
            b_el .= 0.0
            add_element_vector!(b_el, buffers, 1.0, u[1], u[2], u[3], eta_W)
            b_eta[dof_ids] .+= b_el_vec
        end
    end

    K, b_f, b_g, b_eta
end

import LinearAlgebra: Diagonal, kron, isposdef, Symmetric, eigen
import SparseArrays: spdiagm
import ProgressBars: ProgressBar, set_description

function solve_advdiff(settings, fields)
    #=
    Solve the advection-diffusion equation. The full weak form is:

        ∫Ω (∇w, θ∇u) dx                           ( 1. diffusion)
        - ∫ΓD (w, θ∇u⋅n⃗) dσ                       ( 2. consistency term)
        - ∫ΓD (γθ∇w⋅n⃗, u) dσ                      ( 3. outflow boundary Dirichlet term)
        + ∫ΓD ((Cᵣ' θ / hᵣ) w, u) dσ              ( 4. penalty + stabilisation term)

        - ∫Ω (∇w, c⃗u) dx                          ( 5. advection)
        + ∫ΓN (w, u(c⃗⋅n⃗)₊) dσ                     ( 6. boundary term due to integration by parts)
        + ∫ΓD (w, c⃗⋅n⃗u) dσ                        ( 7. consistency term)
        - ∫Γᵢₙ (c⃗⋅n⃗w, u) dσ                       ( 8. inflow boundary Dirichlet term)

        + ∫Ω (τc⃗⋅∇w, -∇⋅(θ∇u) + c⃗⋅∇u) dx          ( 9. SUPG stabilisation)
    =
        ∫Ω (w, f) dx                              (10. volume forcing term)
        + ∫ΓN (w, η) dσ                           (11. Neumann forcing term)
        - ∫ΓD (γ∇w⋅n⃗, g) dσ                       (12. outflow boundary Dirichlet term)
        - ∫Γᵢₙ ((c⃗⋅n⃗)w / θ, g) dσ                 (13. inflow boundary Dirichlet term)
        + ∫ΓD ((Cᵣ' / hᵣ) w, g) dσ                (14. penalty + stabilisation term)

        + ∫Ω (τc⃗⋅∇w, f) dx                        (15. SUPG stabilisation)

    Note that the use of γ here conflicts with its use as the Nitsche stabilisation constant in the
    diffusion case. Because of that, settings["gamma"] still refers to the Nitsche stabilisation,
    while the value of γ in the above formulation is stored as settings["sign"] (since its value
    should be ±1).
    =#
    N = settings["num"]
    gamma = settings["gamma"]
    sign = settings["sign"]
    checkposdef = settings["check"]

    u_degree = settings["u-degree"]
    u_segments = settings["u-segments"]
    u_knots = range(0, 1, u_segments + 1)
    theta_degree = settings["theta-degree"]
    rhs_degree = if settings["rhs-type"] == "bspline"
        settings["rhs-degree"]
    else
        u_degree
    end
    split = get(settings, "split", false)
    max_degree = 2 * u_degree + theta_degree + rhs_degree

    u_basis_1d = BSpline((0, 1), u_segments, u_degree; sparse = true)
    nr_dofs_1d = length(u_basis_1d)

    coeffs_u = Array{Float64, 3}(undef, nr_dofs_1d, nr_dofs_1d, N)
    coeffs_u_f, coeffs_u_g, coeffs_u_eta = if split
        similar(coeffs_u), similar(coeffs_u), similar(coeffs_u)
    else
        nothing, nothing, nothing
    end

    K_stencil = sparsity_pattern(u_basis_1d)

    pb = ProgressBar(1:N)
    Threads.@threads for i in pb
        fields_i = Dict([k => view(v, i) for (k, v) in pairs(fields)])
        K, b_f, b_g, b_eta = assemble_advdiff(settings, fields_i, u_basis_1d, max_degree, K_stencil)

        if checkposdef
            @assert isposdef(Symmetric(K + K'))
        end

        if split
            u_f = reshape(K \ b_f, nr_dofs_1d, nr_dofs_1d)
            u_g = reshape(K \ b_g, nr_dofs_1d, nr_dofs_1d)
            u_eta = reshape(K \ b_eta, nr_dofs_1d, nr_dofs_1d)
            coeffs_u_f[:, :, i] .= u_f
            coeffs_u_g[:, :, i] .= u_g
            coeffs_u_eta[:, :, i] .= u_eta
            coeffs_u[:, :, i] .= u_f .+ u_g .+ u_eta
        else
            b = b_f .+ b_g .+ b_eta
            u = reshape(K \ b, nr_dofs_1d, nr_dofs_1d)
            coeffs_u[:, :, i] .= u
        end
    end

    # Store the results
    if split
        fields["u_f"] = TensorFunctions2D(u_basis_1d, u_basis_1d, coeffs_u_f)
        fields["u_g"] = TensorFunctions2D(u_basis_1d, u_basis_1d, coeffs_u_g)
        fields["u_eta"] = TensorFunctions2D(u_basis_1d, u_basis_1d, coeffs_u_eta)
    end
    fields["u"] = TensorFunctions2D(u_basis_1d, u_basis_1d, coeffs_u)
end

function assemble_advdiff(settings, fields, basis, max_degree, stencil)
    # Initialise an all-zero sparse matrix, but with the correct sparsity pattern stored. This makes
    # assembly much faster compared to starting with spzeros(...)
    K = similar(stencil)
    K .= 0.0
    b_f = zeros(size(K, 1))
    b_g = zeros(size(K, 1))
    b_eta = zeros(size(K, 1))

    sign = Float64(settings["sign"])
    gamma = Float64(settings["gamma"])

    p = basis.degree
    quad = ceil(Int, (max_degree + 1) / 2)

    # Allocate some buffers to reduce the number of allocations in the inner loop
    K_ij = zeros((p + 1)^2, (p + 1)^2)
    b_ij = zeros((p + 1)^2)
    tmp_K = zeros(quad^2, (p + 1)^2)
    tmp_b = zeros(quad^2)
    theta_W = zeros(quad^2)
    supg_mat_left = zeros(quad^2, (p + 1)^2)
    supg_mat_right = zeros(quad^2, (p + 1)^2)

    global_quad, elements = interior(basis, quad)
    (; nodes, weights) = global_quad
    theta_global = fields["theta"](nodes ⊗ nodes)
    theta_dx_global = derivative(fields["theta"], nodes ⊗ nodes, (1, 0))
    theta_dy_global = derivative(fields["theta"], nodes ⊗ nodes, (0, 1))
    f_global = fields["f"](nodes ⊗ nodes)
    v_x_global = fields["v_x"](nodes ⊗ nodes)
    v_y_global = fields["v_y"](nodes ⊗ nodes)
    for element in elements
        (; dof_ids, weights, u, u_dx, u_dy, u_lap, x_ids, y_ids, width, height) =
            element
        thetas = reshape(view(theta_global, x_ids, y_ids), :)
        thetas_dx = reshape(view(theta_dx_global, x_ids, y_ids), :)
        thetas_dy = reshape(view(theta_dy_global, x_ids, y_ids), :)
        v_x = reshape(view(v_x_global, x_ids, y_ids), :)
        v_y = reshape(view(v_y_global, x_ids, y_ids), :)
        f = reshape(view(f_global, x_ids, y_ids), :)

        # 1. ∫Ω (∇w, θ∇u) dx
        theta_W .= thetas .* weights
        tmp_K .= theta_W .* u_dx
        LinearAlgebra.BLAS.gemm!('T', 'N', true, u_dx, tmp_K, false, K_ij)
        tmp_K .= theta_W .* u_dy
        LinearAlgebra.BLAS.gemm!('T', 'N', true, u_dy, tmp_K, true, K_ij)
        # 5. -∫Ω (∇w, c⃗u) dx = -∫ ∂₁w c₁ u dx - ∫ ∂₂w c₂ u dx
        tmp_K .= v_x .* weights .* u
        LinearAlgebra.BLAS.gemm!('T', 'N', -1.0, u_dx, tmp_K, true, K_ij)
        tmp_K .= v_y .* weights .* u
        LinearAlgebra.BLAS.gemm!('T', 'N', -1.0, u_dy, tmp_K, true, K_ij)

        # 9. ∫Ω (τc⃗⋅∇w, -∇⋅(θ∇u) + c⃗⋅∇u) dx
        h = sqrt(width * height)
        th_avg = sum(theta_W) / h^2
        tmp_b .= weights .* hypot.(v_x, v_y)
        v_norm = sum(tmp_b) / h^2
        Pe = v_norm * h / 2th_avg
        tau = h / max(2v_norm, eps()) * min(1, Pe / 3p^2)
        supg_mat_left .= tau .* (v_x .* u_dx .+ v_y .* u_dy)
        supg_mat_right .=
            .-thetas .* u_lap .+ (v_x .- thetas_dx) .* u_dx .+ (v_y .- thetas_dy) .* u_dy

        tmp_K .= weights .* supg_mat_right
        LinearAlgebra.BLAS.gemm!('T', 'N', true, supg_mat_left, tmp_K, true, K_ij)

        tmp_b .= weights .* f
        # 10. ∫Ω (w, f) dx
        LinearAlgebra.BLAS.gemv!('T', true, u, tmp_b, false, b_ij)
        # 15. ∫Ω (τc⃗⋅∇w, f) dx
        LinearAlgebra.BLAS.gemv!('T', true, supg_mat_left, tmp_b, true, b_ij)

        K[dof_ids, dof_ids] .+= K_ij
        b_f[dof_ids] .+= b_ij
    end

    tmp_K = zeros(quad, (p + 1)^2)
    tmp_b = zeros(quad)

    left_data, left_faces = boundary_faces(basis, quad, :left)
    left_nodes = left_data.nodes
    g_left = fields["left:g"](left_nodes)
    theta_left = fields["theta"]([0.0] ⊗ left_nodes)
    v_x_left = fields["v_x"]([0.0] ⊗ left_nodes)

    right_data, right_faces = boundary_faces(basis, quad, :right)
    right_nodes = right_data.nodes
    g_right = fields["right:g"](right_nodes)
    theta_right = fields["theta"]([1.0] ⊗ right_nodes)
    v_x_right = fields["v_x"]([1.0] ⊗ right_nodes)
    for face in Iterators.Flatten((left_faces, right_faces))
        # 2. -∫ΓD (w, θ∇u⋅n⃗) dσ
        (; part, normal, dof_ids, weights, u, u_dn, p_ids) = face
        theta = view(part == :left ? theta_left : theta_right, p_ids)
        v_x = view(part == :left ? v_x_left : v_x_right, p_ids)
        g = view(part == :left ? g_left : g_right, p_ids)

        # 2. -∫ΓD (w, θ∇u⋅n⃗) dσ
        tmp_K .= weights .* theta .* u_dn
        LinearAlgebra.BLAS.gemm!('T', 'N', -1.0, u, tmp_K, false, K_ij)

        # 3. -∫ΓD (γθ∇w⋅n⃗, u) dσ
        tmp_K .= weights .* theta .* u
        LinearAlgebra.BLAS.gemm!('T', 'N', -sign, u_dn, tmp_K, true, K_ij)

        # 4. ∫ΓD ((Cᵣ' |θ| / hᵣ) w, u) dσ
        tmp_K .= weights .* theta .* u
        LinearAlgebra.BLAS.gemm!('T', 'N', gamma, u, tmp_K, true, K_ij)

        # 7. ∫ΓD (w, c⃗⋅n⃗u) dσ = ∫ w c₁n₁u dσ (since Dirichlet boundaries are left/right so n₂ = 0)
        tmp_K .= weights .* v_x .* u
        LinearAlgebra.BLAS.gemm!('T', 'N', normal, u, tmp_K, true, K_ij)

        # 8. -∫Γᵢₙ (c⃗⋅n⃗w, u) dσ = -∫Γᵢₙ c₁ n₁ w u dσ
        tmp_K .= weights .* min.(normal .* v_x, 0.0) .* u
        LinearAlgebra.BLAS.gemm!('T', 'N', -1.0, u, tmp_K, true, K_ij)

        # 12. - ∫ΓD (γ∇w⋅n⃗, g) dσ
        tmp_b .= weights .* g
        LinearAlgebra.BLAS.gemv!('T', -sign, u_dn, tmp_b, false, b_ij)

        # 13. - ∫Γᵢₙ (c⃗⋅n⃗w, g / θ) dσ
        tmp_b .= weights .* min.(normal .* v_x, 0.0) .* g ./ theta
        LinearAlgebra.BLAS.gemv!('T', -1.0, u, tmp_b, true, b_ij)

        # 14. + ∫ΓD ((Cᵣ' / hᵣ) w, g) dσ
        tmp_b .= weights .* g
        LinearAlgebra.BLAS.gemv!('T', gamma, u, tmp_b, true, b_ij)

        K[dof_ids, dof_ids] .+= K_ij
        b_g[dof_ids] .+= b_ij
    end

    bottom_data, bottom_faces = boundary_faces(basis, quad, :bottom)
    bottom_nodes = bottom_data.nodes
    eta_bottom = fields["bottom:eta"](bottom_nodes)
    v_y_bottom = fields["v_y"](bottom_nodes ⊗ [0.0])

    top_data, top_faces = boundary_faces(basis, quad, :top)
    top_nodes = top_data.nodes
    eta_top = fields["top:eta"](top_nodes)
    v_y_top = fields["v_y"](top_nodes ⊗ [0.0])

    for face in Iterators.Flatten((bottom_faces, top_faces))
        (; part, normal, dof_ids, nodes, weights, u, u_dn, p_ids) = face
        v_y = view(part == :bottom ? v_y_bottom : v_y_top, p_ids)
        eta = view(part == :bottom ? eta_bottom : eta_top, p_ids)

        # 6. ∫ΓN (w, u(c⃗⋅n⃗)₊) dσ
        tmp_K .= weights .* max.(normal .* v_y, 0.0) .* u
        LinearAlgebra.BLAS.gemm!('T', 'N', true, u, tmp_K, false, K_ij)

        # 11. + ∫ΓN (w, η) dσ
        tmp_b .= weights .* eta
        LinearAlgebra.BLAS.gemv!('T', true, u, tmp_b, false, b_ij)

        K[dof_ids, dof_ids] .+= K_ij
        b_eta[dof_ids] .+= b_ij
    end

    K, b_f, b_g, b_eta
end

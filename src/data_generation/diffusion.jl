import LinearAlgebra: Diagonal, kron, isposdef, Symmetric, eigen
import SparseArrays: spdiagm
import ProgressBars: ProgressBar, set_description
import Format: format, printfmtln

function solve_diffusion(settings, fields)
    N = settings["num"]
    # Generate trial basis
    u_degree = settings["u-degree"]
    u_segments = settings["u-segments"]
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

    #=
    Evaluate basis functions etc. on quadrature nodes

    Note: in the Julia code I use the ordering [x, y, batch], which in particular means that the
    basis functions of the FEM system are ordered as x-varies-first. In Kronecker products, this
    means that the y-term is the first argument.
    =#

    K_stencil = sparsity_pattern(u_basis_1d)

    progressbar = ProgressBar(1:N)
    set_description(progressbar, "Solving")

    Threads.@threads for i in progressbar
        fields_i = Dict([k => view(v, i) for (k, v) in pairs(fields)])
        K, b_f, b_g, b_eta =
            assemble_diffusion(settings, fields_i, u_basis_1d, max_degree, K_stencil)

        if settings["check"]
            # Check that K is symmetric and positive definite
            relerr = maximum(abs, K .- K') / maximum(abs, K)
            @assert relerr < 1e-12
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
    if split
        fields["u_f"] = TensorFunctions2D(u_basis_1d, u_basis_1d, coeffs_u_f)
        fields["u_g"] = TensorFunctions2D(u_basis_1d, u_basis_1d, coeffs_u_g)
        fields["u_eta"] = TensorFunctions2D(u_basis_1d, u_basis_1d, coeffs_u_eta)
    end
    fields["u"] = TensorFunctions2D(u_basis_1d, u_basis_1d, coeffs_u)
    nothing
end

function assemble_diffusion(settings, fields, basis, max_degree, stencil)
    #=
    Solve the diffusion equation. The weak form is

        ⟨θ∇w, ∇u⟩_Ω - ⟨θ dw/dn, u⟩_ΓD - ⟨θw, du/dn⟩_ΓD + γ⟨θ w, u⟩_ΓD
        =
        ⟨w, f⟩_Ω - ⟨dw/dn, g⟩ + γ⟨w, g⟩ + ⟨w, η⟩_ΓN

    Here, γ is the Nitsche stabilisation constant, which should be large enough that the resulting
    matrix is positive definite.
    =#
    K = similar(stencil)
    K .= 0.0
    b_f = zeros(size(K, 1))
    b_g = zeros(size(K, 1))
    b_eta = zeros(size(K, 1))

    p = basis.degree
    quad = ceil(Int, (max_degree + 1) / 2)

    K_ij = zeros((p + 1)^2, (p + 1)^2)
    b_ij = zeros((p + 1)^2)
    tmp_K = zeros(quad^2, (p + 1)^2)
    tmp_b = zeros(quad^2)
    theta_W = zeros(quad^2)

    global_quad, elements = interior(basis, quad)
    (; nodes, weights) = global_quad
    theta_global = fields["theta"](nodes ⊗ nodes)
    f_global = fields["f"](nodes ⊗ nodes)
    for element in elements
        (; dof_ids, weights, u, u_dx, u_dy, x_ids, y_ids) = element
        theta = reshape(view(theta_global, x_ids, y_ids), :)
        f = reshape(view(f_global, x_ids, y_ids), :)

        #  ∫ ∇w⋅θ∇u dx = ∫ ∂₁w θ ∂₁u dx + ∫ ∂₂ θ ∂₂u dx
        theta_W .= theta .* weights
        tmp_K .= theta_W .* u_dx
        LinearAlgebra.BLAS.gemm!('T', 'N', true, u_dx, tmp_K, false, K_ij)
        tmp_K .= theta_W .* u_dy
        LinearAlgebra.BLAS.gemm!('T', 'N', true, u_dy, tmp_K, true, K_ij)
        K[dof_ids, dof_ids] .+= K_ij

        # ∫ w f dx
        tmp_b .= weights .* f
        LinearAlgebra.BLAS.gemv!('T', true, u, tmp_b, false, b_ij)
        b_f[dof_ids] .+= b_ij
    end

    tmp_K = zeros(quad, (p + 1)^2)
    tmp_b = zeros(quad)

    gamma = settings["gamma"]
    left_quad, left_faces = boundary_faces(basis, quad, :left)
    left_nodes = left_quad.nodes
    theta_left = fields["theta"]([0.0] ⊗ left_nodes)
    g_left = fields["left:g"](left_nodes)

    right_quad, right_faces = boundary_faces(basis, quad, :right)
    right_nodes = right_quad.nodes
    theta_right = fields["theta"]([1.0] ⊗ right_nodes)
    g_right = fields["right:g"](right_nodes)

    for face in Iterators.Flatten((left_faces, right_faces))
        (; part, dof_ids, weights, u, u_dn, p_ids) = face
        theta = view(part == :left ? theta_left : theta_right, p_ids)
        g = view(part == :left ? g_left : g_right, p_ids)

        # - ∫ θ ∇w⋅n u dx - ∫ w θ ∇u⋅n dx (these two terms are each other's adjoint)
        tmp_K .= weights .* theta .* u
        LinearAlgebra.BLAS.gemm!('T', 'N', -1.0, u_dn, tmp_K, false, K_ij)
        # γ∫ w θ u dx (I divide this by 2 and then add the adjoint, to merge some operations with
        # the previous term)
        LinearAlgebra.BLAS.gemm!('T', 'N', gamma / 2, u, tmp_K, true, K_ij)
        K[dof_ids, dof_ids] .+= K_ij .+ K_ij'

        tmp_b .= weights .* g
        # γ∫ w g dx
        LinearAlgebra.BLAS.gemv!('T', gamma, u, tmp_b, false, b_ij)
        # - ∫ ∇w⋅n g dx
        LinearAlgebra.BLAS.gemv!('T', -1.0, u_dn, tmp_b, true, b_ij)
        b_g[dof_ids] .+= b_ij
    end

    bottom_quad, bottom_faces = boundary_faces(basis, quad, :bottom)
    bottom_nodes = bottom_quad.nodes
    eta_bottom = fields["bottom:eta"](bottom_nodes)

    top_data, top_faces = boundary_faces(basis, quad, :top)
    top_nodes = top_data.nodes
    eta_top = fields["top:eta"](top_nodes)
    for face in Iterators.Flatten((bottom_faces, top_faces))
        (; part, dof_ids, nodes, weights, u, p_ids) = face
        eta = view(part == :bottom ? eta_bottom : eta_top, p_ids)

        tmp_b .= weights .* eta
        # ∫ w η dx
        LinearAlgebra.BLAS.gemv!('T', true, u, tmp_b, false, b_ij)
        b_eta[dof_ids] .+= b_ij
    end

    K, b_f, b_g, b_eta
end

import LinearAlgebra: Diagonal, kron, isposdef, Symmetric, eigen
import SparseArrays: spdiagm
import ProgressBars: ProgressBar, set_description
import Format: format, printfmtln

function solve_helmholtz(settings, fields)
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
    coeffs_u = Array{ComplexF64, 3}(undef, nr_dofs_1d, nr_dofs_1d, N)

    K_stencil = sparsity_pattern(u_basis_1d)
    K0, Mb, is_boundary, is_interior = preassemble_helmholtz(settings, u_basis_1d, max_degree, K_stencil)
    if settings["check"]
        # Check that K0 and Mb are symmetric positive (semi-)definite
        relerr = maximum(abs, K0 .- K0') / maximum(abs, K0)
        @assert relerr < 1e-12
        @assert isposdef(Symmetric(K0 + K0' + 1e-15 * I))

        relerr = maximum(abs, Mb .- Mb') / maximum(abs, Mb)
        @assert relerr < 1e-12
        @assert isposdef(Symmetric(Mb + Mb'))
    end

    progressbar = ProgressBar(1:N)
    set_description(progressbar, "Solving")

    Threads.@threads for i in progressbar
        fields_i = Dict([k => view(v, i) for (k, v) in pairs(fields)])
        K, b_f, b_eta, b_g =
            assemble_helmholtz(settings, fields_i, u_basis_1d, max_degree, K_stencil, K0, Mb, is_boundary)

        if settings["check"]
            # Check that K is symmetric (but not Hermitian)
            relerr = maximum(abs, K .- transpose(K)) / maximum(abs, K)
            @assert relerr < 1e-12
        end

        # Solve for the boundaries, substitute those values and solve for the interior
        u = Array{ComplexF64}(undef, nr_dofs_1d^2)
        u[is_boundary] = Mb \ b_g
        b_f[is_interior] .-= K[is_interior, is_boundary] * u[is_boundary]
        u[is_boundary] .= u[is_boundary]
        b_f .+= b_eta
        u[is_interior] .= K[is_interior, is_interior] \ b_f[is_interior]
        coeffs_u[:, :, i] .= reshape(u, nr_dofs_1d, nr_dofs_1d)
    end
    fields["u"] = TensorFunctions2D(u_basis_1d, u_basis_1d, coeffs_u)
    nothing
end

function preassemble_helmholtz(settings, basis, max_degree, stencil)
    # Return some reusable assembled matrices:
    # - The stiffness matrix K0 = ⟨∇w, ∇u⟩_Ω, i.e. the pure diffusion part
    # - The boundary mass matrix M = ⟨w, u⟩_∂Ω, which is used when imposing the boundary conditions
    # - A binary array storing which coefficients correspond to basis functions on the boundary

    K0 = similar(stencil, ComplexF64)
    K0 .= 0.0

    p = basis.degree
    quad = ceil(Int, (max_degree + 1) / 2)

    # Assemble K0
    K_ij = zeros((p + 1)^2, (p + 1)^2)
    tmp_K = zeros(quad^2, (p + 1)^2)
    global_quad, elements = interior(basis, quad)
    (; nodes, weights) = global_quad
    for element in elements
        (; dof_ids, weights, u, u_dx, u_dy, x_ids, y_ids) = element

        #  ∫ ∇w⋅∇u dx = ∫ ∂₁w ∂₁u dx + ∫ ∂₂w ∂₂u dx
        tmp_K .= weights .* u_dx
        LinearAlgebra.BLAS.gemm!('T', 'N', true, u_dx, tmp_K, false, K_ij)
        tmp_K .= weights .* u_dy
        LinearAlgebra.BLAS.gemm!('T', 'N', true, u_dy, tmp_K, true, K_ij)
        K0[dof_ids, dof_ids] .+= K_ij
    end

    # Assemble Mb
    Mb = spzeros(size(K0))
    Mb .= 0.0
    tmp_M = zeros(quad, (p + 1)^2)
    left_quad, left_faces = boundary_faces(basis, quad, :left)
    right_quad, right_faces = boundary_faces(basis, quad, :right)
    left_nodes = left_quad.nodes
    right_nodes = right_quad.nodes
    for face in Iterators.flatten((left_faces, right_faces))
        (; dof_ids, weights, u) = face
        #  ∫ w⋅u dx
        tmp_M .= weights .* u
        LinearAlgebra.BLAS.gemm!('T', 'N', true, u, tmp_M, false, K_ij)
        Mb[dof_ids, dof_ids] .+= K_ij
    end
    is_boundary = Array(diag(Mb) .> 0)
    is_interior = .!is_boundary
    return K0, Mb[is_boundary, is_boundary], is_boundary, is_interior
end

function assemble_helmholtz(settings, fields, basis, max_degree, stencil, K0, Mb, is_boundary)
    K = copy(K0)
    b_f = zeros(ComplexF64, size(K, 1))
    b_g = zeros(ComplexF64, size(K, 1))
    b_eta = zeros(ComplexF64, size(K, 1))

    p = basis.degree
    quad = ceil(Int, (max_degree + 1) / 2)

    K_ij = zeros(ComplexF64, (p + 1)^2, (p + 1)^2)
    b_ij = zeros(ComplexF64, (p + 1)^2)
    tmp_K = zeros(ComplexF64, quad^2, (p + 1)^2)
    tmp_b = zeros(ComplexF64, quad^2)
    theta_W = zeros(ComplexF64, quad^2)

    # Step 1: formulate the linear system for the boundary conditions
    # The matrix is already pre-computed, only need to assemble the right-hand-side vector
    tmp_g = zeros(ComplexF64, quad)
    left_quad, left_faces = boundary_faces(basis, quad, :left)
    right_quad, right_faces = boundary_faces(basis, quad, :right)
    left_nodes = left_quad.nodes
    right_nodes = right_quad.nodes
    g_left = fields["left:g"](left_nodes)
    g_right = fields["right:g"](right_nodes)
    for face in Iterators.flatten((left_faces, right_faces))
        (; part, dof_ids, nodes, weights, u, p_ids) = face
        g_part = if part == :left
            g_left
        elseif part == :right
            g_right
        end
        gs = view(g_part, p_ids)
        tmp_g .= weights .* gs
        LinearAlgebra.mul!(b_ij, u', tmp_g, true, false)
        b_g[dof_ids] .+= b_ij
    end

    # Step 2: assemble the ⟨w, k²u⟩_Ω part
    global_quad, elements = interior(basis, quad)
    (; nodes, weights) = global_quad
    theta_global = fields["theta"](nodes ⊗ nodes)
    f_global = fields["f"](nodes ⊗ nodes)
    for element in elements
        (; dof_ids, weights, u, u_dx, u_dy, x_ids, y_ids) = element
        theta = reshape(view(theta_global, x_ids, y_ids), :)
        f = reshape(view(f_global, x_ids, y_ids), :)

        #  ∫ w θ² u dx
        theta_W .= theta .^ 2 .* weights
        tmp_K .= theta_W .* u
        LinearAlgebra.mul!(K_ij, u', tmp_K, true, false)
        K[dof_ids, dof_ids] .-= K_ij

        # ∫ w f dx
        tmp_b .= weights .* f
        LinearAlgebra.mul!(b_ij, u', tmp_b, true, false)
        b_f[dof_ids] .+= b_ij
    end

    # Step 3: assemble the impedance boundary part
    tmp_eta = zeros(ComplexF64, quad)
    theta_W = zeros(ComplexF64, quad)
    tmp_K = zeros(ComplexF64, quad, (p + 1)^2)
    bottom_quad, bottom_faces = boundary_faces(basis, quad, :bottom)
    top_quad, top_faces = boundary_faces(basis, quad, :top)
    bottom_nodes = bottom_quad.nodes
    top_nodes = top_quad.nodes
    eta_bottom = fields["bottom:eta"](bottom_nodes)
    eta_top = fields["top:eta"](top_nodes)
    theta_bottom = fields["theta"](bottom_nodes ⊗ [0.0])
    theta_top = fields["theta"](top_nodes ⊗ [1.0])
    for face in Iterators.flatten((bottom_faces, top_faces))
        (; part, dof_ids, nodes, weights, u, p_ids) = face
        eta_part, theta_part = if part == :bottom
            eta_bottom, theta_bottom
        elseif part == :top
            eta_top, theta_top
        end
        theta_part = view(theta_part, p_ids)
        etas = view(eta_part, p_ids)
        
        theta_W .= theta_part .* weights
        tmp_K .= theta_W .* u
        LinearAlgebra.mul!(K_ij, u', tmp_K, true, false)
        K[dof_ids, dof_ids] .-= 1im .* K_ij

        tmp_eta .= weights .* etas
        LinearAlgebra.mul!(b_ij, u', tmp_eta, true, false)
        b_eta[dof_ids] .+= b_ij
    end

    return K, b_f, b_eta, view(b_g, is_boundary)
end

The data-generation code is used to generate training data sets for the NGOs. A data set is generated in three steps:

1. Sample random functions for the PDE parameters and source terms. These are $\theta, f, g, \eta$ for diffusion, and additionally $v_x, v_y$ for advection-diffusion.
2. Assemble and solve the resulting FEM discretisations.
3. Export the resulting fields (i.e. all PDE parameters as well as the solution $u$) to an HDF5 file.

# Function sampling

The sampling of random functions is done on `sample_functions.jl`. All PDE parameters are generated on a fixed function basis with random coefficients. The coefficients are either drawn from a uniform distribution (for $\theta$) or a normal distribution. The random coefficients are sampled in such a way that choosing a fixed seed produces a deterministic sequence of functions.

# Assembly

The assembly of the FEM discretisations is done in `diffusion.jl` and `advection_diffusion.jl`, with some code in `assembly.jl`. The general assembly loop works as follows:

1. Iterate over each part of the domain or its boundary
2. For each element, assemble the element stiffness matrix
3. Add the element stiffness matrix to the global matrix at the correct indices

To make use of efficient matrix-matrix and matrix-vector operations, the inner assembly loops make heavy use of BLAS calls, which produces un-intuitive code. Suppose that we want to assemble the following term, adding the result to my element stiffness matrix $K_{jk}$:

$$
K_{jk} \leftarrow K_{jk} + \int_\text{el} \partial_x \varphi_j(x, y)~\theta(x, y)~\partial_x\varphi_k(x, y)~dx~dy,
$$

where $\text{el}$ is one element in $\Omega$ (i.e. a rectangular area consisting of one knot span in $x$ and one knot span in $y$). We assume that the basis functions $\varphi$ and coefficient $\theta$ are sampled on some set of quadrature points $(x_i, y_i)$ with associated quadrature weights $w_i$:

$$
\begin{align}
  \text{udx}_{ij} &= \partial_x\varphi_j(x_i, y_i) \\
  \theta_i &= \theta(x_i, y_i).
\end{align}
$$

We can then compute the integral as

$$
\begin{align}
  \int_\text{el} \partial_x \varphi_j(x, y)~\theta(x, y)~\partial_x\varphi_k(x, y)~dx~dy
  &\approx \sum_i \partial_x\varphi_j(x_i, y_i)~\theta(x_i, y_i)~w_i~\partial_x\varphi_k(x_i, y_i) \\
  &= \sum_i \text{udx}_{ij}~\theta_i~w_i~\text{udx}_{ik}.
\end{align}
$$

I compute this in two steps:

1. Compute $\text{tmp}\_{ik} = \theta_i w_i \text{udx}\_{ik}$, which only requires pointwise multiplications
2. Then, the integrals can be computed as $K_{jk} = \sum_i \text{udx}\_{ij}\text{tmp}\_{ik} = \text{udx}^T \text{tmp},$ which is a single matrix-matrix product.

However, doing this as `udx' * tmp` in Julia allocates a new matrix, which I want to avoid. Instead, I allocate a single buffer $K_{jk}$ and want to perform either

- $K_{jk} \leftarrow \text{udx}^T \text{tmp}$ (for the first term I assemble)
- $K_{jk} \leftarrow K_{jk} + \text{udx}^T \text{tmp}$ (for subsequent terms)

I do this with the BLAS function `gemm!`. Calling `gemm!(tA, tB, alpha, A, B, beta, C)` performs `C .= alpha .* (A * B) .+ beta .* C` in-place without allocating, where `A` or `B` may be transposed: setting `tA` to `'N'`, `'T'`, or `'C'` performs the multiplication with `A`, its transpose, or its complex conjugate transpose, respectively. Changing `tB` analogously changes the way `B` is used.

The first term I assemble should compute `K .= udx' * tmp`, which means calling

	gemm!('T', 'N', true, udx, tmp, false, K)  # K .= udx' * tmp
	
Here setting alpha and beta to `true` and `false` are equivalent to `1` and `0`, respectively.

For the terms that follow, I want to add the matrix-matrix product to the element stiffness matrix, which means setting `beta` to 1, or equivalently, `true`:

	gemm!('T', 'N', true, udx, tmp, true, K)  # K .+= udx' * tmp

For some terms, `alpha` is set to some other constant if the relevant term includes a constant scaling factor as well.

To assemble the right-hand side terms, the process is very similar. However, now we're only dealing with matrix-vector products, meaning I use `gemv!` instead of `gemm!`.

# Exporting

Once all the solutions are computed, the resulting data is exported to an HDF5 file. This process is fairly straightforward:

1. Store the configuration of the data set in the output file's attributes.
2. For each field (i.e. the PDE parameters and the solutions), create a group containing the coefficients of the field as well as a description of the basis.

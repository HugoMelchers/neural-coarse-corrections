includet("diffusion.jl")
includet("advection_diffusion.jl")
includet("helmholtz.jl")

function solve(settings, fields)
    # I don't want to use the parallelism of BLAS, since it's much more efficient to parallelised
    # over the large number of independent problems we want to solve
    if Threads.nthreads() == 1
        @warn "Running on 1 thread. Launch julia with `--threads=<N>` to run on multiple threads"
    end
    LinearAlgebra.BLAS.set_num_threads(1)
    problem = settings["problem"]
    if problem == "diffusion"
        solve_diffusion(settings, fields)
    elseif problem == "advection-diffusion"
        solve_advdiff(settings, fields)
    elseif problem == "helmholtz"
        solve_helmholtz(settings, fields)
    end
end

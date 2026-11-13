#=
Generation of 3D diffusion data sets. Run from the `data_generation` project, e.g.

    julia --project=.. --threads=8
    julia> include("generate.jl")
    julia> generate_diffusion3d()

This re-uses the 1D and 2D code from the parent directory (B-splines, 1D/2D functions, GRFs) without
modifying it. All 3D-specific functions have a `_3d` suffix or distinct names, so that the 2D and 3D
code can be loaded into the same session.
=#
import Revise: includet

const PARENT_DIR = joinpath(@__DIR__, "..")

includet(joinpath(PARENT_DIR, "functions1d.jl"))
includet(joinpath(PARENT_DIR, "functions2d.jl"))
includet(joinpath(PARENT_DIR, "bsplines.jl"))
includet(joinpath(PARENT_DIR, "grfs.jl"))
includet(joinpath(PARENT_DIR, "sample_functions.jl"))
includet(joinpath(PARENT_DIR, "export_hdf5.jl"))
includet(joinpath(@__DIR__, "functions3d.jl"))
includet(joinpath(@__DIR__, "assembly3d.jl"))
includet(joinpath(@__DIR__, "diffusion3d.jl"))
includet(joinpath(@__DIR__, "sample_functions3d.jl"))
includet(joinpath(@__DIR__, "export_hdf5_3d.jl"))

function generate_diffusion3d()
    settings = Dict(
        "problem" => "diffusion3d",
        "num" => 10000,
        "seed" => 1234,
        "check" => false,
        "theta-segments" => 5,
        "theta-degree" => 3,
        "theta-min" => 0.2,
        "theta-max" => 1.0,
        "rhs-type" => "bspline-orthonormal",
        "rhs-degree" => 3,
        "rhs-segments" => 5,
        "f-scale" => 1.0,
        "g-scale" => 1.0,
        "eta-scale" => 1.0,
        "u-degree" => 3,
        "u-segments" => 15,
        "gamma" => 800.0,
        "split" => false,
    )
    fields = sample_functions_3d(settings, "Sampling")
    solve_diffusion_3d(settings, fields)
    export_hdf5_3d(settings, fields, "diffusion3d")
end

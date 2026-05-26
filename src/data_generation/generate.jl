import Revise: includet

includet("functions1d.jl")
includet("functions2d.jl")
includet("bsplines.jl")
includet("grfs.jl")
includet("parse_args.jl")
includet("sample_functions.jl")
includet("solve.jl")
includet("export_hdf5.jl")
includet("assembly.jl")

function generate_diffusion_dataset_1()
    settings = Dict(
        "problem" => "diffusion",
        "num" => 10000,
        "seed" => 1234,
        "check" => true,
        "theta-segments" => 9,
        "theta-degree" => 3,
        "theta-min" => 0.2,
        "theta-max" => 1.0,
        "rhs-type" => "grf",
        "rhs-grf-length" => 0.2,
        "f-scale" => 1.0,
        "g-scale" => 1.0,
        "eta-scale" => 1.0,
        "u-degree" => 3,
        "u-segments" => 54,
        "gamma" => 800.0,
        "split" => false,
    )
    fields = sample_functions(settings, "Sampling")
    solve(settings, fields)
    export_hdf5(settings, fields, "diffusion-dataset-1")
end

function generate_diffusion_dataset_2()
    settings = Dict(
        "problem" => "diffusion",
        "num" => 10000,
        "seed" => 1234,
        "check" => true,
        "theta-segments" => 9,
        "theta-degree" => 3,
        "theta-min" => 0.2,
        "theta-max" => 1.0,
        "rhs-type" => "bspline",
        "rhs-degree" => 3,
        "rhs-segments" => 9,
        "f-scale" => 1.0,
        "g-scale" => 1.0,
        "eta-scale" => 1.0,
        "u-degree" => 3,
        "u-segments" => 54,
        "gamma" => 800.0,
        "split" => false,
    )
    fields = sample_functions(settings, "Sampling")
    solve(settings, fields)
    export_hdf5(settings, fields, "diffusion-dataset-2")
end

function generate_diffusion_dataset_3()
    settings = Dict(
        "problem" => "diffusion",
        "num" => 10000,
        "seed" => 1234,
        "check" => true,
        "theta-segments" => 9,
        "theta-degree" => 3,
        "theta-min" => 0.2,
        "theta-max" => 1.0,
        "rhs-type" => "bspline-orthonormal",
        "rhs-degree" => 3,
        "rhs-segments" => 9,
        "f-scale" => 1.0,
        "g-scale" => 1.0,
        "eta-scale" => 1.0,
        "u-degree" => 3,
        "u-segments" => 54,
        "gamma" => 800.0,
        "split" => false,
    )
    fields = sample_functions(settings, "Sampling")
    solve(settings, fields)
    export_hdf5(settings, fields, "diffusion-dataset-3")
end

function generate_advdiff_low_peclet()
    settings = Dict(
        "problem" => "advection-diffusion",
        "num" => 10000,
        "seed" => 1234,
        "check" => true,
        "theta-segments" => 9,
        "theta-degree" => 3,
        "theta-min" => 0.02,
        "theta-max" => 0.2,
        "rhs-type" => "bspline-orthonormal",
        "rhs-degree" => 3,
        "rhs-segments" => 9,
        "v-type" => "randn",
        "v-scale" => 5.0,
        "f-scale" => 1.0,
        "g-scale" => 1.0,
        "eta-scale" => 1.0,
        "u-degree" => 3,
        "u-segments" => 54,
        "gamma" => 800.0,
        "sign" => 1,
        "split" => false,
    )
    fields = sample_functions(settings, "Sampling")
    solve(settings, fields)
    export_hdf5(settings, fields, "advdiff-low-peclet")
end

function generate_advdiff_high_peclet()
    settings = Dict(
        "problem" => "advection-diffusion",
        "num" => 10000,
        "seed" => 1234,
        "check" => true,
        "theta-segments" => 9,
        "theta-degree" => 3,
        "theta-min" => 0.02,
        "theta-max" => 0.2,
        "rhs-type" => "bspline-orthonormal",
        "rhs-degree" => 3,
        "rhs-segments" => 9,
        "v-type" => "radial",
        "v-dist" => "linear",
        "v-min" => 0.0,
        "v-max" => 100.0,
        "f-scale" => 1.0,
        "g-scale" => 1.0,
        "eta-scale" => 1.0,
        "u-degree" => 3,
        "u-segments" => 54,
        "gamma" => 800.0,
        "sign" => 1,
        "split" => false,
    )
    fields = sample_functions(settings, "Sampling")
    solve(settings, fields)
    export_hdf5(settings, fields, "advdiff-high-peclet")
end

function generate_helmholtz()
    settings = Dict(
        "problem" => "helmholtz",
        "num" => 10000,
        "seed" => 1234,
        "check" => true,
        "theta-segments" => 9,
        "theta-degree" => 3,
        "theta-min" => 0.5,
        "theta-max" => 30.0,
        "rhs-type" => "bspline-orthonormal",
        "rhs-degree" => 3,
        "rhs-segments" => 9,
        "f-scale" => 1.0,
        "g-scale" => 1.0,
        "eta-scale" => 1.0,
        "u-degree" => 3,
        "u-segments" => 54,
        "split" => false,
    )
    fields = sample_functions(settings, "Sampling")
    solve(settings, fields)
    export_hdf5(settings, fields, "helmholtz")
end

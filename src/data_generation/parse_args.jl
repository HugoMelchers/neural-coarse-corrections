import ArgParse: ArgParseSettings, parse_args, @add_arg_table!

function get_args()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--problem"
        help = "Identifier of the PDE (\"diffusion\" or \"advection-diffusion\")"
        "--num", "-n"
        arg_type = Int
        help = "Number of solutions to generate"
        "--seed"
        arg_type = Int
        help = "Seed to use for the random sampling"
        "--check"
        help = "Check whether discretisations are symmetric, positive definite, or both, depending on problem"

        "--u-degree"
        arg_type = Int
        help = "B-spline degree to use for the solution"
        "--u-segments"
        arg_type = Int
        help = "Number of knot segments to use for the solution"

        "--theta-degree"
        arg_type = Int
        help = "B-spline degree to use for θ"
        "--theta-segments"
        arg_type = Int
        help = "Number of knot segments to use for θ"
        "--theta-min"
        arg_type = Int
        help = "Minimum value for θ"
        "--theta-max"
        arg_type = Int
        help = "Maximum value for θ"

        "--rhs-type"
        help = "What kind of basis to use for the right-hand-side data. Should be \"bspline\" or \"grf\""
        "--rhs-grf-length"
        help = "The GRF length scale to use for f, g, and η"
        "--rhs-degree"
        arg_type = Int
        help = "B-spline degree to use for f, g, and η"
        "--rhs-segments"
        arg_type = Int
        help = "Number of knot segments to use for f, g, and η"
        "--f-scale"
        arg_type = Float64
        help = "Scaling factor for the forcing term f"
        "--g-scale"
        arg_type = Float64
        help = "Scaling factor for the Dirichlet boundary data g"
        "--eta-scale"
        arg_type = Float64
        help = "Scaling factor for the Neumann boundary data η"

        "--v-scale"
        arg_type = Float64
        help = "Scaling factor for the velocity field (advection-diffusion only)"

        "--gamma"
        arg_type = Float64
        help = "Stabilisation coefficient"
    end
    return parse_args(s)
end

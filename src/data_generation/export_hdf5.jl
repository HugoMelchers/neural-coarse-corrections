import HDF5: h5open, attributes, create_group

function export_hdf5(settings, fields, filename)
    path = "../../datasets/" * filename * ".hdf5"
    h5open(path, "w") do file
        attrs = attributes(file)
        for (key, value) in pairs(settings)
            attrs[key] = value
        end
        for (name, functions) in pairs(fields)
            _write_field(file, name, functions)
        end
    end
end

function _write_field(file, name, functions)
    group = create_group(file, name)
    attrs = attributes(group)
    functions = flatten(functions)

    group["weights"] = functions.coefficients
    basis_group = create_group(group, "basis")
    if functions isa Functions1D
        _write_basis(basis_group, functions.basis)
    elseif functions isa TensorFunctions2D
        _write_tensor_basis(basis_group, functions.x, functions.y)
    else
        throw(
            ErrorException(
                "in _write_field: got unexpected type $(typeof(functions)) for \"$name\"",
            ),
        )
    end
end

function _write_basis(group, basis)
    attrs = attributes(group)
    if basis isa BSpline
        attrs["type"] = "bspline"
        attrs["degree"] = basis.degree
        attrs["knots"] = basis.knots
    elseif basis isa RBF1D
        attrs["type"] = "rbf1d"
        attrs["l"] = basis.l
        attrs["nodes"] = basis.nodes
    else
        throw(
            ErrorException(
                "in _write_basis: expected BSpline or RBF1D, got type $(typeof(basis))",
            ),
        )
    end
end

function _write_tensor_basis(group, basis_x, basis_y)
    attrs = attributes(group)
    attrs["type"] = "tensor"
    group_x = create_group(group, "x")
    group_y = create_group(group, "y")
    _write_basis(group_x, basis_x)
    _write_basis(group_y, basis_y)
end

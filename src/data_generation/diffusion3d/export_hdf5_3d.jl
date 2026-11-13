import HDF5: h5open, attributes, create_group

#=
3D equivalent of `export_hdf5` (see `../export_hdf5.jl`). The file layout is the same as in 2D,
except that 3D fields have a basis of type "tensor" with three subgroups "x", "y", and "z", and
their weights are 4D arrays. 1D and 2D fields (such as boundary data) are written as in 2D.
=#
function export_hdf5_3d(settings, fields, filename)
    path = joinpath(@__DIR__, "..", "..", "..", "datasets", filename * ".hdf5")
    h5open(path, "w") do file
        attrs = attributes(file)
        for (key, value) in pairs(settings)
            attrs[key] = value
        end
        for (name, functions) in pairs(fields)
            _write_field_3d(file, name, functions)
        end
    end
end

function _write_field_3d(file, name, functions)
    functions = flatten(functions)
    if !(functions isa TensorFunctions3D)
        return _write_field(file, name, functions)
    end
    group = create_group(file, name)
    group["weights"] = functions.coefficients
    basis_group = create_group(group, "basis")
    attributes(basis_group)["type"] = "tensor"
    _write_basis(create_group(basis_group, "x"), functions.x)
    _write_basis(create_group(basis_group, "y"), functions.y)
    _write_basis(create_group(basis_group, "z"), functions.z)
end

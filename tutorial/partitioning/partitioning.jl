using Bcube
using BcubeMPI
using WriteVTK
using PartitionedArrays
const PArray = PartitionedArrays

function tuto_partitioning(parts)
    nparts = num_parts(parts)
    part_id = PArray.get_part(get_part_ids(parts))
    println("I am part_id = $part_id over $nparts")

    if i_am_main(parts)
        println("Running on main part (part_id = $part_id)")

        filename = "../input/mesh/naca0012_o1.msh"
        mesh = read_msh(filename, 2) # '2' indicates the space dimension (3 by default)
        partIds = partitioning(mesh, nparts)

        resultfile = "../myout/partitioning"
        dict_vars = Dict("partIds" => (partIds[:], VTKCellData()))
        write_vtk(resultfile, 0, 0.0, mesh, dict_vars)

        println("Result file :" * resultfile * ".pvd")
    end
end

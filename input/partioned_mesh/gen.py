import gmsh

lc = 1
nx = 3
ny = 5

gmsh.initialize()

#gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
gmsh.option.setNumber("Mesh.PartitionSplitMeshFiles", 1)
gmsh.option.setNumber("Mesh.PartitionCreateGhostCells", 1)

A = gmsh.model.geo.addPoint(-1, -1, 0, lc)
B = gmsh.model.geo.addPoint( 1, -1, 0, lc)
C = gmsh.model.geo.addPoint( 1,  1, 0, lc)
D = gmsh.model.geo.addPoint(-1,  1, 0, lc)

AB = gmsh.model.geo.addLine(A, B)
BC = gmsh.model.geo.addLine(B, C)
CD = gmsh.model.geo.addLine(C, D)
DA = gmsh.model.geo.addLine(D, A)

ABCD = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([AB, BC, CD, DA])])

gmsh.model.geo.synchronize()

for l in [AB, CD]:
    gmsh.model.mesh.setTransfiniteCurve(l,nx)
for l in [BC, DA]:
    gmsh.model.mesh.setTransfiniteCurve(l,ny)
gmsh.model.mesh.setTransfiniteSurface(ABCD)
gmsh.model.mesh.setRecombine(2, ABCD)

gmsh.model.mesh.generate(2)
gmsh.model.mesh.partition(3)

gmsh.write("mesh.msh")
gmsh.finalize()
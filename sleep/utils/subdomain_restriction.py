# |---|---|
# | F | S |
# |---|---|
#
# We will read in a background mesh which is union of solid and fluid
# Then we might want to do seperate fluid/solid computation on the
# corresponding submeshes. For this we might need to put that from the
# union function space to the spaces on the subdomain (and back)

from sleep.utils.fem_eval import DegreeOfFreedom, FEBasisFunction
from sleep.utils.petsc_matrix import petsc_serial_matrix
from petsc4py import PETSc
import dolfin as df
import numpy as np


def restriction_matrix(V, TV, rmesh):
    '''TV's mesh is the EmbeddedMesh of V'''
    assert V.ufl_element() == TV.ufl_element()
    assert TV.mesh().id() == rmesh.id()
    
    mesh = V.mesh()
    tdim = mesh.topology().dim()
    # Let's get the mapping or cell of TV mesh to V mesh cells
    mapping = rmesh.parent_entity_map[mesh.id()][tdim]  
    # The idea is to evaluate TV's degrees of freedom at basis functions of V
    Tdmap = TV.dofmap()
    TV_dof = DegreeOfFreedom(TV)

    dmap = V.dofmap()
    V_basis_f = FEBasisFunction(V)

    # Rows
    visited_dofs = [False]*TV.dim()
    # Column values
    dof_values = np.zeros(V_basis_f.elm.space_dimension(), dtype='double')
    with petsc_serial_matrix(TV, V) as mat:

        for trace_cell in range(TV.mesh().num_cells()):
            TV_dof.cell = trace_cell
            trace_dofs = Tdmap.cell_dofs(trace_cell)
            # The corresponding cell in V mesh
            cell = mapping[trace_cell]
            V_basis_f.cell = cell
            
            dofs = dmap.cell_dofs(cell)
            for local_T, dof_T in enumerate(trace_dofs):

                if visited_dofs[dof_T]:
                    continue
                else:
                    visited_dofs[dof_T] = True

                # Define trace dof
                TV_dof.dof = local_T
                
                # Eval at V basis functions
                for local, dof in enumerate(dofs):
                    # Set which basis foo
                    V_basis_f.dof = local
                    
                    dof_values[local] = TV_dof.eval(V_basis_f)

                # Can fill the matrix now
                col_indices = np.array(dofs, dtype='int32')
                # Insert
                mat.setValues([dof_T], col_indices, dof_values, PETSc.InsertMode.INSERT_VALUES)
    return df.PETScMatrix(mat)

# --------------------------------------------------------------------

if __name__ == '__main__':
    from sleep.utils import EmbeddedMesh
    from sleep.utils import transpose_matrix
    from dolfin import *
    
    fs_domain = UnitSquareMesh(32, 32)
    cell_f = MeshFunction('size_t', fs_domain, 2, 1)
    CompiledSubDomain('x[0] > 0.5 - DOLFIN_EPS').mark(cell_f, 2)

    dx_fs = Measure('dx', domain=fs_domain, subdomain_data=cell_f)
    
    fluid = EmbeddedMesh(cell_f, 1)
    solid = EmbeddedMesh(cell_f, 2)

    u = Expression('x[0]+2*x[1]', degree=1)
    # Test putting scalar to subdomain -------------------------------
    FS = FunctionSpace(fs_domain, 'CG', 1)
    u_fs = interpolate(u, FS)
    # The subdomain space
    F = FunctionSpace(fluid, 'CG', 1)

    R = restriction_matrix(FS, F, fluid)
    # Now we can tranport
    u_f = Function(F)
    R.mult(u_fs.vector(), u_f.vector())

    e = inner(u_f - u, u_f - u)*dx  # Implied, fluid
    norm = inner(u_f, u_f)*dx

    assert sqrt(abs(assemble(e))) < 1E-13 and sqrt(abs(assemble(norm))) > 0

    # Now going back
    u_fs.vector()[:] *= 0
    assert u_fs.vector().norm('linf') < 1E-13
    # We extend by tranpose
    R.transpmult(u_f.vector(), u_fs.vector())
    # We put it there correct
    e = inner(u_fs - u, u_fs - u)*dx_fs(1)
    assert sqrt(abs(assemble(e))) < 1E-13
    # NOTE: we set up a hat so unless on iface the exetended function is 0
    # there will be some contributions of the extension to the cells of the
    # neighbor domain

    # Explict
    u_fs.vector()[:] *= 0
    assert u_fs.vector().norm('linf') < 1E-13
    # We extend by tranpose
    E = transpose_matrix(R)
    E.mult(u_f.vector(), u_fs.vector())
    # We put it there correct
    e = inner(u_fs - u, u_fs - u)*dx_fs(1)
    assert sqrt(abs(assemble(e))) < 1E-13

    u = Expression(('x[0]+2*x[1]', '-2*x[0]+3*x[1]'), degree=1)
    # Test putting vector to subdomain -------------------------------
    FS = VectorFunctionSpace(fs_domain, 'CG', 2)
    u_fs = interpolate(u, FS)
    # The subdomain space
    S = VectorFunctionSpace(solid, 'CG', 2)

    R = restriction_matrix(FS, S, solid)
    # Now we can tranport
    u_s = Function(S)
    R.mult(u_fs.vector(), u_s.vector())

    e = inner(u_s - u, u_s - u)*dx  # Implied, fluid
    norm = inner(u_s, u_s)*dx

    assert sqrt(abs(assemble(e))) < 1E-13 and sqrt(abs(assemble(norm))) > 0

    # Now going back
    u_fs.vector()[:] *= 0
    assert u_fs.vector().norm('linf') < 1E-13
    # We extend by tranpose
    E = R.transpmult(u_s.vector(), u_fs.vector())
    # We put it there correct
    e = inner(u_fs - u, u_fs - u)*dx_fs(2)
    assert sqrt(abs(assemble(e))) < 1E-13

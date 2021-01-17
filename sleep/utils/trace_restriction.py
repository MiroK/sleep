# |---|---|
# | F I S |
# |---|---|
#
# The fluid and structure domains need to communicate to each other
# over interface or just talk to the interface. For this it is important
# to be able to restrict values to manifolds of codimension 1
from sleep.utils.fem_eval import DegreeOfFreedom, FEBasisFunction
from sleep.utils.petsc_matrix import petsc_serial_matrix
from petsc4py import PETSc
import dolfin as df
import numpy as np
import ufl


def trace_cell(o):
    '''
    UFL cell corresponding to restriction of o[cell] to its facets, performing
    this restriction on o[function-like], or objects in o[function space]
    '''
    # Space
    if hasattr(o, 'ufl_cell'):
        return trace_cell(o.ufl_cell())
    # Foo like
    if hasattr(o, 'ufl_element'):
        return trace_cell(o.ufl_element().cell())
    # Elm
    if hasattr(o, 'cell'):
        return trace_cell(o.cell())

    # Another cell
    cell_name = {'tetrahedron': 'triangle',
                 'triangle': 'interval'}[o.cellname()]

    return ufl.Cell(cell_name, o.geometric_dimension())


def trace_matrix(V, TV, trace_mesh):
    '''Take traces of V on trace_mesh in TV space'''
    assert TV.mesh().id() == trace_mesh.id()
    
    # Compatibility of spaces
    assert V.dolfin_element().value_rank() == TV.dolfin_element().value_rank()
    assert V.ufl_element().value_shape() == TV.ufl_element().value_shape()
    assert trace_cell(V) == TV.mesh().ufl_cell()
    assert V.mesh().geometry().dim() == TV.mesh().geometry().dim()
    
    mesh = V.mesh()
    # Trace mesh might be viewed from all the neighbors leading to different
    # TV spaces. This is not convenient as then we will need to embed one
    # into the other. So TV should be made once and we ask for restriction
    # of different neighbors
    fdim = trace_mesh.topology().dim()

    embedding_entity_map = trace_mesh.parent_entity_map
    # If TV's mesh was defined as trace mesh of V
    if mesh.id() in embedding_entity_map:
        assert fdim in embedding_entity_map[mesh.id()]
        mapping = embedding_entity_map[mesh.id()][fdim]


    mesh.init(fdim, fdim+1)
    f2c = mesh.topology()(fdim, fdim+1)  # Facets of V to cell of V

    # The idea is to evaluate TV's degrees of freedom at basis functions
    # of V
    Tdmap = TV.dofmap()
    TV_dof = DegreeOfFreedom(TV)

    dmap = V.dofmap()
    V_basis_f = FEBasisFunction(V)

    # Rows
    visited_dofs = [False]*TV.dim()
    # Column values
    dof_values = np.zeros(V_basis_f.elm.space_dimension(), dtype='double')
    with petsc_serial_matrix(TV, V) as mat:

        for tc in range(trace_mesh.num_cells()):
            # We might 
            TV_dof.cell = tc
            trace_dofs = Tdmap.cell_dofs(tc)

            # Figure out the dofs of V to use here. Does not matter which
            # cell of the connected ones we pick
            cell = f2c(mapping[tc])[0]
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

    # We take fluid domain and master and define interaface wrt to it
    fluid_facets = MeshFunction('size_t', fluid, 1, 0)
    CompiledSubDomain('near(x[0], 0.5)').mark(fluid_facets, 1)
    # Corresponding suraface integral
    dI = Measure('ds', domain=fluid, subdomain_data=fluid_facets, subdomain_id=1)

    interface = EmbeddedMesh(fluid_facets, 1)
    
    f = Expression('x[0]+2*x[1]', degree=1)
    # Test putting scalar to primary ---------------------------------
    F = FunctionSpace(fluid, 'CG', 1)
    uf = interpolate(f, F)

    TF = FunctionSpace(interface, 'CG', 1)
    T = trace_matrix(F, TF, interface)

    # Now we can tranport
    Tuf = Function(TF)
    T.mult(uf.vector(), Tuf.vector())

    e = inner(Tuf - f, Tuf - f)*dx  # Implied, interface
    norm = inner(Tuf, Tuf)*dx

    assert sqrt(abs(assemble(e))) < 1E-13 and sqrt(abs(assemble(norm))) > 0

    # Going back
    uf.vector()[:] *= 0
    assert uf.vector().norm('linf') < 1E-13
    # We extend by tranpose
    T.transpmult(Tuf.vector(), uf.vector())
    # We put it there correct
    e = inner(uf - f, uf - f)*dI
    assert sqrt(abs(assemble(e))) < 1E-13

    # Test putting vector to primary----------------------------------
    f = Expression(('x[0]+2*x[1]', 'x[0]+4*x[1]'), degree=1)
    # Test putting scalar to primary ---------------------------------
    F = VectorFunctionSpace(fluid, 'CG', 2)
    uf = interpolate(f, F)

    TF = VectorFunctionSpace(interface, 'CG', 2)
    T = trace_matrix(F, TF, interface)

    # Now we can tranport
    Tuf = Function(TF)
    T.mult(uf.vector(), Tuf.vector())

    e = inner(Tuf - f, Tuf - f)*dx  # Implied, interface
    norm = inner(Tuf, Tuf)*dx

    assert sqrt(abs(assemble(e))) < 1E-13 and sqrt(abs(assemble(norm))) > 0

    # Going back
    uf.vector()[:] *= 0
    assert uf.vector().norm('linf') < 1E-13
    # We extend by tranpose
    T.transpmult(Tuf.vector(), uf.vector())
    # We put it there correct
    e = inner(uf - f, uf - f)*dI
    assert sqrt(abs(assemble(e))) < 1E-13

    # Now from solid which is the slave side -------------------------

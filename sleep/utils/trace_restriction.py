# |---|---|
# | F I S |
# |---|---|
#
# The fluid and structure domains need to communicate to each other
# over interface or just talk to the interface. For this it is important
# to be able to restrict values to manifolds of codimension 1
from sleep.utils.fem_eval import DegreeOfFreedom, FEBasisFunction
from sleep.utils.petsc_matrix import petsc_serial_matrix
from sleep.utils.embedded_mesh import embed_mesh
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
    else:
        # Now we need to compute how to embed trace_mesh in mesh of V
        embedding_entity_map[mesh.id()] = embed_mesh(trace_mesh, mesh)
        
    # Now makes sense
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
    # Marking facets just for the purpose of checking by integration
    solid_facets = MeshFunction('size_t', solid, 1, 0)
    CompiledSubDomain('near(x[0], 0.5)').mark(solid_facets, 2)
    # Corresponding suraface integral
    dI = Measure('ds', domain=solid, subdomain_data=solid_facets, subdomain_id=2)
    
    f = Expression('x[0]-4*x[1]', degree=1)
    # Test putting scalar to primary ---------------------------------
    S = FunctionSpace(solid, 'CG', 1)
    us = interpolate(f, S)
    # NOTE: we are keeping the trace space as with fluid
    TF = FunctionSpace(interface, 'CG', 1)
    
    T = trace_matrix(S, TF, interface)
    # Now we can tranport
    Tus = Function(TF)
    T.mult(us.vector(), Tus.vector())

    e = inner(Tus - f, Tus - f)*dx  # Implied, interface
    norm = inner(Tus, Tus)*dx

    assert sqrt(abs(assemble(e))) < 1E-13 and sqrt(abs(assemble(norm))) > 0

    # Going back
    us.vector()[:] *= 0
    assert us.vector().norm('linf') < 1E-13
    # We extend by tranpose
    T.transpmult(Tus.vector(), us.vector())
    # We put it there correct
    e = inner(us - f, us - f)*dI
    assert sqrt(abs(assemble(e))) < 1E-13

    ###############
    # A use case: Let there be a deformation on solid domain. Now in the
    # fluid solver we'd like to evaluate it on the interface. Whare could
    # work is
    #
    #     us -> take is trace Tus -> extend to fluid -> compute the stress there
    #
    ###############

    fs_domain = UnitSquareMesh(64, 64)
    cell_f = MeshFunction('size_t', fs_domain, 2, 1)
    CompiledSubDomain('x[0] > 0.5 - DOLFIN_EPS').mark(cell_f, 2)

    dx_fs = Measure('dx', domain=fs_domain, subdomain_data=cell_f)
    
    fluid = EmbeddedMesh(cell_f, 1)
    solid = EmbeddedMesh(cell_f, 2)

    # We take fluid domain and master and define interaface wrt to it
    fluid_facets = MeshFunction('size_t', fluid, 1, 0)
    CompiledSubDomain('near(x[0], 0.5)').mark(fluid_facets, 1)
    # Corresponding suraface integral
    dI_f = Measure('ds', domain=fluid, subdomain_data=fluid_facets, subdomain_id=1)
    # Fluid is master
    interface = EmbeddedMesh(fluid_facets, 1)

    solid_facets = MeshFunction('size_t', solid, 1, 0)
    CompiledSubDomain('near(x[0], 0.5)').mark(solid_facets, 2)
    # Corresponding suraface integral
    dI_s = Measure('ds', domain=solid, subdomain_data=solid_facets, subdomain_id=2)

    def get_u(mesh):
        '''Ground truth'''
        x, y = SpatialCoordinate(mesh)
        return as_vector((x**2+y**2, x-2*y**2))

    uF0 = get_u(fluid)
    
    F = VectorFunctionSpace(fluid, 'CG', 2)
    S = VectorFunctionSpace(solid, 'CG', 2)
    uF = project(uF0, F)

    TF = VectorFunctionSpace(interface, 'CG', 2)
    F2iface = trace_matrix(F, TF, interface)
    S2iface = trace_matrix(S, TF, interface)

    TuF = Function(TF) # Intermediate step
    uS = Function(S)   # Destination

    F2iface.mult(uF.vector(), TuF.vector())
    S2iface.transpmult(TuF.vector(), uS.vector())

    # For comparison
    uS0 = get_u(solid)

    n = Constant((-1, 0))
    # Eval some functions now
    es = [inner(uS0 - uS, uS0 - uS)*dI_s,
          inner(dot(grad(uS0), n) - dot(grad(uS), n),
                dot(grad(uS0), n) - dot(grad(uS), n))*dI_s]
    
    for e in es:
        print sqrt(abs(assemble(e)))
    # So we see that "sharing" displacement is not enough
    # The other option is project the derived quantity, and transform
    # its trace
    F_grad = TensorFunctionSpace(fluid, 'DG', 1)
    stress_f = project(grad(uF), F_grad)

    TF_grad = TensorFunctionSpace(interface, 'DG', 1)
    Tstress = Function(TF_grad)
    trace_matrix(F_grad, TF_grad, interface).mult(stress_f.vector(), Tstress.vector())

    S_grad = TensorFunctionSpace(solid, 'DG', 1)
    stress_s = Function(S_grad)
    trace_matrix(S_grad, TF_grad, interface).transpmult(Tstress.vector(), stress_s.vector())

    es = [inner(dot(grad(uS0), n) - dot(stress_s, n),
                dot(grad(uS0), n) - dot(stress_s, n))*dI_s,
          inner(grad(uS0) - stress_s, grad(uS0) - stress_s)*dI_s]

    for e in es:
        print sqrt(abs(assemble(e)))
    # Now we're close enough again

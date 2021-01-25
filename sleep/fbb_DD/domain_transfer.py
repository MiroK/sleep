from sleep.utils import EmbeddedMesh, trace_matrix
from dolfin import *


def transfer_DD(expr, V, interface, tol=1E-10):
    '''
    Let interface = (bdries-domain-1, bdries-domain-2, shared-bdry-index).
    Using space V defined on bdries-domain-1.mesh() we build a good enough 
    representation of it in V(bdiers-domain-2.mesh()).
    '''
    bdries1, bdries2, tag = interface
    # Share?
    assert tag in set(bdries1.array()) and tag in set(bdries2.array())
    # On first
    mesh1 = bdries1.mesh()
    assert V.mesh().id() == mesh1.id()

    # We assume that the shared surface is on the boundary, duh :). So 
    # the idea is to basically do an L^2 projection of expr onto the trace
    # space of V on the interface
    ds_ = Measure('ds', domain=mesh1, subdomain_data=bdries1)
    # Rhs of the projection problem that we will need to restrict
    v = TestFunction(V)
    b = assemble(inner(v, expr)*ds_(tag))
    # The trace space is
    interface = EmbeddedMesh(bdries1, tag)
    interface.compute_embedding(bdries2, tag, tol)  # Fails if the meshes are far apart
    # The trace space
    TV = FunctionSpace(interface, V.ufl_element().reconstruct(cell=interface.ufl_cell()))

    Tb = Function(TV).vector()
    # Restrict
    trace_matrix(V, TV, interface).mult(b, Tb)
    # Lhs of the projection problem
    u, v = TrialFunction(TV), TestFunction(TV)
    M = assemble(inner(u, v)*dx)
    # Project
    Texpr = Function(TV)  # Trace of the expression (sort of)
    solve(M, Texpr.vector(), Tb)
    # Finally we extend this to V(mesh2). NOTE: it is pretty much only
    # on the interface where the result makes some sense
    mesh2 = bdries2.mesh()
    V2 = FunctionSpace(mesh2, V.ufl_element().reconstruct(cell=mesh2.ufl_cell()))
    u2 = Function(V2)
    # Extend
    trace_matrix(V2, TV, interface).transpmult(Texpr.vector(), u2.vector())
    
    return u2
    
# --------------------------------------------------------------------                       
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from dolfin import *
    import numpy as np
    import sympy as sp
    import ulfy

    n = 64
    
    mesh = UnitSquareMesh(n, n)
    V = VectorFunctionSpace(mesh, 'CG', 2)
    
    x, y = sp.symbols('x y')
    u = Function(V)
    n = Constant((1, 0))
    
    subs = {u: sp.Matrix([sp.sin(x**2+y**2), sp.cos(x**2-3*y**2)])}
    displacement = ulfy.Expression(u, subs=subs, degree=2)
    stress = ulfy.Expression(sym(grad(u)), subs=subs, degree=2)
    traction = ulfy.Expression(dot(n, sym(grad(u))), subs=subs, degree=2)
    traction_n = ulfy.Expression(dot(n, dot(n, sym(grad(u)))), subs=subs, degree=2)
    displacement_n = ulfy.Expression(dot(u, n)*n, subs=subs, degree=2)
    
    cell_f = MeshFunction('size_t', mesh, 2, 2)
    CompiledSubDomain('x[0] < 0.5+DOLFIN_EPS').mark(cell_f, 1)

    left = EmbeddedMesh(cell_f, 1)
    left_bdries = MeshFunction('size_t', left, 1, 0)
    CompiledSubDomain('near(x[0], 0.5)').mark(left_bdries, 1)

    right = EmbeddedMesh(cell_f, 2)
    right_bdries = MeshFunction('size_t', right, 1, 0)
    CompiledSubDomain('near(x[0], 0.5)').mark(right_bdries, 1)

    # Visual inspection - collect interface midpoints
    _, f2v = (left.init(1, 0), left.topology()(1, 0))
    idx, = np.where(left_bdries.array() == 1)

    midpoints = np.mean(left.coordinates()[np.row_stack([f2v(i) for i in idx])], axis=1)
    midpoints = midpoints[np.argsort(midpoints[:, 1])]

    V1 = VectorFunctionSpace(left, 'CG', 2)
    u = interpolate(displacement, V1)
    n = FacetNormal(left)
    # In DD we will need to translate traction
    expr = dot(sym(grad(u)), n)
    I = VectorFunctionSpace(left, 'DG', 1)  # Intermediate one
    f1 = transfer_DD(expr, I, (left_bdries, right_bdries, 1))

    # Normal component of traction
    expr = dot(n, dot(sym(grad(u)), n))
    I = FunctionSpace(left, 'DG', 1)  # Intermediate one
    f2 = transfer_DD(expr, I, (left_bdries, right_bdries, 1))

    # Easy vector
    I = VectorFunctionSpace(left, 'CG', 2)  # Intermediate one
    f3 = transfer_DD(u, I, (left_bdries, right_bdries, 1))
    
    # More involved vector
    I = VectorFunctionSpace(left, 'CG', 2)  # Intermediate one
    f4 = transfer_DD(dot(u, n)*n, I, (left_bdries, right_bdries, 1))
    
    x = midpoints[:, 1]    
    for fh, f in zip((f1, f2, f3, f4), (traction, traction_n, displacement, displacement_n)):
        true = np.row_stack([f(mid) for mid in midpoints])
        num = np.row_stack([fh(mid) for mid in midpoints])

        plt.figure()
        if fh.ufl_shape == ():
            plt.plot(x, true, 'x', color='black', linestyle='none', label='true')
            plt.plot(x, num)
            plt.legend()
        else:
            for i in range(fh.ufl_shape[0]):
                plt.plot(x, true[:, i], 'x', color='black', linestyle='none', label='true')
                plt.plot(x, num[:, i])
            plt.legend()

        print max(np.linalg.norm(true-num, 2, axis=-1))
    plt.show()    

from sleep.utils import trace_matrix
from sleep.utils import EmbeddedMesh
from dolfin import *
import numpy as np
import sympy as sp
import ulfy

# Let's have u on mesh. What I want is grad(u).n computed on the interface
# (in the primal form) the idea being that this could be a function that we
# could extend to other mesh and use it there

def foo(n):
    mesh = UnitSquareMesh(n, n)
    V = VectorFunctionSpace(mesh, 'CG', 2)
    
    x, y = sp.symbols('x y')
    u = Function(V)
    n = Constant((1, 0))
    
    subs = {u: sp.Matrix([sp.sin(x**2+y**2), sp.cos(x**2-3*y**2)])}
    displacement = ulfy.Expression(u, subs=subs, degree=2)
    stress = ulfy.Expression(sym(grad(u)), subs=subs, degree=2)
    traction = ulfy.Expression(dot(n, sym(grad(u))), subs=subs, degree=2)
    
    cell_f = MeshFunction('size_t', mesh, 2, 2)
    CompiledSubDomain('x[0] < 0.5+DOLFIN_EPS').mark(cell_f, 1)

    left = EmbeddedMesh(cell_f, 1)
    left_bdries = MeshFunction('size_t', left, 1, 0)
    CompiledSubDomain('near(x[0], 0.5)').mark(left_bdries, 1)

    interface = EmbeddedMesh(left_bdries, 1)
    
    right = EmbeddedMesh(cell_f, 2)
    right_bdries = MeshFunction('size_t', right, 1, 0)
    CompiledSubDomain('near(x[0], 0.5)').mark(right_bdries, 2)
    # Don't redefine!
    interface.compute_embedding(right_bdries, 2)
    
    # Okay so let's be on left with displacement
    V = VectorFunctionSpace(left, 'CG', 2)
    u = interpolate(displacement, V)
    # Is this a way to compute the stress?
    Q = VectorFunctionSpace(left, 'DG', 1)
    q = TestFunction(Q)  # Use DG1?
    n = FacetNormal(left)
    
    expr = dot(sym(grad(u)), n)
    ds_ = Measure('ds', domain=left, subdomain_data=left_bdries)
    b = assemble(inner(expr, q)*ds_(1))

    # This
    B = VectorFunctionSpace(interface, 'DG', 1)  # Match the test function of form
    MTb = Function(B)
    trace_matrix(Q, B, interface).mult(b, MTb.vector())
    # Now y is in the dual. we need it back
    x = Function(B)
    M = assemble(inner(TrialFunction(B), TestFunction(B))*dx)
    solve(M, x.vector(), MTb.vector())
    # x is now expr|_interface
    
    print assemble(inner(x-traction, x-traction)*dx)

    # What I want to do next is to extend it to right domain so that
    # if I there perform surface integral over the interface I get close
    # to what the left boundary said

    R = VectorFunctionSpace(right, 'DG', 1)
    extend = Function(R)
    trace_matrix(R, B, interface).transpmult(x.vector(), extend.vector())

    import matplotlib.pyplot as plt

    xx = np.sort([cell.midpoint().array()[1] for cell in cells(interface)])
    
    yy = np.array([x(0.5, xxi)[0] for xxi in xx])
    zz = np.array([traction(0.5, xxi)[0] for xxi in xx])
    rr = np.array([extend(0.5, xxi)[0] for xxi in xx])

    plt.figure()
    plt.plot(xx, yy, label='num')
    plt.plot(xx, zz, label='true')
    plt.plot(xx, rr, label='rec')    
    plt.legend()
    plt.show()
    
#for n in (4, 8, 16, 32, 64):
#    foo(n)

foo(128)

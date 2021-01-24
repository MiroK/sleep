from dolfin import *
import itertools
import operator
import sympy as sp
import ulfy  # https://github.com/MiroK/ulfy

# Problem to compute fluid domain displacement
# 
# -div(kappa*grad(u)) = f       
#
# with following bcs:
# 1) Dirichlet boundary sets vector u
# 2) Neumann sets dot(n, grad(u))

# is solved on FE space V
#
# NOTE kappa here could be some function to further enhance/localize
# mesh smoothing

def solve_ale(V, f, bdries, bcs, parameters):
    '''Return velocity and pressure'''
    mesh = V.mesh()
    # Let's see about boundary conditions - they need to be specified on
    # every boundary.
    assert all(k in ('dirichlet', 'neumann') for k in bcs)
    # The tags must be found in bdries
    dirichlet_bcs = bcs.get('dirichlet', ())  
    neumann_bcs = bcs.get('neumann', ())
    # Tuple of pairs (tag, boundary value) is expected
    dirichlet_tags = set(item[0] for item in dirichlet_bcs)
    neumann_tags = set(item[0] for item in neumann_bcs)

    tags = (dirichlet_tags, neumann_tags)
    # Boundary conditions must be on distinct domains
    for this, that in itertools.combinations(tags, 2):
        if this and that: assert not this & that

    # With convention that 0 bdries are inside all the exterior bdries must
    # be given conditions in bcs
    needed = set(bdries.array()) - set((0, ))
    assert needed == reduce(operator.or_, tags)
                 
    u, v = TrialFunction(V), TestFunction(V)
    assert len(u.ufl_shape) == 1
    # All but bc terms
    kappa = parameters['kappa']
    system = inner(kappa*grad(u), grad(v))*dx - inner(f, v)*dx
    # Handle natural bcs
    n = FacetNormal(mesh)
    ds = Measure('ds', domain=mesh, subdomain_data=bdries)
    
    for tag, value in neumann_bcs:
        system += -inner(value, v)*ds(tag)

    # Dirichlet bcs go onto the matrix
    bcs_D = [DirichletBC(V, value, bdries, tag) for tag, value in bcs['dirichlet']]

    # Discrete problem
    a, L = lhs(system), rhs(system)
    A, b = assemble_system(a, L, bcs_D)

    # NOTE: this uses direct solver, might be too slow but we know what
    # to do then
    uh = Function(V)
    solve(A, uh.vector(), b)

    return uh


def mms_ale(kappa_value):
    '''Method of manufactured solutions on [0, 1]^2'''
    mesh = UnitSquareMesh(mpi_comm_self(), 2, 2)  # Dummy
    V = VectorFunctionSpace(mesh, 'CG', 2)
    # Coefficient space
    S = FunctionSpace(mesh, 'DG', 0)
    kappa = Function(S)

    u = Function(V)
    f = -div(kappa*grad(u))
    flux = lambda u, n, kappa=kappa: dot(kappa*grad(u), n)
    
    # What we want to substitute
    x, y, kappa_ = sp.symbols('x y kappa')
    # Expressions
    u_ = sp.Matrix([sp.sin(pi*(x + y)), sp.cos(pi*(x + y))])
    
    subs = {u: u_, kappa: kappa_}  # Function are replaced by symbols

    as_expr = lambda t: ulfy.Expression(t, subs=subs, degree=4, kappa=kappa_value)
    
    # Solution
    u_exact = as_expr(u)
    # Forcing
    f = as_expr(f)
    #  4
    # 1 2
    #  3 so that
    normals = [Constant((-1, 0)), Constant((1, 0)), Constant((0, -1)), Constant((0, 1))]
    fluxes = [as_expr(flux(u, n)) for n in normals]

    return {'solution': u_exact,
            'force': f,
            'fluxes': fluxes}
           
# --------------------------------------------------------------------

if __name__ == '__main__':

    kappa_value = 1E0
    data = mms_ale(kappa_value=kappa_value)
    
    u_exact  = data['solution']
    forcing = data['force']
    fluxes = dict(enumerate(data['fluxes'], 1))

    # Taylor-Hood
    Velm = VectorElement('Lagrange', triangle, 1)

    parameters = {'kappa': Constant(kappa_value)}
    for n in (4, 8, 16, 32, 64):
        mesh = UnitSquareMesh(n, n)
        # Setup similar to coupled problem ...
        bdries = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
        CompiledSubDomain('near(x[0], 0)').mark(bdries, 1)
        CompiledSubDomain('near(x[0], 1)').mark(bdries, 2)
        CompiledSubDomain('near(x[1], 0)').mark(bdries, 3)
        CompiledSubDomain('near(x[1], 1)').mark(bdries, 4)

        # Top bottom are prescribed displacement, and on sides we put
        # stresses
        bcs = {'dirichlet': [(3, u_exact), (4, u_exact)],
               'neumann': [(1, fluxes[1]), (2, fluxes[2])]}

        V = FunctionSpace(mesh, Velm)
        uh = solve_ale(V, f=forcing, bdries=bdries, bcs=bcs,
                       parameters=parameters)
        # Errors
        eu = errornorm(u_exact, uh, 'H1', degree_rise=2)
        print('|u-uh|_1', eu)

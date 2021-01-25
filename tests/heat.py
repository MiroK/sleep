from dolfin import *
import itertools
import operator
import sympy as sp


# 1D heat problem
#
# u_t - u_xx = f 
# u(t, 0) = g
# du/dn(t, 1) = h
#
# u(0, x) = u_0(x)


def solve_heat(V, f, u_0, bdries, bcs, parameters):
    '''Return temperature final time'''
    mesh = V.mesh()

    needed = set(bdries.array()) - set((0, ))    

    assert all(k in ('neumann', 'dirichlet') for k in bcs)

    dirichlet_bcs = bcs.get('dirichlet', ())
    neumann_bcs = bcs.get('neumann', ())      
    # Tuple of pairs (tag, boundary value) is expected
    dirichlet_tags = set(item[0] for item in dirichlet_bcs)
    neumann_tags = set(item[0] for item in neumann_bcs)

    tags = (dirichlet_tags, neumann_tags)
    for this, that in itertools.combinations(tags, 2):
        if this and that: assert not this & that

    assert needed == reduce(operator.or_, tags)

    # Collect bc values for possible temporal update in the integration
    bdry_expressions = sum(([item[1] for item in bc]
                            for bc in (dirichlet_bcs, neumann_bcs)),
                           [])

    # FEM ---
    u, v = TrialFunction(V), TestFunction(V)
    # For weak form also time step is needed
    dt = Constant(parameters['dt'])

    # Previous solutions
    u_0 = interpolate(u_0, V)
    
    # Elasticity
    a = inner(u/dt, v)*dx + inner(grad(u), grad(v))*dx
    L = inner(u_0/dt, v)*dx + inner(f, v)*dx

    # Handle natural bcs
    n = FacetNormal(mesh)
    ds = Measure('ds', domain=mesh, subdomain_data=bdries)

    for tag, value in neumann_bcs:
        L += inner(value, v)*ds(tag)
        
    # Displacement bcs and flux bcs go on the system
    bcs_strong = [DirichletBC(V, value, bdries, tag) for tag, value in dirichlet_bcs]

    # Discrete problem
    assembler = SystemAssembler(a, L, bcs_strong)

    A, b = PETScMatrix(), PETScVector()
    # Assemble once and setup solver for it
    assembler.assemble(A)
    solver = LUSolver(A, 'umfpack')

    # Temporal integration loop
    T0 = parameters['T0']
    for k in range(parameters['nsteps']):
        T0 += dt(0)

        # Update source if possible
        for foo in bdry_expressions + [f]:
            hasattr(foo, 't') and setattr(foo, 't', T0)
            hasattr(foo, 'time') and setattr(foo, 'time', T0)

        assembler.assemble(b)
        solver.solve(u_0.vector(), b)

    return u_0, T0


def mms_heat(parameters):
    '''Method of manufactured solutions on [0, 1]^2'''
    x, t = sp.symbols('x[0] time')
    u = sp.sin(sp.pi*x)*sp.exp(1-t)

    f = u.diff(t, 1) - u.diff(x, 2)
    du_dn_left = -1*u.diff(x, 1)
    du_dn_right = u.diff(x, 1)

    make_expr = lambda foo: Expression(sp.printing.ccode(foo).replace('M_PI', 'pi'),
                                       degree=4, time=0)

    return {'solution': (make_expr(u), ),
            'force': (make_expr(f), ),
            'tractions': (make_expr(du_dn_left), make_expr(du_dn_right))}
           
# --------------------------------------------------------------------

if __name__ == '__main__':

    parameters = {}
    data = mms_heat(parameters)
    
    u_exact,  = data['solution']
    f, = data['force']
    fluxes = dict(enumerate(data['tractions'], 1))

    Velm = FiniteElement('Lagrange', interval, 1)

    parameters['dt'] = 1E-3
    parameters['nsteps'] = int(1E0/parameters['dt'])
    parameters['T0'] = 0.
    for n in (128, 256, 512, 1024):
        mesh = UnitIntervalMesh(n)
        # Setup similar to coupled problem ...
        bdries = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
        CompiledSubDomain('near(x[0], 0)').mark(bdries, 1)
        CompiledSubDomain('near(x[0], 1)').mark(bdries, 2)

        # Reset time
        for things in data.values():
            for thing in things:
                thing.time = 0.
        
        bcs = {'dirichlet': [(1, u_exact)], 'neumann': [(2, fluxes[2])]}

        V = FunctionSpace(mesh, Velm)
        # Get the initial conditions
        u_0 = interpolate(u_exact, V)

        ans = solve_heat(V, f, u_0, bdries=bdries, bcs=bcs, parameters=parameters)

        u_h, time = ans
        u_exact.time = time
        # Errors
        eu = errornorm(u_exact, u_h, 'H1', degree_rise=2)
        print('|u-u_h|_1', eu)


    # ----
    # We really need to resolve the spatial error to see O(dt) of Backward
    # Euler
    n = 16000
    mesh = UnitIntervalMesh(n)
    # Setup similar to coupled problem ...
    bdries = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
    CompiledSubDomain('near(x[0], 0)').mark(bdries, 1)
    CompiledSubDomain('near(x[0], 1)').mark(bdries, 2)


    dt = 1E-2
    for _ in range(4):
        parameters['dt'] = dt
        parameters['nsteps'] = int(1E0/parameters['dt'])
        parameters['T0'] = 0.

        # Reset time
        for things in data.values():
            for thing in things:
                thing.time = 0.
        
        bcs = {'dirichlet': [(1, u_exact)], 'neumann': [(2, fluxes[2])]}

        V = FunctionSpace(mesh, Velm)
        # Get the initial conditions
        u_0 = interpolate(u_exact, V)

        ans = solve_heat(V, f, u_0, bdries=bdries, bcs=bcs, parameters=parameters)

        u_h, time = ans
        u_exact.time = time
        # Errors
        eu = errornorm(u_exact, u_h, 'H1', degree_rise=2)
        print(dt, '|u-u_h|_1', eu)

        dt /= 2.

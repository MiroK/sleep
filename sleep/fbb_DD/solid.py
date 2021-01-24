from dolfin import *
import itertools
import operator
import sympy as sp
import ulfy  # https://github.com/MiroK/ulfy

# We solve Biot in (0, T) x Omega
# 
# -div(sigma(eta, p)) = f1                             [elasticity]
# kappa^{-1}*u + grad(p) = 0                           [Darcy]
# d/dt(s0*p + alpha*div(eta)) + div(u) = f2
#
# where sigma(eta, p) = 2*mu*sym(grad(eta)) + lambda*div(eta)*I - alpha*p*I
#
# with initial condition on p and eta and with following bcs:
#
# Elasticity:
#  1) displacement - sets the entire displacement
#  2) traction - sets sigma.n
#
# Darcy
# 1) pressure - sets pressure
# 2) flux - constrols u.n
#

def solve_ale(W, f1, f2, eta_0, p_0, fbdries, bcs, parameters):
    '''Return displacement, percolation velocity, pressure'''
    # NOTE: this is time dependent problem which we solve with 
    # parameters['dt'] for parameters['nsteps'] time steps and to 
    # update time in f1, f2 or the expressions bcs the physical time is
    # set as parameters['T0'] + dt*(k-th step)
    mesh = V.mesh()

    needed = set(bdries.array()) - set((0, ))    
    # Validate elasticity bcs
    bcs_E = bcs['elasticity']
    assert all(k in ('displacement', 'traction') for k in bcs_E)

    displacement_bcs = bcs_E.get('displacement', ())  
    traction_bcs = bcs_E.get('traction', ())
    # Tuple of pairs (tag, boundary value) is expected
    displacement_tags = set(item[0] for item in displacement_bcs)
    traction_tags = set(item[0] for item in traction_bcs)

    tags = (displacement_tags, traction_tags)
    for this, that in itertools.combinations(tags, 2):
        if this and that: assert not this & that

    assert needed == reduce(operator.or_, tags)

    # Validate Darcy bcs
    bcs_D = bcs['darcy']
    assert all(k in ('pressure', 'flux') for k in bcs_D)

    pressure_bcs = bcs_D.get('pressure', ())  
    flux_bcs = bcs_D.get('flux', ())
    # Tuple of pairs (tag, boundary value) is expected
    pressure_tags = set(item[0] for item in pressure_bcs)
    flux_tags = set(item[0] for item in flux_bcs)

    tags = (pressure_tags, flux_tags)
    for this, that in itertools.combinations(tags, 2):
        if this and that: assert not this & that

    assert needed == reduce(operator.or_, tags)

    # Collect bc values for possible temporal update in the integration
    bdry_expressions = sum(([item[1] for item in bc]
                            for bc in (displacement_bcs, traction_bcs, pressure_bcs, flux_bcs)),
                           [])

    # FEM ---
    eta, u, p = TrialFunctions(W)
    phi, v, q = TestFunctions(W)
    assert len(eta.ufl_shape) == 1 and len(u.ufl_shape) == 1 and len(p.ufl_shape) == 0

    # Material parameters
    kappa, mu, lmbda, alpha, s0 = (parameters[k] for k in ('kappa', 'mu', 'lmbda', 'alpha', 's0'))
    # For weak form also time step is needed
    dt = parameters['dt']

    # Elasticity
    system = (2*mu*inner(sym(grad(eta)), sym(grad(phi)))*dx +
              inner(lmbda*div(eta), div(phi))*dx -
              inner(p, alpha*div(phi))*dx-
              inner(f1, phi)*dx)

    # Darcy
    system += (1/kappa)*inner(u, v)*dx - inner(p, div(v))*dx
    # Mass conservation with backward Euler
    system += (inner(s0*p, q)*dx + inner(alpha*div(eta), q)*dx + dt*inner(div(u), q)*dx
               -dt*inner(f2, q)*dx
               -inner(s0*p_0, q)*dx - inner(alpha*div(eta_0)*dx))

    # Handle natural bcs
    n = FacetNormal(mesh)
    ds = Measure('ds', domain=mesh, subdomain_data=bdries)
    # For elasticity
    for tag, value in tractions_bcs:
        system += -inner(value, phi)*ds(tag)
    # For Darcy
    for tag, value in pressure_bcs:
        system += inner(value, dot(v, n))*ds(tag)
        
    # Displacement bcs and flux bcs go on the system
    bcs_strong = [DirichletBC(W.sub(0), value, bdries, tag) for tag, value in displacement_bcs]
    bcs_strong.extend([DirichletBC(W.sub(1), value, bdries, tag) for tag, value in flux_bcs])

    # Discrete problem
    a, L = lhs(system), rhs(system)
    assembler = SystemAssembler(a, L, bcs_strong)

    A, b = PETScMatrix(), PETScVector()
    # Assemble once and setup solver for it
    assembler.assemble(A)
    solver = LUSolver(A, 'umfpack')

    wh = Function(W)
    # Temporal integration loop
    T0 = parameters['T0']
    for k in range(parameters['nsteps']):
        T0 += dt(0)
        # Update source if possible
        for foo in bdry_expressions:
            hasattr(foo, 't') and setattr(foo, 't', T0)
            hasattr(foo, 'time') and setattr(foo, 'time', T0)

        assembler.assemble(b)
        solver.solve(wh.vector(), b)

        eta_h, u_h, p_h = wh.split(deepcopy=True)

        eta_0.assign(eta_h)
        p_0.assign(p_h)
        
    return eta_h, u_h, p_h


def mms_solid(parameters):
    '''Method of manufactured solutions on [0, 1]^2'''
    mesh = UnitSquareMesh(mpi_comm_self(), 2, 2)  
    V = VectorFunctionSpace(mesh, 'CG', 2)  # Displacement, Flux
    Q = FunctionSpace(mesh, 'CG', 2)  # Pressure
    # Coefficient space
    S = FunctionSpace(mesh, 'DG', 0)

    kappa, mu, lmbda, alpha, s0 = [Function(S) for _ in range(5)]

    eta = Function(V)  # Displacement
    p = Function(Q)  # Pressure
    u = -kappa*grad(p)  # Define flux

    sigma_p = lambda eta, p: 2*mu*sym(grad(eta)) + lmbda*div(eta)*Identity(len(eta)) - alpha*p*Idenity(len(eta))
    
    f1 = -div(sigma_p(eta, p))
    
    # NOTE: we do not have temporal derivative but if we assume eta and p
    # have foo(x, y)*bar(t) then in mass conservation we will have
    # [s0*p + alpha*div(eta)]*dbar(t)
    time = Function(S)
    
    # What we want to substitute
    x, y, kappp_, mu_, lmbda_, alpha_, s0_ = sp.symbols('x y kappa mu lmbda alpha s0')
    # Expressions
    u_ = sp.Matrix([sp.sin(pi*(x + y))*sin(),
                    sp.cos(pi*(x + y))])
    
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

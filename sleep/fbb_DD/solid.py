from dolfin import *
from functools import reduce
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

def solve_solid(W, f1, f2, eta_0, p_0, bdries, bcs, parameters):
    '''Return displacement, percolation velocity, pressure and final time'''
    info('Solving Biot for %d unknowns' % W.dim())
    # NOTE: this is time dependent problem which we solve with 
    # parameters['dt'] for parameters['nsteps'] time steps and to 
    # update time in f1, f2 or the expressions bcs the physical time is
    # set as parameters['T0'] + dt*(k-th step)
    mesh = W.mesh()

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

    assert needed == reduce(operator.or_, tags), (needed, reduce(operator.or_, tags))

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
    dt = Constant(parameters['dt'])

    # Previous solutions
    wh_0 = Function(W)
    assign(wh_0.sub(0), interpolate(eta_0, W.sub(0).collapse()))    
    assign(wh_0.sub(2), interpolate(p_0, W.sub(2).collapse()))
    eta_0, u_0, p_0 = split(wh_0)

    # Elasticity
    a = (2*mu*inner(sym(grad(eta)), sym(grad(phi)))*dx +
         inner(lmbda*div(eta), div(phi))*dx -
         inner(p, alpha*div(phi))*dx)

    L = inner(f1, phi)*dx
              
    # Darcy
    a += (1/kappa)*inner(u, v)*dx - inner(p, div(v))*dx
         
    # Mass conservation with backward Euler
    a += inner(s0*p, q)*dx + inner(alpha*div(eta), q)*dx + dt*inner(div(u), q)*dx
    L += dt*inner(f2, q)*dx + inner(s0*p_0, q)*dx + inner(alpha*div(eta_0), q)*dx         

    # Handle natural bcs
    n = FacetNormal(mesh)
    ds = Measure('ds', domain=mesh, subdomain_data=bdries)
    # For elasticity
    for tag, value in traction_bcs:
        L += inner(value, phi)*ds(tag)
    # For Darcy
    for tag, value in pressure_bcs:
        L += -inner(value, dot(v, n))*ds(tag)
        
    # Displacement bcs and flux bcs go on the system
    bcs_strong = [DirichletBC(W.sub(0), value, bdries, tag) for tag, value in displacement_bcs]
    bcs_strong.extend([DirichletBC(W.sub(1), value, bdries, tag) for tag, value in flux_bcs])

    # Discrete problem
    assembler = SystemAssembler(a, L, bcs_strong)

    A, b = PETScMatrix(), PETScVector()
    # Assemble once and setup solver for it
    assembler.assemble(A)
    solver = LUSolver(A, 'mumps')

    # Temporal integration loop
    T0 = parameters['T0']
    for k in range(parameters['nsteps']):
        # Update source if possible
        for foo in bdry_expressions + [f1, f2]:
            hasattr(foo, 't') and setattr(foo, 't', T0)
            hasattr(foo, 'time') and setattr(foo, 'time', T0)

        assembler.assemble(b)
        solver.solve(wh_0.vector(), b)
        T0 += dt(0)
        
        k % 10 == 0 and info('  Biot at step (%d, %g) |uh|=%g' % (k, T0, wh_0.vector().norm('l2')))    

    eta_h, u_h, p_h = wh_0.split(deepcopy=True)
        
    return eta_h, u_h, p_h, T0


def mms_solid(parameters):
    '''Method of manufactured solutions on [0, 1]^2'''
    mesh = UnitSquareMesh(2, 2)  
    V = VectorFunctionSpace(mesh, 'CG', 2)  # Displacement, Flux
    Q = FunctionSpace(mesh, 'CG', 2)  # Pressure
    # Coefficient space
    S = FunctionSpace(mesh, 'DG', 0)

    kappa, mu, lmbda, alpha, s0 = [Function(S) for _ in range(5)]

    eta = Function(V)  # Displacement
    p = Function(Q)  # Pressure
    u = -kappa*grad(p)  # Define flux

    sigma_p = lambda eta, p: 2*mu*sym(grad(eta)) + lmbda*div(eta)*Identity(len(eta)) - alpha*p*Identity(len(eta))
    
    f1 = -div(sigma_p(eta, p))
    
    # NOTE: we do not have temporal derivative but if we assume eta and p
    # have foo(x, y)*bar(t) then in mass conservation we will have
    # [s0*p + alpha*div(eta)]*dbar(t)
    
    # What we want to substitute
    x, y, kappa_, mu_, lmbda_, alpha_, s0_ = sp.symbols('x y kappa mu lmbda alpha s0')
    time_ = sp.Symbol('time')
    # Expressions
    eta_ = sp.Matrix([sp.sin(pi*(x + y)),
                      sp.cos(pi*(x + y))])*sp.exp(1-time_)

    p_ = sp.cos(pi*(x-y))*sp.exp(1-time_)
    # - here is dbar(t)
    f2 = -(s0*p + alpha*div(eta)) + div(u)

    subs = {eta: eta_, p: p_,
            kappa: kappa_, mu: mu_, lmbda: lmbda_, alpha: alpha_, s0: s0_}

    as_expr = lambda t: ulfy.Expression(t, subs=subs, degree=4,
                                        kappa=parameters['kappa'],
                                        mu=parameters['mu'],
                                        lmbda=parameters['lmbda'],
                                        alpha=parameters['alpha'],
                                        s0=parameters['s0'],
                                        time=0.0)
    
    # Solution
    eta_exact, u_exact, p_exact = map(as_expr, (eta, u, p))
    # Forcing
    f1, f2 = as_expr(f1), as_expr(f2)
    #  4
    # 1 2
    #  3 so that
    normals = [Constant((-1, 0)), Constant((1, 0)), Constant((0, -1)), Constant((0, 1))]
    tractions = [as_expr(dot(sigma_p(eta, p), n)) for n in normals]

    return {'solution': (eta_exact, u_exact, p_exact),
            'force': (f1, f2),
            'tractions': tractions}
           
# --------------------------------------------------------------------

if __name__ == '__main__':

    parameters = {'kappa': Constant(1),
                  'mu': Constant(1),
                  'lmbda': Constant(1),
                  'alpha': Constant(1),
                  's0': Constant(1)}
    data = mms_solid(parameters)
    
    eta_exact, u_exact, p_exact  = data['solution']
    f1, f2 = data['force']
    tractions = dict(enumerate(data['tractions'], 1))

    Eelm = VectorElement('Lagrange', triangle, 2)
    Velm = FiniteElement('Raviart-Thomas', triangle, 1)
    Qelm = FiniteElement('Discontinuous Lagrange', triangle, 0)

    Welm = MixedElement([Eelm, Velm, Qelm])

    parameters['dt'] = 1E-6
    parameters['nsteps'] = int(1E-4/parameters['dt'])
    parameters['T0'] = 0.
    for n in (4, 8, 16, 32, 64):
        mesh = UnitSquareMesh(n, n)
        # Setup similar to coupled problem ...
        bdries = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
        CompiledSubDomain('near(x[0], 0)').mark(bdries, 1)
        CompiledSubDomain('near(x[0], 1)').mark(bdries, 2)
        CompiledSubDomain('near(x[1], 0)').mark(bdries, 3)
        CompiledSubDomain('near(x[1], 1)').mark(bdries, 4)

        # Reset time
        for things in data.values():
            for thing in things:
                thing.time = 0.
        
        # Elasticity: displacement bottom, traction for rest
        # Darcy: pressure bottom and top, flux on sides
        bcs = {'elasticity': {'displacement': [(3, eta_exact)],
                              'traction': [(1, tractions[1]), (2, tractions[2]), (4, tractions[4])]},
               'darcy': {'pressure': [(1, p_exact), (2, p_exact)],
                         'flux': [(3, u_exact), (4, u_exact)]}}

        # Get the initial conditions
        E = FunctionSpace(mesh, Eelm)
        Q = FunctionSpace(mesh, Qelm)

        eta_0 = interpolate(eta_exact, E)
        p_0 = interpolate(p_exact, Q)

        W = FunctionSpace(mesh, Welm)
        ans = solve_solid(W, f1, f2, eta_0, p_0, bdries=bdries, bcs=bcs,
                          parameters=parameters)

        eta_h, u_h, p_h, time = ans
        eta_exact.time, u_exact.time, p_exact.time = (time, )*3
        # Errors
        e_eta = errornorm(eta_exact, eta_h, 'H1', degree_rise=2)        
        e_u = errornorm(u_exact, u_h, 'Hdiv', degree_rise=2)
        e_p = errornorm(p_exact, p_h, 'L2', degree_rise=2)        
        print('|eta-eta_h|_1', e_eta, '|u-uh|_div', e_u, '|p-ph|_0', e_p, '#dofs', W.dim())


    # ----

    n = 128
    mesh = UnitSquareMesh(n, n)
    # Setup similar to coupled problem ...
    bdries = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
    CompiledSubDomain('near(x[0], 0)').mark(bdries, 1)
    CompiledSubDomain('near(x[0], 1)').mark(bdries, 2)
    CompiledSubDomain('near(x[1], 0)').mark(bdries, 3)
    CompiledSubDomain('near(x[1], 1)').mark(bdries, 4)

    dt = 1E-2
    for _ in range(4):
        parameters['dt'] = dt
        parameters['nsteps'] = int(1E-1/dt)
        parameters['T0'] = 0.
        
        # Reset time
        for things in data.values():
            for thing in things:
                thing.time = 0.

        # Elasticity: displacement bottom, traction for rest
        # Darcy: pressure bottom and top, flux on sides
        bcs = {'elasticity': {'displacement': [(3, eta_exact)],
                              'traction': [(1, tractions[1]), (2, tractions[2]), (4, tractions[4])]},
               'darcy': {'pressure': [(1, p_exact), (2, p_exact)],
                         'flux': [(3, u_exact), (4, u_exact)]}}

        # Get the initial conditions
        E = FunctionSpace(mesh, Eelm)
        Q = FunctionSpace(mesh, Qelm)

        eta_0 = interpolate(eta_exact, E)
        p_0 = interpolate(p_exact, Q)

        W = FunctionSpace(mesh, Welm)
        ans = solve_solid(W, f1, f2, eta_0, p_0, bdries=bdries, bcs=bcs,
                          parameters=parameters)

        eta_h, u_h, p_h, time = ans
        eta_exact.time, u_exact.time, p_exact.time = (time, )*3
        # Errors
        e_eta = errornorm(eta_exact, eta_h, 'H1', degree_rise=2)        
        e_u = errornorm(u_exact, u_h, 'Hdiv', degree_rise=2)
        e_p = errornorm(p_exact, p_h, 'L2', degree_rise=2)
        print('\tdt=%.2E T=%.2f nsteps=%d' %  (dt, p_exact.time, parameters['nsteps']))
        print('|eta-eta_h|_1', e_eta, '|u-uh|_div', e_u, '|p-ph|_0', e_p, '#dofs', W.dim())
        
        dt = dt/2

    # for i, (num, true) in enumerate(zip((eta_h, u_h, p_h), (eta_exact, u_exact, p_exact))):
    #     File('x%d_num.pvd' % i) << num
    #     File('x%d_true.pvd' % i) << interpolate(true, num.function_space())

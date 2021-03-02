import sleep.fbb_DD.cylindrical as cyl
from dolfin import *
from functools import reduce
import itertools
import operator
import sympy as sp
import ulfy  # https://github.com/MiroK/ulfy

# Problem
# 
# rho*(du/dt) = div(sigma(u, p)) + rho*f  with sigma(u, p) = mu*grad(u) - p*I
# -div(u) = 0 in Omega
#
# with following bcs:
# 1) velocity boundary sets velocity vector
# 2) Traction boundary sets sigma.n
# 3) Pressure boundary sets pressure, leaving grad(u).n free?
#
# is solved on FE space W

def solve_fluid(W, u_0, f, bdries, bcs, parameters):
    '''Return velocity and pressure'''
    info('Solving Stokes for %d unknowns' % W.dim())
    mesh = W.mesh()
    assert mesh.geometry().dim() == 2
    # Let's see about boundary conditions - they need to be specified on
    # every boundary.
    assert all(k in ('velocity', 'traction', 'pressure') for k in bcs)
    # The tags must be found in bdries
    velocity_bcs = bcs.get('velocity', ())  
    traction_bcs = bcs.get('traction', ())
    pressure_bcs = bcs.get('pressure', ())
    # Tuple of pairs (tag, boundary value) is expected
    velocity_tags = set(item[0] for item in velocity_bcs)
    traction_tags = set(item[0] for item in traction_bcs)
    pressure_tags = set(item[0] for item in pressure_bcs)

    tags = (velocity_tags, traction_tags, pressure_tags)
    # Boundary conditions must be on distinct domains
    for this, that in itertools.combinations(tags, 2):
        if this and that: assert not this & that

    # With convention that 0 bdries are inside all the exterior bdries must
    # be given conditions in bcs
    needed = set(bdries.array()) - set((0, ))
    assert needed == reduce(operator.or_, tags)

    # Things, which we might want to update in the time loop
    bdry_expressions = sum(([item[1] for item in bc]
                            for bc in (velocity_bcs, traction_bcs, pressure_tags)),
                           [])

    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)

    assert len(u.ufl_shape) == 1 and len(p.ufl_shape) == 0

    wh_0 = Function(W)
    assign(wh_0.sub(0), interpolate(u_0, W.sub(0).collapse()))
    
    u_0, p_0 = split(wh_0)
    # All but bc terms
    mu = parameters['mu']
    rho = parameters['rho']
    dt = Constant(parameters['dt'])
    
    system = (rho*inner((u - u_0)/dt, v)*dx +
              inner(mu*grad(u), grad(v))*dx - inner(p, div(v))*dx
              -inner(q, div(u))*dx - rho*inner(f, v)*dx)

    # Handle natural bcs
    n = FacetNormal(mesh)
    ds = Measure('ds', domain=mesh, subdomain_data=bdries)
    
    for tag, value in traction_bcs:
        system += -inner(value, v)*ds(tag)

    for tag, value in pressure_bcs:
        # impose normal component of normal traction do be equal to the imposed pressure
        system += -inner(-value, dot(v, n))*ds(tag) # note the minus sign before the pressure term in the stress
        if parameters.get('add_grad_un_pressure_bdry', False):
            system += -inner(dot(grad(u), n), v)*ds(tag)

    # Velocity bcs go onto the matrix
    bcs_D = [DirichletBC(W.sub(0), value, bdries, tag) for tag, value in velocity_bcs]

    a, L = lhs(system), rhs(system)
    # Discrete problem
    assembler = SystemAssembler(a, L, bcs_D)

    A, b = PETScMatrix(), PETScVector()
    # Assemble once and setup solver for it
    assembler.assemble(A)
    solver = LUSolver(A, 'mumps')

    # Temporal integration loop
    T0 = parameters['T0']
    for k in range(parameters['nsteps']):
        # Update source if possible
        for foo in bdry_expressions + [f]:
            hasattr(foo, 't') and setattr(foo, 't', T0)
            hasattr(foo, 'time') and setattr(foo, 'time', T0)

        assembler.assemble(b)
        solver.solve(wh_0.vector(), b)
        k % 100 == 0 and info('  Stokes at step (%d, %g) |wh|=%g' % (k, T0, wh_0.vector().norm('l2')))    

        T0 += dt(0)        

    u_0, p_0 = wh_0.split(deepcopy='True')
        
    return u_0, p_0, T0



def mms_stokes(mu_value, rho_value):
    '''Method of manufactured solutions on [0, 1]^2'''
    mesh = UnitSquareMesh(2, 2)  # Dummy
    V = FunctionSpace(mesh, 'CG', 2)
    # Coefficient space
    S = FunctionSpace(mesh, 'DG', 0)
    mu, rho, alpha, time = Function(S), Function(S), Function(S), Function(S)

    # Auxiliry function for defining Stokes velocity (to make it div-free)
    phi = Function(V)
    
    u = as_vector((phi.dx(1), -phi.dx(0)))*exp(1 - alpha**2*time)
    p = Function(V)  # Pressure

    stress = lambda u, p, mu=mu: mu*grad(u) - p*Identity(len(u))
    
    # Forcing for Stokes
    f = -div(stress(u, p))/rho + u*(-alpha**2)
    # We will also need traction on boundaries with normal n
    traction = lambda u, p, n: dot(stress(u, p), n)
    
    # What we want to substitute
    x, y, mu_, rho_, alpha_, time_ = sp.symbols('x y mu rho alpha time')
    # Expressions
    phi_ = sp.sin(pi*(x + y))
    p_ = sp.sin(2*pi*x)*sp.sin(4*pi*y)
    
    subs = {phi: phi_, p: p_, mu: mu_, rho: rho_, alpha: alpha_, time: time_}  # Function are replaced by symbols

    as_expr = lambda t: ulfy.Expression(t, subs=subs, degree=4,
                                        mu=mu_value, rho=rho_value,
                                        alpha=0.2, time=0.)
    
    # Solution
    u_exact, p_exact = map(as_expr, (u, p))
    # Forcing
    f = as_expr(f)
    # With u, p we have things for Dirichlet and pressure boudaries. For
    # traction assume boundaries labeled in order
    #  4
    # 1 2
    #  3 so that
    normals = [Constant((-1, 0)), Constant((1, 0)), Constant((0, -1)), Constant((0, 1))]
    tractions = [as_expr(traction(u, p, n)) for n in normals]
    # Finally tangential part
    R = Constant(((0, -1), (1, 0)))

    normal_comps = [as_expr(dot(n, traction(u, p, n))) for n in normals]
    tangent_comps = [as_expr(dot(dot(R, n), traction(u, p, n))) for n in normals]
    stress_components = zip(normal_comps, tangent_comps)

    return {'solution': (u_exact, p_exact),
            'force': f,
            'tractions': tractions,
            'stress_components': stress_components}


# def solve_fluid_cyl(W, f, bdries, bcs, parameters):
#     '''Return velocity and pressure'''
#     info('Solving Stokes for %d unknowns' % W.dim())
#     mesh = W.mesh()
#     assert mesh.geometry().dim() == 2
#     # Let's see about boundary conditions - they need to be specified on
#     # every boundary.
#     assert all(k in ('velocity', 'traction', 'pressure') for k in bcs)
#     # The tags must be found in bdries
#     velocity_bcs = bcs.get('velocity', ())  
#     traction_bcs = bcs.get('traction', ())
#     pressure_bcs = bcs.get('pressure', ())
#     # Tuple of pairs (tag, boundary value) is expected
#     velocity_tags = set(item[0] for item in velocity_bcs)
#     traction_tags = set(item[0] for item in traction_bcs)
#     pressure_tags = set(item[0] for item in pressure_bcs)

#     tags = (velocity_tags, traction_tags, pressure_tags)
#     # Boundary conditions must be on distinct domains
#     for this, that in itertools.combinations(tags, 2):
#         if this and that: assert not this & that

#     # With convention that 0 bdries are inside all the exterior bdries must
#     # be given conditions in bcs
#     needed = set(bdries.array()) - set((0, ))
#     assert needed == reduce(operator.or_, tags)
                 
#     u, p = TrialFunctions(W)
#     v, q = TestFunctions(W)

#     assert len(u.ufl_shape) == 1 and len(p.ufl_shape) == 0

#     z, r = SpatialCoordinate(mesh)
#     # All but bc terms
#     mu = parameters['mu']
#     system = (inner(mu*cyl.GradAxisym(u), cyl.GradAxisym(v))*r*dx - inner(p, cyl.DivAxisym(v))*r*dx
#               -inner(q, cyl.DivAxisym(u))*r*dx - inner(f, v)*r*dx)

#     # Handle natural bcs
#     ds = Measure('ds', domain=mesh, subdomain_data=bdries)
    
#     for tag, value in traction_bcs:
#         system += -inner(value, v)*r*ds(tag)

#     n = FacetNormal(mesh)
#     for tag, value in pressure_bcs:
#         # impose normal component of normal traction do be equal to the imposed pressure
#         system += -inner(-value, dot(v, n))*r*ds(tag) # note the minus sign before the pressure term in the stress

#         # Extend here to 3-vector
#         if parameters.get('add_grad_un_pressure_bdry', False):        
#             n_ = as_vector((n[0], n[1], Constant(0)))
#             v_ = as_vector((v[0], v[1], Constant(0)*v[0]))
#             system += -inner(dot(cyl.GradAxisym(u), n_), v_)*ds(tag)

#     # Velocity bcs go onto the matrix
#     bcs_D = [DirichletBC(W.sub(0), value, bdries, tag) for tag, value in velocity_bcs]

#     # Discrete problem
#     a, L = lhs(system), rhs(system)
#     A, b = assemble_system(a, L, bcs_D)

#     # NOTE: this uses direct solver, might be too slow but we know what
#     # to do then
#     wh = Function(W)
#     timer = Timer('Stokes')
#     solve(A, wh.vector(), b)
#     info('  Stokes done in %f secs |uh|=%g' % (timer.stop(), wh.vector().norm('l2')))    

#     return wh.split(deepcopy=True)


# --------------------------------------------------------------------

if __name__ == '__main__':
    mu_value, rho_value = 2, 5
    data = mms_stokes(mu_value=mu_value, rho_value=rho_value)

    # -- Convergence test case
    u_exact, p_exact = data['solution']
    forcing = data['force']
    tractions = dict(enumerate(data['tractions'], 1))

    # Taylor-Hood
    Velm = VectorElement('Lagrange', triangle, 2)
    Qelm = FiniteElement('Lagrange', triangle, 1)
    Welm = MixedElement([Velm, Qelm])

    # -- Spatial convergence 
    dt = 1E-2
    parameters = {'mu': Constant(mu_value),
                  'rho': Constant(rho_value),
                  'dt': dt,
                  'nsteps': int(1E-1/dt),
                  'T0': 0.}

    for n in (4, 8, 16, 32, 64):
        # Reset time
        for thing in itertools.chain((u_exact, p_exact, forcing), data['tractions']):
            hasattr(thing, 'time') and setattr(thing, 'time', parameters['T0'])
        u_0 = u_exact
        
        mesh = UnitSquareMesh(n, n)
        # Setup similar to coupled problem ...
        bdries = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
        CompiledSubDomain('near(x[0], 0)').mark(bdries, 1)
        CompiledSubDomain('near(x[0], 1)').mark(bdries, 2)
        CompiledSubDomain('near(x[1], 0)').mark(bdries, 3)
        CompiledSubDomain('near(x[1], 1)').mark(bdries, 4)

        bcs = {'velocity': [(1, u_exact), (2, u_exact), (3, u_exact)],
               'traction': [(4, tractions[4])]}

        W = FunctionSpace(mesh, Welm)
        uh, ph, T0 = solve_fluid(W, u_0=u_0, f=forcing, bdries=bdries, bcs=bcs,
                                 parameters=parameters)

        u_exact.time = T0
        p_exact.time = T0
        # Errors
        eu = errornorm(u_exact, uh, 'H1', degree_rise=2)
        ep = errornorm(p_exact, ph, 'L2', degree_rise=2)

        print('|u-uh|_1', eu, '|p-ph|_0', ep)

    # Temporal
    n = 128
    mesh = UnitSquareMesh(n, n)
    # Setup similar to coupled problem ...
    bdries = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
    CompiledSubDomain('near(x[0], 0)').mark(bdries, 1)
    CompiledSubDomain('near(x[0], 1)').mark(bdries, 2)
    CompiledSubDomain('near(x[1], 0)').mark(bdries, 3)
    CompiledSubDomain('near(x[1], 1)').mark(bdries, 4)

    bcs = {'velocity': [(1, u_exact), (2, u_exact), (3, u_exact)],
           'traction': [(4, tractions[4])]}
    
    dt = 1E-2
    parameters = {'mu': Constant(mu_value),
                  'rho': Constant(rho_value),
                  'T0': 0.}

    for k in range(4):
        parameters['dt'] = dt
        parameters['nsteps'] = int(1E-1/dt)
        
        # Reset time
        for thing in itertools.chain((u_exact, p_exact, forcing), data['tractions']):
            hasattr(thing, 'time') and setattr(thing, 'time', parameters['T0'])
        u_0 = u_exact

        W = FunctionSpace(mesh, Welm)
        uh, ph, T0 = solve_fluid(W, u_0=u_0, f=forcing, bdries=bdries, bcs=bcs,
                                 parameters=parameters)

        u_exact.time = T0
        p_exact.time = T0
        # Errors
        eu = errornorm(u_exact, uh, 'H1', degree_rise=2)
        ep = errornorm(p_exact, ph, 'L2', degree_rise=2)

        print('At T0 = {} with dt = {} |u-uh|_1 = {}, |p-ph|_0 = {}'.format(T0, dt, eu, ep))
        dt /= 2.

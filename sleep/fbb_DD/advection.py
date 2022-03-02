import sleep.fbb_DD.cylindrical as cyl
from dolfin import *
from functools import reduce
import numpy as np
import itertools
import operator
import sympy as sp
import ulfy  # https://github.com/MiroK/ulfy

# Problem
# 
# - d c / dt + v.grad(c) - kappa*Delta(c) = f
#
# with following bcs:
# 1) concentration (Dirichlet): c = gD
# 2) flux (Neumann): kappa*dot(grad(c), n) = gN
#
# is solved on FE space W

def solve_adv_diff_cyl(W, velocity,phi, f, c_0, phi_0, bdries, bcs, parameters):
    '''Return concentration field'''
    info('Solving advection-diffusion for %d unknowns' % W.dim())

    assert W.ufl_element().family() == 'Lagrange'
    mesh = W.mesh()
    assert mesh.geometry().dim() == 2
    assert velocity.ufl_shape == (2, )
    # Let's see about boundary conditions - they need to be specified on
    # every boundary.
    assert all(k in ('concentration', 'flux') for k in bcs)
    # The tags must be found in bdries
    dirichlet_bcs = bcs.get('concentration', ())  
    neumann_bcs = bcs.get('flux', ())
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

    # Collect bc values for possible temporal update in the integration
    bdry_expressions = sum(([item[1] for item in bc]
                            for bc in (dirichlet_bcs, neumann_bcs)),
                           [])
    
    c, psi = TrialFunction(W), TestFunction(W)
    assert psi.ufl_shape == (), psi.ufl_shape

    kappa = parameters['kappa']

    dt = Constant(parameters['dt'])

    # Extend velocity to 3d as GradAxisym(scalar) is 3-vector
    velocity = as_vector((velocity[0],
                          velocity[1],
                          Constant(0)))
    # ... however, we are still in 2d
    z, r = SpatialCoordinate(mesh)

    c_0 = interpolate(c_0, W)

    # Should fix error in PVSBRAIN if use psi0 (problem with phi_0 I guess)
    try:
        psi0 = TestFunction(phi_0.function_space())
    except AttributeError:
        psi0 = TestFunction(W)

    phi_0 = interpolate(phi_0, W)
    #
    #(inner((phi - phi_0)/dt, psi)*r*dx 
    # (inner(phi*c/dt, psi)*r*dx - inner(phi_0*c_0/dt, psi0)*r*dx
    # Usual backward Euler
    system =(inner(phi*c/dt, psi)*r*dx - inner(phi_0*c_0/dt, psi)*r*dx + dot(velocity, cyl.GradAxisym(c))*psi*r*dx +
              inner(kappa*phi*cyl.GradAxisym(c), cyl.GradAxisym(psi))*r*dx - inner(f, psi)*r*dx)
    
    # SUPG stabilization
    #if parameters.get('supg', False):
    if False :
        info(' Adding SUPG stabilization')
        h = CellDiameter(mesh)

        mag = sqrt(inner(velocity, velocity))
        # See https://www.hindawi.com/journals/jam/2018/4259634/ for stab.
        stab = 1/(4*kappa/h/h + 2*mag/h)

        # Test the residuum againt        
        # Correct with porosity phi ? 
        system += stab*inner(((1/dt)*(phi*c - phi_0*c_0) - kappa*phi*div(cyl.GradAxisym(c)) + dot(velocity, cyl.GradAxisym(c))) - f,
                             dot(velocity, cyl.GradAxisym(psi)))*r*dx(degree=10)

    # Handle natural bcs
    n = FacetNormal(mesh)
    ds = Measure('ds', domain=mesh, subdomain_data=bdries)
    for tag, value in neumann_bcs:
        system += -inner(value, psi)*r*ds(tag)

    # velocity bcs go onto the matrix
    Q = FunctionSpace(mesh, 'DG', 0)
    q = TestFunction(Q)
    for tag in dirichlet_tags:
        #extend to 3D
        n_ = as_vector((n[0], n[1], Constant(0)))
        v_n = assemble(inner(q, dot(velocity, n_))*ds(tag))
        if np.any(v_n.get_local() > 0):
            print('Dirichlet bcs on outflow?', v_n.max(), tag)
    
    bcs_D = [DirichletBC(W, value, bdries, tag) for tag, value in dirichlet_bcs]

    a, L = lhs(system), rhs(system)
    # Discrete problem
    assembler = SystemAssembler(a, L, bcs_D)

    A, b = PETScMatrix(), PETScVector()
    # Assemble once and setup solver for it
    assembler.assemble(A)
    solver = LUSolver(A, 'mumps')
    timer = Timer('Adv-diff')
    # Temporal integration loop
    T0 = parameters['T0']
    for k in range(parameters['nsteps']):
        # Update source if possible
        for foo in bdry_expressions + [f]:
            hasattr(foo, 't') and setattr(foo, 't', T0)
            hasattr(foo, 'time') and setattr(foo, 'time', T0)

        assembler.assemble(b)
        solver.solve(c_0.vector(), b)
        k % 100 == 0 and info('  Adv-Diff at step (%d, %g) |c_h|=%g' % (k, T0, c_0.vector().norm('l2')))    
        T0 += dt(0)
    
    info('  Adv-diff done in %f secs ' % (timer.stop()))     
    return c_0, T0


def solve_adv_diff(W, velocity,phi, f, c_0, phi_0, bdries, bcs, parameters):
    '''Return concentration field'''
    info('Solving advection-diffusion for %d unknowns' % W.dim())
    assert W.ufl_element().family() == 'Lagrange'
    mesh = W.mesh()
    assert mesh.geometry().dim() == 2
    assert velocity.ufl_shape == (2, )
    # Let's see about boundary conditions - they need to be specified on
    # every boundary.
    assert all(k in ('concentration', 'flux') for k in bcs)
    # The tags must be found in bdries
    dirichlet_bcs = bcs.get('concentration', ())  
    neumann_bcs = bcs.get('flux', ())
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

    # Collect bc values for possible temporal update in the integration
    bdry_expressions = sum(([item[1] for item in bc]
                            for bc in (dirichlet_bcs, neumann_bcs)),
                           [])
    
    c, psi = TrialFunction(W), TestFunction(W)
    assert psi.ufl_shape == (), psi.ufl_shape

    kappa = parameters['kappa']

    dt = Constant(parameters['dt'])

    c_0 = interpolate(c_0, W)

    #correction for dt

    try:
        psi0 = TestFunction(phi_0.function_space())
    except AttributeError:
        psi0 = TestFunction(W)

    phi_0 = interpolate(phi_0, W)


    # Usual backward Euler
    system = (inner(phi*c/dt, psi)*dx - inner(phi_0*c_0/dt, psi)*dx+ dot(velocity, grad(c))*psi*dx +
              kappa*phi*inner(grad(c), grad(psi))*dx - inner(f, psi)*dx)
    
    # SUPG stabilization
    if parameters.get('supg', False):
        info(' Adding SUPG stabilization')
        h = CellDiameter(mesh)

        mag = sqrt(inner(velocity, velocity))
        # See https://www.hindawi.com/journals/jam/2018/4259634/ for stab.
        stab = 1/(4*kappa/h/h + 2*mag/h)

        # Test the residuum againt         
        system += stab*inner(((1/dt)*(c - c_0) - kappa*div(grad(c)) + dot(velocity, grad(c))) - f,
                             dot(velocity, grad(psi)))*dx(degree=10)

    # Handle natural bcs
    n = FacetNormal(mesh)
    ds = Measure('ds', domain=mesh, subdomain_data=bdries)
    
    for tag, value in neumann_bcs:
        system += -inner(value, psi)*ds(tag)

    # velocity bcs go onto the matrix
    Q = FunctionSpace(mesh, 'DG', 0)
    q = TestFunction(Q)
    for tag in dirichlet_tags:
        v_n = assemble(inner(q, dot(velocity, n))*ds(tag))
        if np.any(v_n.get_local() > 0):
            print('Dirichlet bcs on outflow?', v_n.max(), tag)
    
    bcs_D = [DirichletBC(W, value, bdries, tag) for tag, value in dirichlet_bcs]

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
        solver.solve(c_0.vector(), b)
        k % 100 == 0 and info('  Adv-Diff at step (%d, %g) |c_h|=%g' % (k, T0, c_0.vector().norm('l2')))    

        T0 += dt(0)
        
    return c_0, T0


def mms_ad(kappa_value):
    '''Method of manufactured solutions on [0, 1]^2'''
    mesh = UnitSquareMesh(2, 2)  # Dummy
    V = FunctionSpace(mesh, 'CG', 2)
    # Coefficient space
    S = FunctionSpace(mesh, 'DG', 0)
    kappa, alpha = Function(S), Function(S)
    # Velocity
    W = VectorFunctionSpace(mesh, 'CG', 1)
    velocity = Function(W)

    c = Function(V)
    # foo*exp(1-alpha*time) so that d / dt gives us -alpha*foo
    f = -alpha*c + dot(velocity, grad(c)) - kappa*div(grad(c))

    flux = lambda c, n, kappa=kappa: dot(kappa*grad(c), n)
    
    # What we want to substitute
    x, y, kappa_ = sp.symbols('x y kappa')
    time_, alpha_ = sp.symbols('time alpha')
    velocity_ = sp.Matrix([-(y-0.5), (x-0.5)])

    # Expressions
    c_ = sp.sin(pi*(x + y))*sp.exp(1-alpha_*time_)

    subs = {c: c_, kappa: kappa_, alpha: alpha_, velocity: velocity_}
    as_expr = lambda t: ulfy.Expression(t, subs=subs, degree=4,
                                        kappa=kappa_value, alpha=2,
                                        time=0.)

    # Solution
    c_exact, velocity = map(as_expr, (c, velocity))
    # Forcing
    f = as_expr(f)

    normals = [Constant((-1, 0)), Constant((1, 0)), Constant((0, -1)), Constant((0, 1))]
    fluxes = [as_expr(flux(c, n)) for n in normals]

    return {'solution': c_exact,
            'forcing': f,
            'fluxes': fluxes,
            'velocity': velocity}
           
# --------------------------------------------------------------------

if __name__ == '__main__':

    kappa_value = 3E0
    data = mms_ad(kappa_value=kappa_value)
    
    c_exact = data['solution']
    velocity = data['velocity']
    forcing = data['forcing']
    
    fluxes = dict(enumerate(data['fluxes'], 1))

    # Taylor-Hood
    Welm = FiniteElement('Lagrange', triangle, 1)

    dt = 1E-2
    parameters = {'kappa': Constant(kappa_value),    
                  'dt': dt,
                  'nsteps': int(1E-1/dt),
                  'T0': 0.}
        
    # Spatial convergences
    history = []
    for n in (4, 8, 16, 32, 64):
        # Reset time
        for thing in (c_exact, velocity, forcing):
            hasattr(thing, 'time') and setattr(thing, 'time', parameters['T0'])
        c_0 = c_exact
        
        mesh = UnitSquareMesh(n, n)
        # Setup similar to coupled problem ...
        bdries = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
        CompiledSubDomain('near(x[0], 0)').mark(bdries, 1)
        CompiledSubDomain('near(x[0], 1)').mark(bdries, 2)
        CompiledSubDomain('near(x[1], 0)').mark(bdries, 3)
        CompiledSubDomain('near(x[1], 1)').mark(bdries, 4)

        bcs = {'concentration': [], #[(t, c_exact) for t in (1, )],
               'flux': [(t, fluxes[t]) for t in (1, 2, 3, 4)]}

        W = FunctionSpace(mesh, Welm)
        c_h, T = solve_adv_diff_cyl(W, velocity=velocity, f=forcing, c_0=c_0,
                                      bdries=bdries, bcs=bcs, parameters=parameters)
        # Errors
        print(c_exact.time, T)
        c_exact.time = T
        e = errornorm(c_exact, c_h, 'H1', degree_rise=2)

        print('|c-ch|_1', e)
        history.append((mesh.hmin(), e))
    print(history)

    # Temporal cvrg
    n = 128
    mesh = UnitSquareMesh(n, n)
    # Setup similar to coupled problem ...
    bdries = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
    CompiledSubDomain('near(x[0], 0)').mark(bdries, 1)
    CompiledSubDomain('near(x[0], 1)').mark(bdries, 2)
    CompiledSubDomain('near(x[1], 0)').mark(bdries, 3)
    CompiledSubDomain('near(x[1], 1)').mark(bdries, 4)

    bcs = {'concentration': [(t, c_exact) for t in (1, 2)],
           'flux': [(t, fluxes[t]) for t in (3, 4)]}

    W = FunctionSpace(mesh, Welm)

    history = []
    dt = 0.2
    for _ in range(4):
        dt = dt / 2
        parameters = {'kappa': Constant(kappa_value),    
                      'dt': dt,
                      'nsteps': int(1./dt),
                      'T0': 0.,
                      'supg': True}
        
        # Reset time
        for thing in (c_exact, velocity, forcing):
            hasattr(thing, 'time') and setattr(thing, 'time', parameters['T0'])
        c_0 = c_exact
    
        c_h, T = solve_adv_diff(W, velocity=velocity, f=forcing, c_0=c_0,
                                  bdries=bdries, bcs=bcs, parameters=parameters)
        # Errors
        print(c_exact.time, T)
        c_exact.time = T
        e = errornorm(c_exact, c_h, 'H1', degree_rise=2)

        # File('ph.pvd') << c_h
        # c_h.vector().axpy(-1, interpolate(c_exact, c_h.function_space()).vector())
        # File('p.pvd') << c_h

        print('@ T = {} with dt = {} |c-ch|_1 = {}'.format(T, dt, e))
        history.append((dt, mesh.hmin(), e))
    print(history)
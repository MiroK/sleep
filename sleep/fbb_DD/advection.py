from dolfin import *
from functools import reduce
import itertools
import operator
import sympy as sp
import ulfy  # https://github.com/MiroK/ulfy

# Problem
# 
# - d phi / dt + v.grad(phi) - kappa*Delta(phi) = f
#
# with following bcs:
# 1) concentration (Dirichlet): phi = gD
# 2) flux (Neumann): kappa*dot(grad(phi), n) = gN
#
# is solved on FE space W

def solve_adv_diff(W, velocity, f, phi_0, bdries, bcs, parameters):
    '''Return concentration field'''
    info('Solving advection-diffusion for %d unknowns' % W.dim())
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
    
    phi, psi = TrialFunction(W), TestFunction(W)
    assert psi.ufl_shape == (), psi.ufl_shape

    kappa = Constant(parameters['kappa'])

    dt = Constant(parameters['dt'])

    phi_0 = interpolate(phi_0, W)
    # Usual backward Euler
    system = (inner((phi - phi_0)/dt, psi)*dx + dot(velocity, grad(phi))*psi*dx +
              kappa*inner(grad(phi), grad(psi))*dx - inner(f, psi)*dx)
    
    # SUPG stabilization
    if parameters.get('supg', False):
        info(' Adding SUPG stabilization')
        h = CellDiameter(mesh)

        mag = sqrt(inner(velocity, velocity))
        stab = 1/(4*kappa/h/h + 2*mag/h)

        # Test the residuum againt         
        system += stab*inner(((1/dt)*(phi - phi_0) - kappa*div(grad(phi)) + dot(velocity, grad(phi))) - f,
                             dot(velocity, grad(psi)))*dx(degree=10)

    # Handle natural bcs
    n = FacetNormal(mesh)
    ds = Measure('ds', domain=mesh, subdomain_data=bdries)
    
    for tag, value in neumann_bcs:
        system += -inner(value, psi)*ds(tag)

    # velocity bcs go onto the matrix
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
        solver.solve(phi_0.vector(), b)
        k % 100 == 0 and info('  Adv-Diff at step (%d, %g) |phi_h|=%g' % (k, T0, phi_0.vector().norm('l2')))    

        T0 += dt(0)
        
    return phi_0, T0


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

    phi = Function(V)
    # foo*exp(1-alpha*time) so that d / dt gives us -alpha*foo
    f = -alpha*phi + dot(velocity, grad(phi)) - kappa*div(grad(phi))

    flux = lambda phi, n, kappa=kappa: dot(kappa*grad(phi), n)
    
    # What we want to substitute
    x, y, kappa_ = sp.symbols('x y kappa')
    time_, alpha_ = sp.symbols('time alpha')
    velocity_ = sp.Matrix([-(y-0.5), (x-0.5)])

    # Expressions
    phi_ = sp.sin(pi*(x + y))*sp.exp(1-alpha_*time_)

    subs = {phi: phi_, kappa: kappa_, alpha: alpha_, velocity: velocity_}
    as_expr = lambda t: ulfy.Expression(t, subs=subs, degree=4,
                                        kappa=kappa_value, alpha=2,
                                        time=0.)

    # Solution
    phi_exact, velocity = map(as_expr, (phi, velocity))
    # Forcing
    f = as_expr(f)

    normals = [Constant((-1, 0)), Constant((1, 0)), Constant((0, -1)), Constant((0, 1))]
    fluxes = [as_expr(flux(phi, n)) for n in normals]

    return {'solution': phi_exact,
            'forcing': f,
            'fluxes': fluxes,
            'velocity': velocity}
           
# --------------------------------------------------------------------

if __name__ == '__main__':

    kappa_value = 3E0
    data = mms_ad(kappa_value=kappa_value)
    
    phi_exact = data['solution']
    velocity = data['velocity']
    forcing = data['forcing']
    
    fluxes = dict(enumerate(data['fluxes'], 1))

    # Taylor-Hood
    Welm = FiniteElement('Lagrange', triangle, 1)

    dt = 1E-3
    parameters = {'kappa': Constant(kappa_value),    
                  'dt': dt,
                  'nsteps': int(1E-1/dt),
                  'T0': 0.}
        
    # # Spatial convergences
    # history = []
    # for n in (4, 8, 16, 32, 64):
    #     # Reset time
    #     for thing in (phi_exact, velocity, forcing):
    #         hasattr(thing, 'time') and setattr(thing, 'time', parameters['T0'])
    #     phi_0 = phi_exact
        
    #     mesh = UnitSquareMesh(n, n)
    #     # Setup similar to coupled problem ...
    #     bdries = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
    #     CompiledSubDomain('near(x[0], 0)').mark(bdries, 1)
    #     CompiledSubDomain('near(x[0], 1)').mark(bdries, 2)
    #     CompiledSubDomain('near(x[1], 0)').mark(bdries, 3)
    #     CompiledSubDomain('near(x[1], 1)').mark(bdries, 4)

    #     bcs = {'concentration': [], #[(t, phi_exact) for t in (1, )],
    #            'flux': [(t, fluxes[t]) for t in (1, 2, 3, 4)]}

    #     W = FunctionSpace(mesh, Welm)
    #     phi_h, T = solve_adv_diff(W, velocity=velocity, f=forcing, phi_0=phi_0,
    #                               bdries=bdries, bcs=bcs, parameters=parameters)
    #     # Errors
    #     print(phi_exact.time, T)
    #     phi_exact.time = T
    #     e = errornorm(phi_exact, phi_h, 'H1', degree_rise=2)

    #     print('|c-ch|_1', e)
    #     history.append((mesh.hmin(), e))
    # print(history)

    n = 128
    mesh = UnitSquareMesh(n, n)
    # Setup similar to coupled problem ...
    bdries = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
    CompiledSubDomain('near(x[0], 0)').mark(bdries, 1)
    CompiledSubDomain('near(x[0], 1)').mark(bdries, 2)
    CompiledSubDomain('near(x[1], 0)').mark(bdries, 3)
    CompiledSubDomain('near(x[1], 1)').mark(bdries, 4)

    bcs = {'concentration': [(t, phi_exact) for t in (1, 2)],
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
        for thing in (phi_exact, velocity, forcing):
            hasattr(thing, 'time') and setattr(thing, 'time', parameters['T0'])
        phi_0 = phi_exact
    
        phi_h, T = solve_adv_diff(W, velocity=velocity, f=forcing, phi_0=phi_0,
                                  bdries=bdries, bcs=bcs, parameters=parameters)
        # Errors
        print(phi_exact.time, T)
        phi_exact.time = T
        e = errornorm(phi_exact, phi_h, 'H1', degree_rise=2)

        File('ph.pvd') << phi_h
        phi_h.vector().axpy(-1, interpolate(phi_exact, phi_h.function_space()).vector())
        File('p.pvd') << phi_h

        print('@ T = {} with dt = {} |c-ch|_1 = {}'.format(T, dt, e))
        history.append((dt, mesh.hmin(), e))
    print(history)
    
# Does this work for 1d heat equation?

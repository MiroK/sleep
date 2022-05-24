import sleep.fbb_DD.cylindrical as cyl
from dolfin import *
from functools import reduce
import itertools
import operator
import sympy as sp
import ulfy  # https://github.com/MiroK/ulfy

# Problem
# 
# -div(sigma(u, p)) = f  with sigma(u, p) = mu*grad(u) - p*I
# -div(u) = 0 in Omega
#
# with following bcs:
# 1) velocity boundary sets velocity vector
# 2) Traction boundary sets sigma.n
# 3) Pressure boundary sets pressure, leaving grad(u).n free?
#
# is solved on FE space W

def solve_fluid(W, u_0, f,  bdries, bcs, parameters):
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
                 
    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)


    assert len(u.ufl_shape) == 1 and len(p.ufl_shape) == 0
    # All but bc terms
    mu = parameters['mu']
    system = (inner(mu*grad(u), grad(v))*dx - inner(p, div(v))*dx
              -inner(q, div(u))*dx - inner(f, v)*dx)

    # Handle natural bcs
    n = FacetNormal(mesh)
    ds = Measure('ds', domain=mesh, subdomain_data=bdries)
    
    for tag, value in traction_bcs:
        system += -inner(value, v)*ds(tag)

    # For the pressure condition : 
    # We need to impose the normal component of the normal traction on the inlet and outlet to be the pressures we want on each surface
    # and force the normal component of the grad u to be zero

    for tag, value in pressure_bcs:
        # impose normal component of normal traction do be equal to the imposed pressure
        system += -inner(-value, dot(v, n))*ds(tag) # note the minus sign before the pressure term in the stress
        # impose  dot(n, grad(u))=0
        system += inner(dot(grad(u), n), v)*ds(tag)


    # Velocity bcs go onto the matrix
    bcs_D = [DirichletBC(W.sub(0), value, bdries, tag) for tag, value in velocity_bcs]

    # Discrete problem
    a, L = lhs(system), rhs(system)
    A, b = assemble_system(a, L, bcs_D)

    # NOTE: this uses direct solver, might be too slow but we know what
    # to do then
    wh = Function(W)
    timer = Timer('Stokes')
    solve(A, wh.vector(), b)
    info('  Stokes done in %f secs |uh|=%g' % (timer.stop(), wh.vector().norm('l2')))    

    return wh.split(deepcopy=True)


def mms_stokes(mu_value):
    '''Method of manufactured solutions on [0, 1]^2'''
    mesh = UnitSquareMesh(2, 2)  # Dummy
    V = FunctionSpace(mesh, 'CG', 2)
    # Coefficient space
    S = FunctionSpace(mesh, 'DG', 0)
    mu = Function(S)

    # Auxiliry function for defining Stokes velocity (to make it div-free)
    phi = Function(V)
    
    u = as_vector((phi.dx(1), -phi.dx(0)))  # Velocity
    p = Function(V)  # Pressure

    stress = lambda u, p, mu=mu: mu*grad(u) - p*Identity(len(u))
    
    # Forcing for Stokes
    f = -div(stress(u, p))
    # We will also need traction on boundaries with normal n
    traction = lambda u, p, n: dot(stress(u, p), n)
    
    # What we want to substitute
    x, y, mu_ = sp.symbols('x y mu')
    # Expressions
    phi_ = sp.sin(pi*(x + y))
    p_ = sp.sin(2*pi*x)*sp.sin(4*pi*y)
    
    subs = {phi: phi_, p: p_, mu: mu_}  # Function are replaced by symbols

    as_expr = lambda t: ulfy.Expression(t, subs=subs, degree=4, mu=mu_value)
    
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


def solve_fluid_cyl(W,u_0, f,  bdries, bcs, parameters):
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
                 
    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)

    assert len(u.ufl_shape) == 1 and len(p.ufl_shape) == 0

    z, r = SpatialCoordinate(mesh)
    # All but bc terms
    mu = parameters['mu']
    system = (inner(mu*cyl.GradAxisym(u), cyl.GradAxisym(v))*r*dx - inner(p, cyl.DivAxisym(v))*r*dx
              -inner(q, cyl.DivAxisym(u))*r*dx - inner(f, v)*r*dx)

    # Handle natural bcs
    ds = Measure('ds', domain=mesh, subdomain_data=bdries)
    
    for tag, value in traction_bcs:
        system += -inner(value, v)*r*ds(tag)

    n = FacetNormal(mesh)
    for tag, value in pressure_bcs:
        # impose normal component of normal traction do be equal to the imposed pressure
        system += -inner(-value, dot(v, n))*r*ds(tag) # note the minus sign before the pressure term in the stress

        # Extend here to 3-vector
        if parameters.get('add_grad_un_pressure_bdry', False):        
            n_ = as_vector((n[0], n[1], Constant(0)))
            v_ = as_vector((v[0], v[1], Constant(0)*v[0]))
            system += -inner(dot(cyl.GradAxisym(u), n_), v_)*ds(tag)

    # Velocity bcs go onto the matrix
    bcs_D = [DirichletBC(W.sub(0), value, bdries, tag) for tag, value in velocity_bcs]

    # Discrete problem
    a, L = lhs(system), rhs(system)
    A, b = assemble_system(a, L, bcs_D)

    # NOTE: this uses direct solver, might be too slow but we know what
    # to do then
    wh = Function(W)
    timer = Timer('Stokes')
    solve(A, wh.vector(), b)
    info('  Stokes done in %f secs |uh|=%g' % (timer.stop(), wh.vector().norm('l2')))    

    return wh.split(deepcopy=True)


# --------------------------------------------------------------------

if __name__ == '__main__':

    mu_value = 2E0
    data = mms_stokes(mu_value=mu_value)

    # -- Convergence test case
    u_exact, p_exact = data['solution']
    forcing = data['force']
    tractions = dict(enumerate(data['tractions'], 1))
    stress_components = dict(enumerate(data['stress_components'], 1))

    # Taylor-Hood
    Velm = VectorElement('Lagrange', triangle, 2)
    Qelm = FiniteElement('Lagrange', triangle, 1)
    Welm = MixedElement([Velm, Qelm])

    material_parameters = {'mu': Constant(mu_value)}
    for n in (4, 8, 16, 32, 64):
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
        uh, ph = solve_fluid(W, f=forcing, bdries=bdries, bcs=bcs,
                             parameters=material_parameters)
        # Errors
        eu = errornorm(u_exact, uh, 'H1', degree_rise=2)
        ep = errornorm(p_exact, ph, 'L2', degree_rise=2)

        print('|u-uh|_1', eu, '|p-ph|_0', ep)

    # -- Are we able to get Poisseuile flow?
    forcing = Constant((0, 0))
    tractions = dict(enumerate(data['tractions'], 1))
    stress_components = dict(enumerate(data['stress_components'], 1))

    # Taylor-Hood
    Velm = VectorElement('Lagrange', triangle, 2)
    Qelm = FiniteElement('Lagrange', triangle, 1)
    Welm = MixedElement([Velm, Qelm])

    material_parameters = {'mu': Constant(2)}
    for n in (4, 8, 16, 32, 64):
        mesh = UnitSquareMesh(n, n)

        x, y = SpatialCoordinate(mesh)

        mu = material_parameters['mu']
        u_true = as_vector((y*(1-y)/2/mu,
                            Constant(0)*x))
        p_true = 1-x
        # Setup similar to coupled problem ...
        bdries = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
        CompiledSubDomain('near(x[0], 0)').mark(bdries, 1)
        CompiledSubDomain('near(x[0], 1)').mark(bdries, 2)
        CompiledSubDomain('near(x[1], 0)').mark(bdries, 3)
        CompiledSubDomain('near(x[1], 1)').mark(bdries, 4)

        bcs = {'velocity': [(3, Constant((0, 0))), (4, Constant((0, 0)))],
               'pressure': [(1, Constant(1)), (2, Constant(0))]}

        W = FunctionSpace(mesh, Welm)
        uh, ph = solve_fluid(W, f=forcing, bdries=bdries, bcs=bcs,
                             parameters=material_parameters)
        # Errors
        eu = sqrt(abs(assemble(inner(grad(u_true - uh), grad(u_true - uh))*dx)))
        ep = sqrt(abs(assemble(inner(p_true - ph, p_true - ph)*dx)))

        print('|u-uh|_1', eu, '|p-ph|_0', ep)


    # -- Now in cylindrical coordinates
    rIn, rOut, length = 0.5, 1, 3
    delta_P = 2

    forcing = Constant((0, 0))
                         
    # Taylor-Hood
    Velm = VectorElement('Lagrange', triangle, 2)
    Qelm = FiniteElement('Lagrange', triangle, 1)
    Welm = MixedElement([Velm, Qelm])

    material_parameters = {'mu': Constant(mu_value)}
    bcs = {'velocity': [(3, Constant((0, 0))), (4, Constant((0, 0)))],
           # With the following traction bcs Poisseuille works
           #'traction': [(1, Constant((0, 0))), (2, Constant((-delta_P, 0)))],
           'pressure': [(1, Constant(0)), (2, Constant(-delta_P))]
    }

    eu0, ep0, h0 = None, None, None
    for n in (16, 32, 64, 128):
        mesh = RectangleMesh(Point(0, rIn), Point(length, rOut), n, n)

        bdries = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
        CompiledSubDomain('near(x[0], 0)').mark(bdries, 1)
        CompiledSubDomain('near(x[0], L)', L=length).mark(bdries, 2)
        CompiledSubDomain('near(x[1], rIn)', rIn=rIn).mark(bdries, 3)
        CompiledSubDomain('near(x[1], rOut)', rOut=rOut).mark(bdries, 4)

        assert set(bdries.array()) == set((0, 1, 2, 3, 4))
        
        z, r = SpatialCoordinate(mesh)

        delta_P, L, rIn, rOut, mu = map(Constant, (delta_P, length, rIn, rOut, mu_value))
        p_true = -delta_P*z/length
        u_true = as_vector((delta_P/L/4/mu*(ln(r/rOut)/ln(rOut/rIn)*(rOut**2 - rIn**2) + (rOut**2 - r**2)),
                            Constant(0)*p_true))

        W = FunctionSpace(mesh, Welm)
        uh, ph = solve_fluid_cyl(W, f=forcing, bdries=bdries, bcs=bcs,
                                 parameters=material_parameters)

        # Errors
        eu = sqrt(assemble(inner(grad(u_true - uh), grad(u_true - uh))*r*dx))
        ep = sqrt(assemble(inner(ph - p_true, ph - p_true)*r*dx))
        h = float(uh.function_space().mesh().hmin())
        
        if eu0 is not None:
            rate_u = ln(eu/eu0)/ln(h/h0)
            rate_p = ln(ep/ep0)/ln(h/h0)
        else:
            rate_u, rate_p = -1, -1
        
        print('|u-uh|_1 = {:.4E}[{:1.2f}] |p-ph|_0 = {:.4E}[{:1.2f}]'.format(eu, rate_u, ep, rate_p))
        eu0, ep0, h0 = eu, ep, h

    File('uh.pvd') << uh
    File('u.pvd') << project(u_true, uh.function_space())
    File('ph.pvd') << ph
    File('p.pvd') << project(p_true, ph.function_space())

from dolfin import *
from functools import reduce
import itertools
import operator
import sympy as sp
import ulfy  # https://github.com/MiroK/ulfy

# Problem
# 
# -div(sigma(u, p)) = f  with sigma(u, p) = 2*mu*sym(grad(u)) - p*I
# -div(u) = 0 in Omega
#
# with following bcs:
# 1) velocity boundary sets velocity vector
# 2) Traction boundary sets sigma.n
# 3) Pressure boundary set sigma.n.n (pressure part) and (sigma.n).t
#
# is solved on FE space W

def solve_fluid(W, f, bdries, bcs, parameters):
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
    system = (inner(2*mu*sym(grad(u)), sym(grad(v)))*dx - inner(p, div(v))*dx-inner(q, div(u))*dx - inner(f, v)*dx)

    # Handle natural bcs
    n = FacetNormal(mesh)
    ds = Measure('ds', domain=mesh, subdomain_data=bdries)
    
    sigma = lambda u, p, mu=mu: 2*mu*sym(grad(u)) - p*Identity(len(u))

    for tag, value in traction_bcs:
        system += -inner(value, v)*ds(tag)

    # For the pressure condition : 
    # We need to impose the normal component of the normal traction on the inlet and outlet to be the pressures we want on each surface
    # and force the flow to be normal to those surfaces <-- this is incompatible with the moving top and bottom surfaces
    # so force the normal component of the grad u to be zero
    bcs_D =[]
    for tag, value in pressure_bcs:
        # impose normal component of normal traction do be equal to the imposed pressure
        system += inner(value, dot(v, n))*ds(tag)
        # impose  dot(n, grad(u))=0
        system += -inner(Constant((0, 0)), v)*ds(tag)

    # velocity bcs go onto the matrix
    for tag, value in velocity_bcs:
        bcs_D.append(DirichletBC(W.sub(0), value, bdries, tag))

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

    stress = lambda u, p, mu=mu: 2*mu*sym(grad(u)) - p*Identity(len(u))
    
    # Forcing for Stokes
    f = -div(stress(u, p))
    # We will also need traction on boundaries with normal n
    traction = lambda u, p, n: dot(n, stress(u, p))
    
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
           
# --------------------------------------------------------------------

if __name__ == '__main__':

    mu_value = 1E0
    data = mms_stokes(mu_value=mu_value)
    
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

        # Have velocity on the bottom, traction on top. In real application
        # I guess left and right sigma.n.n ~ pressure and we set tangential
        # to 0. Alternative is to have u.t in combination with sigma.n.n
        # but if domain not aligned with axis u.t is tricky - Nitsche?
        bcs = {'velocity': [(3, u_exact)],
               'traction': [(4, tractions[4])],
               'pressure': [(1, stress_components[1]), (2, stress_components[2])]}

        W = FunctionSpace(mesh, Welm)
        uh, ph = solve_fluid(W, f=forcing, bdries=bdries, bcs=bcs,
                             parameters=material_parameters)
        # Errors
        eu = errornorm(u_exact, uh, 'H1', degree_rise=2)
        ep = errornorm(p_exact, ph, 'L2', degree_rise=2)

        print('|u-uh|_1', eu, '|p-ph|_0', ep)

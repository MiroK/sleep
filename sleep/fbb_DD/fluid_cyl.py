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

def Grad(u):
    # z r theta 
    uz, ur = u[0], u[1]
    z, r = SpatialCoordinate(u.ufl_domain().ufl_cargo())
    return as_matrix(((uz.dx(0), uz.dx(1), 0),
                      (ur.dx(0), ur.dx(1), 0),
                      (0,        0,     ur/r)))

def Div(u):
    return tr(Grad(u))


strain = lambda u, mu: 2*mu*sym(Grad(u))
sigma = lambda u, p, mu: strain(u, mu) - p*Identity(3)


def solve_fluid(W, f, bdries, bcs, parameters):
    '''Return velocity and pressure'''
    # Fluid solver in cylindrical coordinates. Assuming that the solution 
    # is axisymmetric we reduce to 2d (z0, z1) x (r0, r1) rectangular domain
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
    system = (inner(2*mu*sym(Grad(u)), sym(Grad(v)))*r*dx - inner(p, Div(v))*r*dx
              -inner(q, Div(u))*r*dx - inner(f, v)*r*dx)

    # Handle natural bcs
    n = FacetNormal(mesh)
    ds = Measure('ds', domain=mesh, subdomain_data=bdries)

    for tag, value in traction_bcs:
        system += -inner(value, v)*r*ds(tag)

    # For the pressure condition : 
    # We need to impose the normal component of the normal traction on the inlet and outlet to be the pressures we want on each surface
    # and force the flow to be normal to those surfaces <-- this is incompatible with the moving top and bottom surfaces
    # so force the normal component of the grad u to be zero

    for tag, value in pressure_bcs:
        # impose normal component of normal traction do be equal to the imposed pressure
        system += -inner(-value, dot(v, n))*r*ds(tag) # note the minus sign before the pressure term in the stress
        # impose  dot(n, grad(u))=0
        #system += -inner(Constant((0, 0)), v)*ds(tag)

    # velocity bcs go onto the matrix
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

    mu_value = 1E0

    rIn, rOut, length = 0.5, 1, 3
    delta_P = 2

    p_exact = Expression('delta_P*(length-x[0])/length', degree=5,
                         delta_P=delta_P, length=length)

    u_exact = Expression(('delta_P/L/4/mu*(std::log(x[1]/rOut)/std::log(rOut/rIn)*(rOut*rOut - rIn*rIn) + (rOut*rOut - x[1]*x[1]))',
                          '0'),
                         rIn=rIn, rOut=rOut, delta_P=delta_P, mu=mu_value, L=length,
                         degree=4)
    
    forcing = Constant((0, 0))
                         
    # Taylor-Hood
    Velm = VectorElement('Lagrange', triangle, 2)
    Qelm = FiniteElement('Lagrange', triangle, 1)
    Welm = MixedElement([Velm, Qelm])

    material_parameters = {'mu': Constant(mu_value)}
    for n in (4, 8, 16, 32, 64):
        mesh = RectangleMesh(Point(0, rIn), Point(length, rOut), n, n)
        # Setup similar to coupled problem ...
        bdries = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
        CompiledSubDomain('near(x[0], 0)').mark(bdries, 1)
        CompiledSubDomain('near(x[0], L)', L=length).mark(bdries, 2)
        CompiledSubDomain('near(x[1], rIn)', rIn=rIn).mark(bdries, 3)
        CompiledSubDomain('near(x[1], rOut)', rOut=rOut).mark(bdries, 4)

        assert set(bdries.array()) == set((0, 1, 2, 3, 4))
        
        bcs = {'velocity': [(3, u_exact), (4, u_exact)],
               'traction': [(1, p_exact*Constant((1, 0))),
                            (2, p_exact*Constant((1, 0)))]}

        W = FunctionSpace(mesh, Welm)
        uh, ph = solve_fluid(W, f=forcing, bdries=bdries, bcs=bcs,
                             parameters=material_parameters)
        # Errors
        eu = errornorm(u_exact, uh, 'H1', degree_rise=2)
        ep = errornorm(p_exact, ph, 'L2', degree_rise=2)

        print('|u-uh|_1', eu, '|p-ph|_0', ep)

        u = interpolate(u_exact, uh.function_space())
        p = interpolate(p_exact, ph.function_space())

        tress = sigma(u, p, Constant(mu_value))
        train = strain(u, Constant(mu_value))
        ds_ = Measure('ds', domain=mesh, subdomain_data=bdries)
        n = Constant((1, 0, 0))
        print(assemble((dot(tress, n)[0])*ds_(2)), assemble(p*ds_(2)))
        print(assemble((dot(tress, n)[1])*ds_(1)))        

        print(assemble((dot(train, n)[0])*ds_(2)), assemble(p*ds_(2)))
        print(assemble((dot(train, n)[1])*ds_(1)))        
        
    File('uh.pvd') << uh
    u = interpolate(u_exact, uh.function_space())
    File('u.pvd') << u
    eu = uh
    eu.vector().axpy(-1, u.vector())
    File('eu.pvd') << eu

    File('ph.pvd') << ph
    p = interpolate(p_exact, ph.function_space())
    File('p.pvd') << p
    ep = ph
    ep.vector().axpy(-1, p.vector())
    File('ep.pvd') << ep

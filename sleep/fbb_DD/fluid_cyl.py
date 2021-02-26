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
    if len(u.ufl_shape) == 1:
        return tr(Grad(u))
    return as_vector(tuple(Div(u[i, :]) for i in range(u.ufl_shape[0])))


strain = lambda u, mu: 2*mu*sym(Grad(u))
sigma = lambda u, p, mu: strain(u, mu) - as_matrix(((p, Constant(0), Constant(0)),
                                                    (0, Constant(0), Constant(0)),
                                                    (Constant(0), Constant(0), Constant(0))))


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
    system = (inner(mu*(Grad(u)), (Grad(v)))*r*dx - inner(p, Div(v))*r*dx
              -inner(q, Div(u))*r*dx - inner(f, v)*r*dx)

    # u x n
    # 
    
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

    forcing = Constant((0, 0))
                         
    # Taylor-Hood
    Velm = VectorElement('Lagrange', triangle, 2)
    Qelm = FiniteElement('Lagrange', triangle, 1)
    Welm = MixedElement([Velm, Qelm])

    material_parameters = {'mu': Constant(mu_value)}

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
        p_true = delta_P*z/length
        u_true = as_vector((-delta_P/L/4/mu*(ln(r/rOut)/ln(rOut/rIn)*(rOut**2 - rIn**2) + (rOut**2 - r**2)),
                            Constant(0)*p_true))

        bcs = {'velocity': [(3, Constant((0, 0))), (4, Constant((0, 0)))],
               'traction': [(1, Constant((0, 0))), (2, Constant((-delta_P, 0)))]}

        W = FunctionSpace(mesh, Welm)
        uh, ph = solve_fluid(W, f=forcing, bdries=bdries, bcs=bcs,
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
    u = project(u_true, uh.function_space())
    File('u.pvd') << u
    eu = uh
    eu.vector().axpy(-1, u.vector())
    File('eu.pvd') << eu

    File('ph.pvd') << ph
    # p = interpolate(p_exact, ph.function_space())
    # File('p.pvd') << p
    # ep = ph
    # ep.vector().axpy(-1, p.vector())
    # File('ep.pvd') << ep

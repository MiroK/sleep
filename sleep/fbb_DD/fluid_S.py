from dolfin import *
from functools import reduce
import itertools
import operator
import sympy as sp
import ulfy  # https://github.com/MiroK/ulfy

# Non steady stokes equations
# 
# rho du/dt-div(sigma(u, p)) = f  with sigma(u, p) = 2*mu*sym(grad(u)) - p*I
# -div(u) = 0 in Omega
#
# with following bcs:
# 1) velocity boundary sets velocity vector
# 2) Traction boundary sets sigma.n
# 3) Pressure boundary set sigma.n.n (pressure part) and (sigma.n).t =0
#
# is solved on FE space W

def solve_fluid(W, f,u_n,p_n, bdries, bcs, parameters):
    '''Return velocity and pressure'''
    info('Solving Stokes for %d unknowns' % W.dim())
    mesh = W.mesh()
    assert mesh.geometry().dim() == 2
    # Let's see about boundary conditions - they need to be specified on
    # every boundary.
    assert all(k in ('velocity','velocity_x', 'traction', 'pressure') for k in bcs)
    # The tags must be found in bdries
    velocity_bcs = bcs.get('velocity', ())  
    velocity_x_bcs = bcs.get('velocity_x', ()) 
    traction_bcs = bcs.get('traction', ())
    pressure_bcs = bcs.get('pressure', ())
    # Tuple of pairs (tag, boundary value) is expected
    velocity_tags = set(item[0] for item in velocity_bcs)
    velocity_x_tags = set(item[0] for item in velocity_x_bcs)
    traction_tags = set(item[0] for item in traction_bcs)
    pressure_tags = set(item[0] for item in pressure_bcs)

    tags = (velocity_tags, velocity_x_tags, traction_tags, pressure_tags)
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
    k = Constant(parameters['dt'])   
    mu  = Constant(parameters['mu'])
    rho = Constant(parameters['rho'])

    # Define Cauchy stress tensor
    def sigma(u,p):
        return 2.0*mu*0.5*(grad(u) + grad(u).T)  - p*Identity(len(u))

    # Define symmetric gradient
    def epsilon(v):
        return  0.5*(grad(v) + grad(v).T)

    system = (inner(2*mu*sym(grad(u)), sym(grad(v)))*dx - inner(p, div(v))*dx-inner(q, div(u))*dx - inner(f, v)*dx)
    
    # Add time derivative term
    #system+=rho*(1/k)*inner(v, u - u_n)*dx 

    # Handle natural bcs
    n = FacetNormal(mesh)
    ds = Measure('ds', domain=mesh, subdomain_data=bdries)
    

    for tag, value in traction_bcs:
        system += -inner(value, v)*ds(tag)

    # For the pressure condition : 
    # We need to impose the normal component of the normal traction on the inlet and outlet to be the pressures we want on each surface
    # and force the normal component of the grad u to be zero

    tau=dot(epsilon(u),n)

    for tag, value in pressure_bcs:
        # set the normal component of traction to the imposed pressure value
        system += inner(value, dot(v, n))*ds(tag)
        system += 2*mu*inner(dot(tau,n),dot(v,n))*ds(tag)
        system += -2*mu*inner(tau,v)*ds(tag)



    # velocity bcs go onto the matrix
    bcs_D = [DirichletBC(W.sub(0), value, bdries, tag) for tag, value in velocity_bcs]
    bcs_D += [DirichletBC(W.sub(0).sub(0), value, bdries, tag) for tag, value in velocity_x_bcs]

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



from dolfin import *
from functools import reduce
import itertools
import operator
import sympy as sp
import ulfy  # https://github.com/MiroK/ulfy

# Problem
# 
# NS equations
#
# with following bcs:
# 1) velocity boundary sets velocity vector
# 2) Traction boundary sets sigma.n
# 3) Pressure boundary set sigma.n.n (pressure part) and (sigma.n).t
#
# is solved on FE space W

def solve_fluid(W, f, u_n, p_n, bdries, bcs, parameters):
    '''Return velocity and pressure'''
    info('Solving Navier Stokes for %d unknowns' % W.dim())
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

    ## Define functions
                 
    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)
    assert len(u.ufl_shape) == 1 and len(p.ufl_shape) == 0

    # Why not taking directly un and pn ? is it because the mesh can have changed ?
    # Previous solutions
    wh_n = Function(W)
    assign(wh_n.sub(0), interpolate(u_n, W.sub(0).collapse()))    
    assign(wh_n.sub(1), interpolate(p_n, W.sub(1).collapse()))
    u_n, p_n = split(wh_n)

    # Define functions at current time step
    wh = Function(W)
    u_,p_ = wh.split(True)
 



    # Define boundary conditions 

    # todo : traction BC

    # Pressure 
    bcu = [DirichletBC(W.sub(1), value, bdries, tag) for tag, value in pressure_bcs]


    # Velocity
    bcp = [DirichletBC(W.sub(0), value, bdries, tag) for tag, value in velocity_bcs]

    # -------------------------------------
    # todo :  I suppose that strain stress definition, variational form definition + assembly of Ai can be done one time outide the time steping


    # Parameters
    U   = 0.5*(u_n + u)

    k = Constant(parameters['dt'])

    n   = FacetNormal(mesh)
    
    mu  = Constant(parameters['mu'])
    rho = Constant(parameters['rho'])

    # Define strain-rate tensor
    def epsilon(u):
        return sym(nabla_grad(u))

    # Define stress tensor
    def sigma(u, p):
        return 2*mu*epsilon(u) - p*Identity(len(u))


    # Define variational problem for step 1
    F1 = rho*dot((u - u_n) / k, v)*dx + \
        rho*dot(dot(u_n, nabla_grad(u_n)), v)*dx \
    + inner(sigma(U, p_n), epsilon(v))*dx \
    + dot(p_n*n, v)*ds - dot(mu*nabla_grad(U)*n, v)*ds \
    - dot(f, v)*dx
    a1 = lhs(F1)
    L1 = rhs(F1)

    # Define variational problem for step 2
    a2 = dot(nabla_grad(p), nabla_grad(q))*dx
    L2 = dot(nabla_grad(p_n), nabla_grad(q))*dx - (1/k)*div(u_)*q*dx

    # Define variational problem for step 3
    a3 = dot(u, v)*dx
    L3 = dot(u_, v)*dx - k*dot(nabla_grad(p_ - p_n), v)*dx

    # Assemble matrices
    A1 = assemble(a1)
    A2 = assemble(a2)
    A3 = assemble(a3)

    #-----------------------------

    # Apply boundary conditions to matrices
    [bc.apply(A1) for bc in bcu]
    [bc.apply(A2) for bc in bcp]

    # Step 1: Tentative velocity step
    b1 = assemble(L1)
    [bc.apply(b1) for bc in bcu]
    solve(A1, u_.vector(), b1)

    # Step 2: Pressure correction step
    b2 = assemble(L2)
    [bc.apply(b2) for bc in bcp]
    solve(A2, p_.vector(), b2)

    # Step 3: Velocity correction step
    b3 = assemble(L3)
    solve(A3, u_.vector(), b3)

    # Update previous solution
    u_n.assign(u_)
    p_n.assign(p_)

    return u_,p_


# --------------------------------------------------------------------

if __name__ == '__main__':

    print('hi')
from dolfin import *
from sleep.utils import preduce
import itertools
import operator
import sympy as sp
import ulfy  # https://github.com/MiroK/ulfy
from petsc4py import PETSc
import numpy as np

print = PETSc.Sys.Print

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
#  3) Nitsche_t: sets u.t = g_u and (sigma.n).n = g_p
#
# Darcy
# 1) pressure - sets pressure
# 2) flux - constrols u.n
#

def solve_solid(W, f1, f2, eta_0, p_0, bdries, bcs, parameters, nitsche_penalty):
    '''Return displacement, percolation velocity, pressure and final time'''
    print('Solving Biot for %d unknowns' % W.dim())
    # NOTE: this is time dependent problem which we solve with 
    # parameters['dt'] for parameters['nsteps'] time steps and to 
    # update time in f1, f2 or the expressions bcs the physical time is
    # set as parameters['T0'] + dt*(k-th step)
    mesh = W.mesh()
    comm = mesh.mpi_comm()

    needed = preduce(comm, operator.or_, (set(bdries.array()), )) - set((0, ))    
    # Validate elasticity bcs
    bcs_E = bcs['elasticity']
    assert all(k in ('displacement', 'traction', 'nitsche_t') for k in bcs_E)

    displacement_bcs = bcs_E.get('displacement', ())  
    traction_bcs = bcs_E.get('traction', ())
    nitsche_t_bcs = bcs_E.get('nitsche_t', ())
    # Tuple of pairs (tag, boundary value) is expected
    displacement_tags = set(item[0] for item in displacement_bcs)
    traction_tags = set(item[0] for item in traction_bcs)
    nitsche_t_tags = set(item[0] for item in nitsche_t_bcs)

    tags = (displacement_tags, traction_tags, nitsche_t_tags)
    for this, that in itertools.combinations(tags, 2):
        if this and that: assert not this & that

    assert needed == preduce(comm, operator.or_, tags)

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

    assert needed == preduce(comm, operator.or_, tags), (needed, preduce(comm, operator.or_, tags))

    # Collect bc values for possible temporal update in the integration
    bdry_expressions = sum(([item[1] for item in bc]
                            for bc in (displacement_bcs, traction_bcs, pressure_bcs, flux_bcs)),
                           [])
    # Nitsche is special (tag, (vel_data, stress_data))
    [bdry_expressions.extend(val[1]) for val in nitsche_t_bcs]

    # FEM ---
    eta, p = TrialFunctions(W)
    phi, q = TestFunctions(W)
    assert len(eta.ufl_shape) == 1 and len(p.ufl_shape) == 0

    # Material parameters
    kappa, mu, lmbda, alpha, s0 = (parameters[k] for k in ('kappa', 'mu', 'lmbda', 'alpha', 's0'))
    # For weak form also time step is needed
    dt = Constant(parameters['dt'])

    # Previous solutions
    wh_0 = Function(W)
    assign(wh_0.sub(0), interpolate(eta_0, W.sub(0).collapse()))    
    assign(wh_0.sub(1), interpolate(p_0, W.sub(1).collapse()))
    eta_0, p_0 = split(wh_0)

    sigma_B = lambda u, p: 2*mu*sym(grad(u)) + lmbda*div(u)*Identity(len(u)) - alpha*p*Identity(len(u))
    # Standard bit
    # Elasticity
    a = (2*mu*inner(sym(grad(eta)), sym(grad(phi)))*dx +
         inner(lmbda*div(eta), div(phi))*dx -
         inner(p, alpha*div(phi))*dx)

    L = inner(f1, phi)*dx
              
    # Darcy
    a += (-alpha*inner(q, div(eta))*dx - inner(s0*p, q)*dx - inner(kappa*dt*grad(p), grad(q))*dx)

    L += -dt*inner(f2, q)*dx - inner(s0*p_0, q)*dx - inner(alpha*div(eta_0), q)*dx         

    # Handle natural bcs
    n = FacetNormal(mesh)
    ds = Measure('ds', domain=mesh, subdomain_data=bdries)
    # For elasticity
    for tag, value in traction_bcs:
        L += inner(value, phi)*ds(tag)
    # For Darcy
    for tag, value in flux_bcs:
        if value.ufl_shape == ():
            L += dt*inner(value, q)*ds(tag)  # Scalar that is -kappa*grad(p).n
        else:
            assert len(value.ufl_shape) == 1 # A vector that is -kappa*grad(p)
            L += dt*inner(dot(value, n), q)*ds(tag)

    # Nitsche
    # We have -sigma.n . phi -> -(sigma.n.n)(phi.n) + -(sigma(eta, p).n.t)(phi.t)
    #                            <bdry data>
    # Symmetry 
    #                                             -(sigma(phi, q).n.t.(eta.t - bdry_data)
    # and penalize
    #                                             + gamma/h (phi.t)*(eta.t - bdry_data)
    
    # Generalize cross product
    def wedge(x, y):
        (n, ) = x.ufl_shape
        (m, ) = y.ufl_shape
        assert n == m
        if n == 2:
            R = Constant(((0, -1), (1, 0)))
            return dot(x, dot(R, y))
        else:
            return cross(x, y)

    hF = CellDiameter(mesh)
    for tag, (vel_data, stress_data) in nitsche_t_bcs:

        a += (-inner(wedge(dot(sigma_B(eta, p), n), n), wedge(phi, n))*ds(tag)
              -inner(wedge(dot(sigma_B(phi, q), n), n), wedge(eta, n))*ds(tag)
              + Constant(nitsche_penalty)/hF*inner(wedge(eta, n), wedge(phi, n))*ds(tag)
        )

        L += (
            # This is the velocity part
            -inner(wedge(dot(sigma_B(phi, q), n), n), vel_data)*ds(tag)
            + Constant(nitsche_penalty)/hF*inner(vel_data, wedge(phi, n))*ds(tag)
            # Not the stress comess from
            + inner(stress_data, dot(phi, n))*ds(tag)
        )
    
    # Displacement bcs and flux bcs go on the system
    bcs_strong = [DirichletBC(W.sub(0), value, bdries, tag) for tag, value in displacement_bcs]
    bcs_strong.extend([DirichletBC(W.sub(1), value, bdries, tag) for tag, value in pressure_bcs])

    # Discrete problem
    assembler = SystemAssembler(a, L, bcs_strong)

    A, b = PETScMatrix(), PETScVector()
    # Assemble once and setup solver for it
    assembler.assemble(A)

    if parameters.get('solver', 'direct') == 'direct':
        solver = LUSolver(A, 'mumps')
    else:
        schur_bit = sqrt(lmbda + 2*mu/len(eta))
        # Preconditioner operator following https://arxiv.org/abs/1705.08842
        a_prec = (2*mu*inner(sym(grad(eta)), sym(grad(phi)))*dx
                  + inner(lmbda*div(eta), div(phi))*dx
                  + inner(s0*p, q)*dx + inner(kappa*dt*grad(p), grad(q))*dx
                  + (alpha**2/schur_bit**2)*inner(p, q)*dx)
        # We do a wishful thinking hoping that the Nitsche contribs to
        # the schur complement can be hidden with the mass matrix
        for tag, (vel_data, stress_data) in nitsche_t_bcs:
            # NOTE: sigma_B.n x n has not contrib to pressure!
            a_prec += (-inner(wedge(dot(sigma_B(eta, p), n), n), wedge(phi, n))*ds(tag)
                       -inner(wedge(dot(sigma_B(phi, q), n), n), wedge(eta, n))*ds(tag)
                       + Constant(nitsche_penalty)/hF*inner(wedge(eta, n), wedge(phi, n))*ds(tag)
            )

        B, b_dummy = PETScMatrix(), PETScVector()
        assemble_system(a_prec, L, bcs_strong, A_tensor=B, b_tensor=b_dummy)

        solver = PETScKrylovSolver()
        
        ksp = solver.ksp()
        ksp.setType(PETSc.KSP.Type.MINRES)
        ksp.setOperators(A.mat(), B.mat())
        ksp.setInitialGuessNonzero(True)  # For time dep problem

        V_dofs, Q_dofs = (W.sub(i).dofmap().dofs() for i in range(2))

        pc = ksp.getPC()
        pc.setType(PETSc.PC.Type.FIELDSPLIT)
        is_V = PETSc.IS().createGeneral(V_dofs)
        is_Q = PETSc.IS().createGeneral(Q_dofs)
        pc.setFieldSplitIS(('0', is_V), ('1', is_Q))
        pc.setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE) 

        ksp.setUp()
        # ... all blocks inverted AMG sweeps
        subksps = pc.getFieldSplitSubKSP()
        subksps[0].setType('preonly')
        # NOTE: ideally this would be done with multigrid but let's see
        # how far LU will take us. For amg also consider grad-grad to help
        # it
        subksps[0].getPC().setType('lu')  
        subksps[1].setType('preonly')
        subksps[1].getPC().setType('hypre')

        opts = PETSc.Options()
        # opts.setValue('ksp_monitor_true_residual', None)
        opts.setValue('ksp_rtol', 1E-8)
        opts.setValue('ksp_atol', 1E-10)

        pc.setFromOptions()
        ksp.setFromOptions()

    # Temporal integration loop
    T0 = parameters['T0']
    niters = []
    for k in range(parameters['nsteps']):
        # Update source if possible
        for foo in bdry_expressions + [f1, f2]:
            hasattr(foo, 't') and setattr(foo, 't', T0)
            hasattr(foo, 'time') and setattr(foo, 'time', T0)

        assembler.assemble(b)
        niters.append(solver.solve(wh_0.vector(), b))
        T0 += dt(0)

        if k % 10 == 0:
            print('  Biot at step (%d, %g) |uh|=%g' % (k, T0, wh_0.vector().norm('l2')))
            print('  KSP stats MIN/MEAN/MAX (%d|%g|%d) %d' % (
                np.min(niters), np.mean(niters), np.max(niters), len(niters)
            ))
            niters.clear()

    eta_h, p_h = wh_0.split(deepcopy=True)
        
    return eta_h, p_h, T0


def mms_solid(parameters, round_top=False):
    '''Method of manufactured solutions on [0, 1]^2'''
    mesh = UnitSquareMesh(MPI.comm_self, 2, 2)  
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
    if not round_top:
        normals = [Constant((-1, 0)), Constant((1, 0)), Constant((0, -1)), Constant((0, 1))]
    else:
        x, y = SpatialCoordinate(mesh)
        radius = Constant(sqrt(0.5**2 + 1**2))
        normals = [Constant((-1, 0)), Constant((1, 0)), Constant((0, -1)),
                   as_vector((x-Constant(0.5), y))/radius]
        
    normal_tractions = [as_expr(dot(n, dot(sigma_p(eta, p), n))) for n in normals]
    normal_displacements = [as_expr(dot(n, eta)) for n in normals]
    
    R = Constant(((0, -1), (1, 0)))
    tangent_tractions = [as_expr(dot(dot(R, n), dot(sigma_p(eta, p), n))) for n in normals]
    tangent_displacements = [as_expr(dot(dot(R, n), eta)) for n in normals]

    tractions = [as_expr(dot(sigma_p(eta, p), n)) for n in normals]    

    return {'solution': (eta_exact, u_exact, p_exact),
            'force': (f1, f2),
            'tractions': tractions,
            'normal_tractions': normal_tractions,
            'tangent_tractions': tangent_tractions, 
            'normal_displacements': normal_displacements, 
            'tangent_displacements': tangent_displacements}
           
# --------------------------------------------------------------------

if __name__ == '__main__':
    from sleep.mesh import load_mesh2d

    round_top = False
    # NOTE: role of curvature
    # In space we have cca 1, 1 rates if rounded
    #                      2, 1 rates if <<FLAT>>
    # maybe this can be improved by penalty but I'd say LM to the rescue
    parameters = {'kappa': Constant(2),
                  'mu': Constant(3),
                  'lmbda': Constant(4),
                  'alpha': Constant(5),
                  's0': Constant(0.2)}
    # Decide solver
    parameters['solver'] = 'iterative'
    
    data = mms_solid(parameters, round_top=round_top)
    
    eta_exact, u_exact, p_exact  = data['solution']
    f1, f2 = data['force']

    (tractions,
     normal_tractions,
     tangent_tractions,
     normal_displacements,
     tangent_displacements) = (dict(enumerate(data[key], 1)) for key in ('tractions',
                                                                         'normal_tractions',
                                                                         'tangent_tractions',
                                                                         'normal_displacements',
                                                                         'tangent_displacements'))

    cell = triangle
    Eelm = VectorElement('Lagrange', cell, 2)
    Qelm = FiniteElement('Lagrange', cell, 1)

    Welm = MixedElement([Eelm, Qelm])

    parameters['dt'] = 1E-6
    parameters['nsteps'] = int(1E-4/parameters['dt'])
    parameters['T0'] = 0.

    if round_top:
        template = '/home/fenics/shared/sleep/sleep/mesh/test/round_domain_{}.h5'
    else:
        template = '/home/fenics/shared/sleep/sleep/mesh/test/square_domain_{}.h5'        

    scales = (1.0, 0.5, 0.25, 0.125, 0.0625)# , 0.03125)
    for scale in scales:
        path = template.format(scale)
        
        mesh, markers, lookup = load_mesh2d(path)
        bdries = markers[1]

    # This is with structured mesh
    # for n in (4, 8, 16, 32, 64):
    #     mesh = UnitSquareMesh(n, n)
        
    #     bdries = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
    #     CompiledSubDomain('near(x[0], 0)').mark(bdries, 1)
    #     CompiledSubDomain('near(x[0], 1)').mark(bdries, 2)
    #     CompiledSubDomain('near(x[1], 0)').mark(bdries, 3)
    #     CompiledSubDomain('near(x[1], 1)').mark(bdries, 4)

        # Reset time
        for things in data.values():
            for thing in things:
                thing.time = 0.
        
        # Elasticity: displacement bottom, traction for rest
        # Darcy: pressure bottom and top, flux on sides
        bcs = {'elasticity': {'displacement': [(3, eta_exact)],
                              'traction': [(1, tractions[1]), (2, tractions[2])],
                              # Nitsche data is a pair of value for eta.t and sigma.n.n
                              'nitsche_t': [(4, (tangent_displacements[4], normal_tractions[4]))],
                              },
               'darcy': {'pressure': [(1, p_exact), (2, p_exact)],
                         'flux': [(3, u_exact), (4, u_exact)]}}

        # Get the initial conditions
        E = FunctionSpace(mesh, Eelm)
        Q = FunctionSpace(mesh, Qelm)

        eta_0 = interpolate(eta_exact, E)
        p_0 = interpolate(p_exact, Q)

        W = FunctionSpace(mesh, Welm)
        ans = solve_solid(W, f1, f2, eta_0, p_0, bdries=bdries, bcs=bcs,
                          parameters=parameters,
                          nitsche_penalty=400)  # NOTE: this might need to be adjusted
        # depending on the material parameter values

        eta_h, p_h, time = ans
        eta_exact.time, p_exact.time = (time, )*2
        # Errors
        e_eta = errornorm(eta_exact, eta_h, 'H1', degree_rise=2)        
        e_p = errornorm(p_exact, p_h, 'H1', degree_rise=2)        
        print('|eta-eta_h|_1', e_eta, '|p-ph|_1', e_p, '#dofs', W.dim())

    # ----

    scale = scales[-1]
    path = template.format(scale)
        
    mesh, markers, lookup = load_mesh2d(path)
    bdries = markers[1]

    dt = 1E-2
    for _ in range(4):
        parameters['dt'] = dt
        parameters['nsteps'] = int(1E-1/dt)
        parameters['T0'] = 0.
        
        # Reset time
        for things in data.values():
            for thing in things:
                thing.time = 0.

        bcs = {'elasticity': {'displacement': [(3, eta_exact)],
                              'traction': [(1, tractions[1]), (2, tractions[2])],
                              # Nitsche data is a pair of value for eta.t and sigma.n.n
                              'nitsche_t': [(4, (tangent_displacements[4], normal_tractions[4]))],
                              },
               'darcy': {'pressure': [(1, p_exact), (2, p_exact)],
                         'flux': [(3, u_exact), (4, u_exact)]}}
        

        # Get the initial conditions
        E = FunctionSpace(mesh, Eelm)
        Q = FunctionSpace(mesh, Qelm)

        eta_0 = interpolate(eta_exact, E)
        p_0 = interpolate(p_exact, Q)

        W = FunctionSpace(mesh, Welm)
        ans = solve_solid(W, f1, f2, eta_0, p_0, bdries=bdries, bcs=bcs,
                          parameters=parameters,
                          nitsche_penalty=400)

        eta_h, p_h, time = ans
        eta_exact.time, p_exact.time = (time, )*2
        # Errors
        e_eta = errornorm(eta_exact, eta_h, 'H1', degree_rise=2)        
        e_p = errornorm(p_exact, p_h, 'H1', degree_rise=2)
        print('\tdt=%.2E T=%.2f nsteps=%d' %  (dt, p_exact.time, parameters['nsteps']))
        print('|eta-eta_h|_1', e_eta, '|p-ph|_1', e_p, '#dofs', W.dim())
        
        dt = dt/2

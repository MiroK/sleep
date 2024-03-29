import petsc4py, sys
petsc4py.init(sys.argv)
from petsc4py import PETSc

# Total pressure formulation
from dolfin import *
from sleep.utils import preduce, KSP_CVRG_REASONS
import itertools
import operator
import sympy as sp
import ulfy  # https://github.com/MiroK/ulfy
import numpy as np
from collections import Counter

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
#
# Darcy
# 1) pressure - sets pressure
# 2) flux - constrols u.n
#

def solve_solid(W, f1, f2, eta_0, pT_0, p_0, bdries, bcs, parameters):
    '''Return displacement, pressure and final time'''
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
    assert all(k in ('displacement', 'traction') for k in bcs_E)

    displacement_bcs = bcs_E.get('displacement', ())  
    traction_bcs = bcs_E.get('traction', ())
    # Tuple of pairs (tag, boundary value) is expected
    displacement_tags = set(item[0] for item in displacement_bcs)
    traction_tags = set(item[0] for item in traction_bcs)

    tags = (displacement_tags, traction_tags)
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

    # FEM ---
    eta, pT, p = TrialFunctions(W)
    phi, qT, q = TestFunctions(W)
    assert len(eta.ufl_shape) == 1 and len(pT.ufl_shape) == 0 and len(p.ufl_shape) == 0

    # Material parameters
    kappa, mu, lmbda, alpha, s0 = (parameters[k] for k in ('kappa', 'mu', 'lmbda', 'alpha', 's0'))
    # For weak form also time step is needed
    dt = Constant(parameters['dt'])

    # Previous solutions
    wh_0 = Function(W)
    assign(wh_0.sub(0), interpolate(eta_0, W.sub(0).collapse()))
    assign(wh_0.sub(1), interpolate(pT_0, W.sub(1).collapse()))        
    assign(wh_0.sub(2), interpolate(p_0, W.sub(2).collapse()))
    eta_0, pT_0, p_0 = split(wh_0)

    # Elasticity
    a = (2*mu*inner(sym(grad(eta)), sym(grad(phi)))*dx - inner(pT, div(phi))*dx)

    L = inner(f1, phi)*dx
    # Total pressure
    a += -inner(div(eta), qT)*dx - (1/lmbda)*inner(pT, qT)*dx + (alpha/lmbda)*inner(p, qT)*dx
    # Darcy
    a += (alpha/lmbda)*inner(q, pT)*dx - inner((s0+alpha**2/lmbda)*p, q)*dx - inner(kappa*dt*grad(p), grad(q))*dx

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
          
    # Displacement bcs and pressure bcs go on the system
    bcs_strong = [DirichletBC(W.sub(0), value, bdries, tag) for tag, value in displacement_bcs]
    bcs_strong.extend([DirichletBC(W.sub(2), value, bdries, tag) for tag, value in pressure_bcs])

    # Discrete problem
    assembler = SystemAssembler(a, L, bcs_strong)

    A, b = PETScMatrix(), PETScVector()
    # Assemble once and setup solver for it
    assembler.assemble(A)

    if parameters.get('solver', 'direct') == 'direct':
        solver = PETScLUSolver(A, 'mumps')
        ksp = solver.ksp()
    else:
        # Lee Mardal Winther preconditioner (assuming here that we are
        # fixing displacement somewhere)
        a_prec = (2*mu*inner(sym(grad(eta)), sym(grad(phi)))*dx  # Full H1 here?
                  + (1 + 1/2/mu)*inner(pT, qT)*dx
                  + inner((s0 + alpha**2/lmbda)*p, q)*dx + inner(kappa*dt*grad(p), grad(q))*dx)

        B, b_dummy = PETScMatrix(), PETScVector()
        assemble_system(a_prec, L, bcs_strong, A_tensor=B, b_tensor=b_dummy)

        solver = PETScKrylovSolver()
        
        ksp = solver.ksp()
        ksp.setType(PETSc.KSP.Type.MINRES)
        ksp.setOperators(A.mat(), B.mat())
        ksp.setInitialGuessNonzero(True)  # For time dep problem

        V_dofs, QT_dofs, Q_dofs = (W.sub(i).dofmap().dofs() for i in range(3))

        pc = ksp.getPC()
        pc.setType(PETSc.PC.Type.FIELDSPLIT)
        is_V = PETSc.IS().createGeneral(V_dofs) 
        is_QT = PETSc.IS().createGeneral(QT_dofs)       
        is_Q = PETSc.IS().createGeneral(Q_dofs)
        pc.setFieldSplitIS(('0', is_V), ('1', is_QT), ('2', is_Q))
        pc.setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE) 

        ksp.setUp()
        # ... all blocks inverted AMG sweeps
        subksps = pc.getFieldSplitSubKSP()
        # NOTE: ideally this would be done with multigrid but let's see
        # how far LU will take us. For amg also consider grad-grad to help
        # it
        subksps[0].setType('preonly')        
        subksps[0].getPC().setType('lu')
        
        subksps[1].setType('preonly')
        subksps[1].getPC().setType('hypre')

        subksps[2].setType('preonly')
        subksps[2].getPC().setType('hypre')

        opts = PETSc.Options()
        # opts.setValue('ksp_monitor_true_residual', None)
        opts.setValue('ksp_rtol', 1E-14)
        opts.setValue('ksp_atol', 1E-8)

        pc.setFromOptions()
        ksp.setFromOptions()
        

    # Temporal integration loop
    T0 = parameters['T0']
    niters, reasons = [], []
    for k in range(parameters['nsteps']):
        # Update source if possible
        for foo in bdry_expressions + [f1, f2]:
            hasattr(foo, 't') and setattr(foo, 't', T0)
            hasattr(foo, 'time') and setattr(foo, 'time', T0)

        assembler.assemble(b)
        niters.append(solver.solve(wh_0.vector(), b))
        reasons.append(ksp.getConvergedReason())        
        T0 += dt(0)
        
        if k % 10 == 0:
            print('  Biot at step (%d, %g) |uh|=%g' % (k, T0, wh_0.vector().norm('l2')))
            print('  KSP stats MIN/MEAN/MAX (%d|%g|%d) %d' % (
                np.min(niters), np.mean(niters), np.max(niters), len(niters)
            ))
            for reason, count in Counter(reasons).items():
                print('      KSP cvrg reason %s -> %d' % (KSP_CVRG_REASONS[reason], count))
            niters.clear()
            reasons.clear()

    eta_h, pT_h, p_h = wh_0.split(deepcopy=True)
        
    return eta_h, pT_h, p_h, T0


def mms_solid(parameters):
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

    pT = -lmbda*div(eta) + alpha*p  # Total pressure

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
    eta_exact, u_exact, p_exact, pT_exact = map(as_expr, (eta, u, p, pT))
    # Forcing
    f1, f2 = as_expr(f1), as_expr(f2)
    #  4
    # 1 2
    #  3 so that
    normals = [Constant((-1, 0)), Constant((1, 0)), Constant((0, -1)), Constant((0, 1))]
    tractions = [as_expr(dot(sigma_p(eta, p), n)) for n in normals]

    return {'solution': (eta_exact, u_exact, p_exact, pT_exact),
            'force': (f1, f2),
            'tractions': tractions}
           
# --------------------------------------------------------------------

if __name__ == '__main__':

    parameters = {'kappa': Constant(2),
                  'mu': Constant(3),
                  'lmbda': Constant(4),
                  'alpha': Constant(5),
                  's0': Constant(0.2)}
    data = mms_solid(parameters)

    # Pick system solver
    parameters['solver'] = 'iterative'
    
    eta_exact, u_exact, p_exact, pT_exact  = data['solution']
    f1, f2 = data['force']
    tractions = dict(enumerate(data['tractions'], 1))

    Eelm = VectorElement('Lagrange', triangle, 2)  # Displ
    QTelm = FiniteElement('Lagrange', triangle, 1)  # Total pressure
    Qelm = FiniteElement('Lagrange', triangle, 1)  # Pressure    

    Welm = MixedElement([Eelm, QTelm, Qelm])

    parameters['dt'] = 1E-6
    parameters['nsteps'] = int(1E-4/parameters['dt'])
    parameters['T0'] = 0.
    for n in (4, 8, 16, 32, 64):
        mesh = UnitSquareMesh(n, n)
        # Setup similar to coupled problem ...
        bdries = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
        CompiledSubDomain('near(x[0], 0)').mark(bdries, 1)
        CompiledSubDomain('near(x[0], 1)').mark(bdries, 2)
        CompiledSubDomain('near(x[1], 0)').mark(bdries, 3)
        CompiledSubDomain('near(x[1], 1)').mark(bdries, 4)

        # Reset time
        for things in data.values():
            for thing in things:
                thing.time = 0.
        
        # Elasticity: displacement bottom, traction for rest
        # Darcy: pressure bottom and top, flux on sides
        bcs = {'elasticity': {'displacement': [(3, eta_exact)],
                              'traction': [(1, tractions[1]), (2, tractions[2]), (4, tractions[4])]},
               'darcy': {'pressure': [(1, p_exact), (2, p_exact)],
                         'flux': [(3, u_exact), (4, u_exact)]}}

        # Get the initial conditions
        E = FunctionSpace(mesh, Eelm)
        QT = FunctionSpace(mesh, QTelm)
        Q = FunctionSpace(mesh, Qelm)

        eta_0 = interpolate(eta_exact, E)
        pT_0 = interpolate(pT_exact, QT)
        p_0 = interpolate(p_exact, Q)

        W = FunctionSpace(mesh, Welm)
        ans = solve_solid(W, f1, f2, eta_0, pT_0, p_0, bdries=bdries, bcs=bcs,
                          parameters=parameters)

        eta_h, pT_h, p_h, time = ans
        eta_exact.time, pT_exact.time, p_exact.time = (time, )*3
        # Errors
        e_eta = errornorm(eta_exact, eta_h, 'H1', degree_rise=2)
        e_pT = errornorm(pT_exact, pT_h, 'L2', degree_rise=2)                
        e_p = errornorm(p_exact, p_h, 'H1', degree_rise=2)        
        print('|eta-eta_h|_1', e_eta,
              '|pT-pTh|_0', e_pT,              
              '|p-ph|_1', e_p,
              '#dofs', W.dim())

    # ----

    n = 128
    mesh = UnitSquareMesh(n, n)
    # Setup similar to coupled problem ...
    bdries = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
    CompiledSubDomain('near(x[0], 0)').mark(bdries, 1)
    CompiledSubDomain('near(x[0], 1)').mark(bdries, 2)
    CompiledSubDomain('near(x[1], 0)').mark(bdries, 3)
    CompiledSubDomain('near(x[1], 1)').mark(bdries, 4)

    dt = 1E-2
    for _ in range(4):
        parameters['dt'] = dt
        parameters['nsteps'] = int(1E-1/dt)
        parameters['T0'] = 0.
        
        # Reset time
        for things in data.values():
            for thing in things:
                thing.time = 0.

        # Elasticity: displacement bottom, traction for rest
        # Darcy: pressure bottom and top, flux on sides
        bcs = {'elasticity': {'displacement': [(3, eta_exact)],
                              'traction': [(1, tractions[1]), (2, tractions[2]), (4, tractions[4])]},
               'darcy': {'pressure': [(1, p_exact), (2, p_exact)],
                         'flux': [(3, u_exact), (4, u_exact)]}}

        # Get the initial conditions
        E = FunctionSpace(mesh, Eelm)
        QT = FunctionSpace(mesh, QTelm)
        Q = FunctionSpace(mesh, Qelm)

        eta_0 = interpolate(eta_exact, E)
        pT_0 = interpolate(pT_exact,QT)
        p_0 = interpolate(p_exact, Q)

        W = FunctionSpace(mesh, Welm)
        ans = solve_solid(W, f1, f2, eta_0, pT_0, p_0, bdries=bdries, bcs=bcs,
                          parameters=parameters)

        eta_h, pT_h, p_h, time = ans
        eta_exact.time, pT_exact.time, p_exact.time = (time, )*3
        # Errors
        e_eta = errornorm(eta_exact, eta_h, 'H1', degree_rise=2)
        e_pT = errornorm(pT_exact, pT_h, 'L2', degree_rise=2)        
        e_p = errornorm(p_exact, p_h, 'H1', degree_rise=2)
        print('\tdt=%.2E T=%.2f nsteps=%d' %  (dt, p_exact.time, parameters['nsteps']))
        print('|eta-eta_h|_1', e_eta,
              '|pT-pTh|_0', e_pT,
              '|p-ph|_1', e_p,
              '#dofs', W.dim())
        
        dt = dt/2

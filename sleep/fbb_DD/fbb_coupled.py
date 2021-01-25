from sleep.fbb_DD.domain_transfer import transfer_into
from sleep.fbb_DD.solid import solve_solid
from sleep.fbb_DD.fluid import solve_fluid
from sleep.fbb_DD.ale import solve_ale
from sleep.utils import EmbeddedMesh
from sleep.mesh import load_mesh2d
from dolfin import *

h5_filename = '../mesh/test/fbb_domain.h5'
# The mesh is typically generated by sleep/mesh/fbb_mesh.py
mesh, markers, lookup = load_mesh2d(h5_filename)
cell_f, facet_f = markers
cell_lookup, facet_lookup = lookup['cell'], lookup['facet']
# The mesh has 3 subdomains: corrsponding to fluid and two Biot domain. 
# Since our approach is to solve the subproblems isolated we now split 
# into fluid and solid meshes ...
mesh_f = EmbeddedMesh(cell_f, cell_lookup['F'])
mesh_s = EmbeddedMesh(cell_f, (cell_lookup['S1'], cell_lookup['S2']))
# ... and get their boundary markers

fluid_markers = ('F_left', 'F_bottom', 'F_right', 'I_bottom')
fluid_markers = tuple(facet_lookup[k] for k in fluid_markers)
fluid_bdries = mesh_f.translate_markers(facet_f, fluid_markers)

solid_markers = ('I_bottom', 'I_top', 'S1_left', 'S1_right', 'S2_left', 'S2_right', 'S2_top')
solid_markers = tuple(facet_lookup[k] for k in solid_markers)
solid_bdries = mesh_s.translate_markers(facet_f, solid_markers)
# We do not need to keep the facets marking the solid-solid interface
values = solid_bdries.array()
values[values == facet_lookup['I_top']] = 0

# Parameters setup ------------------------------------------------ FIXME
fluid_parameters = {'mu': Constant(1.0)}

# For parameters not that Biot has two subdomains (which are marked in the
# mesh so we difine discontinuous functions for them
solid_parameters = {'kappa_2': Constant(1), 'kappa_3': Constant(2),
                    'mu_2': Constant(1), 'mu_3': Constant(2),
                    'lmbda_2': Constant(1), 'lmbda_3': Constant(2),
                    'alpha_2': Constant(1), 'alpha_3': Constant(2),
                    's0_2': Constant(1), 's0_3': Constant(2)}  # FIXME

# NOTE: Here we do P0 projection
dxSolid = Measure('dx', domain=mesh_s, subdomain_data=mesh_s.marking_function)
CoefSpace = FunctionSpace(mesh_s, 'DG', 0)
q = TestFunction(CoefSpace)

for coef in ('kappa', 'mu', 'lmbda', 'alpha', 's0'):
    # Remove
    two = solid_parameters.pop('%s_2' % coef)
    three = solid_parameters.pop('%s_3' % coef)

    form = ((1/CellVolume(mesh_s))*two*q*dxSolid(cell_lookup['S1']) +
            (1/CellVolume(mesh_s))*three*q*dxSolid(cell_lookup['S2']))
    
    solid_parameters[coef] = Function(CoefSpace, assemble(form))
    
ale_parameters = {'kappa': Constant(1.0)}

# Setup fem spaces ---------------------------------------------------
Vf_elm = VectorElement('Lagrange', triangle, 2)
Qf_elm = FiniteElement('Lagrange', triangle, 1)
Wf_elm = MixedElement([Vf_elm, Qf_elm])
Wf = FunctionSpace(mesh_f, Wf_elm)

Es_elm = VectorElement('Lagrange', triangle, 2)
Vs_elm = FiniteElement('Raviart-Thomas', triangle, 1)
Qs_elm = FiniteElement('Discontinuous Lagrange', triangle, 0)
Ws_elm = MixedElement([Es_elm, Vs_elm, Qs_elm])
Ws = FunctionSpace(mesh_s, Ws_elm)

Va_elm = VectorElement('Lagrange', triangle, 1)
Va = FunctionSpace(mesh_f, Va_elm)

# Setup of boundary conditions ----------------------------------- FIXME
# We have some expression (evolving in time possible) that need to be set
pf_in = Constant(0)           # sigma_f.n.n on the inflow F_left boundary
uf_bdry = Constant((0, 0))    # velocity on the driven boundary
ps_out = Constant(0)          # Solid pressure on the top boundary of the Biot domain
ale_u_bdry = Constant((0, 0)) # displacement for ALE of the bottom wall

# We collect them for easier updates in the loop. If they have time attribute
# it will be set
driving_expressions = (pf_in, uf_bdry, ps_out, ale_u_bdry)

# These are realted to interface conditions. We will compute traction from
# solid and apply to fluid. Then from fluid unknowns we will compute pressure
# and displacement. These quantities need to be represented in FEM spaces
# of the solver. Thus
traction_f_iface = Function(VectorFunctionSpace(mesh_f, 'DG', 1))  # FIXME: DG0? to be safe
etas_iface = Function(VectorFunctionSpace(mesh_s, 'CG', 2))
aux = Function(VectorFunctionSpace(mesh_s, 'CG', 2))  # Auxiliary for u_s.np.np
ps_iface = Function(FunctionSpace(mesh_s, 'DG', 0))
# For ALE we will cary the displacement to fluid domain
etaf_iface = Function(VectorFunctionSpace(mesh_f, 'CG', 2))

# Now we wire up
bcs_fluid = {'dirichlet': [(facet_lookup['F_bottom'], uf_bdry)],
             'traction': [(facet_lookup['I_bottom'], traction_f_iface),
                          (facet_lookup['F_right'], Constant((0, 0)))],  # Outlet
             'pressure': [(facet_lookup['F_left'], (pf_in, Constant(0)))]}

bcs_ale = {'dirichlet': [(facet_lookup['F_bottom'], ale_u_bdry),
                         (facet_lookup['I_bottom'], etaf_iface)],
           'neumann': [(facet_lookup['F_left'], Constant((0, 0))),
                       (facet_lookup['F_right'], Constant((0, 0)))]}

bcs_solid = {
    'elasticity': {
        'displacement': [(facet_lookup['I_bottom'], etas_iface)],
        'traction': [(facet_lookup[tag], Constant((0, 0)))
                     for tag in ('S1_left', 'S1_right', 'S2_left', 'S2_right', 'S2_top')]
    },
    'darcy': {
        'pressure': [(facet_lookup['I_bottom'], ps_iface), (facet_lookup['S2_top'], ps_out)],
        'flux': [(facet_lookup[tag], Constant((0, 0))) for tag in ('S1_left', 'S1_right', 'S2_left', 'S2_right')]
    }
}

# Get the initial conditions ------------------------------------- FIXME
Es = Ws.sub(0).collapse()
Qs = Ws.sub(2).collapse()

eta_s0 = interpolate(Constant((0, 0)), Es)

p_s0_expr = Expression('A*x[0]', degree=1, A=1E-4)
p_s0 = interpolate(p_s0_expr, Qs)
# Now that we have the pressure we can also get the flux (to be used for interface bcs)
# so us_0 should be compatible with ps_0
u_s0 = project(-solid_parameters['kappa']*Constant((1, 0)), Ws.sub(1).collapse())

# Things for coupling
n_f, n_p = FacetNormal(mesh_f), FacetNormal(mesh_s)

sigma_f = lambda u, p, mu=fluid_parameters['mu']: 2*mu*sym(grad(u)) - p*Identity(2)

sigma_E = lambda eta, mu=solid_parameters['mu'], lmbda=solid_parameters['lmbda']: (
    2*mu*sym(grad(eta)) + lmbda*div(eta)*Identity(2)
)
sigma_p = lambda eta, p, alpha=solid_parameters['alpha']: sigma_E(eta)-alpha*p*Identity(2)

iface_tag = facet_lookup['I_bottom']
interface = (fluid_bdries, solid_bdries, iface_tag)

# Fenics won't let us move mesh with quadratic displacement so
Va_s = VectorFunctionSpace(mesh_s, 'CG', 1)

# Add things for time stepping
solid_parameters['dt'] = 1E-6  # FIXME
solid_parameters['nsteps'] = 1


# Splitting loop
time = 0.
dt = 1E-6  # Could have it own time
while time < 5*dt:
    time += dt
    # Set sources if they are time dependent
    for expr in driving_expressions:
        hasattr(expr, 'time') and setattr(expr, 'time', time)

    # Given that eta_0 and p_0 we compute traction sigma_f.n_f = -sigma_p.n_p
    transfer_into(traction_f_iface,
                  -dot(sigma_p(eta_s0, p_s0), n_p),
                  interface) 

    # Using traction solve fluid problem
    u_f, p_f = solve_fluid(Wf, f=Constant((0, 0)), bdries=fluid_bdries, bcs=bcs_fluid,
                           parameters=fluid_parameters)
    
    # In solid we want to prescribe pressure = -sigma_f.n_f.n_f
    transfer_into(ps_iface,
                  -dot(n_f, dot(sigma_f(u_f, p_f), n_f)),
                  interface)
    # We set the displacement combining
    # uf.nf + (d/dt eta + u_s).np = 0 and uf.tau - d/dt eta. tau = 0 into
    # uf - d/dt eta - (u_s.nf)*nf = 0 i.e eta = dt*(uf - (u_s.nf)*nf) + eta_0 
    transfer_into(etas_iface,
                  Constant(dt)*u_f,
                  interface)  # eta = dt*uf

    transfer_into(aux,
                  Constant(dt)*dot(u_s0, n_p)*n_p,
                  (solid_bdries, iface_tag))

    etas_iface.vector().axpy(-dt, aux.vector()) # eta = dt*uf - dt*u_s.np*np
    etas_iface.vector().axpy(1.0, eta_s0.vector()) # eta += eta_0
    
    solid_parameters['T0'] = time   
    eta_s, u_s, p_s, time_new = solve_solid(Ws, f1=Constant((0, 0)), f2=Constant(0), eta_0=eta_s0, p_0=p_s0,
                                            bdries=solid_bdries, bcs=bcs_solid, parameters=solid_parameters)
    # It might have done some step on its own so time is not
    time = time_new

    # Move the displacement for ALE
    transfer_into(etaf_iface, eta_s, interface)
    # Finally for ALE
    eta_f = solve_ale(Va, f=Constant((0, 0)), bdries=fluid_bdries, bcs=bcs_ale,
                      parameters=ale_parameters)
    
    # Move both domains(under some sensible condition) FIXME
    if True:
        ALE.move(mesh_s, interpolate(eta_s, Va_s))
        ALE.move(mesh_f, eta_f)

        File('foo.pvd') << mesh_s
        File('bar.pvd') << mesh_f

    # Reassign initial conditions
    eta_s0.assign(eta_s)
    u_s0.assign(u_s)
    p_s0.assign(p_s)

import dolfin as df
import numpy as np


def pcws_constant(cell_f, cases):
    '''Piecewise constant function f
    if x in cell_f[tag] then return cases[tag]
    '''
    mesh = cell_f.mesh()
    assert mesh.topology().dim() == cell_f.dim()
    # Check sanity of values; here we always assme that the value is
    # something that UFL can understand
    value_shape, = set(value.ufl_shape for value in cases.values())
    # We have values for all tags
    my_tags = tuple(np.unique(cell_f.array()))
    all_tags = mesh.mpi_comm().allreduce(my_tags)
    assert set(all_tags) == cases.keys()
    
    # Now we set ourselves up for P0 projection
    V = {0: df.FunctionSpace,
         1: df.VectorFunctionSpace,
         2: df.TensorFunctionSpace}[len(value_shape)](mesh, 'DG', 0)
        
    v = df.TestFunction(V)
    f = df.Function(V)

    hK = df.CellVolume(mesh)
    dx = df.Measure('dx', domain=mesh, subdomain_data=cell_f)

    form = sum((1/hK)*df.inner(v, val)*dx(tag)
               for tag, val in cases.items())

    df.assemble(form, tensor=f.vector())

    return f



# --------------------------------------------------------------------

if __name__ == '__main__':
    from solid_total import solve_solid
    from plane_geometry import plane_geometry
    from dolfin import *

    ll = (0, 0)
    ur = (1, 1)
    holes = [(0.5, 0.5, 0.1, 0.2)]

    # NOTE: `scale` here controls mesh resolutions
    entity_fs = plane_geometry(ll, ur, holes, scale=0.2)

    subdomains, boundaries = entity_fs[2], entity_fs[1]
    mesh = subdomains.mesh()
    
    # In subdomains we have tag 1 for the outer domain and 5 for the rim
    # In facets we have outer boundaries of the square marked as
    #   4
    # 1   2
    #   3
    # and there is tag 5 for the boundary of the hole. In addition we have
    # tag 6 for the other boundary of anulus (r1, r2). This one we wipe out
    boundaries.array()[boundaries.array() == 6] = 0

    # Let's define piecewise constant data based on the subdomains
    parameters = {'kappa': {1: Constant(1), 5: Constant(2)},
                  'mu': {1: Constant(1), 5: Constant(6)},
                  'lmbda': {1: Constant(1), 5: Constant(3)},
                  'alpha': {1: Constant(1), 5: Constant(4)},
                  's0': {1: Constant(1), 5: Constant(0.2)}}
    # Turn it into functions
    parameters = {key: pcws_constant(subdomains, Constant(val)) for key, val in parameters.items()}

    # Just for visual check
    [File(f'{key}.pvd') << val for key, val in parameters.items()]
    
    # Pick system solver
    parameters['solver'] = 'direct'
    
    Eelm = VectorElement('Lagrange', triangle, 2)  # Displ
    QTelm = FiniteElement('Lagrange', triangle, 1)  # Total pressure
    Qelm = FiniteElement('Lagrange', triangle, 1)  # Pressure    

    Welm = MixedElement([Eelm, QTelm, Qelm])

    parameters['dt'] = 1E-6
    parameters['nsteps'] = int(1E-4/parameters['dt'])
    parameters['T0'] = 0.

    # Get the initial conditions
    E = FunctionSpace(mesh, Eelm)
    QT = FunctionSpace(mesh, QTelm)
    Q = FunctionSpace(mesh, Qelm)

    # NOTE: this is all made up
    eta_0 = interpolate(Constant((0, 0)), E)
    pT_0 = interpolate(Constant(0), QT)
    p_0 = interpolate(Constant(0), Q)

    f1 = Constant((0, 0))
    f2 = Constant(0)

    # Elasticity: displacement bottom, traction for rest
    # Darcy: pressure bottom and top, flux on sides
    vzero = Constant((0, 0))
    bcs = {
        'elasticity':
           {
               'displacement': [(1, vzero), (2, vzero), (3, vzero), (4, vzero)],
               'traction': [(5, vzero)]
           },
           #
           'darcy':
           {
               # NOTE: I assume that on the inner wall we prescribed time-dep
               # pressure
               'pressure': [(5, Expression('A*sin(pi*t)', degree=4, A=0.01, t=0))],
               'flux': [(1, vzero), (2, vzero), (3, vzero), (4, vzero)]
           }
    }

    W = FunctionSpace(mesh, Welm)
    ans = solve_solid(W, f1, f2, eta_0, pT_0, p_0, bdries=boundaries, bcs=bcs,
                      parameters=parameters)

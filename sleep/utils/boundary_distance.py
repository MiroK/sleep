# Some papers on Laplacian mesh smoothing use diffusion based on
# diffusion which is computed from the boundary distance
from dolfin import *
import numpy as np


def distance_from_bdry_piece(facet_f, tag=None, elm=None):
    '''Solve Eikonal equation to get distance from tagged facets'''
    mesh = facet_f.mesh()
    assert mesh.topology().dim()-1 == facet_f.dim()

    # Default to CG 1
    if elm is None:
        elm = FiniteElement('Lagrange', mesh.ufl_cell(), 1)
    assert elm.cell() == mesh.ufl_cell()
    assert elm.value_shape() == ()

    if isinstance(tag, int):
        tags = [tag]
    else:
        tags = tag

    V = FunctionSpace(mesh, elm)
    u, v = TrialFunction(V), TestFunction(V)
    f = Constant(1.0)

    phis = []
    for tag in tags:
        phi = Function(V)    
        bc = DirichletBC(V, Constant(0.0), facet_f, tag)

        # Smooth initial guess
        a = inner(grad(u), grad(v))*dx
        L = inner(f, v)*dx
        solve(a == L, phi, bc)

        # Eikonal equation with stabilization
        eps = Constant(mesh.hmax()/100)
        F = sqrt(inner(grad(phi), grad(phi)))*v*dx - inner(f, v)*dx + eps*inner(grad(phi), grad(phi))*v*dx
        solve(F == 0, phi, bc)

        phis.append(phi)

    return function_min(phis)


def function_min(foos):
    '''Function that is min of foos'''
    f = foos.pop()
    V = f.function_space()

    vec = lambda u: as_backend_type(u.vector()).vec()
    # The work vector
    min_f = f.copy()
    f_vec = vec(min_f)
    # Reduce
    while foos:
        g = foos.pop()
        f_vec.pointwiseMin(f_vec, vec(g))
    # Sync
    as_backend_type(min_f.vector()).update_ghost_values()
        
    return min_f


def square_bdry_dist(V):
    '''Distance function of [0, 1]^2'''
    assert V.mesh().geometry().dim() == 2
    
    f = Function(V)
    dofs_x = V.tabulate_dof_coordinates().reshape((-1, 2))
    x, y = dofs_x.T

    possible = np.c_[x, y, 1-x, 1-y]
    dist = np.min(possible, axis=1)

    f.vector().set_local(dist)

    return f

# --------------------------------------------------------------------

if __name__ == '__main__':
    boundaries = [CompiledSubDomain('near(x[0], 0)'),
                  CompiledSubDomain('near(x[0], 1)'),
                  CompiledSubDomain('near(x[1], 0)'),
                  CompiledSubDomain('near(x[1], 1)')]

    # Check convergence
    for n in (8, 16, 32, 64):
        mesh = UnitSquareMesh(n, n)
        facet_f = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
        for tag, bdry in enumerate(boundaries, 1):
            bdry.mark(facet_f, tag)

        dist = distance_from_bdry_piece(facet_f, tag=(1, 2, 3, 4))
        dist0 = square_bdry_dist(dist.function_space())
        e = dist - dist0
        
        print(sqrt(abs(assemble(inner(e, e)*dx))))

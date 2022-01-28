import numpy as np
import gmsh


def build_model(model, geometry_parameters=None):
    '''A unit square'''
    factory = model.occ
    ll = factory.addPoint(0, 0, 0)
    lm = factory.addPoint(0.5, 0, 0)
    lr = factory.addPoint(1, 0, 0)
    ur = factory.addPoint(1, 1, 0)
    ul = factory.addPoint(0, 1, 0)

    # Bottom part is fluid
    #  3 3 2 circle is 4 1
    arc = factory.addCircleArc(ur, lm, ul)
    lines = [(ul, ll), (ll, lm), (lm, lr), (lr, ur)]
    lines = [arc] + [factory.addLine(*p) for p in lines]

    fluid_loop = factory.addCurveLoop(lines)
    fluid = factory.addPlaneSurface([fluid_loop])

    factory.synchronize()

    tags = {'cell': {'F': 1},
            'facet': {}}
    # Physical tags
    model.addPhysicalGroup(2, [fluid], 1)

    model.addPhysicalGroup(1, [lines[0]], 4)
    model.addPhysicalGroup(1, [lines[1]], 1)
    model.addPhysicalGroup(1, [lines[2], lines[3]], 3)
    model.addPhysicalGroup(1, [lines[4]], 2)        

    return model, tags

# --------------------------------------------------------------------

if __name__ == '__main__':
    from sleep.mesh import mesh_model2d, load_mesh2d, set_mesh_size
    import dolfin as df
    import sys

    if '-format' not in sys.argv:
        sys.argv.extend(['-format', 'msh2'])

    try:
        i = sys.argv.index('-clscale')
        clscale = float(sys.argv[i+1])
    except ValueError:
        clscale = 1
    
    gmsh.initialize(sys.argv)

    model = gmsh.model

    model, tags = build_model(model)

    model.occ.synchronize()
    
    h5_filename = './test/round_domain_{}.h5'.format(clscale)
    mesh_model2d(model, tags, h5_filename)

    mesh, markers, lookup = load_mesh2d(h5_filename)

    df.File('./test/foo.pvd') << markers[1]

    # TODO:
    # -Do we get the normal okay-ish on the round surface?
    x = mesh.coordinates()
    facet_f = markers[1].array()
    _, f2v = mesh.init(1, 0), mesh.topology()(1, 0)

    top_vertices = np.unique(np.hstack([f2v(f) for f in np.where(facet_f == 4)[0]]))

    n = lambda x: np.array([x[0] - 0.5, x[1]])/np.sqrt(0.5**2 + 1**2)
    
    center = np.array([0.5, 0])
    for vertex in x[top_vertices]:
        n0 = vertex - center
        n0 /= np.linalg.norm(n0)

        print(vertex, n0, n(vertex))

        
    
    # -Wire up MMS
    # -Compiles in 3d?
    
    
    gmsh.finalize()

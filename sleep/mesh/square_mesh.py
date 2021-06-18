import numpy as np
import gmsh


def build_model(model, geometry_parameters=None):
    '''A unit square'''

    factory = model.occ
    ll = factory.addPoint(0, 0, 0)
    lr = factory.addPoint(1, 0, 0)
    ur = factory.addPoint(1, 1, 0)
    ul = factory.addPoint(0, 1, 0)

    # Bottom part is fluid
    lines = [(ll, lr), (lr, ur), (ur, ul), (ul, ll)]
    lines = [factory.addLine(*p) for p in lines]

    fluid_loop = factory.addCurveLoop(lines)
    fluid = factory.addPlaneSurface([fluid_loop])

    factory.synchronize()

    tags = {'cell': {'F': 1},
            'facet': {}}
    # Physical tags
    model.addPhysicalGroup(2, [fluid], 1)

    # 3 2 4 1
    model.addPhysicalGroup(1, [lines[0]], 3)
    model.addPhysicalGroup(1, [lines[1]], 2)
    model.addPhysicalGroup(1, [lines[2]], 4)
    model.addPhysicalGroup(1, [lines[3]], 1)    


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
    
    h5_filename = './test/square_domain_{}.h5'.format(clscale)
    mesh_model2d(model, tags, h5_filename)

    mesh, markers, lookup = load_mesh2d(h5_filename)

    # df.File('./test/foo.pvd') << markers[1]
    
    gmsh.finalize()

#  YS1                     |
#  |-----------------------|
#  YF/2                    |
# o|-----------X-----------|
#  YF/2                    |
#  |-----------------------|
#  YS2                     |
#  |-----------------------|
import numpy as np
import gmsh


def build_model(model, geometry_parameters):
    '''GMSH model of sleep 2d study'''

    X, YF, YS1, YS2 = (geometry_parameters[k] for k in ('X', 'YF', 'YS1', 'YS2'))
    assert X > 0 and YF > 0 and YS1 > 0 and YS2 > 0
    
    factory = model.occ
    ll = factory.addPoint(0, -YF/2, 0)
    lr = factory.addPoint(X, -YF/2, 0)
    mr = factory.addPoint(X, YF/2, 0)
    ur = factory.addPoint(X, YF/2+YS1, 0)
    ul = factory.addPoint(0, YF/2+YS1, 0)
    ml = factory.addPoint(0, YF/2, 0)

    # Bottom part is fluid
    interfaces = [(mr, ml), (ll, lr)]
    lines = [(lr, mr), (ml, ll)]
    
    fluid_lines = [factory.addLine(*p) for p in lines]
    named_lines = dict(zip(('F_right', 'F_left'), fluid_lines))

    interface_lines = [factory.addLine(*p) for p in interfaces]
    named_lines.update(dict(zip(('I_top', 'I_bottom'), interface_lines)))

    fluid_loop = factory.addCurveLoop(sum(zip(fluid_lines, interface_lines), ()))
    fluid = factory.addPlaneSurface([fluid_loop])

    # Solid top
    lines = [(mr, ur), (ur, ul), (ul, ml)]    
    solid_lines = [factory.addLine(*p) for p in lines]
    named_lines.update(dict(zip(('S1_right', 'S1_top', 'S1_left'), solid_lines)))
    
    solid_loop = factory.addCurveLoop(solid_lines + [interface_lines[0]])
    solid1 = factory.addPlaneSurface([solid_loop])

    # Solid bottom
    left = factory.addPoint(0, -YF/2-YS2, 0)
    right = factory.addPoint(X, -YF/2-YS2, 0)

    lines = [(ll, left), (left, right), (right, lr)]    
    solid_lines = [factory.addLine(*p) for p in lines]
    named_lines.update(dict(zip(('S2_left', 'S2_bottom', 'S2_right'), solid_lines)))

    solid_loop = factory.addCurveLoop([interface_lines[1]] + solid_lines)
    solid2 = factory.addPlaneSurface([solid_loop])

    factory.synchronize()
    # gmsh.write('foo.geo_unrolled')
    tags = {'cell': {'F': 1, 'S1': 2, 'S2': 3},
            'facet': {}}
    # Physical tags
    model.addPhysicalGroup(2, [fluid], 1)
    model.addPhysicalGroup(2, [solid1], 2)
    model.addPhysicalGroup(2, [solid2], 3)    

    for name in named_lines:
        tag = named_lines[name]
        model.addPhysicalGroup(1, [tag], tag)
        tags['facet'][name] = tag

    return model, tags


def check_markers(cell_f, facet_f, lookup, geometry_params):
    '''Things are where we expect them'''
    X, YF, YS1, YS2 = (geometry_parameters[k] for k in ('X', 'YF', 'YS1', 'YS2'))
    # Check boundary marking first
    positions = {'F_left': np.array([0., 0.]),
                 'F_right': np.array([X, 0.]),
                 'S1_right': np.array([X, 0.5*YF+0.5*YS1]),
                 'S1_top': np.array([0.5*X, 0.5*YF+YS1]),
                 'S1_left': np.array([0., 0.5*YF+0.5*YS1]),
                 'S2_right': np.array([X, -0.5*YF-0.5*YS2]),
                 'S2_bottom': np.array([0.5*X, -0.5*YF-YS2]),
                 'S2_left': np.array([0., -0.5*YF-0.5*YS2]),                 
                 'I_top': np.array([0.5*X, 0.5*YF]),
                 'I_bottom': np.array([0.5*X, -0.5*YF])
    }

    mesh = facet_f.mesh()
    x = mesh.coordinates()
    
    mesh.init(1, 0)
    e2v = mesh.topology()(1, 0)
    
    tagged_vertices = lambda tag: np.unique(np.hstack(map(e2v, np.where(facet_f.array() == tag)[0])))
    center = lambda coord: 0.5*(np.max(coord, axis=0) + np.min(coord, axis=0))

    for tag in positions:
        target = positions[tag]
        vertices = x[tagged_vertices(lookup['facet'][tag])] 
        assert np.linalg.norm(target - center(vertices)) < 1E-13
    
    # Now subdomain marking
    positions = {'F': np.array([0.5*X, 0]),
                 'S1': np.array([0.5*X, 0.5*YF+0.5*YS1]),
                 'S2': np.array([0.5*X, -0.5*YF-0.5*YS2])}
    
    mesh.init(1, 0)
    c2v = mesh.topology()(2, 0)
    
    tagged_vertices = lambda tag: np.unique(np.hstack(map(c2v, np.where(cell_f.array() == tag)[0])))
    center = lambda coord: np.array([0.5*(np.min(coord[:, 0]) + np.max(coord[:, 0])),
                                     0.5*(np.min(coord[:, 1]) + np.max(coord[:, 1]))])

    for tag in positions:
        target = positions[tag]
        vertices = x[tagged_vertices(lookup['cell'][tag])] 
        assert np.linalg.norm(target - center(vertices)) < 1E-13
                 
    return True

# --------------------------------------------------------------------

if __name__ == '__main__':
    from sleep.mesh import mesh_model2d, load_mesh2d, set_mesh_size
    import dolfin as df
    import sys

    if '-format' not in sys.argv:
        sys.argv.extend(['-format', 'msh2'])
    
    gmsh.initialize(sys.argv)

    model = gmsh.model

    geometry_parameters = {'X': 2, 'YF': 0.5, 'YS1': 0.25, 'YS2': 0.3}
    model, tags = build_model(model, geometry_parameters)

    # Origin, width, inside, outside sizes
    sizes = {'I1': (0+0.5*geometry_parameters['YF']-0.1, 0.2, 0.01, 0.2),
             'S1': (0+0.5*geometry_parameters['YF']+geometry_parameters['YS1']-0.1, 0.1, 0.2, 0.3),
             'I2': (0-0.5*geometry_parameters['YF']-0.05, 0.1, 0.01, 0.2),
             'S2': (0-0.5*geometry_parameters['YF']-geometry_parameters['YS2'], 0.1, 0.4, 0.4)
    }

    field = model.mesh.field
    fid = 1
    boxes = []
    for (y, w, Vin, Vout) in sizes.values():
         field.add('Box', fid)
         field.setNumber(fid, 'XMin', 0)
         field.setNumber(fid, 'XMax', 0+geometry_parameters['X'])
         field.setNumber(fid, 'YMin', y)
         field.setNumber(fid, 'YMax', y+w)
         field.setNumber(fid, 'VIn', Vin)
         field.setNumber(fid, 'VOut', Vout)

         boxes.append(fid)
         fid += 1
    # Combine
    field.add('Min', fid)
    field.setNumbers(fid, 'FieldsList', boxes)    
    field.setAsBackgroundMesh(fid)

    model.occ.synchronize()
    
    # gmsh.fltk.initialize()
    # gmsh.fltk.run()
    
    h5_filename = './test/two_solid_domain.h5'
    mesh_model2d(model, tags, h5_filename)

    mesh, markers, lookup = load_mesh2d(h5_filename)
    cell_f, facet_f = markers
    
    check_markers(cell_f, facet_f, lookup, geometry_parameters)
    
    gmsh.finalize()
    
    df.File('./test/two_solid_cells.pvd') << cell_f
    df.File('./test/two_solid_facets.pvd') << facet_f

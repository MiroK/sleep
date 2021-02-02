#  |-----------------------|(S2t)
#  YS2                     |  Biot
#  |-----------------------| <------ meant to be a material param interface (Itop)
#  YS1                     |  Biot
#  |-----------------------|(Ibottom)
#  YF                      |  Stokes
#  |---------X-------------|(S1b)
# (0, 0)
import numpy as np
import gmsh


def build_model(model, geometry_parameters):
    '''GMSH model of sleep 2d study'''

    X, YF, YS1, YS2 = (geometry_parameters[k] for k in ('X', 'YF', 'YS1', 'YS2'))
    assert X > 0 and YF > 0 and YS1 > 0 and YS2 > 0
    
    factory = model.occ
    A = factory.addPoint(0, 0, 0)
    B = factory.addPoint(0, YF, 0)
    C = factory.addPoint(0, YF+YS1, 0)
    D = factory.addPoint(0, YF+YS1+YS2, 0)

    a = factory.addPoint(X, 0, 0)
    b = factory.addPoint(X, YF, 0)
    c = factory.addPoint(X, YF+YS1, 0)
    d = factory.addPoint(X, YF+YS1+YS2, 0)

    fluid_lines = [factory.addLine(*p) for p in ((B, A), (A, a), (a, b))]
    named_lines = dict(zip(('F_left', 'F_bottom', 'F_right'), fluid_lines))

    interface_lines = [factory.addLine(*p) for p in ((b, B), (C, c))]
    named_lines.update(dict(zip(('I_bottom', 'I_top'), interface_lines)))

    fluid_loop = factory.addCurveLoop(fluid_lines + [interface_lines[0]])
    fluid = factory.addPlaneSurface([fluid_loop])

    solid1_lines = [factory.addLine(*p) for p in ((B, C), (c, b))]
    named_lines.update(dict(zip(('S1_left', 'S1_right'), solid1_lines)))
    
    solid1_loop = factory.addCurveLoop(sum(zip(interface_lines, solid1_lines), ()))
    solid1 = factory.addPlaneSurface([solid1_loop])

    solid2_lines = [factory.addLine(*p) for p in ((c, d), (d, D), (D, C))]
    named_lines.update(dict(zip(('S2_right', 'S2_top', 'S2_left'), solid2_lines)))

    solid2_loop = factory.addCurveLoop(solid2_lines + [interface_lines[1]])
    solid2 = factory.addPlaneSurface([solid2_loop])

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
    positions = {'F_left': np.array([0., 0.5*YF]),
                 'F_right': np.array([X, 0.5*YF]),
                 'F_bottom': np.array([0.5*X, 0]),
                 'S1_right': np.array([X, YF+0.5*YS1]),
                 'S1_left': np.array([0., YF+0.5*YS1]),
                 'S2_right': np.array([X, YF+YS1+0.5*YS2]),
                 'S2_left': np.array([0., YF+YS1+0.5*YS2]),
                 'S2_top': np.array([0.5*X, YF+YS1+YS2]),
                 'I_top': np.array([0.5*X, YF+YS1]),
                 'I_bottom': np.array([0.5*X, YF])
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
        assert np.linalg.norm(target - center(vertices)) < 1E-13, tag
    
    # Now subdomain marking
    positions = {'F': np.array([0.5*X, 0.5*YF]),
                 'S1': np.array([0.5*X, YF+0.5*YS1]),
                 'S2': np.array([0.5*X, YF+YS1+0.5*YS2])}
    
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

    geometry_parameters = {'X': 500e-4, 'YF': 20e-4, 'YS1': 1e-4, 'YS2': 99e-4}
    model, tags = build_model(model, geometry_parameters)


    Number_cells_vertical_S1=5
    # Origin, width, inside, outside sizes
    sizes = {'I_bottom': (0+geometry_parameters['YF']-geometry_parameters['YF']/10, +geometry_parameters['YF']+geometry_parameters['YS1']/2, geometry_parameters['YS1']/Number_cells_vertical_S1,geometry_parameters['X']/10,geometry_parameters['YF']),
             'I_top': (0+geometry_parameters['YF']+geometry_parameters['YS1']-geometry_parameters['YS1']/2, 0+geometry_parameters['YF']+geometry_parameters['YS1']+geometry_parameters['YS1'], geometry_parameters['YS1']/Number_cells_vertical_S1, geometry_parameters['X']/10,geometry_parameters['YS2'])
    }
    field = model.mesh.field
    fid = 1
    boxes = []
    for (ymin, ymax, Vin, Vout,t) in sizes.values():
         field.add('Box', fid)
         field.setNumber(fid, 'XMin', 0)
         field.setNumber(fid, 'XMax', 0+geometry_parameters['X'])
         field.setNumber(fid, 'YMin', ymin)
         field.setNumber(fid, 'YMax', ymax)
         field.setNumber(fid, 'VIn', Vin)
         field.setNumber(fid, 'VOut', Vout)
         field.setNumber(fid, 'Thickness', t)

         boxes.append(fid)
         fid += 1
    # Combine
    field.add('Min', fid)
    field.setNumbers(fid, 'FieldsList', boxes)    
    field.setAsBackgroundMesh(fid)

    model.occ.synchronize()
    
    # gmsh.fltk.initialize()
    # gmsh.fltk.run()
    
    h5_filename = './test/fbb_domain.h5'
    mesh_model2d(model, tags, h5_filename)

    mesh, markers, lookup = load_mesh2d(h5_filename)
    cell_f, facet_f = markers
    
    check_markers(cell_f, facet_f, lookup, geometry_parameters)
    
    gmsh.finalize()
    
    df.File('./test/fbb_cells.pvd') << cell_f
    df.File('./test/fbb_facets.pvd') << facet_f

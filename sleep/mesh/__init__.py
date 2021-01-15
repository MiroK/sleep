from sleep.mesh.dolfinconvert import convert
import os, gmsh, json
import dolfin as df


def mesh_model2d(model, tags, h5_filename):
    '''Generate fully configured model mesh as h5 file'''
    base, ext = os.path.splitext(h5_filename)
    assert ext == '.h5'

    gmsh.option.setNumber("General.Terminal", 1)
    
    model.occ.synchronize()
    model.mesh.generate(2)
    model.mesh.optimize('')
    
    gmsh_file = '.'.join([base, 'msh'])
    d = os.path.dirname(gmsh_file)
    not os.path.exists(d) and os.mkdir(d)
    
    gmsh.write(gmsh_file)
    # Convert to h5 and remove
    h5_file = convert(gmsh_file)
    
    json_file = '.'.join([base, 'json'])
    with open(json_file, 'w') as handle:
        json.dump(tags, handle)

    return h5_file, json_file


def load_mesh2d(h5_file, json_file=None):
    '''
    Try loading what is needed for simulations; mesh, facet function
    and the mapping from nameed surfaces to tags in facet function
    '''
    base, ext = os.path.splitext(h5_file)
    assert ext == '.h5'
    
    if json_file is None:
        json_file = '.'.join([base, 'json'])
    assert os.path.exists(json_file)

    mesh = df.Mesh()
    h5 = df.HDF5File(mesh.mpi_comm(), h5_file, 'r')
    h5.read(mesh, 'mesh', False)

    cell_f = df.MeshFunction('size_t', mesh, mesh.topology().dim(), 0)
    h5.read(cell_f, 'volumes')

    facet_f = df.MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
    h5.read(facet_f, 'surfaces')
    
    # Otherwise load as dict
    with open(json_file, 'r') as f:
        mapping = json.load(f)

    return (mesh, (cell_f, facet_f), mapping)


def set_mesh_size(model, tags, mesh_sizes):
    '''Set based on distance from facet'''
    # Now set mesh size
    field = model.mesh.field

    thresholds = []
    field_id = 1
    for name, tag in tags['facet'].items():
        # User
        all_okay = True
        for p in ('LcMin', 'LcMax', 'DistMin', 'DistMax'):
            line_sizes = mesh_sizes.get(name, {})
            if line_sizes:
                size = line_sizes[p]
            else:
                size = mesh_sizes.get(p, -1)
            all_okay = all_okay and size > 0

        if all_okay:
            field.add('Distance', field_id)
            field.setNumbers(field_id, 'FacesList', [tag])
                    
            # Then we set the mesh size based on that
            field_id += 1
            thresholds.append(field_id)
            field.add('Threshold', field_id)
            field.setNumber(field_id, 'IField', field_id-1)
            
            # User
            print('Setting', name, tag, 'field', field_id)
            for p in ('LcMin', 'LcMax', 'DistMin', 'DistMax'):
                line_sizes = mesh_sizes.get(name, {})
                if line_sizes:
                    size = line_sizes[p]
                else:
                    size = mesh_sizes.get(p)
                
                field.setNumber(field_id, p, size)
                print('\t', p, size)
    
            field_id += 1
    # Combine
    print('Thresholds', thresholds)
    
    field.add('Min', field_id)
    field.setNumbers(field_id, 'FieldsList', thresholds)
    field.setAsBackgroundMesh(field_id)

    model.occ.synchronize()
    # model.geo.synchronize()    

    return model

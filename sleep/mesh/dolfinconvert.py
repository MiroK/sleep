import sys, os
#from brainpulse.fem.
from .mshconvert import convert2xml
from dolfin import Mesh, HDF5File, MeshFunction, info


def convert(msh_file):
    '''Temporary version of convertin from msh to h5'''
    root, _ = os.path.splitext(msh_file)
    assert os.path.splitext(msh_file)[1] == '.msh'
    
    # Get the xml mesh
    xml_file = '.'.join([root, 'xml'])

    # Convert to XML
    convert2xml(msh_file, xml_file)

    # Success?
    assert os.path.exists(xml_file)

    mesh = Mesh(xml_file)
    h5_file = '.'.join([root, 'h5'])
    out = HDF5File(mesh.mpi_comm(), h5_file, 'w')
    out.write(mesh, 'mesh')

    # Save ALL data as facet_functions
    data_sets = ('curves', 'surfaces', 'volumes')
    regions = ('curve_region.xml', 'facet_region.xml', 'volume_region.xml')

    for data_set, region in zip(data_sets, regions):
        r_xml_file = '_'.join([root, region])

        if os.path.exists(r_xml_file):
            f = MeshFunction('size_t', mesh, r_xml_file)
            out.write(f, data_set)
            # And clean
            os.remove(r_xml_file)
    # and clean
    os.remove(xml_file)

    return h5_file

    
def cleanup(files=None, exts=(), dir='.'):
    '''Get rid of xml'''
    if files is not None:
        return map(os.remove, files)
    else:
        files = [os.path.join(dir, f) for f in filter(lambda f: any(map(f.endswith, exts)), os.listdir(dir))]
        info('Removing %r' % files) 
        return cleanup(files)
                    
# --------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    from dolfin import File, info, mpi_comm_world
    from xii import EmbeddedMesh

    parser = argparse.ArgumentParser(description='Convert msh file to h5')
    parser.add_argument('input', type=str, help='input msh file')
    parser.add_argument('--cleanup', type=str, nargs='+',
                        help='extensions to delete', default=('.xml', ))

    # Save the mesh markers for visualizing
    save_pvd_parser = parser.add_mutually_exclusive_group(required=False)
    save_pvd_parser.add_argument('--save_pvd', dest='save_pvd', action='store_true')
    save_pvd_parser.add_argument('--no_save_pvd', dest='save_pvd', action='store_false')
    parser.set_defaults(save_pvd=False)

    mesh_size_parser = parser.add_mutually_exclusive_group(required=False)
    mesh_size_parser.add_argument('--mesh_size', dest='mesh_size', action='store_true')
    mesh_size_parser.add_argument('--no_mesh_size', dest='mesh_size', action='store_false')
    parser.set_defaults(mesh_size=False)

    args = parser.parse_args()

    h5_file = convert(args.input)
    
    # VTK visualize tags
    if args.save_pvd or args.mesh_size:
        h5 = HDF5File(mpi_comm_world(), h5_file, 'r')
        mesh = Mesh()
        h5.read(mesh, 'mesh', False)

        info('Mesh has %d cells' % mesh.num_cells())
        info('Mesh has %d vertices' % mesh.num_vertices())
        info('Box size %s' % (mesh.coordinates().max(axis=0)-mesh.coordinates().min(axis=0)))

        hmin, hmax = mesh.hmin(), mesh.hmax()
        info('Mesh has sizes %g %g' % (hmin, hmax))
        
        root = os.path.splitext(args.input)[0]
        tdim = mesh.topology().dim()
        
        data_sets = ('curves', 'surfaces', 'volumes')
        dims = (1, tdim-1, tdim)
        for ds, dim in zip(data_sets, dims):
            if h5.has_dataset(ds):
                f = MeshFunction('size_t', mesh, dim, 0)
                h5.read(f, ds)

                if args.save_pvd:
                    File('%s_%s.pvd' % (root, ds)) << f

    cleanup(exts=args.cleanup)

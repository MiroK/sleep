from sleep.utils.make_mesh_cpp import make_mesh
from collections import defaultdict
from itertools import chain
import dolfin as df
import numpy as np
import operator


class EmbeddedMesh(df.Mesh):
    '''
    Construct a mesh of marked entities in marking_function.
    The output is the mesh with cell function which inherited the markers. 
    and an antribute `parent_entity_map` which is dict with a map of new 
    mesh vertices to the old ones, and new mesh cells to the old mesh entities.
    Having several maps in the dict is useful for mortaring.
    '''
    def __init__(self, marking_function, markers):
        if not isinstance(markers, (list, tuple)): markers = [markers]
        
        # Convenience option to specify only subdomains
        is_number = lambda m: isinstance(m, int)
        assert all(map(is_number, markers))
            
        base_mesh = marking_function.mesh()

        assert base_mesh.topology().dim() >= marking_function.dim()

        # Work in serial only (much like submesh)
        assert df.MPI.size(base_mesh.mpi_comm()) == 1

        gdim = base_mesh.geometry().dim()
        tdim = marking_function.dim()
        assert tdim > 0, 'No Embedded mesh from vertices'

        assert markers, markers

        # We reuse a lot of Submesh capabilities if marking by cell_f
        if base_mesh.topology().dim() == marking_function.dim():
            # Submesh works only with one marker so we conform
            color_array = marking_function.array()
            color_cells = dict((m, np.where(color_array == m)[0]) for m in markers)

            # So everybody is marked as 1
            one_cell_f = df.MeshFunction('size_t', base_mesh, tdim, 0)
            for cells in color_cells.itervalues(): one_cell_f.array()[cells] = 1
            
            # The Embedded mesh now steals a lot from submesh
            submesh = df.SubMesh(base_mesh, one_cell_f, 1)

            df.Mesh.__init__(self, submesh)

            # The entity mapping attribute;
            # NOTE: At this point there is not reason to use a dict as
            # a lookup table
            mesh_key = marking_function.mesh().id()
            mapping_0 = submesh.data().array('parent_vertex_indices', 0)

            mapping_tdim = submesh.data().array('parent_cell_indices', tdim)
            self.parent_entity_map = {mesh_key: {0: dict(enumerate(mapping_0)),
                                                 tdim: dict(enumerate(mapping_tdim))}}
            # Finally it remains to preserve the markers
            f = df.MeshFunction('size_t', self, tdim, 0)
            f_values = f.array()
            if len(markers) > 1:
                old2new = dict(zip(mapping_tdim, range(len(mapping_tdim))))
                for color, old_cells in color_cells.iteritems():
                    new_cells = np.array([old2new[o] for o in old_cells], dtype='uintp')
                    f_values[new_cells] = color
            else:
                f.set_all(markers[0])
            
            self.marking_function = f
            # https://stackoverflow.com/questions/2491819/how-to-return-a-value-from-init-in-python            
            return None  

        # Otherwise the mesh needs to by build from scratch
        base_mesh.init(tdim, 0)
        # Collect unique vertices based on their new-mesh indexing, the cells
        # of the embedded mesh are defined in terms of their embedded-numbering
        new_vertices, new_cells = [], []
        # NOTE: new_vertices is actually new -> old vertex map
        # Map from cells of embedded mesh to tdim entities of base mesh, and
        cell_map = []
        cell_colors = defaultdict(list)  # Preserve the markers

        new_cell_index, new_vertex_index = 0, 0
        for marker in markers:
            for entity in df.SubsetIterator(marking_function, marker):
                vs = entity.entities(0)
                cell = []
                # Vertex lookup
                for v in vs:
                    try:
                        local = new_vertices.index(v)
                    except ValueError:
                        local = new_vertex_index
                        new_vertices.append(v)
                        new_vertex_index += 1
                    # Cell, one by one in terms of vertices
                    cell.append(local)
                # The cell
                new_cells.append(cell)
                # Into map
                cell_map.append(entity.index())
                # Colors
                cell_colors[marker].append(new_cell_index)

                new_cell_index += 1
        vertex_coordinates = base_mesh.coordinates()[new_vertices]
        new_cells = np.array(new_cells, dtype='uintp')
        
        # With acquired data build the mesh
        df.Mesh.__init__(self)
        # Fill
        make_mesh(coordinates=vertex_coordinates, cells=new_cells, tdim=tdim, gdim=gdim,
                  mesh=self)

        # The entity mapping attribute
        mesh_key = marking_function.mesh().id()
        self.parent_entity_map = {mesh_key: {0: dict(enumerate(new_vertices)),
                                             tdim: dict(enumerate(cell_map))}}

        f = df.MeshFunction('size_t', self, tdim, 0)
        f_ = f.array()
        # Finally the inherited marking function
        if len(markers) > 1:
            for marker, cells in cell_colors.iteritems():
                f_[cells] = marker
        else:
            f.set_all(markers[0])

        self.marking_function = f


def embed_mesh(child_mesh, parent_mesh, TOL=1E-8):
    '''
    Provided that child_mesh is some 'restriction' mesh of parent compute 
    embedding of vertices and cells of child to enitties of parent
    '''
    assert child_mesh.topology().dim() < parent_mesh.topology().dim()
    assert child_mesh.geometry().dim() == parent_mesh.geometry().dim()
    
    child_x = child_mesh.coordinates().tolist()
    tree = child_mesh.bounding_box_tree()

    parent_x = parent_mesh.coordinates()
    maybe = []
    # Let's see about vertex emebedding - reduce to candidates
    for i, xi in enumerate(parent_x):
        tree.collides(df.Point(xi)) and maybe.append(i)
    assert maybe
    print('Checking {} / {} parent vertices'.format(len(maybe), len(parent_x)))
    maybe_x = parent_x[maybe]

    vertex_mapping = -1*np.ones(len(child_x), dtype=int)
    tol = child_mesh.hmin()*TOL
    # Try to compute pairings
    for child_id, xi in enumerate(child_x):
        distances = np.linalg.norm(maybe_x - xi, 2, axis=1)
        j = np.argmin(distances)
        assert distances[j] < tol
        parent_id = maybe[j]

        vertex_mapping[child_id] = parent_id
    # Done
    assert np.all(vertex_mapping > -1), vertex_mapping

    # Now the candidate entities of parent are those connected to paired
    # vertices
    tdim = child_mesh.topology().dim()
    parent_mesh.init(tdim, 0)
    parent_mesh.init(0, tdim)

    e2v, v2e = parent_mesh.topology()(tdim, 0), parent_mesh.topology()(0, tdim)
    inverse_vertex_mapping = dict(zip(vertex_mapping, np.arange(len(vertex_mapping))))

    cell_mapping = -1*np.ones(child_mesh.num_cells(), dtype=int)
    for idx, child_cell in enumerate(child_mesh.cells()):
        # The cell is given in terms of vertices
        as_parent = set(vertex_mapping[child_cell])

        child_cell = set(child_cell)  # Target
        # Compare to all entities connected to parant_vertices
        entities = chain(*(v2e(v) for v in as_parent))
        found = False
        while not found:
            entity = next(entities)
            # Get it in child_numering
            child_entity = set(inverse_vertex_mapping.get(w, -1) for w in e2v(entity))
            if child_cell == child_entity:
                found = True
                cell_mapping[idx] = entity
    # Done ? 
    assert np.all(cell_mapping > -1)
    
    # For compatibility these are dicts
    return {0: dict(enumerate(vertex_mapping)), tdim: dict(enumerate(cell_mapping))}
    
# --------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import *

    mesh = UnitSquareMesh(64, 32)
    tmesh = BoundaryMesh(mesh, 'exterior')

    mappings = embed_mesh(tmesh, mesh)
    # The coordinates match
    xi, yi = map(list, zip(*mappings[0].items()))
    assert np.linalg.norm(tmesh.coordinates()[xi] - mesh.coordinates()[yi]) < 1E-13

    # Cells match coordinate by coordinates
    tx = tmesh.coordinates()
    x = mesh.coordinates()

    mesh.init(1, 0)
    e2v = mesh.topology()(1, 0)
    for ti, tcell in enumerate(tmesh.cells()):
        # Child coordinates
        cell = tx[tcell]
        entity = x[e2v(mappings[1][ti])]

        for xi in cell:
            assert np.min(np.linalg.norm(entity - xi, 2, axis=1)) < 1E-13, (entity, cell)
        
        
    

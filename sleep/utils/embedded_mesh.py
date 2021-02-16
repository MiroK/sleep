# This is taken from FEniCS_ii to keep the code base free of many depen-
# dencies
from sleep.utils.make_mesh_cpp import make_mesh
from collections import defaultdict
from itertools import chain, permutations
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
        
        base_mesh = marking_function.mesh()
        assert base_mesh.topology().dim() >= marking_function.dim()
        # Work in serial only (much like submesh)
        assert df.MPI.size(base_mesh.mpi_comm()) == 1

        gdim = base_mesh.geometry().dim()
        tdim = marking_function.dim()
        assert tdim > 0, 'No Embedded mesh from vertices'

        assert markers, markers

        # NOTE: treating submesh as a separate case is done for performance
        # as it seems that pure python as done below is about 2x slower
        # We reuse a lot of Submesh capabilities if marking by cell_f
        if base_mesh.topology().dim() == marking_function.dim():
            # Submesh works only with one marker so we conform
            color_array = marking_function.array()
            color_cells = dict((m, np.where(color_array == m)[0]) for m in markers)

            # So everybody is marked as 1
            one_cell_f = df.MeshFunction('size_t', base_mesh, tdim, 0)
            for cells in color_cells.values(): one_cell_f.array()[cells] = 1
            
            # The Embedded mesh now steals a lot from submesh
            submesh = df.SubMesh(base_mesh, one_cell_f, 1)

            df.Mesh.__init__(self, submesh)

            # The entity mapping attribute;
            # NOTE: At this point there is not reason to use a dict as
            # a lookup table            
            mapping_0 = submesh.data().array('parent_vertex_indices', 0)
            mapping_tdim = submesh.data().array('parent_cell_indices', tdim)

            mesh_key = marking_function.mesh().id()            
            self.parent_entity_map = {mesh_key: {0: dict(enumerate(mapping_0)),
                                                 tdim: dict(enumerate(mapping_tdim))}}
            # Finally it remains to preserve the markers
            f = df.MeshFunction('size_t', self, tdim, 0)
            f_values = f.array()
            if len(markers) > 1:
                old2new = dict(zip(mapping_tdim, range(len(mapping_tdim))))
                for color, old_cells in color_cells.items():
                    new_cells = np.array([old2new[o] for o in old_cells], dtype='uintp')
                    f_values[new_cells] = color
            else:
                f.set_all(markers[0])
            
            self.marking_function = f
            # Declare which tagged cells are found
            self.tagged_cells = set(markers)
            # https://stackoverflow.com/questions/2491819/how-to-return-a-value-from-init-in-python            
            return None  

        # Otherwise the mesh needs to by build from scratch
        _, e2v = (base_mesh.init(tdim, 0), base_mesh.topology()(tdim, 0))
        entity_values = marking_function.array()
        colorings = [np.where(entity_values == tag)[0] for tag in markers]
        # Represent the entities as their vertices
        tagged_entities = np.hstack(colorings)

        tagged_entities_v = np.array([e2v(e) for e in tagged_entities], dtype='uintp')
        # Unique vertices that make them up are vertices of our mesh
        tagged_vertices = np.unique(tagged_entities_v.flatten())
        # Representing the entities in the numbering of the new mesh will
        # give us the cell makeup
        mapping = dict(zip(tagged_vertices, range(len(tagged_vertices))))
        # So these are our new cells
        tagged_entities_v.ravel()[:] = np.fromiter((mapping[v] for v in tagged_entities_v.flat),
                                                   dtype='uintp')
        
        # With acquired data build the mesh
        df.Mesh.__init__(self)
        # Fill
        vertex_coordinates = base_mesh.coordinates()[tagged_vertices]
        make_mesh(coordinates=vertex_coordinates, cells=tagged_entities_v, tdim=tdim, gdim=gdim,
                  mesh=self)

        # The entity mapping attribute
        mesh_key = marking_function.mesh().id()
        self.parent_entity_map = {mesh_key: {0: dict(enumerate(tagged_vertices)),
                                             tdim: dict(enumerate(tagged_entities))}}

        f = df.MeshFunction('size_t', self, tdim, 0)
        # Finally the inherited marking function. We colored sequentially so
        if len(markers) > 1:
            f_ = f.array()            
            offsets = np.cumsum(np.r_[0, list(map(len, colorings))])
            for i, marker in enumerate(markers):
                f_[offsets[i]:offsets[i+1]] = marker
        else:
            f.set_all(markers[0])

        self.marking_function = f
        # Declare which tagged cells are found
        self.tagged_cells = set(markers)

    def compute_embedding(self, other_entity_f, tags, tol=1E-10):
        '''
        Compute how self can be viewed as en embeded mesh of other_entity_f.mesh 
        for entities which have the tag.
        '''
        # The use case I have in mind is when we declare [L|R] interface
        # based on L and in the system assembly the view of I from R is needed.
        # Then a 'blind' search is needed because we threw away informations.
        # So that's we avoid here
        tdim = self.topology().dim()
        assert tdim == other_entity_f.dim()
        assert self.geometry().dim() == other_entity_f.mesh().geometry().dim()
        
        parent_mesh = other_entity_f.mesh()
        if parent_mesh.id() in self.parent_entity_map:
            raise ValueError('There is a mapping for {} already'.format(parent_mesh.id()))
        
        # To pair cells with entitities ...
        c2v = self.topology()(tdim, 0)
        # Use vertex comparison
        parent_mesh.init(tdim, 0)
        e2v = parent_mesh.topology()(tdim, 0)

        if isinstance(tags, int): tags = [tags]

        tagged_entities = np.hstack([np.where(other_entity_f.array() == tag)[0] for tag in tags])
        assert len(tagged_entities)

        tree = self.bounding_box_tree()        
        # Collision by coordinates
        x, x_parent = self.coordinates(), parent_mesh.coordinates()

        entity_mapping, vertex_mapping = {}, {}
        for entity in tagged_entities:
            # We need to be able to embed all vertices in order to get a cell
            entity_vertices = e2v(entity).tolist()
            # First seek self cells that collide with the vertex.
            v_cells = set(sum((list(tree.compute_collisions(df.Point(x_parent[v])))
                               for v in entity_vertices), []))
            
            emp = np.mean(x_parent[entity_vertices], axis=0)
            # The one closest by midpoint criterion better embed the entity
            v_cells = sorted(v_cells,
                             key = lambda c: np.linalg.norm(emp - np.mean(x[c2v(c)], axis=0)),
                             reverse=True)
            
            found = False
            # Ideally only one pop is needed as the last(v_cells) is the
            # best match in terms of midpoints. That cell should be able
            # embed the vertices
            while v_cells and not found:
                the_cell = v_cells.pop()
                cell_vertices = c2v(the_cell)
                cell_vertices_x = x[cell_vertices]
                
                # And now pair the vertices
                entity_vertices = e2v(entity)
                entity_vertices_x = x_parent[entity_vertices]
                # There must exist a permutation of cell vertices that matches
                # the entity
                perms = permutations(range(len(cell_vertices)), len(cell_vertices))
                while not found:
                    perm = list(next(perms))
                    found = np.linalg.norm(entity_vertices_x - cell_vertices_x[perm]) < tol
                    
            assert found, 'Embedding failed'
            # Set mappings
            for ev, cv in zip(entity_vertices, cell_vertices[perm]):
                vertex_mapping[cv] = ev
            entity_mapping[the_cell] = entity

        self.parent_entity_map[parent_mesh.id()] = {0: vertex_mapping,
                                                    tdim: entity_mapping}

        return self.parent_entity_map[parent_mesh.id()]

    def translate_markers(self, entity_f, tags=None):
        '''For entity_f.mesh being parent of self tranlate markers'''
        assert entity_f.mesh().id() in self.parent_entity_map
        assert 0 < entity_f.dim() < self.topology().dim()
        
        if tags is None:
            tags = np.unique(entity_f.array())
        if isinstance(tags, int):
            tags = (tags, )

        emesh = entity_f.mesh()
        entity_dim = entity_f.dim()
        cell_dim = self.topology().dim()
        # Entity is connected to parent cells, some of these we can map to
        # from child mesh as cell. Some of its entities is the entity. This
        # is to be determined by vertices
        _, e2v_parent = (emesh.init(entity_dim, 0), emesh.topology()(entity_dim, 0))        
        _, e2c = (emesh.init(entity_dim, cell_dim), emesh.topology()(entity_dim, cell_dim))
        _, c2e = (self.init(cell_dim, entity_dim), self.topology()(cell_dim, entity_dim))
        _, e2v = (self.init(entity_dim, 0), self.topology()(entity_dim, 0))        

        ivertex_mapping = dict((v, k) for k, v in self.parent_entity_map[emesh.id()][0].items())
        icell_mapping = dict((v, k) for k, v in self.parent_entity_map[emesh.id()][cell_dim].items())

        marker_f = df.MeshFunction('size_t', self, entity_dim, 0)
        for tag in tags:
            entities, = np.where(entity_f.array() == tag)  # parent
            # We encode them as vertices in the child
            as_vertices = [set(ivertex_mapping.get(v, -1) for v in e2v_parent(e)) for e in entities]
            # The above was an attempt. Continue with those that could be embeded
            for e, as_vertex in zip(entities, as_vertices):
                if any(v == -1 for v in as_vertex): continue

                parent_cells = [c for c in e2c(e) if c in icell_mapping]

                found = None
                while not found and parent_cells:
                    child_entities = c2e(icell_mapping[parent_cells.pop()])
                    matches = [e_ for e_ in child_entities if set(e2v(e_)) == as_vertex]

                    found = bool(matches)
                    e_, = matches
                    if found:
                        marker_f[int(e_)] = entity_f[e]

        return marker_f


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
        tree.compute_collisions(df.Point(xi)) and maybe.append(i)
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
        
        
    

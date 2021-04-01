""" Module for converting various mesh formats."""
# Stripped down to msh from dolfin-convert

import os.path
from dolfin_utils.meshconvert import xml_writer
import numpy


def gmsh2xml(ifilename, handler):
    """Convert between .gmsh v2.0 format (http://www.geuz.org/gmsh/) and .xml,
    parser implemented as a state machine:

        0 = read 'MeshFormat'
        1 = read  mesh format data
        2 = read 'EndMeshFormat'
        3 = read 'Nodes'
        4 = read  number of vertices
        5 = read  vertices
        6 = read 'EndNodes'
        7 = read 'Elements'
        8 = read  number of cells
        9 = read  cells
        10 = done

    Afterwards, extract physical region numbers if they are defined in
    the mesh file as a mesh function.

    """

    print("Converting from Gmsh format (.msh, .gmsh) to DOLFIN XML format")

    # The dimension of the gmsh element types supported here as well as the dolfin cell types for each dimension
    gmsh_dim = {15: 0, 1: 1, 2: 2, 4: 3}
    cell_type_for_dim = {1: "interval", 2: "triangle", 3: "tetrahedron" }
    # the gmsh element types supported for conversion
    supported_gmsh_element_types = [1, 2, 4, 15]

    # Open files
    ifile = open(ifilename, "r")

    # Scan file for cell type
    cell_type = None
    highest_dim = 0
    line = ifile.readline()
    while line:

        # Remove newline
        if line[-1] == "\n":
            line = line[:-1]

        # Read dimension
        if line.find("$Elements") == 0:

            line = ifile.readline()
            num_elements = int(line)
            if num_elements == 0:
                _error("No elements found in gmsh file.")
            line = ifile.readline()

            # Now iterate through elements to find largest dimension.  Gmsh
            # format might include elements of lower dimensions in the element list.
            # We also need to count number of elements of correct dimensions.
            # Also determine which vertices are not used.
            dim_count = {0: 0, 1: 0, 2: 0, 3: 0}
            vertices_used_for_dim = {0: [], 1: [], 2: [], 3: []}
            # Array used to store gmsh tags for 1D (type 1/line), 2D (type 2/triangular) elements and 3D (type 4/tet) elements
            tags_for_dim = {0: [], 1: [], 2: [], 3: []}

            while line.find("$EndElements") == -1:
                element = line.split()
                elem_type = int(element[1])
                num_tags = int(element[2])
                if elem_type in supported_gmsh_element_types:
                    dim = gmsh_dim[elem_type]
                    if highest_dim < dim:
                        highest_dim = dim
                    node_num_list = [int(node) for node in element[3 + num_tags:]]
                    vertices_used_for_dim[dim].extend(node_num_list)
                    if num_tags > 0:
                        tags_for_dim[dim].append(tuple(int(tag) for tag in element[3:3+num_tags]))
                    dim_count[dim] += 1
                else:
                    #TODO: output a warning here. "gmsh element type %d not supported" % elem_type
                    pass
                line = ifile.readline()
        else:
            # Read next line
            line = ifile.readline()

    # Check that we got the cell type and set num_cells_counted
    if highest_dim == 0:
        _error("Unable to find cells of supported type.")

    num_cells_counted = dim_count[highest_dim]
    vertex_set = set(vertices_used_for_dim[highest_dim])
    vertices_used_for_dim[highest_dim] = None

    vertex_dict = {}
    for n,v in enumerate(vertex_set):
        vertex_dict[v] = n

    # Step to beginning of file
    ifile.seek(0)

    # Set mesh type
    handler.set_mesh_type(cell_type_for_dim[highest_dim], highest_dim)

    # Initialise node list (gmsh does not export all vertexes in order)
    nodelist = {}

    # Current state
    state = 0

    # Write data
    num_vertices_read = 0
    num_cells_read = 0

    # Only import the dolfin objects if facet markings exist
    process_facets = False
    if any(len(tags_for_dim[dim]) > 0 for dim in (highest_dim-1, 1)):
        # first construct the mesh
        try:
            from dolfin import MeshEditor, Mesh
        except ImportError:
            _error("DOLFIN must be installed to handle Gmsh boundary regions")
        mesh = Mesh()
        mesh_editor = MeshEditor()
        cell_type = {1: 'interval', 2: 'triangle', 3: 'tetrahedron'}[highest_dim]
        mesh_editor.open(mesh, cell_type, highest_dim, highest_dim)
        process_facets = True
    else:
        # TODO: Output a warning or an error here
        me = None

    while state != 10:

        # Read next line
        line = ifile.readline()
        if not line: break

        # Skip comments
        if line[0] == '#':
            continue

        # Remove newline
        if line[-1] == "\n":
            line = line[:-1]

        if state == 0:
            if line == "$MeshFormat":
                state = 1
        elif state == 1:
            (version, file_type, data_size) = line.split()
            state = 2
        elif state == 2:
            if line == "$EndMeshFormat":
                state = 3
        elif state == 3:
            if line == "$Nodes":
                state = 4
        elif state == 4:
            num_vertices = len(vertex_dict)
            handler.start_vertices(num_vertices)
            if process_facets:
                mesh_editor.init_vertices_global(num_vertices, num_vertices)
            state = 5
        elif state == 5:
            (node_no, x, y, z) = line.split()
            node_no = int(node_no)
            x,y,z = [float(xx) for xx in (x,y,z)]
            if node_no in vertex_dict:
                node_no = vertex_dict[node_no]
            else:
                continue
            nodelist[int(node_no)] = num_vertices_read
            handler.add_vertex(num_vertices_read, [x, y, z])
            if process_facets:
                if highest_dim == 1:
                    coords = numpy.array([x])
                elif highest_dim == 2:
                    coords = numpy.array([x, y])
                elif highest_dim == 3:
                    coords = numpy.array([x, y, z])
                mesh_editor.add_vertex(num_vertices_read, coords)

            num_vertices_read +=1

            if num_vertices == num_vertices_read:
                handler.end_vertices()
                state = 6
        elif state == 6:
            if line == "$EndNodes":
                state = 7
        elif state == 7:
            if line == "$Elements":
                state = 8
        elif state == 8:
            handler.start_cells(num_cells_counted)
            if process_facets:
                mesh_editor.init_cells_global(num_cells_counted, num_cells_counted)

            state = 9
        elif state == 9:
            element = line.split()
            elem_type = int(element[1])
            num_tags  = int(element[2])
            if elem_type in supported_gmsh_element_types:
                dim = gmsh_dim[elem_type]
            else:
                dim = 0
            if dim == highest_dim:
                node_num_list = [vertex_dict[int(node)] for node in element[3 + num_tags:]]
                for node in node_num_list:
                    if not node in nodelist:
                        _error("Vertex %d of %s %d not previously defined." %
                              (node, cell_type_for_dim[dim], num_cells_read))
                cell_nodes = [nodelist[n] for n in node_num_list]
                handler.add_cell(num_cells_read, cell_nodes)

                if process_facets:
                    cell_nodes = numpy.array([nodelist[n] for n in node_num_list], dtype=numpy.uintp)
                    mesh_editor.add_cell(num_cells_read, cell_nodes)

                num_cells_read +=1

            if num_cells_counted == num_cells_read:
                handler.end_cells()
                if process_facets:
                    mesh_editor.close()
                state = 10
        elif state == 10:
            break

    # Write mesh function based on the Physical Regions defined by
    # gmsh, but only if they are not all zero. All zero physical
    # regions indicate that no physical regions were defined.
    if highest_dim not in [1,2,3]:
        _error("Gmsh tags not supported for dimension %i. Probably a bug" % dim)

    tags = tags_for_dim[highest_dim]
    physical_regions = tuple(tag[0] for tag in tags)
    if not all(tag == 0 for tag in physical_regions):
        handler.start_meshfunction("volume_region", dim, num_cells_counted)
        for i, physical_region in enumerate(physical_regions):
            handler.add_entity_meshfunction(i, physical_region)
        handler.end_meshfunction()

    # Now process the facet markers
    tags = tags_for_dim[highest_dim-1]
    if (len(tags) > 0) and (mesh is not None):
        physical_regions = tuple(tag[0] for tag in tags)
        if not all(tag == 0 for tag in physical_regions):
            mesh.init(highest_dim-1,0)

            # Get the facet-node connectivity information (reshape as a row of node indices per facet)
            if highest_dim==1:
              # for 1d meshes the mesh topology returns the vertex to vertex map, which isn't what we want
              # as facets are vertices
              facets_as_nodes = numpy.array([[i] for i in range(mesh.num_facets())])
            else:
              facets_as_nodes = numpy.array(mesh.topology()(highest_dim-1,0)()).reshape ( mesh.num_facets(), highest_dim )

            # Build the reverse map
            nodes_as_facets = {}
            for facet in range(mesh.num_facets()):
              nodes_as_facets[tuple(facets_as_nodes[facet,:])] = facet

            data = [int(0*k) for k in range(mesh.num_facets()) ]
            for i, physical_region in enumerate(physical_regions):
                nodes = [n-1 for n in vertices_used_for_dim[highest_dim-1][highest_dim*i:(highest_dim*i+highest_dim)]]
                nodes.sort()

                if physical_region != 0:
                    try:
                        index = nodes_as_facets[tuple(nodes)]
                        data[index] = physical_region
                    except IndexError:
                        raise Exception ( "The facet (%d) was not found to mark: %s" % (i, nodes) )

            # Create and initialise the mesh function
            handler.start_meshfunction("facet_region", highest_dim-1, mesh.num_facets() )
            for index, physical_region in enumerate ( data ):
                handler.add_entity_meshfunction(index, physical_region)
            handler.end_meshfunction()

    # Edge markers
    if highest_dim == 3:
        tags = tags_for_dim[1]
        if (len(tags) > 0) and (mesh is not None):
            physical_regions = tuple(tag[0] for tag in tags)
            if not all(tag == 0 for tag in physical_regions):
                mesh.init(1, 0)

                # Get the edge-node connectivity information (reshape as a row
                # of node indices per edge)
                edges_as_nodes = \
                    mesh.topology()(1, 0)().reshape(mesh.num_edges(), 2)

                # Build the reverse map
                nodes_as_edges = {}
                for edge in range(mesh.num_edges()):
                  nodes_as_edges[tuple(edges_as_nodes[edge])] = edge

                data = numpy.zeros(mesh.num_edges())
                for i, physical_region in enumerate(physical_regions):
                    nodes = [n-1 for n in vertices_used_for_dim[1][2*i:(2*i + 2)]]
                    nodes.sort()

                    if physical_region != 0:
                        try:
                            index = nodes_as_edges[tuple(nodes)]
                            data[index] = physical_region
                        except IndexError:
                            raise Exception ( "The edge (%d) was not found to mark: %s" % (i, nodes) )

                # Create and initialise the mesh function
                handler.start_meshfunction("curve_region", 1, mesh.num_edges())
                for index, physical_region in enumerate(data):
                    handler.add_entity_meshfunction(index, physical_region)
                handler.end_meshfunction()

    # Check that we got all data
    if state == 10:
        print("Conversion done")
    else:
       raise ValueError("Missing data, unable to convert \n\ Did you use version 2.0 of the gmsh file format?")

    # Close files
    ifile.close()


class ParseError(Exception):
    """ Error encountered in source file.
    """

class DataHandler(object):
    """ Baseclass for handlers of mesh data.

    The actual handling of mesh data encountered in the source file is
    delegated to a polymorfic object. Typically, the delegate will write the
    data to XML.
    @ivar _state: the state which the handler is in, one of State_*.
    @ivar _cell_type: cell type in mesh. One of CellType_*.
    @ivar _dim: mesh dimensions.
    """
    State_Invalid, State_Init, State_Vertices, State_Cells, \
          State_MeshFunction, State_MeshValueCollection = range(6)
    CellType_Tetrahedron, CellType_Triangle, CellType_Interval = range(3)

    def __init__(self):
        self._state = self.State_Invalid

    def set_mesh_type(self, cell_type, dim):
        assert self._state == self.State_Invalid
        self._state = self.State_Init
        if cell_type == "tetrahedron":
            self._cell_type = self.CellType_Tetrahedron
        elif cell_type == "triangle":
            self._cell_type = self.CellType_Triangle
        elif cell_type == "interval":
            self._cell_type = self.CellType_Interval
        self._dim = dim

    def start_vertices(self, num_vertices):
        assert self._state == self.State_Init
        self._state = self.State_Vertices

    def add_vertex(self, vertex, coords):
        assert self._state == self.State_Vertices

    def end_vertices(self):
        assert self._state == self.State_Vertices
        self._state = self.State_Init

    def start_cells(self, num_cells):
        assert self._state == self.State_Init
        self._state = self.State_Cells

    def add_cell(self, cell, nodes):
        assert self._state == self.State_Cells

    def end_cells(self):
        assert self._state == self.State_Cells
        self._state = self.State_Init

    def start_domains(self):
        assert self._state == self.State_Init

    def end_domains(self):
        self._state = self.State_Init

    def start_meshfunction(self, name, dim, size):
        assert self._state == self.State_Init
        self._state = self.State_MeshFunction

    def add_entity_meshfunction(self, index, value):
        assert self._state == self.State_MeshFunction

    def end_meshfunction(self):
        assert self._state == self.State_MeshFunction
        self._state = self.State_Init

    def start_mesh_value_collection(self, name, dim, size, etype):
        assert self._state == self.State_Init
        self._state = self.State_MeshValueCollection

    def add_entity_mesh_value_collection(self, dim, index, value, local_entity=0):
        assert self._state == self.State_MeshValueCollection

    def end_mesh_value_collection(self):
        assert self._state == self.State_MeshValueCollection
        self._state = self.State_Init

    def warn(self, msg):
        """ Issue warning during parse.
        """
        warnings.warn(msg)

    def error(self, msg):
        """ Raise error during parse.

        This method is expected to raise ParseError.
        """
        raise ParseError(msg)

    def close(self):
        self._state = self.State_Invalid

        
class XmlHandler(DataHandler):
    """ Data handler class which writes to Dolfin XML.
    """
    def __init__(self, ofilename):
        DataHandler.__init__(self)
        self._ofilename = ofilename
        self.__ofile = open(ofilename, "w")
        self.__ofile_meshfunc = None

    def ofile(self):
        return self.__ofile

    def set_mesh_type(self, cell_type, dim):
        DataHandler.set_mesh_type(self, cell_type, dim)
        xml_writer.write_header_mesh(self.__ofile, cell_type, dim)

    def start_vertices(self, num_vertices):
        DataHandler.start_vertices(self, num_vertices)
        xml_writer.write_header_vertices(self.__ofile, num_vertices)

    def add_vertex(self, vertex, coords):
        DataHandler.add_vertex(self, vertex, coords)
        xml_writer.write_vertex(self.__ofile, vertex, *coords)

    def end_vertices(self):
        DataHandler.end_vertices(self)
        xml_writer.write_footer_vertices(self.__ofile)

    def start_cells(self, num_cells):
        DataHandler.start_cells(self, num_cells)
        xml_writer.write_header_cells(self.__ofile, num_cells)

    def add_cell(self, cell, nodes):
        DataHandler.add_cell(self, cell, nodes)
        if self._cell_type == self.CellType_Tetrahedron:
            func = xml_writer.write_cell_tetrahedron
        elif self._cell_type == self.CellType_Triangle:
            func = xml_writer.write_cell_triangle
        elif self._cell_type == self.CellType_Interval:
            func = xml_writer.write_cell_interval

        func(self.__ofile, cell, *nodes)

    def end_cells(self):
        DataHandler.end_cells(self)
        xml_writer.write_footer_cells(self.__ofile)

    def start_meshfunction(self, name, dim, size):
        DataHandler.start_meshfunction(self, name, dim, size)
        fname = os.path.splitext(self.__ofile.name)[0]
        self.__ofile_meshfunc = open("%s_%s.xml" % (fname, name), "w")
        xml_writer.write_header_meshfunction(self.__ofile_meshfunc, dim, size)

    def add_entity_meshfunction(self, index, value):
        DataHandler.add_entity_meshfunction(self, index, value)
        xml_writer.write_entity_meshfunction(self.__ofile_meshfunc, index, value)

    def end_meshfunction(self):
        DataHandler.end_meshfunction(self)
        xml_writer.write_footer_meshfunction(self.__ofile_meshfunc)
        self.__ofile_meshfunc.close()
        self.__ofile_meshfunc = None

    def start_domains(self):
        #DataHandler.start_domains(self)
        xml_writer.write_header_domains(self.__ofile)

    def end_domains(self):
        #DataHandler.end_domains(self)
        xml_writer.write_footer_domains(self.__ofile)

    def start_mesh_value_collection(self, name, dim, size, etype):
        DataHandler.start_mesh_value_collection(self, name, dim, size, etype)
        xml_writer.write_header_meshvaluecollection(self.__ofile, name, dim, size, etype)

    def add_entity_mesh_value_collection(self, dim, index, value, local_entity=0):
        DataHandler.add_entity_mesh_value_collection(self, dim, index, value)
        xml_writer.write_entity_meshvaluecollection(self.__ofile, dim, index, value, local_entity=local_entity)

    def end_mesh_value_collection(self):
        DataHandler.end_mesh_value_collection(self)
        xml_writer.write_footer_meshvaluecollection(self.__ofile)

    def close(self):
        DataHandler.close(self)
        if self.__ofile.closed:
            return
        xml_writer.write_footer_mesh(self.__ofile)
        self.__ofile.close()
        if self.__ofile_meshfunc is not None:
            self.__ofile_meshfunc.close()


def convert2xml(ifilename, ofilename):
    """Convert a file to the DOLFIN XML format."""
    handler = XmlHandler(ofilename)
    gmsh2xml(ifilename, handler)
    handler.close()

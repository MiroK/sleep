# |--------------------|
# |                    |
# |    /\______        |
# |    \/              |
# |                    |
# |--------------------|


def build_model(model, geometry_parameters, mesh_sizes):
    '''GMSH model of Turek benchmark'''
    pass

# --------------------------------------------------------------------

if __name__ == '__main__':
    import sys

    if '-format' not in sys.argv:
        sys.argv.extend(['-format', 'msh2'])
    
    gmsh.initialize(sys.argv)

    model = gmsh.model


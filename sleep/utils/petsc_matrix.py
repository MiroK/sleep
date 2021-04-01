from contextlib import contextmanager
from petsc4py import PETSc
import dolfin as df
import numpy as np


def transpose_matrix(A):
    '''Create a transpose of PETScMatrix/PETSc.Mat'''
    if isinstance(A, PETSc.Mat):
        At = PETSc.Mat()  # Alloc
        A.transpose(At)  # Transpose to At
        return At

    At = transpose_matrix(df.as_backend_type(A).mat())
    return df.PETScMatrix(At)


@contextmanager
def petsc_serial_matrix(test_space, trial_space, nnz=None):
    '''
    PETsc.Mat from trial_space to test_space to be filled in the 
    with block. The spaces can be represented by intergers meaning 
    generic R^n.
    '''
    mesh = test_space.mesh()
    comm = mesh.mpi_comm()
    assert comm.size == 1
    
    row_map = test_space.dofmap()
    col_map = trial_space.dofmap()
    
    sizes = [[row_map.index_map().size(df.IndexMap.MapSize.OWNED),
              row_map.index_map().size(df.IndexMap.MapSize.GLOBAL)],
             [col_map.index_map().size(df.IndexMap.MapSize.OWNED),
              col_map.index_map().size(df.IndexMap.MapSize.GLOBAL)]]
    
    row_map = list(map(int, row_map.tabulate_local_to_global_dofs()))
    col_map = list(map(int, col_map.tabulate_local_to_global_dofs()))
        
    lgmap = lambda indices: (PETSc.LGMap().create(indices, comm=comm)
                             if isinstance(indices, list)
                             else
                             PETSc.LGMap().createIS(indices))
    
    row_lgmap, col_lgmap = list(map(lgmap, (row_map, col_map)))


    # Alloc
    mat = PETSc.Mat().createAIJ(sizes, nnz=nnz, comm=comm)
    mat.setUp()
    
    mat.setLGMap(row_lgmap, col_lgmap)

    mat.assemblyBegin()
    # Fill
    yield mat
    # Tear down
    mat.assemblyEnd()

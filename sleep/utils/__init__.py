from sleep.utils.embedded_mesh import EmbeddedMesh
from sleep.utils.petsc_matrix import transpose_matrix
from sleep.utils.trace_restriction import trace_matrix
from sleep.utils.subdomain_restriction import restriction_matrix
from functools import reduce


def preduce(comm, op, iterable):
    '''Reduce by op on procs of comm'''
    local = reduce(op, iterable)
    print(comm.rank, local, '<<<')
    all_local = comm.allgather(local)
    return reduce(op, all_local)

# --------------------------------------------------------------------

if __name__ == '__main__':
    from mpi4py import MPI as pyMPI
    import dolfin as df
    import operator
    
    mesh = df.UnitSquareMesh(32, 32)
    comm = pyMPI.COMM_WORLD


    print(preduce(comm, operator.or_, ({mesh.mpi_comm().rank},
                                       {mesh.mpi_comm().rank+1})))

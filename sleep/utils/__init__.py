from sleep.utils.embedded_mesh import EmbeddedMesh
from sleep.utils.petsc_matrix import transpose_matrix
from sleep.utils.trace_restriction import trace_matrix
from sleep.utils.subdomain_restriction import restriction_matrix
from functools import reduce


def preduce(comm, op, iterable):
    '''Reduce by op on procs of comm'''
    local = reduce(op, iterable)
    all_local = comm.allgather(local)
    return reduce(op, all_local)


KSP_CVRG_REASONS = {
    1: 'KSP_CONVERGED_RTOL_NORMAL',
    9: 'KSP_CONVERGED_ATOL_NORMAL',
    2: 'KSP_CONVERGED_RTOL' ,         
    3: 'KSP_CONVERGED_ATOL',         
    4: 'KSP_CONVERGED_ITS',                     
    5: 'KSP_CONVERGED_CG_NEG_CURVE',            
    6: 'KSP_CONVERGED_CG_CONSTRAINED',          
    7: 'KSP_CONVERGED_STEP_LENGTH',      
    8: 'KSP_CONVERGED_HAPPY_BREAKDOW',          
    -2: 'KSP_DIVERGED_NULL',                     
    -3: 'KSP_DIVERGED_ITS',                      
    -4: 'KSP_DIVERGED_DTOL',                      
    -5: 'KSP_DIVERGED_BREAKDOWN',                
    -6: 'KSP_DIVERGED_BREAKDOWN_BICG',          
    -7: 'KSP_DIVERGED_NONSYMMETRIC',             
    -8: 'KSP_DIVERGED_INDEFINITE_PC',            
    -9: 'KSP_DIVERGED_NANORINF',                 
    -10: 'KSP_DIVERGED_INDEFINITE_MA',             
    -11: 'KSP_DIVERGED_PC_FAILED'
}

# --------------------------------------------------------------------

if __name__ == '__main__':
    from mpi4py import MPI as pyMPI
    import dolfin as df
    import operator
    
    mesh = df.UnitSquareMesh(32, 32)
    comm = pyMPI.COMM_WORLD


    print(preduce(comm, operator.or_, ({mesh.mpi_comm().rank},
                                       {mesh.mpi_comm().rank+1})))

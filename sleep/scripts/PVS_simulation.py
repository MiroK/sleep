#! /usr/bin/env python3
import argparse
import logging
from datetime import datetime
import os

import numpy as np

from sleep.fbb_DD.fluid import solve_fluid
from sleep.fbb_DD.ale import solve_ale
from sleep.utils import EmbeddedMesh
from sleep.mesh import load_mesh2d
from dolfin import *

#logging.debug
#logging.info
#logging.warning
#logging.error
#logging.critical


#todo : logging initialisation + set for tracer
#todo: add parameter for initialisationt tracer
#todo : add ALE step
#todo : logging for simulaiton run
#todo : add resistance BC
#todo : write 1D profiles for u p c in a file
#todo : return the 1D profiles

#todo : external script to postprocess u p c profiles : figure , compute coef dispertion

def title1(string):
    line1='\n *'*100+'\n'
    line2='**   '+string+'\n'
    line3='*'*100+'\n'
    return line1 + line2 + line3



def PVS_simulation(args):
    """ Solve the flow and tracer transport in the PVS :
    Outputs :
    - a logfile with information about the simulation
    - .pvd files with the u, p and c field at specified args.toutput time period
    
    Return : u, p, c 1D array of the u, p, c fields on the middle line """



    # output folder name
    outputfolder=args.output_folder+'/'+args.job_name+'/'

    if not os.path.exists(outputfolder):
        os.makedirs(outputfolder)

    if not os.path.exists(outputfolder+'/log'):
        os.makedirs(outputfolder+'/log')

    # Create output files
    uf_out, pf_out= File(outputfolder+'/uf.pvd'), File(outputfolder+'/pf.pvd')
    facets_out=File(outputfolder+'/facets.pvd')

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # log to a file
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(outputfolder+'/log/', 'PVS_%s.log' % now)
    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(logging.INFO)
    #formatter = logging.Formatter("%(asctime)s %(filename)s, %(lineno)d, %(funcName)s: %(message)s")
    #file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # log to the console
    console_handler = logging.StreamHandler()
    level = logging.INFO
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    

    # initialise logging

    logging.info(title1("Simulation of the PVS flow and tracer transport using non steady solver and diffusion-advection solvers"))

    logging.info("Date and time:"+datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))

    logging.info('Job name : '+args.job_name)

    logging.debug('logging initialized')


    # Set parameters

    logging.info(title1("Parameters"))

    # Geometry params
    logging.info('\n * Geometry')
    Rv = args.radius_vessel # centimeters
    Rpvs =args.radius_pvs # centimeters
    L = args.length # centimeters

    logging.info('Vessel radius : %e cm'%Rv)
    logging.info('PVS radius : %e cm'%Rpvs)
    logging.info('PVS length : %e cm'%L)

    #Mesh
    logging.info('\n * Mesh')
    #number of cells in the radial direction
    Nr=args.N_radial
    DR=(Rpvs-Rv)/Nr
    #number of cells in the axial direction
    if args.N_axial :
        Nl=args.N_axial
    else :
        Nl=round(L/DR)

    DY=L/Nl
    
    logging.info('N axial : %i'%Nl)
    logging.info('N radial : %e'%Nr)
    logging.info('cell size : %e'%(np.sqrt(DR**2+DY**2)))
    logging.info('nb cells: %i'%(Nl*Nr*2))
    

    #time parameters
    logging.info('\n * Time')
    toutput=args.toutput
    tfinal=args.tend
    dt=args.time_step

    logging.info('final time: %e s'%tfinal)
    logging.info('output period : %e s'%toutput)
    logging.info('time step : %e s'%dt)

    # approximate CFL
    Uapprox=500e-4 #upper limit for extected max velocity
    CFL_dt=0.25*DY/Uapprox
    if  CFL_dt < dt :
        logging.warning('The specified time step of %.2e s does not fullfil the CFL condition. New time step : %.2e s'%(dt, CFL_dt))
    dt=min(dt,CFL_dt)


    # material parameters
    logging.info('\n * Fluid properties')
    mu=args.viscosity
    rho=args.density

    logging.info('density: %e g/cm3'%rho)
    logging.info('dynamic viscosity : %e dyn s/cm2'%mu)

    logging.info('\n* Tracer properties')
    D=args.diffusion_coef

    logging.info('Free diffusion coef: %e cm2/s'%D)

    logging.info('\n * ALE')
    kappa=args.ale_parameter
    logging.info('ALE parameter: %e '%kappa)


    #FEM space
    Vf_elm = VectorElement('Lagrange', triangle, 2)
    Qf_elm = FiniteElement('Lagrange', triangle, 1)

    # Mesh

    mesh_f = RectangleMesh(Point(0, Rv), Point(L, Rpvs), Nl, Nr)

    fluid_bdries = MeshFunction("size_t", mesh_f, mesh_f.topology().dim()-1,0)

    # Label facets
    xy = mesh_f.coordinates().copy()              
    x, y = xy.T

    xmin=x.min()
    xmax=x.max()
    ymin=y.min()
    ymax=y.max()


    tol=min(DR,DY)/2 #cm

    class Boundary_left(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0],xmin,tol) #left


    class Boundary_right(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0],xmax,tol)# right


    class Boundary_bottom(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], ymin,tol) #bottom
        
    class Boundary_top(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], ymax,tol) #top



    btop= Boundary_top()
    bbottom = Boundary_bottom()
    bleft = Boundary_left()
    bright = Boundary_right()


    bbottom.mark(fluid_bdries,  2) 
    btop.mark(fluid_bdries,  4)  
    bleft.mark(fluid_bdries, 1) 
    bright.mark(fluid_bdries,  3) 

    

    facet_lookup = {'x_min': 1 ,'y_min': 2, 'x_max': 3, 'y_max': 4}

    facets_out << fluid_bdries

    # Parameters setup ------------------------------------------------ 

    fluid_parameters = {'mu': mu, 'rho': rho, 'dt':dt}
    ale_parameters = {'kappa': kappa}

    # Setup fem spaces ---------------------------------------------------
    Wf_elm = MixedElement([Vf_elm, Qf_elm])
    Wf = FunctionSpace(mesh_f, Wf_elm)


    # Setup of boundary conditions ----------------------------------- 
    logging.info(title1("Boundary conditions"))
    import sympy
    ts = sympy.symbols("time")
    sin = sympy.sin

    amp=1e-4 #cm
    f=1 #Hz

    logging.info('\n * wall motion parameters')
    ai=args.ai
    fi=args.fi
    phii=args.phii
    logging.info('ai (dimensioless): '+'*e'*len(ai)%ai)
    logging.info('fi (Hz) : '+'*e'*len(fi)%fi)
    logging.info('phii (rad) : '+'*e'*len(phii)%phii)

    functionU = Rv*sum([a*sin(2*pi*f*ts+phi) for a,f,phi in zip(ai,fi,phii)]) # displacement
    U_vessel = sympy.printing.ccode(functionU)

    functionV = sympy.diff(functionU,ts) # velocity
    V_vessel = sympy.printing.ccode(functionV)

   
    logging.info('\n * Fluid')
    logging.info('Left : zero pressure')
    logging.info('Right : resistance')
    logging.info('Top : no slip no gap fixed wall')
    logging.info('Bottom : no slip no gap moving wall')

    logging.info('\n * Tracer concentration')
    logging.info('Left : zero')
    logging.info('Right : no flux')
    logging.info('Top : no flux')
    logging.info('Bottom : no flux')

    # Now we wire up
    bcs_fluid = {'velocity': [(facet_lookup['y_min'], Constant((0,0))),
                            (facet_lookup['y_max'], Constant((0,0)))],
                'traction': [],  
                'pressure': [(facet_lookup['x_min'], Constant(0)),
                            (facet_lookup['x_max'], Constant(0))]}







    # Initialisation : 2 possibilities
    # 1/ Initialise with zero fields
    uf_n = project(Constant((0, 0)), Wf.sub(0).collapse())
    pf_n =  project(Constant(0), Wf.sub(1).collapse())




    # Save initial state
    uf_n.rename("uf", "tmp")
    pf_n.rename("pf", "tmp")

    uf_out << (uf_n, 0)
    pf_out << (pf_n, 0)

    # Time loop
    time = 0.
    timestep=0

    erroru=[]
    errorp=[]
    while time < tfinal:
        time += fluid_parameters['dt']
        timestep+=1

        # Solve fluid problem
        uf_, pf_ = solve_fluid(Wf, f=Constant((0, 0)), bdries=fluid_bdries, bcs=bcs_fluid,
                            parameters=fluid_parameters)

        # Update current solution
        uf_n.assign(uf_)
        pf_n.assign(pf_)

        
        # Save output
        if(timestep % int(toutput/dt) == 0):

            uf_.rename("uf", "tmp")
            pf_.rename("pf", "tmp")

            uf_out << (uf_, time)
            pf_out << (pf_, time)

    #with open('./output/'+outputfolder+'/erroru.txt', 'a') as csv:
    #    row=[N]+erroru
    #    csv.write(('%i'+', %e'*len(erroru)+'\n')%tuple(row))



    #Necessary ?
    #   for i in logging._handlers.copy():
    #       log.removeHandler(i)
    #       i.flush()
    #       i.close()


if __name__ == '__main__':
    # Create the parser
    my_parser = argparse.ArgumentParser(description='Launch a simulation of PVS flow')

    # Add the arguments

    my_parser.add_argument('-j','--job_name',
                        type=str,
                        default="PVS",
                        help='Name of the job')

    my_parser.add_argument('-o','--output_folder',
                            type=str,
                            default="./output",
                            help='Folder where the results are stored')

    my_parser.add_argument('-rv','--radius_vessel',
                        type=float,
                        default=8e-4,
                        help='Vessel radius as rest')

    my_parser.add_argument('-rpvs','--radius_pvs',
                        metavar='Rpvs',
                        type=float,
                        default=10e-4,
                        help='PVS outer radius as rest')

    my_parser.add_argument('-l','--length',
                        type=float,
                        default=100e-4,
                        help='Length of the vessel')

    my_parser.add_argument('-ai',
                        type=float,
                        nargs='+',
                        default=[0.01],
                        help='List of ai')

    my_parser.add_argument('-fi',
                        type=float,
                        nargs='+',
                        default=[10],
                        help='List of fi')


    my_parser.add_argument('-phii',
                        type=float,
                        nargs='+',
                        default=[0],
                        help='List of phii')

    my_parser.add_argument('-tend',
                        type=float,
                        default=1,
                        help='final time')

    my_parser.add_argument('-toutput',
                        type=float,
                        default=0.01,
                        help='output period')                       

    my_parser.add_argument('-dt','--time_step',
                        type=float,
                        default=1e-4,
                        help='time step')

    my_parser.add_argument('-mu','--viscosity',
                        type=float,
                        default=7e-3,
                        help='Dynamic viscosity of the fluid')

    my_parser.add_argument('-rho','--density',
                        type=float,
                        default=1,
                        help='Density of the fluid')

    my_parser.add_argument('-r','--resistance',
                        type=float,
                        default=0,
                        help='Resistance at the inner side of the brain')

    my_parser.add_argument('-k','--ale_parameter',
                        type=float,
                        default=1,
                        help='ALE parameter')

    my_parser.add_argument('-nr','--N_radial',
                        type=int,
                        default=8,
                        help='number of cells in the radial direction')

    my_parser.add_argument('-nl','--N_axial',
                        type=int,
                        default=0,
                        help='number of cells in the axial direction')

    my_parser.add_argument('-d','--diffusion_coef',
                        type=float,
                        default=2e-8,
                        help='Diffusion coefficient of the tracer')

    args = my_parser.parse_args()

    # Execute the PVS simulation

    PVS_simulation(args)



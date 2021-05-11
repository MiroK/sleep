#! /usr/bin/env python3
import argparse
import logging
from datetime import datetime
import os

import numpy as np
from math import pi

from sleep.fbb_DD.domain_transfer import transfer_into
from sleep.fbb_DD.advection import solve_adv_diff as solve_adv_diff
#from sleep.fbb_DD.fluid import solve_fluid as solve_fluid
from sleep.fbb_DD.fluid import solve_fluid
#import sleep.fbb_DD.cylindrical as cyl
#todo : cylindrical for solid
from sleep.fbb_DD.solid2 import solve_solid
from sleep.fbb_DD.ale import solve_ale as solve_ale
from sleep.utils import EmbeddedMesh
from sleep.mesh import load_mesh2d
from dolfin import *

import gmsh


def title1(string):
    line1='\n'+'*'*100+'\n'
    line2='**   '+string+'\n'
    line3='*'*100
    return line1 + line2 + line3


def PVSbrain_simulation(args):
    """ Test case for simple diffusion from PVS to the brain. 
    Outputs :
    - a logfile with information about the simulation
    - .pvd files at specified args.toutput time period with the u, p and c fields in stokes domain and u, p , q, phi and c fields in Biot domain
    - .csv files of u, p, c 1D array of the u, p, c fields on the middle line of the PVS 
    - .csv files of the total mass in the domain, mass flux from the brain to sas, brain to PVS, PVS to SAS

    """



    # output folder name
    outputfolder=args.output_folder+'/'+args.job_name+'/'

    if not os.path.exists(outputfolder):
        os.makedirs(outputfolder)


    if not os.path.exists(outputfolder+'/fields'):
        os.makedirs(outputfolder+'/fields')

    # Create output files
    #pvd files
    c_out=File(outputfolder+'fields'+'/c.pvd')
   
    facets_out_fluid=File(outputfolder+'fields'+'/facets_fluid.pvd')
    facets_out_solid=File(outputfolder+'fields'+'/facets_solid.pvd')

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # log to a file
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(outputfolder+'/', 'PVSBrain_info.log')
    file_handler = logging.FileHandler(filename,mode='w')
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

    logging.info(title1("Test case of simple diffusion from PVS to the brain using the diffusion-advection solver"))

    logging.info("Date and time:"+datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))

    logging.info('Job name : '+args.job_name)

    logging.debug('logging initialized')


    # Set parameters

    logging.info(title1("Parameters"))

    # Geometry params
    logging.info('\n * Geometry')
    Rv = args.radius_vessel # centimeters
    Rpvs =args.radius_pvs # centimeters
    Rbrain =args.radius_brain # centimeters
    L = args.length # centimeters


    logging.info('Vessel radius : %e cm'%Rv)
    logging.info('PVS radius : %e cm'%Rpvs)
    logging.info('Brain radius : %e cm'%Rbrain)
    logging.info('length : %e cm'%L)

    #Mesh
    logging.info('\n * Mesh')
    #number of cells in the radial direction fluid domain
    Nr=args.N_radial_fluid
    #number of cells in the radial direction biot domain
    Nr_biot=args.N_radial_biot
    s_biot=args.biot_progression

    DR=(Rpvs-Rv)/Nr
    #number of cells in the axial direction
    if args.N_axial :
        Nl=args.N_axial
    else :
        Nl=round(L/DR)

    DY=L/Nl
    
    logging.info('N axial: %i'%Nl)
    logging.info('N radial PVS: %e'%Nr)
    logging.info('N radial Biot: %e'%Nr_biot)
    logging.info('progression parameter in biot: %e'%s_biot)

    
    #time parameters
    logging.info('\n * Time')
    toutput=args.toutput
    tfinal=args.tend
    dt=args.time_step

    logging.info('final time: %e s'%tfinal)
    logging.info('output period : %e s'%toutput)
    logging.info('time step : %e s'%dt)


    dt_advdiff=dt



    logging.info('\n* Tracer properties')
    D=args.diffusion_coef
    sigma_gauss=args.sigma
    logging.info('Free diffusion coef: %e cm2/s'%D)
    logging.info('STD of initial gaussian profile: %e '%sigma_gauss)
    xi_gauss=args.initial_pos
    logging.info('Initial position: %e cm2'%xi_gauss)



    logging.info('\n * Porous medium properties')


    porosity_0=args.biot_porosity
    tortuosity=args.biot_tortuosity


    dt_solid=dt


    logging.info('initial porosity: %e '%porosity_0)
    logging.info('tortuosity: %e '%tortuosity)



    ## The tracer is solver in the full domain, so it has two subdomains
    # 1 for solid
    # 0 for fluid

    tracer_parameters={'kappa_0': D, 'kappa_1': D*tortuosity,'dt':dt_advdiff, 'nsteps':1}



    # Mesh
    logging.info(title1('Meshing'))

    meshing=args.mesh_method

    ##### Meshing method : regular

    if meshing=='regular' :
        # Create a mesh using Rectangle mesh : all the cells are regulars but this means a lot of cells

        logging.info('cell size : %e cm'%(np.sqrt(DR**2+DY**2)))

        # Create a rectangle mesh with Nr + Nsolid cells in the radias direction and Nl cells in the axial direction
        # the geometrical progression for the solid mesh is s
        

        #Creation of the uniform mesh
        Rext=1+Nr_biot/Nr
        mesh= RectangleMesh(Point(0,0), Point(L, Rext), Nl, Nr+Nr_biot)
        x = mesh.coordinates()[:,0]
        y = mesh.coordinates()[:,1]

        #Deformation of the mesh

        def deform_mesh(x, y):
            transform_fluid=Rv + (Rpvs-Rv)*y
            transform_solid=Rpvs + (Rbrain-Rpvs)*((y-1)/(Rext-1))**s_biot
            yp=np.where(y <= 1, transform_fluid, transform_solid)
            return [x, yp]

        x_bar, y_bar = deform_mesh(x, y)
        xy_bar_coor = np.array([x_bar, y_bar]).transpose()
        mesh.coordinates()[:] = xy_bar_coor
        mesh.bounding_box_tree().build(mesh)

    else :

    ##### Meshing method : gmsh with box for refinement
        
        from sleep.mesh import mesh_model2d, load_mesh2d, set_mesh_size
        import sys


        gmsh.initialize(['','-format', 'msh2'])


        model = gmsh.model

        import math
        Apvs0=math.pi*Rpvs**2
        Av0=math.pi*Rv**2
        A0=Apvs0-Av0


        # progressive mesh 
        factory = model.occ
        a = factory.addPoint(0, Rv,0)
        b = factory.addPoint(L,Rv,0)
        c = factory.addPoint(L,Rpvs,0)
        d = factory.addPoint(0,Rpvs,0)
        e = factory.addPoint(L,Rbrain,0)
        f = factory.addPoint(0,Rbrain,0)


        fluid_lines = [factory.addLine(*p) for p in ((a,b), (b, c), (c,d), (d,a))]
        named_lines = dict(zip(('bottom', 'pvs_right', 'interface', 'pvs_left'), fluid_lines))

        fluid_loop = factory.addCurveLoop(fluid_lines)
        fluid = factory.addPlaneSurface([fluid_loop])

        solid_lines = [factory.addLine(*p) for p in ((d,c), (c, e), (e,f), (f,d))]
        named_lines.update(dict(zip(('interface', 'brain_right', 'brain_top', 'brain_left'), solid_lines)))

        solid_loop = factory.addCurveLoop(solid_lines)
        solid = factory.addPlaneSurface([solid_loop])


        factory.synchronize()

        tags = {'cell': {'F': 1, 'S': 2},
            'facet': {}}

        model.addPhysicalGroup(2, [fluid], 1)
        model.addPhysicalGroup(2, [solid], 2)

        for name in named_lines:
            tag = named_lines[name]
            model.addPhysicalGroup(1, [tag], tag)

        # boxes for mesh refinement
        cell_size=DR*(Rpvs-Rv)/(Rpvs-Rv)
        boxes = []
        # add box on the PVS for mesh
        field = model.mesh.field


        fid = 1
        field.add('Box', fid)
        field.setNumber(fid, 'XMin', 0)
        field.setNumber(fid, 'XMax', L)
        field.setNumber(fid, 'YMin', Rv)
        field.setNumber(fid, 'YMax', Rpvs)
        field.setNumber(fid, 'VIn', cell_size*2)
        field.setNumber(fid, 'VOut', DR*50 )
        field.setNumber(fid, 'Thickness', (Rpvs-Rv)/4)

        boxes.append(fid)

        
        # Combine
        field.add('Min', fid+1)
        field.setNumbers(fid+1, 'FieldsList', boxes)    
        field.setAsBackgroundMesh(fid+1)

        model.occ.synchronize()

        h5_filename = outputfolder+'/mesh.h5'

        mesh_model2d(model, tags, h5_filename)

        mesh, markers, lookup = load_mesh2d(h5_filename)
        
        gmsh.finalize()

    ## Define subdomains

    x = mesh.coordinates()[:,0]
    y = mesh.coordinates()[:,1]


    tol=1e-7


    class Omega_0(SubDomain):
        def inside(self, x, on_boundary):
            return x[1] < Rpvs + tol

    class Omega_1(SubDomain):
        def inside(self, x, on_boundary):
            return x[1] > Rpvs - tol

    subdomains = MeshFunction("size_t", mesh, mesh.topology().dim(),0)

    subdomain_0 = Omega_0()
    subdomain_1 = Omega_1()
    subdomain_0.mark(subdomains, 0)
    subdomain_1.mark(subdomains, 1)


    mesh_f = EmbeddedMesh(subdomains, 0)
    mesh_s = EmbeddedMesh(subdomains, 1)



    ## Define boundaries
    solid_bdries = MeshFunction("size_t", mesh_s, mesh_s.topology().dim()-1,0)
    fluid_bdries = MeshFunction("size_t", mesh_f, mesh_f.topology().dim()-1,0)
    full_bdries= MeshFunction("size_t", mesh, mesh.topology().dim()-1,0)

    # Label facets
  

    class Boundary_left_fluid(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0],0,tol) and (x[1] <= Rpvs + tol) #left fluid


    class Boundary_right_fluid(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0],L,tol) and (x[1] <= Rpvs + tol) # right fluid


    class Boundary_left_solid(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0],0,tol) and (x[1] >= Rpvs - tol) #left solid


    class Boundary_right_solid(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0],L,tol) and (x[1] >= Rpvs - tol) # right solid


    class Boundary_bottom(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], Rv,tol) #bottom
        
    class Boundary_top(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], Rbrain,tol) #top

    class Boundary_interface(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], Rpvs,tol) #interface      


    #todo : keep separate F and S for right and left in the full bdries

    btop= Boundary_top()
    bbottom = Boundary_bottom()
    bleft_fluid= Boundary_left_fluid()
    bright_fluid = Boundary_right_fluid()
    bleft_solid = Boundary_left_solid()
    bright_solid = Boundary_right_solid()
    binterface = Boundary_interface()

    bbottom.mark(fluid_bdries,  2) 
    binterface.mark(fluid_bdries,  4)  
    bleft_fluid.mark(fluid_bdries, 1) 
    bright_fluid.mark(fluid_bdries,  3) 

    binterface.mark(solid_bdries,  4) 
    bright_solid.mark(solid_bdries,  5) 
    btop.mark(solid_bdries,  6) 
    bleft_solid.mark(solid_bdries,  7) 

    bleft_fluid.mark(full_bdries, 1) 
    bleft_solid.mark(full_bdries,  7)  
    bbottom.mark(full_bdries,  2) 
    bright_fluid.mark(full_bdries,  3) 
    bright_solid.mark(full_bdries,  5) 
    btop.mark(full_bdries,  6) 

    facet_lookup = {'F_left': 1 ,'F_bottom': 2, 'F_right': 3, 'Interface': 4,'S_left': 7 ,'S_top': 6, 'S_right': 5}

    facets_out_fluid << fluid_bdries

    facets_out_solid << solid_bdries


    # define the domain specific parameters for the tracer
    # NOTE: Here we do P0 projection
    dx = Measure('dx', domain=mesh, subdomain_data=subdomains)
    CoefSpace = FunctionSpace(mesh, 'DG', 0)
    q = TestFunction(CoefSpace)

    # Remove
    coef='kappa'

    fluid_coef = tracer_parameters.pop('%s_0' % coef)
    solid_coef = tracer_parameters.pop('%s_1' % coef)

    form = ((1/CellVolume(mesh))*fluid_coef*q*dx(0) +
                (1/CellVolume(mesh))*solid_coef*q*dx(1))
        
    tracer_parameters[coef] = Function(CoefSpace, assemble(form))


    #FEM space

    logging.info(title1("Set FEM spaces"))

    logging.info('\n * Tracer')
    #### Todo : I would like to be able to have discontinuous concentration when we will have the membrane
    #### Beter to solve in two domains or one domain with discontinuous lagrange element ?
    Ct_elm = FiniteElement('Lagrange', triangle, 1)
    Ct = FunctionSpace(mesh, Ct_elm)
    logging.info('Concentration : "Lagrange", triangle, 1')


    #Advection velocity
    FS_advvel= VectorFunctionSpace(mesh, 'CG', 2)


    # Setup of boundary conditions
    logging.info(title1("Boundary conditions"))

    ## to do : allow zero concentration if fluid BC is free on the right
    logging.info('\n * Tracer concentration')   

    bcs_tracer = {'concentration': [(facet_lookup['S_top'], Constant(0))],
                    'flux': [(facet_lookup['S_left'], Constant(0)),
                            (facet_lookup['S_right'], Constant(0)),
                            (facet_lookup['F_right'], Constant(0)),
                            (facet_lookup['F_left'], Constant(0)),
                            (facet_lookup['F_bottom'], Constant(0)),
                            ]}



    # Initialisation : 

    # 1 in the PVS
    cf_0 = Expression('x[1]<= Rpvs ? 1 : 0 ', degree=2, a=1/2/sigma_gauss**2, b=xi_gauss, Rv=Rv, Rpvs=Rpvs)

    c_n =  project(cf_0,Ct)#

    #Initial deformation of the fluid domain
    # We start at a time shift
    tshift=0 # 

    c_n.rename("c", "tmp")
    c_out << (c_n, 0)


    ############# RUN ############3

    logging.info(title1("Run"))

    # Time loop
    time = tshift
    timestep=0





    # Here I dont know if there will be several dt for advdiff and fluid solver
    while time < tfinal+tshift:

        time+=dt
        timestep+=1
        print('time', time-tshift)



        # Solve tracer problem

        tracer_parameters["T0"]=time

        advection_velocity=project(Constant((0,0)),FS_advvel)

        c_, T0= solve_adv_diff(Ct, velocity=advection_velocity, phi=Constant(0.2), f=Constant(0), c_0=c_n, phi_0=Constant(0.2),
                                  bdries=full_bdries, bcs=bcs_tracer, parameters=tracer_parameters)


        # Update current solution

        c_n.assign(c_)


        # Save output
        if(timestep % int(toutput/dt) == 0):

            logging.info("\n*** save output time %e s"%(time-tshift))
            logging.info("number of time steps %i"%timestep)

            # may report Courant number or other important values that indicate how is doing the run

            c_n.rename("c", "tmp")
            c_out << (c_n, time-tshift)

            advection_velocity.rename("adv_vel", "tmp")
            File(outputfolder+'fields'+'/adv_vel.pvd') << (advection_velocity, time-tshift)




if __name__ == '__main__':
    # Create the parser
    my_parser = argparse.ArgumentParser(description='Launch a simulation of PVS flow')

    # Add the arguments

    my_parser.add_argument('-j','--job_name',
                        type=str,
                        default="PVSandBrain",
                        help='Name of the job')

    my_parser.add_argument('-o','--output_folder',
                            type=str,
                            default="../output/",
                            help='Folder where the results are stored')

    my_parser.add_argument('-rv','--radius_vessel',
                        type=float,
                        default=8e-4,
                        help='Vessel radius as rest')

    my_parser.add_argument('-rpvs','--radius_pvs',
                        metavar='Rpvs',
                        type=float,
                        default=11e-4,
                        help='PVS outer radius as rest')

    my_parser.add_argument('-rbrain','--radius_brain',
                        type=float,
                        default=100e-4,
                        help='Outer radius of the brain domain')

    my_parser.add_argument('-l','--length',
                        type=float,
                        default=200e-4,
                        help='Length of the vessel')

    my_parser.add_argument('-ai',
                        type=float,
                        nargs='+',
                        default=[0.1],
                        help='List of ai')

    my_parser.add_argument('-fi',
                        type=float,
                        nargs='+',
                        default=[1],
                        help='List of fi')


    my_parser.add_argument('-phii',
                        type=float,
                        nargs='+',
                        default=[0],
                        help='List of phii')

    my_parser.add_argument('-tend',
                        type=float,
                        default=5,
                        help='final time')

    my_parser.add_argument('-toutput',
                        type=float,
                        default=5e-2,
                        help='output period')                       

    my_parser.add_argument('-dt','--time_step',
                        type=float,
                        default=1e-2,
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
                        default=-1,
                        help='Resistance at the inner side of the brain')

    my_parser.add_argument('-k','--ale_parameter',
                        type=float,
                        default=1,
                        help='ALE parameter')

    my_parser.add_argument('-nrpvs','--N_radial_fluid',
                        type=int,
                        default=8,
                        help='number of cells in the radial direction in the PVS')

    my_parser.add_argument('-nrbiot','--N_radial_biot',
                        type=int,
                        default=20,
                        help='number of cells in the radial direction in the biot domain')

    my_parser.add_argument('-s_biot','--biot_progression',
                        type=float,
                        default=2.5,
                        help='progression coefficient for the mesh in the biot domain')

    my_parser.add_argument('-nl','--N_axial',
                        type=int,
                        default=100,
                        help='number of cells in the axial direction')

    my_parser.add_argument('-d','--diffusion_coef',
                        type=float,
                        default=2e-8,
                        help='Molecular diffusion coefficient of the tracer')

    my_parser.add_argument('-s','--sigma',
                        type=float,
                        default=4e-4,
                        help='STD gaussian init for concentration')

    my_parser.add_argument('-xi','--initial_pos',
                        type=float,
                        default=100e-4,
                        help='Initial center of the gaussian for concentration')

    my_parser.add_argument('-perm','--biot_permeability',
                        type=float,
                        default=3e-13,
                        help='Permeability in the biot domain (cm2)')

    my_parser.add_argument('-E','--biot_youngmodulus',
                        type=float,
                        default=5000e3,
                        help='Young modulus in the Biot domain (dyn/cm2)')

    my_parser.add_argument('-nu','--biot_poisson',
                        type=float,
                        default=0.45,
                        help='Poisson coefficient in the Biot domain')

    my_parser.add_argument('-alpha','--biot_biotcoefficient',
                        type=float,
                        default=1,
                        help='Biot coefficient')

    my_parser.add_argument('-tau','--biot_tortuosity',
                        type=float,
                        default=1,
                        help='Tortuosity in the Biot domain')

    my_parser.add_argument('-phi','--biot_porosity',
                        type=float,
                        default=0.2,
                        help='Porosity in the Biot domain')     

    my_parser.add_argument('-mesh','--mesh_method',
                        type=str,
                        default='regular',
                        help='Method for the mesh construction')   
    args = my_parser.parse_args()


    # Execute the PVS simulation

    PVSbrain_simulation(args)



#! /usr/bin/env python3
import argparse
import logging
from datetime import datetime
import os

import numpy as np
from math import pi,exp

from sleep.fbb_DD.advection import solve_adv_diff_cyl as solve_adv_diff
from sleep.fbb_DD.fluid import solve_fluid_cyl as solve_fluid

from sleep.utils import EmbeddedMesh
from sleep.mesh import load_mesh2d
from dolfin import *


#todo : add time shift like PVS SAS
#add production , absorption


# Define line function for 1D slice evaluation
def line_sample(line, f, fill=np.nan):
    values = fill*np.ones(len(line))
    for i, xi in enumerate(line):
        try:
            values[i] = f(xi)
        except RuntimeError:
            continue
    return values

def slice_integrate_cyl(x,f,ymin,ymax,N=10) :
    vertical_line=line([x,ymin],[x,ymax], N)
    values=line_sample(vertical_line, f)
    r=np.linspace(ymin,ymax,N)
    integral=2*np.trapz(values*r,r)/(ymax**2-ymin**2) #cylindrical coordinate
    return integral

def profile_cyl(f,xmin,xmax,ymin,ymax,Nx=100,Ny=10):
    spanx=np.linspace(xmin,xmax,Nx)
    values=[slice_integrate_cyl(x,f,ymin,ymax,N=Ny) for x in spanx]
    return np.array(values)



def line(A, B, nsteps):
    A=np.array(A)
    B=np.array(B)
    return A + (B-A)*np.linspace(0, 1, nsteps).reshape((-1, 1))


def title1(string):
    line1='\n'+'*'*100+'\n'
    line2='**   '+string+'\n'
    line3='*'*100
    return line1 + line2 + line3



def SAS_simulation(args):
    """ Solve the flow and tracer transport in the PVS :
    Outputs :
    - a logfile with information about the simulation
    - .pvd files with the u, p and c field at specified args.toutput time period
    
    Return : u, p, c 1D array of the u, p, c fields on the middle line """



    # output folder name
    outputfolder=args.output_folder+'/'+args.job_name+'/'

    if not os.path.exists(outputfolder):
        os.makedirs(outputfolder)


    if not os.path.exists(outputfolder+'/profiles'):
        os.makedirs(outputfolder+'/profiles')

    if not os.path.exists(outputfolder+'/fields'):
        os.makedirs(outputfolder+'/fields')

    # Create output files

    #txt files
    csv_p=open(outputfolder+'profiles'+'/pressure.txt', 'w')
    csv_u=open(outputfolder+'profiles'+'/velocity.txt', 'w')
    csv_c=open(outputfolder+'profiles'+'/concentration.txt', 'w')
    csv_rv=open(outputfolder+'profiles'+'/radius.txt', 'w')

    #pvd files
    uf_out, pf_out= File(outputfolder+'fields'+'/uf.pvd'), File(outputfolder+'fields'+'/pf.pvd')
    c_out= File(outputfolder+'fields'+'/c.pvd')
    facets_out=File(outputfolder+'fields'+'/facets.pvd')

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # log to a file
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(outputfolder+'/', 'PVS_info.log')
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

    logging.info(title1("Simulation of the PVS flow and tracer transport using non steady solver and diffusion-advection solvers"))

    logging.info("Date and time:"+datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))

    logging.info('Job name : '+args.job_name)

    logging.debug('logging initialized')


    # Set parameters

    logging.info(title1("Parameters"))

    # Geometry params
    logging.info('\n * Geometry')
    Rv = args.radius_inner # centimeters
    Rpvs =args.radius_outer # centimeters
    L = args.length # centimeters

    logging.info('Inner radius : %e cm'%Rv)
    logging.info('Outer radius : %e cm'%Rpvs)
    logging.info('Length : %e cm'%L)

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

    

    #time parameters
    logging.info('\n * Time')
    toutput=args.toutput
    tfinal=args.tend
    dt=args.time_step

    logging.info('final time: %e s'%tfinal)
    logging.info('output period : %e s'%toutput)
    logging.info('time step : %e s'%dt)

    # approximate CFL for fluid solver : need to compute max velocity depending on the wall displacement... 
    # maybe just add a warning in computation with actual velocity
    #Uapprox=500e-4 #upper limit for extected max velocity
    #CFL_dt=0.25*DY/Uapprox
    #if  CFL_dt < dt :
    #    logging.warning('The specified time step of %.2e s does not fullfil the fluid CFL condition. New fluid time step : %.2e s'%(dt, CFL_dt))
    #dt_fluid=min(dt,CFL_dt)
    dt_fluid=dt

    # approximate CFL for tracer solver
    dt_advdiff=dt


    # material parameters
    logging.info('\n * Fluid properties')
    mu=args.viscosity
    rho=args.density

    logging.info('density: %e g/cm3'%rho)
    logging.info('dynamic viscosity : %e dyn s/cm2'%mu)

    logging.info('\n* Tracer properties')
    D=args.diffusion_coef
    sigma_gauss=args.sigma
    logging.info('Free diffusion coef: %e cm2/s'%D)
    logging.info('STD of initial gaussian profile: %e '%sigma_gauss)
    xi_gauss=args.initial_pos
    logging.info('Initial position: %e cm2'%xi_gauss)





    logging.info('\n * Lateral BC')
    compliance=args.compliance
    logging.info('outer compliance: %e '%compliance)
    if compliance <= 0 :
        lateral_bc='free'
        logging.info('right BC will be set to the free assumption')
    else :
        lateral_bc='compliance'
        logging.info('right BC will be set to the compliance assumption')

    fluid_parameters = {'mu': mu, 'rho': rho, 'dt':dt_fluid}
    tracer_parameters={'kappa': D, 'dt':dt_advdiff}



    # Mesh
    logging.info(title1('Meshing'))

    logging.info('cell size : %e cm'%(np.sqrt(DR**2+DY**2)))
    logging.info('nb cells: %i'%(Nl*Nr*2))

    mesh_f= RectangleMesh(Point(0, Rv), Point(L, Rpvs), Nl, Nr)

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


    #FEM space

    logging.info(title1("Set FEM spaces"))

    logging.info('\n * Fluid')
    Vf_elm = VectorElement('Lagrange', triangle, 2)
    Qf_elm = FiniteElement('Lagrange', triangle, 1)
    Wf_elm = MixedElement([Vf_elm, Qf_elm])
    Wf = FunctionSpace(mesh_f, Wf_elm)
    logging.info('Velocity : "Lagrange", triangle, 2')
    logging.info('Pressure : "Lagrange", triangle, 1')

    logging.info('\n * Tracer')
    Ct_elm = FiniteElement('Lagrange', triangle, 1)
    Ct = FunctionSpace(mesh_f, Ct_elm)
    logging.info('Concentration : "Lagrange", triangle, 1')


    


    # Setup of boundary conditions
    logging.info(title1("Boundary conditions"))

    import sympy
    tn = sympy.symbols("tn")

    sin = sympy.sin
    cos = sympy.cos
    sqrt = sympy.sqrt

    logging.info('\n * inner face flow parameters')
    ai=args.ai
    fi=args.fi
    phii=args.phii
    logging.info('ai (dimensionless): '+'%e '*len(ai)%tuple(ai))
    logging.info('fi (Hz) : '+'%e '*len(fi)%tuple(fi))
    logging.info('phii (rad) : '+'%e '*len(phii)%tuple(phii))


    #Brain outflow due to vessel contraction/dilation
    #blood_vol=12.5e-3#*1.5  #cm3 total cerebral blood volume
    brainvolume= pi*Rv**2*L
    brainsurface=2*pi*Rv*L
    blood_vol=0.035*brainvolume
    pc_arterioles=0.01 #pc of total blood volume in arterioles
    V0_arterioles=pc_arterioles*blood_vol
    Asas=pi*(Rpvs**2)-pi*Rv**2

    #CSF production
    Qprod=6e-6 #cm3/s
    #Steady velocity due to production
    Uprod=Qprod/Asas


    #Concentration in the brain 
    cbrain = Expression('c0*(1-tn)', c0=1, tn = 0, degree=2) 

    logging.info('brain volume: %e cm3 '%brainvolume)
    logging.info('brain surface: %e cm2 '%brainsurface)
    logging.info('blood volume: %e cm3 '%blood_vol)
    logging.info('pourcentage of arterioles: %e pc '%(pc_arterioles*100))

    functionV=V0_arterioles*(1+sum([a*sin(2*pi*f*tn+phi) for a,f,phi in zip(ai,fi,phii)])) # blood volume
    functionU = sympy.diff(functionV,tn)/brainsurface # velocity
    #functionU=V0_arterioles*(sum([a*2*pi*f*cos(2*pi*f*tn+phi) for a,f,phi in zip(ai,fi,phii)]))/brainsurface
    ubottom = sympy.printing.ccode(functionU) 

    vf_bottom = Expression(('0',ubottom), tn = 0, degree=2)  # no slip no gap condition at vessel wall 

    logging.info('\n * Lateral assumption')
    logging.info(lateral_bc)

    logging.info('\n * Fluid')
    logging.info('Left : zero pressure')

    if lateral_bc=='free' :
        logging.info('Right : zero pressure')
    else :
        logging.info('Right : compliance')

    logging.info('Top : no slip no gap fixed wall')
    logging.info('Bottom : no slip and brain inflow')

    logging.info('\n * Tracer concentration')
    logging.info('Left : zero concentration')

    if lateral_bc=='free' :
        logging.info('Right : zero concentration')
    else :
        logging.info('Right : no flux')


    logging.info('Top : no flux')
    logging.info('Bottom : no flux')


    # Now we wire up

    if lateral_bc=='free' :
        bcs_fluid = {'velocity': [(facet_lookup['y_min'],vf_bottom),
                                (facet_lookup['y_max'], Constant((0,0))),
                                (facet_lookup['x_max'], Constant((-Uprod,0)))],
                    'traction': [],  
                    'pressure': [(facet_lookup['x_min'], Constant(0))]}

    else  :

        E=100
        P0=5330
        Pn=5330
        Pout=Expression('Pn', Pn=Pn, degree=0)#
  
        # Compute pressure to impose according to the flow at previous time step and compliance

        bcs_fluid = {'velocity': [(facet_lookup['y_min'],vf_bottom),
                                (facet_lookup['y_max'], Constant((0,0))),
                                (facet_lookup['x_max'], Constant((-Uprod,0)))],
                    'traction': [],  
                    'pressure': [(facet_lookup['x_min'], Pout)]}


    if lateral_bc=='free' :
        bcs_tracer = {'concentration': [(facet_lookup['x_max'], Constant(0)),
                                        (facet_lookup['x_min'], Constant(0)),
                                        (facet_lookup['y_min'], cbrain)],
                    'flux': [(facet_lookup['y_max'], Constant(0))]}
    else :
        bcs_tracer = {'concentration': [(facet_lookup['x_min'], Constant(0)),
                                        (facet_lookup['y_min'], cbrain)],
                    'flux': [(facet_lookup['x_max'], Constant(0)),
                            (facet_lookup['y_max'], Constant(0))]}



    # We collect the time dependent BC for update
    driving_expressions = (vf_bottom)

    # Initialisation : 
    logging.info(title1("Initialisation"))
    logging.info("\n * Fluid")
    logging.info("Velocity : zero field")
    logging.info("Pressure : zero field")
    uf_n = project(Constant((0, 0)), Wf.sub(0).collapse())
    pf_n =  project(Constant(0), Wf.sub(1).collapse())

    #logging.info("\n * Tracer")
    #logging.info("Concentration : Gaussian profile")
    #logging.info("                Centered at mid length")
    #logging.info("                STD parameter = %e"%sigma_gauss)


    #c_0 = Expression('exp(-a*pow(x[0]-b, 2)) ', degree=1, a=1/2/sigma_gauss**2, b=xi_gauss)
    c_n =  project(Constant(0),Ct)

    # Save initial state
    uf_n.rename("uf", "tmp")
    pf_n.rename("pf", "tmp")
    c_n.rename("c", "tmp")
    uf_out << (uf_n, 0)
    pf_out << (pf_n, 0)
    c_out << (c_n, 0)

    files=[csv_p,csv_u,csv_c]
    fields=[pf_n,uf_n.sub(0),c_n]
    
    slice_line = line([0,(Rpvs+Rv)/2],[L,(Rpvs+Rv)/2], 100)

    for csv_file,field in zip(files,fields) :
        #print the x scale
        values=np.linspace(0,L,100)
        row=[0]+list(values)
        csv_file.write(('%e'+', %e'*len(values)+'\n')%tuple(row))
        #print the initial 1D slice
        values = line_sample(slice_line, field) 
        row=[0]+list(values)
        csv_file.write(('%e'+', %e'*len(values)+'\n')%tuple(row))



    ############# RUN ############3

    logging.info(title1("Run"))

    # Time loop
    time = 0.
    timestep=0

    

    intQout=0

    # Here I dont know if there will be several dt for advdiff and fluid solver
    while time < tfinal:

        # Update boundary conditions
        for expr in driving_expressions:
            hasattr(expr, 'tn') and setattr(expr, 'tn', time)

        vf_bottom.tn=time


        if lateral_bc=='compliance' :
            # Compliance law dP/dt=dP/dV.dV/dt=dP/dV.qout=E*P*qout
    
            #qout=2 pi int_Rv^Rpvs u r dr 
            z, r = SpatialCoordinate(mesh_f)
            ds = Measure('ds', domain=mesh_f, subdomain_data=fluid_bdries)
            n = FacetNormal(mesh_f)
            Qout=assemble(2*pi*r*dot(uf_n, n)*ds(facet_lookup['x_max']))
            #Pn+1=pn+dt*(E*P*qoutn)
            #Pout.Pn=Pn+dt*E*Pn*Qout
            # dP/dt=dP/dV.dV/dt=dP/dV.qout=E*P*qout
            # ln(p)-ln(p0)=E int_0^t Qout
            #p=p0exp(E int_0^t Qout) 
            intQout+=Qout*dt
            Pout.Pn=P0*exp(E*intQout)

            print('Right pressure : %e mmHg \n'%(Pout.Pn/1333))


        # Solve fluid problem
        uf_, pf_ = solve_fluid(Wf, u_0=uf_n,  f=Constant((0, 0)), bdries=fluid_bdries, bcs=bcs_fluid,
                            parameters=fluid_parameters)


        tracer_parameters["T0"]=time
        tracer_parameters["nsteps"]=1
        tracer_parameters["dt"]=dt

        # Solve tracer problem
        c_, T0= solve_adv_diff(Ct, velocity=uf_, mesh_displacement=Constant((0,0)), f=Constant(0), phi_0=c_n,
                                  bdries=fluid_bdries, bcs=bcs_tracer, parameters=tracer_parameters)


        # Update current solution
        uf_n.assign(uf_)
        pf_n.assign(pf_)
        c_n.assign(c_)

        #Update time
        time += dt
        timestep+=1

        # Save output
        if(timestep % int(toutput/dt) == 0):

            
            logging.info("\n*** save output time %e s"%time)
            logging.info("number of time steps %i"%timestep)


            # may report Courant number or other important values that indicate how is doing the run

            uf_.rename("uf", "tmp")
            pf_.rename("pf", "tmp")
            c_.rename("c", "tmp")
            uf_out << (uf_, time)
            pf_out << (pf_, time)
            c_out << (c_, time)


            # Get the 1 D profiles at umax (to be changed in cyl coordinate)
            mesh_points=mesh_f.coordinates()                                                      
            x=mesh_points[:,0]
            y=mesh_points[:,1]
            xmin=min(x)
            xmax=max(x)
            ymin=min(y)
            ymax=max(y)
                        
            #slice_line = line([xmin,(ymin+ymax)/2],[xmax,(ymin+ymax)/2], 100)

            logging.info('Rpvs : %e'%ymax)
            logging.info('Rvn : %e'%ymin)

            files=[csv_p,csv_u,csv_c]
            fields=[pf_n,uf_n.sub(0),c_n]
            field_names=['pressure (dyn/cm2)','axial velocity (cm/s)','concentration']

            for csv_file,field,name in zip(files,fields,field_names) :
                #values = line_sample(slice_line, field)
                values =profile_cyl(field,xmin,xmax,ymin,ymax)
                logging.info('Max '+name+' : %.2e'%max(abs(values)))
                #logging.info('Norm '+name+' : %.2e'%field.vector().norm('linf'))
                row=[time]+list(values)
                csv_file.write(('%e'+', %e'*len(values)+'\n')%tuple(row))

            csv_rv.write('%e, %e'%(time,ymin))




if __name__ == '__main__':
    # Create the parser
    my_parser = argparse.ArgumentParser(description='Launch a simulation of PVS flow')

    # Add the arguments

    my_parser.add_argument('-j','--job_name',
                        type=str,
                        default="SAS",
                        help='Name of the job')

    my_parser.add_argument('-o','--output_folder',
                            type=str,
                            default="../output/",
                            help='Folder where the results are stored')

    my_parser.add_argument('-ri','--radius_inner',
                        type=float,
                        default=0.357,
                        help='SAS inner radius')

    my_parser.add_argument('-ro','--radius_outer',
                        metavar='Rpvs',
                        type=float,
                        default=0.357+40e-4,
                        help='SAS outer radius')

    my_parser.add_argument('-l','--length',
                        type=float,
                        default=1,
                        help='Length of the SAS')

    my_parser.add_argument('-ai',
                        type=float,
                        nargs='+',
                        default=[0.005],
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
                        default=0.3,
                        help='final time')

    my_parser.add_argument('-toutput',
                        type=float,
                        default=0.01,
                        help='output period')                       

    my_parser.add_argument('-dt','--time_step',
                        type=float,
                        default=1e-3,
                        help='time step')

    my_parser.add_argument('-mu','--viscosity',
                        type=float,
                        default=7e-3,
                        help='Dynamic viscosity of the fluid')

    my_parser.add_argument('-rho','--density',
                        type=float,
                        default=1,
                        help='Density of the fluid')

    my_parser.add_argument('-c','--compliance',
                        type=float,
                        default=-1,
                        help='Outflow compliance bc for CSF')


    my_parser.add_argument('-nr','--N_radial',
                        type=int,
                        default=4,
                        help='number of cells in the radial direction')

    my_parser.add_argument('-nl','--N_axial',
                        type=int,
                        default=0,
                        help='number of cells in the axial direction')

    my_parser.add_argument('-d','--diffusion_coef',
                        type=float,
                        default=2e-8,
                        help='Diffusion coefficient of the tracer')

    my_parser.add_argument('-s','--sigma',
                        type=float,
                        default=0.01,
                        help='STD gaussian init for concentration')

    my_parser.add_argument('-xi','--initial_pos',
                        type=float,
                        default=0.5,
                        help='Initial center of the gaussian for concentration')

    args = my_parser.parse_args()


    # Execute the PVS simulation

    SAS_simulation(args)



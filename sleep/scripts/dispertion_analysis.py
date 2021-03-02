#! /usr/bin/env python3
from sleep.fbb_DD.PVS_simulation import PVS_simulation


# the diffusion coef of tracers range from 2e-8 to 1e-7 cm2/s
# We define D0 = 2e-8 , D1=4e-8, D3=6e-8,  D4=1e-7

#define a class for the parameters
class params (object):
    def __init__(self):
        self.job_name="default_PVS"
        self.output_folder="./output"
        self.radius_vessel=8e-4
        self.radius_pvs=10e-4
        self.length=100e-4
        self.ai=[0.01]
        self.fi=[10]
        self.phii=[0]
        self.tend=1
        self.toutput=0.01
        self.time_step=1e-3
        self.viscosity=7e-3
        self.density=1
        self.resistance=0
        self.ale_parameter=1
        self.N_radial=8
        self.N_axial=0
        self.diffusion_coef=2e-8
        self.sigma=1e-4

if __name__ == '__main__':
    
    #initialise the parameters with default values
    params=params()

    # Analysis for free lateral boundaries, ie p=0 on left and right

    # run a batch of simulation with varying frequencies / amplitudes

    # reference
    params.job_name='diffusion-D0'
    params.ai=[0]
    params.fi=[0]
    params.tend=10/0.001
    params.toutput=params.tend/100
    params.time_step=params.tend/1000
    params.diffusion_coef=2e-8

    # -j diffusion-D0 -ai 0 -fi 0 -tend 10000 -toutput 100 --time_step 10 --diffusion_coef 2e-8

    PVS_simulation(params)

    # 0.001 Hz (sleep states cycle) - 20 % vessel wall deformation
    params.job_name='freeBC-sleepstates-D0'
    params.ai=[0.2]
    params.fi=[0.001]
    params.tend=10/0.001
    params.toutput=params.tend/100
    params.time_step=params.tend/1000
    params.diffusion_coef=2e-8

    PVS_simulation(params)

    # 1/5 Hz (active dilation contraction) - 10 % vessel wall deformation
    params.job_name='freeBC-activedilation-D0'
    params.ai=[0.1]
    params.fi=[1/5]
    params.tend=10*5
    params.toutput=params.tend/100
    params.time_step=params.tend/1000
    params.diffusion_coef=2e-8

    PVS_simulation(params)

    # 10 Hz cardiac frequencies  - 1 % vessel wall deformation
    params.job_name='freeBC-cardiac-D0'
    params.ai=[0.01]
    params.fi=[10]
    params.tend=1.
    params.toutput=params.tend/100
    params.time_step=params.tend/1000
    params.diffusion_coef=2e-8

    PVS_simulation(params)

    

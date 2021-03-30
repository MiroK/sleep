#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 10:25:13 2021

@author: alexandra
"""
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt, exp, log, pi

import os

def fA(t, a, f, phi=0, Rv0=8e-4, h0=2e-4) :
    w=2*np.pi*f
    Av0=np.pi*Rv0**2
    Aast0=np.pi*(Rv0+h)**2
    A0=Aast0-Av0
    return A0*(1+a*np.sin(w*t+phi*2*np.pi))

def fdAdt(t, a, f, phi=0, Rv0=8e-4, h0=2e-4) :
    w=2*np.pi*f
    Av0=np.pi*Rv0**2
    Aast0=np.pi*(Rv0+h)**2
    A0=Aast0-Av0
    return A0*(w*a*np.cos(w*t+phi*2*np.pi))

def fQ (s, t, a, f, l, phi=0, Rv0=8e-4, h0=2e-4) :
    return -fdAdt(t, a, f, phi, Rv0,h0)*(s-l)

def U (s, t, a, f, l,  phi=0, Rv0=8e-4, h0=2e-4) :
    return fQ(s, t, a, f, l, phi, Rv0, h0)/fA( t, a, f, phi, Rv0, h0)

# compute FWHM using the roots of a fitted spline
from scipy.interpolate import UnivariateSpline

def FWHM(X,Y):
    spline = UnivariateSpline(X, Y-np.max(Y)/2, s=0)
    roots=spline.roots() # find the roots
    if len(roots)==1 :
        #print('Warning : there is only one root to the spline, we return 2*dist(max)')
        fwhm=2*abs(X[np.argmax(Y)]-roots[0])
    elif len(roots)==2 :
        fwhm=abs(roots[1]-roots[0])
    else :
        #print('Error : unexpected number of roots of the fitted spline')
        fwhm=np.nan
    
    return fwhm


from sklearn import linear_model

# With only one linear fit 
def estimate_diff_FWHM_fit(spanTime,spanFWHM) :
    spanSigma=spanFWHM/(2*sqrt(2*log(2)))
    
    X = np.array(spanTime).reshape((-1, 1))
    y = np.array(spanSigma**2)/2

    regressor = linear_model.LinearRegression()
    regressor.fit(X, y)
    a = regressor.coef_[0]
    b = regressor.intercept_

    plt.figure()
    plt.plot(spanTime,spanSigma**2/2,'*')
    if a :
        plt.plot(spanTime,a*(spanTime-(-b/a)))
    else :
        plt.plot(spanTime,a*(spanTime))
    plt.xlabel('time (s)')
    plt.ylabel(u'$\sigma^2/2$')
    plt.title('linear fit for the FWHM method')

    return a

# with the slope, can vary with time
def estimate_diff_FWHM_slope(spanTime,spanFWHM) :
    spanSigma=spanFWHM/(2*sqrt(2*log(2)))
    y = np.array(spanSigma**2)/2

    slope=np.diff(y)/np.diff(spanTime)
    
    plt.plot(spanTime[0:-1],slope,'*')
    plt.xlabel('time (s)')
    plt.ylabel('$Deff$')
    plt.title('Slope of \sigma^2/2 vs t')
    
    return slope

def gaussian (x,t,s,D,L,xi) :
    return s/sqrt(2*D*t)*np.exp(-(x-xi)**2/(4*D*t))

from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score

def estimate_diff_fit(spanC,spanX,t,sigma,L,xi) :
    
    
    def fitfunction(x,D) :
        tini=sigma**2/(2*D)
        return  gaussian (spanX,t+tini,sigma,D,L,xi)
    
    popt, pcov = curve_fit(fitfunction, spanX,spanC,bounds=[1e-9,1e-4])
    
    fit = fitfunction(spanX,*popt)
    
    err=mean_squared_error(spanC, fit) 
        
    return popt[0], err


file_database='/home/alexandra/Documents/Python/sleep/sleep/output/'
file_database+='data_dispersion_nrem_limtmax.csv'


study='nrem'

# set a condition on time analysis to stay in 1D in the PVS
conditiontime=True
conditionalpha=True

Database=[]
datalabel=['job', 'Rv0', 'Rpvs', 'L', 'DX', 'dt', 'rho', 'mu', 'D', 'sigma', 'xi', 'f', 'umax','pmax', 'Pe', 'Re', 'Wo', 'Fo' , 'A', 'beta', 'dPdx', 'T','nPeriod' , 'tend', 'FWHMend','DestFWHM', 'Destfit', 'RFWHM', 'Rfit', 'amp','thetaa', 'tau']

format_joblabel=study+'-%ipm-%imHz-D0'

l=200e-4
D=2e-8



if study=='global':
    Rv=16/2*1e-4
    Rpvs=21/2*1e-4
    spanf=np.array([3,10,50,200,700,2000,5000,10000])/1000
    spana = [0.01,0.02, 0.04, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    
    
if study=='baseline':
    Rv=16/2*1e-4
    Rpvs=21/2*1e-4
    spanf=np.array([5000,6000,7000,8000,9000,10000])/1000
    spana = [0.001,0.002,0.005, 0.01, 0.02, 0.03 , 0.04 ]


if study=='nrem':
    Rv=16.5/2*1e-4
    Rpvs=20.1/2*1e-4
    spanf=np.array([100,200,500,1000,2000,5000,10000])/1000
    spana = [0.005, 0.01, 0.02, 0.05, 0.10, 0.2, 0.5 ]


if study=='rem':
    Rv=19/2*1e-4
    Rpvs=22.5/2*1e-4
    spanf=np.array([5000,6000,7000,8000,9000,10000])/1000
    spana = [0.001,0.002,0.005, 0.01, 0.02, 0.03 , 0.04 ]

h=Rpvs-Rv

Ntot=len(spanf)*len(spana)

iteration=0
for f in spanf :
    for a in spana : 
        iteration+=1
        
        print('iteration %i/%i'%(iteration,Ntot))
        
        plt.close('all')
        
        f=float('%.0e'%f)
        a=round(a*1000)/1000
     
        Umax=U(0, 0, a, f, l, Rv0=Rv, h0=(Rpvs-Rv))
        Pe=h*Umax/D
        if (Umax*1e4<1000)&(Pe>0.8) :
            job=format_joblabel%(round(a*1000), round(f*1000))
            print('job',job)
            
            #### Simulation results
            
            rep='/home/alexandra/Documents/Python/sleep/sleep/output/'+job+'/'
            outputfolder=rep+'postprocess/'
            
            if not os.path.exists(outputfolder):
                os.makedirs(outputfolder)
            
            outputfile = open(outputfolder+'post-process.txt', "w")
            
            outputfile.write('#'*20)
            outputfile.write('\n# Dispersion analysis of '+job)
            outputfile.write('\n'+'#'*20)
            

            
            file='profiles/concentration.txt'
            
            Data=np.loadtxt(rep+file,delimiter=',')
            
            
            t=Data[1:,0]
            
            concentration=Data[:,1:]
            
            pressure=np.loadtxt(rep+'profiles/pressure.txt',delimiter=',')[1:,1:]
            
            velocity=np.loadtxt(rep+'profiles/velocity.txt',delimiter=',')[1:,1:]
            
            #### Simulation parameters
            
            import re
            scinot = re.compile('[+\-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)')
            
            # Should get D, L, rho, nu, f, amp, Rpvs, Rv0,sigma and x0 from the log file
            file='PVS_info.log'
            
            
            with open(rep+file) as fl:
                line = fl.readline()
                while line:
                    line = fl.readline()
                    if line[0:6] == 'Vessel' :
                        Rv0=float(re.findall(scinot, line)[0])
                    elif line[0:6] == 'PVS ra' :
                        Rpvs=float(re.findall(scinot, line)[0])  
                    elif line[0:6] == 'PVS le' :
                        L=float(re.findall(scinot, line)[0])   
                    elif line[0:6] == 'densit' :
                        rho=float(re.findall(scinot, line)[0])
                    elif line[0:6] == 'dynami' :
                        mu=float(re.findall(scinot, line)[0])   
                    elif line[0:6] == 'Free d' :
                        D=float(re.findall(scinot, line)[0]) 
                    elif line[0:6] == 'STD of' :
                        sigma=float(re.findall(scinot, line)[0])  
                    elif line[0:6] == 'Initia' :
                        xi=float(re.findall(scinot, line)[0])
                    elif line[0:6] == 'fi (Hz':
                        f=float(re.findall(scinot, line)[0]) # Check if it works for several frequencies
                    elif line[0:6] == 'ai (di':
                        amp=float(re.findall(scinot, line)[0])*100 # Check if it works for several frequencies
                    elif line[0:6] == 'time s':
                        time_step=float(re.findall(scinot, line)[0])
                    elif line[0:6] == 'cell s':
                        cell_size=float(re.findall(scinot, line)[0])    
                        
            outputfile.write('\n\nRv : %.2f um'%(Rv0*1e4))
            outputfile.write('\nRpvs : %.2f um'%(Rpvs*1e4))
            outputfile.write('\nLength : %.2f um'%(L*1e4))
            
            x=np.linspace(0,L,len(concentration[0,:]))
            
            outputfile.write('\n\nDensity : %.2e g/cm3'%rho)
            outputfile.write('\nViscosity : %.2e dyn s /cm2'%mu)
            nu=mu/rho
            
            outputfile.write('\n\nDiffusion coefficient: %.2e cm2/s'%(D))      
            outputfile.write('\nSTD gausian : %.2e cm'%(sigma))
            outputfile.write('\ncenter gaussian: %.2e um'%(xi*1e4))
            
            outputfile.write('\n\nfrequency : %.2e Hz'%f)
            outputfile.write('\n\namplitude : %.2e pc'%amp)
            
            outputfile.write('\nspatial resolution : %.2e um'%(cell_size*1e4))
            outputfile.write('\ntemporal resolution : %.2e s'%time_step)
            
            ### Compute dimensionless numbers
            
            umax=np.max(abs(velocity))
            pmax=np.max(abs(pressure))
            
            
            w=2*pi*f
            h=Rpvs-Rv0
            
            Pe=h*umax/D
            Re=rho*umax*h/mu
            Wo=h*sqrt(w/mu)
            Fo=D*t[-1]/xi**2
            
            outputfile.write('\nUmax : %.3f um/s'%(umax*1e4))
            outputfile.write('/npmax : %.3f pa \n'%(pmax/10))
            
            outputfile.write('\nmax Reynolds number : %.0e'%Re)
            outputfile.write('\nmax Peclet number : %.0e'%Pe)
            outputfile.write('\nWomersley number : %.0e'%Wo)
            outputfile.write('\nFourier number : %.0e'%Fo)
            
            dPdx=pmax/L
            beta=D/mu
            A=pi*(Rpvs**2-Rv0**2)
            
            
            
            # We want to look to the results at each period (net flow 0)
            dtoutput=t[1]-t[0]
            tend=t[-1]
            
            
            
            if f:
                T=1/f
                ishift=0 #int(T/dtoutput*1/4)
            else :
                T=t[-1]/10
                ishift=0
                
            outputfile.write('\nfinal time simulation : %.2e s'%tend)
            outputfile.write('\noutput period : %.2e s'%dtoutput)
            outputfile.write('/nperiod : %.2e s'%T)
            DX=(x[1]-x[0])*1e4
            outputfile.write('\nspatial resotion : %.2e um'%(DX))
            
            
            iperiodic=(np.arange(ishift,len(t),int(T/dtoutput)))
            
           
            
            span_FWHM=[]
            for c in concentration :
                span_FWHM.append(FWHM(x,c))
                #plt.plot(x,c)
                
            span_FWHM=np.array(span_FWHM) 
            #plt.xlim([35e-4,65e-4])
            
             
            ### Estimation of D from FWHM
            plt.figure()
            Dest=estimate_diff_FWHM_fit(t[iperiodic[::]],span_FWHM[iperiodic[::]])
            DestFWHM=Dest
            
            DestFWHM=max(Dest,D)
            
            tmax=(L/4)**2/DestFWHM/2
            
            if conditiontime :
                #condition diffusion
                while t[iperiodic[-1]]>tmax : 
                    outputfile.write('\n* warning : limitation of tend due to diffusion')
                    it=np.where(t>=tmax)[0][0]
                    ii=np.where(iperiodic>=it)[0][0]
                    iperiodic=iperiodic[0:ii]
  
                    plt.figure()
                    Dest=estimate_diff_FWHM_fit(t[iperiodic[::]],span_FWHM[iperiodic[::]])
                    DestFWHM=max(Dest,D)
                    tmax=(L/2)**2/DestFWHM/2
                    

            thetaa=amp*L/(1+amp)/(L/2-2*np.sqrt(2*DestFWHM*t[iperiodic]))
            
            if conditionalpha :
                #condition advection diffusion
                while thetaa[-1]>3 : 
                    outputfile.write('\n* warning : limitation of tend due to theta a')
                    ii=np.where(thetaa>=3)[0][0]
                    iperiodic=iperiodic[0:ii]
                    Dest=estimate_diff_FWHM_fit(t[iperiodic[::]],span_FWHM[iperiodic[::]])
                    DestFWHM=max(Dest,D)
                    thetaa=amp*L/(1+amp)/(L/2-2*np.sqrt(2*DestFWHM*t[iperiodic]))

            plt.savefig(outputfolder+'disp_FWHM.png')                 
            
            
            outputfile.write('\n\nfinal time analysis : %.2e s'%t[iperiodic[-1]])
            
            
            
            nPeriod=len(iperiodic)-1
            outputfile.write('\nnumber of period analysed: %i'%nPeriod)
            
            if nPeriod==0 :
                outputfile.write('\nOscillatory dispersion analysis aborted')
                continue
            
            plt.figure()
            plt.plot(t[iperiodic],span_FWHM[iperiodic]*1e4,'*')
            plt.xlabel('time (s)')
            plt.ylabel('FWHM (um)')
            plt.title('computed FWHM')
            
            plt.savefig(outputfolder+'FWHM.png')
            
            tau=t[iperiodic[-1]]/(h**2/D)
            outputfile.write('\ntau: %.2e'%tau)
            outputfile.write('\ntheta a : %.1e'%thetaa[-1])
            
            outputfile.write('\n\nEstimation of D from FWHM : %.2e'%DestFWHM)
            
            
            
            #When the apparent Diffusion coefficient is smaller with oscillation than diffusion alone at small time. What is going on? 
            
            ### Estimation of D from fit
            
            # we take the time at last period
            plt.figure()
            spanDest=[]
            for itime in iperiodic[1::] : 
                #print('\nEstimate at time %f s'%t[itime])
            
                Dest, err = estimate_diff_fit(concentration[itime],x,t[itime],sigma,L,xi)
            
                #print('Estimation of D from fit : %.2e'%Dest)
                #print('Fit error : %.2e'%err)
                spanDest.append(Dest)
                plt.plot(t[itime],Dest,'*')
                
            plt.xlabel('time (s)')
            plt.ylabel('D (cm2/s)')
            plt.title('Estimate of D with fit method')
            
            plt.savefig(outputfolder+'disp_gauss.png') 
            
            Destfit=spanDest[-1]
            outputfile.write('\nEstimation of D from fit : %.2e'%Destfit)
            
            
            outputfile.write('\n\nEstimation of R from FWHM : %.2e'%(DestFWHM/D))
            outputfile.write('\nEstimation of R from fit : %.2e'%(Destfit/D))
            
            
            plt.figure()
            color='rgbkcmyrgbkcmyrgbkcmyrgbkcmyrgbkcmyrgbkcmyrgbkcmyrgbkcmyrgbkcmyrgbkcmyrgbkcmyrgbkcmy'
            icol=0
            for i in range(0,len(spanDest)) :
                plt.plot(x,concentration[iperiodic[i+1]],'.',color=color[icol])
                plt.plot(x,gaussian(np.array(x),t[iperiodic[i+1]]+sigma**2/(2*spanDest[i]),sigma,spanDest[i],L,xi),color=color[icol],alpha=0.5)
                icol+=1
            plt.xlabel('x (cm)')
            plt.ylabel('concentration')
            plt.title('Fit of simulation results')
            #plt.xlim([L-50e-4,L])
            
            plt.savefig(outputfolder+'concentration.png') 
            
            outputfile.close()
            
            #update database
            Database.append([job, Rv0, Rpvs, L, cell_size, time_step, rho, mu, D, sigma, xi, f, umax,pmax, Pe, Re, Wo, Fo , A, beta, dPdx, T,nPeriod , t[iperiodic[-1]], span_FWHM[iperiodic[-1]],DestFWHM, Destfit,DestFWHM/D, Destfit/D, amp, thetaa[-1],tau])
                                                            


# save the database
labelstring=''
for d in datalabel:
    labelstring+=d+', '
    
labelstring=labelstring[0:-1]

f = open(file_database, "w")
f.write(labelstring+'\n')
f.close()

formatstring='%s'+' , %.6e'*31

f = open(file_database, "a")
for d in Database :
    #print(formatstring%tuple(d))
    f.write(formatstring%tuple(d)+'\n')
f.close()
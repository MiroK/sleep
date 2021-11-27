#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 10:39:42 2021

@author: alexandra
"""
import numpy as np
from matplotlib import pyplot as plt

from sleep.stages.cycles import State, Cycle 

plt.style.use('styles/muted.mplstyle')

data=np.loadtxt('/home/alexandra/Documents/Python/sleep/sleep/output/sleepcycle/profiles/mass.txt',delimiter=',')
time=data[:,0]
meanCsleep=data[:,3]

data=np.loadtxt('/home/alexandra/Documents/Python/sleep/sleep/output/awake/profiles/mass.txt',delimiter=',')
time=data[:,0]
meanCawake=data[:,3]


data=np.loadtxt('/home/alexandra/Documents/Python/sleep/sleep/output/REMsleep/profiles/mass.txt',delimiter=',')
time=data[:,0]
meanCREM=data[:,3]


data=np.loadtxt('/home/alexandra/Documents/Python/sleep/sleep/output/NREMsleep/profiles/mass.txt',delimiter=',')
time=data[:,0]
meanCNREM=data[:,3]

data=np.loadtxt('/home/alexandra/Documents/Python/sleep/sleep/output/diffusioncycle/profiles/mass.txt',delimiter=',')
time=data[:,0]
meanCdiffusion=data[:,3]

plt.figure(1)
plt.plot(time,meanCdiffusion,'k:',label='Diffusion only')
plt.plot(time,meanCsleep,label='Normal sleeping cycle')
plt.plot(time,meanCawake,label='Awake cycle')
plt.plot(time,meanCNREM,label='NREM sleeping cycle')
plt.plot(time,meanCREM,label='REM sleeping cycle')

plt.legend()

plt.ylabel('mean tracer concentration')
plt.xlabel('time (s)')

plt.yscale('log')
plt.grid(True, which="both", ls="-", color='0.65')



#### illustration for the sleeping stages

#create several states
Awake=State(name='Awake',Rv=6.4e-4,h0=2.7e-4,freqtable={'cardiac':10.1,'resp':3.03,'LF':0.476,'VLF':0.167},amptable={'cardiac':0.04,'resp':0.021,'LF':0.035,'VLF':0.047})
REM=State(name='REM',Rv=7.68e-4,h0=2.214e-4,freqtable={'cardiac':11.36,'resp':2.941,'LF':0.476,'VLF':0.130},amptable={'cardiac':0.042,'resp':0.024,'LF':0.027,'VLF':0.046})
NREM=State(name='NREM',Rv=6.4e-4,h0=2.619e-4,freqtable={'cardiac':10,'resp':2.941,'LF':0.417,'VLF':0.147},amptable={'cardiac':0.037,'resp':0.019,'LF':0.064,'VLF':0.098})
IS=State(name='IS',Rv=7.04e-4,h0=2.511e-4,freqtable={'cardiac':10.1,'resp':2.941,'LF':0.435,'VLF':0.154},amptable={'cardiac':0.042,'resp':0.024,'LF':0.027,'VLF':0.046})

Whisking=State(name='Whisking',Rv=6.592e-4,h0=2.646e-4,freqtable={'cardiac':10.1,'resp':3.03,'LF':0.476,'VLF':0.167},amptable={'cardiac':0.04,'resp':0.021,'LF':0.035,'VLF':0.047})
Locomotion=State(name='Locomotion',Rv=7.168e-4,h0=2.538e-4,freqtable={'cardiac':10.1,'resp':3.03,'LF':0.476,'VLF':0.167},amptable={'cardiac':0.04,'resp':0.021,'LF':0.035,'VLF':0.047})


#create a cycle
sleepcycle=Cycle([(NREM,50),(IS,40),(REM,110),(Awake,10)],transitiontime=2)
awakecycle=Cycle([(Awake,60)],transitiontime=2)
NREMcycle=Cycle([(NREM,50),(Awake,10)],transitiontime=2)
REMcycle=Cycle([(REM,110),(Awake,10)],transitiontime=2)



spantime_awake,listspana_awake,listspanf_awake,spanRv_awake,spanh0_awake,spanRpvs_awake=awakecycle.generatedata(7)
spantime_sleep,listspana_sleep,listspanf_sleep,spanRv_sleep,spanh0_sleep,spanRpvs_sleep=sleepcycle.generatedata(2)
spantime_REM,listspana_REM,listspanf_REM,spanRv_REM,spanh0_REM,spanRpvs_REM=REMcycle.generatedata(4)
spantime_NREM,listspana_NREM,listspanf_NREM,spanRv_NREM,spanh0_NREM,spanRpvs_NREM=NREMcycle.generatedata(7)


plt.figure(2)
plt.plot(spantime_sleep,listspana_sleep['cardiac'],label='Normal sleeping cycle')
plt.plot(spantime_awake,listspana_awake['cardiac'],label='Awake cycle')
plt.plot(spantime_REM,listspana_REM['cardiac'],label='NREM sleeping cycle')
plt.plot(spantime_NREM,listspana_NREM['cardiac'],label='REN sleeping cycle')



import numpy as np
import time
import pickle
import sys
from datetime import datetime

from os import mkdir

def l2n(x): return np.array(x);
def n2l(x): return list(x)

import ecoSystem as es
import greens as gr
import timePropagation as tp

import importlib  #importing the two other files to be used

importlib.reload(es)
importlib.reload(gr)
importlib.reload(tp)

def generateFileBaseName():
    dirName = datetime.today().strftime('%Y%m%d%H%M%S')+'results'+\
            '_'+greenFunc.__name__+\
			'_dt'+'{:.0e}'.format(dt)+\
			'_N'+'{:.0e}'.format(N)+\
			'_T'+'{:.0e}'.format(T)+\
			'_kT'+'{:.0e}'.format(kT)+\
			'_rSteric'+'{:.1e}'.format(eco.sp[eco.rStericC,0])+\
            '_rPassive'+'{:.1e}'.format(eco.sp[eco.rStericC,-1])+\
			'_wS'+'{:.0e}'.format(eco.sp[eco.wSC,0])+\
			'_V'+'{:.0e}'.format(eco.sp[eco.v0C,0])+\
			'_wA'+'{:.0e}'.format(eco.sp[eco.wAC,0])+\
            '_box'+'{:.0e}'.format(boxSize);

    mkdir(dirName)

    fileBaseName = dirName+'/file_'
    return fileBaseName

####### Set up paramters
params = 8; #number of parameters for each particle

######### Number of time steps and particles
dt = 1E-5;
T = 1E5;  # Number of time steps

######### Size of periodic box 
#boxSize = 20


###### Number of snapshots to save
tSave = 10000 # number of snapshots to save
tArray = 10 # Number of files to produce during the run

#### Temperature

#kT = .1;

# Aligment Strength

#wA = -5#-20;#5;


r0active = 0.05

#matching parameters from experiments
robotDiameter = 4.8 # [cm]

passiveSmall = 7 # [cm]
passiveLarge = 28 # [cm]

kAligner = 0.3 # [1/robotDiameter]
kFronter = -0.3 # [1/robotDiameter]


arenaDiameter = 150 # [cm ]

boxSize = r0active*2*(arenaDiameter/robotDiameter)


#number of passive particles
numPassive = 1


#nominal speed of active
#v0 = 30

#Core stiffness
wS0 = 1E2

#greenFunc = gr.greens.grNonInteractingHeteroWithPassivePopPeriodic#grInteractingHeteroPopPeriodic#grNonInteractingHeteroWithPassivePopPeriodic
greenFunc = gr.greens.grInteractingHeteroPopPeriodic#grNonInteractingHeteroWithPassivePopPeriodic

#Take snapshots of the system of only specific particles
particlesToSave = [-1] # -1 saves only the last particle

#To save all particles set flag to 0:
particlesToSaveFlag = 0

# Set up phase space paramters

repeats = 4

#number of particles Matching parameters from expriment of 21 bots +1 passive
Ns = [22]*repeats

#types of alignments
wAs  = l2n([kFronter,kAligner])/r0active # curvity in simulation units of 1/(simulation robot size)
r0passives = l2n([passiveSmall,passiveLarge])/robotDiameter*r0active # passive radius in simulation units of simulation robot size

kTs = l2n([1]) 

#nominal speed of active
v0s = [10]
#kTs = [0.1]

pointsInPhaseSpace = len(r0passives)*len(wAs)*len(kTs)
totTimeEstimate = T/1E5*500 * pointsInPhaseSpace/3600 #[h]

print('Points in phase space :' + str(pointsInPhaseSpace))
print('Estimate total run time: ' +str((totTimeEstimate)) + ' hours')


################# RUN

# Sweep through different number of particles


for N in Ns:
	for wA in wAs:
		for r0passive in r0passives:
			for kT in kTs:
				for v0 in v0s:
					#The following define the parameters of the population in the ecosystem.
					#Can be passed as scalars for homogeneous param or as a vector length of population for heterogenous system.
					N = int(N)

					if particlesToSaveFlag ==0: #Save all
						particlesToSave = range(N)

					#boxSize = 20*(N/128)**0.5 (defined externally for matching conditions in experiment)

					######### Size of periodic box 
					rSteric= r0active*np.ones(N);

					wS=wS0*np.ones(N);
					v = v0*np.ones(N)

					#now make last particles passive and big and passive				
					if numPassive>0:
						rSteric[-numPassive:]=r0passive;
						v[-numPassive:]=0;

					#Initialize the eco system

					eco = es.ecoSystem(N=N,params=params);
					eco.initializeEcosystem(rSteric,wS,v,wA); # generic initiallization of the ecoSystem

					fileBaseName = generateFileBaseName() +'onePassiveBig_'

					#### Special adjustment for initialization "initial conditions"


					# place all active particles around the passive particle pointing at a random direction.
					# Simulation happens in a boxSize X boxSize box with periodic boundary condition
					# passive Particle starts at center of box
					if numPassive>0:
						eco.sp[eco.xC,-1]=boxSize/2 # place passive at center
						eco.sp[eco.yC,-1]=boxSize/2


					#create a halow around passive particle
					x0 = eco.sp[eco.xC,:-1];
					y0 = eco.sp[eco.yC,:-1];
					theta0 = np.arctan2(y0,x0);
					r0 = r0passive + np.random.rand()*(boxSize-r0passive)*0.8;
					dx0 = np.cos(theta0);
					dy0 = np.sin(theta0);
					eco.sp[eco.xC,:-1] = r0*(x0+dx0) + eco.sp[eco.xC,-1]; #spread particles along x direction (outside of large particle)
					eco.sp[eco.yC,:-1] = r0*(y0+dy0) + eco.sp[eco.yC,-1]; #spread particles along y direction (outside of large particle)

					######### Begin the simulation


					#greenFunc = gr.greens.grHeteroPop
					#greenFunc = gr.greens.grNonInteractingHeteroPop


					prop = tp.timePropagation(dt,eco,fileBaseName)
					print('results will be saved to the following base name: ', fileBaseName)
					ti = time.time()

					prop.timeProp5RungePeriodic(T,tSave,tArray,greenFunc,kT,boxSize,particlesToSave)

					tf = time.time()

					print('run time ' +str((tf-ti)) +' seconds' );

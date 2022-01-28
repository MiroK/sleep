from sleep.stages.cycles import Cycle
from sleep.stages.cycles import State

def ReadFixedEffect(file):
    """ Read the mean estimates from a file containing the statistical analysis of the peak to peak data.
        input : 
            - file : a string providing the file where the information must be read 
                    (one file corresponds to a type of mouse, type of vessel, type of measure, frequency band selected)

        output :  a dict. The keys corresponds to the stages. The values are float of the estimates.
                For the stage 'baseline', the value is the Intercept value.
                If the stage is not baseline, then the value is a different value from baseline only if the p value for this stage is < 0.05
                meaning that this stage is significantly different from the baseline.
    """
    import numpy as np

    d={}
    with open(file) as f:
        line=f.readline()# skip first line
        # read data
        line=f.readline()
        lines=[]
        while line:
            line=line.split('\n')[0]
            lines.append(line.split(' '))
            line = f.readline()
    
    #get the intecept value (correspond to the log of the variable)
    intercept=float(lines[0][1])

    #the baseline value correspond to the exp of the intercept
    d['baseline']=np.exp(intercept)

    # the other stages we add the estimate to the intecept only if significative difference with baseline
    for line in lines[1::] :
        d[line[0].replace('"', '')]=np.exp(intercept+float(line[1])*(float(line[3])<0.05))

    return d


#create sleep states :
# the data here are read in the files provided by the statistical analysis of the peak to peak linescan analysis.
folder='../../data/statistics/'
vessel='pen_art_'
mouse='cleanWT_'
analysis='_sleep_'
ftype='_fixedEffects'

data={}

data['Rv']=ReadFixedEffect(folder+vessel+mouse+'lumen'+analysis+'meanRadius'+ftype)
data['h0']=ReadFixedEffect(folder+vessel+mouse+'PVS'+analysis+'meanRadius'+ftype)
data['ampcard']=ReadFixedEffect(folder+vessel+mouse+'PVS'+analysis+'cardiac_amp'+ftype)
data['ampresp']=ReadFixedEffect(folder+vessel+mouse+'PVS'+analysis+'resp_amp'+ftype)
data['periodcard']=ReadFixedEffect(folder+vessel+mouse+'lumen'+analysis+'cardiac_period'+ftype)
data['periodresp']=ReadFixedEffect(folder+vessel+mouse+'lumen'+analysis+'resp_period'+ftype)


data['ampLF']=ReadFixedEffect(folder+vessel+mouse+'PVS'+analysis+'lf_amp'+ftype)
data['ampVLF']=ReadFixedEffect(folder+vessel+mouse+'PVS'+analysis+'vlf_amp'+ftype)

mouse='cleanWT2_' #new frequency band
data['periodLF']=ReadFixedEffect(folder+vessel+mouse+'lumen'+analysis+'lf_period'+ftype)
data['periodVLF']=ReadFixedEffect(folder+vessel+mouse+'lumen'+analysis+'vlf_period'+ftype)
# 

statename='baseline'
freqtable={'cardiac':1/data['periodcard'][statename],'resp':1/data['periodresp'][statename],'LF':1/data['periodLF'][statename],'VLF':1/data['periodVLF'][statename]}
amptable={'cardiac':data['ampcard'][statename]/data['h0'][statename],'resp':data['ampresp'][statename]/data['h0'][statename],'LF':data['ampLF'][statename]/data['h0'][statename],'VLF':data['ampVLF'][statename]/data['h0'][statename]}
#amptable['cardiac']=amptable['cardiac']/2 # I think it is overestimated
amptable['cardiac']=0.03 # arbitrary 3 pc for now
Awake=State(name='Awake',Rv=data['Rv'][statename]*1e-4,h0=data['h0'][statename]*1e-4,freqtable=freqtable,amptable=amptable)


statename='stageNREM'
freqtable={'cardiac':1/data['periodcard'][statename],'resp':1/data['periodresp'][statename],'LF':1/data['periodLF'][statename],'VLF':1/data['periodVLF'][statename]}
amptable={'cardiac':data['ampcard'][statename]/data['h0'][statename],'resp':data['ampresp'][statename]/data['h0'][statename],'LF':data['ampLF'][statename]/data['h0'][statename],'VLF':data['ampVLF'][statename]/data['h0'][statename]}
#amptable['cardiac']=amptable['cardiac']/2 # I think it is overestimated
amptable['cardiac']=0.03 # arbitrary 3 pc for now
NREM=State(name='NREM',Rv=data['Rv'][statename]*1e-4,h0=data['h0'][statename]*1e-4,freqtable=freqtable,amptable=amptable)

statename='stageIS'
freqtable={'cardiac':1/data['periodcard'][statename],'resp':1/data['periodresp'][statename],'LF':1/data['periodLF'][statename],'VLF':1/data['periodVLF'][statename]}
amptable={'cardiac':data['ampcard'][statename]/data['h0'][statename],'resp':data['ampresp'][statename]/data['h0'][statename],'LF':data['ampLF'][statename]/data['h0'][statename],'VLF':data['ampVLF'][statename]/data['h0'][statename]}
#amptable['cardiac']=amptable['cardiac']/2 # I think it is overestimated
amptable['cardiac']=0.03 # arbitrary 3 pc for now
IS=State(name='IS',Rv=data['Rv'][statename]*1e-4,h0=data['h0'][statename]*1e-4,freqtable=freqtable,amptable=amptable)

statename='stageREM'
freqtable={'cardiac':1/data['periodcard'][statename],'resp':1/data['periodresp'][statename],'LF':1/data['periodLF'][statename],'VLF':1/data['periodVLF'][statename]}
amptable={'cardiac':data['ampcard'][statename]/data['h0'][statename],'resp':data['ampresp'][statename]/data['h0'][statename],'LF':data['ampLF'][statename]/data['h0'][statename],'VLF':data['ampVLF'][statename]/data['h0'][statename]}
#amptable['cardiac']=amptable['cardiac']/2 # I think it is overestimated
amptable['cardiac']=0.03 # arbitrary 3 pc for now
REM=State(name='REM',Rv=data['Rv'][statename]*1e-4,h0=data['h0'][statename]*1e-4,freqtable=freqtable,amptable=amptable)


#######
#create awake states :
# the data here are read in the files provided by the statistical analysis of the peak to peak linescan analysis.
analysis='_wake_'


data={}

mouse='cleanWT_'
data['Rv']=ReadFixedEffect(folder+vessel+mouse+'lumen'+analysis+'meanRadius'+ftype)
data['h0']=ReadFixedEffect(folder+vessel+mouse+'PVS'+analysis+'meanRadius'+ftype)
data['ampcard']=ReadFixedEffect(folder+vessel+mouse+'PVS'+analysis+'cardiac_amp'+ftype)
data['ampresp']=ReadFixedEffect(folder+vessel+mouse+'PVS'+analysis+'resp_amp'+ftype)
data['periodcard']=ReadFixedEffect(folder+vessel+mouse+'lumen'+analysis+'cardiac_period'+ftype)
data['periodresp']=ReadFixedEffect(folder+vessel+mouse+'lumen'+analysis+'resp_period'+ftype)

data['ampLF']=ReadFixedEffect(folder+vessel+mouse+'PVS'+analysis+'lf_amp'+ftype)
data['ampVLF']=ReadFixedEffect(folder+vessel+mouse+'PVS'+analysis+'vlf_amp'+ftype)

mouse='cleanWT2_' #new frequency band
data['periodLF']=ReadFixedEffect(folder+vessel+mouse+'lumen'+analysis+'lf_period'+ftype)
data['periodVLF']=ReadFixedEffect(folder+vessel+mouse+'lumen'+analysis+'vlf_period'+ftype)

#We arbitrarily set the VLF component in whisking stage same as in the baseline because we couldnt measure it (too short episodes)
data['ampVLF']['stageWhisking']=data['ampVLF']['baseline']
data['periodVLF']['stageWhisking']=data['periodVLF']['baseline']



statename='baseline'
freqtable={'cardiac':1/data['periodcard'][statename],'resp':1/data['periodresp'][statename],'LF':1/data['periodLF'][statename],'VLF':1/data['periodVLF'][statename]}
amptable={'cardiac':data['ampcard'][statename]/data['h0'][statename],'resp':data['ampresp'][statename]/data['h0'][statename],'LF':data['ampLF'][statename]/data['h0'][statename],'VLF':data['ampVLF'][statename]/data['h0'][statename]}
#amptable['cardiac']=amptable['cardiac']/2 # I think it is overestimated
amptable['cardiac']=0.03 # arbitrary 3 pc for now
Quiet=State(name='Quiet',Rv=data['Rv'][statename]*1e-4,h0=data['h0'][statename]*1e-4,freqtable=freqtable,amptable=amptable)



statename='stageLocomotion'
freqtable={'cardiac':1/data['periodcard'][statename],'resp':1/data['periodresp'][statename],'LF':1/data['periodLF'][statename],'VLF':1/data['periodVLF'][statename]}
amptable={'cardiac':data['ampcard'][statename]/data['h0'][statename],'resp':data['ampresp'][statename]/data['h0'][statename],'LF':data['ampLF'][statename]/data['h0'][statename],'VLF':data['ampVLF'][statename]/data['h0'][statename]}
#amptable['cardiac']=amptable['cardiac']/2 # I think it is overestimated
amptable['cardiac']=0.03 # arbitrary 3 pc for now
Locomotion=State(name='Locomotion',Rv=data['Rv'][statename]*1e-4,h0=data['h0'][statename]*1e-4,freqtable=freqtable,amptable=amptable)



statename='stageWhisking'
freqtable={'cardiac':1/data['periodcard'][statename],'resp':1/data['periodresp'][statename],'LF':1/data['periodLF'][statename],'VLF':1/data['periodVLF'][statename]}
amptable={'cardiac':data['ampcard'][statename]/data['h0'][statename],'resp':data['ampresp'][statename]/data['h0'][statename],'LF':data['ampLF'][statename]/data['h0'][statename],'VLF':data['ampVLF'][statename]/data['h0'][statename]}
#amptable['cardiac']=amptable['cardiac']/2 # I think it is overestimated
amptable['cardiac']=0.03 # arbitrary 3 pc for now
Whisking=State(name='Whisking',Rv=data['Rv'][statename]*1e-4,h0=data['h0'][statename]*1e-4,freqtable=freqtable,amptable=amptable)



stageslist=[Awake,NREM,IS,REM,Quiet,Locomotion,Whisking]
stagesdict={stage.name:stage for stage in stageslist }




def ReadCycle(config_file, name):
    """Read the configuration file and return the corresponding dict.

    Parameters:
        config_file:    The name of the configuration file to read.

    Returns:
        list of config dict
    """

    import yaml

    with open(config_file) as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        config = yaml.load(file, Loader=yaml.FullLoader)


    # We read all the cycles in the cycles config file
    for cycleparam in config:
        # and select the one corresponding to the name given by the user
        if cycleparam['name'] == name:

            cyclestages=[]
            for stagestep in  cycleparam['architecture']:

                try :
                    cyclestages.append((stagesdict[stagestep['stage']],stagestep['duration']))
                except :
                    print("The stage %s is not defined"%stagestep['stage'])
                    exit()
            # create the requested cycle object
            c=Cycle(cyclestages,transitiontime=2, name=cycleparam['name'])
            return c
 
    print("The cycle name doesn't exist")
    exit()  


if __name__ == '__main__':

    cycle=ReadCycle('../stages/cycles.yml','REMsleep')

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
vessel='PenetratingArterioles'
mouse='WT7_'
analysis='sleep_'
ftype='fixedEffects.txt'

datasleep={}

datasleep['Rv']=ReadFixedEffect(folder+vessel+mouse+analysis+'lumen_'+'meanepisode_'+ftype)
datasleep['h0']=ReadFixedEffect(folder+vessel+mouse+analysis+'PVS_'+'meanepisode_'+ftype)

datasleep['ampcard']=ReadFixedEffect(folder+vessel+mouse+analysis+'PVS_'+'cardiac_amp_'+ftype)
datasleep['ampresp']=ReadFixedEffect(folder+vessel+mouse+analysis+'PVS_'+'resp_amp_'+ftype)
datasleep['ampLF']=ReadFixedEffect(folder+vessel+mouse+analysis+'PVS_'+'LF_amp_'+ftype)
datasleep['ampVLF']=ReadFixedEffect(folder+vessel+mouse+analysis+'PVS_'+'VLF_amp_'+ftype)

datasleep['periodcard']=ReadFixedEffect(folder+vessel+mouse+analysis+'PVS_'+'cardiac_period_'+ftype)
datasleep['periodresp']=ReadFixedEffect(folder+vessel+mouse+analysis+'PVS_'+'resp_period_'+ftype)
datasleep['periodLF']=ReadFixedEffect(folder+vessel+mouse+analysis+'PVS_'+'LF_period_'+ftype)
datasleep['periodVLF']=ReadFixedEffect(folder+vessel+mouse+analysis+'PVS_'+'VLF_period_'+ftype)



statename='baseline'
freqtable={'cardiac':1/datasleep['periodcard'][statename],'resp':1/datasleep['periodresp'][statename],'LF':1/datasleep['periodLF'][statename],'VLF':1/datasleep['periodVLF'][statename]}
amptable={'cardiac':datasleep['ampcard'][statename]/datasleep['h0'][statename]/2,'resp':datasleep['ampresp'][statename]/datasleep['h0'][statename]/2,'LF':datasleep['ampLF'][statename]/datasleep['h0'][statename]/2,'VLF':datasleep['ampVLF'][statename]/datasleep['h0'][statename]/2}
amptable['cardiac']/=4 # must be reduced. Could use analytical law
amptable['cardiac']=0
amptable['resp']=0
amptable['LF']=0
amptable['VLF']=0
Awake=State(name='Awake',Rv=datasleep['Rv'][statename]*1e-4,h0=datasleep['h0'][statename]*1e-4,freqtable=freqtable,amptable=amptable)


statename='stageNREM'
freqtable={'cardiac':1/datasleep['periodcard'][statename],'resp':1/datasleep['periodresp'][statename],'LF':1/datasleep['periodLF'][statename],'VLF':1/datasleep['periodVLF'][statename]}
amptable={'cardiac':datasleep['ampcard'][statename]/datasleep['h0'][statename]/2,'resp':datasleep['ampresp'][statename]/datasleep['h0'][statename]/2,'LF':datasleep['ampLF'][statename]/datasleep['h0'][statename]/2,'VLF':datasleep['ampVLF'][statename]/datasleep['h0'][statename]/2}
amptable['cardiac']/=4 # must be reduced. Could use analytical law
amptable['cardiac']=0
amptable['resp']=0
amptable['LF']=0
amptable['VLF']=0
NREM=State(name='NREM',Rv=datasleep['Rv'][statename]*1e-4,h0=datasleep['h0'][statename]*1e-4,freqtable=freqtable,amptable=amptable)

statename='stageIS'
freqtable={'cardiac':1/datasleep['periodcard'][statename],'resp':1/datasleep['periodresp'][statename],'LF':1/datasleep['periodLF'][statename],'VLF':1/datasleep['periodVLF'][statename]}
amptable={'cardiac':datasleep['ampcard'][statename]/datasleep['h0'][statename]/2,'resp':datasleep['ampresp'][statename]/datasleep['h0'][statename]/2,'LF':datasleep['ampLF'][statename]/datasleep['h0'][statename]/2,'VLF':datasleep['ampVLF'][statename]/datasleep['h0'][statename]/2}
amptable['cardiac']/=4 # must be reduced. Could use analytical law
amptable['cardiac']=0
amptable['resp']=0
amptable['LF']=0
amptable['VLF']=0
IS=State(name='IS',Rv=datasleep['Rv'][statename]*1e-4,h0=datasleep['h0'][statename]*1e-4,freqtable=freqtable,amptable=amptable)

statename='stageREM'
freqtable={'cardiac':1/datasleep['periodcard'][statename],'resp':1/datasleep['periodresp'][statename],'LF':1/datasleep['periodLF'][statename],'VLF':1/datasleep['periodVLF'][statename]}
amptable={'cardiac':datasleep['ampcard'][statename]/datasleep['h0'][statename]/2,'resp':datasleep['ampresp'][statename]/datasleep['h0'][statename]/2,'LF':datasleep['ampLF'][statename]/datasleep['h0'][statename]/2,'VLF':datasleep['ampVLF'][statename]/datasleep['h0'][statename]/2}
amptable['cardiac']/=4 # must be reduced. Could use analytical law
amptable['cardiac']=0
amptable['resp']=0
amptable['LF']=0
amptable['VLF']=0
REM=State(name='REM',Rv=datasleep['Rv'][statename]*1e-4,h0=datasleep['h0'][statename]*1e-4,freqtable=freqtable,amptable=amptable)


#######
#create awake states :
# the data here are read in the files provided by the statistical analysis of the peak to peak linescan analysis.
analysis='wake_'


dataawake={}

dataawake['Rv']=ReadFixedEffect(folder+vessel+mouse+analysis+'lumen_'+'meanepisode_'+ftype)
dataawake['h0']=ReadFixedEffect(folder+vessel+mouse+analysis+'PVS_'+'meanepisode_'+ftype)

dataawake['ampcard']=ReadFixedEffect(folder+vessel+mouse+analysis+'PVS_'+'cardiac_amp_'+ftype)
dataawake['ampresp']=ReadFixedEffect(folder+vessel+mouse+analysis+'PVS_'+'resp_amp_'+ftype)
dataawake['ampLF']=ReadFixedEffect(folder+vessel+mouse+analysis+'PVS_'+'LF_amp_'+ftype)
dataawake['ampVLF']=ReadFixedEffect(folder+vessel+mouse+analysis+'PVS_'+'VLF_amp_'+ftype)

dataawake['periodcard']=ReadFixedEffect(folder+vessel+mouse+analysis+'PVS_'+'cardiac_period_'+ftype)
dataawake['periodresp']=ReadFixedEffect(folder+vessel+mouse+analysis+'PVS_'+'resp_period_'+ftype)
dataawake['periodLF']=ReadFixedEffect(folder+vessel+mouse+analysis+'PVS_'+'LF_period_'+ftype)
dataawake['periodVLF']=ReadFixedEffect(folder+vessel+mouse+analysis+'PVS_'+'VLF_period_'+ftype)



#We arbitrarily set the VLF component in whisking stage same as in the baseline because we couldnt measure it (too short episodes)
#dataawake['ampVLF']['stageWhisking']=dataawake['ampVLF']['baseline']
#dataawake['periodVLF']['stageWhisking']=dataawake['periodVLF']['baseline']

#Renormalisation to get the same baseline as sleeping states

for state in ['baseline','stageLocomotion','stageWhisking'] :
    for var in dataawake :
        dataawake[var][state]=dataawake[var][state]/dataawake[var]['baseline']*datasleep[var]['baseline']


statename='baseline'
freqtable={'cardiac':1/dataawake['periodcard'][statename],'resp':1/dataawake['periodresp'][statename],'LF':1/dataawake['periodLF'][statename],'VLF':1/dataawake['periodVLF'][statename]}
amptable={'cardiac':dataawake['ampcard'][statename]/dataawake['h0'][statename]/2,'resp':dataawake['ampresp'][statename]/dataawake['h0'][statename]/2,'LF':0.0,'VLF':0.0}
amptable['cardiac']/=4 # must be reduced. Could use analytical law
Quiet=State(name='QuietnoVLF',Rv=dataawake['Rv'][statename]*1e-4,h0=dataawake['h0'][statename]*1e-4,freqtable=freqtable,amptable=amptable)


statename='baseline'
freqtable={'cardiac':1/dataawake['periodcard'][statename],'resp':1/dataawake['periodresp'][statename],'LF':1/dataawake['periodLF'][statename],'VLF':1/dataawake['periodVLF'][statename]}
amptable={'cardiac':dataawake['ampcard'][statename]/dataawake['h0'][statename]/2,'resp':dataawake['ampresp'][statename]/dataawake['h0'][statename]/2,'LF':dataawake['ampLF'][statename]/dataawake['h0'][statename]/2,'VLF':dataawake['ampVLF'][statename]/dataawake['h0'][statename]/2}
amptable['cardiac']/=4 # must be reduced. Could use analytical law
Quiet=State(name='Quiet',Rv=dataawake['Rv'][statename]*1e-4,h0=dataawake['h0'][statename]*1e-4,freqtable=freqtable,amptable=amptable)


statename='stageLocomotion'
freqtable={'cardiac':1/dataawake['periodcard'][statename],'resp':1/dataawake['periodresp'][statename],'LF':1/dataawake['periodLF'][statename],'VLF':1/dataawake['periodVLF'][statename]}
amptable={'cardiac':dataawake['ampcard'][statename]/dataawake['h0'][statename]/2,'resp':dataawake['ampresp'][statename]/dataawake['h0'][statename]/2,'LF':dataawake['ampLF'][statename]/dataawake['h0'][statename]/2,'VLF':dataawake['ampVLF'][statename]/dataawake['h0'][statename]/2}
amptable['cardiac']/=4 # must be reduced. Could use analytical law
Locomotion=State(name='Locomotion',Rv=dataawake['Rv'][statename]*1e-4,h0=dataawake['h0'][statename]*1e-4,freqtable=freqtable,amptable=amptable)



statename='stageWhisking'
freqtable={'cardiac':1/dataawake['periodcard'][statename],'resp':1/dataawake['periodresp'][statename],'LF':1/dataawake['periodLF'][statename],'VLF':1/dataawake['periodVLF'][statename]}
amptable={'cardiac':dataawake['ampcard'][statename]/dataawake['h0'][statename]/2,'resp':dataawake['ampresp'][statename]/dataawake['h0'][statename]/2,'LF':dataawake['ampLF'][statename]/dataawake['h0'][statename]/2,'VLF':dataawake['ampVLF'][statename]/dataawake['h0'][statename]/2}
amptable['cardiac']/=4 # must be reduced. Could use analytical law
Whisking=State(name='Whisking',Rv=dataawake['Rv'][statename]*1e-4,h0=dataawake['h0'][statename]*1e-4,freqtable=freqtable,amptable=amptable)



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

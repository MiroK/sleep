import numpy as np

class State():
    def __init__(self, name,Rv, h0,freqtable,amptable):
        self.name=name
        self.Rv =Rv
        self.h0 = h0
        self.frequencies=freqtable
        self.amplitudes=amptable

class Cycle():
    def __init__(self,statelist, transitiontime):
        """
        statelist is a list of tuple (state,duration) with state a state object and duration a float
        """
        self.states,self.durations = list(zip(*statelist))
        self.transition=transitiontime
    def generatedata(self,nb_cycle):
        # create empty lists
        spantime=[]
        listspana={}
        listspanf={}
        for fb in self.states[0].amplitudes:
            listspana[fb]=[]
            listspanf[fb]=[]
        spanRv=[]
        spanh0=[]

        # set first time t=0
        time=0
        spantime.append(time)
        for fb in self.states[0].amplitudes:
            listspana[fb].append(self.states[0].amplitudes[fb])
            listspanf[fb].append(self.states[0].frequencies[fb])
        spanRv.append(self.states[0].Rv)
        spanh0.append(self.states[0].h0)

        # fill with the inner cycles
        for cycle in range(nb_cycle):
            for state,duration in zip(self.states,self.durations):
                spantime.append(time+self.transition/2)
                spantime.append(time+duration-self.transition/2)
                time+=duration
                # add two points with sames values to get a plateau for each stage
                for i in range(2) :
                    for fb in state.amplitudes:
                        listspana[fb].append(state.amplitudes[fb])
                        listspanf[fb].append(state.frequencies[fb])
                    spanRv.append(state.Rv)
                    spanh0.append(state.h0)
            
        # last point
        spantime.append(time)
        for fb in state.amplitudes:
            listspana[fb].append(state.amplitudes[fb])
            listspanf[fb].append(state.frequencies[fb])
        spanRv.append(state.Rv)
        spanh0.append(state.h0)

        spanRpvs=np.array(spanRv)+np.array(spanh0)

        return spantime,listspana,listspanf,spanRv,spanh0,spanRpvs
        

if __name__ == '__main__':

    #create several states
    Awake=State(Rv=6.4e-4,h0=2.7e-4,freqtable={'cardiac':10.1,'resp':3.03,'LF':0.476,'VLF':0.167},amptable={'cardiac':0.04,'resp':0.021,'LF':0.035,'VLF':0.047})
    REM=State(Rv=7.68e-4,h0=2.214e-4,freqtable={'cardiac':11.36,'resp':2.941,'LF':0.476,'VLF':0.130},amptable={'cardiac':0.042,'resp':0.024,'LF':0.027,'VLF':0.046})
    NREM=State(Rv=6.4e-4,h0=2.619e-4,freqtable={'cardiac':10,'resp':2.941,'LF':0.417,'VLF':0.147},amptable={'cardiac':0.037,'resp':0.019,'LF':0.064,'VLF':0.098})
    IS=State(Rv=7.04e-4,h0=2.511e-4,freqtable={'cardiac':10.1,'resp':2.941,'LF':0.435,'VLF':0.154},amptable={'cardiac':0.042,'resp':0.024,'LF':0.027,'VLF':0.046})

    Whisking=State(Rv=6.592e-4,h0=2.646e-4,freqtable={'cardiac':10.1,'resp':3.03,'LF':0.476,'VLF':0.167},amptable={'cardiac':0.04,'resp':0.021,'LF':0.035,'VLF':0.047})
    Locomotion=State(Rv=7.168e-4,h0=2.538e-4,freqtable={'cardiac':10.1,'resp':3.03,'LF':0.476,'VLF':0.167},amptable={'cardiac':0.04,'resp':0.021,'LF':0.035,'VLF':0.047})


    #create a cycle
    sleepcycle=Cycle([(NREM,50),(IS,40),(REM,110),(Awake,10)],transitiontime=2)
    awakecycle=Cycle([(NREM,50),(IS,40),(REM,110),(Awake,10)],transitiontime=2)
    NREMcycle=Cycle([(NREM,50),(Awake,10)],transitiontime=2)
    REMcycle=Cycle([(REM,110),(Awake,10)],transitiontime=2)

    spantime,listspana,listspanf,spanRv,spanh0,spanRpvs=sleepcycle.generatedata(2)

    print(spantime)
    print(listspanf)
    
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
    Awake=State(name='Awake',Rv=5.15e-4,h0=3.5e-4,freqtable={'cardiac':1/0.1,'resp':1/0.34,'LF':1/2.25,'VLF':1/6.11},amptable={'cardiac':2.87/100,'resp':1.71/100,'LF':4.31/100,'VLF':0.0})
    NREM=State(name='NREM',Rv=5.52e-4,h0=3.44e-4,freqtable={'cardiac':1/0.11,'resp':1/0.34,'LF':1/2.46,'VLF':1/6.7},amptable={'cardiac':2.02/100,'resp':1.5/100,'LF':6.88/100,'VLF':0.0})
    IS=State(name='IS',Rv=5.6e-4,h0=3.04e-4,freqtable={'cardiac':1/0.11,'resp':1/0.34,'LF':1/2.41,'VLF':1/6.37},amptable={'cardiac':2.07/100,'resp':1.6/100,'LF':6.71/100,'VLF':0.0})
    REM=State(name='REM',Rv=5.97e-4,h0=2.26e-4,freqtable={'cardiac':1/0.1,'resp':1/0.34,'LF':1/2.22,'VLF':1/7.75},amptable={'cardiac':2.73/100,'resp':1.78/100,'LF':2.21/100,'VLF':0.0})

    #create a cycle
    REMcycle=Cycle([(REM,110),(Awake,10)],transitiontime=2) # 2 times

    spantime,listspana,listspanf,spanRv,spanh0,spanRpvs=REMcycle.generatedata(2)

    print(spantime)
    print(spanh0)
    
import numpy as np



class State():
    """
    Define a state of awakenes or sleep as a mean state of the PVS and a supperposition of several oscillations of the lumen wall.

    input : 
    - name : name of the state
    - Rv : mean radius of the lumen
    - h0 : mean thickness of the PVS
    - freqtable : a table containing the frequencies of one or several oscillations freqtable={'osci1':frequency1, ...}
    - amptable : a table containing the amplitudes of one or seceral oscillations 

    freqtable and amptable must be of same size and use the same labels for the oscillations.
    """
    def __init__(self, name,Rv, h0,freqtable,amptable):
        self.name=name
        self.Rv =Rv
        self.h0 = h0
        self.frequencies=freqtable
        self.amplitudes=amptable

        print('Creation of %s stage'%self.name)
        print('Mean lumen radius : %.1e um'%self.Rv)
        print('Mean PVS thickness : %.1e um'%self.h0)
        print('Mean frequencies (Hz): ')
        print(self.frequencies)
        print('Mean amplitudes (ratio): ')
        print(self.amplitudes)
        print('\n')

        #to do : check consistancy between freqtable and amptable and return an error if not consistant

class Cycle():
    """
    Define a cycle as a succession of several states with a given transitiontime between two states.    
    """
    def __init__(self,statelist, transitiontime,name='cycle'):
        """
        Input : 
        - name of the cycle
        - statelist : a list of tuple (state,duration) with state a state object and duration a float
        - tansitiontime : time duration to be imposed between two states, float
        """
        self.name=name
        self.states,self.durations = list(zip(*statelist))
        self.transition=transitiontime
    def generatedata(self,nb_cycles):
        """
        Generate arrays describing the PVS geometry among several cycles.
        
        Input : 
        -nb_cycles : number of cycles to be generated

        Ouputs : 
        - spantime: time array covering the whole succession of cycles
        - listspana : list of amplitude arrays, each array decribe the evolution of the amplitude for one oscillation over time
        - listspanf :  list of frequency arrays, each array decribe the evolution of the frequency for one oscillation over time
        - spanRv : evolution of lumen radius over time
        - spanh0 : evolution of PVS thickness over time
        - spanRpvs :  evolution of the outer radius of the PVS over time
        """
        
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
        for cycle in range(nb_cycles):
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
    cycleobj=Cycle(name='REMcycle',statelist=[(Awake,10),(REM,110)],transitiontime=2) # 2 times

    spantime,listspana,listspanf,spanRv,spanh0,spanRpvs=cycleobj.generatedata(5)

    import matplotlib.pyplot as plt

    plt.plot(spantime,spanh0)
    plt.title(cycleobj.name)
    plt.xlabel('time (s)')
    plt.ylabel('PVS thickness (um)')

    plt.show()

    print(spantime)
    print(spanh0)
    
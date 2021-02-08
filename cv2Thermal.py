import seeed_mlx9064x
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import math
from scipy import ndimage
from skimage.restoration import estimate_sigma
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas

##-------------------------------------------------TWEAK GA---------------------------
genomeFlipChance = 0.35 #experimental, Prerequisite: FlipState = True
genomeFlipState = False #if true, genome flip chance is applied. if false, genome flip chance is not applied
genomeFlipNegativeChance = 0.4
genomeFlipPositiveChance = 0.6

removeStagantGenomes = False
stagnantGenome = 4

DisableStartGeneRate = 0.70

genomePerturbance = 0.90

genomeStepValue = 0.01
maxNeurons = 1000000

mutateInEnvironment = True
genomeEnvironmentMutationChance = 0.5
genomeMutationChance = 0.3

genomeActivationChance = 1
EnableGeneChance = 0.4
DisableGeneChance = 0.6

genomeLinkMutationChance = 3
genomeNodeMutationChance = 0.7

BreedChance = 0.90

stagnantSpecie = 15
BreedingGenomesIsRandom = True #True: elitism doesnt occur. allows more diverse genomes False: elitism occurs
BreedingSpeciesIsRandom = True #True: elitism doesnt occur. allows more diverse species False: elitism occurs
highScoreBased = True #breed/select species based on high score(True) or overall fitness(False) Prerequisite: BreedingSpeciesIsRandom = False

Steps = 2
Exposure = 3 #in seconds

#isAverageDuration = False # for each genomes exposure to environment, if set to true, genome is graded based on average score.
#if set to false, direct genome score is taken for the last timestep

PopulationSize = 50
NumberOfGenerations = 500

disjointmentConstant = 0.3
weightImportanceConstant = 0.84
excessGenesConstant = 0.5

speciesDistanceLimiter = 3

##------------------------------------------------------------------------------------------

##---------------------------THERMAL CAM SETUP----------------------------------------------
#mlx = seeed_mlx9064x.grove_mxl90641()
mlx = seeed_mlx9064x.grove_mxl90640()
mlx.refresh_rate = seeed_mlx9064x.RefreshRate.REFRESH_8_HZ  # The fastest for raspberry 4 
mlx_shape = (24,32)

mlx_interp_val = 10
mlx_interp_shape = (mlx_shape[0]*mlx_interp_val,
                    mlx_shape[1]*mlx_interp_val) #new shape(240 by 320)

fig = plt.figure(figsize=(4,3))#start fig??
ax = fig.add_subplot(111)#subplot?
fig.subplots_adjust(0,0,1,1) # get rid of unnecessary padding
#fig.subplots_adjust(0.002,0.0001,0.9,1)
#fig.subplots_adjust(0.002,0.02,1,1) # get rid of unnecessary padding
therm1 = ax.imshow(np.zeros(mlx_interp_shape),interpolation='none',
                   cmap=plt.cm.bwr,vmin=25,vmax=45) # preemptive image
fig.canvas.draw() # draw figure to copy background
ax_background = fig.canvas.copy_from_bbox(ax.bbox) # copy background

frame = np.zeros(mlx_shape[0]*mlx_shape[1])

def updateplot():
    fig.canvas.restore_region(ax_background) # restore background
    mlx.getFrame(frame) # read mlx90640
    data_array = np.fliplr(np.reshape(frame,mlx_shape)) # reshape, flip data
    data_array = ndimage.zoom(data_array,mlx_interp_val) # interpolate
    therm1.set_array(data_array) # set data
    therm1.set_clim(vmin=np.min(data_array),vmax=np.max(data_array)) # set bounds
    ax.draw_artist(therm1) # draw new thermal image
    fig.canvas.blit(ax.bbox) # draw background
    fig.canvas.flush_events() # show the new image
    return

t_array = []
##------------------------------------------------------------------------------------




##---------------ADJUST I/O SIZE--------------------------------------------------------
testInputs = np.array([])
testInputs = np.tile(0, 16384) #128 BY 128

testOutputs = np.array([])
testOutputs = np.tile(0,6)
##-------------------------------------------------------------------------------------


##----------------ACTIVATION FUNCTION--------------------------------------------------------
def sigmoid(gamma):
    gamma = -4.9 * gamma
    if gamma < 0:
        return 2.0 / (1.0 + math.exp(gamma)) - 1
    return 2.0 / (1.0 + math.exp(-gamma)) - 1
##-------------------------------------------------------------------------------------




##-----------------------------NEURON CLASS------------------------------------------
class Neuron:
    def __init__(this):
        this.geneIndex = [] #stores attatched genes
        this.status = None #2 for normal, 0 for input, 1 for output
        this.inputNumber = None #monitors respective to array onl for i/o
        this.value = None #neuron value
        
    def addGeneIndex(this,value):
        this.geneIndex.append(value)
    
    def setStatus(this,value):
        this.status = value
    
    def setValue(this,value):
        this.value = value
    
    def setInputNumber(this, value):
        this.inputNumber = value
    
    def getInputNumber(this):
        return this.inputNumber
        
    def getValue(this):
        return this.value
    
    def getStatus(this):
        return this.status
    
    def getGeneIndex(this):
        return this.geneIndex
##-------------------------------------------------------------------------------------


##--------------------------GENOME CLASS---------------------------------------------


class createNewGenome:
    def __init__(this):
        this.fitness = 0
        this.gene_dictionary = {}#key is innovation number
        this.neuron_dictionary = {}#key is neuron pos in network
        this.max_neuron = None #max number of neurons
        this.score = None
        this.maxInnovation = None
        this.stagnation = None
        this.linkMutationChance = genomeLinkMutationChance
        this.nodeMutationChance = genomeNodeMutationChance
        this.activationChance = genomeActivationChance
        this.flipSign = genomeFlipChance
        this.step = genomeStepValue
        this.perturbance = genomePerturbance
        this.mutationChance = genomeMutationChance
        this.environmentalMutation = genomeEnvironmentMutationChance
        this.enableGeneChance = EnableGeneChance
        this.disableGeneChance = DisableGeneChance
        this.flipNegativeChance = genomeFlipNegativeChance
        this.flipPositiveChance = genomeFlipPositiveChance
        
    def getFlipPositiveChance(this):
        return this.flipPositiveChance
    def setFlipPositiveChance(this,value):
        this.flipPositiveChance = value
    
    def getFlipNegativeChance(this):
        return this.flipNegativeChance
    def setFlipNegativeChance(this,value):
        this.flipNegativeChance = value
        
    
    def getEnableGeneChance(this):
        return this.enableGeneChance
    def setEnableGeneChance(this,value):
        this.enableGeneChance = value
        
    def getDisableGeneChance(this):
        return this.disableGeneChance
    def setDisableGeneChance(this,value):
        this.disableGeneChance = value
    
    def getEnvironmentalMutationChance(this):
        return this.environmentalMutation
    def setEnvironmentalMutationChance(this,value):
        this.environmentalMutation = value
    
    def getFlipChance(this):
        return this.flipSign
    
    def setFlipChance(this,value):
        this.flipSign = value
        
    def replaceGenomeGeneDictionary(this,dictionary):
        if type(dictionary)!=type(this.gene_dictionary):
            raise Exception("Replacement parameter is not a dictionary")
        else:
            this.gene_dictionary = dictionary
    
    def getGenomeGeneDictionary(this):
        return this.gene_dictionary
    
    def replaceGenomeNeuronDictionary(this, dictionary):
        if type(dictionary)!=type(this.neuron_dictionary):
            raise Exception("Replacement parameter is not a dictionary")
        else:
            this.neuron_dictionary = dictionary
        
    def getGenomeNeuronDictionary(this):
        return this.neuron_dictionary
    
    def getStagnation(this):
        return this.stagnation
    
    def setStagnation(this,value):
        this.stagnation = value
        
    def getFitness(this):
        return this.fitness
    
    def setFitness(this,value):
        this.fitness = value
    
    def getMaxNeuron(this):
        return this.max_neuron
    
    def setMaxNeuron(this,value):
        this.max_neuron = value
    
    def getMaxInnovation(this):
        return this.genome.maxInnovaion
    
    def setMaxInnovation(this,value):
        this.genome.maxInnovation = value
        
    def getScore(this):
        return this.score
    
    def setScore(this,value):
        this.score = value
    
    def getNeuron(this,key):
        return this.neuron_dictionary[key]
    
    def getGene(this,key):
        return this.gene_dictionary[key]
    
    def getLinkMutationChance(this):
        return this.linkMutationChance
    
    def setLinkMutationChance(this,value):
        this.linkMutationChance = value
        
    def getNodeMutationChance(this):
        return this.nodeMutationChance
    
    def setNodeMutationChance(this,value):
        this.nodeMutationChance = value
        
    def getActivationChance(this):
        return this.activationChance
    
    def setActivationChance(this,value):
        this.activationChance = value
        
    def getMutationChance(this):
        return this.mutationChance
    
    def setMutationChance(this,value):
        this.mutationChance = value
        
    
    
    
    def setMaxInnovation(this,value):
        this.maxInnovation = value
    
    def getMaxInnovation(this):
        return this.maxInnovation
    
    def addNeuron(this,key,neuron):
        this.neuron_dictionary[key] = neuron
    
    def addGene(this,key,gene):
        this.gene_dictionary[key] = gene


def copy_genome(genome):
    newGenome = createNewGenome()
    newGenome.replaceGenomeGeneDictionary(genome.getGenomeGeneDictionary())
    newGenome.replaceGenomeNeuronDictionary(genome.getGenomeNeuronDictionary())
    newGenome.setMaxInnovation(genome.getMaxInnovation())
    newGenome.setMaxNeuron(genome.getMaxNeuron())
    return newGenome
##-------------------------------------------------------------------------------------


##-----------------------------------CONNECTION GENE CLASS-----------------------
class connectionGene:
    def __init__(this):
        this.inputNeuron = None  #index
        this.outputNeuron = None #index
        this.weight = None
        this.enable = None
        this.innovation = None   #ancestry monitor
        
    def getInputNeuron(this):
        return this.inputNeuron
    def getOutputNeuron(this):
        return this.outputNeuron
    def getWeight(this):
        return this.weight
    def getEnableStatus(this):
        return this.enable
    def getInnovation(this):
        return this.innovation
    
    def setInnovation(this,value):
        this.innovation = value
    def setEnableStatus(this,value):
        this.enable = value
        
    def setWeight(this,value):
        this.weight = value
    def setInputNeuron(this,value):
        this.inputNeuron = value
    def setOutputNeuron(this,value):
        this.outputNeuron = value
        

def copy_gene(gene):
    copy_ = connectionGene()
    copy_.setWeight(gene.getWeight())
    copy_.setInputNeuron(gene.getInputNeuron())
    copy_.setEnableStatus(gene.getEnableStatus())
    copy_.setOutputNeuron(gene.getOutputNeuron())
    copy_.setInnovation(gene.getInnovation())
    return copy_
        
##------------------------------------------------------------------------------------


##---------------------------BUILD START NETWORK---------------------------------------
class buildStartingNetwork:
    def __init__(self,genome,length_inputs,length_outputs):
        self.genome = genome
        self.inputs = length_inputs
        self.outputs = length_outputs
    def createConnections(self):
        innovationCounter = 0
        neuron_key = 0
        for out_Number in range(self.outputs):
            newOutNeuron = Neuron()
            newOutNeuron.setStatus(1)
            newOutNeuron.setInputNumber(out_Number)
            self.genome.addNeuron(neuron_key,newOutNeuron)
            neuron_key+=1
        for i in range(self.inputs):
            newInNeuron = Neuron()
            newInNeuron.setStatus(0)
            newInNeuron.setInputNumber(i)
            for j in range(self.outputs):
                #each gene is connected to a neuron by a neuron_key
                newConnectionGene = connectionGene()
                newConnectionGene.setInputNeuron(neuron_key) #neuron_key_in
                newConnectionGene.setOutputNeuron(j)#neuron_key_out
                newConnectionGene.setInnovation(innovationCounter)
                newConnectionGene.setWeight(random() * 4.0 - 2.0)
                
                if DisableStartGeneRate > random():
                    newConnectionGene.setEnableStatus(False)
                else:
                    newConnectionGene.setEnableStatus(True)
                    
                newInNeuron.addGeneIndex(newConnectionGene.getInnovation())
                self.genome.neuron_dictionary[j].addGeneIndex(newConnectionGene.getInnovation())
                self.genome.addGene(innovationCounter,newConnectionGene)
                self.genome.setMaxInnovation(innovationCounter)
                innovationCounter+=1
            self.genome.addNeuron(neuron_key,newInNeuron)
            self.genome.setMaxNeuron(neuron_key)
            neuron_key+=1
        #print("NEURONS IN DICT: ", len(self.genome.neuron_dictionary))
        return self.genome
##-------------------------------------------------------------------------------------

##----------------------------CROSSOVER FUNCTIONS-------------------------------------
def childGenes_(genome1,genome2):
    childGenesDict = {}
    if genome1.getFitness() > genome2.getFitness():
        temp = genome2
        genome2 = genome1
        genome1 = temp
        
    for key in genome2.gene_dictionary:
        r = randrange(1,3)
        if key in genome1.gene_dictionary and r ==1:
            g = copy_gene(genome1.gene_dictionary[key])
            childGenesDict[key] = g
        else: 
            g = copy_gene(genome2.gene_dictionary[key])
            childGenesDict[key] = g
            
    return childGenesDict

def matchingGenesInGenome(genome1,genome2):
    averageWeightDifference = 0
    sum__ = 0
    matches = 0
    for key in genome2.gene_dictionary.keys():
        if key in genome1.gene_dictionary.keys():
            matches+=1
            sum__ = sum__ + abs(genome1.gene_dictionary[key].getWeight() - genome2.gene_dictionary[key].getWeight())

    #print("SUM: ",sum__," MATCHES: ",matches) 
    #if sum__ == 0.0:
        #print(len(genome1.getGenomeGeneDictionary()))
        #print(len(genome2.getGenomeGeneDictionary()))
        #raise Exception("Sum of 0 encountered")
    averageWeightDifference = sum__/matches
    return averageWeightDifference

##reworked for dict
def disjointGenesInGenome(genome1,genome2):
    disjoints = 0
    for key in genome2.gene_dictionary:
        if key in genome1.gene_dictionary:
            pass
        elif key < genome1.getMaxInnovation():
            disjoints+=1
            
    for key in genome2.neuron_dictionary:
        if key in genome1.neuron_dictionary:
            pass
        elif key < genome1.getMaxNeuron():
            disjoints+=1
    return disjoints

def excessGenesInGenome(genome1,genome2):
    excess = 0
    for key in genome2.gene_dictionary:
        if key in genome1.gene_dictionary:
            pass
        elif key > genome1.getMaxInnovation():
            excess+=1
            
    for key in genome2.neuron_dictionary:
        if key in genome1.neuron_dictionary:
            pass
        elif key > genome1.getMaxNeuron():
            excess+=1             
    return excess

def crossover(genome1,genome2):
    
    final_dict = {} 
    final_dict = childGenes_(genome1,genome2)
    
    childGenome = buildChildNetwork(final_dict,genome1,genome2)
    childGenome.gene_dictionary = final_dict
    
    genomeMaxNeuron = max(genome1.getMaxNeuron(),genome2.getMaxNeuron())
    if genome1.getFitness() > genome2.getFitness():
        temp = genome2
        genome2 = genome1
        genome1 = temp
        
    childGenome.setMaxNeuron(genomeMaxNeuron)   
    childGenome.setMaxInnovation(genome2.getMaxInnovation())
    childGenome.setLinkMutationChance(genome2.getLinkMutationChance())
    childGenome.setNodeMutationChance(genome2.getNodeMutationChance())
    childGenome.setMutationChance(genome2.getMutationChance())
    childGenome.setEnvironmentalMutationChance(genome2.getEnvironmentalMutationChance())
    
    childGenome.setActivationChance(genome2.getActivationChance())
    childGenome.setEnableGeneChance(genome2.getEnableGeneChance())
    childGenome.setDisableGeneChance(genome2.getDisableGeneChance())
    
    childGenome.setFlipChance(genome2.getFlipChance())
    childGenome.setFlipPositiveChance(genome2.getFlipPositiveChance())
    childGenome.setFlipNegativeChance(genome2.getFlipNegativeChance())
    return childGenome
##------------------------------------------------------------------------------------


##-------------MUTATION FUNCTIONS----------------------------------------------------
def activate_deactivate_mutation(genome,value,chance):
    while(chance > 0):
        if chance > random():
            candidates = {}
            for key in genome.gene_dictionary:
                if genome.gene_dictionary[key].getEnableStatus() == value:
                    candidates[key] = genome.gene_dictionary[key]

            if not candidates:
                break
        
            key, class__ = randchoice.choice(list(candidates.items())) 
            candidates[key].setEnableStatus(not candidates[key].getEnableStatus())
    
            genome.gene_dictionary.update(candidates)
        chance = chance - 1
        
def activate_deactivate_genes(genome):
    if randrange(1,3) == 1:
        genome.setEnableGeneChance(genome.getEnableGeneChance() * 0.95)
    else:
        genome.setEnableGeneChance(genome.getEnableGeneChance() * 1.05263)
    
    if randrange(1,3) == 1:
        genome.setDisableGeneChance(genome.getDisableGeneChance() * 0.95)
    else:
        genome.setDisableGeneChance(genome.getDisableGeneChance() * 1.05263)
        
    activate_deactivate_mutation(genome,False,genome.getEnableGeneChance())
    activate_deactivate_mutation(genome,True,genome.getDisableGeneChance())
    
    
def enable_disable_genes_candidacy(genome):
    if randrange(1,3) == 2:
        genome.setActivationChance(genome.getActivationChance() * 0.95)
    else:
        genome.setActivationChance(genome.getActivationChance() * 1.05263)
    
    p = genome.getActivationChance()
    while(p > 0):
        if p > random():
            activate_deactivate_genes(genome)
        p = p - 1
    return genome

def flip_sign_chance(genome):
    if genomeFlipState == True:
        if randrange(1,3) == 1:
            genome.setFlipChance(genome.getFlipChance() * 0.95)
        else:
            genome.setFlipChance(genome.getFlipChance() * 1.05263)
            
        p = genome.getFlipChance()
        while(p > 0):
            if p > random():
                flip_sign_mutation(genome)
            p = p - 1


def flip_sign_mutation(genome):
    if randrange(1,3) == 1:
        genome.setFlipPositiveChance(genome.getFlipPositiveChance() * 0.95)
    else:
        genome.setFlipPositiveChance(genome.getFlipPositiveChance() * 1.05263)
    
    if randrange(1,3) == 1:
        genome.setFlipNegativeChance(genome.getFlipNegativeChance() * 0.95)
    else:
        genome.setFlipNegativeChance(genome.getFlipNegativeChance() * 1.05263)
        
    flipGenomeSign(genome,genome.getFlipPositiveChance(),1)
    flipGenomeSign(genome,genome.getFlipNegativeChance(),0)
    
def flipGenomeSign(genome,chance,value):
    if value == 1:
        #print("tst p")
        while(chance > 0):
            #print("chk p chnc")
            if chance > random():
                candidates = {}
                for key in genome.gene_dictionary:
                    if genome.gene_dictionary[key].getWeight() < 0:
                        candidates[key] = genome.gene_dictionary[key]

                if not candidates:
                    #print("no candidates p")
                    break
        
                key, class__ = randchoice.choice(list(candidates.items())) 
                class__.setWeight(class__.getWeight() * -1)
                #print("p to n")
                genome.gene_dictionary.update(candidates)
            chance = chance - 1 
    else:
        #print("tst n")
        while(chance > 0):
            #print("chk n chnc")
            if chance > random():
                candidates = {}
                for key in genome.gene_dictionary:
                    if genome.gene_dictionary[key].getWeight() > 0:
                        candidates[key] = genome.gene_dictionary[key]

                if not candidates:
                    #print("no candidates n")
                    break
        
                key, class__ = randchoice.choice(list(candidates.items())) 
                class__.setWeight(class__.getWeight() * -1)
                #print("n to p")
                genome.gene_dictionary.update(candidates)
            chance = chance - 1
            
def pointMutate(genome):
    if randrange(1,3) == 2:
        genome.setMutationChance(genome.getMutationChance() * 0.95)
    else:
        genome.setMutationChance(genome.getMutationChance() * 1.05263)
        
    if genome.getMutationChance() > random():
        for innovation, class_ in genome.gene_dictionary.items():
            if  genome.perturbance > random():
                _number = class_.getWeight() + random() * genome.step*2.0 - genome.step
            else:
                _number = random() * 4.0 - 2.0
            class_.setWeight(_number)
    return genome

def randomNeuron(genome):
    r = randrange(1,3)
    #print("rn: ",r)
    if r == 1:
        neurons = {}
        for key,class__ in genome.neuron_dictionary.items():
            if class__.status == 0:
                neurons[key] = genome.neuron_dictionary[key]
        return randchoice.choice(list(neurons.items()))
    else:
        neurons = {}
        for key,class__ in genome.neuron_dictionary.items():
            if class__.status > 0:
                neurons[key] = genome.neuron_dictionary[key]
        return randchoice.choice(list(neurons.items()))
    
def mutateNodeGeneInGenome(genome):
    if randrange(1,3) == 2:
        genome.setNodeMutationChance(genome.getNodeMutationChance() * 0.95)
    else:
        genome.setNodeMutationChance(genome.getNodeMutationChance() * 1.05263)
        
    p = genome.getNodeMutationChance()
    while(p > 0):
        if p > random():
            maxGenomeInnovationNumber = genome.getMaxInnovation()
            maxNeuron = genome.getMaxNeuron()
            
            geneKey, class__ = randchoice.choice(list(genome.gene_dictionary.items())) 
            
            if class__.getEnableStatus() == False or len(genome.neuron_dictionary) > maxNeurons:
                return genome
            
            class__.setEnableStatus(False)
            genome.addGene(geneKey,class__)
            
            newConnection = connectionGene()
            newConnection.setEnableStatus(True)
            newConnection_1 = connectionGene()
            newConnection_1.setEnableStatus(True)
            maxNeuron +=1
            newNeuron = Neuron()
            newNeuron.setStatus(2)
            newNeuron.setValue(0.0)
            
            _in = genome.gene_dictionary[geneKey].getInputNeuron() #neuron index in
            _out = genome.gene_dictionary[geneKey].getOutputNeuron() #neuron index out
                    
            newConnection.setInputNeuron(_in) #neuron_key 1
            newConnection.setWeight(1)
            maxGenomeInnovationNumber+=1
            newConnection.setInnovation(maxGenomeInnovationNumber)
            newConnection.setOutputNeuron(maxNeuron)#neuron_key 2
            
            genome.neuron_dictionary[_in].addGeneIndex(newConnection.getInnovation())
            newNeuron.addGeneIndex(newConnection.getInnovation())
            
            if newConnection.getInputNeuron() == None:
                raise Exception("Input Neuron cannot be None")
            if newConnection.getOutputNeuron() == None:
                raise Exception("Output Neuron cannot be None")
                        
            genome.addGene(newConnection.getInnovation(),newConnection)
        
            newConnection_1.setInputNeuron(maxNeuron)
            newConnection_1.setWeight(genome.gene_dictionary[geneKey].getWeight())
            maxGenomeInnovationNumber+=1
            newConnection_1.setInnovation(maxGenomeInnovationNumber)
            newConnection_1.setOutputNeuron(_out)
            
            newNeuron.addGeneIndex(newConnection_1.getInnovation())
            genome.neuron_dictionary[_out].addGeneIndex(newConnection_1.getInnovation())
            
            if newConnection_1.getInputNeuron() == None:
                raise Exception("Input Neuron cannot be None")
            if newConnection_1.getOutputNeuron() == None:
                raise Exception("Output Neuron cannot be None")
            
            #add to dict
            genome.addGene(newConnection_1.getInnovation(),newConnection_1)
            genome.addNeuron(maxNeuron,newNeuron)
        #0----0(new)----0   
            #print("NN: ",newNeuron )
            genome.setMaxInnovation(maxGenomeInnovationNumber)
            genome.setMaxNeuron(maxNeuron)
            #print("NODE MUTATE")
        p = p - 1
    return genome


def mutateConnectionGeneInGenome(genome):
    if randrange(1,3) == 2:
        genome.setLinkMutationChance(genome.getLinkMutationChance() * 0.95)
    else:
        genome.setLinkMutationChance(genome.getLinkMutationChance() * 1.05263)
        
    p = genome.getLinkMutationChance()
    while(p > 0):
        if p > random():
            tries = 0
            while(tries < 5):
                maxGenomeInnovationNumber = genome.maxInnovation
                neuron_index_1, neuron_class__ = randomNeuron(genome)
                neuron_index_2, neuron_class__1 = randomNeuron(genome)
        
                arr1 = neuron_class__.getGeneIndex()
                arr2 = neuron_class__1.getGeneIndex()
                arr3 = []
                arr3 = set(arr1).intersection(arr2)
                if len(arr3) > 0:
                    pass
                elif neuron_class__.getStatus() == 0 and neuron_class__1.getStatus() == 0:
                    pass
                elif neuron_class__.getStatus() == 1 and neuron_class__1.getStatus() == 1:
                    pass
                else:
                    newConnection = connectionGene()
                    newConnection.setInnovation(maxGenomeInnovationNumber + 1)
                    newConnection.setWeight(random() * 4.0 - 2.0)
                    newConnection.setEnableStatus(True)
                    
                    if neuron_class__1.getStatus() == 0:
                        #nc1 goes in nc goes out
                        newConnection.setInputNeuron(neuron_index_2)
                        newConnection.setOutputNeuron(neuron_index_1)
                        genome.neuron_dictionary[neuron_index_1].addGeneIndex(newConnection.getInnovation())
                        genome.neuron_dictionary[neuron_index_2].addGeneIndex(newConnection.getInnovation())
                        
                    elif neuron_class__.getStatus() == 0:
                        newConnection.setInputNeuron(neuron_index_1)
                        newConnection.setOutputNeuron(neuron_index_2)
                        genome.neuron_dictionary[neuron_index_1].addGeneIndex(newConnection.getInnovation())
                        genome.neuron_dictionary[neuron_index_2].addGeneIndex(newConnection.getInnovation())
                    
                    elif neuron_class__.getStatus() == 2 and neuron_class__1.getStatus() == 2:
                        newConnection.setInputNeuron(neuron_index_1)
                        newConnection.setOutputNeuron(neuron_index_2)
                        genome.neuron_dictionary[neuron_index_1].addGeneIndex(newConnection.getInnovation())
                        genome.neuron_dictionary[neuron_index_2].addGeneIndex(newConnection.getInnovation())
                    
                    elif neuron_class__.getStatus() == 1:
                        newConnection.setInputNeuron(neuron_index_2)
                        newConnection.setOutputNeuron(neuron_index_1)
                        genome.neuron_dictionary[neuron_index_1].addGeneIndex(newConnection.getInnovation())
                        genome.neuron_dictionary[neuron_index_2].addGeneIndex(newConnection.getInnovation())
                    
                    elif neuron_class__1.getStatus() == 1:
                        newConnection.setInputNeuron(neuron_index_1)
                        newConnection.setOutputNeuron(neuron_index_2)
                        genome.neuron_dictionary[neuron_index_1].addGeneIndex(newConnection.getInnovation())
                        genome.neuron_dictionary[neuron_index_2].addGeneIndex(newConnection.getInnovation())
                        
                    genome.setMaxInnovation(newConnection.getInnovation())    
                    genome.addGene(newConnection.getInnovation(),newConnection)  
                        
                    if newConnection.getInputNeuron() == None:
                        raise Exception("Input Neuron cannot be None")
                    if newConnection.getOutputNeuron() == None:
                        raise Exception("Output Neuron cannot be None")
                    tries = 10
                if tries == 4:
                    pass
                tries+=1
        p = p - 1
    return genome

def mutateGenomeInPopulation(genome):
    genome = pointMutate(genome)
    genome = mutateNodeGeneInGenome(genome)
    genome = mutateConnectionGeneInGenome(genome)
    genome = enable_disable_genes_candidacy(genome)
    flip_sign_chance(genome)
    return genome

##-----------------------------------------------------------------------------------






##-------BUILD CHILD NETWORK-----------------------------------------------------------
def buildChildNetwork(genesDict,genome1,genome2):
    child_genome = createNewGenome()
    final_dict_neurons = {}
    
    final_dict_neurons.update(genome2.neuron_dictionary)
    final_dict_neurons.update(genome1.neuron_dictionary)
    
    for key in final_dict_neurons:
        final_dict_neurons[key].setValue(0.0)
          
    child_genome.neuron_dictionary = final_dict_neurons
    return child_genome

def resetNeuronsInGenome(genome):
    for key in genome.neuron_dictionary:
        genome.neuron_dictionary[key].setValue(0.0)

##------------------------------------------------------------------------------------
        
        
##-----------------------UPDATE NETWORK I/O----------------------------------------
def updateInputs(genome,flattened_array):
    index = 0
    for key in genome.neuron_dictionary:
        if genome.neuron_dictionary[key].getStatus() == 0 and index<=len(flattened_array) and genome.neuron_dictionary[key].getInputNumber() == index:
            genome.neuron_dictionary[key].setValue(flattened_array[index])
            index+=1
            
def obtainOutputs(genome):
    index = 0
    output_values = np.array([])
    for key in genome.neuron_dictionary:
        if genome.neuron_dictionary[key].getStatus() == 1 and index < len(testOutputs) and genome.neuron_dictionary[key].getInputNumber() == index:
            output_values = np.append(output_values,genome.neuron_dictionary[key].getValue())
            print("OUT: ",output_values[index]," INDEX: ",index)
            if index == 0 or index == 2 or index == 3:
                if output_values[index] > 180:
                    output_values[index] = 180
                elif output_values[index] < 0:
                    output_values[index] = 0
            else:
                if output_values[index] > 255:
                    output_values[index] = 255
                elif output_values[index] < 0:
                    output_values[index] = 0
                    
            index+=1
    return output_values
##-------------------------------------------------------------------------------------


##--------FEED FORWARD(eVALUATE GENOME)-------------------------------------------------


def evaluateGenome(genome):
    loop = 0
    while(loop < Steps):
        for neuron_number, neuron_class__ in genome.neuron_dictionary.items():
            sum__ = 0
            activation = 0
            isEvaluated = False
            for gene_innovation in neuron_class__.geneIndex:
                if gene_innovation in genome.gene_dictionary: 
                    if genome.gene_dictionary[gene_innovation].getEnableStatus() == True and neuron_class__.getStatus() > 0: 
                        if genome.gene_dictionary[gene_innovation].getOutputNeuron() == neuron_number:
                            input_number = genome.gene_dictionary[gene_innovation].getInputNeuron()
                            input___     = genome.neuron_dictionary[input_number].getValue()
                            weight       = genome.gene_dictionary[gene_innovation].getWeight()
                            sum__        = sum__ + (input___* weight)
                            isEvaluated  = True
            if isEvaluated == True:
                if neuron_class__.getStatus() == 2:
                    neuron_class__.setValue(sigmoid(sum__))
                elif neuron_class__.getStatus() == 1:
                    #print("sum: ",sum__)
                    neuron_class__.setValue(sum__) 
        loop+=1

##-----------------------------------------------------------------------


##------------------------------HSV SETUP----------------------------------------------------
def hsv_color_space(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    l_h = cv2.getTrackbarPos("L-H","HSV1 BARS")
    l_s = cv2.getTrackbarPos("L-S","HSV1 BARS")
    l_v = cv2.getTrackbarPos("L-V","HSV1 BARS")
    u_h = cv2.getTrackbarPos("U-H","HSV1 BARS")
    u_s = cv2.getTrackbarPos("U-S","HSV1 BARS")
    u_v = cv2.getTrackbarPos("U-V","HSV1 BARS")
    lower_color = np.array([l_h,l_s,l_v])
    upper_color= np.array([u_h,u_s,u_v])
    mask = cv2.inRange(hsv,lower_color,upper_color)
    return mask

def houghLines(img):
    #threshold1 = cv2.getTrackbarPos("T1","HOUGH1 BARS")
    #threshold2 = cv2.getTrackbarPos("T2","HOUGH1 BARS")
    edges = cv2.Canny(img,100,100)
    return edges

def nothing(X):
    pass

def createHSVwindow():
    cv2.namedWindow("HSV1 BARS")
    cv2.createTrackbar("L-H","HSV1 BARS",0,180, nothing)
    cv2.createTrackbar("L-S","HSV1 BARS",0,255, nothing)
    cv2.createTrackbar("L-V","HSV1 BARS",180,180, nothing)
    cv2.createTrackbar("U-H","HSV1 BARS",180,180, nothing)
    cv2.createTrackbar("U-S","HSV1 BARS",255,255, nothing)
    cv2.createTrackbar("U-V","HSV1 BARS",255,255, nothing)
##-------------------------------------------------------------------------------------------------------
    
##--score estimator(s)
def estimate_noise(image_path):
    #img = cv2.imread(image_path)
    return estimate_sigma(image_path, multichannel=False, average_sigmas=True)

def est(img):
    return cv2.Laplacian(img,cv2.CV_64F).var()
##---

createHSVwindow()
while True:
    t1 = time.monotonic() # for determining frame rate
    try:
        updateplot() # update plot
        canvas = FigureCanvas(fig)
        plt.axis('off')
        plt.axis('tight')
        plt.axis('image')
        canvas.draw()
        graph_image = np.array(fig.canvas.get_renderer()._renderer)
        graph_image = hsv_color_space(graph_image)
        graph_image = houghLines(graph_image)
        score = estimate_noise(graph_image)
        if math.isnan(score) == True:
            score = 0.0
        #graph_image = crop_image(graph_image, pixel_value=255)
        #print("shape: ",graph_image.shape)
        print("noise level",score)
        cv2.imshow("graph image",graph_image)
    except:
        continue
    # approximating frame rate
    t_array.append(time.monotonic()-t1)
    if len(t_array)>10:
        t_array = t_array[1:] # recent times for frame rate approx
    print('Frame Rate: {0:2.1f}fps'.format(len(t_array)/np.sum(t_array)))



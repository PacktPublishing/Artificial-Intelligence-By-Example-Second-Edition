# -*- coding: utf-8 -*-

# Genetic Algorithm
#Genetic Algorithm example 
#Copyright Denis Rothman, MIT LICENSE

""" ### Importing libraries """
import math
import random
import datetime

"""### Generating a Parent"""

def gen_parent(length):
  genes=[]                        #genes array
  while len(genes)<length:        #genes is constrained to the length
      #sampleSize: length of target constraint 
      sampleSize=min(length-len(genes),len(geneSet))
      #extend genes with a random sample the size of sampleSize extracted from geneSet
      genes.extend(random.sample(geneSet,sampleSize))
  return ''.join(genes)

"""## Fitness"""

#Fitness function
def get_fitness(this_choice,scenario):
  if(scenario==1):
    fitness=sum(1 for expected,actual in zip(target,this_choice) if expected==actual)
  if(scenario==0):
    cc=list(this_choice) # cc= this choice
    gs=list(geneSet)     # gene set
    cv=list(KPIset)      # value of each KPI in the set
    fitness=0
    for op1 in range(0,len(geneSet)): #2.first find parent gene in gene set
      for op in range(0,len(target)):
        if cc[op]==gs[op1]:             #3.gene identified in gene set
          vc=int(cv[op1])               #4.value of critical path constraint
          fitness+=vc
      for op in range(0,len(target)):
        for op1 in range(0,len(target)):
          if op!=op1 and cc[op]==cc[op1]:
            fitness=0                     # no repetitions allowed, mutation enforcement
  return fitness

"""###  Crossover and Mutate"""

def crossover(parent):
      index=random.randrange(0,len(parent))#producing a random position of the parent gene
      childGenes=list(parent)
      oldGene=childGenes[index]        # for diversity check
      newGene,alternate=random.sample(geneSet,2)
      if(newGene!=oldGene):childGenes[index]=newGene;            #natural crossover
      if(newGene==oldGene):childGenes[index]=alternate;          #mutation introduced to ensure diversity to avoid to get stuck in a local minima
      return ''.join(childGenes)

"""### Display Selection"""

#Display selection
def display(selection,bestFitness,childFitness,startTime):
      timeDiff=datetime.datetime.now()-startTime
      #when the first generation parent is displayed childFitness=bestFitness=parent Fitness 
      print("Selection:",selection,"Fittest:",bestFitness,"This generation Fitness:",childFitness,"Time Difference:",timeDiff)

"""### Main"""

def ga_main():
#I PARENT GENERATION
       startTime=datetime.datetime.now() 
       print("starttime",startTime)
       alphaParent=gen_parent(len(target))
       bestFitness=get_fitness(alphaParent,scenario)
       display(alphaParent,bestFitness,bestFitness,startTime) 
#II. SUBSEQUENT GENERATIONS
       #producing the next generations
       g=0
       bestParent=alphaParent              # at the beginning, the first parent, best or not is our best available choice
       while True:
        g+=1
        child=crossover(bestParent)        #mutation
        childFitness=get_fitness(child,scenario) #number of correct genes
        if bestFitness>=childFitness:   #
                   continue
        display(child,bestFitness,childFitness,startTime)
        bestFitness=childFitness
        bestParent=child
        if scenario==1: goal=len(alphaParent);#number of good genes=parent length
        if scenario==0: goal=threshold;
        if childFitness>=goal:
                   break

#III. SUMMARY
       print("Summary---------------------------------------------------")
       endTime=datetime.datetime.now()
       print("endtime",endTime)
       print("geneSet:",geneSet);print("target:",target)
       print("geneSet length:",len(geneSet))
       print("target length:",len(target))
       print("generations:",g)
       print("Note: the process is stochastic so the number of generations will vary")

"""### Calling the Algorithm"""

print("Genetic Algorithm")

scenario=0   # 1=target provided at start, 0=no target, genetic optimizer
GA=2

#geneSet for all scenarios, other sets for GA==2
geneSet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!.-"   #gene set
KPIset ="0123456772012345674701234569980923456767012345671001234"   #KPI set
threshold=35

#Target 01 with predefined target sequence
#set of genes
if(GA==1):
  geneSet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!.-"
  # target with no space unless specified as a character in the geneSet
  target="Algorithm"  # No space unless specified as a character in the geneSet
  print("geneSet:",geneSet,"\n","target:",target)
  ga_main()

#Target 02 with optimizing values, no target sequence but a KPI to attain
#A coded algorithm sequence: each letter represents a start state for an MDP OR SCM Process: production, services, deliveries or other
 
if(scenario==0 and GA==2):
  target="AAAA"                                                   #unspecified target
  print("geneSet:",geneSet,"\n","target:",target)
  ga_main()

if(scenario==1 and GA==3):
  target="FBDC"  # No space unless specified as a character in the geneSet
  print("geneSet:",geneSet,"\n","target:",target)
  ga_main()

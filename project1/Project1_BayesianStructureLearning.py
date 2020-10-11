import numpy as np
from scipy.special import loggamma
import pandas as pd
import os
import itertools
import networkx as nx

class BayesianStructureLearning:

    def __init__(self, inputCSV, outputGraph):
        self.inputCSV = os.path.abspath(inputCSV)
        self.outputGraph = outputGraph
    
    def importCSV(self):
        contents = pd.read_csv(self.inputCSV)
        return contents.columns, contents.to_numpy()
    
    def buildNoEdgeGraph(self, nodeNames):
        graph = nx.DiGraph()
        nodes = range(len(nodeNames))
        graph.add_nodes_from(nodes)
        return graph
    
    def graphSearch(self, data):        
        print("Graph Search Complete\n")
    
    def indexParentalInstantiation(self, numValues, varParents, parentSample):
        index = 0
        varParentValues = [range(1, numValues[parent]+1) for parent in varParents]
        instants = list(itertools.product(*varParentValues))        
        for currentInstant in instants:
            if np.all(parentSample == currentInstant):
                return index
            index+=1
    
    def graphData(self, data, graph):
        variables = list(graph.nodes())
        parents = [list(graph.predecessors(var)) for var in variables]
        numValues = np.amax(data, axis=0)
        numParentalInstants = np.array([np.prod([numValues[parent] for parent in parents[var]]) for var in variables])
        m = [np.zeros((int(numParentalInstants[var]), int(numValues[var]))) for var in variables]
        for sample in data:
            for var in variables:
                value = sample[var]-1
                instantiation = 0
                if len(parents[var]) != 0:
                    instantiation = self.indexParentalInstantiation(numValues, parents[var], sample[parents[var]])
                m[var][instantiation, value] += 1
        
        return m
    
    def scoreGraph(self, data, graph):
        m = self.graphData(data, graph)
        variables = list(graph.nodes())
        score = 0
        alpha = [np.ones_like(m[var]) for var in variables]
        for var in variables:
            p = np.sum(loggamma(alpha[var]+m[var]))
            p -= np.sum(loggamma(alpha[var]))
            p+= np.sum(loggamma(np.sum(alpha[var])))
            p-= np.sum(loggamma(np.sum(alpha[var]) + np.sum(m[var])))
            score += p
        
        return score #np.sum(p)
        
        
        print("Scoring Complete\n")
    
    def writeFile_gph(self):
        print("Write Graph Complete\n")

    def solve(self):
        while True:
            variableNames, data = self.importCSV()
            graph = self.graphSearch(data)
            score = self.scoreGraph(graph)
            out = self.writeGraph()
            print("Solve Complete\n")
            break

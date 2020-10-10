import numpy as np
from scipy.special import loggamma
import pandas as pd
import os

class BayesianStructureLearning:

    def __init__(self, inputCSV, outputGraph):
        self.inputCSV = os.path.abspath(inputCSV)
        self.outputGraph = outputGraph
    
    def importCSV(self):
        contents = pd.read_csv(self.inputCSV)
        return contents.columns, contents.to_numpy()

    def graphSearch(self, data):        
        print("Graph Search Complete\n")
        
    def graphData(self, data, graph):
        variables = list(graph.nodes())
        parents = [list(graph.predecessors(var)) for var in variables]
        numValues = np.amax(data, axis=0)
        numParentalInstants = [np.prod([numValues[parent] for parent in parents[var]] for var in variables]
        m = np.array([np.zeros((parents[var], numValues[var])) for var in variables])
        for sample in data:
            for var in variables:
                value = sample[var]
                instantiation = 0
                if !isempty(parents[var]):
                    #j = <index for the parental instantiation>
                m[var][instantiation, value]
         return m
    
    def scoreGraph(self, data, graph):
        m = graphData(data, graph)
        alpha = 1
        loggamma(
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

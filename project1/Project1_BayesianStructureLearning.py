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
        graph.add_nodes_from(range(len(nodeNames)))
        return graph
    
    def isAcyclic(self, graph):
        return nx.is_directed_acyclic_graph(graph)
    
    def graphSearch(self, nodeNames, data, method):
        if method == 'K2':
            graph = self.buildNoEdgeGraph(nodeNames)
            orderedVariables = list(graph.nodes)
            np.random.shuffle(orderedVariables)
            for (k, i) in list(enumerate(orderedVariables[1:])):
                score = self.scoreGraph(data, graph)
                while True:
                    score_best, j_best = -1000000, 0
                    for j in orderedVariables[0:k]:
                        if not(graph.has_edge(j, i)):
                            graph.add_edge(j, i)
                            score_new = self.scoreGraph(data, graph)
                            if score_new > score_best:
                                score_best, j_best = score_new, j
                            graph.remove_edge(j, i)
                    if score_best > score and self.isAcyclic(graph):
                        score = score_best
                        graph.add_edge(j_best, i)
                    else:
                        break
            return graph, score
    
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
        numParentalInstants = [len(m[i][:,0]) for i in range(len(variables))]
        score = 0
        alpha = [np.ones_like(m[var]) for var in variables]
        for var in variables:
            for instant in range(numParentalInstants[var]):
                p = np.sum(loggamma(alpha[var][instant,:]+m[var][instant,:]))
                p -= np.sum(loggamma(alpha[var][instant,:]))
                p+= np.sum(loggamma(np.sum(alpha[var][instant,:])))
                p-= np.sum(loggamma(np.sum(alpha[var][instant,:]) + np.sum(m[var][instant,:])))
                score += p
        
        return score
    
    def writeFile_gph(self, nodeNames, graph):
        with open(self.outputGraph, 'w') as f:
            for edge in graph.edges():
                f.write("{}, {}\n".format(nodeNames[edge[0]], nodeNames[edge[1]]))
        f.close()
        print("Write Graph Complete\n")

    def solve(self):
        while True:
            variableNames, data = self.importCSV()
            graph, score = self.graphSearch(variableNames, data, 'K2')
            self.writeFile_gph(variableNames, graph)
            print("Solve Complete with Score:")
            print score
            break
            
    def solve_timed(self):
        import time
        start = time.time()
        self.solve()
        end = time.time()
        print("\nRuntime (s):")
        print end-start

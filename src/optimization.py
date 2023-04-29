"""

This script includes all functions to build and solve the QUBO model.
It requires the modules ``Hardware``, ``InteractionGraph``, ``pyqubo`` and ``neal``. Itself is importet to the file **main.py** as a module.
"""
from pyqubo import Binary
from input import IGraph, CoreSystem

# solver settings: choose from list
#from dwave_qbsolv import QBSolv
#from tabu import TabuSampler
from neal import SimulatedAnnealingSampler

import networkx as nx
import numpy as np
import multiprocessing as mp

import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')

import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Helvetica"],
    "font.size": 16
})

class QUBO:
    """
    This class contains the quadratic unconstrained binary optimization (QUBO) formulation of the mapping problem.
    The method ``build`` builds the model and ``solve``s it with the `neal.SimulatedAnnealingSampeler()` method.

    Args:
        igraph (IGraph): interaction graph declared in module `InteractionGraph`
        cgraph (CoreSystem): multi-core graph declared in module `Hardware`
        start (int): index start slice
        end (int): index last slice
        verbose (bool): default is True, provides additional details on progress
    Attributes:
        slices (list): list of time slice ``networkx.Graph()`` objects
        nsclices (int): number of slices
        laplacians (list): list of Graph Laplacian matrices for all time-slice interaction graphs
        d (nparray): hop-distance matrix between cores
        """

    def __init__(self, igraph:IGraph, cgraph:CoreSystem, start:int, end:int, fact=1, verbose=False):
        
        # sliced quantum circuit and Laplacians
        self.igraph = igraph
        self.slices = igraph.slices

        self.start = start
        self.end = end

        self.nslices = len(self.slices[start:end]) # number of slices
        self.laplacians = igraph.get_laplacians()

        # multi-core system
        self.cgraph = cgraph
        #self.d = cgraph.dgridmatrix #2x5 grid
        self.d = cgraph.dmatrix #all to all

        # problem size
        self.nxvars = self.nslices * self.cgraph.ncores * self.igraph.qc.num_qubits
        self.nyvars = self.nslices * self.cgraph.ncores * self.cgraph.capacity
        self.N = self.nxvars + self.nyvars

        # weighting parameter
        if self.nslices > 0:
            self.fact = fact
            self.lam = self.fact/(self.nslices * self.igraph.qc.num_qubits)
        
        # define model for later
        self.model = None

        self.verbose = verbose
        if verbose:
            print("%d-qubit register circuit is split into %d slices" % (self.igraph.qc.num_qubits, self.nslices))
            print("%d cores of capacity %d" % (self.cgraph.ncores, self.cgraph.capacity))
        

    def build(self, lam = None):
        """ Build and compile the quadratic model to solve the qubit mapping problem to multi-core quantum systems.

        :param lam: weighting parameter >= 0
        :return: quadratic model, lam
        :rtype: cpp_pyqubo.Model, int
        """

        if lam is not None: # set lambda if given
            self.lam = lam

        # set up the binary solution variables
        x = {}
        y = {}
        for t in range(self.start, self.end):  
            for j in range(self.cgraph.ncores):
                x[(t,j)] = [ # solution variables
                    Binary('x_'+str(t)+'_'+str(j)+'-'+str(i)) for i in self.igraph.nodes
                    ] # x = [(slicenr, j)][i] where i refers to a logical qubit and j referes to a core
        
                y[(t,j)] = [ # slack variables
                    Binary('y_'+str(t)+'_'+str(j)+'-'+str(idash)) for idash in range(self.cgraph.capacity)
                    ] # y = [(slicenr, j)][i'] where i' refers to a physical qubit and j referes to a core

        if self.verbose:
            print("number of decision variables: %d + %d " % (self.nxvars,self.nyvars))

        # part 1: assignments
        assignment_model = 0
        nodes = self.igraph.nodes

        for t in range(self.start, self.end):

            L = self.laplacians[t] # graph laplacian of current slice

            # 1. map from interaction graph to the multi-core system
            # 2. such that no gate spans accross cores (cut edge)
            # 3. and the assignment adheres to the cores capacity
            term_assignment = sum( 
                [( sum([x[(t,j)][i] for j in range(self.cgraph.ncores)]) - 1 )**2 for i in self.igraph.nodes]
            )

            term_laplacian = sum([ 
                sum( L[i,i]*x[(t,j)][i] for i in nodes) + 2 * sum( sum([x[(t,j)][i] * L[i,k] * x[(t,j)][k] for i in list(nodes)[k+1:]]) for k in list(nodes)[:-1] ) for j in range(self.cgraph.ncores)
                ])
            

            term_capacity = sum(
                [( sum(x[(t,j)]) - sum(y[(t,j)]) )**2 for j in range(self.cgraph.ncores)]
            )


            assignment_model += term_assignment + term_laplacian + term_capacity

        # part 2: penalize movements
        term_movements = 0

        for t in range(self.start, self.end-1):
            # The distance of each physical qubit which is allocated by the mapping is an estimate for the number of swaps needed in the output circuit
            term_movements += sum(
                                        [sum(
                                            [sum([self.d[j][l]*x[(t,j)][i]*x[(t+1,l)][i] for j in range(self.cgraph.ncores)]) for l in range(self.cgraph.ncores)
                                            ]) for i in nodes
                                        ]
                                        )


        if self.verbose:
            print("Building successful.")
            print("Compile model ..")
        
        # compile the quadratic model
        if self.verbose:
            print('lam: %f' % (self.lam))
        self.model = (assignment_model + self.lam*term_movements).compile()

        if self.verbose:
            print("Compilation successful.")
        
        return self.model, self.lam

    def solve(self):
        """ Solve the optimization problem with simulated annealing solver of the library ``neal`` and generate an element of the child class ``qubosolution``.

        :return: solution to the problem 
        :rtype: ``qubosolution``
        """
        # set parameters for solver heuristic
        self.nsweeps = 1e6 # 1e5 used for benchmark results in Bandic et al. 2023
        self.nreads = 1
        
        solver = SimulatedAnnealingSampler()
        
        if self.verbose:
            print("Start solving ..")
        sampleset = solver.sample(self.model.to_bqm(), num_sweeps=self.nsweeps, num_reads=self.nreads, seed=42)  
        
        decoded_samples = self.model.decode_sampleset(sampleset)
        self.mapping = min(decoded_samples, key=lambda x: x.energy)
        
        if self.verbose:
            print("Potential solution found.")
        
        self.solution = qubosolution(self.mapping, self.igraph, self.cgraph, self.start, self.end, verbose=False)

        return self.solution


    def isvalid(self):
        """ Check if the solution is a valid assignment.

        :return: isvalid 
        :rtype: bool
        """

        nodes = self.igraph.nodes
        x = self.mapping.sample
        assignment_value = 0
        for t in range(self.start, self.end):
            L = self.laplacians[t] # laplacian of current slice

            # 1. map from interaction graph to the multi-core system
            # 2. such that no gate is spanned accross cores
            # 3. constrain to the cores capacity
            term_assignment = sum( 
                [( sum([x['x_'+str(t)+'_'+str(j)+'-'+str(i)] for j in range(self.cgraph.ncores)]) - 1 )**2 for i in nodes]
            )

            term_laplacian = sum([ 
                sum( L[i,i]*x['x_'+str(t)+'_'+str(j)+'-'+str(i)] for i in nodes) + 2 * sum( sum([x['x_'+str(t)+'_'+str(j)+'-'+str(i)] * L[i,k] * x['x_'+str(t)+'_'+str(j)+'-'+str(k)] for i in list(nodes)[k+1:]]) for k in list(nodes)[:-1] ) for j in range(self.cgraph.ncores)
                ])
            

            term_capacity = sum(
                [( sum(x['x_'+str(t)+'_'+str(j)+'-'+str(i)] for i in nodes) - sum(x['y_'+str(t)+'_'+str(j)+'-'+str(idash)] for idash in range(self.cgraph.capacity)) )**2 for j in range(self.cgraph.ncores)]
            )

            assignment_value += term_assignment + term_laplacian + term_capacity
        
        valid = assignment_value == 0
        return valid
    
    def divide_conquer(self, mapping, isvalid, start=0):
        """ Use divide-and-conquer if problem comprises more than 50k variables.

        """
        if self.N == 0: # subproblem is empty --> return
            return
        if self.N < 5e4: # subproblem is smaller than X --> solve, check validity, collect solution array and return
            self.build()
            self.solve()
            v = self.isvalid()
            isvalid.append(v)
            mapping.update(self.mapping.sample)       
            return

        mid = self.nslices // 2 # subproblem is too large --> divide 
        end = self.nslices

        print('left: %d, %d' % (start, start+mid)) # check current state
        # generate new subproblem from left half 
        left_qubo = QUBO(self.igraph, self.cgraph, start=start, end=start+mid, fact=self.fact)

        print('right: %d, %d' % (start+mid, start+end)) #check current state
        # generate new subproblem from right half
        right_qubo = QUBO(self.igraph, self.cgraph, start=start+mid, end=start+end, fact=self.fact)

        # call fun again to solve 
        left_qubo.divide_conquer(mapping, isvalid, start=start)
        right_qubo.divide_conquer(mapping, isvalid, start=start+mid)
    

    # def calc_movements(self):
    #     nmoves = self.mapping.energy*1/(self.lam)
    #     return nmoves
    
    def count_movements(self, x=None):
        """ Count inter-core communications of solution (= movements of qubit states between cores).

        """
        # count movements with solution array 
        if x is None: 
            x = self.mapping.sample
        
        term_movements = 0

        for t in range(self.start, self.end-1):
            # The distance of each physical qubit which is allocated by the mapping is an estimate for the number of swaps needed in the output circuit
            term_movements += sum(
                                        [sum(
                                            [sum([self.d[j][l]*x['x_'+str(t)+'_'+str(j)+'-'+str(i)]*x['x_'+str(t+1)+'_'+str(l)+'-'+str(i)] for j in range(self.cgraph.ncores)]) for l in range(self.cgraph.ncores)
                                            ]) for i in self.igraph.nodes
                                        ]
                                        )
            
        return term_movements
    
    def checkMapping(self):
        """ Test correctness of cost function.
        """
        # should reveal potential implementation errors of the cost function - case: isvalid() is True (bool1) but second part bool2 is False
        # further, bool2 gives insights on misplacements e.g. if isvalid() is False because of misplacements, it prints the affected gates
        bool1 = self.isvalid() 
        bool2 = 0
        for t in range(self.start, self.end):
            for c in range(self.cgraph.ncores):
                nodes = [int(key.split('-')[-1]) for (key,val) in self.mapping.sample.items() if val == 1 and  key.startswith('x_%d_%d' % (t,c))]
                ng = nx.Graph()
                ng.add_nodes_from(nodes)
                
                for edge in self.slices[t]:
                    if (edge[0] in ng.nodes) and (edge[1] not in ng.nodes):
                        print('not valid: slice %d, core %d' % (t,c), edge)  
                        bool2 = 1 
        
        return (bool1 and bool2)
    
    def draw_mapping(self, x=None):
        """ Returns plot of mapping solution (recommened only for small instance sizes up to 100 x-vars)
        """
        
        if x is None:
            x = self.mapping.sample

        options = {
            "font_size": 30,
            "node_size": 1000,
            "edgecolors": "black",
            "linewidths": 0.5,
            "width": 5,
        }

        fig, axs = plt.subplots(self.cgraph.ncores, self.nslices, figsize=(25,10), sharey=True)
        for t in range(self.start, self.end):
            nodes_per_core = []
            for c in range(self.cgraph.ncores):
                nodes = [key.split('-')[-1] for (key,val) in x.items() if val == 1 and  key.startswith('x_%d_%d' % (t,c))]
                ng = nx.Graph()
                ng.add_nodes_from(nodes)

                nodes_per_core.append(nodes)
                # explicitly set positions
                
                pos = dict()
                for i,node in enumerate(nodes):
                    pos[node] = (np.random.uniform(-2,2), np.random.uniform(-2,2))
                
                options["node_color"] = ["white"] * len(nodes)
                if t>0:
                    for i,node in enumerate(nodes):
                        if node not in nodes_prev[c]:
                            options["node_color"][i] = "pink"
                nx.draw_networkx(ng, pos, **options, ax = axs[c,t] )

                axs[c,t].margins(0.7)
                axs[c,t].grid(False)
                axs[c,0].set_ylabel(r"core %d" % (c+1),  fontsize = 30.0)
                axs[0,t].set_title(r"t = %d" % (t+1),  fontsize = 30.0)
            
            nodes_prev = nodes_per_core

        plt.show()

class qubosolution(QUBO):

    def __init__(self, mapping, igraph:IGraph, cgraph:CoreSystem, start:int, end:int, verbose=False):
        super().__init__(igraph, cgraph, start, end, verbose)
        self.mapping = mapping

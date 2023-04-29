"""
This module comprises two class objects, `CGraph`and `IGraph` representing the core-system configuration and Quantum Circuit. Both objects serves as inputs to the QUBO (Optimization.py)
"""

import re
import numpy as np
import networkx as nx
import glob
from qiskit import QuantumCircuit

class CoreSystem:
    """ This class contains all information of the multi-core layout.

    Args:
        ncores (int): number of cores
        capacity (int): number of qubit-capacity of one core       
    Attributes:
        ncores (int): number of cores
        capacity (int): number of qubit-capacity of one core
        dmatrix (nparray): shortest distance matrix in all-to-all layout
        dgridmatrix (dict): shortest distance matrix in 2x5 grid layout
    """

    def __init__(self, ncores, capacity):      
        # extract relevant properties 
        self.ncores =  ncores
        self.capacity = capacity
        self.dmatrix =  np.ones((self.ncores, self.ncores))-np.eye(self.ncores) # all to all multi-core system (each core has a linking edge to all other cores)
        G = nx.convert_node_labels_to_integers(nx.grid_2d_graph(5,2))  # grid multi-core system (2-4 nearest neighbour connections)
        self.dgridmatrix = dict(nx.all_pairs_shortest_path_length(G)) 

class IGraph:
    """This object contains all information retrieved from a quantum circuit and forms the interaction graphs which are part of the input to the optimization problem.

    Args:
        qc (qiskit.QuantumCircuit): Quantum Circuit object
        fname (str): A filename (`.qasm`) where the quantum circuit is stored

    Attributes:
        two_qubit_gates (list): list of all two-qubit gates
        ngates (list): number of gates in total 
        graph (:obj:`networkx graph`): interaction graph comprising all qubits and two-qubits gates as nodes and edges respectively
        edges (list): list of edges stored as tuples (node_i,node_j)
        nodes (list): list of nodes (logical qubits)
        nnodes (int): number of nodes
        slices (list): interaction graphs of type :obj:`networkx graph` for all time slices
        laplacians (list): list of Laplacian matrices for all slices
    """

    def __init__(self, qc=None, fname=None):

        if fname == "test":
            qc = QuantumCircuit(5)
            qc.cx(0,2)
            qc.cx(1,3)
            qc.cx(0,1)
            qc.cx(3,4)
            qc.cx(1,3)
            qc.cx(1,4)
            qc.cx(0,3)
            qc.cx(2,4)
            qc.cx(1,3)
            self.qc = qc
        elif qc != None:         
            self.qc = qc
            self.decompose_to_cx() # decompose to CX-gates
        else: 
            self.qc = QuantumCircuit.from_qasm_file(fname)
            self.decompose_to_cx() # decompose to CX-gates 
            

        # two-qubit gates
        self.two_qubit_gates = [(instruction.qubits[0].index, instruction.qubits[1].index) for instruction in self.qc.data if instruction.operation.num_qubits == 2]

        if len(self.two_qubit_gates) != self.qc.num_nonlocal_gates():
            raise Exception('There are %d non-local gates, but %d CX-gates!' % (self.qc.num_nonlocal_gates(), len(self.two_qubit_gates)))

        # declare storage for slices
        self.slices = [[] for _ in range(self.qc.depth()+1)]

        # slice the circuit
        self.slice_circuit()
        self.nslices = len(self.slices)

        self.nodes = self.slices[0].nodes

    def decompose_to_cx(self):
        """ Decompose circuit until only CX-gates and single-qubit gates left.
        """
        count = 0
        while self.qc.num_nonlocal_gates() != len(self.qc.get_instructions('cx')) and count < 20:
            self.qc = self.qc.decompose()
            count += 1
            if count == 20:
                print('Decomposition stopped by count!')

    def add_gate(self, gate, n):
        """ Recursive function to add a gate to the desired slice.
        
        :param gate: gate of the quantum circuit
        :param n: current slice
        :return: add_gate() if n 
        :rtype: list
        """
        if gate in self.slices[n] or (gate[1], gate[0]) in self.slices[n]: # if gate is already apparent in this slice, add it (excluding this statement gives the depth, how it is defined in literature)
            return n, self.slices
        if any(gate[0] in t for t in self.slices[n]) or any(gate[1] in t for t in self.slices[n]) : # if one of the qubits is already used, begin filling the subsequent slice
            n += 1
            self.slices[n].append(gate)
            return n,self.slices
        elif n == 0: # start filling the first slice
            self.slices[n].append(gate)
            return n, self.slices
        else: # gate will not be assigned to this slice --> try with the previous slice
            return self.add_gate(gate, n-1)

    def slice_circuit(self):
        """ Slice the quantum circuit, where each slice holds a sequence of gates that can be executed on the multi-core system (i.e. in one time-step t) without any moves; generate respective interaction graphs.

        :return: list of interaction graphs, list of time points
        :rtype: list
        """
        for gate in self.two_qubit_gates:
            slice_list = [slice for slice in self.slices if len(slice) > 0]
            n = len(slice_list)
            self.add_gate(gate,n) # call recursive function to add gate to the desired slice

        slice_list = [slice for slice in self.slices if len(slice) > 0]
        n = len(slice_list)

        graphs = []
        for i in range(n):  # generate interaction graphs 
            graph = nx.Graph()
            graph.add_nodes_from(range(self.qc.num_qubits))
            graph.add_edges_from(slice_list[i])
            graphs.append(graph)
        self.slices = graphs

        return  self.slices #return list of nx.Graph() objects
    
    def get_laplacians(self): 
        """ Generate Laplacian matrix from each individual slice.

        :return: list of Laplacian matrices
        :rtype: list of nparrays
        """
        self.laplacians = []
        for slice in self.slices:
            self.laplacians.append(nx.laplacian_matrix(slice).toarray())
        return self.laplacians

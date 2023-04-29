# general modules
import os, csv
import sys
from time import time
from datetime import datetime

# src modules
from input import IGraph, CoreSystem
from optimization import QUBO

# benchmark instances used in Bandic et al. 2023; set-up is with Quantum Volume
from qiskit.circuit.random import random_circuit
from qiskit.circuit.library import DraperQFTAdder, CDKMRippleCarryAdder, QuantumVolume, QFT, MCMTVChain, XGate # use kind='fixed' for Adders!

# pre-processing 
from qiskit import transpile
from qiskit.transpiler import CouplingMap

def evaluate_qubomapping(igraph, cgraph, fact, results, i):
  
    mapping = {}
    valid = []

    qubo_model = QUBO(igraph, cgraph, start=0 , end=igraph.nslices, fact=fact) # initialize QUBO model
    
    build_start = time()
    qubo_model.build() # build QUBO model
    time_build = time() - build_start

    solve_start = time()
    qubo_model.divide_conquer(mapping=mapping, isvalid=valid) # solve QUBO model with divide and conquer if instance > 50,000 x-variables 
    time_solve = time() - solve_start

    isvalid = all(valid) # check if all assignments are valid
    nmoves = qubo_model.count_movements(mapping) # count the inter-core communications

    results[i] = igraph.qc.num_qubits, igraph.qc.depth(), igraph.qc.num_nonlocal_gates(), qubo_model.nslices, qubo_model.lam, time_build, time_solve, nmoves, isvalid, qubo_model.nxvars, qubo_model.nyvars


if __name__ == '__main__': 
    # terminal: nativate to the srouce folder and run: python3 experiment.py testcircuit ../test

    benchmark = sys.argv[1]
    outputdir = sys.argv[2]

    # set multi-core architecture layout (10 x 10 used in Bandic et al. 2023)
    ncores = 3
    ncap = 2
    cgraph = CoreSystem(ncores,ncap)
    
    header = ["nqubits", "cxdepth", "ncxgates", "nslices", "lam", "time to build", "time to solve", "nmoves", "isvalid", "nxvars", "nyvars"] # output header

    data = [] # storage for results
    facts = [1,2] # choose a factor for lambda = fact / (# number of time slices * # two-qubit gates)

    for n in range(5,6): # set number of qubits (50-100 used in Bandic et al. 2023)

        try:
            qc = QuantumVolume(n) # set quantum circuit to be mapped 
            qc = transpile(qc, coupling_map=CouplingMap().from_full(n), optimization_level=3) # parallelize quantum circuit
            igraph = IGraph(qc=qc) # input quantum circuit for QUBO
            
            # result list 
            results = [None]*len(facts)

            for i,fact in enumerate(facts):
                evaluate_qubomapping(igraph, cgraph, fact, results,i)

            if results[0] is None:
                raise Exception("No results here. Something went wrong.")
            
            data+=results
            
        except:
            continue

    # specify output 
    date_time = datetime.now()
    timestamp = date_time.strftime("%d-%b-%Y(%H:%M)")
    output_fname = os.path.join(outputdir+'/{}-{}-results-{}cores-{}cap.csv'.format(benchmark, timestamp, ncores, ncap)) 

    # create a new directory if it does not exist
    ispath = os.path.exists(outputdir)
    if not ispath:
        os.makedirs(outputdir)
        print("New directory is created: "+ outputdir)

    # write output to csv
    with open(output_fname, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)
    f.close()
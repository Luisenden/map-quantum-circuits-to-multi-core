# map-quantum-circuits-to-multi-core

This project reads quantum circuits (.qasm files, [Qiskit Library circuits](https://qiskit.org/documentation/apidoc/circuit_library.html#module-qiskit.circuit.library)) and maps them to the layout of a quantum multi-core system. The mapping is based on a quadratic binary unconstrained optimization (QUBO) formulation. 

If you use this code please cite:

@article{bandic2023mapping,
  title={Mapping quantum circuits to modular architectures with QUBO},
  author={Bandic, Medina and Prielinger, Luise and N{\"u}{\ss}lein, Jonas and Ovide, Gayane and Alarcon, Eduard and Almudever, Carmen G and Feld, Sebastian and others},
  journal={arXiv preprint arXiv:2305.06687},
  year={2023}
}

# How to get started
Make sure you have installed ``Python 3.8``or higher. Then run the following command
```console
pip install -r requirements.txt
````
Get started with the jupyter notebook `playground.ipynb` which contains simple examples to build your first mappings. 
Also, have a look into `experiment.py`, which you can use to reproduce our results.

```
python3 src/experiment.py test-circuit ../test
```

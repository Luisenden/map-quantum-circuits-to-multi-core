# map-quantum-circuits-to-multi-core

This project reads quantum circuits (.qasm files, [Qiskit Library circuits](https://qiskit.org/documentation/apidoc/circuit_library.html#module-qiskit.circuit.library)) and maps them to the layout of a quantum multi-core system. The mapping is based on a quadratic binary unconstrained optimization (QUBO) formulation. 

When using this code, please cite:

@article{bandic2023mapping,
  title={Mapping quantum circuits to modular architectures with QUBO},
  author={Bandic, Medina and Prielinger, Luise and N{\"u}{\ss}lein, Jonas and Ovide, Gayane and Alarcon, Eduard and Almudever, Carmen G and Feld, Sebastian and others},
  journal={arXiv preprint arXiv:2305.06687},
  year={2023}
}

# How to get started
Make sure you have installed ``Python 3.8``or higher; activate your virtual environment and run the following command
```console
pip install -r requirements.txt
````
We recommend to get started with the jupyter notebook `playground.ipynb` which contains simple examples to build your first mappings. 
Also, have a look into `experiment.py`, which you can use to reproduce our results.

```console
python src/experiment.py test-circuit ../test
```
Executing this command will map a small quantum program (test-circuit) to a 3x2 core system (similar to the toy model in the paper) and store the mapping results in the output directory ``test``.
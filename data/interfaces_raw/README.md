Raw interface files

This folder contains raw interface definitions named like:
`<pdb>_chain_<A>_<B>_interface4.5` (example: `2b4c_chain_C_H_interface4.5`).

Each file lists residues that participate in an interface at the chosen cutoff (4.5 Å here). These raw files are parsed by preprocessing scripts in `data_preprocess/` to produce NPZ label arrays used for training and evaluation.

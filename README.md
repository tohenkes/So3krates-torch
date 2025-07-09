# So3krates-torch
Implementation of the So3krates model in pytorch

# TODO

[ ] Atom type embedding always uses z_max=118 for embedding (dumb but should do the same for consistency) -> can just make a large z_table and do len(z_table). just write that as default when creating model from config
[ ] transform params.pkl to torch params and vice versa
[ ] save and load hyperparameter json from torchkrates
[ ] electrostatics
[ ] zbl
[ ] dispersion
[ ] return representations
[ ] training
[ ] ase calculator
[ ] implement other cutoff functions (phys e.g.)
[ ] check if radial basis functions give same results
[ ] enable torch.script (important for openmm)
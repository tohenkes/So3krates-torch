# So3krates-torch
Implementation of the So3krates model in pytorch

# TODO

[ ] Change from max_l to giving the degrees as a list e.g. [1,2,3]
[ ] Energy shifts and scales (learnable and give initial)
[ ] RBF calculated once and not per filter net
[ ] Atom type embedding always uses z_max=118 for embedding (dumb but should do the same for consistency) -> can just make a large z_table and do len(z_table). just write that as default when creating model from config
[ ] transform params.pkl to torch params
[ ] save and load hyperparameter json from torchkrates
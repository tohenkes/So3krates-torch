# So3krates-torch
Implementation of the So3krates model in pytorch

# TODO

[x] !!! Change from max_l to giving the degrees as a list e.g. [1,2,3]
[ ] detach num_features and energy regression dim and make customizable
[ ] !!! Energy shifts and scales (learnable and give initial)
[x] RBF calculated once and not per filter net
[ ] Atom type embedding always uses z_max=118 for embedding (dumb but should do the same for consistency) -> can just make a large z_table and do len(z_table). just write that as default when creating model from config
[ ] !!! transform params.pkl to torch params and vice versa
[ ] !!! save and load hyperparameter json from torchkrates
[ ] !!! residual mlp 1,2
[ ] !!! layer normalization
[ ] long-range modules and zbl
[ ] return representations
[ ] !!! training
[ ] !!! ase calculator
[ ] special embeddings
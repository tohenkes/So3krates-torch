# So3krates-torch

> [!IMPORTANT]
> The code is work in progress! There may be breaking changes!

Lightweight implementation of the So3krates model in pytorch. This package is mostly intended for [aims-PAX](https://github.com/tohenkes/aims-PAX) but is a functional implementation of [So3krates](https://github.com/thorben-frank/mlff) and [SO3LR](https://github.com/general-molecular-simulations/so3lr) in pytorch. For now it uses (modified) source code of the [MACE](https://github.com/ACEsuit/mace) package and follows its style, so many functions are actually compatible.

#### Installation

1. activate your environment
2. clone this repository
3. move to the clone repository
4. `pip install -r requirements.xt`
5. `pip install .`

#### Implemented features:
1. ASE calculator for MD (including pre-trained SO3LR)
2. Inference over ase readable datasets: `torchkrates-eval`
3. Error metrics over ase readable datasets: `torchkrates-test`
4. Transforming pyTorch and JAX parameter formates: `torchkrates-jax2torch` or `torchkrates-torch2jax`
5. Training is WIP but `train` in `tools.train` is already working so you can easily build your own script


> [!IMPORTANT]
> Number 4 means that you can transform the weights from this pytorch version into the JAX version and vice versa. Inference and training is much faster (*at least 1 order of magnitude*) in the JAX version. This implementation is mostly for prototyping and compatability with other packages.



## TODO

- [ ] training
    - [x] hirshfeld loss
    - [ ] load trainings params from yaml
    - [ ] script
- [ ] finetuning
- [ ] save and load hyperparameter json from torchkrates
- [ ] enable torch.script (important for openmm)


## Cite
If you are using the models implemented here please cite:

```bibtex
@article{kabylda2024molecular,
  title={Molecular Simulations with a Pretrained Neural Network and Universal Pairwise Force Fields},
  author={Kabylda, A. and Frank, J. T. and Dou, S. S. and Khabibrakhmanov, A. and Sandonas, L. M.
          and Unke, O. T. and Chmiela, S. and M{\"u}ller, K.R. and Tkatchenko, A.},
  journal={ChemRxiv},
  year={2024},
  doi={10.26434/chemrxiv-2024-bdfr0-v2}
}

@article{frank2024euclidean,
  title={A Euclidean transformer for fast and stable machine learned force fields},
  author={Frank, Thorben and Unke, Oliver and M{\"u}ller, Klaus-Robert and Chmiela, Stefan},
  journal={Nature Communications},
  volume={15},
  number={1},
  pages={6539},
  year={2024}
}
```

Also consider citing MACE, as this software heavlily leans on or uses its code:


```bibtex
@inproceedings{Batatia2022mace,
  title={{MACE}: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields},
  author={Ilyes Batatia and David Peter Kovacs and Gregor N. C. Simm and Christoph Ortner and Gabor Csanyi},
  booktitle={Advances in Neural Information Processing Systems},
  editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
  year={2022},
  url={https://openreview.net/forum?id=YPpSngE-ZU}
}

@misc{Batatia2022Design,
  title = {The Design Space of E(3)-Equivariant Atom-Centered Interatomic Potentials},
  author = {Batatia, Ilyes and Batzner, Simon and Kov{\'a}cs, D{\'a}vid P{\'e}ter and Musaelian, Albert and Simm, Gregor N. C. and Drautz, Ralf and Ortner, Christoph and Kozinsky, Boris and Cs{\'a}nyi, G{\'a}bor},
  year = {2022},
  number = {arXiv:2205.06643},
  eprint = {2205.06643},
  eprinttype = {arxiv},
  doi = {10.48550/arXiv.2205.06643},
  archiveprefix = {arXiv}
 }
```

## Contact

If you have questions you can reach me at: tobias.henkes@uni.lu

For bugs or feature requests, please use [GitHub Issues](https://github.com/tohenkes/So3krates-torch/issues).

## License

The code is published and distributed under the [MIT License](MIT.md).

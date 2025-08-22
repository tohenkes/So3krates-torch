from mlff.config.from_config import run_training
from ml_collections import config_dict
import json

# len of des15k 1272
# len of qm7x_10p 104631
# len of qm7x_1p 10463

# if neighbors_lr_bool is False, it fucks up their stupid shitty as broken ass fuck me code
config_as_dict = json.load(open("hyperparameters.json", "r"))

cfg = config_dict.ConfigDict(
    config_as_dict
)
run_training(
    config=cfg
)
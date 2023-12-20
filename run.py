# run.py

"""
This file is used to run the model with the Lightning CLI. For a reference 
of how to use the Lightning CLI, see 
https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_cli.html.
"""

from pytorch_lightning.cli import LightningCLI

def run():
    LightningCLI(
        save_config_kwargs={"overwrite": True, 'multifile' : True}
        )
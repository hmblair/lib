A boilerplate library build around PyTorch Lightning, containing abstract 
classes and some useful functions.

models.abstract_models contains a base model class, from which all other models
should inherit. The same goes for data.abstract_data.

models also contains some simple models in sklearn.py, as well as some deep
neural network layers in the remaming files. 

finetuning.abstract_finetuning contains a base model for finetuning pre-trained
models, including the ability to freeze and unfreeze certain layers at defined
epochs, and to wrap learnable LoRA parameters around such models.
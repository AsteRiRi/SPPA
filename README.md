# SPPA

This is the reference code for SPPA, this is the initial version of README and will be updated in the future



## requirements

The experiments is performed using the following libraries

- Python (3.9)
- Pytorch (1.9.1)
- torchvision (0.10.1)
- tensorboard (2.5.0)



## Perform Training

The entrance is `main.py`. All supported CLI parameters and their discriptions can be found in `utils/argparser.py`. Most parameters can use their default value and left untouched.



## Key components

The definition and training code of the projector module lie in `modules/projector.py` and `modules/transnet.py`. We the projector is optimized using SGD witm momentum. It is trained with a batch size of 32 for 1.5K iters on VOC and 3K iters on ADE. the learning rate is 1e-1 for the first 75% iters and 1e-2 for the rest. 

All the losses we proposed in this work lie in `utils/loss.py` with detailed Docstrings. They can be easily integrated with little modification in other code base or tasks.



## Hyper-parameters

- L_ali: alpha can be between 10 to 100, we use 30.
- L_str: beta can be between 1 to 100, we use 10. nu * beta can be between 1e-2 to 1e-1, we use 1e-1.
- L_cont: gamma can be between 1e-3 to 1e-1, we use 1e-2. usually mu = 1 is good.
- pseudo label: T_c is selected to keep 80% percent of the raw pseudo labels


# NeuralMidiMachines

 Using neural network based models to generate music using midi files. 

## Implemented models:

 * **LSTM** (Long Short Term Memory), paper [here](http://web.eecs.utk.edu/~itamar/courses/ECE-692/Bobby_paper1.pdf),

 * **VAE** (Variational Autoencoders) based on yOast's [VAE-Torch](https://github.com/y0ast/VAE-Torch), paper [here](https://arxiv.org/abs/1411.7610),

 * **STORN** (Stochastic Recurrent Networks), paper [here](https://arxiv.org/abs/1411.7610),

 * **VRNN** (Variational Recurrent Neural Networks), paper [here](https://arxiv.org/abs/1506.02216).

## Other:
 
 * *data/downloadMidi.py* downloads a midi classical guitar dataset.
 
 * *data/preprocessMidi.py* has various functions, like *parseMidiFile()* (flattens tracks, transposes, ...), *writeMifiFileFromTensor()*, etc...

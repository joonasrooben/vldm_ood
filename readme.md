### This is the codebase of methods and experiments of our paper **"Probability Density from Latent Diffusion Models for Out-of-Distribution Detection"** published in ECAI.

### Requirements

Requirements to run the code are given in the `environment.yml` file.

### Data and Models

Data and encoders (RN18) can be downloaded via OpenOOD repo. Trained VDMs are available at request.

### Training and evaluating
The code and the paths are adjusted to work with OpenOOD encoders but can be easily adapted to needs.

1) With OpenOOD pretrained encoders (like RN18, that we used in our work), one just needs to clone their [repo](https://github.com/Jingkang50/OpenOOD/tree/main) and adapt and run `generate_acts.py` and get the hidden representations
2) To train the VDM with desired hyperparameter one can use `train-py` 
3) To generate new samples and/or calculate densities/losses one can use `generate_and_calculate.py` with accompaning `config.yaml` 
4) To plot the figures, one can use `plotting.ipynb` jupyter notebook with `config_2.yaml` 

### Acknowledegement

Parts of the code used in our work was adapted from the Github repositories such as: [Variational Diffusion Models in Easy PyTorch](https://github.com/myscience/variational-diffusion/tree/main), [OpenOOD](https://github.com/Jingkang50/OpenOOD/tree/main), [Score-Based Generative Modeling through Stochastic Differential Equations](https://github.com/yang-song/score_sde).

### Citation
TBD
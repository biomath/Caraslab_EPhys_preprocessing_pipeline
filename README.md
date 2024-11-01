# EPhys preprocessing pipeline
This pipeline is intended to be run after extracting behavioral timestamps and neuron spike times with our [MatLab pipeline](https://github.com/caraslab/caraslab-spikesortingKS2)

## Install instructions
1. Download and install [ana/miniconda](https://docs.anaconda.com/free/miniconda/index.html)
2. Download and install [python](https://www.python.org/downloads/) >3.11 
3. Clone this repository in your computer
4. Open terminal window
5. Create conda environment
```
conda create -n caraslab_ephys
```
5. Activate conda environment
```
conda activate caraslab_ephys
```
6. Install requirements
```
conda install --yes --file requirements.txt
```
7. Run Jupyter on your browser
```
jupyter notebook
```
8. Double-click on the notebook file: Caraslab_EPhys_analysis_pipeline.ipynb




# Requirements
pip packages:
- pyglet (for rendering)
- pyvirtualdisplay (xvfb wrapper for headless rendering)

modify path:
- module load git gcc intel python/2
- module load cuda # loads cuda 7.5
- export CUDA_HOME=/shared/apps/cuda/7.0
- export LD_LIBRARY_PATH=/lib64:/home/ye30uwyn/cudnn6.5:/shared/apps/cuda/7.0/lib64:$LD_LIBRARY_PATH
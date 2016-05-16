# Deep Deterministic Policy Gradient
Paper: [Continuous control with deep reinforcement learning](http://arxiv.org/abs/1509.02971)

*This is a only a preview, a proper version will be released soon*

## Requirements
python
```bash
pip install pyglet # for rendering
pip install pyvirtualdisplay # xvfb wrapper for headless rendering
```
modify path:
```bash
# module load git gcc intel python/2
# module load cuda # loads cuda 7.5
export CUDA_HOME=/shared/apps/cuda/7.0
export LD_LIBRARY_PATH=/lib64:/home/ye30uwyn/cudnn6.5:/shared/apps/cuda/7.0/lib64:$LD_LIBRARY_PATH
```
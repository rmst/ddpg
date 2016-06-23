# Deep Deterministic Policy Gradient
Paper: ["Continuous control with deep reinforcement learning" - TP Lillicrap, JJ Hunt et al., 2015](http://arxiv.org/abs/1509.02971)

## Installation
- [install Gym](https://github.com/openai/gym#installation)
- [install TensorFlow](https://www.tensorflow.org/versions/r0.9/get_started/os_setup.html)

```bash
pip install pyglet # required for gym rendering
pip install jupyter # required only for visualization (see below)
```

## Usage
Example:
```bash
python run.py --outdir ../ddpg-results/experiment1 --env InvertedDoublePendulum-v1
```
Enter `python run.py -h` to get a complete overview.

If you want to run in the cloud or a university cluster [this](https://github.com/SimonRamstedt/ddpg-darmstadt) might contain additional information.

## Visualization
Example:
```bash
python dashboard.py --exdir ../ddpg-results
```
Enter `python dashboard.py -h` to get a complete overview.

## Known issues
- Not all gym mujoco tasks converging. 
- No batch normalization yet
- No conv nets yet (i.e. only learning from low dimensional states)
- *Please write me or open a github issue if you encounter problems!*

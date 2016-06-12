# Deep Deterministic Policy Gradient
Paper: [Continuous control with deep reinforcement learning](http://arxiv.org/abs/1509.02971)

*This is a only a preview, a proper version will be released soon*

## Installation
```bash
pip install pyglet # for gym rendering
pip install jupyter # for visualization (optional)
```

## Usage
Example:
```bash
python run.py --outdir ../ddpg-results/experiment1 --env Reacher-v1
```
Enter `python run.py -h` to get a complete overview.

## Visualization
Example:
```bash
python dashboard.py --exdir ../ddpg-results
```
Enter `python dashboard.py -h` to get a complete overview.

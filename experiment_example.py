
import argparse # https://docs.python.org/3/library/argparse.html
parser = argparse.ArgumentParser(prog='DDPG')
hp = parser.add_argument_group('hyperparameters')
hp.add_argument('--lr',default=0.001,help='learning rate')
# ... your hyperparameters ...

import experiment
args = experiment.parse(parser) # see [ python <this_file> -h ] for help

# YOUR CODE HERE

print('my learning rate: ', args['lr'])

# YOUR CODE HERE

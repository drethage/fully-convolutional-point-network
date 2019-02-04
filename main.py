""" main.py - Entry point for training, evaluation and inference with a Fully-Convolutional Point Network. """

import argparse
import training
import inference

def main():
    """ Reads command line arguments and starts either training, evaluation or inference. """

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='', help='JSON-formatted file with configuration parameters')
    parser.add_argument('--mode', default='', help='The operational mode - train|evaluate|predict')
    parser.add_argument('--input', default='', help='The path to a PLY file to run through the model')
    parser.add_argument('--device', default='gpu', help='The device to use - cpu|gpu')
    parser.add_argument('--colors', default='', help='A file containing colors')
    cla = parser.parse_args()

    if cla.config == '':
        print 'Error: --config flag is required. Enter the path to a configuration file.'

    if cla.mode == 'evaluate':
        inference.evaluate(cla.config, cla.device)
    elif cla.mode == 'predict':
        inference.predict(cla.config, cla.input, cla.device, cla.colors)
    elif cla.mode == 'train':
        training.train(cla.config)
    else:
        print 'Error: invalid operational mode - options: train, evaluate or predict'

if __name__ == "__main__":
    main()
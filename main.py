import torch
import argparse
import yaml
import datetime
import os

from train import ViTSSLTrainer
from datasets import getVOCDataloader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config_path', dest='config_path', help='config_path')
    parser.add_argument('-checkpoint_path', dest='checkpoint_path', default='', help='checkpoint_path')
    command_line_args = parser.parse_args()
    
    with open(command_line_args.config_path, 'r') as file:
        args = yaml.safe_load(file)
    if args['task_type'] == 'o4p' or args['task_type'] == 'mp':
        train_loader, validate_loader = getVOCDataloader('..', args['batch_size'], ratio=0.01, download=False, shuffle=False)
    elif args['task_type'] == 'lp':
        # train_loader, validate_loader = getVOCDataloader('..', args['batch_size'], ratio=0.01, download=False, shuffle=False)
    save_path = os.path.join('..', 'save_dir', str(datetime.datetime.now()))

    trainer = ViTSSLTrainer(args['task_type'], train_loader, validate_loader, args['epoch_num'], args['learning_rate'], \
                            args['weight_decay'], checkpoint_path=command_line_args.checkpoint_path, \
                            save_path=save_path, save_every=1)

    trainer.train()

if __name__ == '__main__':
    main()

import torch
import argparse
import yaml
import datetime
import os

from train import ViTSSLTrainer
from datasets import getVOCDataloader, getCaltechDataloader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config_path', dest='config_path', help='config_path')
    parser.add_argument('-checkpoint_path', dest='checkpoint_path', default='', help='checkpoint_path')
    command_line_args = parser.parse_args()
    
    with open(command_line_args.config_path, 'r') as file:
        args = yaml.safe_load(file)
    test_dataloader = None
    if args['task_type'] == 'o4p' or args['task_type'] == 'mp':
        train_dataloader, validate_dataloader = getVOCDataloader('..', args['batch_size'], ratio=0.01, download=False, shuffle=True)
    elif args['task_type'] == 'lp':
        train_dataloader, validate_dataloader, test_dataloader = getCaltechDataloader('..', args['batch_size'], ratio=0.3, download=False, shuffle=True)
    save_path = os.path.join('..', 'save_dir', str(datetime.datetime.now()))

    trainer = ViTSSLTrainer(args['task_type'], train_dataloader, validate_dataloader, args['epoch_num'], args['learning_rate'], \
                            args['weight_decay'], test_dataloader=test_dataloader, checkpoint_path=command_line_args.checkpoint_path, \
                            save_path=save_path, save_every=1)

    trainer.train()

if __name__ == '__main__':
    main()

import json
import sys

import hydra
import logging
from omegaconf import DictConfig

from ignite_enterance import run as go

# A logger for this file
log = logging.getLogger(__name__)

# optinal: cifar10, cifar100, svhn, only one.
run_dataset = 'cifar10'


@hydra.main(version_base=None, config_path="configs", config_name=run_dataset)
def run(cfg: DictConfig) -> None:
    """
    Run experiment on certain dataset with DictConfig read from 'configs/[run_dataset].yaml'

    When the program terminates, log files are saved outputs/[day]/[hour-minute-second]
    confusion matrix and checkpoints are saved to logs/[run_dataset]

    options:
        - analyse_time: default False, determines if we should analyse the program progress
        - log_details: default True, determines if we should enable clearml platform & save checkpoints.
        - conclude_train: default Ture, determines analyse confusion matrix and classification report
    """
    log.info(F'START TRAINING ON FOLLOWING PARAMETERS on dataset :{run_dataset}'
             F', config files are saved in [.hydra/(date).yaml]')
    options = {
        'analyse_time': False, 'log_details': True,
        'conclude_train': True
    }
    cfgdict = dict(cfg)

    go(cfgdict, options, log)
    log.info('PROCESS SUCCESSFULLY TERMINATED.')


if __name__ == '__main__':
    run()

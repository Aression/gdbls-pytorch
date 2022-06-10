import hydra, logging
from omegaconf import DictConfig
from ignite_enterance import run as go

run_dataset = 'cifar100'
# A logger for this file
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name=run_dataset)
def run(cfg: DictConfig) -> None:
    options = {
        'analyse_time': False, 'log_details': True,
        'conclude_train': False
    }
    log.info('START TRAINING')
    go(dict(cfg), options, log)
    log.info('PROCESS SUCCESSFULLY TERMINATED.')


if __name__ == '__main__':
    run()

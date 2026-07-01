import hydra
from omegaconf import OmegaConf


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    """Main entry point for PPO training with Hydra configuration management.

    Args:
        config_dict: Hydra configuration dictionary containing training parameters.
    """
    print(config)
    exit(0)

if __name__ == '__main__':
    main()
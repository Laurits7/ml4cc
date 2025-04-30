import hydra
from omegaconf import DictConfig





@hydra.main(config_path="../config", config_name="main.yaml", version_base=None)
def main(cfg: DictConfig):
    for dataset_name, dataset_values in cfg.datasets.items():
        print(k)
        
    # TODO: Load X number of CEPC files: 
    #       - Plot number of primary clusters mean+std for each Particle type and energy






if __name__ == "__main__":
    main()  # pylint: disable=E1120
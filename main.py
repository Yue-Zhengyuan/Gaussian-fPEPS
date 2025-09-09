import os
import logging
import argparse
from omegaconf import OmegaConf
from gfpeps.gaussian_fpeps import gaussian_fpeps


def get_args():
    parser = argparse.ArgumentParser(
        description="Optimization of Gaussian fPEPS with respect to BCS mean field Hamiltonian",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the configuration file.",
        default="./conf/default.yaml",
    )
    args = parser.parse_args()
    return args


def setup_logging(output: str):
    os.makedirs(output, exist_ok=True)
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        # format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        format="%(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(output + "logs.txt", mode="w"),
            # logging.StreamHandler(),  # Optional: still log to console
        ],
    )


if __name__ == "__main__":
    args = get_args()
    cfg = OmegaConf.load(args.config)
    output = cfg.file.OutDir
    assert output[-1] == "/"
    log = logging.getLogger(__name__)
    setup_logging(output)
    gaussian_fpeps(cfg)

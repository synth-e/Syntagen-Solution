import cvprw2024_syntagen_teddybear
from argparse import ArgumentParser
from synthlab.registry import report, print_instruction_of, create as create_component, ClassType
import sys
import yaml
from tqdm import tqdm
import structlog

logger = structlog.get_logger(__name__)

def parse_options():
    parser = ArgumentParser()
    parser.add_argument("-c", "--pipeline-config", type=str, help="Path to the config file (yaml format)")
    return parser.parse_args()

def main():
    options = parse_options()

    if options.pipeline_config is None:
        logger.error("Config file is not provided")
        sys.exit(1)

    with open(options.pipeline_config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    logger.info("[Constructing pipeline] Starting ...")
    pipe = create_component(
        ClassType.PIPELINE,
        cfg["type"],
        cfg["modules"],
        cfg["connections"],
        only_check=False,
        visualize='visualize.png',
    )

    logger.info("[Constructing pipeline] Done!")
    pipe(**cfg)
    del pipe


if __name__ == "__main__":
    main()

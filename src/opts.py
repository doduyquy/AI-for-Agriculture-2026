import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Training Baseline Model")
    parser.add_argument(
        "--configs", 
        nargs='+', 
        default=[
            "src/configs/paths.yaml", 
            "src/configs/model.yaml", 
            "src/configs/train.yaml", 
            "src/configs/dataset.yaml"
        ], 
        help="Paths to config files"
    )
    args = parser.parse_args()
    return args

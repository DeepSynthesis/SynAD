from pathlib import Path


def get_log_dir(dataset_type, sub_dir=""):
    output_dir = Path(__file__).parent / Path(f"../src/logs/{dataset_type}/{sub_dir}")
    output_dir.mkdir(exist_ok=True, parents=True)
    return output_dir

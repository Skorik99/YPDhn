import subprocess
import os
import argparse

# Create outputs directory if it doesn't exist
os.makedirs("outputs", exist_ok=True)

parser = argparse.ArgumentParser(description="Run federated learning experiments.")
parser.add_argument(
    "--device_id",
    type=int,
    default=0,
    help="GPU device IDx to use (default: 0)",
)
args = parser.parse_args()

# Configuration parameters
FEDERATED_METHODS = ["bant", "simbant", "autobant"]
ATTACK_TYPES = ["alie"]
SAMPLE_SIZE = [250, 1000]
BASE_PARAMS = [
    "training_params.batch_size=32",
    f"training_params.device_ids=[{args.device_id}]",
    "federated_params.print_client_metrics=False",
    "federated_params.attack_scheme=constant",
    "federated_params.prop_attack_rounds=1.0",
    "federated_params.clients_attack_types=alie",
    "federated_params.prop_attack_clients=0.4",
]


# Run experiments
for trial_size in SAMPLE_SIZE:
    for method in FEDERATED_METHODS:
        params = [
            f"federated_method={method}",
            f"federated_method.trust_sample_amount={trial_size}",
        ] + BASE_PARAMS
        output_path = f"{method}_alie_trial_size{trial_size}.txt"

        # Create full command
        cmd = ["nohup", "python", "../src/train.py"] + params

        # Convert to string with output redirection
        cmd_str = " ".join(cmd) + f" > {output_path}"

        print(f"Running setup: {method}+{trial_size}", flush=True)
        subprocess.run(cmd_str, shell=True, check=True)

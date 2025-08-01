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
NUM_ITERS = [15, 30, 45, 60, 75]
BASE_PARAMS = [
    "training_params.batch_size=32",
    f"training_params.device_ids=[{args.device_id}]",
    "federated_params.print_client_metrics=False",
    "federated_params.attack_scheme=constant",
    "federated_params.prop_attack_rounds=1.0",
    "federated_params.clients_attack_types=alie",
    "federated_params.prop_attack_clients=0.4",
    "federated_method=autobant",
]


# Run experiments
for i, num_iters in enumerate(NUM_ITERS):
    params = [f"federated_method.num_opt_epochs={i+1}"] + BASE_PARAMS

    output_path = f"autobant_alie_delta_error_{num_iters}iters.txt"

    # Create full command
    cmd = ["nohup", "python", "../src/train.py"] + params

    # Convert to string with output redirection
    cmd_str = " ".join(cmd) + f" > {output_path}"

    print(f"Running setup: {num_iters}", flush=True)
    subprocess.run(cmd_str, shell=True, check=True)

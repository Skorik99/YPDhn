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
ATTACK_TYPES = ["random_grad", "alie", "ipm"]
BASE_PARAMS = [
    "training_params.batch_size=32",
    f"training_params.device_ids=[{args.device_id}]",
    "observed_data_params@dataset=cifar10_dirichlet",
    "federated_params.print_client_metrics=False",
    "federated_params.attack_scheme=constant",
    "federated_params.prop_attack_rounds=1.0",
]
ALPHAS = [0.5, 1]


def build_command(federated_method, attack_type, alpha):
    # Determine prop_attack_clients based on attack type
    if attack_type == "alie":
        prop_attack = 0.4
    elif attack_type == "ipm":
        prop_attack = 0.7
    else:
        prop_attack = 0.5

    # Build parameters list
    params = [
        f"federated_method={federated_method}",
        f"federated_params.clients_attack_types={attack_type}",
        f"federated_params.prop_attack_clients={prop_attack}",
        f"dataset.alpha={alpha}",
    ] + BASE_PARAMS

    # Build output filename
    output_name = f"{federated_method}_{attack_type}_dirichlet_alha{alpha}.txt"

    return params, output_name


# Run experiments
for alpha in ALPHAS:
    for method in FEDERATED_METHODS:
        for attack in ATTACK_TYPES:
            # Build command and output path
            params, output_path = build_command(method, attack, alpha)
            if alpha == 0.5:
                params += ["random_state=39"]

            # Create full command
            cmd = ["nohup", "python", "../src/train.py"] + params

            # Convert to string with output redirection
            cmd_str = " ".join(cmd) + f" > {output_path}"

            # Print and execute
            print(f"Running setup: {method}+{attack}+{alpha}", flush=True)
            subprocess.run(cmd_str, shell=True, check=True)

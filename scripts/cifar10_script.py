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
FEDERATED_METHODS = [
    "bant",
    "autobant",
    "simbant",
    "zeno",
    "recess",
    "central_clip",
    "fbm",
    "bucketing",
    "safeguard",
    "fltrust",
    "byz_vr_marina",
    "fedavg",
]
ATTACK_TYPES = ["no_attack", "label_flip", "sign_flip", "random_grad", "alie", "ipm"]
BASE_PARAMS = [
    "training_params.batch_size=32",
    f"training_params.device_ids=[{args.device_id}]",
    "federated_params.print_client_metrics=False",
]


def build_command(federated_method, attack_type):
    params = [f"federated_method={federated_method}"]
    # Specify command based on attack type
    if attack_type in ["label_flip", "random_gradients"]:
        prop_attack = 0.5
    elif attack_type == "sign_flip":
        prop_attack = 0.6
    elif attack_type == "alie":
        prop_attack = 0.4
    elif attack_type == "ipm":
        prop_attack = 0.7
    else:
        # no attack_type
        prop_attack = 0.0
    if attack_type != "no_attack":
        params += [
            "federated_params.attack_scheme=constant",
            "federated_params.prop_attack_rounds=1.0",
            f"federated_params.clients_attack_types={attack_type}",
            f"federated_params.prop_attack_clients={prop_attack}",
        ]

    # Build parameters list
    params += BASE_PARAMS

    # Build output filename
    output_name = f"{federated_method}_{attack_type}_cifra10.txt"

    return params, output_name


# Run experiments
for attack in ATTACK_TYPES:
    for method in FEDERATED_METHODS:
        # Build command and output path
        params, output_path = build_command(method, attack)
        # Create full command
        cmd = ["nohup", "python", "../src/train.py"] + params
        # Convert to string with output redirection
        cmd_str = " ".join(cmd) + f" > {output_path}"
        # Print and execute
        print(f"Running setup: {method}+{attack}", flush=True)
        subprocess.run(cmd_str, shell=True, check=True)

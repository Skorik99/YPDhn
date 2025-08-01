import os
import tarfile
import requests
import numpy as np
import pickle
import pandas as pd
from PIL import Image
import argparse
import yaml


# Set global seeds for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def download_cifar10(target_dir="cifar10"):
    os.makedirs(target_dir, exist_ok=True)
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    tar_path = os.path.join(target_dir, "cifar-10-python.tar.gz")

    if not os.path.exists(tar_path):
        print("Downloading CIFAR-10...")
        response = requests.get(url, stream=True)
        with open(tar_path, "wb") as f:
            f.write(response.content)

    # Extract files
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=target_dir)
    os.remove(tar_path)


def process_cifar10(base_dir="cifar10"):
    img_dir = os.path.join(base_dir, "images", "data")
    os.makedirs(img_dir, exist_ok=True)

    if not os.path.isabs(img_dir):
        curent_run_path = os.getcwd()
        img_dir = os.path.join(curent_run_path, img_dir)

    with open(os.path.join(base_dir, "cifar-10-batches-py", "batches.meta"), "rb") as f:
        meta = pickle.load(f)
    label_names = meta["label_names"]

    all_data = []
    print("Converting CIFAR-10...")
    for split in ["train", "test"]:
        files = (
            ["data_batch_%d" % i for i in range(1, 6)]
            if split == "train"
            else ["test_batch"]
        )

        for file in files:
            with open(os.path.join(base_dir, "cifar-10-batches-py", file), "rb") as f:
                data = pickle.load(f, encoding="bytes")

            images = data[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            labels = data[b"labels"]

            for i, (img, label) in enumerate(zip(images, labels)):
                if split == "train":
                    file_split = file.split("_")[-1]
                else:
                    file_split = split
                filename = f"{label_names[label]}_{file_split}_split_{i:06d}.png"
                path = os.path.join(img_dir, filename)
                Image.fromarray(img).save(path)
                all_data.append(
                    {
                        "fpath": path,
                        "file_name": filename,
                        "target": label,
                        "split": "train" if split == "train" else "test",
                    }
                )

    return pd.DataFrame(all_data)


def create_splits(df):
    print("Splitting CIFAR-10...")

    def uniform_split(df, num_samples, random_state):
        split_samples = []
        split_ids = []
        for class_id in range(10):
            class_samples = df[df["target"] == class_id].sample(
                num_samples, random_state=random_state
            )
            split_samples.append(class_samples)
            split_ids.extend(class_samples.index)
        return split_samples, split_ids

    def split_train(df, num_clients):
        client_data = []
        available_df = df.copy()
        for client_id in range(1, num_clients + 1):
            client_samples, client_ids = uniform_split(
                available_df,
                num_samples=len(df) // num_clients // 10,
                random_state=RANDOM_STATE + client_id,
            )
            available_df = available_df.drop(client_ids)
            client_df = pd.concat(client_samples)
            client_df["client"] = client_id
            client_data.append(client_df)
        split_train_df = (
            pd.concat(client_data).drop_duplicates().drop(columns=["split"])
        )
        return split_train_df

    full_test_df = df[df["split"] == "test"]
    test_samples, _ = uniform_split(
        full_test_df, num_samples=600, random_state=RANDOM_STATE
    )
    test_df = pd.concat(test_samples).drop(columns=["split"])
    trust_df = full_test_df[~full_test_df.index.isin(test_df.index)].drop(
        columns=["split"]
    )

    train_df = df[df["split"] == "train"]

    # 10 clients split
    train_df_10clients = split_train(train_df, num_clients=10)
    # 100 clients split
    train_df_100clients = split_train(train_df, num_clients=100)

    return train_df_10clients, train_df_100clients, trust_df, test_df


def set_data_configs(target_path):
    print("Setting paths to .yaml files...")
    config_dir = "src/configs/observed_data_params/"
    if not os.path.isdir(config_dir):
        print(
            f"Directory {config_dir} not found. Set paths inside .yaml configs manually"
        )
        return

    config_names = [
        "cifar10.yaml",
        "cifar10_dirichlet.yaml",
        "cifar10_100clients.yaml",
        "cifar10_trust.yaml",
    ]

    if not os.path.isabs(target_path):
        curent_run_path = os.getcwd()
        target_path = os.path.join(curent_run_path, target_path)

    for filename in os.listdir(config_dir):
        if filename not in config_names:
            continue

        filepath = os.path.join(config_dir, filename)
        with open(filepath, "r") as f:
            data = yaml.safe_load(f)

        data_sources = data.get("data_sources", {})

        if "test_directories" in data_sources:
            test_map_path = [os.path.join(target_path, "images", "test_map_file.csv")]
            data_sources["test_directories"] = test_map_path

        if "train_directories" in data_sources:
            if filename in ["cifar10.yaml", "cifar10_dirichlet.yaml"]:
                train_map_name = "10_clients_train_map_file.csv"
            elif filename == "cifar10_100clients.yaml":
                train_map_name = "100_clients_train_map_file.csv"
            elif filename == "cifar10_trust.yaml":
                train_map_name = "trust_map_file.csv"
            else:
                train_map_name = None

            if train_map_name is not None:
                train_map_path = [os.path.join(target_path, "images", train_map_name)]
                data_sources["train_directories"] = train_map_path

        data["data_sources"] = data_sources

        with open(filepath, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

    print("All steps completed successfully!!!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process CIFAR-10 dataset and create splits."
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        default=".",
        help="Directory to save processed files (default: current directory)",
    )
    args = parser.parse_args()

    # 1. Download dataset
    download_cifar10(args.target_dir)
    # 2. Convert to images, create a map-file
    full_df = process_cifar10(args.target_dir)
    # 3. Split map-file
    train_10_clients_df, train_100_clients_df, trust_df, test_df = create_splits(
        full_df
    )
    # 4. Save map-files in target directory
    train_10_clients_df.to_csv(
        os.path.join(args.target_dir, "images", "10_clients_train_map_file.csv"),
        index=False,
    )
    train_100_clients_df.to_csv(
        os.path.join(args.target_dir, "images", "100_clients_train_map_file.csv"),
        index=False,
    )
    trust_df.to_csv(
        os.path.join(args.target_dir, "images", "trust_map_file.csv"), index=False
    )
    test_df.to_csv(
        os.path.join(args.target_dir, "images", "test_map_file.csv"), index=False
    )
    # 5. Set observed_data_params configs
    set_data_configs(args.target_dir)

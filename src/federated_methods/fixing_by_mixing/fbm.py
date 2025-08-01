import torch
import copy
import time

from ..central_clip.central_clip import CentralClip
from utils.attack_utils import set_client_map_round


class FBM(CentralClip):
    def __init__(self, num_byzantines, momentum_beta, tau_clip, clip_iters, ckpt_path):
        self.num_byzantines = num_byzantines
        self.ckpt_path = ckpt_path
        super().__init__(momentum_beta, tau_clip, clip_iters)

    def find_nearest_neighbours(self, rank):
        clients_distance = {}
        client_flat = torch.cat(
            [
                x.flatten()
                for k, x in self.server.client_gradients[rank].items()
                if k != "ipm_eps"
            ]
        )
        for neighbour_rank in range(self.num_clients):
            if neighbour_rank == rank:
                continue
            neighbour_flat = torch.cat(
                [
                    x.flatten()
                    for k, x in self.server.client_gradients[neighbour_rank].items()
                    if k != "ipm_eps"
                ]
            )
            clients_distance[neighbour_rank] = torch.linalg.norm(
                neighbour_flat - client_flat, dtype=torch.float32
            ).item()

        return [
            k for k, _ in sorted(clients_distance.items(), key=lambda item: item[1])
        ]

    def nearest_neighbour_mixing(self):
        new_client_gradients = []
        self.num_clients = len(self.server.client_gradients)
        for rank in range(self.num_clients):
            new_client_grad = {}
            sorted_ranks = self.find_nearest_neighbours(rank)
            print(
                f"For client {rank} the list of nearest clients: {sorted_ranks[:self.num_clients - self.num_byzantines]}"
            )
            for key, weights in self.server.client_gradients[rank].items():
                if key == "ipm_eps":
                    continue
                for i in range(self.num_clients - self.num_byzantines):
                    if key not in new_client_grad:
                        new_client_grad[key] = self.server.client_gradients[
                            sorted_ranks[i]
                        ][key] / (self.num_clients - self.num_byzantines)
                    else:
                        new_client_grad[key] = new_client_grad[
                            key
                        ] + self.server.client_gradients[sorted_ranks[i]][key] / (
                            self.num_clients - self.num_byzantines
                        )

            new_client_gradients.append(new_client_grad)

        self.server.client_gradients = copy.deepcopy(new_client_gradients)

    def aggregate(self):
        self.nearest_neighbour_mixing()
        aggregated_weights = super().aggregate()
        return aggregated_weights

    def begin_train(self):
        self.manager.create_clients(
            self.client_args, self.client_kwargs, self.client_attack_map
        )
        self.clients_loader = self.manager.batches

        for round in range(self.rounds):
            print(f"\nRound number: {round}")
            begin_round_time = time.time()
            self.cur_round = round

            self.server.test_global_model()

            if round == 0 and self.ckpt_path is not None:
                ckpt = torch.load(self.ckpt_path, map_location=self.server.device)
                self.server.global_model.load_state_dict(ckpt["model"])

            print("\nTraining started\n")

            self.client_map_round = set_client_map_round(
                self.client_attack_map, self.attack_rounds, self.attack_scheme, round
            )

            self.train_round()

            self.server.save_best_model(round)

            aggregated_weights = self.aggregate()
            self.server.global_model.load_state_dict(aggregated_weights)

            print(f"Round time: {time.time() - begin_round_time}", flush=True)

        print("Shutdown clients, federated learning end", flush=True)
        self.manager.stop_train()

import torch
import copy
import time

from ..fedavg.fedavg import FedAvg
from utils.attack_utils import set_client_map_round


class ByzVrMarina(FedAvg):
    def __init__(self, p, gamma, tau_clip, clip_iters):
        super().__init__()
        self.p = p
        self.gamma = gamma
        self.tau_clip = tau_clip
        self.clip_iters = clip_iters
        self.quantize_fn = torch.nn.Identity()

    def count_aggregated_client_grads(self, client_gradients):
        aggregated_weights = {key: torch.zeros_like(weights, dtype=torch.float32) for key, weights in client_gradients[0].items()}

        for i in range(len(client_gradients)):
            for key, weights in client_gradients[i].items():
                aggregated_weights[key] += weights * (1 / len(client_gradients))

        return aggregated_weights

    def subtract_gradients(self, g_new, g_old):
        return {key: g_new[key] - g_old[key] for key in g_new}

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

            print("\nTraining started\n")

            self.client_map_round = set_client_map_round(
                self.client_attack_map, self.attack_rounds, self.attack_scheme, round
            )

            if round:
                c_k = torch.bernoulli(torch.tensor([self.p])).item() == 1

                aggregated_grads = self.count_aggregated_client_grads(self.server.client_gradients)
                old_weights = copy.deepcopy(self.server.global_model.state_dict())

                self.train_round()
                old_gradients = self.server.client_gradients

                new_weights = {
                    key: old_weights[key] + self.gamma * aggregated_grads[key].to(old_weights[key].device)
                    for key in old_weights
                }
                self.server.global_model.load_state_dict(new_weights)

                self.train_round()

                if not c_k:
                    new_gradients = self.server.client_gradients
                    diff = [self.subtract_gradients(g_old, g_new) for g_new, g_old in zip(new_gradients, old_gradients)]

                    q_diff = [self.quantize_fn(d) for d in diff]
                    new_client_gradients = [
                        {key: aggregated_grads[key] + q_diff[i][key] for key in aggregated_grads}
                        for i in range(len(q_diff))
                    ]
                    self.server.client_gradients = new_client_gradients
            else:
                self.train_round()

            self.server.save_best_model(round)

            aggregated_weights = self.aggregate()
            self.server.global_model.load_state_dict(aggregated_weights)

            print(f"Round time: {time.time() - begin_round_time}", flush=True)
    
    def aggregate(self):
        aggregated_weights = self.server.global_model.state_dict()
        for l in range(self.clip_iters):
            clipped_clients_updates = []
            # Client Gradients Clipping
            for i in range(len(self.server.client_gradients)):
                clip_update = {}
                for name, _ in self.server.global_model.named_parameters():
                    client_grad = self.server.client_gradients[i][name].to(
                        self.server.device
                    )
                    clip_weight = min(1, self.tau_clip / torch.norm(client_grad))
                    clip_update[name] = client_grad * clip_weight
                clipped_clients_updates.append(clip_update)
            # Server aggregation
            for i in range(len(self.server.client_gradients)):
                for name, _ in self.server.global_model.named_parameters():
                    aggregated_weights[name] = aggregated_weights[
                        name
                    ] + clipped_clients_updates[i][name] * (
                        1 / len(self.server.client_gradients)
                    )

        # Server buffer aggregation
        for name, _ in self.server.global_model.named_buffers():
            aggregated_weights[name] = aggregated_weights[name] + sum(
                self.server.client_gradients[i][name].to(self.server.device)
                for i in range(len(self.server.client_gradients))
            ) / len(self.server.client_gradients)

        return aggregated_weights

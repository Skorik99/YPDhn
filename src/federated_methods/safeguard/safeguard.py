import math
import torch
from collections import OrderedDict

from ..fedavg.fedavg import FedAvg
from .safeguard_server import Safeguard_server


class Safeguard(FedAvg):
    def __init__(
        self, T_0, T_1, multiplier, min_thresh_A, min_thresh_B, noise_std_coef, lr
    ):
        super().__init__()
        self.thresholds = (T_0, T_1)
        self.multiplier = multiplier
        self.min_thresh = (min_thresh_A, min_thresh_B)
        self.noise_std_coef = noise_std_coef
        self.lr = lr

    def _init_server(self, cfg):
        self.server = Safeguard_server(cfg)

    def reset_accumulation(self, index):
        if self.cur_round % self.thresholds[index] == 0:
            for i in range(self.cfg.federated_params.amount_of_clients):
                for key, _ in self.server.global_model.named_parameters():
                    self.server.client_safeguards[index][i][key] = 0

    def accumulate_gradients(self, index):
        for i in range(self.cfg.federated_params.amount_of_clients):
            for key, _ in self.server.global_model.named_parameters():
                self.server.client_safeguards[index][i][
                    key
                ] += self.server.client_gradients[i][key] / len(
                    self.server.good_clients
                )

    def finding_median_grads(self, index):
        client_scores = [0 for _ in range(self.cfg.federated_params.amount_of_clients)]
        for i in range(self.cfg.federated_params.amount_of_clients):
            scores = []
            for j in range(self.cfg.federated_params.amount_of_clients):
                scores.append(
                    (
                        j,
                        self.diff_norm(
                            self.server.client_safeguards[index][i],
                            self.server.client_safeguards[index][j],
                        ),
                    )
                )
            scores.sort(key=lambda dist: dist[1])
            client_scores[i] = scores[
                math.ceil(
                    self.cfg.federated_params.amount_of_clients / 2 - 1
                )  # THIS IS INCORRECT, MINUS SHOULD BE PLUS
            ]
        client_scores = [x for x in client_scores if x != 0]
        client_scores.sort(key=lambda dist: dist[1])
        self.server.min_score[index] = client_scores[0]
        self.server.grad_med_acum[index] = self.server.client_safeguards[index][
            self.server.min_score[index][0]
        ]

    def diff_norm(self, grad1, grad2):
        tmp_grad = OrderedDict()
        for key, _ in self.server.global_model.named_parameters():
            tmp_grad[key] = grad1[key] - grad2[key]
        tmp_grad_flat = torch.cat([x.flatten() for x in list(tmp_grad.values())])
        return torch.norm(tmp_grad_flat)

    def filter_workers(self, index):
        threshold = self.multiplier * min(
            self.min_thresh[index], self.server.min_score[index][1]
        )
        new_good = []
        for i in range(self.cfg.federated_params.amount_of_clients):
            if (
                self.diff_norm(
                    self.server.client_safeguards[index][i],
                    self.server.grad_med_acum[index],
                )
                < 2 * threshold
            ):
                new_good.append(i)
        return new_good

    def add_noise(self, aggregated_weights):
        for key, weights in aggregated_weights.items():
            gaussian_noise = torch.normal(
                0,
                self.noise_std_coef,
                size=weights.shape,
            ).to(self.server.device)
            aggregated_weights[key] = weights - self.lr * gaussian_noise
        return aggregated_weights

    def aggregate(self):
        for i in range(2):
            self.reset_accumulation(i)
            self.accumulate_gradients(i)
            self.finding_median_grads(i)
        self.server.good_clients = list(
            set(self.filter_workers(0)).intersection(self.filter_workers(1))
        )
        print(f"Good workers after filtration: {self.server.good_clients}", flush=True)

        aggregated_weights = self.server.global_model.state_dict()
        for i in self.server.good_clients:
            for key, weights in self.server.client_gradients[i].items():
                aggregated_weights[key] = aggregated_weights[key] + weights.to(
                    self.server.device
                ) * (1 / len(self.server.good_clients))
        aggregated_weights = self.add_noise(aggregated_weights)
        return aggregated_weights

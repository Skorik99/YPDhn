import torch
from collections import OrderedDict
from torch.nn.functional import relu

from ..trial.trial import Trial
from .fltrust_server import FLTrustServer
from utils.data_utils import read_dataframe_from_cfg


class FLTrust(Trial):
    def _init_server(self, cfg):
        fltrust_df = read_dataframe_from_cfg(cfg, "train_directories", "trust_df")
        self.server = FLTrustServer(cfg, fltrust_df)

    def count_trust_scores(self):
        self.server.fltrust_train()
        trust_scores = self.calculate_trust_scores()
        self.normalize_magnitudes()
        return trust_scores

    def calculate_trust_scores(self):
        self.client_directions = []
        self.server_direction = torch.cat(
            [x.flatten() for x in list(self.server.server_grad.values())]
        )
        num_clients = len(self.server.client_gradients)
        trust_scores = []

        for i in range(num_clients):
            self.client_directions.append(
                torch.cat(
                    [
                        self.server.client_gradients[i][k].flatten()
                        for k in list(self.server.server_grad.keys())
                    ]
                )
            )
            trust_scores.append(
                self.client_trust_score(
                    self.server_direction, self.client_directions[i]
                )
            )
        # normalize trust scores
        trust_scores = [ts / sum(trust_scores) for ts in trust_scores]
        return trust_scores

    def normalize_magnitudes(self):
        num_clients = len(self.server.client_gradients)
        for i in range(num_clients):
            normalized_client_gradients = OrderedDict()
            for key, weights in self.server.client_gradients[i].items():
                normalized_client_gradients[key] = (
                    weights
                    * torch.norm(self.server_direction)
                    / torch.norm(self.client_directions[i])
                )
            self.server.client_gradients[i] = normalized_client_gradients

    def client_trust_score(self, server_direction, client_direction):
        return relu(
            torch.dot(server_direction, client_direction)
            / (torch.norm(server_direction) * torch.norm(client_direction))
        )

import copy
import torch
from itertools import islice
from collections import OrderedDict

from ..fedavg.fedavg import FedAvg
from .zeno_server import Zeno_server
from utils.data_utils import read_dataframe_from_cfg


class Zeno(FedAvg):
    def __init__(self, ro, b, use_buffers):
        super().__init__()
        self.ro = ro
        self.b = b
        self.use_buffers = use_buffers
        self.initial_global_model_state = None

    def _init_server(self, cfg):
        trust_df = read_dataframe_from_cfg(cfg, "train_directories", "trust_df")
        self.server = Zeno_server(cfg, trust_df, self.use_buffers)

    def find_sds(self):
        self.initial_global_model_state = copy.deepcopy(
            self.server.global_model
        ).state_dict()
        prev_trust_loader = copy.deepcopy(self.server.trust_loader)
        self.server.trust_loader = [
            next(
                islice(
                    self.server.trust_loader,
                    self.cur_round % len(self.server.trust_loader),
                    (self.cur_round % len(self.server.trust_loader)) + 1,
                )
            )
        ]

        _, _ = self.server.eval_fn("trust")
        global_model_loss = self.server.trust_loss

        for i in range(len(self.server.client_gradients)):
            self.get_loss_for_sds(self.server.client_gradients[i])
            grad_norm = self.find_grad_norm(self.server.client_gradients[i])
            self.server.sds[i] = (
                global_model_loss - self.server.trust_loss - self.ro * (grad_norm**2)
            )
            print(f"Client {i} sds score: {self.server.sds[i]}")

        self.server.global_model.load_state_dict(self.initial_global_model_state)
        self.server.trust_loader = prev_trust_loader

    def overwrite_server_global_model(self, grad_state_dict):
        tmp_weights = OrderedDict()
        for key, weights in grad_state_dict.items():
            tmp_weights[key] = self.initial_global_model_state[key] + weights.to(
                self.server.device
            )
        self.server.global_model.load_state_dict(tmp_weights)

    def get_loss_for_sds(self, grad_state_dict):
        self.overwrite_server_global_model(grad_state_dict)
        _, _ = self.server.eval_fn("trust")

    def find_grad_norm(self, grad):
        needed_state = (
            self.server.global_model.state_dict().items()
            if self.use_buffers
            else self.server.global_model.named_parameters()
        )
        grad_1d = torch.cat([grad[key].flatten() for key, _ in needed_state])
        return torch.norm(grad_1d)

    def find_highest_sds(self):
        amount = len(self.server.client_gradients) - self.b
        threshold = sorted(self.server.sds, reverse=True)[amount - 1]
        self.server.sds = [
            value if value >= threshold else 0 for value in self.server.sds
        ]
        print(
            f"Chosen clients: {[i for i in range(len(self.server.sds)) if self.server.sds[i]]}"
        )

    def aggregate(self):
        self.find_sds()
        self.find_highest_sds()
        aggregated_weights = self.server.global_model.state_dict()
        for i in range(len(self.server.client_gradients)):
            if self.server.sds[i]:
                for key, weights in self.server.client_gradients[i].items():
                    aggregated_weights[key] = aggregated_weights[key] + weights.to(
                        self.server.device
                    ) * (1 / (len(self.server.client_gradients) - self.b))
        return aggregated_weights

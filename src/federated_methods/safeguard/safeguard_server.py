from collections import OrderedDict

from ..fedavg.server import Server


class Safeguard_server(Server):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.good_clients = list(range(cfg.federated_params.amount_of_clients))
        self.client_safeguards = [
            [OrderedDict() for _ in range(cfg.federated_params.amount_of_clients)],
            [OrderedDict() for _ in range(cfg.federated_params.amount_of_clients)],
        ]
        for i in self.good_clients:
            for key, _ in self.global_model.named_parameters():
                self.client_safeguards[0][i][key] = 0
                self.client_safeguards[1][i][key] = 0
        self.min_score = [0, 0]
        self.grad_med_acum = [OrderedDict(), OrderedDict()]

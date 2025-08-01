import torch

from ..fedavg.fedavg import FedAvg


class CentralClip(FedAvg):
    def __init__(self, momentum_beta, tau_clip, clip_iters):
        """
        Central Clipping with momentum (https://arxiv.org/pdf/2012.10333, see Algorithm 2)
        In theory, clipping occurs every local iteration over the client g_i(x)
        We adapted this algorithm to the case of multiple local computations:
        g_i(x) --> Δ_i = x_i^t - x^t

        Args:
            momentum_beta (float): momentum in local Client Optimizer
            tau_clip (float): clipping coef in Central Clipping
            clip_iters (int): number of central clip iterations
        """
        super().__init__()
        self.momentum_beta = momentum_beta
        self.tau_clip = tau_clip
        self.clip_iters = clip_iters

    def _init_client_cls(self):
        assert "SGD" in str(
            self.cfg.optimizer._target_
        ), f"CentralClip works only with SGD client optimizer. You provide: {self.cfg.optimizer._target_}"
        self.cfg.optimizer.momentum = self.momentum_beta
        super()._init_client_cls()

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

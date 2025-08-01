import copy
import numpy as np

from ..central_clip.central_clip import CentralClip
from .bucketing_server import BucketingServer


class Bucketing(CentralClip):
    def __init__(self, beta, s, momentum_beta, tau_clip, clip_iters):
        self.beta = beta
        self.s = s
        self.change_beta = True
        super().__init__(momentum_beta, tau_clip, clip_iters)

    def _init_server(self, cfg):
        self.server = BucketingServer(cfg, self.s)

    def update_client_momentums(self):
        for i in range(len(self.server.client_gradients)):
            for key, _ in self.server.global_model.named_parameters():
                if self.change_beta:
                    self.server.client_momentums[i][key] = self.server.client_gradients[
                        i
                    ][key]
                else:
                    self.server.client_momentums[i][key] = (
                        1 - self.beta
                    ) * self.server.client_gradients[i][
                        key
                    ] + self.beta * self.server.client_momentums[
                        i
                    ][
                        key
                    ]
            for key, _ in self.server.global_model.named_buffers():
                self.server.client_momentums[i][key] = self.server.client_gradients[i][
                    key
                ]
        if self.change_beta:
            self.change_beta = False

    def aragg(self):
        np.random.seed(self.server.cfg.random_state)

        permutations = np.random.permutation(len(self.server.client_momentums))
        for i in range(len(self.server.buckets)):
            bucket = {k: 0 for k in self.server.client_momentums[0].keys()}
            for j in range(i * self.s, min(len(permutations), (i + 1) * self.s)):
                for key, weights in self.server.client_momentums[
                    permutations[j]
                ].items():
                    bucket[key] += weights * (1 / self.s)
            self.server.buckets[i] = bucket

    def aggregate(self):
        self.update_client_momentums()
        self.aragg()
        old_grads = copy.deepcopy(self.server.client_gradients)
        self.server.client_gradients = copy.deepcopy(self.server.buckets)
        aggregated_weights = super().aggregate()
        self.server.client_gradients = copy.deepcopy(old_grads)
        return aggregated_weights

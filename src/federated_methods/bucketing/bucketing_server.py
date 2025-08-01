import math
from collections import OrderedDict

from ..fedavg.server import Server


class BucketingServer(Server):
    def __init__(self, cfg, s):
        super().__init__(cfg)
        self.client_momentums = [
            OrderedDict() for _ in range(self.cfg.federated_params.amount_of_clients)
        ]
        self.buckets = [
            OrderedDict()
            for _ in range(math.ceil(self.cfg.federated_params.amount_of_clients / s))
        ]

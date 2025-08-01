import torch

from ..trial.trial import Trial
from .bant_server import BANTServer
from utils.data_utils import read_dataframe_from_cfg, get_stratified_subsample


class BANT(Trial):
    def __init__(self, trust_sample_amount, momentum_beta):
        super().__init__()
        self.trust_sample_amount = trust_sample_amount
        self.momentum_beta = momentum_beta

    def _init_server(self, cfg):
        trust_df = read_dataframe_from_cfg(cfg, "train_directories", "trust_df")
        _, trust_df = get_stratified_subsample(
            df=trust_df,
            num_samples=self.trust_sample_amount,
            random_state=cfg.random_state,
        )
        self.num_clients = cfg.federated_params.amount_of_clients
        self.prev_trust_scores = [1 / self.num_clients] * self.num_clients
        self.server = BANTServer(cfg, trust_df)

    def count_trust_scores(self):
        server_loss, client_losses = self.server.get_trust_losses()
        trust_scores = self.calculate_trust_score(server_loss, client_losses)
        return trust_scores

    def calculate_trust_score(self, server_loss, client_losses):
        # Calculate loss diff
        trust_scores = [max(server_loss - cl, 0) for cl in client_losses]
        # Create a trust scores with momentum
        sum_ts = sum(trust_scores)
        beta = self.momentum_beta if sum_ts else 0.001
        trust_scores = (
            [ts / sum_ts for ts in trust_scores]
            if sum_ts
            else [1 / self.num_clients] * self.num_clients
        )
        self.prev_trust_scores = [
            (1 - beta) * prev_ts + beta * cur_ts
            for prev_ts, cur_ts in zip(self.prev_trust_scores, trust_scores)
        ]
        # Make idicating
        trust_scores = [
            prev_ts if cur_ts else cur_ts
            for prev_ts, cur_ts in zip(self.prev_trust_scores, trust_scores)
        ]
        # normalize ts
        trust_scores = [ts / sum(trust_scores) for ts in trust_scores]
        return trust_scores

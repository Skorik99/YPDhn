from collections import OrderedDict

from ..trial.trial import Trial
from .autobant_server import AutoBANTServer
from utils.data_utils import read_dataframe_from_cfg, get_stratified_subsample


class AutoBANT(Trial):
    def __init__(
        self,
        trust_sample_amount,
        start_point,
        end_point,
        num_opt_epochs,
        mirror_gamma,
        ts_momentum,
    ):
        super().__init__()
        self.trust_sample_amount = trust_sample_amount
        self.opt_params = [
            start_point,
            end_point,
            num_opt_epochs,
            mirror_gamma,
            ts_momentum,
        ]

    def _init_server(self, cfg):
        trust_df = read_dataframe_from_cfg(cfg, "train_directories", "trust_df")
        _, trust_df = get_stratified_subsample(
            df=trust_df,
            num_samples=self.trust_sample_amount,
            random_state=cfg.random_state,
        )
        self.num_clients = cfg.federated_params.amount_of_clients
        self.server = AutoBANTServer(cfg, trust_df, *self.opt_params)

    def count_trust_scores(self):
        return self.server._count_trust_score()

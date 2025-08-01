from ..fedavg.server import Server
from utils.data_utils import get_dataset_loader


class TrialServer(Server):
    "Abstract Server class for trial function based methods. It adds trust dataset"

    def __init__(self, cfg, trust_df):
        super().__init__(cfg)
        self.trust_df = trust_df
        self.trust_loader = get_dataset_loader(self.trust_df, self.cfg)

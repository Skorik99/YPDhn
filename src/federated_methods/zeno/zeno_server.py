import torch

from ..fedavg.server import Server
from utils.data_utils import get_dataset_loader
from utils.losses import get_loss


class Zeno_server(Server):
    def __init__(self, cfg, trust_df, use_buffers):
        self.trust_loader = None
        super().__init__(cfg)
        self.trust_df = trust_df
        self.trust_loader = get_dataset_loader(self.trust_df, self.cfg)
        self.use_buffers = use_buffers
        self.sds = [0 for _ in range(len(self.client_gradients))]

    def eval_fn(self, mode="test"):
        self.global_model.to(self.device)
        self.global_model.eval()
        self.criterion = get_loss(
            loss_cfg=self.cfg.loss,
            device=self.device,
            df=getattr(self, f"{mode}_df"),
        )

        loss = 0
        fin_targets = []
        fin_outputs = []

        with torch.no_grad():
            for _, batch in enumerate(getattr(self, f"{mode}_loader")):
                _, (input, targets) = batch

                inp = input[0].to(self.device)
                targets = targets.to(self.device)
                outputs = self.global_model(inp)

                loss += self.criterion(outputs, targets)
                fin_targets.extend(targets.tolist())
                fin_outputs.extend(outputs.tolist())

        loss /= len(getattr(self, f"{mode}_loader"))
        setattr(self, f"{mode}_loss", loss)
        return fin_targets, fin_outputs

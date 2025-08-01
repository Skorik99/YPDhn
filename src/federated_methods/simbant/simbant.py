import torch
from collections import OrderedDict

from ..trial.trial import Trial
from ..fltrust.fltrust_server import FLTrustServer
from utils.data_utils import (
    read_dataframe_from_cfg,
    get_stratified_subsample,
    get_dataset_loader,
)
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F


class SimBANT(Trial):
    def __init__(self, trust_sample_amount, prob_temperature, similarity_type):
        super().__init__()
        self.trust_sample_amount = trust_sample_amount
        self.prob_temperature = prob_temperature
        self.similarity_type = similarity_type
        assert self.similarity_type in [
            "cosine",
            "cosine_targets",
        ], f"We support only ['cosine', 'cosine_targets'] similarity metrics, you provide: {self.similarity_type}"

    def _init_server(self, cfg):
        trust_df = read_dataframe_from_cfg(cfg, "train_directories", "trust_df")
        trust_df, self.trust_test_df = get_stratified_subsample(
            df=trust_df,
            num_samples=self.trust_sample_amount,
            random_state=cfg.random_state,
        )
        self.server = FLTrustServer(cfg, trust_df)
        self.num_clients = len(self.server.client_gradients)

    def count_trust_scores(self):
        self.server.fltrust_train()
        targets, server_probs, clients_probs = self.trust_eval_models()
        trust_scores = self.bant_similarity(targets, server_probs, clients_probs)
        return trust_scores

    def trust_eval_models(self):
        def set_gradients(gradients):
            new_weights = OrderedDict()
            for key, weights in gradients.items():
                new_weights[key] = self.server.initial_global_model_state[
                    key
                ] + weights.to(self.server.device)
            self.server.global_model.load_state_dict(new_weights)

        # Set test loader as a trust test part
        self.server.test_loader = get_dataset_loader(
            self.trust_test_df, self.server.cfg, drop_last=False, mode="test"
        )
        # Set server trust train updates
        set_gradients(self.server.server_grad)
        # Get server result
        targets, server_result = self.server.eval_fn()
        # Get probabilities
        server_probs = F.softmax(
            torch.as_tensor(server_result) / self.prob_temperature, dim=-1
        )
        ohe_targets = (
            F.one_hot(torch.tensor(targets), num_classes=server_probs.shape[1])
            .float()
            .to(server_probs.device)
        )
        # Get clients result
        clients_probs = []
        for client_grad in self.server.client_gradients:
            set_gradients(client_grad)
            _, client_result = self.server.eval_fn()
            client_probs = F.softmax(
                torch.as_tensor(client_result) / self.prob_temperature, dim=-1
            )
            clients_probs.append(client_probs)

        # Get back to initial state
        self.server.global_model.load_state_dict(self.server.initial_global_model_state)
        self.server.test_loader = get_dataset_loader(
            self.server.test_df, self.server.cfg, drop_last=False, mode="test"
        )
        return ohe_targets, server_probs, clients_probs

    def bant_similarity(self, targets, server_probs, clients_probs):
        trust_scores = []
        for i in range(self.num_clients):
            if self.similarity_type == "cosine":
                similarity_score = F.cosine_similarity(
                    clients_probs[i], server_probs, dim=1
                )
                mean = similarity_score.mean()
                std = similarity_score.std()
                client_trust_score = max(mean - std, 0.0001)
            if self.similarity_type == "cosine_targets":
                similarity_score = F.cosine_similarity(clients_probs[i], targets, dim=1)
                client_trust_score = similarity_score.mean()

            trust_scores.append(client_trust_score)

        # normalize trust scores
        if self.similarity_type == "cosine":
            trust_scores = [ts / sum(trust_scores) for ts in trust_scores]
        if self.similarity_type == "cosine_targets":
            trust_scores = F.softmax(torch.stack(trust_scores) / 0.05, dim=0)

        return trust_scores

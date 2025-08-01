from ..fedavg.fedavg import FedAvg


class Trial(FedAvg):
    """Abstract class for trial function based methods.
    It overrides `aggregate` method:
    1. Calculate trust scores for clients;
    2. Modifies client updates for FedAvg uniform averaging.

    x^{t+1} = x^t + Σ_i^N(w_i*Δ_i^{t}) => Δ_i^{t} --> Δ_i^{t} * w_i * N =>
    = > x^{t+1} = x^t + 1/N * Σ_i^N (Δ_i^{t})
    """

    def count_trust_scores(self):
        raise NotImplementedError(
            "This function must be implemented in TrialFunction algorithm"
        )

    def _modify_gradients(self, trust_scores):
        num_clients = len(self.server.client_gradients)
        for i in range(num_clients):
            print(f"Client {i} trust score: {trust_scores[i]}")
            modified_client_model_weights = {
                k: v * trust_scores[i] * num_clients
                for k, v in self.server.client_gradients[i].items()
            }
            self.server.client_gradients[i] = modified_client_model_weights

    def aggregate(self):
        trust_scores = self.count_trust_scores()
        self._modify_gradients(trust_scores)
        return super().aggregate()

from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate


class AutoBANTModel(nn.Module):
    def __init__(
        self, cfg, server_model_weights, client_updates, device, init_trust_scores=None
    ):
        super(AutoBANTModel, self).__init__()

        self.amount_of_clients = len(client_updates)
        self.device = device
        if init_trust_scores is None:
            initial_values = torch.tensor(
                [1.0 / self.amount_of_clients] * self.amount_of_clients
            )
        else:
            initial_values = init_trust_scores
        self.trust_scores = nn.Parameter(
            initial_values.clone().detach(), requires_grad=True
        )

        client_states = self.get_client_states(server_model_weights, client_updates)
        self.client_models = self.get_client_models(client_states, cfg)
        self.freeze_client_models_weights()

    def get_client_states(self, server_model_weights, client_updates):
        states = []
        for update in client_updates:
            state = OrderedDict()
            for key, weights1 in update.items():
                state[key] = weights1.to(self.device) + server_model_weights[key]
            states.append(state)
        return states

    def get_client_models(self, client_states, cfg):
        client_models = [
            instantiate(cfg.models[0]).to(self.device)
            for i in range(self.amount_of_clients)
        ]
        for model, state in zip(client_models, client_states):
            model.load_state_dict(state)
        return client_models

    def freeze_client_models_weights(self):
        for model in self.client_models:
            for param in model.parameters():
                param.requires_grad = False


class AutoBANTModel2d(AutoBANTModel):
    def calc_basic_block(self, layers, ts, x):
        # First
        out = sum([ts[i] * layers[i].conv1(x) for i in range(self.amount_of_clients)])
        out = F.relu(
            sum([ts[i] * layers[i].bn1(out) for i in range(self.amount_of_clients)])
        )
        # Second
        out = sum([ts[i] * layers[i].conv2(out) for i in range(self.amount_of_clients)])
        out = sum([ts[i] * layers[i].bn2(out) for i in range(self.amount_of_clients)])
        # Shortcut
        out += self.calc_shortcut(
            [layers[i].shortcut for i in range(self.amount_of_clients)], ts, x
        )
        return F.relu(out)

    def calc_shortcut(self, layers, ts, x):
        if len(layers[0]) == 0:
            return x
        else:
            out = sum([ts[i] * layers[i][0](x) for i in range(self.amount_of_clients)])
            out = sum(
                [ts[i] * layers[i][1](out) for i in range(self.amount_of_clients)]
            )
            return out

    def forward(self, x):
        # to unit simplex
        # trust_scores = torch.softmax(self.trust_scores, dim=0)
        trust_scores = self.trust_scores
        # Init layers
        x = sum(
            [
                trust_scores[i] * self.client_models[i].conv1(x)
                for i in range(self.amount_of_clients)
            ]
        )
        x = F.relu(
            sum(
                [
                    trust_scores[i] * self.client_models[i].bn1(x)
                    for i in range(self.amount_of_clients)
                ]
            )
        )
        # Basic blocks
        for j in range(len(self.client_models[0].layer1)):
            x = self.calc_basic_block(
                [
                    self.client_models[i].layer1[j]
                    for i in range(self.amount_of_clients)
                ],
                trust_scores,
                x,
            )
        for j in range(len(self.client_models[0].layer2)):
            x = self.calc_basic_block(
                [
                    self.client_models[i].layer2[j]
                    for i in range(self.amount_of_clients)
                ],
                trust_scores,
                x,
            )
        for j in range(len(self.client_models[0].layer3)):
            x = self.calc_basic_block(
                [
                    self.client_models[i].layer3[j]
                    for i in range(self.amount_of_clients)
                ],
                trust_scores,
                x,
            )
        for j in range(len(self.client_models[0].layer4)):
            x = self.calc_basic_block(
                [
                    self.client_models[i].layer4[j]
                    for i in range(self.amount_of_clients)
                ],
                trust_scores,
                x,
            )
        # Final layers
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = sum(
            [
                trust_scores[i] * self.client_models[i].linear(x)
                for i in range(self.amount_of_clients)
            ]
        )
        return x


class MemoryEfficientAutoBANTModel(AutoBANTModel2d):
    def __init__(
        self,
        cfg,
        server_model_weights,
        client_updates,
        device,
        init_trust_scores=None,
        batch_size=10,
    ):
        super(AutoBANTModel, self).__init__()

        self.amount_of_clients = len(client_updates)
        self.device = device
        self.batch_size = min(batch_size, self.amount_of_clients)

        # Initialize trust scores
        if init_trust_scores is None:
            initial_values = torch.tensor(
                [1.0 / self.amount_of_clients] * self.amount_of_clients
            )
        else:
            initial_values = init_trust_scores
        self.trust_scores = nn.Parameter(
            initial_values.clone().detach(), requires_grad=True
        )

        # Store client states on CPU
        self.client_states = self.get_client_states(
            server_model_weights, client_updates
        )

        # Create a single template model
        self.template_model = instantiate(cfg.models[0]).to(self.device)

        # Store model configuration for recreation
        self.model_cfg = cfg.models[0]

    def forward(self, x):
        # Initialize accumulator for final output
        result = None

        # Process clients in batches
        for batch_start in range(0, self.amount_of_clients, self.batch_size):
            batch_end = min(batch_start + self.batch_size, self.amount_of_clients)
            batch_size = batch_end - batch_start

            # Create batch models and load their states
            batch_models = []
            for i in range(batch_start, batch_end):
                model = instantiate(self.model_cfg).to(self.device)
                model.load_state_dict(self.client_states[i])
                # Freeze parameters
                for param in model.parameters():
                    param.requires_grad = False
                batch_models.append(model)

            # Process this batch and accumulate results
            batch_result = self.process_batch(x, batch_models, batch_start, batch_end)

            if result is None:
                result = batch_result
            else:
                result += batch_result

            # Delete batch models to free memory
            del batch_models
            torch.cuda.empty_cache()

        return result

    def process_batch(self, x, batch_models, batch_start, batch_end):
        # Get trust scores for this batch
        batch_trust_scores = self.trust_scores[batch_start:batch_end]

        # Process through each layer, accumulating weighted outputs
        # Init layers
        out = torch.zeros_like(self.template_model.conv1(x))
        for i, model in enumerate(batch_models):
            out += batch_trust_scores[i] * model.conv1(x)

        bn_out = torch.zeros_like(self.template_model.bn1(out))
        for i, model in enumerate(batch_models):
            bn_out += batch_trust_scores[i] * model.bn1(out)
        out = F.relu(bn_out)

        # Process through all layers in sequence
        # Layer 1
        out = self.process_layer(
            out, [model.layer1 for model in batch_models], batch_trust_scores
        )
        # Layer 2
        out = self.process_layer(
            out, [model.layer2 for model in batch_models], batch_trust_scores
        )
        # Layer 3
        out = self.process_layer(
            out, [model.layer3 for model in batch_models], batch_trust_scores
        )
        # Layer 4
        out = self.process_layer(
            out, [model.layer4 for model in batch_models], batch_trust_scores
        )

        # Final layers
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)

        final_out = torch.zeros_like(self.template_model.linear(out))
        for i, model in enumerate(batch_models):
            final_out += batch_trust_scores[i] * model.linear(out)

        return final_out

    def process_layer(self, x, layers, batch_trust_scores):
        out = x
        for j in range(len(layers[0])):
            out = self.process_basic_block(
                [layers[i][j] for i in range(len(layers))], batch_trust_scores, out
            )
        return out

    def process_basic_block(self, blocks, batch_trust_scores, x):
        # First
        conv1_out = torch.zeros_like(blocks[0].conv1(x))
        for i, block in enumerate(blocks):
            conv1_out += batch_trust_scores[i] * block.conv1(x)

        bn1_out = torch.zeros_like(blocks[0].bn1(conv1_out))
        for i, block in enumerate(blocks):
            bn1_out += batch_trust_scores[i] * block.bn1(conv1_out)
        out = F.relu(bn1_out)

        # Second
        conv2_out = torch.zeros_like(blocks[0].conv2(out))
        for i, block in enumerate(blocks):
            conv2_out += batch_trust_scores[i] * block.conv2(out)

        bn2_out = torch.zeros_like(blocks[0].bn2(conv2_out))
        for i, block in enumerate(blocks):
            bn2_out += batch_trust_scores[i] * block.bn2(conv2_out)
        out = bn2_out

        # Shortcut
        shortcut = self.process_shortcut(
            [blocks[i].shortcut for i in range(len(blocks))], batch_trust_scores, x
        )
        out += shortcut

        return F.relu(out)

    def process_shortcut(self, shortcuts, batch_trust_scores, x):
        if len(shortcuts[0]) == 0:
            return x
        else:
            sc1_out = torch.zeros_like(shortcuts[0][0](x))
            for i, shortcut in enumerate(shortcuts):
                sc1_out += batch_trust_scores[i] * shortcut[0](x)

            sc2_out = torch.zeros_like(shortcuts[0][1](sc1_out))
            for i, shortcut in enumerate(shortcuts):
                sc2_out += batch_trust_scores[i] * shortcut[1](sc1_out)

            return sc2_out


class AutoBANTModel1d(AutoBANTModel):
    def calc_stem(self, layers, ts, x):
        # MyConv
        out = sum([ts[i] * layers[i][0](x) for i in range(self.amount_of_clients)])
        # Bn
        out = sum([ts[i] * layers[i][1](out) for i in range(self.amount_of_clients)])
        # ReLU
        out = layers[0][2](out)
        # MyMaxPool
        out = sum([ts[i] * layers[i][3](out) for i in range(self.amount_of_clients)])
        return out

    def calc_backbone(self, layers, ts, x):
        for i in range(len(layers[0])):
            for j in range(len(layers[0][0])):
                x = self.calc_basic_block(
                    [layers[p][i][j] for p in range(self.amount_of_clients)], ts, x
                )
        return x

    def calc_basic_block(self, layers, ts, x):
        # print(f"Init shape: {x.size()}")
        identity = x
        # First
        out = sum([ts[i] * layers[i].conv1(x) for i in range(self.amount_of_clients)])
        out = F.relu(
            sum([ts[i] * layers[i].bn1(out) for i in range(self.amount_of_clients)])
        )
        # Check dropout if don't work
        out = sum([ts[i] * layers[i].do1(out) for i in range(self.amount_of_clients)])

        # Second
        out = sum([ts[i] * layers[i].conv2(out) for i in range(self.amount_of_clients)])
        out = F.relu(
            sum([ts[i] * layers[i].bn2(out) for i in range(self.amount_of_clients)])
        )
        # Check dropout if don't work
        out = sum([ts[i] * layers[i].do2(out) for i in range(self.amount_of_clients)])

        # print(f"Out size: {out.size()}")
        # Downsample
        if layers[0].downsample is not None:
            # print("Downsampling...")
            identity = self.calc_downsample(
                [layers[i].downsample for i in range(self.amount_of_clients)],
                ts,
                identity,
            )
            # print(f"Identity size: {identity.size()}")
        # shortcut
        out = out + identity
        return out

    def calc_downsample(self, layers, ts, x):
        out = sum([ts[i] * layers[i][0](x) for i in range(self.amount_of_clients)])
        out = sum([ts[i] * layers[i][1](out) for i in range(self.amount_of_clients)])
        return out

    def calc_head(self, layers, ts, x):
        x = self.calc_pooling_adapter(
            [layers[i].pooling_adapter_head for i in range(self.amount_of_clients)],
            ts,
            x,
        )
        x = self.calc_lin_bn_drop(
            [layers[i].lin_bn_drop_head_final for i in range(self.amount_of_clients)],
            ts,
            x,
        )
        return x

    def calc_pooling_adapter(self, layers, ts, x):
        avg_out = sum(
            [ts[i] * layers[i][0].ap(x) for i in range(self.amount_of_clients)]
        )
        max_out = sum(
            [ts[i] * layers[i][0].mp(x) for i in range(self.amount_of_clients)]
        )
        out = torch.cat([max_out, avg_out], 1)

        out = layers[0][1](out)
        return out

    def calc_lin_bn_drop(self, layers, ts, x):
        out = sum([ts[i] * layers[i][0](x) for i in range(self.amount_of_clients)])
        out = sum([ts[i] * layers[i][1](out) for i in range(self.amount_of_clients)])
        return out

    def forward(self, x):
        trust_scores = self.trust_scores
        x = self.calc_stem(
            [self.client_models[i].stem for i in range(self.amount_of_clients)],
            trust_scores,
            x,
        )
        x = self.calc_backbone(
            [self.client_models[i].backbone for i in range(self.amount_of_clients)],
            trust_scores,
            x,
        )
        x = self.calc_head(
            [self.client_models[i].head for i in range(self.amount_of_clients)],
            trust_scores,
            x,
        )
        return x

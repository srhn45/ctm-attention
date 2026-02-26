import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import math
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download

from models.modules import ParityBackbone, SynapseUNET, Squeeze, SuperLinear, LearnableFourierPositionalEncoding, MultiLearnableFourierPositionalEncoding, CustomRotationalEmbedding, CustomRotationalEmbedding1D, ShallowWide
from models.resnet import prepare_resnet_backbone
from models.utils import compute_normalized_entropy

from models.constants import (
    VALID_NEURON_SELECT_TYPES,
    VALID_BACKBONE_TYPES,
    VALID_POSITIONAL_EMBEDDING_TYPES
)

class ContinuousThoughtMachineAttn(nn.Module, PyTorchModelHubMixin):
    """
    Continuous Thought Machine (CTM) with Attention-based Synchronization.

    Modified to use attention mechanism for computing synchronization instead of
    random sampling of neuron pairs. This preserves more information and uses a
    learnable inductive bias.

    Technical report: https://arxiv.org/abs/2505.05522

    Interactive Website: https://pub.sakana.ai/ctm/

    Blog: https://sakana.ai/ctm/

    Thought takes time and reasoning is a process. 
    
    The CTM consists of three main ideas:
    1. The use of internal recurrence, enabling a dimension over which a concept analogous to thought can occur. 
    1. Neuron-level models, that compute post-activations by applying private (i.e., on a per-neuron basis) MLP 
       models to a history of incoming pre-activations.
    2. Synchronisation as representation, where the neural activity over time is tracked and used to compute how 
       pairs of neurons synchronise with one another over time. This measure of synchronisation is the representation 
       with which the CTM takes action and makes predictions.


    Args:
        iterations (int): Number of internal 'thought' ticks (T, in paper).
        d_model (int): Core dimensionality of the CTM's latent space (D, in paper).
                       NOTE: Note that this is NOT the representation used for action or prediction, but rather that which
                       is fully internal to the model and not directly connected to data.
        d_input (int): Dimensionality of projected attention outputs or direct input features.
        heads (int): Number of attention heads.
        n_synch_out (int): Number of neurons used for output synchronisation (D_out, in paper).
                          NOTE: In attention-based version, this parameter is ignored as we use all neurons.
        n_synch_action (int): Number of neurons used for action/attention synchronisation (D_action, in paper).
                          NOTE: In attention-based version, this parameter is ignored as we use all neurons.
        synapse_depth (int): Depth of the synapse model (U-Net if > 1, else MLP).
        memory_length (int): History length for Neuron-Level Models (M, in paper).
        deep_nlms (bool): Use deeper (2-layer) NLMs if True, else linear.
                        NOTE: we almost always use deep NLMs, but a linear NLM is faster.
        memory_hidden_dims (int): Hidden dimension size for deep NLMs.
        do_layernorm_nlm (bool): Apply LayerNorm within NLMs.
                        NOTE: we never set this to true in the paper. If you set this to true you will get strange behaviour,
                        but you can potentially encourage more periodic behaviour in the dynamics. Untested; be careful.
        backbone_type (str): Type of feature extraction backbone (e.g., 'resnet18-2', 'none').
        positional_embedding_type (str): Type of positional embedding for backbone features.
        out_dims (int): Output dimension size.
                        NOTE: projected from synchronisation!
        prediction_reshaper (list): Shape for reshaping predictions before certainty calculation (task-specific).
                        NOTE: this is used to compute certainty and is needed when applying softmax for probabilities
        dropout (float): Dropout rate.
        neuron_select_type (str): Neuron selection strategy - IGNORED in attention version.
        n_random_pairing_self (int): Number of self-pairing neurons - IGNORED in attention version.
    """                               

    def __init__(self,
                 iterations,
                 d_model,
                 d_input,
                 heads,
                 n_synch_out,
                 n_synch_action,
                 synapse_depth,
                 memory_length,
                 deep_nlms,
                 memory_hidden_dims,
                 do_layernorm_nlm,
                 out_dims,
                 prediction_reshaper=[-1],
                 dropout=0,
                 dropout_nlm=None,
                 backbone_type='resnet18-2',
                 positional_embedding_type='learnable-fourier',
                 neuron_select_type='random-pairing',  
                 n_random_pairing_self=0,
                 num_attn_heads=1
                 ):
        super(ContinuousThoughtMachineAttn, self).__init__()

        # --- Core Parameters ---
        self.iterations = iterations
        self.d_model = d_model
        self.d_input = d_input
        self.memory_length = memory_length
        self.prediction_reshaper = prediction_reshaper
        self.n_synch_out = n_synch_out  # Kept for compatibility, but not used
        self.n_synch_action = n_synch_action  # Kept for compatibility, but not used
        self.backbone_type = backbone_type
        self.out_dims = out_dims
        self.positional_embedding_type = positional_embedding_type
        self.neuron_select_type = neuron_select_type  # Kept for compatibility
        self.memory_length = memory_length
        dropout_nlm = dropout if dropout_nlm is None else dropout_nlm
        
        # --- Attention Parameters ---
        self.temporal_decay = nn.Parameter(torch.zeros(d_model)) # decay term per neuron
        self.wq = nn.Parameter(torch.ones(d_model, d_model))  # scaling for query
        self.W_q = nn.Parameter(torch.Tensor(d_model, d_model))
        self.W_k = nn.Parameter(torch.Tensor(d_model, d_model))
        self.W_v = nn.Parameter(torch.Tensor(d_model, d_model))

        nn.init.xavier_uniform_(self.W_q)
        nn.init.xavier_uniform_(self.W_k)
        nn.init.xavier_uniform_(self.W_v)

        self.attn_dropout_layer = nn.Dropout(dropout)
        self.num_attn_heads = num_attn_heads

        assert d_model % num_attn_heads == 0, "d_model must be divisible by num_attn_heads"
        self.d_head = d_model // num_attn_heads

        # --- Assertions ---
        self.verify_args()

        # --- Input Processing  ---
        d_backbone = self.get_d_backbone()
        self.set_initial_rgb()
        self.set_backbone()
        self.positional_embedding = self.get_positional_embedding(d_backbone)
        self.kv_proj = nn.Sequential(nn.LazyLinear(self.d_input), nn.LayerNorm(self.d_input)) if heads else None
        self.q_proj = nn.LazyLinear(self.d_input) if heads else None
        self.attention = nn.MultiheadAttention(self.d_input, heads, dropout, batch_first=True) if heads else None
        
        # --- Core CTM Modules ---
        self.synapses = self.get_synapses(synapse_depth, d_model, dropout)
        self.trace_processor = self.get_neuron_level_models(deep_nlms, do_layernorm_nlm, memory_length, memory_hidden_dims, d_model, dropout_nlm)

        #  --- Start States ---
        self.register_parameter('start_activated_state', nn.Parameter(torch.zeros((d_model)).uniform_(-math.sqrt(1/(d_model)), math.sqrt(1/(d_model)))))
        self.register_parameter('start_trace', nn.Parameter(torch.zeros((d_model, memory_length)).uniform_(-math.sqrt(1/(d_model+memory_length)), math.sqrt(1/(d_model+memory_length)))))

        # --- Synchronisation (Attention-based) ---
        # Note: In the attention-based version, we use all D neurons, so synchronisation
        # representation size is simply d_model (not a reduced subset)
        self.synch_representation_size_action = d_model
        self.synch_representation_size_out = d_model
        
        print(f"Using attention-based synchronisation with representation size: {d_model}")

        # --- Output Procesing ---
        self.output_projector = nn.Sequential(nn.LazyLinear(self.out_dims))

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision=None,
        cache_dir=None,
        force_download=False,
        proxies=None,
        resume_download=None,
        local_files_only=False,
        token=None,
        map_location="cpu",
        strict=False,
        **model_kwargs,
    ):
        """Override to handle lazy weights initialization."""
        model = cls(**model_kwargs).to(map_location)

        # The CTM contains Lazy modules, so we must run a dummy forward pass to initialize them
        if "imagenet" in model_id:
            dummy_input = torch.randn(1, 3, 224, 224, device=map_location)
        elif "maze-large" in model_id:
            dummy_input = torch.randn(1, 3, 99, 99, device=map_location)
        else:
            raise NotImplementedError

        with torch.no_grad():
            _ = model(dummy_input)

        model_file = hf_hub_download(
            repo_id=model_id,
            filename="model.safetensors",
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            token=token,
            local_files_only=local_files_only,
        )
        from safetensors.torch import load_model as load_model_as_safetensor
        load_model_as_safetensor(model, model_file, strict=strict, device=map_location)

        model.eval()
        return model

    # --- Core CTM Methods ---

    def _split_heads(self, x, batch_size, seq_len):
        """Split the last dimension into (num_heads, d_head)"""
        # x shape: (B, seq_len, D)
        x = x.view(batch_size, seq_len, self.num_attn_heads, self.d_head)
        return x.transpose(1, 2)  # (B, num_heads, seq_len, d_head)

    def compute_attention_synchronisation(self, activated_state_history, current_activated_state):
        """
        Computes synchronisation using attention mechanism over neurons.
        
        Instead of random sampling and cosine similarity, we use attention where:
        - Query & Key: Full history of all neuron post-activations (B, D, T)
        - Value: Current timestep post-activations (B, D, 1)
        
        This allows each neuron to attend to all other neurons based on their
        full temporal history, while only aggregating current-timestep information.
        
        Args:
            activated_state_history: History of post-activations, shape (B, D, T)
            current_activated_state: Current post-activations, shape (B, D)
        
        Returns:
            synchronisation: Attention output, shape (B, D)
        """
        B, D, T = activated_state_history.shape

        # Reshape history: (B, D, T) -> (B, T, D)
        history_reshaped = activated_state_history.transpose(1, 2)  # (B, T, D)
        
        # Prepare Q, K, V for attention

        # Project queries and keys from history
        Q = torch.matmul(history_reshaped, self.W_q.T)  # (B, T, D)
        K = torch.matmul(history_reshaped, self.W_k.T)  # (B, T, D)

        # Current state: (B, D) -> (B, 1, D)
        current_reshaped = current_activated_state.unsqueeze(1)  # (B, 1, D)
        V = torch.matmul(current_reshaped, self.W_v.T) # (B, 1, D)

        # 3. Multi-head reshaping (optional - will be used if num_heads > 1)
        Q = self._split_heads(Q, B, T)  # (B, num_heads, T, d_head)
        K = self._split_heads(K, B, T)  # (B, num_heads, T, d_head)
        V = self._split_heads(V, B, 1)  # (B, num_heads, 1, d_head)

        # 4. Compute attention scores
        # Instead of Q @ K^T, we want cross-neuron attention
        # We need to reshape to get (B, num_heads, D, T) style matrices
        Q_neurons = Q.transpose(2, 3)  # (B, num_heads, d_head, T)
        K_neurons = K  # (B, num_heads, T, d_head)
        
        # Compute attention between neurons based on their temporal patterns
        scores = torch.matmul(Q_neurons, K_neurons) / math.sqrt(self.d_head)  # (B, num_heads, d_head, d_head)

        # 5. Apply attention to values
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.attn_dropout_layer(attention_weights)
        
        # Expand V to match dimensions: (B, num_heads, d_head, 1)
        V_expanded = V.transpose(2, 3)  # (B, num_heads, d_head, 1)
        
        # Apply attention: (B, num_heads, d_head, d_head) @ (B, num_heads, d_head, 1)
        attended = torch.matmul(attention_weights, V_expanded)  # (B, num_heads, d_head, 1)
    
        # 6. Combine heads and produce output
        attended = attended.squeeze(-1).transpose(1, 2)  # (B, d_head, num_heads)
        output = attended.reshape(B, self.d_model)  # (B, D)
        
        return output

    def compute_features(self, x):
        """
        Compute the key-value features from the input data using the backbone. 
        """
        initial_rgb = self.initial_rgb(x)
        self.kv_features = self.backbone(initial_rgb)
        pos_emb = self.positional_embedding(self.kv_features)
        combined_features = (self.kv_features + pos_emb).flatten(2).transpose(1, 2)
        kv = self.kv_proj(combined_features)
        return kv

    def compute_certainty(self, current_prediction):
        """
        Compute the certainty of the current prediction.
        
        We define certainty as being 1-normalised entropy.

        For legacy reasons we stack that in a 2D vector as this can be used for optimisation later.
        """
        B = current_prediction.size(0)
        reshaped_pred = current_prediction.reshape([B] + self.prediction_reshaper)
        ne = compute_normalized_entropy(reshaped_pred)
        current_certainty = torch.stack((ne, 1-ne), -1)
        return current_certainty

    # --- Setup Methods ---

    def set_initial_rgb(self):
        """
        This is largely to accommodate training on grayscale images and is legacy, but it
        doesn't hurt the model in any way that we can tell.
        """
        if 'resnet' in self.backbone_type:
            self.initial_rgb = nn.LazyConv2d(3, 1, 1) # Adapts input channels lazily
        else:
            self.initial_rgb = nn.Identity()

    def get_d_backbone(self):
        """
        Get the dimensionality of the backbone output, to be used for positional embedding setup.

        This is a little bit complicated for resnets, but the logic should be easy enough to read below.        
        """
        if self.backbone_type == 'shallow-wide':
            return 2048
        elif self.backbone_type == 'parity_backbone':
            return self.d_input
        elif 'resnet' in self.backbone_type:
            if '18' in self.backbone_type or '34' in self.backbone_type: 
                if self.backbone_type.split('-')[1]=='1': return 64
                elif self.backbone_type.split('-')[1]=='2': return 128
                elif self.backbone_type.split('-')[1]=='3': return 256
                elif self.backbone_type.split('-')[1]=='4': return 512
                else:
                    raise NotImplementedError
            else:
                if self.backbone_type.split('-')[1]=='1': return 256
                elif self.backbone_type.split('-')[1]=='2': return 512
                elif self.backbone_type.split('-')[1]=='3': return 1024
                elif self.backbone_type.split('-')[1]=='4': return 2048
                else:
                    raise NotImplementedError
        elif self.backbone_type == 'none':
            return None
        else:
            raise ValueError(f"Invalid backbone_type: {self.backbone_type}")

    def set_backbone(self):
        """
        Set the backbone module based on the specified type.
        """
        if self.backbone_type == 'shallow-wide':
            self.backbone = ShallowWide()
        elif self.backbone_type == 'parity_backbone':
            d_backbone = self.get_d_backbone()
            self.backbone = ParityBackbone(n_embeddings=2, d_embedding=d_backbone)
        elif 'resnet' in self.backbone_type:
            self.backbone = prepare_resnet_backbone(self.backbone_type)
        elif self.backbone_type == 'none':
            self.backbone = nn.Identity()
        else:
            raise ValueError(f"Invalid backbone_type: {self.backbone_type}")

    def get_positional_embedding(self, d_backbone):
        """
        Get the positional embedding module.

        For Imagenet and mazes we used NO positional embedding, and largely don't think
        that it is necessary as the CTM can build up its own internal world model when
        observing.

        LearnableFourierPositionalEncoding:
            Implements Algorithm 1 from "Learnable Fourier Features for Multi-Dimensional
            Spatial Positional Encoding" (https://arxiv.org/pdf/2106.02795.pdf).
            Provides positional information for 2D feature maps.      

            (MultiLearnableFourierPositionalEncoding uses multiple feature scales)

        CustomRotationalEmbedding:
            Simple sinusoidal embedding to encourage interpretability
        """
        if self.positional_embedding_type == 'learnable-fourier':
            return LearnableFourierPositionalEncoding(d_backbone, gamma=1 / 2.5)
        elif self.positional_embedding_type == 'multi-learnable-fourier':
            return MultiLearnableFourierPositionalEncoding(d_backbone)
        elif self.positional_embedding_type == 'custom-rotational':
            return CustomRotationalEmbedding(d_backbone)
        elif self.positional_embedding_type == 'custom-rotational-1d':
            return CustomRotationalEmbedding1D(d_backbone)
        elif self.positional_embedding_type == 'none':
            return lambda x: 0  # Default no-op
        else:
            raise ValueError(f"Invalid positional_embedding_type: {self.positional_embedding_type}")

    def get_neuron_level_models(self, deep_nlms, do_layernorm_nlm, memory_length, memory_hidden_dims, d_model, dropout):
        """
        Neuron level models are one of the core innovations of the CTM. They apply separate MLPs/linears to 
        each neuron.
        NOTE: the name 'SuperLinear' is largely legacy, but its purpose is to apply separate linear layers
            per neuron. It is sort of a 'grouped linear' function, where the group size is equal to 1. 
            One could make the group size bigger and use fewer parameters, but that is future work.

        NOTE: We used GLU() nonlinearities because they worked well in practice. 
        """
        if deep_nlms:
            return nn.Sequential(
                nn.Sequential(
                    SuperLinear(in_dims=memory_length, out_dims=2 * memory_hidden_dims, N=d_model,
                                do_norm=do_layernorm_nlm, dropout=dropout),
                    nn.GLU(),
                    SuperLinear(in_dims=memory_hidden_dims, out_dims=2, N=d_model,
                                do_norm=do_layernorm_nlm, dropout=dropout),
                    nn.GLU(),
                    Squeeze(-1)
                )
            )
        else:
            return nn.Sequential(
                nn.Sequential(
                    SuperLinear(in_dims=memory_length, out_dims=2, N=d_model,
                                do_norm=do_layernorm_nlm, dropout=dropout),
                    nn.GLU(),
                    Squeeze(-1)
                )
            )

    def get_synapses(self, synapse_depth, d_model, dropout):
        """
        The synapse model is the recurrent model in the CTM. It's purpose is to share information
        across neurons. If using depth of 1, this is just a simple single layer with nonlinearity and layernomr.
        For deeper synapse models we use a U-NET structure with many skip connections. In practice this performs
        better as it enables multi-level information mixing.

        The intuition with having a deep UNET model for synapses is that the action of synaptic connections is
        not necessarily a linear one, and that approximate a synapose 'update' step in the brain is non trivial. 
        Hence, we set it up so that the CTM can learn some complex internal rule instead of trying to approximate
        it ourselves.
        """
        if synapse_depth == 1:
            return nn.Sequential(
                nn.Dropout(dropout),
                nn.LazyLinear(d_model * 2),
                nn.GLU(),
                nn.LayerNorm(d_model)
            )
        else:
            return SynapseUNET(d_model, synapse_depth, 16, dropout)  # hard-coded minimum width of 16; future work TODO.

    # --- Utilty Methods ---

    def verify_args(self):
        """
        Verify the validity of the input arguments to ensure consistent behaviour. 
        """
        assert self.backbone_type in VALID_BACKBONE_TYPES + ['none'], \
            f"Invalid backbone_type: {self.backbone_type}"
        
        assert self.positional_embedding_type in VALID_POSITIONAL_EMBEDDING_TYPES + ['none'], \
            f"Invalid positional_embedding_type: {self.positional_embedding_type}"

        if self.backbone_type=='none' and self.positional_embedding_type!='none':
            raise AssertionError("There should be no positional embedding if there is no backbone.")




    def forward(self, x, track=False):
        B = x.size(0)
        device = x.device

        # --- Tracking Initialization ---
        pre_activations_tracking = []
        post_activations_tracking = []
        synch_out_tracking = []
        synch_action_tracking = []
        attention_tracking = []

        # --- Featurise Input Data ---
        kv = self.compute_features(x)

        # --- Initialise Recurrent State ---
        state_trace = self.start_trace.unsqueeze(0).expand(B, -1, -1) # Shape: (B, D, M)
        activated_state = self.start_activated_state.unsqueeze(0).expand(B, -1) # Shape: (B, D)
        
        # --- Initialize history of post-activations for attention-based synchronisation ---
        activated_state_history = activated_state.unsqueeze(-1)  # Shape: (B, D, 1)

        # --- Prepare Storage for Outputs per Iteration ---
        predictions = torch.empty(B, self.out_dims, self.iterations, device=device, dtype=torch.float32)
        certainties = torch.empty(B, 2, self.iterations, device=device, dtype=torch.float32)

        # --- Recurrent Loop  ---
        for stepi in range(self.iterations):

            # --- Calculate Synchronisation for Input Data Interaction using Attention ---
            synchronisation_action = self.compute_attention_synchronisation(
                activated_state_history, activated_state
            )

            # --- Interact with Data via Attention ---
            q = self.q_proj(synchronisation_action).unsqueeze(1)
            attn_out, attn_weights = self.attention(q, kv, kv, average_attn_weights=False, need_weights=True)
            attn_out = attn_out.squeeze(1)
            pre_synapse_input = torch.concatenate((attn_out, activated_state), dim=-1)

            # --- Apply Synapses ---
            state = self.synapses(pre_synapse_input)
            # The 'state_trace' is the history of incoming pre-activations
            state_trace = torch.cat((state_trace[:, :, 1:], state.unsqueeze(-1)), dim=-1)

            # --- Apply Neuron-Level Models ---
            activated_state = self.trace_processor(state_trace)
            
            # --- Update history of post-activations ---
            #activated_state_history = torch.cat(
            #    (activated_state_history, activated_state.unsqueeze(-1)), dim=-1
            #)  # Shape: (B, D, T+1)

            # Apply temporal decay to the history
            # decay factor in (0, 1)
            decay = torch.exp(-self.temporal_decay).view(1, -1, 1)  # (1, D, 1)

            # apply EMA decay to entire history
            activated_state_history = activated_state_history * decay

            # append current activation (no decay applied to new step)
            activated_state_history = torch.cat(
                (activated_state_history, activated_state.unsqueeze(-1)), dim=-1
            ) # Shape: (B, D, T+1)

            if activated_state_history.size(-1) > self.memory_length:
                activated_state_history = activated_state_history[:, :, -self.memory_length:]

            # --- Calculate Synchronisation for Output Predictions using Attention ---
            # Note: We use the same synchronisation for both action and output
            synchronisation_out = self.compute_attention_synchronisation(
                activated_state_history, activated_state
            )

            # --- Get Predictions and Certainties ---
            current_prediction = self.output_projector(synchronisation_out)
            current_certainty = self.compute_certainty(current_prediction)

            predictions[..., stepi] = current_prediction
            certainties[..., stepi] = current_certainty

            # --- Tracking ---
            if track:
                pre_activations_tracking.append(state_trace[:,:,-1].detach().cpu().numpy())
                post_activations_tracking.append(activated_state.detach().cpu().numpy())
                attention_tracking.append(attn_weights.detach().cpu().numpy())
                synch_out_tracking.append(synchronisation_out.detach().cpu().numpy())
                synch_action_tracking.append(synchronisation_action.detach().cpu().numpy())

        # --- Return Values ---
        if track:
            return predictions, certainties, (np.array(synch_out_tracking), np.array(synch_action_tracking)), np.array(pre_activations_tracking), np.array(post_activations_tracking), np.array(attention_tracking)
        return predictions, certainties, synchronisation_out
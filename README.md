# CTM with Attention-Based Synchronisation

A fork of [Sakana AI's Continuous Thought Machine](https://github.com/SakanaAI/continuous-thought-machines) that replaces random neuron-pair sampling for synchronisation with a learnable attention mechanism.

**Original paper:** [Continuous Thought Machines](https://arxiv.org/abs/2505.05522)  
**Original repo:** [SakanaAI/continuous-thought-machines](https://github.com/SakanaAI/continuous-thought-machines)

---

## Motivation

In the original CTM, synchronisation representations (S_out and S_action) are computed by randomly sampling pairs of neurons (i, j) and measuring the cosine similarity of their activation histories. This is effective but throws away information — the sampling is random, so potentially important neuron-pair relationships can be missed.

The core question motivating this fork: **can we compute synchronisation using all neurons while keeping a learnable inductive bias?**

---

## The Change: Attention Over Neuron History

Instead of randomly sampling neuron pairs and computing cosine similarity, this implementation computes synchronisation via an attention mechanism over the full activation history.

**The intuition:** treat each neuron as a token, and its history of post-activations as its embedding. This lets every neuron attend to every other neuron based on their full temporal pattern — a direct generalisation of pairwise cosine similarity into a learnable form.

**How the attention is structured:**
- **Query & Key:** Projected from the full history of post-activations `(B, D, T)`, reshaped to `(B, T, D)`
- **Value:** Only the *current* timestep's post-activations `(B, D)`, reshaped to `(B, 1, D)`

Using only the current timestep as Value means the output is a `(B, D)` vector — a drop-in replacement for S_out and S_action with no downstream changes needed.

**Temporal decay:** A per-neuron learnable decay parameter `exp(-λ)` is applied to the activation history before feeding it into attention, giving the model control over how much weight to place on older activations. This replaces the need for a hard memory cutoff as the primary recency mechanism.

**Multi-head support:** The attention block supports multiple heads, grouping neurons into subsets that attend within themselves — analogous to region-level specialisation in the brain, and a practical way to manage the O(D²) complexity of full cross-neuron attention.

The key implementation lives in `compute_attention_synchronisation` inside `ContinuousThoughtMachineAttn` (`models/ctm_attn.py`).

---

## Results

All hyperparameters were kept identical to the baselines in the original example notebooks, except where noted.

| Task | Baseline Train Acc | CTM_Attn Train Acc | Baseline Test Acc | CTM_Attn Test Acc |
|------|-------------------|--------------------|--------------------|-------------------|
| MNIST | 96.9% | **99.2%** | 96.8% | **99.0%** |
| Maze | 70.3% | **84.4%** | **50.8%** | 44.6% |

MNIST used 4 attention heads (~1M params); Maze used 8 attention heads (~13M params). Both used `resnet18-2` backbone and `learnable-fourier` positional embeddings.

The MNIST improvement is consistent across train and test. The maze result shows strong training gains; the test gap likely reflects the added parameters and could likely be addressed with an increase in data.

> Note: a proper ablation (matched parameter count) would be needed to isolate the contribution of the attention mechanism itself vs. the additional parameters.

---

## Usage

The `ContinuousThoughtMachineAttn` class is a drop-in replacement for the original `ContinuousThoughtMachine`. The interface and return values are identical.

```python
from models.ctm_attn import ContinuousThoughtMachineAttn

model = ContinuousThoughtMachineAttn(
    iterations=10,
    d_model=256,
    d_input=64,
    heads=8,
    n_synch_out=64,       # kept for compatibility, not used
    n_synch_action=64,    # kept for compatibility, not used
    synapse_depth=3,
    memory_length=10,
    deep_nlms=True,
    memory_hidden_dims=128,
    do_layernorm_nlm=False,
    out_dims=10,
    backbone_type='resnet18-2',
    positional_embedding_type='learnable-fourier',
    num_attn_heads=4,     # new: number of heads for synchronisation attention
)

predictions, certainties, synch_out = model(x)
```

The existing example notebooks from the original repo work without modification — just swap the import and add `num_attn_heads` to the config.

---

## Key Parameters

| Parameter | Description |
|-----------|-------------|
| `num_attn_heads` | Number of heads in the synchronisation attention block. Must divide `d_model`. |
| `temporal_decay` | Learnable per-neuron decay applied to activation history (initialised to 0, i.e. no decay). |

All other parameters are inherited from the original CTM and are documented there.

---

## Possible Future Directions

- **Ablation study** with matched parameter counts to isolate attention's contribution
- **Bottlenecking cross-neuron attention** — grouping neurons or projecting through an MLP before attention to reduce O(D²) cost
- **Separate S_out and S_action** — currently both use the same attention output; decoupling them may help on tasks where action and prediction benefit from different synchronisation signals
- **Asymmetric Q/K history windows** — using different history lengths for query vs key projections

---

## Citation

If you use this work, please also cite the original CTM paper:

```bibtex
@article{sakana2025ctm,
  title={Continuous Thought Machines},
  author={Sakana AI},
  journal={arXiv preprint arXiv:2505.05522},
  year={2025}
}
```
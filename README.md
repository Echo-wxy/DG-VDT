# DG-VDT

**Graph-Conditioned Reinforcement Learning for Ethereum Smart-Contract Vulnerability Detection and Traceability**

This repository contains the code, data references, and pre-trained model weights for the DG-VDT framework — a reinforcement-learning approach for low-latency vulnerability detection and attacker traceability in Ethereum smart contracts. It accompanies the manuscript submitted to *PeerJ Computer Science*.

---

## Description

DG-VDT reformulates Ethereum smart-contract vulnerability detection and attacker traceability as joint reasoning over a heterogeneous multi-graph abstraction of EVM execution traces. Each trace is decomposed into three complementary graphs (Fund Flow, Contract Creation, Contract Call), and a policy network is trained by pure reinforcement learning (Group Relative Policy Optimization) under a dual-stage progressive reward: an embedding-similarity signal that promotes broad structural exploration transitions, through a sigmoid curriculum, to VF2 subgraph-isomorphism verification for exact signature matching. No supervised fine-tuning is used.

The framework is scoped to three structurally distinct vulnerability categories: **reentrancy**, **short-address**, and **timestamp-dependence**.

---

## Dataset Information

All datasets used in this study are released or referenced with an explicit **DOI or persistent URL** and are distributed in **human- and machine-readable formats** (JSON / JSONL / CSV / `.sol` source files), so that reviewers, replicators, and downstream researchers can access every data source directly from the tables below.

### 1. Author-constructed datasets (released with this repository)

All six author-constructed datasets are deposited on Zenodo under a single record — **DOI: [10.5281/zenodo.20848392](https://doi.org/10.5281/zenodo.20848392)** — and released under CC BY 4.0.

| Dataset | Size | Description | DOI / URL | Format | License |
|---|---|---|---|---|---|
| **BlockTrace-500k** | 500,000 traces | Ethereum mainnet execution traces (blocks 15,000,000–19,500,000, July 2022 – March 2024); used for GRPO training | [10.5281/zenodo.20848392](https://doi.org/10.5281/zenodo.20848392) | JSONL | CC BY 4.0 |
| **EthVulBench** | 4,500 transactions | Curated evaluation set across three vulnerability categories (reentrancy 1,800; short-address 1,350; timestamp-dependence 1,350) | [10.5281/zenodo.20848392](https://doi.org/10.5281/zenodo.20848392) | JSON + CSV | CC BY 4.0 |
| **Etherscan-Public-1k** | 1,024 contracts | Independent validation set from the Etherscan verified-contracts registry (Feb 2024 snapshot) | [10.5281/zenodo.20848392](https://doi.org/10.5281/zenodo.20848392) | `.sol` + JSON | CC BY 4.0 |
| **ToolGen-Block** | 2,500 instances | Blockchain-specific function-calling benchmark following the BFCL annotation schema | [10.5281/zenodo.20848392](https://doi.org/10.5281/zenodo.20848392) | JSONL | CC BY 4.0 |
| **ScamTrace-2024** | 1,200 traces | Real-world attack traces with ground-truth exploiter addresses (sourced from DeFiHackLabs) | [10.5281/zenodo.20848392](https://doi.org/10.5281/zenodo.20848392) | JSONL | CC BY 4.0 |
| **CrossChainAtt** | 600 scenarios | Cross-chain bridge attack scenarios (Ronin, Nomad, Wormhole) | [10.5281/zenodo.20848392](https://doi.org/10.5281/zenodo.20848392) | JSON | CC BY 4.0 |

### 2. Third-party datasets (used for zero-shot evaluation, NOT redistributed)

All third-party datasets are accessed through their **original public repositories**; only the persistent source URLs are given here so that reviewers can obtain the exact versions used in this study. No third-party data is redistributed with this repository.

| Dataset | Description | DOI / URL (main text) | Format | Original license |
|---|---|---|---|---|
| **SolidiFI-Bench** | Injected-bug Solidity benchmark for detector evaluation | <https://github.com/DependableSystemsLab/SolidiFI-benchmark> | `.sol` + JSON (bug reports) | MIT |
| **SmartBugs Curated** | 143 hand-curated vulnerable Solidity contracts across 10 DASP categories | <https://github.com/smartbugs/smartbugs-curated> | `.sol` + YAML metadata | MIT |
| **SmartBugs Wild** | ~47k real-world Solidity contracts scraped from Etherscan | <https://github.com/smartbugs/smartbugs-wild> | `.sol` | MIT |

### 3. Additional public data sources referenced in the study

The following resources are cited in the manuscript as **reference specifications, APIs, or evidence sources**. They are not used as training or evaluation data; the URLs are provided here for full traceability.

| Resource | Role in the study | URL | Format |
|---|---|---|---|
| BFCL annotation schema | Reference schema for the `ToolGen-Block` annotation format | <https://gorilla.cs.berkeley.edu/leaderboard.html> | HTML / JSON |
| Etherscan API | Source for verified contract source code and transaction traces | <https://docs.etherscan.io/> | JSON (REST API) |
| SWC Registry | Canonical taxonomy of smart-contract weaknesses used to align vulnerability labels | <https://swcregistry.io/> | HTML / Markdown |
| DeFiHackLabs | Reproduced real-world exploits used as ground truth for `ScamTrace-2024` | <https://github.com/SunWeb3Sec/DeFiHackLabs> | `.sol` + Markdown |
| Ronin post-mortem | Attack narrative used to construct one `CrossChainAtt` scenario | <https://roninchain.com/blog/posts/back-to-building-ronin-security-breach-6513cc78a5edc1001b03c364> | HTML |
| Nomad analysis | Attack narrative used to construct one `CrossChainAtt` scenario | <https://medium.com/immunefi/hack-analysis-nomad-bridge-august-2022-5aa63d53814a> | HTML |
| Wormhole analysis | Attack narrative used to construct one `CrossChainAtt` scenario | <https://www.halborn.com/blog/post/explained-the-wormhole-hack-february-2022> | HTML |

---

## Code Information

The repository is organized around the four functional components of DG-VDT:

- **Graph extraction** — parsing EVM traces into FFG / CCG / CaG graphs and producing per-window unified snapshots.
- **RGCN graph encoder $\Phi$** — heterogeneous relational graph convolutional encoder (2 layers, hidden dim 256, three relation types: ETH-transfer, function-call, contract-create), pre-trained with a contrastive objective on a held-out 10% split of BlockTrace-500k.
- **VF2 subgraph-isomorphism matcher** — Stage-2 exact signature verification against the reference graphs $G^*$ of each vulnerability category.
- **GRPO training loop** — pure-RL policy optimization with group size $N=8$, dual-stage progressive reward, sigmoid curriculum ($k=0.3$, $m=8\mathrm{k}$ steps), and PPO-style clipping ($\varepsilon=0.2$).

Evaluation scripts and reproducibility artefacts (McNemar / Holm–Bonferroni significance tests, seed control) are also included in the Zenodo release.

---

## Requirements

- Python 3.10+
- PyTorch 2.1+ with CUDA 12.1
- transformers 4.40+
- torch-geometric 2.5+ (for the RGCN encoder)
- networkx 3.2 (for VF2 subgraph isomorphism)
- statsmodels 0.14 (for McNemar / Holm–Bonferroni tests)
- datasets, accelerate

**Hardware used in the paper**

- Training: 4×A100-80GB (DG-VDT-7B, ~48 GPU-hours); 8×A100-80GB (DG-VDT-32B, ~180 GPU-hours)
- Inference: single NVIDIA RTX 3090 (24 GB VRAM)

---

## Usage Instructions

```bash
# 1. Download the Zenodo release and place the datasets under data/
#    DOI: https://doi.org/10.5281/zenodo.20848392

# 2. Install dependencies
pip install -r requirements.txt

# 3. Pre-train the RGCN graph encoder Phi
python rgcn/pretrain.py --config configs/rgcn.yaml

# 4. GRPO training (DG-VDT-7B)
python grpo/train.py --config configs/dgvdt_7b.yaml

# 5. Evaluate on all six benchmarks
bash eval/run_all.sh
```

Reference-graph specifications $G^*$ for the three vulnerability categories (node/edge counts, adjacency lists, node-feature vectors) are included in the Zenodo deposit.

---

## Methodology

The methodological details below (Sections 1–9) are the technical description of DG-VDT. For the full empirical evaluation, statistical analyses, and ablations, please refer to the accompanying manuscript.

### 1. Problem and Motivation

Two things are simultaneously hard on Ethereum: **detecting vulnerabilities precisely** and **tracing attackers reliably**. The two dominant existing lines each have a structural weakness:

- **Tool-augmented LLMs**: trained by supervised fine-tuning (SFT) to imitate large corpora of detection trajectories. They do well *within* the training distribution, but their learning is essentially imitation of surface statistical patterns rather than the causal structure of attacks, so they degrade sharply on structurally novel exploits (new reentrancy variants, cross-chain bridge attacks, etc.).
- **Reward-signal-based RL**: rewards are typically defined over "was the tool call correct / is the output format valid", which lacks any structural understanding of inter-contract interaction topology.

A characteristic failure mode: an SFT model treats the *cyclic fund-flow* topology as the signature of malicious activity, so it achieves high recall on cyclic reentrancy samples but collapses on **timestamp-dependence** attacks whose traces have no cyclic value-transfer structure. The discriminative signal (block-number / timestamp comparisons) is right there in the trace, but the model cannot see it because it over-relies on superficial topology.

**Central thesis**: relational graph structure encodes far richer semantics than tool-invocation counts or output formats; and a **progressive curriculum** is needed to reconcile the breadth-versus-precision tension inherent in open-domain security detection.

### 2. Overview

DG-VDT reformulates "vulnerability detection + attacker traceability" as **joint reasoning over a heterogeneous multi-graph abstraction of execution traces**, driven by a **graph-structured reward** under pure RL (no SFT pretraining). Three interlocking design decisions:

1. **An EVM-instrumented multi-graph representation** — decompose on-chain activity into three semantically complementary graphs;
2. **A dual-stage progressive graph-guided reward** — broad exploration early, strict matching late, with a smooth sigmoid curriculum between them;
3. **A GRPO policy optimizer** — train an LLM backbone with Group Relative Policy Optimization.

It targets three structurally distinct vulnerability classes: **reentrancy**, **short-address**, and **timestamp-dependence**.

### 3. Multi-Graph Representation: One Trace into Three Graphs

Each EVM execution trace is decomposed into three directed graphs that model complementary dimensions of blockchain interaction:

| Graph | Symbol | Captured semantics | Edge meaning |
|---|---|---|---|
| Fund Flow Graph | **FFG** $G_f$ | economic topology of a transaction | Ether transfers between addresses (amount, direction) |
| Contract Creation Graph | **CCG** $G_c$ | contract deployment provenance | deployer -> deployed contract |
| Contract Call Graph | **CaG** $G_\text{call}$ | cross-contract function calls (opcode granularity) | call order, invocation type |

The three are aggregated into a unified snapshot $G_\text{obs} = \{G_f, G_c, G_\text{call}\}$.

**Node types** (address taxonomy + auxiliary trace nodes): EOA (externally owned account), Contract, Miner (block producer), plus auxiliary nodes for state / opcode / branch / transfer-sink.
**Edge types** (interaction ontology): `ETH` (Ether transfer), `CALL` (opcode-level invocation), `CREATE` (deployment), `SLOAD` / `SSTORE` (storage access).

This multi-graph abstraction simultaneously exposes economic topology, deployment provenance, and opcode-level call semantics — exactly what per-contract or single-graph methods leave inaccessible.

### 4. Reference Attack Signatures $G^*$

Each vulnerability class is associated with a canonical **reference graph** $G^*$ that captures its minimal discriminating structure. The structural specifications:

- **Reentrancy $G^*_\text{RE}$ (7 nodes / 8 edges)**: an attacker EOA, a victim contract, an attacker proxy contract, plus balance-ledger state nodes. The discriminating core is the **fund cycle in the FFG** (proxy -> victim -> proxy) together with the **corresponding CALL re-entry edge in the CaG**.
- **Short-address $G^*_\text{SA}$ (5 nodes / 5 edges)**: an attacker EOA, a victim ERC-20 contract, a short recipient address, plus ABI-decoder state nodes. The discriminating core is a CALL edge in the CaG whose **ABI-encoded argument length is < 32 bytes** — the RGCN encoder maps this truncated argument length to an embedding distinct from a valid 32-byte transfer.
- **Timestamp-dependence $G^*_\text{TD}$ (6 nodes / 7 edges)**: a miner account (a high out-degree hub in the CCG), a lottery contract, a TIMESTAMP opcode node, a conditional branch node, a transfer sink, and a balance state node. The discriminating core is the **`TIMESTAMP -> TRANSFER` causal chain in the CaG**, co-occurring with the **miner-account hub in the CCG**.

Signatures are selected by a **median-size** strategy: from the pool of confirmed-positive graphs of a class, take the connected graph whose node count is the median. Compared with min / max / mean size, the median is more robust to outliers in the size distribution — the minimum graph *under-constrains* (insufficient discriminative power), the maximum graph *over-constrains* (rare but valid attack variants fail to match), and the median sits at the optimum between the two.

### 5. Graph-Guided Reward Mechanism

The overall training signal decomposes as:

$$R_\text{final} = R_\text{format} + R_\text{graph}$$

where $R_\text{format} \in \{0,1\}$ is a **prerequisite gate** that checks whether the output satisfies the predefined syntax (`<vulnerability>TYPE</vulnerability>` and `<trace>ADDRESS</trace>`), stabilizing output formatting throughout training. The informationally dominant component is $R_\text{graph}$, which works through the dual-stage progressive mechanism below.

#### 5.1 Stage 1: Broad Graph-Pattern Exploration (soft reward)

Early training is governed by an **embedding-similarity reward** designed to encourage broad coverage of the graph-topology space rather than premature specialization. The observed snapshot $G_\text{obs}$ and the reference graph $G^*$ are both projected into $\mathbb{R}^{256}$ by a heterogeneous graph encoder $\Phi$, and their cosine similarity is taken:

$$r_\text{general} = -0.5 + \frac{\mathbf{z}_\text{obs} \cdot \mathbf{z}^*}{\lVert \mathbf{z}_\text{obs}\rVert\, \lVert \mathbf{z}^*\rVert}$$

Partial reward for topological proximity drives the policy to explore structural variants (e.g., incomplete reentrancy cycles), deferring strict precision to Stage 2.

#### 5.2 Stage 2: Exact Signature Verification (hard reward)

As the policy matures, the reward switches to a strict binary criterion grounded in **subgraph isomorphism**. The VF2 algorithm determines whether the signature $G^*$ appears inside $G_\text{obs}$ under the label assignment (node = address category, edge = interaction semantics):

$$r_\text{strict} = \begin{cases} 1 & \text{if } G^* \text{ is subgraph-isomorphic into } G_\text{obs} \\ 0 & \text{otherwise} \end{cases}$$

The dichotomous signal removes partial-credit ambiguity and provides an unambiguous learning signal for exact pattern matching.

#### 5.3 Curriculum Scheduling: Smooth Sigmoid Transition

The two stages are blended through a continuous sigmoid schedule:

$$R_\text{graph} = \sigma(t)\, r_\text{strict} + \bigl(1 - \sigma(t)\bigr)\, r_\text{general} + R_\text{collab} + R_\text{param}$$

$$\sigma(t) = \bigl(1 + e^{-k(t - m)}\bigr)^{-1}$$

with $k$ controlling transition steepness and $m$ the midpoint step. Unlike a step function, the sigmoid ramp avoids abrupt gradient perturbations at the curriculum boundary.

#### 5.4 Two Auxiliary Terms

- **Multi-graph collaboration bonus** $R_\text{collab} = +0.3$: awarded when corroborating anomalies are detected *simultaneously* across multiple graph views (e.g., a fund cycle in the FFG co-occurring with a circular call sequence in the CaG).
- **Parameter error penalty** $R_\text{param} = -0.3$ (per misaligned numeric field): incentivizes the policy to attend to fine-grained numerical context beyond what topological matching captures.

### 6. Heterogeneous Graph Encoder $\Phi$

$\Phi$ is an extension of the Relational Graph Convolutional Network (**RGCN**) that encodes a graph into a 256-d vector; relations are the edge types, and node features are one-hot node types.

- **Pre-training**: trained on an independent data split with a **graph-contrastive** objective. Positive pairs share the same vulnerability label; hard negatives are the top-$k$ nearest dissimilar-class graphs by cosine distance.
- **Frozen**: after pre-training, $\Phi$ is **kept frozen** throughout GRPO. If $\Phi$ were updated jointly with the policy $\pi_\theta$, the cosine-similarity reward would become a moving target, amplifying gradient variance and introducing reward non-stationarity.

### 7. Policy Optimization: GRPO

The policy network $\pi_\theta$ is optimized with **Group Relative Policy Optimization (GRPO)**. For each query $q$, a group of $N$ candidate completions is sampled from the reference policy, and the objective is

$$\mathcal{J}_\text{GRPO}(\theta) = \mathbb{E}\left[\frac{1}{N}\sum_{i=1}^{N}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \min\!\bigl(\rho_{i,t} A_i,\ \text{clip}(\rho_{i,t}, 1-\varepsilon, 1+\varepsilon) A_i\bigr)\right]$$

where the importance ratio is $\rho_{i,t} = \pi_\theta(o_{i,t}\mid q, o_{i,<t}) / \pi_{\theta_\text{old}}(o_{i,t}\mid q, o_{i,<t})$ and the group-normalized advantage is $A_i = (r_i - \bar{r}) / \text{std}(r)$.

Key design choices:

- **No SFT pretraining** — pure RL, following the R1-style paradigm;
- **KL regularization omitted** — policy drift is implicitly constrained by the clip bound;
- **Reward-collapse guard**: under GRPO, the binary $r_\text{strict}$ can produce *degenerate batches* where all $N$ group samples receive the same reward, yielding zero group-normalized advantage and a vacuous gradient. In that case the gradient from $R_\text{format}$, $R_\text{collab}$, and $R_\text{param}$ (which remain non-constant within the group) is retained, ensuring a non-trivial learning signal.

### 8. Output Format and Vulnerability Classes

The policy must emit markup conforming to the following schema:

```
<vulnerability>TYPE</vulnerability>
<trace>ADDRESS</trace>
```

- `TYPE` in { `reentrancy`, `short_address`, `timestamp` }
- `ADDRESS` is a `0x`-prefixed 40-hex-digit address (the attacker-traceability result)

The predicted vulnerability type selects which reference signature $G^*$ the sample is scored against — this is the mechanism that anchors the textual prediction in graph structure.

### 9. Notation

| Symbol | Meaning |
|---|---|
| $G_\text{obs}$ | unified observed snapshot within a time window, $\{G_f, G_c, G_\text{call}\}$ |
| $G^*$ | canonical reference attack signature of a vulnerability class |
| $\Phi$ | heterogeneous graph encoder (RGCN extension), graph -> $\mathbb{R}^{256}$ |
| $\mathbf{z}_\text{obs}, \mathbf{z}^*$ | graph-level embeddings of $G_\text{obs}$ and $G^*$ |
| $\sigma(t)$ | curriculum sigmoid, interpolating between the two stages over training step $t$ |
| $\pi_\theta$ | the policy network being optimized |
| $N$ | number of candidate completions per query in GRPO |

### In One Sentence

> Decompose an on-chain trace into three complementary graphs (fund-flow / creation / call), and use a sigmoid-curriculum reward that first explores broadly via graph-embedding cosine similarity and then verifies strictly via VF2 subgraph isomorphism, to train an LLM with pure RL (GRPO, no SFT, no KL) so that it both generalizes across structurally heterogeneous attack patterns and precisely classifies vulnerabilities while tracing the attacker.

---

## Citation

If you use DG-VDT, the datasets, or the code in this repository, please cite the accompanying manuscript:

```bibtex
@article{sun2026dgvdt,
  title   = {Graph-Conditioned Reinforcement Learning for Ethereum Smart-Contract Vulnerability Detection and Traceability},
  author  = {Sun, Shiman and Jiang, Wenbao},
  journal = {PeerJ Computer Science},
  year    = {2026},
  note    = {Submitted}
}
```

Please also cite the Zenodo deposit for the datasets and code:

```bibtex
@dataset{sun2026dgvdt_zenodo,
  title     = {DG-VDT: Datasets, code, and pre-trained models for graph-conditioned RL vulnerability detection},
  author    = {Sun, Shiman and Jiang, Wenbao},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20848392},
  url       = {https://doi.org/10.5281/zenodo.20848392}
}
```

---

## License

- **Source code**: MIT License
- **Author-constructed datasets and pre-trained model weights** (via Zenodo): CC BY 4.0
- **Third-party datasets** (SolidiFI-Bench, SmartBugs Curated, SmartBugs Wild): governed by their original licenses; not redistributed here

---

## Contact

For questions about the code or datasets, please contact the corresponding author:

**Wenbao Jiang** — School of Computer Science, Beijing Information Science & Technology University, No.55 Taihe Road, Changping District, Beijing 102206, China
Email: 2024021073@bistu.edu.cn

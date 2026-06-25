# DG-VDT

**Dynamic Graph-guided Vulnerability Detection and Traceability**

A reinforcement-learning framework for low-latency vulnerability detection and attacker traceability in Ethereum smart contracts. This document describes the core idea, the representation design, and the training mechanism of the method.

---

## 1. Problem and Motivation

Two things are simultaneously hard on Ethereum: **detecting vulnerabilities precisely** and **tracing attackers reliably**. The two dominant existing lines each have a structural weakness:

- **Tool-augmented LLMs**: trained by supervised fine-tuning (SFT) to imitate large corpora of detection trajectories. They do well *within* the training distribution, but their learning is essentially imitation of surface statistical patterns rather than the causal structure of attacks, so they degrade sharply on structurally novel exploits (new reentrancy variants, cross-chain bridge attacks, etc.).
- **Reward-signal-based RL**: rewards are typically defined over "was the tool call correct / is the output format valid", which lacks any structural understanding of inter-contract interaction topology.

A characteristic failure mode: an SFT model treats the *cyclic fund-flow* topology as the signature of malicious activity, so it achieves high recall on cyclic reentrancy samples but collapses on **timestamp-dependence** attacks whose traces have no cyclic value-transfer structure. The discriminative signal (block-number / timestamp comparisons) is right there in the trace, but the model cannot see it because it over-relies on superficial topology.

**Central thesis**: relational graph structure encodes far richer semantics than tool-invocation counts or output formats; and a **progressive curriculum** is needed to reconcile the breadth-versus-precision tension inherent in open-domain security detection.

---

## 2. Overview

DG-VDT reformulates "vulnerability detection + attacker traceability" as **joint reasoning over a heterogeneous multi-graph abstraction of execution traces**, driven by a **graph-structured reward** under pure RL (no SFT pretraining). Three interlocking design decisions:

1. **An EVM-instrumented multi-graph representation** — decompose on-chain activity into three semantically complementary graphs;
2. **A dual-stage progressive graph-guided reward** — broad exploration early, strict matching late, with a smooth sigmoid curriculum between them;
3. **A GRPO policy optimizer** — train an LLM backbone with Group Relative Policy Optimization.

It targets three structurally distinct vulnerability classes: **reentrancy**, **short-address**, and **timestamp-dependence**.

---

## 3. Multi-Graph Representation: One Trace into Three Graphs

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

---

## 4. Reference Attack Signatures $G^*$

Each vulnerability class is associated with a canonical **reference graph** $G^*$ that captures its minimal discriminating structure. The structural specifications:

- **Reentrancy $G^*_\text{RE}$ (7 nodes / 8 edges)**: an attacker EOA, a victim contract, an attacker proxy contract, plus balance-ledger state nodes. The discriminating core is the **fund cycle in the FFG** (proxy -> victim -> proxy) together with the **corresponding CALL re-entry edge in the CaG**.
- **Short-address $G^*_\text{SA}$ (5 nodes / 5 edges)**: an attacker EOA, a victim ERC-20 contract, a short recipient address, plus ABI-decoder state nodes. The discriminating core is a CALL edge in the CaG whose **ABI-encoded argument length is < 32 bytes** — the RGCN encoder maps this truncated argument length to an embedding distinct from a valid 32-byte transfer.
- **Timestamp-dependence $G^*_\text{TD}$ (6 nodes / 7 edges)**: a miner account (a high out-degree hub in the CCG), a lottery contract, a TIMESTAMP opcode node, a conditional branch node, a transfer sink, and a balance state node. The discriminating core is the **`TIMESTAMP -> TRANSFER` causal chain in the CaG**, co-occurring with the **miner-account hub in the CCG**.

Signatures are selected by a **median-size** strategy: from the pool of confirmed-positive graphs of a class, take the connected graph whose node count is the median. Compared with min / max / mean size, the median is more robust to outliers in the size distribution — the minimum graph *under-constrains* (insufficient discriminative power), the maximum graph *over-constrains* (rare but valid attack variants fail to match), and the median sits at the optimum between the two.

---

## 5. Graph-Guided Reward Mechanism

The overall training signal decomposes as:

$$R_\text{final} = R_\text{format} + R_\text{graph}$$

where $R_\text{format} \in \{0,1\}$ is a **prerequisite gate** that checks whether the output satisfies the predefined syntax (`<vulnerability>TYPE</vulnerability>` and `<trace>ADDRESS</trace>`), stabilizing output formatting throughout training. The informationally dominant component is $R_\text{graph}$, which works through the dual-stage progressive mechanism below.

### 5.1 Stage 1: Broad Graph-Pattern Exploration (soft reward)

Early training is governed by an **embedding-similarity reward** designed to encourage broad coverage of the graph-topology space rather than premature specialization. The observed snapshot $G_\text{obs}$ and the reference graph $G^*$ are both projected into $\mathbb{R}^{256}$ by a heterogeneous graph encoder $\Phi$, and their cosine similarity is taken:

$$r_\text{general} = -0.5 + \frac{\mathbf{z}_\text{obs} \cdot \mathbf{z}^*}{\lVert \mathbf{z}_\text{obs}\rVert\, \lVert \mathbf{z}^*\rVert}$$

Partial reward for topological proximity drives the policy to explore structural variants (e.g., incomplete reentrancy cycles), deferring strict precision to Stage 2.

### 5.2 Stage 2: Exact Signature Verification (hard reward)

As the policy matures, the reward switches to a strict binary criterion grounded in **subgraph isomorphism**. The VF2 algorithm determines whether the signature $G^*$ appears inside $G_\text{obs}$ under the label assignment (node = address category, edge = interaction semantics):

$$r_\text{strict} = \begin{cases} 1 & \text{if } G^* \text{ is subgraph-isomorphic into } G_\text{obs} \\ 0 & \text{otherwise} \end{cases}$$

The dichotomous signal removes partial-credit ambiguity and provides an unambiguous learning signal for exact pattern matching.

### 5.3 Curriculum Scheduling: Smooth Sigmoid Transition

The two stages are blended through a continuous sigmoid schedule:

$$R_\text{graph} = \sigma(t)\, r_\text{strict} + \bigl(1 - \sigma(t)\bigr)\, r_\text{general} + R_\text{collab} + R_\text{param}$$

$$\sigma(t) = \bigl(1 + e^{-k(t - m)}\bigr)^{-1}$$

with $k$ controlling transition steepness and $m$ the midpoint step. Unlike a step function, the sigmoid ramp avoids abrupt gradient perturbations at the curriculum boundary.

### 5.4 Two Auxiliary Terms

- **Multi-graph collaboration bonus** $R_\text{collab} = +0.3$: awarded when corroborating anomalies are detected *simultaneously* across multiple graph views (e.g., a fund cycle in the FFG co-occurring with a circular call sequence in the CaG).
- **Parameter error penalty** $R_\text{param} = -0.3$ (per misaligned numeric field): incentivizes the policy to attend to fine-grained numerical context beyond what topological matching captures.

---

## 6. Heterogeneous Graph Encoder $\Phi$

$\Phi$ is an extension of the Relational Graph Convolutional Network (**RGCN**) that encodes a graph into a 256-d vector; relations are the edge types, and node features are one-hot node types.

- **Pre-training**: trained on an independent data split with a **graph-contrastive** objective. Positive pairs share the same vulnerability label; hard negatives are the top-$k$ nearest dissimilar-class graphs by cosine distance.
- **Frozen**: after pre-training, $\Phi$ is **kept frozen** throughout GRPO. If $\Phi$ were updated jointly with the policy $\pi_\theta$, the cosine-similarity reward would become a moving target, amplifying gradient variance and introducing reward non-stationarity.

---

## 7. Policy Optimization: GRPO

The policy network $\pi_\theta$ is optimized with **Group Relative Policy Optimization (GRPO)**. For each query $q$, a group of $N$ candidate completions is sampled from the reference policy, and the objective is

$$\mathcal{J}_\text{GRPO}(\theta) = \mathbb{E}\left[\frac{1}{N}\sum_{i=1}^{N}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \min\!\bigl(\rho_{i,t} A_i,\ \text{clip}(\rho_{i,t}, 1-\varepsilon, 1+\varepsilon) A_i\bigr)\right]$$

where the importance ratio is $\rho_{i,t} = \pi_\theta(o_{i,t}\mid q, o_{i,<t}) / \pi_{\theta_\text{old}}(o_{i,t}\mid q, o_{i,<t})$ and the group-normalized advantage is $A_i = (r_i - \bar{r}) / \text{std}(r)$.

Key design choices:

- **No SFT pretraining** — pure RL, following the R1-style paradigm;
- **KL regularization omitted** — policy drift is implicitly constrained by the clip bound;
- **Reward-collapse guard**: under GRPO, the binary $r_\text{strict}$ can produce *degenerate batches* where all $N$ group samples receive the same reward, yielding zero group-normalized advantage and a vacuous gradient. In that case the gradient from $R_\text{format}$, $R_\text{collab}$, and $R_\text{param}$ (which remain non-constant within the group) is retained, ensuring a non-trivial learning signal.

---

## 8. Output Format and Vulnerability Classes

The policy must emit markup conforming to the following schema:

```
<vulnerability>TYPE</vulnerability>
<trace>ADDRESS</trace>
```

- `TYPE` in { `reentrancy`, `short_address`, `timestamp` }
- `ADDRESS` is a `0x`-prefixed 40-hex-digit address (the attacker-traceability result)

The predicted vulnerability type selects which reference signature $G^*$ the sample is scored against — this is the mechanism that anchors the textual prediction in graph structure.

---

## 9. Notation

| Symbol | Meaning |
|---|---|
| $G_\text{obs}$ | unified observed snapshot within a time window, $\{G_f, G_c, G_\text{call}\}$ |
| $G^*$ | canonical reference attack signature of a vulnerability class |
| $\Phi$ | heterogeneous graph encoder (RGCN extension), graph -> $\mathbb{R}^{256}$ |
| $\mathbf{z}_\text{obs}, \mathbf{z}^*$ | graph-level embeddings of $G_\text{obs}$ and $G^*$ |
| $\sigma(t)$ | curriculum sigmoid, interpolating between the two stages over training step $t$ |
| $\pi_\theta$ | the policy network being optimized |
| $N$ | number of candidate completions per query in GRPO |

---

## In One Sentence

> Decompose an on-chain trace into three complementary graphs (fund-flow / creation / call), and use a sigmoid-curriculum reward that first explores broadly via graph-embedding cosine similarity and then verifies strictly via VF2 subgraph isomorphism, to train an LLM with pure RL (GRPO, no SFT, no KL) so that it both generalizes across structurally heterogeneous attack patterns and precisely classifies vulnerabilities while tracing the attacker.

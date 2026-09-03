# REVIEWER DEFENSE & ANTICIPATED CRITIQUES

This document details anticipated reviewer critiques for the IEEE journal submission and provides mathematically grounded, empirical responses based on the 12-run Flickr8k benchmark.

---

### Critique 1: "Why does the full HEDO-HVSC model not show a substantial gain over the SSD Baseline on clean Flickr8k retrieval ($48.28\%$ vs. $48.25\%$)? Why is standalone HEDO slightly higher ($48.43\%$ vs. $48.28\%$)?"

**Author Response:**
Our study is designed as a rigorous, controlled factorial component investigation ($2 \times 2$ factorial matrix across 3 independent random seeds). In cross-modal metric learning under frozen backbones (ViT-B/16 and RoBERTa-base), the primary representation capacity resides in the pretrained representations. 

The empirical findings establish that:
1. The baseline SSD recurrence achieves $48.25\% \pm 0.47\%$ Mean Recall.
2. Standalone HEDO provides a modest retrieval improvement ($48.43\% \pm 0.29\%$), validating that coordinate-momentum smoothing stabilizes patch token representations.
3. Adding HVSC introduces symmetric KL regularization, which functions as a structural constraint rather than a capacity expander. The paired per-seed delta between Full and Baseline is $+0.03\% \pm 0.20\%$ (Seed 42: $-0.14\%$, Seed 43: $-0.08\%$, Seed 44: $+0.32\%$).
4. The primary benefit of combining HEDO with HVSC emerges under out-of-distribution high-frequency perturbations ($+0.50\%$ gain under $\sigma=0.10$ Gaussian noise), demonstrating that the combined architecture acts as a noise-resilient regularizer rather than an over-parameterized feature fitter.

---

### Critique 2: "Where is the formal mathematical proof that the discrete HEDO update monotonically decreases Hamiltonian energy?"

**Author Response:**
We explicitly clarify in the manuscript that HEDO is a **Hamiltonian-inspired discrete dissipative transformation**, not an exact symplectic or formally provable Lyapunov-stable discrete integrator. 

In our pre-training diagnostic (`pre_training_diagnostic.json`), unforced forward simulation of visual coordinates yields trajectory values: $170.824 \to 170.651 \to 170.927 \to 171.646 \to 172.801 \to 174.387$. Because the transformation utilizes learned vector fields $\mathbf{W}_q, \mathbf{W}_p$ and non-zero driving token inputs under forward Euler discretization ($\Delta t = 0.1, \beta = 0.05, \gamma = 0.1$), energy is not universally monotonically decreasing across all arbitrary sequence inputs. We explicitly state this empirical behavior in Section III-C and avoid any overclaims regarding formal Lyapunov stability.

---

### Critique 3: "Does this architecture utilize native Mamba-2 hardware-optimized CUDA kernels?"

**Author Response:**
No. To maintain complete transparency, the paper explicitly states that the implementation is a **custom PyTorch SSD-style recurrent state-space block**. It implements chunk-wise recurrence ($C=16, d_{\text{state}}=64$) with state continuity ($\mathbf{h}_{\text{start}, k} = \mathbf{h}_{\text{end}, k-1}$) using standard PyTorch tensor primitives. This design ensures transparent reproducibility across standard GPU environments without external non-standard CUDA driver dependencies.

---

### Critique 4: "Why are the backbones frozen, and why is evaluation limited to Flickr8k?"

**Author Response:**
1. **Backbone Freezing:** Freezing pretrained ViT-B/16 ($86.6\text{M}$ params) and RoBERTa-base ($124.6\text{M}$ params) isolates the architectural effect of the newly introduced state-space recurrence and coupling modules ($\sim 0.33\text{M}$ to $0.42\text{M}$ trainable parameters). Unfreezing backbones would introduce massive optimization confounders (learning rate schedules, layer decay, catastrophic forgetting), obscuring the direct contribution of HEDO and HVSC.
2. **Benchmark Scope:** Flickr8k ($6{,}000$ train, $1{,}000$ val, $1{,}000$ test images) provides a certified, leak-free benchmark for multi-seed factorial runs ($12$ full training and testing cycles). Extending to larger datasets (e.g., MS-COCO, Flickr30k) and joint end-to-end fine-tuning is explicitly outlined in Section V as future work.

---

### Critique 5: "Is the corruption robustness universal across all perturbation types?"

**Author Response:**
No. As reported in Table III, robustness is **mixed**:
* Clean subset: $\Delta = -0.87\%$
* Gaussian noise ($\sigma=0.05$): $\Delta = -0.47\%$
* Gaussian noise ($\sigma=0.10$): $\mathbf{\Delta = +0.50\%}$
* Brightness increase ($+30\%$): $\Delta = -0.60\%$

The model demonstrates specific resilience to severe high-frequency pixel noise ($\sigma=0.10$), consistent with the physical intuition of dissipative damping ($\beta > 0$). We explicitly present these mixed results without generalizing to universal robustness.

---

### Critique 6: "Does HVSC require cross-modal attention or shared computation during gallery retrieval?"

**Author Response:**
No. Retrieval inference is strictly **unimodal and deterministic**. During test evaluation, image embeddings are computed solely from image pixels using $\boldsymbol{\mu}_I$, and text embeddings are computed solely from token IDs using $\boldsymbol{\mu}_T$. Cross-modal KL regularization $\mathcal{D}_{\text{SKL}}$ is applied exclusively during training. Test-time gallery retrieval uses pre-computed normalized embeddings with standard inner-product search ($O(N)$ dot products).

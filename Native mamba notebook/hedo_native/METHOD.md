# Method: E-HEDO and X-HVSC for multimodal Mamba-2

Phase-1 formulation, matching `models.py` line for line. Notation: token $x_i \in \mathbb{R}^d$ ($d=128$) after the modality projection; image sequence length $L_I = 196$, caption length $L_T \le 64$; chunk size $C = 16$.

## 1. Energy-dissipating operator (E-HEDO)

**Energy.** For each token, a coordinate $q$ and a momentum $p$ carry

$$H(q,p) = \tfrac12\|p\|^2 + V(q), \qquad V(q) = \sum_{r=1}^{R} w_r \,\log\cosh(u_r^\top q + b_r), \quad w_r = \operatorname{softplus}(\omega_r) \ge 0 .$$

$V \ge 0$, so $H \ge 0$. The gradient is $\nabla V(q) = U^\top\big(w \odot \tanh(Uq + b)\big)$, and the Hessian satisfies

$$0 \preceq \nabla^2 V(q) = U^\top \operatorname{diag}\!\big(w \odot \operatorname{sech}^2(Uq+b)\big) U \preceq \hat L\, I, \qquad \hat L = \max_r w_r \,\|U\|_2^2 ,$$

since $0 < \operatorname{sech}^2 \le 1$. $V$ is therefore $\hat L$-smooth.

**Dynamics.** $q_0 = x_i$, $p_0 = W_p x_i$. For $k = 0,\dots,K-1$ (with $K = 3$):

$$p_{k+1} = c_i\, p_k - \Delta t\, \nabla V(q_k), \qquad q_{k+1} = q_k + \Delta t\, p_{k+1},$$

where
- the step size is $\Delta t = \min\!\big(\Delta t_{\max}\,\sigma(\theta),\ \sqrt{(1-\varepsilon)/\hat L}\big)$;
- the token-wise damping rate is $\gamma_i = \operatorname{softplus}(g^\top x_i + g_0)$ (input-dependent: this is what lets the operator treat background and foreground tokens differently);
- the damping factor is $c_i = \min\!\big(e^{-\gamma_i \Delta t},\ \sqrt{1 - \hat L \Delta t^2}\big)$.

**Output.** The dissipated fraction is $\rho_i = \dfrac{H(q_0,p_0) - H(q_K,p_K)}{H(q_0,p_0) + 1} \in [0,1)$, and the output token is

$$\tilde x_i = e^{-\eta \rho_i}\, q_K, \qquad \eta = \operatorname{softplus}(\eta_{\text{raw}}),$$

which is what goes into Mamba-2.

**Theorem 1 (energy non-increase).** If $V$ is $\hat L$-smooth, $\hat L\Delta t^2 < 1$, and $0 \le c \le \sqrt{1-\hat L\Delta t^2}$, then $H(q_{k+1},p_{k+1}) \le H(q_k,p_k)$ for every $k$.

*Proof.* Write $p' = p_{k+1}$ and $q' = q_k + \Delta t\,p'$. By the descent lemma,

$$V(q') \le V(q_k) + \Delta t\langle \nabla V(q_k), p'\rangle + \tfrac{\hat L\Delta t^2}{2}\|p'\|^2 .$$

From the momentum update, $\Delta t\,\nabla V(q_k) = c\,p_k - p'$, so

$$\Delta t\langle \nabla V(q_k), p'\rangle = c\langle p_k, p'\rangle - \|p'\|^2 .$$

Let $a = 1 - \hat L\Delta t^2 > 0$. By Young's inequality, $c\langle p_k,p'\rangle \le \tfrac{c^2}{2a}\|p_k\|^2 + \tfrac a2\|p'\|^2$. Hence

$$H(q',p') \le V(q_k) + \|p'\|^2\Big(\tfrac12 - 1 + \tfrac{\hat L\Delta t^2}{2} + \tfrac a2\Big) + \tfrac{c^2}{2a}\|p_k\|^2 = V(q_k) + \tfrac{c^2}{2a}\|p_k\|^2 \le V(q_k) + \tfrac12\|p_k\|^2 = H(q_k,p_k),$$

because the bracket equals 0 and $c^2 \le a$. $\blacksquare$

**Corollaries.** By induction $H(q_k, p_k) \le H_0 := H(q_0, p_0)$ for all $k$. Hence $\|p_k\| \le \sqrt{2H_0}$ and $\|q_K - x_i\| \le K\,\Delta t\,\sqrt{2H_0}$: the displacement is bounded by the initial energy, whatever the learned parameters. $\rho_i \in [0,1)$, so the attenuation factor lies in $(e^{-\eta}, 1]$.

**What is and is not claimed.** The scheme is a conformally damped semi-implicit (symplectic-Euler-type) step. It is *not* symplectic when $c<1$, and nothing is conserved. "Hamiltonian-inspired" refers to the energy $H$ and the coordinate–momentum structure. Whether dissipation concentrates on background tokens is an empirical question (H1), not a theorem. Implementation checks: the gate (`gate.py`) and the unit tests verify the Hessian bound against autograd, and verify energy non-increase in float64 on real tokens and adversarial parameters. The clamp `dissipated >= 0` only removes float32 round-off.

## 2. Chunk-boundary exchange with variational bottleneck (X-HVSC)

**Pass 1 (independent, per modality $m$).** $y^m = \text{Mamba2}^m(\tilde x^m)$. Let $s^m_k$ be the state at the last valid token of chunk $k$, $k = 1..K_m$, $K_m = \lceil L_m / C\rceil$.

**Bottleneck.** $q(z^m_k \mid s^m_k) = \mathcal N(\mu^m(s^m_k), \operatorname{diag}\sigma^2(s^m_k))$ with $\mu = 10\tanh(\cdot/10)$ and $\log\sigma^2 \in (-4, 1.5)$ through a sigmoid. The rate term is $\mathrm{KL}\big(q(z^m_k\mid s^m_k)\,\|\,\mathcal N(0,I)\big)$, averaged over dimensions and valid chunks.

**Dual embedding.** $e^m = \operatorname{norm}\!\big(h^m(\operatorname{mean}_t y^m_t + \operatorname{mean}_k R^m z^m_k)\big)$, used for scalable retrieval.

**Exchange (pass 2), for a pair (image, caption).** The receiver's chunk $k$ gets the message

$$\text{msg}^{m}_k = \tanh(\alpha^m)\cdot \operatorname{MHA}\big(Q = s^m_{k-1},\ K = V = \{z^{\bar m}_j\}_j\big) \qquad (Q = \text{learned } q_0^m \text{ for } k=1),$$

added to the first token of chunk $k$ only. Then $y'^m = \text{Mamba2}^m(\tilde x^m + \text{msg}^m)$ reuses the same weights, and the exchange embedding is $e'^m = \operatorname{norm}(h^m(\operatorname{mean}_t y'^m_t + \operatorname{mean}_k R^m z^m_k))$. The pair score is $s' = \langle e'^I, e'^T\rangle$. With $\alpha^m = 0$ at initialisation, the exchange is exactly a no-op ($s' = \langle e^I, e^T\rangle$), which the gate checks.

**Retrieval.** Rank all items by the dual score, then re-rank each query's top-$K$ ($K=16$) by $s'$. Both the dual and the re-ranked metrics are reported.

**Objective.**

$$\mathcal L = \text{InfoNCE}_{\text{multi-pos}}(e^I, e^T) + \lambda_x\, \text{InfoNCE}_{\text{multi-pos}}(s'_{U\times N}) + \beta \sum_m \overline{\mathrm{KL}}(q\,\|\,\mathcal N(0,I)),$$

with $\lambda_x = 1$ and $\beta = 10^{-3}$. $s'_{U\times N}$ is computed for all $U{=}8$ images $\times$ $N{=}40$ captions in the batch.

## 3. Complexity

$d$: width; $R$: potential rank; $K_s$: HEDO steps; $d_s$: SSM state size; $h$: attention heads.

| Component | Time per sequence |
|---|---|
| E-HEDO | $O(K_s\, L\, d R)$, linear in $L$ (no token interaction) |
| Mamba-2 pass | $O(L\, d\, d_s)$, linear in $L$ |
| X-HVSC pass 2 | one extra Mamba-2 pass plus attention over chunks, $O\big(L\, d\, d_s + \tfrac{L_I}{C}\tfrac{L_T}{C}\, d\big)$: linear in $L_I$ for a fixed caption |
| Retrieval, $N_I$ images and $N_T$ captions | dual $O(N_I N_T d)$ dot products, plus $K (N_I + N_T)$ pass-2 evaluations (not $N_I N_T$) |

`scaling.py` measures the actual slope (H4). Intra-chunk computation stays the native Mamba-2 scan: messages enter only as additive inputs at chunk-start tokens.

## 4. Computational graph

```
ViT-B/16 (frozen, cached) -> Linear 768->128 -> E-HEDO --+
                                                          |-> Mamba-2 (pass 1) -> boundary states -> q(z|s) --+--> dual embedding e^I
RoBERTa (frozen, cached)  -> Linear 768->128 -> E-HEDO --+                                               |
                                                                                                         v
                                        chunk-start messages MHA(s_{k-1}, z^{other})  -> Mamba-2 (pass 2, same weights) -> e'^I, e'^T -> s'
```

## 5. Hypotheses: operational criteria (fixed before running; implemented in `hypotheses.py`)

Variant names: *ours* = `full_x`, *baseline* = `baseline`. CIs are 95% t-intervals over matched seeds, with at least 3 seeds (5 planned).

- **H1 — energy dissipation.** All four must hold:
  - (a) max float64 per-step energy increase on real test tokens ≤ 1e-9.
  - (b) attenuation(foreground) − attenuation(background) > 0, CI excluding 0. Foreground = top-25% ViT CLS-attention patches, background = bottom 50%.
  - (c1) MR(ours) − MR(baseline) CI lower bound > −1.0 (non-inferiority).
  - (c2) MR(background occluded) − MR(foreground occluded) CI lower > 0 for ours.
- **H2 — modality dominance.** Both must hold:
  - (a) mean |log10 (‖∂L/∂x_img‖ / ‖∂L/∂x_txt‖)| over the last epoch is lower for ours than for `full_x_noexchange`, CI upper < 0.
  - (b) both exchange gates are non-zero, and the message-reliance dominance index $|r_I - r_T|/(r_I + r_T)$ averages below 0.5.
- **H3 — robustness.** Relative MR ($\text{MR}_{\text{corrupt}}/\text{MR}_{\text{clean}}$), averaged over the image corruptions and separately over the text corruptions in `robustness.py`: ours − baseline, CI lower > 0 for both modalities.
- **H4 — efficiency.** Both must hold:
  - (a) head inference latency and training-step time at 196 tokens ≤ 1.5× baseline.
  - (b) log-log slope of latency against image length (784 to 12,544 tokens) ≤ 1.2.

A negative verdict is reported as such.

## 6. Scope and honest limitations

- **Task:** image–text retrieval on Flickr8k (1K test), with frozen ViT-B/16 and RoBERTa-base. Metrics are R@1/5/10, MedR, and mean recall. Accuracy, F1, AUROC and mIoU from the original proposal do not apply to this task. Baselines from other tasks (MambaAD, TimeViper, …) are not comparable; the controls are retrained under an identical protocol instead (param-matched Mamba-2, param-matched Transformer, Transformer with our modules, no-mixer).
- **Resolution:** backbones are fixed at 196 patches. H4 at higher lengths uses random tensors of token shape, so it measures compute, not accuracy.
- **Pairwise cost:** exchange scores are pairwise. The retrieval cost is stated as dual + top-K re-ranking, not as a pure dual encoder.
- **Foreground proxy:** "background" is defined by a saliency proxy (ViT attention), not by human masks.

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 9

out_dir = r"e:\DL Project\final_ieee_paper\figures"
os.makedirs(out_dir, exist_ok=True)

# -------------------------------------------------------------
# FIGURE 2: Flickr8k Dataset & Preprocessing Pipeline
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=300)
ax.axis('off')
ax.set_xlim(0, 10)
ax.set_ylim(0, 4.5)

# Master Dataset Box
rect_m = patches.FancyBboxPatch((3.5, 3.7), 3.0, 0.65, boxstyle="round,pad=0.05", ec="#1f4e79", fc="#d9e1f2", lw=1.3)
ax.add_patch(rect_m)
ax.text(5.0, 4.15, "Official Flickr8k Benchmark", ha='center', va='center', fontsize=9, fontweight='bold', color='#1f4e79')
ax.text(5.0, 3.90, "8,000 Photographic Images | 40,000 Captions (5 caps/img)", ha='center', va='center', fontsize=7.5)

# Three Split Boxes
# Train Split
rect_tr = patches.FancyBboxPatch((0.5, 2.4), 2.6, 0.8, boxstyle="round,pad=0.05", ec="#375623", fc="#e2efda", lw=1.2)
ax.add_patch(rect_tr)
ax.text(1.8, 2.95, "Train Split (75%)", ha='center', va='center', fontsize=8.5, fontweight='bold', color="#375623")
ax.text(1.8, 2.70, "6,000 Images | 30,000 Captions", ha='center', va='center', fontsize=7.5)
ax.text(1.8, 2.50, "Loss Optimization & Tuning", ha='center', va='center', fontsize=7, fontstyle='italic')

# Val Split
rect_va = patches.FancyBboxPatch((3.7, 2.4), 2.6, 0.8, boxstyle="round,pad=0.05", ec="#833c0c", fc="#fce4d6", lw=1.2)
ax.add_patch(rect_va)
ax.text(5.0, 2.95, "Validation Split (12.5%)", ha='center', va='center', fontsize=8.5, fontweight='bold', color="#833c0c")
ax.text(5.0, 2.70, "1,000 Images | 5,000 Captions", ha='center', va='center', fontsize=7.5)
ax.text(5.0, 2.50, "Checkpoint Selection (Val MR)", ha='center', va='center', fontsize=7, fontstyle='italic')

# Test Split
rect_te = patches.FancyBboxPatch((6.9, 2.4), 2.6, 0.8, boxstyle="round,pad=0.05", ec="#7030a0", fc="#f2eaf9", lw=1.2)
ax.add_patch(rect_te)
ax.text(8.2, 2.95, "Held-Out Test Split (12.5%)", ha='center', va='center', fontsize=8.5, fontweight='bold', color="#7030a0")
ax.text(8.2, 2.70, "1,000 Images | 5,000 Captions", ha='center', va='center', fontsize=7.5)
ax.text(8.2, 2.50, "Certified Zero-Leakage Eval", ha='center', va='center', fontsize=7, fontstyle='italic')

# Arrows from Master to Splits
arrow_props = dict(arrowstyle="->", lw=1.1, color="#444444")
ax.annotate("", xy=(1.8, 3.2), xytext=(4.0, 3.7), arrowprops=arrow_props)
ax.annotate("", xy=(5.0, 3.2), xytext=(5.0, 3.7), arrowprops=arrow_props)
ax.annotate("", xy=(8.2, 3.2), xytext=(6.0, 3.7), arrowprops=arrow_props)

# Preprocessing Bottom Stream
# Left: Image Preprocessing
rect_ip = patches.FancyBboxPatch((0.8, 0.2), 3.8, 1.6, boxstyle="round,pad=0.05", ec="#1f4e79", fc="#f2f2f2", lw=1.1)
ax.add_patch(rect_ip)
ax.text(2.7, 1.6, "Visual Preprocessing & Feature Tokenization", ha='center', va='center', fontsize=8, fontweight='bold', color='#1f4e79')
ax.text(2.7, 1.25, "1. Resize & Center Crop: $224 \\times 224$ pixels, 3 channels\n2. Normalization: ImageNet mean & std\n3. Patch Partition: $16 \\times 16$ patches $\\to L_I=196$ + 1 [CLS]\n4. Frozen ViT-B/16: $\\mathbf{X}_I^{(0)} \\in \\mathbb{R}^{B \\times 197 \\times 768}$", ha='center', va='center', fontsize=7)

# Right: Text Preprocessing
rect_tp = patches.FancyBboxPatch((5.4, 0.2), 3.8, 1.6, boxstyle="round,pad=0.05", ec="#1f4e79", fc="#f2f2f2", lw=1.1)
ax.add_patch(rect_tp)
ax.text(7.3, 1.6, "Textual Preprocessing & Tokenization", ha='center', va='center', fontsize=8, fontweight='bold', color='#1f4e79')
ax.text(7.3, 1.25, "1. RoBERTa Byte-Pair Encoding (BPE)\n2. Fixed Sequence Truncation/Padding: $L_T = 64$\n3. Mask Generation: $\\mathbf{M}_T \\in \\{0, 1\\}^{B \\times 64}$ (1=valid, 0=pad)\n4. Frozen RoBERTa: $\\mathbf{X}_T^{(0)} \\in \\mathbb{R}^{B \\times 64 \\times 768}$", ha='center', va='center', fontsize=7)

# Connecting to Preprocessing
ax.annotate("", xy=(2.7, 1.8), xytext=(1.8, 2.4), arrowprops=arrow_props)
ax.annotate("", xy=(7.3, 1.8), xytext=(8.2, 2.4), arrowprops=arrow_props)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "fig2_dataset_pipeline.pdf"), bbox_inches='tight')
plt.savefig(os.path.join(out_dir, "fig2_dataset_pipeline.png"), bbox_inches='tight', dpi=300)
plt.close()
print("Generated Fig 2 Dataset Pipeline.")

# -------------------------------------------------------------
# FIGURE 3: HEDO Coordinate-Momentum Dissipation Operator
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 4.0), dpi=300)
ax.axis('off')
ax.set_xlim(0, 10)
ax.set_ylim(0, 4.2)

# Input Coordinate
r_q0 = patches.FancyBboxPatch((0.5, 2.8), 2.2, 0.7, boxstyle="round,pad=0.05", ec="#1f4e79", fc="#d9e1f2", lw=1.2)
ax.add_patch(r_q0)
ax.text(1.6, 3.25, "Input State Coordinate", ha='center', va='center', fontsize=8, fontweight='bold')
ax.text(1.6, 3.0, "$\\mathbf{q}_0 \\in \\mathbb{R}^{B \\times L \\times d_{\\text{model}}}$", ha='center', va='center', fontsize=7.5)

# Momentum Initialization
r_p0 = patches.FancyBboxPatch((0.5, 0.8), 2.2, 0.7, boxstyle="round,pad=0.05", ec="#833c0c", fc="#fce4d6", lw=1.2)
ax.add_patch(r_p0)
ax.text(1.6, 1.25, "Momentum Initialization", ha='center', va='center', fontsize=8, fontweight='bold')
ax.text(1.6, 1.0, "$\\mathbf{p}_0 = \\tanh(\\mathbf{W}_p \\mathbf{q}_0)$", ha='center', va='center', fontsize=7.5)

ax.annotate("", xy=(1.6, 1.5), xytext=(1.6, 2.8), arrowprops=dict(arrowstyle="->", lw=1.2, color="#833c0c"))

# Dissipative Momentum Update Box
r_pu = patches.FancyBboxPatch((3.5, 0.6), 3.0, 1.2, boxstyle="round,pad=0.05", ec="#c00000", fc="#fbe5d6", lw=1.3)
ax.add_patch(r_pu)
ax.text(5.0, 1.55, "Dissipative Momentum Step", ha='center', va='center', fontsize=8.5, fontweight='bold', color="#c00000")
ax.text(5.0, 1.25, "$\\mathbf{p}_{t+1} = \\mathbf{p}_t(1 - \\beta \\Delta t) - \\Delta t \\tanh(\\mathbf{W}_q \\mathbf{q}_t)$", ha='center', va='center', fontsize=7.5)
ax.text(5.0, 0.95, "Damping Factor: $(1 - \\beta \\Delta t) = 0.995 < 1$\nBounded Potential Force: $\\|\\tanh(\\cdot)\\| \\leq \\sqrt{d}$", ha='center', va='center', fontsize=6.8, fontstyle='italic')

# Coordinate Update Box
r_qu = patches.FancyBboxPatch((3.5, 2.5), 3.0, 1.0, boxstyle="round,pad=0.05", ec="#375623", fc="#e2efda", lw=1.3)
ax.add_patch(r_qu)
ax.text(5.0, 3.25, "Symplectic-Style Coordinate Step", ha='center', va='center', fontsize=8.5, fontweight='bold', color="#375623")
ax.text(5.0, 2.95, "$\\mathbf{q}_{t+1} = \\mathbf{q}_t + \\gamma \\Delta t \\, \\mathbf{p}_{t+1}$", ha='center', va='center', fontsize=7.5)
ax.text(5.0, 2.70, "Perturbation Scale: $\\gamma=0.1, \\Delta t=0.1$", ha='center', va='center', fontsize=7, fontstyle='italic')

# Connect momentum and coordinate updates
ax.annotate("", xy=(3.5, 1.2), xytext=(2.7, 1.15), arrowprops=arrow_props)
ax.annotate("", xy=(3.5, 2.9), xytext=(2.7, 3.15), arrowprops=arrow_props)
ax.annotate("", xy=(5.0, 2.5), xytext=(5.0, 1.8), arrowprops=dict(arrowstyle="->", lw=1.3, color="#c00000"))

# LayerNorm Output Box
r_out = patches.FancyBboxPatch((7.3, 1.6), 2.2, 1.2, boxstyle="round,pad=0.05", ec="#7030a0", fc="#f2eaf9", lw=1.3)
ax.add_patch(r_out)
ax.text(8.4, 2.50, "Layer Normalization", ha='center', va='center', fontsize=8.5, fontweight='bold', color="#7030a0")
ax.text(8.4, 2.15, "$\\text{HEDO}(\\mathbf{q}) = \\text{LN}(\\mathbf{q}_{t+1})$", ha='center', va='center', fontsize=7.5)
ax.text(8.4, 1.85, "Modulates representation energy\n$\\mathcal{H}_{\\text{raw}} = 170.8 \\to \\mathcal{H}_{\\text{HEDO}} = 77.6$", ha='center', va='center', fontsize=6.8, fontstyle='italic')

ax.annotate("", xy=(7.3, 2.2), xytext=(6.5, 3.0), arrowprops=arrow_props)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "fig3_hedo_mechanism.pdf"), bbox_inches='tight')
plt.savefig(os.path.join(out_dir, "fig3_hedo_mechanism.png"), bbox_inches='tight', dpi=300)
plt.close()
print("Generated Fig 3 HEDO Mechanism.")

# -------------------------------------------------------------
# FIGURE 4: State-Continuous Chunk-Wise SSD Recurrence
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 3.6), dpi=300)
ax.axis('off')
ax.set_xlim(0, 10)
ax.set_ylim(0, 3.6)

# Chunk 1
r_c1 = patches.FancyBboxPatch((0.5, 0.8), 2.4, 2.2, boxstyle="round,pad=0.05", ec="#1f4e79", fc="#d9e1f2", lw=1.2)
ax.add_patch(r_c1)
ax.text(1.7, 2.75, "Chunk $k=1$ ($C=16$)", ha='center', va='center', fontsize=8.5, fontweight='bold', color="#1f4e79")
ax.text(1.7, 2.45, "Tokens $[\\mathbf{x}_1, \\dots, \\mathbf{x}_{16}]$", ha='center', va='center', fontsize=7.5)
ax.text(1.7, 2.15, "Init: $\\mathbf{h}_{\\text{start}, 1} = \\mathbf{0}$", ha='center', va='center', fontsize=7)
ax.text(1.7, 1.80, "Recurrence:\n$\\mathbf{h}_t = \\mathbf{A}\\mathbf{h}_{t-1} + \\mathbf{u}_t$", ha='center', va='center', fontsize=7)
ax.text(1.7, 1.15, "Boundary: $\\mathbf{b}_1 = \\mathbf{h}_{\\text{end}, 1}$", ha='center', va='center', fontsize=7.5, fontweight='bold', color="#833c0c")

# Chunk 2
r_c2 = patches.FancyBboxPatch((3.8, 0.8), 2.4, 2.2, boxstyle="round,pad=0.05", ec="#1f4e79", fc="#d9e1f2", lw=1.2)
ax.add_patch(r_c2)
ax.text(5.0, 2.75, "Chunk $k=2$ ($C=16$)", ha='center', va='center', fontsize=8.5, fontweight='bold', color="#1f4e79")
ax.text(5.0, 2.45, "Tokens $[\\mathbf{x}_{17}, \\dots, \\mathbf{x}_{32}]$", ha='center', va='center', fontsize=7.5)
ax.text(5.0, 2.15, "$\\mathbf{h}_{\\text{start}, 2} = \\mathbf{h}_{\\text{end}, 1}$", ha='center', va='center', fontsize=7.5, fontweight='bold', color="#c00000")
ax.text(5.0, 1.80, "Recurrence:\n$\\mathbf{h}_t = \\mathbf{A}\\mathbf{h}_{t-1} + \\mathbf{u}_t$", ha='center', va='center', fontsize=7)
ax.text(5.0, 1.15, "Boundary: $\\mathbf{b}_2 = \\mathbf{h}_{\\text{end}, 2}$", ha='center', va='center', fontsize=7.5, fontweight='bold', color="#833c0c")

# Chunk K
r_ck = patches.FancyBboxPatch((7.1, 0.8), 2.4, 2.2, boxstyle="round,pad=0.05", ec="#1f4e79", fc="#d9e1f2", lw=1.2)
ax.add_patch(r_ck)
ax.text(8.3, 2.75, "Chunk $k=K$ ($C=16$)", ha='center', va='center', fontsize=8.5, fontweight='bold', color="#1f4e79")
ax.text(8.3, 2.45, "Tokens $[\\mathbf{x}_{(K-1)C+1}, \\dots, \\mathbf{x}_L]$", ha='center', va='center', fontsize=7)
ax.text(8.3, 2.15, "$\\mathbf{h}_{\\text{start}, K} = \\mathbf{h}_{\\text{end}, K-1}$", ha='center', va='center', fontsize=7.5, fontweight='bold', color="#c00000")
ax.text(8.3, 1.80, "Recurrence:\n$\\mathbf{h}_t = \\mathbf{A}\\mathbf{h}_{t-1} + \\mathbf{u}_t$", ha='center', va='center', fontsize=7)
ax.text(8.3, 1.15, "Boundary: $\\mathbf{b}_K = \\mathbf{h}_{\\text{end}, K}$", ha='center', va='center', fontsize=7.5, fontweight='bold', color="#833c0c")

# Connecting State Flow Arrows
ax.annotate("NO STATE RESET", xy=(3.8, 1.9), xytext=(2.9, 1.9),
            arrowprops=dict(arrowstyle="->", lw=1.6, color="#c00000"),
            ha='center', va='bottom', fontsize=6.8, fontweight='bold', color="#c00000")
ax.annotate("NO STATE RESET", xy=(7.1, 1.9), xytext=(6.2, 1.9),
            arrowprops=dict(arrowstyle="->", lw=1.6, color="#c00000"),
            ha='center', va='bottom', fontsize=6.8, fontweight='bold', color="#c00000")

# Mask-weighted aggregation arrow at bottom
rect_agg = patches.FancyBboxPatch((1.5, 0.05), 7.0, 0.55, boxstyle="round,pad=0.05", ec="#7030a0", fc="#f2eaf9", lw=1.2)
ax.add_patch(rect_agg)
ax.text(5.0, 0.32, "Mask-Weighted Boundary Aggregation: $\\mathbf{h}_{\\text{bound}} = \\frac{\\sum_{k=1}^K m_k \\mathbf{b}_k}{\\sum_{k=1}^K m_k + \\epsilon} \\in \\mathbb{R}^{B \\times d_{\\text{state}}}$ (Eliminates Padding Bias)", ha='center', va='center', fontsize=7.5, fontweight='bold', color="#7030a0")

ax.annotate("", xy=(2.0, 0.6), xytext=(1.7, 0.8), arrowprops=dict(arrowstyle="->", lw=1.0, color="#7030a0"))
ax.annotate("", xy=(5.0, 0.6), xytext=(5.0, 0.8), arrowprops=dict(arrowstyle="->", lw=1.0, color="#7030a0"))
ax.annotate("", xy=(8.0, 0.6), xytext=(8.3, 0.8), arrowprops=dict(arrowstyle="->", lw=1.0, color="#7030a0"))

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "fig4_chunk_recurrence.pdf"), bbox_inches='tight')
plt.savefig(os.path.join(out_dir, "fig4_chunk_recurrence.png"), bbox_inches='tight', dpi=300)
plt.close()
print("Generated Fig 4 Chunk Recurrence.")

# -------------------------------------------------------------
# FIGURE 5: Training vs. Retrieval Inference Flow
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 5.2), dpi=300)

for ax in [ax1, ax2]:
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.5)

# Subplot 1: Training Regime
ax1.text(5.0, 2.35, "A. Training Phase: Stochastic Sampling & Joint Dual Objective", ha='center', va='center', fontsize=8.5, fontweight='bold', color="#1f4e79")
# Image branch
r_ti = patches.FancyBboxPatch((0.5, 1.1), 3.8, 0.9, boxstyle="round,pad=0.05", ec="#1f4e79", fc="#d9e1f2", lw=1.1)
ax1.add_patch(r_ti)
ax1.text(2.4, 1.70, "Visual Pipeline (Frozen ViT $\\to$ HEDO $\\to$ SSD)", ha='center', va='center', fontsize=7.5, fontweight='bold')
ax1.text(2.4, 1.35, "Latent $\\mathbf{z}_I = \\boldsymbol{\\mu}_I + \\boldsymbol{\\epsilon}_I \\odot \\boldsymbol{\\sigma}_I \\to$ Embedding $\\mathbf{e}_I$", ha='center', va='center', fontsize=7)

# Text branch
r_tt = patches.FancyBboxPatch((5.7, 1.1), 3.8, 0.9, boxstyle="round,pad=0.05", ec="#1f4e79", fc="#d9e1f2", lw=1.1)
ax1.add_patch(r_tt)
ax1.text(7.6, 1.70, "Textual Pipeline (Frozen RoBERTa $\\to$ HEDO $\\to$ SSD)", ha='center', va='center', fontsize=7.5, fontweight='bold')
ax1.text(7.6, 1.35, "Latent $\\mathbf{z}_T = \\boldsymbol{\\mu}_T + \\boldsymbol{\\epsilon}_T \\odot \\boldsymbol{\\sigma}_T \\to$ Embedding $\\mathbf{e}_T$", ha='center', va='center', fontsize=7)

# Objective Box
r_loss = patches.FancyBboxPatch((2.2, 0.1), 5.6, 0.7, boxstyle="round,pad=0.05", ec="#c00000", fc="#fbe5d6", lw=1.3)
ax1.add_patch(r_loss)
ax1.text(5.0, 0.55, "Joint Loss: $\\mathcal{L}_{\\text{total}} = \\mathcal{L}_{\\text{InfoNCE}}(\\mathbf{e}_I, \\mathbf{e}_T; \\tau) + 0.01 \\cdot \\mathcal{D}_{\\text{SKL}}(p_I \\parallel p_T)$", ha='center', va='center', fontsize=7.5, fontweight='bold', color="#c00000")
ax1.text(5.0, 0.28, "Multi-Positive Contrastive Alignment (5:1) + Symmetric FP32 KL Regularization", ha='center', va='center', fontsize=6.8)

ax1.annotate("", xy=(3.5, 0.8), xytext=(2.4, 1.1), arrowprops=arrow_props)
ax1.annotate("", xy=(6.5, 0.8), xytext=(7.6, 1.1), arrowprops=arrow_props)

# Subplot 2: Inference Regime
ax2.text(5.0, 2.35, "B. Test-Time Retrieval: Deterministic Unimodal Evaluation (Zero Cross-Modal Leakage)", ha='center', va='center', fontsize=8.5, fontweight='bold', color="#375623")
# Unimodal Image
r_ii = patches.FancyBboxPatch((0.5, 1.1), 3.8, 0.9, boxstyle="round,pad=0.05", ec="#375623", fc="#e2efda", lw=1.1)
ax2.add_patch(r_ii)
ax2.text(2.4, 1.70, "Query Image $\\mathbf{I} \\in \\mathbb{R}^{3 \\times 224 \\times 224}$", ha='center', va='center', fontsize=7.5, fontweight='bold')
ax2.text(2.4, 1.35, "Deterministic Mean: $\\hat{\\mathbf{e}}_I = \\text{Norm}(\\mathbf{h}_{\\text{pool}, I} + \\alpha \\mathbf{W}_{\\text{out}}\\boldsymbol{\\mu}_I)$", ha='center', va='center', fontsize=7)

# Unimodal Text
r_it = patches.FancyBboxPatch((5.7, 1.1), 3.8, 0.9, boxstyle="round,pad=0.05", ec="#375623", fc="#e2efda", lw=1.1)
ax2.add_patch(r_it)
ax2.text(7.6, 1.70, "Gallery Text Captions $\\mathcal{T}_{\\text{gallery}}$", ha='center', va='center', fontsize=7.5, fontweight='bold')
ax2.text(7.6, 1.35, "Deterministic Mean: $\\hat{\\mathbf{e}}_T = \\text{Norm}(\\mathbf{h}_{\\text{pool}, T} + \\alpha \\mathbf{W}_{\\text{out}}\\boldsymbol{\\mu}_T)$", ha='center', va='center', fontsize=7)

# Evaluation Box
r_eval = patches.FancyBboxPatch((2.2, 0.1), 5.6, 0.7, boxstyle="round,pad=0.05", ec="#203764", fc="#b4c6e7", lw=1.3)
ax2.add_patch(r_eval)
ax2.text(5.0, 0.55, "Cosine Similarity Matrix $\\mathbf{S} = \\hat{\\mathbf{e}}_I \\hat{\\mathbf{e}}_T^\\top \\rightarrow$ Rank @ K (1, 5, 10) & MedR", ha='center', va='center', fontsize=7.5, fontweight='bold', color="#203764")
ax2.text(5.0, 0.28, "No Cross-Modal Attention | No Inter-Modality Recurrent Communication", ha='center', va='center', fontsize=6.8, fontstyle='italic')

ax2.annotate("", xy=(3.5, 0.8), xytext=(2.4, 1.1), arrowprops=arrow_props)
ax2.annotate("", xy=(6.5, 0.8), xytext=(7.6, 1.1), arrowprops=arrow_props)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "fig5_train_inference.pdf"), bbox_inches='tight')
plt.savefig(os.path.join(out_dir, "fig5_train_inference.png"), bbox_inches='tight', dpi=300)
plt.close()
print("Generated Fig 5 Training vs Inference Flow.")

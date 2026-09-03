import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Configure high-quality publication styling
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 9

out_dir = r"e:\DL Project\final_ieee_paper\figures"
os.makedirs(out_dir, exist_ok=True)

# -------------------------------------------------------------
# FIGURE 1: Overall HEDO-HVSC Architecture Diagram
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
ax.axis('off')
ax.set_xlim(0, 10)
ax.set_ylim(0, 5.5)

# Visual Stream (Left)
ax.text(2.5, 5.2, "Visual Stream (Image Query / Gallery)", ha='center', va='center', fontsize=10, fontweight='bold', color='#1f4e79')
# Box 1: Image Input
rect1 = patches.FancyBboxPatch((1.0, 4.4), 3.0, 0.5, boxstyle="round,pad=0.05", ec="#1f4e79", fc="#d9e1f2", lw=1.2)
ax.add_patch(rect1)
ax.text(2.5, 4.65, "Input Image $\\mathbf{I} \\in \\mathbb{R}^{3 \\times 224 \\times 224}$", ha='center', va='center', fontsize=8.5)

# Box 2: Frozen ViT-B/16
rect2 = patches.FancyBboxPatch((1.0, 3.6), 3.0, 0.5, boxstyle="round,pad=0.05", ec="#833c0c", fc="#fce4d6", lw=1.2)
ax.add_patch(rect2)
ax.text(2.5, 3.85, "Frozen ViT-B/16 ($L_I=197, d=768$)", ha='center', va='center', fontsize=8.5, fontweight='bold')

# Box 3: Linear Projection & HEDO
rect3 = patches.FancyBboxPatch((1.0, 2.7), 3.0, 0.6, boxstyle="round,pad=0.05", ec="#375623", fc="#e2efda", lw=1.2)
ax.add_patch(rect3)
ax.text(2.5, 3.05, "Linear Projection $\\mathbf{W}_{\\text{proj}, I} \\to d_{\\text{model}}=128$", ha='center', va='center', fontsize=8)
ax.text(2.5, 2.85, "HEDO: Discrete Dissipative Update (Eq. 10-13)", ha='center', va='center', fontsize=7.5, fontstyle='italic')

# Box 4: Custom SSD Recurrence
rect4 = patches.FancyBboxPatch((1.0, 1.8), 3.0, 0.6, boxstyle="round,pad=0.05", ec="#1f4e79", fc="#d9e1f2", lw=1.2)
ax.add_patch(rect4)
ax.text(2.5, 2.15, "Custom PyTorch SSD Recurrence Block", ha='center', va='center', fontsize=8, fontweight='bold')
ax.text(2.5, 1.95, "Continuous Chunk Boundary Recurrence ($C=16, d_{\\text{state}}=64$)", ha='center', va='center', fontsize=7.5)

# Box 5: HVSC Coupling
rect5 = patches.FancyBboxPatch((1.0, 0.9), 3.0, 0.6, boxstyle="round,pad=0.05", ec="#7030a0", fc="#f2eaf9", lw=1.2)
ax.add_patch(rect5)
ax.text(2.5, 1.25, "HVSC: Chunk Boundary Aggregation & Latent $\\mathbf{z}_I$", ha='center', va='center', fontsize=8, fontweight='bold')
ax.text(2.5, 1.05, "Mask-Weighted Pooling $\\to \\mathcal{N}(\\boldsymbol{\\mu}_I, \\boldsymbol{\\sigma}_I^2)$ (Train) / $\\boldsymbol{\\mu}_I$ (Test)", ha='center', va='center', fontsize=7.5)

# Box 6: Final Embedding
rect6 = patches.FancyBboxPatch((1.2, 0.1), 2.6, 0.5, boxstyle="round,pad=0.05", ec="#203764", fc="#b4c6e7", lw=1.2)
ax.add_patch(rect6)
ax.text(2.5, 0.35, "Image Embedding $\\hat{\\mathbf{e}}_I \\in \\mathbb{S}^{127}$", ha='center', va='center', fontsize=8.5, fontweight='bold')

# Text Stream (Right)
ax.text(7.5, 5.2, "Textual Stream (Caption Query / Gallery)", ha='center', va='center', fontsize=10, fontweight='bold', color='#1f4e79')
# Box 1T: Caption Input
rect1t = patches.FancyBboxPatch((6.0, 4.4), 3.0, 0.5, boxstyle="round,pad=0.05", ec="#1f4e79", fc="#d9e1f2", lw=1.2)
ax.add_patch(rect1t)
ax.text(7.5, 4.65, "Input Caption $\\mathbf{T} \\in \\mathbb{Z}^{64}$ + Mask $\\mathbf{M}_T$", ha='center', va='center', fontsize=8.5)

# Box 2T: Frozen RoBERTa-base
rect2t = patches.FancyBboxPatch((6.0, 3.6), 3.0, 0.5, boxstyle="round,pad=0.05", ec="#833c0c", fc="#fce4d6", lw=1.2)
ax.add_patch(rect2t)
ax.text(7.5, 3.85, "Frozen RoBERTa-base ($L_T=64, d=768$)", ha='center', va='center', fontsize=8.5, fontweight='bold')

# Box 3T: Linear Projection & HEDO
rect3t = patches.FancyBboxPatch((6.0, 2.7), 3.0, 0.6, boxstyle="round,pad=0.05", ec="#375623", fc="#e2efda", lw=1.2)
ax.add_patch(rect3t)
ax.text(7.5, 3.05, "Linear Projection $\\mathbf{W}_{\\text{proj}, T} \\to d_{\\text{model}}=128$", ha='center', va='center', fontsize=8)
ax.text(7.5, 2.85, "HEDO: Discrete Dissipative Update (Eq. 10-13)", ha='center', va='center', fontsize=7.5, fontstyle='italic')

# Box 4T: Custom SSD Recurrence
rect4t = patches.FancyBboxPatch((6.0, 1.8), 3.0, 0.6, boxstyle="round,pad=0.05", ec="#1f4e79", fc="#d9e1f2", lw=1.2)
ax.add_patch(rect4t)
ax.text(7.5, 2.15, "Custom PyTorch SSD Recurrence Block", ha='center', va='center', fontsize=8, fontweight='bold')
ax.text(7.5, 1.95, "Continuous Chunk Boundary Recurrence ($C=16, d_{\\text{state}}=64$)", ha='center', va='center', fontsize=7.5)

# Box 5T: HVSC Coupling
rect5t = patches.FancyBboxPatch((6.0, 0.9), 3.0, 0.6, boxstyle="round,pad=0.05", ec="#7030a0", fc="#f2eaf9", lw=1.2)
ax.add_patch(rect5t)
ax.text(7.5, 1.25, "HVSC: Chunk Boundary Aggregation & Latent $\\mathbf{z}_T$", ha='center', va='center', fontsize=8, fontweight='bold')
ax.text(7.5, 1.05, "Mask-Weighted Pooling $\\to \\mathcal{N}(\\boldsymbol{\\mu}_T, \\boldsymbol{\\sigma}_T^2)$ (Train) / $\\boldsymbol{\\mu}_T$ (Test)", ha='center', va='center', fontsize=7.5)

# Box 6T: Final Embedding
rect6t = patches.FancyBboxPatch((6.2, 0.1), 2.6, 0.5, boxstyle="round,pad=0.05", ec="#203764", fc="#b4c6e7", lw=1.2)
ax.add_patch(rect6t)
ax.text(7.5, 0.35, "Text Embedding $\\hat{\\mathbf{e}}_T \\in \\mathbb{S}^{127}$", ha='center', va='center', fontsize=8.5, fontweight='bold')

# Connecting Arrows (Vertical)
arrow_props = dict(arrowstyle="->", lw=1.2, color="#333333")
for y_start, y_end in [(4.4, 4.1), (3.6, 3.3), (2.7, 2.4), (1.8, 1.5), (0.9, 0.6)]:
    ax.annotate("", xy=(2.5, y_end), xytext=(2.5, y_start), arrowprops=arrow_props)
    ax.annotate("", xy=(7.5, y_end), xytext=(7.5, y_start), arrowprops=arrow_props)

# Center Alignment / Objective Box
center_rect = patches.FancyBboxPatch((4.2, 0.1), 1.6, 1.4, boxstyle="round,pad=0.05", ec="#c00000", fc="#fbe5d6", lw=1.5, ls="--")
ax.add_patch(center_rect)
ax.text(5.0, 1.25, "Metric Alignment", ha='center', va='center', fontsize=8, fontweight='bold', color="#c00000")
ax.text(5.0, 0.95, "Train: Multi-Pos InfoNCE\n+ Sym. KL $\\mathcal{D}_{\\text{SKL}}$\nTest: Cosine $\\hat{\\mathbf{e}}_I^\\top \\hat{\\mathbf{e}}_T$", ha='center', va='center', fontsize=7)
ax.text(5.0, 0.35, "(Unimodal Inference)", ha='center', va='center', fontsize=7, fontstyle='italic')

# Arrows to center box
ax.annotate("", xy=(4.2, 0.35), xytext=(3.8, 0.35), arrowprops=arrow_props)
ax.annotate("", xy=(5.8, 0.35), xytext=(6.2, 0.35), arrowprops=arrow_props)
ax.annotate("", xy=(4.2, 1.15), xytext=(4.0, 1.15), arrowprops=dict(arrowstyle="<->", lw=1.2, color="#7030a0", ls=":"))
ax.annotate("", xy=(5.8, 1.15), xytext=(6.0, 1.15), arrowprops=dict(arrowstyle="<->", lw=1.2, color="#7030a0", ls=":"))

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "fig1_architecture.pdf"), bbox_inches='tight')
plt.savefig(os.path.join(out_dir, "fig1_architecture.png"), bbox_inches='tight', dpi=300)
plt.close()
print("Generated Fig 1 Architecture Diagram successfully.")

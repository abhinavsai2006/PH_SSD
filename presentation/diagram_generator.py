import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Set backend to Agg for non-interactive rendering
plt.switch_backend('Agg')

# Global Font Settings for Clean Vector Rendering
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#CBD5E1'
plt.rcParams['axes.linewidth'] = 1.2

OUTPUT_DIR = r"e:\DL Project\presentation\figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLOR_PRIMARY = '#0F2C59'  # Navy
COLOR_BLUE = '#1A56DB'     # Blue
COLOR_SLATE = '#334155'    # Slate Text

def generate_problem_complexity_fig():
    """Slide 2: Complexity Comparison Chart O(N^2) vs O(N)"""
    fig, ax = plt.subplots(figsize=(6.2, 4.5), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#F8FAFC')
    
    N = np.linspace(128, 4096, 200)
    transformer_cost = (N / 512)**2
    mamba_cost = N / 512
    
    ax.plot(N, transformer_cost, label='Transformer Attention O(N²)', color='#EF4444', linewidth=3, linestyle='-')
    ax.plot(N, mamba_cost, label='PH-SSD / Mamba-2 O(N)', color='#1D4ED8', linewidth=3.5, linestyle='-')
    
    ax.fill_between(N, transformer_cost, mamba_cost, color='#FEE2E2', alpha=0.4, label='Compute & Memory Bottleneck')
    
    ax.set_title('Computational Complexity Scaling', fontsize=12, fontweight='bold', color=COLOR_PRIMARY, pad=12)
    ax.set_xlabel('Sequence Length (N tokens)', fontsize=10, fontweight='bold', color=COLOR_SLATE)
    ax.set_ylabel('Relative Compute Cost', fontsize=10, fontweight='bold', color=COLOR_SLATE)
    ax.legend(loc='upper left', frameon=True, facecolor='#FFFFFF', edgecolor='#CBD5E1', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5, color='#94A3B8')
    
    ax.annotate('Quadratic Memory Explosion', xy=(3200, 39), xytext=(1700, 48),
                arrowprops=dict(facecolor='#EF4444', shrink=0.05, width=1.5, headwidth=6),
                fontsize=9, fontweight='bold', color='#B91C1C')
    ax.annotate('Linear Efficiency', xy=(3500, 7), xytext=(2100, 18),
                arrowprops=dict(facecolor='#1D4ED8', shrink=0.05, width=1.5, headwidth=6),
                fontsize=9, fontweight='bold', color='#1D4ED8')
                
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig_problem_complexity.png")
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Generated {path}")

def generate_sd_npf_energy_fig():
    """Slide 8: SD-NPF Energy Dissipation Curve"""
    fig, ax = plt.subplots(figsize=(6.2, 4.5), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#F8FAFC')
    
    t = np.linspace(0, 10, 200)
    energy_hamiltonian = np.exp(-0.4 * t) * (np.cos(3 * t) + 1.2) + 0.2
    noisy_energy = energy_hamiltonian + 0.25 * np.random.normal(0, 0.15, len(t))
    
    ax.plot(t, noisy_energy, color='#EF4444', alpha=0.4, label='Raw Unfiltered Energy')
    ax.plot(t, energy_hamiltonian, color='#1D4ED8', lw=2.5, label='Dissipative Energy (SD-NPF)')
    ax.axhline(0.2, color='#10B981', linestyle='--', label='Asymptotic Noise Floor')
    
    ax.set_title('Port-Hamiltonian Energy Dissipation', fontsize=12, fontweight='bold', color='#0F2C59', pad=12)
    ax.set_xlabel('Filter Iterations (t)', fontsize=10, fontweight='bold', color='#334155')
    ax.set_ylabel('System Energy H(x)', fontsize=10, fontweight='bold', color='#334155')
    ax.legend(loc='upper right', fontsize=8.5, frameon=True, facecolor='#FFFFFF', edgecolor='#CBD5E1')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig_sd_npf_energy.png")
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Generated {path}")

def main():
    print("Generating Matplotlib plots...")
    generate_problem_complexity_fig()
    generate_sd_npf_energy_fig()
    print("Matplotlib charts successfully generated!")

if __name__ == '__main__':
    main()

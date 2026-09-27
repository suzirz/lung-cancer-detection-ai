"""Generate publication-quality architecture diagram for PulmoScan.
Style: Clean scientific publication diagram (Nature/IEEE style), crisp, no AI glow/hallucination.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_box(ax, x, y, w, h, title, subtitle=None, bg_color='#f8fafc', border_color='#334155', text_color='#0f172a', title_fontsize=10, sub_fontsize=8, corner_radius=0.02):
    # Draw rounded rectangle
    box = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad={corner_radius},rounding_size=0.03",
        facecolor=bg_color,
        edgecolor=border_color,
        linewidth=1.4,
        zorder=2
    )
    ax.add_patch(box)
    
    if subtitle:
        ax.text(x + w / 2, y + h * 0.62, title, ha='center', va='center', fontsize=title_fontsize, fontweight='bold', color=text_color, zorder=3)
        ax.text(x + w / 2, y + h * 0.28, subtitle, ha='center', va='center', fontsize=sub_fontsize, color='#475569', zorder=3)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha='center', va='center', fontsize=title_fontsize, fontweight='bold', color=text_color, zorder=3)

def draw_arrow(ax, start, end, label=None, color='#64748b', style='->', lw=1.5, rad=0.0):
    connectionstyle = f"arc3,rad={rad}" if rad != 0 else "arc3"
    arrow = patches.FancyArrowPatch(
        start, end,
        arrowstyle='-|>',
        connectionstyle=connectionstyle,
        color=color,
        linewidth=lw,
        mutation_scale=14,
        zorder=1
    )
    ax.add_patch(arrow)
    if label:
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2 + 0.02
        ax.text(mid_x, mid_y, label, ha='center', va='bottom', fontsize=8, color='#475569', fontweight='medium', zorder=4)

def main():
    fig, ax = plt.subplots(figsize=(15, 8.5), dpi=300)
    ax.set_xlim(-0.05, 1.45)
    ax.set_ylim(-0.05, 1.05)
    ax.axis('off')
    
    # White background for paper grade quality
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    
    # Title
    ax.text(0.70, 0.98, "PulmoScan Multimodal Diagnostic Architecture", ha='center', va='top', fontsize=16, fontweight='bold', color='#0f172a')
    ax.text(0.70, 0.94, "Dual-Branch Deep Feature Extraction, Radiomics Profiling, and Explainable Saliency Triage", ha='center', va='top', fontsize=10, color='#64748b')

    # Background Group Enclosures (Subgraphs)
    # Stage 1: Feature Extraction Box
    feat_group = patches.FancyBboxPatch(
        (0.24, 0.28), 0.38, 0.58,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        facecolor='#f1f5f9',
        edgecolor='#cbd5e1',
        linewidth=1.0,
        linestyle='--',
        zorder=0
    )
    ax.add_patch(feat_group)
    ax.text(0.43, 0.83, "DUAL-BRANCH FEATURE EXTRACTION", ha='center', va='center', fontsize=9, fontweight='bold', color='#475569')

    # Stage 2: Classification Box
    clf_group = patches.FancyBboxPatch(
        (0.96, 0.16), 0.44, 0.70,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        facecolor='#f8fafc',
        edgecolor='#cbd5e1',
        linewidth=1.0,
        linestyle='--',
        zorder=0
    )
    ax.add_patch(clf_group)
    ax.text(1.18, 0.83, "CLINICAL TRIAGE & INTERPRETABILITY", ha='center', va='center', fontsize=9, fontweight='bold', color='#475569')

    # Node: Input
    draw_box(ax, 0.00, 0.45, 0.18, 0.18, "Axial CT Scan", "224 × 224 RGB Slice\n(DICOM / PNG)", bg_color='#ffffff', border_color='#0284c7', text_color='#0369a1')

    # Branch A: CNN
    draw_box(ax, 0.27, 0.60, 0.32, 0.16, "EfficientNet-B0 Backbone", "Pretrained ImageNet (4.01M params)\n16 MBConv Blocks + GAP", bg_color='#ffffff', border_color='#2563eb', text_color='#1d4ed8')
    draw_box(ax, 0.65, 0.60, 0.24, 0.16, "Latent Embedding", "1,280-Dimensional Vector\nDense Semantic Descriptors", bg_color='#eff6ff', border_color='#3b82f6', text_color='#1e40af')

    # Branch B: Radiomics
    draw_box(ax, 0.27, 0.34, 0.32, 0.16, "Radiomics Extractor", "Intensity Moments (Mean, Std, Skew, Kurt)\nEntropy + Morphological Gradients", bg_color='#ffffff', border_color='#059669', text_color='#047857')
    draw_box(ax, 0.65, 0.34, 0.24, 0.16, "Radiomics Descriptors", "13 Quantitative Features\nBiological / Margin Metrics", bg_color='#ecfdf5', border_color='#10b981', text_color='#065f46')

    # Feature Fusion
    draw_box(ax, 0.94, 0.42, 0.22, 0.15, "Feature Fusion", "Concatenated Vector\n(1,293 Dimensions)", bg_color='#eef2ff', border_color='#6366f1', text_color='#4338ca')

    # Classifiers (Upper right)
    draw_box(ax, 1.22, 0.68, 0.17, 0.11, "Random Forest", "Balanced Weights\n200 Estimators", bg_color='#ffffff', border_color='#ea580c', text_color='#c2410c', title_fontsize=9, sub_fontsize=7.5)
    draw_box(ax, 1.22, 0.52, 0.17, 0.11, "XGBoost", "Gradient Boosted Trees\nSubsample = 0.8", bg_color='#ffffff', border_color='#ea580c', text_color='#c2410c', title_fontsize=9, sub_fontsize=7.5)
    draw_box(ax, 1.22, 0.36, 0.17, 0.11, "Softmax Head", "Direct CNN Classifier\nCross-Entropy Logits", bg_color='#ffffff', border_color='#2563eb', text_color='#1d4ed8', title_fontsize=9, sub_fontsize=7.5)

    # Explainability (Lower right)
    draw_box(ax, 0.94, 0.18, 0.22, 0.14, "Grad-CAM Saliency", "Last Conv Layer (features.8)\nPositive Activation Gradients", bg_color='#fdf2f8', border_color='#db2777', text_color='#be185d')
    draw_box(ax, 1.22, 0.18, 0.17, 0.14, "Saliency Heatmap", "Jet / Viridis Colormap\nSpatial Tumor Focus", bg_color='#fff1f2', border_color='#e11d48', text_color='#9f1239', title_fontsize=9, sub_fontsize=7.5)

    # Connections / Arrows
    # From input to branches
    draw_arrow(ax, (0.18, 0.58), (0.27, 0.68), color='#2563eb', rad=-0.08)
    draw_arrow(ax, (0.18, 0.50), (0.27, 0.42), color='#059669', rad=0.08)

    # CNN branch to embedding
    draw_arrow(ax, (0.59, 0.68), (0.65, 0.68), color='#2563eb')
    # Radiomics branch to descriptors
    draw_arrow(ax, (0.59, 0.42), (0.65, 0.42), color='#059669')

    # To Feature Fusion
    draw_arrow(ax, (0.89, 0.65), (0.94, 0.53), color='#4f46e5', rad=-0.08)
    draw_arrow(ax, (0.89, 0.42), (0.94, 0.48), color='#4f46e5', rad=0.06)

    # Fusion to RF and XGBoost
    draw_arrow(ax, (1.16, 0.53), (1.22, 0.72), color='#ea580c', rad=-0.08)
    draw_arrow(ax, (1.16, 0.49), (1.22, 0.57), color='#ea580c')

    # CNN direct to Softmax Head
    draw_arrow(ax, (0.89, 0.68), (1.22, 0.42), color='#2563eb', rad=-0.22)

    # CNN to Grad-CAM & Heatmap
    draw_arrow(ax, (0.43, 0.60), (0.94, 0.25), color='#db2777', rad=0.28, label="Gradients")
    draw_arrow(ax, (1.16, 0.25), (1.22, 0.25), color='#db2777')

    # Output Box
    draw_box(ax, 0.40, 0.04, 0.60, 0.12, "Clinical Triage Output: Normal Parenchyma | Benign Nodule | Malignant Neoplasm", "Dual-Threshold Stratification: Malignancy Risk Flagging (tau = 0.35) + Spatial Attention Overlay", bg_color='#0f172a', border_color='#0f172a', text_color='#ffffff', title_fontsize=9.5, sub_fontsize=8)
    
    # Arrows to final output box
    draw_arrow(ax, (1.30, 0.36), (0.85, 0.16), color='#475569', rad=-0.12)
    draw_arrow(ax, (1.28, 0.18), (0.75, 0.16), color='#475569', rad=-0.06)

    # Save clean image
    output_path = Path("reports/architecture_diagram.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Diagram successfully generated at: {output_path}")

if __name__ == '__main__':
    main()

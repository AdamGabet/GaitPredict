"""
Gait-centered biomarker infographic - Simple grid layout.
Outputs a grid of body system tiles to output/.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from publication_colors import SYSTEM_COLOR_MAP

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_gait_infographic(save_path=None, extra_systems=None):
    fig, ax = plt.subplots(figsize=(14, 20))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 23)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    systems = [
        [('Cardiovascular', 'cardiovascular_system', ['Blood pressure', 'Heart rate']),
         ('Liver', 'liver', ['ALT', 'AST'])],
        [('Sleep', 'sleep_group', ['AHI', 'REM %']),
         ('Glycemic', 'glycemic_status', ['HbA1c', 'Glucose'])],
        [('Nightingale NMR', 'nightingale', ['Fatty acids', 'Amino acids']),
         ('Lifestyle', 'lifestyle_group', ['Activity', 'Sleep hours'])],
        [('Frailty', 'frailty', ['Grip strength', 'Lean mass']),
         ('Mental', 'mental', ['Depression', 'Anxiety'])],
        [('Body Composition', 'body_composition', ['Fat mass', 'Lean mass']),
         ('Bone Density', 'bone_density', ['BMD', 'BMC'])],
        [('Lipids', 'blood_tests_lipids', ['Cholesterol', 'Triglycerides']),
         ('Renal', 'renal_function', ['Creatinine', 'eGFR'])],
        [('Hematopoietic', 'hematopoietic', ['Hemoglobin', 'RBC']),
         ('Immune', 'immune_system', ['WBC', 'CRP'])],
        [('Anthropometric', 'anthropometric_group', ['Waist', 'Height']),
         ('Metabolomics', 'metabolites', ['Metabolites', 'Amino acids'])],
        [('Microbiome', 'microbiome', ['Bacteria', 'Species']),
         ('Diet', 'high_level_diet_with_stage', ['Vegetables', 'Fiber'])],
    ]

    # Extra rows appended by the Figure 1 composite build (same
    # [[(name, color_key, [labels]), ...], ...] shape as `systems` above).
    if extra_systems:
        systems = systems + [list(r) for r in extra_systems]

    box_width = 5.2
    box_height = 1.5
    gap_x = 0.6
    gap_y = 0.4
    start_x = 1.4
    start_y = 20.4

    for row_idx, row in enumerate(systems):
        y = start_y - row_idx * (box_height + gap_y)
        for col_idx, (name, color_key, labels) in enumerate(row):
            x = start_x + col_idx * (box_width + gap_x)
            rgba = SYSTEM_COLOR_MAP.get(color_key, (0.8, 0.8, 0.8, 1.0))
            color = tuple(0.02 + 0.98 * c for c in rgba[:3])

            box = FancyBboxPatch((x, y), box_width, box_height,
                                  boxstyle="square,pad=0",
                                  facecolor=color, edgecolor='none', alpha=0.2)
            ax.add_patch(box)
            border = FancyBboxPatch((x, y), box_width, box_height,
                                     boxstyle="square,pad=0",
                                     facecolor='none', edgecolor='black', linewidth=2)
            ax.add_patch(border)

            ax.text(x + box_width / 2, y + box_height - 0.22, name,
                    fontsize=25, fontweight='bold', ha='center', va='top', color='black')
            ax.text(x + box_width / 2, y + 0.38, ', '.join(labels),
                    fontsize=24, ha='center', va='center', color='#444444')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    plt.close()


if __name__ == '__main__':
    create_gait_infographic(
        save_path=os.path.join(OUTPUT_DIR, 'gait_biomarker_infographic.png')
    )

"""
Color configuration for publication figures.
Ensures consistent colors across r² contribution and Pearson r plots.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# System name renaming dictionary for cleaner labels
SYSTEM_RENAME_DICT = {
    'anthropometric_group': 'Anthropometric',
    'blood_tests_lipids': 'Lipids',
    'body_composition': 'Body Composition',
    'bone_density': 'Bone Density',
    'cardiovascular_system': 'Cardiovascular',
    'frailty': 'Frailty',
    'glycemic_status': 'Glycemic',
    'hematopoietic': 'Hematopoietic',
    'immune_system': 'Immune',
    'lifestyle': 'Lifestyle',
    'lifestyle_group': 'Lifestyle',
    'lifestyle_ordinal': 'Lifestyle',
    'liver': 'Liver',
    'metabolites': 'Metabolites',
    'metabolites_annotated': 'Metabolites',
    'metabolites_unannotated': 'Metabolites',
    'nightingale': 'Nightingale',
    'proteomics': 'Proteomics',
    'renal_function': 'Renal',
    'sleep_group': 'Sleep',
    'wearable_monthly': 'Wearable',
    'wearable_weekly': 'Wearable Weekly',
    'subject': 'Subject',
    'gait': 'Gait Features',
    'high_level_diet_with_stage': 'High Level Diet',
    'mental': 'Mental',
    'microbiome': 'Microbiome',
}


# Fixed color mapping for each system
SYSTEM_COLOR_MAP = {
    'frailty': (0.95, 0.77, 0.20, 1.0),  # Darker yellow/Mustard
    'body_composition': (0.15, 0.50, 0.13, 1.0),  # Darker green
    'bone_density': (0.22, 0.60, 0.95, 1.0),  # Blue (switched with cardiovascular)
    'liver': (0.60, 0.31, 0.64, 1.0),  # Purple
    'hematopoietic': (1.00, 0.50, 0.05, 1.0),  # Orange
    'blood_tests_lipids': (0.65, 0.34, 0.16, 1.0),  # Brown
    'anthropometric_group': (0.97, 0.51, 0.75, 1.0),  # Pink
    'wearable_monthly': (0.60, 0.60, 0.60, 1.0),  # Gray
    'proteomics': (0.20, 0.63, 0.17, 1.0),  # Green (Set1 original)
    'cardiovascular_system': (0.55, 0.00, 0.00, 1.0),  # Maroon (switched with bone_density)
    'glycemic_status': (0.89, 0.47, 0.76, 1.0),  # Light purple/pink
    'sleep_group': (0.00, 0.75, 1.00, 1.0),  # Electric cyan/blue
    'lifestyle': (0.40, 0.76, 0.65, 1.0),  # Teal
    'metabolites': (0.75, 0.40, 0.10, 1.0),  # Warm amber/burnt orange
    'nightingale': (0.99, 0.75, 0.44, 1.0),  # Light orange
    'immune_system': (0.70, 0.87, 0.54, 1.0),  # Light green
    'renal_function': (0.99, 0.55, 0.38, 1.0),  # Coral
    'wearable_weekly': (0.78, 0.78, 0.78, 1.0),  # Light gray
    'subject': (0.50, 0.50, 0.50, 1.0),  # Medium gray
    'gait': (0.25, 0.88, 0.82, 1.0),  # Turquoise
    'mental': (0.58, 0.40, 0.74, 1.0),  # Purple/Violet
    'high_level_diet_with_stage': (0.94, 0.50, 0.50, 1.0),  # Salmon/Light red
    'microbiome': (0.68, 0.85, 0.90, 1.0),  # Light blue/Sky
    'lifestyle_group': (0.40, 0.76, 0.65, 1.0),  # Teal (same as lifestyle)
    'lifestyle_ordinal': (0.40, 0.76, 0.65, 1.0),  # Teal (same as lifestyle)
    'metabolites_annotated': (0.75, 0.40, 0.10, 1.0),  # Warm amber (same as metabolites)
    'metabolites_unannotated': (0.75, 0.40, 0.10, 1.0),  # Warm amber (same as metabolites)
}

# Activity color palette
ACTIVITY_COLORS = {
    'tm_3kmh': '#27ae60',                  # green
    'self_selected_gait_speed': '#f1c40f', # yellow
    'romberg_closed': '#e74c3c',           # red
    'romberg_open': '#e67e22',             # orange
    'sit_to_stand': '#9b59b6',             # purple
    'stationary_walk': '#3498db',          # blue
    'tandem_walk': '#1abc9c',              # teal
}

# Pretty names for activities
ACTIVITY_NAMES = {
    'tm_3kmh': 'Treadmill 3km/h',
    'self_selected_gait_speed': 'Self-Selected Speed',
    'romberg_closed': 'Romberg Closed',
    'romberg_open': 'Romberg Open',
    'sit_to_stand': 'Sit to Stand',
    'stationary_walk': 'Stationary Walk',
    'tandem_walk': 'Tandem Walk',
}

# Label renaming dictionary for publication figures
LABEL_RENAME_DICT = {
    "age": "Age",
    "bmi": "BMI",
    "ahi": "AHI",
    "hand_grip_left": "L Hand Grip",
    "hand_grip_right": "R Hand Grip",
    "hand_grip_avg": "Hand Grip Avg",
    "frailty_gait_speed": "Gait Speed",
    "frailty_standing_balance": "Standing Balance",
    "frailty_chair_stand": "Chair Stand",
    "bt__hdl_cholesterol": "HDL Cholesterol",
    "bt__ldl_cholesterol": "LDL Cholesterol",
    "bt__triglycerides": "Triglycerides",
    "bt__total_cholesterol": "Total Cholesterol",
    "bt__hemoglobin": "Hemoglobin",
    "bt__hematocrit": "Hematocrit",
    "bt__ferritin": "Ferritin",
    "bt__creatinine": "Creatinine",
    "bt__glucose": "Glucose",
    "bt__urea": "Urea",
    "bt__albumin": "Albumin",
    "bt__bilirubin": "Bilirubin",
    "bt__alkaline_phosphatase": "Alkaline Phosphatase",
    "bt__alt": "ALT",
    "bt__ast": "AST",
    "bt__ggt": "GGT",
    "bt__hba1c": "HbA1c",
    "bt__insulin": "Insulin",
    "bt__wbc": "WBC",
    "bt__rbc": "RBC",
    "bt__platelets": "Platelets",
    "bt__neutrophils": "Neutrophils",
    "bt__lymphocytes": "Lymphocytes",
    "body_comp_android_fat_mass": "Android Fat Mass",
    "body_comp_android_lean_mass": "Android Lean Mass",
    "body_comp_android_fat_free_mass": "Android Fat-Free Mass",
    "body_comp_arm_left_lean_mass": "L Arm Lean Mass",
    "body_comp_arm_right_lean_mass": "R Arm Lean Mass",
    "body_comp_leg_left_lean_mass": "L Leg Lean Mass",
    "body_comp_leg_right_lean_mass": "R Leg Lean Mass",
    "body_comp_trunk_lean_mass": "Trunk Lean Mass",
    "body_comp_total_lean_mass": "Total Lean Mass",
    "body_comp_total_fat_mass": "Total Fat Mass",
    "body_spine_l1_bmd": "L1 BMD",
    "body_spine_l2_bmd": "L2 BMD",
    "body_spine_l3_bmd": "L3 BMD",
    "body_spine_l4_bmd": "L4 BMD",
    "body_spine_total_bmd": "Spine BMD",
    "body_femur_neck_bmd": "Femur Neck BMD",
    "body_femur_total_bmd": "Femur Total BMD",
    "waist": "Waist",
    "hips": "Hips",
    "waist_hip_ratio": "Waist-Hip Ratio",
    "hr_bpm": "Heart Rate",
    "sbp": "SBP",
    "dbp": "DBP",
    "pp": "Pulse Pressure",
    "map": "Mean Arterial Pressure",
    "liver_elasticity": "Liver Elasticity",
    "liver_cap": "Liver CAP",
    "sleep_efficiency": "Sleep Efficiency",
    "sleep_duration": "Sleep Duration",
    "sleep_latency": "Sleep Latency",
    "ecg_mv_p_duration": "ECG P Duration",
    "ecg_mv_qrs_duration": "ECG QRS Duration",
    "ecg_mv_qt_interval": "ECG QT Interval",
    "ecg_mv_pr_interval": "ECG PR Interval",
    "steps_daily_avg": "Daily Steps Avg",
    "active_minutes_daily": "Daily Active Minutes",
    "egfr": "eGFR",
    "uacr": "UACR",
    "automorph_artery_fractal_dimension": "Artery Fractal Dim",
    "automorph_vein_fractal_dimension": "Vein Fractal Dim",
    # Additional labels
    "body_comp_arm_left_fat_free_mass": "L Arm Fat Free Mass",
    "body_comp_arms_fat_free_mass": "Arms Fat Free Mass",
    "body_comp_arms_lean_mass": "Arms Lean Mass",
    "bt__alt_gpt": "ALT",
    "bt__mchc": "MCHC",
    "bt__monocytes_abs": "Monocytes Abs",
    "bt__neutrophils_%": "Neutrophils %",
    "bt__non_hdl_cholesterol": "Non-HDL Cholesterol",
    "csr_percent": "CSR %",
    "heart_rate_min_during_sleep": "Sleep HR Min",
    "iglu_1st_quartile": "Glucose Q1",
    "iglu_mag": "Glucose Magnitude",
    "intima_media_th_1_fit": "IMT",
    "moderate_activity_minutes": "Moderate Activity",
    "r_mv_V3": "R-Wave Peak Amplitude",
    "spine_l1_l3_area": "L1-L3 Area",
    "spine_l1_l3_bmc": "L1-L3 BMC",
    "spine_l1_l4_area": "L1-L4 Area",
    "spine_l2_l4_area": "L2-L4 Area",
    "vigorous_activity_minutes": "Vigorous Activity",
}


# Conceptual near-duplicate pairs between lifestyle_group and lifestyle_ordinal
# Maps lifestyle_group label -> lifestyle_ordinal label measuring the same thing
LIFESTYLE_DUPLICATE_MAP = {
    'physical_activity_maderate_days_a_week': 'moderate_activity_days',
    'walking_10min_days_a_week': 'walking_days_per_week',
    'sleep_hours_in_24H': 'sleep_hours',
    'work_days_a_week': 'work_days_per_week',
    'work_hours': 'work_hours_per_day',
    'work_hours_day': 'work_hours_per_day',
    'hours_using_computer_not_work': 'hours_computer_not_work',
}


def merge_systems(df):
    """
    Merge related systems in the comparisons DataFrame:
    1. metabolites_annotated + metabolites_unannotated -> metabolites
    2. lifestyle + lifestyle_group + lifestyle_ordinal -> lifestyle
       For duplicate/near-duplicate labels, keep the row with highest score.
    """
    df = df.copy()

    # --- Merge metabolites ---
    df.loc[df['system'].isin(['metabolites_annotated', 'metabolites_unannotated']), 'system'] = 'metabolites'

    # --- Merge lifestyle ---
    lifestyle_systems = ['lifestyle', 'lifestyle_group', 'lifestyle_ordinal']
    lifestyle_mask = df['system'].isin(lifestyle_systems)
    df.loc[lifestyle_mask, 'system'] = 'lifestyle'

    # Build a canonical label mapping for near-duplicates
    canonical = {}
    for lg_label, lo_label in LIFESTYLE_DUPLICATE_MAP.items():
        canonical[lg_label] = lo_label
        canonical[lo_label] = lo_label

    # Apply canonical names so duplicates share the same label
    lifestyle_rows = df['system'] == 'lifestyle'
    df.loc[lifestyle_rows, 'label'] = df.loc[lifestyle_rows, 'label'].map(
        lambda x: canonical.get(x, x)
    )

    # Deduplicate: for each (system, label, model, gender, score_type, sub_model) keep highest score
    group_cols = ['system', 'label', 'model', 'gender', 'score_type', 'sub_model']
    is_lifestyle = df['system'] == 'lifestyle'
    non_lifestyle = df[~is_lifestyle]
    lifestyle_deduped = df[is_lifestyle].sort_values('score', ascending=False).drop_duplicates(
        subset=group_cols, keep='first'
    )
    df = pd.concat([non_lifestyle, lifestyle_deduped], ignore_index=True)

    return df


def get_system_colors(systems_ordered):
    """
    Get consistent colors for systems with vibrant colors.
    Each system always gets the same color regardless of ordering.
    
    Args:
        systems_ordered: List of system names in order
    
    Returns:
        Dictionary mapping system name to color
    """
    color_dict = {}
    for sys in systems_ordered:
        if sys in SYSTEM_COLOR_MAP:
            color_dict[sys] = SYSTEM_COLOR_MAP[sys]
        else:
            # Fallback to a default color if system not in map
            color_dict[sys] = (0.5, 0.5, 0.5, 1.0)  # Gray
    
    return color_dict


def get_r2_filtered_data_and_order(r2_path, model_type, gender, score_threshold=0.0, pvalue_threshold=0.01):
    """
    Get filtered r² data and system ordering.
    This is the single source of truth for both r² and Pearson plots.
    
    Returns:
        tuple: (filtered_r2_df, systems_ordered, color_dict)
    """
    import pandas as pd
    
    # Load R2 data
    r2_df = pd.read_csv(r2_path, low_memory=False)
    r2_df = merge_systems(r2_df)
    
    # Filter
    r2_filtered = r2_df[
        (r2_df['sub_model'] == 'ensemble') &
        (r2_df['model'] == model_type) &
        (r2_df['gender'] == gender) &
        (r2_df['score_type'] == 'r2')
    ].copy()
    
    if score_threshold is not None:
        r2_filtered = r2_filtered[r2_filtered['score'] > score_threshold]
    
    if pvalue_threshold is not None:
        r2_filtered = r2_filtered[
            (r2_filtered['wilcox_pvalue_fdr'] < pvalue_threshold) &
            (r2_filtered['score_pvalue'] < 0.05)
        ]
    
    # Calculate contribution (min of delta and score)
    r2_filtered['contribution'] = r2_filtered[['delta', 'score']].min(axis=1)
    
    # Filter delta > 0
    r2_filtered = r2_filtered[r2_filtered['delta'] > 0].copy()
    
    # Exclude specific systems
    exclude_systems = ['wearable_weekly', 'subject']
    r2_filtered = r2_filtered[~r2_filtered['system'].isin(exclude_systems)].copy()
    
    # Exclude specific biomarkers
    exclude_labels = ['frailty_height', 'weight']
    r2_filtered = r2_filtered[~r2_filtered['label'].isin(exclude_labels)].copy()
    
    # Get systems sorted by median contribution
    system_stats = r2_filtered.groupby('system')['contribution'].agg(['median', 'count']).sort_values('median', ascending=False)
    systems_ordered = system_stats.index.tolist()
    
    # Get consistent colors
    color_dict = get_system_colors(systems_ordered)
    
    return r2_filtered, systems_ordered, color_dict


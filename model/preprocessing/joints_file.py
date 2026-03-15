class k4abt_joints(int):
    PELVIS = 0
    SPINE_NAVEL = 1
    SPINE_CHEST = 2
    NECK = 3
    CLAVICLE_LEFT = 4
    SHOULDER_LEFT = 5
    ELBOW_LEFT = 6
    WRIST_LEFT = 7
    HAND_LEFT = 8
    HANDTIP_LEFT = 9
    THUMB_LEFT = 10
    CLAVICLE_RIGHT = 11
    SHOULDER_RIGHT = 12
    ELBOW_RIGHT = 13
    WRIST_RIGHT = 14
    HAND_RIGHT = 15
    HANDTIP_RIGHT = 16
    THUMB_RIGHT = 17
    HIP_LEFT = 18
    KNEE_LEFT = 19
    ANKLE_LEFT = 20
    FOOT_LEFT = 21
    HIP_RIGHT = 22
    KNEE_RIGHT = 23
    ANKLE_RIGHT = 24
    FOOT_RIGHT = 25
    HEAD = 26
    NOSE = 27
    EYE_LEFT = 28
    EAR_LEFT = 29
    EYE_RIGHT = 30
    EAR_RIGHT = 31
    len = 32



k4abt_bones = list([
    (
        k4abt_joints.SPINE_CHEST,
        k4abt_joints.SPINE_NAVEL
    ),
    (
        k4abt_joints.SPINE_NAVEL,
        k4abt_joints.PELVIS
    ),
    (
        k4abt_joints.SPINE_CHEST,
        k4abt_joints.NECK
    ),
    (
        k4abt_joints.NECK,
        k4abt_joints.HEAD
    ),
    (
        k4abt_joints.HEAD,
        k4abt_joints.NOSE
    ),
    (
        k4abt_joints.SPINE_CHEST,
        k4abt_joints.CLAVICLE_LEFT
    ),
    (
        k4abt_joints.CLAVICLE_LEFT,
        k4abt_joints.SHOULDER_LEFT
    ),
    (
        k4abt_joints.SHOULDER_LEFT,
        k4abt_joints.ELBOW_LEFT
    ),
    (
        k4abt_joints.ELBOW_LEFT,
        k4abt_joints.WRIST_LEFT
    ),
    (
        k4abt_joints.WRIST_LEFT,
        k4abt_joints.HAND_LEFT
    ),
    (
        k4abt_joints.HAND_LEFT,
        k4abt_joints.HANDTIP_LEFT
    ),
    (
        k4abt_joints.WRIST_LEFT,
        k4abt_joints.THUMB_LEFT
    ),
    (
        k4abt_joints.PELVIS,
        k4abt_joints.HIP_LEFT
    ),
    (
        k4abt_joints.HIP_LEFT,
        k4abt_joints.KNEE_LEFT
    ),
    (
        k4abt_joints.KNEE_LEFT,
        k4abt_joints.ANKLE_LEFT
    ),
    (
        k4abt_joints.ANKLE_LEFT,
        k4abt_joints.FOOT_LEFT
    ),
    (
        k4abt_joints.NOSE,
        k4abt_joints.EYE_LEFT
    ),
    (
        k4abt_joints.EYE_LEFT,
        k4abt_joints.EAR_LEFT
    ),
    (
        k4abt_joints.SPINE_CHEST,
        k4abt_joints.CLAVICLE_RIGHT
    ),
    (
        k4abt_joints.CLAVICLE_RIGHT,
        k4abt_joints.SHOULDER_RIGHT
    ),
    (
        k4abt_joints.SHOULDER_RIGHT,
        k4abt_joints.ELBOW_RIGHT
    ),
    (
        k4abt_joints.ELBOW_RIGHT,
        k4abt_joints.WRIST_RIGHT
    ),
    (
        k4abt_joints.WRIST_RIGHT,
        k4abt_joints.HAND_RIGHT
    ),
    (
        k4abt_joints.HAND_RIGHT,
        k4abt_joints.HANDTIP_RIGHT
    ),
    (
        k4abt_joints.WRIST_RIGHT,
        k4abt_joints.THUMB_RIGHT
    ),
    (
        k4abt_joints.PELVIS,
        k4abt_joints.HIP_RIGHT
    ),
    (
        k4abt_joints.HIP_RIGHT,
        k4abt_joints.KNEE_RIGHT
    ),
    (
        k4abt_joints.KNEE_RIGHT,
        k4abt_joints.ANKLE_RIGHT
    ),
    (
        k4abt_joints.ANKLE_RIGHT,
        k4abt_joints.FOOT_RIGHT
    ),
    (
        k4abt_joints.NOSE,
        k4abt_joints.EYE_RIGHT
    ),
    (
        k4abt_joints.EYE_RIGHT,
        k4abt_joints.EAR_RIGHT
    )
])

# this is the joints to calculate the centroid from
joints_centroid = [k4abt_joints.PELVIS, k4abt_joints.SPINE_NAVEL] #,
a                = [k4abt_joints.SPINE_NAVEL,
                   k4abt_joints.SPINE_CHEST,
                   k4abt_joints.CLAVICLE_LEFT,
                   k4abt_joints.CLAVICLE_RIGHT,
                   k4abt_joints.HIP_LEFT,
                   k4abt_joints.HIP_RIGHT]
joints_centroid2 = [k4abt_joints.PELVIS]

# this is the joints to cutout in the augmentation function cutout
joints_cutout = [k4abt_joints.HANDTIP_LEFT,
                k4abt_joints.HANDTIP_RIGHT,
                k4abt_joints.THUMB_LEFT,
                k4abt_joints.THUMB_RIGHT,
                k4abt_joints.EYE_RIGHT,
                k4abt_joints.EYE_LEFT,
                k4abt_joints.EAR_RIGHT,
                k4abt_joints.EAR_LEFT,
                k4abt_joints.HAND_LEFT,
                k4abt_joints.HAND_RIGHT]

# this is the joints to use to find a two-step cycle in augmentation method cycle
joints_cycle = [k4abt_joints.ANKLE_LEFT,
                k4abt_joints.ANKLE_RIGHT,
                k4abt_joints.FOOT_LEFT,
                k4abt_joints.FOOT_RIGHT]

# this is the joints to use to find a sit to stand cycle in augmentation method cycle
joints_cycle_sts = [k4abt_joints.PELVIS,
                k4abt_joints.SHOULDER_LEFT,
                k4abt_joints.SHOULDER_RIGHT,
                k4abt_joints.KNEE_LEFT,
                k4abt_joints.KNEE_RIGHT,]

# these are the joints where the confidence is too low to keep ndex(['8_HAND_LEFT_c', '9_HANDTIP_LEFT_c', '10_THUMB_LEFT_c', '15_HAND_RIGHT_c', '16_HANDTIP_RIGHT_c', '17_THUMB_RIGHT_c']
noise_joints = [8, 9, 10,
                15, 16, 17] # eye and ear joints are also noisy

noise_bones = [(2, 1), (1, 0), (2, 3), (3, 20), (20, 21), (2, 4), (4, 5), (5, 6), (6, 7), (0, 12), (12, 13), (13, 14), (14, 15),
                (21, 22), (22, 23), (2, 8), (8, 9), (9, 10), (10, 11), (0, 16), (16, 17), (17, 18), (18, 19), (21, 24), (24, 25)]

noise_pairs =  {"pelvis": (0),
              "spine_navel": (1),
              "spine_chest": (2),
              "neck": (3),
              "clavicles": (4, 8),
              "shoulders": (5, 9),
              "elbows": (6, 10),
              "wrists": (7, 11),
              "hips": (12, 16),
              "knees": (13, 17),
              "ankles": (14, 18),
              "feet": (15, 19),
              "head": (20), 
              "nose": (21), 
              "eyes":(22, 24),
              "ears":(23, 25)}

noise_groups = {
 'L_leg': [12, 13, 14, 15],
 'R_leg': [16, 17, 18, 19],
 'L_arm': [5, 6, 7, 4],
 'R_arm': [9, 10, 11, 8],
 'torso': [0, 1, 2, 3],
 'head_full': [20, 21, 22, 24, 23, 25]
}

# the 17 indices to keep if we want to use the motionbert format
motionbert_keep_idxs = [0, 2, 3, 27, 26,
             5,  6,  7,
             12, 13, 14,
             18, 19, 20,
             22, 23, 24]

# reorder to MotionBERT?s desired order:
motionbert_order = [0,    # root=0
                    22,23,24,   # RHip,RKnee,RAnkle
                    18,19,20,   # LHip, LKnee, LAnkle
                    2,          # torso
                    3,          # neck
                    27,         # nose
                    26,         # head
                    5,6,7,      # LShoulder, LElbow, LWrist
                    12,13,14]   # RShoulder,RElbow,RWrist

motionbert_bones = list([[0, 7], [7, 8], [8, 9], [9, 10], [8, 11], [11, 12], [12, 13],
                          [8, 14], [14, 15], [15, 16], [0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6]])



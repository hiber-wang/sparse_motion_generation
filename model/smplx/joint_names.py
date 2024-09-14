# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

JOINT_NAMES = [
    'pelvis',            # 0
    'left_hip',          # 1
    'right_hip',         # 2
    'spine1',            # 3
    'left_knee',         # 4
    'right_knee',        # 5
    'spine2',            # 6
    'left_ankle',        # 7
    'right_ankle',       # 8
    'spine3',            # 9
    'left_foot',         # 10
    'right_foot',        # 11
    'neck',              # 12
    'left_collar',       # 13
    'right_collar',      # 14
    'head',              # 15
    'left_shoulder',     # 16
    'right_shoulder',    # 17
    'left_elbow',        # 18
    'right_elbow',       # 19
    'left_wrist',        # 20
    'right_wrist',       # 21
    'jaw',               # 22
    'left_eye_smplhf',   # 23
    'right_eye_smplhf',  # 24
    'left_index1',       # 25
    'left_index2',       # 26
    'left_index3',       # 27
    'left_middle1',      # 28
    'left_middle2',      # 29
    'left_middle3',      # 30
    'left_pinky1',       # 31
    'left_pinky2',       # 32
    'left_pinky3',       # 33
    'left_ring1',        # 34
    'left_ring2',        # 35
    'left_ring3',        # 36
    'left_thumb1',       # 37
    'left_thumb2',       # 38
    'left_thumb3',       # 39
    'right_index1',      # 40
    'right_index2',      # 41
    'right_index3',      # 42
    'right_middle1',     # 43
    'right_middle2',     # 44
    'right_middle3',     # 45
    'right_pinky1',      # 46
    'right_pinky2',      # 47
    'right_pinky3',      # 48
    'right_ring1',       # 49
    'right_ring2',       # 50
    'right_ring3',       # 51
    'right_thumb1',      # 52
    'right_thumb2',      # 53
    'right_thumb3',      # 54
    'nose',              # 55
    'right_eye',         # 56
    'left_eye',          # 57
    'right_ear',         # 58
    'left_ear',          # 59
    'left_big_toe',      # 60
    'left_small_toe',    # 61
    'left_heel',         # 62
    'right_big_toe',     # 63
    'right_small_toe',   # 64
    'right_heel',        # 65
    'left_thumb',        # 66
    'left_index',        # 67
    'left_middle',       # 68
    'left_ring',         # 69
    'left_pinky',        # 70
    'right_thumb',       # 71
    'right_index',       # 72
    'right_middle',      # 73
    'right_ring',        # 74
    'right_pinky',       # 75
    'right_eye_brow1',   # 76
    'right_eye_brow2',   # 77
    'right_eye_brow3',   # 78
    'right_eye_brow4',   # 79
    'right_eye_brow5',   # 80
    'left_eye_brow5',    # 81
    'left_eye_brow4',    # 82
    'left_eye_brow3',    # 83
    'left_eye_brow2',    # 84
    'left_eye_brow1',    # 85
    'nose1',             # 86
    'nose2',             # 87
    'nose3',             # 88
    'nose4',             # 89
    'right_nose_2',      # 90
    'right_nose_1',      # 91
    'nose_middle',       # 92
    'left_nose_1',       # 93
    'left_nose_2',       # 94
    'right_eye1',        # 95
    'right_eye2',        # 96
    'right_eye3',        # 97
    'right_eye4',        # 98
    'right_eye5',        # 99
    'right_eye6',        # 100
    'left_eye4',         # 101
    'left_eye3',         # 102
    'left_eye2',         # 103
    'left_eye1',         # 104
    'left_eye6',         # 105
    'left_eye5',         # 106
    'right_mouth_1',     # 107
    'right_mouth_2',     # 108
    'right_mouth_3',     # 109
    'mouth_top',         # 110
    'left_mouth_3',      # 111
    'left_mouth_2',      # 112
    'left_mouth_1',      # 113
    'left_mouth_5',      # 114, 59 in OpenPose output
    'left_mouth_4',      # 115, 58 in OpenPose output
    'mouth_bottom',      # 116
    'right_mouth_4',     # 117
    'right_mouth_5',     # 118
    'right_lip_1',       # 119
    'right_lip_2',       # 120
    'lip_top',           # 121
    'left_lip_2',        # 122
    'left_lip_1',        # 123
    'left_lip_3',        # 124
    'lip_bottom',        # 125
    'right_lip_3',       # 126
    # Face contour
    'right_contour_1',   # 127
    'right_contour_2',   # 128
    'right_contour_3',   # 129
    'right_contour_4',   # 130
    'right_contour_5',   # 131
    'right_contour_6',   # 132
    'right_contour_7',   # 133
    'right_contour_8',   # 134
    'contour_middle',    # 135
    'left_contour_8',    # 136
    'left_contour_7',    # 137
    'left_contour_6',    # 138
    'left_contour_5',    # 139
    'left_contour_4',    # 140
    'left_contour_3',    # 141
    'left_contour_2',    # 142
    'left_contour_1',    # 143
]


SMPLH_JOINT_NAMES = [
    'pelvis',
    'left_hip',
    'right_hip',
    'spine1',
    'left_knee',
    'right_knee',
    'spine2',
    'left_ankle',
    'right_ankle',
    'spine3',
    'left_foot',
    'right_foot',
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_index1',
    'left_index2',
    'left_index3',
    'left_middle1',
    'left_middle2',
    'left_middle3',
    'left_pinky1',
    'left_pinky2',
    'left_pinky3',
    'left_ring1',
    'left_ring2',
    'left_ring3',
    'left_thumb1',
    'left_thumb2',
    'left_thumb3',
    'right_index1',
    'right_index2',
    'right_index3',
    'right_middle1',
    'right_middle2',
    'right_middle3',
    'right_pinky1',
    'right_pinky2',
    'right_pinky3',
    'right_ring1',
    'right_ring2',
    'right_ring3',
    'right_thumb1',
    'right_thumb2',
    'right_thumb3',
    'nose',
    'right_eye',
    'left_eye',
    'right_ear',
    'left_ear',
    'left_big_toe',
    'left_small_toe',
    'left_heel',
    'right_big_toe',
    'right_small_toe',
    'right_heel',
    'left_thumb',
    'left_index',
    'left_middle',
    'left_ring',
    'left_pinky',
    'right_thumb',
    'right_index',
    'right_middle',
    'right_ring',
    'right_pinky',
]

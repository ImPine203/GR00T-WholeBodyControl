from .base import SkeletonBase


class H3Skeleton(SkeletonBase):
    name = "h3skel"

    bone_order_names_with_parents = [
        ("pelvis_skel", None),
        # left hip & left leg & left foot
        ("left_hip_pitch_skel", "pelvis_skel"),
        ("left_hip_roll_skel", "left_hip_pitch_skel"),
        ("left_hip_yaw_skel", "left_hip_roll_skel"),
        ("left_knee_pitch_skel", "left_hip_yaw_skel"),
        ("left_ankle_pitch_skel", "left_knee_pitch_skel"),
        ("left_ankle_roll_skel", "left_ankle_pitch_skel"),
        ("left_toe_base", "left_ankle_roll_skel"),
        # right hip & right leg & right foot
        ("right_hip_pitch_skel", "pelvis_skel"),
        ("right_hip_roll_skel", "right_hip_pitch_skel"),
        ("right_hip_yaw_skel", "right_hip_roll_skel"),
        ("right_knee_pitch_skel", "right_hip_yaw_skel"),
        ("right_ankle_pitch_skel", "right_knee_pitch_skel"),
        ("right_ankle_roll_skel", "right_ankle_pitch_skel"),
        ("right_toe_base", "right_ankle_roll_skel"),
        # waist
        ("waist_yaw_skel", "pelvis_skel"),
        ("waist_roll_skel", "waist_yaw_skel"),
        # left shoulder & left arm & left hand
        ("left_shoulder_pitch_skel", "waist_roll_skel"),
        ("left_shoulder_roll_skel", "left_shoulder_pitch_skel"),
        ("left_shoulder_yaw_skel", "left_shoulder_roll_skel"),
        ("left_elbow_pitch_skel", "left_shoulder_yaw_skel"),
        ("left_wrist_yaw_skel", "left_elbow_pitch_skel"),
        ("left_wrist_roll_skel", "left_wrist_yaw_skel"),
        ("left_wrist_pitch_skel", "left_wrist_roll_skel"),
        # right shoulder & right arm & right hand
        ("right_shoulder_pitch_skel", "waist_roll_skel"),
        ("right_shoulder_roll_skel", "right_shoulder_pitch_skel"),
        ("right_shoulder_yaw_skel", "right_shoulder_roll_skel"),
        ("right_elbow_pitch_skel", "right_shoulder_yaw_skel"),
        ("right_wrist_yaw_skel", "right_elbow_pitch_skel"),
        ("right_wrist_roll_skel", "right_wrist_yaw_skel"),
        ("right_wrist_pitch_skel", "right_wrist_roll_skel"),
        # head
        ("head_yaw_skel", "waist_roll_skel"),
        ("head_pitch_skel", "head_yaw_skel"),
    ]

    left_foot_joint_names = ["left_ankle_roll_skel", "left_toe_base"]
    right_foot_joint_names = ["right_ankle_roll_skel", "right_toe_base"]
    hip_joint_names = ["right_hip_pitch_skel", "left_hip_pitch_skel"]

    def get_skel_slice(self, skeleton: SkeletonBase):
        try:
            return [skeleton.bone_index[x] for x in self.bone_order_names]
        except KeyError:
            raise ValueError("The current skeleton contains joints that are not in the input")

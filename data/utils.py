import torch
import data.matrix as matrix


def resample_motion_fps(motion, target_length):
    N, J, C = motion.shape  # N: 帧数，J: 关节点数，C: 坐标数（通常为3）
    motion = motion.permute(1, 2, 0)

    upsampled_motion = torch.nn.functional.interpolate(motion, size=target_length, mode="linear", align_corners=True)

    upsampled_motion = upsampled_motion.permute(2, 0, 1)
    return upsampled_motion

def smpl_fk(smpl_model, body_pose, betas, global_orient=None, transl=None):
    """
    Args:
        body_pose: (B, L, 63)
        betas: (B, L, 10)
        global_orient: (B, L, 3)
    Returns:
        joints: (B, L, 22, 3)
    """
    if betas is None:
        betas = torch.zeros_like(body_pose[..., :10])
    flag = False
    if len(body_pose.shape) == 2:
        body_pose = body_pose[None]
        betas = betas[None]
        if global_orient is not None:
            global_orient = global_orient[None]
        if transl is not None:
            transl = transl[None]
        flag = True

    B, L = body_pose.shape[:2]
    if global_orient is None:
        global_orient = torch.zeros((B, L, 3), device=body_pose.device)
    aa = torch.cat([global_orient, body_pose], dim=-1).reshape(B, L, -1, 3)
    rotmat = matrix.axis_angle_to_matrix(aa)  # (B, L, 22, 3, 3)
    parents = smpl_model.bm.parents[:22]

    skeleton = smpl_model.get_skeleton(betas)[..., :22, :]  # (B, L, 22, 3)
    local_skeleton = skeleton - skeleton[:, :, parents]
    local_skeleton = torch.cat([skeleton[:, :, :1], local_skeleton[:, :, 1:]], dim=2)

    if transl is not None:
        local_skeleton[..., 0, :] += transl  # B, L, 22, 3

    mat = matrix.get_TRS(rotmat, local_skeleton)  # B, L, 22, 4, 4
    fk_mat = matrix.forward_kinematics(mat, parents)  # B, L, 22, 4, 4
    joints = matrix.get_position(fk_mat)  # B, L, 22, 3
    if flag:
        joints = joints[0]

    return joints

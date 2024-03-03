import Quaternions as Q
import Animation as A
import BVH as BVH
import numpy as np

'''relocate anim so that it starts from start_pos'''
def relocate(anim, start_pos):
    # pos: 3-dimension point
    delta_pos = start_pos - anim.positions[0, 0]
    anim.positions[:,0] += delta_pos
    return anim

'''rotate anim, so that it starts in different facing direction, delta_q is quaternion'''
# reorient: apply delta R to root rotation of all frames
def rotate_root(anim, delta_q):
    anim.rotations[:, 0] = delta_q * anim.rotations[:, 0]
    transform = np.repeat(delta_q.transforms(), anim.shape[0], axis=0)
    # Apply delta R to adjust root translation
    for f in range(0, anim.shape[0]):
        anim.positions[f,0] = np.matmul(transform[f], anim.positions[f,0])
    return anim

'''naively put anim2 at the end of anim1, not smooth, with teleportation issue'''
def concatenate_naive(anim1, anim2):
    anim1.rotations = Q.Quaternions(np.vstack((anim1.rotations.qs, anim2.rotations.qs)))
    anim1.positions = np.vstack((anim1.positions, anim2.positions))
    return anim1

'''NOTE 
for motion editing, only change anim.rotations and anim.positions
no need to change the skeleton, which is anim.orients and anim.offsets
'''

def quaternion_inverse(quaternion):
    w, x, y, z = quaternion.qs[0]
    conjugate = Q.Quaternions(np.array([w, -x, -y, -z]))
    modulus_squared = w**2 + x**2 + y**2 + z**2
    inverse = conjugate / modulus_squared
    return inverse

#TODO: smoothly connects the two motions
def concatenate(anim1, anim2, blend_frames=30):
    # Step 1: Calculate the delta rotation needed to reorient anim2 / reorient m2 to the last frame of m1
    end_rotation_anim1 = anim1.rotations[-blend_frames, 0]
    start_rotation_anim2 = anim2.rotations[0, 0]

    start_rotation_anim2_inv = quaternion_inverse(start_rotation_anim2)
    delta_rotation = end_rotation_anim1 * start_rotation_anim2_inv

    # print(delta_rotation)
    # Reorient anim2 using the calculated delta rotation
    anim2 = rotate_root(anim2, delta_rotation)

    # Step 2: Relocate anim2 to match the last frame's position of anim1
    end_position_anim1 = anim1.positions[-blend_frames, 0]
    anim2 = relocate(anim2, end_position_anim1)

    blend_pos = np.zeros((blend_frames, anim1.positions.shape[1], 3))
    blend_rot = np.zeros((blend_frames, anim1.rotations.shape[1], 4))

    for i in range(blend_frames):
        t = i / float(blend_frames - 1)
        blend_pos[i] = anim1.positions[-blend_frames + i] * (1 - t) + anim2.positions[i] * t
        blend_rot[i] = Q.Quaternions.slerp(anim1.rotations[-blend_frames + i], anim2.rotations[i], t)

    new_positions = np.vstack((anim1.positions[:-blend_frames], blend_pos, anim2.positions[blend_frames+1:]))
    new_rotations = Q.Quaternions(np.vstack((anim1.rotations.qs[:-blend_frames], blend_rot, anim2.rotations.qs[blend_frames+1:])))

    new_anim = A.Animation(new_rotations, new_positions, anim1.orients, anim1.offsets, anim1.parents)

    return new_anim

def find_joint_index(joint_names, joint_name):
    try:
        return joint_names.index(joint_name)
    except ValueError:
        raise ValueError(f"Joint name {joint_name} not found in the list of joint names.")

right_arm_joint_names = [
    'RightShoulder',
    'RightArm',
    'RightForeArm',
    'RightHand',
    'RightFingerBase',
    'RightHandIndex1',
    'RThumb'
]

def resample_motion(motion_data, source_fps, target_fps):
    source_frame_time = 1.0 / source_fps
    target_frame_time = 1.0 / target_fps

    resample_ratio = target_fps / source_fps

    n_source_frames = motion_data.shape[0]

    n_target_frames = int(np.ceil(n_source_frames * resample_ratio))

    resampled_motion_data = np.zeros((n_target_frames, motion_data.shape[1], motion_data.shape[2]), dtype=motion_data.dtype)
    # print(resampled_motion_data.shape)

    for i in range(1, n_target_frames - 1):
        target_time = i * target_frame_time

        source_index = int(target_time // source_frame_time)
        source_next_index = min(source_index + 1, n_source_frames - 1)

        alpha = (target_time - source_index * source_frame_time) / source_frame_time

        q0 = Q.Quaternions(motion_data[source_index])
        q1 = Q.Quaternions(motion_data[source_next_index])
        resampled_motion_data[i] = Q.Quaternions.slerp(q0, q1, alpha).qs

    resampled_motion_data[0] = motion_data[0]
    resampled_motion_data[-1] = motion_data[-1]

    return resampled_motion_data

#TODO: splice right arm waving from anim2 into anim1 walking
def splice(anim1, anim2, joint_names_anim1, joint_names_anim2):
    n_frames_anim1 = anim1.shape[0]
    n_frames_anim2 = anim2.shape[0]

    anim2_resampled = resample_motion(anim2.rotations.qs, n_frames_anim2, n_frames_anim1)

    anim2_resampled = A.Animation(
        Q.Quaternions(anim2_resampled),
        anim2.positions,
        anim2.orients,
        anim2.offsets,
        anim2.parents
    )

    right_arm_indices_anim2 = [joint_names_anim2.index(name) for name in right_arm_joint_names if name in joint_names_anim2]

    for i in range(n_frames_anim1):
        for idx in right_arm_indices_anim2:
            anim1.rotations[i][idx] = anim2_resampled.rotations[i][idx]
            if anim1.positions is not None and anim2_resampled.positions is not None:
                anim1.positions[i][idx] = anim2_resampled.positions[i][idx]

    return anim1

'''load the walking motion'''
filepath = ''
filename_walk = filepath + '16_26.bvh'
anim_walk, joint_names_walk, frametime_walk = BVH.load(filename_walk)
# print(joint_names_walk)

'''load the waving motion'''
filename_wave = filepath + '111_37.bvh'
anim_wave, joint_names_wave, frametime_wave = BVH.load(filename_wave)

# '''simple editing'''
# anim_concat_naive = concatenate_naive(anim_walk, anim_wave)
# filename_concat_naive = filepath + 'concat_naive.bvh'
# BVH.save(filename_concat_naive, anim_concat_naive, joint_names_walk, frametime_walk)

'''concatenate two animation slip'''
concatenated_animation = concatenate(anim_walk, anim_wave)
filename_concatenated = filepath + 'result_concatenated.bvh'
BVH.save(filename_concatenated, concatenated_animation, joint_names_walk, frametime_walk)

spliced_animation = splice(anim_walk, anim_wave, joint_names_walk, joint_names_wave)
filename_spliced = filepath + 'result_spliced.bvh'
BVH.save(filename_spliced, spliced_animation, joint_names_walk, frametime_walk)

# anim_rotated = rotate_root(anim_walk, Q.Quaternions.from_euler(np.array([0, 90, 0])))
# filename_rotated = filepath + 'rotated.bvh'
# BVH.save(filename_rotated, anim_rotated, joint_names_walk, frametime_walk)
# print('DONE!')
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

#TODO: smoothly connects the two motions
def concatenate(anim1, anim2):
    pass

#TODO: splice right arm waving from anim2 into anim1 walking
def splice(anim1, anim2):
    pass


'''load the walking motion'''
filepath = ''
filename_walk = filepath + '16_26.bvh'
anim_walk, joint_names_walk, frametime_walk = BVH.load(filename_walk)

'''load the waving motion'''
filename_wave = filepath + '111_37.bvh'
anim_wave, joint_names_wave, frametime_wave = BVH.load(filename_wave)

'''simple editing'''
anim_concat_naive = concatenate_naive(anim_walk, anim_wave)
filename_concat_naive = filepath + 'concat_naive.bvh'
BVH.save(filename_concat_naive, anim_concat_naive, joint_names_walk, frametime_walk)

anim_rotated = rotate_root(anim_walk, Q.Quaternions.from_euler(np.array([0, 90, 0])))
filename_rotated = filepath + 'rotated.bvh'
BVH.save(filename_rotated, anim_rotated, joint_names_walk, frametime_walk)
print('DONE!')
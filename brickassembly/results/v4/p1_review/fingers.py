import sys; sys.path.insert(0,"/home/bakasakib/Documents/ISDN_Robofab/brickassembly")
import dual_arm_sim as das, numpy as np
b = das.build_arm()
for i,bd in enumerate(b.shape_body):
    if bd in (12,13):
        print(i, bd, b.shape_type[i], "ke",b.shape_material_ke[i],"kd",b.shape_material_kd[i],"mu",b.shape_material_mu[i],"margin",b.shape_margin[i],"scale",b.shape_scale[i])
print("body masses 12,13", b.body_mass[12], b.body_mass[13], "armature", b.joint_armature[7:9])
ex=das.ex
print("brick mass proto", ex.BRICK_MASS, "KE", ex.BRICK_KE, "KD", ex.BRICK_KD, "wall", ex.WALL_THICKNESS, "stud_h", ex.STUD_HEIGHT)

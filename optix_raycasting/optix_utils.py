import optix as ox
import cupy as cp
import numpy as np
import os
import math
from torch.utils.dlpack import to_dlpack

#Read the SIGMA_THRESHOLD in respective cuda_train/forward/gaussians_aabb.h
header_train_forward= os.path.join(os.path.dirname(__file__), "cuda_train/forward", "gaussians_aabb.h")
header_train_backward= os.path.join(os.path.dirname(__file__), "cuda_train/backward", "gaussians_aabb.h")
header_test= os.path.join(os.path.dirname(__file__), "cuda_test", "gaussians_aabb.h")
header_gui= os.path.join(os.path.dirname(__file__), "cuda_gui", "gaussians_aabb.h")

def get_SIGMA_THRESHOLD(file_name):
    file=open(file_name, "r")
    lines=file.readlines()
    file.close()
    for line in lines:
        if "SIGMA_THRESHOLD" in line:
            SIGMA_THRESHOLD=line.split()[2]
            #Remove f at the end of the number
            SIGMA_THRESHOLD=float(SIGMA_THRESHOLD[:-1])
            break
    return SIGMA_THRESHOLD

sigma_threshold_train_forward=get_SIGMA_THRESHOLD(header_train_forward)
sigma_threshold_train_backward=get_SIGMA_THRESHOLD(header_train_backward)
sigma_threshold_test=get_SIGMA_THRESHOLD(header_test)
sigma_threshold_gui=get_SIGMA_THRESHOLD(header_gui)
if sigma_threshold_train_forward!=sigma_threshold_train_backward or sigma_threshold_train_forward!=sigma_threshold_test or sigma_threshold_train_forward!=sigma_threshold_gui:
    raise ValueError("SIGMA_THRESHOLD is not the same in all headers")
else:
    SIGMA_THRESHOLD=sigma_threshold_train_forward


_ELLIPSOID_BBOX = cp.ElementwiseKernel(
    in_params='''
        float32 c0, float32 c1, float32 c2,
        float32 s0, float32 s1, float32 s2,
        float32 w,  float32 x,  float32 y, float32 z,
        float32 dens,
        float32 sigma
    ''',
    out_params='''
        float32 out0, float32 out1, float32 out2,
        float32 out3, float32 out4, float32 out5
    ''',
    operation=r'''
        float t = dens / sigma;
        float d = logf(t * t);
        if (!(d > 0.0f)) d = 0.0f;
        d = sqrtf(d);

        float W = w, X = x, Y = y, Z = z;

        float n = W*W + X*X + Y*Y + Z*Z;
        float invn = rsqrtf(n + 1e-20f);
        W *= invn; X *= invn; Y *= invn; Z *= invn;

        float L1x = 1.f - 2.f*(Y*Y + Z*Z);
        float L1y = 2.f*(X*Y - W*Z);
        float L1z = 2.f*(X*Z + W*Y);

        float L2x = 2.f*(X*Y + W*Z);
        float L2y = 1.f - 2.f*(X*X + Z*Z);
        float L2z = 2.f*(Y*Z - W*X);

        float L3x = 2.f*(X*Z - W*Y);
        float L3y = 2.f*(Y*Z + W*X);
        float L3z = 1.f - 2.f*(X*X + Y*Y);

        float hx = d * sqrtf( (s0*L1x)*(s0*L1x) + (s1*L1y)*(s1*L1y) + (s2*L1z)*(s2*L1z) );
        float hy = d * sqrtf( (s0*L2x)*(s0*L2x) + (s1*L2y)*(s1*L2y) + (s2*L2z)*(s2*L2z) );
        float hz = d * sqrtf( (s0*L3x)*(s0*L3x) + (s1*L3y)*(s1*L3y) + (s2*L3z)*(s2*L3z) );

        out0 = c0 - hx;  out1 = c1 - hy;  out2 = c2 - hz;
        out3 = c0 + hx;  out4 = c1 + hy;  out5 = c2 + hz;
    ''',
    name='ellipsoid_bbox_kernel'
)

def ellipsoids_bbox_from_quat_old(quaternions, centers, scales, densities):
    quaternions = quaternions.astype(cp.float32, copy=False)
    centers     = centers.astype(cp.float32, copy=False)
    scales      = scales.astype(cp.float32, copy=False)
    densities   = densities.astype(cp.float32, copy=False)

    w, x, y, z  = quaternions[:,0], quaternions[:,1], quaternions[:,2], quaternions[:,3]
    c0, c1, c2  = centers[:,0], centers[:,1], centers[:,2]
    s0, s1, s2  = scales[:,0],  scales[:,1],  scales[:,2]

    out = cp.empty((centers.shape[0], 6), dtype=cp.float32)
    _ELLIPSOID_BBOX(
        c0, c1, c2,
        s0, s1, s2,
        w, x, y, z,
        densities, cp.float32(SIGMA_THRESHOLD),
        out[:,0], out[:,1], out[:,2], out[:,3], out[:,4], out[:,5]
    )
    return out

def ellipsoids_bbox_from_quat(quaternions, centers, scales, densities):
    # Cast + colonnes contiguës
    q = quaternions.astype(cp.float32, copy=False)
    c = centers.astype(cp.float32, copy=False)
    s = scales.astype(cp.float32, copy=False)
    d = densities.astype(cp.float32, copy=False)

    w,x,y,z  = [cp.ascontiguousarray(q[:,i]) for i in range(4)]
    c0,c1,c2 = [cp.ascontiguousarray(c[:,i]) for i in range(3)]
    s0,s1,s2 = [cp.ascontiguousarray(s[:,i]) for i in range(3)]

    N = c.shape[0]
    xmin = cp.empty(N, dtype=cp.float32)
    ymin = cp.empty_like(xmin); zmin = cp.empty_like(xmin)
    xmax = cp.empty_like(xmin); ymax = cp.empty_like(xmin); zmax = cp.empty_like(xmin)

    _ELLIPSOID_BBOX(
        c0,c1,c2, s0,s1,s2, w,x,y,z, d, cp.float32(SIGMA_THRESHOLD),
        xmin,ymin,zmin, xmax,ymax,zmax
    )

    # Réductions rapides (CUB)
    bb_min = cp.stack([xmin.min(), ymin.min(), zmin.min()])
    bb_max = cp.stack([xmax.max(), ymax.max(), zmax.max()])

    # Si tu veux aussi la matrice N×6 :
    bboxes = cp.stack([xmin,ymin,zmin,xmax,ymax,zmax], axis=1)
    return bboxes, bb_min, bb_max

def quaternion_to_rotation(quaternion):
    # quaternion = [w, x, y, z]
    w = quaternion[:, 0]
    x = quaternion[:, 1]
    y = quaternion[:, 2]
    z = quaternion[:, 3]
    L1=cp.zeros((quaternion.shape[0],3),dtype=cp.float32)
    L2=cp.zeros((quaternion.shape[0],3),dtype=cp.float32)
    L3=cp.zeros((quaternion.shape[0],3),dtype=cp.float32)
    L1[:,0],L1[:,1],L1[:,2]=1 - 2 * (y**2 + z**2), 2 * (x*y - w*z), 2 * (x*z + w*y)
    L2[:,0],L2[:,1],L2[:,2]=2 * (x*y + w*z), 1 - 2 * (x**2 + z**2), 2 * (y*z - w*x)
    L3[:,0],L3[:,1],L3[:,2]=2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x**2 + y**2)
    return L1,L2,L3

def compute_spheres_bbox(centers,scales):
    out = cp.empty((centers.shape[0], 6), dtype='f4')
    out[:, :3] = centers - 3*scales
    out[:, 3:] = centers + 3*scales
    return out

def compute_ellipsoids_bbox(centers,scales,L1,L2,L3,densities):
    delta=cp.log((densities/SIGMA_THRESHOLD)**2)
    delta[delta<0]=0
    delta=cp.sqrt(delta)

    out = cp.empty((centers.shape[0], 6), dtype='f4')
    scales_L1 = cp.linalg.norm(delta[:,None]*scales * L1, axis=1)
    scales_L2 = cp.linalg.norm(delta[:,None]*scales * L2, axis=1)
    scales_L3 = cp.linalg.norm(delta[:,None]*scales * L3, axis=1)
    #I want a Nx3 array with [L1,L2,L3] for each ellipsoid
    out[:, :3] = centers - cp.vstack([scales_L1,scales_L2,scales_L3]).T
    out[:, 3:] = centers + cp.vstack([scales_L1,scales_L2,scales_L3]).T
    return out

def compute_vol_bbox(centers,scales,L1,L2,L3,densities):

    delta=cp.log((densities/SIGMA_THRESHOLD)**2)
    delta[delta<0]=0
    delta=cp.sqrt(delta)

    out = cp.empty((centers.shape[0], 6), dtype='f4')
    scales_L1 = cp.linalg.norm(delta[:,None]*scales * L1, axis=1)
    scales_L2 = cp.linalg.norm(delta[:,None]*scales * L2, axis=1)
    scales_L3 = cp.linalg.norm(delta[:,None]*scales * L3, axis=1)
    #I want a Nx3 array with [L1,L2,L3] for each ellipsoid
    out[:, :3] = centers - cp.vstack([scales_L1,scales_L2,scales_L3]).T
    out[:, 3:] = centers + cp.vstack([scales_L1,scales_L2,scales_L3]).T
    return out

def create_acceleration_structure(ctx, bboxes):
    build_input = ox.BuildInputCustomPrimitiveArray([bboxes], num_sbt_records=1, flags=[ox.GeometryFlags.REQUIRE_SINGLE_ANYHIT_CALL])
    gas = ox.AccelerationStructure(ctx, [build_input], compact=False,allow_update=True,prefer_fast_build=False)
    return gas

def update_acceleration_structure(gas, bboxes):
    build_input = ox.BuildInputCustomPrimitiveArray([bboxes], num_sbt_records=1, flags=[ox.GeometryFlags.REQUIRE_SINGLE_ANYHIT_CALL])
    gas.update(build_input)

def create_instance_acceleration_structure(ctx, gas, transform):
    instance = ox.Instance(traversable=gas,instance_id=0,flags=ox.InstanceFlags.NONE,transform=transform)
    build_inputs=ox.BuildInputInstanceArray([instance])
    ias = ox.AccelerationStructure(ctx, build_inputs=build_inputs, compact=False,allow_update=True,prefer_fast_build=False)
    return ias

def create_context(log):
    logger = ox.Logger(log)
    ctx = ox.DeviceContext(validation_mode=False, log_callback_function=logger, log_callback_level=4)
    ctx.cache_enabled = False
    return ctx

def create_module(ctx, pipeline_opts,stage,debug_level=ox.CompileDebugLevel.DEFAULT,opt_level=ox.CompileOptimizationLevel.DEFAULT):
    script_dir = os.path.dirname(__file__)
    if stage=="forward":
        cuda_src = os.path.join(script_dir, "cuda_train/forward", "gaussians_aabb.cu")
    elif stage=="backward":
        cuda_src = os.path.join(script_dir, "cuda_train/backward", "gaussians_aabb.cu")
    elif stage=="test":
        cuda_src = os.path.join(script_dir, "cuda_test", "gaussians_aabb.cu")
    elif stage=="gui":
        cuda_src = os.path.join(script_dir, "cuda_gui", "gaussians_aabb.cu")
    else:
        raise ValueError("stage must be 'forward', 'backward', 'test' or 'gui'")
    compile_opts = ox.ModuleCompileOptions(debug_level=debug_level , opt_level=opt_level)
    module = ox.Module(ctx, cuda_src, compile_opts, pipeline_opts)
    return module

def create_program_groups_train(ctx, module):
    raygen_grp = ox.ProgramGroup.create_raygen(ctx, module, "__raygen__rg")
    miss_grp = ox.ProgramGroup.create_miss(ctx, module, "__miss__ms")
    hit_grp = ox.ProgramGroup.create_hitgroup(ctx, module,
                                              entry_function_IS="__intersection__gaussian",
                                              entry_function_AH="__anyhit__ah")
    return raygen_grp, miss_grp, hit_grp

def create_program_groups(ctx, module):
    raygen_grp = ox.ProgramGroup.create_raygen(ctx, module, "__raygen__rg")
    miss_grps = []
    miss_grp_ch=ox.ProgramGroup.create_miss(ctx, module, "__miss__ms_ch")
    miss_grp_ah=ox.ProgramGroup.create_miss(ctx, module, "__miss__ms_ah")
    miss_grps.append(miss_grp_ch)
    miss_grps.append(miss_grp_ah)
    hit_grps = []
    hit_grp_ch = ox.ProgramGroup.create_hitgroup(ctx, module,
                                                entry_function_IS="__intersection__gaussian",
                                                entry_function_CH="__closesthit__ch")
    hit_grp_ah = ox.ProgramGroup.create_hitgroup(ctx, module,
                                              entry_function_IS="__intersection__gaussian",
                                              entry_function_AH="__anyhit__ah")
    hit_grps.append(hit_grp_ch)
    hit_grps.append(hit_grp_ah)
    return [raygen_grp]+miss_grps+hit_grps


def create_pipeline(ctx, program_grps, pipeline_options,debug_level=ox.CompileDebugLevel.DEFAULT):
    link_opts = ox.PipelineLinkOptions(max_trace_depth=1, debug_level=debug_level)

    pipeline = ox.Pipeline(ctx, compile_options=pipeline_options, link_options=link_opts, program_groups=program_grps)
    pipeline.compute_stack_sizes(1,  # max_trace_depth
                                 0,  # max_cc_depth
                                 0)  # max_dc_depth
    return pipeline

def create_sbt(program_grps):
    raygen_grp, miss_grp, hit_grp = program_grps[0], program_grps[1:3], program_grps[3:]

    raygen_sbt = ox.SbtRecord(raygen_grp)
    miss_sbt = ox.SbtRecord(miss_grp)
    hit_sbt=ox.SbtRecord(hit_grp)

    sbt = ox.ShaderBindingTable(raygen_record=raygen_sbt, miss_records=miss_sbt, hitgroup_records=hit_sbt)
    return sbt

def launch_pipeline_forward(pipeline : ox.Pipeline, sbt, gas,bbox_min,bbox_max,dt_step,
                    dynamic_sampling,
                    camera,
                    densities,color_features,positions,scales,quaternions,
                    max_prim_slice,iteration,jitter,rnd_sample, supersampling, white_background,
                    hit_prim_idx):
    ray_size=(camera.image_width*supersampling[0],camera.image_height*supersampling[1])
    params_tmp = [
        ( 'u4', 'rnd_sample'),
        ( 'u4', 'max_prim_slice'),
        ( '3f4', 'bbox_min'),
        ( '3f4', 'bbox_max'),
        ('f4', 'dt_step'),
        ('u4','dynamic_sampling'),
        ('u4', 'image_width'),
        ('u4', 'image_height'),
        ('3f4', 'cam_eye'),
        ('3f4', 'cam_u'),
        ('3f4', 'cam_v'),
        ('3f4', 'cam_w'),
        ('4f4', 'cam_intr'),
        ( 'u8', 'prims'),
        ( 'u8', 'hit_prim_idx'),
        ( 'u8', 'ray_colors'),
        ( 'u8', 'handle'),
        ( 'u4', 'white_background')
    ]

    params = ox.LaunchParamsRecord(names=[p[1] for p in params_tmp],
                                formats=[p[0] for p in params_tmp])
    params['rnd_sample'] = rnd_sample
    params['max_prim_slice'] = max_prim_slice
    params['bbox_min'] = bbox_min.get()
    params['bbox_max'] = bbox_max.get()
    params['dt_step'] = dt_step
    params['dynamic_sampling'] = 1 if dynamic_sampling else 0
    params['image_width'] = camera.image_width*supersampling[0]
    params['image_height'] = camera.image_height*supersampling[1]
    params['cam_eye'] = cp.from_dlpack(to_dlpack(camera.camera_center)).get()
    params['cam_u'] = cp.from_dlpack(to_dlpack(camera.world_view_transform[:3, 0])).get()
    params['cam_v'] = cp.from_dlpack(to_dlpack(camera.world_view_transform[:3, 1])).get()
    params['cam_w'] = cp.from_dlpack(to_dlpack(camera.world_view_transform[:3, 2])).get()

    # cam_intr=cam_tan_half_fovx, cam_tan_half_fovy, principal_point_x, principal_point_y
    params['cam_intr']=np.array([math.tan(camera.FoVx / 2), math.tan(camera.FoVy / 2), camera.Px, camera.Py], dtype=np.float32)

    # N
    N = color_features.shape[0]
    # rgba4 = [rgb, density]
    rgba4 = cp.empty((N, 4), dtype=cp.float32)
    rgba4[:, :3] = color_features*densities[:, None]  # (N,3) * (N,1) -> (N,3)
    rgba4[:,  3] = densities

    # positions_relative4 = [pos_rel, 0]
    positions_origin = camera.camera_center - positions  # (N,3)
    positions_relative4 = cp.empty((N, 4), dtype=cp.float32)
    positions_relative4[:, :3] = positions_origin
    positions_relative4[:,  3] = 0.0

    # inv_scales4 = [1/scale, 0]
    inv_scales = cp.reciprocal(scales)                   # (N,3)
    # inv_scales4 = cp.empty((N, 4), dtype=cp.float32)
    # inv_scales4[:, :3] = inv_scales
    # inv_scales4[:,  3] = 0.0

    # inv_scales_intersect4 = [1/scale_intersect, 0]
    k = cp.sqrt(2.0 * cp.log(densities/ SIGMA_THRESHOLD))
    inv_scales_intersect = inv_scales / k[:,None]
    inv_scales_intersect4 = cp.empty((N, 4), dtype=cp.float32)
    inv_scales_intersect4[:, :3] = inv_scales_intersect
    inv_scales_intersect4[:,  3] = k

    # quaternions : (N,4) float32 déjà prêts

    # --- AoS 64B: packer tout dans (N,16) float32 ---
    prims = cp.empty((N, 16), dtype=cp.float32)
    prims[:,  0: 4] = positions_relative4    # pos4
    prims[:,  4: 8] = inv_scales_intersect4            # inv_inter4
    prims[:,  8:12] = quaternions            # quat4
    prims[:, 12:16] = rgba4                  # rgba4
    prims = cp.ascontiguousarray(prims)

    params['prims'] = prims.data.ptr

    params['hit_prim_idx'] = hit_prim_idx.data.ptr

    ray_colors=cp.zeros((ray_size[0]*ray_size[1],3), dtype=cp.float32)
    params['ray_colors']=ray_colors.data.ptr

    params['handle'] = gas.handle

    if white_background:
        params['white_background'] = 1
    else:
        params['white_background'] = 0

    stream = cp.cuda.Stream()

    pipeline.launch(sbt, dimensions=ray_size, params=params, stream=stream)
    stream.synchronize()
    return ray_colors

##################################################################
def reduce_temp_grads(
    N_temp: int,
    densities_grad,          # shape: (P*N_temp,)
    color_features_grad,     # shape: (P*N_temp, 3)
    positions_grad,          # shape: (P*N_temp, 3)
    scales_grad,             # shape: (P*N_temp, 3)
    quaternions_grad         # shape: (P*N_temp, 4)
):
    # S'assure que le kernel OptiX a fini avant la réduction
    cp.cuda.runtime.deviceSynchronize()

    P = densities_grad.size // N_temp
    assert densities_grad.size % N_temp == 0
    assert color_features_grad.shape[0] == P * N_temp and color_features_grad.shape[1] == 3
    assert positions_grad.shape[0]      == P * N_temp and positions_grad.shape[1]      == 3
    assert scales_grad.shape[0]         == P * N_temp and scales_grad.shape[1]         == 3
    assert quaternions_grad.shape[0]    == P * N_temp and quaternions_grad.shape[1]    == 4

    # Vues (reshape ne copie pas)
    dens_v   = densities_grad.reshape(P, N_temp)
    color_v  = color_features_grad.reshape(P, N_temp, 3)
    pos_v    = positions_grad.reshape(P, N_temp, 3)
    scale_v  = scales_grad.reshape(P, N_temp, 3)
    quat_v   = quaternions_grad.reshape(P, N_temp, 4)

    # Réductions
    densities_grad_final       = dens_v.sum(axis=1)                  # (P,)
    color_features_grad_final  = color_v.sum(axis=1)                 # (P,3)
    positions_grad_final       = pos_v.sum(axis=1)                   # (P,3)
    scales_grad_final          = scale_v.sum(axis=1)                 # (P,3)
    quaternions_grad_final     = quat_v.sum(axis=1)                  # (P,4)

    return (densities_grad_final,
            color_features_grad_final,
            positions_grad_final,
            scales_grad_final,
            quaternions_grad_final)
            
def launch_pipeline_backward(pipeline : ox.Pipeline, sbt, gas,bbox_min,bbox_max,dt_step,
                    dynamic_sampling,
                    camera,
                    densities,color_features,positions,scales,quaternions,
                    ray_colors,dloss_dray_colors,
                    max_prim_slice,iteration,jitter,rnd_sample,supersampling,
                    hit_prim_idx):
    # ray_size=(camera.image_height*camera.image_width*supersampling[0]*supersampling[1],)
    ray_size=(camera.image_width*supersampling[0],camera.image_height*supersampling[1])
    params_tmp = [
        ('u4', 'rnd_sample'),
        ( 'u4', 'max_prim_slice'),
        ( '3f4', 'bbox_min'),
        ( '3f4', 'bbox_max'),
        ('f4', 'dt_step'),
        ('u4','dynamic_sampling'),
        ('u4', 'image_width'),
        ('u4', 'image_height'),
        ('3f4', 'cam_eye'),
        ('3f4', 'cam_u'),
        ('3f4', 'cam_v'),
        ('3f4', 'cam_w'),
        ('4f4', 'cam_intr'),
        ( 'u8', 'prims'),
        ( 'u8', 'hit_prim_idx'),
        ( 'u8', 'ray_colors'),
        ( 'u8', 'dloss_dray_colors'),
        ( 'u8', 'densities_grad'),
        ( 'u8', 'color_features_grad'),
        ( 'u8', 'positions_grad'),
        ( 'u8', 'scales_grad'),
        ( 'u8', 'quaternions_grad'),
        ( 'u8', 'handle')
    ]

    params = ox.LaunchParamsRecord(names=[p[1] for p in params_tmp],
                                formats=[p[0] for p in params_tmp])
    params['rnd_sample'] = rnd_sample
    params['max_prim_slice'] = max_prim_slice
    params['bbox_min'] = bbox_min.get()
    params['bbox_max'] = bbox_max.get()
    params['dt_step'] = dt_step
    params['dynamic_sampling'] = 1 if dynamic_sampling else 0
    params['image_width'] = camera.image_width*supersampling[0]
    params['image_height'] = camera.image_height*supersampling[1]
    params['cam_eye'] = cp.from_dlpack(to_dlpack(camera.camera_center)).get()
    params['cam_u'] = cp.from_dlpack(to_dlpack(camera.world_view_transform[:3, 0])).get()
    params['cam_v'] = cp.from_dlpack(to_dlpack(camera.world_view_transform[:3, 1])).get()
    params['cam_w'] = cp.from_dlpack(to_dlpack(camera.world_view_transform[:3, 2])).get()

    # params['cam_tan_half_fov'] = cp.array([cp.tan(camera.FoVx / 2), cp.tan(camera.FoVy / 2)]).get()
    # params['cam_tan_half_fovx'] = math.tan(camera.FoVx / 2)
    # params['cam_tan_half_fovy'] = math.tan(camera.FoVy / 2)

    # params['principal_point_x'] = camera.Px
    # params['principal_point_y'] = camera.Py
    params['cam_intr']=np.array([math.tan(camera.FoVx / 2), math.tan(camera.FoVy / 2), camera.Px, camera.Py], dtype=np.float32)

    # N
    N = color_features.shape[0]
    # rgba4 = [rgb, density]
    rgba4 = cp.empty((N, 4), dtype=cp.float32)
    rgba4[:, :3] = color_features*densities[:, None]  # (N,3) * (N,1) -> (N,3)
    rgba4[:,  3] = densities

    # positions_relative4 = [pos_rel, 0]
    positions_origin = camera.camera_center - positions  # (N,3)
    positions_relative4 = cp.empty((N, 4), dtype=cp.float32)
    positions_relative4[:, :3] = positions_origin
    positions_relative4[:,  3] = 0.0

    # inv_scales4 = [1/scale, 0]
    inv_scales = cp.reciprocal(scales)                   # (N,3)
    # inv_scales4 = cp.empty((N, 4), dtype=cp.float32)
    # inv_scales4[:, :3] = inv_scales
    # inv_scales4[:,  3] = 0.0

    # inv_scales_intersect4 = [1/scale_intersect, 0]
    k = cp.sqrt(2.0 * cp.log(densities/ SIGMA_THRESHOLD))
    inv_scales_intersect = inv_scales / k[:,None]
    inv_scales_intersect4 = cp.empty((N, 4), dtype=cp.float32)
    inv_scales_intersect4[:, :3] = inv_scales_intersect
    inv_scales_intersect4[:,  3] = k

    # quaternions : (N,4) float32 déjà prêts

    # --- AoS 64B: packer tout dans (N,16) float32 ---
    prims = cp.empty((N, 16), dtype=cp.float32)
    prims[:,  0: 4] = positions_relative4    # pos4
    prims[:,  4: 8] = inv_scales_intersect4            # inv_inter4
    prims[:,  8:12] = quaternions            # quat4
    prims[:, 12:16] = rgba4                  # rgba4
    prims = cp.ascontiguousarray(prims)

    params['prims'] = prims.data.ptr

    # hit_prim_idx=cp.zeros((ray_size[0]*max_prim_slice), dtype=cp.int32)

    params['hit_prim_idx'] = hit_prim_idx.data.ptr

    params['ray_colors']=ray_colors.data.ptr
    params['dloss_dray_colors']=dloss_dray_colors.data.ptr

    # import torch
    # start=torch.cuda.Event(enable_timing=True)
    # end=torch.cuda.Event(enable_timing=True)
    # start.record()

    N_temp=4
    densities_grad=cp.zeros((densities.shape[0]*N_temp), dtype=cp.float32)
    params['densities_grad']=densities_grad.data.ptr
    color_features_grad=cp.zeros((positions.shape[0]*N_temp,3), dtype=cp.float32)
    params['color_features_grad']=color_features_grad.data.ptr
    positions_grad=cp.zeros((positions.shape[0]*N_temp,3), dtype=cp.float32)
    scales_grad=cp.zeros((scales.shape[0]*N_temp,3), dtype=cp.float32)
    params['positions_grad']=positions_grad.data.ptr
    params['scales_grad']=scales_grad.data.ptr
    quaternions_grad=cp.zeros((quaternions.shape[0]*N_temp,4), dtype=cp.float32)
    params['quaternions_grad']=quaternions_grad.data.ptr

    # end.record()
    # torch.cuda.synchronize()
    # print("Alloc grad time (ms):", start.elapsed_time(end))
    params['handle'] = gas.handle

    stream = cp.cuda.Stream()
    pipeline.launch(sbt, dimensions=ray_size, params=params, stream=stream)
    stream.synchronize()

    # return densities_grad,color_features_grad,positions_grad,scales_grad,quaternions_grad
    return reduce_temp_grads(
        N_temp,
        densities_grad,
        color_features_grad,
        positions_grad,
        scales_grad,
        quaternions_grad
    )
##################################################################

def launch_pipeline_test(pipeline : ox.Pipeline, sbt, gas,bbox_min,bbox_max,dt_step,dynamic_sampling,
                    camera,
                    densities,color_features,positions,scales,quaternions,
                    max_prim_slice,rnd_sample,supersampling,white_background,hit_prim_idx):
    ray_size=(camera.image_width*supersampling[0],camera.image_height*supersampling[1])
    params_tmp = [
        ( 'u4', 'rnd_sample'),
        ( 'u4', 'max_prim_slice'),
        ( '3f4', 'bbox_min'),
        ( '3f4', 'bbox_max'),
        ('f4', 'dt_step'),
        ('u4','dynamic_sampling'),
        ('u4', 'image_width'),
        ('u4', 'image_height'),
        ('3f4', 'cam_eye'),
        ('3f4', 'cam_u'),
        ('3f4', 'cam_v'),
        ('3f4', 'cam_w'),
        ('4f4', 'cam_intr'),
        ( 'u8', 'prims'),
        ( 'u8', 'hit_prim_idx'),
        ( 'u8', 'ray_colors'),
        ( 'u8', 'handle'),
        ( 'u4', 'white_background')
    ]

    params = ox.LaunchParamsRecord(names=[p[1] for p in params_tmp],
                                formats=[p[0] for p in params_tmp])
    params['rnd_sample'] = rnd_sample
    params['max_prim_slice'] = max_prim_slice
    params['bbox_min'] = bbox_min.get()
    params['bbox_max'] = bbox_max.get()
    params['dt_step'] = dt_step
    params['dynamic_sampling'] = 1 if dynamic_sampling else 0
    params['image_width'] = camera.image_width*supersampling[0]
    params['image_height'] = camera.image_height*supersampling[1]
    params['cam_eye'] = cp.from_dlpack(to_dlpack(camera.camera_center)).get()
    params['cam_u'] = cp.from_dlpack(to_dlpack(camera.world_view_transform[:3, 0])).get()
    params['cam_v'] = cp.from_dlpack(to_dlpack(camera.world_view_transform[:3, 1])).get()
    params['cam_w'] = cp.from_dlpack(to_dlpack(camera.world_view_transform[:3, 2])).get()

    # cam_intr=cam_tan_half_fovx, cam_tan_half_fovy, principal_point_x, principal_point_y
    params['cam_intr']=np.array([math.tan(camera.FoVx / 2), math.tan(camera.FoVy / 2), camera.Px, camera.Py], dtype=np.float32)

    # N
    N = color_features.shape[0]
    # rgba4 = [rgb, density]
    rgba4 = cp.empty((N, 4), dtype=cp.float32)
    rgba4[:, :3] = color_features*densities[:, None]  # (N,3) * (N,1) -> (N,3)
    rgba4[:,  3] = densities

    # positions_relative4 = [pos_rel, 0]
    positions_origin = camera.camera_center - positions  # (N,3)
    positions_relative4 = cp.empty((N, 4), dtype=cp.float32)
    positions_relative4[:, :3] = positions_origin
    positions_relative4[:,  3] = 0.0

    # inv_scales4 = [1/scale, 0]
    inv_scales = cp.reciprocal(scales)                   # (N,3)
    # inv_scales4 = cp.empty((N, 4), dtype=cp.float32)
    # inv_scales4[:, :3] = inv_scales
    # inv_scales4[:,  3] = 0.0

    # inv_scales_intersect4 = [1/scale_intersect, 0]
    k = cp.sqrt(2.0 * cp.log(densities/ SIGMA_THRESHOLD))
    inv_scales_intersect = inv_scales / k[:,None]
    inv_scales_intersect4 = cp.empty((N, 4), dtype=cp.float32)
    inv_scales_intersect4[:, :3] = inv_scales_intersect
    inv_scales_intersect4[:,  3] = k

    # quaternions : (N,4) float32 déjà prêts

    # --- AoS 64B: packer tout dans (N,16) float32 ---
    prims = cp.empty((N, 16), dtype=cp.float32)
    prims[:,  0: 4] = positions_relative4    # pos4
    prims[:,  4: 8] = inv_scales_intersect4            # inv_inter4
    prims[:,  8:12] = quaternions            # quat4
    prims[:, 12:16] = rgba4                  # rgba4
    prims = cp.ascontiguousarray(prims)

    params['prims'] = prims.data.ptr

    params['hit_prim_idx'] = hit_prim_idx.data.ptr

    ray_colors=cp.zeros((ray_size[0]*ray_size[1],3), dtype=cp.float32)
    params['ray_colors']=ray_colors.data.ptr

    params['handle'] = gas.handle

    if white_background:
        params['white_background'] = 1
    else:
        params['white_background'] = 0

    stream = cp.cuda.Stream()

    pipeline.launch(sbt, dimensions=ray_size, params=params, stream=stream)
    stream.synchronize()
    return ray_colors

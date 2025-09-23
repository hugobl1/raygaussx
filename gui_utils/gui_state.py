import collections
from utils import general_utils as utilities
from gui_utils.trackball import Trackball
from gui_utils.fps_controls import FPCameraControls
import optix as ox
import numpy as np
import cupy as cp
from optix_raycasting import optix_utils as u_ox
from gui_utils.state_handlers import init_camera_state, init_launch_params

SIGMA_THRESHOLD=u_ox.SIGMA_THRESHOLD
#------------------------------------------------------------------------------
# Local types
#------------------------------------------------------------------------------

def _camera_from_first_train_cam(barycenter, first_train_cam_info):
    # Récupère eye/look_at/up/fov à partir d'une cam d'entraînement + barycentre
    T = first_train_cam_info.T
    R = first_train_cam_info.R
    eye = -R @ T

    # Projette le barycentre sur l'axe de vision R[:,2] partant de "eye"
    # Hypothèse: R[:,2] normalisé
    z_axis = R[:, 2]
    len_look_at = np.dot(barycenter - eye, z_axis)
    look_at = eye + z_axis * len_look_at

    up = -R[:, 1]
    fov_y = first_train_cam_info.FoVy * 180.0 / np.pi
    return eye, look_at, up, fov_y

class PrecomputeCtx:
    def __init__(self,positions, color_features, sph_gauss_features, bandwidth_sharpness,
                 lobe_axis, num_sg, degree_sh, max_sg_display, max_sh_degree):
        self.positions=positions
        self.color_features = color_features
        self.sph_gauss_features = sph_gauss_features
        self.bandwidth_sharpness = bandwidth_sharpness
        self.lobe_axis = lobe_axis
        self.num_sg = num_sg
        self.degree_sh = degree_sh
        self.max_sg_display = max_sg_display
        self.max_sh_degree = max_sh_degree

    def __repr__(self):
        return (f"PrecomputeCtx(num_sg={self.num_sg}, degree_sh={self.degree_sh}, ...)")


class Params:
    _params = collections.OrderedDict([
            ('max_prim_slice', 'u4'),
            ('num_prim',         'u4'),
            ('bbox_min',         '3f4'),
            ('bbox_max',         '3f4'),
            ('dt_step',          'f4'),
            ('dynamic_sampling', 'u4'),
            ('prims',            'u8'),
            ('hit_prim_idx',   'u8'),
            ('frame_buffer',   'u8'),
            ('depth_buffer_ptr',   'u8'),
            ('width',          'u4'),
            ('height',         'u4'),
            ('eye',            '3f4'),
            ('u',              '3f4'),
            ('v',              '3f4'),
            ('w',              '3f4'),
            ('trav_handle',    'u8'),
            ('subframe_index', 'i4'),
        ])

    def __init__(self):
        self.handle = ox.LaunchParamsRecord(names=tuple(self._params.keys()),
                                            formats=tuple(self._params.values()))

    def __getattribute__(self, name):
        if name in Params._params.keys():
            item = self.__dict__['handle'][name]
            if isinstance(item, np.ndarray) and item.shape in ((0,), (1,)):
                return item.item()
            if isinstance(item, np.ndarray) and item.shape==(1,3):
                return item[0]
        else:
            return super().__getattribute__(name)

    def __setattr__(self, name, value):
        if name in Params._params.keys():
            self.handle[name] = value
        elif name in {'handle'}:
            super().__setattr__(name, value)
        else:
            raise AttributeError(name)

    def __str__(self):
        return '\n'.join(f'{k}:  {self.handle[k]}' for k in self._params)


class GeometryState:
    __slots__ = ['params', 'time', 'ctx', 'module', 'pipeline', 'pipeline_opts',
            'program_grps', 'sbt',
            'gas',
            'camera_mode','trackball', 'fp_controls',
            'camera_changed', 'mouse_button', 'resize_dirty', 'minimized','is_window_focused',
            'barycenter',
            'path_available_iterations','available_iterations','current_iteration',
            'slider_train_iteration','min_iteration','max_iteration',
            'idx_train_im','idx_test_im','update_train_im','update_test_im',
            'min_idx_test_im','max_idx_test_im','min_idx_train_im','max_idx_train_im',
            'gt_image', 'depth_buffer',
            'slider_used','print_error_image','print_gt_image','print_depth',
            'log_scale_error',
            'show_added_gaussians','update_added_gaussians', 'added_before_iter',
            'train_cam_infos','test_cam_infos',
            'train_images','test_images',
            'pointcloud','prims','pre_ctx','hit_prim_idx',
            'which_item','gui_mode']

    def __init__(self,):
        for slot in self.__slots__:
            setattr(self, slot, None)
        self.params = Params()

        self.camera_mode = 0 # 0 for trackball, 1 for FP camera
        self.trackball = Trackball()
        self.fp_controls = FPCameraControls()
        #FP controls camera is the same as trackball camera
        self.fp_controls.camera = self.trackball.camera

        self.camera_changed = True
        self.mouse_button = -1
        self.resize_dirty = False
        self.minimized = False
        self.is_window_focused = False

        self.current_iteration = -1
        self.slider_train_iteration = 0
        self.min_iteration = 0
        self.max_iteration = 0

        self.idx_train_im = 0
        self.idx_test_im = 0
        self.update_train_im = False
        self.update_test_im = False

        self.min_idx_test_im = 0
        self.max_idx_test_im = 0
        self.min_idx_train_im = 0
        self.max_idx_train_im = 0

        self.slider_used = False
        self.print_error_image = False
        self.print_gt_image = False
        self.print_depth = False

        self.log_scale_error = False

        self.show_added_gaussians = False
        self.update_added_gaussians = False
        self.added_before_iter = 0
        
        self.train_cam_infos = []
        self.test_cam_infos = []
        self.train_images = []
        self.test_images = []
        self.which_item = -1

    @property
    def camera(self):
        if self.camera_mode == 0:
            return self.trackball.camera
        else:
            return self.fp_controls.camera

    @property
    def launch_dimensions(self):
        return (int(self.params.width), int(self.params.height))
    
    def init_state(self, max_prim_slice, width, height, current_iter, pointcloud, data_init, log, optix_info,gui_mode):
        self.time = 0.0
        ### Load pointcloud attributes
        positions,scales,normalized_quaternions,densities,color_features,sph_gauss_features,bandwidth_sharpness,lobe_axis=pointcloud.get_data()
        cp_positions,cp_scales,cp_quaternions,cp_densities,cp_color_features,cp_sph_gauss_features,cp_bandwidth_sharpness,cp_lobe_axis=utilities.torch2cupy(
                            positions,
                            scales,
                            normalized_quaternions,
                            densities,
                            color_features.reshape(-1),
                            sph_gauss_features.reshape(-1),
                            bandwidth_sharpness.reshape(-1),
                            lobe_axis.reshape(-1))
        cp_bboxes, bb_min, bb_max = u_ox.ellipsoids_bbox_from_quat(cp_quaternions,cp_positions,cp_scales,cp_densities)

        self.barycenter=cp.asnumpy(cp_positions).mean(axis=0)
        self.pointcloud=pointcloud
        self.gui_mode=gui_mode

        ### Configure GUI and camera attributes in state
        if gui_mode:
            self.configure_camgui_state_output_mode(data_init["path_available_iterations"],data_init["available_iterations"],
                                                 current_iter,data_init["train_cam_infos"],data_init["test_cam_infos"],
                                                 data_init["train_images"], data_init["test_images"])
        else:
            self.configure_camgui_state_ply_mode()

        ### Initialize OptiX
        self.init_optix(log,cp_bboxes=cp_bboxes,exception_flags=optix_info["exception_flags"],
            debug_level=optix_info["debug_level"], opt_level=optix_info["opt_level"])

        ### Initialize self.pre_ctx
        degree_sh=int(np.sqrt(pointcloud.harmonic_number).item()-1)
        num_sg=pointcloud.sph_gauss_features.shape[2]
        self.pre_ctx=PrecomputeCtx(positions=cp_positions, color_features=cp_color_features, sph_gauss_features=cp_sph_gauss_features,
                                    bandwidth_sharpness=cp_bandwidth_sharpness,
                                    lobe_axis=cp_lobe_axis, num_sg=num_sg, degree_sh=degree_sh,
                                    max_sh_degree=degree_sh, max_sg_display=num_sg)
        ### Initialize params
        params=self.params
        params.max_prim_slice = max_prim_slice
        num_prim=len(cp_positions)
        params.num_prim=num_prim
        params.bbox_min = bb_min.get()
        params.bbox_max = bb_max.get()
        params.dt_step= data_init["dt_step"]
        params.dynamic_sampling=1 if data_init["dynamic_sampling"] else 0

        ###################################
        #### Initialize prims
        diff_ori_pos=cp.zeros((num_prim,4),dtype=cp.float32)
        diff_ori_pos[:,:3]=cp.array(self.camera.eye)-cp_positions
        rgba4=cp.empty((num_prim,4),dtype=cp.float32)
        rgba4[:,3]=cp_densities
        inv_scales = cp.reciprocal(cp_scales)                   # (N,3)
        k = cp.sqrt(2.0 * cp.log(cp_densities/ SIGMA_THRESHOLD))
        inv_scales_intersect = inv_scales / k[:,None]
        inv_scales_intersect4 = cp.empty((num_prim, 4), dtype=cp.float32)
        inv_scales_intersect4[:, :3] = inv_scales_intersect
        inv_scales_intersect4[:,  3] = k
        # --- AoS 64B: packer tout dans (N,16) float32 ---
        prims = cp.empty((num_prim, 16), dtype=cp.float32)
        prims[:, 0: 4] = diff_ori_pos  # diff_ori_pos4
        prims[:,  4: 8] = inv_scales_intersect4            # inv_inter4
        prims[:,  8:12] = cp_quaternions            # quat4
        prims[:, 12:16] = rgba4                  # rgba4
        self.prims = cp.ascontiguousarray(prims)
        params.prims = self.prims.data.ptr
        ###################################

        
        #### Initialize other buffers
        self.hit_prim_idx = cp.zeros((max_prim_slice*width*height),dtype=cp.uint32)
        params.hit_prim_idx = self.hit_prim_idx.data.ptr
        
        self.depth_buffer = cp.zeros((width*height),dtype=cp.float32)
        params.depth_buffer_ptr = self.depth_buffer.data.ptr

        params.width = width
        params.height = height

        self.gt_image = cp.zeros((width*height,4),dtype=cp.uint8)

    def configure_camgui_state_output_mode(self, 
                                        path_available_iterations,
                                        available_iterations,
                                        current_iter,
                                        train_cam_infos,
                                        test_cam_infos,
                                        train_images,
                                        test_images):
        # Méta itérations
        self.path_available_iterations = path_available_iterations
        self.available_iterations = available_iterations
        self.current_iteration = current_iter
        self.min_iteration = available_iterations[0]
        self.max_iteration = available_iterations[-1]
        self.slider_train_iteration = current_iter

        # Index images
        self.min_idx_test_im,  self.max_idx_test_im  = 0, len(test_cam_infos)  - 1
        self.min_idx_train_im, self.max_idx_train_im = 0, len(train_cam_infos) - 1
        self.added_before_iter = self.max_iteration

        # Données cam/images
        self.train_cam_infos, self.test_cam_infos = train_cam_infos, test_cam_infos
        self.train_images,    self.test_images    = train_images,    test_images

        # Caméra initiale: si dispo -> 1ère train cam ; sinon fallback barycentre
        eye, look_at, up, fov_y = _camera_from_first_train_cam(self.barycenter, train_cam_infos[0])
        self.params.eye = eye
        init_camera_state(self, eye=eye, look_at=look_at, up=up, fov_y=fov_y)


    def configure_camgui_state_ply_mode(self):
        ############################################################################################################
        #By default the camera is looking at the barycenter of the positions
        ############################################################################################################
        eye = self.barycenter + np.array([6, 6, 6], dtype=np.float32)
        self.params.eye = eye
        init_camera_state(self, eye=eye, look_at=self.barycenter)

    def init_optix(self, log, cp_bboxes,
                   exception_flags,
                   debug_level,
                   opt_level):
        """Construit le contexte OptiX, pipeline, SBT, GAS et met à jour trav_handle."""
        # 1) Context
        self.ctx = u_ox.create_context(log)

        # 2) Pipeline Options
        self.pipeline_opts = ox.PipelineCompileOptions(
            traversable_graph_flags=ox.TraversableGraphFlags.ALLOW_SINGLE_GAS,
            num_payload_values=2,
            num_attribute_values=0,
            exception_flags=exception_flags,
            pipeline_launch_params_variable_name="params",
        )

        # 3) Module + program groups + pipeline
        self.module       = u_ox.create_module(self.ctx, self.pipeline_opts,
                                               stage="gui",
                                               debug_level=debug_level,
                                               opt_level=opt_level)
        self.program_grps = u_ox.create_program_groups(self.ctx, self.module)
        self.pipeline     = u_ox.create_pipeline(self.ctx,
                                                 program_grps=self.program_grps,
                                                 pipeline_options=self.pipeline_opts,
                                                 debug_level=debug_level)

        # 4) Shader Binding Table
        self.sbt = u_ox.create_sbt(program_grps=self.program_grps)

        # 5) Acceleration Structure + handle
        self.gas = u_ox.create_acceleration_structure(self.ctx, cp_bboxes)
        self.params.trav_handle = self.gas.handle

        # 6) Launch params (écrit self.params.* au besoin)
        init_launch_params(self)

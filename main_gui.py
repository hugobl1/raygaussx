import sys, logging
import optix as ox

import argparse
import glfw, imgui

from gui_utils.gui import init_ui, display_stats
from gui_utils.gl_display import GLDisplay
from gui_utils.cuda_output_buffer import CudaOutputBuffer, CudaOutputBufferType, BufferImageFormat
from gui_utils.gui_state import GeometryState
from gui_utils.loaders import load_from_output, load_from_ply
from gui_utils.glfw_callback import mouse_button_callback, cursor_position_callback, window_size_callback, window_iconify_callback,key_callback,scroll_callback
from gui_utils.state_handlers import update_state, launch_subframe, display_subframe

from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=("Ray Gauss GUI: You can either display the output of a training iteration "
                     "or an rg_ply file. In the first case you must provide the output folder "
                     "and the iteration to display. In the second case you must provide the path "
                     "to the rg_ply file")
    )
    parser.add_argument("-output", type=Path, help="Path to output folder")
    parser.add_argument("-iter", type=int, help="Iteration to display")
    parser.add_argument("-ply_path", type=Path, help="Path to data folder")
    parser.add_argument("--width", type=int, default=1500)
    parser.add_argument("--height", type=int, default=1000)
    parser.add_argument("--max_prim_slice", type=int, default=512)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args(argv)

    # XOR validation: either (ply_path) or (output and iter), but not both
    has_ply = args.ply_path is not None
    has_out_and_iter = args.output is not None and args.iter is not None
    if not (has_ply ^ has_out_and_iter):
        parser.error("You must provide either the output folder and the iteration to display or the data path")

    # Normalize/validate paths similar to your code
    if args.ply_path is not None:
        args.ply_path = args.ply_path.resolve()
        if not args.ply_path.exists():
            parser.error("The path to the rg_ply file is incorrect")

    if args.output is not None:
        args.output = args.output.resolve()
        if not args.output.exists():
            parser.error("The path to the output folder is incorrect")

    return args

def main(argv=None):
    args=parse_args(argv)
    gui_mode= 0 if args.ply_path is not None else 1

    logging.basicConfig(stream=sys.stdout, level= logging.INFO)
    log = logging.getLogger()

    DEBUG = args.debug
    if DEBUG:
        exception_flags=(ox.ExceptionFlags.DEBUG | ox.ExceptionFlags.TRACE_DEPTH | ox.ExceptionFlags.STACK_OVERFLOW)
        debug_level = ox.CompileDebugLevel.FULL
        opt_level = ox.CompileOptimizationLevel.LEVEL_0
    else:
        exception_flags=ox.ExceptionFlags.NONE
        debug_level = ox.CompileDebugLevel.MINIMAL
        opt_level = ox.CompileOptimizationLevel.LEVEL_3
    optix_info = {
        "exception_flags": exception_flags,
        "debug_level": debug_level,
        "opt_level": opt_level
    }

    #------------------------------------------------------------------------------
    # Load scene data
    #------------------------------------------------------------------------------
    logging.info("GUI mode: %s (%s)", gui_mode, "PLY" if gui_mode == 0 else "OUTPUT")
    if gui_mode:
        pointcloud, data_init = load_from_output(args.output, preload_images=True, logger=logging.getLogger(__name__))
        available_iterations      = data_init["available_iterations"]
        path_available_iterations = data_init["path_available_iterations"]
        if args.iter not in available_iterations:
            # au choix: nearest ou erreur
            nearest = min(available_iterations, key=lambda x: abs(x - args.iter))
            logging.warning("Requested iter %s not found; using nearest %s.", args.iter, nearest)
            args.iter = nearest

        pointcloud.restore_model(
            iteration=args.iter,
            checkpoint_folder=path_available_iterations
        )
    else:
        pointcloud, data_init = load_from_ply(args.ply_path)

    #------------------------------------------------------------------------------
    # Main
    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
    # Initialize GeometryState
    #------------------------------------------------------------------------------
    state = GeometryState()
    state.init_state(args.max_prim_slice, args.width, args.height, 
                     args.iter, pointcloud, data_init, log, optix_info, gui_mode)

    buffer_format = BufferImageFormat.UCHAR4
    output_buffer_type = CudaOutputBufferType.enable_gl_interop()

    window, impl = init_ui("optixRadianceField", state.params.width, state.params.height)
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cursor_position_callback)
    glfw.set_window_size_callback(window, window_size_callback)
    glfw.set_window_iconify_callback(window, window_iconify_callback)
    glfw.set_key_callback(window, key_callback)
    glfw.set_scroll_callback(window, scroll_callback)
    glfw.set_window_user_pointer(window, state)

    output_buffer = CudaOutputBuffer(output_buffer_type, buffer_format,
            state.params.width, state.params.height)

    gl_display = GLDisplay(buffer_format)

    state_update_time = 0.0
    render_time = 0.0
    display_time = 0.0

    tstart = glfw.get_time()
    state.fp_controls.dt=tstart
    while not glfw.window_should_close(window):

        t0 = glfw.get_time()
        glfw.poll_events()
        impl.process_inputs()

        state.time = glfw.get_time() - tstart
        update_state(output_buffer, state)
        t1 = glfw.get_time()
        state_update_time += t1 - t0
        t0 = t1

        launch_subframe(output_buffer, state)
        t1 = glfw.get_time()
        render_time += t1 - t0
        t0 = t1

        display_subframe(output_buffer, gl_display, window)
        t1 = glfw.get_time()
        display_time += t1 - t0

        if display_stats(state,state_update_time, render_time, display_time):
            state_update_time = 0.0
            render_time = 0.0
            display_time = 0.0
        imgui.render()
        impl.render(imgui.get_draw_data())
        glfw.swap_buffers(window)
        state.params.subframe_index = state.params.subframe_index+ 1
    impl.shutdown()
    glfw.terminate()
    
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
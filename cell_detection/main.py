"""
main
===============

Runs each part of the cellfinder pipeline in turn.

N.B imports are within functions to prevent tensorflow being imported before
it's warnings are silenced
"""

import os
import logging
import tifffile
from datetime import datetime
import bg_space as bgs
from imlib.IO.cells import save_cells, get_cells
from imlib.cells.cells import MissingCellsError
from imlib.general.system import ensure_directory_exists

from cellfinder_core.main import suppress_tf_logging, tf_suppress_log_messages


def get_downsampled_space(atlas, downsampled_image_path):
    target_shape = tifffile.imread(downsampled_image_path).shape
    downsampled_space = bgs.AnatomicalSpace(
        atlas.metadata["orientation"],
        shape=target_shape,
        resolution=atlas.resolution,
    )
    return downsampled_space


def cells_exist(points_file):
    try:
        get_cells(points_file, cells_only=True)
        return True
    except MissingCellsError:
        return False


def main():
    os.environ["NUMEXPR_NUM_THREADS"] = "18"#自己加的
    yes_no = input('Do you wanna analyze the intensity of each signals? (0 means no, 1 means all, others mean yes) :\n')#自己加的
    try:
        int_yes_no = int(yes_no)#自己加
        if (int_yes_no != 0) and (int_yes_no != 1):#自己加
            rawdatapath = input('INSERT RAW DATA PATH :\n')#自己加
            regions = []#自己加
            while True:#自己加
                region = input('INSERT THE BRAIN REGION TO BE ANALYZED :\n')#自己做
                try:#自己加
                    region = int(region)#自己加
                    if region == 0:
                        break#自己加
                    else:#自己加
                        continue#自己加
                except:#自己加
                    pass#自己加
                print('INSERT THE NEXT REGION OR 0 TO LEAVE')#自己加
                regions.append(region)#自己加
        elif int_yes_no == 1:
            rawdatapath = input('INSERT RAW DATA PATH :\n')#自己加
            regions = []#自己加
        else:#自己加
            rawdatapath = ''#自己加
            regions = []#自己加
    except:#自己加
        pass#自己加
    
    suppress_tf_logging(tf_suppress_log_messages)
    from brainreg.main import main as register
    from cellfinder.tools import prep

    start_time = datetime.now()
    args, arg_groups, what_to_run, atlas = prep.prep_cellfinder_general()

    if what_to_run.register:
        # TODO: add register_part_brain option
        logging.info("Registering to atlas")
        args, additional_images_downsample = prep.prep_registration(args)
        register(
            args.atlas,
            args.orientation,
            args.target_brain_path,
            args.brainreg_paths,
            args.voxel_sizes,
            arg_groups["NiftyReg registration backend options"],
            sort_input_file=args.sort_input_file,
            n_free_cpus=args.n_free_cpus,
            additional_images_downsample=additional_images_downsample,
            backend=args.backend,
            debug=args.debug,
        )

    else:
        logging.info("Skipping registration")

    if len(args.signal_planes_paths) > 1:
        base_directory = args.output_dir

        for idx, signal_paths in enumerate(args.signal_planes_paths):
            channel = args.signal_ch_ids[idx]
            logging.info("Processing channel: " + str(channel))
            channel_directory = os.path.join(
                base_directory, "channel_" + str(channel)
            )
            if not os.path.exists(channel_directory):
                os.makedirs(channel_directory)

            # prep signal channel specific args
            args.signal_planes_paths[0] = signal_paths
            # TODO: don't overwrite args.output_dir - use Paths instead
            args.output_dir = channel_directory
            args.signal_channel = channel
            # Run for each channel
            run_all(args, what_to_run, atlas)

    else:
        args.signal_channel = args.signal_ch_ids[0]
        run_all(args, what_to_run, atlas, rawdatapath, regions, int_yes_no)#自己加
    logging.info(
        "Finished. Total time taken: {}".format(datetime.now() - start_time)
    )


def run_all(args, what_to_run, atlas, rawdatapath, regions, int_yes_no):

    from cellfinder_core.detect import detect
    from cellfinder_core.classify import classify
    from cellfinder_core.tools import prep
    from cellfinder_core.tools.IO import read_with_dask

    from cellfinder.analyse import analyse
    from cellfinder.figures import figures

    from cellfinder.tools.prep import (
        prep_candidate_detection,
        prep_channel_specific_general,
    )

    args, what_to_run = prep_channel_specific_general(args, what_to_run)

    if what_to_run.detect:
        logging.info("Detecting cell candidates")
        args = prep_candidate_detection(args)
        signal_array = read_with_dask(
            args.signal_planes_paths[args.signal_channel]
        )

        points = detect.main(
            signal_array,
            args.start_plane,
            args.end_plane,
            args.voxel_sizes,
            args.soma_diameter,
            args.max_cluster_size,
            args.ball_xy_size,
            args.ball_z_size,
            args.ball_overlap_fraction,
            args.soma_spread_factor,
            args.n_free_cpus,
            args.log_sigma_size,
            args.n_sds_above_mean_thresh,
        )
        ensure_directory_exists(args.paths.points_directory)

        save_cells(
            points,
            args.paths.detected_points,
            save_csv=args.save_csv,
            artifact_keep=args.artifact_keep,
        )

    else:
        logging.info("Skipping cell detection")

    if what_to_run.classify:
        model_weights = prep.prep_classification(
            args.trained_model,
            args.model_weights,
            args.install_path,
            args.model,
            args.n_free_cpus,
        )
        if what_to_run.classify:
            logging.info("Running cell classification")
            background_array = read_with_dask(args.background_planes_path[0])

            points = classify.main(
                points,
                signal_array,
                background_array,
                args.n_free_cpus,
                args.voxel_sizes,
                args.network_voxel_sizes,
                args.batch_size,
                args.cube_height,
                args.cube_width,
                args.cube_depth,
                args.trained_model,
                model_weights,
                args.network_depth,
            )
            save_cells(
                points,
                args.paths.classified_points,
                save_csv=args.save_csv,
            )

            what_to_run.cells_exist = cells_exist(args.paths.classified_points)

        else:
            logging.info("No cells were detected, skipping classification.")

    else:
        logging.info("Skipping cell classification")

    what_to_run.update_if_cells_required()

    if what_to_run.analyse or what_to_run.figures:
        downsampled_space = get_downsampled_space(
            atlas, args.brainreg_paths.boundaries_file_path
        )

    if what_to_run.analyse:
        logging.info("Analysing cell positions")
#自己加的
        def tree(classification_points):
            import xml.etree.ElementTree as ET
            tree = ET.parse(classification_points)
            root = tree.getroot()
            for T in root.iter('Marker_Type'):
                T[0].text = '2'
                tree.write(classification_points)
        tree(args.paths.classified_points)

        analyse.run(args, atlas, downsampled_space, rawdatapath, regions, int_yes_no)#自己加
    else:
        logging.info("Skipping cell position analysis")

    if what_to_run.figures:
        logging.info("Generating figures")
        figures.run(args, atlas, downsampled_space.shape)
    else:
        logging.info("Skipping figure generation")


if __name__ == "__main__":
    main()

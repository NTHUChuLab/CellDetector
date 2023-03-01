import imio
import tifffile
import logging

import numpy as np
import pandas as pd
import bg_space as bgs

from pathlib import Path
from imlib.IO.cells import get_cells
from imlib.pandas.misc import sanitise_df
from imlib.general.system import ensure_directory_exists

from tqdm import tqdm#自己加
from cv2 import circle,imread#自己做
import os#自己加
import multiprocessing as mp#自己加

from cellfinder.export.to_brainrender import export_points


class Point:
    def __init__(
        self, raw_coordinate, atlas_coordinate, structure, hemisphere
    ):
        self.raw_coordinate = raw_coordinate
        self.atlas_coordinate = atlas_coordinate
        self.structure = structure
        self.hemisphere = hemisphere


def calculate_densities(counts, volume_csv_path):
    """
    Use the region volume information from registration to calculate cell
    densities. Based on the atlas names, which must be exactly equal.
    :param counts: dataframe with cell counts
    :param volume_csv_path: path of the volumes of each brain region
    :return:
    """
    volumes = pd.read_csv(volume_csv_path, sep=",", header=0, quotechar='"')
    df = pd.merge(counts, volumes, on="structure_name", how="outer")
    df = df.fillna(0)
    df["left_cells_per_mm3"] = df.left_cell_count / df.left_volume_mm3
    df["right_cells_per_mm3"] = df.right_cell_count / df.right_volume_mm3
    return df


def combine_df_hemispheres(df):
    """
    Combine left and right hemisphere data onto a single row
    :param df:
    :return:
    """
    left = df[df["hemisphere"] == "left"]
    right = df[df["hemisphere"] == "right"]
    left = left.drop(["hemisphere"], axis=1)
    right = right.drop(["hemisphere"], axis=1)
    left.rename(columns={"cell_count": "left_cell_count"}, inplace=True)
    right.rename(columns={"cell_count": "right_cell_count"}, inplace=True)
    both = pd.merge(left, right, on="structure_name", how="outer")
    both = both.fillna(0)
    both["total_cells"] = both.left_cell_count + both.right_cell_count
    both = both.sort_values("total_cells", ascending=False)
    return both


def summarise_points(
    raw_points,
    transformed_points,
    atlas,
    volume_csv_path,
    all_points_filename,
    summary_filename,
    rawdatapath,#自己加
    regions,#自己加
    int_yes_no,#自己加
):
    points = []
    structures_with_points = set()
    for idx, point in enumerate(transformed_points):
        try:
            structure = atlas.structure_from_coords(point)
            structure = atlas.structures[structure]["name"]
            hemisphere = atlas.hemisphere_from_coords(point, as_string=True)
            points.append(Point(raw_points[idx], point, structure, hemisphere))
            structures_with_points.add(structure)
        except:
            continue
    logging.debug("Ensuring output directory exists")
    ensure_directory_exists(Path(all_points_filename).parent)
    if (int_yes_no != 0) and (int_yes_no != 1):#自己加
        t1 = mp.Process(target=create_partial_cell_csv, args=(points[:(len(points)//2)], all_points_filename[:-4]+'1.csv', rawdatapath, regions))  # 建立執行緒
        t2 = mp.Process(target=create_partial_cell_csv, args=(points[(len(points)//2):], all_points_filename[:-4]+'2.csv', rawdatapath, regions))  # 建立執行緒
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        #create_partial_cell_csv(points, all_points_filename, rawdatapath, regions)#自己加
    elif int_yes_no == 1:
        t1 = mp.Process(target=create_all_cell_csv, args=(points[:(len(points)//2)], all_points_filename[:-4]+'1.csv', rawdatapath))  # 建立執行緒
        t2 = mp.Process(target=create_all_cell_csv, args=(points[(len(points)//2):], all_points_filename[:-4]+'2.csv', rawdatapath))  # 建立執行緒
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        #create_all_cell_csv(points, all_points_filename)
    get_region_totals(
        points, structures_with_points, volume_csv_path, summary_filename
    )


def create_partial_cell_csv(points, all_points_filename, rawdatapath, regions):
#自己改的intensity
    rawimagelist = os.listdir(rawdatapath)#自己加
    z = -1#自己做
    df = pd.DataFrame(
        columns=(
            "coordinate_raw_axis_0",
            "coordinate_raw_axis_1",
            "coordinate_raw_axis_2",
            "coordinate_atlas_axis_0",
            "coordinate_atlas_axis_1",
            "coordinate_atlas_axis_2",
            "structure_name",
            "hemisphere",
            "intensity",
        )
    )
    for point in tqdm(points):
        if point.structure in regions:
            if z != point.raw_coordinate[0]:#自己加
                imagename = rawimagelist[point.raw_coordinate[0]]#自己做
                image = imread(rawdatapath+'/'+imagename,2)#自己做
            mask = np.zeros_like(image,dtype='uint16')#自己做
            mask = circle(mask, (point.raw_coordinate[2],point.raw_coordinate[1]), 5, (255,255,255), -1)#自己做
            mask[mask!=0] = 1#自己做
            A = np.multiply(image,mask)#自己做
            S = A.sum()#自己做
            A[A>0] = 1#自己做
            L = A.sum()#自己做
            intensity = S/L#自己做
            z = point.raw_coordinate[0]#自己做
            #intensity = image[point.raw_coordinate[1],point.raw_coordinate[2]]#自己做
            df = df.append(
                {
                    "coordinate_raw_axis_0": point.raw_coordinate[0],
                    "coordinate_raw_axis_1": point.raw_coordinate[1],
                    "coordinate_raw_axis_2": point.raw_coordinate[2],
                    "coordinate_atlas_axis_0": point.atlas_coordinate[0],
                    "coordinate_atlas_axis_1": point.atlas_coordinate[1],
                    "coordinate_atlas_axis_2": point.atlas_coordinate[2],
                    "structure_name": point.structure,
                    "hemisphere": point.hemisphere,
                    "intensity": intensity,
                },
                ignore_index=True,
            )
        
    df.to_csv(all_points_filename, index=False)


def create_all_cell_csv(points, all_points_filename, rawdatapath):
#自己改的intensity
    rawimagelist = os.listdir(rawdatapath)#自己加
    z = -1#自己做
    df = pd.DataFrame(
        columns=(
            "coordinate_raw_axis_0",
            "coordinate_raw_axis_1",
            "coordinate_raw_axis_2",
            "coordinate_atlas_axis_0",
            "coordinate_atlas_axis_1",
            "coordinate_atlas_axis_2",
            "structure_name",
            "hemisphere",
        )
    )
    for point in tqdm(points):
        if z != point.raw_coordinate[0]:#自己加
            imagename = rawimagelist[point.raw_coordinate[0]]#自己做
            image = imread(rawdatapath+'/'+imagename,2)#自己做
        mask = np.zeros_like(image,dtype='uint16')#自己做
        mask = circle(mask, (point.raw_coordinate[2],point.raw_coordinate[1]), 5, (255,255,255), -1)#自己做
        mask[mask!=0] = 1#自己做
        A = np.multiply(image,mask)#自己做
        S = A.sum()#自己做
        A[A>0] = 1#自己做
        L = A.sum()#自己做
        intensity = S/L#自己做
        z = point.raw_coordinate[0]#自己做
        df = df.append(
            {
                "coordinate_raw_axis_0": point.raw_coordinate[0],
                "coordinate_raw_axis_1": point.raw_coordinate[1],
                "coordinate_raw_axis_2": point.raw_coordinate[2],
                "coordinate_atlas_axis_0": point.atlas_coordinate[0],
                "coordinate_atlas_axis_1": point.atlas_coordinate[1],
                "coordinate_atlas_axis_2": point.atlas_coordinate[2],
                "structure_name": point.structure,
                "hemisphere": point.hemisphere,
                "intensity": intensity,
            },
            ignore_index=True,
        )
        
    df.to_csv(all_points_filename, index=False)


def get_region_totals(
    points, structures_with_points, volume_csv_path, output_filename
):
    structures_with_points = list(structures_with_points)

    point_numbers = pd.DataFrame(
        columns=("structure_name", "hemisphere", "cell_count")
    )
    for structure in structures_with_points:
        for hemisphere in ("left", "right"):
            n_points = len(
                [
                    point
                    for point in points
                    if point.structure == structure
                    and point.hemisphere == hemisphere
                ]
            )
            if n_points:
                point_numbers = point_numbers.append(
                    {
                        "structure_name": structure,
                        "hemisphere": hemisphere,
                        "cell_count": n_points,
                    },
                    ignore_index=True,
                )
    sorted_point_numbers = point_numbers.sort_values(
        by=["cell_count"], ascending=False
    )

    combined_hemispheres = combine_df_hemispheres(sorted_point_numbers)
    df = calculate_densities(combined_hemispheres, volume_csv_path)
    df = sanitise_df(df)

    df.to_csv(output_filename, index=False)


def transform_points_to_downsampled_space(
    points, target_space, source_space, output_filename=None
):
    points = source_space.map_points_to(target_space, points)
    if output_filename is not None:
        df = pd.DataFrame(points)
        df.to_hdf(output_filename, key="df", mode="w")
    return points


def transform_points_to_atlas_space(
    points,
    source_space,
    atlas,
    deformation_field_paths,
    downsampled_space,
    downsampled_points_path=None,
    atlas_points_path=None,
):
    downsampled_points = transform_points_to_downsampled_space(
        points,
        downsampled_space,
        source_space,
        output_filename=downsampled_points_path,
    )
    transformed_points = transform_points_downsampled_to_atlas_space(
        downsampled_points,
        atlas,
        deformation_field_paths,
        output_filename=atlas_points_path,
    )
    return transformed_points


def transform_points_downsampled_to_atlas_space(
    downsampled_points, atlas, deformation_field_paths, output_filename=None
):
    field_scales = [int(1000 / resolution) for resolution in atlas.resolution]
    points = [[], [], []]
    for axis, deformation_field_path in enumerate(deformation_field_paths):
        deformation_field = tifffile.imread(deformation_field_path)
        for point in downsampled_points:
            point = [int(round(p)) for p in point]
            points[axis].append(
                int(
                    round(
                        field_scales[axis]
                        * deformation_field[point[0], point[1], point[2]]
                    )
                )
            )

    transformed_points = np.array(points).T

    if output_filename is not None:
        df = pd.DataFrame(transformed_points)
        df.to_hdf(output_filename, key="df", mode="w")

    return transformed_points


def run(args, atlas, downsampled_space, rawdatapath, regions, int_yes_no):#自己加
    deformation_field_paths = [
        args.brainreg_paths.deformation_field_0,
        args.brainreg_paths.deformation_field_1,
        args.brainreg_paths.deformation_field_2,
    ]

    cells = get_cells(args.paths.classified_points, cells_only=True)
    cell_list = []
    for cell in cells:
        cell_list.append([cell.z, cell.y, cell.x])
    cells = np.array(cell_list)

    run_analysis(
        cells,
        args.signal_planes_paths[0],
        args.orientation,
        args.voxel_sizes,
        atlas,
        deformation_field_paths,
        downsampled_space,
        args.paths.downsampled_points,
        args.paths.atlas_points,
        args.paths.brainrender_points,
        args.brainreg_paths.volume_csv_path,
        args.paths.all_points_csv,
        args.paths.summary_csv,
        rawdatapath,#自己加
        regions,#自己加
        int_yes_no,#自己加
    )


def run_analysis(
    cells,
    signal_planes,
    orientation,
    voxel_sizes,
    atlas,
    deformation_field_paths,
    downsampled_space,
    downsampled_points_path,
    atlas_points_path,
    brainrender_points_path,
    volume_csv_path,
    all_points_csv_path,
    summary_csv_path,
    rawdatapath,#自己加
    regions,#自己加
    int_yes_no,#自己加
):

    source_shape = tuple(
        imio.get_size_image_from_file_paths(signal_planes).values()
    )
    source_shape = (source_shape[2], source_shape[1], source_shape[0])

    source_space = bgs.AnatomicalSpace(
        orientation,
        shape=source_shape,
        resolution=[float(i) for i in voxel_sizes],
    )

    transformed_cells = transform_points_to_atlas_space(
        cells,
        source_space,
        atlas,
        deformation_field_paths,
        downsampled_space,
        downsampled_points_path=downsampled_points_path,
        atlas_points_path=atlas_points_path,
    )

    logging.info("Exporting cells to brainrender")
    export_points(
        transformed_cells,
        atlas.resolution[0],
        brainrender_points_path,
    )

    logging.info("Summarising cell positions")
    summarise_points(
        cells,
        transformed_cells,
        atlas,
        volume_csv_path,
        all_points_csv_path,
        summary_csv_path,
        rawdatapath,#自己加
        regions,#自己加
        int_yes_no,#自己加
    )

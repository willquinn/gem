import argparse
import glob

import awkward as ak
import numpy as np
from build_geometry import build_geometry
from lgdo.lh5 import LH5Store, read_as
from lgdo.types import Array, Table, VectorOfVectors
from reboost import units
from reboost.hpge.psd import drift_time, drift_time_heuristic, r90
from reboost.hpge.utils import get_hpge_scalar_rz_field
from reboost.math.stats import gaussian_sample
from reboost.shape import cluster
from utils import load_config, log_config, setup_logger, weighted_avg


def check_bounds(dt_map, xloc, yloc, zloc, coord_offset, logger):
    r_min, r_max = dt_map.φ.grid[0][0], dt_map.φ.grid[0][-1]
    z_min, z_max = dt_map.φ.grid[1][0], dt_map.φ.grid[1][-1]
    z = zloc * 1000 - coord_offset[2].to(dt_map.z_units).m
    r = np.sqrt(
        (xloc * 1000 - coord_offset[0].to(dt_map.r_units).m) ** 2
        + (yloc * 1000 - coord_offset[1].to(dt_map.r_units).m) ** 2
    )
    out_of_bounds_mask = (r < r_min) | (r > r_max) | (z < z_min) | (z > z_max)
    logger.info(f"Total out-of-bounds r, z : {ak.sum(out_of_bounds_mask)}")
    return ak.any(out_of_bounds_mask == True, axis=-1)  # noqa: E712


def process_steps(logger, input_files, config):

    detector_name = config["detector"]["name"]

    logger.info("Loading step data...")
    step_data = read_as("stp/det001/", input_files, library="ak")

    logger.info(
        "Getting the drift time map... "
        + f"data/drift_time_maps/{detector_name}_drift_time_map.lh5"
    )
    dt_map_file = f"data/drift_time_maps/{detector_name}_drift_time_map.lh5"
    dt_map = get_hpge_scalar_rz_field(dt_map_file, "drift_time", "drift_time")

    logger.info("Building the simulation geometry (pre-determined)...")
    reg = build_geometry(logger, detector_name)
    coord_offset = units.pg4_to_pint(
        reg.physicalVolumeDict[detector_name].position
    )

    logger.info("Removing events out of bounds...")
    out_of_bounds_mask = check_bounds(
        dt_map,
        step_data.xloc,
        step_data.yloc,
        step_data.zloc,
        coord_offset,
        logger,
    )

    xloc = step_data.xloc[~out_of_bounds_mask]
    yloc = step_data.yloc[~out_of_bounds_mask]
    zloc = step_data.zloc[~out_of_bounds_mask]
    edep = step_data.edep[~out_of_bounds_mask]

    logger.info("Getting the drift times...")
    drift_times = drift_time(
        xloc * 1000,
        yloc * 1000,
        zloc * 1000,
        dt_map=dt_map,
        coord_offset=coord_offset,
    )

    logger.info("Getting the drift time heuristic...")
    dt_heur = drift_time_heuristic(drift_times, edep)

    logger.info("Calculating r90 and drift times...")
    r90s = r90(edep, xloc * 1000, yloc * 1000, zloc * 1000).view_as("ak")

    logger.info("Clustering...")
    clusters = cluster.cluster_by_step_length(
        step_data.evtid,
        step_data.xloc,
        step_data.yloc,
        step_data.zloc,
        threshold=config["clustering"]["threshold"],
    )

    cluster_x = cluster.apply_cluster(clusters, xloc).view_as("ak")
    cluster_y = cluster.apply_cluster(clusters, yloc).view_as("ak")
    cluster_z = cluster.apply_cluster(clusters, zloc).view_as("ak")
    cluster_e = cluster.apply_cluster(clusters, edep).view_as("ak")

    x_avg = weighted_avg(cluster_x, cluster_e)
    y_avg = weighted_avg(cluster_y, cluster_e)
    z_avg = weighted_avg(cluster_z, cluster_e)
    e_avg = ak.sum(cluster_e, axis=-1)

    logger.info("Clustering may introduce events out of bounds...")
    out_of_bounds_mask = check_bounds(
        dt_map, x_avg, y_avg, z_avg, coord_offset, logger
    )
    x_avg = x_avg[~out_of_bounds_mask]
    y_avg = y_avg[~out_of_bounds_mask]
    z_avg = z_avg[~out_of_bounds_mask]
    e_avg = e_avg[~out_of_bounds_mask]

    logger.info("Computing drift times on clusters...")
    cluster_dt = drift_time(
        x_avg * 1000,
        y_avg * 1000,
        z_avg * 1000,
        dt_map=dt_map,
        coord_offset=coord_offset,
    )

    logger.info("Getting the measured energy...")
    energies = ak.sum(edep, axis=-1)
    energies_smear = gaussian_sample(
        energies, sigma=config["clustering"]["eres_FWHM"] / 2.355
    )

    logger.info("Building output table...")
    clust_tbl = Table(
        {
            "edep": VectorOfVectors(e_avg),
            "xloc": VectorOfVectors(x_avg),
            "yloc": VectorOfVectors(y_avg),
            "zloc": VectorOfVectors(z_avg),
            "dt": VectorOfVectors(cluster_dt),
            "e_smeared": Array(energies_smear[~out_of_bounds_mask]),
            "r90": Array(r90s[~out_of_bounds_mask]),
            "dth": Array(dt_heur[~out_of_bounds_mask]),
            "n_clusters": Array(ak.count(e_avg, axis=-1)),
        }
    )

    output_file = f"data/proc_data/{detector_name}_clustered_output.lh5"
    logger.info(f"Writing cluster data to {output_file}...")
    sto = LH5Store()
    sto.write(clust_tbl, "cluster", output_file)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process and cluster detector step data."
    )
    parser.add_argument(
        "input",
        type=str,
        nargs="+",
        help="Input LH5 file(s) or glob pattern"
        + "(e.g., stp/*.lh5 or stp/file_0.lh5)",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="YAML config file containing run settings",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (default: INFO)",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    log_path = config.get("clustering", {}).get("log_file", None)
    logger = setup_logger(log_path, args.log_level)

    log_config(logger, config)

    input_files = []
    for path in args.input:
        input_files.extend(glob.glob(path))
    input_files = sorted(set(input_files))

    if not input_files:
        logger.error("No input files found. Exiting.")
        return

    logger.info(f"Found {len(input_files)} input files.")
    logger.debug(f"Input files: {input_files}")

    # Now call your processing function:
    process_steps(logger, input_files, config)
    logger.info("Finished.")


if __name__ == "__main__":
    main()

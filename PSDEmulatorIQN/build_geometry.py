import argparse

import pyg4ometry as pg4
import pygeomtools
from legendhpges import make_hpge
from legendmeta import LegendMetadata
from numpy import pi
from utils import load_config, log_config, setup_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build detector geometry and write GDML file."
    )
    parser.add_argument(
        "--config", required=True, help="Path to config YAML file"
    )
    parser.add_argument(
        "--log-level", default="INFO", help="Logging level (default: INFO)"
    )
    return parser.parse_args()


def build_geometry(logger, detector_name, write=False):
    logger.info(f"Building geometry for detector: {detector_name}")
    lmeta = LegendMetadata()
    detector_meta = lmeta.hardware.detectors.germanium.diodes[detector_name]
    registry = pg4.geant4.Registry()

    hpge_lv = make_hpge(
        detector_meta, name=f"{detector_name}_L", registry=registry
    )

    world_s = pg4.geant4.solid.Orb(
        "World_s", 20, registry=registry, lunit="cm"
    )
    world_lv = pg4.geant4.LogicalVolume(
        world_s, "G4_Galactic", "World", registry=registry
    )
    registry.setWorld(world_lv)

    lar_s = pg4.geant4.solid.Orb("LAr_s", 15, registry=registry, lunit="cm")
    lar_lv = pg4.geant4.LogicalVolume(
        lar_s, "G4_lAr", "LAr_l", registry=registry
    )
    pg4.geant4.PhysicalVolume(
        [0, 0, 0], [0, 0, 0], lar_lv, "LAr", world_lv, registry=registry
    )

    detector_pv = pg4.geant4.PhysicalVolume(
        [0, 0, 0],
        [-5, 0, -3, "cm"],
        hpge_lv,
        detector_name,
        lar_lv,
        registry=registry,
    )
    detector_pv.pygeom_active_detector = pygeomtools.RemageDetectorInfo(
        "germanium", 1, detector_meta
    )

    logger.info("Adding source...")
    source_s = pg4.geant4.solid.Tubs(
        "Source_s", 0, 1, 1, 0, 2 * pi, registry=registry
    )
    source_lv = pg4.geant4.LogicalVolume(
        source_s, "G4_BRAIN_ICRP", "Source_L", registry=registry
    )
    pg4.geant4.PhysicalVolume(
        [0, 0, 0],
        [0, 5, 0, "cm"],
        source_lv,
        "Source",
        lar_lv,
        registry=registry,
    )

    if write:
        pygeomtools.write_pygeom(
            registry, f"geometries/{detector_name}_geometry.gdml"
        )
        logger.info(f"data/geometries/{detector_name}_geometry.gdml written.")

    return registry


def main():
    args = parse_args()
    config = load_config(args.config)

    log_path = config.get("geometry", {}).get("log_file", None)
    logger = setup_logger(log_path, args.log_level)
    log_config(logger, config)

    build_geometry(logger, config["detector"]["name"], write=True)


if __name__ == "__main__":
    main()

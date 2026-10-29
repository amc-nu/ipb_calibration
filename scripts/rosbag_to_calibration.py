import click
import yaml
from ipb_calibration.rosbag_converter import convert


@click.command()
@click.argument("config", type=click.Path(exists=True))
def main(config):
    """Convert a ROS2 bag to the calibration input folder. CONFIG: yaml, see config/rosbag_to_calibration.yaml."""
    convert(yaml.safe_load(open(config)))


if __name__ == "__main__":
    main()

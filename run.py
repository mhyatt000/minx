import tyro
import os
from rich.pretty import pprint

from dataclasses import dataclass, asdict

from enum import Enum


class Cluster(Enum):
    polaris = "polaris"
    sophia = "sophia"


@dataclass
class Job:
    """Job schedule helper with reasonable defaults
    #PBS -l select=1:system=polaris
    #PBS -l place=scatter
    #PBS -l walltime=0:05:00
    #PBS -l filesystems=home:grand
    #PBS -j oe
    #PBS -q debug
    #PBS -A EVITA
    """

    n: int = 1  # num nodes
    system: Cluster = Cluster.polaris
    place: str = "scatter"
    time: str = "00:05:00"

    def create(self):

        assert self.time

        args = [
            f"#PBS -l select={self.n}:system={self.system.value}",
            f"#PBS -l place={self.place}",
            f"#PBS -l walltime={self.time}",
        ]
        args += [
            "#PBS -l filesystems=home:grand",
            "#PBS -j oe",
            "#PBS -q debug",
            "#PBS -A EVITA",
        ]

        pprint(args)
        return " ".join(args)


def main(cfg: Job):

    pprint(cfg)
    pprint(cfg.create())
    # os.system(cfg.create())


if __name__ == "__main__":
    main(tyro.cli(Job))

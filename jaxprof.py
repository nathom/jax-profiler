import logging
import json
import argparse
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import humanfriendly
import jax.profiler
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


class JaxProfiler:
    def __init__(self, profile_dir=None):
        if profile_dir is None:
            now = datetime.now()
            now_str = now.strftime("%Y-%m-%d_%H:%M:%S")
            profile_dir = Path(f"profiles/{now_str}")
            profile_dir.mkdir(parents=True, exist_ok=False)

        self.profile_dir = Path(profile_dir)
        assert self.profile_dir.exists(), f"{self.profile_dir} does not exist"
        self._stop_event = threading.Event()
        self.i = 0

    def capture(self, label: str | None = None, desc: str | None = None):
        filename = f"prof_{self.i:03d}"
        path = self.profile_dir / f"{filename}.prof"
        meta_path = self.profile_dir / f"{filename}_meta.json"
        meta = {"desc": desc, "label": label, "i": self.i}
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        logger.info("Saving profile at %s", path)
        jax.profiler.save_device_memory_profile(path)
        self.i += 1

    def capture_in_background(self, delta_s: float = 3.0):
        def profiling_loop():
            while not self._stop_event.is_set():
                self.capture()
                time.sleep(delta_s)

        self._stop_event.clear()
        self._thread = threading.Thread(target=profiling_loop)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join()


@dataclass(slots=True)
class ProfileEntry:
    function_name: str
    cum_percentage: float
    cum_mem: float
    flat_mem: float


@dataclass(slots=True)
class Profile:
    total_mem: float
    entries: list[ProfileEntry]


total_mem_re = re.compile(r"(\d+(?:\.\d+)?\w*) total")


def parse_profile_file(prof_file) -> Profile:
    result = subprocess.run(
        ["pprof", "--text", prof_file],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"pprof command failed: {result.stderr}")

    output = result.stdout.splitlines()
    if len(output) == 0:
        raise ValueError("No output from pprof command")

    total_mem = total_mem_re.search(output[1])

    if total_mem is None:
        raise RuntimeError(f"Could not find total usage: {result.stdout}")

    entries = []
    for entry_line in output[4:]:
        data = entry_line.split()
        [flat, flat_perc, sum_perc, cum_mem, cum_perc, func_name] = data
        entries.append(
            ProfileEntry(
                function_name=func_name,
                cum_percentage=float(cum_perc[:-1]),
                cum_mem=float(humanfriendly.parse_size(cum_mem)),
                flat_mem=float(humanfriendly.parse_size(flat)),
            )
        )

    return Profile(
        total_mem=humanfriendly.parse_size(total_mem.group(1)), entries=entries
    )


def parse_profile_dir(profile_dir: str | Path) -> list[Profile]:
    profile_dir = Path(profile_dir)
    prof_files = sorted([p for p in profile_dir.iterdir() if p.is_file()])
    profiles = [parse_profile_file(f) for f in tqdm(prof_files, unit="file")]
    return profiles


def plot_total_mem(
    profiles: list[Profile], delta_s: float = 3.0, save_path: str | None = None
):
    total_mem_data = [p.total_mem for p in profiles]
    xs = delta_s * np.arange(len(total_mem_data))
    plt.figure(figsize=(10, 6))
    plt.plot(xs, total_mem_data, marker="o", linestyle="-", color="b")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Memory Usage (bytes)")
    plt.title("Total Memory Usage Over Time")
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()


def plot_function_mem(profiles: list[Profile], delta_s: float = 3.0):
    func_to_mem = {
        entry.function_name: [] for prof in profiles for entry in prof.entries
    }
    seen = set()
    for prof in profiles:
        for entry in prof.entries:
            seen.add(entry.function_name)
            func_to_mem[entry.function_name].append(entry.flat_mem)
        for func, mem_sizes in func_to_mem.items():
            if func not in seen:
                func_to_mem[func].append(0)
        seen.clear()

    del func_to_mem["<unknown>"]
    xs = delta_s * np.arange(len(profiles))

    plt.figure(figsize=(10, 6))

    for func, mem_sizes in func_to_mem.items():
        plt.plot(xs, mem_sizes, marker="o", linestyle="-", label=func)

    plt.xlabel("Capture #")
    plt.ylabel("Memory Usage (bytes)")
    plt.title("Memory Usage Over Time by Function")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process profile files and save plot.")
    parser.add_argument(
        "--save", type=str, required=True, help="Path to save the PNG file for the plot"
    )
    parser.add_argument(
        "--profile-path",
        type=str,
        required=True,
        help="Path to the directory containing profile files",
    )
    parser.add_argument(
        "--delta", type=float, default=3.0, help="Interval for profiler captures"
    )

    args = parser.parse_args()

    profiles = parse_profile_dir(args.profile_path)
    plot_total_mem(profiles, delta_s=args.delta, save_path=args.save)

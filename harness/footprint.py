"""Physical footprint of the Ollama runner, sampled from outside the process."""

from __future__ import annotations

import ctypes
import subprocess
import threading
import time
from dataclasses import dataclass


class RUsageInfoV6(ctypes.Structure):
    """Matches struct rusage_info_v6 in <sys/resource.h>."""

    _fields_ = [
        ("ri_uuid", ctypes.c_uint8 * 16),
        ("ri_user_time", ctypes.c_uint64),
        ("ri_system_time", ctypes.c_uint64),
        ("ri_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_interrupt_wkups", ctypes.c_uint64),
        ("ri_pageins", ctypes.c_uint64),
        ("ri_wired_size", ctypes.c_uint64),
        ("ri_resident_size", ctypes.c_uint64),
        ("ri_phys_footprint", ctypes.c_uint64),
        ("ri_proc_start_abstime", ctypes.c_uint64),
        ("ri_proc_exit_abstime", ctypes.c_uint64),
        ("ri_child_user_time", ctypes.c_uint64),
        ("ri_child_system_time", ctypes.c_uint64),
        ("ri_child_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_child_interrupt_wkups", ctypes.c_uint64),
        ("ri_child_pageins", ctypes.c_uint64),
        ("ri_child_elapsed_abstime", ctypes.c_uint64),
        ("ri_diskio_bytesread", ctypes.c_uint64),
        ("ri_diskio_byteswritten", ctypes.c_uint64),
        ("ri_cpu_time_qos_default", ctypes.c_uint64),
        ("ri_cpu_time_qos_maintenance", ctypes.c_uint64),
        ("ri_cpu_time_qos_background", ctypes.c_uint64),
        ("ri_cpu_time_qos_utility", ctypes.c_uint64),
        ("ri_cpu_time_qos_legacy", ctypes.c_uint64),
        ("ri_cpu_time_qos_user_initiated", ctypes.c_uint64),
        ("ri_cpu_time_qos_user_interactive", ctypes.c_uint64),
        ("ri_billed_system_time", ctypes.c_uint64),
        ("ri_serviced_system_time", ctypes.c_uint64),
        ("ri_logical_writes", ctypes.c_uint64),
        ("ri_lifetime_max_phys_footprint", ctypes.c_uint64),
        ("ri_instructions", ctypes.c_uint64),
        ("ri_cycles", ctypes.c_uint64),
        ("ri_billed_energy", ctypes.c_uint64),
        ("ri_serviced_energy", ctypes.c_uint64),
        ("ri_interval_max_phys_footprint", ctypes.c_uint64),
        ("ri_runnable_time", ctypes.c_uint64),
        ("ri_flags", ctypes.c_uint64),
        ("ri_user_ptime", ctypes.c_uint64),
        ("ri_system_ptime", ctypes.c_uint64),
        ("ri_pinstructions", ctypes.c_uint64),
        ("ri_pcycles", ctypes.c_uint64),
        ("ri_energy_nj", ctypes.c_uint64),
        ("ri_penergy_nj", ctypes.c_uint64),
        ("ri_secure_time_in_system", ctypes.c_uint64),
        ("ri_secure_ptime_in_system", ctypes.c_uint64),
        ("ri_neural_footprint", ctypes.c_uint64),
        ("ri_lifetime_max_neural_footprint", ctypes.c_uint64),
        ("ri_interval_max_neural_footprint", ctypes.c_uint64),
        ("ri_conclave_footprint", ctypes.c_uint64),
        ("ri_page_wait_time_mach", ctypes.c_uint64),
        ("ri_page_cache_hits", ctypes.c_uint64),
        ("ri_reserved", ctypes.c_uint64 * 6),
    ]


_libproc = ctypes.CDLL("/usr/lib/libproc.dylib")
_libproc.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
_libproc.proc_pid_rusage.restype = ctypes.c_int
_RUSAGE_INFO_V6 = 6


def process_memory(pid: int) -> tuple[int, int]:
    """Return (phys_footprint, resident_size) in bytes for pid."""
    info = RUsageInfoV6()
    rc = _libproc.proc_pid_rusage(pid, _RUSAGE_INFO_V6, ctypes.byref(info))
    if rc != 0:
        raise OSError(f"proc_pid_rusage failed for pid {pid} (rc {rc})")
    return int(info.ri_phys_footprint), int(info.ri_resident_size)


def list_runners() -> list[tuple[int, int, str]]:
    """Return (pid, port, command) for each Ollama llama-server."""
    out = subprocess.check_output(["ps", "-ax", "-o", "pid=,command="], text=True)
    found = []
    for line in out.splitlines():
        stripped = line.strip()
        if "llama-server" not in stripped or "--model" not in stripped:
            continue
        pid_text, _, command = stripped.partition(" ")
        port = _flag_value(command, "--port")
        if port is None:
            continue
        found.append((int(pid_text), int(port), command))
    return found


def find_runner(weight_digest: str) -> tuple[int, int]:
    """Find the runner whose --model path contains this blob digest."""
    hex_digest = weight_digest.removeprefix("sha256:")
    matches = [item for item in list_runners() if hex_digest in item[2]]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one llama-server for blob {hex_digest[:12]}, found {len(matches)}"
        )
    pid, port, _command = matches[0]
    return pid, port


def find_server(port: int) -> tuple[int, int]:
    """Find the llama-server listening on this port."""
    matches = [item for item in list_runners() if item[1] == port]
    if len(matches) != 1:
        raise RuntimeError(f"expected one llama-server on port {port}, found {len(matches)}")
    pid, found_port, _command = matches[0]
    return pid, found_port


def _flag_value(command: str, flag: str) -> str | None:
    parts = command.split()
    for index, part in enumerate(parts):
        if part == flag and index + 1 < len(parts):
            return parts[index + 1]
    return None


@dataclass
class Sample:
    t_s: float
    phys_footprint: int
    resident_size: int = 0


class FootprintSampler:
    """Poll phys_footprint at a fixed rate. The clock is perf_counter minus t0."""

    def __init__(self, pid: int, interval_s: float = 0.1):
        self.pid = pid
        self.interval_s = interval_s
        self.samples: list[Sample] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="footprint", daemon=True)
        self._t0 = 0.0

    def start(self, t0: float) -> None:
        self._t0 = t0
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()

    def _run(self) -> None:
        next_t = time.perf_counter()
        while not self._stop.is_set():
            now = time.perf_counter()
            if now < next_t:
                self._stop.wait(next_t - now)
                if self._stop.is_set():
                    break
            taken = time.perf_counter()
            footprint, resident = process_memory(self.pid)
            self.samples.append(Sample(taken - self._t0, footprint, resident))
            next_t += self.interval_s
            if next_t < taken:
                next_t = taken + self.interval_s

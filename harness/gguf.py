"""Read the GGUF header far enough to name the file and its tensors."""

from __future__ import annotations

import struct
from pathlib import Path

_SCALAR_SIZE = {
    0: 1,
    1: 1,
    2: 2,
    3: 2,
    4: 4,
    5: 4,
    6: 4,
    7: 1,
    10: 8,
    11: 8,
    12: 8,
}

FILE_TYPE_NAMES = {
    2: "Q4_0",
    14: "Q4_K_S",
    15: "Q4_K_M",
    16: "Q5_K_S",
    17: "Q5_K_M",
}


def summarize(path: Path) -> dict:
    with path.open("rb") as handle:
        magic, version, n_tensors, n_kv = struct.unpack("<4sIQQ", handle.read(24))
        if magic != b"GGUF":
            raise ValueError(f"{path} is not a GGUF file")
        metadata: dict[str, object] = {}
        for _ in range(n_kv):
            key = _read_string(handle)
            value_type = struct.unpack("<I", handle.read(4))[0]
            metadata[key] = _read_metadata_value(handle, value_type)
        tensor_names = [_read_tensor_name(handle) for _ in range(n_tensors)]
    file_type = metadata.get("general.file_type")
    return {
        "version": version,
        "architecture": metadata.get("general.architecture"),
        "name": metadata.get("general.name"),
        "file_type": file_type,
        "file_type_name": FILE_TYPE_NAMES.get(file_type, str(file_type)),
        "tensor_count": n_tensors,
        "tensor_names": tensor_names,
        "vision_tensors": vision_tensors(tensor_names),
        "bytes": path.stat().st_size,
    }


def vision_tensors(names: list[str]) -> list[str]:
    """Tensor names that belong to a vision tower or projector, not the language model."""
    found = []
    for name in names:
        if name.startswith(("v.", "mm.", "vision.")) or "mmproj" in name:
            found.append(name)
    return found


def _read_string(handle) -> str:
    length = struct.unpack("<Q", handle.read(8))[0]
    return handle.read(length).decode("utf-8")


def _read_metadata_value(handle, value_type: int):
    if value_type == 8:
        return _read_string(handle)
    if value_type == 9:
        element_type, count = struct.unpack("<IQ", handle.read(12))
        _skip_array(handle, element_type, count)
        return {"array_type": element_type, "count": count}
    size = _SCALAR_SIZE[value_type]
    raw = handle.read(size)
    if value_type == 7:
        return raw != b"\x00"
    formats = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 10: "<Q", 11: "<q", 12: "<d"}
    return struct.unpack(formats[value_type], raw)[0]


def _skip_array(handle, element_type: int, count: int) -> None:
    if element_type == 8:
        for _ in range(count):
            length = struct.unpack("<Q", handle.read(8))[0]
            handle.seek(length, 1)
        return
    if element_type == 9:
        raise ValueError("nested GGUF arrays are not expected in this file")
    handle.seek(_SCALAR_SIZE[element_type] * count, 1)


def _read_tensor_name(handle) -> str:
    name = _read_string(handle)
    n_dim = struct.unpack("<I", handle.read(4))[0]
    handle.seek(8 * n_dim, 1)
    handle.seek(4, 1)  # dtype
    handle.seek(8, 1)  # offset
    return name

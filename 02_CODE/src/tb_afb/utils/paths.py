from pathlib import Path
from typing import Union

PathLike = Union[str, Path]


def resolve_within(base_dir: PathLike, candidate: PathLike, *, must_exist: bool = False) -> Path:
    """Resolve candidate and require it to remain inside base_dir.

    Absolute paths are accepted only when they resolve below base_dir. Using
    Path.relative_to avoids sibling-prefix bypasses such as /data2 matching
    /data.
    """
    base = Path(base_dir).resolve()
    path = Path(candidate)
    resolved = path.resolve() if path.is_absolute() else (base / path).resolve()

    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise PermissionError(f"Path escapes allowed directory: {resolved}") from exc

    if must_exist and not resolved.exists():
        raise FileNotFoundError(resolved)
    return resolved

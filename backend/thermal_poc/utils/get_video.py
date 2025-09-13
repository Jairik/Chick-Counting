''' Helpers for getting the video path for testing the thermal model '''

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Union

__all__ = ["get_local_source"]

# Main function, get the correct source path
def get_source():
    ''' Return the local, correct path for the video source for testing the thermal model '''
    win = "" 


''' Path conversion on Windows when using WSL (helpers) '''

_PathLike = Union[str, os.PathLike]
_WIN_PREFIX = re.compile(r'^[A-Za-z]:[\\/]|^[\\/]{2}[^\\/]+[\\/]')

def is_windows_path(path: _PathLike) -> bool:
    ''' Return True if the string looks like a Windows path '''
    s = str(path)
    if not s:
        return False
    return bool(_WIN_PREFIX.match(s)) or ('\\' in s and not s.startswith('/'))

def to_wsl_path(path: str, *, must_exist: bool = False) -> str:
    ''' Convert a Windows path to a WSL-compatible path '''
    """
    Convert a Windows path to a WSL/Linux path.
    - If `p` already looks POSIX, return it unchanged.
    - If `must_exist` is True, raise FileNotFoundError if the converted path isn't found.
    """
    s = str(path)

    # Already POSIX-ish (WSL/Linux)
    if s.startswith('/'):
        if must_exist and not Path(s).exists():
            raise FileNotFoundError(f"{s} does not exist")
        return s

    # If not obviously Windows, return as-is
    if not is_windows_path(s):
        if must_exist and not Path(s).exists():
            raise FileNotFoundError(f"{s} does not exist")
        return s

    # Prefer wslpath for correctness (handles edge cases & UNC cleanly)
    if shutil.which('wslpath'):
        try:
            out = subprocess.check_output(['wslpath', '-u', s], stderr=subprocess.STDOUT)
            w = out.decode().strip()
            if must_exist and not Path(w).exists():
                raise FileNotFoundError(f"{w} does not exist")
            return w
        except subprocess.CalledProcessError:
            pass  # fall back to manual conversion

    # --- Manual fallback ---
    # UNC path: \\server\share\dir\file -> /mnt/unc/server/share/dir/file
    if s.startswith(('\\\\', '//')):
        # remove leading slashes and normalize separators
        tail = s.lstrip('\\/').replace('\\', '/')
        parts = tail.split('/', 2)  # server, share, rest...
        if len(parts) == 1:
            w = f"/mnt/unc/{parts[0]}"
        elif len(parts) == 2:
            w = f"/mnt/unc/{parts[0]}/{parts[1]}"
        else:
            w = f"/mnt/unc/{parts[0]}/{parts[1]}/{parts[2]}"
    else:
        # Drive path: C:\dir\file -> /mnt/c/dir/file
        drive = s[0].lower()
        rest = s[2:].replace('\\', '/')
        w = f"/mnt/{drive}/{rest.lstrip('/')}"

    if must_exist and not Path(w).exists():
        raise FileNotFoundError(f"{w} does not exist")
    return w
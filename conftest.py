import pytest
from pathlib import Path
import uuid
import os


@pytest.fixture
def tmp_path(tmp_path_factory):
    """Override tmp_path to use a snap-accessible directory.

    This is needed for snap-confined ffmpeg which has limited access.
    Uses .pytest_tmp in the project directory instead of /tmp.
    """
    # Get project root (where conftest.py lives)
    proj_root = Path(__file__).parent
    base = proj_root / ".pytest_tmp"
    base.mkdir(exist_ok=True, parents=True)

    # Generate a unique temp directory
    tmp = base / str(uuid.uuid4())
    tmp.mkdir(exist_ok=True, parents=True)
    yield tmp

    # Cleanup
    import shutil
    if tmp.exists():
        shutil.rmtree(tmp)

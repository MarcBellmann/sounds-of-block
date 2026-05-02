import shutil
import uuid
from pathlib import Path

import pytest


@pytest.fixture
def tmp_path():
    """Override tmp_path to use a project-local directory.

    snap-confined ffmpeg cannot access /tmp, so we use .pytest_tmp/ in the
    project root instead. This fixture is scoped to tests/audio/ only.
    """
    proj_root = Path(__file__).parent.parent.parent
    base = proj_root / ".pytest_tmp"
    base.mkdir(exist_ok=True, parents=True)

    tmp = base / str(uuid.uuid4())
    tmp.mkdir(exist_ok=True, parents=True)
    yield tmp

    if tmp.exists():
        shutil.rmtree(tmp)

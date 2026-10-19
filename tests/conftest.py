import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from churnkit import synth  # noqa: E402


@pytest.fixture(scope="session")
def data():
    return synth.generate(n_customers=800, seed=11)

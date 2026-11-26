from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def spark():
    from engine.spark_session import get_spark

    session = get_spark(app_name="EcommerceWorkbenchTests", shuffle_partitions=4)
    # Run the suite in strict ANSI mode to prove ingestion never relies on lenient casts
    session.conf.set("spark.sql.ansi.enabled", "true")
    yield session
    session.stop()

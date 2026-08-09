"""
tests/conftest.py
==================
Shared pytest fixtures. A single session-scoped SparkSession is reused
across every test file that needs one, since creating a SparkSession (a
real JVM) per test would be needlessly slow.
"""

import pytest

from src.spark_session import get_spark_session


@pytest.fixture(scope="session")
def spark():
    session = get_spark_session()
    yield session
    session.stop()

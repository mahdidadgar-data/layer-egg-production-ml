"""Run the Streamlit app headless and interact with it."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture(scope="module")
def app():
    return AppTest.from_file(APP, default_timeout=240).run()


def test_app_runs_without_errors(app):
    assert not app.exception
    assert not app.error
    labels = [m.label for m in app.metric]
    assert "Eggs per hen to 72 wk" in labels


def test_changing_strain_updates_the_plan(app):
    before = next(m.value for m in app.metric if m.label == "Mean egg weight")
    next(s for s in app.selectbox if s.label == "Strain").set_value("Hy-Line W-80")
    app.run()
    assert not app.exception
    after = next(m.value for m in app.metric if m.label == "Mean egg weight")
    assert after != before

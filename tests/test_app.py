"""Smoke tests for the Streamlit UI using Streamlit's AppTest harness.

These verify the app renders and the JD-requirements flow works; the
file-upload path is covered by the CLI/e2e tests since AppTest cannot
drive file_uploader widgets.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).parent.parent / "app.py")
JD = (Path(__file__).parent.parent / "sample_data" / "job_description.txt").read_text()


@pytest.fixture()
def app():
    return AppTest.from_file(APP, default_timeout=30).run()


class TestAppSmoke:
    def test_initial_render_has_no_exception(self, app):
        assert not app.exception

    def test_jd_detection_populates_must_haves(self, app):
        app.text_area[0].set_value(JD).run()
        assert not app.exception
        must = app.multiselect[0].value
        assert "Python" in must
        assert "PyTorch" in must
        # nice-to-haves should NOT be preselected as must-haves
        assert "Terraform" not in must

    def test_adjusting_must_haves_reruns_cleanly(self, app):
        app.text_area[0].set_value(JD).run()
        app.multiselect[0].unselect("TensorFlow").run()
        assert not app.exception

    def test_sidebar_weight_sliders(self, app):
        assert len(app.sidebar.slider) == 4
        app.sidebar.slider[0].set_value(80).run()
        assert not app.exception

    def test_short_jd_shows_hint_not_crash(self, app):
        app.text_area[0].set_value("too short").run()
        assert not app.exception

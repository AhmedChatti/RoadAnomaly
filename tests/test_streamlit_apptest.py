"""Streamlit AppTest smoke coverage (no browser; no E2b weight mutation).

Enable with ROADANOMALY_RUN_APPTEST=1 (loads TensorFlow via app imports — slow).
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = os.environ.get("ROADANOMALY_RUN_APPTEST", "").strip() in {"1", "true", "yes"}


@unittest.skipUnless(RUN, "Set ROADANOMALY_RUN_APPTEST=1 to run Streamlit AppTest (slow TF import)")
class StreamlitAppTestSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / "src"))

    def test_app_boots_and_exposes_nav_buttons(self):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
        at.run()
        self.assertFalse(at.exception, msg=str(getattr(at, "exception", None)))
        labels = [b.label for b in at.button]
        for needed in ("Analyser", "Jouer", "Revue", "Aide"):
            self.assertIn(needed, labels)

    def test_switch_to_jouer_page(self):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
        at.run()
        jouer = next(b for b in at.button if b.label == "Jouer")
        jouer.click().run()
        self.assertFalse(at.exception, msg=str(getattr(at, "exception", None)))
        markdown = "\n".join(m.value for m in at.markdown)
        self.assertIn("Anomaly Hunter", markdown)


if __name__ == "__main__":
    unittest.main()

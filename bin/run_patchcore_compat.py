"""Compatibility launcher for run_patchcore.py.

The command-line interface uses the public flag name
``--anomaly_scorer_num_nn``. Some PatchCore classes historically used the
internal argument name ``anomaly_score_num_nn``. This launcher makes both names
compatible before executing the original runner.

Use this file when comparing older experiment scripts that still rely on the
public CLI spelling.
"""

from pathlib import Path
import runpy

import patchcore.patchcore
import patchcore.geometry_patchcore


def _patch_load_method(cls):
    original_load = cls.load

    def load_with_alias(self, *args, **kwargs):
        if "anomaly_scorer_num_nn" in kwargs:
            value = kwargs.pop("anomaly_scorer_num_nn")
            kwargs.setdefault("anomaly_score_num_nn", value)
        return original_load(self, *args, **kwargs)

    cls.load = load_with_alias


_patch_load_method(patchcore.patchcore.PatchCore)
_patch_load_method(patchcore.geometry_patchcore.PatchCore)

runpy.run_path(str(Path(__file__).with_name("run_patchcore.py")), run_name="__main__")

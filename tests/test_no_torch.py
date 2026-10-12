import subprocess
import sys


def test_analysis_and_sim_do_not_import_torch():
    code = (
        "import sys\n"
        "import moescope.analysis.report, moescope.analysis.compare, moescope.analysis.coactivation\n"
        "import moescope.sim.simulate, moescope.sim.recommend, moescope.trace.reader\n"
        "assert 'torch' not in sys.modules, 'torch was imported'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)

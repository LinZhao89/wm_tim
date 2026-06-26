# wm_tim

This repository contains experimental code for wafer map anomaly detection research.

The project is under active development. Detailed method descriptions, experiment settings, data organization, and paper-specific materials are intentionally not documented here before publication.

## Setup

```bash
pip install -r requirements.txt
export PYTHONPATH=src
```

On Windows PowerShell:

```powershell
$env:PYTHONPATH='src'
```

## Usage

The main experiment runner is:

```bash
PYTHONPATH=src python bin/run_patchcore.py --help
```

A compatibility launcher is also available for older experiment scripts:

```bash
PYTHONPATH=src python bin/run_patchcore_compat.py --help
```

## Notes

- This repository is for internal research use before paper submission.
- Dataset paths, experiment commands, and paper-specific methodology are not included in this public README.
- Please keep detailed experiment protocols in private notes or paper source files until the work is ready for release.

## License

See the repository license file.

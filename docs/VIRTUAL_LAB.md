# Running TerraSpectraIQ on the BITS Virtual Lab

The goal is one screenshot showing `train.py` running to completion on the Virtual Lab machine.
The script prints the host name, Python and scikit-learn versions, and the timestamp in its header,
then ends with the five-model comparison table.

## 1. Get the code

```bash
git clone https://github.com/venkatmanish/Terraspectraiq-satellite-ml.git
cd Terraspectraiq-satellite-ml
```

If the lab has no internet access to GitHub, download the repository ZIP on your own machine
(GitHub → Code → Download ZIP), upload it through the lab's file manager, then
`unzip Terraspectraiq-satellite-ml-main.zip && cd Terraspectraiq-satellite-ml-main`.

## 2. Install the dependencies

```bash
python3 --version
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins the versions used for the deployed models and needs Python ≥ 3.11.
If the lab has an older Python or the install fails, install compatible versions instead.
The training code works with scikit-learn ≥ 1.3:

```bash
pip install "scikit-learn>=1.3" pandas numpy scipy joblib
```

If `venv` is unavailable, use `pip install --user ...` with the same package list.
On a Jupyter-only lab, run each command in a notebook cell prefixed with `!`, and run
`%run train.py` for step 3.

## 3. Run the experiment

```bash
python train.py
```

It takes about 30 seconds on a laptop; a shared lab VM may be slower. With older library versions the
numbers can differ slightly. For example, with Python 3.11 and scikit-learn 1.3.2, Logistic Regression
gives accuracy 0.8365 instead of 0.8380 (the lbfgs solver changed between versions). The other four
models and the winner are unchanged. Use `python train.py --quick`
only if the full run is too slow, and say so in the submission. Optionally confirm the test suite:

```bash
pip install pytest && python -m pytest -q
```

## 4. Take the screenshot

Capture the whole lab window after the run finishes, so that these are visible:

- the Virtual Lab desktop or browser frame, which shows it is the BITS lab;
- the header lines (`Host: …`, Python / scikit-learn versions, timestamp);
- the `FINAL COMPARISON — official sat.tst hold-out` table and the winner lines.

If everything does not fit on one screen, maximise the terminal or shrink the font (Ctrl −).
The assignment asks for **one** screenshot.

Do not commit or push from the lab. Running `train.py` there rewrites `model/` with the lab's library
versions, and the deployed app needs the pinned versions.

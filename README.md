# Confidence Calibration for Neural Networks

This project provides tools and evaluations for calibrating the output
probabilities of neural network models. Good calibration ensures that
a model's confidence scores match the true likelihood of an event, which
is critical for trustworthy decision making in safety‑critical and
scientific applications.


## Features

- **Platt Scaling**: Logistic regression-based calibration.
- **Isotonic Regression**: Non‑parametric calibration method.
- **Temperature Scaling**: Neural‑network compatible scalar temperature
  for softmax correction.
- **Ensemble Calibration**: Weighted combination of the above methods.
- **Evaluation utilities**: Brier score, ECE, MCE and reliability
diagram data.
- **Device helpers**: Automatically select GPU/CPU devices and report
  information.


## 🛠️ Installation

```bash
# clone the repository
git clone https://github.com/kryptologyst/Confidence-Calibration-for-Neural-Networks.git
cd Confidence-Calibration-for-Neural-Networks

# create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# or `.venv\Scripts\activate` on Windows

# install the core dependencies
pip install -r requirements.txt
# or use pyproject.toml:
# pip install "[dev]" .
```

Development extras (testing, formatting) are available via the `dev`
optional dependency in `pyproject.toml`:

```bash
pip install .[dev]
```


## Quick Start

```python
from calibration import PlattScaling, IsotonicCalibration, TemperatureScaling
import numpy as np

# create synthetic binary logits + labels
rng = np.random.RandomState(0)
X = rng.randn(100, 2)  # two-class logits
y = rng.randint(0, 2, size=100)

# fit a calibrator
calibrator = PlattScaling(random_state=42).fit(X, y)
probs = calibrator.predict_proba(X)

print(probs.shape)  # (100, 2)
```

A demo Streamlit app is also provided; after installing dependencies you
can run it with:

```bash
calibration-demo
```


## ✅ Running the Tests

This project includes a small unit test suite to verify basic
functionality. To execute the tests:

```bash
pytest
```

Coverage reports are generated automatically; the tests exercise
calibration methods and the evaluation utilities.


## Project Structure

```
0758.py                    # illustrative script and project description
pyproject.toml             # build metadata
requirements.txt           # installation pins
src/                       # package sources
  calibration/             # calibration algorithms & evaluator
  utils/                   # device and seeding helpers
tests/                     # unit tests
```


## Contributing

1. Fork the repository.
2. Create a feature branch.
3. Add or update tests for any new behavior.
4. Run `black`/`ruff` and make sure `pytest` passes.
5. Submit a pull request with a clear description of your changes.


## License

This project is licensed under the MIT License. See the
[LICENSE](LICENSE) file for details.
# Confidence-Calibration-for-Neural-Networks

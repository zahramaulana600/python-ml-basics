# Python ML Basics 🧠

Fundamental machine learning algorithms and text/data preprocessing utilities implemented from scratch in standard Python.

Designed for self-study, learning mathematical intuitions, and reference.

## 📁 Repository Structure
- `ml_basics/`
  - `linear_regression.py`: Univariate and multivariate linear regression using gradient descent.
  - `preprocessing.py`: Min-Max normalization, standard scaling, and basic text cleaning.
- `tests/`
  - `test_basics.py`: Unit test suite covering mathematical consistency and edge cases.
- `.github/workflows/`
  - `ci.yml`: Automated testing pipeline on Python 3.10 and 3.11.

## 🚀 Quick Example

```python
from ml_basics.linear_regression import LinearRegression

# Train a simple model (y = 2x + 1)
X = [[1.0], [2.0], [3.0], [4.0]]
y = [3.0, 5.0, 7.0, 9.0]

model = LinearRegression(learning_rate=0.01, epochs=500)
model.fit(X, y)

pred = model.predict([[5.0]])
print(f"Prediction for x=5: {pred[0]:.2f}")  # ~11.0
```

## 🧪 Running Tests

```bash
python3 -m unittest discover -s tests
```

## 📄 License
MIT License

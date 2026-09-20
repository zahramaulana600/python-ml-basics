import unittest
from ml_basics.linear_regression import LinearRegression
from ml_basics.preprocessing import min_max_scale, standard_scale, clean_text

class TestMLBasics(unittest.TestCase):
    def test_preprocessing(self):
        vals = [10.0, 20.0, 30.0, 40.0, 50.0]
        scaled = min_max_scale(vals)
        self.assertAlmostEqual(scaled[0], 0.0)
        self.assertAlmostEqual(scaled[-1], 1.0)
        
        std_scaled = standard_scale(vals)
        self.assertAlmostEqual(sum(std_scaled) / len(std_scaled), 0.0, places=5)

    def test_clean_text(self):
        txt = "Hello, World! Learning AI & Machine Learning."
        tokens = clean_text(txt)
        self.assertEqual(tokens, ["hello", "world", "learning", "ai", "machine", "learning"])

    def test_linear_regression(self):
        # y = 2x + 1
        X = [[1.0], [2.0], [3.0], [4.0], [5.0]]
        y = [3.0, 5.0, 7.0, 9.0, 11.0]

        model = LinearRegression(learning_rate=0.02, epochs=800)
        model.fit(X, y)
        preds = model.predict([[6.0], [7.0]])
        self.assertAlmostEqual(preds[0], 13.0, delta=0.2)
        self.assertAlmostEqual(preds[1], 15.0, delta=0.2)

if __name__ == '__main__':
    unittest.main()

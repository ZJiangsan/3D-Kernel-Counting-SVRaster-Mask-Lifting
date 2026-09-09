import unittest

from kernel3d.metrics import count_metrics


class TestMetrics(unittest.TestCase):
    def test_reported_proposed_mae(self):
        gt = [716, 337, 729, 314, 539, 463, 659, 496, 724]
        pred = [716, 340, 732, 317, 533, 461, 654, 492, 725]
        self.assertAlmostEqual(count_metrics(gt, pred)["MAE"], 3.0)

    def test_reported_no_split_mae(self):
        gt = [716, 337, 729, 314, 539, 463, 659, 496, 724]
        pred = [684, 335, 721, 295, 520, 442, 643, 488, 717]
        self.assertAlmostEqual(count_metrics(gt, pred)["MAE"], 14.666666666666666)


if __name__ == "__main__":
    unittest.main()

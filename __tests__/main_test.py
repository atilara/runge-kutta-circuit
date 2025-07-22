import unittest

import numpy as np
from numpy.testing import assert_almost_equal

from core.equations import fonte_v, sistema, rk4


class TestCircuitoRLC(unittest.TestCase):

    def test_fonte_v_dc(self):
        self.assertEqual(fonte_v(0, "DC", V0=5), 5)
        self.assertEqual(fonte_v(10, "DC", V0=3.2), 3.2)

    def test_fonte_v_ac(self):
        t = 0.25
        A = 2
        f = 1  # 1 Hz
        esperado = A * np.sin(2 * np.pi * f * t)
        self.assertAlmostEqual(fonte_v(t, "AC", A=A, f=f), esperado)

    def test_fonte_v_default(self):
        self.assertEqual(fonte_v(0, "XYZ"), 0)

    def test_sistema_com_dc(self):
        R, L, C = 1.0, 1.0, 1.0
        t = 0
        x = [0, 0]
        resultado = sistema(t, x, R, L, C, "DC", V0=5)
        esperado = np.array([0, 5])
        assert_almost_equal(resultado, esperado, decimal=5)

    def test_rk4_com_dc(self):
        R, L, C = 1.0, 1.0, 1.0
        x0 = [0, 0]
        t0, tf, h = 0, 1, 0.1
        t_vals, x_vals = rk4(sistema, x0, t0, tf, h, R, L, C, "DC", V0=5)
        self.assertEqual(len(t_vals), len(x_vals))

        self.assertTrue(np.any(x_vals[-1] > 0))

    def test_rk4_stability_ac(self):
        R, L, C = 0.5, 1.0, 1.0
        x0 = [0.0, 0.0]
        t0, tf, h = 0.0, 2.0, 0.01
        t_vals, x_vals = rk4(sistema, x0, t0, tf, h, R, L, C, "AC", A=1.0, f_onda=60.0)
        self.assertEqual(len(t_vals), len(x_vals))

        self.assertLess(np.max(np.abs(x_vals)), 1000)

    def test_fonte_v_dc_random(self):
        for _ in range(10):
            V0 = np.random.uniform(-100, 100)
            t = np.random.uniform(0, 10)
            self.assertAlmostEqual(fonte_v(t, "DC", V0=V0), V0)

    def test_fonte_v_ac_random(self):
        for _ in range(10):
            A = np.random.uniform(0.1, 10)
            f = np.random.uniform(0.1, 100)
            t = np.random.uniform(0, 1)
            esperado = A * np.sin(2 * np.pi * f * t)
            resultado = fonte_v(t, "AC", A=A, f=f)
            self.assertAlmostEqual(resultado, esperado, places=5)

    def test_sistema_random(self):
        for _ in range(10):
            R = np.random.uniform(0.1, 10)
            L = np.random.uniform(0.1, 10)
            C = np.random.uniform(0.1, 10)
            x = np.random.uniform(-10, 10, size=2)
            t = np.random.uniform(0, 10)
            V0 = np.random.uniform(0, 100)
            resultado = sistema(t, x, R, L, C, "DC", V0=V0)
            self.assertEqual(resultado.shape, (2,))
            self.assertTrue(np.all(np.isfinite(resultado)))  # Sem NaN ou inf

    def test_rk4_random_parameters(self):
        for _ in range(5):
            R = np.random.uniform(0.1, 5)
            L = np.random.uniform(0.1, 5)
            C = np.random.uniform(0.1, 5)
            x0 = np.random.uniform(-1, 1, size=2)
            t0 = 0.0
            tf = np.random.uniform(0.5, 5)
            h = np.random.uniform(0.001, 0.05)
            V0 = np.random.uniform(0, 20)

            t_vals, x_vals = rk4(sistema, x0, t0, tf, h, R, L, C, "DC", V0=V0)

            self.assertEqual(len(t_vals), len(x_vals))
            self.assertFalse(np.any(np.isnan(x_vals)))
            self.assertFalse(np.any(np.isinf(x_vals)))
            self.assertEqual(x_vals.shape[1], 2)

if __name__ == '__main__':
    unittest.main()
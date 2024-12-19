"""
Created on 28 December 2020
@author: Peter Corke
"""

import numpy.testing as nt
import roboticstoolbox as rtb
import numpy as np
import matplotlib.pyplot as plt
import unittest

from roboticstoolbox.mobile.Bug2 import edgelist
from roboticstoolbox.mobile.landmarkmap import LandmarkMap
from roboticstoolbox.mobile.drivers import base
from roboticstoolbox.mobile.sensors import RangeBearingSensor
from roboticstoolbox.mobile.Vehicle import Unicycle, Bicycle

# from roboticstoolbox.mobile import Planner

# ======================================================================== #


class TestNavigation(unittest.TestCase):
    def test_edgelist(self):
        im = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 0, 0],
                [0, 0, 0, 1, 1, 1, 0, 0],
                [0, 0, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )

        seeds = [(2, 4), (3, 5), (5, 5), (3, 4), (1, 4), (2, 5), (3, 6), (1, 5)]
        for seed in seeds:
            # clockwise
            edge, _ = edgelist(im, seed)
            for e in edge:
                self.assertEqual(im[e[1], e[0]], im[seed[1], seed[0]])

            # counter clockwise
            edge, _ = edgelist(im, seed, -1)
            for e in edge:
                self.assertEqual(im[e[1], e[0]], im[seed[1], seed[0]])


# ======================================================================== #


class RangeBearingSensorTest(unittest.TestCase):
    def setUp(self):
        self.veh = rtb.Bicycle()
        self.map = rtb.LandmarkMap(20)
        self.rs = RangeBearingSensor(self.veh, self.map)

    def test_init(self):
        self.assertIsInstance(self.rs.map, rtb.LandmarkMap)
        self.assertIsInstance(self.rs.robot, rtb.Bicycle)

        self.assertIsInstance(str(self.rs), str)

    def test_reading(self):
        z, lm_id = self.rs.reading()
        self.assertIsInstance(z, np.ndarray)
        self.assertEqual(z.shape, (2,))

        # test missing samples
        rs = RangeBearingSensor(self.veh, self.map, every=2)

        # first return is (None, None)
        z, lm_id = rs.reading()
        self.assertEqual(z, None)

        z, lm_id = rs.reading()
        self.assertIsInstance(z, np.ndarray)
        self.assertEqual(z.shape, (2,))

        z, lm_id = rs.reading()
        self.assertEqual(z, None)

    def test_h(self):
        xv = np.r_[2, 3, 0.5]
        p = np.r_[3, 4]
        z = self.rs.h(xv, 10)
        self.assertIsInstance(z, np.ndarray)
        self.assertEqual(z.shape, (2,))
        self.assertAlmostEqual(z[0], np.linalg.norm(self.rs.map[10] - xv[:2]))
        theta = z[1] + xv[2]
        nt.assert_almost_equal(
            self.rs.map[10],
            xv[:2] + z[0] * np.r_[np.cos(theta), np.sin(theta)],
        )

        z = self.rs.h(xv, [3, 4])
        self.assertIsInstance(z, np.ndarray)
        self.assertEqual(z.shape, (2,))
        self.assertAlmostEqual(z[0], np.linalg.norm(p - xv[:2]))
        theta = z[1] + 0.5
        nt.assert_almost_equal(
            [3, 4], xv[:2] + z[0] * np.r_[np.cos(theta), np.sin(theta)]
        )

        # all landmarks
        z = self.rs.h(xv)
        self.assertIsInstance(z, np.ndarray)
        self.assertEqual(z.shape, (20, 2))
        for k in range(20):
            nt.assert_almost_equal(z[k, :], self.rs.h(xv, k))

        # if vehicle at landmark 10 range=bearing=0
        x = np.r_[self.map[10], 0]
        z = self.rs.h(x, 10)
        self.assertEqual(tuple(z), (0, 0))

        # vectorized forms
        xv = np.array([[2, 3, 0.5], [3, 4, 0], [4, 5, -0.5]])
        z = self.rs.h(xv, 10)
        self.assertIsInstance(z, np.ndarray)
        self.assertEqual(z.shape, (3, 2))
        for i in range(3):
            nt.assert_almost_equal(z[i, :], self.rs.h(xv[i, :], 10))

    def test_H_jacobians(self):
        xv = np.r_[1, 2, np.pi / 4]
        p = np.r_[5, 7]
        id = 10

        nt.assert_almost_equal(
            self.rs.Hx(xv, id), base.numjac(lambda x: self.rs.h(x, id), xv), decimal=4
        )

        nt.assert_almost_equal(
            self.rs.Hp(xv, p), base.numjac(lambda p: self.rs.h(xv, p), p), decimal=4
        )

        xv = [1, 2, np.pi / 4]
        p = [5, 7]
        id = 10

        nt.assert_almost_equal(
            self.rs.Hx(xv, id), base.numjac(lambda x: self.rs.h(x, id), xv), decimal=4
        )

        nt.assert_almost_equal(
            self.rs.Hp(xv, p), base.numjac(lambda p: self.rs.h(xv, p), p), decimal=4
        )

    def test_g(self):
        xv = np.r_[1, 2, np.pi / 4]
        p = np.r_[5, 7]

        z = self.rs.h(xv, p)
        nt.assert_almost_equal(p, self.rs.g(xv, z))

    def test_G_jacobians(self):
        xv = np.r_[1, 2, np.pi / 4]
        p = np.r_[5, 7]

        z = self.rs.h(xv, p)

        nt.assert_almost_equal(
            self.rs.Gx(xv, z), base.numjac(lambda x: self.rs.g(x, z), xv), decimal=4
        )

        nt.assert_almost_equal(
            self.rs.Gz(xv, z), base.numjac(lambda z: self.rs.g(xv, z), z), decimal=4
        )

    def test_plot(self):
        # map = LandmarkMap(20)
        # map.plot(block=False)
        pass


# ======================================================================== #


class LandMarkTest(unittest.TestCase):
    def test_init(self):
        map = LandmarkMap(20)

        self.assertEqual(len(map), 20)

        lm = map[0]
        self.assertIsInstance(lm, np.ndarray)
        self.assertTrue(lm.shape, (2,))

        self.assertIsInstance(str(lm), str)

    def test_range(self):
        map = LandmarkMap(1000, workspace=[-10, 10, 100, 200])

        self.assertTrue(map._map.shape, (2, 1000))

        for x, y in map:
            self.assertTrue(-10 <= x <= 10)
            self.assertTrue(100 <= y <= 200)

    def test_plot(self):
        plt.clf()
        map = LandmarkMap(20)
        map.plot(block=False)


# ======================================================================== #


class DriversTest(unittest.TestCase):
    def test_init(self):
        rp = rtb.RandomPath(10)

        self.assertIsInstance(str(rp), str)

        rp.init()

        veh = rtb.Bicycle()

        veh.control = rp

        self.assertIs(veh.control, rp)
        self.assertIs(rp.vehicle, veh)

        u = rp.demand()
        self.assertIsInstance(u, np.ndarray)
        self.assertTrue(u.shape, (2,))


class TestBicycle(unittest.TestCase):
    def test_jacobians(self):
        xv = np.r_[1, 2, np.pi / 4]
        odo = np.r_[0.1, 0.2]
        veh = Bicycle()

        nt.assert_almost_equal(
            veh.Fx(xv, odo), base.numjac(lambda x: veh.f(x, odo), xv), decimal=4
        )

        nt.assert_almost_equal(
            veh.Fv(xv, odo), base.numjac(lambda d: veh.f(xv, d), odo), decimal=4
        )


class TestUnicycle(unittest.TestCase):
    def test_str(self):
        """
        check the string representation of the unicycle
        """
        uni = Unicycle()
        self.assertEqual(
            str(uni),
            """Unicycle: x = [ 0, 0, 0 ]
  W=1, steer_max=inf, vel_max=inf, accel_max=inf""",
        )

        uni = Unicycle(steer_max=0.7)
        self.assertEqual(
            str(uni),
            """Unicycle: x = [ 0, 0, 0 ]
  W=1, steer_max=0.7, vel_max=inf, accel_max=inf""",
        )

    def test_deriv(self):
        """
        test the derivative function
        """
        uni = Unicycle()

        state = np.r_[0, 0, 0]
        input = [1, 0]  # no rotation
        nt.assert_almost_equal(uni.deriv(state, input), np.r_[1.0, 0, 0])

        input = [0, 1]  # only rotate
        nt.assert_almost_equal(uni.deriv(state, input), np.r_[0, 0, 1])

        input = [1, 1]  # turn and rotate
        nt.assert_almost_equal(uni.deriv(state, input), np.r_[1, 0, 1])


if __name__ == "__main__":
    unittest.main()

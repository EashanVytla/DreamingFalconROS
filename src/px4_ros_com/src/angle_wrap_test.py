from utils import unwrap_angle

import math
import unittest

class TestUnwrapAngle(unittest.TestCase):

    def test_no_wrap(self):
        # No wrapping needed.
        prev_angle = 0.5
        new_angle = 1.0
        result = unwrap_angle(new_angle, prev_angle)
        self.assertAlmostEqual(result, new_angle)
        
    def test_positive_wrap(self):
        # Test a positive wrap: if prev_angle = -3 and new_angle = 3,
        # then the difference is more than π, so the angle should be wrapped.
        prev_angle = -3.0
        new_angle = 3.0
        result = unwrap_angle(new_angle, prev_angle)
        expected = new_angle - 2 * math.pi  # 3.0 - 2π; with while loop, one subtraction is enough
        self.assertAlmostEqual(result, expected)
        
    def test_negative_wrap(self):
        # Test a negative wrap: if prev_angle = 3 and new_angle = -3,
        # then the difference is less than -π, so the angle should be wrapped.
        prev_angle = 3.0
        new_angle = -3.0
        result = unwrap_angle(new_angle, prev_angle)
        expected = new_angle + 2 * math.pi  # -3.0 + 2π
        self.assertAlmostEqual(result, expected)
        
    def test_boundary_positive(self):
        # When delta equals exactly π, no adjustment should be made.
        prev_angle = 0
        new_angle = math.pi
        result = unwrap_angle(new_angle, prev_angle)
        self.assertAlmostEqual(result, new_angle)
        
    def test_boundary_negative(self):
        # When delta equals exactly -π, no adjustment should be made.
        prev_angle = 0
        new_angle = -math.pi
        result = unwrap_angle(new_angle, prev_angle)
        self.assertAlmostEqual(result, new_angle)

    def test_multiple_rotations(self):
        # For example, with prev_angle = 0 and new_angle = 5π:
        #   First loop: 5π > π, so subtract 2π → new_angle becomes 3π.
        #   Second loop: 3π > π, so subtract 2π again → new_angle becomes π.
        # So, the final unwrapped angle is π.
        prev_angle = 0
        new_angle = 5 * math.pi
        result = unwrap_angle(new_angle, prev_angle)
        expected = math.pi  # As explained above.
        self.assertAlmostEqual(result, expected)

if __name__ == '__main__':
    unittest.main()

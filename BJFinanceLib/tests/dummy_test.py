from unittest import TestCase

import BJFinanceLib

class TestExample(TestCase):
    def test_is_string(self):
        s = "this is a string"
        self.assertTrue(isinstance(s, basestring))
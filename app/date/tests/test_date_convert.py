import unittest
import datetime

from .. import date_convert

class TestDateConversion(unittest.TestCase):

    def setUp(self):
        pass

    def test_time_to_pydate(self):
        ttpd = date_convert.time_to_pydate
        py_time = datetime.datetime(2003, 7, 30, 0, 27, 27, 272000)
        self.assertEqual(ttpd('2003-07-30T00:27:27.272'),py_time)

if __name__ == '__main__':
    unittest.main()

from llvm.common import find_library
from llvm.object import ObjectFile

import unittest

class TestObjectFile(unittest.TestCase):
    def test_create_from_file(self):
        source = find_library()
        of = ObjectFile(filename=source)

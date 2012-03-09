from llvm.common import find_library
from llvm.core import MemoryBuffer

import unittest

class TestCore(unittest.TestCase):
    def test_memory_buffer_create_from_file(self):
        source = find_library()
        self.assertIsNotNone(source)

        mb = MemoryBuffer(filename=source)

# Check the lit adaption to run under unittest.
#
# RUN: python %s %{inputs}/unittest-adaptor 2> %t.err
# RUN: FileCheck < %t.err %s
#
# CHECK: unittest-adaptor :: test-one.txt ... ok
# CHECK: unittest-adaptor :: test-two.txt ... FAIL

import unittest
import sys

import lit
import lit.discovery

input_path = sys.argv[1]
unittest_suite = lit.discovery.load_test_suite([input_path])
runner = unittest.TextTestRunner(verbosity=2)
runner.run(unittest_suite)

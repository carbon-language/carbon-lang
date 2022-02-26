#!/usr/bin/env python

import os
import sys

if len(sys.argv) != 2:
    raise ValueError("unexpected number of args")

if sys.argv[1] == "--gtest_list_tests":
    print(f"""\
Running main() from {os.getcwd()}/gtest_main.cc
FirstTest.
  subTestA
  subTestB
  subTestC
  subTestD
ParameterizedTest/0.
  subTest
ParameterizedTest/1.
  subTest""")
    sys.exit(0)
elif not sys.argv[1].startswith("--gtest_filter="):
    raise ValueError("unexpected argument: %r" % (sys.argv[1]))

test_name = sys.argv[1].split('=',1)[1]
print('Running main() from gtest_main.cc')
if test_name == 'FirstTest.subTestA':
    print('I am subTest A, I PASS')
    print('[  PASSED  ] 1 test.')
    sys.exit(0)
elif test_name == 'FirstTest.subTestB':
    print('I am subTest B, I FAIL')
    print('And I have two lines of output')
    sys.exit(1)
elif test_name == 'FirstTest.subTestC':
    print('I am subTest C, I am SKIPPED')
    print('[  PASSED  ] 0 tests.')
    print('[  SKIPPED ] 1 test, listed below:')
    print('[  SKIPPED ] FirstTest.subTestC')
    sys.exit(0)
elif test_name == 'FirstTest.subTestD':
    print('I am subTest D, I am UNRESOLVED')
    sys.exit(0)
elif test_name in ('ParameterizedTest/0.subTest',
                   'ParameterizedTest/1.subTest'):
    print('I am a parameterized test, I also PASS')
    print('[  PASSED  ] 1 test.')
    sys.exit(0)
else:
    raise SystemExit("error: invalid test name: %r" % (test_name,))

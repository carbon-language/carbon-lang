"""
'blacklist' is a Python dictionary, it stores the mapping of a string describing
either a testclass or a testcase, i.e, testclass.testmethod, to the reason (a
string) it is blacklisted.

Following is an example which states that test class IntegerTypesExprTestCase
should be skipped because 'This test class crashed' and the test case
FoundationTestCase.test_data_type_and_expr_with_dsym should be skipped because
it is 'Temporarily disabled'.

blacklist = {'IntegerTypesExprTestCase': 'This test class crashed',
             'FoundationTestCase.test_data_type_and_expr_with_dsym': 'Temporarily disabled'
             }
"""

blacklist = {'BasicExprCommandsTestCase.test_evaluate_expression_python': 'Crashed while running the entire test suite with CC=clang'
             # To reproduce the crash: CC=clang ./dotest.py -v -w 2> ~/Developer/Log/lldbtest.log
             # The clang version used is clang-126.
             # Two radars filed for the crashes: rdar://problem/8769826 and rdar://problem/8773329.
             # To skip this test case: CC=clang ./dotest.py -b blacklist.py -v -w 2> ~/Developer/Log/lldbtest.log
             }

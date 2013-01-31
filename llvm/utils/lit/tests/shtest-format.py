# Check the various features of the ShTest format.
#
# RUN: not %{lit} -j 1 -v %{inputs}/shtest-format > %t.out
# RUN: FileCheck < %t.out %s
#
# END.

# CHECK: -- Testing:

# CHECK: FAIL: shtest-format :: external_shell/fail.txt
# CHECK: *** TEST 'shtest-format :: external_shell/fail.txt' FAILED ***
# CHECK: Command Output (stderr):
# CHECK: cat: does-not-exist: No such file or directory
# CHECK: --

# CHECK: PASS: shtest-format :: external_shell/pass.txt

# CHECK: FAIL: shtest-format :: fail.txt

# CHECK: UNRESOLVED: shtest-format :: no-test-line.txt
# CHECK: PASS: shtest-format :: pass.txt
# CHECK: UNSUPPORTED: shtest-format :: requires-missing.txt
# CHECK: PASS: shtest-format :: requires-present.txt
# CHECK: UNSUPPORTED: shtest-format :: unsupported_dir/some-test.txt
# CHECK: XFAIL: shtest-format :: xfail-feature.txt
# CHECK: XFAIL: shtest-format :: xfail-target.txt
# CHECK: XFAIL: shtest-format :: xfail.txt
# CHECK: XPASS: shtest-format :: xpass.txt
# CHECK: Testing Time

# CHECK: Unexpected Passing Tests (1)
# CHECK: shtest-format :: xpass.txt

# CHECK: Failing Tests (2)
# CHECK: shtest-format :: external_shell/fail.txt
# CHECK: shtest-format :: fail.txt

# CHECK: Expected Passes    : 3
# CHECK: Expected Failures  : 3
# CHECK: Unsupported Tests  : 2
# CHECK: Unresolved Tests   : 1
# CHECK: Unexpected Passes  : 1
# CHECK: Unexpected Failures: 2

# Check the basic discovery process, including a sub-suite.
#
# RUN: %{lit} %{inputs}/discovery \
# RUN:   -j 1 --debug --no-execute --show-suites -v > %t.out 2> %t.err
# RUN: FileCheck --check-prefix=CHECK-BASIC-OUT < %t.out %s
# RUN: FileCheck --check-prefix=CHECK-BASIC-ERR < %t.err %s
#
# CHECK-BASIC-ERR: loading suite config '{{.*}}/tests/Inputs/discovery/lit.cfg'
# CHECK-BASIC-ERR: loading local config '{{.*}}/tests/Inputs/discovery/subdir/lit.local.cfg'
# CHECK-BASIC-ERR: loading suite config '{{.*}}/tests/Inputs/discovery/subsuite/lit.cfg'
#
# CHECK-BASIC-OUT: -- Test Suites --
# CHECK-BASIC-OUT:   sub-suite - 2 tests
# CHECK-BASIC-OUT:     Source Root:
# CHECK-BASIC-OUT:     Exec Root  :
# CHECK-BASIC-OUT:   top-level-suite - 3 tests
# CHECK-BASIC-OUT:     Source Root:
# CHECK-BASIC-OUT:     Exec Root  :
#
# CHECK-BASIC-OUT: -- Testing: 5 tests, 1 threads --
# CHECK-BASIC-OUT: PASS: sub-suite :: test-one
# CHECK-BASIC-OUT: PASS: sub-suite :: test-two
# CHECK-BASIC-OUT: PASS: top-level-suite :: subdir/test-three
# CHECK-BASIC-OUT: PASS: top-level-suite :: test-one
# CHECK-BASIC-OUT: PASS: top-level-suite :: test-two


# Check discovery when exact test names are given.
#
# RUN: %{lit} \
# RUN:     %{inputs}/discovery/subdir/test-three.py \
# RUN:     %{inputs}/discovery/subsuite/test-one.txt \
# RUN:   -j 1 --no-execute --show-suites -v > %t.out
# RUN: FileCheck --check-prefix=CHECK-EXACT-TEST < %t.out %s
#
# CHECK-EXACT-TEST: -- Testing: 2 tests, 1 threads --
# CHECK-EXACT-TEST: PASS: sub-suite :: test-one
# CHECK-EXACT-TEST: PASS: top-level-suite :: subdir/test-three



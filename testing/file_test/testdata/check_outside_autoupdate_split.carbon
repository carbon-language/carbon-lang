// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //testing/file_test:file_test_base_test --test_arg=--file_tests=testing/file_test/testdata/check_outside_autoupdate_split.carbon
// TIP: To dump output, run:
// TIP:   bazel run //testing/file_test:file_test_base_test -- --dump_output --file_tests=testing/file_test/testdata/check_outside_autoupdate_split.carbon

// When the file uses an `AUTOUPDATE-SPLIT`, `CHECK` lines elsewhere are
// content rather than expectations, so a split can hold a test file of its
// own. Such a split still can't contain a literal `// ---` line, so the `/`
// characters are written as `[[@0x2f]]`.
//
// The echoed line numbers below show that both lines are part of `a.carbon`:
// if the `CHECK` had been consumed it would be an unmatched expectation, and
// if the `// ---` had split, there would be a second file in the arguments.

// --- a.carbon
// CHECK:STDOUT: not an expectation
this line follows a CHECK line
[[@0x2f]][[@0x2f]] --- not a split
this line follows a split marker

// --- AUTOUPDATE-SPLIT

// CHECK:STDOUT: 2 args: `default_args`, `a.carbon`
// CHECK:STDOUT: a.carbon:2: this line follows a CHECK line
// CHECK:STDOUT: a.carbon:4: this line follows a split marker

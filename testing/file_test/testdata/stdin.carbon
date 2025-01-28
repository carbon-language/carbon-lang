// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //testing/file_test:file_test_base_test --test_arg=--file_tests=testing/file_test/testdata/stdin.carbon
// TIP: To dump output, run:
// TIP:   bazel run //testing/file_test:file_test_base_test -- --dump_output --file_tests=testing/file_test/testdata/stdin.carbon

// --- a.carbon

// --- STDIN

S
t
d
i
n

// --- AUTOUPDATE-SPLIT

// CHECK:STDERR: --- STDIN:
// CHECK:STDERR:
// CHECK:STDERR: S
// CHECK:STDERR: t
// CHECK:STDERR: d
// CHECK:STDERR: i
// CHECK:STDERR: n
// CHECK:STDERR:
// CHECK:STDOUT: 2 args: `default_args`, `a.carbon`

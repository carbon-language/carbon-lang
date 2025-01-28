// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lex/testdata/fail_multifile.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lex/testdata/fail_multifile.carbon

// --- fail_a.carbon
// CHECK:STDOUT: - filename: fail_a.carbon
// CHECK:STDOUT:   tokens:
// CHECK:STDERR: fail_a.carbon:[[@LINE+4]]:3: error: empty digit sequence in numeric literal [EmptyDigitSequence]
// CHECK:STDERR: 1.a
// CHECK:STDERR:   ^
// CHECK:STDERR:
1.a
// CHECK:STDOUT:   - { index: 1, kind:     "Error", line: {{ *}}[[@LINE-1]], column: 1, indent: 1, spelling: "1.a", has_leading_space: true }

// --- fail_b.carbon
// CHECK:STDOUT: - filename: fail_b.carbon
// CHECK:STDOUT:   tokens:
// CHECK:STDERR: fail_b.carbon:[[@LINE+4]]:3: error: empty digit sequence in numeric literal [EmptyDigitSequence]
// CHECK:STDERR: 2.b
// CHECK:STDERR:   ^
// CHECK:STDERR:
2.b
// CHECK:STDOUT:   - { index: 1, kind:     "Error", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "2.b", has_leading_space: true }

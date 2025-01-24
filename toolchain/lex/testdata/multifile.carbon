// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lex/testdata/multifile.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lex/testdata/multifile.carbon

// --- a.carbon
// CHECK:STDOUT: - filename: a.carbon
// CHECK:STDOUT:   tokens:
a;
// CHECK:STDOUT:   - { index: 1, kind: "Identifier", line: {{ *}}[[@LINE-1]], column: 1, indent: 1, spelling: "a", identifier: 0, has_leading_space: true }
// CHECK:STDOUT:   - { index: 2, kind:       "Semi", line: {{ *}}[[@LINE-2]], column: 2, indent: 1, spelling: ";" }

// --- b.carbon
// CHECK:STDOUT: - filename: b.carbon
// CHECK:STDOUT:   tokens:
b;
// CHECK:STDOUT:   - { index: 1, kind: "Identifier", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "b", identifier: 0, has_leading_space: true }
// CHECK:STDOUT:   - { index: 2, kind:       "Semi", line: {{ *}}[[@LINE-2]], column:   2, indent: 1, spelling: ";" }
a;
// CHECK:STDOUT:   - { index: 3, kind: "Identifier", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "a", identifier: 1, has_leading_space: true }
// CHECK:STDOUT:   - { index: 4, kind:       "Semi", line: {{ *}}[[@LINE-2]], column:   2, indent: 1, spelling: ";" }

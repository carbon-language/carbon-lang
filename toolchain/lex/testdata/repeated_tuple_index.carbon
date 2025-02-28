// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lex/testdata/repeated_tuple_index.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lex/testdata/repeated_tuple_index.carbon
// CHECK:STDOUT: - filename: repeated_tuple_index.carbon
// CHECK:STDOUT:   tokens:

x.0.1.2.3.foo.4.5
// CHECK:STDOUT:   - { index:  1, kind:    "Identifier", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "x", identifier: 0, has_leading_space: true }
// CHECK:STDOUT:   - { index:  2, kind:        "Period", line: {{ *}}[[@LINE-2]], column:   2, indent: 1, spelling: "." }
// CHECK:STDOUT:   - { index:  3, kind:    "IntLiteral", line: {{ *}}[[@LINE-3]], column:   3, indent: 1, spelling: "0", value: "0" }
// CHECK:STDOUT:   - { index:  4, kind:        "Period", line: {{ *}}[[@LINE-4]], column:   4, indent: 1, spelling: "." }
// CHECK:STDOUT:   - { index:  5, kind:    "IntLiteral", line: {{ *}}[[@LINE-5]], column:   5, indent: 1, spelling: "1", value: "1" }
// CHECK:STDOUT:   - { index:  6, kind:        "Period", line: {{ *}}[[@LINE-6]], column:   6, indent: 1, spelling: "." }
// CHECK:STDOUT:   - { index:  7, kind:    "IntLiteral", line: {{ *}}[[@LINE-7]], column:   7, indent: 1, spelling: "2", value: "2" }
// CHECK:STDOUT:   - { index:  8, kind:        "Period", line: {{ *}}[[@LINE-8]], column:   8, indent: 1, spelling: "." }
// CHECK:STDOUT:   - { index:  9, kind:    "IntLiteral", line: {{ *}}[[@LINE-9]], column:   9, indent: 1, spelling: "3", value: "3" }
// CHECK:STDOUT:   - { index: 10, kind:        "Period", line: {{ *}}[[@LINE-10]], column:  10, indent: 1, spelling: "." }
// CHECK:STDOUT:   - { index: 11, kind:    "Identifier", line: {{ *}}[[@LINE-11]], column:  11, indent: 1, spelling: "foo", identifier: 1 }
// CHECK:STDOUT:   - { index: 12, kind:        "Period", line: {{ *}}[[@LINE-12]], column:  14, indent: 1, spelling: "." }
// CHECK:STDOUT:   - { index: 13, kind:    "IntLiteral", line: {{ *}}[[@LINE-13]], column:  15, indent: 1, spelling: "4", value: "4" }
// CHECK:STDOUT:   - { index: 14, kind:        "Period", line: {{ *}}[[@LINE-14]], column:  16, indent: 1, spelling: "." }
// CHECK:STDOUT:   - { index: 15, kind:    "IntLiteral", line: {{ *}}[[@LINE-15]], column:  17, indent: 1, spelling: "5", value: "5" }

1.2.3.4
// CHECK:STDOUT:   - { index: 16, kind:   "RealLiteral", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "1.2", value: "12*10^-1", has_leading_space: true }
// CHECK:STDOUT:   - { index: 17, kind:        "Period", line: {{ *}}[[@LINE-2]], column:   4, indent: 1, spelling: "." }
// CHECK:STDOUT:   - { index: 18, kind:    "IntLiteral", line: {{ *}}[[@LINE-3]], column:   5, indent: 1, spelling: "3", value: "3" }
// CHECK:STDOUT:   - { index: 19, kind:        "Period", line: {{ *}}[[@LINE-4]], column:   6, indent: 1, spelling: "." }
// CHECK:STDOUT:   - { index: 20, kind:    "IntLiteral", line: {{ *}}[[@LINE-5]], column:   7, indent: 1, spelling: "4", value: "4" }

tuple.1.2
// CHECK:STDOUT:   - { index: 21, kind:    "Identifier", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "tuple", identifier: 2, has_leading_space: true }
// CHECK:STDOUT:   - { index: 22, kind:        "Period", line: {{ *}}[[@LINE-2]], column:   6, indent: 1, spelling: "." }
// CHECK:STDOUT:   - { index: 23, kind:    "IntLiteral", line: {{ *}}[[@LINE-3]], column:   7, indent: 1, spelling: "1", value: "1" }
// CHECK:STDOUT:   - { index: 24, kind:        "Period", line: {{ *}}[[@LINE-4]], column:   8, indent: 1, spelling: "." }
// CHECK:STDOUT:   - { index: 25, kind:    "IntLiteral", line: {{ *}}[[@LINE-5]], column:   9, indent: 1, spelling: "2", value: "2" }

tuple . 1.2
// CHECK:STDOUT:   - { index: 26, kind:    "Identifier", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "tuple", identifier: 2, has_leading_space: true }
// CHECK:STDOUT:   - { index: 27, kind:        "Period", line: {{ *}}[[@LINE-2]], column:   7, indent: 1, spelling: ".", has_leading_space: true }
// CHECK:STDOUT:   - { index: 28, kind:   "RealLiteral", line: {{ *}}[[@LINE-3]], column:   9, indent: 1, spelling: "1.2", value: "12*10^-1", has_leading_space: true }

"hello"1.2
// CHECK:STDOUT:   - { index: 29, kind: "StringLiteral", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "\"hello\"", value: "hello", has_leading_space: true }
// CHECK:STDOUT:   - { index: 30, kind:   "RealLiteral", line: {{ *}}[[@LINE-2]], column:   8, indent: 1, spelling: "1.2", value: "12*10^-1" }

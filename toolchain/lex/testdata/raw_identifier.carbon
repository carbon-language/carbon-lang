// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lex/testdata/raw_identifier.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lex/testdata/raw_identifier.carbon
// CHECK:STDOUT: - filename: raw_identifier.carbon
// CHECK:STDOUT:   tokens:

// A non-keyword identifier.
r#foo
// CHECK:STDOUT:   - { index: 1, kind:          "Identifier", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "foo", identifier: 1, has_leading_space: true }

// The same non-keyword identifier, for comparison.
foo
// CHECK:STDOUT:   - { index: 2, kind:          "Identifier", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "foo", identifier: 1, has_leading_space: true }

// A keyword as a raw identifier.
r#self
// CHECK:STDOUT:   - { index: 3, kind:          "Identifier", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "self", identifier: 2, has_leading_space: true }

// The same keyword, for comparison.
self
// CHECK:STDOUT:   - { index: 4, kind: "SelfValueIdentifier", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "self", has_leading_space: true }

// A type literal as a raw identifier.
r#i32
// CHECK:STDOUT:   - { index: 5, kind:          "Identifier", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "i32", identifier: 3, has_leading_space: true }

// The same type literal, for comparison.
i32
// CHECK:STDOUT:   - { index: 6, kind:      "IntTypeLiteral", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "i32", has_leading_space: true }

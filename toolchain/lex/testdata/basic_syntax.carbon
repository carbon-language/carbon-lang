// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Dump all tokens so that we include the file start and end.
// EXTRA-ARGS: --no-omit-file-boundary-tokens
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lex/testdata/basic_syntax.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lex/testdata/basic_syntax.carbon
// CHECK:STDOUT: - filename: basic_syntax.carbon
// CHECK:STDOUT:   tokens:
// CHECK:STDOUT:   - { index:  0, kind:         "FileStart", line: {{ *\d+}}, column:   1, indent: 1, spelling: "" }

fn run(String program) {
// CHECK:STDOUT:   - { index:  1, kind:                "Fn", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "fn", has_leading_space: true }
// CHECK:STDOUT:   - { index:  2, kind:        "Identifier", line: {{ *}}[[@LINE-2]], column:   4, indent: 1, spelling: "run", identifier: 0, has_leading_space: true }
// CHECK:STDOUT:   - { index:  3, kind:         "OpenParen", line: {{ *}}[[@LINE-3]], column:   7, indent: 1, spelling: "(", closing_token: 6 }
// CHECK:STDOUT:   - { index:  4, kind: "StringTypeLiteral", line: {{ *}}[[@LINE-4]], column:   8, indent: 1, spelling: "String" }
// CHECK:STDOUT:   - { index:  5, kind:        "Identifier", line: {{ *}}[[@LINE-5]], column:  15, indent: 1, spelling: "program", identifier: 1, has_leading_space: true }
// CHECK:STDOUT:   - { index:  6, kind:        "CloseParen", line: {{ *}}[[@LINE-6]], column:  22, indent: 1, spelling: ")", opening_token: 3 }
// CHECK:STDOUT:   - { index:  7, kind:    "OpenCurlyBrace", line: {{ *}}[[@LINE-7]], column:  24, indent: 1, spelling: "{", closing_token: 11, has_leading_space: true }
  return True;
  // CHECK:STDOUT:   - { index:  8, kind:            "Return", line: {{ *}}[[@LINE-1]], column:   3, indent: 3, spelling: "return", has_leading_space: true }
  // CHECK:STDOUT:   - { index:  9, kind:        "Identifier", line: {{ *}}[[@LINE-2]], column:  10, indent: 3, spelling: "True", identifier: 2, has_leading_space: true }
  // CHECK:STDOUT:   - { index: 10, kind:              "Semi", line: {{ *}}[[@LINE-3]], column:  14, indent: 3, spelling: ";" }
}
// CHECK:STDOUT:   - { index: 11, kind:   "CloseCurlyBrace", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "}", opening_token: 7, has_leading_space: true }

// CHECK:STDOUT:   - { index: 12, kind:           "FileEnd", line: {{ *}}[[@LINE+0]], column: {{ *\d+}}, indent: 1, spelling: "", has_leading_space: true }

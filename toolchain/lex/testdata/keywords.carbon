// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lex/testdata/keywords.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lex/testdata/keywords.carbon

// --- indented.carbon
// CHECK:STDOUT: - filename: indented.carbon
// CHECK:STDOUT:   tokens:

   fn
   // CHECK:STDOUT:   - { index: 1, kind:        "Fn", line: {{ *}}[[@LINE-1]], column: 4, indent: 4, spelling: "fn", has_leading_space: true }

// --- chain.carbon
// CHECK:STDOUT: - filename: chain.carbon
// CHECK:STDOUT:   tokens:

and or not if else for return var break continue _
// CHECK:STDOUT:   - { index:  1, kind:        "And", line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: "and", has_leading_space: true }
// CHECK:STDOUT:   - { index:  2, kind:         "Or", line: {{ *}}[[@LINE-2]], column:  5, indent: 1, spelling: "or", has_leading_space: true }
// CHECK:STDOUT:   - { index:  3, kind:        "Not", line: {{ *}}[[@LINE-3]], column:  8, indent: 1, spelling: "not", has_leading_space: true }
// CHECK:STDOUT:   - { index:  4, kind:         "If", line: {{ *}}[[@LINE-4]], column: 12, indent: 1, spelling: "if", has_leading_space: true }
// CHECK:STDOUT:   - { index:  5, kind:       "Else", line: {{ *}}[[@LINE-5]], column: 15, indent: 1, spelling: "else", has_leading_space: true }
// CHECK:STDOUT:   - { index:  6, kind:        "For", line: {{ *}}[[@LINE-6]], column: 20, indent: 1, spelling: "for", has_leading_space: true }
// CHECK:STDOUT:   - { index:  7, kind:     "Return", line: {{ *}}[[@LINE-7]], column: 24, indent: 1, spelling: "return", has_leading_space: true }
// CHECK:STDOUT:   - { index:  8, kind:        "Var", line: {{ *}}[[@LINE-8]], column: 31, indent: 1, spelling: "var", has_leading_space: true }
// CHECK:STDOUT:   - { index:  9, kind:      "Break", line: {{ *}}[[@LINE-9]], column: 35, indent: 1, spelling: "break", has_leading_space: true }
// CHECK:STDOUT:   - { index: 10, kind:   "Continue", line: {{ *}}[[@LINE-10]], column: 41, indent: 1, spelling: "continue", has_leading_space: true }
// CHECK:STDOUT:   - { index: 11, kind: "Underscore", line: {{ *}}[[@LINE-11]], column: 50, indent: 1, spelling: "_", has_leading_space: true }

// --- notakeyword.carbon
// CHECK:STDOUT: - filename: notakeyword.carbon
// CHECK:STDOUT:   tokens:

notakeyword
// CHECK:STDOUT:   - { index: 1, kind: "Identifier", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "notakeyword", identifier: 0, has_leading_space: true }

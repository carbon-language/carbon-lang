// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/format/testdata/basics/strings.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/format/testdata/basics/strings.carbon

// --- test.carbon

// A multi-line string literal's body is emitted verbatim, and the statement
// around it is formatted normally.
var text: String = '''
  line one
    indented line
  ''';

var single: String = "one  line, spacing kept";

// --- AUTOUPDATE-SPLIT

// CHECK:STDOUT: // A multi-line string literal's body is emitted verbatim, and the statement
// CHECK:STDOUT: // around it is formatted normally.
// CHECK:STDOUT: var text: String = '''
// CHECK:STDOUT:   line one
// CHECK:STDOUT:     indented line
// CHECK:STDOUT:   ''';
// CHECK:STDOUT:
// CHECK:STDOUT: var single: String = "one  line, spacing kept";

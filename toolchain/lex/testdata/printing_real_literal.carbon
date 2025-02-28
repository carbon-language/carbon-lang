// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lex/testdata/printing_real_literal.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lex/testdata/printing_real_literal.carbon
// CHECK:STDOUT: - filename: printing_real_literal.carbon
// CHECK:STDOUT:   tokens:

2.5
// CHECK:STDOUT:   - { index: 1, kind: "RealLiteral", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "2.5", value: "25*10^-1", has_leading_space: true }

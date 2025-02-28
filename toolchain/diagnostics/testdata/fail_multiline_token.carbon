// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: --include-diagnostic-kind compile --phase=parse %s
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/diagnostics/testdata/fail_multiline_token.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/diagnostics/testdata/fail_multiline_token.carbon

// CHECK:STDERR: fail_multiline_token.carbon:[[@LINE+4]]:1: error: unrecognized declaration introducer [UnrecognizedDecl]
// CHECK:STDERR: '''
// CHECK:STDERR: ^~~
// CHECK:STDERR:
'''
hello
'''

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This is validating that the multi-line token's hint prints correctly.
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/basics/multiline_token.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/basics/multiline_token.carbon

// --- fail_multiline_token.carbon

// CHECK:STDERR: fail_multiline_token.carbon:[[@LINE+4]]:1: error: unrecognized declaration introducer [UnrecognizedDecl]
// CHECK:STDERR: '''
// CHECK:STDERR: ^~~
// CHECK:STDERR:
'''
hello
'''

// CHECK:STDOUT: - filename: fail_multiline_token.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'InvalidParseStart', text: ''''
// CHECK:STDOUT: hello
// CHECK:STDOUT: '''', has_error: yes},
// CHECK:STDOUT:     {kind: 'InvalidParseSubtree', text: ''''
// CHECK:STDOUT: hello
// CHECK:STDOUT: '''', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]

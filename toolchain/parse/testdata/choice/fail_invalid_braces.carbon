// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/choice/fail_invalid_braces.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/choice/fail_invalid_braces.carbon

// CHECK:STDERR: fail_invalid_braces.carbon:[[@LINE+4]]:22: error: choice definition expected [ExpectedChoiceDefinition]
// CHECK:STDERR: choice InvalidBraces |
// CHECK:STDERR:                      ^
// CHECK:STDERR:
choice InvalidBraces |
  Some
// CHECK:STDERR: fail_invalid_braces.carbon:[[@LINE+4]]:1: error: unrecognized declaration introducer [UnrecognizedDecl]
// CHECK:STDERR: |
// CHECK:STDERR: ^
// CHECK:STDERR:
|

// CHECK:STDOUT: - filename: fail_invalid_braces.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'ChoiceIntroducer', text: 'choice'},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: 'InvalidBraces'},
// CHECK:STDOUT:       {kind: 'ChoiceDefinitionStart', text: '|', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:     {kind: 'ChoiceDefinition', text: '|', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:       {kind: 'InvalidParseStart', text: '|', has_error: yes},
// CHECK:STDOUT:     {kind: 'InvalidParseSubtree', text: '|', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]

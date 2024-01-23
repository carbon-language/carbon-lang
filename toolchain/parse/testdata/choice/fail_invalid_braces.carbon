// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_invalid_braces.carbon:[[@LINE+3]]:22: ERROR: Choice definition expected.
// CHECK:STDERR: choice InvalidBraces |
// CHECK:STDERR:                      ^
choice InvalidBraces |
  Some
// CHECK:STDERR: fail_invalid_braces.carbon:[[@LINE+3]]:1: ERROR: Unrecognized declaration introducer.
// CHECK:STDERR: |
// CHECK:STDERR: ^
|

// CHECK:STDOUT: - filename: fail_invalid_braces.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'ChoiceIntroducer', text: 'choice'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'InvalidBraces'},
// CHECK:STDOUT:       {kind: 'ChoiceDefinitionStart', text: '|', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:     {kind: 'ChoiceDefinition', text: '|', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:       {kind: 'InvalidParseStart', text: '|', has_error: yes},
// CHECK:STDOUT:     {kind: 'InvalidParseSubtree', text: '|', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]

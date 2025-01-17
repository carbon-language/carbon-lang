// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/array/fail_require_close_bracket.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/array/fail_require_close_bracket.carbon
// TODO: It should emit only one error message.

// CHECK:STDERR: fail_require_close_bracket.carbon:[[@LINE+4]]:8: error: opening symbol without a corresponding closing symbol [UnmatchedOpening]
// CHECK:STDERR: var x: [i32;;
// CHECK:STDERR:        ^
// CHECK:STDERR:
var x: [i32;;

// CHECK:STDOUT: - filename: fail_require_close_bracket.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'x'},
// CHECK:STDOUT:           {kind: 'InvalidParse', text: '[', has_error: yes},
// CHECK:STDOUT:         {kind: 'VarBindingPattern', text: ':', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:       {kind: 'VariablePattern', text: 'var', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:     {kind: 'VariableDecl', text: ';', subtree_size: 6},
// CHECK:STDOUT:     {kind: 'EmptyDecl', text: ';'},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]

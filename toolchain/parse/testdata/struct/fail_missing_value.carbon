// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/struct/fail_missing_value.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/struct/fail_missing_value.carbon

// CHECK:STDERR: fail_missing_value.carbon:[[@LINE+4]]:12: error: expected expression [ExpectedExpr]
// CHECK:STDERR: var x: {.a=} = {};
// CHECK:STDERR:            ^
// CHECK:STDERR:
var x: {.a=} = {};

// CHECK:STDOUT: - filename: fail_missing_value.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'x'},
// CHECK:STDOUT:             {kind: 'StructLiteralStart', text: '{'},
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeParams', text: 'a'},
// CHECK:STDOUT:             {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'InvalidParse', text: '}', has_error: yes},
// CHECK:STDOUT:             {kind: 'InvalidParse', text: '=', has_error: yes},
// CHECK:STDOUT:           {kind: 'StructLiteral', text: '}', has_error: yes, subtree_size: 6},
// CHECK:STDOUT:         {kind: 'VarBindingPattern', text: ':', subtree_size: 8},
// CHECK:STDOUT:       {kind: 'VariablePattern', text: 'var', subtree_size: 9},
// CHECK:STDOUT:       {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'StructLiteralStart', text: '{'},
// CHECK:STDOUT:       {kind: 'StructLiteral', text: '}', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'VariableDecl', text: ';', subtree_size: 14},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]

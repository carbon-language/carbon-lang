// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/struct/fail_type_no_designator.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/struct/fail_type_no_designator.carbon

// CHECK:STDERR: fail_type_no_designator.carbon:[[@LINE+4]]:9: error: expected `.field: field_type` or `.field = value` [ExpectedStructLiteralField]
// CHECK:STDERR: var x: {i32} = {};
// CHECK:STDERR:         ^~~
// CHECK:STDERR:
var x: {i32} = {};

// CHECK:STDOUT: - filename: fail_type_no_designator.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'x'},
// CHECK:STDOUT:             {kind: 'StructLiteralStart', text: '{'},
// CHECK:STDOUT:             {kind: 'InvalidParse', text: 'i32', has_error: yes},
// CHECK:STDOUT:           {kind: 'StructLiteral', text: '}', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:         {kind: 'VarBindingPattern', text: ':', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'VariablePattern', text: 'var', subtree_size: 6},
// CHECK:STDOUT:       {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'StructLiteralStart', text: '{'},
// CHECK:STDOUT:       {kind: 'StructLiteral', text: '}', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'VariableDecl', text: ';', subtree_size: 11},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]

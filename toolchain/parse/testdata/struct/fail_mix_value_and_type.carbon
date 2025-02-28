// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/struct/fail_mix_value_and_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/struct/fail_mix_value_and_type.carbon

// CHECK:STDERR: fail_mix_value_and_type.carbon:[[@LINE+4]]:17: error: expected `.field = value` [ExpectedStructLiteralField]
// CHECK:STDERR: var x: {.a = 0, b: i32} = {};
// CHECK:STDERR:                 ^
// CHECK:STDERR:
var x: {.a = 0, b: i32} = {};

// CHECK:STDOUT: - filename: fail_mix_value_and_type.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'x'},
// CHECK:STDOUT:             {kind: 'StructLiteralStart', text: '{'},
// CHECK:STDOUT:                 {kind: 'IdentifierNameNotBeforeParams', text: 'a'},
// CHECK:STDOUT:               {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:               {kind: 'IntLiteral', text: '0'},
// CHECK:STDOUT:             {kind: 'StructLiteralField', text: '=', subtree_size: 4},
// CHECK:STDOUT:             {kind: 'StructLiteralComma', text: ','},
// CHECK:STDOUT:             {kind: 'InvalidParse', text: 'b', has_error: yes},
// CHECK:STDOUT:           {kind: 'StructLiteral', text: '}', has_error: yes, subtree_size: 8},
// CHECK:STDOUT:         {kind: 'VarBindingPattern', text: ':', subtree_size: 10},
// CHECK:STDOUT:       {kind: 'VariablePattern', text: 'var', subtree_size: 11},
// CHECK:STDOUT:       {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'StructLiteralStart', text: '{'},
// CHECK:STDOUT:       {kind: 'StructLiteral', text: '}', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'VariableDecl', text: ';', subtree_size: 16},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/struct/fail_extra_token_in_value.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/struct/fail_extra_token_in_value.carbon

// CHECK:STDERR: fail_extra_token_in_value.carbon:[[@LINE+4]]:28: error: expected `,` or `}` [UnexpectedTokenAfterListElement]
// CHECK:STDERR: var x: {.a: i32} = {.a = 0 banana};
// CHECK:STDERR:                            ^~~~~~
// CHECK:STDERR:
var x: {.a: i32} = {.a = 0 banana};

// CHECK:STDOUT: - filename: fail_extra_token_in_value.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'x'},
// CHECK:STDOUT:             {kind: 'StructTypeLiteralStart', text: '{'},
// CHECK:STDOUT:                 {kind: 'IdentifierNameNotBeforeParams', text: 'a'},
// CHECK:STDOUT:               {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:               {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:             {kind: 'StructTypeLiteralField', text: ':', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'StructTypeLiteral', text: '}', subtree_size: 6},
// CHECK:STDOUT:         {kind: 'VarBindingPattern', text: ':', subtree_size: 8},
// CHECK:STDOUT:       {kind: 'VariablePattern', text: 'var', subtree_size: 9},
// CHECK:STDOUT:       {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'StructLiteralStart', text: '{'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeParams', text: 'a'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IntLiteral', text: '0'},
// CHECK:STDOUT:         {kind: 'StructLiteralField', text: '=', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'StructLiteral', text: '}', has_error: yes, subtree_size: 6},
// CHECK:STDOUT:     {kind: 'VariableDecl', text: ';', subtree_size: 18},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]

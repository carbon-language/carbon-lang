// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/struct/fail_mix_with_unknown.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/struct/fail_mix_with_unknown.carbon

// CHECK:STDERR: fail_mix_with_unknown.carbon:[[@LINE+8]]:25: error: expected `.field = value` [ExpectedStructLiteralField]
// CHECK:STDERR: var x: i32 = {.a = 1, .b, .c: i32};
// CHECK:STDERR:                         ^
// CHECK:STDERR:
// CHECK:STDERR: fail_mix_with_unknown.carbon:[[@LINE+4]]:29: error: expected `.field = value` [ExpectedStructLiteralField]
// CHECK:STDERR: var x: i32 = {.a = 1, .b, .c: i32};
// CHECK:STDERR:                             ^
// CHECK:STDERR:
var x: i32 = {.a = 1, .b, .c: i32};

// CHECK:STDERR: fail_mix_with_unknown.carbon:[[@LINE+8]]:26: error: expected `.field: field_type` [ExpectedStructLiteralField]
// CHECK:STDERR: var x: i32 = {.a: i32, .b, .c = 1};
// CHECK:STDERR:                          ^
// CHECK:STDERR:
// CHECK:STDERR: fail_mix_with_unknown.carbon:[[@LINE+4]]:31: error: expected `.field: field_type` [ExpectedStructLiteralField]
// CHECK:STDERR: var x: i32 = {.a: i32, .b, .c = 1};
// CHECK:STDERR:                               ^
// CHECK:STDERR:
var x: i32 = {.a: i32, .b, .c = 1};

// CHECK:STDOUT: - filename: fail_mix_with_unknown.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'x'},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'VarBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'VariablePattern', text: 'var', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'StructLiteralStart', text: '{'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeParams', text: 'a'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IntLiteral', text: '1'},
// CHECK:STDOUT:         {kind: 'StructLiteralField', text: '=', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'StructLiteralComma', text: ','},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'b'},
// CHECK:STDOUT:         {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'InvalidParse', text: '.', has_error: yes},
// CHECK:STDOUT:         {kind: 'StructLiteralComma', text: ','},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'c'},
// CHECK:STDOUT:         {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'InvalidParse', text: '.', has_error: yes},
// CHECK:STDOUT:       {kind: 'StructLiteral', text: '}', has_error: yes, subtree_size: 14},
// CHECK:STDOUT:     {kind: 'VariableDecl', text: ';', subtree_size: 21},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'x'},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'VarBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'VariablePattern', text: 'var', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'StructTypeLiteralStart', text: '{'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeParams', text: 'a'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'StructTypeLiteralField', text: ':', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'StructTypeLiteralComma', text: ','},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'b'},
// CHECK:STDOUT:         {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'InvalidParse', text: '.', has_error: yes},
// CHECK:STDOUT:         {kind: 'StructTypeLiteralComma', text: ','},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'c'},
// CHECK:STDOUT:         {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'InvalidParse', text: '.', has_error: yes},
// CHECK:STDOUT:       {kind: 'StructTypeLiteral', text: '}', has_error: yes, subtree_size: 14},
// CHECK:STDOUT:     {kind: 'VariableDecl', text: ';', subtree_size: 21},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]

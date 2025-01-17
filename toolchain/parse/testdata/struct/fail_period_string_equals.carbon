// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/struct/fail_period_string_equals.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/struct/fail_period_string_equals.carbon

// CHECK:STDERR: fail_period_string_equals.carbon:[[@LINE+4]]:10: error: expected identifier after `.` [ExpectedIdentifierAfterPeriodOrArrow]
// CHECK:STDERR: var x: {."hello" = 0, .y = 4} = {};
// CHECK:STDERR:          ^~~~~~~
// CHECK:STDERR:
var x: {."hello" = 0, .y = 4} = {};

// CHECK:STDOUT: - filename: fail_period_string_equals.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'x'},
// CHECK:STDOUT:             {kind: 'StructLiteralStart', text: '{'},
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeParams', text: '"hello"', has_error: yes},
// CHECK:STDOUT:             {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'IntLiteral', text: '0'},
// CHECK:STDOUT:             {kind: 'InvalidParse', text: '=', has_error: yes},
// CHECK:STDOUT:             {kind: 'StructLiteralComma', text: ','},
// CHECK:STDOUT:                 {kind: 'IdentifierNameNotBeforeParams', text: 'y'},
// CHECK:STDOUT:               {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:               {kind: 'IntLiteral', text: '4'},
// CHECK:STDOUT:             {kind: 'StructLiteralField', text: '=', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'StructLiteral', text: '}', has_error: yes, subtree_size: 11},
// CHECK:STDOUT:         {kind: 'VarBindingPattern', text: ':', subtree_size: 13},
// CHECK:STDOUT:       {kind: 'VariablePattern', text: 'var', subtree_size: 14},
// CHECK:STDOUT:       {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'StructLiteralStart', text: '{'},
// CHECK:STDOUT:       {kind: 'StructLiteral', text: '}', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'VariableDecl', text: ';', subtree_size: 19},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]

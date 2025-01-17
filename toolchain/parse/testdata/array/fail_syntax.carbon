// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/array/fail_syntax.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/array/fail_syntax.carbon

// --- fail_type.carbon

// CHECK:STDERR: fail_type.carbon:[[@LINE+4]]:12: error: expected `;` in array type [ExpectedArraySemi]
// CHECK:STDERR: var x: [i32];
// CHECK:STDERR:            ^
// CHECK:STDERR:
var x: [i32];

// --- fail_invalid_char.carbon

// CHECK:STDERR: fail_invalid_char.carbon:[[@LINE+16]]:12: error: expected expression [ExpectedExpr]
// CHECK:STDERR: fn X() -> [:];
// CHECK:STDERR:            ^
// CHECK:STDERR:
// CHECK:STDERR: fail_invalid_char.carbon:[[@LINE+12]]:12: error: expected `;` in array type [ExpectedArraySemi]
// CHECK:STDERR: fn X() -> [:];
// CHECK:STDERR:            ^
// CHECK:STDERR:
// CHECK:STDERR: fail_invalid_char.carbon:[[@LINE+8]]:12: error: expected expression [ExpectedExpr]
// CHECK:STDERR: fn X() -> [:];
// CHECK:STDERR:            ^
// CHECK:STDERR:
// CHECK:STDERR: fail_invalid_char.carbon:[[@LINE+4]]:12: error: unexpected tokens before `]` [ExpectedCloseSymbol]
// CHECK:STDERR: fn X() -> [:];
// CHECK:STDERR:            ^
// CHECK:STDERR:
fn X() -> [:];

// --- fail_unlexed_expr.carbon

// CHECK:STDERR: fail_unlexed_expr.carbon:[[@LINE+12]]:9: error: encountered unrecognized characters while parsing [UnrecognizedCharacters]
// CHECK:STDERR: var y: [`];
// CHECK:STDERR:         ^
// CHECK:STDERR:
// CHECK:STDERR: fail_unlexed_expr.carbon:[[@LINE+8]]:9: error: expected `;` in array type [ExpectedArraySemi]
// CHECK:STDERR: var y: [`];
// CHECK:STDERR:         ^
// CHECK:STDERR:
// CHECK:STDERR: fail_unlexed_expr.carbon:[[@LINE+4]]:9: error: unexpected tokens before `]` [ExpectedCloseSymbol]
// CHECK:STDERR: var y: [`];
// CHECK:STDERR:         ^
// CHECK:STDERR:
var y: [`];

// --- fail_no_close_bracket.carbon

// CHECK:STDERR: fail_no_close_bracket.carbon:[[@LINE+4]]:8: error: opening symbol without a corresponding closing symbol [UnmatchedOpening]
// CHECK:STDERR: var x: [i32;;
// CHECK:STDERR:        ^
// CHECK:STDERR:
var x: [i32;;

// --- fail_no_semi.carbon

// CHECK:STDERR: fail_no_semi.carbon:[[@LINE+4]]:12: error: expected `;` in array type [ExpectedArraySemi]
// CHECK:STDERR: var x: [i32];
// CHECK:STDERR:            ^
// CHECK:STDERR:
var x: [i32];

// CHECK:STDOUT: - filename: fail_type.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'x'},
// CHECK:STDOUT:               {kind: 'ArrayExprStart', text: '['},
// CHECK:STDOUT:               {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:             {kind: 'ArrayExprSemi', text: ']', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:           {kind: 'ArrayExpr', text: ']', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:         {kind: 'VarBindingPattern', text: ':', subtree_size: 6},
// CHECK:STDOUT:       {kind: 'VariablePattern', text: 'var', subtree_size: 7},
// CHECK:STDOUT:     {kind: 'VariableDecl', text: ';', subtree_size: 9},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_invalid_char.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'IdentifierNameBeforeParams', text: 'X'},
// CHECK:STDOUT:         {kind: 'TuplePatternStart', text: '('},
// CHECK:STDOUT:       {kind: 'TuplePattern', text: ')', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'ArrayExprStart', text: '['},
// CHECK:STDOUT:             {kind: 'InvalidParse', text: ':', has_error: yes},
// CHECK:STDOUT:           {kind: 'ArrayExprSemi', text: ':', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:           {kind: 'InvalidParse', text: ':', has_error: yes},
// CHECK:STDOUT:         {kind: 'ArrayExpr', text: ']', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ReturnType', text: '->', subtree_size: 6},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: ';', subtree_size: 11},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_unlexed_expr.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'y'},
// CHECK:STDOUT:               {kind: 'ArrayExprStart', text: '['},
// CHECK:STDOUT:               {kind: 'InvalidParse', text: '`', has_error: yes},
// CHECK:STDOUT:             {kind: 'ArrayExprSemi', text: '`', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:             {kind: 'InvalidParse', text: '`', has_error: yes},
// CHECK:STDOUT:           {kind: 'ArrayExpr', text: ']', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:         {kind: 'VarBindingPattern', text: ':', subtree_size: 7},
// CHECK:STDOUT:       {kind: 'VariablePattern', text: 'var', subtree_size: 8},
// CHECK:STDOUT:     {kind: 'VariableDecl', text: ';', subtree_size: 10},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_no_close_bracket.carbon
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
// CHECK:STDOUT: - filename: fail_no_semi.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'x'},
// CHECK:STDOUT:               {kind: 'ArrayExprStart', text: '['},
// CHECK:STDOUT:               {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:             {kind: 'ArrayExprSemi', text: ']', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:           {kind: 'ArrayExpr', text: ']', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:         {kind: 'VarBindingPattern', text: ':', subtree_size: 6},
// CHECK:STDOUT:       {kind: 'VariablePattern', text: 'var', subtree_size: 7},
// CHECK:STDOUT:     {kind: 'VariableDecl', text: ';', subtree_size: 9},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- type.carbon

// CHECK:STDERR: type.carbon:[[@LINE+3]]:12: ERROR: Expected `;` in array type.
// CHECK:STDERR: var x: [i32];
// CHECK:STDERR:            ^
var x: [i32];

// --- invalid_char.carbon

// CHECK:STDERR: invalid_char.carbon:[[@LINE+12]]:12: ERROR: Expected expression.
// CHECK:STDERR: fn X() -> [:];
// CHECK:STDERR:            ^
// CHECK:STDERR: invalid_char.carbon:[[@LINE+9]]:12: ERROR: Expected `;` in array type.
// CHECK:STDERR: fn X() -> [:];
// CHECK:STDERR:            ^
// CHECK:STDERR: invalid_char.carbon:[[@LINE+6]]:12: ERROR: Expected expression.
// CHECK:STDERR: fn X() -> [:];
// CHECK:STDERR:            ^
// CHECK:STDERR: invalid_char.carbon:[[@LINE+3]]:12: ERROR: Unexpected tokens before `]`.
// CHECK:STDERR: fn X() -> [:];
// CHECK:STDERR:            ^
fn X() -> [:];

// --- unlexed_expr.carbon

// CHECK:STDERR: unlexed_expr.carbon:[[@LINE+15]]:9: ERROR: Encountered unrecognized characters while parsing.
// CHECK:STDERR: var y: [`];
// CHECK:STDERR:         ^
// CHECK:STDERR: unlexed_expr.carbon:[[@LINE+12]]:9: ERROR: Expected expression.
// CHECK:STDERR: var y: [`];
// CHECK:STDERR:         ^
// CHECK:STDERR: unlexed_expr.carbon:[[@LINE+9]]:9: ERROR: Expected `;` in array type.
// CHECK:STDERR: var y: [`];
// CHECK:STDERR:         ^
// CHECK:STDERR: unlexed_expr.carbon:[[@LINE+6]]:9: ERROR: Expected expression.
// CHECK:STDERR: var y: [`];
// CHECK:STDERR:         ^
// CHECK:STDERR: unlexed_expr.carbon:[[@LINE+3]]:9: ERROR: Unexpected tokens before `]`.
// CHECK:STDERR: var y: [`];
// CHECK:STDERR:         ^
var y: [`];

// --- no_close_bracket.carbon

// CHECK:STDERR: no_close_bracket.carbon:[[@LINE+9]]:8: ERROR: Closing symbol does not match most recent opening symbol.
// CHECK:STDERR: var x: [i32;;
// CHECK:STDERR:        ^
// CHECK:STDERR: no_close_bracket.carbon:[[@LINE+6]]:13: ERROR: Expected expression.
// CHECK:STDERR: var x: [i32;;
// CHECK:STDERR:             ^
// CHECK:STDERR: no_close_bracket.carbon:[[@LINE+3]]:13: ERROR: Unexpected tokens before `]`.
// CHECK:STDERR: var x: [i32;;
// CHECK:STDERR:             ^
var x: [i32;;
// CHECK:STDERR: no_close_bracket.carbon:[[@LINE+3]]:1: ERROR: `var` declarations must end with a `;`.
// CHECK:STDERR:
// CHECK:STDERR: ^

// --- no_semi.carbon

// CHECK:STDERR: no_semi.carbon:[[@LINE+3]]:12: ERROR: Expected `;` in array type.
// CHECK:STDERR: var x: [i32];
// CHECK:STDERR:            ^
var x: [i32];

// CHECK:STDOUT: - filename: type.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'x'},
// CHECK:STDOUT:             {kind: 'ArrayExprStart', text: '['},
// CHECK:STDOUT:             {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'ArrayExprSemi', text: ']', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:         {kind: 'ArrayExpr', text: ']', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:       {kind: 'BindingPattern', text: ':', subtree_size: 6},
// CHECK:STDOUT:     {kind: 'VariableDecl', text: ';', subtree_size: 8},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: invalid_char.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'IdentifierName', text: 'X'},
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
// CHECK:STDOUT: - filename: unlexed_expr.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'y'},
// CHECK:STDOUT:             {kind: 'ArrayExprStart', text: '['},
// CHECK:STDOUT:             {kind: 'InvalidParse', text: '`', has_error: yes},
// CHECK:STDOUT:           {kind: 'ArrayExprSemi', text: '`', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:           {kind: 'InvalidParse', text: '`', has_error: yes},
// CHECK:STDOUT:         {kind: 'ArrayExpr', text: ']', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:       {kind: 'BindingPattern', text: ':', subtree_size: 7},
// CHECK:STDOUT:     {kind: 'VariableDecl', text: ';', subtree_size: 9},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: no_close_bracket.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'x'},
// CHECK:STDOUT:             {kind: 'ArrayExprStart', text: '['},
// CHECK:STDOUT:             {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'ArrayExprSemi', text: ';', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'InvalidParse', text: ';', has_error: yes},
// CHECK:STDOUT:         {kind: 'ArrayExpr', text: ']', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:       {kind: 'BindingPattern', text: ':', subtree_size: 7},
// CHECK:STDOUT:     {kind: 'VariableDecl', text: 'var', has_error: yes, subtree_size: 9},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: no_semi.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'x'},
// CHECK:STDOUT:             {kind: 'ArrayExprStart', text: '['},
// CHECK:STDOUT:             {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'ArrayExprSemi', text: ']', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:         {kind: 'ArrayExpr', text: ']', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:       {kind: 'BindingPattern', text: ':', subtree_size: 6},
// CHECK:STDOUT:     {kind: 'VariableDecl', text: ';', subtree_size: 8},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// TODO: We try to form (n*)*p and reject due to missing parentheses
// before we notice the missing whitespace around the second `*`.
// It'd be better to (somehow) form n*(*p) and reject due to the missing
// whitespace around the first `*`.
// CHECK:STDERR: fail_star_star_no_space.carbon:[[@LINE+3]]:16: ERROR: `var` declarations must end with a `;`.
// CHECK:STDERR: var n: i8 = n**p;
// CHECK:STDERR:                ^
var n: i8 = n**p;

// CHECK:STDOUT: - filename: fail_star_star_no_space.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:         {kind: 'Name', text: 'n'},
// CHECK:STDOUT:         {kind: 'Literal', text: 'i8'},
// CHECK:STDOUT:       {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:           {kind: 'NameExpression', text: 'n'},
// CHECK:STDOUT:         {kind: 'PostfixOperator', text: '*', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'PostfixOperator', text: '*', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'VariableDeclaration', text: ';', has_error: yes, subtree_size: 9},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]

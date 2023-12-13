// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: recover_prefix_repeat.carbon:[[@LINE+3]]:17: ERROR: Parentheses are required around this unary `const` operator.
// CHECK:STDERR: fn F() -> const const i32* {
// CHECK:STDERR:                 ^~~~~
fn F() -> const const i32* {
}

// CHECK:STDOUT: - filename: recover_prefix_repeat.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'F'},
// CHECK:STDOUT:           {kind: 'TuplePatternStart', text: '('},
// CHECK:STDOUT:         {kind: 'TuplePattern', text: ')', subtree_size: 2},
// CHECK:STDOUT:                 {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:               {kind: 'PrefixOperator', text: 'const', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'PrefixOperator', text: 'const', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'PostfixOperator', text: '*', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'ReturnType', text: '->', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 10},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 11},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]

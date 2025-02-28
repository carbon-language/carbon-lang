// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/pointer/const_pointer.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/pointer/const_pointer.carbon

fn A() -> const i32* { return A(); }
fn B() -> (const i32)* { return B(); }
fn C() -> const (i32*) { return C(); }

// CHECK:STDOUT: - filename: const_pointer.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'A'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:               {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:             {kind: 'PrefixOperatorConst', text: 'const', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'PostfixOperatorStar', text: '*', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'ReturnType', text: '->', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 9},
// CHECK:STDOUT:         {kind: 'ReturnStatementStart', text: 'return'},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'A'},
// CHECK:STDOUT:           {kind: 'CallExprStart', text: '(', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'CallExpr', text: ')', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ReturnStatement', text: ';', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 15},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'B'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:               {kind: 'ParenExprStart', text: '('},
// CHECK:STDOUT:                 {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:               {kind: 'PrefixOperatorConst', text: 'const', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'ParenExpr', text: ')', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'PostfixOperatorStar', text: '*', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'ReturnType', text: '->', subtree_size: 6},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 11},
// CHECK:STDOUT:         {kind: 'ReturnStatementStart', text: 'return'},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'B'},
// CHECK:STDOUT:           {kind: 'CallExprStart', text: '(', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'CallExpr', text: ')', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ReturnStatement', text: ';', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 17},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'C'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:               {kind: 'ParenExprStart', text: '('},
// CHECK:STDOUT:                 {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:               {kind: 'PostfixOperatorStar', text: '*', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'ParenExpr', text: ')', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'PrefixOperatorConst', text: 'const', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'ReturnType', text: '->', subtree_size: 6},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 11},
// CHECK:STDOUT:         {kind: 'ReturnStatementStart', text: 'return'},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'C'},
// CHECK:STDOUT:           {kind: 'CallExprStart', text: '(', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'CallExpr', text: ')', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ReturnStatement', text: ';', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 17},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]

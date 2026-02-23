// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/operators/ref.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/operators/ref.carbon

// --- ref_precedence.carbon

var x: i32 = ref y as i32;

// --- fail_if_ref.carbon

// CHECK:STDERR: fail_if_ref.carbon:[[@LINE+4]]:18: error: parentheses are required around this unary `if` operator [UnaryOperatorRequiresParentheses]
// CHECK:STDERR: var x: i32 = ref if b then c else d;
// CHECK:STDERR:                  ^~
// CHECK:STDERR:
var x: i32 = ref if b then c else d;
var x: i32 = ref (if b then c else d);
var x: i32 = if b then ref c else ref d;

// CHECK:STDOUT: - filename: ref_precedence.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'VarBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'VariablePattern', text: 'var', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'y'},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'InfixOperatorAs', text: 'as', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'PrefixOperatorRef', text: 'ref', subtree_size: 4},
// CHECK:STDOUT:     {kind: 'VariableDecl', text: ';', subtree_size: 11},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_if_ref.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'VarBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'VariablePattern', text: 'var', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'b'},
// CHECK:STDOUT:           {kind: 'IfExprIf', text: 'if', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'c'},
// CHECK:STDOUT:           {kind: 'IfExprThen', text: 'then', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'd'},
// CHECK:STDOUT:         {kind: 'IfExprElse', text: 'else', subtree_size: 6},
// CHECK:STDOUT:       {kind: 'PrefixOperatorRef', text: 'ref', subtree_size: 7},
// CHECK:STDOUT:     {kind: 'VariableDecl', text: ';', subtree_size: 14},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'VarBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'VariablePattern', text: 'var', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:           {kind: 'ParenExprStart', text: '('},
// CHECK:STDOUT:               {kind: 'IdentifierNameExpr', text: 'b'},
// CHECK:STDOUT:             {kind: 'IfExprIf', text: 'if', subtree_size: 2},
// CHECK:STDOUT:               {kind: 'IdentifierNameExpr', text: 'c'},
// CHECK:STDOUT:             {kind: 'IfExprThen', text: 'then', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'd'},
// CHECK:STDOUT:           {kind: 'IfExprElse', text: 'else', subtree_size: 6},
// CHECK:STDOUT:         {kind: 'ParenExpr', text: ')', subtree_size: 8},
// CHECK:STDOUT:       {kind: 'PrefixOperatorRef', text: 'ref', subtree_size: 9},
// CHECK:STDOUT:     {kind: 'VariableDecl', text: ';', subtree_size: 16},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'VarBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'VariablePattern', text: 'var', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'b'},
// CHECK:STDOUT:         {kind: 'IfExprIf', text: 'if', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'c'},
// CHECK:STDOUT:           {kind: 'PrefixOperatorRef', text: 'ref', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'IfExprThen', text: 'then', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'd'},
// CHECK:STDOUT:         {kind: 'PrefixOperatorRef', text: 'ref', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'IfExprElse', text: 'else', subtree_size: 8},
// CHECK:STDOUT:     {kind: 'VariableDecl', text: ';', subtree_size: 15},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]

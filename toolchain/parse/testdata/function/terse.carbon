// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/function/terse.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/function/terse.carbon

fn F(x: i32) => x;

fn G(x: i32) -> i32 => x;

// CHECK:STDOUT: - filename: terse.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameMaybeBeforeSignature', text: 'F'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:             {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '=>', subtree_size: 8},
// CHECK:STDOUT:       {kind: 'TerseBodyArrow', text: '=>'},
// CHECK:STDOUT:       {kind: 'IdentifierNameExpr', text: 'x'},
// CHECK:STDOUT:     {kind: 'FunctionTerseDefinition', text: ';', subtree_size: 11},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameMaybeBeforeSignature', text: 'G'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:             {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'ReturnType', text: '->', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '=>', subtree_size: 10},
// CHECK:STDOUT:       {kind: 'TerseBodyArrow', text: '=>'},
// CHECK:STDOUT:       {kind: 'IdentifierNameExpr', text: 'x'},
// CHECK:STDOUT:     {kind: 'FunctionTerseDefinition', text: ';', subtree_size: 13},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]

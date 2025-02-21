// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/where_expr/basic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/where_expr/basic.carbon

fn Foo(T: type where .Self == i32);

interface I {
  let T:! type where .Foo impls Bar;
}

let T:! type where .U = .V;

fn F() {
  let U:! type where .W == .X;
}

// CHECK:STDOUT: - filename: basic.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'IdentifierNameBeforeParams', text: 'Foo'},
// CHECK:STDOUT:         {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'T'},
// CHECK:STDOUT:               {kind: 'TypeTypeLiteral', text: 'type'},
// CHECK:STDOUT:             {kind: 'WhereOperand', text: 'where', subtree_size: 2},
// CHECK:STDOUT:                 {kind: 'SelfTypeName', text: 'Self'},
// CHECK:STDOUT:               {kind: 'DesignatorExpr', text: '.', subtree_size: 2},
// CHECK:STDOUT:               {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:             {kind: 'RequirementEqualEqual', text: '==', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'WhereExpr', text: 'where', subtree_size: 7},
// CHECK:STDOUT:         {kind: 'LetBindingPattern', text: ':', subtree_size: 9},
// CHECK:STDOUT:       {kind: 'ExplicitParamList', text: ')', subtree_size: 11},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: ';', subtree_size: 14},
// CHECK:STDOUT:         {kind: 'InterfaceIntroducer', text: 'interface'},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: 'I'},
// CHECK:STDOUT:       {kind: 'InterfaceDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'T'},
// CHECK:STDOUT:               {kind: 'TypeTypeLiteral', text: 'type'},
// CHECK:STDOUT:             {kind: 'WhereOperand', text: 'where', subtree_size: 2},
// CHECK:STDOUT:                 {kind: 'IdentifierNameNotBeforeParams', text: 'Foo'},
// CHECK:STDOUT:               {kind: 'DesignatorExpr', text: '.', subtree_size: 2},
// CHECK:STDOUT:               {kind: 'IdentifierNameExpr', text: 'Bar'},
// CHECK:STDOUT:             {kind: 'RequirementImpls', text: 'impls', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'WhereExpr', text: 'where', subtree_size: 7},
// CHECK:STDOUT:         {kind: 'CompileTimeBindingPattern', text: ':!', subtree_size: 9},
// CHECK:STDOUT:       {kind: 'LetDecl', text: ';', subtree_size: 11},
// CHECK:STDOUT:     {kind: 'InterfaceDefinition', text: '}', subtree_size: 15},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: 'T'},
// CHECK:STDOUT:             {kind: 'TypeTypeLiteral', text: 'type'},
// CHECK:STDOUT:           {kind: 'WhereOperand', text: 'where', subtree_size: 2},
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeParams', text: 'U'},
// CHECK:STDOUT:             {kind: 'DesignatorExpr', text: '.', subtree_size: 2},
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeParams', text: 'V'},
// CHECK:STDOUT:             {kind: 'DesignatorExpr', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'RequirementEqual', text: '=', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'WhereExpr', text: 'where', subtree_size: 8},
// CHECK:STDOUT:       {kind: 'CompileTimeBindingPattern', text: ':!', subtree_size: 10},
// CHECK:STDOUT:     {kind: 'LetDecl', text: ';', subtree_size: 12},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'F'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'U'},
// CHECK:STDOUT:               {kind: 'TypeTypeLiteral', text: 'type'},
// CHECK:STDOUT:             {kind: 'WhereOperand', text: 'where', subtree_size: 2},
// CHECK:STDOUT:                 {kind: 'IdentifierNameNotBeforeParams', text: 'W'},
// CHECK:STDOUT:               {kind: 'DesignatorExpr', text: '.', subtree_size: 2},
// CHECK:STDOUT:                 {kind: 'IdentifierNameNotBeforeParams', text: 'X'},
// CHECK:STDOUT:               {kind: 'DesignatorExpr', text: '.', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'RequirementEqualEqual', text: '==', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'WhereExpr', text: 'where', subtree_size: 8},
// CHECK:STDOUT:         {kind: 'CompileTimeBindingPattern', text: ':!', subtree_size: 10},
// CHECK:STDOUT:       {kind: 'LetDecl', text: ';', subtree_size: 12},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 18},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/function/definition/decl_statement.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/function/definition/decl_statement.carbon

// --- valid.carbon

// These are expected to work.
fn F() {
  alias A = i32;
  class A {}
  base class B {}
  abstract class Abstract {}
  fn F() {}
  constraint C {}
  interface I {}
  impl A as I {}
  final impl A as I {}
  choice C {}
  let n: i32 = 0;
  var v: i32;
}

// --- syntactically_valid.carbon

// These parse but should not type-check.
fn F() {
  adapt A;
  extend adapt A;
  extend base: B;
  extend impl as I;
  namespace N;
  default fn F();
  impl fn G();
  virtual fn H();
  private var v: i32;
  protected var v: i32;
}

// CHECK:STDOUT: - filename: valid.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'F'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'AliasIntroducer', text: 'alias'},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: 'A'},
// CHECK:STDOUT:         {kind: 'AliasInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:       {kind: 'Alias', text: ';', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'ClassIntroducer', text: 'class'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'A'},
// CHECK:STDOUT:         {kind: 'ClassDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ClassDefinition', text: '}', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'ClassIntroducer', text: 'class'},
// CHECK:STDOUT:           {kind: 'BaseModifier', text: 'base'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'B'},
// CHECK:STDOUT:         {kind: 'ClassDefinitionStart', text: '{', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'ClassDefinition', text: '}', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'ClassIntroducer', text: 'class'},
// CHECK:STDOUT:           {kind: 'AbstractModifier', text: 'abstract'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'Abstract'},
// CHECK:STDOUT:         {kind: 'ClassDefinitionStart', text: '{', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'ClassDefinition', text: '}', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:           {kind: 'IdentifierNameBeforeParams', text: 'F'},
// CHECK:STDOUT:             {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:           {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'FunctionDefinition', text: '}', subtree_size: 6},
// CHECK:STDOUT:           {kind: 'NamedConstraintIntroducer', text: 'constraint'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'C'},
// CHECK:STDOUT:         {kind: 'NamedConstraintDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'NamedConstraintDefinition', text: '}', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'InterfaceIntroducer', text: 'interface'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'I'},
// CHECK:STDOUT:         {kind: 'InterfaceDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'InterfaceDefinition', text: '}', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'ImplIntroducer', text: 'impl'},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'A'},
// CHECK:STDOUT:           {kind: 'TypeImplAs', text: 'as', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'I'},
// CHECK:STDOUT:         {kind: 'ImplDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ImplDefinition', text: '}', subtree_size: 6},
// CHECK:STDOUT:           {kind: 'ImplIntroducer', text: 'impl'},
// CHECK:STDOUT:           {kind: 'FinalModifier', text: 'final'},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'A'},
// CHECK:STDOUT:           {kind: 'TypeImplAs', text: 'as', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'I'},
// CHECK:STDOUT:         {kind: 'ImplDefinitionStart', text: '{', subtree_size: 6},
// CHECK:STDOUT:       {kind: 'ImplDefinition', text: '}', subtree_size: 7},
// CHECK:STDOUT:           {kind: 'ChoiceIntroducer', text: 'choice'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'C'},
// CHECK:STDOUT:         {kind: 'ChoiceDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ChoiceDefinition', text: '}', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'n'},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'LetInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'IntLiteral', text: '0'},
// CHECK:STDOUT:       {kind: 'LetDecl', text: ';', subtree_size: 7},
// CHECK:STDOUT:         {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeParams', text: 'v'},
// CHECK:STDOUT:             {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'VarBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'VariablePattern', text: 'var', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'VariableDecl', text: ';', subtree_size: 6},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 69},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: syntactically_valid.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'F'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'AdaptIntroducer', text: 'adapt'},
// CHECK:STDOUT:         {kind: 'IdentifierNameExpr', text: 'A'},
// CHECK:STDOUT:       {kind: 'AdaptDecl', text: ';', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'AdaptIntroducer', text: 'adapt'},
// CHECK:STDOUT:         {kind: 'ExtendModifier', text: 'extend'},
// CHECK:STDOUT:         {kind: 'IdentifierNameExpr', text: 'A'},
// CHECK:STDOUT:       {kind: 'AdaptDecl', text: ';', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'BaseIntroducer', text: 'base'},
// CHECK:STDOUT:         {kind: 'ExtendModifier', text: 'extend'},
// CHECK:STDOUT:         {kind: 'BaseColon', text: ':'},
// CHECK:STDOUT:         {kind: 'IdentifierNameExpr', text: 'B'},
// CHECK:STDOUT:       {kind: 'BaseDecl', text: ';', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'ImplIntroducer', text: 'impl'},
// CHECK:STDOUT:         {kind: 'ExtendModifier', text: 'extend'},
// CHECK:STDOUT:         {kind: 'DefaultSelfImplAs', text: 'as'},
// CHECK:STDOUT:         {kind: 'IdentifierNameExpr', text: 'I'},
// CHECK:STDOUT:       {kind: 'ImplDecl', text: ';', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'NamespaceStart', text: 'namespace'},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: 'N'},
// CHECK:STDOUT:       {kind: 'Namespace', text: ';', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'DefaultModifier', text: 'default'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'F'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDecl', text: ';', subtree_size: 6},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'ImplModifier', text: 'impl'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'G'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDecl', text: ';', subtree_size: 6},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'VirtualModifier', text: 'virtual'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'H'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDecl', text: ';', subtree_size: 6},
// CHECK:STDOUT:         {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:         {kind: 'PrivateModifier', text: 'private'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeParams', text: 'v'},
// CHECK:STDOUT:             {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'VarBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'VariablePattern', text: 'var', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'VariableDecl', text: ';', subtree_size: 7},
// CHECK:STDOUT:         {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:         {kind: 'ProtectedModifier', text: 'protected'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeParams', text: 'v'},
// CHECK:STDOUT:             {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'VarBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'VariablePattern', text: 'var', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'VariableDecl', text: ';', subtree_size: 7},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 58},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/function/decl_statement.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/function/decl_statement.carbon

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
  override fn H();
  virtual fn H();
  private var v: i32;
  protected var v: i32;
}

// CHECK:STDOUT: - filename: valid.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─AliasIntroducer 'alias'
// CHECK:STDOUT:   │ │ ├─IdentifierNameNotBeforeSignature 'A'
// CHECK:STDOUT:   │ │ ├─AliasInitializer '='
// CHECK:STDOUT:   │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ ├─Alias ';'
// CHECK:STDOUT:   │ │   ╭─ClassIntroducer 'class'
// CHECK:STDOUT:   │ │   ├─IdentifierNameNotBeforeSignature 'A'
// CHECK:STDOUT:   │ │ ╭─ClassDefinitionStart '{'
// CHECK:STDOUT:   │ ├─ClassDefinition '}'
// CHECK:STDOUT:   │ │   ╭─ClassIntroducer 'class'
// CHECK:STDOUT:   │ │   ├─BaseModifier 'base'
// CHECK:STDOUT:   │ │   ├─IdentifierNameNotBeforeSignature 'B'
// CHECK:STDOUT:   │ │ ╭─ClassDefinitionStart '{'
// CHECK:STDOUT:   │ ├─ClassDefinition '}'
// CHECK:STDOUT:   │ │   ╭─ClassIntroducer 'class'
// CHECK:STDOUT:   │ │   ├─AbstractModifier 'abstract'
// CHECK:STDOUT:   │ │   ├─IdentifierNameNotBeforeSignature 'Abstract'
// CHECK:STDOUT:   │ │ ╭─ClassDefinitionStart '{'
// CHECK:STDOUT:   │ ├─ClassDefinition '}'
// CHECK:STDOUT:   │ │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │ │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ ├─FunctionDefinition '}'
// CHECK:STDOUT:   │ │   ╭─NamedConstraintIntroducer 'constraint'
// CHECK:STDOUT:   │ │   ├─IdentifierNameNotBeforeSignature 'C'
// CHECK:STDOUT:   │ │ ╭─NamedConstraintDefinitionStart '{'
// CHECK:STDOUT:   │ ├─NamedConstraintDefinition '}'
// CHECK:STDOUT:   │ │   ╭─InterfaceIntroducer 'interface'
// CHECK:STDOUT:   │ │   ├─IdentifierNameNotBeforeSignature 'I'
// CHECK:STDOUT:   │ │ ╭─InterfaceDefinitionStart '{'
// CHECK:STDOUT:   │ ├─InterfaceDefinition '}'
// CHECK:STDOUT:   │ │   ╭─ImplIntroducer 'impl'
// CHECK:STDOUT:   │ │   │ ╭─IdentifierNameExpr 'A'
// CHECK:STDOUT:   │ │   ├─ImplTypeAs 'as'
// CHECK:STDOUT:   │ │   ├─IdentifierNameExpr 'I'
// CHECK:STDOUT:   │ │ ╭─ImplDefinitionStart '{'
// CHECK:STDOUT:   │ ├─ImplDefinition '}'
// CHECK:STDOUT:   │ │   ╭─ImplIntroducer 'impl'
// CHECK:STDOUT:   │ │   ├─FinalModifier 'final'
// CHECK:STDOUT:   │ │   │ ╭─IdentifierNameExpr 'A'
// CHECK:STDOUT:   │ │   ├─ImplTypeAs 'as'
// CHECK:STDOUT:   │ │   ├─IdentifierNameExpr 'I'
// CHECK:STDOUT:   │ │ ╭─ImplDefinitionStart '{'
// CHECK:STDOUT:   │ ├─ImplDefinition '}'
// CHECK:STDOUT:   │ │   ╭─ChoiceIntroducer 'choice'
// CHECK:STDOUT:   │ │   ├─IdentifierNameNotBeforeSignature 'C'
// CHECK:STDOUT:   │ │ ╭─ChoiceDefinitionStart '{'
// CHECK:STDOUT:   │ ├─ChoiceDefinition '}'
// CHECK:STDOUT:   │ │ ╭─LetIntroducer 'let'
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameNotBeforeSignature 'n'
// CHECK:STDOUT:   │ │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─LetInitializer '='
// CHECK:STDOUT:   │ │ ├─IntLiteral '0'
// CHECK:STDOUT:   │ ├─LetDecl ';'
// CHECK:STDOUT:   │ │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'v'
// CHECK:STDOUT:   │ │ │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │   ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─VariablePattern 'var'
// CHECK:STDOUT:   │ ├─VariableDecl ';'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: syntactically_valid.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─AdaptIntroducer 'adapt'
// CHECK:STDOUT:   │ │ ├─IdentifierNameExpr 'A'
// CHECK:STDOUT:   │ ├─AdaptDecl ';'
// CHECK:STDOUT:   │ │ ╭─AdaptIntroducer 'adapt'
// CHECK:STDOUT:   │ │ ├─ExtendModifier 'extend'
// CHECK:STDOUT:   │ │ ├─IdentifierNameExpr 'A'
// CHECK:STDOUT:   │ ├─AdaptDecl ';'
// CHECK:STDOUT:   │ │ ╭─BaseIntroducer 'base'
// CHECK:STDOUT:   │ │ ├─ExtendModifier 'extend'
// CHECK:STDOUT:   │ │ ├─BaseColon ':'
// CHECK:STDOUT:   │ │ ├─IdentifierNameExpr 'B'
// CHECK:STDOUT:   │ ├─BaseDecl ';'
// CHECK:STDOUT:   │ │ ╭─ImplIntroducer 'impl'
// CHECK:STDOUT:   │ │ ├─ExtendModifier 'extend'
// CHECK:STDOUT:   │ │ ├─ImplDefaultSelfAs 'as'
// CHECK:STDOUT:   │ │ ├─IdentifierNameExpr 'I'
// CHECK:STDOUT:   │ ├─ImplDecl ';'
// CHECK:STDOUT:   │ │ ╭─NamespaceStart 'namespace'
// CHECK:STDOUT:   │ │ ├─IdentifierNameNotBeforeSignature 'N'
// CHECK:STDOUT:   │ ├─Namespace ';'
// CHECK:STDOUT:   │ │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ │ ├─DefaultModifier 'default'
// CHECK:STDOUT:   │ │ ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │ │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ │ ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ├─FunctionDecl ';'
// CHECK:STDOUT:   │ │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ │ ├─ImplModifier 'impl'
// CHECK:STDOUT:   │ │ ├─IdentifierNameMaybeBeforeSignature 'G'
// CHECK:STDOUT:   │ │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ │ ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ├─FunctionDecl ';'
// CHECK:STDOUT:   │ │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ │ ├─OverrideModifier 'override'
// CHECK:STDOUT:   │ │ ├─IdentifierNameMaybeBeforeSignature 'H'
// CHECK:STDOUT:   │ │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ │ ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ├─FunctionDecl ';'
// CHECK:STDOUT:   │ │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ │ ├─VirtualModifier 'virtual'
// CHECK:STDOUT:   │ │ ├─IdentifierNameMaybeBeforeSignature 'H'
// CHECK:STDOUT:   │ │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ │ ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ├─FunctionDecl ';'
// CHECK:STDOUT:   │ │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │ ├─PrivateModifier 'private'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'v'
// CHECK:STDOUT:   │ │ │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │   ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─VariablePattern 'var'
// CHECK:STDOUT:   │ ├─VariableDecl ';'
// CHECK:STDOUT:   │ │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │ ├─ProtectedModifier 'protected'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'v'
// CHECK:STDOUT:   │ │ │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │   ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─VariablePattern 'var'
// CHECK:STDOUT:   │ ├─VariableDecl ';'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)

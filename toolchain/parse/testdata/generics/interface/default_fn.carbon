// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/generics/interface/default_fn.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/generics/interface/default_fn.carbon

interface Foo {
  fn Scale(self, s: i32) -> Self;

  // Default definition of `Invert` calls `Scale`.
  default fn Invert(self) -> Self {
    return self.Scale(-1);
  }
}

// CHECK:STDOUT: - filename: default_fn.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─InterfaceIntroducer 'interface'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'Foo'
// CHECK:STDOUT:   │ ╭─InterfaceDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ │ ├─IdentifierNameMaybeBeforeSignature 'Scale'
// CHECK:STDOUT:   │ │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ │ │ │ ╭─SelfValueName 'self'
// CHECK:STDOUT:   │ │ │ ├─SelfBindingPattern 'self'
// CHECK:STDOUT:   │ │ │ ├─PatternListComma ','
// CHECK:STDOUT:   │ │ │ │ ╭─IdentifierNameNotBeforeSignature 's'
// CHECK:STDOUT:   │ │ │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ │ │ ╭─SelfTypeNameExpr 'Self'
// CHECK:STDOUT:   │ │ ├─ReturnType '->'
// CHECK:STDOUT:   │ ├─FunctionDecl ';'
// CHECK:STDOUT:   │ │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ │   ├─DefaultModifier 'default'
// CHECK:STDOUT:   │ │   ├─IdentifierNameMaybeBeforeSignature 'Invert'
// CHECK:STDOUT:   │ │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ │   │ │ ╭─SelfValueName 'self'
// CHECK:STDOUT:   │ │   │ ├─SelfBindingPattern 'self'
// CHECK:STDOUT:   │ │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ │   │ ╭─SelfTypeNameExpr 'Self'
// CHECK:STDOUT:   │ │   ├─ReturnType '->'
// CHECK:STDOUT:   │ │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │ │ ╭─ReturnStatementStart 'return'
// CHECK:STDOUT:   │ │ │ │     ╭─SelfValueNameExpr 'self'
// CHECK:STDOUT:   │ │ │ │     ├─IdentifierNameNotBeforeSignature 'Scale'
// CHECK:STDOUT:   │ │ │ │   ╭─MemberAccessExpr '.'
// CHECK:STDOUT:   │ │ │ │ ╭─CallExprStart '('
// CHECK:STDOUT:   │ │ │ │ │ ╭─IntLiteral '1'
// CHECK:STDOUT:   │ │ │ │ ├─PrefixOperatorMinus '-'
// CHECK:STDOUT:   │ │ │ ├─CallExpr ')'
// CHECK:STDOUT:   │ │ ├─ReturnStatement ';'
// CHECK:STDOUT:   │ ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─InterfaceDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)

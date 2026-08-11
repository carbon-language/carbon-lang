// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/generics/interface/final_fn.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/generics/interface/final_fn.carbon

interface Order {
  fn Less(self, right: Self) -> bool;
  final fn Greater(self, right: Self) -> bool {
    return right.Less(self);
  }
}

// CHECK:STDOUT: - filename: final_fn.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─InterfaceIntroducer 'interface'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'Order'
// CHECK:STDOUT:   │ ╭─InterfaceDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ │ ├─IdentifierNameMaybeBeforeSignature 'Less'
// CHECK:STDOUT:   │ │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ │ │ │ ╭─SelfValueName 'self'
// CHECK:STDOUT:   │ │ │ ├─SelfBindingPattern 'self'
// CHECK:STDOUT:   │ │ │ ├─PatternListComma ','
// CHECK:STDOUT:   │ │ │ │ ╭─IdentifierNameNotBeforeSignature 'right'
// CHECK:STDOUT:   │ │ │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ │ ├─SelfTypeNameExpr 'Self'
// CHECK:STDOUT:   │ │ │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ │ │ ╭─BoolTypeLiteral 'bool'
// CHECK:STDOUT:   │ │ ├─ReturnType '->'
// CHECK:STDOUT:   │ ├─FunctionDecl ';'
// CHECK:STDOUT:   │ │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ │   ├─FinalModifier 'final'
// CHECK:STDOUT:   │ │   ├─IdentifierNameMaybeBeforeSignature 'Greater'
// CHECK:STDOUT:   │ │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ │   │ │ ╭─SelfValueName 'self'
// CHECK:STDOUT:   │ │   │ ├─SelfBindingPattern 'self'
// CHECK:STDOUT:   │ │   │ ├─PatternListComma ','
// CHECK:STDOUT:   │ │   │ │ ╭─IdentifierNameNotBeforeSignature 'right'
// CHECK:STDOUT:   │ │   │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │   │ │ ├─SelfTypeNameExpr 'Self'
// CHECK:STDOUT:   │ │   │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ │   │ ╭─BoolTypeLiteral 'bool'
// CHECK:STDOUT:   │ │   ├─ReturnType '->'
// CHECK:STDOUT:   │ │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │ │ ╭─ReturnStatementStart 'return'
// CHECK:STDOUT:   │ │ │ │     ╭─IdentifierNameExpr 'right'
// CHECK:STDOUT:   │ │ │ │     ├─IdentifierNameNotBeforeSignature 'Less'
// CHECK:STDOUT:   │ │ │ │   ╭─MemberAccessExpr '.'
// CHECK:STDOUT:   │ │ │ │ ╭─CallExprStart '('
// CHECK:STDOUT:   │ │ │ │ ├─SelfValueNameExpr 'self'
// CHECK:STDOUT:   │ │ │ ├─CallExpr ')'
// CHECK:STDOUT:   │ │ ├─ReturnStatement ';'
// CHECK:STDOUT:   │ ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─InterfaceDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)

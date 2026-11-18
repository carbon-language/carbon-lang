// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/generics/named_constraint/invalid_method.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/generics/named_constraint/invalid_method.carbon

constraint E {
  // This is an error in the check phase.
  fn DeclaredMethod(self, b: Self) -> Self;
}

constraint Foo {
  // This is an error in the check phase.
  fn DefinedMethod(unused self, b: Self) -> Self {}
}

// CHECK:STDOUT: - filename: invalid_method.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─NamedConstraintIntroducer 'constraint'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'E'
// CHECK:STDOUT:   │ ╭─NamedConstraintDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ │ ├─IdentifierNameMaybeBeforeSignature 'DeclaredMethod'
// CHECK:STDOUT:   │ │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ │ │ │ ╭─SelfValueName 'self'
// CHECK:STDOUT:   │ │ │ ├─SelfBindingPattern 'self'
// CHECK:STDOUT:   │ │ │ ├─PatternListComma ','
// CHECK:STDOUT:   │ │ │ │ ╭─IdentifierNameNotBeforeSignature 'b'
// CHECK:STDOUT:   │ │ │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ │ ├─SelfTypeNameExpr 'Self'
// CHECK:STDOUT:   │ │ │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ │ │ ╭─SelfTypeNameExpr 'Self'
// CHECK:STDOUT:   │ │ ├─ReturnType '->'
// CHECK:STDOUT:   │ ├─FunctionDecl ';'
// CHECK:STDOUT:   ├─NamedConstraintDefinition '}'
// CHECK:STDOUT:   │   ╭─NamedConstraintIntroducer 'constraint'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'Foo'
// CHECK:STDOUT:   │ ╭─NamedConstraintDefinitionStart '{'
// CHECK:STDOUT:   │ │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ │   ├─IdentifierNameMaybeBeforeSignature 'DefinedMethod'
// CHECK:STDOUT:   │ │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ │   │ │   ╭─SelfValueName 'self'
// CHECK:STDOUT:   │ │   │ │ ╭─SelfBindingPattern 'self'
// CHECK:STDOUT:   │ │   │ ├─UnusedPattern 'unused'
// CHECK:STDOUT:   │ │   │ ├─PatternListComma ','
// CHECK:STDOUT:   │ │   │ │ ╭─IdentifierNameNotBeforeSignature 'b'
// CHECK:STDOUT:   │ │   │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │   │ │ ├─SelfTypeNameExpr 'Self'
// CHECK:STDOUT:   │ │   │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ │   │ ╭─SelfTypeNameExpr 'Self'
// CHECK:STDOUT:   │ │   ├─ReturnType '->'
// CHECK:STDOUT:   │ │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─NamedConstraintDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)

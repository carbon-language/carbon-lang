// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/generics/interface/self_ref.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/generics/interface/self_ref.carbon

interface Foo {
  fn Sub(ref self, b: Self) -> Self;
  fn Sub(ref self: foo, b: Self) -> Self;
}

// CHECK:STDOUT: - filename: self_ref.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─InterfaceIntroducer 'interface'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'Foo'
// CHECK:STDOUT:   │ ╭─InterfaceDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ │ ├─IdentifierNameMaybeBeforeSignature 'Sub'
// CHECK:STDOUT:   │ │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ │ │ │   ╭─SelfValueName 'self'
// CHECK:STDOUT:   │ │ │ │ ╭─RefBindingName 'ref'
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
// CHECK:STDOUT:   │ │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ │ ├─IdentifierNameMaybeBeforeSignature 'Sub'
// CHECK:STDOUT:   │ │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ │ │ │   ╭─SelfValueName 'self'
// CHECK:STDOUT:   │ │ │ │ ╭─RefBindingName 'ref'
// CHECK:STDOUT:   │ │ │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ │ ├─IdentifierNameExpr 'foo'
// CHECK:STDOUT:   │ │ │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ │ │ ├─PatternListComma ','
// CHECK:STDOUT:   │ │ │ │ ╭─IdentifierNameNotBeforeSignature 'b'
// CHECK:STDOUT:   │ │ │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ │ ├─SelfTypeNameExpr 'Self'
// CHECK:STDOUT:   │ │ │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ │ │ ╭─SelfTypeNameExpr 'Self'
// CHECK:STDOUT:   │ │ ├─ReturnType '->'
// CHECK:STDOUT:   │ ├─FunctionDecl ';'
// CHECK:STDOUT:   ├─InterfaceDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/generics/interface/fail_self_param_syntax.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/generics/interface/fail_self_param_syntax.carbon

interface Foo {
  // CHECK:STDERR: fail_self_param_syntax.carbon:[[@LINE+4]]:13: error: expected `,` or `]` [UnexpectedTokenAfterListElement]
  // CHECK:STDERR:   fn Sub[me Self](b: Self) -> Self;
  // CHECK:STDERR:             ^~~~
  // CHECK:STDERR:
  fn Sub[me Self](b: Self) -> Self;

  fn Mul[Self](b: Self) -> Self;
}

// CHECK:STDOUT: - filename: fail_self_param_syntax.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─InterfaceIntroducer 'interface'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'Foo'
// CHECK:STDOUT:   │ ╭─InterfaceDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ │ ├─IdentifierNameMaybeBeforeSignature 'Sub'
// CHECK:STDOUT:   │ │ │ ╭─ImplicitParamListStart '['
// CHECK:STDOUT:   │ │ │ ├─IdentifierNameExpr 'me'
// CHECK:STDOUT:   │ │ ├─ImplicitParamList ']' has_error
// CHECK:STDOUT:   │ │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ │ │ │ ╭─IdentifierNameNotBeforeSignature 'b'
// CHECK:STDOUT:   │ │ │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ │ ├─SelfTypeNameExpr 'Self'
// CHECK:STDOUT:   │ │ │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ │ │ ╭─SelfTypeNameExpr 'Self'
// CHECK:STDOUT:   │ │ ├─ReturnType '->'
// CHECK:STDOUT:   │ ├─FunctionDecl ';'
// CHECK:STDOUT:   │ │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ │ ├─IdentifierNameMaybeBeforeSignature 'Mul'
// CHECK:STDOUT:   │ │ │ ╭─ImplicitParamListStart '['
// CHECK:STDOUT:   │ │ │ ├─SelfTypeNameExpr 'Self'
// CHECK:STDOUT:   │ │ ├─ImplicitParamList ']'
// CHECK:STDOUT:   │ │ │ ╭─ExplicitParamListStart '('
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

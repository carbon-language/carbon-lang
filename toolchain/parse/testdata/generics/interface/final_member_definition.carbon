// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/generics/interface/final_member_definition.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/generics/interface/final_member_definition.carbon

interface Foo {
  final fn Add(unused self, b: Self) -> Self {
    print("Allowed.");
  }
}

// CHECK:STDOUT: - filename: final_member_definition.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─InterfaceIntroducer 'interface'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'Foo'
// CHECK:STDOUT:   │ ╭─InterfaceDefinitionStart '{'
// CHECK:STDOUT:   │ │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ │   ├─FinalModifier 'final'
// CHECK:STDOUT:   │ │   ├─IdentifierNameMaybeBeforeSignature 'Add'
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
// CHECK:STDOUT:   │ │ │     ╭─IdentifierNameExpr 'print'
// CHECK:STDOUT:   │ │ │   ╭─CallExprStart '('
// CHECK:STDOUT:   │ │ │   ├─StringLiteral '"Allowed."'
// CHECK:STDOUT:   │ │ │ ╭─CallExpr ')'
// CHECK:STDOUT:   │ │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─InterfaceDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)

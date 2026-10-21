// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/class/fn_definitions.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/class/fn_definitions.carbon

class Foo {
  fn Make() -> Self { return {.x = 0}; }

  fn Baz(self) -> i32 { return self.x; }

  fn Modify(ref self) { self->x = 1; }
}

// CHECK:STDOUT: - filename: fn_definitions.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─ClassIntroducer 'class'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'Foo'
// CHECK:STDOUT:   │ ╭─ClassDefinitionStart '{'
// CHECK:STDOUT:   │ │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ │   ├─IdentifierNameMaybeBeforeSignature 'Make'
// CHECK:STDOUT:   │ │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ │   │ ╭─SelfTypeNameExpr 'Self'
// CHECK:STDOUT:   │ │   ├─ReturnType '->'
// CHECK:STDOUT:   │ │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │ │ ╭─ReturnStatementStart 'return'
// CHECK:STDOUT:   │ │ │ │ ╭─StructLiteralStart '{'
// CHECK:STDOUT:   │ │ │ │ │   ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │ │ │ │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │ │ │ │ ├─IntLiteral '0'
// CHECK:STDOUT:   │ │ │ │ ├─StructLiteralField '='
// CHECK:STDOUT:   │ │ │ ├─StructLiteral '}'
// CHECK:STDOUT:   │ │ ├─ReturnStatement ';'
// CHECK:STDOUT:   │ ├─FunctionDefinition '}'
// CHECK:STDOUT:   │ │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ │   ├─IdentifierNameMaybeBeforeSignature 'Baz'
// CHECK:STDOUT:   │ │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ │   │ │ ╭─SelfValueName 'self'
// CHECK:STDOUT:   │ │   │ ├─SelfBindingPattern 'self'
// CHECK:STDOUT:   │ │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ │   │ ╭─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │   ├─ReturnType '->'
// CHECK:STDOUT:   │ │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │ │ ╭─ReturnStatementStart 'return'
// CHECK:STDOUT:   │ │ │ │ ╭─SelfValueNameExpr 'self'
// CHECK:STDOUT:   │ │ │ │ ├─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │ │ ├─MemberAccessExpr '.'
// CHECK:STDOUT:   │ │ ├─ReturnStatement ';'
// CHECK:STDOUT:   │ ├─FunctionDefinition '}'
// CHECK:STDOUT:   │ │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ │   ├─IdentifierNameMaybeBeforeSignature 'Modify'
// CHECK:STDOUT:   │ │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ │   │ │   ╭─SelfValueName 'self'
// CHECK:STDOUT:   │ │   │ │ ╭─RefBindingName 'ref'
// CHECK:STDOUT:   │ │   │ ├─SelfBindingPattern 'self'
// CHECK:STDOUT:   │ │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │ │     ╭─SelfValueNameExpr 'self'
// CHECK:STDOUT:   │ │ │     ├─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │ │   ╭─PointerMemberAccessExpr '->'
// CHECK:STDOUT:   │ │ │   ├─IntLiteral '1'
// CHECK:STDOUT:   │ │ │ ╭─InfixOperatorEqual '='
// CHECK:STDOUT:   │ │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─ClassDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)

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
  let T: type where .Foo impls Bar;
}

let generic T: type where .U = .V;

fn F() {
  let generic U: type where .W == .X;
}

// CHECK:STDOUT: - filename: basic.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ ├─IdentifierNameMaybeBeforeSignature 'Foo'
// CHECK:STDOUT:   │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameNotBeforeSignature 'T'
// CHECK:STDOUT:   │ │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ │   ╭─TypeTypeLiteral 'type'
// CHECK:STDOUT:   │ │ │ │ ╭─WhereOperand 'where'
// CHECK:STDOUT:   │ │ │ │ │   ╭─SelfTypeName 'Self'
// CHECK:STDOUT:   │ │ │ │ │ ╭─DesignatorExpr '.'
// CHECK:STDOUT:   │ │ │ │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ │ │ ├─RequirementEqualEqual '=='
// CHECK:STDOUT:   │ │ │ ├─WhereExpr 'where'
// CHECK:STDOUT:   │ │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ ├─ExplicitParamList ')'
// CHECK:STDOUT:   ├─FunctionDecl ';'
// CHECK:STDOUT:   │   ╭─InterfaceIntroducer 'interface'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'I'
// CHECK:STDOUT:   │ ╭─InterfaceDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─AssociatedConstantIntroducer 'let'
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameNotBeforeSignature 'T'
// CHECK:STDOUT:   │ │ │ │   ╭─TypeTypeLiteral 'type'
// CHECK:STDOUT:   │ │ │ │ ╭─WhereOperand 'where'
// CHECK:STDOUT:   │ │ │ │ │   ╭─IdentifierNameNotBeforeSignature 'Foo'
// CHECK:STDOUT:   │ │ │ │ │ ╭─DesignatorExpr '.'
// CHECK:STDOUT:   │ │ │ │ │ ├─IdentifierNameExpr 'Bar'
// CHECK:STDOUT:   │ │ │ │ ├─RequirementImpls 'impls'
// CHECK:STDOUT:   │ │ │ ├─WhereExpr 'where'
// CHECK:STDOUT:   │ │ ├─AssociatedConstantNameAndType ':'
// CHECK:STDOUT:   │ ├─AssociatedConstantDecl ';'
// CHECK:STDOUT:   ├─InterfaceDefinition '}'
// CHECK:STDOUT:   │ ╭─LetIntroducer 'let'
// CHECK:STDOUT:   │ │ ╭─IdentifierNameNotBeforeSignature 'T'
// CHECK:STDOUT:   │ │ ├─CompileTimeBindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │   ╭─TypeTypeLiteral 'type'
// CHECK:STDOUT:   │ │ │ ╭─WhereOperand 'where'
// CHECK:STDOUT:   │ │ │ │   ╭─IdentifierNameNotBeforeSignature 'U'
// CHECK:STDOUT:   │ │ │ │ ╭─DesignatorExpr '.'
// CHECK:STDOUT:   │ │ │ │ │ ╭─IdentifierNameNotBeforeSignature 'V'
// CHECK:STDOUT:   │ │ │ │ ├─DesignatorExpr '.'
// CHECK:STDOUT:   │ │ │ ├─RequirementEqual '='
// CHECK:STDOUT:   │ │ ├─WhereExpr 'where'
// CHECK:STDOUT:   │ ├─CompileTimeBindingPattern ':'
// CHECK:STDOUT:   ├─LetDecl ';'
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─LetIntroducer 'let'
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameNotBeforeSignature 'U'
// CHECK:STDOUT:   │ │ │ ├─CompileTimeBindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ │   ╭─TypeTypeLiteral 'type'
// CHECK:STDOUT:   │ │ │ │ ╭─WhereOperand 'where'
// CHECK:STDOUT:   │ │ │ │ │   ╭─IdentifierNameNotBeforeSignature 'W'
// CHECK:STDOUT:   │ │ │ │ │ ╭─DesignatorExpr '.'
// CHECK:STDOUT:   │ │ │ │ │ │ ╭─IdentifierNameNotBeforeSignature 'X'
// CHECK:STDOUT:   │ │ │ │ │ ├─DesignatorExpr '.'
// CHECK:STDOUT:   │ │ │ │ ├─RequirementEqualEqual '=='
// CHECK:STDOUT:   │ │ │ ├─WhereExpr 'where'
// CHECK:STDOUT:   │ │ ├─CompileTimeBindingPattern ':'
// CHECK:STDOUT:   │ ├─LetDecl ';'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)

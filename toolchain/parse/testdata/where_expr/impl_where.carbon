// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/where_expr/impl_where.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/where_expr/impl_where.carbon

interface Interface {
  let T: type;
}

impl i32 as Interface where .T = () {
}

impl bool as Interface where .T = i32;

impl () as (Interface where .T = {}) {}

impl {} as (Interface where .T = ()) where .T = ();

impl f64 as Interface where .T = (type where .Self impls type) {
}

// CHECK:STDOUT: - filename: impl_where.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─InterfaceIntroducer 'interface'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'Interface'
// CHECK:STDOUT:   │ ╭─InterfaceDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─AssociatedConstantIntroducer 'let'
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameNotBeforeSignature 'T'
// CHECK:STDOUT:   │ │ │ ├─TypeTypeLiteral 'type'
// CHECK:STDOUT:   │ │ ├─AssociatedConstantNameAndType ':'
// CHECK:STDOUT:   │ ├─AssociatedConstantDecl ';'
// CHECK:STDOUT:   ├─InterfaceDefinition '}'
// CHECK:STDOUT:   │   ╭─ImplIntroducer 'impl'
// CHECK:STDOUT:   │   │ ╭─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │   ├─ImplTypeAs 'as'
// CHECK:STDOUT:   │   │   ╭─IdentifierNameExpr 'Interface'
// CHECK:STDOUT:   │   │ ╭─WhereOperand 'where'
// CHECK:STDOUT:   │   │ │   ╭─IdentifierNameNotBeforeSignature 'T'
// CHECK:STDOUT:   │   │ │ ╭─DesignatorExpr '.'
// CHECK:STDOUT:   │   │ │ │ ╭─TupleLiteralStart '('
// CHECK:STDOUT:   │   │ │ ├─TupleLiteral ')'
// CHECK:STDOUT:   │   │ ├─RequirementEqual '='
// CHECK:STDOUT:   │   ├─WhereExpr 'where'
// CHECK:STDOUT:   │ ╭─ImplDefinitionStart '{'
// CHECK:STDOUT:   ├─ImplDefinition '}'
// CHECK:STDOUT:   │ ╭─ImplIntroducer 'impl'
// CHECK:STDOUT:   │ │ ╭─BoolTypeLiteral 'bool'
// CHECK:STDOUT:   │ ├─ImplTypeAs 'as'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameExpr 'Interface'
// CHECK:STDOUT:   │ │ ╭─WhereOperand 'where'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'T'
// CHECK:STDOUT:   │ │ │ ╭─DesignatorExpr '.'
// CHECK:STDOUT:   │ │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ ├─RequirementEqual '='
// CHECK:STDOUT:   │ ├─WhereExpr 'where'
// CHECK:STDOUT:   ├─ImplDecl ';'
// CHECK:STDOUT:   │   ╭─ImplIntroducer 'impl'
// CHECK:STDOUT:   │   │   ╭─TupleLiteralStart '('
// CHECK:STDOUT:   │   │ ╭─TupleLiteral ')'
// CHECK:STDOUT:   │   ├─ImplTypeAs 'as'
// CHECK:STDOUT:   │   │ ╭─ParenExprStart '('
// CHECK:STDOUT:   │   │ │   ╭─IdentifierNameExpr 'Interface'
// CHECK:STDOUT:   │   │ │ ╭─WhereOperand 'where'
// CHECK:STDOUT:   │   │ │ │   ╭─IdentifierNameNotBeforeSignature 'T'
// CHECK:STDOUT:   │   │ │ │ ╭─DesignatorExpr '.'
// CHECK:STDOUT:   │   │ │ │ │ ╭─StructLiteralStart '{'
// CHECK:STDOUT:   │   │ │ │ ├─StructLiteral '}'
// CHECK:STDOUT:   │   │ │ ├─RequirementEqual '='
// CHECK:STDOUT:   │   │ ├─WhereExpr 'where'
// CHECK:STDOUT:   │   ├─ParenExpr ')'
// CHECK:STDOUT:   │ ╭─ImplDefinitionStart '{'
// CHECK:STDOUT:   ├─ImplDefinition '}'
// CHECK:STDOUT:   │ ╭─ImplIntroducer 'impl'
// CHECK:STDOUT:   │ │   ╭─StructLiteralStart '{'
// CHECK:STDOUT:   │ │ ╭─StructLiteral '}'
// CHECK:STDOUT:   │ ├─ImplTypeAs 'as'
// CHECK:STDOUT:   │ │     ╭─ParenExprStart '('
// CHECK:STDOUT:   │ │     │   ╭─IdentifierNameExpr 'Interface'
// CHECK:STDOUT:   │ │     │ ╭─WhereOperand 'where'
// CHECK:STDOUT:   │ │     │ │   ╭─IdentifierNameNotBeforeSignature 'T'
// CHECK:STDOUT:   │ │     │ │ ╭─DesignatorExpr '.'
// CHECK:STDOUT:   │ │     │ │ │ ╭─TupleLiteralStart '('
// CHECK:STDOUT:   │ │     │ │ ├─TupleLiteral ')'
// CHECK:STDOUT:   │ │     │ ├─RequirementEqual '='
// CHECK:STDOUT:   │ │     ├─WhereExpr 'where'
// CHECK:STDOUT:   │ │   ╭─ParenExpr ')'
// CHECK:STDOUT:   │ │ ╭─WhereOperand 'where'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'T'
// CHECK:STDOUT:   │ │ │ ╭─DesignatorExpr '.'
// CHECK:STDOUT:   │ │ │ │ ╭─TupleLiteralStart '('
// CHECK:STDOUT:   │ │ │ ├─TupleLiteral ')'
// CHECK:STDOUT:   │ │ ├─RequirementEqual '='
// CHECK:STDOUT:   │ ├─WhereExpr 'where'
// CHECK:STDOUT:   ├─ImplDecl ';'
// CHECK:STDOUT:   │   ╭─ImplIntroducer 'impl'
// CHECK:STDOUT:   │   │ ╭─FloatTypeLiteral 'f64'
// CHECK:STDOUT:   │   ├─ImplTypeAs 'as'
// CHECK:STDOUT:   │   │   ╭─IdentifierNameExpr 'Interface'
// CHECK:STDOUT:   │   │ ╭─WhereOperand 'where'
// CHECK:STDOUT:   │   │ │   ╭─IdentifierNameNotBeforeSignature 'T'
// CHECK:STDOUT:   │   │ │ ╭─DesignatorExpr '.'
// CHECK:STDOUT:   │   │ │ │ ╭─ParenExprStart '('
// CHECK:STDOUT:   │   │ │ │ │   ╭─TypeTypeLiteral 'type'
// CHECK:STDOUT:   │   │ │ │ │ ╭─WhereOperand 'where'
// CHECK:STDOUT:   │   │ │ │ │ │   ╭─SelfTypeName 'Self'
// CHECK:STDOUT:   │   │ │ │ │ │ ╭─DesignatorExpr '.'
// CHECK:STDOUT:   │   │ │ │ │ │ ├─TypeTypeLiteral 'type'
// CHECK:STDOUT:   │   │ │ │ │ ├─RequirementImpls 'impls'
// CHECK:STDOUT:   │   │ │ │ ├─WhereExpr 'where'
// CHECK:STDOUT:   │   │ │ ├─ParenExpr ')'
// CHECK:STDOUT:   │   │ ├─RequirementEqual '='
// CHECK:STDOUT:   │   ├─WhereExpr 'where'
// CHECK:STDOUT:   │ ╭─ImplDefinitionStart '{'
// CHECK:STDOUT:   ├─ImplDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/where_assignment/basic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/where_assignment/basic.carbon

impl C as Z {
  where Z1 = D.(Y.Y1);
  where Z2 = Z1;
}

// CHECK:STDOUT: - filename: basic.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─ImplIntroducer 'impl'
// CHECK:STDOUT:   │   │ ╭─IdentifierNameExpr 'C'
// CHECK:STDOUT:   │   ├─ImplTypeAs 'as'
// CHECK:STDOUT:   │   ├─IdentifierNameExpr 'Z'
// CHECK:STDOUT:   │ ╭─ImplDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─WhereAssignmentIntroducer 'where'
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameExpr 'Z1'
// CHECK:STDOUT:   │ │ │ │ ╭─IdentifierNameExpr 'D'
// CHECK:STDOUT:   │ │ │ │ │ ╭─ParenExprStart '('
// CHECK:STDOUT:   │ │ │ │ │ │ ╭─IdentifierNameExpr 'Y'
// CHECK:STDOUT:   │ │ │ │ │ │ ├─IdentifierNameNotBeforeSignature 'Y1'
// CHECK:STDOUT:   │ │ │ │ │ ├─MemberAccessExpr '.'
// CHECK:STDOUT:   │ │ │ │ ├─ParenExpr ')'
// CHECK:STDOUT:   │ │ │ ├─MemberAccessExpr '.'
// CHECK:STDOUT:   │ │ ├─WhereAssignmentOperator '='
// CHECK:STDOUT:   │ ├─WhereAssignmentDecl ';'
// CHECK:STDOUT:   │ │ ╭─WhereAssignmentIntroducer 'where'
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameExpr 'Z2'
// CHECK:STDOUT:   │ │ │ ├─IdentifierNameExpr 'Z1'
// CHECK:STDOUT:   │ │ ├─WhereAssignmentOperator '='
// CHECK:STDOUT:   │ ├─WhereAssignmentDecl ';'
// CHECK:STDOUT:   ├─ImplDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)

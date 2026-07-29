// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/generics/impl/fail_out_of_line_member.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/generics/impl/fail_out_of_line_member.carbon

// TODO: Decide if we want to support this syntax. See #3763.

interface Interface {
  fn F();
}

impl bool as Interface {
  fn F();
}

// CHECK:STDERR: fail_out_of_line_member.carbon:[[@LINE+4]]:4: error: `fn` introducer should be followed by a name [ExpectedDeclName]
// CHECK:STDERR: fn (bool as Interface).F() {}
// CHECK:STDERR:    ^
// CHECK:STDERR:
fn (bool as Interface).F() {}

class C {
  impl Self as Interface {
    fn F();
  }
}

// CHECK:STDERR: fail_out_of_line_member.carbon:[[@LINE+4]]:6: error: `.` should be followed by a name [ExpectedDeclNameAfterPeriod]
// CHECK:STDERR: fn C.(Self as Interface).F() {}
// CHECK:STDERR:      ^
// CHECK:STDERR:
fn C.(Self as Interface).F() {}

// CHECK:STDOUT: - filename: fail_out_of_line_member.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─InterfaceIntroducer 'interface'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'Interface'
// CHECK:STDOUT:   │ ╭─InterfaceDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ │ ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │ │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ │ ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ├─FunctionDecl ';'
// CHECK:STDOUT:   ├─InterfaceDefinition '}'
// CHECK:STDOUT:   │   ╭─ImplIntroducer 'impl'
// CHECK:STDOUT:   │   │ ╭─BoolTypeLiteral 'bool'
// CHECK:STDOUT:   │   ├─ImplTypeAs 'as'
// CHECK:STDOUT:   │   ├─IdentifierNameExpr 'Interface'
// CHECK:STDOUT:   │ ╭─ImplDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ │ ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │ │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ │ ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ├─FunctionDecl ';'
// CHECK:STDOUT:   ├─ImplDefinition '}'
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ ├─InvalidParse '(' ❌
// CHECK:STDOUT:   ├─FunctionDecl '}' ❌
// CHECK:STDOUT:   │   ╭─ClassIntroducer 'class'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'C'
// CHECK:STDOUT:   │ ╭─ClassDefinitionStart '{'
// CHECK:STDOUT:   │ │   ╭─ImplIntroducer 'impl'
// CHECK:STDOUT:   │ │   │ ╭─SelfTypeNameExpr 'Self'
// CHECK:STDOUT:   │ │   ├─ImplTypeAs 'as'
// CHECK:STDOUT:   │ │   ├─IdentifierNameExpr 'Interface'
// CHECK:STDOUT:   │ │ ╭─ImplDefinitionStart '{'
// CHECK:STDOUT:   │ │ │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ │ │ ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │ │ │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ │ │ ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ │ ├─FunctionDecl ';'
// CHECK:STDOUT:   │ ├─ImplDefinition '}'
// CHECK:STDOUT:   ├─ClassDefinition '}'
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ │ ╭─IdentifierNameNotBeforeSignature 'C'
// CHECK:STDOUT:   │ ├─IdentifierNameQualifierWithoutParams '.'
// CHECK:STDOUT:   │ ├─InvalidParse '(' ❌
// CHECK:STDOUT:   ├─FunctionDecl '}' ❌
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/packages/package/fail_in_nested_scope.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/packages/package/fail_in_nested_scope.carbon

// api.carbon

class C {
  // CHECK:STDERR: fail_in_nested_scope.carbon:[[@LINE+7]]:3: error: the `package` declaration must be the first non-comment line [PackageTooLate]
  // CHECK:STDERR:   package P;
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR: fail_in_nested_scope.carbon:[[@LINE-4]]:1: note: first non-comment line is here [FirstNonCommentLine]
  // CHECK:STDERR: class C {
  // CHECK:STDERR: ^~~~~
  // CHECK:STDERR:
  package P;
}

fn F() {
  // `package` here is an expression naming the package scope.
  // CHECK:STDERR: fail_in_nested_scope.carbon:[[@LINE+4]]:11: error: expected `.` after `package` expression [ExpectedPeriodAfterPackage]
  // CHECK:STDERR:   package P;
  // CHECK:STDERR:           ^
  // CHECK:STDERR:
  package P;

  // OK
  package.P;
}

// impl.carbon

class C {
  // CHECK:STDERR: fail_in_nested_scope.carbon:[[@LINE+7]]:3: error: the `package` declaration must be the first non-comment line [PackageTooLate]
  // CHECK:STDERR:   impl package P;
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR: fail_in_nested_scope.carbon:[[@LINE-29]]:1: note: first non-comment line is here [FirstNonCommentLine]
  // CHECK:STDERR: class C {
  // CHECK:STDERR: ^~~~~
  // CHECK:STDERR:
  impl package P;
}

fn F() {
  // This reaches decl parsing because `impl` is a valid modifier or introducer.
  // CHECK:STDERR: fail_in_nested_scope.carbon:[[@LINE+7]]:3: error: the `package` declaration must be the first non-comment line [PackageTooLate]
  // CHECK:STDERR:   impl package P;
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR: fail_in_nested_scope.carbon:[[@LINE-41]]:1: note: first non-comment line is here [FirstNonCommentLine]
  // CHECK:STDERR: class C {
  // CHECK:STDERR: ^~~~~
  // CHECK:STDERR:
  impl package P;
}

// CHECK:STDOUT: - filename: fail_in_nested_scope.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─ClassIntroducer 'class'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'C'
// CHECK:STDOUT:   │ ╭─ClassDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─PackageIntroducer 'package'
// CHECK:STDOUT:   │ ├─PackageDecl ';' ❌
// CHECK:STDOUT:   ├─ClassDefinition '}'
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─PackageExpr 'package'
// CHECK:STDOUT:   │ ├─ExprStatement ';' ❌
// CHECK:STDOUT:   │ │   ╭─PackageExpr 'package'
// CHECK:STDOUT:   │ │   ├─IdentifierNameNotBeforeSignature 'P'
// CHECK:STDOUT:   │ │ ╭─MemberAccessExpr '.'
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   │   ╭─ClassIntroducer 'class'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'C'
// CHECK:STDOUT:   │ ╭─ClassDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─PackageIntroducer 'package'
// CHECK:STDOUT:   │ │ ├─ImplModifier 'impl'
// CHECK:STDOUT:   │ ├─PackageDecl ';' ❌
// CHECK:STDOUT:   ├─ClassDefinition '}'
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─PackageIntroducer 'package'
// CHECK:STDOUT:   │ │ ├─ImplModifier 'impl'
// CHECK:STDOUT:   │ ├─PackageDecl ';' ❌
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)

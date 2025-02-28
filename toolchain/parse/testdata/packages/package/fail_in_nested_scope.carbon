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
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'ClassIntroducer', text: 'class'},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: 'C'},
// CHECK:STDOUT:       {kind: 'ClassDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'PackageIntroducer', text: 'package'},
// CHECK:STDOUT:       {kind: 'PackageDecl', text: ';', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:     {kind: 'ClassDefinition', text: '}', subtree_size: 6},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'F'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'PackageExpr', text: 'package'},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:           {kind: 'PackageExpr', text: 'package'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'P'},
// CHECK:STDOUT:         {kind: 'MemberAccessExpr', text: '.', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 4},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 12},
// CHECK:STDOUT:         {kind: 'ClassIntroducer', text: 'class'},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: 'C'},
// CHECK:STDOUT:       {kind: 'ClassDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'PackageIntroducer', text: 'package'},
// CHECK:STDOUT:         {kind: 'ImplModifier', text: 'impl'},
// CHECK:STDOUT:       {kind: 'PackageDecl', text: ';', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:     {kind: 'ClassDefinition', text: '}', subtree_size: 7},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'F'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'PackageIntroducer', text: 'package'},
// CHECK:STDOUT:         {kind: 'ImplModifier', text: 'impl'},
// CHECK:STDOUT:       {kind: 'PackageDecl', text: ';', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 9},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]

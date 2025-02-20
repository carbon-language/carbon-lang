// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/packages/library/fail_in_nested_scope.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/packages/library/fail_in_nested_scope.carbon

// api.carbon

class C {
  // CHECK:STDERR: fail_in_nested_scope.carbon:[[@LINE+7]]:3: error: the `library` declaration must be the first non-comment line [PackageTooLate]
  // CHECK:STDERR:   library "foo";
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR: fail_in_nested_scope.carbon:[[@LINE-4]]:1: note: first non-comment line is here [FirstNonCommentLine]
  // CHECK:STDERR: class C {
  // CHECK:STDERR: ^~~~~
  // CHECK:STDERR:
  library "foo";
}

fn F() {
  // CHECK:STDERR: fail_in_nested_scope.carbon:[[@LINE+7]]:3: error: the `library` declaration must be the first non-comment line [PackageTooLate]
  // CHECK:STDERR:   library "foo";
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR: fail_in_nested_scope.carbon:[[@LINE-15]]:1: note: first non-comment line is here [FirstNonCommentLine]
  // CHECK:STDERR: class C {
  // CHECK:STDERR: ^~~~~
  // CHECK:STDERR:
  library "foo";
}

// impl.carbon

class C {
  // CHECK:STDERR: fail_in_nested_scope.carbon:[[@LINE+7]]:3: error: the `library` declaration must be the first non-comment line [PackageTooLate]
  // CHECK:STDERR:   impl library "foo";
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR: fail_in_nested_scope.carbon:[[@LINE-28]]:1: note: first non-comment line is here [FirstNonCommentLine]
  // CHECK:STDERR: class C {
  // CHECK:STDERR: ^~~~~
  // CHECK:STDERR:
  impl library "foo";
}

fn F() {
  // CHECK:STDERR: fail_in_nested_scope.carbon:[[@LINE+7]]:3: error: the `library` declaration must be the first non-comment line [PackageTooLate]
  // CHECK:STDERR:   impl library "foo";
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR: fail_in_nested_scope.carbon:[[@LINE-39]]:1: note: first non-comment line is here [FirstNonCommentLine]
  // CHECK:STDERR: class C {
  // CHECK:STDERR: ^~~~~
  // CHECK:STDERR:
  impl library "foo";
}

// CHECK:STDOUT: - filename: fail_in_nested_scope.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'ClassIntroducer', text: 'class'},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: 'C'},
// CHECK:STDOUT:       {kind: 'ClassDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'LibraryIntroducer', text: 'library'},
// CHECK:STDOUT:       {kind: 'LibraryDecl', text: ';', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:     {kind: 'ClassDefinition', text: '}', subtree_size: 6},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'F'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'LibraryIntroducer', text: 'library'},
// CHECK:STDOUT:       {kind: 'LibraryDecl', text: ';', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 8},
// CHECK:STDOUT:         {kind: 'ClassIntroducer', text: 'class'},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: 'C'},
// CHECK:STDOUT:       {kind: 'ClassDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'LibraryIntroducer', text: 'library'},
// CHECK:STDOUT:         {kind: 'ImplModifier', text: 'impl'},
// CHECK:STDOUT:       {kind: 'LibraryDecl', text: ';', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:     {kind: 'ClassDefinition', text: '}', subtree_size: 7},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'F'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'LibraryIntroducer', text: 'library'},
// CHECK:STDOUT:         {kind: 'ImplModifier', text: 'impl'},
// CHECK:STDOUT:       {kind: 'LibraryDecl', text: ';', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 9},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]

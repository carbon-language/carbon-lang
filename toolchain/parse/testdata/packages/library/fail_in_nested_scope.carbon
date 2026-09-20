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
  // CHECK:STDERR: fail_in_nested_scope.carbon:[[@LINE+3]]:3-9: error: the `library` declaration must be the first non-comment line [PackageTooLate]
  // CHECK:STDERR: fail_in_nested_scope.carbon:[[@LINE-2]]:1-5: note: first non-comment line is here [FirstNonCommentLine]
  // CHECK:STDERR:
  library "foo";
}

fn F() {
  // CHECK:STDERR: fail_in_nested_scope.carbon:[[@LINE+3]]:3-9: error: the `library` declaration must be the first non-comment line [PackageTooLate]
  // CHECK:STDERR: fail_in_nested_scope.carbon:[[@LINE-9]]:1-5: note: first non-comment line is here [FirstNonCommentLine]
  // CHECK:STDERR:
  library "foo";
}

// impl.carbon

class C {
  // CHECK:STDERR: fail_in_nested_scope.carbon:[[@LINE+3]]:3-6: error: the `library` declaration must be the first non-comment line [PackageTooLate]
  // CHECK:STDERR: fail_in_nested_scope.carbon:[[@LINE-18]]:1-5: note: first non-comment line is here [FirstNonCommentLine]
  // CHECK:STDERR:
  impl library "foo";
}

fn F() {
  // CHECK:STDERR: fail_in_nested_scope.carbon:[[@LINE+3]]:3-6: error: the `library` declaration must be the first non-comment line [PackageTooLate]
  // CHECK:STDERR: fail_in_nested_scope.carbon:[[@LINE-25]]:1-5: note: first non-comment line is here [FirstNonCommentLine]
  // CHECK:STDERR:
  impl library "foo";
}

// CHECK:STDOUT: - filename: fail_in_nested_scope.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─ClassIntroducer 'class'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'C'
// CHECK:STDOUT:   │ ╭─ClassDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─LibraryIntroducer 'library'
// CHECK:STDOUT:   │ ├─LibraryDecl ';' has_error
// CHECK:STDOUT:   ├─ClassDefinition '}'
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─LibraryIntroducer 'library'
// CHECK:STDOUT:   │ ├─LibraryDecl ';' has_error
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   │   ╭─ClassIntroducer 'class'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'C'
// CHECK:STDOUT:   │ ╭─ClassDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─LibraryIntroducer 'library'
// CHECK:STDOUT:   │ │ ├─ImplModifier 'impl'
// CHECK:STDOUT:   │ ├─LibraryDecl ';' has_error
// CHECK:STDOUT:   ├─ClassDefinition '}'
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─LibraryIntroducer 'library'
// CHECK:STDOUT:   │ │ ├─ImplModifier 'impl'
// CHECK:STDOUT:   │ ├─LibraryDecl ';' has_error
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)

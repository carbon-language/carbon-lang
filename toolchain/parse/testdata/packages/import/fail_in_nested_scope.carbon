// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/packages/import/fail_in_nested_scope.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/packages/import/fail_in_nested_scope.carbon

class A {
  // CHECK:STDERR: fail_in_nested_scope.carbon:[[@LINE+3]]:3-8: error: `import` declarations must come after the `package` declaration (if present) and before any other entities in the file [ImportTooLate]
  // CHECK:STDERR: fail_in_nested_scope.carbon:[[@LINE-2]]:1-5: note: first declaration is here [FirstDecl]
  // CHECK:STDERR:
  import B;
}

fn F() {
  // CHECK:STDERR: fail_in_nested_scope.carbon:[[@LINE+3]]:3-8: error: `import` declarations must come after the `package` declaration (if present) and before any other entities in the file [ImportTooLate]
  // CHECK:STDERR: fail_in_nested_scope.carbon:[[@LINE-9]]:1-5: note: first declaration is here [FirstDecl]
  // CHECK:STDERR:
  import C;
}

// CHECK:STDOUT: - filename: fail_in_nested_scope.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─ClassIntroducer 'class'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'A'
// CHECK:STDOUT:   │ ╭─ClassDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─ImportIntroducer 'import'
// CHECK:STDOUT:   │ ├─ImportDecl ';' has_error
// CHECK:STDOUT:   ├─ClassDefinition '}'
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─ImportIntroducer 'import'
// CHECK:STDOUT:   │ ├─ImportDecl ';' has_error
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)

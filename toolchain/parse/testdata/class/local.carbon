// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/class/local.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/class/local.carbon

fn F() {
  class C {
    fn G() {}
    fn H();
  }
  fn C.H() {}
}

// CHECK:STDOUT: - filename: local.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │   ╭─ClassIntroducer 'class'
// CHECK:STDOUT:   │ │   ├─IdentifierNameNotBeforeSignature 'C'
// CHECK:STDOUT:   │ │ ╭─ClassDefinitionStart '{'
// CHECK:STDOUT:   │ │ │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ │ │   ├─IdentifierNameMaybeBeforeSignature 'G'
// CHECK:STDOUT:   │ │ │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ │ │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ │ │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │ ├─FunctionDefinition '}'
// CHECK:STDOUT:   │ │ │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ │ │ ├─IdentifierNameMaybeBeforeSignature 'H'
// CHECK:STDOUT:   │ │ │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ │ │ ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ │ ├─FunctionDecl ';'
// CHECK:STDOUT:   │ ├─ClassDefinition '}'
// CHECK:STDOUT:   │ │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ │   │ ╭─IdentifierNameNotBeforeSignature 'C'
// CHECK:STDOUT:   │ │   ├─IdentifierNameQualifierWithoutParams '.'
// CHECK:STDOUT:   │ │   ├─IdentifierNameMaybeBeforeSignature 'H'
// CHECK:STDOUT:   │ │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)

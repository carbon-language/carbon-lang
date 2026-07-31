// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/choice/local.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/choice/local.carbon

fn F() {
  choice C {
    X,
    Y
  }
}

// CHECK:STDOUT: - filename: local.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │   ╭─ChoiceIntroducer 'choice'
// CHECK:STDOUT:   │ │   ├─IdentifierNameNotBeforeSignature 'C'
// CHECK:STDOUT:   │ │ ╭─ChoiceDefinitionStart '{'
// CHECK:STDOUT:   │ │ ├─IdentifierNameNotBeforeSignature 'X'
// CHECK:STDOUT:   │ │ ├─ChoiceAlternativeListComma ','
// CHECK:STDOUT:   │ │ ├─IdentifierNameNotBeforeSignature 'Y'
// CHECK:STDOUT:   │ ├─ChoiceDefinition '}'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)

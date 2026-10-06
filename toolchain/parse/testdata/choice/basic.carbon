// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/choice/basic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/choice/basic.carbon

choice Ordering {
  Less,
  Equivalent,
  Greater,
  Incomparable
}

// CHECK:STDOUT: - filename: basic.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─ChoiceIntroducer 'choice'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'Ordering'
// CHECK:STDOUT:   │ ╭─ChoiceDefinitionStart '{'
// CHECK:STDOUT:   │ ├─IdentifierNameNotBeforeSignature 'Less'
// CHECK:STDOUT:   │ ├─ChoiceAlternativeListComma ','
// CHECK:STDOUT:   │ ├─IdentifierNameNotBeforeSignature 'Equivalent'
// CHECK:STDOUT:   │ ├─ChoiceAlternativeListComma ','
// CHECK:STDOUT:   │ ├─IdentifierNameNotBeforeSignature 'Greater'
// CHECK:STDOUT:   │ ├─ChoiceAlternativeListComma ','
// CHECK:STDOUT:   │ ├─IdentifierNameNotBeforeSignature 'Incomparable'
// CHECK:STDOUT:   ├─ChoiceDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)

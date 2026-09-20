// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/choice/fail_invalid_alternative_identifier.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/choice/fail_invalid_alternative_identifier.carbon

choice InvalidAlternativeIdentifier {
    // CHECK:STDERR: fail_invalid_alternative_identifier.carbon:[[@LINE+2]]:5: error: expected choice alternative name [ExpectedChoiceAlternativeName]
    // CHECK:STDERR:
    ,Some,
    // CHECK:STDERR: fail_invalid_alternative_identifier.carbon:[[@LINE+2]]:5-6: error: expected choice alternative name [ExpectedChoiceAlternativeName]
    // CHECK:STDERR:
    42,
    Other
}

// CHECK:STDOUT: - filename: fail_invalid_alternative_identifier.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─ChoiceIntroducer 'choice'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'InvalidAlternativeIdentifier'
// CHECK:STDOUT:   │ ╭─ChoiceDefinitionStart '{'
// CHECK:STDOUT:   │ ├─IdentifierNameNotBeforeSignature 'Other'
// CHECK:STDOUT:   ├─ChoiceDefinition '}' has_error
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)

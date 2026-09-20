// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/choice/fail_missing_definition_parameterized.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/choice/fail_missing_definition_parameterized.carbon

// CHECK:STDERR: fail_missing_definition_parameterized.carbon:[[@LINE+2]]:34: error: choice definition expected [ExpectedChoiceDefinition]
// CHECK:STDERR:
choice MissingDefinition(T: type);

// CHECK:STDOUT: - filename: fail_missing_definition_parameterized.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─ChoiceIntroducer 'choice'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'MissingDefinition'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   │ │ ╭─IdentifierNameNotBeforeSignature 'T'
// CHECK:STDOUT:   │   │ │ ├─CompileTimeBindingPatternTypeStart ':'
// CHECK:STDOUT:   │   │ │ ├─TypeTypeLiteral 'type'
// CHECK:STDOUT:   │   │ ├─CompileTimeBindingPattern ':'
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─ChoiceDefinitionStart ';' has_error
// CHECK:STDOUT:   ├─ChoiceDefinition ';' has_error
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)

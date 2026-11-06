// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/choice/fail_keyword_alternative_name.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/choice/fail_keyword_alternative_name.carbon

choice C {
  // CHECK:STDERR:        ╭─┤ fail_keyword_alternative_name.carbon:[[@LINE+8]]:3
  // CHECK:STDERR: ->     │   class,
  // CHECK:STDERR:        ·   ┳━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: expected choice alternative name; prefix reserved word with `r#`
  // CHECK:STDERR:        ·      ┃        to form a valid identifier
  // CHECK:STDERR:        ·      ┃        [ExpectedChoiceAlternativeName]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  class,
  Other
}

// CHECK:STDOUT: - filename: fail_keyword_alternative_name.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─ChoiceIntroducer 'choice'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'C'
// CHECK:STDOUT:   │ ╭─ChoiceDefinitionStart '{'
// CHECK:STDOUT:   │ ├─IdentifierNameNotBeforeSignature 'class'
// CHECK:STDOUT:   │ ├─ChoiceAlternativeListComma ','
// CHECK:STDOUT:   │ ├─IdentifierNameNotBeforeSignature 'Other'
// CHECK:STDOUT:   ├─ChoiceDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)

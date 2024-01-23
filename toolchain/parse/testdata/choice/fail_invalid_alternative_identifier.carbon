// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

choice InvalidAlternativeIdentifier {
    // CHECK:STDERR: fail_invalid_alternative_identifier.carbon:[[@LINE+3]]:5: ERROR: Expected choice alternative name.
    // CHECK:STDERR:     ,Some,
    // CHECK:STDERR:     ^
    ,Some,
    // CHECK:STDERR: fail_invalid_alternative_identifier.carbon:[[@LINE+3]]:5: ERROR: Expected choice alternative name.
    // CHECK:STDERR:     42,
    // CHECK:STDERR:     ^~
    42,
    Other
}

// CHECK:STDOUT: - filename: fail_invalid_alternative_identifier.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'ChoiceIntroducer', text: 'choice'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'InvalidAlternativeIdentifier'},
// CHECK:STDOUT:       {kind: 'ChoiceDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'IdentifierName', text: 'Other'},
// CHECK:STDOUT:     {kind: 'ChoiceDefinition', text: '}', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]

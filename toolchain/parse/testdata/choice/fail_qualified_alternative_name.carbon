// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

choice X {
  // CHECK:STDERR: fail_qualified_alternative_name.carbon:[[@LINE+3]]:4: ERROR: Expected `,` or `}`.
  // CHECK:STDERR:   A.B
  // CHECK:STDERR:    ^
  A.B
}

// CHECK:STDOUT: - filename: fail_qualified_alternative_name.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'ChoiceIntroducer', text: 'choice'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'X'},
// CHECK:STDOUT:       {kind: 'ChoiceDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'IdentifierName', text: 'A'},
// CHECK:STDOUT:     {kind: 'ChoiceDefinition', text: '}', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]

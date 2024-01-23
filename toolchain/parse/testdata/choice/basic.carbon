// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

choice Ordering {
  Less,
  Equivalent,
  Greater,
  Incomparable
}

// CHECK:STDOUT: - filename: basic.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'ChoiceIntroducer', text: 'choice'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'Ordering'},
// CHECK:STDOUT:       {kind: 'ChoiceDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'IdentifierName', text: 'Less'},
// CHECK:STDOUT:       {kind: 'ChoiceAlternativeListComma', text: ','},
// CHECK:STDOUT:       {kind: 'IdentifierName', text: 'Equivalent'},
// CHECK:STDOUT:       {kind: 'ChoiceAlternativeListComma', text: ','},
// CHECK:STDOUT:       {kind: 'IdentifierName', text: 'Greater'},
// CHECK:STDOUT:       {kind: 'ChoiceAlternativeListComma', text: ','},
// CHECK:STDOUT:       {kind: 'IdentifierName', text: 'Incomparable'},
// CHECK:STDOUT:     {kind: 'ChoiceDefinition', text: '}', subtree_size: 11},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]

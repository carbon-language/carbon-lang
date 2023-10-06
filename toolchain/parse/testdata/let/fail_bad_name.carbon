// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_bad_name.carbon:[[@LINE+3]]:5: ERROR: Expected pattern in `let` declaration.
// CHECK:STDERR: let ? = 4;
// CHECK:STDERR:     ^
let ? = 4;

// CHECK:STDOUT: - filename: fail_bad_name.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:         {kind: 'Name', text: '?', has_error: yes},
// CHECK:STDOUT:         {kind: 'InvalidParse', text: '?', has_error: yes},
// CHECK:STDOUT:       {kind: 'PatternBinding', text: '?', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:       {kind: 'LetInitializer', text: '='},
// CHECK:STDOUT:       {kind: 'Literal', text: '4'},
// CHECK:STDOUT:     {kind: 'LetDeclaration', text: ';', subtree_size: 7},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]

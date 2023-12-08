// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

let

// CHECK:STDERR: fail_no_semi.carbon:[[@LINE+15]]:21: ERROR: Expected binding pattern.
// CHECK:STDERR: // CHECK:STDOUT:   ]
// CHECK:STDERR:                     ^
// CHECK:STDERR: fail_no_semi.carbon:[[@LINE+12]]:21: ERROR: `let` declarations must end with a `;`.
// CHECK:STDERR: // CHECK:STDOUT:   ]
// CHECK:STDERR:                     ^
// CHECK:STDOUT: - filename: fail_no_semi.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: '', has_error: yes},
// CHECK:STDOUT:         {kind: 'InvalidParse', text: '', has_error: yes},
// CHECK:STDOUT:       {kind: 'BindingPattern', text: '', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:     {kind: 'LetDecl', text: 'let', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]

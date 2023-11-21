// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_without_name_and_many_tokens_in_params.carbon:[[@LINE+3]]:4: ERROR: `fn` introducer should be followed by a name.
// CHECK:STDERR: fn (a tokens c d e f g h i j k l m n o p q r s t u v w x y z);
// CHECK:STDERR:    ^
fn (a tokens c d e f g h i j k l m n o p q r s t u v w x y z);

// CHECK:STDOUT: - filename: fail_without_name_and_many_tokens_in_params.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'InvalidParse', text: '(', has_error: yes},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: ';', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]

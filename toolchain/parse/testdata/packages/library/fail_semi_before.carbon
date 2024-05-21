// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

;

// CHECK:STDERR: fail_semi_before.carbon:[[@LINE+6]]:1: ERROR: The `library` declaration must be the first non-comment line.
// CHECK:STDERR: library "foo";
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR: fail_semi_before.carbon:[[@LINE-5]]:1: First non-comment line is here.
// CHECK:STDERR: ;
// CHECK:STDERR: ^
library "foo";

// CHECK:STDOUT: - filename: fail_semi_before.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:     {kind: 'EmptyDecl', text: ';'},
// CHECK:STDOUT:       {kind: 'LibraryIntroducer', text: 'library'},
// CHECK:STDOUT:     {kind: 'LibraryDecl', text: ';', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]

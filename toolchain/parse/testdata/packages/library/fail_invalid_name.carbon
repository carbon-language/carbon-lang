// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- fail_identifier.carbon

// CHECK:STDERR: fail_identifier.carbon:[[@LINE+4]]:9: ERROR: Expected `default` or a string literal to specify the library name.
// CHECK:STDERR: library Shapes api;
// CHECK:STDERR:         ^~~~~~
// CHECK:STDERR:
library Shapes api;

// --- fail_raw_identifier.carbon

// CHECK:STDERR: fail_raw_identifier.carbon:[[@LINE+4]]:9: ERROR: Expected `default` or a string literal to specify the library name.
// CHECK:STDERR: library r#default api;
// CHECK:STDERR:         ^~~~~~~
// CHECK:STDERR:
library r#default api;

// --- fail_missing.carbon

// CHECK:STDERR: fail_missing.carbon:[[@LINE+4]]:8: ERROR: Expected `default` or a string literal to specify the library name.
// CHECK:STDERR: library;
// CHECK:STDERR:        ^
// CHECK:STDERR:
library;

// --- fail_missing_with_api.carbon

// CHECK:STDERR: fail_missing_with_api.carbon:[[@LINE+4]]:9: ERROR: Expected `default` or a string literal to specify the library name.
// CHECK:STDERR: library api;
// CHECK:STDERR:         ^~~
// CHECK:STDERR:
library api;

// --- fail_no_semi.carbon

library "NoSemi" api

// CHECK:STDERR: fail_no_semi.carbon:[[@LINE+39]]:21: ERROR: `library` declarations must end with a `;`.
// CHECK:STDERR: // CHECK:STDOUT:   ]
// CHECK:STDERR:                     ^
// CHECK:STDOUT: - filename: fail_identifier.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'LibraryIntroducer', text: 'library'},
// CHECK:STDOUT:     {kind: 'LibraryDirective', text: ';', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_raw_identifier.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'LibraryIntroducer', text: 'library'},
// CHECK:STDOUT:     {kind: 'LibraryDirective', text: ';', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_missing.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'LibraryIntroducer', text: 'library'},
// CHECK:STDOUT:     {kind: 'LibraryDirective', text: ';', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_missing_with_api.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'LibraryIntroducer', text: 'library'},
// CHECK:STDOUT:     {kind: 'LibraryDirective', text: ';', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_no_semi.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'LibraryIntroducer', text: 'library'},
// CHECK:STDOUT:       {kind: 'LibraryName', text: '"NoSemi"'},
// CHECK:STDOUT:       {kind: 'PackageApi', text: 'api'},
// CHECK:STDOUT:     {kind: 'LibraryDirective', text: 'api', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]

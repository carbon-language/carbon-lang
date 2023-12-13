// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- identifier.carbon

// CHECK:STDERR: identifier.carbon:[[@LINE+3]]:9: ERROR: Expected `default` or a string literal to specify the library name.
// CHECK:STDERR: library Shapes api;
// CHECK:STDERR:         ^~~~~~
library Shapes api;

// --- raw_identifier.carbon

// CHECK:STDERR: raw_identifier.carbon:[[@LINE+3]]:9: ERROR: Expected `default` or a string literal to specify the library name.
// CHECK:STDERR: library r#default api;
// CHECK:STDERR:         ^~~~~~~
library r#default api;

// --- missing.carbon

// CHECK:STDERR: missing.carbon:[[@LINE+3]]:8: ERROR: Expected `default` or a string literal to specify the library name.
// CHECK:STDERR: library;
// CHECK:STDERR:        ^
library;

// --- missing_with_api.carbon

// CHECK:STDERR: missing_with_api.carbon:[[@LINE+3]]:9: ERROR: Expected `default` or a string literal to specify the library name.
// CHECK:STDERR: library api;
// CHECK:STDERR:         ^~~
library api;

// --- no_semi.carbon

library "NoSemi" api

// CHECK:STDERR: no_semi.carbon:[[@LINE+39]]:21: ERROR: `library` declarations must end with a `;`.
// CHECK:STDERR: // CHECK:STDOUT:   ]
// CHECK:STDERR:                     ^
// CHECK:STDOUT: - filename: identifier.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'LibraryIntroducer', text: 'library'},
// CHECK:STDOUT:     {kind: 'LibraryDirective', text: ';', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: raw_identifier.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'LibraryIntroducer', text: 'library'},
// CHECK:STDOUT:     {kind: 'LibraryDirective', text: ';', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: missing.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'LibraryIntroducer', text: 'library'},
// CHECK:STDOUT:     {kind: 'LibraryDirective', text: ';', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: missing_with_api.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'LibraryIntroducer', text: 'library'},
// CHECK:STDOUT:     {kind: 'LibraryDirective', text: ';', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: no_semi.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'LibraryIntroducer', text: 'library'},
// CHECK:STDOUT:       {kind: 'LibraryName', text: '"NoSemi"'},
// CHECK:STDOUT:       {kind: 'PackageApi', text: 'api'},
// CHECK:STDOUT:     {kind: 'LibraryDirective', text: 'api', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_incomplete_name.carbon:[[@LINE+3]]:15: ERROR: Expected identifier after `.`.
// CHECK:STDERR: namespace Foo.;
// CHECK:STDERR:               ^
namespace Foo.;

// CHECK:STDOUT: - filename: fail_incomplete_name.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'NamespaceStart', text: 'namespace'},
// CHECK:STDOUT:         {kind: 'NameExpression', text: 'Foo'},
// CHECK:STDOUT:         {kind: 'Name', text: ';', has_error: yes},
// CHECK:STDOUT:       {kind: 'QualifiedDeclaration', text: '.', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'Namespace', text: ';', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]

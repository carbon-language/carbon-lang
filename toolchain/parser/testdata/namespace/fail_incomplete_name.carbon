// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: [
// CHECK:STDOUT:   {kind: 'NamespaceStart', text: 'namespace'},
// CHECK:STDOUT:     {kind: 'NameExpression', text: 'Foo'},
// CHECK:STDOUT:     {kind: 'Name', text: ';', has_error: yes},
// CHECK:STDOUT:   {kind: 'QualifiedDeclaration', text: '.', subtree_size: 3},
// CHECK:STDOUT: {kind: 'Namespace', text: ';', subtree_size: 5},
// CHECK:STDOUT: {kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

// CHECK:STDERR: fail_incomplete_name.carbon:[[@LINE+1]]:15: Expected identifier after `.`.
namespace Foo.;

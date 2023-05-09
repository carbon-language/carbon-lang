// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: [
// CHECK:STDOUT:   {kind: 'InterfaceIntroducer', text: 'interface'},
// CHECK:STDOUT: {kind: 'InterfaceDeclaration', text: 'interface', has_error: yes, subtree_size: 2},
// CHECK:STDOUT: {kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

// CHECK:STDERR: toolchain/parser/testdata/generics/interface/fail_missing_name.carbon:[[@LINE+1]]:11: `interface` introducer should be followed by a name.
interface {
}

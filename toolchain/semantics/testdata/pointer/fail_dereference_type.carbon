// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_dereference_type.carbon:[[@LINE+6]]:8: Cannot dereference operand of non-pointer type `type`.
// CHECK:STDERR: var p: *i32;
// CHECK:STDERR:        ^
// CHECK:STDERR: fail_dereference_type.carbon:[[@LINE+3]]:8: To form a pointer type, write the `*` after the pointee type.
// CHECK:STDERR: var p: *i32;
// CHECK:STDERR:        ^
var p: *i32;

// CHECK:STDOUT: cross_reference_irs_size: 1
// CHECK:STDOUT: functions: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: integer_literals: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: real_literals: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: strings: [
// CHECK:STDOUT:   p,
// CHECK:STDOUT: ]
// CHECK:STDOUT: types: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: type_blocks: [
// CHECK:STDOUT: ]
// CHECK:STDOUT: nodes: [
// CHECK:STDOUT:   {kind: Dereference, arg0: nodeIntegerType, type: typeError},
// CHECK:STDOUT:   {kind: VarStorage, type: typeError},
// CHECK:STDOUT:   {kind: BindName, arg0: str0, arg1: node+1, type: typeError},
// CHECK:STDOUT: ]
// CHECK:STDOUT: node_blocks: [
// CHECK:STDOUT:   [
// CHECK:STDOUT:   ],
// CHECK:STDOUT:   [
// CHECK:STDOUT:     node+0,
// CHECK:STDOUT:     node+1,
// CHECK:STDOUT:     node+2,
// CHECK:STDOUT:   ],
// CHECK:STDOUT: ]

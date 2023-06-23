// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: ; ModuleID = 'empty_struct.carbon'
// CHECK:STDOUT: source_filename = "empty_struct.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: %EmptyTupleType = type {}
// CHECK:STDOUT: %StructLiteralType = type {}
// CHECK:STDOUT:
// CHECK:STDOUT: define %EmptyTupleType @Echo(%StructLiteralType %a) {
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }

fn Echo(a: {}) {
}

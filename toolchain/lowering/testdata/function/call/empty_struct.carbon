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
// CHECK:STDOUT: entry:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define %EmptyTupleType @Main() {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %StructLiteralValue = alloca %StructLiteralType, align 8
// CHECK:STDOUT:   %0 = load %StructLiteralType, ptr %StructLiteralValue, align 1
// CHECK:STDOUT:   %Echo = call %EmptyTupleType @Echo(%StructLiteralType %0)
// CHECK:STDOUT: }

fn Echo(a: {}) {
  // TODO: Set return type to `{}` and return `a` back.
}

fn Main() {
  // TODO: Store the returned `{}`.
  Echo({});
}

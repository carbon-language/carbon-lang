// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn I32() -> type {
  return i32;
}

fn F64() -> type {
  return f64;
}

// CHECK:STDOUT: ; ModuleID = 'type_values.carbon'
// CHECK:STDOUT: source_filename = "type_values.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define void @I32() {
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @F64() {
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Int() -> type = "int.make_32";
fn Float(size: i32) -> type = "float.make";
fn Bool() -> type = "bool.make";

fn F() {
  var i: Int() = 0;
  var f: Float(64) = 0.0;
  var b: Bool() = false;
}

// CHECK:STDOUT: ; ModuleID = 'types.carbon'
// CHECK:STDOUT: source_filename = "types.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define void @F() {
// CHECK:STDOUT:   %i = alloca i32, align 4
// CHECK:STDOUT:   store i32 0, ptr %i, align 4
// CHECK:STDOUT:   %f = alloca double, align 8
// CHECK:STDOUT:   store double 0.000000e+00, ptr %f, align 8
// CHECK:STDOUT:   %b = alloca i1, align 1
// CHECK:STDOUT:   store i1 false, ptr %b, align 1
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }

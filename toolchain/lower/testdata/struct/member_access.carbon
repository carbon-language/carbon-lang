// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Run() -> i32 {
  var x: {.a: f64, .b: i32} = {.a = 0.0, .b = 1};
  var y: i32 = x.b;
  var z: i32 = y;
  return 0;
}

// CHECK:STDOUT: ; ModuleID = 'member_access.carbon'
// CHECK:STDOUT: source_filename = "member_access.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @main() {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x.var = alloca { double, i32 }, align 8
// CHECK:STDOUT:   %.loc8_48.2.a = getelementptr inbounds { double, i32 }, ptr %x.var, i32 0, i32 0
// CHECK:STDOUT:   store double 0.000000e+00, ptr %.loc8_48.2.a, align 8
// CHECK:STDOUT:   %.loc8_48.4.b = getelementptr inbounds { double, i32 }, ptr %x.var, i32 0, i32 1
// CHECK:STDOUT:   store i32 1, ptr %.loc8_48.4.b, align 4
// CHECK:STDOUT:   %y.var = alloca i32, align 4
// CHECK:STDOUT:   %.loc9_17.1.b = getelementptr inbounds { double, i32 }, ptr %x.var, i32 0, i32 1
// CHECK:STDOUT:   %.loc9_17.2 = load i32, ptr %.loc9_17.1.b, align 4
// CHECK:STDOUT:   store i32 %.loc9_17.2, ptr %y.var, align 4
// CHECK:STDOUT:   %z.var = alloca i32, align 4
// CHECK:STDOUT:   %.loc10 = load i32, ptr %y.var, align 4
// CHECK:STDOUT:   store i32 %.loc10, ptr %z.var, align 4
// CHECK:STDOUT:   ret i32 0
// CHECK:STDOUT: }

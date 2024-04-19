// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Run() -> i32 {
  var x: {.a: i32, .b: i32} = {.a = 1, .b = 2};
  var y: {.a: i32, .b: i32} = x;
  return 0;
}

// CHECK:STDOUT: ; ModuleID = 'two_entries.carbon'
// CHECK:STDOUT: source_filename = "two_entries.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @main() {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x.var = alloca { i32, i32 }, align 8
// CHECK:STDOUT:   %.loc8_46.2.a = getelementptr inbounds { i32, i32 }, ptr %x.var, i32 0, i32 0
// CHECK:STDOUT:   store i32 1, ptr %.loc8_46.2.a, align 4
// CHECK:STDOUT:   %.loc8_46.4.b = getelementptr inbounds { i32, i32 }, ptr %x.var, i32 0, i32 1
// CHECK:STDOUT:   store i32 2, ptr %.loc8_46.4.b, align 4
// CHECK:STDOUT:   %y.var = alloca { i32, i32 }, align 8
// CHECK:STDOUT:   %.loc9_31.1.a = getelementptr inbounds { i32, i32 }, ptr %x.var, i32 0, i32 0
// CHECK:STDOUT:   %.loc9_31.2 = load i32, ptr %.loc9_31.1.a, align 4
// CHECK:STDOUT:   %.loc9_31.3.a = getelementptr inbounds { i32, i32 }, ptr %y.var, i32 0, i32 0
// CHECK:STDOUT:   store i32 %.loc9_31.2, ptr %.loc9_31.3.a, align 4
// CHECK:STDOUT:   %.loc9_31.5.b = getelementptr inbounds { i32, i32 }, ptr %x.var, i32 0, i32 1
// CHECK:STDOUT:   %.loc9_31.6 = load i32, ptr %.loc9_31.5.b, align 4
// CHECK:STDOUT:   %.loc9_31.7.b = getelementptr inbounds { i32, i32 }, ptr %y.var, i32 0, i32 1
// CHECK:STDOUT:   store i32 %.loc9_31.6, ptr %.loc9_31.7.b, align 4
// CHECK:STDOUT:   ret i32 0
// CHECK:STDOUT: }

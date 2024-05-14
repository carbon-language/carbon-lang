// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Run() -> i32 {
  var x: {.a: i32} = {.a = 4};
  var y: {.a: i32} = x;
  return 0;
}

// CHECK:STDOUT: ; ModuleID = 'one_entry.carbon'
// CHECK:STDOUT: source_filename = "one_entry.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @main() {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x.var = alloca { i32 }, align 8
// CHECK:STDOUT:   store { i32 } { i32 4 }, ptr %x.var, align 4
// CHECK:STDOUT:   %y.var = alloca { i32 }, align 8
// CHECK:STDOUT:   %.loc9_22.1.a = getelementptr inbounds { i32 }, ptr %x.var, i32 0, i32 0
// CHECK:STDOUT:   %.loc9_22.2 = load i32, ptr %.loc9_22.1.a, align 4
// CHECK:STDOUT:   %.loc9_22.3.struct.init = insertvalue { i32 } poison, i32 %.loc9_22.2, 0
// CHECK:STDOUT:   store { i32 } %.loc9_22.3.struct.init, ptr %y.var, align 4
// CHECK:STDOUT:   ret i32 0
// CHECK:STDOUT: }

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() -> i32 {
  var a: i32 = 0;
  alias b = a;
  return b;
}

// CHECK:STDOUT: ; ModuleID = 'local.carbon'
// CHECK:STDOUT: source_filename = "local.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @F() {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca i32, align 4
// CHECK:STDOUT:   store i32 0, ptr %a.var, align 4
// CHECK:STDOUT:   %.loc10 = load i32, ptr %a.var, align 4
// CHECK:STDOUT:   ret i32 %.loc10
// CHECK:STDOUT: }

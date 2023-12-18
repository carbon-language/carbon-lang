// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Run() -> i32 {
  var x: () = ();
  var y: () = x;
  return 0;
}

// CHECK:STDOUT: ; ModuleID = 'empty.carbon'
// CHECK:STDOUT: source_filename = "empty.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @main() {
// CHECK:STDOUT:   %x = alloca {}, align 8
// CHECK:STDOUT:   %y = alloca {}, align 8
// CHECK:STDOUT:   ret i32 0
// CHECK:STDOUT: }

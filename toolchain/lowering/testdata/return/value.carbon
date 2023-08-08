// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Main() -> i32 {
  return 0;
}

// CHECK:STDOUT: ; ModuleID = 'value.carbon'
// CHECK:STDOUT: source_filename = "value.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @Main() {
// CHECK:STDOUT:   ret i32 0
// CHECK:STDOUT: }

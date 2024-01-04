// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F(T:! type) {
  var p: T*;
  let n: T = *p;
}

// CHECK:STDOUT: ; ModuleID = 'type_param.carbon'
// CHECK:STDOUT: source_filename = "type_param.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: %type = type {}
// CHECK:STDOUT:
// CHECK:STDOUT: define void @F(%type %T) {
// CHECK:STDOUT:   %p = alloca ptr, align 8
// CHECK:STDOUT:   %1 = load ptr, ptr %p, align 8
// CHECK:STDOUT:   %2 = load {}, ptr %1, align 1
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }

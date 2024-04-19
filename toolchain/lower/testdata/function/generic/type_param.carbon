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
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %p.var = alloca ptr, align 8
// CHECK:STDOUT:   %.loc9_15 = load ptr, ptr %p.var, align 8
// CHECK:STDOUT:   %.loc9_14.2 = load {}, ptr %.loc9_15, align 1
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }

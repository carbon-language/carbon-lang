// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Echo(a: {}) -> {} {
  return {};
}

fn Main() {
  var b: {} = Echo({});
}

// CHECK:STDOUT: ; ModuleID = 'empty_struct.carbon'
// CHECK:STDOUT: source_filename = "empty_struct.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define {} @Echo({} %a) {
// CHECK:STDOUT:   %struct = alloca {}, align 8
// CHECK:STDOUT:   %1 = load {}, ptr %struct, align 1
// CHECK:STDOUT:   ret {} %1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @Main() {
// CHECK:STDOUT:   %struct = alloca {}, align 8
// CHECK:STDOUT:   %b = alloca {}, align 8
// CHECK:STDOUT:   %struct1 = alloca {}, align 8
// CHECK:STDOUT:   %1 = load {}, ptr %struct1, align 1
// CHECK:STDOUT:   %Echo = call {} @Echo({} %1)
// CHECK:STDOUT:   store {} %Echo, ptr %b, align 1
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }

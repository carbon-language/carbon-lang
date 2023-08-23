// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Echo(a: ()) -> () {
  return a;
}

fn Main() {
  var b: () = Echo(());
}

// CHECK:STDOUT: ; ModuleID = 'empty_tuple.carbon'
// CHECK:STDOUT: source_filename = "empty_tuple.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define {} @Echo({} %a) {
// CHECK:STDOUT:   ret {} %a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @Main() {
// CHECK:STDOUT:   %tuple = alloca {}, align 8
// CHECK:STDOUT:   %b = alloca {}, align 8
// CHECK:STDOUT:   %tuple1 = alloca {}, align 8
// CHECK:STDOUT:   %1 = load {}, ptr %tuple1, align 1
// CHECK:STDOUT:   %Echo = call {} @Echo({} %1)
// CHECK:STDOUT:   store {} %Echo, ptr %b, align 1
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE


fn Foo() {}

fn Bar(a: ()) -> () { return a; }

fn Main() {
  var x: () = Bar(Foo());
}

// CHECK:STDOUT: ; ModuleID = 'implicit_empty_tuple_as_arg.carbon'
// CHECK:STDOUT: source_filename = "implicit_empty_tuple_as_arg.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define void @Foo() {
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define {} @Bar({} %a) {
// CHECK:STDOUT:   ret {} %a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @Main() {
// CHECK:STDOUT:   %tuple = alloca {}, align 8
// CHECK:STDOUT:   %var = alloca {}, align 8
// CHECK:STDOUT:   call void @Foo()
// CHECK:STDOUT:   %call.result = alloca {}, align 8
// CHECK:STDOUT:   %1 = load {}, ptr %call.result, align 1
// CHECK:STDOUT:   %2 = load {}, ptr %call.result, align 1
// CHECK:STDOUT:   %Bar = call {} @Bar({} %2)
// CHECK:STDOUT:   store {} %Bar, ptr %var, align 1
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }

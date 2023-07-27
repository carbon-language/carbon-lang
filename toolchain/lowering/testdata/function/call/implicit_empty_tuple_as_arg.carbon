// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: ; ModuleID = 'implicit_empty_tuple_as_arg.carbon'
// CHECK:STDOUT: source_filename = "implicit_empty_tuple_as_arg.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: %TupleLiteralType = type {}
// CHECK:STDOUT:
// CHECK:STDOUT: define void @Foo() {
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define %TupleLiteralType @Bar(%TupleLiteralType %a) {
// CHECK:STDOUT:   ret %TupleLiteralType %a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @Main() {
// CHECK:STDOUT:   %TupleLiteralValue = alloca %TupleLiteralType, align 8
// CHECK:STDOUT:   %var = alloca %TupleLiteralType, align 8
// CHECK:STDOUT:   call void @Foo()
// CHECK:STDOUT:   %TupleLiteralValue1 = alloca {}, align 8
// CHECK:STDOUT:   %1 = load %TupleLiteralType, ptr %TupleLiteralValue1, align 1
// CHECK:STDOUT:   %Bar = call %TupleLiteralType @Bar(%TupleLiteralType %1)
// CHECK:STDOUT:   store %TupleLiteralType %Bar, ptr %var, align 1
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }


fn Foo() {}

fn Bar(a: ()) -> () { return a; }

fn Main() {
  var x: () = Bar(Foo());
}

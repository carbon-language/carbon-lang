// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn AddNonmethod(a: i32, b: i32) -> i32 = "int.add";
fn AddMethod[self: i32](b: i32) -> i32 = "int.add";

fn TestAddNonmethod(a: i32, b: i32) -> i32 { return AddNonmethod(a, b); }
fn TestAddMethod(a: i32, b: i32) -> i32 { return a.(AddMethod)(b); }

// CHECK:STDOUT: ; ModuleID = 'method_vs_nonmethod.carbon'
// CHECK:STDOUT: source_filename = "method_vs_nonmethod.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @TestAddNonmethod(i32 %a, i32 %b) {
// CHECK:STDOUT:   %add = add i32 %a, %b
// CHECK:STDOUT:   ret i32 %add
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @TestAddMethod(i32 %a, i32 %b) {
// CHECK:STDOUT:   %add = add i32 %a, %b
// CHECK:STDOUT:   ret i32 %add
// CHECK:STDOUT: }

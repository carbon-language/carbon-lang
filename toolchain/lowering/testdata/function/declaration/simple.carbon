// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: ; ModuleID = 'simple.carbon'
// CHECK:STDOUT: source_filename = "simple.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: %EmptyTupleType = type {}
// CHECK:STDOUT:
// CHECK:STDOUT: declare %EmptyTupleType @F(i32)
// CHECK:STDOUT:
// CHECK:STDOUT: define %EmptyTupleType @G(i32 %n) {
// CHECK:STDOUT:   %F = call %EmptyTupleType @F(i32 %n)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }

fn F(n: i32);

fn G(n: i32) { F(n); }

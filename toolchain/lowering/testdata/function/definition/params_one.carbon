// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: ; ModuleID = 'params_one.carbon'
// CHECK:STDOUT: source_filename = "params_one.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: %EmptyTuple = type {}
// CHECK:STDOUT:
// CHECK:STDOUT: define %EmptyTuple @Foo(i32 %a) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT: }

fn Foo(a: i32) {}

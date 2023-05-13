// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: ; ModuleID = 'toolchain/lowering/testdata/function/definition/params_two.carbon'
// CHECK:STDOUT: source_filename = "toolchain/lowering/testdata/function/definition/params_two.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define void @Foo(i32 %a, i32 %b) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT: }

fn Foo(a: i32, b: i32) {}

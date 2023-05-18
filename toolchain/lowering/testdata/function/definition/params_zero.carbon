// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: ; ModuleID = 'params_zero.carbon'
// CHECK:STDOUT: source_filename = "params_zero.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: %EmptyTuple = type {}
// CHECK:STDOUT:
// CHECK:STDOUT: define %EmptyTuple @Foo() {
// CHECK:STDOUT: entry:
// CHECK:STDOUT: }

fn Foo() {}

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var x: i32;

// CHECK:STDOUT: --- global_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.x = %x} [template]
// CHECK:STDOUT:   %x.var: ref i32 = var x
// CHECK:STDOUT:   %x: ref i32 = bind_name x, %x.var
// CHECK:STDOUT: }
// CHECK:STDOUT:

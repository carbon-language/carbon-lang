// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var x: i32 = 0;

// CHECK:STDOUT: --- global_decl_with_init.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 0 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.x = %x} [template]
// CHECK:STDOUT:   %x.var: ref i32 = var x
// CHECK:STDOUT:   %x: ref i32 = bind_name x, %x.var
// CHECK:STDOUT:   %.loc7: i32 = int_literal 0 [template = constants.%.1]
// CHECK:STDOUT:   assign %x.var, %.loc7
// CHECK:STDOUT: }
// CHECK:STDOUT:

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var a: i32 = (1);
var b: i32 = ((2));

// CHECK:STDOUT: --- parens.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc7: i32 = int_literal 1, const
// CHECK:STDOUT:   %.loc8: i32 = int_literal 2, const
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace package, {.a = %a, .b = %b}
// CHECK:STDOUT:   %a.var: ref i32 = var a
// CHECK:STDOUT:   %a: ref i32 = bind_name a, %a.var
// CHECK:STDOUT:   %.loc7: i32 = int_literal 1, const = constants.%.loc7
// CHECK:STDOUT:   assign %a.var, %.loc7
// CHECK:STDOUT:   %b.var: ref i32 = var b
// CHECK:STDOUT:   %b: ref i32 = bind_name b, %b.var
// CHECK:STDOUT:   %.loc8: i32 = int_literal 2, const = constants.%.loc8
// CHECK:STDOUT:   assign %b.var, %.loc8
// CHECK:STDOUT: }
// CHECK:STDOUT:

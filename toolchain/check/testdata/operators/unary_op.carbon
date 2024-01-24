// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Not(b: bool) -> bool {
  return not b;
}

let not_true: bool = not true;
let not_false: bool = not false;

// CHECK:STDOUT: --- unary_op.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: bool = bool_literal true [template]
// CHECK:STDOUT:   %.2: bool = bool_literal false [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.Not = %Not} [template]
// CHECK:STDOUT:   %Not: <function> = fn_decl @Not [template]
// CHECK:STDOUT:   %.loc11_26: bool = bool_literal true [template = constants.%.1]
// CHECK:STDOUT:   %.loc11_22: bool = not %.loc11_26 [template = constants.%.2]
// CHECK:STDOUT:   %not_true: bool = bind_name not_true, %.loc11_22
// CHECK:STDOUT:   %.loc12_27: bool = bool_literal false [template = constants.%.2]
// CHECK:STDOUT:   %.loc12_23: bool = not %.loc12_27 [template = constants.%.1]
// CHECK:STDOUT:   %not_false: bool = bind_name not_false, %.loc12_23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Not(%b: bool) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %b.ref: bool = name_ref b, %b
// CHECK:STDOUT:   %.loc8: bool = not %b.ref
// CHECK:STDOUT:   return %.loc8
// CHECK:STDOUT: }
// CHECK:STDOUT:

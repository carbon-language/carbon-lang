// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Main() -> {.a: i32} {
  return {.a = 3};
}

// CHECK:STDOUT: --- struct.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = struct_type {.a: i32} [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 3 [template]
// CHECK:STDOUT:   %.3: {.a: i32} = struct_value (%.2) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.Main = %Main} [template]
// CHECK:STDOUT:   %Main: <function> = fn_decl @Main [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() -> {.a: i32} {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8_16: i32 = int_literal 3 [template = constants.%.2]
// CHECK:STDOUT:   %.loc8_17.1: {.a: i32} = struct_literal (%.loc8_16)
// CHECK:STDOUT:   %.loc8_17.2: {.a: i32} = struct_value (%.loc8_16) [template = constants.%.3]
// CHECK:STDOUT:   %.loc8_17.3: {.a: i32} = converted %.loc8_17.1, %.loc8_17.2 [template = constants.%.3]
// CHECK:STDOUT:   return %.loc8_17.3
// CHECK:STDOUT: }
// CHECK:STDOUT:

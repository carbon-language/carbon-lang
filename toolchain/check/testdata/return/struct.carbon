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
// CHECK:STDOUT:   %Main: type = fn_type @Main [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Main = struct_value () [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 3 [template]
// CHECK:STDOUT:   %struct.2: {.a: i32} = struct_value (%.3) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Main.decl: Main = fn_decl @Main [template = constants.%struct.1] {
// CHECK:STDOUT:     %.loc7: type = struct_type {.a: i32} [template = constants.%.1]
// CHECK:STDOUT:     @Main.%return: ref {.a: i32} = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() -> {.a: i32} {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8_16: i32 = int_literal 3 [template = constants.%.3]
// CHECK:STDOUT:   %.loc8_17: {.a: i32} = struct_literal (%.loc8_16)
// CHECK:STDOUT:   %struct: {.a: i32} = struct_value (%.loc8_16) [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc8_18: {.a: i32} = converted %.loc8_17, %struct [template = constants.%struct.2]
// CHECK:STDOUT:   return %.loc8_18
// CHECK:STDOUT: }
// CHECK:STDOUT:

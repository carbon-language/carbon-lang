// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() -> (i32,) { return (0,); }

fn Run() -> i32 {
  return F()[0];
}

// CHECK:STDOUT: --- tuple_return_value_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type (type) [template]
// CHECK:STDOUT:   %.2: type = tuple_type (i32) [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.4: (i32,) = tuple_value (%.3) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.F = %F, .Run = %Run} [template]
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template]
// CHECK:STDOUT:   %Run: <function> = fn_decl @Run [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> (i32,) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_28: i32 = int_literal 0 [template = constants.%.3]
// CHECK:STDOUT:   %.loc7_30.1: (i32,) = tuple_literal (%.loc7_28)
// CHECK:STDOUT:   %.loc7_30.2: (i32,) = tuple_value (%.loc7_28) [template = constants.%.4]
// CHECK:STDOUT:   %.loc7_30.3: (i32,) = converted %.loc7_30.1, %.loc7_30.2 [template = constants.%.4]
// CHECK:STDOUT:   return %.loc7_30.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: <function> = name_ref F, file.%F [template = file.%F]
// CHECK:STDOUT:   %.loc10_11.1: init (i32,) = call %F.ref()
// CHECK:STDOUT:   %.loc10_14: i32 = int_literal 0 [template = constants.%.3]
// CHECK:STDOUT:   %.loc10_11.2: ref (i32,) = temporary_storage
// CHECK:STDOUT:   %.loc10_11.3: ref (i32,) = temporary %.loc10_11.2, %.loc10_11.1
// CHECK:STDOUT:   %.loc10_15.1: ref i32 = tuple_index %.loc10_11.3, %.loc10_14
// CHECK:STDOUT:   %.loc10_15.2: i32 = bind_value %.loc10_15.1
// CHECK:STDOUT:   return %.loc10_15.2
// CHECK:STDOUT: }
// CHECK:STDOUT:

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() -> (i32,) { return (0,); }

fn Run() -> i32 {
  return F()[0];
}

// CHECK:STDOUT: file "tuple_return_value_access.carbon" {
// CHECK:STDOUT:   %F: <function> = fn_decl @F
// CHECK:STDOUT:   %Run: <function> = fn_decl @Run
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> (i32,) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_28: i32 = int_literal 0
// CHECK:STDOUT:   %.loc7_30.1: (i32,) = tuple_literal (%.loc7_28)
// CHECK:STDOUT:   %.loc7_30.2: (i32,) = tuple_value %.loc7_30.1, (%.loc7_28)
// CHECK:STDOUT:   return %.loc7_30.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: <function> = name_reference "F", package.%F
// CHECK:STDOUT:   %.loc10_11.1: init (i32,) = call %F.ref()
// CHECK:STDOUT:   %.loc10_14: i32 = int_literal 0
// CHECK:STDOUT:   %.loc10_11.2: ref (i32,) = temporary_storage
// CHECK:STDOUT:   %.loc10_11.3: ref (i32,) = temporary %.loc10_11.2, %.loc10_11.1
// CHECK:STDOUT:   %.loc10_15.1: ref i32 = tuple_index %.loc10_11.3, %.loc10_14
// CHECK:STDOUT:   %.loc10_15.2: i32 = bind_value %.loc10_15.1
// CHECK:STDOUT:   return %.loc10_15.2
// CHECK:STDOUT: }

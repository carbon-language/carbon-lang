// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F(arr: [i32; 3], i: i32) -> i32 {
  return arr[i];
}

fn G() -> i32 {
  return F((1, 2, 3), 1);
}

// CHECK:STDOUT: file "function_param.carbon" {
// CHECK:STDOUT:   %.loc7 = fn_decl @F
// CHECK:STDOUT:   %.loc11 = fn_decl @G
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%arr: [i32; 3], %i: i32) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8: i32 = array_index %arr, %i
// CHECK:STDOUT:   return %.loc8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc12_13: i32 = int_literal 1
// CHECK:STDOUT:   %.loc12_16: i32 = int_literal 2
// CHECK:STDOUT:   %.loc12_19: i32 = int_literal 3
// CHECK:STDOUT:   %.loc12_20.1: type = tuple_type (i32, i32, i32)
// CHECK:STDOUT:   %.loc12_20.2: (i32, i32, i32) = tuple_literal (%.loc12_13, %.loc12_16, %.loc12_19)
// CHECK:STDOUT:   %.loc12_23: i32 = int_literal 1
// CHECK:STDOUT:   %.loc12_20.3: ref [i32; 3] = temporary_storage
// CHECK:STDOUT:   %.loc12_20.4: i32 = int_literal 0
// CHECK:STDOUT:   %.loc12_20.5: ref i32 = array_index %.loc12_20.3, %.loc12_20.4
// CHECK:STDOUT:   %.loc12_20.6: init i32 = initialize_from %.loc12_13 to %.loc12_20.5
// CHECK:STDOUT:   %.loc12_20.7: i32 = int_literal 1
// CHECK:STDOUT:   %.loc12_20.8: ref i32 = array_index %.loc12_20.3, %.loc12_20.7
// CHECK:STDOUT:   %.loc12_20.9: init i32 = initialize_from %.loc12_16 to %.loc12_20.8
// CHECK:STDOUT:   %.loc12_20.10: i32 = int_literal 2
// CHECK:STDOUT:   %.loc12_20.11: ref i32 = array_index %.loc12_20.3, %.loc12_20.10
// CHECK:STDOUT:   %.loc12_20.12: init i32 = initialize_from %.loc12_19 to %.loc12_20.11
// CHECK:STDOUT:   %.loc12_20.13: init [i32; 3] = array_init %.loc12_20.2, (%.loc12_20.6, %.loc12_20.9, %.loc12_20.12) to %.loc12_20.3
// CHECK:STDOUT:   %.loc12_20.14: ref [i32; 3] = temporary %.loc12_20.3, %.loc12_20.13
// CHECK:STDOUT:   %.loc12_20.15: [i32; 3] = bind_value %.loc12_20.14
// CHECK:STDOUT:   %.loc12_11.1: init i32 = call @F(%.loc12_20.15, %.loc12_23)
// CHECK:STDOUT:   %.loc12_11.2: ref i32 = temporary_storage
// CHECK:STDOUT:   %.loc12_11.3: ref i32 = temporary %.loc12_11.2, %.loc12_11.1
// CHECK:STDOUT:   %.loc12_11.4: i32 = bind_value %.loc12_11.3
// CHECK:STDOUT:   return %.loc12_11.4
// CHECK:STDOUT: }

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
// CHECK:STDOUT:   %.loc12_13.1: i32 = int_literal 1
// CHECK:STDOUT:   %.loc12_13.2: i32 = stub_reference %.loc12_13.1
// CHECK:STDOUT:   %.loc12_16.1: i32 = int_literal 2
// CHECK:STDOUT:   %.loc12_16.2: i32 = stub_reference %.loc12_16.1
// CHECK:STDOUT:   %.loc12_19.1: i32 = int_literal 3
// CHECK:STDOUT:   %.loc12_19.2: i32 = stub_reference %.loc12_19.1
// CHECK:STDOUT:   %.loc12_20.1: type = tuple_type (i32, i32, i32)
// CHECK:STDOUT:   %.loc12_20.2: (i32, i32, i32) = tuple_literal (%.loc12_13.2, %.loc12_16.2, %.loc12_19.2)
// CHECK:STDOUT:   %.loc12_20.3: (i32, i32, i32) = stub_reference %.loc12_20.2
// CHECK:STDOUT:   %.loc12_23.1: i32 = int_literal 1
// CHECK:STDOUT:   %.loc12_23.2: i32 = stub_reference %.loc12_23.1
// CHECK:STDOUT:   %.loc12_20.4: ref [i32; 3] = temporary_storage
// CHECK:STDOUT:   %.loc12_20.5: i32 = int_literal 0
// CHECK:STDOUT:   %.loc12_20.6: ref i32 = array_index %.loc12_20.4, %.loc12_20.5
// CHECK:STDOUT:   %.loc12_20.7: init i32 = initialize_from %.loc12_13.2 to %.loc12_20.6
// CHECK:STDOUT:   %.loc12_20.8: i32 = int_literal 1
// CHECK:STDOUT:   %.loc12_20.9: ref i32 = array_index %.loc12_20.4, %.loc12_20.8
// CHECK:STDOUT:   %.loc12_20.10: init i32 = initialize_from %.loc12_16.2 to %.loc12_20.9
// CHECK:STDOUT:   %.loc12_20.11: i32 = int_literal 2
// CHECK:STDOUT:   %.loc12_20.12: ref i32 = array_index %.loc12_20.4, %.loc12_20.11
// CHECK:STDOUT:   %.loc12_20.13: init i32 = initialize_from %.loc12_19.2 to %.loc12_20.12
// CHECK:STDOUT:   %.loc12_20.14: init [i32; 3] = array_init %.loc12_20.2, (%.loc12_20.7, %.loc12_20.10, %.loc12_20.13) to %.loc12_20.4
// CHECK:STDOUT:   %.loc12_20.15: ref [i32; 3] = temporary %.loc12_20.4, %.loc12_20.14
// CHECK:STDOUT:   %.loc12_20.16: [i32; 3] = bind_value %.loc12_20.15
// CHECK:STDOUT:   %.loc12_11.1: init i32 = call @F(%.loc12_20.16, %.loc12_23.2)
// CHECK:STDOUT:   %.loc12_11.2: ref i32 = temporary_storage
// CHECK:STDOUT:   %.loc12_11.3: ref i32 = temporary %.loc12_11.2, %.loc12_11.1
// CHECK:STDOUT:   %.loc12_11.4: i32 = bind_value %.loc12_11.3
// CHECK:STDOUT:   return %.loc12_11.4
// CHECK:STDOUT: }

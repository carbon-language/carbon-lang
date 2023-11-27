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

// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc7: type = ptr_type [i32; 3]
// CHECK:STDOUT:   %.loc12: type = tuple_type (i32, i32, i32)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file "function_param.carbon" {
// CHECK:STDOUT:   %F: <function> = fn_decl @F
// CHECK:STDOUT:   %G: <function> = fn_decl @G
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%arr: [i32; 3], %i: i32) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %arr.ref: [i32; 3] = name_ref arr, %arr
// CHECK:STDOUT:   %i.ref: i32 = name_ref i, %i
// CHECK:STDOUT:   %.loc8_15.1: ref [i32; 3] = value_as_ref %arr.ref
// CHECK:STDOUT:   %.loc8_15.2: ref i32 = array_index %.loc8_15.1, %i.ref
// CHECK:STDOUT:   %.loc8_15.3: i32 = bind_value %.loc8_15.2
// CHECK:STDOUT:   return %.loc8_15.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: <function> = name_ref F, file.%F
// CHECK:STDOUT:   %.loc12_13: i32 = int_literal 1
// CHECK:STDOUT:   %.loc12_16: i32 = int_literal 2
// CHECK:STDOUT:   %.loc12_19: i32 = int_literal 3
// CHECK:STDOUT:   %.loc12_20.1: (i32, i32, i32) = tuple_literal (%.loc12_13, %.loc12_16, %.loc12_19)
// CHECK:STDOUT:   %.loc12_23: i32 = int_literal 1
// CHECK:STDOUT:   %.loc12_20.2: ref [i32; 3] = temporary_storage
// CHECK:STDOUT:   %.loc12_20.3: i32 = int_literal 0
// CHECK:STDOUT:   %.loc12_20.4: ref i32 = array_index %.loc12_20.2, %.loc12_20.3
// CHECK:STDOUT:   %.loc12_20.5: init i32 = initialize_from %.loc12_13 to %.loc12_20.4
// CHECK:STDOUT:   %.loc12_20.6: i32 = int_literal 1
// CHECK:STDOUT:   %.loc12_20.7: ref i32 = array_index %.loc12_20.2, %.loc12_20.6
// CHECK:STDOUT:   %.loc12_20.8: init i32 = initialize_from %.loc12_16 to %.loc12_20.7
// CHECK:STDOUT:   %.loc12_20.9: i32 = int_literal 2
// CHECK:STDOUT:   %.loc12_20.10: ref i32 = array_index %.loc12_20.2, %.loc12_20.9
// CHECK:STDOUT:   %.loc12_20.11: init i32 = initialize_from %.loc12_19 to %.loc12_20.10
// CHECK:STDOUT:   %.loc12_20.12: init [i32; 3] = array_init (%.loc12_20.5, %.loc12_20.8, %.loc12_20.11) to %.loc12_20.2
// CHECK:STDOUT:   %.loc12_20.13: init [i32; 3] = converted %.loc12_20.1, %.loc12_20.12
// CHECK:STDOUT:   %.loc12_20.14: ref [i32; 3] = temporary %.loc12_20.2, %.loc12_20.13
// CHECK:STDOUT:   %.loc12_20.15: [i32; 3] = bind_value %.loc12_20.14
// CHECK:STDOUT:   %.loc12_11: init i32 = call %F.ref(%.loc12_20.15, %.loc12_23)
// CHECK:STDOUT:   %.loc12_25.1: i32 = value_of_initializer %.loc12_11
// CHECK:STDOUT:   %.loc12_25.2: i32 = converted %.loc12_11, %.loc12_25.1
// CHECK:STDOUT:   return %.loc12_25.2
// CHECK:STDOUT: }

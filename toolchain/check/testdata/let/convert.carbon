// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() -> i32 {
  var v: (i32, i32, i32) = (1, 2, 3);
  // Convert from object representation to value representation.
  let w: (i32, i32, i32) = v;
  return w[1];
}

// CHECK:STDOUT: file "convert.carbon" {
// CHECK:STDOUT:   %.loc7 = fn_decl @F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8_24.1: type = tuple_type (type, type, type)
// CHECK:STDOUT:   %.loc8_24.2: (type, type, type) = tuple_literal (i32, i32, i32)
// CHECK:STDOUT:   %.loc8_24.3: type = tuple_type (i32, i32, i32)
// CHECK:STDOUT:   %v: ref (i32, i32, i32) = var "v"
// CHECK:STDOUT:   %.loc8_29: i32 = int_literal 1
// CHECK:STDOUT:   %.loc8_32: i32 = int_literal 2
// CHECK:STDOUT:   %.loc8_35: i32 = int_literal 3
// CHECK:STDOUT:   %.loc8_36.1: (i32, i32, i32) = tuple_literal (%.loc8_29, %.loc8_32, %.loc8_35)
// CHECK:STDOUT:   %.loc8_36.2: ref i32 = tuple_access %v, member0
// CHECK:STDOUT:   %.loc8_36.3: init i32 = initialize_from %.loc8_29 to %.loc8_36.2
// CHECK:STDOUT:   %.loc8_36.4: ref i32 = tuple_access %v, member1
// CHECK:STDOUT:   %.loc8_36.5: init i32 = initialize_from %.loc8_32 to %.loc8_36.4
// CHECK:STDOUT:   %.loc8_36.6: ref i32 = tuple_access %v, member2
// CHECK:STDOUT:   %.loc8_36.7: init i32 = initialize_from %.loc8_35 to %.loc8_36.6
// CHECK:STDOUT:   %.loc8_36.8: init (i32, i32, i32) = tuple_init %.loc8_36.1, (%.loc8_36.3, %.loc8_36.5, %.loc8_36.7)
// CHECK:STDOUT:   assign %v, %.loc8_36.8
// CHECK:STDOUT:   %.loc10: (type, type, type) = tuple_literal (i32, i32, i32)
// CHECK:STDOUT:   %.loc8_7.1: ref i32 = tuple_access %v, member0
// CHECK:STDOUT:   %.loc8_7.2: i32 = bind_value %.loc8_7.1
// CHECK:STDOUT:   %.loc8_7.3: ref i32 = tuple_access %v, member1
// CHECK:STDOUT:   %.loc8_7.4: i32 = bind_value %.loc8_7.3
// CHECK:STDOUT:   %.loc8_7.5: ref i32 = tuple_access %v, member2
// CHECK:STDOUT:   %.loc8_7.6: i32 = bind_value %.loc8_7.5
// CHECK:STDOUT:   %.loc8_7.7: (i32, i32, i32) = tuple_value %v, (%.loc8_7.2, %.loc8_7.4, %.loc8_7.6)
// CHECK:STDOUT:   %w: (i32, i32, i32) = bind_name "w", %.loc8_7.7
// CHECK:STDOUT:   %.loc11_12: i32 = int_literal 1
// CHECK:STDOUT:   %.loc11_13: i32 = tuple_index %w, %.loc11_12
// CHECK:STDOUT:   return %.loc11_13
// CHECK:STDOUT: }

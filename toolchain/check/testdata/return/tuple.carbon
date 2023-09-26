// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE


fn Main() -> (i32, i32) {
  return (15, 35);
}

// CHECK:STDOUT: file "tuple.carbon" {
// CHECK:STDOUT:   %.loc8 = fn_decl @Main
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() -> %return: (i32, i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc9_11: i32 = int_literal 15
// CHECK:STDOUT:   %.loc9_15: i32 = int_literal 35
// CHECK:STDOUT:   %.loc9_17: (i32, i32) = tuple_literal (%.loc9_11, %.loc9_15)
// CHECK:STDOUT:   %.loc9_18.1: ref i32 = tuple_access %return, member0
// CHECK:STDOUT:   %.loc9_18.2: init i32 = initialize_from %.loc9_11 to %.loc9_18.1
// CHECK:STDOUT:   %.loc9_18.3: ref i32 = tuple_access %return, member1
// CHECK:STDOUT:   %.loc9_18.4: init i32 = initialize_from %.loc9_15 to %.loc9_18.3
// CHECK:STDOUT:   %.loc9_18.5: init (i32, i32) = tuple_init %.loc9_17, (%.loc9_18.2, %.loc9_18.4)
// CHECK:STDOUT:   return %.loc9_18.5
// CHECK:STDOUT: }

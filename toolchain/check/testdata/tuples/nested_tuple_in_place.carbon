// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() -> (i32, i32, i32);

fn G() {
  var v: ((i32, i32, i32), (i32, i32, i32)) = (F(), F());
}

fn H() {
  var v: (i32, (i32, i32, i32), i32) = (1, F(), 2);
}

// CHECK:STDOUT: file "nested_tuple_in_place.carbon" {
// CHECK:STDOUT:   %.loc7 = fn_decl @F
// CHECK:STDOUT:   %.loc9 = fn_decl @G
// CHECK:STDOUT:   %.loc13 = fn_decl @H
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %return: (i32, i32, i32);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_25: (type, type, type) = tuple_literal (i32, i32, i32)
// CHECK:STDOUT:   %.loc10_42: (type, type, type) = tuple_literal (i32, i32, i32)
// CHECK:STDOUT:   %.loc10_43.1: type = tuple_type ((type, type, type), (type, type, type))
// CHECK:STDOUT:   %.loc10_43.2: ((type, type, type), (type, type, type)) = tuple_literal (%.loc10_25, %.loc10_42)
// CHECK:STDOUT:   %.loc10_43.3: type = tuple_type ((i32, i32, i32), (i32, i32, i32))
// CHECK:STDOUT:   %v: ref ((i32, i32, i32), (i32, i32, i32)) = var "v"
// CHECK:STDOUT:   %.loc10_57.1: ref (i32, i32, i32) = tuple_access %v, member0
// CHECK:STDOUT:   %.loc10_49: init (i32, i32, i32) = call @F() to %.loc10_57.1
// CHECK:STDOUT:   %.loc10_57.2: ref (i32, i32, i32) = tuple_access %v, member1
// CHECK:STDOUT:   %.loc10_54: init (i32, i32, i32) = call @F() to %.loc10_57.2
// CHECK:STDOUT:   %.loc10_56: ((i32, i32, i32), (i32, i32, i32)) = tuple_literal (%.loc10_49, %.loc10_54)
// CHECK:STDOUT:   %.loc10_57.3: init ((i32, i32, i32), (i32, i32, i32)) = tuple_init %.loc10_56, (%.loc10_49, %.loc10_54)
// CHECK:STDOUT:   assign %v, %.loc10_57.3
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc14_30: (type, type, type) = tuple_literal (i32, i32, i32)
// CHECK:STDOUT:   %.loc14_36.1: type = tuple_type (type, (type, type, type), type)
// CHECK:STDOUT:   %.loc14_36.2: (type, (type, type, type), type) = tuple_literal (i32, %.loc14_30, i32)
// CHECK:STDOUT:   %.loc14_36.3: type = tuple_type (i32, (i32, i32, i32), i32)
// CHECK:STDOUT:   %v: ref (i32, (i32, i32, i32), i32) = var "v"
// CHECK:STDOUT:   %.loc14_41: i32 = int_literal 1
// CHECK:STDOUT:   %.loc14_51.1: ref (i32, i32, i32) = tuple_access %v, member1
// CHECK:STDOUT:   %.loc14_45: init (i32, i32, i32) = call @F() to %.loc14_51.1
// CHECK:STDOUT:   %.loc14_49: i32 = int_literal 2
// CHECK:STDOUT:   %.loc14_50: (i32, (i32, i32, i32), i32) = tuple_literal (%.loc14_41, %.loc14_45, %.loc14_49)
// CHECK:STDOUT:   %.loc14_51.2: ref i32 = tuple_access %v, member0
// CHECK:STDOUT:   %.loc14_51.3: init i32 = initialize_from %.loc14_41 to %.loc14_51.2
// CHECK:STDOUT:   %.loc14_51.4: ref i32 = tuple_access %v, member2
// CHECK:STDOUT:   %.loc14_51.5: init i32 = initialize_from %.loc14_49 to %.loc14_51.4
// CHECK:STDOUT:   %.loc14_51.6: init (i32, (i32, i32, i32), i32) = tuple_init %.loc14_50, (%.loc14_51.3, %.loc14_45, %.loc14_51.5)
// CHECK:STDOUT:   assign %v, %.loc14_51.6
// CHECK:STDOUT:   return
// CHECK:STDOUT: }

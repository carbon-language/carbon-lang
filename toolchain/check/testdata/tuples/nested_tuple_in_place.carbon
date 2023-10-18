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
// CHECK:STDOUT:   %.loc7: type = ptr_type (i32, i32, i32)
// CHECK:STDOUT:   %F: <function> = fn_decl @F
// CHECK:STDOUT:   %G: <function> = fn_decl @G
// CHECK:STDOUT:   %H: <function> = fn_decl @H
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
// CHECK:STDOUT:   %.loc10_43.4: type = tuple_type ((i32, i32, i32)*, (i32, i32, i32)*)
// CHECK:STDOUT:   %.loc10_43.5: type = ptr_type ((i32, i32, i32)*, (i32, i32, i32)*)
// CHECK:STDOUT:   %v: ref ((i32, i32, i32), (i32, i32, i32)) = var "v"
// CHECK:STDOUT:   %F.ref.loc10_48: <function> = name_reference "F", package.%F
// CHECK:STDOUT:   %.loc10_56.1: ref (i32, i32, i32) = tuple_access %v, member0
// CHECK:STDOUT:   %.loc10_49: init (i32, i32, i32) = call %F.ref.loc10_48() to %.loc10_56.1
// CHECK:STDOUT:   %F.ref.loc10_53: <function> = name_reference "F", package.%F
// CHECK:STDOUT:   %.loc10_56.2: ref (i32, i32, i32) = tuple_access %v, member1
// CHECK:STDOUT:   %.loc10_54: init (i32, i32, i32) = call %F.ref.loc10_53() to %.loc10_56.2
// CHECK:STDOUT:   %.loc10_56.3: ((i32, i32, i32), (i32, i32, i32)) = tuple_literal (%.loc10_49, %.loc10_54)
// CHECK:STDOUT:   %.loc10_56.4: init ((i32, i32, i32), (i32, i32, i32)) = tuple_init %.loc10_56.3, (%.loc10_49, %.loc10_54)
// CHECK:STDOUT:   assign %v, %.loc10_56.4
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc14_30: (type, type, type) = tuple_literal (i32, i32, i32)
// CHECK:STDOUT:   %.loc14_36.1: type = tuple_type (type, (type, type, type), type)
// CHECK:STDOUT:   %.loc14_36.2: (type, (type, type, type), type) = tuple_literal (i32, %.loc14_30, i32)
// CHECK:STDOUT:   %.loc14_36.3: type = tuple_type (i32, (i32, i32, i32), i32)
// CHECK:STDOUT:   %.loc14_36.4: type = tuple_type (i32, (i32, i32, i32)*, i32)
// CHECK:STDOUT:   %.loc14_36.5: type = ptr_type (i32, (i32, i32, i32)*, i32)
// CHECK:STDOUT:   %v: ref (i32, (i32, i32, i32), i32) = var "v"
// CHECK:STDOUT:   %.loc14_41: i32 = int_literal 1
// CHECK:STDOUT:   %F.ref: <function> = name_reference "F", package.%F
// CHECK:STDOUT:   %.loc14_50.1: ref (i32, i32, i32) = tuple_access %v, member1
// CHECK:STDOUT:   %.loc14_45: init (i32, i32, i32) = call %F.ref() to %.loc14_50.1
// CHECK:STDOUT:   %.loc14_49: i32 = int_literal 2
// CHECK:STDOUT:   %.loc14_50.2: (i32, (i32, i32, i32), i32) = tuple_literal (%.loc14_41, %.loc14_45, %.loc14_49)
// CHECK:STDOUT:   %.loc14_50.3: ref i32 = tuple_access %v, member0
// CHECK:STDOUT:   %.loc14_50.4: init i32 = initialize_from %.loc14_41 to %.loc14_50.3
// CHECK:STDOUT:   %.loc14_50.5: ref i32 = tuple_access %v, member2
// CHECK:STDOUT:   %.loc14_50.6: init i32 = initialize_from %.loc14_49 to %.loc14_50.5
// CHECK:STDOUT:   %.loc14_50.7: init (i32, (i32, i32, i32), i32) = tuple_init %.loc14_50.2, (%.loc14_50.4, %.loc14_45, %.loc14_50.6)
// CHECK:STDOUT:   assign %v, %.loc14_50.7
// CHECK:STDOUT:   return
// CHECK:STDOUT: }

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() -> (i32, i32, i32);

fn G() {
  var v: {.a: (i32, i32, i32), .b: (i32, i32, i32)} = {.a = F(), .b = F()};
}

// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc7_25.1: type = tuple_type (type, type, type)
// CHECK:STDOUT:   %.loc7_25.2: type = tuple_type (i32, i32, i32)
// CHECK:STDOUT:   %.loc7_25.3: type = ptr_type (i32, i32, i32)
// CHECK:STDOUT:   %.loc10_51.1: type = struct_type {.a: (i32, i32, i32)*, .b: (i32, i32, i32)*}
// CHECK:STDOUT:   %.loc10_51.2: type = ptr_type {.a: (i32, i32, i32)*, .b: (i32, i32, i32)*}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file "nested_struct_in_place.carbon" {
// CHECK:STDOUT:   %F: <function> = fn_decl @F
// CHECK:STDOUT:   %G: <function> = fn_decl @G
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %return: (i32, i32, i32);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_29: (type, type, type) = tuple_literal (i32, i32, i32)
// CHECK:STDOUT:   %.loc7_25.1: type = converted %.loc10_29, constants.%.loc7_25.2
// CHECK:STDOUT:   %.loc10_50: (type, type, type) = tuple_literal (i32, i32, i32)
// CHECK:STDOUT:   %.loc7_25.2: type = converted %.loc10_50, constants.%.loc7_25.2
// CHECK:STDOUT:   %.loc10_51: type = struct_type {.a: (i32, i32, i32), .b: (i32, i32, i32)}
// CHECK:STDOUT:   %v.var: ref {.a: (i32, i32, i32), .b: (i32, i32, i32)} = var v
// CHECK:STDOUT:   %v: ref {.a: (i32, i32, i32), .b: (i32, i32, i32)} = bind_name v, %v.var
// CHECK:STDOUT:   %F.ref.loc10_61: <function> = name_ref F, file.%F
// CHECK:STDOUT:   %.loc10_74.1: ref (i32, i32, i32) = struct_access %v.var, member0
// CHECK:STDOUT:   %.loc10_62: init (i32, i32, i32) = call %F.ref.loc10_61() to %.loc10_74.1
// CHECK:STDOUT:   %F.ref.loc10_71: <function> = name_ref F, file.%F
// CHECK:STDOUT:   %.loc10_74.2: ref (i32, i32, i32) = struct_access %v.var, member1
// CHECK:STDOUT:   %.loc10_72: init (i32, i32, i32) = call %F.ref.loc10_71() to %.loc10_74.2
// CHECK:STDOUT:   %.loc10_74.3: {.a: (i32, i32, i32), .b: (i32, i32, i32)} = struct_literal (%.loc10_62, %.loc10_72)
// CHECK:STDOUT:   %.loc10_74.4: init {.a: (i32, i32, i32), .b: (i32, i32, i32)} = struct_init (%.loc10_62, %.loc10_72) to %v.var
// CHECK:STDOUT:   %.loc10_74.5: init {.a: (i32, i32, i32), .b: (i32, i32, i32)} = converted %.loc10_74.3, %.loc10_74.4
// CHECK:STDOUT:   assign %v.var, %.loc10_74.5
// CHECK:STDOUT:   return
// CHECK:STDOUT: }

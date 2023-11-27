// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class C {
  var a: i32;
  var b: i32;
}

fn F() -> C {
  returned var result: C = {.a = 1, .b = 2};
  return var;
}

fn G() -> i32 {
  returned var result: i32 = 0;
  return var;
}

// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc10_1.1: type = struct_type {.a: i32, .b: i32}
// CHECK:STDOUT:   %.loc10_1.2: type = ptr_type {.a: i32, .b: i32}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file "returned_var.carbon" {
// CHECK:STDOUT:   class_decl @C, ()
// CHECK:STDOUT:   %C: type = class_type @C
// CHECK:STDOUT:   %F: <function> = fn_decl @F
// CHECK:STDOUT:   %G: <function> = fn_decl @G
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc8_8.1: type = unbound_field_type C, i32
// CHECK:STDOUT:   %.loc8_8.2: <unbound field of class C> = field a, member0
// CHECK:STDOUT:   %a: <unbound field of class C> = bind_name a, %.loc8_8.2
// CHECK:STDOUT:   %.loc9_8.1: type = unbound_field_type C, i32
// CHECK:STDOUT:   %.loc9_8.2: <unbound field of class C> = field b, member1
// CHECK:STDOUT:   %b: <unbound field of class C> = bind_name b, %.loc9_8.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .a = %a
// CHECK:STDOUT:   .b = %b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %return: C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C
// CHECK:STDOUT:   %result: ref C = bind_name result, %return
// CHECK:STDOUT:   %.loc13_34: i32 = int_literal 1
// CHECK:STDOUT:   %.loc13_42: i32 = int_literal 2
// CHECK:STDOUT:   %.loc13_43.1: {.a: i32, .b: i32} = struct_literal (%.loc13_34, %.loc13_42)
// CHECK:STDOUT:   %.loc13_43.2: ref i32 = class_field_access %return, member0
// CHECK:STDOUT:   %.loc13_43.3: init i32 = initialize_from %.loc13_34 to %.loc13_43.2
// CHECK:STDOUT:   %.loc13_43.4: ref i32 = class_field_access %return, member1
// CHECK:STDOUT:   %.loc13_43.5: init i32 = initialize_from %.loc13_42 to %.loc13_43.4
// CHECK:STDOUT:   %.loc13_43.6: init C = class_init (%.loc13_43.3, %.loc13_43.5), %return
// CHECK:STDOUT:   %.loc13_43.7: init C = converted %.loc13_43.1, %.loc13_43.6
// CHECK:STDOUT:   assign %return, %.loc13_43.7
// CHECK:STDOUT:   return %result
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %result.var: ref i32 = var result
// CHECK:STDOUT:   %result: ref i32 = bind_name result, %result.var
// CHECK:STDOUT:   %.loc18_30: i32 = int_literal 0
// CHECK:STDOUT:   assign %result.var, %.loc18_30
// CHECK:STDOUT:   %.loc18_16: i32 = bind_value %result
// CHECK:STDOUT:   return %.loc18_16
// CHECK:STDOUT: }

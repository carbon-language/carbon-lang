// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class Class {
  var j: i32;
  var k: i32;
}

fn Run() -> i32 {
  var cv: Class;
  cv.j = 1;
  cv.k = 2;
  let c: Class = cv;
  return c.j + c.k;
}

// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc10_1.1: type = struct_type {.j: i32, .k: i32}
// CHECK:STDOUT:   %.loc10_1.2: type = ptr_type {.j: i32, .k: i32}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file "field_access_in_value.carbon" {
// CHECK:STDOUT:   class_decl @Class, ()
// CHECK:STDOUT:   %Class: type = class_type @Class
// CHECK:STDOUT:   %Run: <function> = fn_decl @Run
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %.loc8_8.1: type = unbound_field_type Class, i32
// CHECK:STDOUT:   %.loc8_8.2: <unbound field of class Class> = field j, member0
// CHECK:STDOUT:   %j: <unbound field of class Class> = bind_name j, %.loc8_8.2
// CHECK:STDOUT:   %.loc9_8.1: type = unbound_field_type Class, i32
// CHECK:STDOUT:   %.loc9_8.2: <unbound field of class Class> = field k, member1
// CHECK:STDOUT:   %k: <unbound field of class Class> = bind_name k, %.loc9_8.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .j = %j
// CHECK:STDOUT:   .k = %k
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Class.ref.loc13: type = name_ref Class, file.%Class
// CHECK:STDOUT:   %cv.var: ref Class = var cv
// CHECK:STDOUT:   %cv: ref Class = bind_name cv, %cv.var
// CHECK:STDOUT:   %cv.ref.loc14: ref Class = name_ref cv, %cv
// CHECK:STDOUT:   %.loc14_5: ref i32 = class_field_access %cv.ref.loc14, member0
// CHECK:STDOUT:   %.loc14_10: i32 = int_literal 1
// CHECK:STDOUT:   assign %.loc14_5, %.loc14_10
// CHECK:STDOUT:   %cv.ref.loc15: ref Class = name_ref cv, %cv
// CHECK:STDOUT:   %.loc15_5: ref i32 = class_field_access %cv.ref.loc15, member1
// CHECK:STDOUT:   %.loc15_10: i32 = int_literal 2
// CHECK:STDOUT:   assign %.loc15_5, %.loc15_10
// CHECK:STDOUT:   %Class.ref.loc16: type = name_ref Class, file.%Class
// CHECK:STDOUT:   %cv.ref.loc16: ref Class = name_ref cv, %cv
// CHECK:STDOUT:   %.loc16: Class = bind_value %cv.ref.loc16
// CHECK:STDOUT:   %c: Class = bind_name c, %.loc16
// CHECK:STDOUT:   %c.ref.loc17_10: Class = name_ref c, %c
// CHECK:STDOUT:   %.loc17_11.1: ref i32 = class_field_access %c.ref.loc17_10, member0
// CHECK:STDOUT:   %.loc17_11.2: i32 = bind_value %.loc17_11.1
// CHECK:STDOUT:   %c.ref.loc17_16: Class = name_ref c, %c
// CHECK:STDOUT:   %.loc17_17.1: ref i32 = class_field_access %c.ref.loc17_16, member1
// CHECK:STDOUT:   %.loc17_17.2: i32 = bind_value %.loc17_17.1
// CHECK:STDOUT:   %.loc17_14: i32 = add %.loc17_11.2, %.loc17_17.2
// CHECK:STDOUT:   return %.loc17_14
// CHECK:STDOUT: }

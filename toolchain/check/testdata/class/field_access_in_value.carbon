// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class Class {
  var j: i32;
  var k: i32;
}

fn Test() {
  var cv: Class;
  cv.j = 1;
  cv.k = 2;
  let c: Class = cv;
  var cj: i32 = c.j;
  var ck: i32 = c.k;
}

// CHECK:STDOUT: --- field_access_in_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [template]
// CHECK:STDOUT:   %.1: type = unbound_element_type Class, i32 [template]
// CHECK:STDOUT:   %.2: type = struct_type {.j: i32, .k: i32} [template]
// CHECK:STDOUT:   %Test: type = fn_type @Test [template]
// CHECK:STDOUT:   %.3: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: Test = struct_value () [template]
// CHECK:STDOUT:   %.4: type = ptr_type {.j: i32, .k: i32} [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.6: i32 = int_literal 2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .Test = %Test.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [template = constants.%Class] {}
// CHECK:STDOUT:   %Test.decl: Test = fn_decl @Test [template = constants.%struct] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %.loc8: <unbound element of class Class> = field_decl j, element0 [template]
// CHECK:STDOUT:   %.loc9: <unbound element of class Class> = field_decl k, element1 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .j = %.loc8
// CHECK:STDOUT:   .k = %.loc9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Class.ref.loc13: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:   %cv.var: ref Class = var cv
// CHECK:STDOUT:   %cv: ref Class = bind_name cv, %cv.var
// CHECK:STDOUT:   %cv.ref.loc14: ref Class = name_ref cv, %cv
// CHECK:STDOUT:   %j.ref.loc14: <unbound element of class Class> = name_ref j, @Class.%.loc8 [template = @Class.%.loc8]
// CHECK:STDOUT:   %.loc14_5: ref i32 = class_element_access %cv.ref.loc14, element0
// CHECK:STDOUT:   %.loc14_10: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   assign %.loc14_5, %.loc14_10
// CHECK:STDOUT:   %cv.ref.loc15: ref Class = name_ref cv, %cv
// CHECK:STDOUT:   %k.ref.loc15: <unbound element of class Class> = name_ref k, @Class.%.loc9 [template = @Class.%.loc9]
// CHECK:STDOUT:   %.loc15_5: ref i32 = class_element_access %cv.ref.loc15, element1
// CHECK:STDOUT:   %.loc15_10: i32 = int_literal 2 [template = constants.%.6]
// CHECK:STDOUT:   assign %.loc15_5, %.loc15_10
// CHECK:STDOUT:   %Class.ref.loc16: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:   %cv.ref.loc16: ref Class = name_ref cv, %cv
// CHECK:STDOUT:   %.loc16: Class = bind_value %cv.ref.loc16
// CHECK:STDOUT:   %c: Class = bind_name c, %.loc16
// CHECK:STDOUT:   %cj.var: ref i32 = var cj
// CHECK:STDOUT:   %cj: ref i32 = bind_name cj, %cj.var
// CHECK:STDOUT:   %c.ref.loc17: Class = name_ref c, %c
// CHECK:STDOUT:   %j.ref.loc17: <unbound element of class Class> = name_ref j, @Class.%.loc8 [template = @Class.%.loc8]
// CHECK:STDOUT:   %.loc17_18.1: ref i32 = class_element_access %c.ref.loc17, element0
// CHECK:STDOUT:   %.loc17_18.2: i32 = bind_value %.loc17_18.1
// CHECK:STDOUT:   assign %cj.var, %.loc17_18.2
// CHECK:STDOUT:   %ck.var: ref i32 = var ck
// CHECK:STDOUT:   %ck: ref i32 = bind_name ck, %ck.var
// CHECK:STDOUT:   %c.ref.loc18: Class = name_ref c, %c
// CHECK:STDOUT:   %k.ref.loc18: <unbound element of class Class> = name_ref k, @Class.%.loc9 [template = @Class.%.loc9]
// CHECK:STDOUT:   %.loc18_18.1: ref i32 = class_element_access %c.ref.loc18, element1
// CHECK:STDOUT:   %.loc18_18.2: i32 = bind_value %.loc18_18.1
// CHECK:STDOUT:   assign %ck.var, %.loc18_18.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

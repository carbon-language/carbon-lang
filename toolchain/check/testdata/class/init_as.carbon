// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class Class {
  var a: i32;
  var b: i32;
}

fn F() -> i32 {
  return ({.a = 1, .b = 2} as Class).a;
}

// CHECK:STDOUT: --- init_as.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [template]
// CHECK:STDOUT:   %.1: type = unbound_element_type Class, i32 [template]
// CHECK:STDOUT:   %.2: type = struct_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT:   %F: type = fn_type @F [template]
// CHECK:STDOUT:   %.3: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: F = struct_value () [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.6: type = ptr_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT:   %struct.2: Class = struct_value (%.4, %.5) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [template = constants.%Class] {}
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct.1] {
// CHECK:STDOUT:     @F.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %.loc8: <unbound element of class Class> = field_decl a, element0 [template]
// CHECK:STDOUT:   %.loc9: <unbound element of class Class> = field_decl b, element1 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .a = %.loc8
// CHECK:STDOUT:   .b = %.loc9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc13_17: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc13_25: i32 = int_literal 2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc13_26.1: {.a: i32, .b: i32} = struct_literal (%.loc13_17, %.loc13_25)
// CHECK:STDOUT:   %Class.ref: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:   %.loc13_26.2: ref Class = temporary_storage
// CHECK:STDOUT:   %.loc13_26.3: ref i32 = class_element_access %.loc13_26.2, element0
// CHECK:STDOUT:   %.loc13_26.4: init i32 = initialize_from %.loc13_17 to %.loc13_26.3 [template = constants.%.4]
// CHECK:STDOUT:   %.loc13_26.5: ref i32 = class_element_access %.loc13_26.2, element1
// CHECK:STDOUT:   %.loc13_26.6: init i32 = initialize_from %.loc13_25 to %.loc13_26.5 [template = constants.%.5]
// CHECK:STDOUT:   %.loc13_26.7: init Class = class_init (%.loc13_26.4, %.loc13_26.6), %.loc13_26.2 [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc13_26.8: ref Class = temporary %.loc13_26.2, %.loc13_26.7
// CHECK:STDOUT:   %.loc13_28: ref Class = converted %.loc13_26.1, %.loc13_26.8
// CHECK:STDOUT:   %a.ref: <unbound element of class Class> = name_ref a, @Class.%.loc8 [template = @Class.%.loc8]
// CHECK:STDOUT:   %.loc13_37.1: ref i32 = class_element_access %.loc13_28, element0
// CHECK:STDOUT:   %.loc13_37.2: i32 = bind_value %.loc13_37.1
// CHECK:STDOUT:   return %.loc13_37.2
// CHECK:STDOUT: }
// CHECK:STDOUT:

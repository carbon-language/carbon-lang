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

// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc10_1.1: type = struct_type {.a: i32, .b: i32}
// CHECK:STDOUT:   %.loc10_1.2: type = ptr_type {.a: i32, .b: i32}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file "init_as.carbon" {
// CHECK:STDOUT:   class_decl @Class, ()
// CHECK:STDOUT:   %Class: type = class_type @Class
// CHECK:STDOUT:   %F: <function> = fn_decl @F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %.loc8_8.1: type = unbound_field_type Class, i32
// CHECK:STDOUT:   %.loc8_8.2: <unbound field of class Class> = field a, member0
// CHECK:STDOUT:   %a: <unbound field of class Class> = bind_name a, %.loc8_8.2
// CHECK:STDOUT:   %.loc9_8.1: type = unbound_field_type Class, i32
// CHECK:STDOUT:   %.loc9_8.2: <unbound field of class Class> = field b, member1
// CHECK:STDOUT:   %b: <unbound field of class Class> = bind_name b, %.loc9_8.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .a = %a
// CHECK:STDOUT:   .b = %b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc13_17: i32 = int_literal 1
// CHECK:STDOUT:   %.loc13_25: i32 = int_literal 2
// CHECK:STDOUT:   %.loc13_26.1: {.a: i32, .b: i32} = struct_literal (%.loc13_17, %.loc13_25)
// CHECK:STDOUT:   %Class.ref: type = name_ref Class, file.%Class
// CHECK:STDOUT:   %.loc13_26.2: ref Class = temporary_storage
// CHECK:STDOUT:   %.loc13_26.3: ref i32 = class_field_access %.loc13_26.2, member0
// CHECK:STDOUT:   %.loc13_26.4: init i32 = initialize_from %.loc13_17 to %.loc13_26.3
// CHECK:STDOUT:   %.loc13_26.5: ref i32 = class_field_access %.loc13_26.2, member1
// CHECK:STDOUT:   %.loc13_26.6: init i32 = initialize_from %.loc13_25 to %.loc13_26.5
// CHECK:STDOUT:   %.loc13_26.7: init Class = class_init (%.loc13_26.4, %.loc13_26.6), %.loc13_26.2
// CHECK:STDOUT:   %.loc13_26.8: ref Class = temporary %.loc13_26.2, %.loc13_26.7
// CHECK:STDOUT:   %.loc13_26.9: ref Class = converted %.loc13_26.1, %.loc13_26.8
// CHECK:STDOUT:   %.loc13_37.1: ref i32 = class_field_access %.loc13_26.9, member0
// CHECK:STDOUT:   %.loc13_37.2: i32 = bind_value %.loc13_37.1
// CHECK:STDOUT:   return %.loc13_37.2
// CHECK:STDOUT: }

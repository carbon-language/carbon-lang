// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class Class {
  var a: i32;
  var b: i32;
}

fn G(p: Class*);

fn F() {
  // TODO: This case should presumably work: `{...} as Class` should be an
  // initializing expression, not a value expression.
  //
  // CHECK:STDERR: fail_init_as_inplace.carbon:[[@LINE+3]]:18: ERROR: Cannot copy value of type `Class`.
  // CHECK:STDERR:   var c: Class = {.a = 1, .b = 2} as Class;
  // CHECK:STDERR:                  ^~~~~~~~~~~~~~~~
  var c: Class = {.a = 1, .b = 2} as Class;
  G(&c);
}

// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc10_1.1: type = struct_type {.a: i32, .b: i32}
// CHECK:STDOUT:   %.loc10_1.2: type = ptr_type {.a: i32, .b: i32}
// CHECK:STDOUT:   %.loc22: type = tuple_type ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file "fail_init_as_inplace.carbon" {
// CHECK:STDOUT:   class_decl @Class, ()
// CHECK:STDOUT:   %Class: type = class_type @Class
// CHECK:STDOUT:   %G: <function> = fn_decl @G
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
// CHECK:STDOUT: fn @G(%p: Class*);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Class.ref.loc21_10: type = name_ref Class, file.%Class
// CHECK:STDOUT:   %c.var: ref Class = var c
// CHECK:STDOUT:   %c: ref Class = bind_name c, %c.var
// CHECK:STDOUT:   %.loc21_24: i32 = int_literal 1
// CHECK:STDOUT:   %.loc21_32: i32 = int_literal 2
// CHECK:STDOUT:   %.loc21_33.1: {.a: i32, .b: i32} = struct_literal (%.loc21_24, %.loc21_32)
// CHECK:STDOUT:   %Class.ref.loc21_38: type = name_ref Class, file.%Class
// CHECK:STDOUT:   %.loc21_33.2: ref Class = temporary_storage
// CHECK:STDOUT:   %.loc21_33.3: ref i32 = class_field_access %.loc21_33.2, member0
// CHECK:STDOUT:   %.loc21_33.4: init i32 = initialize_from %.loc21_24 to %.loc21_33.3
// CHECK:STDOUT:   %.loc21_33.5: ref i32 = class_field_access %.loc21_33.2, member1
// CHECK:STDOUT:   %.loc21_33.6: init i32 = initialize_from %.loc21_32 to %.loc21_33.5
// CHECK:STDOUT:   %.loc21_33.7: init Class = class_init (%.loc21_33.4, %.loc21_33.6), %.loc21_33.2
// CHECK:STDOUT:   %.loc21_33.8: ref Class = temporary %.loc21_33.2, %.loc21_33.7
// CHECK:STDOUT:   %.loc21_33.9: ref Class = converted %.loc21_33.1, %.loc21_33.8
// CHECK:STDOUT:   %.loc21_33.10: Class = bind_value %.loc21_33.9
// CHECK:STDOUT:   assign %c.var, <error>
// CHECK:STDOUT:   %G.ref: <function> = name_ref G, file.%G
// CHECK:STDOUT:   %c.ref: ref Class = name_ref c, %c
// CHECK:STDOUT:   %.loc22_5: Class* = address_of %c.ref
// CHECK:STDOUT:   %.loc22_4: init () = call %G.ref(%.loc22_5)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }

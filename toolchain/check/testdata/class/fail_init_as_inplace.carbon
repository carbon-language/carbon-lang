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

// CHECK:STDOUT: --- fail_init_as_inplace.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [template]
// CHECK:STDOUT:   %.1: type = unbound_element_type Class, i32 [template]
// CHECK:STDOUT:   %.2: type = struct_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT:   %.3: type = ptr_type Class [template]
// CHECK:STDOUT:   %.4: type = ptr_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.6: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.7: Class = struct_value (%.5, %.6) [template]
// CHECK:STDOUT:   %.8: type = tuple_type () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.Class = %Class.decl, .G = %G, .F = %F} [template]
// CHECK:STDOUT:   %Class.decl = class_decl @Class, ()
// CHECK:STDOUT:   %G: <function> = fn_decl @G [template]
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %.loc8: <unbound element of class Class> = field_decl a, element0 [template]
// CHECK:STDOUT:   %.loc9: <unbound element of class Class> = field_decl b, element1 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .a = %.loc8
// CHECK:STDOUT:   .b = %.loc9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%p: Class*);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Class.ref.loc21_10: type = name_ref Class, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:   %c.var: ref Class = var c
// CHECK:STDOUT:   %c: ref Class = bind_name c, %c.var
// CHECK:STDOUT:   %.loc21_24: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc21_32: i32 = int_literal 2 [template = constants.%.6]
// CHECK:STDOUT:   %.loc21_33.1: {.a: i32, .b: i32} = struct_literal (%.loc21_24, %.loc21_32)
// CHECK:STDOUT:   %Class.ref.loc21_38: type = name_ref Class, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:   %.loc21_33.2: ref Class = temporary_storage
// CHECK:STDOUT:   %.loc21_33.3: ref i32 = class_element_access %.loc21_33.2, element0
// CHECK:STDOUT:   %.loc21_33.4: init i32 = initialize_from %.loc21_24 to %.loc21_33.3 [template = constants.%.5]
// CHECK:STDOUT:   %.loc21_33.5: ref i32 = class_element_access %.loc21_33.2, element1
// CHECK:STDOUT:   %.loc21_33.6: init i32 = initialize_from %.loc21_32 to %.loc21_33.5 [template = constants.%.6]
// CHECK:STDOUT:   %.loc21_33.7: init Class = class_init (%.loc21_33.4, %.loc21_33.6), %.loc21_33.2 [template = constants.%.7]
// CHECK:STDOUT:   %.loc21_33.8: ref Class = temporary %.loc21_33.2, %.loc21_33.7
// CHECK:STDOUT:   %.loc21_33.9: ref Class = converted %.loc21_33.1, %.loc21_33.8
// CHECK:STDOUT:   %.loc21_33.10: Class = bind_value %.loc21_33.9
// CHECK:STDOUT:   assign %c.var, <error>
// CHECK:STDOUT:   %G.ref: <function> = name_ref G, file.%G [template = file.%G]
// CHECK:STDOUT:   %c.ref: ref Class = name_ref c, %c
// CHECK:STDOUT:   %.loc22_5: Class* = addr_of %c.ref
// CHECK:STDOUT:   %.loc22_4: init () = call %G.ref(%.loc22_5)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

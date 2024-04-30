// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class Class {
  var a: i32;
  var b: i32;
}

fn F() {
  // CHECK:STDERR: fail_init.carbon:[[@LINE+4]]:3: ERROR: Cannot initialize class with 2 field(s) from struct with 1 field(s).
  // CHECK:STDERR:   {.a = 1} as Class;
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR:
  {.a = 1} as Class;
  // CHECK:STDERR: fail_init.carbon:[[@LINE+4]]:3: ERROR: Missing value for field `b` in struct initialization.
  // CHECK:STDERR:   {.a = 1, .c = 2} as Class;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  {.a = 1, .c = 2} as Class;
  // CHECK:STDERR: fail_init.carbon:[[@LINE+3]]:3: ERROR: Cannot initialize class with 2 field(s) from struct with 3 field(s).
  // CHECK:STDERR:   {.a = 1, .b = 2, .c = 3} as Class;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~
  {.a = 1, .b = 2, .c = 3} as Class;
}

// CHECK:STDOUT: --- fail_init.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [template]
// CHECK:STDOUT:   %.1: type = unbound_element_type Class, i32 [template]
// CHECK:STDOUT:   %.2: type = struct_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT:   %F: type = fn_type @F [template]
// CHECK:STDOUT:   %.3: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: F = struct_value () [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.5: type = struct_type {.a: i32} [template]
// CHECK:STDOUT:   %.6: type = ptr_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT:   %.7: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.8: type = struct_type {.a: i32, .c: i32} [template]
// CHECK:STDOUT:   %.9: i32 = int_literal 3 [template]
// CHECK:STDOUT:   %.10: type = struct_type {.a: i32, .b: i32, .c: i32} [template]
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
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct] {}
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
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc17_9: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc17_10.1: {.a: i32} = struct_literal (%.loc17_9)
// CHECK:STDOUT:   %Class.ref.loc17: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:   %.loc17_10.2: ref Class = temporary_storage
// CHECK:STDOUT:   %.loc17_10.3: ref Class = temporary %.loc17_10.2, <error>
// CHECK:STDOUT:   %.loc17_12: ref Class = converted %.loc17_10.1, %.loc17_10.3
// CHECK:STDOUT:   %.loc22_9: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc22_17: i32 = int_literal 2 [template = constants.%.7]
// CHECK:STDOUT:   %.loc22_18.1: {.a: i32, .c: i32} = struct_literal (%.loc22_9, %.loc22_17)
// CHECK:STDOUT:   %Class.ref.loc22: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:   %.loc22_18.2: ref Class = temporary_storage
// CHECK:STDOUT:   %.loc22_18.3: ref i32 = class_element_access %.loc22_18.2, element0
// CHECK:STDOUT:   %.loc22_18.4: init i32 = initialize_from %.loc22_9 to %.loc22_18.3 [template = constants.%.4]
// CHECK:STDOUT:   %.loc22_18.5: ref Class = temporary %.loc22_18.2, <error>
// CHECK:STDOUT:   %.loc22_20: ref Class = converted %.loc22_18.1, %.loc22_18.5
// CHECK:STDOUT:   %.loc26_9: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc26_17: i32 = int_literal 2 [template = constants.%.7]
// CHECK:STDOUT:   %.loc26_25: i32 = int_literal 3 [template = constants.%.9]
// CHECK:STDOUT:   %.loc26_26.1: {.a: i32, .b: i32, .c: i32} = struct_literal (%.loc26_9, %.loc26_17, %.loc26_25)
// CHECK:STDOUT:   %Class.ref.loc26: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:   %.loc26_26.2: ref Class = temporary_storage
// CHECK:STDOUT:   %.loc26_26.3: ref Class = temporary %.loc26_26.2, <error>
// CHECK:STDOUT:   %.loc26_28: ref Class = converted %.loc26_26.1, %.loc26_26.3
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

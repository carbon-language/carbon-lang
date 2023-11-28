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
  // CHECK:STDERR: fail_init.carbon:[[@LINE+3]]:3: ERROR: Cannot initialize class with 2 field(s) from struct with 1 field(s).
  // CHECK:STDERR:   {.a = 1} as Class;
  // CHECK:STDERR:   ^~~~~~~~
  {.a = 1} as Class;
  // CHECK:STDERR: fail_init.carbon:[[@LINE+3]]:3: ERROR: Missing value for field `b` in struct initialization.
  // CHECK:STDERR:   {.a = 1, .c = 2} as Class;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
  {.a = 1, .c = 2} as Class;
  // CHECK:STDERR: fail_init.carbon:[[@LINE+3]]:3: ERROR: Cannot initialize class with 2 field(s) from struct with 3 field(s).
  // CHECK:STDERR:   {.a = 1, .b = 2, .c = 3} as Class;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~
  {.a = 1, .b = 2, .c = 3} as Class;
}

// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc10_1.1: type = struct_type {.a: i32, .b: i32}
// CHECK:STDOUT:   %.loc16: type = struct_type {.a: i32}
// CHECK:STDOUT:   %.loc10_1.2: type = ptr_type {.a: i32, .b: i32}
// CHECK:STDOUT:   %.loc20: type = struct_type {.a: i32, .c: i32}
// CHECK:STDOUT:   %.loc24: type = struct_type {.a: i32, .b: i32, .c: i32}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file "fail_init.carbon" {
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
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc16_9: i32 = int_literal 1
// CHECK:STDOUT:   %.loc16_10.1: {.a: i32} = struct_literal (%.loc16_9)
// CHECK:STDOUT:   %Class.ref.loc16: type = name_ref Class, file.%Class
// CHECK:STDOUT:   %.loc16_10.2: ref Class = temporary_storage
// CHECK:STDOUT:   %.loc16_10.3: ref Class = temporary %.loc16_10.2, <error>
// CHECK:STDOUT:   %.loc16_10.4: ref Class = converted %.loc16_10.1, %.loc16_10.3
// CHECK:STDOUT:   %.loc20_9: i32 = int_literal 1
// CHECK:STDOUT:   %.loc20_17: i32 = int_literal 2
// CHECK:STDOUT:   %.loc20_18.1: {.a: i32, .c: i32} = struct_literal (%.loc20_9, %.loc20_17)
// CHECK:STDOUT:   %Class.ref.loc20: type = name_ref Class, file.%Class
// CHECK:STDOUT:   %.loc20_18.2: ref Class = temporary_storage
// CHECK:STDOUT:   %.loc20_18.3: ref i32 = class_field_access %.loc20_18.2, member0
// CHECK:STDOUT:   %.loc20_18.4: init i32 = initialize_from %.loc20_9 to %.loc20_18.3
// CHECK:STDOUT:   %.loc20_18.5: ref Class = temporary %.loc20_18.2, <error>
// CHECK:STDOUT:   %.loc20_18.6: ref Class = converted %.loc20_18.1, %.loc20_18.5
// CHECK:STDOUT:   %.loc24_9: i32 = int_literal 1
// CHECK:STDOUT:   %.loc24_17: i32 = int_literal 2
// CHECK:STDOUT:   %.loc24_25: i32 = int_literal 3
// CHECK:STDOUT:   %.loc24_26.1: {.a: i32, .b: i32, .c: i32} = struct_literal (%.loc24_9, %.loc24_17, %.loc24_25)
// CHECK:STDOUT:   %Class.ref.loc24: type = name_ref Class, file.%Class
// CHECK:STDOUT:   %.loc24_26.2: ref Class = temporary_storage
// CHECK:STDOUT:   %.loc24_26.3: ref Class = temporary %.loc24_26.2, <error>
// CHECK:STDOUT:   %.loc24_26.4: ref Class = converted %.loc24_26.1, %.loc24_26.3
// CHECK:STDOUT:   return
// CHECK:STDOUT: }

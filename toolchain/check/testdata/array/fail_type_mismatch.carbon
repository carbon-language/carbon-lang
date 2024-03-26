// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+4]]:19: ERROR: Cannot implicitly convert from `String` to `i32`.
// CHECK:STDERR: var a: [i32; 3] = (1, "Hello", "World");
// CHECK:STDERR:                   ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var a: [i32; 3] = (1, "Hello", "World");

var t1: (i32, String, String);
// CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+4]]:19: ERROR: Cannot implicitly convert from `String` to `i32`.
// CHECK:STDERR: var b: [i32; 3] = t1;
// CHECK:STDERR:                   ^~
// CHECK:STDERR:
var b: [i32; 3] = t1;

// CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+4]]:19: ERROR: Cannot initialize array of 3 element(s) from 2 initializer(s).
// CHECK:STDERR: var c: [i32; 3] = (1, 2);
// CHECK:STDERR:                   ^~~~~~
// CHECK:STDERR:
var c: [i32; 3] = (1, 2);

var t2: (i32, i32);
// CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+3]]:19: ERROR: Cannot initialize array of 3 element(s) from tuple with 2 element(s).
// CHECK:STDERR: var d: [i32; 3] = t2;
// CHECK:STDERR:                   ^~
var d: [i32; 3] = t2;

// CHECK:STDOUT: --- fail_type_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 3 [template]
// CHECK:STDOUT:   %.2: type = array_type %.1, i32 [template]
// CHECK:STDOUT:   %.3: type = ptr_type [i32; 3] [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.5: type = ptr_type String [template]
// CHECK:STDOUT:   %.6: String = string_literal "Hello" [template]
// CHECK:STDOUT:   %.7: String = string_literal "World" [template]
// CHECK:STDOUT:   %.8: type = tuple_type (i32, String, String) [template]
// CHECK:STDOUT:   %.9: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.10: type = tuple_type (type, type, type) [template]
// CHECK:STDOUT:   %.11: type = tuple_type (i32, String*, String*) [template]
// CHECK:STDOUT:   %.12: type = ptr_type (i32, String*, String*) [template]
// CHECK:STDOUT:   %.13: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.14: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT:   %.15: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %.16: type = ptr_type (i32, i32) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .t1 = %t1
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:     .t2 = %t2
// CHECK:STDOUT:     .d = %d
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc11_14: i32 = int_literal 3 [template = constants.%.1]
// CHECK:STDOUT:   %.loc11_15: type = array_type %.loc11_14, i32 [template = constants.%.2]
// CHECK:STDOUT:   %a.var: ref [i32; 3] = var a
// CHECK:STDOUT:   %a: ref [i32; 3] = bind_name a, %a.var
// CHECK:STDOUT:   %.loc13_29.1: (type, type, type) = tuple_literal (i32, String, String)
// CHECK:STDOUT:   %.loc13_29.2: type = converted %.loc13_29.1, constants.%.8 [template = constants.%.8]
// CHECK:STDOUT:   %t1.var: ref (i32, String, String) = var t1
// CHECK:STDOUT:   %t1: ref (i32, String, String) = bind_name t1, %t1.var
// CHECK:STDOUT:   %.loc18_14: i32 = int_literal 3 [template = constants.%.1]
// CHECK:STDOUT:   %.loc18_15: type = array_type %.loc18_14, i32 [template = constants.%.2]
// CHECK:STDOUT:   %b.var: ref [i32; 3] = var b
// CHECK:STDOUT:   %b: ref [i32; 3] = bind_name b, %b.var
// CHECK:STDOUT:   %.loc24_14: i32 = int_literal 3 [template = constants.%.1]
// CHECK:STDOUT:   %.loc24_15: type = array_type %.loc24_14, i32 [template = constants.%.2]
// CHECK:STDOUT:   %c.var: ref [i32; 3] = var c
// CHECK:STDOUT:   %c: ref [i32; 3] = bind_name c, %c.var
// CHECK:STDOUT:   %.loc26_18.1: (type, type) = tuple_literal (i32, i32)
// CHECK:STDOUT:   %.loc26_18.2: type = converted %.loc26_18.1, constants.%.14 [template = constants.%.14]
// CHECK:STDOUT:   %t2.var: ref (i32, i32) = var t2
// CHECK:STDOUT:   %t2: ref (i32, i32) = bind_name t2, %t2.var
// CHECK:STDOUT:   %.loc30_14: i32 = int_literal 3 [template = constants.%.1]
// CHECK:STDOUT:   %.loc30_15: type = array_type %.loc30_14, i32 [template = constants.%.2]
// CHECK:STDOUT:   %d.var: ref [i32; 3] = var d
// CHECK:STDOUT:   %d: ref [i32; 3] = bind_name d, %d.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_20: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc11_23: String = string_literal "Hello" [template = constants.%.6]
// CHECK:STDOUT:   %.loc11_32: String = string_literal "World" [template = constants.%.7]
// CHECK:STDOUT:   %.loc11_39.1: (i32, String, String) = tuple_literal (%.loc11_20, %.loc11_23, %.loc11_32)
// CHECK:STDOUT:   %.loc11_39.2: i32 = int_literal 0 [template = constants.%.9]
// CHECK:STDOUT:   %.loc11_39.3: ref i32 = array_index file.%a.var, %.loc11_39.2
// CHECK:STDOUT:   %.loc11_39.4: init i32 = initialize_from %.loc11_20 to %.loc11_39.3 [template = constants.%.4]
// CHECK:STDOUT:   assign file.%a.var, <error>
// CHECK:STDOUT:   %t1.ref: ref (i32, String, String) = name_ref t1, file.%t1
// CHECK:STDOUT:   %.loc18_19.1: ref i32 = tuple_access %t1.ref, element0
// CHECK:STDOUT:   %.loc18_19.2: i32 = bind_value %.loc18_19.1
// CHECK:STDOUT:   %.loc18_19.3: i32 = int_literal 0 [template = constants.%.9]
// CHECK:STDOUT:   %.loc18_19.4: ref i32 = array_index file.%b.var, %.loc18_19.3
// CHECK:STDOUT:   %.loc18_19.5: init i32 = initialize_from %.loc18_19.2 to %.loc18_19.4
// CHECK:STDOUT:   %.loc18_19.6: ref String = tuple_access %t1.ref, element1
// CHECK:STDOUT:   assign file.%b.var, <error>
// CHECK:STDOUT:   %.loc24_20: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc24_23: i32 = int_literal 2 [template = constants.%.13]
// CHECK:STDOUT:   %.loc24_24: (i32, i32) = tuple_literal (%.loc24_20, %.loc24_23)
// CHECK:STDOUT:   assign file.%c.var, <error>
// CHECK:STDOUT:   %t2.ref: ref (i32, i32) = name_ref t2, file.%t2
// CHECK:STDOUT:   assign file.%d.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

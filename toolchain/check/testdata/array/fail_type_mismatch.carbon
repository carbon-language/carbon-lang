// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+3]]:39: ERROR: Cannot implicitly convert from `String` to `i32`.
// CHECK:STDERR: var a: [i32; 3] = (1, "Hello", "World");
// CHECK:STDERR:                                       ^
var a: [i32; 3] = (1, "Hello", "World");

var t1: (i32, String, String);
// CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+3]]:19: ERROR: Cannot implicitly convert from `String` to `i32`.
// CHECK:STDERR: var b: [i32; 3] = t1;
// CHECK:STDERR:                   ^
var b: [i32; 3] = t1;

// CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+3]]:24: ERROR: Cannot initialize array of 3 element(s) from 2 initializer(s).
// CHECK:STDERR: var c: [i32; 3] = (1, 2);
// CHECK:STDERR:                        ^
var c: [i32; 3] = (1, 2);

var t2: (i32, i32);
// CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+3]]:19: ERROR: Cannot initialize array of 3 element(s) from tuple with 2 element(s).
// CHECK:STDERR: var d: [i32; 3] = t2;
// CHECK:STDERR:                   ^
var d: [i32; 3] = t2;

// CHECK:STDOUT: file "fail_type_mismatch.carbon" {
// CHECK:STDOUT:   %.loc10_14: i32 = int_literal 3
// CHECK:STDOUT:   %.loc10_15.1: type = array_type %.loc10_14, i32
// CHECK:STDOUT:   %.loc10_15.2: type = ptr_type [i32; 3]
// CHECK:STDOUT:   %a: ref [i32; 3] = var "a"
// CHECK:STDOUT:   %.loc10_20: i32 = int_literal 1
// CHECK:STDOUT:   %.1: type = ptr_type String
// CHECK:STDOUT:   %.loc10_23: String = string_literal "Hello"
// CHECK:STDOUT:   %.loc10_32: String = string_literal "World"
// CHECK:STDOUT:   %.loc10_39.1: type = tuple_type (i32, String, String)
// CHECK:STDOUT:   %.loc10_39.2: (i32, String, String) = tuple_literal (%.loc10_20, %.loc10_23, %.loc10_32)
// CHECK:STDOUT:   %.loc10_39.3: i32 = int_literal 0
// CHECK:STDOUT:   %.loc10_39.4: ref i32 = array_index %a, %.loc10_39.3
// CHECK:STDOUT:   %.loc10_39.5: init i32 = initialize_from %.loc10_20 to %.loc10_39.4
// CHECK:STDOUT:   assign %a, <error>
// CHECK:STDOUT:   %.loc12_29.1: type = tuple_type (type, type, type)
// CHECK:STDOUT:   %.loc12_29.2: (type, type, type) = tuple_literal (i32, String, String)
// CHECK:STDOUT:   %.loc10_39.6: type = tuple_type (i32, String*, String*)
// CHECK:STDOUT:   %.loc10_39.7: type = ptr_type (i32, String*, String*)
// CHECK:STDOUT:   %t1: ref (i32, String, String) = var "t1"
// CHECK:STDOUT:   %.loc16_14: i32 = int_literal 3
// CHECK:STDOUT:   %.loc16_15: type = array_type %.loc16_14, i32
// CHECK:STDOUT:   %b: ref [i32; 3] = var "b"
// CHECK:STDOUT:   %t1.ref: ref (i32, String, String) = name_reference "t1", %t1
// CHECK:STDOUT:   %.loc16_19.1: ref i32 = tuple_access %t1.ref, member0
// CHECK:STDOUT:   %.loc16_19.2: i32 = bind_value %.loc16_19.1
// CHECK:STDOUT:   %.loc16_19.3: i32 = int_literal 0
// CHECK:STDOUT:   %.loc16_19.4: ref i32 = array_index %b, %.loc16_19.3
// CHECK:STDOUT:   %.loc16_19.5: init i32 = initialize_from %.loc16_19.2 to %.loc16_19.4
// CHECK:STDOUT:   %.loc16_19.6: ref String = tuple_access %t1.ref, member1
// CHECK:STDOUT:   assign %b, <error>
// CHECK:STDOUT:   %.loc21_14: i32 = int_literal 3
// CHECK:STDOUT:   %.loc21_15: type = array_type %.loc21_14, i32
// CHECK:STDOUT:   %c: ref [i32; 3] = var "c"
// CHECK:STDOUT:   %.loc21_20: i32 = int_literal 1
// CHECK:STDOUT:   %.loc21_23: i32 = int_literal 2
// CHECK:STDOUT:   %.loc21_24.1: type = tuple_type (i32, i32)
// CHECK:STDOUT:   %.loc21_24.2: (i32, i32) = tuple_literal (%.loc21_20, %.loc21_23)
// CHECK:STDOUT:   assign %c, <error>
// CHECK:STDOUT:   %.loc23_18.1: type = tuple_type (type, type)
// CHECK:STDOUT:   %.loc23_18.2: (type, type) = tuple_literal (i32, i32)
// CHECK:STDOUT:   %.loc21_24.3: type = ptr_type (i32, i32)
// CHECK:STDOUT:   %t2: ref (i32, i32) = var "t2"
// CHECK:STDOUT:   %.loc27_14: i32 = int_literal 3
// CHECK:STDOUT:   %.loc27_15: type = array_type %.loc27_14, i32
// CHECK:STDOUT:   %d: ref [i32; 3] = var "d"
// CHECK:STDOUT:   %t2.ref: ref (i32, i32) = name_reference "t2", %t2
// CHECK:STDOUT:   assign %d, <error>
// CHECK:STDOUT: }

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+3]]:39: Cannot implicitly convert from `String` to `i32`.
// CHECK:STDERR: var a: [i32; 3] = (1, "Hello", "World");
// CHECK:STDERR:                                       ^
var a: [i32; 3] = (1, "Hello", "World");

// CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+3]]:5: Cannot implicitly convert from `String` to `i32`.
// CHECK:STDERR: var t1: (i32, String, String);
// CHECK:STDERR:     ^
var t1: (i32, String, String);
var b: [i32; 3] = t1;

// CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+3]]:24: Cannot initialize array of 3 element(s) from 2 initializer(s).
// CHECK:STDERR: var c: [i32; 3] = (1, 2);
// CHECK:STDERR:                        ^
var c: [i32; 3] = (1, 2);

// CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+3]]:5: Cannot initialize array of 3 element(s) from tuple with 2 element(s).
// CHECK:STDERR: var t2: (i32, i32);
// CHECK:STDERR:     ^
var t2: (i32, i32);
var d: [i32; 3] = t2;

// CHECK:STDOUT: file "fail_type_mismatch.carbon" {
// CHECK:STDOUT:   %.loc10_14: i32 = int_literal 3
// CHECK:STDOUT:   %.loc10_15: type = array_type %.loc10_14, i32
// CHECK:STDOUT:   %a: ref [i32; 3] = var "a"
// CHECK:STDOUT:   %.loc10_20.1: i32 = int_literal 1
// CHECK:STDOUT:   %.loc10_20.2: i32 = stub_reference %.loc10_20.1
// CHECK:STDOUT:   %.loc10_23.1: String = string_literal "Hello"
// CHECK:STDOUT:   %.loc10_23.2: String = stub_reference %.loc10_23.1
// CHECK:STDOUT:   %.loc10_32.1: String = string_literal "World"
// CHECK:STDOUT:   %.loc10_32.2: String = stub_reference %.loc10_32.1
// CHECK:STDOUT:   %.loc10_39.1: type = tuple_type (i32, String, String)
// CHECK:STDOUT:   %.loc10_39.2: (i32, String, String) = tuple_literal (%.loc10_20.2, %.loc10_23.2, %.loc10_32.2)
// CHECK:STDOUT:   %.loc10_39.3: ref [i32; 3] = temporary_storage
// CHECK:STDOUT:   %.loc10_39.4: i32 = int_literal 0
// CHECK:STDOUT:   %.loc10_39.5: ref i32 = array_index %.loc10_39.3, %.loc10_39.4
// CHECK:STDOUT:   %.loc10_39.6: init i32 = initialize_from %.loc10_20.2 to %.loc10_39.5
// CHECK:STDOUT:   %.loc10_39.7: i32 = int_literal 1
// CHECK:STDOUT:   %.loc10_39.8: ref i32 = array_index %.loc10_39.3, %.loc10_39.7
// CHECK:STDOUT:   assign %a, <error>
// CHECK:STDOUT:   %.loc15_10: type = stub_reference i32
// CHECK:STDOUT:   %.loc15_15: type = stub_reference String
// CHECK:STDOUT:   %.loc15_23: type = stub_reference String
// CHECK:STDOUT:   %.loc15_29.1: type = tuple_type (type, type, type)
// CHECK:STDOUT:   %.loc15_29.2: (type, type, type) = tuple_literal (%.loc15_10, %.loc15_15, %.loc15_23)
// CHECK:STDOUT:   %t1: ref (i32, String, String) = var "t1"
// CHECK:STDOUT:   %.loc16_14: i32 = int_literal 3
// CHECK:STDOUT:   %.loc16_15: type = array_type %.loc16_14, i32
// CHECK:STDOUT:   %b: ref [i32; 3] = var "b"
// CHECK:STDOUT:   %.loc15_5.1: ref [i32; 3] = temporary_storage
// CHECK:STDOUT:   %.loc15_5.2: i32 = int_literal 0
// CHECK:STDOUT:   %.loc15_5.3: ref i32 = array_index %.loc15_5.1, %.loc15_5.2
// CHECK:STDOUT:   %.loc15_5.4: ref i32 = tuple_index %t1, %.loc15_5.2
// CHECK:STDOUT:   %.loc15_5.5: i32 = bind_value %.loc15_5.4
// CHECK:STDOUT:   %.loc15_5.6: init i32 = initialize_from %.loc15_5.5 to %.loc15_5.3
// CHECK:STDOUT:   %.loc15_5.7: i32 = int_literal 1
// CHECK:STDOUT:   %.loc15_5.8: ref i32 = array_index %.loc15_5.1, %.loc15_5.7
// CHECK:STDOUT:   %.loc15_5.9: ref String = tuple_index %t1, %.loc15_5.7
// CHECK:STDOUT:   assign %b, <error>
// CHECK:STDOUT:   %.loc21_14: i32 = int_literal 3
// CHECK:STDOUT:   %.loc21_15: type = array_type %.loc21_14, i32
// CHECK:STDOUT:   %c: ref [i32; 3] = var "c"
// CHECK:STDOUT:   %.loc21_20.1: i32 = int_literal 1
// CHECK:STDOUT:   %.loc21_20.2: i32 = stub_reference %.loc21_20.1
// CHECK:STDOUT:   %.loc21_23.1: i32 = int_literal 2
// CHECK:STDOUT:   %.loc21_23.2: i32 = stub_reference %.loc21_23.1
// CHECK:STDOUT:   %.loc21_24.1: type = tuple_type (i32, i32)
// CHECK:STDOUT:   %.loc21_24.2: (i32, i32) = tuple_literal (%.loc21_20.2, %.loc21_23.2)
// CHECK:STDOUT:   assign %c, <error>
// CHECK:STDOUT:   %.loc26_10: type = stub_reference i32
// CHECK:STDOUT:   %.loc26_15: type = stub_reference i32
// CHECK:STDOUT:   %.loc26_18.1: type = tuple_type (type, type)
// CHECK:STDOUT:   %.loc26_18.2: (type, type) = tuple_literal (%.loc26_10, %.loc26_15)
// CHECK:STDOUT:   %t2: ref (i32, i32) = var "t2"
// CHECK:STDOUT:   %.loc27_14: i32 = int_literal 3
// CHECK:STDOUT:   %.loc27_15: type = array_type %.loc27_14, i32
// CHECK:STDOUT:   %d: ref [i32; 3] = var "d"
// CHECK:STDOUT:   assign %d, <error>
// CHECK:STDOUT: }

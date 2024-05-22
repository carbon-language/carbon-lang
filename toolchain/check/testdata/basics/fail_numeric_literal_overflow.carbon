// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_numeric_literal_overflow.carbon:[[@LINE+4]]:14: ERROR: Integer literal with value 39999999999999999993 does not fit in i32.
// CHECK:STDERR: let a: i32 = 39999999999999999993;
// CHECK:STDERR:              ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let a: i32 = 39999999999999999993;

// CHECK:STDERR: fail_numeric_literal_overflow.carbon:[[@LINE+4]]:14: ERROR: Integer literal with value 2147483648 does not fit in i32.
// CHECK:STDERR: let b: i32 = 2_147_483_648;
// CHECK:STDERR:              ^~~~~~~~~~~~~
// CHECK:STDERR:
let b: i32 = 2_147_483_648;

// CHECK:STDERR: fail_numeric_literal_overflow.carbon:[[@LINE+4]]:14: ERROR: Integer literal with value 2147483648 does not fit in i32.
// CHECK:STDERR: let c: i32 = 0x8000_0000;
// CHECK:STDERR:              ^~~~~~~~~~~
// CHECK:STDERR:
let c: i32 = 0x8000_0000;

// CHECK:STDERR: fail_numeric_literal_overflow.carbon:[[@LINE+4]]:14: ERROR: Real mantissa with value 399999999999999999930 does not fit in i64.
// CHECK:STDERR: let d: f64 = 39999999999999999993.0e3;
// CHECK:STDERR:              ^~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let d: f64 = 39999999999999999993.0e3;

// CHECK:STDERR: fail_numeric_literal_overflow.carbon:[[@LINE+3]]:14: ERROR: Real exponent with value 39999999999999999992 does not fit in i64.
// CHECK:STDERR: let e: f64 = 5.0e39999999999999999993;
// CHECK:STDERR:              ^~~~~~~~~~~~~~~~~~~~~~~~
let e: f64 = 5.0e39999999999999999993;

// CHECK:STDOUT: --- fail_numeric_literal_overflow.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Int32 = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 64 [template]
// CHECK:STDOUT:   %Float: type = fn_type @Float [template]
// CHECK:STDOUT:   %struct.2: Float = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %import_ref.1: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %int.make_type_32.loc11: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %.loc11_8.1: type = value_of_initializer %int.make_type_32.loc11 [template = i32]
// CHECK:STDOUT:   %.loc11_8.2: type = converted %int.make_type_32.loc11, %.loc11_8.1 [template = i32]
// CHECK:STDOUT:   %a: i32 = bind_name a, <error>
// CHECK:STDOUT:   %import_ref.2: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %int.make_type_32.loc17: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %.loc17_8.1: type = value_of_initializer %int.make_type_32.loc17 [template = i32]
// CHECK:STDOUT:   %.loc17_8.2: type = converted %int.make_type_32.loc17, %.loc17_8.1 [template = i32]
// CHECK:STDOUT:   %b: i32 = bind_name b, <error>
// CHECK:STDOUT:   %import_ref.3: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %int.make_type_32.loc23: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %.loc23_8.1: type = value_of_initializer %int.make_type_32.loc23 [template = i32]
// CHECK:STDOUT:   %.loc23_8.2: type = converted %int.make_type_32.loc23, %.loc23_8.1 [template = i32]
// CHECK:STDOUT:   %c: i32 = bind_name c, <error>
// CHECK:STDOUT:   %.loc29_8.1: i32 = int_literal 64 [template = constants.%.2]
// CHECK:STDOUT:   %import_ref.4: Float = import_ref ir4, inst+30, loaded [template = constants.%struct.2]
// CHECK:STDOUT:   %float.make_type.loc29: init type = call constants.%struct.2(%.loc29_8.1) [template = f64]
// CHECK:STDOUT:   %.loc29_8.2: type = value_of_initializer %float.make_type.loc29 [template = f64]
// CHECK:STDOUT:   %.loc29_8.3: type = converted %float.make_type.loc29, %.loc29_8.2 [template = f64]
// CHECK:STDOUT:   %d: f64 = bind_name d, <error>
// CHECK:STDOUT:   %.loc34_8.1: i32 = int_literal 64 [template = constants.%.2]
// CHECK:STDOUT:   %import_ref.5: Float = import_ref ir4, inst+30, loaded [template = constants.%struct.2]
// CHECK:STDOUT:   %float.make_type.loc34: init type = call constants.%struct.2(%.loc34_8.1) [template = f64]
// CHECK:STDOUT:   %.loc34_8.2: type = value_of_initializer %float.make_type.loc34 [template = f64]
// CHECK:STDOUT:   %.loc34_8.3: type = converted %float.make_type.loc34, %.loc34_8.2 [template = f64]
// CHECK:STDOUT:   %e: f64 = bind_name e, <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Float(%size: i32) -> type = "float.make_type";
// CHECK:STDOUT:

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/basics/fail_numeric_literal_overflow.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/basics/fail_numeric_literal_overflow.carbon

// CHECK:STDERR: fail_numeric_literal_overflow.carbon:[[@LINE+4]]:1: error: integer value 39999999999999999993 too large for type `i32` [IntTooLargeForType]
// CHECK:STDERR: let a: i32 = 39999999999999999993;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let a: i32 = 39999999999999999993;

// CHECK:STDERR: fail_numeric_literal_overflow.carbon:[[@LINE+4]]:1: error: integer value 2147483648 too large for type `i32` [IntTooLargeForType]
// CHECK:STDERR: let b: i32 = 2_147_483_648;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let b: i32 = 2_147_483_648;

// CHECK:STDERR: fail_numeric_literal_overflow.carbon:[[@LINE+4]]:1: error: integer value 2147483648 too large for type `i32` [IntTooLargeForType]
// CHECK:STDERR: let c: i32 = 0x8000_0000;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let c: i32 = 0x8000_0000;

// CHECK:STDERR: fail_numeric_literal_overflow.carbon:[[@LINE+4]]:14: error: real mantissa with value 399999999999999999930 does not fit in i64 [RealMantissaTooLargeForI64]
// CHECK:STDERR: let d: f64 = 39999999999999999993.0e3;
// CHECK:STDERR:              ^~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let d: f64 = 39999999999999999993.0e3;

// CHECK:STDERR: fail_numeric_literal_overflow.carbon:[[@LINE+3]]:14: error: real exponent with value 39999999999999999992 does not fit in i64 [RealExponentTooLargeForI64]
// CHECK:STDERR: let e: f64 = 5.0e39999999999999999993;
// CHECK:STDERR:              ^~~~~~~~~~~~~~~~~~~~~~~~
let e: f64 = 5.0e39999999999999999993;

// CHECK:STDOUT: --- fail_numeric_literal_overflow.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 39999999999999999993 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(i32) [template]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.11 [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %.25: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %.26: <bound method> = bound_method %.1, %Convert.15 [template]
// CHECK:STDOUT:   %.27: i32 = int_value 39999999999999999993 [template]
// CHECK:STDOUT:   %.28: Core.IntLiteral = int_value 2147483648 [template]
// CHECK:STDOUT:   %.29: <bound method> = bound_method %.28, %Convert.15 [template]
// CHECK:STDOUT:   %.30: i32 = int_value 2147483648 [template]
// CHECK:STDOUT:   %.31: Core.IntLiteral = int_value 64 [template]
// CHECK:STDOUT:   %Float.type: type = fn_type @Float [template]
// CHECK:STDOUT:   %Float: %Float.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     .Float = %import_ref.51
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .a = @__global_init.%a
// CHECK:STDOUT:     .b = @__global_init.%b
// CHECK:STDOUT:     .c = @__global_init.%c
// CHECK:STDOUT:     .d = @__global_init.%d
// CHECK:STDOUT:     .e = @__global_init.%e
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %int.make_type_32.loc15: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc15_8.1: type = value_of_initializer %int.make_type_32.loc15 [template = i32]
// CHECK:STDOUT:   %.loc15_8.2: type = converted %int.make_type_32.loc15, %.loc15_8.1 [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc21: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc21_8.1: type = value_of_initializer %int.make_type_32.loc21 [template = i32]
// CHECK:STDOUT:   %.loc21_8.2: type = converted %int.make_type_32.loc21, %.loc21_8.1 [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc27: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc27_8.1: type = value_of_initializer %int.make_type_32.loc27 [template = i32]
// CHECK:STDOUT:   %.loc27_8.2: type = converted %int.make_type_32.loc27, %.loc27_8.1 [template = i32]
// CHECK:STDOUT:   %.loc33_8.1: Core.IntLiteral = int_value 64 [template = constants.%.31]
// CHECK:STDOUT:   %float.make_type.loc33: init type = call constants.%Float(%.loc33_8.1) [template = f64]
// CHECK:STDOUT:   %.loc33_8.2: type = value_of_initializer %float.make_type.loc33 [template = f64]
// CHECK:STDOUT:   %.loc33_8.3: type = converted %float.make_type.loc33, %.loc33_8.2 [template = f64]
// CHECK:STDOUT:   %.loc38_8.1: Core.IntLiteral = int_value 64 [template = constants.%.31]
// CHECK:STDOUT:   %float.make_type.loc38: init type = call constants.%Float(%.loc38_8.1) [template = f64]
// CHECK:STDOUT:   %.loc38_8.2: type = value_of_initializer %float.make_type.loc38 [template = f64]
// CHECK:STDOUT:   %.loc38_8.3: type = converted %float.make_type.loc38, %.loc38_8.2 [template = f64]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc15_14: Core.IntLiteral = int_value 39999999999999999993 [template = constants.%.1]
// CHECK:STDOUT:   %.loc15_34.1: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc15_34.2: <bound method> = bound_method %.loc15_14, %.loc15_34.1 [template = constants.%.26]
// CHECK:STDOUT:   %int.convert_checked.loc15: init i32 = call %.loc15_34.2(%.loc15_14) [template = constants.%.27]
// CHECK:STDOUT:   %.loc15_34.3: i32 = value_of_initializer %int.convert_checked.loc15 [template = constants.%.27]
// CHECK:STDOUT:   %.loc15_34.4: i32 = converted %.loc15_14, %.loc15_34.3 [template = constants.%.27]
// CHECK:STDOUT:   %a: i32 = bind_name a, %.loc15_34.4
// CHECK:STDOUT:   %.loc21_14: Core.IntLiteral = int_value 2147483648 [template = constants.%.28]
// CHECK:STDOUT:   %.loc21_27.1: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc21_27.2: <bound method> = bound_method %.loc21_14, %.loc21_27.1 [template = constants.%.29]
// CHECK:STDOUT:   %int.convert_checked.loc21: init i32 = call %.loc21_27.2(%.loc21_14) [template = constants.%.30]
// CHECK:STDOUT:   %.loc21_27.3: i32 = value_of_initializer %int.convert_checked.loc21 [template = constants.%.30]
// CHECK:STDOUT:   %.loc21_27.4: i32 = converted %.loc21_14, %.loc21_27.3 [template = constants.%.30]
// CHECK:STDOUT:   %b: i32 = bind_name b, %.loc21_27.4
// CHECK:STDOUT:   %.loc27_14: Core.IntLiteral = int_value 2147483648 [template = constants.%.28]
// CHECK:STDOUT:   %.loc27_25.1: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc27_25.2: <bound method> = bound_method %.loc27_14, %.loc27_25.1 [template = constants.%.29]
// CHECK:STDOUT:   %int.convert_checked.loc27: init i32 = call %.loc27_25.2(%.loc27_14) [template = constants.%.30]
// CHECK:STDOUT:   %.loc27_25.3: i32 = value_of_initializer %int.convert_checked.loc27 [template = constants.%.30]
// CHECK:STDOUT:   %.loc27_25.4: i32 = converted %.loc27_14, %.loc27_25.3 [template = constants.%.30]
// CHECK:STDOUT:   %c: i32 = bind_name c, %.loc27_25.4
// CHECK:STDOUT:   %d: f64 = bind_name d, <error>
// CHECK:STDOUT:   %e: f64 = bind_name e, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

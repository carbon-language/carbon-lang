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
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %int_39999999999999999993.af6: Core.IntLiteral = int_value 39999999999999999993 [template]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %impl_witness.d39: <witness> = impl_witness (imports.%import_ref.a5b), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [template]
// CHECK:STDOUT:   %Convert.bound.7ef: <bound method> = bound_method %int_39999999999999999993.af6, %Convert.956 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2ed: <specific function> = specific_function %Convert.bound.7ef, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_39999999999999999993.dee: %i32 = int_value 39999999999999999993 [template]
// CHECK:STDOUT:   %int_2147483648.1db: Core.IntLiteral = int_value 2147483648 [template]
// CHECK:STDOUT:   %Convert.bound.85f: <bound method> = bound_method %int_2147483648.1db, %Convert.956 [template]
// CHECK:STDOUT:   %Convert.specific_fn.387: <specific function> = specific_function %Convert.bound.85f, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2147483648.8df: %i32 = int_value 2147483648 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.485
// CHECK:STDOUT:     .ImplicitAs = %import_ref.d44
// CHECK:STDOUT:     .Float = %import_ref.1d6
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
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_39999999999999999993: Core.IntLiteral = int_value 39999999999999999993 [template = constants.%int_39999999999999999993.af6]
// CHECK:STDOUT:   %impl.elem0.loc15: %Convert.type.1b6 = impl_witness_access constants.%impl_witness.d39, element0 [template = constants.%Convert.956]
// CHECK:STDOUT:   %Convert.bound.loc15: <bound method> = bound_method %int_39999999999999999993, %impl.elem0.loc15 [template = constants.%Convert.bound.7ef]
// CHECK:STDOUT:   %Convert.specific_fn.loc15: <specific function> = specific_function %Convert.bound.loc15, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2ed]
// CHECK:STDOUT:   %int.convert_checked.loc15: init %i32 = call %Convert.specific_fn.loc15(%int_39999999999999999993) [template = constants.%int_39999999999999999993.dee]
// CHECK:STDOUT:   %.loc15_34.1: %i32 = value_of_initializer %int.convert_checked.loc15 [template = constants.%int_39999999999999999993.dee]
// CHECK:STDOUT:   %.loc15_34.2: %i32 = converted %int_39999999999999999993, %.loc15_34.1 [template = constants.%int_39999999999999999993.dee]
// CHECK:STDOUT:   %a: %i32 = bind_name a, %.loc15_34.2
// CHECK:STDOUT:   %int_2147483648.loc21: Core.IntLiteral = int_value 2147483648 [template = constants.%int_2147483648.1db]
// CHECK:STDOUT:   %impl.elem0.loc21: %Convert.type.1b6 = impl_witness_access constants.%impl_witness.d39, element0 [template = constants.%Convert.956]
// CHECK:STDOUT:   %Convert.bound.loc21: <bound method> = bound_method %int_2147483648.loc21, %impl.elem0.loc21 [template = constants.%Convert.bound.85f]
// CHECK:STDOUT:   %Convert.specific_fn.loc21: <specific function> = specific_function %Convert.bound.loc21, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.387]
// CHECK:STDOUT:   %int.convert_checked.loc21: init %i32 = call %Convert.specific_fn.loc21(%int_2147483648.loc21) [template = constants.%int_2147483648.8df]
// CHECK:STDOUT:   %.loc21_27.1: %i32 = value_of_initializer %int.convert_checked.loc21 [template = constants.%int_2147483648.8df]
// CHECK:STDOUT:   %.loc21_27.2: %i32 = converted %int_2147483648.loc21, %.loc21_27.1 [template = constants.%int_2147483648.8df]
// CHECK:STDOUT:   %b: %i32 = bind_name b, %.loc21_27.2
// CHECK:STDOUT:   %int_2147483648.loc27: Core.IntLiteral = int_value 2147483648 [template = constants.%int_2147483648.1db]
// CHECK:STDOUT:   %impl.elem0.loc27: %Convert.type.1b6 = impl_witness_access constants.%impl_witness.d39, element0 [template = constants.%Convert.956]
// CHECK:STDOUT:   %Convert.bound.loc27: <bound method> = bound_method %int_2147483648.loc27, %impl.elem0.loc27 [template = constants.%Convert.bound.85f]
// CHECK:STDOUT:   %Convert.specific_fn.loc27: <specific function> = specific_function %Convert.bound.loc27, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.387]
// CHECK:STDOUT:   %int.convert_checked.loc27: init %i32 = call %Convert.specific_fn.loc27(%int_2147483648.loc27) [template = constants.%int_2147483648.8df]
// CHECK:STDOUT:   %.loc27_25.1: %i32 = value_of_initializer %int.convert_checked.loc27 [template = constants.%int_2147483648.8df]
// CHECK:STDOUT:   %.loc27_25.2: %i32 = converted %int_2147483648.loc27, %.loc27_25.1 [template = constants.%int_2147483648.8df]
// CHECK:STDOUT:   %c: %i32 = bind_name c, %.loc27_25.2
// CHECK:STDOUT:   %d: f64 = bind_name d, <error>
// CHECK:STDOUT:   %e: f64 = bind_name e, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

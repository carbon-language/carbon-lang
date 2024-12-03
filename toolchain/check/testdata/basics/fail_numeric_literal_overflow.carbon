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
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %int_39999999999999999993.1: Core.IntLiteral = int_value 39999999999999999993 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_39999999999999999993.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_39999999999999999993.2: %i32 = int_value 39999999999999999993 [template]
// CHECK:STDOUT:   %int_2147483648.1: Core.IntLiteral = int_value 2147483648 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_2147483648.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2147483648.2: %i32 = int_value 2147483648 [template]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [template]
// CHECK:STDOUT:   %Float.type: type = fn_type @Float [template]
// CHECK:STDOUT:   %Float: %Float.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     .Float = %import_ref.38
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
// CHECK:STDOUT:   %int_32.loc15: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc15: init type = call constants.%Int(%int_32.loc15) [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_8.1: type = value_of_initializer %int.make_type_signed.loc15 [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_8.2: type = converted %int.make_type_signed.loc15, %.loc15_8.1 [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc21: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc21: init type = call constants.%Int(%int_32.loc21) [template = constants.%i32]
// CHECK:STDOUT:   %.loc21_8.1: type = value_of_initializer %int.make_type_signed.loc21 [template = constants.%i32]
// CHECK:STDOUT:   %.loc21_8.2: type = converted %int.make_type_signed.loc21, %.loc21_8.1 [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc27: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc27: init type = call constants.%Int(%int_32.loc27) [template = constants.%i32]
// CHECK:STDOUT:   %.loc27_8.1: type = value_of_initializer %int.make_type_signed.loc27 [template = constants.%i32]
// CHECK:STDOUT:   %.loc27_8.2: type = converted %int.make_type_signed.loc27, %.loc27_8.1 [template = constants.%i32]
// CHECK:STDOUT:   %int_64.loc33: Core.IntLiteral = int_value 64 [template = constants.%int_64]
// CHECK:STDOUT:   %float.make_type.loc33: init type = call constants.%Float(%int_64.loc33) [template = f64]
// CHECK:STDOUT:   %.loc33_8.1: type = value_of_initializer %float.make_type.loc33 [template = f64]
// CHECK:STDOUT:   %.loc33_8.2: type = converted %float.make_type.loc33, %.loc33_8.1 [template = f64]
// CHECK:STDOUT:   %int_64.loc38: Core.IntLiteral = int_value 64 [template = constants.%int_64]
// CHECK:STDOUT:   %float.make_type.loc38: init type = call constants.%Float(%int_64.loc38) [template = f64]
// CHECK:STDOUT:   %.loc38_8.1: type = value_of_initializer %float.make_type.loc38 [template = f64]
// CHECK:STDOUT:   %.loc38_8.2: type = converted %float.make_type.loc38, %.loc38_8.1 [template = f64]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_39999999999999999993: Core.IntLiteral = int_value 39999999999999999993 [template = constants.%int_39999999999999999993.1]
// CHECK:STDOUT:   %impl.elem0.loc15: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc15: <bound method> = bound_method %int_39999999999999999993, %impl.elem0.loc15 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc15: <specific function> = specific_function %Convert.bound.loc15, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc15: init %i32 = call %Convert.specific_fn.loc15(%int_39999999999999999993) [template = constants.%int_39999999999999999993.2]
// CHECK:STDOUT:   %.loc15_34.1: %i32 = value_of_initializer %int.convert_checked.loc15 [template = constants.%int_39999999999999999993.2]
// CHECK:STDOUT:   %.loc15_34.2: %i32 = converted %int_39999999999999999993, %.loc15_34.1 [template = constants.%int_39999999999999999993.2]
// CHECK:STDOUT:   %a: %i32 = bind_name a, %.loc15_34.2
// CHECK:STDOUT:   %int_2147483648.loc21: Core.IntLiteral = int_value 2147483648 [template = constants.%int_2147483648.1]
// CHECK:STDOUT:   %impl.elem0.loc21: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc21: <bound method> = bound_method %int_2147483648.loc21, %impl.elem0.loc21 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc21: <specific function> = specific_function %Convert.bound.loc21, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc21: init %i32 = call %Convert.specific_fn.loc21(%int_2147483648.loc21) [template = constants.%int_2147483648.2]
// CHECK:STDOUT:   %.loc21_27.1: %i32 = value_of_initializer %int.convert_checked.loc21 [template = constants.%int_2147483648.2]
// CHECK:STDOUT:   %.loc21_27.2: %i32 = converted %int_2147483648.loc21, %.loc21_27.1 [template = constants.%int_2147483648.2]
// CHECK:STDOUT:   %b: %i32 = bind_name b, %.loc21_27.2
// CHECK:STDOUT:   %int_2147483648.loc27: Core.IntLiteral = int_value 2147483648 [template = constants.%int_2147483648.1]
// CHECK:STDOUT:   %impl.elem0.loc27: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc27: <bound method> = bound_method %int_2147483648.loc27, %impl.elem0.loc27 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc27: <specific function> = specific_function %Convert.bound.loc27, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc27: init %i32 = call %Convert.specific_fn.loc27(%int_2147483648.loc27) [template = constants.%int_2147483648.2]
// CHECK:STDOUT:   %.loc27_25.1: %i32 = value_of_initializer %int.convert_checked.loc27 [template = constants.%int_2147483648.2]
// CHECK:STDOUT:   %.loc27_25.2: %i32 = converted %int_2147483648.loc27, %.loc27_25.1 [template = constants.%int_2147483648.2]
// CHECK:STDOUT:   %c: %i32 = bind_name c, %.loc27_25.2
// CHECK:STDOUT:   %d: f64 = bind_name d, <error>
// CHECK:STDOUT:   %e: f64 = bind_name e, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

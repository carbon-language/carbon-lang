// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/basics/fail_numeric_literal_overflow.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/basics/fail_numeric_literal_overflow.carbon

// CHECK:STDERR: fail_numeric_literal_overflow.carbon:[[@LINE+4]]:14: error: integer value 39999999999999999993 too large for type `i32` [IntTooLargeForType]
// CHECK:STDERR: let a: i32 = 39999999999999999993;
// CHECK:STDERR:              ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let a: i32 = 39999999999999999993;

// CHECK:STDERR: fail_numeric_literal_overflow.carbon:[[@LINE+4]]:14: error: integer value 2147483648 too large for type `i32` [IntTooLargeForType]
// CHECK:STDERR: let b: i32 = 2_147_483_648;
// CHECK:STDERR:              ^~~~~~~~~~~~~
// CHECK:STDERR:
let b: i32 = 2_147_483_648;

// CHECK:STDERR: fail_numeric_literal_overflow.carbon:[[@LINE+4]]:14: error: integer value 2147483648 too large for type `i32` [IntTooLargeForType]
// CHECK:STDERR: let c: i32 = 0x8000_0000;
// CHECK:STDERR:              ^~~~~~~~~~~
// CHECK:STDERR:
let c: i32 = 0x8000_0000;

// CHECK:STDERR: fail_numeric_literal_overflow.carbon:[[@LINE+4]]:14: error: real mantissa with value 399999999999999999930 does not fit in i64 [RealMantissaTooLargeForI64]
// CHECK:STDERR: let d: f64 = 39999999999999999993.0e3;
// CHECK:STDERR:              ^~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let d: f64 = 39999999999999999993.0e3;

// CHECK:STDERR: fail_numeric_literal_overflow.carbon:[[@LINE+4]]:14: error: real exponent with value 39999999999999999992 does not fit in i64 [RealExponentTooLargeForI64]
// CHECK:STDERR: let e: f64 = 5.0e39999999999999999993;
// CHECK:STDERR:              ^~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let e: f64 = 5.0e39999999999999999993;

// CHECK:STDOUT: --- fail_numeric_literal_overflow.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %int_39999999999999999993.af6: Core.IntLiteral = int_value 39999999999999999993 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %impl_witness.d39: <witness> = impl_witness (imports.%Core.import_ref.a5b), @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, %impl_witness.d39 [concrete]
// CHECK:STDOUT:   %.a0b: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound.7ef: <bound method> = bound_method %int_39999999999999999993.af6, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.75f: <bound method> = bound_method %int_39999999999999999993.af6, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_39999999999999999993.dee: %i32 = int_value 39999999999999999993 [concrete]
// CHECK:STDOUT:   %int_2147483648.1db: Core.IntLiteral = int_value 2147483648 [concrete]
// CHECK:STDOUT:   %Convert.bound.85f: <bound method> = bound_method %int_2147483648.1db, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.f79: <bound method> = bound_method %int_2147483648.1db, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2147483648.8df: %i32 = int_value 2147483648 [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %Float.type: type = fn_type @Float [concrete]
// CHECK:STDOUT:   %Float: %Float.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Float = %Core.Float
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:     .d = %d
// CHECK:STDOUT:     .e = %e
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc15_8: type = splice_block %i32.loc15 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc15: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc15: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc15: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc15_14.1: <bound method> = bound_method @__global_init.%int_39999999999999999993, %impl.elem0.loc15 [concrete = constants.%Convert.bound.7ef]
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem0.loc15, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_14.2: <bound method> = bound_method @__global_init.%int_39999999999999999993, %specific_fn.loc15 [concrete = constants.%bound_method.75f]
// CHECK:STDOUT:   %int.convert_checked.loc15: init %i32 = call %bound_method.loc15_14.2(@__global_init.%int_39999999999999999993) [concrete = constants.%int_39999999999999999993.dee]
// CHECK:STDOUT:   %.loc15_14.1: %i32 = value_of_initializer %int.convert_checked.loc15 [concrete = constants.%int_39999999999999999993.dee]
// CHECK:STDOUT:   %.loc15_14.2: %i32 = converted @__global_init.%int_39999999999999999993, %.loc15_14.1 [concrete = constants.%int_39999999999999999993.dee]
// CHECK:STDOUT:   %a: %i32 = bind_name a, %.loc15_14.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc21_8: type = splice_block %i32.loc21 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc21: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc21: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc21: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc21_14.1: <bound method> = bound_method @__global_init.%int_2147483648.loc21, %impl.elem0.loc21 [concrete = constants.%Convert.bound.85f]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem0.loc21, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc21_14.2: <bound method> = bound_method @__global_init.%int_2147483648.loc21, %specific_fn.loc21 [concrete = constants.%bound_method.f79]
// CHECK:STDOUT:   %int.convert_checked.loc21: init %i32 = call %bound_method.loc21_14.2(@__global_init.%int_2147483648.loc21) [concrete = constants.%int_2147483648.8df]
// CHECK:STDOUT:   %.loc21_14.1: %i32 = value_of_initializer %int.convert_checked.loc21 [concrete = constants.%int_2147483648.8df]
// CHECK:STDOUT:   %.loc21_14.2: %i32 = converted @__global_init.%int_2147483648.loc21, %.loc21_14.1 [concrete = constants.%int_2147483648.8df]
// CHECK:STDOUT:   %b: %i32 = bind_name b, %.loc21_14.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %i32 = binding_pattern c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc27_8: type = splice_block %i32.loc27 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc27: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc27: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc27: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc27_14.1: <bound method> = bound_method @__global_init.%int_2147483648.loc27, %impl.elem0.loc27 [concrete = constants.%Convert.bound.85f]
// CHECK:STDOUT:   %specific_fn.loc27: <specific function> = specific_function %impl.elem0.loc27, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc27_14.2: <bound method> = bound_method @__global_init.%int_2147483648.loc27, %specific_fn.loc27 [concrete = constants.%bound_method.f79]
// CHECK:STDOUT:   %int.convert_checked.loc27: init %i32 = call %bound_method.loc27_14.2(@__global_init.%int_2147483648.loc27) [concrete = constants.%int_2147483648.8df]
// CHECK:STDOUT:   %.loc27_14.1: %i32 = value_of_initializer %int.convert_checked.loc27 [concrete = constants.%int_2147483648.8df]
// CHECK:STDOUT:   %.loc27_14.2: %i32 = converted @__global_init.%int_2147483648.loc27, %.loc27_14.1 [concrete = constants.%int_2147483648.8df]
// CHECK:STDOUT:   %c: %i32 = bind_name c, %.loc27_14.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: f64 = binding_pattern d
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc33_8.1: type = splice_block %.loc33_8.3 [concrete = f64] {
// CHECK:STDOUT:     %int_64.loc33: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %float.make_type.loc33: init type = call constants.%Float(%int_64.loc33) [concrete = f64]
// CHECK:STDOUT:     %.loc33_8.2: type = value_of_initializer %float.make_type.loc33 [concrete = f64]
// CHECK:STDOUT:     %.loc33_8.3: type = converted %float.make_type.loc33, %.loc33_8.2 [concrete = f64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d: f64 = bind_name d, <error>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %e.patt: f64 = binding_pattern e
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc39_8.1: type = splice_block %.loc39_8.3 [concrete = f64] {
// CHECK:STDOUT:     %int_64.loc39: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %float.make_type.loc39: init type = call constants.%Float(%int_64.loc39) [concrete = f64]
// CHECK:STDOUT:     %.loc39_8.2: type = value_of_initializer %float.make_type.loc39 [concrete = f64]
// CHECK:STDOUT:     %.loc39_8.3: type = converted %float.make_type.loc39, %.loc39_8.2 [concrete = f64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %e: f64 = bind_name e, <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_39999999999999999993: Core.IntLiteral = int_value 39999999999999999993 [concrete = constants.%int_39999999999999999993.af6]
// CHECK:STDOUT:   %int_2147483648.loc21: Core.IntLiteral = int_value 2147483648 [concrete = constants.%int_2147483648.1db]
// CHECK:STDOUT:   %int_2147483648.loc27: Core.IntLiteral = int_value 2147483648 [concrete = constants.%int_2147483648.1db]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

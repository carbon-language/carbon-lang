// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/float/convert_int.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/float/convert_int.carbon

// --- float_to_int.carbon
library "[[@TEST_NAME]]";

fn ConvertF64ToI32(a: f64) -> i32 = "float.convert_int";
fn ConvertF64ToU32(a: f64) -> u32 = "float.convert_int";
fn ConvertFloatLiteralToI32(a: Core.FloatLiteral) -> i32 = "float.convert_int";
fn ConvertFloatLiteralToU32(a: Core.FloatLiteral) -> u32 = "float.convert_int";
fn ConvertFloatLiteralToIntLiteral(a: Core.FloatLiteral) -> Core.IntLiteral = "float.convert_int";
fn ConvertF64ToIntLiteral(a: f64) -> Core.IntLiteral = "float.convert_int";

fn Negate(a: f64) -> f64 = "float.negate";
fn Div(a: f64, b: f64) -> f64 = "float.div";

class Expect[T:! type](N:! T) {}
fn Test[T:! type](N:! T) -> Expect(N) { return {}; }

// --- identity.carbon
library "[[@TEST_NAME]]";
import library "float_to_int";

fn F() {
  Test(ConvertF64ToI32(0.0)) as Expect(0 as i32);
  Test(ConvertF64ToI32(1.0)) as Expect(1 as i32);
  Test(ConvertF64ToI32(Negate(1.0))) as Expect(-1 as i32);
  Test(ConvertF64ToI32(1.99)) as Expect(1 as i32);
  Test(ConvertF64ToI32(Negate(1.99))) as Expect(-1 as i32);
  Test(ConvertF64ToI32(2147483647.0)) as Expect(2147483647 as i32);
  Test(ConvertF64ToI32(Negate(2147483648.0))) as Expect(-2147483648 as i32);

  Test(ConvertF64ToU32(0.0)) as Expect(0 as u32);
  Test(ConvertF64ToU32(1.0)) as Expect(1 as u32);
  Test(ConvertF64ToU32(4294967295.0)) as Expect(4294967295 as u32);

  Test(ConvertFloatLiteralToI32(1.0)) as Expect(1 as i32);
  Test(ConvertFloatLiteralToI32(0.0)) as Expect(0 as i32);
  Test(ConvertFloatLiteralToI32(0.0e1000000000000000000000000000000000)) as Expect(0 as i32);
  Test(ConvertFloatLiteralToI32(1.0e-1000000000000000000000000000000000)) as Expect(0 as i32);

  Test(ConvertFloatLiteralToI32(0x1.0p4)) as Expect(16 as i32);
  Test(ConvertFloatLiteralToI32(0x3.4p4)) as Expect(52 as i32);
  Test(ConvertFloatLiteralToI32(0x1.0p-4)) as Expect(0 as i32);
  Test(ConvertFloatLiteralToI32(0x10.0p-2)) as Expect(4 as i32);

  Test(ConvertFloatLiteralToI32(0x1000000000000000000000000.0p-96)) as Expect(1 as i32);
  Test(ConvertFloatLiteralToI32(0x1000000000000000000000000.1p-96)) as Expect(1 as i32);
  Test(ConvertFloatLiteralToI32(0x0FFFFFFFFFFFFFFFFFFFFFFFF.Fp-96)) as Expect(0 as i32);

  Test(ConvertFloatLiteralToI32(100000000000000000000000000000000000000000000000000.0e-50)) as Expect(1 as i32);
  Test(ConvertFloatLiteralToI32(100000000000000000000000000000000000000000000000000.1e-50)) as Expect(1 as i32);
  Test(ConvertFloatLiteralToI32(99999999999999999999999999999999999999999999999999.9e-50)) as Expect(0 as i32);

  Test(ConvertFloatLiteralToI32(214748364.7e1)) as Expect(2147483647 as i32);
  Test(ConvertFloatLiteralToI32(21474.83647e5)) as Expect(2147483647 as i32);
  Test(ConvertFloatLiteralToI32(0x7FFF.FFFFp16)) as Expect(2147483647 as i32);
  Test(ConvertFloatLiteralToI32(1.0e9)) as Expect(1000000000 as i32);
  Test(ConvertFloatLiteralToI32(0x1.0p30)) as Expect(1073741824 as i32);

  Test(ConvertFloatLiteralToU32(0x8000.0000p16)) as Expect(2147483648 as u32);
  Test(ConvertFloatLiteralToU32(4.0e9)) as Expect(4000000000 as u32);
  Test(ConvertFloatLiteralToU32(0x1.0p31)) as Expect(2147483648 as u32);

  Test(ConvertFloatLiteralToIntLiteral(1.0)) as Expect(1);
  Test(ConvertFloatLiteralToIntLiteral(0.0)) as Expect(0);
  Test(ConvertFloatLiteralToIntLiteral(0.0e1000000000000000000000000000000000)) as Expect(0);
  Test(ConvertFloatLiteralToIntLiteral(1.0e-1000000000000000000000000000000000)) as Expect(0);

  Test(ConvertFloatLiteralToIntLiteral(0x1.0p8)) as Expect(256);
  Test(ConvertFloatLiteralToIntLiteral(0x1.0p-8)) as Expect(0);

  Test(ConvertFloatLiteralToIntLiteral(0x1000000000000000000000000.0p-96)) as Expect(1);
  Test(ConvertFloatLiteralToIntLiteral(0x1000000000000000000000000.1p-96)) as Expect(1);
  Test(ConvertFloatLiteralToIntLiteral(0x0FFFFFFFFFFFFFFFFFFFFFFFF.Fp-96)) as Expect(0);

  Test(ConvertFloatLiteralToIntLiteral(100000000000000000000000000000000000000000000000000.0e-50)) as Expect(1);
  Test(ConvertFloatLiteralToIntLiteral(100000000000000000000000000000000000000000000000000.1e-50)) as Expect(1);
  Test(ConvertFloatLiteralToIntLiteral(99999999999999999999999999999999999999999999999999.9e-50)) as Expect(0);

  Test(ConvertF64ToIntLiteral(1.0)) as Expect(1);
  Test(ConvertF64ToIntLiteral(Negate(1.0))) as Expect(-1);
}

// --- fail_overflow.carbon
library "[[@TEST_NAME]]";
import library "float_to_int";

//@dump-sem-ir-begin
// CHECK:STDERR: fail_overflow.carbon:[[@LINE+4]]:21: error: floating-point value 2147483648 too large for integer type `i32` [FloatTooLargeForIntType]
// CHECK:STDERR: let overflow: i32 = ConvertF64ToI32(2147483648.0);
// CHECK:STDERR:                     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let overflow: i32 = ConvertF64ToI32(2147483648.0);

// CHECK:STDERR: fail_overflow.carbon:[[@LINE+4]]:22: error: floating-point value -2147483649 too large for integer type `i32` [FloatTooLargeForIntType]
// CHECK:STDERR: let underflow: i32 = ConvertF64ToI32(Negate(2147483649.0));
// CHECK:STDERR:                      ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let underflow: i32 = ConvertF64ToI32(Negate(2147483649.0));

// CHECK:STDERR: fail_overflow.carbon:[[@LINE+4]]:25: error: floating-point value -1 too large for integer type `u32` [FloatTooLargeForIntType]
// CHECK:STDERR: let unsigned_neg: u32 = ConvertF64ToU32(Negate(1.0));
// CHECK:STDERR:                         ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let unsigned_neg: u32 = ConvertF64ToU32(Negate(1.0));

// CHECK:STDERR: fail_overflow.carbon:[[@LINE+4]]:29: error: floating-point value 21474836480*10^-1 too large for integer type `i32` [RealLiteralTooLargeForIntType]
// CHECK:STDERR: let literal_overflow: i32 = ConvertFloatLiteralToI32(2147483648.0);
// CHECK:STDERR:                             ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let literal_overflow: i32 = ConvertFloatLiteralToI32(2147483648.0);

// CHECK:STDERR: fail_overflow.carbon:[[@LINE+4]]:35: error: floating-point value 10*10^999999 too large for integer type `i32` [RealLiteralTooLargeForIntType]
// CHECK:STDERR: let giant_literal_overflow: i32 = ConvertFloatLiteralToI32(1.0e1000000);
// CHECK:STDERR:                                   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let giant_literal_overflow: i32 = ConvertFloatLiteralToI32(1.0e1000000);

//
// CHECK:STDERR: fail_overflow.carbon:[[@LINE+4]]:62: error: floating-point value 10*10^9999999 too large to convert: result would be an integer whose width is greater than the maximum supported width of 8388608 [RealLiteralTooLargeForUnsizedInt]
// CHECK:STDERR: let giant_literal_to_int_literal_overflow: Core.IntLiteral = ConvertFloatLiteralToIntLiteral(1.0e10000000);
// CHECK:STDERR:                                                              ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let giant_literal_to_int_literal_overflow: Core.IntLiteral = ConvertFloatLiteralToIntLiteral(1.0e10000000);

//
// CHECK:STDERR: fail_overflow.carbon:[[@LINE+4]]:44: error: floating-point value 10*10^999999999999999999999999999999999 too large for integer type `i32` [RealLiteralTooLargeForIntType]
// CHECK:STDERR: let giant_exponent_literal_overflow: i32 = ConvertFloatLiteralToI32(1.0e1000000000000000000000000000000000);
// CHECK:STDERR:                                            ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let giant_exponent_literal_overflow: i32 = ConvertFloatLiteralToI32(1.0e1000000000000000000000000000000000);

//
//
// CHECK:STDERR: fail_overflow.carbon:[[@LINE+4]]:71: error: floating-point value 10*10^999999999999999999999999999999999 too large to convert: result would be an integer whose width is greater than the maximum supported width of 8388608 [RealLiteralTooLargeForUnsizedInt]
// CHECK:STDERR: let giant_exponent_literal_to_int_literal_overflow: Core.IntLiteral = ConvertFloatLiteralToIntLiteral(1.0e1000000000000000000000000000000000);
// CHECK:STDERR:                                                                       ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let giant_exponent_literal_to_int_literal_overflow: Core.IntLiteral = ConvertFloatLiteralToIntLiteral(1.0e1000000000000000000000000000000000);

//
// CHECK:STDERR: fail_overflow.carbon:[[@LINE+4]]:44: error: floating-point value 2147483648*10^0 too large for integer type `i32` [RealLiteralTooLargeForIntType]
// CHECK:STDERR: let literal_overflow_large_mantissa: i32 = ConvertFloatLiteralToI32(214748364.8e1);
// CHECK:STDERR:                                            ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let literal_overflow_large_mantissa: i32 = ConvertFloatLiteralToI32(214748364.8e1);

//
// CHECK:STDERR: fail_overflow.carbon:[[@LINE+4]]:42: error: floating-point value 2147483648*10^0 too large for integer type `i32` [RealLiteralTooLargeForIntType]
// CHECK:STDERR: let literal_overflow_med_mantissa: i32 = ConvertFloatLiteralToI32(21474.83648e5);
// CHECK:STDERR:                                          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let literal_overflow_med_mantissa: i32 = ConvertFloatLiteralToI32(21474.83648e5);

//
// CHECK:STDERR: fail_overflow.carbon:[[@LINE+4]]:46: error: floating-point value 2147483648*2^0 too large for integer type `i32` [RealLiteralTooLargeForIntType]
// CHECK:STDERR: let literal_overflow_hex_med_mantissa: i32 = ConvertFloatLiteralToI32(0x8000.0000p16);
// CHECK:STDERR:                                              ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let literal_overflow_hex_med_mantissa: i32 = ConvertFloatLiteralToI32(0x8000.0000p16);

//
// CHECK:STDERR: fail_overflow.carbon:[[@LINE+4]]:44: error: floating-point value 10*10^9 too large for integer type `i32` [RealLiteralTooLargeForIntType]
// CHECK:STDERR: let literal_overflow_small_mantissa: i32 = ConvertFloatLiteralToI32(1.0e10);
// CHECK:STDERR:                                            ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let literal_overflow_small_mantissa: i32 = ConvertFloatLiteralToI32(1.0e10);

//
// CHECK:STDERR: fail_overflow.carbon:[[@LINE+4]]:48: error: floating-point value 16*2^27 too large for integer type `i32` [RealLiteralTooLargeForIntType]
// CHECK:STDERR: let literal_overflow_bin_small_mantissa: i32 = ConvertFloatLiteralToI32(0x1.0p31);
// CHECK:STDERR:                                                ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let literal_overflow_bin_small_mantissa: i32 = ConvertFloatLiteralToI32(0x1.0p31);

//
// CHECK:STDERR: fail_overflow.carbon:[[@LINE+4]]:42: error: floating-point value 50*10^8 too large for integer type `u32` [RealLiteralTooLargeForIntType]
// CHECK:STDERR: let literal_overflow_unsigned_neg: u32 = ConvertFloatLiteralToU32(5.0e9);
// CHECK:STDERR:                                          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let literal_overflow_unsigned_neg: u32 = ConvertFloatLiteralToU32(5.0e9);

//
// CHECK:STDERR: fail_overflow.carbon:[[@LINE+4]]:46: error: floating-point value 16*2^28 too large for integer type `u32` [RealLiteralTooLargeForIntType]
// CHECK:STDERR: let literal_overflow_bin_unsigned_neg: u32 = ConvertFloatLiteralToU32(0x1.0p32);
// CHECK:STDERR:                                              ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let literal_overflow_bin_unsigned_neg: u32 = ConvertFloatLiteralToU32(0x1.0p32);
//@dump-sem-ir-end

// --- fail_nan_inf.carbon
library "[[@TEST_NAME]]";
import library "float_to_int";

//@dump-sem-ir-begin
// CHECK:STDERR: fail_nan_inf.carbon:[[@LINE+4]]:20: error: cannot convert NaN to integer type `i32` [FloatNaNConvertedToInt]
// CHECK:STDERR: let nan_val: i32 = ConvertF64ToI32(Div(0.0, 0.0));
// CHECK:STDERR:                    ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let nan_val: i32 = ConvertF64ToI32(Div(0.0, 0.0));

// CHECK:STDERR: fail_nan_inf.carbon:[[@LINE+4]]:20: error: cannot convert infinity to integer type `i32` [FloatInfinityConvertedToInt]
// CHECK:STDERR: let inf_val: i32 = ConvertF64ToI32(Div(1.0, 0.0));
// CHECK:STDERR:                    ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let inf_val: i32 = ConvertF64ToI32(Div(1.0, 0.0));
//@dump-sem-ir-end

// CHECK:STDOUT: --- fail_overflow.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.dc0: type = pattern_type Core.IntLiteral [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %overflow.patt: %pattern_type.6b6 = value_binding_pattern overflow [concrete]
// CHECK:STDOUT:   %ConvertF64ToI32.type: type = fn_type @ConvertF64ToI32 [concrete]
// CHECK:STDOUT:   %ConvertF64ToI32: %ConvertF64ToI32.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %f64.dc1: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %float.bab60c.1: Core.FloatLiteral = float_literal_value 21474836480e-1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.91e: type = facet_type <@ImplicitAs, @ImplicitAs(%f64.dc1)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.063: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.d4f: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.063 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.43c: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table, @Core.FloatLiteral.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.ef0: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.cf0: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.ef0 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.91e = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.43c) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.6c5: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%f64.dc1, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.2d7: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.6c5, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.d5f: <bound method> = bound_method %float.bab60c.1, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.cf0 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.ImplicitAs.impl.Convert.cf0, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.a8b: <bound method> = bound_method %float.bab60c.1, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.d5c: %f64.dc1 = float_value 2147483648 [concrete]
// CHECK:STDOUT:   %underflow.patt: %pattern_type.6b6 = value_binding_pattern underflow [concrete]
// CHECK:STDOUT:   %Negate.type: type = fn_type @Negate [concrete]
// CHECK:STDOUT:   %Negate: %Negate.type = struct_value () [concrete]
// CHECK:STDOUT:   %float.449: Core.FloatLiteral = float_literal_value 21474836490e-1 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.92c: <bound method> = bound_method %float.449, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.cf0 [concrete]
// CHECK:STDOUT:   %bound_method.af3: <bound method> = bound_method %float.449, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.f2a: %f64.dc1 = float_value 2147483649 [concrete]
// CHECK:STDOUT:   %float.caa: %f64.dc1 = float_value -2147483649 [concrete]
// CHECK:STDOUT:   %UInt.type: type = generic_class_type @UInt [concrete]
// CHECK:STDOUT:   %UInt.generic: %UInt.type = struct_value () [concrete]
// CHECK:STDOUT:   %u32: type = class_type @UInt, @UInt(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.668: type = pattern_type %u32 [concrete]
// CHECK:STDOUT:   %unsigned_neg.patt: %pattern_type.668 = value_binding_pattern unsigned_neg [concrete]
// CHECK:STDOUT:   %ConvertF64ToU32.type: type = fn_type @ConvertF64ToU32 [concrete]
// CHECK:STDOUT:   %ConvertF64ToU32: %ConvertF64ToU32.type = struct_value () [concrete]
// CHECK:STDOUT:   %float.6da: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.757: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.cf0 [concrete]
// CHECK:STDOUT:   %bound_method.143: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.3b9: %f64.dc1 = float_value 1 [concrete]
// CHECK:STDOUT:   %float.b2f: %f64.dc1 = float_value -1 [concrete]
// CHECK:STDOUT:   %literal_overflow.patt: %pattern_type.6b6 = value_binding_pattern literal_overflow [concrete]
// CHECK:STDOUT:   %ConvertFloatLiteralToI32.type: type = fn_type @ConvertFloatLiteralToI32 [concrete]
// CHECK:STDOUT:   %ConvertFloatLiteralToI32: %ConvertFloatLiteralToI32.type = struct_value () [concrete]
// CHECK:STDOUT:   %float.bab60c.2: Core.FloatLiteral = float_literal_value 21474836480e-1 [concrete]
// CHECK:STDOUT:   %giant_literal_overflow.patt: %pattern_type.6b6 = value_binding_pattern giant_literal_overflow [concrete]
// CHECK:STDOUT:   %float.53d: Core.FloatLiteral = float_literal_value 10e999999 [concrete]
// CHECK:STDOUT:   %giant_literal_to_int_literal_overflow.patt: %pattern_type.dc0 = value_binding_pattern giant_literal_to_int_literal_overflow [concrete]
// CHECK:STDOUT:   %ConvertFloatLiteralToIntLiteral.type: type = fn_type @ConvertFloatLiteralToIntLiteral [concrete]
// CHECK:STDOUT:   %ConvertFloatLiteralToIntLiteral: %ConvertFloatLiteralToIntLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %float.ef09: Core.FloatLiteral = float_literal_value 10e9999999 [concrete]
// CHECK:STDOUT:   %giant_exponent_literal_overflow.patt: %pattern_type.6b6 = value_binding_pattern giant_exponent_literal_overflow [concrete]
// CHECK:STDOUT:   %float.f461ee.1: Core.FloatLiteral = float_literal_value 10e999999999999999999999999999999999 [concrete]
// CHECK:STDOUT:   %giant_exponent_literal_to_int_literal_overflow.patt: %pattern_type.dc0 = value_binding_pattern giant_exponent_literal_to_int_literal_overflow [concrete]
// CHECK:STDOUT:   %float.f461ee.2: Core.FloatLiteral = float_literal_value 10e999999999999999999999999999999999 [concrete]
// CHECK:STDOUT:   %literal_overflow_large_mantissa.patt: %pattern_type.6b6 = value_binding_pattern literal_overflow_large_mantissa [concrete]
// CHECK:STDOUT:   %float.e12922.1: Core.FloatLiteral = float_literal_value 2147483648e0 [concrete]
// CHECK:STDOUT:   %literal_overflow_med_mantissa.patt: %pattern_type.6b6 = value_binding_pattern literal_overflow_med_mantissa [concrete]
// CHECK:STDOUT:   %float.e12922.2: Core.FloatLiteral = float_literal_value 2147483648e0 [concrete]
// CHECK:STDOUT:   %literal_overflow_hex_med_mantissa.patt: %pattern_type.6b6 = value_binding_pattern literal_overflow_hex_med_mantissa [concrete]
// CHECK:STDOUT:   %float.ef03: Core.FloatLiteral = float_literal_value 2147483648p0 [concrete]
// CHECK:STDOUT:   %literal_overflow_small_mantissa.patt: %pattern_type.6b6 = value_binding_pattern literal_overflow_small_mantissa [concrete]
// CHECK:STDOUT:   %float.9d5: Core.FloatLiteral = float_literal_value 10e9 [concrete]
// CHECK:STDOUT:   %literal_overflow_bin_small_mantissa.patt: %pattern_type.6b6 = value_binding_pattern literal_overflow_bin_small_mantissa [concrete]
// CHECK:STDOUT:   %float.c0c: Core.FloatLiteral = float_literal_value 16p27 [concrete]
// CHECK:STDOUT:   %literal_overflow_unsigned_neg.patt: %pattern_type.668 = value_binding_pattern literal_overflow_unsigned_neg [concrete]
// CHECK:STDOUT:   %ConvertFloatLiteralToU32.type: type = fn_type @ConvertFloatLiteralToU32 [concrete]
// CHECK:STDOUT:   %ConvertFloatLiteralToU32: %ConvertFloatLiteralToU32.type = struct_value () [concrete]
// CHECK:STDOUT:   %float.4f4: Core.FloatLiteral = float_literal_value 50e8 [concrete]
// CHECK:STDOUT:   %literal_overflow_bin_unsigned_neg.patt: %pattern_type.668 = value_binding_pattern literal_overflow_bin_unsigned_neg [concrete]
// CHECK:STDOUT:   %float.97b: Core.FloatLiteral = float_literal_value 16p28 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.ConvertF64ToI32: %ConvertF64ToI32.type = import_ref Main//float_to_int, ConvertF64ToI32, loaded [concrete = constants.%ConvertF64ToI32]
// CHECK:STDOUT:   %Main.ConvertF64ToU32: %ConvertF64ToU32.type = import_ref Main//float_to_int, ConvertF64ToU32, loaded [concrete = constants.%ConvertF64ToU32]
// CHECK:STDOUT:   %Main.ConvertFloatLiteralToI32: %ConvertFloatLiteralToI32.type = import_ref Main//float_to_int, ConvertFloatLiteralToI32, loaded [concrete = constants.%ConvertFloatLiteralToI32]
// CHECK:STDOUT:   %Main.ConvertFloatLiteralToU32: %ConvertFloatLiteralToU32.type = import_ref Main//float_to_int, ConvertFloatLiteralToU32, loaded [concrete = constants.%ConvertFloatLiteralToU32]
// CHECK:STDOUT:   %Main.ConvertFloatLiteralToIntLiteral: %ConvertFloatLiteralToIntLiteral.type = import_ref Main//float_to_int, ConvertFloatLiteralToIntLiteral, loaded [concrete = constants.%ConvertFloatLiteralToIntLiteral]
// CHECK:STDOUT:   %Main.Negate: %Negate.type = import_ref Main//float_to_int, Negate, loaded [concrete = constants.%Negate]
// CHECK:STDOUT:   %Core.048aa3.1: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .UInt = %Core.UInt
// CHECK:STDOUT:     .IntLiteral = %Core.IntLiteral
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.c4c: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.063) = import_ref Core//prelude/parts/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.d4f)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%Core.import_ref.c4c), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.UInt: %UInt.type = import_ref Core//prelude/parts/uint, UInt, loaded [concrete = constants.%UInt.generic]
// CHECK:STDOUT:   %Core.IntLiteral: type = import_ref Core//prelude/parts/int_literal, IntLiteral, loaded [concrete = Core.IntLiteral]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %overflow.patt: %pattern_type.6b6 = value_binding_pattern overflow [concrete = constants.%overflow.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9_49.1: %i32 = value_of_initializer @__global_init.%ConvertF64ToI32.call.loc9 [concrete = <error>]
// CHECK:STDOUT:   %.loc9_49.2: %i32 = converted @__global_init.%ConvertF64ToI32.call.loc9, %.loc9_49.1 [concrete = <error>]
// CHECK:STDOUT:   %i32.loc9: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %overflow: %i32 = value_binding overflow, %.loc9_49.2 [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %underflow.patt: %pattern_type.6b6 = value_binding_pattern underflow [concrete = constants.%underflow.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc15_58.1: %i32 = value_of_initializer @__global_init.%ConvertF64ToI32.call.loc15 [concrete = <error>]
// CHECK:STDOUT:   %.loc15_58.2: %i32 = converted @__global_init.%ConvertF64ToI32.call.loc15, %.loc15_58.1 [concrete = <error>]
// CHECK:STDOUT:   %i32.loc15: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %underflow: %i32 = value_binding underflow, %.loc15_58.2 [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %unsigned_neg.patt: %pattern_type.668 = value_binding_pattern unsigned_neg [concrete = constants.%unsigned_neg.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc21_52.1: %u32 = value_of_initializer @__global_init.%ConvertF64ToU32.call [concrete = <error>]
// CHECK:STDOUT:   %.loc21_52.2: %u32 = converted @__global_init.%ConvertF64ToU32.call, %.loc21_52.1 [concrete = <error>]
// CHECK:STDOUT:   %u32.loc21: type = type_literal constants.%u32 [concrete = constants.%u32]
// CHECK:STDOUT:   %unsigned_neg: %u32 = value_binding unsigned_neg, %.loc21_52.2 [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %literal_overflow.patt: %pattern_type.6b6 = value_binding_pattern literal_overflow [concrete = constants.%literal_overflow.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc27_66.1: %i32 = value_of_initializer @__global_init.%ConvertFloatLiteralToI32.call.loc27 [concrete = <error>]
// CHECK:STDOUT:   %.loc27_66.2: %i32 = converted @__global_init.%ConvertFloatLiteralToI32.call.loc27, %.loc27_66.1 [concrete = <error>]
// CHECK:STDOUT:   %i32.loc27: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %literal_overflow: %i32 = value_binding literal_overflow, %.loc27_66.2 [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %giant_literal_overflow.patt: %pattern_type.6b6 = value_binding_pattern giant_literal_overflow [concrete = constants.%giant_literal_overflow.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc33_71.1: %i32 = value_of_initializer @__global_init.%ConvertFloatLiteralToI32.call.loc33 [concrete = <error>]
// CHECK:STDOUT:   %.loc33_71.2: %i32 = converted @__global_init.%ConvertFloatLiteralToI32.call.loc33, %.loc33_71.1 [concrete = <error>]
// CHECK:STDOUT:   %i32.loc33: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %giant_literal_overflow: %i32 = value_binding giant_literal_overflow, %.loc33_71.2 [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %giant_literal_to_int_literal_overflow.patt: %pattern_type.dc0 = value_binding_pattern giant_literal_to_int_literal_overflow [concrete = constants.%giant_literal_to_int_literal_overflow.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc40_106.1: Core.IntLiteral = value_of_initializer @__global_init.%ConvertFloatLiteralToIntLiteral.call.loc40 [concrete = <error>]
// CHECK:STDOUT:   %.loc40_106.2: Core.IntLiteral = converted @__global_init.%ConvertFloatLiteralToIntLiteral.call.loc40, %.loc40_106.1 [concrete = <error>]
// CHECK:STDOUT:   %.loc40_48: type = splice_block %IntLiteral.ref.loc40 [concrete = Core.IntLiteral] {
// CHECK:STDOUT:     %Core.ref.loc40: <namespace> = name_ref Core, imports.%Core.048aa3.1 [concrete = imports.%Core.048aa3.1]
// CHECK:STDOUT:     %IntLiteral.ref.loc40: type = name_ref IntLiteral, imports.%Core.IntLiteral [concrete = Core.IntLiteral]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %giant_literal_to_int_literal_overflow: Core.IntLiteral = value_binding giant_literal_to_int_literal_overflow, %.loc40_106.2 [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %giant_exponent_literal_overflow.patt: %pattern_type.6b6 = value_binding_pattern giant_exponent_literal_overflow [concrete = constants.%giant_exponent_literal_overflow.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc47_107.1: %i32 = value_of_initializer @__global_init.%ConvertFloatLiteralToI32.call.loc47 [concrete = <error>]
// CHECK:STDOUT:   %.loc47_107.2: %i32 = converted @__global_init.%ConvertFloatLiteralToI32.call.loc47, %.loc47_107.1 [concrete = <error>]
// CHECK:STDOUT:   %i32.loc47: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %giant_exponent_literal_overflow: %i32 = value_binding giant_exponent_literal_overflow, %.loc47_107.2 [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %giant_exponent_literal_to_int_literal_overflow.patt: %pattern_type.dc0 = value_binding_pattern giant_exponent_literal_to_int_literal_overflow [concrete = constants.%giant_exponent_literal_to_int_literal_overflow.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc55_141.1: Core.IntLiteral = value_of_initializer @__global_init.%ConvertFloatLiteralToIntLiteral.call.loc55 [concrete = <error>]
// CHECK:STDOUT:   %.loc55_141.2: Core.IntLiteral = converted @__global_init.%ConvertFloatLiteralToIntLiteral.call.loc55, %.loc55_141.1 [concrete = <error>]
// CHECK:STDOUT:   %.loc55_57: type = splice_block %IntLiteral.ref.loc55 [concrete = Core.IntLiteral] {
// CHECK:STDOUT:     %Core.ref.loc55: <namespace> = name_ref Core, imports.%Core.048aa3.1 [concrete = imports.%Core.048aa3.1]
// CHECK:STDOUT:     %IntLiteral.ref.loc55: type = name_ref IntLiteral, imports.%Core.IntLiteral [concrete = Core.IntLiteral]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %giant_exponent_literal_to_int_literal_overflow: Core.IntLiteral = value_binding giant_exponent_literal_to_int_literal_overflow, %.loc55_141.2 [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %literal_overflow_large_mantissa.patt: %pattern_type.6b6 = value_binding_pattern literal_overflow_large_mantissa [concrete = constants.%literal_overflow_large_mantissa.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc62_82.1: %i32 = value_of_initializer @__global_init.%ConvertFloatLiteralToI32.call.loc62 [concrete = <error>]
// CHECK:STDOUT:   %.loc62_82.2: %i32 = converted @__global_init.%ConvertFloatLiteralToI32.call.loc62, %.loc62_82.1 [concrete = <error>]
// CHECK:STDOUT:   %i32.loc62: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %literal_overflow_large_mantissa: %i32 = value_binding literal_overflow_large_mantissa, %.loc62_82.2 [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %literal_overflow_med_mantissa.patt: %pattern_type.6b6 = value_binding_pattern literal_overflow_med_mantissa [concrete = constants.%literal_overflow_med_mantissa.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc69_80.1: %i32 = value_of_initializer @__global_init.%ConvertFloatLiteralToI32.call.loc69 [concrete = <error>]
// CHECK:STDOUT:   %.loc69_80.2: %i32 = converted @__global_init.%ConvertFloatLiteralToI32.call.loc69, %.loc69_80.1 [concrete = <error>]
// CHECK:STDOUT:   %i32.loc69: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %literal_overflow_med_mantissa: %i32 = value_binding literal_overflow_med_mantissa, %.loc69_80.2 [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %literal_overflow_hex_med_mantissa.patt: %pattern_type.6b6 = value_binding_pattern literal_overflow_hex_med_mantissa [concrete = constants.%literal_overflow_hex_med_mantissa.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc76_85.1: %i32 = value_of_initializer @__global_init.%ConvertFloatLiteralToI32.call.loc76 [concrete = <error>]
// CHECK:STDOUT:   %.loc76_85.2: %i32 = converted @__global_init.%ConvertFloatLiteralToI32.call.loc76, %.loc76_85.1 [concrete = <error>]
// CHECK:STDOUT:   %i32.loc76: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %literal_overflow_hex_med_mantissa: %i32 = value_binding literal_overflow_hex_med_mantissa, %.loc76_85.2 [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %literal_overflow_small_mantissa.patt: %pattern_type.6b6 = value_binding_pattern literal_overflow_small_mantissa [concrete = constants.%literal_overflow_small_mantissa.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc83_75.1: %i32 = value_of_initializer @__global_init.%ConvertFloatLiteralToI32.call.loc83 [concrete = <error>]
// CHECK:STDOUT:   %.loc83_75.2: %i32 = converted @__global_init.%ConvertFloatLiteralToI32.call.loc83, %.loc83_75.1 [concrete = <error>]
// CHECK:STDOUT:   %i32.loc83: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %literal_overflow_small_mantissa: %i32 = value_binding literal_overflow_small_mantissa, %.loc83_75.2 [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %literal_overflow_bin_small_mantissa.patt: %pattern_type.6b6 = value_binding_pattern literal_overflow_bin_small_mantissa [concrete = constants.%literal_overflow_bin_small_mantissa.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc90_81.1: %i32 = value_of_initializer @__global_init.%ConvertFloatLiteralToI32.call.loc90 [concrete = <error>]
// CHECK:STDOUT:   %.loc90_81.2: %i32 = converted @__global_init.%ConvertFloatLiteralToI32.call.loc90, %.loc90_81.1 [concrete = <error>]
// CHECK:STDOUT:   %i32.loc90: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %literal_overflow_bin_small_mantissa: %i32 = value_binding literal_overflow_bin_small_mantissa, %.loc90_81.2 [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %literal_overflow_unsigned_neg.patt: %pattern_type.668 = value_binding_pattern literal_overflow_unsigned_neg [concrete = constants.%literal_overflow_unsigned_neg.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc97_72.1: %u32 = value_of_initializer @__global_init.%ConvertFloatLiteralToU32.call.loc97 [concrete = <error>]
// CHECK:STDOUT:   %.loc97_72.2: %u32 = converted @__global_init.%ConvertFloatLiteralToU32.call.loc97, %.loc97_72.1 [concrete = <error>]
// CHECK:STDOUT:   %u32.loc97: type = type_literal constants.%u32 [concrete = constants.%u32]
// CHECK:STDOUT:   %literal_overflow_unsigned_neg: %u32 = value_binding literal_overflow_unsigned_neg, %.loc97_72.2 [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %literal_overflow_bin_unsigned_neg.patt: %pattern_type.668 = value_binding_pattern literal_overflow_bin_unsigned_neg [concrete = constants.%literal_overflow_bin_unsigned_neg.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc104_79.1: %u32 = value_of_initializer @__global_init.%ConvertFloatLiteralToU32.call.loc104 [concrete = <error>]
// CHECK:STDOUT:   %.loc104_79.2: %u32 = converted @__global_init.%ConvertFloatLiteralToU32.call.loc104, %.loc104_79.1 [concrete = <error>]
// CHECK:STDOUT:   %u32.loc104: type = type_literal constants.%u32 [concrete = constants.%u32]
// CHECK:STDOUT:   %literal_overflow_bin_unsigned_neg: %u32 = value_binding literal_overflow_bin_unsigned_neg, %.loc104_79.2 [concrete = <error>]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %ConvertF64ToI32.ref.loc9: %ConvertF64ToI32.type = name_ref ConvertF64ToI32, imports.%Main.ConvertF64ToI32 [concrete = constants.%ConvertF64ToI32]
// CHECK:STDOUT:   %float.loc9: Core.FloatLiteral = float_literal_value 21474836480e-1 [concrete = constants.%float.bab60c.1]
// CHECK:STDOUT:   %impl.elem0.loc9: %.2d7 = impl_witness_access constants.%ImplicitAs.impl_witness.43c, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.cf0]
// CHECK:STDOUT:   %bound_method.loc9_37.1: <bound method> = bound_method %float.loc9, %impl.elem0.loc9 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.d5f]
// CHECK:STDOUT:   %specific_fn.loc9: <specific function> = specific_function %impl.elem0.loc9, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_37.2: <bound method> = bound_method %float.loc9, %specific_fn.loc9 [concrete = constants.%bound_method.a8b]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %f64.dc1 = call %bound_method.loc9_37.2(%float.loc9) [concrete = constants.%float.d5c]
// CHECK:STDOUT:   %.loc9_37.1: %f64.dc1 = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%float.d5c]
// CHECK:STDOUT:   %.loc9_37.2: %f64.dc1 = converted %float.loc9, %.loc9_37.1 [concrete = constants.%float.d5c]
// CHECK:STDOUT:   %ConvertF64ToI32.call.loc9: init %i32 = call %ConvertF64ToI32.ref.loc9(%.loc9_37.2) [concrete = <error>]
// CHECK:STDOUT:   %ConvertF64ToI32.ref.loc15: %ConvertF64ToI32.type = name_ref ConvertF64ToI32, imports.%Main.ConvertF64ToI32 [concrete = constants.%ConvertF64ToI32]
// CHECK:STDOUT:   %Negate.ref.loc15: %Negate.type = name_ref Negate, imports.%Main.Negate [concrete = constants.%Negate]
// CHECK:STDOUT:   %float.loc15: Core.FloatLiteral = float_literal_value 21474836490e-1 [concrete = constants.%float.449]
// CHECK:STDOUT:   %impl.elem0.loc15: %.2d7 = impl_witness_access constants.%ImplicitAs.impl_witness.43c, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.cf0]
// CHECK:STDOUT:   %bound_method.loc15_45.1: <bound method> = bound_method %float.loc15, %impl.elem0.loc15 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.92c]
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem0.loc15, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_45.2: <bound method> = bound_method %float.loc15, %specific_fn.loc15 [concrete = constants.%bound_method.af3]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc15: init %f64.dc1 = call %bound_method.loc15_45.2(%float.loc15) [concrete = constants.%float.f2a]
// CHECK:STDOUT:   %.loc15_45.1: %f64.dc1 = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc15 [concrete = constants.%float.f2a]
// CHECK:STDOUT:   %.loc15_45.2: %f64.dc1 = converted %float.loc15, %.loc15_45.1 [concrete = constants.%float.f2a]
// CHECK:STDOUT:   %Negate.call.loc15: init %f64.dc1 = call %Negate.ref.loc15(%.loc15_45.2) [concrete = constants.%float.caa]
// CHECK:STDOUT:   %.loc15_57.1: %f64.dc1 = value_of_initializer %Negate.call.loc15 [concrete = constants.%float.caa]
// CHECK:STDOUT:   %.loc15_57.2: %f64.dc1 = converted %Negate.call.loc15, %.loc15_57.1 [concrete = constants.%float.caa]
// CHECK:STDOUT:   %ConvertF64ToI32.call.loc15: init %i32 = call %ConvertF64ToI32.ref.loc15(%.loc15_57.2) [concrete = <error>]
// CHECK:STDOUT:   %ConvertF64ToU32.ref: %ConvertF64ToU32.type = name_ref ConvertF64ToU32, imports.%Main.ConvertF64ToU32 [concrete = constants.%ConvertF64ToU32]
// CHECK:STDOUT:   %Negate.ref.loc21: %Negate.type = name_ref Negate, imports.%Main.Negate [concrete = constants.%Negate]
// CHECK:STDOUT:   %float.loc21: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %impl.elem0.loc21: %.2d7 = impl_witness_access constants.%ImplicitAs.impl_witness.43c, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.cf0]
// CHECK:STDOUT:   %bound_method.loc21_48.1: <bound method> = bound_method %float.loc21, %impl.elem0.loc21 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.757]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem0.loc21, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc21_48.2: <bound method> = bound_method %float.loc21, %specific_fn.loc21 [concrete = constants.%bound_method.143]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc21: init %f64.dc1 = call %bound_method.loc21_48.2(%float.loc21) [concrete = constants.%float.3b9]
// CHECK:STDOUT:   %.loc21_48.1: %f64.dc1 = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc21 [concrete = constants.%float.3b9]
// CHECK:STDOUT:   %.loc21_48.2: %f64.dc1 = converted %float.loc21, %.loc21_48.1 [concrete = constants.%float.3b9]
// CHECK:STDOUT:   %Negate.call.loc21: init %f64.dc1 = call %Negate.ref.loc21(%.loc21_48.2) [concrete = constants.%float.b2f]
// CHECK:STDOUT:   %.loc21_51.1: %f64.dc1 = value_of_initializer %Negate.call.loc21 [concrete = constants.%float.b2f]
// CHECK:STDOUT:   %.loc21_51.2: %f64.dc1 = converted %Negate.call.loc21, %.loc21_51.1 [concrete = constants.%float.b2f]
// CHECK:STDOUT:   %ConvertF64ToU32.call: init %u32 = call %ConvertF64ToU32.ref(%.loc21_51.2) [concrete = <error>]
// CHECK:STDOUT:   %ConvertFloatLiteralToI32.ref.loc27: %ConvertFloatLiteralToI32.type = name_ref ConvertFloatLiteralToI32, imports.%Main.ConvertFloatLiteralToI32 [concrete = constants.%ConvertFloatLiteralToI32]
// CHECK:STDOUT:   %float.loc27: Core.FloatLiteral = float_literal_value 21474836480e-1 [concrete = constants.%float.bab60c.2]
// CHECK:STDOUT:   %ConvertFloatLiteralToI32.call.loc27: init %i32 = call %ConvertFloatLiteralToI32.ref.loc27(%float.loc27) [concrete = <error>]
// CHECK:STDOUT:   %ConvertFloatLiteralToI32.ref.loc33: %ConvertFloatLiteralToI32.type = name_ref ConvertFloatLiteralToI32, imports.%Main.ConvertFloatLiteralToI32 [concrete = constants.%ConvertFloatLiteralToI32]
// CHECK:STDOUT:   %float.loc33: Core.FloatLiteral = float_literal_value 10e999999 [concrete = constants.%float.53d]
// CHECK:STDOUT:   %ConvertFloatLiteralToI32.call.loc33: init %i32 = call %ConvertFloatLiteralToI32.ref.loc33(%float.loc33) [concrete = <error>]
// CHECK:STDOUT:   %ConvertFloatLiteralToIntLiteral.ref.loc40: %ConvertFloatLiteralToIntLiteral.type = name_ref ConvertFloatLiteralToIntLiteral, imports.%Main.ConvertFloatLiteralToIntLiteral [concrete = constants.%ConvertFloatLiteralToIntLiteral]
// CHECK:STDOUT:   %float.loc40: Core.FloatLiteral = float_literal_value 10e9999999 [concrete = constants.%float.ef09]
// CHECK:STDOUT:   %ConvertFloatLiteralToIntLiteral.call.loc40: init Core.IntLiteral = call %ConvertFloatLiteralToIntLiteral.ref.loc40(%float.loc40) [concrete = <error>]
// CHECK:STDOUT:   %ConvertFloatLiteralToI32.ref.loc47: %ConvertFloatLiteralToI32.type = name_ref ConvertFloatLiteralToI32, imports.%Main.ConvertFloatLiteralToI32 [concrete = constants.%ConvertFloatLiteralToI32]
// CHECK:STDOUT:   %float.loc47: Core.FloatLiteral = float_literal_value 10e999999999999999999999999999999999 [concrete = constants.%float.f461ee.1]
// CHECK:STDOUT:   %ConvertFloatLiteralToI32.call.loc47: init %i32 = call %ConvertFloatLiteralToI32.ref.loc47(%float.loc47) [concrete = <error>]
// CHECK:STDOUT:   %ConvertFloatLiteralToIntLiteral.ref.loc55: %ConvertFloatLiteralToIntLiteral.type = name_ref ConvertFloatLiteralToIntLiteral, imports.%Main.ConvertFloatLiteralToIntLiteral [concrete = constants.%ConvertFloatLiteralToIntLiteral]
// CHECK:STDOUT:   %float.loc55: Core.FloatLiteral = float_literal_value 10e999999999999999999999999999999999 [concrete = constants.%float.f461ee.2]
// CHECK:STDOUT:   %ConvertFloatLiteralToIntLiteral.call.loc55: init Core.IntLiteral = call %ConvertFloatLiteralToIntLiteral.ref.loc55(%float.loc55) [concrete = <error>]
// CHECK:STDOUT:   %ConvertFloatLiteralToI32.ref.loc62: %ConvertFloatLiteralToI32.type = name_ref ConvertFloatLiteralToI32, imports.%Main.ConvertFloatLiteralToI32 [concrete = constants.%ConvertFloatLiteralToI32]
// CHECK:STDOUT:   %float.loc62: Core.FloatLiteral = float_literal_value 2147483648e0 [concrete = constants.%float.e12922.1]
// CHECK:STDOUT:   %ConvertFloatLiteralToI32.call.loc62: init %i32 = call %ConvertFloatLiteralToI32.ref.loc62(%float.loc62) [concrete = <error>]
// CHECK:STDOUT:   %ConvertFloatLiteralToI32.ref.loc69: %ConvertFloatLiteralToI32.type = name_ref ConvertFloatLiteralToI32, imports.%Main.ConvertFloatLiteralToI32 [concrete = constants.%ConvertFloatLiteralToI32]
// CHECK:STDOUT:   %float.loc69: Core.FloatLiteral = float_literal_value 2147483648e0 [concrete = constants.%float.e12922.2]
// CHECK:STDOUT:   %ConvertFloatLiteralToI32.call.loc69: init %i32 = call %ConvertFloatLiteralToI32.ref.loc69(%float.loc69) [concrete = <error>]
// CHECK:STDOUT:   %ConvertFloatLiteralToI32.ref.loc76: %ConvertFloatLiteralToI32.type = name_ref ConvertFloatLiteralToI32, imports.%Main.ConvertFloatLiteralToI32 [concrete = constants.%ConvertFloatLiteralToI32]
// CHECK:STDOUT:   %float.loc76: Core.FloatLiteral = float_literal_value 2147483648p0 [concrete = constants.%float.ef03]
// CHECK:STDOUT:   %ConvertFloatLiteralToI32.call.loc76: init %i32 = call %ConvertFloatLiteralToI32.ref.loc76(%float.loc76) [concrete = <error>]
// CHECK:STDOUT:   %ConvertFloatLiteralToI32.ref.loc83: %ConvertFloatLiteralToI32.type = name_ref ConvertFloatLiteralToI32, imports.%Main.ConvertFloatLiteralToI32 [concrete = constants.%ConvertFloatLiteralToI32]
// CHECK:STDOUT:   %float.loc83: Core.FloatLiteral = float_literal_value 10e9 [concrete = constants.%float.9d5]
// CHECK:STDOUT:   %ConvertFloatLiteralToI32.call.loc83: init %i32 = call %ConvertFloatLiteralToI32.ref.loc83(%float.loc83) [concrete = <error>]
// CHECK:STDOUT:   %ConvertFloatLiteralToI32.ref.loc90: %ConvertFloatLiteralToI32.type = name_ref ConvertFloatLiteralToI32, imports.%Main.ConvertFloatLiteralToI32 [concrete = constants.%ConvertFloatLiteralToI32]
// CHECK:STDOUT:   %float.loc90: Core.FloatLiteral = float_literal_value 16p27 [concrete = constants.%float.c0c]
// CHECK:STDOUT:   %ConvertFloatLiteralToI32.call.loc90: init %i32 = call %ConvertFloatLiteralToI32.ref.loc90(%float.loc90) [concrete = <error>]
// CHECK:STDOUT:   %ConvertFloatLiteralToU32.ref.loc97: %ConvertFloatLiteralToU32.type = name_ref ConvertFloatLiteralToU32, imports.%Main.ConvertFloatLiteralToU32 [concrete = constants.%ConvertFloatLiteralToU32]
// CHECK:STDOUT:   %float.loc97: Core.FloatLiteral = float_literal_value 50e8 [concrete = constants.%float.4f4]
// CHECK:STDOUT:   %ConvertFloatLiteralToU32.call.loc97: init %u32 = call %ConvertFloatLiteralToU32.ref.loc97(%float.loc97) [concrete = <error>]
// CHECK:STDOUT:   %ConvertFloatLiteralToU32.ref.loc104: %ConvertFloatLiteralToU32.type = name_ref ConvertFloatLiteralToU32, imports.%Main.ConvertFloatLiteralToU32 [concrete = constants.%ConvertFloatLiteralToU32]
// CHECK:STDOUT:   %float.loc104: Core.FloatLiteral = float_literal_value 16p28 [concrete = constants.%float.97b]
// CHECK:STDOUT:   %ConvertFloatLiteralToU32.call.loc104: init %u32 = call %ConvertFloatLiteralToU32.ref.loc104(%float.loc104) [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_nan_inf.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %nan_val.patt: %pattern_type.6b6 = value_binding_pattern nan_val [concrete]
// CHECK:STDOUT:   %ConvertF64ToI32.type: type = fn_type @ConvertF64ToI32 [concrete]
// CHECK:STDOUT:   %ConvertF64ToI32: %ConvertF64ToI32.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %f64.dc1: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %Div.type: type = fn_type @Div [concrete]
// CHECK:STDOUT:   %Div: %Div.type = struct_value () [concrete]
// CHECK:STDOUT:   %float.1f75e4.1: Core.FloatLiteral = float_literal_value 0e-1 [concrete]
// CHECK:STDOUT:   %float.1f75e4.2: Core.FloatLiteral = float_literal_value 0e-1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.91e: type = facet_type <@ImplicitAs, @ImplicitAs(%f64.dc1)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.063: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.d4f: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.063 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.43c: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table, @Core.FloatLiteral.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.ef0: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.cf0: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.ef0 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.91e = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.43c) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.6c5: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%f64.dc1, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.2d7: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.6c5, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.af318c.1: <bound method> = bound_method %float.1f75e4.1, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.cf0 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.ImplicitAs.impl.Convert.cf0, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.819b88.1: <bound method> = bound_method %float.1f75e4.1, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.895: %f64.dc1 = float_value 0 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.af318c.2: <bound method> = bound_method %float.1f75e4.2, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.cf0 [concrete]
// CHECK:STDOUT:   %bound_method.819b88.2: <bound method> = bound_method %float.1f75e4.2, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.a8a: %f64.dc1 = float_value NaN [concrete]
// CHECK:STDOUT:   %inf_val.patt: %pattern_type.6b6 = value_binding_pattern inf_val [concrete]
// CHECK:STDOUT:   %float.6da: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %float.1f75e4.3: Core.FloatLiteral = float_literal_value 0e-1 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.757: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.cf0 [concrete]
// CHECK:STDOUT:   %bound_method.143: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.3b9: %f64.dc1 = float_value 1 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.af318c.3: <bound method> = bound_method %float.1f75e4.3, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.cf0 [concrete]
// CHECK:STDOUT:   %bound_method.819b88.3: <bound method> = bound_method %float.1f75e4.3, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.7ae: %f64.dc1 = float_value +Inf [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.ConvertF64ToI32: %ConvertF64ToI32.type = import_ref Main//float_to_int, ConvertF64ToI32, loaded [concrete = constants.%ConvertF64ToI32]
// CHECK:STDOUT:   %Main.Div: %Div.type = import_ref Main//float_to_int, Div, loaded [concrete = constants.%Div]
// CHECK:STDOUT:   %Core.import_ref.c4c: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.063) = import_ref Core//prelude/parts/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.d4f)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%Core.import_ref.c4c), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %nan_val.patt: %pattern_type.6b6 = value_binding_pattern nan_val [concrete = constants.%nan_val.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9_49.1: %i32 = value_of_initializer @__global_init.%ConvertF64ToI32.call.loc9 [concrete = <error>]
// CHECK:STDOUT:   %.loc9_49.2: %i32 = converted @__global_init.%ConvertF64ToI32.call.loc9, %.loc9_49.1 [concrete = <error>]
// CHECK:STDOUT:   %i32.loc9: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %nan_val: %i32 = value_binding nan_val, %.loc9_49.2 [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %inf_val.patt: %pattern_type.6b6 = value_binding_pattern inf_val [concrete = constants.%inf_val.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc15_49.1: %i32 = value_of_initializer @__global_init.%ConvertF64ToI32.call.loc15 [concrete = <error>]
// CHECK:STDOUT:   %.loc15_49.2: %i32 = converted @__global_init.%ConvertF64ToI32.call.loc15, %.loc15_49.1 [concrete = <error>]
// CHECK:STDOUT:   %i32.loc15: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %inf_val: %i32 = value_binding inf_val, %.loc15_49.2 [concrete = <error>]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %ConvertF64ToI32.ref.loc9: %ConvertF64ToI32.type = name_ref ConvertF64ToI32, imports.%Main.ConvertF64ToI32 [concrete = constants.%ConvertF64ToI32]
// CHECK:STDOUT:   %Div.ref.loc9: %Div.type = name_ref Div, imports.%Main.Div [concrete = constants.%Div]
// CHECK:STDOUT:   %float.loc9_40: Core.FloatLiteral = float_literal_value 0e-1 [concrete = constants.%float.1f75e4.1]
// CHECK:STDOUT:   %float.loc9_45: Core.FloatLiteral = float_literal_value 0e-1 [concrete = constants.%float.1f75e4.2]
// CHECK:STDOUT:   %impl.elem0.loc9_40: %.2d7 = impl_witness_access constants.%ImplicitAs.impl_witness.43c, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.cf0]
// CHECK:STDOUT:   %bound_method.loc9_40.1: <bound method> = bound_method %float.loc9_40, %impl.elem0.loc9_40 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.af318c.1]
// CHECK:STDOUT:   %specific_fn.loc9_40: <specific function> = specific_function %impl.elem0.loc9_40, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_40.2: <bound method> = bound_method %float.loc9_40, %specific_fn.loc9_40 [concrete = constants.%bound_method.819b88.1]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc9_40: init %f64.dc1 = call %bound_method.loc9_40.2(%float.loc9_40) [concrete = constants.%float.895]
// CHECK:STDOUT:   %.loc9_40.1: %f64.dc1 = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc9_40 [concrete = constants.%float.895]
// CHECK:STDOUT:   %.loc9_40.2: %f64.dc1 = converted %float.loc9_40, %.loc9_40.1 [concrete = constants.%float.895]
// CHECK:STDOUT:   %impl.elem0.loc9_45: %.2d7 = impl_witness_access constants.%ImplicitAs.impl_witness.43c, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.cf0]
// CHECK:STDOUT:   %bound_method.loc9_45.1: <bound method> = bound_method %float.loc9_45, %impl.elem0.loc9_45 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.af318c.2]
// CHECK:STDOUT:   %specific_fn.loc9_45: <specific function> = specific_function %impl.elem0.loc9_45, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_45.2: <bound method> = bound_method %float.loc9_45, %specific_fn.loc9_45 [concrete = constants.%bound_method.819b88.2]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc9_45: init %f64.dc1 = call %bound_method.loc9_45.2(%float.loc9_45) [concrete = constants.%float.895]
// CHECK:STDOUT:   %.loc9_45.1: %f64.dc1 = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc9_45 [concrete = constants.%float.895]
// CHECK:STDOUT:   %.loc9_45.2: %f64.dc1 = converted %float.loc9_45, %.loc9_45.1 [concrete = constants.%float.895]
// CHECK:STDOUT:   %Div.call.loc9: init %f64.dc1 = call %Div.ref.loc9(%.loc9_40.2, %.loc9_45.2) [concrete = constants.%float.a8a]
// CHECK:STDOUT:   %.loc9_48.1: %f64.dc1 = value_of_initializer %Div.call.loc9 [concrete = constants.%float.a8a]
// CHECK:STDOUT:   %.loc9_48.2: %f64.dc1 = converted %Div.call.loc9, %.loc9_48.1 [concrete = constants.%float.a8a]
// CHECK:STDOUT:   %ConvertF64ToI32.call.loc9: init %i32 = call %ConvertF64ToI32.ref.loc9(%.loc9_48.2) [concrete = <error>]
// CHECK:STDOUT:   %ConvertF64ToI32.ref.loc15: %ConvertF64ToI32.type = name_ref ConvertF64ToI32, imports.%Main.ConvertF64ToI32 [concrete = constants.%ConvertF64ToI32]
// CHECK:STDOUT:   %Div.ref.loc15: %Div.type = name_ref Div, imports.%Main.Div [concrete = constants.%Div]
// CHECK:STDOUT:   %float.loc15_40: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %float.loc15_45: Core.FloatLiteral = float_literal_value 0e-1 [concrete = constants.%float.1f75e4.3]
// CHECK:STDOUT:   %impl.elem0.loc15_40: %.2d7 = impl_witness_access constants.%ImplicitAs.impl_witness.43c, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.cf0]
// CHECK:STDOUT:   %bound_method.loc15_40.1: <bound method> = bound_method %float.loc15_40, %impl.elem0.loc15_40 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.757]
// CHECK:STDOUT:   %specific_fn.loc15_40: <specific function> = specific_function %impl.elem0.loc15_40, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_40.2: <bound method> = bound_method %float.loc15_40, %specific_fn.loc15_40 [concrete = constants.%bound_method.143]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc15_40: init %f64.dc1 = call %bound_method.loc15_40.2(%float.loc15_40) [concrete = constants.%float.3b9]
// CHECK:STDOUT:   %.loc15_40.1: %f64.dc1 = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc15_40 [concrete = constants.%float.3b9]
// CHECK:STDOUT:   %.loc15_40.2: %f64.dc1 = converted %float.loc15_40, %.loc15_40.1 [concrete = constants.%float.3b9]
// CHECK:STDOUT:   %impl.elem0.loc15_45: %.2d7 = impl_witness_access constants.%ImplicitAs.impl_witness.43c, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.cf0]
// CHECK:STDOUT:   %bound_method.loc15_45.1: <bound method> = bound_method %float.loc15_45, %impl.elem0.loc15_45 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.af318c.3]
// CHECK:STDOUT:   %specific_fn.loc15_45: <specific function> = specific_function %impl.elem0.loc15_45, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_45.2: <bound method> = bound_method %float.loc15_45, %specific_fn.loc15_45 [concrete = constants.%bound_method.819b88.3]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc15_45: init %f64.dc1 = call %bound_method.loc15_45.2(%float.loc15_45) [concrete = constants.%float.895]
// CHECK:STDOUT:   %.loc15_45.1: %f64.dc1 = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc15_45 [concrete = constants.%float.895]
// CHECK:STDOUT:   %.loc15_45.2: %f64.dc1 = converted %float.loc15_45, %.loc15_45.1 [concrete = constants.%float.895]
// CHECK:STDOUT:   %Div.call.loc15: init %f64.dc1 = call %Div.ref.loc15(%.loc15_40.2, %.loc15_45.2) [concrete = constants.%float.7ae]
// CHECK:STDOUT:   %.loc15_48.1: %f64.dc1 = value_of_initializer %Div.call.loc15 [concrete = constants.%float.7ae]
// CHECK:STDOUT:   %.loc15_48.2: %f64.dc1 = converted %Div.call.loc15, %.loc15_48.1 [concrete = constants.%float.7ae]
// CHECK:STDOUT:   %ConvertF64ToI32.call.loc15: init %i32 = call %ConvertF64ToI32.ref.loc15(%.loc15_48.2) [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:

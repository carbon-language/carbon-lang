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

class Expect[T: type](N: T) {}
fn Test[T: type](generic N: T) -> Expect(N) { return {}; }

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
// CHECK:STDERR:        ╭─┤ fail_overflow.carbon:[[@LINE+8]]:21
// CHECK:STDERR: ->     │ let overflow: i32 = ConvertF64ToI32(2147483648.0);
// CHECK:STDERR:        ·                     ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        ·                     ┗━━┫ error: floating-point value 2147483648 too large for
// CHECK:STDERR:        ·                        ┃        integer type `i32`
// CHECK:STDERR:        ·                        ┃        [FloatTooLargeForIntType]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
let overflow: i32 = ConvertF64ToI32(2147483648.0);

// CHECK:STDERR:        ╭─┤ fail_overflow.carbon:[[@LINE+8]]:22
// CHECK:STDERR: ->     │ let underflow: i32 = ConvertF64ToI32(Negate(2147483649.0));
// CHECK:STDERR:        ·                      ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        ·                      ┗━━┫ error: floating-point value -2147483649 too large for
// CHECK:STDERR:        ·                         ┃        integer type `i32`
// CHECK:STDERR:        ·                         ┃        [FloatTooLargeForIntType]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
let underflow: i32 = ConvertF64ToI32(Negate(2147483649.0));

// CHECK:STDERR:        ╭─┤ fail_overflow.carbon:[[@LINE+8]]:25
// CHECK:STDERR: ->     │ let unsigned_neg: u32 = ConvertF64ToU32(Negate(1.0));
// CHECK:STDERR:        ·                         ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        ·                         ┗━━┫ error: floating-point value -1 too large for
// CHECK:STDERR:        ·                            ┃        integer type `u32`
// CHECK:STDERR:        ·                            ┃        [FloatTooLargeForIntType]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
let unsigned_neg: u32 = ConvertF64ToU32(Negate(1.0));

// CHECK:STDERR:        ╭─┤ fail_overflow.carbon:[[@LINE+8]]:29
// CHECK:STDERR: ->     │ let literal_overflow: i32 = ConvertFloatLiteralToI32(2147483648.0);
// CHECK:STDERR:        ·                             ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        ·                             ┗━━┫ error: floating-point value 21474836480*10^-1
// CHECK:STDERR:        ·                                ┃        too large for integer type `i32`
// CHECK:STDERR:        ·                                ┃        [RealLiteralTooLargeForIntType]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
let literal_overflow: i32 = ConvertFloatLiteralToI32(2147483648.0);

// CHECK:STDERR:        ╭─┤ fail_overflow.carbon:[[@LINE+8]]:35
// CHECK:STDERR: ->     │ let giant_literal_overflow: i32 = ConvertFloatLiteralToI32(1.0e1000000);
// CHECK:STDERR:        ·                                   ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        ·                                   ┗━━┫ error: floating-point value 10*10^999999
// CHECK:STDERR:        ·                                      ┃        too large for integer type `i32`
// CHECK:STDERR:        ·                                      ┃        [RealLiteralTooLargeForIntType]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
let giant_literal_overflow: i32 = ConvertFloatLiteralToI32(1.0e1000000);

//
// CHECK:STDERR:        ╭─┤ fail_overflow.carbon:[[@LINE+11]]:62
// CHECK:STDERR: ->     │ ...overflow: Core.IntLiteral = ConvertFloatLiteralToIntLiteral(1.0e10000000);
// CHECK:STDERR:        ·                                ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        ·                                ┗━━┫ error: floating-point value 10*10^9999999
// CHECK:STDERR:        ·                                   ┃        too large to convert: result would be
// CHECK:STDERR:        ·                                   ┃        an integer whose width is greater
// CHECK:STDERR:        ·                                   ┃        than the maximum supported width of
// CHECK:STDERR:        ·                                   ┃        8388608
// CHECK:STDERR:        ·                                   ┃        [RealLiteralTooLargeForUnsizedInt]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
let giant_literal_to_int_literal_overflow: Core.IntLiteral = ConvertFloatLiteralToIntLiteral(1.0e10000000);

//
// CHECK:STDERR:        ╭─┤ fail_overflow.carbon:[[@LINE+9]]:44
// CHECK:STDERR: ->     │ ...l_overflow: i32 = ConvertFloatLiteralToI32(1.0e100000000000000000000000000...
// CHECK:STDERR:        ·                      ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        ·                      ┗━━┫ error: floating-point value
// CHECK:STDERR:        ·                         ┃        10*10^999999999999999999999999999999999 too
// CHECK:STDERR:        ·                         ┃        large for integer type `i32`
// CHECK:STDERR:        ·                         ┃        [RealLiteralTooLargeForIntType]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
let giant_exponent_literal_overflow: i32 = ConvertFloatLiteralToI32(1.0e1000000000000000000000000000000000);

//
//
// CHECK:STDERR:        ╭─┤ fail_overflow.carbon:[[@LINE+11]]:71
// CHECK:STDERR: ->     │ ...Core.IntLiteral = ConvertFloatLiteralToIntLiteral(1.0e10000000000000000000...
// CHECK:STDERR:        ·                      ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        ·                      ┗━━┫ error: floating-point value
// CHECK:STDERR:        ·                         ┃        10*10^999999999999999999999999999999999 too
// CHECK:STDERR:        ·                         ┃        large to convert: result would be an integer
// CHECK:STDERR:        ·                         ┃        whose width is greater than the maximum
// CHECK:STDERR:        ·                         ┃        supported width of 8388608
// CHECK:STDERR:        ·                         ┃        [RealLiteralTooLargeForUnsizedInt]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
let giant_exponent_literal_to_int_literal_overflow: Core.IntLiteral = ConvertFloatLiteralToIntLiteral(1.0e1000000000000000000000000000000000);

//
// CHECK:STDERR:        ╭─┤ fail_overflow.carbon:[[@LINE+9]]:44
// CHECK:STDERR: ->     │ ...al_overflow_large_mantissa: i32 = ConvertFloatLiteralToI32(214748364.8e1);
// CHECK:STDERR:        ·                                      ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        ·                                      ┗━━┫ error: floating-point value
// CHECK:STDERR:        ·                                         ┃        2147483648*10^0 too large for
// CHECK:STDERR:        ·                                         ┃        integer type `i32`
// CHECK:STDERR:        ·                                         ┃        [RealLiteralTooLargeForIntType]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
let literal_overflow_large_mantissa: i32 = ConvertFloatLiteralToI32(214748364.8e1);

//
// CHECK:STDERR:        ╭─┤ fail_overflow.carbon:[[@LINE+9]]:42
// CHECK:STDERR: ->     │ ...eral_overflow_med_mantissa: i32 = ConvertFloatLiteralToI32(21474.83648e5);
// CHECK:STDERR:        ·                                      ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        ·                                      ┗━━┫ error: floating-point value
// CHECK:STDERR:        ·                                         ┃        2147483648*10^0 too large for
// CHECK:STDERR:        ·                                         ┃        integer type `i32`
// CHECK:STDERR:        ·                                         ┃        [RealLiteralTooLargeForIntType]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
let literal_overflow_med_mantissa: i32 = ConvertFloatLiteralToI32(21474.83648e5);

//
// CHECK:STDERR:        ╭─┤ fail_overflow.carbon:[[@LINE+9]]:46
// CHECK:STDERR: ->     │ ...overflow_hex_med_mantissa: i32 = ConvertFloatLiteralToI32(0x8000.0000p16);
// CHECK:STDERR:        ·                                     ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        ·                                     ┗━━┫ error: floating-point value
// CHECK:STDERR:        ·                                        ┃        2147483648*2^0 too large for
// CHECK:STDERR:        ·                                        ┃        integer type `i32`
// CHECK:STDERR:        ·                                        ┃        [RealLiteralTooLargeForIntType]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
let literal_overflow_hex_med_mantissa: i32 = ConvertFloatLiteralToI32(0x8000.0000p16);

//
// CHECK:STDERR:        ╭─┤ fail_overflow.carbon:[[@LINE+9]]:44
// CHECK:STDERR: ->     │ let literal_overflow_small_mantissa: i32 = ConvertFloatLiteralToI32(1.0e10);
// CHECK:STDERR:        ·                                            ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        ·                                            ┗━━┫ error: floating-point value
// CHECK:STDERR:        ·                                               ┃        10*10^9 too large for
// CHECK:STDERR:        ·                                               ┃        integer type `i32`
// CHECK:STDERR:        ·                                               ┃        [RealLiteralTooLargeForIntType]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
let literal_overflow_small_mantissa: i32 = ConvertFloatLiteralToI32(1.0e10);

//
// CHECK:STDERR:        ╭─┤ fail_overflow.carbon:[[@LINE+9]]:48
// CHECK:STDERR: ->     │ ...ral_overflow_bin_small_mantissa: i32 = ConvertFloatLiteralToI32(0x1.0p31);
// CHECK:STDERR:        ·                                           ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        ·                                           ┗━━┫ error: floating-point value
// CHECK:STDERR:        ·                                              ┃        16*2^27 too large for
// CHECK:STDERR:        ·                                              ┃        integer type `i32`
// CHECK:STDERR:        ·                                              ┃        [RealLiteralTooLargeForIntType]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
let literal_overflow_bin_small_mantissa: i32 = ConvertFloatLiteralToI32(0x1.0p31);

//
// CHECK:STDERR:        ╭─┤ fail_overflow.carbon:[[@LINE+9]]:42
// CHECK:STDERR: ->     │ let literal_overflow_unsigned_neg: u32 = ConvertFloatLiteralToU32(5.0e9);
// CHECK:STDERR:        ·                                          ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        ·                                          ┗━━┫ error: floating-point value
// CHECK:STDERR:        ·                                             ┃        50*10^8 too large for
// CHECK:STDERR:        ·                                             ┃        integer type `u32`
// CHECK:STDERR:        ·                                             ┃        [RealLiteralTooLargeForIntType]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
let literal_overflow_unsigned_neg: u32 = ConvertFloatLiteralToU32(5.0e9);

//
// CHECK:STDERR:        ╭─┤ fail_overflow.carbon:[[@LINE+9]]:46
// CHECK:STDERR: ->     │ let literal_overflow_bin_unsigned_neg: u32 = ConvertFloatLiteralToU32(0x1.0p32);
// CHECK:STDERR:        ·                                              ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        ·                                              ┗━━┫ error: floating-point value
// CHECK:STDERR:        ·                                                 ┃        16*2^28 too large for
// CHECK:STDERR:        ·                                                 ┃        integer type `u32`
// CHECK:STDERR:        ·                                                 ┃        [RealLiteralTooLargeForIntType]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
let literal_overflow_bin_unsigned_neg: u32 = ConvertFloatLiteralToU32(0x1.0p32);
//@dump-sem-ir-end

// --- fail_nan_inf.carbon
library "[[@TEST_NAME]]";
import library "float_to_int";

//@dump-sem-ir-begin
// CHECK:STDERR:        ╭─┤ fail_nan_inf.carbon:[[@LINE+7]]:20
// CHECK:STDERR: ->     │ let nan_val: i32 = ConvertF64ToI32(Div(0.0, 0.0));
// CHECK:STDERR:        ·                    ━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        ·                             ┗━━┫ error: cannot convert NaN to integer type `i32`
// CHECK:STDERR:        ·                                ┃        [FloatNaNConvertedToInt]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
let nan_val: i32 = ConvertF64ToI32(Div(0.0, 0.0));

// CHECK:STDERR:        ╭─┤ fail_nan_inf.carbon:[[@LINE+7]]:20
// CHECK:STDERR: ->     │ let inf_val: i32 = ConvertF64ToI32(Div(1.0, 0.0));
// CHECK:STDERR:        ·                    ━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        ·                        ┗━━┫ error: cannot convert infinity to integer type `i32`
// CHECK:STDERR:        ·                           ┃        [FloatInfinityConvertedToInt]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
let inf_val: i32 = ConvertF64ToI32(Div(1.0, 0.0));
//@dump-sem-ir-end

// CHECK:STDOUT: --- fail_overflow.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = parameterized_class_name_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.dc0: type = pattern_type Core.IntLiteral [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %overflow.patt: %pattern_type.d94 = value_binding_pattern overflow [concrete]
// CHECK:STDOUT:   %ConvertF64ToI32.type: type = fn_type @ConvertF64ToI32 [concrete]
// CHECK:STDOUT:   %ConvertF64ToI32: %ConvertF64ToI32.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %f64.761: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %float.bab60c.1: Core.FloatLiteral = float_literal_value 21474836480e-1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.068: type = facet_type <@ImplicitAs, @ImplicitAs(%f64.761)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.c1a: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.48f: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.c1a = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.1d6: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table, @Core.FloatLiteral.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.b39: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.9d0: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.b39 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.068 = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.1d6) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.62d: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%f64.761, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.b5b: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.62d, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.9eb: <bound method> = bound_method %float.bab60c.1, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.9d0 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.ImplicitAs.impl.Convert.9d0, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.847: <bound method> = bound_method %float.bab60c.1, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.ab7: %f64.761 = float_value 2147483648 [concrete]
// CHECK:STDOUT:   %underflow.patt: %pattern_type.d94 = value_binding_pattern underflow [concrete]
// CHECK:STDOUT:   %Negate.type: type = fn_type @Negate [concrete]
// CHECK:STDOUT:   %Negate: %Negate.type = struct_value () [concrete]
// CHECK:STDOUT:   %float.449: Core.FloatLiteral = float_literal_value 21474836490e-1 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.00d: <bound method> = bound_method %float.449, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.9d0 [concrete]
// CHECK:STDOUT:   %bound_method.a16: <bound method> = bound_method %float.449, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.8e0: %f64.761 = float_value 2147483649 [concrete]
// CHECK:STDOUT:   %float.5e1: %f64.761 = float_value -2147483649 [concrete]
// CHECK:STDOUT:   %UInt.type: type = parameterized_class_name_type @UInt [concrete]
// CHECK:STDOUT:   %UInt.generic: %UInt.type = struct_value () [concrete]
// CHECK:STDOUT:   %u32: type = class_type @UInt, @UInt(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.129: type = pattern_type %u32 [concrete]
// CHECK:STDOUT:   %unsigned_neg.patt: %pattern_type.129 = value_binding_pattern unsigned_neg [concrete]
// CHECK:STDOUT:   %ConvertF64ToU32.type: type = fn_type @ConvertF64ToU32 [concrete]
// CHECK:STDOUT:   %ConvertF64ToU32: %ConvertF64ToU32.type = struct_value () [concrete]
// CHECK:STDOUT:   %float.6da: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.1a9: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.9d0 [concrete]
// CHECK:STDOUT:   %bound_method.04a: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.a71: %f64.761 = float_value 1 [concrete]
// CHECK:STDOUT:   %float.c4b: %f64.761 = float_value -1 [concrete]
// CHECK:STDOUT:   %literal_overflow.patt: %pattern_type.d94 = value_binding_pattern literal_overflow [concrete]
// CHECK:STDOUT:   %ConvertFloatLiteralToI32.type: type = fn_type @ConvertFloatLiteralToI32 [concrete]
// CHECK:STDOUT:   %ConvertFloatLiteralToI32: %ConvertFloatLiteralToI32.type = struct_value () [concrete]
// CHECK:STDOUT:   %float.bab60c.2: Core.FloatLiteral = float_literal_value 21474836480e-1 [concrete]
// CHECK:STDOUT:   %giant_literal_overflow.patt: %pattern_type.d94 = value_binding_pattern giant_literal_overflow [concrete]
// CHECK:STDOUT:   %float.53d: Core.FloatLiteral = float_literal_value 10e999999 [concrete]
// CHECK:STDOUT:   %giant_literal_to_int_literal_overflow.patt: %pattern_type.dc0 = value_binding_pattern giant_literal_to_int_literal_overflow [concrete]
// CHECK:STDOUT:   %ConvertFloatLiteralToIntLiteral.type: type = fn_type @ConvertFloatLiteralToIntLiteral [concrete]
// CHECK:STDOUT:   %ConvertFloatLiteralToIntLiteral: %ConvertFloatLiteralToIntLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %float.ef09: Core.FloatLiteral = float_literal_value 10e9999999 [concrete]
// CHECK:STDOUT:   %giant_exponent_literal_overflow.patt: %pattern_type.d94 = value_binding_pattern giant_exponent_literal_overflow [concrete]
// CHECK:STDOUT:   %float.f461ee.1: Core.FloatLiteral = float_literal_value 10e999999999999999999999999999999999 [concrete]
// CHECK:STDOUT:   %giant_exponent_literal_to_int_literal_overflow.patt: %pattern_type.dc0 = value_binding_pattern giant_exponent_literal_to_int_literal_overflow [concrete]
// CHECK:STDOUT:   %float.f461ee.2: Core.FloatLiteral = float_literal_value 10e999999999999999999999999999999999 [concrete]
// CHECK:STDOUT:   %literal_overflow_large_mantissa.patt: %pattern_type.d94 = value_binding_pattern literal_overflow_large_mantissa [concrete]
// CHECK:STDOUT:   %float.e12922.1: Core.FloatLiteral = float_literal_value 2147483648e0 [concrete]
// CHECK:STDOUT:   %literal_overflow_med_mantissa.patt: %pattern_type.d94 = value_binding_pattern literal_overflow_med_mantissa [concrete]
// CHECK:STDOUT:   %float.e12922.2: Core.FloatLiteral = float_literal_value 2147483648e0 [concrete]
// CHECK:STDOUT:   %literal_overflow_hex_med_mantissa.patt: %pattern_type.d94 = value_binding_pattern literal_overflow_hex_med_mantissa [concrete]
// CHECK:STDOUT:   %float.ef03: Core.FloatLiteral = float_literal_value 2147483648p0 [concrete]
// CHECK:STDOUT:   %literal_overflow_small_mantissa.patt: %pattern_type.d94 = value_binding_pattern literal_overflow_small_mantissa [concrete]
// CHECK:STDOUT:   %float.9d5: Core.FloatLiteral = float_literal_value 10e9 [concrete]
// CHECK:STDOUT:   %literal_overflow_bin_small_mantissa.patt: %pattern_type.d94 = value_binding_pattern literal_overflow_bin_small_mantissa [concrete]
// CHECK:STDOUT:   %float.c0c: Core.FloatLiteral = float_literal_value 16p27 [concrete]
// CHECK:STDOUT:   %literal_overflow_unsigned_neg.patt: %pattern_type.129 = value_binding_pattern literal_overflow_unsigned_neg [concrete]
// CHECK:STDOUT:   %ConvertFloatLiteralToU32.type: type = fn_type @ConvertFloatLiteralToU32 [concrete]
// CHECK:STDOUT:   %ConvertFloatLiteralToU32: %ConvertFloatLiteralToU32.type = struct_value () [concrete]
// CHECK:STDOUT:   %float.4f4: Core.FloatLiteral = float_literal_value 50e8 [concrete]
// CHECK:STDOUT:   %literal_overflow_bin_unsigned_neg.patt: %pattern_type.129 = value_binding_pattern literal_overflow_bin_unsigned_neg [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.0c7: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.c1a) = import_ref Core//prelude/parts/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.48f)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%Core.import_ref.0c7), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.UInt: %UInt.type = import_ref Core//prelude/parts/uint, UInt, loaded [concrete = constants.%UInt.generic]
// CHECK:STDOUT:   %Core.IntLiteral: type = import_ref Core//prelude/parts/int_literal, IntLiteral, loaded [concrete = Core.IntLiteral]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %.loc13_49.1: %i32 = value_of_initializer @__global_init.%ConvertF64ToI32.call.loc13 [concrete = <error>]
// CHECK:STDOUT:   %.loc13_49.2: %i32 = converted @__global_init.%ConvertF64ToI32.call.loc13, %.loc13_49.1 [concrete = <error>]
// CHECK:STDOUT:   %i32.loc13: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %overflow: %i32 = wrapper_binding overflow, %.loc13_49.2 [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %overflow.patt: %pattern_type.d94 = value_binding_pattern overflow [concrete = constants.%overflow.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc23_58.1: %i32 = value_of_initializer @__global_init.%ConvertF64ToI32.call.loc23 [concrete = <error>]
// CHECK:STDOUT:   %.loc23_58.2: %i32 = converted @__global_init.%ConvertF64ToI32.call.loc23, %.loc23_58.1 [concrete = <error>]
// CHECK:STDOUT:   %i32.loc23: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %underflow: %i32 = wrapper_binding underflow, %.loc23_58.2 [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %underflow.patt: %pattern_type.d94 = value_binding_pattern underflow [concrete = constants.%underflow.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc33_52.1: %u32 = value_of_initializer @__global_init.%ConvertF64ToU32.call [concrete = <error>]
// CHECK:STDOUT:   %.loc33_52.2: %u32 = converted @__global_init.%ConvertF64ToU32.call, %.loc33_52.1 [concrete = <error>]
// CHECK:STDOUT:   %u32.loc33: type = type_literal constants.%u32 [concrete = constants.%u32]
// CHECK:STDOUT:   %unsigned_neg: %u32 = wrapper_binding unsigned_neg, %.loc33_52.2 [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %unsigned_neg.patt: %pattern_type.129 = value_binding_pattern unsigned_neg [concrete = constants.%unsigned_neg.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc43_66.1: %i32 = value_of_initializer @__global_init.%ConvertFloatLiteralToI32.call.loc43 [concrete = <error>]
// CHECK:STDOUT:   %.loc43_66.2: %i32 = converted @__global_init.%ConvertFloatLiteralToI32.call.loc43, %.loc43_66.1 [concrete = <error>]
// CHECK:STDOUT:   %i32.loc43: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %literal_overflow: %i32 = wrapper_binding literal_overflow, %.loc43_66.2 [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %literal_overflow.patt: %pattern_type.d94 = value_binding_pattern literal_overflow [concrete = constants.%literal_overflow.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc53_71.1: %i32 = value_of_initializer @__global_init.%ConvertFloatLiteralToI32.call.loc53 [concrete = <error>]
// CHECK:STDOUT:   %.loc53_71.2: %i32 = converted @__global_init.%ConvertFloatLiteralToI32.call.loc53, %.loc53_71.1 [concrete = <error>]
// CHECK:STDOUT:   %i32.loc53: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %giant_literal_overflow: %i32 = wrapper_binding giant_literal_overflow, %.loc53_71.2 [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %giant_literal_overflow.patt: %pattern_type.d94 = value_binding_pattern giant_literal_overflow [concrete = constants.%giant_literal_overflow.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc67_106.1: Core.IntLiteral = value_of_initializer @__global_init.%ConvertFloatLiteralToIntLiteral.call.loc67 [concrete = <error>]
// CHECK:STDOUT:   %.loc67_106.2: Core.IntLiteral = converted @__global_init.%ConvertFloatLiteralToIntLiteral.call.loc67, %.loc67_106.1 [concrete = <error>]
// CHECK:STDOUT:   %.loc67_48: type = splice_block %IntLiteral.ref.loc67 [concrete = Core.IntLiteral] {
// CHECK:STDOUT:     %Core.ref.loc67: <namespace> = name_ref Core, imports.%Core.048aa3.1 [concrete = imports.%Core.048aa3.1]
// CHECK:STDOUT:     %IntLiteral.ref.loc67: type = name_ref IntLiteral, imports.%Core.IntLiteral [concrete = Core.IntLiteral]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %giant_literal_to_int_literal_overflow: Core.IntLiteral = wrapper_binding giant_literal_to_int_literal_overflow, %.loc67_106.2 [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %giant_literal_to_int_literal_overflow.patt: %pattern_type.dc0 = value_binding_pattern giant_literal_to_int_literal_overflow [concrete = constants.%giant_literal_to_int_literal_overflow.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc79_107.1: %i32 = value_of_initializer @__global_init.%ConvertFloatLiteralToI32.call.loc79 [concrete = <error>]
// CHECK:STDOUT:   %.loc79_107.2: %i32 = converted @__global_init.%ConvertFloatLiteralToI32.call.loc79, %.loc79_107.1 [concrete = <error>]
// CHECK:STDOUT:   %i32.loc79: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %giant_exponent_literal_overflow: %i32 = wrapper_binding giant_exponent_literal_overflow, %.loc79_107.2 [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %giant_exponent_literal_overflow.patt: %pattern_type.d94 = value_binding_pattern giant_exponent_literal_overflow [concrete = constants.%giant_exponent_literal_overflow.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc94_141.1: Core.IntLiteral = value_of_initializer @__global_init.%ConvertFloatLiteralToIntLiteral.call.loc94 [concrete = <error>]
// CHECK:STDOUT:   %.loc94_141.2: Core.IntLiteral = converted @__global_init.%ConvertFloatLiteralToIntLiteral.call.loc94, %.loc94_141.1 [concrete = <error>]
// CHECK:STDOUT:   %.loc94_57: type = splice_block %IntLiteral.ref.loc94 [concrete = Core.IntLiteral] {
// CHECK:STDOUT:     %Core.ref.loc94: <namespace> = name_ref Core, imports.%Core.048aa3.1 [concrete = imports.%Core.048aa3.1]
// CHECK:STDOUT:     %IntLiteral.ref.loc94: type = name_ref IntLiteral, imports.%Core.IntLiteral [concrete = Core.IntLiteral]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %giant_exponent_literal_to_int_literal_overflow: Core.IntLiteral = wrapper_binding giant_exponent_literal_to_int_literal_overflow, %.loc94_141.2 [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %giant_exponent_literal_to_int_literal_overflow.patt: %pattern_type.dc0 = value_binding_pattern giant_exponent_literal_to_int_literal_overflow [concrete = constants.%giant_exponent_literal_to_int_literal_overflow.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc106_82.1: %i32 = value_of_initializer @__global_init.%ConvertFloatLiteralToI32.call.loc106 [concrete = <error>]
// CHECK:STDOUT:   %.loc106_82.2: %i32 = converted @__global_init.%ConvertFloatLiteralToI32.call.loc106, %.loc106_82.1 [concrete = <error>]
// CHECK:STDOUT:   %i32.loc106: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %literal_overflow_large_mantissa: %i32 = wrapper_binding literal_overflow_large_mantissa, %.loc106_82.2 [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %literal_overflow_large_mantissa.patt: %pattern_type.d94 = value_binding_pattern literal_overflow_large_mantissa [concrete = constants.%literal_overflow_large_mantissa.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc118_80.1: %i32 = value_of_initializer @__global_init.%ConvertFloatLiteralToI32.call.loc118 [concrete = <error>]
// CHECK:STDOUT:   %.loc118_80.2: %i32 = converted @__global_init.%ConvertFloatLiteralToI32.call.loc118, %.loc118_80.1 [concrete = <error>]
// CHECK:STDOUT:   %i32.loc118: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %literal_overflow_med_mantissa: %i32 = wrapper_binding literal_overflow_med_mantissa, %.loc118_80.2 [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %literal_overflow_med_mantissa.patt: %pattern_type.d94 = value_binding_pattern literal_overflow_med_mantissa [concrete = constants.%literal_overflow_med_mantissa.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc130_85.1: %i32 = value_of_initializer @__global_init.%ConvertFloatLiteralToI32.call.loc130 [concrete = <error>]
// CHECK:STDOUT:   %.loc130_85.2: %i32 = converted @__global_init.%ConvertFloatLiteralToI32.call.loc130, %.loc130_85.1 [concrete = <error>]
// CHECK:STDOUT:   %i32.loc130: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %literal_overflow_hex_med_mantissa: %i32 = wrapper_binding literal_overflow_hex_med_mantissa, %.loc130_85.2 [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %literal_overflow_hex_med_mantissa.patt: %pattern_type.d94 = value_binding_pattern literal_overflow_hex_med_mantissa [concrete = constants.%literal_overflow_hex_med_mantissa.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc142_75.1: %i32 = value_of_initializer @__global_init.%ConvertFloatLiteralToI32.call.loc142 [concrete = <error>]
// CHECK:STDOUT:   %.loc142_75.2: %i32 = converted @__global_init.%ConvertFloatLiteralToI32.call.loc142, %.loc142_75.1 [concrete = <error>]
// CHECK:STDOUT:   %i32.loc142: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %literal_overflow_small_mantissa: %i32 = wrapper_binding literal_overflow_small_mantissa, %.loc142_75.2 [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %literal_overflow_small_mantissa.patt: %pattern_type.d94 = value_binding_pattern literal_overflow_small_mantissa [concrete = constants.%literal_overflow_small_mantissa.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc154_81.1: %i32 = value_of_initializer @__global_init.%ConvertFloatLiteralToI32.call.loc154 [concrete = <error>]
// CHECK:STDOUT:   %.loc154_81.2: %i32 = converted @__global_init.%ConvertFloatLiteralToI32.call.loc154, %.loc154_81.1 [concrete = <error>]
// CHECK:STDOUT:   %i32.loc154: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %literal_overflow_bin_small_mantissa: %i32 = wrapper_binding literal_overflow_bin_small_mantissa, %.loc154_81.2 [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %literal_overflow_bin_small_mantissa.patt: %pattern_type.d94 = value_binding_pattern literal_overflow_bin_small_mantissa [concrete = constants.%literal_overflow_bin_small_mantissa.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc166_72.1: %u32 = value_of_initializer @__global_init.%ConvertFloatLiteralToU32.call.loc166 [concrete = <error>]
// CHECK:STDOUT:   %.loc166_72.2: %u32 = converted @__global_init.%ConvertFloatLiteralToU32.call.loc166, %.loc166_72.1 [concrete = <error>]
// CHECK:STDOUT:   %u32.loc166: type = type_literal constants.%u32 [concrete = constants.%u32]
// CHECK:STDOUT:   %literal_overflow_unsigned_neg: %u32 = wrapper_binding literal_overflow_unsigned_neg, %.loc166_72.2 [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %literal_overflow_unsigned_neg.patt: %pattern_type.129 = value_binding_pattern literal_overflow_unsigned_neg [concrete = constants.%literal_overflow_unsigned_neg.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc178_79.1: %u32 = value_of_initializer @__global_init.%ConvertFloatLiteralToU32.call.loc178 [concrete = <error>]
// CHECK:STDOUT:   %.loc178_79.2: %u32 = converted @__global_init.%ConvertFloatLiteralToU32.call.loc178, %.loc178_79.1 [concrete = <error>]
// CHECK:STDOUT:   %u32.loc178: type = type_literal constants.%u32 [concrete = constants.%u32]
// CHECK:STDOUT:   %literal_overflow_bin_unsigned_neg: %u32 = wrapper_binding literal_overflow_bin_unsigned_neg, %.loc178_79.2 [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %literal_overflow_bin_unsigned_neg.patt: %pattern_type.129 = value_binding_pattern literal_overflow_bin_unsigned_neg [concrete = constants.%literal_overflow_bin_unsigned_neg.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %ConvertF64ToI32.ref.loc13: %ConvertF64ToI32.type = name_ref ConvertF64ToI32, imports.%Main.ConvertF64ToI32 [concrete = constants.%ConvertF64ToI32]
// CHECK:STDOUT:   %float.loc13: Core.FloatLiteral = float_literal_value 21474836480e-1 [concrete = constants.%float.bab60c.1]
// CHECK:STDOUT:   %impl.elem0.loc13: %.b5b = impl_witness_access constants.%ImplicitAs.impl_witness.1d6, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.9d0]
// CHECK:STDOUT:   %bound_method.loc13_37.1: <bound method> = bound_method %float.loc13, %impl.elem0.loc13 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.9eb]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_37.2: <bound method> = bound_method %float.loc13, %specific_fn.loc13 [concrete = constants.%bound_method.847]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc13: init %f64.761 = call %bound_method.loc13_37.2(%float.loc13) [concrete = constants.%float.ab7]
// CHECK:STDOUT:   %.loc13_37.1: %f64.761 = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc13 [concrete = constants.%float.ab7]
// CHECK:STDOUT:   %.loc13_37.2: %f64.761 = converted %float.loc13, %.loc13_37.1 [concrete = constants.%float.ab7]
// CHECK:STDOUT:   %ConvertF64ToI32.call.loc13: init %i32 = call %ConvertF64ToI32.ref.loc13(%.loc13_37.2) [concrete = <error>]
// CHECK:STDOUT:   %ConvertF64ToI32.ref.loc23: %ConvertF64ToI32.type = name_ref ConvertF64ToI32, imports.%Main.ConvertF64ToI32 [concrete = constants.%ConvertF64ToI32]
// CHECK:STDOUT:   %Negate.ref.loc23: %Negate.type = name_ref Negate, imports.%Main.Negate [concrete = constants.%Negate]
// CHECK:STDOUT:   %float.loc23: Core.FloatLiteral = float_literal_value 21474836490e-1 [concrete = constants.%float.449]
// CHECK:STDOUT:   %impl.elem0.loc23: %.b5b = impl_witness_access constants.%ImplicitAs.impl_witness.1d6, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.9d0]
// CHECK:STDOUT:   %bound_method.loc23_45.1: <bound method> = bound_method %float.loc23, %impl.elem0.loc23 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.00d]
// CHECK:STDOUT:   %specific_fn.loc23: <specific function> = specific_function %impl.elem0.loc23, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc23_45.2: <bound method> = bound_method %float.loc23, %specific_fn.loc23 [concrete = constants.%bound_method.a16]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc23: init %f64.761 = call %bound_method.loc23_45.2(%float.loc23) [concrete = constants.%float.8e0]
// CHECK:STDOUT:   %.loc23_45.1: %f64.761 = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc23 [concrete = constants.%float.8e0]
// CHECK:STDOUT:   %.loc23_45.2: %f64.761 = converted %float.loc23, %.loc23_45.1 [concrete = constants.%float.8e0]
// CHECK:STDOUT:   %Negate.call.loc23: init %f64.761 = call %Negate.ref.loc23(%.loc23_45.2) [concrete = constants.%float.5e1]
// CHECK:STDOUT:   %.loc23_57.1: %f64.761 = value_of_initializer %Negate.call.loc23 [concrete = constants.%float.5e1]
// CHECK:STDOUT:   %.loc23_57.2: %f64.761 = converted %Negate.call.loc23, %.loc23_57.1 [concrete = constants.%float.5e1]
// CHECK:STDOUT:   %ConvertF64ToI32.call.loc23: init %i32 = call %ConvertF64ToI32.ref.loc23(%.loc23_57.2) [concrete = <error>]
// CHECK:STDOUT:   %ConvertF64ToU32.ref: %ConvertF64ToU32.type = name_ref ConvertF64ToU32, imports.%Main.ConvertF64ToU32 [concrete = constants.%ConvertF64ToU32]
// CHECK:STDOUT:   %Negate.ref.loc33: %Negate.type = name_ref Negate, imports.%Main.Negate [concrete = constants.%Negate]
// CHECK:STDOUT:   %float.loc33: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %impl.elem0.loc33: %.b5b = impl_witness_access constants.%ImplicitAs.impl_witness.1d6, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.9d0]
// CHECK:STDOUT:   %bound_method.loc33_48.1: <bound method> = bound_method %float.loc33, %impl.elem0.loc33 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.1a9]
// CHECK:STDOUT:   %specific_fn.loc33: <specific function> = specific_function %impl.elem0.loc33, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc33_48.2: <bound method> = bound_method %float.loc33, %specific_fn.loc33 [concrete = constants.%bound_method.04a]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc33: init %f64.761 = call %bound_method.loc33_48.2(%float.loc33) [concrete = constants.%float.a71]
// CHECK:STDOUT:   %.loc33_48.1: %f64.761 = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc33 [concrete = constants.%float.a71]
// CHECK:STDOUT:   %.loc33_48.2: %f64.761 = converted %float.loc33, %.loc33_48.1 [concrete = constants.%float.a71]
// CHECK:STDOUT:   %Negate.call.loc33: init %f64.761 = call %Negate.ref.loc33(%.loc33_48.2) [concrete = constants.%float.c4b]
// CHECK:STDOUT:   %.loc33_51.1: %f64.761 = value_of_initializer %Negate.call.loc33 [concrete = constants.%float.c4b]
// CHECK:STDOUT:   %.loc33_51.2: %f64.761 = converted %Negate.call.loc33, %.loc33_51.1 [concrete = constants.%float.c4b]
// CHECK:STDOUT:   %ConvertF64ToU32.call: init %u32 = call %ConvertF64ToU32.ref(%.loc33_51.2) [concrete = <error>]
// CHECK:STDOUT:   %ConvertFloatLiteralToI32.ref.loc43: %ConvertFloatLiteralToI32.type = name_ref ConvertFloatLiteralToI32, imports.%Main.ConvertFloatLiteralToI32 [concrete = constants.%ConvertFloatLiteralToI32]
// CHECK:STDOUT:   %float.loc43: Core.FloatLiteral = float_literal_value 21474836480e-1 [concrete = constants.%float.bab60c.2]
// CHECK:STDOUT:   %ConvertFloatLiteralToI32.call.loc43: init %i32 = call %ConvertFloatLiteralToI32.ref.loc43(%float.loc43) [concrete = <error>]
// CHECK:STDOUT:   %ConvertFloatLiteralToI32.ref.loc53: %ConvertFloatLiteralToI32.type = name_ref ConvertFloatLiteralToI32, imports.%Main.ConvertFloatLiteralToI32 [concrete = constants.%ConvertFloatLiteralToI32]
// CHECK:STDOUT:   %float.loc53: Core.FloatLiteral = float_literal_value 10e999999 [concrete = constants.%float.53d]
// CHECK:STDOUT:   %ConvertFloatLiteralToI32.call.loc53: init %i32 = call %ConvertFloatLiteralToI32.ref.loc53(%float.loc53) [concrete = <error>]
// CHECK:STDOUT:   %ConvertFloatLiteralToIntLiteral.ref.loc67: %ConvertFloatLiteralToIntLiteral.type = name_ref ConvertFloatLiteralToIntLiteral, imports.%Main.ConvertFloatLiteralToIntLiteral [concrete = constants.%ConvertFloatLiteralToIntLiteral]
// CHECK:STDOUT:   %float.loc67: Core.FloatLiteral = float_literal_value 10e9999999 [concrete = constants.%float.ef09]
// CHECK:STDOUT:   %ConvertFloatLiteralToIntLiteral.call.loc67: init Core.IntLiteral = call %ConvertFloatLiteralToIntLiteral.ref.loc67(%float.loc67) [concrete = <error>]
// CHECK:STDOUT:   %ConvertFloatLiteralToI32.ref.loc79: %ConvertFloatLiteralToI32.type = name_ref ConvertFloatLiteralToI32, imports.%Main.ConvertFloatLiteralToI32 [concrete = constants.%ConvertFloatLiteralToI32]
// CHECK:STDOUT:   %float.loc79: Core.FloatLiteral = float_literal_value 10e999999999999999999999999999999999 [concrete = constants.%float.f461ee.1]
// CHECK:STDOUT:   %ConvertFloatLiteralToI32.call.loc79: init %i32 = call %ConvertFloatLiteralToI32.ref.loc79(%float.loc79) [concrete = <error>]
// CHECK:STDOUT:   %ConvertFloatLiteralToIntLiteral.ref.loc94: %ConvertFloatLiteralToIntLiteral.type = name_ref ConvertFloatLiteralToIntLiteral, imports.%Main.ConvertFloatLiteralToIntLiteral [concrete = constants.%ConvertFloatLiteralToIntLiteral]
// CHECK:STDOUT:   %float.loc94: Core.FloatLiteral = float_literal_value 10e999999999999999999999999999999999 [concrete = constants.%float.f461ee.2]
// CHECK:STDOUT:   %ConvertFloatLiteralToIntLiteral.call.loc94: init Core.IntLiteral = call %ConvertFloatLiteralToIntLiteral.ref.loc94(%float.loc94) [concrete = <error>]
// CHECK:STDOUT:   %ConvertFloatLiteralToI32.ref.loc106: %ConvertFloatLiteralToI32.type = name_ref ConvertFloatLiteralToI32, imports.%Main.ConvertFloatLiteralToI32 [concrete = constants.%ConvertFloatLiteralToI32]
// CHECK:STDOUT:   %float.loc106: Core.FloatLiteral = float_literal_value 2147483648e0 [concrete = constants.%float.e12922.1]
// CHECK:STDOUT:   %ConvertFloatLiteralToI32.call.loc106: init %i32 = call %ConvertFloatLiteralToI32.ref.loc106(%float.loc106) [concrete = <error>]
// CHECK:STDOUT:   %ConvertFloatLiteralToI32.ref.loc118: %ConvertFloatLiteralToI32.type = name_ref ConvertFloatLiteralToI32, imports.%Main.ConvertFloatLiteralToI32 [concrete = constants.%ConvertFloatLiteralToI32]
// CHECK:STDOUT:   %float.loc118: Core.FloatLiteral = float_literal_value 2147483648e0 [concrete = constants.%float.e12922.2]
// CHECK:STDOUT:   %ConvertFloatLiteralToI32.call.loc118: init %i32 = call %ConvertFloatLiteralToI32.ref.loc118(%float.loc118) [concrete = <error>]
// CHECK:STDOUT:   %ConvertFloatLiteralToI32.ref.loc130: %ConvertFloatLiteralToI32.type = name_ref ConvertFloatLiteralToI32, imports.%Main.ConvertFloatLiteralToI32 [concrete = constants.%ConvertFloatLiteralToI32]
// CHECK:STDOUT:   %float.loc130: Core.FloatLiteral = float_literal_value 2147483648p0 [concrete = constants.%float.ef03]
// CHECK:STDOUT:   %ConvertFloatLiteralToI32.call.loc130: init %i32 = call %ConvertFloatLiteralToI32.ref.loc130(%float.loc130) [concrete = <error>]
// CHECK:STDOUT:   %ConvertFloatLiteralToI32.ref.loc142: %ConvertFloatLiteralToI32.type = name_ref ConvertFloatLiteralToI32, imports.%Main.ConvertFloatLiteralToI32 [concrete = constants.%ConvertFloatLiteralToI32]
// CHECK:STDOUT:   %float.loc142: Core.FloatLiteral = float_literal_value 10e9 [concrete = constants.%float.9d5]
// CHECK:STDOUT:   %ConvertFloatLiteralToI32.call.loc142: init %i32 = call %ConvertFloatLiteralToI32.ref.loc142(%float.loc142) [concrete = <error>]
// CHECK:STDOUT:   %ConvertFloatLiteralToI32.ref.loc154: %ConvertFloatLiteralToI32.type = name_ref ConvertFloatLiteralToI32, imports.%Main.ConvertFloatLiteralToI32 [concrete = constants.%ConvertFloatLiteralToI32]
// CHECK:STDOUT:   %float.loc154: Core.FloatLiteral = float_literal_value 16p27 [concrete = constants.%float.c0c]
// CHECK:STDOUT:   %ConvertFloatLiteralToI32.call.loc154: init %i32 = call %ConvertFloatLiteralToI32.ref.loc154(%float.loc154) [concrete = <error>]
// CHECK:STDOUT:   %ConvertFloatLiteralToU32.ref.loc166: %ConvertFloatLiteralToU32.type = name_ref ConvertFloatLiteralToU32, imports.%Main.ConvertFloatLiteralToU32 [concrete = constants.%ConvertFloatLiteralToU32]
// CHECK:STDOUT:   %float.loc166: Core.FloatLiteral = float_literal_value 50e8 [concrete = constants.%float.4f4]
// CHECK:STDOUT:   %ConvertFloatLiteralToU32.call.loc166: init %u32 = call %ConvertFloatLiteralToU32.ref.loc166(%float.loc166) [concrete = <error>]
// CHECK:STDOUT:   %ConvertFloatLiteralToU32.ref.loc178: %ConvertFloatLiteralToU32.type = name_ref ConvertFloatLiteralToU32, imports.%Main.ConvertFloatLiteralToU32 [concrete = constants.%ConvertFloatLiteralToU32]
// CHECK:STDOUT:   %float.loc178: Core.FloatLiteral = float_literal_value 16p28 [concrete = constants.%float.97b]
// CHECK:STDOUT:   %ConvertFloatLiteralToU32.call.loc178: init %u32 = call %ConvertFloatLiteralToU32.ref.loc178(%float.loc178) [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_nan_inf.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %nan_val.patt: %pattern_type.d94 = value_binding_pattern nan_val [concrete]
// CHECK:STDOUT:   %ConvertF64ToI32.type: type = fn_type @ConvertF64ToI32 [concrete]
// CHECK:STDOUT:   %ConvertF64ToI32: %ConvertF64ToI32.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %f64.761: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %Div.type: type = fn_type @Div [concrete]
// CHECK:STDOUT:   %Div: %Div.type = struct_value () [concrete]
// CHECK:STDOUT:   %float.1f75e4.1: Core.FloatLiteral = float_literal_value 0e-1 [concrete]
// CHECK:STDOUT:   %float.1f75e4.2: Core.FloatLiteral = float_literal_value 0e-1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.068: type = facet_type <@ImplicitAs, @ImplicitAs(%f64.761)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.c1a: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.48f: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.c1a = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.1d6: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table, @Core.FloatLiteral.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.b39: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.9d0: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.b39 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.068 = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.1d6) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.62d: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%f64.761, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.b5b: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.62d, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.7c43ce.1: <bound method> = bound_method %float.1f75e4.1, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.9d0 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.ImplicitAs.impl.Convert.9d0, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.49396d.1: <bound method> = bound_method %float.1f75e4.1, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.73e: %f64.761 = float_value 0 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.7c43ce.2: <bound method> = bound_method %float.1f75e4.2, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.9d0 [concrete]
// CHECK:STDOUT:   %bound_method.49396d.2: <bound method> = bound_method %float.1f75e4.2, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.61c: %f64.761 = float_value NaN [concrete]
// CHECK:STDOUT:   %inf_val.patt: %pattern_type.d94 = value_binding_pattern inf_val [concrete]
// CHECK:STDOUT:   %float.6da: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %float.1f75e4.3: Core.FloatLiteral = float_literal_value 0e-1 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.1a9: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.9d0 [concrete]
// CHECK:STDOUT:   %bound_method.04a: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.a71: %f64.761 = float_value 1 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.7c43ce.3: <bound method> = bound_method %float.1f75e4.3, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.9d0 [concrete]
// CHECK:STDOUT:   %bound_method.49396d.3: <bound method> = bound_method %float.1f75e4.3, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.62c: %f64.761 = float_value +Inf [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.ConvertF64ToI32: %ConvertF64ToI32.type = import_ref Main//float_to_int, ConvertF64ToI32, loaded [concrete = constants.%ConvertF64ToI32]
// CHECK:STDOUT:   %Main.Div: %Div.type = import_ref Main//float_to_int, Div, loaded [concrete = constants.%Div]
// CHECK:STDOUT:   %Core.import_ref.0c7: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.c1a) = import_ref Core//prelude/parts/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.48f)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%Core.import_ref.0c7), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %.loc12_49.1: %i32 = value_of_initializer @__global_init.%ConvertF64ToI32.call.loc12 [concrete = <error>]
// CHECK:STDOUT:   %.loc12_49.2: %i32 = converted @__global_init.%ConvertF64ToI32.call.loc12, %.loc12_49.1 [concrete = <error>]
// CHECK:STDOUT:   %i32.loc12: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %nan_val: %i32 = wrapper_binding nan_val, %.loc12_49.2 [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %nan_val.patt: %pattern_type.d94 = value_binding_pattern nan_val [concrete = constants.%nan_val.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc21_49.1: %i32 = value_of_initializer @__global_init.%ConvertF64ToI32.call.loc21 [concrete = <error>]
// CHECK:STDOUT:   %.loc21_49.2: %i32 = converted @__global_init.%ConvertF64ToI32.call.loc21, %.loc21_49.1 [concrete = <error>]
// CHECK:STDOUT:   %i32.loc21: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %inf_val: %i32 = wrapper_binding inf_val, %.loc21_49.2 [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %inf_val.patt: %pattern_type.d94 = value_binding_pattern inf_val [concrete = constants.%inf_val.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %ConvertF64ToI32.ref.loc12: %ConvertF64ToI32.type = name_ref ConvertF64ToI32, imports.%Main.ConvertF64ToI32 [concrete = constants.%ConvertF64ToI32]
// CHECK:STDOUT:   %Div.ref.loc12: %Div.type = name_ref Div, imports.%Main.Div [concrete = constants.%Div]
// CHECK:STDOUT:   %float.loc12_40: Core.FloatLiteral = float_literal_value 0e-1 [concrete = constants.%float.1f75e4.1]
// CHECK:STDOUT:   %float.loc12_45: Core.FloatLiteral = float_literal_value 0e-1 [concrete = constants.%float.1f75e4.2]
// CHECK:STDOUT:   %impl.elem0.loc12_40: %.b5b = impl_witness_access constants.%ImplicitAs.impl_witness.1d6, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.9d0]
// CHECK:STDOUT:   %bound_method.loc12_40.1: <bound method> = bound_method %float.loc12_40, %impl.elem0.loc12_40 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.7c43ce.1]
// CHECK:STDOUT:   %specific_fn.loc12_40: <specific function> = specific_function %impl.elem0.loc12_40, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_40.2: <bound method> = bound_method %float.loc12_40, %specific_fn.loc12_40 [concrete = constants.%bound_method.49396d.1]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc12_40: init %f64.761 = call %bound_method.loc12_40.2(%float.loc12_40) [concrete = constants.%float.73e]
// CHECK:STDOUT:   %.loc12_40.1: %f64.761 = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc12_40 [concrete = constants.%float.73e]
// CHECK:STDOUT:   %.loc12_40.2: %f64.761 = converted %float.loc12_40, %.loc12_40.1 [concrete = constants.%float.73e]
// CHECK:STDOUT:   %impl.elem0.loc12_45: %.b5b = impl_witness_access constants.%ImplicitAs.impl_witness.1d6, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.9d0]
// CHECK:STDOUT:   %bound_method.loc12_45.1: <bound method> = bound_method %float.loc12_45, %impl.elem0.loc12_45 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.7c43ce.2]
// CHECK:STDOUT:   %specific_fn.loc12_45: <specific function> = specific_function %impl.elem0.loc12_45, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_45.2: <bound method> = bound_method %float.loc12_45, %specific_fn.loc12_45 [concrete = constants.%bound_method.49396d.2]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc12_45: init %f64.761 = call %bound_method.loc12_45.2(%float.loc12_45) [concrete = constants.%float.73e]
// CHECK:STDOUT:   %.loc12_45.1: %f64.761 = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc12_45 [concrete = constants.%float.73e]
// CHECK:STDOUT:   %.loc12_45.2: %f64.761 = converted %float.loc12_45, %.loc12_45.1 [concrete = constants.%float.73e]
// CHECK:STDOUT:   %Div.call.loc12: init %f64.761 = call %Div.ref.loc12(%.loc12_40.2, %.loc12_45.2) [concrete = constants.%float.61c]
// CHECK:STDOUT:   %.loc12_48.1: %f64.761 = value_of_initializer %Div.call.loc12 [concrete = constants.%float.61c]
// CHECK:STDOUT:   %.loc12_48.2: %f64.761 = converted %Div.call.loc12, %.loc12_48.1 [concrete = constants.%float.61c]
// CHECK:STDOUT:   %ConvertF64ToI32.call.loc12: init %i32 = call %ConvertF64ToI32.ref.loc12(%.loc12_48.2) [concrete = <error>]
// CHECK:STDOUT:   %ConvertF64ToI32.ref.loc21: %ConvertF64ToI32.type = name_ref ConvertF64ToI32, imports.%Main.ConvertF64ToI32 [concrete = constants.%ConvertF64ToI32]
// CHECK:STDOUT:   %Div.ref.loc21: %Div.type = name_ref Div, imports.%Main.Div [concrete = constants.%Div]
// CHECK:STDOUT:   %float.loc21_40: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %float.loc21_45: Core.FloatLiteral = float_literal_value 0e-1 [concrete = constants.%float.1f75e4.3]
// CHECK:STDOUT:   %impl.elem0.loc21_40: %.b5b = impl_witness_access constants.%ImplicitAs.impl_witness.1d6, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.9d0]
// CHECK:STDOUT:   %bound_method.loc21_40.1: <bound method> = bound_method %float.loc21_40, %impl.elem0.loc21_40 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.1a9]
// CHECK:STDOUT:   %specific_fn.loc21_40: <specific function> = specific_function %impl.elem0.loc21_40, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc21_40.2: <bound method> = bound_method %float.loc21_40, %specific_fn.loc21_40 [concrete = constants.%bound_method.04a]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc21_40: init %f64.761 = call %bound_method.loc21_40.2(%float.loc21_40) [concrete = constants.%float.a71]
// CHECK:STDOUT:   %.loc21_40.1: %f64.761 = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc21_40 [concrete = constants.%float.a71]
// CHECK:STDOUT:   %.loc21_40.2: %f64.761 = converted %float.loc21_40, %.loc21_40.1 [concrete = constants.%float.a71]
// CHECK:STDOUT:   %impl.elem0.loc21_45: %.b5b = impl_witness_access constants.%ImplicitAs.impl_witness.1d6, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.9d0]
// CHECK:STDOUT:   %bound_method.loc21_45.1: <bound method> = bound_method %float.loc21_45, %impl.elem0.loc21_45 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.7c43ce.3]
// CHECK:STDOUT:   %specific_fn.loc21_45: <specific function> = specific_function %impl.elem0.loc21_45, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc21_45.2: <bound method> = bound_method %float.loc21_45, %specific_fn.loc21_45 [concrete = constants.%bound_method.49396d.3]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc21_45: init %f64.761 = call %bound_method.loc21_45.2(%float.loc21_45) [concrete = constants.%float.73e]
// CHECK:STDOUT:   %.loc21_45.1: %f64.761 = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc21_45 [concrete = constants.%float.73e]
// CHECK:STDOUT:   %.loc21_45.2: %f64.761 = converted %float.loc21_45, %.loc21_45.1 [concrete = constants.%float.73e]
// CHECK:STDOUT:   %Div.call.loc21: init %f64.761 = call %Div.ref.loc21(%.loc21_40.2, %.loc21_45.2) [concrete = constants.%float.62c]
// CHECK:STDOUT:   %.loc21_48.1: %f64.761 = value_of_initializer %Div.call.loc21 [concrete = constants.%float.62c]
// CHECK:STDOUT:   %.loc21_48.2: %f64.761 = converted %Div.call.loc21, %.loc21_48.1 [concrete = constants.%float.62c]
// CHECK:STDOUT:   %ConvertF64ToI32.call.loc21: init %i32 = call %ConvertF64ToI32.ref.loc21(%.loc21_48.2) [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/primitives/type_literals.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/primitives/type_literals.carbon

// --- iN.carbon
library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
var test_i8: i8;
var test_i16: i16;
var test_i32: i32;
var test_i64: i64;
//@dump-sem-ir-end

// --- uN.carbon
library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
var test_u8: u8;
var test_u16: u16;
var test_u32: u32;
var test_u64: u64;
//@dump-sem-ir-end

// --- fN.carbon
library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
var test_f16: f16;
var test_f32: f32;
var test_f64: f64;
var test_f128: f128;
//@dump-sem-ir-end

// --- char.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
let test_char: char = 'c';
//@dump-sem-ir-end

// --- string.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
let test_str: str = "Test";
//@dump-sem-ir-end

// --- type.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
var test_type: type = i32;
//@dump-sem-ir-end

// --- fail_iN_bad_width.carbon
library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_iN_bad_width.carbon:[[@LINE+4]]:14: error: name `i0` not found [NameNotFound]
// CHECK:STDERR: var test_i0: i0;
// CHECK:STDERR:              ^~
// CHECK:STDERR:
var test_i0: i0;
// CHECK:STDERR: fail_iN_bad_width.carbon:[[@LINE+4]]:14: error: bit width of integer type literal must be a multiple of 8; use `Core.Int(1)` instead [IntWidthNotMultipleOf8]
// CHECK:STDERR: var test_i1: i1;
// CHECK:STDERR:              ^~
// CHECK:STDERR:
var test_i1: i1;
// CHECK:STDERR: fail_iN_bad_width.carbon:[[@LINE+4]]:15: error: bit width of integer type literal must be a multiple of 8; use `Core.Int(15)` instead [IntWidthNotMultipleOf8]
// CHECK:STDERR: var test_i15: i15;
// CHECK:STDERR:               ^~~
// CHECK:STDERR:
var test_i15: i15;
// CHECK:STDERR: fail_iN_bad_width.carbon:[[@LINE+4]]:14: error: bit width of integer type literal must be a multiple of 8; use `Core.Int(4)` instead [IntWidthNotMultipleOf8]
// CHECK:STDERR: var test_i4: i4;
// CHECK:STDERR:              ^~
// CHECK:STDERR:
var test_i4: i4;
// CHECK:STDERR: fail_iN_bad_width.carbon:[[@LINE+4]]:15: error: bit width of integer type literal must be a multiple of 8; use `Core.Int(12)` instead [IntWidthNotMultipleOf8]
// CHECK:STDERR: var test_i12: i12;
// CHECK:STDERR:               ^~~
// CHECK:STDERR:
var test_i12: i12;
// CHECK:STDERR: fail_iN_bad_width.carbon:[[@LINE+7]]:23: error: binding pattern has incomplete type `i1000000000` in name binding declaration [IncompleteTypeInBindingDecl]
// CHECK:STDERR: var test_i1000000000: i1000000000;
// CHECK:STDERR:                       ^~~~~~~~~~~
// CHECK:STDERR: min_prelude/parts/int.carbon:11:9: note: integer type width of 1000000000 is greater than the maximum supported width of 8388608 [IntWidthTooLarge]
// CHECK:STDERR:   adapt MakeInt(N);
// CHECK:STDERR:         ^~~~~~~~~~
// CHECK:STDERR:
var test_i1000000000: i1000000000;
// TODO: This diagnostic is not very good.
// CHECK:STDERR: fail_iN_bad_width.carbon:[[@LINE+8]]:33: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
// CHECK:STDERR: var test_i10000000000000000000: i10000000000000000000;
// CHECK:STDERR:                                 ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_iN_bad_width.carbon:[[@LINE+4]]:34: error: found a type literal with a bit width using 20 digits, which is greater than the limit of 18 [TooManyTypeBitWidthDigits]
// CHECK:STDERR: var test_i10000000000000000000: i10000000000000000000;
// CHECK:STDERR:                                  ^
// CHECK:STDERR:
var test_i10000000000000000000: i10000000000000000000;

// --- fail_uN_bad_width.carbon
library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_uN_bad_width.carbon:[[@LINE+4]]:14: error: name `u0` not found [NameNotFound]
// CHECK:STDERR: var test_u0: u0;
// CHECK:STDERR:              ^~
// CHECK:STDERR:
var test_u0: u0;
// CHECK:STDERR: fail_uN_bad_width.carbon:[[@LINE+4]]:14: error: bit width of integer type literal must be a multiple of 8; use `Core.UInt(1)` instead [IntWidthNotMultipleOf8]
// CHECK:STDERR: var test_u1: u1;
// CHECK:STDERR:              ^~
// CHECK:STDERR:
var test_u1: u1;
// CHECK:STDERR: fail_uN_bad_width.carbon:[[@LINE+4]]:15: error: bit width of integer type literal must be a multiple of 8; use `Core.UInt(15)` instead [IntWidthNotMultipleOf8]
// CHECK:STDERR: var test_u15: u15;
// CHECK:STDERR:               ^~~
// CHECK:STDERR:
var test_u15: u15;
// CHECK:STDERR: fail_uN_bad_width.carbon:[[@LINE+4]]:14: error: bit width of integer type literal must be a multiple of 8; use `Core.UInt(4)` instead [IntWidthNotMultipleOf8]
// CHECK:STDERR: var test_u4: u4;
// CHECK:STDERR:              ^~
// CHECK:STDERR:
var test_u4: u4;
// CHECK:STDERR: fail_uN_bad_width.carbon:[[@LINE+4]]:15: error: bit width of integer type literal must be a multiple of 8; use `Core.UInt(12)` instead [IntWidthNotMultipleOf8]
// CHECK:STDERR: var test_u12: u12;
// CHECK:STDERR:               ^~~
// CHECK:STDERR:
var test_u12: u12;
// CHECK:STDERR: fail_uN_bad_width.carbon:[[@LINE+7]]:23: error: binding pattern has incomplete type `u1000000000` in name binding declaration [IncompleteTypeInBindingDecl]
// CHECK:STDERR: var test_u1000000000: u1000000000;
// CHECK:STDERR:                       ^~~~~~~~~~~
// CHECK:STDERR: min_prelude/parts/uint.carbon:11:9: note: integer type width of 1000000000 is greater than the maximum supported width of 8388608 [IntWidthTooLarge]
// CHECK:STDERR:   adapt MakeUInt(N);
// CHECK:STDERR:         ^~~~~~~~~~~
// CHECK:STDERR:
var test_u1000000000: u1000000000;
// TODO: This diagnostic is not very good.
// CHECK:STDERR: fail_uN_bad_width.carbon:[[@LINE+8]]:33: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
// CHECK:STDERR: var test_u10000000000000000000: u10000000000000000000;
// CHECK:STDERR:                                 ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_uN_bad_width.carbon:[[@LINE+4]]:34: error: found a type literal with a bit width using 20 digits, which is greater than the limit of 18 [TooManyTypeBitWidthDigits]
// CHECK:STDERR: var test_u10000000000000000000: u10000000000000000000;
// CHECK:STDERR:                                  ^
// CHECK:STDERR:
var test_u10000000000000000000: u10000000000000000000;

// --- fail_fN_bad_width.carbon
library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_fN_bad_width.carbon:[[@LINE+4]]:14: error: name `f0` not found [NameNotFound]
// CHECK:STDERR: var test_f0: f0;
// CHECK:STDERR:              ^~
// CHECK:STDERR:
var test_f0: f0;
// CHECK:STDERR: fail_fN_bad_width.carbon:[[@LINE+7]]:14: error: binding pattern has incomplete type `f1` in name binding declaration [IncompleteTypeInBindingDecl]
// CHECK:STDERR: var test_f1: f1;
// CHECK:STDERR:              ^~
// CHECK:STDERR: min_prelude/parts/float.carbon:12:9: note: unsupported floating-point bit width 1 [CompileTimeFloatBitWidth]
// CHECK:STDERR:   adapt MakeFloat(N);
// CHECK:STDERR:         ^~~~~~~~~~~~
// CHECK:STDERR:
var test_f1: f1;
// CHECK:STDERR: fail_fN_bad_width.carbon:[[@LINE+7]]:15: error: binding pattern has incomplete type `f15` in name binding declaration [IncompleteTypeInBindingDecl]
// CHECK:STDERR: var test_f15: f15;
// CHECK:STDERR:               ^~~
// CHECK:STDERR: min_prelude/parts/float.carbon:12:9: note: unsupported floating-point bit width 15 [CompileTimeFloatBitWidth]
// CHECK:STDERR:   adapt MakeFloat(N);
// CHECK:STDERR:         ^~~~~~~~~~~~
// CHECK:STDERR:
var test_f15: f15;
// TODO: Decide if we want to expose the x87 80-bit floating point type this way.
// CHECK:STDERR: fail_fN_bad_width.carbon:[[@LINE+7]]:15: error: binding pattern has incomplete type `f80` in name binding declaration [IncompleteTypeInBindingDecl]
// CHECK:STDERR: var test_f80: f80;
// CHECK:STDERR:               ^~~
// CHECK:STDERR: min_prelude/parts/float.carbon:12:9: note: unsupported floating-point bit width 80 [CompileTimeFloatBitWidth]
// CHECK:STDERR:   adapt MakeFloat(N);
// CHECK:STDERR:         ^~~~~~~~~~~~
// CHECK:STDERR:
var test_f80: f80;
// CHECK:STDERR: fail_fN_bad_width.carbon:[[@LINE+7]]:16: error: binding pattern has incomplete type `f100` in name binding declaration [IncompleteTypeInBindingDecl]
// CHECK:STDERR: var test_f100: f100;
// CHECK:STDERR:                ^~~~
// CHECK:STDERR: min_prelude/parts/float.carbon:12:9: note: unsupported floating-point bit width 100 [CompileTimeFloatBitWidth]
// CHECK:STDERR:   adapt MakeFloat(N);
// CHECK:STDERR:         ^~~~~~~~~~~~
// CHECK:STDERR:
var test_f100: f100;
// CHECK:STDERR: fail_fN_bad_width.carbon:[[@LINE+7]]:23: error: binding pattern has incomplete type `f1000000000` in name binding declaration [IncompleteTypeInBindingDecl]
// CHECK:STDERR: var test_f1000000000: f1000000000;
// CHECK:STDERR:                       ^~~~~~~~~~~
// CHECK:STDERR: min_prelude/parts/float.carbon:12:9: note: unsupported floating-point bit width 1000000000 [CompileTimeFloatBitWidth]
// CHECK:STDERR:   adapt MakeFloat(N);
// CHECK:STDERR:         ^~~~~~~~~~~~
// CHECK:STDERR:
var test_f1000000000: f1000000000;
// CHECK:STDERR: fail_fN_bad_width.carbon:[[@LINE+7]]:26: error: binding pattern has incomplete type `f1000000000000` in name binding declaration [IncompleteTypeInBindingDecl]
// CHECK:STDERR: var test_f1000000000000: f1000000000000;
// CHECK:STDERR:                          ^~~~~~~~~~~~~~
// CHECK:STDERR: min_prelude/parts/float.carbon:12:9: note: unsupported floating-point bit width 1000000000000 [CompileTimeFloatBitWidth]
// CHECK:STDERR:   adapt MakeFloat(N);
// CHECK:STDERR:         ^~~~~~~~~~~~
// CHECK:STDERR:
var test_f1000000000000: f1000000000000;

// --- fail_non_type_as_type.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_non_type_as_type.carbon:[[@LINE+7]]:1: error: cannot implicitly convert non-type value of type `Core.IntLiteral` to `type` [ConversionFailureNonTypeToFacet]
// CHECK:STDERR: var x: type = 42;
// CHECK:STDERR: ^~~~~~~~~~~
// CHECK:STDERR: fail_non_type_as_type.carbon:[[@LINE+4]]:1: note: type `Core.IntLiteral` does not implement interface `Core.ImplicitAs(type)` [MissingImplInMemberAccessInContext]
// CHECK:STDERR: var x: type = 42;
// CHECK:STDERR: ^~~~~~~~~~~
// CHECK:STDERR:
var x: type = 42;

// --- fail_stringify_type_literals.carbon

library "[[@TEST_NAME]]";

// Ensure that type literals get stringified properly.

// CHECK:STDERR: fail_stringify_type_literals.carbon:[[@LINE+7]]:1: error: cannot implicitly convert expression of type `()` to `i32` [ConversionFailure]
// CHECK:STDERR: var test_i32: i32 = ();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_stringify_type_literals.carbon:[[@LINE+4]]:1: note: type `()` does not implement interface `Core.ImplicitAs(i32)` [MissingImplInMemberAccessInContext]
// CHECK:STDERR: var test_i32: i32 = ();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var test_i32: i32 = ();

// CHECK:STDERR: fail_stringify_type_literals.carbon:[[@LINE+7]]:1: error: cannot implicitly convert expression of type `()` to `f32` [ConversionFailure]
// CHECK:STDERR: var test_f32: f32 = ();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_stringify_type_literals.carbon:[[@LINE+4]]:1: note: type `()` does not implement interface `Core.ImplicitAs(f32)` [MissingImplInMemberAccessInContext]
// CHECK:STDERR: var test_f32: f32 = ();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var test_f32: f32 = ();

// CHECK:STDERR: fail_stringify_type_literals.carbon:[[@LINE+7]]:1: error: cannot implicitly convert expression of type `()` to `u32` [ConversionFailure]
// CHECK:STDERR: var test_u32: u32 = ();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_stringify_type_literals.carbon:[[@LINE+4]]:1: note: type `()` does not implement interface `Core.ImplicitAs(u32)` [MissingImplInMemberAccessInContext]
// CHECK:STDERR: var test_u32: u32 = ();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var test_u32: u32 = ();

// CHECK:STDERR: fail_stringify_type_literals.carbon:[[@LINE+7]]:1: error: cannot implicitly convert expression of type `()` to `bool` [ConversionFailure]
// CHECK:STDERR: var test_bool: bool = ();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_stringify_type_literals.carbon:[[@LINE+4]]:1: note: type `()` does not implement interface `Core.ImplicitAs(bool)` [MissingImplInMemberAccessInContext]
// CHECK:STDERR: var test_bool: bool = ();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var test_bool: bool = ();

// CHECK:STDERR: fail_stringify_type_literals.carbon:[[@LINE+7]]:1: error: cannot implicitly convert expression of type `()` to `char` [ConversionFailure]
// CHECK:STDERR: var test_char: char = ();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_stringify_type_literals.carbon:[[@LINE+4]]:1: note: type `()` does not implement interface `Core.ImplicitAs(char)` [MissingImplInMemberAccessInContext]
// CHECK:STDERR: var test_char: char = ();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var test_char: char = ();

// CHECK:STDERR: fail_stringify_type_literals.carbon:[[@LINE+7]]:1: error: cannot implicitly convert expression of type `()` to `str` [ConversionFailure]
// CHECK:STDERR: var test_str: str = ();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_stringify_type_literals.carbon:[[@LINE+4]]:1: note: type `()` does not implement interface `Core.ImplicitAs(str)` [MissingImplInMemberAccessInContext]
// CHECK:STDERR: var test_str: str = ();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var test_str: str = ();

// CHECK:STDOUT: --- iN.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_8: Core.IntLiteral = int_value 8 [concrete]
// CHECK:STDOUT:   %i8: type = class_type @Int, @Int(%int_8) [concrete]
// CHECK:STDOUT:   %pattern_type.e3f: type = pattern_type %i8 [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.type.6c6: type = fn_type @T.as.DefaultOrUnformed.impl.Op, @T.as.DefaultOrUnformed.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.ae1: %T.as.DefaultOrUnformed.impl.Op.type.6c6 = struct_value () [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.8de: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.349, @T.as.DefaultOrUnformed.impl(%i8) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.e22: %DefaultOrUnformed.type = facet_value %i8, (%DefaultOrUnformed.impl_witness.8de) [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(%int_16) [concrete]
// CHECK:STDOUT:   %pattern_type.2f8: type = pattern_type %i16 [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.e93: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.349, @T.as.DefaultOrUnformed.impl(%i16) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.cc9: %DefaultOrUnformed.type = facet_value %i16, (%DefaultOrUnformed.impl_witness.e93) [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.8d3: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.349, @T.as.DefaultOrUnformed.impl(%i32) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.266: %DefaultOrUnformed.type = facet_value %i32, (%DefaultOrUnformed.impl_witness.8d3) [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.95b: type = pattern_type %i64 [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.3a1: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.349, @T.as.DefaultOrUnformed.impl(%i64) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.161: %DefaultOrUnformed.type = facet_value %i64, (%DefaultOrUnformed.impl_witness.3a1) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.01d: @T.as.DefaultOrUnformed.impl.%T.as.DefaultOrUnformed.impl.Op.type (%T.as.DefaultOrUnformed.impl.Op.type.6c6) = import_ref Core//prelude/parts/default, loc{{\d+_\d+}}, loaded [symbolic = @T.as.DefaultOrUnformed.impl.%T.as.DefaultOrUnformed.impl.Op (constants.%T.as.DefaultOrUnformed.impl.Op.ae1)]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness_table.349 = impl_witness_table (%Core.import_ref.01d), @T.as.DefaultOrUnformed.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %test_i8.patt: %pattern_type.e3f = ref_binding_pattern test_i8 [concrete]
// CHECK:STDOUT:     %test_i8.var_patt: %pattern_type.e3f = var_pattern %test_i8.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %test_i8.var: ref %i8 = var %test_i8.var_patt [concrete]
// CHECK:STDOUT:   %i8: type = type_literal constants.%i8 [concrete = constants.%i8]
// CHECK:STDOUT:   %test_i8: ref %i8 = ref_binding test_i8, %test_i8.var [concrete = %test_i8.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %test_i16.patt: %pattern_type.2f8 = ref_binding_pattern test_i16 [concrete]
// CHECK:STDOUT:     %test_i16.var_patt: %pattern_type.2f8 = var_pattern %test_i16.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %test_i16.var: ref %i16 = var %test_i16.var_patt [concrete]
// CHECK:STDOUT:   %i16: type = type_literal constants.%i16 [concrete = constants.%i16]
// CHECK:STDOUT:   %test_i16: ref %i16 = ref_binding test_i16, %test_i16.var [concrete = %test_i16.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %test_i32.patt: %pattern_type.7ce = ref_binding_pattern test_i32 [concrete]
// CHECK:STDOUT:     %test_i32.var_patt: %pattern_type.7ce = var_pattern %test_i32.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %test_i32.var: ref %i32 = var %test_i32.var_patt [concrete]
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %test_i32: ref %i32 = ref_binding test_i32, %test_i32.var [concrete = %test_i32.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %test_i64.patt: %pattern_type.95b = ref_binding_pattern test_i64 [concrete]
// CHECK:STDOUT:     %test_i64.var_patt: %pattern_type.95b = var_pattern %test_i64.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %test_i64.var: ref %i64 = var %test_i64.var_patt [concrete]
// CHECK:STDOUT:   %i64: type = type_literal constants.%i64 [concrete = constants.%i64]
// CHECK:STDOUT:   %test_i64: ref %i64 = ref_binding test_i64, %test_i64.var [concrete = %test_i64.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc4: %DefaultOrUnformed.type = facet_value constants.%i8, (constants.%DefaultOrUnformed.impl_witness.8de) [concrete = constants.%DefaultOrUnformed.facet.e22]
// CHECK:STDOUT:   %.loc4_16.1: %DefaultOrUnformed.type = converted constants.%i8, %DefaultOrUnformed.facet.loc4 [concrete = constants.%DefaultOrUnformed.facet.e22]
// CHECK:STDOUT:   %as_type.loc4: type = facet_access_type %.loc4_16.1 [concrete = constants.%i8]
// CHECK:STDOUT:   %.loc4_16.2: type = converted %.loc4_16.1, %as_type.loc4 [concrete = constants.%i8]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.call.loc4: init %i8 = call %T.as.DefaultOrUnformed.impl.Op.specific_fn.1()
// CHECK:STDOUT:   assign file.%test_i8.var, %T.as.DefaultOrUnformed.impl.Op.call.loc4
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc5: %DefaultOrUnformed.type = facet_value constants.%i16, (constants.%DefaultOrUnformed.impl_witness.e93) [concrete = constants.%DefaultOrUnformed.facet.cc9]
// CHECK:STDOUT:   %.loc5_18.1: %DefaultOrUnformed.type = converted constants.%i16, %DefaultOrUnformed.facet.loc5 [concrete = constants.%DefaultOrUnformed.facet.cc9]
// CHECK:STDOUT:   %as_type.loc5: type = facet_access_type %.loc5_18.1 [concrete = constants.%i16]
// CHECK:STDOUT:   %.loc5_18.2: type = converted %.loc5_18.1, %as_type.loc5 [concrete = constants.%i16]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.call.loc5: init %i16 = call %T.as.DefaultOrUnformed.impl.Op.specific_fn.2()
// CHECK:STDOUT:   assign file.%test_i16.var, %T.as.DefaultOrUnformed.impl.Op.call.loc5
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc6: %DefaultOrUnformed.type = facet_value constants.%i32, (constants.%DefaultOrUnformed.impl_witness.8d3) [concrete = constants.%DefaultOrUnformed.facet.266]
// CHECK:STDOUT:   %.loc6_18.1: %DefaultOrUnformed.type = converted constants.%i32, %DefaultOrUnformed.facet.loc6 [concrete = constants.%DefaultOrUnformed.facet.266]
// CHECK:STDOUT:   %as_type.loc6: type = facet_access_type %.loc6_18.1 [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc6_18.2: type = converted %.loc6_18.1, %as_type.loc6 [concrete = constants.%i32]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.call.loc6: init %i32 = call %T.as.DefaultOrUnformed.impl.Op.specific_fn.3()
// CHECK:STDOUT:   assign file.%test_i32.var, %T.as.DefaultOrUnformed.impl.Op.call.loc6
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc7: %DefaultOrUnformed.type = facet_value constants.%i64, (constants.%DefaultOrUnformed.impl_witness.3a1) [concrete = constants.%DefaultOrUnformed.facet.161]
// CHECK:STDOUT:   %.loc7_18.1: %DefaultOrUnformed.type = converted constants.%i64, %DefaultOrUnformed.facet.loc7 [concrete = constants.%DefaultOrUnformed.facet.161]
// CHECK:STDOUT:   %as_type.loc7: type = facet_access_type %.loc7_18.1 [concrete = constants.%i64]
// CHECK:STDOUT:   %.loc7_18.2: type = converted %.loc7_18.1, %as_type.loc7 [concrete = constants.%i64]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.call.loc7: init %i64 = call %T.as.DefaultOrUnformed.impl.Op.specific_fn.4()
// CHECK:STDOUT:   assign file.%test_i64.var, %T.as.DefaultOrUnformed.impl.Op.call.loc7
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- uN.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_8: Core.IntLiteral = int_value 8 [concrete]
// CHECK:STDOUT:   %u8: type = class_type @UInt, @UInt(%int_8) [concrete]
// CHECK:STDOUT:   %pattern_type.8f3: type = pattern_type %u8 [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.type.6c6: type = fn_type @T.as.DefaultOrUnformed.impl.Op, @T.as.DefaultOrUnformed.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.ae1: %T.as.DefaultOrUnformed.impl.Op.type.6c6 = struct_value () [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.341: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.349, @T.as.DefaultOrUnformed.impl(%u8) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.56df: %DefaultOrUnformed.type = facet_value %u8, (%DefaultOrUnformed.impl_witness.341) [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %u16: type = class_type @UInt, @UInt(%int_16) [concrete]
// CHECK:STDOUT:   %pattern_type.9db: type = pattern_type %u16 [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.f79: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.349, @T.as.DefaultOrUnformed.impl(%u16) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.ad7: %DefaultOrUnformed.type = facet_value %u16, (%DefaultOrUnformed.impl_witness.f79) [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %u32: type = class_type @UInt, @UInt(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.4a9: type = pattern_type %u32 [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.486: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.349, @T.as.DefaultOrUnformed.impl(%u32) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.5f8: %DefaultOrUnformed.type = facet_value %u32, (%DefaultOrUnformed.impl_witness.486) [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %u64: type = class_type @UInt, @UInt(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.157: type = pattern_type %u64 [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.8ea: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.349, @T.as.DefaultOrUnformed.impl(%u64) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.56d3: %DefaultOrUnformed.type = facet_value %u64, (%DefaultOrUnformed.impl_witness.8ea) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.01d: @T.as.DefaultOrUnformed.impl.%T.as.DefaultOrUnformed.impl.Op.type (%T.as.DefaultOrUnformed.impl.Op.type.6c6) = import_ref Core//prelude/parts/default, loc{{\d+_\d+}}, loaded [symbolic = @T.as.DefaultOrUnformed.impl.%T.as.DefaultOrUnformed.impl.Op (constants.%T.as.DefaultOrUnformed.impl.Op.ae1)]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness_table.349 = impl_witness_table (%Core.import_ref.01d), @T.as.DefaultOrUnformed.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %test_u8.patt: %pattern_type.8f3 = ref_binding_pattern test_u8 [concrete]
// CHECK:STDOUT:     %test_u8.var_patt: %pattern_type.8f3 = var_pattern %test_u8.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %test_u8.var: ref %u8 = var %test_u8.var_patt [concrete]
// CHECK:STDOUT:   %u8: type = type_literal constants.%u8 [concrete = constants.%u8]
// CHECK:STDOUT:   %test_u8: ref %u8 = ref_binding test_u8, %test_u8.var [concrete = %test_u8.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %test_u16.patt: %pattern_type.9db = ref_binding_pattern test_u16 [concrete]
// CHECK:STDOUT:     %test_u16.var_patt: %pattern_type.9db = var_pattern %test_u16.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %test_u16.var: ref %u16 = var %test_u16.var_patt [concrete]
// CHECK:STDOUT:   %u16: type = type_literal constants.%u16 [concrete = constants.%u16]
// CHECK:STDOUT:   %test_u16: ref %u16 = ref_binding test_u16, %test_u16.var [concrete = %test_u16.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %test_u32.patt: %pattern_type.4a9 = ref_binding_pattern test_u32 [concrete]
// CHECK:STDOUT:     %test_u32.var_patt: %pattern_type.4a9 = var_pattern %test_u32.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %test_u32.var: ref %u32 = var %test_u32.var_patt [concrete]
// CHECK:STDOUT:   %u32: type = type_literal constants.%u32 [concrete = constants.%u32]
// CHECK:STDOUT:   %test_u32: ref %u32 = ref_binding test_u32, %test_u32.var [concrete = %test_u32.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %test_u64.patt: %pattern_type.157 = ref_binding_pattern test_u64 [concrete]
// CHECK:STDOUT:     %test_u64.var_patt: %pattern_type.157 = var_pattern %test_u64.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %test_u64.var: ref %u64 = var %test_u64.var_patt [concrete]
// CHECK:STDOUT:   %u64: type = type_literal constants.%u64 [concrete = constants.%u64]
// CHECK:STDOUT:   %test_u64: ref %u64 = ref_binding test_u64, %test_u64.var [concrete = %test_u64.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc4: %DefaultOrUnformed.type = facet_value constants.%u8, (constants.%DefaultOrUnformed.impl_witness.341) [concrete = constants.%DefaultOrUnformed.facet.56df]
// CHECK:STDOUT:   %.loc4_16.1: %DefaultOrUnformed.type = converted constants.%u8, %DefaultOrUnformed.facet.loc4 [concrete = constants.%DefaultOrUnformed.facet.56df]
// CHECK:STDOUT:   %as_type.loc4: type = facet_access_type %.loc4_16.1 [concrete = constants.%u8]
// CHECK:STDOUT:   %.loc4_16.2: type = converted %.loc4_16.1, %as_type.loc4 [concrete = constants.%u8]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.call.loc4: init %u8 = call %T.as.DefaultOrUnformed.impl.Op.specific_fn.1()
// CHECK:STDOUT:   assign file.%test_u8.var, %T.as.DefaultOrUnformed.impl.Op.call.loc4
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc5: %DefaultOrUnformed.type = facet_value constants.%u16, (constants.%DefaultOrUnformed.impl_witness.f79) [concrete = constants.%DefaultOrUnformed.facet.ad7]
// CHECK:STDOUT:   %.loc5_18.1: %DefaultOrUnformed.type = converted constants.%u16, %DefaultOrUnformed.facet.loc5 [concrete = constants.%DefaultOrUnformed.facet.ad7]
// CHECK:STDOUT:   %as_type.loc5: type = facet_access_type %.loc5_18.1 [concrete = constants.%u16]
// CHECK:STDOUT:   %.loc5_18.2: type = converted %.loc5_18.1, %as_type.loc5 [concrete = constants.%u16]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.call.loc5: init %u16 = call %T.as.DefaultOrUnformed.impl.Op.specific_fn.2()
// CHECK:STDOUT:   assign file.%test_u16.var, %T.as.DefaultOrUnformed.impl.Op.call.loc5
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc6: %DefaultOrUnformed.type = facet_value constants.%u32, (constants.%DefaultOrUnformed.impl_witness.486) [concrete = constants.%DefaultOrUnformed.facet.5f8]
// CHECK:STDOUT:   %.loc6_18.1: %DefaultOrUnformed.type = converted constants.%u32, %DefaultOrUnformed.facet.loc6 [concrete = constants.%DefaultOrUnformed.facet.5f8]
// CHECK:STDOUT:   %as_type.loc6: type = facet_access_type %.loc6_18.1 [concrete = constants.%u32]
// CHECK:STDOUT:   %.loc6_18.2: type = converted %.loc6_18.1, %as_type.loc6 [concrete = constants.%u32]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.call.loc6: init %u32 = call %T.as.DefaultOrUnformed.impl.Op.specific_fn.3()
// CHECK:STDOUT:   assign file.%test_u32.var, %T.as.DefaultOrUnformed.impl.Op.call.loc6
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc7: %DefaultOrUnformed.type = facet_value constants.%u64, (constants.%DefaultOrUnformed.impl_witness.8ea) [concrete = constants.%DefaultOrUnformed.facet.56d3]
// CHECK:STDOUT:   %.loc7_18.1: %DefaultOrUnformed.type = converted constants.%u64, %DefaultOrUnformed.facet.loc7 [concrete = constants.%DefaultOrUnformed.facet.56d3]
// CHECK:STDOUT:   %as_type.loc7: type = facet_access_type %.loc7_18.1 [concrete = constants.%u64]
// CHECK:STDOUT:   %.loc7_18.2: type = converted %.loc7_18.1, %as_type.loc7 [concrete = constants.%u64]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.call.loc7: init %u64 = call %T.as.DefaultOrUnformed.impl.Op.specific_fn.4()
// CHECK:STDOUT:   assign file.%test_u64.var, %T.as.DefaultOrUnformed.impl.Op.call.loc7
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fN.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %f16.a6a: type = class_type @Float, @Float(%int_16) [concrete]
// CHECK:STDOUT:   %pattern_type.cd1: type = pattern_type %f16.a6a [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.type.6c6: type = fn_type @T.as.DefaultOrUnformed.impl.Op, @T.as.DefaultOrUnformed.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.ae1: %T.as.DefaultOrUnformed.impl.Op.type.6c6 = struct_value () [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.a55: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.349, @T.as.DefaultOrUnformed.impl(%f16.a6a) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.671: %DefaultOrUnformed.type = facet_value %f16.a6a, (%DefaultOrUnformed.impl_witness.a55) [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %f32.97e: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.201: type = pattern_type %f32.97e [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.567: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.349, @T.as.DefaultOrUnformed.impl(%f32.97e) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.60c: %DefaultOrUnformed.type = facet_value %f32.97e, (%DefaultOrUnformed.impl_witness.567) [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %f64.d77: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.0ae: type = pattern_type %f64.d77 [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.461: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.349, @T.as.DefaultOrUnformed.impl(%f64.d77) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.625: %DefaultOrUnformed.type = facet_value %f64.d77, (%DefaultOrUnformed.impl_witness.461) [concrete]
// CHECK:STDOUT:   %int_128: Core.IntLiteral = int_value 128 [concrete]
// CHECK:STDOUT:   %f128.b8c: type = class_type @Float, @Float(%int_128) [concrete]
// CHECK:STDOUT:   %pattern_type.22c: type = pattern_type %f128.b8c [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.39e: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.349, @T.as.DefaultOrUnformed.impl(%f128.b8c) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.560: %DefaultOrUnformed.type = facet_value %f128.b8c, (%DefaultOrUnformed.impl_witness.39e) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.01d: @T.as.DefaultOrUnformed.impl.%T.as.DefaultOrUnformed.impl.Op.type (%T.as.DefaultOrUnformed.impl.Op.type.6c6) = import_ref Core//prelude/parts/default, loc{{\d+_\d+}}, loaded [symbolic = @T.as.DefaultOrUnformed.impl.%T.as.DefaultOrUnformed.impl.Op (constants.%T.as.DefaultOrUnformed.impl.Op.ae1)]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness_table.349 = impl_witness_table (%Core.import_ref.01d), @T.as.DefaultOrUnformed.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %test_f16.patt: %pattern_type.cd1 = ref_binding_pattern test_f16 [concrete]
// CHECK:STDOUT:     %test_f16.var_patt: %pattern_type.cd1 = var_pattern %test_f16.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %test_f16.var: ref %f16.a6a = var %test_f16.var_patt [concrete]
// CHECK:STDOUT:   %f16: type = type_literal constants.%f16.a6a [concrete = constants.%f16.a6a]
// CHECK:STDOUT:   %test_f16: ref %f16.a6a = ref_binding test_f16, %test_f16.var [concrete = %test_f16.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %test_f32.patt: %pattern_type.201 = ref_binding_pattern test_f32 [concrete]
// CHECK:STDOUT:     %test_f32.var_patt: %pattern_type.201 = var_pattern %test_f32.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %test_f32.var: ref %f32.97e = var %test_f32.var_patt [concrete]
// CHECK:STDOUT:   %f32: type = type_literal constants.%f32.97e [concrete = constants.%f32.97e]
// CHECK:STDOUT:   %test_f32: ref %f32.97e = ref_binding test_f32, %test_f32.var [concrete = %test_f32.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %test_f64.patt: %pattern_type.0ae = ref_binding_pattern test_f64 [concrete]
// CHECK:STDOUT:     %test_f64.var_patt: %pattern_type.0ae = var_pattern %test_f64.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %test_f64.var: ref %f64.d77 = var %test_f64.var_patt [concrete]
// CHECK:STDOUT:   %f64: type = type_literal constants.%f64.d77 [concrete = constants.%f64.d77]
// CHECK:STDOUT:   %test_f64: ref %f64.d77 = ref_binding test_f64, %test_f64.var [concrete = %test_f64.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %test_f128.patt: %pattern_type.22c = ref_binding_pattern test_f128 [concrete]
// CHECK:STDOUT:     %test_f128.var_patt: %pattern_type.22c = var_pattern %test_f128.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %test_f128.var: ref %f128.b8c = var %test_f128.var_patt [concrete]
// CHECK:STDOUT:   %f128: type = type_literal constants.%f128.b8c [concrete = constants.%f128.b8c]
// CHECK:STDOUT:   %test_f128: ref %f128.b8c = ref_binding test_f128, %test_f128.var [concrete = %test_f128.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc4: %DefaultOrUnformed.type = facet_value constants.%f16.a6a, (constants.%DefaultOrUnformed.impl_witness.a55) [concrete = constants.%DefaultOrUnformed.facet.671]
// CHECK:STDOUT:   %.loc4_18.1: %DefaultOrUnformed.type = converted constants.%f16.a6a, %DefaultOrUnformed.facet.loc4 [concrete = constants.%DefaultOrUnformed.facet.671]
// CHECK:STDOUT:   %as_type.loc4: type = facet_access_type %.loc4_18.1 [concrete = constants.%f16.a6a]
// CHECK:STDOUT:   %.loc4_18.2: type = converted %.loc4_18.1, %as_type.loc4 [concrete = constants.%f16.a6a]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.call.loc4: init %f16.a6a = call %T.as.DefaultOrUnformed.impl.Op.specific_fn.1()
// CHECK:STDOUT:   assign file.%test_f16.var, %T.as.DefaultOrUnformed.impl.Op.call.loc4
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc5: %DefaultOrUnformed.type = facet_value constants.%f32.97e, (constants.%DefaultOrUnformed.impl_witness.567) [concrete = constants.%DefaultOrUnformed.facet.60c]
// CHECK:STDOUT:   %.loc5_18.1: %DefaultOrUnformed.type = converted constants.%f32.97e, %DefaultOrUnformed.facet.loc5 [concrete = constants.%DefaultOrUnformed.facet.60c]
// CHECK:STDOUT:   %as_type.loc5: type = facet_access_type %.loc5_18.1 [concrete = constants.%f32.97e]
// CHECK:STDOUT:   %.loc5_18.2: type = converted %.loc5_18.1, %as_type.loc5 [concrete = constants.%f32.97e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.call.loc5: init %f32.97e = call %T.as.DefaultOrUnformed.impl.Op.specific_fn.2()
// CHECK:STDOUT:   assign file.%test_f32.var, %T.as.DefaultOrUnformed.impl.Op.call.loc5
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc6: %DefaultOrUnformed.type = facet_value constants.%f64.d77, (constants.%DefaultOrUnformed.impl_witness.461) [concrete = constants.%DefaultOrUnformed.facet.625]
// CHECK:STDOUT:   %.loc6_18.1: %DefaultOrUnformed.type = converted constants.%f64.d77, %DefaultOrUnformed.facet.loc6 [concrete = constants.%DefaultOrUnformed.facet.625]
// CHECK:STDOUT:   %as_type.loc6: type = facet_access_type %.loc6_18.1 [concrete = constants.%f64.d77]
// CHECK:STDOUT:   %.loc6_18.2: type = converted %.loc6_18.1, %as_type.loc6 [concrete = constants.%f64.d77]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.call.loc6: init %f64.d77 = call %T.as.DefaultOrUnformed.impl.Op.specific_fn.3()
// CHECK:STDOUT:   assign file.%test_f64.var, %T.as.DefaultOrUnformed.impl.Op.call.loc6
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc7: %DefaultOrUnformed.type = facet_value constants.%f128.b8c, (constants.%DefaultOrUnformed.impl_witness.39e) [concrete = constants.%DefaultOrUnformed.facet.560]
// CHECK:STDOUT:   %.loc7_20.1: %DefaultOrUnformed.type = converted constants.%f128.b8c, %DefaultOrUnformed.facet.loc7 [concrete = constants.%DefaultOrUnformed.facet.560]
// CHECK:STDOUT:   %as_type.loc7: type = facet_access_type %.loc7_20.1 [concrete = constants.%f128.b8c]
// CHECK:STDOUT:   %.loc7_20.2: type = converted %.loc7_20.1, %as_type.loc7 [concrete = constants.%f128.b8c]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.call.loc7: init %f128.b8c = call %T.as.DefaultOrUnformed.impl.Op.specific_fn.4()
// CHECK:STDOUT:   assign file.%test_f128.var, %T.as.DefaultOrUnformed.impl.Op.call.loc7
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- char.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %char: type = class_type @Char [concrete]
// CHECK:STDOUT:   %pattern_type.b09: type = pattern_type %char [concrete]
// CHECK:STDOUT:   %.b9b: Core.CharLiteral = char_value U+0063 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.d99: type = facet_type <@ImplicitAs, @ImplicitAs(%char)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.d99 = facet_value Core.CharLiteral, (%ImplicitAs.impl_witness) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.95b: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%char, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.734: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.95b, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.CharLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.CharLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.CharLiteral.as.ImplicitAs.impl.Convert: %Core.CharLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.CharLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %.b9b, %Core.CharLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_99: %char = int_value 99 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.b73: %Core.CharLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/parts/char, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.CharLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%Core.import_ref.b73), @Core.CharLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %test_char.patt: %pattern_type.b09 = value_binding_pattern test_char [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %char: type = type_literal constants.%char [concrete = constants.%char]
// CHECK:STDOUT:   %impl.elem0: %.734 = impl_witness_access constants.%ImplicitAs.impl_witness, element0 [concrete = constants.%Core.CharLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method @__global_init.%.loc5, %impl.elem0 [concrete = constants.%Core.CharLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.CharLiteral.as.ImplicitAs.impl.Convert.call: init %char = call %bound_method(@__global_init.%.loc5) [concrete = constants.%int_99]
// CHECK:STDOUT:   %.loc5_23.1: %char = value_of_initializer %Core.CharLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_99]
// CHECK:STDOUT:   %.loc5_23.2: %char = converted @__global_init.%.loc5, %.loc5_23.1 [concrete = constants.%int_99]
// CHECK:STDOUT:   %test_char: %char = value_binding test_char, %.loc5_23.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc5: Core.CharLiteral = char_value U+0063 [concrete = constants.%.b9b]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- string.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %str.ee0: type = class_type @String [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %u64: type = class_type @UInt, @UInt(%int_64) [concrete]
// CHECK:STDOUT:   %int_8: Core.IntLiteral = int_value 8 [concrete]
// CHECK:STDOUT:   %u8: type = class_type @UInt, @UInt(%int_8) [concrete]
// CHECK:STDOUT:   %ptr.3e8: type = ptr_type %u8 [concrete]
// CHECK:STDOUT:   %pattern_type.461: type = pattern_type %str.ee0 [concrete]
// CHECK:STDOUT:   %str.c1c: %ptr.3e8 = string_literal "Test" [concrete]
// CHECK:STDOUT:   %int_4: %u64 = int_value 4 [concrete]
// CHECK:STDOUT:   %String.val: %str.ee0 = struct_value (%str.c1c, %int_4) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %test_str.patt: %pattern_type.461 = value_binding_pattern test_str [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %str: type = type_literal constants.%str.ee0 [concrete = constants.%str.ee0]
// CHECK:STDOUT:   %test_str: %str.ee0 = value_binding test_str, @__global_init.%String.val
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %str: %ptr.3e8 = string_literal "Test" [concrete = constants.%str.c1c]
// CHECK:STDOUT:   %int_4: %u64 = int_value 4 [concrete = constants.%int_4]
// CHECK:STDOUT:   %String.val: %str.ee0 = struct_value (%str, %int_4) [concrete = constants.%String.val]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.b47: <witness> = impl_witness imports.%Copy.impl_witness_table.b1c [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value type, (%Copy.impl_witness.b47) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.a4f: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.070: type = fn_type_with_self_type %Copy.WithSelf.Op.type.a4f, %Copy.facet [concrete]
// CHECK:STDOUT:   %type.as.Copy.impl.Op.type: type = fn_type @type.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %type.as.Copy.impl.Op: %type.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %type.as.Copy.impl.Op.bound: <bound method> = bound_method %i32, %type.as.Copy.impl.Op [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.1a7: %type.as.Copy.impl.Op.type = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [concrete = constants.%type.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.b1c = impl_witness_table (%Core.import_ref.1a7), @type.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %test_type.patt: %pattern_type.98f = ref_binding_pattern test_type [concrete]
// CHECK:STDOUT:     %test_type.var_patt: %pattern_type.98f = var_pattern %test_type.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %test_type.var: ref type = var %test_type.var_patt [concrete]
// CHECK:STDOUT:   %.loc5: type = type_literal type [concrete = type]
// CHECK:STDOUT:   %test_type: ref type = ref_binding test_type, %test_type.var [concrete = %test_type.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0: %.070 = impl_witness_access constants.%Copy.impl_witness.b47, element0 [concrete = constants.%type.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %i32, %impl.elem0 [concrete = constants.%type.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %type.as.Copy.impl.Op.call: init type = call %bound_method(%i32) [concrete = constants.%i32]
// CHECK:STDOUT:   assign file.%test_type.var, %type.as.Copy.impl.Op.call
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:

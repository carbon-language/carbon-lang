// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/basics/type_literals.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/basics/type_literals.carbon

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
var test_f64: f64;
//@dump-sem-ir-end

// --- fail_fN_todo_unsupported.carbon
library "[[@TEST_NAME]]";

// TODO: Some or all of these should eventually work.
// CHECK:STDERR: fail_fN_todo_unsupported.carbon:[[@LINE+4]]:15: error: semantics TODO: `Currently only f64 is allowed` [SemanticsTodo]
// CHECK:STDERR: var test_f16: f16;
// CHECK:STDERR:               ^~~
// CHECK:STDERR:
var test_f16: f16;
var test_f32: f32;
var test_f128: f128;

// --- string.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
let test_str: String = "Test";
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
// CHECK:STDERR: fail_iN_bad_width.carbon:[[@LINE+7]]:15: error: bit width of integer type literal must be a multiple of 8; use `Core.Int(15)` instead [IntWidthNotMultipleOf8]
// CHECK:STDERR: var test_i15: i15;
// CHECK:STDERR:               ^~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:13:9: error: integer type width of 1000000000 is greater than the maximum supported width of 8388608 [IntWidthTooLarge]
// CHECK:STDERR:   adapt MakeInt(N);
// CHECK:STDERR:         ^~~~~~~~~~
var test_i15: i15;
// CHECK:STDERR: fail_iN_bad_width.carbon:[[@LINE+4]]:23: note: in `i1000000000` used here [ResolvingSpecificHere]
// CHECK:STDERR: var test_i1000000000: i1000000000;
// CHECK:STDERR:                       ^~~~~~~~~~~
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
// CHECK:STDERR: fail_uN_bad_width.carbon:[[@LINE+7]]:15: error: bit width of integer type literal must be a multiple of 8; use `Core.UInt(15)` instead [IntWidthNotMultipleOf8]
// CHECK:STDERR: var test_u15: u15;
// CHECK:STDERR:               ^~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:14:9: error: integer type width of 1000000000 is greater than the maximum supported width of 8388608 [IntWidthTooLarge]
// CHECK:STDERR:   adapt MakeUInt(N);
// CHECK:STDERR:         ^~~~~~~~~~~
var test_u15: u15;
// CHECK:STDERR: fail_uN_bad_width.carbon:[[@LINE+4]]:23: note: in `u1000000000` used here [ResolvingSpecificHere]
// CHECK:STDERR: var test_u1000000000: u1000000000;
// CHECK:STDERR:                       ^~~~~~~~~~~
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
// CHECK:STDERR: fail_fN_bad_width.carbon:[[@LINE+4]]:14: error: semantics TODO: `Currently only f64 is allowed` [SemanticsTodo]
// CHECK:STDERR: var test_f1: f1;
// CHECK:STDERR:              ^~
// CHECK:STDERR:
var test_f1: f1;
var test_f15: f15;
var test_f100: f100;
var test_f1000000000: f1000000000;
var test_f1000000000000: f1000000000000;

// --- fail_non_type_as_type.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_non_type_as_type.carbon:[[@LINE+7]]:1: error: cannot implicitly convert non-type value of type `Core.IntLiteral` to `type` [ConversionFailureNonTypeToFacet]
// CHECK:STDERR: var x: type = 42;
// CHECK:STDERR: ^~~~~~~~~~~
// CHECK:STDERR: fail_non_type_as_type.carbon:[[@LINE+4]]:1: note: type `Core.IntLiteral` does not implement interface `Core.ImplicitAs(type)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: var x: type = 42;
// CHECK:STDERR: ^~~~~~~~~~~
// CHECK:STDERR:
var x: type = 42;

// CHECK:STDOUT: --- iN.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_8: Core.IntLiteral = int_value 8 [concrete]
// CHECK:STDOUT:   %i8: type = class_type @Int, @Int(%int_8) [concrete]
// CHECK:STDOUT:   %pattern_type.e3f: type = pattern_type %i8 [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(%int_16) [concrete]
// CHECK:STDOUT:   %pattern_type.2f8: type = pattern_type %i16 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.95b: type = pattern_type %i64 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %test_i8.patt: %pattern_type.e3f = binding_pattern test_i8 [concrete]
// CHECK:STDOUT:     %test_i8.var_patt: %pattern_type.e3f = var_pattern %test_i8.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %test_i8.var: ref %i8 = var %test_i8.var_patt [concrete]
// CHECK:STDOUT:   %.loc4: type = splice_block %i8 [concrete = constants.%i8] {
// CHECK:STDOUT:     %int_8: Core.IntLiteral = int_value 8 [concrete = constants.%int_8]
// CHECK:STDOUT:     %i8: type = class_type @Int, @Int(constants.%int_8) [concrete = constants.%i8]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %test_i8: ref %i8 = bind_name test_i8, %test_i8.var [concrete = %test_i8.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %test_i16.patt: %pattern_type.2f8 = binding_pattern test_i16 [concrete]
// CHECK:STDOUT:     %test_i16.var_patt: %pattern_type.2f8 = var_pattern %test_i16.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %test_i16.var: ref %i16 = var %test_i16.var_patt [concrete]
// CHECK:STDOUT:   %.loc5: type = splice_block %i16 [concrete = constants.%i16] {
// CHECK:STDOUT:     %int_16: Core.IntLiteral = int_value 16 [concrete = constants.%int_16]
// CHECK:STDOUT:     %i16: type = class_type @Int, @Int(constants.%int_16) [concrete = constants.%i16]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %test_i16: ref %i16 = bind_name test_i16, %test_i16.var [concrete = %test_i16.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %test_i32.patt: %pattern_type.7ce = binding_pattern test_i32 [concrete]
// CHECK:STDOUT:     %test_i32.var_patt: %pattern_type.7ce = var_pattern %test_i32.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %test_i32.var: ref %i32 = var %test_i32.var_patt [concrete]
// CHECK:STDOUT:   %.loc6: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %test_i32: ref %i32 = bind_name test_i32, %test_i32.var [concrete = %test_i32.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %test_i64.patt: %pattern_type.95b = binding_pattern test_i64 [concrete]
// CHECK:STDOUT:     %test_i64.var_patt: %pattern_type.95b = var_pattern %test_i64.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %test_i64.var: ref %i64 = var %test_i64.var_patt [concrete]
// CHECK:STDOUT:   %.loc7: type = splice_block %i64 [concrete = constants.%i64] {
// CHECK:STDOUT:     %int_64: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %i64: type = class_type @Int, @Int(constants.%int_64) [concrete = constants.%i64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %test_i64: ref %i64 = bind_name test_i64, %test_i64.var [concrete = %test_i64.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- uN.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_8: Core.IntLiteral = int_value 8 [concrete]
// CHECK:STDOUT:   %u8: type = class_type @UInt, @UInt(%int_8) [concrete]
// CHECK:STDOUT:   %pattern_type.8f3: type = pattern_type %u8 [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %u16: type = class_type @UInt, @UInt(%int_16) [concrete]
// CHECK:STDOUT:   %pattern_type.9db: type = pattern_type %u16 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %u32: type = class_type @UInt, @UInt(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.4a9: type = pattern_type %u32 [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %u64: type = class_type @UInt, @UInt(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.157: type = pattern_type %u64 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %test_u8.patt: %pattern_type.8f3 = binding_pattern test_u8 [concrete]
// CHECK:STDOUT:     %test_u8.var_patt: %pattern_type.8f3 = var_pattern %test_u8.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %test_u8.var: ref %u8 = var %test_u8.var_patt [concrete]
// CHECK:STDOUT:   %.loc4: type = splice_block %u8 [concrete = constants.%u8] {
// CHECK:STDOUT:     %int_8: Core.IntLiteral = int_value 8 [concrete = constants.%int_8]
// CHECK:STDOUT:     %u8: type = class_type @UInt, @UInt(constants.%int_8) [concrete = constants.%u8]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %test_u8: ref %u8 = bind_name test_u8, %test_u8.var [concrete = %test_u8.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %test_u16.patt: %pattern_type.9db = binding_pattern test_u16 [concrete]
// CHECK:STDOUT:     %test_u16.var_patt: %pattern_type.9db = var_pattern %test_u16.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %test_u16.var: ref %u16 = var %test_u16.var_patt [concrete]
// CHECK:STDOUT:   %.loc5: type = splice_block %u16 [concrete = constants.%u16] {
// CHECK:STDOUT:     %int_16: Core.IntLiteral = int_value 16 [concrete = constants.%int_16]
// CHECK:STDOUT:     %u16: type = class_type @UInt, @UInt(constants.%int_16) [concrete = constants.%u16]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %test_u16: ref %u16 = bind_name test_u16, %test_u16.var [concrete = %test_u16.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %test_u32.patt: %pattern_type.4a9 = binding_pattern test_u32 [concrete]
// CHECK:STDOUT:     %test_u32.var_patt: %pattern_type.4a9 = var_pattern %test_u32.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %test_u32.var: ref %u32 = var %test_u32.var_patt [concrete]
// CHECK:STDOUT:   %.loc6: type = splice_block %u32 [concrete = constants.%u32] {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %u32: type = class_type @UInt, @UInt(constants.%int_32) [concrete = constants.%u32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %test_u32: ref %u32 = bind_name test_u32, %test_u32.var [concrete = %test_u32.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %test_u64.patt: %pattern_type.157 = binding_pattern test_u64 [concrete]
// CHECK:STDOUT:     %test_u64.var_patt: %pattern_type.157 = var_pattern %test_u64.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %test_u64.var: ref %u64 = var %test_u64.var_patt [concrete]
// CHECK:STDOUT:   %.loc7: type = splice_block %u64 [concrete = constants.%u64] {
// CHECK:STDOUT:     %int_64: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %u64: type = class_type @UInt, @UInt(constants.%int_64) [concrete = constants.%u64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %test_u64: ref %u64 = bind_name test_u64, %test_u64.var [concrete = %test_u64.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fN.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %Float.type: type = fn_type @Float [concrete]
// CHECK:STDOUT:   %Float: %Float.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.3de: type = pattern_type f64 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %test_f64.patt: %pattern_type.3de = binding_pattern test_f64 [concrete]
// CHECK:STDOUT:     %test_f64.var_patt: %pattern_type.3de = var_pattern %test_f64.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %test_f64.var: ref f64 = var %test_f64.var_patt [concrete]
// CHECK:STDOUT:   %.loc4_15.1: type = splice_block %.loc4_15.3 [concrete = f64] {
// CHECK:STDOUT:     %int_64: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %float.make_type: init type = call constants.%Float(%int_64) [concrete = f64]
// CHECK:STDOUT:     %.loc4_15.2: type = value_of_initializer %float.make_type [concrete = f64]
// CHECK:STDOUT:     %.loc4_15.3: type = converted %float.make_type, %.loc4_15.2 [concrete = f64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %test_f64: ref f64 = bind_name test_f64, %test_f64.var [concrete = %test_f64.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- string.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %pattern_type: type = pattern_type String [concrete]
// CHECK:STDOUT:   %str: String = string_literal "Test" [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %test_str.patt: %pattern_type = binding_pattern test_str [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %test_str: String = bind_name test_str, @__global_init.%str
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %str: String = string_literal "Test" [concrete = constants.%str]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %test_type.patt: %pattern_type.98f = binding_pattern test_type [concrete]
// CHECK:STDOUT:     %test_type.var_patt: %pattern_type.98f = var_pattern %test_type.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %test_type.var: ref type = var %test_type.var_patt [concrete]
// CHECK:STDOUT:   %test_type: ref type = bind_name test_type, %test_type.var [concrete = %test_type.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   assign file.%test_type.var, %i32
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:

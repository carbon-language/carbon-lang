// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/float/convert_checked.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/float/convert_checked.carbon

// --- literal.carbon

library "[[@TEST_NAME]]";

fn FloatLiteral() -> type = "float_literal.make_type";

fn FloatLiteralToFloatLiteral(a: FloatLiteral()) -> FloatLiteral() =
    "float.convert_checked";

// --- identity_literal.carbon

library "[[@TEST_NAME]]";
import library "literal";

//@dump-sem-ir-begin
let f: FloatLiteral() = FloatLiteralToFloatLiteral(1.0);
//@dump-sem-ir-end

// --- f64.carbon

library "[[@TEST_NAME]]";
export import library "literal";

fn Float64ToFloat64(a: f64) -> f64 = "float.convert_checked";
fn Float64ToFloatLiteral(a: f64) -> FloatLiteral() = "float.convert_checked";
fn FloatLiteralToFloat64(a: FloatLiteral()) -> f64 = "float.convert_checked";

// --- f32.carbon

library "[[@TEST_NAME]]";
export import library "literal";

fn Float32ToFloat32(a: f32) -> f32 = "float.convert_checked";
fn Float32ToFloatLiteral(a: f32) -> FloatLiteral() = "float.convert_checked";
fn FloatLiteralToFloat32(a: FloatLiteral()) -> f32 = "float.convert_checked";

fn Float32ToFloat64(a: f32) -> f64 = "float.convert_checked";
fn Float64ToFloat32(a: f64) -> f32 = "float.convert_checked";

// --- literal_f64.carbon

library "[[@TEST_NAME]]";
import library "f64";

//@dump-sem-ir-begin
let a: f64 = FloatLiteralToFloat64(0.0);
let b: f64 = FloatLiteralToFloat64(1.0);
let c: f64 = FloatLiteralToFloat64(1.0e308);
//@dump-sem-ir-end

// --- literal_f32.carbon

library "[[@TEST_NAME]]";
import library "f32";

//@dump-sem-ir-begin
let a: f32 = FloatLiteralToFloat32(0.0);
let b: f32 = FloatLiteralToFloat32(1.0);
let c: f32 = FloatLiteralToFloat32(1.0e38);
//@dump-sem-ir-end

// --- identity_f64.carbon

library "[[@TEST_NAME]]";
import library "f64";

//@dump-sem-ir-begin
let a: f64 = Float64ToFloat64(0.0);
let b: f64 = Float64ToFloat64(1.0);
let c: f64 = Float64ToFloat64(1.0e308);
//@dump-sem-ir-end

// --- identity_f32.carbon

library "[[@TEST_NAME]]";
import library "f32";

//@dump-sem-ir-begin
let a: f32 = Float32ToFloat32(0.0);
let b: f32 = Float32ToFloat32(1.0);
let c: f32 = Float32ToFloat32(1.0e38);
//@dump-sem-ir-end

// --- truncate.carbon

library "[[@TEST_NAME]]";
import library "f32";

//@dump-sem-ir-begin
let a: f32 = Float64ToFloat32(1.0);
//@dump-sem-ir-end

// --- fail_truncate_overflow.carbon

library "[[@TEST_NAME]]";
import library "f32";
import library "f64";

//@dump-sem-ir-begin
// CHECK:STDERR: fail_truncate_overflow.carbon:[[@LINE+4]]:14: error: value 9.9999999999999994E+38 too large for floating-point type `f32` [FloatTooLargeForType]
// CHECK:STDERR: let a: f32 = Float64ToFloat32(1.0e39);
// CHECK:STDERR:              ^~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let a: f32 = Float64ToFloat32(1.0e39);
// CHECK:STDERR: fail_truncate_overflow.carbon:[[@LINE+4]]:14: error: value 10*10^38 too large for floating-point type `f32` [FloatLiteralTooLargeForType]
// CHECK:STDERR: let b: f32 = FloatLiteralToFloat32(1.0e39);
// CHECK:STDERR:              ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let b: f32 = FloatLiteralToFloat32(1.0e39);
// CHECK:STDERR: fail_truncate_overflow.carbon:[[@LINE+4]]:14: error: value 10*10^308 too large for floating-point type `f64` [FloatLiteralTooLargeForType]
// CHECK:STDERR: let c: f64 = FloatLiteralToFloat64(1.0e309);
// CHECK:STDERR:              ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let c: f64 = FloatLiteralToFloat64(1.0e309);
//@dump-sem-ir-end

// --- extend.carbon

library "[[@TEST_NAME]]";
import library "f32";

//@dump-sem-ir-begin
let a: f64 = Float32ToFloat64(1.0);
let b: f64 = Float32ToFloat64(1.0e30);
//@dump-sem-ir-end

// --- fail_not_constant.carbon

library "[[@TEST_NAME]]";
import library "f64";

//@dump-sem-ir-begin
let not_constant_64: f64 = 0.0;
// CHECK:STDERR: fail_not_constant.carbon:[[@LINE+8]]:33: error: non-constant call to compile-time-only function [NonConstantCallToCompTimeOnlyFunction]
// CHECK:STDERR: let convert_not_constant: f64 = Float64ToFloat64(not_constant_64);
// CHECK:STDERR:                                 ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_not_constant.carbon:[[@LINE-7]]:1: in import [InImport]
// CHECK:STDERR: f64.carbon:36:1: note: compile-time-only function declared here [CompTimeOnlyFunctionHere]
// CHECK:STDERR: fn Float64ToFloat64(a: f64) -> f64 = "float.convert_checked";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let convert_not_constant: f64 = Float64ToFloat64(not_constant_64);
//@dump-sem-ir-end

// CHECK:STDOUT: --- identity_literal.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %FloatLiteral.type: type = fn_type @FloatLiteral [concrete]
// CHECK:STDOUT:   %FloatLiteral: %FloatLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.dab: type = pattern_type Core.FloatLiteral [concrete]
// CHECK:STDOUT:   %FloatLiteralToFloatLiteral.type: type = fn_type @FloatLiteralToFloatLiteral [concrete]
// CHECK:STDOUT:   %FloatLiteralToFloatLiteral: %FloatLiteralToFloatLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.FloatLiteral: %FloatLiteral.type = import_ref Main//literal, FloatLiteral, loaded [concrete = constants.%FloatLiteral]
// CHECK:STDOUT:   %Main.FloatLiteralToFloatLiteral: %FloatLiteralToFloatLiteral.type = import_ref Main//literal, FloatLiteralToFloatLiteral, loaded [concrete = constants.%FloatLiteralToFloatLiteral]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %f.patt: %pattern_type.dab = value_binding_pattern f [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc28_21.1: type = splice_block %.loc28_21.3 [concrete = Core.FloatLiteral] {
// CHECK:STDOUT:     %FloatLiteral.ref: %FloatLiteral.type = name_ref FloatLiteral, imports.%Main.FloatLiteral [concrete = constants.%FloatLiteral]
// CHECK:STDOUT:     %FloatLiteral.call: init type = call %FloatLiteral.ref() [concrete = Core.FloatLiteral]
// CHECK:STDOUT:     %.loc28_21.2: type = value_of_initializer %FloatLiteral.call [concrete = Core.FloatLiteral]
// CHECK:STDOUT:     %.loc28_21.3: type = converted %FloatLiteral.call, %.loc28_21.2 [concrete = Core.FloatLiteral]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc28_55.1: Core.FloatLiteral = value_of_initializer @__global_init.%FloatLiteralToFloatLiteral.call [concrete = constants.%float]
// CHECK:STDOUT:   %.loc28_55.2: Core.FloatLiteral = converted @__global_init.%FloatLiteralToFloatLiteral.call, %.loc28_55.1 [concrete = constants.%float]
// CHECK:STDOUT:   %f: Core.FloatLiteral = value_binding f, %.loc28_55.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %FloatLiteralToFloatLiteral.ref: %FloatLiteralToFloatLiteral.type = name_ref FloatLiteralToFloatLiteral, imports.%Main.FloatLiteralToFloatLiteral [concrete = constants.%FloatLiteralToFloatLiteral]
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float]
// CHECK:STDOUT:   %FloatLiteralToFloatLiteral.call: init Core.FloatLiteral = call %FloatLiteralToFloatLiteral.ref(%float) [concrete = constants.%float]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- literal_f64.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %f64.d77: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.0ae: type = pattern_type %f64.d77 [concrete]
// CHECK:STDOUT:   %FloatLiteralToFloat64.type: type = fn_type @FloatLiteralToFloat64 [concrete]
// CHECK:STDOUT:   %FloatLiteralToFloat64: %FloatLiteralToFloat64.type = struct_value () [concrete]
// CHECK:STDOUT:   %float.1f7: Core.FloatLiteral = float_literal_value 0e-1 [concrete]
// CHECK:STDOUT:   %float.0a8: %f64.d77 = float_value 0 [concrete]
// CHECK:STDOUT:   %float.6da: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %float.d20: %f64.d77 = float_value 1 [concrete]
// CHECK:STDOUT:   %float.12a: Core.FloatLiteral = float_literal_value 10e307 [concrete]
// CHECK:STDOUT:   %float.bde: %f64.d77 = float_value 1.0E+308 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.FloatLiteralToFloat64: %FloatLiteralToFloat64.type = import_ref Main//f64, FloatLiteralToFloat64, loaded [concrete = constants.%FloatLiteralToFloat64]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.0ae = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f64.loc58: type = type_literal constants.%f64.d77 [concrete = constants.%f64.d77]
// CHECK:STDOUT:   %.loc58_39.1: %f64.d77 = value_of_initializer @__global_init.%FloatLiteralToFloat64.call.loc58 [concrete = constants.%float.0a8]
// CHECK:STDOUT:   %.loc58_39.2: %f64.d77 = converted @__global_init.%FloatLiteralToFloat64.call.loc58, %.loc58_39.1 [concrete = constants.%float.0a8]
// CHECK:STDOUT:   %a: %f64.d77 = value_binding a, %.loc58_39.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.0ae = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f64.loc59: type = type_literal constants.%f64.d77 [concrete = constants.%f64.d77]
// CHECK:STDOUT:   %.loc59_39.1: %f64.d77 = value_of_initializer @__global_init.%FloatLiteralToFloat64.call.loc59 [concrete = constants.%float.d20]
// CHECK:STDOUT:   %.loc59_39.2: %f64.d77 = converted @__global_init.%FloatLiteralToFloat64.call.loc59, %.loc59_39.1 [concrete = constants.%float.d20]
// CHECK:STDOUT:   %b: %f64.d77 = value_binding b, %.loc59_39.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.0ae = value_binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f64.loc60: type = type_literal constants.%f64.d77 [concrete = constants.%f64.d77]
// CHECK:STDOUT:   %.loc60_43.1: %f64.d77 = value_of_initializer @__global_init.%FloatLiteralToFloat64.call.loc60 [concrete = constants.%float.bde]
// CHECK:STDOUT:   %.loc60_43.2: %f64.d77 = converted @__global_init.%FloatLiteralToFloat64.call.loc60, %.loc60_43.1 [concrete = constants.%float.bde]
// CHECK:STDOUT:   %c: %f64.d77 = value_binding c, %.loc60_43.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %FloatLiteralToFloat64.ref.loc58: %FloatLiteralToFloat64.type = name_ref FloatLiteralToFloat64, imports.%Main.FloatLiteralToFloat64 [concrete = constants.%FloatLiteralToFloat64]
// CHECK:STDOUT:   %float.loc58: Core.FloatLiteral = float_literal_value 0e-1 [concrete = constants.%float.1f7]
// CHECK:STDOUT:   %FloatLiteralToFloat64.call.loc58: init %f64.d77 = call %FloatLiteralToFloat64.ref.loc58(%float.loc58) [concrete = constants.%float.0a8]
// CHECK:STDOUT:   %FloatLiteralToFloat64.ref.loc59: %FloatLiteralToFloat64.type = name_ref FloatLiteralToFloat64, imports.%Main.FloatLiteralToFloat64 [concrete = constants.%FloatLiteralToFloat64]
// CHECK:STDOUT:   %float.loc59: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %FloatLiteralToFloat64.call.loc59: init %f64.d77 = call %FloatLiteralToFloat64.ref.loc59(%float.loc59) [concrete = constants.%float.d20]
// CHECK:STDOUT:   %FloatLiteralToFloat64.ref.loc60: %FloatLiteralToFloat64.type = name_ref FloatLiteralToFloat64, imports.%Main.FloatLiteralToFloat64 [concrete = constants.%FloatLiteralToFloat64]
// CHECK:STDOUT:   %float.loc60: Core.FloatLiteral = float_literal_value 10e307 [concrete = constants.%float.12a]
// CHECK:STDOUT:   %FloatLiteralToFloat64.call.loc60: init %f64.d77 = call %FloatLiteralToFloat64.ref.loc60(%float.loc60) [concrete = constants.%float.bde]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- literal_f32.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %f32.97e: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.201: type = pattern_type %f32.97e [concrete]
// CHECK:STDOUT:   %FloatLiteralToFloat32.type: type = fn_type @FloatLiteralToFloat32 [concrete]
// CHECK:STDOUT:   %FloatLiteralToFloat32: %FloatLiteralToFloat32.type = struct_value () [concrete]
// CHECK:STDOUT:   %float.1f7: Core.FloatLiteral = float_literal_value 0e-1 [concrete]
// CHECK:STDOUT:   %float.4db: %f32.97e = float_value 0 [concrete]
// CHECK:STDOUT:   %float.6da: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %float.e3b: %f32.97e = float_value 1 [concrete]
// CHECK:STDOUT:   %float.8d2: Core.FloatLiteral = float_literal_value 10e37 [concrete]
// CHECK:STDOUT:   %float.520: %f32.97e = float_value 9.99999968E+37 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.FloatLiteralToFloat32: %FloatLiteralToFloat32.type = import_ref Main//f32, FloatLiteralToFloat32, loaded [concrete = constants.%FloatLiteralToFloat32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.201 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f32.loc69: type = type_literal constants.%f32.97e [concrete = constants.%f32.97e]
// CHECK:STDOUT:   %.loc69_39.1: %f32.97e = value_of_initializer @__global_init.%FloatLiteralToFloat32.call.loc69 [concrete = constants.%float.4db]
// CHECK:STDOUT:   %.loc69_39.2: %f32.97e = converted @__global_init.%FloatLiteralToFloat32.call.loc69, %.loc69_39.1 [concrete = constants.%float.4db]
// CHECK:STDOUT:   %a: %f32.97e = value_binding a, %.loc69_39.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.201 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f32.loc70: type = type_literal constants.%f32.97e [concrete = constants.%f32.97e]
// CHECK:STDOUT:   %.loc70_39.1: %f32.97e = value_of_initializer @__global_init.%FloatLiteralToFloat32.call.loc70 [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc70_39.2: %f32.97e = converted @__global_init.%FloatLiteralToFloat32.call.loc70, %.loc70_39.1 [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %b: %f32.97e = value_binding b, %.loc70_39.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.201 = value_binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f32.loc71: type = type_literal constants.%f32.97e [concrete = constants.%f32.97e]
// CHECK:STDOUT:   %.loc71_42.1: %f32.97e = value_of_initializer @__global_init.%FloatLiteralToFloat32.call.loc71 [concrete = constants.%float.520]
// CHECK:STDOUT:   %.loc71_42.2: %f32.97e = converted @__global_init.%FloatLiteralToFloat32.call.loc71, %.loc71_42.1 [concrete = constants.%float.520]
// CHECK:STDOUT:   %c: %f32.97e = value_binding c, %.loc71_42.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %FloatLiteralToFloat32.ref.loc69: %FloatLiteralToFloat32.type = name_ref FloatLiteralToFloat32, imports.%Main.FloatLiteralToFloat32 [concrete = constants.%FloatLiteralToFloat32]
// CHECK:STDOUT:   %float.loc69: Core.FloatLiteral = float_literal_value 0e-1 [concrete = constants.%float.1f7]
// CHECK:STDOUT:   %FloatLiteralToFloat32.call.loc69: init %f32.97e = call %FloatLiteralToFloat32.ref.loc69(%float.loc69) [concrete = constants.%float.4db]
// CHECK:STDOUT:   %FloatLiteralToFloat32.ref.loc70: %FloatLiteralToFloat32.type = name_ref FloatLiteralToFloat32, imports.%Main.FloatLiteralToFloat32 [concrete = constants.%FloatLiteralToFloat32]
// CHECK:STDOUT:   %float.loc70: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %FloatLiteralToFloat32.call.loc70: init %f32.97e = call %FloatLiteralToFloat32.ref.loc70(%float.loc70) [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %FloatLiteralToFloat32.ref.loc71: %FloatLiteralToFloat32.type = name_ref FloatLiteralToFloat32, imports.%Main.FloatLiteralToFloat32 [concrete = constants.%FloatLiteralToFloat32]
// CHECK:STDOUT:   %float.loc71: Core.FloatLiteral = float_literal_value 10e37 [concrete = constants.%float.8d2]
// CHECK:STDOUT:   %FloatLiteralToFloat32.call.loc71: init %f32.97e = call %FloatLiteralToFloat32.ref.loc71(%float.loc71) [concrete = constants.%float.520]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- identity_f64.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %f64.d77: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.0ae: type = pattern_type %f64.d77 [concrete]
// CHECK:STDOUT:   %Float64ToFloat64.type: type = fn_type @Float64ToFloat64 [concrete]
// CHECK:STDOUT:   %Float64ToFloat64: %Float64ToFloat64.type = struct_value () [concrete]
// CHECK:STDOUT:   %float.1f7: Core.FloatLiteral = float_literal_value 0e-1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.4a8: type = facet_type <@ImplicitAs, @ImplicitAs(%f64.d77)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.02f: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.1f0: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.02f = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.cb2: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table, @Core.FloatLiteral.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.2c7: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.239: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.2c7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.4a8 = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.cb2) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.a33: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%f64.d77, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.6c5: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.a33, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.8c6: <bound method> = bound_method %float.1f7, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.239 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.ImplicitAs.impl.Convert.239, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.f2b: <bound method> = bound_method %float.1f7, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.0a8: %f64.d77 = float_value 0 [concrete]
// CHECK:STDOUT:   %float.6da: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.1ed: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.239 [concrete]
// CHECK:STDOUT:   %bound_method.50c: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.d20: %f64.d77 = float_value 1 [concrete]
// CHECK:STDOUT:   %float.12a: Core.FloatLiteral = float_literal_value 10e307 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.dee: <bound method> = bound_method %float.12a, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.239 [concrete]
// CHECK:STDOUT:   %bound_method.d56: <bound method> = bound_method %float.12a, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.bde: %f64.d77 = float_value 1.0E+308 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Float64ToFloat64: %Float64ToFloat64.type = import_ref Main//f64, Float64ToFloat64, loaded [concrete = constants.%Float64ToFloat64]
// CHECK:STDOUT:   %Core.import_ref.38a: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.02f) = import_ref Core//prelude/parts/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.1f0)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%Core.import_ref.38a), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.0ae = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f64.loc80: type = type_literal constants.%f64.d77 [concrete = constants.%f64.d77]
// CHECK:STDOUT:   %.loc80_34.1: %f64.d77 = value_of_initializer @__global_init.%Float64ToFloat64.call.loc80 [concrete = constants.%float.0a8]
// CHECK:STDOUT:   %.loc80_34.2: %f64.d77 = converted @__global_init.%Float64ToFloat64.call.loc80, %.loc80_34.1 [concrete = constants.%float.0a8]
// CHECK:STDOUT:   %a: %f64.d77 = value_binding a, %.loc80_34.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.0ae = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f64.loc81: type = type_literal constants.%f64.d77 [concrete = constants.%f64.d77]
// CHECK:STDOUT:   %.loc81_34.1: %f64.d77 = value_of_initializer @__global_init.%Float64ToFloat64.call.loc81 [concrete = constants.%float.d20]
// CHECK:STDOUT:   %.loc81_34.2: %f64.d77 = converted @__global_init.%Float64ToFloat64.call.loc81, %.loc81_34.1 [concrete = constants.%float.d20]
// CHECK:STDOUT:   %b: %f64.d77 = value_binding b, %.loc81_34.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.0ae = value_binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f64.loc82: type = type_literal constants.%f64.d77 [concrete = constants.%f64.d77]
// CHECK:STDOUT:   %.loc82_38.1: %f64.d77 = value_of_initializer @__global_init.%Float64ToFloat64.call.loc82 [concrete = constants.%float.bde]
// CHECK:STDOUT:   %.loc82_38.2: %f64.d77 = converted @__global_init.%Float64ToFloat64.call.loc82, %.loc82_38.1 [concrete = constants.%float.bde]
// CHECK:STDOUT:   %c: %f64.d77 = value_binding c, %.loc82_38.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Float64ToFloat64.ref.loc80: %Float64ToFloat64.type = name_ref Float64ToFloat64, imports.%Main.Float64ToFloat64 [concrete = constants.%Float64ToFloat64]
// CHECK:STDOUT:   %float.loc80: Core.FloatLiteral = float_literal_value 0e-1 [concrete = constants.%float.1f7]
// CHECK:STDOUT:   %impl.elem0.loc80: %.6c5 = impl_witness_access constants.%ImplicitAs.impl_witness.cb2, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.239]
// CHECK:STDOUT:   %bound_method.loc80_31.1: <bound method> = bound_method %float.loc80, %impl.elem0.loc80 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.8c6]
// CHECK:STDOUT:   %specific_fn.loc80: <specific function> = specific_function %impl.elem0.loc80, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc80_31.2: <bound method> = bound_method %float.loc80, %specific_fn.loc80 [concrete = constants.%bound_method.f2b]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc80: init %f64.d77 = call %bound_method.loc80_31.2(%float.loc80) [concrete = constants.%float.0a8]
// CHECK:STDOUT:   %.loc80_31.1: %f64.d77 = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc80 [concrete = constants.%float.0a8]
// CHECK:STDOUT:   %.loc80_31.2: %f64.d77 = converted %float.loc80, %.loc80_31.1 [concrete = constants.%float.0a8]
// CHECK:STDOUT:   %Float64ToFloat64.call.loc80: init %f64.d77 = call %Float64ToFloat64.ref.loc80(%.loc80_31.2) [concrete = constants.%float.0a8]
// CHECK:STDOUT:   %Float64ToFloat64.ref.loc81: %Float64ToFloat64.type = name_ref Float64ToFloat64, imports.%Main.Float64ToFloat64 [concrete = constants.%Float64ToFloat64]
// CHECK:STDOUT:   %float.loc81: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %impl.elem0.loc81: %.6c5 = impl_witness_access constants.%ImplicitAs.impl_witness.cb2, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.239]
// CHECK:STDOUT:   %bound_method.loc81_31.1: <bound method> = bound_method %float.loc81, %impl.elem0.loc81 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.1ed]
// CHECK:STDOUT:   %specific_fn.loc81: <specific function> = specific_function %impl.elem0.loc81, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc81_31.2: <bound method> = bound_method %float.loc81, %specific_fn.loc81 [concrete = constants.%bound_method.50c]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc81: init %f64.d77 = call %bound_method.loc81_31.2(%float.loc81) [concrete = constants.%float.d20]
// CHECK:STDOUT:   %.loc81_31.1: %f64.d77 = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc81 [concrete = constants.%float.d20]
// CHECK:STDOUT:   %.loc81_31.2: %f64.d77 = converted %float.loc81, %.loc81_31.1 [concrete = constants.%float.d20]
// CHECK:STDOUT:   %Float64ToFloat64.call.loc81: init %f64.d77 = call %Float64ToFloat64.ref.loc81(%.loc81_31.2) [concrete = constants.%float.d20]
// CHECK:STDOUT:   %Float64ToFloat64.ref.loc82: %Float64ToFloat64.type = name_ref Float64ToFloat64, imports.%Main.Float64ToFloat64 [concrete = constants.%Float64ToFloat64]
// CHECK:STDOUT:   %float.loc82: Core.FloatLiteral = float_literal_value 10e307 [concrete = constants.%float.12a]
// CHECK:STDOUT:   %impl.elem0.loc82: %.6c5 = impl_witness_access constants.%ImplicitAs.impl_witness.cb2, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.239]
// CHECK:STDOUT:   %bound_method.loc82_31.1: <bound method> = bound_method %float.loc82, %impl.elem0.loc82 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.dee]
// CHECK:STDOUT:   %specific_fn.loc82: <specific function> = specific_function %impl.elem0.loc82, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc82_31.2: <bound method> = bound_method %float.loc82, %specific_fn.loc82 [concrete = constants.%bound_method.d56]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc82: init %f64.d77 = call %bound_method.loc82_31.2(%float.loc82) [concrete = constants.%float.bde]
// CHECK:STDOUT:   %.loc82_31.1: %f64.d77 = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc82 [concrete = constants.%float.bde]
// CHECK:STDOUT:   %.loc82_31.2: %f64.d77 = converted %float.loc82, %.loc82_31.1 [concrete = constants.%float.bde]
// CHECK:STDOUT:   %Float64ToFloat64.call.loc82: init %f64.d77 = call %Float64ToFloat64.ref.loc82(%.loc82_31.2) [concrete = constants.%float.bde]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- identity_f32.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %f32.97e: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.201: type = pattern_type %f32.97e [concrete]
// CHECK:STDOUT:   %Float32ToFloat32.type: type = fn_type @Float32ToFloat32 [concrete]
// CHECK:STDOUT:   %Float32ToFloat32: %Float32ToFloat32.type = struct_value () [concrete]
// CHECK:STDOUT:   %float.1f7: Core.FloatLiteral = float_literal_value 0e-1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.223: type = facet_type <@ImplicitAs, @ImplicitAs(%f32.97e)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.02f: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.1f0: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.02f = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.bc6: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.461: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.461 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.223 = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.bc6) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.e4d: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%f32.97e, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.98d: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.e4d, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.c2d: <bound method> = bound_method %float.1f7, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.577: <bound method> = bound_method %float.1f7, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.4db: %f32.97e = float_value 0 [concrete]
// CHECK:STDOUT:   %float.6da: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.c4b: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55 [concrete]
// CHECK:STDOUT:   %bound_method.57d: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.e3b: %f32.97e = float_value 1 [concrete]
// CHECK:STDOUT:   %float.8d2: Core.FloatLiteral = float_literal_value 10e37 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.d1f: <bound method> = bound_method %float.8d2, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55 [concrete]
// CHECK:STDOUT:   %bound_method.bed: <bound method> = bound_method %float.8d2, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.520: %f32.97e = float_value 9.99999968E+37 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Float32ToFloat32: %Float32ToFloat32.type = import_ref Main//f32, Float32ToFloat32, loaded [concrete = constants.%Float32ToFloat32]
// CHECK:STDOUT:   %Core.import_ref.38a: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.02f) = import_ref Core//prelude/parts/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.1f0)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%Core.import_ref.38a), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.201 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f32.loc91: type = type_literal constants.%f32.97e [concrete = constants.%f32.97e]
// CHECK:STDOUT:   %.loc91_34.1: %f32.97e = value_of_initializer @__global_init.%Float32ToFloat32.call.loc91 [concrete = constants.%float.4db]
// CHECK:STDOUT:   %.loc91_34.2: %f32.97e = converted @__global_init.%Float32ToFloat32.call.loc91, %.loc91_34.1 [concrete = constants.%float.4db]
// CHECK:STDOUT:   %a: %f32.97e = value_binding a, %.loc91_34.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.201 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f32.loc92: type = type_literal constants.%f32.97e [concrete = constants.%f32.97e]
// CHECK:STDOUT:   %.loc92_34.1: %f32.97e = value_of_initializer @__global_init.%Float32ToFloat32.call.loc92 [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc92_34.2: %f32.97e = converted @__global_init.%Float32ToFloat32.call.loc92, %.loc92_34.1 [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %b: %f32.97e = value_binding b, %.loc92_34.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.201 = value_binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f32.loc93: type = type_literal constants.%f32.97e [concrete = constants.%f32.97e]
// CHECK:STDOUT:   %.loc93_37.1: %f32.97e = value_of_initializer @__global_init.%Float32ToFloat32.call.loc93 [concrete = constants.%float.520]
// CHECK:STDOUT:   %.loc93_37.2: %f32.97e = converted @__global_init.%Float32ToFloat32.call.loc93, %.loc93_37.1 [concrete = constants.%float.520]
// CHECK:STDOUT:   %c: %f32.97e = value_binding c, %.loc93_37.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Float32ToFloat32.ref.loc91: %Float32ToFloat32.type = name_ref Float32ToFloat32, imports.%Main.Float32ToFloat32 [concrete = constants.%Float32ToFloat32]
// CHECK:STDOUT:   %float.loc91: Core.FloatLiteral = float_literal_value 0e-1 [concrete = constants.%float.1f7]
// CHECK:STDOUT:   %impl.elem0.loc91: %.98d = impl_witness_access constants.%ImplicitAs.impl_witness.bc6, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55]
// CHECK:STDOUT:   %bound_method.loc91_31.1: <bound method> = bound_method %float.loc91, %impl.elem0.loc91 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.c2d]
// CHECK:STDOUT:   %specific_fn.loc91: <specific function> = specific_function %impl.elem0.loc91, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc91_31.2: <bound method> = bound_method %float.loc91, %specific_fn.loc91 [concrete = constants.%bound_method.577]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc91: init %f32.97e = call %bound_method.loc91_31.2(%float.loc91) [concrete = constants.%float.4db]
// CHECK:STDOUT:   %.loc91_31.1: %f32.97e = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc91 [concrete = constants.%float.4db]
// CHECK:STDOUT:   %.loc91_31.2: %f32.97e = converted %float.loc91, %.loc91_31.1 [concrete = constants.%float.4db]
// CHECK:STDOUT:   %Float32ToFloat32.call.loc91: init %f32.97e = call %Float32ToFloat32.ref.loc91(%.loc91_31.2) [concrete = constants.%float.4db]
// CHECK:STDOUT:   %Float32ToFloat32.ref.loc92: %Float32ToFloat32.type = name_ref Float32ToFloat32, imports.%Main.Float32ToFloat32 [concrete = constants.%Float32ToFloat32]
// CHECK:STDOUT:   %float.loc92: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %impl.elem0.loc92: %.98d = impl_witness_access constants.%ImplicitAs.impl_witness.bc6, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55]
// CHECK:STDOUT:   %bound_method.loc92_31.1: <bound method> = bound_method %float.loc92, %impl.elem0.loc92 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.c4b]
// CHECK:STDOUT:   %specific_fn.loc92: <specific function> = specific_function %impl.elem0.loc92, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc92_31.2: <bound method> = bound_method %float.loc92, %specific_fn.loc92 [concrete = constants.%bound_method.57d]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc92: init %f32.97e = call %bound_method.loc92_31.2(%float.loc92) [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc92_31.1: %f32.97e = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc92 [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc92_31.2: %f32.97e = converted %float.loc92, %.loc92_31.1 [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %Float32ToFloat32.call.loc92: init %f32.97e = call %Float32ToFloat32.ref.loc92(%.loc92_31.2) [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %Float32ToFloat32.ref.loc93: %Float32ToFloat32.type = name_ref Float32ToFloat32, imports.%Main.Float32ToFloat32 [concrete = constants.%Float32ToFloat32]
// CHECK:STDOUT:   %float.loc93: Core.FloatLiteral = float_literal_value 10e37 [concrete = constants.%float.8d2]
// CHECK:STDOUT:   %impl.elem0.loc93: %.98d = impl_witness_access constants.%ImplicitAs.impl_witness.bc6, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55]
// CHECK:STDOUT:   %bound_method.loc93_31.1: <bound method> = bound_method %float.loc93, %impl.elem0.loc93 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.d1f]
// CHECK:STDOUT:   %specific_fn.loc93: <specific function> = specific_function %impl.elem0.loc93, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc93_31.2: <bound method> = bound_method %float.loc93, %specific_fn.loc93 [concrete = constants.%bound_method.bed]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc93: init %f32.97e = call %bound_method.loc93_31.2(%float.loc93) [concrete = constants.%float.520]
// CHECK:STDOUT:   %.loc93_31.1: %f32.97e = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc93 [concrete = constants.%float.520]
// CHECK:STDOUT:   %.loc93_31.2: %f32.97e = converted %float.loc93, %.loc93_31.1 [concrete = constants.%float.520]
// CHECK:STDOUT:   %Float32ToFloat32.call.loc93: init %f32.97e = call %Float32ToFloat32.ref.loc93(%.loc93_31.2) [concrete = constants.%float.520]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- truncate.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %f32.97e: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.201: type = pattern_type %f32.97e [concrete]
// CHECK:STDOUT:   %Float64ToFloat32.type: type = fn_type @Float64ToFloat32 [concrete]
// CHECK:STDOUT:   %Float64ToFloat32: %Float64ToFloat32.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %f64.d77: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %float.6da: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.4a8: type = facet_type <@ImplicitAs, @ImplicitAs(%f64.d77)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.02f: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.1f0: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.02f = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.cb2: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table, @Core.FloatLiteral.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.2c7: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.239: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.2c7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.4a8 = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.cb2) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.a33: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%f64.d77, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.6c5: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.a33, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.239 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.ImplicitAs.impl.Convert.239, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.d20: %f64.d77 = float_value 1 [concrete]
// CHECK:STDOUT:   %float.e3b: %f32.97e = float_value 1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Float64ToFloat32: %Float64ToFloat32.type = import_ref Main//f32, Float64ToFloat32, loaded [concrete = constants.%Float64ToFloat32]
// CHECK:STDOUT:   %Core.import_ref.38a: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.02f) = import_ref Core//prelude/parts/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.1f0)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%Core.import_ref.38a), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.201 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f32: type = type_literal constants.%f32.97e [concrete = constants.%f32.97e]
// CHECK:STDOUT:   %.loc102_34.1: %f32.97e = value_of_initializer @__global_init.%Float64ToFloat32.call [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc102_34.2: %f32.97e = converted @__global_init.%Float64ToFloat32.call, %.loc102_34.1 [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %a: %f32.97e = value_binding a, %.loc102_34.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Float64ToFloat32.ref: %Float64ToFloat32.type = name_ref Float64ToFloat32, imports.%Main.Float64ToFloat32 [concrete = constants.%Float64ToFloat32]
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %impl.elem0: %.6c5 = impl_witness_access constants.%ImplicitAs.impl_witness.cb2, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.239]
// CHECK:STDOUT:   %bound_method.loc102_31.1: <bound method> = bound_method %float, %impl.elem0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc102_31.2: <bound method> = bound_method %float, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f64.d77 = call %bound_method.loc102_31.2(%float) [concrete = constants.%float.d20]
// CHECK:STDOUT:   %.loc102_31.1: %f64.d77 = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%float.d20]
// CHECK:STDOUT:   %.loc102_31.2: %f64.d77 = converted %float, %.loc102_31.1 [concrete = constants.%float.d20]
// CHECK:STDOUT:   %Float64ToFloat32.call: init %f32.97e = call %Float64ToFloat32.ref(%.loc102_31.2) [concrete = constants.%float.e3b]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_truncate_overflow.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %f32.97e: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.201: type = pattern_type %f32.97e [concrete]
// CHECK:STDOUT:   %Float64ToFloat32.type: type = fn_type @Float64ToFloat32 [concrete]
// CHECK:STDOUT:   %Float64ToFloat32: %Float64ToFloat32.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %f64.d77: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.0ae: type = pattern_type %f64.d77 [concrete]
// CHECK:STDOUT:   %float.cfcfd2.1: Core.FloatLiteral = float_literal_value 10e38 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.4a8: type = facet_type <@ImplicitAs, @ImplicitAs(%f64.d77)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.02f: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.1f0: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.02f = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.cb2: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table, @Core.FloatLiteral.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.2c7: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.239: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.2c7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.4a8 = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.cb2) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.a33: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%f64.d77, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.6c5: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.a33, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %float.cfcfd2.1, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.239 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.ImplicitAs.impl.Convert.239, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %float.cfcfd2.1, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.c37: %f64.d77 = float_value 9.9999999999999994E+38 [concrete]
// CHECK:STDOUT:   %FloatLiteralToFloat32.type: type = fn_type @FloatLiteralToFloat32 [concrete]
// CHECK:STDOUT:   %FloatLiteralToFloat32: %FloatLiteralToFloat32.type = struct_value () [concrete]
// CHECK:STDOUT:   %float.cfcfd2.2: Core.FloatLiteral = float_literal_value 10e38 [concrete]
// CHECK:STDOUT:   %FloatLiteralToFloat64.type: type = fn_type @FloatLiteralToFloat64 [concrete]
// CHECK:STDOUT:   %FloatLiteralToFloat64: %FloatLiteralToFloat64.type = struct_value () [concrete]
// CHECK:STDOUT:   %float.82f: Core.FloatLiteral = float_literal_value 10e308 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.FloatLiteralToFloat32: %FloatLiteralToFloat32.type = import_ref Main//f32, FloatLiteralToFloat32, loaded [concrete = constants.%FloatLiteralToFloat32]
// CHECK:STDOUT:   %Main.Float64ToFloat32: %Float64ToFloat32.type = import_ref Main//f32, Float64ToFloat32, loaded [concrete = constants.%Float64ToFloat32]
// CHECK:STDOUT:   %Main.FloatLiteralToFloat64: %FloatLiteralToFloat64.type = import_ref Main//f64, FloatLiteralToFloat64, loaded [concrete = constants.%FloatLiteralToFloat64]
// CHECK:STDOUT:   %Core.import_ref.38a: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.02f) = import_ref Core//prelude/parts/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.1f0)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%Core.import_ref.38a), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.201 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f32.loc116: type = type_literal constants.%f32.97e [concrete = constants.%f32.97e]
// CHECK:STDOUT:   %.loc116_37.1: %f32.97e = value_of_initializer @__global_init.%Float64ToFloat32.call [concrete = <error>]
// CHECK:STDOUT:   %.loc116_37.2: %f32.97e = converted @__global_init.%Float64ToFloat32.call, %.loc116_37.1 [concrete = <error>]
// CHECK:STDOUT:   %a: %f32.97e = value_binding a, %.loc116_37.2 [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.201 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f32.loc121: type = type_literal constants.%f32.97e [concrete = constants.%f32.97e]
// CHECK:STDOUT:   %.loc121_42.1: %f32.97e = value_of_initializer @__global_init.%FloatLiteralToFloat32.call [concrete = <error>]
// CHECK:STDOUT:   %.loc121_42.2: %f32.97e = converted @__global_init.%FloatLiteralToFloat32.call, %.loc121_42.1 [concrete = <error>]
// CHECK:STDOUT:   %b: %f32.97e = value_binding b, %.loc121_42.2 [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.0ae = value_binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f64: type = type_literal constants.%f64.d77 [concrete = constants.%f64.d77]
// CHECK:STDOUT:   %.loc126_43.1: %f64.d77 = value_of_initializer @__global_init.%FloatLiteralToFloat64.call [concrete = <error>]
// CHECK:STDOUT:   %.loc126_43.2: %f64.d77 = converted @__global_init.%FloatLiteralToFloat64.call, %.loc126_43.1 [concrete = <error>]
// CHECK:STDOUT:   %c: %f64.d77 = value_binding c, %.loc126_43.2 [concrete = <error>]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Float64ToFloat32.ref: %Float64ToFloat32.type = name_ref Float64ToFloat32, imports.%Main.Float64ToFloat32 [concrete = constants.%Float64ToFloat32]
// CHECK:STDOUT:   %float.loc116: Core.FloatLiteral = float_literal_value 10e38 [concrete = constants.%float.cfcfd2.1]
// CHECK:STDOUT:   %impl.elem0: %.6c5 = impl_witness_access constants.%ImplicitAs.impl_witness.cb2, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.239]
// CHECK:STDOUT:   %bound_method.loc116_31.1: <bound method> = bound_method %float.loc116, %impl.elem0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc116_31.2: <bound method> = bound_method %float.loc116, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f64.d77 = call %bound_method.loc116_31.2(%float.loc116) [concrete = constants.%float.c37]
// CHECK:STDOUT:   %.loc116_31.1: %f64.d77 = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%float.c37]
// CHECK:STDOUT:   %.loc116_31.2: %f64.d77 = converted %float.loc116, %.loc116_31.1 [concrete = constants.%float.c37]
// CHECK:STDOUT:   %Float64ToFloat32.call: init %f32.97e = call %Float64ToFloat32.ref(%.loc116_31.2) [concrete = <error>]
// CHECK:STDOUT:   %FloatLiteralToFloat32.ref: %FloatLiteralToFloat32.type = name_ref FloatLiteralToFloat32, imports.%Main.FloatLiteralToFloat32 [concrete = constants.%FloatLiteralToFloat32]
// CHECK:STDOUT:   %float.loc121: Core.FloatLiteral = float_literal_value 10e38 [concrete = constants.%float.cfcfd2.2]
// CHECK:STDOUT:   %FloatLiteralToFloat32.call: init %f32.97e = call %FloatLiteralToFloat32.ref(%float.loc121) [concrete = <error>]
// CHECK:STDOUT:   %FloatLiteralToFloat64.ref: %FloatLiteralToFloat64.type = name_ref FloatLiteralToFloat64, imports.%Main.FloatLiteralToFloat64 [concrete = constants.%FloatLiteralToFloat64]
// CHECK:STDOUT:   %float.loc126: Core.FloatLiteral = float_literal_value 10e308 [concrete = constants.%float.82f]
// CHECK:STDOUT:   %FloatLiteralToFloat64.call: init %f64.d77 = call %FloatLiteralToFloat64.ref(%float.loc126) [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- extend.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %f64.d77: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.0ae: type = pattern_type %f64.d77 [concrete]
// CHECK:STDOUT:   %Float32ToFloat64.type: type = fn_type @Float32ToFloat64 [concrete]
// CHECK:STDOUT:   %Float32ToFloat64: %Float32ToFloat64.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %f32.97e: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %float.6da: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.223: type = facet_type <@ImplicitAs, @ImplicitAs(%f32.97e)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.02f: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.1f0: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.02f = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.bc6: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.461: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.461 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.223 = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.bc6) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.e4d: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%f32.97e, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.98d: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.e4d, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.c4b: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.57d: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.e3b: %f32.97e = float_value 1 [concrete]
// CHECK:STDOUT:   %float.d20: %f64.d77 = float_value 1 [concrete]
// CHECK:STDOUT:   %float.c02: Core.FloatLiteral = float_literal_value 10e29 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.9ad: <bound method> = bound_method %float.c02, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55 [concrete]
// CHECK:STDOUT:   %bound_method.601: <bound method> = bound_method %float.c02, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.7d4: %f32.97e = float_value 1.00000002E+30 [concrete]
// CHECK:STDOUT:   %float.6a7: %f64.d77 = float_value 1.0000000150474662E+30 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Float32ToFloat64: %Float32ToFloat64.type = import_ref Main//f32, Float32ToFloat64, loaded [concrete = constants.%Float32ToFloat64]
// CHECK:STDOUT:   %Core.import_ref.38a: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.02f) = import_ref Core//prelude/parts/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.1f0)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%Core.import_ref.38a), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.0ae = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f64.loc135: type = type_literal constants.%f64.d77 [concrete = constants.%f64.d77]
// CHECK:STDOUT:   %.loc135_34.1: %f64.d77 = value_of_initializer @__global_init.%Float32ToFloat64.call.loc135 [concrete = constants.%float.d20]
// CHECK:STDOUT:   %.loc135_34.2: %f64.d77 = converted @__global_init.%Float32ToFloat64.call.loc135, %.loc135_34.1 [concrete = constants.%float.d20]
// CHECK:STDOUT:   %a: %f64.d77 = value_binding a, %.loc135_34.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.0ae = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f64.loc136: type = type_literal constants.%f64.d77 [concrete = constants.%f64.d77]
// CHECK:STDOUT:   %.loc136_37.1: %f64.d77 = value_of_initializer @__global_init.%Float32ToFloat64.call.loc136 [concrete = constants.%float.6a7]
// CHECK:STDOUT:   %.loc136_37.2: %f64.d77 = converted @__global_init.%Float32ToFloat64.call.loc136, %.loc136_37.1 [concrete = constants.%float.6a7]
// CHECK:STDOUT:   %b: %f64.d77 = value_binding b, %.loc136_37.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Float32ToFloat64.ref.loc135: %Float32ToFloat64.type = name_ref Float32ToFloat64, imports.%Main.Float32ToFloat64 [concrete = constants.%Float32ToFloat64]
// CHECK:STDOUT:   %float.loc135: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %impl.elem0.loc135: %.98d = impl_witness_access constants.%ImplicitAs.impl_witness.bc6, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55]
// CHECK:STDOUT:   %bound_method.loc135_31.1: <bound method> = bound_method %float.loc135, %impl.elem0.loc135 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.c4b]
// CHECK:STDOUT:   %specific_fn.loc135: <specific function> = specific_function %impl.elem0.loc135, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc135_31.2: <bound method> = bound_method %float.loc135, %specific_fn.loc135 [concrete = constants.%bound_method.57d]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc135: init %f32.97e = call %bound_method.loc135_31.2(%float.loc135) [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc135_31.1: %f32.97e = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc135 [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc135_31.2: %f32.97e = converted %float.loc135, %.loc135_31.1 [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %Float32ToFloat64.call.loc135: init %f64.d77 = call %Float32ToFloat64.ref.loc135(%.loc135_31.2) [concrete = constants.%float.d20]
// CHECK:STDOUT:   %Float32ToFloat64.ref.loc136: %Float32ToFloat64.type = name_ref Float32ToFloat64, imports.%Main.Float32ToFloat64 [concrete = constants.%Float32ToFloat64]
// CHECK:STDOUT:   %float.loc136: Core.FloatLiteral = float_literal_value 10e29 [concrete = constants.%float.c02]
// CHECK:STDOUT:   %impl.elem0.loc136: %.98d = impl_witness_access constants.%ImplicitAs.impl_witness.bc6, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55]
// CHECK:STDOUT:   %bound_method.loc136_31.1: <bound method> = bound_method %float.loc136, %impl.elem0.loc136 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.9ad]
// CHECK:STDOUT:   %specific_fn.loc136: <specific function> = specific_function %impl.elem0.loc136, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc136_31.2: <bound method> = bound_method %float.loc136, %specific_fn.loc136 [concrete = constants.%bound_method.601]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc136: init %f32.97e = call %bound_method.loc136_31.2(%float.loc136) [concrete = constants.%float.7d4]
// CHECK:STDOUT:   %.loc136_31.1: %f32.97e = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc136 [concrete = constants.%float.7d4]
// CHECK:STDOUT:   %.loc136_31.2: %f32.97e = converted %float.loc136, %.loc136_31.1 [concrete = constants.%float.7d4]
// CHECK:STDOUT:   %Float32ToFloat64.call.loc136: init %f64.d77 = call %Float32ToFloat64.ref.loc136(%.loc136_31.2) [concrete = constants.%float.6a7]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_not_constant.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %f64.d77: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.0ae: type = pattern_type %f64.d77 [concrete]
// CHECK:STDOUT:   %float.1f7: Core.FloatLiteral = float_literal_value 0e-1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.4a8: type = facet_type <@ImplicitAs, @ImplicitAs(%f64.d77)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.02f: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.1f0: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.02f = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.cb2: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table, @Core.FloatLiteral.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.2c7: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.239: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.2c7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.4a8 = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.cb2) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.a33: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%f64.d77, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.6c5: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.a33, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %float.1f7, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.239 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.ImplicitAs.impl.Convert.239, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %float.1f7, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.0a8: %f64.d77 = float_value 0 [concrete]
// CHECK:STDOUT:   %Float64ToFloat64.type: type = fn_type @Float64ToFloat64 [concrete]
// CHECK:STDOUT:   %Float64ToFloat64: %Float64ToFloat64.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Float64ToFloat64: %Float64ToFloat64.type = import_ref Main//f64, Float64ToFloat64, loaded [concrete = constants.%Float64ToFloat64]
// CHECK:STDOUT:   %Core.import_ref.38a: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.02f) = import_ref Core//prelude/parts/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.1f0)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%Core.import_ref.38a), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %not_constant_64.patt: %pattern_type.0ae = value_binding_pattern not_constant_64 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f64.loc145: type = type_literal constants.%f64.d77 [concrete = constants.%f64.d77]
// CHECK:STDOUT:   %impl.elem0: %.6c5 = impl_witness_access constants.%ImplicitAs.impl_witness.cb2, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.239]
// CHECK:STDOUT:   %bound_method.loc145_28.1: <bound method> = bound_method @__global_init.%float, %impl.elem0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc145_28.2: <bound method> = bound_method @__global_init.%float, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f64.d77 = call %bound_method.loc145_28.2(@__global_init.%float) [concrete = constants.%float.0a8]
// CHECK:STDOUT:   %.loc145_28.1: %f64.d77 = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%float.0a8]
// CHECK:STDOUT:   %.loc145_28.2: %f64.d77 = converted @__global_init.%float, %.loc145_28.1 [concrete = constants.%float.0a8]
// CHECK:STDOUT:   %not_constant_64: %f64.d77 = value_binding not_constant_64, %.loc145_28.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %convert_not_constant.patt: %pattern_type.0ae = value_binding_pattern convert_not_constant [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f64.loc154: type = type_literal constants.%f64.d77 [concrete = constants.%f64.d77]
// CHECK:STDOUT:   %.loc154_65.1: %f64.d77 = value_of_initializer @__global_init.%Float64ToFloat64.call
// CHECK:STDOUT:   %.loc154_65.2: %f64.d77 = converted @__global_init.%Float64ToFloat64.call, %.loc154_65.1
// CHECK:STDOUT:   %convert_not_constant: %f64.d77 = value_binding convert_not_constant, %.loc154_65.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 0e-1 [concrete = constants.%float.1f7]
// CHECK:STDOUT:   %Float64ToFloat64.ref: %Float64ToFloat64.type = name_ref Float64ToFloat64, imports.%Main.Float64ToFloat64 [concrete = constants.%Float64ToFloat64]
// CHECK:STDOUT:   %not_constant_64.ref: %f64.d77 = name_ref not_constant_64, file.%not_constant_64
// CHECK:STDOUT:   %Float64ToFloat64.call: init %f64.d77 = call %Float64ToFloat64.ref(%not_constant_64.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:

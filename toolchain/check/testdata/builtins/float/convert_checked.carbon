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
// CHECK:STDERR: f64.carbon:5:1: note: compile-time-only function declared here [CompTimeOnlyFunctionHere]
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
// CHECK:STDOUT:   %.loc13.1: type = splice_block %.loc13.3 [concrete = Core.FloatLiteral] {
// CHECK:STDOUT:     %FloatLiteral.ref: %FloatLiteral.type = name_ref FloatLiteral, imports.%Main.FloatLiteral [concrete = constants.%FloatLiteral]
// CHECK:STDOUT:     %FloatLiteral.call: init type = call %FloatLiteral.ref() [concrete = Core.FloatLiteral]
// CHECK:STDOUT:     %.loc13.2: type = value_of_initializer %FloatLiteral.call [concrete = Core.FloatLiteral]
// CHECK:STDOUT:     %.loc13.3: type = converted %FloatLiteral.call, %.loc13.2 [concrete = Core.FloatLiteral]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc18.1: Core.FloatLiteral = value_of_initializer @__global_init.%FloatLiteralToFloatLiteral.call [concrete = constants.%float]
// CHECK:STDOUT:   %.loc18.2: Core.FloatLiteral = converted @__global_init.%FloatLiteralToFloatLiteral.call, %.loc18.1 [concrete = constants.%float]
// CHECK:STDOUT:   %f: Core.FloatLiteral = value_binding f, %.loc18.2
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
// CHECK:STDOUT:   %f64.loc11: type = type_literal constants.%f64.d77 [concrete = constants.%f64.d77]
// CHECK:STDOUT:   %.loc16.1: %f64.d77 = value_of_initializer @__global_init.%FloatLiteralToFloat64.call.loc16 [concrete = constants.%float.0a8]
// CHECK:STDOUT:   %.loc16.2: %f64.d77 = converted @__global_init.%FloatLiteralToFloat64.call.loc16, %.loc16.1 [concrete = constants.%float.0a8]
// CHECK:STDOUT:   %a: %f64.d77 = value_binding a, %.loc16.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.0ae = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f64.loc21: type = type_literal constants.%f64.d77 [concrete = constants.%f64.d77]
// CHECK:STDOUT:   %.loc26.1: %f64.d77 = value_of_initializer @__global_init.%FloatLiteralToFloat64.call.loc26 [concrete = constants.%float.d20]
// CHECK:STDOUT:   %.loc26.2: %f64.d77 = converted @__global_init.%FloatLiteralToFloat64.call.loc26, %.loc26.1 [concrete = constants.%float.d20]
// CHECK:STDOUT:   %b: %f64.d77 = value_binding b, %.loc26.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.0ae = value_binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f64.loc31: type = type_literal constants.%f64.d77 [concrete = constants.%f64.d77]
// CHECK:STDOUT:   %.loc36.1: %f64.d77 = value_of_initializer @__global_init.%FloatLiteralToFloat64.call.loc36 [concrete = constants.%float.bde]
// CHECK:STDOUT:   %.loc36.2: %f64.d77 = converted @__global_init.%FloatLiteralToFloat64.call.loc36, %.loc36.1 [concrete = constants.%float.bde]
// CHECK:STDOUT:   %c: %f64.d77 = value_binding c, %.loc36.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %FloatLiteralToFloat64.ref.loc13: %FloatLiteralToFloat64.type = name_ref FloatLiteralToFloat64, imports.%Main.FloatLiteralToFloat64 [concrete = constants.%FloatLiteralToFloat64]
// CHECK:STDOUT:   %float.loc15: Core.FloatLiteral = float_literal_value 0e-1 [concrete = constants.%float.1f7]
// CHECK:STDOUT:   %FloatLiteralToFloat64.call.loc16: init %f64.d77 = call %FloatLiteralToFloat64.ref.loc13(%float.loc15) [concrete = constants.%float.0a8]
// CHECK:STDOUT:   %FloatLiteralToFloat64.ref.loc23: %FloatLiteralToFloat64.type = name_ref FloatLiteralToFloat64, imports.%Main.FloatLiteralToFloat64 [concrete = constants.%FloatLiteralToFloat64]
// CHECK:STDOUT:   %float.loc25: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %FloatLiteralToFloat64.call.loc26: init %f64.d77 = call %FloatLiteralToFloat64.ref.loc23(%float.loc25) [concrete = constants.%float.d20]
// CHECK:STDOUT:   %FloatLiteralToFloat64.ref.loc33: %FloatLiteralToFloat64.type = name_ref FloatLiteralToFloat64, imports.%Main.FloatLiteralToFloat64 [concrete = constants.%FloatLiteralToFloat64]
// CHECK:STDOUT:   %float.loc35: Core.FloatLiteral = float_literal_value 10e307 [concrete = constants.%float.12a]
// CHECK:STDOUT:   %FloatLiteralToFloat64.call.loc36: init %f64.d77 = call %FloatLiteralToFloat64.ref.loc33(%float.loc35) [concrete = constants.%float.bde]
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
// CHECK:STDOUT:   %f32.loc11: type = type_literal constants.%f32.97e [concrete = constants.%f32.97e]
// CHECK:STDOUT:   %.loc16.1: %f32.97e = value_of_initializer @__global_init.%FloatLiteralToFloat32.call.loc16 [concrete = constants.%float.4db]
// CHECK:STDOUT:   %.loc16.2: %f32.97e = converted @__global_init.%FloatLiteralToFloat32.call.loc16, %.loc16.1 [concrete = constants.%float.4db]
// CHECK:STDOUT:   %a: %f32.97e = value_binding a, %.loc16.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.201 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f32.loc21: type = type_literal constants.%f32.97e [concrete = constants.%f32.97e]
// CHECK:STDOUT:   %.loc26.1: %f32.97e = value_of_initializer @__global_init.%FloatLiteralToFloat32.call.loc26 [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc26.2: %f32.97e = converted @__global_init.%FloatLiteralToFloat32.call.loc26, %.loc26.1 [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %b: %f32.97e = value_binding b, %.loc26.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.201 = value_binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f32.loc31: type = type_literal constants.%f32.97e [concrete = constants.%f32.97e]
// CHECK:STDOUT:   %.loc36.1: %f32.97e = value_of_initializer @__global_init.%FloatLiteralToFloat32.call.loc36 [concrete = constants.%float.520]
// CHECK:STDOUT:   %.loc36.2: %f32.97e = converted @__global_init.%FloatLiteralToFloat32.call.loc36, %.loc36.1 [concrete = constants.%float.520]
// CHECK:STDOUT:   %c: %f32.97e = value_binding c, %.loc36.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %FloatLiteralToFloat32.ref.loc13: %FloatLiteralToFloat32.type = name_ref FloatLiteralToFloat32, imports.%Main.FloatLiteralToFloat32 [concrete = constants.%FloatLiteralToFloat32]
// CHECK:STDOUT:   %float.loc15: Core.FloatLiteral = float_literal_value 0e-1 [concrete = constants.%float.1f7]
// CHECK:STDOUT:   %FloatLiteralToFloat32.call.loc16: init %f32.97e = call %FloatLiteralToFloat32.ref.loc13(%float.loc15) [concrete = constants.%float.4db]
// CHECK:STDOUT:   %FloatLiteralToFloat32.ref.loc23: %FloatLiteralToFloat32.type = name_ref FloatLiteralToFloat32, imports.%Main.FloatLiteralToFloat32 [concrete = constants.%FloatLiteralToFloat32]
// CHECK:STDOUT:   %float.loc25: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %FloatLiteralToFloat32.call.loc26: init %f32.97e = call %FloatLiteralToFloat32.ref.loc23(%float.loc25) [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %FloatLiteralToFloat32.ref.loc33: %FloatLiteralToFloat32.type = name_ref FloatLiteralToFloat32, imports.%Main.FloatLiteralToFloat32 [concrete = constants.%FloatLiteralToFloat32]
// CHECK:STDOUT:   %float.loc35: Core.FloatLiteral = float_literal_value 10e37 [concrete = constants.%float.8d2]
// CHECK:STDOUT:   %FloatLiteralToFloat32.call.loc36: init %f32.97e = call %FloatLiteralToFloat32.ref.loc33(%float.loc35) [concrete = constants.%float.520]
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
// CHECK:STDOUT:   %Core.import_ref.38a: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.02f) = import_ref Core//prelude/parts/float, loc124, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.1f0)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%Core.import_ref.38a), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.0ae = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f64.loc11: type = type_literal constants.%f64.d77 [concrete = constants.%f64.d77]
// CHECK:STDOUT:   %.loc16.1: %f64.d77 = value_of_initializer @__global_init.%Float64ToFloat64.call.loc16 [concrete = constants.%float.0a8]
// CHECK:STDOUT:   %.loc16.2: %f64.d77 = converted @__global_init.%Float64ToFloat64.call.loc16, %.loc16.1 [concrete = constants.%float.0a8]
// CHECK:STDOUT:   %a: %f64.d77 = value_binding a, %.loc16.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.0ae = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f64.loc21: type = type_literal constants.%f64.d77 [concrete = constants.%f64.d77]
// CHECK:STDOUT:   %.loc26.1: %f64.d77 = value_of_initializer @__global_init.%Float64ToFloat64.call.loc26 [concrete = constants.%float.d20]
// CHECK:STDOUT:   %.loc26.2: %f64.d77 = converted @__global_init.%Float64ToFloat64.call.loc26, %.loc26.1 [concrete = constants.%float.d20]
// CHECK:STDOUT:   %b: %f64.d77 = value_binding b, %.loc26.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.0ae = value_binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f64.loc31: type = type_literal constants.%f64.d77 [concrete = constants.%f64.d77]
// CHECK:STDOUT:   %.loc36.1: %f64.d77 = value_of_initializer @__global_init.%Float64ToFloat64.call.loc36 [concrete = constants.%float.bde]
// CHECK:STDOUT:   %.loc36.2: %f64.d77 = converted @__global_init.%Float64ToFloat64.call.loc36, %.loc36.1 [concrete = constants.%float.bde]
// CHECK:STDOUT:   %c: %f64.d77 = value_binding c, %.loc36.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Float64ToFloat64.ref.loc13: %Float64ToFloat64.type = name_ref Float64ToFloat64, imports.%Main.Float64ToFloat64 [concrete = constants.%Float64ToFloat64]
// CHECK:STDOUT:   %float.loc15: Core.FloatLiteral = float_literal_value 0e-1 [concrete = constants.%float.1f7]
// CHECK:STDOUT:   %impl.elem0.loc15: %.6c5 = impl_witness_access constants.%ImplicitAs.impl_witness.cb2, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.239]
// CHECK:STDOUT:   %bound_method.loc15.1: <bound method> = bound_method %float.loc15, %impl.elem0.loc15 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.8c6]
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem0.loc15, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15.2: <bound method> = bound_method %float.loc15, %specific_fn.loc15 [concrete = constants.%bound_method.f2b]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc15: init %f64.d77 = call %bound_method.loc15.2(%float.loc15) [concrete = constants.%float.0a8]
// CHECK:STDOUT:   %.loc15.1: %f64.d77 = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc15 [concrete = constants.%float.0a8]
// CHECK:STDOUT:   %.loc15.2: %f64.d77 = converted %float.loc15, %.loc15.1 [concrete = constants.%float.0a8]
// CHECK:STDOUT:   %Float64ToFloat64.call.loc16: init %f64.d77 = call %Float64ToFloat64.ref.loc13(%.loc15.2) [concrete = constants.%float.0a8]
// CHECK:STDOUT:   %Float64ToFloat64.ref.loc23: %Float64ToFloat64.type = name_ref Float64ToFloat64, imports.%Main.Float64ToFloat64 [concrete = constants.%Float64ToFloat64]
// CHECK:STDOUT:   %float.loc25: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %impl.elem0.loc25: %.6c5 = impl_witness_access constants.%ImplicitAs.impl_witness.cb2, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.239]
// CHECK:STDOUT:   %bound_method.loc25.1: <bound method> = bound_method %float.loc25, %impl.elem0.loc25 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.1ed]
// CHECK:STDOUT:   %specific_fn.loc25: <specific function> = specific_function %impl.elem0.loc25, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc25.2: <bound method> = bound_method %float.loc25, %specific_fn.loc25 [concrete = constants.%bound_method.50c]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc25: init %f64.d77 = call %bound_method.loc25.2(%float.loc25) [concrete = constants.%float.d20]
// CHECK:STDOUT:   %.loc25.1: %f64.d77 = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc25 [concrete = constants.%float.d20]
// CHECK:STDOUT:   %.loc25.2: %f64.d77 = converted %float.loc25, %.loc25.1 [concrete = constants.%float.d20]
// CHECK:STDOUT:   %Float64ToFloat64.call.loc26: init %f64.d77 = call %Float64ToFloat64.ref.loc23(%.loc25.2) [concrete = constants.%float.d20]
// CHECK:STDOUT:   %Float64ToFloat64.ref.loc33: %Float64ToFloat64.type = name_ref Float64ToFloat64, imports.%Main.Float64ToFloat64 [concrete = constants.%Float64ToFloat64]
// CHECK:STDOUT:   %float.loc35: Core.FloatLiteral = float_literal_value 10e307 [concrete = constants.%float.12a]
// CHECK:STDOUT:   %impl.elem0.loc35: %.6c5 = impl_witness_access constants.%ImplicitAs.impl_witness.cb2, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.239]
// CHECK:STDOUT:   %bound_method.loc35.1: <bound method> = bound_method %float.loc35, %impl.elem0.loc35 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.dee]
// CHECK:STDOUT:   %specific_fn.loc35: <specific function> = specific_function %impl.elem0.loc35, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc35.2: <bound method> = bound_method %float.loc35, %specific_fn.loc35 [concrete = constants.%bound_method.d56]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc35: init %f64.d77 = call %bound_method.loc35.2(%float.loc35) [concrete = constants.%float.bde]
// CHECK:STDOUT:   %.loc35.1: %f64.d77 = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc35 [concrete = constants.%float.bde]
// CHECK:STDOUT:   %.loc35.2: %f64.d77 = converted %float.loc35, %.loc35.1 [concrete = constants.%float.bde]
// CHECK:STDOUT:   %Float64ToFloat64.call.loc36: init %f64.d77 = call %Float64ToFloat64.ref.loc33(%.loc35.2) [concrete = constants.%float.bde]
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
// CHECK:STDOUT:   %Core.import_ref.38a: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.02f) = import_ref Core//prelude/parts/float, loc124, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.1f0)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%Core.import_ref.38a), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.201 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f32.loc11: type = type_literal constants.%f32.97e [concrete = constants.%f32.97e]
// CHECK:STDOUT:   %.loc16.1: %f32.97e = value_of_initializer @__global_init.%Float32ToFloat32.call.loc16 [concrete = constants.%float.4db]
// CHECK:STDOUT:   %.loc16.2: %f32.97e = converted @__global_init.%Float32ToFloat32.call.loc16, %.loc16.1 [concrete = constants.%float.4db]
// CHECK:STDOUT:   %a: %f32.97e = value_binding a, %.loc16.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.201 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f32.loc21: type = type_literal constants.%f32.97e [concrete = constants.%f32.97e]
// CHECK:STDOUT:   %.loc26.1: %f32.97e = value_of_initializer @__global_init.%Float32ToFloat32.call.loc26 [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc26.2: %f32.97e = converted @__global_init.%Float32ToFloat32.call.loc26, %.loc26.1 [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %b: %f32.97e = value_binding b, %.loc26.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.201 = value_binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f32.loc31: type = type_literal constants.%f32.97e [concrete = constants.%f32.97e]
// CHECK:STDOUT:   %.loc36.1: %f32.97e = value_of_initializer @__global_init.%Float32ToFloat32.call.loc36 [concrete = constants.%float.520]
// CHECK:STDOUT:   %.loc36.2: %f32.97e = converted @__global_init.%Float32ToFloat32.call.loc36, %.loc36.1 [concrete = constants.%float.520]
// CHECK:STDOUT:   %c: %f32.97e = value_binding c, %.loc36.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Float32ToFloat32.ref.loc13: %Float32ToFloat32.type = name_ref Float32ToFloat32, imports.%Main.Float32ToFloat32 [concrete = constants.%Float32ToFloat32]
// CHECK:STDOUT:   %float.loc15: Core.FloatLiteral = float_literal_value 0e-1 [concrete = constants.%float.1f7]
// CHECK:STDOUT:   %impl.elem0.loc15: %.98d = impl_witness_access constants.%ImplicitAs.impl_witness.bc6, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55]
// CHECK:STDOUT:   %bound_method.loc15.1: <bound method> = bound_method %float.loc15, %impl.elem0.loc15 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.c2d]
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem0.loc15, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15.2: <bound method> = bound_method %float.loc15, %specific_fn.loc15 [concrete = constants.%bound_method.577]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc15: init %f32.97e = call %bound_method.loc15.2(%float.loc15) [concrete = constants.%float.4db]
// CHECK:STDOUT:   %.loc15.1: %f32.97e = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc15 [concrete = constants.%float.4db]
// CHECK:STDOUT:   %.loc15.2: %f32.97e = converted %float.loc15, %.loc15.1 [concrete = constants.%float.4db]
// CHECK:STDOUT:   %Float32ToFloat32.call.loc16: init %f32.97e = call %Float32ToFloat32.ref.loc13(%.loc15.2) [concrete = constants.%float.4db]
// CHECK:STDOUT:   %Float32ToFloat32.ref.loc23: %Float32ToFloat32.type = name_ref Float32ToFloat32, imports.%Main.Float32ToFloat32 [concrete = constants.%Float32ToFloat32]
// CHECK:STDOUT:   %float.loc25: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %impl.elem0.loc25: %.98d = impl_witness_access constants.%ImplicitAs.impl_witness.bc6, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55]
// CHECK:STDOUT:   %bound_method.loc25.1: <bound method> = bound_method %float.loc25, %impl.elem0.loc25 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.c4b]
// CHECK:STDOUT:   %specific_fn.loc25: <specific function> = specific_function %impl.elem0.loc25, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc25.2: <bound method> = bound_method %float.loc25, %specific_fn.loc25 [concrete = constants.%bound_method.57d]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc25: init %f32.97e = call %bound_method.loc25.2(%float.loc25) [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc25.1: %f32.97e = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc25 [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc25.2: %f32.97e = converted %float.loc25, %.loc25.1 [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %Float32ToFloat32.call.loc26: init %f32.97e = call %Float32ToFloat32.ref.loc23(%.loc25.2) [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %Float32ToFloat32.ref.loc33: %Float32ToFloat32.type = name_ref Float32ToFloat32, imports.%Main.Float32ToFloat32 [concrete = constants.%Float32ToFloat32]
// CHECK:STDOUT:   %float.loc35: Core.FloatLiteral = float_literal_value 10e37 [concrete = constants.%float.8d2]
// CHECK:STDOUT:   %impl.elem0.loc35: %.98d = impl_witness_access constants.%ImplicitAs.impl_witness.bc6, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55]
// CHECK:STDOUT:   %bound_method.loc35.1: <bound method> = bound_method %float.loc35, %impl.elem0.loc35 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.d1f]
// CHECK:STDOUT:   %specific_fn.loc35: <specific function> = specific_function %impl.elem0.loc35, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc35.2: <bound method> = bound_method %float.loc35, %specific_fn.loc35 [concrete = constants.%bound_method.bed]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc35: init %f32.97e = call %bound_method.loc35.2(%float.loc35) [concrete = constants.%float.520]
// CHECK:STDOUT:   %.loc35.1: %f32.97e = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc35 [concrete = constants.%float.520]
// CHECK:STDOUT:   %.loc35.2: %f32.97e = converted %float.loc35, %.loc35.1 [concrete = constants.%float.520]
// CHECK:STDOUT:   %Float32ToFloat32.call.loc36: init %f32.97e = call %Float32ToFloat32.ref.loc33(%.loc35.2) [concrete = constants.%float.520]
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
// CHECK:STDOUT:   %Core.import_ref.38a: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.02f) = import_ref Core//prelude/parts/float, loc124, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.1f0)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%Core.import_ref.38a), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.201 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f32: type = type_literal constants.%f32.97e [concrete = constants.%f32.97e]
// CHECK:STDOUT:   %.loc16.1: %f32.97e = value_of_initializer @__global_init.%Float64ToFloat32.call [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc16.2: %f32.97e = converted @__global_init.%Float64ToFloat32.call, %.loc16.1 [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %a: %f32.97e = value_binding a, %.loc16.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Float64ToFloat32.ref: %Float64ToFloat32.type = name_ref Float64ToFloat32, imports.%Main.Float64ToFloat32 [concrete = constants.%Float64ToFloat32]
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %impl.elem0: %.6c5 = impl_witness_access constants.%ImplicitAs.impl_witness.cb2, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.239]
// CHECK:STDOUT:   %bound_method.loc15.1: <bound method> = bound_method %float, %impl.elem0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15.2: <bound method> = bound_method %float, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f64.d77 = call %bound_method.loc15.2(%float) [concrete = constants.%float.d20]
// CHECK:STDOUT:   %.loc15.1: %f64.d77 = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%float.d20]
// CHECK:STDOUT:   %.loc15.2: %f64.d77 = converted %float, %.loc15.1 [concrete = constants.%float.d20]
// CHECK:STDOUT:   %Float64ToFloat32.call: init %f32.97e = call %Float64ToFloat32.ref(%.loc15.2) [concrete = constants.%float.e3b]
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
// CHECK:STDOUT:   %Core.import_ref.38a: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.02f) = import_ref Core//prelude/parts/float, loc124, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.1f0)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%Core.import_ref.38a), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.201 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f32.loc15: type = type_literal constants.%f32.97e [concrete = constants.%f32.97e]
// CHECK:STDOUT:   %.loc20.1: %f32.97e = value_of_initializer @__global_init.%Float64ToFloat32.call [concrete = <error>]
// CHECK:STDOUT:   %.loc20.2: %f32.97e = converted @__global_init.%Float64ToFloat32.call, %.loc20.1 [concrete = <error>]
// CHECK:STDOUT:   %a: %f32.97e = value_binding a, %.loc20.2 [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.201 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f32.loc25: type = type_literal constants.%f32.97e [concrete = constants.%f32.97e]
// CHECK:STDOUT:   %.loc30.1: %f32.97e = value_of_initializer @__global_init.%FloatLiteralToFloat32.call [concrete = <error>]
// CHECK:STDOUT:   %.loc30.2: %f32.97e = converted @__global_init.%FloatLiteralToFloat32.call, %.loc30.1 [concrete = <error>]
// CHECK:STDOUT:   %b: %f32.97e = value_binding b, %.loc30.2 [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.0ae = value_binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f64: type = type_literal constants.%f64.d77 [concrete = constants.%f64.d77]
// CHECK:STDOUT:   %.loc40.1: %f64.d77 = value_of_initializer @__global_init.%FloatLiteralToFloat64.call [concrete = <error>]
// CHECK:STDOUT:   %.loc40.2: %f64.d77 = converted @__global_init.%FloatLiteralToFloat64.call, %.loc40.1 [concrete = <error>]
// CHECK:STDOUT:   %c: %f64.d77 = value_binding c, %.loc40.2 [concrete = <error>]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Float64ToFloat32.ref: %Float64ToFloat32.type = name_ref Float64ToFloat32, imports.%Main.Float64ToFloat32 [concrete = constants.%Float64ToFloat32]
// CHECK:STDOUT:   %float.loc19: Core.FloatLiteral = float_literal_value 10e38 [concrete = constants.%float.cfcfd2.1]
// CHECK:STDOUT:   %impl.elem0: %.6c5 = impl_witness_access constants.%ImplicitAs.impl_witness.cb2, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.239]
// CHECK:STDOUT:   %bound_method.loc19.1: <bound method> = bound_method %float.loc19, %impl.elem0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc19.2: <bound method> = bound_method %float.loc19, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f64.d77 = call %bound_method.loc19.2(%float.loc19) [concrete = constants.%float.c37]
// CHECK:STDOUT:   %.loc19.1: %f64.d77 = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%float.c37]
// CHECK:STDOUT:   %.loc19.2: %f64.d77 = converted %float.loc19, %.loc19.1 [concrete = constants.%float.c37]
// CHECK:STDOUT:   %Float64ToFloat32.call: init %f32.97e = call %Float64ToFloat32.ref(%.loc19.2) [concrete = <error>]
// CHECK:STDOUT:   %FloatLiteralToFloat32.ref: %FloatLiteralToFloat32.type = name_ref FloatLiteralToFloat32, imports.%Main.FloatLiteralToFloat32 [concrete = constants.%FloatLiteralToFloat32]
// CHECK:STDOUT:   %float.loc29: Core.FloatLiteral = float_literal_value 10e38 [concrete = constants.%float.cfcfd2.2]
// CHECK:STDOUT:   %FloatLiteralToFloat32.call: init %f32.97e = call %FloatLiteralToFloat32.ref(%float.loc29) [concrete = <error>]
// CHECK:STDOUT:   %FloatLiteralToFloat64.ref: %FloatLiteralToFloat64.type = name_ref FloatLiteralToFloat64, imports.%Main.FloatLiteralToFloat64 [concrete = constants.%FloatLiteralToFloat64]
// CHECK:STDOUT:   %float.loc39: Core.FloatLiteral = float_literal_value 10e308 [concrete = constants.%float.82f]
// CHECK:STDOUT:   %FloatLiteralToFloat64.call: init %f64.d77 = call %FloatLiteralToFloat64.ref(%float.loc39) [concrete = <error>]
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
// CHECK:STDOUT:   %Core.import_ref.38a: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.02f) = import_ref Core//prelude/parts/float, loc124, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.1f0)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%Core.import_ref.38a), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.0ae = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f64.loc11: type = type_literal constants.%f64.d77 [concrete = constants.%f64.d77]
// CHECK:STDOUT:   %.loc16.1: %f64.d77 = value_of_initializer @__global_init.%Float32ToFloat64.call.loc16 [concrete = constants.%float.d20]
// CHECK:STDOUT:   %.loc16.2: %f64.d77 = converted @__global_init.%Float32ToFloat64.call.loc16, %.loc16.1 [concrete = constants.%float.d20]
// CHECK:STDOUT:   %a: %f64.d77 = value_binding a, %.loc16.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.0ae = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f64.loc21: type = type_literal constants.%f64.d77 [concrete = constants.%f64.d77]
// CHECK:STDOUT:   %.loc26.1: %f64.d77 = value_of_initializer @__global_init.%Float32ToFloat64.call.loc26 [concrete = constants.%float.6a7]
// CHECK:STDOUT:   %.loc26.2: %f64.d77 = converted @__global_init.%Float32ToFloat64.call.loc26, %.loc26.1 [concrete = constants.%float.6a7]
// CHECK:STDOUT:   %b: %f64.d77 = value_binding b, %.loc26.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Float32ToFloat64.ref.loc13: %Float32ToFloat64.type = name_ref Float32ToFloat64, imports.%Main.Float32ToFloat64 [concrete = constants.%Float32ToFloat64]
// CHECK:STDOUT:   %float.loc15: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %impl.elem0.loc15: %.98d = impl_witness_access constants.%ImplicitAs.impl_witness.bc6, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55]
// CHECK:STDOUT:   %bound_method.loc15.1: <bound method> = bound_method %float.loc15, %impl.elem0.loc15 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.c4b]
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem0.loc15, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15.2: <bound method> = bound_method %float.loc15, %specific_fn.loc15 [concrete = constants.%bound_method.57d]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc15: init %f32.97e = call %bound_method.loc15.2(%float.loc15) [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc15.1: %f32.97e = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc15 [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc15.2: %f32.97e = converted %float.loc15, %.loc15.1 [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %Float32ToFloat64.call.loc16: init %f64.d77 = call %Float32ToFloat64.ref.loc13(%.loc15.2) [concrete = constants.%float.d20]
// CHECK:STDOUT:   %Float32ToFloat64.ref.loc23: %Float32ToFloat64.type = name_ref Float32ToFloat64, imports.%Main.Float32ToFloat64 [concrete = constants.%Float32ToFloat64]
// CHECK:STDOUT:   %float.loc25: Core.FloatLiteral = float_literal_value 10e29 [concrete = constants.%float.c02]
// CHECK:STDOUT:   %impl.elem0.loc25: %.98d = impl_witness_access constants.%ImplicitAs.impl_witness.bc6, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55]
// CHECK:STDOUT:   %bound_method.loc25.1: <bound method> = bound_method %float.loc25, %impl.elem0.loc25 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.9ad]
// CHECK:STDOUT:   %specific_fn.loc25: <specific function> = specific_function %impl.elem0.loc25, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc25.2: <bound method> = bound_method %float.loc25, %specific_fn.loc25 [concrete = constants.%bound_method.601]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc25: init %f32.97e = call %bound_method.loc25.2(%float.loc25) [concrete = constants.%float.7d4]
// CHECK:STDOUT:   %.loc25.1: %f32.97e = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc25 [concrete = constants.%float.7d4]
// CHECK:STDOUT:   %.loc25.2: %f32.97e = converted %float.loc25, %.loc25.1 [concrete = constants.%float.7d4]
// CHECK:STDOUT:   %Float32ToFloat64.call.loc26: init %f64.d77 = call %Float32ToFloat64.ref.loc23(%.loc25.2) [concrete = constants.%float.6a7]
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
// CHECK:STDOUT:   %Core.import_ref.38a: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.02f) = import_ref Core//prelude/parts/float, loc124, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.1f0)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%Core.import_ref.38a), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %not_constant_64.patt: %pattern_type.0ae = value_binding_pattern not_constant_64 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f64.loc11: type = type_literal constants.%f64.d77 [concrete = constants.%f64.d77]
// CHECK:STDOUT:   %impl.elem0: %.6c5 = impl_witness_access constants.%ImplicitAs.impl_witness.cb2, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.239]
// CHECK:STDOUT:   %bound_method.loc13.1: <bound method> = bound_method @__global_init.%float, %impl.elem0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13.2: <bound method> = bound_method @__global_init.%float, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f64.d77 = call %bound_method.loc13.2(@__global_init.%float) [concrete = constants.%float.0a8]
// CHECK:STDOUT:   %.loc13.1: %f64.d77 = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%float.0a8]
// CHECK:STDOUT:   %.loc13.2: %f64.d77 = converted @__global_init.%float, %.loc13.1 [concrete = constants.%float.0a8]
// CHECK:STDOUT:   %not_constant_64: %f64.d77 = value_binding not_constant_64, %.loc13.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %convert_not_constant.patt: %pattern_type.0ae = value_binding_pattern convert_not_constant [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f64.loc18: type = type_literal constants.%f64.d77 [concrete = constants.%f64.d77]
// CHECK:STDOUT:   %.loc23.1: %f64.d77 = value_of_initializer @__global_init.%Float64ToFloat64.call
// CHECK:STDOUT:   %.loc23.2: %f64.d77 = converted @__global_init.%Float64ToFloat64.call, %.loc23.1
// CHECK:STDOUT:   %convert_not_constant: %f64.d77 = value_binding convert_not_constant, %.loc23.2
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

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
// CHECK:STDOUT:     %f.patt: %pattern_type.dab = binding_pattern f [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc6_21.1: type = splice_block %.loc6_21.3 [concrete = Core.FloatLiteral] {
// CHECK:STDOUT:     %FloatLiteral.ref: %FloatLiteral.type = name_ref FloatLiteral, imports.%Main.FloatLiteral [concrete = constants.%FloatLiteral]
// CHECK:STDOUT:     %FloatLiteral.call: init type = call %FloatLiteral.ref() [concrete = Core.FloatLiteral]
// CHECK:STDOUT:     %.loc6_21.2: type = value_of_initializer %FloatLiteral.call [concrete = Core.FloatLiteral]
// CHECK:STDOUT:     %.loc6_21.3: type = converted %FloatLiteral.call, %.loc6_21.2 [concrete = Core.FloatLiteral]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc6_55.1: Core.FloatLiteral = value_of_initializer @__global_init.%FloatLiteralToFloatLiteral.call [concrete = constants.%float]
// CHECK:STDOUT:   %.loc6_55.2: Core.FloatLiteral = converted @__global_init.%FloatLiteralToFloatLiteral.call, %.loc6_55.1 [concrete = constants.%float]
// CHECK:STDOUT:   %f: Core.FloatLiteral = bind_name f, %.loc6_55.2
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
// CHECK:STDOUT:   %float.674: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %float.d20: %f64.d77 = float_value 1 [concrete]
// CHECK:STDOUT:   %float.173: Core.FloatLiteral = float_literal_value 10e307 [concrete]
// CHECK:STDOUT:   %float.bde: %f64.d77 = float_value 1.0E+308 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.FloatLiteralToFloat64: %FloatLiteralToFloat64.type = import_ref Main//f64, FloatLiteralToFloat64, loaded [concrete = constants.%FloatLiteralToFloat64]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.0ae = binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc6_8: type = splice_block %f64.loc6 [concrete = constants.%f64.d77] {
// CHECK:STDOUT:     %int_64.loc6: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %f64.loc6: type = class_type @Float, @Float(constants.%int_64) [concrete = constants.%f64.d77]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc6_39.1: %f64.d77 = value_of_initializer @__global_init.%FloatLiteralToFloat64.call.loc6 [concrete = constants.%float.0a8]
// CHECK:STDOUT:   %.loc6_39.2: %f64.d77 = converted @__global_init.%FloatLiteralToFloat64.call.loc6, %.loc6_39.1 [concrete = constants.%float.0a8]
// CHECK:STDOUT:   %a: %f64.d77 = bind_name a, %.loc6_39.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.0ae = binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc7_8: type = splice_block %f64.loc7 [concrete = constants.%f64.d77] {
// CHECK:STDOUT:     %int_64.loc7: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %f64.loc7: type = class_type @Float, @Float(constants.%int_64) [concrete = constants.%f64.d77]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc7_39.1: %f64.d77 = value_of_initializer @__global_init.%FloatLiteralToFloat64.call.loc7 [concrete = constants.%float.d20]
// CHECK:STDOUT:   %.loc7_39.2: %f64.d77 = converted @__global_init.%FloatLiteralToFloat64.call.loc7, %.loc7_39.1 [concrete = constants.%float.d20]
// CHECK:STDOUT:   %b: %f64.d77 = bind_name b, %.loc7_39.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.0ae = binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_8: type = splice_block %f64.loc8 [concrete = constants.%f64.d77] {
// CHECK:STDOUT:     %int_64.loc8: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %f64.loc8: type = class_type @Float, @Float(constants.%int_64) [concrete = constants.%f64.d77]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_43.1: %f64.d77 = value_of_initializer @__global_init.%FloatLiteralToFloat64.call.loc8 [concrete = constants.%float.bde]
// CHECK:STDOUT:   %.loc8_43.2: %f64.d77 = converted @__global_init.%FloatLiteralToFloat64.call.loc8, %.loc8_43.1 [concrete = constants.%float.bde]
// CHECK:STDOUT:   %c: %f64.d77 = bind_name c, %.loc8_43.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %FloatLiteralToFloat64.ref.loc6: %FloatLiteralToFloat64.type = name_ref FloatLiteralToFloat64, imports.%Main.FloatLiteralToFloat64 [concrete = constants.%FloatLiteralToFloat64]
// CHECK:STDOUT:   %float.loc6: Core.FloatLiteral = float_literal_value 0e-1 [concrete = constants.%float.1f7]
// CHECK:STDOUT:   %FloatLiteralToFloat64.call.loc6: init %f64.d77 = call %FloatLiteralToFloat64.ref.loc6(%float.loc6) [concrete = constants.%float.0a8]
// CHECK:STDOUT:   %FloatLiteralToFloat64.ref.loc7: %FloatLiteralToFloat64.type = name_ref FloatLiteralToFloat64, imports.%Main.FloatLiteralToFloat64 [concrete = constants.%FloatLiteralToFloat64]
// CHECK:STDOUT:   %float.loc7: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.674]
// CHECK:STDOUT:   %FloatLiteralToFloat64.call.loc7: init %f64.d77 = call %FloatLiteralToFloat64.ref.loc7(%float.loc7) [concrete = constants.%float.d20]
// CHECK:STDOUT:   %FloatLiteralToFloat64.ref.loc8: %FloatLiteralToFloat64.type = name_ref FloatLiteralToFloat64, imports.%Main.FloatLiteralToFloat64 [concrete = constants.%FloatLiteralToFloat64]
// CHECK:STDOUT:   %float.loc8: Core.FloatLiteral = float_literal_value 10e307 [concrete = constants.%float.173]
// CHECK:STDOUT:   %FloatLiteralToFloat64.call.loc8: init %f64.d77 = call %FloatLiteralToFloat64.ref.loc8(%float.loc8) [concrete = constants.%float.bde]
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
// CHECK:STDOUT:   %float.674: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %float.e3b: %f32.97e = float_value 1 [concrete]
// CHECK:STDOUT:   %float.516: Core.FloatLiteral = float_literal_value 10e37 [concrete]
// CHECK:STDOUT:   %float.520: %f32.97e = float_value 9.99999968E+37 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.FloatLiteralToFloat32: %FloatLiteralToFloat32.type = import_ref Main//f32, FloatLiteralToFloat32, loaded [concrete = constants.%FloatLiteralToFloat32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.201 = binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc6_8: type = splice_block %f32.loc6 [concrete = constants.%f32.97e] {
// CHECK:STDOUT:     %int_32.loc6: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %f32.loc6: type = class_type @Float, @Float(constants.%int_32) [concrete = constants.%f32.97e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc6_39.1: %f32.97e = value_of_initializer @__global_init.%FloatLiteralToFloat32.call.loc6 [concrete = constants.%float.4db]
// CHECK:STDOUT:   %.loc6_39.2: %f32.97e = converted @__global_init.%FloatLiteralToFloat32.call.loc6, %.loc6_39.1 [concrete = constants.%float.4db]
// CHECK:STDOUT:   %a: %f32.97e = bind_name a, %.loc6_39.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.201 = binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc7_8: type = splice_block %f32.loc7 [concrete = constants.%f32.97e] {
// CHECK:STDOUT:     %int_32.loc7: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %f32.loc7: type = class_type @Float, @Float(constants.%int_32) [concrete = constants.%f32.97e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc7_39.1: %f32.97e = value_of_initializer @__global_init.%FloatLiteralToFloat32.call.loc7 [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc7_39.2: %f32.97e = converted @__global_init.%FloatLiteralToFloat32.call.loc7, %.loc7_39.1 [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %b: %f32.97e = bind_name b, %.loc7_39.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.201 = binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_8: type = splice_block %f32.loc8 [concrete = constants.%f32.97e] {
// CHECK:STDOUT:     %int_32.loc8: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %f32.loc8: type = class_type @Float, @Float(constants.%int_32) [concrete = constants.%f32.97e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_42.1: %f32.97e = value_of_initializer @__global_init.%FloatLiteralToFloat32.call.loc8 [concrete = constants.%float.520]
// CHECK:STDOUT:   %.loc8_42.2: %f32.97e = converted @__global_init.%FloatLiteralToFloat32.call.loc8, %.loc8_42.1 [concrete = constants.%float.520]
// CHECK:STDOUT:   %c: %f32.97e = bind_name c, %.loc8_42.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %FloatLiteralToFloat32.ref.loc6: %FloatLiteralToFloat32.type = name_ref FloatLiteralToFloat32, imports.%Main.FloatLiteralToFloat32 [concrete = constants.%FloatLiteralToFloat32]
// CHECK:STDOUT:   %float.loc6: Core.FloatLiteral = float_literal_value 0e-1 [concrete = constants.%float.1f7]
// CHECK:STDOUT:   %FloatLiteralToFloat32.call.loc6: init %f32.97e = call %FloatLiteralToFloat32.ref.loc6(%float.loc6) [concrete = constants.%float.4db]
// CHECK:STDOUT:   %FloatLiteralToFloat32.ref.loc7: %FloatLiteralToFloat32.type = name_ref FloatLiteralToFloat32, imports.%Main.FloatLiteralToFloat32 [concrete = constants.%FloatLiteralToFloat32]
// CHECK:STDOUT:   %float.loc7: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.674]
// CHECK:STDOUT:   %FloatLiteralToFloat32.call.loc7: init %f32.97e = call %FloatLiteralToFloat32.ref.loc7(%float.loc7) [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %FloatLiteralToFloat32.ref.loc8: %FloatLiteralToFloat32.type = name_ref FloatLiteralToFloat32, imports.%Main.FloatLiteralToFloat32 [concrete = constants.%FloatLiteralToFloat32]
// CHECK:STDOUT:   %float.loc8: Core.FloatLiteral = float_literal_value 10e37 [concrete = constants.%float.516]
// CHECK:STDOUT:   %FloatLiteralToFloat32.call.loc8: init %f32.97e = call %FloatLiteralToFloat32.ref.loc8(%float.loc8) [concrete = constants.%float.520]
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
// CHECK:STDOUT:   %ImplicitAs.Convert.type.726: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%f64.d77) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.261: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.4dd: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.261 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c09: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table, @Core.FloatLiteral.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.d68: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.052: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.d68 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.4a8 = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.c09) [concrete]
// CHECK:STDOUT:   %.eb2: type = fn_type_with_self_type %ImplicitAs.Convert.type.726, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.4de: <bound method> = bound_method %float.1f7, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.052 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.ImplicitAs.impl.Convert.052, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.4dc: <bound method> = bound_method %float.1f7, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.0a8: %f64.d77 = float_value 0 [concrete]
// CHECK:STDOUT:   %float.674: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.45b: <bound method> = bound_method %float.674, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.052 [concrete]
// CHECK:STDOUT:   %bound_method.93c: <bound method> = bound_method %float.674, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.d20: %f64.d77 = float_value 1 [concrete]
// CHECK:STDOUT:   %float.173: Core.FloatLiteral = float_literal_value 10e307 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.6de: <bound method> = bound_method %float.173, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.052 [concrete]
// CHECK:STDOUT:   %bound_method.c13: <bound method> = bound_method %float.173, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.bde: %f64.d77 = float_value 1.0E+308 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Float64ToFloat64: %Float64ToFloat64.type = import_ref Main//f64, Float64ToFloat64, loaded [concrete = constants.%Float64ToFloat64]
// CHECK:STDOUT:   %Core.import_ref.175: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.261) = import_ref Core//prelude/parts/float, loc15_41, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.4dd)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%Core.import_ref.175), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.0ae = binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc6_8: type = splice_block %f64.loc6 [concrete = constants.%f64.d77] {
// CHECK:STDOUT:     %int_64.loc6: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %f64.loc6: type = class_type @Float, @Float(constants.%int_64) [concrete = constants.%f64.d77]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc6_34.1: %f64.d77 = value_of_initializer @__global_init.%Float64ToFloat64.call.loc6 [concrete = constants.%float.0a8]
// CHECK:STDOUT:   %.loc6_34.2: %f64.d77 = converted @__global_init.%Float64ToFloat64.call.loc6, %.loc6_34.1 [concrete = constants.%float.0a8]
// CHECK:STDOUT:   %a: %f64.d77 = bind_name a, %.loc6_34.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.0ae = binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc7_8: type = splice_block %f64.loc7 [concrete = constants.%f64.d77] {
// CHECK:STDOUT:     %int_64.loc7: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %f64.loc7: type = class_type @Float, @Float(constants.%int_64) [concrete = constants.%f64.d77]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc7_34.1: %f64.d77 = value_of_initializer @__global_init.%Float64ToFloat64.call.loc7 [concrete = constants.%float.d20]
// CHECK:STDOUT:   %.loc7_34.2: %f64.d77 = converted @__global_init.%Float64ToFloat64.call.loc7, %.loc7_34.1 [concrete = constants.%float.d20]
// CHECK:STDOUT:   %b: %f64.d77 = bind_name b, %.loc7_34.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.0ae = binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_8: type = splice_block %f64.loc8 [concrete = constants.%f64.d77] {
// CHECK:STDOUT:     %int_64.loc8: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %f64.loc8: type = class_type @Float, @Float(constants.%int_64) [concrete = constants.%f64.d77]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_38.1: %f64.d77 = value_of_initializer @__global_init.%Float64ToFloat64.call.loc8 [concrete = constants.%float.bde]
// CHECK:STDOUT:   %.loc8_38.2: %f64.d77 = converted @__global_init.%Float64ToFloat64.call.loc8, %.loc8_38.1 [concrete = constants.%float.bde]
// CHECK:STDOUT:   %c: %f64.d77 = bind_name c, %.loc8_38.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Float64ToFloat64.ref.loc6: %Float64ToFloat64.type = name_ref Float64ToFloat64, imports.%Main.Float64ToFloat64 [concrete = constants.%Float64ToFloat64]
// CHECK:STDOUT:   %float.loc6: Core.FloatLiteral = float_literal_value 0e-1 [concrete = constants.%float.1f7]
// CHECK:STDOUT:   %impl.elem0.loc6: %.eb2 = impl_witness_access constants.%ImplicitAs.impl_witness.c09, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.052]
// CHECK:STDOUT:   %bound_method.loc6_31.1: <bound method> = bound_method %float.loc6, %impl.elem0.loc6 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.4de]
// CHECK:STDOUT:   %specific_fn.loc6: <specific function> = specific_function %impl.elem0.loc6, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_31.2: <bound method> = bound_method %float.loc6, %specific_fn.loc6 [concrete = constants.%bound_method.4dc]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc6: init %f64.d77 = call %bound_method.loc6_31.2(%float.loc6) [concrete = constants.%float.0a8]
// CHECK:STDOUT:   %.loc6_31.1: %f64.d77 = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc6 [concrete = constants.%float.0a8]
// CHECK:STDOUT:   %.loc6_31.2: %f64.d77 = converted %float.loc6, %.loc6_31.1 [concrete = constants.%float.0a8]
// CHECK:STDOUT:   %Float64ToFloat64.call.loc6: init %f64.d77 = call %Float64ToFloat64.ref.loc6(%.loc6_31.2) [concrete = constants.%float.0a8]
// CHECK:STDOUT:   %Float64ToFloat64.ref.loc7: %Float64ToFloat64.type = name_ref Float64ToFloat64, imports.%Main.Float64ToFloat64 [concrete = constants.%Float64ToFloat64]
// CHECK:STDOUT:   %float.loc7: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.674]
// CHECK:STDOUT:   %impl.elem0.loc7: %.eb2 = impl_witness_access constants.%ImplicitAs.impl_witness.c09, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.052]
// CHECK:STDOUT:   %bound_method.loc7_31.1: <bound method> = bound_method %float.loc7, %impl.elem0.loc7 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.45b]
// CHECK:STDOUT:   %specific_fn.loc7: <specific function> = specific_function %impl.elem0.loc7, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_31.2: <bound method> = bound_method %float.loc7, %specific_fn.loc7 [concrete = constants.%bound_method.93c]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc7: init %f64.d77 = call %bound_method.loc7_31.2(%float.loc7) [concrete = constants.%float.d20]
// CHECK:STDOUT:   %.loc7_31.1: %f64.d77 = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc7 [concrete = constants.%float.d20]
// CHECK:STDOUT:   %.loc7_31.2: %f64.d77 = converted %float.loc7, %.loc7_31.1 [concrete = constants.%float.d20]
// CHECK:STDOUT:   %Float64ToFloat64.call.loc7: init %f64.d77 = call %Float64ToFloat64.ref.loc7(%.loc7_31.2) [concrete = constants.%float.d20]
// CHECK:STDOUT:   %Float64ToFloat64.ref.loc8: %Float64ToFloat64.type = name_ref Float64ToFloat64, imports.%Main.Float64ToFloat64 [concrete = constants.%Float64ToFloat64]
// CHECK:STDOUT:   %float.loc8: Core.FloatLiteral = float_literal_value 10e307 [concrete = constants.%float.173]
// CHECK:STDOUT:   %impl.elem0.loc8: %.eb2 = impl_witness_access constants.%ImplicitAs.impl_witness.c09, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.052]
// CHECK:STDOUT:   %bound_method.loc8_31.1: <bound method> = bound_method %float.loc8, %impl.elem0.loc8 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.6de]
// CHECK:STDOUT:   %specific_fn.loc8: <specific function> = specific_function %impl.elem0.loc8, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_31.2: <bound method> = bound_method %float.loc8, %specific_fn.loc8 [concrete = constants.%bound_method.c13]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc8: init %f64.d77 = call %bound_method.loc8_31.2(%float.loc8) [concrete = constants.%float.bde]
// CHECK:STDOUT:   %.loc8_31.1: %f64.d77 = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc8 [concrete = constants.%float.bde]
// CHECK:STDOUT:   %.loc8_31.2: %f64.d77 = converted %float.loc8, %.loc8_31.1 [concrete = constants.%float.bde]
// CHECK:STDOUT:   %Float64ToFloat64.call.loc8: init %f64.d77 = call %Float64ToFloat64.ref.loc8(%.loc8_31.2) [concrete = constants.%float.bde]
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
// CHECK:STDOUT:   %ImplicitAs.Convert.type.b8c: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%f32.97e) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.261: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.4dd: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.261 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c4d: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.4bf: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.563: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.4bf = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.223 = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.c4d) [concrete]
// CHECK:STDOUT:   %.870: type = fn_type_with_self_type %ImplicitAs.Convert.type.b8c, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.927: <bound method> = bound_method %float.1f7, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.563 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.ImplicitAs.impl.Convert.563, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.bb9: <bound method> = bound_method %float.1f7, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.4db: %f32.97e = float_value 0 [concrete]
// CHECK:STDOUT:   %float.674: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.6f2: <bound method> = bound_method %float.674, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.563 [concrete]
// CHECK:STDOUT:   %bound_method.56a: <bound method> = bound_method %float.674, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.e3b: %f32.97e = float_value 1 [concrete]
// CHECK:STDOUT:   %float.516: Core.FloatLiteral = float_literal_value 10e37 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.6fb: <bound method> = bound_method %float.516, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.563 [concrete]
// CHECK:STDOUT:   %bound_method.8c9: <bound method> = bound_method %float.516, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.520: %f32.97e = float_value 9.99999968E+37 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Float32ToFloat32: %Float32ToFloat32.type = import_ref Main//f32, Float32ToFloat32, loaded [concrete = constants.%Float32ToFloat32]
// CHECK:STDOUT:   %Core.import_ref.175: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.261) = import_ref Core//prelude/parts/float, loc15_41, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.4dd)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%Core.import_ref.175), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.201 = binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc6_8: type = splice_block %f32.loc6 [concrete = constants.%f32.97e] {
// CHECK:STDOUT:     %int_32.loc6: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %f32.loc6: type = class_type @Float, @Float(constants.%int_32) [concrete = constants.%f32.97e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc6_34.1: %f32.97e = value_of_initializer @__global_init.%Float32ToFloat32.call.loc6 [concrete = constants.%float.4db]
// CHECK:STDOUT:   %.loc6_34.2: %f32.97e = converted @__global_init.%Float32ToFloat32.call.loc6, %.loc6_34.1 [concrete = constants.%float.4db]
// CHECK:STDOUT:   %a: %f32.97e = bind_name a, %.loc6_34.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.201 = binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc7_8: type = splice_block %f32.loc7 [concrete = constants.%f32.97e] {
// CHECK:STDOUT:     %int_32.loc7: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %f32.loc7: type = class_type @Float, @Float(constants.%int_32) [concrete = constants.%f32.97e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc7_34.1: %f32.97e = value_of_initializer @__global_init.%Float32ToFloat32.call.loc7 [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc7_34.2: %f32.97e = converted @__global_init.%Float32ToFloat32.call.loc7, %.loc7_34.1 [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %b: %f32.97e = bind_name b, %.loc7_34.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.201 = binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_8: type = splice_block %f32.loc8 [concrete = constants.%f32.97e] {
// CHECK:STDOUT:     %int_32.loc8: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %f32.loc8: type = class_type @Float, @Float(constants.%int_32) [concrete = constants.%f32.97e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_37.1: %f32.97e = value_of_initializer @__global_init.%Float32ToFloat32.call.loc8 [concrete = constants.%float.520]
// CHECK:STDOUT:   %.loc8_37.2: %f32.97e = converted @__global_init.%Float32ToFloat32.call.loc8, %.loc8_37.1 [concrete = constants.%float.520]
// CHECK:STDOUT:   %c: %f32.97e = bind_name c, %.loc8_37.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Float32ToFloat32.ref.loc6: %Float32ToFloat32.type = name_ref Float32ToFloat32, imports.%Main.Float32ToFloat32 [concrete = constants.%Float32ToFloat32]
// CHECK:STDOUT:   %float.loc6: Core.FloatLiteral = float_literal_value 0e-1 [concrete = constants.%float.1f7]
// CHECK:STDOUT:   %impl.elem0.loc6: %.870 = impl_witness_access constants.%ImplicitAs.impl_witness.c4d, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.563]
// CHECK:STDOUT:   %bound_method.loc6_31.1: <bound method> = bound_method %float.loc6, %impl.elem0.loc6 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.927]
// CHECK:STDOUT:   %specific_fn.loc6: <specific function> = specific_function %impl.elem0.loc6, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_31.2: <bound method> = bound_method %float.loc6, %specific_fn.loc6 [concrete = constants.%bound_method.bb9]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc6: init %f32.97e = call %bound_method.loc6_31.2(%float.loc6) [concrete = constants.%float.4db]
// CHECK:STDOUT:   %.loc6_31.1: %f32.97e = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc6 [concrete = constants.%float.4db]
// CHECK:STDOUT:   %.loc6_31.2: %f32.97e = converted %float.loc6, %.loc6_31.1 [concrete = constants.%float.4db]
// CHECK:STDOUT:   %Float32ToFloat32.call.loc6: init %f32.97e = call %Float32ToFloat32.ref.loc6(%.loc6_31.2) [concrete = constants.%float.4db]
// CHECK:STDOUT:   %Float32ToFloat32.ref.loc7: %Float32ToFloat32.type = name_ref Float32ToFloat32, imports.%Main.Float32ToFloat32 [concrete = constants.%Float32ToFloat32]
// CHECK:STDOUT:   %float.loc7: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.674]
// CHECK:STDOUT:   %impl.elem0.loc7: %.870 = impl_witness_access constants.%ImplicitAs.impl_witness.c4d, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.563]
// CHECK:STDOUT:   %bound_method.loc7_31.1: <bound method> = bound_method %float.loc7, %impl.elem0.loc7 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.6f2]
// CHECK:STDOUT:   %specific_fn.loc7: <specific function> = specific_function %impl.elem0.loc7, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_31.2: <bound method> = bound_method %float.loc7, %specific_fn.loc7 [concrete = constants.%bound_method.56a]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc7: init %f32.97e = call %bound_method.loc7_31.2(%float.loc7) [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc7_31.1: %f32.97e = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc7 [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc7_31.2: %f32.97e = converted %float.loc7, %.loc7_31.1 [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %Float32ToFloat32.call.loc7: init %f32.97e = call %Float32ToFloat32.ref.loc7(%.loc7_31.2) [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %Float32ToFloat32.ref.loc8: %Float32ToFloat32.type = name_ref Float32ToFloat32, imports.%Main.Float32ToFloat32 [concrete = constants.%Float32ToFloat32]
// CHECK:STDOUT:   %float.loc8: Core.FloatLiteral = float_literal_value 10e37 [concrete = constants.%float.516]
// CHECK:STDOUT:   %impl.elem0.loc8: %.870 = impl_witness_access constants.%ImplicitAs.impl_witness.c4d, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.563]
// CHECK:STDOUT:   %bound_method.loc8_31.1: <bound method> = bound_method %float.loc8, %impl.elem0.loc8 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.6fb]
// CHECK:STDOUT:   %specific_fn.loc8: <specific function> = specific_function %impl.elem0.loc8, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_31.2: <bound method> = bound_method %float.loc8, %specific_fn.loc8 [concrete = constants.%bound_method.8c9]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc8: init %f32.97e = call %bound_method.loc8_31.2(%float.loc8) [concrete = constants.%float.520]
// CHECK:STDOUT:   %.loc8_31.1: %f32.97e = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc8 [concrete = constants.%float.520]
// CHECK:STDOUT:   %.loc8_31.2: %f32.97e = converted %float.loc8, %.loc8_31.1 [concrete = constants.%float.520]
// CHECK:STDOUT:   %Float32ToFloat32.call.loc8: init %f32.97e = call %Float32ToFloat32.ref.loc8(%.loc8_31.2) [concrete = constants.%float.520]
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
// CHECK:STDOUT:   %float.674: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.4a8: type = facet_type <@ImplicitAs, @ImplicitAs(%f64.d77)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.726: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%f64.d77) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.261: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.4dd: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.261 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c09: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table, @Core.FloatLiteral.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.d68: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.052: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.d68 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.4a8 = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.c09) [concrete]
// CHECK:STDOUT:   %.eb2: type = fn_type_with_self_type %ImplicitAs.Convert.type.726, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %float.674, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.052 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.ImplicitAs.impl.Convert.052, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %float.674, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.d20: %f64.d77 = float_value 1 [concrete]
// CHECK:STDOUT:   %float.e3b: %f32.97e = float_value 1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Float64ToFloat32: %Float64ToFloat32.type = import_ref Main//f32, Float64ToFloat32, loaded [concrete = constants.%Float64ToFloat32]
// CHECK:STDOUT:   %Core.import_ref.175: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.261) = import_ref Core//prelude/parts/float, loc15_41, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.4dd)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%Core.import_ref.175), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.201 = binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc6_8: type = splice_block %f32 [concrete = constants.%f32.97e] {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %f32: type = class_type @Float, @Float(constants.%int_32) [concrete = constants.%f32.97e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc6_34.1: %f32.97e = value_of_initializer @__global_init.%Float64ToFloat32.call [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc6_34.2: %f32.97e = converted @__global_init.%Float64ToFloat32.call, %.loc6_34.1 [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %a: %f32.97e = bind_name a, %.loc6_34.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Float64ToFloat32.ref: %Float64ToFloat32.type = name_ref Float64ToFloat32, imports.%Main.Float64ToFloat32 [concrete = constants.%Float64ToFloat32]
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.674]
// CHECK:STDOUT:   %impl.elem0: %.eb2 = impl_witness_access constants.%ImplicitAs.impl_witness.c09, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.052]
// CHECK:STDOUT:   %bound_method.loc6_31.1: <bound method> = bound_method %float, %impl.elem0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_31.2: <bound method> = bound_method %float, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f64.d77 = call %bound_method.loc6_31.2(%float) [concrete = constants.%float.d20]
// CHECK:STDOUT:   %.loc6_31.1: %f64.d77 = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%float.d20]
// CHECK:STDOUT:   %.loc6_31.2: %f64.d77 = converted %float, %.loc6_31.1 [concrete = constants.%float.d20]
// CHECK:STDOUT:   %Float64ToFloat32.call: init %f32.97e = call %Float64ToFloat32.ref(%.loc6_31.2) [concrete = constants.%float.e3b]
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
// CHECK:STDOUT:   %float.bfd691.1: Core.FloatLiteral = float_literal_value 10e38 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.4a8: type = facet_type <@ImplicitAs, @ImplicitAs(%f64.d77)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.726: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%f64.d77) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.261: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.4dd: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.261 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c09: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table, @Core.FloatLiteral.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.d68: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.052: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.d68 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.4a8 = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.c09) [concrete]
// CHECK:STDOUT:   %.eb2: type = fn_type_with_self_type %ImplicitAs.Convert.type.726, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %float.bfd691.1, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.052 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.ImplicitAs.impl.Convert.052, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %float.bfd691.1, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.c37: %f64.d77 = float_value 9.9999999999999994E+38 [concrete]
// CHECK:STDOUT:   %FloatLiteralToFloat32.type: type = fn_type @FloatLiteralToFloat32 [concrete]
// CHECK:STDOUT:   %FloatLiteralToFloat32: %FloatLiteralToFloat32.type = struct_value () [concrete]
// CHECK:STDOUT:   %float.bfd691.2: Core.FloatLiteral = float_literal_value 10e38 [concrete]
// CHECK:STDOUT:   %FloatLiteralToFloat64.type: type = fn_type @FloatLiteralToFloat64 [concrete]
// CHECK:STDOUT:   %FloatLiteralToFloat64: %FloatLiteralToFloat64.type = struct_value () [concrete]
// CHECK:STDOUT:   %float.bb5: Core.FloatLiteral = float_literal_value 10e308 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.FloatLiteralToFloat32: %FloatLiteralToFloat32.type = import_ref Main//f32, FloatLiteralToFloat32, loaded [concrete = constants.%FloatLiteralToFloat32]
// CHECK:STDOUT:   %Main.Float64ToFloat32: %Float64ToFloat32.type = import_ref Main//f32, Float64ToFloat32, loaded [concrete = constants.%Float64ToFloat32]
// CHECK:STDOUT:   %Main.FloatLiteralToFloat64: %FloatLiteralToFloat64.type = import_ref Main//f64, FloatLiteralToFloat64, loaded [concrete = constants.%FloatLiteralToFloat64]
// CHECK:STDOUT:   %Core.import_ref.175: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.261) = import_ref Core//prelude/parts/float, loc15_41, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.4dd)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%Core.import_ref.175), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.201 = binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc11_8: type = splice_block %f32.loc11 [concrete = constants.%f32.97e] {
// CHECK:STDOUT:     %int_32.loc11: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %f32.loc11: type = class_type @Float, @Float(constants.%int_32) [concrete = constants.%f32.97e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc11_37.1: %f32.97e = value_of_initializer @__global_init.%Float64ToFloat32.call [concrete = <error>]
// CHECK:STDOUT:   %.loc11_37.2: %f32.97e = converted @__global_init.%Float64ToFloat32.call, %.loc11_37.1 [concrete = <error>]
// CHECK:STDOUT:   %a: %f32.97e = bind_name a, %.loc11_37.2 [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.201 = binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc16_8: type = splice_block %f32.loc16 [concrete = constants.%f32.97e] {
// CHECK:STDOUT:     %int_32.loc16: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %f32.loc16: type = class_type @Float, @Float(constants.%int_32) [concrete = constants.%f32.97e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc16_42.1: %f32.97e = value_of_initializer @__global_init.%FloatLiteralToFloat32.call [concrete = <error>]
// CHECK:STDOUT:   %.loc16_42.2: %f32.97e = converted @__global_init.%FloatLiteralToFloat32.call, %.loc16_42.1 [concrete = <error>]
// CHECK:STDOUT:   %b: %f32.97e = bind_name b, %.loc16_42.2 [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.0ae = binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc21_8: type = splice_block %f64 [concrete = constants.%f64.d77] {
// CHECK:STDOUT:     %int_64: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %f64: type = class_type @Float, @Float(constants.%int_64) [concrete = constants.%f64.d77]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc21_43.1: %f64.d77 = value_of_initializer @__global_init.%FloatLiteralToFloat64.call [concrete = <error>]
// CHECK:STDOUT:   %.loc21_43.2: %f64.d77 = converted @__global_init.%FloatLiteralToFloat64.call, %.loc21_43.1 [concrete = <error>]
// CHECK:STDOUT:   %c: %f64.d77 = bind_name c, %.loc21_43.2 [concrete = <error>]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Float64ToFloat32.ref: %Float64ToFloat32.type = name_ref Float64ToFloat32, imports.%Main.Float64ToFloat32 [concrete = constants.%Float64ToFloat32]
// CHECK:STDOUT:   %float.loc11: Core.FloatLiteral = float_literal_value 10e38 [concrete = constants.%float.bfd691.1]
// CHECK:STDOUT:   %impl.elem0: %.eb2 = impl_witness_access constants.%ImplicitAs.impl_witness.c09, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.052]
// CHECK:STDOUT:   %bound_method.loc11_31.1: <bound method> = bound_method %float.loc11, %impl.elem0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_31.2: <bound method> = bound_method %float.loc11, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f64.d77 = call %bound_method.loc11_31.2(%float.loc11) [concrete = constants.%float.c37]
// CHECK:STDOUT:   %.loc11_31.1: %f64.d77 = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%float.c37]
// CHECK:STDOUT:   %.loc11_31.2: %f64.d77 = converted %float.loc11, %.loc11_31.1 [concrete = constants.%float.c37]
// CHECK:STDOUT:   %Float64ToFloat32.call: init %f32.97e = call %Float64ToFloat32.ref(%.loc11_31.2) [concrete = <error>]
// CHECK:STDOUT:   %FloatLiteralToFloat32.ref: %FloatLiteralToFloat32.type = name_ref FloatLiteralToFloat32, imports.%Main.FloatLiteralToFloat32 [concrete = constants.%FloatLiteralToFloat32]
// CHECK:STDOUT:   %float.loc16: Core.FloatLiteral = float_literal_value 10e38 [concrete = constants.%float.bfd691.2]
// CHECK:STDOUT:   %FloatLiteralToFloat32.call: init %f32.97e = call %FloatLiteralToFloat32.ref(%float.loc16) [concrete = <error>]
// CHECK:STDOUT:   %FloatLiteralToFloat64.ref: %FloatLiteralToFloat64.type = name_ref FloatLiteralToFloat64, imports.%Main.FloatLiteralToFloat64 [concrete = constants.%FloatLiteralToFloat64]
// CHECK:STDOUT:   %float.loc21: Core.FloatLiteral = float_literal_value 10e308 [concrete = constants.%float.bb5]
// CHECK:STDOUT:   %FloatLiteralToFloat64.call: init %f64.d77 = call %FloatLiteralToFloat64.ref(%float.loc21) [concrete = <error>]
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
// CHECK:STDOUT:   %float.674: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.223: type = facet_type <@ImplicitAs, @ImplicitAs(%f32.97e)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.b8c: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%f32.97e) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.261: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.4dd: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.261 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c4d: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.4bf: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.563: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.4bf = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.223 = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.c4d) [concrete]
// CHECK:STDOUT:   %.870: type = fn_type_with_self_type %ImplicitAs.Convert.type.b8c, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.6f2: <bound method> = bound_method %float.674, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.563 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.ImplicitAs.impl.Convert.563, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.56a: <bound method> = bound_method %float.674, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.e3b: %f32.97e = float_value 1 [concrete]
// CHECK:STDOUT:   %float.d20: %f64.d77 = float_value 1 [concrete]
// CHECK:STDOUT:   %float.9bd: Core.FloatLiteral = float_literal_value 10e29 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.342: <bound method> = bound_method %float.9bd, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.563 [concrete]
// CHECK:STDOUT:   %bound_method.335: <bound method> = bound_method %float.9bd, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.7d4: %f32.97e = float_value 1.00000002E+30 [concrete]
// CHECK:STDOUT:   %float.6a7: %f64.d77 = float_value 1.0000000150474662E+30 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Float32ToFloat64: %Float32ToFloat64.type = import_ref Main//f32, Float32ToFloat64, loaded [concrete = constants.%Float32ToFloat64]
// CHECK:STDOUT:   %Core.import_ref.175: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.261) = import_ref Core//prelude/parts/float, loc15_41, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.4dd)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%Core.import_ref.175), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.0ae = binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc6_8: type = splice_block %f64.loc6 [concrete = constants.%f64.d77] {
// CHECK:STDOUT:     %int_64.loc6: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %f64.loc6: type = class_type @Float, @Float(constants.%int_64) [concrete = constants.%f64.d77]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc6_34.1: %f64.d77 = value_of_initializer @__global_init.%Float32ToFloat64.call.loc6 [concrete = constants.%float.d20]
// CHECK:STDOUT:   %.loc6_34.2: %f64.d77 = converted @__global_init.%Float32ToFloat64.call.loc6, %.loc6_34.1 [concrete = constants.%float.d20]
// CHECK:STDOUT:   %a: %f64.d77 = bind_name a, %.loc6_34.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.0ae = binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc7_8: type = splice_block %f64.loc7 [concrete = constants.%f64.d77] {
// CHECK:STDOUT:     %int_64.loc7: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %f64.loc7: type = class_type @Float, @Float(constants.%int_64) [concrete = constants.%f64.d77]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc7_37.1: %f64.d77 = value_of_initializer @__global_init.%Float32ToFloat64.call.loc7 [concrete = constants.%float.6a7]
// CHECK:STDOUT:   %.loc7_37.2: %f64.d77 = converted @__global_init.%Float32ToFloat64.call.loc7, %.loc7_37.1 [concrete = constants.%float.6a7]
// CHECK:STDOUT:   %b: %f64.d77 = bind_name b, %.loc7_37.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Float32ToFloat64.ref.loc6: %Float32ToFloat64.type = name_ref Float32ToFloat64, imports.%Main.Float32ToFloat64 [concrete = constants.%Float32ToFloat64]
// CHECK:STDOUT:   %float.loc6: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.674]
// CHECK:STDOUT:   %impl.elem0.loc6: %.870 = impl_witness_access constants.%ImplicitAs.impl_witness.c4d, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.563]
// CHECK:STDOUT:   %bound_method.loc6_31.1: <bound method> = bound_method %float.loc6, %impl.elem0.loc6 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.6f2]
// CHECK:STDOUT:   %specific_fn.loc6: <specific function> = specific_function %impl.elem0.loc6, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_31.2: <bound method> = bound_method %float.loc6, %specific_fn.loc6 [concrete = constants.%bound_method.56a]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc6: init %f32.97e = call %bound_method.loc6_31.2(%float.loc6) [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc6_31.1: %f32.97e = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc6 [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc6_31.2: %f32.97e = converted %float.loc6, %.loc6_31.1 [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %Float32ToFloat64.call.loc6: init %f64.d77 = call %Float32ToFloat64.ref.loc6(%.loc6_31.2) [concrete = constants.%float.d20]
// CHECK:STDOUT:   %Float32ToFloat64.ref.loc7: %Float32ToFloat64.type = name_ref Float32ToFloat64, imports.%Main.Float32ToFloat64 [concrete = constants.%Float32ToFloat64]
// CHECK:STDOUT:   %float.loc7: Core.FloatLiteral = float_literal_value 10e29 [concrete = constants.%float.9bd]
// CHECK:STDOUT:   %impl.elem0.loc7: %.870 = impl_witness_access constants.%ImplicitAs.impl_witness.c4d, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.563]
// CHECK:STDOUT:   %bound_method.loc7_31.1: <bound method> = bound_method %float.loc7, %impl.elem0.loc7 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.342]
// CHECK:STDOUT:   %specific_fn.loc7: <specific function> = specific_function %impl.elem0.loc7, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_31.2: <bound method> = bound_method %float.loc7, %specific_fn.loc7 [concrete = constants.%bound_method.335]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc7: init %f32.97e = call %bound_method.loc7_31.2(%float.loc7) [concrete = constants.%float.7d4]
// CHECK:STDOUT:   %.loc7_31.1: %f32.97e = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc7 [concrete = constants.%float.7d4]
// CHECK:STDOUT:   %.loc7_31.2: %f32.97e = converted %float.loc7, %.loc7_31.1 [concrete = constants.%float.7d4]
// CHECK:STDOUT:   %Float32ToFloat64.call.loc7: init %f64.d77 = call %Float32ToFloat64.ref.loc7(%.loc7_31.2) [concrete = constants.%float.6a7]
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
// CHECK:STDOUT:   %ImplicitAs.Convert.type.726: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%f64.d77) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.261: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.4dd: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.261 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c09: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table, @Core.FloatLiteral.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.d68: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.052: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.d68 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.4a8 = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.c09) [concrete]
// CHECK:STDOUT:   %.eb2: type = fn_type_with_self_type %ImplicitAs.Convert.type.726, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %float.1f7, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.052 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.ImplicitAs.impl.Convert.052, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %float.1f7, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.0a8: %f64.d77 = float_value 0 [concrete]
// CHECK:STDOUT:   %Float64ToFloat64.type: type = fn_type @Float64ToFloat64 [concrete]
// CHECK:STDOUT:   %Float64ToFloat64: %Float64ToFloat64.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Float64ToFloat64: %Float64ToFloat64.type = import_ref Main//f64, Float64ToFloat64, loaded [concrete = constants.%Float64ToFloat64]
// CHECK:STDOUT:   %Core.import_ref.175: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.261) = import_ref Core//prelude/parts/float, loc15_41, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.4dd)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%Core.import_ref.175), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %not_constant_64.patt: %pattern_type.0ae = binding_pattern not_constant_64 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc6_22: type = splice_block %f64.loc6 [concrete = constants.%f64.d77] {
// CHECK:STDOUT:     %int_64.loc6: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %f64.loc6: type = class_type @Float, @Float(constants.%int_64) [concrete = constants.%f64.d77]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0: %.eb2 = impl_witness_access constants.%ImplicitAs.impl_witness.c09, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.052]
// CHECK:STDOUT:   %bound_method.loc6_28.1: <bound method> = bound_method @__global_init.%float, %impl.elem0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_28.2: <bound method> = bound_method @__global_init.%float, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f64.d77 = call %bound_method.loc6_28.2(@__global_init.%float) [concrete = constants.%float.0a8]
// CHECK:STDOUT:   %.loc6_28.1: %f64.d77 = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%float.0a8]
// CHECK:STDOUT:   %.loc6_28.2: %f64.d77 = converted @__global_init.%float, %.loc6_28.1 [concrete = constants.%float.0a8]
// CHECK:STDOUT:   %not_constant_64: %f64.d77 = bind_name not_constant_64, %.loc6_28.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %convert_not_constant.patt: %pattern_type.0ae = binding_pattern convert_not_constant [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc15_27: type = splice_block %f64.loc15 [concrete = constants.%f64.d77] {
// CHECK:STDOUT:     %int_64.loc15: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %f64.loc15: type = class_type @Float, @Float(constants.%int_64) [concrete = constants.%f64.d77]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc15_65.1: %f64.d77 = value_of_initializer @__global_init.%Float64ToFloat64.call
// CHECK:STDOUT:   %.loc15_65.2: %f64.d77 = converted @__global_init.%Float64ToFloat64.call, %.loc15_65.1
// CHECK:STDOUT:   %convert_not_constant: %f64.d77 = bind_name convert_not_constant, %.loc15_65.2
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

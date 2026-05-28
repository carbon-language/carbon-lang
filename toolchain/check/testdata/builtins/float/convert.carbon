// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/float/convert.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/float/convert.carbon

// --- literal.carbon

library "literal";

fn FloatLiteral() -> type = "float_literal.make_type";

fn FloatLiteralToFloatLiteral(a: FloatLiteral()) -> FloatLiteral() =
    "float.convert";

// --- identity_literal.carbon

library "identity_literal";
import library "literal";

//@dump-sem-ir-begin
let f: FloatLiteral() = FloatLiteralToFloatLiteral(1.0);
//@dump-sem-ir-end

// --- f64.carbon

library "f64";
export import library "literal";

fn Float64ToFloat64(a: f64) -> f64 = "float.convert";
fn FloatLiteralToFloat64(a: FloatLiteral()) -> f64 = "float.convert";

// --- f32.carbon

library "f32";
export import library "literal";

fn Float32ToFloat32(a: f32) -> f32 = "float.convert";
fn FloatLiteralToFloat32(a: FloatLiteral()) -> f32 = "float.convert";

fn Float32ToFloat64(a: f32) -> f64 = "float.convert";
fn Float64ToFloat32(a: f64) -> f32 = "float.convert";

// --- literal_f64.carbon

library "literal_f64";
import library "f64";

//@dump-sem-ir-begin
let a: f64 = FloatLiteralToFloat64(0.0);
let b: f64 = FloatLiteralToFloat64(1.0);
let c: f64 = FloatLiteralToFloat64(1.0e308);
// Large value that overflows is allowed in unchecked convert (rounds to infinity)
let d: f64 = FloatLiteralToFloat64(1.0e309);
//@dump-sem-ir-end

// --- literal_f32.carbon

library "literal_f32";
import library "f32";

//@dump-sem-ir-begin
let a: f32 = FloatLiteralToFloat32(0.0);
let b: f32 = FloatLiteralToFloat32(1.0);
let c: f32 = FloatLiteralToFloat32(1.0e38);
// Large value that overflows is allowed in unchecked convert (rounds to infinity)
let d: f32 = FloatLiteralToFloat32(1.0e39);
//@dump-sem-ir-end

// --- identity_f64.carbon

library "identity_f64";
import library "f64";

//@dump-sem-ir-begin
let a: f64 = Float64ToFloat64(0.0);
let b: f64 = Float64ToFloat64(1.0);
let c: f64 = Float64ToFloat64(1.0e308);
//@dump-sem-ir-end

// --- identity_f32.carbon

library "identity_f32";
import library "f32";

//@dump-sem-ir-begin
let a: f32 = Float32ToFloat32(0.0);
let b: f32 = Float32ToFloat32(1.0);
let c: f32 = Float32ToFloat32(1.0e38);
//@dump-sem-ir-end

// --- truncate.carbon

library "truncate";
import library "f32";
import library "f64";

//@dump-sem-ir-begin
let a: f32 = Float64ToFloat32(1.0);
// Narrowing with overflow is allowed in unchecked convert (rounds to infinity)
let b: f32 = Float64ToFloat32(1.0e39);
//@dump-sem-ir-end

// --- extend.carbon

library "extend";
import library "f32";
import library "f64";

//@dump-sem-ir-begin
let a: f64 = Float32ToFloat64(1.0);
let b: f64 = Float32ToFloat64(1.0e30);
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
// CHECK:STDOUT:   %.loc6_21.1: type = splice_block %.loc6_21.3 [concrete = Core.FloatLiteral] {
// CHECK:STDOUT:     %FloatLiteral.ref: %FloatLiteral.type = name_ref FloatLiteral, imports.%Main.FloatLiteral [concrete = constants.%FloatLiteral]
// CHECK:STDOUT:     %FloatLiteral.call: init type = call %FloatLiteral.ref() [concrete = Core.FloatLiteral]
// CHECK:STDOUT:     %.loc6_21.2: type = value_of_initializer %FloatLiteral.call [concrete = Core.FloatLiteral]
// CHECK:STDOUT:     %.loc6_21.3: type = converted %FloatLiteral.call, %.loc6_21.2 [concrete = Core.FloatLiteral]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc6_55.1: Core.FloatLiteral = value_of_initializer @__global_init.%FloatLiteralToFloatLiteral.call [concrete = constants.%float]
// CHECK:STDOUT:   %.loc6_55.2: Core.FloatLiteral = converted @__global_init.%FloatLiteralToFloatLiteral.call, %.loc6_55.1 [concrete = constants.%float]
// CHECK:STDOUT:   %f: Core.FloatLiteral = value_binding f, %.loc6_55.2
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
// CHECK:STDOUT:   %f64.b55: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.f3c: type = pattern_type %f64.b55 [concrete]
// CHECK:STDOUT:   %FloatLiteralToFloat64.type: type = fn_type @FloatLiteralToFloat64 [concrete]
// CHECK:STDOUT:   %FloatLiteralToFloat64: %FloatLiteralToFloat64.type = struct_value () [concrete]
// CHECK:STDOUT:   %float.1f7: Core.FloatLiteral = float_literal_value 0e-1 [concrete]
// CHECK:STDOUT:   %float.1af: %f64.b55 = float_value 0 [concrete]
// CHECK:STDOUT:   %float.6da: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %float.642: %f64.b55 = float_value 1 [concrete]
// CHECK:STDOUT:   %float.12a: Core.FloatLiteral = float_literal_value 10e307 [concrete]
// CHECK:STDOUT:   %float.3b9: %f64.b55 = float_value 1.0E+308 [concrete]
// CHECK:STDOUT:   %float.82f: Core.FloatLiteral = float_literal_value 10e308 [concrete]
// CHECK:STDOUT:   %float.f70: %f64.b55 = float_value +Inf [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.FloatLiteralToFloat64: %FloatLiteralToFloat64.type = import_ref Main//f64, FloatLiteralToFloat64, loaded [concrete = constants.%FloatLiteralToFloat64]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.f3c = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f64.loc6: type = type_literal constants.%f64.b55 [concrete = constants.%f64.b55]
// CHECK:STDOUT:   %.loc6_39.1: %f64.b55 = value_of_initializer @__global_init.%FloatLiteralToFloat64.call.loc6 [concrete = constants.%float.1af]
// CHECK:STDOUT:   %.loc6_39.2: %f64.b55 = converted @__global_init.%FloatLiteralToFloat64.call.loc6, %.loc6_39.1 [concrete = constants.%float.1af]
// CHECK:STDOUT:   %a: %f64.b55 = value_binding a, %.loc6_39.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.f3c = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f64.loc7: type = type_literal constants.%f64.b55 [concrete = constants.%f64.b55]
// CHECK:STDOUT:   %.loc7_39.1: %f64.b55 = value_of_initializer @__global_init.%FloatLiteralToFloat64.call.loc7 [concrete = constants.%float.642]
// CHECK:STDOUT:   %.loc7_39.2: %f64.b55 = converted @__global_init.%FloatLiteralToFloat64.call.loc7, %.loc7_39.1 [concrete = constants.%float.642]
// CHECK:STDOUT:   %b: %f64.b55 = value_binding b, %.loc7_39.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.f3c = value_binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f64.loc8: type = type_literal constants.%f64.b55 [concrete = constants.%f64.b55]
// CHECK:STDOUT:   %.loc8_43.1: %f64.b55 = value_of_initializer @__global_init.%FloatLiteralToFloat64.call.loc8 [concrete = constants.%float.3b9]
// CHECK:STDOUT:   %.loc8_43.2: %f64.b55 = converted @__global_init.%FloatLiteralToFloat64.call.loc8, %.loc8_43.1 [concrete = constants.%float.3b9]
// CHECK:STDOUT:   %c: %f64.b55 = value_binding c, %.loc8_43.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.f3c = value_binding_pattern d [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f64.loc10: type = type_literal constants.%f64.b55 [concrete = constants.%f64.b55]
// CHECK:STDOUT:   %.loc10_43.1: %f64.b55 = value_of_initializer @__global_init.%FloatLiteralToFloat64.call.loc10 [concrete = constants.%float.f70]
// CHECK:STDOUT:   %.loc10_43.2: %f64.b55 = converted @__global_init.%FloatLiteralToFloat64.call.loc10, %.loc10_43.1 [concrete = constants.%float.f70]
// CHECK:STDOUT:   %d: %f64.b55 = value_binding d, %.loc10_43.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %FloatLiteralToFloat64.ref.loc6: %FloatLiteralToFloat64.type = name_ref FloatLiteralToFloat64, imports.%Main.FloatLiteralToFloat64 [concrete = constants.%FloatLiteralToFloat64]
// CHECK:STDOUT:   %float.loc6: Core.FloatLiteral = float_literal_value 0e-1 [concrete = constants.%float.1f7]
// CHECK:STDOUT:   %FloatLiteralToFloat64.call.loc6: init %f64.b55 = call %FloatLiteralToFloat64.ref.loc6(%float.loc6) [concrete = constants.%float.1af]
// CHECK:STDOUT:   %FloatLiteralToFloat64.ref.loc7: %FloatLiteralToFloat64.type = name_ref FloatLiteralToFloat64, imports.%Main.FloatLiteralToFloat64 [concrete = constants.%FloatLiteralToFloat64]
// CHECK:STDOUT:   %float.loc7: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %FloatLiteralToFloat64.call.loc7: init %f64.b55 = call %FloatLiteralToFloat64.ref.loc7(%float.loc7) [concrete = constants.%float.642]
// CHECK:STDOUT:   %FloatLiteralToFloat64.ref.loc8: %FloatLiteralToFloat64.type = name_ref FloatLiteralToFloat64, imports.%Main.FloatLiteralToFloat64 [concrete = constants.%FloatLiteralToFloat64]
// CHECK:STDOUT:   %float.loc8: Core.FloatLiteral = float_literal_value 10e307 [concrete = constants.%float.12a]
// CHECK:STDOUT:   %FloatLiteralToFloat64.call.loc8: init %f64.b55 = call %FloatLiteralToFloat64.ref.loc8(%float.loc8) [concrete = constants.%float.3b9]
// CHECK:STDOUT:   %FloatLiteralToFloat64.ref.loc10: %FloatLiteralToFloat64.type = name_ref FloatLiteralToFloat64, imports.%Main.FloatLiteralToFloat64 [concrete = constants.%FloatLiteralToFloat64]
// CHECK:STDOUT:   %float.loc10: Core.FloatLiteral = float_literal_value 10e308 [concrete = constants.%float.82f]
// CHECK:STDOUT:   %FloatLiteralToFloat64.call.loc10: init %f64.b55 = call %FloatLiteralToFloat64.ref.loc10(%float.loc10) [concrete = constants.%float.f70]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- literal_f32.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %f32.a26: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.f3b: type = pattern_type %f32.a26 [concrete]
// CHECK:STDOUT:   %FloatLiteralToFloat32.type: type = fn_type @FloatLiteralToFloat32 [concrete]
// CHECK:STDOUT:   %FloatLiteralToFloat32: %FloatLiteralToFloat32.type = struct_value () [concrete]
// CHECK:STDOUT:   %float.1f7: Core.FloatLiteral = float_literal_value 0e-1 [concrete]
// CHECK:STDOUT:   %float.e89: %f32.a26 = float_value 0 [concrete]
// CHECK:STDOUT:   %float.6da: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %float.2ec: %f32.a26 = float_value 1 [concrete]
// CHECK:STDOUT:   %float.8d2: Core.FloatLiteral = float_literal_value 10e37 [concrete]
// CHECK:STDOUT:   %float.16a: %f32.a26 = float_value 9.99999968E+37 [concrete]
// CHECK:STDOUT:   %float.cfc: Core.FloatLiteral = float_literal_value 10e38 [concrete]
// CHECK:STDOUT:   %float.81c: %f32.a26 = float_value +Inf [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.FloatLiteralToFloat32: %FloatLiteralToFloat32.type = import_ref Main//f32, FloatLiteralToFloat32, loaded [concrete = constants.%FloatLiteralToFloat32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.f3b = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f32.loc6: type = type_literal constants.%f32.a26 [concrete = constants.%f32.a26]
// CHECK:STDOUT:   %.loc6_39.1: %f32.a26 = value_of_initializer @__global_init.%FloatLiteralToFloat32.call.loc6 [concrete = constants.%float.e89]
// CHECK:STDOUT:   %.loc6_39.2: %f32.a26 = converted @__global_init.%FloatLiteralToFloat32.call.loc6, %.loc6_39.1 [concrete = constants.%float.e89]
// CHECK:STDOUT:   %a: %f32.a26 = value_binding a, %.loc6_39.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.f3b = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f32.loc7: type = type_literal constants.%f32.a26 [concrete = constants.%f32.a26]
// CHECK:STDOUT:   %.loc7_39.1: %f32.a26 = value_of_initializer @__global_init.%FloatLiteralToFloat32.call.loc7 [concrete = constants.%float.2ec]
// CHECK:STDOUT:   %.loc7_39.2: %f32.a26 = converted @__global_init.%FloatLiteralToFloat32.call.loc7, %.loc7_39.1 [concrete = constants.%float.2ec]
// CHECK:STDOUT:   %b: %f32.a26 = value_binding b, %.loc7_39.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.f3b = value_binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f32.loc8: type = type_literal constants.%f32.a26 [concrete = constants.%f32.a26]
// CHECK:STDOUT:   %.loc8_42.1: %f32.a26 = value_of_initializer @__global_init.%FloatLiteralToFloat32.call.loc8 [concrete = constants.%float.16a]
// CHECK:STDOUT:   %.loc8_42.2: %f32.a26 = converted @__global_init.%FloatLiteralToFloat32.call.loc8, %.loc8_42.1 [concrete = constants.%float.16a]
// CHECK:STDOUT:   %c: %f32.a26 = value_binding c, %.loc8_42.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.f3b = value_binding_pattern d [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f32.loc10: type = type_literal constants.%f32.a26 [concrete = constants.%f32.a26]
// CHECK:STDOUT:   %.loc10_42.1: %f32.a26 = value_of_initializer @__global_init.%FloatLiteralToFloat32.call.loc10 [concrete = constants.%float.81c]
// CHECK:STDOUT:   %.loc10_42.2: %f32.a26 = converted @__global_init.%FloatLiteralToFloat32.call.loc10, %.loc10_42.1 [concrete = constants.%float.81c]
// CHECK:STDOUT:   %d: %f32.a26 = value_binding d, %.loc10_42.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %FloatLiteralToFloat32.ref.loc6: %FloatLiteralToFloat32.type = name_ref FloatLiteralToFloat32, imports.%Main.FloatLiteralToFloat32 [concrete = constants.%FloatLiteralToFloat32]
// CHECK:STDOUT:   %float.loc6: Core.FloatLiteral = float_literal_value 0e-1 [concrete = constants.%float.1f7]
// CHECK:STDOUT:   %FloatLiteralToFloat32.call.loc6: init %f32.a26 = call %FloatLiteralToFloat32.ref.loc6(%float.loc6) [concrete = constants.%float.e89]
// CHECK:STDOUT:   %FloatLiteralToFloat32.ref.loc7: %FloatLiteralToFloat32.type = name_ref FloatLiteralToFloat32, imports.%Main.FloatLiteralToFloat32 [concrete = constants.%FloatLiteralToFloat32]
// CHECK:STDOUT:   %float.loc7: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %FloatLiteralToFloat32.call.loc7: init %f32.a26 = call %FloatLiteralToFloat32.ref.loc7(%float.loc7) [concrete = constants.%float.2ec]
// CHECK:STDOUT:   %FloatLiteralToFloat32.ref.loc8: %FloatLiteralToFloat32.type = name_ref FloatLiteralToFloat32, imports.%Main.FloatLiteralToFloat32 [concrete = constants.%FloatLiteralToFloat32]
// CHECK:STDOUT:   %float.loc8: Core.FloatLiteral = float_literal_value 10e37 [concrete = constants.%float.8d2]
// CHECK:STDOUT:   %FloatLiteralToFloat32.call.loc8: init %f32.a26 = call %FloatLiteralToFloat32.ref.loc8(%float.loc8) [concrete = constants.%float.16a]
// CHECK:STDOUT:   %FloatLiteralToFloat32.ref.loc10: %FloatLiteralToFloat32.type = name_ref FloatLiteralToFloat32, imports.%Main.FloatLiteralToFloat32 [concrete = constants.%FloatLiteralToFloat32]
// CHECK:STDOUT:   %float.loc10: Core.FloatLiteral = float_literal_value 10e38 [concrete = constants.%float.cfc]
// CHECK:STDOUT:   %FloatLiteralToFloat32.call.loc10: init %f32.a26 = call %FloatLiteralToFloat32.ref.loc10(%float.loc10) [concrete = constants.%float.81c]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- identity_f64.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %f64.b55: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.f3c: type = pattern_type %f64.b55 [concrete]
// CHECK:STDOUT:   %Float64ToFloat64.type: type = fn_type @Float64ToFloat64 [concrete]
// CHECK:STDOUT:   %Float64ToFloat64: %Float64ToFloat64.type = struct_value () [concrete]
// CHECK:STDOUT:   %float.1f7: Core.FloatLiteral = float_literal_value 0e-1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.857: type = facet_type <@ImplicitAs, @ImplicitAs(%f64.b55)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.5e6: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.539: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.5e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.83f: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table, @Core.FloatLiteral.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.5c5: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.881: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.5c5 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.857 = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.83f) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.096: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%f64.b55, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.119: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.096, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.4d4: <bound method> = bound_method %float.1f7, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.881 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.ImplicitAs.impl.Convert.881, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.a29: <bound method> = bound_method %float.1f7, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.1af: %f64.b55 = float_value 0 [concrete]
// CHECK:STDOUT:   %float.6da: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.bef: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.881 [concrete]
// CHECK:STDOUT:   %bound_method.c5a: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.642: %f64.b55 = float_value 1 [concrete]
// CHECK:STDOUT:   %float.12a: Core.FloatLiteral = float_literal_value 10e307 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.72f: <bound method> = bound_method %float.12a, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.881 [concrete]
// CHECK:STDOUT:   %bound_method.b4f: <bound method> = bound_method %float.12a, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.3b9: %f64.b55 = float_value 1.0E+308 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Float64ToFloat64: %Float64ToFloat64.type = import_ref Main//f64, Float64ToFloat64, loaded [concrete = constants.%Float64ToFloat64]
// CHECK:STDOUT:   %Core.import_ref.63c: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.5e6) = import_ref Core//prelude/parts/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.539)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%Core.import_ref.63c), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.f3c = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f64.loc6: type = type_literal constants.%f64.b55 [concrete = constants.%f64.b55]
// CHECK:STDOUT:   %.loc6_34.1: %f64.b55 = value_of_initializer @__global_init.%Float64ToFloat64.call.loc6 [concrete = constants.%float.1af]
// CHECK:STDOUT:   %.loc6_34.2: %f64.b55 = converted @__global_init.%Float64ToFloat64.call.loc6, %.loc6_34.1 [concrete = constants.%float.1af]
// CHECK:STDOUT:   %a: %f64.b55 = value_binding a, %.loc6_34.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.f3c = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f64.loc7: type = type_literal constants.%f64.b55 [concrete = constants.%f64.b55]
// CHECK:STDOUT:   %.loc7_34.1: %f64.b55 = value_of_initializer @__global_init.%Float64ToFloat64.call.loc7 [concrete = constants.%float.642]
// CHECK:STDOUT:   %.loc7_34.2: %f64.b55 = converted @__global_init.%Float64ToFloat64.call.loc7, %.loc7_34.1 [concrete = constants.%float.642]
// CHECK:STDOUT:   %b: %f64.b55 = value_binding b, %.loc7_34.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.f3c = value_binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f64.loc8: type = type_literal constants.%f64.b55 [concrete = constants.%f64.b55]
// CHECK:STDOUT:   %.loc8_38.1: %f64.b55 = value_of_initializer @__global_init.%Float64ToFloat64.call.loc8 [concrete = constants.%float.3b9]
// CHECK:STDOUT:   %.loc8_38.2: %f64.b55 = converted @__global_init.%Float64ToFloat64.call.loc8, %.loc8_38.1 [concrete = constants.%float.3b9]
// CHECK:STDOUT:   %c: %f64.b55 = value_binding c, %.loc8_38.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Float64ToFloat64.ref.loc6: %Float64ToFloat64.type = name_ref Float64ToFloat64, imports.%Main.Float64ToFloat64 [concrete = constants.%Float64ToFloat64]
// CHECK:STDOUT:   %float.loc6: Core.FloatLiteral = float_literal_value 0e-1 [concrete = constants.%float.1f7]
// CHECK:STDOUT:   %impl.elem0.loc6: %.119 = impl_witness_access constants.%ImplicitAs.impl_witness.83f, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.881]
// CHECK:STDOUT:   %bound_method.loc6_31.1: <bound method> = bound_method %float.loc6, %impl.elem0.loc6 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.4d4]
// CHECK:STDOUT:   %specific_fn.loc6: <specific function> = specific_function %impl.elem0.loc6, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_31.2: <bound method> = bound_method %float.loc6, %specific_fn.loc6 [concrete = constants.%bound_method.a29]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc6: init %f64.b55 = call %bound_method.loc6_31.2(%float.loc6) [concrete = constants.%float.1af]
// CHECK:STDOUT:   %.loc6_31.1: %f64.b55 = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc6 [concrete = constants.%float.1af]
// CHECK:STDOUT:   %.loc6_31.2: %f64.b55 = converted %float.loc6, %.loc6_31.1 [concrete = constants.%float.1af]
// CHECK:STDOUT:   %Float64ToFloat64.call.loc6: init %f64.b55 = call %Float64ToFloat64.ref.loc6(%.loc6_31.2) [concrete = constants.%float.1af]
// CHECK:STDOUT:   %Float64ToFloat64.ref.loc7: %Float64ToFloat64.type = name_ref Float64ToFloat64, imports.%Main.Float64ToFloat64 [concrete = constants.%Float64ToFloat64]
// CHECK:STDOUT:   %float.loc7: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %impl.elem0.loc7: %.119 = impl_witness_access constants.%ImplicitAs.impl_witness.83f, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.881]
// CHECK:STDOUT:   %bound_method.loc7_31.1: <bound method> = bound_method %float.loc7, %impl.elem0.loc7 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.bef]
// CHECK:STDOUT:   %specific_fn.loc7: <specific function> = specific_function %impl.elem0.loc7, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_31.2: <bound method> = bound_method %float.loc7, %specific_fn.loc7 [concrete = constants.%bound_method.c5a]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc7: init %f64.b55 = call %bound_method.loc7_31.2(%float.loc7) [concrete = constants.%float.642]
// CHECK:STDOUT:   %.loc7_31.1: %f64.b55 = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc7 [concrete = constants.%float.642]
// CHECK:STDOUT:   %.loc7_31.2: %f64.b55 = converted %float.loc7, %.loc7_31.1 [concrete = constants.%float.642]
// CHECK:STDOUT:   %Float64ToFloat64.call.loc7: init %f64.b55 = call %Float64ToFloat64.ref.loc7(%.loc7_31.2) [concrete = constants.%float.642]
// CHECK:STDOUT:   %Float64ToFloat64.ref.loc8: %Float64ToFloat64.type = name_ref Float64ToFloat64, imports.%Main.Float64ToFloat64 [concrete = constants.%Float64ToFloat64]
// CHECK:STDOUT:   %float.loc8: Core.FloatLiteral = float_literal_value 10e307 [concrete = constants.%float.12a]
// CHECK:STDOUT:   %impl.elem0.loc8: %.119 = impl_witness_access constants.%ImplicitAs.impl_witness.83f, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.881]
// CHECK:STDOUT:   %bound_method.loc8_31.1: <bound method> = bound_method %float.loc8, %impl.elem0.loc8 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.72f]
// CHECK:STDOUT:   %specific_fn.loc8: <specific function> = specific_function %impl.elem0.loc8, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_31.2: <bound method> = bound_method %float.loc8, %specific_fn.loc8 [concrete = constants.%bound_method.b4f]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc8: init %f64.b55 = call %bound_method.loc8_31.2(%float.loc8) [concrete = constants.%float.3b9]
// CHECK:STDOUT:   %.loc8_31.1: %f64.b55 = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc8 [concrete = constants.%float.3b9]
// CHECK:STDOUT:   %.loc8_31.2: %f64.b55 = converted %float.loc8, %.loc8_31.1 [concrete = constants.%float.3b9]
// CHECK:STDOUT:   %Float64ToFloat64.call.loc8: init %f64.b55 = call %Float64ToFloat64.ref.loc8(%.loc8_31.2) [concrete = constants.%float.3b9]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- identity_f32.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %f32.a26: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.f3b: type = pattern_type %f32.a26 [concrete]
// CHECK:STDOUT:   %Float32ToFloat32.type: type = fn_type @Float32ToFloat32 [concrete]
// CHECK:STDOUT:   %Float32ToFloat32: %Float32ToFloat32.type = struct_value () [concrete]
// CHECK:STDOUT:   %float.1f7: Core.FloatLiteral = float_literal_value 0e-1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.d0d: type = facet_type <@ImplicitAs, @ImplicitAs(%f32.a26)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.5e6: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.539: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.5e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.65d: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.f03: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.2cd: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.f03 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.d0d = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.65d) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.c87: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%f32.a26, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.3a8: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.c87, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.3a0: <bound method> = bound_method %float.1f7, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.2cd [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.ImplicitAs.impl.Convert.2cd, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.7ae: <bound method> = bound_method %float.1f7, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.e89: %f32.a26 = float_value 0 [concrete]
// CHECK:STDOUT:   %float.6da: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.1ac: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.2cd [concrete]
// CHECK:STDOUT:   %bound_method.aaa: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.2ec: %f32.a26 = float_value 1 [concrete]
// CHECK:STDOUT:   %float.8d2: Core.FloatLiteral = float_literal_value 10e37 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.f18: <bound method> = bound_method %float.8d2, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.2cd [concrete]
// CHECK:STDOUT:   %bound_method.137: <bound method> = bound_method %float.8d2, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.16a: %f32.a26 = float_value 9.99999968E+37 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Float32ToFloat32: %Float32ToFloat32.type = import_ref Main//f32, Float32ToFloat32, loaded [concrete = constants.%Float32ToFloat32]
// CHECK:STDOUT:   %Core.import_ref.63c: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.5e6) = import_ref Core//prelude/parts/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.539)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%Core.import_ref.63c), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.f3b = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f32.loc6: type = type_literal constants.%f32.a26 [concrete = constants.%f32.a26]
// CHECK:STDOUT:   %.loc6_34.1: %f32.a26 = value_of_initializer @__global_init.%Float32ToFloat32.call.loc6 [concrete = constants.%float.e89]
// CHECK:STDOUT:   %.loc6_34.2: %f32.a26 = converted @__global_init.%Float32ToFloat32.call.loc6, %.loc6_34.1 [concrete = constants.%float.e89]
// CHECK:STDOUT:   %a: %f32.a26 = value_binding a, %.loc6_34.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.f3b = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f32.loc7: type = type_literal constants.%f32.a26 [concrete = constants.%f32.a26]
// CHECK:STDOUT:   %.loc7_34.1: %f32.a26 = value_of_initializer @__global_init.%Float32ToFloat32.call.loc7 [concrete = constants.%float.2ec]
// CHECK:STDOUT:   %.loc7_34.2: %f32.a26 = converted @__global_init.%Float32ToFloat32.call.loc7, %.loc7_34.1 [concrete = constants.%float.2ec]
// CHECK:STDOUT:   %b: %f32.a26 = value_binding b, %.loc7_34.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.f3b = value_binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f32.loc8: type = type_literal constants.%f32.a26 [concrete = constants.%f32.a26]
// CHECK:STDOUT:   %.loc8_37.1: %f32.a26 = value_of_initializer @__global_init.%Float32ToFloat32.call.loc8 [concrete = constants.%float.16a]
// CHECK:STDOUT:   %.loc8_37.2: %f32.a26 = converted @__global_init.%Float32ToFloat32.call.loc8, %.loc8_37.1 [concrete = constants.%float.16a]
// CHECK:STDOUT:   %c: %f32.a26 = value_binding c, %.loc8_37.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Float32ToFloat32.ref.loc6: %Float32ToFloat32.type = name_ref Float32ToFloat32, imports.%Main.Float32ToFloat32 [concrete = constants.%Float32ToFloat32]
// CHECK:STDOUT:   %float.loc6: Core.FloatLiteral = float_literal_value 0e-1 [concrete = constants.%float.1f7]
// CHECK:STDOUT:   %impl.elem0.loc6: %.3a8 = impl_witness_access constants.%ImplicitAs.impl_witness.65d, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.2cd]
// CHECK:STDOUT:   %bound_method.loc6_31.1: <bound method> = bound_method %float.loc6, %impl.elem0.loc6 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.3a0]
// CHECK:STDOUT:   %specific_fn.loc6: <specific function> = specific_function %impl.elem0.loc6, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_31.2: <bound method> = bound_method %float.loc6, %specific_fn.loc6 [concrete = constants.%bound_method.7ae]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc6: init %f32.a26 = call %bound_method.loc6_31.2(%float.loc6) [concrete = constants.%float.e89]
// CHECK:STDOUT:   %.loc6_31.1: %f32.a26 = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc6 [concrete = constants.%float.e89]
// CHECK:STDOUT:   %.loc6_31.2: %f32.a26 = converted %float.loc6, %.loc6_31.1 [concrete = constants.%float.e89]
// CHECK:STDOUT:   %Float32ToFloat32.call.loc6: init %f32.a26 = call %Float32ToFloat32.ref.loc6(%.loc6_31.2) [concrete = constants.%float.e89]
// CHECK:STDOUT:   %Float32ToFloat32.ref.loc7: %Float32ToFloat32.type = name_ref Float32ToFloat32, imports.%Main.Float32ToFloat32 [concrete = constants.%Float32ToFloat32]
// CHECK:STDOUT:   %float.loc7: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %impl.elem0.loc7: %.3a8 = impl_witness_access constants.%ImplicitAs.impl_witness.65d, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.2cd]
// CHECK:STDOUT:   %bound_method.loc7_31.1: <bound method> = bound_method %float.loc7, %impl.elem0.loc7 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.1ac]
// CHECK:STDOUT:   %specific_fn.loc7: <specific function> = specific_function %impl.elem0.loc7, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_31.2: <bound method> = bound_method %float.loc7, %specific_fn.loc7 [concrete = constants.%bound_method.aaa]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc7: init %f32.a26 = call %bound_method.loc7_31.2(%float.loc7) [concrete = constants.%float.2ec]
// CHECK:STDOUT:   %.loc7_31.1: %f32.a26 = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc7 [concrete = constants.%float.2ec]
// CHECK:STDOUT:   %.loc7_31.2: %f32.a26 = converted %float.loc7, %.loc7_31.1 [concrete = constants.%float.2ec]
// CHECK:STDOUT:   %Float32ToFloat32.call.loc7: init %f32.a26 = call %Float32ToFloat32.ref.loc7(%.loc7_31.2) [concrete = constants.%float.2ec]
// CHECK:STDOUT:   %Float32ToFloat32.ref.loc8: %Float32ToFloat32.type = name_ref Float32ToFloat32, imports.%Main.Float32ToFloat32 [concrete = constants.%Float32ToFloat32]
// CHECK:STDOUT:   %float.loc8: Core.FloatLiteral = float_literal_value 10e37 [concrete = constants.%float.8d2]
// CHECK:STDOUT:   %impl.elem0.loc8: %.3a8 = impl_witness_access constants.%ImplicitAs.impl_witness.65d, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.2cd]
// CHECK:STDOUT:   %bound_method.loc8_31.1: <bound method> = bound_method %float.loc8, %impl.elem0.loc8 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.f18]
// CHECK:STDOUT:   %specific_fn.loc8: <specific function> = specific_function %impl.elem0.loc8, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_31.2: <bound method> = bound_method %float.loc8, %specific_fn.loc8 [concrete = constants.%bound_method.137]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc8: init %f32.a26 = call %bound_method.loc8_31.2(%float.loc8) [concrete = constants.%float.16a]
// CHECK:STDOUT:   %.loc8_31.1: %f32.a26 = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc8 [concrete = constants.%float.16a]
// CHECK:STDOUT:   %.loc8_31.2: %f32.a26 = converted %float.loc8, %.loc8_31.1 [concrete = constants.%float.16a]
// CHECK:STDOUT:   %Float32ToFloat32.call.loc8: init %f32.a26 = call %Float32ToFloat32.ref.loc8(%.loc8_31.2) [concrete = constants.%float.16a]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- truncate.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %f32.a26: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.f3b: type = pattern_type %f32.a26 [concrete]
// CHECK:STDOUT:   %Float64ToFloat32.type: type = fn_type @Float64ToFloat32 [concrete]
// CHECK:STDOUT:   %Float64ToFloat32: %Float64ToFloat32.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %f64.b55: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %float.6da: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.857: type = facet_type <@ImplicitAs, @ImplicitAs(%f64.b55)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.5e6: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.539: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.5e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.83f: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table, @Core.FloatLiteral.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.5c5: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.881: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.5c5 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.857 = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.83f) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.096: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%f64.b55, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.119: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.096, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.bef: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.881 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.ImplicitAs.impl.Convert.881, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.c5a: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.642: %f64.b55 = float_value 1 [concrete]
// CHECK:STDOUT:   %float.2ec: %f32.a26 = float_value 1 [concrete]
// CHECK:STDOUT:   %float.cfc: Core.FloatLiteral = float_literal_value 10e38 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.af3: <bound method> = bound_method %float.cfc, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.881 [concrete]
// CHECK:STDOUT:   %bound_method.8d2: <bound method> = bound_method %float.cfc, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.dae: %f64.b55 = float_value 9.9999999999999994E+38 [concrete]
// CHECK:STDOUT:   %float.81c: %f32.a26 = float_value +Inf [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Float64ToFloat32: %Float64ToFloat32.type = import_ref Main//f32, Float64ToFloat32, loaded [concrete = constants.%Float64ToFloat32]
// CHECK:STDOUT:   %Core.import_ref.63c: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.5e6) = import_ref Core//prelude/parts/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.539)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%Core.import_ref.63c), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.f3b = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f32.loc7: type = type_literal constants.%f32.a26 [concrete = constants.%f32.a26]
// CHECK:STDOUT:   %.loc7_34.1: %f32.a26 = value_of_initializer @__global_init.%Float64ToFloat32.call.loc7 [concrete = constants.%float.2ec]
// CHECK:STDOUT:   %.loc7_34.2: %f32.a26 = converted @__global_init.%Float64ToFloat32.call.loc7, %.loc7_34.1 [concrete = constants.%float.2ec]
// CHECK:STDOUT:   %a: %f32.a26 = value_binding a, %.loc7_34.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.f3b = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f32.loc9: type = type_literal constants.%f32.a26 [concrete = constants.%f32.a26]
// CHECK:STDOUT:   %.loc9_37.1: %f32.a26 = value_of_initializer @__global_init.%Float64ToFloat32.call.loc9 [concrete = constants.%float.81c]
// CHECK:STDOUT:   %.loc9_37.2: %f32.a26 = converted @__global_init.%Float64ToFloat32.call.loc9, %.loc9_37.1 [concrete = constants.%float.81c]
// CHECK:STDOUT:   %b: %f32.a26 = value_binding b, %.loc9_37.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Float64ToFloat32.ref.loc7: %Float64ToFloat32.type = name_ref Float64ToFloat32, imports.%Main.Float64ToFloat32 [concrete = constants.%Float64ToFloat32]
// CHECK:STDOUT:   %float.loc7: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %impl.elem0.loc7: %.119 = impl_witness_access constants.%ImplicitAs.impl_witness.83f, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.881]
// CHECK:STDOUT:   %bound_method.loc7_31.1: <bound method> = bound_method %float.loc7, %impl.elem0.loc7 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.bef]
// CHECK:STDOUT:   %specific_fn.loc7: <specific function> = specific_function %impl.elem0.loc7, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_31.2: <bound method> = bound_method %float.loc7, %specific_fn.loc7 [concrete = constants.%bound_method.c5a]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc7: init %f64.b55 = call %bound_method.loc7_31.2(%float.loc7) [concrete = constants.%float.642]
// CHECK:STDOUT:   %.loc7_31.1: %f64.b55 = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc7 [concrete = constants.%float.642]
// CHECK:STDOUT:   %.loc7_31.2: %f64.b55 = converted %float.loc7, %.loc7_31.1 [concrete = constants.%float.642]
// CHECK:STDOUT:   %Float64ToFloat32.call.loc7: init %f32.a26 = call %Float64ToFloat32.ref.loc7(%.loc7_31.2) [concrete = constants.%float.2ec]
// CHECK:STDOUT:   %Float64ToFloat32.ref.loc9: %Float64ToFloat32.type = name_ref Float64ToFloat32, imports.%Main.Float64ToFloat32 [concrete = constants.%Float64ToFloat32]
// CHECK:STDOUT:   %float.loc9: Core.FloatLiteral = float_literal_value 10e38 [concrete = constants.%float.cfc]
// CHECK:STDOUT:   %impl.elem0.loc9: %.119 = impl_witness_access constants.%ImplicitAs.impl_witness.83f, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.881]
// CHECK:STDOUT:   %bound_method.loc9_31.1: <bound method> = bound_method %float.loc9, %impl.elem0.loc9 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.af3]
// CHECK:STDOUT:   %specific_fn.loc9: <specific function> = specific_function %impl.elem0.loc9, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_31.2: <bound method> = bound_method %float.loc9, %specific_fn.loc9 [concrete = constants.%bound_method.8d2]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %f64.b55 = call %bound_method.loc9_31.2(%float.loc9) [concrete = constants.%float.dae]
// CHECK:STDOUT:   %.loc9_31.1: %f64.b55 = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%float.dae]
// CHECK:STDOUT:   %.loc9_31.2: %f64.b55 = converted %float.loc9, %.loc9_31.1 [concrete = constants.%float.dae]
// CHECK:STDOUT:   %Float64ToFloat32.call.loc9: init %f32.a26 = call %Float64ToFloat32.ref.loc9(%.loc9_31.2) [concrete = constants.%float.81c]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- extend.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %f64.b55: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.f3c: type = pattern_type %f64.b55 [concrete]
// CHECK:STDOUT:   %Float32ToFloat64.type: type = fn_type @Float32ToFloat64 [concrete]
// CHECK:STDOUT:   %Float32ToFloat64: %Float32ToFloat64.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %f32.a26: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %float.6da: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.d0d: type = facet_type <@ImplicitAs, @ImplicitAs(%f32.a26)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.5e6: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.539: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.5e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.65d: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.f03: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.2cd: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.f03 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.d0d = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.65d) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.c87: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%f32.a26, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.3a8: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.c87, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.1ac: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.2cd [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.ImplicitAs.impl.Convert.2cd, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.aaa: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.2ec: %f32.a26 = float_value 1 [concrete]
// CHECK:STDOUT:   %float.642: %f64.b55 = float_value 1 [concrete]
// CHECK:STDOUT:   %float.c02: Core.FloatLiteral = float_literal_value 10e29 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.1c5: <bound method> = bound_method %float.c02, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.2cd [concrete]
// CHECK:STDOUT:   %bound_method.dc8: <bound method> = bound_method %float.c02, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.419: %f32.a26 = float_value 1.00000002E+30 [concrete]
// CHECK:STDOUT:   %float.e6a: %f64.b55 = float_value 1.0000000150474662E+30 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Float32ToFloat64: %Float32ToFloat64.type = import_ref Main//f32, Float32ToFloat64, loaded [concrete = constants.%Float32ToFloat64]
// CHECK:STDOUT:   %Core.import_ref.63c: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.5e6) = import_ref Core//prelude/parts/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.539)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%Core.import_ref.63c), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.f3c = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f64.loc7: type = type_literal constants.%f64.b55 [concrete = constants.%f64.b55]
// CHECK:STDOUT:   %.loc7_34.1: %f64.b55 = value_of_initializer @__global_init.%Float32ToFloat64.call.loc7 [concrete = constants.%float.642]
// CHECK:STDOUT:   %.loc7_34.2: %f64.b55 = converted @__global_init.%Float32ToFloat64.call.loc7, %.loc7_34.1 [concrete = constants.%float.642]
// CHECK:STDOUT:   %a: %f64.b55 = value_binding a, %.loc7_34.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.f3c = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f64.loc8: type = type_literal constants.%f64.b55 [concrete = constants.%f64.b55]
// CHECK:STDOUT:   %.loc8_37.1: %f64.b55 = value_of_initializer @__global_init.%Float32ToFloat64.call.loc8 [concrete = constants.%float.e6a]
// CHECK:STDOUT:   %.loc8_37.2: %f64.b55 = converted @__global_init.%Float32ToFloat64.call.loc8, %.loc8_37.1 [concrete = constants.%float.e6a]
// CHECK:STDOUT:   %b: %f64.b55 = value_binding b, %.loc8_37.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Float32ToFloat64.ref.loc7: %Float32ToFloat64.type = name_ref Float32ToFloat64, imports.%Main.Float32ToFloat64 [concrete = constants.%Float32ToFloat64]
// CHECK:STDOUT:   %float.loc7: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %impl.elem0.loc7: %.3a8 = impl_witness_access constants.%ImplicitAs.impl_witness.65d, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.2cd]
// CHECK:STDOUT:   %bound_method.loc7_31.1: <bound method> = bound_method %float.loc7, %impl.elem0.loc7 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.1ac]
// CHECK:STDOUT:   %specific_fn.loc7: <specific function> = specific_function %impl.elem0.loc7, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_31.2: <bound method> = bound_method %float.loc7, %specific_fn.loc7 [concrete = constants.%bound_method.aaa]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc7: init %f32.a26 = call %bound_method.loc7_31.2(%float.loc7) [concrete = constants.%float.2ec]
// CHECK:STDOUT:   %.loc7_31.1: %f32.a26 = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc7 [concrete = constants.%float.2ec]
// CHECK:STDOUT:   %.loc7_31.2: %f32.a26 = converted %float.loc7, %.loc7_31.1 [concrete = constants.%float.2ec]
// CHECK:STDOUT:   %Float32ToFloat64.call.loc7: init %f64.b55 = call %Float32ToFloat64.ref.loc7(%.loc7_31.2) [concrete = constants.%float.642]
// CHECK:STDOUT:   %Float32ToFloat64.ref.loc8: %Float32ToFloat64.type = name_ref Float32ToFloat64, imports.%Main.Float32ToFloat64 [concrete = constants.%Float32ToFloat64]
// CHECK:STDOUT:   %float.loc8: Core.FloatLiteral = float_literal_value 10e29 [concrete = constants.%float.c02]
// CHECK:STDOUT:   %impl.elem0.loc8: %.3a8 = impl_witness_access constants.%ImplicitAs.impl_witness.65d, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.2cd]
// CHECK:STDOUT:   %bound_method.loc8_31.1: <bound method> = bound_method %float.loc8, %impl.elem0.loc8 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.1c5]
// CHECK:STDOUT:   %specific_fn.loc8: <specific function> = specific_function %impl.elem0.loc8, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_31.2: <bound method> = bound_method %float.loc8, %specific_fn.loc8 [concrete = constants.%bound_method.dc8]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc8: init %f32.a26 = call %bound_method.loc8_31.2(%float.loc8) [concrete = constants.%float.419]
// CHECK:STDOUT:   %.loc8_31.1: %f32.a26 = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc8 [concrete = constants.%float.419]
// CHECK:STDOUT:   %.loc8_31.2: %f32.a26 = converted %float.loc8, %.loc8_31.1 [concrete = constants.%float.419]
// CHECK:STDOUT:   %Float32ToFloat64.call.loc8: init %f64.b55 = call %Float32ToFloat64.ref.loc8(%.loc8_31.2) [concrete = constants.%float.e6a]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:

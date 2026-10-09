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

fn MakeFloatLiteral() -> type = "float_literal.make_type";
alias FloatLiteral = MakeFloatLiteral();

fn FloatLiteralToFloatLiteral(a: FloatLiteral) -> FloatLiteral =
    "float.convert";

// --- identity_literal.carbon

library "identity_literal";
import library "literal";

//@dump-sem-ir-begin
let f: FloatLiteral = FloatLiteralToFloatLiteral(1.0);
//@dump-sem-ir-end

// --- f64.carbon

library "f64";
export import library "literal";

fn Float64ToFloat64(a: f64) -> f64 = "float.convert";
fn FloatLiteralToFloat64(a: FloatLiteral) -> f64 = "float.convert";

// --- f32.carbon

library "f32";
export import library "literal";

fn Float32ToFloat32(a: f32) -> f32 = "float.convert";
fn FloatLiteralToFloat32(a: FloatLiteral) -> f32 = "float.convert";

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
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type Core.FloatLiteral [concrete]
// CHECK:STDOUT:   %f.patt: %pattern_type = value_binding_pattern f [concrete]
// CHECK:STDOUT:   %FloatLiteralToFloatLiteral.type: type = fn_type @FloatLiteralToFloatLiteral [concrete]
// CHECK:STDOUT:   %FloatLiteralToFloatLiteral: %FloatLiteralToFloatLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.FloatLiteral: type = import_ref Main//literal, FloatLiteral, loaded [concrete = Core.FloatLiteral]
// CHECK:STDOUT:   %Main.FloatLiteralToFloatLiteral: %FloatLiteralToFloatLiteral.type = import_ref Main//literal, FloatLiteralToFloatLiteral, loaded [concrete = constants.%FloatLiteralToFloatLiteral]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %.loc6_53.1: Core.FloatLiteral = value_of_initializer @__global_init.%FloatLiteralToFloatLiteral.call [concrete = constants.%float]
// CHECK:STDOUT:   %.loc6_53.2: Core.FloatLiteral = converted @__global_init.%FloatLiteralToFloatLiteral.call, %.loc6_53.1 [concrete = constants.%float]
// CHECK:STDOUT:   %FloatLiteral.ref: type = name_ref FloatLiteral, imports.%Main.FloatLiteral [concrete = Core.FloatLiteral]
// CHECK:STDOUT:   %f: Core.FloatLiteral = wrapper_binding f, %.loc6_53.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %f.patt: %pattern_type = value_binding_pattern f [concrete = constants.%f.patt]
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %f64.84c: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.39b: type = pattern_type %f64.84c [concrete]
// CHECK:STDOUT:   %a.patt.b3a: %pattern_type.39b = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %FloatLiteralToFloat64.type: type = fn_type @FloatLiteralToFloat64 [concrete]
// CHECK:STDOUT:   %FloatLiteralToFloat64: %FloatLiteralToFloat64.type = struct_value () [concrete]
// CHECK:STDOUT:   %float.1f7: Core.FloatLiteral = float_literal_value 0e-1 [concrete]
// CHECK:STDOUT:   %float.484: %f64.84c = float_value 0 [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.39b = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %float.6da: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %float.b4b: %f64.84c = float_value 1 [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.39b = value_binding_pattern c [concrete]
// CHECK:STDOUT:   %float.12a: Core.FloatLiteral = float_literal_value 10e307 [concrete]
// CHECK:STDOUT:   %float.3a0: %f64.84c = float_value 1.0E+308 [concrete]
// CHECK:STDOUT:   %d.patt: %pattern_type.39b = value_binding_pattern d [concrete]
// CHECK:STDOUT:   %float.82f: Core.FloatLiteral = float_literal_value 10e308 [concrete]
// CHECK:STDOUT:   %float.5ea: %f64.84c = float_value +Inf [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.FloatLiteralToFloat64: %FloatLiteralToFloat64.type = import_ref Main//f64, FloatLiteralToFloat64, loaded [concrete = constants.%FloatLiteralToFloat64]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %.loc6_39.1: %f64.84c = value_of_initializer @__global_init.%FloatLiteralToFloat64.call.loc6 [concrete = constants.%float.484]
// CHECK:STDOUT:   %.loc6_39.2: %f64.84c = converted @__global_init.%FloatLiteralToFloat64.call.loc6, %.loc6_39.1 [concrete = constants.%float.484]
// CHECK:STDOUT:   %f64.loc6: type = type_literal constants.%f64.84c [concrete = constants.%f64.84c]
// CHECK:STDOUT:   %a: %f64.84c = wrapper_binding a, %.loc6_39.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.39b = value_binding_pattern a [concrete = constants.%a.patt.b3a]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc7_39.1: %f64.84c = value_of_initializer @__global_init.%FloatLiteralToFloat64.call.loc7 [concrete = constants.%float.b4b]
// CHECK:STDOUT:   %.loc7_39.2: %f64.84c = converted @__global_init.%FloatLiteralToFloat64.call.loc7, %.loc7_39.1 [concrete = constants.%float.b4b]
// CHECK:STDOUT:   %f64.loc7: type = type_literal constants.%f64.84c [concrete = constants.%f64.84c]
// CHECK:STDOUT:   %b: %f64.84c = wrapper_binding b, %.loc7_39.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.39b = value_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_43.1: %f64.84c = value_of_initializer @__global_init.%FloatLiteralToFloat64.call.loc8 [concrete = constants.%float.3a0]
// CHECK:STDOUT:   %.loc8_43.2: %f64.84c = converted @__global_init.%FloatLiteralToFloat64.call.loc8, %.loc8_43.1 [concrete = constants.%float.3a0]
// CHECK:STDOUT:   %f64.loc8: type = type_literal constants.%f64.84c [concrete = constants.%f64.84c]
// CHECK:STDOUT:   %c: %f64.84c = wrapper_binding c, %.loc8_43.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.39b = value_binding_pattern c [concrete = constants.%c.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10_43.1: %f64.84c = value_of_initializer @__global_init.%FloatLiteralToFloat64.call.loc10 [concrete = constants.%float.5ea]
// CHECK:STDOUT:   %.loc10_43.2: %f64.84c = converted @__global_init.%FloatLiteralToFloat64.call.loc10, %.loc10_43.1 [concrete = constants.%float.5ea]
// CHECK:STDOUT:   %f64.loc10: type = type_literal constants.%f64.84c [concrete = constants.%f64.84c]
// CHECK:STDOUT:   %d: %f64.84c = wrapper_binding d, %.loc10_43.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.39b = value_binding_pattern d [concrete = constants.%d.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %FloatLiteralToFloat64.ref.loc6: %FloatLiteralToFloat64.type = name_ref FloatLiteralToFloat64, imports.%Main.FloatLiteralToFloat64 [concrete = constants.%FloatLiteralToFloat64]
// CHECK:STDOUT:   %float.loc6: Core.FloatLiteral = float_literal_value 0e-1 [concrete = constants.%float.1f7]
// CHECK:STDOUT:   %FloatLiteralToFloat64.call.loc6: init %f64.84c = call %FloatLiteralToFloat64.ref.loc6(%float.loc6) [concrete = constants.%float.484]
// CHECK:STDOUT:   %FloatLiteralToFloat64.ref.loc7: %FloatLiteralToFloat64.type = name_ref FloatLiteralToFloat64, imports.%Main.FloatLiteralToFloat64 [concrete = constants.%FloatLiteralToFloat64]
// CHECK:STDOUT:   %float.loc7: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %FloatLiteralToFloat64.call.loc7: init %f64.84c = call %FloatLiteralToFloat64.ref.loc7(%float.loc7) [concrete = constants.%float.b4b]
// CHECK:STDOUT:   %FloatLiteralToFloat64.ref.loc8: %FloatLiteralToFloat64.type = name_ref FloatLiteralToFloat64, imports.%Main.FloatLiteralToFloat64 [concrete = constants.%FloatLiteralToFloat64]
// CHECK:STDOUT:   %float.loc8: Core.FloatLiteral = float_literal_value 10e307 [concrete = constants.%float.12a]
// CHECK:STDOUT:   %FloatLiteralToFloat64.call.loc8: init %f64.84c = call %FloatLiteralToFloat64.ref.loc8(%float.loc8) [concrete = constants.%float.3a0]
// CHECK:STDOUT:   %FloatLiteralToFloat64.ref.loc10: %FloatLiteralToFloat64.type = name_ref FloatLiteralToFloat64, imports.%Main.FloatLiteralToFloat64 [concrete = constants.%FloatLiteralToFloat64]
// CHECK:STDOUT:   %float.loc10: Core.FloatLiteral = float_literal_value 10e308 [concrete = constants.%float.82f]
// CHECK:STDOUT:   %FloatLiteralToFloat64.call.loc10: init %f64.84c = call %FloatLiteralToFloat64.ref.loc10(%float.loc10) [concrete = constants.%float.5ea]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- literal_f32.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %f32.2d9: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.afb: type = pattern_type %f32.2d9 [concrete]
// CHECK:STDOUT:   %a.patt.80d: %pattern_type.afb = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %FloatLiteralToFloat32.type: type = fn_type @FloatLiteralToFloat32 [concrete]
// CHECK:STDOUT:   %FloatLiteralToFloat32: %FloatLiteralToFloat32.type = struct_value () [concrete]
// CHECK:STDOUT:   %float.1f7: Core.FloatLiteral = float_literal_value 0e-1 [concrete]
// CHECK:STDOUT:   %float.ece: %f32.2d9 = float_value 0 [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.afb = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %float.6da: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %float.3a8: %f32.2d9 = float_value 1 [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.afb = value_binding_pattern c [concrete]
// CHECK:STDOUT:   %float.8d2: Core.FloatLiteral = float_literal_value 10e37 [concrete]
// CHECK:STDOUT:   %float.0e0: %f32.2d9 = float_value 9.99999968E+37 [concrete]
// CHECK:STDOUT:   %d.patt: %pattern_type.afb = value_binding_pattern d [concrete]
// CHECK:STDOUT:   %float.cfc: Core.FloatLiteral = float_literal_value 10e38 [concrete]
// CHECK:STDOUT:   %float.d2f: %f32.2d9 = float_value +Inf [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.FloatLiteralToFloat32: %FloatLiteralToFloat32.type = import_ref Main//f32, FloatLiteralToFloat32, loaded [concrete = constants.%FloatLiteralToFloat32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %.loc6_39.1: %f32.2d9 = value_of_initializer @__global_init.%FloatLiteralToFloat32.call.loc6 [concrete = constants.%float.ece]
// CHECK:STDOUT:   %.loc6_39.2: %f32.2d9 = converted @__global_init.%FloatLiteralToFloat32.call.loc6, %.loc6_39.1 [concrete = constants.%float.ece]
// CHECK:STDOUT:   %f32.loc6: type = type_literal constants.%f32.2d9 [concrete = constants.%f32.2d9]
// CHECK:STDOUT:   %a: %f32.2d9 = wrapper_binding a, %.loc6_39.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.afb = value_binding_pattern a [concrete = constants.%a.patt.80d]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc7_39.1: %f32.2d9 = value_of_initializer @__global_init.%FloatLiteralToFloat32.call.loc7 [concrete = constants.%float.3a8]
// CHECK:STDOUT:   %.loc7_39.2: %f32.2d9 = converted @__global_init.%FloatLiteralToFloat32.call.loc7, %.loc7_39.1 [concrete = constants.%float.3a8]
// CHECK:STDOUT:   %f32.loc7: type = type_literal constants.%f32.2d9 [concrete = constants.%f32.2d9]
// CHECK:STDOUT:   %b: %f32.2d9 = wrapper_binding b, %.loc7_39.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.afb = value_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_42.1: %f32.2d9 = value_of_initializer @__global_init.%FloatLiteralToFloat32.call.loc8 [concrete = constants.%float.0e0]
// CHECK:STDOUT:   %.loc8_42.2: %f32.2d9 = converted @__global_init.%FloatLiteralToFloat32.call.loc8, %.loc8_42.1 [concrete = constants.%float.0e0]
// CHECK:STDOUT:   %f32.loc8: type = type_literal constants.%f32.2d9 [concrete = constants.%f32.2d9]
// CHECK:STDOUT:   %c: %f32.2d9 = wrapper_binding c, %.loc8_42.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.afb = value_binding_pattern c [concrete = constants.%c.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10_42.1: %f32.2d9 = value_of_initializer @__global_init.%FloatLiteralToFloat32.call.loc10 [concrete = constants.%float.d2f]
// CHECK:STDOUT:   %.loc10_42.2: %f32.2d9 = converted @__global_init.%FloatLiteralToFloat32.call.loc10, %.loc10_42.1 [concrete = constants.%float.d2f]
// CHECK:STDOUT:   %f32.loc10: type = type_literal constants.%f32.2d9 [concrete = constants.%f32.2d9]
// CHECK:STDOUT:   %d: %f32.2d9 = wrapper_binding d, %.loc10_42.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.afb = value_binding_pattern d [concrete = constants.%d.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %FloatLiteralToFloat32.ref.loc6: %FloatLiteralToFloat32.type = name_ref FloatLiteralToFloat32, imports.%Main.FloatLiteralToFloat32 [concrete = constants.%FloatLiteralToFloat32]
// CHECK:STDOUT:   %float.loc6: Core.FloatLiteral = float_literal_value 0e-1 [concrete = constants.%float.1f7]
// CHECK:STDOUT:   %FloatLiteralToFloat32.call.loc6: init %f32.2d9 = call %FloatLiteralToFloat32.ref.loc6(%float.loc6) [concrete = constants.%float.ece]
// CHECK:STDOUT:   %FloatLiteralToFloat32.ref.loc7: %FloatLiteralToFloat32.type = name_ref FloatLiteralToFloat32, imports.%Main.FloatLiteralToFloat32 [concrete = constants.%FloatLiteralToFloat32]
// CHECK:STDOUT:   %float.loc7: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %FloatLiteralToFloat32.call.loc7: init %f32.2d9 = call %FloatLiteralToFloat32.ref.loc7(%float.loc7) [concrete = constants.%float.3a8]
// CHECK:STDOUT:   %FloatLiteralToFloat32.ref.loc8: %FloatLiteralToFloat32.type = name_ref FloatLiteralToFloat32, imports.%Main.FloatLiteralToFloat32 [concrete = constants.%FloatLiteralToFloat32]
// CHECK:STDOUT:   %float.loc8: Core.FloatLiteral = float_literal_value 10e37 [concrete = constants.%float.8d2]
// CHECK:STDOUT:   %FloatLiteralToFloat32.call.loc8: init %f32.2d9 = call %FloatLiteralToFloat32.ref.loc8(%float.loc8) [concrete = constants.%float.0e0]
// CHECK:STDOUT:   %FloatLiteralToFloat32.ref.loc10: %FloatLiteralToFloat32.type = name_ref FloatLiteralToFloat32, imports.%Main.FloatLiteralToFloat32 [concrete = constants.%FloatLiteralToFloat32]
// CHECK:STDOUT:   %float.loc10: Core.FloatLiteral = float_literal_value 10e38 [concrete = constants.%float.cfc]
// CHECK:STDOUT:   %FloatLiteralToFloat32.call.loc10: init %f32.2d9 = call %FloatLiteralToFloat32.ref.loc10(%float.loc10) [concrete = constants.%float.d2f]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- identity_f64.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %f64.84c: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.39b: type = pattern_type %f64.84c [concrete]
// CHECK:STDOUT:   %a.patt.b3a: %pattern_type.39b = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %Float64ToFloat64.type: type = fn_type @Float64ToFloat64 [concrete]
// CHECK:STDOUT:   %Float64ToFloat64: %Float64ToFloat64.type = struct_value () [concrete]
// CHECK:STDOUT:   %float.1f7: Core.FloatLiteral = float_literal_value 0e-1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.838: type = facet_type <@ImplicitAs, @ImplicitAs(%f64.84c)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.ca8: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bb2: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.ca8 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.afc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table, @Core.FloatLiteral.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.228: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.7ae: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.228 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.838 = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.afc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.01d: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%f64.84c, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.fa7: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.01d, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.98c: <bound method> = bound_method %float.1f7, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.7ae [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.ImplicitAs.impl.Convert.7ae, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.d82: <bound method> = bound_method %float.1f7, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.484: %f64.84c = float_value 0 [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.39b = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %float.6da: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.6da: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.7ae [concrete]
// CHECK:STDOUT:   %bound_method.066: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.b4b: %f64.84c = float_value 1 [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.39b = value_binding_pattern c [concrete]
// CHECK:STDOUT:   %float.12a: Core.FloatLiteral = float_literal_value 10e307 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.6cf: <bound method> = bound_method %float.12a, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.7ae [concrete]
// CHECK:STDOUT:   %bound_method.ad6: <bound method> = bound_method %float.12a, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.3a0: %f64.84c = float_value 1.0E+308 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Float64ToFloat64: %Float64ToFloat64.type = import_ref Main//f64, Float64ToFloat64, loaded [concrete = constants.%Float64ToFloat64]
// CHECK:STDOUT:   %Core.import_ref.f33: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.ca8) = import_ref Core//prelude/parts/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bb2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%Core.import_ref.f33), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %.loc6_34.1: %f64.84c = value_of_initializer @__global_init.%Float64ToFloat64.call.loc6 [concrete = constants.%float.484]
// CHECK:STDOUT:   %.loc6_34.2: %f64.84c = converted @__global_init.%Float64ToFloat64.call.loc6, %.loc6_34.1 [concrete = constants.%float.484]
// CHECK:STDOUT:   %f64.loc6: type = type_literal constants.%f64.84c [concrete = constants.%f64.84c]
// CHECK:STDOUT:   %a: %f64.84c = wrapper_binding a, %.loc6_34.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.39b = value_binding_pattern a [concrete = constants.%a.patt.b3a]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc7_34.1: %f64.84c = value_of_initializer @__global_init.%Float64ToFloat64.call.loc7 [concrete = constants.%float.b4b]
// CHECK:STDOUT:   %.loc7_34.2: %f64.84c = converted @__global_init.%Float64ToFloat64.call.loc7, %.loc7_34.1 [concrete = constants.%float.b4b]
// CHECK:STDOUT:   %f64.loc7: type = type_literal constants.%f64.84c [concrete = constants.%f64.84c]
// CHECK:STDOUT:   %b: %f64.84c = wrapper_binding b, %.loc7_34.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.39b = value_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_38.1: %f64.84c = value_of_initializer @__global_init.%Float64ToFloat64.call.loc8 [concrete = constants.%float.3a0]
// CHECK:STDOUT:   %.loc8_38.2: %f64.84c = converted @__global_init.%Float64ToFloat64.call.loc8, %.loc8_38.1 [concrete = constants.%float.3a0]
// CHECK:STDOUT:   %f64.loc8: type = type_literal constants.%f64.84c [concrete = constants.%f64.84c]
// CHECK:STDOUT:   %c: %f64.84c = wrapper_binding c, %.loc8_38.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.39b = value_binding_pattern c [concrete = constants.%c.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Float64ToFloat64.ref.loc6: %Float64ToFloat64.type = name_ref Float64ToFloat64, imports.%Main.Float64ToFloat64 [concrete = constants.%Float64ToFloat64]
// CHECK:STDOUT:   %float.loc6: Core.FloatLiteral = float_literal_value 0e-1 [concrete = constants.%float.1f7]
// CHECK:STDOUT:   %impl.elem0.loc6: %.fa7 = impl_witness_access constants.%ImplicitAs.impl_witness.afc, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.7ae]
// CHECK:STDOUT:   %bound_method.loc6_31.1: <bound method> = bound_method %float.loc6, %impl.elem0.loc6 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.98c]
// CHECK:STDOUT:   %specific_fn.loc6: <specific function> = specific_function %impl.elem0.loc6, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_31.2: <bound method> = bound_method %float.loc6, %specific_fn.loc6 [concrete = constants.%bound_method.d82]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc6: init %f64.84c = call %bound_method.loc6_31.2(%float.loc6) [concrete = constants.%float.484]
// CHECK:STDOUT:   %.loc6_31.1: %f64.84c = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc6 [concrete = constants.%float.484]
// CHECK:STDOUT:   %.loc6_31.2: %f64.84c = converted %float.loc6, %.loc6_31.1 [concrete = constants.%float.484]
// CHECK:STDOUT:   %Float64ToFloat64.call.loc6: init %f64.84c = call %Float64ToFloat64.ref.loc6(%.loc6_31.2) [concrete = constants.%float.484]
// CHECK:STDOUT:   %Float64ToFloat64.ref.loc7: %Float64ToFloat64.type = name_ref Float64ToFloat64, imports.%Main.Float64ToFloat64 [concrete = constants.%Float64ToFloat64]
// CHECK:STDOUT:   %float.loc7: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %impl.elem0.loc7: %.fa7 = impl_witness_access constants.%ImplicitAs.impl_witness.afc, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.7ae]
// CHECK:STDOUT:   %bound_method.loc7_31.1: <bound method> = bound_method %float.loc7, %impl.elem0.loc7 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.6da]
// CHECK:STDOUT:   %specific_fn.loc7: <specific function> = specific_function %impl.elem0.loc7, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_31.2: <bound method> = bound_method %float.loc7, %specific_fn.loc7 [concrete = constants.%bound_method.066]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc7: init %f64.84c = call %bound_method.loc7_31.2(%float.loc7) [concrete = constants.%float.b4b]
// CHECK:STDOUT:   %.loc7_31.1: %f64.84c = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc7 [concrete = constants.%float.b4b]
// CHECK:STDOUT:   %.loc7_31.2: %f64.84c = converted %float.loc7, %.loc7_31.1 [concrete = constants.%float.b4b]
// CHECK:STDOUT:   %Float64ToFloat64.call.loc7: init %f64.84c = call %Float64ToFloat64.ref.loc7(%.loc7_31.2) [concrete = constants.%float.b4b]
// CHECK:STDOUT:   %Float64ToFloat64.ref.loc8: %Float64ToFloat64.type = name_ref Float64ToFloat64, imports.%Main.Float64ToFloat64 [concrete = constants.%Float64ToFloat64]
// CHECK:STDOUT:   %float.loc8: Core.FloatLiteral = float_literal_value 10e307 [concrete = constants.%float.12a]
// CHECK:STDOUT:   %impl.elem0.loc8: %.fa7 = impl_witness_access constants.%ImplicitAs.impl_witness.afc, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.7ae]
// CHECK:STDOUT:   %bound_method.loc8_31.1: <bound method> = bound_method %float.loc8, %impl.elem0.loc8 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.6cf]
// CHECK:STDOUT:   %specific_fn.loc8: <specific function> = specific_function %impl.elem0.loc8, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_31.2: <bound method> = bound_method %float.loc8, %specific_fn.loc8 [concrete = constants.%bound_method.ad6]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc8: init %f64.84c = call %bound_method.loc8_31.2(%float.loc8) [concrete = constants.%float.3a0]
// CHECK:STDOUT:   %.loc8_31.1: %f64.84c = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc8 [concrete = constants.%float.3a0]
// CHECK:STDOUT:   %.loc8_31.2: %f64.84c = converted %float.loc8, %.loc8_31.1 [concrete = constants.%float.3a0]
// CHECK:STDOUT:   %Float64ToFloat64.call.loc8: init %f64.84c = call %Float64ToFloat64.ref.loc8(%.loc8_31.2) [concrete = constants.%float.3a0]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- identity_f32.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %f32.2d9: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.afb: type = pattern_type %f32.2d9 [concrete]
// CHECK:STDOUT:   %a.patt.80d: %pattern_type.afb = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %Float32ToFloat32.type: type = fn_type @Float32ToFloat32 [concrete]
// CHECK:STDOUT:   %Float32ToFloat32: %Float32ToFloat32.type = struct_value () [concrete]
// CHECK:STDOUT:   %float.1f7: Core.FloatLiteral = float_literal_value 0e-1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.484: type = facet_type <@ImplicitAs, @ImplicitAs(%f32.2d9)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.ca8: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bb2: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.ca8 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a1a: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.a18: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.089: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.a18 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.484 = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.a1a) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.057: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%f32.2d9, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.637: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.057, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.5ea: <bound method> = bound_method %float.1f7, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.089 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.ImplicitAs.impl.Convert.089, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.663: <bound method> = bound_method %float.1f7, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.ece: %f32.2d9 = float_value 0 [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.afb = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %float.6da: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.4c1: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.089 [concrete]
// CHECK:STDOUT:   %bound_method.29b: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.3a8: %f32.2d9 = float_value 1 [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.afb = value_binding_pattern c [concrete]
// CHECK:STDOUT:   %float.8d2: Core.FloatLiteral = float_literal_value 10e37 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.b07: <bound method> = bound_method %float.8d2, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.089 [concrete]
// CHECK:STDOUT:   %bound_method.10f: <bound method> = bound_method %float.8d2, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.0e0: %f32.2d9 = float_value 9.99999968E+37 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Float32ToFloat32: %Float32ToFloat32.type = import_ref Main//f32, Float32ToFloat32, loaded [concrete = constants.%Float32ToFloat32]
// CHECK:STDOUT:   %Core.import_ref.f33: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.ca8) = import_ref Core//prelude/parts/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bb2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%Core.import_ref.f33), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %.loc6_34.1: %f32.2d9 = value_of_initializer @__global_init.%Float32ToFloat32.call.loc6 [concrete = constants.%float.ece]
// CHECK:STDOUT:   %.loc6_34.2: %f32.2d9 = converted @__global_init.%Float32ToFloat32.call.loc6, %.loc6_34.1 [concrete = constants.%float.ece]
// CHECK:STDOUT:   %f32.loc6: type = type_literal constants.%f32.2d9 [concrete = constants.%f32.2d9]
// CHECK:STDOUT:   %a: %f32.2d9 = wrapper_binding a, %.loc6_34.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.afb = value_binding_pattern a [concrete = constants.%a.patt.80d]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc7_34.1: %f32.2d9 = value_of_initializer @__global_init.%Float32ToFloat32.call.loc7 [concrete = constants.%float.3a8]
// CHECK:STDOUT:   %.loc7_34.2: %f32.2d9 = converted @__global_init.%Float32ToFloat32.call.loc7, %.loc7_34.1 [concrete = constants.%float.3a8]
// CHECK:STDOUT:   %f32.loc7: type = type_literal constants.%f32.2d9 [concrete = constants.%f32.2d9]
// CHECK:STDOUT:   %b: %f32.2d9 = wrapper_binding b, %.loc7_34.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.afb = value_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_37.1: %f32.2d9 = value_of_initializer @__global_init.%Float32ToFloat32.call.loc8 [concrete = constants.%float.0e0]
// CHECK:STDOUT:   %.loc8_37.2: %f32.2d9 = converted @__global_init.%Float32ToFloat32.call.loc8, %.loc8_37.1 [concrete = constants.%float.0e0]
// CHECK:STDOUT:   %f32.loc8: type = type_literal constants.%f32.2d9 [concrete = constants.%f32.2d9]
// CHECK:STDOUT:   %c: %f32.2d9 = wrapper_binding c, %.loc8_37.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.afb = value_binding_pattern c [concrete = constants.%c.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Float32ToFloat32.ref.loc6: %Float32ToFloat32.type = name_ref Float32ToFloat32, imports.%Main.Float32ToFloat32 [concrete = constants.%Float32ToFloat32]
// CHECK:STDOUT:   %float.loc6: Core.FloatLiteral = float_literal_value 0e-1 [concrete = constants.%float.1f7]
// CHECK:STDOUT:   %impl.elem0.loc6: %.637 = impl_witness_access constants.%ImplicitAs.impl_witness.a1a, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.089]
// CHECK:STDOUT:   %bound_method.loc6_31.1: <bound method> = bound_method %float.loc6, %impl.elem0.loc6 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.5ea]
// CHECK:STDOUT:   %specific_fn.loc6: <specific function> = specific_function %impl.elem0.loc6, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_31.2: <bound method> = bound_method %float.loc6, %specific_fn.loc6 [concrete = constants.%bound_method.663]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc6: init %f32.2d9 = call %bound_method.loc6_31.2(%float.loc6) [concrete = constants.%float.ece]
// CHECK:STDOUT:   %.loc6_31.1: %f32.2d9 = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc6 [concrete = constants.%float.ece]
// CHECK:STDOUT:   %.loc6_31.2: %f32.2d9 = converted %float.loc6, %.loc6_31.1 [concrete = constants.%float.ece]
// CHECK:STDOUT:   %Float32ToFloat32.call.loc6: init %f32.2d9 = call %Float32ToFloat32.ref.loc6(%.loc6_31.2) [concrete = constants.%float.ece]
// CHECK:STDOUT:   %Float32ToFloat32.ref.loc7: %Float32ToFloat32.type = name_ref Float32ToFloat32, imports.%Main.Float32ToFloat32 [concrete = constants.%Float32ToFloat32]
// CHECK:STDOUT:   %float.loc7: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %impl.elem0.loc7: %.637 = impl_witness_access constants.%ImplicitAs.impl_witness.a1a, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.089]
// CHECK:STDOUT:   %bound_method.loc7_31.1: <bound method> = bound_method %float.loc7, %impl.elem0.loc7 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.4c1]
// CHECK:STDOUT:   %specific_fn.loc7: <specific function> = specific_function %impl.elem0.loc7, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_31.2: <bound method> = bound_method %float.loc7, %specific_fn.loc7 [concrete = constants.%bound_method.29b]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc7: init %f32.2d9 = call %bound_method.loc7_31.2(%float.loc7) [concrete = constants.%float.3a8]
// CHECK:STDOUT:   %.loc7_31.1: %f32.2d9 = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc7 [concrete = constants.%float.3a8]
// CHECK:STDOUT:   %.loc7_31.2: %f32.2d9 = converted %float.loc7, %.loc7_31.1 [concrete = constants.%float.3a8]
// CHECK:STDOUT:   %Float32ToFloat32.call.loc7: init %f32.2d9 = call %Float32ToFloat32.ref.loc7(%.loc7_31.2) [concrete = constants.%float.3a8]
// CHECK:STDOUT:   %Float32ToFloat32.ref.loc8: %Float32ToFloat32.type = name_ref Float32ToFloat32, imports.%Main.Float32ToFloat32 [concrete = constants.%Float32ToFloat32]
// CHECK:STDOUT:   %float.loc8: Core.FloatLiteral = float_literal_value 10e37 [concrete = constants.%float.8d2]
// CHECK:STDOUT:   %impl.elem0.loc8: %.637 = impl_witness_access constants.%ImplicitAs.impl_witness.a1a, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.089]
// CHECK:STDOUT:   %bound_method.loc8_31.1: <bound method> = bound_method %float.loc8, %impl.elem0.loc8 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.b07]
// CHECK:STDOUT:   %specific_fn.loc8: <specific function> = specific_function %impl.elem0.loc8, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_31.2: <bound method> = bound_method %float.loc8, %specific_fn.loc8 [concrete = constants.%bound_method.10f]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc8: init %f32.2d9 = call %bound_method.loc8_31.2(%float.loc8) [concrete = constants.%float.0e0]
// CHECK:STDOUT:   %.loc8_31.1: %f32.2d9 = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc8 [concrete = constants.%float.0e0]
// CHECK:STDOUT:   %.loc8_31.2: %f32.2d9 = converted %float.loc8, %.loc8_31.1 [concrete = constants.%float.0e0]
// CHECK:STDOUT:   %Float32ToFloat32.call.loc8: init %f32.2d9 = call %Float32ToFloat32.ref.loc8(%.loc8_31.2) [concrete = constants.%float.0e0]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- truncate.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %f32.2d9: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.afb: type = pattern_type %f32.2d9 [concrete]
// CHECK:STDOUT:   %a.patt.80d: %pattern_type.afb = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %Float64ToFloat32.type: type = fn_type @Float64ToFloat32 [concrete]
// CHECK:STDOUT:   %Float64ToFloat32: %Float64ToFloat32.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %f64.84c: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %float.6da: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.838: type = facet_type <@ImplicitAs, @ImplicitAs(%f64.84c)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.ca8: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bb2: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.ca8 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.afc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table, @Core.FloatLiteral.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.228: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.7ae: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.228 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.838 = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.afc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.01d: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%f64.84c, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.fa7: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.01d, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.6da: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.7ae [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.ImplicitAs.impl.Convert.7ae, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.066: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.b4b: %f64.84c = float_value 1 [concrete]
// CHECK:STDOUT:   %float.3a8: %f32.2d9 = float_value 1 [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.afb = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %float.cfc: Core.FloatLiteral = float_literal_value 10e38 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.e16: <bound method> = bound_method %float.cfc, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.7ae [concrete]
// CHECK:STDOUT:   %bound_method.fa1: <bound method> = bound_method %float.cfc, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.71c: %f64.84c = float_value 9.9999999999999994E+38 [concrete]
// CHECK:STDOUT:   %float.d2f: %f32.2d9 = float_value +Inf [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Float64ToFloat32: %Float64ToFloat32.type = import_ref Main//f32, Float64ToFloat32, loaded [concrete = constants.%Float64ToFloat32]
// CHECK:STDOUT:   %Core.import_ref.f33: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.ca8) = import_ref Core//prelude/parts/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bb2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%Core.import_ref.f33), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %.loc7_34.1: %f32.2d9 = value_of_initializer @__global_init.%Float64ToFloat32.call.loc7 [concrete = constants.%float.3a8]
// CHECK:STDOUT:   %.loc7_34.2: %f32.2d9 = converted @__global_init.%Float64ToFloat32.call.loc7, %.loc7_34.1 [concrete = constants.%float.3a8]
// CHECK:STDOUT:   %f32.loc7: type = type_literal constants.%f32.2d9 [concrete = constants.%f32.2d9]
// CHECK:STDOUT:   %a: %f32.2d9 = wrapper_binding a, %.loc7_34.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.afb = value_binding_pattern a [concrete = constants.%a.patt.80d]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9_37.1: %f32.2d9 = value_of_initializer @__global_init.%Float64ToFloat32.call.loc9 [concrete = constants.%float.d2f]
// CHECK:STDOUT:   %.loc9_37.2: %f32.2d9 = converted @__global_init.%Float64ToFloat32.call.loc9, %.loc9_37.1 [concrete = constants.%float.d2f]
// CHECK:STDOUT:   %f32.loc9: type = type_literal constants.%f32.2d9 [concrete = constants.%f32.2d9]
// CHECK:STDOUT:   %b: %f32.2d9 = wrapper_binding b, %.loc9_37.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.afb = value_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Float64ToFloat32.ref.loc7: %Float64ToFloat32.type = name_ref Float64ToFloat32, imports.%Main.Float64ToFloat32 [concrete = constants.%Float64ToFloat32]
// CHECK:STDOUT:   %float.loc7: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %impl.elem0.loc7: %.fa7 = impl_witness_access constants.%ImplicitAs.impl_witness.afc, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.7ae]
// CHECK:STDOUT:   %bound_method.loc7_31.1: <bound method> = bound_method %float.loc7, %impl.elem0.loc7 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.6da]
// CHECK:STDOUT:   %specific_fn.loc7: <specific function> = specific_function %impl.elem0.loc7, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_31.2: <bound method> = bound_method %float.loc7, %specific_fn.loc7 [concrete = constants.%bound_method.066]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc7: init %f64.84c = call %bound_method.loc7_31.2(%float.loc7) [concrete = constants.%float.b4b]
// CHECK:STDOUT:   %.loc7_31.1: %f64.84c = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc7 [concrete = constants.%float.b4b]
// CHECK:STDOUT:   %.loc7_31.2: %f64.84c = converted %float.loc7, %.loc7_31.1 [concrete = constants.%float.b4b]
// CHECK:STDOUT:   %Float64ToFloat32.call.loc7: init %f32.2d9 = call %Float64ToFloat32.ref.loc7(%.loc7_31.2) [concrete = constants.%float.3a8]
// CHECK:STDOUT:   %Float64ToFloat32.ref.loc9: %Float64ToFloat32.type = name_ref Float64ToFloat32, imports.%Main.Float64ToFloat32 [concrete = constants.%Float64ToFloat32]
// CHECK:STDOUT:   %float.loc9: Core.FloatLiteral = float_literal_value 10e38 [concrete = constants.%float.cfc]
// CHECK:STDOUT:   %impl.elem0.loc9: %.fa7 = impl_witness_access constants.%ImplicitAs.impl_witness.afc, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.7ae]
// CHECK:STDOUT:   %bound_method.loc9_31.1: <bound method> = bound_method %float.loc9, %impl.elem0.loc9 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.e16]
// CHECK:STDOUT:   %specific_fn.loc9: <specific function> = specific_function %impl.elem0.loc9, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_31.2: <bound method> = bound_method %float.loc9, %specific_fn.loc9 [concrete = constants.%bound_method.fa1]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %f64.84c = call %bound_method.loc9_31.2(%float.loc9) [concrete = constants.%float.71c]
// CHECK:STDOUT:   %.loc9_31.1: %f64.84c = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%float.71c]
// CHECK:STDOUT:   %.loc9_31.2: %f64.84c = converted %float.loc9, %.loc9_31.1 [concrete = constants.%float.71c]
// CHECK:STDOUT:   %Float64ToFloat32.call.loc9: init %f32.2d9 = call %Float64ToFloat32.ref.loc9(%.loc9_31.2) [concrete = constants.%float.d2f]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- extend.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %f64.84c: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.39b: type = pattern_type %f64.84c [concrete]
// CHECK:STDOUT:   %a.patt.b3a: %pattern_type.39b = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %Float32ToFloat64.type: type = fn_type @Float32ToFloat64 [concrete]
// CHECK:STDOUT:   %Float32ToFloat64: %Float32ToFloat64.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %f32.2d9: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %float.6da: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.484: type = facet_type <@ImplicitAs, @ImplicitAs(%f32.2d9)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.ca8: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bb2: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.ca8 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a1a: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.a18: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.089: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.a18 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.484 = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.a1a) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.057: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%f32.2d9, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.637: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.057, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.4c1: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.089 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.ImplicitAs.impl.Convert.089, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.29b: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.3a8: %f32.2d9 = float_value 1 [concrete]
// CHECK:STDOUT:   %float.b4b: %f64.84c = float_value 1 [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.39b = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %float.c02: Core.FloatLiteral = float_literal_value 10e29 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.ef1: <bound method> = bound_method %float.c02, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.089 [concrete]
// CHECK:STDOUT:   %bound_method.ad0: <bound method> = bound_method %float.c02, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.a6d: %f32.2d9 = float_value 1.00000002E+30 [concrete]
// CHECK:STDOUT:   %float.80b: %f64.84c = float_value 1.0000000150474662E+30 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Float32ToFloat64: %Float32ToFloat64.type = import_ref Main//f32, Float32ToFloat64, loaded [concrete = constants.%Float32ToFloat64]
// CHECK:STDOUT:   %Core.import_ref.f33: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.ca8) = import_ref Core//prelude/parts/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bb2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%Core.import_ref.f33), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %.loc7_34.1: %f64.84c = value_of_initializer @__global_init.%Float32ToFloat64.call.loc7 [concrete = constants.%float.b4b]
// CHECK:STDOUT:   %.loc7_34.2: %f64.84c = converted @__global_init.%Float32ToFloat64.call.loc7, %.loc7_34.1 [concrete = constants.%float.b4b]
// CHECK:STDOUT:   %f64.loc7: type = type_literal constants.%f64.84c [concrete = constants.%f64.84c]
// CHECK:STDOUT:   %a: %f64.84c = wrapper_binding a, %.loc7_34.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.39b = value_binding_pattern a [concrete = constants.%a.patt.b3a]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_37.1: %f64.84c = value_of_initializer @__global_init.%Float32ToFloat64.call.loc8 [concrete = constants.%float.80b]
// CHECK:STDOUT:   %.loc8_37.2: %f64.84c = converted @__global_init.%Float32ToFloat64.call.loc8, %.loc8_37.1 [concrete = constants.%float.80b]
// CHECK:STDOUT:   %f64.loc8: type = type_literal constants.%f64.84c [concrete = constants.%f64.84c]
// CHECK:STDOUT:   %b: %f64.84c = wrapper_binding b, %.loc8_37.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.39b = value_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Float32ToFloat64.ref.loc7: %Float32ToFloat64.type = name_ref Float32ToFloat64, imports.%Main.Float32ToFloat64 [concrete = constants.%Float32ToFloat64]
// CHECK:STDOUT:   %float.loc7: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %impl.elem0.loc7: %.637 = impl_witness_access constants.%ImplicitAs.impl_witness.a1a, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.089]
// CHECK:STDOUT:   %bound_method.loc7_31.1: <bound method> = bound_method %float.loc7, %impl.elem0.loc7 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.4c1]
// CHECK:STDOUT:   %specific_fn.loc7: <specific function> = specific_function %impl.elem0.loc7, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_31.2: <bound method> = bound_method %float.loc7, %specific_fn.loc7 [concrete = constants.%bound_method.29b]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc7: init %f32.2d9 = call %bound_method.loc7_31.2(%float.loc7) [concrete = constants.%float.3a8]
// CHECK:STDOUT:   %.loc7_31.1: %f32.2d9 = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc7 [concrete = constants.%float.3a8]
// CHECK:STDOUT:   %.loc7_31.2: %f32.2d9 = converted %float.loc7, %.loc7_31.1 [concrete = constants.%float.3a8]
// CHECK:STDOUT:   %Float32ToFloat64.call.loc7: init %f64.84c = call %Float32ToFloat64.ref.loc7(%.loc7_31.2) [concrete = constants.%float.b4b]
// CHECK:STDOUT:   %Float32ToFloat64.ref.loc8: %Float32ToFloat64.type = name_ref Float32ToFloat64, imports.%Main.Float32ToFloat64 [concrete = constants.%Float32ToFloat64]
// CHECK:STDOUT:   %float.loc8: Core.FloatLiteral = float_literal_value 10e29 [concrete = constants.%float.c02]
// CHECK:STDOUT:   %impl.elem0.loc8: %.637 = impl_witness_access constants.%ImplicitAs.impl_witness.a1a, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.089]
// CHECK:STDOUT:   %bound_method.loc8_31.1: <bound method> = bound_method %float.loc8, %impl.elem0.loc8 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.ef1]
// CHECK:STDOUT:   %specific_fn.loc8: <specific function> = specific_function %impl.elem0.loc8, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_31.2: <bound method> = bound_method %float.loc8, %specific_fn.loc8 [concrete = constants.%bound_method.ad0]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc8: init %f32.2d9 = call %bound_method.loc8_31.2(%float.loc8) [concrete = constants.%float.a6d]
// CHECK:STDOUT:   %.loc8_31.1: %f32.2d9 = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc8 [concrete = constants.%float.a6d]
// CHECK:STDOUT:   %.loc8_31.2: %f32.2d9 = converted %float.loc8, %.loc8_31.1 [concrete = constants.%float.a6d]
// CHECK:STDOUT:   %Float32ToFloat64.call.loc8: init %f64.84c = call %Float32ToFloat64.ref.loc8(%.loc8_31.2) [concrete = constants.%float.80b]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:

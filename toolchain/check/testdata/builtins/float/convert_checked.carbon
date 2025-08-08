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

// Narrowing
fn FloatLiteralToFloat64(a: FloatLiteral()) -> f64 = "float.convert_checked";

// --- call_literal.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: call_literal.carbon:[[@LINE+7]]:14: error: name `FloatLiteralToFloat64` not found [NameNotFound]
// CHECK:STDERR: let f: f64 = FloatLiteralToFloat64(1.0);
// CHECK:STDERR:              ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: min_prelude/parts/float.carbon:11:9: error: bit width must be 64 [CompileTimeFloatBitWidth]
// CHECK:STDERR:   adapt MakeFloat(N);
// CHECK:STDERR:         ^~~~~~~~~~~~
let f: f64 = FloatLiteralToFloat64(1.0);

// --- fail_todo_convert.carbon

library "[[@TEST_NAME]]";

// Size preserving
// CHECK:STDERR: fail_todo_convert.carbon:[[@LINE+8]]:29: note: in `f32` used here [ResolvingSpecificHere]
// CHECK:STDERR: fn Float32ToFloat32(a: f32) -> f32 = "float.convert_checked";
// CHECK:STDERR:                             ^~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_convert.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "float.convert_checked" [InvalidBuiltinSignature]
// CHECK:STDERR: fn Float32ToFloat32(a: f32) -> f32 = "float.convert_checked";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn Float32ToFloat32(a: f32) -> f32 = "float.convert_checked";
// CHECK:STDERR: fail_todo_convert.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "float.convert_checked" [InvalidBuiltinSignature]
// CHECK:STDERR: fn Float64ToFloat64(a: f64) -> f64 = "float.convert_checked";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn Float64ToFloat64(a: f64) -> f64 = "float.convert_checked";
// CHECK:STDERR: fail_todo_convert.carbon:[[@LINE+12]]:34: error: name `FloatLiteral` not found [NameNotFound]
// CHECK:STDERR: fn FloatLiteralToFloatLiteral(a: FloatLiteral()) -> FloatLiteral() =
// CHECK:STDERR:                                  ^~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_convert.carbon:[[@LINE+8]]:53: error: name `FloatLiteral` not found [NameNotFound]
// CHECK:STDERR: fn FloatLiteralToFloatLiteral(a: FloatLiteral()) -> FloatLiteral() =
// CHECK:STDERR:                                                     ^~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_convert.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "float.convert_checked" [InvalidBuiltinSignature]
// CHECK:STDERR: fn FloatLiteralToFloatLiteral(a: FloatLiteral()) -> FloatLiteral() =
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn FloatLiteralToFloatLiteral(a: FloatLiteral()) -> FloatLiteral() =
    "float.convert_checked";

// Narrowing
// CHECK:STDERR: fail_todo_convert.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "float.convert_checked" [InvalidBuiltinSignature]
// CHECK:STDERR: fn Float64ToFloat32(a: f64) -> f32 = "float.convert_checked";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn Float64ToFloat32(a: f64) -> f32 = "float.convert_checked";
// CHECK:STDERR: fail_todo_convert.carbon:[[@LINE+8]]:29: error: name `FloatLiteral` not found [NameNotFound]
// CHECK:STDERR: fn FloatLiteralToFloat32(a: FloatLiteral()) -> f32 = "float.convert_checked";
// CHECK:STDERR:                             ^~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_convert.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "float.convert_checked" [InvalidBuiltinSignature]
// CHECK:STDERR: fn FloatLiteralToFloat32(a: FloatLiteral()) -> f32 = "float.convert_checked";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn FloatLiteralToFloat32(a: FloatLiteral()) -> f32 = "float.convert_checked";

// Widening
// CHECK:STDERR: fail_todo_convert.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "float.convert_checked" [InvalidBuiltinSignature]
// CHECK:STDERR: fn Float32ToFloat64(a: f32) -> f64 = "float.convert_checked";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn Float32ToFloat64(a: f32) -> f64 = "float.convert_checked";
// CHECK:STDERR: fail_todo_convert.carbon:[[@LINE+8]]:37: error: name `FloatLiteral` not found [NameNotFound]
// CHECK:STDERR: fn Float32ToFloatLiteral(a: f32) -> FloatLiteral() = "float.convert_checked";
// CHECK:STDERR:                                     ^~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_convert.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "float.convert_checked" [InvalidBuiltinSignature]
// CHECK:STDERR: fn Float32ToFloatLiteral(a: f32) -> FloatLiteral() = "float.convert_checked";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn Float32ToFloatLiteral(a: f32) -> FloatLiteral() = "float.convert_checked";
// CHECK:STDERR: fail_todo_convert.carbon:[[@LINE+11]]:37: error: name `FloatLiteral` not found [NameNotFound]
// CHECK:STDERR: fn Float64ToFloatLiteral(a: f64) -> FloatLiteral() = "float.convert_checked";
// CHECK:STDERR:                                     ^~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_convert.carbon:[[@LINE+7]]:1: error: invalid signature for builtin function "float.convert_checked" [InvalidBuiltinSignature]
// CHECK:STDERR: fn Float64ToFloatLiteral(a: f64) -> FloatLiteral() = "float.convert_checked";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: min_prelude/parts/float.carbon:11:9: error: bit width must be 64 [CompileTimeFloatBitWidth]
// CHECK:STDERR:   adapt MakeFloat(N);
// CHECK:STDERR:         ^~~~~~~~~~~~
fn Float64ToFloatLiteral(a: f64) -> FloatLiteral() = "float.convert_checked";

// --- fail_todo_runtime_call.carbon

library "[[@TEST_NAME]]";
import library "convert";

//@dump-sem-ir-begin
let SizePreserving: f64 = Float64ToFloat64(1.0);

// CHECK:STDERR: fail_todo_runtime_call.carbon:[[@LINE+7]]:16: note: in `f32` used here [ResolvingSpecificHere]
// CHECK:STDERR: let Narrowing: f32 = Float64ToFloat32(1.0);
// CHECK:STDERR:                ^~~
// CHECK:STDERR:
// CHECK:STDERR: min_prelude/parts/float.carbon:11:9: error: bit width must be 64 [CompileTimeFloatBitWidth]
// CHECK:STDERR:   adapt MakeFloat(N);
// CHECK:STDERR:         ^~~~~~~~~~~~
let Narrowing: f32 = Float64ToFloat32(1.0);
let Widening: f64 = Float32ToFloat64(1.0);
//@dump-sem-ir-end

// --- fail_todo_identity.carbon

library "[[@TEST_NAME]]";
import library "convert";

// CHECK:STDERR: fail_todo_identity.carbon:[[@LINE+4]]:8: note: in `f32` used here [ResolvingSpecificHere]
// CHECK:STDERR: let a: f32 = Float32ToFloat32(0.0);
// CHECK:STDERR:        ^~~
// CHECK:STDERR:
let a: f32 = Float32ToFloat32(0.0);
let b: f32 = Float32ToFloat32(1.0);
// CHECK:STDERR: fail_todo_identity.carbon:[[@LINE+4]]:31: error: cannot access member of interface `Core.Negate` in type `Core.FloatLiteral` that does not implement that interface [MissingImplInMemberAccess]
// CHECK:STDERR: let c: f32 = Float32ToFloat32(-1.0);
// CHECK:STDERR:                               ^~~~
// CHECK:STDERR:
let c: f32 = Float32ToFloat32(-1.0);
// CHECK:STDERR: fail_todo_identity.carbon:[[@LINE+11]]:8: error: name `FloatLiteral` not found [NameNotFound]
// CHECK:STDERR: let d: FloatLiteral() = FloatLiteralToFloatLiteral(Float64ToFloatLiteral(-1.0));
// CHECK:STDERR:        ^~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_identity.carbon:[[@LINE+7]]:74: error: cannot access member of interface `Core.Negate` in type `Core.FloatLiteral` that does not implement that interface [MissingImplInMemberAccess]
// CHECK:STDERR: let d: FloatLiteral() = FloatLiteralToFloatLiteral(Float64ToFloatLiteral(-1.0));
// CHECK:STDERR:                                                                          ^~~~
// CHECK:STDERR:
// CHECK:STDERR: min_prelude/parts/float.carbon:11:9: error: bit width must be 64 [CompileTimeFloatBitWidth]
// CHECK:STDERR:   adapt MakeFloat(N);
// CHECK:STDERR:         ^~~~~~~~~~~~
let d: FloatLiteral() = FloatLiteralToFloatLiteral(Float64ToFloatLiteral(-1.0));

// --- fail_todo_truncate.carbon

library "[[@TEST_NAME]]";
import library "convert";

// CHECK:STDERR: fail_todo_truncate.carbon:[[@LINE+4]]:8: note: in `f32` used here [ResolvingSpecificHere]
// CHECK:STDERR: let a: f32 = Float64ToFloat32(1.0);
// CHECK:STDERR:        ^~~
// CHECK:STDERR:
let a: f32 = Float64ToFloat32(1.0);
let b: f32 = Float64ToFloat32(1.0e100);
// CHECK:STDERR: fail_todo_truncate.carbon:[[@LINE+4]]:31: error: cannot access member of interface `Core.Negate` in type `Core.FloatLiteral` that does not implement that interface [MissingImplInMemberAccess]
// CHECK:STDERR: let c: f32 = Float64ToFloat32(-1.0e100);
// CHECK:STDERR:                               ^~~~~~~~
// CHECK:STDERR:
let c: f32 = Float64ToFloat32(-1.0e100);

let lit_f32_one: f32 = FloatLiteralToFloat32(1.0);
// CHECK:STDERR: fail_todo_truncate.carbon:[[@LINE+4]]:24: error: name `FloatLiteralToFloat64` not found [NameNotFound]
// CHECK:STDERR: let lit_f64_one: f64 = FloatLiteralToFloat64(1.0);
// CHECK:STDERR:                        ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let lit_f64_one: f64 = FloatLiteralToFloat64(1.0);

// --- fail_todo_extend.carbon

library "[[@TEST_NAME]]";
import library "convert";

let a: f64 = Float32ToFloat64(1.0);
let b: f64 = Float32ToFloat64(1.0e30);
// CHECK:STDERR: fail_todo_extend.carbon:[[@LINE+7]]:31: error: cannot access member of interface `Core.Negate` in type `Core.FloatLiteral` that does not implement that interface [MissingImplInMemberAccess]
// CHECK:STDERR: let c: f64 = Float32ToFloat64(-1.0e30);
// CHECK:STDERR:                               ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: min_prelude/parts/float.carbon:11:9: error: bit width must be 64 [CompileTimeFloatBitWidth]
// CHECK:STDERR:   adapt MakeFloat(N);
// CHECK:STDERR:         ^~~~~~~~~~~~
let c: f64 = Float32ToFloat64(-1.0e30);

// --- fail_not_constant.carbon

library "[[@TEST_NAME]]";
import library "convert";

let not_constant_64: f64 = 0.0;
// CHECK:STDERR: fail_not_constant.carbon:[[@LINE+4]]:22: note: in `f32` used here [ResolvingSpecificHere]
// CHECK:STDERR: let not_constant_32: f32 = 0.0;
// CHECK:STDERR:                      ^~~
// CHECK:STDERR:
let not_constant_32: f32 = 0.0;

let convert_not_constant_narrow: f32 = Float64ToFloat32(not_constant_64);
let convert_not_constant_same: f32 = Float32ToFloat32(not_constant_32);
let convert_not_constant_widen: f64 = Float32ToFloat64(not_constant_32);

// CHECK:STDOUT: --- fail_todo_runtime_call.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %f64.d77: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.0ae: type = pattern_type %f64.d77 [concrete]
// CHECK:STDOUT:   %Float64ToFloat64.type: type = fn_type @Float64ToFloat64 [concrete]
// CHECK:STDOUT:   %Float64ToFloat64: %Float64ToFloat64.type = struct_value () [concrete]
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6ec: type = facet_type <@ImplicitAs, @ImplicitAs(%f64.d77)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.726: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%f64.d77) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.644: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.b45: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.644 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.857: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table, @Core.FloatLiteral.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.baf: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.a03: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.baf = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.8d0: %ImplicitAs.type.6ec = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.857) [concrete]
// CHECK:STDOUT:   %.678: type = fn_type_with_self_type %ImplicitAs.Convert.type.726, %ImplicitAs.facet.8d0 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.64b: <bound method> = bound_method %float, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.a03 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn.c3e: <specific function> = specific_function %Core.FloatLiteral.as.ImplicitAs.impl.Convert.a03, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.20c: <bound method> = bound_method %float, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn.c3e [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %f32: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.201: type = pattern_type %f32 [concrete]
// CHECK:STDOUT:   %Float64ToFloat32.type: type = fn_type @Float64ToFloat32 [concrete]
// CHECK:STDOUT:   %Float64ToFloat32: %Float64ToFloat32.type = struct_value () [concrete]
// CHECK:STDOUT:   %Float32ToFloat64.type: type = fn_type @Float32ToFloat64 [concrete]
// CHECK:STDOUT:   %Float32ToFloat64: %Float32ToFloat64.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.31d: type = facet_type <@ImplicitAs, @ImplicitAs(%f32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.b8c: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%f32) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.e3f: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.43b: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.1a3: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.43b = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.f6d: %ImplicitAs.type.31d = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.e3f) [concrete]
// CHECK:STDOUT:   %.a5d: type = fn_type_with_self_type %ImplicitAs.Convert.type.b8c, %ImplicitAs.facet.f6d [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.bcb: <bound method> = bound_method %float, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.1a3 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn.23b: <specific function> = specific_function %Core.FloatLiteral.as.ImplicitAs.impl.Convert.1a3, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.495: <bound method> = bound_method %float, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn.23b [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Float64ToFloat64: %Float64ToFloat64.type = import_ref Main//convert, Float64ToFloat64, loaded [concrete = constants.%Float64ToFloat64]
// CHECK:STDOUT:   %Main.Float64ToFloat32: %Float64ToFloat32.type = import_ref Main//convert, Float64ToFloat32, loaded [concrete = constants.%Float64ToFloat32]
// CHECK:STDOUT:   %Main.Float32ToFloat64: %Float32ToFloat64.type = import_ref Main//convert, Float32ToFloat64, loaded [concrete = constants.%Float32ToFloat64]
// CHECK:STDOUT:   %Core.import_ref.c0c: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.644) = import_ref Core//prelude/parts/float, loc15_41, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.b45)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%Core.import_ref.c0c), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %SizePreserving.patt: %pattern_type.0ae = binding_pattern SizePreserving [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc6_21: type = splice_block %f64.loc6 [concrete = constants.%f64.d77] {
// CHECK:STDOUT:     %int_64.loc6: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %f64.loc6: type = class_type @Float, @Float(constants.%int_64) [concrete = constants.%f64.d77]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc6_47.1: %f64.d77 = value_of_initializer @__global_init.%Float64ToFloat64.call
// CHECK:STDOUT:   %.loc6_47.2: %f64.d77 = converted @__global_init.%Float64ToFloat64.call, %.loc6_47.1
// CHECK:STDOUT:   %SizePreserving: %f64.d77 = bind_name SizePreserving, %.loc6_47.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %Narrowing.patt: %pattern_type.201 = binding_pattern Narrowing [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc15_16: type = splice_block %f32 [concrete = constants.%f32] {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %f32: type = class_type @Float, @Float(constants.%int_32) [concrete = constants.%f32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc15_42.1: %f32 = value_of_initializer @__global_init.%Float64ToFloat32.call
// CHECK:STDOUT:   %.loc15_42.2: %f32 = converted @__global_init.%Float64ToFloat32.call, %.loc15_42.1
// CHECK:STDOUT:   %Narrowing: %f32 = bind_name Narrowing, %.loc15_42.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %Widening.patt: %pattern_type.0ae = binding_pattern Widening [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc16_15: type = splice_block %f64.loc16 [concrete = constants.%f64.d77] {
// CHECK:STDOUT:     %int_64.loc16: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %f64.loc16: type = class_type @Float, @Float(constants.%int_64) [concrete = constants.%f64.d77]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc16_41.1: %f64.d77 = value_of_initializer @__global_init.%Float32ToFloat64.call
// CHECK:STDOUT:   %.loc16_41.2: %f64.d77 = converted @__global_init.%Float32ToFloat64.call, %.loc16_41.1
// CHECK:STDOUT:   %Widening: %f64.d77 = bind_name Widening, %.loc16_41.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Float64ToFloat64.ref: %Float64ToFloat64.type = name_ref Float64ToFloat64, imports.%Main.Float64ToFloat64 [concrete = constants.%Float64ToFloat64]
// CHECK:STDOUT:   %float.loc6: Core.FloatLiteral = float_value 1 [concrete = constants.%float]
// CHECK:STDOUT:   %impl.elem0.loc6: %.678 = impl_witness_access constants.%ImplicitAs.impl_witness.857, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.a03]
// CHECK:STDOUT:   %bound_method.loc6_44.1: <bound method> = bound_method %float.loc6, %impl.elem0.loc6 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.64b]
// CHECK:STDOUT:   %specific_fn.loc6: <specific function> = specific_function %impl.elem0.loc6, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn.c3e]
// CHECK:STDOUT:   %bound_method.loc6_44.2: <bound method> = bound_method %float.loc6, %specific_fn.loc6 [concrete = constants.%bound_method.20c]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc6: init %f64.d77 = call %bound_method.loc6_44.2(%float.loc6) [concrete = constants.%float]
// CHECK:STDOUT:   %.loc6_44.1: %f64.d77 = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc6 [concrete = constants.%float]
// CHECK:STDOUT:   %.loc6_44.2: %f64.d77 = converted %float.loc6, %.loc6_44.1 [concrete = constants.%float]
// CHECK:STDOUT:   %Float64ToFloat64.call: init %f64.d77 = call %Float64ToFloat64.ref(%.loc6_44.2)
// CHECK:STDOUT:   %Float64ToFloat32.ref: %Float64ToFloat32.type = name_ref Float64ToFloat32, imports.%Main.Float64ToFloat32 [concrete = constants.%Float64ToFloat32]
// CHECK:STDOUT:   %float.loc15: Core.FloatLiteral = float_value 1 [concrete = constants.%float]
// CHECK:STDOUT:   %impl.elem0.loc15: %.678 = impl_witness_access constants.%ImplicitAs.impl_witness.857, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.a03]
// CHECK:STDOUT:   %bound_method.loc15_39.1: <bound method> = bound_method %float.loc15, %impl.elem0.loc15 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.64b]
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem0.loc15, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn.c3e]
// CHECK:STDOUT:   %bound_method.loc15_39.2: <bound method> = bound_method %float.loc15, %specific_fn.loc15 [concrete = constants.%bound_method.20c]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc15: init %f64.d77 = call %bound_method.loc15_39.2(%float.loc15) [concrete = constants.%float]
// CHECK:STDOUT:   %.loc15_39.1: %f64.d77 = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc15 [concrete = constants.%float]
// CHECK:STDOUT:   %.loc15_39.2: %f64.d77 = converted %float.loc15, %.loc15_39.1 [concrete = constants.%float]
// CHECK:STDOUT:   %Float64ToFloat32.call: init %f32 = call %Float64ToFloat32.ref(%.loc15_39.2)
// CHECK:STDOUT:   %Float32ToFloat64.ref: %Float32ToFloat64.type = name_ref Float32ToFloat64, imports.%Main.Float32ToFloat64 [concrete = constants.%Float32ToFloat64]
// CHECK:STDOUT:   %float.loc16: Core.FloatLiteral = float_value 1 [concrete = constants.%float]
// CHECK:STDOUT:   %impl.elem0.loc16: %.a5d = impl_witness_access constants.%ImplicitAs.impl_witness.e3f, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.1a3]
// CHECK:STDOUT:   %bound_method.loc16_38.1: <bound method> = bound_method %float.loc16, %impl.elem0.loc16 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound.bcb]
// CHECK:STDOUT:   %specific_fn.loc16: <specific function> = specific_function %impl.elem0.loc16, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn.23b]
// CHECK:STDOUT:   %bound_method.loc16_38.2: <bound method> = bound_method %float.loc16, %specific_fn.loc16 [concrete = constants.%bound_method.495]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc16: init %f32 = call %bound_method.loc16_38.2(%float.loc16) [concrete = constants.%float]
// CHECK:STDOUT:   %.loc16_38.1: %f32 = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call.loc16 [concrete = constants.%float]
// CHECK:STDOUT:   %.loc16_38.2: %f32 = converted %float.loc16, %.loc16_38.1 [concrete = constants.%float]
// CHECK:STDOUT:   %Float32ToFloat64.call: init %f64.d77 = call %Float32ToFloat64.ref(%.loc16_38.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:

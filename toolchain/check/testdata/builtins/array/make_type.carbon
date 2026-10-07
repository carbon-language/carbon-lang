// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/array/make_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/array/make_type.carbon

// --- types.carbon

library "[[@TEST_NAME]]";

fn MakeArray(T: type, N: Core.IntLiteral) -> type = "array.make_type";

// --- use_types.carbon

library "[[@TEST_NAME]]";

import library "types";

//@dump-sem-ir-begin
var a: MakeArray(i32, 3) = (1, 2, 3);
var b: MakeArray(MakeArray(i32, 2), 0) = ();
//@dump-sem-ir-end

fn Generic[T: type](N: Core.IntLiteral) -> type {
  //@dump-sem-ir-begin
  return MakeArray(T, N);
  //@dump-sem-ir-end
}

fn Symbolic(generic T: type, generic N: Core.IntLiteral, x: MakeArray(T, N)*)
    -> MakeArray(T, N)* {
  return x;
}

// --- import_types.carbon

library "[[@TEST_NAME]]";

import library "types";
import library "use_types";

fn UseA() -> MakeArray(i32, 3)* {
  return &a;
}

fn UseB() -> MakeArray(MakeArray(i32, 2), 0)* {
  return &b;
}

fn UseSymbolic(x: MakeArray(i32, 24)*) -> MakeArray(i32, 24)* {
  return Symbolic(i32, 24, x);
}

// --- fail_negative_bound.carbon

library "[[@TEST_NAME]]";

import library "types";

// CHECK:STDERR: fail_negative_bound.carbon:[[@LINE+4]]:8: error: array bound of -1 is negative [ArrayBoundNegative]
// CHECK:STDERR: var n: MakeArray(i32, -1);
// CHECK:STDERR:        ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var n: MakeArray(i32, -1);

// --- fail_bound_too_large.carbon

library "[[@TEST_NAME]]";

import library "types";

// CHECK:STDERR: fail_bound_too_large.carbon:[[@LINE+4]]:8: error: array bound of 18446744073709551616 is too large [ArrayBoundTooLarge]
// CHECK:STDERR: var n: MakeArray(i32, 0x1_0000_0000_0000_0000);
// CHECK:STDERR:        ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var n: MakeArray(i32, 0x1_0000_0000_0000_0000);

// --- fail_dyn_bound.carbon

library "[[@TEST_NAME]]";

import library "types";

fn F(n: Core.IntLiteral) {
  // CHECK:STDERR: fail_dyn_bound.carbon:[[@LINE+4]]:17: error: cannot evaluate type expression [TypeExprEvaluationFailure]
  // CHECK:STDERR:   var unused a: MakeArray(i32, n);
  // CHECK:STDERR:                 ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var unused a: MakeArray(i32, n);
}

// CHECK:STDOUT: --- use_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %MakeArray.type: type = fn_type @MakeArray [concrete]
// CHECK:STDOUT:   %MakeArray: %MakeArray.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type type [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %array_type.dc7: type = array_type %int_3.1ba, %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.771: type = pattern_type %array_type.dc7 [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.771 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.771 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.type.37f: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.2d5: %tuple.type.37f = tuple_value (%int_1.5b8, %int_2.ecc, %int_3.1ba) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.914: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.845: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a2a: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1aa, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.914 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a2a) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.8eb: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.b7a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.8eb, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.094: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.953: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.0c6: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.9f3: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %bound_method.3cb: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.295: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.485: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %bound_method.763: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.410: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %array.f8d: %array_type.dc7 = tuple_value (%int_1.0c6, %int_2.295, %int_3.410) [concrete]
// CHECK:STDOUT:   %array_type.801: type = array_type %int_2.ecc, %i32 [concrete]
// CHECK:STDOUT:   %array_type.da1: type = array_type %int_0, %array_type.801 [concrete]
// CHECK:STDOUT:   %pattern_type.1f1: type = pattern_type %array_type.da1 [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.1f1 = ref_binding_pattern b [concrete]
// CHECK:STDOUT:   %b.var_patt: %pattern_type.1f1 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %array.c40: %array_type.da1 = tuple_value () [concrete]
// CHECK:STDOUT:   %T.patt.3a0: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.MakeArray: %MakeArray.type = import_ref Main//types, MakeArray, loaded [concrete = constants.%MakeArray]
// CHECK:STDOUT:   %Core.import_ref.edf: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.845)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1aa = impl_witness_table (%Core.import_ref.edf), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %a.var: ref %array_type.dc7 = var_storage %a.var_patt [concrete]
// CHECK:STDOUT:   %.loc7_24.1: type = splice_block %.loc7_24.3 [concrete = constants.%array_type.dc7] {
// CHECK:STDOUT:     %MakeArray.ref.loc7: %MakeArray.type = name_ref MakeArray, imports.%Main.MakeArray [concrete = constants.%MakeArray]
// CHECK:STDOUT:     %i32.loc7: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %MakeArray.call.loc7: init type = call %MakeArray.ref.loc7(%i32.loc7, %int_3) [concrete = constants.%array_type.dc7]
// CHECK:STDOUT:     %.loc7_24.2: type = value_of_initializer %MakeArray.call.loc7 [concrete = constants.%array_type.dc7]
// CHECK:STDOUT:     %.loc7_24.3: type = converted %MakeArray.call.loc7, %.loc7_24.2 [concrete = constants.%array_type.dc7]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type.dc7 = wrapper_binding a, %a.var [concrete = %a.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.771 = ref_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.771 = var_pattern %a.patt [concrete = constants.%a.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %array_type.da1 = var_storage %b.var_patt [concrete]
// CHECK:STDOUT:   %.loc8_38.1: type = splice_block %.loc8_38.3 [concrete = constants.%array_type.da1] {
// CHECK:STDOUT:     %MakeArray.ref.loc8_8: %MakeArray.type = name_ref MakeArray, imports.%Main.MakeArray [concrete = constants.%MakeArray]
// CHECK:STDOUT:     %MakeArray.ref.loc8_18: %MakeArray.type = name_ref MakeArray, imports.%Main.MakeArray [concrete = constants.%MakeArray]
// CHECK:STDOUT:     %i32.loc8: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %MakeArray.call.loc8_34: init type = call %MakeArray.ref.loc8_18(%i32.loc8, %int_2) [concrete = constants.%array_type.801]
// CHECK:STDOUT:     %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:     %.loc8_34.1: type = value_of_initializer %MakeArray.call.loc8_34 [concrete = constants.%array_type.801]
// CHECK:STDOUT:     %.loc8_34.2: type = converted %MakeArray.call.loc8_34, %.loc8_34.1 [concrete = constants.%array_type.801]
// CHECK:STDOUT:     %MakeArray.call.loc8_38: init type = call %MakeArray.ref.loc8_8(%.loc8_34.2, %int_0) [concrete = constants.%array_type.da1]
// CHECK:STDOUT:     %.loc8_38.2: type = value_of_initializer %MakeArray.call.loc8_38 [concrete = constants.%array_type.da1]
// CHECK:STDOUT:     %.loc8_38.3: type = converted %MakeArray.call.loc8_38, %.loc8_38.2 [concrete = constants.%array_type.da1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %array_type.da1 = wrapper_binding b, %b.var [concrete = %b.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.1f1 = ref_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.1f1 = var_pattern %b.patt [concrete = constants.%b.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Generic(%T.loc11_13.2: type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%N.param: Core.IntLiteral) -> out %return.param: type {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %MakeArray.ref: %MakeArray.type = name_ref MakeArray, imports.%Main.MakeArray [concrete = constants.%MakeArray]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc11_13.2 [symbolic = %T.loc11_13.1 (constants.%T.67d)]
// CHECK:STDOUT:     %N.ref: Core.IntLiteral = name_ref N, %N
// CHECK:STDOUT:     %MakeArray.call: init type = call %MakeArray.ref(%T.ref, %N.ref)
// CHECK:STDOUT:     return %MakeArray.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1.loc7_29: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc7_32: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %.loc7_36.1: %tuple.type.37f = tuple_literal (%int_1.loc7_29, %int_2.loc7_32, %int_3) [concrete = constants.%tuple.2d5]
// CHECK:STDOUT:   %impl.elem0.loc7_36.1: %.b7a = impl_witness_access constants.%ImplicitAs.impl_witness.a2a, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc7_36.1: <bound method> = bound_method %int_1.loc7_29, %impl.elem0.loc7_36.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.094]
// CHECK:STDOUT:   %specific_fn.loc7_36.1: <specific function> = specific_function %impl.elem0.loc7_36.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_36.2: <bound method> = bound_method %int_1.loc7_29, %specific_fn.loc7_36.1 [concrete = constants.%bound_method.953]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7_36.1: init %i32 = call %bound_method.loc7_36.2(%int_1.loc7_29) [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %.loc7_36.2: init %i32 = converted %int_1.loc7_29, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7_36.1 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc7_36.3: ref %i32 = array_index file.%a.var, %int_0
// CHECK:STDOUT:   %.loc7_36.4: init %i32 to %.loc7_36.3 = in_place_init %.loc7_36.2 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %impl.elem0.loc7_36.2: %.b7a = impl_witness_access constants.%ImplicitAs.impl_witness.a2a, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc7_36.3: <bound method> = bound_method %int_2.loc7_32, %impl.elem0.loc7_36.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.9f3]
// CHECK:STDOUT:   %specific_fn.loc7_36.2: <specific function> = specific_function %impl.elem0.loc7_36.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_36.4: <bound method> = bound_method %int_2.loc7_32, %specific_fn.loc7_36.2 [concrete = constants.%bound_method.3cb]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7_36.2: init %i32 = call %bound_method.loc7_36.4(%int_2.loc7_32) [concrete = constants.%int_2.295]
// CHECK:STDOUT:   %.loc7_36.5: init %i32 = converted %int_2.loc7_32, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7_36.2 [concrete = constants.%int_2.295]
// CHECK:STDOUT:   %int_1.loc7_36: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc7_36.6: ref %i32 = array_index file.%a.var, %int_1.loc7_36
// CHECK:STDOUT:   %.loc7_36.7: init %i32 to %.loc7_36.6 = in_place_init %.loc7_36.5 [concrete = constants.%int_2.295]
// CHECK:STDOUT:   %impl.elem0.loc7_36.3: %.b7a = impl_witness_access constants.%ImplicitAs.impl_witness.a2a, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc7_36.5: <bound method> = bound_method %int_3, %impl.elem0.loc7_36.3 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.485]
// CHECK:STDOUT:   %specific_fn.loc7_36.3: <specific function> = specific_function %impl.elem0.loc7_36.3, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_36.6: <bound method> = bound_method %int_3, %specific_fn.loc7_36.3 [concrete = constants.%bound_method.763]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7_36.3: init %i32 = call %bound_method.loc7_36.6(%int_3) [concrete = constants.%int_3.410]
// CHECK:STDOUT:   %.loc7_36.8: init %i32 = converted %int_3, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7_36.3 [concrete = constants.%int_3.410]
// CHECK:STDOUT:   %int_2.loc7_36: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc7_36.9: ref %i32 = array_index file.%a.var, %int_2.loc7_36
// CHECK:STDOUT:   %.loc7_36.10: init %i32 to %.loc7_36.9 = in_place_init %.loc7_36.8 [concrete = constants.%int_3.410]
// CHECK:STDOUT:   %.loc7_36.11: init %array_type.dc7 to file.%a.var = array_init (%.loc7_36.4, %.loc7_36.7, %.loc7_36.10) [concrete = constants.%array.f8d]
// CHECK:STDOUT:   %.loc7_1: init %array_type.dc7 = converted %.loc7_36.1, %.loc7_36.11 [concrete = constants.%array.f8d]
// CHECK:STDOUT:   assign file.%a.var, %.loc7_1
// CHECK:STDOUT:   %.loc8_43.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc8_43.2: init %array_type.da1 to file.%b.var = array_init () [concrete = constants.%array.c40]
// CHECK:STDOUT:   %.loc8_1: init %array_type.da1 = converted %.loc8_43.1, %.loc8_43.2 [concrete = constants.%array.c40]
// CHECK:STDOUT:   assign file.%b.var, %.loc8_1
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%T.67d) {
// CHECK:STDOUT:   %T.patt.loc11_13.2 => constants.%T.patt.3a0
// CHECK:STDOUT:   %T.loc11_13.1 => constants.%T.67d
// CHECK:STDOUT: }
// CHECK:STDOUT:

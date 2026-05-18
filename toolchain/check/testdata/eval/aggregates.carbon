// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/eval/aggregates.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/eval/aggregates.carbon

// --- basics.carbon
library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
var tuple_copy: (i32, i32) = (1, 2) as (i32, i32);

var struct_copy: {.a: i32, .b: i32, .c: i32} = {.c = 3, .b = 2, .a = 1} as {.b: i32, .a: i32, .c: i32};

var tuple_index: array(i32, 1) = (0,) as array(i32, (5, 7, 1, 9).2);

var struct_access: array(i32, 1) = (0,) as array(i32, {.a = 3, .b = 1}.b);
//@dump-sem-ir-end

// --- fail_todo_array_temporary.carbon
library "[[@TEST_NAME]]";

// TODO: This creates an array temporary, which we don't yet support evaluating.
//@dump-sem-ir-begin
// CHECK:STDERR: fail_todo_array_temporary.carbon:[[@LINE+4]]:53: error: array bound is not a constant [InvalidArrayExpr]
// CHECK:STDERR: var array_index: array(i32, 1) = (0,) as array(i32, ((5, 7, 1, 9) as array(i32, 4))[2]);
// CHECK:STDERR:                                                     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var array_index: array(i32, 1) = (0,) as array(i32, ((5, 7, 1, 9) as array(i32, 4))[2]);
//@dump-sem-ir-end

// --- symbolic.carbon
library "[[@TEST_NAME]]";

// Check that we propagate the `symbolic` tag through evaluations.
fn F(T:! Core.Destroy) {
  //@dump-sem-ir-begin
  var unused u: (T*, const T);
  var unused v: {.a: T};
  var unused w: array(T, 5);
  //@dump-sem-ir-end
}

fn G(N:! i32) {
  //@dump-sem-ir-begin
  var unused k: array(i32, N);
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- basics.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.519: %tuple.type.24b = tuple_value (%i32, %i32) [concrete]
// CHECK:STDOUT:   %tuple.type.e53: type = tuple_type (%i32, %i32) [concrete]
// CHECK:STDOUT:   %pattern_type.a63: type = pattern_type %tuple.type.e53 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.type.f94: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.ad8: %tuple.type.f94 = tuple_value (%int_1.5b8, %int_2.ecc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ef: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.aad: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.abb: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.aad = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.f4c: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.e6c, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.9ca: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.9ca = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.0ef = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.f4c) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.61c: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.cbf: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.61c, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.0b7: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.1c4: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.d72: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.099: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5 [concrete]
// CHECK:STDOUT:   %bound_method.c74: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.cb7: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.21c: %tuple.type.e53 = tuple_value (%int_1.d72, %int_2.cb7) [concrete]
// CHECK:STDOUT:   %tuple.elem0: ref %i32 = tuple_access file.%tuple_copy.var, element0 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.521: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.783: %Int.as.Copy.impl.Op.type.521 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.f1f: <witness> = impl_witness imports.%Copy.impl_witness_table.592, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.f98: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.6d3: %Int.as.Copy.impl.Op.type.f98 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.f1f) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.83a: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.2ec: type = fn_type_with_self_type %Copy.WithSelf.Op.type.83a, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound.a11: <bound method> = bound_method %int_1.d72, %Int.as.Copy.impl.Op.6d3 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.6d3, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.1de: <bound method> = bound_method %int_1.d72, %Int.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %tuple.elem1: ref %i32 = tuple_access file.%tuple_copy.var, element1 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound.110: <bound method> = bound_method %int_2.cb7, %Int.as.Copy.impl.Op.6d3 [concrete]
// CHECK:STDOUT:   %bound_method.56d: <bound method> = bound_method %int_2.cb7, %Int.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %struct_type.a.b.c: type = struct_type {.a: %i32, .b: %i32, .c: %i32} [concrete]
// CHECK:STDOUT:   %pattern_type.fba: type = pattern_type %struct_type.a.b.c [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %struct_type.c.b.a: type = struct_type {.c: Core.IntLiteral, .b: Core.IntLiteral, .a: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct.c75: %struct_type.c.b.a = struct_value (%int_3.1ba, %int_2.ecc, %int_1.5b8) [concrete]
// CHECK:STDOUT:   %struct_type.b.a.c: type = struct_type {.b: %i32, .a: %i32, .c: %i32} [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.eab: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5 [concrete]
// CHECK:STDOUT:   %bound_method.77a: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.781: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %struct.e7e: %struct_type.b.a.c = struct_value (%int_2.cb7, %int_1.d72, %int_3.781) [concrete]
// CHECK:STDOUT:   %.f8b: ref %i32 = struct_access file.%struct_copy.var, element1 [concrete]
// CHECK:STDOUT:   %.74a: ref %i32 = struct_access file.%struct_copy.var, element0 [concrete]
// CHECK:STDOUT:   %.850: ref %i32 = struct_access file.%struct_copy.var, element2 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound.22c: <bound method> = bound_method %int_3.781, %Int.as.Copy.impl.Op.6d3 [concrete]
// CHECK:STDOUT:   %bound_method.a5f: <bound method> = bound_method %int_3.781, %Int.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %struct.a9a: %struct_type.a.b.c = struct_value (%int_1.d72, %int_2.cb7, %int_3.781) [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_1.5b8, %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.e1f: type = pattern_type %array_type [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %tuple.type.985: type = tuple_type (Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.4f2: %tuple.type.985 = tuple_value (%int_0.5c6) [concrete]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %int_7: Core.IntLiteral = int_value 7 [concrete]
// CHECK:STDOUT:   %int_9: Core.IntLiteral = int_value 9 [concrete]
// CHECK:STDOUT:   %tuple.type.d46: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.869: %tuple.type.d46 = tuple_value (%int_5, %int_7, %int_1.5b8, %int_9) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.ce0: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5 [concrete]
// CHECK:STDOUT:   %bound_method.e76: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.fed: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%int_0.fed) [concrete]
// CHECK:STDOUT:   %struct_type.a.b: type = struct_type {.a: Core.IntLiteral, .b: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct.a81: %struct_type.a.b = struct_value (%int_3.1ba, %int_1.5b8) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.fa9: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.aad) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.abb)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.e6c = impl_witness_table (%Core.import_ref.fa9), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.0bb: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.521) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.783)]
// CHECK:STDOUT:   %Copy.impl_witness_table.592 = impl_witness_table (%Core.import_ref.0bb), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %tuple_copy.patt: %pattern_type.a63 = ref_binding_pattern tuple_copy [concrete]
// CHECK:STDOUT:     %tuple_copy.var_patt: %pattern_type.a63 = var_pattern %tuple_copy.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %tuple_copy.var: ref %tuple.type.e53 = var %tuple_copy.var_patt [concrete]
// CHECK:STDOUT:   %.loc4_26.1: type = splice_block %.loc4_26.3 [concrete = constants.%tuple.type.e53] {
// CHECK:STDOUT:     %i32.loc4_18: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc4_23: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc4_26.2: %tuple.type.24b = tuple_literal (%i32.loc4_18, %i32.loc4_23) [concrete = constants.%tuple.519]
// CHECK:STDOUT:     %.loc4_26.3: type = converted %.loc4_26.2, constants.%tuple.type.e53 [concrete = constants.%tuple.type.e53]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %tuple_copy: ref %tuple.type.e53 = ref_binding tuple_copy, %tuple_copy.var [concrete = %tuple_copy.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %struct_copy.patt: %pattern_type.fba = ref_binding_pattern struct_copy [concrete]
// CHECK:STDOUT:     %struct_copy.var_patt: %pattern_type.fba = var_pattern %struct_copy.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %struct_copy.var: ref %struct_type.a.b.c = var %struct_copy.var_patt [concrete]
// CHECK:STDOUT:   %.loc6: type = splice_block %struct_type.a.b.c [concrete = constants.%struct_type.a.b.c] {
// CHECK:STDOUT:     %i32.loc6_23: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc6_32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc6_41: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.a.b.c: type = struct_type {.a: %i32, .b: %i32, .c: %i32} [concrete = constants.%struct_type.a.b.c]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %struct_copy: ref %struct_type.a.b.c = ref_binding struct_copy, %struct_copy.var [concrete = %struct_copy.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %tuple_index.patt: %pattern_type.e1f = ref_binding_pattern tuple_index [concrete]
// CHECK:STDOUT:     %tuple_index.var_patt: %pattern_type.e1f = var_pattern %tuple_index.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %tuple_index.var: ref %array_type = var %tuple_index.var_patt [concrete]
// CHECK:STDOUT:   %.loc8: type = splice_block %array_type.loc8 [concrete = constants.%array_type] {
// CHECK:STDOUT:     %i32.loc8: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %array_type.loc8: type = array_type %int_1.loc8, %i32.loc8 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %tuple_index: ref %array_type = ref_binding tuple_index, %tuple_index.var [concrete = %tuple_index.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %struct_access.patt: %pattern_type.e1f = ref_binding_pattern struct_access [concrete]
// CHECK:STDOUT:     %struct_access.var_patt: %pattern_type.e1f = var_pattern %struct_access.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %struct_access.var: ref %array_type = var %struct_access.var_patt [concrete]
// CHECK:STDOUT:   %.loc10: type = splice_block %array_type.loc10 [concrete = constants.%array_type] {
// CHECK:STDOUT:     %i32.loc10: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %int_1.loc10: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %array_type.loc10: type = array_type %int_1.loc10, %i32.loc10 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %struct_access: ref %array_type = ref_binding struct_access, %struct_access.var [concrete = %struct_access.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1.loc4: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc4: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc4_35.1: %tuple.type.f94 = tuple_literal (%int_1.loc4, %int_2.loc4) [concrete = constants.%tuple.ad8]
// CHECK:STDOUT:   %i32.loc4_41: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %i32.loc4_46: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc4_49.1: %tuple.type.24b = tuple_literal (%i32.loc4_41, %i32.loc4_46) [concrete = constants.%tuple.519]
// CHECK:STDOUT:   %.loc4_49.2: type = converted %.loc4_49.1, constants.%tuple.type.e53 [concrete = constants.%tuple.type.e53]
// CHECK:STDOUT:   %impl.elem0.loc4_35.1: %.cbf = impl_witness_access constants.%ImplicitAs.impl_witness.f4c, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5]
// CHECK:STDOUT:   %bound_method.loc4_35.1: <bound method> = bound_method %int_1.loc4, %impl.elem0.loc4_35.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.0b7]
// CHECK:STDOUT:   %specific_fn.loc4_35.1: <specific function> = specific_function %impl.elem0.loc4_35.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc4_35.2: <bound method> = bound_method %int_1.loc4, %specific_fn.loc4_35.1 [concrete = constants.%bound_method.1c4]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc4_35.1: init %i32 = call %bound_method.loc4_35.2(%int_1.loc4) [concrete = constants.%int_1.d72]
// CHECK:STDOUT:   %.loc4_35.2: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc4_35.1 [concrete = constants.%int_1.d72]
// CHECK:STDOUT:   %.loc4_35.3: %i32 = converted %int_1.loc4, %.loc4_35.2 [concrete = constants.%int_1.d72]
// CHECK:STDOUT:   %impl.elem0.loc4_35.2: %.cbf = impl_witness_access constants.%ImplicitAs.impl_witness.f4c, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5]
// CHECK:STDOUT:   %bound_method.loc4_35.3: <bound method> = bound_method %int_2.loc4, %impl.elem0.loc4_35.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.099]
// CHECK:STDOUT:   %specific_fn.loc4_35.2: <specific function> = specific_function %impl.elem0.loc4_35.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc4_35.4: <bound method> = bound_method %int_2.loc4, %specific_fn.loc4_35.2 [concrete = constants.%bound_method.c74]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc4_35.2: init %i32 = call %bound_method.loc4_35.4(%int_2.loc4) [concrete = constants.%int_2.cb7]
// CHECK:STDOUT:   %.loc4_35.4: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc4_35.2 [concrete = constants.%int_2.cb7]
// CHECK:STDOUT:   %.loc4_35.5: %i32 = converted %int_2.loc4, %.loc4_35.4 [concrete = constants.%int_2.cb7]
// CHECK:STDOUT:   %tuple.loc4: %tuple.type.e53 = tuple_value (%.loc4_35.3, %.loc4_35.5) [concrete = constants.%tuple.21c]
// CHECK:STDOUT:   %.loc4_37.1: %tuple.type.e53 = converted %.loc4_35.1, %tuple.loc4 [concrete = constants.%tuple.21c]
// CHECK:STDOUT:   %tuple.elem0.loc4_37.1: %i32 = tuple_access %.loc4_37.1, element0 [concrete = constants.%int_1.d72]
// CHECK:STDOUT:   %impl.elem0.loc4_37.1: %.2ec = impl_witness_access constants.%Copy.impl_witness.f1f, element0 [concrete = constants.%Int.as.Copy.impl.Op.6d3]
// CHECK:STDOUT:   %bound_method.loc4_37.1: <bound method> = bound_method %tuple.elem0.loc4_37.1, %impl.elem0.loc4_37.1 [concrete = constants.%Int.as.Copy.impl.Op.bound.a11]
// CHECK:STDOUT:   %specific_fn.loc4_37.1: <specific function> = specific_function %impl.elem0.loc4_37.1, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc4_37.2: <bound method> = bound_method %tuple.elem0.loc4_37.1, %specific_fn.loc4_37.1 [concrete = constants.%bound_method.1de]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc4_37.1: init %i32 = call %bound_method.loc4_37.2(%tuple.elem0.loc4_37.1) [concrete = constants.%int_1.d72]
// CHECK:STDOUT:   %tuple.elem0.loc4_37.2: ref %i32 = tuple_access file.%tuple_copy.var, element0 [concrete = constants.%tuple.elem0]
// CHECK:STDOUT:   %.loc4_37.2: init %i32 to %tuple.elem0.loc4_37.2 = in_place_init %Int.as.Copy.impl.Op.call.loc4_37.1 [concrete = constants.%int_1.d72]
// CHECK:STDOUT:   %tuple.elem1.loc4_37.1: %i32 = tuple_access %.loc4_37.1, element1 [concrete = constants.%int_2.cb7]
// CHECK:STDOUT:   %impl.elem0.loc4_37.2: %.2ec = impl_witness_access constants.%Copy.impl_witness.f1f, element0 [concrete = constants.%Int.as.Copy.impl.Op.6d3]
// CHECK:STDOUT:   %bound_method.loc4_37.3: <bound method> = bound_method %tuple.elem1.loc4_37.1, %impl.elem0.loc4_37.2 [concrete = constants.%Int.as.Copy.impl.Op.bound.110]
// CHECK:STDOUT:   %specific_fn.loc4_37.2: <specific function> = specific_function %impl.elem0.loc4_37.2, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc4_37.4: <bound method> = bound_method %tuple.elem1.loc4_37.1, %specific_fn.loc4_37.2 [concrete = constants.%bound_method.56d]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc4_37.2: init %i32 = call %bound_method.loc4_37.4(%tuple.elem1.loc4_37.1) [concrete = constants.%int_2.cb7]
// CHECK:STDOUT:   %tuple.elem1.loc4_37.2: ref %i32 = tuple_access file.%tuple_copy.var, element1 [concrete = constants.%tuple.elem1]
// CHECK:STDOUT:   %.loc4_37.3: init %i32 to %tuple.elem1.loc4_37.2 = in_place_init %Int.as.Copy.impl.Op.call.loc4_37.2 [concrete = constants.%int_2.cb7]
// CHECK:STDOUT:   %.loc4_37.4: init %tuple.type.e53 to file.%tuple_copy.var = tuple_init (%.loc4_37.2, %.loc4_37.3) [concrete = constants.%tuple.21c]
// CHECK:STDOUT:   %.loc4_1: init %tuple.type.e53 = converted %.loc4_37.1, %.loc4_37.4 [concrete = constants.%tuple.21c]
// CHECK:STDOUT:   assign file.%tuple_copy.var, %.loc4_1
// CHECK:STDOUT:   %int_3.loc6: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %int_2.loc6: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %int_1.loc6: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc6_71.1: %struct_type.c.b.a = struct_literal (%int_3.loc6, %int_2.loc6, %int_1.loc6) [concrete = constants.%struct.c75]
// CHECK:STDOUT:   %i32.loc6_81: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %i32.loc6_90: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %i32.loc6_99: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %struct_type.b.a.c: type = struct_type {.b: %i32, .a: %i32, .c: %i32} [concrete = constants.%struct_type.b.a.c]
// CHECK:STDOUT:   %impl.elem0.loc6_71.1: %.cbf = impl_witness_access constants.%ImplicitAs.impl_witness.f4c, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5]
// CHECK:STDOUT:   %bound_method.loc6_71.1: <bound method> = bound_method %int_2.loc6, %impl.elem0.loc6_71.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.099]
// CHECK:STDOUT:   %specific_fn.loc6_71.1: <specific function> = specific_function %impl.elem0.loc6_71.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_71.2: <bound method> = bound_method %int_2.loc6, %specific_fn.loc6_71.1 [concrete = constants.%bound_method.c74]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_71.1: init %i32 = call %bound_method.loc6_71.2(%int_2.loc6) [concrete = constants.%int_2.cb7]
// CHECK:STDOUT:   %.loc6_71.2: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_71.1 [concrete = constants.%int_2.cb7]
// CHECK:STDOUT:   %.loc6_71.3: %i32 = converted %int_2.loc6, %.loc6_71.2 [concrete = constants.%int_2.cb7]
// CHECK:STDOUT:   %impl.elem0.loc6_71.2: %.cbf = impl_witness_access constants.%ImplicitAs.impl_witness.f4c, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5]
// CHECK:STDOUT:   %bound_method.loc6_71.3: <bound method> = bound_method %int_1.loc6, %impl.elem0.loc6_71.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.0b7]
// CHECK:STDOUT:   %specific_fn.loc6_71.2: <specific function> = specific_function %impl.elem0.loc6_71.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_71.4: <bound method> = bound_method %int_1.loc6, %specific_fn.loc6_71.2 [concrete = constants.%bound_method.1c4]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_71.2: init %i32 = call %bound_method.loc6_71.4(%int_1.loc6) [concrete = constants.%int_1.d72]
// CHECK:STDOUT:   %.loc6_71.4: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_71.2 [concrete = constants.%int_1.d72]
// CHECK:STDOUT:   %.loc6_71.5: %i32 = converted %int_1.loc6, %.loc6_71.4 [concrete = constants.%int_1.d72]
// CHECK:STDOUT:   %impl.elem0.loc6_71.3: %.cbf = impl_witness_access constants.%ImplicitAs.impl_witness.f4c, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5]
// CHECK:STDOUT:   %bound_method.loc6_71.5: <bound method> = bound_method %int_3.loc6, %impl.elem0.loc6_71.3 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.eab]
// CHECK:STDOUT:   %specific_fn.loc6_71.3: <specific function> = specific_function %impl.elem0.loc6_71.3, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_71.6: <bound method> = bound_method %int_3.loc6, %specific_fn.loc6_71.3 [concrete = constants.%bound_method.77a]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_71.3: init %i32 = call %bound_method.loc6_71.6(%int_3.loc6) [concrete = constants.%int_3.781]
// CHECK:STDOUT:   %.loc6_71.6: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_71.3 [concrete = constants.%int_3.781]
// CHECK:STDOUT:   %.loc6_71.7: %i32 = converted %int_3.loc6, %.loc6_71.6 [concrete = constants.%int_3.781]
// CHECK:STDOUT:   %struct.loc6: %struct_type.b.a.c = struct_value (%.loc6_71.3, %.loc6_71.5, %.loc6_71.7) [concrete = constants.%struct.e7e]
// CHECK:STDOUT:   %.loc6_73.1: %struct_type.b.a.c = converted %.loc6_71.1, %struct.loc6 [concrete = constants.%struct.e7e]
// CHECK:STDOUT:   %.loc6_73.2: %i32 = struct_access %.loc6_73.1, element1 [concrete = constants.%int_1.d72]
// CHECK:STDOUT:   %impl.elem0.loc6_73.1: %.2ec = impl_witness_access constants.%Copy.impl_witness.f1f, element0 [concrete = constants.%Int.as.Copy.impl.Op.6d3]
// CHECK:STDOUT:   %bound_method.loc6_73.1: <bound method> = bound_method %.loc6_73.2, %impl.elem0.loc6_73.1 [concrete = constants.%Int.as.Copy.impl.Op.bound.a11]
// CHECK:STDOUT:   %specific_fn.loc6_73.1: <specific function> = specific_function %impl.elem0.loc6_73.1, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_73.2: <bound method> = bound_method %.loc6_73.2, %specific_fn.loc6_73.1 [concrete = constants.%bound_method.1de]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc6_73.1: init %i32 = call %bound_method.loc6_73.2(%.loc6_73.2) [concrete = constants.%int_1.d72]
// CHECK:STDOUT:   %.loc6_73.3: ref %i32 = struct_access file.%struct_copy.var, element1 [concrete = constants.%.f8b]
// CHECK:STDOUT:   %.loc6_73.4: init %i32 to %.loc6_73.3 = in_place_init %Int.as.Copy.impl.Op.call.loc6_73.1 [concrete = constants.%int_1.d72]
// CHECK:STDOUT:   %.loc6_73.5: %i32 = struct_access %.loc6_73.1, element0 [concrete = constants.%int_2.cb7]
// CHECK:STDOUT:   %impl.elem0.loc6_73.2: %.2ec = impl_witness_access constants.%Copy.impl_witness.f1f, element0 [concrete = constants.%Int.as.Copy.impl.Op.6d3]
// CHECK:STDOUT:   %bound_method.loc6_73.3: <bound method> = bound_method %.loc6_73.5, %impl.elem0.loc6_73.2 [concrete = constants.%Int.as.Copy.impl.Op.bound.110]
// CHECK:STDOUT:   %specific_fn.loc6_73.2: <specific function> = specific_function %impl.elem0.loc6_73.2, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_73.4: <bound method> = bound_method %.loc6_73.5, %specific_fn.loc6_73.2 [concrete = constants.%bound_method.56d]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc6_73.2: init %i32 = call %bound_method.loc6_73.4(%.loc6_73.5) [concrete = constants.%int_2.cb7]
// CHECK:STDOUT:   %.loc6_73.6: ref %i32 = struct_access file.%struct_copy.var, element0 [concrete = constants.%.74a]
// CHECK:STDOUT:   %.loc6_73.7: init %i32 to %.loc6_73.6 = in_place_init %Int.as.Copy.impl.Op.call.loc6_73.2 [concrete = constants.%int_2.cb7]
// CHECK:STDOUT:   %.loc6_73.8: %i32 = struct_access %.loc6_73.1, element2 [concrete = constants.%int_3.781]
// CHECK:STDOUT:   %impl.elem0.loc6_73.3: %.2ec = impl_witness_access constants.%Copy.impl_witness.f1f, element0 [concrete = constants.%Int.as.Copy.impl.Op.6d3]
// CHECK:STDOUT:   %bound_method.loc6_73.5: <bound method> = bound_method %.loc6_73.8, %impl.elem0.loc6_73.3 [concrete = constants.%Int.as.Copy.impl.Op.bound.22c]
// CHECK:STDOUT:   %specific_fn.loc6_73.3: <specific function> = specific_function %impl.elem0.loc6_73.3, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_73.6: <bound method> = bound_method %.loc6_73.8, %specific_fn.loc6_73.3 [concrete = constants.%bound_method.a5f]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc6_73.3: init %i32 = call %bound_method.loc6_73.6(%.loc6_73.8) [concrete = constants.%int_3.781]
// CHECK:STDOUT:   %.loc6_73.9: ref %i32 = struct_access file.%struct_copy.var, element2 [concrete = constants.%.850]
// CHECK:STDOUT:   %.loc6_73.10: init %i32 to %.loc6_73.9 = in_place_init %Int.as.Copy.impl.Op.call.loc6_73.3 [concrete = constants.%int_3.781]
// CHECK:STDOUT:   %.loc6_73.11: init %struct_type.a.b.c to file.%struct_copy.var = struct_init (%.loc6_73.4, %.loc6_73.7, %.loc6_73.10) [concrete = constants.%struct.a9a]
// CHECK:STDOUT:   %.loc6_1: init %struct_type.a.b.c = converted %.loc6_73.1, %.loc6_73.11 [concrete = constants.%struct.a9a]
// CHECK:STDOUT:   assign file.%struct_copy.var, %.loc6_1
// CHECK:STDOUT:   %int_0.loc8_35: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc8_37.1: %tuple.type.985 = tuple_literal (%int_0.loc8_35) [concrete = constants.%tuple.4f2]
// CHECK:STDOUT:   %i32.loc8: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5]
// CHECK:STDOUT:   %int_7: Core.IntLiteral = int_value 7 [concrete = constants.%int_7]
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_9: Core.IntLiteral = int_value 9 [concrete = constants.%int_9]
// CHECK:STDOUT:   %.loc8_64.1: %tuple.type.d46 = tuple_literal (%int_5, %int_7, %int_1.loc8, %int_9) [concrete = constants.%tuple.869]
// CHECK:STDOUT:   %int_2.loc8: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %tuple.loc8: %tuple.type.d46 = tuple_value (%int_5, %int_7, %int_1.loc8, %int_9) [concrete = constants.%tuple.869]
// CHECK:STDOUT:   %.loc8_64.2: %tuple.type.d46 = converted %.loc8_64.1, %tuple.loc8 [concrete = constants.%tuple.869]
// CHECK:STDOUT:   %tuple.elem2: Core.IntLiteral = tuple_access %.loc8_64.2, element2 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %array_type.loc8: type = array_type %tuple.elem2, %i32.loc8 [concrete = constants.%array_type]
// CHECK:STDOUT:   %impl.elem0.loc8: %.cbf = impl_witness_access constants.%ImplicitAs.impl_witness.f4c, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5]
// CHECK:STDOUT:   %bound_method.loc8_37.1: <bound method> = bound_method %int_0.loc8_35, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.ce0]
// CHECK:STDOUT:   %specific_fn.loc8: <specific function> = specific_function %impl.elem0.loc8, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_37.2: <bound method> = bound_method %int_0.loc8_35, %specific_fn.loc8 [concrete = constants.%bound_method.e76]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8: init %i32 = call %bound_method.loc8_37.2(%int_0.loc8_35) [concrete = constants.%int_0.fed]
// CHECK:STDOUT:   %.loc8_37.2: init %i32 = converted %int_0.loc8_35, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8 [concrete = constants.%int_0.fed]
// CHECK:STDOUT:   %.loc8_1: ref %array_type = splice_block file.%tuple_index.var [concrete = file.%tuple_index.var] {}
// CHECK:STDOUT:   %int_0.loc8_37: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc8_37.3: ref %i32 = array_index %.loc8_1, %int_0.loc8_37
// CHECK:STDOUT:   %.loc8_37.4: init %i32 to %.loc8_37.3 = in_place_init %.loc8_37.2 [concrete = constants.%int_0.fed]
// CHECK:STDOUT:   %.loc8_37.5: init %array_type to %.loc8_1 = array_init (%.loc8_37.4) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc8_39: init %array_type = converted %.loc8_37.1, %.loc8_37.5 [concrete = constants.%array]
// CHECK:STDOUT:   assign file.%tuple_index.var, %.loc8_39
// CHECK:STDOUT:   %int_0.loc10_37: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc10_39.1: %tuple.type.985 = tuple_literal (%int_0.loc10_37) [concrete = constants.%tuple.4f2]
// CHECK:STDOUT:   %i32.loc10: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %int_3.loc10: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %int_1.loc10: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc10_70.1: %struct_type.a.b = struct_literal (%int_3.loc10, %int_1.loc10) [concrete = constants.%struct.a81]
// CHECK:STDOUT:   %struct.loc10: %struct_type.a.b = struct_value (%int_3.loc10, %int_1.loc10) [concrete = constants.%struct.a81]
// CHECK:STDOUT:   %.loc10_70.2: %struct_type.a.b = converted %.loc10_70.1, %struct.loc10 [concrete = constants.%struct.a81]
// CHECK:STDOUT:   %.loc10_71: Core.IntLiteral = struct_access %.loc10_70.2, element1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %array_type.loc10: type = array_type %.loc10_71, %i32.loc10 [concrete = constants.%array_type]
// CHECK:STDOUT:   %impl.elem0.loc10: %.cbf = impl_witness_access constants.%ImplicitAs.impl_witness.f4c, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5]
// CHECK:STDOUT:   %bound_method.loc10_39.1: <bound method> = bound_method %int_0.loc10_37, %impl.elem0.loc10 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.ce0]
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_39.2: <bound method> = bound_method %int_0.loc10_37, %specific_fn.loc10 [concrete = constants.%bound_method.e76]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10: init %i32 = call %bound_method.loc10_39.2(%int_0.loc10_37) [concrete = constants.%int_0.fed]
// CHECK:STDOUT:   %.loc10_39.2: init %i32 = converted %int_0.loc10_37, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10 [concrete = constants.%int_0.fed]
// CHECK:STDOUT:   %.loc10_1: ref %array_type = splice_block file.%struct_access.var [concrete = file.%struct_access.var] {}
// CHECK:STDOUT:   %int_0.loc10_39: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc10_39.3: ref %i32 = array_index %.loc10_1, %int_0.loc10_39
// CHECK:STDOUT:   %.loc10_39.4: init %i32 to %.loc10_39.3 = in_place_init %.loc10_39.2 [concrete = constants.%int_0.fed]
// CHECK:STDOUT:   %.loc10_39.5: init %array_type to %.loc10_1 = array_init (%.loc10_39.4) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc10_41: init %array_type = converted %.loc10_39.1, %.loc10_39.5 [concrete = constants.%array]
// CHECK:STDOUT:   assign file.%struct_access.var, %.loc10_41
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_array_temporary.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %array_type.850: type = array_type %int_1.5b8, %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.e1f: type = pattern_type %array_type.850 [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %tuple.type.985: type = tuple_type (Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.4f2: %tuple.type.985 = tuple_value (%int_0) [concrete]
// CHECK:STDOUT:   %int_5.64b: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %int_7.29f: Core.IntLiteral = int_value 7 [concrete]
// CHECK:STDOUT:   %int_9.988: Core.IntLiteral = int_value 9 [concrete]
// CHECK:STDOUT:   %tuple.type.d46: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.869: %tuple.type.d46 = tuple_value (%int_5.64b, %int_7.29f, %int_1.5b8, %int_9.988) [concrete]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %array_type.475: type = array_type %int_4, %i32 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ef: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.aad: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.abb: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.aad = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.f4c: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.e6c, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.9ca: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.9ca = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.0ef = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.f4c) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.61c: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.cbf: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.61c, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.e8f: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.469: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_5.bb1: %i32 = int_value 5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.0bb: <bound method> = bound_method %int_7.29f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5 [concrete]
// CHECK:STDOUT:   %bound_method.48c: <bound method> = bound_method %int_7.29f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_7.075: %i32 = int_value 7 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.0b7: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5 [concrete]
// CHECK:STDOUT:   %bound_method.1c4: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.d72: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.fc6: <bound method> = bound_method %int_9.988, %Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5 [concrete]
// CHECK:STDOUT:   %bound_method.86f: <bound method> = bound_method %int_9.988, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_9.839: %i32 = int_value 9 [concrete]
// CHECK:STDOUT:   %array: %array_type.475 = tuple_value (%int_5.bb1, %int_7.075, %int_1.d72, %int_9.839) [concrete]
// CHECK:STDOUT:   %.d3a: ref %array_type.475 = temporary invalid, %array [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.099: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5 [concrete]
// CHECK:STDOUT:   %bound_method.c74: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.cb7: %i32 = int_value 2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.fa9: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.aad) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.abb)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.e6c = impl_witness_table (%Core.import_ref.fa9), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %array_index.patt: %pattern_type.e1f = ref_binding_pattern array_index [concrete]
// CHECK:STDOUT:     %array_index.var_patt: %pattern_type.e1f = var_pattern %array_index.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %array_index.var: ref %array_type.850 = var %array_index.var_patt [concrete]
// CHECK:STDOUT:   %.loc9: type = splice_block %array_type [concrete = constants.%array_type.850] {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %array_type: type = array_type %int_1, %i32 [concrete = constants.%array_type.850]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %array_index: ref %array_type.850 = ref_binding array_index, %array_index.var [concrete = %array_index.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_0.loc9_35: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc9_37: %tuple.type.985 = tuple_literal (%int_0.loc9_35) [concrete = constants.%tuple.4f2]
// CHECK:STDOUT:   %i32.loc9_48: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5.64b]
// CHECK:STDOUT:   %int_7: Core.IntLiteral = int_value 7 [concrete = constants.%int_7.29f]
// CHECK:STDOUT:   %int_1.loc9_61: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_9: Core.IntLiteral = int_value 9 [concrete = constants.%int_9.988]
// CHECK:STDOUT:   %.loc9_65.1: %tuple.type.d46 = tuple_literal (%int_5, %int_7, %int_1.loc9_61, %int_9) [concrete = constants.%tuple.869]
// CHECK:STDOUT:   %i32.loc9_76: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4]
// CHECK:STDOUT:   %array_type: type = array_type %int_4, %i32.loc9_76 [concrete = constants.%array_type.475]
// CHECK:STDOUT:   %impl.elem0.loc9_65.1: %.cbf = impl_witness_access constants.%ImplicitAs.impl_witness.f4c, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5]
// CHECK:STDOUT:   %bound_method.loc9_65.1: <bound method> = bound_method %int_5, %impl.elem0.loc9_65.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.e8f]
// CHECK:STDOUT:   %specific_fn.loc9_65.1: <specific function> = specific_function %impl.elem0.loc9_65.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_65.2: <bound method> = bound_method %int_5, %specific_fn.loc9_65.1 [concrete = constants.%bound_method.469]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_65.1: init %i32 = call %bound_method.loc9_65.2(%int_5) [concrete = constants.%int_5.bb1]
// CHECK:STDOUT:   %.loc9_65.2: init %i32 = converted %int_5, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_65.1 [concrete = constants.%int_5.bb1]
// CHECK:STDOUT:   %.loc9_65.3: ref %array_type.475 = temporary_storage
// CHECK:STDOUT:   %int_0.loc9_65: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc9_65.4: ref %i32 = array_index %.loc9_65.3, %int_0.loc9_65
// CHECK:STDOUT:   %.loc9_65.5: init %i32 to %.loc9_65.4 = in_place_init %.loc9_65.2 [concrete = constants.%int_5.bb1]
// CHECK:STDOUT:   %impl.elem0.loc9_65.2: %.cbf = impl_witness_access constants.%ImplicitAs.impl_witness.f4c, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5]
// CHECK:STDOUT:   %bound_method.loc9_65.3: <bound method> = bound_method %int_7, %impl.elem0.loc9_65.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.0bb]
// CHECK:STDOUT:   %specific_fn.loc9_65.2: <specific function> = specific_function %impl.elem0.loc9_65.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_65.4: <bound method> = bound_method %int_7, %specific_fn.loc9_65.2 [concrete = constants.%bound_method.48c]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_65.2: init %i32 = call %bound_method.loc9_65.4(%int_7) [concrete = constants.%int_7.075]
// CHECK:STDOUT:   %.loc9_65.6: init %i32 = converted %int_7, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_65.2 [concrete = constants.%int_7.075]
// CHECK:STDOUT:   %int_1.loc9_65: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc9_65.7: ref %i32 = array_index %.loc9_65.3, %int_1.loc9_65
// CHECK:STDOUT:   %.loc9_65.8: init %i32 to %.loc9_65.7 = in_place_init %.loc9_65.6 [concrete = constants.%int_7.075]
// CHECK:STDOUT:   %impl.elem0.loc9_65.3: %.cbf = impl_witness_access constants.%ImplicitAs.impl_witness.f4c, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5]
// CHECK:STDOUT:   %bound_method.loc9_65.5: <bound method> = bound_method %int_1.loc9_61, %impl.elem0.loc9_65.3 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.0b7]
// CHECK:STDOUT:   %specific_fn.loc9_65.3: <specific function> = specific_function %impl.elem0.loc9_65.3, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_65.6: <bound method> = bound_method %int_1.loc9_61, %specific_fn.loc9_65.3 [concrete = constants.%bound_method.1c4]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_65.3: init %i32 = call %bound_method.loc9_65.6(%int_1.loc9_61) [concrete = constants.%int_1.d72]
// CHECK:STDOUT:   %.loc9_65.9: init %i32 = converted %int_1.loc9_61, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_65.3 [concrete = constants.%int_1.d72]
// CHECK:STDOUT:   %int_2.loc9_65: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc9_65.10: ref %i32 = array_index %.loc9_65.3, %int_2.loc9_65
// CHECK:STDOUT:   %.loc9_65.11: init %i32 to %.loc9_65.10 = in_place_init %.loc9_65.9 [concrete = constants.%int_1.d72]
// CHECK:STDOUT:   %impl.elem0.loc9_65.4: %.cbf = impl_witness_access constants.%ImplicitAs.impl_witness.f4c, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5]
// CHECK:STDOUT:   %bound_method.loc9_65.7: <bound method> = bound_method %int_9, %impl.elem0.loc9_65.4 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.fc6]
// CHECK:STDOUT:   %specific_fn.loc9_65.4: <specific function> = specific_function %impl.elem0.loc9_65.4, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_65.8: <bound method> = bound_method %int_9, %specific_fn.loc9_65.4 [concrete = constants.%bound_method.86f]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_65.4: init %i32 = call %bound_method.loc9_65.8(%int_9) [concrete = constants.%int_9.839]
// CHECK:STDOUT:   %.loc9_65.12: init %i32 = converted %int_9, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_65.4 [concrete = constants.%int_9.839]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:   %.loc9_65.13: ref %i32 = array_index %.loc9_65.3, %int_3
// CHECK:STDOUT:   %.loc9_65.14: init %i32 to %.loc9_65.13 = in_place_init %.loc9_65.12 [concrete = constants.%int_9.839]
// CHECK:STDOUT:   %.loc9_65.15: init %array_type.475 to %.loc9_65.3 = array_init (%.loc9_65.5, %.loc9_65.8, %.loc9_65.11, %.loc9_65.14) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc9_67.1: init %array_type.475 = converted %.loc9_65.1, %.loc9_65.15 [concrete = constants.%array]
// CHECK:STDOUT:   %int_2.loc9_85: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc9_67.2: ref %array_type.475 = temporary %.loc9_65.3, %.loc9_67.1 [concrete = constants.%.d3a]
// CHECK:STDOUT:   %impl.elem0.loc9_85: %.cbf = impl_witness_access constants.%ImplicitAs.impl_witness.f4c, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5]
// CHECK:STDOUT:   %bound_method.loc9_85.1: <bound method> = bound_method %int_2.loc9_85, %impl.elem0.loc9_85 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.099]
// CHECK:STDOUT:   %specific_fn.loc9_85: <specific function> = specific_function %impl.elem0.loc9_85, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_85.2: <bound method> = bound_method %int_2.loc9_85, %specific_fn.loc9_85 [concrete = constants.%bound_method.c74]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_85: init %i32 = call %bound_method.loc9_85.2(%int_2.loc9_85) [concrete = constants.%int_2.cb7]
// CHECK:STDOUT:   %.loc9_85.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_85 [concrete = constants.%int_2.cb7]
// CHECK:STDOUT:   %.loc9_85.2: %i32 = converted %int_2.loc9_85, %.loc9_85.1 [concrete = constants.%int_2.cb7]
// CHECK:STDOUT:   %.loc9_86.1: ref %i32 = array_index %.loc9_67.2, %.loc9_85.2
// CHECK:STDOUT:   %.loc9_86.2: %i32 = acquire_value %.loc9_86.1
// CHECK:STDOUT:   assign file.%array_index.var, <error>
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- symbolic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.761: %Destroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %T.as_type.415: type = facet_access_type %T.761 [symbolic]
// CHECK:STDOUT:   %ptr.f64: type = ptr_type %T.as_type.415 [symbolic]
// CHECK:STDOUT:   %const: type = const_type %T.as_type.415 [symbolic]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.24b = tuple_value (%ptr.f64, %const) [symbolic]
// CHECK:STDOUT:   %tuple.type.b7c: type = tuple_type (%ptr.f64, %const) [symbolic]
// CHECK:STDOUT:   %require_complete.5af: <witness> = require_complete_type %tuple.type.b7c [symbolic]
// CHECK:STDOUT:   %pattern_type.b89: type = pattern_type %tuple.type.b7c [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.lookup_impl_witness.505: <witness> = lookup_impl_witness %tuple.type.b7c, @DefaultOrUnformed [symbolic]
// CHECK:STDOUT:   %.a07: require_specific_def_type = require_specific_def @T.as.DefaultOrUnformed.impl(%tuple.type.b7c) [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.be9: %DefaultOrUnformed.type = facet_value %tuple.type.b7c, (%DefaultOrUnformed.lookup_impl_witness.505) [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.WithSelf.Op.type.0ff: type = fn_type @DefaultOrUnformed.WithSelf.Op, @DefaultOrUnformed.WithSelf(%DefaultOrUnformed.facet.be9) [symbolic]
// CHECK:STDOUT:   %.98a: type = fn_type_with_self_type %DefaultOrUnformed.WithSelf.Op.type.0ff, %DefaultOrUnformed.facet.be9 [symbolic]
// CHECK:STDOUT:   %impl.elem0.831: %.98a = impl_witness_access %DefaultOrUnformed.lookup_impl_witness.505, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.bcb: <specific function> = specific_impl_function %impl.elem0.831, @DefaultOrUnformed.WithSelf.Op(%DefaultOrUnformed.facet.be9) [symbolic]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %T.as_type.415} [symbolic]
// CHECK:STDOUT:   %require_complete.87f: <witness> = require_complete_type %struct_type.a [symbolic]
// CHECK:STDOUT:   %pattern_type.9ba: type = pattern_type %struct_type.a [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.lookup_impl_witness.9af: <witness> = lookup_impl_witness %struct_type.a, @DefaultOrUnformed [symbolic]
// CHECK:STDOUT:   %.7d8: require_specific_def_type = require_specific_def @T.as.DefaultOrUnformed.impl(%struct_type.a) [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.0c9: %DefaultOrUnformed.type = facet_value %struct_type.a, (%DefaultOrUnformed.lookup_impl_witness.9af) [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.WithSelf.Op.type.411: type = fn_type @DefaultOrUnformed.WithSelf.Op, @DefaultOrUnformed.WithSelf(%DefaultOrUnformed.facet.0c9) [symbolic]
// CHECK:STDOUT:   %.d72: type = fn_type_with_self_type %DefaultOrUnformed.WithSelf.Op.type.411, %DefaultOrUnformed.facet.0c9 [symbolic]
// CHECK:STDOUT:   %impl.elem0.bc2: %.d72 = impl_witness_access %DefaultOrUnformed.lookup_impl_witness.9af, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.219: <specific function> = specific_impl_function %impl.elem0.bc2, @DefaultOrUnformed.WithSelf.Op(%DefaultOrUnformed.facet.0c9) [symbolic]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %array_type.3df: type = array_type %int_5, %T.as_type.415 [symbolic]
// CHECK:STDOUT:   %require_complete.743: <witness> = require_complete_type %array_type.3df [symbolic]
// CHECK:STDOUT:   %pattern_type.ddf: type = pattern_type %array_type.3df [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.lookup_impl_witness.04e: <witness> = lookup_impl_witness %array_type.3df, @DefaultOrUnformed [symbolic]
// CHECK:STDOUT:   %.b27: require_specific_def_type = require_specific_def @T.as.DefaultOrUnformed.impl(%array_type.3df) [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.b87: %DefaultOrUnformed.type = facet_value %array_type.3df, (%DefaultOrUnformed.lookup_impl_witness.04e) [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.WithSelf.Op.type.e2e: type = fn_type @DefaultOrUnformed.WithSelf.Op, @DefaultOrUnformed.WithSelf(%DefaultOrUnformed.facet.b87) [symbolic]
// CHECK:STDOUT:   %.bc3: type = fn_type_with_self_type %DefaultOrUnformed.WithSelf.Op.type.e2e, %DefaultOrUnformed.facet.b87 [symbolic]
// CHECK:STDOUT:   %impl.elem0.43c: %.bc3 = impl_witness_access %DefaultOrUnformed.lookup_impl_witness.04e, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.5e3: <specific function> = specific_impl_function %impl.elem0.43c, @DefaultOrUnformed.WithSelf.Op(%DefaultOrUnformed.facet.b87) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.499: <witness> = lookup_impl_witness %array_type.3df, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.151: %Destroy.type = facet_value %array_type.3df, (%Destroy.lookup_impl_witness.499) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.991: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.151) [symbolic]
// CHECK:STDOUT:   %.c90: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.991, %Destroy.facet.151 [symbolic]
// CHECK:STDOUT:   %impl.elem0.cac: %.c90 = impl_witness_access %Destroy.lookup_impl_witness.499, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.c68: <specific function> = specific_impl_function %impl.elem0.cac, @Destroy.WithSelf.Op(%Destroy.facet.151) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.b08: <witness> = lookup_impl_witness %struct_type.a, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.525: %Destroy.type = facet_value %struct_type.a, (%Destroy.lookup_impl_witness.b08) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.a8e: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.525) [symbolic]
// CHECK:STDOUT:   %.343: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.a8e, %Destroy.facet.525 [symbolic]
// CHECK:STDOUT:   %impl.elem0.4e5: %.343 = impl_witness_access %Destroy.lookup_impl_witness.b08, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.730: <specific function> = specific_impl_function %impl.elem0.4e5, @Destroy.WithSelf.Op(%Destroy.facet.525) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.b88: <witness> = lookup_impl_witness %tuple.type.b7c, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.dc1: %Destroy.type = facet_value %tuple.type.b7c, (%Destroy.lookup_impl_witness.b88) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.156: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.dc1) [symbolic]
// CHECK:STDOUT:   %.225: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.156, %Destroy.facet.dc1 [symbolic]
// CHECK:STDOUT:   %impl.elem0.e8c: %.225 = impl_witness_access %Destroy.lookup_impl_witness.b88, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.883: <specific function> = specific_impl_function %impl.elem0.e8c, @Destroy.WithSelf.Op(%Destroy.facet.dc1) [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %N.921: %i32 = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.ab8: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %From: Core.IntLiteral = symbolic_binding From, 0 [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.d24: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%From) [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.8bc: %Int.as.ImplicitAs.impl.Convert.type.d24 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.f73: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.94d, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.25b: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.338: %Int.as.ImplicitAs.impl.Convert.type.25b = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.ab8 = facet_value %i32, (%ImplicitAs.impl_witness.f73) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.abd: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.d21: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.abd, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %N.921, %Int.as.ImplicitAs.impl.Convert.338 [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Int.as.ImplicitAs.impl.Convert.338, @Int.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %N.921, %Int.as.ImplicitAs.impl.Convert.specific_fn [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method(%N.921) [symbolic]
// CHECK:STDOUT:   %array_type.b1a: type = array_type %Int.as.ImplicitAs.impl.Convert.call, %i32 [symbolic]
// CHECK:STDOUT:   %require_complete.553: <witness> = require_complete_type %array_type.b1a [symbolic]
// CHECK:STDOUT:   %pattern_type.bc6: type = pattern_type %array_type.b1a [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.lookup_impl_witness.dfc: <witness> = lookup_impl_witness %array_type.b1a, @DefaultOrUnformed [symbolic]
// CHECK:STDOUT:   %.174: require_specific_def_type = require_specific_def @T.as.DefaultOrUnformed.impl(%array_type.b1a) [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.fb3: %DefaultOrUnformed.type = facet_value %array_type.b1a, (%DefaultOrUnformed.lookup_impl_witness.dfc) [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.WithSelf.Op.type.554: type = fn_type @DefaultOrUnformed.WithSelf.Op, @DefaultOrUnformed.WithSelf(%DefaultOrUnformed.facet.fb3) [symbolic]
// CHECK:STDOUT:   %.4b5: type = fn_type_with_self_type %DefaultOrUnformed.WithSelf.Op.type.554, %DefaultOrUnformed.facet.fb3 [symbolic]
// CHECK:STDOUT:   %impl.elem0.291: %.4b5 = impl_witness_access %DefaultOrUnformed.lookup_impl_witness.dfc, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.2ee: <specific function> = specific_impl_function %impl.elem0.291, @DefaultOrUnformed.WithSelf.Op(%DefaultOrUnformed.facet.fb3) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.397: <witness> = lookup_impl_witness %array_type.b1a, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.9a6: %Destroy.type = facet_value %array_type.b1a, (%Destroy.lookup_impl_witness.397) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.6ac: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.9a6) [symbolic]
// CHECK:STDOUT:   %.c61: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.6ac, %Destroy.facet.9a6 [symbolic]
// CHECK:STDOUT:   %impl.elem0.e0c: %.c61 = impl_witness_access %Destroy.lookup_impl_witness.397, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.028: <specific function> = specific_impl_function %impl.elem0.e0c, @Destroy.WithSelf.Op(%Destroy.facet.9a6) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.2cc: @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert.type (%Int.as.ImplicitAs.impl.Convert.type.d24) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert (constants.%Int.as.ImplicitAs.impl.Convert.8bc)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.94d = impl_witness_table (%Core.import_ref.2cc), @Int.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc4_7.2: %Destroy.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc6_19.2: type = facet_access_type %T.loc4_7.1 [symbolic = %T.as_type.loc6_19.2 (constants.%T.as_type.415)]
// CHECK:STDOUT:   %ptr.loc6_19.2: type = ptr_type %T.as_type.loc6_19.2 [symbolic = %ptr.loc6_19.2 (constants.%ptr.f64)]
// CHECK:STDOUT:   %const.loc6_22.2: type = const_type %T.as_type.loc6_19.2 [symbolic = %const.loc6_22.2 (constants.%const)]
// CHECK:STDOUT:   %tuple: %tuple.type.24b = tuple_value (%ptr.loc6_19.2, %const.loc6_22.2) [symbolic = %tuple (constants.%tuple)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%ptr.loc6_19.2, %const.loc6_22.2) [symbolic = %tuple.type (constants.%tuple.type.b7c)]
// CHECK:STDOUT:   %require_complete.loc6: <witness> = require_complete_type %tuple.type [symbolic = %require_complete.loc6 (constants.%require_complete.5af)]
// CHECK:STDOUT:   %pattern_type.loc6: type = pattern_type %tuple.type [symbolic = %pattern_type.loc6 (constants.%pattern_type.b89)]
// CHECK:STDOUT:   %.loc6_30.3: require_specific_def_type = require_specific_def @T.as.DefaultOrUnformed.impl(%tuple.type) [symbolic = %.loc6_30.3 (constants.%.a07)]
// CHECK:STDOUT:   %DefaultOrUnformed.lookup_impl_witness.loc6: <witness> = lookup_impl_witness %tuple.type, @DefaultOrUnformed [symbolic = %DefaultOrUnformed.lookup_impl_witness.loc6 (constants.%DefaultOrUnformed.lookup_impl_witness.505)]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc6_30.2: %DefaultOrUnformed.type = facet_value %tuple.type, (%DefaultOrUnformed.lookup_impl_witness.loc6) [symbolic = %DefaultOrUnformed.facet.loc6_30.2 (constants.%DefaultOrUnformed.facet.be9)]
// CHECK:STDOUT:   %DefaultOrUnformed.WithSelf.Op.type.loc6: type = fn_type @DefaultOrUnformed.WithSelf.Op, @DefaultOrUnformed.WithSelf(%DefaultOrUnformed.facet.loc6_30.2) [symbolic = %DefaultOrUnformed.WithSelf.Op.type.loc6 (constants.%DefaultOrUnformed.WithSelf.Op.type.0ff)]
// CHECK:STDOUT:   %.loc6_30.4: type = fn_type_with_self_type %DefaultOrUnformed.WithSelf.Op.type.loc6, %DefaultOrUnformed.facet.loc6_30.2 [symbolic = %.loc6_30.4 (constants.%.98a)]
// CHECK:STDOUT:   %impl.elem0.loc6_30.2: @F.%.loc6_30.4 (%.98a) = impl_witness_access %DefaultOrUnformed.lookup_impl_witness.loc6, element0 [symbolic = %impl.elem0.loc6_30.2 (constants.%impl.elem0.831)]
// CHECK:STDOUT:   %specific_impl_fn.loc6_30.2: <specific function> = specific_impl_function %impl.elem0.loc6_30.2, @DefaultOrUnformed.WithSelf.Op(%DefaultOrUnformed.facet.loc6_30.2) [symbolic = %specific_impl_fn.loc6_30.2 (constants.%specific_impl_fn.bcb)]
// CHECK:STDOUT:   %struct_type.a.loc7_23.2: type = struct_type {.a: @F.%T.as_type.loc6_19.2 (%T.as_type.415)} [symbolic = %struct_type.a.loc7_23.2 (constants.%struct_type.a)]
// CHECK:STDOUT:   %require_complete.loc7: <witness> = require_complete_type %struct_type.a.loc7_23.2 [symbolic = %require_complete.loc7 (constants.%require_complete.87f)]
// CHECK:STDOUT:   %pattern_type.loc7: type = pattern_type %struct_type.a.loc7_23.2 [symbolic = %pattern_type.loc7 (constants.%pattern_type.9ba)]
// CHECK:STDOUT:   %.loc7_24.3: require_specific_def_type = require_specific_def @T.as.DefaultOrUnformed.impl(%struct_type.a.loc7_23.2) [symbolic = %.loc7_24.3 (constants.%.7d8)]
// CHECK:STDOUT:   %DefaultOrUnformed.lookup_impl_witness.loc7: <witness> = lookup_impl_witness %struct_type.a.loc7_23.2, @DefaultOrUnformed [symbolic = %DefaultOrUnformed.lookup_impl_witness.loc7 (constants.%DefaultOrUnformed.lookup_impl_witness.9af)]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc7_24.2: %DefaultOrUnformed.type = facet_value %struct_type.a.loc7_23.2, (%DefaultOrUnformed.lookup_impl_witness.loc7) [symbolic = %DefaultOrUnformed.facet.loc7_24.2 (constants.%DefaultOrUnformed.facet.0c9)]
// CHECK:STDOUT:   %DefaultOrUnformed.WithSelf.Op.type.loc7: type = fn_type @DefaultOrUnformed.WithSelf.Op, @DefaultOrUnformed.WithSelf(%DefaultOrUnformed.facet.loc7_24.2) [symbolic = %DefaultOrUnformed.WithSelf.Op.type.loc7 (constants.%DefaultOrUnformed.WithSelf.Op.type.411)]
// CHECK:STDOUT:   %.loc7_24.4: type = fn_type_with_self_type %DefaultOrUnformed.WithSelf.Op.type.loc7, %DefaultOrUnformed.facet.loc7_24.2 [symbolic = %.loc7_24.4 (constants.%.d72)]
// CHECK:STDOUT:   %impl.elem0.loc7_24.2: @F.%.loc7_24.4 (%.d72) = impl_witness_access %DefaultOrUnformed.lookup_impl_witness.loc7, element0 [symbolic = %impl.elem0.loc7_24.2 (constants.%impl.elem0.bc2)]
// CHECK:STDOUT:   %specific_impl_fn.loc7_24.2: <specific function> = specific_impl_function %impl.elem0.loc7_24.2, @DefaultOrUnformed.WithSelf.Op(%DefaultOrUnformed.facet.loc7_24.2) [symbolic = %specific_impl_fn.loc7_24.2 (constants.%specific_impl_fn.219)]
// CHECK:STDOUT:   %array_type.loc8_27.2: type = array_type constants.%int_5, %T.as_type.loc6_19.2 [symbolic = %array_type.loc8_27.2 (constants.%array_type.3df)]
// CHECK:STDOUT:   %require_complete.loc8: <witness> = require_complete_type %array_type.loc8_27.2 [symbolic = %require_complete.loc8 (constants.%require_complete.743)]
// CHECK:STDOUT:   %pattern_type.loc8: type = pattern_type %array_type.loc8_27.2 [symbolic = %pattern_type.loc8 (constants.%pattern_type.ddf)]
// CHECK:STDOUT:   %.loc8_28.3: require_specific_def_type = require_specific_def @T.as.DefaultOrUnformed.impl(%array_type.loc8_27.2) [symbolic = %.loc8_28.3 (constants.%.b27)]
// CHECK:STDOUT:   %DefaultOrUnformed.lookup_impl_witness.loc8: <witness> = lookup_impl_witness %array_type.loc8_27.2, @DefaultOrUnformed [symbolic = %DefaultOrUnformed.lookup_impl_witness.loc8 (constants.%DefaultOrUnformed.lookup_impl_witness.04e)]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc8_28.2: %DefaultOrUnformed.type = facet_value %array_type.loc8_27.2, (%DefaultOrUnformed.lookup_impl_witness.loc8) [symbolic = %DefaultOrUnformed.facet.loc8_28.2 (constants.%DefaultOrUnformed.facet.b87)]
// CHECK:STDOUT:   %DefaultOrUnformed.WithSelf.Op.type.loc8: type = fn_type @DefaultOrUnformed.WithSelf.Op, @DefaultOrUnformed.WithSelf(%DefaultOrUnformed.facet.loc8_28.2) [symbolic = %DefaultOrUnformed.WithSelf.Op.type.loc8 (constants.%DefaultOrUnformed.WithSelf.Op.type.e2e)]
// CHECK:STDOUT:   %.loc8_28.4: type = fn_type_with_self_type %DefaultOrUnformed.WithSelf.Op.type.loc8, %DefaultOrUnformed.facet.loc8_28.2 [symbolic = %.loc8_28.4 (constants.%.bc3)]
// CHECK:STDOUT:   %impl.elem0.loc8_28.2: @F.%.loc8_28.4 (%.bc3) = impl_witness_access %DefaultOrUnformed.lookup_impl_witness.loc8, element0 [symbolic = %impl.elem0.loc8_28.2 (constants.%impl.elem0.43c)]
// CHECK:STDOUT:   %specific_impl_fn.loc8_28.2: <specific function> = specific_impl_function %impl.elem0.loc8_28.2, @DefaultOrUnformed.WithSelf.Op(%DefaultOrUnformed.facet.loc8_28.2) [symbolic = %specific_impl_fn.loc8_28.2 (constants.%specific_impl_fn.5e3)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc8: <witness> = lookup_impl_witness %array_type.loc8_27.2, @Destroy [symbolic = %Destroy.lookup_impl_witness.loc8 (constants.%Destroy.lookup_impl_witness.499)]
// CHECK:STDOUT:   %Destroy.facet.loc8: %Destroy.type = facet_value %array_type.loc8_27.2, (%Destroy.lookup_impl_witness.loc8) [symbolic = %Destroy.facet.loc8 (constants.%Destroy.facet.151)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc8: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc8) [symbolic = %Destroy.WithSelf.Op.type.loc8 (constants.%Destroy.WithSelf.Op.type.991)]
// CHECK:STDOUT:   %.loc8_3.2: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.loc8, %Destroy.facet.loc8 [symbolic = %.loc8_3.2 (constants.%.c90)]
// CHECK:STDOUT:   %impl.elem0.loc8_3.2: @F.%.loc8_3.2 (%.c90) = impl_witness_access %Destroy.lookup_impl_witness.loc8, element0 [symbolic = %impl.elem0.loc8_3.2 (constants.%impl.elem0.cac)]
// CHECK:STDOUT:   %specific_impl_fn.loc8_3.2: <specific function> = specific_impl_function %impl.elem0.loc8_3.2, @Destroy.WithSelf.Op(%Destroy.facet.loc8) [symbolic = %specific_impl_fn.loc8_3.2 (constants.%specific_impl_fn.c68)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc7: <witness> = lookup_impl_witness %struct_type.a.loc7_23.2, @Destroy [symbolic = %Destroy.lookup_impl_witness.loc7 (constants.%Destroy.lookup_impl_witness.b08)]
// CHECK:STDOUT:   %Destroy.facet.loc7: %Destroy.type = facet_value %struct_type.a.loc7_23.2, (%Destroy.lookup_impl_witness.loc7) [symbolic = %Destroy.facet.loc7 (constants.%Destroy.facet.525)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc7: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc7) [symbolic = %Destroy.WithSelf.Op.type.loc7 (constants.%Destroy.WithSelf.Op.type.a8e)]
// CHECK:STDOUT:   %.loc7_3: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.loc7, %Destroy.facet.loc7 [symbolic = %.loc7_3 (constants.%.343)]
// CHECK:STDOUT:   %impl.elem0.loc7_3.2: @F.%.loc7_3 (%.343) = impl_witness_access %Destroy.lookup_impl_witness.loc7, element0 [symbolic = %impl.elem0.loc7_3.2 (constants.%impl.elem0.4e5)]
// CHECK:STDOUT:   %specific_impl_fn.loc7_3.2: <specific function> = specific_impl_function %impl.elem0.loc7_3.2, @Destroy.WithSelf.Op(%Destroy.facet.loc7) [symbolic = %specific_impl_fn.loc7_3.2 (constants.%specific_impl_fn.730)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc6: <witness> = lookup_impl_witness %tuple.type, @Destroy [symbolic = %Destroy.lookup_impl_witness.loc6 (constants.%Destroy.lookup_impl_witness.b88)]
// CHECK:STDOUT:   %Destroy.facet.loc6: %Destroy.type = facet_value %tuple.type, (%Destroy.lookup_impl_witness.loc6) [symbolic = %Destroy.facet.loc6 (constants.%Destroy.facet.dc1)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc6: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc6) [symbolic = %Destroy.WithSelf.Op.type.loc6 (constants.%Destroy.WithSelf.Op.type.156)]
// CHECK:STDOUT:   %.loc6_3.2: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.loc6, %Destroy.facet.loc6 [symbolic = %.loc6_3.2 (constants.%.225)]
// CHECK:STDOUT:   %impl.elem0.loc6_3.2: @F.%.loc6_3.2 (%.225) = impl_witness_access %Destroy.lookup_impl_witness.loc6, element0 [symbolic = %impl.elem0.loc6_3.2 (constants.%impl.elem0.e8c)]
// CHECK:STDOUT:   %specific_impl_fn.loc6_3.2: <specific function> = specific_impl_function %impl.elem0.loc6_3.2, @Destroy.WithSelf.Op(%Destroy.facet.loc6) [symbolic = %specific_impl_fn.loc6_3.2 (constants.%specific_impl_fn.883)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %u.patt: @F.%pattern_type.loc6 (%pattern_type.b89) = ref_binding_pattern u [concrete]
// CHECK:STDOUT:       %u.var_patt: @F.%pattern_type.loc6 (%pattern_type.b89) = var_pattern %u.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u.var: ref @F.%tuple.type (%tuple.type.b7c) = var %u.var_patt
// CHECK:STDOUT:     %DefaultOrUnformed.facet.loc6_30.1: %DefaultOrUnformed.type = facet_value constants.%tuple.type.b7c, (constants.%DefaultOrUnformed.lookup_impl_witness.505) [symbolic = %DefaultOrUnformed.facet.loc6_30.2 (constants.%DefaultOrUnformed.facet.be9)]
// CHECK:STDOUT:     %.loc6_30.1: %DefaultOrUnformed.type = converted constants.%tuple.type.b7c, %DefaultOrUnformed.facet.loc6_30.1 [symbolic = %DefaultOrUnformed.facet.loc6_30.2 (constants.%DefaultOrUnformed.facet.be9)]
// CHECK:STDOUT:     %as_type.loc6: type = facet_access_type %.loc6_30.1 [symbolic = %tuple.type (constants.%tuple.type.b7c)]
// CHECK:STDOUT:     %.loc6_30.2: type = converted %.loc6_30.1, %as_type.loc6 [symbolic = %tuple.type (constants.%tuple.type.b7c)]
// CHECK:STDOUT:     %impl.elem0.loc6_30.1: @F.%.loc6_30.4 (%.98a) = impl_witness_access constants.%DefaultOrUnformed.lookup_impl_witness.505, element0 [symbolic = %impl.elem0.loc6_30.2 (constants.%impl.elem0.831)]
// CHECK:STDOUT:     %specific_impl_fn.loc6_30.1: <specific function> = specific_impl_function %impl.elem0.loc6_30.1, @DefaultOrUnformed.WithSelf.Op(constants.%DefaultOrUnformed.facet.be9) [symbolic = %specific_impl_fn.loc6_30.2 (constants.%specific_impl_fn.bcb)]
// CHECK:STDOUT:     %.loc6_3.1: ref @F.%tuple.type (%tuple.type.b7c) = splice_block %u.var {}
// CHECK:STDOUT:     %DefaultOrUnformed.WithSelf.Op.call.loc6: init @F.%tuple.type (%tuple.type.b7c) to %.loc6_3.1 = call %specific_impl_fn.loc6_30.1()
// CHECK:STDOUT:     assign %u.var, %DefaultOrUnformed.WithSelf.Op.call.loc6
// CHECK:STDOUT:     %.loc6_29.1: type = splice_block %.loc6_29.3 [symbolic = %tuple.type (constants.%tuple.type.b7c)] {
// CHECK:STDOUT:       %T.ref.loc6_18: %Destroy.type = name_ref T, %T.loc4_7.2 [symbolic = %T.loc4_7.1 (constants.%T.761)]
// CHECK:STDOUT:       %T.as_type.loc6_19.1: type = facet_access_type %T.ref.loc6_18 [symbolic = %T.as_type.loc6_19.2 (constants.%T.as_type.415)]
// CHECK:STDOUT:       %.loc6_19: type = converted %T.ref.loc6_18, %T.as_type.loc6_19.1 [symbolic = %T.as_type.loc6_19.2 (constants.%T.as_type.415)]
// CHECK:STDOUT:       %ptr.loc6_19.1: type = ptr_type %.loc6_19 [symbolic = %ptr.loc6_19.2 (constants.%ptr.f64)]
// CHECK:STDOUT:       %T.ref.loc6_28: %Destroy.type = name_ref T, %T.loc4_7.2 [symbolic = %T.loc4_7.1 (constants.%T.761)]
// CHECK:STDOUT:       %T.as_type.loc6_22: type = facet_access_type %T.ref.loc6_28 [symbolic = %T.as_type.loc6_19.2 (constants.%T.as_type.415)]
// CHECK:STDOUT:       %.loc6_22: type = converted %T.ref.loc6_28, %T.as_type.loc6_22 [symbolic = %T.as_type.loc6_19.2 (constants.%T.as_type.415)]
// CHECK:STDOUT:       %const.loc6_22.1: type = const_type %.loc6_22 [symbolic = %const.loc6_22.2 (constants.%const)]
// CHECK:STDOUT:       %.loc6_29.2: %tuple.type.24b = tuple_literal (%ptr.loc6_19.1, %const.loc6_22.1) [symbolic = %tuple (constants.%tuple)]
// CHECK:STDOUT:       %.loc6_29.3: type = converted %.loc6_29.2, constants.%tuple.type.b7c [symbolic = %tuple.type (constants.%tuple.type.b7c)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: ref @F.%tuple.type (%tuple.type.b7c) = ref_binding u, %u.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %v.patt: @F.%pattern_type.loc7 (%pattern_type.9ba) = ref_binding_pattern v [concrete]
// CHECK:STDOUT:       %v.var_patt: @F.%pattern_type.loc7 (%pattern_type.9ba) = var_pattern %v.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v.var: ref @F.%struct_type.a.loc7_23.2 (%struct_type.a) = var %v.var_patt
// CHECK:STDOUT:     %DefaultOrUnformed.facet.loc7_24.1: %DefaultOrUnformed.type = facet_value constants.%struct_type.a, (constants.%DefaultOrUnformed.lookup_impl_witness.9af) [symbolic = %DefaultOrUnformed.facet.loc7_24.2 (constants.%DefaultOrUnformed.facet.0c9)]
// CHECK:STDOUT:     %.loc7_24.1: %DefaultOrUnformed.type = converted constants.%struct_type.a, %DefaultOrUnformed.facet.loc7_24.1 [symbolic = %DefaultOrUnformed.facet.loc7_24.2 (constants.%DefaultOrUnformed.facet.0c9)]
// CHECK:STDOUT:     %as_type.loc7: type = facet_access_type %.loc7_24.1 [symbolic = %struct_type.a.loc7_23.2 (constants.%struct_type.a)]
// CHECK:STDOUT:     %.loc7_24.2: type = converted %.loc7_24.1, %as_type.loc7 [symbolic = %struct_type.a.loc7_23.2 (constants.%struct_type.a)]
// CHECK:STDOUT:     %impl.elem0.loc7_24.1: @F.%.loc7_24.4 (%.d72) = impl_witness_access constants.%DefaultOrUnformed.lookup_impl_witness.9af, element0 [symbolic = %impl.elem0.loc7_24.2 (constants.%impl.elem0.bc2)]
// CHECK:STDOUT:     %specific_impl_fn.loc7_24.1: <specific function> = specific_impl_function %impl.elem0.loc7_24.1, @DefaultOrUnformed.WithSelf.Op(constants.%DefaultOrUnformed.facet.0c9) [symbolic = %specific_impl_fn.loc7_24.2 (constants.%specific_impl_fn.219)]
// CHECK:STDOUT:     %DefaultOrUnformed.WithSelf.Op.call.loc7: init @F.%struct_type.a.loc7_23.2 (%struct_type.a) = call %specific_impl_fn.loc7_24.1()
// CHECK:STDOUT:     assign %v.var, %DefaultOrUnformed.WithSelf.Op.call.loc7
// CHECK:STDOUT:     %.loc7_23: type = splice_block %struct_type.a.loc7_23.1 [symbolic = %struct_type.a.loc7_23.2 (constants.%struct_type.a)] {
// CHECK:STDOUT:       %T.ref.loc7: %Destroy.type = name_ref T, %T.loc4_7.2 [symbolic = %T.loc4_7.1 (constants.%T.761)]
// CHECK:STDOUT:       %T.as_type.loc7: type = facet_access_type %T.ref.loc7 [symbolic = %T.as_type.loc6_19.2 (constants.%T.as_type.415)]
// CHECK:STDOUT:       %.loc7_22: type = converted %T.ref.loc7, %T.as_type.loc7 [symbolic = %T.as_type.loc6_19.2 (constants.%T.as_type.415)]
// CHECK:STDOUT:       %struct_type.a.loc7_23.1: type = struct_type {.a: @F.%T.as_type.loc6_19.2 (%T.as_type.415)} [symbolic = %struct_type.a.loc7_23.2 (constants.%struct_type.a)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: ref @F.%struct_type.a.loc7_23.2 (%struct_type.a) = ref_binding v, %v.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %w.patt: @F.%pattern_type.loc8 (%pattern_type.ddf) = ref_binding_pattern w [concrete]
// CHECK:STDOUT:       %w.var_patt: @F.%pattern_type.loc8 (%pattern_type.ddf) = var_pattern %w.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %w.var: ref @F.%array_type.loc8_27.2 (%array_type.3df) = var %w.var_patt
// CHECK:STDOUT:     %DefaultOrUnformed.facet.loc8_28.1: %DefaultOrUnformed.type = facet_value constants.%array_type.3df, (constants.%DefaultOrUnformed.lookup_impl_witness.04e) [symbolic = %DefaultOrUnformed.facet.loc8_28.2 (constants.%DefaultOrUnformed.facet.b87)]
// CHECK:STDOUT:     %.loc8_28.1: %DefaultOrUnformed.type = converted constants.%array_type.3df, %DefaultOrUnformed.facet.loc8_28.1 [symbolic = %DefaultOrUnformed.facet.loc8_28.2 (constants.%DefaultOrUnformed.facet.b87)]
// CHECK:STDOUT:     %as_type.loc8: type = facet_access_type %.loc8_28.1 [symbolic = %array_type.loc8_27.2 (constants.%array_type.3df)]
// CHECK:STDOUT:     %.loc8_28.2: type = converted %.loc8_28.1, %as_type.loc8 [symbolic = %array_type.loc8_27.2 (constants.%array_type.3df)]
// CHECK:STDOUT:     %impl.elem0.loc8_28.1: @F.%.loc8_28.4 (%.bc3) = impl_witness_access constants.%DefaultOrUnformed.lookup_impl_witness.04e, element0 [symbolic = %impl.elem0.loc8_28.2 (constants.%impl.elem0.43c)]
// CHECK:STDOUT:     %specific_impl_fn.loc8_28.1: <specific function> = specific_impl_function %impl.elem0.loc8_28.1, @DefaultOrUnformed.WithSelf.Op(constants.%DefaultOrUnformed.facet.b87) [symbolic = %specific_impl_fn.loc8_28.2 (constants.%specific_impl_fn.5e3)]
// CHECK:STDOUT:     %.loc8_3.1: ref @F.%array_type.loc8_27.2 (%array_type.3df) = splice_block %w.var {}
// CHECK:STDOUT:     %DefaultOrUnformed.WithSelf.Op.call.loc8: init @F.%array_type.loc8_27.2 (%array_type.3df) to %.loc8_3.1 = call %specific_impl_fn.loc8_28.1()
// CHECK:STDOUT:     assign %w.var, %DefaultOrUnformed.WithSelf.Op.call.loc8
// CHECK:STDOUT:     %.loc8_27: type = splice_block %array_type.loc8_27.1 [symbolic = %array_type.loc8_27.2 (constants.%array_type.3df)] {
// CHECK:STDOUT:       %T.ref.loc8: %Destroy.type = name_ref T, %T.loc4_7.2 [symbolic = %T.loc4_7.1 (constants.%T.761)]
// CHECK:STDOUT:       %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5]
// CHECK:STDOUT:       %T.as_type.loc8: type = facet_access_type %T.ref.loc8 [symbolic = %T.as_type.loc6_19.2 (constants.%T.as_type.415)]
// CHECK:STDOUT:       %.loc8_23: type = converted %T.ref.loc8, %T.as_type.loc8 [symbolic = %T.as_type.loc6_19.2 (constants.%T.as_type.415)]
// CHECK:STDOUT:       %array_type.loc8_27.1: type = array_type %int_5, %.loc8_23 [symbolic = %array_type.loc8_27.2 (constants.%array_type.3df)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %w: ref @F.%array_type.loc8_27.2 (%array_type.3df) = ref_binding w, %w.var
// CHECK:STDOUT:     %impl.elem0.loc8_3.1: @F.%.loc8_3.2 (%.c90) = impl_witness_access constants.%Destroy.lookup_impl_witness.499, element0 [symbolic = %impl.elem0.loc8_3.2 (constants.%impl.elem0.cac)]
// CHECK:STDOUT:     %bound_method.loc8_3.1: <bound method> = bound_method %w.var, %impl.elem0.loc8_3.1
// CHECK:STDOUT:     %specific_impl_fn.loc8_3.1: <specific function> = specific_impl_function %impl.elem0.loc8_3.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.151) [symbolic = %specific_impl_fn.loc8_3.2 (constants.%specific_impl_fn.c68)]
// CHECK:STDOUT:     %bound_method.loc8_3.2: <bound method> = bound_method %w.var, %specific_impl_fn.loc8_3.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call.loc8: init %empty_tuple.type = call %bound_method.loc8_3.2(%w.var)
// CHECK:STDOUT:     %impl.elem0.loc7_3.1: @F.%.loc7_3 (%.343) = impl_witness_access constants.%Destroy.lookup_impl_witness.b08, element0 [symbolic = %impl.elem0.loc7_3.2 (constants.%impl.elem0.4e5)]
// CHECK:STDOUT:     %bound_method.loc7_3.1: <bound method> = bound_method %v.var, %impl.elem0.loc7_3.1
// CHECK:STDOUT:     %specific_impl_fn.loc7_3.1: <specific function> = specific_impl_function %impl.elem0.loc7_3.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.525) [symbolic = %specific_impl_fn.loc7_3.2 (constants.%specific_impl_fn.730)]
// CHECK:STDOUT:     %bound_method.loc7_3.2: <bound method> = bound_method %v.var, %specific_impl_fn.loc7_3.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call.loc7: init %empty_tuple.type = call %bound_method.loc7_3.2(%v.var)
// CHECK:STDOUT:     %impl.elem0.loc6_3.1: @F.%.loc6_3.2 (%.225) = impl_witness_access constants.%Destroy.lookup_impl_witness.b88, element0 [symbolic = %impl.elem0.loc6_3.2 (constants.%impl.elem0.e8c)]
// CHECK:STDOUT:     %bound_method.loc6_3.1: <bound method> = bound_method %u.var, %impl.elem0.loc6_3.1
// CHECK:STDOUT:     %specific_impl_fn.loc6_3.1: <specific function> = specific_impl_function %impl.elem0.loc6_3.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.dc1) [symbolic = %specific_impl_fn.loc6_3.2 (constants.%specific_impl_fn.883)]
// CHECK:STDOUT:     %bound_method.loc6_3.2: <bound method> = bound_method %u.var, %specific_impl_fn.loc6_3.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call.loc6: init %empty_tuple.type = call %bound_method.loc6_3.2(%u.var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%N.loc12_7.2: %i32) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %N.loc12_7.1, constants.%Int.as.ImplicitAs.impl.Convert.338 [symbolic = %Int.as.ImplicitAs.impl.Convert.bound (constants.%Int.as.ImplicitAs.impl.Convert.bound)]
// CHECK:STDOUT:   %bound_method.loc14_28.3: <bound method> = bound_method %N.loc12_7.1, constants.%Int.as.ImplicitAs.impl.Convert.specific_fn [symbolic = %bound_method.loc14_28.3 (constants.%bound_method)]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.call.loc14_28.2: init Core.IntLiteral = call %bound_method.loc14_28.3(%N.loc12_7.1) [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc14_28.2 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:   %array_type.loc14_29.2: type = array_type %Int.as.ImplicitAs.impl.Convert.call.loc14_28.2, constants.%i32 [symbolic = %array_type.loc14_29.2 (constants.%array_type.b1a)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %array_type.loc14_29.2 [symbolic = %require_complete (constants.%require_complete.553)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %array_type.loc14_29.2 [symbolic = %pattern_type (constants.%pattern_type.bc6)]
// CHECK:STDOUT:   %.loc14_30.3: require_specific_def_type = require_specific_def @T.as.DefaultOrUnformed.impl(%array_type.loc14_29.2) [symbolic = %.loc14_30.3 (constants.%.174)]
// CHECK:STDOUT:   %DefaultOrUnformed.lookup_impl_witness: <witness> = lookup_impl_witness %array_type.loc14_29.2, @DefaultOrUnformed [symbolic = %DefaultOrUnformed.lookup_impl_witness (constants.%DefaultOrUnformed.lookup_impl_witness.dfc)]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc14_30.2: %DefaultOrUnformed.type = facet_value %array_type.loc14_29.2, (%DefaultOrUnformed.lookup_impl_witness) [symbolic = %DefaultOrUnformed.facet.loc14_30.2 (constants.%DefaultOrUnformed.facet.fb3)]
// CHECK:STDOUT:   %DefaultOrUnformed.WithSelf.Op.type: type = fn_type @DefaultOrUnformed.WithSelf.Op, @DefaultOrUnformed.WithSelf(%DefaultOrUnformed.facet.loc14_30.2) [symbolic = %DefaultOrUnformed.WithSelf.Op.type (constants.%DefaultOrUnformed.WithSelf.Op.type.554)]
// CHECK:STDOUT:   %.loc14_30.4: type = fn_type_with_self_type %DefaultOrUnformed.WithSelf.Op.type, %DefaultOrUnformed.facet.loc14_30.2 [symbolic = %.loc14_30.4 (constants.%.4b5)]
// CHECK:STDOUT:   %impl.elem0.loc14_30.2: @G.%.loc14_30.4 (%.4b5) = impl_witness_access %DefaultOrUnformed.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc14_30.2 (constants.%impl.elem0.291)]
// CHECK:STDOUT:   %specific_impl_fn.loc14_30.2: <specific function> = specific_impl_function %impl.elem0.loc14_30.2, @DefaultOrUnformed.WithSelf.Op(%DefaultOrUnformed.facet.loc14_30.2) [symbolic = %specific_impl_fn.loc14_30.2 (constants.%specific_impl_fn.2ee)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %array_type.loc14_29.2, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.397)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %array_type.loc14_29.2, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet (constants.%Destroy.facet.9a6)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.6ac)]
// CHECK:STDOUT:   %.loc14_3.2: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Destroy.facet [symbolic = %.loc14_3.2 (constants.%.c61)]
// CHECK:STDOUT:   %impl.elem0.loc14_3.2: @G.%.loc14_3.2 (%.c61) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc14_3.2 (constants.%impl.elem0.e0c)]
// CHECK:STDOUT:   %specific_impl_fn.loc14_3.2: <specific function> = specific_impl_function %impl.elem0.loc14_3.2, @Destroy.WithSelf.Op(%Destroy.facet) [symbolic = %specific_impl_fn.loc14_3.2 (constants.%specific_impl_fn.028)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %k.patt: @G.%pattern_type (%pattern_type.bc6) = ref_binding_pattern k [concrete]
// CHECK:STDOUT:       %k.var_patt: @G.%pattern_type (%pattern_type.bc6) = var_pattern %k.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %k.var: ref @G.%array_type.loc14_29.2 (%array_type.b1a) = var %k.var_patt
// CHECK:STDOUT:     %DefaultOrUnformed.facet.loc14_30.1: %DefaultOrUnformed.type = facet_value constants.%array_type.b1a, (constants.%DefaultOrUnformed.lookup_impl_witness.dfc) [symbolic = %DefaultOrUnformed.facet.loc14_30.2 (constants.%DefaultOrUnformed.facet.fb3)]
// CHECK:STDOUT:     %.loc14_30.1: %DefaultOrUnformed.type = converted constants.%array_type.b1a, %DefaultOrUnformed.facet.loc14_30.1 [symbolic = %DefaultOrUnformed.facet.loc14_30.2 (constants.%DefaultOrUnformed.facet.fb3)]
// CHECK:STDOUT:     %as_type: type = facet_access_type %.loc14_30.1 [symbolic = %array_type.loc14_29.2 (constants.%array_type.b1a)]
// CHECK:STDOUT:     %.loc14_30.2: type = converted %.loc14_30.1, %as_type [symbolic = %array_type.loc14_29.2 (constants.%array_type.b1a)]
// CHECK:STDOUT:     %impl.elem0.loc14_30.1: @G.%.loc14_30.4 (%.4b5) = impl_witness_access constants.%DefaultOrUnformed.lookup_impl_witness.dfc, element0 [symbolic = %impl.elem0.loc14_30.2 (constants.%impl.elem0.291)]
// CHECK:STDOUT:     %specific_impl_fn.loc14_30.1: <specific function> = specific_impl_function %impl.elem0.loc14_30.1, @DefaultOrUnformed.WithSelf.Op(constants.%DefaultOrUnformed.facet.fb3) [symbolic = %specific_impl_fn.loc14_30.2 (constants.%specific_impl_fn.2ee)]
// CHECK:STDOUT:     %.loc14_3.1: ref @G.%array_type.loc14_29.2 (%array_type.b1a) = splice_block %k.var {}
// CHECK:STDOUT:     %DefaultOrUnformed.WithSelf.Op.call: init @G.%array_type.loc14_29.2 (%array_type.b1a) to %.loc14_3.1 = call %specific_impl_fn.loc14_30.1()
// CHECK:STDOUT:     assign %k.var, %DefaultOrUnformed.WithSelf.Op.call
// CHECK:STDOUT:     %.loc14_29: type = splice_block %array_type.loc14_29.1 [symbolic = %array_type.loc14_29.2 (constants.%array_type.b1a)] {
// CHECK:STDOUT:       %i32.loc14: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %N.ref: %i32 = name_ref N, %N.loc12_7.2 [symbolic = %N.loc12_7.1 (constants.%N.921)]
// CHECK:STDOUT:       %impl.elem0.loc14_28: %.d21 = impl_witness_access constants.%ImplicitAs.impl_witness.f73, element0 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.338]
// CHECK:STDOUT:       %bound_method.loc14_28.1: <bound method> = bound_method %N.ref, %impl.elem0.loc14_28 [symbolic = %Int.as.ImplicitAs.impl.Convert.bound (constants.%Int.as.ImplicitAs.impl.Convert.bound)]
// CHECK:STDOUT:       %specific_fn: <specific function> = specific_function %impl.elem0.loc14_28, @Int.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Int.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:       %bound_method.loc14_28.2: <bound method> = bound_method %N.ref, %specific_fn [symbolic = %bound_method.loc14_28.3 (constants.%bound_method)]
// CHECK:STDOUT:       %Int.as.ImplicitAs.impl.Convert.call.loc14_28.1: init Core.IntLiteral = call %bound_method.loc14_28.2(%N.ref) [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc14_28.2 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:       %.loc14_28.1: Core.IntLiteral = value_of_initializer %Int.as.ImplicitAs.impl.Convert.call.loc14_28.1 [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc14_28.2 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:       %.loc14_28.2: Core.IntLiteral = converted %N.ref, %.loc14_28.1 [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc14_28.2 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:       %array_type.loc14_29.1: type = array_type %.loc14_28.2, %i32.loc14 [symbolic = %array_type.loc14_29.2 (constants.%array_type.b1a)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %k: ref @G.%array_type.loc14_29.2 (%array_type.b1a) = ref_binding k, %k.var
// CHECK:STDOUT:     %impl.elem0.loc14_3.1: @G.%.loc14_3.2 (%.c61) = impl_witness_access constants.%Destroy.lookup_impl_witness.397, element0 [symbolic = %impl.elem0.loc14_3.2 (constants.%impl.elem0.e0c)]
// CHECK:STDOUT:     %bound_method.loc14_3.1: <bound method> = bound_method %k.var, %impl.elem0.loc14_3.1
// CHECK:STDOUT:     %specific_impl_fn.loc14_3.1: <specific function> = specific_impl_function %impl.elem0.loc14_3.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.9a6) [symbolic = %specific_impl_fn.loc14_3.2 (constants.%specific_impl_fn.028)]
// CHECK:STDOUT:     %bound_method.loc14_3.2: <bound method> = bound_method %k.var, %specific_impl_fn.loc14_3.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc14_3.2(%k.var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.761) {
// CHECK:STDOUT:   %T.loc4_7.1 => constants.%T.761
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%N.921) {
// CHECK:STDOUT:   %N.loc12_7.1 => constants.%N.921
// CHECK:STDOUT: }
// CHECK:STDOUT:

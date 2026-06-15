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
// CHECK:STDOUT:   %tuple.b59: %tuple.type.24b = tuple_value (%i32, %i32) [concrete]
// CHECK:STDOUT:   %tuple.type.e55: type = tuple_type (%i32, %i32) [concrete]
// CHECK:STDOUT:   %pattern_type.394: type = pattern_type %tuple.type.e55 [concrete]
// CHECK:STDOUT:   %tuple_copy.patt: %pattern_type.394 = ref_binding_pattern tuple_copy [concrete]
// CHECK:STDOUT:   %tuple_copy.var_patt: %pattern_type.394 = var_pattern %tuple_copy.patt [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.type.f94: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.ad8: %tuple.type.f94 = tuple_value (%int_1.5b8, %int_2.ecc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.914: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.845: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a23: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b3c, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.914 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a23) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.740: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.1c5: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.740, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.094: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.953: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.0c6: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.9f3: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %bound_method.3cb: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.295: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.102: %tuple.type.e55 = tuple_value (%int_1.0c6, %int_2.295) [concrete]
// CHECK:STDOUT:   %tuple.elem0: ref %i32 = tuple_access file.%tuple_copy.var, element0 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.ac8: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.5e0: %Int.as.Copy.impl.Op.type.ac8 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.0e0: <witness> = impl_witness imports.%Copy.impl_witness_table.367, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.3f6: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.4f6: %Int.as.Copy.impl.Op.type.3f6 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.0e0) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.d09: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.7a6: type = fn_type_with_self_type %Copy.WithSelf.Op.type.d09, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound.102: <bound method> = bound_method %int_1.0c6, %Int.as.Copy.impl.Op.4f6 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.4f6, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.681: <bound method> = bound_method %int_1.0c6, %Int.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %tuple.elem1: ref %i32 = tuple_access file.%tuple_copy.var, element1 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound.245: <bound method> = bound_method %int_2.295, %Int.as.Copy.impl.Op.4f6 [concrete]
// CHECK:STDOUT:   %bound_method.03a: <bound method> = bound_method %int_2.295, %Int.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %struct_type.a.b.c: type = struct_type {.a: %i32, .b: %i32, .c: %i32} [concrete]
// CHECK:STDOUT:   %pattern_type.9c3: type = pattern_type %struct_type.a.b.c [concrete]
// CHECK:STDOUT:   %struct_copy.patt: %pattern_type.9c3 = ref_binding_pattern struct_copy [concrete]
// CHECK:STDOUT:   %struct_copy.var_patt: %pattern_type.9c3 = var_pattern %struct_copy.patt [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %struct_type.c.b.a: type = struct_type {.c: Core.IntLiteral, .b: Core.IntLiteral, .a: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct.c75: %struct_type.c.b.a = struct_value (%int_3.1ba, %int_2.ecc, %int_1.5b8) [concrete]
// CHECK:STDOUT:   %struct_type.b.a.c: type = struct_type {.b: %i32, .a: %i32, .c: %i32} [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.485: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %bound_method.763: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.410: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %struct.55d: %struct_type.b.a.c = struct_value (%int_2.295, %int_1.0c6, %int_3.410) [concrete]
// CHECK:STDOUT:   %.6b5: ref %i32 = struct_access file.%struct_copy.var, element1 [concrete]
// CHECK:STDOUT:   %.e89: ref %i32 = struct_access file.%struct_copy.var, element0 [concrete]
// CHECK:STDOUT:   %.dd4: ref %i32 = struct_access file.%struct_copy.var, element2 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound.4ec: <bound method> = bound_method %int_3.410, %Int.as.Copy.impl.Op.4f6 [concrete]
// CHECK:STDOUT:   %bound_method.1b6: <bound method> = bound_method %int_3.410, %Int.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %struct.11a: %struct_type.a.b.c = struct_value (%int_1.0c6, %int_2.295, %int_3.410) [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_1.5b8, %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.97e: type = pattern_type %array_type [concrete]
// CHECK:STDOUT:   %tuple_index.patt: %pattern_type.97e = ref_binding_pattern tuple_index [concrete]
// CHECK:STDOUT:   %tuple_index.var_patt: %pattern_type.97e = var_pattern %tuple_index.patt [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %tuple.type.985: type = tuple_type (Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.4f2: %tuple.type.985 = tuple_value (%int_0.5c6) [concrete]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %int_7: Core.IntLiteral = int_value 7 [concrete]
// CHECK:STDOUT:   %int_9: Core.IntLiteral = int_value 9 [concrete]
// CHECK:STDOUT:   %tuple.type.d46: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.869: %tuple.type.d46 = tuple_value (%int_5, %int_7, %int_1.5b8, %int_9) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.7b4: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %bound_method.6e0: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.3c0: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%int_0.3c0) [concrete]
// CHECK:STDOUT:   %struct_access.patt: %pattern_type.97e = ref_binding_pattern struct_access [concrete]
// CHECK:STDOUT:   %struct_access.var_patt: %pattern_type.97e = var_pattern %struct_access.patt [concrete]
// CHECK:STDOUT:   %struct_type.a.b: type = struct_type {.a: Core.IntLiteral, .b: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct.a81: %struct_type.a.b = struct_value (%int_3.1ba, %int_1.5b8) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.edf: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.845)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b3c = impl_witness_table (%Core.import_ref.edf), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.cd6: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.ac8) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.5e0)]
// CHECK:STDOUT:   %Copy.impl_witness_table.367 = impl_witness_table (%Core.import_ref.cd6), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %tuple_copy.patt: %pattern_type.394 = ref_binding_pattern tuple_copy [concrete = constants.%tuple_copy.patt]
// CHECK:STDOUT:     %tuple_copy.var_patt: %pattern_type.394 = var_pattern %tuple_copy.patt [concrete = constants.%tuple_copy.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %tuple_copy.var: ref %tuple.type.e55 = var %tuple_copy.var_patt [concrete]
// CHECK:STDOUT:   %.loc4_26.1: type = splice_block %.loc4_26.3 [concrete = constants.%tuple.type.e55] {
// CHECK:STDOUT:     %i32.loc4_18: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc4_23: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc4_26.2: %tuple.type.24b = tuple_literal (%i32.loc4_18, %i32.loc4_23) [concrete = constants.%tuple.b59]
// CHECK:STDOUT:     %.loc4_26.3: type = converted %.loc4_26.2, constants.%tuple.type.e55 [concrete = constants.%tuple.type.e55]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %tuple_copy: ref %tuple.type.e55 = ref_binding tuple_copy, %tuple_copy.var [concrete = %tuple_copy.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %struct_copy.patt: %pattern_type.9c3 = ref_binding_pattern struct_copy [concrete = constants.%struct_copy.patt]
// CHECK:STDOUT:     %struct_copy.var_patt: %pattern_type.9c3 = var_pattern %struct_copy.patt [concrete = constants.%struct_copy.var_patt]
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
// CHECK:STDOUT:     %tuple_index.patt: %pattern_type.97e = ref_binding_pattern tuple_index [concrete = constants.%tuple_index.patt]
// CHECK:STDOUT:     %tuple_index.var_patt: %pattern_type.97e = var_pattern %tuple_index.patt [concrete = constants.%tuple_index.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %tuple_index.var: ref %array_type = var %tuple_index.var_patt [concrete]
// CHECK:STDOUT:   %.loc8: type = splice_block %array_type.loc8 [concrete = constants.%array_type] {
// CHECK:STDOUT:     %i32.loc8: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %array_type.loc8: type = array_type %int_1.loc8, %i32.loc8 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %tuple_index: ref %array_type = ref_binding tuple_index, %tuple_index.var [concrete = %tuple_index.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %struct_access.patt: %pattern_type.97e = ref_binding_pattern struct_access [concrete = constants.%struct_access.patt]
// CHECK:STDOUT:     %struct_access.var_patt: %pattern_type.97e = var_pattern %struct_access.patt [concrete = constants.%struct_access.var_patt]
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
// CHECK:STDOUT:   %.loc4_49.1: %tuple.type.24b = tuple_literal (%i32.loc4_41, %i32.loc4_46) [concrete = constants.%tuple.b59]
// CHECK:STDOUT:   %.loc4_49.2: type = converted %.loc4_49.1, constants.%tuple.type.e55 [concrete = constants.%tuple.type.e55]
// CHECK:STDOUT:   %impl.elem0.loc4_35.1: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc4_35.1: <bound method> = bound_method %int_1.loc4, %impl.elem0.loc4_35.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.094]
// CHECK:STDOUT:   %specific_fn.loc4_35.1: <specific function> = specific_function %impl.elem0.loc4_35.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc4_35.2: <bound method> = bound_method %int_1.loc4, %specific_fn.loc4_35.1 [concrete = constants.%bound_method.953]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc4_35.1: init %i32 = call %bound_method.loc4_35.2(%int_1.loc4) [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %.loc4_35.2: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc4_35.1 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %.loc4_35.3: %i32 = converted %int_1.loc4, %.loc4_35.2 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %impl.elem0.loc4_35.2: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc4_35.3: <bound method> = bound_method %int_2.loc4, %impl.elem0.loc4_35.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.9f3]
// CHECK:STDOUT:   %specific_fn.loc4_35.2: <specific function> = specific_function %impl.elem0.loc4_35.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc4_35.4: <bound method> = bound_method %int_2.loc4, %specific_fn.loc4_35.2 [concrete = constants.%bound_method.3cb]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc4_35.2: init %i32 = call %bound_method.loc4_35.4(%int_2.loc4) [concrete = constants.%int_2.295]
// CHECK:STDOUT:   %.loc4_35.4: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc4_35.2 [concrete = constants.%int_2.295]
// CHECK:STDOUT:   %.loc4_35.5: %i32 = converted %int_2.loc4, %.loc4_35.4 [concrete = constants.%int_2.295]
// CHECK:STDOUT:   %tuple.loc4: %tuple.type.e55 = tuple_value (%.loc4_35.3, %.loc4_35.5) [concrete = constants.%tuple.102]
// CHECK:STDOUT:   %.loc4_37.1: %tuple.type.e55 = converted %.loc4_35.1, %tuple.loc4 [concrete = constants.%tuple.102]
// CHECK:STDOUT:   %tuple.elem0.loc4_37.1: %i32 = tuple_access %.loc4_37.1, element0 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %impl.elem0.loc4_37.1: %.7a6 = impl_witness_access constants.%Copy.impl_witness.0e0, element0 [concrete = constants.%Int.as.Copy.impl.Op.4f6]
// CHECK:STDOUT:   %bound_method.loc4_37.1: <bound method> = bound_method %tuple.elem0.loc4_37.1, %impl.elem0.loc4_37.1 [concrete = constants.%Int.as.Copy.impl.Op.bound.102]
// CHECK:STDOUT:   %specific_fn.loc4_37.1: <specific function> = specific_function %impl.elem0.loc4_37.1, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc4_37.2: <bound method> = bound_method %tuple.elem0.loc4_37.1, %specific_fn.loc4_37.1 [concrete = constants.%bound_method.681]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc4_37.1: init %i32 = call %bound_method.loc4_37.2(%tuple.elem0.loc4_37.1) [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %tuple.elem0.loc4_37.2: ref %i32 = tuple_access file.%tuple_copy.var, element0 [concrete = constants.%tuple.elem0]
// CHECK:STDOUT:   %.loc4_37.2: init %i32 to %tuple.elem0.loc4_37.2 = in_place_init %Int.as.Copy.impl.Op.call.loc4_37.1 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %tuple.elem1.loc4_37.1: %i32 = tuple_access %.loc4_37.1, element1 [concrete = constants.%int_2.295]
// CHECK:STDOUT:   %impl.elem0.loc4_37.2: %.7a6 = impl_witness_access constants.%Copy.impl_witness.0e0, element0 [concrete = constants.%Int.as.Copy.impl.Op.4f6]
// CHECK:STDOUT:   %bound_method.loc4_37.3: <bound method> = bound_method %tuple.elem1.loc4_37.1, %impl.elem0.loc4_37.2 [concrete = constants.%Int.as.Copy.impl.Op.bound.245]
// CHECK:STDOUT:   %specific_fn.loc4_37.2: <specific function> = specific_function %impl.elem0.loc4_37.2, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc4_37.4: <bound method> = bound_method %tuple.elem1.loc4_37.1, %specific_fn.loc4_37.2 [concrete = constants.%bound_method.03a]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc4_37.2: init %i32 = call %bound_method.loc4_37.4(%tuple.elem1.loc4_37.1) [concrete = constants.%int_2.295]
// CHECK:STDOUT:   %tuple.elem1.loc4_37.2: ref %i32 = tuple_access file.%tuple_copy.var, element1 [concrete = constants.%tuple.elem1]
// CHECK:STDOUT:   %.loc4_37.3: init %i32 to %tuple.elem1.loc4_37.2 = in_place_init %Int.as.Copy.impl.Op.call.loc4_37.2 [concrete = constants.%int_2.295]
// CHECK:STDOUT:   %.loc4_37.4: init %tuple.type.e55 to file.%tuple_copy.var = tuple_init (%.loc4_37.2, %.loc4_37.3) [concrete = constants.%tuple.102]
// CHECK:STDOUT:   %.loc4_1: init %tuple.type.e55 = converted %.loc4_37.1, %.loc4_37.4 [concrete = constants.%tuple.102]
// CHECK:STDOUT:   assign file.%tuple_copy.var, %.loc4_1
// CHECK:STDOUT:   %int_3.loc6: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %int_2.loc6: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %int_1.loc6: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc6_71.1: %struct_type.c.b.a = struct_literal (%int_3.loc6, %int_2.loc6, %int_1.loc6) [concrete = constants.%struct.c75]
// CHECK:STDOUT:   %i32.loc6_81: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %i32.loc6_90: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %i32.loc6_99: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %struct_type.b.a.c: type = struct_type {.b: %i32, .a: %i32, .c: %i32} [concrete = constants.%struct_type.b.a.c]
// CHECK:STDOUT:   %impl.elem0.loc6_71.1: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc6_71.1: <bound method> = bound_method %int_2.loc6, %impl.elem0.loc6_71.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.9f3]
// CHECK:STDOUT:   %specific_fn.loc6_71.1: <specific function> = specific_function %impl.elem0.loc6_71.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_71.2: <bound method> = bound_method %int_2.loc6, %specific_fn.loc6_71.1 [concrete = constants.%bound_method.3cb]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_71.1: init %i32 = call %bound_method.loc6_71.2(%int_2.loc6) [concrete = constants.%int_2.295]
// CHECK:STDOUT:   %.loc6_71.2: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_71.1 [concrete = constants.%int_2.295]
// CHECK:STDOUT:   %.loc6_71.3: %i32 = converted %int_2.loc6, %.loc6_71.2 [concrete = constants.%int_2.295]
// CHECK:STDOUT:   %impl.elem0.loc6_71.2: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc6_71.3: <bound method> = bound_method %int_1.loc6, %impl.elem0.loc6_71.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.094]
// CHECK:STDOUT:   %specific_fn.loc6_71.2: <specific function> = specific_function %impl.elem0.loc6_71.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_71.4: <bound method> = bound_method %int_1.loc6, %specific_fn.loc6_71.2 [concrete = constants.%bound_method.953]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_71.2: init %i32 = call %bound_method.loc6_71.4(%int_1.loc6) [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %.loc6_71.4: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_71.2 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %.loc6_71.5: %i32 = converted %int_1.loc6, %.loc6_71.4 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %impl.elem0.loc6_71.3: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc6_71.5: <bound method> = bound_method %int_3.loc6, %impl.elem0.loc6_71.3 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.485]
// CHECK:STDOUT:   %specific_fn.loc6_71.3: <specific function> = specific_function %impl.elem0.loc6_71.3, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_71.6: <bound method> = bound_method %int_3.loc6, %specific_fn.loc6_71.3 [concrete = constants.%bound_method.763]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_71.3: init %i32 = call %bound_method.loc6_71.6(%int_3.loc6) [concrete = constants.%int_3.410]
// CHECK:STDOUT:   %.loc6_71.6: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_71.3 [concrete = constants.%int_3.410]
// CHECK:STDOUT:   %.loc6_71.7: %i32 = converted %int_3.loc6, %.loc6_71.6 [concrete = constants.%int_3.410]
// CHECK:STDOUT:   %struct.loc6: %struct_type.b.a.c = struct_value (%.loc6_71.3, %.loc6_71.5, %.loc6_71.7) [concrete = constants.%struct.55d]
// CHECK:STDOUT:   %.loc6_73.1: %struct_type.b.a.c = converted %.loc6_71.1, %struct.loc6 [concrete = constants.%struct.55d]
// CHECK:STDOUT:   %.loc6_73.2: %i32 = struct_access %.loc6_73.1, element1 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %impl.elem0.loc6_73.1: %.7a6 = impl_witness_access constants.%Copy.impl_witness.0e0, element0 [concrete = constants.%Int.as.Copy.impl.Op.4f6]
// CHECK:STDOUT:   %bound_method.loc6_73.1: <bound method> = bound_method %.loc6_73.2, %impl.elem0.loc6_73.1 [concrete = constants.%Int.as.Copy.impl.Op.bound.102]
// CHECK:STDOUT:   %specific_fn.loc6_73.1: <specific function> = specific_function %impl.elem0.loc6_73.1, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_73.2: <bound method> = bound_method %.loc6_73.2, %specific_fn.loc6_73.1 [concrete = constants.%bound_method.681]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc6_73.1: init %i32 = call %bound_method.loc6_73.2(%.loc6_73.2) [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %.loc6_73.3: ref %i32 = struct_access file.%struct_copy.var, element1 [concrete = constants.%.6b5]
// CHECK:STDOUT:   %.loc6_73.4: init %i32 to %.loc6_73.3 = in_place_init %Int.as.Copy.impl.Op.call.loc6_73.1 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %.loc6_73.5: %i32 = struct_access %.loc6_73.1, element0 [concrete = constants.%int_2.295]
// CHECK:STDOUT:   %impl.elem0.loc6_73.2: %.7a6 = impl_witness_access constants.%Copy.impl_witness.0e0, element0 [concrete = constants.%Int.as.Copy.impl.Op.4f6]
// CHECK:STDOUT:   %bound_method.loc6_73.3: <bound method> = bound_method %.loc6_73.5, %impl.elem0.loc6_73.2 [concrete = constants.%Int.as.Copy.impl.Op.bound.245]
// CHECK:STDOUT:   %specific_fn.loc6_73.2: <specific function> = specific_function %impl.elem0.loc6_73.2, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_73.4: <bound method> = bound_method %.loc6_73.5, %specific_fn.loc6_73.2 [concrete = constants.%bound_method.03a]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc6_73.2: init %i32 = call %bound_method.loc6_73.4(%.loc6_73.5) [concrete = constants.%int_2.295]
// CHECK:STDOUT:   %.loc6_73.6: ref %i32 = struct_access file.%struct_copy.var, element0 [concrete = constants.%.e89]
// CHECK:STDOUT:   %.loc6_73.7: init %i32 to %.loc6_73.6 = in_place_init %Int.as.Copy.impl.Op.call.loc6_73.2 [concrete = constants.%int_2.295]
// CHECK:STDOUT:   %.loc6_73.8: %i32 = struct_access %.loc6_73.1, element2 [concrete = constants.%int_3.410]
// CHECK:STDOUT:   %impl.elem0.loc6_73.3: %.7a6 = impl_witness_access constants.%Copy.impl_witness.0e0, element0 [concrete = constants.%Int.as.Copy.impl.Op.4f6]
// CHECK:STDOUT:   %bound_method.loc6_73.5: <bound method> = bound_method %.loc6_73.8, %impl.elem0.loc6_73.3 [concrete = constants.%Int.as.Copy.impl.Op.bound.4ec]
// CHECK:STDOUT:   %specific_fn.loc6_73.3: <specific function> = specific_function %impl.elem0.loc6_73.3, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_73.6: <bound method> = bound_method %.loc6_73.8, %specific_fn.loc6_73.3 [concrete = constants.%bound_method.1b6]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc6_73.3: init %i32 = call %bound_method.loc6_73.6(%.loc6_73.8) [concrete = constants.%int_3.410]
// CHECK:STDOUT:   %.loc6_73.9: ref %i32 = struct_access file.%struct_copy.var, element2 [concrete = constants.%.dd4]
// CHECK:STDOUT:   %.loc6_73.10: init %i32 to %.loc6_73.9 = in_place_init %Int.as.Copy.impl.Op.call.loc6_73.3 [concrete = constants.%int_3.410]
// CHECK:STDOUT:   %.loc6_73.11: init %struct_type.a.b.c to file.%struct_copy.var = struct_init (%.loc6_73.4, %.loc6_73.7, %.loc6_73.10) [concrete = constants.%struct.11a]
// CHECK:STDOUT:   %.loc6_1: init %struct_type.a.b.c = converted %.loc6_73.1, %.loc6_73.11 [concrete = constants.%struct.11a]
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
// CHECK:STDOUT:   %impl.elem0.loc8: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc8_37.1: <bound method> = bound_method %int_0.loc8_35, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.7b4]
// CHECK:STDOUT:   %specific_fn.loc8: <specific function> = specific_function %impl.elem0.loc8, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_37.2: <bound method> = bound_method %int_0.loc8_35, %specific_fn.loc8 [concrete = constants.%bound_method.6e0]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8: init %i32 = call %bound_method.loc8_37.2(%int_0.loc8_35) [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:   %.loc8_37.2: init %i32 = converted %int_0.loc8_35, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8 [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:   %.loc8_1: ref %array_type = splice_block file.%tuple_index.var [concrete = file.%tuple_index.var] {}
// CHECK:STDOUT:   %int_0.loc8_37: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc8_37.3: ref %i32 = array_index %.loc8_1, %int_0.loc8_37
// CHECK:STDOUT:   %.loc8_37.4: init %i32 to %.loc8_37.3 = in_place_init %.loc8_37.2 [concrete = constants.%int_0.3c0]
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
// CHECK:STDOUT:   %impl.elem0.loc10: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc10_39.1: <bound method> = bound_method %int_0.loc10_37, %impl.elem0.loc10 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.7b4]
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_39.2: <bound method> = bound_method %int_0.loc10_37, %specific_fn.loc10 [concrete = constants.%bound_method.6e0]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10: init %i32 = call %bound_method.loc10_39.2(%int_0.loc10_37) [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:   %.loc10_39.2: init %i32 = converted %int_0.loc10_37, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10 [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:   %.loc10_1: ref %array_type = splice_block file.%struct_access.var [concrete = file.%struct_access.var] {}
// CHECK:STDOUT:   %int_0.loc10_39: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc10_39.3: ref %i32 = array_index %.loc10_1, %int_0.loc10_39
// CHECK:STDOUT:   %.loc10_39.4: init %i32 to %.loc10_39.3 = in_place_init %.loc10_39.2 [concrete = constants.%int_0.3c0]
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
// CHECK:STDOUT:   %array_type.d27: type = array_type %int_1.5b8, %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.97e: type = pattern_type %array_type.d27 [concrete]
// CHECK:STDOUT:   %array_index.patt: %pattern_type.97e = ref_binding_pattern array_index [concrete]
// CHECK:STDOUT:   %array_index.var_patt: %pattern_type.97e = var_pattern %array_index.patt [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %tuple.type.985: type = tuple_type (Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.4f2: %tuple.type.985 = tuple_value (%int_0) [concrete]
// CHECK:STDOUT:   %int_5.64b: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %int_7.29f: Core.IntLiteral = int_value 7 [concrete]
// CHECK:STDOUT:   %int_9.988: Core.IntLiteral = int_value 9 [concrete]
// CHECK:STDOUT:   %tuple.type.d46: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.869: %tuple.type.d46 = tuple_value (%int_5.64b, %int_7.29f, %int_1.5b8, %int_9.988) [concrete]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %array_type.88c: type = array_type %int_4, %i32 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.914: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.845: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a23: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b3c, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.914 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a23) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.740: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.1c5: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.740, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.f6c: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.c9d: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_5.eef: %i32 = int_value 5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.66b: <bound method> = bound_method %int_7.29f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %bound_method.7ff: <bound method> = bound_method %int_7.29f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_7.690: %i32 = int_value 7 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.094: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %bound_method.953: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.0c6: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.f46: <bound method> = bound_method %int_9.988, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %bound_method.9bf: <bound method> = bound_method %int_9.988, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_9.056: %i32 = int_value 9 [concrete]
// CHECK:STDOUT:   %array: %array_type.88c = tuple_value (%int_5.eef, %int_7.690, %int_1.0c6, %int_9.056) [concrete]
// CHECK:STDOUT:   %.653: ref %array_type.88c = temporary invalid, %array [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.9f3: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %bound_method.3cb: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.295: %i32 = int_value 2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.edf: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.845)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b3c = impl_witness_table (%Core.import_ref.edf), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %array_index.patt: %pattern_type.97e = ref_binding_pattern array_index [concrete = constants.%array_index.patt]
// CHECK:STDOUT:     %array_index.var_patt: %pattern_type.97e = var_pattern %array_index.patt [concrete = constants.%array_index.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %array_index.var: ref %array_type.d27 = var %array_index.var_patt [concrete]
// CHECK:STDOUT:   %.loc9: type = splice_block %array_type [concrete = constants.%array_type.d27] {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %array_type: type = array_type %int_1, %i32 [concrete = constants.%array_type.d27]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %array_index: ref %array_type.d27 = ref_binding array_index, %array_index.var [concrete = %array_index.var]
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
// CHECK:STDOUT:   %array_type: type = array_type %int_4, %i32.loc9_76 [concrete = constants.%array_type.88c]
// CHECK:STDOUT:   %impl.elem0.loc9_65.1: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc9_65.1: <bound method> = bound_method %int_5, %impl.elem0.loc9_65.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.f6c]
// CHECK:STDOUT:   %specific_fn.loc9_65.1: <specific function> = specific_function %impl.elem0.loc9_65.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_65.2: <bound method> = bound_method %int_5, %specific_fn.loc9_65.1 [concrete = constants.%bound_method.c9d]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_65.1: init %i32 = call %bound_method.loc9_65.2(%int_5) [concrete = constants.%int_5.eef]
// CHECK:STDOUT:   %.loc9_65.2: init %i32 = converted %int_5, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_65.1 [concrete = constants.%int_5.eef]
// CHECK:STDOUT:   %.loc9_65.3: ref %array_type.88c = temporary_storage
// CHECK:STDOUT:   %int_0.loc9_65: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc9_65.4: ref %i32 = array_index %.loc9_65.3, %int_0.loc9_65
// CHECK:STDOUT:   %.loc9_65.5: init %i32 to %.loc9_65.4 = in_place_init %.loc9_65.2 [concrete = constants.%int_5.eef]
// CHECK:STDOUT:   %impl.elem0.loc9_65.2: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc9_65.3: <bound method> = bound_method %int_7, %impl.elem0.loc9_65.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.66b]
// CHECK:STDOUT:   %specific_fn.loc9_65.2: <specific function> = specific_function %impl.elem0.loc9_65.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_65.4: <bound method> = bound_method %int_7, %specific_fn.loc9_65.2 [concrete = constants.%bound_method.7ff]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_65.2: init %i32 = call %bound_method.loc9_65.4(%int_7) [concrete = constants.%int_7.690]
// CHECK:STDOUT:   %.loc9_65.6: init %i32 = converted %int_7, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_65.2 [concrete = constants.%int_7.690]
// CHECK:STDOUT:   %int_1.loc9_65: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc9_65.7: ref %i32 = array_index %.loc9_65.3, %int_1.loc9_65
// CHECK:STDOUT:   %.loc9_65.8: init %i32 to %.loc9_65.7 = in_place_init %.loc9_65.6 [concrete = constants.%int_7.690]
// CHECK:STDOUT:   %impl.elem0.loc9_65.3: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc9_65.5: <bound method> = bound_method %int_1.loc9_61, %impl.elem0.loc9_65.3 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.094]
// CHECK:STDOUT:   %specific_fn.loc9_65.3: <specific function> = specific_function %impl.elem0.loc9_65.3, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_65.6: <bound method> = bound_method %int_1.loc9_61, %specific_fn.loc9_65.3 [concrete = constants.%bound_method.953]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_65.3: init %i32 = call %bound_method.loc9_65.6(%int_1.loc9_61) [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %.loc9_65.9: init %i32 = converted %int_1.loc9_61, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_65.3 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %int_2.loc9_65: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc9_65.10: ref %i32 = array_index %.loc9_65.3, %int_2.loc9_65
// CHECK:STDOUT:   %.loc9_65.11: init %i32 to %.loc9_65.10 = in_place_init %.loc9_65.9 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %impl.elem0.loc9_65.4: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc9_65.7: <bound method> = bound_method %int_9, %impl.elem0.loc9_65.4 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.f46]
// CHECK:STDOUT:   %specific_fn.loc9_65.4: <specific function> = specific_function %impl.elem0.loc9_65.4, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_65.8: <bound method> = bound_method %int_9, %specific_fn.loc9_65.4 [concrete = constants.%bound_method.9bf]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_65.4: init %i32 = call %bound_method.loc9_65.8(%int_9) [concrete = constants.%int_9.056]
// CHECK:STDOUT:   %.loc9_65.12: init %i32 = converted %int_9, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_65.4 [concrete = constants.%int_9.056]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:   %.loc9_65.13: ref %i32 = array_index %.loc9_65.3, %int_3
// CHECK:STDOUT:   %.loc9_65.14: init %i32 to %.loc9_65.13 = in_place_init %.loc9_65.12 [concrete = constants.%int_9.056]
// CHECK:STDOUT:   %.loc9_65.15: init %array_type.88c to %.loc9_65.3 = array_init (%.loc9_65.5, %.loc9_65.8, %.loc9_65.11, %.loc9_65.14) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc9_67.1: init %array_type.88c = converted %.loc9_65.1, %.loc9_65.15 [concrete = constants.%array]
// CHECK:STDOUT:   %int_2.loc9_85: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc9_67.2: ref %array_type.88c = temporary %.loc9_65.3, %.loc9_67.1 [concrete = constants.%.653]
// CHECK:STDOUT:   %impl.elem0.loc9_85: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc9_85.1: <bound method> = bound_method %int_2.loc9_85, %impl.elem0.loc9_85 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.9f3]
// CHECK:STDOUT:   %specific_fn.loc9_85: <specific function> = specific_function %impl.elem0.loc9_85, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_85.2: <bound method> = bound_method %int_2.loc9_85, %specific_fn.loc9_85 [concrete = constants.%bound_method.3cb]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_85: init %i32 = call %bound_method.loc9_85.2(%int_2.loc9_85) [concrete = constants.%int_2.295]
// CHECK:STDOUT:   %.loc9_85.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_85 [concrete = constants.%int_2.295]
// CHECK:STDOUT:   %.loc9_85.2: %i32 = converted %int_2.loc9_85, %.loc9_85.1 [concrete = constants.%int_2.295]
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
// CHECK:STDOUT:   %pattern_type.429: type = pattern_type %Destroy.type [concrete]
// CHECK:STDOUT:   %T.patt.351: %pattern_type.429 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.0e7: %Destroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %T.as_type.700: type = facet_access_type %T.0e7 [symbolic]
// CHECK:STDOUT:   %ptr.b37: type = ptr_type %T.as_type.700 [symbolic]
// CHECK:STDOUT:   %const: type = const_type %T.as_type.700 [symbolic]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.24b = tuple_value (%ptr.b37, %const) [symbolic]
// CHECK:STDOUT:   %tuple.type.cd5: type = tuple_type (%ptr.b37, %const) [symbolic]
// CHECK:STDOUT:   %require_complete.3b6: <witness> = require_complete_type %tuple.type.cd5 [symbolic]
// CHECK:STDOUT:   %pattern_type.02b: type = pattern_type %tuple.type.cd5 [symbolic]
// CHECK:STDOUT:   %u.patt: %pattern_type.02b = ref_binding_pattern u [symbolic]
// CHECK:STDOUT:   %u.var_patt: %pattern_type.02b = var_pattern %u.patt [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.lookup_impl_witness.2ab: <witness> = lookup_impl_witness %tuple.type.cd5, @DefaultOrUnformed [symbolic]
// CHECK:STDOUT:   %.f98: require_specific_def_type = require_specific_def @T.as.DefaultOrUnformed.impl(%tuple.type.cd5) [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.c98: %DefaultOrUnformed.type = facet_value %tuple.type.cd5, (%DefaultOrUnformed.lookup_impl_witness.2ab) [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.WithSelf.Op.type.f2b: type = fn_type @DefaultOrUnformed.WithSelf.Op, @DefaultOrUnformed.WithSelf(%DefaultOrUnformed.facet.c98) [symbolic]
// CHECK:STDOUT:   %.c04: type = fn_type_with_self_type %DefaultOrUnformed.WithSelf.Op.type.f2b, %DefaultOrUnformed.facet.c98 [symbolic]
// CHECK:STDOUT:   %impl.elem0.91d: %.c04 = impl_witness_access %DefaultOrUnformed.lookup_impl_witness.2ab, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.495: <specific function> = specific_impl_function %impl.elem0.91d, @DefaultOrUnformed.WithSelf.Op(%DefaultOrUnformed.facet.c98) [symbolic]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %T.as_type.700} [symbolic]
// CHECK:STDOUT:   %require_complete.f8e: <witness> = require_complete_type %struct_type.a [symbolic]
// CHECK:STDOUT:   %pattern_type.915: type = pattern_type %struct_type.a [symbolic]
// CHECK:STDOUT:   %v.patt: %pattern_type.915 = ref_binding_pattern v [symbolic]
// CHECK:STDOUT:   %v.var_patt: %pattern_type.915 = var_pattern %v.patt [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.lookup_impl_witness.53f: <witness> = lookup_impl_witness %struct_type.a, @DefaultOrUnformed [symbolic]
// CHECK:STDOUT:   %.b78: require_specific_def_type = require_specific_def @T.as.DefaultOrUnformed.impl(%struct_type.a) [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.103: %DefaultOrUnformed.type = facet_value %struct_type.a, (%DefaultOrUnformed.lookup_impl_witness.53f) [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.WithSelf.Op.type.4de: type = fn_type @DefaultOrUnformed.WithSelf.Op, @DefaultOrUnformed.WithSelf(%DefaultOrUnformed.facet.103) [symbolic]
// CHECK:STDOUT:   %.883: type = fn_type_with_self_type %DefaultOrUnformed.WithSelf.Op.type.4de, %DefaultOrUnformed.facet.103 [symbolic]
// CHECK:STDOUT:   %impl.elem0.178: %.883 = impl_witness_access %DefaultOrUnformed.lookup_impl_witness.53f, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.b12: <specific function> = specific_impl_function %impl.elem0.178, @DefaultOrUnformed.WithSelf.Op(%DefaultOrUnformed.facet.103) [symbolic]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %array_type.50f: type = array_type %int_5, %T.as_type.700 [symbolic]
// CHECK:STDOUT:   %require_complete.f3e: <witness> = require_complete_type %array_type.50f [symbolic]
// CHECK:STDOUT:   %pattern_type.78a: type = pattern_type %array_type.50f [symbolic]
// CHECK:STDOUT:   %w.patt: %pattern_type.78a = ref_binding_pattern w [symbolic]
// CHECK:STDOUT:   %w.var_patt: %pattern_type.78a = var_pattern %w.patt [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.lookup_impl_witness.1fb: <witness> = lookup_impl_witness %array_type.50f, @DefaultOrUnformed [symbolic]
// CHECK:STDOUT:   %.376: require_specific_def_type = require_specific_def @T.as.DefaultOrUnformed.impl(%array_type.50f) [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.d60: %DefaultOrUnformed.type = facet_value %array_type.50f, (%DefaultOrUnformed.lookup_impl_witness.1fb) [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.WithSelf.Op.type.e22: type = fn_type @DefaultOrUnformed.WithSelf.Op, @DefaultOrUnformed.WithSelf(%DefaultOrUnformed.facet.d60) [symbolic]
// CHECK:STDOUT:   %.478: type = fn_type_with_self_type %DefaultOrUnformed.WithSelf.Op.type.e22, %DefaultOrUnformed.facet.d60 [symbolic]
// CHECK:STDOUT:   %impl.elem0.276: %.478 = impl_witness_access %DefaultOrUnformed.lookup_impl_witness.1fb, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.6bf: <specific function> = specific_impl_function %impl.elem0.276, @DefaultOrUnformed.WithSelf.Op(%DefaultOrUnformed.facet.d60) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.1f5: <witness> = lookup_impl_witness %array_type.50f, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.664: %Destroy.type = facet_value %array_type.50f, (%Destroy.lookup_impl_witness.1f5) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.373: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.664) [symbolic]
// CHECK:STDOUT:   %.260: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.373, %Destroy.facet.664 [symbolic]
// CHECK:STDOUT:   %impl.elem0.375: %.260 = impl_witness_access %Destroy.lookup_impl_witness.1f5, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.f05: <specific function> = specific_impl_function %impl.elem0.375, @Destroy.WithSelf.Op(%Destroy.facet.664) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.7f9: <witness> = lookup_impl_witness %struct_type.a, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.aa3: %Destroy.type = facet_value %struct_type.a, (%Destroy.lookup_impl_witness.7f9) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ae9: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.aa3) [symbolic]
// CHECK:STDOUT:   %.3c7: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.ae9, %Destroy.facet.aa3 [symbolic]
// CHECK:STDOUT:   %impl.elem0.bb8: %.3c7 = impl_witness_access %Destroy.lookup_impl_witness.7f9, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.799: <specific function> = specific_impl_function %impl.elem0.bb8, @Destroy.WithSelf.Op(%Destroy.facet.aa3) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.d26: <witness> = lookup_impl_witness %tuple.type.cd5, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.7ba: %Destroy.type = facet_value %tuple.type.cd5, (%Destroy.lookup_impl_witness.d26) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.a30: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.7ba) [symbolic]
// CHECK:STDOUT:   %.a0a: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.a30, %Destroy.facet.7ba [symbolic]
// CHECK:STDOUT:   %impl.elem0.667: %.a0a = impl_witness_access %Destroy.lookup_impl_witness.d26, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.96b: <specific function> = specific_impl_function %impl.elem0.667, @Destroy.WithSelf.Op(%Destroy.facet.7ba) [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %N.patt.38a: %pattern_type.6b6 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %N.37f: %i32 = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.7cb: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %From: Core.IntLiteral = symbolic_binding From, 0 [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.48d: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%From) [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.0f9: %Int.as.ImplicitAs.impl.Convert.type.48d = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.416: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b8b, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.683: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.69e: %Int.as.ImplicitAs.impl.Convert.type.683 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.7cb = facet_value %i32, (%ImplicitAs.impl_witness.416) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.dc6: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.b08: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.dc6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %N.37f, %Int.as.ImplicitAs.impl.Convert.69e [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Int.as.ImplicitAs.impl.Convert.69e, @Int.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %N.37f, %Int.as.ImplicitAs.impl.Convert.specific_fn [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method(%N.37f) [symbolic]
// CHECK:STDOUT:   %array_type.359: type = array_type %Int.as.ImplicitAs.impl.Convert.call, %i32 [symbolic]
// CHECK:STDOUT:   %require_complete.f47: <witness> = require_complete_type %array_type.359 [symbolic]
// CHECK:STDOUT:   %pattern_type.486: type = pattern_type %array_type.359 [symbolic]
// CHECK:STDOUT:   %k.patt: %pattern_type.486 = ref_binding_pattern k [symbolic]
// CHECK:STDOUT:   %k.var_patt: %pattern_type.486 = var_pattern %k.patt [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.lookup_impl_witness.5c0: <witness> = lookup_impl_witness %array_type.359, @DefaultOrUnformed [symbolic]
// CHECK:STDOUT:   %.63b: require_specific_def_type = require_specific_def @T.as.DefaultOrUnformed.impl(%array_type.359) [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.b1c: %DefaultOrUnformed.type = facet_value %array_type.359, (%DefaultOrUnformed.lookup_impl_witness.5c0) [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.WithSelf.Op.type.d4c: type = fn_type @DefaultOrUnformed.WithSelf.Op, @DefaultOrUnformed.WithSelf(%DefaultOrUnformed.facet.b1c) [symbolic]
// CHECK:STDOUT:   %.608: type = fn_type_with_self_type %DefaultOrUnformed.WithSelf.Op.type.d4c, %DefaultOrUnformed.facet.b1c [symbolic]
// CHECK:STDOUT:   %impl.elem0.37d: %.608 = impl_witness_access %DefaultOrUnformed.lookup_impl_witness.5c0, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.b63: <specific function> = specific_impl_function %impl.elem0.37d, @DefaultOrUnformed.WithSelf.Op(%DefaultOrUnformed.facet.b1c) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.822: <witness> = lookup_impl_witness %array_type.359, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.a58: %Destroy.type = facet_value %array_type.359, (%Destroy.lookup_impl_witness.822) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.6c9: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.a58) [symbolic]
// CHECK:STDOUT:   %.a73: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.6c9, %Destroy.facet.a58 [symbolic]
// CHECK:STDOUT:   %impl.elem0.fa7: %.a73 = impl_witness_access %Destroy.lookup_impl_witness.822, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.075: <specific function> = specific_impl_function %impl.elem0.fa7, @Destroy.WithSelf.Op(%Destroy.facet.a58) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.32e: @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert.type (%Int.as.ImplicitAs.impl.Convert.type.48d) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert (constants.%Int.as.ImplicitAs.impl.Convert.0f9)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b8b = impl_witness_table (%Core.import_ref.32e), @Int.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc4_7.2: %Destroy.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc6_19.2: type = facet_access_type %T.loc4_7.1 [symbolic = %T.as_type.loc6_19.2 (constants.%T.as_type.700)]
// CHECK:STDOUT:   %ptr.loc6_19.2: type = ptr_type %T.as_type.loc6_19.2 [symbolic = %ptr.loc6_19.2 (constants.%ptr.b37)]
// CHECK:STDOUT:   %const.loc6_22.2: type = const_type %T.as_type.loc6_19.2 [symbolic = %const.loc6_22.2 (constants.%const)]
// CHECK:STDOUT:   %tuple: %tuple.type.24b = tuple_value (%ptr.loc6_19.2, %const.loc6_22.2) [symbolic = %tuple (constants.%tuple)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%ptr.loc6_19.2, %const.loc6_22.2) [symbolic = %tuple.type (constants.%tuple.type.cd5)]
// CHECK:STDOUT:   %require_complete.loc6: <witness> = require_complete_type %tuple.type [symbolic = %require_complete.loc6 (constants.%require_complete.3b6)]
// CHECK:STDOUT:   %pattern_type.loc6: type = pattern_type %tuple.type [symbolic = %pattern_type.loc6 (constants.%pattern_type.02b)]
// CHECK:STDOUT:   %u.patt.loc6_15.2: @F.%pattern_type.loc6 (%pattern_type.02b) = ref_binding_pattern u [symbolic = %u.patt.loc6_15.2 (constants.%u.patt)]
// CHECK:STDOUT:   %u.var_patt.loc6_3.2: @F.%pattern_type.loc6 (%pattern_type.02b) = var_pattern %u.patt.loc6_15.2 [symbolic = %u.var_patt.loc6_3.2 (constants.%u.var_patt)]
// CHECK:STDOUT:   %.loc6_30.3: require_specific_def_type = require_specific_def @T.as.DefaultOrUnformed.impl(%tuple.type) [symbolic = %.loc6_30.3 (constants.%.f98)]
// CHECK:STDOUT:   %DefaultOrUnformed.lookup_impl_witness.loc6: <witness> = lookup_impl_witness %tuple.type, @DefaultOrUnformed [symbolic = %DefaultOrUnformed.lookup_impl_witness.loc6 (constants.%DefaultOrUnformed.lookup_impl_witness.2ab)]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc6_30.2: %DefaultOrUnformed.type = facet_value %tuple.type, (%DefaultOrUnformed.lookup_impl_witness.loc6) [symbolic = %DefaultOrUnformed.facet.loc6_30.2 (constants.%DefaultOrUnformed.facet.c98)]
// CHECK:STDOUT:   %DefaultOrUnformed.WithSelf.Op.type.loc6: type = fn_type @DefaultOrUnformed.WithSelf.Op, @DefaultOrUnformed.WithSelf(%DefaultOrUnformed.facet.loc6_30.2) [symbolic = %DefaultOrUnformed.WithSelf.Op.type.loc6 (constants.%DefaultOrUnformed.WithSelf.Op.type.f2b)]
// CHECK:STDOUT:   %.loc6_30.4: type = fn_type_with_self_type %DefaultOrUnformed.WithSelf.Op.type.loc6, %DefaultOrUnformed.facet.loc6_30.2 [symbolic = %.loc6_30.4 (constants.%.c04)]
// CHECK:STDOUT:   %impl.elem0.loc6_30.2: @F.%.loc6_30.4 (%.c04) = impl_witness_access %DefaultOrUnformed.lookup_impl_witness.loc6, element0 [symbolic = %impl.elem0.loc6_30.2 (constants.%impl.elem0.91d)]
// CHECK:STDOUT:   %specific_impl_fn.loc6_30.2: <specific function> = specific_impl_function %impl.elem0.loc6_30.2, @DefaultOrUnformed.WithSelf.Op(%DefaultOrUnformed.facet.loc6_30.2) [symbolic = %specific_impl_fn.loc6_30.2 (constants.%specific_impl_fn.495)]
// CHECK:STDOUT:   %struct_type.a.loc7_23.2: type = struct_type {.a: @F.%T.as_type.loc6_19.2 (%T.as_type.700)} [symbolic = %struct_type.a.loc7_23.2 (constants.%struct_type.a)]
// CHECK:STDOUT:   %require_complete.loc7: <witness> = require_complete_type %struct_type.a.loc7_23.2 [symbolic = %require_complete.loc7 (constants.%require_complete.f8e)]
// CHECK:STDOUT:   %pattern_type.loc7: type = pattern_type %struct_type.a.loc7_23.2 [symbolic = %pattern_type.loc7 (constants.%pattern_type.915)]
// CHECK:STDOUT:   %v.patt.loc7_15.2: @F.%pattern_type.loc7 (%pattern_type.915) = ref_binding_pattern v [symbolic = %v.patt.loc7_15.2 (constants.%v.patt)]
// CHECK:STDOUT:   %v.var_patt.loc7_3.2: @F.%pattern_type.loc7 (%pattern_type.915) = var_pattern %v.patt.loc7_15.2 [symbolic = %v.var_patt.loc7_3.2 (constants.%v.var_patt)]
// CHECK:STDOUT:   %.loc7_24.3: require_specific_def_type = require_specific_def @T.as.DefaultOrUnformed.impl(%struct_type.a.loc7_23.2) [symbolic = %.loc7_24.3 (constants.%.b78)]
// CHECK:STDOUT:   %DefaultOrUnformed.lookup_impl_witness.loc7: <witness> = lookup_impl_witness %struct_type.a.loc7_23.2, @DefaultOrUnformed [symbolic = %DefaultOrUnformed.lookup_impl_witness.loc7 (constants.%DefaultOrUnformed.lookup_impl_witness.53f)]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc7_24.2: %DefaultOrUnformed.type = facet_value %struct_type.a.loc7_23.2, (%DefaultOrUnformed.lookup_impl_witness.loc7) [symbolic = %DefaultOrUnformed.facet.loc7_24.2 (constants.%DefaultOrUnformed.facet.103)]
// CHECK:STDOUT:   %DefaultOrUnformed.WithSelf.Op.type.loc7: type = fn_type @DefaultOrUnformed.WithSelf.Op, @DefaultOrUnformed.WithSelf(%DefaultOrUnformed.facet.loc7_24.2) [symbolic = %DefaultOrUnformed.WithSelf.Op.type.loc7 (constants.%DefaultOrUnformed.WithSelf.Op.type.4de)]
// CHECK:STDOUT:   %.loc7_24.4: type = fn_type_with_self_type %DefaultOrUnformed.WithSelf.Op.type.loc7, %DefaultOrUnformed.facet.loc7_24.2 [symbolic = %.loc7_24.4 (constants.%.883)]
// CHECK:STDOUT:   %impl.elem0.loc7_24.2: @F.%.loc7_24.4 (%.883) = impl_witness_access %DefaultOrUnformed.lookup_impl_witness.loc7, element0 [symbolic = %impl.elem0.loc7_24.2 (constants.%impl.elem0.178)]
// CHECK:STDOUT:   %specific_impl_fn.loc7_24.2: <specific function> = specific_impl_function %impl.elem0.loc7_24.2, @DefaultOrUnformed.WithSelf.Op(%DefaultOrUnformed.facet.loc7_24.2) [symbolic = %specific_impl_fn.loc7_24.2 (constants.%specific_impl_fn.b12)]
// CHECK:STDOUT:   %array_type.loc8_27.2: type = array_type constants.%int_5, %T.as_type.loc6_19.2 [symbolic = %array_type.loc8_27.2 (constants.%array_type.50f)]
// CHECK:STDOUT:   %require_complete.loc8: <witness> = require_complete_type %array_type.loc8_27.2 [symbolic = %require_complete.loc8 (constants.%require_complete.f3e)]
// CHECK:STDOUT:   %pattern_type.loc8: type = pattern_type %array_type.loc8_27.2 [symbolic = %pattern_type.loc8 (constants.%pattern_type.78a)]
// CHECK:STDOUT:   %w.patt.loc8_15.2: @F.%pattern_type.loc8 (%pattern_type.78a) = ref_binding_pattern w [symbolic = %w.patt.loc8_15.2 (constants.%w.patt)]
// CHECK:STDOUT:   %w.var_patt.loc8_3.2: @F.%pattern_type.loc8 (%pattern_type.78a) = var_pattern %w.patt.loc8_15.2 [symbolic = %w.var_patt.loc8_3.2 (constants.%w.var_patt)]
// CHECK:STDOUT:   %.loc8_28.3: require_specific_def_type = require_specific_def @T.as.DefaultOrUnformed.impl(%array_type.loc8_27.2) [symbolic = %.loc8_28.3 (constants.%.376)]
// CHECK:STDOUT:   %DefaultOrUnformed.lookup_impl_witness.loc8: <witness> = lookup_impl_witness %array_type.loc8_27.2, @DefaultOrUnformed [symbolic = %DefaultOrUnformed.lookup_impl_witness.loc8 (constants.%DefaultOrUnformed.lookup_impl_witness.1fb)]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc8_28.2: %DefaultOrUnformed.type = facet_value %array_type.loc8_27.2, (%DefaultOrUnformed.lookup_impl_witness.loc8) [symbolic = %DefaultOrUnformed.facet.loc8_28.2 (constants.%DefaultOrUnformed.facet.d60)]
// CHECK:STDOUT:   %DefaultOrUnformed.WithSelf.Op.type.loc8: type = fn_type @DefaultOrUnformed.WithSelf.Op, @DefaultOrUnformed.WithSelf(%DefaultOrUnformed.facet.loc8_28.2) [symbolic = %DefaultOrUnformed.WithSelf.Op.type.loc8 (constants.%DefaultOrUnformed.WithSelf.Op.type.e22)]
// CHECK:STDOUT:   %.loc8_28.4: type = fn_type_with_self_type %DefaultOrUnformed.WithSelf.Op.type.loc8, %DefaultOrUnformed.facet.loc8_28.2 [symbolic = %.loc8_28.4 (constants.%.478)]
// CHECK:STDOUT:   %impl.elem0.loc8_28.2: @F.%.loc8_28.4 (%.478) = impl_witness_access %DefaultOrUnformed.lookup_impl_witness.loc8, element0 [symbolic = %impl.elem0.loc8_28.2 (constants.%impl.elem0.276)]
// CHECK:STDOUT:   %specific_impl_fn.loc8_28.2: <specific function> = specific_impl_function %impl.elem0.loc8_28.2, @DefaultOrUnformed.WithSelf.Op(%DefaultOrUnformed.facet.loc8_28.2) [symbolic = %specific_impl_fn.loc8_28.2 (constants.%specific_impl_fn.6bf)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc8: <witness> = lookup_impl_witness %array_type.loc8_27.2, @Destroy [symbolic = %Destroy.lookup_impl_witness.loc8 (constants.%Destroy.lookup_impl_witness.1f5)]
// CHECK:STDOUT:   %Destroy.facet.loc8_3.3: %Destroy.type = facet_value %array_type.loc8_27.2, (%Destroy.lookup_impl_witness.loc8) [symbolic = %Destroy.facet.loc8_3.3 (constants.%Destroy.facet.664)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc8: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc8_3.3) [symbolic = %Destroy.WithSelf.Op.type.loc8 (constants.%Destroy.WithSelf.Op.type.373)]
// CHECK:STDOUT:   %.loc8_3.4: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.loc8, %Destroy.facet.loc8_3.3 [symbolic = %.loc8_3.4 (constants.%.260)]
// CHECK:STDOUT:   %impl.elem0.loc8_3.2: @F.%.loc8_3.4 (%.260) = impl_witness_access %Destroy.lookup_impl_witness.loc8, element0 [symbolic = %impl.elem0.loc8_3.2 (constants.%impl.elem0.375)]
// CHECK:STDOUT:   %specific_impl_fn.loc8_3.2: <specific function> = specific_impl_function %impl.elem0.loc8_3.2, @Destroy.WithSelf.Op(%Destroy.facet.loc8_3.3) [symbolic = %specific_impl_fn.loc8_3.2 (constants.%specific_impl_fn.f05)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc7: <witness> = lookup_impl_witness %struct_type.a.loc7_23.2, @Destroy [symbolic = %Destroy.lookup_impl_witness.loc7 (constants.%Destroy.lookup_impl_witness.7f9)]
// CHECK:STDOUT:   %Destroy.facet.loc7_3.3: %Destroy.type = facet_value %struct_type.a.loc7_23.2, (%Destroy.lookup_impl_witness.loc7) [symbolic = %Destroy.facet.loc7_3.3 (constants.%Destroy.facet.aa3)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc7: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc7_3.3) [symbolic = %Destroy.WithSelf.Op.type.loc7 (constants.%Destroy.WithSelf.Op.type.ae9)]
// CHECK:STDOUT:   %.loc7_3.3: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.loc7, %Destroy.facet.loc7_3.3 [symbolic = %.loc7_3.3 (constants.%.3c7)]
// CHECK:STDOUT:   %impl.elem0.loc7_3.2: @F.%.loc7_3.3 (%.3c7) = impl_witness_access %Destroy.lookup_impl_witness.loc7, element0 [symbolic = %impl.elem0.loc7_3.2 (constants.%impl.elem0.bb8)]
// CHECK:STDOUT:   %specific_impl_fn.loc7_3.2: <specific function> = specific_impl_function %impl.elem0.loc7_3.2, @Destroy.WithSelf.Op(%Destroy.facet.loc7_3.3) [symbolic = %specific_impl_fn.loc7_3.2 (constants.%specific_impl_fn.799)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc6: <witness> = lookup_impl_witness %tuple.type, @Destroy [symbolic = %Destroy.lookup_impl_witness.loc6 (constants.%Destroy.lookup_impl_witness.d26)]
// CHECK:STDOUT:   %Destroy.facet.loc6_3.3: %Destroy.type = facet_value %tuple.type, (%Destroy.lookup_impl_witness.loc6) [symbolic = %Destroy.facet.loc6_3.3 (constants.%Destroy.facet.7ba)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc6: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc6_3.3) [symbolic = %Destroy.WithSelf.Op.type.loc6 (constants.%Destroy.WithSelf.Op.type.a30)]
// CHECK:STDOUT:   %.loc6_3.4: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.loc6, %Destroy.facet.loc6_3.3 [symbolic = %.loc6_3.4 (constants.%.a0a)]
// CHECK:STDOUT:   %impl.elem0.loc6_3.2: @F.%.loc6_3.4 (%.a0a) = impl_witness_access %Destroy.lookup_impl_witness.loc6, element0 [symbolic = %impl.elem0.loc6_3.2 (constants.%impl.elem0.667)]
// CHECK:STDOUT:   %specific_impl_fn.loc6_3.2: <specific function> = specific_impl_function %impl.elem0.loc6_3.2, @Destroy.WithSelf.Op(%Destroy.facet.loc6_3.3) [symbolic = %specific_impl_fn.loc6_3.2 (constants.%specific_impl_fn.96b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %u.patt.loc6_15.1: @F.%pattern_type.loc6 (%pattern_type.02b) = ref_binding_pattern u [symbolic = %u.patt.loc6_15.2 (constants.%u.patt)]
// CHECK:STDOUT:       %u.var_patt.loc6_3.1: @F.%pattern_type.loc6 (%pattern_type.02b) = var_pattern %u.patt.loc6_15.1 [symbolic = %u.var_patt.loc6_3.2 (constants.%u.var_patt)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u.var: ref @F.%tuple.type (%tuple.type.cd5) = var %u.var_patt.loc6_3.1
// CHECK:STDOUT:     %DefaultOrUnformed.facet.loc6_30.1: %DefaultOrUnformed.type = facet_value constants.%tuple.type.cd5, (constants.%DefaultOrUnformed.lookup_impl_witness.2ab) [symbolic = %DefaultOrUnformed.facet.loc6_30.2 (constants.%DefaultOrUnformed.facet.c98)]
// CHECK:STDOUT:     %.loc6_30.1: %DefaultOrUnformed.type = converted constants.%tuple.type.cd5, %DefaultOrUnformed.facet.loc6_30.1 [symbolic = %DefaultOrUnformed.facet.loc6_30.2 (constants.%DefaultOrUnformed.facet.c98)]
// CHECK:STDOUT:     %as_type.loc6: type = facet_access_type %.loc6_30.1 [symbolic = %tuple.type (constants.%tuple.type.cd5)]
// CHECK:STDOUT:     %.loc6_30.2: type = converted %.loc6_30.1, %as_type.loc6 [symbolic = %tuple.type (constants.%tuple.type.cd5)]
// CHECK:STDOUT:     %impl.elem0.loc6_30.1: @F.%.loc6_30.4 (%.c04) = impl_witness_access constants.%DefaultOrUnformed.lookup_impl_witness.2ab, element0 [symbolic = %impl.elem0.loc6_30.2 (constants.%impl.elem0.91d)]
// CHECK:STDOUT:     %specific_impl_fn.loc6_30.1: <specific function> = specific_impl_function %impl.elem0.loc6_30.1, @DefaultOrUnformed.WithSelf.Op(constants.%DefaultOrUnformed.facet.c98) [symbolic = %specific_impl_fn.loc6_30.2 (constants.%specific_impl_fn.495)]
// CHECK:STDOUT:     %.loc6_3.1: ref @F.%tuple.type (%tuple.type.cd5) = splice_block %u.var {}
// CHECK:STDOUT:     %DefaultOrUnformed.WithSelf.Op.call.loc6: init @F.%tuple.type (%tuple.type.cd5) to %.loc6_3.1 = call %specific_impl_fn.loc6_30.1()
// CHECK:STDOUT:     assign %u.var, %DefaultOrUnformed.WithSelf.Op.call.loc6
// CHECK:STDOUT:     %.loc6_29.1: type = splice_block %.loc6_29.3 [symbolic = %tuple.type (constants.%tuple.type.cd5)] {
// CHECK:STDOUT:       %T.ref.loc6_18: %Destroy.type = name_ref T, %T.loc4_7.2 [symbolic = %T.loc4_7.1 (constants.%T.0e7)]
// CHECK:STDOUT:       %T.as_type.loc6_19.1: type = facet_access_type %T.ref.loc6_18 [symbolic = %T.as_type.loc6_19.2 (constants.%T.as_type.700)]
// CHECK:STDOUT:       %.loc6_19: type = converted %T.ref.loc6_18, %T.as_type.loc6_19.1 [symbolic = %T.as_type.loc6_19.2 (constants.%T.as_type.700)]
// CHECK:STDOUT:       %ptr.loc6_19.1: type = ptr_type %.loc6_19 [symbolic = %ptr.loc6_19.2 (constants.%ptr.b37)]
// CHECK:STDOUT:       %T.ref.loc6_28: %Destroy.type = name_ref T, %T.loc4_7.2 [symbolic = %T.loc4_7.1 (constants.%T.0e7)]
// CHECK:STDOUT:       %T.as_type.loc6_22: type = facet_access_type %T.ref.loc6_28 [symbolic = %T.as_type.loc6_19.2 (constants.%T.as_type.700)]
// CHECK:STDOUT:       %.loc6_22: type = converted %T.ref.loc6_28, %T.as_type.loc6_22 [symbolic = %T.as_type.loc6_19.2 (constants.%T.as_type.700)]
// CHECK:STDOUT:       %const.loc6_22.1: type = const_type %.loc6_22 [symbolic = %const.loc6_22.2 (constants.%const)]
// CHECK:STDOUT:       %.loc6_29.2: %tuple.type.24b = tuple_literal (%ptr.loc6_19.1, %const.loc6_22.1) [symbolic = %tuple (constants.%tuple)]
// CHECK:STDOUT:       %.loc6_29.3: type = converted %.loc6_29.2, constants.%tuple.type.cd5 [symbolic = %tuple.type (constants.%tuple.type.cd5)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: ref @F.%tuple.type (%tuple.type.cd5) = ref_binding u, %u.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %v.patt.loc7_15.1: @F.%pattern_type.loc7 (%pattern_type.915) = ref_binding_pattern v [symbolic = %v.patt.loc7_15.2 (constants.%v.patt)]
// CHECK:STDOUT:       %v.var_patt.loc7_3.1: @F.%pattern_type.loc7 (%pattern_type.915) = var_pattern %v.patt.loc7_15.1 [symbolic = %v.var_patt.loc7_3.2 (constants.%v.var_patt)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v.var: ref @F.%struct_type.a.loc7_23.2 (%struct_type.a) = var %v.var_patt.loc7_3.1
// CHECK:STDOUT:     %DefaultOrUnformed.facet.loc7_24.1: %DefaultOrUnformed.type = facet_value constants.%struct_type.a, (constants.%DefaultOrUnformed.lookup_impl_witness.53f) [symbolic = %DefaultOrUnformed.facet.loc7_24.2 (constants.%DefaultOrUnformed.facet.103)]
// CHECK:STDOUT:     %.loc7_24.1: %DefaultOrUnformed.type = converted constants.%struct_type.a, %DefaultOrUnformed.facet.loc7_24.1 [symbolic = %DefaultOrUnformed.facet.loc7_24.2 (constants.%DefaultOrUnformed.facet.103)]
// CHECK:STDOUT:     %as_type.loc7: type = facet_access_type %.loc7_24.1 [symbolic = %struct_type.a.loc7_23.2 (constants.%struct_type.a)]
// CHECK:STDOUT:     %.loc7_24.2: type = converted %.loc7_24.1, %as_type.loc7 [symbolic = %struct_type.a.loc7_23.2 (constants.%struct_type.a)]
// CHECK:STDOUT:     %impl.elem0.loc7_24.1: @F.%.loc7_24.4 (%.883) = impl_witness_access constants.%DefaultOrUnformed.lookup_impl_witness.53f, element0 [symbolic = %impl.elem0.loc7_24.2 (constants.%impl.elem0.178)]
// CHECK:STDOUT:     %specific_impl_fn.loc7_24.1: <specific function> = specific_impl_function %impl.elem0.loc7_24.1, @DefaultOrUnformed.WithSelf.Op(constants.%DefaultOrUnformed.facet.103) [symbolic = %specific_impl_fn.loc7_24.2 (constants.%specific_impl_fn.b12)]
// CHECK:STDOUT:     %DefaultOrUnformed.WithSelf.Op.call.loc7: init @F.%struct_type.a.loc7_23.2 (%struct_type.a) = call %specific_impl_fn.loc7_24.1()
// CHECK:STDOUT:     assign %v.var, %DefaultOrUnformed.WithSelf.Op.call.loc7
// CHECK:STDOUT:     %.loc7_23: type = splice_block %struct_type.a.loc7_23.1 [symbolic = %struct_type.a.loc7_23.2 (constants.%struct_type.a)] {
// CHECK:STDOUT:       %T.ref.loc7: %Destroy.type = name_ref T, %T.loc4_7.2 [symbolic = %T.loc4_7.1 (constants.%T.0e7)]
// CHECK:STDOUT:       %T.as_type.loc7: type = facet_access_type %T.ref.loc7 [symbolic = %T.as_type.loc6_19.2 (constants.%T.as_type.700)]
// CHECK:STDOUT:       %.loc7_22: type = converted %T.ref.loc7, %T.as_type.loc7 [symbolic = %T.as_type.loc6_19.2 (constants.%T.as_type.700)]
// CHECK:STDOUT:       %struct_type.a.loc7_23.1: type = struct_type {.a: @F.%T.as_type.loc6_19.2 (%T.as_type.700)} [symbolic = %struct_type.a.loc7_23.2 (constants.%struct_type.a)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: ref @F.%struct_type.a.loc7_23.2 (%struct_type.a) = ref_binding v, %v.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %w.patt.loc8_15.1: @F.%pattern_type.loc8 (%pattern_type.78a) = ref_binding_pattern w [symbolic = %w.patt.loc8_15.2 (constants.%w.patt)]
// CHECK:STDOUT:       %w.var_patt.loc8_3.1: @F.%pattern_type.loc8 (%pattern_type.78a) = var_pattern %w.patt.loc8_15.1 [symbolic = %w.var_patt.loc8_3.2 (constants.%w.var_patt)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %w.var: ref @F.%array_type.loc8_27.2 (%array_type.50f) = var %w.var_patt.loc8_3.1
// CHECK:STDOUT:     %DefaultOrUnformed.facet.loc8_28.1: %DefaultOrUnformed.type = facet_value constants.%array_type.50f, (constants.%DefaultOrUnformed.lookup_impl_witness.1fb) [symbolic = %DefaultOrUnformed.facet.loc8_28.2 (constants.%DefaultOrUnformed.facet.d60)]
// CHECK:STDOUT:     %.loc8_28.1: %DefaultOrUnformed.type = converted constants.%array_type.50f, %DefaultOrUnformed.facet.loc8_28.1 [symbolic = %DefaultOrUnformed.facet.loc8_28.2 (constants.%DefaultOrUnformed.facet.d60)]
// CHECK:STDOUT:     %as_type.loc8: type = facet_access_type %.loc8_28.1 [symbolic = %array_type.loc8_27.2 (constants.%array_type.50f)]
// CHECK:STDOUT:     %.loc8_28.2: type = converted %.loc8_28.1, %as_type.loc8 [symbolic = %array_type.loc8_27.2 (constants.%array_type.50f)]
// CHECK:STDOUT:     %impl.elem0.loc8_28.1: @F.%.loc8_28.4 (%.478) = impl_witness_access constants.%DefaultOrUnformed.lookup_impl_witness.1fb, element0 [symbolic = %impl.elem0.loc8_28.2 (constants.%impl.elem0.276)]
// CHECK:STDOUT:     %specific_impl_fn.loc8_28.1: <specific function> = specific_impl_function %impl.elem0.loc8_28.1, @DefaultOrUnformed.WithSelf.Op(constants.%DefaultOrUnformed.facet.d60) [symbolic = %specific_impl_fn.loc8_28.2 (constants.%specific_impl_fn.6bf)]
// CHECK:STDOUT:     %.loc8_3.1: ref @F.%array_type.loc8_27.2 (%array_type.50f) = splice_block %w.var {}
// CHECK:STDOUT:     %DefaultOrUnformed.WithSelf.Op.call.loc8: init @F.%array_type.loc8_27.2 (%array_type.50f) to %.loc8_3.1 = call %specific_impl_fn.loc8_28.1()
// CHECK:STDOUT:     assign %w.var, %DefaultOrUnformed.WithSelf.Op.call.loc8
// CHECK:STDOUT:     %.loc8_27: type = splice_block %array_type.loc8_27.1 [symbolic = %array_type.loc8_27.2 (constants.%array_type.50f)] {
// CHECK:STDOUT:       %T.ref.loc8: %Destroy.type = name_ref T, %T.loc4_7.2 [symbolic = %T.loc4_7.1 (constants.%T.0e7)]
// CHECK:STDOUT:       %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5]
// CHECK:STDOUT:       %T.as_type.loc8: type = facet_access_type %T.ref.loc8 [symbolic = %T.as_type.loc6_19.2 (constants.%T.as_type.700)]
// CHECK:STDOUT:       %.loc8_23: type = converted %T.ref.loc8, %T.as_type.loc8 [symbolic = %T.as_type.loc6_19.2 (constants.%T.as_type.700)]
// CHECK:STDOUT:       %array_type.loc8_27.1: type = array_type %int_5, %.loc8_23 [symbolic = %array_type.loc8_27.2 (constants.%array_type.50f)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %w: ref @F.%array_type.loc8_27.2 (%array_type.50f) = ref_binding w, %w.var
// CHECK:STDOUT:     %impl.elem0.loc8_3.1: @F.%.loc8_3.4 (%.260) = impl_witness_access constants.%Destroy.lookup_impl_witness.1f5, element0 [symbolic = %impl.elem0.loc8_3.2 (constants.%impl.elem0.375)]
// CHECK:STDOUT:     %bound_method.loc8_3.1: <bound method> = bound_method %w.var, %impl.elem0.loc8_3.1
// CHECK:STDOUT:     %Destroy.facet.loc8_3.1: %Destroy.type = facet_value constants.%array_type.50f, (constants.%Destroy.lookup_impl_witness.1f5) [symbolic = %Destroy.facet.loc8_3.3 (constants.%Destroy.facet.664)]
// CHECK:STDOUT:     %.loc8_3.2: %Destroy.type = converted constants.%array_type.50f, %Destroy.facet.loc8_3.1 [symbolic = %Destroy.facet.loc8_3.3 (constants.%Destroy.facet.664)]
// CHECK:STDOUT:     %Destroy.facet.loc8_3.2: %Destroy.type = facet_value constants.%array_type.50f, (constants.%Destroy.lookup_impl_witness.1f5) [symbolic = %Destroy.facet.loc8_3.3 (constants.%Destroy.facet.664)]
// CHECK:STDOUT:     %.loc8_3.3: %Destroy.type = converted constants.%array_type.50f, %Destroy.facet.loc8_3.2 [symbolic = %Destroy.facet.loc8_3.3 (constants.%Destroy.facet.664)]
// CHECK:STDOUT:     %specific_impl_fn.loc8_3.1: <specific function> = specific_impl_function %impl.elem0.loc8_3.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.664) [symbolic = %specific_impl_fn.loc8_3.2 (constants.%specific_impl_fn.f05)]
// CHECK:STDOUT:     %bound_method.loc8_3.2: <bound method> = bound_method %w.var, %specific_impl_fn.loc8_3.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call.loc8: init %empty_tuple.type = call %bound_method.loc8_3.2(%w.var)
// CHECK:STDOUT:     %impl.elem0.loc7_3.1: @F.%.loc7_3.3 (%.3c7) = impl_witness_access constants.%Destroy.lookup_impl_witness.7f9, element0 [symbolic = %impl.elem0.loc7_3.2 (constants.%impl.elem0.bb8)]
// CHECK:STDOUT:     %bound_method.loc7_3.1: <bound method> = bound_method %v.var, %impl.elem0.loc7_3.1
// CHECK:STDOUT:     %Destroy.facet.loc7_3.1: %Destroy.type = facet_value constants.%struct_type.a, (constants.%Destroy.lookup_impl_witness.7f9) [symbolic = %Destroy.facet.loc7_3.3 (constants.%Destroy.facet.aa3)]
// CHECK:STDOUT:     %.loc7_3.1: %Destroy.type = converted constants.%struct_type.a, %Destroy.facet.loc7_3.1 [symbolic = %Destroy.facet.loc7_3.3 (constants.%Destroy.facet.aa3)]
// CHECK:STDOUT:     %Destroy.facet.loc7_3.2: %Destroy.type = facet_value constants.%struct_type.a, (constants.%Destroy.lookup_impl_witness.7f9) [symbolic = %Destroy.facet.loc7_3.3 (constants.%Destroy.facet.aa3)]
// CHECK:STDOUT:     %.loc7_3.2: %Destroy.type = converted constants.%struct_type.a, %Destroy.facet.loc7_3.2 [symbolic = %Destroy.facet.loc7_3.3 (constants.%Destroy.facet.aa3)]
// CHECK:STDOUT:     %specific_impl_fn.loc7_3.1: <specific function> = specific_impl_function %impl.elem0.loc7_3.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.aa3) [symbolic = %specific_impl_fn.loc7_3.2 (constants.%specific_impl_fn.799)]
// CHECK:STDOUT:     %bound_method.loc7_3.2: <bound method> = bound_method %v.var, %specific_impl_fn.loc7_3.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call.loc7: init %empty_tuple.type = call %bound_method.loc7_3.2(%v.var)
// CHECK:STDOUT:     %impl.elem0.loc6_3.1: @F.%.loc6_3.4 (%.a0a) = impl_witness_access constants.%Destroy.lookup_impl_witness.d26, element0 [symbolic = %impl.elem0.loc6_3.2 (constants.%impl.elem0.667)]
// CHECK:STDOUT:     %bound_method.loc6_3.1: <bound method> = bound_method %u.var, %impl.elem0.loc6_3.1
// CHECK:STDOUT:     %Destroy.facet.loc6_3.1: %Destroy.type = facet_value constants.%tuple.type.cd5, (constants.%Destroy.lookup_impl_witness.d26) [symbolic = %Destroy.facet.loc6_3.3 (constants.%Destroy.facet.7ba)]
// CHECK:STDOUT:     %.loc6_3.2: %Destroy.type = converted constants.%tuple.type.cd5, %Destroy.facet.loc6_3.1 [symbolic = %Destroy.facet.loc6_3.3 (constants.%Destroy.facet.7ba)]
// CHECK:STDOUT:     %Destroy.facet.loc6_3.2: %Destroy.type = facet_value constants.%tuple.type.cd5, (constants.%Destroy.lookup_impl_witness.d26) [symbolic = %Destroy.facet.loc6_3.3 (constants.%Destroy.facet.7ba)]
// CHECK:STDOUT:     %.loc6_3.3: %Destroy.type = converted constants.%tuple.type.cd5, %Destroy.facet.loc6_3.2 [symbolic = %Destroy.facet.loc6_3.3 (constants.%Destroy.facet.7ba)]
// CHECK:STDOUT:     %specific_impl_fn.loc6_3.1: <specific function> = specific_impl_function %impl.elem0.loc6_3.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.7ba) [symbolic = %specific_impl_fn.loc6_3.2 (constants.%specific_impl_fn.96b)]
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
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %N.loc12_7.1, constants.%Int.as.ImplicitAs.impl.Convert.69e [symbolic = %Int.as.ImplicitAs.impl.Convert.bound (constants.%Int.as.ImplicitAs.impl.Convert.bound)]
// CHECK:STDOUT:   %bound_method.loc14_28.3: <bound method> = bound_method %N.loc12_7.1, constants.%Int.as.ImplicitAs.impl.Convert.specific_fn [symbolic = %bound_method.loc14_28.3 (constants.%bound_method)]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.call.loc14_28.2: init Core.IntLiteral = call %bound_method.loc14_28.3(%N.loc12_7.1) [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc14_28.2 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:   %array_type.loc14_29.2: type = array_type %Int.as.ImplicitAs.impl.Convert.call.loc14_28.2, constants.%i32 [symbolic = %array_type.loc14_29.2 (constants.%array_type.359)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %array_type.loc14_29.2 [symbolic = %require_complete (constants.%require_complete.f47)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %array_type.loc14_29.2 [symbolic = %pattern_type (constants.%pattern_type.486)]
// CHECK:STDOUT:   %k.patt.loc14_15.2: @G.%pattern_type (%pattern_type.486) = ref_binding_pattern k [symbolic = %k.patt.loc14_15.2 (constants.%k.patt)]
// CHECK:STDOUT:   %k.var_patt.loc14_3.2: @G.%pattern_type (%pattern_type.486) = var_pattern %k.patt.loc14_15.2 [symbolic = %k.var_patt.loc14_3.2 (constants.%k.var_patt)]
// CHECK:STDOUT:   %.loc14_30.3: require_specific_def_type = require_specific_def @T.as.DefaultOrUnformed.impl(%array_type.loc14_29.2) [symbolic = %.loc14_30.3 (constants.%.63b)]
// CHECK:STDOUT:   %DefaultOrUnformed.lookup_impl_witness: <witness> = lookup_impl_witness %array_type.loc14_29.2, @DefaultOrUnformed [symbolic = %DefaultOrUnformed.lookup_impl_witness (constants.%DefaultOrUnformed.lookup_impl_witness.5c0)]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc14_30.2: %DefaultOrUnformed.type = facet_value %array_type.loc14_29.2, (%DefaultOrUnformed.lookup_impl_witness) [symbolic = %DefaultOrUnformed.facet.loc14_30.2 (constants.%DefaultOrUnformed.facet.b1c)]
// CHECK:STDOUT:   %DefaultOrUnformed.WithSelf.Op.type: type = fn_type @DefaultOrUnformed.WithSelf.Op, @DefaultOrUnformed.WithSelf(%DefaultOrUnformed.facet.loc14_30.2) [symbolic = %DefaultOrUnformed.WithSelf.Op.type (constants.%DefaultOrUnformed.WithSelf.Op.type.d4c)]
// CHECK:STDOUT:   %.loc14_30.4: type = fn_type_with_self_type %DefaultOrUnformed.WithSelf.Op.type, %DefaultOrUnformed.facet.loc14_30.2 [symbolic = %.loc14_30.4 (constants.%.608)]
// CHECK:STDOUT:   %impl.elem0.loc14_30.2: @G.%.loc14_30.4 (%.608) = impl_witness_access %DefaultOrUnformed.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc14_30.2 (constants.%impl.elem0.37d)]
// CHECK:STDOUT:   %specific_impl_fn.loc14_30.2: <specific function> = specific_impl_function %impl.elem0.loc14_30.2, @DefaultOrUnformed.WithSelf.Op(%DefaultOrUnformed.facet.loc14_30.2) [symbolic = %specific_impl_fn.loc14_30.2 (constants.%specific_impl_fn.b63)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %array_type.loc14_29.2, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.822)]
// CHECK:STDOUT:   %Destroy.facet.loc14_3.3: %Destroy.type = facet_value %array_type.loc14_29.2, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet.loc14_3.3 (constants.%Destroy.facet.a58)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc14_3.3) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.6c9)]
// CHECK:STDOUT:   %.loc14_3.4: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Destroy.facet.loc14_3.3 [symbolic = %.loc14_3.4 (constants.%.a73)]
// CHECK:STDOUT:   %impl.elem0.loc14_3.2: @G.%.loc14_3.4 (%.a73) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc14_3.2 (constants.%impl.elem0.fa7)]
// CHECK:STDOUT:   %specific_impl_fn.loc14_3.2: <specific function> = specific_impl_function %impl.elem0.loc14_3.2, @Destroy.WithSelf.Op(%Destroy.facet.loc14_3.3) [symbolic = %specific_impl_fn.loc14_3.2 (constants.%specific_impl_fn.075)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %k.patt.loc14_15.1: @G.%pattern_type (%pattern_type.486) = ref_binding_pattern k [symbolic = %k.patt.loc14_15.2 (constants.%k.patt)]
// CHECK:STDOUT:       %k.var_patt.loc14_3.1: @G.%pattern_type (%pattern_type.486) = var_pattern %k.patt.loc14_15.1 [symbolic = %k.var_patt.loc14_3.2 (constants.%k.var_patt)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %k.var: ref @G.%array_type.loc14_29.2 (%array_type.359) = var %k.var_patt.loc14_3.1
// CHECK:STDOUT:     %DefaultOrUnformed.facet.loc14_30.1: %DefaultOrUnformed.type = facet_value constants.%array_type.359, (constants.%DefaultOrUnformed.lookup_impl_witness.5c0) [symbolic = %DefaultOrUnformed.facet.loc14_30.2 (constants.%DefaultOrUnformed.facet.b1c)]
// CHECK:STDOUT:     %.loc14_30.1: %DefaultOrUnformed.type = converted constants.%array_type.359, %DefaultOrUnformed.facet.loc14_30.1 [symbolic = %DefaultOrUnformed.facet.loc14_30.2 (constants.%DefaultOrUnformed.facet.b1c)]
// CHECK:STDOUT:     %as_type: type = facet_access_type %.loc14_30.1 [symbolic = %array_type.loc14_29.2 (constants.%array_type.359)]
// CHECK:STDOUT:     %.loc14_30.2: type = converted %.loc14_30.1, %as_type [symbolic = %array_type.loc14_29.2 (constants.%array_type.359)]
// CHECK:STDOUT:     %impl.elem0.loc14_30.1: @G.%.loc14_30.4 (%.608) = impl_witness_access constants.%DefaultOrUnformed.lookup_impl_witness.5c0, element0 [symbolic = %impl.elem0.loc14_30.2 (constants.%impl.elem0.37d)]
// CHECK:STDOUT:     %specific_impl_fn.loc14_30.1: <specific function> = specific_impl_function %impl.elem0.loc14_30.1, @DefaultOrUnformed.WithSelf.Op(constants.%DefaultOrUnformed.facet.b1c) [symbolic = %specific_impl_fn.loc14_30.2 (constants.%specific_impl_fn.b63)]
// CHECK:STDOUT:     %.loc14_3.1: ref @G.%array_type.loc14_29.2 (%array_type.359) = splice_block %k.var {}
// CHECK:STDOUT:     %DefaultOrUnformed.WithSelf.Op.call: init @G.%array_type.loc14_29.2 (%array_type.359) to %.loc14_3.1 = call %specific_impl_fn.loc14_30.1()
// CHECK:STDOUT:     assign %k.var, %DefaultOrUnformed.WithSelf.Op.call
// CHECK:STDOUT:     %.loc14_29: type = splice_block %array_type.loc14_29.1 [symbolic = %array_type.loc14_29.2 (constants.%array_type.359)] {
// CHECK:STDOUT:       %i32.loc14: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %N.ref: %i32 = name_ref N, %N.loc12_7.2 [symbolic = %N.loc12_7.1 (constants.%N.37f)]
// CHECK:STDOUT:       %impl.elem0.loc14_28: %.b08 = impl_witness_access constants.%ImplicitAs.impl_witness.416, element0 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.69e]
// CHECK:STDOUT:       %bound_method.loc14_28.1: <bound method> = bound_method %N.ref, %impl.elem0.loc14_28 [symbolic = %Int.as.ImplicitAs.impl.Convert.bound (constants.%Int.as.ImplicitAs.impl.Convert.bound)]
// CHECK:STDOUT:       %specific_fn: <specific function> = specific_function %impl.elem0.loc14_28, @Int.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Int.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:       %bound_method.loc14_28.2: <bound method> = bound_method %N.ref, %specific_fn [symbolic = %bound_method.loc14_28.3 (constants.%bound_method)]
// CHECK:STDOUT:       %Int.as.ImplicitAs.impl.Convert.call.loc14_28.1: init Core.IntLiteral = call %bound_method.loc14_28.2(%N.ref) [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc14_28.2 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:       %.loc14_28.1: Core.IntLiteral = value_of_initializer %Int.as.ImplicitAs.impl.Convert.call.loc14_28.1 [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc14_28.2 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:       %.loc14_28.2: Core.IntLiteral = converted %N.ref, %.loc14_28.1 [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc14_28.2 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:       %array_type.loc14_29.1: type = array_type %.loc14_28.2, %i32.loc14 [symbolic = %array_type.loc14_29.2 (constants.%array_type.359)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %k: ref @G.%array_type.loc14_29.2 (%array_type.359) = ref_binding k, %k.var
// CHECK:STDOUT:     %impl.elem0.loc14_3.1: @G.%.loc14_3.4 (%.a73) = impl_witness_access constants.%Destroy.lookup_impl_witness.822, element0 [symbolic = %impl.elem0.loc14_3.2 (constants.%impl.elem0.fa7)]
// CHECK:STDOUT:     %bound_method.loc14_3.1: <bound method> = bound_method %k.var, %impl.elem0.loc14_3.1
// CHECK:STDOUT:     %Destroy.facet.loc14_3.1: %Destroy.type = facet_value constants.%array_type.359, (constants.%Destroy.lookup_impl_witness.822) [symbolic = %Destroy.facet.loc14_3.3 (constants.%Destroy.facet.a58)]
// CHECK:STDOUT:     %.loc14_3.2: %Destroy.type = converted constants.%array_type.359, %Destroy.facet.loc14_3.1 [symbolic = %Destroy.facet.loc14_3.3 (constants.%Destroy.facet.a58)]
// CHECK:STDOUT:     %Destroy.facet.loc14_3.2: %Destroy.type = facet_value constants.%array_type.359, (constants.%Destroy.lookup_impl_witness.822) [symbolic = %Destroy.facet.loc14_3.3 (constants.%Destroy.facet.a58)]
// CHECK:STDOUT:     %.loc14_3.3: %Destroy.type = converted constants.%array_type.359, %Destroy.facet.loc14_3.2 [symbolic = %Destroy.facet.loc14_3.3 (constants.%Destroy.facet.a58)]
// CHECK:STDOUT:     %specific_impl_fn.loc14_3.1: <specific function> = specific_impl_function %impl.elem0.loc14_3.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.a58) [symbolic = %specific_impl_fn.loc14_3.2 (constants.%specific_impl_fn.075)]
// CHECK:STDOUT:     %bound_method.loc14_3.2: <bound method> = bound_method %k.var, %specific_impl_fn.loc14_3.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc14_3.2(%k.var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.0e7) {
// CHECK:STDOUT:   %T.patt.loc4_7.2 => constants.%T.patt.351
// CHECK:STDOUT:   %T.loc4_7.1 => constants.%T.0e7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%N.37f) {
// CHECK:STDOUT:   %N.patt.loc12_7.2 => constants.%N.patt.38a
// CHECK:STDOUT:   %N.loc12_7.1 => constants.%N.37f
// CHECK:STDOUT: }
// CHECK:STDOUT:

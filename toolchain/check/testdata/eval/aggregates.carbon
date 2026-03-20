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
fn F(T:! type) {
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
// CHECK:STDOUT:   %tuple.95a: %tuple.type.24b = tuple_value (%i32, %i32) [concrete]
// CHECK:STDOUT:   %tuple.type.d07: type = tuple_type (%i32, %i32) [concrete]
// CHECK:STDOUT:   %pattern_type.511: type = pattern_type %tuple.type.d07 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.type.f94: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.ad8: %tuple.type.f94 = tuple_value (%int_1.5b8, %int_2.ecc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.6bc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.74f, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.6bc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.b37: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.545: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.b37, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.38b: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4e5: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.646: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.21c: %tuple.type.d07 = tuple_value (%int_1.5d2, %int_2.ef8) [concrete]
// CHECK:STDOUT:   %tuple.elem0: ref %i32 = tuple_access file.%tuple_copy.var, element0 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.824: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.9b9: %Int.as.Copy.impl.Op.type.824 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.f17: <witness> = impl_witness imports.%Copy.impl_witness_table.e76, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.546: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.664: %Int.as.Copy.impl.Op.type.546 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.f17) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.081: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.8e2: type = fn_type_with_self_type %Copy.WithSelf.Op.type.081, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound.ee3: <bound method> = bound_method %int_1.5d2, %Int.as.Copy.impl.Op.664 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.664, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.4d7: <bound method> = bound_method %int_1.5d2, %Int.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %tuple.elem1: ref %i32 = tuple_access file.%tuple_copy.var, element1 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound.5e8: <bound method> = bound_method %int_2.ef8, %Int.as.Copy.impl.Op.664 [concrete]
// CHECK:STDOUT:   %bound_method.f15: <bound method> = bound_method %int_2.ef8, %Int.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %struct_type.a.b.c: type = struct_type {.a: %i32, .b: %i32, .c: %i32} [concrete]
// CHECK:STDOUT:   %pattern_type.8ae: type = pattern_type %struct_type.a.b.c [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %struct_type.c.b.a: type = struct_type {.c: Core.IntLiteral, .b: Core.IntLiteral, .a: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct.c75: %struct_type.c.b.a = struct_value (%int_3.1ba, %int_2.ecc, %int_1.5b8) [concrete]
// CHECK:STDOUT:   %struct_type.b.a.c: type = struct_type {.b: %i32, .a: %i32, .c: %i32} [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.061: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.fa7: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.822: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %struct.21d: %struct_type.b.a.c = struct_value (%int_2.ef8, %int_1.5d2, %int_3.822) [concrete]
// CHECK:STDOUT:   %.460: ref %i32 = struct_access file.%struct_copy.var, element1 [concrete]
// CHECK:STDOUT:   %.61f: ref %i32 = struct_access file.%struct_copy.var, element0 [concrete]
// CHECK:STDOUT:   %.f68: ref %i32 = struct_access file.%struct_copy.var, element2 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound.ecc: <bound method> = bound_method %int_3.822, %Int.as.Copy.impl.Op.664 [concrete]
// CHECK:STDOUT:   %bound_method.4f5: <bound method> = bound_method %int_3.822, %Int.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %struct.cff: %struct_type.a.b.c = struct_value (%int_1.5d2, %int_2.ef8, %int_3.822) [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_1.5b8, %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.a98: type = pattern_type %array_type [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %tuple.type.985: type = tuple_type (Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.4f2: %tuple.type.985 = tuple_value (%int_0.5c6) [concrete]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %int_7: Core.IntLiteral = int_value 7 [concrete]
// CHECK:STDOUT:   %int_9: Core.IntLiteral = int_value 9 [concrete]
// CHECK:STDOUT:   %tuple.type.d46: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.869: %tuple.type.d46 = tuple_value (%int_5, %int_7, %int_1.5b8, %int_9) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.897: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.d2e: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%int_0.6a9) [concrete]
// CHECK:STDOUT:   %struct_type.a.b: type = struct_type {.a: Core.IntLiteral, .b: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct.a81: %struct_type.a.b = struct_value (%int_3.1ba, %int_1.5b8) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.42d: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.74f = impl_witness_table (%Core.import_ref.42d), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.18d: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.824) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.9b9)]
// CHECK:STDOUT:   %Copy.impl_witness_table.e76 = impl_witness_table (%Core.import_ref.18d), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %tuple_copy.patt: %pattern_type.511 = ref_binding_pattern tuple_copy [concrete]
// CHECK:STDOUT:     %tuple_copy.var_patt: %pattern_type.511 = var_pattern %tuple_copy.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %tuple_copy.var: ref %tuple.type.d07 = var %tuple_copy.var_patt [concrete]
// CHECK:STDOUT:   %.loc4_26.1: type = splice_block %.loc4_26.3 [concrete = constants.%tuple.type.d07] {
// CHECK:STDOUT:     %i32.loc4_18: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc4_23: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc4_26.2: %tuple.type.24b = tuple_literal (%i32.loc4_18, %i32.loc4_23) [concrete = constants.%tuple.95a]
// CHECK:STDOUT:     %.loc4_26.3: type = converted %.loc4_26.2, constants.%tuple.type.d07 [concrete = constants.%tuple.type.d07]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %tuple_copy: ref %tuple.type.d07 = ref_binding tuple_copy, %tuple_copy.var [concrete = %tuple_copy.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %struct_copy.patt: %pattern_type.8ae = ref_binding_pattern struct_copy [concrete]
// CHECK:STDOUT:     %struct_copy.var_patt: %pattern_type.8ae = var_pattern %struct_copy.patt [concrete]
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
// CHECK:STDOUT:     %tuple_index.patt: %pattern_type.a98 = ref_binding_pattern tuple_index [concrete]
// CHECK:STDOUT:     %tuple_index.var_patt: %pattern_type.a98 = var_pattern %tuple_index.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %tuple_index.var: ref %array_type = var %tuple_index.var_patt [concrete]
// CHECK:STDOUT:   %.loc8: type = splice_block %array_type.loc8 [concrete = constants.%array_type] {
// CHECK:STDOUT:     %i32.loc8: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %array_type.loc8: type = array_type %int_1.loc8, %i32.loc8 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %tuple_index: ref %array_type = ref_binding tuple_index, %tuple_index.var [concrete = %tuple_index.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %struct_access.patt: %pattern_type.a98 = ref_binding_pattern struct_access [concrete]
// CHECK:STDOUT:     %struct_access.var_patt: %pattern_type.a98 = var_pattern %struct_access.patt [concrete]
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
// CHECK:STDOUT:   %.loc4_49.1: %tuple.type.24b = tuple_literal (%i32.loc4_41, %i32.loc4_46) [concrete = constants.%tuple.95a]
// CHECK:STDOUT:   %.loc4_49.2: type = converted %.loc4_49.1, constants.%tuple.type.d07 [concrete = constants.%tuple.type.d07]
// CHECK:STDOUT:   %impl.elem0.loc4_35.1: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc4_35.1: <bound method> = bound_method %int_1.loc4, %impl.elem0.loc4_35.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215]
// CHECK:STDOUT:   %specific_fn.loc4_35.1: <specific function> = specific_function %impl.elem0.loc4_35.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc4_35.2: <bound method> = bound_method %int_1.loc4, %specific_fn.loc4_35.1 [concrete = constants.%bound_method.38b]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc4_35.1: init %i32 = call %bound_method.loc4_35.2(%int_1.loc4) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc4_35.2: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc4_35.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc4_35.3: %i32 = converted %int_1.loc4, %.loc4_35.2 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc4_35.2: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc4_35.3: <bound method> = bound_method %int_2.loc4, %impl.elem0.loc4_35.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4e5]
// CHECK:STDOUT:   %specific_fn.loc4_35.2: <specific function> = specific_function %impl.elem0.loc4_35.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc4_35.4: <bound method> = bound_method %int_2.loc4, %specific_fn.loc4_35.2 [concrete = constants.%bound_method.646]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc4_35.2: init %i32 = call %bound_method.loc4_35.4(%int_2.loc4) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc4_35.4: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc4_35.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc4_35.5: %i32 = converted %int_2.loc4, %.loc4_35.4 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %tuple.loc4: %tuple.type.d07 = tuple_value (%.loc4_35.3, %.loc4_35.5) [concrete = constants.%tuple.21c]
// CHECK:STDOUT:   %.loc4_37.1: %tuple.type.d07 = converted %.loc4_35.1, %tuple.loc4 [concrete = constants.%tuple.21c]
// CHECK:STDOUT:   %tuple.elem0.loc4_37.1: %i32 = tuple_access %.loc4_37.1, element0 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc4_37.1: %.8e2 = impl_witness_access constants.%Copy.impl_witness.f17, element0 [concrete = constants.%Int.as.Copy.impl.Op.664]
// CHECK:STDOUT:   %bound_method.loc4_37.1: <bound method> = bound_method %tuple.elem0.loc4_37.1, %impl.elem0.loc4_37.1 [concrete = constants.%Int.as.Copy.impl.Op.bound.ee3]
// CHECK:STDOUT:   %specific_fn.loc4_37.1: <specific function> = specific_function %impl.elem0.loc4_37.1, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc4_37.2: <bound method> = bound_method %tuple.elem0.loc4_37.1, %specific_fn.loc4_37.1 [concrete = constants.%bound_method.4d7]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc4_37.1: init %i32 = call %bound_method.loc4_37.2(%tuple.elem0.loc4_37.1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %tuple.elem0.loc4_37.2: ref %i32 = tuple_access file.%tuple_copy.var, element0 [concrete = constants.%tuple.elem0]
// CHECK:STDOUT:   %.loc4_37.2: init %i32 to %tuple.elem0.loc4_37.2 = in_place_init %Int.as.Copy.impl.Op.call.loc4_37.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %tuple.elem1.loc4_37.1: %i32 = tuple_access %.loc4_37.1, element1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %impl.elem0.loc4_37.2: %.8e2 = impl_witness_access constants.%Copy.impl_witness.f17, element0 [concrete = constants.%Int.as.Copy.impl.Op.664]
// CHECK:STDOUT:   %bound_method.loc4_37.3: <bound method> = bound_method %tuple.elem1.loc4_37.1, %impl.elem0.loc4_37.2 [concrete = constants.%Int.as.Copy.impl.Op.bound.5e8]
// CHECK:STDOUT:   %specific_fn.loc4_37.2: <specific function> = specific_function %impl.elem0.loc4_37.2, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc4_37.4: <bound method> = bound_method %tuple.elem1.loc4_37.1, %specific_fn.loc4_37.2 [concrete = constants.%bound_method.f15]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc4_37.2: init %i32 = call %bound_method.loc4_37.4(%tuple.elem1.loc4_37.1) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %tuple.elem1.loc4_37.2: ref %i32 = tuple_access file.%tuple_copy.var, element1 [concrete = constants.%tuple.elem1]
// CHECK:STDOUT:   %.loc4_37.3: init %i32 to %tuple.elem1.loc4_37.2 = in_place_init %Int.as.Copy.impl.Op.call.loc4_37.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc4_37.4: init %tuple.type.d07 to file.%tuple_copy.var = tuple_init (%.loc4_37.2, %.loc4_37.3) [concrete = constants.%tuple.21c]
// CHECK:STDOUT:   %.loc4_1: init %tuple.type.d07 = converted %.loc4_37.1, %.loc4_37.4 [concrete = constants.%tuple.21c]
// CHECK:STDOUT:   assign file.%tuple_copy.var, %.loc4_1
// CHECK:STDOUT:   %int_3.loc6: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %int_2.loc6: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %int_1.loc6: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc6_71.1: %struct_type.c.b.a = struct_literal (%int_3.loc6, %int_2.loc6, %int_1.loc6) [concrete = constants.%struct.c75]
// CHECK:STDOUT:   %i32.loc6_81: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %i32.loc6_90: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %i32.loc6_99: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %struct_type.b.a.c: type = struct_type {.b: %i32, .a: %i32, .c: %i32} [concrete = constants.%struct_type.b.a.c]
// CHECK:STDOUT:   %impl.elem0.loc6_71.1: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc6_71.1: <bound method> = bound_method %int_2.loc6, %impl.elem0.loc6_71.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4e5]
// CHECK:STDOUT:   %specific_fn.loc6_71.1: <specific function> = specific_function %impl.elem0.loc6_71.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_71.2: <bound method> = bound_method %int_2.loc6, %specific_fn.loc6_71.1 [concrete = constants.%bound_method.646]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_71.1: init %i32 = call %bound_method.loc6_71.2(%int_2.loc6) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc6_71.2: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_71.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc6_71.3: %i32 = converted %int_2.loc6, %.loc6_71.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %impl.elem0.loc6_71.2: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc6_71.3: <bound method> = bound_method %int_1.loc6, %impl.elem0.loc6_71.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215]
// CHECK:STDOUT:   %specific_fn.loc6_71.2: <specific function> = specific_function %impl.elem0.loc6_71.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_71.4: <bound method> = bound_method %int_1.loc6, %specific_fn.loc6_71.2 [concrete = constants.%bound_method.38b]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_71.2: init %i32 = call %bound_method.loc6_71.4(%int_1.loc6) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc6_71.4: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_71.2 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc6_71.5: %i32 = converted %int_1.loc6, %.loc6_71.4 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc6_71.3: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc6_71.5: <bound method> = bound_method %int_3.loc6, %impl.elem0.loc6_71.3 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.061]
// CHECK:STDOUT:   %specific_fn.loc6_71.3: <specific function> = specific_function %impl.elem0.loc6_71.3, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_71.6: <bound method> = bound_method %int_3.loc6, %specific_fn.loc6_71.3 [concrete = constants.%bound_method.fa7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_71.3: init %i32 = call %bound_method.loc6_71.6(%int_3.loc6) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc6_71.6: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_71.3 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc6_71.7: %i32 = converted %int_3.loc6, %.loc6_71.6 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %struct.loc6: %struct_type.b.a.c = struct_value (%.loc6_71.3, %.loc6_71.5, %.loc6_71.7) [concrete = constants.%struct.21d]
// CHECK:STDOUT:   %.loc6_73.1: %struct_type.b.a.c = converted %.loc6_71.1, %struct.loc6 [concrete = constants.%struct.21d]
// CHECK:STDOUT:   %.loc6_73.2: %i32 = struct_access %.loc6_73.1, element1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc6_73.1: %.8e2 = impl_witness_access constants.%Copy.impl_witness.f17, element0 [concrete = constants.%Int.as.Copy.impl.Op.664]
// CHECK:STDOUT:   %bound_method.loc6_73.1: <bound method> = bound_method %.loc6_73.2, %impl.elem0.loc6_73.1 [concrete = constants.%Int.as.Copy.impl.Op.bound.ee3]
// CHECK:STDOUT:   %specific_fn.loc6_73.1: <specific function> = specific_function %impl.elem0.loc6_73.1, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_73.2: <bound method> = bound_method %.loc6_73.2, %specific_fn.loc6_73.1 [concrete = constants.%bound_method.4d7]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc6_73.1: init %i32 = call %bound_method.loc6_73.2(%.loc6_73.2) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc6_73.3: ref %i32 = struct_access file.%struct_copy.var, element1 [concrete = constants.%.460]
// CHECK:STDOUT:   %.loc6_73.4: init %i32 to %.loc6_73.3 = in_place_init %Int.as.Copy.impl.Op.call.loc6_73.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc6_73.5: %i32 = struct_access %.loc6_73.1, element0 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %impl.elem0.loc6_73.2: %.8e2 = impl_witness_access constants.%Copy.impl_witness.f17, element0 [concrete = constants.%Int.as.Copy.impl.Op.664]
// CHECK:STDOUT:   %bound_method.loc6_73.3: <bound method> = bound_method %.loc6_73.5, %impl.elem0.loc6_73.2 [concrete = constants.%Int.as.Copy.impl.Op.bound.5e8]
// CHECK:STDOUT:   %specific_fn.loc6_73.2: <specific function> = specific_function %impl.elem0.loc6_73.2, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_73.4: <bound method> = bound_method %.loc6_73.5, %specific_fn.loc6_73.2 [concrete = constants.%bound_method.f15]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc6_73.2: init %i32 = call %bound_method.loc6_73.4(%.loc6_73.5) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc6_73.6: ref %i32 = struct_access file.%struct_copy.var, element0 [concrete = constants.%.61f]
// CHECK:STDOUT:   %.loc6_73.7: init %i32 to %.loc6_73.6 = in_place_init %Int.as.Copy.impl.Op.call.loc6_73.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc6_73.8: %i32 = struct_access %.loc6_73.1, element2 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %impl.elem0.loc6_73.3: %.8e2 = impl_witness_access constants.%Copy.impl_witness.f17, element0 [concrete = constants.%Int.as.Copy.impl.Op.664]
// CHECK:STDOUT:   %bound_method.loc6_73.5: <bound method> = bound_method %.loc6_73.8, %impl.elem0.loc6_73.3 [concrete = constants.%Int.as.Copy.impl.Op.bound.ecc]
// CHECK:STDOUT:   %specific_fn.loc6_73.3: <specific function> = specific_function %impl.elem0.loc6_73.3, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_73.6: <bound method> = bound_method %.loc6_73.8, %specific_fn.loc6_73.3 [concrete = constants.%bound_method.4f5]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc6_73.3: init %i32 = call %bound_method.loc6_73.6(%.loc6_73.8) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc6_73.9: ref %i32 = struct_access file.%struct_copy.var, element2 [concrete = constants.%.f68]
// CHECK:STDOUT:   %.loc6_73.10: init %i32 to %.loc6_73.9 = in_place_init %Int.as.Copy.impl.Op.call.loc6_73.3 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc6_73.11: init %struct_type.a.b.c to file.%struct_copy.var = struct_init (%.loc6_73.4, %.loc6_73.7, %.loc6_73.10) [concrete = constants.%struct.cff]
// CHECK:STDOUT:   %.loc6_1: init %struct_type.a.b.c = converted %.loc6_73.1, %.loc6_73.11 [concrete = constants.%struct.cff]
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
// CHECK:STDOUT:   %impl.elem0.loc8: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc8_37.1: <bound method> = bound_method %int_0.loc8_35, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.897]
// CHECK:STDOUT:   %specific_fn.loc8: <specific function> = specific_function %impl.elem0.loc8, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_37.2: <bound method> = bound_method %int_0.loc8_35, %specific_fn.loc8 [concrete = constants.%bound_method.d2e]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8: init %i32 = call %bound_method.loc8_37.2(%int_0.loc8_35) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc8_37.2: init %i32 = converted %int_0.loc8_35, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc8_1: ref %array_type = splice_block file.%tuple_index.var [concrete = file.%tuple_index.var] {}
// CHECK:STDOUT:   %int_0.loc8_37: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc8_37.3: ref %i32 = array_index %.loc8_1, %int_0.loc8_37
// CHECK:STDOUT:   %.loc8_37.4: init %i32 to %.loc8_37.3 = in_place_init %.loc8_37.2 [concrete = constants.%int_0.6a9]
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
// CHECK:STDOUT:   %impl.elem0.loc10: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc10_39.1: <bound method> = bound_method %int_0.loc10_37, %impl.elem0.loc10 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.897]
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_39.2: <bound method> = bound_method %int_0.loc10_37, %specific_fn.loc10 [concrete = constants.%bound_method.d2e]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10: init %i32 = call %bound_method.loc10_39.2(%int_0.loc10_37) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc10_39.2: init %i32 = converted %int_0.loc10_37, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc10_1: ref %array_type = splice_block file.%struct_access.var [concrete = file.%struct_access.var] {}
// CHECK:STDOUT:   %int_0.loc10_39: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc10_39.3: ref %i32 = array_index %.loc10_1, %int_0.loc10_39
// CHECK:STDOUT:   %.loc10_39.4: init %i32 to %.loc10_39.3 = in_place_init %.loc10_39.2 [concrete = constants.%int_0.6a9]
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
// CHECK:STDOUT:   %array_type.0cb: type = array_type %int_1.5b8, %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.a98: type = pattern_type %array_type.0cb [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %tuple.type.985: type = tuple_type (Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.4f2: %tuple.type.985 = tuple_value (%int_0) [concrete]
// CHECK:STDOUT:   %int_5.64b: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %int_7.29f: Core.IntLiteral = int_value 7 [concrete]
// CHECK:STDOUT:   %int_9.988: Core.IntLiteral = int_value 9 [concrete]
// CHECK:STDOUT:   %tuple.type.d46: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.869: %tuple.type.d46 = tuple_value (%int_5.64b, %int_7.29f, %int_1.5b8, %int_9.988) [concrete]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %array_type.f32: type = array_type %int_4, %i32 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.6bc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.74f, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.6bc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.b37: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.545: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.b37, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.005: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.e9d: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_5.0f6: %i32 = int_value 5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.1e0: <bound method> = bound_method %int_7.29f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.bf2: <bound method> = bound_method %int_7.29f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_7.0b1: %i32 = int_value 7 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.38b: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.87d: <bound method> = bound_method %int_9.988, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.661: <bound method> = bound_method %int_9.988, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_9.f88: %i32 = int_value 9 [concrete]
// CHECK:STDOUT:   %array: %array_type.f32 = tuple_value (%int_5.0f6, %int_7.0b1, %int_1.5d2, %int_9.f88) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4e5: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.646: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.42d: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.74f = impl_witness_table (%Core.import_ref.42d), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %array_index.patt: %pattern_type.a98 = ref_binding_pattern array_index [concrete]
// CHECK:STDOUT:     %array_index.var_patt: %pattern_type.a98 = var_pattern %array_index.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %array_index.var: ref %array_type.0cb = var %array_index.var_patt [concrete]
// CHECK:STDOUT:   %.loc9: type = splice_block %array_type [concrete = constants.%array_type.0cb] {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %array_type: type = array_type %int_1, %i32 [concrete = constants.%array_type.0cb]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %array_index: ref %array_type.0cb = ref_binding array_index, %array_index.var [concrete = %array_index.var]
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
// CHECK:STDOUT:   %array_type: type = array_type %int_4, %i32.loc9_76 [concrete = constants.%array_type.f32]
// CHECK:STDOUT:   %impl.elem0.loc9_65.1: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc9_65.1: <bound method> = bound_method %int_5, %impl.elem0.loc9_65.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.005]
// CHECK:STDOUT:   %specific_fn.loc9_65.1: <specific function> = specific_function %impl.elem0.loc9_65.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_65.2: <bound method> = bound_method %int_5, %specific_fn.loc9_65.1 [concrete = constants.%bound_method.e9d]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_65.1: init %i32 = call %bound_method.loc9_65.2(%int_5) [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc9_65.2: init %i32 = converted %int_5, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_65.1 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc9_65.3: ref %array_type.f32 = temporary_storage
// CHECK:STDOUT:   %int_0.loc9_65: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc9_65.4: ref %i32 = array_index %.loc9_65.3, %int_0.loc9_65
// CHECK:STDOUT:   %.loc9_65.5: init %i32 to %.loc9_65.4 = in_place_init %.loc9_65.2 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %impl.elem0.loc9_65.2: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc9_65.3: <bound method> = bound_method %int_7, %impl.elem0.loc9_65.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.1e0]
// CHECK:STDOUT:   %specific_fn.loc9_65.2: <specific function> = specific_function %impl.elem0.loc9_65.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_65.4: <bound method> = bound_method %int_7, %specific_fn.loc9_65.2 [concrete = constants.%bound_method.bf2]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_65.2: init %i32 = call %bound_method.loc9_65.4(%int_7) [concrete = constants.%int_7.0b1]
// CHECK:STDOUT:   %.loc9_65.6: init %i32 = converted %int_7, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_65.2 [concrete = constants.%int_7.0b1]
// CHECK:STDOUT:   %int_1.loc9_65: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc9_65.7: ref %i32 = array_index %.loc9_65.3, %int_1.loc9_65
// CHECK:STDOUT:   %.loc9_65.8: init %i32 to %.loc9_65.7 = in_place_init %.loc9_65.6 [concrete = constants.%int_7.0b1]
// CHECK:STDOUT:   %impl.elem0.loc9_65.3: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc9_65.5: <bound method> = bound_method %int_1.loc9_61, %impl.elem0.loc9_65.3 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215]
// CHECK:STDOUT:   %specific_fn.loc9_65.3: <specific function> = specific_function %impl.elem0.loc9_65.3, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_65.6: <bound method> = bound_method %int_1.loc9_61, %specific_fn.loc9_65.3 [concrete = constants.%bound_method.38b]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_65.3: init %i32 = call %bound_method.loc9_65.6(%int_1.loc9_61) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc9_65.9: init %i32 = converted %int_1.loc9_61, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_65.3 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %int_2.loc9_65: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc9_65.10: ref %i32 = array_index %.loc9_65.3, %int_2.loc9_65
// CHECK:STDOUT:   %.loc9_65.11: init %i32 to %.loc9_65.10 = in_place_init %.loc9_65.9 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc9_65.4: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc9_65.7: <bound method> = bound_method %int_9, %impl.elem0.loc9_65.4 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.87d]
// CHECK:STDOUT:   %specific_fn.loc9_65.4: <specific function> = specific_function %impl.elem0.loc9_65.4, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_65.8: <bound method> = bound_method %int_9, %specific_fn.loc9_65.4 [concrete = constants.%bound_method.661]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_65.4: init %i32 = call %bound_method.loc9_65.8(%int_9) [concrete = constants.%int_9.f88]
// CHECK:STDOUT:   %.loc9_65.12: init %i32 = converted %int_9, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_65.4 [concrete = constants.%int_9.f88]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:   %.loc9_65.13: ref %i32 = array_index %.loc9_65.3, %int_3
// CHECK:STDOUT:   %.loc9_65.14: init %i32 to %.loc9_65.13 = in_place_init %.loc9_65.12 [concrete = constants.%int_9.f88]
// CHECK:STDOUT:   %.loc9_65.15: init %array_type.f32 to %.loc9_65.3 = array_init (%.loc9_65.5, %.loc9_65.8, %.loc9_65.11, %.loc9_65.14) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc9_67.1: init %array_type.f32 = converted %.loc9_65.1, %.loc9_65.15 [concrete = constants.%array]
// CHECK:STDOUT:   %int_2.loc9_85: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc9_67.2: ref %array_type.f32 = temporary %.loc9_65.3, %.loc9_67.1
// CHECK:STDOUT:   %impl.elem0.loc9_85: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc9_85.1: <bound method> = bound_method %int_2.loc9_85, %impl.elem0.loc9_85 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4e5]
// CHECK:STDOUT:   %specific_fn.loc9_85: <specific function> = specific_function %impl.elem0.loc9_85, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_85.2: <bound method> = bound_method %int_2.loc9_85, %specific_fn.loc9_85 [concrete = constants.%bound_method.646]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_85: init %i32 = call %bound_method.loc9_85.2(%int_2.loc9_85) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc9_85.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_85 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc9_85.2: %i32 = converted %int_2.loc9_85, %.loc9_85.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc9_86.1: ref %i32 = array_index %.loc9_67.2, %.loc9_85.2
// CHECK:STDOUT:   %.loc9_86.2: %i32 = acquire_value %.loc9_86.1
// CHECK:STDOUT:   assign file.%array_index.var, <error>
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- symbolic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %const: type = const_type %T.67d [symbolic]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.24b = tuple_value (%ptr.e8f, %const) [symbolic]
// CHECK:STDOUT:   %tuple.type.3c8: type = tuple_type (%ptr.e8f, %const) [symbolic]
// CHECK:STDOUT:   %require_complete.666: <witness> = require_complete_type %tuple.type.3c8 [symbolic]
// CHECK:STDOUT:   %pattern_type.4ac: type = pattern_type %tuple.type.3c8 [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %.f7e: require_specific_def_type = require_specific_def @T.as.DefaultOrUnformed.impl(%tuple.type.3c8) [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.lookup_impl_witness.2ad: <witness> = lookup_impl_witness %tuple.type.3c8, @DefaultOrUnformed [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.9a0: %DefaultOrUnformed.type = facet_value %tuple.type.3c8, (%DefaultOrUnformed.lookup_impl_witness.2ad) [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.WithSelf.Op.type.0af: type = fn_type @DefaultOrUnformed.WithSelf.Op, @DefaultOrUnformed.WithSelf(%DefaultOrUnformed.facet.9a0) [symbolic]
// CHECK:STDOUT:   %.246: type = fn_type_with_self_type %DefaultOrUnformed.WithSelf.Op.type.0af, %DefaultOrUnformed.facet.9a0 [symbolic]
// CHECK:STDOUT:   %impl.elem0.c6f: %.246 = impl_witness_access %DefaultOrUnformed.lookup_impl_witness.2ad, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.58b: <specific function> = specific_impl_function %impl.elem0.c6f, @DefaultOrUnformed.WithSelf.Op(%DefaultOrUnformed.facet.9a0) [symbolic]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %T.67d} [symbolic]
// CHECK:STDOUT:   %require_complete.5d6: <witness> = require_complete_type %struct_type.a [symbolic]
// CHECK:STDOUT:   %pattern_type.7b9: type = pattern_type %struct_type.a [symbolic]
// CHECK:STDOUT:   %.d26: require_specific_def_type = require_specific_def @T.as.DefaultOrUnformed.impl(%struct_type.a) [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.lookup_impl_witness.169: <witness> = lookup_impl_witness %struct_type.a, @DefaultOrUnformed [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.ca5: %DefaultOrUnformed.type = facet_value %struct_type.a, (%DefaultOrUnformed.lookup_impl_witness.169) [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.WithSelf.Op.type.0c5: type = fn_type @DefaultOrUnformed.WithSelf.Op, @DefaultOrUnformed.WithSelf(%DefaultOrUnformed.facet.ca5) [symbolic]
// CHECK:STDOUT:   %.9ac: type = fn_type_with_self_type %DefaultOrUnformed.WithSelf.Op.type.0c5, %DefaultOrUnformed.facet.ca5 [symbolic]
// CHECK:STDOUT:   %impl.elem0.e9c: %.9ac = impl_witness_access %DefaultOrUnformed.lookup_impl_witness.169, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.cf6: <specific function> = specific_impl_function %impl.elem0.e9c, @DefaultOrUnformed.WithSelf.Op(%DefaultOrUnformed.facet.ca5) [symbolic]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %array_type.742: type = array_type %int_5, %T.67d [symbolic]
// CHECK:STDOUT:   %require_complete.345: <witness> = require_complete_type %array_type.742 [symbolic]
// CHECK:STDOUT:   %pattern_type.d52: type = pattern_type %array_type.742 [symbolic]
// CHECK:STDOUT:   %.617: require_specific_def_type = require_specific_def @T.as.DefaultOrUnformed.impl(%array_type.742) [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.lookup_impl_witness.945: <witness> = lookup_impl_witness %array_type.742, @DefaultOrUnformed [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.ad6: %DefaultOrUnformed.type = facet_value %array_type.742, (%DefaultOrUnformed.lookup_impl_witness.945) [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.WithSelf.Op.type.2f0: type = fn_type @DefaultOrUnformed.WithSelf.Op, @DefaultOrUnformed.WithSelf(%DefaultOrUnformed.facet.ad6) [symbolic]
// CHECK:STDOUT:   %.544: type = fn_type_with_self_type %DefaultOrUnformed.WithSelf.Op.type.2f0, %DefaultOrUnformed.facet.ad6 [symbolic]
// CHECK:STDOUT:   %impl.elem0.97e: %.544 = impl_witness_access %DefaultOrUnformed.lookup_impl_witness.945, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.261: <specific function> = specific_impl_function %impl.elem0.97e, @DefaultOrUnformed.WithSelf.Op(%DefaultOrUnformed.facet.ad6) [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.b29: <witness> = lookup_impl_witness %array_type.742, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.ddc: %Destroy.type = facet_value %array_type.742, (%Destroy.lookup_impl_witness.b29) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.3c6: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.ddc) [symbolic]
// CHECK:STDOUT:   %.950: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.3c6, %Destroy.facet.ddc [symbolic]
// CHECK:STDOUT:   %impl.elem0.fa5: %.950 = impl_witness_access %Destroy.lookup_impl_witness.b29, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.ede: <specific function> = specific_impl_function %impl.elem0.fa5, @Destroy.WithSelf.Op(%Destroy.facet.ddc) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.bed: <witness> = lookup_impl_witness %struct_type.a, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.d21: %Destroy.type = facet_value %struct_type.a, (%Destroy.lookup_impl_witness.bed) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.873: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.d21) [symbolic]
// CHECK:STDOUT:   %.511: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.873, %Destroy.facet.d21 [symbolic]
// CHECK:STDOUT:   %impl.elem0.12c: %.511 = impl_witness_access %Destroy.lookup_impl_witness.bed, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.c3a: <specific function> = specific_impl_function %impl.elem0.12c, @Destroy.WithSelf.Op(%Destroy.facet.d21) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.7eb: <witness> = lookup_impl_witness %tuple.type.3c8, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.66d: %Destroy.type = facet_value %tuple.type.3c8, (%Destroy.lookup_impl_witness.7eb) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.f23: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.66d) [symbolic]
// CHECK:STDOUT:   %.f20: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.f23, %Destroy.facet.66d [symbolic]
// CHECK:STDOUT:   %impl.elem0.f5f: %.f20 = impl_witness_access %Destroy.lookup_impl_witness.7eb, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.87c: <specific function> = specific_impl_function %impl.elem0.f5f, @Destroy.WithSelf.Op(%Destroy.facet.66d) [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %N.5de: %i32 = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.139: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %From: Core.IntLiteral = symbolic_binding From, 0 [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.2ed: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%From) [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.d29: %Int.as.ImplicitAs.impl.Convert.type.2ed = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.640: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.ea2, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.240: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.dd4: %Int.as.ImplicitAs.impl.Convert.type.240 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.139 = facet_value %i32, (%ImplicitAs.impl_witness.640) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.462: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.0a7: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.462, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %N.5de, %Int.as.ImplicitAs.impl.Convert.dd4 [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Int.as.ImplicitAs.impl.Convert.dd4, @Int.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %N.5de, %Int.as.ImplicitAs.impl.Convert.specific_fn [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method(%N.5de) [symbolic]
// CHECK:STDOUT:   %array_type.2ec: type = array_type %Int.as.ImplicitAs.impl.Convert.call, %i32 [symbolic]
// CHECK:STDOUT:   %require_complete.5d1: <witness> = require_complete_type %array_type.2ec [symbolic]
// CHECK:STDOUT:   %pattern_type.99c: type = pattern_type %array_type.2ec [symbolic]
// CHECK:STDOUT:   %.d6f: require_specific_def_type = require_specific_def @T.as.DefaultOrUnformed.impl(%array_type.2ec) [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.lookup_impl_witness.162: <witness> = lookup_impl_witness %array_type.2ec, @DefaultOrUnformed [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.0c7: %DefaultOrUnformed.type = facet_value %array_type.2ec, (%DefaultOrUnformed.lookup_impl_witness.162) [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.WithSelf.Op.type.839: type = fn_type @DefaultOrUnformed.WithSelf.Op, @DefaultOrUnformed.WithSelf(%DefaultOrUnformed.facet.0c7) [symbolic]
// CHECK:STDOUT:   %.78e: type = fn_type_with_self_type %DefaultOrUnformed.WithSelf.Op.type.839, %DefaultOrUnformed.facet.0c7 [symbolic]
// CHECK:STDOUT:   %impl.elem0.82a: %.78e = impl_witness_access %DefaultOrUnformed.lookup_impl_witness.162, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.fcf: <specific function> = specific_impl_function %impl.elem0.82a, @DefaultOrUnformed.WithSelf.Op(%DefaultOrUnformed.facet.0c7) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.f5c: <witness> = lookup_impl_witness %array_type.2ec, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.d1e: %Destroy.type = facet_value %array_type.2ec, (%Destroy.lookup_impl_witness.f5c) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ff9: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.d1e) [symbolic]
// CHECK:STDOUT:   %.63e: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.ff9, %Destroy.facet.d1e [symbolic]
// CHECK:STDOUT:   %impl.elem0.2bf: %.63e = impl_witness_access %Destroy.lookup_impl_witness.f5c, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.0a5: <specific function> = specific_impl_function %impl.elem0.2bf, @Destroy.WithSelf.Op(%Destroy.facet.d1e) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.0bc: @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert.type (%Int.as.ImplicitAs.impl.Convert.type.2ed) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert (constants.%Int.as.ImplicitAs.impl.Convert.d29)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.ea2 = impl_witness_table (%Core.import_ref.0bc), @Int.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc4_6.2: type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ptr.loc6_19.2: type = ptr_type %T.loc4_6.1 [symbolic = %ptr.loc6_19.2 (constants.%ptr.e8f)]
// CHECK:STDOUT:   %const.loc6_22.2: type = const_type %T.loc4_6.1 [symbolic = %const.loc6_22.2 (constants.%const)]
// CHECK:STDOUT:   %tuple: %tuple.type.24b = tuple_value (%ptr.loc6_19.2, %const.loc6_22.2) [symbolic = %tuple (constants.%tuple)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%ptr.loc6_19.2, %const.loc6_22.2) [symbolic = %tuple.type (constants.%tuple.type.3c8)]
// CHECK:STDOUT:   %require_complete.loc6: <witness> = require_complete_type %tuple.type [symbolic = %require_complete.loc6 (constants.%require_complete.666)]
// CHECK:STDOUT:   %pattern_type.loc6: type = pattern_type %tuple.type [symbolic = %pattern_type.loc6 (constants.%pattern_type.4ac)]
// CHECK:STDOUT:   %.loc6_30.3: require_specific_def_type = require_specific_def @T.as.DefaultOrUnformed.impl(%tuple.type) [symbolic = %.loc6_30.3 (constants.%.f7e)]
// CHECK:STDOUT:   %DefaultOrUnformed.lookup_impl_witness.loc6: <witness> = lookup_impl_witness %tuple.type, @DefaultOrUnformed [symbolic = %DefaultOrUnformed.lookup_impl_witness.loc6 (constants.%DefaultOrUnformed.lookup_impl_witness.2ad)]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc6_30.2: %DefaultOrUnformed.type = facet_value %tuple.type, (%DefaultOrUnformed.lookup_impl_witness.loc6) [symbolic = %DefaultOrUnformed.facet.loc6_30.2 (constants.%DefaultOrUnformed.facet.9a0)]
// CHECK:STDOUT:   %DefaultOrUnformed.WithSelf.Op.type.loc6: type = fn_type @DefaultOrUnformed.WithSelf.Op, @DefaultOrUnformed.WithSelf(%DefaultOrUnformed.facet.loc6_30.2) [symbolic = %DefaultOrUnformed.WithSelf.Op.type.loc6 (constants.%DefaultOrUnformed.WithSelf.Op.type.0af)]
// CHECK:STDOUT:   %.loc6_30.4: type = fn_type_with_self_type %DefaultOrUnformed.WithSelf.Op.type.loc6, %DefaultOrUnformed.facet.loc6_30.2 [symbolic = %.loc6_30.4 (constants.%.246)]
// CHECK:STDOUT:   %impl.elem0.loc6_30.2: @F.%.loc6_30.4 (%.246) = impl_witness_access %DefaultOrUnformed.lookup_impl_witness.loc6, element0 [symbolic = %impl.elem0.loc6_30.2 (constants.%impl.elem0.c6f)]
// CHECK:STDOUT:   %specific_impl_fn.loc6_30.2: <specific function> = specific_impl_function %impl.elem0.loc6_30.2, @DefaultOrUnformed.WithSelf.Op(%DefaultOrUnformed.facet.loc6_30.2) [symbolic = %specific_impl_fn.loc6_30.2 (constants.%specific_impl_fn.58b)]
// CHECK:STDOUT:   %struct_type.a.loc7_23.2: type = struct_type {.a: @F.%T.loc4_6.1 (%T.67d)} [symbolic = %struct_type.a.loc7_23.2 (constants.%struct_type.a)]
// CHECK:STDOUT:   %require_complete.loc7: <witness> = require_complete_type %struct_type.a.loc7_23.2 [symbolic = %require_complete.loc7 (constants.%require_complete.5d6)]
// CHECK:STDOUT:   %pattern_type.loc7: type = pattern_type %struct_type.a.loc7_23.2 [symbolic = %pattern_type.loc7 (constants.%pattern_type.7b9)]
// CHECK:STDOUT:   %.loc7_24.3: require_specific_def_type = require_specific_def @T.as.DefaultOrUnformed.impl(%struct_type.a.loc7_23.2) [symbolic = %.loc7_24.3 (constants.%.d26)]
// CHECK:STDOUT:   %DefaultOrUnformed.lookup_impl_witness.loc7: <witness> = lookup_impl_witness %struct_type.a.loc7_23.2, @DefaultOrUnformed [symbolic = %DefaultOrUnformed.lookup_impl_witness.loc7 (constants.%DefaultOrUnformed.lookup_impl_witness.169)]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc7_24.2: %DefaultOrUnformed.type = facet_value %struct_type.a.loc7_23.2, (%DefaultOrUnformed.lookup_impl_witness.loc7) [symbolic = %DefaultOrUnformed.facet.loc7_24.2 (constants.%DefaultOrUnformed.facet.ca5)]
// CHECK:STDOUT:   %DefaultOrUnformed.WithSelf.Op.type.loc7: type = fn_type @DefaultOrUnformed.WithSelf.Op, @DefaultOrUnformed.WithSelf(%DefaultOrUnformed.facet.loc7_24.2) [symbolic = %DefaultOrUnformed.WithSelf.Op.type.loc7 (constants.%DefaultOrUnformed.WithSelf.Op.type.0c5)]
// CHECK:STDOUT:   %.loc7_24.4: type = fn_type_with_self_type %DefaultOrUnformed.WithSelf.Op.type.loc7, %DefaultOrUnformed.facet.loc7_24.2 [symbolic = %.loc7_24.4 (constants.%.9ac)]
// CHECK:STDOUT:   %impl.elem0.loc7_24.2: @F.%.loc7_24.4 (%.9ac) = impl_witness_access %DefaultOrUnformed.lookup_impl_witness.loc7, element0 [symbolic = %impl.elem0.loc7_24.2 (constants.%impl.elem0.e9c)]
// CHECK:STDOUT:   %specific_impl_fn.loc7_24.2: <specific function> = specific_impl_function %impl.elem0.loc7_24.2, @DefaultOrUnformed.WithSelf.Op(%DefaultOrUnformed.facet.loc7_24.2) [symbolic = %specific_impl_fn.loc7_24.2 (constants.%specific_impl_fn.cf6)]
// CHECK:STDOUT:   %array_type.loc8_27.2: type = array_type constants.%int_5, %T.loc4_6.1 [symbolic = %array_type.loc8_27.2 (constants.%array_type.742)]
// CHECK:STDOUT:   %require_complete.loc8: <witness> = require_complete_type %array_type.loc8_27.2 [symbolic = %require_complete.loc8 (constants.%require_complete.345)]
// CHECK:STDOUT:   %pattern_type.loc8: type = pattern_type %array_type.loc8_27.2 [symbolic = %pattern_type.loc8 (constants.%pattern_type.d52)]
// CHECK:STDOUT:   %.loc8_28.3: require_specific_def_type = require_specific_def @T.as.DefaultOrUnformed.impl(%array_type.loc8_27.2) [symbolic = %.loc8_28.3 (constants.%.617)]
// CHECK:STDOUT:   %DefaultOrUnformed.lookup_impl_witness.loc8: <witness> = lookup_impl_witness %array_type.loc8_27.2, @DefaultOrUnformed [symbolic = %DefaultOrUnformed.lookup_impl_witness.loc8 (constants.%DefaultOrUnformed.lookup_impl_witness.945)]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc8_28.2: %DefaultOrUnformed.type = facet_value %array_type.loc8_27.2, (%DefaultOrUnformed.lookup_impl_witness.loc8) [symbolic = %DefaultOrUnformed.facet.loc8_28.2 (constants.%DefaultOrUnformed.facet.ad6)]
// CHECK:STDOUT:   %DefaultOrUnformed.WithSelf.Op.type.loc8: type = fn_type @DefaultOrUnformed.WithSelf.Op, @DefaultOrUnformed.WithSelf(%DefaultOrUnformed.facet.loc8_28.2) [symbolic = %DefaultOrUnformed.WithSelf.Op.type.loc8 (constants.%DefaultOrUnformed.WithSelf.Op.type.2f0)]
// CHECK:STDOUT:   %.loc8_28.4: type = fn_type_with_self_type %DefaultOrUnformed.WithSelf.Op.type.loc8, %DefaultOrUnformed.facet.loc8_28.2 [symbolic = %.loc8_28.4 (constants.%.544)]
// CHECK:STDOUT:   %impl.elem0.loc8_28.2: @F.%.loc8_28.4 (%.544) = impl_witness_access %DefaultOrUnformed.lookup_impl_witness.loc8, element0 [symbolic = %impl.elem0.loc8_28.2 (constants.%impl.elem0.97e)]
// CHECK:STDOUT:   %specific_impl_fn.loc8_28.2: <specific function> = specific_impl_function %impl.elem0.loc8_28.2, @DefaultOrUnformed.WithSelf.Op(%DefaultOrUnformed.facet.loc8_28.2) [symbolic = %specific_impl_fn.loc8_28.2 (constants.%specific_impl_fn.261)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc8: <witness> = lookup_impl_witness %array_type.loc8_27.2, @Destroy [symbolic = %Destroy.lookup_impl_witness.loc8 (constants.%Destroy.lookup_impl_witness.b29)]
// CHECK:STDOUT:   %Destroy.facet.loc8: %Destroy.type = facet_value %array_type.loc8_27.2, (%Destroy.lookup_impl_witness.loc8) [symbolic = %Destroy.facet.loc8 (constants.%Destroy.facet.ddc)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc8: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc8) [symbolic = %Destroy.WithSelf.Op.type.loc8 (constants.%Destroy.WithSelf.Op.type.3c6)]
// CHECK:STDOUT:   %.loc8_3.2: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.loc8, %Destroy.facet.loc8 [symbolic = %.loc8_3.2 (constants.%.950)]
// CHECK:STDOUT:   %impl.elem0.loc8_3.2: @F.%.loc8_3.2 (%.950) = impl_witness_access %Destroy.lookup_impl_witness.loc8, element0 [symbolic = %impl.elem0.loc8_3.2 (constants.%impl.elem0.fa5)]
// CHECK:STDOUT:   %specific_impl_fn.loc8_3.2: <specific function> = specific_impl_function %impl.elem0.loc8_3.2, @Destroy.WithSelf.Op(%Destroy.facet.loc8) [symbolic = %specific_impl_fn.loc8_3.2 (constants.%specific_impl_fn.ede)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc7: <witness> = lookup_impl_witness %struct_type.a.loc7_23.2, @Destroy [symbolic = %Destroy.lookup_impl_witness.loc7 (constants.%Destroy.lookup_impl_witness.bed)]
// CHECK:STDOUT:   %Destroy.facet.loc7: %Destroy.type = facet_value %struct_type.a.loc7_23.2, (%Destroy.lookup_impl_witness.loc7) [symbolic = %Destroy.facet.loc7 (constants.%Destroy.facet.d21)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc7: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc7) [symbolic = %Destroy.WithSelf.Op.type.loc7 (constants.%Destroy.WithSelf.Op.type.873)]
// CHECK:STDOUT:   %.loc7_3: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.loc7, %Destroy.facet.loc7 [symbolic = %.loc7_3 (constants.%.511)]
// CHECK:STDOUT:   %impl.elem0.loc7_3.2: @F.%.loc7_3 (%.511) = impl_witness_access %Destroy.lookup_impl_witness.loc7, element0 [symbolic = %impl.elem0.loc7_3.2 (constants.%impl.elem0.12c)]
// CHECK:STDOUT:   %specific_impl_fn.loc7_3.2: <specific function> = specific_impl_function %impl.elem0.loc7_3.2, @Destroy.WithSelf.Op(%Destroy.facet.loc7) [symbolic = %specific_impl_fn.loc7_3.2 (constants.%specific_impl_fn.c3a)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc6: <witness> = lookup_impl_witness %tuple.type, @Destroy [symbolic = %Destroy.lookup_impl_witness.loc6 (constants.%Destroy.lookup_impl_witness.7eb)]
// CHECK:STDOUT:   %Destroy.facet.loc6: %Destroy.type = facet_value %tuple.type, (%Destroy.lookup_impl_witness.loc6) [symbolic = %Destroy.facet.loc6 (constants.%Destroy.facet.66d)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc6: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc6) [symbolic = %Destroy.WithSelf.Op.type.loc6 (constants.%Destroy.WithSelf.Op.type.f23)]
// CHECK:STDOUT:   %.loc6_3.2: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.loc6, %Destroy.facet.loc6 [symbolic = %.loc6_3.2 (constants.%.f20)]
// CHECK:STDOUT:   %impl.elem0.loc6_3.2: @F.%.loc6_3.2 (%.f20) = impl_witness_access %Destroy.lookup_impl_witness.loc6, element0 [symbolic = %impl.elem0.loc6_3.2 (constants.%impl.elem0.f5f)]
// CHECK:STDOUT:   %specific_impl_fn.loc6_3.2: <specific function> = specific_impl_function %impl.elem0.loc6_3.2, @Destroy.WithSelf.Op(%Destroy.facet.loc6) [symbolic = %specific_impl_fn.loc6_3.2 (constants.%specific_impl_fn.87c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %u.patt: @F.%pattern_type.loc6 (%pattern_type.4ac) = ref_binding_pattern u [concrete]
// CHECK:STDOUT:       %u.var_patt: @F.%pattern_type.loc6 (%pattern_type.4ac) = var_pattern %u.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u.var: ref @F.%tuple.type (%tuple.type.3c8) = var %u.var_patt
// CHECK:STDOUT:     %DefaultOrUnformed.facet.loc6_30.1: %DefaultOrUnformed.type = facet_value constants.%tuple.type.3c8, (constants.%DefaultOrUnformed.lookup_impl_witness.2ad) [symbolic = %DefaultOrUnformed.facet.loc6_30.2 (constants.%DefaultOrUnformed.facet.9a0)]
// CHECK:STDOUT:     %.loc6_30.1: %DefaultOrUnformed.type = converted constants.%tuple.type.3c8, %DefaultOrUnformed.facet.loc6_30.1 [symbolic = %DefaultOrUnformed.facet.loc6_30.2 (constants.%DefaultOrUnformed.facet.9a0)]
// CHECK:STDOUT:     %as_type.loc6: type = facet_access_type %.loc6_30.1 [symbolic = %tuple.type (constants.%tuple.type.3c8)]
// CHECK:STDOUT:     %.loc6_30.2: type = converted %.loc6_30.1, %as_type.loc6 [symbolic = %tuple.type (constants.%tuple.type.3c8)]
// CHECK:STDOUT:     %impl.elem0.loc6_30.1: @F.%.loc6_30.4 (%.246) = impl_witness_access constants.%DefaultOrUnformed.lookup_impl_witness.2ad, element0 [symbolic = %impl.elem0.loc6_30.2 (constants.%impl.elem0.c6f)]
// CHECK:STDOUT:     %specific_impl_fn.loc6_30.1: <specific function> = specific_impl_function %impl.elem0.loc6_30.1, @DefaultOrUnformed.WithSelf.Op(constants.%DefaultOrUnformed.facet.9a0) [symbolic = %specific_impl_fn.loc6_30.2 (constants.%specific_impl_fn.58b)]
// CHECK:STDOUT:     %.loc6_3.1: ref @F.%tuple.type (%tuple.type.3c8) = splice_block %u.var {}
// CHECK:STDOUT:     %DefaultOrUnformed.WithSelf.Op.call.loc6: init @F.%tuple.type (%tuple.type.3c8) to %.loc6_3.1 = call %specific_impl_fn.loc6_30.1()
// CHECK:STDOUT:     assign %u.var, %DefaultOrUnformed.WithSelf.Op.call.loc6
// CHECK:STDOUT:     %.loc6_29.1: type = splice_block %.loc6_29.3 [symbolic = %tuple.type (constants.%tuple.type.3c8)] {
// CHECK:STDOUT:       %T.ref.loc6_18: type = name_ref T, %T.loc4_6.2 [symbolic = %T.loc4_6.1 (constants.%T.67d)]
// CHECK:STDOUT:       %ptr.loc6_19.1: type = ptr_type %T.ref.loc6_18 [symbolic = %ptr.loc6_19.2 (constants.%ptr.e8f)]
// CHECK:STDOUT:       %T.ref.loc6_28: type = name_ref T, %T.loc4_6.2 [symbolic = %T.loc4_6.1 (constants.%T.67d)]
// CHECK:STDOUT:       %const.loc6_22.1: type = const_type %T.ref.loc6_28 [symbolic = %const.loc6_22.2 (constants.%const)]
// CHECK:STDOUT:       %.loc6_29.2: %tuple.type.24b = tuple_literal (%ptr.loc6_19.1, %const.loc6_22.1) [symbolic = %tuple (constants.%tuple)]
// CHECK:STDOUT:       %.loc6_29.3: type = converted %.loc6_29.2, constants.%tuple.type.3c8 [symbolic = %tuple.type (constants.%tuple.type.3c8)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: ref @F.%tuple.type (%tuple.type.3c8) = ref_binding u, %u.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %v.patt: @F.%pattern_type.loc7 (%pattern_type.7b9) = ref_binding_pattern v [concrete]
// CHECK:STDOUT:       %v.var_patt: @F.%pattern_type.loc7 (%pattern_type.7b9) = var_pattern %v.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v.var: ref @F.%struct_type.a.loc7_23.2 (%struct_type.a) = var %v.var_patt
// CHECK:STDOUT:     %DefaultOrUnformed.facet.loc7_24.1: %DefaultOrUnformed.type = facet_value constants.%struct_type.a, (constants.%DefaultOrUnformed.lookup_impl_witness.169) [symbolic = %DefaultOrUnformed.facet.loc7_24.2 (constants.%DefaultOrUnformed.facet.ca5)]
// CHECK:STDOUT:     %.loc7_24.1: %DefaultOrUnformed.type = converted constants.%struct_type.a, %DefaultOrUnformed.facet.loc7_24.1 [symbolic = %DefaultOrUnformed.facet.loc7_24.2 (constants.%DefaultOrUnformed.facet.ca5)]
// CHECK:STDOUT:     %as_type.loc7: type = facet_access_type %.loc7_24.1 [symbolic = %struct_type.a.loc7_23.2 (constants.%struct_type.a)]
// CHECK:STDOUT:     %.loc7_24.2: type = converted %.loc7_24.1, %as_type.loc7 [symbolic = %struct_type.a.loc7_23.2 (constants.%struct_type.a)]
// CHECK:STDOUT:     %impl.elem0.loc7_24.1: @F.%.loc7_24.4 (%.9ac) = impl_witness_access constants.%DefaultOrUnformed.lookup_impl_witness.169, element0 [symbolic = %impl.elem0.loc7_24.2 (constants.%impl.elem0.e9c)]
// CHECK:STDOUT:     %specific_impl_fn.loc7_24.1: <specific function> = specific_impl_function %impl.elem0.loc7_24.1, @DefaultOrUnformed.WithSelf.Op(constants.%DefaultOrUnformed.facet.ca5) [symbolic = %specific_impl_fn.loc7_24.2 (constants.%specific_impl_fn.cf6)]
// CHECK:STDOUT:     %DefaultOrUnformed.WithSelf.Op.call.loc7: init @F.%struct_type.a.loc7_23.2 (%struct_type.a) = call %specific_impl_fn.loc7_24.1()
// CHECK:STDOUT:     assign %v.var, %DefaultOrUnformed.WithSelf.Op.call.loc7
// CHECK:STDOUT:     %.loc7_23: type = splice_block %struct_type.a.loc7_23.1 [symbolic = %struct_type.a.loc7_23.2 (constants.%struct_type.a)] {
// CHECK:STDOUT:       %T.ref.loc7: type = name_ref T, %T.loc4_6.2 [symbolic = %T.loc4_6.1 (constants.%T.67d)]
// CHECK:STDOUT:       %struct_type.a.loc7_23.1: type = struct_type {.a: @F.%T.loc4_6.1 (%T.67d)} [symbolic = %struct_type.a.loc7_23.2 (constants.%struct_type.a)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: ref @F.%struct_type.a.loc7_23.2 (%struct_type.a) = ref_binding v, %v.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %w.patt: @F.%pattern_type.loc8 (%pattern_type.d52) = ref_binding_pattern w [concrete]
// CHECK:STDOUT:       %w.var_patt: @F.%pattern_type.loc8 (%pattern_type.d52) = var_pattern %w.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %w.var: ref @F.%array_type.loc8_27.2 (%array_type.742) = var %w.var_patt
// CHECK:STDOUT:     %DefaultOrUnformed.facet.loc8_28.1: %DefaultOrUnformed.type = facet_value constants.%array_type.742, (constants.%DefaultOrUnformed.lookup_impl_witness.945) [symbolic = %DefaultOrUnformed.facet.loc8_28.2 (constants.%DefaultOrUnformed.facet.ad6)]
// CHECK:STDOUT:     %.loc8_28.1: %DefaultOrUnformed.type = converted constants.%array_type.742, %DefaultOrUnformed.facet.loc8_28.1 [symbolic = %DefaultOrUnformed.facet.loc8_28.2 (constants.%DefaultOrUnformed.facet.ad6)]
// CHECK:STDOUT:     %as_type.loc8: type = facet_access_type %.loc8_28.1 [symbolic = %array_type.loc8_27.2 (constants.%array_type.742)]
// CHECK:STDOUT:     %.loc8_28.2: type = converted %.loc8_28.1, %as_type.loc8 [symbolic = %array_type.loc8_27.2 (constants.%array_type.742)]
// CHECK:STDOUT:     %impl.elem0.loc8_28.1: @F.%.loc8_28.4 (%.544) = impl_witness_access constants.%DefaultOrUnformed.lookup_impl_witness.945, element0 [symbolic = %impl.elem0.loc8_28.2 (constants.%impl.elem0.97e)]
// CHECK:STDOUT:     %specific_impl_fn.loc8_28.1: <specific function> = specific_impl_function %impl.elem0.loc8_28.1, @DefaultOrUnformed.WithSelf.Op(constants.%DefaultOrUnformed.facet.ad6) [symbolic = %specific_impl_fn.loc8_28.2 (constants.%specific_impl_fn.261)]
// CHECK:STDOUT:     %.loc8_3.1: ref @F.%array_type.loc8_27.2 (%array_type.742) = splice_block %w.var {}
// CHECK:STDOUT:     %DefaultOrUnformed.WithSelf.Op.call.loc8: init @F.%array_type.loc8_27.2 (%array_type.742) to %.loc8_3.1 = call %specific_impl_fn.loc8_28.1()
// CHECK:STDOUT:     assign %w.var, %DefaultOrUnformed.WithSelf.Op.call.loc8
// CHECK:STDOUT:     %.loc8_27: type = splice_block %array_type.loc8_27.1 [symbolic = %array_type.loc8_27.2 (constants.%array_type.742)] {
// CHECK:STDOUT:       %T.ref.loc8: type = name_ref T, %T.loc4_6.2 [symbolic = %T.loc4_6.1 (constants.%T.67d)]
// CHECK:STDOUT:       %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5]
// CHECK:STDOUT:       %array_type.loc8_27.1: type = array_type %int_5, %T.ref.loc8 [symbolic = %array_type.loc8_27.2 (constants.%array_type.742)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %w: ref @F.%array_type.loc8_27.2 (%array_type.742) = ref_binding w, %w.var
// CHECK:STDOUT:     %impl.elem0.loc8_3.1: @F.%.loc8_3.2 (%.950) = impl_witness_access constants.%Destroy.lookup_impl_witness.b29, element0 [symbolic = %impl.elem0.loc8_3.2 (constants.%impl.elem0.fa5)]
// CHECK:STDOUT:     %bound_method.loc8_3.1: <bound method> = bound_method %w.var, %impl.elem0.loc8_3.1
// CHECK:STDOUT:     %specific_impl_fn.loc8_3.1: <specific function> = specific_impl_function %impl.elem0.loc8_3.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.ddc) [symbolic = %specific_impl_fn.loc8_3.2 (constants.%specific_impl_fn.ede)]
// CHECK:STDOUT:     %bound_method.loc8_3.2: <bound method> = bound_method %w.var, %specific_impl_fn.loc8_3.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call.loc8: init %empty_tuple.type = call %bound_method.loc8_3.2(%w.var)
// CHECK:STDOUT:     %impl.elem0.loc7_3.1: @F.%.loc7_3 (%.511) = impl_witness_access constants.%Destroy.lookup_impl_witness.bed, element0 [symbolic = %impl.elem0.loc7_3.2 (constants.%impl.elem0.12c)]
// CHECK:STDOUT:     %bound_method.loc7_3.1: <bound method> = bound_method %v.var, %impl.elem0.loc7_3.1
// CHECK:STDOUT:     %specific_impl_fn.loc7_3.1: <specific function> = specific_impl_function %impl.elem0.loc7_3.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.d21) [symbolic = %specific_impl_fn.loc7_3.2 (constants.%specific_impl_fn.c3a)]
// CHECK:STDOUT:     %bound_method.loc7_3.2: <bound method> = bound_method %v.var, %specific_impl_fn.loc7_3.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call.loc7: init %empty_tuple.type = call %bound_method.loc7_3.2(%v.var)
// CHECK:STDOUT:     %impl.elem0.loc6_3.1: @F.%.loc6_3.2 (%.f20) = impl_witness_access constants.%Destroy.lookup_impl_witness.7eb, element0 [symbolic = %impl.elem0.loc6_3.2 (constants.%impl.elem0.f5f)]
// CHECK:STDOUT:     %bound_method.loc6_3.1: <bound method> = bound_method %u.var, %impl.elem0.loc6_3.1
// CHECK:STDOUT:     %specific_impl_fn.loc6_3.1: <specific function> = specific_impl_function %impl.elem0.loc6_3.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.66d) [symbolic = %specific_impl_fn.loc6_3.2 (constants.%specific_impl_fn.87c)]
// CHECK:STDOUT:     %bound_method.loc6_3.2: <bound method> = bound_method %u.var, %specific_impl_fn.loc6_3.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call.loc6: init %empty_tuple.type = call %bound_method.loc6_3.2(%u.var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%N.loc12_6.2: %i32) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %N.loc12_6.1, constants.%Int.as.ImplicitAs.impl.Convert.dd4 [symbolic = %Int.as.ImplicitAs.impl.Convert.bound (constants.%Int.as.ImplicitAs.impl.Convert.bound)]
// CHECK:STDOUT:   %bound_method.loc14_28.3: <bound method> = bound_method %N.loc12_6.1, constants.%Int.as.ImplicitAs.impl.Convert.specific_fn [symbolic = %bound_method.loc14_28.3 (constants.%bound_method)]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.call.loc14_28.2: init Core.IntLiteral = call %bound_method.loc14_28.3(%N.loc12_6.1) [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc14_28.2 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:   %array_type.loc14_29.2: type = array_type %Int.as.ImplicitAs.impl.Convert.call.loc14_28.2, constants.%i32 [symbolic = %array_type.loc14_29.2 (constants.%array_type.2ec)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %array_type.loc14_29.2 [symbolic = %require_complete (constants.%require_complete.5d1)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %array_type.loc14_29.2 [symbolic = %pattern_type (constants.%pattern_type.99c)]
// CHECK:STDOUT:   %.loc14_30.3: require_specific_def_type = require_specific_def @T.as.DefaultOrUnformed.impl(%array_type.loc14_29.2) [symbolic = %.loc14_30.3 (constants.%.d6f)]
// CHECK:STDOUT:   %DefaultOrUnformed.lookup_impl_witness: <witness> = lookup_impl_witness %array_type.loc14_29.2, @DefaultOrUnformed [symbolic = %DefaultOrUnformed.lookup_impl_witness (constants.%DefaultOrUnformed.lookup_impl_witness.162)]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc14_30.2: %DefaultOrUnformed.type = facet_value %array_type.loc14_29.2, (%DefaultOrUnformed.lookup_impl_witness) [symbolic = %DefaultOrUnformed.facet.loc14_30.2 (constants.%DefaultOrUnformed.facet.0c7)]
// CHECK:STDOUT:   %DefaultOrUnformed.WithSelf.Op.type: type = fn_type @DefaultOrUnformed.WithSelf.Op, @DefaultOrUnformed.WithSelf(%DefaultOrUnformed.facet.loc14_30.2) [symbolic = %DefaultOrUnformed.WithSelf.Op.type (constants.%DefaultOrUnformed.WithSelf.Op.type.839)]
// CHECK:STDOUT:   %.loc14_30.4: type = fn_type_with_self_type %DefaultOrUnformed.WithSelf.Op.type, %DefaultOrUnformed.facet.loc14_30.2 [symbolic = %.loc14_30.4 (constants.%.78e)]
// CHECK:STDOUT:   %impl.elem0.loc14_30.2: @G.%.loc14_30.4 (%.78e) = impl_witness_access %DefaultOrUnformed.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc14_30.2 (constants.%impl.elem0.82a)]
// CHECK:STDOUT:   %specific_impl_fn.loc14_30.2: <specific function> = specific_impl_function %impl.elem0.loc14_30.2, @DefaultOrUnformed.WithSelf.Op(%DefaultOrUnformed.facet.loc14_30.2) [symbolic = %specific_impl_fn.loc14_30.2 (constants.%specific_impl_fn.fcf)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %array_type.loc14_29.2, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.f5c)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %array_type.loc14_29.2, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet (constants.%Destroy.facet.d1e)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.ff9)]
// CHECK:STDOUT:   %.loc14_3.2: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Destroy.facet [symbolic = %.loc14_3.2 (constants.%.63e)]
// CHECK:STDOUT:   %impl.elem0.loc14_3.2: @G.%.loc14_3.2 (%.63e) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc14_3.2 (constants.%impl.elem0.2bf)]
// CHECK:STDOUT:   %specific_impl_fn.loc14_3.2: <specific function> = specific_impl_function %impl.elem0.loc14_3.2, @Destroy.WithSelf.Op(%Destroy.facet) [symbolic = %specific_impl_fn.loc14_3.2 (constants.%specific_impl_fn.0a5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %k.patt: @G.%pattern_type (%pattern_type.99c) = ref_binding_pattern k [concrete]
// CHECK:STDOUT:       %k.var_patt: @G.%pattern_type (%pattern_type.99c) = var_pattern %k.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %k.var: ref @G.%array_type.loc14_29.2 (%array_type.2ec) = var %k.var_patt
// CHECK:STDOUT:     %DefaultOrUnformed.facet.loc14_30.1: %DefaultOrUnformed.type = facet_value constants.%array_type.2ec, (constants.%DefaultOrUnformed.lookup_impl_witness.162) [symbolic = %DefaultOrUnformed.facet.loc14_30.2 (constants.%DefaultOrUnformed.facet.0c7)]
// CHECK:STDOUT:     %.loc14_30.1: %DefaultOrUnformed.type = converted constants.%array_type.2ec, %DefaultOrUnformed.facet.loc14_30.1 [symbolic = %DefaultOrUnformed.facet.loc14_30.2 (constants.%DefaultOrUnformed.facet.0c7)]
// CHECK:STDOUT:     %as_type: type = facet_access_type %.loc14_30.1 [symbolic = %array_type.loc14_29.2 (constants.%array_type.2ec)]
// CHECK:STDOUT:     %.loc14_30.2: type = converted %.loc14_30.1, %as_type [symbolic = %array_type.loc14_29.2 (constants.%array_type.2ec)]
// CHECK:STDOUT:     %impl.elem0.loc14_30.1: @G.%.loc14_30.4 (%.78e) = impl_witness_access constants.%DefaultOrUnformed.lookup_impl_witness.162, element0 [symbolic = %impl.elem0.loc14_30.2 (constants.%impl.elem0.82a)]
// CHECK:STDOUT:     %specific_impl_fn.loc14_30.1: <specific function> = specific_impl_function %impl.elem0.loc14_30.1, @DefaultOrUnformed.WithSelf.Op(constants.%DefaultOrUnformed.facet.0c7) [symbolic = %specific_impl_fn.loc14_30.2 (constants.%specific_impl_fn.fcf)]
// CHECK:STDOUT:     %.loc14_3.1: ref @G.%array_type.loc14_29.2 (%array_type.2ec) = splice_block %k.var {}
// CHECK:STDOUT:     %DefaultOrUnformed.WithSelf.Op.call: init @G.%array_type.loc14_29.2 (%array_type.2ec) to %.loc14_3.1 = call %specific_impl_fn.loc14_30.1()
// CHECK:STDOUT:     assign %k.var, %DefaultOrUnformed.WithSelf.Op.call
// CHECK:STDOUT:     %.loc14_29: type = splice_block %array_type.loc14_29.1 [symbolic = %array_type.loc14_29.2 (constants.%array_type.2ec)] {
// CHECK:STDOUT:       %i32.loc14: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %N.ref: %i32 = name_ref N, %N.loc12_6.2 [symbolic = %N.loc12_6.1 (constants.%N.5de)]
// CHECK:STDOUT:       %impl.elem0.loc14_28: %.0a7 = impl_witness_access constants.%ImplicitAs.impl_witness.640, element0 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.dd4]
// CHECK:STDOUT:       %bound_method.loc14_28.1: <bound method> = bound_method %N.ref, %impl.elem0.loc14_28 [symbolic = %Int.as.ImplicitAs.impl.Convert.bound (constants.%Int.as.ImplicitAs.impl.Convert.bound)]
// CHECK:STDOUT:       %specific_fn: <specific function> = specific_function %impl.elem0.loc14_28, @Int.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Int.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:       %bound_method.loc14_28.2: <bound method> = bound_method %N.ref, %specific_fn [symbolic = %bound_method.loc14_28.3 (constants.%bound_method)]
// CHECK:STDOUT:       %Int.as.ImplicitAs.impl.Convert.call.loc14_28.1: init Core.IntLiteral = call %bound_method.loc14_28.2(%N.ref) [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc14_28.2 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:       %.loc14_28.1: Core.IntLiteral = value_of_initializer %Int.as.ImplicitAs.impl.Convert.call.loc14_28.1 [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc14_28.2 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:       %.loc14_28.2: Core.IntLiteral = converted %N.ref, %.loc14_28.1 [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc14_28.2 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:       %array_type.loc14_29.1: type = array_type %.loc14_28.2, %i32.loc14 [symbolic = %array_type.loc14_29.2 (constants.%array_type.2ec)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %k: ref @G.%array_type.loc14_29.2 (%array_type.2ec) = ref_binding k, %k.var
// CHECK:STDOUT:     %impl.elem0.loc14_3.1: @G.%.loc14_3.2 (%.63e) = impl_witness_access constants.%Destroy.lookup_impl_witness.f5c, element0 [symbolic = %impl.elem0.loc14_3.2 (constants.%impl.elem0.2bf)]
// CHECK:STDOUT:     %bound_method.loc14_3.1: <bound method> = bound_method %k.var, %impl.elem0.loc14_3.1
// CHECK:STDOUT:     %specific_impl_fn.loc14_3.1: <specific function> = specific_impl_function %impl.elem0.loc14_3.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.d1e) [symbolic = %specific_impl_fn.loc14_3.2 (constants.%specific_impl_fn.0a5)]
// CHECK:STDOUT:     %bound_method.loc14_3.2: <bound method> = bound_method %k.var, %specific_impl_fn.loc14_3.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc14_3.2(%k.var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.67d) {
// CHECK:STDOUT:   %T.loc4_6.1 => constants.%T.67d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%N.5de) {
// CHECK:STDOUT:   %N.loc12_6.1 => constants.%N.5de
// CHECK:STDOUT: }
// CHECK:STDOUT:

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
  var u: (T*, const T);
  var v: {.a: T};
  var w: array(T, 5);
  //@dump-sem-ir-end
}

fn G(N:! i32) {
  //@dump-sem-ir-begin
  var k: array(i32, N);
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
// CHECK:STDOUT:   %ImplicitAs.type.d14: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f51: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.2a1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f51 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.bc9: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.132, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.51e: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.51e = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.d14 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.bc9) [concrete]
// CHECK:STDOUT:   %.322: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.265: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.dfc: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.393: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b [concrete]
// CHECK:STDOUT:   %bound_method.9f7: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.21c: %tuple.type.d07 = tuple_value (%int_1.5d2, %int_2.ef8) [concrete]
// CHECK:STDOUT:   %tuple.elem0: ref %i32 = tuple_access file.%tuple_copy.var, element0 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.24b: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.c95: %Int.as.Copy.impl.Op.type.24b = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.fb7: <witness> = impl_witness imports.%Copy.impl_witness_table.b6a, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.469: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.dfd: %Int.as.Copy.impl.Op.type.469 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.fb7) [concrete]
// CHECK:STDOUT:   %.65f: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound.89c: <bound method> = bound_method %int_1.5d2, %Int.as.Copy.impl.Op.dfd [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.dfd, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.8c6: <bound method> = bound_method %int_1.5d2, %Int.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %tuple.elem1: ref %i32 = tuple_access file.%tuple_copy.var, element1 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound.b0d: <bound method> = bound_method %int_2.ef8, %Int.as.Copy.impl.Op.dfd [concrete]
// CHECK:STDOUT:   %bound_method.d76: <bound method> = bound_method %int_2.ef8, %Int.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %struct_type.a.b.c: type = struct_type {.a: %i32, .b: %i32, .c: %i32} [concrete]
// CHECK:STDOUT:   %pattern_type.8ae: type = pattern_type %struct_type.a.b.c [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %struct_type.c.b.a: type = struct_type {.c: Core.IntLiteral, .b: Core.IntLiteral, .a: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct.c75: %struct_type.c.b.a = struct_value (%int_3.1ba, %int_2.ecc, %int_1.5b8) [concrete]
// CHECK:STDOUT:   %struct_type.b.a.c: type = struct_type {.b: %i32, .a: %i32, .c: %i32} [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.87d: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b [concrete]
// CHECK:STDOUT:   %bound_method.def: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.822: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %struct.21d: %struct_type.b.a.c = struct_value (%int_2.ef8, %int_1.5d2, %int_3.822) [concrete]
// CHECK:STDOUT:   %.44a: ref %i32 = struct_access file.%struct_copy.var, element1 [concrete]
// CHECK:STDOUT:   %.5b7: ref %i32 = struct_access file.%struct_copy.var, element0 [concrete]
// CHECK:STDOUT:   %.c61: ref %i32 = struct_access file.%struct_copy.var, element2 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound.b68: <bound method> = bound_method %int_3.822, %Int.as.Copy.impl.Op.dfd [concrete]
// CHECK:STDOUT:   %bound_method.996: <bound method> = bound_method %int_3.822, %Int.as.Copy.impl.Op.specific_fn [concrete]
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
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.0b3: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b [concrete]
// CHECK:STDOUT:   %bound_method.9be: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%int_0.6a9) [concrete]
// CHECK:STDOUT:   %struct_type.a.b: type = struct_type {.a: Core.IntLiteral, .b: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct.a81: %struct_type.a.b = struct_value (%int_3.1ba, %int_1.5b8) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.e24: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f51) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.2a1)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.132 = impl_witness_table (%Core.import_ref.e24), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.d12: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.24b) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.c95)]
// CHECK:STDOUT:   %Copy.impl_witness_table.b6a = impl_witness_table (%Core.import_ref.d12), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %tuple_copy.patt: %pattern_type.511 = ref_binding_pattern tuple_copy [concrete]
// CHECK:STDOUT:     %tuple_copy.var_patt: %pattern_type.511 = var_pattern %tuple_copy.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %tuple_copy.var: ref %tuple.type.d07 = var %tuple_copy.var_patt [concrete]
// CHECK:STDOUT:   %.loc4_26.1: type = splice_block %.loc4_26.3 [concrete = constants.%tuple.type.d07] {
// CHECK:STDOUT:     %int_32.loc4_18: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc4_18: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc4_23: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc4_23: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
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
// CHECK:STDOUT:     %int_32.loc6_23: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc6_23: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc6_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc6_32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc6_41: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc6_41: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.a.b.c: type = struct_type {.a: %i32, .b: %i32, .c: %i32} [concrete = constants.%struct_type.a.b.c]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %struct_copy: ref %struct_type.a.b.c = ref_binding struct_copy, %struct_copy.var [concrete = %struct_copy.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %tuple_index.patt: %pattern_type.a98 = ref_binding_pattern tuple_index [concrete]
// CHECK:STDOUT:     %tuple_index.var_patt: %pattern_type.a98 = var_pattern %tuple_index.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %tuple_index.var: ref %array_type = var %tuple_index.var_patt [concrete]
// CHECK:STDOUT:   %.loc8: type = splice_block %array_type.loc8 [concrete = constants.%array_type] {
// CHECK:STDOUT:     %int_32.loc8: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc8: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
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
// CHECK:STDOUT:     %int_32.loc10: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc10: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
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
// CHECK:STDOUT:   %int_32.loc4_41: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc4_41: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %int_32.loc4_46: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc4_46: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc4_49.1: %tuple.type.24b = tuple_literal (%i32.loc4_41, %i32.loc4_46) [concrete = constants.%tuple.95a]
// CHECK:STDOUT:   %.loc4_49.2: type = converted %.loc4_49.1, constants.%tuple.type.d07 [concrete = constants.%tuple.type.d07]
// CHECK:STDOUT:   %impl.elem0.loc4_35.1: %.322 = impl_witness_access constants.%ImplicitAs.impl_witness.bc9, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b]
// CHECK:STDOUT:   %bound_method.loc4_35.1: <bound method> = bound_method %int_1.loc4, %impl.elem0.loc4_35.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.265]
// CHECK:STDOUT:   %specific_fn.loc4_35.1: <specific function> = specific_function %impl.elem0.loc4_35.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc4_35.2: <bound method> = bound_method %int_1.loc4, %specific_fn.loc4_35.1 [concrete = constants.%bound_method.dfc]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc4_35.1: init %i32 = call %bound_method.loc4_35.2(%int_1.loc4) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc4_35.2: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc4_35.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc4_35.3: %i32 = converted %int_1.loc4, %.loc4_35.2 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc4_35.2: %.322 = impl_witness_access constants.%ImplicitAs.impl_witness.bc9, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b]
// CHECK:STDOUT:   %bound_method.loc4_35.3: <bound method> = bound_method %int_2.loc4, %impl.elem0.loc4_35.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.393]
// CHECK:STDOUT:   %specific_fn.loc4_35.2: <specific function> = specific_function %impl.elem0.loc4_35.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc4_35.4: <bound method> = bound_method %int_2.loc4, %specific_fn.loc4_35.2 [concrete = constants.%bound_method.9f7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc4_35.2: init %i32 = call %bound_method.loc4_35.4(%int_2.loc4) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc4_35.4: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc4_35.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc4_35.5: %i32 = converted %int_2.loc4, %.loc4_35.4 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %tuple.loc4: %tuple.type.d07 = tuple_value (%.loc4_35.3, %.loc4_35.5) [concrete = constants.%tuple.21c]
// CHECK:STDOUT:   %.loc4_37.1: %tuple.type.d07 = converted %.loc4_35.1, %tuple.loc4 [concrete = constants.%tuple.21c]
// CHECK:STDOUT:   %tuple.elem0.loc4_37.1: %i32 = tuple_access %.loc4_37.1, element0 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc4_37.1: %.65f = impl_witness_access constants.%Copy.impl_witness.fb7, element0 [concrete = constants.%Int.as.Copy.impl.Op.dfd]
// CHECK:STDOUT:   %bound_method.loc4_37.1: <bound method> = bound_method %tuple.elem0.loc4_37.1, %impl.elem0.loc4_37.1 [concrete = constants.%Int.as.Copy.impl.Op.bound.89c]
// CHECK:STDOUT:   %specific_fn.loc4_37.1: <specific function> = specific_function %impl.elem0.loc4_37.1, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc4_37.2: <bound method> = bound_method %tuple.elem0.loc4_37.1, %specific_fn.loc4_37.1 [concrete = constants.%bound_method.8c6]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc4_37.1: init %i32 = call %bound_method.loc4_37.2(%tuple.elem0.loc4_37.1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %tuple.elem0.loc4_37.2: ref %i32 = tuple_access file.%tuple_copy.var, element0 [concrete = constants.%tuple.elem0]
// CHECK:STDOUT:   %.loc4_37.2: init %i32 = initialize_from %Int.as.Copy.impl.Op.call.loc4_37.1 to %tuple.elem0.loc4_37.2 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %tuple.elem1.loc4_37.1: %i32 = tuple_access %.loc4_37.1, element1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %impl.elem0.loc4_37.2: %.65f = impl_witness_access constants.%Copy.impl_witness.fb7, element0 [concrete = constants.%Int.as.Copy.impl.Op.dfd]
// CHECK:STDOUT:   %bound_method.loc4_37.3: <bound method> = bound_method %tuple.elem1.loc4_37.1, %impl.elem0.loc4_37.2 [concrete = constants.%Int.as.Copy.impl.Op.bound.b0d]
// CHECK:STDOUT:   %specific_fn.loc4_37.2: <specific function> = specific_function %impl.elem0.loc4_37.2, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc4_37.4: <bound method> = bound_method %tuple.elem1.loc4_37.1, %specific_fn.loc4_37.2 [concrete = constants.%bound_method.d76]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc4_37.2: init %i32 = call %bound_method.loc4_37.4(%tuple.elem1.loc4_37.1) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %tuple.elem1.loc4_37.2: ref %i32 = tuple_access file.%tuple_copy.var, element1 [concrete = constants.%tuple.elem1]
// CHECK:STDOUT:   %.loc4_37.3: init %i32 = initialize_from %Int.as.Copy.impl.Op.call.loc4_37.2 to %tuple.elem1.loc4_37.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc4_37.4: init %tuple.type.d07 = tuple_init (%.loc4_37.2, %.loc4_37.3) to file.%tuple_copy.var [concrete = constants.%tuple.21c]
// CHECK:STDOUT:   %.loc4_1: init %tuple.type.d07 = converted %.loc4_37.1, %.loc4_37.4 [concrete = constants.%tuple.21c]
// CHECK:STDOUT:   assign file.%tuple_copy.var, %.loc4_1
// CHECK:STDOUT:   %int_3.loc6: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %int_2.loc6: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %int_1.loc6: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc6_71.1: %struct_type.c.b.a = struct_literal (%int_3.loc6, %int_2.loc6, %int_1.loc6) [concrete = constants.%struct.c75]
// CHECK:STDOUT:   %int_32.loc6_81: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc6_81: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %int_32.loc6_90: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc6_90: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %int_32.loc6_99: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc6_99: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %struct_type.b.a.c: type = struct_type {.b: %i32, .a: %i32, .c: %i32} [concrete = constants.%struct_type.b.a.c]
// CHECK:STDOUT:   %impl.elem0.loc6_71.1: %.322 = impl_witness_access constants.%ImplicitAs.impl_witness.bc9, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b]
// CHECK:STDOUT:   %bound_method.loc6_71.1: <bound method> = bound_method %int_2.loc6, %impl.elem0.loc6_71.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.393]
// CHECK:STDOUT:   %specific_fn.loc6_71.1: <specific function> = specific_function %impl.elem0.loc6_71.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_71.2: <bound method> = bound_method %int_2.loc6, %specific_fn.loc6_71.1 [concrete = constants.%bound_method.9f7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_71.1: init %i32 = call %bound_method.loc6_71.2(%int_2.loc6) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc6_71.2: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_71.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc6_71.3: %i32 = converted %int_2.loc6, %.loc6_71.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %impl.elem0.loc6_71.2: %.322 = impl_witness_access constants.%ImplicitAs.impl_witness.bc9, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b]
// CHECK:STDOUT:   %bound_method.loc6_71.3: <bound method> = bound_method %int_1.loc6, %impl.elem0.loc6_71.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.265]
// CHECK:STDOUT:   %specific_fn.loc6_71.2: <specific function> = specific_function %impl.elem0.loc6_71.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_71.4: <bound method> = bound_method %int_1.loc6, %specific_fn.loc6_71.2 [concrete = constants.%bound_method.dfc]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_71.2: init %i32 = call %bound_method.loc6_71.4(%int_1.loc6) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc6_71.4: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_71.2 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc6_71.5: %i32 = converted %int_1.loc6, %.loc6_71.4 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc6_71.3: %.322 = impl_witness_access constants.%ImplicitAs.impl_witness.bc9, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b]
// CHECK:STDOUT:   %bound_method.loc6_71.5: <bound method> = bound_method %int_3.loc6, %impl.elem0.loc6_71.3 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.87d]
// CHECK:STDOUT:   %specific_fn.loc6_71.3: <specific function> = specific_function %impl.elem0.loc6_71.3, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_71.6: <bound method> = bound_method %int_3.loc6, %specific_fn.loc6_71.3 [concrete = constants.%bound_method.def]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_71.3: init %i32 = call %bound_method.loc6_71.6(%int_3.loc6) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc6_71.6: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_71.3 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc6_71.7: %i32 = converted %int_3.loc6, %.loc6_71.6 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %struct.loc6: %struct_type.b.a.c = struct_value (%.loc6_71.3, %.loc6_71.5, %.loc6_71.7) [concrete = constants.%struct.21d]
// CHECK:STDOUT:   %.loc6_73.1: %struct_type.b.a.c = converted %.loc6_71.1, %struct.loc6 [concrete = constants.%struct.21d]
// CHECK:STDOUT:   %.loc6_73.2: %i32 = struct_access %.loc6_73.1, element1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc6_73.1: %.65f = impl_witness_access constants.%Copy.impl_witness.fb7, element0 [concrete = constants.%Int.as.Copy.impl.Op.dfd]
// CHECK:STDOUT:   %bound_method.loc6_73.1: <bound method> = bound_method %.loc6_73.2, %impl.elem0.loc6_73.1 [concrete = constants.%Int.as.Copy.impl.Op.bound.89c]
// CHECK:STDOUT:   %specific_fn.loc6_73.1: <specific function> = specific_function %impl.elem0.loc6_73.1, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_73.2: <bound method> = bound_method %.loc6_73.2, %specific_fn.loc6_73.1 [concrete = constants.%bound_method.8c6]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc6_73.1: init %i32 = call %bound_method.loc6_73.2(%.loc6_73.2) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc6_73.3: ref %i32 = struct_access file.%struct_copy.var, element1 [concrete = constants.%.44a]
// CHECK:STDOUT:   %.loc6_73.4: init %i32 = initialize_from %Int.as.Copy.impl.Op.call.loc6_73.1 to %.loc6_73.3 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc6_73.5: %i32 = struct_access %.loc6_73.1, element0 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %impl.elem0.loc6_73.2: %.65f = impl_witness_access constants.%Copy.impl_witness.fb7, element0 [concrete = constants.%Int.as.Copy.impl.Op.dfd]
// CHECK:STDOUT:   %bound_method.loc6_73.3: <bound method> = bound_method %.loc6_73.5, %impl.elem0.loc6_73.2 [concrete = constants.%Int.as.Copy.impl.Op.bound.b0d]
// CHECK:STDOUT:   %specific_fn.loc6_73.2: <specific function> = specific_function %impl.elem0.loc6_73.2, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_73.4: <bound method> = bound_method %.loc6_73.5, %specific_fn.loc6_73.2 [concrete = constants.%bound_method.d76]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc6_73.2: init %i32 = call %bound_method.loc6_73.4(%.loc6_73.5) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc6_73.6: ref %i32 = struct_access file.%struct_copy.var, element0 [concrete = constants.%.5b7]
// CHECK:STDOUT:   %.loc6_73.7: init %i32 = initialize_from %Int.as.Copy.impl.Op.call.loc6_73.2 to %.loc6_73.6 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc6_73.8: %i32 = struct_access %.loc6_73.1, element2 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %impl.elem0.loc6_73.3: %.65f = impl_witness_access constants.%Copy.impl_witness.fb7, element0 [concrete = constants.%Int.as.Copy.impl.Op.dfd]
// CHECK:STDOUT:   %bound_method.loc6_73.5: <bound method> = bound_method %.loc6_73.8, %impl.elem0.loc6_73.3 [concrete = constants.%Int.as.Copy.impl.Op.bound.b68]
// CHECK:STDOUT:   %specific_fn.loc6_73.3: <specific function> = specific_function %impl.elem0.loc6_73.3, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_73.6: <bound method> = bound_method %.loc6_73.8, %specific_fn.loc6_73.3 [concrete = constants.%bound_method.996]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc6_73.3: init %i32 = call %bound_method.loc6_73.6(%.loc6_73.8) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc6_73.9: ref %i32 = struct_access file.%struct_copy.var, element2 [concrete = constants.%.c61]
// CHECK:STDOUT:   %.loc6_73.10: init %i32 = initialize_from %Int.as.Copy.impl.Op.call.loc6_73.3 to %.loc6_73.9 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc6_73.11: init %struct_type.a.b.c = struct_init (%.loc6_73.4, %.loc6_73.7, %.loc6_73.10) to file.%struct_copy.var [concrete = constants.%struct.cff]
// CHECK:STDOUT:   %.loc6_1: init %struct_type.a.b.c = converted %.loc6_73.1, %.loc6_73.11 [concrete = constants.%struct.cff]
// CHECK:STDOUT:   assign file.%struct_copy.var, %.loc6_1
// CHECK:STDOUT:   %int_0.loc8_35: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc8_37.1: %tuple.type.985 = tuple_literal (%int_0.loc8_35) [concrete = constants.%tuple.4f2]
// CHECK:STDOUT:   %int_32.loc8: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc8: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
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
// CHECK:STDOUT:   %impl.elem0.loc8: %.322 = impl_witness_access constants.%ImplicitAs.impl_witness.bc9, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b]
// CHECK:STDOUT:   %bound_method.loc8_37.1: <bound method> = bound_method %int_0.loc8_35, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.0b3]
// CHECK:STDOUT:   %specific_fn.loc8: <specific function> = specific_function %impl.elem0.loc8, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_37.2: <bound method> = bound_method %int_0.loc8_35, %specific_fn.loc8 [concrete = constants.%bound_method.9be]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8: init %i32 = call %bound_method.loc8_37.2(%int_0.loc8_35) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc8_37.2: init %i32 = converted %int_0.loc8_35, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc8_1: ref %array_type = splice_block file.%tuple_index.var [concrete = file.%tuple_index.var] {}
// CHECK:STDOUT:   %int_0.loc8_37: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc8_37.3: ref %i32 = array_index %.loc8_1, %int_0.loc8_37
// CHECK:STDOUT:   %.loc8_37.4: init %i32 = initialize_from %.loc8_37.2 to %.loc8_37.3 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc8_37.5: init %array_type = array_init (%.loc8_37.4) to %.loc8_1 [concrete = constants.%array]
// CHECK:STDOUT:   %.loc8_39: init %array_type = converted %.loc8_37.1, %.loc8_37.5 [concrete = constants.%array]
// CHECK:STDOUT:   assign file.%tuple_index.var, %.loc8_39
// CHECK:STDOUT:   %int_0.loc10_37: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc10_39.1: %tuple.type.985 = tuple_literal (%int_0.loc10_37) [concrete = constants.%tuple.4f2]
// CHECK:STDOUT:   %int_32.loc10: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc10: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %int_3.loc10: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %int_1.loc10: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc10_70.1: %struct_type.a.b = struct_literal (%int_3.loc10, %int_1.loc10) [concrete = constants.%struct.a81]
// CHECK:STDOUT:   %struct.loc10: %struct_type.a.b = struct_value (%int_3.loc10, %int_1.loc10) [concrete = constants.%struct.a81]
// CHECK:STDOUT:   %.loc10_70.2: %struct_type.a.b = converted %.loc10_70.1, %struct.loc10 [concrete = constants.%struct.a81]
// CHECK:STDOUT:   %.loc10_71: Core.IntLiteral = struct_access %.loc10_70.2, element1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %array_type.loc10: type = array_type %.loc10_71, %i32.loc10 [concrete = constants.%array_type]
// CHECK:STDOUT:   %impl.elem0.loc10: %.322 = impl_witness_access constants.%ImplicitAs.impl_witness.bc9, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b]
// CHECK:STDOUT:   %bound_method.loc10_39.1: <bound method> = bound_method %int_0.loc10_37, %impl.elem0.loc10 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.0b3]
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_39.2: <bound method> = bound_method %int_0.loc10_37, %specific_fn.loc10 [concrete = constants.%bound_method.9be]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10: init %i32 = call %bound_method.loc10_39.2(%int_0.loc10_37) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc10_39.2: init %i32 = converted %int_0.loc10_37, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc10_1: ref %array_type = splice_block file.%struct_access.var [concrete = file.%struct_access.var] {}
// CHECK:STDOUT:   %int_0.loc10_39: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc10_39.3: ref %i32 = array_index %.loc10_1, %int_0.loc10_39
// CHECK:STDOUT:   %.loc10_39.4: init %i32 = initialize_from %.loc10_39.2 to %.loc10_39.3 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc10_39.5: init %array_type = array_init (%.loc10_39.4) to %.loc10_1 [concrete = constants.%array]
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
// CHECK:STDOUT:   %ImplicitAs.type.d14: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f51: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.2a1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f51 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.bc9: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.132, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.51e: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.51e = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.d14 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.bc9) [concrete]
// CHECK:STDOUT:   %.322: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.451: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.c57: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_5.0f6: %i32 = int_value 5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.a66: <bound method> = bound_method %int_7.29f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b [concrete]
// CHECK:STDOUT:   %bound_method.1db: <bound method> = bound_method %int_7.29f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_7.0b1: %i32 = int_value 7 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.265: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b [concrete]
// CHECK:STDOUT:   %bound_method.dfc: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.268: <bound method> = bound_method %int_9.988, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b [concrete]
// CHECK:STDOUT:   %bound_method.e3f: <bound method> = bound_method %int_9.988, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_9.f88: %i32 = int_value 9 [concrete]
// CHECK:STDOUT:   %array: %array_type.f32 = tuple_value (%int_5.0f6, %int_7.0b1, %int_1.5d2, %int_9.f88) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.393: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b [concrete]
// CHECK:STDOUT:   %bound_method.9f7: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.e24: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f51) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.2a1)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.132 = impl_witness_table (%Core.import_ref.e24), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %array_index.patt: %pattern_type.a98 = ref_binding_pattern array_index [concrete]
// CHECK:STDOUT:     %array_index.var_patt: %pattern_type.a98 = var_pattern %array_index.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %array_index.var: ref %array_type.0cb = var %array_index.var_patt [concrete]
// CHECK:STDOUT:   %.loc9: type = splice_block %array_type [concrete = constants.%array_type.0cb] {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
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
// CHECK:STDOUT:   %int_32.loc9_48: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc9_48: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5.64b]
// CHECK:STDOUT:   %int_7: Core.IntLiteral = int_value 7 [concrete = constants.%int_7.29f]
// CHECK:STDOUT:   %int_1.loc9_61: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_9: Core.IntLiteral = int_value 9 [concrete = constants.%int_9.988]
// CHECK:STDOUT:   %.loc9_65.1: %tuple.type.d46 = tuple_literal (%int_5, %int_7, %int_1.loc9_61, %int_9) [concrete = constants.%tuple.869]
// CHECK:STDOUT:   %int_32.loc9_76: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc9_76: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4]
// CHECK:STDOUT:   %array_type: type = array_type %int_4, %i32.loc9_76 [concrete = constants.%array_type.f32]
// CHECK:STDOUT:   %impl.elem0.loc9_65.1: %.322 = impl_witness_access constants.%ImplicitAs.impl_witness.bc9, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b]
// CHECK:STDOUT:   %bound_method.loc9_65.1: <bound method> = bound_method %int_5, %impl.elem0.loc9_65.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.451]
// CHECK:STDOUT:   %specific_fn.loc9_65.1: <specific function> = specific_function %impl.elem0.loc9_65.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_65.2: <bound method> = bound_method %int_5, %specific_fn.loc9_65.1 [concrete = constants.%bound_method.c57]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_65.1: init %i32 = call %bound_method.loc9_65.2(%int_5) [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc9_65.2: init %i32 = converted %int_5, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_65.1 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc9_65.3: ref %array_type.f32 = temporary_storage
// CHECK:STDOUT:   %int_0.loc9_65: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc9_65.4: ref %i32 = array_index %.loc9_65.3, %int_0.loc9_65
// CHECK:STDOUT:   %.loc9_65.5: init %i32 = initialize_from %.loc9_65.2 to %.loc9_65.4 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %impl.elem0.loc9_65.2: %.322 = impl_witness_access constants.%ImplicitAs.impl_witness.bc9, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b]
// CHECK:STDOUT:   %bound_method.loc9_65.3: <bound method> = bound_method %int_7, %impl.elem0.loc9_65.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.a66]
// CHECK:STDOUT:   %specific_fn.loc9_65.2: <specific function> = specific_function %impl.elem0.loc9_65.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_65.4: <bound method> = bound_method %int_7, %specific_fn.loc9_65.2 [concrete = constants.%bound_method.1db]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_65.2: init %i32 = call %bound_method.loc9_65.4(%int_7) [concrete = constants.%int_7.0b1]
// CHECK:STDOUT:   %.loc9_65.6: init %i32 = converted %int_7, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_65.2 [concrete = constants.%int_7.0b1]
// CHECK:STDOUT:   %int_1.loc9_65: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc9_65.7: ref %i32 = array_index %.loc9_65.3, %int_1.loc9_65
// CHECK:STDOUT:   %.loc9_65.8: init %i32 = initialize_from %.loc9_65.6 to %.loc9_65.7 [concrete = constants.%int_7.0b1]
// CHECK:STDOUT:   %impl.elem0.loc9_65.3: %.322 = impl_witness_access constants.%ImplicitAs.impl_witness.bc9, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b]
// CHECK:STDOUT:   %bound_method.loc9_65.5: <bound method> = bound_method %int_1.loc9_61, %impl.elem0.loc9_65.3 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.265]
// CHECK:STDOUT:   %specific_fn.loc9_65.3: <specific function> = specific_function %impl.elem0.loc9_65.3, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_65.6: <bound method> = bound_method %int_1.loc9_61, %specific_fn.loc9_65.3 [concrete = constants.%bound_method.dfc]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_65.3: init %i32 = call %bound_method.loc9_65.6(%int_1.loc9_61) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc9_65.9: init %i32 = converted %int_1.loc9_61, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_65.3 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %int_2.loc9_65: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc9_65.10: ref %i32 = array_index %.loc9_65.3, %int_2.loc9_65
// CHECK:STDOUT:   %.loc9_65.11: init %i32 = initialize_from %.loc9_65.9 to %.loc9_65.10 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc9_65.4: %.322 = impl_witness_access constants.%ImplicitAs.impl_witness.bc9, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b]
// CHECK:STDOUT:   %bound_method.loc9_65.7: <bound method> = bound_method %int_9, %impl.elem0.loc9_65.4 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.268]
// CHECK:STDOUT:   %specific_fn.loc9_65.4: <specific function> = specific_function %impl.elem0.loc9_65.4, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_65.8: <bound method> = bound_method %int_9, %specific_fn.loc9_65.4 [concrete = constants.%bound_method.e3f]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_65.4: init %i32 = call %bound_method.loc9_65.8(%int_9) [concrete = constants.%int_9.f88]
// CHECK:STDOUT:   %.loc9_65.12: init %i32 = converted %int_9, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_65.4 [concrete = constants.%int_9.f88]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:   %.loc9_65.13: ref %i32 = array_index %.loc9_65.3, %int_3
// CHECK:STDOUT:   %.loc9_65.14: init %i32 = initialize_from %.loc9_65.12 to %.loc9_65.13 [concrete = constants.%int_9.f88]
// CHECK:STDOUT:   %.loc9_65.15: init %array_type.f32 = array_init (%.loc9_65.5, %.loc9_65.8, %.loc9_65.11, %.loc9_65.14) to %.loc9_65.3 [concrete = constants.%array]
// CHECK:STDOUT:   %.loc9_67.1: init %array_type.f32 = converted %.loc9_65.1, %.loc9_65.15 [concrete = constants.%array]
// CHECK:STDOUT:   %int_2.loc9_85: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc9_67.2: ref %array_type.f32 = temporary %.loc9_65.3, %.loc9_67.1
// CHECK:STDOUT:   %int_32.loc9_86: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc9_86: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc9_85: %.322 = impl_witness_access constants.%ImplicitAs.impl_witness.bc9, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b]
// CHECK:STDOUT:   %bound_method.loc9_85.1: <bound method> = bound_method %int_2.loc9_85, %impl.elem0.loc9_85 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.393]
// CHECK:STDOUT:   %specific_fn.loc9_85: <specific function> = specific_function %impl.elem0.loc9_85, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_85.2: <bound method> = bound_method %int_2.loc9_85, %specific_fn.loc9_85 [concrete = constants.%bound_method.9f7]
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
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %ptr.4f0: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %const: type = const_type %T [symbolic]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.24b = tuple_value (%ptr.4f0, %const) [symbolic]
// CHECK:STDOUT:   %tuple.type.2d0: type = tuple_type (%ptr.4f0, %const) [symbolic]
// CHECK:STDOUT:   %require_complete.b52: <witness> = require_complete_type %tuple.type.2d0 [symbolic]
// CHECK:STDOUT:   %pattern_type.9dc: type = pattern_type %tuple.type.2d0 [symbolic]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %T} [symbolic]
// CHECK:STDOUT:   %require_complete.849: <witness> = require_complete_type %struct_type.a [symbolic]
// CHECK:STDOUT:   %pattern_type.68a: type = pattern_type %struct_type.a [symbolic]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %array_type.297: type = array_type %int_5, %T [symbolic]
// CHECK:STDOUT:   %require_complete.e3e: <witness> = require_complete_type %array_type.297 [symbolic]
// CHECK:STDOUT:   %pattern_type.9a2: type = pattern_type %array_type.297 [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %DestroyT: %type_where = symbolic_binding DestroyT, 0 [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.15c: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%DestroyT) [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.fd5: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.15c = struct_value () [symbolic]
// CHECK:STDOUT:   %facet_value.f78: %type_where = facet_value %array_type.297, () [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.fed: <witness> = impl_witness imports.%Destroy.impl_witness_table, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.f78) [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.6fe: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.f78) [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.440: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.6fe = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.facet.d98: %Destroy.type = facet_value %array_type.297, (%Destroy.impl_witness.fed) [symbolic]
// CHECK:STDOUT:   %.539: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet.d98 [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.cd0: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op.440, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value.f78) [symbolic]
// CHECK:STDOUT:   %facet_value.d02: %type_where = facet_value %struct_type.a, () [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.e1b: <witness> = impl_witness imports.%Destroy.impl_witness_table, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.d02) [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.722: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.d02) [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.a8b: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.722 = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.facet.1ca: %Destroy.type = facet_value %struct_type.a, (%Destroy.impl_witness.e1b) [symbolic]
// CHECK:STDOUT:   %.506: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet.1ca [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.df3: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op.a8b, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value.d02) [symbolic]
// CHECK:STDOUT:   %facet_value.0f0: %type_where = facet_value %tuple.type.2d0, () [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.ef2: <witness> = impl_witness imports.%Destroy.impl_witness_table, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.0f0) [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.36c: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.0f0) [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.2b6: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.36c = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.facet.ce6: %Destroy.type = facet_value %tuple.type.2d0, (%Destroy.impl_witness.ef2) [symbolic]
// CHECK:STDOUT:   %.271: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet.ce6 [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.e81: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op.2b6, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value.0f0) [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %N.fe8: %i32 = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.7a9: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.71e: type = fn_type @ImplicitAs.Convert, @ImplicitAs(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %From: Core.IntLiteral = symbolic_binding From, 0 [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.2c2: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%From) [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.cdf: %Int.as.ImplicitAs.impl.Convert.type.2c2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.972: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.783, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.592: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.047: %Int.as.ImplicitAs.impl.Convert.type.592 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.7a9 = facet_value %i32, (%ImplicitAs.impl_witness.972) [concrete]
// CHECK:STDOUT:   %.63b: type = fn_type_with_self_type %ImplicitAs.Convert.type.71e, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %N.fe8, %Int.as.ImplicitAs.impl.Convert.047 [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Int.as.ImplicitAs.impl.Convert.047, @Int.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %N.fe8, %Int.as.ImplicitAs.impl.Convert.specific_fn [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method(%N.fe8) [symbolic]
// CHECK:STDOUT:   %array_type.58c: type = array_type %Int.as.ImplicitAs.impl.Convert.call, %i32 [symbolic]
// CHECK:STDOUT:   %require_complete.33a: <witness> = require_complete_type %array_type.58c [symbolic]
// CHECK:STDOUT:   %pattern_type.e6c: type = pattern_type %array_type.58c [symbolic]
// CHECK:STDOUT:   %facet_value.8c5: %type_where = facet_value %array_type.58c, () [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.db6: <witness> = impl_witness imports.%Destroy.impl_witness_table, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.8c5) [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.405: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.8c5) [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.a68: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.405 = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.facet.bb1: %Destroy.type = facet_value %array_type.58c, (%Destroy.impl_witness.db6) [symbolic]
// CHECK:STDOUT:   %.cc4: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet.bb1 [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.695: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op.a68, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value.8c5) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.5c4: @DestroyT.binding.as_type.as.Destroy.impl.%DestroyT.binding.as_type.as.Destroy.impl.Op.type (%DestroyT.binding.as_type.as.Destroy.impl.Op.type.15c) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @DestroyT.binding.as_type.as.Destroy.impl.%DestroyT.binding.as_type.as.Destroy.impl.Op (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.fd5)]
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (%Core.import_ref.5c4), @DestroyT.binding.as_type.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.a86: @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert.type (%Int.as.ImplicitAs.impl.Convert.type.2c2) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert (constants.%Int.as.ImplicitAs.impl.Convert.cdf)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.783 = impl_witness_table (%Core.import_ref.a86), @Int.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc4_6.2: type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ptr.loc6_12.2: type = ptr_type %T.loc4_6.1 [symbolic = %ptr.loc6_12.2 (constants.%ptr.4f0)]
// CHECK:STDOUT:   %const.loc6_15.2: type = const_type %T.loc4_6.1 [symbolic = %const.loc6_15.2 (constants.%const)]
// CHECK:STDOUT:   %tuple: %tuple.type.24b = tuple_value (%ptr.loc6_12.2, %const.loc6_15.2) [symbolic = %tuple (constants.%tuple)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%ptr.loc6_12.2, %const.loc6_15.2) [symbolic = %tuple.type (constants.%tuple.type.2d0)]
// CHECK:STDOUT:   %require_complete.loc6: <witness> = require_complete_type %tuple.type [symbolic = %require_complete.loc6 (constants.%require_complete.b52)]
// CHECK:STDOUT:   %pattern_type.loc6: type = pattern_type %tuple.type [symbolic = %pattern_type.loc6 (constants.%pattern_type.9dc)]
// CHECK:STDOUT:   %struct_type.a.loc7_16.2: type = struct_type {.a: @F.%T.loc4_6.1 (%T)} [symbolic = %struct_type.a.loc7_16.2 (constants.%struct_type.a)]
// CHECK:STDOUT:   %require_complete.loc7: <witness> = require_complete_type %struct_type.a.loc7_16.2 [symbolic = %require_complete.loc7 (constants.%require_complete.849)]
// CHECK:STDOUT:   %pattern_type.loc7: type = pattern_type %struct_type.a.loc7_16.2 [symbolic = %pattern_type.loc7 (constants.%pattern_type.68a)]
// CHECK:STDOUT:   %array_type.loc8_20.2: type = array_type constants.%int_5, %T.loc4_6.1 [symbolic = %array_type.loc8_20.2 (constants.%array_type.297)]
// CHECK:STDOUT:   %require_complete.loc8: <witness> = require_complete_type %array_type.loc8_20.2 [symbolic = %require_complete.loc8 (constants.%require_complete.e3e)]
// CHECK:STDOUT:   %pattern_type.loc8: type = pattern_type %array_type.loc8_20.2 [symbolic = %pattern_type.loc8 (constants.%pattern_type.9a2)]
// CHECK:STDOUT:   %facet_value.loc8: %type_where = facet_value %array_type.loc8_20.2, () [symbolic = %facet_value.loc8 (constants.%facet_value.f78)]
// CHECK:STDOUT:   %Destroy.impl_witness.loc8: <witness> = impl_witness imports.%Destroy.impl_witness_table, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.loc8) [symbolic = %Destroy.impl_witness.loc8 (constants.%Destroy.impl_witness.fed)]
// CHECK:STDOUT:   %Destroy.facet.loc8: %Destroy.type = facet_value %array_type.loc8_20.2, (%Destroy.impl_witness.loc8) [symbolic = %Destroy.facet.loc8 (constants.%Destroy.facet.d98)]
// CHECK:STDOUT:   %.loc8_3: type = fn_type_with_self_type constants.%Destroy.Op.type, %Destroy.facet.loc8 [symbolic = %.loc8_3 (constants.%.539)]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.loc8: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.loc8) [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op.type.loc8 (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.type.6fe)]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.loc8: @F.%DestroyT.binding.as_type.as.Destroy.impl.Op.type.loc8 (%DestroyT.binding.as_type.as.Destroy.impl.Op.type.6fe) = struct_value () [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op.loc8 (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.440)]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.loc8: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op.loc8, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value.loc8) [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.loc8 (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.cd0)]
// CHECK:STDOUT:   %facet_value.loc7: %type_where = facet_value %struct_type.a.loc7_16.2, () [symbolic = %facet_value.loc7 (constants.%facet_value.d02)]
// CHECK:STDOUT:   %Destroy.impl_witness.loc7: <witness> = impl_witness imports.%Destroy.impl_witness_table, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.loc7) [symbolic = %Destroy.impl_witness.loc7 (constants.%Destroy.impl_witness.e1b)]
// CHECK:STDOUT:   %Destroy.facet.loc7: %Destroy.type = facet_value %struct_type.a.loc7_16.2, (%Destroy.impl_witness.loc7) [symbolic = %Destroy.facet.loc7 (constants.%Destroy.facet.1ca)]
// CHECK:STDOUT:   %.loc7_3: type = fn_type_with_self_type constants.%Destroy.Op.type, %Destroy.facet.loc7 [symbolic = %.loc7_3 (constants.%.506)]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.loc7: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.loc7) [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op.type.loc7 (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.type.722)]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.loc7: @F.%DestroyT.binding.as_type.as.Destroy.impl.Op.type.loc7 (%DestroyT.binding.as_type.as.Destroy.impl.Op.type.722) = struct_value () [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op.loc7 (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.a8b)]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.loc7: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op.loc7, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value.loc7) [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.loc7 (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.df3)]
// CHECK:STDOUT:   %facet_value.loc6: %type_where = facet_value %tuple.type, () [symbolic = %facet_value.loc6 (constants.%facet_value.0f0)]
// CHECK:STDOUT:   %Destroy.impl_witness.loc6: <witness> = impl_witness imports.%Destroy.impl_witness_table, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.loc6) [symbolic = %Destroy.impl_witness.loc6 (constants.%Destroy.impl_witness.ef2)]
// CHECK:STDOUT:   %Destroy.facet.loc6: %Destroy.type = facet_value %tuple.type, (%Destroy.impl_witness.loc6) [symbolic = %Destroy.facet.loc6 (constants.%Destroy.facet.ce6)]
// CHECK:STDOUT:   %.loc6_3: type = fn_type_with_self_type constants.%Destroy.Op.type, %Destroy.facet.loc6 [symbolic = %.loc6_3 (constants.%.271)]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.loc6: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.loc6) [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op.type.loc6 (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.type.36c)]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.loc6: @F.%DestroyT.binding.as_type.as.Destroy.impl.Op.type.loc6 (%DestroyT.binding.as_type.as.Destroy.impl.Op.type.36c) = struct_value () [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op.loc6 (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.2b6)]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.loc6: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op.loc6, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value.loc6) [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.loc6 (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.e81)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %u.patt: @F.%pattern_type.loc6 (%pattern_type.9dc) = ref_binding_pattern u [concrete]
// CHECK:STDOUT:       %u.var_patt: @F.%pattern_type.loc6 (%pattern_type.9dc) = var_pattern %u.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u.var: ref @F.%tuple.type (%tuple.type.2d0) = var %u.var_patt
// CHECK:STDOUT:     %.loc6_22.1: type = splice_block %.loc6_22.3 [symbolic = %tuple.type (constants.%tuple.type.2d0)] {
// CHECK:STDOUT:       %T.ref.loc6_11: type = name_ref T, %T.loc4_6.2 [symbolic = %T.loc4_6.1 (constants.%T)]
// CHECK:STDOUT:       %ptr.loc6_12.1: type = ptr_type %T.ref.loc6_11 [symbolic = %ptr.loc6_12.2 (constants.%ptr.4f0)]
// CHECK:STDOUT:       %T.ref.loc6_21: type = name_ref T, %T.loc4_6.2 [symbolic = %T.loc4_6.1 (constants.%T)]
// CHECK:STDOUT:       %const.loc6_15.1: type = const_type %T.ref.loc6_21 [symbolic = %const.loc6_15.2 (constants.%const)]
// CHECK:STDOUT:       %.loc6_22.2: %tuple.type.24b = tuple_literal (%ptr.loc6_12.1, %const.loc6_15.1) [symbolic = %tuple (constants.%tuple)]
// CHECK:STDOUT:       %.loc6_22.3: type = converted %.loc6_22.2, constants.%tuple.type.2d0 [symbolic = %tuple.type (constants.%tuple.type.2d0)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: ref @F.%tuple.type (%tuple.type.2d0) = ref_binding u, %u.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %v.patt: @F.%pattern_type.loc7 (%pattern_type.68a) = ref_binding_pattern v [concrete]
// CHECK:STDOUT:       %v.var_patt: @F.%pattern_type.loc7 (%pattern_type.68a) = var_pattern %v.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v.var: ref @F.%struct_type.a.loc7_16.2 (%struct_type.a) = var %v.var_patt
// CHECK:STDOUT:     %.loc7_16: type = splice_block %struct_type.a.loc7_16.1 [symbolic = %struct_type.a.loc7_16.2 (constants.%struct_type.a)] {
// CHECK:STDOUT:       %T.ref.loc7: type = name_ref T, %T.loc4_6.2 [symbolic = %T.loc4_6.1 (constants.%T)]
// CHECK:STDOUT:       %struct_type.a.loc7_16.1: type = struct_type {.a: @F.%T.loc4_6.1 (%T)} [symbolic = %struct_type.a.loc7_16.2 (constants.%struct_type.a)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: ref @F.%struct_type.a.loc7_16.2 (%struct_type.a) = ref_binding v, %v.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %w.patt: @F.%pattern_type.loc8 (%pattern_type.9a2) = ref_binding_pattern w [concrete]
// CHECK:STDOUT:       %w.var_patt: @F.%pattern_type.loc8 (%pattern_type.9a2) = var_pattern %w.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %w.var: ref @F.%array_type.loc8_20.2 (%array_type.297) = var %w.var_patt
// CHECK:STDOUT:     %.loc8_20: type = splice_block %array_type.loc8_20.1 [symbolic = %array_type.loc8_20.2 (constants.%array_type.297)] {
// CHECK:STDOUT:       %T.ref.loc8: type = name_ref T, %T.loc4_6.2 [symbolic = %T.loc4_6.1 (constants.%T)]
// CHECK:STDOUT:       %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5]
// CHECK:STDOUT:       %array_type.loc8_20.1: type = array_type %int_5, %T.ref.loc8 [symbolic = %array_type.loc8_20.2 (constants.%array_type.297)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %w: ref @F.%array_type.loc8_20.2 (%array_type.297) = ref_binding w, %w.var
// CHECK:STDOUT:     %impl.elem0.loc8: @F.%.loc8_3 (%.539) = impl_witness_access constants.%Destroy.impl_witness.fed, element0 [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op.loc8 (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.440)]
// CHECK:STDOUT:     %bound_method.loc8_3.1: <bound method> = bound_method %w.var, %impl.elem0.loc8
// CHECK:STDOUT:     %specific_fn.loc8: <specific function> = specific_function %impl.elem0.loc8, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.f78) [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.loc8 (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.cd0)]
// CHECK:STDOUT:     %bound_method.loc8_3.2: <bound method> = bound_method %w.var, %specific_fn.loc8
// CHECK:STDOUT:     %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc8: init %empty_tuple.type = call %bound_method.loc8_3.2(%w.var)
// CHECK:STDOUT:     %impl.elem0.loc7: @F.%.loc7_3 (%.506) = impl_witness_access constants.%Destroy.impl_witness.e1b, element0 [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op.loc7 (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.a8b)]
// CHECK:STDOUT:     %bound_method.loc7_3.1: <bound method> = bound_method %v.var, %impl.elem0.loc7
// CHECK:STDOUT:     %specific_fn.loc7: <specific function> = specific_function %impl.elem0.loc7, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.d02) [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.loc7 (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.df3)]
// CHECK:STDOUT:     %bound_method.loc7_3.2: <bound method> = bound_method %v.var, %specific_fn.loc7
// CHECK:STDOUT:     %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc7: init %empty_tuple.type = call %bound_method.loc7_3.2(%v.var)
// CHECK:STDOUT:     %impl.elem0.loc6: @F.%.loc6_3 (%.271) = impl_witness_access constants.%Destroy.impl_witness.ef2, element0 [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op.loc6 (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.2b6)]
// CHECK:STDOUT:     %bound_method.loc6_3.1: <bound method> = bound_method %u.var, %impl.elem0.loc6
// CHECK:STDOUT:     %specific_fn.loc6: <specific function> = specific_function %impl.elem0.loc6, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.0f0) [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.loc6 (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.e81)]
// CHECK:STDOUT:     %bound_method.loc6_3.2: <bound method> = bound_method %u.var, %specific_fn.loc6
// CHECK:STDOUT:     %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc6: init %empty_tuple.type = call %bound_method.loc6_3.2(%u.var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%N.loc12_6.2: %i32) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %N.loc12_6.1, constants.%Int.as.ImplicitAs.impl.Convert.047 [symbolic = %Int.as.ImplicitAs.impl.Convert.bound (constants.%Int.as.ImplicitAs.impl.Convert.bound)]
// CHECK:STDOUT:   %bound_method.loc14_21.3: <bound method> = bound_method %N.loc12_6.1, constants.%Int.as.ImplicitAs.impl.Convert.specific_fn [symbolic = %bound_method.loc14_21.3 (constants.%bound_method)]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.call.loc14_21.2: init Core.IntLiteral = call %bound_method.loc14_21.3(%N.loc12_6.1) [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc14_21.2 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:   %array_type.loc14_22.2: type = array_type %Int.as.ImplicitAs.impl.Convert.call.loc14_21.2, constants.%i32 [symbolic = %array_type.loc14_22.2 (constants.%array_type.58c)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %array_type.loc14_22.2 [symbolic = %require_complete (constants.%require_complete.33a)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %array_type.loc14_22.2 [symbolic = %pattern_type (constants.%pattern_type.e6c)]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %array_type.loc14_22.2, () [symbolic = %facet_value (constants.%facet_value.8c5)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness imports.%Destroy.impl_witness_table, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness.db6)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %array_type.loc14_22.2, (%Destroy.impl_witness) [symbolic = %Destroy.facet (constants.%Destroy.facet.bb1)]
// CHECK:STDOUT:   %.loc14_3: type = fn_type_with_self_type constants.%Destroy.Op.type, %Destroy.facet [symbolic = %.loc14_3 (constants.%.cc4)]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op.type (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.type.405)]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op: @G.%DestroyT.binding.as_type.as.Destroy.impl.Op.type (%DestroyT.binding.as_type.as.Destroy.impl.Op.type.405) = struct_value () [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.a68)]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value) [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.695)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %k.patt: @G.%pattern_type (%pattern_type.e6c) = ref_binding_pattern k [concrete]
// CHECK:STDOUT:       %k.var_patt: @G.%pattern_type (%pattern_type.e6c) = var_pattern %k.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %k.var: ref @G.%array_type.loc14_22.2 (%array_type.58c) = var %k.var_patt
// CHECK:STDOUT:     %.loc14_22: type = splice_block %array_type.loc14_22.1 [symbolic = %array_type.loc14_22.2 (constants.%array_type.58c)] {
// CHECK:STDOUT:       %int_32.loc14: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc14: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %N.ref: %i32 = name_ref N, %N.loc12_6.2 [symbolic = %N.loc12_6.1 (constants.%N.fe8)]
// CHECK:STDOUT:       %impl.elem0.loc14_21: %.63b = impl_witness_access constants.%ImplicitAs.impl_witness.972, element0 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.047]
// CHECK:STDOUT:       %bound_method.loc14_21.1: <bound method> = bound_method %N.ref, %impl.elem0.loc14_21 [symbolic = %Int.as.ImplicitAs.impl.Convert.bound (constants.%Int.as.ImplicitAs.impl.Convert.bound)]
// CHECK:STDOUT:       %specific_fn.loc14_21: <specific function> = specific_function %impl.elem0.loc14_21, @Int.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Int.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:       %bound_method.loc14_21.2: <bound method> = bound_method %N.ref, %specific_fn.loc14_21 [symbolic = %bound_method.loc14_21.3 (constants.%bound_method)]
// CHECK:STDOUT:       %Int.as.ImplicitAs.impl.Convert.call.loc14_21.1: init Core.IntLiteral = call %bound_method.loc14_21.2(%N.ref) [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc14_21.2 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:       %.loc14_21.1: Core.IntLiteral = value_of_initializer %Int.as.ImplicitAs.impl.Convert.call.loc14_21.1 [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc14_21.2 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:       %.loc14_21.2: Core.IntLiteral = converted %N.ref, %.loc14_21.1 [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc14_21.2 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:       %array_type.loc14_22.1: type = array_type %.loc14_21.2, %i32.loc14 [symbolic = %array_type.loc14_22.2 (constants.%array_type.58c)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %k: ref @G.%array_type.loc14_22.2 (%array_type.58c) = ref_binding k, %k.var
// CHECK:STDOUT:     %impl.elem0.loc14_3: @G.%.loc14_3 (%.cc4) = impl_witness_access constants.%Destroy.impl_witness.db6, element0 [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.a68)]
// CHECK:STDOUT:     %bound_method.loc14_3.1: <bound method> = bound_method %k.var, %impl.elem0.loc14_3
// CHECK:STDOUT:     %specific_fn.loc14_3: <specific function> = specific_function %impl.elem0.loc14_3, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.8c5) [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.695)]
// CHECK:STDOUT:     %bound_method.loc14_3.2: <bound method> = bound_method %k.var, %specific_fn.loc14_3
// CHECK:STDOUT:     %DestroyT.binding.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc14_3.2(%k.var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc4_6.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%N.fe8) {
// CHECK:STDOUT:   %N.loc12_6.1 => constants.%N.fe8
// CHECK:STDOUT: }
// CHECK:STDOUT:

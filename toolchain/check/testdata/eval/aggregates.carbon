// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
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
// CHECK:STDERR: fail_todo_array_temporary.carbon:[[@LINE+2]]:53-86: error: array bound is not a constant [InvalidArrayExpr]
// CHECK:STDERR:
var array_index: array(i32, 1) = (0,) as array(i32, ((5, 7, 1, 9) as array(i32, 4))[2]);
//@dump-sem-ir-end

// --- symbolic.carbon
library "[[@TEST_NAME]]";

// Check that we propagate the `symbolic` tag through evaluations.
fn F(generic T: Core.Destroy) {
  //@dump-sem-ir-begin
  var unused u: (T*, const T);
  var unused v: {.a: T};
  var unused w: array(T, 5);
  //@dump-sem-ir-end
}

fn G(generic N: i32) {
  //@dump-sem-ir-begin
  var unused k: array(i32, N);
  //@dump-sem-ir-end
}


// CHECK:STDOUT: --- basics.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N.fe9: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.693: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.be9: %tuple.type.693 = tuple_value (%i32, %i32) [concrete]
// CHECK:STDOUT:   %tuple.type.35f: type = tuple_type (%i32, %i32) [concrete]
// CHECK:STDOUT:   %pattern_type.354: type = pattern_type %tuple.type.35f [concrete]
// CHECK:STDOUT:   %tuple_copy.patt: %pattern_type.354 = ref_binding_pattern tuple_copy [concrete]
// CHECK:STDOUT:   %tuple_copy.var_patt: %pattern_type.354 = var_pattern %tuple_copy.patt [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.type.f94: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.ad8: %tuple.type.f94 = tuple_value (%int_1.5b8, %int_2.ecc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6b0: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.d73: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f53, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.6b0 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.d73) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.58f: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.33a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.58f, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.282: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.3bd: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.68a: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.8a6: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %bound_method.a55: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.2e0: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.380: %tuple.type.35f = tuple_value (%int_1.68a, %int_2.2e0) [concrete]
// CHECK:STDOUT:   %tuple.elem0: ref %i32 = tuple_access file.%tuple_copy.var, element0 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.a1d: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.3c1: %Int.as.Copy.impl.Op.type.a1d = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.8af: <witness> = impl_witness imports.%Copy.impl_witness_table.f44, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.648: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.cae: %Int.as.Copy.impl.Op.type.648 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.8af) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.f76: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.2b9: type = fn_type_with_self_type %Copy.WithSelf.Op.type.f76, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound.c26: <bound method> = bound_method %int_1.68a, %Int.as.Copy.impl.Op.cae [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.cae, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.a13: <bound method> = bound_method %int_1.68a, %Int.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %tuple.elem1: ref %i32 = tuple_access file.%tuple_copy.var, element1 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound.f7d: <bound method> = bound_method %int_2.2e0, %Int.as.Copy.impl.Op.cae [concrete]
// CHECK:STDOUT:   %bound_method.3ba: <bound method> = bound_method %int_2.2e0, %Int.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %struct_type.a.b.c: type = struct_type {.a: %i32, .b: %i32, .c: %i32} [concrete]
// CHECK:STDOUT:   %pattern_type.9f0: type = pattern_type %struct_type.a.b.c [concrete]
// CHECK:STDOUT:   %struct_copy.patt: %pattern_type.9f0 = ref_binding_pattern struct_copy [concrete]
// CHECK:STDOUT:   %struct_copy.var_patt: %pattern_type.9f0 = var_pattern %struct_copy.patt [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %struct_type.c.b.a: type = struct_type {.c: Core.IntLiteral, .b: Core.IntLiteral, .a: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct.c75: %struct_type.c.b.a = struct_value (%int_3.1ba, %int_2.ecc, %int_1.5b8) [concrete]
// CHECK:STDOUT:   %struct_type.b.a.c: type = struct_type {.b: %i32, .a: %i32, .c: %i32} [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.b3c: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %bound_method.fb0: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.9f3: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %struct.a6e: %struct_type.b.a.c = struct_value (%int_2.2e0, %int_1.68a, %int_3.9f3) [concrete]
// CHECK:STDOUT:   %.cde: ref %i32 = struct_access file.%struct_copy.var, element1 [concrete]
// CHECK:STDOUT:   %.21d: ref %i32 = struct_access file.%struct_copy.var, element0 [concrete]
// CHECK:STDOUT:   %.dfa: ref %i32 = struct_access file.%struct_copy.var, element2 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound.a06: <bound method> = bound_method %int_3.9f3, %Int.as.Copy.impl.Op.cae [concrete]
// CHECK:STDOUT:   %bound_method.c89: <bound method> = bound_method %int_3.9f3, %Int.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %struct.d64: %struct_type.a.b.c = struct_value (%int_1.68a, %int_2.2e0, %int_3.9f3) [concrete]
// CHECK:STDOUT:   %Array.00c: type = class_type @Array, @Array(%i32, %int_1.5b8) [concrete]
// CHECK:STDOUT:   %pattern_type.dc1: type = pattern_type %Array.00c [concrete]
// CHECK:STDOUT:   %tuple_index.patt: %pattern_type.dc1 = ref_binding_pattern tuple_index [concrete]
// CHECK:STDOUT:   %tuple_index.var_patt: %pattern_type.dc1 = var_pattern %tuple_index.patt [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %tuple.type.985: type = tuple_type (Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.4f2: %tuple.type.985 = tuple_value (%int_0.5c6) [concrete]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %int_7: Core.IntLiteral = int_value 7 [concrete]
// CHECK:STDOUT:   %int_9: Core.IntLiteral = int_value 9 [concrete]
// CHECK:STDOUT:   %tuple.type.d46: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.869: %tuple.type.d46 = tuple_value (%int_5, %int_7, %int_1.5b8, %int_9) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.762: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %bound_method.036: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.753: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %array: %Array.00c = tuple_value (%int_0.753) [concrete]
// CHECK:STDOUT:   %struct_access.patt: %pattern_type.dc1 = ref_binding_pattern struct_access [concrete]
// CHECK:STDOUT:   %struct_access.var_patt: %pattern_type.dc1 = var_pattern %struct_access.patt [concrete]
// CHECK:STDOUT:   %struct_type.a.b: type = struct_type {.a: Core.IntLiteral, .b: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct.a81: %struct_type.a.b = struct_value (%int_3.1ba, %int_1.5b8) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.e47: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f53 = impl_witness_table (%Core.import_ref.e47), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.72f: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.a1d) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.3c1)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f44 = impl_witness_table (%Core.import_ref.72f), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %tuple_copy.var: ref %tuple.type.35f = var_storage %tuple_copy.var_patt [concrete]
// CHECK:STDOUT:   %.loc4_26.1: type = splice_block %.loc4_26.3 [concrete = constants.%tuple.type.35f] {
// CHECK:STDOUT:     %i32.loc4_18: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc4_23: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc4_26.2: %tuple.type.693 = tuple_literal (%i32.loc4_18, %i32.loc4_23) [concrete = constants.%tuple.be9]
// CHECK:STDOUT:     %.loc4_26.3: type = converted %.loc4_26.2, constants.%tuple.type.35f [concrete = constants.%tuple.type.35f]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %tuple_copy: ref %tuple.type.35f = wrapper_binding tuple_copy, %tuple_copy.var [concrete = %tuple_copy.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %tuple_copy.patt: %pattern_type.354 = ref_binding_pattern tuple_copy [concrete = constants.%tuple_copy.patt]
// CHECK:STDOUT:     %tuple_copy.var_patt: %pattern_type.354 = var_pattern %tuple_copy.patt [concrete = constants.%tuple_copy.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %struct_copy.var: ref %struct_type.a.b.c = var_storage %struct_copy.var_patt [concrete]
// CHECK:STDOUT:   %.loc6: type = splice_block %struct_type.a.b.c [concrete = constants.%struct_type.a.b.c] {
// CHECK:STDOUT:     %i32.loc6_23: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc6_32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc6_41: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.a.b.c: type = struct_type {.a: %i32, .b: %i32, .c: %i32} [concrete = constants.%struct_type.a.b.c]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %struct_copy: ref %struct_type.a.b.c = wrapper_binding struct_copy, %struct_copy.var [concrete = %struct_copy.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %struct_copy.patt: %pattern_type.9f0 = ref_binding_pattern struct_copy [concrete = constants.%struct_copy.patt]
// CHECK:STDOUT:     %struct_copy.var_patt: %pattern_type.9f0 = var_pattern %struct_copy.patt [concrete = constants.%struct_copy.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %tuple_index.var: ref %Array.00c = var_storage %tuple_index.var_patt [concrete]
// CHECK:STDOUT:   %.loc8: type = splice_block %Array.loc8 [concrete = constants.%Array.00c] {
// CHECK:STDOUT:     %i32.loc8: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %Array.loc8: type = class_type @Array, @Array(constants.%i32, constants.%int_1.5b8) [concrete = constants.%Array.00c]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %tuple_index: ref %Array.00c = wrapper_binding tuple_index, %tuple_index.var [concrete = %tuple_index.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %tuple_index.patt: %pattern_type.dc1 = ref_binding_pattern tuple_index [concrete = constants.%tuple_index.patt]
// CHECK:STDOUT:     %tuple_index.var_patt: %pattern_type.dc1 = var_pattern %tuple_index.patt [concrete = constants.%tuple_index.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %struct_access.var: ref %Array.00c = var_storage %struct_access.var_patt [concrete]
// CHECK:STDOUT:   %.loc10: type = splice_block %Array.loc10 [concrete = constants.%Array.00c] {
// CHECK:STDOUT:     %i32.loc10: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %int_1.loc10: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %Array.loc10: type = class_type @Array, @Array(constants.%i32, constants.%int_1.5b8) [concrete = constants.%Array.00c]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %struct_access: ref %Array.00c = wrapper_binding struct_access, %struct_access.var [concrete = %struct_access.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %struct_access.patt: %pattern_type.dc1 = ref_binding_pattern struct_access [concrete = constants.%struct_access.patt]
// CHECK:STDOUT:     %struct_access.var_patt: %pattern_type.dc1 = var_pattern %struct_access.patt [concrete = constants.%struct_access.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1.loc4: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc4: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc4_35.1: %tuple.type.f94 = tuple_literal (%int_1.loc4, %int_2.loc4) [concrete = constants.%tuple.ad8]
// CHECK:STDOUT:   %i32.loc4_41: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %i32.loc4_46: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc4_49.1: %tuple.type.693 = tuple_literal (%i32.loc4_41, %i32.loc4_46) [concrete = constants.%tuple.be9]
// CHECK:STDOUT:   %.loc4_49.2: type = converted %.loc4_49.1, constants.%tuple.type.35f [concrete = constants.%tuple.type.35f]
// CHECK:STDOUT:   %impl.elem0.loc4_35.1: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc4_35.1: <bound method> = bound_method %int_1.loc4, %impl.elem0.loc4_35.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.282]
// CHECK:STDOUT:   %specific_fn.loc4_35.1: <specific function> = specific_function %impl.elem0.loc4_35.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc4_35.2: <bound method> = bound_method %int_1.loc4, %specific_fn.loc4_35.1 [concrete = constants.%bound_method.3bd]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc4_35.1: init %i32 = call %bound_method.loc4_35.2(%int_1.loc4) [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc4_35.2: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc4_35.1 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc4_35.3: %i32 = converted %int_1.loc4, %.loc4_35.2 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %impl.elem0.loc4_35.2: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc4_35.3: <bound method> = bound_method %int_2.loc4, %impl.elem0.loc4_35.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.8a6]
// CHECK:STDOUT:   %specific_fn.loc4_35.2: <specific function> = specific_function %impl.elem0.loc4_35.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc4_35.4: <bound method> = bound_method %int_2.loc4, %specific_fn.loc4_35.2 [concrete = constants.%bound_method.a55]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc4_35.2: init %i32 = call %bound_method.loc4_35.4(%int_2.loc4) [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %.loc4_35.4: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc4_35.2 [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %.loc4_35.5: %i32 = converted %int_2.loc4, %.loc4_35.4 [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %tuple.loc4: %tuple.type.35f = tuple_value (%.loc4_35.3, %.loc4_35.5) [concrete = constants.%tuple.380]
// CHECK:STDOUT:   %.loc4_37.1: %tuple.type.35f = converted %.loc4_35.1, %tuple.loc4 [concrete = constants.%tuple.380]
// CHECK:STDOUT:   %tuple.elem0.loc4_37.1: %i32 = tuple_access %.loc4_37.1, element0 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %impl.elem0.loc4_37.1: %.2b9 = impl_witness_access constants.%Copy.impl_witness.8af, element0 [concrete = constants.%Int.as.Copy.impl.Op.cae]
// CHECK:STDOUT:   %bound_method.loc4_37.1: <bound method> = bound_method %tuple.elem0.loc4_37.1, %impl.elem0.loc4_37.1 [concrete = constants.%Int.as.Copy.impl.Op.bound.c26]
// CHECK:STDOUT:   %specific_fn.loc4_37.1: <specific function> = specific_function %impl.elem0.loc4_37.1, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc4_37.2: <bound method> = bound_method %tuple.elem0.loc4_37.1, %specific_fn.loc4_37.1 [concrete = constants.%bound_method.a13]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc4_37.1: init %i32 = call %bound_method.loc4_37.2(%tuple.elem0.loc4_37.1) [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %tuple.elem0.loc4_37.2: ref %i32 = tuple_access file.%tuple_copy.var, element0 [concrete = constants.%tuple.elem0]
// CHECK:STDOUT:   %.loc4_37.2: init %i32 to %tuple.elem0.loc4_37.2 = in_place_init %Int.as.Copy.impl.Op.call.loc4_37.1 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %tuple.elem1.loc4_37.1: %i32 = tuple_access %.loc4_37.1, element1 [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %impl.elem0.loc4_37.2: %.2b9 = impl_witness_access constants.%Copy.impl_witness.8af, element0 [concrete = constants.%Int.as.Copy.impl.Op.cae]
// CHECK:STDOUT:   %bound_method.loc4_37.3: <bound method> = bound_method %tuple.elem1.loc4_37.1, %impl.elem0.loc4_37.2 [concrete = constants.%Int.as.Copy.impl.Op.bound.f7d]
// CHECK:STDOUT:   %specific_fn.loc4_37.2: <specific function> = specific_function %impl.elem0.loc4_37.2, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc4_37.4: <bound method> = bound_method %tuple.elem1.loc4_37.1, %specific_fn.loc4_37.2 [concrete = constants.%bound_method.3ba]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc4_37.2: init %i32 = call %bound_method.loc4_37.4(%tuple.elem1.loc4_37.1) [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %tuple.elem1.loc4_37.2: ref %i32 = tuple_access file.%tuple_copy.var, element1 [concrete = constants.%tuple.elem1]
// CHECK:STDOUT:   %.loc4_37.3: init %i32 to %tuple.elem1.loc4_37.2 = in_place_init %Int.as.Copy.impl.Op.call.loc4_37.2 [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %.loc4_37.4: init %tuple.type.35f to file.%tuple_copy.var = tuple_init (%.loc4_37.2, %.loc4_37.3) [concrete = constants.%tuple.380]
// CHECK:STDOUT:   %.loc4_1: init %tuple.type.35f = converted %.loc4_37.1, %.loc4_37.4 [concrete = constants.%tuple.380]
// CHECK:STDOUT:   assign file.%tuple_copy.var, %.loc4_1
// CHECK:STDOUT:   %int_3.loc6: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %int_2.loc6: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %int_1.loc6: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc6_71.1: %struct_type.c.b.a = struct_literal (%int_3.loc6, %int_2.loc6, %int_1.loc6) [concrete = constants.%struct.c75]
// CHECK:STDOUT:   %i32.loc6_81: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %i32.loc6_90: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %i32.loc6_99: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %struct_type.b.a.c: type = struct_type {.b: %i32, .a: %i32, .c: %i32} [concrete = constants.%struct_type.b.a.c]
// CHECK:STDOUT:   %impl.elem0.loc6_71.1: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc6_71.1: <bound method> = bound_method %int_2.loc6, %impl.elem0.loc6_71.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.8a6]
// CHECK:STDOUT:   %specific_fn.loc6_71.1: <specific function> = specific_function %impl.elem0.loc6_71.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_71.2: <bound method> = bound_method %int_2.loc6, %specific_fn.loc6_71.1 [concrete = constants.%bound_method.a55]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_71.1: init %i32 = call %bound_method.loc6_71.2(%int_2.loc6) [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %.loc6_71.2: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_71.1 [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %.loc6_71.3: %i32 = converted %int_2.loc6, %.loc6_71.2 [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %impl.elem0.loc6_71.2: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc6_71.3: <bound method> = bound_method %int_1.loc6, %impl.elem0.loc6_71.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.282]
// CHECK:STDOUT:   %specific_fn.loc6_71.2: <specific function> = specific_function %impl.elem0.loc6_71.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_71.4: <bound method> = bound_method %int_1.loc6, %specific_fn.loc6_71.2 [concrete = constants.%bound_method.3bd]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_71.2: init %i32 = call %bound_method.loc6_71.4(%int_1.loc6) [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc6_71.4: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_71.2 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc6_71.5: %i32 = converted %int_1.loc6, %.loc6_71.4 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %impl.elem0.loc6_71.3: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc6_71.5: <bound method> = bound_method %int_3.loc6, %impl.elem0.loc6_71.3 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.b3c]
// CHECK:STDOUT:   %specific_fn.loc6_71.3: <specific function> = specific_function %impl.elem0.loc6_71.3, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_71.6: <bound method> = bound_method %int_3.loc6, %specific_fn.loc6_71.3 [concrete = constants.%bound_method.fb0]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_71.3: init %i32 = call %bound_method.loc6_71.6(%int_3.loc6) [concrete = constants.%int_3.9f3]
// CHECK:STDOUT:   %.loc6_71.6: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_71.3 [concrete = constants.%int_3.9f3]
// CHECK:STDOUT:   %.loc6_71.7: %i32 = converted %int_3.loc6, %.loc6_71.6 [concrete = constants.%int_3.9f3]
// CHECK:STDOUT:   %struct.loc6: %struct_type.b.a.c = struct_value (%.loc6_71.3, %.loc6_71.5, %.loc6_71.7) [concrete = constants.%struct.a6e]
// CHECK:STDOUT:   %.loc6_73.1: %struct_type.b.a.c = converted %.loc6_71.1, %struct.loc6 [concrete = constants.%struct.a6e]
// CHECK:STDOUT:   %.loc6_73.2: %i32 = struct_access %.loc6_73.1, element1 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %impl.elem0.loc6_73.1: %.2b9 = impl_witness_access constants.%Copy.impl_witness.8af, element0 [concrete = constants.%Int.as.Copy.impl.Op.cae]
// CHECK:STDOUT:   %bound_method.loc6_73.1: <bound method> = bound_method %.loc6_73.2, %impl.elem0.loc6_73.1 [concrete = constants.%Int.as.Copy.impl.Op.bound.c26]
// CHECK:STDOUT:   %specific_fn.loc6_73.1: <specific function> = specific_function %impl.elem0.loc6_73.1, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_73.2: <bound method> = bound_method %.loc6_73.2, %specific_fn.loc6_73.1 [concrete = constants.%bound_method.a13]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc6_73.1: init %i32 = call %bound_method.loc6_73.2(%.loc6_73.2) [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc6_73.3: ref %i32 = struct_access file.%struct_copy.var, element1 [concrete = constants.%.cde]
// CHECK:STDOUT:   %.loc6_73.4: init %i32 to %.loc6_73.3 = in_place_init %Int.as.Copy.impl.Op.call.loc6_73.1 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc6_73.5: %i32 = struct_access %.loc6_73.1, element0 [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %impl.elem0.loc6_73.2: %.2b9 = impl_witness_access constants.%Copy.impl_witness.8af, element0 [concrete = constants.%Int.as.Copy.impl.Op.cae]
// CHECK:STDOUT:   %bound_method.loc6_73.3: <bound method> = bound_method %.loc6_73.5, %impl.elem0.loc6_73.2 [concrete = constants.%Int.as.Copy.impl.Op.bound.f7d]
// CHECK:STDOUT:   %specific_fn.loc6_73.2: <specific function> = specific_function %impl.elem0.loc6_73.2, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_73.4: <bound method> = bound_method %.loc6_73.5, %specific_fn.loc6_73.2 [concrete = constants.%bound_method.3ba]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc6_73.2: init %i32 = call %bound_method.loc6_73.4(%.loc6_73.5) [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %.loc6_73.6: ref %i32 = struct_access file.%struct_copy.var, element0 [concrete = constants.%.21d]
// CHECK:STDOUT:   %.loc6_73.7: init %i32 to %.loc6_73.6 = in_place_init %Int.as.Copy.impl.Op.call.loc6_73.2 [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %.loc6_73.8: %i32 = struct_access %.loc6_73.1, element2 [concrete = constants.%int_3.9f3]
// CHECK:STDOUT:   %impl.elem0.loc6_73.3: %.2b9 = impl_witness_access constants.%Copy.impl_witness.8af, element0 [concrete = constants.%Int.as.Copy.impl.Op.cae]
// CHECK:STDOUT:   %bound_method.loc6_73.5: <bound method> = bound_method %.loc6_73.8, %impl.elem0.loc6_73.3 [concrete = constants.%Int.as.Copy.impl.Op.bound.a06]
// CHECK:STDOUT:   %specific_fn.loc6_73.3: <specific function> = specific_function %impl.elem0.loc6_73.3, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_73.6: <bound method> = bound_method %.loc6_73.8, %specific_fn.loc6_73.3 [concrete = constants.%bound_method.c89]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc6_73.3: init %i32 = call %bound_method.loc6_73.6(%.loc6_73.8) [concrete = constants.%int_3.9f3]
// CHECK:STDOUT:   %.loc6_73.9: ref %i32 = struct_access file.%struct_copy.var, element2 [concrete = constants.%.dfa]
// CHECK:STDOUT:   %.loc6_73.10: init %i32 to %.loc6_73.9 = in_place_init %Int.as.Copy.impl.Op.call.loc6_73.3 [concrete = constants.%int_3.9f3]
// CHECK:STDOUT:   %.loc6_73.11: init %struct_type.a.b.c to file.%struct_copy.var = struct_init (%.loc6_73.4, %.loc6_73.7, %.loc6_73.10) [concrete = constants.%struct.d64]
// CHECK:STDOUT:   %.loc6_1: init %struct_type.a.b.c = converted %.loc6_73.1, %.loc6_73.11 [concrete = constants.%struct.d64]
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
// CHECK:STDOUT:   %Array.loc8: type = class_type @Array, @Array(constants.%i32, constants.%int_1.5b8) [concrete = constants.%Array.00c]
// CHECK:STDOUT:   %impl.elem0.loc8: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc8_37.1: <bound method> = bound_method %int_0.loc8_35, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.762]
// CHECK:STDOUT:   %specific_fn.loc8: <specific function> = specific_function %impl.elem0.loc8, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_37.2: <bound method> = bound_method %int_0.loc8_35, %specific_fn.loc8 [concrete = constants.%bound_method.036]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8: init %i32 = call %bound_method.loc8_37.2(%int_0.loc8_35) [concrete = constants.%int_0.753]
// CHECK:STDOUT:   %.loc8_37.2: init %i32 = converted %int_0.loc8_35, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8 [concrete = constants.%int_0.753]
// CHECK:STDOUT:   %.loc8_1: ref %Array.00c = splice_block file.%tuple_index.var [concrete = file.%tuple_index.var] {}
// CHECK:STDOUT:   %int_0.loc8_37: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc8_37.3: ref %i32 = array_index %.loc8_1, %int_0.loc8_37
// CHECK:STDOUT:   %.loc8_37.4: init %i32 to %.loc8_37.3 = in_place_init %.loc8_37.2 [concrete = constants.%int_0.753]
// CHECK:STDOUT:   %.loc8_37.5: init %Array.00c to %.loc8_1 = array_init (%.loc8_37.4) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc8_39: init %Array.00c = converted %.loc8_37.1, %.loc8_37.5 [concrete = constants.%array]
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
// CHECK:STDOUT:   %Array.loc10: type = class_type @Array, @Array(constants.%i32, constants.%int_1.5b8) [concrete = constants.%Array.00c]
// CHECK:STDOUT:   %impl.elem0.loc10: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc10_39.1: <bound method> = bound_method %int_0.loc10_37, %impl.elem0.loc10 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.762]
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_39.2: <bound method> = bound_method %int_0.loc10_37, %specific_fn.loc10 [concrete = constants.%bound_method.036]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10: init %i32 = call %bound_method.loc10_39.2(%int_0.loc10_37) [concrete = constants.%int_0.753]
// CHECK:STDOUT:   %.loc10_39.2: init %i32 = converted %int_0.loc10_37, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10 [concrete = constants.%int_0.753]
// CHECK:STDOUT:   %.loc10_1: ref %Array.00c = splice_block file.%struct_access.var [concrete = file.%struct_access.var] {}
// CHECK:STDOUT:   %int_0.loc10_39: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc10_39.3: ref %i32 = array_index %.loc10_1, %int_0.loc10_39
// CHECK:STDOUT:   %.loc10_39.4: init %i32 to %.loc10_39.3 = in_place_init %.loc10_39.2 [concrete = constants.%int_0.753]
// CHECK:STDOUT:   %.loc10_39.5: init %Array.00c to %.loc10_1 = array_init (%.loc10_39.4) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc10_41: init %Array.00c = converted %.loc10_39.1, %.loc10_39.5 [concrete = constants.%array]
// CHECK:STDOUT:   assign file.%struct_access.var, %.loc10_41
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_array_temporary.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Array.00c: type = class_type @Array, @Array(%i32, %int_1.5b8) [concrete]
// CHECK:STDOUT:   %pattern_type.dc1: type = pattern_type %Array.00c [concrete]
// CHECK:STDOUT:   %array_index.patt: %pattern_type.dc1 = ref_binding_pattern array_index [concrete]
// CHECK:STDOUT:   %array_index.var_patt: %pattern_type.dc1 = var_pattern %array_index.patt [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %tuple.type.985: type = tuple_type (Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.4f2: %tuple.type.985 = tuple_value (%int_0) [concrete]
// CHECK:STDOUT:   %int_5.64b: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %int_7.29f: Core.IntLiteral = int_value 7 [concrete]
// CHECK:STDOUT:   %int_9.988: Core.IntLiteral = int_value 9 [concrete]
// CHECK:STDOUT:   %tuple.type.d46: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.869: %tuple.type.d46 = tuple_value (%int_5.64b, %int_7.29f, %int_1.5b8, %int_9.988) [concrete]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %Array.f09: type = class_type @Array, @Array(%i32, %int_4) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6b0: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.d73: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f53, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.6b0 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.d73) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.58f: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.33a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.58f, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.f65: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.ab2: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_5.708: %i32 = int_value 5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5f1: <bound method> = bound_method %int_7.29f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %bound_method.a5e: <bound method> = bound_method %int_7.29f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_7.a13: %i32 = int_value 7 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.282: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %bound_method.3bd: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.68a: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cf2: <bound method> = bound_method %int_9.988, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %bound_method.015: <bound method> = bound_method %int_9.988, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_9.607: %i32 = int_value 9 [concrete]
// CHECK:STDOUT:   %array: %Array.f09 = tuple_value (%int_5.708, %int_7.a13, %int_1.68a, %int_9.607) [concrete]
// CHECK:STDOUT:   %.4ae: ref %Array.f09 = temporary invalid, %array [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.8a6: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %bound_method.a55: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.2e0: %i32 = int_value 2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.e47: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f53 = impl_witness_table (%Core.import_ref.e47), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %array_index.var: ref %Array.00c = var_storage %array_index.var_patt [concrete]
// CHECK:STDOUT:   %.loc7: type = splice_block %Array [concrete = constants.%Array.00c] {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %Array: type = class_type @Array, @Array(constants.%i32, constants.%int_1.5b8) [concrete = constants.%Array.00c]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %array_index: ref %Array.00c = wrapper_binding array_index, %array_index.var [concrete = %array_index.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %array_index.patt: %pattern_type.dc1 = ref_binding_pattern array_index [concrete = constants.%array_index.patt]
// CHECK:STDOUT:     %array_index.var_patt: %pattern_type.dc1 = var_pattern %array_index.patt [concrete = constants.%array_index.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_0.loc7_35: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc7_37: %tuple.type.985 = tuple_literal (%int_0.loc7_35) [concrete = constants.%tuple.4f2]
// CHECK:STDOUT:   %i32.loc7_48: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5.64b]
// CHECK:STDOUT:   %int_7: Core.IntLiteral = int_value 7 [concrete = constants.%int_7.29f]
// CHECK:STDOUT:   %int_1.loc7_61: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_9: Core.IntLiteral = int_value 9 [concrete = constants.%int_9.988]
// CHECK:STDOUT:   %.loc7_65.1: %tuple.type.d46 = tuple_literal (%int_5, %int_7, %int_1.loc7_61, %int_9) [concrete = constants.%tuple.869]
// CHECK:STDOUT:   %i32.loc7_76: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4]
// CHECK:STDOUT:   %Array: type = class_type @Array, @Array(constants.%i32, constants.%int_4) [concrete = constants.%Array.f09]
// CHECK:STDOUT:   %impl.elem0.loc7_65.1: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc7_65.1: <bound method> = bound_method %int_5, %impl.elem0.loc7_65.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.f65]
// CHECK:STDOUT:   %specific_fn.loc7_65.1: <specific function> = specific_function %impl.elem0.loc7_65.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_65.2: <bound method> = bound_method %int_5, %specific_fn.loc7_65.1 [concrete = constants.%bound_method.ab2]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7_65.1: init %i32 = call %bound_method.loc7_65.2(%int_5) [concrete = constants.%int_5.708]
// CHECK:STDOUT:   %.loc7_65.2: init %i32 = converted %int_5, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7_65.1 [concrete = constants.%int_5.708]
// CHECK:STDOUT:   %.loc7_65.3: ref %Array.f09 = temporary_storage
// CHECK:STDOUT:   %int_0.loc7_65: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc7_65.4: ref %i32 = array_index %.loc7_65.3, %int_0.loc7_65
// CHECK:STDOUT:   %.loc7_65.5: init %i32 to %.loc7_65.4 = in_place_init %.loc7_65.2 [concrete = constants.%int_5.708]
// CHECK:STDOUT:   %impl.elem0.loc7_65.2: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc7_65.3: <bound method> = bound_method %int_7, %impl.elem0.loc7_65.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5f1]
// CHECK:STDOUT:   %specific_fn.loc7_65.2: <specific function> = specific_function %impl.elem0.loc7_65.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_65.4: <bound method> = bound_method %int_7, %specific_fn.loc7_65.2 [concrete = constants.%bound_method.a5e]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7_65.2: init %i32 = call %bound_method.loc7_65.4(%int_7) [concrete = constants.%int_7.a13]
// CHECK:STDOUT:   %.loc7_65.6: init %i32 = converted %int_7, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7_65.2 [concrete = constants.%int_7.a13]
// CHECK:STDOUT:   %int_1.loc7_65: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc7_65.7: ref %i32 = array_index %.loc7_65.3, %int_1.loc7_65
// CHECK:STDOUT:   %.loc7_65.8: init %i32 to %.loc7_65.7 = in_place_init %.loc7_65.6 [concrete = constants.%int_7.a13]
// CHECK:STDOUT:   %impl.elem0.loc7_65.3: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc7_65.5: <bound method> = bound_method %int_1.loc7_61, %impl.elem0.loc7_65.3 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.282]
// CHECK:STDOUT:   %specific_fn.loc7_65.3: <specific function> = specific_function %impl.elem0.loc7_65.3, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_65.6: <bound method> = bound_method %int_1.loc7_61, %specific_fn.loc7_65.3 [concrete = constants.%bound_method.3bd]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7_65.3: init %i32 = call %bound_method.loc7_65.6(%int_1.loc7_61) [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc7_65.9: init %i32 = converted %int_1.loc7_61, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7_65.3 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %int_2.loc7_65: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc7_65.10: ref %i32 = array_index %.loc7_65.3, %int_2.loc7_65
// CHECK:STDOUT:   %.loc7_65.11: init %i32 to %.loc7_65.10 = in_place_init %.loc7_65.9 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %impl.elem0.loc7_65.4: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc7_65.7: <bound method> = bound_method %int_9, %impl.elem0.loc7_65.4 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cf2]
// CHECK:STDOUT:   %specific_fn.loc7_65.4: <specific function> = specific_function %impl.elem0.loc7_65.4, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_65.8: <bound method> = bound_method %int_9, %specific_fn.loc7_65.4 [concrete = constants.%bound_method.015]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7_65.4: init %i32 = call %bound_method.loc7_65.8(%int_9) [concrete = constants.%int_9.607]
// CHECK:STDOUT:   %.loc7_65.12: init %i32 = converted %int_9, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7_65.4 [concrete = constants.%int_9.607]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:   %.loc7_65.13: ref %i32 = array_index %.loc7_65.3, %int_3
// CHECK:STDOUT:   %.loc7_65.14: init %i32 to %.loc7_65.13 = in_place_init %.loc7_65.12 [concrete = constants.%int_9.607]
// CHECK:STDOUT:   %.loc7_65.15: init %Array.f09 to %.loc7_65.3 = array_init (%.loc7_65.5, %.loc7_65.8, %.loc7_65.11, %.loc7_65.14) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc7_67.1: init %Array.f09 = converted %.loc7_65.1, %.loc7_65.15 [concrete = constants.%array]
// CHECK:STDOUT:   %int_2.loc7_85: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc7_67.2: ref %Array.f09 = temporary %.loc7_65.3, %.loc7_67.1 [concrete = constants.%.4ae]
// CHECK:STDOUT:   %impl.elem0.loc7_85: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc7_85.1: <bound method> = bound_method %int_2.loc7_85, %impl.elem0.loc7_85 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.8a6]
// CHECK:STDOUT:   %specific_fn.loc7_85: <specific function> = specific_function %impl.elem0.loc7_85, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_85.2: <bound method> = bound_method %int_2.loc7_85, %specific_fn.loc7_85 [concrete = constants.%bound_method.a55]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7_85: init %i32 = call %bound_method.loc7_85.2(%int_2.loc7_85) [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %.loc7_85.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7_85 [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %.loc7_85.2: %i32 = converted %int_2.loc7_85, %.loc7_85.1 [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %.loc7_86.1: ref %i32 = array_index %.loc7_67.2, %.loc7_85.2
// CHECK:STDOUT:   %.loc7_86.2: %i32 = acquire_value %.loc7_86.1
// CHECK:STDOUT:   assign file.%array_index.var, <error>
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- symbolic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.429: type = pattern_type %Destroy.type [concrete]
// CHECK:STDOUT:   %T.patt.351: %pattern_type.429 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.0e7: %Destroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %T.as_type.700: type = facet_access_type %T.0e7 [symbolic]
// CHECK:STDOUT:   %ptr.b37: type = ptr_type %T.as_type.700 [symbolic]
// CHECK:STDOUT:   %const: type = const_type %T.as_type.700 [symbolic]
// CHECK:STDOUT:   %tuple.type.693: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.693 = tuple_value (%ptr.b37, %const) [symbolic]
// CHECK:STDOUT:   %tuple.type.cd5: type = tuple_type (%ptr.b37, %const) [symbolic]
// CHECK:STDOUT:   %require_complete.3b6: <witness> = require_complete_type %tuple.type.cd5 [symbolic]
// CHECK:STDOUT:   %pattern_type.02b: type = pattern_type %tuple.type.cd5 [symbolic]
// CHECK:STDOUT:   %u.patt: %pattern_type.02b = ref_binding_pattern u [symbolic]
// CHECK:STDOUT:   %u.var_patt: %pattern_type.02b = var_pattern %u.patt [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.lookup_impl_witness.2ab: <witness> = lookup_impl_witness %tuple.type.cd5, @DefaultOrUnformed [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.c98: %DefaultOrUnformed.type = facet_value %tuple.type.cd5, (%DefaultOrUnformed.lookup_impl_witness.2ab) [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.WithSelf.Op.type.f14: type = fn_type @DefaultOrUnformed.WithSelf.Op, @DefaultOrUnformed.WithSelf(%DefaultOrUnformed.facet.c98) [symbolic]
// CHECK:STDOUT:   %.ae7: type = fn_type_with_self_type %DefaultOrUnformed.WithSelf.Op.type.f14, %DefaultOrUnformed.facet.c98 [symbolic]
// CHECK:STDOUT:   %impl.elem0.fec: %.ae7 = impl_witness_access %DefaultOrUnformed.lookup_impl_witness.2ab, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.619: <specific function> = specific_impl_function %impl.elem0.fec, @DefaultOrUnformed.WithSelf.Op(%DefaultOrUnformed.facet.c98) [symbolic]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %T.as_type.700} [symbolic]
// CHECK:STDOUT:   %require_complete.f8e: <witness> = require_complete_type %struct_type.a [symbolic]
// CHECK:STDOUT:   %pattern_type.915: type = pattern_type %struct_type.a [symbolic]
// CHECK:STDOUT:   %v.patt: %pattern_type.915 = ref_binding_pattern v [symbolic]
// CHECK:STDOUT:   %v.var_patt: %pattern_type.915 = var_pattern %v.patt [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.lookup_impl_witness.53f: <witness> = lookup_impl_witness %struct_type.a, @DefaultOrUnformed [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.103: %DefaultOrUnformed.type = facet_value %struct_type.a, (%DefaultOrUnformed.lookup_impl_witness.53f) [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.WithSelf.Op.type.a16: type = fn_type @DefaultOrUnformed.WithSelf.Op, @DefaultOrUnformed.WithSelf(%DefaultOrUnformed.facet.103) [symbolic]
// CHECK:STDOUT:   %.9be: type = fn_type_with_self_type %DefaultOrUnformed.WithSelf.Op.type.a16, %DefaultOrUnformed.facet.103 [symbolic]
// CHECK:STDOUT:   %impl.elem0.873: %.9be = impl_witness_access %DefaultOrUnformed.lookup_impl_witness.53f, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.1d1: <specific function> = specific_impl_function %impl.elem0.873, @DefaultOrUnformed.WithSelf.Op(%DefaultOrUnformed.facet.103) [symbolic]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %Array.e06: type = class_type @Array, @Array(%T.as_type.700, %int_5) [symbolic]
// CHECK:STDOUT:   %require_complete.05a: <witness> = require_complete_type %Array.e06 [symbolic]
// CHECK:STDOUT:   %pattern_type.38c: type = pattern_type %Array.e06 [symbolic]
// CHECK:STDOUT:   %w.patt: %pattern_type.38c = ref_binding_pattern w [symbolic]
// CHECK:STDOUT:   %w.var_patt: %pattern_type.38c = var_pattern %w.patt [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.lookup_impl_witness.f03: <witness> = lookup_impl_witness %Array.e06, @DefaultOrUnformed [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.61d: %DefaultOrUnformed.type = facet_value %Array.e06, (%DefaultOrUnformed.lookup_impl_witness.f03) [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.WithSelf.Op.type.25b: type = fn_type @DefaultOrUnformed.WithSelf.Op, @DefaultOrUnformed.WithSelf(%DefaultOrUnformed.facet.61d) [symbolic]
// CHECK:STDOUT:   %.b93: type = fn_type_with_self_type %DefaultOrUnformed.WithSelf.Op.type.25b, %DefaultOrUnformed.facet.61d [symbolic]
// CHECK:STDOUT:   %impl.elem0.1c3: %.b93 = impl_witness_access %DefaultOrUnformed.lookup_impl_witness.f03, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.aef: <specific function> = specific_impl_function %impl.elem0.1c3, @DefaultOrUnformed.WithSelf.Op(%DefaultOrUnformed.facet.61d) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.fc7: <witness> = lookup_impl_witness %Array.e06, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.aae: %Destroy.type = facet_value %Array.e06, (%Destroy.lookup_impl_witness.fc7) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.ddd: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%Destroy.facet.aae) [symbolic]
// CHECK:STDOUT:   %.44d: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.ddd, %Destroy.facet.aae [symbolic]
// CHECK:STDOUT:   %impl.elem2.f1a: %.44d = impl_witness_access %Destroy.lookup_impl_witness.fc7, element2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.5b6: <specific function> = specific_impl_function %impl.elem2.f1a, @Destroy.WithSelf.SelfDestruct(%Destroy.facet.aae) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.7f9: <witness> = lookup_impl_witness %struct_type.a, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.aa3: %Destroy.type = facet_value %struct_type.a, (%Destroy.lookup_impl_witness.7f9) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.af4: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%Destroy.facet.aa3) [symbolic]
// CHECK:STDOUT:   %.8a1: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.af4, %Destroy.facet.aa3 [symbolic]
// CHECK:STDOUT:   %impl.elem2.86f: %.8a1 = impl_witness_access %Destroy.lookup_impl_witness.7f9, element2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.86e: <specific function> = specific_impl_function %impl.elem2.86f, @Destroy.WithSelf.SelfDestruct(%Destroy.facet.aa3) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.d26: <witness> = lookup_impl_witness %tuple.type.cd5, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.7ba: %Destroy.type = facet_value %tuple.type.cd5, (%Destroy.lookup_impl_witness.d26) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.1e6: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%Destroy.facet.7ba) [symbolic]
// CHECK:STDOUT:   %.b75: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.1e6, %Destroy.facet.7ba [symbolic]
// CHECK:STDOUT:   %impl.elem2.dad: %.b75 = impl_witness_access %Destroy.lookup_impl_witness.d26, element2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.d5b: <specific function> = specific_impl_function %impl.elem2.dad, @Destroy.WithSelf.SelfDestruct(%Destroy.facet.7ba) [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %N.patt.d64: %pattern_type.d94 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %N.c34: %i32 = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.5be: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %From: Core.IntLiteral = symbolic_binding From, 0 [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.dfd: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%From) [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.914: %Int.as.ImplicitAs.impl.Convert.type.dfd = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.323: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b7c, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.f39: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.779: %Int.as.ImplicitAs.impl.Convert.type.f39 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.5be = facet_value %i32, (%ImplicitAs.impl_witness.323) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.4c7: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.ce7: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.4c7, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %N.c34, %Int.as.ImplicitAs.impl.Convert.779 [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Int.as.ImplicitAs.impl.Convert.779, @Int.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %N.c34, %Int.as.ImplicitAs.impl.Convert.specific_fn [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method(%N.c34) [symbolic]
// CHECK:STDOUT:   %Array.e80: type = class_type @Array, @Array(%i32, %Int.as.ImplicitAs.impl.Convert.call) [symbolic]
// CHECK:STDOUT:   %require_complete.fcb: <witness> = require_complete_type %Array.e80 [symbolic]
// CHECK:STDOUT:   %pattern_type.fc0: type = pattern_type %Array.e80 [symbolic]
// CHECK:STDOUT:   %k.patt: %pattern_type.fc0 = ref_binding_pattern k [symbolic]
// CHECK:STDOUT:   %k.var_patt: %pattern_type.fc0 = var_pattern %k.patt [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.lookup_impl_witness.b5f: <witness> = lookup_impl_witness %Array.e80, @DefaultOrUnformed [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.48a: %DefaultOrUnformed.type = facet_value %Array.e80, (%DefaultOrUnformed.lookup_impl_witness.b5f) [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.WithSelf.Op.type.8de: type = fn_type @DefaultOrUnformed.WithSelf.Op, @DefaultOrUnformed.WithSelf(%DefaultOrUnformed.facet.48a) [symbolic]
// CHECK:STDOUT:   %.431: type = fn_type_with_self_type %DefaultOrUnformed.WithSelf.Op.type.8de, %DefaultOrUnformed.facet.48a [symbolic]
// CHECK:STDOUT:   %impl.elem0.b7f: %.431 = impl_witness_access %DefaultOrUnformed.lookup_impl_witness.b5f, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.971: <specific function> = specific_impl_function %impl.elem0.b7f, @DefaultOrUnformed.WithSelf.Op(%DefaultOrUnformed.facet.48a) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.27d: <witness> = lookup_impl_witness %Array.e80, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.ca8: %Destroy.type = facet_value %Array.e80, (%Destroy.lookup_impl_witness.27d) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.4da: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%Destroy.facet.ca8) [symbolic]
// CHECK:STDOUT:   %.48f: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.4da, %Destroy.facet.ca8 [symbolic]
// CHECK:STDOUT:   %impl.elem2.af7: %.48f = impl_witness_access %Destroy.lookup_impl_witness.27d, element2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.a3f: <specific function> = specific_impl_function %impl.elem2.af7, @Destroy.WithSelf.SelfDestruct(%Destroy.facet.ca8) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.ed0: @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert.type (%Int.as.ImplicitAs.impl.Convert.type.dfd) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert (constants.%Int.as.ImplicitAs.impl.Convert.914)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b7c = impl_witness_table (%Core.import_ref.ed0), @Int.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc4_15.2: %Destroy.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc6_19.2: type = facet_access_type %T.loc4_15.1 [symbolic = %T.as_type.loc6_19.2 (constants.%T.as_type.700)]
// CHECK:STDOUT:   %ptr.loc6_19.2: type = ptr_type %T.as_type.loc6_19.2 [symbolic = %ptr.loc6_19.2 (constants.%ptr.b37)]
// CHECK:STDOUT:   %const.loc6_22.2: type = const_type %T.as_type.loc6_19.2 [symbolic = %const.loc6_22.2 (constants.%const)]
// CHECK:STDOUT:   %tuple: %tuple.type.693 = tuple_value (%ptr.loc6_19.2, %const.loc6_22.2) [symbolic = %tuple (constants.%tuple)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%ptr.loc6_19.2, %const.loc6_22.2) [symbolic = %tuple.type (constants.%tuple.type.cd5)]
// CHECK:STDOUT:   %require_complete.loc6: <witness> = require_complete_type %tuple.type [symbolic = %require_complete.loc6 (constants.%require_complete.3b6)]
// CHECK:STDOUT:   %pattern_type.loc6: type = pattern_type %tuple.type [symbolic = %pattern_type.loc6 (constants.%pattern_type.02b)]
// CHECK:STDOUT:   %u.patt.loc6_15.2: @F.%pattern_type.loc6 (%pattern_type.02b) = ref_binding_pattern u [symbolic = %u.patt.loc6_15.2 (constants.%u.patt)]
// CHECK:STDOUT:   %u.var_patt.loc6_3.2: @F.%pattern_type.loc6 (%pattern_type.02b) = var_pattern %u.patt.loc6_15.2 [symbolic = %u.var_patt.loc6_3.2 (constants.%u.var_patt)]
// CHECK:STDOUT:   %DefaultOrUnformed.lookup_impl_witness.loc6: <witness> = lookup_impl_witness %tuple.type, @DefaultOrUnformed [symbolic = %DefaultOrUnformed.lookup_impl_witness.loc6 (constants.%DefaultOrUnformed.lookup_impl_witness.2ab)]
// CHECK:STDOUT:   %require_witness_defn.loc6_30: require_witness_defn_type = require_witness_defn %DefaultOrUnformed.lookup_impl_witness.loc6 [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc6_30.2: %DefaultOrUnformed.type = facet_value %tuple.type, (%DefaultOrUnformed.lookup_impl_witness.loc6) [symbolic = %DefaultOrUnformed.facet.loc6_30.2 (constants.%DefaultOrUnformed.facet.c98)]
// CHECK:STDOUT:   %DefaultOrUnformed.WithSelf.Op.type.loc6: type = fn_type @DefaultOrUnformed.WithSelf.Op, @DefaultOrUnformed.WithSelf(%DefaultOrUnformed.facet.loc6_30.2) [symbolic = %DefaultOrUnformed.WithSelf.Op.type.loc6 (constants.%DefaultOrUnformed.WithSelf.Op.type.f14)]
// CHECK:STDOUT:   %.loc6_30.3: type = fn_type_with_self_type %DefaultOrUnformed.WithSelf.Op.type.loc6, %DefaultOrUnformed.facet.loc6_30.2 [symbolic = %.loc6_30.3 (constants.%.ae7)]
// CHECK:STDOUT:   %impl.elem0.loc6_30.2: @F.%.loc6_30.3 (%.ae7) = impl_witness_access %DefaultOrUnformed.lookup_impl_witness.loc6, element0 [symbolic = %impl.elem0.loc6_30.2 (constants.%impl.elem0.fec)]
// CHECK:STDOUT:   %specific_impl_fn.loc6_30.2: <specific function> = specific_impl_function %impl.elem0.loc6_30.2, @DefaultOrUnformed.WithSelf.Op(%DefaultOrUnformed.facet.loc6_30.2) [symbolic = %specific_impl_fn.loc6_30.2 (constants.%specific_impl_fn.619)]
// CHECK:STDOUT:   %struct_type.a.loc7_23.2: type = struct_type {.a: @F.%T.as_type.loc6_19.2 (%T.as_type.700)} [symbolic = %struct_type.a.loc7_23.2 (constants.%struct_type.a)]
// CHECK:STDOUT:   %require_complete.loc7: <witness> = require_complete_type %struct_type.a.loc7_23.2 [symbolic = %require_complete.loc7 (constants.%require_complete.f8e)]
// CHECK:STDOUT:   %pattern_type.loc7: type = pattern_type %struct_type.a.loc7_23.2 [symbolic = %pattern_type.loc7 (constants.%pattern_type.915)]
// CHECK:STDOUT:   %v.patt.loc7_15.2: @F.%pattern_type.loc7 (%pattern_type.915) = ref_binding_pattern v [symbolic = %v.patt.loc7_15.2 (constants.%v.patt)]
// CHECK:STDOUT:   %v.var_patt.loc7_3.2: @F.%pattern_type.loc7 (%pattern_type.915) = var_pattern %v.patt.loc7_15.2 [symbolic = %v.var_patt.loc7_3.2 (constants.%v.var_patt)]
// CHECK:STDOUT:   %DefaultOrUnformed.lookup_impl_witness.loc7: <witness> = lookup_impl_witness %struct_type.a.loc7_23.2, @DefaultOrUnformed [symbolic = %DefaultOrUnformed.lookup_impl_witness.loc7 (constants.%DefaultOrUnformed.lookup_impl_witness.53f)]
// CHECK:STDOUT:   %require_witness_defn.loc7_24: require_witness_defn_type = require_witness_defn %DefaultOrUnformed.lookup_impl_witness.loc7 [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc7_24.2: %DefaultOrUnformed.type = facet_value %struct_type.a.loc7_23.2, (%DefaultOrUnformed.lookup_impl_witness.loc7) [symbolic = %DefaultOrUnformed.facet.loc7_24.2 (constants.%DefaultOrUnformed.facet.103)]
// CHECK:STDOUT:   %DefaultOrUnformed.WithSelf.Op.type.loc7: type = fn_type @DefaultOrUnformed.WithSelf.Op, @DefaultOrUnformed.WithSelf(%DefaultOrUnformed.facet.loc7_24.2) [symbolic = %DefaultOrUnformed.WithSelf.Op.type.loc7 (constants.%DefaultOrUnformed.WithSelf.Op.type.a16)]
// CHECK:STDOUT:   %.loc7_24.3: type = fn_type_with_self_type %DefaultOrUnformed.WithSelf.Op.type.loc7, %DefaultOrUnformed.facet.loc7_24.2 [symbolic = %.loc7_24.3 (constants.%.9be)]
// CHECK:STDOUT:   %impl.elem0.loc7_24.2: @F.%.loc7_24.3 (%.9be) = impl_witness_access %DefaultOrUnformed.lookup_impl_witness.loc7, element0 [symbolic = %impl.elem0.loc7_24.2 (constants.%impl.elem0.873)]
// CHECK:STDOUT:   %specific_impl_fn.loc7_24.2: <specific function> = specific_impl_function %impl.elem0.loc7_24.2, @DefaultOrUnformed.WithSelf.Op(%DefaultOrUnformed.facet.loc7_24.2) [symbolic = %specific_impl_fn.loc7_24.2 (constants.%specific_impl_fn.1d1)]
// CHECK:STDOUT:   %Array.loc8_27.2: type = class_type @Array, @Array(%T.as_type.loc6_19.2, constants.%int_5) [symbolic = %Array.loc8_27.2 (constants.%Array.e06)]
// CHECK:STDOUT:   %require_complete.loc8: <witness> = require_complete_type %Array.loc8_27.2 [symbolic = %require_complete.loc8 (constants.%require_complete.05a)]
// CHECK:STDOUT:   %pattern_type.loc8: type = pattern_type %Array.loc8_27.2 [symbolic = %pattern_type.loc8 (constants.%pattern_type.38c)]
// CHECK:STDOUT:   %w.patt.loc8_15.2: @F.%pattern_type.loc8 (%pattern_type.38c) = ref_binding_pattern w [symbolic = %w.patt.loc8_15.2 (constants.%w.patt)]
// CHECK:STDOUT:   %w.var_patt.loc8_3.2: @F.%pattern_type.loc8 (%pattern_type.38c) = var_pattern %w.patt.loc8_15.2 [symbolic = %w.var_patt.loc8_3.2 (constants.%w.var_patt)]
// CHECK:STDOUT:   %DefaultOrUnformed.lookup_impl_witness.loc8: <witness> = lookup_impl_witness %Array.loc8_27.2, @DefaultOrUnformed [symbolic = %DefaultOrUnformed.lookup_impl_witness.loc8 (constants.%DefaultOrUnformed.lookup_impl_witness.f03)]
// CHECK:STDOUT:   %require_witness_defn.loc8_28: require_witness_defn_type = require_witness_defn %DefaultOrUnformed.lookup_impl_witness.loc8 [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc8_28.2: %DefaultOrUnformed.type = facet_value %Array.loc8_27.2, (%DefaultOrUnformed.lookup_impl_witness.loc8) [symbolic = %DefaultOrUnformed.facet.loc8_28.2 (constants.%DefaultOrUnformed.facet.61d)]
// CHECK:STDOUT:   %DefaultOrUnformed.WithSelf.Op.type.loc8: type = fn_type @DefaultOrUnformed.WithSelf.Op, @DefaultOrUnformed.WithSelf(%DefaultOrUnformed.facet.loc8_28.2) [symbolic = %DefaultOrUnformed.WithSelf.Op.type.loc8 (constants.%DefaultOrUnformed.WithSelf.Op.type.25b)]
// CHECK:STDOUT:   %.loc8_28.3: type = fn_type_with_self_type %DefaultOrUnformed.WithSelf.Op.type.loc8, %DefaultOrUnformed.facet.loc8_28.2 [symbolic = %.loc8_28.3 (constants.%.b93)]
// CHECK:STDOUT:   %impl.elem0.loc8_28.2: @F.%.loc8_28.3 (%.b93) = impl_witness_access %DefaultOrUnformed.lookup_impl_witness.loc8, element0 [symbolic = %impl.elem0.loc8_28.2 (constants.%impl.elem0.1c3)]
// CHECK:STDOUT:   %specific_impl_fn.loc8_28.2: <specific function> = specific_impl_function %impl.elem0.loc8_28.2, @DefaultOrUnformed.WithSelf.Op(%DefaultOrUnformed.facet.loc8_28.2) [symbolic = %specific_impl_fn.loc8_28.2 (constants.%specific_impl_fn.aef)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc8: <witness> = lookup_impl_witness %Array.loc8_27.2, @Destroy [symbolic = %Destroy.lookup_impl_witness.loc8 (constants.%Destroy.lookup_impl_witness.fc7)]
// CHECK:STDOUT:   %require_witness_defn.loc8_3: require_witness_defn_type = require_witness_defn %Destroy.lookup_impl_witness.loc8 [symbolic]
// CHECK:STDOUT:   %Destroy.facet.loc8_3.3: %Destroy.type = facet_value %Array.loc8_27.2, (%Destroy.lookup_impl_witness.loc8) [symbolic = %Destroy.facet.loc8_3.3 (constants.%Destroy.facet.aae)]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.loc8: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%Destroy.facet.loc8_3.3) [symbolic = %Destroy.WithSelf.SelfDestruct.type.loc8 (constants.%Destroy.WithSelf.SelfDestruct.type.ddd)]
// CHECK:STDOUT:   %.loc8_3.4: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.loc8, %Destroy.facet.loc8_3.3 [symbolic = %.loc8_3.4 (constants.%.44d)]
// CHECK:STDOUT:   %impl.elem2.loc8_3.2: @F.%.loc8_3.4 (%.44d) = impl_witness_access %Destroy.lookup_impl_witness.loc8, element2 [symbolic = %impl.elem2.loc8_3.2 (constants.%impl.elem2.f1a)]
// CHECK:STDOUT:   %specific_impl_fn.loc8_3.2: <specific function> = specific_impl_function %impl.elem2.loc8_3.2, @Destroy.WithSelf.SelfDestruct(%Destroy.facet.loc8_3.3) [symbolic = %specific_impl_fn.loc8_3.2 (constants.%specific_impl_fn.5b6)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc7: <witness> = lookup_impl_witness %struct_type.a.loc7_23.2, @Destroy [symbolic = %Destroy.lookup_impl_witness.loc7 (constants.%Destroy.lookup_impl_witness.7f9)]
// CHECK:STDOUT:   %require_witness_defn.loc7_3: require_witness_defn_type = require_witness_defn %Destroy.lookup_impl_witness.loc7 [symbolic]
// CHECK:STDOUT:   %Destroy.facet.loc7_3.3: %Destroy.type = facet_value %struct_type.a.loc7_23.2, (%Destroy.lookup_impl_witness.loc7) [symbolic = %Destroy.facet.loc7_3.3 (constants.%Destroy.facet.aa3)]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.loc7: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%Destroy.facet.loc7_3.3) [symbolic = %Destroy.WithSelf.SelfDestruct.type.loc7 (constants.%Destroy.WithSelf.SelfDestruct.type.af4)]
// CHECK:STDOUT:   %.loc7_3.3: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.loc7, %Destroy.facet.loc7_3.3 [symbolic = %.loc7_3.3 (constants.%.8a1)]
// CHECK:STDOUT:   %impl.elem2.loc7_3.2: @F.%.loc7_3.3 (%.8a1) = impl_witness_access %Destroy.lookup_impl_witness.loc7, element2 [symbolic = %impl.elem2.loc7_3.2 (constants.%impl.elem2.86f)]
// CHECK:STDOUT:   %specific_impl_fn.loc7_3.2: <specific function> = specific_impl_function %impl.elem2.loc7_3.2, @Destroy.WithSelf.SelfDestruct(%Destroy.facet.loc7_3.3) [symbolic = %specific_impl_fn.loc7_3.2 (constants.%specific_impl_fn.86e)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc6: <witness> = lookup_impl_witness %tuple.type, @Destroy [symbolic = %Destroy.lookup_impl_witness.loc6 (constants.%Destroy.lookup_impl_witness.d26)]
// CHECK:STDOUT:   %require_witness_defn.loc6_3: require_witness_defn_type = require_witness_defn %Destroy.lookup_impl_witness.loc6 [symbolic]
// CHECK:STDOUT:   %Destroy.facet.loc6_3.3: %Destroy.type = facet_value %tuple.type, (%Destroy.lookup_impl_witness.loc6) [symbolic = %Destroy.facet.loc6_3.3 (constants.%Destroy.facet.7ba)]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.loc6: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%Destroy.facet.loc6_3.3) [symbolic = %Destroy.WithSelf.SelfDestruct.type.loc6 (constants.%Destroy.WithSelf.SelfDestruct.type.1e6)]
// CHECK:STDOUT:   %.loc6_3.4: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.loc6, %Destroy.facet.loc6_3.3 [symbolic = %.loc6_3.4 (constants.%.b75)]
// CHECK:STDOUT:   %impl.elem2.loc6_3.2: @F.%.loc6_3.4 (%.b75) = impl_witness_access %Destroy.lookup_impl_witness.loc6, element2 [symbolic = %impl.elem2.loc6_3.2 (constants.%impl.elem2.dad)]
// CHECK:STDOUT:   %specific_impl_fn.loc6_3.2: <specific function> = specific_impl_function %impl.elem2.loc6_3.2, @Destroy.WithSelf.SelfDestruct(%Destroy.facet.loc6_3.3) [symbolic = %specific_impl_fn.loc6_3.2 (constants.%specific_impl_fn.d5b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %u.var: ref @F.%tuple.type (%tuple.type.cd5) = var_storage %u.var_patt.loc6_3.1
// CHECK:STDOUT:     %DefaultOrUnformed.facet.loc6_30.1: %DefaultOrUnformed.type = facet_value constants.%tuple.type.cd5, (constants.%DefaultOrUnformed.lookup_impl_witness.2ab) [symbolic = %DefaultOrUnformed.facet.loc6_30.2 (constants.%DefaultOrUnformed.facet.c98)]
// CHECK:STDOUT:     %.loc6_30.1: %DefaultOrUnformed.type = converted constants.%tuple.type.cd5, %DefaultOrUnformed.facet.loc6_30.1 [symbolic = %DefaultOrUnformed.facet.loc6_30.2 (constants.%DefaultOrUnformed.facet.c98)]
// CHECK:STDOUT:     %as_type.loc6: type = facet_access_type %.loc6_30.1 [symbolic = %tuple.type (constants.%tuple.type.cd5)]
// CHECK:STDOUT:     %.loc6_30.2: type = converted %.loc6_30.1, %as_type.loc6 [symbolic = %tuple.type (constants.%tuple.type.cd5)]
// CHECK:STDOUT:     %impl.elem0.loc6_30.1: @F.%.loc6_30.3 (%.ae7) = impl_witness_access constants.%DefaultOrUnformed.lookup_impl_witness.2ab, element0 [symbolic = %impl.elem0.loc6_30.2 (constants.%impl.elem0.fec)]
// CHECK:STDOUT:     %specific_impl_fn.loc6_30.1: <specific function> = specific_impl_function %impl.elem0.loc6_30.1, @DefaultOrUnformed.WithSelf.Op(constants.%DefaultOrUnformed.facet.c98) [symbolic = %specific_impl_fn.loc6_30.2 (constants.%specific_impl_fn.619)]
// CHECK:STDOUT:     %.loc6_3.1: ref @F.%tuple.type (%tuple.type.cd5) = splice_block %u.var {}
// CHECK:STDOUT:     %DefaultOrUnformed.WithSelf.Op.call.loc6: init @F.%tuple.type (%tuple.type.cd5) to %.loc6_3.1 = call %specific_impl_fn.loc6_30.1()
// CHECK:STDOUT:     assign %u.var, %DefaultOrUnformed.WithSelf.Op.call.loc6
// CHECK:STDOUT:     %.loc6_29.1: type = splice_block %.loc6_29.3 [symbolic = %tuple.type (constants.%tuple.type.cd5)] {
// CHECK:STDOUT:       %T.ref.loc6_18: %Destroy.type = name_ref T, %T.loc4_15.2 [symbolic = %T.loc4_15.1 (constants.%T.0e7)]
// CHECK:STDOUT:       %T.as_type.loc6_19.1: type = facet_access_type %T.ref.loc6_18 [symbolic = %T.as_type.loc6_19.2 (constants.%T.as_type.700)]
// CHECK:STDOUT:       %.loc6_19: type = converted %T.ref.loc6_18, %T.as_type.loc6_19.1 [symbolic = %T.as_type.loc6_19.2 (constants.%T.as_type.700)]
// CHECK:STDOUT:       %ptr.loc6_19.1: type = ptr_type %.loc6_19 [symbolic = %ptr.loc6_19.2 (constants.%ptr.b37)]
// CHECK:STDOUT:       %T.ref.loc6_28: %Destroy.type = name_ref T, %T.loc4_15.2 [symbolic = %T.loc4_15.1 (constants.%T.0e7)]
// CHECK:STDOUT:       %T.as_type.loc6_22: type = facet_access_type %T.ref.loc6_28 [symbolic = %T.as_type.loc6_19.2 (constants.%T.as_type.700)]
// CHECK:STDOUT:       %.loc6_22: type = converted %T.ref.loc6_28, %T.as_type.loc6_22 [symbolic = %T.as_type.loc6_19.2 (constants.%T.as_type.700)]
// CHECK:STDOUT:       %const.loc6_22.1: type = const_type %.loc6_22 [symbolic = %const.loc6_22.2 (constants.%const)]
// CHECK:STDOUT:       %.loc6_29.2: %tuple.type.693 = tuple_literal (%ptr.loc6_19.1, %const.loc6_22.1) [symbolic = %tuple (constants.%tuple)]
// CHECK:STDOUT:       %.loc6_29.3: type = converted %.loc6_29.2, constants.%tuple.type.cd5 [symbolic = %tuple.type (constants.%tuple.type.cd5)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: ref @F.%tuple.type (%tuple.type.cd5) = wrapper_binding u, %u.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %u.patt.loc6_15.1: @F.%pattern_type.loc6 (%pattern_type.02b) = ref_binding_pattern u [symbolic = %u.patt.loc6_15.2 (constants.%u.patt)]
// CHECK:STDOUT:       %u.var_patt.loc6_3.1: @F.%pattern_type.loc6 (%pattern_type.02b) = var_pattern %u.patt.loc6_15.1 [symbolic = %u.var_patt.loc6_3.2 (constants.%u.var_patt)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v.var: ref @F.%struct_type.a.loc7_23.2 (%struct_type.a) = var_storage %v.var_patt.loc7_3.1
// CHECK:STDOUT:     %DefaultOrUnformed.facet.loc7_24.1: %DefaultOrUnformed.type = facet_value constants.%struct_type.a, (constants.%DefaultOrUnformed.lookup_impl_witness.53f) [symbolic = %DefaultOrUnformed.facet.loc7_24.2 (constants.%DefaultOrUnformed.facet.103)]
// CHECK:STDOUT:     %.loc7_24.1: %DefaultOrUnformed.type = converted constants.%struct_type.a, %DefaultOrUnformed.facet.loc7_24.1 [symbolic = %DefaultOrUnformed.facet.loc7_24.2 (constants.%DefaultOrUnformed.facet.103)]
// CHECK:STDOUT:     %as_type.loc7: type = facet_access_type %.loc7_24.1 [symbolic = %struct_type.a.loc7_23.2 (constants.%struct_type.a)]
// CHECK:STDOUT:     %.loc7_24.2: type = converted %.loc7_24.1, %as_type.loc7 [symbolic = %struct_type.a.loc7_23.2 (constants.%struct_type.a)]
// CHECK:STDOUT:     %impl.elem0.loc7_24.1: @F.%.loc7_24.3 (%.9be) = impl_witness_access constants.%DefaultOrUnformed.lookup_impl_witness.53f, element0 [symbolic = %impl.elem0.loc7_24.2 (constants.%impl.elem0.873)]
// CHECK:STDOUT:     %specific_impl_fn.loc7_24.1: <specific function> = specific_impl_function %impl.elem0.loc7_24.1, @DefaultOrUnformed.WithSelf.Op(constants.%DefaultOrUnformed.facet.103) [symbolic = %specific_impl_fn.loc7_24.2 (constants.%specific_impl_fn.1d1)]
// CHECK:STDOUT:     %DefaultOrUnformed.WithSelf.Op.call.loc7: init @F.%struct_type.a.loc7_23.2 (%struct_type.a) = call %specific_impl_fn.loc7_24.1()
// CHECK:STDOUT:     assign %v.var, %DefaultOrUnformed.WithSelf.Op.call.loc7
// CHECK:STDOUT:     %.loc7_23: type = splice_block %struct_type.a.loc7_23.1 [symbolic = %struct_type.a.loc7_23.2 (constants.%struct_type.a)] {
// CHECK:STDOUT:       %T.ref.loc7: %Destroy.type = name_ref T, %T.loc4_15.2 [symbolic = %T.loc4_15.1 (constants.%T.0e7)]
// CHECK:STDOUT:       %T.as_type.loc7: type = facet_access_type %T.ref.loc7 [symbolic = %T.as_type.loc6_19.2 (constants.%T.as_type.700)]
// CHECK:STDOUT:       %.loc7_22: type = converted %T.ref.loc7, %T.as_type.loc7 [symbolic = %T.as_type.loc6_19.2 (constants.%T.as_type.700)]
// CHECK:STDOUT:       %struct_type.a.loc7_23.1: type = struct_type {.a: @F.%T.as_type.loc6_19.2 (%T.as_type.700)} [symbolic = %struct_type.a.loc7_23.2 (constants.%struct_type.a)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: ref @F.%struct_type.a.loc7_23.2 (%struct_type.a) = wrapper_binding v, %v.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %v.patt.loc7_15.1: @F.%pattern_type.loc7 (%pattern_type.915) = ref_binding_pattern v [symbolic = %v.patt.loc7_15.2 (constants.%v.patt)]
// CHECK:STDOUT:       %v.var_patt.loc7_3.1: @F.%pattern_type.loc7 (%pattern_type.915) = var_pattern %v.patt.loc7_15.1 [symbolic = %v.var_patt.loc7_3.2 (constants.%v.var_patt)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %w.var: ref @F.%Array.loc8_27.2 (%Array.e06) = var_storage %w.var_patt.loc8_3.1
// CHECK:STDOUT:     %DefaultOrUnformed.facet.loc8_28.1: %DefaultOrUnformed.type = facet_value constants.%Array.e06, (constants.%DefaultOrUnformed.lookup_impl_witness.f03) [symbolic = %DefaultOrUnformed.facet.loc8_28.2 (constants.%DefaultOrUnformed.facet.61d)]
// CHECK:STDOUT:     %.loc8_28.1: %DefaultOrUnformed.type = converted constants.%Array.e06, %DefaultOrUnformed.facet.loc8_28.1 [symbolic = %DefaultOrUnformed.facet.loc8_28.2 (constants.%DefaultOrUnformed.facet.61d)]
// CHECK:STDOUT:     %as_type.loc8: type = facet_access_type %.loc8_28.1 [symbolic = %Array.loc8_27.2 (constants.%Array.e06)]
// CHECK:STDOUT:     %.loc8_28.2: type = converted %.loc8_28.1, %as_type.loc8 [symbolic = %Array.loc8_27.2 (constants.%Array.e06)]
// CHECK:STDOUT:     %impl.elem0.loc8_28.1: @F.%.loc8_28.3 (%.b93) = impl_witness_access constants.%DefaultOrUnformed.lookup_impl_witness.f03, element0 [symbolic = %impl.elem0.loc8_28.2 (constants.%impl.elem0.1c3)]
// CHECK:STDOUT:     %specific_impl_fn.loc8_28.1: <specific function> = specific_impl_function %impl.elem0.loc8_28.1, @DefaultOrUnformed.WithSelf.Op(constants.%DefaultOrUnformed.facet.61d) [symbolic = %specific_impl_fn.loc8_28.2 (constants.%specific_impl_fn.aef)]
// CHECK:STDOUT:     %.loc8_3.1: ref @F.%Array.loc8_27.2 (%Array.e06) = splice_block %w.var {}
// CHECK:STDOUT:     %DefaultOrUnformed.WithSelf.Op.call.loc8: init @F.%Array.loc8_27.2 (%Array.e06) to %.loc8_3.1 = call %specific_impl_fn.loc8_28.1()
// CHECK:STDOUT:     assign %w.var, %DefaultOrUnformed.WithSelf.Op.call.loc8
// CHECK:STDOUT:     %.loc8_27: type = splice_block %Array.loc8_27.1 [symbolic = %Array.loc8_27.2 (constants.%Array.e06)] {
// CHECK:STDOUT:       %T.ref.loc8: %Destroy.type = name_ref T, %T.loc4_15.2 [symbolic = %T.loc4_15.1 (constants.%T.0e7)]
// CHECK:STDOUT:       %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5]
// CHECK:STDOUT:       %T.as_type.loc8: type = facet_access_type %T.ref.loc8 [symbolic = %T.as_type.loc6_19.2 (constants.%T.as_type.700)]
// CHECK:STDOUT:       %.loc8_23: type = converted %T.ref.loc8, %T.as_type.loc8 [symbolic = %T.as_type.loc6_19.2 (constants.%T.as_type.700)]
// CHECK:STDOUT:       %Array.loc8_27.1: type = class_type @Array, @Array(constants.%T.as_type.700, constants.%int_5) [symbolic = %Array.loc8_27.2 (constants.%Array.e06)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %w: ref @F.%Array.loc8_27.2 (%Array.e06) = wrapper_binding w, %w.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %w.patt.loc8_15.1: @F.%pattern_type.loc8 (%pattern_type.38c) = ref_binding_pattern w [symbolic = %w.patt.loc8_15.2 (constants.%w.patt)]
// CHECK:STDOUT:       %w.var_patt.loc8_3.1: @F.%pattern_type.loc8 (%pattern_type.38c) = var_pattern %w.patt.loc8_15.1 [symbolic = %w.var_patt.loc8_3.2 (constants.%w.var_patt)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %impl.elem2.loc8_3.1: @F.%.loc8_3.4 (%.44d) = impl_witness_access constants.%Destroy.lookup_impl_witness.fc7, element2 [symbolic = %impl.elem2.loc8_3.2 (constants.%impl.elem2.f1a)]
// CHECK:STDOUT:     %bound_method.loc8_3.1: <bound method> = bound_method %w.var, %impl.elem2.loc8_3.1
// CHECK:STDOUT:     %Destroy.facet.loc8_3.1: %Destroy.type = facet_value constants.%Array.e06, (constants.%Destroy.lookup_impl_witness.fc7) [symbolic = %Destroy.facet.loc8_3.3 (constants.%Destroy.facet.aae)]
// CHECK:STDOUT:     %.loc8_3.2: %Destroy.type = converted constants.%Array.e06, %Destroy.facet.loc8_3.1 [symbolic = %Destroy.facet.loc8_3.3 (constants.%Destroy.facet.aae)]
// CHECK:STDOUT:     %Destroy.facet.loc8_3.2: %Destroy.type = facet_value constants.%Array.e06, (constants.%Destroy.lookup_impl_witness.fc7) [symbolic = %Destroy.facet.loc8_3.3 (constants.%Destroy.facet.aae)]
// CHECK:STDOUT:     %.loc8_3.3: %Destroy.type = converted constants.%Array.e06, %Destroy.facet.loc8_3.2 [symbolic = %Destroy.facet.loc8_3.3 (constants.%Destroy.facet.aae)]
// CHECK:STDOUT:     %specific_impl_fn.loc8_3.1: <specific function> = specific_impl_function %impl.elem2.loc8_3.1, @Destroy.WithSelf.SelfDestruct(constants.%Destroy.facet.aae) [symbolic = %specific_impl_fn.loc8_3.2 (constants.%specific_impl_fn.5b6)]
// CHECK:STDOUT:     %bound_method.loc8_3.2: <bound method> = bound_method %w.var, %specific_impl_fn.loc8_3.1
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc8: init %empty_tuple.type = call %bound_method.loc8_3.2(%w.var)
// CHECK:STDOUT:     %impl.elem2.loc7_3.1: @F.%.loc7_3.3 (%.8a1) = impl_witness_access constants.%Destroy.lookup_impl_witness.7f9, element2 [symbolic = %impl.elem2.loc7_3.2 (constants.%impl.elem2.86f)]
// CHECK:STDOUT:     %bound_method.loc7_3.1: <bound method> = bound_method %v.var, %impl.elem2.loc7_3.1
// CHECK:STDOUT:     %Destroy.facet.loc7_3.1: %Destroy.type = facet_value constants.%struct_type.a, (constants.%Destroy.lookup_impl_witness.7f9) [symbolic = %Destroy.facet.loc7_3.3 (constants.%Destroy.facet.aa3)]
// CHECK:STDOUT:     %.loc7_3.1: %Destroy.type = converted constants.%struct_type.a, %Destroy.facet.loc7_3.1 [symbolic = %Destroy.facet.loc7_3.3 (constants.%Destroy.facet.aa3)]
// CHECK:STDOUT:     %Destroy.facet.loc7_3.2: %Destroy.type = facet_value constants.%struct_type.a, (constants.%Destroy.lookup_impl_witness.7f9) [symbolic = %Destroy.facet.loc7_3.3 (constants.%Destroy.facet.aa3)]
// CHECK:STDOUT:     %.loc7_3.2: %Destroy.type = converted constants.%struct_type.a, %Destroy.facet.loc7_3.2 [symbolic = %Destroy.facet.loc7_3.3 (constants.%Destroy.facet.aa3)]
// CHECK:STDOUT:     %specific_impl_fn.loc7_3.1: <specific function> = specific_impl_function %impl.elem2.loc7_3.1, @Destroy.WithSelf.SelfDestruct(constants.%Destroy.facet.aa3) [symbolic = %specific_impl_fn.loc7_3.2 (constants.%specific_impl_fn.86e)]
// CHECK:STDOUT:     %bound_method.loc7_3.2: <bound method> = bound_method %v.var, %specific_impl_fn.loc7_3.1
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc7: init %empty_tuple.type = call %bound_method.loc7_3.2(%v.var)
// CHECK:STDOUT:     %impl.elem2.loc6_3.1: @F.%.loc6_3.4 (%.b75) = impl_witness_access constants.%Destroy.lookup_impl_witness.d26, element2 [symbolic = %impl.elem2.loc6_3.2 (constants.%impl.elem2.dad)]
// CHECK:STDOUT:     %bound_method.loc6_3.1: <bound method> = bound_method %u.var, %impl.elem2.loc6_3.1
// CHECK:STDOUT:     %Destroy.facet.loc6_3.1: %Destroy.type = facet_value constants.%tuple.type.cd5, (constants.%Destroy.lookup_impl_witness.d26) [symbolic = %Destroy.facet.loc6_3.3 (constants.%Destroy.facet.7ba)]
// CHECK:STDOUT:     %.loc6_3.2: %Destroy.type = converted constants.%tuple.type.cd5, %Destroy.facet.loc6_3.1 [symbolic = %Destroy.facet.loc6_3.3 (constants.%Destroy.facet.7ba)]
// CHECK:STDOUT:     %Destroy.facet.loc6_3.2: %Destroy.type = facet_value constants.%tuple.type.cd5, (constants.%Destroy.lookup_impl_witness.d26) [symbolic = %Destroy.facet.loc6_3.3 (constants.%Destroy.facet.7ba)]
// CHECK:STDOUT:     %.loc6_3.3: %Destroy.type = converted constants.%tuple.type.cd5, %Destroy.facet.loc6_3.2 [symbolic = %Destroy.facet.loc6_3.3 (constants.%Destroy.facet.7ba)]
// CHECK:STDOUT:     %specific_impl_fn.loc6_3.1: <specific function> = specific_impl_function %impl.elem2.loc6_3.1, @Destroy.WithSelf.SelfDestruct(constants.%Destroy.facet.7ba) [symbolic = %specific_impl_fn.loc6_3.2 (constants.%specific_impl_fn.d5b)]
// CHECK:STDOUT:     %bound_method.loc6_3.2: <bound method> = bound_method %u.var, %specific_impl_fn.loc6_3.1
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc6: init %empty_tuple.type = call %bound_method.loc6_3.2(%u.var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%N.loc12_15.2: %i32) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %N.loc12_15.1, constants.%Int.as.ImplicitAs.impl.Convert.779 [symbolic = %Int.as.ImplicitAs.impl.Convert.bound (constants.%Int.as.ImplicitAs.impl.Convert.bound)]
// CHECK:STDOUT:   %bound_method.loc14_28.3: <bound method> = bound_method %N.loc12_15.1, constants.%Int.as.ImplicitAs.impl.Convert.specific_fn [symbolic = %bound_method.loc14_28.3 (constants.%bound_method)]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.call.loc14_28.2: init Core.IntLiteral = call %bound_method.loc14_28.3(%N.loc12_15.1) [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc14_28.2 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:   %Array.loc14_29.2: type = class_type @Array, @Array(constants.%i32, %Int.as.ImplicitAs.impl.Convert.call.loc14_28.2) [symbolic = %Array.loc14_29.2 (constants.%Array.e80)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Array.loc14_29.2 [symbolic = %require_complete (constants.%require_complete.fcb)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Array.loc14_29.2 [symbolic = %pattern_type (constants.%pattern_type.fc0)]
// CHECK:STDOUT:   %k.patt.loc14_15.2: @G.%pattern_type (%pattern_type.fc0) = ref_binding_pattern k [symbolic = %k.patt.loc14_15.2 (constants.%k.patt)]
// CHECK:STDOUT:   %k.var_patt.loc14_3.2: @G.%pattern_type (%pattern_type.fc0) = var_pattern %k.patt.loc14_15.2 [symbolic = %k.var_patt.loc14_3.2 (constants.%k.var_patt)]
// CHECK:STDOUT:   %DefaultOrUnformed.lookup_impl_witness: <witness> = lookup_impl_witness %Array.loc14_29.2, @DefaultOrUnformed [symbolic = %DefaultOrUnformed.lookup_impl_witness (constants.%DefaultOrUnformed.lookup_impl_witness.b5f)]
// CHECK:STDOUT:   %require_witness_defn.loc14_30: require_witness_defn_type = require_witness_defn %DefaultOrUnformed.lookup_impl_witness [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc14_30.2: %DefaultOrUnformed.type = facet_value %Array.loc14_29.2, (%DefaultOrUnformed.lookup_impl_witness) [symbolic = %DefaultOrUnformed.facet.loc14_30.2 (constants.%DefaultOrUnformed.facet.48a)]
// CHECK:STDOUT:   %DefaultOrUnformed.WithSelf.Op.type: type = fn_type @DefaultOrUnformed.WithSelf.Op, @DefaultOrUnformed.WithSelf(%DefaultOrUnformed.facet.loc14_30.2) [symbolic = %DefaultOrUnformed.WithSelf.Op.type (constants.%DefaultOrUnformed.WithSelf.Op.type.8de)]
// CHECK:STDOUT:   %.loc14_30.3: type = fn_type_with_self_type %DefaultOrUnformed.WithSelf.Op.type, %DefaultOrUnformed.facet.loc14_30.2 [symbolic = %.loc14_30.3 (constants.%.431)]
// CHECK:STDOUT:   %impl.elem0.loc14_30.2: @G.%.loc14_30.3 (%.431) = impl_witness_access %DefaultOrUnformed.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc14_30.2 (constants.%impl.elem0.b7f)]
// CHECK:STDOUT:   %specific_impl_fn.loc14_30.2: <specific function> = specific_impl_function %impl.elem0.loc14_30.2, @DefaultOrUnformed.WithSelf.Op(%DefaultOrUnformed.facet.loc14_30.2) [symbolic = %specific_impl_fn.loc14_30.2 (constants.%specific_impl_fn.971)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %Array.loc14_29.2, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.27d)]
// CHECK:STDOUT:   %require_witness_defn.loc14_3: require_witness_defn_type = require_witness_defn %Destroy.lookup_impl_witness [symbolic]
// CHECK:STDOUT:   %Destroy.facet.loc14_3.3: %Destroy.type = facet_value %Array.loc14_29.2, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet.loc14_3.3 (constants.%Destroy.facet.ca8)]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%Destroy.facet.loc14_3.3) [symbolic = %Destroy.WithSelf.SelfDestruct.type (constants.%Destroy.WithSelf.SelfDestruct.type.4da)]
// CHECK:STDOUT:   %.loc14_3.4: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type, %Destroy.facet.loc14_3.3 [symbolic = %.loc14_3.4 (constants.%.48f)]
// CHECK:STDOUT:   %impl.elem2.loc14_3.2: @G.%.loc14_3.4 (%.48f) = impl_witness_access %Destroy.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc14_3.2 (constants.%impl.elem2.af7)]
// CHECK:STDOUT:   %specific_impl_fn.loc14_3.2: <specific function> = specific_impl_function %impl.elem2.loc14_3.2, @Destroy.WithSelf.SelfDestruct(%Destroy.facet.loc14_3.3) [symbolic = %specific_impl_fn.loc14_3.2 (constants.%specific_impl_fn.a3f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %k.var: ref @G.%Array.loc14_29.2 (%Array.e80) = var_storage %k.var_patt.loc14_3.1
// CHECK:STDOUT:     %DefaultOrUnformed.facet.loc14_30.1: %DefaultOrUnformed.type = facet_value constants.%Array.e80, (constants.%DefaultOrUnformed.lookup_impl_witness.b5f) [symbolic = %DefaultOrUnformed.facet.loc14_30.2 (constants.%DefaultOrUnformed.facet.48a)]
// CHECK:STDOUT:     %.loc14_30.1: %DefaultOrUnformed.type = converted constants.%Array.e80, %DefaultOrUnformed.facet.loc14_30.1 [symbolic = %DefaultOrUnformed.facet.loc14_30.2 (constants.%DefaultOrUnformed.facet.48a)]
// CHECK:STDOUT:     %as_type: type = facet_access_type %.loc14_30.1 [symbolic = %Array.loc14_29.2 (constants.%Array.e80)]
// CHECK:STDOUT:     %.loc14_30.2: type = converted %.loc14_30.1, %as_type [symbolic = %Array.loc14_29.2 (constants.%Array.e80)]
// CHECK:STDOUT:     %impl.elem0.loc14_30.1: @G.%.loc14_30.3 (%.431) = impl_witness_access constants.%DefaultOrUnformed.lookup_impl_witness.b5f, element0 [symbolic = %impl.elem0.loc14_30.2 (constants.%impl.elem0.b7f)]
// CHECK:STDOUT:     %specific_impl_fn.loc14_30.1: <specific function> = specific_impl_function %impl.elem0.loc14_30.1, @DefaultOrUnformed.WithSelf.Op(constants.%DefaultOrUnformed.facet.48a) [symbolic = %specific_impl_fn.loc14_30.2 (constants.%specific_impl_fn.971)]
// CHECK:STDOUT:     %.loc14_3.1: ref @G.%Array.loc14_29.2 (%Array.e80) = splice_block %k.var {}
// CHECK:STDOUT:     %DefaultOrUnformed.WithSelf.Op.call: init @G.%Array.loc14_29.2 (%Array.e80) to %.loc14_3.1 = call %specific_impl_fn.loc14_30.1()
// CHECK:STDOUT:     assign %k.var, %DefaultOrUnformed.WithSelf.Op.call
// CHECK:STDOUT:     %.loc14_29: type = splice_block %Array.loc14_29.1 [symbolic = %Array.loc14_29.2 (constants.%Array.e80)] {
// CHECK:STDOUT:       %i32.loc14: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %N.ref: %i32 = name_ref N, %N.loc12_15.2 [symbolic = %N.loc12_15.1 (constants.%N.c34)]
// CHECK:STDOUT:       %impl.elem0.loc14_28: %.ce7 = impl_witness_access constants.%ImplicitAs.impl_witness.323, element0 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.779]
// CHECK:STDOUT:       %bound_method.loc14_28.1: <bound method> = bound_method %N.ref, %impl.elem0.loc14_28 [symbolic = %Int.as.ImplicitAs.impl.Convert.bound (constants.%Int.as.ImplicitAs.impl.Convert.bound)]
// CHECK:STDOUT:       %specific_fn: <specific function> = specific_function %impl.elem0.loc14_28, @Int.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Int.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:       %bound_method.loc14_28.2: <bound method> = bound_method %N.ref, %specific_fn [symbolic = %bound_method.loc14_28.3 (constants.%bound_method)]
// CHECK:STDOUT:       %Int.as.ImplicitAs.impl.Convert.call.loc14_28.1: init Core.IntLiteral = call %bound_method.loc14_28.2(%N.ref) [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc14_28.2 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:       %.loc14_28.1: Core.IntLiteral = value_of_initializer %Int.as.ImplicitAs.impl.Convert.call.loc14_28.1 [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc14_28.2 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:       %.loc14_28.2: Core.IntLiteral = converted %N.ref, %.loc14_28.1 [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc14_28.2 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:       %Array.loc14_29.1: type = class_type @Array, @Array(constants.%i32, constants.%Int.as.ImplicitAs.impl.Convert.call) [symbolic = %Array.loc14_29.2 (constants.%Array.e80)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %k: ref @G.%Array.loc14_29.2 (%Array.e80) = wrapper_binding k, %k.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %k.patt.loc14_15.1: @G.%pattern_type (%pattern_type.fc0) = ref_binding_pattern k [symbolic = %k.patt.loc14_15.2 (constants.%k.patt)]
// CHECK:STDOUT:       %k.var_patt.loc14_3.1: @G.%pattern_type (%pattern_type.fc0) = var_pattern %k.patt.loc14_15.1 [symbolic = %k.var_patt.loc14_3.2 (constants.%k.var_patt)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %impl.elem2.loc14_3.1: @G.%.loc14_3.4 (%.48f) = impl_witness_access constants.%Destroy.lookup_impl_witness.27d, element2 [symbolic = %impl.elem2.loc14_3.2 (constants.%impl.elem2.af7)]
// CHECK:STDOUT:     %bound_method.loc14_3.1: <bound method> = bound_method %k.var, %impl.elem2.loc14_3.1
// CHECK:STDOUT:     %Destroy.facet.loc14_3.1: %Destroy.type = facet_value constants.%Array.e80, (constants.%Destroy.lookup_impl_witness.27d) [symbolic = %Destroy.facet.loc14_3.3 (constants.%Destroy.facet.ca8)]
// CHECK:STDOUT:     %.loc14_3.2: %Destroy.type = converted constants.%Array.e80, %Destroy.facet.loc14_3.1 [symbolic = %Destroy.facet.loc14_3.3 (constants.%Destroy.facet.ca8)]
// CHECK:STDOUT:     %Destroy.facet.loc14_3.2: %Destroy.type = facet_value constants.%Array.e80, (constants.%Destroy.lookup_impl_witness.27d) [symbolic = %Destroy.facet.loc14_3.3 (constants.%Destroy.facet.ca8)]
// CHECK:STDOUT:     %.loc14_3.3: %Destroy.type = converted constants.%Array.e80, %Destroy.facet.loc14_3.2 [symbolic = %Destroy.facet.loc14_3.3 (constants.%Destroy.facet.ca8)]
// CHECK:STDOUT:     %specific_impl_fn.loc14_3.1: <specific function> = specific_impl_function %impl.elem2.loc14_3.1, @Destroy.WithSelf.SelfDestruct(constants.%Destroy.facet.ca8) [symbolic = %specific_impl_fn.loc14_3.2 (constants.%specific_impl_fn.a3f)]
// CHECK:STDOUT:     %bound_method.loc14_3.2: <bound method> = bound_method %k.var, %specific_impl_fn.loc14_3.1
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %bound_method.loc14_3.2(%k.var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.0e7) {
// CHECK:STDOUT:   %T.patt.loc4_15.2 => constants.%T.patt.351
// CHECK:STDOUT:   %T.loc4_15.1 => constants.%T.0e7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%N.c34) {
// CHECK:STDOUT:   %N.patt.loc12_15.2 => constants.%N.patt.d64
// CHECK:STDOUT:   %N.loc12_15.1 => constants.%N.c34
// CHECK:STDOUT: }
// CHECK:STDOUT:

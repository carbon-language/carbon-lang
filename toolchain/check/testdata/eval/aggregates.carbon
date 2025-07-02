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
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.d07: type = tuple_type (%i32, %i32) [concrete]
// CHECK:STDOUT:   %pattern_type.511: type = pattern_type %tuple.type.d07 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.type.f94: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.0f9: type = fn_type @Convert.2, @impl.4f9(%To) [symbolic]
// CHECK:STDOUT:   %Convert.f06: %Convert.type.0f9 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c75: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a2f, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.c75) [concrete]
// CHECK:STDOUT:   %.9c3: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound.ab5: <bound method> = bound_method %int_1.5b8, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.9a1: <bound method> = bound_method %int_1.5b8, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Convert.bound.ef9: <bound method> = bound_method %int_2.ecc, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.b92: <bound method> = bound_method %int_2.ecc, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.21c: %tuple.type.d07 = tuple_value (%int_1.5d2, %int_2.ef8) [concrete]
// CHECK:STDOUT:   %tuple.elem0: ref %i32 = tuple_access file.%tuple_copy.var, element0 [concrete]
// CHECK:STDOUT:   %tuple.elem1: ref %i32 = tuple_access file.%tuple_copy.var, element1 [concrete]
// CHECK:STDOUT:   %struct_type.a.b.c: type = struct_type {.a: %i32, .b: %i32, .c: %i32} [concrete]
// CHECK:STDOUT:   %pattern_type.8ae: type = pattern_type %struct_type.a.b.c [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %struct_type.c.b.a: type = struct_type {.c: Core.IntLiteral, .b: Core.IntLiteral, .a: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct_type.b.a.c: type = struct_type {.b: %i32, .a: %i32, .c: %i32} [concrete]
// CHECK:STDOUT:   %Convert.bound.b30: <bound method> = bound_method %int_3.1ba, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.047: <bound method> = bound_method %int_3.1ba, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.822: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %struct.21d: %struct_type.b.a.c = struct_value (%int_2.ef8, %int_1.5d2, %int_3.822) [concrete]
// CHECK:STDOUT:   %.91e: ref %i32 = struct_access file.%struct_copy.var, element1 [concrete]
// CHECK:STDOUT:   %.657: ref %i32 = struct_access file.%struct_copy.var, element0 [concrete]
// CHECK:STDOUT:   %.8fb: ref %i32 = struct_access file.%struct_copy.var, element2 [concrete]
// CHECK:STDOUT:   %struct.cff: %struct_type.a.b.c = struct_value (%int_1.5d2, %int_2.ef8, %int_3.822) [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_1.5b8, %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.a98: type = pattern_type %array_type [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %tuple.type.985: type = tuple_type (Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %int_7: Core.IntLiteral = int_value 7 [concrete]
// CHECK:STDOUT:   %int_9: Core.IntLiteral = int_value 9 [concrete]
// CHECK:STDOUT:   %tuple.type.d46: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.869: %tuple.type.d46 = tuple_value (%int_5, %int_7, %int_1.5b8, %int_9) [concrete]
// CHECK:STDOUT:   %Convert.bound.d04: <bound method> = bound_method %int_0.5c6, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.b6e: <bound method> = bound_method %int_0.5c6, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%int_0.6a9) [concrete]
// CHECK:STDOUT:   %struct_type.a.b: type = struct_type {.a: Core.IntLiteral, .b: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct.a81: %struct_type.a.b = struct_value (%int_3.1ba, %int_1.5b8) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.a5b: @impl.4f9.%Convert.type (%Convert.type.0f9) = import_ref Core//prelude/parts/int, loc16_39, loaded [symbolic = @impl.4f9.%Convert (constants.%Convert.f06)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a2f = impl_witness_table (%Core.import_ref.a5b), @impl.4f9 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %tuple_copy.patt: %pattern_type.511 = binding_pattern tuple_copy [concrete]
// CHECK:STDOUT:     %tuple_copy.var_patt: %pattern_type.511 = var_pattern %tuple_copy.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %tuple_copy.var: ref %tuple.type.d07 = var %tuple_copy.var_patt [concrete]
// CHECK:STDOUT:   %.loc4_26.1: type = splice_block %.loc4_26.3 [concrete = constants.%tuple.type.d07] {
// CHECK:STDOUT:     %int_32.loc4_18: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc4_18: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc4_23: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc4_23: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc4_26.2: %tuple.type.24b = tuple_literal (%i32.loc4_18, %i32.loc4_23)
// CHECK:STDOUT:     %.loc4_26.3: type = converted %.loc4_26.2, constants.%tuple.type.d07 [concrete = constants.%tuple.type.d07]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %tuple_copy: ref %tuple.type.d07 = bind_name tuple_copy, %tuple_copy.var [concrete = %tuple_copy.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %struct_copy.patt: %pattern_type.8ae = binding_pattern struct_copy [concrete]
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
// CHECK:STDOUT:   %struct_copy: ref %struct_type.a.b.c = bind_name struct_copy, %struct_copy.var [concrete = %struct_copy.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %tuple_index.patt: %pattern_type.a98 = binding_pattern tuple_index [concrete]
// CHECK:STDOUT:     %tuple_index.var_patt: %pattern_type.a98 = var_pattern %tuple_index.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %tuple_index.var: ref %array_type = var %tuple_index.var_patt [concrete]
// CHECK:STDOUT:   %.loc8: type = splice_block %array_type.loc8 [concrete = constants.%array_type] {
// CHECK:STDOUT:     %int_32.loc8: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc8: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %array_type.loc8: type = array_type %int_1.loc8, %i32.loc8 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %tuple_index: ref %array_type = bind_name tuple_index, %tuple_index.var [concrete = %tuple_index.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %struct_access.patt: %pattern_type.a98 = binding_pattern struct_access [concrete]
// CHECK:STDOUT:     %struct_access.var_patt: %pattern_type.a98 = var_pattern %struct_access.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %struct_access.var: ref %array_type = var %struct_access.var_patt [concrete]
// CHECK:STDOUT:   %.loc10: type = splice_block %array_type.loc10 [concrete = constants.%array_type] {
// CHECK:STDOUT:     %int_32.loc10: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc10: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_1.loc10: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %array_type.loc10: type = array_type %int_1.loc10, %i32.loc10 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %struct_access: ref %array_type = bind_name struct_access, %struct_access.var [concrete = %struct_access.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1.loc4: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc4: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc4_35.1: %tuple.type.f94 = tuple_literal (%int_1.loc4, %int_2.loc4)
// CHECK:STDOUT:   %int_32.loc4_41: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc4_41: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %int_32.loc4_46: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc4_46: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc4_49.1: %tuple.type.24b = tuple_literal (%i32.loc4_41, %i32.loc4_46)
// CHECK:STDOUT:   %.loc4_49.2: type = converted %.loc4_49.1, constants.%tuple.type.d07 [concrete = constants.%tuple.type.d07]
// CHECK:STDOUT:   %impl.elem0.loc4_35.1: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc4_35.1: <bound method> = bound_method %int_1.loc4, %impl.elem0.loc4_35.1 [concrete = constants.%Convert.bound.ab5]
// CHECK:STDOUT:   %specific_fn.loc4_35.1: <specific function> = specific_function %impl.elem0.loc4_35.1, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc4_35.2: <bound method> = bound_method %int_1.loc4, %specific_fn.loc4_35.1 [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:   %int.convert_checked.loc4_35.1: init %i32 = call %bound_method.loc4_35.2(%int_1.loc4) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc4_35.2: %i32 = value_of_initializer %int.convert_checked.loc4_35.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc4_35.3: %i32 = converted %int_1.loc4, %.loc4_35.2 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc4_35.2: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc4_35.3: <bound method> = bound_method %int_2.loc4, %impl.elem0.loc4_35.2 [concrete = constants.%Convert.bound.ef9]
// CHECK:STDOUT:   %specific_fn.loc4_35.2: <specific function> = specific_function %impl.elem0.loc4_35.2, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc4_35.4: <bound method> = bound_method %int_2.loc4, %specific_fn.loc4_35.2 [concrete = constants.%bound_method.b92]
// CHECK:STDOUT:   %int.convert_checked.loc4_35.2: init %i32 = call %bound_method.loc4_35.4(%int_2.loc4) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc4_35.4: %i32 = value_of_initializer %int.convert_checked.loc4_35.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc4_35.5: %i32 = converted %int_2.loc4, %.loc4_35.4 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %tuple.loc4: %tuple.type.d07 = tuple_value (%.loc4_35.3, %.loc4_35.5) [concrete = constants.%tuple.21c]
// CHECK:STDOUT:   %.loc4_37.1: %tuple.type.d07 = converted %.loc4_35.1, %tuple.loc4 [concrete = constants.%tuple.21c]
// CHECK:STDOUT:   %tuple.elem0.loc4_37.1: %i32 = tuple_access %.loc4_37.1, element0 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %tuple.elem0.loc4_37.2: ref %i32 = tuple_access file.%tuple_copy.var, element0 [concrete = constants.%tuple.elem0]
// CHECK:STDOUT:   %.loc4_37.2: init %i32 = initialize_from %tuple.elem0.loc4_37.1 to %tuple.elem0.loc4_37.2 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %tuple.elem1.loc4_37.1: %i32 = tuple_access %.loc4_37.1, element1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %tuple.elem1.loc4_37.2: ref %i32 = tuple_access file.%tuple_copy.var, element1 [concrete = constants.%tuple.elem1]
// CHECK:STDOUT:   %.loc4_37.3: init %i32 = initialize_from %tuple.elem1.loc4_37.1 to %tuple.elem1.loc4_37.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc4_37.4: init %tuple.type.d07 = tuple_init (%.loc4_37.2, %.loc4_37.3) to file.%tuple_copy.var [concrete = constants.%tuple.21c]
// CHECK:STDOUT:   %.loc4_1: init %tuple.type.d07 = converted %.loc4_37.1, %.loc4_37.4 [concrete = constants.%tuple.21c]
// CHECK:STDOUT:   assign file.%tuple_copy.var, %.loc4_1
// CHECK:STDOUT:   %int_3.loc6: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %int_2.loc6: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %int_1.loc6: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc6_71.1: %struct_type.c.b.a = struct_literal (%int_3.loc6, %int_2.loc6, %int_1.loc6)
// CHECK:STDOUT:   %int_32.loc6_81: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc6_81: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %int_32.loc6_90: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc6_90: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %int_32.loc6_99: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc6_99: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %struct_type.b.a.c: type = struct_type {.b: %i32, .a: %i32, .c: %i32} [concrete = constants.%struct_type.b.a.c]
// CHECK:STDOUT:   %impl.elem0.loc6_71.1: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc6_71.1: <bound method> = bound_method %int_2.loc6, %impl.elem0.loc6_71.1 [concrete = constants.%Convert.bound.ef9]
// CHECK:STDOUT:   %specific_fn.loc6_71.1: <specific function> = specific_function %impl.elem0.loc6_71.1, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_71.2: <bound method> = bound_method %int_2.loc6, %specific_fn.loc6_71.1 [concrete = constants.%bound_method.b92]
// CHECK:STDOUT:   %int.convert_checked.loc6_71.1: init %i32 = call %bound_method.loc6_71.2(%int_2.loc6) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc6_71.2: %i32 = value_of_initializer %int.convert_checked.loc6_71.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc6_71.3: %i32 = converted %int_2.loc6, %.loc6_71.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %impl.elem0.loc6_71.2: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc6_71.3: <bound method> = bound_method %int_1.loc6, %impl.elem0.loc6_71.2 [concrete = constants.%Convert.bound.ab5]
// CHECK:STDOUT:   %specific_fn.loc6_71.2: <specific function> = specific_function %impl.elem0.loc6_71.2, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_71.4: <bound method> = bound_method %int_1.loc6, %specific_fn.loc6_71.2 [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:   %int.convert_checked.loc6_71.2: init %i32 = call %bound_method.loc6_71.4(%int_1.loc6) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc6_71.4: %i32 = value_of_initializer %int.convert_checked.loc6_71.2 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc6_71.5: %i32 = converted %int_1.loc6, %.loc6_71.4 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc6_71.3: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc6_71.5: <bound method> = bound_method %int_3.loc6, %impl.elem0.loc6_71.3 [concrete = constants.%Convert.bound.b30]
// CHECK:STDOUT:   %specific_fn.loc6_71.3: <specific function> = specific_function %impl.elem0.loc6_71.3, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_71.6: <bound method> = bound_method %int_3.loc6, %specific_fn.loc6_71.3 [concrete = constants.%bound_method.047]
// CHECK:STDOUT:   %int.convert_checked.loc6_71.3: init %i32 = call %bound_method.loc6_71.6(%int_3.loc6) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc6_71.6: %i32 = value_of_initializer %int.convert_checked.loc6_71.3 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc6_71.7: %i32 = converted %int_3.loc6, %.loc6_71.6 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %struct.loc6: %struct_type.b.a.c = struct_value (%.loc6_71.3, %.loc6_71.5, %.loc6_71.7) [concrete = constants.%struct.21d]
// CHECK:STDOUT:   %.loc6_73.1: %struct_type.b.a.c = converted %.loc6_71.1, %struct.loc6 [concrete = constants.%struct.21d]
// CHECK:STDOUT:   %.loc6_73.2: %i32 = struct_access %.loc6_73.1, element1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc6_73.3: ref %i32 = struct_access file.%struct_copy.var, element1 [concrete = constants.%.91e]
// CHECK:STDOUT:   %.loc6_73.4: init %i32 = initialize_from %.loc6_73.2 to %.loc6_73.3 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc6_73.5: %i32 = struct_access %.loc6_73.1, element0 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc6_73.6: ref %i32 = struct_access file.%struct_copy.var, element0 [concrete = constants.%.657]
// CHECK:STDOUT:   %.loc6_73.7: init %i32 = initialize_from %.loc6_73.5 to %.loc6_73.6 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc6_73.8: %i32 = struct_access %.loc6_73.1, element2 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc6_73.9: ref %i32 = struct_access file.%struct_copy.var, element2 [concrete = constants.%.8fb]
// CHECK:STDOUT:   %.loc6_73.10: init %i32 = initialize_from %.loc6_73.8 to %.loc6_73.9 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc6_73.11: init %struct_type.a.b.c = struct_init (%.loc6_73.4, %.loc6_73.7, %.loc6_73.10) to file.%struct_copy.var [concrete = constants.%struct.cff]
// CHECK:STDOUT:   %.loc6_1: init %struct_type.a.b.c = converted %.loc6_73.1, %.loc6_73.11 [concrete = constants.%struct.cff]
// CHECK:STDOUT:   assign file.%struct_copy.var, %.loc6_1
// CHECK:STDOUT:   %int_0.loc8_35: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc8_37.1: %tuple.type.985 = tuple_literal (%int_0.loc8_35)
// CHECK:STDOUT:   %int_32.loc8: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc8: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5]
// CHECK:STDOUT:   %int_7: Core.IntLiteral = int_value 7 [concrete = constants.%int_7]
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_9: Core.IntLiteral = int_value 9 [concrete = constants.%int_9]
// CHECK:STDOUT:   %.loc8_64.1: %tuple.type.d46 = tuple_literal (%int_5, %int_7, %int_1.loc8, %int_9)
// CHECK:STDOUT:   %int_2.loc8: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %tuple.loc8: %tuple.type.d46 = tuple_value (%int_5, %int_7, %int_1.loc8, %int_9) [concrete = constants.%tuple.869]
// CHECK:STDOUT:   %.loc8_64.2: %tuple.type.d46 = converted %.loc8_64.1, %tuple.loc8 [concrete = constants.%tuple.869]
// CHECK:STDOUT:   %tuple.elem2: Core.IntLiteral = tuple_access %.loc8_64.2, element2 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %array_type.loc8: type = array_type %tuple.elem2, %i32.loc8 [concrete = constants.%array_type]
// CHECK:STDOUT:   %impl.elem0.loc8: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc8_37.1: <bound method> = bound_method %int_0.loc8_35, %impl.elem0.loc8 [concrete = constants.%Convert.bound.d04]
// CHECK:STDOUT:   %specific_fn.loc8: <specific function> = specific_function %impl.elem0.loc8, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_37.2: <bound method> = bound_method %int_0.loc8_35, %specific_fn.loc8 [concrete = constants.%bound_method.b6e]
// CHECK:STDOUT:   %int.convert_checked.loc8: init %i32 = call %bound_method.loc8_37.2(%int_0.loc8_35) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc8_37.2: init %i32 = converted %int_0.loc8_35, %int.convert_checked.loc8 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc8_1: ref %array_type = splice_block file.%tuple_index.var [concrete = file.%tuple_index.var] {}
// CHECK:STDOUT:   %int_0.loc8_37: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc8_37.3: ref %i32 = array_index %.loc8_1, %int_0.loc8_37
// CHECK:STDOUT:   %.loc8_37.4: init %i32 = initialize_from %.loc8_37.2 to %.loc8_37.3 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc8_37.5: init %array_type = array_init (%.loc8_37.4) to %.loc8_1 [concrete = constants.%array]
// CHECK:STDOUT:   %.loc8_39: init %array_type = converted %.loc8_37.1, %.loc8_37.5 [concrete = constants.%array]
// CHECK:STDOUT:   assign file.%tuple_index.var, %.loc8_39
// CHECK:STDOUT:   %int_0.loc10_37: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc10_39.1: %tuple.type.985 = tuple_literal (%int_0.loc10_37)
// CHECK:STDOUT:   %int_32.loc10: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc10: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %int_3.loc10: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %int_1.loc10: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc10_70.1: %struct_type.a.b = struct_literal (%int_3.loc10, %int_1.loc10)
// CHECK:STDOUT:   %struct.loc10: %struct_type.a.b = struct_value (%int_3.loc10, %int_1.loc10) [concrete = constants.%struct.a81]
// CHECK:STDOUT:   %.loc10_70.2: %struct_type.a.b = converted %.loc10_70.1, %struct.loc10 [concrete = constants.%struct.a81]
// CHECK:STDOUT:   %.loc10_71: Core.IntLiteral = struct_access %.loc10_70.2, element1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %array_type.loc10: type = array_type %.loc10_71, %i32.loc10 [concrete = constants.%array_type]
// CHECK:STDOUT:   %impl.elem0.loc10: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc10_39.1: <bound method> = bound_method %int_0.loc10_37, %impl.elem0.loc10 [concrete = constants.%Convert.bound.d04]
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_39.2: <bound method> = bound_method %int_0.loc10_37, %specific_fn.loc10 [concrete = constants.%bound_method.b6e]
// CHECK:STDOUT:   %int.convert_checked.loc10: init %i32 = call %bound_method.loc10_39.2(%int_0.loc10_37) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc10_39.2: init %i32 = converted %int_0.loc10_37, %int.convert_checked.loc10 [concrete = constants.%int_0.6a9]
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
// CHECK:STDOUT:   %int_5.64b: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %int_7.29f: Core.IntLiteral = int_value 7 [concrete]
// CHECK:STDOUT:   %int_9.988: Core.IntLiteral = int_value 9 [concrete]
// CHECK:STDOUT:   %tuple.type.d46: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %array_type.f32: type = array_type %int_4, %i32 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.0f9: type = fn_type @Convert.2, @impl.4f9(%To) [symbolic]
// CHECK:STDOUT:   %Convert.f06: %Convert.type.0f9 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c75: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a2f, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.c75) [concrete]
// CHECK:STDOUT:   %.9c3: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound.4e6: <bound method> = bound_method %int_5.64b, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.a25: <bound method> = bound_method %int_5.64b, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_5.0f6: %i32 = int_value 5 [concrete]
// CHECK:STDOUT:   %Convert.bound.208: <bound method> = bound_method %int_7.29f, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.3bd: <bound method> = bound_method %int_7.29f, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_7.0b1: %i32 = int_value 7 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %Convert.bound.ab5: <bound method> = bound_method %int_1.5b8, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.9a1: <bound method> = bound_method %int_1.5b8, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %Convert.bound.9e2: <bound method> = bound_method %int_9.988, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.cd3: <bound method> = bound_method %int_9.988, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_9.f88: %i32 = int_value 9 [concrete]
// CHECK:STDOUT:   %array: %array_type.f32 = tuple_value (%int_5.0f6, %int_7.0b1, %int_1.5d2, %int_9.f88) [concrete]
// CHECK:STDOUT:   %Convert.bound.ef9: <bound method> = bound_method %int_2.ecc, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.b92: <bound method> = bound_method %int_2.ecc, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.a5b: @impl.4f9.%Convert.type (%Convert.type.0f9) = import_ref Core//prelude/parts/int, loc16_39, loaded [symbolic = @impl.4f9.%Convert (constants.%Convert.f06)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a2f = impl_witness_table (%Core.import_ref.a5b), @impl.4f9 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %array_index.patt: %pattern_type.a98 = binding_pattern array_index [concrete]
// CHECK:STDOUT:     %array_index.var_patt: %pattern_type.a98 = var_pattern %array_index.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %array_index.var: ref %array_type.0cb = var %array_index.var_patt [concrete]
// CHECK:STDOUT:   %.loc9: type = splice_block %array_type [concrete = constants.%array_type.0cb] {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %array_type: type = array_type %int_1, %i32 [concrete = constants.%array_type.0cb]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %array_index: ref %array_type.0cb = bind_name array_index, %array_index.var [concrete = %array_index.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_0.loc9_35: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc9_37: %tuple.type.985 = tuple_literal (%int_0.loc9_35)
// CHECK:STDOUT:   %int_32.loc9_48: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc9_48: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5.64b]
// CHECK:STDOUT:   %int_7: Core.IntLiteral = int_value 7 [concrete = constants.%int_7.29f]
// CHECK:STDOUT:   %int_1.loc9_61: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_9: Core.IntLiteral = int_value 9 [concrete = constants.%int_9.988]
// CHECK:STDOUT:   %.loc9_65.1: %tuple.type.d46 = tuple_literal (%int_5, %int_7, %int_1.loc9_61, %int_9)
// CHECK:STDOUT:   %int_32.loc9_76: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc9_76: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4]
// CHECK:STDOUT:   %array_type: type = array_type %int_4, %i32.loc9_76 [concrete = constants.%array_type.f32]
// CHECK:STDOUT:   %impl.elem0.loc9_65.1: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc9_65.1: <bound method> = bound_method %int_5, %impl.elem0.loc9_65.1 [concrete = constants.%Convert.bound.4e6]
// CHECK:STDOUT:   %specific_fn.loc9_65.1: <specific function> = specific_function %impl.elem0.loc9_65.1, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_65.2: <bound method> = bound_method %int_5, %specific_fn.loc9_65.1 [concrete = constants.%bound_method.a25]
// CHECK:STDOUT:   %int.convert_checked.loc9_65.1: init %i32 = call %bound_method.loc9_65.2(%int_5) [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc9_65.2: init %i32 = converted %int_5, %int.convert_checked.loc9_65.1 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc9_65.3: ref %array_type.f32 = temporary_storage
// CHECK:STDOUT:   %int_0.loc9_65: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc9_65.4: ref %i32 = array_index %.loc9_65.3, %int_0.loc9_65
// CHECK:STDOUT:   %.loc9_65.5: init %i32 = initialize_from %.loc9_65.2 to %.loc9_65.4 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %impl.elem0.loc9_65.2: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc9_65.3: <bound method> = bound_method %int_7, %impl.elem0.loc9_65.2 [concrete = constants.%Convert.bound.208]
// CHECK:STDOUT:   %specific_fn.loc9_65.2: <specific function> = specific_function %impl.elem0.loc9_65.2, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_65.4: <bound method> = bound_method %int_7, %specific_fn.loc9_65.2 [concrete = constants.%bound_method.3bd]
// CHECK:STDOUT:   %int.convert_checked.loc9_65.2: init %i32 = call %bound_method.loc9_65.4(%int_7) [concrete = constants.%int_7.0b1]
// CHECK:STDOUT:   %.loc9_65.6: init %i32 = converted %int_7, %int.convert_checked.loc9_65.2 [concrete = constants.%int_7.0b1]
// CHECK:STDOUT:   %int_1.loc9_65: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc9_65.7: ref %i32 = array_index %.loc9_65.3, %int_1.loc9_65
// CHECK:STDOUT:   %.loc9_65.8: init %i32 = initialize_from %.loc9_65.6 to %.loc9_65.7 [concrete = constants.%int_7.0b1]
// CHECK:STDOUT:   %impl.elem0.loc9_65.3: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc9_65.5: <bound method> = bound_method %int_1.loc9_61, %impl.elem0.loc9_65.3 [concrete = constants.%Convert.bound.ab5]
// CHECK:STDOUT:   %specific_fn.loc9_65.3: <specific function> = specific_function %impl.elem0.loc9_65.3, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_65.6: <bound method> = bound_method %int_1.loc9_61, %specific_fn.loc9_65.3 [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:   %int.convert_checked.loc9_65.3: init %i32 = call %bound_method.loc9_65.6(%int_1.loc9_61) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc9_65.9: init %i32 = converted %int_1.loc9_61, %int.convert_checked.loc9_65.3 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %int_2.loc9_65: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc9_65.10: ref %i32 = array_index %.loc9_65.3, %int_2.loc9_65
// CHECK:STDOUT:   %.loc9_65.11: init %i32 = initialize_from %.loc9_65.9 to %.loc9_65.10 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc9_65.4: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc9_65.7: <bound method> = bound_method %int_9, %impl.elem0.loc9_65.4 [concrete = constants.%Convert.bound.9e2]
// CHECK:STDOUT:   %specific_fn.loc9_65.4: <specific function> = specific_function %impl.elem0.loc9_65.4, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_65.8: <bound method> = bound_method %int_9, %specific_fn.loc9_65.4 [concrete = constants.%bound_method.cd3]
// CHECK:STDOUT:   %int.convert_checked.loc9_65.4: init %i32 = call %bound_method.loc9_65.8(%int_9) [concrete = constants.%int_9.f88]
// CHECK:STDOUT:   %.loc9_65.12: init %i32 = converted %int_9, %int.convert_checked.loc9_65.4 [concrete = constants.%int_9.f88]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:   %.loc9_65.13: ref %i32 = array_index %.loc9_65.3, %int_3
// CHECK:STDOUT:   %.loc9_65.14: init %i32 = initialize_from %.loc9_65.12 to %.loc9_65.13 [concrete = constants.%int_9.f88]
// CHECK:STDOUT:   %.loc9_65.15: init %array_type.f32 = array_init (%.loc9_65.5, %.loc9_65.8, %.loc9_65.11, %.loc9_65.14) to %.loc9_65.3 [concrete = constants.%array]
// CHECK:STDOUT:   %.loc9_67.1: init %array_type.f32 = converted %.loc9_65.1, %.loc9_65.15 [concrete = constants.%array]
// CHECK:STDOUT:   %int_2.loc9_85: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc9_67.2: ref %array_type.f32 = temporary %.loc9_65.3, %.loc9_67.1
// CHECK:STDOUT:   %int_32.loc9_86: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc9_86: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc9_85: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc9_85.1: <bound method> = bound_method %int_2.loc9_85, %impl.elem0.loc9_85 [concrete = constants.%Convert.bound.ef9]
// CHECK:STDOUT:   %specific_fn.loc9_85: <specific function> = specific_function %impl.elem0.loc9_85, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_85.2: <bound method> = bound_method %int_2.loc9_85, %specific_fn.loc9_85 [concrete = constants.%bound_method.b92]
// CHECK:STDOUT:   %int.convert_checked.loc9_85: init %i32 = call %bound_method.loc9_85.2(%int_2.loc9_85) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc9_85.1: %i32 = value_of_initializer %int.convert_checked.loc9_85 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc9_85.2: %i32 = converted %int_2.loc9_85, %.loc9_85.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc9_86.1: ref %i32 = array_index %.loc9_67.2, %.loc9_85.2
// CHECK:STDOUT:   %.loc9_86.2: %i32 = bind_value %.loc9_86.1
// CHECK:STDOUT:   assign file.%array_index.var, <error>
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- symbolic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %ptr.79f: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %const: type = const_type %T [symbolic]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.4f2: type = tuple_type (%ptr.79f, %const) [symbolic]
// CHECK:STDOUT:   %ptr.6cd: type = ptr_type %tuple.type.4f2 [symbolic]
// CHECK:STDOUT:   %require_complete.155: <witness> = require_complete_type %tuple.type.4f2 [symbolic]
// CHECK:STDOUT:   %pattern_type.973: type = pattern_type %tuple.type.4f2 [symbolic]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %T} [symbolic]
// CHECK:STDOUT:   %require_complete.28a: <witness> = require_complete_type %struct_type.a [symbolic]
// CHECK:STDOUT:   %pattern_type.95a: type = pattern_type %struct_type.a [symbolic]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %array_type.ec2: type = array_type %int_5, %T [symbolic]
// CHECK:STDOUT:   %ptr.1a0: type = ptr_type %array_type.ec2 [symbolic]
// CHECK:STDOUT:   %require_complete.fe1: <witness> = require_complete_type %array_type.ec2 [symbolic]
// CHECK:STDOUT:   %pattern_type.035: type = pattern_type %array_type.ec2 [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Op.type.bae: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %require_complete.b09: <witness> = require_complete_type %ptr.1a0 [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.b45: <witness> = lookup_impl_witness %array_type.ec2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.0d3: %Destroy.type = facet_value %array_type.ec2, (%Destroy.lookup_impl_witness.b45) [symbolic]
// CHECK:STDOUT:   %.03f: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet.0d3 [symbolic]
// CHECK:STDOUT:   %impl.elem0.1ce: %.03f = impl_witness_access %Destroy.lookup_impl_witness.b45, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.236: <specific function> = specific_impl_function %impl.elem0.1ce, @Op.1(%Destroy.facet.0d3) [symbolic]
// CHECK:STDOUT:   %ptr.48a: type = ptr_type %struct_type.a [symbolic]
// CHECK:STDOUT:   %require_complete.86d: <witness> = require_complete_type %ptr.48a [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.b34: <witness> = lookup_impl_witness %struct_type.a, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.2c9: %Destroy.type = facet_value %struct_type.a, (%Destroy.lookup_impl_witness.b34) [symbolic]
// CHECK:STDOUT:   %.b85: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet.2c9 [symbolic]
// CHECK:STDOUT:   %impl.elem0.dc9: %.b85 = impl_witness_access %Destroy.lookup_impl_witness.b34, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.2a6: <specific function> = specific_impl_function %impl.elem0.dc9, @Op.1(%Destroy.facet.2c9) [symbolic]
// CHECK:STDOUT:   %require_complete.66e: <witness> = require_complete_type %ptr.6cd [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.a8a: <witness> = lookup_impl_witness %tuple.type.4f2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.7be: %Destroy.type = facet_value %tuple.type.4f2, (%Destroy.lookup_impl_witness.a8a) [symbolic]
// CHECK:STDOUT:   %.f5c: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet.7be [symbolic]
// CHECK:STDOUT:   %impl.elem0.87c: %.f5c = impl_witness_access %Destroy.lookup_impl_witness.a8a, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.4b8: <specific function> = specific_impl_function %impl.elem0.87c, @Op.1(%Destroy.facet.7be) [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %N.51e: %i32 = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.2fd: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %Convert.type.71e: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %From: Core.IntLiteral = bind_symbolic_name From, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.708: type = fn_type @Convert.3, @impl.971(%From) [symbolic]
// CHECK:STDOUT:   %Convert.c68: %Convert.type.708 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a11: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1d9, @impl.971(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.4ad: type = fn_type @Convert.3, @impl.971(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.960: %Convert.type.4ad = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.2fd = facet_value %i32, (%ImplicitAs.impl_witness.a11) [concrete]
// CHECK:STDOUT:   %.0ea: type = fn_type_with_self_type %Convert.type.71e, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %N.51e, %Convert.960 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.960, @Convert.3(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %N.51e, %Convert.specific_fn [symbolic]
// CHECK:STDOUT:   %int.convert_checked: init Core.IntLiteral = call %bound_method(%N.51e) [symbolic]
// CHECK:STDOUT:   %array_type.c7c: type = array_type %int.convert_checked, %i32 [symbolic]
// CHECK:STDOUT:   %ptr.0ad: type = ptr_type %array_type.c7c [symbolic]
// CHECK:STDOUT:   %require_complete.7cb: <witness> = require_complete_type %array_type.c7c [symbolic]
// CHECK:STDOUT:   %pattern_type.ccc: type = pattern_type %array_type.c7c [symbolic]
// CHECK:STDOUT:   %require_complete.7d5: <witness> = require_complete_type %ptr.0ad [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.4fa: <witness> = lookup_impl_witness %array_type.c7c, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.ad4: %Destroy.type = facet_value %array_type.c7c, (%Destroy.lookup_impl_witness.4fa) [symbolic]
// CHECK:STDOUT:   %.a57: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet.ad4 [symbolic]
// CHECK:STDOUT:   %impl.elem0.f15: %.a57 = impl_witness_access %Destroy.lookup_impl_witness.4fa, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.bae: <specific function> = specific_impl_function %impl.elem0.f15, @Op.1(%Destroy.facet.ad4) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.85c: @impl.971.%Convert.type (%Convert.type.708) = import_ref Core//prelude/parts/int, loc20_44, loaded [symbolic = @impl.971.%Convert (constants.%Convert.c68)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1d9 = impl_witness_table (%Core.import_ref.85c), @impl.971 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc4_6.1: type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ptr.loc6_12.2: type = ptr_type %T.loc4_6.2 [symbolic = %ptr.loc6_12.2 (constants.%ptr.79f)]
// CHECK:STDOUT:   %const.loc6_15.2: type = const_type %T.loc4_6.2 [symbolic = %const.loc6_15.2 (constants.%const)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%ptr.loc6_12.2, %const.loc6_15.2) [symbolic = %tuple.type (constants.%tuple.type.4f2)]
// CHECK:STDOUT:   %require_complete.loc6_22: <witness> = require_complete_type %tuple.type [symbolic = %require_complete.loc6_22 (constants.%require_complete.155)]
// CHECK:STDOUT:   %pattern_type.loc6: type = pattern_type %tuple.type [symbolic = %pattern_type.loc6 (constants.%pattern_type.973)]
// CHECK:STDOUT:   %struct_type.a.loc7_16.2: type = struct_type {.a: @F.%T.loc4_6.2 (%T)} [symbolic = %struct_type.a.loc7_16.2 (constants.%struct_type.a)]
// CHECK:STDOUT:   %require_complete.loc7_16: <witness> = require_complete_type %struct_type.a.loc7_16.2 [symbolic = %require_complete.loc7_16 (constants.%require_complete.28a)]
// CHECK:STDOUT:   %pattern_type.loc7: type = pattern_type %struct_type.a.loc7_16.2 [symbolic = %pattern_type.loc7 (constants.%pattern_type.95a)]
// CHECK:STDOUT:   %array_type.loc8_20.2: type = array_type constants.%int_5, %T.loc4_6.2 [symbolic = %array_type.loc8_20.2 (constants.%array_type.ec2)]
// CHECK:STDOUT:   %require_complete.loc8_20: <witness> = require_complete_type %array_type.loc8_20.2 [symbolic = %require_complete.loc8_20 (constants.%require_complete.fe1)]
// CHECK:STDOUT:   %pattern_type.loc8: type = pattern_type %array_type.loc8_20.2 [symbolic = %pattern_type.loc8 (constants.%pattern_type.035)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc8: <witness> = lookup_impl_witness %array_type.loc8_20.2, @Destroy [symbolic = %Destroy.lookup_impl_witness.loc8 (constants.%Destroy.lookup_impl_witness.b45)]
// CHECK:STDOUT:   %Destroy.facet.loc8: %Destroy.type = facet_value %array_type.loc8_20.2, (%Destroy.lookup_impl_witness.loc8) [symbolic = %Destroy.facet.loc8 (constants.%Destroy.facet.0d3)]
// CHECK:STDOUT:   %.loc8_3.2: type = fn_type_with_self_type constants.%Op.type.bae, %Destroy.facet.loc8 [symbolic = %.loc8_3.2 (constants.%.03f)]
// CHECK:STDOUT:   %impl.elem0.loc8_3.2: @F.%.loc8_3.2 (%.03f) = impl_witness_access %Destroy.lookup_impl_witness.loc8, element0 [symbolic = %impl.elem0.loc8_3.2 (constants.%impl.elem0.1ce)]
// CHECK:STDOUT:   %specific_impl_fn.loc8_3.2: <specific function> = specific_impl_function %impl.elem0.loc8_3.2, @Op.1(%Destroy.facet.loc8) [symbolic = %specific_impl_fn.loc8_3.2 (constants.%specific_impl_fn.236)]
// CHECK:STDOUT:   %ptr.loc8: type = ptr_type %array_type.loc8_20.2 [symbolic = %ptr.loc8 (constants.%ptr.1a0)]
// CHECK:STDOUT:   %require_complete.loc8_3: <witness> = require_complete_type %ptr.loc8 [symbolic = %require_complete.loc8_3 (constants.%require_complete.b09)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc7: <witness> = lookup_impl_witness %struct_type.a.loc7_16.2, @Destroy [symbolic = %Destroy.lookup_impl_witness.loc7 (constants.%Destroy.lookup_impl_witness.b34)]
// CHECK:STDOUT:   %Destroy.facet.loc7: %Destroy.type = facet_value %struct_type.a.loc7_16.2, (%Destroy.lookup_impl_witness.loc7) [symbolic = %Destroy.facet.loc7 (constants.%Destroy.facet.2c9)]
// CHECK:STDOUT:   %.loc7_3.2: type = fn_type_with_self_type constants.%Op.type.bae, %Destroy.facet.loc7 [symbolic = %.loc7_3.2 (constants.%.b85)]
// CHECK:STDOUT:   %impl.elem0.loc7_3.2: @F.%.loc7_3.2 (%.b85) = impl_witness_access %Destroy.lookup_impl_witness.loc7, element0 [symbolic = %impl.elem0.loc7_3.2 (constants.%impl.elem0.dc9)]
// CHECK:STDOUT:   %specific_impl_fn.loc7_3.2: <specific function> = specific_impl_function %impl.elem0.loc7_3.2, @Op.1(%Destroy.facet.loc7) [symbolic = %specific_impl_fn.loc7_3.2 (constants.%specific_impl_fn.2a6)]
// CHECK:STDOUT:   %ptr.loc7: type = ptr_type %struct_type.a.loc7_16.2 [symbolic = %ptr.loc7 (constants.%ptr.48a)]
// CHECK:STDOUT:   %require_complete.loc7_3: <witness> = require_complete_type %ptr.loc7 [symbolic = %require_complete.loc7_3 (constants.%require_complete.86d)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc6: <witness> = lookup_impl_witness %tuple.type, @Destroy [symbolic = %Destroy.lookup_impl_witness.loc6 (constants.%Destroy.lookup_impl_witness.a8a)]
// CHECK:STDOUT:   %Destroy.facet.loc6: %Destroy.type = facet_value %tuple.type, (%Destroy.lookup_impl_witness.loc6) [symbolic = %Destroy.facet.loc6 (constants.%Destroy.facet.7be)]
// CHECK:STDOUT:   %.loc6_3.2: type = fn_type_with_self_type constants.%Op.type.bae, %Destroy.facet.loc6 [symbolic = %.loc6_3.2 (constants.%.f5c)]
// CHECK:STDOUT:   %impl.elem0.loc6_3.2: @F.%.loc6_3.2 (%.f5c) = impl_witness_access %Destroy.lookup_impl_witness.loc6, element0 [symbolic = %impl.elem0.loc6_3.2 (constants.%impl.elem0.87c)]
// CHECK:STDOUT:   %specific_impl_fn.loc6_3.2: <specific function> = specific_impl_function %impl.elem0.loc6_3.2, @Op.1(%Destroy.facet.loc6) [symbolic = %specific_impl_fn.loc6_3.2 (constants.%specific_impl_fn.4b8)]
// CHECK:STDOUT:   %ptr.loc6_3: type = ptr_type %tuple.type [symbolic = %ptr.loc6_3 (constants.%ptr.6cd)]
// CHECK:STDOUT:   %require_complete.loc6_3: <witness> = require_complete_type %ptr.loc6_3 [symbolic = %require_complete.loc6_3 (constants.%require_complete.66e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %u.patt: @F.%pattern_type.loc6 (%pattern_type.973) = binding_pattern u [concrete]
// CHECK:STDOUT:       %u.var_patt: @F.%pattern_type.loc6 (%pattern_type.973) = var_pattern %u.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u.var: ref @F.%tuple.type (%tuple.type.4f2) = var %u.var_patt
// CHECK:STDOUT:     %.loc6_22.1: type = splice_block %.loc6_22.3 [symbolic = %tuple.type (constants.%tuple.type.4f2)] {
// CHECK:STDOUT:       %T.ref.loc6_11: type = name_ref T, %T.loc4_6.1 [symbolic = %T.loc4_6.2 (constants.%T)]
// CHECK:STDOUT:       %ptr.loc6_12.1: type = ptr_type %T.ref.loc6_11 [symbolic = %ptr.loc6_12.2 (constants.%ptr.79f)]
// CHECK:STDOUT:       %T.ref.loc6_21: type = name_ref T, %T.loc4_6.1 [symbolic = %T.loc4_6.2 (constants.%T)]
// CHECK:STDOUT:       %const.loc6_15.1: type = const_type %T.ref.loc6_21 [symbolic = %const.loc6_15.2 (constants.%const)]
// CHECK:STDOUT:       %.loc6_22.2: %tuple.type.24b = tuple_literal (%ptr.loc6_12.1, %const.loc6_15.1)
// CHECK:STDOUT:       %.loc6_22.3: type = converted %.loc6_22.2, constants.%tuple.type.4f2 [symbolic = %tuple.type (constants.%tuple.type.4f2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: ref @F.%tuple.type (%tuple.type.4f2) = bind_name u, %u.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %v.patt: @F.%pattern_type.loc7 (%pattern_type.95a) = binding_pattern v [concrete]
// CHECK:STDOUT:       %v.var_patt: @F.%pattern_type.loc7 (%pattern_type.95a) = var_pattern %v.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v.var: ref @F.%struct_type.a.loc7_16.2 (%struct_type.a) = var %v.var_patt
// CHECK:STDOUT:     %.loc7_16: type = splice_block %struct_type.a.loc7_16.1 [symbolic = %struct_type.a.loc7_16.2 (constants.%struct_type.a)] {
// CHECK:STDOUT:       %T.ref.loc7: type = name_ref T, %T.loc4_6.1 [symbolic = %T.loc4_6.2 (constants.%T)]
// CHECK:STDOUT:       %struct_type.a.loc7_16.1: type = struct_type {.a: @F.%T.loc4_6.2 (%T)} [symbolic = %struct_type.a.loc7_16.2 (constants.%struct_type.a)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: ref @F.%struct_type.a.loc7_16.2 (%struct_type.a) = bind_name v, %v.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %w.patt: @F.%pattern_type.loc8 (%pattern_type.035) = binding_pattern w [concrete]
// CHECK:STDOUT:       %w.var_patt: @F.%pattern_type.loc8 (%pattern_type.035) = var_pattern %w.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %w.var: ref @F.%array_type.loc8_20.2 (%array_type.ec2) = var %w.var_patt
// CHECK:STDOUT:     %.loc8_20: type = splice_block %array_type.loc8_20.1 [symbolic = %array_type.loc8_20.2 (constants.%array_type.ec2)] {
// CHECK:STDOUT:       %T.ref.loc8: type = name_ref T, %T.loc4_6.1 [symbolic = %T.loc4_6.2 (constants.%T)]
// CHECK:STDOUT:       %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5]
// CHECK:STDOUT:       %array_type.loc8_20.1: type = array_type %int_5, %T.ref.loc8 [symbolic = %array_type.loc8_20.2 (constants.%array_type.ec2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %w: ref @F.%array_type.loc8_20.2 (%array_type.ec2) = bind_name w, %w.var
// CHECK:STDOUT:     %impl.elem0.loc8_3.1: @F.%.loc8_3.2 (%.03f) = impl_witness_access constants.%Destroy.lookup_impl_witness.b45, element0 [symbolic = %impl.elem0.loc8_3.2 (constants.%impl.elem0.1ce)]
// CHECK:STDOUT:     %bound_method.loc8_3.1: <bound method> = bound_method %w.var, %impl.elem0.loc8_3.1
// CHECK:STDOUT:     %specific_impl_fn.loc8_3.1: <specific function> = specific_impl_function %impl.elem0.loc8_3.1, @Op.1(constants.%Destroy.facet.0d3) [symbolic = %specific_impl_fn.loc8_3.2 (constants.%specific_impl_fn.236)]
// CHECK:STDOUT:     %bound_method.loc8_3.2: <bound method> = bound_method %w.var, %specific_impl_fn.loc8_3.1
// CHECK:STDOUT:     %addr.loc8: @F.%ptr.loc8 (%ptr.1a0) = addr_of %w.var
// CHECK:STDOUT:     %.loc8_3.1: init %empty_tuple.type = call %bound_method.loc8_3.2(%addr.loc8)
// CHECK:STDOUT:     %impl.elem0.loc7_3.1: @F.%.loc7_3.2 (%.b85) = impl_witness_access constants.%Destroy.lookup_impl_witness.b34, element0 [symbolic = %impl.elem0.loc7_3.2 (constants.%impl.elem0.dc9)]
// CHECK:STDOUT:     %bound_method.loc7_3.1: <bound method> = bound_method %v.var, %impl.elem0.loc7_3.1
// CHECK:STDOUT:     %specific_impl_fn.loc7_3.1: <specific function> = specific_impl_function %impl.elem0.loc7_3.1, @Op.1(constants.%Destroy.facet.2c9) [symbolic = %specific_impl_fn.loc7_3.2 (constants.%specific_impl_fn.2a6)]
// CHECK:STDOUT:     %bound_method.loc7_3.2: <bound method> = bound_method %v.var, %specific_impl_fn.loc7_3.1
// CHECK:STDOUT:     %addr.loc7: @F.%ptr.loc7 (%ptr.48a) = addr_of %v.var
// CHECK:STDOUT:     %.loc7_3.1: init %empty_tuple.type = call %bound_method.loc7_3.2(%addr.loc7)
// CHECK:STDOUT:     %impl.elem0.loc6_3.1: @F.%.loc6_3.2 (%.f5c) = impl_witness_access constants.%Destroy.lookup_impl_witness.a8a, element0 [symbolic = %impl.elem0.loc6_3.2 (constants.%impl.elem0.87c)]
// CHECK:STDOUT:     %bound_method.loc6_3.1: <bound method> = bound_method %u.var, %impl.elem0.loc6_3.1
// CHECK:STDOUT:     %specific_impl_fn.loc6_3.1: <specific function> = specific_impl_function %impl.elem0.loc6_3.1, @Op.1(constants.%Destroy.facet.7be) [symbolic = %specific_impl_fn.loc6_3.2 (constants.%specific_impl_fn.4b8)]
// CHECK:STDOUT:     %bound_method.loc6_3.2: <bound method> = bound_method %u.var, %specific_impl_fn.loc6_3.1
// CHECK:STDOUT:     %addr.loc6: @F.%ptr.loc6_3 (%ptr.6cd) = addr_of %u.var
// CHECK:STDOUT:     %.loc6_3.1: init %empty_tuple.type = call %bound_method.loc6_3.2(%addr.loc6)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%N.loc12_6.1: %i32) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %N.loc12_6.2, constants.%Convert.960 [symbolic = %Convert.bound (constants.%Convert.bound)]
// CHECK:STDOUT:   %bound_method.loc14_21.3: <bound method> = bound_method %N.loc12_6.2, constants.%Convert.specific_fn [symbolic = %bound_method.loc14_21.3 (constants.%bound_method)]
// CHECK:STDOUT:   %int.convert_checked.loc14_21.2: init Core.IntLiteral = call %bound_method.loc14_21.3(%N.loc12_6.2) [symbolic = %int.convert_checked.loc14_21.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:   %array_type.loc14_22.2: type = array_type %int.convert_checked.loc14_21.2, constants.%i32 [symbolic = %array_type.loc14_22.2 (constants.%array_type.c7c)]
// CHECK:STDOUT:   %require_complete.loc14_22: <witness> = require_complete_type %array_type.loc14_22.2 [symbolic = %require_complete.loc14_22 (constants.%require_complete.7cb)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %array_type.loc14_22.2 [symbolic = %pattern_type (constants.%pattern_type.ccc)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %array_type.loc14_22.2, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.4fa)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %array_type.loc14_22.2, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet (constants.%Destroy.facet.ad4)]
// CHECK:STDOUT:   %.loc14_3.2: type = fn_type_with_self_type constants.%Op.type.bae, %Destroy.facet [symbolic = %.loc14_3.2 (constants.%.a57)]
// CHECK:STDOUT:   %impl.elem0.loc14_3.2: @G.%.loc14_3.2 (%.a57) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc14_3.2 (constants.%impl.elem0.f15)]
// CHECK:STDOUT:   %specific_impl_fn.loc14_3.2: <specific function> = specific_impl_function %impl.elem0.loc14_3.2, @Op.1(%Destroy.facet) [symbolic = %specific_impl_fn.loc14_3.2 (constants.%specific_impl_fn.bae)]
// CHECK:STDOUT:   %ptr: type = ptr_type %array_type.loc14_22.2 [symbolic = %ptr (constants.%ptr.0ad)]
// CHECK:STDOUT:   %require_complete.loc14_3: <witness> = require_complete_type %ptr [symbolic = %require_complete.loc14_3 (constants.%require_complete.7d5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %k.patt: @G.%pattern_type (%pattern_type.ccc) = binding_pattern k [concrete]
// CHECK:STDOUT:       %k.var_patt: @G.%pattern_type (%pattern_type.ccc) = var_pattern %k.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %k.var: ref @G.%array_type.loc14_22.2 (%array_type.c7c) = var %k.var_patt
// CHECK:STDOUT:     %.loc14_22: type = splice_block %array_type.loc14_22.1 [symbolic = %array_type.loc14_22.2 (constants.%array_type.c7c)] {
// CHECK:STDOUT:       %int_32.loc14: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc14: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %N.ref: %i32 = name_ref N, %N.loc12_6.1 [symbolic = %N.loc12_6.2 (constants.%N.51e)]
// CHECK:STDOUT:       %impl.elem0.loc14_21: %.0ea = impl_witness_access constants.%ImplicitAs.impl_witness.a11, element0 [concrete = constants.%Convert.960]
// CHECK:STDOUT:       %bound_method.loc14_21.1: <bound method> = bound_method %N.ref, %impl.elem0.loc14_21 [symbolic = %Convert.bound (constants.%Convert.bound)]
// CHECK:STDOUT:       %specific_fn: <specific function> = specific_function %impl.elem0.loc14_21, @Convert.3(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:       %bound_method.loc14_21.2: <bound method> = bound_method %N.ref, %specific_fn [symbolic = %bound_method.loc14_21.3 (constants.%bound_method)]
// CHECK:STDOUT:       %int.convert_checked.loc14_21.1: init Core.IntLiteral = call %bound_method.loc14_21.2(%N.ref) [symbolic = %int.convert_checked.loc14_21.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:       %.loc14_21.1: Core.IntLiteral = value_of_initializer %int.convert_checked.loc14_21.1 [symbolic = %int.convert_checked.loc14_21.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:       %.loc14_21.2: Core.IntLiteral = converted %N.ref, %.loc14_21.1 [symbolic = %int.convert_checked.loc14_21.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:       %array_type.loc14_22.1: type = array_type %.loc14_21.2, %i32.loc14 [symbolic = %array_type.loc14_22.2 (constants.%array_type.c7c)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %k: ref @G.%array_type.loc14_22.2 (%array_type.c7c) = bind_name k, %k.var
// CHECK:STDOUT:     %impl.elem0.loc14_3.1: @G.%.loc14_3.2 (%.a57) = impl_witness_access constants.%Destroy.lookup_impl_witness.4fa, element0 [symbolic = %impl.elem0.loc14_3.2 (constants.%impl.elem0.f15)]
// CHECK:STDOUT:     %bound_method.loc14_3.1: <bound method> = bound_method %k.var, %impl.elem0.loc14_3.1
// CHECK:STDOUT:     %specific_impl_fn.loc14_3.1: <specific function> = specific_impl_function %impl.elem0.loc14_3.1, @Op.1(constants.%Destroy.facet.ad4) [symbolic = %specific_impl_fn.loc14_3.2 (constants.%specific_impl_fn.bae)]
// CHECK:STDOUT:     %bound_method.loc14_3.2: <bound method> = bound_method %k.var, %specific_impl_fn.loc14_3.1
// CHECK:STDOUT:     %addr: @G.%ptr (%ptr.0ad) = addr_of %k.var
// CHECK:STDOUT:     %.loc14_3.1: init %empty_tuple.type = call %bound_method.loc14_3.2(%addr)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc4_6.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%N.51e) {
// CHECK:STDOUT:   %N.loc12_6.2 => constants.%N.51e
// CHECK:STDOUT: }
// CHECK:STDOUT:

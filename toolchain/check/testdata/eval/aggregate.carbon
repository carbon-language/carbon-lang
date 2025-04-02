// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/eval/aggregate.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/eval/aggregate.carbon

var tuple_copy: (i32, i32) = (1, 2) as (i32, i32);

var struct_copy: {.a: i32, .b: i32, .c: i32} = {.c = 3, .b = 2, .a = 1} as {.b: i32, .a: i32, .c: i32};

var tuple_index: array(i32, 1) = (0,) as array(i32, (5, 7, 1, 9).2);

var struct_access: array(i32, 1) = (0,) as array(i32, {.a = 3, .b = 1}.b);

// CHECK:STDOUT: --- aggregate.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.d07: type = tuple_type (%i32, %i32) [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.type.f94: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %impl_witness.d39: <witness> = impl_witness (imports.%Core.import_ref.a5b), @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%impl_witness.d39) [concrete]
// CHECK:STDOUT:   %.be7: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound.ab5: <bound method> = bound_method %int_1.5b8, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.9a1: <bound method> = bound_method %int_1.5b8, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Convert.bound.ef9: <bound method> = bound_method %int_2.ecc, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.b92: <bound method> = bound_method %int_2.ecc, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.21c: %tuple.type.d07 = tuple_value (%int_1.5d2, %int_2.ef8) [concrete]
// CHECK:STDOUT:   %struct_type.a.b.c: type = struct_type {.a: %i32, .b: %i32, .c: %i32} [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %struct_type.c.b.a: type = struct_type {.c: Core.IntLiteral, .b: Core.IntLiteral, .a: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct_type.b.a.c: type = struct_type {.b: %i32, .a: %i32, .c: %i32} [concrete]
// CHECK:STDOUT:   %Convert.bound.b30: <bound method> = bound_method %int_3.1ba, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.047: <bound method> = bound_method %int_3.1ba, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.822: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %struct.21d: %struct_type.b.a.c = struct_value (%int_2.ef8, %int_1.5d2, %int_3.822) [concrete]
// CHECK:STDOUT:   %struct.cff: %struct_type.a.b.c = struct_value (%int_1.5d2, %int_2.ef8, %int_3.822) [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_1.5b8, %i32 [concrete]
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
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .tuple_copy = %tuple_copy
// CHECK:STDOUT:     .struct_copy = %struct_copy
// CHECK:STDOUT:     .tuple_index = %tuple_index
// CHECK:STDOUT:     .struct_access = %struct_access
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %tuple_copy.patt: %tuple.type.d07 = binding_pattern tuple_copy
// CHECK:STDOUT:     %.loc11_1: %tuple.type.d07 = var_pattern %tuple_copy.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %tuple_copy.var: ref %tuple.type.d07 = var tuple_copy
// CHECK:STDOUT:   %.loc11_26.1: type = splice_block %.loc11_26.3 [concrete = constants.%tuple.type.d07] {
// CHECK:STDOUT:     %int_32.loc11_18: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc11_18: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc11_23: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc11_23: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc11_26.2: %tuple.type.24b = tuple_literal (%i32.loc11_18, %i32.loc11_23)
// CHECK:STDOUT:     %.loc11_26.3: type = converted %.loc11_26.2, constants.%tuple.type.d07 [concrete = constants.%tuple.type.d07]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %tuple_copy: ref %tuple.type.d07 = bind_name tuple_copy, %tuple_copy.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %struct_copy.patt: %struct_type.a.b.c = binding_pattern struct_copy
// CHECK:STDOUT:     %.loc13_1: %struct_type.a.b.c = var_pattern %struct_copy.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %struct_copy.var: ref %struct_type.a.b.c = var struct_copy
// CHECK:STDOUT:   %.loc13_44: type = splice_block %struct_type.a.b.c [concrete = constants.%struct_type.a.b.c] {
// CHECK:STDOUT:     %int_32.loc13_23: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc13_23: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc13_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc13_32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc13_41: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc13_41: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.a.b.c: type = struct_type {.a: %i32, .b: %i32, .c: %i32} [concrete = constants.%struct_type.a.b.c]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %struct_copy: ref %struct_type.a.b.c = bind_name struct_copy, %struct_copy.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %tuple_index.patt: %array_type = binding_pattern tuple_index
// CHECK:STDOUT:     %.loc15_1: %array_type = var_pattern %tuple_index.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %tuple_index.var: ref %array_type = var tuple_index
// CHECK:STDOUT:   %.loc15_30: type = splice_block %array_type.loc15 [concrete = constants.%array_type] {
// CHECK:STDOUT:     %int_32.loc15: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc15: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_1.loc15: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %array_type.loc15: type = array_type %int_1.loc15, %i32.loc15 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %tuple_index: ref %array_type = bind_name tuple_index, %tuple_index.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %struct_access.patt: %array_type = binding_pattern struct_access
// CHECK:STDOUT:     %.loc17_1: %array_type = var_pattern %struct_access.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %struct_access.var: ref %array_type = var struct_access
// CHECK:STDOUT:   %.loc17_32: type = splice_block %array_type.loc17 [concrete = constants.%array_type] {
// CHECK:STDOUT:     %int_32.loc17: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc17: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_1.loc17: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %array_type.loc17: type = array_type %int_1.loc17, %i32.loc17 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %struct_access: ref %array_type = bind_name struct_access, %struct_access.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1.loc11: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc11: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc11_35.1: %tuple.type.f94 = tuple_literal (%int_1.loc11, %int_2.loc11)
// CHECK:STDOUT:   %int_32.loc11_41: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc11_41: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %int_32.loc11_46: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc11_46: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc11_49.1: %tuple.type.24b = tuple_literal (%i32.loc11_41, %i32.loc11_46)
// CHECK:STDOUT:   %.loc11_49.2: type = converted %.loc11_49.1, constants.%tuple.type.d07 [concrete = constants.%tuple.type.d07]
// CHECK:STDOUT:   %impl.elem0.loc11_35.1: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc11_35.1: <bound method> = bound_method %int_1.loc11, %impl.elem0.loc11_35.1 [concrete = constants.%Convert.bound.ab5]
// CHECK:STDOUT:   %specific_fn.loc11_35.1: <specific function> = specific_function %impl.elem0.loc11_35.1, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_35.2: <bound method> = bound_method %int_1.loc11, %specific_fn.loc11_35.1 [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:   %int.convert_checked.loc11_35.1: init %i32 = call %bound_method.loc11_35.2(%int_1.loc11) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc11_35.2: %i32 = value_of_initializer %int.convert_checked.loc11_35.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc11_35.3: %i32 = converted %int_1.loc11, %.loc11_35.2 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc11_35.2: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc11_35.3: <bound method> = bound_method %int_2.loc11, %impl.elem0.loc11_35.2 [concrete = constants.%Convert.bound.ef9]
// CHECK:STDOUT:   %specific_fn.loc11_35.2: <specific function> = specific_function %impl.elem0.loc11_35.2, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_35.4: <bound method> = bound_method %int_2.loc11, %specific_fn.loc11_35.2 [concrete = constants.%bound_method.b92]
// CHECK:STDOUT:   %int.convert_checked.loc11_35.2: init %i32 = call %bound_method.loc11_35.4(%int_2.loc11) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc11_35.4: %i32 = value_of_initializer %int.convert_checked.loc11_35.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc11_35.5: %i32 = converted %int_2.loc11, %.loc11_35.4 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %tuple.loc11: %tuple.type.d07 = tuple_value (%.loc11_35.3, %.loc11_35.5) [concrete = constants.%tuple.21c]
// CHECK:STDOUT:   %.loc11_37.1: %tuple.type.d07 = converted %.loc11_35.1, %tuple.loc11 [concrete = constants.%tuple.21c]
// CHECK:STDOUT:   %tuple.elem0.loc11_37.1: %i32 = tuple_access %.loc11_37.1, element0 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %tuple.elem0.loc11_37.2: ref %i32 = tuple_access file.%tuple_copy.var, element0
// CHECK:STDOUT:   %.loc11_37.2: init %i32 = initialize_from %tuple.elem0.loc11_37.1 to %tuple.elem0.loc11_37.2 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %tuple.elem1.loc11_37.1: %i32 = tuple_access %.loc11_37.1, element1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %tuple.elem1.loc11_37.2: ref %i32 = tuple_access file.%tuple_copy.var, element1
// CHECK:STDOUT:   %.loc11_37.3: init %i32 = initialize_from %tuple.elem1.loc11_37.1 to %tuple.elem1.loc11_37.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc11_37.4: init %tuple.type.d07 = tuple_init (%.loc11_37.2, %.loc11_37.3) to file.%tuple_copy.var [concrete = constants.%tuple.21c]
// CHECK:STDOUT:   %.loc11_1: init %tuple.type.d07 = converted %.loc11_37.1, %.loc11_37.4 [concrete = constants.%tuple.21c]
// CHECK:STDOUT:   assign file.%tuple_copy.var, %.loc11_1
// CHECK:STDOUT:   %int_3.loc13: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %int_2.loc13: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %int_1.loc13: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc13_71.1: %struct_type.c.b.a = struct_literal (%int_3.loc13, %int_2.loc13, %int_1.loc13)
// CHECK:STDOUT:   %int_32.loc13_81: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc13_81: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %int_32.loc13_90: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc13_90: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %int_32.loc13_99: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc13_99: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %struct_type.b.a.c: type = struct_type {.b: %i32, .a: %i32, .c: %i32} [concrete = constants.%struct_type.b.a.c]
// CHECK:STDOUT:   %impl.elem0.loc13_71.1: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc13_71.1: <bound method> = bound_method %int_2.loc13, %impl.elem0.loc13_71.1 [concrete = constants.%Convert.bound.ef9]
// CHECK:STDOUT:   %specific_fn.loc13_71.1: <specific function> = specific_function %impl.elem0.loc13_71.1, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_71.2: <bound method> = bound_method %int_2.loc13, %specific_fn.loc13_71.1 [concrete = constants.%bound_method.b92]
// CHECK:STDOUT:   %int.convert_checked.loc13_71.1: init %i32 = call %bound_method.loc13_71.2(%int_2.loc13) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc13_71.2: %i32 = value_of_initializer %int.convert_checked.loc13_71.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc13_71.3: %i32 = converted %int_2.loc13, %.loc13_71.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %impl.elem0.loc13_71.2: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc13_71.3: <bound method> = bound_method %int_1.loc13, %impl.elem0.loc13_71.2 [concrete = constants.%Convert.bound.ab5]
// CHECK:STDOUT:   %specific_fn.loc13_71.2: <specific function> = specific_function %impl.elem0.loc13_71.2, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_71.4: <bound method> = bound_method %int_1.loc13, %specific_fn.loc13_71.2 [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:   %int.convert_checked.loc13_71.2: init %i32 = call %bound_method.loc13_71.4(%int_1.loc13) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc13_71.4: %i32 = value_of_initializer %int.convert_checked.loc13_71.2 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc13_71.5: %i32 = converted %int_1.loc13, %.loc13_71.4 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc13_71.3: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc13_71.5: <bound method> = bound_method %int_3.loc13, %impl.elem0.loc13_71.3 [concrete = constants.%Convert.bound.b30]
// CHECK:STDOUT:   %specific_fn.loc13_71.3: <specific function> = specific_function %impl.elem0.loc13_71.3, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_71.6: <bound method> = bound_method %int_3.loc13, %specific_fn.loc13_71.3 [concrete = constants.%bound_method.047]
// CHECK:STDOUT:   %int.convert_checked.loc13_71.3: init %i32 = call %bound_method.loc13_71.6(%int_3.loc13) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc13_71.6: %i32 = value_of_initializer %int.convert_checked.loc13_71.3 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc13_71.7: %i32 = converted %int_3.loc13, %.loc13_71.6 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %struct.loc13: %struct_type.b.a.c = struct_value (%.loc13_71.3, %.loc13_71.5, %.loc13_71.7) [concrete = constants.%struct.21d]
// CHECK:STDOUT:   %.loc13_73.1: %struct_type.b.a.c = converted %.loc13_71.1, %struct.loc13 [concrete = constants.%struct.21d]
// CHECK:STDOUT:   %.loc13_73.2: %i32 = struct_access %.loc13_73.1, element1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc13_73.3: ref %i32 = struct_access file.%struct_copy.var, element1
// CHECK:STDOUT:   %.loc13_73.4: init %i32 = initialize_from %.loc13_73.2 to %.loc13_73.3 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc13_73.5: %i32 = struct_access %.loc13_73.1, element0 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc13_73.6: ref %i32 = struct_access file.%struct_copy.var, element0
// CHECK:STDOUT:   %.loc13_73.7: init %i32 = initialize_from %.loc13_73.5 to %.loc13_73.6 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc13_73.8: %i32 = struct_access %.loc13_73.1, element2 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc13_73.9: ref %i32 = struct_access file.%struct_copy.var, element2
// CHECK:STDOUT:   %.loc13_73.10: init %i32 = initialize_from %.loc13_73.8 to %.loc13_73.9 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc13_73.11: init %struct_type.a.b.c = struct_init (%.loc13_73.4, %.loc13_73.7, %.loc13_73.10) to file.%struct_copy.var [concrete = constants.%struct.cff]
// CHECK:STDOUT:   %.loc13_1: init %struct_type.a.b.c = converted %.loc13_73.1, %.loc13_73.11 [concrete = constants.%struct.cff]
// CHECK:STDOUT:   assign file.%struct_copy.var, %.loc13_1
// CHECK:STDOUT:   %int_0.loc15_35: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc15_37.1: %tuple.type.985 = tuple_literal (%int_0.loc15_35)
// CHECK:STDOUT:   %int_32.loc15: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc15: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5]
// CHECK:STDOUT:   %int_7: Core.IntLiteral = int_value 7 [concrete = constants.%int_7]
// CHECK:STDOUT:   %int_1.loc15: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_9: Core.IntLiteral = int_value 9 [concrete = constants.%int_9]
// CHECK:STDOUT:   %.loc15_64.1: %tuple.type.d46 = tuple_literal (%int_5, %int_7, %int_1.loc15, %int_9)
// CHECK:STDOUT:   %int_2.loc15: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %tuple.loc15: %tuple.type.d46 = tuple_value (%int_5, %int_7, %int_1.loc15, %int_9) [concrete = constants.%tuple.869]
// CHECK:STDOUT:   %.loc15_64.2: %tuple.type.d46 = converted %.loc15_64.1, %tuple.loc15 [concrete = constants.%tuple.869]
// CHECK:STDOUT:   %tuple.elem2: Core.IntLiteral = tuple_access %.loc15_64.2, element2 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %array_type.loc15: type = array_type %tuple.elem2, %i32.loc15 [concrete = constants.%array_type]
// CHECK:STDOUT:   %impl.elem0.loc15: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc15_37.1: <bound method> = bound_method %int_0.loc15_35, %impl.elem0.loc15 [concrete = constants.%Convert.bound.d04]
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem0.loc15, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_37.2: <bound method> = bound_method %int_0.loc15_35, %specific_fn.loc15 [concrete = constants.%bound_method.b6e]
// CHECK:STDOUT:   %int.convert_checked.loc15: init %i32 = call %bound_method.loc15_37.2(%int_0.loc15_35) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc15_37.2: init %i32 = converted %int_0.loc15_35, %int.convert_checked.loc15 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc15_1: ref %array_type = splice_block file.%tuple_index.var {}
// CHECK:STDOUT:   %int_0.loc15_37: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc15_37.3: ref %i32 = array_index %.loc15_1, %int_0.loc15_37
// CHECK:STDOUT:   %.loc15_37.4: init %i32 = initialize_from %.loc15_37.2 to %.loc15_37.3 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc15_37.5: init %array_type = array_init (%.loc15_37.4) to %.loc15_1 [concrete = constants.%array]
// CHECK:STDOUT:   %.loc15_39: init %array_type = converted %.loc15_37.1, %.loc15_37.5 [concrete = constants.%array]
// CHECK:STDOUT:   assign file.%tuple_index.var, %.loc15_39
// CHECK:STDOUT:   %int_0.loc17_37: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc17_39.1: %tuple.type.985 = tuple_literal (%int_0.loc17_37)
// CHECK:STDOUT:   %int_32.loc17: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc17: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %int_3.loc17: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %int_1.loc17: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc17_70.1: %struct_type.a.b = struct_literal (%int_3.loc17, %int_1.loc17)
// CHECK:STDOUT:   %struct.loc17: %struct_type.a.b = struct_value (%int_3.loc17, %int_1.loc17) [concrete = constants.%struct.a81]
// CHECK:STDOUT:   %.loc17_70.2: %struct_type.a.b = converted %.loc17_70.1, %struct.loc17 [concrete = constants.%struct.a81]
// CHECK:STDOUT:   %.loc17_71: Core.IntLiteral = struct_access %.loc17_70.2, element1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %array_type.loc17: type = array_type %.loc17_71, %i32.loc17 [concrete = constants.%array_type]
// CHECK:STDOUT:   %impl.elem0.loc17: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc17_39.1: <bound method> = bound_method %int_0.loc17_37, %impl.elem0.loc17 [concrete = constants.%Convert.bound.d04]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem0.loc17, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc17_39.2: <bound method> = bound_method %int_0.loc17_37, %specific_fn.loc17 [concrete = constants.%bound_method.b6e]
// CHECK:STDOUT:   %int.convert_checked.loc17: init %i32 = call %bound_method.loc17_39.2(%int_0.loc17_37) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc17_39.2: init %i32 = converted %int_0.loc17_37, %int.convert_checked.loc17 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc17_1: ref %array_type = splice_block file.%struct_access.var {}
// CHECK:STDOUT:   %int_0.loc17_39: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc17_39.3: ref %i32 = array_index %.loc17_1, %int_0.loc17_39
// CHECK:STDOUT:   %.loc17_39.4: init %i32 = initialize_from %.loc17_39.2 to %.loc17_39.3 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc17_39.5: init %array_type = array_init (%.loc17_39.4) to %.loc17_1 [concrete = constants.%array]
// CHECK:STDOUT:   %.loc17_41: init %array_type = converted %.loc17_39.1, %.loc17_39.5 [concrete = constants.%array]
// CHECK:STDOUT:   assign file.%struct_access.var, %.loc17_41
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

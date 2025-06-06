// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/array/index_not_literal.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/array/index_not_literal.carbon

// --- function_param.carbon

fn F(arr: array(i32, 3), i: i32) -> i32 {
  //@dump-sem-ir-begin
  return arr[i];
  //@dump-sem-ir-end
}

fn G() -> i32 {
  //@dump-sem-ir-begin
  return F((1, 2, 3), 1);
  //@dump-sem-ir-end
}

// --- index_non_literal.carbon

library "[[@TEST_NAME]]";

fn F(a: array({}, 3)) -> {} {
  //@dump-sem-ir-begin
  return a[{.index = 2}.index];
  //@dump-sem-ir-end
}

// --- fail_out_of_bound_non_literal.carbon

library "[[@TEST_NAME]]";

fn F(a: array({}, 3)) -> {} {
  // CHECK:STDERR: fail_out_of_bound_non_literal.carbon:[[@LINE+4]]:12: error: array index `3` is past the end of type `array({}, 3)` [ArrayIndexOutOfBounds]
  // CHECK:STDERR:   return a[{.index = 3}.index];
  // CHECK:STDERR:            ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return a[{.index = 3}.index];
}

// CHECK:STDOUT: --- function_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_3.1ba, %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To.c80: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.0f9: type = fn_type @Convert.2, @impl.4f9(%To.c80) [symbolic]
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
// CHECK:STDOUT:   %Convert.bound.b30: <bound method> = bound_method %int_3.1ba, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.047: <bound method> = bound_method %int_3.1ba, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.822: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%int_1.5d2, %int_2.ef8, %int_3.822) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.a5b: @impl.4f9.%Convert.type (%Convert.type.0f9) = import_ref Core//prelude/types/int, loc19_39, loaded [symbolic = @impl.4f9.%Convert (constants.%Convert.f06)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a2f = impl_witness_table (%Core.import_ref.a5b), @impl.4f9 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%arr.param: %array_type, %i.param: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %arr.ref: %array_type = name_ref arr, %arr
// CHECK:STDOUT:   %i.ref: %i32 = name_ref i, %i
// CHECK:STDOUT:   %int_32.loc4: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc4: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc4_15.1: ref %array_type = value_as_ref %arr.ref
// CHECK:STDOUT:   %.loc4_15.2: ref %i32 = array_index %.loc4_15.1, %i.ref
// CHECK:STDOUT:   %.loc4_15.3: %i32 = bind_value %.loc4_15.2
// CHECK:STDOUT:   return %.loc4_15.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %int_1.loc10_13: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc10_16: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %.loc10_20.1: %tuple.type = tuple_literal (%int_1.loc10_13, %int_2.loc10_16, %int_3)
// CHECK:STDOUT:   %int_1.loc10_23: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc10_20.1: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc10_20.1: <bound method> = bound_method %int_1.loc10_13, %impl.elem0.loc10_20.1 [concrete = constants.%Convert.bound.ab5]
// CHECK:STDOUT:   %specific_fn.loc10_20.1: <specific function> = specific_function %impl.elem0.loc10_20.1, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_20.2: <bound method> = bound_method %int_1.loc10_13, %specific_fn.loc10_20.1 [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:   %int.convert_checked.loc10_20.1: init %i32 = call %bound_method.loc10_20.2(%int_1.loc10_13) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc10_20.2: init %i32 = converted %int_1.loc10_13, %int.convert_checked.loc10_20.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc10_20.3: ref %array_type = temporary_storage
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc10_20.4: ref %i32 = array_index %.loc10_20.3, %int_0
// CHECK:STDOUT:   %.loc10_20.5: init %i32 = initialize_from %.loc10_20.2 to %.loc10_20.4 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc10_20.2: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc10_20.3: <bound method> = bound_method %int_2.loc10_16, %impl.elem0.loc10_20.2 [concrete = constants.%Convert.bound.ef9]
// CHECK:STDOUT:   %specific_fn.loc10_20.2: <specific function> = specific_function %impl.elem0.loc10_20.2, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_20.4: <bound method> = bound_method %int_2.loc10_16, %specific_fn.loc10_20.2 [concrete = constants.%bound_method.b92]
// CHECK:STDOUT:   %int.convert_checked.loc10_20.2: init %i32 = call %bound_method.loc10_20.4(%int_2.loc10_16) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc10_20.6: init %i32 = converted %int_2.loc10_16, %int.convert_checked.loc10_20.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %int_1.loc10_20: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc10_20.7: ref %i32 = array_index %.loc10_20.3, %int_1.loc10_20
// CHECK:STDOUT:   %.loc10_20.8: init %i32 = initialize_from %.loc10_20.6 to %.loc10_20.7 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %impl.elem0.loc10_20.3: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc10_20.5: <bound method> = bound_method %int_3, %impl.elem0.loc10_20.3 [concrete = constants.%Convert.bound.b30]
// CHECK:STDOUT:   %specific_fn.loc10_20.3: <specific function> = specific_function %impl.elem0.loc10_20.3, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_20.6: <bound method> = bound_method %int_3, %specific_fn.loc10_20.3 [concrete = constants.%bound_method.047]
// CHECK:STDOUT:   %int.convert_checked.loc10_20.3: init %i32 = call %bound_method.loc10_20.6(%int_3) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc10_20.9: init %i32 = converted %int_3, %int.convert_checked.loc10_20.3 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %int_2.loc10_20: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc10_20.10: ref %i32 = array_index %.loc10_20.3, %int_2.loc10_20
// CHECK:STDOUT:   %.loc10_20.11: init %i32 = initialize_from %.loc10_20.9 to %.loc10_20.10 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc10_20.12: init %array_type = array_init (%.loc10_20.5, %.loc10_20.8, %.loc10_20.11) to %.loc10_20.3 [concrete = constants.%array]
// CHECK:STDOUT:   %.loc10_20.13: init %array_type = converted %.loc10_20.1, %.loc10_20.12 [concrete = constants.%array]
// CHECK:STDOUT:   %.loc10_20.14: ref %array_type = temporary %.loc10_20.3, %.loc10_20.13
// CHECK:STDOUT:   %.loc10_20.15: %array_type = bind_value %.loc10_20.14
// CHECK:STDOUT:   %impl.elem0.loc10_23: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc10_23.1: <bound method> = bound_method %int_1.loc10_23, %impl.elem0.loc10_23 [concrete = constants.%Convert.bound.ab5]
// CHECK:STDOUT:   %specific_fn.loc10_23: <specific function> = specific_function %impl.elem0.loc10_23, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_23.2: <bound method> = bound_method %int_1.loc10_23, %specific_fn.loc10_23 [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:   %int.convert_checked.loc10_23: init %i32 = call %bound_method.loc10_23.2(%int_1.loc10_23) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc10_23.1: %i32 = value_of_initializer %int.convert_checked.loc10_23 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc10_23.2: %i32 = converted %int_1.loc10_23, %.loc10_23.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %F.call: init %i32 = call %F.ref(%.loc10_20.15, %.loc10_23.2)
// CHECK:STDOUT:   %.loc10_25.1: %i32 = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc10_25.2: %i32 = converted %F.call, %.loc10_25.1
// CHECK:STDOUT:   return %.loc10_25.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- index_non_literal.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_3, %empty_struct_type [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %struct_type.index: type = struct_type {.index: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct: %struct_type.index = struct_value (%int_2.ecc) [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To.c80: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.0f9: type = fn_type @Convert.2, @impl.4f9(%To.c80) [symbolic]
// CHECK:STDOUT:   %Convert.f06: %Convert.type.0f9 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c75: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a2f, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.c75) [concrete]
// CHECK:STDOUT:   %.9c3: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_2.ecc, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_2.ecc, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.a5b: @impl.4f9.%Convert.type (%Convert.type.0f9) = import_ref Core//prelude/types/int, loc19_39, loaded [symbolic = @impl.4f9.%Convert (constants.%Convert.f06)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a2f = impl_witness_table (%Core.import_ref.a5b), @impl.4f9 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%a.param: %array_type) -> %empty_struct_type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %array_type = name_ref a, %a
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc6_23.1: %struct_type.index = struct_literal (%int_2)
// CHECK:STDOUT:   %struct: %struct_type.index = struct_value (%int_2) [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc6_23.2: %struct_type.index = converted %.loc6_23.1, %struct [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc6_24.1: Core.IntLiteral = struct_access %.loc6_23.2, element0 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc6_24.1: <bound method> = bound_method %.loc6_24.1, %impl.elem0 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_24.2: <bound method> = bound_method %.loc6_24.1, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %bound_method.loc6_24.2(%.loc6_24.1) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc6_24.2: %i32 = value_of_initializer %int.convert_checked [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc6_24.3: %i32 = converted %.loc6_24.1, %.loc6_24.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc6_30.1: ref %array_type = value_as_ref %a.ref
// CHECK:STDOUT:   %.loc6_30.2: ref %empty_struct_type = array_index %.loc6_30.1, %.loc6_24.3
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc6_30.3: %empty_struct_type = converted %.loc6_30.2, %empty_struct [concrete = constants.%empty_struct]
// CHECK:STDOUT:   return %.loc6_30.3
// CHECK:STDOUT: }
// CHECK:STDOUT:

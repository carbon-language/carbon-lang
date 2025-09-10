// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
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
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_3.1ba, %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.f01: type = ptr_type %array_type [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.49e: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f01: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.49e = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.acc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b6b, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.9ec: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.592: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.9ec = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.acc) [concrete]
// CHECK:STDOUT:   %.7ea: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.a02: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.592 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.592, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.b59: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.2d9: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.592 [concrete]
// CHECK:STDOUT:   %bound_method.f6f: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.c36: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.592 [concrete]
// CHECK:STDOUT:   %bound_method.f79: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.822: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%int_1.5d2, %int_2.ef8, %int_3.822) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.330: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%array_type) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.a44: %T.as.Destroy.impl.Op.type.330 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.428: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.49e) = import_ref Core//prelude/parts/int, loc16_39, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f01)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b6b = impl_witness_table (%Core.import_ref.428), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:   %impl.elem0.loc10_20.1: %.7ea = impl_witness_access constants.%ImplicitAs.impl_witness.acc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.592]
// CHECK:STDOUT:   %bound_method.loc10_20.1: <bound method> = bound_method %int_1.loc10_13, %impl.elem0.loc10_20.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.a02]
// CHECK:STDOUT:   %specific_fn.loc10_20.1: <specific function> = specific_function %impl.elem0.loc10_20.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_20.2: <bound method> = bound_method %int_1.loc10_13, %specific_fn.loc10_20.1 [concrete = constants.%bound_method.b59]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_20.1: init %i32 = call %bound_method.loc10_20.2(%int_1.loc10_13) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc10_20.2: init %i32 = converted %int_1.loc10_13, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_20.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc10_20.3: ref %array_type = temporary_storage
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc10_20.4: ref %i32 = array_index %.loc10_20.3, %int_0
// CHECK:STDOUT:   %.loc10_20.5: init %i32 = initialize_from %.loc10_20.2 to %.loc10_20.4 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc10_20.2: %.7ea = impl_witness_access constants.%ImplicitAs.impl_witness.acc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.592]
// CHECK:STDOUT:   %bound_method.loc10_20.3: <bound method> = bound_method %int_2.loc10_16, %impl.elem0.loc10_20.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.2d9]
// CHECK:STDOUT:   %specific_fn.loc10_20.2: <specific function> = specific_function %impl.elem0.loc10_20.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_20.4: <bound method> = bound_method %int_2.loc10_16, %specific_fn.loc10_20.2 [concrete = constants.%bound_method.f6f]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_20.2: init %i32 = call %bound_method.loc10_20.4(%int_2.loc10_16) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc10_20.6: init %i32 = converted %int_2.loc10_16, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_20.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %int_1.loc10_20: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc10_20.7: ref %i32 = array_index %.loc10_20.3, %int_1.loc10_20
// CHECK:STDOUT:   %.loc10_20.8: init %i32 = initialize_from %.loc10_20.6 to %.loc10_20.7 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %impl.elem0.loc10_20.3: %.7ea = impl_witness_access constants.%ImplicitAs.impl_witness.acc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.592]
// CHECK:STDOUT:   %bound_method.loc10_20.5: <bound method> = bound_method %int_3, %impl.elem0.loc10_20.3 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.c36]
// CHECK:STDOUT:   %specific_fn.loc10_20.3: <specific function> = specific_function %impl.elem0.loc10_20.3, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_20.6: <bound method> = bound_method %int_3, %specific_fn.loc10_20.3 [concrete = constants.%bound_method.f79]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_20.3: init %i32 = call %bound_method.loc10_20.6(%int_3) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc10_20.9: init %i32 = converted %int_3, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_20.3 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %int_2.loc10_20: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc10_20.10: ref %i32 = array_index %.loc10_20.3, %int_2.loc10_20
// CHECK:STDOUT:   %.loc10_20.11: init %i32 = initialize_from %.loc10_20.9 to %.loc10_20.10 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc10_20.12: init %array_type = array_init (%.loc10_20.5, %.loc10_20.8, %.loc10_20.11) to %.loc10_20.3 [concrete = constants.%array]
// CHECK:STDOUT:   %.loc10_20.13: init %array_type = converted %.loc10_20.1, %.loc10_20.12 [concrete = constants.%array]
// CHECK:STDOUT:   %.loc10_20.14: ref %array_type = temporary %.loc10_20.3, %.loc10_20.13
// CHECK:STDOUT:   %.loc10_20.15: %array_type = bind_value %.loc10_20.14
// CHECK:STDOUT:   %impl.elem0.loc10_23: %.7ea = impl_witness_access constants.%ImplicitAs.impl_witness.acc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.592]
// CHECK:STDOUT:   %bound_method.loc10_23.1: <bound method> = bound_method %int_1.loc10_23, %impl.elem0.loc10_23 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.a02]
// CHECK:STDOUT:   %specific_fn.loc10_23: <specific function> = specific_function %impl.elem0.loc10_23, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_23.2: <bound method> = bound_method %int_1.loc10_23, %specific_fn.loc10_23 [concrete = constants.%bound_method.b59]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_23: init %i32 = call %bound_method.loc10_23.2(%int_1.loc10_23) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc10_23.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_23 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc10_23.2: %i32 = converted %int_1.loc10_23, %.loc10_23.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %F.call: init %i32 = call %F.ref(%.loc10_20.15, %.loc10_23.2)
// CHECK:STDOUT:   %.loc10_25.1: %i32 = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc10_25.2: %i32 = converted %F.call, %.loc10_25.1
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc10_20.14, constants.%T.as.Destroy.impl.Op.a44
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10_20.7: <bound method> = bound_method %.loc10_20.14, %T.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.f01 = addr_of %.loc10_20.14
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc10_20.7(%addr)
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
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.49e: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f01: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.49e = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.acc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b6b, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.9ec: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.592: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.9ec = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.acc) [concrete]
// CHECK:STDOUT:   %.7ea: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.592 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.592, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.428: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.49e) = import_ref Core//prelude/parts/int, loc16_39, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f01)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b6b = impl_witness_table (%Core.import_ref.428), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:   %impl.elem0: %.7ea = impl_witness_access constants.%ImplicitAs.impl_witness.acc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.592]
// CHECK:STDOUT:   %bound_method.loc6_24.1: <bound method> = bound_method %.loc6_24.1, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_24.2: <bound method> = bound_method %.loc6_24.1, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc6_24.2(%.loc6_24.1) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc6_24.2: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc6_24.3: %i32 = converted %.loc6_24.1, %.loc6_24.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc6_30.1: ref %array_type = value_as_ref %a.ref
// CHECK:STDOUT:   %.loc6_30.2: ref %empty_struct_type = array_index %.loc6_30.1, %.loc6_24.3
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc6_30.3: %empty_struct_type = converted %.loc6_30.2, %empty_struct [concrete = constants.%empty_struct]
// CHECK:STDOUT:   return %.loc6_30.3
// CHECK:STDOUT: }
// CHECK:STDOUT:

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
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_3.1ba, %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.2f2: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.a64: %Int.as.Copy.impl.Op.type.2f2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.1a4: <witness> = impl_witness imports.%Copy.impl_witness_table.0ac, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.422: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.ce7: %Int.as.Copy.impl.Op.type.422 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.1a4) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.7e2: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.604: type = fn_type_with_self_type %Copy.WithSelf.Op.type.7e2, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.ce7, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.type.37f: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.2d5: %tuple.type.37f = tuple_value (%int_1.5b8, %int_2.ecc, %int_3.1ba) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.360: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.048: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e2c: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.048 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.3d6: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.0bd, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0a4: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0a4 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.360 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.3d6) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.da5: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.474: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.da5, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.c1a: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.e5b: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.d9c: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.bf7: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1 [concrete]
// CHECK:STDOUT:   %bound_method.7d5: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.228: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.8c0: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1 [concrete]
// CHECK:STDOUT:   %bound_method.655: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.6cc: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%int_1.d9c, %int_2.228, %int_3.6cc) [concrete]
// CHECK:STDOUT:   %.2d4: ref %array_type = temporary invalid, %array [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.3: type = fn_type @Destroy.Op.loc10_20.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.3: %Destroy.Op.type.790b3b.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.2d4, %Destroy.Op.2dd605.3 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.55f: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.2f2) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.a64)]
// CHECK:STDOUT:   %Copy.impl_witness_table.0ac = impl_witness_table (%Core.import_ref.55f), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.b42: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.048) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e2c)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.0bd = impl_witness_table (%Core.import_ref.b42), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%arr.param: %array_type, %i.param: %i32) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %arr.ref: %array_type = name_ref arr, %arr
// CHECK:STDOUT:   %i.ref: %i32 = name_ref i, %i
// CHECK:STDOUT:   %.loc4_15.1: ref %array_type = value_as_ref %arr.ref
// CHECK:STDOUT:   %.loc4_15.2: ref %i32 = array_index %.loc4_15.1, %i.ref
// CHECK:STDOUT:   %.loc4_15.3: %i32 = acquire_value %.loc4_15.2
// CHECK:STDOUT:   %impl.elem0: %.604 = impl_witness_access constants.%Copy.impl_witness.1a4, element0 [concrete = constants.%Int.as.Copy.impl.Op.ce7]
// CHECK:STDOUT:   %bound_method.loc4_15.1: <bound method> = bound_method %.loc4_15.3, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc4_15.2: <bound method> = bound_method %.loc4_15.3, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc4_15.2(%.loc4_15.3)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %int_1.loc10_13: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc10_16: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %.loc10_20.1: %tuple.type.37f = tuple_literal (%int_1.loc10_13, %int_2.loc10_16, %int_3) [concrete = constants.%tuple.2d5]
// CHECK:STDOUT:   %int_1.loc10_23: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc10_20.1: %.474 = impl_witness_access constants.%ImplicitAs.impl_witness.3d6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1]
// CHECK:STDOUT:   %bound_method.loc10_20.1: <bound method> = bound_method %int_1.loc10_13, %impl.elem0.loc10_20.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.c1a]
// CHECK:STDOUT:   %specific_fn.loc10_20.1: <specific function> = specific_function %impl.elem0.loc10_20.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_20.2: <bound method> = bound_method %int_1.loc10_13, %specific_fn.loc10_20.1 [concrete = constants.%bound_method.e5b]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_20.1: init %i32 = call %bound_method.loc10_20.2(%int_1.loc10_13) [concrete = constants.%int_1.d9c]
// CHECK:STDOUT:   %.loc10_20.2: init %i32 = converted %int_1.loc10_13, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_20.1 [concrete = constants.%int_1.d9c]
// CHECK:STDOUT:   %.loc10_20.3: ref %array_type = temporary_storage
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc10_20.4: ref %i32 = array_index %.loc10_20.3, %int_0
// CHECK:STDOUT:   %.loc10_20.5: init %i32 to %.loc10_20.4 = in_place_init %.loc10_20.2 [concrete = constants.%int_1.d9c]
// CHECK:STDOUT:   %impl.elem0.loc10_20.2: %.474 = impl_witness_access constants.%ImplicitAs.impl_witness.3d6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1]
// CHECK:STDOUT:   %bound_method.loc10_20.3: <bound method> = bound_method %int_2.loc10_16, %impl.elem0.loc10_20.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.bf7]
// CHECK:STDOUT:   %specific_fn.loc10_20.2: <specific function> = specific_function %impl.elem0.loc10_20.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_20.4: <bound method> = bound_method %int_2.loc10_16, %specific_fn.loc10_20.2 [concrete = constants.%bound_method.7d5]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_20.2: init %i32 = call %bound_method.loc10_20.4(%int_2.loc10_16) [concrete = constants.%int_2.228]
// CHECK:STDOUT:   %.loc10_20.6: init %i32 = converted %int_2.loc10_16, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_20.2 [concrete = constants.%int_2.228]
// CHECK:STDOUT:   %int_1.loc10_20: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc10_20.7: ref %i32 = array_index %.loc10_20.3, %int_1.loc10_20
// CHECK:STDOUT:   %.loc10_20.8: init %i32 to %.loc10_20.7 = in_place_init %.loc10_20.6 [concrete = constants.%int_2.228]
// CHECK:STDOUT:   %impl.elem0.loc10_20.3: %.474 = impl_witness_access constants.%ImplicitAs.impl_witness.3d6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1]
// CHECK:STDOUT:   %bound_method.loc10_20.5: <bound method> = bound_method %int_3, %impl.elem0.loc10_20.3 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.8c0]
// CHECK:STDOUT:   %specific_fn.loc10_20.3: <specific function> = specific_function %impl.elem0.loc10_20.3, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_20.6: <bound method> = bound_method %int_3, %specific_fn.loc10_20.3 [concrete = constants.%bound_method.655]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_20.3: init %i32 = call %bound_method.loc10_20.6(%int_3) [concrete = constants.%int_3.6cc]
// CHECK:STDOUT:   %.loc10_20.9: init %i32 = converted %int_3, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_20.3 [concrete = constants.%int_3.6cc]
// CHECK:STDOUT:   %int_2.loc10_20: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc10_20.10: ref %i32 = array_index %.loc10_20.3, %int_2.loc10_20
// CHECK:STDOUT:   %.loc10_20.11: init %i32 to %.loc10_20.10 = in_place_init %.loc10_20.9 [concrete = constants.%int_3.6cc]
// CHECK:STDOUT:   %.loc10_20.12: init %array_type to %.loc10_20.3 = array_init (%.loc10_20.5, %.loc10_20.8, %.loc10_20.11) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc10_20.13: init %array_type = converted %.loc10_20.1, %.loc10_20.12 [concrete = constants.%array]
// CHECK:STDOUT:   %.loc10_20.14: ref %array_type = temporary %.loc10_20.3, %.loc10_20.13 [concrete = constants.%.2d4]
// CHECK:STDOUT:   %.loc10_20.15: %array_type = acquire_value %.loc10_20.14 [concrete = constants.%array]
// CHECK:STDOUT:   %impl.elem0.loc10_23: %.474 = impl_witness_access constants.%ImplicitAs.impl_witness.3d6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1]
// CHECK:STDOUT:   %bound_method.loc10_23.1: <bound method> = bound_method %int_1.loc10_23, %impl.elem0.loc10_23 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.c1a]
// CHECK:STDOUT:   %specific_fn.loc10_23: <specific function> = specific_function %impl.elem0.loc10_23, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_23.2: <bound method> = bound_method %int_1.loc10_23, %specific_fn.loc10_23 [concrete = constants.%bound_method.e5b]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_23: init %i32 = call %bound_method.loc10_23.2(%int_1.loc10_23) [concrete = constants.%int_1.d9c]
// CHECK:STDOUT:   %.loc10_23.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_23 [concrete = constants.%int_1.d9c]
// CHECK:STDOUT:   %.loc10_23.2: %i32 = converted %int_1.loc10_23, %.loc10_23.1 [concrete = constants.%int_1.d9c]
// CHECK:STDOUT:   %F.call: init %i32 = call %F.ref(%.loc10_20.15, %.loc10_23.2)
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call constants.%Destroy.Op.bound(constants.%.2d4)
// CHECK:STDOUT:   return %F.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_20.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_20.2(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_20.3(%self.param: ref %array_type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- index_non_literal.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_3, %empty_struct_type [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %struct_type.index: type = struct_type {.index: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct: %struct_type.index = struct_value (%int_2.ecc) [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.360: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.048: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e2c: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.048 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.3d6: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.0bd, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0a4: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0a4 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.360 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.3d6) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.da5: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.474: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.da5, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.228: %i32 = int_value 2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.b42: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.048) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e2c)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.0bd = impl_witness_table (%Core.import_ref.b42), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%a.param: %array_type) -> out %return.param: %empty_struct_type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %array_type = name_ref a, %a
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc6_23.1: %struct_type.index = struct_literal (%int_2) [concrete = constants.%struct]
// CHECK:STDOUT:   %struct: %struct_type.index = struct_value (%int_2) [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc6_23.2: %struct_type.index = converted %.loc6_23.1, %struct [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc6_24.1: Core.IntLiteral = struct_access %.loc6_23.2, element0 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %impl.elem0: %.474 = impl_witness_access constants.%ImplicitAs.impl_witness.3d6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1]
// CHECK:STDOUT:   %bound_method.loc6_24.1: <bound method> = bound_method %.loc6_24.1, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_24.2: <bound method> = bound_method %.loc6_24.1, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc6_24.2(%.loc6_24.1) [concrete = constants.%int_2.228]
// CHECK:STDOUT:   %.loc6_24.2: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_2.228]
// CHECK:STDOUT:   %.loc6_24.3: %i32 = converted %.loc6_24.1, %.loc6_24.2 [concrete = constants.%int_2.228]
// CHECK:STDOUT:   %.loc6_30.1: ref %array_type = value_as_ref %a.ref
// CHECK:STDOUT:   %.loc6_30.2: ref %empty_struct_type = array_index %.loc6_30.1, %.loc6_24.3
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc6_30.3: %empty_struct_type = converted %.loc6_30.2, %empty_struct [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc6_30.4: init %empty_struct_type = struct_init () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc6_31: init %empty_struct_type = converted %.loc6_30.3, %.loc6_30.4 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   return %.loc6_31
// CHECK:STDOUT: }
// CHECK:STDOUT:

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/lookup/int_literal_convert.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/lookup/int_literal_convert.carbon

// --- generic_convert.carbon

library "[[@TEST_NAME]]";

fn ConvertZero(N:! Core.IntLiteral()) -> Core.Int(N) {
  //@dump-sem-ir-begin
  return 0;
  //@dump-sem-ir-end
}

fn ConvertOneMillion(N:! Core.IntLiteral()) -> Core.Int(N) {
  //@dump-sem-ir-begin
  return 1_000_000;
  //@dump-sem-ir-end
}

fn ValidConversions() {
  //@dump-sem-ir-begin
  var a: i8 = ConvertZero(8);
  var b: i64 = ConvertZero(64);

  var c: i32 = ConvertOneMillion(32);
  var d: i64 = ConvertOneMillion(64);
  //@dump-sem-ir-end
}

// --- fail_convert_range.carbon

library "[[@TEST_NAME]]";

fn ConvertOneMillion(N:! Core.IntLiteral()) -> Core.Int(N) {
  // CHECK:STDERR: fail_convert_range.carbon:[[@LINE+3]]:3: error: integer value 1000000 too large for type `i8` [IntTooLargeForType]
  // CHECK:STDERR:   return 1_000_000;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  return 1_000_000;
}

fn ValidConversions() {
  // CHECK:STDERR: fail_convert_range.carbon:[[@LINE+7]]:3: note: in `ConvertOneMillion(8)` used here [ResolvingSpecificHere]
  // CHECK:STDERR:   ConvertOneMillion(8);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_convert_range.carbon:[[@LINE-8]]:3: error: integer value 1000000 too large for type `i16` [IntTooLargeForType]
  // CHECK:STDERR:   return 1_000_000;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  ConvertOneMillion(8);

  // CHECK:STDERR: fail_convert_range.carbon:[[@LINE+4]]:3: note: in `ConvertOneMillion(16)` used here [ResolvingSpecificHere]
  // CHECK:STDERR:   ConvertOneMillion(16);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  ConvertOneMillion(16);
}

// --- fail_generic_convert_not_constant.carbon

library "[[@TEST_NAME]]";

fn Make() -> Core.IntLiteral();

fn ConvertRuntime(N:! Core.IntLiteral()) -> Core.Int(N) {
  // CHECK:STDERR: fail_generic_convert_not_constant.carbon:[[@LINE+7]]:3: error: non-constant call to compile-time-only function [NonConstantCallToCompTimeOnlyFunction]
  // CHECK:STDERR:   return Make();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR: min_prelude/parts/int.carbon:23:3: note: compile-time-only function declared here [CompTimeOnlyFunctionHere]
  // CHECK:STDERR:   fn Convert[self: Self]() -> Int(To) = "int.convert_checked";
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return Make();
}

// CHECK:STDOUT: --- generic_convert.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %Int.fc6021.1: type = class_type @Int, @Int(%N) [symbolic]
// CHECK:STDOUT:   %pattern_type.764eab.1: type = pattern_type %Int.fc6021.1 [symbolic]
// CHECK:STDOUT:   %ConvertZero.type: type = fn_type @ConvertZero [concrete]
// CHECK:STDOUT:   %ConvertZero: %ConvertZero.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.9019d7.1: <witness> = require_complete_type %Int.fc6021.1 [symbolic]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %Dest: type = symbolic_binding Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.ff3: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.6d3: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.Convert.50f: %ImplicitAs.Convert.type.6d3 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.a2b600.1: type = facet_type <@ImplicitAs, @ImplicitAs(%Int.fc6021.1)> [symbolic]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.45c09f.1: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%Int.fc6021.1) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.d9e959.1: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%Int.fc6021.1) [symbolic]
// CHECK:STDOUT:   %assoc0.ec6: %ImplicitAs.assoc_type.d9e959.1 = assoc_entity element0, imports.%Core.import_ref.218 [symbolic]
// CHECK:STDOUT:   %require_complete.a2ba56.1: <witness> = require_complete_type %ImplicitAs.type.a2b600.1 [symbolic]
// CHECK:STDOUT:   %assoc0.0bc: %ImplicitAs.assoc_type.ff3 = assoc_entity element0, imports.%Core.import_ref.fa2 [symbolic]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.41218a.1: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.740682.1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.41218a.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.23aa8c.2: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.8c3, @Core.IntLiteral.as.ImplicitAs.impl(%N) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.41218a.2: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%N) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.740682.2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.41218a.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %.0d3: require_specific_def_type = require_specific_def @Core.IntLiteral.as.ImplicitAs.impl(%N) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.facet.1ac: %ImplicitAs.type.a2b600.1 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.23aa8c.2) [symbolic]
// CHECK:STDOUT:   %.cef: type = fn_type_with_self_type %ImplicitAs.Convert.type.45c09f.1, %ImplicitAs.facet.1ac [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.285: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.740682.2 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.cd7: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.740682.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%N) [symbolic]
// CHECK:STDOUT:   %bound_method.6aa: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.cd7 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.0ef: init %Int.fc6021.1 = call %bound_method.6aa(%int_0.5c6) [symbolic]
// CHECK:STDOUT:   %ConvertOneMillion.type: type = fn_type @ConvertOneMillion [concrete]
// CHECK:STDOUT:   %ConvertOneMillion: %ConvertOneMillion.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_1000000.01f: Core.IntLiteral = int_value 1000000 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.835: <bound method> = bound_method %int_1000000.01f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.740682.2 [symbolic]
// CHECK:STDOUT:   %bound_method.b3e: <bound method> = bound_method %int_1000000.01f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.cd7 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.489: init %Int.fc6021.1 = call %bound_method.b3e(%int_1000000.01f) [symbolic]
// CHECK:STDOUT:   %int_8: Core.IntLiteral = int_value 8 [concrete]
// CHECK:STDOUT:   %i8: type = class_type @Int, @Int(%int_8) [concrete]
// CHECK:STDOUT:   %i8.builtin: type = int_type signed, %int_8 [concrete]
// CHECK:STDOUT:   %complete_type.58c: <witness> = complete_type_witness %i8.builtin [concrete]
// CHECK:STDOUT:   %pattern_type.e3f: type = pattern_type %i8 [concrete]
// CHECK:STDOUT:   %ConvertZero.specific_fn.d56: <specific function> = specific_function %ConvertZero, @ConvertZero(%int_8) [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
// CHECK:STDOUT:   %i64.builtin: type = int_type signed, %int_64 [concrete]
// CHECK:STDOUT:   %complete_type.4a1: <witness> = complete_type_witness %i64.builtin [concrete]
// CHECK:STDOUT:   %pattern_type.95b: type = pattern_type %i64 [concrete]
// CHECK:STDOUT:   %ConvertZero.specific_fn.021: <specific function> = specific_function %ConvertZero, @ConvertZero(%int_64) [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %ConvertOneMillion.specific_fn.29f: <specific function> = specific_function %ConvertOneMillion, @ConvertOneMillion(%int_32) [concrete]
// CHECK:STDOUT:   %ConvertOneMillion.specific_fn.fe7: <specific function> = specific_function %ConvertOneMillion, @ConvertOneMillion(%int_64) [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value.f6a: %type_where = facet_value %i64, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.545: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.f6a) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.b3c: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.545 = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.d23: %type_where = facet_value %i32, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.9df: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.d23) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.ae2: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.9df = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.7c8: %type_where = facet_value %i8, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.1fc: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.7c8) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.644: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.1fc = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.3ce: type = facet_type <@ImplicitAs, @ImplicitAs(%i8)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.e79: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i8) [concrete]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.1b0: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%i8) [concrete]
// CHECK:STDOUT:   %assoc0.4d1: %ImplicitAs.assoc_type.1b0 = assoc_entity element0, imports.%Core.import_ref.218 [concrete]
// CHECK:STDOUT:   %complete_type.69d: <witness> = complete_type_witness %ImplicitAs.type.3ce [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.64d: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.8c3, @Core.IntLiteral.as.ImplicitAs.impl(%int_8) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.1b1: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_8) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.1dc: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.1b1 = struct_value () [concrete]
// CHECK:STDOUT:   %.151: require_specific_def_type = require_specific_def @Core.IntLiteral.as.ImplicitAs.impl(%int_8) [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.67f: %ImplicitAs.type.3ce = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.64d) [concrete]
// CHECK:STDOUT:   %.246: type = fn_type_with_self_type %ImplicitAs.Convert.type.e79, %ImplicitAs.facet.67f [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.55e: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.1dc [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.d87: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.1dc, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_8) [concrete]
// CHECK:STDOUT:   %bound_method.3e6: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.d87 [concrete]
// CHECK:STDOUT:   %int_0.f20: %i8 = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e50: type = facet_type <@ImplicitAs, @ImplicitAs(%i64)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.94e: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i64) [concrete]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.849: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%i64) [concrete]
// CHECK:STDOUT:   %assoc0.bd1: %ImplicitAs.assoc_type.849 = assoc_entity element0, imports.%Core.import_ref.218 [concrete]
// CHECK:STDOUT:   %complete_type.ce8: <witness> = complete_type_witness %ImplicitAs.type.e50 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.47f: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.8c3, @Core.IntLiteral.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.c10: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f8e: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.c10 = struct_value () [concrete]
// CHECK:STDOUT:   %.013: require_specific_def_type = require_specific_def @Core.IntLiteral.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.673: %ImplicitAs.type.e50 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.47f) [concrete]
// CHECK:STDOUT:   %.23a: type = fn_type_with_self_type %ImplicitAs.Convert.type.94e, %ImplicitAs.facet.673 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.ece: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f8e [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.d59cc: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.f8e, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.c1b: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.d59cc [concrete]
// CHECK:STDOUT:   %int_0.6ab: %i64 = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.d14: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.3d9: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %assoc0.a71: %ImplicitAs.assoc_type.3d9 = assoc_entity element0, imports.%Core.import_ref.218 [concrete]
// CHECK:STDOUT:   %complete_type.8b8: <witness> = complete_type_witness %ImplicitAs.type.d14 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.57f: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.8c3, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.640: %ImplicitAs.type.d14 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.57f) [concrete]
// CHECK:STDOUT:   %.dbf: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet.640 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.84b: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f0: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.84b = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5e3: <bound method> = bound_method %int_1000000.01f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f0 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.d59c4: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.b76: <bound method> = bound_method %int_1000000.01f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.d59c4 [concrete]
// CHECK:STDOUT:   %int_1000000.272: %i32 = int_value 1000000 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.545: <bound method> = bound_method %int_1000000.01f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f8e [concrete]
// CHECK:STDOUT:   %bound_method.bf1: <bound method> = bound_method %int_1000000.01f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.d59cc [concrete]
// CHECK:STDOUT:   %int_1000000.7e7: %i64 = int_value 1000000 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.a80: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.ff3) = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.%assoc0 (constants.%assoc0.0bc)]
// CHECK:STDOUT:   %Core.import_ref.218: @ImplicitAs.%ImplicitAs.Convert.type (%ImplicitAs.Convert.type.6d3) = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.%ImplicitAs.Convert (constants.%ImplicitAs.Convert.50f)]
// CHECK:STDOUT:   %Core.import_ref.fa2 = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.1b5: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.41218a.1) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.740682.1)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.8c3 = impl_witness_table (%Core.import_ref.1b5), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ConvertZero(%N.loc4_16.2: Core.IntLiteral) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %ImplicitAs.type.loc6_11.2: type = facet_type <@ImplicitAs, @ImplicitAs(%Int.loc4_52.1)> [symbolic = %ImplicitAs.type.loc6_11.2 (constants.%ImplicitAs.type.a2b600.1)]
// CHECK:STDOUT:   %require_complete.loc6: <witness> = require_complete_type %ImplicitAs.type.loc6_11.2 [symbolic = %require_complete.loc6 (constants.%require_complete.a2ba56.1)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%Int.loc4_52.1) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type.d9e959.1)]
// CHECK:STDOUT:   %assoc0: @ConvertZero.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.d9e959.1) = assoc_entity element0, imports.%Core.import_ref.218 [symbolic = %assoc0 (constants.%assoc0.ec6)]
// CHECK:STDOUT:   %.loc6_11.3: require_specific_def_type = require_specific_def @Core.IntLiteral.as.ImplicitAs.impl(%N.loc4_16.1) [symbolic = %.loc6_11.3 (constants.%.0d3)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.8c3, @Core.IntLiteral.as.ImplicitAs.impl(%N.loc4_16.1) [symbolic = %ImplicitAs.impl_witness (constants.%ImplicitAs.impl_witness.23aa8c.2)]
// CHECK:STDOUT:   %ImplicitAs.Convert.type: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%Int.loc4_52.1) [symbolic = %ImplicitAs.Convert.type (constants.%ImplicitAs.Convert.type.45c09f.1)]
// CHECK:STDOUT:   %ImplicitAs.facet: @ConvertZero.%ImplicitAs.type.loc6_11.2 (%ImplicitAs.type.a2b600.1) = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness) [symbolic = %ImplicitAs.facet (constants.%ImplicitAs.facet.1ac)]
// CHECK:STDOUT:   %.loc6_11.4: type = fn_type_with_self_type %ImplicitAs.Convert.type, %ImplicitAs.facet [symbolic = %.loc6_11.4 (constants.%.cef)]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%N.loc4_16.1) [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.type (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.41218a.2)]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: @ConvertZero.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.41218a.2) = struct_value () [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.740682.2)]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method constants.%int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.285)]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%N.loc4_16.1) [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.cd7)]
// CHECK:STDOUT:   %bound_method.loc6_11.3: <bound method> = bound_method constants.%int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [symbolic = %bound_method.loc6_11.3 (constants.%bound_method.6aa)]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_11.2: init @ConvertZero.%Int.loc4_52.1 (%Int.fc6021.1) = call %bound_method.loc6_11.3(constants.%int_0.5c6) [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_11.2 (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.call.0ef)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> @ConvertZero.%Int.loc4_52.1 (%Int.fc6021.1) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:     %ImplicitAs.type.loc6_11.1: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%Int.fc6021.1)> [symbolic = %ImplicitAs.type.loc6_11.2 (constants.%ImplicitAs.type.a2b600.1)]
// CHECK:STDOUT:     %.loc6_11.1: @ConvertZero.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.d9e959.1) = specific_constant imports.%Core.import_ref.a80, @ImplicitAs(constants.%Int.fc6021.1) [symbolic = %assoc0 (constants.%assoc0.ec6)]
// CHECK:STDOUT:     %Convert.ref: @ConvertZero.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.d9e959.1) = name_ref Convert, %.loc6_11.1 [symbolic = %assoc0 (constants.%assoc0.ec6)]
// CHECK:STDOUT:     %impl.elem0: @ConvertZero.%.loc6_11.4 (%.cef) = impl_witness_access constants.%ImplicitAs.impl_witness.23aa8c.2, element0 [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.740682.2)]
// CHECK:STDOUT:     %bound_method.loc6_11.1: <bound method> = bound_method %int_0, %impl.elem0 [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.285)]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%N) [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.cd7)]
// CHECK:STDOUT:     %bound_method.loc6_11.2: <bound method> = bound_method %int_0, %specific_fn [symbolic = %bound_method.loc6_11.3 (constants.%bound_method.6aa)]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_11.1: init @ConvertZero.%Int.loc4_52.1 (%Int.fc6021.1) = call %bound_method.loc6_11.2(%int_0) [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_11.2 (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.call.0ef)]
// CHECK:STDOUT:     %.loc6_11.2: init @ConvertZero.%Int.loc4_52.1 (%Int.fc6021.1) = converted %int_0, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_11.1 [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_11.2 (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.call.0ef)]
// CHECK:STDOUT:     return %.loc6_11.2 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ConvertOneMillion(%N.loc10_22.2: Core.IntLiteral) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc12_19.1: <witness> = require_complete_type %Int.loc10_58.1 [symbolic = %require_complete.loc12_19.1 (constants.%require_complete.9019d7.1)]
// CHECK:STDOUT:   %ImplicitAs.type.loc12_19.2: type = facet_type <@ImplicitAs, @ImplicitAs(%Int.loc10_58.1)> [symbolic = %ImplicitAs.type.loc12_19.2 (constants.%ImplicitAs.type.a2b600.1)]
// CHECK:STDOUT:   %require_complete.loc12_19.2: <witness> = require_complete_type %ImplicitAs.type.loc12_19.2 [symbolic = %require_complete.loc12_19.2 (constants.%require_complete.a2ba56.1)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%Int.loc10_58.1) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type.d9e959.1)]
// CHECK:STDOUT:   %assoc0: @ConvertOneMillion.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.d9e959.1) = assoc_entity element0, imports.%Core.import_ref.218 [symbolic = %assoc0 (constants.%assoc0.ec6)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.8c3, @Core.IntLiteral.as.ImplicitAs.impl(%N.loc10_22.1) [symbolic = %ImplicitAs.impl_witness (constants.%ImplicitAs.impl_witness.23aa8c.2)]
// CHECK:STDOUT:   %ImplicitAs.Convert.type: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%Int.loc10_58.1) [symbolic = %ImplicitAs.Convert.type (constants.%ImplicitAs.Convert.type.45c09f.1)]
// CHECK:STDOUT:   %ImplicitAs.facet: @ConvertOneMillion.%ImplicitAs.type.loc12_19.2 (%ImplicitAs.type.a2b600.1) = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness) [symbolic = %ImplicitAs.facet (constants.%ImplicitAs.facet.1ac)]
// CHECK:STDOUT:   %.loc12_19.3: type = fn_type_with_self_type %ImplicitAs.Convert.type, %ImplicitAs.facet [symbolic = %.loc12_19.3 (constants.%.cef)]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%N.loc10_22.1) [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.type (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.41218a.2)]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: @ConvertOneMillion.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.41218a.2) = struct_value () [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.740682.2)]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method constants.%int_1000000.01f, %Core.IntLiteral.as.ImplicitAs.impl.Convert [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.835)]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%N.loc10_22.1) [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.cd7)]
// CHECK:STDOUT:   %bound_method.loc12_19.3: <bound method> = bound_method constants.%int_1000000.01f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [symbolic = %bound_method.loc12_19.3 (constants.%bound_method.b3e)]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12_19.2: init @ConvertOneMillion.%Int.loc10_58.1 (%Int.fc6021.1) = call %bound_method.loc12_19.3(constants.%int_1000000.01f) [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12_19.2 (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.call.489)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> @ConvertOneMillion.%Int.loc10_58.1 (%Int.fc6021.1) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %int_1000000: Core.IntLiteral = int_value 1000000 [concrete = constants.%int_1000000.01f]
// CHECK:STDOUT:     %ImplicitAs.type.loc12_19.1: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%Int.fc6021.1)> [symbolic = %ImplicitAs.type.loc12_19.2 (constants.%ImplicitAs.type.a2b600.1)]
// CHECK:STDOUT:     %.loc12_19.1: @ConvertOneMillion.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.d9e959.1) = specific_constant imports.%Core.import_ref.a80, @ImplicitAs(constants.%Int.fc6021.1) [symbolic = %assoc0 (constants.%assoc0.ec6)]
// CHECK:STDOUT:     %Convert.ref: @ConvertOneMillion.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.d9e959.1) = name_ref Convert, %.loc12_19.1 [symbolic = %assoc0 (constants.%assoc0.ec6)]
// CHECK:STDOUT:     %impl.elem0: @ConvertOneMillion.%.loc12_19.3 (%.cef) = impl_witness_access constants.%ImplicitAs.impl_witness.23aa8c.2, element0 [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.740682.2)]
// CHECK:STDOUT:     %bound_method.loc12_19.1: <bound method> = bound_method %int_1000000, %impl.elem0 [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.835)]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%N) [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.cd7)]
// CHECK:STDOUT:     %bound_method.loc12_19.2: <bound method> = bound_method %int_1000000, %specific_fn [symbolic = %bound_method.loc12_19.3 (constants.%bound_method.b3e)]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12_19.1: init @ConvertOneMillion.%Int.loc10_58.1 (%Int.fc6021.1) = call %bound_method.loc12_19.2(%int_1000000) [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12_19.2 (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.call.489)]
// CHECK:STDOUT:     %.loc12_19.2: init @ConvertOneMillion.%Int.loc10_58.1 (%Int.fc6021.1) = converted %int_1000000, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12_19.1 [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12_19.2 (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.call.489)]
// CHECK:STDOUT:     return %.loc12_19.2 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ValidConversions() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.e3f = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.e3f = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %i8 = var %a.var_patt
// CHECK:STDOUT:   %ConvertZero.ref.loc18: %ConvertZero.type = name_ref ConvertZero, file.%ConvertZero.decl [concrete = constants.%ConvertZero]
// CHECK:STDOUT:   %int_8.loc18_27: Core.IntLiteral = int_value 8 [concrete = constants.%int_8]
// CHECK:STDOUT:   %ConvertZero.specific_fn.loc18: <specific function> = specific_function %ConvertZero.ref.loc18, @ConvertZero(constants.%int_8) [concrete = constants.%ConvertZero.specific_fn.d56]
// CHECK:STDOUT:   %ConvertZero.call.loc18: init %i8 = call %ConvertZero.specific_fn.loc18()
// CHECK:STDOUT:   assign %a.var, %ConvertZero.call.loc18
// CHECK:STDOUT:   %.loc18: type = splice_block %i8 [concrete = constants.%i8] {
// CHECK:STDOUT:     %int_8.loc18_10: Core.IntLiteral = int_value 8 [concrete = constants.%int_8]
// CHECK:STDOUT:     %i8: type = class_type @Int, @Int(constants.%int_8) [concrete = constants.%i8]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %i8 = ref_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.95b = ref_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.95b = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %i64 = var %b.var_patt
// CHECK:STDOUT:   %ConvertZero.ref.loc19: %ConvertZero.type = name_ref ConvertZero, file.%ConvertZero.decl [concrete = constants.%ConvertZero]
// CHECK:STDOUT:   %int_64.loc19_28: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:   %ConvertZero.specific_fn.loc19: <specific function> = specific_function %ConvertZero.ref.loc19, @ConvertZero(constants.%int_64) [concrete = constants.%ConvertZero.specific_fn.021]
// CHECK:STDOUT:   %ConvertZero.call.loc19: init %i64 = call %ConvertZero.specific_fn.loc19()
// CHECK:STDOUT:   assign %b.var, %ConvertZero.call.loc19
// CHECK:STDOUT:   %.loc19: type = splice_block %i64.loc19 [concrete = constants.%i64] {
// CHECK:STDOUT:     %int_64.loc19_10: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %i64.loc19: type = class_type @Int, @Int(constants.%int_64) [concrete = constants.%i64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %i64 = ref_binding b, %b.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.7ce = ref_binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.7ce = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %i32 = var %c.var_patt
// CHECK:STDOUT:   %ConvertOneMillion.ref.loc21: %ConvertOneMillion.type = name_ref ConvertOneMillion, file.%ConvertOneMillion.decl [concrete = constants.%ConvertOneMillion]
// CHECK:STDOUT:   %int_32.loc21_34: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %ConvertOneMillion.specific_fn.loc21: <specific function> = specific_function %ConvertOneMillion.ref.loc21, @ConvertOneMillion(constants.%int_32) [concrete = constants.%ConvertOneMillion.specific_fn.29f]
// CHECK:STDOUT:   %ConvertOneMillion.call.loc21: init %i32 = call %ConvertOneMillion.specific_fn.loc21()
// CHECK:STDOUT:   assign %c.var, %ConvertOneMillion.call.loc21
// CHECK:STDOUT:   %.loc21: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc21_10: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %i32 = ref_binding c, %c.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.95b = ref_binding_pattern d [concrete]
// CHECK:STDOUT:     %d.var_patt: %pattern_type.95b = var_pattern %d.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.var: ref %i64 = var %d.var_patt
// CHECK:STDOUT:   %ConvertOneMillion.ref.loc22: %ConvertOneMillion.type = name_ref ConvertOneMillion, file.%ConvertOneMillion.decl [concrete = constants.%ConvertOneMillion]
// CHECK:STDOUT:   %int_64.loc22_34: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:   %ConvertOneMillion.specific_fn.loc22: <specific function> = specific_function %ConvertOneMillion.ref.loc22, @ConvertOneMillion(constants.%int_64) [concrete = constants.%ConvertOneMillion.specific_fn.fe7]
// CHECK:STDOUT:   %ConvertOneMillion.call.loc22: init %i64 = call %ConvertOneMillion.specific_fn.loc22()
// CHECK:STDOUT:   assign %d.var, %ConvertOneMillion.call.loc22
// CHECK:STDOUT:   %.loc22: type = splice_block %i64.loc22 [concrete = constants.%i64] {
// CHECK:STDOUT:     %int_64.loc22_10: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %i64.loc22: type = class_type @Int, @Int(constants.%int_64) [concrete = constants.%i64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d: ref %i64 = ref_binding d, %d.var
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc22: <bound method> = bound_method %d.var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.b3c
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc22: <bound method> = bound_method %d.var, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc22: init %empty_tuple.type = call %bound_method.loc22(%d.var)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc21: <bound method> = bound_method %c.var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.ae2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc21: <bound method> = bound_method %c.var, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc21: init %empty_tuple.type = call %bound_method.loc21(%c.var)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc19: <bound method> = bound_method %b.var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.b3c
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc19: <bound method> = bound_method %b.var, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.3
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc19: init %empty_tuple.type = call %bound_method.loc19(%b.var)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc18: <bound method> = bound_method %a.var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.644
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc18: <bound method> = bound_method %a.var, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc18: init %empty_tuple.type = call %bound_method.loc18(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ConvertZero(constants.%N) {
// CHECK:STDOUT:   %N.loc4_16.1 => constants.%N
// CHECK:STDOUT:   %Int.loc4_52.1 => constants.%Int.fc6021.1
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.764eab.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ConvertOneMillion(constants.%N) {
// CHECK:STDOUT:   %N.loc10_22.1 => constants.%N
// CHECK:STDOUT:   %Int.loc10_58.1 => constants.%Int.fc6021.1
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.764eab.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ConvertZero(constants.%int_8) {
// CHECK:STDOUT:   %N.loc4_16.1 => constants.%int_8
// CHECK:STDOUT:   %Int.loc4_52.1 => constants.%i8
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.e3f
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc4 => constants.%complete_type.58c
// CHECK:STDOUT:   %ImplicitAs.type.loc6_11.2 => constants.%ImplicitAs.type.3ce
// CHECK:STDOUT:   %require_complete.loc6 => constants.%complete_type.69d
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.1b0
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.4d1
// CHECK:STDOUT:   %.loc6_11.3 => constants.%.151
// CHECK:STDOUT:   %ImplicitAs.impl_witness => constants.%ImplicitAs.impl_witness.64d
// CHECK:STDOUT:   %ImplicitAs.Convert.type => constants.%ImplicitAs.Convert.type.e79
// CHECK:STDOUT:   %ImplicitAs.facet => constants.%ImplicitAs.facet.67f
// CHECK:STDOUT:   %.loc6_11.4 => constants.%.246
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type => constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.1b1
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert => constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.1dc
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound => constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.55e
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn => constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.d87
// CHECK:STDOUT:   %bound_method.loc6_11.3 => constants.%bound_method.3e6
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_11.2 => constants.%int_0.f20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ConvertZero(constants.%int_64) {
// CHECK:STDOUT:   %N.loc4_16.1 => constants.%int_64
// CHECK:STDOUT:   %Int.loc4_52.1 => constants.%i64
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.95b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc4 => constants.%complete_type.4a1
// CHECK:STDOUT:   %ImplicitAs.type.loc6_11.2 => constants.%ImplicitAs.type.e50
// CHECK:STDOUT:   %require_complete.loc6 => constants.%complete_type.ce8
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.849
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.bd1
// CHECK:STDOUT:   %.loc6_11.3 => constants.%.013
// CHECK:STDOUT:   %ImplicitAs.impl_witness => constants.%ImplicitAs.impl_witness.47f
// CHECK:STDOUT:   %ImplicitAs.Convert.type => constants.%ImplicitAs.Convert.type.94e
// CHECK:STDOUT:   %ImplicitAs.facet => constants.%ImplicitAs.facet.673
// CHECK:STDOUT:   %.loc6_11.4 => constants.%.23a
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type => constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.c10
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert => constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f8e
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound => constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.ece
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn => constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.d59cc
// CHECK:STDOUT:   %bound_method.loc6_11.3 => constants.%bound_method.c1b
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_11.2 => constants.%int_0.6ab
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ConvertOneMillion(constants.%int_32) {
// CHECK:STDOUT:   %N.loc10_22.1 => constants.%int_32
// CHECK:STDOUT:   %Int.loc10_58.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7ce
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc12_19.1 => constants.%complete_type.f8a
// CHECK:STDOUT:   %ImplicitAs.type.loc12_19.2 => constants.%ImplicitAs.type.d14
// CHECK:STDOUT:   %require_complete.loc12_19.2 => constants.%complete_type.8b8
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.3d9
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.a71
// CHECK:STDOUT:   %ImplicitAs.impl_witness => constants.%ImplicitAs.impl_witness.57f
// CHECK:STDOUT:   %ImplicitAs.Convert.type => constants.%ImplicitAs.Convert.type.1b6
// CHECK:STDOUT:   %ImplicitAs.facet => constants.%ImplicitAs.facet.640
// CHECK:STDOUT:   %.loc12_19.3 => constants.%.dbf
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type => constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.84b
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert => constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6f0
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound => constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5e3
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn => constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.d59c4
// CHECK:STDOUT:   %bound_method.loc12_19.3 => constants.%bound_method.b76
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12_19.2 => constants.%int_1000000.272
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ConvertOneMillion(constants.%int_64) {
// CHECK:STDOUT:   %N.loc10_22.1 => constants.%int_64
// CHECK:STDOUT:   %Int.loc10_58.1 => constants.%i64
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.95b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc12_19.1 => constants.%complete_type.4a1
// CHECK:STDOUT:   %ImplicitAs.type.loc12_19.2 => constants.%ImplicitAs.type.e50
// CHECK:STDOUT:   %require_complete.loc12_19.2 => constants.%complete_type.ce8
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.849
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.bd1
// CHECK:STDOUT:   %ImplicitAs.impl_witness => constants.%ImplicitAs.impl_witness.47f
// CHECK:STDOUT:   %ImplicitAs.Convert.type => constants.%ImplicitAs.Convert.type.94e
// CHECK:STDOUT:   %ImplicitAs.facet => constants.%ImplicitAs.facet.673
// CHECK:STDOUT:   %.loc12_19.3 => constants.%.23a
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type => constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.c10
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert => constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f8e
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound => constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.545
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn => constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.d59cc
// CHECK:STDOUT:   %bound_method.loc12_19.3 => constants.%bound_method.bf1
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12_19.2 => constants.%int_1000000.7e7
// CHECK:STDOUT: }
// CHECK:STDOUT:

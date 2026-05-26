// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/return/tuple.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/return/tuple.carbon


fn Main() -> (i32, i32) {
  return (15, 35);
}

// CHECK:STDOUT: --- tuple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.8e3: %tuple.type.24b = tuple_value (%i32, %i32) [concrete]
// CHECK:STDOUT:   %tuple.type.87c: type = tuple_type (%i32, %i32) [concrete]
// CHECK:STDOUT:   %.88c: Core.Form = init_form %tuple.type.87c [concrete]
// CHECK:STDOUT:   %pattern_type.624: type = pattern_type %tuple.type.87c [concrete]
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [concrete]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_15.447: Core.IntLiteral = int_value 15 [concrete]
// CHECK:STDOUT:   %int_35.c79: Core.IntLiteral = int_value 35 [concrete]
// CHECK:STDOUT:   %tuple.type.f94: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.1b5: %tuple.type.f94 = tuple_value (%int_15.447, %int_35.c79) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.649: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.649 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.544: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.ec3: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.fb6: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1a5, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ef7: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ef7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.544 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.fb6) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.367: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.205: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.367, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.90d: <bound method> = bound_method %int_15.447, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.896: <bound method> = bound_method %int_15.447, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_15.e49: %i32 = int_value 15 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.ab0: <bound method> = bound_method %int_35.c79, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b [concrete]
// CHECK:STDOUT:   %bound_method.6eb: <bound method> = bound_method %int_35.c79, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_35.334: %i32 = int_value 35 [concrete]
// CHECK:STDOUT:   %tuple.fc2: %tuple.type.87c = tuple_value (%int_15.e49, %int_35.334) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.649 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.dd3: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ec3)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1a5 = impl_witness_table (%Core.import_ref.dd3), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Main.decl: %Main.type = fn_decl @Main [concrete = constants.%Main] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.624 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.624 = return_slot_pattern %return.param_patt, %.loc16_23.2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc16_15: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc16_20: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc16_23.1: %tuple.type.24b = tuple_literal (%i32.loc16_15, %i32.loc16_20) [concrete = constants.%tuple.8e3]
// CHECK:STDOUT:     %.loc16_23.2: type = converted %.loc16_23.1, constants.%tuple.type.87c [concrete = constants.%tuple.type.87c]
// CHECK:STDOUT:     %.loc16_23.3: Core.Form = init_form %.loc16_23.2 [concrete = constants.%.88c]
// CHECK:STDOUT:     %return.param: ref %tuple.type.87c = out_param call_param0
// CHECK:STDOUT:     %return: ref %tuple.type.87c = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() -> out %return.param: %tuple.type.87c {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_15: Core.IntLiteral = int_value 15 [concrete = constants.%int_15.447]
// CHECK:STDOUT:   %int_35: Core.IntLiteral = int_value 35 [concrete = constants.%int_35.c79]
// CHECK:STDOUT:   %.loc17_17.1: %tuple.type.f94 = tuple_literal (%int_15, %int_35) [concrete = constants.%tuple.1b5]
// CHECK:STDOUT:   %impl.elem0.loc17_17.1: %.205 = impl_witness_access constants.%ImplicitAs.impl_witness.fb6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b]
// CHECK:STDOUT:   %bound_method.loc17_17.1: <bound method> = bound_method %int_15, %impl.elem0.loc17_17.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.90d]
// CHECK:STDOUT:   %specific_fn.loc17_17.1: <specific function> = specific_function %impl.elem0.loc17_17.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc17_17.2: <bound method> = bound_method %int_15, %specific_fn.loc17_17.1 [concrete = constants.%bound_method.896]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17_17.1: init %i32 = call %bound_method.loc17_17.2(%int_15) [concrete = constants.%int_15.e49]
// CHECK:STDOUT:   %.loc17_17.2: init %i32 = converted %int_15, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17_17.1 [concrete = constants.%int_15.e49]
// CHECK:STDOUT:   %tuple.elem0: ref %i32 = tuple_access %return.param, element0
// CHECK:STDOUT:   %.loc17_17.3: init %i32 to %tuple.elem0 = in_place_init %.loc17_17.2 [concrete = constants.%int_15.e49]
// CHECK:STDOUT:   %impl.elem0.loc17_17.2: %.205 = impl_witness_access constants.%ImplicitAs.impl_witness.fb6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b]
// CHECK:STDOUT:   %bound_method.loc17_17.3: <bound method> = bound_method %int_35, %impl.elem0.loc17_17.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.ab0]
// CHECK:STDOUT:   %specific_fn.loc17_17.2: <specific function> = specific_function %impl.elem0.loc17_17.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc17_17.4: <bound method> = bound_method %int_35, %specific_fn.loc17_17.2 [concrete = constants.%bound_method.6eb]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17_17.2: init %i32 = call %bound_method.loc17_17.4(%int_35) [concrete = constants.%int_35.334]
// CHECK:STDOUT:   %.loc17_17.4: init %i32 = converted %int_35, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17_17.2 [concrete = constants.%int_35.334]
// CHECK:STDOUT:   %tuple.elem1: ref %i32 = tuple_access %return.param, element1
// CHECK:STDOUT:   %.loc17_17.5: init %i32 to %tuple.elem1 = in_place_init %.loc17_17.4 [concrete = constants.%int_35.334]
// CHECK:STDOUT:   %.loc17_17.6: init %tuple.type.87c to %return.param = tuple_init (%.loc17_17.3, %.loc17_17.5) [concrete = constants.%tuple.fc2]
// CHECK:STDOUT:   %.loc17_18: init %tuple.type.87c = converted %.loc17_17.1, %.loc17_17.6 [concrete = constants.%tuple.fc2]
// CHECK:STDOUT:   return %.loc17_18 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:

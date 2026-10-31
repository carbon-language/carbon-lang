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
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = parameterized_class_name_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.693: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.be9: %tuple.type.693 = tuple_value (%i32, %i32) [concrete]
// CHECK:STDOUT:   %tuple.type.35f: type = tuple_type (%i32, %i32) [concrete]
// CHECK:STDOUT:   %.30d: Core.Form = init_form %tuple.type.35f [concrete]
// CHECK:STDOUT:   %pattern_type.354: type = pattern_type %tuple.type.35f [concrete]
// CHECK:STDOUT:   %return.param_patt.46d: %pattern_type.354 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.d16: %pattern_type.354 = return_slot_pattern %return.param_patt.46d, %tuple.type.35f [concrete]
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [concrete]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_15.447: Core.IntLiteral = int_value 15 [concrete]
// CHECK:STDOUT:   %int_35.c79: Core.IntLiteral = int_value 35 [concrete]
// CHECK:STDOUT:   %tuple.type.f94: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.1b5: %tuple.type.f94 = tuple_value (%int_15.447, %int_35.c79) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6b0: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.d73: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f53, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.6b0 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.d73) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.58f: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.33a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.58f, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.af2: <bound method> = bound_method %int_15.447, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.e89: <bound method> = bound_method %int_15.447, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_15.b25: %i32 = int_value 15 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.04b: <bound method> = bound_method %int_35.c79, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %bound_method.117: <bound method> = bound_method %int_35.c79, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_35.a20: %i32 = int_value 35 [concrete]
// CHECK:STDOUT:   %tuple.c67: %tuple.type.35f = tuple_value (%int_15.b25, %int_35.a20) [concrete]
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
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.e47: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f53 = impl_witness_table (%Core.import_ref.e47), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Main.decl: %Main.type = fn_decl @Main [concrete = constants.%Main] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.354 = out_param_pattern [concrete = constants.%return.param_patt.46d]
// CHECK:STDOUT:     %return.patt: %pattern_type.354 = return_slot_pattern %return.param_patt, %.loc16_23.2 [concrete = constants.%return.patt.d16]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc16_15: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc16_20: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc16_23.1: %tuple.type.693 = tuple_literal (%i32.loc16_15, %i32.loc16_20) [concrete = constants.%tuple.be9]
// CHECK:STDOUT:     %.loc16_23.2: type = converted %.loc16_23.1, constants.%tuple.type.35f [concrete = constants.%tuple.type.35f]
// CHECK:STDOUT:     %.loc16_23.3: Core.Form = init_form %.loc16_23.2 [concrete = constants.%.30d]
// CHECK:STDOUT:     %return.param: ref %tuple.type.35f = out_param call_param0
// CHECK:STDOUT:     %return: ref %tuple.type.35f = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() -> out %return.param: %tuple.type.35f {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_15: Core.IntLiteral = int_value 15 [concrete = constants.%int_15.447]
// CHECK:STDOUT:   %int_35: Core.IntLiteral = int_value 35 [concrete = constants.%int_35.c79]
// CHECK:STDOUT:   %.loc17_17.1: %tuple.type.f94 = tuple_literal (%int_15, %int_35) [concrete = constants.%tuple.1b5]
// CHECK:STDOUT:   %impl.elem0.loc17_17.1: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc17_17.1: <bound method> = bound_method %int_15, %impl.elem0.loc17_17.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.af2]
// CHECK:STDOUT:   %specific_fn.loc17_17.1: <specific function> = specific_function %impl.elem0.loc17_17.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc17_17.2: <bound method> = bound_method %int_15, %specific_fn.loc17_17.1 [concrete = constants.%bound_method.e89]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17_17.1: init %i32 = call %bound_method.loc17_17.2(%int_15) [concrete = constants.%int_15.b25]
// CHECK:STDOUT:   %.loc17_17.2: init %i32 = converted %int_15, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17_17.1 [concrete = constants.%int_15.b25]
// CHECK:STDOUT:   %tuple.elem0: ref %i32 = tuple_access %return.param, element0
// CHECK:STDOUT:   %.loc17_17.3: init %i32 to %tuple.elem0 = in_place_init %.loc17_17.2 [concrete = constants.%int_15.b25]
// CHECK:STDOUT:   %impl.elem0.loc17_17.2: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc17_17.3: <bound method> = bound_method %int_35, %impl.elem0.loc17_17.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.04b]
// CHECK:STDOUT:   %specific_fn.loc17_17.2: <specific function> = specific_function %impl.elem0.loc17_17.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc17_17.4: <bound method> = bound_method %int_35, %specific_fn.loc17_17.2 [concrete = constants.%bound_method.117]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17_17.2: init %i32 = call %bound_method.loc17_17.4(%int_35) [concrete = constants.%int_35.a20]
// CHECK:STDOUT:   %.loc17_17.4: init %i32 = converted %int_35, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17_17.2 [concrete = constants.%int_35.a20]
// CHECK:STDOUT:   %tuple.elem1: ref %i32 = tuple_access %return.param, element1
// CHECK:STDOUT:   %.loc17_17.5: init %i32 to %tuple.elem1 = in_place_init %.loc17_17.4 [concrete = constants.%int_35.a20]
// CHECK:STDOUT:   %.loc17_17.6: init %tuple.type.35f to %return.param = tuple_init (%.loc17_17.3, %.loc17_17.5) [concrete = constants.%tuple.c67]
// CHECK:STDOUT:   %.loc17_18: init %tuple.type.35f = converted %.loc17_17.1, %.loc17_17.6 [concrete = constants.%tuple.c67]
// CHECK:STDOUT:   return %.loc17_18 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:

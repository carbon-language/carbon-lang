// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.d07: type = tuple_type (%i32, %i32) [concrete]
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [concrete]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_15.447: Core.IntLiteral = int_value 15 [concrete]
// CHECK:STDOUT:   %int_35.c79: Core.IntLiteral = int_value 35 [concrete]
// CHECK:STDOUT:   %tuple.type.f94: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %impl_witness.d39: <witness> = impl_witness (imports.%Core.import_ref.a5b), @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, %impl_witness.d39 [concrete]
// CHECK:STDOUT:   %.a0b: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound.75f: <bound method> = bound_method %int_15.447, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.984: <bound method> = bound_method %int_15.447, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_15.7f7: %i32 = int_value 15 [concrete]
// CHECK:STDOUT:   %Convert.bound.76a: <bound method> = bound_method %int_35.c79, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.231: <bound method> = bound_method %int_35.c79, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_35.c78: %i32 = int_value 35 [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.d07 = tuple_value (%int_15.7f7, %int_35.c78) [concrete]
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
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Main.decl: %Main.type = fn_decl @Main [concrete = constants.%Main] {
// CHECK:STDOUT:     %return.patt: %tuple.type.d07 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %tuple.type.d07 = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc12_15: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc12_15: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc12_20: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc12_20: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc12_23.1: %tuple.type.24b = tuple_literal (%i32.loc12_15, %i32.loc12_20)
// CHECK:STDOUT:     %.loc12_23.2: type = converted %.loc12_23.1, constants.%tuple.type.d07 [concrete = constants.%tuple.type.d07]
// CHECK:STDOUT:     %return.param: ref %tuple.type.d07 = out_param call_param0
// CHECK:STDOUT:     %return: ref %tuple.type.d07 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() -> %return.param_patt: %tuple.type.d07 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_15: Core.IntLiteral = int_value 15 [concrete = constants.%int_15.447]
// CHECK:STDOUT:   %int_35: Core.IntLiteral = int_value 35 [concrete = constants.%int_35.c79]
// CHECK:STDOUT:   %.loc13_17.1: %tuple.type.f94 = tuple_literal (%int_15, %int_35)
// CHECK:STDOUT:   %impl.elem0.loc13_17.1: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc13_17.1: <bound method> = bound_method %int_15, %impl.elem0.loc13_17.1 [concrete = constants.%Convert.bound.75f]
// CHECK:STDOUT:   %specific_fn.loc13_17.1: <specific function> = specific_function %impl.elem0.loc13_17.1, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_17.2: <bound method> = bound_method %int_15, %specific_fn.loc13_17.1 [concrete = constants.%bound_method.984]
// CHECK:STDOUT:   %int.convert_checked.loc13_17.1: init %i32 = call %bound_method.loc13_17.2(%int_15) [concrete = constants.%int_15.7f7]
// CHECK:STDOUT:   %.loc13_17.2: init %i32 = converted %int_15, %int.convert_checked.loc13_17.1 [concrete = constants.%int_15.7f7]
// CHECK:STDOUT:   %tuple.elem0: ref %i32 = tuple_access %return, element0
// CHECK:STDOUT:   %.loc13_17.3: init %i32 = initialize_from %.loc13_17.2 to %tuple.elem0 [concrete = constants.%int_15.7f7]
// CHECK:STDOUT:   %impl.elem0.loc13_17.2: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc13_17.3: <bound method> = bound_method %int_35, %impl.elem0.loc13_17.2 [concrete = constants.%Convert.bound.76a]
// CHECK:STDOUT:   %specific_fn.loc13_17.2: <specific function> = specific_function %impl.elem0.loc13_17.2, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_17.4: <bound method> = bound_method %int_35, %specific_fn.loc13_17.2 [concrete = constants.%bound_method.231]
// CHECK:STDOUT:   %int.convert_checked.loc13_17.2: init %i32 = call %bound_method.loc13_17.4(%int_35) [concrete = constants.%int_35.c78]
// CHECK:STDOUT:   %.loc13_17.4: init %i32 = converted %int_35, %int.convert_checked.loc13_17.2 [concrete = constants.%int_35.c78]
// CHECK:STDOUT:   %tuple.elem1: ref %i32 = tuple_access %return, element1
// CHECK:STDOUT:   %.loc13_17.5: init %i32 = initialize_from %.loc13_17.4 to %tuple.elem1 [concrete = constants.%int_35.c78]
// CHECK:STDOUT:   %.loc13_17.6: init %tuple.type.d07 = tuple_init (%.loc13_17.3, %.loc13_17.5) to %return [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc13_18: init %tuple.type.d07 = converted %.loc13_17.1, %.loc13_17.6 [concrete = constants.%tuple]
// CHECK:STDOUT:   return %.loc13_18 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:

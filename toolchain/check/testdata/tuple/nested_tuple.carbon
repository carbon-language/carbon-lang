// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/tuple/nested_tuple.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/tuple/nested_tuple.carbon

var x: ((i32, i32), i32) = ((12, 76), 6);

// CHECK:STDOUT: --- nested_tuple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.45d: type = tuple_type (%tuple.type.24b, type) [concrete]
// CHECK:STDOUT:   %tuple.type.d07: type = tuple_type (%i32, %i32) [concrete]
// CHECK:STDOUT:   %tuple.type.f72: type = tuple_type (%tuple.type.d07, %i32) [concrete]
// CHECK:STDOUT:   %int_12.6a3: Core.IntLiteral = int_value 12 [concrete]
// CHECK:STDOUT:   %int_76.872: Core.IntLiteral = int_value 76 [concrete]
// CHECK:STDOUT:   %tuple.type.f94: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %int_6.462: Core.IntLiteral = int_value 6 [concrete]
// CHECK:STDOUT:   %tuple.type.acf: type = tuple_type (%tuple.type.f94, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %impl_witness.d39: <witness> = impl_witness (imports.%Core.import_ref.a5b), @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, %impl_witness.d39 [concrete]
// CHECK:STDOUT:   %.a0b: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound.221: <bound method> = bound_method %int_12.6a3, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.dae: <bound method> = bound_method %int_12.6a3, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_12.1e1: %i32 = int_value 12 [concrete]
// CHECK:STDOUT:   %Convert.bound.033: <bound method> = bound_method %int_76.872, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.ce4: <bound method> = bound_method %int_76.872, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_76.8dd: %i32 = int_value 76 [concrete]
// CHECK:STDOUT:   %tuple.737: %tuple.type.d07 = tuple_value (%int_12.1e1, %int_76.8dd) [concrete]
// CHECK:STDOUT:   %Convert.bound.ce9: <bound method> = bound_method %int_6.462, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.efa: <bound method> = bound_method %int_6.462, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_6.e56: %i32 = int_value 6 [concrete]
// CHECK:STDOUT:   %tuple.51f: %tuple.type.f72 = tuple_value (%tuple.737, %int_6.e56) [concrete]
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
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %tuple.type.f72 = binding_pattern x
// CHECK:STDOUT:     %.loc11_1: %tuple.type.f72 = var_pattern %x.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %tuple.type.f72 = var x
// CHECK:STDOUT:   %.loc11_24.1: type = splice_block %.loc11_24.4 [concrete = constants.%tuple.type.f72] {
// CHECK:STDOUT:     %int_32.loc11_10: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc11_10: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc11_15: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc11_15: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc11_18: %tuple.type.24b = tuple_literal (%i32.loc11_10, %i32.loc11_15)
// CHECK:STDOUT:     %int_32.loc11_21: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc11_21: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc11_24.2: %tuple.type.45d = tuple_literal (%.loc11_18, %i32.loc11_21)
// CHECK:STDOUT:     %.loc11_24.3: type = converted %.loc11_18, constants.%tuple.type.d07 [concrete = constants.%tuple.type.d07]
// CHECK:STDOUT:     %.loc11_24.4: type = converted %.loc11_24.2, constants.%tuple.type.f72 [concrete = constants.%tuple.type.f72]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %tuple.type.f72 = bind_name x, %x.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_12: Core.IntLiteral = int_value 12 [concrete = constants.%int_12.6a3]
// CHECK:STDOUT:   %int_76: Core.IntLiteral = int_value 76 [concrete = constants.%int_76.872]
// CHECK:STDOUT:   %.loc11_36.1: %tuple.type.f94 = tuple_literal (%int_12, %int_76)
// CHECK:STDOUT:   %int_6: Core.IntLiteral = int_value 6 [concrete = constants.%int_6.462]
// CHECK:STDOUT:   %.loc11_40.1: %tuple.type.acf = tuple_literal (%.loc11_36.1, %int_6)
// CHECK:STDOUT:   %impl.elem0.loc11_36.1: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc11_36.1: <bound method> = bound_method %int_12, %impl.elem0.loc11_36.1 [concrete = constants.%Convert.bound.221]
// CHECK:STDOUT:   %specific_fn.loc11_36.1: <specific function> = specific_function %impl.elem0.loc11_36.1, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_36.2: <bound method> = bound_method %int_12, %specific_fn.loc11_36.1 [concrete = constants.%bound_method.dae]
// CHECK:STDOUT:   %int.convert_checked.loc11_36.1: init %i32 = call %bound_method.loc11_36.2(%int_12) [concrete = constants.%int_12.1e1]
// CHECK:STDOUT:   %.loc11_36.2: init %i32 = converted %int_12, %int.convert_checked.loc11_36.1 [concrete = constants.%int_12.1e1]
// CHECK:STDOUT:   %tuple.elem0.loc11_40: ref %tuple.type.d07 = tuple_access file.%x.var, element0
// CHECK:STDOUT:   %tuple.elem0.loc11_36: ref %i32 = tuple_access %tuple.elem0.loc11_40, element0
// CHECK:STDOUT:   %.loc11_36.3: init %i32 = initialize_from %.loc11_36.2 to %tuple.elem0.loc11_36 [concrete = constants.%int_12.1e1]
// CHECK:STDOUT:   %impl.elem0.loc11_36.2: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc11_36.3: <bound method> = bound_method %int_76, %impl.elem0.loc11_36.2 [concrete = constants.%Convert.bound.033]
// CHECK:STDOUT:   %specific_fn.loc11_36.2: <specific function> = specific_function %impl.elem0.loc11_36.2, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_36.4: <bound method> = bound_method %int_76, %specific_fn.loc11_36.2 [concrete = constants.%bound_method.ce4]
// CHECK:STDOUT:   %int.convert_checked.loc11_36.2: init %i32 = call %bound_method.loc11_36.4(%int_76) [concrete = constants.%int_76.8dd]
// CHECK:STDOUT:   %.loc11_36.4: init %i32 = converted %int_76, %int.convert_checked.loc11_36.2 [concrete = constants.%int_76.8dd]
// CHECK:STDOUT:   %tuple.elem1.loc11_36: ref %i32 = tuple_access %tuple.elem0.loc11_40, element1
// CHECK:STDOUT:   %.loc11_36.5: init %i32 = initialize_from %.loc11_36.4 to %tuple.elem1.loc11_36 [concrete = constants.%int_76.8dd]
// CHECK:STDOUT:   %.loc11_36.6: init %tuple.type.d07 = tuple_init (%.loc11_36.3, %.loc11_36.5) to %tuple.elem0.loc11_40 [concrete = constants.%tuple.737]
// CHECK:STDOUT:   %.loc11_40.2: init %tuple.type.d07 = converted %.loc11_36.1, %.loc11_36.6 [concrete = constants.%tuple.737]
// CHECK:STDOUT:   %impl.elem0.loc11_40: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc11_40.1: <bound method> = bound_method %int_6, %impl.elem0.loc11_40 [concrete = constants.%Convert.bound.ce9]
// CHECK:STDOUT:   %specific_fn.loc11_40: <specific function> = specific_function %impl.elem0.loc11_40, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_40.2: <bound method> = bound_method %int_6, %specific_fn.loc11_40 [concrete = constants.%bound_method.efa]
// CHECK:STDOUT:   %int.convert_checked.loc11_40: init %i32 = call %bound_method.loc11_40.2(%int_6) [concrete = constants.%int_6.e56]
// CHECK:STDOUT:   %.loc11_40.3: init %i32 = converted %int_6, %int.convert_checked.loc11_40 [concrete = constants.%int_6.e56]
// CHECK:STDOUT:   %tuple.elem1.loc11_40: ref %i32 = tuple_access file.%x.var, element1
// CHECK:STDOUT:   %.loc11_40.4: init %i32 = initialize_from %.loc11_40.3 to %tuple.elem1.loc11_40 [concrete = constants.%int_6.e56]
// CHECK:STDOUT:   %.loc11_40.5: init %tuple.type.f72 = tuple_init (%.loc11_40.2, %.loc11_40.4) to file.%x.var [concrete = constants.%tuple.51f]
// CHECK:STDOUT:   %.loc11_1: init %tuple.type.f72 = converted %.loc11_40.1, %.loc11_40.5 [concrete = constants.%tuple.51f]
// CHECK:STDOUT:   assign file.%x.var, %.loc11_1
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

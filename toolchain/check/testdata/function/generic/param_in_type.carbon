// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/generic/param_in_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/generic/param_in_type.carbon

fn F(N:! i32, a: array(i32, N)*);

// CHECK:STDOUT: --- param_in_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %N.51e: %i32 = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt.8e2: %i32 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.2fd: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %Convert.type.71e: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %impl_witness.023: <witness> = impl_witness (imports.%Core.import_ref.85c), @impl.971(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.4ad: type = fn_type @Convert.3, @impl.971(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.960: %Convert.type.4ad = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.2fd = facet_value %i32, (%impl_witness.023) [concrete]
// CHECK:STDOUT:   %.4d8: type = fn_type_with_self_type %Convert.type.71e, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %N.51e, %Convert.960 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.960, @Convert.3(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %N.51e, %Convert.specific_fn [symbolic]
// CHECK:STDOUT:   %int.convert_checked: init Core.IntLiteral = call %bound_method(%N.51e) [symbolic]
// CHECK:STDOUT:   %array_type: type = array_type %int.convert_checked, %i32 [symbolic]
// CHECK:STDOUT:   %ptr: type = ptr_type %array_type [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
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
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %N.patt.loc11_6.1: %i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc11_6.2 (constants.%N.patt.8e2)]
// CHECK:STDOUT:     %a.patt: @F.%ptr.loc11_31.2 (%ptr) = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: @F.%ptr.loc11_31.2 (%ptr) = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11_10: type = splice_block %i32.loc11_10 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc11_10: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc11_10: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc11_6.1: %i32 = bind_symbolic_name N, 0 [symbolic = %N.loc11_6.2 (constants.%N.51e)]
// CHECK:STDOUT:     %a.param: @F.%ptr.loc11_31.2 (%ptr) = value_param call_param0
// CHECK:STDOUT:     %.loc11_31: type = splice_block %ptr.loc11_31.1 [symbolic = %ptr.loc11_31.2 (constants.%ptr)] {
// CHECK:STDOUT:       %int_32.loc11_24: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc11_24: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %N.ref: %i32 = name_ref N, %N.loc11_6.1 [symbolic = %N.loc11_6.2 (constants.%N.51e)]
// CHECK:STDOUT:       %impl.elem0: %.4d8 = impl_witness_access constants.%impl_witness.023, element0 [concrete = constants.%Convert.960]
// CHECK:STDOUT:       %bound_method.loc11_29.1: <bound method> = bound_method %N.ref, %impl.elem0 [symbolic = %Convert.bound (constants.%Convert.bound)]
// CHECK:STDOUT:       %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.3(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:       %bound_method.loc11_29.2: <bound method> = bound_method %N.ref, %specific_fn [symbolic = %bound_method.loc11_29.3 (constants.%bound_method)]
// CHECK:STDOUT:       %int.convert_checked.loc11_29.1: init Core.IntLiteral = call %bound_method.loc11_29.2(%N.ref) [symbolic = %int.convert_checked.loc11_29.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:       %.loc11_29.1: Core.IntLiteral = value_of_initializer %int.convert_checked.loc11_29.1 [symbolic = %int.convert_checked.loc11_29.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:       %.loc11_29.2: Core.IntLiteral = converted %N.ref, %.loc11_29.1 [symbolic = %int.convert_checked.loc11_29.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:       %array_type.loc11_30.1: type = array_type %.loc11_29.2, %i32.loc11_24 [symbolic = %array_type.loc11_30.2 (constants.%array_type)]
// CHECK:STDOUT:       %ptr.loc11_31.1: type = ptr_type %array_type.loc11_30.1 [symbolic = %ptr.loc11_31.2 (constants.%ptr)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @F.%ptr.loc11_31.2 (%ptr) = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%N.loc11_6.1: %i32) {
// CHECK:STDOUT:   %N.loc11_6.2: %i32 = bind_symbolic_name N, 0 [symbolic = %N.loc11_6.2 (constants.%N.51e)]
// CHECK:STDOUT:   %N.patt.loc11_6.2: %i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc11_6.2 (constants.%N.patt.8e2)]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %N.loc11_6.2, constants.%Convert.960 [symbolic = %Convert.bound (constants.%Convert.bound)]
// CHECK:STDOUT:   %bound_method.loc11_29.3: <bound method> = bound_method %N.loc11_6.2, constants.%Convert.specific_fn [symbolic = %bound_method.loc11_29.3 (constants.%bound_method)]
// CHECK:STDOUT:   %int.convert_checked.loc11_29.2: init Core.IntLiteral = call %bound_method.loc11_29.3(%N.loc11_6.2) [symbolic = %int.convert_checked.loc11_29.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:   %array_type.loc11_30.2: type = array_type %int.convert_checked.loc11_29.2, constants.%i32 [symbolic = %array_type.loc11_30.2 (constants.%array_type)]
// CHECK:STDOUT:   %ptr.loc11_31.2: type = ptr_type %array_type.loc11_30.2 [symbolic = %ptr.loc11_31.2 (constants.%ptr)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%N.patt.loc11_6.1: %i32, %a.param_patt: @F.%ptr.loc11_31.2 (%ptr));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%N.51e) {
// CHECK:STDOUT:   %N.loc11_6.2 => constants.%N.51e
// CHECK:STDOUT:   %N.patt.loc11_6.2 => constants.%N.patt.8e2
// CHECK:STDOUT:   %Convert.bound => constants.%Convert.bound
// CHECK:STDOUT:   %bound_method.loc11_29.3 => constants.%bound_method
// CHECK:STDOUT:   %int.convert_checked.loc11_29.2 => constants.%int.convert_checked
// CHECK:STDOUT:   %array_type.loc11_30.2 => constants.%array_type
// CHECK:STDOUT:   %ptr.loc11_31.2 => constants.%ptr
// CHECK:STDOUT: }
// CHECK:STDOUT:

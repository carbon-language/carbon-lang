// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/eval/symbolic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/eval/symbolic.carbon

// Check that we propagate the `symbolic` tag through evaluations.
fn F(T:! type) {
  var u: (T*, const T);
  var v: {.a: T};
  var w: array(T, 5);
}

fn G(N:! i32) {
  var k: array(i32, N);
}

// CHECK:STDOUT: --- symbolic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.79f: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %const: type = const_type %T [symbolic]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.4f2: type = tuple_type (%ptr.79f, %const) [symbolic]
// CHECK:STDOUT:   %require_complete.155: <witness> = require_complete_type %tuple.type.4f2 [symbolic]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %T} [symbolic]
// CHECK:STDOUT:   %require_complete.28a: <witness> = require_complete_type %struct_type.a [symbolic]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %array_type.ec2: type = array_type %int_5, %T [symbolic]
// CHECK:STDOUT:   %require_complete.fe1: <witness> = require_complete_type %array_type.ec2 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %N.51e: %i32 = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt.8e2: %i32 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.2fd: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %Convert.type.71e: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %impl_witness.023: <witness> = impl_witness (imports.%Core.import_ref.85c), @impl.2(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.4ad: type = fn_type @Convert.3, @impl.2(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.960: %Convert.type.4ad = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.2fd = facet_value %i32, %impl_witness.023 [concrete]
// CHECK:STDOUT:   %.10e: type = fn_type_with_self_type %Convert.type.71e, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %N.51e, %Convert.960 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.3(%int_32) [symbolic]
// CHECK:STDOUT:   %int.convert_checked: init Core.IntLiteral = call %Convert.specific_fn(%N.51e) [symbolic]
// CHECK:STDOUT:   %array_type.b04: type = array_type %int.convert_checked, %i32 [symbolic]
// CHECK:STDOUT:   %require_complete.9dc: <witness> = require_complete_type %array_type.b04 [symbolic]
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
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc12_6.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc12_6.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc12_6.1, runtime_param<none> [symbolic = %T.patt.loc12_6.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<none>
// CHECK:STDOUT:     %T.loc12_6.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc12_6.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %N.patt.loc18_6.1: %i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc18_6.2 (constants.%N.patt.8e2)]
// CHECK:STDOUT:     %N.param_patt: %i32 = value_param_pattern %N.patt.loc18_6.1, runtime_param<none> [symbolic = %N.patt.loc18_6.2 (constants.%N.patt.8e2)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %N.param: %i32 = value_param runtime_param<none>
// CHECK:STDOUT:     %.loc18: type = splice_block %i32.loc18 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc18: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc18: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc18_6.1: %i32 = bind_symbolic_name N, 0, %N.param [symbolic = %N.loc18_6.2 (constants.%N.51e)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc12_6.1: type) {
// CHECK:STDOUT:   %T.loc12_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc12_6.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc12_6.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc12_6.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ptr.loc13_12.2: type = ptr_type @F.%T.loc12_6.2 (%T) [symbolic = %ptr.loc13_12.2 (constants.%ptr.79f)]
// CHECK:STDOUT:   %const.loc13_15.2: type = const_type @F.%T.loc12_6.2 (%T) [symbolic = %const.loc13_15.2 (constants.%const)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (@F.%ptr.loc13_12.2 (%ptr.79f), @F.%const.loc13_15.2 (%const)) [symbolic = %tuple.type (constants.%tuple.type.4f2)]
// CHECK:STDOUT:   %require_complete.loc13: <witness> = require_complete_type @F.%tuple.type (%tuple.type.4f2) [symbolic = %require_complete.loc13 (constants.%require_complete.155)]
// CHECK:STDOUT:   %struct_type.a.loc14_16.2: type = struct_type {.a: @F.%T.loc12_6.2 (%T)} [symbolic = %struct_type.a.loc14_16.2 (constants.%struct_type.a)]
// CHECK:STDOUT:   %require_complete.loc14: <witness> = require_complete_type @F.%struct_type.a.loc14_16.2 (%struct_type.a) [symbolic = %require_complete.loc14 (constants.%require_complete.28a)]
// CHECK:STDOUT:   %array_type.loc15_20.2: type = array_type constants.%int_5, @F.%T.loc12_6.2 (%T) [symbolic = %array_type.loc15_20.2 (constants.%array_type.ec2)]
// CHECK:STDOUT:   %require_complete.loc15: <witness> = require_complete_type @F.%array_type.loc15_20.2 (%array_type.ec2) [symbolic = %require_complete.loc15 (constants.%require_complete.fe1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.param_patt: type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %u.patt: @F.%tuple.type (%tuple.type.4f2) = binding_pattern u
// CHECK:STDOUT:       %.loc13_3: @F.%tuple.type (%tuple.type.4f2) = var_pattern %u.patt
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u.var: ref @F.%tuple.type (%tuple.type.4f2) = var u
// CHECK:STDOUT:     %.loc13_22.1: type = splice_block %.loc13_22.3 [symbolic = %tuple.type (constants.%tuple.type.4f2)] {
// CHECK:STDOUT:       %T.ref.loc13_11: type = name_ref T, %T.loc12_6.1 [symbolic = %T.loc12_6.2 (constants.%T)]
// CHECK:STDOUT:       %ptr.loc13_12.1: type = ptr_type %T [symbolic = %ptr.loc13_12.2 (constants.%ptr.79f)]
// CHECK:STDOUT:       %T.ref.loc13_21: type = name_ref T, %T.loc12_6.1 [symbolic = %T.loc12_6.2 (constants.%T)]
// CHECK:STDOUT:       %const.loc13_15.1: type = const_type %T [symbolic = %const.loc13_15.2 (constants.%const)]
// CHECK:STDOUT:       %.loc13_22.2: %tuple.type.24b = tuple_literal (%ptr.loc13_12.1, %const.loc13_15.1)
// CHECK:STDOUT:       %.loc13_22.3: type = converted %.loc13_22.2, constants.%tuple.type.4f2 [symbolic = %tuple.type (constants.%tuple.type.4f2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: ref @F.%tuple.type (%tuple.type.4f2) = bind_name u, %u.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %v.patt: @F.%struct_type.a.loc14_16.2 (%struct_type.a) = binding_pattern v
// CHECK:STDOUT:       %.loc14_3: @F.%struct_type.a.loc14_16.2 (%struct_type.a) = var_pattern %v.patt
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v.var: ref @F.%struct_type.a.loc14_16.2 (%struct_type.a) = var v
// CHECK:STDOUT:     %.loc14_16: type = splice_block %struct_type.a.loc14_16.1 [symbolic = %struct_type.a.loc14_16.2 (constants.%struct_type.a)] {
// CHECK:STDOUT:       %T.ref.loc14: type = name_ref T, %T.loc12_6.1 [symbolic = %T.loc12_6.2 (constants.%T)]
// CHECK:STDOUT:       %struct_type.a.loc14_16.1: type = struct_type {.a: %T} [symbolic = %struct_type.a.loc14_16.2 (constants.%struct_type.a)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: ref @F.%struct_type.a.loc14_16.2 (%struct_type.a) = bind_name v, %v.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %w.patt: @F.%array_type.loc15_20.2 (%array_type.ec2) = binding_pattern w
// CHECK:STDOUT:       %.loc15_3: @F.%array_type.loc15_20.2 (%array_type.ec2) = var_pattern %w.patt
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %w.var: ref @F.%array_type.loc15_20.2 (%array_type.ec2) = var w
// CHECK:STDOUT:     %.loc15_20: type = splice_block %array_type.loc15_20.1 [symbolic = %array_type.loc15_20.2 (constants.%array_type.ec2)] {
// CHECK:STDOUT:       %T.ref.loc15: type = name_ref T, %T.loc12_6.1 [symbolic = %T.loc12_6.2 (constants.%T)]
// CHECK:STDOUT:       %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5]
// CHECK:STDOUT:       %array_type.loc15_20.1: type = array_type %int_5, %T [symbolic = %array_type.loc15_20.2 (constants.%array_type.ec2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %w: ref @F.%array_type.loc15_20.2 (%array_type.ec2) = bind_name w, %w.var
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%N.loc18_6.1: %i32) {
// CHECK:STDOUT:   %N.loc18_6.2: %i32 = bind_symbolic_name N, 0 [symbolic = %N.loc18_6.2 (constants.%N.51e)]
// CHECK:STDOUT:   %N.patt.loc18_6.2: %i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc18_6.2 (constants.%N.patt.8e2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %N.loc18_6.2, constants.%Convert.960 [symbolic = %Convert.bound (constants.%Convert.bound)]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.3(constants.%int_32) [symbolic = %Convert.specific_fn (constants.%Convert.specific_fn)]
// CHECK:STDOUT:   %int.convert_checked.loc19_21.2: init Core.IntLiteral = call %Convert.specific_fn(%N.loc18_6.2) [symbolic = %int.convert_checked.loc19_21.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:   %array_type.loc19_22.2: type = array_type %int.convert_checked.loc19_21.2, %i32 [symbolic = %array_type.loc19_22.2 (constants.%array_type.b04)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @G.%array_type.loc19_22.2 (%array_type.b04) [symbolic = %require_complete (constants.%require_complete.9dc)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%N.param_patt: %i32) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %k.patt: @G.%array_type.loc19_22.2 (%array_type.b04) = binding_pattern k
// CHECK:STDOUT:       %.loc19_3: @G.%array_type.loc19_22.2 (%array_type.b04) = var_pattern %k.patt
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %k.var: ref @G.%array_type.loc19_22.2 (%array_type.b04) = var k
// CHECK:STDOUT:     %.loc19_22: type = splice_block %array_type.loc19_22.1 [symbolic = %array_type.loc19_22.2 (constants.%array_type.b04)] {
// CHECK:STDOUT:       %int_32.loc19: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc19: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %N.ref: %i32 = name_ref N, %N.loc18_6.1 [symbolic = %N.loc18_6.2 (constants.%N.51e)]
// CHECK:STDOUT:       %impl.elem0: %.10e = impl_witness_access constants.%impl_witness.023, element0 [concrete = constants.%Convert.960]
// CHECK:STDOUT:       %bound_method: <bound method> = bound_method %N.ref, %impl.elem0 [symbolic = %Convert.bound (constants.%Convert.bound)]
// CHECK:STDOUT:       %specific_fn: <specific function> = specific_function %bound_method, @Convert.3(constants.%int_32) [symbolic = %Convert.specific_fn (constants.%Convert.specific_fn)]
// CHECK:STDOUT:       %int.convert_checked.loc19_21.1: init Core.IntLiteral = call %specific_fn(%N.ref) [symbolic = %int.convert_checked.loc19_21.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:       %.loc19_21.1: Core.IntLiteral = value_of_initializer %int.convert_checked.loc19_21.1 [symbolic = %int.convert_checked.loc19_21.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:       %.loc19_21.2: Core.IntLiteral = converted %N.ref, %.loc19_21.1 [symbolic = %int.convert_checked.loc19_21.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:       %array_type.loc19_22.1: type = array_type %.loc19_21.2, %i32 [symbolic = %array_type.loc19_22.2 (constants.%array_type.b04)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %k: ref @G.%array_type.loc19_22.2 (%array_type.b04) = bind_name k, %k.var
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc12_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc12_6.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%N.51e) {
// CHECK:STDOUT:   %N.loc18_6.2 => constants.%N.51e
// CHECK:STDOUT:   %N.patt.loc18_6.2 => constants.%N.51e
// CHECK:STDOUT: }
// CHECK:STDOUT:

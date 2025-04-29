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
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.79f: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %const: type = const_type %T [symbolic]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.4f2: type = tuple_type (%ptr.79f, %const) [symbolic]
// CHECK:STDOUT:   %require_complete.155: <witness> = require_complete_type %tuple.type.4f2 [symbolic]
// CHECK:STDOUT:   %pattern_type.973: type = pattern_type %tuple.type.4f2 [symbolic]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %T} [symbolic]
// CHECK:STDOUT:   %require_complete.28a: <witness> = require_complete_type %struct_type.a [symbolic]
// CHECK:STDOUT:   %pattern_type.95a: type = pattern_type %struct_type.a [symbolic]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %array_type.ec2: type = array_type %int_5, %T [symbolic]
// CHECK:STDOUT:   %require_complete.fe1: <witness> = require_complete_type %array_type.ec2 [symbolic]
// CHECK:STDOUT:   %pattern_type.035: type = pattern_type %array_type.ec2 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %N.51e: %i32 = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.2fd: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %Convert.type.71e: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1d9 = impl_witness_table (imports.%Core.import_ref.85c), @impl.971 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a11: <witness> = impl_witness %ImplicitAs.impl_witness_table.1d9, @impl.971(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.4ad: type = fn_type @Convert.3, @impl.971(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.960: %Convert.type.4ad = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.2fd = facet_value %i32, (%ImplicitAs.impl_witness.a11) [concrete]
// CHECK:STDOUT:   %.0ea: type = fn_type_with_self_type %Convert.type.71e, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %N.51e, %Convert.960 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.960, @Convert.3(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %N.51e, %Convert.specific_fn [symbolic]
// CHECK:STDOUT:   %int.convert_checked: init Core.IntLiteral = call %bound_method(%N.51e) [symbolic]
// CHECK:STDOUT:   %array_type.c7c: type = array_type %int.convert_checked, %i32 [symbolic]
// CHECK:STDOUT:   %require_complete.7cb: <witness> = require_complete_type %array_type.c7c [symbolic]
// CHECK:STDOUT:   %pattern_type.ccc: type = pattern_type %array_type.c7c [symbolic]
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
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc12_6.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc12_6.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %N.patt: %pattern_type.7ce = symbolic_binding_pattern N, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc18: type = splice_block %i32.loc18 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc18: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc18: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc18_6.1: %i32 = bind_symbolic_name N, 0 [symbolic = %N.loc18_6.2 (constants.%N.51e)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc12_6.1: type) {
// CHECK:STDOUT:   %T.loc12_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc12_6.2 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ptr.loc13_12.2: type = ptr_type %T.loc12_6.2 [symbolic = %ptr.loc13_12.2 (constants.%ptr.79f)]
// CHECK:STDOUT:   %const.loc13_15.2: type = const_type %T.loc12_6.2 [symbolic = %const.loc13_15.2 (constants.%const)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%ptr.loc13_12.2, %const.loc13_15.2) [symbolic = %tuple.type (constants.%tuple.type.4f2)]
// CHECK:STDOUT:   %require_complete.loc13: <witness> = require_complete_type %tuple.type [symbolic = %require_complete.loc13 (constants.%require_complete.155)]
// CHECK:STDOUT:   %pattern_type.loc13: type = pattern_type %tuple.type [symbolic = %pattern_type.loc13 (constants.%pattern_type.973)]
// CHECK:STDOUT:   %struct_type.a.loc14_16.2: type = struct_type {.a: @F.%T.loc12_6.2 (%T)} [symbolic = %struct_type.a.loc14_16.2 (constants.%struct_type.a)]
// CHECK:STDOUT:   %require_complete.loc14: <witness> = require_complete_type %struct_type.a.loc14_16.2 [symbolic = %require_complete.loc14 (constants.%require_complete.28a)]
// CHECK:STDOUT:   %pattern_type.loc14: type = pattern_type %struct_type.a.loc14_16.2 [symbolic = %pattern_type.loc14 (constants.%pattern_type.95a)]
// CHECK:STDOUT:   %array_type.loc15_20.2: type = array_type constants.%int_5, %T.loc12_6.2 [symbolic = %array_type.loc15_20.2 (constants.%array_type.ec2)]
// CHECK:STDOUT:   %require_complete.loc15: <witness> = require_complete_type %array_type.loc15_20.2 [symbolic = %require_complete.loc15 (constants.%require_complete.fe1)]
// CHECK:STDOUT:   %pattern_type.loc15: type = pattern_type %array_type.loc15_20.2 [symbolic = %pattern_type.loc15 (constants.%pattern_type.035)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %u.patt: @F.%pattern_type.loc13 (%pattern_type.973) = binding_pattern u [concrete]
// CHECK:STDOUT:       %.loc13_3: @F.%pattern_type.loc13 (%pattern_type.973) = var_pattern %u.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u.var: ref @F.%tuple.type (%tuple.type.4f2) = var u
// CHECK:STDOUT:     %.loc13_22.1: type = splice_block %.loc13_22.3 [symbolic = %tuple.type (constants.%tuple.type.4f2)] {
// CHECK:STDOUT:       %T.ref.loc13_11: type = name_ref T, %T.loc12_6.1 [symbolic = %T.loc12_6.2 (constants.%T)]
// CHECK:STDOUT:       %ptr.loc13_12.1: type = ptr_type %T.ref.loc13_11 [symbolic = %ptr.loc13_12.2 (constants.%ptr.79f)]
// CHECK:STDOUT:       %T.ref.loc13_21: type = name_ref T, %T.loc12_6.1 [symbolic = %T.loc12_6.2 (constants.%T)]
// CHECK:STDOUT:       %const.loc13_15.1: type = const_type %T.ref.loc13_21 [symbolic = %const.loc13_15.2 (constants.%const)]
// CHECK:STDOUT:       %.loc13_22.2: %tuple.type.24b = tuple_literal (%ptr.loc13_12.1, %const.loc13_15.1)
// CHECK:STDOUT:       %.loc13_22.3: type = converted %.loc13_22.2, constants.%tuple.type.4f2 [symbolic = %tuple.type (constants.%tuple.type.4f2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: ref @F.%tuple.type (%tuple.type.4f2) = bind_name u, %u.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %v.patt: @F.%pattern_type.loc14 (%pattern_type.95a) = binding_pattern v [concrete]
// CHECK:STDOUT:       %.loc14_3: @F.%pattern_type.loc14 (%pattern_type.95a) = var_pattern %v.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v.var: ref @F.%struct_type.a.loc14_16.2 (%struct_type.a) = var v
// CHECK:STDOUT:     %.loc14_16: type = splice_block %struct_type.a.loc14_16.1 [symbolic = %struct_type.a.loc14_16.2 (constants.%struct_type.a)] {
// CHECK:STDOUT:       %T.ref.loc14: type = name_ref T, %T.loc12_6.1 [symbolic = %T.loc12_6.2 (constants.%T)]
// CHECK:STDOUT:       %struct_type.a.loc14_16.1: type = struct_type {.a: @F.%T.loc12_6.2 (%T)} [symbolic = %struct_type.a.loc14_16.2 (constants.%struct_type.a)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: ref @F.%struct_type.a.loc14_16.2 (%struct_type.a) = bind_name v, %v.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %w.patt: @F.%pattern_type.loc15 (%pattern_type.035) = binding_pattern w [concrete]
// CHECK:STDOUT:       %.loc15_3: @F.%pattern_type.loc15 (%pattern_type.035) = var_pattern %w.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %w.var: ref @F.%array_type.loc15_20.2 (%array_type.ec2) = var w
// CHECK:STDOUT:     %.loc15_20: type = splice_block %array_type.loc15_20.1 [symbolic = %array_type.loc15_20.2 (constants.%array_type.ec2)] {
// CHECK:STDOUT:       %T.ref.loc15: type = name_ref T, %T.loc12_6.1 [symbolic = %T.loc12_6.2 (constants.%T)]
// CHECK:STDOUT:       %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5]
// CHECK:STDOUT:       %array_type.loc15_20.1: type = array_type %int_5, %T.ref.loc15 [symbolic = %array_type.loc15_20.2 (constants.%array_type.ec2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %w: ref @F.%array_type.loc15_20.2 (%array_type.ec2) = bind_name w, %w.var
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%N.loc18_6.1: %i32) {
// CHECK:STDOUT:   %N.loc18_6.2: %i32 = bind_symbolic_name N, 0 [symbolic = %N.loc18_6.2 (constants.%N.51e)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %N.loc18_6.2, constants.%Convert.960 [symbolic = %Convert.bound (constants.%Convert.bound)]
// CHECK:STDOUT:   %bound_method.loc19_21.3: <bound method> = bound_method %N.loc18_6.2, constants.%Convert.specific_fn [symbolic = %bound_method.loc19_21.3 (constants.%bound_method)]
// CHECK:STDOUT:   %int.convert_checked.loc19_21.2: init Core.IntLiteral = call %bound_method.loc19_21.3(%N.loc18_6.2) [symbolic = %int.convert_checked.loc19_21.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:   %array_type.loc19_22.2: type = array_type %int.convert_checked.loc19_21.2, constants.%i32 [symbolic = %array_type.loc19_22.2 (constants.%array_type.c7c)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %array_type.loc19_22.2 [symbolic = %require_complete (constants.%require_complete.7cb)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %array_type.loc19_22.2 [symbolic = %pattern_type (constants.%pattern_type.ccc)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %k.patt: @G.%pattern_type (%pattern_type.ccc) = binding_pattern k [concrete]
// CHECK:STDOUT:       %.loc19_3: @G.%pattern_type (%pattern_type.ccc) = var_pattern %k.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %k.var: ref @G.%array_type.loc19_22.2 (%array_type.c7c) = var k
// CHECK:STDOUT:     %.loc19_22: type = splice_block %array_type.loc19_22.1 [symbolic = %array_type.loc19_22.2 (constants.%array_type.c7c)] {
// CHECK:STDOUT:       %int_32.loc19: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc19: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %N.ref: %i32 = name_ref N, %N.loc18_6.1 [symbolic = %N.loc18_6.2 (constants.%N.51e)]
// CHECK:STDOUT:       %impl.elem0: %.0ea = impl_witness_access constants.%ImplicitAs.impl_witness.a11, element0 [concrete = constants.%Convert.960]
// CHECK:STDOUT:       %bound_method.loc19_21.1: <bound method> = bound_method %N.ref, %impl.elem0 [symbolic = %Convert.bound (constants.%Convert.bound)]
// CHECK:STDOUT:       %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.3(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:       %bound_method.loc19_21.2: <bound method> = bound_method %N.ref, %specific_fn [symbolic = %bound_method.loc19_21.3 (constants.%bound_method)]
// CHECK:STDOUT:       %int.convert_checked.loc19_21.1: init Core.IntLiteral = call %bound_method.loc19_21.2(%N.ref) [symbolic = %int.convert_checked.loc19_21.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:       %.loc19_21.1: Core.IntLiteral = value_of_initializer %int.convert_checked.loc19_21.1 [symbolic = %int.convert_checked.loc19_21.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:       %.loc19_21.2: Core.IntLiteral = converted %N.ref, %.loc19_21.1 [symbolic = %int.convert_checked.loc19_21.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:       %array_type.loc19_22.1: type = array_type %.loc19_21.2, %i32.loc19 [symbolic = %array_type.loc19_22.2 (constants.%array_type.c7c)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %k: ref @G.%array_type.loc19_22.2 (%array_type.c7c) = bind_name k, %k.var
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc12_6.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%N.51e) {
// CHECK:STDOUT:   %N.loc18_6.2 => constants.%N.51e
// CHECK:STDOUT: }
// CHECK:STDOUT:

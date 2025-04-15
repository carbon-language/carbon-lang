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
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.79f: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %const: type = const_type %T [symbolic]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.4f2: type = tuple_type (%ptr.79f, %const) [symbolic]
// CHECK:STDOUT:   %require_complete.155: <witness> = require_complete_type %tuple.type.4f2 [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Op.type.bae: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.a8a: <witness> = lookup_impl_witness %tuple.type.4f2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.7be: %Destroy.type = facet_value %tuple.type.4f2, (%Destroy.lookup_impl_witness.a8a) [symbolic]
// CHECK:STDOUT:   %.f5c: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet.7be [symbolic]
// CHECK:STDOUT:   %impl.elem0.87c: %.f5c = impl_witness_access %Destroy.lookup_impl_witness.a8a, element0 [symbolic]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %T} [symbolic]
// CHECK:STDOUT:   %require_complete.28a: <witness> = require_complete_type %struct_type.a [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.b34: <witness> = lookup_impl_witness %struct_type.a, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.2c9: %Destroy.type = facet_value %struct_type.a, (%Destroy.lookup_impl_witness.b34) [symbolic]
// CHECK:STDOUT:   %.b85: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet.2c9 [symbolic]
// CHECK:STDOUT:   %impl.elem0.dc9: %.b85 = impl_witness_access %Destroy.lookup_impl_witness.b34, element0 [symbolic]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %array_type.ec2: type = array_type %int_5, %T [symbolic]
// CHECK:STDOUT:   %require_complete.fe1: <witness> = require_complete_type %array_type.ec2 [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.b45: <witness> = lookup_impl_witness %array_type.ec2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.0d3: %Destroy.type = facet_value %array_type.ec2, (%Destroy.lookup_impl_witness.b45) [symbolic]
// CHECK:STDOUT:   %.03f: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet.0d3 [symbolic]
// CHECK:STDOUT:   %impl.elem0.1ce: %.03f = impl_witness_access %Destroy.lookup_impl_witness.b45, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.236: <specific function> = specific_impl_function %impl.elem0.1ce, @Op.1(%Destroy.facet.0d3) [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.2a6: <specific function> = specific_impl_function %impl.elem0.dc9, @Op.1(%Destroy.facet.2c9) [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.4b8: <specific function> = specific_impl_function %impl.elem0.87c, @Op.1(%Destroy.facet.7be) [symbolic]
// CHECK:STDOUT:   %require_complete.6e5: <witness> = require_complete_type %ptr.79f [symbolic]
// CHECK:STDOUT:   %require_complete.7cc: <witness> = require_complete_type %const [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %N.51e: %i32 = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt.8e2: %i32 = symbolic_binding_pattern N, 0 [symbolic]
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
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.4fa: <witness> = lookup_impl_witness %array_type.c7c, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.ad4: %Destroy.type = facet_value %array_type.c7c, (%Destroy.lookup_impl_witness.4fa) [symbolic]
// CHECK:STDOUT:   %.a57: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet.ad4 [symbolic]
// CHECK:STDOUT:   %impl.elem0.f15: %.a57 = impl_witness_access %Destroy.lookup_impl_witness.4fa, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.bae: <specific function> = specific_impl_function %impl.elem0.f15, @Op.1(%Destroy.facet.ad4) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
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
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc12_6.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc12_6.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %N.patt.loc18_6.1: %i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc18_6.2 (constants.%N.patt.8e2)]
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
// CHECK:STDOUT:   %T.patt.loc12_6.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc12_6.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ptr.loc13_12.2: type = ptr_type %T.loc12_6.2 [symbolic = %ptr.loc13_12.2 (constants.%ptr.79f)]
// CHECK:STDOUT:   %const.loc13_15.2: type = const_type %T.loc12_6.2 [symbolic = %const.loc13_15.2 (constants.%const)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%ptr.loc13_12.2, %const.loc13_15.2) [symbolic = %tuple.type (constants.%tuple.type.4f2)]
// CHECK:STDOUT:   %require_complete.loc13_22: <witness> = require_complete_type %tuple.type [symbolic = %require_complete.loc13_22 (constants.%require_complete.155)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc13: <witness> = lookup_impl_witness %tuple.type, @Destroy [symbolic = %Destroy.lookup_impl_witness.loc13 (constants.%Destroy.lookup_impl_witness.a8a)]
// CHECK:STDOUT:   %Destroy.facet.loc13: %Destroy.type = facet_value %tuple.type, (%Destroy.lookup_impl_witness.loc13) [symbolic = %Destroy.facet.loc13 (constants.%Destroy.facet.7be)]
// CHECK:STDOUT:   %.loc13_3.5: type = fn_type_with_self_type constants.%Op.type.bae, %Destroy.facet.loc13 [symbolic = %.loc13_3.5 (constants.%.f5c)]
// CHECK:STDOUT:   %impl.elem0.loc13_3.2: @F.%.loc13_3.5 (%.f5c) = impl_witness_access %Destroy.lookup_impl_witness.loc13, element0 [symbolic = %impl.elem0.loc13_3.2 (constants.%impl.elem0.87c)]
// CHECK:STDOUT:   %struct_type.a.loc14_16.2: type = struct_type {.a: @F.%T.loc12_6.2 (%T)} [symbolic = %struct_type.a.loc14_16.2 (constants.%struct_type.a)]
// CHECK:STDOUT:   %require_complete.loc14_16: <witness> = require_complete_type %struct_type.a.loc14_16.2 [symbolic = %require_complete.loc14_16 (constants.%require_complete.28a)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc14: <witness> = lookup_impl_witness %struct_type.a.loc14_16.2, @Destroy [symbolic = %Destroy.lookup_impl_witness.loc14 (constants.%Destroy.lookup_impl_witness.b34)]
// CHECK:STDOUT:   %Destroy.facet.loc14: %Destroy.type = facet_value %struct_type.a.loc14_16.2, (%Destroy.lookup_impl_witness.loc14) [symbolic = %Destroy.facet.loc14 (constants.%Destroy.facet.2c9)]
// CHECK:STDOUT:   %.loc14_3.5: type = fn_type_with_self_type constants.%Op.type.bae, %Destroy.facet.loc14 [symbolic = %.loc14_3.5 (constants.%.b85)]
// CHECK:STDOUT:   %impl.elem0.loc14_3.2: @F.%.loc14_3.5 (%.b85) = impl_witness_access %Destroy.lookup_impl_witness.loc14, element0 [symbolic = %impl.elem0.loc14_3.2 (constants.%impl.elem0.dc9)]
// CHECK:STDOUT:   %array_type.loc15_20.2: type = array_type constants.%int_5, %T.loc12_6.2 [symbolic = %array_type.loc15_20.2 (constants.%array_type.ec2)]
// CHECK:STDOUT:   %require_complete.loc15: <witness> = require_complete_type %array_type.loc15_20.2 [symbolic = %require_complete.loc15 (constants.%require_complete.fe1)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc15: <witness> = lookup_impl_witness %array_type.loc15_20.2, @Destroy [symbolic = %Destroy.lookup_impl_witness.loc15 (constants.%Destroy.lookup_impl_witness.b45)]
// CHECK:STDOUT:   %Destroy.facet.loc15: %Destroy.type = facet_value %array_type.loc15_20.2, (%Destroy.lookup_impl_witness.loc15) [symbolic = %Destroy.facet.loc15 (constants.%Destroy.facet.0d3)]
// CHECK:STDOUT:   %.loc15_3.3: type = fn_type_with_self_type constants.%Op.type.bae, %Destroy.facet.loc15 [symbolic = %.loc15_3.3 (constants.%.03f)]
// CHECK:STDOUT:   %impl.elem0.loc15_3.2: @F.%.loc15_3.3 (%.03f) = impl_witness_access %Destroy.lookup_impl_witness.loc15, element0 [symbolic = %impl.elem0.loc15_3.2 (constants.%impl.elem0.1ce)]
// CHECK:STDOUT:   %specific_impl_fn.loc15_3.2: <specific function> = specific_impl_function %impl.elem0.loc15_3.2, @Op.1(%Destroy.facet.loc15) [symbolic = %specific_impl_fn.loc15_3.2 (constants.%specific_impl_fn.236)]
// CHECK:STDOUT:   %specific_impl_fn.loc14_3.2: <specific function> = specific_impl_function %impl.elem0.loc14_3.2, @Op.1(%Destroy.facet.loc14) [symbolic = %specific_impl_fn.loc14_3.2 (constants.%specific_impl_fn.2a6)]
// CHECK:STDOUT:   %require_complete.loc14_3: <witness> = require_complete_type %T.loc12_6.2 [symbolic = %require_complete.loc14_3 (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %specific_impl_fn.loc13_3.2: <specific function> = specific_impl_function %impl.elem0.loc13_3.2, @Op.1(%Destroy.facet.loc13) [symbolic = %specific_impl_fn.loc13_3.2 (constants.%specific_impl_fn.4b8)]
// CHECK:STDOUT:   %require_complete.loc13_3.1: <witness> = require_complete_type %ptr.loc13_12.2 [symbolic = %require_complete.loc13_3.1 (constants.%require_complete.6e5)]
// CHECK:STDOUT:   %require_complete.loc13_3.2: <witness> = require_complete_type %const.loc13_15.2 [symbolic = %require_complete.loc13_3.2 (constants.%require_complete.7cc)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %u.patt: @F.%tuple.type (%tuple.type.4f2) = binding_pattern u
// CHECK:STDOUT:       %.loc13_3.1: @F.%tuple.type (%tuple.type.4f2) = var_pattern %u.patt
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u.var: ref @F.%tuple.type (%tuple.type.4f2) = var u
// CHECK:STDOUT:     %impl.elem0.loc13_3.1: @F.%.loc13_3.5 (%.f5c) = impl_witness_access constants.%Destroy.lookup_impl_witness.a8a, element0 [symbolic = %impl.elem0.loc13_3.2 (constants.%impl.elem0.87c)]
// CHECK:STDOUT:     %bound_method.loc13: <bound method> = bound_method %u.var, %impl.elem0.loc13_3.1
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
// CHECK:STDOUT:       %v.patt: @F.%struct_type.a.loc14_16.2 (%struct_type.a) = binding_pattern v
// CHECK:STDOUT:       %.loc14_3.1: @F.%struct_type.a.loc14_16.2 (%struct_type.a) = var_pattern %v.patt
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v.var: ref @F.%struct_type.a.loc14_16.2 (%struct_type.a) = var v
// CHECK:STDOUT:     %impl.elem0.loc14_3.1: @F.%.loc14_3.5 (%.b85) = impl_witness_access constants.%Destroy.lookup_impl_witness.b34, element0 [symbolic = %impl.elem0.loc14_3.2 (constants.%impl.elem0.dc9)]
// CHECK:STDOUT:     %bound_method.loc14: <bound method> = bound_method %v.var, %impl.elem0.loc14_3.1
// CHECK:STDOUT:     %.loc14_16: type = splice_block %struct_type.a.loc14_16.1 [symbolic = %struct_type.a.loc14_16.2 (constants.%struct_type.a)] {
// CHECK:STDOUT:       %T.ref.loc14: type = name_ref T, %T.loc12_6.1 [symbolic = %T.loc12_6.2 (constants.%T)]
// CHECK:STDOUT:       %struct_type.a.loc14_16.1: type = struct_type {.a: @F.%T.loc12_6.2 (%T)} [symbolic = %struct_type.a.loc14_16.2 (constants.%struct_type.a)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: ref @F.%struct_type.a.loc14_16.2 (%struct_type.a) = bind_name v, %v.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %w.patt: @F.%array_type.loc15_20.2 (%array_type.ec2) = binding_pattern w
// CHECK:STDOUT:       %.loc15_3.1: @F.%array_type.loc15_20.2 (%array_type.ec2) = var_pattern %w.patt
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %w.var: ref @F.%array_type.loc15_20.2 (%array_type.ec2) = var w
// CHECK:STDOUT:     %impl.elem0.loc15_3.1: @F.%.loc15_3.3 (%.03f) = impl_witness_access constants.%Destroy.lookup_impl_witness.b45, element0 [symbolic = %impl.elem0.loc15_3.2 (constants.%impl.elem0.1ce)]
// CHECK:STDOUT:     %bound_method.loc15: <bound method> = bound_method %w.var, %impl.elem0.loc15_3.1
// CHECK:STDOUT:     %.loc15_20: type = splice_block %array_type.loc15_20.1 [symbolic = %array_type.loc15_20.2 (constants.%array_type.ec2)] {
// CHECK:STDOUT:       %T.ref.loc15: type = name_ref T, %T.loc12_6.1 [symbolic = %T.loc12_6.2 (constants.%T)]
// CHECK:STDOUT:       %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5]
// CHECK:STDOUT:       %array_type.loc15_20.1: type = array_type %int_5, %T.ref.loc15 [symbolic = %array_type.loc15_20.2 (constants.%array_type.ec2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %w: ref @F.%array_type.loc15_20.2 (%array_type.ec2) = bind_name w, %w.var
// CHECK:STDOUT:     %specific_impl_fn.loc15_3.1: <specific function> = specific_impl_function %impl.elem0.loc15_3.1, @Op.1(constants.%Destroy.facet.0d3) [symbolic = %specific_impl_fn.loc15_3.2 (constants.%specific_impl_fn.236)]
// CHECK:STDOUT:     %bound_method.1: <bound method> = bound_method %w.var, %specific_impl_fn.loc15_3.1
// CHECK:STDOUT:     %.loc15_3.2: @F.%array_type.loc15_20.2 (%array_type.ec2) = bind_value %w.var
// CHECK:STDOUT:     %.1: init %empty_tuple.type = call %bound_method.1(%.loc15_3.2)
// CHECK:STDOUT:     %specific_impl_fn.loc14_3.1: <specific function> = specific_impl_function %impl.elem0.loc14_3.1, @Op.1(constants.%Destroy.facet.2c9) [symbolic = %specific_impl_fn.loc14_3.2 (constants.%specific_impl_fn.2a6)]
// CHECK:STDOUT:     %bound_method.2: <bound method> = bound_method %v.var, %specific_impl_fn.loc14_3.1
// CHECK:STDOUT:     %.loc14_3.2: ref @F.%T.loc12_6.2 (%T) = struct_access %v.var, element0
// CHECK:STDOUT:     %.loc14_3.3: @F.%T.loc12_6.2 (%T) = bind_value %.loc14_3.2
// CHECK:STDOUT:     %struct: @F.%struct_type.a.loc14_16.2 (%struct_type.a) = struct_value (%.loc14_3.3)
// CHECK:STDOUT:     %.loc14_3.4: @F.%struct_type.a.loc14_16.2 (%struct_type.a) = converted %v.var, %struct
// CHECK:STDOUT:     %.2: init %empty_tuple.type = call %bound_method.2(%.loc14_3.4)
// CHECK:STDOUT:     %specific_impl_fn.loc13_3.1: <specific function> = specific_impl_function %impl.elem0.loc13_3.1, @Op.1(constants.%Destroy.facet.7be) [symbolic = %specific_impl_fn.loc13_3.2 (constants.%specific_impl_fn.4b8)]
// CHECK:STDOUT:     %bound_method.3: <bound method> = bound_method %u.var, %specific_impl_fn.loc13_3.1
// CHECK:STDOUT:     %tuple.elem0: ref @F.%ptr.loc13_12.2 (%ptr.79f) = tuple_access %u.var, element0
// CHECK:STDOUT:     %.loc13_3.2: @F.%ptr.loc13_12.2 (%ptr.79f) = bind_value %tuple.elem0
// CHECK:STDOUT:     %tuple.elem1: ref @F.%const.loc13_15.2 (%const) = tuple_access %u.var, element1
// CHECK:STDOUT:     %.loc13_3.3: @F.%const.loc13_15.2 (%const) = bind_value %tuple.elem1
// CHECK:STDOUT:     %tuple: @F.%tuple.type (%tuple.type.4f2) = tuple_value (%.loc13_3.2, %.loc13_3.3)
// CHECK:STDOUT:     %.loc13_3.4: @F.%tuple.type (%tuple.type.4f2) = converted %u.var, %tuple
// CHECK:STDOUT:     %.3: init %empty_tuple.type = call %bound_method.3(%.loc13_3.4)
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
// CHECK:STDOUT:   %bound_method.loc19_21.3: <bound method> = bound_method %N.loc18_6.2, constants.%Convert.specific_fn [symbolic = %bound_method.loc19_21.3 (constants.%bound_method)]
// CHECK:STDOUT:   %int.convert_checked.loc19_21.2: init Core.IntLiteral = call %bound_method.loc19_21.3(%N.loc18_6.2) [symbolic = %int.convert_checked.loc19_21.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:   %array_type.loc19_22.2: type = array_type %int.convert_checked.loc19_21.2, constants.%i32 [symbolic = %array_type.loc19_22.2 (constants.%array_type.c7c)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %array_type.loc19_22.2 [symbolic = %require_complete (constants.%require_complete.7cb)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %array_type.loc19_22.2, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.4fa)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %array_type.loc19_22.2, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet (constants.%Destroy.facet.ad4)]
// CHECK:STDOUT:   %.loc19_3.3: type = fn_type_with_self_type constants.%Op.type.bae, %Destroy.facet [symbolic = %.loc19_3.3 (constants.%.a57)]
// CHECK:STDOUT:   %impl.elem0.loc19_3.2: @G.%.loc19_3.3 (%.a57) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc19_3.2 (constants.%impl.elem0.f15)]
// CHECK:STDOUT:   %specific_impl_fn.loc19_3.2: <specific function> = specific_impl_function %impl.elem0.loc19_3.2, @Op.1(%Destroy.facet) [symbolic = %specific_impl_fn.loc19_3.2 (constants.%specific_impl_fn.bae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %k.patt: @G.%array_type.loc19_22.2 (%array_type.c7c) = binding_pattern k
// CHECK:STDOUT:       %.loc19_3.1: @G.%array_type.loc19_22.2 (%array_type.c7c) = var_pattern %k.patt
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %k.var: ref @G.%array_type.loc19_22.2 (%array_type.c7c) = var k
// CHECK:STDOUT:     %impl.elem0.loc19_3.1: @G.%.loc19_3.3 (%.a57) = impl_witness_access constants.%Destroy.lookup_impl_witness.4fa, element0 [symbolic = %impl.elem0.loc19_3.2 (constants.%impl.elem0.f15)]
// CHECK:STDOUT:     %bound_method.loc19_3: <bound method> = bound_method %k.var, %impl.elem0.loc19_3.1
// CHECK:STDOUT:     %.loc19_22: type = splice_block %array_type.loc19_22.1 [symbolic = %array_type.loc19_22.2 (constants.%array_type.c7c)] {
// CHECK:STDOUT:       %int_32.loc19: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc19: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %N.ref: %i32 = name_ref N, %N.loc18_6.1 [symbolic = %N.loc18_6.2 (constants.%N.51e)]
// CHECK:STDOUT:       %impl.elem0.loc19_21: %.0ea = impl_witness_access constants.%ImplicitAs.impl_witness.a11, element0 [concrete = constants.%Convert.960]
// CHECK:STDOUT:       %bound_method.loc19_21.1: <bound method> = bound_method %N.ref, %impl.elem0.loc19_21 [symbolic = %Convert.bound (constants.%Convert.bound)]
// CHECK:STDOUT:       %specific_fn: <specific function> = specific_function %impl.elem0.loc19_21, @Convert.3(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:       %bound_method.loc19_21.2: <bound method> = bound_method %N.ref, %specific_fn [symbolic = %bound_method.loc19_21.3 (constants.%bound_method)]
// CHECK:STDOUT:       %int.convert_checked.loc19_21.1: init Core.IntLiteral = call %bound_method.loc19_21.2(%N.ref) [symbolic = %int.convert_checked.loc19_21.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:       %.loc19_21.1: Core.IntLiteral = value_of_initializer %int.convert_checked.loc19_21.1 [symbolic = %int.convert_checked.loc19_21.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:       %.loc19_21.2: Core.IntLiteral = converted %N.ref, %.loc19_21.1 [symbolic = %int.convert_checked.loc19_21.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:       %array_type.loc19_22.1: type = array_type %.loc19_21.2, %i32.loc19 [symbolic = %array_type.loc19_22.2 (constants.%array_type.c7c)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %k: ref @G.%array_type.loc19_22.2 (%array_type.c7c) = bind_name k, %k.var
// CHECK:STDOUT:     %specific_impl_fn.loc19_3.1: <specific function> = specific_impl_function %impl.elem0.loc19_3.1, @Op.1(constants.%Destroy.facet.ad4) [symbolic = %specific_impl_fn.loc19_3.2 (constants.%specific_impl_fn.bae)]
// CHECK:STDOUT:     %bound_method.1: <bound method> = bound_method %k.var, %specific_impl_fn.loc19_3.1
// CHECK:STDOUT:     %.loc19_3.2: @G.%array_type.loc19_22.2 (%array_type.c7c) = bind_value %k.var
// CHECK:STDOUT:     %.1: init %empty_tuple.type = call %bound_method.1(%.loc19_3.2)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc12_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc12_6.2 => constants.%T.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%N.51e) {
// CHECK:STDOUT:   %N.loc18_6.2 => constants.%N.51e
// CHECK:STDOUT:   %N.patt.loc18_6.2 => constants.%N.patt.8e2
// CHECK:STDOUT: }
// CHECK:STDOUT:

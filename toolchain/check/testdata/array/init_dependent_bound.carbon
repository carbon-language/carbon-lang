// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/array/init_dependent_bound.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/array/init_dependent_bound.carbon

// --- generic_empty.carbon

library "[[@TEST_NAME]]";

fn G(T:! type) {
  // We can initialize this without knowing T.
  //@dump-sem-ir-begin
  var arr: array(T, 0) = ();
  //@dump-sem-ir-end
}

class C {}

fn H() {
  G(C);
}

// --- fail_init_dependent_bound.carbon

library "[[@TEST_NAME]]";

fn F(N:! i32) {
  // CHECK:STDERR: fail_init_dependent_bound.carbon:[[@LINE+4]]:28: error: cannot initialize array with dependent bound from a list of initializers [ArrayInitDependentBound]
  // CHECK:STDERR:   var arr: array(i32, N) = (1, 2, 3);
  // CHECK:STDERR:                            ^~~~~~~~~
  // CHECK:STDERR:
  var arr: array(i32, N) = (1, 2, 3);
}

// --- fail_todo_init_template_dependent_bound.carbon

library "[[@TEST_NAME]]";

// TODO: This should be valid.
fn G(template N:! i32) {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_init_template_dependent_bound.carbon:[[@LINE+4]]:12: error: cannot evaluate type expression [TypeExprEvaluationFailure]
  // CHECK:STDERR:   var arr: array(i32, N) = (1, 2, 3);
  // CHECK:STDERR:            ^~~~~~~~~~~~~
  // CHECK:STDERR:
  var arr: array(i32, N) = (1, 2, 3);
  //@dump-sem-ir-end
}

fn H() { G(3); }

// CHECK:STDOUT: --- generic_empty.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %array_type.281: type = array_type %int_0, %T [symbolic]
// CHECK:STDOUT:   %ptr.e06: type = ptr_type %array_type.281 [symbolic]
// CHECK:STDOUT:   %require_complete.b7f: <witness> = require_complete_type %array_type.281 [symbolic]
// CHECK:STDOUT:   %pattern_type.d48: type = pattern_type %array_type.281 [symbolic]
// CHECK:STDOUT:   %array.2ed: %array_type.281 = tuple_value () [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Op.type.bae: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %Op.type.bc9: type = fn_type @Op.2, @impl(%T) [symbolic]
// CHECK:STDOUT:   %Op.46f: %Op.type.bc9 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.662: <witness> = require_complete_type %ptr.e06 [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %array_type.281, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.cdc: %Destroy.type = facet_value %array_type.281, (%Destroy.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %.77e: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet.cdc [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.77e = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @Op.1(%Destroy.facet.cdc) [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %array_type.6f1: type = array_type %int_0, %C [concrete]
// CHECK:STDOUT:   %ptr.cf4: type = ptr_type %array_type.6f1 [concrete]
// CHECK:STDOUT:   %complete_type.ed6: <witness> = complete_type_witness %array_type.6f1 [concrete]
// CHECK:STDOUT:   %pattern_type.9c8: type = pattern_type %array_type.6f1 [concrete]
// CHECK:STDOUT:   %array.2e5: %array_type.6f1 = tuple_value () [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.8b8: <witness> = impl_witness imports.%Destroy.impl_witness_table, @impl(%array_type.6f1) [concrete]
// CHECK:STDOUT:   %Op.type.a3e: type = fn_type @Op.2, @impl(%array_type.6f1) [concrete]
// CHECK:STDOUT:   %Op.dc5: %Op.type.a3e = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type.3e1: <witness> = complete_type_witness %ptr.cf4 [concrete]
// CHECK:STDOUT:   %Destroy.facet.da7: %Destroy.type = facet_value %array_type.6f1, (%Destroy.impl_witness.8b8) [concrete]
// CHECK:STDOUT:   %.a21: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet.da7 [concrete]
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function %Op.dc5, @Op.2(%array_type.6f1) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.0b9: @impl.%Op.type (%Op.type.bc9) = import_ref Core//prelude/parts/destroy, loc8_29, loaded [symbolic = @impl.%Op (constants.%Op.46f)]
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (%Core.import_ref.0b9), @impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%T.loc4_6.1: type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %array_type.loc7_22.2: type = array_type constants.%int_0, %T.loc4_6.2 [symbolic = %array_type.loc7_22.2 (constants.%array_type.281)]
// CHECK:STDOUT:   %require_complete.loc7_22: <witness> = require_complete_type %array_type.loc7_22.2 [symbolic = %require_complete.loc7_22 (constants.%require_complete.b7f)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %array_type.loc7_22.2 [symbolic = %pattern_type (constants.%pattern_type.d48)]
// CHECK:STDOUT:   %array: @G.%array_type.loc7_22.2 (%array_type.281) = tuple_value () [symbolic = %array (constants.%array.2ed)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %array_type.loc7_22.2, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %array_type.loc7_22.2, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet (constants.%Destroy.facet.cdc)]
// CHECK:STDOUT:   %.loc7_3.3: type = fn_type_with_self_type constants.%Op.type.bae, %Destroy.facet [symbolic = %.loc7_3.3 (constants.%.77e)]
// CHECK:STDOUT:   %impl.elem0.loc7_3.2: @G.%.loc7_3.3 (%.77e) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc7_3.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc7_3.2: <specific function> = specific_impl_function %impl.elem0.loc7_3.2, @Op.1(%Destroy.facet) [symbolic = %specific_impl_fn.loc7_3.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:   %ptr: type = ptr_type %array_type.loc7_22.2 [symbolic = %ptr (constants.%ptr.e06)]
// CHECK:STDOUT:   %require_complete.loc7_3: <witness> = require_complete_type %ptr [symbolic = %require_complete.loc7_3 (constants.%require_complete.662)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %arr.patt: @G.%pattern_type (%pattern_type.d48) = binding_pattern arr [concrete]
// CHECK:STDOUT:       %arr.var_patt: @G.%pattern_type (%pattern_type.d48) = var_pattern %arr.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %arr.var: ref @G.%array_type.loc7_22.2 (%array_type.281) = var %arr.var_patt
// CHECK:STDOUT:     %.loc7_27.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc7_27.2: init @G.%array_type.loc7_22.2 (%array_type.281) = array_init () to %arr.var [symbolic = %array (constants.%array.2ed)]
// CHECK:STDOUT:     %.loc7_3.1: init @G.%array_type.loc7_22.2 (%array_type.281) = converted %.loc7_27.1, %.loc7_27.2 [symbolic = %array (constants.%array.2ed)]
// CHECK:STDOUT:     assign %arr.var, %.loc7_3.1
// CHECK:STDOUT:     %.loc7_22: type = splice_block %array_type.loc7_22.1 [symbolic = %array_type.loc7_22.2 (constants.%array_type.281)] {
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc4_6.1 [symbolic = %T.loc4_6.2 (constants.%T)]
// CHECK:STDOUT:       %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:       %array_type.loc7_22.1: type = array_type %int_0, %T.ref [symbolic = %array_type.loc7_22.2 (constants.%array_type.281)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %arr: ref @G.%array_type.loc7_22.2 (%array_type.281) = bind_name arr, %arr.var
// CHECK:STDOUT:     %impl.elem0.loc7_3.1: @G.%.loc7_3.3 (%.77e) = impl_witness_access constants.%Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc7_3.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc7_3.1: <bound method> = bound_method %arr.var, %impl.elem0.loc7_3.1
// CHECK:STDOUT:     %specific_impl_fn.loc7_3.1: <specific function> = specific_impl_function %impl.elem0.loc7_3.1, @Op.1(constants.%Destroy.facet.cdc) [symbolic = %specific_impl_fn.loc7_3.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %bound_method.loc7_3.2: <bound method> = bound_method %arr.var, %specific_impl_fn.loc7_3.1
// CHECK:STDOUT:     %addr: @G.%ptr (%ptr.e06) = addr_of %arr.var
// CHECK:STDOUT:     %.loc7_3.2: init %empty_tuple.type = call %bound_method.loc7_3.2(%addr)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%T) {
// CHECK:STDOUT:   %T.loc4_6.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%C) {
// CHECK:STDOUT:   %T.loc4_6.2 => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %array_type.loc7_22.2 => constants.%array_type.6f1
// CHECK:STDOUT:   %require_complete.loc7_22 => constants.%complete_type.ed6
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9c8
// CHECK:STDOUT:   %array => constants.%array.2e5
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%Destroy.impl_witness.8b8
// CHECK:STDOUT:   %Destroy.facet => constants.%Destroy.facet.da7
// CHECK:STDOUT:   %.loc7_3.3 => constants.%.a21
// CHECK:STDOUT:   %impl.elem0.loc7_3.2 => constants.%Op.dc5
// CHECK:STDOUT:   %specific_impl_fn.loc7_3.2 => constants.%Op.specific_fn
// CHECK:STDOUT:   %ptr => constants.%ptr.cf4
// CHECK:STDOUT:   %require_complete.loc7_3 => constants.%complete_type.3e1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_init_template_dependent_bound.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %N.51e: %i32 = bind_symbolic_name N, 0, template [template]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.2fd: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %From: Core.IntLiteral = bind_symbolic_name From, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.708: type = fn_type @Convert.3, @impl.971(%From) [symbolic]
// CHECK:STDOUT:   %Convert.c68: %Convert.type.708 = struct_value () [symbolic]
// CHECK:STDOUT:   %Convert.type.71e: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %int_3.822: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a11: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1d9, @impl.971(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.4ad: type = fn_type @Convert.3, @impl.971(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.960: %Convert.type.4ad = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.f49: %ImplicitAs.type.2fd = facet_value %i32, (%ImplicitAs.impl_witness.a11) [concrete]
// CHECK:STDOUT:   %.0ea: type = fn_type_with_self_type %Convert.type.71e, %ImplicitAs.facet.f49 [concrete]
// CHECK:STDOUT:   %Convert.bound.2d6: <bound method> = bound_method %int_3.822, %Convert.960 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn.8a8: <specific function> = specific_function %Convert.960, @Convert.3(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.c6f: <bound method> = bound_method %int_3.822, %Convert.specific_fn.8a8 [concrete]
// CHECK:STDOUT:   %inst.splice_block: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.df7: Core.IntLiteral = splice_block %.dd0 [concrete = %int_3.1ba] {
// CHECK:STDOUT:       %impl.elem0.c2d: %.0ea = impl_witness_access %ImplicitAs.impl_witness.a11, element0 [concrete = %Convert.960]
// CHECK:STDOUT:       %bound_method.8d1: <bound method> = bound_method %int_3.822, %impl.elem0.c2d [concrete = %Convert.bound.2d6]
// CHECK:STDOUT:       %specific_fn: <specific function> = specific_function %impl.elem0.c2d, @Convert.3(%int_32) [concrete = %Convert.specific_fn.8a8]
// CHECK:STDOUT:       %bound_method.9ef: <bound method> = bound_method %int_3.822, %specific_fn [concrete = %bound_method.c6f]
// CHECK:STDOUT:       %int.convert_checked: init Core.IntLiteral = call %bound_method.9ef(%int_3.822) [concrete = %int_3.1ba]
// CHECK:STDOUT:       %.5d1: Core.IntLiteral = value_of_initializer %int.convert_checked [concrete = %int_3.1ba]
// CHECK:STDOUT:       %.dd0: Core.IntLiteral = converted %int_3.822, %.5d1 [concrete = %int_3.1ba]
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.85c: @impl.971.%Convert.type (%Convert.type.708) = import_ref Core//prelude/parts/int, loc20_44, loaded [symbolic = @impl.971.%Convert (constants.%Convert.c68)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1d9 = impl_witness_table (%Core.import_ref.85c), @impl.971 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%N.loc5_15.1: %i32) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc11_23.2: <instruction> = convert_to_value_action %N.ref, Core.IntLiteral [template]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %arr.patt: <error> = binding_pattern arr [concrete]
// CHECK:STDOUT:       %arr.var_patt: <error> = var_pattern %arr.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %arr.var: ref <error> = var %arr.var_patt [concrete = <error>]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %.loc11_36: %tuple.type = tuple_literal (%int_1, %int_2, %int_3)
// CHECK:STDOUT:     assign %arr.var, <error>
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %arr: <error> = bind_name arr, <error> [concrete = <error>]
// CHECK:STDOUT:     %impl.elem0: <error> = impl_witness_access <error>, element0 [concrete = <error>]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%N.51e) {
// CHECK:STDOUT:   %N.loc5_15.2 => constants.%N.51e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%int_3.822) {
// CHECK:STDOUT:   %N.loc5_15.2 => constants.%int_3.822
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc11_23.2 => constants.%inst.splice_block
// CHECK:STDOUT: }
// CHECK:STDOUT:

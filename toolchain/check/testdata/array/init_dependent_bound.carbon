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
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %array_type.512: type = array_type %int_0, %T [symbolic]
// CHECK:STDOUT:   %ptr.a86: type = ptr_type %array_type.512 [symbolic]
// CHECK:STDOUT:   %require_complete.cec: <witness> = require_complete_type %array_type.512 [symbolic]
// CHECK:STDOUT:   %pattern_type.b5c: type = pattern_type %array_type.512 [symbolic]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %array.8ba: %array_type.512 = tuple_value () [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %DestroyT: %type_where = symbolic_binding DestroyT, 0 [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.15c: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%DestroyT) [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.fd5: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.15c = struct_value () [symbolic]
// CHECK:STDOUT:   %facet_value.30f: %type_where = facet_value %array_type.512, () [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.aa0: <witness> = impl_witness imports.%Destroy.impl_witness_table, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.30f) [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.cc4: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.30f) [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.44f: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.cc4 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.2e8: <witness> = require_complete_type %ptr.a86 [symbolic]
// CHECK:STDOUT:   %Destroy.facet.49e: %Destroy.type = facet_value %array_type.512, (%Destroy.impl_witness.aa0) [symbolic]
// CHECK:STDOUT:   %.fb1: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet.49e [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.91d: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op.44f, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value.30f) [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %array_type.6f1: type = array_type %int_0, %C [concrete]
// CHECK:STDOUT:   %ptr.cf4: type = ptr_type %array_type.6f1 [concrete]
// CHECK:STDOUT:   %complete_type.ed6: <witness> = complete_type_witness %array_type.6f1 [concrete]
// CHECK:STDOUT:   %pattern_type.9c8: type = pattern_type %array_type.6f1 [concrete]
// CHECK:STDOUT:   %array.2e5: %array_type.6f1 = tuple_value () [concrete]
// CHECK:STDOUT:   %facet_value.cba: %type_where = facet_value %array_type.6f1, () [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.dff: <witness> = impl_witness imports.%Destroy.impl_witness_table, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.cba) [concrete]
// CHECK:STDOUT:   %Destroy.facet.105: %Destroy.type = facet_value %array_type.6f1, (%Destroy.impl_witness.dff) [concrete]
// CHECK:STDOUT:   %.0c7: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet.105 [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.cc8: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.cba) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.dcd: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.cc8 = struct_value () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.bf3: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op.dcd, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value.cba) [concrete]
// CHECK:STDOUT:   %complete_type.3e1: <witness> = complete_type_witness %ptr.cf4 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.5c4: @DestroyT.binding.as_type.as.Destroy.impl.%DestroyT.binding.as_type.as.Destroy.impl.Op.type (%DestroyT.binding.as_type.as.Destroy.impl.Op.type.15c) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @DestroyT.binding.as_type.as.Destroy.impl.%DestroyT.binding.as_type.as.Destroy.impl.Op (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.fd5)]
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (%Core.import_ref.5c4), @DestroyT.binding.as_type.as.Destroy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%T.loc4_6.2: type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %array_type.loc7_22.2: type = array_type constants.%int_0, %T.loc4_6.1 [symbolic = %array_type.loc7_22.2 (constants.%array_type.512)]
// CHECK:STDOUT:   %require_complete.loc7_22: <witness> = require_complete_type %array_type.loc7_22.2 [symbolic = %require_complete.loc7_22 (constants.%require_complete.cec)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %array_type.loc7_22.2 [symbolic = %pattern_type (constants.%pattern_type.b5c)]
// CHECK:STDOUT:   %array: @G.%array_type.loc7_22.2 (%array_type.512) = tuple_value () [symbolic = %array (constants.%array.8ba)]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %array_type.loc7_22.2, () [symbolic = %facet_value (constants.%facet_value.30f)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness imports.%Destroy.impl_witness_table, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness.aa0)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %array_type.loc7_22.2, (%Destroy.impl_witness) [symbolic = %Destroy.facet (constants.%Destroy.facet.49e)]
// CHECK:STDOUT:   %.loc7_3.2: type = fn_type_with_self_type constants.%Destroy.Op.type, %Destroy.facet [symbolic = %.loc7_3.2 (constants.%.fb1)]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op.type (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.type.cc4)]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op: @G.%DestroyT.binding.as_type.as.Destroy.impl.Op.type (%DestroyT.binding.as_type.as.Destroy.impl.Op.type.cc4) = struct_value () [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.44f)]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value) [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.91d)]
// CHECK:STDOUT:   %ptr: type = ptr_type %array_type.loc7_22.2 [symbolic = %ptr (constants.%ptr.a86)]
// CHECK:STDOUT:   %require_complete.loc7_3: <witness> = require_complete_type %ptr [symbolic = %require_complete.loc7_3 (constants.%require_complete.2e8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %arr.patt: @G.%pattern_type (%pattern_type.b5c) = ref_binding_pattern arr [concrete]
// CHECK:STDOUT:       %arr.var_patt: @G.%pattern_type (%pattern_type.b5c) = var_pattern %arr.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %arr.var: ref @G.%array_type.loc7_22.2 (%array_type.512) = var %arr.var_patt
// CHECK:STDOUT:     %.loc7_27.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc7_27.2: init @G.%array_type.loc7_22.2 (%array_type.512) = array_init () to %arr.var [symbolic = %array (constants.%array.8ba)]
// CHECK:STDOUT:     %.loc7_3.1: init @G.%array_type.loc7_22.2 (%array_type.512) = converted %.loc7_27.1, %.loc7_27.2 [symbolic = %array (constants.%array.8ba)]
// CHECK:STDOUT:     assign %arr.var, %.loc7_3.1
// CHECK:STDOUT:     %.loc7_22: type = splice_block %array_type.loc7_22.1 [symbolic = %array_type.loc7_22.2 (constants.%array_type.512)] {
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc4_6.2 [symbolic = %T.loc4_6.1 (constants.%T)]
// CHECK:STDOUT:       %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:       %array_type.loc7_22.1: type = array_type %int_0, %T.ref [symbolic = %array_type.loc7_22.2 (constants.%array_type.512)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %arr: ref @G.%array_type.loc7_22.2 (%array_type.512) = ref_binding arr, %arr.var
// CHECK:STDOUT:     %impl.elem0: @G.%.loc7_3.2 (%.fb1) = impl_witness_access constants.%Destroy.impl_witness.aa0, element0 [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.44f)]
// CHECK:STDOUT:     %bound_method.loc7_3.1: <bound method> = bound_method %arr.var, %impl.elem0
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.30f) [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.91d)]
// CHECK:STDOUT:     %bound_method.loc7_3.2: <bound method> = bound_method %arr.var, %specific_fn
// CHECK:STDOUT:     %addr: @G.%ptr (%ptr.a86) = addr_of %arr.var
// CHECK:STDOUT:     %DestroyT.binding.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc7_3.2(%addr)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%T) {
// CHECK:STDOUT:   %T.loc4_6.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%C) {
// CHECK:STDOUT:   %T.loc4_6.1 => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %array_type.loc7_22.2 => constants.%array_type.6f1
// CHECK:STDOUT:   %require_complete.loc7_22 => constants.%complete_type.ed6
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9c8
// CHECK:STDOUT:   %array => constants.%array.2e5
// CHECK:STDOUT:   %facet_value => constants.%facet_value.cba
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.dff
// CHECK:STDOUT:   %Destroy.facet => constants.%Destroy.facet.105
// CHECK:STDOUT:   %.loc7_3.2 => constants.%.0c7
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type => constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.type.cc8
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op => constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.dcd
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn => constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.bf3
// CHECK:STDOUT:   %ptr => constants.%ptr.cf4
// CHECK:STDOUT:   %require_complete.loc7_3 => constants.%complete_type.3e1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_init_template_dependent_bound.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %N.fe8: %i32 = symbolic_binding N, 0, template [template]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%int_1, %int_2, %int_3.1ba) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.7a9: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %From: Core.IntLiteral = symbolic_binding From, 0 [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.2c2: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%From) [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.cdf: %Int.as.ImplicitAs.impl.Convert.type.2c2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.71e: type = fn_type @ImplicitAs.Convert, @ImplicitAs(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %int_3.822: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.972: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.783, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.592: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.047: %Int.as.ImplicitAs.impl.Convert.type.592 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.64a: %ImplicitAs.type.7a9 = facet_value %i32, (%ImplicitAs.impl_witness.972) [concrete]
// CHECK:STDOUT:   %.63b: type = fn_type_with_self_type %ImplicitAs.Convert.type.71e, %ImplicitAs.facet.64a [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_3.822, %Int.as.ImplicitAs.impl.Convert.047 [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Int.as.ImplicitAs.impl.Convert.047, @Int.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.ee5: <bound method> = bound_method %int_3.822, %Int.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %inst.splice_block: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.17f: Core.IntLiteral = splice_block %.2c5 [concrete = %int_3.1ba] {
// CHECK:STDOUT:       %impl.elem0: %.63b = impl_witness_access %ImplicitAs.impl_witness.972, element0 [concrete = %Int.as.ImplicitAs.impl.Convert.047]
// CHECK:STDOUT:       %bound_method.de9: <bound method> = bound_method %int_3.822, %impl.elem0 [concrete = %Int.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:       %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.ImplicitAs.impl.Convert(%int_32) [concrete = %Int.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:       %bound_method.14b: <bound method> = bound_method %int_3.822, %specific_fn [concrete = %bound_method.ee5]
// CHECK:STDOUT:       %Int.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.14b(%int_3.822) [concrete = %int_3.1ba]
// CHECK:STDOUT:       %.d2b: Core.IntLiteral = value_of_initializer %Int.as.ImplicitAs.impl.Convert.call [concrete = %int_3.1ba]
// CHECK:STDOUT:       %.2c5: Core.IntLiteral = converted %int_3.822, %.d2b [concrete = %int_3.1ba]
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.a86: @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert.type (%Int.as.ImplicitAs.impl.Convert.type.2c2) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert (constants.%Int.as.ImplicitAs.impl.Convert.cdf)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.783 = impl_witness_table (%Core.import_ref.a86), @Int.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%N.loc5_15.2: %i32) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc11_23.2: <instruction> = convert_to_value_action %N.ref, Core.IntLiteral [template]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %arr.patt: <error> = ref_binding_pattern arr [concrete]
// CHECK:STDOUT:       %arr.var_patt: <error> = var_pattern %arr.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %arr.var: ref <error> = var %arr.var_patt [concrete = <error>]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %.loc11_36: %tuple.type = tuple_literal (%int_1, %int_2, %int_3) [concrete = constants.%tuple]
// CHECK:STDOUT:     assign %arr.var, <error>
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %arr: ref <error> = ref_binding arr, <error> [concrete = <error>]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%N.fe8) {
// CHECK:STDOUT:   %N.loc5_15.1 => constants.%N.fe8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%int_3.822) {
// CHECK:STDOUT:   %N.loc5_15.1 => constants.%int_3.822
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc11_23.2 => constants.%inst.splice_block
// CHECK:STDOUT: }
// CHECK:STDOUT:

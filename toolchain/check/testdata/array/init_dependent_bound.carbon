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
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %DestroyT: %type_where = bind_symbolic_name DestroyT, 0 [symbolic]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.type.190: type = fn_type @DestroyT.as_type.as.Destroy.impl.Op, @DestroyT.as_type.as.Destroy.impl(%DestroyT) [symbolic]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.8a0: %DestroyT.as_type.as.Destroy.impl.Op.type.190 = struct_value () [symbolic]
// CHECK:STDOUT:   %facet_value.bf1: %type_where = facet_value %array_type.281, () [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.f4a: <witness> = impl_witness imports.%Destroy.impl_witness_table, @DestroyT.as_type.as.Destroy.impl(%facet_value.bf1) [symbolic]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.type.565: type = fn_type @DestroyT.as_type.as.Destroy.impl.Op, @DestroyT.as_type.as.Destroy.impl(%facet_value.bf1) [symbolic]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.dd2: %DestroyT.as_type.as.Destroy.impl.Op.type.565 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.662: <witness> = require_complete_type %ptr.e06 [symbolic]
// CHECK:STDOUT:   %Destroy.facet.f25: %Destroy.type = facet_value %array_type.281, (%Destroy.impl_witness.f4a) [symbolic]
// CHECK:STDOUT:   %.3d9: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet.f25 [symbolic]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.specific_fn.c7c: <specific function> = specific_function %DestroyT.as_type.as.Destroy.impl.Op.dd2, @DestroyT.as_type.as.Destroy.impl.Op(%facet_value.bf1) [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %array_type.6f1: type = array_type %int_0, %C [concrete]
// CHECK:STDOUT:   %ptr.cf4: type = ptr_type %array_type.6f1 [concrete]
// CHECK:STDOUT:   %complete_type.ed6: <witness> = complete_type_witness %array_type.6f1 [concrete]
// CHECK:STDOUT:   %pattern_type.9c8: type = pattern_type %array_type.6f1 [concrete]
// CHECK:STDOUT:   %array.2e5: %array_type.6f1 = tuple_value () [concrete]
// CHECK:STDOUT:   %facet_value.cba: %type_where = facet_value %array_type.6f1, () [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.20d: <witness> = impl_witness imports.%Destroy.impl_witness_table, @DestroyT.as_type.as.Destroy.impl(%facet_value.cba) [concrete]
// CHECK:STDOUT:   %Destroy.facet.ad5: %Destroy.type = facet_value %array_type.6f1, (%Destroy.impl_witness.20d) [concrete]
// CHECK:STDOUT:   %.600: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet.ad5 [concrete]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.type.710: type = fn_type @DestroyT.as_type.as.Destroy.impl.Op, @DestroyT.as_type.as.Destroy.impl(%facet_value.cba) [concrete]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.0db: %DestroyT.as_type.as.Destroy.impl.Op.type.710 = struct_value () [concrete]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.specific_fn.f19: <specific function> = specific_function %DestroyT.as_type.as.Destroy.impl.Op.0db, @DestroyT.as_type.as.Destroy.impl.Op(%facet_value.cba) [concrete]
// CHECK:STDOUT:   %complete_type.3e1: <witness> = complete_type_witness %ptr.cf4 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.d51: @DestroyT.as_type.as.Destroy.impl.%DestroyT.as_type.as.Destroy.impl.Op.type (%DestroyT.as_type.as.Destroy.impl.Op.type.190) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @DestroyT.as_type.as.Destroy.impl.%DestroyT.as_type.as.Destroy.impl.Op (constants.%DestroyT.as_type.as.Destroy.impl.Op.8a0)]
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (%Core.import_ref.d51), @DestroyT.as_type.as.Destroy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%T.loc4_6.2: type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %array_type.loc7_22.2: type = array_type constants.%int_0, %T.loc4_6.1 [symbolic = %array_type.loc7_22.2 (constants.%array_type.281)]
// CHECK:STDOUT:   %require_complete.loc7_22: <witness> = require_complete_type %array_type.loc7_22.2 [symbolic = %require_complete.loc7_22 (constants.%require_complete.b7f)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %array_type.loc7_22.2 [symbolic = %pattern_type (constants.%pattern_type.d48)]
// CHECK:STDOUT:   %array: @G.%array_type.loc7_22.2 (%array_type.281) = tuple_value () [symbolic = %array (constants.%array.2ed)]
// CHECK:STDOUT:   %facet_value.loc7_3.2: %type_where = facet_value %array_type.loc7_22.2, () [symbolic = %facet_value.loc7_3.2 (constants.%facet_value.bf1)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness imports.%Destroy.impl_witness_table, @DestroyT.as_type.as.Destroy.impl(%facet_value.loc7_3.2) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness.f4a)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %array_type.loc7_22.2, (%Destroy.impl_witness) [symbolic = %Destroy.facet (constants.%Destroy.facet.f25)]
// CHECK:STDOUT:   %.loc7_3.3: type = fn_type_with_self_type constants.%Destroy.Op.type, %Destroy.facet [symbolic = %.loc7_3.3 (constants.%.3d9)]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.type: type = fn_type @DestroyT.as_type.as.Destroy.impl.Op, @DestroyT.as_type.as.Destroy.impl(%facet_value.loc7_3.2) [symbolic = %DestroyT.as_type.as.Destroy.impl.Op.type (constants.%DestroyT.as_type.as.Destroy.impl.Op.type.565)]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op: @G.%DestroyT.as_type.as.Destroy.impl.Op.type (%DestroyT.as_type.as.Destroy.impl.Op.type.565) = struct_value () [symbolic = %DestroyT.as_type.as.Destroy.impl.Op (constants.%DestroyT.as_type.as.Destroy.impl.Op.dd2)]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %DestroyT.as_type.as.Destroy.impl.Op, @DestroyT.as_type.as.Destroy.impl.Op(%facet_value.loc7_3.2) [symbolic = %DestroyT.as_type.as.Destroy.impl.Op.specific_fn (constants.%DestroyT.as_type.as.Destroy.impl.Op.specific_fn.c7c)]
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
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc4_6.2 [symbolic = %T.loc4_6.1 (constants.%T)]
// CHECK:STDOUT:       %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:       %array_type.loc7_22.1: type = array_type %int_0, %T.ref [symbolic = %array_type.loc7_22.2 (constants.%array_type.281)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %arr: ref @G.%array_type.loc7_22.2 (%array_type.281) = bind_name arr, %arr.var
// CHECK:STDOUT:     %facet_value.loc7_3.1: %type_where = facet_value constants.%array_type.281, () [symbolic = %facet_value.loc7_3.2 (constants.%facet_value.bf1)]
// CHECK:STDOUT:     %.loc7_3.2: %type_where = converted constants.%array_type.281, %facet_value.loc7_3.1 [symbolic = %facet_value.loc7_3.2 (constants.%facet_value.bf1)]
// CHECK:STDOUT:     %impl.elem0: @G.%.loc7_3.3 (%.3d9) = impl_witness_access constants.%Destroy.impl_witness.f4a, element0 [symbolic = %DestroyT.as_type.as.Destroy.impl.Op (constants.%DestroyT.as_type.as.Destroy.impl.Op.dd2)]
// CHECK:STDOUT:     %bound_method.loc7_3.1: <bound method> = bound_method %arr.var, %impl.elem0
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @DestroyT.as_type.as.Destroy.impl.Op(constants.%facet_value.bf1) [symbolic = %DestroyT.as_type.as.Destroy.impl.Op.specific_fn (constants.%DestroyT.as_type.as.Destroy.impl.Op.specific_fn.c7c)]
// CHECK:STDOUT:     %bound_method.loc7_3.2: <bound method> = bound_method %arr.var, %specific_fn
// CHECK:STDOUT:     %addr: @G.%ptr (%ptr.e06) = addr_of %arr.var
// CHECK:STDOUT:     %DestroyT.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc7_3.2(%addr)
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
// CHECK:STDOUT:   %facet_value.loc7_3.2 => constants.%facet_value.cba
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.20d
// CHECK:STDOUT:   %Destroy.facet => constants.%Destroy.facet.ad5
// CHECK:STDOUT:   %.loc7_3.3 => constants.%.600
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.type => constants.%DestroyT.as_type.as.Destroy.impl.Op.type.710
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op => constants.%DestroyT.as_type.as.Destroy.impl.Op.0db
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.specific_fn => constants.%DestroyT.as_type.as.Destroy.impl.Op.specific_fn.f19
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
// CHECK:STDOUT:   %ImplicitAs.type.7a9: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %From: Core.IntLiteral = bind_symbolic_name From, 0 [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.543: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%From) [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.c08: %Int.as.ImplicitAs.impl.Convert.type.543 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.71e: type = fn_type @ImplicitAs.Convert, @ImplicitAs(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %int_3.822: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.6bf: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.e99, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.516: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.b09: %Int.as.ImplicitAs.impl.Convert.type.516 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.1a1: %ImplicitAs.type.7a9 = facet_value %i32, (%ImplicitAs.impl_witness.6bf) [concrete]
// CHECK:STDOUT:   %.81e: type = fn_type_with_self_type %ImplicitAs.Convert.type.71e, %ImplicitAs.facet.1a1 [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_3.822, %Int.as.ImplicitAs.impl.Convert.b09 [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Int.as.ImplicitAs.impl.Convert.b09, @Int.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.e83: <bound method> = bound_method %int_3.822, %Int.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %inst.splice_block: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.7f6: Core.IntLiteral = splice_block %.efe [concrete = %int_3.1ba] {
// CHECK:STDOUT:       %impl.elem0: %.81e = impl_witness_access %ImplicitAs.impl_witness.6bf, element0 [concrete = %Int.as.ImplicitAs.impl.Convert.b09]
// CHECK:STDOUT:       %bound_method.d9e: <bound method> = bound_method %int_3.822, %impl.elem0 [concrete = %Int.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:       %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.ImplicitAs.impl.Convert(%int_32) [concrete = %Int.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:       %bound_method.320: <bound method> = bound_method %int_3.822, %specific_fn [concrete = %bound_method.e83]
// CHECK:STDOUT:       %Int.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.320(%int_3.822) [concrete = %int_3.1ba]
// CHECK:STDOUT:       %.f92: Core.IntLiteral = value_of_initializer %Int.as.ImplicitAs.impl.Convert.call [concrete = %int_3.1ba]
// CHECK:STDOUT:       %.efe: Core.IntLiteral = converted %int_3.822, %.f92 [concrete = %int_3.1ba]
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.25c: @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert.type (%Int.as.ImplicitAs.impl.Convert.type.543) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert (constants.%Int.as.ImplicitAs.impl.Convert.c08)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.e99 = impl_witness_table (%Core.import_ref.25c), @Int.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%N.51e) {
// CHECK:STDOUT:   %N.loc5_15.1 => constants.%N.51e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%int_3.822) {
// CHECK:STDOUT:   %N.loc5_15.1 => constants.%int_3.822
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc11_23.2 => constants.%inst.splice_block
// CHECK:STDOUT: }
// CHECK:STDOUT:

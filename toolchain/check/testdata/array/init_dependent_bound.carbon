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
  var unused arr: array(T, 0) = ();
  //@dump-sem-ir-end
}

class C {}

fn H() {
  G(C);
}

// --- fail_init_dependent_bound.carbon

library "[[@TEST_NAME]]";

fn F(N:! i32) {
  // CHECK:STDERR: fail_init_dependent_bound.carbon:[[@LINE+4]]:35: error: cannot initialize array with dependent bound from a list of initializers [ArrayInitDependentBound]
  // CHECK:STDERR:   var unused arr: array(i32, N) = (1, 2, 3);
  // CHECK:STDERR:                                   ^~~~~~~~~
  // CHECK:STDERR:
  var unused arr: array(i32, N) = (1, 2, 3);
}

// --- fail_todo_init_template_dependent_bound.carbon

library "[[@TEST_NAME]]";

// TODO: This should be valid.
fn G(template N:! i32) {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_init_template_dependent_bound.carbon:[[@LINE+4]]:19: error: cannot evaluate type expression [TypeExprEvaluationFailure]
  // CHECK:STDERR:   var unused arr: array(i32, N) = (1, 2, 3);
  // CHECK:STDERR:                   ^~~~~~~~~~~~~
  // CHECK:STDERR:
  var unused arr: array(i32, N) = (1, 2, 3);
  //@dump-sem-ir-end
}

fn H() { G(3); }

// CHECK:STDOUT: --- generic_empty.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %array_type.1b3: type = array_type %int_0, %T [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %array_type.1b3 [symbolic]
// CHECK:STDOUT:   %pattern_type.bd6: type = pattern_type %array_type.1b3 [symbolic]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %array.ca4: %array_type.1b3 = tuple_value () [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %array_type.1b3, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.20b: %Destroy.type = facet_value %array_type.1b3, (%Destroy.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.140: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.20b) [symbolic]
// CHECK:STDOUT:   %.032: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.140, %Destroy.facet.20b [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.032 = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @Destroy.WithSelf.Op(%Destroy.facet.20b) [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %array_type.c56: type = array_type %int_0, %C [concrete]
// CHECK:STDOUT:   %complete_type.152: <witness> = complete_type_witness %array_type.c56 [concrete]
// CHECK:STDOUT:   %pattern_type.739: type = pattern_type %array_type.c56 [concrete]
// CHECK:STDOUT:   %array.610: %array_type.c56 = tuple_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.340: <witness> = custom_witness (%Destroy.Op), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.a8e: %Destroy.type = facet_value %array_type.c56, (%custom_witness.340) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.6ec: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.a8e) [concrete]
// CHECK:STDOUT:   %.40c: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.6ec, %Destroy.facet.a8e [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%T.loc4_7.2: type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %array_type.loc7_29.2: type = array_type constants.%int_0, %T.loc4_7.1 [symbolic = %array_type.loc7_29.2 (constants.%array_type.1b3)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %array_type.loc7_29.2 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %array_type.loc7_29.2 [symbolic = %pattern_type (constants.%pattern_type.bd6)]
// CHECK:STDOUT:   %array: @G.%array_type.loc7_29.2 (%array_type.1b3) = tuple_value () [symbolic = %array (constants.%array.ca4)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %array_type.loc7_29.2, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %array_type.loc7_29.2, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet (constants.%Destroy.facet.20b)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.140)]
// CHECK:STDOUT:   %.loc7_3.2: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Destroy.facet [symbolic = %.loc7_3.2 (constants.%.032)]
// CHECK:STDOUT:   %impl.elem0.loc7_3.2: @G.%.loc7_3.2 (%.032) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc7_3.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc7_3.2: <specific function> = specific_impl_function %impl.elem0.loc7_3.2, @Destroy.WithSelf.Op(%Destroy.facet) [symbolic = %specific_impl_fn.loc7_3.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %arr.patt: @G.%pattern_type (%pattern_type.bd6) = ref_binding_pattern arr [concrete]
// CHECK:STDOUT:       %arr.var_patt: @G.%pattern_type (%pattern_type.bd6) = var_pattern %arr.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %arr.var: ref @G.%array_type.loc7_29.2 (%array_type.1b3) = var %arr.var_patt
// CHECK:STDOUT:     %.loc7_34.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc7_34.2: init @G.%array_type.loc7_29.2 (%array_type.1b3) to %arr.var = array_init () [symbolic = %array (constants.%array.ca4)]
// CHECK:STDOUT:     %.loc7_3.1: init @G.%array_type.loc7_29.2 (%array_type.1b3) = converted %.loc7_34.1, %.loc7_34.2 [symbolic = %array (constants.%array.ca4)]
// CHECK:STDOUT:     assign %arr.var, %.loc7_3.1
// CHECK:STDOUT:     %.loc7_29: type = splice_block %array_type.loc7_29.1 [symbolic = %array_type.loc7_29.2 (constants.%array_type.1b3)] {
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc4_7.2 [symbolic = %T.loc4_7.1 (constants.%T)]
// CHECK:STDOUT:       %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:       %array_type.loc7_29.1: type = array_type %int_0, %T.ref [symbolic = %array_type.loc7_29.2 (constants.%array_type.1b3)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %arr: ref @G.%array_type.loc7_29.2 (%array_type.1b3) = ref_binding arr, %arr.var
// CHECK:STDOUT:     %impl.elem0.loc7_3.1: @G.%.loc7_3.2 (%.032) = impl_witness_access constants.%Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc7_3.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc7_3.1: <bound method> = bound_method %arr.var, %impl.elem0.loc7_3.1
// CHECK:STDOUT:     %specific_impl_fn.loc7_3.1: <specific function> = specific_impl_function %impl.elem0.loc7_3.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.20b) [symbolic = %specific_impl_fn.loc7_3.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %bound_method.loc7_3.2: <bound method> = bound_method %arr.var, %specific_impl_fn.loc7_3.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc7_3.2(%arr.var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %array_type.c56) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%T) {
// CHECK:STDOUT:   %T.loc4_7.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%C) {
// CHECK:STDOUT:   %T.loc4_7.1 => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %array_type.loc7_29.2 => constants.%array_type.c56
// CHECK:STDOUT:   %require_complete => constants.%complete_type.152
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.739
// CHECK:STDOUT:   %array => constants.%array.610
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.340
// CHECK:STDOUT:   %Destroy.facet => constants.%Destroy.facet.a8e
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.6ec
// CHECK:STDOUT:   %.loc7_3.2 => constants.%.40c
// CHECK:STDOUT:   %impl.elem0.loc7_3.2 => constants.%Destroy.Op
// CHECK:STDOUT:   %specific_impl_fn.loc7_3.2 => constants.%Destroy.Op
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_init_template_dependent_bound.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %N.afc: %i32 = symbolic_binding N, 0, template [template]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%int_1, %int_2, %int_3.1ba) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.4bd: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %From: Core.IntLiteral = symbolic_binding From, 0 [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.2f8: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%From) [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.7c2: %Int.as.ImplicitAs.impl.Convert.type.2f8 = struct_value () [symbolic]
// CHECK:STDOUT:   %int_3.49a: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.d2b: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.07d, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.73a: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.735: %Int.as.ImplicitAs.impl.Convert.type.73a = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.4dd: %ImplicitAs.type.4bd = facet_value %i32, (%ImplicitAs.impl_witness.d2b) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.f57: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet.4dd) [concrete]
// CHECK:STDOUT:   %.516: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.f57, %ImplicitAs.facet.4dd [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_3.49a, %Int.as.ImplicitAs.impl.Convert.735 [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Int.as.ImplicitAs.impl.Convert.735, @Int.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.9a4: <bound method> = bound_method %int_3.49a, %Int.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %inst.splice_block: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.994: Core.IntLiteral = splice_block %.bb5 [concrete = %int_3.1ba] {
// CHECK:STDOUT:       %impl.elem0: %.516 = impl_witness_access %ImplicitAs.impl_witness.d2b, element0 [concrete = %Int.as.ImplicitAs.impl.Convert.735]
// CHECK:STDOUT:       %bound_method.be8: <bound method> = bound_method %int_3.49a, %impl.elem0 [concrete = %Int.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:       %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.ImplicitAs.impl.Convert(%int_32) [concrete = %Int.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:       %bound_method.2df: <bound method> = bound_method %int_3.49a, %specific_fn [concrete = %bound_method.9a4]
// CHECK:STDOUT:       %Int.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.2df(%int_3.49a) [concrete = %int_3.1ba]
// CHECK:STDOUT:       %.aa0: Core.IntLiteral = value_of_initializer %Int.as.ImplicitAs.impl.Convert.call [concrete = %int_3.1ba]
// CHECK:STDOUT:       %.bb5: Core.IntLiteral = converted %int_3.49a, %.aa0 [concrete = %int_3.1ba]
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.dd0: @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert.type (%Int.as.ImplicitAs.impl.Convert.type.2f8) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert (constants.%Int.as.ImplicitAs.impl.Convert.7c2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.07d = impl_witness_table (%Core.import_ref.dd0), @Int.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%N.loc5_16.2: %i32) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc11_30.2: <instruction> = convert_to_value_action %N.ref, Core.IntLiteral [template]
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
// CHECK:STDOUT:     %.loc11_43: %tuple.type = tuple_literal (%int_1, %int_2, %int_3) [concrete = constants.%tuple]
// CHECK:STDOUT:     assign %arr.var, <error>
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %arr: ref <error> = ref_binding arr, <error> [concrete = <error>]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%N.afc) {
// CHECK:STDOUT:   %N.loc5_16.1 => constants.%N.afc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%int_3.49a) {
// CHECK:STDOUT:   %N.loc5_16.1 => constants.%int_3.49a
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc11_30.2 => constants.%inst.splice_block
// CHECK:STDOUT: }
// CHECK:STDOUT:

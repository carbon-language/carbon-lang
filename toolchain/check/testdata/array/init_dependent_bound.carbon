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

fn G(generic T: type) {
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

fn F(generic N: i32) {
  // CHECK:STDERR: fail_init_dependent_bound.carbon:[[@LINE+4]]:35: error: cannot initialize array with dependent bound from a list of initializers [ArrayInitDependentBound]
  // CHECK:STDERR:   var unused arr: array(i32, N) = (1, 2, 3);
  // CHECK:STDERR:                                   ^~~~~~~~~
  // CHECK:STDERR:
  var unused arr: array(i32, N) = (1, 2, 3);
}

// --- fail_todo_init_template_dependent_bound.carbon

library "[[@TEST_NAME]]";

// TODO: This should be valid.
fn G(template N: i32) {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_init_template_dependent_bound.carbon:[[@LINE+4]]:35: error: cannot initialize array with dependent bound from a list of initializers [ArrayInitDependentBound]
  // CHECK:STDERR:   var unused arr: array(i32, N) = (1, 2, 3);
  // CHECK:STDERR:                                   ^~~~~~~~~
  // CHECK:STDERR:
  var unused arr: array(i32, N) = (1, 2, 3);
  //@dump-sem-ir-end
}

fn H() { G(3); }

// CHECK:STDOUT: --- generic_empty.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %array_type.1b3: type = array_type %int_0, %T [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %array_type.1b3 [symbolic]
// CHECK:STDOUT:   %pattern_type.bd6: type = pattern_type %array_type.1b3 [symbolic]
// CHECK:STDOUT:   %arr.patt.770: %pattern_type.bd6 = ref_binding_pattern arr [symbolic]
// CHECK:STDOUT:   %arr.var_patt.5fc: %pattern_type.bd6 = var_pattern %arr.patt.770 [symbolic]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %array.ca4: %array_type.1b3 = tuple_value () [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %array_type.1b3, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.a52: %Destroy.type = facet_value %array_type.1b3, (%Destroy.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.a67: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.a52) [symbolic]
// CHECK:STDOUT:   %.1e5: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.a67, %Destroy.facet.a52 [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.1e5 = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @Destroy.WithSelf.Op(%Destroy.facet.a52) [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %array_type.988: type = array_type %int_0, %C [concrete]
// CHECK:STDOUT:   %complete_type.e80: <witness> = complete_type_witness %array_type.988 [concrete]
// CHECK:STDOUT:   %pattern_type.6be: type = pattern_type %array_type.988 [concrete]
// CHECK:STDOUT:   %arr.patt.c91: %pattern_type.6be = ref_binding_pattern arr [concrete]
// CHECK:STDOUT:   %arr.var_patt.9dd: %pattern_type.6be = var_pattern %arr.patt.c91 [concrete]
// CHECK:STDOUT:   %array.497: %array_type.988 = tuple_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df9: <witness> = custom_witness (%Destroy.Op), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.396: %Destroy.type = facet_value %array_type.988, (%custom_witness.df9) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ccc: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.396) [concrete]
// CHECK:STDOUT:   %.a02: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.ccc, %Destroy.facet.396 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%T.loc4_15.2: type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %array_type.loc7_29.2: type = array_type constants.%int_0, %T.loc4_15.1 [symbolic = %array_type.loc7_29.2 (constants.%array_type.1b3)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %array_type.loc7_29.2 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %array_type.loc7_29.2 [symbolic = %pattern_type (constants.%pattern_type.bd6)]
// CHECK:STDOUT:   %arr.patt.loc7_17.2: @G.%pattern_type (%pattern_type.bd6) = ref_binding_pattern arr [symbolic = %arr.patt.loc7_17.2 (constants.%arr.patt.770)]
// CHECK:STDOUT:   %arr.var_patt.loc7_3.2: @G.%pattern_type (%pattern_type.bd6) = var_pattern %arr.patt.loc7_17.2 [symbolic = %arr.var_patt.loc7_3.2 (constants.%arr.var_patt.5fc)]
// CHECK:STDOUT:   %array: @G.%array_type.loc7_29.2 (%array_type.1b3) = tuple_value () [symbolic = %array (constants.%array.ca4)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %array_type.loc7_29.2, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness)]
// CHECK:STDOUT:   %Destroy.facet.loc7_3.3: %Destroy.type = facet_value %array_type.loc7_29.2, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet.loc7_3.3 (constants.%Destroy.facet.a52)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc7_3.3) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.a67)]
// CHECK:STDOUT:   %.loc7_3.4: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Destroy.facet.loc7_3.3 [symbolic = %.loc7_3.4 (constants.%.1e5)]
// CHECK:STDOUT:   %impl.elem0.loc7_3.2: @G.%.loc7_3.4 (%.1e5) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc7_3.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc7_3.2: <specific function> = specific_impl_function %impl.elem0.loc7_3.2, @Destroy.WithSelf.Op(%Destroy.facet.loc7_3.3) [symbolic = %specific_impl_fn.loc7_3.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %arr.var: ref @G.%array_type.loc7_29.2 (%array_type.1b3) = var_storage %arr.var_patt.loc7_3.1
// CHECK:STDOUT:     %.loc7_34.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc7_34.2: init @G.%array_type.loc7_29.2 (%array_type.1b3) to %arr.var = array_init () [symbolic = %array (constants.%array.ca4)]
// CHECK:STDOUT:     %.loc7_3.1: init @G.%array_type.loc7_29.2 (%array_type.1b3) = converted %.loc7_34.1, %.loc7_34.2 [symbolic = %array (constants.%array.ca4)]
// CHECK:STDOUT:     assign %arr.var, %.loc7_3.1
// CHECK:STDOUT:     %.loc7_29: type = splice_block %array_type.loc7_29.1 [symbolic = %array_type.loc7_29.2 (constants.%array_type.1b3)] {
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc4_15.2 [symbolic = %T.loc4_15.1 (constants.%T)]
// CHECK:STDOUT:       %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:       %array_type.loc7_29.1: type = array_type %int_0, %T.ref [symbolic = %array_type.loc7_29.2 (constants.%array_type.1b3)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %arr: ref @G.%array_type.loc7_29.2 (%array_type.1b3) = wrapper_binding arr, %arr.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %arr.patt.loc7_17.1: @G.%pattern_type (%pattern_type.bd6) = ref_binding_pattern arr [symbolic = %arr.patt.loc7_17.2 (constants.%arr.patt.770)]
// CHECK:STDOUT:       %arr.var_patt.loc7_3.1: @G.%pattern_type (%pattern_type.bd6) = var_pattern %arr.patt.loc7_17.1 [symbolic = %arr.var_patt.loc7_3.2 (constants.%arr.var_patt.5fc)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %impl.elem0.loc7_3.1: @G.%.loc7_3.4 (%.1e5) = impl_witness_access constants.%Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc7_3.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc7_3.1: <bound method> = bound_method %arr.var, %impl.elem0.loc7_3.1
// CHECK:STDOUT:     %Destroy.facet.loc7_3.1: %Destroy.type = facet_value constants.%array_type.1b3, (constants.%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet.loc7_3.3 (constants.%Destroy.facet.a52)]
// CHECK:STDOUT:     %.loc7_3.2: %Destroy.type = converted constants.%array_type.1b3, %Destroy.facet.loc7_3.1 [symbolic = %Destroy.facet.loc7_3.3 (constants.%Destroy.facet.a52)]
// CHECK:STDOUT:     %Destroy.facet.loc7_3.2: %Destroy.type = facet_value constants.%array_type.1b3, (constants.%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet.loc7_3.3 (constants.%Destroy.facet.a52)]
// CHECK:STDOUT:     %.loc7_3.3: %Destroy.type = converted constants.%array_type.1b3, %Destroy.facet.loc7_3.2 [symbolic = %Destroy.facet.loc7_3.3 (constants.%Destroy.facet.a52)]
// CHECK:STDOUT:     %specific_impl_fn.loc7_3.1: <specific function> = specific_impl_function %impl.elem0.loc7_3.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.a52) [symbolic = %specific_impl_fn.loc7_3.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %bound_method.loc7_3.2: <bound method> = bound_method %arr.var, %specific_impl_fn.loc7_3.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc7_3.2(%arr.var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %array_type.988) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc4_15.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc4_15.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%C) {
// CHECK:STDOUT:   %T.patt.loc4_15.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc4_15.1 => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %array_type.loc7_29.2 => constants.%array_type.988
// CHECK:STDOUT:   %require_complete => constants.%complete_type.e80
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.6be
// CHECK:STDOUT:   %arr.patt.loc7_17.2 => constants.%arr.patt.c91
// CHECK:STDOUT:   %arr.var_patt.loc7_3.2 => constants.%arr.var_patt.9dd
// CHECK:STDOUT:   %array => constants.%array.497
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.df9
// CHECK:STDOUT:   %Destroy.facet.loc7_3.3 => constants.%Destroy.facet.396
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.ccc
// CHECK:STDOUT:   %.loc7_3.4 => constants.%.a02
// CHECK:STDOUT:   %impl.elem0.loc7_3.2 => constants.%Destroy.Op
// CHECK:STDOUT:   %specific_impl_fn.loc7_3.2 => constants.%Destroy.Op
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_init_template_dependent_bound.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %N.patt.38a: %pattern_type.6b6 = symbolic_binding_pattern N, 0, template [template]
// CHECK:STDOUT:   %N.37f: %i32 = symbolic_binding N, 0, template [template]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %.233: Core.IntLiteral = splice_inst @G.%.loc11_30.2 [template]
// CHECK:STDOUT:   %array_type.3cc: type = array_type %.233, %i32 [template]
// CHECK:STDOUT:   %require_complete.2bf: <witness> = require_complete_type %array_type.3cc [template]
// CHECK:STDOUT:   %pattern_type.ed3: type = pattern_type %array_type.3cc [template]
// CHECK:STDOUT:   %arr.patt.515: %pattern_type.ed3 = ref_binding_pattern arr [template]
// CHECK:STDOUT:   %arr.var_patt.821: %pattern_type.ed3 = var_pattern %arr.patt.515 [template]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%int_1, %int_2, %int_3.1ba) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Self.0e7: %Destroy.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.d3e: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Self.0e7) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.42b: %Destroy.WithSelf.Op.type.d3e = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.assoc_type: type = assoc_entity_type @Destroy [concrete]
// CHECK:STDOUT:   %assoc0.ae8: %Destroy.assoc_type = assoc_entity element0, imports.%Core.import_ref.918 [concrete]
// CHECK:STDOUT:   %.a09: %array_type.3cc = splice_inst @G.%.loc11_3.4 [template]
// CHECK:STDOUT:   %.b71: @G.%.loc11_3.7 (@G.%.loc11_3.7) = splice_inst @G.%.loc11_3.6 [template]
// CHECK:STDOUT:   %.098: @G.%.loc11_3.10 (@G.%.loc11_3.10) = splice_inst @G.%.loc11_3.9 [template]
// CHECK:STDOUT:   %From: Core.IntLiteral = symbolic_binding From, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.7cb: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.48d: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%From) [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.0f9: %Int.as.ImplicitAs.impl.Convert.type.48d = struct_value () [symbolic]
// CHECK:STDOUT:   %int_3.410: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.590: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.204, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.683: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.69e: %Int.as.ImplicitAs.impl.Convert.type.683 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.8de: %ImplicitAs.type.7cb = facet_value %i32, (%ImplicitAs.impl_witness.590) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.177: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet.8de) [concrete]
// CHECK:STDOUT:   %.e3f: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.177, %ImplicitAs.facet.8de [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_3.410, %Int.as.ImplicitAs.impl.Convert.69e [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Int.as.ImplicitAs.impl.Convert.69e, @Int.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.545: <bound method> = bound_method %int_3.410, %Int.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %inst.splice_block.ca0: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.354: Core.IntLiteral = splice_block %.238 [concrete = %int_3.1ba] {
// CHECK:STDOUT:       %impl.elem0.923: %.e3f = impl_witness_access %ImplicitAs.impl_witness.590, element0 [concrete = %Int.as.ImplicitAs.impl.Convert.69e]
// CHECK:STDOUT:       %bound_method.1cb: <bound method> = bound_method %int_3.410, %impl.elem0.923 [concrete = %Int.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:       %specific_fn: <specific function> = specific_function %impl.elem0.923, @Int.as.ImplicitAs.impl.Convert(%int_32) [concrete = %Int.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:       %bound_method.ae8: <bound method> = bound_method %int_3.410, %specific_fn [concrete = %bound_method.545]
// CHECK:STDOUT:       %Int.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.ae8(%int_3.410) [concrete = %int_3.1ba]
// CHECK:STDOUT:       %.238: Core.IntLiteral = value_of_initializer %Int.as.ImplicitAs.impl.Convert.call [concrete = %int_3.1ba]
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %array_type.dc7: type = array_type %int_3.1ba, %i32 [concrete]
// CHECK:STDOUT:   %complete_type.b77: <witness> = complete_type_witness %array_type.dc7 [concrete]
// CHECK:STDOUT:   %pattern_type.771: type = pattern_type %array_type.dc7 [concrete]
// CHECK:STDOUT:   %arr.patt.316: %pattern_type.771 = ref_binding_pattern arr [concrete]
// CHECK:STDOUT:   %arr.var_patt.c63: %pattern_type.771 = var_pattern %arr.patt.316 [concrete]
// CHECK:STDOUT:   %inst.as_compatible: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.723: ref %array_type.dc7 = as_compatible @G.%arr.var
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.3: type = fn_type @Destroy.Op.loc11_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.3: %Destroy.Op.type.1d8f74.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df9cc1.3: <witness> = custom_witness (%Destroy.Op.1a2547.3), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.9a9: %Destroy.type = facet_value %array_type.dc7, (%custom_witness.df9cc1.3) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.642: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.9a9) [concrete]
// CHECK:STDOUT:   %.6cd: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.642, %Destroy.facet.9a9 [concrete]
// CHECK:STDOUT:   %inst.splice_block.adf: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.6fe: <bound method> = splice_block %bound_method.69b {
// CHECK:STDOUT:       %impl.elem0.ef2: %.6cd = impl_witness_access %custom_witness.df9cc1.3, element0 [concrete = %Destroy.Op.1a2547.3]
// CHECK:STDOUT:       %bound_method.69b: <bound method> = bound_method %.723, %impl.elem0.ef2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.call: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %Destroy.Op.call: init %empty_tuple.type = call %.6fe(%.723)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.918: @Destroy.WithSelf.%Destroy.WithSelf.Op.type (%Destroy.WithSelf.Op.type.d3e) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @Destroy.WithSelf.%Destroy.WithSelf.Op (constants.%Destroy.WithSelf.Op.42b)]
// CHECK:STDOUT:   %Core.import_ref.32e: @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert.type (%Int.as.ImplicitAs.impl.Convert.type.48d) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert (constants.%Int.as.ImplicitAs.impl.Convert.0f9)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.204 = impl_witness_table (%Core.import_ref.32e), @Int.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%N.loc5_16.2: %i32) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc11_30.2: <instruction> = convert_to_value_action %N.ref, Core.IntLiteral [template]
// CHECK:STDOUT:   %.loc11_30.3: Core.IntLiteral = splice_inst %.loc11_30.2 [template = %.loc11_30.3 (constants.%.233)]
// CHECK:STDOUT:   %array_type.loc11_31.2: type = array_type %.loc11_30.3, constants.%i32 [template = %array_type.loc11_31.2 (constants.%array_type.3cc)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %array_type.loc11_31.2 [template = %require_complete (constants.%require_complete.2bf)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %array_type.loc11_31.2 [template = %pattern_type (constants.%pattern_type.ed3)]
// CHECK:STDOUT:   %arr.patt.loc11_17.2: @G.%pattern_type (%pattern_type.ed3) = ref_binding_pattern arr [template = %arr.patt.loc11_17.2 (constants.%arr.patt.515)]
// CHECK:STDOUT:   %arr.var_patt.loc11_3.2: @G.%pattern_type (%pattern_type.ed3) = var_pattern %arr.patt.loc11_17.2 [template = %arr.var_patt.loc11_3.2 (constants.%arr.var_patt.821)]
// CHECK:STDOUT:   %.loc11_3.4: <instruction> = refine_type_action %arr.var, %array_type.loc11_31.2 [template]
// CHECK:STDOUT:   %.loc11_3.5: @G.%array_type.loc11_31.2 (%array_type.3cc) = splice_inst %.loc11_3.4 [template = %.loc11_3.5 (constants.%.a09)]
// CHECK:STDOUT:   %.loc11_3.6: <instruction> = compound_member_access_action %.loc11_3.1, constants.%assoc0.ae8 [template]
// CHECK:STDOUT:   %.loc11_3.7: type = type_of_inst %.loc11_3.6 [template]
// CHECK:STDOUT:   %.loc11_3.8: @G.%.loc11_3.7 (@G.%.loc11_3.7) = splice_inst %.loc11_3.6 [template = %.loc11_3.8 (constants.%.b71)]
// CHECK:STDOUT:   %.loc11_3.9: <instruction> = call_action (%.loc11_3.2), true [template]
// CHECK:STDOUT:   %.loc11_3.10: type = type_of_inst %.loc11_3.9 [template]
// CHECK:STDOUT:   %.loc11_3.11: @G.%.loc11_3.10 (@G.%.loc11_3.10) = splice_inst %.loc11_3.9 [template = %.loc11_3.11 (constants.%.098)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %arr.var: ref @G.%array_type.loc11_31.2 (%array_type.3cc) = var_storage %arr.var_patt.loc11_3.1
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %.loc11_43: %tuple.type = tuple_literal (%int_1, %int_2, %int_3) [concrete = constants.%tuple]
// CHECK:STDOUT:     assign %arr.var, <error>
// CHECK:STDOUT:     %.loc11_31: type = splice_block %array_type.loc11_31.1 [template = %array_type.loc11_31.2 (constants.%array_type.3cc)] {
// CHECK:STDOUT:       %i32.loc11: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %N.ref: %i32 = name_ref N, %N.loc5_16.2 [template = %N.loc5_16.1 (constants.%N.37f)]
// CHECK:STDOUT:       %.loc11_30.1: Core.IntLiteral = splice_inst %.loc11_30.2 [template = %.loc11_30.3 (constants.%.233)]
// CHECK:STDOUT:       %array_type.loc11_31.1: type = array_type %.loc11_30.1, %i32.loc11 [template = %array_type.loc11_31.2 (constants.%array_type.3cc)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %arr: ref @G.%array_type.loc11_31.2 (%array_type.3cc) = wrapper_binding arr, %arr.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %arr.patt.loc11_17.1: @G.%pattern_type (%pattern_type.ed3) = ref_binding_pattern arr [template = %arr.patt.loc11_17.2 (constants.%arr.patt.515)]
// CHECK:STDOUT:       %arr.var_patt.loc11_3.1: @G.%pattern_type (%pattern_type.ed3) = var_pattern %arr.patt.loc11_17.1 [template = %arr.var_patt.loc11_3.2 (constants.%arr.var_patt.821)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc11_3.1: @G.%array_type.loc11_31.2 (%array_type.3cc) = splice_inst %.loc11_3.4 [template = %.loc11_3.5 (constants.%.a09)]
// CHECK:STDOUT:     %.loc11_3.2: @G.%.loc11_3.7 (@G.%.loc11_3.7) = splice_inst %.loc11_3.6 [template = %.loc11_3.8 (constants.%.b71)]
// CHECK:STDOUT:     %.loc11_3.3: @G.%.loc11_3.10 (@G.%.loc11_3.10) = splice_inst %.loc11_3.9 [template = %.loc11_3.11 (constants.%.098)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc11_3.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc11_3.2(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc11_3.3(%self.param: ref %array_type.dc7) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%N.37f) {
// CHECK:STDOUT:   %N.patt.loc5_16.2 => constants.%N.patt.38a
// CHECK:STDOUT:   %N.loc5_16.1 => constants.%N.37f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%int_3.410) {
// CHECK:STDOUT:   %N.patt.loc5_16.2 => constants.%N.patt.38a
// CHECK:STDOUT:   %N.loc5_16.1 => constants.%int_3.410
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc11_30.2 => constants.%inst.splice_block.ca0
// CHECK:STDOUT:   %.loc11_30.3 => constants.%int_3.1ba
// CHECK:STDOUT:   %array_type.loc11_31.2 => constants.%array_type.dc7
// CHECK:STDOUT:   %require_complete => constants.%complete_type.b77
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.771
// CHECK:STDOUT:   %arr.patt.loc11_17.2 => constants.%arr.patt.316
// CHECK:STDOUT:   %arr.var_patt.loc11_3.2 => constants.%arr.var_patt.c63
// CHECK:STDOUT:   %.loc11_3.4 => constants.%inst.as_compatible
// CHECK:STDOUT:   %.loc11_3.5 => invalid
// CHECK:STDOUT:   %.loc11_3.6 => constants.%inst.splice_block.adf
// CHECK:STDOUT:   %.loc11_3.7 => <bound method>
// CHECK:STDOUT:   %.loc11_3.8 => invalid
// CHECK:STDOUT:   %.loc11_3.9 => constants.%inst.call
// CHECK:STDOUT:   %.loc11_3.10 => constants.%empty_tuple.type
// CHECK:STDOUT:   %.loc11_3.11 => invalid
// CHECK:STDOUT: }
// CHECK:STDOUT:

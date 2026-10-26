// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/array.carbon
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
  // CHECK:STDERR: fail_init_dependent_bound.carbon:[[@LINE+2]]:35-43: error: cannot initialize array with dependent bound from a list of initializers [ArrayInitDependentBound]
  // CHECK:STDERR:
  var unused arr: array(i32, N) = (1, 2, 3);
}

// --- fail_init_dependent_bound_in_constant.carbon
library "[[@TEST_NAME]]";

interface Y(T: type, U: T) {}

constraint Z(N: i32) {
  // Forming the specific of `Y` uses constant values which are canonical and
  // have no location. The error should still point somewhere reasonable.
  // CHECK:STDERR: fail_init_dependent_bound_in_constant.carbon:[[@LINE+3]]:24-42: error: cannot initialize array with dependent bound from a list of initializers [ArrayInitDependentBound]
  // CHECK:STDERR: fail_init_dependent_bound_in_constant.carbon:[[@LINE-6]]:1-28: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  extend require impls Y(array((), N), ());
}

// --- init_template_dependent_bound.carbon

library "[[@TEST_NAME]]";

fn G(template N: i32) {
  //@dump-sem-ir-begin
  var unused arr: array(i32, N) = (1, 2, 3);
  //@dump-sem-ir-end
}

fn H() { G(3); }

// CHECK:STDOUT: --- generic_empty.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %pattern_type.9a587c.1: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt.3a0592.1: %pattern_type.9a587c.1 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %Array.744: type = class_type @Array, @Array(%T.67d, %int_0) [symbolic]
// CHECK:STDOUT:   %require_complete.d2d: <witness> = require_complete_type %Array.744 [symbolic]
// CHECK:STDOUT:   %pattern_type.197: type = pattern_type %Array.744 [symbolic]
// CHECK:STDOUT:   %arr.patt.a3c: %pattern_type.197 = ref_binding_pattern arr [symbolic]
// CHECK:STDOUT:   %arr.var_patt.aeb: %pattern_type.197 = var_pattern %arr.patt.a3c [symbolic]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %array.3f3: %Array.744 = tuple_value () [symbolic]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.140: <witness> = lookup_impl_witness %Array.744, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.2ad: <witness> = lookup_impl_witness %Array.744, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.facet.b3e: %SubobjectDestroy.type = facet_value %Array.744, (%SubobjectDestroy.lookup_impl_witness.2ad) [symbolic]
// CHECK:STDOUT:   %.b07: require_specific_def_type = require_specific_def @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.b3e) [symbolic]
// CHECK:STDOUT:   %Destroy.facet.5cf: %Destroy.type = facet_value %Array.744, (%DestroyImpl.lookup_impl_witness.140, %SubobjectDestroy.lookup_impl_witness.2ad) [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %Array.184: type = class_type @Array, @Array(%C, %int_0) [concrete]
// CHECK:STDOUT:   %array_type.988: type = array_type %int_0, %C [concrete]
// CHECK:STDOUT:   %complete_type.e80: <witness> = complete_type_witness %array_type.988 [concrete]
// CHECK:STDOUT:   %pattern_type.feb: type = pattern_type %Array.184 [concrete]
// CHECK:STDOUT:   %arr.patt.081: %pattern_type.feb = ref_binding_pattern arr [concrete]
// CHECK:STDOUT:   %arr.var_patt.c85: %pattern_type.feb = var_pattern %arr.patt.081 [concrete]
// CHECK:STDOUT:   %array.27e: %Array.184 = tuple_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc7_3.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.466: %SubobjectDestroy.type = facet_value %Array.184, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.714: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.466) [concrete]
// CHECK:STDOUT:   %.426: require_specific_def_type = require_specific_def @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.466) [concrete]
// CHECK:STDOUT:   %Destroy.facet.29c: %Destroy.type = facet_value %Array.184, (%DestroyImpl.impl_witness.714, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.cac: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%Destroy.facet.29c) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%T.loc4_15.2: type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Array.loc7_29.2: type = class_type @Array, @Array(%T.loc4_15.1, constants.%int_0) [symbolic = %Array.loc7_29.2 (constants.%Array.744)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Array.loc7_29.2 [symbolic = %require_complete (constants.%require_complete.d2d)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Array.loc7_29.2 [symbolic = %pattern_type (constants.%pattern_type.197)]
// CHECK:STDOUT:   %arr.patt.loc7_17.2: @G.%pattern_type (%pattern_type.197) = ref_binding_pattern arr [symbolic = %arr.patt.loc7_17.2 (constants.%arr.patt.a3c)]
// CHECK:STDOUT:   %arr.var_patt.loc7_3.2: @G.%pattern_type (%pattern_type.197) = var_pattern %arr.patt.loc7_17.2 [symbolic = %arr.var_patt.loc7_3.2 (constants.%arr.var_patt.aeb)]
// CHECK:STDOUT:   %array: @G.%Array.loc7_29.2 (%Array.744) = tuple_value () [symbolic = %array (constants.%array.3f3)]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness: <witness> = lookup_impl_witness %Array.loc7_29.2, @SubobjectDestroy [symbolic = %SubobjectDestroy.lookup_impl_witness (constants.%SubobjectDestroy.lookup_impl_witness.2ad)]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %Array.loc7_29.2, (%SubobjectDestroy.lookup_impl_witness) [symbolic = %SubobjectDestroy.facet (constants.%SubobjectDestroy.facet.b3e)]
// CHECK:STDOUT:   %.loc7_3.4: require_specific_def_type = require_specific_def @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet) [symbolic = %.loc7_3.4 (constants.%.b07)]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness: <witness> = lookup_impl_witness %Array.loc7_29.2, @DestroyImpl [symbolic = %DestroyImpl.lookup_impl_witness (constants.%DestroyImpl.lookup_impl_witness.140)]
// CHECK:STDOUT:   %Destroy.facet.loc7_3.3: %Destroy.type = facet_value %Array.loc7_29.2, (%DestroyImpl.lookup_impl_witness, %SubobjectDestroy.lookup_impl_witness) [symbolic = %Destroy.facet.loc7_3.3 (constants.%Destroy.facet.5cf)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %arr.var: ref @G.%Array.loc7_29.2 (%Array.744) = var_storage %arr.var_patt.loc7_3.1
// CHECK:STDOUT:     %.loc7_34.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc7_34.2: init @G.%Array.loc7_29.2 (%Array.744) to %arr.var = array_init () [symbolic = %array (constants.%array.3f3)]
// CHECK:STDOUT:     %.loc7_3.1: init @G.%Array.loc7_29.2 (%Array.744) = converted %.loc7_34.1, %.loc7_34.2 [symbolic = %array (constants.%array.3f3)]
// CHECK:STDOUT:     assign %arr.var, %.loc7_3.1
// CHECK:STDOUT:     %.loc7_29: type = splice_block %Array.loc7_29.1 [symbolic = %Array.loc7_29.2 (constants.%Array.744)] {
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc4_15.2 [symbolic = %T.loc4_15.1 (constants.%T.67d)]
// CHECK:STDOUT:       %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:       %Array.loc7_29.1: type = class_type @Array, @Array(constants.%T.67d, constants.%int_0) [symbolic = %Array.loc7_29.2 (constants.%Array.744)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %arr: ref @G.%Array.loc7_29.2 (%Array.744) = wrapper_binding arr, %arr.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %arr.patt.loc7_17.1: @G.%pattern_type (%pattern_type.197) = ref_binding_pattern arr [symbolic = %arr.patt.loc7_17.2 (constants.%arr.patt.a3c)]
// CHECK:STDOUT:       %arr.var_patt.loc7_3.1: @G.%pattern_type (%pattern_type.197) = var_pattern %arr.patt.loc7_17.1 [symbolic = %arr.var_patt.loc7_3.2 (constants.%arr.var_patt.aeb)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Destroy.facet.loc7_3.1: %Destroy.type = facet_value constants.%Array.744, (constants.%DestroyImpl.lookup_impl_witness.140, constants.%SubobjectDestroy.lookup_impl_witness.2ad) [symbolic = %Destroy.facet.loc7_3.3 (constants.%Destroy.facet.5cf)]
// CHECK:STDOUT:     %.loc7_3.2: %Destroy.type = converted constants.%Array.744, %Destroy.facet.loc7_3.1 [symbolic = %Destroy.facet.loc7_3.3 (constants.%Destroy.facet.5cf)]
// CHECK:STDOUT:     %Destroy.facet.loc7_3.2: %Destroy.type = facet_value constants.%Array.744, (constants.%DestroyImpl.lookup_impl_witness.140, constants.%SubobjectDestroy.lookup_impl_witness.2ad) [symbolic = %Destroy.facet.loc7_3.3 (constants.%Destroy.facet.5cf)]
// CHECK:STDOUT:     %.loc7_3.3: %Destroy.type = converted constants.%Array.744, %Destroy.facet.loc7_3.2 [symbolic = %Destroy.facet.loc7_3.3 (constants.%Destroy.facet.5cf)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%arr.var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc7_3.1(%self.param: ref %array_type.988) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc7_3.2(%self.param: ref %Array.184) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%T.67d) {
// CHECK:STDOUT:   %T.patt.loc4_15.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc4_15.1 => constants.%T.67d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%C) {
// CHECK:STDOUT:   %T.patt.loc4_15.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc4_15.1 => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Array.loc7_29.2 => constants.%Array.184
// CHECK:STDOUT:   %require_complete => constants.%complete_type.e80
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.feb
// CHECK:STDOUT:   %arr.patt.loc7_17.2 => constants.%arr.patt.081
// CHECK:STDOUT:   %arr.var_patt.loc7_3.2 => constants.%arr.var_patt.c85
// CHECK:STDOUT:   %array => constants.%array.27e
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness => constants.%custom_witness.b80ef6.2
// CHECK:STDOUT:   %SubobjectDestroy.facet => constants.%SubobjectDestroy.facet.466
// CHECK:STDOUT:   %.loc7_3.4 => constants.%.426
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness => constants.%DestroyImpl.impl_witness.714
// CHECK:STDOUT:   %Destroy.facet.loc7_3.3 => constants.%Destroy.facet.29c
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.cac
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- init_template_dependent_bound.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %N.patt.d64: %pattern_type.d94 = symbolic_binding_pattern N, 0, template [template]
// CHECK:STDOUT:   %N.c34: %i32 = symbolic_binding N, 0, template [template]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %.da2: Core.IntLiteral = splice_inst @G.%.loc6_30.2 [template]
// CHECK:STDOUT:   %Array.type: type = parameterized_class_name_type @Array [concrete]
// CHECK:STDOUT:   %Array.generic: %Array.type = struct_value () [concrete]
// CHECK:STDOUT:   %.d25: type = type_of_inst @G.%.loc6_31.5 [template]
// CHECK:STDOUT:   %.24e: %.d25 = splice_inst @G.%.loc6_31.5 [template]
// CHECK:STDOUT:   %.537: type = splice_inst @G.%.loc6_31.8 [template]
// CHECK:STDOUT:   %require_complete.cb0: <witness> = require_complete_type %.537 [template]
// CHECK:STDOUT:   %pattern_type.fcb: type = pattern_type %.537 [template]
// CHECK:STDOUT:   %arr.patt.c8d: %pattern_type.fcb = ref_binding_pattern arr [template]
// CHECK:STDOUT:   %arr.var_patt.1e0: %pattern_type.fcb = var_pattern %arr.patt.c8d [template]
// CHECK:STDOUT:   %int_1.5b8115.1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc8af.1: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %tuple.type.37f: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.2d5: %tuple.type.37f = tuple_value (%int_1.5b8115.1, %int_2.ecc8af.1, %int_3.1ba) [concrete]
// CHECK:STDOUT:   %tuple.type.f7a: type = tuple_type (<instruction>) [concrete]
// CHECK:STDOUT:   %.5d6: %.537 = splice_inst @G.%tuple.elem0 [template]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %.8e2: type = type_of_inst @G.%.loc6_3.6 [template]
// CHECK:STDOUT:   %.9bf: %.8e2 = splice_inst @G.%.loc6_3.6 [template]
// CHECK:STDOUT:   %ImplicitAs.type.6b0: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6 = struct_value () [symbolic]
// CHECK:STDOUT:   %From: Core.IntLiteral = symbolic_binding From, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.5be: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.dfd: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%From) [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.914: %Int.as.ImplicitAs.impl.Convert.type.dfd = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.d73: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f53, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.fc6: %ImplicitAs.type.6b0 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.d73) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.58f: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet.fc6) [concrete]
// CHECK:STDOUT:   %.33a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.58f, %ImplicitAs.facet.fc6 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.b3c: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.fb0: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.9f3: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.323: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b7c, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.f39: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.779: %Int.as.ImplicitAs.impl.Convert.type.f39 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.0aa: %ImplicitAs.type.5be = facet_value %i32, (%ImplicitAs.impl_witness.323) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.4c7: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet.0aa) [concrete]
// CHECK:STDOUT:   %.ce7: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.4c7, %ImplicitAs.facet.0aa [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_3.9f3, %Int.as.ImplicitAs.impl.Convert.779 [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Int.as.ImplicitAs.impl.Convert.779, @Int.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.746: <bound method> = bound_method %int_3.9f3, %Int.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %inst.splice_block.426619.1: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.7c4: Core.IntLiteral = splice_block %.ef3 [concrete = %int_3.1ba] {
// CHECK:STDOUT:       %impl.elem0.2d4: %.ce7 = impl_witness_access %ImplicitAs.impl_witness.323, element0 [concrete = %Int.as.ImplicitAs.impl.Convert.779]
// CHECK:STDOUT:       %bound_method.657: <bound method> = bound_method %int_3.9f3, %impl.elem0.2d4 [concrete = %Int.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:       %specific_fn.e3b: <specific function> = specific_function %impl.elem0.2d4, @Int.as.ImplicitAs.impl.Convert(%int_32) [concrete = %Int.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:       %bound_method.93a: <bound method> = bound_method %int_3.9f3, %specific_fn.e3b [concrete = %bound_method.746]
// CHECK:STDOUT:       %Int.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.93a(%int_3.9f3) [concrete = %int_3.1ba]
// CHECK:STDOUT:       %.ef3: Core.IntLiteral = value_of_initializer %Int.as.ImplicitAs.impl.Convert.call [concrete = %int_3.1ba]
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Array.95c1f3.1: type = class_type @Array, @Array(%i32, %int_3.1ba) [concrete]
// CHECK:STDOUT:   %inst.class_type: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %Array.95c1f3.2: type = class_type @Array, @Array(%i32, %int_3.1ba) [concrete = %Array.95c1f3.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.426619.2: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.ce5: type = splice_block %Array.95c1f3.1 [concrete = %Array.95c1f3.1] {}
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %array_type.afd: type = array_type %int_3.1ba, %i32 [concrete]
// CHECK:STDOUT:   %complete_type.4fe: <witness> = complete_type_witness %array_type.afd [concrete]
// CHECK:STDOUT:   %pattern_type.d4e: type = pattern_type %Array.95c1f3.1 [concrete]
// CHECK:STDOUT:   %arr.patt.403: %pattern_type.d4e = ref_binding_pattern arr [concrete]
// CHECK:STDOUT:   %arr.var_patt.074: %pattern_type.d4e = var_pattern %arr.patt.403 [concrete]
// CHECK:STDOUT:   %int_0.5c61a1.1: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.282: <bound method> = bound_method %int_1.5b8115.1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %bound_method.3bd: <bound method> = bound_method %int_1.5b8115.1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.68a: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.8a6: <bound method> = bound_method %int_2.ecc8af.1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %bound_method.a55: <bound method> = bound_method %int_2.ecc8af.1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.2e0: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %array: %Array.95c1f3.1 = tuple_value (%int_1.68a, %int_2.2e0, %int_3.9f3) [concrete]
// CHECK:STDOUT:   %inst.splice_block.426619.3: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.e8f: init %Array.95c1f3.1 = splice_block %.0a1 [concrete = %array] {
// CHECK:STDOUT:       %tuple.elem0: Core.IntLiteral = tuple_access %tuple.2d5, element0 [concrete = %int_1.5b8115.1]
// CHECK:STDOUT:       %impl.elem0.5f53f7.1: %.33a = impl_witness_access %ImplicitAs.impl_witness.d73, element0 [concrete = %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:       %bound_method.273: <bound method> = bound_method %tuple.elem0, %impl.elem0.5f53f7.1 [concrete = %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.282]
// CHECK:STDOUT:       %specific_fn.54a37f.1: <specific function> = specific_function %impl.elem0.5f53f7.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete = %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:       %bound_method.109: <bound method> = bound_method %tuple.elem0, %specific_fn.54a37f.1 [concrete = %bound_method.3bd]
// CHECK:STDOUT:       %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.84a: init %i32 = call %bound_method.109(%tuple.elem0) [concrete = %int_1.68a]
// CHECK:STDOUT:       %.64d: init %i32 = converted %tuple.elem0, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.84a [concrete = %int_1.68a]
// CHECK:STDOUT:       %.202040.2: ref %Array.95c1f3.1 = specific_inst @G.%arr.var, @G(%int_3.9f3)
// CHECK:STDOUT:       %int_0.5c61a1.2: Core.IntLiteral = int_value 0 [concrete = %int_0.5c61a1.1]
// CHECK:STDOUT:       %.1c0: ref %i32 = array_index %.202040.2, %int_0.5c61a1.2
// CHECK:STDOUT:       %.e90: init %i32 to %.1c0 = in_place_init %.64d [concrete = %int_1.68a]
// CHECK:STDOUT:       %tuple.elem1: Core.IntLiteral = tuple_access %tuple.2d5, element1 [concrete = %int_2.ecc8af.1]
// CHECK:STDOUT:       %impl.elem0.5f53f7.2: %.33a = impl_witness_access %ImplicitAs.impl_witness.d73, element0 [concrete = %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:       %bound_method.166: <bound method> = bound_method %tuple.elem1, %impl.elem0.5f53f7.2 [concrete = %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.8a6]
// CHECK:STDOUT:       %specific_fn.54a37f.2: <specific function> = specific_function %impl.elem0.5f53f7.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete = %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:       %bound_method.1fb: <bound method> = bound_method %tuple.elem1, %specific_fn.54a37f.2 [concrete = %bound_method.a55]
// CHECK:STDOUT:       %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.acf: init %i32 = call %bound_method.1fb(%tuple.elem1) [concrete = %int_2.2e0]
// CHECK:STDOUT:       %.081: init %i32 = converted %tuple.elem1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.acf [concrete = %int_2.2e0]
// CHECK:STDOUT:       %int_1.5b8115.2: Core.IntLiteral = int_value 1 [concrete = %int_1.5b8115.1]
// CHECK:STDOUT:       %.873: ref %i32 = array_index %.202040.2, %int_1.5b8115.2
// CHECK:STDOUT:       %.ee8: init %i32 to %.873 = in_place_init %.081 [concrete = %int_2.2e0]
// CHECK:STDOUT:       %tuple.elem2: Core.IntLiteral = tuple_access %tuple.2d5, element2 [concrete = %int_3.1ba]
// CHECK:STDOUT:       %impl.elem0.5f53f7.3: %.33a = impl_witness_access %ImplicitAs.impl_witness.d73, element0 [concrete = %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:       %bound_method.44f: <bound method> = bound_method %tuple.elem2, %impl.elem0.5f53f7.3 [concrete = %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.b3c]
// CHECK:STDOUT:       %specific_fn.54a37f.3: <specific function> = specific_function %impl.elem0.5f53f7.3, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete = %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:       %bound_method.f04: <bound method> = bound_method %tuple.elem2, %specific_fn.54a37f.3 [concrete = %bound_method.fb0]
// CHECK:STDOUT:       %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.44c: init %i32 = call %bound_method.f04(%tuple.elem2) [concrete = %int_3.9f3]
// CHECK:STDOUT:       %.1ac: init %i32 = converted %tuple.elem2, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.44c [concrete = %int_3.9f3]
// CHECK:STDOUT:       %int_2.ecc8af.2: Core.IntLiteral = int_value 2 [concrete = %int_2.ecc8af.1]
// CHECK:STDOUT:       %.ea8: ref %i32 = array_index %.202040.2, %int_2.ecc8af.2
// CHECK:STDOUT:       %.823: init %i32 to %.ea8 = in_place_init %.1ac [concrete = %int_3.9f3]
// CHECK:STDOUT:       %.0a1: init %Array.95c1f3.1 to %.202040.2 = array_init (%.e90, %.ee8, %.823) [concrete = %array]
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %tuple.242: %tuple.type.f7a = tuple_value (%inst.splice_block.426619.3) [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc6_3.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.79c: %SubobjectDestroy.type = facet_value %Array.95c1f3.1, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.90e: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.79c) [concrete]
// CHECK:STDOUT:   %Destroy.facet.4164cf.1: %Destroy.type = facet_value %Array.95c1f3.1, (%DestroyImpl.impl_witness.90e, %custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %inst.splice_block.426619.4: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.de2: init %empty_tuple.type = splice_block %SelfDestruct.call {
// CHECK:STDOUT:       %.202040.1: ref %Array.95c1f3.1 = specific_inst @G.%arr.var, @G(%int_3.9f3)
// CHECK:STDOUT:       %Destroy.facet.4164cf.2: %Destroy.type = facet_value %Array.95c1f3.1, (%DestroyImpl.impl_witness.90e, %custom_witness.b80ef6.4) [concrete = %Destroy.facet.4164cf.1]
// CHECK:STDOUT:       %.c19077.1: %Destroy.type = converted %Array.95c1f3.1, %Destroy.facet.4164cf.2 [concrete = %Destroy.facet.4164cf.1]
// CHECK:STDOUT:       %Destroy.facet.4164cf.3: %Destroy.type = facet_value %Array.95c1f3.1, (%DestroyImpl.impl_witness.90e, %custom_witness.b80ef6.4) [concrete = %Destroy.facet.4164cf.1]
// CHECK:STDOUT:       %.c19077.2: %Destroy.type = converted %Array.95c1f3.1, %Destroy.facet.4164cf.3 [concrete = %Destroy.facet.4164cf.1]
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn.cd831d.2(%.202040.1)
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.e47: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f53 = impl_witness_table (%Core.import_ref.e47), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.ed0: @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert.type (%Int.as.ImplicitAs.impl.Convert.type.dfd) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert (constants.%Int.as.ImplicitAs.impl.Convert.914)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b7c = impl_witness_table (%Core.import_ref.ed0), @Int.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%N.loc4_16.2: %i32) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc6_30.2: <instruction> = convert_to_value_action %N.ref, Core.IntLiteral [template]
// CHECK:STDOUT:   %.loc6_30.3: Core.IntLiteral = splice_inst %.loc6_30.2 [template = %.loc6_30.3 (constants.%.da2)]
// CHECK:STDOUT:   %.loc6_31.5: <instruction> = call_action (constants.%Array.generic, %i32.loc6, %.loc6_30.1), false [template]
// CHECK:STDOUT:   %.loc6_31.6: type = type_of_inst %.loc6_31.5 [template = %.loc6_31.6 (constants.%.d25)]
// CHECK:STDOUT:   %.loc6_31.7: @G.%.loc6_31.6 (%.d25) = splice_inst %.loc6_31.5 [template = %.loc6_31.7 (constants.%.24e)]
// CHECK:STDOUT:   %.loc6_31.8: <instruction> = convert_to_value_action %.loc6_31.3, type [template]
// CHECK:STDOUT:   %.loc6_31.9: type = splice_inst %.loc6_31.8 [template = %.loc6_31.9 (constants.%.537)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %.loc6_31.9 [template = %require_complete (constants.%require_complete.cb0)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %.loc6_31.9 [template = %pattern_type (constants.%pattern_type.fcb)]
// CHECK:STDOUT:   %arr.patt.loc6_17.2: @G.%pattern_type (%pattern_type.fcb) = ref_binding_pattern arr [template = %arr.patt.loc6_17.2 (constants.%arr.patt.c8d)]
// CHECK:STDOUT:   %arr.var_patt.loc6_3.2: @G.%pattern_type (%pattern_type.fcb) = var_pattern %arr.patt.loc6_17.2 [template = %arr.var_patt.loc6_3.2 (constants.%arr.var_patt.1e0)]
// CHECK:STDOUT:   %.loc6_3.4: %tuple.type.f7a = initialize_action %.loc6_43, %.loc6_31.9, %arr.var, none, false [template]
// CHECK:STDOUT:   %tuple.elem0: <instruction> = tuple_access %.loc6_3.4, element0 [template]
// CHECK:STDOUT:   %.loc6_3.5: @G.%.loc6_31.9 (%.537) = splice_inst %tuple.elem0 [template = %.loc6_3.5 (constants.%.5d6)]
// CHECK:STDOUT:   %.loc6_3.6: <instruction> = call_action (constants.%SelfDestruct, %arr.var), true [template]
// CHECK:STDOUT:   %.loc6_3.7: type = type_of_inst %.loc6_3.6 [template = %.loc6_3.7 (constants.%.8e2)]
// CHECK:STDOUT:   %.loc6_3.8: @G.%.loc6_3.7 (%.8e2) = splice_inst %.loc6_3.6 [template = %.loc6_3.8 (constants.%.9bf)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %arr.var: ref @G.%.loc6_31.9 (%.537) = var_storage %arr.var_patt.loc6_3.1
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8115.1]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc8af.1]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %.loc6_43: %tuple.type.37f = tuple_literal (%int_1, %int_2, %int_3) [concrete = constants.%tuple.2d5]
// CHECK:STDOUT:     %.loc6_3.1: @G.%.loc6_31.9 (%.537) = splice_inst %tuple.elem0 [template = %.loc6_3.5 (constants.%.5d6)]
// CHECK:STDOUT:     assign %arr.var, %.loc6_3.1
// CHECK:STDOUT:     %.loc6_31.1: type = splice_block %.loc6_31.4 [template = %.loc6_31.9 (constants.%.537)] {
// CHECK:STDOUT:       %i32.loc6: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %N.ref: %i32 = name_ref N, %N.loc4_16.2 [template = %N.loc4_16.1 (constants.%N.c34)]
// CHECK:STDOUT:       %.loc6_30.1: Core.IntLiteral = splice_inst %.loc6_30.2 [template = %.loc6_30.3 (constants.%.da2)]
// CHECK:STDOUT:       %.loc6_31.2: type = type_of_inst %.loc6_31.5 [template = %.loc6_31.6 (constants.%.d25)]
// CHECK:STDOUT:       %.loc6_31.3: @G.%.loc6_31.6 (%.d25) = splice_inst %.loc6_31.5 [template = %.loc6_31.7 (constants.%.24e)]
// CHECK:STDOUT:       %.loc6_31.4: type = splice_inst %.loc6_31.8 [template = %.loc6_31.9 (constants.%.537)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %arr: ref @G.%.loc6_31.9 (%.537) = wrapper_binding arr, %arr.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %arr.patt.loc6_17.1: @G.%pattern_type (%pattern_type.fcb) = ref_binding_pattern arr [template = %arr.patt.loc6_17.2 (constants.%arr.patt.c8d)]
// CHECK:STDOUT:       %arr.var_patt.loc6_3.1: @G.%pattern_type (%pattern_type.fcb) = var_pattern %arr.patt.loc6_17.1 [template = %arr.var_patt.loc6_3.2 (constants.%arr.var_patt.1e0)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc6_3.2: type = type_of_inst %.loc6_3.6 [template = %.loc6_3.7 (constants.%.8e2)]
// CHECK:STDOUT:     %.loc6_3.3: @G.%.loc6_3.7 (%.8e2) = splice_inst %.loc6_3.6 [template = %.loc6_3.8 (constants.%.9bf)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc6_3.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc6_3.2(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc6_3.3(%self.param: ref %array_type.afd) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc6_3.4(%self.param: ref %Array.95c1f3.1) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%N.c34) {
// CHECK:STDOUT:   %N.patt.loc4_16.2 => constants.%N.patt.d64
// CHECK:STDOUT:   %N.loc4_16.1 => constants.%N.c34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%int_3.9f3) {
// CHECK:STDOUT:   %N.patt.loc4_16.2 => constants.%N.patt.d64
// CHECK:STDOUT:   %N.loc4_16.1 => constants.%int_3.9f3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc6_30.2 => constants.%inst.splice_block.426619.1
// CHECK:STDOUT:   %.loc6_30.3 => constants.%int_3.1ba
// CHECK:STDOUT:   %.loc6_31.5 => constants.%inst.class_type
// CHECK:STDOUT:   %.loc6_31.6 => type
// CHECK:STDOUT:   %.loc6_31.7 => constants.%Array.95c1f3.1
// CHECK:STDOUT:   %.loc6_31.8 => constants.%inst.splice_block.426619.2
// CHECK:STDOUT:   %.loc6_31.9 => constants.%Array.95c1f3.1
// CHECK:STDOUT:   %require_complete => constants.%complete_type.4fe
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.d4e
// CHECK:STDOUT:   %arr.patt.loc6_17.2 => constants.%arr.patt.403
// CHECK:STDOUT:   %arr.var_patt.loc6_3.2 => constants.%arr.var_patt.074
// CHECK:STDOUT:   %.loc6_3.4 => constants.%tuple.242
// CHECK:STDOUT:   %tuple.elem0 => constants.%inst.splice_block.426619.3
// CHECK:STDOUT:   %.loc6_3.5 => constants.%array
// CHECK:STDOUT:   %.loc6_3.6 => constants.%inst.splice_block.426619.4
// CHECK:STDOUT:   %.loc6_3.7 => constants.%empty_tuple.type
// CHECK:STDOUT:   %.loc6_3.8 => <not constant>
// CHECK:STDOUT: }
// CHECK:STDOUT:

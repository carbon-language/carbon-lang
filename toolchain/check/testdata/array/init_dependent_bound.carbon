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
  // CHECK:STDERR: fail_init_dependent_bound.carbon:[[@LINE+4]]:35: error: cannot initialize array with dependent bound from a list of initializers [ArrayInitDependentBound]
  // CHECK:STDERR:   var unused arr: array(i32, N) = (1, 2, 3);
  // CHECK:STDERR:                                   ^~~~~~~~~
  // CHECK:STDERR:
  var unused arr: array(i32, N) = (1, 2, 3);
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
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %Array.c64: type = class_type @Array, @Array(%T, %int_0) [symbolic]
// CHECK:STDOUT:   %require_complete.d78: <witness> = require_complete_type %Array.c64 [symbolic]
// CHECK:STDOUT:   %pattern_type.630: type = pattern_type %Array.c64 [symbolic]
// CHECK:STDOUT:   %arr.patt.504: %pattern_type.630 = ref_binding_pattern arr [symbolic]
// CHECK:STDOUT:   %arr.var_patt.567: %pattern_type.630 = var_pattern %arr.patt.504 [symbolic]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %array.343: %Array.c64 = tuple_value () [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.0d8: <witness> = lookup_impl_witness %Array.c64, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.bcc: %Destroy.type = facet_value %Array.c64, (%Destroy.lookup_impl_witness.0d8) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.887: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Destroy.facet.bcc) [symbolic]
// CHECK:STDOUT:   %.4c0: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.887, %Destroy.facet.bcc [symbolic]
// CHECK:STDOUT:   %impl.elem2: %.4c0 = impl_witness_access %Destroy.lookup_impl_witness.0d8, element2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.e29: <specific function> = specific_impl_function %impl.elem2, @Destroy.WithSelf.SelfDestruct.1(%Destroy.facet.bcc) [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %Array.909: type = class_type @Array, @Array(%C, %int_0) [concrete]
// CHECK:STDOUT:   %array_type.988: type = array_type %int_0, %C [concrete]
// CHECK:STDOUT:   %complete_type.e80: <witness> = complete_type_witness %array_type.988 [concrete]
// CHECK:STDOUT:   %pattern_type.2b0: type = pattern_type %Array.909 [concrete]
// CHECK:STDOUT:   %arr.patt.84c: %pattern_type.2b0 = ref_binding_pattern arr [concrete]
// CHECK:STDOUT:   %arr.var_patt.3a1: %pattern_type.2b0 = var_pattern %arr.patt.84c [concrete]
// CHECK:STDOUT:   %array.276: %Array.909 = tuple_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.bff: %pattern_type.2b0 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.d07: %pattern_type.2b0 = wrapper_binding_pattern self, %self.param_patt.bff [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfc: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc7 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01: %Destroy.WithSelf.SubobjectDestroy.type.dfc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef0: type = fn_type @Destroy.WithSelf.Op.loc7 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403: %Destroy.WithSelf.Op.type.ef0 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbc: type = fn_type @Destroy.WithSelf.SelfDestruct.loc7 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3: %Destroy.WithSelf.SelfDestruct.type.fbc = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f8f: <witness> = custom_witness (%Destroy.WithSelf.Op.403, %Destroy.WithSelf.SubobjectDestroy.d01, %Destroy.WithSelf.SelfDestruct.db3), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.70f: %Destroy.type = facet_value %Array.909, (%custom_witness.f8f) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.40d: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Destroy.facet.70f) [concrete]
// CHECK:STDOUT:   %.221: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.40d, %Destroy.facet.70f [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl: %Destroy.WithSelf.SubobjectDestroy.type.dfc = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc7 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.2b0 = ref_param_pattern [concrete = constants.%self.param_patt.bff]
// CHECK:STDOUT:     %self.patt: %pattern_type.2b0 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.d07]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Array.909 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Array.909 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl: %Destroy.WithSelf.Op.type.ef0 = fn_decl @Destroy.WithSelf.Op.loc7 [concrete = constants.%Destroy.WithSelf.Op.403] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.2b0 = ref_param_pattern [concrete = constants.%self.param_patt.bff]
// CHECK:STDOUT:     %self.patt: %pattern_type.2b0 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.d07]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Array.909 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Array.909 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%T.loc4_15.2: type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Array.loc7_29.2: type = class_type @Array, @Array(%T.loc4_15.1, constants.%int_0) [symbolic = %Array.loc7_29.2 (constants.%Array.c64)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Array.loc7_29.2 [symbolic = %require_complete (constants.%require_complete.d78)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Array.loc7_29.2 [symbolic = %pattern_type (constants.%pattern_type.630)]
// CHECK:STDOUT:   %arr.patt.loc7_17.2: @G.%pattern_type (%pattern_type.630) = ref_binding_pattern arr [symbolic = %arr.patt.loc7_17.2 (constants.%arr.patt.504)]
// CHECK:STDOUT:   %arr.var_patt.loc7_3.2: @G.%pattern_type (%pattern_type.630) = var_pattern %arr.patt.loc7_17.2 [symbolic = %arr.var_patt.loc7_3.2 (constants.%arr.var_patt.567)]
// CHECK:STDOUT:   %array: @G.%Array.loc7_29.2 (%Array.c64) = tuple_value () [symbolic = %array (constants.%array.343)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %Array.loc7_29.2, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.0d8)]
// CHECK:STDOUT:   %Destroy.facet.loc7_3.3: %Destroy.type = facet_value %Array.loc7_29.2, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet.loc7_3.3 (constants.%Destroy.facet.bcc)]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Destroy.facet.loc7_3.3) [symbolic = %Destroy.WithSelf.SelfDestruct.type (constants.%Destroy.WithSelf.SelfDestruct.type.887)]
// CHECK:STDOUT:   %.loc7_3.4: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type, %Destroy.facet.loc7_3.3 [symbolic = %.loc7_3.4 (constants.%.4c0)]
// CHECK:STDOUT:   %impl.elem2.loc7_3.2: @G.%.loc7_3.4 (%.4c0) = impl_witness_access %Destroy.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc7_3.2 (constants.%impl.elem2)]
// CHECK:STDOUT:   %specific_impl_fn.loc7_3.2: <specific function> = specific_impl_function %impl.elem2.loc7_3.2, @Destroy.WithSelf.SelfDestruct.1(%Destroy.facet.loc7_3.3) [symbolic = %specific_impl_fn.loc7_3.2 (constants.%specific_impl_fn.e29)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %arr.var: ref @G.%Array.loc7_29.2 (%Array.c64) = var_storage %arr.var_patt.loc7_3.1
// CHECK:STDOUT:     %.loc7_34.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc7_34.2: init @G.%Array.loc7_29.2 (%Array.c64) to %arr.var = array_init () [symbolic = %array (constants.%array.343)]
// CHECK:STDOUT:     %.loc7_3.1: init @G.%Array.loc7_29.2 (%Array.c64) = converted %.loc7_34.1, %.loc7_34.2 [symbolic = %array (constants.%array.343)]
// CHECK:STDOUT:     assign %arr.var, %.loc7_3.1
// CHECK:STDOUT:     %.loc7_29: type = splice_block %Array.loc7_29.1 [symbolic = %Array.loc7_29.2 (constants.%Array.c64)] {
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc4_15.2 [symbolic = %T.loc4_15.1 (constants.%T)]
// CHECK:STDOUT:       %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:       %Array.loc7_29.1: type = class_type @Array, @Array(constants.%T, constants.%int_0) [symbolic = %Array.loc7_29.2 (constants.%Array.c64)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %arr: ref @G.%Array.loc7_29.2 (%Array.c64) = wrapper_binding arr, %arr.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %arr.patt.loc7_17.1: @G.%pattern_type (%pattern_type.630) = ref_binding_pattern arr [symbolic = %arr.patt.loc7_17.2 (constants.%arr.patt.504)]
// CHECK:STDOUT:       %arr.var_patt.loc7_3.1: @G.%pattern_type (%pattern_type.630) = var_pattern %arr.patt.loc7_17.1 [symbolic = %arr.var_patt.loc7_3.2 (constants.%arr.var_patt.567)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %impl.elem2.loc7_3.1: @G.%.loc7_3.4 (%.4c0) = impl_witness_access constants.%Destroy.lookup_impl_witness.0d8, element2 [symbolic = %impl.elem2.loc7_3.2 (constants.%impl.elem2)]
// CHECK:STDOUT:     %bound_method.loc7_3.1: <bound method> = bound_method %arr.var, %impl.elem2.loc7_3.1
// CHECK:STDOUT:     %Destroy.facet.loc7_3.1: %Destroy.type = facet_value constants.%Array.c64, (constants.%Destroy.lookup_impl_witness.0d8) [symbolic = %Destroy.facet.loc7_3.3 (constants.%Destroy.facet.bcc)]
// CHECK:STDOUT:     %.loc7_3.2: %Destroy.type = converted constants.%Array.c64, %Destroy.facet.loc7_3.1 [symbolic = %Destroy.facet.loc7_3.3 (constants.%Destroy.facet.bcc)]
// CHECK:STDOUT:     %Destroy.facet.loc7_3.2: %Destroy.type = facet_value constants.%Array.c64, (constants.%Destroy.lookup_impl_witness.0d8) [symbolic = %Destroy.facet.loc7_3.3 (constants.%Destroy.facet.bcc)]
// CHECK:STDOUT:     %.loc7_3.3: %Destroy.type = converted constants.%Array.c64, %Destroy.facet.loc7_3.2 [symbolic = %Destroy.facet.loc7_3.3 (constants.%Destroy.facet.bcc)]
// CHECK:STDOUT:     %specific_impl_fn.loc7_3.1: <specific function> = specific_impl_function %impl.elem2.loc7_3.1, @Destroy.WithSelf.SelfDestruct.1(constants.%Destroy.facet.bcc) [symbolic = %specific_impl_fn.loc7_3.2 (constants.%specific_impl_fn.e29)]
// CHECK:STDOUT:     %bound_method.loc7_3.2: <bound method> = bound_method %arr.var, %specific_impl_fn.loc7_3.1
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %bound_method.loc7_3.2(%arr.var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc7(%self.param: ref %Array.909) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc7(%self.param: ref %Array.909) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc7(%self.param: ref %Array.909) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
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
// CHECK:STDOUT:   %Array.loc7_29.2 => constants.%Array.909
// CHECK:STDOUT:   %require_complete => constants.%complete_type.e80
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2b0
// CHECK:STDOUT:   %arr.patt.loc7_17.2 => constants.%arr.patt.84c
// CHECK:STDOUT:   %arr.var_patt.loc7_3.2 => constants.%arr.var_patt.3a1
// CHECK:STDOUT:   %array => constants.%array.276
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.f8f
// CHECK:STDOUT:   %Destroy.facet.loc7_3.3 => constants.%Destroy.facet.70f
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type => constants.%Destroy.WithSelf.SelfDestruct.type.40d
// CHECK:STDOUT:   %.loc7_3.4 => constants.%.221
// CHECK:STDOUT:   %impl.elem2.loc7_3.2 => constants.%Destroy.WithSelf.SelfDestruct.db3
// CHECK:STDOUT:   %specific_impl_fn.loc7_3.2 => constants.%Destroy.WithSelf.SelfDestruct.db3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- init_template_dependent_bound.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.9d9: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %N.patt.b2d: %pattern_type.9d9 = symbolic_binding_pattern N, 0, template [template]
// CHECK:STDOUT:   %N.5d3: %i32 = symbolic_binding N, 0, template [template]
// CHECK:STDOUT:   %.da2: Core.IntLiteral = splice_inst @G.%.loc6_30.2 [template]
// CHECK:STDOUT:   %Array.type: type = generic_class_type @Array [concrete]
// CHECK:STDOUT:   %Array.generic: %Array.type = struct_value () [concrete]
// CHECK:STDOUT:   %.4b2: type = type_of_inst @G.%.loc6_31.5 [template]
// CHECK:STDOUT:   %.787: %.4b2 = splice_inst @G.%.loc6_31.5 [template]
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
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Self.0e7: %Destroy.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.869: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Self.0e7) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.c97: %Destroy.WithSelf.SelfDestruct.type.869 = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.assoc_type: type = assoc_entity_type @Destroy [concrete]
// CHECK:STDOUT:   %assoc2: %Destroy.assoc_type = assoc_entity element2, imports.%Core.import_ref.df0 [concrete]
// CHECK:STDOUT:   %.8cb: type = type_of_inst @G.%.loc6_3.8 [template]
// CHECK:STDOUT:   %.c23: %.8cb = splice_inst @G.%.loc6_3.8 [template]
// CHECK:STDOUT:   %.57c: type = type_of_inst @G.%.loc6_3.11 [template]
// CHECK:STDOUT:   %.e63: %.57c = splice_inst @G.%.loc6_3.11 [template]
// CHECK:STDOUT:   %From: Core.IntLiteral = symbolic_binding From, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.5be: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.500: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%From) [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bfc: %Int.as.ImplicitAs.impl.Convert.type.500 = struct_value () [symbolic]
// CHECK:STDOUT:   %int_3.af5: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.803: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.6ce, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.de1: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.76f: %Int.as.ImplicitAs.impl.Convert.type.de1 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.eb0: %ImplicitAs.type.5be = facet_value %i32, (%ImplicitAs.impl_witness.803) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.797: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet.eb0) [concrete]
// CHECK:STDOUT:   %.917: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.797, %ImplicitAs.facet.eb0 [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_3.af5, %Int.as.ImplicitAs.impl.Convert.76f [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Int.as.ImplicitAs.impl.Convert.76f, @Int.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.343: <bound method> = bound_method %int_3.af5, %Int.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %inst.splice_block.426619.1: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.03d: Core.IntLiteral = splice_block %.20e [concrete = %int_3.1ba] {
// CHECK:STDOUT:       %impl.elem0.0f8: %.917 = impl_witness_access %ImplicitAs.impl_witness.803, element0 [concrete = %Int.as.ImplicitAs.impl.Convert.76f]
// CHECK:STDOUT:       %bound_method.6ee: <bound method> = bound_method %int_3.af5, %impl.elem0.0f8 [concrete = %Int.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:       %specific_fn.c27: <specific function> = specific_function %impl.elem0.0f8, @Int.as.ImplicitAs.impl.Convert(%int_32) [concrete = %Int.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:       %bound_method.4b9: <bound method> = bound_method %int_3.af5, %specific_fn.c27 [concrete = %bound_method.343]
// CHECK:STDOUT:       %Int.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.4b9(%int_3.af5) [concrete = %int_3.1ba]
// CHECK:STDOUT:       %.20e: Core.IntLiteral = value_of_initializer %Int.as.ImplicitAs.impl.Convert.call [concrete = %int_3.1ba]
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Array.3c49d1.1: type = class_type @Array, @Array(%i32, %int_3.1ba) [concrete]
// CHECK:STDOUT:   %inst.class_type: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %Array.3c49d1.2: type = class_type @Array, @Array(%i32, %int_3.1ba) [concrete = %Array.3c49d1.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.426619.2: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.23d: type = splice_block %Array.3c49d1.1 [concrete = %Array.3c49d1.1] {}
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %array_type.4db: type = array_type %int_3.1ba, %i32 [concrete]
// CHECK:STDOUT:   %complete_type.ea1: <witness> = complete_type_witness %array_type.4db [concrete]
// CHECK:STDOUT:   %pattern_type.ce9: type = pattern_type %Array.3c49d1.1 [concrete]
// CHECK:STDOUT:   %arr.patt.d62: %pattern_type.ce9 = ref_binding_pattern arr [concrete]
// CHECK:STDOUT:   %arr.var_patt.1b8: %pattern_type.ce9 = var_pattern %arr.patt.d62 [concrete]
// CHECK:STDOUT:   %int_0.5c61a1.1: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %int_1.8e8: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.cb7: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %array: %Array.3c49d1.1 = tuple_value (%int_1.8e8, %int_2.cb7, %int_3.af5) [concrete]
// CHECK:STDOUT:   %inst.splice_block.426619.3: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.a79: init %Array.3c49d1.1 = splice_block %.317 [concrete = %array] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %.18a44e.2: ref %Array.3c49d1.1 = specific_inst @G.%arr.var, @G(%int_3.af5)
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %tuple.242: %tuple.type.f7a = tuple_value (%inst.splice_block.426619.3) [concrete]
// CHECK:STDOUT:   %self.param_patt.6c0: %pattern_type.9d9 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.1ea: %pattern_type.9d9 = wrapper_binding_pattern self, %self.param_patt.6c0 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc6_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.1: type = fn_type @Destroy.WithSelf.Op.loc6_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.1: %Destroy.WithSelf.Op.type.ef016f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.1: type = fn_type @Destroy.WithSelf.SelfDestruct.loc6_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.1: %Destroy.WithSelf.SelfDestruct.type.fbceb5.1 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.258: %pattern_type.ce9 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.979: %pattern_type.ce9 = wrapper_binding_pattern self, %self.param_patt.258 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc6_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc6_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.2: type = fn_type @Destroy.WithSelf.SelfDestruct.loc6_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.2: %Destroy.WithSelf.SelfDestruct.type.fbceb5.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f8f19d.2: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.2, %Destroy.WithSelf.SubobjectDestroy.d01daf.2, %Destroy.WithSelf.SelfDestruct.db3fdb.2), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.b76: %Destroy.type = facet_value %Array.3c49d1.1, (%custom_witness.f8f19d.2) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.ab4: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Destroy.facet.b76) [concrete]
// CHECK:STDOUT:   %.b7c: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.ab4, %Destroy.facet.b76 [concrete]
// CHECK:STDOUT:   %inst.splice_block.426619.4: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.c8c: <bound method> = splice_block %bound_method.a27 {
// CHECK:STDOUT:       %.18a44e.1: ref %Array.3c49d1.1 = specific_inst @G.%arr.var, @G(%int_3.af5)
// CHECK:STDOUT:       %impl.elem2: %.b7c = impl_witness_access %custom_witness.f8f19d.2, element2 [concrete = %Destroy.WithSelf.SelfDestruct.db3fdb.2]
// CHECK:STDOUT:       %bound_method.a27: <bound method> = bound_method %.18a44e.1, %impl.elem2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.call: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %.c8c(%.18a44e.1)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.df0: @Destroy.WithSelf.%Destroy.WithSelf.SelfDestruct.type (%Destroy.WithSelf.SelfDestruct.type.869) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @Destroy.WithSelf.%Destroy.WithSelf.SelfDestruct (constants.%Destroy.WithSelf.SelfDestruct.c97)]
// CHECK:STDOUT:   %Core.import_ref.5b7: @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert.type (%Int.as.ImplicitAs.impl.Convert.type.500) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert (constants.%Int.as.ImplicitAs.impl.Convert.bfc)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.6ce = impl_witness_table (%Core.import_ref.5b7), @Int.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc6_3.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.9d9 = ref_param_pattern [concrete = constants.%self.param_patt.6c0]
// CHECK:STDOUT:     %self.patt: %pattern_type.9d9 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.1ea]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i32 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i32 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.1: %Destroy.WithSelf.Op.type.ef016f.1 = fn_decl @Destroy.WithSelf.Op.loc6_3.1 [concrete = constants.%Destroy.WithSelf.Op.403171.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.9d9 = ref_param_pattern [concrete = constants.%self.param_patt.6c0]
// CHECK:STDOUT:     %self.patt: %pattern_type.9d9 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.1ea]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i32 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i32 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc6_3.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.ce9 = ref_param_pattern [concrete = constants.%self.param_patt.258]
// CHECK:STDOUT:     %self.patt: %pattern_type.ce9 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.979]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Array.3c49d1.1 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Array.3c49d1.1 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.2 = fn_decl @Destroy.WithSelf.Op.loc6_3.2 [concrete = constants.%Destroy.WithSelf.Op.403171.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.ce9 = ref_param_pattern [concrete = constants.%self.param_patt.258]
// CHECK:STDOUT:     %self.patt: %pattern_type.ce9 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.979]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Array.3c49d1.1 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Array.3c49d1.1 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%N.loc4_16.2: %i32) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc6_30.2: <instruction> = convert_to_value_action %N.ref, Core.IntLiteral [template]
// CHECK:STDOUT:   %.loc6_30.3: Core.IntLiteral = splice_inst %.loc6_30.2 [template = %.loc6_30.3 (constants.%.da2)]
// CHECK:STDOUT:   %.loc6_31.5: <instruction> = call_action (constants.%Array.generic, %i32.loc6, %.loc6_30.1), false [template]
// CHECK:STDOUT:   %.loc6_31.6: type = type_of_inst %.loc6_31.5 [template = %.loc6_31.6 (constants.%.4b2)]
// CHECK:STDOUT:   %.loc6_31.7: @G.%.loc6_31.6 (%.4b2) = splice_inst %.loc6_31.5 [template = %.loc6_31.7 (constants.%.787)]
// CHECK:STDOUT:   %.loc6_31.8: <instruction> = convert_to_value_action %.loc6_31.3, type [template]
// CHECK:STDOUT:   %.loc6_31.9: type = splice_inst %.loc6_31.8 [template = %.loc6_31.9 (constants.%.537)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %.loc6_31.9 [template = %require_complete (constants.%require_complete.cb0)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %.loc6_31.9 [template = %pattern_type (constants.%pattern_type.fcb)]
// CHECK:STDOUT:   %arr.patt.loc6_17.2: @G.%pattern_type (%pattern_type.fcb) = ref_binding_pattern arr [template = %arr.patt.loc6_17.2 (constants.%arr.patt.c8d)]
// CHECK:STDOUT:   %arr.var_patt.loc6_3.2: @G.%pattern_type (%pattern_type.fcb) = var_pattern %arr.patt.loc6_17.2 [template = %arr.var_patt.loc6_3.2 (constants.%arr.var_patt.1e0)]
// CHECK:STDOUT:   %.loc6_3.6: %tuple.type.f7a = initialize_action %.loc6_43, %.loc6_31.9, %arr.var, none, false [template]
// CHECK:STDOUT:   %tuple.elem0: <instruction> = tuple_access %.loc6_3.6, element0 [template]
// CHECK:STDOUT:   %.loc6_3.7: @G.%.loc6_31.9 (%.537) = splice_inst %tuple.elem0 [template = %.loc6_3.7 (constants.%.5d6)]
// CHECK:STDOUT:   %.loc6_3.8: <instruction> = compound_member_access_action %arr.var, constants.%assoc2 [template]
// CHECK:STDOUT:   %.loc6_3.9: type = type_of_inst %.loc6_3.8 [template = %.loc6_3.9 (constants.%.8cb)]
// CHECK:STDOUT:   %.loc6_3.10: @G.%.loc6_3.9 (%.8cb) = splice_inst %.loc6_3.8 [template = %.loc6_3.10 (constants.%.c23)]
// CHECK:STDOUT:   %.loc6_3.11: <instruction> = call_action (%.loc6_3.3), true [template]
// CHECK:STDOUT:   %.loc6_3.12: type = type_of_inst %.loc6_3.11 [template = %.loc6_3.12 (constants.%.57c)]
// CHECK:STDOUT:   %.loc6_3.13: @G.%.loc6_3.12 (%.57c) = splice_inst %.loc6_3.11 [template = %.loc6_3.13 (constants.%.e63)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %arr.var: ref @G.%.loc6_31.9 (%.537) = var_storage %arr.var_patt.loc6_3.1
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8115.1]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc8af.1]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %.loc6_43: %tuple.type.37f = tuple_literal (%int_1, %int_2, %int_3) [concrete = constants.%tuple.2d5]
// CHECK:STDOUT:     %.loc6_3.1: @G.%.loc6_31.9 (%.537) = splice_inst %tuple.elem0 [template = %.loc6_3.7 (constants.%.5d6)]
// CHECK:STDOUT:     assign %arr.var, %.loc6_3.1
// CHECK:STDOUT:     %.loc6_31.1: type = splice_block %.loc6_31.4 [template = %.loc6_31.9 (constants.%.537)] {
// CHECK:STDOUT:       %i32.loc6: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %N.ref: %i32 = name_ref N, %N.loc4_16.2 [template = %N.loc4_16.1 (constants.%N.5d3)]
// CHECK:STDOUT:       %.loc6_30.1: Core.IntLiteral = splice_inst %.loc6_30.2 [template = %.loc6_30.3 (constants.%.da2)]
// CHECK:STDOUT:       %.loc6_31.2: type = type_of_inst %.loc6_31.5 [template = %.loc6_31.6 (constants.%.4b2)]
// CHECK:STDOUT:       %.loc6_31.3: @G.%.loc6_31.6 (%.4b2) = splice_inst %.loc6_31.5 [template = %.loc6_31.7 (constants.%.787)]
// CHECK:STDOUT:       %.loc6_31.4: type = splice_inst %.loc6_31.8 [template = %.loc6_31.9 (constants.%.537)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %arr: ref @G.%.loc6_31.9 (%.537) = wrapper_binding arr, %arr.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %arr.patt.loc6_17.1: @G.%pattern_type (%pattern_type.fcb) = ref_binding_pattern arr [template = %arr.patt.loc6_17.2 (constants.%arr.patt.c8d)]
// CHECK:STDOUT:       %arr.var_patt.loc6_3.1: @G.%pattern_type (%pattern_type.fcb) = var_pattern %arr.patt.loc6_17.1 [template = %arr.var_patt.loc6_3.2 (constants.%arr.var_patt.1e0)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc6_3.2: type = type_of_inst %.loc6_3.8 [template = %.loc6_3.9 (constants.%.8cb)]
// CHECK:STDOUT:     %.loc6_3.3: @G.%.loc6_3.9 (%.8cb) = splice_inst %.loc6_3.8 [template = %.loc6_3.10 (constants.%.c23)]
// CHECK:STDOUT:     %.loc6_3.4: type = type_of_inst %.loc6_3.11 [template = %.loc6_3.12 (constants.%.57c)]
// CHECK:STDOUT:     %.loc6_3.5: @G.%.loc6_3.12 (%.57c) = splice_inst %.loc6_3.11 [template = %.loc6_3.13 (constants.%.e63)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc6_3.1(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc6_3.1(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc6_3.1(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.1(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.1(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc6_3.2(%self.param: ref %Array.3c49d1.1) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc8af.1]
// CHECK:STDOUT:   %.loc6_3.1: ref %i32 = array_index %self.param, %int_2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc6_3.1: <bound method> = bound_method %.loc6_3.1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc6_3.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc6_3.1(%.loc6_3.1)
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8115.1]
// CHECK:STDOUT:   %.loc6_3.2: ref %i32 = array_index %self.param, %int_1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc6_3.2: <bound method> = bound_method %.loc6_3.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc6_3.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc6_3.2(%.loc6_3.2)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c61a1.1]
// CHECK:STDOUT:   %.loc6_3.3: ref %i32 = array_index %self.param, %int_0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc6_3.3: <bound method> = bound_method %.loc6_3.3, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc6_3.3: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc6_3.3(%.loc6_3.3)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc6_3.2(%self.param: ref %Array.3c49d1.1) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc6_3.2(%self.param: ref %Array.3c49d1.1) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.2(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%N.5d3) {
// CHECK:STDOUT:   %N.patt.loc4_16.2 => constants.%N.patt.b2d
// CHECK:STDOUT:   %N.loc4_16.1 => constants.%N.5d3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%int_3.af5) {
// CHECK:STDOUT:   %N.patt.loc4_16.2 => constants.%N.patt.b2d
// CHECK:STDOUT:   %N.loc4_16.1 => constants.%int_3.af5
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc6_30.2 => constants.%inst.splice_block.426619.1
// CHECK:STDOUT:   %.loc6_30.3 => constants.%int_3.1ba
// CHECK:STDOUT:   %.loc6_31.5 => constants.%inst.class_type
// CHECK:STDOUT:   %.loc6_31.6 => type
// CHECK:STDOUT:   %.loc6_31.7 => constants.%Array.3c49d1.1
// CHECK:STDOUT:   %.loc6_31.8 => constants.%inst.splice_block.426619.2
// CHECK:STDOUT:   %.loc6_31.9 => constants.%Array.3c49d1.1
// CHECK:STDOUT:   %require_complete => constants.%complete_type.ea1
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.ce9
// CHECK:STDOUT:   %arr.patt.loc6_17.2 => constants.%arr.patt.d62
// CHECK:STDOUT:   %arr.var_patt.loc6_3.2 => constants.%arr.var_patt.1b8
// CHECK:STDOUT:   %.loc6_3.6 => constants.%tuple.242
// CHECK:STDOUT:   %tuple.elem0 => constants.%inst.splice_block.426619.3
// CHECK:STDOUT:   %.loc6_3.7 => constants.%array
// CHECK:STDOUT:   %.loc6_3.8 => constants.%inst.splice_block.426619.4
// CHECK:STDOUT:   %.loc6_3.9 => <bound method>
// CHECK:STDOUT:   %.loc6_3.10 => <not constant>
// CHECK:STDOUT:   %.loc6_3.11 => constants.%inst.call
// CHECK:STDOUT:   %.loc6_3.12 => constants.%empty_tuple.type
// CHECK:STDOUT:   %.loc6_3.13 => <not constant>
// CHECK:STDOUT: }
// CHECK:STDOUT:

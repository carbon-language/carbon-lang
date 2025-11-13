// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/lookup/impl_forall_addr.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/lookup/impl_forall_addr.carbon

// --- impl_forall.carbon
library "[[@TEST_NAME]]";

class A(T:! type) {
  var n: T;
}

interface I(U:! type) {
  fn F[addr self: Self*]() -> U*;
}

//@dump-sem-ir-begin
impl forall [V:! type] A(V) as I(V) {
  fn F[addr self: Self*]() -> V* {
    return &self->n;
  }
}
//@dump-sem-ir-end

fn TestGeneric[W:! type](a: A(W)*) -> W* {
  //@dump-sem-ir-begin
  return a->(I(W).F)();
  //@dump-sem-ir-end
}

fn TestSpecific(a: A({})*) -> {}* {
  //@dump-sem-ir-begin
  return a->(I({}).F)();
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- impl_forall.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T.d9f: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %A.type: type = generic_class_type @A [concrete]
// CHECK:STDOUT:   %A.generic: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.dac: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.dac = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %V.d9f: type = symbolic_binding V, 0 [symbolic]
// CHECK:STDOUT:   %A.2ddf89.2: type = class_type @A, @A(%V.d9f) [symbolic]
// CHECK:STDOUT:   %I.type.07036d.2: type = facet_type <@I, @I(%V.d9f)> [symbolic]
// CHECK:STDOUT:   %require_complete.c94ab4.1: <witness> = require_complete_type %I.type.07036d.2 [symbolic]
// CHECK:STDOUT:   %I.impl_witness.0c179e.1: <witness> = impl_witness file.%I.impl_witness_table, @A.as.I.impl(%V.d9f) [symbolic]
// CHECK:STDOUT:   %ptr.6f46b6.1: type = ptr_type %A.2ddf89.2 [symbolic]
// CHECK:STDOUT:   %pattern_type.2180b5.1: type = pattern_type %ptr.6f46b6.1 [symbolic]
// CHECK:STDOUT:   %ptr.4f0b5c.2: type = ptr_type %V.d9f [symbolic]
// CHECK:STDOUT:   %pattern_type.a603a0.2: type = pattern_type %ptr.4f0b5c.2 [symbolic]
// CHECK:STDOUT:   %A.as.I.impl.F.type.2974bf.1: type = fn_type @A.as.I.impl.F, @A.as.I.impl(%V.d9f) [symbolic]
// CHECK:STDOUT:   %A.as.I.impl.F.786476.1: %A.as.I.impl.F.type.2974bf.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.482d3f.1: <witness> = require_complete_type %ptr.4f0b5c.2 [symbolic]
// CHECK:STDOUT:   %require_complete.743ecf.1: <witness> = require_complete_type %ptr.6f46b6.1 [symbolic]
// CHECK:STDOUT:   %A.elem.ee5a68.2: type = unbound_element_type %A.2ddf89.2, %V.d9f [symbolic]
// CHECK:STDOUT:   %require_complete.f6b5ff.1: <witness> = require_complete_type %A.2ddf89.2 [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.75bcbe.1: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.d9f) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.692ac2.1: %ptr.as.Copy.impl.Op.type.75bcbe.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.c66: <witness> = lookup_impl_witness %ptr.4f0b5c.2, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.facet.747: %Copy.type = facet_value %ptr.4f0b5c.2, (%Copy.lookup_impl_witness.c66) [symbolic]
// CHECK:STDOUT:   %.89d: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.747 [symbolic]
// CHECK:STDOUT:   %impl.elem0.928: %.89d = impl_witness_access %Copy.lookup_impl_witness.c66, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.6a1: <specific function> = specific_impl_function %impl.elem0.928, @Copy.Op(%Copy.facet.747) [symbolic]
// CHECK:STDOUT:   %W: type = symbolic_binding W, 0 [symbolic]
// CHECK:STDOUT:   %A.2ddf89.3: type = class_type @A, @A(%W) [symbolic]
// CHECK:STDOUT:   %ptr.6f46b6.2: type = ptr_type %A.2ddf89.3 [symbolic]
// CHECK:STDOUT:   %pattern_type.2180b5.2: type = pattern_type %ptr.6f46b6.2 [symbolic]
// CHECK:STDOUT:   %ptr.4f0b5c.4: type = ptr_type %W [symbolic]
// CHECK:STDOUT:   %pattern_type.a603a0.4: type = pattern_type %ptr.4f0b5c.4 [symbolic]
// CHECK:STDOUT:   %I.type.07036d.3: type = facet_type <@I, @I(%W)> [symbolic]
// CHECK:STDOUT:   %I.F.type.76d346.3: type = fn_type @I.F, @I(%W) [symbolic]
// CHECK:STDOUT:   %I.assoc_type.b650b2.3: type = assoc_entity_type @I, @I(%W) [symbolic]
// CHECK:STDOUT:   %assoc0.b4fda0.3: %I.assoc_type.b650b2.3 = assoc_entity element0, @I.%I.F.decl [symbolic]
// CHECK:STDOUT:   %require_complete.c94ab4.2: <witness> = require_complete_type %I.type.07036d.3 [symbolic]
// CHECK:STDOUT:   %I.impl_witness.0c179e.2: <witness> = impl_witness file.%I.impl_witness_table, @A.as.I.impl(%W) [symbolic]
// CHECK:STDOUT:   %A.as.I.impl.F.type.2974bf.2: type = fn_type @A.as.I.impl.F, @A.as.I.impl(%W) [symbolic]
// CHECK:STDOUT:   %A.as.I.impl.F.786476.2: %A.as.I.impl.F.type.2974bf.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %A.2ddf89.3, @I, @I(%W) [symbolic]
// CHECK:STDOUT:   %I.facet.84d: %I.type.07036d.3 = facet_value %A.2ddf89.3, (%I.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %.e50: type = fn_type_with_self_type %I.F.type.76d346.3, %I.facet.84d [symbolic]
// CHECK:STDOUT:   %impl.elem0.a9e: %.e50 = impl_witness_access %I.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.2e3: <specific function> = specific_impl_function %impl.elem0.a9e, @I.F(%W, %I.facet.84d) [symbolic]
// CHECK:STDOUT:   %require_complete.f6b5ff.2: <witness> = require_complete_type %A.2ddf89.3 [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %A.235: type = class_type @A, @A(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %ptr.197: type = ptr_type %A.235 [concrete]
// CHECK:STDOUT:   %pattern_type.e47: type = pattern_type %ptr.197 [concrete]
// CHECK:STDOUT:   %ptr.c28: type = ptr_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.1cc: type = pattern_type %ptr.c28 [concrete]
// CHECK:STDOUT:   %I.type.399: type = facet_type <@I, @I(%empty_struct_type)> [concrete]
// CHECK:STDOUT:   %I.F.type.684: type = fn_type @I.F, @I(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %I.assoc_type.22c: type = assoc_entity_type @I, @I(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %assoc0.a7f: %I.assoc_type.22c = assoc_entity element0, @I.%I.F.decl [concrete]
// CHECK:STDOUT:   %complete_type.e4b: <witness> = complete_type_witness %I.type.399 [concrete]
// CHECK:STDOUT:   %I.impl_witness.915: <witness> = impl_witness file.%I.impl_witness_table, @A.as.I.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %A.as.I.impl.F.type.184: type = fn_type @A.as.I.impl.F, @A.as.I.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %A.as.I.impl.F.041: %A.as.I.impl.F.type.184 = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet.6ea: %I.type.399 = facet_value %A.235, (%I.impl_witness.915) [concrete]
// CHECK:STDOUT:   %.824: type = fn_type_with_self_type %I.F.type.684, %I.facet.6ea [concrete]
// CHECK:STDOUT:   %A.as.I.impl.F.specific_fn: <specific function> = specific_function %A.as.I.impl.F.041, @A.as.I.impl.F(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %A.elem.2af: type = unbound_element_type %A.235, %empty_struct_type [concrete]
// CHECK:STDOUT:   %struct_type.n.91c: type = struct_type {.n: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %complete_type.0a6: <witness> = complete_type_witness %struct_type.n.91c [concrete]
// CHECK:STDOUT:   %complete_type.38e: <witness> = complete_type_witness %ptr.c28 [concrete]
// CHECK:STDOUT:   %complete_type.d27: <witness> = complete_type_witness %ptr.197 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.433: <witness> = impl_witness imports.%Copy.impl_witness_table.67d, @ptr.as.Copy.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.424: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.15e: %ptr.as.Copy.impl.Op.type.424 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.48c: %Copy.type = facet_value %ptr.c28, (%Copy.impl_witness.433) [concrete]
// CHECK:STDOUT:   %.fb0: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.48c [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.15e, @ptr.as.Copy.impl.Op(%empty_struct_type) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.659: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.75bcbe.1) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.692ac2.1)]
// CHECK:STDOUT:   %Copy.impl_witness_table.67d = impl_witness_table (%Core.import_ref.659), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   impl_decl @A.as.I.impl [concrete] {
// CHECK:STDOUT:     %V.patt: %pattern_type.98f = symbolic_binding_pattern V, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A.ref: %A.type = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:     %V.ref.loc12_26: type = name_ref V, %V.loc12_14.2 [symbolic = %V.loc12_14.1 (constants.%V.d9f)]
// CHECK:STDOUT:     %A.loc12_27.2: type = class_type @A, @A(constants.%V.d9f) [symbolic = %A.loc12_27.1 (constants.%A.2ddf89.2)]
// CHECK:STDOUT:     %I.ref: %I.type.dac = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:     %V.ref.loc12_34: type = name_ref V, %V.loc12_14.2 [symbolic = %V.loc12_14.1 (constants.%V.d9f)]
// CHECK:STDOUT:     %I.type.loc12_35.2: type = facet_type <@I, @I(constants.%V.d9f)> [symbolic = %I.type.loc12_35.1 (constants.%I.type.07036d.2)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %V.loc12_14.2: type = symbolic_binding V, 0 [symbolic = %V.loc12_14.1 (constants.%V.d9f)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (@A.as.I.impl.%A.as.I.impl.F.decl), @A.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table, @A.as.I.impl(constants.%V.d9f) [symbolic = @A.as.I.impl.%I.impl_witness (constants.%I.impl_witness.0c179e.1)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @A.as.I.impl(%V.loc12_14.2: type) {
// CHECK:STDOUT:   %V.loc12_14.1: type = symbolic_binding V, 0 [symbolic = %V.loc12_14.1 (constants.%V.d9f)]
// CHECK:STDOUT:   %A.loc12_27.1: type = class_type @A, @A(%V.loc12_14.1) [symbolic = %A.loc12_27.1 (constants.%A.2ddf89.2)]
// CHECK:STDOUT:   %I.type.loc12_35.1: type = facet_type <@I, @I(%V.loc12_14.1)> [symbolic = %I.type.loc12_35.1 (constants.%I.type.07036d.2)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.type.loc12_35.1 [symbolic = %require_complete (constants.%require_complete.c94ab4.1)]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness file.%I.impl_witness_table, @A.as.I.impl(%V.loc12_14.1) [symbolic = %I.impl_witness (constants.%I.impl_witness.0c179e.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.as.I.impl.F.type: type = fn_type @A.as.I.impl.F, @A.as.I.impl(%V.loc12_14.1) [symbolic = %A.as.I.impl.F.type (constants.%A.as.I.impl.F.type.2974bf.1)]
// CHECK:STDOUT:   %A.as.I.impl.F: @A.as.I.impl.%A.as.I.impl.F.type (%A.as.I.impl.F.type.2974bf.1) = struct_value () [symbolic = %A.as.I.impl.F (constants.%A.as.I.impl.F.786476.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %A.loc12_27.2 as %I.type.loc12_35.2 {
// CHECK:STDOUT:     %A.as.I.impl.F.decl: @A.as.I.impl.%A.as.I.impl.F.type (%A.as.I.impl.F.type.2974bf.1) = fn_decl @A.as.I.impl.F [symbolic = @A.as.I.impl.%A.as.I.impl.F (constants.%A.as.I.impl.F.786476.1)] {
// CHECK:STDOUT:       %self.patt: @A.as.I.impl.F.%pattern_type.loc13_13 (%pattern_type.2180b5.1) = value_binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @A.as.I.impl.F.%pattern_type.loc13_13 (%pattern_type.2180b5.1) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %.loc13_8: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:       %return.patt: @A.as.I.impl.F.%pattern_type.loc13_28 (%pattern_type.a603a0.2) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @A.as.I.impl.F.%pattern_type.loc13_28 (%pattern_type.a603a0.2) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %V.ref: type = name_ref V, @A.as.I.impl.%V.loc12_14.2 [symbolic = %V (constants.%V.d9f)]
// CHECK:STDOUT:       %ptr.loc13_32.2: type = ptr_type %V.ref [symbolic = %ptr.loc13_32.1 (constants.%ptr.4f0b5c.2)]
// CHECK:STDOUT:       %self.param: @A.as.I.impl.F.%ptr.loc13_23.1 (%ptr.6f46b6.1) = value_param call_param0
// CHECK:STDOUT:       %.loc13_23: type = splice_block %ptr.loc13_23.2 [symbolic = %ptr.loc13_23.1 (constants.%ptr.6f46b6.1)] {
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, @A.as.I.impl.%A.loc12_27.2 [symbolic = %A (constants.%A.2ddf89.2)]
// CHECK:STDOUT:         %ptr.loc13_23.2: type = ptr_type %Self.ref [symbolic = %ptr.loc13_23.1 (constants.%ptr.6f46b6.1)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @A.as.I.impl.F.%ptr.loc13_23.1 (%ptr.6f46b6.1) = value_binding self, %self.param
// CHECK:STDOUT:       %return.param: ref @A.as.I.impl.F.%ptr.loc13_32.1 (%ptr.4f0b5c.2) = out_param call_param1
// CHECK:STDOUT:       %return: ref @A.as.I.impl.F.%ptr.loc13_32.1 (%ptr.4f0b5c.2) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .V = <poisoned>
// CHECK:STDOUT:     .F = %A.as.I.impl.F.decl
// CHECK:STDOUT:     witness = file.%I.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @A.as.I.impl.F(@A.as.I.impl.%V.loc12_14.2: type) {
// CHECK:STDOUT:   %V: type = symbolic_binding V, 0 [symbolic = %V (constants.%V.d9f)]
// CHECK:STDOUT:   %A: type = class_type @A, @A(%V) [symbolic = %A (constants.%A.2ddf89.2)]
// CHECK:STDOUT:   %ptr.loc13_23.1: type = ptr_type %A [symbolic = %ptr.loc13_23.1 (constants.%ptr.6f46b6.1)]
// CHECK:STDOUT:   %pattern_type.loc13_13: type = pattern_type %ptr.loc13_23.1 [symbolic = %pattern_type.loc13_13 (constants.%pattern_type.2180b5.1)]
// CHECK:STDOUT:   %ptr.loc13_32.1: type = ptr_type %V [symbolic = %ptr.loc13_32.1 (constants.%ptr.4f0b5c.2)]
// CHECK:STDOUT:   %pattern_type.loc13_28: type = pattern_type %ptr.loc13_32.1 [symbolic = %pattern_type.loc13_28 (constants.%pattern_type.a603a0.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc13_28: <witness> = require_complete_type %ptr.loc13_32.1 [symbolic = %require_complete.loc13_28 (constants.%require_complete.482d3f.1)]
// CHECK:STDOUT:   %require_complete.loc13_17: <witness> = require_complete_type %ptr.loc13_23.1 [symbolic = %require_complete.loc13_17 (constants.%require_complete.743ecf.1)]
// CHECK:STDOUT:   %require_complete.loc14: <witness> = require_complete_type %A [symbolic = %require_complete.loc14 (constants.%require_complete.f6b5ff.1)]
// CHECK:STDOUT:   %A.elem: type = unbound_element_type %A, %V [symbolic = %A.elem (constants.%A.elem.ee5a68.2)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr.loc13_32.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.c66)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.loc13_32.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet.747)]
// CHECK:STDOUT:   %.loc14_12.2: type = fn_type_with_self_type constants.%Copy.Op.type, %Copy.facet [symbolic = %.loc14_12.2 (constants.%.89d)]
// CHECK:STDOUT:   %impl.elem0.loc14_12.2: @A.as.I.impl.F.%.loc14_12.2 (%.89d) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc14_12.2 (constants.%impl.elem0.928)]
// CHECK:STDOUT:   %specific_impl_fn.loc14_12.2: <specific function> = specific_impl_function %impl.elem0.loc14_12.2, @Copy.Op(%Copy.facet) [symbolic = %specific_impl_fn.loc14_12.2 (constants.%specific_impl_fn.6a1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @A.as.I.impl.F.%ptr.loc13_23.1 (%ptr.6f46b6.1)) -> @A.as.I.impl.F.%ptr.loc13_32.1 (%ptr.4f0b5c.2) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @A.as.I.impl.F.%ptr.loc13_23.1 (%ptr.6f46b6.1) = name_ref self, %self
// CHECK:STDOUT:     %.loc14_17.1: ref @A.as.I.impl.F.%A (%A.2ddf89.2) = deref %self.ref
// CHECK:STDOUT:     %n.ref: @A.as.I.impl.F.%A.elem (%A.elem.ee5a68.2) = name_ref n, @A.%.loc4 [concrete = @A.%.loc4]
// CHECK:STDOUT:     %.loc14_17.2: ref @A.as.I.impl.F.%V (%V.d9f) = class_element_access %.loc14_17.1, element0
// CHECK:STDOUT:     %addr: @A.as.I.impl.F.%ptr.loc13_32.1 (%ptr.4f0b5c.2) = addr_of %.loc14_17.2
// CHECK:STDOUT:     %impl.elem0.loc14_12.1: @A.as.I.impl.F.%.loc14_12.2 (%.89d) = impl_witness_access constants.%Copy.lookup_impl_witness.c66, element0 [symbolic = %impl.elem0.loc14_12.2 (constants.%impl.elem0.928)]
// CHECK:STDOUT:     %bound_method.loc14_12.1: <bound method> = bound_method %addr, %impl.elem0.loc14_12.1
// CHECK:STDOUT:     %specific_impl_fn.loc14_12.1: <specific function> = specific_impl_function %impl.elem0.loc14_12.1, @Copy.Op(constants.%Copy.facet.747) [symbolic = %specific_impl_fn.loc14_12.2 (constants.%specific_impl_fn.6a1)]
// CHECK:STDOUT:     %bound_method.loc14_12.2: <bound method> = bound_method %addr, %specific_impl_fn.loc14_12.1
// CHECK:STDOUT:     %.loc14_12.1: init @A.as.I.impl.F.%ptr.loc13_32.1 (%ptr.4f0b5c.2) = call %bound_method.loc14_12.2(%addr)
// CHECK:STDOUT:     return %.loc14_12.1 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestGeneric(%W.loc19_16.2: type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %I.type.loc21_17.2: type = facet_type <@I, @I(%W.loc19_16.1)> [symbolic = %I.type.loc21_17.2 (constants.%I.type.07036d.3)]
// CHECK:STDOUT:   %require_complete.loc21_18: <witness> = require_complete_type %I.type.loc21_17.2 [symbolic = %require_complete.loc21_18 (constants.%require_complete.c94ab4.2)]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I, @I(%W.loc19_16.1) [symbolic = %I.assoc_type (constants.%I.assoc_type.b650b2.3)]
// CHECK:STDOUT:   %assoc0: @TestGeneric.%I.assoc_type (%I.assoc_type.b650b2.3) = assoc_entity element0, @I.%I.F.decl [symbolic = %assoc0 (constants.%assoc0.b4fda0.3)]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %A.loc19_32.1, @I, @I(%W.loc19_16.1) [symbolic = %I.lookup_impl_witness (constants.%I.lookup_impl_witness)]
// CHECK:STDOUT:   %I.F.type: type = fn_type @I.F, @I(%W.loc19_16.1) [symbolic = %I.F.type (constants.%I.F.type.76d346.3)]
// CHECK:STDOUT:   %I.facet: @TestGeneric.%I.type.loc21_17.2 (%I.type.07036d.3) = facet_value %A.loc19_32.1, (%I.lookup_impl_witness) [symbolic = %I.facet (constants.%I.facet.84d)]
// CHECK:STDOUT:   %.loc21_11.2: type = fn_type_with_self_type %I.F.type, %I.facet [symbolic = %.loc21_11.2 (constants.%.e50)]
// CHECK:STDOUT:   %impl.elem0.loc21_11.2: @TestGeneric.%.loc21_11.2 (%.e50) = impl_witness_access %I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc21_11.2 (constants.%impl.elem0.a9e)]
// CHECK:STDOUT:   %specific_impl_fn.loc21_11.2: <specific function> = specific_impl_function %impl.elem0.loc21_11.2, @I.F(%W.loc19_16.1, %I.facet) [symbolic = %specific_impl_fn.loc21_11.2 (constants.%specific_impl_fn.2e3)]
// CHECK:STDOUT:   %require_complete.loc21_11: <witness> = require_complete_type %A.loc19_32.1 [symbolic = %require_complete.loc21_11 (constants.%require_complete.f6b5ff.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @TestGeneric.%ptr.loc19_33.1 (%ptr.6f46b6.2)) -> @TestGeneric.%ptr.loc19_40.1 (%ptr.4f0b5c.4) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %a.ref: @TestGeneric.%ptr.loc19_33.1 (%ptr.6f46b6.2) = name_ref a, %a
// CHECK:STDOUT:     %I.ref: %I.type.dac = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:     %W.ref.loc21: type = name_ref W, %W.loc19_16.2 [symbolic = %W.loc19_16.1 (constants.%W)]
// CHECK:STDOUT:     %I.type.loc21_17.1: type = facet_type <@I, @I(constants.%W)> [symbolic = %I.type.loc21_17.2 (constants.%I.type.07036d.3)]
// CHECK:STDOUT:     %.loc21_18: @TestGeneric.%I.assoc_type (%I.assoc_type.b650b2.3) = specific_constant @I.%assoc0.loc8_33.1, @I(constants.%W) [symbolic = %assoc0 (constants.%assoc0.b4fda0.3)]
// CHECK:STDOUT:     %F.ref: @TestGeneric.%I.assoc_type (%I.assoc_type.b650b2.3) = name_ref F, %.loc21_18 [symbolic = %assoc0 (constants.%assoc0.b4fda0.3)]
// CHECK:STDOUT:     %.loc21_11.1: ref @TestGeneric.%A.loc19_32.1 (%A.2ddf89.3) = deref %a.ref
// CHECK:STDOUT:     %impl.elem0.loc21_11.1: @TestGeneric.%.loc21_11.2 (%.e50) = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc21_11.2 (constants.%impl.elem0.a9e)]
// CHECK:STDOUT:     %bound_method.loc21_11: <bound method> = bound_method %.loc21_11.1, %impl.elem0.loc21_11.1
// CHECK:STDOUT:     %specific_impl_fn.loc21_11.1: <specific function> = specific_impl_function %impl.elem0.loc21_11.1, @I.F(constants.%W, constants.%I.facet.84d) [symbolic = %specific_impl_fn.loc21_11.2 (constants.%specific_impl_fn.2e3)]
// CHECK:STDOUT:     %bound_method.loc21_22: <bound method> = bound_method %.loc21_11.1, %specific_impl_fn.loc21_11.1
// CHECK:STDOUT:     %addr: @TestGeneric.%ptr.loc19_33.1 (%ptr.6f46b6.2) = addr_of %.loc21_11.1
// CHECK:STDOUT:     %.loc21_22: init @TestGeneric.%ptr.loc19_40.1 (%ptr.4f0b5c.4) = call %bound_method.loc21_22(%addr)
// CHECK:STDOUT:     return %.loc21_22 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestSpecific(%a.param: %ptr.197) -> %ptr.c28 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %ptr.197 = name_ref a, %a
// CHECK:STDOUT:   %I.ref: %I.type.dac = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:   %.loc27_17: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc27_18: type = converted %.loc27_17, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(constants.%empty_struct_type)> [concrete = constants.%I.type.399]
// CHECK:STDOUT:   %.loc27_19: %I.assoc_type.22c = specific_constant @I.%assoc0.loc8_33.1, @I(constants.%empty_struct_type) [concrete = constants.%assoc0.a7f]
// CHECK:STDOUT:   %F.ref: %I.assoc_type.22c = name_ref F, %.loc27_19 [concrete = constants.%assoc0.a7f]
// CHECK:STDOUT:   %.loc27_11: ref %A.235 = deref %a.ref
// CHECK:STDOUT:   %impl.elem0: %.824 = impl_witness_access constants.%I.impl_witness.915, element0 [concrete = constants.%A.as.I.impl.F.041]
// CHECK:STDOUT:   %bound_method.loc27_11: <bound method> = bound_method %.loc27_11, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @A.as.I.impl.F(constants.%empty_struct_type) [concrete = constants.%A.as.I.impl.F.specific_fn]
// CHECK:STDOUT:   %bound_method.loc27_23: <bound method> = bound_method %.loc27_11, %specific_fn
// CHECK:STDOUT:   %addr: %ptr.197 = addr_of %.loc27_11
// CHECK:STDOUT:   %A.as.I.impl.F.call: init %ptr.c28 = call %bound_method.loc27_23(%addr)
// CHECK:STDOUT:   return %A.as.I.impl.F.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.as.I.impl(constants.%V.d9f) {
// CHECK:STDOUT:   %V.loc12_14.1 => constants.%V.d9f
// CHECK:STDOUT:   %A.loc12_27.1 => constants.%A.2ddf89.2
// CHECK:STDOUT:   %I.type.loc12_35.1 => constants.%I.type.07036d.2
// CHECK:STDOUT:   %require_complete => constants.%require_complete.c94ab4.1
// CHECK:STDOUT:   %I.impl_witness => constants.%I.impl_witness.0c179e.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.as.I.impl.F.type => constants.%A.as.I.impl.F.type.2974bf.1
// CHECK:STDOUT:   %A.as.I.impl.F => constants.%A.as.I.impl.F.786476.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.as.I.impl.F(constants.%V.d9f) {
// CHECK:STDOUT:   %V => constants.%V.d9f
// CHECK:STDOUT:   %A => constants.%A.2ddf89.2
// CHECK:STDOUT:   %ptr.loc13_23.1 => constants.%ptr.6f46b6.1
// CHECK:STDOUT:   %pattern_type.loc13_13 => constants.%pattern_type.2180b5.1
// CHECK:STDOUT:   %ptr.loc13_32.1 => constants.%ptr.4f0b5c.2
// CHECK:STDOUT:   %pattern_type.loc13_28 => constants.%pattern_type.a603a0.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGeneric(constants.%W) {
// CHECK:STDOUT:   %W.loc19_16.1 => constants.%W
// CHECK:STDOUT:   %A.loc19_32.1 => constants.%A.2ddf89.3
// CHECK:STDOUT:   %ptr.loc19_33.1 => constants.%ptr.6f46b6.2
// CHECK:STDOUT:   %pattern_type.loc19_26 => constants.%pattern_type.2180b5.2
// CHECK:STDOUT:   %ptr.loc19_40.1 => constants.%ptr.4f0b5c.4
// CHECK:STDOUT:   %pattern_type.loc19_36 => constants.%pattern_type.a603a0.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.as.I.impl(constants.%W) {
// CHECK:STDOUT:   %V.loc12_14.1 => constants.%W
// CHECK:STDOUT:   %A.loc12_27.1 => constants.%A.2ddf89.3
// CHECK:STDOUT:   %I.type.loc12_35.1 => constants.%I.type.07036d.3
// CHECK:STDOUT:   %require_complete => constants.%require_complete.c94ab4.2
// CHECK:STDOUT:   %I.impl_witness => constants.%I.impl_witness.0c179e.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.as.I.impl.F.type => constants.%A.as.I.impl.F.type.2974bf.2
// CHECK:STDOUT:   %A.as.I.impl.F => constants.%A.as.I.impl.F.786476.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.as.I.impl(constants.%empty_struct_type) {
// CHECK:STDOUT:   %V.loc12_14.1 => constants.%empty_struct_type
// CHECK:STDOUT:   %A.loc12_27.1 => constants.%A.235
// CHECK:STDOUT:   %I.type.loc12_35.1 => constants.%I.type.399
// CHECK:STDOUT:   %require_complete => constants.%complete_type.e4b
// CHECK:STDOUT:   %I.impl_witness => constants.%I.impl_witness.915
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.as.I.impl.F.type => constants.%A.as.I.impl.F.type.184
// CHECK:STDOUT:   %A.as.I.impl.F => constants.%A.as.I.impl.F.041
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.as.I.impl.F(constants.%empty_struct_type) {
// CHECK:STDOUT:   %V => constants.%empty_struct_type
// CHECK:STDOUT:   %A => constants.%A.235
// CHECK:STDOUT:   %ptr.loc13_23.1 => constants.%ptr.197
// CHECK:STDOUT:   %pattern_type.loc13_13 => constants.%pattern_type.e47
// CHECK:STDOUT:   %ptr.loc13_32.1 => constants.%ptr.c28
// CHECK:STDOUT:   %pattern_type.loc13_28 => constants.%pattern_type.1cc
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc13_28 => constants.%complete_type.38e
// CHECK:STDOUT:   %require_complete.loc13_17 => constants.%complete_type.d27
// CHECK:STDOUT:   %require_complete.loc14 => constants.%complete_type.0a6
// CHECK:STDOUT:   %A.elem => constants.%A.elem.2af
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.impl_witness.433
// CHECK:STDOUT:   %Copy.facet => constants.%Copy.facet.48c
// CHECK:STDOUT:   %.loc14_12.2 => constants.%.fb0
// CHECK:STDOUT:   %impl.elem0.loc14_12.2 => constants.%ptr.as.Copy.impl.Op.15e
// CHECK:STDOUT:   %specific_impl_fn.loc14_12.2 => constants.%ptr.as.Copy.impl.Op.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:

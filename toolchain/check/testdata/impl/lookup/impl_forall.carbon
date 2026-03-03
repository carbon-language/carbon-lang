// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/lookup/impl_forall.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/lookup/impl_forall.carbon

// --- impl_forall.carbon
library "[[@TEST_NAME]]";

class A(T:! type) {
  var n: T;
}

interface I(U:! type) {
  fn F[ref self: Self]() -> U*;
}

//@dump-sem-ir-begin
impl forall [V:! type] A(V) as I(V) {
  fn F[ref self: Self]() -> V* {
    return &self.n;
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
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %A.type: type = generic_class_type @A [concrete]
// CHECK:STDOUT:   %A.generic: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %U.67d: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %I.type.609: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.609 = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.1ab3e4.1: type = facet_type <@I, @I(%U.67d)> [symbolic]
// CHECK:STDOUT:   %Self.fdb544.1: %I.type.1ab3e4.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %V.67d: type = symbolic_binding V, 0 [symbolic]
// CHECK:STDOUT:   %A.95c0c7.2: type = class_type @A, @A(%V.67d) [symbolic]
// CHECK:STDOUT:   %I.type.1ab3e4.2: type = facet_type <@I, @I(%V.67d)> [symbolic]
// CHECK:STDOUT:   %I.impl_witness.f3f3bd.1: <witness> = impl_witness @A.as.I.impl.%I.impl_witness_table, @A.as.I.impl(%V.67d) [symbolic]
// CHECK:STDOUT:   %require_complete.e360af.1: <witness> = require_complete_type %I.type.1ab3e4.2 [symbolic]
// CHECK:STDOUT:   %pattern_type.2c404c.1: type = pattern_type %A.95c0c7.2 [symbolic]
// CHECK:STDOUT:   %ptr.e8f8f9.2: type = ptr_type %V.67d [symbolic]
// CHECK:STDOUT:   %.4fe7d1.2: Core.Form = init_form %ptr.e8f8f9.2, call_param1 [symbolic]
// CHECK:STDOUT:   %pattern_type.4f4b84.2: type = pattern_type %ptr.e8f8f9.2 [symbolic]
// CHECK:STDOUT:   %A.as.I.impl.F.type.58747b.1: type = fn_type @A.as.I.impl.F, @A.as.I.impl(%V.67d) [symbolic]
// CHECK:STDOUT:   %A.as.I.impl.F.afeb91.1: %A.as.I.impl.F.type.58747b.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %A.elem.8a20fa.2: type = unbound_element_type %A.95c0c7.2, %V.67d [symbolic]
// CHECK:STDOUT:   %require_complete.fd656a.1: <witness> = require_complete_type %A.95c0c7.2 [symbolic]
// CHECK:STDOUT:   %require_complete.ef162c.1: <witness> = require_complete_type %ptr.e8f8f9.2 [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.2d4455.1: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.74e93b.1: %ptr.as.Copy.impl.Op.type.2d4455.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.2f2: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%V.67d) [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.2e6: <witness> = lookup_impl_witness %ptr.e8f8f9.2, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.facet.c25: %Copy.type = facet_value %ptr.e8f8f9.2, (%Copy.lookup_impl_witness.2e6) [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.d82: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.c25) [symbolic]
// CHECK:STDOUT:   %.299: type = fn_type_with_self_type %Copy.WithSelf.Op.type.d82, %Copy.facet.c25 [symbolic]
// CHECK:STDOUT:   %impl.elem0.1c7: %.299 = impl_witness_access %Copy.lookup_impl_witness.2e6, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.366: <specific function> = specific_impl_function %impl.elem0.1c7, @Copy.WithSelf.Op(%Copy.facet.c25) [symbolic]
// CHECK:STDOUT:   %W: type = symbolic_binding W, 0 [symbolic]
// CHECK:STDOUT:   %A.95c0c7.3: type = class_type @A, @A(%W) [symbolic]
// CHECK:STDOUT:   %ptr.5da: type = ptr_type %A.95c0c7.3 [symbolic]
// CHECK:STDOUT:   %pattern_type.605: type = pattern_type %ptr.5da [symbolic]
// CHECK:STDOUT:   %ptr.e8f8f9.4: type = ptr_type %W [symbolic]
// CHECK:STDOUT:   %.4fe7d1.4: Core.Form = init_form %ptr.e8f8f9.4, call_param1 [symbolic]
// CHECK:STDOUT:   %pattern_type.4f4b84.4: type = pattern_type %ptr.e8f8f9.4 [symbolic]
// CHECK:STDOUT:   %I.type.1ab3e4.3: type = facet_type <@I, @I(%W)> [symbolic]
// CHECK:STDOUT:   %I.assoc_type.76c031.3: type = assoc_entity_type @I, @I(%W) [symbolic]
// CHECK:STDOUT:   %assoc0.203667.3: %I.assoc_type.76c031.3 = assoc_entity element0, @I.WithSelf.%I.WithSelf.F.decl [symbolic]
// CHECK:STDOUT:   %require_complete.e360af.2: <witness> = require_complete_type %I.type.1ab3e4.3 [symbolic]
// CHECK:STDOUT:   %I.impl_witness.f3f3bd.2: <witness> = impl_witness @A.as.I.impl.%I.impl_witness_table, @A.as.I.impl(%W) [symbolic]
// CHECK:STDOUT:   %A.as.I.impl.F.type.58747b.2: type = fn_type @A.as.I.impl.F, @A.as.I.impl(%W) [symbolic]
// CHECK:STDOUT:   %A.as.I.impl.F.afeb91.2: %A.as.I.impl.F.type.58747b.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %.0fe: require_specific_def_type = require_specific_def @A.as.I.impl(%W) [symbolic]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %A.95c0c7.3, @I, @I(%W) [symbolic]
// CHECK:STDOUT:   %I.facet.558: %I.type.1ab3e4.3 = facet_value %A.95c0c7.3, (%I.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type.912: type = fn_type @I.WithSelf.F, @I.WithSelf(%W, %I.facet.558) [symbolic]
// CHECK:STDOUT:   %.695: type = fn_type_with_self_type %I.WithSelf.F.type.912, %I.facet.558 [symbolic]
// CHECK:STDOUT:   %impl.elem0.976: %.695 = impl_witness_access %I.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.3c2: <specific function> = specific_impl_function %impl.elem0.976, @I.WithSelf.F(%W, %I.facet.558) [symbolic]
// CHECK:STDOUT:   %require_complete.fd656a.2: <witness> = require_complete_type %A.95c0c7.3 [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %A.ed7: type = class_type @A, @A(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %ptr.ed2: type = ptr_type %A.ed7 [concrete]
// CHECK:STDOUT:   %ptr.c28: type = ptr_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %.98f: Core.Form = init_form %ptr.c28, call_param1 [concrete]
// CHECK:STDOUT:   %pattern_type.1cc: type = pattern_type %ptr.c28 [concrete]
// CHECK:STDOUT:   %I.type.ab5: type = facet_type <@I, @I(%empty_struct_type)> [concrete]
// CHECK:STDOUT:   %I.assoc_type.294: type = assoc_entity_type @I, @I(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %assoc0.c64: %I.assoc_type.294 = assoc_entity element0, @I.WithSelf.%I.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %I.impl_witness.708: <witness> = impl_witness @A.as.I.impl.%I.impl_witness_table, @A.as.I.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %complete_type.5b1: <witness> = complete_type_witness %I.type.ab5 [concrete]
// CHECK:STDOUT:   %A.as.I.impl.F.type.aa2: type = fn_type @A.as.I.impl.F, @A.as.I.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %A.as.I.impl.F.c38: %A.as.I.impl.F.type.aa2 = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet.681: %I.type.ab5 = facet_value %A.ed7, (%I.impl_witness.708) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.type.d75: type = fn_type @I.WithSelf.F, @I.WithSelf(%empty_struct_type, %I.facet.681) [concrete]
// CHECK:STDOUT:   %.d59: type = fn_type_with_self_type %I.WithSelf.F.type.d75, %I.facet.681 [concrete]
// CHECK:STDOUT:   %pattern_type.853: type = pattern_type %A.ed7 [concrete]
// CHECK:STDOUT:   %A.as.I.impl.F.specific_fn: <specific function> = specific_function %A.as.I.impl.F.c38, @A.as.I.impl.F(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %A.elem.599: type = unbound_element_type %A.ed7, %empty_struct_type [concrete]
// CHECK:STDOUT:   %struct_type.n.91c: type = struct_type {.n: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %complete_type.0a6: <witness> = complete_type_witness %struct_type.n.91c [concrete]
// CHECK:STDOUT:   %complete_type.38e: <witness> = complete_type_witness %ptr.c28 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.3fd: <witness> = impl_witness imports.%Copy.impl_witness_table.c3a, @ptr.as.Copy.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.032: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.cbf: %ptr.as.Copy.impl.Op.type.032 = struct_value () [concrete]
// CHECK:STDOUT:   %.07c: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %Copy.facet.111: %Copy.type = facet_value %ptr.c28, (%Copy.impl_witness.3fd) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.514: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.111) [concrete]
// CHECK:STDOUT:   %.237: type = fn_type_with_self_type %Copy.WithSelf.Op.type.514, %Copy.facet.111 [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.cbf, @ptr.as.Copy.impl.Op(%empty_struct_type) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.203: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.2d4455.1) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.74e93b.1)]
// CHECK:STDOUT:   %Copy.impl_witness_table.c3a = impl_witness_table (%Core.import_ref.203), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   impl_decl @A.as.I.impl [concrete] {
// CHECK:STDOUT:     %V.patt: %pattern_type.98f = symbolic_binding_pattern V, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A.ref: %A.type = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:     %V.ref.loc12_26: type = name_ref V, %V.loc12_14.2 [symbolic = %V.loc12_14.1 (constants.%V.67d)]
// CHECK:STDOUT:     %A.loc12_27.2: type = class_type @A, @A(constants.%V.67d) [symbolic = %A.loc12_27.1 (constants.%A.95c0c7.2)]
// CHECK:STDOUT:     %I.ref: %I.type.609 = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:     %V.ref.loc12_34: type = name_ref V, %V.loc12_14.2 [symbolic = %V.loc12_14.1 (constants.%V.67d)]
// CHECK:STDOUT:     %I.type.loc12_35.2: type = facet_type <@I, @I(constants.%V.67d)> [symbolic = %I.type.loc12_35.1 (constants.%I.type.1ab3e4.2)]
// CHECK:STDOUT:     %.loc12_18.1: type = splice_block %.loc12_18.2 [concrete = type] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %.loc12_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %V.loc12_14.2: type = symbolic_binding V, 0 [symbolic = %V.loc12_14.1 (constants.%V.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @A.as.I.impl(%V.loc12_14.2: type) {
// CHECK:STDOUT:   %V.loc12_14.1: type = symbolic_binding V, 0 [symbolic = %V.loc12_14.1 (constants.%V.67d)]
// CHECK:STDOUT:   %A.loc12_27.1: type = class_type @A, @A(%V.loc12_14.1) [symbolic = %A.loc12_27.1 (constants.%A.95c0c7.2)]
// CHECK:STDOUT:   %I.type.loc12_35.1: type = facet_type <@I, @I(%V.loc12_14.1)> [symbolic = %I.type.loc12_35.1 (constants.%I.type.1ab3e4.2)]
// CHECK:STDOUT:   %I.impl_witness.loc12_37.2: <witness> = impl_witness %I.impl_witness_table, @A.as.I.impl(%V.loc12_14.1) [symbolic = %I.impl_witness.loc12_37.2 (constants.%I.impl_witness.f3f3bd.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.type.loc12_35.1 [symbolic = %require_complete (constants.%require_complete.e360af.1)]
// CHECK:STDOUT:   %A.as.I.impl.F.type: type = fn_type @A.as.I.impl.F, @A.as.I.impl(%V.loc12_14.1) [symbolic = %A.as.I.impl.F.type (constants.%A.as.I.impl.F.type.58747b.1)]
// CHECK:STDOUT:   %A.as.I.impl.F: @A.as.I.impl.%A.as.I.impl.F.type (%A.as.I.impl.F.type.58747b.1) = struct_value () [symbolic = %A.as.I.impl.F (constants.%A.as.I.impl.F.afeb91.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %A.loc12_27.2 as %I.type.loc12_35.2 {
// CHECK:STDOUT:     %A.as.I.impl.F.decl: @A.as.I.impl.%A.as.I.impl.F.type (%A.as.I.impl.F.type.58747b.1) = fn_decl @A.as.I.impl.F [symbolic = @A.as.I.impl.%A.as.I.impl.F (constants.%A.as.I.impl.F.afeb91.1)] {
// CHECK:STDOUT:       %self.patt: @A.as.I.impl.F.%pattern_type.loc13_12 (%pattern_type.2c404c.1) = ref_binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @A.as.I.impl.F.%pattern_type.loc13_12 (%pattern_type.2c404c.1) = ref_param_pattern %self.patt [concrete]
// CHECK:STDOUT:       %return.patt: @A.as.I.impl.F.%pattern_type.loc13_26 (%pattern_type.4f4b84.2) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @A.as.I.impl.F.%pattern_type.loc13_26 (%pattern_type.4f4b84.2) = out_param_pattern %return.patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %V.ref: type = name_ref V, @A.as.I.impl.%V.loc12_14.2 [symbolic = %V (constants.%V.67d)]
// CHECK:STDOUT:       %ptr.loc13_30.2: type = ptr_type %V.ref [symbolic = %ptr.loc13_30.1 (constants.%ptr.e8f8f9.2)]
// CHECK:STDOUT:       %.loc13_30.2: Core.Form = init_form %ptr.loc13_30.2, call_param1 [symbolic = %.loc13_30.1 (constants.%.4fe7d1.2)]
// CHECK:STDOUT:       %self.param: ref @A.as.I.impl.F.%A (%A.95c0c7.2) = ref_param call_param0
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, @A.as.I.impl.%A.loc12_27.2 [symbolic = %A (constants.%A.95c0c7.2)]
// CHECK:STDOUT:       %self: ref @A.as.I.impl.F.%A (%A.95c0c7.2) = ref_binding self, %self.param
// CHECK:STDOUT:       %return.param: ref @A.as.I.impl.F.%ptr.loc13_30.1 (%ptr.e8f8f9.2) = out_param call_param1
// CHECK:STDOUT:       %return: ref @A.as.I.impl.F.%ptr.loc13_30.1 (%ptr.e8f8f9.2) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %I.impl_witness_table = impl_witness_table (%A.as.I.impl.F.decl), @A.as.I.impl [concrete]
// CHECK:STDOUT:     %I.impl_witness.loc12_37.1: <witness> = impl_witness %I.impl_witness_table, @A.as.I.impl(constants.%V.67d) [symbolic = %I.impl_witness.loc12_37.2 (constants.%I.impl_witness.f3f3bd.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .V = <poisoned>
// CHECK:STDOUT:     .F = %A.as.I.impl.F.decl
// CHECK:STDOUT:     witness = %I.impl_witness.loc12_37.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @A.as.I.impl.F(@A.as.I.impl.%V.loc12_14.2: type) {
// CHECK:STDOUT:   %V: type = symbolic_binding V, 0 [symbolic = %V (constants.%V.67d)]
// CHECK:STDOUT:   %A: type = class_type @A, @A(%V) [symbolic = %A (constants.%A.95c0c7.2)]
// CHECK:STDOUT:   %pattern_type.loc13_12: type = pattern_type %A [symbolic = %pattern_type.loc13_12 (constants.%pattern_type.2c404c.1)]
// CHECK:STDOUT:   %ptr.loc13_30.1: type = ptr_type %V [symbolic = %ptr.loc13_30.1 (constants.%ptr.e8f8f9.2)]
// CHECK:STDOUT:   %.loc13_30.1: Core.Form = init_form %ptr.loc13_30.1, call_param1 [symbolic = %.loc13_30.1 (constants.%.4fe7d1.2)]
// CHECK:STDOUT:   %pattern_type.loc13_26: type = pattern_type %ptr.loc13_30.1 [symbolic = %pattern_type.loc13_26 (constants.%pattern_type.4f4b84.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc13_16: <witness> = require_complete_type %A [symbolic = %require_complete.loc13_16 (constants.%require_complete.fd656a.1)]
// CHECK:STDOUT:   %require_complete.loc13_30: <witness> = require_complete_type %ptr.loc13_30.1 [symbolic = %require_complete.loc13_30 (constants.%require_complete.ef162c.1)]
// CHECK:STDOUT:   %A.elem: type = unbound_element_type %A, %V [symbolic = %A.elem (constants.%A.elem.8a20fa.2)]
// CHECK:STDOUT:   %.loc14_12.1: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%V) [symbolic = %.loc14_12.1 (constants.%.2f2)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr.loc13_30.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.2e6)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.loc13_30.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet.c25)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.d82)]
// CHECK:STDOUT:   %.loc14_12.2: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %Copy.facet [symbolic = %.loc14_12.2 (constants.%.299)]
// CHECK:STDOUT:   %impl.elem0.loc14_12.2: @A.as.I.impl.F.%.loc14_12.2 (%.299) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc14_12.2 (constants.%impl.elem0.1c7)]
// CHECK:STDOUT:   %specific_impl_fn.loc14_12.2: <specific function> = specific_impl_function %impl.elem0.loc14_12.2, @Copy.WithSelf.Op(%Copy.facet) [symbolic = %specific_impl_fn.loc14_12.2 (constants.%specific_impl_fn.366)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: ref @A.as.I.impl.F.%A (%A.95c0c7.2)) -> out %return.param: @A.as.I.impl.F.%ptr.loc13_30.1 (%ptr.e8f8f9.2) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: ref @A.as.I.impl.F.%A (%A.95c0c7.2) = name_ref self, %self
// CHECK:STDOUT:     %n.ref: @A.as.I.impl.F.%A.elem (%A.elem.8a20fa.2) = name_ref n, @A.%.loc4 [concrete = @A.%.loc4]
// CHECK:STDOUT:     %.loc14_17: ref @A.as.I.impl.F.%V (%V.67d) = class_element_access %self.ref, element0
// CHECK:STDOUT:     %addr: @A.as.I.impl.F.%ptr.loc13_30.1 (%ptr.e8f8f9.2) = addr_of %.loc14_17
// CHECK:STDOUT:     %impl.elem0.loc14_12.1: @A.as.I.impl.F.%.loc14_12.2 (%.299) = impl_witness_access constants.%Copy.lookup_impl_witness.2e6, element0 [symbolic = %impl.elem0.loc14_12.2 (constants.%impl.elem0.1c7)]
// CHECK:STDOUT:     %bound_method.loc14_12.1: <bound method> = bound_method %addr, %impl.elem0.loc14_12.1
// CHECK:STDOUT:     %specific_impl_fn.loc14_12.1: <specific function> = specific_impl_function %impl.elem0.loc14_12.1, @Copy.WithSelf.Op(constants.%Copy.facet.c25) [symbolic = %specific_impl_fn.loc14_12.2 (constants.%specific_impl_fn.366)]
// CHECK:STDOUT:     %bound_method.loc14_12.2: <bound method> = bound_method %addr, %specific_impl_fn.loc14_12.1
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @A.as.I.impl.F.%ptr.loc13_30.1 (%ptr.e8f8f9.2) = call %bound_method.loc14_12.2(%addr)
// CHECK:STDOUT:     return %Copy.WithSelf.Op.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestGeneric(%W.loc19_16.2: type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %I.type.loc21_17.2: type = facet_type <@I, @I(%W.loc19_16.1)> [symbolic = %I.type.loc21_17.2 (constants.%I.type.1ab3e4.3)]
// CHECK:STDOUT:   %require_complete.loc21_18: <witness> = require_complete_type %I.type.loc21_17.2 [symbolic = %require_complete.loc21_18 (constants.%require_complete.e360af.2)]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I, @I(%W.loc19_16.1) [symbolic = %I.assoc_type (constants.%I.assoc_type.76c031.3)]
// CHECK:STDOUT:   %assoc0: @TestGeneric.%I.assoc_type (%I.assoc_type.76c031.3) = assoc_entity element0, @I.WithSelf.%I.WithSelf.F.decl [symbolic = %assoc0 (constants.%assoc0.203667.3)]
// CHECK:STDOUT:   %.loc21_11.2: require_specific_def_type = require_specific_def @A.as.I.impl(%W.loc19_16.1) [symbolic = %.loc21_11.2 (constants.%.0fe)]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %A.loc19_32.1, @I, @I(%W.loc19_16.1) [symbolic = %I.lookup_impl_witness (constants.%I.lookup_impl_witness)]
// CHECK:STDOUT:   %I.facet: @TestGeneric.%I.type.loc21_17.2 (%I.type.1ab3e4.3) = facet_value %A.loc19_32.1, (%I.lookup_impl_witness) [symbolic = %I.facet (constants.%I.facet.558)]
// CHECK:STDOUT:   %I.WithSelf.F.type: type = fn_type @I.WithSelf.F, @I.WithSelf(%W.loc19_16.1, %I.facet) [symbolic = %I.WithSelf.F.type (constants.%I.WithSelf.F.type.912)]
// CHECK:STDOUT:   %.loc21_11.3: type = fn_type_with_self_type %I.WithSelf.F.type, %I.facet [symbolic = %.loc21_11.3 (constants.%.695)]
// CHECK:STDOUT:   %impl.elem0.loc21_11.2: @TestGeneric.%.loc21_11.3 (%.695) = impl_witness_access %I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc21_11.2 (constants.%impl.elem0.976)]
// CHECK:STDOUT:   %specific_impl_fn.loc21_11.2: <specific function> = specific_impl_function %impl.elem0.loc21_11.2, @I.WithSelf.F(%W.loc19_16.1, %I.facet) [symbolic = %specific_impl_fn.loc21_11.2 (constants.%specific_impl_fn.3c2)]
// CHECK:STDOUT:   %require_complete.loc21_11: <witness> = require_complete_type %A.loc19_32.1 [symbolic = %require_complete.loc21_11 (constants.%require_complete.fd656a.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @TestGeneric.%ptr.loc19_33.1 (%ptr.5da)) -> out %return.param: @TestGeneric.%ptr.loc19_40.1 (%ptr.e8f8f9.4) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %a.ref: @TestGeneric.%ptr.loc19_33.1 (%ptr.5da) = name_ref a, %a
// CHECK:STDOUT:     %I.ref: %I.type.609 = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:     %W.ref.loc21: type = name_ref W, %W.loc19_16.2 [symbolic = %W.loc19_16.1 (constants.%W)]
// CHECK:STDOUT:     %I.type.loc21_17.1: type = facet_type <@I, @I(constants.%W)> [symbolic = %I.type.loc21_17.2 (constants.%I.type.1ab3e4.3)]
// CHECK:STDOUT:     %.loc21_18: @TestGeneric.%I.assoc_type (%I.assoc_type.76c031.3) = specific_constant @I.WithSelf.%assoc0.loc8_31.1, @I.WithSelf(constants.%W, constants.%Self.fdb544.1) [symbolic = %assoc0 (constants.%assoc0.203667.3)]
// CHECK:STDOUT:     %F.ref: @TestGeneric.%I.assoc_type (%I.assoc_type.76c031.3) = name_ref F, %.loc21_18 [symbolic = %assoc0 (constants.%assoc0.203667.3)]
// CHECK:STDOUT:     %.loc21_11.1: ref @TestGeneric.%A.loc19_32.1 (%A.95c0c7.3) = deref %a.ref
// CHECK:STDOUT:     %impl.elem0.loc21_11.1: @TestGeneric.%.loc21_11.3 (%.695) = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc21_11.2 (constants.%impl.elem0.976)]
// CHECK:STDOUT:     %bound_method.loc21_11: <bound method> = bound_method %.loc21_11.1, %impl.elem0.loc21_11.1
// CHECK:STDOUT:     %specific_impl_fn.loc21_11.1: <specific function> = specific_impl_function %impl.elem0.loc21_11.1, @I.WithSelf.F(constants.%W, constants.%I.facet.558) [symbolic = %specific_impl_fn.loc21_11.2 (constants.%specific_impl_fn.3c2)]
// CHECK:STDOUT:     %bound_method.loc21_22: <bound method> = bound_method %.loc21_11.1, %specific_impl_fn.loc21_11.1
// CHECK:STDOUT:     %I.WithSelf.F.call: init @TestGeneric.%ptr.loc19_40.1 (%ptr.e8f8f9.4) = call %bound_method.loc21_22(%.loc21_11.1)
// CHECK:STDOUT:     return %I.WithSelf.F.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestSpecific(%a.param: %ptr.ed2) -> out %return.param: %ptr.c28 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %ptr.ed2 = name_ref a, %a
// CHECK:STDOUT:   %I.ref: %I.type.609 = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:   %.loc27_17: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc27_18: type = converted %.loc27_17, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(constants.%empty_struct_type)> [concrete = constants.%I.type.ab5]
// CHECK:STDOUT:   %.loc27_19: %I.assoc_type.294 = specific_constant @I.WithSelf.%assoc0.loc8_31.1, @I.WithSelf(constants.%empty_struct_type, constants.%Self.fdb544.1) [concrete = constants.%assoc0.c64]
// CHECK:STDOUT:   %F.ref: %I.assoc_type.294 = name_ref F, %.loc27_19 [concrete = constants.%assoc0.c64]
// CHECK:STDOUT:   %.loc27_11: ref %A.ed7 = deref %a.ref
// CHECK:STDOUT:   %impl.elem0: %.d59 = impl_witness_access constants.%I.impl_witness.708, element0 [concrete = constants.%A.as.I.impl.F.c38]
// CHECK:STDOUT:   %bound_method.loc27_11: <bound method> = bound_method %.loc27_11, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @A.as.I.impl.F(constants.%empty_struct_type) [concrete = constants.%A.as.I.impl.F.specific_fn]
// CHECK:STDOUT:   %bound_method.loc27_23: <bound method> = bound_method %.loc27_11, %specific_fn
// CHECK:STDOUT:   %A.as.I.impl.F.call: init %ptr.c28 = call %bound_method.loc27_23(%.loc27_11)
// CHECK:STDOUT:   return %A.as.I.impl.F.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.as.I.impl(constants.%V.67d) {
// CHECK:STDOUT:   %V.loc12_14.1 => constants.%V.67d
// CHECK:STDOUT:   %A.loc12_27.1 => constants.%A.95c0c7.2
// CHECK:STDOUT:   %I.type.loc12_35.1 => constants.%I.type.1ab3e4.2
// CHECK:STDOUT:   %I.impl_witness.loc12_37.2 => constants.%I.impl_witness.f3f3bd.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.e360af.1
// CHECK:STDOUT:   %A.as.I.impl.F.type => constants.%A.as.I.impl.F.type.58747b.1
// CHECK:STDOUT:   %A.as.I.impl.F => constants.%A.as.I.impl.F.afeb91.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.as.I.impl.F(constants.%V.67d) {
// CHECK:STDOUT:   %V => constants.%V.67d
// CHECK:STDOUT:   %A => constants.%A.95c0c7.2
// CHECK:STDOUT:   %pattern_type.loc13_12 => constants.%pattern_type.2c404c.1
// CHECK:STDOUT:   %ptr.loc13_30.1 => constants.%ptr.e8f8f9.2
// CHECK:STDOUT:   %.loc13_30.1 => constants.%.4fe7d1.2
// CHECK:STDOUT:   %pattern_type.loc13_26 => constants.%pattern_type.4f4b84.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGeneric(constants.%W) {
// CHECK:STDOUT:   %W.loc19_16.1 => constants.%W
// CHECK:STDOUT:   %A.loc19_32.1 => constants.%A.95c0c7.3
// CHECK:STDOUT:   %ptr.loc19_33.1 => constants.%ptr.5da
// CHECK:STDOUT:   %pattern_type.loc19_26 => constants.%pattern_type.605
// CHECK:STDOUT:   %ptr.loc19_40.1 => constants.%ptr.e8f8f9.4
// CHECK:STDOUT:   %.loc19_40.1 => constants.%.4fe7d1.4
// CHECK:STDOUT:   %pattern_type.loc19_36 => constants.%pattern_type.4f4b84.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.as.I.impl(constants.%W) {
// CHECK:STDOUT:   %V.loc12_14.1 => constants.%W
// CHECK:STDOUT:   %A.loc12_27.1 => constants.%A.95c0c7.3
// CHECK:STDOUT:   %I.type.loc12_35.1 => constants.%I.type.1ab3e4.3
// CHECK:STDOUT:   %I.impl_witness.loc12_37.2 => constants.%I.impl_witness.f3f3bd.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.e360af.2
// CHECK:STDOUT:   %A.as.I.impl.F.type => constants.%A.as.I.impl.F.type.58747b.2
// CHECK:STDOUT:   %A.as.I.impl.F => constants.%A.as.I.impl.F.afeb91.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.as.I.impl(constants.%empty_struct_type) {
// CHECK:STDOUT:   %V.loc12_14.1 => constants.%empty_struct_type
// CHECK:STDOUT:   %A.loc12_27.1 => constants.%A.ed7
// CHECK:STDOUT:   %I.type.loc12_35.1 => constants.%I.type.ab5
// CHECK:STDOUT:   %I.impl_witness.loc12_37.2 => constants.%I.impl_witness.708
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.5b1
// CHECK:STDOUT:   %A.as.I.impl.F.type => constants.%A.as.I.impl.F.type.aa2
// CHECK:STDOUT:   %A.as.I.impl.F => constants.%A.as.I.impl.F.c38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.as.I.impl.F(constants.%empty_struct_type) {
// CHECK:STDOUT:   %V => constants.%empty_struct_type
// CHECK:STDOUT:   %A => constants.%A.ed7
// CHECK:STDOUT:   %pattern_type.loc13_12 => constants.%pattern_type.853
// CHECK:STDOUT:   %ptr.loc13_30.1 => constants.%ptr.c28
// CHECK:STDOUT:   %.loc13_30.1 => constants.%.98f
// CHECK:STDOUT:   %pattern_type.loc13_26 => constants.%pattern_type.1cc
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc13_16 => constants.%complete_type.0a6
// CHECK:STDOUT:   %require_complete.loc13_30 => constants.%complete_type.38e
// CHECK:STDOUT:   %A.elem => constants.%A.elem.599
// CHECK:STDOUT:   %.loc14_12.1 => constants.%.07c
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.impl_witness.3fd
// CHECK:STDOUT:   %Copy.facet => constants.%Copy.facet.111
// CHECK:STDOUT:   %Copy.WithSelf.Op.type => constants.%Copy.WithSelf.Op.type.514
// CHECK:STDOUT:   %.loc14_12.2 => constants.%.237
// CHECK:STDOUT:   %impl.elem0.loc14_12.2 => constants.%ptr.as.Copy.impl.Op.cbf
// CHECK:STDOUT:   %specific_impl_fn.loc14_12.2 => constants.%ptr.as.Copy.impl.Op.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:

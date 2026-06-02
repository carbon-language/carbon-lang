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
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %A.type: type = generic_class_type @A [concrete]
// CHECK:STDOUT:   %A.generic: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %U.67d: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %I.type.48c: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.48c = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.50feb0.1: type = facet_type <@I, @I(%U.67d)> [symbolic]
// CHECK:STDOUT:   %Self.6263de.1: %I.type.50feb0.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %V.67d: type = symbolic_binding V, 0 [symbolic]
// CHECK:STDOUT:   %A.6eda0e.2: type = class_type @A, @A(%V.67d) [symbolic]
// CHECK:STDOUT:   %I.type.50feb0.2: type = facet_type <@I, @I(%V.67d)> [symbolic]
// CHECK:STDOUT:   %I.impl_witness.580224.1: <witness> = impl_witness @A.as.I.impl.%I.impl_witness_table, @A.as.I.impl(%V.67d) [symbolic]
// CHECK:STDOUT:   %require_complete.df9dcc.1: <witness> = require_complete_type %I.type.50feb0.2 [symbolic]
// CHECK:STDOUT:   %pattern_type.2e18a7.1: type = pattern_type %A.6eda0e.2 [symbolic]
// CHECK:STDOUT:   %ptr.e8f8f9.2: type = ptr_type %V.67d [symbolic]
// CHECK:STDOUT:   %.cb6cb9.2: Core.Form = init_form %ptr.e8f8f9.2 [symbolic]
// CHECK:STDOUT:   %pattern_type.4f4b84.2: type = pattern_type %ptr.e8f8f9.2 [symbolic]
// CHECK:STDOUT:   %A.as.I.impl.F.type.f4acbf.1: type = fn_type @A.as.I.impl.F, @A.as.I.impl(%V.67d) [symbolic]
// CHECK:STDOUT:   %A.as.I.impl.F.91160c.1: %A.as.I.impl.F.type.f4acbf.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %A.elem.e8eca3.2: type = unbound_element_type %A.6eda0e.2, %V.67d [symbolic]
// CHECK:STDOUT:   %require_complete.4ad: <witness> = require_complete_type %A.6eda0e.2 [symbolic]
// CHECK:STDOUT:   %require_complete.ef162c.1: <witness> = require_complete_type %ptr.e8f8f9.2 [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.aa424f.1: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.5a1c66.1: %ptr.as.Copy.impl.Op.type.aa424f.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.2bf: <witness> = lookup_impl_witness %ptr.e8f8f9.2, @Copy [symbolic]
// CHECK:STDOUT:   %.b6a: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%V.67d) [symbolic]
// CHECK:STDOUT:   %Copy.facet.44d: %Copy.type = facet_value %ptr.e8f8f9.2, (%Copy.lookup_impl_witness.2bf) [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.72b: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.44d) [symbolic]
// CHECK:STDOUT:   %.39b: type = fn_type_with_self_type %Copy.WithSelf.Op.type.72b, %Copy.facet.44d [symbolic]
// CHECK:STDOUT:   %impl.elem0.8a3: %.39b = impl_witness_access %Copy.lookup_impl_witness.2bf, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.a13: <specific function> = specific_impl_function %impl.elem0.8a3, @Copy.WithSelf.Op(%Copy.facet.44d) [symbolic]
// CHECK:STDOUT:   %W: type = symbolic_binding W, 0 [symbolic]
// CHECK:STDOUT:   %A.6eda0e.3: type = class_type @A, @A(%W) [symbolic]
// CHECK:STDOUT:   %ptr.670: type = ptr_type %A.6eda0e.3 [symbolic]
// CHECK:STDOUT:   %pattern_type.aa5: type = pattern_type %ptr.670 [symbolic]
// CHECK:STDOUT:   %ptr.e8f8f9.4: type = ptr_type %W [symbolic]
// CHECK:STDOUT:   %.cb6cb9.4: Core.Form = init_form %ptr.e8f8f9.4 [symbolic]
// CHECK:STDOUT:   %pattern_type.4f4b84.4: type = pattern_type %ptr.e8f8f9.4 [symbolic]
// CHECK:STDOUT:   %I.type.50feb0.3: type = facet_type <@I, @I(%W)> [symbolic]
// CHECK:STDOUT:   %I.assoc_type.bd7023.3: type = assoc_entity_type @I, @I(%W) [symbolic]
// CHECK:STDOUT:   %assoc0.631ba7.3: %I.assoc_type.bd7023.3 = assoc_entity element0, @I.WithSelf.%I.WithSelf.F.decl [symbolic]
// CHECK:STDOUT:   %require_complete.df9dcc.2: <witness> = require_complete_type %I.type.50feb0.3 [symbolic]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %A.6eda0e.3, @I, @I(%W) [symbolic]
// CHECK:STDOUT:   %I.impl_witness.580224.2: <witness> = impl_witness @A.as.I.impl.%I.impl_witness_table, @A.as.I.impl(%W) [symbolic]
// CHECK:STDOUT:   %A.as.I.impl.F.type.f4acbf.2: type = fn_type @A.as.I.impl.F, @A.as.I.impl(%W) [symbolic]
// CHECK:STDOUT:   %A.as.I.impl.F.91160c.2: %A.as.I.impl.F.type.f4acbf.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %.f9e: require_specific_def_type = require_specific_def @A.as.I.impl(%W) [symbolic]
// CHECK:STDOUT:   %I.facet.42f: %I.type.50feb0.3 = facet_value %A.6eda0e.3, (%I.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type.125: type = fn_type @I.WithSelf.F, @I.WithSelf(%W, %I.facet.42f) [symbolic]
// CHECK:STDOUT:   %.00e: type = fn_type_with_self_type %I.WithSelf.F.type.125, %I.facet.42f [symbolic]
// CHECK:STDOUT:   %impl.elem0.ac1: %.00e = impl_witness_access %I.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.4e2: <specific function> = specific_impl_function %impl.elem0.ac1, @I.WithSelf.F(%W, %I.facet.42f) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %A.966: type = class_type @A, @A(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %ptr.f6a: type = ptr_type %A.966 [concrete]
// CHECK:STDOUT:   %ptr.c28: type = ptr_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %.9ba: Core.Form = init_form %ptr.c28 [concrete]
// CHECK:STDOUT:   %pattern_type.1cc: type = pattern_type %ptr.c28 [concrete]
// CHECK:STDOUT:   %I.type.317: type = facet_type <@I, @I(%empty_struct_type)> [concrete]
// CHECK:STDOUT:   %I.assoc_type.4a0: type = assoc_entity_type @I, @I(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %assoc0.379: %I.assoc_type.4a0 = assoc_entity element0, @I.WithSelf.%I.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %I.impl_witness.865: <witness> = impl_witness @A.as.I.impl.%I.impl_witness_table, @A.as.I.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %complete_type.d9a: <witness> = complete_type_witness %I.type.317 [concrete]
// CHECK:STDOUT:   %A.as.I.impl.F.type.0a1: type = fn_type @A.as.I.impl.F, @A.as.I.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %A.as.I.impl.F.36b: %A.as.I.impl.F.type.0a1 = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet.7fa: %I.type.317 = facet_value %A.966, (%I.impl_witness.865) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.type.69a: type = fn_type @I.WithSelf.F, @I.WithSelf(%empty_struct_type, %I.facet.7fa) [concrete]
// CHECK:STDOUT:   %.329: type = fn_type_with_self_type %I.WithSelf.F.type.69a, %I.facet.7fa [concrete]
// CHECK:STDOUT:   %pattern_type.53f: type = pattern_type %A.966 [concrete]
// CHECK:STDOUT:   %A.as.I.impl.F.specific_fn: <specific function> = specific_function %A.as.I.impl.F.36b, @A.as.I.impl.F(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %A.elem.71e: type = unbound_element_type %A.966, %empty_struct_type [concrete]
// CHECK:STDOUT:   %struct_type.n.91c: type = struct_type {.n: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %complete_type.0a6: <witness> = complete_type_witness %struct_type.n.91c [concrete]
// CHECK:STDOUT:   %complete_type.38e: <witness> = complete_type_witness %ptr.c28 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.1c0: <witness> = impl_witness imports.%Copy.impl_witness_table.6b9, @ptr.as.Copy.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.91f: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.d7b: %ptr.as.Copy.impl.Op.type.91f = struct_value () [concrete]
// CHECK:STDOUT:   %.41e: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %Copy.facet.e29: %Copy.type = facet_value %ptr.c28, (%Copy.impl_witness.1c0) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.980: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.e29) [concrete]
// CHECK:STDOUT:   %.f8c: type = fn_type_with_self_type %Copy.WithSelf.Op.type.980, %Copy.facet.e29 [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.d7b, @ptr.as.Copy.impl.Op(%empty_struct_type) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.aa4: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.aa424f.1) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.5a1c66.1)]
// CHECK:STDOUT:   %Copy.impl_witness_table.6b9 = impl_witness_table (%Core.import_ref.aa4), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   impl_decl @A.as.I.impl [concrete] {
// CHECK:STDOUT:     %V.patt: %pattern_type.98f = symbolic_binding_pattern V, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A.ref: %A.type = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:     %V.ref.loc12_26: type = name_ref V, %V.loc12_15.2 [symbolic = %V.loc12_15.1 (constants.%V.67d)]
// CHECK:STDOUT:     %A.loc12_27.2: type = class_type @A, @A(constants.%V.67d) [symbolic = %A.loc12_27.1 (constants.%A.6eda0e.2)]
// CHECK:STDOUT:     %I.ref: %I.type.48c = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:     %V.ref.loc12_34: type = name_ref V, %V.loc12_15.2 [symbolic = %V.loc12_15.1 (constants.%V.67d)]
// CHECK:STDOUT:     %I.type.loc12_35.2: type = facet_type <@I, @I(constants.%V.67d)> [symbolic = %I.type.loc12_35.1 (constants.%I.type.50feb0.2)]
// CHECK:STDOUT:     %.loc12_18.1: type = splice_block %.loc12_18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc12_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %V.loc12_15.2: type = symbolic_binding V, 0 [symbolic = %V.loc12_15.1 (constants.%V.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @A.as.I.impl(%V.loc12_15.2: type) {
// CHECK:STDOUT:   %V.loc12_15.1: type = symbolic_binding V, 0 [symbolic = %V.loc12_15.1 (constants.%V.67d)]
// CHECK:STDOUT:   %A.loc12_27.1: type = class_type @A, @A(%V.loc12_15.1) [symbolic = %A.loc12_27.1 (constants.%A.6eda0e.2)]
// CHECK:STDOUT:   %I.type.loc12_35.1: type = facet_type <@I, @I(%V.loc12_15.1)> [symbolic = %I.type.loc12_35.1 (constants.%I.type.50feb0.2)]
// CHECK:STDOUT:   %I.impl_witness.loc12_37.2: <witness> = impl_witness %I.impl_witness_table, @A.as.I.impl(%V.loc12_15.1) [symbolic = %I.impl_witness.loc12_37.2 (constants.%I.impl_witness.580224.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.type.loc12_35.1 [symbolic = %require_complete (constants.%require_complete.df9dcc.1)]
// CHECK:STDOUT:   %A.as.I.impl.F.type: type = fn_type @A.as.I.impl.F, @A.as.I.impl(%V.loc12_15.1) [symbolic = %A.as.I.impl.F.type (constants.%A.as.I.impl.F.type.f4acbf.1)]
// CHECK:STDOUT:   %A.as.I.impl.F: @A.as.I.impl.%A.as.I.impl.F.type (%A.as.I.impl.F.type.f4acbf.1) = struct_value () [symbolic = %A.as.I.impl.F (constants.%A.as.I.impl.F.91160c.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %A.loc12_27.2 as %I.type.loc12_35.2 {
// CHECK:STDOUT:     %A.as.I.impl.F.decl: @A.as.I.impl.%A.as.I.impl.F.type (%A.as.I.impl.F.type.f4acbf.1) = fn_decl @A.as.I.impl.F [symbolic = @A.as.I.impl.%A.as.I.impl.F (constants.%A.as.I.impl.F.91160c.1)] {
// CHECK:STDOUT:       %self.param_patt: @A.as.I.impl.F.%pattern_type.loc13_16 (%pattern_type.2e18a7.1) = ref_param_pattern [concrete]
// CHECK:STDOUT:       %self.patt: @A.as.I.impl.F.%pattern_type.loc13_16 (%pattern_type.2e18a7.1) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:       %return.param_patt: @A.as.I.impl.F.%pattern_type.loc13_30 (%pattern_type.4f4b84.2) = out_param_pattern [concrete]
// CHECK:STDOUT:       %return.patt: @A.as.I.impl.F.%pattern_type.loc13_30 (%pattern_type.4f4b84.2) = return_slot_pattern %return.param_patt, %ptr.loc13_30.2 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %V.ref: type = name_ref V, @A.as.I.impl.%V.loc12_15.2 [symbolic = %V (constants.%V.67d)]
// CHECK:STDOUT:       %ptr.loc13_30.2: type = ptr_type %V.ref [symbolic = %ptr.loc13_30.1 (constants.%ptr.e8f8f9.2)]
// CHECK:STDOUT:       %.loc13_30.2: Core.Form = init_form %ptr.loc13_30.2 [symbolic = %.loc13_30.1 (constants.%.cb6cb9.2)]
// CHECK:STDOUT:       %self.param: ref @A.as.I.impl.F.%A (%A.6eda0e.2) = ref_param call_param0
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, @A.as.I.impl.%A.loc12_27.2 [symbolic = %A (constants.%A.6eda0e.2)]
// CHECK:STDOUT:       %self: ref @A.as.I.impl.F.%A (%A.6eda0e.2) = ref_binding self, %self.param
// CHECK:STDOUT:       %return.param: ref @A.as.I.impl.F.%ptr.loc13_30.1 (%ptr.e8f8f9.2) = out_param call_param1
// CHECK:STDOUT:       %return: ref @A.as.I.impl.F.%ptr.loc13_30.1 (%ptr.e8f8f9.2) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %I.impl_witness_table = impl_witness_table (%A.as.I.impl.F.decl), @A.as.I.impl [concrete]
// CHECK:STDOUT:     %I.impl_witness.loc12_37.1: <witness> = impl_witness %I.impl_witness_table, @A.as.I.impl(constants.%V.67d) [symbolic = %I.impl_witness.loc12_37.2 (constants.%I.impl_witness.580224.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .V = <poisoned>
// CHECK:STDOUT:     .F = %A.as.I.impl.F.decl
// CHECK:STDOUT:     extend %I.type.loc12_35.2
// CHECK:STDOUT:     witness = %I.impl_witness.loc12_37.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @A.as.I.impl.F(@A.as.I.impl.%V.loc12_15.2: type) {
// CHECK:STDOUT:   %V: type = symbolic_binding V, 0 [symbolic = %V (constants.%V.67d)]
// CHECK:STDOUT:   %A: type = class_type @A, @A(%V) [symbolic = %A (constants.%A.6eda0e.2)]
// CHECK:STDOUT:   %pattern_type.loc13_16: type = pattern_type %A [symbolic = %pattern_type.loc13_16 (constants.%pattern_type.2e18a7.1)]
// CHECK:STDOUT:   %ptr.loc13_30.1: type = ptr_type %V [symbolic = %ptr.loc13_30.1 (constants.%ptr.e8f8f9.2)]
// CHECK:STDOUT:   %.loc13_30.1: Core.Form = init_form %ptr.loc13_30.1 [symbolic = %.loc13_30.1 (constants.%.cb6cb9.2)]
// CHECK:STDOUT:   %pattern_type.loc13_30: type = pattern_type %ptr.loc13_30.1 [symbolic = %pattern_type.loc13_30 (constants.%pattern_type.4f4b84.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc13_16: <witness> = require_complete_type %A [symbolic = %require_complete.loc13_16 (constants.%require_complete.4ad)]
// CHECK:STDOUT:   %require_complete.loc13_26: <witness> = require_complete_type %ptr.loc13_30.1 [symbolic = %require_complete.loc13_26 (constants.%require_complete.ef162c.1)]
// CHECK:STDOUT:   %A.elem: type = unbound_element_type %A, %V [symbolic = %A.elem (constants.%A.elem.e8eca3.2)]
// CHECK:STDOUT:   %.loc14_12.1: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%V) [symbolic = %.loc14_12.1 (constants.%.b6a)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr.loc13_30.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.2bf)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.loc13_30.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet.44d)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.72b)]
// CHECK:STDOUT:   %.loc14_12.2: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %Copy.facet [symbolic = %.loc14_12.2 (constants.%.39b)]
// CHECK:STDOUT:   %impl.elem0.loc14_12.2: @A.as.I.impl.F.%.loc14_12.2 (%.39b) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc14_12.2 (constants.%impl.elem0.8a3)]
// CHECK:STDOUT:   %specific_impl_fn.loc14_12.2: <specific function> = specific_impl_function %impl.elem0.loc14_12.2, @Copy.WithSelf.Op(%Copy.facet) [symbolic = %specific_impl_fn.loc14_12.2 (constants.%specific_impl_fn.a13)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: ref @A.as.I.impl.F.%A (%A.6eda0e.2)) -> out %return.param: @A.as.I.impl.F.%ptr.loc13_30.1 (%ptr.e8f8f9.2) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: ref @A.as.I.impl.F.%A (%A.6eda0e.2) = name_ref self, %self
// CHECK:STDOUT:     %n.ref: @A.as.I.impl.F.%A.elem (%A.elem.e8eca3.2) = name_ref n, @A.%.loc4 [concrete = @A.%.loc4]
// CHECK:STDOUT:     %.loc14_17: ref @A.as.I.impl.F.%V (%V.67d) = class_element_access %self.ref, element0
// CHECK:STDOUT:     %addr: @A.as.I.impl.F.%ptr.loc13_30.1 (%ptr.e8f8f9.2) = addr_of %.loc14_17
// CHECK:STDOUT:     %impl.elem0.loc14_12.1: @A.as.I.impl.F.%.loc14_12.2 (%.39b) = impl_witness_access constants.%Copy.lookup_impl_witness.2bf, element0 [symbolic = %impl.elem0.loc14_12.2 (constants.%impl.elem0.8a3)]
// CHECK:STDOUT:     %bound_method.loc14_12.1: <bound method> = bound_method %addr, %impl.elem0.loc14_12.1
// CHECK:STDOUT:     %specific_impl_fn.loc14_12.1: <specific function> = specific_impl_function %impl.elem0.loc14_12.1, @Copy.WithSelf.Op(constants.%Copy.facet.44d) [symbolic = %specific_impl_fn.loc14_12.2 (constants.%specific_impl_fn.a13)]
// CHECK:STDOUT:     %bound_method.loc14_12.2: <bound method> = bound_method %addr, %specific_impl_fn.loc14_12.1
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @A.as.I.impl.F.%ptr.loc13_30.1 (%ptr.e8f8f9.2) = call %bound_method.loc14_12.2(%addr)
// CHECK:STDOUT:     return %Copy.WithSelf.Op.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestGeneric(%W.loc19_17.2: type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %I.type.loc21_17.2: type = facet_type <@I, @I(%W.loc19_17.1)> [symbolic = %I.type.loc21_17.2 (constants.%I.type.50feb0.3)]
// CHECK:STDOUT:   %require_complete.loc21: <witness> = require_complete_type %I.type.loc21_17.2 [symbolic = %require_complete.loc21 (constants.%require_complete.df9dcc.2)]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I, @I(%W.loc19_17.1) [symbolic = %I.assoc_type (constants.%I.assoc_type.bd7023.3)]
// CHECK:STDOUT:   %assoc0: @TestGeneric.%I.assoc_type (%I.assoc_type.bd7023.3) = assoc_entity element0, @I.WithSelf.%I.WithSelf.F.decl [symbolic = %assoc0 (constants.%assoc0.631ba7.3)]
// CHECK:STDOUT:   %.loc21_11.2: require_specific_def_type = require_specific_def @A.as.I.impl(%W.loc19_17.1) [symbolic = %.loc21_11.2 (constants.%.f9e)]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %A.loc19_32.1, @I, @I(%W.loc19_17.1) [symbolic = %I.lookup_impl_witness (constants.%I.lookup_impl_witness)]
// CHECK:STDOUT:   %I.facet: @TestGeneric.%I.type.loc21_17.2 (%I.type.50feb0.3) = facet_value %A.loc19_32.1, (%I.lookup_impl_witness) [symbolic = %I.facet (constants.%I.facet.42f)]
// CHECK:STDOUT:   %I.WithSelf.F.type: type = fn_type @I.WithSelf.F, @I.WithSelf(%W.loc19_17.1, %I.facet) [symbolic = %I.WithSelf.F.type (constants.%I.WithSelf.F.type.125)]
// CHECK:STDOUT:   %.loc21_11.3: type = fn_type_with_self_type %I.WithSelf.F.type, %I.facet [symbolic = %.loc21_11.3 (constants.%.00e)]
// CHECK:STDOUT:   %impl.elem0.loc21_11.2: @TestGeneric.%.loc21_11.3 (%.00e) = impl_witness_access %I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc21_11.2 (constants.%impl.elem0.ac1)]
// CHECK:STDOUT:   %specific_impl_fn.loc21_11.2: <specific function> = specific_impl_function %impl.elem0.loc21_11.2, @I.WithSelf.F(%W.loc19_17.1, %I.facet) [symbolic = %specific_impl_fn.loc21_11.2 (constants.%specific_impl_fn.4e2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @TestGeneric.%ptr.loc19_33.1 (%ptr.670)) -> out %return.param: @TestGeneric.%ptr.loc19_40.1 (%ptr.e8f8f9.4) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %a.ref: @TestGeneric.%ptr.loc19_33.1 (%ptr.670) = name_ref a, %a
// CHECK:STDOUT:     %I.ref: %I.type.48c = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:     %W.ref.loc21: type = name_ref W, %W.loc19_17.2 [symbolic = %W.loc19_17.1 (constants.%W)]
// CHECK:STDOUT:     %I.type.loc21_17.1: type = facet_type <@I, @I(constants.%W)> [symbolic = %I.type.loc21_17.2 (constants.%I.type.50feb0.3)]
// CHECK:STDOUT:     %.loc21_18: @TestGeneric.%I.assoc_type (%I.assoc_type.bd7023.3) = specific_constant @I.WithSelf.%assoc0.loc8_31.1, @I.WithSelf(constants.%W, constants.%Self.6263de.1) [symbolic = %assoc0 (constants.%assoc0.631ba7.3)]
// CHECK:STDOUT:     %F.ref: @TestGeneric.%I.assoc_type (%I.assoc_type.bd7023.3) = name_ref F, %.loc21_18 [symbolic = %assoc0 (constants.%assoc0.631ba7.3)]
// CHECK:STDOUT:     %.loc21_11.1: ref @TestGeneric.%A.loc19_32.1 (%A.6eda0e.3) = deref %a.ref
// CHECK:STDOUT:     %impl.elem0.loc21_11.1: @TestGeneric.%.loc21_11.3 (%.00e) = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc21_11.2 (constants.%impl.elem0.ac1)]
// CHECK:STDOUT:     %bound_method.loc21_11: <bound method> = bound_method %.loc21_11.1, %impl.elem0.loc21_11.1
// CHECK:STDOUT:     %specific_impl_fn.loc21_11.1: <specific function> = specific_impl_function %impl.elem0.loc21_11.1, @I.WithSelf.F(constants.%W, constants.%I.facet.42f) [symbolic = %specific_impl_fn.loc21_11.2 (constants.%specific_impl_fn.4e2)]
// CHECK:STDOUT:     %bound_method.loc21_22: <bound method> = bound_method %.loc21_11.1, %specific_impl_fn.loc21_11.1
// CHECK:STDOUT:     %I.WithSelf.F.call: init @TestGeneric.%ptr.loc19_40.1 (%ptr.e8f8f9.4) = call %bound_method.loc21_22(%.loc21_11.1)
// CHECK:STDOUT:     return %I.WithSelf.F.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestSpecific(%a.param: %ptr.f6a) -> out %return.param: %ptr.c28 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %ptr.f6a = name_ref a, %a
// CHECK:STDOUT:   %I.ref: %I.type.48c = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:   %.loc27_17: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc27_18: type = converted %.loc27_17, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(constants.%empty_struct_type)> [concrete = constants.%I.type.317]
// CHECK:STDOUT:   %.loc27_19: %I.assoc_type.4a0 = specific_constant @I.WithSelf.%assoc0.loc8_31.1, @I.WithSelf(constants.%empty_struct_type, constants.%Self.6263de.1) [concrete = constants.%assoc0.379]
// CHECK:STDOUT:   %F.ref: %I.assoc_type.4a0 = name_ref F, %.loc27_19 [concrete = constants.%assoc0.379]
// CHECK:STDOUT:   %.loc27_11: ref %A.966 = deref %a.ref
// CHECK:STDOUT:   %impl.elem0: %.329 = impl_witness_access constants.%I.impl_witness.865, element0 [concrete = constants.%A.as.I.impl.F.36b]
// CHECK:STDOUT:   %bound_method.loc27_11: <bound method> = bound_method %.loc27_11, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @A.as.I.impl.F(constants.%empty_struct_type) [concrete = constants.%A.as.I.impl.F.specific_fn]
// CHECK:STDOUT:   %bound_method.loc27_23: <bound method> = bound_method %.loc27_11, %specific_fn
// CHECK:STDOUT:   %A.as.I.impl.F.call: init %ptr.c28 = call %bound_method.loc27_23(%.loc27_11)
// CHECK:STDOUT:   return %A.as.I.impl.F.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.as.I.impl(constants.%V.67d) {
// CHECK:STDOUT:   %V.loc12_15.1 => constants.%V.67d
// CHECK:STDOUT:   %A.loc12_27.1 => constants.%A.6eda0e.2
// CHECK:STDOUT:   %I.type.loc12_35.1 => constants.%I.type.50feb0.2
// CHECK:STDOUT:   %I.impl_witness.loc12_37.2 => constants.%I.impl_witness.580224.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.df9dcc.1
// CHECK:STDOUT:   %A.as.I.impl.F.type => constants.%A.as.I.impl.F.type.f4acbf.1
// CHECK:STDOUT:   %A.as.I.impl.F => constants.%A.as.I.impl.F.91160c.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.as.I.impl.F(constants.%V.67d) {
// CHECK:STDOUT:   %V => constants.%V.67d
// CHECK:STDOUT:   %A => constants.%A.6eda0e.2
// CHECK:STDOUT:   %pattern_type.loc13_16 => constants.%pattern_type.2e18a7.1
// CHECK:STDOUT:   %ptr.loc13_30.1 => constants.%ptr.e8f8f9.2
// CHECK:STDOUT:   %.loc13_30.1 => constants.%.cb6cb9.2
// CHECK:STDOUT:   %pattern_type.loc13_30 => constants.%pattern_type.4f4b84.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGeneric(constants.%W) {
// CHECK:STDOUT:   %W.loc19_17.1 => constants.%W
// CHECK:STDOUT:   %A.loc19_32.1 => constants.%A.6eda0e.3
// CHECK:STDOUT:   %ptr.loc19_33.1 => constants.%ptr.670
// CHECK:STDOUT:   %pattern_type.loc19_27 => constants.%pattern_type.aa5
// CHECK:STDOUT:   %ptr.loc19_40.1 => constants.%ptr.e8f8f9.4
// CHECK:STDOUT:   %.loc19_40.1 => constants.%.cb6cb9.4
// CHECK:STDOUT:   %pattern_type.loc19_40 => constants.%pattern_type.4f4b84.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.as.I.impl(constants.%W) {
// CHECK:STDOUT:   %V.loc12_15.1 => constants.%W
// CHECK:STDOUT:   %A.loc12_27.1 => constants.%A.6eda0e.3
// CHECK:STDOUT:   %I.type.loc12_35.1 => constants.%I.type.50feb0.3
// CHECK:STDOUT:   %I.impl_witness.loc12_37.2 => constants.%I.impl_witness.580224.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.df9dcc.2
// CHECK:STDOUT:   %A.as.I.impl.F.type => constants.%A.as.I.impl.F.type.f4acbf.2
// CHECK:STDOUT:   %A.as.I.impl.F => constants.%A.as.I.impl.F.91160c.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.as.I.impl(constants.%empty_struct_type) {
// CHECK:STDOUT:   %V.loc12_15.1 => constants.%empty_struct_type
// CHECK:STDOUT:   %A.loc12_27.1 => constants.%A.966
// CHECK:STDOUT:   %I.type.loc12_35.1 => constants.%I.type.317
// CHECK:STDOUT:   %I.impl_witness.loc12_37.2 => constants.%I.impl_witness.865
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.d9a
// CHECK:STDOUT:   %A.as.I.impl.F.type => constants.%A.as.I.impl.F.type.0a1
// CHECK:STDOUT:   %A.as.I.impl.F => constants.%A.as.I.impl.F.36b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.as.I.impl.F(constants.%empty_struct_type) {
// CHECK:STDOUT:   %V => constants.%empty_struct_type
// CHECK:STDOUT:   %A => constants.%A.966
// CHECK:STDOUT:   %pattern_type.loc13_16 => constants.%pattern_type.53f
// CHECK:STDOUT:   %ptr.loc13_30.1 => constants.%ptr.c28
// CHECK:STDOUT:   %.loc13_30.1 => constants.%.9ba
// CHECK:STDOUT:   %pattern_type.loc13_30 => constants.%pattern_type.1cc
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc13_16 => constants.%complete_type.0a6
// CHECK:STDOUT:   %require_complete.loc13_26 => constants.%complete_type.38e
// CHECK:STDOUT:   %A.elem => constants.%A.elem.71e
// CHECK:STDOUT:   %.loc14_12.1 => constants.%.41e
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.impl_witness.1c0
// CHECK:STDOUT:   %Copy.facet => constants.%Copy.facet.e29
// CHECK:STDOUT:   %Copy.WithSelf.Op.type => constants.%Copy.WithSelf.Op.type.980
// CHECK:STDOUT:   %.loc14_12.2 => constants.%.f8c
// CHECK:STDOUT:   %impl.elem0.loc14_12.2 => constants.%ptr.as.Copy.impl.Op.d7b
// CHECK:STDOUT:   %specific_impl_fn.loc14_12.2 => constants.%ptr.as.Copy.impl.Op.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:

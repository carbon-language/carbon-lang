// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/overloaded/sub.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/overloaded/sub.carbon

// This file was generated from binary_op.carbon.tmpl. Run make_tests.sh to regenerate.

package User;

class C {};

impl C as Core.SubWith(C) where .Result = C {
  fn Op(unused self: C, unused other: C) -> C {
    return {};
  }
}
impl C as Core.SubAssignWith(C) {
  fn Op(unused ref self: C, unused other: C) {}
}

fn TestOp(a: C, b: C) -> C {
  //@dump-sem-ir-begin
  return a - b;
  //@dump-sem-ir-end
}

fn TestAssign(a: C*, b: C) {
  //@dump-sem-ir-begin
  *a -= b;
  //@dump-sem-ir-end
}

class D {};

impl D as Core.Sub {
  fn Op(unused self: D, unused other: D) -> D {
    return {};
  }
}
impl D as Core.SubAssign {
  fn Op(unused ref self: D, unused other: D) {}
}

fn TestNamedConstraintOp(a: D, b: D) -> D {
  //@dump-sem-ir-begin
  return a - b;
  //@dump-sem-ir-end
}

fn TestNamedConstraintAssign(a: D*, b: D) {
  //@dump-sem-ir-begin
  *a -= b;
  //@dump-sem-ir-end
}

fn TestGenericOp[T: Core.Sub](a: T, b: T) -> T {
  //@dump-sem-ir-begin
  return a - b;
  //@dump-sem-ir-end
}

fn TestGenericAssign[T: Core.SubAssign](a: T*, b: T) {
  //@dump-sem-ir-begin
  *a -= b;
  //@dump-sem-ir-end
}

fn CallGeneric(c: C, cp: C*, d: D, dp: D*, i: i32, ip: i32*) {
  let _: C = TestGenericOp(c, c);
  TestGenericAssign(cp, c);
  let _: D = TestGenericOp(d, d);
  TestGenericAssign(dp, d);
  let _: i32 = TestGenericOp(i, i);
  TestGenericAssign(ip, i);
}

// CHECK:STDOUT: --- sub.carbon = $sub
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Other: type = symbolic_binding Other, 0 [symbolic]
// CHECK:STDOUT:   %SubWith.type.73b: type = facet_type <@SubWith, @SubWith(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.18a: %SubWith.type.73b = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %SubWith.assoc_type.300: type = assoc_entity_type @SubWith, @SubWith(%Other) [symbolic]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.028: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%Other, %Self.18a) [symbolic]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.e84: %SubWith.WithSelf.Op.type.028 = struct_value () [symbolic]
// CHECK:STDOUT:   %SubWith.type.605: type = facet_type <@SubWith, @SubWith(%C)> [concrete]
// CHECK:STDOUT:   %SubWith.assoc_type.4d9: type = assoc_entity_type @SubWith, @SubWith(%C) [concrete]
// CHECK:STDOUT:   %assoc1.fb7: %SubWith.assoc_type.4d9 = assoc_entity element1, imports.%Core.import_ref.2f4 [concrete]
// CHECK:STDOUT:   %SubWith.impl_witness.5c3: <witness> = impl_witness @C.as.SubWith.impl.%SubWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.4a7: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %.79c: Core.Form = init_form %C [concrete]
// CHECK:STDOUT:   %return.param_patt.415: %pattern_type.4a7 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.340: %pattern_type.4a7 = return_slot_pattern %return.param_patt.415, %C [concrete]
// CHECK:STDOUT:   %C.as.SubWith.impl.Op.type: type = fn_type @C.as.SubWith.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.SubWith.impl.Op: %C.as.SubWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubWith.facet.aa6: %SubWith.type.605 = facet_value %C, (%SubWith.impl_witness.5c3) [concrete]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.619: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%C, %SubWith.facet.aa6) [concrete]
// CHECK:STDOUT:   %SubAssignWith.type.0c6: type = facet_type <@SubAssignWith, @SubAssignWith(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.0dd: %SubAssignWith.type.0c6 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %SubAssignWith.assoc_type.821: type = assoc_entity_type @SubAssignWith, @SubAssignWith(%Other) [symbolic]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type.7e7: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%Other, %Self.0dd) [symbolic]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.002: %SubAssignWith.WithSelf.Op.type.7e7 = struct_value () [symbolic]
// CHECK:STDOUT:   %SubAssignWith.type.9cb: type = facet_type <@SubAssignWith, @SubAssignWith(%C)> [concrete]
// CHECK:STDOUT:   %SubAssignWith.impl_witness.219: <witness> = impl_witness @C.as.SubAssignWith.impl.%SubAssignWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %SubAssignWith.assoc_type.4d8: type = assoc_entity_type @SubAssignWith, @SubAssignWith(%C) [concrete]
// CHECK:STDOUT:   %assoc0.f8e: %SubAssignWith.assoc_type.4d8 = assoc_entity element0, imports.%Core.import_ref.8e5 [concrete]
// CHECK:STDOUT:   %C.as.SubAssignWith.impl.Op.type: type = fn_type @C.as.SubAssignWith.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.SubAssignWith.impl.Op: %C.as.SubAssignWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubAssignWith.facet.346: %SubAssignWith.type.9cb = facet_value %C, (%SubAssignWith.impl_witness.219) [concrete]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type.9ed: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%C, %SubAssignWith.facet.346) [concrete]
// CHECK:STDOUT:   %a.param_patt.2a5: %pattern_type.4a7 = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt.a30: %pattern_type.4a7 = wrapper_binding_pattern a, %a.param_patt.2a5 [concrete]
// CHECK:STDOUT:   %b.param_patt.45f: %pattern_type.4a7 = value_param_pattern [concrete]
// CHECK:STDOUT:   %b.patt.562: %pattern_type.4a7 = wrapper_binding_pattern b, %b.param_patt.45f [concrete]
// CHECK:STDOUT:   %assoc1.b58: %SubWith.assoc_type.300 = assoc_entity element1, imports.%Core.import_ref.124 [symbolic]
// CHECK:STDOUT:   %.af0: type = fn_type_with_self_type %SubWith.WithSelf.Op.type.619, %SubWith.facet.aa6 [concrete]
// CHECK:STDOUT:   %ptr.00e: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.842: type = pattern_type %ptr.00e [concrete]
// CHECK:STDOUT:   %a.param_patt.4fa: %pattern_type.842 = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt.eba: %pattern_type.842 = wrapper_binding_pattern a, %a.param_patt.4fa [concrete]
// CHECK:STDOUT:   %assoc0.467: %SubAssignWith.assoc_type.821 = assoc_entity element0, imports.%Core.import_ref.91a [symbolic]
// CHECK:STDOUT:   %.7c8: type = fn_type_with_self_type %SubAssignWith.WithSelf.Op.type.9ed, %SubAssignWith.facet.346 [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %Sub.type: type = facet_type <@Sub> [concrete]
// CHECK:STDOUT:   %SubWith.impl_witness.736: <witness> = impl_witness @D.as.SubWith.impl.%SubWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.331: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %.84d: Core.Form = init_form %D [concrete]
// CHECK:STDOUT:   %return.param_patt.835: %pattern_type.331 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.5ef: %pattern_type.331 = return_slot_pattern %return.param_patt.835, %D [concrete]
// CHECK:STDOUT:   %D.as.SubWith.impl.Op.type: type = fn_type @D.as.SubWith.impl.Op [concrete]
// CHECK:STDOUT:   %D.as.SubWith.impl.Op: %D.as.SubWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubWith.type.9a5: type = facet_type <@SubWith, @SubWith(%D)> [concrete]
// CHECK:STDOUT:   %SubWith.facet.65d: %SubWith.type.9a5 = facet_value %D, (%SubWith.impl_witness.736) [concrete]
// CHECK:STDOUT:   %SubWith.assoc_type.12c: type = assoc_entity_type @SubWith, @SubWith(%D) [concrete]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.1e1: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%D, %SubWith.facet.65d) [concrete]
// CHECK:STDOUT:   %assoc1.db1: %SubWith.assoc_type.12c = assoc_entity element1, imports.%Core.import_ref.2f4 [concrete]
// CHECK:STDOUT:   %SubAssign.type: type = facet_type <@SubAssign> [concrete]
// CHECK:STDOUT:   %SubAssignWith.impl_witness.736: <witness> = impl_witness @D.as.SubAssignWith.impl.%SubAssignWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %D.as.SubAssignWith.impl.Op.type: type = fn_type @D.as.SubAssignWith.impl.Op [concrete]
// CHECK:STDOUT:   %D.as.SubAssignWith.impl.Op: %D.as.SubAssignWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubAssignWith.type.41d: type = facet_type <@SubAssignWith, @SubAssignWith(%D)> [concrete]
// CHECK:STDOUT:   %SubAssignWith.facet.ff0: %SubAssignWith.type.41d = facet_value %D, (%SubAssignWith.impl_witness.736) [concrete]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type.4c8: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%D, %SubAssignWith.facet.ff0) [concrete]
// CHECK:STDOUT:   %SubAssignWith.assoc_type.3bb: type = assoc_entity_type @SubAssignWith, @SubAssignWith(%D) [concrete]
// CHECK:STDOUT:   %assoc0.06d: %SubAssignWith.assoc_type.3bb = assoc_entity element0, imports.%Core.import_ref.8e5 [concrete]
// CHECK:STDOUT:   %a.param_patt.d0e: %pattern_type.331 = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt.ec5: %pattern_type.331 = wrapper_binding_pattern a, %a.param_patt.d0e [concrete]
// CHECK:STDOUT:   %b.param_patt.110: %pattern_type.331 = value_param_pattern [concrete]
// CHECK:STDOUT:   %b.patt.14c: %pattern_type.331 = wrapper_binding_pattern b, %b.param_patt.110 [concrete]
// CHECK:STDOUT:   %.bfd: type = fn_type_with_self_type %SubWith.WithSelf.Op.type.1e1, %SubWith.facet.65d [concrete]
// CHECK:STDOUT:   %ptr.4a5: type = ptr_type %D [concrete]
// CHECK:STDOUT:   %pattern_type.ceb: type = pattern_type %ptr.4a5 [concrete]
// CHECK:STDOUT:   %a.param_patt.609: %pattern_type.ceb = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt.fac: %pattern_type.ceb = wrapper_binding_pattern a, %a.param_patt.609 [concrete]
// CHECK:STDOUT:   %.079: type = fn_type_with_self_type %SubAssignWith.WithSelf.Op.type.4c8, %SubAssignWith.facet.ff0 [concrete]
// CHECK:STDOUT:   %pattern_type.9a587c.2: type = pattern_type %Sub.type [concrete]
// CHECK:STDOUT:   %T.patt.3a0592.1: %pattern_type.9a587c.2 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.f45530.1: %Sub.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.190bb7.1: type = facet_access_type %T.f45530.1 [symbolic]
// CHECK:STDOUT:   %pattern_type.86a9db.1: type = pattern_type %T.as_type.190bb7.1 [symbolic]
// CHECK:STDOUT:   %a.param_patt.e33: %pattern_type.86a9db.1 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %a.patt.e02: %pattern_type.86a9db.1 = wrapper_binding_pattern a, %a.param_patt.e33 [symbolic]
// CHECK:STDOUT:   %b.param_patt.4cd741.1: %pattern_type.86a9db.1 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %b.patt.60acf6.1: %pattern_type.86a9db.1 = wrapper_binding_pattern b, %b.param_patt.4cd741.1 [symbolic]
// CHECK:STDOUT:   %.1f0: Core.Form = init_form %T.as_type.190bb7.1 [symbolic]
// CHECK:STDOUT:   %return.param_patt.5f6: %pattern_type.86a9db.1 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.96a: %pattern_type.86a9db.1 = return_slot_pattern %return.param_patt.5f6, %T.as_type.190bb7.1 [symbolic]
// CHECK:STDOUT:   %SubWith.type.d7cdcb.2: type = facet_type <@SubWith, @SubWith(%T.as_type.190bb7.1)> [symbolic]
// CHECK:STDOUT:   %SubWith.assoc_type.6a64d1.2: type = assoc_entity_type @SubWith, @SubWith(%T.as_type.190bb7.1) [symbolic]
// CHECK:STDOUT:   %assoc1.3dbfe7.2: %SubWith.assoc_type.6a64d1.2 = assoc_entity element1, imports.%Core.import_ref.2f4 [symbolic]
// CHECK:STDOUT:   %require_complete.e26: <witness> = require_complete_type %SubWith.type.d7cdcb.2 [symbolic]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.2dcf1a.2: <witness> = lookup_impl_witness %T.f45530.1, @SubWith, @SubWith(%T.as_type.190bb7.1) [symbolic]
// CHECK:STDOUT:   %SubWith.facet.76b: %SubWith.type.d7cdcb.2 = facet_value %T.as_type.190bb7.1, (%SubWith.lookup_impl_witness.2dcf1a.2) [symbolic]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.6c2: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%T.as_type.190bb7.1, %SubWith.facet.76b) [symbolic]
// CHECK:STDOUT:   %.790: type = fn_type_with_self_type %SubWith.WithSelf.Op.type.6c2, %SubWith.facet.76b [symbolic]
// CHECK:STDOUT:   %impl.elem1.ee8: %.790 = impl_witness_access %SubWith.lookup_impl_witness.2dcf1a.2, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.9df: <specific function> = specific_impl_function %impl.elem1.ee8, @SubWith.WithSelf.Op(%T.as_type.190bb7.1, %SubWith.facet.76b) [symbolic]
// CHECK:STDOUT:   %pattern_type.9a587c.3: type = pattern_type %SubAssign.type [concrete]
// CHECK:STDOUT:   %T.patt.3a0592.2: %pattern_type.9a587c.3 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.f45530.2: %SubAssign.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.190bb7.2: type = facet_access_type %T.f45530.2 [symbolic]
// CHECK:STDOUT:   %ptr.360: type = ptr_type %T.as_type.190bb7.2 [symbolic]
// CHECK:STDOUT:   %pattern_type.ff5: type = pattern_type %ptr.360 [symbolic]
// CHECK:STDOUT:   %a.param_patt.f50: %pattern_type.ff5 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %a.patt.193: %pattern_type.ff5 = wrapper_binding_pattern a, %a.param_patt.f50 [symbolic]
// CHECK:STDOUT:   %pattern_type.86a9db.2: type = pattern_type %T.as_type.190bb7.2 [symbolic]
// CHECK:STDOUT:   %b.param_patt.4cd741.2: %pattern_type.86a9db.2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %b.patt.60acf6.2: %pattern_type.86a9db.2 = wrapper_binding_pattern b, %b.param_patt.4cd741.2 [symbolic]
// CHECK:STDOUT:   %SubAssignWith.type.a02458.2: type = facet_type <@SubAssignWith, @SubAssignWith(%T.as_type.190bb7.2)> [symbolic]
// CHECK:STDOUT:   %SubAssignWith.assoc_type.3cd17d.2: type = assoc_entity_type @SubAssignWith, @SubAssignWith(%T.as_type.190bb7.2) [symbolic]
// CHECK:STDOUT:   %assoc0.a46656.2: %SubAssignWith.assoc_type.3cd17d.2 = assoc_entity element0, imports.%Core.import_ref.8e5 [symbolic]
// CHECK:STDOUT:   %require_complete.96caf9.2: <witness> = require_complete_type %SubAssignWith.type.a02458.2 [symbolic]
// CHECK:STDOUT:   %SubAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.f45530.2, @SubAssignWith, @SubAssignWith(%T.as_type.190bb7.2) [symbolic]
// CHECK:STDOUT:   %SubAssignWith.facet.dc4: %SubAssignWith.type.a02458.2 = facet_value %T.as_type.190bb7.2, (%SubAssignWith.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type.51b: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%T.as_type.190bb7.2, %SubAssignWith.facet.dc4) [symbolic]
// CHECK:STDOUT:   %.b28: type = fn_type_with_self_type %SubAssignWith.WithSelf.Op.type.51b, %SubAssignWith.facet.dc4 [symbolic]
// CHECK:STDOUT:   %impl.elem0.009: %.b28 = impl_witness_access %SubAssignWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.3c4: <specific function> = specific_impl_function %impl.elem0.009, @SubAssignWith.WithSelf.Op(%T.as_type.190bb7.2, %SubAssignWith.facet.dc4) [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %Int.d4bbe0.1: type = class_type @Int, @Int(%N) [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %ptr.a17: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %ptr.a17 [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %Sub.facet.fbd: %Sub.type = facet_value %C, (%SubWith.impl_witness.5c3) [concrete]
// CHECK:STDOUT:   %SubAssign.facet.208: %SubAssign.type = facet_value %C, (%SubAssignWith.impl_witness.219) [concrete]
// CHECK:STDOUT:   %Sub.facet.6f9: %Sub.type = facet_value %D, (%SubWith.impl_witness.736) [concrete]
// CHECK:STDOUT:   %SubAssign.facet.6f9: %SubAssign.type = facet_value %D, (%SubAssignWith.impl_witness.736) [concrete]
// CHECK:STDOUT:   %Int.as.SubWith.impl.Op.type.dc6: type = fn_type @Int.as.SubWith.impl.Op.1, @Int.as.SubWith.impl.dfb(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.SubWith.impl.Op.b73: %Int.as.SubWith.impl.Op.type.dc6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.81c7d0.1: type = facet_type <@ImplicitAs, @ImplicitAs(%Int.d4bbe0.1)> [symbolic]
// CHECK:STDOUT:   %U.f16: %ImplicitAs.type.81c7d0.1 = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %SubWith.type.476: type = facet_type <@SubWith, @SubWith(%i32)> [concrete]
// CHECK:STDOUT:   %SubWith.impl_witness.344: <witness> = impl_witness imports.%SubWith.impl_witness_table.d9d, @Int.as.SubWith.impl.dfb(%int_32) [concrete]
// CHECK:STDOUT:   %SubWith.assoc_type.9b6: type = assoc_entity_type @SubWith, @SubWith(%i32) [concrete]
// CHECK:STDOUT:   %assoc1.0d5: %SubWith.assoc_type.9b6 = assoc_entity element1, imports.%Core.import_ref.2f4 [concrete]
// CHECK:STDOUT:   %complete_type.d56: <witness> = complete_type_witness %SubWith.type.476 [concrete]
// CHECK:STDOUT:   %Int.as.SubWith.impl.Op.type.3ad: type = fn_type @Int.as.SubWith.impl.Op.1, @Int.as.SubWith.impl.dfb(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.SubWith.impl.Op.7bf: %Int.as.SubWith.impl.Op.type.3ad = struct_value () [concrete]
// CHECK:STDOUT:   %Sub.facet.d11: %Sub.type = facet_value %i32, (%SubWith.impl_witness.344) [concrete]
// CHECK:STDOUT:   %a.param_patt.076: %pattern_type.d94 = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt.96f: %pattern_type.d94 = wrapper_binding_pattern a, %a.param_patt.076 [concrete]
// CHECK:STDOUT:   %b.param_patt.1a2: %pattern_type.d94 = value_param_pattern [concrete]
// CHECK:STDOUT:   %b.patt.3ca: %pattern_type.d94 = wrapper_binding_pattern b, %b.param_patt.1a2 [concrete]
// CHECK:STDOUT:   %.0b3: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.299: %pattern_type.d94 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.6fa: %pattern_type.d94 = return_slot_pattern %return.param_patt.299, %i32 [concrete]
// CHECK:STDOUT:   %Int.as.SubAssignWith.impl.Op.type.8e60c4.1: type = fn_type @Int.as.SubAssignWith.impl.Op.1, @Int.as.SubAssignWith.impl(%N, %U.f16) [symbolic]
// CHECK:STDOUT:   %Int.as.SubAssignWith.impl.Op.f08cab.1: %Int.as.SubAssignWith.impl.Op.type.8e60c4.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.6b0: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Int.d4bbe0.2: type = class_type @Int, @Int(%To) [symbolic]
// CHECK:STDOUT:   %facet_type.08a: type = facet_type <@IntFitsIn, @IntFitsIn(%Int.d4bbe0.2) & @AnyInt> [symbolic]
// CHECK:STDOUT:   %From.170: %facet_type.08a = symbolic_binding From, 1 [symbolic]
// CHECK:STDOUT:   %From.as_type.as.ImplicitAs.impl.Convert.type.386: type = fn_type @From.as_type.as.ImplicitAs.impl.Convert, @From.as_type.as.ImplicitAs.impl(%To, %From.170) [symbolic]
// CHECK:STDOUT:   %From.as_type.as.ImplicitAs.impl.Convert.85a: %From.as_type.as.ImplicitAs.impl.Convert.type.386 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.AnyInt.impl.AsInt.type.0ee: type = fn_type @Int.as.AnyInt.impl.AsInt, @Int.as.AnyInt.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.AnyInt.impl.AsInt.497: %Int.as.AnyInt.impl.AsInt.type.0ee = struct_value () [symbolic]
// CHECK:STDOUT:   %facet_type.0f0: type = facet_type <@IntFitsIn, @IntFitsIn(%i32) & @AnyInt> [concrete]
// CHECK:STDOUT:   %custom_witness.633: <witness> = custom_witness (), @IntFitsIn, @IntFitsIn(%i32) [concrete]
// CHECK:STDOUT:   %AnyInt.impl_witness.83b: <witness> = impl_witness imports.%AnyInt.impl_witness_table, @Int.as.AnyInt.impl(%int_32) [concrete]
// CHECK:STDOUT:   %facet_value: %facet_type.0f0 = facet_value %i32, (%custom_witness.633, %AnyInt.impl_witness.83b) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.943: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.8f1, @From.as_type.as.ImplicitAs.impl(%int_32, %facet_value) [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.6b0 = facet_value %i32, (%ImplicitAs.impl_witness.943) [concrete]
// CHECK:STDOUT:   %SubAssignWith.type.da8: type = facet_type <@SubAssignWith, @SubAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %SubAssignWith.impl_witness.f4a: <witness> = impl_witness imports.%SubAssignWith.impl_witness_table, @Int.as.SubAssignWith.impl(%int_32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %SubAssign.facet.b10: %SubAssign.type = facet_value %i32, (%SubAssignWith.impl_witness.f4a) [concrete]
// CHECK:STDOUT:   %a.param_patt.977: %pattern_type.7dc = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt.b70: %pattern_type.7dc = wrapper_binding_pattern a, %a.param_patt.977 [concrete]
// CHECK:STDOUT:   %complete_type.4fc: <witness> = complete_type_witness %SubWith.type.605 [concrete]
// CHECK:STDOUT:   %complete_type.a56: <witness> = complete_type_witness %ptr.00e [concrete]
// CHECK:STDOUT:   %complete_type.aaa: <witness> = complete_type_witness %SubAssignWith.type.9cb [concrete]
// CHECK:STDOUT:   %complete_type.984: <witness> = complete_type_witness %SubWith.type.9a5 [concrete]
// CHECK:STDOUT:   %complete_type.320: <witness> = complete_type_witness %ptr.4a5 [concrete]
// CHECK:STDOUT:   %complete_type.d17: <witness> = complete_type_witness %SubAssignWith.type.41d [concrete]
// CHECK:STDOUT:   %SubWith.facet.996: %SubWith.type.476 = facet_value %i32, (%SubWith.impl_witness.344) [concrete]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.74c: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%i32, %SubWith.facet.996) [concrete]
// CHECK:STDOUT:   %.896: type = fn_type_with_self_type %SubWith.WithSelf.Op.type.74c, %SubWith.facet.996 [concrete]
// CHECK:STDOUT:   %Int.as.SubWith.impl.Op.specific_fn.ae0: <specific function> = specific_function %Int.as.SubWith.impl.Op.7bf, @Int.as.SubWith.impl.Op.1(%int_32) [concrete]
// CHECK:STDOUT:   %complete_type.1b1: <witness> = complete_type_witness %ptr.a17 [concrete]
// CHECK:STDOUT:   %SubAssignWith.assoc_type.ec0: type = assoc_entity_type @SubAssignWith, @SubAssignWith(%i32) [concrete]
// CHECK:STDOUT:   %assoc0.9646: %SubAssignWith.assoc_type.ec0 = assoc_entity element0, imports.%Core.import_ref.8e5 [concrete]
// CHECK:STDOUT:   %complete_type.956: <witness> = complete_type_witness %SubAssignWith.type.da8 [concrete]
// CHECK:STDOUT:   %SubAssignWith.facet.750: %SubAssignWith.type.da8 = facet_value %i32, (%SubAssignWith.impl_witness.f4a) [concrete]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type.c33: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%i32, %SubAssignWith.facet.750) [concrete]
// CHECK:STDOUT:   %.e9d: type = fn_type_with_self_type %SubAssignWith.WithSelf.Op.type.c33, %SubAssignWith.facet.750 [concrete]
// CHECK:STDOUT:   %Int.as.SubAssignWith.impl.Op.type.6d3d4e.2: type = fn_type @Int.as.SubAssignWith.impl.Op.1, @Int.as.SubAssignWith.impl(%int_32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Int.as.SubAssignWith.impl.Op.ae2978.2: %Int.as.SubAssignWith.impl.Op.type.6d3d4e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.SubAssignWith.impl.Op.specific_fn.dc0c2b.1: <specific function> = specific_function %Int.as.SubAssignWith.impl.Op.ae2978.2, @Int.as.SubAssignWith.impl.Op.1(%int_32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.5fe: @SubWith.WithSelf.%SubWith.assoc_type (%SubWith.assoc_type.300) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @SubWith.WithSelf.%assoc1 (constants.%assoc1.b58)]
// CHECK:STDOUT:   %Core.import_ref.2f4: @SubWith.WithSelf.%SubWith.WithSelf.Op.type (%SubWith.WithSelf.Op.type.028) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @SubWith.WithSelf.%SubWith.WithSelf.Op (constants.%SubWith.WithSelf.Op.e84)]
// CHECK:STDOUT:   %Core.import_ref.1f6: @SubAssignWith.WithSelf.%SubAssignWith.assoc_type (%SubAssignWith.assoc_type.821) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @SubAssignWith.WithSelf.%assoc0 (constants.%assoc0.467)]
// CHECK:STDOUT:   %Core.import_ref.8e5: @SubAssignWith.WithSelf.%SubAssignWith.WithSelf.Op.type (%SubAssignWith.WithSelf.Op.type.7e7) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @SubAssignWith.WithSelf.%SubAssignWith.WithSelf.Op (constants.%SubAssignWith.WithSelf.Op.002)]
// CHECK:STDOUT:   %Core.import_ref.124 = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.91a = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.ebd: type = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.SubWith.impl.dfb.%Int (constants.%Int.d4bbe0.1)]
// CHECK:STDOUT:   %Core.import_ref.a82f: @Int.as.SubWith.impl.dfb.%Int.as.SubWith.impl.Op.type (%Int.as.SubWith.impl.Op.type.dc6) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.SubWith.impl.dfb.%Int.as.SubWith.impl.Op (constants.%Int.as.SubWith.impl.Op.b73)]
// CHECK:STDOUT:   %SubWith.impl_witness_table.d9d = impl_witness_table (%Core.import_ref.ebd, %Core.import_ref.a82f), @Int.as.SubWith.impl.dfb [concrete]
// CHECK:STDOUT:   %Core.import_ref.4ec: @Int.as.SubAssignWith.impl.%Int.as.SubAssignWith.impl.Op.type.2 (%Int.as.SubAssignWith.impl.Op.type.8e60c4.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.SubAssignWith.impl.%Int.as.SubAssignWith.impl.Op.2 (constants.%Int.as.SubAssignWith.impl.Op.f08cab.1)]
// CHECK:STDOUT:   %SubAssignWith.impl_witness_table = impl_witness_table (%Core.import_ref.4ec), @Int.as.SubAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.39b: @From.as_type.as.ImplicitAs.impl.%From.as_type.as.ImplicitAs.impl.Convert.type (%From.as_type.as.ImplicitAs.impl.Convert.type.386) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @From.as_type.as.ImplicitAs.impl.%From.as_type.as.ImplicitAs.impl.Convert (constants.%From.as_type.as.ImplicitAs.impl.Convert.85a)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.8f1 = impl_witness_table (%Core.import_ref.39b), @From.as_type.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.47b: Core.IntLiteral = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.AnyInt.impl.%N (constants.%N)]
// CHECK:STDOUT:   %Core.import_ref.323: @Int.as.AnyInt.impl.%Int.as.AnyInt.impl.AsInt.type (%Int.as.AnyInt.impl.AsInt.type.0ee) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.AnyInt.impl.%Int.as.AnyInt.impl.AsInt (constants.%Int.as.AnyInt.impl.AsInt.497)]
// CHECK:STDOUT:   %AnyInt.impl_witness_table = impl_witness_table (%Core.import_ref.47b, %Core.import_ref.323), @Int.as.AnyInt.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestOp(%a.param: %C, %b.param: %C) -> out %return.param: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1: %.af0 = impl_witness_access constants.%SubWith.impl_witness.5c3, element1 [concrete = constants.%C.as.SubWith.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.ref, %impl.elem1
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.as.SubWith.impl.Op.call: init %C to %.loc30_26.1 = call %bound_method(%a.ref, %b.ref)
// CHECK:STDOUT:   return %C.as.SubWith.impl.Op.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestAssign(%a.param: %ptr.00e, %b.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %ptr.00e = name_ref a, %a
// CHECK:STDOUT:   %.loc38: ref %C = deref %a.ref
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0: %.7c8 = impl_witness_access constants.%SubAssignWith.impl_witness.219, element0 [concrete = constants.%C.as.SubAssignWith.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc38, %impl.elem0
// CHECK:STDOUT:   %C.as.SubAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method(%.loc38, %b.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestNamedConstraintOp(%a.param: %D, %b.param: %D) -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %D = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %D = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1: %.bfd = impl_witness_access constants.%SubWith.impl_witness.736, element1 [concrete = constants.%D.as.SubWith.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.ref, %impl.elem1
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %D.as.SubWith.impl.Op.call: init %D to %.loc53_41.1 = call %bound_method(%a.ref, %b.ref)
// CHECK:STDOUT:   return %D.as.SubWith.impl.Op.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestNamedConstraintAssign(%a.param: %ptr.4a5, %b.param: %D) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %ptr.4a5 = name_ref a, %a
// CHECK:STDOUT:   %.loc61: ref %D = deref %a.ref
// CHECK:STDOUT:   %b.ref: %D = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0: %.079 = impl_witness_access constants.%SubAssignWith.impl_witness.736, element0 [concrete = constants.%D.as.SubAssignWith.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc61, %impl.elem0
// CHECK:STDOUT:   %D.as.SubAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method(%.loc61, %b.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestGenericOp(%T.loc65_19.2: %Sub.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SubWith.type.loc67_12.2: type = facet_type <@SubWith, @SubWith(%T.as_type.loc65_34.1)> [symbolic = %SubWith.type.loc67_12.2 (constants.%SubWith.type.d7cdcb.2)]
// CHECK:STDOUT:   %require_complete.loc67: <witness> = require_complete_type %SubWith.type.loc67_12.2 [symbolic = %require_complete.loc67 (constants.%require_complete.e26)]
// CHECK:STDOUT:   %SubWith.assoc_type: type = assoc_entity_type @SubWith, @SubWith(%T.as_type.loc65_34.1) [symbolic = %SubWith.assoc_type (constants.%SubWith.assoc_type.6a64d1.2)]
// CHECK:STDOUT:   %assoc1: @TestGenericOp.%SubWith.assoc_type (%SubWith.assoc_type.6a64d1.2) = assoc_entity element1, imports.%Core.import_ref.2f4 [symbolic = %assoc1 (constants.%assoc1.3dbfe7.2)]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc65_19.1, @SubWith, @SubWith(%T.as_type.loc65_34.1) [symbolic = %SubWith.lookup_impl_witness (constants.%SubWith.lookup_impl_witness.2dcf1a.2)]
// CHECK:STDOUT:   %SubWith.facet: @TestGenericOp.%SubWith.type.loc67_12.2 (%SubWith.type.d7cdcb.2) = facet_value %T.as_type.loc65_34.1, (%SubWith.lookup_impl_witness) [symbolic = %SubWith.facet (constants.%SubWith.facet.76b)]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%T.as_type.loc65_34.1, %SubWith.facet) [symbolic = %SubWith.WithSelf.Op.type (constants.%SubWith.WithSelf.Op.type.6c2)]
// CHECK:STDOUT:   %.loc67_12.2: type = fn_type_with_self_type %SubWith.WithSelf.Op.type, %SubWith.facet [symbolic = %.loc67_12.2 (constants.%.790)]
// CHECK:STDOUT:   %impl.elem1.loc67_12.2: @TestGenericOp.%.loc67_12.2 (%.790) = impl_witness_access %SubWith.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc67_12.2 (constants.%impl.elem1.ee8)]
// CHECK:STDOUT:   %specific_impl_fn.loc67_12.2: <specific function> = specific_impl_function %impl.elem1.loc67_12.2, @SubWith.WithSelf.Op(%T.as_type.loc65_34.1, %SubWith.facet) [symbolic = %specific_impl_fn.loc67_12.2 (constants.%specific_impl_fn.9df)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @TestGenericOp.%T.as_type.loc65_34.1 (%T.as_type.190bb7.1), %b.param: @TestGenericOp.%T.as_type.loc65_34.1 (%T.as_type.190bb7.1)) -> out %return.param: @TestGenericOp.%T.as_type.loc65_34.1 (%T.as_type.190bb7.1) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %a.ref: @TestGenericOp.%T.as_type.loc65_34.1 (%T.as_type.190bb7.1) = name_ref a, %a
// CHECK:STDOUT:     %b.ref: @TestGenericOp.%T.as_type.loc65_34.1 (%T.as_type.190bb7.1) = name_ref b, %b
// CHECK:STDOUT:     %SubWith.type.loc67_12.1: type = facet_type <@SubWith, @SubWith(constants.%T.as_type.190bb7.1)> [symbolic = %SubWith.type.loc67_12.2 (constants.%SubWith.type.d7cdcb.2)]
// CHECK:STDOUT:     %.loc67_12.1: @TestGenericOp.%SubWith.assoc_type (%SubWith.assoc_type.6a64d1.2) = specific_constant imports.%Core.import_ref.5fe, @SubWith.WithSelf(constants.%T.as_type.190bb7.1, constants.%Self.18a) [symbolic = %assoc1 (constants.%assoc1.3dbfe7.2)]
// CHECK:STDOUT:     %Op.ref: @TestGenericOp.%SubWith.assoc_type (%SubWith.assoc_type.6a64d1.2) = name_ref Op, %.loc67_12.1 [symbolic = %assoc1 (constants.%assoc1.3dbfe7.2)]
// CHECK:STDOUT:     %impl.elem1.loc67_12.1: @TestGenericOp.%.loc67_12.2 (%.790) = impl_witness_access constants.%SubWith.lookup_impl_witness.2dcf1a.2, element1 [symbolic = %impl.elem1.loc67_12.2 (constants.%impl.elem1.ee8)]
// CHECK:STDOUT:     %bound_method.loc67_12.1: <bound method> = bound_method %a.ref, %impl.elem1.loc67_12.1
// CHECK:STDOUT:     %specific_impl_fn.loc67_12.1: <specific function> = specific_impl_function %impl.elem1.loc67_12.1, @SubWith.WithSelf.Op(constants.%T.as_type.190bb7.1, constants.%SubWith.facet.76b) [symbolic = %specific_impl_fn.loc67_12.2 (constants.%specific_impl_fn.9df)]
// CHECK:STDOUT:     %bound_method.loc67_12.2: <bound method> = bound_method %a.ref, %specific_impl_fn.loc67_12.1
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SubWith.WithSelf.Op.call: init @TestGenericOp.%T.as_type.loc65_34.1 (%T.as_type.190bb7.1) to %.loc65_46.1 = call %bound_method.loc67_12.2(%a.ref, %b.ref)
// CHECK:STDOUT:     return %SubWith.WithSelf.Op.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestGenericAssign(%T.loc71_23.2: %SubAssign.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SubAssignWith.type.loc73_6.2: type = facet_type <@SubAssignWith, @SubAssignWith(%T.as_type.loc71_45.1)> [symbolic = %SubAssignWith.type.loc73_6.2 (constants.%SubAssignWith.type.a02458.2)]
// CHECK:STDOUT:   %require_complete.loc73: <witness> = require_complete_type %SubAssignWith.type.loc73_6.2 [symbolic = %require_complete.loc73 (constants.%require_complete.96caf9.2)]
// CHECK:STDOUT:   %SubAssignWith.assoc_type: type = assoc_entity_type @SubAssignWith, @SubAssignWith(%T.as_type.loc71_45.1) [symbolic = %SubAssignWith.assoc_type (constants.%SubAssignWith.assoc_type.3cd17d.2)]
// CHECK:STDOUT:   %assoc0: @TestGenericAssign.%SubAssignWith.assoc_type (%SubAssignWith.assoc_type.3cd17d.2) = assoc_entity element0, imports.%Core.import_ref.8e5 [symbolic = %assoc0 (constants.%assoc0.a46656.2)]
// CHECK:STDOUT:   %SubAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc71_23.1, @SubAssignWith, @SubAssignWith(%T.as_type.loc71_45.1) [symbolic = %SubAssignWith.lookup_impl_witness (constants.%SubAssignWith.lookup_impl_witness)]
// CHECK:STDOUT:   %SubAssignWith.facet: @TestGenericAssign.%SubAssignWith.type.loc73_6.2 (%SubAssignWith.type.a02458.2) = facet_value %T.as_type.loc71_45.1, (%SubAssignWith.lookup_impl_witness) [symbolic = %SubAssignWith.facet (constants.%SubAssignWith.facet.dc4)]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%T.as_type.loc71_45.1, %SubAssignWith.facet) [symbolic = %SubAssignWith.WithSelf.Op.type (constants.%SubAssignWith.WithSelf.Op.type.51b)]
// CHECK:STDOUT:   %.loc73_6.2: type = fn_type_with_self_type %SubAssignWith.WithSelf.Op.type, %SubAssignWith.facet [symbolic = %.loc73_6.2 (constants.%.b28)]
// CHECK:STDOUT:   %impl.elem0.loc73_6.2: @TestGenericAssign.%.loc73_6.2 (%.b28) = impl_witness_access %SubAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc73_6.2 (constants.%impl.elem0.009)]
// CHECK:STDOUT:   %specific_impl_fn.loc73_6.2: <specific function> = specific_impl_function %impl.elem0.loc73_6.2, @SubAssignWith.WithSelf.Op(%T.as_type.loc71_45.1, %SubAssignWith.facet) [symbolic = %specific_impl_fn.loc73_6.2 (constants.%specific_impl_fn.3c4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @TestGenericAssign.%ptr.loc71_45.1 (%ptr.360), %b.param: @TestGenericAssign.%T.as_type.loc71_45.1 (%T.as_type.190bb7.2)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %a.ref: @TestGenericAssign.%ptr.loc71_45.1 (%ptr.360) = name_ref a, %a
// CHECK:STDOUT:     %.loc73_3: ref @TestGenericAssign.%T.as_type.loc71_45.1 (%T.as_type.190bb7.2) = deref %a.ref
// CHECK:STDOUT:     %b.ref: @TestGenericAssign.%T.as_type.loc71_45.1 (%T.as_type.190bb7.2) = name_ref b, %b
// CHECK:STDOUT:     %SubAssignWith.type.loc73_6.1: type = facet_type <@SubAssignWith, @SubAssignWith(constants.%T.as_type.190bb7.2)> [symbolic = %SubAssignWith.type.loc73_6.2 (constants.%SubAssignWith.type.a02458.2)]
// CHECK:STDOUT:     %.loc73_6.1: @TestGenericAssign.%SubAssignWith.assoc_type (%SubAssignWith.assoc_type.3cd17d.2) = specific_constant imports.%Core.import_ref.1f6, @SubAssignWith.WithSelf(constants.%T.as_type.190bb7.2, constants.%Self.0dd) [symbolic = %assoc0 (constants.%assoc0.a46656.2)]
// CHECK:STDOUT:     %Op.ref: @TestGenericAssign.%SubAssignWith.assoc_type (%SubAssignWith.assoc_type.3cd17d.2) = name_ref Op, %.loc73_6.1 [symbolic = %assoc0 (constants.%assoc0.a46656.2)]
// CHECK:STDOUT:     %impl.elem0.loc73_6.1: @TestGenericAssign.%.loc73_6.2 (%.b28) = impl_witness_access constants.%SubAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc73_6.2 (constants.%impl.elem0.009)]
// CHECK:STDOUT:     %bound_method.loc73_6.1: <bound method> = bound_method %.loc73_3, %impl.elem0.loc73_6.1
// CHECK:STDOUT:     %specific_impl_fn.loc73_6.1: <specific function> = specific_impl_function %impl.elem0.loc73_6.1, @SubAssignWith.WithSelf.Op(constants.%T.as_type.190bb7.2, constants.%SubAssignWith.facet.dc4) [symbolic = %specific_impl_fn.loc73_6.2 (constants.%specific_impl_fn.3c4)]
// CHECK:STDOUT:     %bound_method.loc73_6.2: <bound method> = bound_method %.loc73_3, %specific_impl_fn.loc73_6.1
// CHECK:STDOUT:     %SubAssignWith.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc73_6.2(%.loc73_3, %b.ref)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGenericOp(constants.%T.f45530.1) {
// CHECK:STDOUT:   %T.patt.loc65_19.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc65_19.1 => constants.%T.f45530.1
// CHECK:STDOUT:   %T.as_type.loc65_34.1 => constants.%T.as_type.190bb7.1
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.86a9db.1
// CHECK:STDOUT:   %a.param_patt.loc65_32.2 => constants.%a.param_patt.e33
// CHECK:STDOUT:   %a.patt.loc65_32.2 => constants.%a.patt.e02
// CHECK:STDOUT:   %b.param_patt.loc65_38.2 => constants.%b.param_patt.4cd741.1
// CHECK:STDOUT:   %b.patt.loc65_38.2 => constants.%b.patt.60acf6.1
// CHECK:STDOUT:   %.loc65_46.2 => constants.%.1f0
// CHECK:STDOUT:   %return.param_patt.loc65_46.2 => constants.%return.param_patt.5f6
// CHECK:STDOUT:   %return.patt.loc65_43.2 => constants.%return.patt.96a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGenericAssign(constants.%T.f45530.2) {
// CHECK:STDOUT:   %T.patt.loc71_23.2 => constants.%T.patt.3a0592.2
// CHECK:STDOUT:   %T.loc71_23.1 => constants.%T.f45530.2
// CHECK:STDOUT:   %T.as_type.loc71_45.1 => constants.%T.as_type.190bb7.2
// CHECK:STDOUT:   %ptr.loc71_45.1 => constants.%ptr.360
// CHECK:STDOUT:   %pattern_type.loc71_42 => constants.%pattern_type.ff5
// CHECK:STDOUT:   %a.param_patt.loc71_42.2 => constants.%a.param_patt.f50
// CHECK:STDOUT:   %a.patt.loc71_42.2 => constants.%a.patt.193
// CHECK:STDOUT:   %pattern_type.loc71_49 => constants.%pattern_type.86a9db.2
// CHECK:STDOUT:   %b.param_patt.loc71_49.2 => constants.%b.param_patt.4cd741.2
// CHECK:STDOUT:   %b.patt.loc71_49.2 => constants.%b.patt.60acf6.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGenericOp(constants.%Sub.facet.fbd) {
// CHECK:STDOUT:   %T.patt.loc65_19.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc65_19.1 => constants.%Sub.facet.fbd
// CHECK:STDOUT:   %T.as_type.loc65_34.1 => constants.%C
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.4a7
// CHECK:STDOUT:   %a.param_patt.loc65_32.2 => constants.%a.param_patt.2a5
// CHECK:STDOUT:   %a.patt.loc65_32.2 => constants.%a.patt.a30
// CHECK:STDOUT:   %b.param_patt.loc65_38.2 => constants.%b.param_patt.45f
// CHECK:STDOUT:   %b.patt.loc65_38.2 => constants.%b.patt.562
// CHECK:STDOUT:   %.loc65_46.2 => constants.%.79c
// CHECK:STDOUT:   %return.param_patt.loc65_46.2 => constants.%return.param_patt.415
// CHECK:STDOUT:   %return.patt.loc65_43.2 => constants.%return.patt.340
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc65 => constants.%complete_type.357
// CHECK:STDOUT:   %SubWith.type.loc67_12.2 => constants.%SubWith.type.605
// CHECK:STDOUT:   %require_complete.loc67 => constants.%complete_type.4fc
// CHECK:STDOUT:   %SubWith.assoc_type => constants.%SubWith.assoc_type.4d9
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.fb7
// CHECK:STDOUT:   %SubWith.lookup_impl_witness => constants.%SubWith.impl_witness.5c3
// CHECK:STDOUT:   %SubWith.facet => constants.%SubWith.facet.aa6
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type => constants.%SubWith.WithSelf.Op.type.619
// CHECK:STDOUT:   %.loc67_12.2 => constants.%.af0
// CHECK:STDOUT:   %impl.elem1.loc67_12.2 => constants.%C.as.SubWith.impl.Op
// CHECK:STDOUT:   %specific_impl_fn.loc67_12.2 => constants.%C.as.SubWith.impl.Op
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGenericAssign(constants.%SubAssign.facet.208) {
// CHECK:STDOUT:   %T.patt.loc71_23.2 => constants.%T.patt.3a0592.2
// CHECK:STDOUT:   %T.loc71_23.1 => constants.%SubAssign.facet.208
// CHECK:STDOUT:   %T.as_type.loc71_45.1 => constants.%C
// CHECK:STDOUT:   %ptr.loc71_45.1 => constants.%ptr.00e
// CHECK:STDOUT:   %pattern_type.loc71_42 => constants.%pattern_type.842
// CHECK:STDOUT:   %a.param_patt.loc71_42.2 => constants.%a.param_patt.4fa
// CHECK:STDOUT:   %a.patt.loc71_42.2 => constants.%a.patt.eba
// CHECK:STDOUT:   %pattern_type.loc71_49 => constants.%pattern_type.4a7
// CHECK:STDOUT:   %b.param_patt.loc71_49.2 => constants.%b.param_patt.45f
// CHECK:STDOUT:   %b.patt.loc71_49.2 => constants.%b.patt.562
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc71_42 => constants.%complete_type.a56
// CHECK:STDOUT:   %require_complete.loc71_49 => constants.%complete_type.357
// CHECK:STDOUT:   %SubAssignWith.type.loc73_6.2 => constants.%SubAssignWith.type.9cb
// CHECK:STDOUT:   %require_complete.loc73 => constants.%complete_type.aaa
// CHECK:STDOUT:   %SubAssignWith.assoc_type => constants.%SubAssignWith.assoc_type.4d8
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.f8e
// CHECK:STDOUT:   %SubAssignWith.lookup_impl_witness => constants.%SubAssignWith.impl_witness.219
// CHECK:STDOUT:   %SubAssignWith.facet => constants.%SubAssignWith.facet.346
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type => constants.%SubAssignWith.WithSelf.Op.type.9ed
// CHECK:STDOUT:   %.loc73_6.2 => constants.%.7c8
// CHECK:STDOUT:   %impl.elem0.loc73_6.2 => constants.%C.as.SubAssignWith.impl.Op
// CHECK:STDOUT:   %specific_impl_fn.loc73_6.2 => constants.%C.as.SubAssignWith.impl.Op
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGenericOp(constants.%Sub.facet.6f9) {
// CHECK:STDOUT:   %T.patt.loc65_19.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc65_19.1 => constants.%Sub.facet.6f9
// CHECK:STDOUT:   %T.as_type.loc65_34.1 => constants.%D
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.331
// CHECK:STDOUT:   %a.param_patt.loc65_32.2 => constants.%a.param_patt.d0e
// CHECK:STDOUT:   %a.patt.loc65_32.2 => constants.%a.patt.ec5
// CHECK:STDOUT:   %b.param_patt.loc65_38.2 => constants.%b.param_patt.110
// CHECK:STDOUT:   %b.patt.loc65_38.2 => constants.%b.patt.14c
// CHECK:STDOUT:   %.loc65_46.2 => constants.%.84d
// CHECK:STDOUT:   %return.param_patt.loc65_46.2 => constants.%return.param_patt.835
// CHECK:STDOUT:   %return.patt.loc65_43.2 => constants.%return.patt.5ef
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc65 => constants.%complete_type.357
// CHECK:STDOUT:   %SubWith.type.loc67_12.2 => constants.%SubWith.type.9a5
// CHECK:STDOUT:   %require_complete.loc67 => constants.%complete_type.984
// CHECK:STDOUT:   %SubWith.assoc_type => constants.%SubWith.assoc_type.12c
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.db1
// CHECK:STDOUT:   %SubWith.lookup_impl_witness => constants.%SubWith.impl_witness.736
// CHECK:STDOUT:   %SubWith.facet => constants.%SubWith.facet.65d
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type => constants.%SubWith.WithSelf.Op.type.1e1
// CHECK:STDOUT:   %.loc67_12.2 => constants.%.bfd
// CHECK:STDOUT:   %impl.elem1.loc67_12.2 => constants.%D.as.SubWith.impl.Op
// CHECK:STDOUT:   %specific_impl_fn.loc67_12.2 => constants.%D.as.SubWith.impl.Op
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGenericAssign(constants.%SubAssign.facet.6f9) {
// CHECK:STDOUT:   %T.patt.loc71_23.2 => constants.%T.patt.3a0592.2
// CHECK:STDOUT:   %T.loc71_23.1 => constants.%SubAssign.facet.6f9
// CHECK:STDOUT:   %T.as_type.loc71_45.1 => constants.%D
// CHECK:STDOUT:   %ptr.loc71_45.1 => constants.%ptr.4a5
// CHECK:STDOUT:   %pattern_type.loc71_42 => constants.%pattern_type.ceb
// CHECK:STDOUT:   %a.param_patt.loc71_42.2 => constants.%a.param_patt.609
// CHECK:STDOUT:   %a.patt.loc71_42.2 => constants.%a.patt.fac
// CHECK:STDOUT:   %pattern_type.loc71_49 => constants.%pattern_type.331
// CHECK:STDOUT:   %b.param_patt.loc71_49.2 => constants.%b.param_patt.110
// CHECK:STDOUT:   %b.patt.loc71_49.2 => constants.%b.patt.14c
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc71_42 => constants.%complete_type.320
// CHECK:STDOUT:   %require_complete.loc71_49 => constants.%complete_type.357
// CHECK:STDOUT:   %SubAssignWith.type.loc73_6.2 => constants.%SubAssignWith.type.41d
// CHECK:STDOUT:   %require_complete.loc73 => constants.%complete_type.d17
// CHECK:STDOUT:   %SubAssignWith.assoc_type => constants.%SubAssignWith.assoc_type.3bb
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.06d
// CHECK:STDOUT:   %SubAssignWith.lookup_impl_witness => constants.%SubAssignWith.impl_witness.736
// CHECK:STDOUT:   %SubAssignWith.facet => constants.%SubAssignWith.facet.ff0
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type => constants.%SubAssignWith.WithSelf.Op.type.4c8
// CHECK:STDOUT:   %.loc73_6.2 => constants.%.079
// CHECK:STDOUT:   %impl.elem0.loc73_6.2 => constants.%D.as.SubAssignWith.impl.Op
// CHECK:STDOUT:   %specific_impl_fn.loc73_6.2 => constants.%D.as.SubAssignWith.impl.Op
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGenericOp(constants.%Sub.facet.d11) {
// CHECK:STDOUT:   %T.patt.loc65_19.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc65_19.1 => constants.%Sub.facet.d11
// CHECK:STDOUT:   %T.as_type.loc65_34.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.d94
// CHECK:STDOUT:   %a.param_patt.loc65_32.2 => constants.%a.param_patt.076
// CHECK:STDOUT:   %a.patt.loc65_32.2 => constants.%a.patt.96f
// CHECK:STDOUT:   %b.param_patt.loc65_38.2 => constants.%b.param_patt.1a2
// CHECK:STDOUT:   %b.patt.loc65_38.2 => constants.%b.patt.3ca
// CHECK:STDOUT:   %.loc65_46.2 => constants.%.0b3
// CHECK:STDOUT:   %return.param_patt.loc65_46.2 => constants.%return.param_patt.299
// CHECK:STDOUT:   %return.patt.loc65_43.2 => constants.%return.patt.6fa
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc65 => constants.%complete_type.f8a
// CHECK:STDOUT:   %SubWith.type.loc67_12.2 => constants.%SubWith.type.476
// CHECK:STDOUT:   %require_complete.loc67 => constants.%complete_type.d56
// CHECK:STDOUT:   %SubWith.assoc_type => constants.%SubWith.assoc_type.9b6
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.0d5
// CHECK:STDOUT:   %SubWith.lookup_impl_witness => constants.%SubWith.impl_witness.344
// CHECK:STDOUT:   %SubWith.facet => constants.%SubWith.facet.996
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type => constants.%SubWith.WithSelf.Op.type.74c
// CHECK:STDOUT:   %.loc67_12.2 => constants.%.896
// CHECK:STDOUT:   %impl.elem1.loc67_12.2 => constants.%Int.as.SubWith.impl.Op.7bf
// CHECK:STDOUT:   %specific_impl_fn.loc67_12.2 => constants.%Int.as.SubWith.impl.Op.specific_fn.ae0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGenericAssign(constants.%SubAssign.facet.b10) {
// CHECK:STDOUT:   %T.patt.loc71_23.2 => constants.%T.patt.3a0592.2
// CHECK:STDOUT:   %T.loc71_23.1 => constants.%SubAssign.facet.b10
// CHECK:STDOUT:   %T.as_type.loc71_45.1 => constants.%i32
// CHECK:STDOUT:   %ptr.loc71_45.1 => constants.%ptr.a17
// CHECK:STDOUT:   %pattern_type.loc71_42 => constants.%pattern_type.7dc
// CHECK:STDOUT:   %a.param_patt.loc71_42.2 => constants.%a.param_patt.977
// CHECK:STDOUT:   %a.patt.loc71_42.2 => constants.%a.patt.b70
// CHECK:STDOUT:   %pattern_type.loc71_49 => constants.%pattern_type.d94
// CHECK:STDOUT:   %b.param_patt.loc71_49.2 => constants.%b.param_patt.1a2
// CHECK:STDOUT:   %b.patt.loc71_49.2 => constants.%b.patt.3ca
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc71_42 => constants.%complete_type.1b1
// CHECK:STDOUT:   %require_complete.loc71_49 => constants.%complete_type.f8a
// CHECK:STDOUT:   %SubAssignWith.type.loc73_6.2 => constants.%SubAssignWith.type.da8
// CHECK:STDOUT:   %require_complete.loc73 => constants.%complete_type.956
// CHECK:STDOUT:   %SubAssignWith.assoc_type => constants.%SubAssignWith.assoc_type.ec0
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.9646
// CHECK:STDOUT:   %SubAssignWith.lookup_impl_witness => constants.%SubAssignWith.impl_witness.f4a
// CHECK:STDOUT:   %SubAssignWith.facet => constants.%SubAssignWith.facet.750
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type => constants.%SubAssignWith.WithSelf.Op.type.c33
// CHECK:STDOUT:   %.loc73_6.2 => constants.%.e9d
// CHECK:STDOUT:   %impl.elem0.loc73_6.2 => constants.%Int.as.SubAssignWith.impl.Op.ae2978.2
// CHECK:STDOUT:   %specific_impl_fn.loc73_6.2 => constants.%Int.as.SubAssignWith.impl.Op.specific_fn.dc0c2b.1
// CHECK:STDOUT: }
// CHECK:STDOUT:

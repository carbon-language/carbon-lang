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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/overloaded/bit_xor.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/overloaded/bit_xor.carbon

// This file was generated from binary_op.carbon.tmpl. Run make_tests.sh to regenerate.

package User;

class C {};

impl C as Core.BitXorWith(C) where .Result = C {
  fn Op(unused self: C, unused other: C) -> C {
    return {};
  }
}
impl C as Core.BitXorAssignWith(C) {
  fn Op(unused ref self: C, unused other: C) {}
}

fn TestOp(a: C, b: C) -> C {
  //@dump-sem-ir-begin
  return a ^ b;
  //@dump-sem-ir-end
}

fn TestAssign(a: C*, b: C) {
  //@dump-sem-ir-begin
  *a ^= b;
  //@dump-sem-ir-end
}

class D {};

impl D as Core.BitXor {
  fn Op(unused self: D, unused other: D) -> D {
    return {};
  }
}
impl D as Core.BitXorAssign {
  fn Op(unused ref self: D, unused other: D) {}
}

fn TestNamedConstraintOp(a: D, b: D) -> D {
  //@dump-sem-ir-begin
  return a ^ b;
  //@dump-sem-ir-end
}

fn TestNamedConstraintAssign(a: D*, b: D) {
  //@dump-sem-ir-begin
  *a ^= b;
  //@dump-sem-ir-end
}

fn TestGenericOp[T: Core.BitXor](a: T, b: T) -> T {
  //@dump-sem-ir-begin
  return a ^ b;
  //@dump-sem-ir-end
}

fn TestGenericAssign[T: Core.BitXorAssign](a: T*, b: T) {
  //@dump-sem-ir-begin
  *a ^= b;
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

// CHECK:STDOUT: --- bit_xor.carbon = $bit_xor
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Other: type = symbolic_binding Other, 0 [symbolic]
// CHECK:STDOUT:   %BitXorWith.type.3e2: type = facet_type <@BitXorWith, @BitXorWith(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.40b: %BitXorWith.type.3e2 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %BitXorWith.assoc_type.b45: type = assoc_entity_type @BitXorWith, @BitXorWith(%Other) [symbolic]
// CHECK:STDOUT:   %BitXorWith.WithSelf.Op.type.b7c: type = fn_type @BitXorWith.WithSelf.Op, @BitXorWith.WithSelf(%Other, %Self.40b) [symbolic]
// CHECK:STDOUT:   %BitXorWith.WithSelf.Op.f80: %BitXorWith.WithSelf.Op.type.b7c = struct_value () [symbolic]
// CHECK:STDOUT:   %BitXorWith.type.12f: type = facet_type <@BitXorWith, @BitXorWith(%C)> [concrete]
// CHECK:STDOUT:   %BitXorWith.assoc_type.2ce: type = assoc_entity_type @BitXorWith, @BitXorWith(%C) [concrete]
// CHECK:STDOUT:   %assoc1.f29: %BitXorWith.assoc_type.2ce = assoc_entity element1, imports.%Core.import_ref.729 [concrete]
// CHECK:STDOUT:   %BitXorWith.impl_witness.45f: <witness> = impl_witness @C.as.BitXorWith.impl.%BitXorWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.4a7: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %.79c: Core.Form = init_form %C [concrete]
// CHECK:STDOUT:   %return.param_patt.415: %pattern_type.4a7 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.340: %pattern_type.4a7 = return_slot_pattern %return.param_patt.415, %C [concrete]
// CHECK:STDOUT:   %C.as.BitXorWith.impl.Op.type: type = fn_type @C.as.BitXorWith.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.BitXorWith.impl.Op: %C.as.BitXorWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %BitXorWith.facet.f38: %BitXorWith.type.12f = facet_value %C, (%BitXorWith.impl_witness.45f) [concrete]
// CHECK:STDOUT:   %BitXorWith.WithSelf.Op.type.f48: type = fn_type @BitXorWith.WithSelf.Op, @BitXorWith.WithSelf(%C, %BitXorWith.facet.f38) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.type.2e4: type = facet_type <@BitXorAssignWith, @BitXorAssignWith(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.c68: %BitXorAssignWith.type.2e4 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %BitXorAssignWith.assoc_type.832: type = assoc_entity_type @BitXorAssignWith, @BitXorAssignWith(%Other) [symbolic]
// CHECK:STDOUT:   %BitXorAssignWith.WithSelf.Op.type.f53: type = fn_type @BitXorAssignWith.WithSelf.Op, @BitXorAssignWith.WithSelf(%Other, %Self.c68) [symbolic]
// CHECK:STDOUT:   %BitXorAssignWith.WithSelf.Op.d75: %BitXorAssignWith.WithSelf.Op.type.f53 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitXorAssignWith.type.b89: type = facet_type <@BitXorAssignWith, @BitXorAssignWith(%C)> [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.impl_witness.05c: <witness> = impl_witness @C.as.BitXorAssignWith.impl.%BitXorAssignWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.assoc_type.ec9: type = assoc_entity_type @BitXorAssignWith, @BitXorAssignWith(%C) [concrete]
// CHECK:STDOUT:   %assoc0.5eb: %BitXorAssignWith.assoc_type.ec9 = assoc_entity element0, imports.%Core.import_ref.f06 [concrete]
// CHECK:STDOUT:   %C.as.BitXorAssignWith.impl.Op.type: type = fn_type @C.as.BitXorAssignWith.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.BitXorAssignWith.impl.Op: %C.as.BitXorAssignWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.facet.fc8: %BitXorAssignWith.type.b89 = facet_value %C, (%BitXorAssignWith.impl_witness.05c) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.WithSelf.Op.type.ffb: type = fn_type @BitXorAssignWith.WithSelf.Op, @BitXorAssignWith.WithSelf(%C, %BitXorAssignWith.facet.fc8) [concrete]
// CHECK:STDOUT:   %a.param_patt.2a5: %pattern_type.4a7 = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt.a30: %pattern_type.4a7 = wrapper_binding_pattern a, %a.param_patt.2a5 [concrete]
// CHECK:STDOUT:   %b.param_patt.45f: %pattern_type.4a7 = value_param_pattern [concrete]
// CHECK:STDOUT:   %b.patt.562: %pattern_type.4a7 = wrapper_binding_pattern b, %b.param_patt.45f [concrete]
// CHECK:STDOUT:   %assoc1.f10: %BitXorWith.assoc_type.b45 = assoc_entity element1, imports.%Core.import_ref.699 [symbolic]
// CHECK:STDOUT:   %.434: type = fn_type_with_self_type %BitXorWith.WithSelf.Op.type.f48, %BitXorWith.facet.f38 [concrete]
// CHECK:STDOUT:   %ptr.00e: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.842: type = pattern_type %ptr.00e [concrete]
// CHECK:STDOUT:   %a.param_patt.4fa: %pattern_type.842 = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt.eba: %pattern_type.842 = wrapper_binding_pattern a, %a.param_patt.4fa [concrete]
// CHECK:STDOUT:   %assoc0.dbd: %BitXorAssignWith.assoc_type.832 = assoc_entity element0, imports.%Core.import_ref.fa3 [symbolic]
// CHECK:STDOUT:   %.e48: type = fn_type_with_self_type %BitXorAssignWith.WithSelf.Op.type.ffb, %BitXorAssignWith.facet.fc8 [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %BitXor.type: type = facet_type <@BitXor> [concrete]
// CHECK:STDOUT:   %BitXorWith.impl_witness.736: <witness> = impl_witness @D.as.BitXorWith.impl.%BitXorWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.331: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %.84d: Core.Form = init_form %D [concrete]
// CHECK:STDOUT:   %return.param_patt.835: %pattern_type.331 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.5ef: %pattern_type.331 = return_slot_pattern %return.param_patt.835, %D [concrete]
// CHECK:STDOUT:   %D.as.BitXorWith.impl.Op.type: type = fn_type @D.as.BitXorWith.impl.Op [concrete]
// CHECK:STDOUT:   %D.as.BitXorWith.impl.Op: %D.as.BitXorWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %BitXorWith.type.95e: type = facet_type <@BitXorWith, @BitXorWith(%D)> [concrete]
// CHECK:STDOUT:   %BitXorWith.facet.4af: %BitXorWith.type.95e = facet_value %D, (%BitXorWith.impl_witness.736) [concrete]
// CHECK:STDOUT:   %BitXorWith.assoc_type.a37: type = assoc_entity_type @BitXorWith, @BitXorWith(%D) [concrete]
// CHECK:STDOUT:   %BitXorWith.WithSelf.Op.type.289: type = fn_type @BitXorWith.WithSelf.Op, @BitXorWith.WithSelf(%D, %BitXorWith.facet.4af) [concrete]
// CHECK:STDOUT:   %assoc1.dc9: %BitXorWith.assoc_type.a37 = assoc_entity element1, imports.%Core.import_ref.729 [concrete]
// CHECK:STDOUT:   %BitXorAssign.type: type = facet_type <@BitXorAssign> [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.impl_witness.736: <witness> = impl_witness @D.as.BitXorAssignWith.impl.%BitXorAssignWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %D.as.BitXorAssignWith.impl.Op.type: type = fn_type @D.as.BitXorAssignWith.impl.Op [concrete]
// CHECK:STDOUT:   %D.as.BitXorAssignWith.impl.Op: %D.as.BitXorAssignWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.type.4c4: type = facet_type <@BitXorAssignWith, @BitXorAssignWith(%D)> [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.facet.58a: %BitXorAssignWith.type.4c4 = facet_value %D, (%BitXorAssignWith.impl_witness.736) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.WithSelf.Op.type.1eb: type = fn_type @BitXorAssignWith.WithSelf.Op, @BitXorAssignWith.WithSelf(%D, %BitXorAssignWith.facet.58a) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.assoc_type.545: type = assoc_entity_type @BitXorAssignWith, @BitXorAssignWith(%D) [concrete]
// CHECK:STDOUT:   %assoc0.9ee: %BitXorAssignWith.assoc_type.545 = assoc_entity element0, imports.%Core.import_ref.f06 [concrete]
// CHECK:STDOUT:   %a.param_patt.d0e: %pattern_type.331 = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt.ec5: %pattern_type.331 = wrapper_binding_pattern a, %a.param_patt.d0e [concrete]
// CHECK:STDOUT:   %b.param_patt.110: %pattern_type.331 = value_param_pattern [concrete]
// CHECK:STDOUT:   %b.patt.14c: %pattern_type.331 = wrapper_binding_pattern b, %b.param_patt.110 [concrete]
// CHECK:STDOUT:   %.b0b: type = fn_type_with_self_type %BitXorWith.WithSelf.Op.type.289, %BitXorWith.facet.4af [concrete]
// CHECK:STDOUT:   %ptr.4a5: type = ptr_type %D [concrete]
// CHECK:STDOUT:   %pattern_type.ceb: type = pattern_type %ptr.4a5 [concrete]
// CHECK:STDOUT:   %a.param_patt.609: %pattern_type.ceb = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt.fac: %pattern_type.ceb = wrapper_binding_pattern a, %a.param_patt.609 [concrete]
// CHECK:STDOUT:   %.61a: type = fn_type_with_self_type %BitXorAssignWith.WithSelf.Op.type.1eb, %BitXorAssignWith.facet.58a [concrete]
// CHECK:STDOUT:   %pattern_type.9a587c.2: type = pattern_type %BitXor.type [concrete]
// CHECK:STDOUT:   %T.patt.3a0592.1: %pattern_type.9a587c.2 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.f45530.1: %BitXor.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.190bb7.1: type = facet_access_type %T.f45530.1 [symbolic]
// CHECK:STDOUT:   %pattern_type.86a9db.1: type = pattern_type %T.as_type.190bb7.1 [symbolic]
// CHECK:STDOUT:   %a.param_patt.e33: %pattern_type.86a9db.1 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %a.patt.e02: %pattern_type.86a9db.1 = wrapper_binding_pattern a, %a.param_patt.e33 [symbolic]
// CHECK:STDOUT:   %b.param_patt.4cd741.1: %pattern_type.86a9db.1 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %b.patt.60acf6.1: %pattern_type.86a9db.1 = wrapper_binding_pattern b, %b.param_patt.4cd741.1 [symbolic]
// CHECK:STDOUT:   %.1f0: Core.Form = init_form %T.as_type.190bb7.1 [symbolic]
// CHECK:STDOUT:   %return.param_patt.5f6: %pattern_type.86a9db.1 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.96a: %pattern_type.86a9db.1 = return_slot_pattern %return.param_patt.5f6, %T.as_type.190bb7.1 [symbolic]
// CHECK:STDOUT:   %BitXorWith.type.0183c7.2: type = facet_type <@BitXorWith, @BitXorWith(%T.as_type.190bb7.1)> [symbolic]
// CHECK:STDOUT:   %BitXorWith.assoc_type.477171.2: type = assoc_entity_type @BitXorWith, @BitXorWith(%T.as_type.190bb7.1) [symbolic]
// CHECK:STDOUT:   %assoc1.d23edb.2: %BitXorWith.assoc_type.477171.2 = assoc_entity element1, imports.%Core.import_ref.729 [symbolic]
// CHECK:STDOUT:   %require_complete.c5c: <witness> = require_complete_type %BitXorWith.type.0183c7.2 [symbolic]
// CHECK:STDOUT:   %BitXorWith.lookup_impl_witness.ed23b2.2: <witness> = lookup_impl_witness %T.f45530.1, @BitXorWith, @BitXorWith(%T.as_type.190bb7.1) [symbolic]
// CHECK:STDOUT:   %BitXorWith.facet.80e: %BitXorWith.type.0183c7.2 = facet_value %T.as_type.190bb7.1, (%BitXorWith.lookup_impl_witness.ed23b2.2) [symbolic]
// CHECK:STDOUT:   %BitXorWith.WithSelf.Op.type.8e7: type = fn_type @BitXorWith.WithSelf.Op, @BitXorWith.WithSelf(%T.as_type.190bb7.1, %BitXorWith.facet.80e) [symbolic]
// CHECK:STDOUT:   %.087: type = fn_type_with_self_type %BitXorWith.WithSelf.Op.type.8e7, %BitXorWith.facet.80e [symbolic]
// CHECK:STDOUT:   %impl.elem1.44a: %.087 = impl_witness_access %BitXorWith.lookup_impl_witness.ed23b2.2, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.ef2: <specific function> = specific_impl_function %impl.elem1.44a, @BitXorWith.WithSelf.Op(%T.as_type.190bb7.1, %BitXorWith.facet.80e) [symbolic]
// CHECK:STDOUT:   %pattern_type.9a587c.3: type = pattern_type %BitXorAssign.type [concrete]
// CHECK:STDOUT:   %T.patt.3a0592.2: %pattern_type.9a587c.3 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.f45530.2: %BitXorAssign.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.190bb7.2: type = facet_access_type %T.f45530.2 [symbolic]
// CHECK:STDOUT:   %ptr.360: type = ptr_type %T.as_type.190bb7.2 [symbolic]
// CHECK:STDOUT:   %pattern_type.ff5: type = pattern_type %ptr.360 [symbolic]
// CHECK:STDOUT:   %a.param_patt.f50: %pattern_type.ff5 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %a.patt.193: %pattern_type.ff5 = wrapper_binding_pattern a, %a.param_patt.f50 [symbolic]
// CHECK:STDOUT:   %pattern_type.86a9db.2: type = pattern_type %T.as_type.190bb7.2 [symbolic]
// CHECK:STDOUT:   %b.param_patt.4cd741.2: %pattern_type.86a9db.2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %b.patt.60acf6.2: %pattern_type.86a9db.2 = wrapper_binding_pattern b, %b.param_patt.4cd741.2 [symbolic]
// CHECK:STDOUT:   %BitXorAssignWith.type.ad9b0c.2: type = facet_type <@BitXorAssignWith, @BitXorAssignWith(%T.as_type.190bb7.2)> [symbolic]
// CHECK:STDOUT:   %BitXorAssignWith.assoc_type.fd0b33.2: type = assoc_entity_type @BitXorAssignWith, @BitXorAssignWith(%T.as_type.190bb7.2) [symbolic]
// CHECK:STDOUT:   %assoc0.5e4786.2: %BitXorAssignWith.assoc_type.fd0b33.2 = assoc_entity element0, imports.%Core.import_ref.f06 [symbolic]
// CHECK:STDOUT:   %require_complete.03f253.2: <witness> = require_complete_type %BitXorAssignWith.type.ad9b0c.2 [symbolic]
// CHECK:STDOUT:   %BitXorAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.f45530.2, @BitXorAssignWith, @BitXorAssignWith(%T.as_type.190bb7.2) [symbolic]
// CHECK:STDOUT:   %BitXorAssignWith.facet.38b: %BitXorAssignWith.type.ad9b0c.2 = facet_value %T.as_type.190bb7.2, (%BitXorAssignWith.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %BitXorAssignWith.WithSelf.Op.type.46c: type = fn_type @BitXorAssignWith.WithSelf.Op, @BitXorAssignWith.WithSelf(%T.as_type.190bb7.2, %BitXorAssignWith.facet.38b) [symbolic]
// CHECK:STDOUT:   %.7b7: type = fn_type_with_self_type %BitXorAssignWith.WithSelf.Op.type.46c, %BitXorAssignWith.facet.38b [symbolic]
// CHECK:STDOUT:   %impl.elem0.cd6: %.7b7 = impl_witness_access %BitXorAssignWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.6f5: <specific function> = specific_impl_function %impl.elem0.cd6, @BitXorAssignWith.WithSelf.Op(%T.as_type.190bb7.2, %BitXorAssignWith.facet.38b) [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %Int.d4bbe0.1: type = class_type @Int, @Int(%N) [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %ptr.a17: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %ptr.a17 [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %BitXor.facet.8c0: %BitXor.type = facet_value %C, (%BitXorWith.impl_witness.45f) [concrete]
// CHECK:STDOUT:   %BitXorAssign.facet.3ca: %BitXorAssign.type = facet_value %C, (%BitXorAssignWith.impl_witness.05c) [concrete]
// CHECK:STDOUT:   %BitXor.facet.6f9: %BitXor.type = facet_value %D, (%BitXorWith.impl_witness.736) [concrete]
// CHECK:STDOUT:   %BitXorAssign.facet.6f9: %BitXorAssign.type = facet_value %D, (%BitXorAssignWith.impl_witness.736) [concrete]
// CHECK:STDOUT:   %Int.as.BitXorWith.impl.Op.type.96f: type = fn_type @Int.as.BitXorWith.impl.Op.1, @Int.as.BitXorWith.impl.3c9(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.BitXorWith.impl.Op.016: %Int.as.BitXorWith.impl.Op.type.96f = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.81c7d0.1: type = facet_type <@ImplicitAs, @ImplicitAs(%Int.d4bbe0.1)> [symbolic]
// CHECK:STDOUT:   %U.f16: %ImplicitAs.type.81c7d0.1 = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %BitXorWith.type.bfb: type = facet_type <@BitXorWith, @BitXorWith(%i32)> [concrete]
// CHECK:STDOUT:   %BitXorWith.impl_witness.5c7: <witness> = impl_witness imports.%BitXorWith.impl_witness_table.7f9, @Int.as.BitXorWith.impl.3c9(%int_32) [concrete]
// CHECK:STDOUT:   %BitXorWith.assoc_type.38b: type = assoc_entity_type @BitXorWith, @BitXorWith(%i32) [concrete]
// CHECK:STDOUT:   %assoc1.e08: %BitXorWith.assoc_type.38b = assoc_entity element1, imports.%Core.import_ref.729 [concrete]
// CHECK:STDOUT:   %complete_type.f01: <witness> = complete_type_witness %BitXorWith.type.bfb [concrete]
// CHECK:STDOUT:   %Int.as.BitXorWith.impl.Op.type.c44: type = fn_type @Int.as.BitXorWith.impl.Op.1, @Int.as.BitXorWith.impl.3c9(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.BitXorWith.impl.Op.153: %Int.as.BitXorWith.impl.Op.type.c44 = struct_value () [concrete]
// CHECK:STDOUT:   %BitXor.facet.82f: %BitXor.type = facet_value %i32, (%BitXorWith.impl_witness.5c7) [concrete]
// CHECK:STDOUT:   %a.param_patt.076: %pattern_type.d94 = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt.96f: %pattern_type.d94 = wrapper_binding_pattern a, %a.param_patt.076 [concrete]
// CHECK:STDOUT:   %b.param_patt.1a2: %pattern_type.d94 = value_param_pattern [concrete]
// CHECK:STDOUT:   %b.patt.3ca: %pattern_type.d94 = wrapper_binding_pattern b, %b.param_patt.1a2 [concrete]
// CHECK:STDOUT:   %.0b3: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.299: %pattern_type.d94 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.6fa: %pattern_type.d94 = return_slot_pattern %return.param_patt.299, %i32 [concrete]
// CHECK:STDOUT:   %Int.as.BitXorAssignWith.impl.Op.type.bf8238.1: type = fn_type @Int.as.BitXorAssignWith.impl.Op.1, @Int.as.BitXorAssignWith.impl(%N, %U.f16) [symbolic]
// CHECK:STDOUT:   %Int.as.BitXorAssignWith.impl.Op.fed844.1: %Int.as.BitXorAssignWith.impl.Op.type.bf8238.1 = struct_value () [symbolic]
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
// CHECK:STDOUT:   %BitXorAssignWith.type.e76: type = facet_type <@BitXorAssignWith, @BitXorAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.impl_witness.b11: <witness> = impl_witness imports.%BitXorAssignWith.impl_witness_table, @Int.as.BitXorAssignWith.impl(%int_32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %BitXorAssign.facet.fc2: %BitXorAssign.type = facet_value %i32, (%BitXorAssignWith.impl_witness.b11) [concrete]
// CHECK:STDOUT:   %a.param_patt.977: %pattern_type.7dc = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt.b70: %pattern_type.7dc = wrapper_binding_pattern a, %a.param_patt.977 [concrete]
// CHECK:STDOUT:   %complete_type.259: <witness> = complete_type_witness %BitXorWith.type.12f [concrete]
// CHECK:STDOUT:   %complete_type.a56: <witness> = complete_type_witness %ptr.00e [concrete]
// CHECK:STDOUT:   %complete_type.f05: <witness> = complete_type_witness %BitXorAssignWith.type.b89 [concrete]
// CHECK:STDOUT:   %complete_type.b51: <witness> = complete_type_witness %BitXorWith.type.95e [concrete]
// CHECK:STDOUT:   %complete_type.320: <witness> = complete_type_witness %ptr.4a5 [concrete]
// CHECK:STDOUT:   %complete_type.ce2: <witness> = complete_type_witness %BitXorAssignWith.type.4c4 [concrete]
// CHECK:STDOUT:   %BitXorWith.facet.365: %BitXorWith.type.bfb = facet_value %i32, (%BitXorWith.impl_witness.5c7) [concrete]
// CHECK:STDOUT:   %BitXorWith.WithSelf.Op.type.53c: type = fn_type @BitXorWith.WithSelf.Op, @BitXorWith.WithSelf(%i32, %BitXorWith.facet.365) [concrete]
// CHECK:STDOUT:   %.3d5: type = fn_type_with_self_type %BitXorWith.WithSelf.Op.type.53c, %BitXorWith.facet.365 [concrete]
// CHECK:STDOUT:   %Int.as.BitXorWith.impl.Op.specific_fn.3a2: <specific function> = specific_function %Int.as.BitXorWith.impl.Op.153, @Int.as.BitXorWith.impl.Op.1(%int_32) [concrete]
// CHECK:STDOUT:   %complete_type.1b1: <witness> = complete_type_witness %ptr.a17 [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.assoc_type.874: type = assoc_entity_type @BitXorAssignWith, @BitXorAssignWith(%i32) [concrete]
// CHECK:STDOUT:   %assoc0.5f9: %BitXorAssignWith.assoc_type.874 = assoc_entity element0, imports.%Core.import_ref.f06 [concrete]
// CHECK:STDOUT:   %complete_type.baf: <witness> = complete_type_witness %BitXorAssignWith.type.e76 [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.facet.0e5: %BitXorAssignWith.type.e76 = facet_value %i32, (%BitXorAssignWith.impl_witness.b11) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.WithSelf.Op.type.944: type = fn_type @BitXorAssignWith.WithSelf.Op, @BitXorAssignWith.WithSelf(%i32, %BitXorAssignWith.facet.0e5) [concrete]
// CHECK:STDOUT:   %.a85: type = fn_type_with_self_type %BitXorAssignWith.WithSelf.Op.type.944, %BitXorAssignWith.facet.0e5 [concrete]
// CHECK:STDOUT:   %Int.as.BitXorAssignWith.impl.Op.type.43b4a6.2: type = fn_type @Int.as.BitXorAssignWith.impl.Op.1, @Int.as.BitXorAssignWith.impl(%int_32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Int.as.BitXorAssignWith.impl.Op.62f34b.2: %Int.as.BitXorAssignWith.impl.Op.type.43b4a6.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.BitXorAssignWith.impl.Op.specific_fn.5f6c9b.1: <specific function> = specific_function %Int.as.BitXorAssignWith.impl.Op.62f34b.2, @Int.as.BitXorAssignWith.impl.Op.1(%int_32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.27e: @BitXorWith.WithSelf.%BitXorWith.assoc_type (%BitXorWith.assoc_type.b45) = import_ref Core//prelude/operators/bitwise, loc{{\d+_\d+}}, loaded [symbolic = @BitXorWith.WithSelf.%assoc1 (constants.%assoc1.f10)]
// CHECK:STDOUT:   %Core.import_ref.729: @BitXorWith.WithSelf.%BitXorWith.WithSelf.Op.type (%BitXorWith.WithSelf.Op.type.b7c) = import_ref Core//prelude/operators/bitwise, loc{{\d+_\d+}}, loaded [symbolic = @BitXorWith.WithSelf.%BitXorWith.WithSelf.Op (constants.%BitXorWith.WithSelf.Op.f80)]
// CHECK:STDOUT:   %Core.import_ref.b36: @BitXorAssignWith.WithSelf.%BitXorAssignWith.assoc_type (%BitXorAssignWith.assoc_type.832) = import_ref Core//prelude/operators/bitwise, loc{{\d+_\d+}}, loaded [symbolic = @BitXorAssignWith.WithSelf.%assoc0 (constants.%assoc0.dbd)]
// CHECK:STDOUT:   %Core.import_ref.f06: @BitXorAssignWith.WithSelf.%BitXorAssignWith.WithSelf.Op.type (%BitXorAssignWith.WithSelf.Op.type.f53) = import_ref Core//prelude/operators/bitwise, loc{{\d+_\d+}}, loaded [symbolic = @BitXorAssignWith.WithSelf.%BitXorAssignWith.WithSelf.Op (constants.%BitXorAssignWith.WithSelf.Op.d75)]
// CHECK:STDOUT:   %Core.import_ref.699 = import_ref Core//prelude/operators/bitwise, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.fa3 = import_ref Core//prelude/operators/bitwise, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.ebd: type = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.BitXorWith.impl.3c9.%Int (constants.%Int.d4bbe0.1)]
// CHECK:STDOUT:   %Core.import_ref.299: @Int.as.BitXorWith.impl.3c9.%Int.as.BitXorWith.impl.Op.type (%Int.as.BitXorWith.impl.Op.type.96f) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.BitXorWith.impl.3c9.%Int.as.BitXorWith.impl.Op (constants.%Int.as.BitXorWith.impl.Op.016)]
// CHECK:STDOUT:   %BitXorWith.impl_witness_table.7f9 = impl_witness_table (%Core.import_ref.ebd, %Core.import_ref.299), @Int.as.BitXorWith.impl.3c9 [concrete]
// CHECK:STDOUT:   %Core.import_ref.bf4: @Int.as.BitXorAssignWith.impl.%Int.as.BitXorAssignWith.impl.Op.type.2 (%Int.as.BitXorAssignWith.impl.Op.type.bf8238.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.BitXorAssignWith.impl.%Int.as.BitXorAssignWith.impl.Op.2 (constants.%Int.as.BitXorAssignWith.impl.Op.fed844.1)]
// CHECK:STDOUT:   %BitXorAssignWith.impl_witness_table = impl_witness_table (%Core.import_ref.bf4), @Int.as.BitXorAssignWith.impl [concrete]
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
// CHECK:STDOUT:   %impl.elem1: %.434 = impl_witness_access constants.%BitXorWith.impl_witness.45f, element1 [concrete = constants.%C.as.BitXorWith.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.ref, %impl.elem1
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.as.BitXorWith.impl.Op.call: init %C to %.loc30_26.1 = call %bound_method(%a.ref, %b.ref)
// CHECK:STDOUT:   return %C.as.BitXorWith.impl.Op.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestAssign(%a.param: %ptr.00e, %b.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %ptr.00e = name_ref a, %a
// CHECK:STDOUT:   %.loc38: ref %C = deref %a.ref
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0: %.e48 = impl_witness_access constants.%BitXorAssignWith.impl_witness.05c, element0 [concrete = constants.%C.as.BitXorAssignWith.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc38, %impl.elem0
// CHECK:STDOUT:   %C.as.BitXorAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method(%.loc38, %b.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestNamedConstraintOp(%a.param: %D, %b.param: %D) -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %D = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %D = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1: %.b0b = impl_witness_access constants.%BitXorWith.impl_witness.736, element1 [concrete = constants.%D.as.BitXorWith.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.ref, %impl.elem1
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %D.as.BitXorWith.impl.Op.call: init %D to %.loc53_41.1 = call %bound_method(%a.ref, %b.ref)
// CHECK:STDOUT:   return %D.as.BitXorWith.impl.Op.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestNamedConstraintAssign(%a.param: %ptr.4a5, %b.param: %D) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %ptr.4a5 = name_ref a, %a
// CHECK:STDOUT:   %.loc61: ref %D = deref %a.ref
// CHECK:STDOUT:   %b.ref: %D = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0: %.61a = impl_witness_access constants.%BitXorAssignWith.impl_witness.736, element0 [concrete = constants.%D.as.BitXorAssignWith.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc61, %impl.elem0
// CHECK:STDOUT:   %D.as.BitXorAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method(%.loc61, %b.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestGenericOp(%T.loc65_19.2: %BitXor.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %BitXorWith.type.loc67_12.2: type = facet_type <@BitXorWith, @BitXorWith(%T.as_type.loc65_37.1)> [symbolic = %BitXorWith.type.loc67_12.2 (constants.%BitXorWith.type.0183c7.2)]
// CHECK:STDOUT:   %require_complete.loc67: <witness> = require_complete_type %BitXorWith.type.loc67_12.2 [symbolic = %require_complete.loc67 (constants.%require_complete.c5c)]
// CHECK:STDOUT:   %BitXorWith.assoc_type: type = assoc_entity_type @BitXorWith, @BitXorWith(%T.as_type.loc65_37.1) [symbolic = %BitXorWith.assoc_type (constants.%BitXorWith.assoc_type.477171.2)]
// CHECK:STDOUT:   %assoc1: @TestGenericOp.%BitXorWith.assoc_type (%BitXorWith.assoc_type.477171.2) = assoc_entity element1, imports.%Core.import_ref.729 [symbolic = %assoc1 (constants.%assoc1.d23edb.2)]
// CHECK:STDOUT:   %BitXorWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc65_19.1, @BitXorWith, @BitXorWith(%T.as_type.loc65_37.1) [symbolic = %BitXorWith.lookup_impl_witness (constants.%BitXorWith.lookup_impl_witness.ed23b2.2)]
// CHECK:STDOUT:   %BitXorWith.facet: @TestGenericOp.%BitXorWith.type.loc67_12.2 (%BitXorWith.type.0183c7.2) = facet_value %T.as_type.loc65_37.1, (%BitXorWith.lookup_impl_witness) [symbolic = %BitXorWith.facet (constants.%BitXorWith.facet.80e)]
// CHECK:STDOUT:   %BitXorWith.WithSelf.Op.type: type = fn_type @BitXorWith.WithSelf.Op, @BitXorWith.WithSelf(%T.as_type.loc65_37.1, %BitXorWith.facet) [symbolic = %BitXorWith.WithSelf.Op.type (constants.%BitXorWith.WithSelf.Op.type.8e7)]
// CHECK:STDOUT:   %.loc67_12.2: type = fn_type_with_self_type %BitXorWith.WithSelf.Op.type, %BitXorWith.facet [symbolic = %.loc67_12.2 (constants.%.087)]
// CHECK:STDOUT:   %impl.elem1.loc67_12.2: @TestGenericOp.%.loc67_12.2 (%.087) = impl_witness_access %BitXorWith.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc67_12.2 (constants.%impl.elem1.44a)]
// CHECK:STDOUT:   %specific_impl_fn.loc67_12.2: <specific function> = specific_impl_function %impl.elem1.loc67_12.2, @BitXorWith.WithSelf.Op(%T.as_type.loc65_37.1, %BitXorWith.facet) [symbolic = %specific_impl_fn.loc67_12.2 (constants.%specific_impl_fn.ef2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @TestGenericOp.%T.as_type.loc65_37.1 (%T.as_type.190bb7.1), %b.param: @TestGenericOp.%T.as_type.loc65_37.1 (%T.as_type.190bb7.1)) -> out %return.param: @TestGenericOp.%T.as_type.loc65_37.1 (%T.as_type.190bb7.1) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %a.ref: @TestGenericOp.%T.as_type.loc65_37.1 (%T.as_type.190bb7.1) = name_ref a, %a
// CHECK:STDOUT:     %b.ref: @TestGenericOp.%T.as_type.loc65_37.1 (%T.as_type.190bb7.1) = name_ref b, %b
// CHECK:STDOUT:     %BitXorWith.type.loc67_12.1: type = facet_type <@BitXorWith, @BitXorWith(constants.%T.as_type.190bb7.1)> [symbolic = %BitXorWith.type.loc67_12.2 (constants.%BitXorWith.type.0183c7.2)]
// CHECK:STDOUT:     %.loc67_12.1: @TestGenericOp.%BitXorWith.assoc_type (%BitXorWith.assoc_type.477171.2) = specific_constant imports.%Core.import_ref.27e, @BitXorWith.WithSelf(constants.%T.as_type.190bb7.1, constants.%Self.40b) [symbolic = %assoc1 (constants.%assoc1.d23edb.2)]
// CHECK:STDOUT:     %Op.ref: @TestGenericOp.%BitXorWith.assoc_type (%BitXorWith.assoc_type.477171.2) = name_ref Op, %.loc67_12.1 [symbolic = %assoc1 (constants.%assoc1.d23edb.2)]
// CHECK:STDOUT:     %impl.elem1.loc67_12.1: @TestGenericOp.%.loc67_12.2 (%.087) = impl_witness_access constants.%BitXorWith.lookup_impl_witness.ed23b2.2, element1 [symbolic = %impl.elem1.loc67_12.2 (constants.%impl.elem1.44a)]
// CHECK:STDOUT:     %bound_method.loc67_12.1: <bound method> = bound_method %a.ref, %impl.elem1.loc67_12.1
// CHECK:STDOUT:     %specific_impl_fn.loc67_12.1: <specific function> = specific_impl_function %impl.elem1.loc67_12.1, @BitXorWith.WithSelf.Op(constants.%T.as_type.190bb7.1, constants.%BitXorWith.facet.80e) [symbolic = %specific_impl_fn.loc67_12.2 (constants.%specific_impl_fn.ef2)]
// CHECK:STDOUT:     %bound_method.loc67_12.2: <bound method> = bound_method %a.ref, %specific_impl_fn.loc67_12.1
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %BitXorWith.WithSelf.Op.call: init @TestGenericOp.%T.as_type.loc65_37.1 (%T.as_type.190bb7.1) to %.loc65_49.1 = call %bound_method.loc67_12.2(%a.ref, %b.ref)
// CHECK:STDOUT:     return %BitXorWith.WithSelf.Op.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestGenericAssign(%T.loc71_23.2: %BitXorAssign.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %BitXorAssignWith.type.loc73_6.2: type = facet_type <@BitXorAssignWith, @BitXorAssignWith(%T.as_type.loc71_48.1)> [symbolic = %BitXorAssignWith.type.loc73_6.2 (constants.%BitXorAssignWith.type.ad9b0c.2)]
// CHECK:STDOUT:   %require_complete.loc73: <witness> = require_complete_type %BitXorAssignWith.type.loc73_6.2 [symbolic = %require_complete.loc73 (constants.%require_complete.03f253.2)]
// CHECK:STDOUT:   %BitXorAssignWith.assoc_type: type = assoc_entity_type @BitXorAssignWith, @BitXorAssignWith(%T.as_type.loc71_48.1) [symbolic = %BitXorAssignWith.assoc_type (constants.%BitXorAssignWith.assoc_type.fd0b33.2)]
// CHECK:STDOUT:   %assoc0: @TestGenericAssign.%BitXorAssignWith.assoc_type (%BitXorAssignWith.assoc_type.fd0b33.2) = assoc_entity element0, imports.%Core.import_ref.f06 [symbolic = %assoc0 (constants.%assoc0.5e4786.2)]
// CHECK:STDOUT:   %BitXorAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc71_23.1, @BitXorAssignWith, @BitXorAssignWith(%T.as_type.loc71_48.1) [symbolic = %BitXorAssignWith.lookup_impl_witness (constants.%BitXorAssignWith.lookup_impl_witness)]
// CHECK:STDOUT:   %BitXorAssignWith.facet: @TestGenericAssign.%BitXorAssignWith.type.loc73_6.2 (%BitXorAssignWith.type.ad9b0c.2) = facet_value %T.as_type.loc71_48.1, (%BitXorAssignWith.lookup_impl_witness) [symbolic = %BitXorAssignWith.facet (constants.%BitXorAssignWith.facet.38b)]
// CHECK:STDOUT:   %BitXorAssignWith.WithSelf.Op.type: type = fn_type @BitXorAssignWith.WithSelf.Op, @BitXorAssignWith.WithSelf(%T.as_type.loc71_48.1, %BitXorAssignWith.facet) [symbolic = %BitXorAssignWith.WithSelf.Op.type (constants.%BitXorAssignWith.WithSelf.Op.type.46c)]
// CHECK:STDOUT:   %.loc73_6.2: type = fn_type_with_self_type %BitXorAssignWith.WithSelf.Op.type, %BitXorAssignWith.facet [symbolic = %.loc73_6.2 (constants.%.7b7)]
// CHECK:STDOUT:   %impl.elem0.loc73_6.2: @TestGenericAssign.%.loc73_6.2 (%.7b7) = impl_witness_access %BitXorAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc73_6.2 (constants.%impl.elem0.cd6)]
// CHECK:STDOUT:   %specific_impl_fn.loc73_6.2: <specific function> = specific_impl_function %impl.elem0.loc73_6.2, @BitXorAssignWith.WithSelf.Op(%T.as_type.loc71_48.1, %BitXorAssignWith.facet) [symbolic = %specific_impl_fn.loc73_6.2 (constants.%specific_impl_fn.6f5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @TestGenericAssign.%ptr.loc71_48.1 (%ptr.360), %b.param: @TestGenericAssign.%T.as_type.loc71_48.1 (%T.as_type.190bb7.2)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %a.ref: @TestGenericAssign.%ptr.loc71_48.1 (%ptr.360) = name_ref a, %a
// CHECK:STDOUT:     %.loc73_3: ref @TestGenericAssign.%T.as_type.loc71_48.1 (%T.as_type.190bb7.2) = deref %a.ref
// CHECK:STDOUT:     %b.ref: @TestGenericAssign.%T.as_type.loc71_48.1 (%T.as_type.190bb7.2) = name_ref b, %b
// CHECK:STDOUT:     %BitXorAssignWith.type.loc73_6.1: type = facet_type <@BitXorAssignWith, @BitXorAssignWith(constants.%T.as_type.190bb7.2)> [symbolic = %BitXorAssignWith.type.loc73_6.2 (constants.%BitXorAssignWith.type.ad9b0c.2)]
// CHECK:STDOUT:     %.loc73_6.1: @TestGenericAssign.%BitXorAssignWith.assoc_type (%BitXorAssignWith.assoc_type.fd0b33.2) = specific_constant imports.%Core.import_ref.b36, @BitXorAssignWith.WithSelf(constants.%T.as_type.190bb7.2, constants.%Self.c68) [symbolic = %assoc0 (constants.%assoc0.5e4786.2)]
// CHECK:STDOUT:     %Op.ref: @TestGenericAssign.%BitXorAssignWith.assoc_type (%BitXorAssignWith.assoc_type.fd0b33.2) = name_ref Op, %.loc73_6.1 [symbolic = %assoc0 (constants.%assoc0.5e4786.2)]
// CHECK:STDOUT:     %impl.elem0.loc73_6.1: @TestGenericAssign.%.loc73_6.2 (%.7b7) = impl_witness_access constants.%BitXorAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc73_6.2 (constants.%impl.elem0.cd6)]
// CHECK:STDOUT:     %bound_method.loc73_6.1: <bound method> = bound_method %.loc73_3, %impl.elem0.loc73_6.1
// CHECK:STDOUT:     %specific_impl_fn.loc73_6.1: <specific function> = specific_impl_function %impl.elem0.loc73_6.1, @BitXorAssignWith.WithSelf.Op(constants.%T.as_type.190bb7.2, constants.%BitXorAssignWith.facet.38b) [symbolic = %specific_impl_fn.loc73_6.2 (constants.%specific_impl_fn.6f5)]
// CHECK:STDOUT:     %bound_method.loc73_6.2: <bound method> = bound_method %.loc73_3, %specific_impl_fn.loc73_6.1
// CHECK:STDOUT:     %BitXorAssignWith.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc73_6.2(%.loc73_3, %b.ref)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGenericOp(constants.%T.f45530.1) {
// CHECK:STDOUT:   %T.patt.loc65_19.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc65_19.1 => constants.%T.f45530.1
// CHECK:STDOUT:   %T.as_type.loc65_37.1 => constants.%T.as_type.190bb7.1
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.86a9db.1
// CHECK:STDOUT:   %a.param_patt.loc65_35.2 => constants.%a.param_patt.e33
// CHECK:STDOUT:   %a.patt.loc65_35.2 => constants.%a.patt.e02
// CHECK:STDOUT:   %b.param_patt.loc65_41.2 => constants.%b.param_patt.4cd741.1
// CHECK:STDOUT:   %b.patt.loc65_41.2 => constants.%b.patt.60acf6.1
// CHECK:STDOUT:   %.loc65_49.2 => constants.%.1f0
// CHECK:STDOUT:   %return.param_patt.loc65_49.2 => constants.%return.param_patt.5f6
// CHECK:STDOUT:   %return.patt.loc65_46.2 => constants.%return.patt.96a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGenericAssign(constants.%T.f45530.2) {
// CHECK:STDOUT:   %T.patt.loc71_23.2 => constants.%T.patt.3a0592.2
// CHECK:STDOUT:   %T.loc71_23.1 => constants.%T.f45530.2
// CHECK:STDOUT:   %T.as_type.loc71_48.1 => constants.%T.as_type.190bb7.2
// CHECK:STDOUT:   %ptr.loc71_48.1 => constants.%ptr.360
// CHECK:STDOUT:   %pattern_type.loc71_45 => constants.%pattern_type.ff5
// CHECK:STDOUT:   %a.param_patt.loc71_45.2 => constants.%a.param_patt.f50
// CHECK:STDOUT:   %a.patt.loc71_45.2 => constants.%a.patt.193
// CHECK:STDOUT:   %pattern_type.loc71_52 => constants.%pattern_type.86a9db.2
// CHECK:STDOUT:   %b.param_patt.loc71_52.2 => constants.%b.param_patt.4cd741.2
// CHECK:STDOUT:   %b.patt.loc71_52.2 => constants.%b.patt.60acf6.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGenericOp(constants.%BitXor.facet.8c0) {
// CHECK:STDOUT:   %T.patt.loc65_19.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc65_19.1 => constants.%BitXor.facet.8c0
// CHECK:STDOUT:   %T.as_type.loc65_37.1 => constants.%C
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.4a7
// CHECK:STDOUT:   %a.param_patt.loc65_35.2 => constants.%a.param_patt.2a5
// CHECK:STDOUT:   %a.patt.loc65_35.2 => constants.%a.patt.a30
// CHECK:STDOUT:   %b.param_patt.loc65_41.2 => constants.%b.param_patt.45f
// CHECK:STDOUT:   %b.patt.loc65_41.2 => constants.%b.patt.562
// CHECK:STDOUT:   %.loc65_49.2 => constants.%.79c
// CHECK:STDOUT:   %return.param_patt.loc65_49.2 => constants.%return.param_patt.415
// CHECK:STDOUT:   %return.patt.loc65_46.2 => constants.%return.patt.340
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc65 => constants.%complete_type.357
// CHECK:STDOUT:   %BitXorWith.type.loc67_12.2 => constants.%BitXorWith.type.12f
// CHECK:STDOUT:   %require_complete.loc67 => constants.%complete_type.259
// CHECK:STDOUT:   %BitXorWith.assoc_type => constants.%BitXorWith.assoc_type.2ce
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.f29
// CHECK:STDOUT:   %BitXorWith.lookup_impl_witness => constants.%BitXorWith.impl_witness.45f
// CHECK:STDOUT:   %BitXorWith.facet => constants.%BitXorWith.facet.f38
// CHECK:STDOUT:   %BitXorWith.WithSelf.Op.type => constants.%BitXorWith.WithSelf.Op.type.f48
// CHECK:STDOUT:   %.loc67_12.2 => constants.%.434
// CHECK:STDOUT:   %impl.elem1.loc67_12.2 => constants.%C.as.BitXorWith.impl.Op
// CHECK:STDOUT:   %specific_impl_fn.loc67_12.2 => constants.%C.as.BitXorWith.impl.Op
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGenericAssign(constants.%BitXorAssign.facet.3ca) {
// CHECK:STDOUT:   %T.patt.loc71_23.2 => constants.%T.patt.3a0592.2
// CHECK:STDOUT:   %T.loc71_23.1 => constants.%BitXorAssign.facet.3ca
// CHECK:STDOUT:   %T.as_type.loc71_48.1 => constants.%C
// CHECK:STDOUT:   %ptr.loc71_48.1 => constants.%ptr.00e
// CHECK:STDOUT:   %pattern_type.loc71_45 => constants.%pattern_type.842
// CHECK:STDOUT:   %a.param_patt.loc71_45.2 => constants.%a.param_patt.4fa
// CHECK:STDOUT:   %a.patt.loc71_45.2 => constants.%a.patt.eba
// CHECK:STDOUT:   %pattern_type.loc71_52 => constants.%pattern_type.4a7
// CHECK:STDOUT:   %b.param_patt.loc71_52.2 => constants.%b.param_patt.45f
// CHECK:STDOUT:   %b.patt.loc71_52.2 => constants.%b.patt.562
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc71_45 => constants.%complete_type.a56
// CHECK:STDOUT:   %require_complete.loc71_52 => constants.%complete_type.357
// CHECK:STDOUT:   %BitXorAssignWith.type.loc73_6.2 => constants.%BitXorAssignWith.type.b89
// CHECK:STDOUT:   %require_complete.loc73 => constants.%complete_type.f05
// CHECK:STDOUT:   %BitXorAssignWith.assoc_type => constants.%BitXorAssignWith.assoc_type.ec9
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.5eb
// CHECK:STDOUT:   %BitXorAssignWith.lookup_impl_witness => constants.%BitXorAssignWith.impl_witness.05c
// CHECK:STDOUT:   %BitXorAssignWith.facet => constants.%BitXorAssignWith.facet.fc8
// CHECK:STDOUT:   %BitXorAssignWith.WithSelf.Op.type => constants.%BitXorAssignWith.WithSelf.Op.type.ffb
// CHECK:STDOUT:   %.loc73_6.2 => constants.%.e48
// CHECK:STDOUT:   %impl.elem0.loc73_6.2 => constants.%C.as.BitXorAssignWith.impl.Op
// CHECK:STDOUT:   %specific_impl_fn.loc73_6.2 => constants.%C.as.BitXorAssignWith.impl.Op
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGenericOp(constants.%BitXor.facet.6f9) {
// CHECK:STDOUT:   %T.patt.loc65_19.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc65_19.1 => constants.%BitXor.facet.6f9
// CHECK:STDOUT:   %T.as_type.loc65_37.1 => constants.%D
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.331
// CHECK:STDOUT:   %a.param_patt.loc65_35.2 => constants.%a.param_patt.d0e
// CHECK:STDOUT:   %a.patt.loc65_35.2 => constants.%a.patt.ec5
// CHECK:STDOUT:   %b.param_patt.loc65_41.2 => constants.%b.param_patt.110
// CHECK:STDOUT:   %b.patt.loc65_41.2 => constants.%b.patt.14c
// CHECK:STDOUT:   %.loc65_49.2 => constants.%.84d
// CHECK:STDOUT:   %return.param_patt.loc65_49.2 => constants.%return.param_patt.835
// CHECK:STDOUT:   %return.patt.loc65_46.2 => constants.%return.patt.5ef
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc65 => constants.%complete_type.357
// CHECK:STDOUT:   %BitXorWith.type.loc67_12.2 => constants.%BitXorWith.type.95e
// CHECK:STDOUT:   %require_complete.loc67 => constants.%complete_type.b51
// CHECK:STDOUT:   %BitXorWith.assoc_type => constants.%BitXorWith.assoc_type.a37
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.dc9
// CHECK:STDOUT:   %BitXorWith.lookup_impl_witness => constants.%BitXorWith.impl_witness.736
// CHECK:STDOUT:   %BitXorWith.facet => constants.%BitXorWith.facet.4af
// CHECK:STDOUT:   %BitXorWith.WithSelf.Op.type => constants.%BitXorWith.WithSelf.Op.type.289
// CHECK:STDOUT:   %.loc67_12.2 => constants.%.b0b
// CHECK:STDOUT:   %impl.elem1.loc67_12.2 => constants.%D.as.BitXorWith.impl.Op
// CHECK:STDOUT:   %specific_impl_fn.loc67_12.2 => constants.%D.as.BitXorWith.impl.Op
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGenericAssign(constants.%BitXorAssign.facet.6f9) {
// CHECK:STDOUT:   %T.patt.loc71_23.2 => constants.%T.patt.3a0592.2
// CHECK:STDOUT:   %T.loc71_23.1 => constants.%BitXorAssign.facet.6f9
// CHECK:STDOUT:   %T.as_type.loc71_48.1 => constants.%D
// CHECK:STDOUT:   %ptr.loc71_48.1 => constants.%ptr.4a5
// CHECK:STDOUT:   %pattern_type.loc71_45 => constants.%pattern_type.ceb
// CHECK:STDOUT:   %a.param_patt.loc71_45.2 => constants.%a.param_patt.609
// CHECK:STDOUT:   %a.patt.loc71_45.2 => constants.%a.patt.fac
// CHECK:STDOUT:   %pattern_type.loc71_52 => constants.%pattern_type.331
// CHECK:STDOUT:   %b.param_patt.loc71_52.2 => constants.%b.param_patt.110
// CHECK:STDOUT:   %b.patt.loc71_52.2 => constants.%b.patt.14c
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc71_45 => constants.%complete_type.320
// CHECK:STDOUT:   %require_complete.loc71_52 => constants.%complete_type.357
// CHECK:STDOUT:   %BitXorAssignWith.type.loc73_6.2 => constants.%BitXorAssignWith.type.4c4
// CHECK:STDOUT:   %require_complete.loc73 => constants.%complete_type.ce2
// CHECK:STDOUT:   %BitXorAssignWith.assoc_type => constants.%BitXorAssignWith.assoc_type.545
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.9ee
// CHECK:STDOUT:   %BitXorAssignWith.lookup_impl_witness => constants.%BitXorAssignWith.impl_witness.736
// CHECK:STDOUT:   %BitXorAssignWith.facet => constants.%BitXorAssignWith.facet.58a
// CHECK:STDOUT:   %BitXorAssignWith.WithSelf.Op.type => constants.%BitXorAssignWith.WithSelf.Op.type.1eb
// CHECK:STDOUT:   %.loc73_6.2 => constants.%.61a
// CHECK:STDOUT:   %impl.elem0.loc73_6.2 => constants.%D.as.BitXorAssignWith.impl.Op
// CHECK:STDOUT:   %specific_impl_fn.loc73_6.2 => constants.%D.as.BitXorAssignWith.impl.Op
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGenericOp(constants.%BitXor.facet.82f) {
// CHECK:STDOUT:   %T.patt.loc65_19.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc65_19.1 => constants.%BitXor.facet.82f
// CHECK:STDOUT:   %T.as_type.loc65_37.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.d94
// CHECK:STDOUT:   %a.param_patt.loc65_35.2 => constants.%a.param_patt.076
// CHECK:STDOUT:   %a.patt.loc65_35.2 => constants.%a.patt.96f
// CHECK:STDOUT:   %b.param_patt.loc65_41.2 => constants.%b.param_patt.1a2
// CHECK:STDOUT:   %b.patt.loc65_41.2 => constants.%b.patt.3ca
// CHECK:STDOUT:   %.loc65_49.2 => constants.%.0b3
// CHECK:STDOUT:   %return.param_patt.loc65_49.2 => constants.%return.param_patt.299
// CHECK:STDOUT:   %return.patt.loc65_46.2 => constants.%return.patt.6fa
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc65 => constants.%complete_type.f8a
// CHECK:STDOUT:   %BitXorWith.type.loc67_12.2 => constants.%BitXorWith.type.bfb
// CHECK:STDOUT:   %require_complete.loc67 => constants.%complete_type.f01
// CHECK:STDOUT:   %BitXorWith.assoc_type => constants.%BitXorWith.assoc_type.38b
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.e08
// CHECK:STDOUT:   %BitXorWith.lookup_impl_witness => constants.%BitXorWith.impl_witness.5c7
// CHECK:STDOUT:   %BitXorWith.facet => constants.%BitXorWith.facet.365
// CHECK:STDOUT:   %BitXorWith.WithSelf.Op.type => constants.%BitXorWith.WithSelf.Op.type.53c
// CHECK:STDOUT:   %.loc67_12.2 => constants.%.3d5
// CHECK:STDOUT:   %impl.elem1.loc67_12.2 => constants.%Int.as.BitXorWith.impl.Op.153
// CHECK:STDOUT:   %specific_impl_fn.loc67_12.2 => constants.%Int.as.BitXorWith.impl.Op.specific_fn.3a2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGenericAssign(constants.%BitXorAssign.facet.fc2) {
// CHECK:STDOUT:   %T.patt.loc71_23.2 => constants.%T.patt.3a0592.2
// CHECK:STDOUT:   %T.loc71_23.1 => constants.%BitXorAssign.facet.fc2
// CHECK:STDOUT:   %T.as_type.loc71_48.1 => constants.%i32
// CHECK:STDOUT:   %ptr.loc71_48.1 => constants.%ptr.a17
// CHECK:STDOUT:   %pattern_type.loc71_45 => constants.%pattern_type.7dc
// CHECK:STDOUT:   %a.param_patt.loc71_45.2 => constants.%a.param_patt.977
// CHECK:STDOUT:   %a.patt.loc71_45.2 => constants.%a.patt.b70
// CHECK:STDOUT:   %pattern_type.loc71_52 => constants.%pattern_type.d94
// CHECK:STDOUT:   %b.param_patt.loc71_52.2 => constants.%b.param_patt.1a2
// CHECK:STDOUT:   %b.patt.loc71_52.2 => constants.%b.patt.3ca
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc71_45 => constants.%complete_type.1b1
// CHECK:STDOUT:   %require_complete.loc71_52 => constants.%complete_type.f8a
// CHECK:STDOUT:   %BitXorAssignWith.type.loc73_6.2 => constants.%BitXorAssignWith.type.e76
// CHECK:STDOUT:   %require_complete.loc73 => constants.%complete_type.baf
// CHECK:STDOUT:   %BitXorAssignWith.assoc_type => constants.%BitXorAssignWith.assoc_type.874
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.5f9
// CHECK:STDOUT:   %BitXorAssignWith.lookup_impl_witness => constants.%BitXorAssignWith.impl_witness.b11
// CHECK:STDOUT:   %BitXorAssignWith.facet => constants.%BitXorAssignWith.facet.0e5
// CHECK:STDOUT:   %BitXorAssignWith.WithSelf.Op.type => constants.%BitXorAssignWith.WithSelf.Op.type.944
// CHECK:STDOUT:   %.loc73_6.2 => constants.%.a85
// CHECK:STDOUT:   %impl.elem0.loc73_6.2 => constants.%Int.as.BitXorAssignWith.impl.Op.62f34b.2
// CHECK:STDOUT:   %specific_impl_fn.loc73_6.2 => constants.%Int.as.BitXorAssignWith.impl.Op.specific_fn.5f6c9b.1
// CHECK:STDOUT: }
// CHECK:STDOUT:

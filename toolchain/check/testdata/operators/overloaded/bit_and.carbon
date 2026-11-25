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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/overloaded/bit_and.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/overloaded/bit_and.carbon

// This file was generated from binary_op.carbon.tmpl. Run make_tests.sh to regenerate.

package User;

class C {};

impl C as Core.BitAndWith(C) where .Result = C {
  fn Op(unused self: C, unused other: C) -> C {
    return {};
  }
}
impl C as Core.BitAndAssignWith(C) {
  fn Op(unused ref self: C, unused other: C) {}
}

fn TestOp(a: C, b: C) -> C {
  //@dump-sem-ir-begin
  return a & b;
  //@dump-sem-ir-end
}

fn TestAssign(a: C*, b: C) {
  //@dump-sem-ir-begin
  *a &= b;
  //@dump-sem-ir-end
}

class D {};

impl D as Core.BitAnd {
  fn Op(unused self: D, unused other: D) -> D {
    return {};
  }
}
impl D as Core.BitAndAssign {
  fn Op(unused ref self: D, unused other: D) {}
}

fn TestNamedConstraintOp(a: D, b: D) -> D {
  //@dump-sem-ir-begin
  return a & b;
  //@dump-sem-ir-end
}

fn TestNamedConstraintAssign(a: D*, b: D) {
  //@dump-sem-ir-begin
  *a &= b;
  //@dump-sem-ir-end
}

fn TestGenericOp[T: Core.BitAnd](a: T, b: T) -> T {
  //@dump-sem-ir-begin
  return a & b;
  //@dump-sem-ir-end
}

fn TestGenericAssign[T: Core.BitAndAssign](a: T*, b: T) {
  //@dump-sem-ir-begin
  *a &= b;
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

// CHECK:STDOUT: --- bit_and.carbon = $bit_and
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Other: type = symbolic_binding Other, 0 [symbolic]
// CHECK:STDOUT:   %BitAndWith.type.221: type = facet_type <@BitAndWith, @BitAndWith(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.48b: %BitAndWith.type.221 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %BitAndWith.assoc_type.657: type = assoc_entity_type @BitAndWith, @BitAndWith(%Other) [symbolic]
// CHECK:STDOUT:   %BitAndWith.WithSelf.Op.type.273: type = fn_type @BitAndWith.WithSelf.Op, @BitAndWith.WithSelf(%Other, %Self.48b) [symbolic]
// CHECK:STDOUT:   %BitAndWith.WithSelf.Op.cf3: %BitAndWith.WithSelf.Op.type.273 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitAndWith.type.01b: type = facet_type <@BitAndWith, @BitAndWith(%C)> [concrete]
// CHECK:STDOUT:   %BitAndWith.assoc_type.409: type = assoc_entity_type @BitAndWith, @BitAndWith(%C) [concrete]
// CHECK:STDOUT:   %assoc1.ea0: %BitAndWith.assoc_type.409 = assoc_entity element1, imports.%Core.import_ref.7c1 [concrete]
// CHECK:STDOUT:   %BitAndWith.impl_witness.b9f: <witness> = impl_witness @C.as.BitAndWith.impl.%BitAndWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.4a7: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %.79c: Core.Form = init_form %C [concrete]
// CHECK:STDOUT:   %return.param_patt.415: %pattern_type.4a7 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.340: %pattern_type.4a7 = return_slot_pattern %return.param_patt.415, %C [concrete]
// CHECK:STDOUT:   %C.as.BitAndWith.impl.Op.type: type = fn_type @C.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.BitAndWith.impl.Op: %C.as.BitAndWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndWith.facet.423: %BitAndWith.type.01b = facet_value %C, (%BitAndWith.impl_witness.b9f) [concrete]
// CHECK:STDOUT:   %BitAndWith.WithSelf.Op.type.c29: type = fn_type @BitAndWith.WithSelf.Op, @BitAndWith.WithSelf(%C, %BitAndWith.facet.423) [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.type.60d: type = facet_type <@BitAndAssignWith, @BitAndAssignWith(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.591: %BitAndAssignWith.type.60d = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %BitAndAssignWith.assoc_type.867: type = assoc_entity_type @BitAndAssignWith, @BitAndAssignWith(%Other) [symbolic]
// CHECK:STDOUT:   %BitAndAssignWith.WithSelf.Op.type.e9d: type = fn_type @BitAndAssignWith.WithSelf.Op, @BitAndAssignWith.WithSelf(%Other, %Self.591) [symbolic]
// CHECK:STDOUT:   %BitAndAssignWith.WithSelf.Op.13a: %BitAndAssignWith.WithSelf.Op.type.e9d = struct_value () [symbolic]
// CHECK:STDOUT:   %BitAndAssignWith.type.0a8: type = facet_type <@BitAndAssignWith, @BitAndAssignWith(%C)> [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.impl_witness.ea1: <witness> = impl_witness @C.as.BitAndAssignWith.impl.%BitAndAssignWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.assoc_type.c8f: type = assoc_entity_type @BitAndAssignWith, @BitAndAssignWith(%C) [concrete]
// CHECK:STDOUT:   %assoc0.9a4: %BitAndAssignWith.assoc_type.c8f = assoc_entity element0, imports.%Core.import_ref.7e7 [concrete]
// CHECK:STDOUT:   %C.as.BitAndAssignWith.impl.Op.type: type = fn_type @C.as.BitAndAssignWith.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.BitAndAssignWith.impl.Op: %C.as.BitAndAssignWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.facet.edd: %BitAndAssignWith.type.0a8 = facet_value %C, (%BitAndAssignWith.impl_witness.ea1) [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.WithSelf.Op.type.fe6: type = fn_type @BitAndAssignWith.WithSelf.Op, @BitAndAssignWith.WithSelf(%C, %BitAndAssignWith.facet.edd) [concrete]
// CHECK:STDOUT:   %a.param_patt.2a5: %pattern_type.4a7 = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt.a30: %pattern_type.4a7 = wrapper_binding_pattern a, %a.param_patt.2a5 [concrete]
// CHECK:STDOUT:   %b.param_patt.45f: %pattern_type.4a7 = value_param_pattern [concrete]
// CHECK:STDOUT:   %b.patt.562: %pattern_type.4a7 = wrapper_binding_pattern b, %b.param_patt.45f [concrete]
// CHECK:STDOUT:   %assoc1.9bb: %BitAndWith.assoc_type.657 = assoc_entity element1, imports.%Core.import_ref.0f5 [symbolic]
// CHECK:STDOUT:   %.df9: type = fn_type_with_self_type %BitAndWith.WithSelf.Op.type.c29, %BitAndWith.facet.423 [concrete]
// CHECK:STDOUT:   %ptr.00e: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.842: type = pattern_type %ptr.00e [concrete]
// CHECK:STDOUT:   %a.param_patt.4fa: %pattern_type.842 = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt.eba: %pattern_type.842 = wrapper_binding_pattern a, %a.param_patt.4fa [concrete]
// CHECK:STDOUT:   %assoc0.b97: %BitAndAssignWith.assoc_type.867 = assoc_entity element0, imports.%Core.import_ref.012 [symbolic]
// CHECK:STDOUT:   %.09a: type = fn_type_with_self_type %BitAndAssignWith.WithSelf.Op.type.fe6, %BitAndAssignWith.facet.edd [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %BitAnd.type: type = facet_type <@BitAnd> [concrete]
// CHECK:STDOUT:   %BitAndWith.impl_witness.736: <witness> = impl_witness @D.as.BitAndWith.impl.%BitAndWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.331: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %.84d: Core.Form = init_form %D [concrete]
// CHECK:STDOUT:   %return.param_patt.835: %pattern_type.331 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.5ef: %pattern_type.331 = return_slot_pattern %return.param_patt.835, %D [concrete]
// CHECK:STDOUT:   %D.as.BitAndWith.impl.Op.type: type = fn_type @D.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %D.as.BitAndWith.impl.Op: %D.as.BitAndWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndWith.type.6fa: type = facet_type <@BitAndWith, @BitAndWith(%D)> [concrete]
// CHECK:STDOUT:   %BitAndWith.facet.744: %BitAndWith.type.6fa = facet_value %D, (%BitAndWith.impl_witness.736) [concrete]
// CHECK:STDOUT:   %BitAndWith.assoc_type.aa1: type = assoc_entity_type @BitAndWith, @BitAndWith(%D) [concrete]
// CHECK:STDOUT:   %BitAndWith.WithSelf.Op.type.20b: type = fn_type @BitAndWith.WithSelf.Op, @BitAndWith.WithSelf(%D, %BitAndWith.facet.744) [concrete]
// CHECK:STDOUT:   %assoc1.1eb: %BitAndWith.assoc_type.aa1 = assoc_entity element1, imports.%Core.import_ref.7c1 [concrete]
// CHECK:STDOUT:   %BitAndAssign.type: type = facet_type <@BitAndAssign> [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.impl_witness.736: <witness> = impl_witness @D.as.BitAndAssignWith.impl.%BitAndAssignWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %D.as.BitAndAssignWith.impl.Op.type: type = fn_type @D.as.BitAndAssignWith.impl.Op [concrete]
// CHECK:STDOUT:   %D.as.BitAndAssignWith.impl.Op: %D.as.BitAndAssignWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.type.59f: type = facet_type <@BitAndAssignWith, @BitAndAssignWith(%D)> [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.facet.c70: %BitAndAssignWith.type.59f = facet_value %D, (%BitAndAssignWith.impl_witness.736) [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.WithSelf.Op.type.64c: type = fn_type @BitAndAssignWith.WithSelf.Op, @BitAndAssignWith.WithSelf(%D, %BitAndAssignWith.facet.c70) [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.assoc_type.dd9: type = assoc_entity_type @BitAndAssignWith, @BitAndAssignWith(%D) [concrete]
// CHECK:STDOUT:   %assoc0.003: %BitAndAssignWith.assoc_type.dd9 = assoc_entity element0, imports.%Core.import_ref.7e7 [concrete]
// CHECK:STDOUT:   %a.param_patt.d0e: %pattern_type.331 = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt.ec5: %pattern_type.331 = wrapper_binding_pattern a, %a.param_patt.d0e [concrete]
// CHECK:STDOUT:   %b.param_patt.110: %pattern_type.331 = value_param_pattern [concrete]
// CHECK:STDOUT:   %b.patt.14c: %pattern_type.331 = wrapper_binding_pattern b, %b.param_patt.110 [concrete]
// CHECK:STDOUT:   %.315: type = fn_type_with_self_type %BitAndWith.WithSelf.Op.type.20b, %BitAndWith.facet.744 [concrete]
// CHECK:STDOUT:   %ptr.4a5: type = ptr_type %D [concrete]
// CHECK:STDOUT:   %pattern_type.ceb: type = pattern_type %ptr.4a5 [concrete]
// CHECK:STDOUT:   %a.param_patt.609: %pattern_type.ceb = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt.fac: %pattern_type.ceb = wrapper_binding_pattern a, %a.param_patt.609 [concrete]
// CHECK:STDOUT:   %.707: type = fn_type_with_self_type %BitAndAssignWith.WithSelf.Op.type.64c, %BitAndAssignWith.facet.c70 [concrete]
// CHECK:STDOUT:   %pattern_type.9a587c.2: type = pattern_type %BitAnd.type [concrete]
// CHECK:STDOUT:   %T.patt.3a0592.1: %pattern_type.9a587c.2 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.f45530.1: %BitAnd.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.190bb7.1: type = facet_access_type %T.f45530.1 [symbolic]
// CHECK:STDOUT:   %pattern_type.86a9db.1: type = pattern_type %T.as_type.190bb7.1 [symbolic]
// CHECK:STDOUT:   %a.param_patt.e33: %pattern_type.86a9db.1 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %a.patt.e02: %pattern_type.86a9db.1 = wrapper_binding_pattern a, %a.param_patt.e33 [symbolic]
// CHECK:STDOUT:   %b.param_patt.4cd741.1: %pattern_type.86a9db.1 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %b.patt.60acf6.1: %pattern_type.86a9db.1 = wrapper_binding_pattern b, %b.param_patt.4cd741.1 [symbolic]
// CHECK:STDOUT:   %.1f0: Core.Form = init_form %T.as_type.190bb7.1 [symbolic]
// CHECK:STDOUT:   %return.param_patt.5f6: %pattern_type.86a9db.1 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.96a: %pattern_type.86a9db.1 = return_slot_pattern %return.param_patt.5f6, %T.as_type.190bb7.1 [symbolic]
// CHECK:STDOUT:   %BitAndWith.type.bbeb52.2: type = facet_type <@BitAndWith, @BitAndWith(%T.as_type.190bb7.1)> [symbolic]
// CHECK:STDOUT:   %BitAndWith.assoc_type.884223.2: type = assoc_entity_type @BitAndWith, @BitAndWith(%T.as_type.190bb7.1) [symbolic]
// CHECK:STDOUT:   %assoc1.b5b429.2: %BitAndWith.assoc_type.884223.2 = assoc_entity element1, imports.%Core.import_ref.7c1 [symbolic]
// CHECK:STDOUT:   %require_complete.8b5: <witness> = require_complete_type %BitAndWith.type.bbeb52.2 [symbolic]
// CHECK:STDOUT:   %BitAndWith.lookup_impl_witness.3a3af1.2: <witness> = lookup_impl_witness %T.f45530.1, @BitAndWith, @BitAndWith(%T.as_type.190bb7.1) [symbolic]
// CHECK:STDOUT:   %BitAndWith.facet.948: %BitAndWith.type.bbeb52.2 = facet_value %T.as_type.190bb7.1, (%BitAndWith.lookup_impl_witness.3a3af1.2) [symbolic]
// CHECK:STDOUT:   %BitAndWith.WithSelf.Op.type.a48: type = fn_type @BitAndWith.WithSelf.Op, @BitAndWith.WithSelf(%T.as_type.190bb7.1, %BitAndWith.facet.948) [symbolic]
// CHECK:STDOUT:   %.daa: type = fn_type_with_self_type %BitAndWith.WithSelf.Op.type.a48, %BitAndWith.facet.948 [symbolic]
// CHECK:STDOUT:   %impl.elem1.a77: %.daa = impl_witness_access %BitAndWith.lookup_impl_witness.3a3af1.2, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.7aa: <specific function> = specific_impl_function %impl.elem1.a77, @BitAndWith.WithSelf.Op(%T.as_type.190bb7.1, %BitAndWith.facet.948) [symbolic]
// CHECK:STDOUT:   %pattern_type.9a587c.3: type = pattern_type %BitAndAssign.type [concrete]
// CHECK:STDOUT:   %T.patt.3a0592.2: %pattern_type.9a587c.3 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.f45530.2: %BitAndAssign.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.190bb7.2: type = facet_access_type %T.f45530.2 [symbolic]
// CHECK:STDOUT:   %ptr.360: type = ptr_type %T.as_type.190bb7.2 [symbolic]
// CHECK:STDOUT:   %pattern_type.ff5: type = pattern_type %ptr.360 [symbolic]
// CHECK:STDOUT:   %a.param_patt.f50: %pattern_type.ff5 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %a.patt.193: %pattern_type.ff5 = wrapper_binding_pattern a, %a.param_patt.f50 [symbolic]
// CHECK:STDOUT:   %pattern_type.86a9db.2: type = pattern_type %T.as_type.190bb7.2 [symbolic]
// CHECK:STDOUT:   %b.param_patt.4cd741.2: %pattern_type.86a9db.2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %b.patt.60acf6.2: %pattern_type.86a9db.2 = wrapper_binding_pattern b, %b.param_patt.4cd741.2 [symbolic]
// CHECK:STDOUT:   %BitAndAssignWith.type.9743d5.2: type = facet_type <@BitAndAssignWith, @BitAndAssignWith(%T.as_type.190bb7.2)> [symbolic]
// CHECK:STDOUT:   %BitAndAssignWith.assoc_type.fc666d.2: type = assoc_entity_type @BitAndAssignWith, @BitAndAssignWith(%T.as_type.190bb7.2) [symbolic]
// CHECK:STDOUT:   %assoc0.b4dd45.2: %BitAndAssignWith.assoc_type.fc666d.2 = assoc_entity element0, imports.%Core.import_ref.7e7 [symbolic]
// CHECK:STDOUT:   %require_complete.247c45.2: <witness> = require_complete_type %BitAndAssignWith.type.9743d5.2 [symbolic]
// CHECK:STDOUT:   %BitAndAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.f45530.2, @BitAndAssignWith, @BitAndAssignWith(%T.as_type.190bb7.2) [symbolic]
// CHECK:STDOUT:   %BitAndAssignWith.facet.420: %BitAndAssignWith.type.9743d5.2 = facet_value %T.as_type.190bb7.2, (%BitAndAssignWith.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %BitAndAssignWith.WithSelf.Op.type.e07: type = fn_type @BitAndAssignWith.WithSelf.Op, @BitAndAssignWith.WithSelf(%T.as_type.190bb7.2, %BitAndAssignWith.facet.420) [symbolic]
// CHECK:STDOUT:   %.854: type = fn_type_with_self_type %BitAndAssignWith.WithSelf.Op.type.e07, %BitAndAssignWith.facet.420 [symbolic]
// CHECK:STDOUT:   %impl.elem0.86c: %.854 = impl_witness_access %BitAndAssignWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.0ff: <specific function> = specific_impl_function %impl.elem0.86c, @BitAndAssignWith.WithSelf.Op(%T.as_type.190bb7.2, %BitAndAssignWith.facet.420) [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %Int.d4bbe0.1: type = class_type @Int, @Int(%N) [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %ptr.a17: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %ptr.a17 [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %BitAnd.facet.50e: %BitAnd.type = facet_value %C, (%BitAndWith.impl_witness.b9f) [concrete]
// CHECK:STDOUT:   %BitAndAssign.facet.4df: %BitAndAssign.type = facet_value %C, (%BitAndAssignWith.impl_witness.ea1) [concrete]
// CHECK:STDOUT:   %BitAnd.facet.6f9: %BitAnd.type = facet_value %D, (%BitAndWith.impl_witness.736) [concrete]
// CHECK:STDOUT:   %BitAndAssign.facet.6f9: %BitAndAssign.type = facet_value %D, (%BitAndAssignWith.impl_witness.736) [concrete]
// CHECK:STDOUT:   %Int.as.BitAndWith.impl.Op.type.919: type = fn_type @Int.as.BitAndWith.impl.Op.1, @Int.as.BitAndWith.impl.136(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.BitAndWith.impl.Op.8d9: %Int.as.BitAndWith.impl.Op.type.919 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.81c7d0.1: type = facet_type <@ImplicitAs, @ImplicitAs(%Int.d4bbe0.1)> [symbolic]
// CHECK:STDOUT:   %U.f16: %ImplicitAs.type.81c7d0.1 = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %BitAndWith.type.2ea: type = facet_type <@BitAndWith, @BitAndWith(%i32)> [concrete]
// CHECK:STDOUT:   %BitAndWith.impl_witness.128: <witness> = impl_witness imports.%BitAndWith.impl_witness_table.949, @Int.as.BitAndWith.impl.136(%int_32) [concrete]
// CHECK:STDOUT:   %BitAndWith.assoc_type.4cc: type = assoc_entity_type @BitAndWith, @BitAndWith(%i32) [concrete]
// CHECK:STDOUT:   %assoc1.342: %BitAndWith.assoc_type.4cc = assoc_entity element1, imports.%Core.import_ref.7c1 [concrete]
// CHECK:STDOUT:   %complete_type.b42: <witness> = complete_type_witness %BitAndWith.type.2ea [concrete]
// CHECK:STDOUT:   %Int.as.BitAndWith.impl.Op.type.b33: type = fn_type @Int.as.BitAndWith.impl.Op.1, @Int.as.BitAndWith.impl.136(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.BitAndWith.impl.Op.272: %Int.as.BitAndWith.impl.Op.type.b33 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAnd.facet.8cd: %BitAnd.type = facet_value %i32, (%BitAndWith.impl_witness.128) [concrete]
// CHECK:STDOUT:   %a.param_patt.076: %pattern_type.d94 = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt.96f: %pattern_type.d94 = wrapper_binding_pattern a, %a.param_patt.076 [concrete]
// CHECK:STDOUT:   %b.param_patt.1a2: %pattern_type.d94 = value_param_pattern [concrete]
// CHECK:STDOUT:   %b.patt.3ca: %pattern_type.d94 = wrapper_binding_pattern b, %b.param_patt.1a2 [concrete]
// CHECK:STDOUT:   %.0b3: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.299: %pattern_type.d94 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.6fa: %pattern_type.d94 = return_slot_pattern %return.param_patt.299, %i32 [concrete]
// CHECK:STDOUT:   %Int.as.BitAndAssignWith.impl.Op.type.bc5330.1: type = fn_type @Int.as.BitAndAssignWith.impl.Op.1, @Int.as.BitAndAssignWith.impl(%N, %U.f16) [symbolic]
// CHECK:STDOUT:   %Int.as.BitAndAssignWith.impl.Op.7b2ec1.1: %Int.as.BitAndAssignWith.impl.Op.type.bc5330.1 = struct_value () [symbolic]
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
// CHECK:STDOUT:   %BitAndAssignWith.type.f47: type = facet_type <@BitAndAssignWith, @BitAndAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.impl_witness.f23: <witness> = impl_witness imports.%BitAndAssignWith.impl_witness_table, @Int.as.BitAndAssignWith.impl(%int_32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %BitAndAssign.facet.d8d: %BitAndAssign.type = facet_value %i32, (%BitAndAssignWith.impl_witness.f23) [concrete]
// CHECK:STDOUT:   %a.param_patt.977: %pattern_type.7dc = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt.b70: %pattern_type.7dc = wrapper_binding_pattern a, %a.param_patt.977 [concrete]
// CHECK:STDOUT:   %complete_type.8d4: <witness> = complete_type_witness %BitAndWith.type.01b [concrete]
// CHECK:STDOUT:   %complete_type.a56: <witness> = complete_type_witness %ptr.00e [concrete]
// CHECK:STDOUT:   %complete_type.b80: <witness> = complete_type_witness %BitAndAssignWith.type.0a8 [concrete]
// CHECK:STDOUT:   %complete_type.64d: <witness> = complete_type_witness %BitAndWith.type.6fa [concrete]
// CHECK:STDOUT:   %complete_type.320: <witness> = complete_type_witness %ptr.4a5 [concrete]
// CHECK:STDOUT:   %complete_type.d4a: <witness> = complete_type_witness %BitAndAssignWith.type.59f [concrete]
// CHECK:STDOUT:   %BitAndWith.facet.b1d: %BitAndWith.type.2ea = facet_value %i32, (%BitAndWith.impl_witness.128) [concrete]
// CHECK:STDOUT:   %BitAndWith.WithSelf.Op.type.03c: type = fn_type @BitAndWith.WithSelf.Op, @BitAndWith.WithSelf(%i32, %BitAndWith.facet.b1d) [concrete]
// CHECK:STDOUT:   %.a35: type = fn_type_with_self_type %BitAndWith.WithSelf.Op.type.03c, %BitAndWith.facet.b1d [concrete]
// CHECK:STDOUT:   %Int.as.BitAndWith.impl.Op.specific_fn.fbb: <specific function> = specific_function %Int.as.BitAndWith.impl.Op.272, @Int.as.BitAndWith.impl.Op.1(%int_32) [concrete]
// CHECK:STDOUT:   %complete_type.1b1: <witness> = complete_type_witness %ptr.a17 [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.assoc_type.875: type = assoc_entity_type @BitAndAssignWith, @BitAndAssignWith(%i32) [concrete]
// CHECK:STDOUT:   %assoc0.f312: %BitAndAssignWith.assoc_type.875 = assoc_entity element0, imports.%Core.import_ref.7e7 [concrete]
// CHECK:STDOUT:   %complete_type.cee: <witness> = complete_type_witness %BitAndAssignWith.type.f47 [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.facet.cb4: %BitAndAssignWith.type.f47 = facet_value %i32, (%BitAndAssignWith.impl_witness.f23) [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.WithSelf.Op.type.74a: type = fn_type @BitAndAssignWith.WithSelf.Op, @BitAndAssignWith.WithSelf(%i32, %BitAndAssignWith.facet.cb4) [concrete]
// CHECK:STDOUT:   %.2db: type = fn_type_with_self_type %BitAndAssignWith.WithSelf.Op.type.74a, %BitAndAssignWith.facet.cb4 [concrete]
// CHECK:STDOUT:   %Int.as.BitAndAssignWith.impl.Op.type.d7e008.2: type = fn_type @Int.as.BitAndAssignWith.impl.Op.1, @Int.as.BitAndAssignWith.impl(%int_32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Int.as.BitAndAssignWith.impl.Op.43022d.2: %Int.as.BitAndAssignWith.impl.Op.type.d7e008.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.BitAndAssignWith.impl.Op.specific_fn.6ac8d6.1: <specific function> = specific_function %Int.as.BitAndAssignWith.impl.Op.43022d.2, @Int.as.BitAndAssignWith.impl.Op.1(%int_32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.e78: @BitAndWith.WithSelf.%BitAndWith.assoc_type (%BitAndWith.assoc_type.657) = import_ref Core//prelude/operators/bitwise, loc{{\d+_\d+}}, loaded [symbolic = @BitAndWith.WithSelf.%assoc1 (constants.%assoc1.9bb)]
// CHECK:STDOUT:   %Core.import_ref.7c1: @BitAndWith.WithSelf.%BitAndWith.WithSelf.Op.type (%BitAndWith.WithSelf.Op.type.273) = import_ref Core//prelude/operators/bitwise, loc{{\d+_\d+}}, loaded [symbolic = @BitAndWith.WithSelf.%BitAndWith.WithSelf.Op (constants.%BitAndWith.WithSelf.Op.cf3)]
// CHECK:STDOUT:   %Core.import_ref.7b10: @BitAndAssignWith.WithSelf.%BitAndAssignWith.assoc_type (%BitAndAssignWith.assoc_type.867) = import_ref Core//prelude/operators/bitwise, loc{{\d+_\d+}}, loaded [symbolic = @BitAndAssignWith.WithSelf.%assoc0 (constants.%assoc0.b97)]
// CHECK:STDOUT:   %Core.import_ref.7e7: @BitAndAssignWith.WithSelf.%BitAndAssignWith.WithSelf.Op.type (%BitAndAssignWith.WithSelf.Op.type.e9d) = import_ref Core//prelude/operators/bitwise, loc{{\d+_\d+}}, loaded [symbolic = @BitAndAssignWith.WithSelf.%BitAndAssignWith.WithSelf.Op (constants.%BitAndAssignWith.WithSelf.Op.13a)]
// CHECK:STDOUT:   %Core.import_ref.0f5 = import_ref Core//prelude/operators/bitwise, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.012 = import_ref Core//prelude/operators/bitwise, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.ebd: type = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.BitAndWith.impl.136.%Int (constants.%Int.d4bbe0.1)]
// CHECK:STDOUT:   %Core.import_ref.3c0: @Int.as.BitAndWith.impl.136.%Int.as.BitAndWith.impl.Op.type (%Int.as.BitAndWith.impl.Op.type.919) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.BitAndWith.impl.136.%Int.as.BitAndWith.impl.Op (constants.%Int.as.BitAndWith.impl.Op.8d9)]
// CHECK:STDOUT:   %BitAndWith.impl_witness_table.949 = impl_witness_table (%Core.import_ref.ebd, %Core.import_ref.3c0), @Int.as.BitAndWith.impl.136 [concrete]
// CHECK:STDOUT:   %Core.import_ref.5fa: @Int.as.BitAndAssignWith.impl.%Int.as.BitAndAssignWith.impl.Op.type.2 (%Int.as.BitAndAssignWith.impl.Op.type.bc5330.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.BitAndAssignWith.impl.%Int.as.BitAndAssignWith.impl.Op.2 (constants.%Int.as.BitAndAssignWith.impl.Op.7b2ec1.1)]
// CHECK:STDOUT:   %BitAndAssignWith.impl_witness_table = impl_witness_table (%Core.import_ref.5fa), @Int.as.BitAndAssignWith.impl [concrete]
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
// CHECK:STDOUT:   %impl.elem1: %.df9 = impl_witness_access constants.%BitAndWith.impl_witness.b9f, element1 [concrete = constants.%C.as.BitAndWith.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.ref, %impl.elem1
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.as.BitAndWith.impl.Op.call: init %C to %.loc30_26.1 = call %bound_method(%a.ref, %b.ref)
// CHECK:STDOUT:   return %C.as.BitAndWith.impl.Op.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestAssign(%a.param: %ptr.00e, %b.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %ptr.00e = name_ref a, %a
// CHECK:STDOUT:   %.loc38: ref %C = deref %a.ref
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0: %.09a = impl_witness_access constants.%BitAndAssignWith.impl_witness.ea1, element0 [concrete = constants.%C.as.BitAndAssignWith.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc38, %impl.elem0
// CHECK:STDOUT:   %C.as.BitAndAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method(%.loc38, %b.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestNamedConstraintOp(%a.param: %D, %b.param: %D) -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %D = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %D = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1: %.315 = impl_witness_access constants.%BitAndWith.impl_witness.736, element1 [concrete = constants.%D.as.BitAndWith.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.ref, %impl.elem1
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %D.as.BitAndWith.impl.Op.call: init %D to %.loc53_41.1 = call %bound_method(%a.ref, %b.ref)
// CHECK:STDOUT:   return %D.as.BitAndWith.impl.Op.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestNamedConstraintAssign(%a.param: %ptr.4a5, %b.param: %D) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %ptr.4a5 = name_ref a, %a
// CHECK:STDOUT:   %.loc61: ref %D = deref %a.ref
// CHECK:STDOUT:   %b.ref: %D = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0: %.707 = impl_witness_access constants.%BitAndAssignWith.impl_witness.736, element0 [concrete = constants.%D.as.BitAndAssignWith.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc61, %impl.elem0
// CHECK:STDOUT:   %D.as.BitAndAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method(%.loc61, %b.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestGenericOp(%T.loc65_19.2: %BitAnd.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %BitAndWith.type.loc67_12.2: type = facet_type <@BitAndWith, @BitAndWith(%T.as_type.loc65_37.1)> [symbolic = %BitAndWith.type.loc67_12.2 (constants.%BitAndWith.type.bbeb52.2)]
// CHECK:STDOUT:   %require_complete.loc67: <witness> = require_complete_type %BitAndWith.type.loc67_12.2 [symbolic = %require_complete.loc67 (constants.%require_complete.8b5)]
// CHECK:STDOUT:   %BitAndWith.assoc_type: type = assoc_entity_type @BitAndWith, @BitAndWith(%T.as_type.loc65_37.1) [symbolic = %BitAndWith.assoc_type (constants.%BitAndWith.assoc_type.884223.2)]
// CHECK:STDOUT:   %assoc1: @TestGenericOp.%BitAndWith.assoc_type (%BitAndWith.assoc_type.884223.2) = assoc_entity element1, imports.%Core.import_ref.7c1 [symbolic = %assoc1 (constants.%assoc1.b5b429.2)]
// CHECK:STDOUT:   %BitAndWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc65_19.1, @BitAndWith, @BitAndWith(%T.as_type.loc65_37.1) [symbolic = %BitAndWith.lookup_impl_witness (constants.%BitAndWith.lookup_impl_witness.3a3af1.2)]
// CHECK:STDOUT:   %BitAndWith.facet: @TestGenericOp.%BitAndWith.type.loc67_12.2 (%BitAndWith.type.bbeb52.2) = facet_value %T.as_type.loc65_37.1, (%BitAndWith.lookup_impl_witness) [symbolic = %BitAndWith.facet (constants.%BitAndWith.facet.948)]
// CHECK:STDOUT:   %BitAndWith.WithSelf.Op.type: type = fn_type @BitAndWith.WithSelf.Op, @BitAndWith.WithSelf(%T.as_type.loc65_37.1, %BitAndWith.facet) [symbolic = %BitAndWith.WithSelf.Op.type (constants.%BitAndWith.WithSelf.Op.type.a48)]
// CHECK:STDOUT:   %.loc67_12.2: type = fn_type_with_self_type %BitAndWith.WithSelf.Op.type, %BitAndWith.facet [symbolic = %.loc67_12.2 (constants.%.daa)]
// CHECK:STDOUT:   %impl.elem1.loc67_12.2: @TestGenericOp.%.loc67_12.2 (%.daa) = impl_witness_access %BitAndWith.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc67_12.2 (constants.%impl.elem1.a77)]
// CHECK:STDOUT:   %specific_impl_fn.loc67_12.2: <specific function> = specific_impl_function %impl.elem1.loc67_12.2, @BitAndWith.WithSelf.Op(%T.as_type.loc65_37.1, %BitAndWith.facet) [symbolic = %specific_impl_fn.loc67_12.2 (constants.%specific_impl_fn.7aa)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @TestGenericOp.%T.as_type.loc65_37.1 (%T.as_type.190bb7.1), %b.param: @TestGenericOp.%T.as_type.loc65_37.1 (%T.as_type.190bb7.1)) -> out %return.param: @TestGenericOp.%T.as_type.loc65_37.1 (%T.as_type.190bb7.1) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %a.ref: @TestGenericOp.%T.as_type.loc65_37.1 (%T.as_type.190bb7.1) = name_ref a, %a
// CHECK:STDOUT:     %b.ref: @TestGenericOp.%T.as_type.loc65_37.1 (%T.as_type.190bb7.1) = name_ref b, %b
// CHECK:STDOUT:     %BitAndWith.type.loc67_12.1: type = facet_type <@BitAndWith, @BitAndWith(constants.%T.as_type.190bb7.1)> [symbolic = %BitAndWith.type.loc67_12.2 (constants.%BitAndWith.type.bbeb52.2)]
// CHECK:STDOUT:     %.loc67_12.1: @TestGenericOp.%BitAndWith.assoc_type (%BitAndWith.assoc_type.884223.2) = specific_constant imports.%Core.import_ref.e78, @BitAndWith.WithSelf(constants.%T.as_type.190bb7.1, constants.%Self.48b) [symbolic = %assoc1 (constants.%assoc1.b5b429.2)]
// CHECK:STDOUT:     %Op.ref: @TestGenericOp.%BitAndWith.assoc_type (%BitAndWith.assoc_type.884223.2) = name_ref Op, %.loc67_12.1 [symbolic = %assoc1 (constants.%assoc1.b5b429.2)]
// CHECK:STDOUT:     %impl.elem1.loc67_12.1: @TestGenericOp.%.loc67_12.2 (%.daa) = impl_witness_access constants.%BitAndWith.lookup_impl_witness.3a3af1.2, element1 [symbolic = %impl.elem1.loc67_12.2 (constants.%impl.elem1.a77)]
// CHECK:STDOUT:     %bound_method.loc67_12.1: <bound method> = bound_method %a.ref, %impl.elem1.loc67_12.1
// CHECK:STDOUT:     %specific_impl_fn.loc67_12.1: <specific function> = specific_impl_function %impl.elem1.loc67_12.1, @BitAndWith.WithSelf.Op(constants.%T.as_type.190bb7.1, constants.%BitAndWith.facet.948) [symbolic = %specific_impl_fn.loc67_12.2 (constants.%specific_impl_fn.7aa)]
// CHECK:STDOUT:     %bound_method.loc67_12.2: <bound method> = bound_method %a.ref, %specific_impl_fn.loc67_12.1
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %BitAndWith.WithSelf.Op.call: init @TestGenericOp.%T.as_type.loc65_37.1 (%T.as_type.190bb7.1) to %.loc65_49.1 = call %bound_method.loc67_12.2(%a.ref, %b.ref)
// CHECK:STDOUT:     return %BitAndWith.WithSelf.Op.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestGenericAssign(%T.loc71_23.2: %BitAndAssign.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %BitAndAssignWith.type.loc73_6.2: type = facet_type <@BitAndAssignWith, @BitAndAssignWith(%T.as_type.loc71_48.1)> [symbolic = %BitAndAssignWith.type.loc73_6.2 (constants.%BitAndAssignWith.type.9743d5.2)]
// CHECK:STDOUT:   %require_complete.loc73: <witness> = require_complete_type %BitAndAssignWith.type.loc73_6.2 [symbolic = %require_complete.loc73 (constants.%require_complete.247c45.2)]
// CHECK:STDOUT:   %BitAndAssignWith.assoc_type: type = assoc_entity_type @BitAndAssignWith, @BitAndAssignWith(%T.as_type.loc71_48.1) [symbolic = %BitAndAssignWith.assoc_type (constants.%BitAndAssignWith.assoc_type.fc666d.2)]
// CHECK:STDOUT:   %assoc0: @TestGenericAssign.%BitAndAssignWith.assoc_type (%BitAndAssignWith.assoc_type.fc666d.2) = assoc_entity element0, imports.%Core.import_ref.7e7 [symbolic = %assoc0 (constants.%assoc0.b4dd45.2)]
// CHECK:STDOUT:   %BitAndAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc71_23.1, @BitAndAssignWith, @BitAndAssignWith(%T.as_type.loc71_48.1) [symbolic = %BitAndAssignWith.lookup_impl_witness (constants.%BitAndAssignWith.lookup_impl_witness)]
// CHECK:STDOUT:   %BitAndAssignWith.facet: @TestGenericAssign.%BitAndAssignWith.type.loc73_6.2 (%BitAndAssignWith.type.9743d5.2) = facet_value %T.as_type.loc71_48.1, (%BitAndAssignWith.lookup_impl_witness) [symbolic = %BitAndAssignWith.facet (constants.%BitAndAssignWith.facet.420)]
// CHECK:STDOUT:   %BitAndAssignWith.WithSelf.Op.type: type = fn_type @BitAndAssignWith.WithSelf.Op, @BitAndAssignWith.WithSelf(%T.as_type.loc71_48.1, %BitAndAssignWith.facet) [symbolic = %BitAndAssignWith.WithSelf.Op.type (constants.%BitAndAssignWith.WithSelf.Op.type.e07)]
// CHECK:STDOUT:   %.loc73_6.2: type = fn_type_with_self_type %BitAndAssignWith.WithSelf.Op.type, %BitAndAssignWith.facet [symbolic = %.loc73_6.2 (constants.%.854)]
// CHECK:STDOUT:   %impl.elem0.loc73_6.2: @TestGenericAssign.%.loc73_6.2 (%.854) = impl_witness_access %BitAndAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc73_6.2 (constants.%impl.elem0.86c)]
// CHECK:STDOUT:   %specific_impl_fn.loc73_6.2: <specific function> = specific_impl_function %impl.elem0.loc73_6.2, @BitAndAssignWith.WithSelf.Op(%T.as_type.loc71_48.1, %BitAndAssignWith.facet) [symbolic = %specific_impl_fn.loc73_6.2 (constants.%specific_impl_fn.0ff)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @TestGenericAssign.%ptr.loc71_48.1 (%ptr.360), %b.param: @TestGenericAssign.%T.as_type.loc71_48.1 (%T.as_type.190bb7.2)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %a.ref: @TestGenericAssign.%ptr.loc71_48.1 (%ptr.360) = name_ref a, %a
// CHECK:STDOUT:     %.loc73_3: ref @TestGenericAssign.%T.as_type.loc71_48.1 (%T.as_type.190bb7.2) = deref %a.ref
// CHECK:STDOUT:     %b.ref: @TestGenericAssign.%T.as_type.loc71_48.1 (%T.as_type.190bb7.2) = name_ref b, %b
// CHECK:STDOUT:     %BitAndAssignWith.type.loc73_6.1: type = facet_type <@BitAndAssignWith, @BitAndAssignWith(constants.%T.as_type.190bb7.2)> [symbolic = %BitAndAssignWith.type.loc73_6.2 (constants.%BitAndAssignWith.type.9743d5.2)]
// CHECK:STDOUT:     %.loc73_6.1: @TestGenericAssign.%BitAndAssignWith.assoc_type (%BitAndAssignWith.assoc_type.fc666d.2) = specific_constant imports.%Core.import_ref.7b10, @BitAndAssignWith.WithSelf(constants.%T.as_type.190bb7.2, constants.%Self.591) [symbolic = %assoc0 (constants.%assoc0.b4dd45.2)]
// CHECK:STDOUT:     %Op.ref: @TestGenericAssign.%BitAndAssignWith.assoc_type (%BitAndAssignWith.assoc_type.fc666d.2) = name_ref Op, %.loc73_6.1 [symbolic = %assoc0 (constants.%assoc0.b4dd45.2)]
// CHECK:STDOUT:     %impl.elem0.loc73_6.1: @TestGenericAssign.%.loc73_6.2 (%.854) = impl_witness_access constants.%BitAndAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc73_6.2 (constants.%impl.elem0.86c)]
// CHECK:STDOUT:     %bound_method.loc73_6.1: <bound method> = bound_method %.loc73_3, %impl.elem0.loc73_6.1
// CHECK:STDOUT:     %specific_impl_fn.loc73_6.1: <specific function> = specific_impl_function %impl.elem0.loc73_6.1, @BitAndAssignWith.WithSelf.Op(constants.%T.as_type.190bb7.2, constants.%BitAndAssignWith.facet.420) [symbolic = %specific_impl_fn.loc73_6.2 (constants.%specific_impl_fn.0ff)]
// CHECK:STDOUT:     %bound_method.loc73_6.2: <bound method> = bound_method %.loc73_3, %specific_impl_fn.loc73_6.1
// CHECK:STDOUT:     %BitAndAssignWith.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc73_6.2(%.loc73_3, %b.ref)
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
// CHECK:STDOUT: specific @TestGenericOp(constants.%BitAnd.facet.50e) {
// CHECK:STDOUT:   %T.patt.loc65_19.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc65_19.1 => constants.%BitAnd.facet.50e
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
// CHECK:STDOUT:   %BitAndWith.type.loc67_12.2 => constants.%BitAndWith.type.01b
// CHECK:STDOUT:   %require_complete.loc67 => constants.%complete_type.8d4
// CHECK:STDOUT:   %BitAndWith.assoc_type => constants.%BitAndWith.assoc_type.409
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.ea0
// CHECK:STDOUT:   %BitAndWith.lookup_impl_witness => constants.%BitAndWith.impl_witness.b9f
// CHECK:STDOUT:   %BitAndWith.facet => constants.%BitAndWith.facet.423
// CHECK:STDOUT:   %BitAndWith.WithSelf.Op.type => constants.%BitAndWith.WithSelf.Op.type.c29
// CHECK:STDOUT:   %.loc67_12.2 => constants.%.df9
// CHECK:STDOUT:   %impl.elem1.loc67_12.2 => constants.%C.as.BitAndWith.impl.Op
// CHECK:STDOUT:   %specific_impl_fn.loc67_12.2 => constants.%C.as.BitAndWith.impl.Op
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGenericAssign(constants.%BitAndAssign.facet.4df) {
// CHECK:STDOUT:   %T.patt.loc71_23.2 => constants.%T.patt.3a0592.2
// CHECK:STDOUT:   %T.loc71_23.1 => constants.%BitAndAssign.facet.4df
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
// CHECK:STDOUT:   %BitAndAssignWith.type.loc73_6.2 => constants.%BitAndAssignWith.type.0a8
// CHECK:STDOUT:   %require_complete.loc73 => constants.%complete_type.b80
// CHECK:STDOUT:   %BitAndAssignWith.assoc_type => constants.%BitAndAssignWith.assoc_type.c8f
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.9a4
// CHECK:STDOUT:   %BitAndAssignWith.lookup_impl_witness => constants.%BitAndAssignWith.impl_witness.ea1
// CHECK:STDOUT:   %BitAndAssignWith.facet => constants.%BitAndAssignWith.facet.edd
// CHECK:STDOUT:   %BitAndAssignWith.WithSelf.Op.type => constants.%BitAndAssignWith.WithSelf.Op.type.fe6
// CHECK:STDOUT:   %.loc73_6.2 => constants.%.09a
// CHECK:STDOUT:   %impl.elem0.loc73_6.2 => constants.%C.as.BitAndAssignWith.impl.Op
// CHECK:STDOUT:   %specific_impl_fn.loc73_6.2 => constants.%C.as.BitAndAssignWith.impl.Op
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGenericOp(constants.%BitAnd.facet.6f9) {
// CHECK:STDOUT:   %T.patt.loc65_19.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc65_19.1 => constants.%BitAnd.facet.6f9
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
// CHECK:STDOUT:   %BitAndWith.type.loc67_12.2 => constants.%BitAndWith.type.6fa
// CHECK:STDOUT:   %require_complete.loc67 => constants.%complete_type.64d
// CHECK:STDOUT:   %BitAndWith.assoc_type => constants.%BitAndWith.assoc_type.aa1
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.1eb
// CHECK:STDOUT:   %BitAndWith.lookup_impl_witness => constants.%BitAndWith.impl_witness.736
// CHECK:STDOUT:   %BitAndWith.facet => constants.%BitAndWith.facet.744
// CHECK:STDOUT:   %BitAndWith.WithSelf.Op.type => constants.%BitAndWith.WithSelf.Op.type.20b
// CHECK:STDOUT:   %.loc67_12.2 => constants.%.315
// CHECK:STDOUT:   %impl.elem1.loc67_12.2 => constants.%D.as.BitAndWith.impl.Op
// CHECK:STDOUT:   %specific_impl_fn.loc67_12.2 => constants.%D.as.BitAndWith.impl.Op
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGenericAssign(constants.%BitAndAssign.facet.6f9) {
// CHECK:STDOUT:   %T.patt.loc71_23.2 => constants.%T.patt.3a0592.2
// CHECK:STDOUT:   %T.loc71_23.1 => constants.%BitAndAssign.facet.6f9
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
// CHECK:STDOUT:   %BitAndAssignWith.type.loc73_6.2 => constants.%BitAndAssignWith.type.59f
// CHECK:STDOUT:   %require_complete.loc73 => constants.%complete_type.d4a
// CHECK:STDOUT:   %BitAndAssignWith.assoc_type => constants.%BitAndAssignWith.assoc_type.dd9
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.003
// CHECK:STDOUT:   %BitAndAssignWith.lookup_impl_witness => constants.%BitAndAssignWith.impl_witness.736
// CHECK:STDOUT:   %BitAndAssignWith.facet => constants.%BitAndAssignWith.facet.c70
// CHECK:STDOUT:   %BitAndAssignWith.WithSelf.Op.type => constants.%BitAndAssignWith.WithSelf.Op.type.64c
// CHECK:STDOUT:   %.loc73_6.2 => constants.%.707
// CHECK:STDOUT:   %impl.elem0.loc73_6.2 => constants.%D.as.BitAndAssignWith.impl.Op
// CHECK:STDOUT:   %specific_impl_fn.loc73_6.2 => constants.%D.as.BitAndAssignWith.impl.Op
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGenericOp(constants.%BitAnd.facet.8cd) {
// CHECK:STDOUT:   %T.patt.loc65_19.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc65_19.1 => constants.%BitAnd.facet.8cd
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
// CHECK:STDOUT:   %BitAndWith.type.loc67_12.2 => constants.%BitAndWith.type.2ea
// CHECK:STDOUT:   %require_complete.loc67 => constants.%complete_type.b42
// CHECK:STDOUT:   %BitAndWith.assoc_type => constants.%BitAndWith.assoc_type.4cc
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.342
// CHECK:STDOUT:   %BitAndWith.lookup_impl_witness => constants.%BitAndWith.impl_witness.128
// CHECK:STDOUT:   %BitAndWith.facet => constants.%BitAndWith.facet.b1d
// CHECK:STDOUT:   %BitAndWith.WithSelf.Op.type => constants.%BitAndWith.WithSelf.Op.type.03c
// CHECK:STDOUT:   %.loc67_12.2 => constants.%.a35
// CHECK:STDOUT:   %impl.elem1.loc67_12.2 => constants.%Int.as.BitAndWith.impl.Op.272
// CHECK:STDOUT:   %specific_impl_fn.loc67_12.2 => constants.%Int.as.BitAndWith.impl.Op.specific_fn.fbb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGenericAssign(constants.%BitAndAssign.facet.d8d) {
// CHECK:STDOUT:   %T.patt.loc71_23.2 => constants.%T.patt.3a0592.2
// CHECK:STDOUT:   %T.loc71_23.1 => constants.%BitAndAssign.facet.d8d
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
// CHECK:STDOUT:   %BitAndAssignWith.type.loc73_6.2 => constants.%BitAndAssignWith.type.f47
// CHECK:STDOUT:   %require_complete.loc73 => constants.%complete_type.cee
// CHECK:STDOUT:   %BitAndAssignWith.assoc_type => constants.%BitAndAssignWith.assoc_type.875
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.f312
// CHECK:STDOUT:   %BitAndAssignWith.lookup_impl_witness => constants.%BitAndAssignWith.impl_witness.f23
// CHECK:STDOUT:   %BitAndAssignWith.facet => constants.%BitAndAssignWith.facet.cb4
// CHECK:STDOUT:   %BitAndAssignWith.WithSelf.Op.type => constants.%BitAndAssignWith.WithSelf.Op.type.74a
// CHECK:STDOUT:   %.loc73_6.2 => constants.%.2db
// CHECK:STDOUT:   %impl.elem0.loc73_6.2 => constants.%Int.as.BitAndAssignWith.impl.Op.43022d.2
// CHECK:STDOUT:   %specific_impl_fn.loc73_6.2 => constants.%Int.as.BitAndAssignWith.impl.Op.specific_fn.6ac8d6.1
// CHECK:STDOUT: }
// CHECK:STDOUT:

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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/overloaded/left_shift.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/overloaded/left_shift.carbon

// This file was generated from binary_op.carbon.tmpl. Run make_tests.sh to regenerate.

package User;

class C {};

impl C as Core.LeftShiftWith(C) where .Result = C {
  fn Op(unused self: C, unused other: C) -> C {
    return {};
  }
}
impl C as Core.LeftShiftAssignWith(C) {
  fn Op(unused ref self: C, unused other: C) {}
}

fn TestOp(a: C, b: C) -> C {
  //@dump-sem-ir-begin
  return a << b;
  //@dump-sem-ir-end
}

fn TestAssign(a: C*, b: C) {
  //@dump-sem-ir-begin
  *a <<= b;
  //@dump-sem-ir-end
}

class D {};

impl D as Core.LeftShift {
  fn Op(unused self: D, unused other: D) -> D {
    return {};
  }
}
impl D as Core.LeftShiftAssign {
  fn Op(unused ref self: D, unused other: D) {}
}

fn TestNamedConstraintOp(a: D, b: D) -> D {
  //@dump-sem-ir-begin
  return a << b;
  //@dump-sem-ir-end
}

fn TestNamedConstraintAssign(a: D*, b: D) {
  //@dump-sem-ir-begin
  *a <<= b;
  //@dump-sem-ir-end
}

fn TestGenericOp[T: Core.LeftShift](a: T, b: T) -> T {
  //@dump-sem-ir-begin
  return a << b;
  //@dump-sem-ir-end
}

fn TestGenericAssign[T: Core.LeftShiftAssign](a: T*, b: T) {
  //@dump-sem-ir-begin
  *a <<= b;
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

// CHECK:STDOUT: --- left_shift.carbon = $left_shift
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Other: type = symbolic_binding Other, 0 [symbolic]
// CHECK:STDOUT:   %LeftShiftWith.type.f8b: type = facet_type <@LeftShiftWith, @LeftShiftWith(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.a4f: %LeftShiftWith.type.f8b = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %LeftShiftWith.assoc_type.1c1: type = assoc_entity_type @LeftShiftWith, @LeftShiftWith(%Other) [symbolic]
// CHECK:STDOUT:   %LeftShiftWith.WithSelf.Op.type.1ec: type = fn_type @LeftShiftWith.WithSelf.Op, @LeftShiftWith.WithSelf(%Other, %Self.a4f) [symbolic]
// CHECK:STDOUT:   %LeftShiftWith.WithSelf.Op.173: %LeftShiftWith.WithSelf.Op.type.1ec = struct_value () [symbolic]
// CHECK:STDOUT:   %LeftShiftWith.type.daa: type = facet_type <@LeftShiftWith, @LeftShiftWith(%C)> [concrete]
// CHECK:STDOUT:   %LeftShiftWith.assoc_type.deb: type = assoc_entity_type @LeftShiftWith, @LeftShiftWith(%C) [concrete]
// CHECK:STDOUT:   %assoc1.b77: %LeftShiftWith.assoc_type.deb = assoc_entity element1, imports.%Core.import_ref.c79 [concrete]
// CHECK:STDOUT:   %LeftShiftWith.impl_witness.541: <witness> = impl_witness @C.as.LeftShiftWith.impl.%LeftShiftWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.4a7: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %.79c: Core.Form = init_form %C [concrete]
// CHECK:STDOUT:   %return.param_patt.415: %pattern_type.4a7 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.340: %pattern_type.4a7 = return_slot_pattern %return.param_patt.415, %C [concrete]
// CHECK:STDOUT:   %C.as.LeftShiftWith.impl.Op.type: type = fn_type @C.as.LeftShiftWith.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.LeftShiftWith.impl.Op: %C.as.LeftShiftWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %LeftShiftWith.facet.d87: %LeftShiftWith.type.daa = facet_value %C, (%LeftShiftWith.impl_witness.541) [concrete]
// CHECK:STDOUT:   %LeftShiftWith.WithSelf.Op.type.aba: type = fn_type @LeftShiftWith.WithSelf.Op, @LeftShiftWith.WithSelf(%C, %LeftShiftWith.facet.d87) [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.type.414: type = facet_type <@LeftShiftAssignWith, @LeftShiftAssignWith(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.72e: %LeftShiftAssignWith.type.414 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %LeftShiftAssignWith.assoc_type.dd3: type = assoc_entity_type @LeftShiftAssignWith, @LeftShiftAssignWith(%Other) [symbolic]
// CHECK:STDOUT:   %LeftShiftAssignWith.WithSelf.Op.type.f9a: type = fn_type @LeftShiftAssignWith.WithSelf.Op, @LeftShiftAssignWith.WithSelf(%Other, %Self.72e) [symbolic]
// CHECK:STDOUT:   %LeftShiftAssignWith.WithSelf.Op.07a: %LeftShiftAssignWith.WithSelf.Op.type.f9a = struct_value () [symbolic]
// CHECK:STDOUT:   %LeftShiftAssignWith.type.7c8: type = facet_type <@LeftShiftAssignWith, @LeftShiftAssignWith(%C)> [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.impl_witness.607: <witness> = impl_witness @C.as.LeftShiftAssignWith.impl.%LeftShiftAssignWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.assoc_type.5fb: type = assoc_entity_type @LeftShiftAssignWith, @LeftShiftAssignWith(%C) [concrete]
// CHECK:STDOUT:   %assoc0.bae: %LeftShiftAssignWith.assoc_type.5fb = assoc_entity element0, imports.%Core.import_ref.321 [concrete]
// CHECK:STDOUT:   %C.as.LeftShiftAssignWith.impl.Op.type: type = fn_type @C.as.LeftShiftAssignWith.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.LeftShiftAssignWith.impl.Op: %C.as.LeftShiftAssignWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.facet.c65: %LeftShiftAssignWith.type.7c8 = facet_value %C, (%LeftShiftAssignWith.impl_witness.607) [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.WithSelf.Op.type.3a4: type = fn_type @LeftShiftAssignWith.WithSelf.Op, @LeftShiftAssignWith.WithSelf(%C, %LeftShiftAssignWith.facet.c65) [concrete]
// CHECK:STDOUT:   %a.param_patt.2a5: %pattern_type.4a7 = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt.a30: %pattern_type.4a7 = wrapper_binding_pattern a, %a.param_patt.2a5 [concrete]
// CHECK:STDOUT:   %b.param_patt.45f: %pattern_type.4a7 = value_param_pattern [concrete]
// CHECK:STDOUT:   %b.patt.562: %pattern_type.4a7 = wrapper_binding_pattern b, %b.param_patt.45f [concrete]
// CHECK:STDOUT:   %assoc1.240: %LeftShiftWith.assoc_type.1c1 = assoc_entity element1, imports.%Core.import_ref.930 [symbolic]
// CHECK:STDOUT:   %.c42: type = fn_type_with_self_type %LeftShiftWith.WithSelf.Op.type.aba, %LeftShiftWith.facet.d87 [concrete]
// CHECK:STDOUT:   %ptr.00e: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.842: type = pattern_type %ptr.00e [concrete]
// CHECK:STDOUT:   %a.param_patt.4fa: %pattern_type.842 = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt.eba: %pattern_type.842 = wrapper_binding_pattern a, %a.param_patt.4fa [concrete]
// CHECK:STDOUT:   %assoc0.504: %LeftShiftAssignWith.assoc_type.dd3 = assoc_entity element0, imports.%Core.import_ref.151 [symbolic]
// CHECK:STDOUT:   %.6c6: type = fn_type_with_self_type %LeftShiftAssignWith.WithSelf.Op.type.3a4, %LeftShiftAssignWith.facet.c65 [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %LeftShift.type: type = facet_type <@LeftShift> [concrete]
// CHECK:STDOUT:   %LeftShiftWith.impl_witness.736: <witness> = impl_witness @D.as.LeftShiftWith.impl.%LeftShiftWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.331: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %.84d: Core.Form = init_form %D [concrete]
// CHECK:STDOUT:   %return.param_patt.835: %pattern_type.331 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.5ef: %pattern_type.331 = return_slot_pattern %return.param_patt.835, %D [concrete]
// CHECK:STDOUT:   %D.as.LeftShiftWith.impl.Op.type: type = fn_type @D.as.LeftShiftWith.impl.Op [concrete]
// CHECK:STDOUT:   %D.as.LeftShiftWith.impl.Op: %D.as.LeftShiftWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %LeftShiftWith.type.aec: type = facet_type <@LeftShiftWith, @LeftShiftWith(%D)> [concrete]
// CHECK:STDOUT:   %LeftShiftWith.facet.e2e: %LeftShiftWith.type.aec = facet_value %D, (%LeftShiftWith.impl_witness.736) [concrete]
// CHECK:STDOUT:   %LeftShiftWith.assoc_type.c17: type = assoc_entity_type @LeftShiftWith, @LeftShiftWith(%D) [concrete]
// CHECK:STDOUT:   %LeftShiftWith.WithSelf.Op.type.b14: type = fn_type @LeftShiftWith.WithSelf.Op, @LeftShiftWith.WithSelf(%D, %LeftShiftWith.facet.e2e) [concrete]
// CHECK:STDOUT:   %assoc1.2c6: %LeftShiftWith.assoc_type.c17 = assoc_entity element1, imports.%Core.import_ref.c79 [concrete]
// CHECK:STDOUT:   %LeftShiftAssign.type: type = facet_type <@LeftShiftAssign> [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.impl_witness.736: <witness> = impl_witness @D.as.LeftShiftAssignWith.impl.%LeftShiftAssignWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %D.as.LeftShiftAssignWith.impl.Op.type: type = fn_type @D.as.LeftShiftAssignWith.impl.Op [concrete]
// CHECK:STDOUT:   %D.as.LeftShiftAssignWith.impl.Op: %D.as.LeftShiftAssignWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.type.27e: type = facet_type <@LeftShiftAssignWith, @LeftShiftAssignWith(%D)> [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.facet.a72: %LeftShiftAssignWith.type.27e = facet_value %D, (%LeftShiftAssignWith.impl_witness.736) [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.WithSelf.Op.type.73a: type = fn_type @LeftShiftAssignWith.WithSelf.Op, @LeftShiftAssignWith.WithSelf(%D, %LeftShiftAssignWith.facet.a72) [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.assoc_type.2fa: type = assoc_entity_type @LeftShiftAssignWith, @LeftShiftAssignWith(%D) [concrete]
// CHECK:STDOUT:   %assoc0.742: %LeftShiftAssignWith.assoc_type.2fa = assoc_entity element0, imports.%Core.import_ref.321 [concrete]
// CHECK:STDOUT:   %a.param_patt.d0e: %pattern_type.331 = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt.ec5: %pattern_type.331 = wrapper_binding_pattern a, %a.param_patt.d0e [concrete]
// CHECK:STDOUT:   %b.param_patt.110: %pattern_type.331 = value_param_pattern [concrete]
// CHECK:STDOUT:   %b.patt.14c: %pattern_type.331 = wrapper_binding_pattern b, %b.param_patt.110 [concrete]
// CHECK:STDOUT:   %.bd1: type = fn_type_with_self_type %LeftShiftWith.WithSelf.Op.type.b14, %LeftShiftWith.facet.e2e [concrete]
// CHECK:STDOUT:   %ptr.4a5: type = ptr_type %D [concrete]
// CHECK:STDOUT:   %pattern_type.ceb: type = pattern_type %ptr.4a5 [concrete]
// CHECK:STDOUT:   %a.param_patt.609: %pattern_type.ceb = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt.fac: %pattern_type.ceb = wrapper_binding_pattern a, %a.param_patt.609 [concrete]
// CHECK:STDOUT:   %.c0f: type = fn_type_with_self_type %LeftShiftAssignWith.WithSelf.Op.type.73a, %LeftShiftAssignWith.facet.a72 [concrete]
// CHECK:STDOUT:   %pattern_type.9a587c.2: type = pattern_type %LeftShift.type [concrete]
// CHECK:STDOUT:   %T.patt.3a0592.1: %pattern_type.9a587c.2 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.f45530.1: %LeftShift.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.190bb7.1: type = facet_access_type %T.f45530.1 [symbolic]
// CHECK:STDOUT:   %pattern_type.86a9db.1: type = pattern_type %T.as_type.190bb7.1 [symbolic]
// CHECK:STDOUT:   %a.param_patt.e33: %pattern_type.86a9db.1 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %a.patt.e02: %pattern_type.86a9db.1 = wrapper_binding_pattern a, %a.param_patt.e33 [symbolic]
// CHECK:STDOUT:   %b.param_patt.4cd741.1: %pattern_type.86a9db.1 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %b.patt.60acf6.1: %pattern_type.86a9db.1 = wrapper_binding_pattern b, %b.param_patt.4cd741.1 [symbolic]
// CHECK:STDOUT:   %.1f0: Core.Form = init_form %T.as_type.190bb7.1 [symbolic]
// CHECK:STDOUT:   %return.param_patt.5f6: %pattern_type.86a9db.1 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.96a: %pattern_type.86a9db.1 = return_slot_pattern %return.param_patt.5f6, %T.as_type.190bb7.1 [symbolic]
// CHECK:STDOUT:   %LeftShiftWith.type.3e428a.2: type = facet_type <@LeftShiftWith, @LeftShiftWith(%T.as_type.190bb7.1)> [symbolic]
// CHECK:STDOUT:   %LeftShiftWith.assoc_type.7e6d0d.2: type = assoc_entity_type @LeftShiftWith, @LeftShiftWith(%T.as_type.190bb7.1) [symbolic]
// CHECK:STDOUT:   %assoc1.8fd5dc.2: %LeftShiftWith.assoc_type.7e6d0d.2 = assoc_entity element1, imports.%Core.import_ref.c79 [symbolic]
// CHECK:STDOUT:   %require_complete.1dd: <witness> = require_complete_type %LeftShiftWith.type.3e428a.2 [symbolic]
// CHECK:STDOUT:   %LeftShiftWith.lookup_impl_witness.27975c.2: <witness> = lookup_impl_witness %T.f45530.1, @LeftShiftWith, @LeftShiftWith(%T.as_type.190bb7.1) [symbolic]
// CHECK:STDOUT:   %LeftShiftWith.facet.7d2: %LeftShiftWith.type.3e428a.2 = facet_value %T.as_type.190bb7.1, (%LeftShiftWith.lookup_impl_witness.27975c.2) [symbolic]
// CHECK:STDOUT:   %LeftShiftWith.WithSelf.Op.type.7ee: type = fn_type @LeftShiftWith.WithSelf.Op, @LeftShiftWith.WithSelf(%T.as_type.190bb7.1, %LeftShiftWith.facet.7d2) [symbolic]
// CHECK:STDOUT:   %.08c: type = fn_type_with_self_type %LeftShiftWith.WithSelf.Op.type.7ee, %LeftShiftWith.facet.7d2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.542: %.08c = impl_witness_access %LeftShiftWith.lookup_impl_witness.27975c.2, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.348: <specific function> = specific_impl_function %impl.elem1.542, @LeftShiftWith.WithSelf.Op(%T.as_type.190bb7.1, %LeftShiftWith.facet.7d2) [symbolic]
// CHECK:STDOUT:   %pattern_type.9a587c.3: type = pattern_type %LeftShiftAssign.type [concrete]
// CHECK:STDOUT:   %T.patt.3a0592.2: %pattern_type.9a587c.3 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.f45530.2: %LeftShiftAssign.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.190bb7.2: type = facet_access_type %T.f45530.2 [symbolic]
// CHECK:STDOUT:   %ptr.360: type = ptr_type %T.as_type.190bb7.2 [symbolic]
// CHECK:STDOUT:   %pattern_type.ff5: type = pattern_type %ptr.360 [symbolic]
// CHECK:STDOUT:   %a.param_patt.f50: %pattern_type.ff5 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %a.patt.193: %pattern_type.ff5 = wrapper_binding_pattern a, %a.param_patt.f50 [symbolic]
// CHECK:STDOUT:   %pattern_type.86a9db.2: type = pattern_type %T.as_type.190bb7.2 [symbolic]
// CHECK:STDOUT:   %b.param_patt.4cd741.2: %pattern_type.86a9db.2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %b.patt.60acf6.2: %pattern_type.86a9db.2 = wrapper_binding_pattern b, %b.param_patt.4cd741.2 [symbolic]
// CHECK:STDOUT:   %LeftShiftAssignWith.type.93876f.2: type = facet_type <@LeftShiftAssignWith, @LeftShiftAssignWith(%T.as_type.190bb7.2)> [symbolic]
// CHECK:STDOUT:   %LeftShiftAssignWith.assoc_type.5ac4b3.2: type = assoc_entity_type @LeftShiftAssignWith, @LeftShiftAssignWith(%T.as_type.190bb7.2) [symbolic]
// CHECK:STDOUT:   %assoc0.64dfba.2: %LeftShiftAssignWith.assoc_type.5ac4b3.2 = assoc_entity element0, imports.%Core.import_ref.321 [symbolic]
// CHECK:STDOUT:   %require_complete.851c02.2: <witness> = require_complete_type %LeftShiftAssignWith.type.93876f.2 [symbolic]
// CHECK:STDOUT:   %LeftShiftAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.f45530.2, @LeftShiftAssignWith, @LeftShiftAssignWith(%T.as_type.190bb7.2) [symbolic]
// CHECK:STDOUT:   %LeftShiftAssignWith.facet.509: %LeftShiftAssignWith.type.93876f.2 = facet_value %T.as_type.190bb7.2, (%LeftShiftAssignWith.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %LeftShiftAssignWith.WithSelf.Op.type.574: type = fn_type @LeftShiftAssignWith.WithSelf.Op, @LeftShiftAssignWith.WithSelf(%T.as_type.190bb7.2, %LeftShiftAssignWith.facet.509) [symbolic]
// CHECK:STDOUT:   %.508: type = fn_type_with_self_type %LeftShiftAssignWith.WithSelf.Op.type.574, %LeftShiftAssignWith.facet.509 [symbolic]
// CHECK:STDOUT:   %impl.elem0.ce7: %.508 = impl_witness_access %LeftShiftAssignWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.9b0: <specific function> = specific_impl_function %impl.elem0.ce7, @LeftShiftAssignWith.WithSelf.Op(%T.as_type.190bb7.2, %LeftShiftAssignWith.facet.509) [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %Int.d4bbe0.1: type = class_type @Int, @Int(%N) [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %ptr.a17: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %ptr.a17 [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %LeftShift.facet.aff: %LeftShift.type = facet_value %C, (%LeftShiftWith.impl_witness.541) [concrete]
// CHECK:STDOUT:   %LeftShiftAssign.facet.4c3: %LeftShiftAssign.type = facet_value %C, (%LeftShiftAssignWith.impl_witness.607) [concrete]
// CHECK:STDOUT:   %LeftShift.facet.6f9: %LeftShift.type = facet_value %D, (%LeftShiftWith.impl_witness.736) [concrete]
// CHECK:STDOUT:   %LeftShiftAssign.facet.6f9: %LeftShiftAssign.type = facet_value %D, (%LeftShiftAssignWith.impl_witness.736) [concrete]
// CHECK:STDOUT:   %Int.as.LeftShiftWith.impl.Op.type.a8a: type = fn_type @Int.as.LeftShiftWith.impl.Op.1, @Int.as.LeftShiftWith.impl.af8(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.LeftShiftWith.impl.Op.b16: %Int.as.LeftShiftWith.impl.Op.type.a8a = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.81c7d0.1: type = facet_type <@ImplicitAs, @ImplicitAs(%Int.d4bbe0.1)> [symbolic]
// CHECK:STDOUT:   %U.f16: %ImplicitAs.type.81c7d0.1 = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %LeftShiftWith.type.795: type = facet_type <@LeftShiftWith, @LeftShiftWith(%i32)> [concrete]
// CHECK:STDOUT:   %LeftShiftWith.impl_witness.c59: <witness> = impl_witness imports.%LeftShiftWith.impl_witness_table.f13, @Int.as.LeftShiftWith.impl.af8(%int_32) [concrete]
// CHECK:STDOUT:   %LeftShiftWith.assoc_type.3a2: type = assoc_entity_type @LeftShiftWith, @LeftShiftWith(%i32) [concrete]
// CHECK:STDOUT:   %assoc1.1d4: %LeftShiftWith.assoc_type.3a2 = assoc_entity element1, imports.%Core.import_ref.c79 [concrete]
// CHECK:STDOUT:   %complete_type.1f8: <witness> = complete_type_witness %LeftShiftWith.type.795 [concrete]
// CHECK:STDOUT:   %Int.as.LeftShiftWith.impl.Op.type.3e4: type = fn_type @Int.as.LeftShiftWith.impl.Op.1, @Int.as.LeftShiftWith.impl.af8(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.LeftShiftWith.impl.Op.750: %Int.as.LeftShiftWith.impl.Op.type.3e4 = struct_value () [concrete]
// CHECK:STDOUT:   %LeftShift.facet.f35: %LeftShift.type = facet_value %i32, (%LeftShiftWith.impl_witness.c59) [concrete]
// CHECK:STDOUT:   %a.param_patt.076: %pattern_type.d94 = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt.96f: %pattern_type.d94 = wrapper_binding_pattern a, %a.param_patt.076 [concrete]
// CHECK:STDOUT:   %b.param_patt.1a2: %pattern_type.d94 = value_param_pattern [concrete]
// CHECK:STDOUT:   %b.patt.3ca: %pattern_type.d94 = wrapper_binding_pattern b, %b.param_patt.1a2 [concrete]
// CHECK:STDOUT:   %.0b3: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.299: %pattern_type.d94 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.6fa: %pattern_type.d94 = return_slot_pattern %return.param_patt.299, %i32 [concrete]
// CHECK:STDOUT:   %Int.as.LeftShiftAssignWith.impl.Op.type.c2c614.1: type = fn_type @Int.as.LeftShiftAssignWith.impl.Op.1, @Int.as.LeftShiftAssignWith.impl(%N, %U.f16) [symbolic]
// CHECK:STDOUT:   %Int.as.LeftShiftAssignWith.impl.Op.17c79c.1: %Int.as.LeftShiftAssignWith.impl.Op.type.c2c614.1 = struct_value () [symbolic]
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
// CHECK:STDOUT:   %LeftShiftAssignWith.type.e04: type = facet_type <@LeftShiftAssignWith, @LeftShiftAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.impl_witness.b4d: <witness> = impl_witness imports.%LeftShiftAssignWith.impl_witness_table, @Int.as.LeftShiftAssignWith.impl(%int_32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %LeftShiftAssign.facet.deb: %LeftShiftAssign.type = facet_value %i32, (%LeftShiftAssignWith.impl_witness.b4d) [concrete]
// CHECK:STDOUT:   %a.param_patt.977: %pattern_type.7dc = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt.b70: %pattern_type.7dc = wrapper_binding_pattern a, %a.param_patt.977 [concrete]
// CHECK:STDOUT:   %complete_type.d86: <witness> = complete_type_witness %LeftShiftWith.type.daa [concrete]
// CHECK:STDOUT:   %complete_type.a56: <witness> = complete_type_witness %ptr.00e [concrete]
// CHECK:STDOUT:   %complete_type.5cc: <witness> = complete_type_witness %LeftShiftAssignWith.type.7c8 [concrete]
// CHECK:STDOUT:   %complete_type.f58: <witness> = complete_type_witness %LeftShiftWith.type.aec [concrete]
// CHECK:STDOUT:   %complete_type.320: <witness> = complete_type_witness %ptr.4a5 [concrete]
// CHECK:STDOUT:   %complete_type.125: <witness> = complete_type_witness %LeftShiftAssignWith.type.27e [concrete]
// CHECK:STDOUT:   %LeftShiftWith.facet.bdb: %LeftShiftWith.type.795 = facet_value %i32, (%LeftShiftWith.impl_witness.c59) [concrete]
// CHECK:STDOUT:   %LeftShiftWith.WithSelf.Op.type.8ba: type = fn_type @LeftShiftWith.WithSelf.Op, @LeftShiftWith.WithSelf(%i32, %LeftShiftWith.facet.bdb) [concrete]
// CHECK:STDOUT:   %.a55: type = fn_type_with_self_type %LeftShiftWith.WithSelf.Op.type.8ba, %LeftShiftWith.facet.bdb [concrete]
// CHECK:STDOUT:   %Int.as.LeftShiftWith.impl.Op.specific_fn.040: <specific function> = specific_function %Int.as.LeftShiftWith.impl.Op.750, @Int.as.LeftShiftWith.impl.Op.1(%int_32) [concrete]
// CHECK:STDOUT:   %complete_type.1b1: <witness> = complete_type_witness %ptr.a17 [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.assoc_type.7f9: type = assoc_entity_type @LeftShiftAssignWith, @LeftShiftAssignWith(%i32) [concrete]
// CHECK:STDOUT:   %assoc0.674: %LeftShiftAssignWith.assoc_type.7f9 = assoc_entity element0, imports.%Core.import_ref.321 [concrete]
// CHECK:STDOUT:   %complete_type.8aa: <witness> = complete_type_witness %LeftShiftAssignWith.type.e04 [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.facet.28f: %LeftShiftAssignWith.type.e04 = facet_value %i32, (%LeftShiftAssignWith.impl_witness.b4d) [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.WithSelf.Op.type.353: type = fn_type @LeftShiftAssignWith.WithSelf.Op, @LeftShiftAssignWith.WithSelf(%i32, %LeftShiftAssignWith.facet.28f) [concrete]
// CHECK:STDOUT:   %.1df: type = fn_type_with_self_type %LeftShiftAssignWith.WithSelf.Op.type.353, %LeftShiftAssignWith.facet.28f [concrete]
// CHECK:STDOUT:   %Int.as.LeftShiftAssignWith.impl.Op.type.6c0a70.2: type = fn_type @Int.as.LeftShiftAssignWith.impl.Op.1, @Int.as.LeftShiftAssignWith.impl(%int_32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Int.as.LeftShiftAssignWith.impl.Op.00cc00.2: %Int.as.LeftShiftAssignWith.impl.Op.type.6c0a70.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.LeftShiftAssignWith.impl.Op.specific_fn.3e135e.1: <specific function> = specific_function %Int.as.LeftShiftAssignWith.impl.Op.00cc00.2, @Int.as.LeftShiftAssignWith.impl.Op.1(%int_32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.0b1: @LeftShiftWith.WithSelf.%LeftShiftWith.assoc_type (%LeftShiftWith.assoc_type.1c1) = import_ref Core//prelude/operators/bitwise, loc{{\d+_\d+}}, loaded [symbolic = @LeftShiftWith.WithSelf.%assoc1 (constants.%assoc1.240)]
// CHECK:STDOUT:   %Core.import_ref.c79: @LeftShiftWith.WithSelf.%LeftShiftWith.WithSelf.Op.type (%LeftShiftWith.WithSelf.Op.type.1ec) = import_ref Core//prelude/operators/bitwise, loc{{\d+_\d+}}, loaded [symbolic = @LeftShiftWith.WithSelf.%LeftShiftWith.WithSelf.Op (constants.%LeftShiftWith.WithSelf.Op.173)]
// CHECK:STDOUT:   %Core.import_ref.54c: @LeftShiftAssignWith.WithSelf.%LeftShiftAssignWith.assoc_type (%LeftShiftAssignWith.assoc_type.dd3) = import_ref Core//prelude/operators/bitwise, loc{{\d+_\d+}}, loaded [symbolic = @LeftShiftAssignWith.WithSelf.%assoc0 (constants.%assoc0.504)]
// CHECK:STDOUT:   %Core.import_ref.321: @LeftShiftAssignWith.WithSelf.%LeftShiftAssignWith.WithSelf.Op.type (%LeftShiftAssignWith.WithSelf.Op.type.f9a) = import_ref Core//prelude/operators/bitwise, loc{{\d+_\d+}}, loaded [symbolic = @LeftShiftAssignWith.WithSelf.%LeftShiftAssignWith.WithSelf.Op (constants.%LeftShiftAssignWith.WithSelf.Op.07a)]
// CHECK:STDOUT:   %Core.import_ref.930 = import_ref Core//prelude/operators/bitwise, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.151 = import_ref Core//prelude/operators/bitwise, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.ebd: type = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.LeftShiftWith.impl.af8.%Int (constants.%Int.d4bbe0.1)]
// CHECK:STDOUT:   %Core.import_ref.b26: @Int.as.LeftShiftWith.impl.af8.%Int.as.LeftShiftWith.impl.Op.type (%Int.as.LeftShiftWith.impl.Op.type.a8a) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.LeftShiftWith.impl.af8.%Int.as.LeftShiftWith.impl.Op (constants.%Int.as.LeftShiftWith.impl.Op.b16)]
// CHECK:STDOUT:   %LeftShiftWith.impl_witness_table.f13 = impl_witness_table (%Core.import_ref.ebd, %Core.import_ref.b26), @Int.as.LeftShiftWith.impl.af8 [concrete]
// CHECK:STDOUT:   %Core.import_ref.cf1: @Int.as.LeftShiftAssignWith.impl.%Int.as.LeftShiftAssignWith.impl.Op.type.2 (%Int.as.LeftShiftAssignWith.impl.Op.type.c2c614.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.LeftShiftAssignWith.impl.%Int.as.LeftShiftAssignWith.impl.Op.2 (constants.%Int.as.LeftShiftAssignWith.impl.Op.17c79c.1)]
// CHECK:STDOUT:   %LeftShiftAssignWith.impl_witness_table = impl_witness_table (%Core.import_ref.cf1), @Int.as.LeftShiftAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.39b: @From.as_type.as.ImplicitAs.impl.%From.as_type.as.ImplicitAs.impl.Convert.type (%From.as_type.as.ImplicitAs.impl.Convert.type.386) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @From.as_type.as.ImplicitAs.impl.%From.as_type.as.ImplicitAs.impl.Convert (constants.%From.as_type.as.ImplicitAs.impl.Convert.85a)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.8f1 = impl_witness_table (%Core.import_ref.39b), @From.as_type.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.47b: Core.IntLiteral = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.AnyInt.impl.%N (constants.%N)]
// CHECK:STDOUT:   %Core.import_ref.323f: @Int.as.AnyInt.impl.%Int.as.AnyInt.impl.AsInt.type (%Int.as.AnyInt.impl.AsInt.type.0ee) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.AnyInt.impl.%Int.as.AnyInt.impl.AsInt (constants.%Int.as.AnyInt.impl.AsInt.497)]
// CHECK:STDOUT:   %AnyInt.impl_witness_table = impl_witness_table (%Core.import_ref.47b, %Core.import_ref.323f), @Int.as.AnyInt.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestOp(%a.param: %C, %b.param: %C) -> out %return.param: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1: %.c42 = impl_witness_access constants.%LeftShiftWith.impl_witness.541, element1 [concrete = constants.%C.as.LeftShiftWith.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.ref, %impl.elem1
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.as.LeftShiftWith.impl.Op.call: init %C to %.loc30_26.1 = call %bound_method(%a.ref, %b.ref)
// CHECK:STDOUT:   return %C.as.LeftShiftWith.impl.Op.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestAssign(%a.param: %ptr.00e, %b.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %ptr.00e = name_ref a, %a
// CHECK:STDOUT:   %.loc38: ref %C = deref %a.ref
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0: %.6c6 = impl_witness_access constants.%LeftShiftAssignWith.impl_witness.607, element0 [concrete = constants.%C.as.LeftShiftAssignWith.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc38, %impl.elem0
// CHECK:STDOUT:   %C.as.LeftShiftAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method(%.loc38, %b.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestNamedConstraintOp(%a.param: %D, %b.param: %D) -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %D = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %D = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1: %.bd1 = impl_witness_access constants.%LeftShiftWith.impl_witness.736, element1 [concrete = constants.%D.as.LeftShiftWith.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.ref, %impl.elem1
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %D.as.LeftShiftWith.impl.Op.call: init %D to %.loc53_41.1 = call %bound_method(%a.ref, %b.ref)
// CHECK:STDOUT:   return %D.as.LeftShiftWith.impl.Op.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestNamedConstraintAssign(%a.param: %ptr.4a5, %b.param: %D) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %ptr.4a5 = name_ref a, %a
// CHECK:STDOUT:   %.loc61: ref %D = deref %a.ref
// CHECK:STDOUT:   %b.ref: %D = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0: %.c0f = impl_witness_access constants.%LeftShiftAssignWith.impl_witness.736, element0 [concrete = constants.%D.as.LeftShiftAssignWith.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc61, %impl.elem0
// CHECK:STDOUT:   %D.as.LeftShiftAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method(%.loc61, %b.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestGenericOp(%T.loc65_19.2: %LeftShift.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %LeftShiftWith.type.loc67_12.2: type = facet_type <@LeftShiftWith, @LeftShiftWith(%T.as_type.loc65_40.1)> [symbolic = %LeftShiftWith.type.loc67_12.2 (constants.%LeftShiftWith.type.3e428a.2)]
// CHECK:STDOUT:   %require_complete.loc67: <witness> = require_complete_type %LeftShiftWith.type.loc67_12.2 [symbolic = %require_complete.loc67 (constants.%require_complete.1dd)]
// CHECK:STDOUT:   %LeftShiftWith.assoc_type: type = assoc_entity_type @LeftShiftWith, @LeftShiftWith(%T.as_type.loc65_40.1) [symbolic = %LeftShiftWith.assoc_type (constants.%LeftShiftWith.assoc_type.7e6d0d.2)]
// CHECK:STDOUT:   %assoc1: @TestGenericOp.%LeftShiftWith.assoc_type (%LeftShiftWith.assoc_type.7e6d0d.2) = assoc_entity element1, imports.%Core.import_ref.c79 [symbolic = %assoc1 (constants.%assoc1.8fd5dc.2)]
// CHECK:STDOUT:   %LeftShiftWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc65_19.1, @LeftShiftWith, @LeftShiftWith(%T.as_type.loc65_40.1) [symbolic = %LeftShiftWith.lookup_impl_witness (constants.%LeftShiftWith.lookup_impl_witness.27975c.2)]
// CHECK:STDOUT:   %LeftShiftWith.facet: @TestGenericOp.%LeftShiftWith.type.loc67_12.2 (%LeftShiftWith.type.3e428a.2) = facet_value %T.as_type.loc65_40.1, (%LeftShiftWith.lookup_impl_witness) [symbolic = %LeftShiftWith.facet (constants.%LeftShiftWith.facet.7d2)]
// CHECK:STDOUT:   %LeftShiftWith.WithSelf.Op.type: type = fn_type @LeftShiftWith.WithSelf.Op, @LeftShiftWith.WithSelf(%T.as_type.loc65_40.1, %LeftShiftWith.facet) [symbolic = %LeftShiftWith.WithSelf.Op.type (constants.%LeftShiftWith.WithSelf.Op.type.7ee)]
// CHECK:STDOUT:   %.loc67_12.2: type = fn_type_with_self_type %LeftShiftWith.WithSelf.Op.type, %LeftShiftWith.facet [symbolic = %.loc67_12.2 (constants.%.08c)]
// CHECK:STDOUT:   %impl.elem1.loc67_12.2: @TestGenericOp.%.loc67_12.2 (%.08c) = impl_witness_access %LeftShiftWith.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc67_12.2 (constants.%impl.elem1.542)]
// CHECK:STDOUT:   %specific_impl_fn.loc67_12.2: <specific function> = specific_impl_function %impl.elem1.loc67_12.2, @LeftShiftWith.WithSelf.Op(%T.as_type.loc65_40.1, %LeftShiftWith.facet) [symbolic = %specific_impl_fn.loc67_12.2 (constants.%specific_impl_fn.348)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @TestGenericOp.%T.as_type.loc65_40.1 (%T.as_type.190bb7.1), %b.param: @TestGenericOp.%T.as_type.loc65_40.1 (%T.as_type.190bb7.1)) -> out %return.param: @TestGenericOp.%T.as_type.loc65_40.1 (%T.as_type.190bb7.1) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %a.ref: @TestGenericOp.%T.as_type.loc65_40.1 (%T.as_type.190bb7.1) = name_ref a, %a
// CHECK:STDOUT:     %b.ref: @TestGenericOp.%T.as_type.loc65_40.1 (%T.as_type.190bb7.1) = name_ref b, %b
// CHECK:STDOUT:     %LeftShiftWith.type.loc67_12.1: type = facet_type <@LeftShiftWith, @LeftShiftWith(constants.%T.as_type.190bb7.1)> [symbolic = %LeftShiftWith.type.loc67_12.2 (constants.%LeftShiftWith.type.3e428a.2)]
// CHECK:STDOUT:     %.loc67_12.1: @TestGenericOp.%LeftShiftWith.assoc_type (%LeftShiftWith.assoc_type.7e6d0d.2) = specific_constant imports.%Core.import_ref.0b1, @LeftShiftWith.WithSelf(constants.%T.as_type.190bb7.1, constants.%Self.a4f) [symbolic = %assoc1 (constants.%assoc1.8fd5dc.2)]
// CHECK:STDOUT:     %Op.ref: @TestGenericOp.%LeftShiftWith.assoc_type (%LeftShiftWith.assoc_type.7e6d0d.2) = name_ref Op, %.loc67_12.1 [symbolic = %assoc1 (constants.%assoc1.8fd5dc.2)]
// CHECK:STDOUT:     %impl.elem1.loc67_12.1: @TestGenericOp.%.loc67_12.2 (%.08c) = impl_witness_access constants.%LeftShiftWith.lookup_impl_witness.27975c.2, element1 [symbolic = %impl.elem1.loc67_12.2 (constants.%impl.elem1.542)]
// CHECK:STDOUT:     %bound_method.loc67_12.1: <bound method> = bound_method %a.ref, %impl.elem1.loc67_12.1
// CHECK:STDOUT:     %specific_impl_fn.loc67_12.1: <specific function> = specific_impl_function %impl.elem1.loc67_12.1, @LeftShiftWith.WithSelf.Op(constants.%T.as_type.190bb7.1, constants.%LeftShiftWith.facet.7d2) [symbolic = %specific_impl_fn.loc67_12.2 (constants.%specific_impl_fn.348)]
// CHECK:STDOUT:     %bound_method.loc67_12.2: <bound method> = bound_method %a.ref, %specific_impl_fn.loc67_12.1
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %LeftShiftWith.WithSelf.Op.call: init @TestGenericOp.%T.as_type.loc65_40.1 (%T.as_type.190bb7.1) to %.loc65_52.1 = call %bound_method.loc67_12.2(%a.ref, %b.ref)
// CHECK:STDOUT:     return %LeftShiftWith.WithSelf.Op.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestGenericAssign(%T.loc71_23.2: %LeftShiftAssign.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %LeftShiftAssignWith.type.loc73_6.2: type = facet_type <@LeftShiftAssignWith, @LeftShiftAssignWith(%T.as_type.loc71_51.1)> [symbolic = %LeftShiftAssignWith.type.loc73_6.2 (constants.%LeftShiftAssignWith.type.93876f.2)]
// CHECK:STDOUT:   %require_complete.loc73: <witness> = require_complete_type %LeftShiftAssignWith.type.loc73_6.2 [symbolic = %require_complete.loc73 (constants.%require_complete.851c02.2)]
// CHECK:STDOUT:   %LeftShiftAssignWith.assoc_type: type = assoc_entity_type @LeftShiftAssignWith, @LeftShiftAssignWith(%T.as_type.loc71_51.1) [symbolic = %LeftShiftAssignWith.assoc_type (constants.%LeftShiftAssignWith.assoc_type.5ac4b3.2)]
// CHECK:STDOUT:   %assoc0: @TestGenericAssign.%LeftShiftAssignWith.assoc_type (%LeftShiftAssignWith.assoc_type.5ac4b3.2) = assoc_entity element0, imports.%Core.import_ref.321 [symbolic = %assoc0 (constants.%assoc0.64dfba.2)]
// CHECK:STDOUT:   %LeftShiftAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc71_23.1, @LeftShiftAssignWith, @LeftShiftAssignWith(%T.as_type.loc71_51.1) [symbolic = %LeftShiftAssignWith.lookup_impl_witness (constants.%LeftShiftAssignWith.lookup_impl_witness)]
// CHECK:STDOUT:   %LeftShiftAssignWith.facet: @TestGenericAssign.%LeftShiftAssignWith.type.loc73_6.2 (%LeftShiftAssignWith.type.93876f.2) = facet_value %T.as_type.loc71_51.1, (%LeftShiftAssignWith.lookup_impl_witness) [symbolic = %LeftShiftAssignWith.facet (constants.%LeftShiftAssignWith.facet.509)]
// CHECK:STDOUT:   %LeftShiftAssignWith.WithSelf.Op.type: type = fn_type @LeftShiftAssignWith.WithSelf.Op, @LeftShiftAssignWith.WithSelf(%T.as_type.loc71_51.1, %LeftShiftAssignWith.facet) [symbolic = %LeftShiftAssignWith.WithSelf.Op.type (constants.%LeftShiftAssignWith.WithSelf.Op.type.574)]
// CHECK:STDOUT:   %.loc73_6.2: type = fn_type_with_self_type %LeftShiftAssignWith.WithSelf.Op.type, %LeftShiftAssignWith.facet [symbolic = %.loc73_6.2 (constants.%.508)]
// CHECK:STDOUT:   %impl.elem0.loc73_6.2: @TestGenericAssign.%.loc73_6.2 (%.508) = impl_witness_access %LeftShiftAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc73_6.2 (constants.%impl.elem0.ce7)]
// CHECK:STDOUT:   %specific_impl_fn.loc73_6.2: <specific function> = specific_impl_function %impl.elem0.loc73_6.2, @LeftShiftAssignWith.WithSelf.Op(%T.as_type.loc71_51.1, %LeftShiftAssignWith.facet) [symbolic = %specific_impl_fn.loc73_6.2 (constants.%specific_impl_fn.9b0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @TestGenericAssign.%ptr.loc71_51.1 (%ptr.360), %b.param: @TestGenericAssign.%T.as_type.loc71_51.1 (%T.as_type.190bb7.2)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %a.ref: @TestGenericAssign.%ptr.loc71_51.1 (%ptr.360) = name_ref a, %a
// CHECK:STDOUT:     %.loc73_3: ref @TestGenericAssign.%T.as_type.loc71_51.1 (%T.as_type.190bb7.2) = deref %a.ref
// CHECK:STDOUT:     %b.ref: @TestGenericAssign.%T.as_type.loc71_51.1 (%T.as_type.190bb7.2) = name_ref b, %b
// CHECK:STDOUT:     %LeftShiftAssignWith.type.loc73_6.1: type = facet_type <@LeftShiftAssignWith, @LeftShiftAssignWith(constants.%T.as_type.190bb7.2)> [symbolic = %LeftShiftAssignWith.type.loc73_6.2 (constants.%LeftShiftAssignWith.type.93876f.2)]
// CHECK:STDOUT:     %.loc73_6.1: @TestGenericAssign.%LeftShiftAssignWith.assoc_type (%LeftShiftAssignWith.assoc_type.5ac4b3.2) = specific_constant imports.%Core.import_ref.54c, @LeftShiftAssignWith.WithSelf(constants.%T.as_type.190bb7.2, constants.%Self.72e) [symbolic = %assoc0 (constants.%assoc0.64dfba.2)]
// CHECK:STDOUT:     %Op.ref: @TestGenericAssign.%LeftShiftAssignWith.assoc_type (%LeftShiftAssignWith.assoc_type.5ac4b3.2) = name_ref Op, %.loc73_6.1 [symbolic = %assoc0 (constants.%assoc0.64dfba.2)]
// CHECK:STDOUT:     %impl.elem0.loc73_6.1: @TestGenericAssign.%.loc73_6.2 (%.508) = impl_witness_access constants.%LeftShiftAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc73_6.2 (constants.%impl.elem0.ce7)]
// CHECK:STDOUT:     %bound_method.loc73_6.1: <bound method> = bound_method %.loc73_3, %impl.elem0.loc73_6.1
// CHECK:STDOUT:     %specific_impl_fn.loc73_6.1: <specific function> = specific_impl_function %impl.elem0.loc73_6.1, @LeftShiftAssignWith.WithSelf.Op(constants.%T.as_type.190bb7.2, constants.%LeftShiftAssignWith.facet.509) [symbolic = %specific_impl_fn.loc73_6.2 (constants.%specific_impl_fn.9b0)]
// CHECK:STDOUT:     %bound_method.loc73_6.2: <bound method> = bound_method %.loc73_3, %specific_impl_fn.loc73_6.1
// CHECK:STDOUT:     %LeftShiftAssignWith.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc73_6.2(%.loc73_3, %b.ref)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGenericOp(constants.%T.f45530.1) {
// CHECK:STDOUT:   %T.patt.loc65_19.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc65_19.1 => constants.%T.f45530.1
// CHECK:STDOUT:   %T.as_type.loc65_40.1 => constants.%T.as_type.190bb7.1
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.86a9db.1
// CHECK:STDOUT:   %a.param_patt.loc65_38.2 => constants.%a.param_patt.e33
// CHECK:STDOUT:   %a.patt.loc65_38.2 => constants.%a.patt.e02
// CHECK:STDOUT:   %b.param_patt.loc65_44.2 => constants.%b.param_patt.4cd741.1
// CHECK:STDOUT:   %b.patt.loc65_44.2 => constants.%b.patt.60acf6.1
// CHECK:STDOUT:   %.loc65_52.2 => constants.%.1f0
// CHECK:STDOUT:   %return.param_patt.loc65_52.2 => constants.%return.param_patt.5f6
// CHECK:STDOUT:   %return.patt.loc65_49.2 => constants.%return.patt.96a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGenericAssign(constants.%T.f45530.2) {
// CHECK:STDOUT:   %T.patt.loc71_23.2 => constants.%T.patt.3a0592.2
// CHECK:STDOUT:   %T.loc71_23.1 => constants.%T.f45530.2
// CHECK:STDOUT:   %T.as_type.loc71_51.1 => constants.%T.as_type.190bb7.2
// CHECK:STDOUT:   %ptr.loc71_51.1 => constants.%ptr.360
// CHECK:STDOUT:   %pattern_type.loc71_48 => constants.%pattern_type.ff5
// CHECK:STDOUT:   %a.param_patt.loc71_48.2 => constants.%a.param_patt.f50
// CHECK:STDOUT:   %a.patt.loc71_48.2 => constants.%a.patt.193
// CHECK:STDOUT:   %pattern_type.loc71_55 => constants.%pattern_type.86a9db.2
// CHECK:STDOUT:   %b.param_patt.loc71_55.2 => constants.%b.param_patt.4cd741.2
// CHECK:STDOUT:   %b.patt.loc71_55.2 => constants.%b.patt.60acf6.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGenericOp(constants.%LeftShift.facet.aff) {
// CHECK:STDOUT:   %T.patt.loc65_19.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc65_19.1 => constants.%LeftShift.facet.aff
// CHECK:STDOUT:   %T.as_type.loc65_40.1 => constants.%C
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.4a7
// CHECK:STDOUT:   %a.param_patt.loc65_38.2 => constants.%a.param_patt.2a5
// CHECK:STDOUT:   %a.patt.loc65_38.2 => constants.%a.patt.a30
// CHECK:STDOUT:   %b.param_patt.loc65_44.2 => constants.%b.param_patt.45f
// CHECK:STDOUT:   %b.patt.loc65_44.2 => constants.%b.patt.562
// CHECK:STDOUT:   %.loc65_52.2 => constants.%.79c
// CHECK:STDOUT:   %return.param_patt.loc65_52.2 => constants.%return.param_patt.415
// CHECK:STDOUT:   %return.patt.loc65_49.2 => constants.%return.patt.340
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc65 => constants.%complete_type.357
// CHECK:STDOUT:   %LeftShiftWith.type.loc67_12.2 => constants.%LeftShiftWith.type.daa
// CHECK:STDOUT:   %require_complete.loc67 => constants.%complete_type.d86
// CHECK:STDOUT:   %LeftShiftWith.assoc_type => constants.%LeftShiftWith.assoc_type.deb
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.b77
// CHECK:STDOUT:   %LeftShiftWith.lookup_impl_witness => constants.%LeftShiftWith.impl_witness.541
// CHECK:STDOUT:   %LeftShiftWith.facet => constants.%LeftShiftWith.facet.d87
// CHECK:STDOUT:   %LeftShiftWith.WithSelf.Op.type => constants.%LeftShiftWith.WithSelf.Op.type.aba
// CHECK:STDOUT:   %.loc67_12.2 => constants.%.c42
// CHECK:STDOUT:   %impl.elem1.loc67_12.2 => constants.%C.as.LeftShiftWith.impl.Op
// CHECK:STDOUT:   %specific_impl_fn.loc67_12.2 => constants.%C.as.LeftShiftWith.impl.Op
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGenericAssign(constants.%LeftShiftAssign.facet.4c3) {
// CHECK:STDOUT:   %T.patt.loc71_23.2 => constants.%T.patt.3a0592.2
// CHECK:STDOUT:   %T.loc71_23.1 => constants.%LeftShiftAssign.facet.4c3
// CHECK:STDOUT:   %T.as_type.loc71_51.1 => constants.%C
// CHECK:STDOUT:   %ptr.loc71_51.1 => constants.%ptr.00e
// CHECK:STDOUT:   %pattern_type.loc71_48 => constants.%pattern_type.842
// CHECK:STDOUT:   %a.param_patt.loc71_48.2 => constants.%a.param_patt.4fa
// CHECK:STDOUT:   %a.patt.loc71_48.2 => constants.%a.patt.eba
// CHECK:STDOUT:   %pattern_type.loc71_55 => constants.%pattern_type.4a7
// CHECK:STDOUT:   %b.param_patt.loc71_55.2 => constants.%b.param_patt.45f
// CHECK:STDOUT:   %b.patt.loc71_55.2 => constants.%b.patt.562
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc71_48 => constants.%complete_type.a56
// CHECK:STDOUT:   %require_complete.loc71_55 => constants.%complete_type.357
// CHECK:STDOUT:   %LeftShiftAssignWith.type.loc73_6.2 => constants.%LeftShiftAssignWith.type.7c8
// CHECK:STDOUT:   %require_complete.loc73 => constants.%complete_type.5cc
// CHECK:STDOUT:   %LeftShiftAssignWith.assoc_type => constants.%LeftShiftAssignWith.assoc_type.5fb
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.bae
// CHECK:STDOUT:   %LeftShiftAssignWith.lookup_impl_witness => constants.%LeftShiftAssignWith.impl_witness.607
// CHECK:STDOUT:   %LeftShiftAssignWith.facet => constants.%LeftShiftAssignWith.facet.c65
// CHECK:STDOUT:   %LeftShiftAssignWith.WithSelf.Op.type => constants.%LeftShiftAssignWith.WithSelf.Op.type.3a4
// CHECK:STDOUT:   %.loc73_6.2 => constants.%.6c6
// CHECK:STDOUT:   %impl.elem0.loc73_6.2 => constants.%C.as.LeftShiftAssignWith.impl.Op
// CHECK:STDOUT:   %specific_impl_fn.loc73_6.2 => constants.%C.as.LeftShiftAssignWith.impl.Op
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGenericOp(constants.%LeftShift.facet.6f9) {
// CHECK:STDOUT:   %T.patt.loc65_19.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc65_19.1 => constants.%LeftShift.facet.6f9
// CHECK:STDOUT:   %T.as_type.loc65_40.1 => constants.%D
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.331
// CHECK:STDOUT:   %a.param_patt.loc65_38.2 => constants.%a.param_patt.d0e
// CHECK:STDOUT:   %a.patt.loc65_38.2 => constants.%a.patt.ec5
// CHECK:STDOUT:   %b.param_patt.loc65_44.2 => constants.%b.param_patt.110
// CHECK:STDOUT:   %b.patt.loc65_44.2 => constants.%b.patt.14c
// CHECK:STDOUT:   %.loc65_52.2 => constants.%.84d
// CHECK:STDOUT:   %return.param_patt.loc65_52.2 => constants.%return.param_patt.835
// CHECK:STDOUT:   %return.patt.loc65_49.2 => constants.%return.patt.5ef
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc65 => constants.%complete_type.357
// CHECK:STDOUT:   %LeftShiftWith.type.loc67_12.2 => constants.%LeftShiftWith.type.aec
// CHECK:STDOUT:   %require_complete.loc67 => constants.%complete_type.f58
// CHECK:STDOUT:   %LeftShiftWith.assoc_type => constants.%LeftShiftWith.assoc_type.c17
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.2c6
// CHECK:STDOUT:   %LeftShiftWith.lookup_impl_witness => constants.%LeftShiftWith.impl_witness.736
// CHECK:STDOUT:   %LeftShiftWith.facet => constants.%LeftShiftWith.facet.e2e
// CHECK:STDOUT:   %LeftShiftWith.WithSelf.Op.type => constants.%LeftShiftWith.WithSelf.Op.type.b14
// CHECK:STDOUT:   %.loc67_12.2 => constants.%.bd1
// CHECK:STDOUT:   %impl.elem1.loc67_12.2 => constants.%D.as.LeftShiftWith.impl.Op
// CHECK:STDOUT:   %specific_impl_fn.loc67_12.2 => constants.%D.as.LeftShiftWith.impl.Op
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGenericAssign(constants.%LeftShiftAssign.facet.6f9) {
// CHECK:STDOUT:   %T.patt.loc71_23.2 => constants.%T.patt.3a0592.2
// CHECK:STDOUT:   %T.loc71_23.1 => constants.%LeftShiftAssign.facet.6f9
// CHECK:STDOUT:   %T.as_type.loc71_51.1 => constants.%D
// CHECK:STDOUT:   %ptr.loc71_51.1 => constants.%ptr.4a5
// CHECK:STDOUT:   %pattern_type.loc71_48 => constants.%pattern_type.ceb
// CHECK:STDOUT:   %a.param_patt.loc71_48.2 => constants.%a.param_patt.609
// CHECK:STDOUT:   %a.patt.loc71_48.2 => constants.%a.patt.fac
// CHECK:STDOUT:   %pattern_type.loc71_55 => constants.%pattern_type.331
// CHECK:STDOUT:   %b.param_patt.loc71_55.2 => constants.%b.param_patt.110
// CHECK:STDOUT:   %b.patt.loc71_55.2 => constants.%b.patt.14c
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc71_48 => constants.%complete_type.320
// CHECK:STDOUT:   %require_complete.loc71_55 => constants.%complete_type.357
// CHECK:STDOUT:   %LeftShiftAssignWith.type.loc73_6.2 => constants.%LeftShiftAssignWith.type.27e
// CHECK:STDOUT:   %require_complete.loc73 => constants.%complete_type.125
// CHECK:STDOUT:   %LeftShiftAssignWith.assoc_type => constants.%LeftShiftAssignWith.assoc_type.2fa
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.742
// CHECK:STDOUT:   %LeftShiftAssignWith.lookup_impl_witness => constants.%LeftShiftAssignWith.impl_witness.736
// CHECK:STDOUT:   %LeftShiftAssignWith.facet => constants.%LeftShiftAssignWith.facet.a72
// CHECK:STDOUT:   %LeftShiftAssignWith.WithSelf.Op.type => constants.%LeftShiftAssignWith.WithSelf.Op.type.73a
// CHECK:STDOUT:   %.loc73_6.2 => constants.%.c0f
// CHECK:STDOUT:   %impl.elem0.loc73_6.2 => constants.%D.as.LeftShiftAssignWith.impl.Op
// CHECK:STDOUT:   %specific_impl_fn.loc73_6.2 => constants.%D.as.LeftShiftAssignWith.impl.Op
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGenericOp(constants.%LeftShift.facet.f35) {
// CHECK:STDOUT:   %T.patt.loc65_19.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc65_19.1 => constants.%LeftShift.facet.f35
// CHECK:STDOUT:   %T.as_type.loc65_40.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.d94
// CHECK:STDOUT:   %a.param_patt.loc65_38.2 => constants.%a.param_patt.076
// CHECK:STDOUT:   %a.patt.loc65_38.2 => constants.%a.patt.96f
// CHECK:STDOUT:   %b.param_patt.loc65_44.2 => constants.%b.param_patt.1a2
// CHECK:STDOUT:   %b.patt.loc65_44.2 => constants.%b.patt.3ca
// CHECK:STDOUT:   %.loc65_52.2 => constants.%.0b3
// CHECK:STDOUT:   %return.param_patt.loc65_52.2 => constants.%return.param_patt.299
// CHECK:STDOUT:   %return.patt.loc65_49.2 => constants.%return.patt.6fa
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc65 => constants.%complete_type.f8a
// CHECK:STDOUT:   %LeftShiftWith.type.loc67_12.2 => constants.%LeftShiftWith.type.795
// CHECK:STDOUT:   %require_complete.loc67 => constants.%complete_type.1f8
// CHECK:STDOUT:   %LeftShiftWith.assoc_type => constants.%LeftShiftWith.assoc_type.3a2
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.1d4
// CHECK:STDOUT:   %LeftShiftWith.lookup_impl_witness => constants.%LeftShiftWith.impl_witness.c59
// CHECK:STDOUT:   %LeftShiftWith.facet => constants.%LeftShiftWith.facet.bdb
// CHECK:STDOUT:   %LeftShiftWith.WithSelf.Op.type => constants.%LeftShiftWith.WithSelf.Op.type.8ba
// CHECK:STDOUT:   %.loc67_12.2 => constants.%.a55
// CHECK:STDOUT:   %impl.elem1.loc67_12.2 => constants.%Int.as.LeftShiftWith.impl.Op.750
// CHECK:STDOUT:   %specific_impl_fn.loc67_12.2 => constants.%Int.as.LeftShiftWith.impl.Op.specific_fn.040
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGenericAssign(constants.%LeftShiftAssign.facet.deb) {
// CHECK:STDOUT:   %T.patt.loc71_23.2 => constants.%T.patt.3a0592.2
// CHECK:STDOUT:   %T.loc71_23.1 => constants.%LeftShiftAssign.facet.deb
// CHECK:STDOUT:   %T.as_type.loc71_51.1 => constants.%i32
// CHECK:STDOUT:   %ptr.loc71_51.1 => constants.%ptr.a17
// CHECK:STDOUT:   %pattern_type.loc71_48 => constants.%pattern_type.7dc
// CHECK:STDOUT:   %a.param_patt.loc71_48.2 => constants.%a.param_patt.977
// CHECK:STDOUT:   %a.patt.loc71_48.2 => constants.%a.patt.b70
// CHECK:STDOUT:   %pattern_type.loc71_55 => constants.%pattern_type.d94
// CHECK:STDOUT:   %b.param_patt.loc71_55.2 => constants.%b.param_patt.1a2
// CHECK:STDOUT:   %b.patt.loc71_55.2 => constants.%b.patt.3ca
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc71_48 => constants.%complete_type.1b1
// CHECK:STDOUT:   %require_complete.loc71_55 => constants.%complete_type.f8a
// CHECK:STDOUT:   %LeftShiftAssignWith.type.loc73_6.2 => constants.%LeftShiftAssignWith.type.e04
// CHECK:STDOUT:   %require_complete.loc73 => constants.%complete_type.8aa
// CHECK:STDOUT:   %LeftShiftAssignWith.assoc_type => constants.%LeftShiftAssignWith.assoc_type.7f9
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.674
// CHECK:STDOUT:   %LeftShiftAssignWith.lookup_impl_witness => constants.%LeftShiftAssignWith.impl_witness.b4d
// CHECK:STDOUT:   %LeftShiftAssignWith.facet => constants.%LeftShiftAssignWith.facet.28f
// CHECK:STDOUT:   %LeftShiftAssignWith.WithSelf.Op.type => constants.%LeftShiftAssignWith.WithSelf.Op.type.353
// CHECK:STDOUT:   %.loc73_6.2 => constants.%.1df
// CHECK:STDOUT:   %impl.elem0.loc73_6.2 => constants.%Int.as.LeftShiftAssignWith.impl.Op.00cc00.2
// CHECK:STDOUT:   %specific_impl_fn.loc73_6.2 => constants.%Int.as.LeftShiftAssignWith.impl.Op.specific_fn.3e135e.1
// CHECK:STDOUT: }
// CHECK:STDOUT:

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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/overloaded/bit_or.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/overloaded/bit_or.carbon

// This file was generated from binary_op.carbon.tmpl. Run make_tests.sh to regenerate.

package User;

class C {};

impl C as Core.BitOrWith(C) where .Result = C {
  fn Op(unused self: C, unused other: C) -> C {
    return {};
  }
}
impl C as Core.BitOrAssignWith(C) {
  fn Op(unused ref self: C, unused other: C) {}
}

fn TestOp(a: C, b: C) -> C {
  //@dump-sem-ir-begin
  return a | b;
  //@dump-sem-ir-end
}

fn TestAssign(a: C*, b: C) {
  //@dump-sem-ir-begin
  *a |= b;
  //@dump-sem-ir-end
}

class D {};

impl D as Core.BitOr {
  fn Op(unused self: D, unused other: D) -> D {
    return {};
  }
}
impl D as Core.BitOrAssign {
  fn Op(unused ref self: D, unused other: D) {}
}

fn TestNamedConstraintOp(a: D, b: D) -> D {
  //@dump-sem-ir-begin
  return a | b;
  //@dump-sem-ir-end
}

fn TestNamedConstraintAssign(a: D*, b: D) {
  //@dump-sem-ir-begin
  *a |= b;
  //@dump-sem-ir-end
}

fn TestGenericOp[T: Core.BitOr](a: T, b: T) -> T {
  //@dump-sem-ir-begin
  return a | b;
  //@dump-sem-ir-end
}

fn TestGenericAssign[T: Core.BitOrAssign](a: T*, b: T) {
  //@dump-sem-ir-begin
  *a |= b;
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

// CHECK:STDOUT: --- bit_or.carbon = $bit_or
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Other: type = symbolic_binding Other, 0 [symbolic]
// CHECK:STDOUT:   %BitOrWith.type.c71: type = facet_type <@BitOrWith, @BitOrWith(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.a48: %BitOrWith.type.c71 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %BitOrWith.assoc_type.ca1: type = assoc_entity_type @BitOrWith, @BitOrWith(%Other) [symbolic]
// CHECK:STDOUT:   %BitOrWith.WithSelf.Op.type.d90: type = fn_type @BitOrWith.WithSelf.Op, @BitOrWith.WithSelf(%Other, %Self.a48) [symbolic]
// CHECK:STDOUT:   %BitOrWith.WithSelf.Op.767: %BitOrWith.WithSelf.Op.type.d90 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitOrWith.type.ba6: type = facet_type <@BitOrWith, @BitOrWith(%C)> [concrete]
// CHECK:STDOUT:   %BitOrWith.assoc_type.a18: type = assoc_entity_type @BitOrWith, @BitOrWith(%C) [concrete]
// CHECK:STDOUT:   %assoc1.0c4: %BitOrWith.assoc_type.a18 = assoc_entity element1, imports.%Core.import_ref.c46 [concrete]
// CHECK:STDOUT:   %BitOrWith.impl_witness.d4d: <witness> = impl_witness @C.as.BitOrWith.impl.%BitOrWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.4a7: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %.79c: Core.Form = init_form %C [concrete]
// CHECK:STDOUT:   %return.param_patt.415: %pattern_type.4a7 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.340: %pattern_type.4a7 = return_slot_pattern %return.param_patt.415, %C [concrete]
// CHECK:STDOUT:   %C.as.BitOrWith.impl.Op.type: type = fn_type @C.as.BitOrWith.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.BitOrWith.impl.Op: %C.as.BitOrWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %BitOrWith.facet.c79: %BitOrWith.type.ba6 = facet_value %C, (%BitOrWith.impl_witness.d4d) [concrete]
// CHECK:STDOUT:   %BitOrWith.WithSelf.Op.type.1aa: type = fn_type @BitOrWith.WithSelf.Op, @BitOrWith.WithSelf(%C, %BitOrWith.facet.c79) [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.type.f23: type = facet_type <@BitOrAssignWith, @BitOrAssignWith(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.102: %BitOrAssignWith.type.f23 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %BitOrAssignWith.assoc_type.13c: type = assoc_entity_type @BitOrAssignWith, @BitOrAssignWith(%Other) [symbolic]
// CHECK:STDOUT:   %BitOrAssignWith.WithSelf.Op.type.e9d: type = fn_type @BitOrAssignWith.WithSelf.Op, @BitOrAssignWith.WithSelf(%Other, %Self.102) [symbolic]
// CHECK:STDOUT:   %BitOrAssignWith.WithSelf.Op.cbb: %BitOrAssignWith.WithSelf.Op.type.e9d = struct_value () [symbolic]
// CHECK:STDOUT:   %BitOrAssignWith.type.c39: type = facet_type <@BitOrAssignWith, @BitOrAssignWith(%C)> [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.impl_witness.fbe: <witness> = impl_witness @C.as.BitOrAssignWith.impl.%BitOrAssignWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.assoc_type.57a: type = assoc_entity_type @BitOrAssignWith, @BitOrAssignWith(%C) [concrete]
// CHECK:STDOUT:   %assoc0.975: %BitOrAssignWith.assoc_type.57a = assoc_entity element0, imports.%Core.import_ref.c9b [concrete]
// CHECK:STDOUT:   %C.as.BitOrAssignWith.impl.Op.type: type = fn_type @C.as.BitOrAssignWith.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.BitOrAssignWith.impl.Op: %C.as.BitOrAssignWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.facet.f0c: %BitOrAssignWith.type.c39 = facet_value %C, (%BitOrAssignWith.impl_witness.fbe) [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.WithSelf.Op.type.0fd: type = fn_type @BitOrAssignWith.WithSelf.Op, @BitOrAssignWith.WithSelf(%C, %BitOrAssignWith.facet.f0c) [concrete]
// CHECK:STDOUT:   %a.param_patt.2a5: %pattern_type.4a7 = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt.a30: %pattern_type.4a7 = wrapper_binding_pattern a, %a.param_patt.2a5 [concrete]
// CHECK:STDOUT:   %b.param_patt.45f: %pattern_type.4a7 = value_param_pattern [concrete]
// CHECK:STDOUT:   %b.patt.562: %pattern_type.4a7 = wrapper_binding_pattern b, %b.param_patt.45f [concrete]
// CHECK:STDOUT:   %assoc1.629: %BitOrWith.assoc_type.ca1 = assoc_entity element1, imports.%Core.import_ref.cac [symbolic]
// CHECK:STDOUT:   %.dcd: type = fn_type_with_self_type %BitOrWith.WithSelf.Op.type.1aa, %BitOrWith.facet.c79 [concrete]
// CHECK:STDOUT:   %ptr.00e: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.842: type = pattern_type %ptr.00e [concrete]
// CHECK:STDOUT:   %a.param_patt.4fa: %pattern_type.842 = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt.eba: %pattern_type.842 = wrapper_binding_pattern a, %a.param_patt.4fa [concrete]
// CHECK:STDOUT:   %assoc0.015: %BitOrAssignWith.assoc_type.13c = assoc_entity element0, imports.%Core.import_ref.b197 [symbolic]
// CHECK:STDOUT:   %.778: type = fn_type_with_self_type %BitOrAssignWith.WithSelf.Op.type.0fd, %BitOrAssignWith.facet.f0c [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %BitOr.type: type = facet_type <@BitOr> [concrete]
// CHECK:STDOUT:   %BitOrWith.impl_witness.736: <witness> = impl_witness @D.as.BitOrWith.impl.%BitOrWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.331: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %.84d: Core.Form = init_form %D [concrete]
// CHECK:STDOUT:   %return.param_patt.835: %pattern_type.331 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.5ef: %pattern_type.331 = return_slot_pattern %return.param_patt.835, %D [concrete]
// CHECK:STDOUT:   %D.as.BitOrWith.impl.Op.type: type = fn_type @D.as.BitOrWith.impl.Op [concrete]
// CHECK:STDOUT:   %D.as.BitOrWith.impl.Op: %D.as.BitOrWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %BitOrWith.type.d72: type = facet_type <@BitOrWith, @BitOrWith(%D)> [concrete]
// CHECK:STDOUT:   %BitOrWith.facet.d76: %BitOrWith.type.d72 = facet_value %D, (%BitOrWith.impl_witness.736) [concrete]
// CHECK:STDOUT:   %BitOrWith.assoc_type.35d: type = assoc_entity_type @BitOrWith, @BitOrWith(%D) [concrete]
// CHECK:STDOUT:   %BitOrWith.WithSelf.Op.type.51c: type = fn_type @BitOrWith.WithSelf.Op, @BitOrWith.WithSelf(%D, %BitOrWith.facet.d76) [concrete]
// CHECK:STDOUT:   %assoc1.ae2: %BitOrWith.assoc_type.35d = assoc_entity element1, imports.%Core.import_ref.c46 [concrete]
// CHECK:STDOUT:   %BitOrAssign.type: type = facet_type <@BitOrAssign> [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.impl_witness.736: <witness> = impl_witness @D.as.BitOrAssignWith.impl.%BitOrAssignWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %D.as.BitOrAssignWith.impl.Op.type: type = fn_type @D.as.BitOrAssignWith.impl.Op [concrete]
// CHECK:STDOUT:   %D.as.BitOrAssignWith.impl.Op: %D.as.BitOrAssignWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.type.b08: type = facet_type <@BitOrAssignWith, @BitOrAssignWith(%D)> [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.facet.fc9: %BitOrAssignWith.type.b08 = facet_value %D, (%BitOrAssignWith.impl_witness.736) [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.WithSelf.Op.type.c7e: type = fn_type @BitOrAssignWith.WithSelf.Op, @BitOrAssignWith.WithSelf(%D, %BitOrAssignWith.facet.fc9) [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.assoc_type.2dc: type = assoc_entity_type @BitOrAssignWith, @BitOrAssignWith(%D) [concrete]
// CHECK:STDOUT:   %assoc0.7bb: %BitOrAssignWith.assoc_type.2dc = assoc_entity element0, imports.%Core.import_ref.c9b [concrete]
// CHECK:STDOUT:   %a.param_patt.d0e: %pattern_type.331 = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt.ec5: %pattern_type.331 = wrapper_binding_pattern a, %a.param_patt.d0e [concrete]
// CHECK:STDOUT:   %b.param_patt.110: %pattern_type.331 = value_param_pattern [concrete]
// CHECK:STDOUT:   %b.patt.14c: %pattern_type.331 = wrapper_binding_pattern b, %b.param_patt.110 [concrete]
// CHECK:STDOUT:   %.571: type = fn_type_with_self_type %BitOrWith.WithSelf.Op.type.51c, %BitOrWith.facet.d76 [concrete]
// CHECK:STDOUT:   %ptr.4a5: type = ptr_type %D [concrete]
// CHECK:STDOUT:   %pattern_type.ceb: type = pattern_type %ptr.4a5 [concrete]
// CHECK:STDOUT:   %a.param_patt.609: %pattern_type.ceb = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt.fac: %pattern_type.ceb = wrapper_binding_pattern a, %a.param_patt.609 [concrete]
// CHECK:STDOUT:   %.dde: type = fn_type_with_self_type %BitOrAssignWith.WithSelf.Op.type.c7e, %BitOrAssignWith.facet.fc9 [concrete]
// CHECK:STDOUT:   %pattern_type.9a587c.2: type = pattern_type %BitOr.type [concrete]
// CHECK:STDOUT:   %T.patt.3a0592.1: %pattern_type.9a587c.2 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.f45530.1: %BitOr.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.190bb7.1: type = facet_access_type %T.f45530.1 [symbolic]
// CHECK:STDOUT:   %pattern_type.86a9db.1: type = pattern_type %T.as_type.190bb7.1 [symbolic]
// CHECK:STDOUT:   %a.param_patt.e33: %pattern_type.86a9db.1 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %a.patt.e02: %pattern_type.86a9db.1 = wrapper_binding_pattern a, %a.param_patt.e33 [symbolic]
// CHECK:STDOUT:   %b.param_patt.4cd741.1: %pattern_type.86a9db.1 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %b.patt.60acf6.1: %pattern_type.86a9db.1 = wrapper_binding_pattern b, %b.param_patt.4cd741.1 [symbolic]
// CHECK:STDOUT:   %.1f0: Core.Form = init_form %T.as_type.190bb7.1 [symbolic]
// CHECK:STDOUT:   %return.param_patt.5f6: %pattern_type.86a9db.1 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.96a: %pattern_type.86a9db.1 = return_slot_pattern %return.param_patt.5f6, %T.as_type.190bb7.1 [symbolic]
// CHECK:STDOUT:   %BitOrWith.type.d86ffd.2: type = facet_type <@BitOrWith, @BitOrWith(%T.as_type.190bb7.1)> [symbolic]
// CHECK:STDOUT:   %BitOrWith.assoc_type.e41a2a.2: type = assoc_entity_type @BitOrWith, @BitOrWith(%T.as_type.190bb7.1) [symbolic]
// CHECK:STDOUT:   %assoc1.941bc6.2: %BitOrWith.assoc_type.e41a2a.2 = assoc_entity element1, imports.%Core.import_ref.c46 [symbolic]
// CHECK:STDOUT:   %require_complete.6d6: <witness> = require_complete_type %BitOrWith.type.d86ffd.2 [symbolic]
// CHECK:STDOUT:   %BitOrWith.lookup_impl_witness.f5b0d3.2: <witness> = lookup_impl_witness %T.f45530.1, @BitOrWith, @BitOrWith(%T.as_type.190bb7.1) [symbolic]
// CHECK:STDOUT:   %BitOrWith.facet.d9c: %BitOrWith.type.d86ffd.2 = facet_value %T.as_type.190bb7.1, (%BitOrWith.lookup_impl_witness.f5b0d3.2) [symbolic]
// CHECK:STDOUT:   %BitOrWith.WithSelf.Op.type.c2e: type = fn_type @BitOrWith.WithSelf.Op, @BitOrWith.WithSelf(%T.as_type.190bb7.1, %BitOrWith.facet.d9c) [symbolic]
// CHECK:STDOUT:   %.720: type = fn_type_with_self_type %BitOrWith.WithSelf.Op.type.c2e, %BitOrWith.facet.d9c [symbolic]
// CHECK:STDOUT:   %impl.elem1.9f6: %.720 = impl_witness_access %BitOrWith.lookup_impl_witness.f5b0d3.2, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.1a1: <specific function> = specific_impl_function %impl.elem1.9f6, @BitOrWith.WithSelf.Op(%T.as_type.190bb7.1, %BitOrWith.facet.d9c) [symbolic]
// CHECK:STDOUT:   %pattern_type.9a587c.3: type = pattern_type %BitOrAssign.type [concrete]
// CHECK:STDOUT:   %T.patt.3a0592.2: %pattern_type.9a587c.3 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.f45530.2: %BitOrAssign.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.190bb7.2: type = facet_access_type %T.f45530.2 [symbolic]
// CHECK:STDOUT:   %ptr.360: type = ptr_type %T.as_type.190bb7.2 [symbolic]
// CHECK:STDOUT:   %pattern_type.ff5: type = pattern_type %ptr.360 [symbolic]
// CHECK:STDOUT:   %a.param_patt.f50: %pattern_type.ff5 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %a.patt.193: %pattern_type.ff5 = wrapper_binding_pattern a, %a.param_patt.f50 [symbolic]
// CHECK:STDOUT:   %pattern_type.86a9db.2: type = pattern_type %T.as_type.190bb7.2 [symbolic]
// CHECK:STDOUT:   %b.param_patt.4cd741.2: %pattern_type.86a9db.2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %b.patt.60acf6.2: %pattern_type.86a9db.2 = wrapper_binding_pattern b, %b.param_patt.4cd741.2 [symbolic]
// CHECK:STDOUT:   %BitOrAssignWith.type.21b79c.2: type = facet_type <@BitOrAssignWith, @BitOrAssignWith(%T.as_type.190bb7.2)> [symbolic]
// CHECK:STDOUT:   %BitOrAssignWith.assoc_type.ff74e4.2: type = assoc_entity_type @BitOrAssignWith, @BitOrAssignWith(%T.as_type.190bb7.2) [symbolic]
// CHECK:STDOUT:   %assoc0.154089.2: %BitOrAssignWith.assoc_type.ff74e4.2 = assoc_entity element0, imports.%Core.import_ref.c9b [symbolic]
// CHECK:STDOUT:   %require_complete.11742c.2: <witness> = require_complete_type %BitOrAssignWith.type.21b79c.2 [symbolic]
// CHECK:STDOUT:   %BitOrAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.f45530.2, @BitOrAssignWith, @BitOrAssignWith(%T.as_type.190bb7.2) [symbolic]
// CHECK:STDOUT:   %BitOrAssignWith.facet.c2c: %BitOrAssignWith.type.21b79c.2 = facet_value %T.as_type.190bb7.2, (%BitOrAssignWith.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %BitOrAssignWith.WithSelf.Op.type.ed3: type = fn_type @BitOrAssignWith.WithSelf.Op, @BitOrAssignWith.WithSelf(%T.as_type.190bb7.2, %BitOrAssignWith.facet.c2c) [symbolic]
// CHECK:STDOUT:   %.9d4: type = fn_type_with_self_type %BitOrAssignWith.WithSelf.Op.type.ed3, %BitOrAssignWith.facet.c2c [symbolic]
// CHECK:STDOUT:   %impl.elem0.4ed: %.9d4 = impl_witness_access %BitOrAssignWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.f6b: <specific function> = specific_impl_function %impl.elem0.4ed, @BitOrAssignWith.WithSelf.Op(%T.as_type.190bb7.2, %BitOrAssignWith.facet.c2c) [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %Int.d4bbe0.1: type = class_type @Int, @Int(%N) [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %ptr.a17: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %ptr.a17 [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %BitOr.facet.423: %BitOr.type = facet_value %C, (%BitOrWith.impl_witness.d4d) [concrete]
// CHECK:STDOUT:   %BitOrAssign.facet.37c: %BitOrAssign.type = facet_value %C, (%BitOrAssignWith.impl_witness.fbe) [concrete]
// CHECK:STDOUT:   %BitOr.facet.6f9: %BitOr.type = facet_value %D, (%BitOrWith.impl_witness.736) [concrete]
// CHECK:STDOUT:   %BitOrAssign.facet.6f9: %BitOrAssign.type = facet_value %D, (%BitOrAssignWith.impl_witness.736) [concrete]
// CHECK:STDOUT:   %Int.as.BitOrWith.impl.Op.type.8a5: type = fn_type @Int.as.BitOrWith.impl.Op.1, @Int.as.BitOrWith.impl.ac1(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.BitOrWith.impl.Op.3d4: %Int.as.BitOrWith.impl.Op.type.8a5 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.81c7d0.1: type = facet_type <@ImplicitAs, @ImplicitAs(%Int.d4bbe0.1)> [symbolic]
// CHECK:STDOUT:   %U.f16: %ImplicitAs.type.81c7d0.1 = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %BitOrWith.type.06c: type = facet_type <@BitOrWith, @BitOrWith(%i32)> [concrete]
// CHECK:STDOUT:   %BitOrWith.impl_witness.58e: <witness> = impl_witness imports.%BitOrWith.impl_witness_table.ac1, @Int.as.BitOrWith.impl.ac1(%int_32) [concrete]
// CHECK:STDOUT:   %BitOrWith.assoc_type.e85: type = assoc_entity_type @BitOrWith, @BitOrWith(%i32) [concrete]
// CHECK:STDOUT:   %assoc1.8e2: %BitOrWith.assoc_type.e85 = assoc_entity element1, imports.%Core.import_ref.c46 [concrete]
// CHECK:STDOUT:   %complete_type.7b9: <witness> = complete_type_witness %BitOrWith.type.06c [concrete]
// CHECK:STDOUT:   %Int.as.BitOrWith.impl.Op.type.2f5: type = fn_type @Int.as.BitOrWith.impl.Op.1, @Int.as.BitOrWith.impl.ac1(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.BitOrWith.impl.Op.78a: %Int.as.BitOrWith.impl.Op.type.2f5 = struct_value () [concrete]
// CHECK:STDOUT:   %BitOr.facet.85b: %BitOr.type = facet_value %i32, (%BitOrWith.impl_witness.58e) [concrete]
// CHECK:STDOUT:   %a.param_patt.076: %pattern_type.d94 = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt.96f: %pattern_type.d94 = wrapper_binding_pattern a, %a.param_patt.076 [concrete]
// CHECK:STDOUT:   %b.param_patt.1a2: %pattern_type.d94 = value_param_pattern [concrete]
// CHECK:STDOUT:   %b.patt.3ca: %pattern_type.d94 = wrapper_binding_pattern b, %b.param_patt.1a2 [concrete]
// CHECK:STDOUT:   %.0b3: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.299: %pattern_type.d94 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.6fa: %pattern_type.d94 = return_slot_pattern %return.param_patt.299, %i32 [concrete]
// CHECK:STDOUT:   %Int.as.BitOrAssignWith.impl.Op.type.906d57.1: type = fn_type @Int.as.BitOrAssignWith.impl.Op.1, @Int.as.BitOrAssignWith.impl(%N, %U.f16) [symbolic]
// CHECK:STDOUT:   %Int.as.BitOrAssignWith.impl.Op.2043b8.1: %Int.as.BitOrAssignWith.impl.Op.type.906d57.1 = struct_value () [symbolic]
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
// CHECK:STDOUT:   %BitOrAssignWith.type.8b6: type = facet_type <@BitOrAssignWith, @BitOrAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.impl_witness.a9c: <witness> = impl_witness imports.%BitOrAssignWith.impl_witness_table, @Int.as.BitOrAssignWith.impl(%int_32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %BitOrAssign.facet.471: %BitOrAssign.type = facet_value %i32, (%BitOrAssignWith.impl_witness.a9c) [concrete]
// CHECK:STDOUT:   %a.param_patt.977: %pattern_type.7dc = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt.b70: %pattern_type.7dc = wrapper_binding_pattern a, %a.param_patt.977 [concrete]
// CHECK:STDOUT:   %complete_type.8f6: <witness> = complete_type_witness %BitOrWith.type.ba6 [concrete]
// CHECK:STDOUT:   %complete_type.a56: <witness> = complete_type_witness %ptr.00e [concrete]
// CHECK:STDOUT:   %complete_type.c5a: <witness> = complete_type_witness %BitOrAssignWith.type.c39 [concrete]
// CHECK:STDOUT:   %complete_type.ae2: <witness> = complete_type_witness %BitOrWith.type.d72 [concrete]
// CHECK:STDOUT:   %complete_type.320: <witness> = complete_type_witness %ptr.4a5 [concrete]
// CHECK:STDOUT:   %complete_type.9d8: <witness> = complete_type_witness %BitOrAssignWith.type.b08 [concrete]
// CHECK:STDOUT:   %BitOrWith.facet.603: %BitOrWith.type.06c = facet_value %i32, (%BitOrWith.impl_witness.58e) [concrete]
// CHECK:STDOUT:   %BitOrWith.WithSelf.Op.type.0dc: type = fn_type @BitOrWith.WithSelf.Op, @BitOrWith.WithSelf(%i32, %BitOrWith.facet.603) [concrete]
// CHECK:STDOUT:   %.6d9: type = fn_type_with_self_type %BitOrWith.WithSelf.Op.type.0dc, %BitOrWith.facet.603 [concrete]
// CHECK:STDOUT:   %Int.as.BitOrWith.impl.Op.specific_fn.18a: <specific function> = specific_function %Int.as.BitOrWith.impl.Op.78a, @Int.as.BitOrWith.impl.Op.1(%int_32) [concrete]
// CHECK:STDOUT:   %complete_type.1b1: <witness> = complete_type_witness %ptr.a17 [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.assoc_type.f2c: type = assoc_entity_type @BitOrAssignWith, @BitOrAssignWith(%i32) [concrete]
// CHECK:STDOUT:   %assoc0.774: %BitOrAssignWith.assoc_type.f2c = assoc_entity element0, imports.%Core.import_ref.c9b [concrete]
// CHECK:STDOUT:   %complete_type.b58: <witness> = complete_type_witness %BitOrAssignWith.type.8b6 [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.facet.5cc: %BitOrAssignWith.type.8b6 = facet_value %i32, (%BitOrAssignWith.impl_witness.a9c) [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.WithSelf.Op.type.d0d: type = fn_type @BitOrAssignWith.WithSelf.Op, @BitOrAssignWith.WithSelf(%i32, %BitOrAssignWith.facet.5cc) [concrete]
// CHECK:STDOUT:   %.0cdb: type = fn_type_with_self_type %BitOrAssignWith.WithSelf.Op.type.d0d, %BitOrAssignWith.facet.5cc [concrete]
// CHECK:STDOUT:   %Int.as.BitOrAssignWith.impl.Op.type.96dfc8.2: type = fn_type @Int.as.BitOrAssignWith.impl.Op.1, @Int.as.BitOrAssignWith.impl(%int_32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Int.as.BitOrAssignWith.impl.Op.e3e6a9.2: %Int.as.BitOrAssignWith.impl.Op.type.96dfc8.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.BitOrAssignWith.impl.Op.specific_fn.0e69c3.1: <specific function> = specific_function %Int.as.BitOrAssignWith.impl.Op.e3e6a9.2, @Int.as.BitOrAssignWith.impl.Op.1(%int_32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.412: @BitOrWith.WithSelf.%BitOrWith.assoc_type (%BitOrWith.assoc_type.ca1) = import_ref Core//prelude/operators/bitwise, loc{{\d+_\d+}}, loaded [symbolic = @BitOrWith.WithSelf.%assoc1 (constants.%assoc1.629)]
// CHECK:STDOUT:   %Core.import_ref.c46: @BitOrWith.WithSelf.%BitOrWith.WithSelf.Op.type (%BitOrWith.WithSelf.Op.type.d90) = import_ref Core//prelude/operators/bitwise, loc{{\d+_\d+}}, loaded [symbolic = @BitOrWith.WithSelf.%BitOrWith.WithSelf.Op (constants.%BitOrWith.WithSelf.Op.767)]
// CHECK:STDOUT:   %Core.import_ref.4c2: @BitOrAssignWith.WithSelf.%BitOrAssignWith.assoc_type (%BitOrAssignWith.assoc_type.13c) = import_ref Core//prelude/operators/bitwise, loc{{\d+_\d+}}, loaded [symbolic = @BitOrAssignWith.WithSelf.%assoc0 (constants.%assoc0.015)]
// CHECK:STDOUT:   %Core.import_ref.c9b: @BitOrAssignWith.WithSelf.%BitOrAssignWith.WithSelf.Op.type (%BitOrAssignWith.WithSelf.Op.type.e9d) = import_ref Core//prelude/operators/bitwise, loc{{\d+_\d+}}, loaded [symbolic = @BitOrAssignWith.WithSelf.%BitOrAssignWith.WithSelf.Op (constants.%BitOrAssignWith.WithSelf.Op.cbb)]
// CHECK:STDOUT:   %Core.import_ref.cac = import_ref Core//prelude/operators/bitwise, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.b197 = import_ref Core//prelude/operators/bitwise, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.ebd: type = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.BitOrWith.impl.ac1.%Int (constants.%Int.d4bbe0.1)]
// CHECK:STDOUT:   %Core.import_ref.043: @Int.as.BitOrWith.impl.ac1.%Int.as.BitOrWith.impl.Op.type (%Int.as.BitOrWith.impl.Op.type.8a5) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.BitOrWith.impl.ac1.%Int.as.BitOrWith.impl.Op (constants.%Int.as.BitOrWith.impl.Op.3d4)]
// CHECK:STDOUT:   %BitOrWith.impl_witness_table.ac1 = impl_witness_table (%Core.import_ref.ebd, %Core.import_ref.043), @Int.as.BitOrWith.impl.ac1 [concrete]
// CHECK:STDOUT:   %Core.import_ref.a19: @Int.as.BitOrAssignWith.impl.%Int.as.BitOrAssignWith.impl.Op.type.2 (%Int.as.BitOrAssignWith.impl.Op.type.906d57.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.BitOrAssignWith.impl.%Int.as.BitOrAssignWith.impl.Op.2 (constants.%Int.as.BitOrAssignWith.impl.Op.2043b8.1)]
// CHECK:STDOUT:   %BitOrAssignWith.impl_witness_table = impl_witness_table (%Core.import_ref.a19), @Int.as.BitOrAssignWith.impl [concrete]
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
// CHECK:STDOUT:   %impl.elem1: %.dcd = impl_witness_access constants.%BitOrWith.impl_witness.d4d, element1 [concrete = constants.%C.as.BitOrWith.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.ref, %impl.elem1
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.as.BitOrWith.impl.Op.call: init %C to %.loc30_26.1 = call %bound_method(%a.ref, %b.ref)
// CHECK:STDOUT:   return %C.as.BitOrWith.impl.Op.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestAssign(%a.param: %ptr.00e, %b.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %ptr.00e = name_ref a, %a
// CHECK:STDOUT:   %.loc38: ref %C = deref %a.ref
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0: %.778 = impl_witness_access constants.%BitOrAssignWith.impl_witness.fbe, element0 [concrete = constants.%C.as.BitOrAssignWith.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc38, %impl.elem0
// CHECK:STDOUT:   %C.as.BitOrAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method(%.loc38, %b.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestNamedConstraintOp(%a.param: %D, %b.param: %D) -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %D = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %D = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1: %.571 = impl_witness_access constants.%BitOrWith.impl_witness.736, element1 [concrete = constants.%D.as.BitOrWith.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.ref, %impl.elem1
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %D.as.BitOrWith.impl.Op.call: init %D to %.loc53_41.1 = call %bound_method(%a.ref, %b.ref)
// CHECK:STDOUT:   return %D.as.BitOrWith.impl.Op.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestNamedConstraintAssign(%a.param: %ptr.4a5, %b.param: %D) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %ptr.4a5 = name_ref a, %a
// CHECK:STDOUT:   %.loc61: ref %D = deref %a.ref
// CHECK:STDOUT:   %b.ref: %D = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0: %.dde = impl_witness_access constants.%BitOrAssignWith.impl_witness.736, element0 [concrete = constants.%D.as.BitOrAssignWith.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc61, %impl.elem0
// CHECK:STDOUT:   %D.as.BitOrAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method(%.loc61, %b.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestGenericOp(%T.loc65_19.2: %BitOr.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %BitOrWith.type.loc67_12.2: type = facet_type <@BitOrWith, @BitOrWith(%T.as_type.loc65_36.1)> [symbolic = %BitOrWith.type.loc67_12.2 (constants.%BitOrWith.type.d86ffd.2)]
// CHECK:STDOUT:   %require_complete.loc67: <witness> = require_complete_type %BitOrWith.type.loc67_12.2 [symbolic = %require_complete.loc67 (constants.%require_complete.6d6)]
// CHECK:STDOUT:   %BitOrWith.assoc_type: type = assoc_entity_type @BitOrWith, @BitOrWith(%T.as_type.loc65_36.1) [symbolic = %BitOrWith.assoc_type (constants.%BitOrWith.assoc_type.e41a2a.2)]
// CHECK:STDOUT:   %assoc1: @TestGenericOp.%BitOrWith.assoc_type (%BitOrWith.assoc_type.e41a2a.2) = assoc_entity element1, imports.%Core.import_ref.c46 [symbolic = %assoc1 (constants.%assoc1.941bc6.2)]
// CHECK:STDOUT:   %BitOrWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc65_19.1, @BitOrWith, @BitOrWith(%T.as_type.loc65_36.1) [symbolic = %BitOrWith.lookup_impl_witness (constants.%BitOrWith.lookup_impl_witness.f5b0d3.2)]
// CHECK:STDOUT:   %BitOrWith.facet: @TestGenericOp.%BitOrWith.type.loc67_12.2 (%BitOrWith.type.d86ffd.2) = facet_value %T.as_type.loc65_36.1, (%BitOrWith.lookup_impl_witness) [symbolic = %BitOrWith.facet (constants.%BitOrWith.facet.d9c)]
// CHECK:STDOUT:   %BitOrWith.WithSelf.Op.type: type = fn_type @BitOrWith.WithSelf.Op, @BitOrWith.WithSelf(%T.as_type.loc65_36.1, %BitOrWith.facet) [symbolic = %BitOrWith.WithSelf.Op.type (constants.%BitOrWith.WithSelf.Op.type.c2e)]
// CHECK:STDOUT:   %.loc67_12.2: type = fn_type_with_self_type %BitOrWith.WithSelf.Op.type, %BitOrWith.facet [symbolic = %.loc67_12.2 (constants.%.720)]
// CHECK:STDOUT:   %impl.elem1.loc67_12.2: @TestGenericOp.%.loc67_12.2 (%.720) = impl_witness_access %BitOrWith.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc67_12.2 (constants.%impl.elem1.9f6)]
// CHECK:STDOUT:   %specific_impl_fn.loc67_12.2: <specific function> = specific_impl_function %impl.elem1.loc67_12.2, @BitOrWith.WithSelf.Op(%T.as_type.loc65_36.1, %BitOrWith.facet) [symbolic = %specific_impl_fn.loc67_12.2 (constants.%specific_impl_fn.1a1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @TestGenericOp.%T.as_type.loc65_36.1 (%T.as_type.190bb7.1), %b.param: @TestGenericOp.%T.as_type.loc65_36.1 (%T.as_type.190bb7.1)) -> out %return.param: @TestGenericOp.%T.as_type.loc65_36.1 (%T.as_type.190bb7.1) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %a.ref: @TestGenericOp.%T.as_type.loc65_36.1 (%T.as_type.190bb7.1) = name_ref a, %a
// CHECK:STDOUT:     %b.ref: @TestGenericOp.%T.as_type.loc65_36.1 (%T.as_type.190bb7.1) = name_ref b, %b
// CHECK:STDOUT:     %BitOrWith.type.loc67_12.1: type = facet_type <@BitOrWith, @BitOrWith(constants.%T.as_type.190bb7.1)> [symbolic = %BitOrWith.type.loc67_12.2 (constants.%BitOrWith.type.d86ffd.2)]
// CHECK:STDOUT:     %.loc67_12.1: @TestGenericOp.%BitOrWith.assoc_type (%BitOrWith.assoc_type.e41a2a.2) = specific_constant imports.%Core.import_ref.412, @BitOrWith.WithSelf(constants.%T.as_type.190bb7.1, constants.%Self.a48) [symbolic = %assoc1 (constants.%assoc1.941bc6.2)]
// CHECK:STDOUT:     %Op.ref: @TestGenericOp.%BitOrWith.assoc_type (%BitOrWith.assoc_type.e41a2a.2) = name_ref Op, %.loc67_12.1 [symbolic = %assoc1 (constants.%assoc1.941bc6.2)]
// CHECK:STDOUT:     %impl.elem1.loc67_12.1: @TestGenericOp.%.loc67_12.2 (%.720) = impl_witness_access constants.%BitOrWith.lookup_impl_witness.f5b0d3.2, element1 [symbolic = %impl.elem1.loc67_12.2 (constants.%impl.elem1.9f6)]
// CHECK:STDOUT:     %bound_method.loc67_12.1: <bound method> = bound_method %a.ref, %impl.elem1.loc67_12.1
// CHECK:STDOUT:     %specific_impl_fn.loc67_12.1: <specific function> = specific_impl_function %impl.elem1.loc67_12.1, @BitOrWith.WithSelf.Op(constants.%T.as_type.190bb7.1, constants.%BitOrWith.facet.d9c) [symbolic = %specific_impl_fn.loc67_12.2 (constants.%specific_impl_fn.1a1)]
// CHECK:STDOUT:     %bound_method.loc67_12.2: <bound method> = bound_method %a.ref, %specific_impl_fn.loc67_12.1
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %BitOrWith.WithSelf.Op.call: init @TestGenericOp.%T.as_type.loc65_36.1 (%T.as_type.190bb7.1) to %.loc65_48.1 = call %bound_method.loc67_12.2(%a.ref, %b.ref)
// CHECK:STDOUT:     return %BitOrWith.WithSelf.Op.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestGenericAssign(%T.loc71_23.2: %BitOrAssign.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %BitOrAssignWith.type.loc73_6.2: type = facet_type <@BitOrAssignWith, @BitOrAssignWith(%T.as_type.loc71_47.1)> [symbolic = %BitOrAssignWith.type.loc73_6.2 (constants.%BitOrAssignWith.type.21b79c.2)]
// CHECK:STDOUT:   %require_complete.loc73: <witness> = require_complete_type %BitOrAssignWith.type.loc73_6.2 [symbolic = %require_complete.loc73 (constants.%require_complete.11742c.2)]
// CHECK:STDOUT:   %BitOrAssignWith.assoc_type: type = assoc_entity_type @BitOrAssignWith, @BitOrAssignWith(%T.as_type.loc71_47.1) [symbolic = %BitOrAssignWith.assoc_type (constants.%BitOrAssignWith.assoc_type.ff74e4.2)]
// CHECK:STDOUT:   %assoc0: @TestGenericAssign.%BitOrAssignWith.assoc_type (%BitOrAssignWith.assoc_type.ff74e4.2) = assoc_entity element0, imports.%Core.import_ref.c9b [symbolic = %assoc0 (constants.%assoc0.154089.2)]
// CHECK:STDOUT:   %BitOrAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc71_23.1, @BitOrAssignWith, @BitOrAssignWith(%T.as_type.loc71_47.1) [symbolic = %BitOrAssignWith.lookup_impl_witness (constants.%BitOrAssignWith.lookup_impl_witness)]
// CHECK:STDOUT:   %BitOrAssignWith.facet: @TestGenericAssign.%BitOrAssignWith.type.loc73_6.2 (%BitOrAssignWith.type.21b79c.2) = facet_value %T.as_type.loc71_47.1, (%BitOrAssignWith.lookup_impl_witness) [symbolic = %BitOrAssignWith.facet (constants.%BitOrAssignWith.facet.c2c)]
// CHECK:STDOUT:   %BitOrAssignWith.WithSelf.Op.type: type = fn_type @BitOrAssignWith.WithSelf.Op, @BitOrAssignWith.WithSelf(%T.as_type.loc71_47.1, %BitOrAssignWith.facet) [symbolic = %BitOrAssignWith.WithSelf.Op.type (constants.%BitOrAssignWith.WithSelf.Op.type.ed3)]
// CHECK:STDOUT:   %.loc73_6.2: type = fn_type_with_self_type %BitOrAssignWith.WithSelf.Op.type, %BitOrAssignWith.facet [symbolic = %.loc73_6.2 (constants.%.9d4)]
// CHECK:STDOUT:   %impl.elem0.loc73_6.2: @TestGenericAssign.%.loc73_6.2 (%.9d4) = impl_witness_access %BitOrAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc73_6.2 (constants.%impl.elem0.4ed)]
// CHECK:STDOUT:   %specific_impl_fn.loc73_6.2: <specific function> = specific_impl_function %impl.elem0.loc73_6.2, @BitOrAssignWith.WithSelf.Op(%T.as_type.loc71_47.1, %BitOrAssignWith.facet) [symbolic = %specific_impl_fn.loc73_6.2 (constants.%specific_impl_fn.f6b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @TestGenericAssign.%ptr.loc71_47.1 (%ptr.360), %b.param: @TestGenericAssign.%T.as_type.loc71_47.1 (%T.as_type.190bb7.2)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %a.ref: @TestGenericAssign.%ptr.loc71_47.1 (%ptr.360) = name_ref a, %a
// CHECK:STDOUT:     %.loc73_3: ref @TestGenericAssign.%T.as_type.loc71_47.1 (%T.as_type.190bb7.2) = deref %a.ref
// CHECK:STDOUT:     %b.ref: @TestGenericAssign.%T.as_type.loc71_47.1 (%T.as_type.190bb7.2) = name_ref b, %b
// CHECK:STDOUT:     %BitOrAssignWith.type.loc73_6.1: type = facet_type <@BitOrAssignWith, @BitOrAssignWith(constants.%T.as_type.190bb7.2)> [symbolic = %BitOrAssignWith.type.loc73_6.2 (constants.%BitOrAssignWith.type.21b79c.2)]
// CHECK:STDOUT:     %.loc73_6.1: @TestGenericAssign.%BitOrAssignWith.assoc_type (%BitOrAssignWith.assoc_type.ff74e4.2) = specific_constant imports.%Core.import_ref.4c2, @BitOrAssignWith.WithSelf(constants.%T.as_type.190bb7.2, constants.%Self.102) [symbolic = %assoc0 (constants.%assoc0.154089.2)]
// CHECK:STDOUT:     %Op.ref: @TestGenericAssign.%BitOrAssignWith.assoc_type (%BitOrAssignWith.assoc_type.ff74e4.2) = name_ref Op, %.loc73_6.1 [symbolic = %assoc0 (constants.%assoc0.154089.2)]
// CHECK:STDOUT:     %impl.elem0.loc73_6.1: @TestGenericAssign.%.loc73_6.2 (%.9d4) = impl_witness_access constants.%BitOrAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc73_6.2 (constants.%impl.elem0.4ed)]
// CHECK:STDOUT:     %bound_method.loc73_6.1: <bound method> = bound_method %.loc73_3, %impl.elem0.loc73_6.1
// CHECK:STDOUT:     %specific_impl_fn.loc73_6.1: <specific function> = specific_impl_function %impl.elem0.loc73_6.1, @BitOrAssignWith.WithSelf.Op(constants.%T.as_type.190bb7.2, constants.%BitOrAssignWith.facet.c2c) [symbolic = %specific_impl_fn.loc73_6.2 (constants.%specific_impl_fn.f6b)]
// CHECK:STDOUT:     %bound_method.loc73_6.2: <bound method> = bound_method %.loc73_3, %specific_impl_fn.loc73_6.1
// CHECK:STDOUT:     %BitOrAssignWith.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc73_6.2(%.loc73_3, %b.ref)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGenericOp(constants.%T.f45530.1) {
// CHECK:STDOUT:   %T.patt.loc65_19.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc65_19.1 => constants.%T.f45530.1
// CHECK:STDOUT:   %T.as_type.loc65_36.1 => constants.%T.as_type.190bb7.1
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.86a9db.1
// CHECK:STDOUT:   %a.param_patt.loc65_34.2 => constants.%a.param_patt.e33
// CHECK:STDOUT:   %a.patt.loc65_34.2 => constants.%a.patt.e02
// CHECK:STDOUT:   %b.param_patt.loc65_40.2 => constants.%b.param_patt.4cd741.1
// CHECK:STDOUT:   %b.patt.loc65_40.2 => constants.%b.patt.60acf6.1
// CHECK:STDOUT:   %.loc65_48.2 => constants.%.1f0
// CHECK:STDOUT:   %return.param_patt.loc65_48.2 => constants.%return.param_patt.5f6
// CHECK:STDOUT:   %return.patt.loc65_45.2 => constants.%return.patt.96a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGenericAssign(constants.%T.f45530.2) {
// CHECK:STDOUT:   %T.patt.loc71_23.2 => constants.%T.patt.3a0592.2
// CHECK:STDOUT:   %T.loc71_23.1 => constants.%T.f45530.2
// CHECK:STDOUT:   %T.as_type.loc71_47.1 => constants.%T.as_type.190bb7.2
// CHECK:STDOUT:   %ptr.loc71_47.1 => constants.%ptr.360
// CHECK:STDOUT:   %pattern_type.loc71_44 => constants.%pattern_type.ff5
// CHECK:STDOUT:   %a.param_patt.loc71_44.2 => constants.%a.param_patt.f50
// CHECK:STDOUT:   %a.patt.loc71_44.2 => constants.%a.patt.193
// CHECK:STDOUT:   %pattern_type.loc71_51 => constants.%pattern_type.86a9db.2
// CHECK:STDOUT:   %b.param_patt.loc71_51.2 => constants.%b.param_patt.4cd741.2
// CHECK:STDOUT:   %b.patt.loc71_51.2 => constants.%b.patt.60acf6.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGenericOp(constants.%BitOr.facet.423) {
// CHECK:STDOUT:   %T.patt.loc65_19.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc65_19.1 => constants.%BitOr.facet.423
// CHECK:STDOUT:   %T.as_type.loc65_36.1 => constants.%C
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.4a7
// CHECK:STDOUT:   %a.param_patt.loc65_34.2 => constants.%a.param_patt.2a5
// CHECK:STDOUT:   %a.patt.loc65_34.2 => constants.%a.patt.a30
// CHECK:STDOUT:   %b.param_patt.loc65_40.2 => constants.%b.param_patt.45f
// CHECK:STDOUT:   %b.patt.loc65_40.2 => constants.%b.patt.562
// CHECK:STDOUT:   %.loc65_48.2 => constants.%.79c
// CHECK:STDOUT:   %return.param_patt.loc65_48.2 => constants.%return.param_patt.415
// CHECK:STDOUT:   %return.patt.loc65_45.2 => constants.%return.patt.340
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc65 => constants.%complete_type.357
// CHECK:STDOUT:   %BitOrWith.type.loc67_12.2 => constants.%BitOrWith.type.ba6
// CHECK:STDOUT:   %require_complete.loc67 => constants.%complete_type.8f6
// CHECK:STDOUT:   %BitOrWith.assoc_type => constants.%BitOrWith.assoc_type.a18
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.0c4
// CHECK:STDOUT:   %BitOrWith.lookup_impl_witness => constants.%BitOrWith.impl_witness.d4d
// CHECK:STDOUT:   %BitOrWith.facet => constants.%BitOrWith.facet.c79
// CHECK:STDOUT:   %BitOrWith.WithSelf.Op.type => constants.%BitOrWith.WithSelf.Op.type.1aa
// CHECK:STDOUT:   %.loc67_12.2 => constants.%.dcd
// CHECK:STDOUT:   %impl.elem1.loc67_12.2 => constants.%C.as.BitOrWith.impl.Op
// CHECK:STDOUT:   %specific_impl_fn.loc67_12.2 => constants.%C.as.BitOrWith.impl.Op
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGenericAssign(constants.%BitOrAssign.facet.37c) {
// CHECK:STDOUT:   %T.patt.loc71_23.2 => constants.%T.patt.3a0592.2
// CHECK:STDOUT:   %T.loc71_23.1 => constants.%BitOrAssign.facet.37c
// CHECK:STDOUT:   %T.as_type.loc71_47.1 => constants.%C
// CHECK:STDOUT:   %ptr.loc71_47.1 => constants.%ptr.00e
// CHECK:STDOUT:   %pattern_type.loc71_44 => constants.%pattern_type.842
// CHECK:STDOUT:   %a.param_patt.loc71_44.2 => constants.%a.param_patt.4fa
// CHECK:STDOUT:   %a.patt.loc71_44.2 => constants.%a.patt.eba
// CHECK:STDOUT:   %pattern_type.loc71_51 => constants.%pattern_type.4a7
// CHECK:STDOUT:   %b.param_patt.loc71_51.2 => constants.%b.param_patt.45f
// CHECK:STDOUT:   %b.patt.loc71_51.2 => constants.%b.patt.562
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc71_44 => constants.%complete_type.a56
// CHECK:STDOUT:   %require_complete.loc71_51 => constants.%complete_type.357
// CHECK:STDOUT:   %BitOrAssignWith.type.loc73_6.2 => constants.%BitOrAssignWith.type.c39
// CHECK:STDOUT:   %require_complete.loc73 => constants.%complete_type.c5a
// CHECK:STDOUT:   %BitOrAssignWith.assoc_type => constants.%BitOrAssignWith.assoc_type.57a
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.975
// CHECK:STDOUT:   %BitOrAssignWith.lookup_impl_witness => constants.%BitOrAssignWith.impl_witness.fbe
// CHECK:STDOUT:   %BitOrAssignWith.facet => constants.%BitOrAssignWith.facet.f0c
// CHECK:STDOUT:   %BitOrAssignWith.WithSelf.Op.type => constants.%BitOrAssignWith.WithSelf.Op.type.0fd
// CHECK:STDOUT:   %.loc73_6.2 => constants.%.778
// CHECK:STDOUT:   %impl.elem0.loc73_6.2 => constants.%C.as.BitOrAssignWith.impl.Op
// CHECK:STDOUT:   %specific_impl_fn.loc73_6.2 => constants.%C.as.BitOrAssignWith.impl.Op
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGenericOp(constants.%BitOr.facet.6f9) {
// CHECK:STDOUT:   %T.patt.loc65_19.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc65_19.1 => constants.%BitOr.facet.6f9
// CHECK:STDOUT:   %T.as_type.loc65_36.1 => constants.%D
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.331
// CHECK:STDOUT:   %a.param_patt.loc65_34.2 => constants.%a.param_patt.d0e
// CHECK:STDOUT:   %a.patt.loc65_34.2 => constants.%a.patt.ec5
// CHECK:STDOUT:   %b.param_patt.loc65_40.2 => constants.%b.param_patt.110
// CHECK:STDOUT:   %b.patt.loc65_40.2 => constants.%b.patt.14c
// CHECK:STDOUT:   %.loc65_48.2 => constants.%.84d
// CHECK:STDOUT:   %return.param_patt.loc65_48.2 => constants.%return.param_patt.835
// CHECK:STDOUT:   %return.patt.loc65_45.2 => constants.%return.patt.5ef
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc65 => constants.%complete_type.357
// CHECK:STDOUT:   %BitOrWith.type.loc67_12.2 => constants.%BitOrWith.type.d72
// CHECK:STDOUT:   %require_complete.loc67 => constants.%complete_type.ae2
// CHECK:STDOUT:   %BitOrWith.assoc_type => constants.%BitOrWith.assoc_type.35d
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.ae2
// CHECK:STDOUT:   %BitOrWith.lookup_impl_witness => constants.%BitOrWith.impl_witness.736
// CHECK:STDOUT:   %BitOrWith.facet => constants.%BitOrWith.facet.d76
// CHECK:STDOUT:   %BitOrWith.WithSelf.Op.type => constants.%BitOrWith.WithSelf.Op.type.51c
// CHECK:STDOUT:   %.loc67_12.2 => constants.%.571
// CHECK:STDOUT:   %impl.elem1.loc67_12.2 => constants.%D.as.BitOrWith.impl.Op
// CHECK:STDOUT:   %specific_impl_fn.loc67_12.2 => constants.%D.as.BitOrWith.impl.Op
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGenericAssign(constants.%BitOrAssign.facet.6f9) {
// CHECK:STDOUT:   %T.patt.loc71_23.2 => constants.%T.patt.3a0592.2
// CHECK:STDOUT:   %T.loc71_23.1 => constants.%BitOrAssign.facet.6f9
// CHECK:STDOUT:   %T.as_type.loc71_47.1 => constants.%D
// CHECK:STDOUT:   %ptr.loc71_47.1 => constants.%ptr.4a5
// CHECK:STDOUT:   %pattern_type.loc71_44 => constants.%pattern_type.ceb
// CHECK:STDOUT:   %a.param_patt.loc71_44.2 => constants.%a.param_patt.609
// CHECK:STDOUT:   %a.patt.loc71_44.2 => constants.%a.patt.fac
// CHECK:STDOUT:   %pattern_type.loc71_51 => constants.%pattern_type.331
// CHECK:STDOUT:   %b.param_patt.loc71_51.2 => constants.%b.param_patt.110
// CHECK:STDOUT:   %b.patt.loc71_51.2 => constants.%b.patt.14c
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc71_44 => constants.%complete_type.320
// CHECK:STDOUT:   %require_complete.loc71_51 => constants.%complete_type.357
// CHECK:STDOUT:   %BitOrAssignWith.type.loc73_6.2 => constants.%BitOrAssignWith.type.b08
// CHECK:STDOUT:   %require_complete.loc73 => constants.%complete_type.9d8
// CHECK:STDOUT:   %BitOrAssignWith.assoc_type => constants.%BitOrAssignWith.assoc_type.2dc
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.7bb
// CHECK:STDOUT:   %BitOrAssignWith.lookup_impl_witness => constants.%BitOrAssignWith.impl_witness.736
// CHECK:STDOUT:   %BitOrAssignWith.facet => constants.%BitOrAssignWith.facet.fc9
// CHECK:STDOUT:   %BitOrAssignWith.WithSelf.Op.type => constants.%BitOrAssignWith.WithSelf.Op.type.c7e
// CHECK:STDOUT:   %.loc73_6.2 => constants.%.dde
// CHECK:STDOUT:   %impl.elem0.loc73_6.2 => constants.%D.as.BitOrAssignWith.impl.Op
// CHECK:STDOUT:   %specific_impl_fn.loc73_6.2 => constants.%D.as.BitOrAssignWith.impl.Op
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGenericOp(constants.%BitOr.facet.85b) {
// CHECK:STDOUT:   %T.patt.loc65_19.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc65_19.1 => constants.%BitOr.facet.85b
// CHECK:STDOUT:   %T.as_type.loc65_36.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.d94
// CHECK:STDOUT:   %a.param_patt.loc65_34.2 => constants.%a.param_patt.076
// CHECK:STDOUT:   %a.patt.loc65_34.2 => constants.%a.patt.96f
// CHECK:STDOUT:   %b.param_patt.loc65_40.2 => constants.%b.param_patt.1a2
// CHECK:STDOUT:   %b.patt.loc65_40.2 => constants.%b.patt.3ca
// CHECK:STDOUT:   %.loc65_48.2 => constants.%.0b3
// CHECK:STDOUT:   %return.param_patt.loc65_48.2 => constants.%return.param_patt.299
// CHECK:STDOUT:   %return.patt.loc65_45.2 => constants.%return.patt.6fa
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc65 => constants.%complete_type.f8a
// CHECK:STDOUT:   %BitOrWith.type.loc67_12.2 => constants.%BitOrWith.type.06c
// CHECK:STDOUT:   %require_complete.loc67 => constants.%complete_type.7b9
// CHECK:STDOUT:   %BitOrWith.assoc_type => constants.%BitOrWith.assoc_type.e85
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.8e2
// CHECK:STDOUT:   %BitOrWith.lookup_impl_witness => constants.%BitOrWith.impl_witness.58e
// CHECK:STDOUT:   %BitOrWith.facet => constants.%BitOrWith.facet.603
// CHECK:STDOUT:   %BitOrWith.WithSelf.Op.type => constants.%BitOrWith.WithSelf.Op.type.0dc
// CHECK:STDOUT:   %.loc67_12.2 => constants.%.6d9
// CHECK:STDOUT:   %impl.elem1.loc67_12.2 => constants.%Int.as.BitOrWith.impl.Op.78a
// CHECK:STDOUT:   %specific_impl_fn.loc67_12.2 => constants.%Int.as.BitOrWith.impl.Op.specific_fn.18a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGenericAssign(constants.%BitOrAssign.facet.471) {
// CHECK:STDOUT:   %T.patt.loc71_23.2 => constants.%T.patt.3a0592.2
// CHECK:STDOUT:   %T.loc71_23.1 => constants.%BitOrAssign.facet.471
// CHECK:STDOUT:   %T.as_type.loc71_47.1 => constants.%i32
// CHECK:STDOUT:   %ptr.loc71_47.1 => constants.%ptr.a17
// CHECK:STDOUT:   %pattern_type.loc71_44 => constants.%pattern_type.7dc
// CHECK:STDOUT:   %a.param_patt.loc71_44.2 => constants.%a.param_patt.977
// CHECK:STDOUT:   %a.patt.loc71_44.2 => constants.%a.patt.b70
// CHECK:STDOUT:   %pattern_type.loc71_51 => constants.%pattern_type.d94
// CHECK:STDOUT:   %b.param_patt.loc71_51.2 => constants.%b.param_patt.1a2
// CHECK:STDOUT:   %b.patt.loc71_51.2 => constants.%b.patt.3ca
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc71_44 => constants.%complete_type.1b1
// CHECK:STDOUT:   %require_complete.loc71_51 => constants.%complete_type.f8a
// CHECK:STDOUT:   %BitOrAssignWith.type.loc73_6.2 => constants.%BitOrAssignWith.type.8b6
// CHECK:STDOUT:   %require_complete.loc73 => constants.%complete_type.b58
// CHECK:STDOUT:   %BitOrAssignWith.assoc_type => constants.%BitOrAssignWith.assoc_type.f2c
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.774
// CHECK:STDOUT:   %BitOrAssignWith.lookup_impl_witness => constants.%BitOrAssignWith.impl_witness.a9c
// CHECK:STDOUT:   %BitOrAssignWith.facet => constants.%BitOrAssignWith.facet.5cc
// CHECK:STDOUT:   %BitOrAssignWith.WithSelf.Op.type => constants.%BitOrAssignWith.WithSelf.Op.type.d0d
// CHECK:STDOUT:   %.loc73_6.2 => constants.%.0cdb
// CHECK:STDOUT:   %impl.elem0.loc73_6.2 => constants.%Int.as.BitOrAssignWith.impl.Op.e3e6a9.2
// CHECK:STDOUT:   %specific_impl_fn.loc73_6.2 => constants.%Int.as.BitOrAssignWith.impl.Op.specific_fn.0e69c3.1
// CHECK:STDOUT: }
// CHECK:STDOUT:

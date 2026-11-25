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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/overloaded/right_shift.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/overloaded/right_shift.carbon

// This file was generated from binary_op.carbon.tmpl. Run make_tests.sh to regenerate.

package User;

class C {};

impl C as Core.RightShiftWith(C) where .Result = C {
  fn Op(unused self: C, unused other: C) -> C {
    return {};
  }
}
impl C as Core.RightShiftAssignWith(C) {
  fn Op(unused ref self: C, unused other: C) {}
}

fn TestOp(a: C, b: C) -> C {
  //@dump-sem-ir-begin
  return a >> b;
  //@dump-sem-ir-end
}

fn TestAssign(a: C*, b: C) {
  //@dump-sem-ir-begin
  *a >>= b;
  //@dump-sem-ir-end
}

class D {};

impl D as Core.RightShift {
  fn Op(unused self: D, unused other: D) -> D {
    return {};
  }
}
impl D as Core.RightShiftAssign {
  fn Op(unused ref self: D, unused other: D) {}
}

fn TestNamedConstraintOp(a: D, b: D) -> D {
  //@dump-sem-ir-begin
  return a >> b;
  //@dump-sem-ir-end
}

fn TestNamedConstraintAssign(a: D*, b: D) {
  //@dump-sem-ir-begin
  *a >>= b;
  //@dump-sem-ir-end
}

fn TestGenericOp[T: Core.RightShift](a: T, b: T) -> T {
  //@dump-sem-ir-begin
  return a >> b;
  //@dump-sem-ir-end
}

fn TestGenericAssign[T: Core.RightShiftAssign](a: T*, b: T) {
  //@dump-sem-ir-begin
  *a >>= b;
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

// CHECK:STDOUT: --- right_shift.carbon = $right_shift
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Other: type = symbolic_binding Other, 0 [symbolic]
// CHECK:STDOUT:   %RightShiftWith.type.d22: type = facet_type <@RightShiftWith, @RightShiftWith(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.8d4: %RightShiftWith.type.d22 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %RightShiftWith.assoc_type.377: type = assoc_entity_type @RightShiftWith, @RightShiftWith(%Other) [symbolic]
// CHECK:STDOUT:   %RightShiftWith.WithSelf.Op.type.8ac: type = fn_type @RightShiftWith.WithSelf.Op, @RightShiftWith.WithSelf(%Other, %Self.8d4) [symbolic]
// CHECK:STDOUT:   %RightShiftWith.WithSelf.Op.525: %RightShiftWith.WithSelf.Op.type.8ac = struct_value () [symbolic]
// CHECK:STDOUT:   %RightShiftWith.type.2f7: type = facet_type <@RightShiftWith, @RightShiftWith(%C)> [concrete]
// CHECK:STDOUT:   %RightShiftWith.assoc_type.a32: type = assoc_entity_type @RightShiftWith, @RightShiftWith(%C) [concrete]
// CHECK:STDOUT:   %assoc1.2d2: %RightShiftWith.assoc_type.a32 = assoc_entity element1, imports.%Core.import_ref.34d [concrete]
// CHECK:STDOUT:   %RightShiftWith.impl_witness.d46: <witness> = impl_witness @C.as.RightShiftWith.impl.%RightShiftWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.4a7: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %.79c: Core.Form = init_form %C [concrete]
// CHECK:STDOUT:   %return.param_patt.415: %pattern_type.4a7 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.340: %pattern_type.4a7 = return_slot_pattern %return.param_patt.415, %C [concrete]
// CHECK:STDOUT:   %C.as.RightShiftWith.impl.Op.type: type = fn_type @C.as.RightShiftWith.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.RightShiftWith.impl.Op: %C.as.RightShiftWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %RightShiftWith.facet.da5: %RightShiftWith.type.2f7 = facet_value %C, (%RightShiftWith.impl_witness.d46) [concrete]
// CHECK:STDOUT:   %RightShiftWith.WithSelf.Op.type.235: type = fn_type @RightShiftWith.WithSelf.Op, @RightShiftWith.WithSelf(%C, %RightShiftWith.facet.da5) [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.type.bff: type = facet_type <@RightShiftAssignWith, @RightShiftAssignWith(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.e5e: %RightShiftAssignWith.type.bff = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %RightShiftAssignWith.assoc_type.573: type = assoc_entity_type @RightShiftAssignWith, @RightShiftAssignWith(%Other) [symbolic]
// CHECK:STDOUT:   %RightShiftAssignWith.WithSelf.Op.type.6fe: type = fn_type @RightShiftAssignWith.WithSelf.Op, @RightShiftAssignWith.WithSelf(%Other, %Self.e5e) [symbolic]
// CHECK:STDOUT:   %RightShiftAssignWith.WithSelf.Op.c39: %RightShiftAssignWith.WithSelf.Op.type.6fe = struct_value () [symbolic]
// CHECK:STDOUT:   %RightShiftAssignWith.type.967: type = facet_type <@RightShiftAssignWith, @RightShiftAssignWith(%C)> [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.impl_witness.725: <witness> = impl_witness @C.as.RightShiftAssignWith.impl.%RightShiftAssignWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.assoc_type.407: type = assoc_entity_type @RightShiftAssignWith, @RightShiftAssignWith(%C) [concrete]
// CHECK:STDOUT:   %assoc0.f72: %RightShiftAssignWith.assoc_type.407 = assoc_entity element0, imports.%Core.import_ref.e99 [concrete]
// CHECK:STDOUT:   %C.as.RightShiftAssignWith.impl.Op.type: type = fn_type @C.as.RightShiftAssignWith.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.RightShiftAssignWith.impl.Op: %C.as.RightShiftAssignWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.facet.fb9: %RightShiftAssignWith.type.967 = facet_value %C, (%RightShiftAssignWith.impl_witness.725) [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.WithSelf.Op.type.961: type = fn_type @RightShiftAssignWith.WithSelf.Op, @RightShiftAssignWith.WithSelf(%C, %RightShiftAssignWith.facet.fb9) [concrete]
// CHECK:STDOUT:   %a.param_patt.2a5: %pattern_type.4a7 = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt.a30: %pattern_type.4a7 = wrapper_binding_pattern a, %a.param_patt.2a5 [concrete]
// CHECK:STDOUT:   %b.param_patt.45f: %pattern_type.4a7 = value_param_pattern [concrete]
// CHECK:STDOUT:   %b.patt.562: %pattern_type.4a7 = wrapper_binding_pattern b, %b.param_patt.45f [concrete]
// CHECK:STDOUT:   %assoc1.354: %RightShiftWith.assoc_type.377 = assoc_entity element1, imports.%Core.import_ref.011 [symbolic]
// CHECK:STDOUT:   %.300: type = fn_type_with_self_type %RightShiftWith.WithSelf.Op.type.235, %RightShiftWith.facet.da5 [concrete]
// CHECK:STDOUT:   %ptr.00e: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.842: type = pattern_type %ptr.00e [concrete]
// CHECK:STDOUT:   %a.param_patt.4fa: %pattern_type.842 = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt.eba: %pattern_type.842 = wrapper_binding_pattern a, %a.param_patt.4fa [concrete]
// CHECK:STDOUT:   %assoc0.9d8: %RightShiftAssignWith.assoc_type.573 = assoc_entity element0, imports.%Core.import_ref.495 [symbolic]
// CHECK:STDOUT:   %.e4d: type = fn_type_with_self_type %RightShiftAssignWith.WithSelf.Op.type.961, %RightShiftAssignWith.facet.fb9 [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %RightShift.type: type = facet_type <@RightShift> [concrete]
// CHECK:STDOUT:   %RightShiftWith.impl_witness.736: <witness> = impl_witness @D.as.RightShiftWith.impl.%RightShiftWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.331: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %.84d: Core.Form = init_form %D [concrete]
// CHECK:STDOUT:   %return.param_patt.835: %pattern_type.331 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.5ef: %pattern_type.331 = return_slot_pattern %return.param_patt.835, %D [concrete]
// CHECK:STDOUT:   %D.as.RightShiftWith.impl.Op.type: type = fn_type @D.as.RightShiftWith.impl.Op [concrete]
// CHECK:STDOUT:   %D.as.RightShiftWith.impl.Op: %D.as.RightShiftWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %RightShiftWith.type.7ba: type = facet_type <@RightShiftWith, @RightShiftWith(%D)> [concrete]
// CHECK:STDOUT:   %RightShiftWith.facet.711: %RightShiftWith.type.7ba = facet_value %D, (%RightShiftWith.impl_witness.736) [concrete]
// CHECK:STDOUT:   %RightShiftWith.assoc_type.080: type = assoc_entity_type @RightShiftWith, @RightShiftWith(%D) [concrete]
// CHECK:STDOUT:   %RightShiftWith.WithSelf.Op.type.6e0: type = fn_type @RightShiftWith.WithSelf.Op, @RightShiftWith.WithSelf(%D, %RightShiftWith.facet.711) [concrete]
// CHECK:STDOUT:   %assoc1.6a6: %RightShiftWith.assoc_type.080 = assoc_entity element1, imports.%Core.import_ref.34d [concrete]
// CHECK:STDOUT:   %RightShiftAssign.type: type = facet_type <@RightShiftAssign> [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.impl_witness.736: <witness> = impl_witness @D.as.RightShiftAssignWith.impl.%RightShiftAssignWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %D.as.RightShiftAssignWith.impl.Op.type: type = fn_type @D.as.RightShiftAssignWith.impl.Op [concrete]
// CHECK:STDOUT:   %D.as.RightShiftAssignWith.impl.Op: %D.as.RightShiftAssignWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.type.e5f: type = facet_type <@RightShiftAssignWith, @RightShiftAssignWith(%D)> [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.facet.649: %RightShiftAssignWith.type.e5f = facet_value %D, (%RightShiftAssignWith.impl_witness.736) [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.WithSelf.Op.type.0fe: type = fn_type @RightShiftAssignWith.WithSelf.Op, @RightShiftAssignWith.WithSelf(%D, %RightShiftAssignWith.facet.649) [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.assoc_type.845: type = assoc_entity_type @RightShiftAssignWith, @RightShiftAssignWith(%D) [concrete]
// CHECK:STDOUT:   %assoc0.517: %RightShiftAssignWith.assoc_type.845 = assoc_entity element0, imports.%Core.import_ref.e99 [concrete]
// CHECK:STDOUT:   %a.param_patt.d0e: %pattern_type.331 = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt.ec5: %pattern_type.331 = wrapper_binding_pattern a, %a.param_patt.d0e [concrete]
// CHECK:STDOUT:   %b.param_patt.110: %pattern_type.331 = value_param_pattern [concrete]
// CHECK:STDOUT:   %b.patt.14c: %pattern_type.331 = wrapper_binding_pattern b, %b.param_patt.110 [concrete]
// CHECK:STDOUT:   %.027: type = fn_type_with_self_type %RightShiftWith.WithSelf.Op.type.6e0, %RightShiftWith.facet.711 [concrete]
// CHECK:STDOUT:   %ptr.4a5: type = ptr_type %D [concrete]
// CHECK:STDOUT:   %pattern_type.ceb: type = pattern_type %ptr.4a5 [concrete]
// CHECK:STDOUT:   %a.param_patt.609: %pattern_type.ceb = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt.fac: %pattern_type.ceb = wrapper_binding_pattern a, %a.param_patt.609 [concrete]
// CHECK:STDOUT:   %.585: type = fn_type_with_self_type %RightShiftAssignWith.WithSelf.Op.type.0fe, %RightShiftAssignWith.facet.649 [concrete]
// CHECK:STDOUT:   %pattern_type.9a587c.2: type = pattern_type %RightShift.type [concrete]
// CHECK:STDOUT:   %T.patt.3a0592.1: %pattern_type.9a587c.2 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.f45530.1: %RightShift.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.190bb7.1: type = facet_access_type %T.f45530.1 [symbolic]
// CHECK:STDOUT:   %pattern_type.86a9db.1: type = pattern_type %T.as_type.190bb7.1 [symbolic]
// CHECK:STDOUT:   %a.param_patt.e33: %pattern_type.86a9db.1 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %a.patt.e02: %pattern_type.86a9db.1 = wrapper_binding_pattern a, %a.param_patt.e33 [symbolic]
// CHECK:STDOUT:   %b.param_patt.4cd741.1: %pattern_type.86a9db.1 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %b.patt.60acf6.1: %pattern_type.86a9db.1 = wrapper_binding_pattern b, %b.param_patt.4cd741.1 [symbolic]
// CHECK:STDOUT:   %.1f0: Core.Form = init_form %T.as_type.190bb7.1 [symbolic]
// CHECK:STDOUT:   %return.param_patt.5f6: %pattern_type.86a9db.1 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.96a: %pattern_type.86a9db.1 = return_slot_pattern %return.param_patt.5f6, %T.as_type.190bb7.1 [symbolic]
// CHECK:STDOUT:   %RightShiftWith.type.678ef0.2: type = facet_type <@RightShiftWith, @RightShiftWith(%T.as_type.190bb7.1)> [symbolic]
// CHECK:STDOUT:   %RightShiftWith.assoc_type.2a5500.2: type = assoc_entity_type @RightShiftWith, @RightShiftWith(%T.as_type.190bb7.1) [symbolic]
// CHECK:STDOUT:   %assoc1.e9e1f7.2: %RightShiftWith.assoc_type.2a5500.2 = assoc_entity element1, imports.%Core.import_ref.34d [symbolic]
// CHECK:STDOUT:   %require_complete.52a: <witness> = require_complete_type %RightShiftWith.type.678ef0.2 [symbolic]
// CHECK:STDOUT:   %RightShiftWith.lookup_impl_witness.8d6d2c.2: <witness> = lookup_impl_witness %T.f45530.1, @RightShiftWith, @RightShiftWith(%T.as_type.190bb7.1) [symbolic]
// CHECK:STDOUT:   %RightShiftWith.facet.af5: %RightShiftWith.type.678ef0.2 = facet_value %T.as_type.190bb7.1, (%RightShiftWith.lookup_impl_witness.8d6d2c.2) [symbolic]
// CHECK:STDOUT:   %RightShiftWith.WithSelf.Op.type.3ec: type = fn_type @RightShiftWith.WithSelf.Op, @RightShiftWith.WithSelf(%T.as_type.190bb7.1, %RightShiftWith.facet.af5) [symbolic]
// CHECK:STDOUT:   %.209: type = fn_type_with_self_type %RightShiftWith.WithSelf.Op.type.3ec, %RightShiftWith.facet.af5 [symbolic]
// CHECK:STDOUT:   %impl.elem1.290: %.209 = impl_witness_access %RightShiftWith.lookup_impl_witness.8d6d2c.2, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.717: <specific function> = specific_impl_function %impl.elem1.290, @RightShiftWith.WithSelf.Op(%T.as_type.190bb7.1, %RightShiftWith.facet.af5) [symbolic]
// CHECK:STDOUT:   %pattern_type.9a587c.3: type = pattern_type %RightShiftAssign.type [concrete]
// CHECK:STDOUT:   %T.patt.3a0592.2: %pattern_type.9a587c.3 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.f45530.2: %RightShiftAssign.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.190bb7.2: type = facet_access_type %T.f45530.2 [symbolic]
// CHECK:STDOUT:   %ptr.360: type = ptr_type %T.as_type.190bb7.2 [symbolic]
// CHECK:STDOUT:   %pattern_type.ff5: type = pattern_type %ptr.360 [symbolic]
// CHECK:STDOUT:   %a.param_patt.f50: %pattern_type.ff5 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %a.patt.193: %pattern_type.ff5 = wrapper_binding_pattern a, %a.param_patt.f50 [symbolic]
// CHECK:STDOUT:   %pattern_type.86a9db.2: type = pattern_type %T.as_type.190bb7.2 [symbolic]
// CHECK:STDOUT:   %b.param_patt.4cd741.2: %pattern_type.86a9db.2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %b.patt.60acf6.2: %pattern_type.86a9db.2 = wrapper_binding_pattern b, %b.param_patt.4cd741.2 [symbolic]
// CHECK:STDOUT:   %RightShiftAssignWith.type.56546a.2: type = facet_type <@RightShiftAssignWith, @RightShiftAssignWith(%T.as_type.190bb7.2)> [symbolic]
// CHECK:STDOUT:   %RightShiftAssignWith.assoc_type.7f5f89.2: type = assoc_entity_type @RightShiftAssignWith, @RightShiftAssignWith(%T.as_type.190bb7.2) [symbolic]
// CHECK:STDOUT:   %assoc0.b4b7ba.2: %RightShiftAssignWith.assoc_type.7f5f89.2 = assoc_entity element0, imports.%Core.import_ref.e99 [symbolic]
// CHECK:STDOUT:   %require_complete.1e3a95.2: <witness> = require_complete_type %RightShiftAssignWith.type.56546a.2 [symbolic]
// CHECK:STDOUT:   %RightShiftAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.f45530.2, @RightShiftAssignWith, @RightShiftAssignWith(%T.as_type.190bb7.2) [symbolic]
// CHECK:STDOUT:   %RightShiftAssignWith.facet.6e0: %RightShiftAssignWith.type.56546a.2 = facet_value %T.as_type.190bb7.2, (%RightShiftAssignWith.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %RightShiftAssignWith.WithSelf.Op.type.ffc: type = fn_type @RightShiftAssignWith.WithSelf.Op, @RightShiftAssignWith.WithSelf(%T.as_type.190bb7.2, %RightShiftAssignWith.facet.6e0) [symbolic]
// CHECK:STDOUT:   %.326: type = fn_type_with_self_type %RightShiftAssignWith.WithSelf.Op.type.ffc, %RightShiftAssignWith.facet.6e0 [symbolic]
// CHECK:STDOUT:   %impl.elem0.bb3: %.326 = impl_witness_access %RightShiftAssignWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.086: <specific function> = specific_impl_function %impl.elem0.bb3, @RightShiftAssignWith.WithSelf.Op(%T.as_type.190bb7.2, %RightShiftAssignWith.facet.6e0) [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %Int.d4bbe0.1: type = class_type @Int, @Int(%N) [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %ptr.a17: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %ptr.a17 [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %RightShift.facet.f8f: %RightShift.type = facet_value %C, (%RightShiftWith.impl_witness.d46) [concrete]
// CHECK:STDOUT:   %RightShiftAssign.facet.a10: %RightShiftAssign.type = facet_value %C, (%RightShiftAssignWith.impl_witness.725) [concrete]
// CHECK:STDOUT:   %RightShift.facet.6f9: %RightShift.type = facet_value %D, (%RightShiftWith.impl_witness.736) [concrete]
// CHECK:STDOUT:   %RightShiftAssign.facet.6f9: %RightShiftAssign.type = facet_value %D, (%RightShiftAssignWith.impl_witness.736) [concrete]
// CHECK:STDOUT:   %Int.as.RightShiftWith.impl.Op.type.ccd: type = fn_type @Int.as.RightShiftWith.impl.Op.1, @Int.as.RightShiftWith.impl.796(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.RightShiftWith.impl.Op.d3c: %Int.as.RightShiftWith.impl.Op.type.ccd = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.81c7d0.1: type = facet_type <@ImplicitAs, @ImplicitAs(%Int.d4bbe0.1)> [symbolic]
// CHECK:STDOUT:   %U.f16: %ImplicitAs.type.81c7d0.1 = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %RightShiftWith.type.efe: type = facet_type <@RightShiftWith, @RightShiftWith(%i32)> [concrete]
// CHECK:STDOUT:   %RightShiftWith.impl_witness.77d: <witness> = impl_witness imports.%RightShiftWith.impl_witness_table.20d, @Int.as.RightShiftWith.impl.796(%int_32) [concrete]
// CHECK:STDOUT:   %RightShiftWith.assoc_type.415: type = assoc_entity_type @RightShiftWith, @RightShiftWith(%i32) [concrete]
// CHECK:STDOUT:   %assoc1.ec7: %RightShiftWith.assoc_type.415 = assoc_entity element1, imports.%Core.import_ref.34d [concrete]
// CHECK:STDOUT:   %complete_type.310: <witness> = complete_type_witness %RightShiftWith.type.efe [concrete]
// CHECK:STDOUT:   %Int.as.RightShiftWith.impl.Op.type.df7: type = fn_type @Int.as.RightShiftWith.impl.Op.1, @Int.as.RightShiftWith.impl.796(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.RightShiftWith.impl.Op.d9f: %Int.as.RightShiftWith.impl.Op.type.df7 = struct_value () [concrete]
// CHECK:STDOUT:   %RightShift.facet.c40: %RightShift.type = facet_value %i32, (%RightShiftWith.impl_witness.77d) [concrete]
// CHECK:STDOUT:   %a.param_patt.076: %pattern_type.d94 = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt.96f: %pattern_type.d94 = wrapper_binding_pattern a, %a.param_patt.076 [concrete]
// CHECK:STDOUT:   %b.param_patt.1a2: %pattern_type.d94 = value_param_pattern [concrete]
// CHECK:STDOUT:   %b.patt.3ca: %pattern_type.d94 = wrapper_binding_pattern b, %b.param_patt.1a2 [concrete]
// CHECK:STDOUT:   %.0b3e: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.299: %pattern_type.d94 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.6fa: %pattern_type.d94 = return_slot_pattern %return.param_patt.299, %i32 [concrete]
// CHECK:STDOUT:   %Int.as.RightShiftAssignWith.impl.Op.type.484b98.1: type = fn_type @Int.as.RightShiftAssignWith.impl.Op.1, @Int.as.RightShiftAssignWith.impl(%N, %U.f16) [symbolic]
// CHECK:STDOUT:   %Int.as.RightShiftAssignWith.impl.Op.78f28b.1: %Int.as.RightShiftAssignWith.impl.Op.type.484b98.1 = struct_value () [symbolic]
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
// CHECK:STDOUT:   %RightShiftAssignWith.type.db2: type = facet_type <@RightShiftAssignWith, @RightShiftAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.impl_witness.67f: <witness> = impl_witness imports.%RightShiftAssignWith.impl_witness_table, @Int.as.RightShiftAssignWith.impl(%int_32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %RightShiftAssign.facet.706: %RightShiftAssign.type = facet_value %i32, (%RightShiftAssignWith.impl_witness.67f) [concrete]
// CHECK:STDOUT:   %a.param_patt.977: %pattern_type.7dc = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt.b70: %pattern_type.7dc = wrapper_binding_pattern a, %a.param_patt.977 [concrete]
// CHECK:STDOUT:   %complete_type.4d6: <witness> = complete_type_witness %RightShiftWith.type.2f7 [concrete]
// CHECK:STDOUT:   %complete_type.a56: <witness> = complete_type_witness %ptr.00e [concrete]
// CHECK:STDOUT:   %complete_type.315: <witness> = complete_type_witness %RightShiftAssignWith.type.967 [concrete]
// CHECK:STDOUT:   %complete_type.915: <witness> = complete_type_witness %RightShiftWith.type.7ba [concrete]
// CHECK:STDOUT:   %complete_type.320: <witness> = complete_type_witness %ptr.4a5 [concrete]
// CHECK:STDOUT:   %complete_type.c20: <witness> = complete_type_witness %RightShiftAssignWith.type.e5f [concrete]
// CHECK:STDOUT:   %RightShiftWith.facet.449: %RightShiftWith.type.efe = facet_value %i32, (%RightShiftWith.impl_witness.77d) [concrete]
// CHECK:STDOUT:   %RightShiftWith.WithSelf.Op.type.61f: type = fn_type @RightShiftWith.WithSelf.Op, @RightShiftWith.WithSelf(%i32, %RightShiftWith.facet.449) [concrete]
// CHECK:STDOUT:   %.a2f: type = fn_type_with_self_type %RightShiftWith.WithSelf.Op.type.61f, %RightShiftWith.facet.449 [concrete]
// CHECK:STDOUT:   %Int.as.RightShiftWith.impl.Op.specific_fn.66a: <specific function> = specific_function %Int.as.RightShiftWith.impl.Op.d9f, @Int.as.RightShiftWith.impl.Op.1(%int_32) [concrete]
// CHECK:STDOUT:   %complete_type.1b1: <witness> = complete_type_witness %ptr.a17 [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.assoc_type.0da: type = assoc_entity_type @RightShiftAssignWith, @RightShiftAssignWith(%i32) [concrete]
// CHECK:STDOUT:   %assoc0.466: %RightShiftAssignWith.assoc_type.0da = assoc_entity element0, imports.%Core.import_ref.e99 [concrete]
// CHECK:STDOUT:   %complete_type.70c: <witness> = complete_type_witness %RightShiftAssignWith.type.db2 [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.facet.17c: %RightShiftAssignWith.type.db2 = facet_value %i32, (%RightShiftAssignWith.impl_witness.67f) [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.WithSelf.Op.type.0b8: type = fn_type @RightShiftAssignWith.WithSelf.Op, @RightShiftAssignWith.WithSelf(%i32, %RightShiftAssignWith.facet.17c) [concrete]
// CHECK:STDOUT:   %.6c5: type = fn_type_with_self_type %RightShiftAssignWith.WithSelf.Op.type.0b8, %RightShiftAssignWith.facet.17c [concrete]
// CHECK:STDOUT:   %Int.as.RightShiftAssignWith.impl.Op.type.c37433.2: type = fn_type @Int.as.RightShiftAssignWith.impl.Op.1, @Int.as.RightShiftAssignWith.impl(%int_32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Int.as.RightShiftAssignWith.impl.Op.2676d3.2: %Int.as.RightShiftAssignWith.impl.Op.type.c37433.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.RightShiftAssignWith.impl.Op.specific_fn.16bcbd.1: <specific function> = specific_function %Int.as.RightShiftAssignWith.impl.Op.2676d3.2, @Int.as.RightShiftAssignWith.impl.Op.1(%int_32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.f42: @RightShiftWith.WithSelf.%RightShiftWith.assoc_type (%RightShiftWith.assoc_type.377) = import_ref Core//prelude/operators/bitwise, loc{{\d+_\d+}}, loaded [symbolic = @RightShiftWith.WithSelf.%assoc1 (constants.%assoc1.354)]
// CHECK:STDOUT:   %Core.import_ref.34d: @RightShiftWith.WithSelf.%RightShiftWith.WithSelf.Op.type (%RightShiftWith.WithSelf.Op.type.8ac) = import_ref Core//prelude/operators/bitwise, loc{{\d+_\d+}}, loaded [symbolic = @RightShiftWith.WithSelf.%RightShiftWith.WithSelf.Op (constants.%RightShiftWith.WithSelf.Op.525)]
// CHECK:STDOUT:   %Core.import_ref.fdf: @RightShiftAssignWith.WithSelf.%RightShiftAssignWith.assoc_type (%RightShiftAssignWith.assoc_type.573) = import_ref Core//prelude/operators/bitwise, loc{{\d+_\d+}}, loaded [symbolic = @RightShiftAssignWith.WithSelf.%assoc0 (constants.%assoc0.9d8)]
// CHECK:STDOUT:   %Core.import_ref.e99: @RightShiftAssignWith.WithSelf.%RightShiftAssignWith.WithSelf.Op.type (%RightShiftAssignWith.WithSelf.Op.type.6fe) = import_ref Core//prelude/operators/bitwise, loc{{\d+_\d+}}, loaded [symbolic = @RightShiftAssignWith.WithSelf.%RightShiftAssignWith.WithSelf.Op (constants.%RightShiftAssignWith.WithSelf.Op.c39)]
// CHECK:STDOUT:   %Core.import_ref.011 = import_ref Core//prelude/operators/bitwise, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.495 = import_ref Core//prelude/operators/bitwise, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.ebd: type = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.RightShiftWith.impl.796.%Int (constants.%Int.d4bbe0.1)]
// CHECK:STDOUT:   %Core.import_ref.86e: @Int.as.RightShiftWith.impl.796.%Int.as.RightShiftWith.impl.Op.type (%Int.as.RightShiftWith.impl.Op.type.ccd) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.RightShiftWith.impl.796.%Int.as.RightShiftWith.impl.Op (constants.%Int.as.RightShiftWith.impl.Op.d3c)]
// CHECK:STDOUT:   %RightShiftWith.impl_witness_table.20d = impl_witness_table (%Core.import_ref.ebd, %Core.import_ref.86e), @Int.as.RightShiftWith.impl.796 [concrete]
// CHECK:STDOUT:   %Core.import_ref.6bab: @Int.as.RightShiftAssignWith.impl.%Int.as.RightShiftAssignWith.impl.Op.type.2 (%Int.as.RightShiftAssignWith.impl.Op.type.484b98.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.RightShiftAssignWith.impl.%Int.as.RightShiftAssignWith.impl.Op.2 (constants.%Int.as.RightShiftAssignWith.impl.Op.78f28b.1)]
// CHECK:STDOUT:   %RightShiftAssignWith.impl_witness_table = impl_witness_table (%Core.import_ref.6bab), @Int.as.RightShiftAssignWith.impl [concrete]
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
// CHECK:STDOUT:   %impl.elem1: %.300 = impl_witness_access constants.%RightShiftWith.impl_witness.d46, element1 [concrete = constants.%C.as.RightShiftWith.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.ref, %impl.elem1
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.as.RightShiftWith.impl.Op.call: init %C to %.loc30_26.1 = call %bound_method(%a.ref, %b.ref)
// CHECK:STDOUT:   return %C.as.RightShiftWith.impl.Op.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestAssign(%a.param: %ptr.00e, %b.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %ptr.00e = name_ref a, %a
// CHECK:STDOUT:   %.loc38: ref %C = deref %a.ref
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0: %.e4d = impl_witness_access constants.%RightShiftAssignWith.impl_witness.725, element0 [concrete = constants.%C.as.RightShiftAssignWith.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc38, %impl.elem0
// CHECK:STDOUT:   %C.as.RightShiftAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method(%.loc38, %b.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestNamedConstraintOp(%a.param: %D, %b.param: %D) -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %D = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %D = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1: %.027 = impl_witness_access constants.%RightShiftWith.impl_witness.736, element1 [concrete = constants.%D.as.RightShiftWith.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.ref, %impl.elem1
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %D.as.RightShiftWith.impl.Op.call: init %D to %.loc53_41.1 = call %bound_method(%a.ref, %b.ref)
// CHECK:STDOUT:   return %D.as.RightShiftWith.impl.Op.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestNamedConstraintAssign(%a.param: %ptr.4a5, %b.param: %D) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %ptr.4a5 = name_ref a, %a
// CHECK:STDOUT:   %.loc61: ref %D = deref %a.ref
// CHECK:STDOUT:   %b.ref: %D = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0: %.585 = impl_witness_access constants.%RightShiftAssignWith.impl_witness.736, element0 [concrete = constants.%D.as.RightShiftAssignWith.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc61, %impl.elem0
// CHECK:STDOUT:   %D.as.RightShiftAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method(%.loc61, %b.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestGenericOp(%T.loc65_19.2: %RightShift.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %RightShiftWith.type.loc67_12.2: type = facet_type <@RightShiftWith, @RightShiftWith(%T.as_type.loc65_41.1)> [symbolic = %RightShiftWith.type.loc67_12.2 (constants.%RightShiftWith.type.678ef0.2)]
// CHECK:STDOUT:   %require_complete.loc67: <witness> = require_complete_type %RightShiftWith.type.loc67_12.2 [symbolic = %require_complete.loc67 (constants.%require_complete.52a)]
// CHECK:STDOUT:   %RightShiftWith.assoc_type: type = assoc_entity_type @RightShiftWith, @RightShiftWith(%T.as_type.loc65_41.1) [symbolic = %RightShiftWith.assoc_type (constants.%RightShiftWith.assoc_type.2a5500.2)]
// CHECK:STDOUT:   %assoc1: @TestGenericOp.%RightShiftWith.assoc_type (%RightShiftWith.assoc_type.2a5500.2) = assoc_entity element1, imports.%Core.import_ref.34d [symbolic = %assoc1 (constants.%assoc1.e9e1f7.2)]
// CHECK:STDOUT:   %RightShiftWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc65_19.1, @RightShiftWith, @RightShiftWith(%T.as_type.loc65_41.1) [symbolic = %RightShiftWith.lookup_impl_witness (constants.%RightShiftWith.lookup_impl_witness.8d6d2c.2)]
// CHECK:STDOUT:   %RightShiftWith.facet: @TestGenericOp.%RightShiftWith.type.loc67_12.2 (%RightShiftWith.type.678ef0.2) = facet_value %T.as_type.loc65_41.1, (%RightShiftWith.lookup_impl_witness) [symbolic = %RightShiftWith.facet (constants.%RightShiftWith.facet.af5)]
// CHECK:STDOUT:   %RightShiftWith.WithSelf.Op.type: type = fn_type @RightShiftWith.WithSelf.Op, @RightShiftWith.WithSelf(%T.as_type.loc65_41.1, %RightShiftWith.facet) [symbolic = %RightShiftWith.WithSelf.Op.type (constants.%RightShiftWith.WithSelf.Op.type.3ec)]
// CHECK:STDOUT:   %.loc67_12.2: type = fn_type_with_self_type %RightShiftWith.WithSelf.Op.type, %RightShiftWith.facet [symbolic = %.loc67_12.2 (constants.%.209)]
// CHECK:STDOUT:   %impl.elem1.loc67_12.2: @TestGenericOp.%.loc67_12.2 (%.209) = impl_witness_access %RightShiftWith.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc67_12.2 (constants.%impl.elem1.290)]
// CHECK:STDOUT:   %specific_impl_fn.loc67_12.2: <specific function> = specific_impl_function %impl.elem1.loc67_12.2, @RightShiftWith.WithSelf.Op(%T.as_type.loc65_41.1, %RightShiftWith.facet) [symbolic = %specific_impl_fn.loc67_12.2 (constants.%specific_impl_fn.717)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @TestGenericOp.%T.as_type.loc65_41.1 (%T.as_type.190bb7.1), %b.param: @TestGenericOp.%T.as_type.loc65_41.1 (%T.as_type.190bb7.1)) -> out %return.param: @TestGenericOp.%T.as_type.loc65_41.1 (%T.as_type.190bb7.1) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %a.ref: @TestGenericOp.%T.as_type.loc65_41.1 (%T.as_type.190bb7.1) = name_ref a, %a
// CHECK:STDOUT:     %b.ref: @TestGenericOp.%T.as_type.loc65_41.1 (%T.as_type.190bb7.1) = name_ref b, %b
// CHECK:STDOUT:     %RightShiftWith.type.loc67_12.1: type = facet_type <@RightShiftWith, @RightShiftWith(constants.%T.as_type.190bb7.1)> [symbolic = %RightShiftWith.type.loc67_12.2 (constants.%RightShiftWith.type.678ef0.2)]
// CHECK:STDOUT:     %.loc67_12.1: @TestGenericOp.%RightShiftWith.assoc_type (%RightShiftWith.assoc_type.2a5500.2) = specific_constant imports.%Core.import_ref.f42, @RightShiftWith.WithSelf(constants.%T.as_type.190bb7.1, constants.%Self.8d4) [symbolic = %assoc1 (constants.%assoc1.e9e1f7.2)]
// CHECK:STDOUT:     %Op.ref: @TestGenericOp.%RightShiftWith.assoc_type (%RightShiftWith.assoc_type.2a5500.2) = name_ref Op, %.loc67_12.1 [symbolic = %assoc1 (constants.%assoc1.e9e1f7.2)]
// CHECK:STDOUT:     %impl.elem1.loc67_12.1: @TestGenericOp.%.loc67_12.2 (%.209) = impl_witness_access constants.%RightShiftWith.lookup_impl_witness.8d6d2c.2, element1 [symbolic = %impl.elem1.loc67_12.2 (constants.%impl.elem1.290)]
// CHECK:STDOUT:     %bound_method.loc67_12.1: <bound method> = bound_method %a.ref, %impl.elem1.loc67_12.1
// CHECK:STDOUT:     %specific_impl_fn.loc67_12.1: <specific function> = specific_impl_function %impl.elem1.loc67_12.1, @RightShiftWith.WithSelf.Op(constants.%T.as_type.190bb7.1, constants.%RightShiftWith.facet.af5) [symbolic = %specific_impl_fn.loc67_12.2 (constants.%specific_impl_fn.717)]
// CHECK:STDOUT:     %bound_method.loc67_12.2: <bound method> = bound_method %a.ref, %specific_impl_fn.loc67_12.1
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %RightShiftWith.WithSelf.Op.call: init @TestGenericOp.%T.as_type.loc65_41.1 (%T.as_type.190bb7.1) to %.loc65_53.1 = call %bound_method.loc67_12.2(%a.ref, %b.ref)
// CHECK:STDOUT:     return %RightShiftWith.WithSelf.Op.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestGenericAssign(%T.loc71_23.2: %RightShiftAssign.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %RightShiftAssignWith.type.loc73_6.2: type = facet_type <@RightShiftAssignWith, @RightShiftAssignWith(%T.as_type.loc71_52.1)> [symbolic = %RightShiftAssignWith.type.loc73_6.2 (constants.%RightShiftAssignWith.type.56546a.2)]
// CHECK:STDOUT:   %require_complete.loc73: <witness> = require_complete_type %RightShiftAssignWith.type.loc73_6.2 [symbolic = %require_complete.loc73 (constants.%require_complete.1e3a95.2)]
// CHECK:STDOUT:   %RightShiftAssignWith.assoc_type: type = assoc_entity_type @RightShiftAssignWith, @RightShiftAssignWith(%T.as_type.loc71_52.1) [symbolic = %RightShiftAssignWith.assoc_type (constants.%RightShiftAssignWith.assoc_type.7f5f89.2)]
// CHECK:STDOUT:   %assoc0: @TestGenericAssign.%RightShiftAssignWith.assoc_type (%RightShiftAssignWith.assoc_type.7f5f89.2) = assoc_entity element0, imports.%Core.import_ref.e99 [symbolic = %assoc0 (constants.%assoc0.b4b7ba.2)]
// CHECK:STDOUT:   %RightShiftAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc71_23.1, @RightShiftAssignWith, @RightShiftAssignWith(%T.as_type.loc71_52.1) [symbolic = %RightShiftAssignWith.lookup_impl_witness (constants.%RightShiftAssignWith.lookup_impl_witness)]
// CHECK:STDOUT:   %RightShiftAssignWith.facet: @TestGenericAssign.%RightShiftAssignWith.type.loc73_6.2 (%RightShiftAssignWith.type.56546a.2) = facet_value %T.as_type.loc71_52.1, (%RightShiftAssignWith.lookup_impl_witness) [symbolic = %RightShiftAssignWith.facet (constants.%RightShiftAssignWith.facet.6e0)]
// CHECK:STDOUT:   %RightShiftAssignWith.WithSelf.Op.type: type = fn_type @RightShiftAssignWith.WithSelf.Op, @RightShiftAssignWith.WithSelf(%T.as_type.loc71_52.1, %RightShiftAssignWith.facet) [symbolic = %RightShiftAssignWith.WithSelf.Op.type (constants.%RightShiftAssignWith.WithSelf.Op.type.ffc)]
// CHECK:STDOUT:   %.loc73_6.2: type = fn_type_with_self_type %RightShiftAssignWith.WithSelf.Op.type, %RightShiftAssignWith.facet [symbolic = %.loc73_6.2 (constants.%.326)]
// CHECK:STDOUT:   %impl.elem0.loc73_6.2: @TestGenericAssign.%.loc73_6.2 (%.326) = impl_witness_access %RightShiftAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc73_6.2 (constants.%impl.elem0.bb3)]
// CHECK:STDOUT:   %specific_impl_fn.loc73_6.2: <specific function> = specific_impl_function %impl.elem0.loc73_6.2, @RightShiftAssignWith.WithSelf.Op(%T.as_type.loc71_52.1, %RightShiftAssignWith.facet) [symbolic = %specific_impl_fn.loc73_6.2 (constants.%specific_impl_fn.086)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @TestGenericAssign.%ptr.loc71_52.1 (%ptr.360), %b.param: @TestGenericAssign.%T.as_type.loc71_52.1 (%T.as_type.190bb7.2)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %a.ref: @TestGenericAssign.%ptr.loc71_52.1 (%ptr.360) = name_ref a, %a
// CHECK:STDOUT:     %.loc73_3: ref @TestGenericAssign.%T.as_type.loc71_52.1 (%T.as_type.190bb7.2) = deref %a.ref
// CHECK:STDOUT:     %b.ref: @TestGenericAssign.%T.as_type.loc71_52.1 (%T.as_type.190bb7.2) = name_ref b, %b
// CHECK:STDOUT:     %RightShiftAssignWith.type.loc73_6.1: type = facet_type <@RightShiftAssignWith, @RightShiftAssignWith(constants.%T.as_type.190bb7.2)> [symbolic = %RightShiftAssignWith.type.loc73_6.2 (constants.%RightShiftAssignWith.type.56546a.2)]
// CHECK:STDOUT:     %.loc73_6.1: @TestGenericAssign.%RightShiftAssignWith.assoc_type (%RightShiftAssignWith.assoc_type.7f5f89.2) = specific_constant imports.%Core.import_ref.fdf, @RightShiftAssignWith.WithSelf(constants.%T.as_type.190bb7.2, constants.%Self.e5e) [symbolic = %assoc0 (constants.%assoc0.b4b7ba.2)]
// CHECK:STDOUT:     %Op.ref: @TestGenericAssign.%RightShiftAssignWith.assoc_type (%RightShiftAssignWith.assoc_type.7f5f89.2) = name_ref Op, %.loc73_6.1 [symbolic = %assoc0 (constants.%assoc0.b4b7ba.2)]
// CHECK:STDOUT:     %impl.elem0.loc73_6.1: @TestGenericAssign.%.loc73_6.2 (%.326) = impl_witness_access constants.%RightShiftAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc73_6.2 (constants.%impl.elem0.bb3)]
// CHECK:STDOUT:     %bound_method.loc73_6.1: <bound method> = bound_method %.loc73_3, %impl.elem0.loc73_6.1
// CHECK:STDOUT:     %specific_impl_fn.loc73_6.1: <specific function> = specific_impl_function %impl.elem0.loc73_6.1, @RightShiftAssignWith.WithSelf.Op(constants.%T.as_type.190bb7.2, constants.%RightShiftAssignWith.facet.6e0) [symbolic = %specific_impl_fn.loc73_6.2 (constants.%specific_impl_fn.086)]
// CHECK:STDOUT:     %bound_method.loc73_6.2: <bound method> = bound_method %.loc73_3, %specific_impl_fn.loc73_6.1
// CHECK:STDOUT:     %RightShiftAssignWith.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc73_6.2(%.loc73_3, %b.ref)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGenericOp(constants.%T.f45530.1) {
// CHECK:STDOUT:   %T.patt.loc65_19.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc65_19.1 => constants.%T.f45530.1
// CHECK:STDOUT:   %T.as_type.loc65_41.1 => constants.%T.as_type.190bb7.1
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.86a9db.1
// CHECK:STDOUT:   %a.param_patt.loc65_39.2 => constants.%a.param_patt.e33
// CHECK:STDOUT:   %a.patt.loc65_39.2 => constants.%a.patt.e02
// CHECK:STDOUT:   %b.param_patt.loc65_45.2 => constants.%b.param_patt.4cd741.1
// CHECK:STDOUT:   %b.patt.loc65_45.2 => constants.%b.patt.60acf6.1
// CHECK:STDOUT:   %.loc65_53.2 => constants.%.1f0
// CHECK:STDOUT:   %return.param_patt.loc65_53.2 => constants.%return.param_patt.5f6
// CHECK:STDOUT:   %return.patt.loc65_50.2 => constants.%return.patt.96a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGenericAssign(constants.%T.f45530.2) {
// CHECK:STDOUT:   %T.patt.loc71_23.2 => constants.%T.patt.3a0592.2
// CHECK:STDOUT:   %T.loc71_23.1 => constants.%T.f45530.2
// CHECK:STDOUT:   %T.as_type.loc71_52.1 => constants.%T.as_type.190bb7.2
// CHECK:STDOUT:   %ptr.loc71_52.1 => constants.%ptr.360
// CHECK:STDOUT:   %pattern_type.loc71_49 => constants.%pattern_type.ff5
// CHECK:STDOUT:   %a.param_patt.loc71_49.2 => constants.%a.param_patt.f50
// CHECK:STDOUT:   %a.patt.loc71_49.2 => constants.%a.patt.193
// CHECK:STDOUT:   %pattern_type.loc71_56 => constants.%pattern_type.86a9db.2
// CHECK:STDOUT:   %b.param_patt.loc71_56.2 => constants.%b.param_patt.4cd741.2
// CHECK:STDOUT:   %b.patt.loc71_56.2 => constants.%b.patt.60acf6.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGenericOp(constants.%RightShift.facet.f8f) {
// CHECK:STDOUT:   %T.patt.loc65_19.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc65_19.1 => constants.%RightShift.facet.f8f
// CHECK:STDOUT:   %T.as_type.loc65_41.1 => constants.%C
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.4a7
// CHECK:STDOUT:   %a.param_patt.loc65_39.2 => constants.%a.param_patt.2a5
// CHECK:STDOUT:   %a.patt.loc65_39.2 => constants.%a.patt.a30
// CHECK:STDOUT:   %b.param_patt.loc65_45.2 => constants.%b.param_patt.45f
// CHECK:STDOUT:   %b.patt.loc65_45.2 => constants.%b.patt.562
// CHECK:STDOUT:   %.loc65_53.2 => constants.%.79c
// CHECK:STDOUT:   %return.param_patt.loc65_53.2 => constants.%return.param_patt.415
// CHECK:STDOUT:   %return.patt.loc65_50.2 => constants.%return.patt.340
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc65 => constants.%complete_type.357
// CHECK:STDOUT:   %RightShiftWith.type.loc67_12.2 => constants.%RightShiftWith.type.2f7
// CHECK:STDOUT:   %require_complete.loc67 => constants.%complete_type.4d6
// CHECK:STDOUT:   %RightShiftWith.assoc_type => constants.%RightShiftWith.assoc_type.a32
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.2d2
// CHECK:STDOUT:   %RightShiftWith.lookup_impl_witness => constants.%RightShiftWith.impl_witness.d46
// CHECK:STDOUT:   %RightShiftWith.facet => constants.%RightShiftWith.facet.da5
// CHECK:STDOUT:   %RightShiftWith.WithSelf.Op.type => constants.%RightShiftWith.WithSelf.Op.type.235
// CHECK:STDOUT:   %.loc67_12.2 => constants.%.300
// CHECK:STDOUT:   %impl.elem1.loc67_12.2 => constants.%C.as.RightShiftWith.impl.Op
// CHECK:STDOUT:   %specific_impl_fn.loc67_12.2 => constants.%C.as.RightShiftWith.impl.Op
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGenericAssign(constants.%RightShiftAssign.facet.a10) {
// CHECK:STDOUT:   %T.patt.loc71_23.2 => constants.%T.patt.3a0592.2
// CHECK:STDOUT:   %T.loc71_23.1 => constants.%RightShiftAssign.facet.a10
// CHECK:STDOUT:   %T.as_type.loc71_52.1 => constants.%C
// CHECK:STDOUT:   %ptr.loc71_52.1 => constants.%ptr.00e
// CHECK:STDOUT:   %pattern_type.loc71_49 => constants.%pattern_type.842
// CHECK:STDOUT:   %a.param_patt.loc71_49.2 => constants.%a.param_patt.4fa
// CHECK:STDOUT:   %a.patt.loc71_49.2 => constants.%a.patt.eba
// CHECK:STDOUT:   %pattern_type.loc71_56 => constants.%pattern_type.4a7
// CHECK:STDOUT:   %b.param_patt.loc71_56.2 => constants.%b.param_patt.45f
// CHECK:STDOUT:   %b.patt.loc71_56.2 => constants.%b.patt.562
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc71_49 => constants.%complete_type.a56
// CHECK:STDOUT:   %require_complete.loc71_56 => constants.%complete_type.357
// CHECK:STDOUT:   %RightShiftAssignWith.type.loc73_6.2 => constants.%RightShiftAssignWith.type.967
// CHECK:STDOUT:   %require_complete.loc73 => constants.%complete_type.315
// CHECK:STDOUT:   %RightShiftAssignWith.assoc_type => constants.%RightShiftAssignWith.assoc_type.407
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.f72
// CHECK:STDOUT:   %RightShiftAssignWith.lookup_impl_witness => constants.%RightShiftAssignWith.impl_witness.725
// CHECK:STDOUT:   %RightShiftAssignWith.facet => constants.%RightShiftAssignWith.facet.fb9
// CHECK:STDOUT:   %RightShiftAssignWith.WithSelf.Op.type => constants.%RightShiftAssignWith.WithSelf.Op.type.961
// CHECK:STDOUT:   %.loc73_6.2 => constants.%.e4d
// CHECK:STDOUT:   %impl.elem0.loc73_6.2 => constants.%C.as.RightShiftAssignWith.impl.Op
// CHECK:STDOUT:   %specific_impl_fn.loc73_6.2 => constants.%C.as.RightShiftAssignWith.impl.Op
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGenericOp(constants.%RightShift.facet.6f9) {
// CHECK:STDOUT:   %T.patt.loc65_19.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc65_19.1 => constants.%RightShift.facet.6f9
// CHECK:STDOUT:   %T.as_type.loc65_41.1 => constants.%D
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.331
// CHECK:STDOUT:   %a.param_patt.loc65_39.2 => constants.%a.param_patt.d0e
// CHECK:STDOUT:   %a.patt.loc65_39.2 => constants.%a.patt.ec5
// CHECK:STDOUT:   %b.param_patt.loc65_45.2 => constants.%b.param_patt.110
// CHECK:STDOUT:   %b.patt.loc65_45.2 => constants.%b.patt.14c
// CHECK:STDOUT:   %.loc65_53.2 => constants.%.84d
// CHECK:STDOUT:   %return.param_patt.loc65_53.2 => constants.%return.param_patt.835
// CHECK:STDOUT:   %return.patt.loc65_50.2 => constants.%return.patt.5ef
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc65 => constants.%complete_type.357
// CHECK:STDOUT:   %RightShiftWith.type.loc67_12.2 => constants.%RightShiftWith.type.7ba
// CHECK:STDOUT:   %require_complete.loc67 => constants.%complete_type.915
// CHECK:STDOUT:   %RightShiftWith.assoc_type => constants.%RightShiftWith.assoc_type.080
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.6a6
// CHECK:STDOUT:   %RightShiftWith.lookup_impl_witness => constants.%RightShiftWith.impl_witness.736
// CHECK:STDOUT:   %RightShiftWith.facet => constants.%RightShiftWith.facet.711
// CHECK:STDOUT:   %RightShiftWith.WithSelf.Op.type => constants.%RightShiftWith.WithSelf.Op.type.6e0
// CHECK:STDOUT:   %.loc67_12.2 => constants.%.027
// CHECK:STDOUT:   %impl.elem1.loc67_12.2 => constants.%D.as.RightShiftWith.impl.Op
// CHECK:STDOUT:   %specific_impl_fn.loc67_12.2 => constants.%D.as.RightShiftWith.impl.Op
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGenericAssign(constants.%RightShiftAssign.facet.6f9) {
// CHECK:STDOUT:   %T.patt.loc71_23.2 => constants.%T.patt.3a0592.2
// CHECK:STDOUT:   %T.loc71_23.1 => constants.%RightShiftAssign.facet.6f9
// CHECK:STDOUT:   %T.as_type.loc71_52.1 => constants.%D
// CHECK:STDOUT:   %ptr.loc71_52.1 => constants.%ptr.4a5
// CHECK:STDOUT:   %pattern_type.loc71_49 => constants.%pattern_type.ceb
// CHECK:STDOUT:   %a.param_patt.loc71_49.2 => constants.%a.param_patt.609
// CHECK:STDOUT:   %a.patt.loc71_49.2 => constants.%a.patt.fac
// CHECK:STDOUT:   %pattern_type.loc71_56 => constants.%pattern_type.331
// CHECK:STDOUT:   %b.param_patt.loc71_56.2 => constants.%b.param_patt.110
// CHECK:STDOUT:   %b.patt.loc71_56.2 => constants.%b.patt.14c
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc71_49 => constants.%complete_type.320
// CHECK:STDOUT:   %require_complete.loc71_56 => constants.%complete_type.357
// CHECK:STDOUT:   %RightShiftAssignWith.type.loc73_6.2 => constants.%RightShiftAssignWith.type.e5f
// CHECK:STDOUT:   %require_complete.loc73 => constants.%complete_type.c20
// CHECK:STDOUT:   %RightShiftAssignWith.assoc_type => constants.%RightShiftAssignWith.assoc_type.845
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.517
// CHECK:STDOUT:   %RightShiftAssignWith.lookup_impl_witness => constants.%RightShiftAssignWith.impl_witness.736
// CHECK:STDOUT:   %RightShiftAssignWith.facet => constants.%RightShiftAssignWith.facet.649
// CHECK:STDOUT:   %RightShiftAssignWith.WithSelf.Op.type => constants.%RightShiftAssignWith.WithSelf.Op.type.0fe
// CHECK:STDOUT:   %.loc73_6.2 => constants.%.585
// CHECK:STDOUT:   %impl.elem0.loc73_6.2 => constants.%D.as.RightShiftAssignWith.impl.Op
// CHECK:STDOUT:   %specific_impl_fn.loc73_6.2 => constants.%D.as.RightShiftAssignWith.impl.Op
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGenericOp(constants.%RightShift.facet.c40) {
// CHECK:STDOUT:   %T.patt.loc65_19.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc65_19.1 => constants.%RightShift.facet.c40
// CHECK:STDOUT:   %T.as_type.loc65_41.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.d94
// CHECK:STDOUT:   %a.param_patt.loc65_39.2 => constants.%a.param_patt.076
// CHECK:STDOUT:   %a.patt.loc65_39.2 => constants.%a.patt.96f
// CHECK:STDOUT:   %b.param_patt.loc65_45.2 => constants.%b.param_patt.1a2
// CHECK:STDOUT:   %b.patt.loc65_45.2 => constants.%b.patt.3ca
// CHECK:STDOUT:   %.loc65_53.2 => constants.%.0b3e
// CHECK:STDOUT:   %return.param_patt.loc65_53.2 => constants.%return.param_patt.299
// CHECK:STDOUT:   %return.patt.loc65_50.2 => constants.%return.patt.6fa
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc65 => constants.%complete_type.f8a
// CHECK:STDOUT:   %RightShiftWith.type.loc67_12.2 => constants.%RightShiftWith.type.efe
// CHECK:STDOUT:   %require_complete.loc67 => constants.%complete_type.310
// CHECK:STDOUT:   %RightShiftWith.assoc_type => constants.%RightShiftWith.assoc_type.415
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.ec7
// CHECK:STDOUT:   %RightShiftWith.lookup_impl_witness => constants.%RightShiftWith.impl_witness.77d
// CHECK:STDOUT:   %RightShiftWith.facet => constants.%RightShiftWith.facet.449
// CHECK:STDOUT:   %RightShiftWith.WithSelf.Op.type => constants.%RightShiftWith.WithSelf.Op.type.61f
// CHECK:STDOUT:   %.loc67_12.2 => constants.%.a2f
// CHECK:STDOUT:   %impl.elem1.loc67_12.2 => constants.%Int.as.RightShiftWith.impl.Op.d9f
// CHECK:STDOUT:   %specific_impl_fn.loc67_12.2 => constants.%Int.as.RightShiftWith.impl.Op.specific_fn.66a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGenericAssign(constants.%RightShiftAssign.facet.706) {
// CHECK:STDOUT:   %T.patt.loc71_23.2 => constants.%T.patt.3a0592.2
// CHECK:STDOUT:   %T.loc71_23.1 => constants.%RightShiftAssign.facet.706
// CHECK:STDOUT:   %T.as_type.loc71_52.1 => constants.%i32
// CHECK:STDOUT:   %ptr.loc71_52.1 => constants.%ptr.a17
// CHECK:STDOUT:   %pattern_type.loc71_49 => constants.%pattern_type.7dc
// CHECK:STDOUT:   %a.param_patt.loc71_49.2 => constants.%a.param_patt.977
// CHECK:STDOUT:   %a.patt.loc71_49.2 => constants.%a.patt.b70
// CHECK:STDOUT:   %pattern_type.loc71_56 => constants.%pattern_type.d94
// CHECK:STDOUT:   %b.param_patt.loc71_56.2 => constants.%b.param_patt.1a2
// CHECK:STDOUT:   %b.patt.loc71_56.2 => constants.%b.patt.3ca
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc71_49 => constants.%complete_type.1b1
// CHECK:STDOUT:   %require_complete.loc71_56 => constants.%complete_type.f8a
// CHECK:STDOUT:   %RightShiftAssignWith.type.loc73_6.2 => constants.%RightShiftAssignWith.type.db2
// CHECK:STDOUT:   %require_complete.loc73 => constants.%complete_type.70c
// CHECK:STDOUT:   %RightShiftAssignWith.assoc_type => constants.%RightShiftAssignWith.assoc_type.0da
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.466
// CHECK:STDOUT:   %RightShiftAssignWith.lookup_impl_witness => constants.%RightShiftAssignWith.impl_witness.67f
// CHECK:STDOUT:   %RightShiftAssignWith.facet => constants.%RightShiftAssignWith.facet.17c
// CHECK:STDOUT:   %RightShiftAssignWith.WithSelf.Op.type => constants.%RightShiftAssignWith.WithSelf.Op.type.0b8
// CHECK:STDOUT:   %.loc73_6.2 => constants.%.6c5
// CHECK:STDOUT:   %impl.elem0.loc73_6.2 => constants.%Int.as.RightShiftAssignWith.impl.Op.2676d3.2
// CHECK:STDOUT:   %specific_impl_fn.loc73_6.2 => constants.%Int.as.RightShiftAssignWith.impl.Op.specific_fn.16bcbd.1
// CHECK:STDOUT: }
// CHECK:STDOUT:

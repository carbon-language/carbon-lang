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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/overloaded/div.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/overloaded/div.carbon

// This file was generated from binary_op.carbon.tmpl. Run make_tests.sh to regenerate.

package User;

class C {};

impl C as Core.DivWith(C) where .Result = C {
  fn Op(unused self: C, unused other: C) -> C {
    return {};
  }
}
impl C as Core.DivAssignWith(C) {
  fn Op(unused ref self: C, unused other: C) {}
}

fn TestOp(a: C, b: C) -> C {
  //@dump-sem-ir-begin
  return a / b;
  //@dump-sem-ir-end
}

fn TestAssign(a: C*, b: C) {
  //@dump-sem-ir-begin
  *a /= b;
  //@dump-sem-ir-end
}

class D {};

impl D as Core.Div {
  fn Op(unused self: D, unused other: D) -> D {
    return {};
  }
}
impl D as Core.DivAssign {
  fn Op(unused ref self: D, unused other: D) {}
}

fn TestNamedConstraintOp(a: D, b: D) -> D {
  //@dump-sem-ir-begin
  return a / b;
  //@dump-sem-ir-end
}

fn TestNamedConstraintAssign(a: D*, b: D) {
  //@dump-sem-ir-begin
  *a /= b;
  //@dump-sem-ir-end
}

fn TestGenericOp[T: Core.Div](a: T, b: T) -> T {
  //@dump-sem-ir-begin
  return a / b;
  //@dump-sem-ir-end
}

fn TestGenericAssign[T: Core.DivAssign](a: T*, b: T) {
  //@dump-sem-ir-begin
  *a /= b;
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

// CHECK:STDOUT: --- div.carbon = $div
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Other: type = symbolic_binding Other, 0 [symbolic]
// CHECK:STDOUT:   %DivWith.type.472: type = facet_type <@DivWith, @DivWith(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.5b3: %DivWith.type.472 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %DivWith.assoc_type.e7d: type = assoc_entity_type @DivWith, @DivWith(%Other) [symbolic]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.207: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%Other, %Self.5b3) [symbolic]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.4f1: %DivWith.WithSelf.Op.type.207 = struct_value () [symbolic]
// CHECK:STDOUT:   %DivWith.type.46f: type = facet_type <@DivWith, @DivWith(%C)> [concrete]
// CHECK:STDOUT:   %DivWith.assoc_type.64f: type = assoc_entity_type @DivWith, @DivWith(%C) [concrete]
// CHECK:STDOUT:   %assoc1.259: %DivWith.assoc_type.64f = assoc_entity element1, imports.%Core.import_ref.f2f [concrete]
// CHECK:STDOUT:   %DivWith.impl_witness.8d9: <witness> = impl_witness @C.as.DivWith.impl.%DivWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.4a7: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %.79c: Core.Form = init_form %C [concrete]
// CHECK:STDOUT:   %return.param_patt.415: %pattern_type.4a7 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.340: %pattern_type.4a7 = return_slot_pattern %return.param_patt.415, %C [concrete]
// CHECK:STDOUT:   %C.as.DivWith.impl.Op.type: type = fn_type @C.as.DivWith.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.DivWith.impl.Op: %C.as.DivWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %DivWith.facet.990: %DivWith.type.46f = facet_value %C, (%DivWith.impl_witness.8d9) [concrete]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.fe2: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%C, %DivWith.facet.990) [concrete]
// CHECK:STDOUT:   %DivAssignWith.type.4b7: type = facet_type <@DivAssignWith, @DivAssignWith(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.5bb: %DivAssignWith.type.4b7 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %DivAssignWith.assoc_type.4e8a: type = assoc_entity_type @DivAssignWith, @DivAssignWith(%Other) [symbolic]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type.42b: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%Other, %Self.5bb) [symbolic]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.298: %DivAssignWith.WithSelf.Op.type.42b = struct_value () [symbolic]
// CHECK:STDOUT:   %DivAssignWith.type.881: type = facet_type <@DivAssignWith, @DivAssignWith(%C)> [concrete]
// CHECK:STDOUT:   %DivAssignWith.impl_witness.34d: <witness> = impl_witness @C.as.DivAssignWith.impl.%DivAssignWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %DivAssignWith.assoc_type.3a7: type = assoc_entity_type @DivAssignWith, @DivAssignWith(%C) [concrete]
// CHECK:STDOUT:   %assoc0.44e: %DivAssignWith.assoc_type.3a7 = assoc_entity element0, imports.%Core.import_ref.ee5 [concrete]
// CHECK:STDOUT:   %C.as.DivAssignWith.impl.Op.type: type = fn_type @C.as.DivAssignWith.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.DivAssignWith.impl.Op: %C.as.DivAssignWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %DivAssignWith.facet.464: %DivAssignWith.type.881 = facet_value %C, (%DivAssignWith.impl_witness.34d) [concrete]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type.5b5: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%C, %DivAssignWith.facet.464) [concrete]
// CHECK:STDOUT:   %a.param_patt.2a5: %pattern_type.4a7 = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt.a30: %pattern_type.4a7 = wrapper_binding_pattern a, %a.param_patt.2a5 [concrete]
// CHECK:STDOUT:   %b.param_patt.45f: %pattern_type.4a7 = value_param_pattern [concrete]
// CHECK:STDOUT:   %b.patt.562: %pattern_type.4a7 = wrapper_binding_pattern b, %b.param_patt.45f [concrete]
// CHECK:STDOUT:   %assoc1.2b8: %DivWith.assoc_type.e7d = assoc_entity element1, imports.%Core.import_ref.94d [symbolic]
// CHECK:STDOUT:   %.2e6: type = fn_type_with_self_type %DivWith.WithSelf.Op.type.fe2, %DivWith.facet.990 [concrete]
// CHECK:STDOUT:   %ptr.00e: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.842: type = pattern_type %ptr.00e [concrete]
// CHECK:STDOUT:   %a.param_patt.4fa: %pattern_type.842 = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt.eba: %pattern_type.842 = wrapper_binding_pattern a, %a.param_patt.4fa [concrete]
// CHECK:STDOUT:   %assoc0.45a: %DivAssignWith.assoc_type.4e8a = assoc_entity element0, imports.%Core.import_ref.50b [symbolic]
// CHECK:STDOUT:   %.c05: type = fn_type_with_self_type %DivAssignWith.WithSelf.Op.type.5b5, %DivAssignWith.facet.464 [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %Div.type: type = facet_type <@Div> [concrete]
// CHECK:STDOUT:   %DivWith.impl_witness.736: <witness> = impl_witness @D.as.DivWith.impl.%DivWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.331: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %.84d: Core.Form = init_form %D [concrete]
// CHECK:STDOUT:   %return.param_patt.835: %pattern_type.331 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.5ef: %pattern_type.331 = return_slot_pattern %return.param_patt.835, %D [concrete]
// CHECK:STDOUT:   %D.as.DivWith.impl.Op.type: type = fn_type @D.as.DivWith.impl.Op [concrete]
// CHECK:STDOUT:   %D.as.DivWith.impl.Op: %D.as.DivWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %DivWith.type.2da: type = facet_type <@DivWith, @DivWith(%D)> [concrete]
// CHECK:STDOUT:   %DivWith.facet.efa: %DivWith.type.2da = facet_value %D, (%DivWith.impl_witness.736) [concrete]
// CHECK:STDOUT:   %DivWith.assoc_type.3ec: type = assoc_entity_type @DivWith, @DivWith(%D) [concrete]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.4a0: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%D, %DivWith.facet.efa) [concrete]
// CHECK:STDOUT:   %assoc1.f36: %DivWith.assoc_type.3ec = assoc_entity element1, imports.%Core.import_ref.f2f [concrete]
// CHECK:STDOUT:   %DivAssign.type: type = facet_type <@DivAssign> [concrete]
// CHECK:STDOUT:   %DivAssignWith.impl_witness.736: <witness> = impl_witness @D.as.DivAssignWith.impl.%DivAssignWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %D.as.DivAssignWith.impl.Op.type: type = fn_type @D.as.DivAssignWith.impl.Op [concrete]
// CHECK:STDOUT:   %D.as.DivAssignWith.impl.Op: %D.as.DivAssignWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %DivAssignWith.type.723: type = facet_type <@DivAssignWith, @DivAssignWith(%D)> [concrete]
// CHECK:STDOUT:   %DivAssignWith.facet.447: %DivAssignWith.type.723 = facet_value %D, (%DivAssignWith.impl_witness.736) [concrete]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type.9f3: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%D, %DivAssignWith.facet.447) [concrete]
// CHECK:STDOUT:   %DivAssignWith.assoc_type.4e8b: type = assoc_entity_type @DivAssignWith, @DivAssignWith(%D) [concrete]
// CHECK:STDOUT:   %assoc0.04c: %DivAssignWith.assoc_type.4e8b = assoc_entity element0, imports.%Core.import_ref.ee5 [concrete]
// CHECK:STDOUT:   %a.param_patt.d0e: %pattern_type.331 = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt.ec5: %pattern_type.331 = wrapper_binding_pattern a, %a.param_patt.d0e [concrete]
// CHECK:STDOUT:   %b.param_patt.110: %pattern_type.331 = value_param_pattern [concrete]
// CHECK:STDOUT:   %b.patt.14c: %pattern_type.331 = wrapper_binding_pattern b, %b.param_patt.110 [concrete]
// CHECK:STDOUT:   %.502: type = fn_type_with_self_type %DivWith.WithSelf.Op.type.4a0, %DivWith.facet.efa [concrete]
// CHECK:STDOUT:   %ptr.4a5: type = ptr_type %D [concrete]
// CHECK:STDOUT:   %pattern_type.ceb: type = pattern_type %ptr.4a5 [concrete]
// CHECK:STDOUT:   %a.param_patt.609: %pattern_type.ceb = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt.fac: %pattern_type.ceb = wrapper_binding_pattern a, %a.param_patt.609 [concrete]
// CHECK:STDOUT:   %.037: type = fn_type_with_self_type %DivAssignWith.WithSelf.Op.type.9f3, %DivAssignWith.facet.447 [concrete]
// CHECK:STDOUT:   %pattern_type.9a587c.2: type = pattern_type %Div.type [concrete]
// CHECK:STDOUT:   %T.patt.3a0592.1: %pattern_type.9a587c.2 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.f45530.1: %Div.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.190bb7.1: type = facet_access_type %T.f45530.1 [symbolic]
// CHECK:STDOUT:   %pattern_type.86a9db.1: type = pattern_type %T.as_type.190bb7.1 [symbolic]
// CHECK:STDOUT:   %a.param_patt.e33: %pattern_type.86a9db.1 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %a.patt.e02: %pattern_type.86a9db.1 = wrapper_binding_pattern a, %a.param_patt.e33 [symbolic]
// CHECK:STDOUT:   %b.param_patt.4cd741.1: %pattern_type.86a9db.1 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %b.patt.60acf6.1: %pattern_type.86a9db.1 = wrapper_binding_pattern b, %b.param_patt.4cd741.1 [symbolic]
// CHECK:STDOUT:   %.1f0: Core.Form = init_form %T.as_type.190bb7.1 [symbolic]
// CHECK:STDOUT:   %return.param_patt.5f6: %pattern_type.86a9db.1 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.96a: %pattern_type.86a9db.1 = return_slot_pattern %return.param_patt.5f6, %T.as_type.190bb7.1 [symbolic]
// CHECK:STDOUT:   %DivWith.type.db1403.2: type = facet_type <@DivWith, @DivWith(%T.as_type.190bb7.1)> [symbolic]
// CHECK:STDOUT:   %DivWith.assoc_type.5b4429.2: type = assoc_entity_type @DivWith, @DivWith(%T.as_type.190bb7.1) [symbolic]
// CHECK:STDOUT:   %assoc1.948c2c.2: %DivWith.assoc_type.5b4429.2 = assoc_entity element1, imports.%Core.import_ref.f2f [symbolic]
// CHECK:STDOUT:   %require_complete.816: <witness> = require_complete_type %DivWith.type.db1403.2 [symbolic]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.c11c24.2: <witness> = lookup_impl_witness %T.f45530.1, @DivWith, @DivWith(%T.as_type.190bb7.1) [symbolic]
// CHECK:STDOUT:   %DivWith.facet.461: %DivWith.type.db1403.2 = facet_value %T.as_type.190bb7.1, (%DivWith.lookup_impl_witness.c11c24.2) [symbolic]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.11d: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%T.as_type.190bb7.1, %DivWith.facet.461) [symbolic]
// CHECK:STDOUT:   %.f46: type = fn_type_with_self_type %DivWith.WithSelf.Op.type.11d, %DivWith.facet.461 [symbolic]
// CHECK:STDOUT:   %impl.elem1.e15: %.f46 = impl_witness_access %DivWith.lookup_impl_witness.c11c24.2, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.021: <specific function> = specific_impl_function %impl.elem1.e15, @DivWith.WithSelf.Op(%T.as_type.190bb7.1, %DivWith.facet.461) [symbolic]
// CHECK:STDOUT:   %pattern_type.9a587c.3: type = pattern_type %DivAssign.type [concrete]
// CHECK:STDOUT:   %T.patt.3a0592.2: %pattern_type.9a587c.3 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.f45530.2: %DivAssign.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.190bb7.2: type = facet_access_type %T.f45530.2 [symbolic]
// CHECK:STDOUT:   %ptr.360: type = ptr_type %T.as_type.190bb7.2 [symbolic]
// CHECK:STDOUT:   %pattern_type.ff5: type = pattern_type %ptr.360 [symbolic]
// CHECK:STDOUT:   %a.param_patt.f50: %pattern_type.ff5 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %a.patt.193: %pattern_type.ff5 = wrapper_binding_pattern a, %a.param_patt.f50 [symbolic]
// CHECK:STDOUT:   %pattern_type.86a9db.2: type = pattern_type %T.as_type.190bb7.2 [symbolic]
// CHECK:STDOUT:   %b.param_patt.4cd741.2: %pattern_type.86a9db.2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %b.patt.60acf6.2: %pattern_type.86a9db.2 = wrapper_binding_pattern b, %b.param_patt.4cd741.2 [symbolic]
// CHECK:STDOUT:   %DivAssignWith.type.61703e.2: type = facet_type <@DivAssignWith, @DivAssignWith(%T.as_type.190bb7.2)> [symbolic]
// CHECK:STDOUT:   %DivAssignWith.assoc_type.80389e.2: type = assoc_entity_type @DivAssignWith, @DivAssignWith(%T.as_type.190bb7.2) [symbolic]
// CHECK:STDOUT:   %assoc0.aafe77.2: %DivAssignWith.assoc_type.80389e.2 = assoc_entity element0, imports.%Core.import_ref.ee5 [symbolic]
// CHECK:STDOUT:   %require_complete.2a3600.2: <witness> = require_complete_type %DivAssignWith.type.61703e.2 [symbolic]
// CHECK:STDOUT:   %DivAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.f45530.2, @DivAssignWith, @DivAssignWith(%T.as_type.190bb7.2) [symbolic]
// CHECK:STDOUT:   %DivAssignWith.facet.029: %DivAssignWith.type.61703e.2 = facet_value %T.as_type.190bb7.2, (%DivAssignWith.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type.876: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%T.as_type.190bb7.2, %DivAssignWith.facet.029) [symbolic]
// CHECK:STDOUT:   %.e62: type = fn_type_with_self_type %DivAssignWith.WithSelf.Op.type.876, %DivAssignWith.facet.029 [symbolic]
// CHECK:STDOUT:   %impl.elem0.a2e: %.e62 = impl_witness_access %DivAssignWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.9c3: <specific function> = specific_impl_function %impl.elem0.a2e, @DivAssignWith.WithSelf.Op(%T.as_type.190bb7.2, %DivAssignWith.facet.029) [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %Int.d4bbe0.1: type = class_type @Int, @Int(%N) [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %ptr.a17: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %ptr.a17 [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %Div.facet.ca6: %Div.type = facet_value %C, (%DivWith.impl_witness.8d9) [concrete]
// CHECK:STDOUT:   %DivAssign.facet.9fc: %DivAssign.type = facet_value %C, (%DivAssignWith.impl_witness.34d) [concrete]
// CHECK:STDOUT:   %Div.facet.6f9: %Div.type = facet_value %D, (%DivWith.impl_witness.736) [concrete]
// CHECK:STDOUT:   %DivAssign.facet.6f9: %DivAssign.type = facet_value %D, (%DivAssignWith.impl_witness.736) [concrete]
// CHECK:STDOUT:   %Int.as.DivWith.impl.Op.type.1cb: type = fn_type @Int.as.DivWith.impl.Op.1, @Int.as.DivWith.impl.a48(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.DivWith.impl.Op.814: %Int.as.DivWith.impl.Op.type.1cb = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.81c7d0.1: type = facet_type <@ImplicitAs, @ImplicitAs(%Int.d4bbe0.1)> [symbolic]
// CHECK:STDOUT:   %U.f16: %ImplicitAs.type.81c7d0.1 = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %DivWith.type.6d0: type = facet_type <@DivWith, @DivWith(%i32)> [concrete]
// CHECK:STDOUT:   %DivWith.impl_witness.863: <witness> = impl_witness imports.%DivWith.impl_witness_table.5df, @Int.as.DivWith.impl.a48(%int_32) [concrete]
// CHECK:STDOUT:   %DivWith.assoc_type.75d: type = assoc_entity_type @DivWith, @DivWith(%i32) [concrete]
// CHECK:STDOUT:   %assoc1.fdc: %DivWith.assoc_type.75d = assoc_entity element1, imports.%Core.import_ref.f2f [concrete]
// CHECK:STDOUT:   %complete_type.43e: <witness> = complete_type_witness %DivWith.type.6d0 [concrete]
// CHECK:STDOUT:   %Int.as.DivWith.impl.Op.type.cf0: type = fn_type @Int.as.DivWith.impl.Op.1, @Int.as.DivWith.impl.a48(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.DivWith.impl.Op.aaf: %Int.as.DivWith.impl.Op.type.cf0 = struct_value () [concrete]
// CHECK:STDOUT:   %Div.facet.3a5: %Div.type = facet_value %i32, (%DivWith.impl_witness.863) [concrete]
// CHECK:STDOUT:   %a.param_patt.076: %pattern_type.d94 = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt.96f: %pattern_type.d94 = wrapper_binding_pattern a, %a.param_patt.076 [concrete]
// CHECK:STDOUT:   %b.param_patt.1a2: %pattern_type.d94 = value_param_pattern [concrete]
// CHECK:STDOUT:   %b.patt.3ca: %pattern_type.d94 = wrapper_binding_pattern b, %b.param_patt.1a2 [concrete]
// CHECK:STDOUT:   %.0b3: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.299: %pattern_type.d94 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.6fa: %pattern_type.d94 = return_slot_pattern %return.param_patt.299, %i32 [concrete]
// CHECK:STDOUT:   %Int.as.DivAssignWith.impl.Op.type.694d6e.1: type = fn_type @Int.as.DivAssignWith.impl.Op.1, @Int.as.DivAssignWith.impl(%N, %U.f16) [symbolic]
// CHECK:STDOUT:   %Int.as.DivAssignWith.impl.Op.f16edf.1: %Int.as.DivAssignWith.impl.Op.type.694d6e.1 = struct_value () [symbolic]
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
// CHECK:STDOUT:   %DivAssignWith.type.49e: type = facet_type <@DivAssignWith, @DivAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %DivAssignWith.impl_witness.0b2: <witness> = impl_witness imports.%DivAssignWith.impl_witness_table, @Int.as.DivAssignWith.impl(%int_32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %DivAssign.facet.e5a: %DivAssign.type = facet_value %i32, (%DivAssignWith.impl_witness.0b2) [concrete]
// CHECK:STDOUT:   %a.param_patt.977: %pattern_type.7dc = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt.b70: %pattern_type.7dc = wrapper_binding_pattern a, %a.param_patt.977 [concrete]
// CHECK:STDOUT:   %complete_type.8a3: <witness> = complete_type_witness %DivWith.type.46f [concrete]
// CHECK:STDOUT:   %complete_type.a56: <witness> = complete_type_witness %ptr.00e [concrete]
// CHECK:STDOUT:   %complete_type.3e7: <witness> = complete_type_witness %DivAssignWith.type.881 [concrete]
// CHECK:STDOUT:   %complete_type.8bc: <witness> = complete_type_witness %DivWith.type.2da [concrete]
// CHECK:STDOUT:   %complete_type.320: <witness> = complete_type_witness %ptr.4a5 [concrete]
// CHECK:STDOUT:   %complete_type.61a: <witness> = complete_type_witness %DivAssignWith.type.723 [concrete]
// CHECK:STDOUT:   %DivWith.facet.e66: %DivWith.type.6d0 = facet_value %i32, (%DivWith.impl_witness.863) [concrete]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.07a: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%i32, %DivWith.facet.e66) [concrete]
// CHECK:STDOUT:   %.7fe: type = fn_type_with_self_type %DivWith.WithSelf.Op.type.07a, %DivWith.facet.e66 [concrete]
// CHECK:STDOUT:   %Int.as.DivWith.impl.Op.specific_fn.7a5: <specific function> = specific_function %Int.as.DivWith.impl.Op.aaf, @Int.as.DivWith.impl.Op.1(%int_32) [concrete]
// CHECK:STDOUT:   %complete_type.1b1: <witness> = complete_type_witness %ptr.a17 [concrete]
// CHECK:STDOUT:   %DivAssignWith.assoc_type.4b8: type = assoc_entity_type @DivAssignWith, @DivAssignWith(%i32) [concrete]
// CHECK:STDOUT:   %assoc0.df8: %DivAssignWith.assoc_type.4b8 = assoc_entity element0, imports.%Core.import_ref.ee5 [concrete]
// CHECK:STDOUT:   %complete_type.eec: <witness> = complete_type_witness %DivAssignWith.type.49e [concrete]
// CHECK:STDOUT:   %DivAssignWith.facet.7b0: %DivAssignWith.type.49e = facet_value %i32, (%DivAssignWith.impl_witness.0b2) [concrete]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type.1df: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%i32, %DivAssignWith.facet.7b0) [concrete]
// CHECK:STDOUT:   %.4e9: type = fn_type_with_self_type %DivAssignWith.WithSelf.Op.type.1df, %DivAssignWith.facet.7b0 [concrete]
// CHECK:STDOUT:   %Int.as.DivAssignWith.impl.Op.type.4499ab.2: type = fn_type @Int.as.DivAssignWith.impl.Op.1, @Int.as.DivAssignWith.impl(%int_32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Int.as.DivAssignWith.impl.Op.559dc5.2: %Int.as.DivAssignWith.impl.Op.type.4499ab.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.DivAssignWith.impl.Op.specific_fn.b672b8.1: <specific function> = specific_function %Int.as.DivAssignWith.impl.Op.559dc5.2, @Int.as.DivAssignWith.impl.Op.1(%int_32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.972: @DivWith.WithSelf.%DivWith.assoc_type (%DivWith.assoc_type.e7d) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @DivWith.WithSelf.%assoc1 (constants.%assoc1.2b8)]
// CHECK:STDOUT:   %Core.import_ref.f2f: @DivWith.WithSelf.%DivWith.WithSelf.Op.type (%DivWith.WithSelf.Op.type.207) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @DivWith.WithSelf.%DivWith.WithSelf.Op (constants.%DivWith.WithSelf.Op.4f1)]
// CHECK:STDOUT:   %Core.import_ref.5dd: @DivAssignWith.WithSelf.%DivAssignWith.assoc_type (%DivAssignWith.assoc_type.4e8a) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @DivAssignWith.WithSelf.%assoc0 (constants.%assoc0.45a)]
// CHECK:STDOUT:   %Core.import_ref.ee5: @DivAssignWith.WithSelf.%DivAssignWith.WithSelf.Op.type (%DivAssignWith.WithSelf.Op.type.42b) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @DivAssignWith.WithSelf.%DivAssignWith.WithSelf.Op (constants.%DivAssignWith.WithSelf.Op.298)]
// CHECK:STDOUT:   %Core.import_ref.94d = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.50b = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.ebd: type = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.DivWith.impl.a48.%Int (constants.%Int.d4bbe0.1)]
// CHECK:STDOUT:   %Core.import_ref.c9d: @Int.as.DivWith.impl.a48.%Int.as.DivWith.impl.Op.type (%Int.as.DivWith.impl.Op.type.1cb) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.DivWith.impl.a48.%Int.as.DivWith.impl.Op (constants.%Int.as.DivWith.impl.Op.814)]
// CHECK:STDOUT:   %DivWith.impl_witness_table.5df = impl_witness_table (%Core.import_ref.ebd, %Core.import_ref.c9d), @Int.as.DivWith.impl.a48 [concrete]
// CHECK:STDOUT:   %Core.import_ref.ac5: @Int.as.DivAssignWith.impl.%Int.as.DivAssignWith.impl.Op.type.2 (%Int.as.DivAssignWith.impl.Op.type.694d6e.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.DivAssignWith.impl.%Int.as.DivAssignWith.impl.Op.2 (constants.%Int.as.DivAssignWith.impl.Op.f16edf.1)]
// CHECK:STDOUT:   %DivAssignWith.impl_witness_table = impl_witness_table (%Core.import_ref.ac5), @Int.as.DivAssignWith.impl [concrete]
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
// CHECK:STDOUT:   %impl.elem1: %.2e6 = impl_witness_access constants.%DivWith.impl_witness.8d9, element1 [concrete = constants.%C.as.DivWith.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.ref, %impl.elem1
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.as.DivWith.impl.Op.call: init %C to %.loc30_26.1 = call %bound_method(%a.ref, %b.ref)
// CHECK:STDOUT:   return %C.as.DivWith.impl.Op.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestAssign(%a.param: %ptr.00e, %b.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %ptr.00e = name_ref a, %a
// CHECK:STDOUT:   %.loc38: ref %C = deref %a.ref
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0: %.c05 = impl_witness_access constants.%DivAssignWith.impl_witness.34d, element0 [concrete = constants.%C.as.DivAssignWith.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc38, %impl.elem0
// CHECK:STDOUT:   %C.as.DivAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method(%.loc38, %b.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestNamedConstraintOp(%a.param: %D, %b.param: %D) -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %D = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %D = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1: %.502 = impl_witness_access constants.%DivWith.impl_witness.736, element1 [concrete = constants.%D.as.DivWith.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.ref, %impl.elem1
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %D.as.DivWith.impl.Op.call: init %D to %.loc53_41.1 = call %bound_method(%a.ref, %b.ref)
// CHECK:STDOUT:   return %D.as.DivWith.impl.Op.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestNamedConstraintAssign(%a.param: %ptr.4a5, %b.param: %D) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %ptr.4a5 = name_ref a, %a
// CHECK:STDOUT:   %.loc61: ref %D = deref %a.ref
// CHECK:STDOUT:   %b.ref: %D = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0: %.037 = impl_witness_access constants.%DivAssignWith.impl_witness.736, element0 [concrete = constants.%D.as.DivAssignWith.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc61, %impl.elem0
// CHECK:STDOUT:   %D.as.DivAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method(%.loc61, %b.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestGenericOp(%T.loc65_19.2: %Div.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %DivWith.type.loc67_12.2: type = facet_type <@DivWith, @DivWith(%T.as_type.loc65_34.1)> [symbolic = %DivWith.type.loc67_12.2 (constants.%DivWith.type.db1403.2)]
// CHECK:STDOUT:   %require_complete.loc67: <witness> = require_complete_type %DivWith.type.loc67_12.2 [symbolic = %require_complete.loc67 (constants.%require_complete.816)]
// CHECK:STDOUT:   %DivWith.assoc_type: type = assoc_entity_type @DivWith, @DivWith(%T.as_type.loc65_34.1) [symbolic = %DivWith.assoc_type (constants.%DivWith.assoc_type.5b4429.2)]
// CHECK:STDOUT:   %assoc1: @TestGenericOp.%DivWith.assoc_type (%DivWith.assoc_type.5b4429.2) = assoc_entity element1, imports.%Core.import_ref.f2f [symbolic = %assoc1 (constants.%assoc1.948c2c.2)]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc65_19.1, @DivWith, @DivWith(%T.as_type.loc65_34.1) [symbolic = %DivWith.lookup_impl_witness (constants.%DivWith.lookup_impl_witness.c11c24.2)]
// CHECK:STDOUT:   %DivWith.facet: @TestGenericOp.%DivWith.type.loc67_12.2 (%DivWith.type.db1403.2) = facet_value %T.as_type.loc65_34.1, (%DivWith.lookup_impl_witness) [symbolic = %DivWith.facet (constants.%DivWith.facet.461)]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%T.as_type.loc65_34.1, %DivWith.facet) [symbolic = %DivWith.WithSelf.Op.type (constants.%DivWith.WithSelf.Op.type.11d)]
// CHECK:STDOUT:   %.loc67_12.2: type = fn_type_with_self_type %DivWith.WithSelf.Op.type, %DivWith.facet [symbolic = %.loc67_12.2 (constants.%.f46)]
// CHECK:STDOUT:   %impl.elem1.loc67_12.2: @TestGenericOp.%.loc67_12.2 (%.f46) = impl_witness_access %DivWith.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc67_12.2 (constants.%impl.elem1.e15)]
// CHECK:STDOUT:   %specific_impl_fn.loc67_12.2: <specific function> = specific_impl_function %impl.elem1.loc67_12.2, @DivWith.WithSelf.Op(%T.as_type.loc65_34.1, %DivWith.facet) [symbolic = %specific_impl_fn.loc67_12.2 (constants.%specific_impl_fn.021)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @TestGenericOp.%T.as_type.loc65_34.1 (%T.as_type.190bb7.1), %b.param: @TestGenericOp.%T.as_type.loc65_34.1 (%T.as_type.190bb7.1)) -> out %return.param: @TestGenericOp.%T.as_type.loc65_34.1 (%T.as_type.190bb7.1) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %a.ref: @TestGenericOp.%T.as_type.loc65_34.1 (%T.as_type.190bb7.1) = name_ref a, %a
// CHECK:STDOUT:     %b.ref: @TestGenericOp.%T.as_type.loc65_34.1 (%T.as_type.190bb7.1) = name_ref b, %b
// CHECK:STDOUT:     %DivWith.type.loc67_12.1: type = facet_type <@DivWith, @DivWith(constants.%T.as_type.190bb7.1)> [symbolic = %DivWith.type.loc67_12.2 (constants.%DivWith.type.db1403.2)]
// CHECK:STDOUT:     %.loc67_12.1: @TestGenericOp.%DivWith.assoc_type (%DivWith.assoc_type.5b4429.2) = specific_constant imports.%Core.import_ref.972, @DivWith.WithSelf(constants.%T.as_type.190bb7.1, constants.%Self.5b3) [symbolic = %assoc1 (constants.%assoc1.948c2c.2)]
// CHECK:STDOUT:     %Op.ref: @TestGenericOp.%DivWith.assoc_type (%DivWith.assoc_type.5b4429.2) = name_ref Op, %.loc67_12.1 [symbolic = %assoc1 (constants.%assoc1.948c2c.2)]
// CHECK:STDOUT:     %impl.elem1.loc67_12.1: @TestGenericOp.%.loc67_12.2 (%.f46) = impl_witness_access constants.%DivWith.lookup_impl_witness.c11c24.2, element1 [symbolic = %impl.elem1.loc67_12.2 (constants.%impl.elem1.e15)]
// CHECK:STDOUT:     %bound_method.loc67_12.1: <bound method> = bound_method %a.ref, %impl.elem1.loc67_12.1
// CHECK:STDOUT:     %specific_impl_fn.loc67_12.1: <specific function> = specific_impl_function %impl.elem1.loc67_12.1, @DivWith.WithSelf.Op(constants.%T.as_type.190bb7.1, constants.%DivWith.facet.461) [symbolic = %specific_impl_fn.loc67_12.2 (constants.%specific_impl_fn.021)]
// CHECK:STDOUT:     %bound_method.loc67_12.2: <bound method> = bound_method %a.ref, %specific_impl_fn.loc67_12.1
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %DivWith.WithSelf.Op.call: init @TestGenericOp.%T.as_type.loc65_34.1 (%T.as_type.190bb7.1) to %.loc65_46.1 = call %bound_method.loc67_12.2(%a.ref, %b.ref)
// CHECK:STDOUT:     return %DivWith.WithSelf.Op.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestGenericAssign(%T.loc71_23.2: %DivAssign.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %DivAssignWith.type.loc73_6.2: type = facet_type <@DivAssignWith, @DivAssignWith(%T.as_type.loc71_45.1)> [symbolic = %DivAssignWith.type.loc73_6.2 (constants.%DivAssignWith.type.61703e.2)]
// CHECK:STDOUT:   %require_complete.loc73: <witness> = require_complete_type %DivAssignWith.type.loc73_6.2 [symbolic = %require_complete.loc73 (constants.%require_complete.2a3600.2)]
// CHECK:STDOUT:   %DivAssignWith.assoc_type: type = assoc_entity_type @DivAssignWith, @DivAssignWith(%T.as_type.loc71_45.1) [symbolic = %DivAssignWith.assoc_type (constants.%DivAssignWith.assoc_type.80389e.2)]
// CHECK:STDOUT:   %assoc0: @TestGenericAssign.%DivAssignWith.assoc_type (%DivAssignWith.assoc_type.80389e.2) = assoc_entity element0, imports.%Core.import_ref.ee5 [symbolic = %assoc0 (constants.%assoc0.aafe77.2)]
// CHECK:STDOUT:   %DivAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc71_23.1, @DivAssignWith, @DivAssignWith(%T.as_type.loc71_45.1) [symbolic = %DivAssignWith.lookup_impl_witness (constants.%DivAssignWith.lookup_impl_witness)]
// CHECK:STDOUT:   %DivAssignWith.facet: @TestGenericAssign.%DivAssignWith.type.loc73_6.2 (%DivAssignWith.type.61703e.2) = facet_value %T.as_type.loc71_45.1, (%DivAssignWith.lookup_impl_witness) [symbolic = %DivAssignWith.facet (constants.%DivAssignWith.facet.029)]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%T.as_type.loc71_45.1, %DivAssignWith.facet) [symbolic = %DivAssignWith.WithSelf.Op.type (constants.%DivAssignWith.WithSelf.Op.type.876)]
// CHECK:STDOUT:   %.loc73_6.2: type = fn_type_with_self_type %DivAssignWith.WithSelf.Op.type, %DivAssignWith.facet [symbolic = %.loc73_6.2 (constants.%.e62)]
// CHECK:STDOUT:   %impl.elem0.loc73_6.2: @TestGenericAssign.%.loc73_6.2 (%.e62) = impl_witness_access %DivAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc73_6.2 (constants.%impl.elem0.a2e)]
// CHECK:STDOUT:   %specific_impl_fn.loc73_6.2: <specific function> = specific_impl_function %impl.elem0.loc73_6.2, @DivAssignWith.WithSelf.Op(%T.as_type.loc71_45.1, %DivAssignWith.facet) [symbolic = %specific_impl_fn.loc73_6.2 (constants.%specific_impl_fn.9c3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @TestGenericAssign.%ptr.loc71_45.1 (%ptr.360), %b.param: @TestGenericAssign.%T.as_type.loc71_45.1 (%T.as_type.190bb7.2)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %a.ref: @TestGenericAssign.%ptr.loc71_45.1 (%ptr.360) = name_ref a, %a
// CHECK:STDOUT:     %.loc73_3: ref @TestGenericAssign.%T.as_type.loc71_45.1 (%T.as_type.190bb7.2) = deref %a.ref
// CHECK:STDOUT:     %b.ref: @TestGenericAssign.%T.as_type.loc71_45.1 (%T.as_type.190bb7.2) = name_ref b, %b
// CHECK:STDOUT:     %DivAssignWith.type.loc73_6.1: type = facet_type <@DivAssignWith, @DivAssignWith(constants.%T.as_type.190bb7.2)> [symbolic = %DivAssignWith.type.loc73_6.2 (constants.%DivAssignWith.type.61703e.2)]
// CHECK:STDOUT:     %.loc73_6.1: @TestGenericAssign.%DivAssignWith.assoc_type (%DivAssignWith.assoc_type.80389e.2) = specific_constant imports.%Core.import_ref.5dd, @DivAssignWith.WithSelf(constants.%T.as_type.190bb7.2, constants.%Self.5bb) [symbolic = %assoc0 (constants.%assoc0.aafe77.2)]
// CHECK:STDOUT:     %Op.ref: @TestGenericAssign.%DivAssignWith.assoc_type (%DivAssignWith.assoc_type.80389e.2) = name_ref Op, %.loc73_6.1 [symbolic = %assoc0 (constants.%assoc0.aafe77.2)]
// CHECK:STDOUT:     %impl.elem0.loc73_6.1: @TestGenericAssign.%.loc73_6.2 (%.e62) = impl_witness_access constants.%DivAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc73_6.2 (constants.%impl.elem0.a2e)]
// CHECK:STDOUT:     %bound_method.loc73_6.1: <bound method> = bound_method %.loc73_3, %impl.elem0.loc73_6.1
// CHECK:STDOUT:     %specific_impl_fn.loc73_6.1: <specific function> = specific_impl_function %impl.elem0.loc73_6.1, @DivAssignWith.WithSelf.Op(constants.%T.as_type.190bb7.2, constants.%DivAssignWith.facet.029) [symbolic = %specific_impl_fn.loc73_6.2 (constants.%specific_impl_fn.9c3)]
// CHECK:STDOUT:     %bound_method.loc73_6.2: <bound method> = bound_method %.loc73_3, %specific_impl_fn.loc73_6.1
// CHECK:STDOUT:     %DivAssignWith.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc73_6.2(%.loc73_3, %b.ref)
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
// CHECK:STDOUT: specific @TestGenericOp(constants.%Div.facet.ca6) {
// CHECK:STDOUT:   %T.patt.loc65_19.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc65_19.1 => constants.%Div.facet.ca6
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
// CHECK:STDOUT:   %DivWith.type.loc67_12.2 => constants.%DivWith.type.46f
// CHECK:STDOUT:   %require_complete.loc67 => constants.%complete_type.8a3
// CHECK:STDOUT:   %DivWith.assoc_type => constants.%DivWith.assoc_type.64f
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.259
// CHECK:STDOUT:   %DivWith.lookup_impl_witness => constants.%DivWith.impl_witness.8d9
// CHECK:STDOUT:   %DivWith.facet => constants.%DivWith.facet.990
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type => constants.%DivWith.WithSelf.Op.type.fe2
// CHECK:STDOUT:   %.loc67_12.2 => constants.%.2e6
// CHECK:STDOUT:   %impl.elem1.loc67_12.2 => constants.%C.as.DivWith.impl.Op
// CHECK:STDOUT:   %specific_impl_fn.loc67_12.2 => constants.%C.as.DivWith.impl.Op
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGenericAssign(constants.%DivAssign.facet.9fc) {
// CHECK:STDOUT:   %T.patt.loc71_23.2 => constants.%T.patt.3a0592.2
// CHECK:STDOUT:   %T.loc71_23.1 => constants.%DivAssign.facet.9fc
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
// CHECK:STDOUT:   %DivAssignWith.type.loc73_6.2 => constants.%DivAssignWith.type.881
// CHECK:STDOUT:   %require_complete.loc73 => constants.%complete_type.3e7
// CHECK:STDOUT:   %DivAssignWith.assoc_type => constants.%DivAssignWith.assoc_type.3a7
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.44e
// CHECK:STDOUT:   %DivAssignWith.lookup_impl_witness => constants.%DivAssignWith.impl_witness.34d
// CHECK:STDOUT:   %DivAssignWith.facet => constants.%DivAssignWith.facet.464
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type => constants.%DivAssignWith.WithSelf.Op.type.5b5
// CHECK:STDOUT:   %.loc73_6.2 => constants.%.c05
// CHECK:STDOUT:   %impl.elem0.loc73_6.2 => constants.%C.as.DivAssignWith.impl.Op
// CHECK:STDOUT:   %specific_impl_fn.loc73_6.2 => constants.%C.as.DivAssignWith.impl.Op
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGenericOp(constants.%Div.facet.6f9) {
// CHECK:STDOUT:   %T.patt.loc65_19.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc65_19.1 => constants.%Div.facet.6f9
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
// CHECK:STDOUT:   %DivWith.type.loc67_12.2 => constants.%DivWith.type.2da
// CHECK:STDOUT:   %require_complete.loc67 => constants.%complete_type.8bc
// CHECK:STDOUT:   %DivWith.assoc_type => constants.%DivWith.assoc_type.3ec
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.f36
// CHECK:STDOUT:   %DivWith.lookup_impl_witness => constants.%DivWith.impl_witness.736
// CHECK:STDOUT:   %DivWith.facet => constants.%DivWith.facet.efa
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type => constants.%DivWith.WithSelf.Op.type.4a0
// CHECK:STDOUT:   %.loc67_12.2 => constants.%.502
// CHECK:STDOUT:   %impl.elem1.loc67_12.2 => constants.%D.as.DivWith.impl.Op
// CHECK:STDOUT:   %specific_impl_fn.loc67_12.2 => constants.%D.as.DivWith.impl.Op
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGenericAssign(constants.%DivAssign.facet.6f9) {
// CHECK:STDOUT:   %T.patt.loc71_23.2 => constants.%T.patt.3a0592.2
// CHECK:STDOUT:   %T.loc71_23.1 => constants.%DivAssign.facet.6f9
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
// CHECK:STDOUT:   %DivAssignWith.type.loc73_6.2 => constants.%DivAssignWith.type.723
// CHECK:STDOUT:   %require_complete.loc73 => constants.%complete_type.61a
// CHECK:STDOUT:   %DivAssignWith.assoc_type => constants.%DivAssignWith.assoc_type.4e8b
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.04c
// CHECK:STDOUT:   %DivAssignWith.lookup_impl_witness => constants.%DivAssignWith.impl_witness.736
// CHECK:STDOUT:   %DivAssignWith.facet => constants.%DivAssignWith.facet.447
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type => constants.%DivAssignWith.WithSelf.Op.type.9f3
// CHECK:STDOUT:   %.loc73_6.2 => constants.%.037
// CHECK:STDOUT:   %impl.elem0.loc73_6.2 => constants.%D.as.DivAssignWith.impl.Op
// CHECK:STDOUT:   %specific_impl_fn.loc73_6.2 => constants.%D.as.DivAssignWith.impl.Op
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGenericOp(constants.%Div.facet.3a5) {
// CHECK:STDOUT:   %T.patt.loc65_19.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc65_19.1 => constants.%Div.facet.3a5
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
// CHECK:STDOUT:   %DivWith.type.loc67_12.2 => constants.%DivWith.type.6d0
// CHECK:STDOUT:   %require_complete.loc67 => constants.%complete_type.43e
// CHECK:STDOUT:   %DivWith.assoc_type => constants.%DivWith.assoc_type.75d
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.fdc
// CHECK:STDOUT:   %DivWith.lookup_impl_witness => constants.%DivWith.impl_witness.863
// CHECK:STDOUT:   %DivWith.facet => constants.%DivWith.facet.e66
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type => constants.%DivWith.WithSelf.Op.type.07a
// CHECK:STDOUT:   %.loc67_12.2 => constants.%.7fe
// CHECK:STDOUT:   %impl.elem1.loc67_12.2 => constants.%Int.as.DivWith.impl.Op.aaf
// CHECK:STDOUT:   %specific_impl_fn.loc67_12.2 => constants.%Int.as.DivWith.impl.Op.specific_fn.7a5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGenericAssign(constants.%DivAssign.facet.e5a) {
// CHECK:STDOUT:   %T.patt.loc71_23.2 => constants.%T.patt.3a0592.2
// CHECK:STDOUT:   %T.loc71_23.1 => constants.%DivAssign.facet.e5a
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
// CHECK:STDOUT:   %DivAssignWith.type.loc73_6.2 => constants.%DivAssignWith.type.49e
// CHECK:STDOUT:   %require_complete.loc73 => constants.%complete_type.eec
// CHECK:STDOUT:   %DivAssignWith.assoc_type => constants.%DivAssignWith.assoc_type.4b8
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.df8
// CHECK:STDOUT:   %DivAssignWith.lookup_impl_witness => constants.%DivAssignWith.impl_witness.0b2
// CHECK:STDOUT:   %DivAssignWith.facet => constants.%DivAssignWith.facet.7b0
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type => constants.%DivAssignWith.WithSelf.Op.type.1df
// CHECK:STDOUT:   %.loc73_6.2 => constants.%.4e9
// CHECK:STDOUT:   %impl.elem0.loc73_6.2 => constants.%Int.as.DivAssignWith.impl.Op.559dc5.2
// CHECK:STDOUT:   %specific_impl_fn.loc73_6.2 => constants.%Int.as.DivAssignWith.impl.Op.specific_fn.b672b8.1
// CHECK:STDOUT: }
// CHECK:STDOUT:

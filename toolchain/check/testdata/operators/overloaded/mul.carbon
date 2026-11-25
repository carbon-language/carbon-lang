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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/overloaded/mul.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/overloaded/mul.carbon

// This file was generated from binary_op.carbon.tmpl. Run make_tests.sh to regenerate.

package User;

class C {};

impl C as Core.MulWith(C) where .Result = C {
  fn Op(unused self: C, unused other: C) -> C {
    return {};
  }
}
impl C as Core.MulAssignWith(C) {
  fn Op(unused ref self: C, unused other: C) {}
}

fn TestOp(a: C, b: C) -> C {
  //@dump-sem-ir-begin
  return a * b;
  //@dump-sem-ir-end
}

fn TestAssign(a: C*, b: C) {
  //@dump-sem-ir-begin
  *a *= b;
  //@dump-sem-ir-end
}

class D {};

impl D as Core.Mul {
  fn Op(unused self: D, unused other: D) -> D {
    return {};
  }
}
impl D as Core.MulAssign {
  fn Op(unused ref self: D, unused other: D) {}
}

fn TestNamedConstraintOp(a: D, b: D) -> D {
  //@dump-sem-ir-begin
  return a * b;
  //@dump-sem-ir-end
}

fn TestNamedConstraintAssign(a: D*, b: D) {
  //@dump-sem-ir-begin
  *a *= b;
  //@dump-sem-ir-end
}

fn TestGenericOp[T: Core.Mul](a: T, b: T) -> T {
  //@dump-sem-ir-begin
  return a * b;
  //@dump-sem-ir-end
}

fn TestGenericAssign[T: Core.MulAssign](a: T*, b: T) {
  //@dump-sem-ir-begin
  *a *= b;
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

// CHECK:STDOUT: --- mul.carbon = $mul
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Other: type = symbolic_binding Other, 0 [symbolic]
// CHECK:STDOUT:   %MulWith.type.616: type = facet_type <@MulWith, @MulWith(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.742: %MulWith.type.616 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %MulWith.assoc_type.c81: type = assoc_entity_type @MulWith, @MulWith(%Other) [symbolic]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.dad: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%Other, %Self.742) [symbolic]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.507: %MulWith.WithSelf.Op.type.dad = struct_value () [symbolic]
// CHECK:STDOUT:   %MulWith.type.883: type = facet_type <@MulWith, @MulWith(%C)> [concrete]
// CHECK:STDOUT:   %MulWith.assoc_type.813: type = assoc_entity_type @MulWith, @MulWith(%C) [concrete]
// CHECK:STDOUT:   %assoc1.2e6: %MulWith.assoc_type.813 = assoc_entity element1, imports.%Core.import_ref.fc2 [concrete]
// CHECK:STDOUT:   %MulWith.impl_witness.278: <witness> = impl_witness @C.as.MulWith.impl.%MulWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.4a7: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %.79c: Core.Form = init_form %C [concrete]
// CHECK:STDOUT:   %return.param_patt.415: %pattern_type.4a7 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.340: %pattern_type.4a7 = return_slot_pattern %return.param_patt.415, %C [concrete]
// CHECK:STDOUT:   %C.as.MulWith.impl.Op.type: type = fn_type @C.as.MulWith.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.MulWith.impl.Op: %C.as.MulWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %MulWith.facet.5f2: %MulWith.type.883 = facet_value %C, (%MulWith.impl_witness.278) [concrete]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.03e: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%C, %MulWith.facet.5f2) [concrete]
// CHECK:STDOUT:   %MulAssignWith.type.233: type = facet_type <@MulAssignWith, @MulAssignWith(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.9e4: %MulAssignWith.type.233 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %MulAssignWith.assoc_type.5ad: type = assoc_entity_type @MulAssignWith, @MulAssignWith(%Other) [symbolic]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type.c40: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%Other, %Self.9e4) [symbolic]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.b36: %MulAssignWith.WithSelf.Op.type.c40 = struct_value () [symbolic]
// CHECK:STDOUT:   %MulAssignWith.type.24f: type = facet_type <@MulAssignWith, @MulAssignWith(%C)> [concrete]
// CHECK:STDOUT:   %MulAssignWith.impl_witness.7ee: <witness> = impl_witness @C.as.MulAssignWith.impl.%MulAssignWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %MulAssignWith.assoc_type.06c: type = assoc_entity_type @MulAssignWith, @MulAssignWith(%C) [concrete]
// CHECK:STDOUT:   %assoc0.187: %MulAssignWith.assoc_type.06c = assoc_entity element0, imports.%Core.import_ref.c11 [concrete]
// CHECK:STDOUT:   %C.as.MulAssignWith.impl.Op.type: type = fn_type @C.as.MulAssignWith.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.MulAssignWith.impl.Op: %C.as.MulAssignWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %MulAssignWith.facet.9be: %MulAssignWith.type.24f = facet_value %C, (%MulAssignWith.impl_witness.7ee) [concrete]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type.534: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%C, %MulAssignWith.facet.9be) [concrete]
// CHECK:STDOUT:   %a.param_patt.2a5: %pattern_type.4a7 = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt.a30: %pattern_type.4a7 = wrapper_binding_pattern a, %a.param_patt.2a5 [concrete]
// CHECK:STDOUT:   %b.param_patt.45f: %pattern_type.4a7 = value_param_pattern [concrete]
// CHECK:STDOUT:   %b.patt.562: %pattern_type.4a7 = wrapper_binding_pattern b, %b.param_patt.45f [concrete]
// CHECK:STDOUT:   %assoc1.d9e: %MulWith.assoc_type.c81 = assoc_entity element1, imports.%Core.import_ref.29d [symbolic]
// CHECK:STDOUT:   %.387: type = fn_type_with_self_type %MulWith.WithSelf.Op.type.03e, %MulWith.facet.5f2 [concrete]
// CHECK:STDOUT:   %ptr.00e: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.842: type = pattern_type %ptr.00e [concrete]
// CHECK:STDOUT:   %a.param_patt.4fa: %pattern_type.842 = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt.eba: %pattern_type.842 = wrapper_binding_pattern a, %a.param_patt.4fa [concrete]
// CHECK:STDOUT:   %assoc0.911: %MulAssignWith.assoc_type.5ad = assoc_entity element0, imports.%Core.import_ref.d9a [symbolic]
// CHECK:STDOUT:   %.d01: type = fn_type_with_self_type %MulAssignWith.WithSelf.Op.type.534, %MulAssignWith.facet.9be [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %Mul.type: type = facet_type <@Mul> [concrete]
// CHECK:STDOUT:   %MulWith.impl_witness.736: <witness> = impl_witness @D.as.MulWith.impl.%MulWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.331: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %.84d: Core.Form = init_form %D [concrete]
// CHECK:STDOUT:   %return.param_patt.835: %pattern_type.331 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.5ef: %pattern_type.331 = return_slot_pattern %return.param_patt.835, %D [concrete]
// CHECK:STDOUT:   %D.as.MulWith.impl.Op.type: type = fn_type @D.as.MulWith.impl.Op [concrete]
// CHECK:STDOUT:   %D.as.MulWith.impl.Op: %D.as.MulWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %MulWith.type.ac3: type = facet_type <@MulWith, @MulWith(%D)> [concrete]
// CHECK:STDOUT:   %MulWith.facet.e7a: %MulWith.type.ac3 = facet_value %D, (%MulWith.impl_witness.736) [concrete]
// CHECK:STDOUT:   %MulWith.assoc_type.1e9: type = assoc_entity_type @MulWith, @MulWith(%D) [concrete]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.cbd: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%D, %MulWith.facet.e7a) [concrete]
// CHECK:STDOUT:   %assoc1.cc9: %MulWith.assoc_type.1e9 = assoc_entity element1, imports.%Core.import_ref.fc2 [concrete]
// CHECK:STDOUT:   %MulAssign.type: type = facet_type <@MulAssign> [concrete]
// CHECK:STDOUT:   %MulAssignWith.impl_witness.736: <witness> = impl_witness @D.as.MulAssignWith.impl.%MulAssignWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %D.as.MulAssignWith.impl.Op.type: type = fn_type @D.as.MulAssignWith.impl.Op [concrete]
// CHECK:STDOUT:   %D.as.MulAssignWith.impl.Op: %D.as.MulAssignWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %MulAssignWith.type.2e2: type = facet_type <@MulAssignWith, @MulAssignWith(%D)> [concrete]
// CHECK:STDOUT:   %MulAssignWith.facet.ddf: %MulAssignWith.type.2e2 = facet_value %D, (%MulAssignWith.impl_witness.736) [concrete]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type.523: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%D, %MulAssignWith.facet.ddf) [concrete]
// CHECK:STDOUT:   %MulAssignWith.assoc_type.26b: type = assoc_entity_type @MulAssignWith, @MulAssignWith(%D) [concrete]
// CHECK:STDOUT:   %assoc0.c2b: %MulAssignWith.assoc_type.26b = assoc_entity element0, imports.%Core.import_ref.c11 [concrete]
// CHECK:STDOUT:   %a.param_patt.d0e: %pattern_type.331 = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt.ec5: %pattern_type.331 = wrapper_binding_pattern a, %a.param_patt.d0e [concrete]
// CHECK:STDOUT:   %b.param_patt.110: %pattern_type.331 = value_param_pattern [concrete]
// CHECK:STDOUT:   %b.patt.14c: %pattern_type.331 = wrapper_binding_pattern b, %b.param_patt.110 [concrete]
// CHECK:STDOUT:   %.891: type = fn_type_with_self_type %MulWith.WithSelf.Op.type.cbd, %MulWith.facet.e7a [concrete]
// CHECK:STDOUT:   %ptr.4a5: type = ptr_type %D [concrete]
// CHECK:STDOUT:   %pattern_type.ceb: type = pattern_type %ptr.4a5 [concrete]
// CHECK:STDOUT:   %a.param_patt.609: %pattern_type.ceb = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt.fac: %pattern_type.ceb = wrapper_binding_pattern a, %a.param_patt.609 [concrete]
// CHECK:STDOUT:   %.dcb: type = fn_type_with_self_type %MulAssignWith.WithSelf.Op.type.523, %MulAssignWith.facet.ddf [concrete]
// CHECK:STDOUT:   %pattern_type.9a587c.2: type = pattern_type %Mul.type [concrete]
// CHECK:STDOUT:   %T.patt.3a0592.1: %pattern_type.9a587c.2 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.f45530.1: %Mul.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.190bb7.1: type = facet_access_type %T.f45530.1 [symbolic]
// CHECK:STDOUT:   %pattern_type.86a9db.1: type = pattern_type %T.as_type.190bb7.1 [symbolic]
// CHECK:STDOUT:   %a.param_patt.e33: %pattern_type.86a9db.1 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %a.patt.e02: %pattern_type.86a9db.1 = wrapper_binding_pattern a, %a.param_patt.e33 [symbolic]
// CHECK:STDOUT:   %b.param_patt.4cd741.1: %pattern_type.86a9db.1 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %b.patt.60acf6.1: %pattern_type.86a9db.1 = wrapper_binding_pattern b, %b.param_patt.4cd741.1 [symbolic]
// CHECK:STDOUT:   %.1f0: Core.Form = init_form %T.as_type.190bb7.1 [symbolic]
// CHECK:STDOUT:   %return.param_patt.5f6: %pattern_type.86a9db.1 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.96a: %pattern_type.86a9db.1 = return_slot_pattern %return.param_patt.5f6, %T.as_type.190bb7.1 [symbolic]
// CHECK:STDOUT:   %MulWith.type.4cc99f.2: type = facet_type <@MulWith, @MulWith(%T.as_type.190bb7.1)> [symbolic]
// CHECK:STDOUT:   %MulWith.assoc_type.7e2ea5.2: type = assoc_entity_type @MulWith, @MulWith(%T.as_type.190bb7.1) [symbolic]
// CHECK:STDOUT:   %assoc1.77a6ba.2: %MulWith.assoc_type.7e2ea5.2 = assoc_entity element1, imports.%Core.import_ref.fc2 [symbolic]
// CHECK:STDOUT:   %require_complete.f57: <witness> = require_complete_type %MulWith.type.4cc99f.2 [symbolic]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.f1f681.2: <witness> = lookup_impl_witness %T.f45530.1, @MulWith, @MulWith(%T.as_type.190bb7.1) [symbolic]
// CHECK:STDOUT:   %MulWith.facet.d92: %MulWith.type.4cc99f.2 = facet_value %T.as_type.190bb7.1, (%MulWith.lookup_impl_witness.f1f681.2) [symbolic]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.07f: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%T.as_type.190bb7.1, %MulWith.facet.d92) [symbolic]
// CHECK:STDOUT:   %.d69: type = fn_type_with_self_type %MulWith.WithSelf.Op.type.07f, %MulWith.facet.d92 [symbolic]
// CHECK:STDOUT:   %impl.elem1.1c4: %.d69 = impl_witness_access %MulWith.lookup_impl_witness.f1f681.2, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.bdf: <specific function> = specific_impl_function %impl.elem1.1c4, @MulWith.WithSelf.Op(%T.as_type.190bb7.1, %MulWith.facet.d92) [symbolic]
// CHECK:STDOUT:   %pattern_type.9a587c.3: type = pattern_type %MulAssign.type [concrete]
// CHECK:STDOUT:   %T.patt.3a0592.2: %pattern_type.9a587c.3 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.f45530.2: %MulAssign.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.190bb7.2: type = facet_access_type %T.f45530.2 [symbolic]
// CHECK:STDOUT:   %ptr.360: type = ptr_type %T.as_type.190bb7.2 [symbolic]
// CHECK:STDOUT:   %pattern_type.ff5: type = pattern_type %ptr.360 [symbolic]
// CHECK:STDOUT:   %a.param_patt.f50: %pattern_type.ff5 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %a.patt.193: %pattern_type.ff5 = wrapper_binding_pattern a, %a.param_patt.f50 [symbolic]
// CHECK:STDOUT:   %pattern_type.86a9db.2: type = pattern_type %T.as_type.190bb7.2 [symbolic]
// CHECK:STDOUT:   %b.param_patt.4cd741.2: %pattern_type.86a9db.2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %b.patt.60acf6.2: %pattern_type.86a9db.2 = wrapper_binding_pattern b, %b.param_patt.4cd741.2 [symbolic]
// CHECK:STDOUT:   %MulAssignWith.type.995568.2: type = facet_type <@MulAssignWith, @MulAssignWith(%T.as_type.190bb7.2)> [symbolic]
// CHECK:STDOUT:   %MulAssignWith.assoc_type.3ae895.2: type = assoc_entity_type @MulAssignWith, @MulAssignWith(%T.as_type.190bb7.2) [symbolic]
// CHECK:STDOUT:   %assoc0.e8f545.2: %MulAssignWith.assoc_type.3ae895.2 = assoc_entity element0, imports.%Core.import_ref.c11 [symbolic]
// CHECK:STDOUT:   %require_complete.c642d9.2: <witness> = require_complete_type %MulAssignWith.type.995568.2 [symbolic]
// CHECK:STDOUT:   %MulAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.f45530.2, @MulAssignWith, @MulAssignWith(%T.as_type.190bb7.2) [symbolic]
// CHECK:STDOUT:   %MulAssignWith.facet.961: %MulAssignWith.type.995568.2 = facet_value %T.as_type.190bb7.2, (%MulAssignWith.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type.631: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%T.as_type.190bb7.2, %MulAssignWith.facet.961) [symbolic]
// CHECK:STDOUT:   %.8bfa: type = fn_type_with_self_type %MulAssignWith.WithSelf.Op.type.631, %MulAssignWith.facet.961 [symbolic]
// CHECK:STDOUT:   %impl.elem0.3dc: %.8bfa = impl_witness_access %MulAssignWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.d73: <specific function> = specific_impl_function %impl.elem0.3dc, @MulAssignWith.WithSelf.Op(%T.as_type.190bb7.2, %MulAssignWith.facet.961) [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %Int.d4bbe0.1: type = class_type @Int, @Int(%N) [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %ptr.a17: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %ptr.a17 [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %Mul.facet.e08: %Mul.type = facet_value %C, (%MulWith.impl_witness.278) [concrete]
// CHECK:STDOUT:   %MulAssign.facet.60f: %MulAssign.type = facet_value %C, (%MulAssignWith.impl_witness.7ee) [concrete]
// CHECK:STDOUT:   %Mul.facet.6f9: %Mul.type = facet_value %D, (%MulWith.impl_witness.736) [concrete]
// CHECK:STDOUT:   %MulAssign.facet.6f9: %MulAssign.type = facet_value %D, (%MulAssignWith.impl_witness.736) [concrete]
// CHECK:STDOUT:   %Int.as.MulWith.impl.Op.type.994: type = fn_type @Int.as.MulWith.impl.Op.1, @Int.as.MulWith.impl.750(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.MulWith.impl.Op.319: %Int.as.MulWith.impl.Op.type.994 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.81c7d0.1: type = facet_type <@ImplicitAs, @ImplicitAs(%Int.d4bbe0.1)> [symbolic]
// CHECK:STDOUT:   %U.f16: %ImplicitAs.type.81c7d0.1 = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %MulWith.type.8c1: type = facet_type <@MulWith, @MulWith(%i32)> [concrete]
// CHECK:STDOUT:   %MulWith.impl_witness.481: <witness> = impl_witness imports.%MulWith.impl_witness_table.fe8, @Int.as.MulWith.impl.750(%int_32) [concrete]
// CHECK:STDOUT:   %MulWith.assoc_type.dd0: type = assoc_entity_type @MulWith, @MulWith(%i32) [concrete]
// CHECK:STDOUT:   %assoc1.072: %MulWith.assoc_type.dd0 = assoc_entity element1, imports.%Core.import_ref.fc2 [concrete]
// CHECK:STDOUT:   %complete_type.50e: <witness> = complete_type_witness %MulWith.type.8c1 [concrete]
// CHECK:STDOUT:   %Int.as.MulWith.impl.Op.type.594: type = fn_type @Int.as.MulWith.impl.Op.1, @Int.as.MulWith.impl.750(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.MulWith.impl.Op.8f2: %Int.as.MulWith.impl.Op.type.594 = struct_value () [concrete]
// CHECK:STDOUT:   %Mul.facet.201: %Mul.type = facet_value %i32, (%MulWith.impl_witness.481) [concrete]
// CHECK:STDOUT:   %a.param_patt.076: %pattern_type.d94 = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt.96f: %pattern_type.d94 = wrapper_binding_pattern a, %a.param_patt.076 [concrete]
// CHECK:STDOUT:   %b.param_patt.1a2: %pattern_type.d94 = value_param_pattern [concrete]
// CHECK:STDOUT:   %b.patt.3ca: %pattern_type.d94 = wrapper_binding_pattern b, %b.param_patt.1a2 [concrete]
// CHECK:STDOUT:   %.0b3: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.2999: %pattern_type.d94 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.6fa: %pattern_type.d94 = return_slot_pattern %return.param_patt.2999, %i32 [concrete]
// CHECK:STDOUT:   %Int.as.MulAssignWith.impl.Op.type.bfafd7.1: type = fn_type @Int.as.MulAssignWith.impl.Op.1, @Int.as.MulAssignWith.impl(%N, %U.f16) [symbolic]
// CHECK:STDOUT:   %Int.as.MulAssignWith.impl.Op.d5bc95.1: %Int.as.MulAssignWith.impl.Op.type.bfafd7.1 = struct_value () [symbolic]
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
// CHECK:STDOUT:   %MulAssignWith.type.a99: type = facet_type <@MulAssignWith, @MulAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %MulAssignWith.impl_witness.4ab: <witness> = impl_witness imports.%MulAssignWith.impl_witness_table, @Int.as.MulAssignWith.impl(%int_32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %MulAssign.facet.e14: %MulAssign.type = facet_value %i32, (%MulAssignWith.impl_witness.4ab) [concrete]
// CHECK:STDOUT:   %a.param_patt.977: %pattern_type.7dc = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt.b70: %pattern_type.7dc = wrapper_binding_pattern a, %a.param_patt.977 [concrete]
// CHECK:STDOUT:   %complete_type.9bb: <witness> = complete_type_witness %MulWith.type.883 [concrete]
// CHECK:STDOUT:   %complete_type.a56: <witness> = complete_type_witness %ptr.00e [concrete]
// CHECK:STDOUT:   %complete_type.fb2: <witness> = complete_type_witness %MulAssignWith.type.24f [concrete]
// CHECK:STDOUT:   %complete_type.230: <witness> = complete_type_witness %MulWith.type.ac3 [concrete]
// CHECK:STDOUT:   %complete_type.320: <witness> = complete_type_witness %ptr.4a5 [concrete]
// CHECK:STDOUT:   %complete_type.f31: <witness> = complete_type_witness %MulAssignWith.type.2e2 [concrete]
// CHECK:STDOUT:   %MulWith.facet.7a6: %MulWith.type.8c1 = facet_value %i32, (%MulWith.impl_witness.481) [concrete]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.e31: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%i32, %MulWith.facet.7a6) [concrete]
// CHECK:STDOUT:   %.4d2: type = fn_type_with_self_type %MulWith.WithSelf.Op.type.e31, %MulWith.facet.7a6 [concrete]
// CHECK:STDOUT:   %Int.as.MulWith.impl.Op.specific_fn.94c: <specific function> = specific_function %Int.as.MulWith.impl.Op.8f2, @Int.as.MulWith.impl.Op.1(%int_32) [concrete]
// CHECK:STDOUT:   %complete_type.1b1: <witness> = complete_type_witness %ptr.a17 [concrete]
// CHECK:STDOUT:   %MulAssignWith.assoc_type.fc8: type = assoc_entity_type @MulAssignWith, @MulAssignWith(%i32) [concrete]
// CHECK:STDOUT:   %assoc0.f84: %MulAssignWith.assoc_type.fc8 = assoc_entity element0, imports.%Core.import_ref.c11 [concrete]
// CHECK:STDOUT:   %complete_type.044: <witness> = complete_type_witness %MulAssignWith.type.a99 [concrete]
// CHECK:STDOUT:   %MulAssignWith.facet.630: %MulAssignWith.type.a99 = facet_value %i32, (%MulAssignWith.impl_witness.4ab) [concrete]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type.3c4: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%i32, %MulAssignWith.facet.630) [concrete]
// CHECK:STDOUT:   %.d46: type = fn_type_with_self_type %MulAssignWith.WithSelf.Op.type.3c4, %MulAssignWith.facet.630 [concrete]
// CHECK:STDOUT:   %Int.as.MulAssignWith.impl.Op.type.0e2e2d.2: type = fn_type @Int.as.MulAssignWith.impl.Op.1, @Int.as.MulAssignWith.impl(%int_32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Int.as.MulAssignWith.impl.Op.7939a1.2: %Int.as.MulAssignWith.impl.Op.type.0e2e2d.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.MulAssignWith.impl.Op.specific_fn.45bc50.1: <specific function> = specific_function %Int.as.MulAssignWith.impl.Op.7939a1.2, @Int.as.MulAssignWith.impl.Op.1(%int_32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.0a8: @MulWith.WithSelf.%MulWith.assoc_type (%MulWith.assoc_type.c81) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @MulWith.WithSelf.%assoc1 (constants.%assoc1.d9e)]
// CHECK:STDOUT:   %Core.import_ref.fc2: @MulWith.WithSelf.%MulWith.WithSelf.Op.type (%MulWith.WithSelf.Op.type.dad) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @MulWith.WithSelf.%MulWith.WithSelf.Op (constants.%MulWith.WithSelf.Op.507)]
// CHECK:STDOUT:   %Core.import_ref.575: @MulAssignWith.WithSelf.%MulAssignWith.assoc_type (%MulAssignWith.assoc_type.5ad) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @MulAssignWith.WithSelf.%assoc0 (constants.%assoc0.911)]
// CHECK:STDOUT:   %Core.import_ref.c11: @MulAssignWith.WithSelf.%MulAssignWith.WithSelf.Op.type (%MulAssignWith.WithSelf.Op.type.c40) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @MulAssignWith.WithSelf.%MulAssignWith.WithSelf.Op (constants.%MulAssignWith.WithSelf.Op.b36)]
// CHECK:STDOUT:   %Core.import_ref.29d = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.d9a = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.ebd: type = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.MulWith.impl.750.%Int (constants.%Int.d4bbe0.1)]
// CHECK:STDOUT:   %Core.import_ref.aa6d: @Int.as.MulWith.impl.750.%Int.as.MulWith.impl.Op.type (%Int.as.MulWith.impl.Op.type.994) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.MulWith.impl.750.%Int.as.MulWith.impl.Op (constants.%Int.as.MulWith.impl.Op.319)]
// CHECK:STDOUT:   %MulWith.impl_witness_table.fe8 = impl_witness_table (%Core.import_ref.ebd, %Core.import_ref.aa6d), @Int.as.MulWith.impl.750 [concrete]
// CHECK:STDOUT:   %Core.import_ref.e3f: @Int.as.MulAssignWith.impl.%Int.as.MulAssignWith.impl.Op.type.2 (%Int.as.MulAssignWith.impl.Op.type.bfafd7.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.MulAssignWith.impl.%Int.as.MulAssignWith.impl.Op.2 (constants.%Int.as.MulAssignWith.impl.Op.d5bc95.1)]
// CHECK:STDOUT:   %MulAssignWith.impl_witness_table = impl_witness_table (%Core.import_ref.e3f), @Int.as.MulAssignWith.impl [concrete]
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
// CHECK:STDOUT:   %impl.elem1: %.387 = impl_witness_access constants.%MulWith.impl_witness.278, element1 [concrete = constants.%C.as.MulWith.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.ref, %impl.elem1
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.as.MulWith.impl.Op.call: init %C to %.loc30_26.1 = call %bound_method(%a.ref, %b.ref)
// CHECK:STDOUT:   return %C.as.MulWith.impl.Op.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestAssign(%a.param: %ptr.00e, %b.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %ptr.00e = name_ref a, %a
// CHECK:STDOUT:   %.loc38: ref %C = deref %a.ref
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0: %.d01 = impl_witness_access constants.%MulAssignWith.impl_witness.7ee, element0 [concrete = constants.%C.as.MulAssignWith.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc38, %impl.elem0
// CHECK:STDOUT:   %C.as.MulAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method(%.loc38, %b.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestNamedConstraintOp(%a.param: %D, %b.param: %D) -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %D = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %D = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1: %.891 = impl_witness_access constants.%MulWith.impl_witness.736, element1 [concrete = constants.%D.as.MulWith.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.ref, %impl.elem1
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %D.as.MulWith.impl.Op.call: init %D to %.loc53_41.1 = call %bound_method(%a.ref, %b.ref)
// CHECK:STDOUT:   return %D.as.MulWith.impl.Op.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestNamedConstraintAssign(%a.param: %ptr.4a5, %b.param: %D) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %ptr.4a5 = name_ref a, %a
// CHECK:STDOUT:   %.loc61: ref %D = deref %a.ref
// CHECK:STDOUT:   %b.ref: %D = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0: %.dcb = impl_witness_access constants.%MulAssignWith.impl_witness.736, element0 [concrete = constants.%D.as.MulAssignWith.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc61, %impl.elem0
// CHECK:STDOUT:   %D.as.MulAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method(%.loc61, %b.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestGenericOp(%T.loc65_19.2: %Mul.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %MulWith.type.loc67_12.2: type = facet_type <@MulWith, @MulWith(%T.as_type.loc65_34.1)> [symbolic = %MulWith.type.loc67_12.2 (constants.%MulWith.type.4cc99f.2)]
// CHECK:STDOUT:   %require_complete.loc67: <witness> = require_complete_type %MulWith.type.loc67_12.2 [symbolic = %require_complete.loc67 (constants.%require_complete.f57)]
// CHECK:STDOUT:   %MulWith.assoc_type: type = assoc_entity_type @MulWith, @MulWith(%T.as_type.loc65_34.1) [symbolic = %MulWith.assoc_type (constants.%MulWith.assoc_type.7e2ea5.2)]
// CHECK:STDOUT:   %assoc1: @TestGenericOp.%MulWith.assoc_type (%MulWith.assoc_type.7e2ea5.2) = assoc_entity element1, imports.%Core.import_ref.fc2 [symbolic = %assoc1 (constants.%assoc1.77a6ba.2)]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc65_19.1, @MulWith, @MulWith(%T.as_type.loc65_34.1) [symbolic = %MulWith.lookup_impl_witness (constants.%MulWith.lookup_impl_witness.f1f681.2)]
// CHECK:STDOUT:   %MulWith.facet: @TestGenericOp.%MulWith.type.loc67_12.2 (%MulWith.type.4cc99f.2) = facet_value %T.as_type.loc65_34.1, (%MulWith.lookup_impl_witness) [symbolic = %MulWith.facet (constants.%MulWith.facet.d92)]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%T.as_type.loc65_34.1, %MulWith.facet) [symbolic = %MulWith.WithSelf.Op.type (constants.%MulWith.WithSelf.Op.type.07f)]
// CHECK:STDOUT:   %.loc67_12.2: type = fn_type_with_self_type %MulWith.WithSelf.Op.type, %MulWith.facet [symbolic = %.loc67_12.2 (constants.%.d69)]
// CHECK:STDOUT:   %impl.elem1.loc67_12.2: @TestGenericOp.%.loc67_12.2 (%.d69) = impl_witness_access %MulWith.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc67_12.2 (constants.%impl.elem1.1c4)]
// CHECK:STDOUT:   %specific_impl_fn.loc67_12.2: <specific function> = specific_impl_function %impl.elem1.loc67_12.2, @MulWith.WithSelf.Op(%T.as_type.loc65_34.1, %MulWith.facet) [symbolic = %specific_impl_fn.loc67_12.2 (constants.%specific_impl_fn.bdf)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @TestGenericOp.%T.as_type.loc65_34.1 (%T.as_type.190bb7.1), %b.param: @TestGenericOp.%T.as_type.loc65_34.1 (%T.as_type.190bb7.1)) -> out %return.param: @TestGenericOp.%T.as_type.loc65_34.1 (%T.as_type.190bb7.1) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %a.ref: @TestGenericOp.%T.as_type.loc65_34.1 (%T.as_type.190bb7.1) = name_ref a, %a
// CHECK:STDOUT:     %b.ref: @TestGenericOp.%T.as_type.loc65_34.1 (%T.as_type.190bb7.1) = name_ref b, %b
// CHECK:STDOUT:     %MulWith.type.loc67_12.1: type = facet_type <@MulWith, @MulWith(constants.%T.as_type.190bb7.1)> [symbolic = %MulWith.type.loc67_12.2 (constants.%MulWith.type.4cc99f.2)]
// CHECK:STDOUT:     %.loc67_12.1: @TestGenericOp.%MulWith.assoc_type (%MulWith.assoc_type.7e2ea5.2) = specific_constant imports.%Core.import_ref.0a8, @MulWith.WithSelf(constants.%T.as_type.190bb7.1, constants.%Self.742) [symbolic = %assoc1 (constants.%assoc1.77a6ba.2)]
// CHECK:STDOUT:     %Op.ref: @TestGenericOp.%MulWith.assoc_type (%MulWith.assoc_type.7e2ea5.2) = name_ref Op, %.loc67_12.1 [symbolic = %assoc1 (constants.%assoc1.77a6ba.2)]
// CHECK:STDOUT:     %impl.elem1.loc67_12.1: @TestGenericOp.%.loc67_12.2 (%.d69) = impl_witness_access constants.%MulWith.lookup_impl_witness.f1f681.2, element1 [symbolic = %impl.elem1.loc67_12.2 (constants.%impl.elem1.1c4)]
// CHECK:STDOUT:     %bound_method.loc67_12.1: <bound method> = bound_method %a.ref, %impl.elem1.loc67_12.1
// CHECK:STDOUT:     %specific_impl_fn.loc67_12.1: <specific function> = specific_impl_function %impl.elem1.loc67_12.1, @MulWith.WithSelf.Op(constants.%T.as_type.190bb7.1, constants.%MulWith.facet.d92) [symbolic = %specific_impl_fn.loc67_12.2 (constants.%specific_impl_fn.bdf)]
// CHECK:STDOUT:     %bound_method.loc67_12.2: <bound method> = bound_method %a.ref, %specific_impl_fn.loc67_12.1
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %MulWith.WithSelf.Op.call: init @TestGenericOp.%T.as_type.loc65_34.1 (%T.as_type.190bb7.1) to %.loc65_46.1 = call %bound_method.loc67_12.2(%a.ref, %b.ref)
// CHECK:STDOUT:     return %MulWith.WithSelf.Op.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestGenericAssign(%T.loc71_23.2: %MulAssign.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %MulAssignWith.type.loc73_6.2: type = facet_type <@MulAssignWith, @MulAssignWith(%T.as_type.loc71_45.1)> [symbolic = %MulAssignWith.type.loc73_6.2 (constants.%MulAssignWith.type.995568.2)]
// CHECK:STDOUT:   %require_complete.loc73: <witness> = require_complete_type %MulAssignWith.type.loc73_6.2 [symbolic = %require_complete.loc73 (constants.%require_complete.c642d9.2)]
// CHECK:STDOUT:   %MulAssignWith.assoc_type: type = assoc_entity_type @MulAssignWith, @MulAssignWith(%T.as_type.loc71_45.1) [symbolic = %MulAssignWith.assoc_type (constants.%MulAssignWith.assoc_type.3ae895.2)]
// CHECK:STDOUT:   %assoc0: @TestGenericAssign.%MulAssignWith.assoc_type (%MulAssignWith.assoc_type.3ae895.2) = assoc_entity element0, imports.%Core.import_ref.c11 [symbolic = %assoc0 (constants.%assoc0.e8f545.2)]
// CHECK:STDOUT:   %MulAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc71_23.1, @MulAssignWith, @MulAssignWith(%T.as_type.loc71_45.1) [symbolic = %MulAssignWith.lookup_impl_witness (constants.%MulAssignWith.lookup_impl_witness)]
// CHECK:STDOUT:   %MulAssignWith.facet: @TestGenericAssign.%MulAssignWith.type.loc73_6.2 (%MulAssignWith.type.995568.2) = facet_value %T.as_type.loc71_45.1, (%MulAssignWith.lookup_impl_witness) [symbolic = %MulAssignWith.facet (constants.%MulAssignWith.facet.961)]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%T.as_type.loc71_45.1, %MulAssignWith.facet) [symbolic = %MulAssignWith.WithSelf.Op.type (constants.%MulAssignWith.WithSelf.Op.type.631)]
// CHECK:STDOUT:   %.loc73_6.2: type = fn_type_with_self_type %MulAssignWith.WithSelf.Op.type, %MulAssignWith.facet [symbolic = %.loc73_6.2 (constants.%.8bfa)]
// CHECK:STDOUT:   %impl.elem0.loc73_6.2: @TestGenericAssign.%.loc73_6.2 (%.8bfa) = impl_witness_access %MulAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc73_6.2 (constants.%impl.elem0.3dc)]
// CHECK:STDOUT:   %specific_impl_fn.loc73_6.2: <specific function> = specific_impl_function %impl.elem0.loc73_6.2, @MulAssignWith.WithSelf.Op(%T.as_type.loc71_45.1, %MulAssignWith.facet) [symbolic = %specific_impl_fn.loc73_6.2 (constants.%specific_impl_fn.d73)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @TestGenericAssign.%ptr.loc71_45.1 (%ptr.360), %b.param: @TestGenericAssign.%T.as_type.loc71_45.1 (%T.as_type.190bb7.2)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %a.ref: @TestGenericAssign.%ptr.loc71_45.1 (%ptr.360) = name_ref a, %a
// CHECK:STDOUT:     %.loc73_3: ref @TestGenericAssign.%T.as_type.loc71_45.1 (%T.as_type.190bb7.2) = deref %a.ref
// CHECK:STDOUT:     %b.ref: @TestGenericAssign.%T.as_type.loc71_45.1 (%T.as_type.190bb7.2) = name_ref b, %b
// CHECK:STDOUT:     %MulAssignWith.type.loc73_6.1: type = facet_type <@MulAssignWith, @MulAssignWith(constants.%T.as_type.190bb7.2)> [symbolic = %MulAssignWith.type.loc73_6.2 (constants.%MulAssignWith.type.995568.2)]
// CHECK:STDOUT:     %.loc73_6.1: @TestGenericAssign.%MulAssignWith.assoc_type (%MulAssignWith.assoc_type.3ae895.2) = specific_constant imports.%Core.import_ref.575, @MulAssignWith.WithSelf(constants.%T.as_type.190bb7.2, constants.%Self.9e4) [symbolic = %assoc0 (constants.%assoc0.e8f545.2)]
// CHECK:STDOUT:     %Op.ref: @TestGenericAssign.%MulAssignWith.assoc_type (%MulAssignWith.assoc_type.3ae895.2) = name_ref Op, %.loc73_6.1 [symbolic = %assoc0 (constants.%assoc0.e8f545.2)]
// CHECK:STDOUT:     %impl.elem0.loc73_6.1: @TestGenericAssign.%.loc73_6.2 (%.8bfa) = impl_witness_access constants.%MulAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc73_6.2 (constants.%impl.elem0.3dc)]
// CHECK:STDOUT:     %bound_method.loc73_6.1: <bound method> = bound_method %.loc73_3, %impl.elem0.loc73_6.1
// CHECK:STDOUT:     %specific_impl_fn.loc73_6.1: <specific function> = specific_impl_function %impl.elem0.loc73_6.1, @MulAssignWith.WithSelf.Op(constants.%T.as_type.190bb7.2, constants.%MulAssignWith.facet.961) [symbolic = %specific_impl_fn.loc73_6.2 (constants.%specific_impl_fn.d73)]
// CHECK:STDOUT:     %bound_method.loc73_6.2: <bound method> = bound_method %.loc73_3, %specific_impl_fn.loc73_6.1
// CHECK:STDOUT:     %MulAssignWith.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc73_6.2(%.loc73_3, %b.ref)
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
// CHECK:STDOUT: specific @TestGenericOp(constants.%Mul.facet.e08) {
// CHECK:STDOUT:   %T.patt.loc65_19.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc65_19.1 => constants.%Mul.facet.e08
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
// CHECK:STDOUT:   %MulWith.type.loc67_12.2 => constants.%MulWith.type.883
// CHECK:STDOUT:   %require_complete.loc67 => constants.%complete_type.9bb
// CHECK:STDOUT:   %MulWith.assoc_type => constants.%MulWith.assoc_type.813
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.2e6
// CHECK:STDOUT:   %MulWith.lookup_impl_witness => constants.%MulWith.impl_witness.278
// CHECK:STDOUT:   %MulWith.facet => constants.%MulWith.facet.5f2
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type => constants.%MulWith.WithSelf.Op.type.03e
// CHECK:STDOUT:   %.loc67_12.2 => constants.%.387
// CHECK:STDOUT:   %impl.elem1.loc67_12.2 => constants.%C.as.MulWith.impl.Op
// CHECK:STDOUT:   %specific_impl_fn.loc67_12.2 => constants.%C.as.MulWith.impl.Op
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGenericAssign(constants.%MulAssign.facet.60f) {
// CHECK:STDOUT:   %T.patt.loc71_23.2 => constants.%T.patt.3a0592.2
// CHECK:STDOUT:   %T.loc71_23.1 => constants.%MulAssign.facet.60f
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
// CHECK:STDOUT:   %MulAssignWith.type.loc73_6.2 => constants.%MulAssignWith.type.24f
// CHECK:STDOUT:   %require_complete.loc73 => constants.%complete_type.fb2
// CHECK:STDOUT:   %MulAssignWith.assoc_type => constants.%MulAssignWith.assoc_type.06c
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.187
// CHECK:STDOUT:   %MulAssignWith.lookup_impl_witness => constants.%MulAssignWith.impl_witness.7ee
// CHECK:STDOUT:   %MulAssignWith.facet => constants.%MulAssignWith.facet.9be
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type => constants.%MulAssignWith.WithSelf.Op.type.534
// CHECK:STDOUT:   %.loc73_6.2 => constants.%.d01
// CHECK:STDOUT:   %impl.elem0.loc73_6.2 => constants.%C.as.MulAssignWith.impl.Op
// CHECK:STDOUT:   %specific_impl_fn.loc73_6.2 => constants.%C.as.MulAssignWith.impl.Op
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGenericOp(constants.%Mul.facet.6f9) {
// CHECK:STDOUT:   %T.patt.loc65_19.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc65_19.1 => constants.%Mul.facet.6f9
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
// CHECK:STDOUT:   %MulWith.type.loc67_12.2 => constants.%MulWith.type.ac3
// CHECK:STDOUT:   %require_complete.loc67 => constants.%complete_type.230
// CHECK:STDOUT:   %MulWith.assoc_type => constants.%MulWith.assoc_type.1e9
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.cc9
// CHECK:STDOUT:   %MulWith.lookup_impl_witness => constants.%MulWith.impl_witness.736
// CHECK:STDOUT:   %MulWith.facet => constants.%MulWith.facet.e7a
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type => constants.%MulWith.WithSelf.Op.type.cbd
// CHECK:STDOUT:   %.loc67_12.2 => constants.%.891
// CHECK:STDOUT:   %impl.elem1.loc67_12.2 => constants.%D.as.MulWith.impl.Op
// CHECK:STDOUT:   %specific_impl_fn.loc67_12.2 => constants.%D.as.MulWith.impl.Op
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGenericAssign(constants.%MulAssign.facet.6f9) {
// CHECK:STDOUT:   %T.patt.loc71_23.2 => constants.%T.patt.3a0592.2
// CHECK:STDOUT:   %T.loc71_23.1 => constants.%MulAssign.facet.6f9
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
// CHECK:STDOUT:   %MulAssignWith.type.loc73_6.2 => constants.%MulAssignWith.type.2e2
// CHECK:STDOUT:   %require_complete.loc73 => constants.%complete_type.f31
// CHECK:STDOUT:   %MulAssignWith.assoc_type => constants.%MulAssignWith.assoc_type.26b
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.c2b
// CHECK:STDOUT:   %MulAssignWith.lookup_impl_witness => constants.%MulAssignWith.impl_witness.736
// CHECK:STDOUT:   %MulAssignWith.facet => constants.%MulAssignWith.facet.ddf
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type => constants.%MulAssignWith.WithSelf.Op.type.523
// CHECK:STDOUT:   %.loc73_6.2 => constants.%.dcb
// CHECK:STDOUT:   %impl.elem0.loc73_6.2 => constants.%D.as.MulAssignWith.impl.Op
// CHECK:STDOUT:   %specific_impl_fn.loc73_6.2 => constants.%D.as.MulAssignWith.impl.Op
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGenericOp(constants.%Mul.facet.201) {
// CHECK:STDOUT:   %T.patt.loc65_19.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc65_19.1 => constants.%Mul.facet.201
// CHECK:STDOUT:   %T.as_type.loc65_34.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.d94
// CHECK:STDOUT:   %a.param_patt.loc65_32.2 => constants.%a.param_patt.076
// CHECK:STDOUT:   %a.patt.loc65_32.2 => constants.%a.patt.96f
// CHECK:STDOUT:   %b.param_patt.loc65_38.2 => constants.%b.param_patt.1a2
// CHECK:STDOUT:   %b.patt.loc65_38.2 => constants.%b.patt.3ca
// CHECK:STDOUT:   %.loc65_46.2 => constants.%.0b3
// CHECK:STDOUT:   %return.param_patt.loc65_46.2 => constants.%return.param_patt.2999
// CHECK:STDOUT:   %return.patt.loc65_43.2 => constants.%return.patt.6fa
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc65 => constants.%complete_type.f8a
// CHECK:STDOUT:   %MulWith.type.loc67_12.2 => constants.%MulWith.type.8c1
// CHECK:STDOUT:   %require_complete.loc67 => constants.%complete_type.50e
// CHECK:STDOUT:   %MulWith.assoc_type => constants.%MulWith.assoc_type.dd0
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.072
// CHECK:STDOUT:   %MulWith.lookup_impl_witness => constants.%MulWith.impl_witness.481
// CHECK:STDOUT:   %MulWith.facet => constants.%MulWith.facet.7a6
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type => constants.%MulWith.WithSelf.Op.type.e31
// CHECK:STDOUT:   %.loc67_12.2 => constants.%.4d2
// CHECK:STDOUT:   %impl.elem1.loc67_12.2 => constants.%Int.as.MulWith.impl.Op.8f2
// CHECK:STDOUT:   %specific_impl_fn.loc67_12.2 => constants.%Int.as.MulWith.impl.Op.specific_fn.94c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGenericAssign(constants.%MulAssign.facet.e14) {
// CHECK:STDOUT:   %T.patt.loc71_23.2 => constants.%T.patt.3a0592.2
// CHECK:STDOUT:   %T.loc71_23.1 => constants.%MulAssign.facet.e14
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
// CHECK:STDOUT:   %MulAssignWith.type.loc73_6.2 => constants.%MulAssignWith.type.a99
// CHECK:STDOUT:   %require_complete.loc73 => constants.%complete_type.044
// CHECK:STDOUT:   %MulAssignWith.assoc_type => constants.%MulAssignWith.assoc_type.fc8
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.f84
// CHECK:STDOUT:   %MulAssignWith.lookup_impl_witness => constants.%MulAssignWith.impl_witness.4ab
// CHECK:STDOUT:   %MulAssignWith.facet => constants.%MulAssignWith.facet.630
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type => constants.%MulAssignWith.WithSelf.Op.type.3c4
// CHECK:STDOUT:   %.loc73_6.2 => constants.%.d46
// CHECK:STDOUT:   %impl.elem0.loc73_6.2 => constants.%Int.as.MulAssignWith.impl.Op.7939a1.2
// CHECK:STDOUT:   %specific_impl_fn.loc73_6.2 => constants.%Int.as.MulAssignWith.impl.Op.specific_fn.45bc50.1
// CHECK:STDOUT: }
// CHECK:STDOUT:

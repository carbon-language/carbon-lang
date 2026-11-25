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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/overloaded/add.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/overloaded/add.carbon

// This file was generated from binary_op.carbon.tmpl. Run make_tests.sh to regenerate.

package User;

class C {};

impl C as Core.AddWith(C) where .Result = C {
  fn Op(unused self: C, unused other: C) -> C {
    return {};
  }
}
impl C as Core.AddAssignWith(C) {
  fn Op(unused ref self: C, unused other: C) {}
}

fn TestOp(a: C, b: C) -> C {
  //@dump-sem-ir-begin
  return a + b;
  //@dump-sem-ir-end
}

fn TestAssign(a: C*, b: C) {
  //@dump-sem-ir-begin
  *a += b;
  //@dump-sem-ir-end
}

class D {};

impl D as Core.Add {
  fn Op(unused self: D, unused other: D) -> D {
    return {};
  }
}
impl D as Core.AddAssign {
  fn Op(unused ref self: D, unused other: D) {}
}

fn TestNamedConstraintOp(a: D, b: D) -> D {
  //@dump-sem-ir-begin
  return a + b;
  //@dump-sem-ir-end
}

fn TestNamedConstraintAssign(a: D*, b: D) {
  //@dump-sem-ir-begin
  *a += b;
  //@dump-sem-ir-end
}

fn TestGenericOp[T: Core.Add](a: T, b: T) -> T {
  //@dump-sem-ir-begin
  return a + b;
  //@dump-sem-ir-end
}

fn TestGenericAssign[T: Core.AddAssign](a: T*, b: T) {
  //@dump-sem-ir-begin
  *a += b;
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

// CHECK:STDOUT: --- add.carbon = $add
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Other: type = symbolic_binding Other, 0 [symbolic]
// CHECK:STDOUT:   %AddWith.type.872: type = facet_type <@AddWith, @AddWith(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.958: %AddWith.type.872 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %AddWith.assoc_type.1d5: type = assoc_entity_type @AddWith, @AddWith(%Other) [symbolic]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.70b: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%Other, %Self.958) [symbolic]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.bb1: %AddWith.WithSelf.Op.type.70b = struct_value () [symbolic]
// CHECK:STDOUT:   %AddWith.type.f3e: type = facet_type <@AddWith, @AddWith(%C)> [concrete]
// CHECK:STDOUT:   %AddWith.assoc_type.d7a: type = assoc_entity_type @AddWith, @AddWith(%C) [concrete]
// CHECK:STDOUT:   %assoc1.546: %AddWith.assoc_type.d7a = assoc_entity element1, imports.%Core.import_ref.8ad [concrete]
// CHECK:STDOUT:   %AddWith.impl_witness.e33: <witness> = impl_witness @C.as.AddWith.impl.%AddWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.4a7: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %.79c: Core.Form = init_form %C [concrete]
// CHECK:STDOUT:   %return.param_patt.415: %pattern_type.4a7 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.340: %pattern_type.4a7 = return_slot_pattern %return.param_patt.415, %C [concrete]
// CHECK:STDOUT:   %C.as.AddWith.impl.Op.type: type = fn_type @C.as.AddWith.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.AddWith.impl.Op: %C.as.AddWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %AddWith.facet.1f6: %AddWith.type.f3e = facet_value %C, (%AddWith.impl_witness.e33) [concrete]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.97e: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%C, %AddWith.facet.1f6) [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.3a4: type = facet_type <@AddAssignWith, @AddAssignWith(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.73b: %AddAssignWith.type.3a4 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %AddAssignWith.assoc_type.084: type = assoc_entity_type @AddAssignWith, @AddAssignWith(%Other) [symbolic]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.9de: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%Other, %Self.73b) [symbolic]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.cfc: %AddAssignWith.WithSelf.Op.type.9de = struct_value () [symbolic]
// CHECK:STDOUT:   %AddAssignWith.type.cf4: type = facet_type <@AddAssignWith, @AddAssignWith(%C)> [concrete]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.3ff: <witness> = impl_witness @C.as.AddAssignWith.impl.%AddAssignWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %AddAssignWith.assoc_type.a0e: type = assoc_entity_type @AddAssignWith, @AddAssignWith(%C) [concrete]
// CHECK:STDOUT:   %assoc0.da3: %AddAssignWith.assoc_type.a0e = assoc_entity element0, imports.%Core.import_ref.90f [concrete]
// CHECK:STDOUT:   %C.as.AddAssignWith.impl.Op.type: type = fn_type @C.as.AddAssignWith.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.AddAssignWith.impl.Op: %C.as.AddAssignWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet.dd1: %AddAssignWith.type.cf4 = facet_value %C, (%AddAssignWith.impl_witness.3ff) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.d8d: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%C, %AddAssignWith.facet.dd1) [concrete]
// CHECK:STDOUT:   %a.param_patt.2a5: %pattern_type.4a7 = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt.a30: %pattern_type.4a7 = wrapper_binding_pattern a, %a.param_patt.2a5 [concrete]
// CHECK:STDOUT:   %b.param_patt.45f: %pattern_type.4a7 = value_param_pattern [concrete]
// CHECK:STDOUT:   %b.patt.562: %pattern_type.4a7 = wrapper_binding_pattern b, %b.param_patt.45f [concrete]
// CHECK:STDOUT:   %assoc1.84a: %AddWith.assoc_type.1d5 = assoc_entity element1, imports.%Core.import_ref.e9a [symbolic]
// CHECK:STDOUT:   %.da3: type = fn_type_with_self_type %AddWith.WithSelf.Op.type.97e, %AddWith.facet.1f6 [concrete]
// CHECK:STDOUT:   %ptr.00e: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.842: type = pattern_type %ptr.00e [concrete]
// CHECK:STDOUT:   %a.param_patt.4fa: %pattern_type.842 = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt.eba: %pattern_type.842 = wrapper_binding_pattern a, %a.param_patt.4fa [concrete]
// CHECK:STDOUT:   %assoc0.3b5: %AddAssignWith.assoc_type.084 = assoc_entity element0, imports.%Core.import_ref.278 [symbolic]
// CHECK:STDOUT:   %.964: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.d8d, %AddAssignWith.facet.dd1 [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %Add.type: type = facet_type <@Add> [concrete]
// CHECK:STDOUT:   %AddWith.impl_witness.736: <witness> = impl_witness @D.as.AddWith.impl.%AddWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.331: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %.84d: Core.Form = init_form %D [concrete]
// CHECK:STDOUT:   %return.param_patt.835: %pattern_type.331 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.5ef: %pattern_type.331 = return_slot_pattern %return.param_patt.835, %D [concrete]
// CHECK:STDOUT:   %D.as.AddWith.impl.Op.type: type = fn_type @D.as.AddWith.impl.Op [concrete]
// CHECK:STDOUT:   %D.as.AddWith.impl.Op: %D.as.AddWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %AddWith.type.710: type = facet_type <@AddWith, @AddWith(%D)> [concrete]
// CHECK:STDOUT:   %AddWith.facet.f08: %AddWith.type.710 = facet_value %D, (%AddWith.impl_witness.736) [concrete]
// CHECK:STDOUT:   %AddWith.assoc_type.805: type = assoc_entity_type @AddWith, @AddWith(%D) [concrete]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.5f4: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%D, %AddWith.facet.f08) [concrete]
// CHECK:STDOUT:   %assoc1.3fd: %AddWith.assoc_type.805 = assoc_entity element1, imports.%Core.import_ref.8ad [concrete]
// CHECK:STDOUT:   %AddAssign.type: type = facet_type <@AddAssign> [concrete]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.736: <witness> = impl_witness @D.as.AddAssignWith.impl.%AddAssignWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %D.as.AddAssignWith.impl.Op.type: type = fn_type @D.as.AddAssignWith.impl.Op [concrete]
// CHECK:STDOUT:   %D.as.AddAssignWith.impl.Op: %D.as.AddAssignWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.702: type = facet_type <@AddAssignWith, @AddAssignWith(%D)> [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet.c87: %AddAssignWith.type.702 = facet_value %D, (%AddAssignWith.impl_witness.736) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.306: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%D, %AddAssignWith.facet.c87) [concrete]
// CHECK:STDOUT:   %AddAssignWith.assoc_type.fe5: type = assoc_entity_type @AddAssignWith, @AddAssignWith(%D) [concrete]
// CHECK:STDOUT:   %assoc0.928: %AddAssignWith.assoc_type.fe5 = assoc_entity element0, imports.%Core.import_ref.90f [concrete]
// CHECK:STDOUT:   %a.param_patt.d0e: %pattern_type.331 = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt.ec5: %pattern_type.331 = wrapper_binding_pattern a, %a.param_patt.d0e [concrete]
// CHECK:STDOUT:   %b.param_patt.110: %pattern_type.331 = value_param_pattern [concrete]
// CHECK:STDOUT:   %b.patt.14c: %pattern_type.331 = wrapper_binding_pattern b, %b.param_patt.110 [concrete]
// CHECK:STDOUT:   %.2bc: type = fn_type_with_self_type %AddWith.WithSelf.Op.type.5f4, %AddWith.facet.f08 [concrete]
// CHECK:STDOUT:   %ptr.4a5: type = ptr_type %D [concrete]
// CHECK:STDOUT:   %pattern_type.ceb: type = pattern_type %ptr.4a5 [concrete]
// CHECK:STDOUT:   %a.param_patt.609: %pattern_type.ceb = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt.fac: %pattern_type.ceb = wrapper_binding_pattern a, %a.param_patt.609 [concrete]
// CHECK:STDOUT:   %.123: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.306, %AddAssignWith.facet.c87 [concrete]
// CHECK:STDOUT:   %pattern_type.9a587c.2: type = pattern_type %Add.type [concrete]
// CHECK:STDOUT:   %T.patt.3a0592.1: %pattern_type.9a587c.2 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.f45530.1: %Add.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.190bb7.1: type = facet_access_type %T.f45530.1 [symbolic]
// CHECK:STDOUT:   %pattern_type.86a9db.1: type = pattern_type %T.as_type.190bb7.1 [symbolic]
// CHECK:STDOUT:   %a.param_patt.e33: %pattern_type.86a9db.1 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %a.patt.e02: %pattern_type.86a9db.1 = wrapper_binding_pattern a, %a.param_patt.e33 [symbolic]
// CHECK:STDOUT:   %b.param_patt.4cd741.1: %pattern_type.86a9db.1 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %b.patt.60acf6.1: %pattern_type.86a9db.1 = wrapper_binding_pattern b, %b.param_patt.4cd741.1 [symbolic]
// CHECK:STDOUT:   %.1f0: Core.Form = init_form %T.as_type.190bb7.1 [symbolic]
// CHECK:STDOUT:   %return.param_patt.5f6: %pattern_type.86a9db.1 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.96a: %pattern_type.86a9db.1 = return_slot_pattern %return.param_patt.5f6, %T.as_type.190bb7.1 [symbolic]
// CHECK:STDOUT:   %AddWith.type.2309ff.2: type = facet_type <@AddWith, @AddWith(%T.as_type.190bb7.1)> [symbolic]
// CHECK:STDOUT:   %AddWith.assoc_type.d6e422.2: type = assoc_entity_type @AddWith, @AddWith(%T.as_type.190bb7.1) [symbolic]
// CHECK:STDOUT:   %assoc1.857cf7.2: %AddWith.assoc_type.d6e422.2 = assoc_entity element1, imports.%Core.import_ref.8ad [symbolic]
// CHECK:STDOUT:   %require_complete.a92: <witness> = require_complete_type %AddWith.type.2309ff.2 [symbolic]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.82cdf5.2: <witness> = lookup_impl_witness %T.f45530.1, @AddWith, @AddWith(%T.as_type.190bb7.1) [symbolic]
// CHECK:STDOUT:   %AddWith.facet.a58: %AddWith.type.2309ff.2 = facet_value %T.as_type.190bb7.1, (%AddWith.lookup_impl_witness.82cdf5.2) [symbolic]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.593: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%T.as_type.190bb7.1, %AddWith.facet.a58) [symbolic]
// CHECK:STDOUT:   %.50a: type = fn_type_with_self_type %AddWith.WithSelf.Op.type.593, %AddWith.facet.a58 [symbolic]
// CHECK:STDOUT:   %impl.elem1.ce3: %.50a = impl_witness_access %AddWith.lookup_impl_witness.82cdf5.2, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.844: <specific function> = specific_impl_function %impl.elem1.ce3, @AddWith.WithSelf.Op(%T.as_type.190bb7.1, %AddWith.facet.a58) [symbolic]
// CHECK:STDOUT:   %pattern_type.9a587c.3: type = pattern_type %AddAssign.type [concrete]
// CHECK:STDOUT:   %T.patt.3a0592.2: %pattern_type.9a587c.3 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.f45530.2: %AddAssign.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.190bb7.2: type = facet_access_type %T.f45530.2 [symbolic]
// CHECK:STDOUT:   %ptr.360: type = ptr_type %T.as_type.190bb7.2 [symbolic]
// CHECK:STDOUT:   %pattern_type.ff5: type = pattern_type %ptr.360 [symbolic]
// CHECK:STDOUT:   %a.param_patt.f50: %pattern_type.ff5 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %a.patt.193: %pattern_type.ff5 = wrapper_binding_pattern a, %a.param_patt.f50 [symbolic]
// CHECK:STDOUT:   %pattern_type.86a9db.2: type = pattern_type %T.as_type.190bb7.2 [symbolic]
// CHECK:STDOUT:   %b.param_patt.4cd741.2: %pattern_type.86a9db.2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %b.patt.60acf6.2: %pattern_type.86a9db.2 = wrapper_binding_pattern b, %b.param_patt.4cd741.2 [symbolic]
// CHECK:STDOUT:   %AddAssignWith.type.339202.2: type = facet_type <@AddAssignWith, @AddAssignWith(%T.as_type.190bb7.2)> [symbolic]
// CHECK:STDOUT:   %AddAssignWith.assoc_type.0b4a9d.2: type = assoc_entity_type @AddAssignWith, @AddAssignWith(%T.as_type.190bb7.2) [symbolic]
// CHECK:STDOUT:   %assoc0.498608.2: %AddAssignWith.assoc_type.0b4a9d.2 = assoc_entity element0, imports.%Core.import_ref.90f [symbolic]
// CHECK:STDOUT:   %require_complete.58e780.2: <witness> = require_complete_type %AddAssignWith.type.339202.2 [symbolic]
// CHECK:STDOUT:   %AddAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.f45530.2, @AddAssignWith, @AddAssignWith(%T.as_type.190bb7.2) [symbolic]
// CHECK:STDOUT:   %AddAssignWith.facet.2b4: %AddAssignWith.type.339202.2 = facet_value %T.as_type.190bb7.2, (%AddAssignWith.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.396: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%T.as_type.190bb7.2, %AddAssignWith.facet.2b4) [symbolic]
// CHECK:STDOUT:   %.c82: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.396, %AddAssignWith.facet.2b4 [symbolic]
// CHECK:STDOUT:   %impl.elem0.9b7: %.c82 = impl_witness_access %AddAssignWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.d5e: <specific function> = specific_impl_function %impl.elem0.9b7, @AddAssignWith.WithSelf.Op(%T.as_type.190bb7.2, %AddAssignWith.facet.2b4) [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %Int.d4bbe0.1: type = class_type @Int, @Int(%N) [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %ptr.a17: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %ptr.a17 [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %Add.facet.cf6: %Add.type = facet_value %C, (%AddWith.impl_witness.e33) [concrete]
// CHECK:STDOUT:   %AddAssign.facet.e2c: %AddAssign.type = facet_value %C, (%AddAssignWith.impl_witness.3ff) [concrete]
// CHECK:STDOUT:   %Add.facet.6f9: %Add.type = facet_value %D, (%AddWith.impl_witness.736) [concrete]
// CHECK:STDOUT:   %AddAssign.facet.6f9: %AddAssign.type = facet_value %D, (%AddAssignWith.impl_witness.736) [concrete]
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.type.543: type = fn_type @Int.as.AddWith.impl.Op.1, @Int.as.AddWith.impl.d0c(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.9b8: %Int.as.AddWith.impl.Op.type.543 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.81c7d0.1: type = facet_type <@ImplicitAs, @ImplicitAs(%Int.d4bbe0.1)> [symbolic]
// CHECK:STDOUT:   %U.f16: %ImplicitAs.type.81c7d0.1 = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %AddWith.type.ff2: type = facet_type <@AddWith, @AddWith(%i32)> [concrete]
// CHECK:STDOUT:   %AddWith.impl_witness.d3a: <witness> = impl_witness imports.%AddWith.impl_witness_table.174, @Int.as.AddWith.impl.d0c(%int_32) [concrete]
// CHECK:STDOUT:   %AddWith.assoc_type.ddd: type = assoc_entity_type @AddWith, @AddWith(%i32) [concrete]
// CHECK:STDOUT:   %assoc1.6d3: %AddWith.assoc_type.ddd = assoc_entity element1, imports.%Core.import_ref.8ad [concrete]
// CHECK:STDOUT:   %complete_type.00d: <witness> = complete_type_witness %AddWith.type.ff2 [concrete]
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.type.e40: type = fn_type @Int.as.AddWith.impl.Op.1, @Int.as.AddWith.impl.d0c(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.f9e: %Int.as.AddWith.impl.Op.type.e40 = struct_value () [concrete]
// CHECK:STDOUT:   %Add.facet.b9e: %Add.type = facet_value %i32, (%AddWith.impl_witness.d3a) [concrete]
// CHECK:STDOUT:   %a.param_patt.076: %pattern_type.d94 = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt.96f: %pattern_type.d94 = wrapper_binding_pattern a, %a.param_patt.076 [concrete]
// CHECK:STDOUT:   %b.param_patt.1a2: %pattern_type.d94 = value_param_pattern [concrete]
// CHECK:STDOUT:   %b.patt.3ca: %pattern_type.d94 = wrapper_binding_pattern b, %b.param_patt.1a2 [concrete]
// CHECK:STDOUT:   %.0b3: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.299: %pattern_type.d94 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.6fa: %pattern_type.d94 = return_slot_pattern %return.param_patt.299, %i32 [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.type.93140c.1: type = fn_type @Int.as.AddAssignWith.impl.Op.1, @Int.as.AddAssignWith.impl(%N, %U.f16) [symbolic]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.75e08e.1: %Int.as.AddAssignWith.impl.Op.type.93140c.1 = struct_value () [symbolic]
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
// CHECK:STDOUT:   %AddAssignWith.type.1ad: type = facet_type <@AddAssignWith, @AddAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.95a: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table, @Int.as.AddAssignWith.impl(%int_32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %AddAssign.facet.690: %AddAssign.type = facet_value %i32, (%AddAssignWith.impl_witness.95a) [concrete]
// CHECK:STDOUT:   %a.param_patt.977: %pattern_type.7dc = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt.b70: %pattern_type.7dc = wrapper_binding_pattern a, %a.param_patt.977 [concrete]
// CHECK:STDOUT:   %complete_type.18f: <witness> = complete_type_witness %AddWith.type.f3e [concrete]
// CHECK:STDOUT:   %complete_type.a56: <witness> = complete_type_witness %ptr.00e [concrete]
// CHECK:STDOUT:   %complete_type.03e: <witness> = complete_type_witness %AddAssignWith.type.cf4 [concrete]
// CHECK:STDOUT:   %complete_type.2f3: <witness> = complete_type_witness %AddWith.type.710 [concrete]
// CHECK:STDOUT:   %complete_type.320: <witness> = complete_type_witness %ptr.4a5 [concrete]
// CHECK:STDOUT:   %complete_type.53b: <witness> = complete_type_witness %AddAssignWith.type.702 [concrete]
// CHECK:STDOUT:   %AddWith.facet.06d: %AddWith.type.ff2 = facet_value %i32, (%AddWith.impl_witness.d3a) [concrete]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.590: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%i32, %AddWith.facet.06d) [concrete]
// CHECK:STDOUT:   %.abe: type = fn_type_with_self_type %AddWith.WithSelf.Op.type.590, %AddWith.facet.06d [concrete]
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.specific_fn.3df: <specific function> = specific_function %Int.as.AddWith.impl.Op.f9e, @Int.as.AddWith.impl.Op.1(%int_32) [concrete]
// CHECK:STDOUT:   %complete_type.1b1: <witness> = complete_type_witness %ptr.a17 [concrete]
// CHECK:STDOUT:   %AddAssignWith.assoc_type.144: type = assoc_entity_type @AddAssignWith, @AddAssignWith(%i32) [concrete]
// CHECK:STDOUT:   %assoc0.c38: %AddAssignWith.assoc_type.144 = assoc_entity element0, imports.%Core.import_ref.90f [concrete]
// CHECK:STDOUT:   %complete_type.eb5: <witness> = complete_type_witness %AddAssignWith.type.1ad [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet.d53: %AddAssignWith.type.1ad = facet_value %i32, (%AddAssignWith.impl_witness.95a) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.313: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%i32, %AddAssignWith.facet.d53) [concrete]
// CHECK:STDOUT:   %.316: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.313, %AddAssignWith.facet.d53 [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.type.47f4e6.2: type = fn_type @Int.as.AddAssignWith.impl.Op.1, @Int.as.AddAssignWith.impl(%int_32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.bff4f9.2: %Int.as.AddAssignWith.impl.Op.type.47f4e6.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.specific_fn.ec28e6.1: <specific function> = specific_function %Int.as.AddAssignWith.impl.Op.bff4f9.2, @Int.as.AddAssignWith.impl.Op.1(%int_32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.5c64: @AddWith.WithSelf.%AddWith.assoc_type (%AddWith.assoc_type.1d5) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @AddWith.WithSelf.%assoc1 (constants.%assoc1.84a)]
// CHECK:STDOUT:   %Core.import_ref.8ad: @AddWith.WithSelf.%AddWith.WithSelf.Op.type (%AddWith.WithSelf.Op.type.70b) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @AddWith.WithSelf.%AddWith.WithSelf.Op (constants.%AddWith.WithSelf.Op.bb1)]
// CHECK:STDOUT:   %Core.import_ref.36d: @AddAssignWith.WithSelf.%AddAssignWith.assoc_type (%AddAssignWith.assoc_type.084) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @AddAssignWith.WithSelf.%assoc0 (constants.%assoc0.3b5)]
// CHECK:STDOUT:   %Core.import_ref.90f: @AddAssignWith.WithSelf.%AddAssignWith.WithSelf.Op.type (%AddAssignWith.WithSelf.Op.type.9de) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @AddAssignWith.WithSelf.%AddAssignWith.WithSelf.Op (constants.%AddAssignWith.WithSelf.Op.cfc)]
// CHECK:STDOUT:   %Core.import_ref.e9a = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.278 = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.ebd: type = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.AddWith.impl.d0c.%Int (constants.%Int.d4bbe0.1)]
// CHECK:STDOUT:   %Core.import_ref.7d7: @Int.as.AddWith.impl.d0c.%Int.as.AddWith.impl.Op.type (%Int.as.AddWith.impl.Op.type.543) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.AddWith.impl.d0c.%Int.as.AddWith.impl.Op (constants.%Int.as.AddWith.impl.Op.9b8)]
// CHECK:STDOUT:   %AddWith.impl_witness_table.174 = impl_witness_table (%Core.import_ref.ebd, %Core.import_ref.7d7), @Int.as.AddWith.impl.d0c [concrete]
// CHECK:STDOUT:   %Core.import_ref.b25: @Int.as.AddAssignWith.impl.%Int.as.AddAssignWith.impl.Op.type.2 (%Int.as.AddAssignWith.impl.Op.type.93140c.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.AddAssignWith.impl.%Int.as.AddAssignWith.impl.Op.2 (constants.%Int.as.AddAssignWith.impl.Op.75e08e.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table = impl_witness_table (%Core.import_ref.b25), @Int.as.AddAssignWith.impl [concrete]
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
// CHECK:STDOUT:   %impl.elem1: %.da3 = impl_witness_access constants.%AddWith.impl_witness.e33, element1 [concrete = constants.%C.as.AddWith.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.ref, %impl.elem1
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.as.AddWith.impl.Op.call: init %C to %.loc30_26.1 = call %bound_method(%a.ref, %b.ref)
// CHECK:STDOUT:   return %C.as.AddWith.impl.Op.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestAssign(%a.param: %ptr.00e, %b.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %ptr.00e = name_ref a, %a
// CHECK:STDOUT:   %.loc38: ref %C = deref %a.ref
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0: %.964 = impl_witness_access constants.%AddAssignWith.impl_witness.3ff, element0 [concrete = constants.%C.as.AddAssignWith.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc38, %impl.elem0
// CHECK:STDOUT:   %C.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method(%.loc38, %b.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestNamedConstraintOp(%a.param: %D, %b.param: %D) -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %D = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %D = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1: %.2bc = impl_witness_access constants.%AddWith.impl_witness.736, element1 [concrete = constants.%D.as.AddWith.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.ref, %impl.elem1
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %D.as.AddWith.impl.Op.call: init %D to %.loc53_41.1 = call %bound_method(%a.ref, %b.ref)
// CHECK:STDOUT:   return %D.as.AddWith.impl.Op.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestNamedConstraintAssign(%a.param: %ptr.4a5, %b.param: %D) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %ptr.4a5 = name_ref a, %a
// CHECK:STDOUT:   %.loc61: ref %D = deref %a.ref
// CHECK:STDOUT:   %b.ref: %D = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0: %.123 = impl_witness_access constants.%AddAssignWith.impl_witness.736, element0 [concrete = constants.%D.as.AddAssignWith.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc61, %impl.elem0
// CHECK:STDOUT:   %D.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method(%.loc61, %b.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestGenericOp(%T.loc65_19.2: %Add.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %AddWith.type.loc67_12.2: type = facet_type <@AddWith, @AddWith(%T.as_type.loc65_34.1)> [symbolic = %AddWith.type.loc67_12.2 (constants.%AddWith.type.2309ff.2)]
// CHECK:STDOUT:   %require_complete.loc67: <witness> = require_complete_type %AddWith.type.loc67_12.2 [symbolic = %require_complete.loc67 (constants.%require_complete.a92)]
// CHECK:STDOUT:   %AddWith.assoc_type: type = assoc_entity_type @AddWith, @AddWith(%T.as_type.loc65_34.1) [symbolic = %AddWith.assoc_type (constants.%AddWith.assoc_type.d6e422.2)]
// CHECK:STDOUT:   %assoc1: @TestGenericOp.%AddWith.assoc_type (%AddWith.assoc_type.d6e422.2) = assoc_entity element1, imports.%Core.import_ref.8ad [symbolic = %assoc1 (constants.%assoc1.857cf7.2)]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc65_19.1, @AddWith, @AddWith(%T.as_type.loc65_34.1) [symbolic = %AddWith.lookup_impl_witness (constants.%AddWith.lookup_impl_witness.82cdf5.2)]
// CHECK:STDOUT:   %AddWith.facet: @TestGenericOp.%AddWith.type.loc67_12.2 (%AddWith.type.2309ff.2) = facet_value %T.as_type.loc65_34.1, (%AddWith.lookup_impl_witness) [symbolic = %AddWith.facet (constants.%AddWith.facet.a58)]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%T.as_type.loc65_34.1, %AddWith.facet) [symbolic = %AddWith.WithSelf.Op.type (constants.%AddWith.WithSelf.Op.type.593)]
// CHECK:STDOUT:   %.loc67_12.2: type = fn_type_with_self_type %AddWith.WithSelf.Op.type, %AddWith.facet [symbolic = %.loc67_12.2 (constants.%.50a)]
// CHECK:STDOUT:   %impl.elem1.loc67_12.2: @TestGenericOp.%.loc67_12.2 (%.50a) = impl_witness_access %AddWith.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc67_12.2 (constants.%impl.elem1.ce3)]
// CHECK:STDOUT:   %specific_impl_fn.loc67_12.2: <specific function> = specific_impl_function %impl.elem1.loc67_12.2, @AddWith.WithSelf.Op(%T.as_type.loc65_34.1, %AddWith.facet) [symbolic = %specific_impl_fn.loc67_12.2 (constants.%specific_impl_fn.844)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @TestGenericOp.%T.as_type.loc65_34.1 (%T.as_type.190bb7.1), %b.param: @TestGenericOp.%T.as_type.loc65_34.1 (%T.as_type.190bb7.1)) -> out %return.param: @TestGenericOp.%T.as_type.loc65_34.1 (%T.as_type.190bb7.1) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %a.ref: @TestGenericOp.%T.as_type.loc65_34.1 (%T.as_type.190bb7.1) = name_ref a, %a
// CHECK:STDOUT:     %b.ref: @TestGenericOp.%T.as_type.loc65_34.1 (%T.as_type.190bb7.1) = name_ref b, %b
// CHECK:STDOUT:     %AddWith.type.loc67_12.1: type = facet_type <@AddWith, @AddWith(constants.%T.as_type.190bb7.1)> [symbolic = %AddWith.type.loc67_12.2 (constants.%AddWith.type.2309ff.2)]
// CHECK:STDOUT:     %.loc67_12.1: @TestGenericOp.%AddWith.assoc_type (%AddWith.assoc_type.d6e422.2) = specific_constant imports.%Core.import_ref.5c64, @AddWith.WithSelf(constants.%T.as_type.190bb7.1, constants.%Self.958) [symbolic = %assoc1 (constants.%assoc1.857cf7.2)]
// CHECK:STDOUT:     %Op.ref: @TestGenericOp.%AddWith.assoc_type (%AddWith.assoc_type.d6e422.2) = name_ref Op, %.loc67_12.1 [symbolic = %assoc1 (constants.%assoc1.857cf7.2)]
// CHECK:STDOUT:     %impl.elem1.loc67_12.1: @TestGenericOp.%.loc67_12.2 (%.50a) = impl_witness_access constants.%AddWith.lookup_impl_witness.82cdf5.2, element1 [symbolic = %impl.elem1.loc67_12.2 (constants.%impl.elem1.ce3)]
// CHECK:STDOUT:     %bound_method.loc67_12.1: <bound method> = bound_method %a.ref, %impl.elem1.loc67_12.1
// CHECK:STDOUT:     %specific_impl_fn.loc67_12.1: <specific function> = specific_impl_function %impl.elem1.loc67_12.1, @AddWith.WithSelf.Op(constants.%T.as_type.190bb7.1, constants.%AddWith.facet.a58) [symbolic = %specific_impl_fn.loc67_12.2 (constants.%specific_impl_fn.844)]
// CHECK:STDOUT:     %bound_method.loc67_12.2: <bound method> = bound_method %a.ref, %specific_impl_fn.loc67_12.1
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %AddWith.WithSelf.Op.call: init @TestGenericOp.%T.as_type.loc65_34.1 (%T.as_type.190bb7.1) to %.loc65_46.1 = call %bound_method.loc67_12.2(%a.ref, %b.ref)
// CHECK:STDOUT:     return %AddWith.WithSelf.Op.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestGenericAssign(%T.loc71_23.2: %AddAssign.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %AddAssignWith.type.loc73_6.2: type = facet_type <@AddAssignWith, @AddAssignWith(%T.as_type.loc71_45.1)> [symbolic = %AddAssignWith.type.loc73_6.2 (constants.%AddAssignWith.type.339202.2)]
// CHECK:STDOUT:   %require_complete.loc73: <witness> = require_complete_type %AddAssignWith.type.loc73_6.2 [symbolic = %require_complete.loc73 (constants.%require_complete.58e780.2)]
// CHECK:STDOUT:   %AddAssignWith.assoc_type: type = assoc_entity_type @AddAssignWith, @AddAssignWith(%T.as_type.loc71_45.1) [symbolic = %AddAssignWith.assoc_type (constants.%AddAssignWith.assoc_type.0b4a9d.2)]
// CHECK:STDOUT:   %assoc0: @TestGenericAssign.%AddAssignWith.assoc_type (%AddAssignWith.assoc_type.0b4a9d.2) = assoc_entity element0, imports.%Core.import_ref.90f [symbolic = %assoc0 (constants.%assoc0.498608.2)]
// CHECK:STDOUT:   %AddAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc71_23.1, @AddAssignWith, @AddAssignWith(%T.as_type.loc71_45.1) [symbolic = %AddAssignWith.lookup_impl_witness (constants.%AddAssignWith.lookup_impl_witness)]
// CHECK:STDOUT:   %AddAssignWith.facet: @TestGenericAssign.%AddAssignWith.type.loc73_6.2 (%AddAssignWith.type.339202.2) = facet_value %T.as_type.loc71_45.1, (%AddAssignWith.lookup_impl_witness) [symbolic = %AddAssignWith.facet (constants.%AddAssignWith.facet.2b4)]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%T.as_type.loc71_45.1, %AddAssignWith.facet) [symbolic = %AddAssignWith.WithSelf.Op.type (constants.%AddAssignWith.WithSelf.Op.type.396)]
// CHECK:STDOUT:   %.loc73_6.2: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type, %AddAssignWith.facet [symbolic = %.loc73_6.2 (constants.%.c82)]
// CHECK:STDOUT:   %impl.elem0.loc73_6.2: @TestGenericAssign.%.loc73_6.2 (%.c82) = impl_witness_access %AddAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc73_6.2 (constants.%impl.elem0.9b7)]
// CHECK:STDOUT:   %specific_impl_fn.loc73_6.2: <specific function> = specific_impl_function %impl.elem0.loc73_6.2, @AddAssignWith.WithSelf.Op(%T.as_type.loc71_45.1, %AddAssignWith.facet) [symbolic = %specific_impl_fn.loc73_6.2 (constants.%specific_impl_fn.d5e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @TestGenericAssign.%ptr.loc71_45.1 (%ptr.360), %b.param: @TestGenericAssign.%T.as_type.loc71_45.1 (%T.as_type.190bb7.2)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %a.ref: @TestGenericAssign.%ptr.loc71_45.1 (%ptr.360) = name_ref a, %a
// CHECK:STDOUT:     %.loc73_3: ref @TestGenericAssign.%T.as_type.loc71_45.1 (%T.as_type.190bb7.2) = deref %a.ref
// CHECK:STDOUT:     %b.ref: @TestGenericAssign.%T.as_type.loc71_45.1 (%T.as_type.190bb7.2) = name_ref b, %b
// CHECK:STDOUT:     %AddAssignWith.type.loc73_6.1: type = facet_type <@AddAssignWith, @AddAssignWith(constants.%T.as_type.190bb7.2)> [symbolic = %AddAssignWith.type.loc73_6.2 (constants.%AddAssignWith.type.339202.2)]
// CHECK:STDOUT:     %.loc73_6.1: @TestGenericAssign.%AddAssignWith.assoc_type (%AddAssignWith.assoc_type.0b4a9d.2) = specific_constant imports.%Core.import_ref.36d, @AddAssignWith.WithSelf(constants.%T.as_type.190bb7.2, constants.%Self.73b) [symbolic = %assoc0 (constants.%assoc0.498608.2)]
// CHECK:STDOUT:     %Op.ref: @TestGenericAssign.%AddAssignWith.assoc_type (%AddAssignWith.assoc_type.0b4a9d.2) = name_ref Op, %.loc73_6.1 [symbolic = %assoc0 (constants.%assoc0.498608.2)]
// CHECK:STDOUT:     %impl.elem0.loc73_6.1: @TestGenericAssign.%.loc73_6.2 (%.c82) = impl_witness_access constants.%AddAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc73_6.2 (constants.%impl.elem0.9b7)]
// CHECK:STDOUT:     %bound_method.loc73_6.1: <bound method> = bound_method %.loc73_3, %impl.elem0.loc73_6.1
// CHECK:STDOUT:     %specific_impl_fn.loc73_6.1: <specific function> = specific_impl_function %impl.elem0.loc73_6.1, @AddAssignWith.WithSelf.Op(constants.%T.as_type.190bb7.2, constants.%AddAssignWith.facet.2b4) [symbolic = %specific_impl_fn.loc73_6.2 (constants.%specific_impl_fn.d5e)]
// CHECK:STDOUT:     %bound_method.loc73_6.2: <bound method> = bound_method %.loc73_3, %specific_impl_fn.loc73_6.1
// CHECK:STDOUT:     %AddAssignWith.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc73_6.2(%.loc73_3, %b.ref)
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
// CHECK:STDOUT: specific @TestGenericOp(constants.%Add.facet.cf6) {
// CHECK:STDOUT:   %T.patt.loc65_19.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc65_19.1 => constants.%Add.facet.cf6
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
// CHECK:STDOUT:   %AddWith.type.loc67_12.2 => constants.%AddWith.type.f3e
// CHECK:STDOUT:   %require_complete.loc67 => constants.%complete_type.18f
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.d7a
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.546
// CHECK:STDOUT:   %AddWith.lookup_impl_witness => constants.%AddWith.impl_witness.e33
// CHECK:STDOUT:   %AddWith.facet => constants.%AddWith.facet.1f6
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type => constants.%AddWith.WithSelf.Op.type.97e
// CHECK:STDOUT:   %.loc67_12.2 => constants.%.da3
// CHECK:STDOUT:   %impl.elem1.loc67_12.2 => constants.%C.as.AddWith.impl.Op
// CHECK:STDOUT:   %specific_impl_fn.loc67_12.2 => constants.%C.as.AddWith.impl.Op
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGenericAssign(constants.%AddAssign.facet.e2c) {
// CHECK:STDOUT:   %T.patt.loc71_23.2 => constants.%T.patt.3a0592.2
// CHECK:STDOUT:   %T.loc71_23.1 => constants.%AddAssign.facet.e2c
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
// CHECK:STDOUT:   %AddAssignWith.type.loc73_6.2 => constants.%AddAssignWith.type.cf4
// CHECK:STDOUT:   %require_complete.loc73 => constants.%complete_type.03e
// CHECK:STDOUT:   %AddAssignWith.assoc_type => constants.%AddAssignWith.assoc_type.a0e
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.da3
// CHECK:STDOUT:   %AddAssignWith.lookup_impl_witness => constants.%AddAssignWith.impl_witness.3ff
// CHECK:STDOUT:   %AddAssignWith.facet => constants.%AddAssignWith.facet.dd1
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type => constants.%AddAssignWith.WithSelf.Op.type.d8d
// CHECK:STDOUT:   %.loc73_6.2 => constants.%.964
// CHECK:STDOUT:   %impl.elem0.loc73_6.2 => constants.%C.as.AddAssignWith.impl.Op
// CHECK:STDOUT:   %specific_impl_fn.loc73_6.2 => constants.%C.as.AddAssignWith.impl.Op
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGenericOp(constants.%Add.facet.6f9) {
// CHECK:STDOUT:   %T.patt.loc65_19.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc65_19.1 => constants.%Add.facet.6f9
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
// CHECK:STDOUT:   %AddWith.type.loc67_12.2 => constants.%AddWith.type.710
// CHECK:STDOUT:   %require_complete.loc67 => constants.%complete_type.2f3
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.805
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.3fd
// CHECK:STDOUT:   %AddWith.lookup_impl_witness => constants.%AddWith.impl_witness.736
// CHECK:STDOUT:   %AddWith.facet => constants.%AddWith.facet.f08
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type => constants.%AddWith.WithSelf.Op.type.5f4
// CHECK:STDOUT:   %.loc67_12.2 => constants.%.2bc
// CHECK:STDOUT:   %impl.elem1.loc67_12.2 => constants.%D.as.AddWith.impl.Op
// CHECK:STDOUT:   %specific_impl_fn.loc67_12.2 => constants.%D.as.AddWith.impl.Op
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGenericAssign(constants.%AddAssign.facet.6f9) {
// CHECK:STDOUT:   %T.patt.loc71_23.2 => constants.%T.patt.3a0592.2
// CHECK:STDOUT:   %T.loc71_23.1 => constants.%AddAssign.facet.6f9
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
// CHECK:STDOUT:   %AddAssignWith.type.loc73_6.2 => constants.%AddAssignWith.type.702
// CHECK:STDOUT:   %require_complete.loc73 => constants.%complete_type.53b
// CHECK:STDOUT:   %AddAssignWith.assoc_type => constants.%AddAssignWith.assoc_type.fe5
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.928
// CHECK:STDOUT:   %AddAssignWith.lookup_impl_witness => constants.%AddAssignWith.impl_witness.736
// CHECK:STDOUT:   %AddAssignWith.facet => constants.%AddAssignWith.facet.c87
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type => constants.%AddAssignWith.WithSelf.Op.type.306
// CHECK:STDOUT:   %.loc73_6.2 => constants.%.123
// CHECK:STDOUT:   %impl.elem0.loc73_6.2 => constants.%D.as.AddAssignWith.impl.Op
// CHECK:STDOUT:   %specific_impl_fn.loc73_6.2 => constants.%D.as.AddAssignWith.impl.Op
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGenericOp(constants.%Add.facet.b9e) {
// CHECK:STDOUT:   %T.patt.loc65_19.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc65_19.1 => constants.%Add.facet.b9e
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
// CHECK:STDOUT:   %AddWith.type.loc67_12.2 => constants.%AddWith.type.ff2
// CHECK:STDOUT:   %require_complete.loc67 => constants.%complete_type.00d
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.ddd
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.6d3
// CHECK:STDOUT:   %AddWith.lookup_impl_witness => constants.%AddWith.impl_witness.d3a
// CHECK:STDOUT:   %AddWith.facet => constants.%AddWith.facet.06d
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type => constants.%AddWith.WithSelf.Op.type.590
// CHECK:STDOUT:   %.loc67_12.2 => constants.%.abe
// CHECK:STDOUT:   %impl.elem1.loc67_12.2 => constants.%Int.as.AddWith.impl.Op.f9e
// CHECK:STDOUT:   %specific_impl_fn.loc67_12.2 => constants.%Int.as.AddWith.impl.Op.specific_fn.3df
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGenericAssign(constants.%AddAssign.facet.690) {
// CHECK:STDOUT:   %T.patt.loc71_23.2 => constants.%T.patt.3a0592.2
// CHECK:STDOUT:   %T.loc71_23.1 => constants.%AddAssign.facet.690
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
// CHECK:STDOUT:   %AddAssignWith.type.loc73_6.2 => constants.%AddAssignWith.type.1ad
// CHECK:STDOUT:   %require_complete.loc73 => constants.%complete_type.eb5
// CHECK:STDOUT:   %AddAssignWith.assoc_type => constants.%AddAssignWith.assoc_type.144
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.c38
// CHECK:STDOUT:   %AddAssignWith.lookup_impl_witness => constants.%AddAssignWith.impl_witness.95a
// CHECK:STDOUT:   %AddAssignWith.facet => constants.%AddAssignWith.facet.d53
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type => constants.%AddAssignWith.WithSelf.Op.type.313
// CHECK:STDOUT:   %.loc73_6.2 => constants.%.316
// CHECK:STDOUT:   %impl.elem0.loc73_6.2 => constants.%Int.as.AddAssignWith.impl.Op.bff4f9.2
// CHECK:STDOUT:   %specific_impl_fn.loc73_6.2 => constants.%Int.as.AddAssignWith.impl.Op.specific_fn.ec28e6.1
// CHECK:STDOUT: }
// CHECK:STDOUT:

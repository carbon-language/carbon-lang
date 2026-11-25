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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/overloaded/mod.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/overloaded/mod.carbon

// This file was generated from binary_op.carbon.tmpl. Run make_tests.sh to regenerate.

package User;

class C {};

impl C as Core.ModWith(C) where .Result = C {
  fn Op(unused self: C, unused other: C) -> C {
    return {};
  }
}
impl C as Core.ModAssignWith(C) {
  fn Op(unused ref self: C, unused other: C) {}
}

fn TestOp(a: C, b: C) -> C {
  //@dump-sem-ir-begin
  return a % b;
  //@dump-sem-ir-end
}

fn TestAssign(a: C*, b: C) {
  //@dump-sem-ir-begin
  *a %= b;
  //@dump-sem-ir-end
}

class D {};

impl D as Core.Mod {
  fn Op(unused self: D, unused other: D) -> D {
    return {};
  }
}
impl D as Core.ModAssign {
  fn Op(unused ref self: D, unused other: D) {}
}

fn TestNamedConstraintOp(a: D, b: D) -> D {
  //@dump-sem-ir-begin
  return a % b;
  //@dump-sem-ir-end
}

fn TestNamedConstraintAssign(a: D*, b: D) {
  //@dump-sem-ir-begin
  *a %= b;
  //@dump-sem-ir-end
}

fn TestGenericOp[T: Core.Mod](a: T, b: T) -> T {
  //@dump-sem-ir-begin
  return a % b;
  //@dump-sem-ir-end
}

fn TestGenericAssign[T: Core.ModAssign](a: T*, b: T) {
  //@dump-sem-ir-begin
  *a %= b;
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

// CHECK:STDOUT: --- mod.carbon = $mod
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Other: type = symbolic_binding Other, 0 [symbolic]
// CHECK:STDOUT:   %ModWith.type.11e: type = facet_type <@ModWith, @ModWith(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.c18: %ModWith.type.11e = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %ModWith.assoc_type.1f1: type = assoc_entity_type @ModWith, @ModWith(%Other) [symbolic]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.04e: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%Other, %Self.c18) [symbolic]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.6fb: %ModWith.WithSelf.Op.type.04e = struct_value () [symbolic]
// CHECK:STDOUT:   %ModWith.type.bf9: type = facet_type <@ModWith, @ModWith(%C)> [concrete]
// CHECK:STDOUT:   %ModWith.assoc_type.828: type = assoc_entity_type @ModWith, @ModWith(%C) [concrete]
// CHECK:STDOUT:   %assoc1.871: %ModWith.assoc_type.828 = assoc_entity element1, imports.%Core.import_ref.9f8f [concrete]
// CHECK:STDOUT:   %ModWith.impl_witness.f41: <witness> = impl_witness @C.as.ModWith.impl.%ModWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.4a7: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %.79c: Core.Form = init_form %C [concrete]
// CHECK:STDOUT:   %return.param_patt.415: %pattern_type.4a7 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.340: %pattern_type.4a7 = return_slot_pattern %return.param_patt.415, %C [concrete]
// CHECK:STDOUT:   %C.as.ModWith.impl.Op.type: type = fn_type @C.as.ModWith.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.ModWith.impl.Op: %C.as.ModWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %ModWith.facet.03b: %ModWith.type.bf9 = facet_value %C, (%ModWith.impl_witness.f41) [concrete]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.06b: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%C, %ModWith.facet.03b) [concrete]
// CHECK:STDOUT:   %ModAssignWith.type.3a0: type = facet_type <@ModAssignWith, @ModAssignWith(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.cc0: %ModAssignWith.type.3a0 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %ModAssignWith.assoc_type.01d: type = assoc_entity_type @ModAssignWith, @ModAssignWith(%Other) [symbolic]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type.7f2: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%Other, %Self.cc0) [symbolic]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.9b8: %ModAssignWith.WithSelf.Op.type.7f2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ModAssignWith.type.d51: type = facet_type <@ModAssignWith, @ModAssignWith(%C)> [concrete]
// CHECK:STDOUT:   %ModAssignWith.impl_witness.d6f: <witness> = impl_witness @C.as.ModAssignWith.impl.%ModAssignWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %ModAssignWith.assoc_type.183: type = assoc_entity_type @ModAssignWith, @ModAssignWith(%C) [concrete]
// CHECK:STDOUT:   %assoc0.aeb: %ModAssignWith.assoc_type.183 = assoc_entity element0, imports.%Core.import_ref.97f [concrete]
// CHECK:STDOUT:   %C.as.ModAssignWith.impl.Op.type: type = fn_type @C.as.ModAssignWith.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.ModAssignWith.impl.Op: %C.as.ModAssignWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %ModAssignWith.facet.6e5: %ModAssignWith.type.d51 = facet_value %C, (%ModAssignWith.impl_witness.d6f) [concrete]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type.cb7: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%C, %ModAssignWith.facet.6e5) [concrete]
// CHECK:STDOUT:   %a.param_patt.2a5: %pattern_type.4a7 = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt.a30: %pattern_type.4a7 = wrapper_binding_pattern a, %a.param_patt.2a5 [concrete]
// CHECK:STDOUT:   %b.param_patt.45f: %pattern_type.4a7 = value_param_pattern [concrete]
// CHECK:STDOUT:   %b.patt.562: %pattern_type.4a7 = wrapper_binding_pattern b, %b.param_patt.45f [concrete]
// CHECK:STDOUT:   %assoc1.f4b: %ModWith.assoc_type.1f1 = assoc_entity element1, imports.%Core.import_ref.be0 [symbolic]
// CHECK:STDOUT:   %.1bb: type = fn_type_with_self_type %ModWith.WithSelf.Op.type.06b, %ModWith.facet.03b [concrete]
// CHECK:STDOUT:   %ptr.00e: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.842: type = pattern_type %ptr.00e [concrete]
// CHECK:STDOUT:   %a.param_patt.4fa: %pattern_type.842 = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt.eba: %pattern_type.842 = wrapper_binding_pattern a, %a.param_patt.4fa [concrete]
// CHECK:STDOUT:   %assoc0.3a8: %ModAssignWith.assoc_type.01d = assoc_entity element0, imports.%Core.import_ref.ffb [symbolic]
// CHECK:STDOUT:   %.651: type = fn_type_with_self_type %ModAssignWith.WithSelf.Op.type.cb7, %ModAssignWith.facet.6e5 [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %Mod.type: type = facet_type <@Mod> [concrete]
// CHECK:STDOUT:   %ModWith.impl_witness.736: <witness> = impl_witness @D.as.ModWith.impl.%ModWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.331: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %.84d: Core.Form = init_form %D [concrete]
// CHECK:STDOUT:   %return.param_patt.835: %pattern_type.331 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.5ef: %pattern_type.331 = return_slot_pattern %return.param_patt.835, %D [concrete]
// CHECK:STDOUT:   %D.as.ModWith.impl.Op.type: type = fn_type @D.as.ModWith.impl.Op [concrete]
// CHECK:STDOUT:   %D.as.ModWith.impl.Op: %D.as.ModWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %ModWith.type.d8b: type = facet_type <@ModWith, @ModWith(%D)> [concrete]
// CHECK:STDOUT:   %ModWith.facet.485: %ModWith.type.d8b = facet_value %D, (%ModWith.impl_witness.736) [concrete]
// CHECK:STDOUT:   %ModWith.assoc_type.53e: type = assoc_entity_type @ModWith, @ModWith(%D) [concrete]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.525: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%D, %ModWith.facet.485) [concrete]
// CHECK:STDOUT:   %assoc1.c03: %ModWith.assoc_type.53e = assoc_entity element1, imports.%Core.import_ref.9f8f [concrete]
// CHECK:STDOUT:   %ModAssign.type: type = facet_type <@ModAssign> [concrete]
// CHECK:STDOUT:   %ModAssignWith.impl_witness.736: <witness> = impl_witness @D.as.ModAssignWith.impl.%ModAssignWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %D.as.ModAssignWith.impl.Op.type: type = fn_type @D.as.ModAssignWith.impl.Op [concrete]
// CHECK:STDOUT:   %D.as.ModAssignWith.impl.Op: %D.as.ModAssignWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %ModAssignWith.type.14f: type = facet_type <@ModAssignWith, @ModAssignWith(%D)> [concrete]
// CHECK:STDOUT:   %ModAssignWith.facet.0ce: %ModAssignWith.type.14f = facet_value %D, (%ModAssignWith.impl_witness.736) [concrete]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type.ec3: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%D, %ModAssignWith.facet.0ce) [concrete]
// CHECK:STDOUT:   %ModAssignWith.assoc_type.d05: type = assoc_entity_type @ModAssignWith, @ModAssignWith(%D) [concrete]
// CHECK:STDOUT:   %assoc0.67a: %ModAssignWith.assoc_type.d05 = assoc_entity element0, imports.%Core.import_ref.97f [concrete]
// CHECK:STDOUT:   %a.param_patt.d0e: %pattern_type.331 = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt.ec5: %pattern_type.331 = wrapper_binding_pattern a, %a.param_patt.d0e [concrete]
// CHECK:STDOUT:   %b.param_patt.110: %pattern_type.331 = value_param_pattern [concrete]
// CHECK:STDOUT:   %b.patt.14c: %pattern_type.331 = wrapper_binding_pattern b, %b.param_patt.110 [concrete]
// CHECK:STDOUT:   %.8b2: type = fn_type_with_self_type %ModWith.WithSelf.Op.type.525, %ModWith.facet.485 [concrete]
// CHECK:STDOUT:   %ptr.4a5: type = ptr_type %D [concrete]
// CHECK:STDOUT:   %pattern_type.ceb: type = pattern_type %ptr.4a5 [concrete]
// CHECK:STDOUT:   %a.param_patt.609: %pattern_type.ceb = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt.fac: %pattern_type.ceb = wrapper_binding_pattern a, %a.param_patt.609 [concrete]
// CHECK:STDOUT:   %.2df: type = fn_type_with_self_type %ModAssignWith.WithSelf.Op.type.ec3, %ModAssignWith.facet.0ce [concrete]
// CHECK:STDOUT:   %pattern_type.9a587c.2: type = pattern_type %Mod.type [concrete]
// CHECK:STDOUT:   %T.patt.3a0592.1: %pattern_type.9a587c.2 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.f45530.1: %Mod.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.190bb7.1: type = facet_access_type %T.f45530.1 [symbolic]
// CHECK:STDOUT:   %pattern_type.86a9db.1: type = pattern_type %T.as_type.190bb7.1 [symbolic]
// CHECK:STDOUT:   %a.param_patt.e33: %pattern_type.86a9db.1 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %a.patt.e02: %pattern_type.86a9db.1 = wrapper_binding_pattern a, %a.param_patt.e33 [symbolic]
// CHECK:STDOUT:   %b.param_patt.4cd741.1: %pattern_type.86a9db.1 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %b.patt.60acf6.1: %pattern_type.86a9db.1 = wrapper_binding_pattern b, %b.param_patt.4cd741.1 [symbolic]
// CHECK:STDOUT:   %.1f0: Core.Form = init_form %T.as_type.190bb7.1 [symbolic]
// CHECK:STDOUT:   %return.param_patt.5f6: %pattern_type.86a9db.1 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.96a: %pattern_type.86a9db.1 = return_slot_pattern %return.param_patt.5f6, %T.as_type.190bb7.1 [symbolic]
// CHECK:STDOUT:   %ModWith.type.280069.2: type = facet_type <@ModWith, @ModWith(%T.as_type.190bb7.1)> [symbolic]
// CHECK:STDOUT:   %ModWith.assoc_type.63a975.2: type = assoc_entity_type @ModWith, @ModWith(%T.as_type.190bb7.1) [symbolic]
// CHECK:STDOUT:   %assoc1.01ac1f.2: %ModWith.assoc_type.63a975.2 = assoc_entity element1, imports.%Core.import_ref.9f8f [symbolic]
// CHECK:STDOUT:   %require_complete.887: <witness> = require_complete_type %ModWith.type.280069.2 [symbolic]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.8b3b39.2: <witness> = lookup_impl_witness %T.f45530.1, @ModWith, @ModWith(%T.as_type.190bb7.1) [symbolic]
// CHECK:STDOUT:   %ModWith.facet.ba5: %ModWith.type.280069.2 = facet_value %T.as_type.190bb7.1, (%ModWith.lookup_impl_witness.8b3b39.2) [symbolic]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.8b2: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%T.as_type.190bb7.1, %ModWith.facet.ba5) [symbolic]
// CHECK:STDOUT:   %.16f: type = fn_type_with_self_type %ModWith.WithSelf.Op.type.8b2, %ModWith.facet.ba5 [symbolic]
// CHECK:STDOUT:   %impl.elem1.bf5: %.16f = impl_witness_access %ModWith.lookup_impl_witness.8b3b39.2, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.1ab: <specific function> = specific_impl_function %impl.elem1.bf5, @ModWith.WithSelf.Op(%T.as_type.190bb7.1, %ModWith.facet.ba5) [symbolic]
// CHECK:STDOUT:   %pattern_type.9a587c.3: type = pattern_type %ModAssign.type [concrete]
// CHECK:STDOUT:   %T.patt.3a0592.2: %pattern_type.9a587c.3 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.f45530.2: %ModAssign.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.190bb7.2: type = facet_access_type %T.f45530.2 [symbolic]
// CHECK:STDOUT:   %ptr.360: type = ptr_type %T.as_type.190bb7.2 [symbolic]
// CHECK:STDOUT:   %pattern_type.ff5: type = pattern_type %ptr.360 [symbolic]
// CHECK:STDOUT:   %a.param_patt.f50: %pattern_type.ff5 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %a.patt.193: %pattern_type.ff5 = wrapper_binding_pattern a, %a.param_patt.f50 [symbolic]
// CHECK:STDOUT:   %pattern_type.86a9db.2: type = pattern_type %T.as_type.190bb7.2 [symbolic]
// CHECK:STDOUT:   %b.param_patt.4cd741.2: %pattern_type.86a9db.2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %b.patt.60acf6.2: %pattern_type.86a9db.2 = wrapper_binding_pattern b, %b.param_patt.4cd741.2 [symbolic]
// CHECK:STDOUT:   %ModAssignWith.type.4760d9.2: type = facet_type <@ModAssignWith, @ModAssignWith(%T.as_type.190bb7.2)> [symbolic]
// CHECK:STDOUT:   %ModAssignWith.assoc_type.6b6315.2: type = assoc_entity_type @ModAssignWith, @ModAssignWith(%T.as_type.190bb7.2) [symbolic]
// CHECK:STDOUT:   %assoc0.c31ad7.2: %ModAssignWith.assoc_type.6b6315.2 = assoc_entity element0, imports.%Core.import_ref.97f [symbolic]
// CHECK:STDOUT:   %require_complete.e8eb0e.2: <witness> = require_complete_type %ModAssignWith.type.4760d9.2 [symbolic]
// CHECK:STDOUT:   %ModAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.f45530.2, @ModAssignWith, @ModAssignWith(%T.as_type.190bb7.2) [symbolic]
// CHECK:STDOUT:   %ModAssignWith.facet.4ee: %ModAssignWith.type.4760d9.2 = facet_value %T.as_type.190bb7.2, (%ModAssignWith.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type.760: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%T.as_type.190bb7.2, %ModAssignWith.facet.4ee) [symbolic]
// CHECK:STDOUT:   %.2b3: type = fn_type_with_self_type %ModAssignWith.WithSelf.Op.type.760, %ModAssignWith.facet.4ee [symbolic]
// CHECK:STDOUT:   %impl.elem0.44b: %.2b3 = impl_witness_access %ModAssignWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.c3a: <specific function> = specific_impl_function %impl.elem0.44b, @ModAssignWith.WithSelf.Op(%T.as_type.190bb7.2, %ModAssignWith.facet.4ee) [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %Int.d4bbe0.1: type = class_type @Int, @Int(%N) [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %ptr.a17: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %ptr.a17 [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %Mod.facet.2a8: %Mod.type = facet_value %C, (%ModWith.impl_witness.f41) [concrete]
// CHECK:STDOUT:   %ModAssign.facet.85e: %ModAssign.type = facet_value %C, (%ModAssignWith.impl_witness.d6f) [concrete]
// CHECK:STDOUT:   %Mod.facet.6f9: %Mod.type = facet_value %D, (%ModWith.impl_witness.736) [concrete]
// CHECK:STDOUT:   %ModAssign.facet.6f9: %ModAssign.type = facet_value %D, (%ModAssignWith.impl_witness.736) [concrete]
// CHECK:STDOUT:   %Int.as.ModWith.impl.Op.type.a81: type = fn_type @Int.as.ModWith.impl.Op.1, @Int.as.ModWith.impl.658(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.ModWith.impl.Op.2ff: %Int.as.ModWith.impl.Op.type.a81 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.81c7d0.1: type = facet_type <@ImplicitAs, @ImplicitAs(%Int.d4bbe0.1)> [symbolic]
// CHECK:STDOUT:   %U.f16: %ImplicitAs.type.81c7d0.1 = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %ModWith.type.c3d: type = facet_type <@ModWith, @ModWith(%i32)> [concrete]
// CHECK:STDOUT:   %ModWith.impl_witness.218: <witness> = impl_witness imports.%ModWith.impl_witness_table.f40, @Int.as.ModWith.impl.658(%int_32) [concrete]
// CHECK:STDOUT:   %ModWith.assoc_type.61d: type = assoc_entity_type @ModWith, @ModWith(%i32) [concrete]
// CHECK:STDOUT:   %assoc1.3f4: %ModWith.assoc_type.61d = assoc_entity element1, imports.%Core.import_ref.9f8f [concrete]
// CHECK:STDOUT:   %complete_type.093: <witness> = complete_type_witness %ModWith.type.c3d [concrete]
// CHECK:STDOUT:   %Int.as.ModWith.impl.Op.type.46c: type = fn_type @Int.as.ModWith.impl.Op.1, @Int.as.ModWith.impl.658(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ModWith.impl.Op.0bd: %Int.as.ModWith.impl.Op.type.46c = struct_value () [concrete]
// CHECK:STDOUT:   %Mod.facet.a1a: %Mod.type = facet_value %i32, (%ModWith.impl_witness.218) [concrete]
// CHECK:STDOUT:   %a.param_patt.076: %pattern_type.d94 = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt.96f: %pattern_type.d94 = wrapper_binding_pattern a, %a.param_patt.076 [concrete]
// CHECK:STDOUT:   %b.param_patt.1a2: %pattern_type.d94 = value_param_pattern [concrete]
// CHECK:STDOUT:   %b.patt.3ca: %pattern_type.d94 = wrapper_binding_pattern b, %b.param_patt.1a2 [concrete]
// CHECK:STDOUT:   %.0b3: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.299: %pattern_type.d94 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.6fa: %pattern_type.d94 = return_slot_pattern %return.param_patt.299, %i32 [concrete]
// CHECK:STDOUT:   %Int.as.ModAssignWith.impl.Op.type.45f996.1: type = fn_type @Int.as.ModAssignWith.impl.Op.1, @Int.as.ModAssignWith.impl(%N, %U.f16) [symbolic]
// CHECK:STDOUT:   %Int.as.ModAssignWith.impl.Op.1121c1.1: %Int.as.ModAssignWith.impl.Op.type.45f996.1 = struct_value () [symbolic]
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
// CHECK:STDOUT:   %ModAssignWith.type.124: type = facet_type <@ModAssignWith, @ModAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %ModAssignWith.impl_witness.8db: <witness> = impl_witness imports.%ModAssignWith.impl_witness_table, @Int.as.ModAssignWith.impl(%int_32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %ModAssign.facet.bf7: %ModAssign.type = facet_value %i32, (%ModAssignWith.impl_witness.8db) [concrete]
// CHECK:STDOUT:   %a.param_patt.977: %pattern_type.7dc = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt.b70: %pattern_type.7dc = wrapper_binding_pattern a, %a.param_patt.977 [concrete]
// CHECK:STDOUT:   %complete_type.9cd: <witness> = complete_type_witness %ModWith.type.bf9 [concrete]
// CHECK:STDOUT:   %complete_type.a56: <witness> = complete_type_witness %ptr.00e [concrete]
// CHECK:STDOUT:   %complete_type.4bb: <witness> = complete_type_witness %ModAssignWith.type.d51 [concrete]
// CHECK:STDOUT:   %complete_type.232: <witness> = complete_type_witness %ModWith.type.d8b [concrete]
// CHECK:STDOUT:   %complete_type.320: <witness> = complete_type_witness %ptr.4a5 [concrete]
// CHECK:STDOUT:   %complete_type.4c0: <witness> = complete_type_witness %ModAssignWith.type.14f [concrete]
// CHECK:STDOUT:   %ModWith.facet.3c6: %ModWith.type.c3d = facet_value %i32, (%ModWith.impl_witness.218) [concrete]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.759: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%i32, %ModWith.facet.3c6) [concrete]
// CHECK:STDOUT:   %.b3b: type = fn_type_with_self_type %ModWith.WithSelf.Op.type.759, %ModWith.facet.3c6 [concrete]
// CHECK:STDOUT:   %Int.as.ModWith.impl.Op.specific_fn.8c2: <specific function> = specific_function %Int.as.ModWith.impl.Op.0bd, @Int.as.ModWith.impl.Op.1(%int_32) [concrete]
// CHECK:STDOUT:   %complete_type.1b1: <witness> = complete_type_witness %ptr.a17 [concrete]
// CHECK:STDOUT:   %ModAssignWith.assoc_type.dae: type = assoc_entity_type @ModAssignWith, @ModAssignWith(%i32) [concrete]
// CHECK:STDOUT:   %assoc0.c87: %ModAssignWith.assoc_type.dae = assoc_entity element0, imports.%Core.import_ref.97f [concrete]
// CHECK:STDOUT:   %complete_type.f89: <witness> = complete_type_witness %ModAssignWith.type.124 [concrete]
// CHECK:STDOUT:   %ModAssignWith.facet.d8e: %ModAssignWith.type.124 = facet_value %i32, (%ModAssignWith.impl_witness.8db) [concrete]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type.b54: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%i32, %ModAssignWith.facet.d8e) [concrete]
// CHECK:STDOUT:   %.0bb: type = fn_type_with_self_type %ModAssignWith.WithSelf.Op.type.b54, %ModAssignWith.facet.d8e [concrete]
// CHECK:STDOUT:   %Int.as.ModAssignWith.impl.Op.type.40db86.2: type = fn_type @Int.as.ModAssignWith.impl.Op.1, @Int.as.ModAssignWith.impl(%int_32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Int.as.ModAssignWith.impl.Op.12ab9f.2: %Int.as.ModAssignWith.impl.Op.type.40db86.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.ModAssignWith.impl.Op.specific_fn.e61a9e.1: <specific function> = specific_function %Int.as.ModAssignWith.impl.Op.12ab9f.2, @Int.as.ModAssignWith.impl.Op.1(%int_32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.ea5e3: @ModWith.WithSelf.%ModWith.assoc_type (%ModWith.assoc_type.1f1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @ModWith.WithSelf.%assoc1 (constants.%assoc1.f4b)]
// CHECK:STDOUT:   %Core.import_ref.9f8f: @ModWith.WithSelf.%ModWith.WithSelf.Op.type (%ModWith.WithSelf.Op.type.04e) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @ModWith.WithSelf.%ModWith.WithSelf.Op (constants.%ModWith.WithSelf.Op.6fb)]
// CHECK:STDOUT:   %Core.import_ref.1a2: @ModAssignWith.WithSelf.%ModAssignWith.assoc_type (%ModAssignWith.assoc_type.01d) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @ModAssignWith.WithSelf.%assoc0 (constants.%assoc0.3a8)]
// CHECK:STDOUT:   %Core.import_ref.97f: @ModAssignWith.WithSelf.%ModAssignWith.WithSelf.Op.type (%ModAssignWith.WithSelf.Op.type.7f2) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @ModAssignWith.WithSelf.%ModAssignWith.WithSelf.Op (constants.%ModAssignWith.WithSelf.Op.9b8)]
// CHECK:STDOUT:   %Core.import_ref.be0 = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.ffb = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.ebd: type = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.ModWith.impl.658.%Int (constants.%Int.d4bbe0.1)]
// CHECK:STDOUT:   %Core.import_ref.7cb: @Int.as.ModWith.impl.658.%Int.as.ModWith.impl.Op.type (%Int.as.ModWith.impl.Op.type.a81) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.ModWith.impl.658.%Int.as.ModWith.impl.Op (constants.%Int.as.ModWith.impl.Op.2ff)]
// CHECK:STDOUT:   %ModWith.impl_witness_table.f40 = impl_witness_table (%Core.import_ref.ebd, %Core.import_ref.7cb), @Int.as.ModWith.impl.658 [concrete]
// CHECK:STDOUT:   %Core.import_ref.28c: @Int.as.ModAssignWith.impl.%Int.as.ModAssignWith.impl.Op.type.2 (%Int.as.ModAssignWith.impl.Op.type.45f996.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.ModAssignWith.impl.%Int.as.ModAssignWith.impl.Op.2 (constants.%Int.as.ModAssignWith.impl.Op.1121c1.1)]
// CHECK:STDOUT:   %ModAssignWith.impl_witness_table = impl_witness_table (%Core.import_ref.28c), @Int.as.ModAssignWith.impl [concrete]
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
// CHECK:STDOUT:   %impl.elem1: %.1bb = impl_witness_access constants.%ModWith.impl_witness.f41, element1 [concrete = constants.%C.as.ModWith.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.ref, %impl.elem1
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.as.ModWith.impl.Op.call: init %C to %.loc30_26.1 = call %bound_method(%a.ref, %b.ref)
// CHECK:STDOUT:   return %C.as.ModWith.impl.Op.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestAssign(%a.param: %ptr.00e, %b.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %ptr.00e = name_ref a, %a
// CHECK:STDOUT:   %.loc38: ref %C = deref %a.ref
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0: %.651 = impl_witness_access constants.%ModAssignWith.impl_witness.d6f, element0 [concrete = constants.%C.as.ModAssignWith.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc38, %impl.elem0
// CHECK:STDOUT:   %C.as.ModAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method(%.loc38, %b.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestNamedConstraintOp(%a.param: %D, %b.param: %D) -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %D = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %D = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1: %.8b2 = impl_witness_access constants.%ModWith.impl_witness.736, element1 [concrete = constants.%D.as.ModWith.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.ref, %impl.elem1
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %D.as.ModWith.impl.Op.call: init %D to %.loc53_41.1 = call %bound_method(%a.ref, %b.ref)
// CHECK:STDOUT:   return %D.as.ModWith.impl.Op.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestNamedConstraintAssign(%a.param: %ptr.4a5, %b.param: %D) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %ptr.4a5 = name_ref a, %a
// CHECK:STDOUT:   %.loc61: ref %D = deref %a.ref
// CHECK:STDOUT:   %b.ref: %D = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0: %.2df = impl_witness_access constants.%ModAssignWith.impl_witness.736, element0 [concrete = constants.%D.as.ModAssignWith.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc61, %impl.elem0
// CHECK:STDOUT:   %D.as.ModAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method(%.loc61, %b.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestGenericOp(%T.loc65_19.2: %Mod.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %ModWith.type.loc67_12.2: type = facet_type <@ModWith, @ModWith(%T.as_type.loc65_34.1)> [symbolic = %ModWith.type.loc67_12.2 (constants.%ModWith.type.280069.2)]
// CHECK:STDOUT:   %require_complete.loc67: <witness> = require_complete_type %ModWith.type.loc67_12.2 [symbolic = %require_complete.loc67 (constants.%require_complete.887)]
// CHECK:STDOUT:   %ModWith.assoc_type: type = assoc_entity_type @ModWith, @ModWith(%T.as_type.loc65_34.1) [symbolic = %ModWith.assoc_type (constants.%ModWith.assoc_type.63a975.2)]
// CHECK:STDOUT:   %assoc1: @TestGenericOp.%ModWith.assoc_type (%ModWith.assoc_type.63a975.2) = assoc_entity element1, imports.%Core.import_ref.9f8f [symbolic = %assoc1 (constants.%assoc1.01ac1f.2)]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc65_19.1, @ModWith, @ModWith(%T.as_type.loc65_34.1) [symbolic = %ModWith.lookup_impl_witness (constants.%ModWith.lookup_impl_witness.8b3b39.2)]
// CHECK:STDOUT:   %ModWith.facet: @TestGenericOp.%ModWith.type.loc67_12.2 (%ModWith.type.280069.2) = facet_value %T.as_type.loc65_34.1, (%ModWith.lookup_impl_witness) [symbolic = %ModWith.facet (constants.%ModWith.facet.ba5)]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%T.as_type.loc65_34.1, %ModWith.facet) [symbolic = %ModWith.WithSelf.Op.type (constants.%ModWith.WithSelf.Op.type.8b2)]
// CHECK:STDOUT:   %.loc67_12.2: type = fn_type_with_self_type %ModWith.WithSelf.Op.type, %ModWith.facet [symbolic = %.loc67_12.2 (constants.%.16f)]
// CHECK:STDOUT:   %impl.elem1.loc67_12.2: @TestGenericOp.%.loc67_12.2 (%.16f) = impl_witness_access %ModWith.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc67_12.2 (constants.%impl.elem1.bf5)]
// CHECK:STDOUT:   %specific_impl_fn.loc67_12.2: <specific function> = specific_impl_function %impl.elem1.loc67_12.2, @ModWith.WithSelf.Op(%T.as_type.loc65_34.1, %ModWith.facet) [symbolic = %specific_impl_fn.loc67_12.2 (constants.%specific_impl_fn.1ab)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @TestGenericOp.%T.as_type.loc65_34.1 (%T.as_type.190bb7.1), %b.param: @TestGenericOp.%T.as_type.loc65_34.1 (%T.as_type.190bb7.1)) -> out %return.param: @TestGenericOp.%T.as_type.loc65_34.1 (%T.as_type.190bb7.1) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %a.ref: @TestGenericOp.%T.as_type.loc65_34.1 (%T.as_type.190bb7.1) = name_ref a, %a
// CHECK:STDOUT:     %b.ref: @TestGenericOp.%T.as_type.loc65_34.1 (%T.as_type.190bb7.1) = name_ref b, %b
// CHECK:STDOUT:     %ModWith.type.loc67_12.1: type = facet_type <@ModWith, @ModWith(constants.%T.as_type.190bb7.1)> [symbolic = %ModWith.type.loc67_12.2 (constants.%ModWith.type.280069.2)]
// CHECK:STDOUT:     %.loc67_12.1: @TestGenericOp.%ModWith.assoc_type (%ModWith.assoc_type.63a975.2) = specific_constant imports.%Core.import_ref.ea5e3, @ModWith.WithSelf(constants.%T.as_type.190bb7.1, constants.%Self.c18) [symbolic = %assoc1 (constants.%assoc1.01ac1f.2)]
// CHECK:STDOUT:     %Op.ref: @TestGenericOp.%ModWith.assoc_type (%ModWith.assoc_type.63a975.2) = name_ref Op, %.loc67_12.1 [symbolic = %assoc1 (constants.%assoc1.01ac1f.2)]
// CHECK:STDOUT:     %impl.elem1.loc67_12.1: @TestGenericOp.%.loc67_12.2 (%.16f) = impl_witness_access constants.%ModWith.lookup_impl_witness.8b3b39.2, element1 [symbolic = %impl.elem1.loc67_12.2 (constants.%impl.elem1.bf5)]
// CHECK:STDOUT:     %bound_method.loc67_12.1: <bound method> = bound_method %a.ref, %impl.elem1.loc67_12.1
// CHECK:STDOUT:     %specific_impl_fn.loc67_12.1: <specific function> = specific_impl_function %impl.elem1.loc67_12.1, @ModWith.WithSelf.Op(constants.%T.as_type.190bb7.1, constants.%ModWith.facet.ba5) [symbolic = %specific_impl_fn.loc67_12.2 (constants.%specific_impl_fn.1ab)]
// CHECK:STDOUT:     %bound_method.loc67_12.2: <bound method> = bound_method %a.ref, %specific_impl_fn.loc67_12.1
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %ModWith.WithSelf.Op.call: init @TestGenericOp.%T.as_type.loc65_34.1 (%T.as_type.190bb7.1) to %.loc65_46.1 = call %bound_method.loc67_12.2(%a.ref, %b.ref)
// CHECK:STDOUT:     return %ModWith.WithSelf.Op.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestGenericAssign(%T.loc71_23.2: %ModAssign.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %ModAssignWith.type.loc73_6.2: type = facet_type <@ModAssignWith, @ModAssignWith(%T.as_type.loc71_45.1)> [symbolic = %ModAssignWith.type.loc73_6.2 (constants.%ModAssignWith.type.4760d9.2)]
// CHECK:STDOUT:   %require_complete.loc73: <witness> = require_complete_type %ModAssignWith.type.loc73_6.2 [symbolic = %require_complete.loc73 (constants.%require_complete.e8eb0e.2)]
// CHECK:STDOUT:   %ModAssignWith.assoc_type: type = assoc_entity_type @ModAssignWith, @ModAssignWith(%T.as_type.loc71_45.1) [symbolic = %ModAssignWith.assoc_type (constants.%ModAssignWith.assoc_type.6b6315.2)]
// CHECK:STDOUT:   %assoc0: @TestGenericAssign.%ModAssignWith.assoc_type (%ModAssignWith.assoc_type.6b6315.2) = assoc_entity element0, imports.%Core.import_ref.97f [symbolic = %assoc0 (constants.%assoc0.c31ad7.2)]
// CHECK:STDOUT:   %ModAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc71_23.1, @ModAssignWith, @ModAssignWith(%T.as_type.loc71_45.1) [symbolic = %ModAssignWith.lookup_impl_witness (constants.%ModAssignWith.lookup_impl_witness)]
// CHECK:STDOUT:   %ModAssignWith.facet: @TestGenericAssign.%ModAssignWith.type.loc73_6.2 (%ModAssignWith.type.4760d9.2) = facet_value %T.as_type.loc71_45.1, (%ModAssignWith.lookup_impl_witness) [symbolic = %ModAssignWith.facet (constants.%ModAssignWith.facet.4ee)]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%T.as_type.loc71_45.1, %ModAssignWith.facet) [symbolic = %ModAssignWith.WithSelf.Op.type (constants.%ModAssignWith.WithSelf.Op.type.760)]
// CHECK:STDOUT:   %.loc73_6.2: type = fn_type_with_self_type %ModAssignWith.WithSelf.Op.type, %ModAssignWith.facet [symbolic = %.loc73_6.2 (constants.%.2b3)]
// CHECK:STDOUT:   %impl.elem0.loc73_6.2: @TestGenericAssign.%.loc73_6.2 (%.2b3) = impl_witness_access %ModAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc73_6.2 (constants.%impl.elem0.44b)]
// CHECK:STDOUT:   %specific_impl_fn.loc73_6.2: <specific function> = specific_impl_function %impl.elem0.loc73_6.2, @ModAssignWith.WithSelf.Op(%T.as_type.loc71_45.1, %ModAssignWith.facet) [symbolic = %specific_impl_fn.loc73_6.2 (constants.%specific_impl_fn.c3a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @TestGenericAssign.%ptr.loc71_45.1 (%ptr.360), %b.param: @TestGenericAssign.%T.as_type.loc71_45.1 (%T.as_type.190bb7.2)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %a.ref: @TestGenericAssign.%ptr.loc71_45.1 (%ptr.360) = name_ref a, %a
// CHECK:STDOUT:     %.loc73_3: ref @TestGenericAssign.%T.as_type.loc71_45.1 (%T.as_type.190bb7.2) = deref %a.ref
// CHECK:STDOUT:     %b.ref: @TestGenericAssign.%T.as_type.loc71_45.1 (%T.as_type.190bb7.2) = name_ref b, %b
// CHECK:STDOUT:     %ModAssignWith.type.loc73_6.1: type = facet_type <@ModAssignWith, @ModAssignWith(constants.%T.as_type.190bb7.2)> [symbolic = %ModAssignWith.type.loc73_6.2 (constants.%ModAssignWith.type.4760d9.2)]
// CHECK:STDOUT:     %.loc73_6.1: @TestGenericAssign.%ModAssignWith.assoc_type (%ModAssignWith.assoc_type.6b6315.2) = specific_constant imports.%Core.import_ref.1a2, @ModAssignWith.WithSelf(constants.%T.as_type.190bb7.2, constants.%Self.cc0) [symbolic = %assoc0 (constants.%assoc0.c31ad7.2)]
// CHECK:STDOUT:     %Op.ref: @TestGenericAssign.%ModAssignWith.assoc_type (%ModAssignWith.assoc_type.6b6315.2) = name_ref Op, %.loc73_6.1 [symbolic = %assoc0 (constants.%assoc0.c31ad7.2)]
// CHECK:STDOUT:     %impl.elem0.loc73_6.1: @TestGenericAssign.%.loc73_6.2 (%.2b3) = impl_witness_access constants.%ModAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc73_6.2 (constants.%impl.elem0.44b)]
// CHECK:STDOUT:     %bound_method.loc73_6.1: <bound method> = bound_method %.loc73_3, %impl.elem0.loc73_6.1
// CHECK:STDOUT:     %specific_impl_fn.loc73_6.1: <specific function> = specific_impl_function %impl.elem0.loc73_6.1, @ModAssignWith.WithSelf.Op(constants.%T.as_type.190bb7.2, constants.%ModAssignWith.facet.4ee) [symbolic = %specific_impl_fn.loc73_6.2 (constants.%specific_impl_fn.c3a)]
// CHECK:STDOUT:     %bound_method.loc73_6.2: <bound method> = bound_method %.loc73_3, %specific_impl_fn.loc73_6.1
// CHECK:STDOUT:     %ModAssignWith.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc73_6.2(%.loc73_3, %b.ref)
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
// CHECK:STDOUT: specific @TestGenericOp(constants.%Mod.facet.2a8) {
// CHECK:STDOUT:   %T.patt.loc65_19.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc65_19.1 => constants.%Mod.facet.2a8
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
// CHECK:STDOUT:   %ModWith.type.loc67_12.2 => constants.%ModWith.type.bf9
// CHECK:STDOUT:   %require_complete.loc67 => constants.%complete_type.9cd
// CHECK:STDOUT:   %ModWith.assoc_type => constants.%ModWith.assoc_type.828
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.871
// CHECK:STDOUT:   %ModWith.lookup_impl_witness => constants.%ModWith.impl_witness.f41
// CHECK:STDOUT:   %ModWith.facet => constants.%ModWith.facet.03b
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type => constants.%ModWith.WithSelf.Op.type.06b
// CHECK:STDOUT:   %.loc67_12.2 => constants.%.1bb
// CHECK:STDOUT:   %impl.elem1.loc67_12.2 => constants.%C.as.ModWith.impl.Op
// CHECK:STDOUT:   %specific_impl_fn.loc67_12.2 => constants.%C.as.ModWith.impl.Op
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGenericAssign(constants.%ModAssign.facet.85e) {
// CHECK:STDOUT:   %T.patt.loc71_23.2 => constants.%T.patt.3a0592.2
// CHECK:STDOUT:   %T.loc71_23.1 => constants.%ModAssign.facet.85e
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
// CHECK:STDOUT:   %ModAssignWith.type.loc73_6.2 => constants.%ModAssignWith.type.d51
// CHECK:STDOUT:   %require_complete.loc73 => constants.%complete_type.4bb
// CHECK:STDOUT:   %ModAssignWith.assoc_type => constants.%ModAssignWith.assoc_type.183
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.aeb
// CHECK:STDOUT:   %ModAssignWith.lookup_impl_witness => constants.%ModAssignWith.impl_witness.d6f
// CHECK:STDOUT:   %ModAssignWith.facet => constants.%ModAssignWith.facet.6e5
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type => constants.%ModAssignWith.WithSelf.Op.type.cb7
// CHECK:STDOUT:   %.loc73_6.2 => constants.%.651
// CHECK:STDOUT:   %impl.elem0.loc73_6.2 => constants.%C.as.ModAssignWith.impl.Op
// CHECK:STDOUT:   %specific_impl_fn.loc73_6.2 => constants.%C.as.ModAssignWith.impl.Op
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGenericOp(constants.%Mod.facet.6f9) {
// CHECK:STDOUT:   %T.patt.loc65_19.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc65_19.1 => constants.%Mod.facet.6f9
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
// CHECK:STDOUT:   %ModWith.type.loc67_12.2 => constants.%ModWith.type.d8b
// CHECK:STDOUT:   %require_complete.loc67 => constants.%complete_type.232
// CHECK:STDOUT:   %ModWith.assoc_type => constants.%ModWith.assoc_type.53e
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.c03
// CHECK:STDOUT:   %ModWith.lookup_impl_witness => constants.%ModWith.impl_witness.736
// CHECK:STDOUT:   %ModWith.facet => constants.%ModWith.facet.485
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type => constants.%ModWith.WithSelf.Op.type.525
// CHECK:STDOUT:   %.loc67_12.2 => constants.%.8b2
// CHECK:STDOUT:   %impl.elem1.loc67_12.2 => constants.%D.as.ModWith.impl.Op
// CHECK:STDOUT:   %specific_impl_fn.loc67_12.2 => constants.%D.as.ModWith.impl.Op
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGenericAssign(constants.%ModAssign.facet.6f9) {
// CHECK:STDOUT:   %T.patt.loc71_23.2 => constants.%T.patt.3a0592.2
// CHECK:STDOUT:   %T.loc71_23.1 => constants.%ModAssign.facet.6f9
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
// CHECK:STDOUT:   %ModAssignWith.type.loc73_6.2 => constants.%ModAssignWith.type.14f
// CHECK:STDOUT:   %require_complete.loc73 => constants.%complete_type.4c0
// CHECK:STDOUT:   %ModAssignWith.assoc_type => constants.%ModAssignWith.assoc_type.d05
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.67a
// CHECK:STDOUT:   %ModAssignWith.lookup_impl_witness => constants.%ModAssignWith.impl_witness.736
// CHECK:STDOUT:   %ModAssignWith.facet => constants.%ModAssignWith.facet.0ce
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type => constants.%ModAssignWith.WithSelf.Op.type.ec3
// CHECK:STDOUT:   %.loc73_6.2 => constants.%.2df
// CHECK:STDOUT:   %impl.elem0.loc73_6.2 => constants.%D.as.ModAssignWith.impl.Op
// CHECK:STDOUT:   %specific_impl_fn.loc73_6.2 => constants.%D.as.ModAssignWith.impl.Op
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGenericOp(constants.%Mod.facet.a1a) {
// CHECK:STDOUT:   %T.patt.loc65_19.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc65_19.1 => constants.%Mod.facet.a1a
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
// CHECK:STDOUT:   %ModWith.type.loc67_12.2 => constants.%ModWith.type.c3d
// CHECK:STDOUT:   %require_complete.loc67 => constants.%complete_type.093
// CHECK:STDOUT:   %ModWith.assoc_type => constants.%ModWith.assoc_type.61d
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.3f4
// CHECK:STDOUT:   %ModWith.lookup_impl_witness => constants.%ModWith.impl_witness.218
// CHECK:STDOUT:   %ModWith.facet => constants.%ModWith.facet.3c6
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type => constants.%ModWith.WithSelf.Op.type.759
// CHECK:STDOUT:   %.loc67_12.2 => constants.%.b3b
// CHECK:STDOUT:   %impl.elem1.loc67_12.2 => constants.%Int.as.ModWith.impl.Op.0bd
// CHECK:STDOUT:   %specific_impl_fn.loc67_12.2 => constants.%Int.as.ModWith.impl.Op.specific_fn.8c2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGenericAssign(constants.%ModAssign.facet.bf7) {
// CHECK:STDOUT:   %T.patt.loc71_23.2 => constants.%T.patt.3a0592.2
// CHECK:STDOUT:   %T.loc71_23.1 => constants.%ModAssign.facet.bf7
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
// CHECK:STDOUT:   %ModAssignWith.type.loc73_6.2 => constants.%ModAssignWith.type.124
// CHECK:STDOUT:   %require_complete.loc73 => constants.%complete_type.f89
// CHECK:STDOUT:   %ModAssignWith.assoc_type => constants.%ModAssignWith.assoc_type.dae
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.c87
// CHECK:STDOUT:   %ModAssignWith.lookup_impl_witness => constants.%ModAssignWith.impl_witness.8db
// CHECK:STDOUT:   %ModAssignWith.facet => constants.%ModAssignWith.facet.d8e
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type => constants.%ModAssignWith.WithSelf.Op.type.b54
// CHECK:STDOUT:   %.loc73_6.2 => constants.%.0bb
// CHECK:STDOUT:   %impl.elem0.loc73_6.2 => constants.%Int.as.ModAssignWith.impl.Op.12ab9f.2
// CHECK:STDOUT:   %specific_impl_fn.loc73_6.2 => constants.%Int.as.ModAssignWith.impl.Op.specific_fn.e61a9e.1
// CHECK:STDOUT: }
// CHECK:STDOUT:

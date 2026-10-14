// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/generic_method.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/generic_method.carbon

// --- generic_method_fewer_params.carbon

library "[[@TEST_NAME]]";

// T has index 0, Self has index 1, U has index 2.
interface A(T: type) {
  fn F(generic U: type, u: U*) -> (T, Self, U*);
}

class X {}
class Y {}
class Z {}

impl Y as A(X) {
  // Here, U has index 0. The specific used for A.F needs to renumber its U from
  // index 2 to index 0.
  fn F(generic U: type, u: U*) -> (X, Y, U*) {
    return ({}, {}, u);
  }
}

fn Call() {
  var u: Z;
  (Y as A(X)).F(Z, &u);
}

fn CallGeneric(generic T: A(X) & Core.Destroy) {
  var u: Z;
  T.F(Z, &u);
}

fn CallIndirect() {
  CallGeneric(Y);
}

// --- generic_method_more_params.carbon

library "[[@TEST_NAME]]";

// T has index 0, Self has index 1, U has index 2.
interface A(T: type) {
  fn F(generic U: type, u: U) -> (T, Self, U);
}

class X {}
class Y1 {}
class Y2 {}
class Z {}

impl forall [V1: type, V2: type, W: type] (V1, V2) as A(W) {
  // Here, U has index 3. The specific used for A.F needs to renumber its U from
  // index 2 to index 3.
  fn F(generic U: type, u: U) -> (W, (V1, V2), U) {
    return F(U, u);
  }
}

fn Call() {
  // TODO: The `as type` here should not be necessary.
  (((Y1, Y2) as type) as A(X)).F(Z, {});
}

fn CallGeneric(generic T: A(X) & Core.Destroy) {
  T.F(Z, {});
}

fn CallIndirect() {
  // TODO: The `as type` here should not be necessary.
  CallGeneric((Y1, Y2) as type);
}


// CHECK:STDOUT: --- generic_method_fewer_params.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt.3a0: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %A.type.55a: type = generic_interface_type @A [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %A.generic: %A.type.55a = struct_value () [concrete]
// CHECK:STDOUT:   %A.type.acd: type = facet_type <@A, @A(%T.67d)> [symbolic]
// CHECK:STDOUT:   %Self.49a: %A.type.acd = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %U.patt.5c4: %pattern_type.9a5 = symbolic_binding_pattern U, 2 [symbolic]
// CHECK:STDOUT:   %U.4e7: type = symbolic_binding U, 2 [symbolic]
// CHECK:STDOUT:   %ptr.8f6: type = ptr_type %U.4e7 [symbolic]
// CHECK:STDOUT:   %pattern_type.986: type = pattern_type %ptr.8f6 [symbolic]
// CHECK:STDOUT:   %u.param_patt.a01: %pattern_type.986 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %u.patt.d42: %pattern_type.986 = wrapper_binding_pattern u, %u.param_patt.a01 [symbolic]
// CHECK:STDOUT:   %tuple.type.a31: type = tuple_type (type, %A.type.acd, type) [symbolic]
// CHECK:STDOUT:   %tuple.a84: %tuple.type.a31 = tuple_value (%T.67d, %Self.49a, %ptr.8f6) [symbolic]
// CHECK:STDOUT:   %Self.as_type.763: type = facet_access_type %Self.49a [symbolic]
// CHECK:STDOUT:   %tuple.type.640: type = tuple_type (%T.67d, %Self.as_type.763, %ptr.8f6) [symbolic]
// CHECK:STDOUT:   %.f0d: Core.Form = init_form %tuple.type.640 [symbolic]
// CHECK:STDOUT:   %pattern_type.0d7: type = pattern_type %tuple.type.640 [symbolic]
// CHECK:STDOUT:   %return.param_patt.9d3: %pattern_type.0d7 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.8aa: %pattern_type.0d7 = return_slot_pattern %return.param_patt.9d3, %tuple.type.640 [symbolic]
// CHECK:STDOUT:   %A.WithSelf.F.type.08d: type = fn_type @A.WithSelf.F, @A.WithSelf(%T.67d, %Self.49a) [symbolic]
// CHECK:STDOUT:   %A.WithSelf.F.3fa: %A.WithSelf.F.type.08d = struct_value () [symbolic]
// CHECK:STDOUT:   %A.assoc_type.447: type = assoc_entity_type @A, @A(%T.67d) [symbolic]
// CHECK:STDOUT:   %assoc0.d4b: %A.assoc_type.447 = assoc_entity element0, @A.WithSelf.%A.WithSelf.F.decl [symbolic]
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Y: type = class_type @Y [concrete]
// CHECK:STDOUT:   %Z: type = class_type @Z [concrete]
// CHECK:STDOUT:   %A.type.df0: type = facet_type <@A, @A(%X)> [concrete]
// CHECK:STDOUT:   %.ef7: <witness> = impl_self_witness %Y, @A, @A(%X) [concrete]
// CHECK:STDOUT:   %A.impl_witness: <witness> = impl_witness @Y.as.A.impl.%A.impl_witness_table [concrete]
// CHECK:STDOUT:   %Self.891: %A.type.df0 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %A.WithSelf.F.type.ad4: type = fn_type @A.WithSelf.F, @A.WithSelf(%X, %Self.49a) [symbolic]
// CHECK:STDOUT:   %A.WithSelf.F.1e2: %A.WithSelf.F.type.ad4 = struct_value () [symbolic]
// CHECK:STDOUT:   %A.assoc_type.a54: type = assoc_entity_type @A, @A(%X) [concrete]
// CHECK:STDOUT:   %assoc0.987: %A.assoc_type.a54 = assoc_entity element0, @A.WithSelf.%A.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %U.patt.3a0: %pattern_type.9a5 = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %U.67d: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %ptr.e8f8f9.1: type = ptr_type %U.67d [symbolic]
// CHECK:STDOUT:   %pattern_type.4f4b84.1: type = pattern_type %ptr.e8f8f9.1 [symbolic]
// CHECK:STDOUT:   %u.param_patt.58b: %pattern_type.4f4b84.1 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %u.patt.60f: %pattern_type.4f4b84.1 = wrapper_binding_pattern u, %u.param_patt.58b [symbolic]
// CHECK:STDOUT:   %tuple.type.531: type = tuple_type (type, type, type) [concrete]
// CHECK:STDOUT:   %tuple.250: %tuple.type.531 = tuple_value (%X, %Y, %ptr.e8f8f9.1) [symbolic]
// CHECK:STDOUT:   %tuple.type.316: type = tuple_type (%X, %Y, %ptr.e8f8f9.1) [symbolic]
// CHECK:STDOUT:   %.3c1: Core.Form = init_form %tuple.type.316 [symbolic]
// CHECK:STDOUT:   %pattern_type.178: type = pattern_type %tuple.type.316 [symbolic]
// CHECK:STDOUT:   %return.param_patt.b96: %pattern_type.178 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.f2f: %pattern_type.178 = return_slot_pattern %return.param_patt.b96, %tuple.type.316 [symbolic]
// CHECK:STDOUT:   %Y.as.A.impl.F.type: type = fn_type @Y.as.A.impl.F [concrete]
// CHECK:STDOUT:   %Y.as.A.impl.F: %Y.as.A.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %A.facet.d1f: %A.type.df0 = facet_value %Y, (%A.impl_witness) [concrete]
// CHECK:STDOUT:   %A.WithSelf.F.type.781: type = fn_type @A.WithSelf.F, @A.WithSelf(%X, %A.facet.d1f) [concrete]
// CHECK:STDOUT:   %A.WithSelf.F.b66: %A.WithSelf.F.type.781 = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.202: type = tuple_type (type, %A.type.df0, type) [concrete]
// CHECK:STDOUT:   %tuple.e37: %tuple.type.202 = tuple_value (%X, %A.facet.d1f, %ptr.e8f8f9.1) [symbolic]
// CHECK:STDOUT:   %require_complete.ef162c.1: <witness> = require_complete_type %ptr.e8f8f9.1 [symbolic]
// CHECK:STDOUT:   %require_complete.44f: <witness> = require_complete_type %tuple.type.316 [symbolic]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.dd2: type = tuple_type (%empty_struct_type, %empty_struct_type, %ptr.e8f8f9.1) [symbolic]
// CHECK:STDOUT:   %X.val: %X = struct_value () [concrete]
// CHECK:STDOUT:   %Y.val: %Y = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.ff5f0d.1: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.ce912e.1: %ptr.as.Copy.impl.Op.type.ff5f0d.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.1da: <witness> = lookup_impl_witness %ptr.e8f8f9.1, @Copy [symbolic]
// CHECK:STDOUT:   %.0e9: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%U.67d) [symbolic]
// CHECK:STDOUT:   %Copy.facet.302: %Copy.type = facet_value %ptr.e8f8f9.1, (%Copy.lookup_impl_witness.1da) [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.884: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.302) [symbolic]
// CHECK:STDOUT:   %.be5: type = fn_type_with_self_type %Copy.WithSelf.Op.type.884, %Copy.facet.302 [symbolic]
// CHECK:STDOUT:   %impl.elem0.484: %.be5 = impl_witness_access %Copy.lookup_impl_witness.1da, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.a76: <specific function> = specific_impl_function %impl.elem0.484, @Copy.WithSelf.Op(%Copy.facet.302) [symbolic]
// CHECK:STDOUT:   %Call.type: type = fn_type @Call [concrete]
// CHECK:STDOUT:   %Call: %Call.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.34e: type = pattern_type %Z [concrete]
// CHECK:STDOUT:   %u.patt.352: %pattern_type.34e = ref_binding_pattern u [concrete]
// CHECK:STDOUT:   %u.var_patt: %pattern_type.34e = var_pattern %u.patt.352 [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.type.150: type = fn_type @T.as.DefaultOrUnformed.impl.Op, @T.as.DefaultOrUnformed.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.53a: %T.as.DefaultOrUnformed.impl.Op.type.150 = struct_value () [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.88e: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.856, @T.as.DefaultOrUnformed.impl(%Z) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.type.1d9: type = fn_type @T.as.DefaultOrUnformed.impl.Op, @T.as.DefaultOrUnformed.impl(%Z) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.f06: %T.as.DefaultOrUnformed.impl.Op.type.1d9 = struct_value () [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value %Z, (%DefaultOrUnformed.impl_witness.88e) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.specific_fn: <specific function> = specific_function %T.as.DefaultOrUnformed.impl.Op.f06, @T.as.DefaultOrUnformed.impl.Op(%Z) [concrete]
// CHECK:STDOUT:   %.261: type = fn_type_with_self_type %A.WithSelf.F.type.781, %A.facet.d1f [concrete]
// CHECK:STDOUT:   %ptr.415: type = ptr_type %Z [concrete]
// CHECK:STDOUT:   %pattern_type.b4b: type = pattern_type %ptr.415 [concrete]
// CHECK:STDOUT:   %u.param_patt.5c1: %pattern_type.b4b = value_param_pattern [concrete]
// CHECK:STDOUT:   %u.patt.d7f: %pattern_type.b4b = wrapper_binding_pattern u, %u.param_patt.5c1 [concrete]
// CHECK:STDOUT:   %tuple.b32: %tuple.type.531 = tuple_value (%X, %Y, %ptr.415) [concrete]
// CHECK:STDOUT:   %tuple.type.2cd: type = tuple_type (%X, %Y, %ptr.415) [concrete]
// CHECK:STDOUT:   %.5bc: Core.Form = init_form %tuple.type.2cd [concrete]
// CHECK:STDOUT:   %pattern_type.b33: type = pattern_type %tuple.type.2cd [concrete]
// CHECK:STDOUT:   %return.param_patt.75d: %pattern_type.b33 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.8bb: %pattern_type.b33 = return_slot_pattern %return.param_patt.75d, %tuple.type.2cd [concrete]
// CHECK:STDOUT:   %Y.as.A.impl.F.specific_fn: <specific function> = specific_function %Y.as.A.impl.F, @Y.as.A.impl.F(%Z) [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %facet_type.ece: type = facet_type <@DestroyImpl & @SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.273: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.5b5: %T.as_type.as.DestroyImpl.impl.Op.type.273 = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc23_22.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc23_22.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc23_22.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc23_22.5 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.ceb: %SubobjectDestroy.type = facet_value %ptr.415, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.3a0: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.ceb) [concrete]
// CHECK:STDOUT:   %complete_type.458: <witness> = complete_type_witness %ptr.415 [concrete]
// CHECK:STDOUT:   %facet_value.da6: %facet_type.ece = facet_value %ptr.415, (%DestroyImpl.impl_witness.3a0, %custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.b83: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%facet_value.da6) [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.8db: %SubobjectDestroy.type = facet_value %Y, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.691: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.8db) [concrete]
// CHECK:STDOUT:   %facet_value.391: %facet_type.ece = facet_value %Y, (%DestroyImpl.impl_witness.691, %custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.e36: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%facet_value.391) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.fc5: %SubobjectDestroy.type = facet_value %X, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.678: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.fc5) [concrete]
// CHECK:STDOUT:   %facet_value.9d2: %facet_type.ece = facet_value %X, (%DestroyImpl.impl_witness.678, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.e0d: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%facet_value.9d2) [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.c56: %SubobjectDestroy.type = facet_value %tuple.type.2cd, (%custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.cb9: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.c56) [concrete]
// CHECK:STDOUT:   %complete_type.dec: <witness> = complete_type_witness %tuple.type.2cd [concrete]
// CHECK:STDOUT:   %.081: require_specific_def_type = require_specific_def @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.c56) [concrete]
// CHECK:STDOUT:   %facet_value.219: %facet_type.ece = facet_value %tuple.type.2cd, (%DestroyImpl.impl_witness.cb9, %custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.da0: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%facet_value.219) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.6: type = fn_type @SubobjectDestroy.WithSelf.Op.loc22 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.6: %SubobjectDestroy.WithSelf.Op.type.6870fd.6 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.6: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.6), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.5d5: %SubobjectDestroy.type = facet_value %Z, (%custom_witness.b80ef6.6) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.72a: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.5d5) [concrete]
// CHECK:STDOUT:   %facet_value.469: %facet_type.ece = facet_value %Z, (%DestroyImpl.impl_witness.72a, %custom_witness.b80ef6.6) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.c54: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%facet_value.469) [concrete]
// CHECK:STDOUT:   %BitAndWith.type.a06: type = generic_interface_type @BitAndWith [concrete]
// CHECK:STDOUT:   %BitAndWith.generic: %BitAndWith.type.a06 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndWith.type.411: type = facet_type <@BitAndWith, @BitAndWith(type)> [concrete]
// CHECK:STDOUT:   %BitAndWith.impl_witness: <witness> = impl_witness imports.%BitAndWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %BitAndWith.facet: %BitAndWith.type.411 = facet_value type, (%BitAndWith.impl_witness) [concrete]
// CHECK:STDOUT:   %BitAndWith.WithSelf.Op.type.0cf: type = fn_type @BitAndWith.WithSelf.Op, @BitAndWith.WithSelf(type, %BitAndWith.facet) [concrete]
// CHECK:STDOUT:   %.85c: type = fn_type_with_self_type %BitAndWith.WithSelf.Op.type.0cf, %BitAndWith.facet [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.type: type = fn_type @type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op: %type.as.BitAndWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.bound: <bound method> = bound_method %A.type.df0, %type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %facet_type.572: type = facet_type <@A, @A(%X) & @DestroyImpl & @SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %pattern_type.03d: type = pattern_type %facet_type.572 [concrete]
// CHECK:STDOUT:   %T.patt.4c0: %pattern_type.03d = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.80e: %facet_type.572 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %CallGeneric.type: type = fn_type @CallGeneric [concrete]
// CHECK:STDOUT:   %CallGeneric: %CallGeneric.type = struct_value () [concrete]
// CHECK:STDOUT:   %A.WithSelf.F.type.298: type = fn_type @A.WithSelf.F, @A.WithSelf(%X, %T.80e) [symbolic]
// CHECK:STDOUT:   %A.WithSelf.F.d25: %A.WithSelf.F.type.298 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.8e3: type = facet_access_type %T.80e [symbolic]
// CHECK:STDOUT:   %A.lookup_impl_witness: <witness> = lookup_impl_witness %T.80e, @A, @A(%X) [symbolic]
// CHECK:STDOUT:   %A.facet.5d2: %A.type.df0 = facet_value %T.as_type.8e3, (%A.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %A.WithSelf.F.type.9bb: type = fn_type @A.WithSelf.F, @A.WithSelf(%X, %A.facet.5d2) [symbolic]
// CHECK:STDOUT:   %A.WithSelf.F.8c8: %A.WithSelf.F.type.9bb = struct_value () [symbolic]
// CHECK:STDOUT:   %.ad0: type = fn_type_with_self_type %A.WithSelf.F.type.9bb, %A.facet.5d2 [symbolic]
// CHECK:STDOUT:   %impl.elem0.327: %.ad0 = impl_witness_access %A.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %tuple.664: %tuple.type.202 = tuple_value (%X, %A.facet.5d2, %ptr.415) [symbolic]
// CHECK:STDOUT:   %tuple.type.e1c: type = tuple_type (%X, %T.as_type.8e3, %ptr.415) [symbolic]
// CHECK:STDOUT:   %.1c24: Core.Form = init_form %tuple.type.e1c [symbolic]
// CHECK:STDOUT:   %pattern_type.75d: type = pattern_type %tuple.type.e1c [symbolic]
// CHECK:STDOUT:   %return.param_patt.32a: %pattern_type.75d = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.13c: %pattern_type.75d = return_slot_pattern %return.param_patt.32a, %tuple.type.e1c [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.b41: <specific function> = specific_impl_function %impl.elem0.327, @A.WithSelf.F(%X, %A.facet.5d2, %Z) [symbolic]
// CHECK:STDOUT:   %require_complete.067: <witness> = require_complete_type %tuple.type.e1c [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.6a7: <witness> = lookup_impl_witness %tuple.type.e1c, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.a7e: <witness> = lookup_impl_witness %tuple.type.e1c, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.facet.f49: %SubobjectDestroy.type = facet_value %tuple.type.e1c, (%SubobjectDestroy.lookup_impl_witness.a7e) [symbolic]
// CHECK:STDOUT:   %.d22: require_specific_def_type = require_specific_def @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.f49) [symbolic]
// CHECK:STDOUT:   %facet_value.744: %facet_type.ece = facet_value %tuple.type.e1c, (%DestroyImpl.lookup_impl_witness.6a7, %SubobjectDestroy.lookup_impl_witness.a7e) [symbolic]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.3c0: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%facet_value.744) [symbolic]
// CHECK:STDOUT:   %CallIndirect.type: type = fn_type @CallIndirect [concrete]
// CHECK:STDOUT:   %CallIndirect: %CallIndirect.type = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.a83: %facet_type.572 = facet_value %Y, (%A.impl_witness, %DestroyImpl.impl_witness.691, %custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %CallGeneric.specific_fn: <specific function> = specific_function %CallGeneric, @CallGeneric(%facet_value.a83) [concrete]
// CHECK:STDOUT:   %tuple.type.2c5: type = tuple_type (%empty_struct_type, %empty_struct_type, %ptr.415) [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.cff: <witness> = impl_witness imports.%Copy.impl_witness_table.fb0, @ptr.as.Copy.impl(%Z) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.905: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%Z) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.ebe: %ptr.as.Copy.impl.Op.type.905 = struct_value () [concrete]
// CHECK:STDOUT:   %.072: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%Z) [concrete]
// CHECK:STDOUT:   %Copy.facet.9c6: %Copy.type = facet_value %ptr.415, (%Copy.impl_witness.cff) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.107: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.9c6) [concrete]
// CHECK:STDOUT:   %.08f: type = fn_type_with_self_type %Copy.WithSelf.Op.type.107, %Copy.facet.9c6 [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.ebe, @ptr.as.Copy.impl.Op(%Z) [concrete]
// CHECK:STDOUT:   %tuple.d83: %tuple.type.202 = tuple_value (%X, %A.facet.d1f, %ptr.415) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .DefaultOrUnformed = %Core.DefaultOrUnformed
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .BitAndWith = %Core.BitAndWith
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.e5f: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.ff5f0d.1) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.ce912e.1)]
// CHECK:STDOUT:   %Copy.impl_witness_table.fb0 = impl_witness_table (%Core.import_ref.e5f), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.DefaultOrUnformed: type = import_ref Core//prelude/parts/default, DefaultOrUnformed, loaded [concrete = constants.%DefaultOrUnformed.type]
// CHECK:STDOUT:   %Core.import_ref.cc8: @T.as.DefaultOrUnformed.impl.%T.as.DefaultOrUnformed.impl.Op.type (%T.as.DefaultOrUnformed.impl.Op.type.150) = import_ref Core//prelude/parts/default, loc{{\d+_\d+}}, loaded [symbolic = @T.as.DefaultOrUnformed.impl.%T.as.DefaultOrUnformed.impl.Op (constants.%T.as.DefaultOrUnformed.impl.Op.53a)]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness_table.856 = impl_witness_table (%Core.import_ref.cc8), @T.as.DefaultOrUnformed.impl [concrete]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT:   %Core.import_ref.688: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.273) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.5b5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.688), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%facet_type.ece]
// CHECK:STDOUT:   %Core.BitAndWith: %BitAndWith.type.a06 = import_ref Core//prelude/parts/as, BitAndWith, loaded [concrete = constants.%BitAndWith.generic]
// CHECK:STDOUT:   %Core.import_ref.9c3: %type.as.BitAndWith.impl.Op.type = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, loaded [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:   %BitAndWith.impl_witness_table = impl_witness_table (%Core.import_ref.9c3), @type.as.BitAndWith.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .X = %X.decl
// CHECK:STDOUT:     .Y = %Y.decl
// CHECK:STDOUT:     .Z = %Z.decl
// CHECK:STDOUT:     .Call = %Call.decl
// CHECK:STDOUT:     .CallGeneric = %CallGeneric.decl
// CHECK:STDOUT:     .CallIndirect = %CallIndirect.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: %A.type.55a = interface_decl @A [concrete = constants.%A.generic] {
// CHECK:STDOUT:     %T.patt.loc5_14.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc5_14.2 (constants.%T.patt.3a0)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5_16.1: type = splice_block %.loc5_16.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc5_16.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc5_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc5_14.1 (constants.%T.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.decl: type = class_decl @X [concrete = constants.%X] {} {}
// CHECK:STDOUT:   %Y.decl: type = class_decl @Y [concrete = constants.%Y] {} {}
// CHECK:STDOUT:   %Z.decl: type = class_decl @Z [concrete = constants.%Z] {} {}
// CHECK:STDOUT:   impl_decl @Y.as.A.impl [concrete] {} {
// CHECK:STDOUT:     %Y.ref: type = name_ref Y, file.%Y.decl [concrete = constants.%Y]
// CHECK:STDOUT:     %A.ref: %A.type.55a = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %A.type: type = facet_type <@A, @A(constants.%X)> [concrete = constants.%A.type.df0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc13: <witness> = impl_self_witness @Y.as.A.impl.%Y.ref, @A, @A(constants.%X) [concrete = constants.%.ef7]
// CHECK:STDOUT:   %Call.decl: %Call.type = fn_decl @Call [concrete = constants.%Call] {} {}
// CHECK:STDOUT:   %CallGeneric.decl: %CallGeneric.type = fn_decl @CallGeneric [concrete = constants.%CallGeneric] {
// CHECK:STDOUT:     %T.patt.loc26_25.1: %pattern_type.03d = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc26_25.2 (constants.%T.patt.4c0)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc26_32.1: type = splice_block %.loc26_32.3 [concrete = constants.%facet_type.572] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %A.ref: %A.type.55a = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:       %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:       %A.type: type = facet_type <@A, @A(constants.%X)> [concrete = constants.%A.type.df0]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%facet_type.ece]
// CHECK:STDOUT:       %impl.elem0.loc26: %.85c = impl_witness_access constants.%BitAndWith.impl_witness, element0 [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:       %bound_method: <bound method> = bound_method %A.type, %impl.elem0.loc26 [concrete = constants.%type.as.BitAndWith.impl.Op.bound]
// CHECK:STDOUT:       %type.as.BitAndWith.impl.Op.call: init type = call %bound_method(%A.type, %Destroy.ref) [concrete = constants.%facet_type.572]
// CHECK:STDOUT:       %.loc26_32.2: type = value_of_initializer %type.as.BitAndWith.impl.Op.call [concrete = constants.%facet_type.572]
// CHECK:STDOUT:       %.loc26_32.3: type = converted %type.as.BitAndWith.impl.Op.call, %.loc26_32.2 [concrete = constants.%facet_type.572]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc26_25.2: %facet_type.572 = symbolic_binding T, 0 [symbolic = %T.loc26_25.1 (constants.%T.80e)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallIndirect.decl: %CallIndirect.type = fn_decl @CallIndirect [concrete = constants.%CallIndirect] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @A(%T.loc5_14.2: type) {
// CHECK:STDOUT:   %T.patt.loc5_14.2: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc5_14.2 (constants.%T.patt.3a0)]
// CHECK:STDOUT:   %T.loc5_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc5_14.1 (constants.%T.67d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.type: type = facet_type <@A, @A(%T.loc5_14.1)> [symbolic = %A.type (constants.%A.type.acd)]
// CHECK:STDOUT:   %Self.loc5_22.2: @A.%A.type (%A.type.acd) = symbolic_binding Self, 1 [symbolic = %Self.loc5_22.2 (constants.%Self.49a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc5_22.1: @A.%A.type (%A.type.acd) = symbolic_binding Self, 1 [symbolic = %Self.loc5_22.2 (constants.%Self.49a)]
// CHECK:STDOUT:     %A.WithSelf.decl = interface_with_self_decl @A [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %A.WithSelf.F.decl: @A.WithSelf.%A.WithSelf.F.type (%A.WithSelf.F.type.08d) = fn_decl @A.WithSelf.F [symbolic = @A.WithSelf.%A.WithSelf.F (constants.%A.WithSelf.F.3fa)] {
// CHECK:STDOUT:       %U.patt.loc6_17.1: %pattern_type.9a5 = symbolic_binding_pattern U, 2 [symbolic = %U.patt.loc6_17.2 (constants.%U.patt.5c4)]
// CHECK:STDOUT:       %u.param_patt.loc6_26.1: @A.WithSelf.F.%pattern_type.loc6_26 (%pattern_type.986) = value_param_pattern [symbolic = %u.param_patt.loc6_26.2 (constants.%u.param_patt.a01)]
// CHECK:STDOUT:       %u.patt.loc6_26.1: @A.WithSelf.F.%pattern_type.loc6_26 (%pattern_type.986) = wrapper_binding_pattern u, %u.param_patt.loc6_26.1 [symbolic = %u.patt.loc6_26.2 (constants.%u.patt.d42)]
// CHECK:STDOUT:       %return.param_patt.loc6_47.1: @A.WithSelf.F.%pattern_type.loc6_47 (%pattern_type.0d7) = out_param_pattern [symbolic = %return.param_patt.loc6_47.2 (constants.%return.param_patt.9d3)]
// CHECK:STDOUT:       %return.patt.loc6_32.1: @A.WithSelf.F.%pattern_type.loc6_47 (%pattern_type.0d7) = return_slot_pattern %return.param_patt.loc6_47.1, %.loc6_47.4 [symbolic = %return.patt.loc6_32.2 (constants.%return.patt.8aa)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @A.%T.loc5_14.2 [symbolic = %T (constants.%T.67d)]
// CHECK:STDOUT:       %.loc6_39: @A.WithSelf.F.%A.type (%A.type.acd) = specific_constant @A.%Self.loc5_22.1, @A(constants.%T.67d) [symbolic = %Self (constants.%Self.49a)]
// CHECK:STDOUT:       %Self.ref: @A.WithSelf.F.%A.type (%A.type.acd) = name_ref Self, %.loc6_39 [symbolic = %Self (constants.%Self.49a)]
// CHECK:STDOUT:       %U.ref.loc6_45: type = name_ref U, %U.loc6_17.2 [symbolic = %U.loc6_17.1 (constants.%U.4e7)]
// CHECK:STDOUT:       %ptr.loc6_46: type = ptr_type %U.ref.loc6_45 [symbolic = %ptr.loc6_29.1 (constants.%ptr.8f6)]
// CHECK:STDOUT:       %.loc6_47.2: @A.WithSelf.F.%tuple.type.loc6_47.1 (%tuple.type.a31) = tuple_literal (%T.ref, %Self.ref, %ptr.loc6_46) [symbolic = %tuple (constants.%tuple.a84)]
// CHECK:STDOUT:       %Self.as_type.loc6_47.2: type = facet_access_type constants.%Self.49a [symbolic = %Self.as_type.loc6_47.1 (constants.%Self.as_type.763)]
// CHECK:STDOUT:       %.loc6_47.3: type = converted constants.%Self.49a, %Self.as_type.loc6_47.2 [symbolic = %Self.as_type.loc6_47.1 (constants.%Self.as_type.763)]
// CHECK:STDOUT:       %.loc6_47.4: type = converted %.loc6_47.2, constants.%tuple.type.640 [symbolic = %tuple.type.loc6_47.2 (constants.%tuple.type.640)]
// CHECK:STDOUT:       %.loc6_47.5: Core.Form = init_form %.loc6_47.4 [symbolic = %.loc6_47.1 (constants.%.f0d)]
// CHECK:STDOUT:       %.loc6_19.1: type = splice_block %.loc6_19.2 [concrete = type] {
// CHECK:STDOUT:         %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:         %.loc6_19.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %U.loc6_17.2: type = symbolic_binding U, 2 [symbolic = %U.loc6_17.1 (constants.%U.4e7)]
// CHECK:STDOUT:       %u.param: @A.WithSelf.F.%ptr.loc6_29.1 (%ptr.8f6) = value_param call_param0
// CHECK:STDOUT:       %.loc6_29: type = splice_block %ptr.loc6_29.2 [symbolic = %ptr.loc6_29.1 (constants.%ptr.8f6)] {
// CHECK:STDOUT:         %U.ref.loc6_28: type = name_ref U, %U.loc6_17.2 [symbolic = %U.loc6_17.1 (constants.%U.4e7)]
// CHECK:STDOUT:         %ptr.loc6_29.2: type = ptr_type %U.ref.loc6_28 [symbolic = %ptr.loc6_29.1 (constants.%ptr.8f6)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %u: @A.WithSelf.F.%ptr.loc6_29.1 (%ptr.8f6) = wrapper_binding u, %u.param
// CHECK:STDOUT:       %return.param: ref @A.WithSelf.F.%tuple.type.loc6_47.2 (%tuple.type.640) = out_param call_param1
// CHECK:STDOUT:       %return: ref @A.WithSelf.F.%tuple.type.loc6_47.2 (%tuple.type.640) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc6_48.1: @A.WithSelf.%A.assoc_type (%A.assoc_type.447) = assoc_entity element0, %A.WithSelf.F.decl [symbolic = %assoc0.loc6_48.2 (constants.%assoc0.d4b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc5_22.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .F = @A.WithSelf.%assoc0.loc6_48.1
// CHECK:STDOUT:     .X = <poisoned>
// CHECK:STDOUT:     .Y = <poisoned>
// CHECK:STDOUT:     witness = (@A.WithSelf.%A.WithSelf.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Y.as.A.impl: %Y.ref as %A.type {
// CHECK:STDOUT:   %Y.as.A.impl.F.decl: %Y.as.A.impl.F.type = fn_decl @Y.as.A.impl.F [concrete = constants.%Y.as.A.impl.F] {
// CHECK:STDOUT:     %U.patt.loc16_17.1: %pattern_type.9a5 = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc16_17.2 (constants.%U.patt.3a0)]
// CHECK:STDOUT:     %u.param_patt.loc16_26.1: @Y.as.A.impl.F.%pattern_type.loc16_26 (%pattern_type.4f4b84.1) = value_param_pattern [symbolic = %u.param_patt.loc16_26.2 (constants.%u.param_patt.58b)]
// CHECK:STDOUT:     %u.patt.loc16_26.1: @Y.as.A.impl.F.%pattern_type.loc16_26 (%pattern_type.4f4b84.1) = wrapper_binding_pattern u, %u.param_patt.loc16_26.1 [symbolic = %u.patt.loc16_26.2 (constants.%u.patt.60f)]
// CHECK:STDOUT:     %return.param_patt.loc16_44.1: @Y.as.A.impl.F.%pattern_type.loc16_44 (%pattern_type.178) = out_param_pattern [symbolic = %return.param_patt.loc16_44.2 (constants.%return.param_patt.b96)]
// CHECK:STDOUT:     %return.patt.loc16_32.1: @Y.as.A.impl.F.%pattern_type.loc16_44 (%pattern_type.178) = return_slot_pattern %return.param_patt.loc16_44.1, %.loc16_44.3 [symbolic = %return.patt.loc16_32.2 (constants.%return.patt.f2f)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %Y.ref: type = name_ref Y, file.%Y.decl [concrete = constants.%Y]
// CHECK:STDOUT:     %U.ref.loc16_42: type = name_ref U, %U.loc16_17.2 [symbolic = %U.loc16_17.1 (constants.%U.67d)]
// CHECK:STDOUT:     %ptr.loc16_43: type = ptr_type %U.ref.loc16_42 [symbolic = %ptr.loc16_29.1 (constants.%ptr.e8f8f9.1)]
// CHECK:STDOUT:     %.loc16_44.2: %tuple.type.531 = tuple_literal (%X.ref, %Y.ref, %ptr.loc16_43) [symbolic = %tuple (constants.%tuple.250)]
// CHECK:STDOUT:     %.loc16_44.3: type = converted %.loc16_44.2, constants.%tuple.type.316 [symbolic = %tuple.type.loc16 (constants.%tuple.type.316)]
// CHECK:STDOUT:     %.loc16_44.4: Core.Form = init_form %.loc16_44.3 [symbolic = %.loc16_44.1 (constants.%.3c1)]
// CHECK:STDOUT:     %.loc16_19.1: type = splice_block %.loc16_19.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc16_19.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc16_17.2: type = symbolic_binding U, 0 [symbolic = %U.loc16_17.1 (constants.%U.67d)]
// CHECK:STDOUT:     %u.param: @Y.as.A.impl.F.%ptr.loc16_29.1 (%ptr.e8f8f9.1) = value_param call_param0
// CHECK:STDOUT:     %.loc16_29: type = splice_block %ptr.loc16_29.2 [symbolic = %ptr.loc16_29.1 (constants.%ptr.e8f8f9.1)] {
// CHECK:STDOUT:       %U.ref.loc16_28: type = name_ref U, %U.loc16_17.2 [symbolic = %U.loc16_17.1 (constants.%U.67d)]
// CHECK:STDOUT:       %ptr.loc16_29.2: type = ptr_type %U.ref.loc16_28 [symbolic = %ptr.loc16_29.1 (constants.%ptr.e8f8f9.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: @Y.as.A.impl.F.%ptr.loc16_29.1 (%ptr.e8f8f9.1) = wrapper_binding u, %u.param
// CHECK:STDOUT:     %return.param: ref @Y.as.A.impl.F.%tuple.type.loc16 (%tuple.type.316) = out_param call_param1
// CHECK:STDOUT:     %return: ref @Y.as.A.impl.F.%tuple.type.loc16 (%tuple.type.316) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.impl_witness_table = impl_witness_table (%Y.as.A.impl.F.decl), @Y.as.A.impl [concrete]
// CHECK:STDOUT:   %A.impl_witness: <witness> = impl_witness %A.impl_witness_table [concrete = constants.%A.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .X = <poisoned>
// CHECK:STDOUT:   .Y = <poisoned>
// CHECK:STDOUT:   .F = %Y.as.A.impl.F.decl
// CHECK:STDOUT:   extend %A.type
// CHECK:STDOUT:   witness = %A.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @X {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Y {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Y
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Z {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Z
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @A.WithSelf.F(@A.%T.loc5_14.2: type, @A.%Self.loc5_22.1: @A.%A.type (%A.type.acd), %U.loc6_17.2: type) {
// CHECK:STDOUT:   %U.patt.loc6_17.2: %pattern_type.9a5 = symbolic_binding_pattern U, 2 [symbolic = %U.patt.loc6_17.2 (constants.%U.patt.5c4)]
// CHECK:STDOUT:   %U.loc6_17.1: type = symbolic_binding U, 2 [symbolic = %U.loc6_17.1 (constants.%U.4e7)]
// CHECK:STDOUT:   %ptr.loc6_29.1: type = ptr_type %U.loc6_17.1 [symbolic = %ptr.loc6_29.1 (constants.%ptr.8f6)]
// CHECK:STDOUT:   %pattern_type.loc6_26: type = pattern_type %ptr.loc6_29.1 [symbolic = %pattern_type.loc6_26 (constants.%pattern_type.986)]
// CHECK:STDOUT:   %u.param_patt.loc6_26.2: @A.WithSelf.F.%pattern_type.loc6_26 (%pattern_type.986) = value_param_pattern [symbolic = %u.param_patt.loc6_26.2 (constants.%u.param_patt.a01)]
// CHECK:STDOUT:   %u.patt.loc6_26.2: @A.WithSelf.F.%pattern_type.loc6_26 (%pattern_type.986) = wrapper_binding_pattern u, %u.param_patt.loc6_26.2 [symbolic = %u.patt.loc6_26.2 (constants.%u.patt.d42)]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T.67d)]
// CHECK:STDOUT:   %A.type: type = facet_type <@A, @A(%T)> [symbolic = %A.type (constants.%A.type.acd)]
// CHECK:STDOUT:   %Self: @A.WithSelf.F.%A.type (%A.type.acd) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.49a)]
// CHECK:STDOUT:   %tuple.type.loc6_47.1: type = tuple_type (type, %A.type, type) [symbolic = %tuple.type.loc6_47.1 (constants.%tuple.type.a31)]
// CHECK:STDOUT:   %tuple: @A.WithSelf.F.%tuple.type.loc6_47.1 (%tuple.type.a31) = tuple_value (%T, %Self, %ptr.loc6_29.1) [symbolic = %tuple (constants.%tuple.a84)]
// CHECK:STDOUT:   %Self.as_type.loc6_47.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc6_47.1 (constants.%Self.as_type.763)]
// CHECK:STDOUT:   %tuple.type.loc6_47.2: type = tuple_type (%T, %Self.as_type.loc6_47.1, %ptr.loc6_29.1) [symbolic = %tuple.type.loc6_47.2 (constants.%tuple.type.640)]
// CHECK:STDOUT:   %.loc6_47.1: Core.Form = init_form %tuple.type.loc6_47.2 [symbolic = %.loc6_47.1 (constants.%.f0d)]
// CHECK:STDOUT:   %pattern_type.loc6_47: type = pattern_type %tuple.type.loc6_47.2 [symbolic = %pattern_type.loc6_47 (constants.%pattern_type.0d7)]
// CHECK:STDOUT:   %return.param_patt.loc6_47.2: @A.WithSelf.F.%pattern_type.loc6_47 (%pattern_type.0d7) = out_param_pattern [symbolic = %return.param_patt.loc6_47.2 (constants.%return.param_patt.9d3)]
// CHECK:STDOUT:   %return.patt.loc6_32.2: @A.WithSelf.F.%pattern_type.loc6_47 (%pattern_type.0d7) = return_slot_pattern %return.param_patt.loc6_47.2, %tuple.type.loc6_47.2 [symbolic = %return.patt.loc6_32.2 (constants.%return.patt.8aa)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param: @A.WithSelf.F.%ptr.loc6_29.1 (%ptr.8f6)) -> out %return.param: @A.WithSelf.F.%tuple.type.loc6_47.2 (%tuple.type.640);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Y.as.A.impl.F(%U.loc16_17.2: type) {
// CHECK:STDOUT:   %U.patt.loc16_17.2: %pattern_type.9a5 = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc16_17.2 (constants.%U.patt.3a0)]
// CHECK:STDOUT:   %U.loc16_17.1: type = symbolic_binding U, 0 [symbolic = %U.loc16_17.1 (constants.%U.67d)]
// CHECK:STDOUT:   %ptr.loc16_29.1: type = ptr_type %U.loc16_17.1 [symbolic = %ptr.loc16_29.1 (constants.%ptr.e8f8f9.1)]
// CHECK:STDOUT:   %pattern_type.loc16_26: type = pattern_type %ptr.loc16_29.1 [symbolic = %pattern_type.loc16_26 (constants.%pattern_type.4f4b84.1)]
// CHECK:STDOUT:   %u.param_patt.loc16_26.2: @Y.as.A.impl.F.%pattern_type.loc16_26 (%pattern_type.4f4b84.1) = value_param_pattern [symbolic = %u.param_patt.loc16_26.2 (constants.%u.param_patt.58b)]
// CHECK:STDOUT:   %u.patt.loc16_26.2: @Y.as.A.impl.F.%pattern_type.loc16_26 (%pattern_type.4f4b84.1) = wrapper_binding_pattern u, %u.param_patt.loc16_26.2 [symbolic = %u.patt.loc16_26.2 (constants.%u.patt.60f)]
// CHECK:STDOUT:   %tuple: %tuple.type.531 = tuple_value (constants.%X, constants.%Y, %ptr.loc16_29.1) [symbolic = %tuple (constants.%tuple.250)]
// CHECK:STDOUT:   %tuple.type.loc16: type = tuple_type (constants.%X, constants.%Y, %ptr.loc16_29.1) [symbolic = %tuple.type.loc16 (constants.%tuple.type.316)]
// CHECK:STDOUT:   %.loc16_44.1: Core.Form = init_form %tuple.type.loc16 [symbolic = %.loc16_44.1 (constants.%.3c1)]
// CHECK:STDOUT:   %pattern_type.loc16_44: type = pattern_type %tuple.type.loc16 [symbolic = %pattern_type.loc16_44 (constants.%pattern_type.178)]
// CHECK:STDOUT:   %return.param_patt.loc16_44.2: @Y.as.A.impl.F.%pattern_type.loc16_44 (%pattern_type.178) = out_param_pattern [symbolic = %return.param_patt.loc16_44.2 (constants.%return.param_patt.b96)]
// CHECK:STDOUT:   %return.patt.loc16_32.2: @Y.as.A.impl.F.%pattern_type.loc16_44 (%pattern_type.178) = return_slot_pattern %return.param_patt.loc16_44.2, %tuple.type.loc16 [symbolic = %return.patt.loc16_32.2 (constants.%return.patt.f2f)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc16_26: <witness> = require_complete_type %ptr.loc16_29.1 [symbolic = %require_complete.loc16_26 (constants.%require_complete.ef162c.1)]
// CHECK:STDOUT:   %require_complete.loc16_32: <witness> = require_complete_type %tuple.type.loc16 [symbolic = %require_complete.loc16_32 (constants.%require_complete.44f)]
// CHECK:STDOUT:   %tuple.type.loc17: type = tuple_type (constants.%empty_struct_type, constants.%empty_struct_type, %ptr.loc16_29.1) [symbolic = %tuple.type.loc17 (constants.%tuple.type.dd2)]
// CHECK:STDOUT:   %.loc17_21.3: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%U.loc16_17.1) [symbolic = %.loc17_21.3 (constants.%.0e9)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr.loc16_29.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.1da)]
// CHECK:STDOUT:   %Copy.facet.loc17_21.3: %Copy.type = facet_value %ptr.loc16_29.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet.loc17_21.3 (constants.%Copy.facet.302)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.loc17_21.3) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.884)]
// CHECK:STDOUT:   %.loc17_21.4: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %Copy.facet.loc17_21.3 [symbolic = %.loc17_21.4 (constants.%.be5)]
// CHECK:STDOUT:   %impl.elem0.loc17_21.2: @Y.as.A.impl.F.%.loc17_21.4 (%.be5) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc17_21.2 (constants.%impl.elem0.484)]
// CHECK:STDOUT:   %specific_impl_fn.loc17_21.2: <specific function> = specific_impl_function %impl.elem0.loc17_21.2, @Copy.WithSelf.Op(%Copy.facet.loc17_21.3) [symbolic = %specific_impl_fn.loc17_21.2 (constants.%specific_impl_fn.a76)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param: @Y.as.A.impl.F.%ptr.loc16_29.1 (%ptr.e8f8f9.1)) -> out %return.param: @Y.as.A.impl.F.%tuple.type.loc16 (%tuple.type.316) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc17_14.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc17_18.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %u.ref: @Y.as.A.impl.F.%ptr.loc16_29.1 (%ptr.e8f8f9.1) = name_ref u, %u
// CHECK:STDOUT:     %.loc17_22.1: @Y.as.A.impl.F.%tuple.type.loc17 (%tuple.type.dd2) = tuple_literal (%.loc17_14.1, %.loc17_18.1, %u.ref)
// CHECK:STDOUT:     %tuple.elem0: ref %X = tuple_access %return.param, element0
// CHECK:STDOUT:     %.loc17_14.2: init %X to %tuple.elem0 = class_init () [concrete = constants.%X.val]
// CHECK:STDOUT:     %.loc17_22.2: init %X = converted %.loc17_14.1, %.loc17_14.2 [concrete = constants.%X.val]
// CHECK:STDOUT:     %tuple.elem1: ref %Y = tuple_access %return.param, element1
// CHECK:STDOUT:     %.loc17_18.2: init %Y to %tuple.elem1 = class_init () [concrete = constants.%Y.val]
// CHECK:STDOUT:     %.loc17_22.3: init %Y = converted %.loc17_18.1, %.loc17_18.2 [concrete = constants.%Y.val]
// CHECK:STDOUT:     %impl.elem0.loc17_21.1: @Y.as.A.impl.F.%.loc17_21.4 (%.be5) = impl_witness_access constants.%Copy.lookup_impl_witness.1da, element0 [symbolic = %impl.elem0.loc17_21.2 (constants.%impl.elem0.484)]
// CHECK:STDOUT:     %bound_method.loc17_21.1: <bound method> = bound_method %u.ref, %impl.elem0.loc17_21.1
// CHECK:STDOUT:     %Copy.facet.loc17_21.1: %Copy.type = facet_value constants.%ptr.e8f8f9.1, (constants.%Copy.lookup_impl_witness.1da) [symbolic = %Copy.facet.loc17_21.3 (constants.%Copy.facet.302)]
// CHECK:STDOUT:     %.loc17_21.1: %Copy.type = converted constants.%ptr.e8f8f9.1, %Copy.facet.loc17_21.1 [symbolic = %Copy.facet.loc17_21.3 (constants.%Copy.facet.302)]
// CHECK:STDOUT:     %Copy.facet.loc17_21.2: %Copy.type = facet_value constants.%ptr.e8f8f9.1, (constants.%Copy.lookup_impl_witness.1da) [symbolic = %Copy.facet.loc17_21.3 (constants.%Copy.facet.302)]
// CHECK:STDOUT:     %.loc17_21.2: %Copy.type = converted constants.%ptr.e8f8f9.1, %Copy.facet.loc17_21.2 [symbolic = %Copy.facet.loc17_21.3 (constants.%Copy.facet.302)]
// CHECK:STDOUT:     %specific_impl_fn.loc17_21.1: <specific function> = specific_impl_function %impl.elem0.loc17_21.1, @Copy.WithSelf.Op(constants.%Copy.facet.302) [symbolic = %specific_impl_fn.loc17_21.2 (constants.%specific_impl_fn.a76)]
// CHECK:STDOUT:     %bound_method.loc17_21.2: <bound method> = bound_method %u.ref, %specific_impl_fn.loc17_21.1
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @Y.as.A.impl.F.%ptr.loc16_29.1 (%ptr.e8f8f9.1) = call %bound_method.loc17_21.2(%u.ref)
// CHECK:STDOUT:     %tuple.elem2: ref @Y.as.A.impl.F.%ptr.loc16_29.1 (%ptr.e8f8f9.1) = tuple_access %return.param, element2
// CHECK:STDOUT:     %.loc17_22.4: init @Y.as.A.impl.F.%ptr.loc16_29.1 (%ptr.e8f8f9.1) to %tuple.elem2 = in_place_init %Copy.WithSelf.Op.call
// CHECK:STDOUT:     %.loc17_22.5: init @Y.as.A.impl.F.%tuple.type.loc16 (%tuple.type.316) to %return.param = tuple_init (%.loc17_22.2, %.loc17_22.3, %.loc17_22.4)
// CHECK:STDOUT:     %.loc17_23: init @Y.as.A.impl.F.%tuple.type.loc16 (%tuple.type.316) = converted %.loc17_22.1, %.loc17_22.5
// CHECK:STDOUT:     return %.loc17_23 to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %u.var: ref %Z = var_storage %u.var_patt
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value constants.%Z, (constants.%DefaultOrUnformed.impl_witness.88e) [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %.loc22_11.1: %DefaultOrUnformed.type = converted constants.%Z, %DefaultOrUnformed.facet [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %as_type.loc22: type = facet_access_type %.loc22_11.1 [concrete = constants.%Z]
// CHECK:STDOUT:   %.loc22_11.2: type = converted %.loc22_11.1, %as_type.loc22 [concrete = constants.%Z]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.specific_fn: <specific function> = specific_function constants.%T.as.DefaultOrUnformed.impl.Op.f06, @T.as.DefaultOrUnformed.impl.Op(constants.%Z) [concrete = constants.%T.as.DefaultOrUnformed.impl.Op.specific_fn]
// CHECK:STDOUT:   %.loc22_3.1: ref %Z = splice_block %u.var {}
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.call: init %Z to %.loc22_3.1 = call %T.as.DefaultOrUnformed.impl.Op.specific_fn()
// CHECK:STDOUT:   assign %u.var, %T.as.DefaultOrUnformed.impl.Op.call
// CHECK:STDOUT:   %Z.ref.loc22: type = name_ref Z, file.%Z.decl [concrete = constants.%Z]
// CHECK:STDOUT:   %u: ref %Z = wrapper_binding u, %u.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %u.patt: %pattern_type.34e = ref_binding_pattern u [concrete = constants.%u.patt.352]
// CHECK:STDOUT:     %u.var_patt: %pattern_type.34e = var_pattern %u.patt [concrete = constants.%u.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Y.ref: type = name_ref Y, file.%Y.decl [concrete = constants.%Y]
// CHECK:STDOUT:   %A.ref: %A.type.55a = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:   %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %A.type: type = facet_type <@A, @A(constants.%X)> [concrete = constants.%A.type.df0]
// CHECK:STDOUT:   %A.facet: %A.type.df0 = facet_value %Y.ref, (constants.%A.impl_witness) [concrete = constants.%A.facet.d1f]
// CHECK:STDOUT:   %.loc23_6: %A.type.df0 = converted %Y.ref, %A.facet [concrete = constants.%A.facet.d1f]
// CHECK:STDOUT:   %as_type.loc23: type = facet_access_type %.loc23_6 [concrete = constants.%Y]
// CHECK:STDOUT:   %.loc23_14.1: type = converted %.loc23_6, %as_type.loc23 [concrete = constants.%Y]
// CHECK:STDOUT:   %.loc23_14.2: %A.assoc_type.a54 = specific_constant @A.WithSelf.%assoc0.loc6_48.1, @A.WithSelf(constants.%X, constants.%A.facet.d1f) [concrete = constants.%assoc0.987]
// CHECK:STDOUT:   %F.ref: %A.assoc_type.a54 = name_ref F, %.loc23_14.2 [concrete = constants.%assoc0.987]
// CHECK:STDOUT:   %impl.elem0: %.261 = impl_witness_access constants.%A.impl_witness, element0 [concrete = constants.%Y.as.A.impl.F]
// CHECK:STDOUT:   %Z.ref.loc23: type = name_ref Z, file.%Z.decl [concrete = constants.%Z]
// CHECK:STDOUT:   %u.ref: ref %Z = name_ref u, %u
// CHECK:STDOUT:   %addr: %ptr.415 = addr_of %u.ref
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Y.as.A.impl.F(constants.%Z) [concrete = constants.%Y.as.A.impl.F.specific_fn]
// CHECK:STDOUT:   %.loc23_22.1: ref %tuple.type.2cd = temporary_storage
// CHECK:STDOUT:   %Y.as.A.impl.F.call: init %tuple.type.2cd to %.loc23_22.1 = call %specific_fn(%addr)
// CHECK:STDOUT:   %.loc23_22.2: ref %tuple.type.2cd = temporary %.loc23_22.1, %Y.as.A.impl.F.call
// CHECK:STDOUT:   %facet_value.loc23_22.1: %facet_type.ece = facet_value constants.%tuple.type.2cd, (constants.%DestroyImpl.impl_witness.cb9, constants.%custom_witness.b80ef6.5) [concrete = constants.%facet_value.219]
// CHECK:STDOUT:   %.loc23_22.3: %facet_type.ece = converted constants.%tuple.type.2cd, %facet_value.loc23_22.1 [concrete = constants.%facet_value.219]
// CHECK:STDOUT:   %facet_value.loc23_22.2: %facet_type.ece = facet_value constants.%tuple.type.2cd, (constants.%DestroyImpl.impl_witness.cb9, constants.%custom_witness.b80ef6.5) [concrete = constants.%facet_value.219]
// CHECK:STDOUT:   %.loc23_22.4: %facet_type.ece = converted constants.%tuple.type.2cd, %facet_value.loc23_22.2 [concrete = constants.%facet_value.219]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.1: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.219) [concrete = constants.%SelfDestruct.specific_fn.da0]
// CHECK:STDOUT:   %SelfDestruct.call.loc23: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc23_22.2)
// CHECK:STDOUT:   %facet_value.loc22_3.1: %facet_type.ece = facet_value constants.%Z, (constants.%DestroyImpl.impl_witness.72a, constants.%custom_witness.b80ef6.6) [concrete = constants.%facet_value.469]
// CHECK:STDOUT:   %.loc22_3.2: %facet_type.ece = converted constants.%Z, %facet_value.loc22_3.1 [concrete = constants.%facet_value.469]
// CHECK:STDOUT:   %facet_value.loc22_3.2: %facet_type.ece = facet_value constants.%Z, (constants.%DestroyImpl.impl_witness.72a, constants.%custom_witness.b80ef6.6) [concrete = constants.%facet_value.469]
// CHECK:STDOUT:   %.loc22_3.3: %facet_type.ece = converted constants.%Z, %facet_value.loc22_3.2 [concrete = constants.%facet_value.469]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.469) [concrete = constants.%SelfDestruct.specific_fn.c54]
// CHECK:STDOUT:   %SelfDestruct.call.loc22: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%u.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc23_22.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc23_22.2(%self.param: ref %X) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc23_22.3(%self.param: ref %Y) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc23_22.4(%self.param: ref %ptr.415) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc23_22.5(%self.param: ref %tuple.type.2cd) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %tuple.elem2: ref %ptr.415 = tuple_access %self.param, element2
// CHECK:STDOUT:   %facet_value.loc23_22.1: %facet_type.ece = facet_value constants.%ptr.415, (constants.%DestroyImpl.impl_witness.3a0, constants.%custom_witness.b80ef6.4) [concrete = constants.%facet_value.da6]
// CHECK:STDOUT:   %.loc23_22.1: %facet_type.ece = converted constants.%ptr.415, %facet_value.loc23_22.1 [concrete = constants.%facet_value.da6]
// CHECK:STDOUT:   %facet_value.loc23_22.2: %facet_type.ece = facet_value constants.%ptr.415, (constants.%DestroyImpl.impl_witness.3a0, constants.%custom_witness.b80ef6.4) [concrete = constants.%facet_value.da6]
// CHECK:STDOUT:   %.loc23_22.2: %facet_type.ece = converted constants.%ptr.415, %facet_value.loc23_22.2 [concrete = constants.%facet_value.da6]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.1: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.da6) [concrete = constants.%SelfDestruct.specific_fn.b83]
// CHECK:STDOUT:   %SelfDestruct.call.loc23_22.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%tuple.elem2)
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %tuple.elem1: ref %Y = tuple_access %self.param, element1
// CHECK:STDOUT:   %facet_value.loc23_22.3: %facet_type.ece = facet_value constants.%Y, (constants.%DestroyImpl.impl_witness.691, constants.%custom_witness.b80ef6.3) [concrete = constants.%facet_value.391]
// CHECK:STDOUT:   %.loc23_22.3: %facet_type.ece = converted constants.%Y, %facet_value.loc23_22.3 [concrete = constants.%facet_value.391]
// CHECK:STDOUT:   %facet_value.loc23_22.4: %facet_type.ece = facet_value constants.%Y, (constants.%DestroyImpl.impl_witness.691, constants.%custom_witness.b80ef6.3) [concrete = constants.%facet_value.391]
// CHECK:STDOUT:   %.loc23_22.4: %facet_type.ece = converted constants.%Y, %facet_value.loc23_22.4 [concrete = constants.%facet_value.391]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.391) [concrete = constants.%SelfDestruct.specific_fn.e36]
// CHECK:STDOUT:   %SelfDestruct.call.loc23_22.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%tuple.elem1)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %tuple.elem0: ref %X = tuple_access %self.param, element0
// CHECK:STDOUT:   %facet_value.loc23_22.5: %facet_type.ece = facet_value constants.%X, (constants.%DestroyImpl.impl_witness.678, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.9d2]
// CHECK:STDOUT:   %.loc23_22.5: %facet_type.ece = converted constants.%X, %facet_value.loc23_22.5 [concrete = constants.%facet_value.9d2]
// CHECK:STDOUT:   %facet_value.loc23_22.6: %facet_type.ece = facet_value constants.%X, (constants.%DestroyImpl.impl_witness.678, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.9d2]
// CHECK:STDOUT:   %.loc23_22.6: %facet_type.ece = converted constants.%X, %facet_value.loc23_22.6 [concrete = constants.%facet_value.9d2]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.3: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.9d2) [concrete = constants.%SelfDestruct.specific_fn.e0d]
// CHECK:STDOUT:   %SelfDestruct.call.loc23_22.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%tuple.elem0)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc22(%self.param: ref %Z) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallGeneric(%T.loc26_25.2: %facet_type.572) {
// CHECK:STDOUT:   %T.patt.loc26_25.2: %pattern_type.03d = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc26_25.2 (constants.%T.patt.4c0)]
// CHECK:STDOUT:   %T.loc26_25.1: %facet_type.572 = symbolic_binding T, 0 [symbolic = %T.loc26_25.1 (constants.%T.80e)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc28_4.2: type = facet_access_type %T.loc26_25.1 [symbolic = %T.as_type.loc28_4.2 (constants.%T.as_type.8e3)]
// CHECK:STDOUT:   %A.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc26_25.1, @A, @A(constants.%X) [symbolic = %A.lookup_impl_witness (constants.%A.lookup_impl_witness)]
// CHECK:STDOUT:   %A.facet: %A.type.df0 = facet_value %T.as_type.loc28_4.2, (%A.lookup_impl_witness) [symbolic = %A.facet (constants.%A.facet.5d2)]
// CHECK:STDOUT:   %A.WithSelf.F.type: type = fn_type @A.WithSelf.F, @A.WithSelf(constants.%X, %A.facet) [symbolic = %A.WithSelf.F.type (constants.%A.WithSelf.F.type.9bb)]
// CHECK:STDOUT:   %.loc28_4.3: type = fn_type_with_self_type %A.WithSelf.F.type, %A.facet [symbolic = %.loc28_4.3 (constants.%.ad0)]
// CHECK:STDOUT:   %impl.elem0.loc28_4.2: @CallGeneric.%.loc28_4.3 (%.ad0) = impl_witness_access %A.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc28_4.2 (constants.%impl.elem0.327)]
// CHECK:STDOUT:   %specific_impl_fn.loc28_4.2: <specific function> = specific_impl_function %impl.elem0.loc28_4.2, @A.WithSelf.F(constants.%X, %A.facet, constants.%Z) [symbolic = %specific_impl_fn.loc28_4.2 (constants.%specific_impl_fn.b41)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (constants.%X, %T.as_type.loc28_4.2, constants.%ptr.415) [symbolic = %tuple.type (constants.%tuple.type.e1c)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %tuple.type [symbolic = %require_complete (constants.%require_complete.067)]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness: <witness> = lookup_impl_witness %tuple.type, @SubobjectDestroy [symbolic = %SubobjectDestroy.lookup_impl_witness (constants.%SubobjectDestroy.lookup_impl_witness.a7e)]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %tuple.type, (%SubobjectDestroy.lookup_impl_witness) [symbolic = %SubobjectDestroy.facet (constants.%SubobjectDestroy.facet.f49)]
// CHECK:STDOUT:   %.loc28_12.5: require_specific_def_type = require_specific_def @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet) [symbolic = %.loc28_12.5 (constants.%.d22)]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness: <witness> = lookup_impl_witness %tuple.type, @DestroyImpl [symbolic = %DestroyImpl.lookup_impl_witness (constants.%DestroyImpl.lookup_impl_witness.6a7)]
// CHECK:STDOUT:   %facet_value.loc28_12.3: %facet_type.ece = facet_value %tuple.type, (%DestroyImpl.lookup_impl_witness, %SubobjectDestroy.lookup_impl_witness) [symbolic = %facet_value.loc28_12.3 (constants.%facet_value.744)]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.3: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(%facet_value.loc28_12.3) [symbolic = %SelfDestruct.specific_fn.3 (constants.%SelfDestruct.specific_fn.3c0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %u.var: ref %Z = var_storage %u.var_patt
// CHECK:STDOUT:     %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value constants.%Z, (constants.%DefaultOrUnformed.impl_witness.88e) [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:     %.loc27_11.1: %DefaultOrUnformed.type = converted constants.%Z, %DefaultOrUnformed.facet [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:     %as_type: type = facet_access_type %.loc27_11.1 [concrete = constants.%Z]
// CHECK:STDOUT:     %.loc27_11.2: type = converted %.loc27_11.1, %as_type [concrete = constants.%Z]
// CHECK:STDOUT:     %T.as.DefaultOrUnformed.impl.Op.specific_fn: <specific function> = specific_function constants.%T.as.DefaultOrUnformed.impl.Op.f06, @T.as.DefaultOrUnformed.impl.Op(constants.%Z) [concrete = constants.%T.as.DefaultOrUnformed.impl.Op.specific_fn]
// CHECK:STDOUT:     %.loc27_3.1: ref %Z = splice_block %u.var {}
// CHECK:STDOUT:     %T.as.DefaultOrUnformed.impl.Op.call: init %Z to %.loc27_3.1 = call %T.as.DefaultOrUnformed.impl.Op.specific_fn()
// CHECK:STDOUT:     assign %u.var, %T.as.DefaultOrUnformed.impl.Op.call
// CHECK:STDOUT:     %Z.ref.loc27: type = name_ref Z, file.%Z.decl [concrete = constants.%Z]
// CHECK:STDOUT:     %u: ref %Z = wrapper_binding u, %u.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %u.patt: %pattern_type.34e = ref_binding_pattern u [concrete = constants.%u.patt.352]
// CHECK:STDOUT:       %u.var_patt: %pattern_type.34e = var_pattern %u.patt [concrete = constants.%u.var_patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.ref: %facet_type.572 = name_ref T, %T.loc26_25.2 [symbolic = %T.loc26_25.1 (constants.%T.80e)]
// CHECK:STDOUT:     %T.as_type.loc28_4.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc28_4.2 (constants.%T.as_type.8e3)]
// CHECK:STDOUT:     %.loc28_4.1: type = converted %T.ref, %T.as_type.loc28_4.1 [symbolic = %T.as_type.loc28_4.2 (constants.%T.as_type.8e3)]
// CHECK:STDOUT:     %.loc28_4.2: %A.assoc_type.a54 = specific_constant @A.WithSelf.%assoc0.loc6_48.1, @A.WithSelf(constants.%X, constants.%T.80e) [concrete = constants.%assoc0.987]
// CHECK:STDOUT:     %F.ref: %A.assoc_type.a54 = name_ref F, %.loc28_4.2 [concrete = constants.%assoc0.987]
// CHECK:STDOUT:     %impl.elem0.loc28_4.1: @CallGeneric.%.loc28_4.3 (%.ad0) = impl_witness_access constants.%A.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc28_4.2 (constants.%impl.elem0.327)]
// CHECK:STDOUT:     %Z.ref.loc28: type = name_ref Z, file.%Z.decl [concrete = constants.%Z]
// CHECK:STDOUT:     %u.ref: ref %Z = name_ref u, %u
// CHECK:STDOUT:     %addr: %ptr.415 = addr_of %u.ref
// CHECK:STDOUT:     %specific_impl_fn.loc28_4.1: <specific function> = specific_impl_function %impl.elem0.loc28_4.1, @A.WithSelf.F(constants.%X, constants.%A.facet.5d2, constants.%Z) [symbolic = %specific_impl_fn.loc28_4.2 (constants.%specific_impl_fn.b41)]
// CHECK:STDOUT:     %.loc28_12.1: ref @CallGeneric.%tuple.type (%tuple.type.e1c) = temporary_storage
// CHECK:STDOUT:     %A.WithSelf.F.call: init @CallGeneric.%tuple.type (%tuple.type.e1c) to %.loc28_12.1 = call %specific_impl_fn.loc28_4.1(%addr)
// CHECK:STDOUT:     %.loc28_12.2: ref @CallGeneric.%tuple.type (%tuple.type.e1c) = temporary %.loc28_12.1, %A.WithSelf.F.call
// CHECK:STDOUT:     %facet_value.loc28_12.1: %facet_type.ece = facet_value constants.%tuple.type.e1c, (constants.%DestroyImpl.lookup_impl_witness.6a7, constants.%SubobjectDestroy.lookup_impl_witness.a7e) [symbolic = %facet_value.loc28_12.3 (constants.%facet_value.744)]
// CHECK:STDOUT:     %.loc28_12.3: %facet_type.ece = converted constants.%tuple.type.e1c, %facet_value.loc28_12.1 [symbolic = %facet_value.loc28_12.3 (constants.%facet_value.744)]
// CHECK:STDOUT:     %facet_value.loc28_12.2: %facet_type.ece = facet_value constants.%tuple.type.e1c, (constants.%DestroyImpl.lookup_impl_witness.6a7, constants.%SubobjectDestroy.lookup_impl_witness.a7e) [symbolic = %facet_value.loc28_12.3 (constants.%facet_value.744)]
// CHECK:STDOUT:     %.loc28_12.4: %facet_type.ece = converted constants.%tuple.type.e1c, %facet_value.loc28_12.2 [symbolic = %facet_value.loc28_12.3 (constants.%facet_value.744)]
// CHECK:STDOUT:     %SelfDestruct.specific_fn.1: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.744) [symbolic = %SelfDestruct.specific_fn.3 (constants.%SelfDestruct.specific_fn.3c0)]
// CHECK:STDOUT:     %SelfDestruct.call.loc28: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc28_12.2)
// CHECK:STDOUT:     %facet_value.loc27_3.1: %facet_type.ece = facet_value constants.%Z, (constants.%DestroyImpl.impl_witness.72a, constants.%custom_witness.b80ef6.6) [concrete = constants.%facet_value.469]
// CHECK:STDOUT:     %.loc27_3.2: %facet_type.ece = converted constants.%Z, %facet_value.loc27_3.1 [concrete = constants.%facet_value.469]
// CHECK:STDOUT:     %facet_value.loc27_3.2: %facet_type.ece = facet_value constants.%Z, (constants.%DestroyImpl.impl_witness.72a, constants.%custom_witness.b80ef6.6) [concrete = constants.%facet_value.469]
// CHECK:STDOUT:     %.loc27_3.3: %facet_type.ece = converted constants.%Z, %facet_value.loc27_3.2 [concrete = constants.%facet_value.469]
// CHECK:STDOUT:     %SelfDestruct.specific_fn.2: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.469) [concrete = constants.%SelfDestruct.specific_fn.c54]
// CHECK:STDOUT:     %SelfDestruct.call.loc27: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%u.var)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallIndirect() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CallGeneric.ref: %CallGeneric.type = name_ref CallGeneric, file.%CallGeneric.decl [concrete = constants.%CallGeneric]
// CHECK:STDOUT:   %Y.ref: type = name_ref Y, file.%Y.decl [concrete = constants.%Y]
// CHECK:STDOUT:   %facet_value: %facet_type.572 = facet_value %Y.ref, (constants.%A.impl_witness, constants.%DestroyImpl.impl_witness.691, constants.%custom_witness.b80ef6.3) [concrete = constants.%facet_value.a83]
// CHECK:STDOUT:   %.loc32: %facet_type.572 = converted %Y.ref, %facet_value [concrete = constants.%facet_value.a83]
// CHECK:STDOUT:   %CallGeneric.specific_fn: <specific function> = specific_function %CallGeneric.ref, @CallGeneric(constants.%facet_value.a83) [concrete = constants.%CallGeneric.specific_fn]
// CHECK:STDOUT:   %CallGeneric.call: init %empty_tuple.type = call %CallGeneric.specific_fn()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%T.67d) {
// CHECK:STDOUT:   %T.patt.loc5_14.2 => constants.%T.patt.3a0
// CHECK:STDOUT:   %T.loc5_14.1 => constants.%T.67d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%T.67d, constants.%Self.49a) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf.F(constants.%T.67d, constants.%Self.49a, constants.%U.4e7) {
// CHECK:STDOUT:   %U.patt.loc6_17.2 => constants.%U.patt.5c4
// CHECK:STDOUT:   %U.loc6_17.1 => constants.%U.4e7
// CHECK:STDOUT:   %ptr.loc6_29.1 => constants.%ptr.8f6
// CHECK:STDOUT:   %pattern_type.loc6_26 => constants.%pattern_type.986
// CHECK:STDOUT:   %u.param_patt.loc6_26.2 => constants.%u.param_patt.a01
// CHECK:STDOUT:   %u.patt.loc6_26.2 => constants.%u.patt.d42
// CHECK:STDOUT:   %T => constants.%T.67d
// CHECK:STDOUT:   %A.type => constants.%A.type.acd
// CHECK:STDOUT:   %Self => constants.%Self.49a
// CHECK:STDOUT:   %tuple.type.loc6_47.1 => constants.%tuple.type.a31
// CHECK:STDOUT:   %tuple => constants.%tuple.a84
// CHECK:STDOUT:   %Self.as_type.loc6_47.1 => constants.%Self.as_type.763
// CHECK:STDOUT:   %tuple.type.loc6_47.2 => constants.%tuple.type.640
// CHECK:STDOUT:   %.loc6_47.1 => constants.%.f0d
// CHECK:STDOUT:   %pattern_type.loc6_47 => constants.%pattern_type.0d7
// CHECK:STDOUT:   %return.param_patt.loc6_47.2 => constants.%return.param_patt.9d3
// CHECK:STDOUT:   %return.patt.loc6_32.2 => constants.%return.patt.8aa
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%X) {
// CHECK:STDOUT:   %T.patt.loc5_14.2 => constants.%T.patt.3a0
// CHECK:STDOUT:   %T.loc5_14.1 => constants.%X
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.type => constants.%A.type.df0
// CHECK:STDOUT:   %Self.loc5_22.2 => constants.%Self.891
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%X, constants.%Self.49a) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.df0
// CHECK:STDOUT:   %Self => constants.%Self.49a
// CHECK:STDOUT:   %A.WithSelf.F.type => constants.%A.WithSelf.F.type.ad4
// CHECK:STDOUT:   %A.WithSelf.F => constants.%A.WithSelf.F.1e2
// CHECK:STDOUT:   %A.assoc_type => constants.%A.assoc_type.a54
// CHECK:STDOUT:   %assoc0.loc6_48.2 => constants.%assoc0.987
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y.as.A.impl.F(constants.%U.67d) {
// CHECK:STDOUT:   %U.patt.loc16_17.2 => constants.%U.patt.3a0
// CHECK:STDOUT:   %U.loc16_17.1 => constants.%U.67d
// CHECK:STDOUT:   %ptr.loc16_29.1 => constants.%ptr.e8f8f9.1
// CHECK:STDOUT:   %pattern_type.loc16_26 => constants.%pattern_type.4f4b84.1
// CHECK:STDOUT:   %u.param_patt.loc16_26.2 => constants.%u.param_patt.58b
// CHECK:STDOUT:   %u.patt.loc16_26.2 => constants.%u.patt.60f
// CHECK:STDOUT:   %tuple => constants.%tuple.250
// CHECK:STDOUT:   %tuple.type.loc16 => constants.%tuple.type.316
// CHECK:STDOUT:   %.loc16_44.1 => constants.%.3c1
// CHECK:STDOUT:   %pattern_type.loc16_44 => constants.%pattern_type.178
// CHECK:STDOUT:   %return.param_patt.loc16_44.2 => constants.%return.param_patt.b96
// CHECK:STDOUT:   %return.patt.loc16_32.2 => constants.%return.patt.f2f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%X, constants.%A.facet.d1f) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.df0
// CHECK:STDOUT:   %Self => constants.%A.facet.d1f
// CHECK:STDOUT:   %A.WithSelf.F.type => constants.%A.WithSelf.F.type.781
// CHECK:STDOUT:   %A.WithSelf.F => constants.%A.WithSelf.F.b66
// CHECK:STDOUT:   %A.assoc_type => constants.%A.assoc_type.a54
// CHECK:STDOUT:   %assoc0.loc6_48.2 => constants.%assoc0.987
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf.F(constants.%X, constants.%A.facet.d1f, constants.%U.67d) {
// CHECK:STDOUT:   %U.patt.loc6_17.2 => constants.%U.patt.5c4
// CHECK:STDOUT:   %U.loc6_17.1 => constants.%U.67d
// CHECK:STDOUT:   %ptr.loc6_29.1 => constants.%ptr.e8f8f9.1
// CHECK:STDOUT:   %pattern_type.loc6_26 => constants.%pattern_type.4f4b84.1
// CHECK:STDOUT:   %u.param_patt.loc6_26.2 => constants.%u.param_patt.58b
// CHECK:STDOUT:   %u.patt.loc6_26.2 => constants.%u.patt.60f
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.df0
// CHECK:STDOUT:   %Self => constants.%A.facet.d1f
// CHECK:STDOUT:   %tuple.type.loc6_47.1 => constants.%tuple.type.202
// CHECK:STDOUT:   %tuple => constants.%tuple.e37
// CHECK:STDOUT:   %Self.as_type.loc6_47.1 => constants.%Y
// CHECK:STDOUT:   %tuple.type.loc6_47.2 => constants.%tuple.type.316
// CHECK:STDOUT:   %.loc6_47.1 => constants.%.3c1
// CHECK:STDOUT:   %pattern_type.loc6_47 => constants.%pattern_type.178
// CHECK:STDOUT:   %return.param_patt.loc6_47.2 => constants.%return.param_patt.b96
// CHECK:STDOUT:   %return.patt.loc6_32.2 => constants.%return.patt.f2f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y.as.A.impl.F(constants.%Z) {
// CHECK:STDOUT:   %U.patt.loc16_17.2 => constants.%U.patt.3a0
// CHECK:STDOUT:   %U.loc16_17.1 => constants.%Z
// CHECK:STDOUT:   %ptr.loc16_29.1 => constants.%ptr.415
// CHECK:STDOUT:   %pattern_type.loc16_26 => constants.%pattern_type.b4b
// CHECK:STDOUT:   %u.param_patt.loc16_26.2 => constants.%u.param_patt.5c1
// CHECK:STDOUT:   %u.patt.loc16_26.2 => constants.%u.patt.d7f
// CHECK:STDOUT:   %tuple => constants.%tuple.b32
// CHECK:STDOUT:   %tuple.type.loc16 => constants.%tuple.type.2cd
// CHECK:STDOUT:   %.loc16_44.1 => constants.%.5bc
// CHECK:STDOUT:   %pattern_type.loc16_44 => constants.%pattern_type.b33
// CHECK:STDOUT:   %return.param_patt.loc16_44.2 => constants.%return.param_patt.75d
// CHECK:STDOUT:   %return.patt.loc16_32.2 => constants.%return.patt.8bb
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc16_26 => constants.%complete_type.458
// CHECK:STDOUT:   %require_complete.loc16_32 => constants.%complete_type.dec
// CHECK:STDOUT:   %tuple.type.loc17 => constants.%tuple.type.2c5
// CHECK:STDOUT:   %.loc17_21.3 => constants.%.072
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.impl_witness.cff
// CHECK:STDOUT:   %Copy.facet.loc17_21.3 => constants.%Copy.facet.9c6
// CHECK:STDOUT:   %Copy.WithSelf.Op.type => constants.%Copy.WithSelf.Op.type.107
// CHECK:STDOUT:   %.loc17_21.4 => constants.%.08f
// CHECK:STDOUT:   %impl.elem0.loc17_21.2 => constants.%ptr.as.Copy.impl.Op.ebe
// CHECK:STDOUT:   %specific_impl_fn.loc17_21.2 => constants.%ptr.as.Copy.impl.Op.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGeneric(constants.%T.80e) {
// CHECK:STDOUT:   %T.patt.loc26_25.2 => constants.%T.patt.4c0
// CHECK:STDOUT:   %T.loc26_25.1 => constants.%T.80e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%X, constants.%T.80e) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.df0
// CHECK:STDOUT:   %Self => constants.%T.80e
// CHECK:STDOUT:   %A.WithSelf.F.type => constants.%A.WithSelf.F.type.298
// CHECK:STDOUT:   %A.WithSelf.F => constants.%A.WithSelf.F.d25
// CHECK:STDOUT:   %A.assoc_type => constants.%A.assoc_type.a54
// CHECK:STDOUT:   %assoc0.loc6_48.2 => constants.%assoc0.987
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%X, constants.%A.facet.5d2) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.df0
// CHECK:STDOUT:   %Self => constants.%A.facet.5d2
// CHECK:STDOUT:   %A.WithSelf.F.type => constants.%A.WithSelf.F.type.9bb
// CHECK:STDOUT:   %A.WithSelf.F => constants.%A.WithSelf.F.8c8
// CHECK:STDOUT:   %A.assoc_type => constants.%A.assoc_type.a54
// CHECK:STDOUT:   %assoc0.loc6_48.2 => constants.%assoc0.987
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf.F(constants.%X, constants.%A.facet.5d2, constants.%Z) {
// CHECK:STDOUT:   %U.patt.loc6_17.2 => constants.%U.patt.5c4
// CHECK:STDOUT:   %U.loc6_17.1 => constants.%Z
// CHECK:STDOUT:   %ptr.loc6_29.1 => constants.%ptr.415
// CHECK:STDOUT:   %pattern_type.loc6_26 => constants.%pattern_type.b4b
// CHECK:STDOUT:   %u.param_patt.loc6_26.2 => constants.%u.param_patt.5c1
// CHECK:STDOUT:   %u.patt.loc6_26.2 => constants.%u.patt.d7f
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.df0
// CHECK:STDOUT:   %Self => constants.%A.facet.5d2
// CHECK:STDOUT:   %tuple.type.loc6_47.1 => constants.%tuple.type.202
// CHECK:STDOUT:   %tuple => constants.%tuple.664
// CHECK:STDOUT:   %Self.as_type.loc6_47.1 => constants.%T.as_type.8e3
// CHECK:STDOUT:   %tuple.type.loc6_47.2 => constants.%tuple.type.e1c
// CHECK:STDOUT:   %.loc6_47.1 => constants.%.1c24
// CHECK:STDOUT:   %pattern_type.loc6_47 => constants.%pattern_type.75d
// CHECK:STDOUT:   %return.param_patt.loc6_47.2 => constants.%return.param_patt.32a
// CHECK:STDOUT:   %return.patt.loc6_32.2 => constants.%return.patt.13c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGeneric(constants.%facet_value.a83) {
// CHECK:STDOUT:   %T.patt.loc26_25.2 => constants.%T.patt.4c0
// CHECK:STDOUT:   %T.loc26_25.1 => constants.%facet_value.a83
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc28_4.2 => constants.%Y
// CHECK:STDOUT:   %A.lookup_impl_witness => constants.%A.impl_witness
// CHECK:STDOUT:   %A.facet => constants.%A.facet.d1f
// CHECK:STDOUT:   %A.WithSelf.F.type => constants.%A.WithSelf.F.type.781
// CHECK:STDOUT:   %.loc28_4.3 => constants.%.261
// CHECK:STDOUT:   %impl.elem0.loc28_4.2 => constants.%Y.as.A.impl.F
// CHECK:STDOUT:   %specific_impl_fn.loc28_4.2 => constants.%Y.as.A.impl.F.specific_fn
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.2cd
// CHECK:STDOUT:   %require_complete => constants.%complete_type.dec
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness => constants.%custom_witness.b80ef6.5
// CHECK:STDOUT:   %SubobjectDestroy.facet => constants.%SubobjectDestroy.facet.c56
// CHECK:STDOUT:   %.loc28_12.5 => constants.%.081
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness => constants.%DestroyImpl.impl_witness.cb9
// CHECK:STDOUT:   %facet_value.loc28_12.3 => constants.%facet_value.219
// CHECK:STDOUT:   %SelfDestruct.specific_fn.3 => constants.%SelfDestruct.specific_fn.da0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf.F(constants.%X, constants.%A.facet.d1f, constants.%Z) {
// CHECK:STDOUT:   %U.patt.loc6_17.2 => constants.%U.patt.5c4
// CHECK:STDOUT:   %U.loc6_17.1 => constants.%Z
// CHECK:STDOUT:   %ptr.loc6_29.1 => constants.%ptr.415
// CHECK:STDOUT:   %pattern_type.loc6_26 => constants.%pattern_type.b4b
// CHECK:STDOUT:   %u.param_patt.loc6_26.2 => constants.%u.param_patt.5c1
// CHECK:STDOUT:   %u.patt.loc6_26.2 => constants.%u.patt.d7f
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.df0
// CHECK:STDOUT:   %Self => constants.%A.facet.d1f
// CHECK:STDOUT:   %tuple.type.loc6_47.1 => constants.%tuple.type.202
// CHECK:STDOUT:   %tuple => constants.%tuple.d83
// CHECK:STDOUT:   %Self.as_type.loc6_47.1 => constants.%Y
// CHECK:STDOUT:   %tuple.type.loc6_47.2 => constants.%tuple.type.2cd
// CHECK:STDOUT:   %.loc6_47.1 => constants.%.5bc
// CHECK:STDOUT:   %pattern_type.loc6_47 => constants.%pattern_type.b33
// CHECK:STDOUT:   %return.param_patt.loc6_47.2 => constants.%return.param_patt.75d
// CHECK:STDOUT:   %return.patt.loc6_32.2 => constants.%return.patt.8bb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- generic_method_more_params.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt.3a0: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %A.type.55a: type = generic_interface_type @A [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %A.generic: %A.type.55a = struct_value () [concrete]
// CHECK:STDOUT:   %A.type.acd: type = facet_type <@A, @A(%T.67d)> [symbolic]
// CHECK:STDOUT:   %Self.49a: %A.type.acd = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %U.patt.5c4: %pattern_type.9a5 = symbolic_binding_pattern U, 2 [symbolic]
// CHECK:STDOUT:   %U.4e7: type = symbolic_binding U, 2 [symbolic]
// CHECK:STDOUT:   %pattern_type.62e: type = pattern_type %U.4e7 [symbolic]
// CHECK:STDOUT:   %u.param_patt.6ef: %pattern_type.62e = value_param_pattern [symbolic]
// CHECK:STDOUT:   %u.patt.9b1: %pattern_type.62e = wrapper_binding_pattern u, %u.param_patt.6ef [symbolic]
// CHECK:STDOUT:   %tuple.type.a31: type = tuple_type (type, %A.type.acd, type) [symbolic]
// CHECK:STDOUT:   %tuple.62b: %tuple.type.a31 = tuple_value (%T.67d, %Self.49a, %U.4e7) [symbolic]
// CHECK:STDOUT:   %Self.as_type.763: type = facet_access_type %Self.49a [symbolic]
// CHECK:STDOUT:   %tuple.type.d73: type = tuple_type (%T.67d, %Self.as_type.763, %U.4e7) [symbolic]
// CHECK:STDOUT:   %.bc3: Core.Form = init_form %tuple.type.d73 [symbolic]
// CHECK:STDOUT:   %pattern_type.d6c: type = pattern_type %tuple.type.d73 [symbolic]
// CHECK:STDOUT:   %return.param_patt.52a: %pattern_type.d6c = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.808: %pattern_type.d6c = return_slot_pattern %return.param_patt.52a, %tuple.type.d73 [symbolic]
// CHECK:STDOUT:   %A.WithSelf.F.type.08d: type = fn_type @A.WithSelf.F, @A.WithSelf(%T.67d, %Self.49a) [symbolic]
// CHECK:STDOUT:   %A.WithSelf.F.3fa: %A.WithSelf.F.type.08d = struct_value () [symbolic]
// CHECK:STDOUT:   %A.assoc_type.447: type = assoc_entity_type @A, @A(%T.67d) [symbolic]
// CHECK:STDOUT:   %assoc0.d4b: %A.assoc_type.447 = assoc_entity element0, @A.WithSelf.%A.WithSelf.F.decl [symbolic]
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Y1: type = class_type @Y1 [concrete]
// CHECK:STDOUT:   %Y2: type = class_type @Y2 [concrete]
// CHECK:STDOUT:   %Z: type = class_type @Z [concrete]
// CHECK:STDOUT:   %V1.patt: %pattern_type.9a5 = symbolic_binding_pattern V1, 0 [symbolic]
// CHECK:STDOUT:   %V1: type = symbolic_binding V1, 0 [symbolic]
// CHECK:STDOUT:   %V2.patt: %pattern_type.9a5 = symbolic_binding_pattern V2, 1 [symbolic]
// CHECK:STDOUT:   %V2: type = symbolic_binding V2, 1 [symbolic]
// CHECK:STDOUT:   %W.patt: %pattern_type.9a5 = symbolic_binding_pattern W, 2 [symbolic]
// CHECK:STDOUT:   %W: type = symbolic_binding W, 2 [symbolic]
// CHECK:STDOUT:   %tuple.type.693: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.fa2: %tuple.type.693 = tuple_value (%V1, %V2) [symbolic]
// CHECK:STDOUT:   %tuple.type.a5e: type = tuple_type (%V1, %V2) [symbolic]
// CHECK:STDOUT:   %A.type.3ad: type = facet_type <@A, @A(%W)> [symbolic]
// CHECK:STDOUT:   %.b95: <witness> = impl_self_witness %tuple.type.a5e, @A, @A(%W) [symbolic]
// CHECK:STDOUT:   %A.impl_witness.07b: <witness> = impl_witness @tuple.type.as.A.impl.%A.impl_witness_table, @tuple.type.as.A.impl(%V1, %V2, %W) [symbolic]
// CHECK:STDOUT:   %Self.0f8: %A.type.3ad = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %A.WithSelf.F.type.64e: type = fn_type @A.WithSelf.F, @A.WithSelf(%W, %Self.49a) [symbolic]
// CHECK:STDOUT:   %A.WithSelf.F.1fb: %A.WithSelf.F.type.64e = struct_value () [symbolic]
// CHECK:STDOUT:   %A.assoc_type.250: type = assoc_entity_type @A, @A(%W) [symbolic]
// CHECK:STDOUT:   %assoc0.6de: %A.assoc_type.250 = assoc_entity element0, @A.WithSelf.%A.WithSelf.F.decl [symbolic]
// CHECK:STDOUT:   %require_complete.7b2: <witness> = require_complete_type %A.type.3ad [symbolic]
// CHECK:STDOUT:   %U.patt.755: %pattern_type.9a5 = symbolic_binding_pattern U, 3 [symbolic]
// CHECK:STDOUT:   %U.ce4: type = symbolic_binding U, 3 [symbolic]
// CHECK:STDOUT:   %pattern_type.e4a: type = pattern_type %U.ce4 [symbolic]
// CHECK:STDOUT:   %u.param_patt.afe: %pattern_type.e4a = value_param_pattern [symbolic]
// CHECK:STDOUT:   %u.patt.60c: %pattern_type.e4a = wrapper_binding_pattern u, %u.param_patt.afe [symbolic]
// CHECK:STDOUT:   %tuple.type.649: type = tuple_type (type, %tuple.type.693, type) [concrete]
// CHECK:STDOUT:   %tuple.cd3: %tuple.type.649 = tuple_value (%W, %tuple.fa2, %U.ce4) [symbolic]
// CHECK:STDOUT:   %tuple.type.897: type = tuple_type (%W, %tuple.type.a5e, %U.ce4) [symbolic]
// CHECK:STDOUT:   %.f8e: Core.Form = init_form %tuple.type.897 [symbolic]
// CHECK:STDOUT:   %pattern_type.5cc: type = pattern_type %tuple.type.897 [symbolic]
// CHECK:STDOUT:   %return.param_patt.b1c: %pattern_type.5cc = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.bad9: %pattern_type.5cc = return_slot_pattern %return.param_patt.b1c, %tuple.type.897 [symbolic]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type.44a: type = fn_type @tuple.type.as.A.impl.F, @tuple.type.as.A.impl(%V1, %V2, %W) [symbolic]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.742: %tuple.type.as.A.impl.F.type.44a = struct_value () [symbolic]
// CHECK:STDOUT:   %A.facet.de4: %A.type.3ad = facet_value %tuple.type.a5e, (%A.impl_witness.07b) [symbolic]
// CHECK:STDOUT:   %A.WithSelf.F.type.3e0: type = fn_type @A.WithSelf.F, @A.WithSelf(%W, %A.facet.de4) [symbolic]
// CHECK:STDOUT:   %A.WithSelf.F.43a: %A.WithSelf.F.type.3e0 = struct_value () [symbolic]
// CHECK:STDOUT:   %tuple.type.038: type = tuple_type (type, %A.type.3ad, type) [symbolic]
// CHECK:STDOUT:   %tuple.c3f: %tuple.type.038 = tuple_value (%W, %A.facet.de4, %U.ce4) [symbolic]
// CHECK:STDOUT:   %require_complete.5a4: <witness> = require_complete_type %U.ce4 [symbolic]
// CHECK:STDOUT:   %require_complete.da0: <witness> = require_complete_type %tuple.type.897 [symbolic]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.specific_fn.37b: <specific function> = specific_function %tuple.type.as.A.impl.F.742, @tuple.type.as.A.impl.F(%V1, %V2, %W, %U.ce4) [symbolic]
// CHECK:STDOUT:   %Call.type: type = fn_type @Call [concrete]
// CHECK:STDOUT:   %Call: %Call.type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.71f: %tuple.type.693 = tuple_value (%Y1, %Y2) [concrete]
// CHECK:STDOUT:   %tuple.type.59f: type = tuple_type (%Y1, %Y2) [concrete]
// CHECK:STDOUT:   %A.type.df0: type = facet_type <@A, @A(%X)> [concrete]
// CHECK:STDOUT:   %.de6: <witness> = impl_self_witness %tuple.type.59f, @A, @A(%X) [concrete]
// CHECK:STDOUT:   %A.impl_witness.0e0: <witness> = impl_witness @tuple.type.as.A.impl.%A.impl_witness_table, @tuple.type.as.A.impl(%Y1, %Y2, %X) [concrete]
// CHECK:STDOUT:   %Self.891: %A.type.df0 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %A.WithSelf.F.type.ad4: type = fn_type @A.WithSelf.F, @A.WithSelf(%X, %Self.49a) [symbolic]
// CHECK:STDOUT:   %A.WithSelf.F.1e2: %A.WithSelf.F.type.ad4 = struct_value () [symbolic]
// CHECK:STDOUT:   %A.assoc_type.a54: type = assoc_entity_type @A, @A(%X) [concrete]
// CHECK:STDOUT:   %assoc0.987: %A.assoc_type.a54 = assoc_entity element0, @A.WithSelf.%A.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %complete_type.e33: <witness> = complete_type_witness %A.type.df0 [concrete]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type.745: type = fn_type @tuple.type.as.A.impl.F, @tuple.type.as.A.impl(%Y1, %Y2, %X) [concrete]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.749: %tuple.type.as.A.impl.F.type.745 = struct_value () [concrete]
// CHECK:STDOUT:   %A.facet.89e: %A.type.df0 = facet_value %tuple.type.59f, (%A.impl_witness.0e0) [concrete]
// CHECK:STDOUT:   %A.WithSelf.F.type.d74: type = fn_type @A.WithSelf.F, @A.WithSelf(%X, %A.facet.89e) [concrete]
// CHECK:STDOUT:   %A.WithSelf.F.cd4: %A.WithSelf.F.type.d74 = struct_value () [concrete]
// CHECK:STDOUT:   %.744: type = fn_type_with_self_type %A.WithSelf.F.type.d74, %A.facet.89e [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.34e: type = pattern_type %Z [concrete]
// CHECK:STDOUT:   %u.param_patt.a3e: %pattern_type.34e = value_param_pattern [concrete]
// CHECK:STDOUT:   %u.patt.5d3: %pattern_type.34e = wrapper_binding_pattern u, %u.param_patt.a3e [concrete]
// CHECK:STDOUT:   %tuple.415: %tuple.type.649 = tuple_value (%X, %tuple.71f, %Z) [concrete]
// CHECK:STDOUT:   %tuple.type.d40: type = tuple_type (%X, %tuple.type.59f, %Z) [concrete]
// CHECK:STDOUT:   %.1a9: Core.Form = init_form %tuple.type.d40 [concrete]
// CHECK:STDOUT:   %pattern_type.6a8: type = pattern_type %tuple.type.d40 [concrete]
// CHECK:STDOUT:   %return.param_patt.fb7: %pattern_type.6a8 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.badc: %pattern_type.6a8 = return_slot_pattern %return.param_patt.fb7, %tuple.type.d40 [concrete]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.specific_fn.981: <specific function> = specific_function %tuple.type.as.A.impl.F.749, @tuple.type.as.A.impl.F(%Y1, %Y2, %X, %Z) [concrete]
// CHECK:STDOUT:   %Z.val: %Z = struct_value () [concrete]
// CHECK:STDOUT:   %.7dc: ref %Z = temporary invalid, %Z.val [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %facet_type.ece: type = facet_type <@DestroyImpl & @SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.273: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.5b5b: %T.as_type.as.DestroyImpl.impl.Op.type.273 = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc24_39.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc24_39.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc24_39.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc24_39.5 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.e85: %SubobjectDestroy.type = facet_value %Y2, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.0db: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.e85) [concrete]
// CHECK:STDOUT:   %facet_value.ed4: %facet_type.ece = facet_value %Y2, (%DestroyImpl.impl_witness.0db, %custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.51d: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%facet_value.ed4) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.920: %SubobjectDestroy.type = facet_value %Y1, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.6c4: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.920) [concrete]
// CHECK:STDOUT:   %facet_value.cf6: %facet_type.ece = facet_value %Y1, (%DestroyImpl.impl_witness.6c4, %custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.ead: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%facet_value.cf6) [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.6: type = fn_type @SubobjectDestroy.WithSelf.Op.loc24_39.6 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.6: %SubobjectDestroy.WithSelf.Op.type.6870fd.6 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.6: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.6), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.7: type = fn_type @SubobjectDestroy.WithSelf.Op.loc24_39.7 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.7: %SubobjectDestroy.WithSelf.Op.type.6870fd.7 = struct_value () [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.5d5: %SubobjectDestroy.type = facet_value %Z, (%custom_witness.b80ef6.6) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.72a: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.5d5) [concrete]
// CHECK:STDOUT:   %facet_value.469: %facet_type.ece = facet_value %Z, (%DestroyImpl.impl_witness.72a, %custom_witness.b80ef6.6) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.c54: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%facet_value.469) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.d46: %SubobjectDestroy.type = facet_value %tuple.type.59f, (%custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.937: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.d46) [concrete]
// CHECK:STDOUT:   %facet_value.7f4: %facet_type.ece = facet_value %tuple.type.59f, (%DestroyImpl.impl_witness.937, %custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.868: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%facet_value.7f4) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.fc5: %SubobjectDestroy.type = facet_value %X, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.678: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.fc5) [concrete]
// CHECK:STDOUT:   %facet_value.9d2: %facet_type.ece = facet_value %X, (%DestroyImpl.impl_witness.678, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.e0d: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%facet_value.9d2) [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.7: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.7), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.392: %SubobjectDestroy.type = facet_value %tuple.type.d40, (%custom_witness.b80ef6.7) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.f58: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.392) [concrete]
// CHECK:STDOUT:   %complete_type.8f6: <witness> = complete_type_witness %tuple.type.d40 [concrete]
// CHECK:STDOUT:   %.422: require_specific_def_type = require_specific_def @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.392) [concrete]
// CHECK:STDOUT:   %facet_value.409: %facet_type.ece = facet_value %tuple.type.d40, (%DestroyImpl.impl_witness.f58, %custom_witness.b80ef6.7) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.faa: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%facet_value.409) [concrete]
// CHECK:STDOUT:   %BitAndWith.type.a06: type = generic_interface_type @BitAndWith [concrete]
// CHECK:STDOUT:   %BitAndWith.generic: %BitAndWith.type.a06 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndWith.type.411: type = facet_type <@BitAndWith, @BitAndWith(type)> [concrete]
// CHECK:STDOUT:   %BitAndWith.impl_witness: <witness> = impl_witness imports.%BitAndWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %BitAndWith.facet: %BitAndWith.type.411 = facet_value type, (%BitAndWith.impl_witness) [concrete]
// CHECK:STDOUT:   %BitAndWith.WithSelf.Op.type.0cf: type = fn_type @BitAndWith.WithSelf.Op, @BitAndWith.WithSelf(type, %BitAndWith.facet) [concrete]
// CHECK:STDOUT:   %.85c: type = fn_type_with_self_type %BitAndWith.WithSelf.Op.type.0cf, %BitAndWith.facet [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.type: type = fn_type @type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op: %type.as.BitAndWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.bound: <bound method> = bound_method %A.type.df0, %type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %facet_type.572: type = facet_type <@A, @A(%X) & @DestroyImpl & @SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %pattern_type.03d: type = pattern_type %facet_type.572 [concrete]
// CHECK:STDOUT:   %T.patt.4c0: %pattern_type.03d = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.80e: %facet_type.572 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %CallGeneric.type: type = fn_type @CallGeneric [concrete]
// CHECK:STDOUT:   %CallGeneric: %CallGeneric.type = struct_value () [concrete]
// CHECK:STDOUT:   %A.WithSelf.F.type.298: type = fn_type @A.WithSelf.F, @A.WithSelf(%X, %T.80e) [symbolic]
// CHECK:STDOUT:   %A.WithSelf.F.d25: %A.WithSelf.F.type.298 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.8e3: type = facet_access_type %T.80e [symbolic]
// CHECK:STDOUT:   %A.lookup_impl_witness: <witness> = lookup_impl_witness %T.80e, @A, @A(%X) [symbolic]
// CHECK:STDOUT:   %A.facet.5d2: %A.type.df0 = facet_value %T.as_type.8e3, (%A.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %A.WithSelf.F.type.9bb: type = fn_type @A.WithSelf.F, @A.WithSelf(%X, %A.facet.5d2) [symbolic]
// CHECK:STDOUT:   %A.WithSelf.F.8c8: %A.WithSelf.F.type.9bb = struct_value () [symbolic]
// CHECK:STDOUT:   %.ad0: type = fn_type_with_self_type %A.WithSelf.F.type.9bb, %A.facet.5d2 [symbolic]
// CHECK:STDOUT:   %impl.elem0.327: %.ad0 = impl_witness_access %A.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %tuple.type.202: type = tuple_type (type, %A.type.df0, type) [concrete]
// CHECK:STDOUT:   %tuple.612: %tuple.type.202 = tuple_value (%X, %A.facet.5d2, %Z) [symbolic]
// CHECK:STDOUT:   %tuple.type.b25: type = tuple_type (%X, %T.as_type.8e3, %Z) [symbolic]
// CHECK:STDOUT:   %.55e: Core.Form = init_form %tuple.type.b25 [symbolic]
// CHECK:STDOUT:   %pattern_type.25a: type = pattern_type %tuple.type.b25 [symbolic]
// CHECK:STDOUT:   %return.param_patt.08d: %pattern_type.25a = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.e9a: %pattern_type.25a = return_slot_pattern %return.param_patt.08d, %tuple.type.b25 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.b41: <specific function> = specific_impl_function %impl.elem0.327, @A.WithSelf.F(%X, %A.facet.5d2, %Z) [symbolic]
// CHECK:STDOUT:   %require_complete.ac9: <witness> = require_complete_type %tuple.type.b25 [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.dff: <witness> = lookup_impl_witness %tuple.type.b25, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.30d: <witness> = lookup_impl_witness %tuple.type.b25, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.facet.0af: %SubobjectDestroy.type = facet_value %tuple.type.b25, (%SubobjectDestroy.lookup_impl_witness.30d) [symbolic]
// CHECK:STDOUT:   %.e37: require_specific_def_type = require_specific_def @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.0af) [symbolic]
// CHECK:STDOUT:   %facet_value.9e6: %facet_type.ece = facet_value %tuple.type.b25, (%DestroyImpl.lookup_impl_witness.dff, %SubobjectDestroy.lookup_impl_witness.30d) [symbolic]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.4cf: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%facet_value.9e6) [symbolic]
// CHECK:STDOUT:   %CallIndirect.type: type = fn_type @CallIndirect [concrete]
// CHECK:STDOUT:   %CallIndirect: %CallIndirect.type = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.e23: %facet_type.572 = facet_value %tuple.type.59f, (%A.impl_witness.0e0, %DestroyImpl.impl_witness.937, %custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %CallGeneric.specific_fn: <specific function> = specific_function %CallGeneric, @CallGeneric(%facet_value.e23) [concrete]
// CHECK:STDOUT:   %tuple.b5b: %tuple.type.202 = tuple_value (%X, %A.facet.89e, %Z) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .BitAndWith = %Core.BitAndWith
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT:   %Core.import_ref.688: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.273) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.5b5b)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.688), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%facet_type.ece]
// CHECK:STDOUT:   %Core.BitAndWith: %BitAndWith.type.a06 = import_ref Core//prelude/parts/as, BitAndWith, loaded [concrete = constants.%BitAndWith.generic]
// CHECK:STDOUT:   %Core.import_ref.9c3: %type.as.BitAndWith.impl.Op.type = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, loaded [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:   %BitAndWith.impl_witness_table = impl_witness_table (%Core.import_ref.9c3), @type.as.BitAndWith.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .X = %X.decl
// CHECK:STDOUT:     .Y1 = %Y1.decl
// CHECK:STDOUT:     .Y2 = %Y2.decl
// CHECK:STDOUT:     .Z = %Z.decl
// CHECK:STDOUT:     .Call = %Call.decl
// CHECK:STDOUT:     .CallGeneric = %CallGeneric.decl
// CHECK:STDOUT:     .CallIndirect = %CallIndirect.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: %A.type.55a = interface_decl @A [concrete = constants.%A.generic] {
// CHECK:STDOUT:     %T.patt.loc5_14.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc5_14.2 (constants.%T.patt.3a0)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5_16.1: type = splice_block %.loc5_16.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc5_16.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc5_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc5_14.1 (constants.%T.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.decl: type = class_decl @X [concrete = constants.%X] {} {}
// CHECK:STDOUT:   %Y1.decl: type = class_decl @Y1 [concrete = constants.%Y1] {} {}
// CHECK:STDOUT:   %Y2.decl: type = class_decl @Y2 [concrete = constants.%Y2] {} {}
// CHECK:STDOUT:   %Z.decl: type = class_decl @Z [concrete = constants.%Z] {} {}
// CHECK:STDOUT:   impl_decl @tuple.type.as.A.impl [concrete] {
// CHECK:STDOUT:     %V1.patt.loc14_16.1: %pattern_type.9a5 = symbolic_binding_pattern V1, 0 [symbolic = %V1.patt.loc14_16.2 (constants.%V1.patt)]
// CHECK:STDOUT:     %V2.patt.loc14_26.1: %pattern_type.9a5 = symbolic_binding_pattern V2, 1 [symbolic = %V2.patt.loc14_26.2 (constants.%V2.patt)]
// CHECK:STDOUT:     %W.patt.loc14_35.1: %pattern_type.9a5 = symbolic_binding_pattern W, 2 [symbolic = %W.patt.loc14_35.2 (constants.%W.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %V1.ref: type = name_ref V1, %V1.loc14_16.2 [symbolic = %V1.loc14_16.1 (constants.%V1)]
// CHECK:STDOUT:     %V2.ref: type = name_ref V2, %V2.loc14_26.2 [symbolic = %V2.loc14_26.1 (constants.%V2)]
// CHECK:STDOUT:     %.loc14_50.1: %tuple.type.693 = tuple_literal (%V1.ref, %V2.ref) [symbolic = %tuple (constants.%tuple.fa2)]
// CHECK:STDOUT:     %.loc14_50.2: type = converted %.loc14_50.1, constants.%tuple.type.a5e [symbolic = %tuple.type (constants.%tuple.type.a5e)]
// CHECK:STDOUT:     %A.ref: %A.type.55a = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:     %W.ref: type = name_ref W, %W.loc14_35.2 [symbolic = %W.loc14_35.1 (constants.%W)]
// CHECK:STDOUT:     %A.type.loc14_58.2: type = facet_type <@A, @A(constants.%W)> [symbolic = %A.type.loc14_58.1 (constants.%A.type.3ad)]
// CHECK:STDOUT:     %.loc14_18.1: type = splice_block %.loc14_18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen.loc14_16: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc14_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %V1.loc14_16.2: type = symbolic_binding V1, 0 [symbolic = %V1.loc14_16.1 (constants.%V1)]
// CHECK:STDOUT:     %.loc14_28.1: type = splice_block %.loc14_28.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen.loc14_26: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc14_28.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %V2.loc14_26.2: type = symbolic_binding V2, 1 [symbolic = %V2.loc14_26.1 (constants.%V2)]
// CHECK:STDOUT:     %.loc14_37.1: type = splice_block %.loc14_37.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen.loc14_35: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc14_37.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %W.loc14_35.2: type = symbolic_binding W, 2 [symbolic = %W.loc14_35.1 (constants.%W)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc14: <witness> = impl_self_witness @tuple.type.as.A.impl.%.loc14_50.2, @A, @A(constants.%W) [symbolic = @tuple.type.as.A.impl.%.loc14_60 (constants.%.b95)]
// CHECK:STDOUT:   %Call.decl: %Call.type = fn_decl @Call [concrete = constants.%Call] {} {}
// CHECK:STDOUT:   %CallGeneric.decl: %CallGeneric.type = fn_decl @CallGeneric [concrete = constants.%CallGeneric] {
// CHECK:STDOUT:     %T.patt.loc27_25.1: %pattern_type.03d = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc27_25.2 (constants.%T.patt.4c0)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc27_32.1: type = splice_block %.loc27_32.3 [concrete = constants.%facet_type.572] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %A.ref: %A.type.55a = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:       %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:       %A.type: type = facet_type <@A, @A(constants.%X)> [concrete = constants.%A.type.df0]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%facet_type.ece]
// CHECK:STDOUT:       %impl.elem0.loc27: %.85c = impl_witness_access constants.%BitAndWith.impl_witness, element0 [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:       %bound_method: <bound method> = bound_method %A.type, %impl.elem0.loc27 [concrete = constants.%type.as.BitAndWith.impl.Op.bound]
// CHECK:STDOUT:       %type.as.BitAndWith.impl.Op.call: init type = call %bound_method(%A.type, %Destroy.ref) [concrete = constants.%facet_type.572]
// CHECK:STDOUT:       %.loc27_32.2: type = value_of_initializer %type.as.BitAndWith.impl.Op.call [concrete = constants.%facet_type.572]
// CHECK:STDOUT:       %.loc27_32.3: type = converted %type.as.BitAndWith.impl.Op.call, %.loc27_32.2 [concrete = constants.%facet_type.572]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc27_25.2: %facet_type.572 = symbolic_binding T, 0 [symbolic = %T.loc27_25.1 (constants.%T.80e)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallIndirect.decl: %CallIndirect.type = fn_decl @CallIndirect [concrete = constants.%CallIndirect] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @A(%T.loc5_14.2: type) {
// CHECK:STDOUT:   %T.patt.loc5_14.2: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc5_14.2 (constants.%T.patt.3a0)]
// CHECK:STDOUT:   %T.loc5_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc5_14.1 (constants.%T.67d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.type: type = facet_type <@A, @A(%T.loc5_14.1)> [symbolic = %A.type (constants.%A.type.acd)]
// CHECK:STDOUT:   %Self.loc5_22.2: @A.%A.type (%A.type.acd) = symbolic_binding Self, 1 [symbolic = %Self.loc5_22.2 (constants.%Self.49a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc5_22.1: @A.%A.type (%A.type.acd) = symbolic_binding Self, 1 [symbolic = %Self.loc5_22.2 (constants.%Self.49a)]
// CHECK:STDOUT:     %A.WithSelf.decl = interface_with_self_decl @A [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %A.WithSelf.F.decl: @A.WithSelf.%A.WithSelf.F.type (%A.WithSelf.F.type.08d) = fn_decl @A.WithSelf.F [symbolic = @A.WithSelf.%A.WithSelf.F (constants.%A.WithSelf.F.3fa)] {
// CHECK:STDOUT:       %U.patt.loc6_17.1: %pattern_type.9a5 = symbolic_binding_pattern U, 2 [symbolic = %U.patt.loc6_17.2 (constants.%U.patt.5c4)]
// CHECK:STDOUT:       %u.param_patt.loc6_26.1: @A.WithSelf.F.%pattern_type.loc6_26 (%pattern_type.62e) = value_param_pattern [symbolic = %u.param_patt.loc6_26.2 (constants.%u.param_patt.6ef)]
// CHECK:STDOUT:       %u.patt.loc6_26.1: @A.WithSelf.F.%pattern_type.loc6_26 (%pattern_type.62e) = wrapper_binding_pattern u, %u.param_patt.loc6_26.1 [symbolic = %u.patt.loc6_26.2 (constants.%u.patt.9b1)]
// CHECK:STDOUT:       %return.param_patt.loc6_45.1: @A.WithSelf.F.%pattern_type.loc6_45 (%pattern_type.d6c) = out_param_pattern [symbolic = %return.param_patt.loc6_45.2 (constants.%return.param_patt.52a)]
// CHECK:STDOUT:       %return.patt.loc6_31.1: @A.WithSelf.F.%pattern_type.loc6_45 (%pattern_type.d6c) = return_slot_pattern %return.param_patt.loc6_45.1, %.loc6_45.4 [symbolic = %return.patt.loc6_31.2 (constants.%return.patt.808)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @A.%T.loc5_14.2 [symbolic = %T (constants.%T.67d)]
// CHECK:STDOUT:       %.loc6_38: @A.WithSelf.F.%A.type (%A.type.acd) = specific_constant @A.%Self.loc5_22.1, @A(constants.%T.67d) [symbolic = %Self (constants.%Self.49a)]
// CHECK:STDOUT:       %Self.ref: @A.WithSelf.F.%A.type (%A.type.acd) = name_ref Self, %.loc6_38 [symbolic = %Self (constants.%Self.49a)]
// CHECK:STDOUT:       %U.ref.loc6_44: type = name_ref U, %U.loc6_17.2 [symbolic = %U.loc6_17.1 (constants.%U.4e7)]
// CHECK:STDOUT:       %.loc6_45.2: @A.WithSelf.F.%tuple.type.loc6_45.1 (%tuple.type.a31) = tuple_literal (%T.ref, %Self.ref, %U.ref.loc6_44) [symbolic = %tuple (constants.%tuple.62b)]
// CHECK:STDOUT:       %Self.as_type.loc6_45.2: type = facet_access_type constants.%Self.49a [symbolic = %Self.as_type.loc6_45.1 (constants.%Self.as_type.763)]
// CHECK:STDOUT:       %.loc6_45.3: type = converted constants.%Self.49a, %Self.as_type.loc6_45.2 [symbolic = %Self.as_type.loc6_45.1 (constants.%Self.as_type.763)]
// CHECK:STDOUT:       %.loc6_45.4: type = converted %.loc6_45.2, constants.%tuple.type.d73 [symbolic = %tuple.type.loc6_45.2 (constants.%tuple.type.d73)]
// CHECK:STDOUT:       %.loc6_45.5: Core.Form = init_form %.loc6_45.4 [symbolic = %.loc6_45.1 (constants.%.bc3)]
// CHECK:STDOUT:       %.loc6_19.1: type = splice_block %.loc6_19.2 [concrete = type] {
// CHECK:STDOUT:         %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:         %.loc6_19.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %U.loc6_17.2: type = symbolic_binding U, 2 [symbolic = %U.loc6_17.1 (constants.%U.4e7)]
// CHECK:STDOUT:       %u.param: @A.WithSelf.F.%U.loc6_17.1 (%U.4e7) = value_param call_param0
// CHECK:STDOUT:       %U.ref.loc6_28: type = name_ref U, %U.loc6_17.2 [symbolic = %U.loc6_17.1 (constants.%U.4e7)]
// CHECK:STDOUT:       %u: @A.WithSelf.F.%U.loc6_17.1 (%U.4e7) = wrapper_binding u, %u.param
// CHECK:STDOUT:       %return.param: ref @A.WithSelf.F.%tuple.type.loc6_45.2 (%tuple.type.d73) = out_param call_param1
// CHECK:STDOUT:       %return: ref @A.WithSelf.F.%tuple.type.loc6_45.2 (%tuple.type.d73) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc6_46.1: @A.WithSelf.%A.assoc_type (%A.assoc_type.447) = assoc_entity element0, %A.WithSelf.F.decl [symbolic = %assoc0.loc6_46.2 (constants.%assoc0.d4b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc5_22.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .F = @A.WithSelf.%assoc0.loc6_46.1
// CHECK:STDOUT:     .W = <poisoned>
// CHECK:STDOUT:     .V1 = <poisoned>
// CHECK:STDOUT:     .V2 = <poisoned>
// CHECK:STDOUT:     witness = (@A.WithSelf.%A.WithSelf.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @tuple.type.as.A.impl(%V1.loc14_16.2: type, %V2.loc14_26.2: type, %W.loc14_35.2: type) {
// CHECK:STDOUT:   %V1.patt.loc14_16.2: %pattern_type.9a5 = symbolic_binding_pattern V1, 0 [symbolic = %V1.patt.loc14_16.2 (constants.%V1.patt)]
// CHECK:STDOUT:   %V1.loc14_16.1: type = symbolic_binding V1, 0 [symbolic = %V1.loc14_16.1 (constants.%V1)]
// CHECK:STDOUT:   %V2.patt.loc14_26.2: %pattern_type.9a5 = symbolic_binding_pattern V2, 1 [symbolic = %V2.patt.loc14_26.2 (constants.%V2.patt)]
// CHECK:STDOUT:   %V2.loc14_26.1: type = symbolic_binding V2, 1 [symbolic = %V2.loc14_26.1 (constants.%V2)]
// CHECK:STDOUT:   %W.patt.loc14_35.2: %pattern_type.9a5 = symbolic_binding_pattern W, 2 [symbolic = %W.patt.loc14_35.2 (constants.%W.patt)]
// CHECK:STDOUT:   %W.loc14_35.1: type = symbolic_binding W, 2 [symbolic = %W.loc14_35.1 (constants.%W)]
// CHECK:STDOUT:   %tuple: %tuple.type.693 = tuple_value (%V1.loc14_16.1, %V2.loc14_26.1) [symbolic = %tuple (constants.%tuple.fa2)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%V1.loc14_16.1, %V2.loc14_26.1) [symbolic = %tuple.type (constants.%tuple.type.a5e)]
// CHECK:STDOUT:   %A.type.loc14_58.1: type = facet_type <@A, @A(%W.loc14_35.1)> [symbolic = %A.type.loc14_58.1 (constants.%A.type.3ad)]
// CHECK:STDOUT:   %.loc14_60: <witness> = impl_self_witness %tuple.type, @A, @A(%W.loc14_35.1) [symbolic = %.loc14_60 (constants.%.b95)]
// CHECK:STDOUT:   %A.impl_witness.loc14_60.2: <witness> = impl_witness %A.impl_witness_table, @tuple.type.as.A.impl(%V1.loc14_16.1, %V2.loc14_26.1, %W.loc14_35.1) [symbolic = %A.impl_witness.loc14_60.2 (constants.%A.impl_witness.07b)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %A.type.loc14_58.1 [symbolic = %require_complete (constants.%require_complete.7b2)]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type: type = fn_type @tuple.type.as.A.impl.F, @tuple.type.as.A.impl(%V1.loc14_16.1, %V2.loc14_26.1, %W.loc14_35.1) [symbolic = %tuple.type.as.A.impl.F.type (constants.%tuple.type.as.A.impl.F.type.44a)]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F: @tuple.type.as.A.impl.%tuple.type.as.A.impl.F.type (%tuple.type.as.A.impl.F.type.44a) = struct_value () [symbolic = %tuple.type.as.A.impl.F (constants.%tuple.type.as.A.impl.F.742)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc14_50.2 as %A.type.loc14_58.2 {
// CHECK:STDOUT:     %tuple.type.as.A.impl.F.decl: @tuple.type.as.A.impl.%tuple.type.as.A.impl.F.type (%tuple.type.as.A.impl.F.type.44a) = fn_decl @tuple.type.as.A.impl.F [symbolic = @tuple.type.as.A.impl.%tuple.type.as.A.impl.F (constants.%tuple.type.as.A.impl.F.742)] {
// CHECK:STDOUT:       %U.patt.loc17_17.1: %pattern_type.9a5 = symbolic_binding_pattern U, 3 [symbolic = %U.patt.loc17_17.2 (constants.%U.patt.755)]
// CHECK:STDOUT:       %u.param_patt.loc17_26.1: @tuple.type.as.A.impl.F.%pattern_type.loc17_26 (%pattern_type.e4a) = value_param_pattern [symbolic = %u.param_patt.loc17_26.2 (constants.%u.param_patt.afe)]
// CHECK:STDOUT:       %u.patt.loc17_26.1: @tuple.type.as.A.impl.F.%pattern_type.loc17_26 (%pattern_type.e4a) = wrapper_binding_pattern u, %u.param_patt.loc17_26.1 [symbolic = %u.patt.loc17_26.2 (constants.%u.patt.60c)]
// CHECK:STDOUT:       %return.param_patt.loc17_49.1: @tuple.type.as.A.impl.F.%pattern_type.loc17_49 (%pattern_type.5cc) = out_param_pattern [symbolic = %return.param_patt.loc17_49.2 (constants.%return.param_patt.b1c)]
// CHECK:STDOUT:       %return.patt.loc17_31.1: @tuple.type.as.A.impl.F.%pattern_type.loc17_49 (%pattern_type.5cc) = return_slot_pattern %return.param_patt.loc17_49.1, %.loc17_49.5 [symbolic = %return.patt.loc17_31.2 (constants.%return.patt.bad9)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %W.ref: type = name_ref W, @tuple.type.as.A.impl.%W.loc14_35.2 [symbolic = %W (constants.%W)]
// CHECK:STDOUT:       %V1.ref: type = name_ref V1, @tuple.type.as.A.impl.%V1.loc14_16.2 [symbolic = %V1 (constants.%V1)]
// CHECK:STDOUT:       %V2.ref: type = name_ref V2, @tuple.type.as.A.impl.%V2.loc14_26.2 [symbolic = %V2 (constants.%V2)]
// CHECK:STDOUT:       %.loc17_45: %tuple.type.693 = tuple_literal (%V1.ref, %V2.ref) [symbolic = %tuple.loc17_45 (constants.%tuple.fa2)]
// CHECK:STDOUT:       %U.ref.loc17_48: type = name_ref U, %U.loc17_17.2 [symbolic = %U.loc17_17.1 (constants.%U.ce4)]
// CHECK:STDOUT:       %.loc17_49.3: %tuple.type.649 = tuple_literal (%W.ref, %.loc17_45, %U.ref.loc17_48) [symbolic = %tuple.loc17_49 (constants.%tuple.cd3)]
// CHECK:STDOUT:       %.loc17_49.4: type = converted constants.%tuple.fa2, constants.%tuple.type.a5e [symbolic = %tuple.type.loc17_49.1 (constants.%tuple.type.a5e)]
// CHECK:STDOUT:       %.loc17_49.5: type = converted %.loc17_49.3, constants.%tuple.type.897 [symbolic = %tuple.type.loc17_49.2 (constants.%tuple.type.897)]
// CHECK:STDOUT:       %.loc17_49.6: Core.Form = init_form %.loc17_49.5 [symbolic = %.loc17_49.2 (constants.%.f8e)]
// CHECK:STDOUT:       %.loc17_19.1: type = splice_block %.loc17_19.2 [concrete = type] {
// CHECK:STDOUT:         %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:         %.loc17_19.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %U.loc17_17.2: type = symbolic_binding U, 3 [symbolic = %U.loc17_17.1 (constants.%U.ce4)]
// CHECK:STDOUT:       %u.param: @tuple.type.as.A.impl.F.%U.loc17_17.1 (%U.ce4) = value_param call_param0
// CHECK:STDOUT:       %U.ref.loc17_28: type = name_ref U, %U.loc17_17.2 [symbolic = %U.loc17_17.1 (constants.%U.ce4)]
// CHECK:STDOUT:       %u: @tuple.type.as.A.impl.F.%U.loc17_17.1 (%U.ce4) = wrapper_binding u, %u.param
// CHECK:STDOUT:       %return.param: ref @tuple.type.as.A.impl.F.%tuple.type.loc17_49.2 (%tuple.type.897) = out_param call_param1
// CHECK:STDOUT:       %return: ref @tuple.type.as.A.impl.F.%tuple.type.loc17_49.2 (%tuple.type.897) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %A.impl_witness_table = impl_witness_table (%tuple.type.as.A.impl.F.decl), @tuple.type.as.A.impl [concrete]
// CHECK:STDOUT:     %A.impl_witness.loc14_60.1: <witness> = impl_witness %A.impl_witness_table, @tuple.type.as.A.impl(constants.%V1, constants.%V2, constants.%W) [symbolic = %A.impl_witness.loc14_60.2 (constants.%A.impl_witness.07b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .W = <poisoned>
// CHECK:STDOUT:     .V1 = <poisoned>
// CHECK:STDOUT:     .V2 = <poisoned>
// CHECK:STDOUT:     .F = %tuple.type.as.A.impl.F.decl
// CHECK:STDOUT:     extend %A.type.loc14_58.2
// CHECK:STDOUT:     witness = %A.impl_witness.loc14_60.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @X {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Y1 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Y1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Y2 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Y2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Z {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Z
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @A.WithSelf.F(@A.%T.loc5_14.2: type, @A.%Self.loc5_22.1: @A.%A.type (%A.type.acd), %U.loc6_17.2: type) {
// CHECK:STDOUT:   %U.patt.loc6_17.2: %pattern_type.9a5 = symbolic_binding_pattern U, 2 [symbolic = %U.patt.loc6_17.2 (constants.%U.patt.5c4)]
// CHECK:STDOUT:   %U.loc6_17.1: type = symbolic_binding U, 2 [symbolic = %U.loc6_17.1 (constants.%U.4e7)]
// CHECK:STDOUT:   %pattern_type.loc6_26: type = pattern_type %U.loc6_17.1 [symbolic = %pattern_type.loc6_26 (constants.%pattern_type.62e)]
// CHECK:STDOUT:   %u.param_patt.loc6_26.2: @A.WithSelf.F.%pattern_type.loc6_26 (%pattern_type.62e) = value_param_pattern [symbolic = %u.param_patt.loc6_26.2 (constants.%u.param_patt.6ef)]
// CHECK:STDOUT:   %u.patt.loc6_26.2: @A.WithSelf.F.%pattern_type.loc6_26 (%pattern_type.62e) = wrapper_binding_pattern u, %u.param_patt.loc6_26.2 [symbolic = %u.patt.loc6_26.2 (constants.%u.patt.9b1)]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T.67d)]
// CHECK:STDOUT:   %A.type: type = facet_type <@A, @A(%T)> [symbolic = %A.type (constants.%A.type.acd)]
// CHECK:STDOUT:   %Self: @A.WithSelf.F.%A.type (%A.type.acd) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.49a)]
// CHECK:STDOUT:   %tuple.type.loc6_45.1: type = tuple_type (type, %A.type, type) [symbolic = %tuple.type.loc6_45.1 (constants.%tuple.type.a31)]
// CHECK:STDOUT:   %tuple: @A.WithSelf.F.%tuple.type.loc6_45.1 (%tuple.type.a31) = tuple_value (%T, %Self, %U.loc6_17.1) [symbolic = %tuple (constants.%tuple.62b)]
// CHECK:STDOUT:   %Self.as_type.loc6_45.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc6_45.1 (constants.%Self.as_type.763)]
// CHECK:STDOUT:   %tuple.type.loc6_45.2: type = tuple_type (%T, %Self.as_type.loc6_45.1, %U.loc6_17.1) [symbolic = %tuple.type.loc6_45.2 (constants.%tuple.type.d73)]
// CHECK:STDOUT:   %.loc6_45.1: Core.Form = init_form %tuple.type.loc6_45.2 [symbolic = %.loc6_45.1 (constants.%.bc3)]
// CHECK:STDOUT:   %pattern_type.loc6_45: type = pattern_type %tuple.type.loc6_45.2 [symbolic = %pattern_type.loc6_45 (constants.%pattern_type.d6c)]
// CHECK:STDOUT:   %return.param_patt.loc6_45.2: @A.WithSelf.F.%pattern_type.loc6_45 (%pattern_type.d6c) = out_param_pattern [symbolic = %return.param_patt.loc6_45.2 (constants.%return.param_patt.52a)]
// CHECK:STDOUT:   %return.patt.loc6_31.2: @A.WithSelf.F.%pattern_type.loc6_45 (%pattern_type.d6c) = return_slot_pattern %return.param_patt.loc6_45.2, %tuple.type.loc6_45.2 [symbolic = %return.patt.loc6_31.2 (constants.%return.patt.808)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param: @A.WithSelf.F.%U.loc6_17.1 (%U.4e7)) -> out %return.param: @A.WithSelf.F.%tuple.type.loc6_45.2 (%tuple.type.d73);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @tuple.type.as.A.impl.F(@tuple.type.as.A.impl.%V1.loc14_16.2: type, @tuple.type.as.A.impl.%V2.loc14_26.2: type, @tuple.type.as.A.impl.%W.loc14_35.2: type, %U.loc17_17.2: type) {
// CHECK:STDOUT:   %U.patt.loc17_17.2: %pattern_type.9a5 = symbolic_binding_pattern U, 3 [symbolic = %U.patt.loc17_17.2 (constants.%U.patt.755)]
// CHECK:STDOUT:   %U.loc17_17.1: type = symbolic_binding U, 3 [symbolic = %U.loc17_17.1 (constants.%U.ce4)]
// CHECK:STDOUT:   %pattern_type.loc17_26: type = pattern_type %U.loc17_17.1 [symbolic = %pattern_type.loc17_26 (constants.%pattern_type.e4a)]
// CHECK:STDOUT:   %u.param_patt.loc17_26.2: @tuple.type.as.A.impl.F.%pattern_type.loc17_26 (%pattern_type.e4a) = value_param_pattern [symbolic = %u.param_patt.loc17_26.2 (constants.%u.param_patt.afe)]
// CHECK:STDOUT:   %u.patt.loc17_26.2: @tuple.type.as.A.impl.F.%pattern_type.loc17_26 (%pattern_type.e4a) = wrapper_binding_pattern u, %u.param_patt.loc17_26.2 [symbolic = %u.patt.loc17_26.2 (constants.%u.patt.60c)]
// CHECK:STDOUT:   %W: type = symbolic_binding W, 2 [symbolic = %W (constants.%W)]
// CHECK:STDOUT:   %V1: type = symbolic_binding V1, 0 [symbolic = %V1 (constants.%V1)]
// CHECK:STDOUT:   %V2: type = symbolic_binding V2, 1 [symbolic = %V2 (constants.%V2)]
// CHECK:STDOUT:   %tuple.loc17_45: %tuple.type.693 = tuple_value (%V1, %V2) [symbolic = %tuple.loc17_45 (constants.%tuple.fa2)]
// CHECK:STDOUT:   %tuple.loc17_49: %tuple.type.649 = tuple_value (%W, %tuple.loc17_45, %U.loc17_17.1) [symbolic = %tuple.loc17_49 (constants.%tuple.cd3)]
// CHECK:STDOUT:   %tuple.type.loc17_49.1: type = tuple_type (%V1, %V2) [symbolic = %tuple.type.loc17_49.1 (constants.%tuple.type.a5e)]
// CHECK:STDOUT:   %tuple.type.loc17_49.2: type = tuple_type (%W, %tuple.type.loc17_49.1, %U.loc17_17.1) [symbolic = %tuple.type.loc17_49.2 (constants.%tuple.type.897)]
// CHECK:STDOUT:   %.loc17_49.2: Core.Form = init_form %tuple.type.loc17_49.2 [symbolic = %.loc17_49.2 (constants.%.f8e)]
// CHECK:STDOUT:   %pattern_type.loc17_49: type = pattern_type %tuple.type.loc17_49.2 [symbolic = %pattern_type.loc17_49 (constants.%pattern_type.5cc)]
// CHECK:STDOUT:   %return.param_patt.loc17_49.2: @tuple.type.as.A.impl.F.%pattern_type.loc17_49 (%pattern_type.5cc) = out_param_pattern [symbolic = %return.param_patt.loc17_49.2 (constants.%return.param_patt.b1c)]
// CHECK:STDOUT:   %return.patt.loc17_31.2: @tuple.type.as.A.impl.F.%pattern_type.loc17_49 (%pattern_type.5cc) = return_slot_pattern %return.param_patt.loc17_49.2, %tuple.type.loc17_49.2 [symbolic = %return.patt.loc17_31.2 (constants.%return.patt.bad9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc17_26: <witness> = require_complete_type %U.loc17_17.1 [symbolic = %require_complete.loc17_26 (constants.%require_complete.5a4)]
// CHECK:STDOUT:   %require_complete.loc17_31: <witness> = require_complete_type %tuple.type.loc17_49.2 [symbolic = %require_complete.loc17_31 (constants.%require_complete.da0)]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type: type = fn_type @tuple.type.as.A.impl.F, @tuple.type.as.A.impl(%V1, %V2, %W) [symbolic = %tuple.type.as.A.impl.F.type (constants.%tuple.type.as.A.impl.F.type.44a)]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F: @tuple.type.as.A.impl.F.%tuple.type.as.A.impl.F.type (%tuple.type.as.A.impl.F.type.44a) = struct_value () [symbolic = %tuple.type.as.A.impl.F (constants.%tuple.type.as.A.impl.F.742)]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.specific_fn.loc18_12.2: <specific function> = specific_function %tuple.type.as.A.impl.F, @tuple.type.as.A.impl.F(%V1, %V2, %W, %U.loc17_17.1) [symbolic = %tuple.type.as.A.impl.F.specific_fn.loc18_12.2 (constants.%tuple.type.as.A.impl.F.specific_fn.37b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param: @tuple.type.as.A.impl.F.%U.loc17_17.1 (%U.ce4)) -> out %return.param: @tuple.type.as.A.impl.F.%tuple.type.loc17_49.2 (%tuple.type.897) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc18: @tuple.type.as.A.impl.F.%tuple.type.as.A.impl.F.type (%tuple.type.as.A.impl.F.type.44a) = specific_constant @tuple.type.as.A.impl.%tuple.type.as.A.impl.F.decl, @tuple.type.as.A.impl(constants.%V1, constants.%V2, constants.%W) [symbolic = %tuple.type.as.A.impl.F (constants.%tuple.type.as.A.impl.F.742)]
// CHECK:STDOUT:     %F.ref: @tuple.type.as.A.impl.F.%tuple.type.as.A.impl.F.type (%tuple.type.as.A.impl.F.type.44a) = name_ref F, %.loc18 [symbolic = %tuple.type.as.A.impl.F (constants.%tuple.type.as.A.impl.F.742)]
// CHECK:STDOUT:     %U.ref.loc18: type = name_ref U, %U.loc17_17.2 [symbolic = %U.loc17_17.1 (constants.%U.ce4)]
// CHECK:STDOUT:     %u.ref: @tuple.type.as.A.impl.F.%U.loc17_17.1 (%U.ce4) = name_ref u, %u
// CHECK:STDOUT:     %tuple.type.as.A.impl.F.specific_fn.loc18_12.1: <specific function> = specific_function %F.ref, @tuple.type.as.A.impl.F(constants.%V1, constants.%V2, constants.%W, constants.%U.ce4) [symbolic = %tuple.type.as.A.impl.F.specific_fn.loc18_12.2 (constants.%tuple.type.as.A.impl.F.specific_fn.37b)]
// CHECK:STDOUT:     %.loc17_49.1: ref @tuple.type.as.A.impl.F.%tuple.type.loc17_49.2 (%tuple.type.897) = splice_block %return.param {}
// CHECK:STDOUT:     %tuple.type.as.A.impl.F.call: init @tuple.type.as.A.impl.F.%tuple.type.loc17_49.2 (%tuple.type.897) to %.loc17_49.1 = call %tuple.type.as.A.impl.F.specific_fn.loc18_12.1(%u.ref)
// CHECK:STDOUT:     return %tuple.type.as.A.impl.F.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Y1.ref: type = name_ref Y1, file.%Y1.decl [concrete = constants.%Y1]
// CHECK:STDOUT:   %Y2.ref: type = name_ref Y2, file.%Y2.decl [concrete = constants.%Y2]
// CHECK:STDOUT:   %.loc24_12: %tuple.type.693 = tuple_literal (%Y1.ref, %Y2.ref) [concrete = constants.%tuple.71f]
// CHECK:STDOUT:   %.loc24_17: type = type_literal type [concrete = type]
// CHECK:STDOUT:   %.loc24_14: type = converted %.loc24_12, constants.%tuple.type.59f [concrete = constants.%tuple.type.59f]
// CHECK:STDOUT:   %A.ref: %A.type.55a = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:   %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %A.type: type = facet_type <@A, @A(constants.%X)> [concrete = constants.%A.type.df0]
// CHECK:STDOUT:   %A.facet: %A.type.df0 = facet_value %.loc24_14, (constants.%A.impl_witness.0e0) [concrete = constants.%A.facet.89e]
// CHECK:STDOUT:   %.loc24_23: %A.type.df0 = converted %.loc24_14, %A.facet [concrete = constants.%A.facet.89e]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc24_23 [concrete = constants.%tuple.type.59f]
// CHECK:STDOUT:   %.loc24_31.1: type = converted %.loc24_23, %as_type [concrete = constants.%tuple.type.59f]
// CHECK:STDOUT:   %.loc24_31.2: %A.assoc_type.a54 = specific_constant @A.WithSelf.%assoc0.loc6_46.1, @A.WithSelf(constants.%X, constants.%A.facet.89e) [concrete = constants.%assoc0.987]
// CHECK:STDOUT:   %F.ref: %A.assoc_type.a54 = name_ref F, %.loc24_31.2 [concrete = constants.%assoc0.987]
// CHECK:STDOUT:   %impl.elem0: %.744 = impl_witness_access constants.%A.impl_witness.0e0, element0 [concrete = constants.%tuple.type.as.A.impl.F.749]
// CHECK:STDOUT:   %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z]
// CHECK:STDOUT:   %.loc24_38.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @tuple.type.as.A.impl.F(constants.%Y1, constants.%Y2, constants.%X, constants.%Z) [concrete = constants.%tuple.type.as.A.impl.F.specific_fn.981]
// CHECK:STDOUT:   %.loc24_39.1: ref %tuple.type.d40 = temporary_storage
// CHECK:STDOUT:   %.loc24_38.2: ref %Z = temporary_storage
// CHECK:STDOUT:   %.loc24_38.3: init %Z to %.loc24_38.2 = class_init () [concrete = constants.%Z.val]
// CHECK:STDOUT:   %.loc24_38.4: init %Z = converted %.loc24_38.1, %.loc24_38.3 [concrete = constants.%Z.val]
// CHECK:STDOUT:   %.loc24_38.5: ref %Z = temporary %.loc24_38.2, %.loc24_38.4 [concrete = constants.%.7dc]
// CHECK:STDOUT:   %.loc24_38.6: %Z = acquire_value %.loc24_38.5 [concrete = constants.%Z.val]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.call: init %tuple.type.d40 to %.loc24_39.1 = call %specific_fn(%.loc24_38.6)
// CHECK:STDOUT:   %.loc24_39.2: ref %tuple.type.d40 = temporary %.loc24_39.1, %tuple.type.as.A.impl.F.call
// CHECK:STDOUT:   %facet_value.loc24_39.1: %facet_type.ece = facet_value constants.%tuple.type.d40, (constants.%DestroyImpl.impl_witness.f58, constants.%custom_witness.b80ef6.7) [concrete = constants.%facet_value.409]
// CHECK:STDOUT:   %.loc24_39.3: %facet_type.ece = converted constants.%tuple.type.d40, %facet_value.loc24_39.1 [concrete = constants.%facet_value.409]
// CHECK:STDOUT:   %facet_value.loc24_39.2: %facet_type.ece = facet_value constants.%tuple.type.d40, (constants.%DestroyImpl.impl_witness.f58, constants.%custom_witness.b80ef6.7) [concrete = constants.%facet_value.409]
// CHECK:STDOUT:   %.loc24_39.4: %facet_type.ece = converted constants.%tuple.type.d40, %facet_value.loc24_39.2 [concrete = constants.%facet_value.409]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.1: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.409) [concrete = constants.%SelfDestruct.specific_fn.faa]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_39: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc24_39.2)
// CHECK:STDOUT:   %facet_value.loc24_38.1: %facet_type.ece = facet_value constants.%Z, (constants.%DestroyImpl.impl_witness.72a, constants.%custom_witness.b80ef6.6) [concrete = constants.%facet_value.469]
// CHECK:STDOUT:   %.loc24_38.7: %facet_type.ece = converted constants.%Z, %facet_value.loc24_38.1 [concrete = constants.%facet_value.469]
// CHECK:STDOUT:   %facet_value.loc24_38.2: %facet_type.ece = facet_value constants.%Z, (constants.%DestroyImpl.impl_witness.72a, constants.%custom_witness.b80ef6.6) [concrete = constants.%facet_value.469]
// CHECK:STDOUT:   %.loc24_38.8: %facet_type.ece = converted constants.%Z, %facet_value.loc24_38.2 [concrete = constants.%facet_value.469]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.469) [concrete = constants.%SelfDestruct.specific_fn.c54]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_38: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc24_38.5)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc24_39.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc24_39.2(%self.param: ref %X) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc24_39.3(%self.param: ref %Y1) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc24_39.4(%self.param: ref %Y2) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc24_39.5(%self.param: ref %tuple.type.59f) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %tuple.elem1: ref %Y2 = tuple_access %self.param, element1
// CHECK:STDOUT:   %facet_value.loc24_39.1: %facet_type.ece = facet_value constants.%Y2, (constants.%DestroyImpl.impl_witness.0db, constants.%custom_witness.b80ef6.4) [concrete = constants.%facet_value.ed4]
// CHECK:STDOUT:   %.loc24_39.1: %facet_type.ece = converted constants.%Y2, %facet_value.loc24_39.1 [concrete = constants.%facet_value.ed4]
// CHECK:STDOUT:   %facet_value.loc24_39.2: %facet_type.ece = facet_value constants.%Y2, (constants.%DestroyImpl.impl_witness.0db, constants.%custom_witness.b80ef6.4) [concrete = constants.%facet_value.ed4]
// CHECK:STDOUT:   %.loc24_39.2: %facet_type.ece = converted constants.%Y2, %facet_value.loc24_39.2 [concrete = constants.%facet_value.ed4]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.1: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.ed4) [concrete = constants.%SelfDestruct.specific_fn.51d]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_39.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%tuple.elem1)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %tuple.elem0: ref %Y1 = tuple_access %self.param, element0
// CHECK:STDOUT:   %facet_value.loc24_39.3: %facet_type.ece = facet_value constants.%Y1, (constants.%DestroyImpl.impl_witness.6c4, constants.%custom_witness.b80ef6.3) [concrete = constants.%facet_value.cf6]
// CHECK:STDOUT:   %.loc24_39.3: %facet_type.ece = converted constants.%Y1, %facet_value.loc24_39.3 [concrete = constants.%facet_value.cf6]
// CHECK:STDOUT:   %facet_value.loc24_39.4: %facet_type.ece = facet_value constants.%Y1, (constants.%DestroyImpl.impl_witness.6c4, constants.%custom_witness.b80ef6.3) [concrete = constants.%facet_value.cf6]
// CHECK:STDOUT:   %.loc24_39.4: %facet_type.ece = converted constants.%Y1, %facet_value.loc24_39.4 [concrete = constants.%facet_value.cf6]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.cf6) [concrete = constants.%SelfDestruct.specific_fn.ead]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_39.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%tuple.elem0)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc24_39.6(%self.param: ref %Z) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc24_39.7(%self.param: ref %tuple.type.d40) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %tuple.elem2: ref %Z = tuple_access %self.param, element2
// CHECK:STDOUT:   %facet_value.loc24_39.1: %facet_type.ece = facet_value constants.%Z, (constants.%DestroyImpl.impl_witness.72a, constants.%custom_witness.b80ef6.6) [concrete = constants.%facet_value.469]
// CHECK:STDOUT:   %.loc24_39.1: %facet_type.ece = converted constants.%Z, %facet_value.loc24_39.1 [concrete = constants.%facet_value.469]
// CHECK:STDOUT:   %facet_value.loc24_39.2: %facet_type.ece = facet_value constants.%Z, (constants.%DestroyImpl.impl_witness.72a, constants.%custom_witness.b80ef6.6) [concrete = constants.%facet_value.469]
// CHECK:STDOUT:   %.loc24_39.2: %facet_type.ece = converted constants.%Z, %facet_value.loc24_39.2 [concrete = constants.%facet_value.469]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.1: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.469) [concrete = constants.%SelfDestruct.specific_fn.c54]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_39.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%tuple.elem2)
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %tuple.elem1: ref %tuple.type.59f = tuple_access %self.param, element1
// CHECK:STDOUT:   %facet_value.loc24_39.3: %facet_type.ece = facet_value constants.%tuple.type.59f, (constants.%DestroyImpl.impl_witness.937, constants.%custom_witness.b80ef6.5) [concrete = constants.%facet_value.7f4]
// CHECK:STDOUT:   %.loc24_39.3: %facet_type.ece = converted constants.%tuple.type.59f, %facet_value.loc24_39.3 [concrete = constants.%facet_value.7f4]
// CHECK:STDOUT:   %facet_value.loc24_39.4: %facet_type.ece = facet_value constants.%tuple.type.59f, (constants.%DestroyImpl.impl_witness.937, constants.%custom_witness.b80ef6.5) [concrete = constants.%facet_value.7f4]
// CHECK:STDOUT:   %.loc24_39.4: %facet_type.ece = converted constants.%tuple.type.59f, %facet_value.loc24_39.4 [concrete = constants.%facet_value.7f4]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.7f4) [concrete = constants.%SelfDestruct.specific_fn.868]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_39.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%tuple.elem1)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %tuple.elem0: ref %X = tuple_access %self.param, element0
// CHECK:STDOUT:   %facet_value.loc24_39.5: %facet_type.ece = facet_value constants.%X, (constants.%DestroyImpl.impl_witness.678, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.9d2]
// CHECK:STDOUT:   %.loc24_39.5: %facet_type.ece = converted constants.%X, %facet_value.loc24_39.5 [concrete = constants.%facet_value.9d2]
// CHECK:STDOUT:   %facet_value.loc24_39.6: %facet_type.ece = facet_value constants.%X, (constants.%DestroyImpl.impl_witness.678, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.9d2]
// CHECK:STDOUT:   %.loc24_39.6: %facet_type.ece = converted constants.%X, %facet_value.loc24_39.6 [concrete = constants.%facet_value.9d2]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.3: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.9d2) [concrete = constants.%SelfDestruct.specific_fn.e0d]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_39.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%tuple.elem0)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallGeneric(%T.loc27_25.2: %facet_type.572) {
// CHECK:STDOUT:   %T.patt.loc27_25.2: %pattern_type.03d = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc27_25.2 (constants.%T.patt.4c0)]
// CHECK:STDOUT:   %T.loc27_25.1: %facet_type.572 = symbolic_binding T, 0 [symbolic = %T.loc27_25.1 (constants.%T.80e)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc28_4.2: type = facet_access_type %T.loc27_25.1 [symbolic = %T.as_type.loc28_4.2 (constants.%T.as_type.8e3)]
// CHECK:STDOUT:   %A.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc27_25.1, @A, @A(constants.%X) [symbolic = %A.lookup_impl_witness (constants.%A.lookup_impl_witness)]
// CHECK:STDOUT:   %A.facet: %A.type.df0 = facet_value %T.as_type.loc28_4.2, (%A.lookup_impl_witness) [symbolic = %A.facet (constants.%A.facet.5d2)]
// CHECK:STDOUT:   %A.WithSelf.F.type: type = fn_type @A.WithSelf.F, @A.WithSelf(constants.%X, %A.facet) [symbolic = %A.WithSelf.F.type (constants.%A.WithSelf.F.type.9bb)]
// CHECK:STDOUT:   %.loc28_4.3: type = fn_type_with_self_type %A.WithSelf.F.type, %A.facet [symbolic = %.loc28_4.3 (constants.%.ad0)]
// CHECK:STDOUT:   %impl.elem0.loc28_4.2: @CallGeneric.%.loc28_4.3 (%.ad0) = impl_witness_access %A.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc28_4.2 (constants.%impl.elem0.327)]
// CHECK:STDOUT:   %specific_impl_fn.loc28_4.2: <specific function> = specific_impl_function %impl.elem0.loc28_4.2, @A.WithSelf.F(constants.%X, %A.facet, constants.%Z) [symbolic = %specific_impl_fn.loc28_4.2 (constants.%specific_impl_fn.b41)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (constants.%X, %T.as_type.loc28_4.2, constants.%Z) [symbolic = %tuple.type (constants.%tuple.type.b25)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %tuple.type [symbolic = %require_complete (constants.%require_complete.ac9)]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness: <witness> = lookup_impl_witness %tuple.type, @SubobjectDestroy [symbolic = %SubobjectDestroy.lookup_impl_witness (constants.%SubobjectDestroy.lookup_impl_witness.30d)]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %tuple.type, (%SubobjectDestroy.lookup_impl_witness) [symbolic = %SubobjectDestroy.facet (constants.%SubobjectDestroy.facet.0af)]
// CHECK:STDOUT:   %.loc28_12.5: require_specific_def_type = require_specific_def @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet) [symbolic = %.loc28_12.5 (constants.%.e37)]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness: <witness> = lookup_impl_witness %tuple.type, @DestroyImpl [symbolic = %DestroyImpl.lookup_impl_witness (constants.%DestroyImpl.lookup_impl_witness.dff)]
// CHECK:STDOUT:   %facet_value.loc28_12.3: %facet_type.ece = facet_value %tuple.type, (%DestroyImpl.lookup_impl_witness, %SubobjectDestroy.lookup_impl_witness) [symbolic = %facet_value.loc28_12.3 (constants.%facet_value.9e6)]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.3: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(%facet_value.loc28_12.3) [symbolic = %SelfDestruct.specific_fn.3 (constants.%SelfDestruct.specific_fn.4cf)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref: %facet_type.572 = name_ref T, %T.loc27_25.2 [symbolic = %T.loc27_25.1 (constants.%T.80e)]
// CHECK:STDOUT:     %T.as_type.loc28_4.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc28_4.2 (constants.%T.as_type.8e3)]
// CHECK:STDOUT:     %.loc28_4.1: type = converted %T.ref, %T.as_type.loc28_4.1 [symbolic = %T.as_type.loc28_4.2 (constants.%T.as_type.8e3)]
// CHECK:STDOUT:     %.loc28_4.2: %A.assoc_type.a54 = specific_constant @A.WithSelf.%assoc0.loc6_46.1, @A.WithSelf(constants.%X, constants.%T.80e) [concrete = constants.%assoc0.987]
// CHECK:STDOUT:     %F.ref: %A.assoc_type.a54 = name_ref F, %.loc28_4.2 [concrete = constants.%assoc0.987]
// CHECK:STDOUT:     %impl.elem0.loc28_4.1: @CallGeneric.%.loc28_4.3 (%.ad0) = impl_witness_access constants.%A.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc28_4.2 (constants.%impl.elem0.327)]
// CHECK:STDOUT:     %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z]
// CHECK:STDOUT:     %.loc28_11.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %specific_impl_fn.loc28_4.1: <specific function> = specific_impl_function %impl.elem0.loc28_4.1, @A.WithSelf.F(constants.%X, constants.%A.facet.5d2, constants.%Z) [symbolic = %specific_impl_fn.loc28_4.2 (constants.%specific_impl_fn.b41)]
// CHECK:STDOUT:     %.loc28_12.1: ref @CallGeneric.%tuple.type (%tuple.type.b25) = temporary_storage
// CHECK:STDOUT:     %.loc28_11.2: ref %Z = temporary_storage
// CHECK:STDOUT:     %.loc28_11.3: init %Z to %.loc28_11.2 = class_init () [concrete = constants.%Z.val]
// CHECK:STDOUT:     %.loc28_11.4: init %Z = converted %.loc28_11.1, %.loc28_11.3 [concrete = constants.%Z.val]
// CHECK:STDOUT:     %.loc28_11.5: ref %Z = temporary %.loc28_11.2, %.loc28_11.4 [concrete = constants.%.7dc]
// CHECK:STDOUT:     %.loc28_11.6: %Z = acquire_value %.loc28_11.5 [concrete = constants.%Z.val]
// CHECK:STDOUT:     %A.WithSelf.F.call: init @CallGeneric.%tuple.type (%tuple.type.b25) to %.loc28_12.1 = call %specific_impl_fn.loc28_4.1(%.loc28_11.6)
// CHECK:STDOUT:     %.loc28_12.2: ref @CallGeneric.%tuple.type (%tuple.type.b25) = temporary %.loc28_12.1, %A.WithSelf.F.call
// CHECK:STDOUT:     %facet_value.loc28_12.1: %facet_type.ece = facet_value constants.%tuple.type.b25, (constants.%DestroyImpl.lookup_impl_witness.dff, constants.%SubobjectDestroy.lookup_impl_witness.30d) [symbolic = %facet_value.loc28_12.3 (constants.%facet_value.9e6)]
// CHECK:STDOUT:     %.loc28_12.3: %facet_type.ece = converted constants.%tuple.type.b25, %facet_value.loc28_12.1 [symbolic = %facet_value.loc28_12.3 (constants.%facet_value.9e6)]
// CHECK:STDOUT:     %facet_value.loc28_12.2: %facet_type.ece = facet_value constants.%tuple.type.b25, (constants.%DestroyImpl.lookup_impl_witness.dff, constants.%SubobjectDestroy.lookup_impl_witness.30d) [symbolic = %facet_value.loc28_12.3 (constants.%facet_value.9e6)]
// CHECK:STDOUT:     %.loc28_12.4: %facet_type.ece = converted constants.%tuple.type.b25, %facet_value.loc28_12.2 [symbolic = %facet_value.loc28_12.3 (constants.%facet_value.9e6)]
// CHECK:STDOUT:     %SelfDestruct.specific_fn.1: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.9e6) [symbolic = %SelfDestruct.specific_fn.3 (constants.%SelfDestruct.specific_fn.4cf)]
// CHECK:STDOUT:     %SelfDestruct.call.loc28_12: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc28_12.2)
// CHECK:STDOUT:     %facet_value.loc28_11.1: %facet_type.ece = facet_value constants.%Z, (constants.%DestroyImpl.impl_witness.72a, constants.%custom_witness.b80ef6.6) [concrete = constants.%facet_value.469]
// CHECK:STDOUT:     %.loc28_11.7: %facet_type.ece = converted constants.%Z, %facet_value.loc28_11.1 [concrete = constants.%facet_value.469]
// CHECK:STDOUT:     %facet_value.loc28_11.2: %facet_type.ece = facet_value constants.%Z, (constants.%DestroyImpl.impl_witness.72a, constants.%custom_witness.b80ef6.6) [concrete = constants.%facet_value.469]
// CHECK:STDOUT:     %.loc28_11.8: %facet_type.ece = converted constants.%Z, %facet_value.loc28_11.2 [concrete = constants.%facet_value.469]
// CHECK:STDOUT:     %SelfDestruct.specific_fn.2: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.469) [concrete = constants.%SelfDestruct.specific_fn.c54]
// CHECK:STDOUT:     %SelfDestruct.call.loc28_11: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc28_11.5)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallIndirect() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CallGeneric.ref: %CallGeneric.type = name_ref CallGeneric, file.%CallGeneric.decl [concrete = constants.%CallGeneric]
// CHECK:STDOUT:   %Y1.ref: type = name_ref Y1, file.%Y1.decl [concrete = constants.%Y1]
// CHECK:STDOUT:   %Y2.ref: type = name_ref Y2, file.%Y2.decl [concrete = constants.%Y2]
// CHECK:STDOUT:   %.loc33_22: %tuple.type.693 = tuple_literal (%Y1.ref, %Y2.ref) [concrete = constants.%tuple.71f]
// CHECK:STDOUT:   %.loc33_27: type = type_literal type [concrete = type]
// CHECK:STDOUT:   %.loc33_24: type = converted %.loc33_22, constants.%tuple.type.59f [concrete = constants.%tuple.type.59f]
// CHECK:STDOUT:   %facet_value: %facet_type.572 = facet_value %.loc33_24, (constants.%A.impl_witness.0e0, constants.%DestroyImpl.impl_witness.937, constants.%custom_witness.b80ef6.5) [concrete = constants.%facet_value.e23]
// CHECK:STDOUT:   %.loc33_31: %facet_type.572 = converted %.loc33_24, %facet_value [concrete = constants.%facet_value.e23]
// CHECK:STDOUT:   %CallGeneric.specific_fn: <specific function> = specific_function %CallGeneric.ref, @CallGeneric(constants.%facet_value.e23) [concrete = constants.%CallGeneric.specific_fn]
// CHECK:STDOUT:   %CallGeneric.call: init %empty_tuple.type = call %CallGeneric.specific_fn()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%T.67d) {
// CHECK:STDOUT:   %T.patt.loc5_14.2 => constants.%T.patt.3a0
// CHECK:STDOUT:   %T.loc5_14.1 => constants.%T.67d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%T.67d, constants.%Self.49a) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf.F(constants.%T.67d, constants.%Self.49a, constants.%U.4e7) {
// CHECK:STDOUT:   %U.patt.loc6_17.2 => constants.%U.patt.5c4
// CHECK:STDOUT:   %U.loc6_17.1 => constants.%U.4e7
// CHECK:STDOUT:   %pattern_type.loc6_26 => constants.%pattern_type.62e
// CHECK:STDOUT:   %u.param_patt.loc6_26.2 => constants.%u.param_patt.6ef
// CHECK:STDOUT:   %u.patt.loc6_26.2 => constants.%u.patt.9b1
// CHECK:STDOUT:   %T => constants.%T.67d
// CHECK:STDOUT:   %A.type => constants.%A.type.acd
// CHECK:STDOUT:   %Self => constants.%Self.49a
// CHECK:STDOUT:   %tuple.type.loc6_45.1 => constants.%tuple.type.a31
// CHECK:STDOUT:   %tuple => constants.%tuple.62b
// CHECK:STDOUT:   %Self.as_type.loc6_45.1 => constants.%Self.as_type.763
// CHECK:STDOUT:   %tuple.type.loc6_45.2 => constants.%tuple.type.d73
// CHECK:STDOUT:   %.loc6_45.1 => constants.%.bc3
// CHECK:STDOUT:   %pattern_type.loc6_45 => constants.%pattern_type.d6c
// CHECK:STDOUT:   %return.param_patt.loc6_45.2 => constants.%return.param_patt.52a
// CHECK:STDOUT:   %return.patt.loc6_31.2 => constants.%return.patt.808
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%W) {
// CHECK:STDOUT:   %T.patt.loc5_14.2 => constants.%T.patt.3a0
// CHECK:STDOUT:   %T.loc5_14.1 => constants.%W
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.type => constants.%A.type.3ad
// CHECK:STDOUT:   %Self.loc5_22.2 => constants.%Self.0f8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @tuple.type.as.A.impl(constants.%V1, constants.%V2, constants.%W) {
// CHECK:STDOUT:   %V1.patt.loc14_16.2 => constants.%V1.patt
// CHECK:STDOUT:   %V1.loc14_16.1 => constants.%V1
// CHECK:STDOUT:   %V2.patt.loc14_26.2 => constants.%V2.patt
// CHECK:STDOUT:   %V2.loc14_26.1 => constants.%V2
// CHECK:STDOUT:   %W.patt.loc14_35.2 => constants.%W.patt
// CHECK:STDOUT:   %W.loc14_35.1 => constants.%W
// CHECK:STDOUT:   %tuple => constants.%tuple.fa2
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.a5e
// CHECK:STDOUT:   %A.type.loc14_58.1 => constants.%A.type.3ad
// CHECK:STDOUT:   %.loc14_60 => constants.%.b95
// CHECK:STDOUT:   %A.impl_witness.loc14_60.2 => constants.%A.impl_witness.07b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.7b2
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type => constants.%tuple.type.as.A.impl.F.type.44a
// CHECK:STDOUT:   %tuple.type.as.A.impl.F => constants.%tuple.type.as.A.impl.F.742
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%W, constants.%Self.49a) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%W
// CHECK:STDOUT:   %A.type => constants.%A.type.3ad
// CHECK:STDOUT:   %Self => constants.%Self.49a
// CHECK:STDOUT:   %A.WithSelf.F.type => constants.%A.WithSelf.F.type.64e
// CHECK:STDOUT:   %A.WithSelf.F => constants.%A.WithSelf.F.1fb
// CHECK:STDOUT:   %A.assoc_type => constants.%A.assoc_type.250
// CHECK:STDOUT:   %assoc0.loc6_46.2 => constants.%assoc0.6de
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @tuple.type.as.A.impl.F(constants.%V1, constants.%V2, constants.%W, constants.%U.ce4) {
// CHECK:STDOUT:   %U.patt.loc17_17.2 => constants.%U.patt.755
// CHECK:STDOUT:   %U.loc17_17.1 => constants.%U.ce4
// CHECK:STDOUT:   %pattern_type.loc17_26 => constants.%pattern_type.e4a
// CHECK:STDOUT:   %u.param_patt.loc17_26.2 => constants.%u.param_patt.afe
// CHECK:STDOUT:   %u.patt.loc17_26.2 => constants.%u.patt.60c
// CHECK:STDOUT:   %W => constants.%W
// CHECK:STDOUT:   %V1 => constants.%V1
// CHECK:STDOUT:   %V2 => constants.%V2
// CHECK:STDOUT:   %tuple.loc17_45 => constants.%tuple.fa2
// CHECK:STDOUT:   %tuple.loc17_49 => constants.%tuple.cd3
// CHECK:STDOUT:   %tuple.type.loc17_49.1 => constants.%tuple.type.a5e
// CHECK:STDOUT:   %tuple.type.loc17_49.2 => constants.%tuple.type.897
// CHECK:STDOUT:   %.loc17_49.2 => constants.%.f8e
// CHECK:STDOUT:   %pattern_type.loc17_49 => constants.%pattern_type.5cc
// CHECK:STDOUT:   %return.param_patt.loc17_49.2 => constants.%return.param_patt.b1c
// CHECK:STDOUT:   %return.patt.loc17_31.2 => constants.%return.patt.bad9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc17_26 => constants.%require_complete.5a4
// CHECK:STDOUT:   %require_complete.loc17_31 => constants.%require_complete.da0
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type => constants.%tuple.type.as.A.impl.F.type.44a
// CHECK:STDOUT:   %tuple.type.as.A.impl.F => constants.%tuple.type.as.A.impl.F.742
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.specific_fn.loc18_12.2 => constants.%tuple.type.as.A.impl.F.specific_fn.37b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%W, constants.%A.facet.de4) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%W
// CHECK:STDOUT:   %A.type => constants.%A.type.3ad
// CHECK:STDOUT:   %Self => constants.%A.facet.de4
// CHECK:STDOUT:   %A.WithSelf.F.type => constants.%A.WithSelf.F.type.3e0
// CHECK:STDOUT:   %A.WithSelf.F => constants.%A.WithSelf.F.43a
// CHECK:STDOUT:   %A.assoc_type => constants.%A.assoc_type.250
// CHECK:STDOUT:   %assoc0.loc6_46.2 => constants.%assoc0.6de
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf.F(constants.%W, constants.%A.facet.de4, constants.%U.ce4) {
// CHECK:STDOUT:   %U.patt.loc6_17.2 => constants.%U.patt.5c4
// CHECK:STDOUT:   %U.loc6_17.1 => constants.%U.ce4
// CHECK:STDOUT:   %pattern_type.loc6_26 => constants.%pattern_type.e4a
// CHECK:STDOUT:   %u.param_patt.loc6_26.2 => constants.%u.param_patt.afe
// CHECK:STDOUT:   %u.patt.loc6_26.2 => constants.%u.patt.60c
// CHECK:STDOUT:   %T => constants.%W
// CHECK:STDOUT:   %A.type => constants.%A.type.3ad
// CHECK:STDOUT:   %Self => constants.%A.facet.de4
// CHECK:STDOUT:   %tuple.type.loc6_45.1 => constants.%tuple.type.038
// CHECK:STDOUT:   %tuple => constants.%tuple.c3f
// CHECK:STDOUT:   %Self.as_type.loc6_45.1 => constants.%tuple.type.a5e
// CHECK:STDOUT:   %tuple.type.loc6_45.2 => constants.%tuple.type.897
// CHECK:STDOUT:   %.loc6_45.1 => constants.%.f8e
// CHECK:STDOUT:   %pattern_type.loc6_45 => constants.%pattern_type.5cc
// CHECK:STDOUT:   %return.param_patt.loc6_45.2 => constants.%return.param_patt.b1c
// CHECK:STDOUT:   %return.patt.loc6_31.2 => constants.%return.patt.bad9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%X) {
// CHECK:STDOUT:   %T.patt.loc5_14.2 => constants.%T.patt.3a0
// CHECK:STDOUT:   %T.loc5_14.1 => constants.%X
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.type => constants.%A.type.df0
// CHECK:STDOUT:   %Self.loc5_22.2 => constants.%Self.891
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @tuple.type.as.A.impl(constants.%Y1, constants.%Y2, constants.%X) {
// CHECK:STDOUT:   %V1.patt.loc14_16.2 => constants.%V1.patt
// CHECK:STDOUT:   %V1.loc14_16.1 => constants.%Y1
// CHECK:STDOUT:   %V2.patt.loc14_26.2 => constants.%V2.patt
// CHECK:STDOUT:   %V2.loc14_26.1 => constants.%Y2
// CHECK:STDOUT:   %W.patt.loc14_35.2 => constants.%W.patt
// CHECK:STDOUT:   %W.loc14_35.1 => constants.%X
// CHECK:STDOUT:   %tuple => constants.%tuple.71f
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.59f
// CHECK:STDOUT:   %A.type.loc14_58.1 => constants.%A.type.df0
// CHECK:STDOUT:   %.loc14_60 => constants.%.de6
// CHECK:STDOUT:   %A.impl_witness.loc14_60.2 => constants.%A.impl_witness.0e0
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.e33
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type => constants.%tuple.type.as.A.impl.F.type.745
// CHECK:STDOUT:   %tuple.type.as.A.impl.F => constants.%tuple.type.as.A.impl.F.749
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%X, constants.%Self.49a) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.df0
// CHECK:STDOUT:   %Self => constants.%Self.49a
// CHECK:STDOUT:   %A.WithSelf.F.type => constants.%A.WithSelf.F.type.ad4
// CHECK:STDOUT:   %A.WithSelf.F => constants.%A.WithSelf.F.1e2
// CHECK:STDOUT:   %A.assoc_type => constants.%A.assoc_type.a54
// CHECK:STDOUT:   %assoc0.loc6_46.2 => constants.%assoc0.987
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%X, constants.%A.facet.89e) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.df0
// CHECK:STDOUT:   %Self => constants.%A.facet.89e
// CHECK:STDOUT:   %A.WithSelf.F.type => constants.%A.WithSelf.F.type.d74
// CHECK:STDOUT:   %A.WithSelf.F => constants.%A.WithSelf.F.cd4
// CHECK:STDOUT:   %A.assoc_type => constants.%A.assoc_type.a54
// CHECK:STDOUT:   %assoc0.loc6_46.2 => constants.%assoc0.987
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @tuple.type.as.A.impl.F(constants.%Y1, constants.%Y2, constants.%X, constants.%Z) {
// CHECK:STDOUT:   %U.patt.loc17_17.2 => constants.%U.patt.755
// CHECK:STDOUT:   %U.loc17_17.1 => constants.%Z
// CHECK:STDOUT:   %pattern_type.loc17_26 => constants.%pattern_type.34e
// CHECK:STDOUT:   %u.param_patt.loc17_26.2 => constants.%u.param_patt.a3e
// CHECK:STDOUT:   %u.patt.loc17_26.2 => constants.%u.patt.5d3
// CHECK:STDOUT:   %W => constants.%X
// CHECK:STDOUT:   %V1 => constants.%Y1
// CHECK:STDOUT:   %V2 => constants.%Y2
// CHECK:STDOUT:   %tuple.loc17_45 => constants.%tuple.71f
// CHECK:STDOUT:   %tuple.loc17_49 => constants.%tuple.415
// CHECK:STDOUT:   %tuple.type.loc17_49.1 => constants.%tuple.type.59f
// CHECK:STDOUT:   %tuple.type.loc17_49.2 => constants.%tuple.type.d40
// CHECK:STDOUT:   %.loc17_49.2 => constants.%.1a9
// CHECK:STDOUT:   %pattern_type.loc17_49 => constants.%pattern_type.6a8
// CHECK:STDOUT:   %return.param_patt.loc17_49.2 => constants.%return.param_patt.fb7
// CHECK:STDOUT:   %return.patt.loc17_31.2 => constants.%return.patt.badc
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc17_26 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc17_31 => constants.%complete_type.8f6
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type => constants.%tuple.type.as.A.impl.F.type.745
// CHECK:STDOUT:   %tuple.type.as.A.impl.F => constants.%tuple.type.as.A.impl.F.749
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.specific_fn.loc18_12.2 => constants.%tuple.type.as.A.impl.F.specific_fn.981
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGeneric(constants.%T.80e) {
// CHECK:STDOUT:   %T.patt.loc27_25.2 => constants.%T.patt.4c0
// CHECK:STDOUT:   %T.loc27_25.1 => constants.%T.80e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%X, constants.%T.80e) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.df0
// CHECK:STDOUT:   %Self => constants.%T.80e
// CHECK:STDOUT:   %A.WithSelf.F.type => constants.%A.WithSelf.F.type.298
// CHECK:STDOUT:   %A.WithSelf.F => constants.%A.WithSelf.F.d25
// CHECK:STDOUT:   %A.assoc_type => constants.%A.assoc_type.a54
// CHECK:STDOUT:   %assoc0.loc6_46.2 => constants.%assoc0.987
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%X, constants.%A.facet.5d2) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.df0
// CHECK:STDOUT:   %Self => constants.%A.facet.5d2
// CHECK:STDOUT:   %A.WithSelf.F.type => constants.%A.WithSelf.F.type.9bb
// CHECK:STDOUT:   %A.WithSelf.F => constants.%A.WithSelf.F.8c8
// CHECK:STDOUT:   %A.assoc_type => constants.%A.assoc_type.a54
// CHECK:STDOUT:   %assoc0.loc6_46.2 => constants.%assoc0.987
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf.F(constants.%X, constants.%A.facet.5d2, constants.%Z) {
// CHECK:STDOUT:   %U.patt.loc6_17.2 => constants.%U.patt.5c4
// CHECK:STDOUT:   %U.loc6_17.1 => constants.%Z
// CHECK:STDOUT:   %pattern_type.loc6_26 => constants.%pattern_type.34e
// CHECK:STDOUT:   %u.param_patt.loc6_26.2 => constants.%u.param_patt.a3e
// CHECK:STDOUT:   %u.patt.loc6_26.2 => constants.%u.patt.5d3
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.df0
// CHECK:STDOUT:   %Self => constants.%A.facet.5d2
// CHECK:STDOUT:   %tuple.type.loc6_45.1 => constants.%tuple.type.202
// CHECK:STDOUT:   %tuple => constants.%tuple.612
// CHECK:STDOUT:   %Self.as_type.loc6_45.1 => constants.%T.as_type.8e3
// CHECK:STDOUT:   %tuple.type.loc6_45.2 => constants.%tuple.type.b25
// CHECK:STDOUT:   %.loc6_45.1 => constants.%.55e
// CHECK:STDOUT:   %pattern_type.loc6_45 => constants.%pattern_type.25a
// CHECK:STDOUT:   %return.param_patt.loc6_45.2 => constants.%return.param_patt.08d
// CHECK:STDOUT:   %return.patt.loc6_31.2 => constants.%return.patt.e9a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGeneric(constants.%facet_value.e23) {
// CHECK:STDOUT:   %T.patt.loc27_25.2 => constants.%T.patt.4c0
// CHECK:STDOUT:   %T.loc27_25.1 => constants.%facet_value.e23
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc28_4.2 => constants.%tuple.type.59f
// CHECK:STDOUT:   %A.lookup_impl_witness => constants.%A.impl_witness.0e0
// CHECK:STDOUT:   %A.facet => constants.%A.facet.89e
// CHECK:STDOUT:   %A.WithSelf.F.type => constants.%A.WithSelf.F.type.d74
// CHECK:STDOUT:   %.loc28_4.3 => constants.%.744
// CHECK:STDOUT:   %impl.elem0.loc28_4.2 => constants.%tuple.type.as.A.impl.F.749
// CHECK:STDOUT:   %specific_impl_fn.loc28_4.2 => constants.%tuple.type.as.A.impl.F.specific_fn.981
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.d40
// CHECK:STDOUT:   %require_complete => constants.%complete_type.8f6
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness => constants.%custom_witness.b80ef6.7
// CHECK:STDOUT:   %SubobjectDestroy.facet => constants.%SubobjectDestroy.facet.392
// CHECK:STDOUT:   %.loc28_12.5 => constants.%.422
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness => constants.%DestroyImpl.impl_witness.f58
// CHECK:STDOUT:   %facet_value.loc28_12.3 => constants.%facet_value.409
// CHECK:STDOUT:   %SelfDestruct.specific_fn.3 => constants.%SelfDestruct.specific_fn.faa
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf.F(constants.%X, constants.%A.facet.89e, constants.%Z) {
// CHECK:STDOUT:   %U.patt.loc6_17.2 => constants.%U.patt.5c4
// CHECK:STDOUT:   %U.loc6_17.1 => constants.%Z
// CHECK:STDOUT:   %pattern_type.loc6_26 => constants.%pattern_type.34e
// CHECK:STDOUT:   %u.param_patt.loc6_26.2 => constants.%u.param_patt.a3e
// CHECK:STDOUT:   %u.patt.loc6_26.2 => constants.%u.patt.5d3
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.df0
// CHECK:STDOUT:   %Self => constants.%A.facet.89e
// CHECK:STDOUT:   %tuple.type.loc6_45.1 => constants.%tuple.type.202
// CHECK:STDOUT:   %tuple => constants.%tuple.b5b
// CHECK:STDOUT:   %Self.as_type.loc6_45.1 => constants.%tuple.type.59f
// CHECK:STDOUT:   %tuple.type.loc6_45.2 => constants.%tuple.type.d40
// CHECK:STDOUT:   %.loc6_45.1 => constants.%.1a9
// CHECK:STDOUT:   %pattern_type.loc6_45 => constants.%pattern_type.6a8
// CHECK:STDOUT:   %return.param_patt.loc6_45.2 => constants.%return.param_patt.fb7
// CHECK:STDOUT:   %return.patt.loc6_31.2 => constants.%return.patt.badc
// CHECK:STDOUT: }
// CHECK:STDOUT:

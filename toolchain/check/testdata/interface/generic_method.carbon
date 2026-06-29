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
interface A(T:! type) {
  fn F(U:! type, u: U*) -> (T, Self, U*);
}

class X {}
class Y {}
class Z {}

impl Y as A(X) {
  // Here, U has index 0. The specific used for A.F needs to renumber its U from
  // index 2 to index 0.
  fn F(U:! type, u: U*) -> (X, Y, U*) {
    return ({}, {}, u);
  }
}

fn Call() {
  var u: Z;
  (Y as A(X)).F(Z, &u);
}

fn CallGeneric(T:! A(X) & Core.Destroy) {
  var u: Z;
  T.F(Z, &u);
}

fn CallIndirect() {
  CallGeneric(Y);
}

// --- generic_method_more_params.carbon

library "[[@TEST_NAME]]";

// T has index 0, Self has index 1, U has index 2.
interface A(T:! type) {
  fn F(U:! type, u: U) -> (T, Self, U);
}

class X {}
class Y1 {}
class Y2 {}
class Z {}

impl forall [V1:! type, V2:! type, W:! type] (V1, V2) as A(W) {
  // Here, U has index 3. The specific used for A.F needs to renumber its U from
  // index 2 to index 3.
  fn F(U:! type, u: U) -> (W, (V1, V2), U) {
    return F(U, u);
  }
}

fn Call() {
  // TODO: The `as type` here should not be necessary.
  (((Y1, Y2) as type) as A(X)).F(Z, {});
}

fn CallGeneric(T:! A(X) & Core.Destroy) {
  T.F(Z, {});
}

fn CallIndirect() {
  // TODO: The `as type` here should not be necessary.
  CallGeneric((Y1, Y2) as type);
}


// CHECK:STDOUT: --- generic_method_fewer_params.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt.d47: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %A.type.55a: type = generic_interface_type @A [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %A.generic: %A.type.55a = struct_value () [concrete]
// CHECK:STDOUT:   %A.type.acd: type = facet_type <@A, @A(%T.67d)> [symbolic]
// CHECK:STDOUT:   %Self.49a: %A.type.acd = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %U.patt.b22: %pattern_type.98f = symbolic_binding_pattern U, 2 [symbolic]
// CHECK:STDOUT:   %U.4e7: type = symbolic_binding U, 2 [symbolic]
// CHECK:STDOUT:   %ptr.8f6: type = ptr_type %U.4e7 [symbolic]
// CHECK:STDOUT:   %pattern_type.986: type = pattern_type %ptr.8f6 [symbolic]
// CHECK:STDOUT:   %u.param_patt.a01: %pattern_type.986 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %u.patt.c78: %pattern_type.986 = at_binding_pattern u, %u.param_patt.a01 [symbolic]
// CHECK:STDOUT:   %tuple.type.76f: type = tuple_type (type, %A.type.acd, type) [symbolic]
// CHECK:STDOUT:   %tuple.0f4: %tuple.type.76f = tuple_value (%T.67d, %Self.49a, %ptr.8f6) [symbolic]
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
// CHECK:STDOUT:   %A.impl_witness: <witness> = impl_witness @Y.as.A.impl.%A.impl_witness_table [concrete]
// CHECK:STDOUT:   %Self.891: %A.type.df0 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %A.WithSelf.F.type.ad4: type = fn_type @A.WithSelf.F, @A.WithSelf(%X, %Self.49a) [symbolic]
// CHECK:STDOUT:   %A.WithSelf.F.1e2: %A.WithSelf.F.type.ad4 = struct_value () [symbolic]
// CHECK:STDOUT:   %A.assoc_type.a54: type = assoc_entity_type @A, @A(%X) [concrete]
// CHECK:STDOUT:   %assoc0.987: %A.assoc_type.a54 = assoc_entity element0, @A.WithSelf.%A.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %U.patt.d47: %pattern_type.98f = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %U.67d: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %ptr.e8f8f9.1: type = ptr_type %U.67d [symbolic]
// CHECK:STDOUT:   %pattern_type.4f4b84.1: type = pattern_type %ptr.e8f8f9.1 [symbolic]
// CHECK:STDOUT:   %u.param_patt.58b: %pattern_type.4f4b84.1 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %u.patt.542: %pattern_type.4f4b84.1 = at_binding_pattern u, %u.param_patt.58b [symbolic]
// CHECK:STDOUT:   %tuple.type.ff9: type = tuple_type (type, type, type) [concrete]
// CHECK:STDOUT:   %tuple.f57: %tuple.type.ff9 = tuple_value (%X, %Y, %ptr.e8f8f9.1) [symbolic]
// CHECK:STDOUT:   %tuple.type.316: type = tuple_type (%X, %Y, %ptr.e8f8f9.1) [symbolic]
// CHECK:STDOUT:   %.3c1: Core.Form = init_form %tuple.type.316 [symbolic]
// CHECK:STDOUT:   %pattern_type.178: type = pattern_type %tuple.type.316 [symbolic]
// CHECK:STDOUT:   %return.param_patt.b96: %pattern_type.178 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.f2f: %pattern_type.178 = return_slot_pattern %return.param_patt.b96, %tuple.type.316 [symbolic]
// CHECK:STDOUT:   %Y.as.A.impl.F.type: type = fn_type @Y.as.A.impl.F [concrete]
// CHECK:STDOUT:   %Y.as.A.impl.F: %Y.as.A.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %A.facet.c65: %A.type.df0 = facet_value %Y, (%A.impl_witness) [concrete]
// CHECK:STDOUT:   %A.WithSelf.F.type.22c: type = fn_type @A.WithSelf.F, @A.WithSelf(%X, %A.facet.c65) [concrete]
// CHECK:STDOUT:   %A.WithSelf.F.06a: %A.WithSelf.F.type.22c = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.966: type = tuple_type (type, %A.type.df0, type) [concrete]
// CHECK:STDOUT:   %tuple.7f7: %tuple.type.966 = tuple_value (%X, %A.facet.c65, %ptr.e8f8f9.1) [symbolic]
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
// CHECK:STDOUT:   %Copy.facet.302b: %Copy.type = facet_value %ptr.e8f8f9.1, (%Copy.lookup_impl_witness.1da) [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.884: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.302b) [symbolic]
// CHECK:STDOUT:   %.be5: type = fn_type_with_self_type %Copy.WithSelf.Op.type.884, %Copy.facet.302b [symbolic]
// CHECK:STDOUT:   %impl.elem0.484: %.be5 = impl_witness_access %Copy.lookup_impl_witness.1da, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.a76: <specific function> = specific_impl_function %impl.elem0.484, @Copy.WithSelf.Op(%Copy.facet.302b) [symbolic]
// CHECK:STDOUT:   %Call.type: type = fn_type @Call [concrete]
// CHECK:STDOUT:   %Call: %Call.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.34e: type = pattern_type %Z [concrete]
// CHECK:STDOUT:   %u.patt.352: %pattern_type.34e = ref_binding_pattern u [concrete]
// CHECK:STDOUT:   %u.var_patt: %pattern_type.34e = var_pattern %u.patt.352 [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.type.150: type = fn_type @T.as.DefaultOrUnformed.impl.Op, @T.as.DefaultOrUnformed.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.53a: %T.as.DefaultOrUnformed.impl.Op.type.150 = struct_value () [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.e88: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.374, @T.as.DefaultOrUnformed.impl(%Z) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.type.1d9: type = fn_type @T.as.DefaultOrUnformed.impl.Op, @T.as.DefaultOrUnformed.impl(%Z) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.f06: %T.as.DefaultOrUnformed.impl.Op.type.1d9 = struct_value () [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value %Z, (%DefaultOrUnformed.impl_witness.e88) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.specific_fn: <specific function> = specific_function %T.as.DefaultOrUnformed.impl.Op.f06, @T.as.DefaultOrUnformed.impl.Op(%Z) [concrete]
// CHECK:STDOUT:   %.7d5: type = fn_type_with_self_type %A.WithSelf.F.type.22c, %A.facet.c65 [concrete]
// CHECK:STDOUT:   %ptr.415: type = ptr_type %Z [concrete]
// CHECK:STDOUT:   %pattern_type.b4b: type = pattern_type %ptr.415 [concrete]
// CHECK:STDOUT:   %u.param_patt.5c1: %pattern_type.b4b = value_param_pattern [concrete]
// CHECK:STDOUT:   %u.patt.5b6: %pattern_type.b4b = at_binding_pattern u, %u.param_patt.5c1 [concrete]
// CHECK:STDOUT:   %tuple.8e4: %tuple.type.ff9 = tuple_value (%X, %Y, %ptr.415) [concrete]
// CHECK:STDOUT:   %tuple.type.2cd: type = tuple_type (%X, %Y, %ptr.415) [concrete]
// CHECK:STDOUT:   %.5bc: Core.Form = init_form %tuple.type.2cd [concrete]
// CHECK:STDOUT:   %pattern_type.b33: type = pattern_type %tuple.type.2cd [concrete]
// CHECK:STDOUT:   %return.param_patt.75d: %pattern_type.b33 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.8bb: %pattern_type.b33 = return_slot_pattern %return.param_patt.75d, %tuple.type.2cd [concrete]
// CHECK:STDOUT:   %Y.as.A.impl.F.specific_fn: <specific function> = specific_function %Y.as.A.impl.F, @Y.as.A.impl.F(%Z) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.3: type = fn_type @Destroy.Op.loc23_22.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.3: %Destroy.Op.type.1d8f74.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df9cc1.3: <witness> = custom_witness (%Destroy.Op.1a2547.3), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.5: type = fn_type @Destroy.Op.loc23_22.5 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.5: %Destroy.Op.type.1d8f74.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df9cc1.5: <witness> = custom_witness (%Destroy.Op.1a2547.5), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.e95: %Destroy.type = facet_value %tuple.type.2cd, (%custom_witness.df9cc1.5) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.964: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.e95) [concrete]
// CHECK:STDOUT:   %.41a: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.964, %Destroy.facet.e95 [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.6: type = fn_type @Destroy.Op.loc22 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.6: %Destroy.Op.type.1d8f74.6 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndWith.type.a06: type = generic_interface_type @BitAndWith [concrete]
// CHECK:STDOUT:   %BitAndWith.generic: %BitAndWith.type.a06 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndWith.type.802: type = facet_type <@BitAndWith, @BitAndWith(type)> [concrete]
// CHECK:STDOUT:   %BitAndWith.impl_witness: <witness> = impl_witness imports.%BitAndWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %BitAndWith.facet: %BitAndWith.type.802 = facet_value type, (%BitAndWith.impl_witness) [concrete]
// CHECK:STDOUT:   %BitAndWith.WithSelf.Op.type.38f: type = fn_type @BitAndWith.WithSelf.Op, @BitAndWith.WithSelf(type, %BitAndWith.facet) [concrete]
// CHECK:STDOUT:   %.d56: type = fn_type_with_self_type %BitAndWith.WithSelf.Op.type.38f, %BitAndWith.facet [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.type: type = fn_type @type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op: %type.as.BitAndWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.bound: <bound method> = bound_method %A.type.df0, %type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@A, @A(%X) & @Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.871: type = pattern_type %facet_type [concrete]
// CHECK:STDOUT:   %T.patt.c17: %pattern_type.871 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.5b7: %facet_type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %CallGeneric.type: type = fn_type @CallGeneric [concrete]
// CHECK:STDOUT:   %CallGeneric: %CallGeneric.type = struct_value () [concrete]
// CHECK:STDOUT:   %A.WithSelf.F.type.91d: type = fn_type @A.WithSelf.F, @A.WithSelf(%X, %T.5b7) [symbolic]
// CHECK:STDOUT:   %A.WithSelf.F.3f4: %A.WithSelf.F.type.91d = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.0af: type = facet_access_type %T.5b7 [symbolic]
// CHECK:STDOUT:   %A.lookup_impl_witness: <witness> = lookup_impl_witness %T.5b7, @A, @A(%X) [symbolic]
// CHECK:STDOUT:   %A.facet.318: %A.type.df0 = facet_value %T.as_type.0af, (%A.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %A.WithSelf.F.type.497: type = fn_type @A.WithSelf.F, @A.WithSelf(%X, %A.facet.318) [symbolic]
// CHECK:STDOUT:   %A.WithSelf.F.a4d: %A.WithSelf.F.type.497 = struct_value () [symbolic]
// CHECK:STDOUT:   %.151: type = fn_type_with_self_type %A.WithSelf.F.type.497, %A.facet.318 [symbolic]
// CHECK:STDOUT:   %impl.elem0.1a3: %.151 = impl_witness_access %A.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %tuple.575: %tuple.type.966 = tuple_value (%X, %A.facet.318, %ptr.415) [symbolic]
// CHECK:STDOUT:   %tuple.type.c2a: type = tuple_type (%X, %T.as_type.0af, %ptr.415) [symbolic]
// CHECK:STDOUT:   %.546: Core.Form = init_form %tuple.type.c2a [symbolic]
// CHECK:STDOUT:   %pattern_type.0e9: type = pattern_type %tuple.type.c2a [symbolic]
// CHECK:STDOUT:   %return.param_patt.579: %pattern_type.0e9 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.62e: %pattern_type.0e9 = return_slot_pattern %return.param_patt.579, %tuple.type.c2a [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.74d: <specific function> = specific_impl_function %impl.elem0.1a3, @A.WithSelf.F(%X, %A.facet.318, %Z) [symbolic]
// CHECK:STDOUT:   %require_complete.d14: <witness> = require_complete_type %tuple.type.c2a [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.a02: <witness> = lookup_impl_witness %tuple.type.c2a, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.2f8: %Destroy.type = facet_value %tuple.type.c2a, (%Destroy.lookup_impl_witness.a02) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.2ac: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.2f8) [symbolic]
// CHECK:STDOUT:   %.89c: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.2ac, %Destroy.facet.2f8 [symbolic]
// CHECK:STDOUT:   %impl.elem0.a1a: %.89c = impl_witness_access %Destroy.lookup_impl_witness.a02, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.65e: <specific function> = specific_impl_function %impl.elem0.a1a, @Destroy.WithSelf.Op(%Destroy.facet.2f8) [symbolic]
// CHECK:STDOUT:   %CallIndirect.type: type = fn_type @CallIndirect [concrete]
// CHECK:STDOUT:   %CallIndirect: %CallIndirect.type = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value: %facet_type = facet_value %Y, (%A.impl_witness, %custom_witness.df9cc1.3) [concrete]
// CHECK:STDOUT:   %CallGeneric.specific_fn: <specific function> = specific_function %CallGeneric, @CallGeneric(%facet_value) [concrete]
// CHECK:STDOUT:   %complete_type.458: <witness> = complete_type_witness %ptr.415 [concrete]
// CHECK:STDOUT:   %complete_type.dec: <witness> = complete_type_witness %tuple.type.2cd [concrete]
// CHECK:STDOUT:   %tuple.type.2c5: type = tuple_type (%empty_struct_type, %empty_struct_type, %ptr.415) [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.d9a: <witness> = impl_witness imports.%Copy.impl_witness_table.4f1, @ptr.as.Copy.impl(%Z) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.905: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%Z) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.ebe: %ptr.as.Copy.impl.Op.type.905 = struct_value () [concrete]
// CHECK:STDOUT:   %.072: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%Z) [concrete]
// CHECK:STDOUT:   %Copy.facet.302f: %Copy.type = facet_value %ptr.415, (%Copy.impl_witness.d9a) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.b68: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.302f) [concrete]
// CHECK:STDOUT:   %.6df: type = fn_type_with_self_type %Copy.WithSelf.Op.type.b68, %Copy.facet.302f [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.ebe, @ptr.as.Copy.impl.Op(%Z) [concrete]
// CHECK:STDOUT:   %tuple.1b5: %tuple.type.966 = tuple_value (%X, %A.facet.c65, %ptr.415) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .DefaultOrUnformed = %Core.DefaultOrUnformed
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .BitAndWith = %Core.BitAndWith
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.e5f: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.ff5f0d.1) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.ce912e.1)]
// CHECK:STDOUT:   %Copy.impl_witness_table.4f1 = impl_witness_table (%Core.import_ref.e5f), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.DefaultOrUnformed: type = import_ref Core//prelude/parts/default, DefaultOrUnformed, loaded [concrete = constants.%DefaultOrUnformed.type]
// CHECK:STDOUT:   %Core.import_ref.cc8: @T.as.DefaultOrUnformed.impl.%T.as.DefaultOrUnformed.impl.Op.type (%T.as.DefaultOrUnformed.impl.Op.type.150) = import_ref Core//prelude/parts/default, loc{{\d+_\d+}}, loaded [symbolic = @T.as.DefaultOrUnformed.impl.%T.as.DefaultOrUnformed.impl.Op (constants.%T.as.DefaultOrUnformed.impl.Op.53a)]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness_table.374 = impl_witness_table (%Core.import_ref.cc8), @T.as.DefaultOrUnformed.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
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
// CHECK:STDOUT:     %T.patt.loc5_14.1: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc5_14.2 (constants.%T.patt.d47)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5_17.1: type = splice_block %.loc5_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc5_17.2: type = type_literal type [concrete = type]
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
// CHECK:STDOUT:   %Call.decl: %Call.type = fn_decl @Call [concrete = constants.%Call] {} {}
// CHECK:STDOUT:   %CallGeneric.decl: %CallGeneric.type = fn_decl @CallGeneric [concrete = constants.%CallGeneric] {
// CHECK:STDOUT:     %T.patt.loc26_17.1: %pattern_type.871 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc26_17.2 (constants.%T.patt.c17)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc26_25.1: type = splice_block %.loc26_25.3 [concrete = constants.%facet_type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %A.ref: %A.type.55a = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:       %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:       %A.type: type = facet_type <@A, @A(constants.%X)> [concrete = constants.%A.type.df0]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:       %impl.elem0.loc26: %.d56 = impl_witness_access constants.%BitAndWith.impl_witness, element0 [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:       %bound_method.loc26: <bound method> = bound_method %A.type, %impl.elem0.loc26 [concrete = constants.%type.as.BitAndWith.impl.Op.bound]
// CHECK:STDOUT:       %type.as.BitAndWith.impl.Op.call: init type = call %bound_method.loc26(%A.type, %Destroy.ref) [concrete = constants.%facet_type]
// CHECK:STDOUT:       %.loc26_25.2: type = value_of_initializer %type.as.BitAndWith.impl.Op.call [concrete = constants.%facet_type]
// CHECK:STDOUT:       %.loc26_25.3: type = converted %type.as.BitAndWith.impl.Op.call, %.loc26_25.2 [concrete = constants.%facet_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc26_17.2: %facet_type = symbolic_binding T, 0 [symbolic = %T.loc26_17.1 (constants.%T.5b7)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallIndirect.decl: %CallIndirect.type = fn_decl @CallIndirect [concrete = constants.%CallIndirect] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @A(%T.loc5_14.2: type) {
// CHECK:STDOUT:   %T.patt.loc5_14.2: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc5_14.2 (constants.%T.patt.d47)]
// CHECK:STDOUT:   %T.loc5_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc5_14.1 (constants.%T.67d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.type: type = facet_type <@A, @A(%T.loc5_14.1)> [symbolic = %A.type (constants.%A.type.acd)]
// CHECK:STDOUT:   %Self.loc5_23.2: @A.%A.type (%A.type.acd) = symbolic_binding Self, 1 [symbolic = %Self.loc5_23.2 (constants.%Self.49a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc5_23.1: @A.%A.type (%A.type.acd) = symbolic_binding Self, 1 [symbolic = %Self.loc5_23.2 (constants.%Self.49a)]
// CHECK:STDOUT:     %A.WithSelf.decl = interface_with_self_decl @A [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %A.WithSelf.F.decl: @A.WithSelf.%A.WithSelf.F.type (%A.WithSelf.F.type.08d) = fn_decl @A.WithSelf.F [symbolic = @A.WithSelf.%A.WithSelf.F (constants.%A.WithSelf.F.3fa)] {
// CHECK:STDOUT:       %U.patt.loc6_9.1: %pattern_type.98f = symbolic_binding_pattern U, 2 [symbolic = %U.patt.loc6_9.2 (constants.%U.patt.b22)]
// CHECK:STDOUT:       %u.param_patt.loc6_19.1: @A.WithSelf.F.%pattern_type.loc6_19 (%pattern_type.986) = value_param_pattern [symbolic = %u.param_patt.loc6_19.2 (constants.%u.param_patt.a01)]
// CHECK:STDOUT:       %u.patt.loc6_19.1: @A.WithSelf.F.%pattern_type.loc6_19 (%pattern_type.986) = at_binding_pattern u, %u.param_patt.loc6_19.1 [symbolic = %u.patt.loc6_19.2 (constants.%u.patt.c78)]
// CHECK:STDOUT:       %return.param_patt.loc6_40.1: @A.WithSelf.F.%pattern_type.loc6_40 (%pattern_type.0d7) = out_param_pattern [symbolic = %return.param_patt.loc6_40.2 (constants.%return.param_patt.9d3)]
// CHECK:STDOUT:       %return.patt.loc6_25.1: @A.WithSelf.F.%pattern_type.loc6_40 (%pattern_type.0d7) = return_slot_pattern %return.param_patt.loc6_40.1, %.loc6_40.4 [symbolic = %return.patt.loc6_25.2 (constants.%return.patt.8aa)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @A.%T.loc5_14.2 [symbolic = %T (constants.%T.67d)]
// CHECK:STDOUT:       %.loc6_32: @A.WithSelf.F.%A.type (%A.type.acd) = specific_constant @A.%Self.loc5_23.1, @A(constants.%T.67d) [symbolic = %Self (constants.%Self.49a)]
// CHECK:STDOUT:       %Self.ref: @A.WithSelf.F.%A.type (%A.type.acd) = name_ref Self, %.loc6_32 [symbolic = %Self (constants.%Self.49a)]
// CHECK:STDOUT:       %U.ref.loc6_38: type = name_ref U, %U.loc6_9.2 [symbolic = %U.loc6_9.1 (constants.%U.4e7)]
// CHECK:STDOUT:       %ptr.loc6_39: type = ptr_type %U.ref.loc6_38 [symbolic = %ptr.loc6_22.1 (constants.%ptr.8f6)]
// CHECK:STDOUT:       %.loc6_40.2: @A.WithSelf.F.%tuple.type.loc6_40.1 (%tuple.type.76f) = tuple_literal (%T.ref, %Self.ref, %ptr.loc6_39) [symbolic = %tuple (constants.%tuple.0f4)]
// CHECK:STDOUT:       %Self.as_type.loc6_40.2: type = facet_access_type constants.%Self.49a [symbolic = %Self.as_type.loc6_40.1 (constants.%Self.as_type.763)]
// CHECK:STDOUT:       %.loc6_40.3: type = converted constants.%Self.49a, %Self.as_type.loc6_40.2 [symbolic = %Self.as_type.loc6_40.1 (constants.%Self.as_type.763)]
// CHECK:STDOUT:       %.loc6_40.4: type = converted %.loc6_40.2, constants.%tuple.type.640 [symbolic = %tuple.type.loc6_40.2 (constants.%tuple.type.640)]
// CHECK:STDOUT:       %.loc6_40.5: Core.Form = init_form %.loc6_40.4 [symbolic = %.loc6_40.1 (constants.%.f0d)]
// CHECK:STDOUT:       %.loc6_12.1: type = splice_block %.loc6_12.2 [concrete = type] {
// CHECK:STDOUT:         %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:         %.loc6_12.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %U.loc6_9.2: type = symbolic_binding U, 2 [symbolic = %U.loc6_9.1 (constants.%U.4e7)]
// CHECK:STDOUT:       %u.param: @A.WithSelf.F.%ptr.loc6_22.1 (%ptr.8f6) = value_param call_param0
// CHECK:STDOUT:       %.loc6_22: type = splice_block %ptr.loc6_22.2 [symbolic = %ptr.loc6_22.1 (constants.%ptr.8f6)] {
// CHECK:STDOUT:         %U.ref.loc6_21: type = name_ref U, %U.loc6_9.2 [symbolic = %U.loc6_9.1 (constants.%U.4e7)]
// CHECK:STDOUT:         %ptr.loc6_22.2: type = ptr_type %U.ref.loc6_21 [symbolic = %ptr.loc6_22.1 (constants.%ptr.8f6)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %u: @A.WithSelf.F.%ptr.loc6_22.1 (%ptr.8f6) = wrapper_binding u, %u.param
// CHECK:STDOUT:       %return.param: ref @A.WithSelf.F.%tuple.type.loc6_40.2 (%tuple.type.640) = out_param call_param1
// CHECK:STDOUT:       %return: ref @A.WithSelf.F.%tuple.type.loc6_40.2 (%tuple.type.640) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc6_41.1: @A.WithSelf.%A.assoc_type (%A.assoc_type.447) = assoc_entity element0, %A.WithSelf.F.decl [symbolic = %assoc0.loc6_41.2 (constants.%assoc0.d4b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc5_23.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .F = @A.WithSelf.%assoc0.loc6_41.1
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
// CHECK:STDOUT:     %U.patt.loc16_9.1: %pattern_type.98f = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc16_9.2 (constants.%U.patt.d47)]
// CHECK:STDOUT:     %u.param_patt.loc16_19.1: @Y.as.A.impl.F.%pattern_type.loc16_19 (%pattern_type.4f4b84.1) = value_param_pattern [symbolic = %u.param_patt.loc16_19.2 (constants.%u.param_patt.58b)]
// CHECK:STDOUT:     %u.patt.loc16_19.1: @Y.as.A.impl.F.%pattern_type.loc16_19 (%pattern_type.4f4b84.1) = at_binding_pattern u, %u.param_patt.loc16_19.1 [symbolic = %u.patt.loc16_19.2 (constants.%u.patt.542)]
// CHECK:STDOUT:     %return.param_patt.loc16_37.1: @Y.as.A.impl.F.%pattern_type.loc16_37 (%pattern_type.178) = out_param_pattern [symbolic = %return.param_patt.loc16_37.2 (constants.%return.param_patt.b96)]
// CHECK:STDOUT:     %return.patt.loc16_25.1: @Y.as.A.impl.F.%pattern_type.loc16_37 (%pattern_type.178) = return_slot_pattern %return.param_patt.loc16_37.1, %.loc16_37.3 [symbolic = %return.patt.loc16_25.2 (constants.%return.patt.f2f)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %Y.ref: type = name_ref Y, file.%Y.decl [concrete = constants.%Y]
// CHECK:STDOUT:     %U.ref.loc16_35: type = name_ref U, %U.loc16_9.2 [symbolic = %U.loc16_9.1 (constants.%U.67d)]
// CHECK:STDOUT:     %ptr.loc16_36: type = ptr_type %U.ref.loc16_35 [symbolic = %ptr.loc16_22.1 (constants.%ptr.e8f8f9.1)]
// CHECK:STDOUT:     %.loc16_37.2: %tuple.type.ff9 = tuple_literal (%X.ref, %Y.ref, %ptr.loc16_36) [symbolic = %tuple (constants.%tuple.f57)]
// CHECK:STDOUT:     %.loc16_37.3: type = converted %.loc16_37.2, constants.%tuple.type.316 [symbolic = %tuple.type.loc16 (constants.%tuple.type.316)]
// CHECK:STDOUT:     %.loc16_37.4: Core.Form = init_form %.loc16_37.3 [symbolic = %.loc16_37.1 (constants.%.3c1)]
// CHECK:STDOUT:     %.loc16_12.1: type = splice_block %.loc16_12.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc16_12.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc16_9.2: type = symbolic_binding U, 0 [symbolic = %U.loc16_9.1 (constants.%U.67d)]
// CHECK:STDOUT:     %u.param: @Y.as.A.impl.F.%ptr.loc16_22.1 (%ptr.e8f8f9.1) = value_param call_param0
// CHECK:STDOUT:     %.loc16_22: type = splice_block %ptr.loc16_22.2 [symbolic = %ptr.loc16_22.1 (constants.%ptr.e8f8f9.1)] {
// CHECK:STDOUT:       %U.ref.loc16_21: type = name_ref U, %U.loc16_9.2 [symbolic = %U.loc16_9.1 (constants.%U.67d)]
// CHECK:STDOUT:       %ptr.loc16_22.2: type = ptr_type %U.ref.loc16_21 [symbolic = %ptr.loc16_22.1 (constants.%ptr.e8f8f9.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: @Y.as.A.impl.F.%ptr.loc16_22.1 (%ptr.e8f8f9.1) = wrapper_binding u, %u.param
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
// CHECK:STDOUT: generic fn @A.WithSelf.F(@A.%T.loc5_14.2: type, @A.%Self.loc5_23.1: @A.%A.type (%A.type.acd), %U.loc6_9.2: type) {
// CHECK:STDOUT:   %U.patt.loc6_9.2: %pattern_type.98f = symbolic_binding_pattern U, 2 [symbolic = %U.patt.loc6_9.2 (constants.%U.patt.b22)]
// CHECK:STDOUT:   %U.loc6_9.1: type = symbolic_binding U, 2 [symbolic = %U.loc6_9.1 (constants.%U.4e7)]
// CHECK:STDOUT:   %ptr.loc6_22.1: type = ptr_type %U.loc6_9.1 [symbolic = %ptr.loc6_22.1 (constants.%ptr.8f6)]
// CHECK:STDOUT:   %pattern_type.loc6_19: type = pattern_type %ptr.loc6_22.1 [symbolic = %pattern_type.loc6_19 (constants.%pattern_type.986)]
// CHECK:STDOUT:   %u.param_patt.loc6_19.2: @A.WithSelf.F.%pattern_type.loc6_19 (%pattern_type.986) = value_param_pattern [symbolic = %u.param_patt.loc6_19.2 (constants.%u.param_patt.a01)]
// CHECK:STDOUT:   %u.patt.loc6_19.2: @A.WithSelf.F.%pattern_type.loc6_19 (%pattern_type.986) = at_binding_pattern u, %u.param_patt.loc6_19.2 [symbolic = %u.patt.loc6_19.2 (constants.%u.patt.c78)]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T.67d)]
// CHECK:STDOUT:   %A.type: type = facet_type <@A, @A(%T)> [symbolic = %A.type (constants.%A.type.acd)]
// CHECK:STDOUT:   %Self: @A.WithSelf.F.%A.type (%A.type.acd) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.49a)]
// CHECK:STDOUT:   %tuple.type.loc6_40.1: type = tuple_type (type, %A.type, type) [symbolic = %tuple.type.loc6_40.1 (constants.%tuple.type.76f)]
// CHECK:STDOUT:   %tuple: @A.WithSelf.F.%tuple.type.loc6_40.1 (%tuple.type.76f) = tuple_value (%T, %Self, %ptr.loc6_22.1) [symbolic = %tuple (constants.%tuple.0f4)]
// CHECK:STDOUT:   %Self.as_type.loc6_40.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc6_40.1 (constants.%Self.as_type.763)]
// CHECK:STDOUT:   %tuple.type.loc6_40.2: type = tuple_type (%T, %Self.as_type.loc6_40.1, %ptr.loc6_22.1) [symbolic = %tuple.type.loc6_40.2 (constants.%tuple.type.640)]
// CHECK:STDOUT:   %.loc6_40.1: Core.Form = init_form %tuple.type.loc6_40.2 [symbolic = %.loc6_40.1 (constants.%.f0d)]
// CHECK:STDOUT:   %pattern_type.loc6_40: type = pattern_type %tuple.type.loc6_40.2 [symbolic = %pattern_type.loc6_40 (constants.%pattern_type.0d7)]
// CHECK:STDOUT:   %return.param_patt.loc6_40.2: @A.WithSelf.F.%pattern_type.loc6_40 (%pattern_type.0d7) = out_param_pattern [symbolic = %return.param_patt.loc6_40.2 (constants.%return.param_patt.9d3)]
// CHECK:STDOUT:   %return.patt.loc6_25.2: @A.WithSelf.F.%pattern_type.loc6_40 (%pattern_type.0d7) = return_slot_pattern %return.param_patt.loc6_40.2, %tuple.type.loc6_40.2 [symbolic = %return.patt.loc6_25.2 (constants.%return.patt.8aa)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param: @A.WithSelf.F.%ptr.loc6_22.1 (%ptr.8f6)) -> out %return.param: @A.WithSelf.F.%tuple.type.loc6_40.2 (%tuple.type.640);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Y.as.A.impl.F(%U.loc16_9.2: type) {
// CHECK:STDOUT:   %U.patt.loc16_9.2: %pattern_type.98f = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc16_9.2 (constants.%U.patt.d47)]
// CHECK:STDOUT:   %U.loc16_9.1: type = symbolic_binding U, 0 [symbolic = %U.loc16_9.1 (constants.%U.67d)]
// CHECK:STDOUT:   %ptr.loc16_22.1: type = ptr_type %U.loc16_9.1 [symbolic = %ptr.loc16_22.1 (constants.%ptr.e8f8f9.1)]
// CHECK:STDOUT:   %pattern_type.loc16_19: type = pattern_type %ptr.loc16_22.1 [symbolic = %pattern_type.loc16_19 (constants.%pattern_type.4f4b84.1)]
// CHECK:STDOUT:   %u.param_patt.loc16_19.2: @Y.as.A.impl.F.%pattern_type.loc16_19 (%pattern_type.4f4b84.1) = value_param_pattern [symbolic = %u.param_patt.loc16_19.2 (constants.%u.param_patt.58b)]
// CHECK:STDOUT:   %u.patt.loc16_19.2: @Y.as.A.impl.F.%pattern_type.loc16_19 (%pattern_type.4f4b84.1) = at_binding_pattern u, %u.param_patt.loc16_19.2 [symbolic = %u.patt.loc16_19.2 (constants.%u.patt.542)]
// CHECK:STDOUT:   %tuple: %tuple.type.ff9 = tuple_value (constants.%X, constants.%Y, %ptr.loc16_22.1) [symbolic = %tuple (constants.%tuple.f57)]
// CHECK:STDOUT:   %tuple.type.loc16: type = tuple_type (constants.%X, constants.%Y, %ptr.loc16_22.1) [symbolic = %tuple.type.loc16 (constants.%tuple.type.316)]
// CHECK:STDOUT:   %.loc16_37.1: Core.Form = init_form %tuple.type.loc16 [symbolic = %.loc16_37.1 (constants.%.3c1)]
// CHECK:STDOUT:   %pattern_type.loc16_37: type = pattern_type %tuple.type.loc16 [symbolic = %pattern_type.loc16_37 (constants.%pattern_type.178)]
// CHECK:STDOUT:   %return.param_patt.loc16_37.2: @Y.as.A.impl.F.%pattern_type.loc16_37 (%pattern_type.178) = out_param_pattern [symbolic = %return.param_patt.loc16_37.2 (constants.%return.param_patt.b96)]
// CHECK:STDOUT:   %return.patt.loc16_25.2: @Y.as.A.impl.F.%pattern_type.loc16_37 (%pattern_type.178) = return_slot_pattern %return.param_patt.loc16_37.2, %tuple.type.loc16 [symbolic = %return.patt.loc16_25.2 (constants.%return.patt.f2f)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc16_19: <witness> = require_complete_type %ptr.loc16_22.1 [symbolic = %require_complete.loc16_19 (constants.%require_complete.ef162c.1)]
// CHECK:STDOUT:   %require_complete.loc16_25: <witness> = require_complete_type %tuple.type.loc16 [symbolic = %require_complete.loc16_25 (constants.%require_complete.44f)]
// CHECK:STDOUT:   %tuple.type.loc17: type = tuple_type (constants.%empty_struct_type, constants.%empty_struct_type, %ptr.loc16_22.1) [symbolic = %tuple.type.loc17 (constants.%tuple.type.dd2)]
// CHECK:STDOUT:   %.loc17_21.3: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%U.loc16_9.1) [symbolic = %.loc17_21.3 (constants.%.0e9)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr.loc16_22.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.1da)]
// CHECK:STDOUT:   %Copy.facet.loc17_21.3: %Copy.type = facet_value %ptr.loc16_22.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet.loc17_21.3 (constants.%Copy.facet.302b)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.loc17_21.3) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.884)]
// CHECK:STDOUT:   %.loc17_21.4: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %Copy.facet.loc17_21.3 [symbolic = %.loc17_21.4 (constants.%.be5)]
// CHECK:STDOUT:   %impl.elem0.loc17_21.2: @Y.as.A.impl.F.%.loc17_21.4 (%.be5) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc17_21.2 (constants.%impl.elem0.484)]
// CHECK:STDOUT:   %specific_impl_fn.loc17_21.2: <specific function> = specific_impl_function %impl.elem0.loc17_21.2, @Copy.WithSelf.Op(%Copy.facet.loc17_21.3) [symbolic = %specific_impl_fn.loc17_21.2 (constants.%specific_impl_fn.a76)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param: @Y.as.A.impl.F.%ptr.loc16_22.1 (%ptr.e8f8f9.1)) -> out %return.param: @Y.as.A.impl.F.%tuple.type.loc16 (%tuple.type.316) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc17_14.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc17_18.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %u.ref: @Y.as.A.impl.F.%ptr.loc16_22.1 (%ptr.e8f8f9.1) = name_ref u, %u
// CHECK:STDOUT:     %.loc17_22.1: @Y.as.A.impl.F.%tuple.type.loc17 (%tuple.type.dd2) = tuple_literal (%.loc17_14.1, %.loc17_18.1, %u.ref)
// CHECK:STDOUT:     %tuple.elem0: ref %X = tuple_access %return.param, element0
// CHECK:STDOUT:     %.loc17_14.2: init %X to %tuple.elem0 = class_init () [concrete = constants.%X.val]
// CHECK:STDOUT:     %.loc17_22.2: init %X = converted %.loc17_14.1, %.loc17_14.2 [concrete = constants.%X.val]
// CHECK:STDOUT:     %tuple.elem1: ref %Y = tuple_access %return.param, element1
// CHECK:STDOUT:     %.loc17_18.2: init %Y to %tuple.elem1 = class_init () [concrete = constants.%Y.val]
// CHECK:STDOUT:     %.loc17_22.3: init %Y = converted %.loc17_18.1, %.loc17_18.2 [concrete = constants.%Y.val]
// CHECK:STDOUT:     %impl.elem0.loc17_21.1: @Y.as.A.impl.F.%.loc17_21.4 (%.be5) = impl_witness_access constants.%Copy.lookup_impl_witness.1da, element0 [symbolic = %impl.elem0.loc17_21.2 (constants.%impl.elem0.484)]
// CHECK:STDOUT:     %bound_method.loc17_21.1: <bound method> = bound_method %u.ref, %impl.elem0.loc17_21.1
// CHECK:STDOUT:     %Copy.facet.loc17_21.1: %Copy.type = facet_value constants.%ptr.e8f8f9.1, (constants.%Copy.lookup_impl_witness.1da) [symbolic = %Copy.facet.loc17_21.3 (constants.%Copy.facet.302b)]
// CHECK:STDOUT:     %.loc17_21.1: %Copy.type = converted constants.%ptr.e8f8f9.1, %Copy.facet.loc17_21.1 [symbolic = %Copy.facet.loc17_21.3 (constants.%Copy.facet.302b)]
// CHECK:STDOUT:     %Copy.facet.loc17_21.2: %Copy.type = facet_value constants.%ptr.e8f8f9.1, (constants.%Copy.lookup_impl_witness.1da) [symbolic = %Copy.facet.loc17_21.3 (constants.%Copy.facet.302b)]
// CHECK:STDOUT:     %.loc17_21.2: %Copy.type = converted constants.%ptr.e8f8f9.1, %Copy.facet.loc17_21.2 [symbolic = %Copy.facet.loc17_21.3 (constants.%Copy.facet.302b)]
// CHECK:STDOUT:     %specific_impl_fn.loc17_21.1: <specific function> = specific_impl_function %impl.elem0.loc17_21.1, @Copy.WithSelf.Op(constants.%Copy.facet.302b) [symbolic = %specific_impl_fn.loc17_21.2 (constants.%specific_impl_fn.a76)]
// CHECK:STDOUT:     %bound_method.loc17_21.2: <bound method> = bound_method %u.ref, %specific_impl_fn.loc17_21.1
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @Y.as.A.impl.F.%ptr.loc16_22.1 (%ptr.e8f8f9.1) = call %bound_method.loc17_21.2(%u.ref)
// CHECK:STDOUT:     %tuple.elem2: ref @Y.as.A.impl.F.%ptr.loc16_22.1 (%ptr.e8f8f9.1) = tuple_access %return.param, element2
// CHECK:STDOUT:     %.loc17_22.4: init @Y.as.A.impl.F.%ptr.loc16_22.1 (%ptr.e8f8f9.1) to %tuple.elem2 = in_place_init %Copy.WithSelf.Op.call
// CHECK:STDOUT:     %.loc17_22.5: init @Y.as.A.impl.F.%tuple.type.loc16 (%tuple.type.316) to %return.param = tuple_init (%.loc17_22.2, %.loc17_22.3, %.loc17_22.4)
// CHECK:STDOUT:     %.loc17_23: init @Y.as.A.impl.F.%tuple.type.loc16 (%tuple.type.316) = converted %.loc17_22.1, %.loc17_22.5
// CHECK:STDOUT:     return %.loc17_23 to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %u.patt: %pattern_type.34e = ref_binding_pattern u [concrete = constants.%u.patt.352]
// CHECK:STDOUT:     %u.var_patt: %pattern_type.34e = var_pattern %u.patt [concrete = constants.%u.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %u.var: ref %Z = var %u.var_patt
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value constants.%Z, (constants.%DefaultOrUnformed.impl_witness.e88) [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %.loc22_11.1: %DefaultOrUnformed.type = converted constants.%Z, %DefaultOrUnformed.facet [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %as_type.loc22: type = facet_access_type %.loc22_11.1 [concrete = constants.%Z]
// CHECK:STDOUT:   %.loc22_11.2: type = converted %.loc22_11.1, %as_type.loc22 [concrete = constants.%Z]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.specific_fn: <specific function> = specific_function constants.%T.as.DefaultOrUnformed.impl.Op.f06, @T.as.DefaultOrUnformed.impl.Op(constants.%Z) [concrete = constants.%T.as.DefaultOrUnformed.impl.Op.specific_fn]
// CHECK:STDOUT:   %.loc22_3: ref %Z = splice_block %u.var {}
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.call: init %Z to %.loc22_3 = call %T.as.DefaultOrUnformed.impl.Op.specific_fn()
// CHECK:STDOUT:   assign %u.var, %T.as.DefaultOrUnformed.impl.Op.call
// CHECK:STDOUT:   %Z.ref.loc22: type = name_ref Z, file.%Z.decl [concrete = constants.%Z]
// CHECK:STDOUT:   %u: ref %Z = wrapper_binding u, %u.var
// CHECK:STDOUT:   %Y.ref: type = name_ref Y, file.%Y.decl [concrete = constants.%Y]
// CHECK:STDOUT:   %A.ref: %A.type.55a = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:   %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %A.type: type = facet_type <@A, @A(constants.%X)> [concrete = constants.%A.type.df0]
// CHECK:STDOUT:   %A.facet: %A.type.df0 = facet_value %Y.ref, (constants.%A.impl_witness) [concrete = constants.%A.facet.c65]
// CHECK:STDOUT:   %.loc23_6: %A.type.df0 = converted %Y.ref, %A.facet [concrete = constants.%A.facet.c65]
// CHECK:STDOUT:   %as_type.loc23: type = facet_access_type %.loc23_6 [concrete = constants.%Y]
// CHECK:STDOUT:   %.loc23_14.1: type = converted %.loc23_6, %as_type.loc23 [concrete = constants.%Y]
// CHECK:STDOUT:   %.loc23_14.2: %A.assoc_type.a54 = specific_constant @A.WithSelf.%assoc0.loc6_41.1, @A.WithSelf(constants.%X, constants.%A.facet.c65) [concrete = constants.%assoc0.987]
// CHECK:STDOUT:   %F.ref: %A.assoc_type.a54 = name_ref F, %.loc23_14.2 [concrete = constants.%assoc0.987]
// CHECK:STDOUT:   %impl.elem0: %.7d5 = impl_witness_access constants.%A.impl_witness, element0 [concrete = constants.%Y.as.A.impl.F]
// CHECK:STDOUT:   %Z.ref.loc23: type = name_ref Z, file.%Z.decl [concrete = constants.%Z]
// CHECK:STDOUT:   %u.ref: ref %Z = name_ref u, %u
// CHECK:STDOUT:   %addr: %ptr.415 = addr_of %u.ref
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Y.as.A.impl.F(constants.%Z) [concrete = constants.%Y.as.A.impl.F.specific_fn]
// CHECK:STDOUT:   %.loc23_22.1: ref %tuple.type.2cd = temporary_storage
// CHECK:STDOUT:   %Y.as.A.impl.F.call: init %tuple.type.2cd to %.loc23_22.1 = call %specific_fn(%addr)
// CHECK:STDOUT:   %.loc23_22.2: ref %tuple.type.2cd = temporary %.loc23_22.1, %Y.as.A.impl.F.call
// CHECK:STDOUT:   %Destroy.Op.bound.loc23: <bound method> = bound_method %.loc23_22.2, constants.%Destroy.Op.1a2547.5
// CHECK:STDOUT:   %Destroy.Op.call.loc23: init %empty_tuple.type = call %Destroy.Op.bound.loc23(%.loc23_22.2)
// CHECK:STDOUT:   %Destroy.Op.bound.loc22: <bound method> = bound_method %u.var, constants.%Destroy.Op.1a2547.6
// CHECK:STDOUT:   %Destroy.Op.call.loc22: init %empty_tuple.type = call %Destroy.Op.bound.loc22(%u.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc23_22.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc23_22.2(%self.param: ref %X) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc23_22.3(%self.param: ref %Y) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc23_22.4(%self.param: ref %ptr.415) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc23_22.5(%self.param: ref %tuple.type.2cd) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc22(%self.param: ref %Z) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallGeneric(%T.loc26_17.2: %facet_type) {
// CHECK:STDOUT:   %T.patt.loc26_17.2: %pattern_type.871 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc26_17.2 (constants.%T.patt.c17)]
// CHECK:STDOUT:   %T.loc26_17.1: %facet_type = symbolic_binding T, 0 [symbolic = %T.loc26_17.1 (constants.%T.5b7)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc28_4.2: type = facet_access_type %T.loc26_17.1 [symbolic = %T.as_type.loc28_4.2 (constants.%T.as_type.0af)]
// CHECK:STDOUT:   %A.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc26_17.1, @A, @A(constants.%X) [symbolic = %A.lookup_impl_witness (constants.%A.lookup_impl_witness)]
// CHECK:STDOUT:   %A.facet: %A.type.df0 = facet_value %T.as_type.loc28_4.2, (%A.lookup_impl_witness) [symbolic = %A.facet (constants.%A.facet.318)]
// CHECK:STDOUT:   %A.WithSelf.F.type: type = fn_type @A.WithSelf.F, @A.WithSelf(constants.%X, %A.facet) [symbolic = %A.WithSelf.F.type (constants.%A.WithSelf.F.type.497)]
// CHECK:STDOUT:   %.loc28_4.3: type = fn_type_with_self_type %A.WithSelf.F.type, %A.facet [symbolic = %.loc28_4.3 (constants.%.151)]
// CHECK:STDOUT:   %impl.elem0.loc28_4.2: @CallGeneric.%.loc28_4.3 (%.151) = impl_witness_access %A.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc28_4.2 (constants.%impl.elem0.1a3)]
// CHECK:STDOUT:   %specific_impl_fn.loc28_4.2: <specific function> = specific_impl_function %impl.elem0.loc28_4.2, @A.WithSelf.F(constants.%X, %A.facet, constants.%Z) [symbolic = %specific_impl_fn.loc28_4.2 (constants.%specific_impl_fn.74d)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (constants.%X, %T.as_type.loc28_4.2, constants.%ptr.415) [symbolic = %tuple.type (constants.%tuple.type.c2a)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %tuple.type [symbolic = %require_complete (constants.%require_complete.d14)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %tuple.type, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.a02)]
// CHECK:STDOUT:   %Destroy.facet.loc28_12.3: %Destroy.type = facet_value %tuple.type, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet.loc28_12.3 (constants.%Destroy.facet.2f8)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc28_12.3) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.2ac)]
// CHECK:STDOUT:   %.loc28_12.5: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Destroy.facet.loc28_12.3 [symbolic = %.loc28_12.5 (constants.%.89c)]
// CHECK:STDOUT:   %impl.elem0.loc28_12.2: @CallGeneric.%.loc28_12.5 (%.89c) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc28_12.2 (constants.%impl.elem0.a1a)]
// CHECK:STDOUT:   %specific_impl_fn.loc28_12.2: <specific function> = specific_impl_function %impl.elem0.loc28_12.2, @Destroy.WithSelf.Op(%Destroy.facet.loc28_12.3) [symbolic = %specific_impl_fn.loc28_12.2 (constants.%specific_impl_fn.65e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %u.patt: %pattern_type.34e = ref_binding_pattern u [concrete = constants.%u.patt.352]
// CHECK:STDOUT:       %u.var_patt: %pattern_type.34e = var_pattern %u.patt [concrete = constants.%u.var_patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u.var: ref %Z = var %u.var_patt
// CHECK:STDOUT:     %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value constants.%Z, (constants.%DefaultOrUnformed.impl_witness.e88) [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:     %.loc27_11.1: %DefaultOrUnformed.type = converted constants.%Z, %DefaultOrUnformed.facet [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:     %as_type: type = facet_access_type %.loc27_11.1 [concrete = constants.%Z]
// CHECK:STDOUT:     %.loc27_11.2: type = converted %.loc27_11.1, %as_type [concrete = constants.%Z]
// CHECK:STDOUT:     %T.as.DefaultOrUnformed.impl.Op.specific_fn: <specific function> = specific_function constants.%T.as.DefaultOrUnformed.impl.Op.f06, @T.as.DefaultOrUnformed.impl.Op(constants.%Z) [concrete = constants.%T.as.DefaultOrUnformed.impl.Op.specific_fn]
// CHECK:STDOUT:     %.loc27_3: ref %Z = splice_block %u.var {}
// CHECK:STDOUT:     %T.as.DefaultOrUnformed.impl.Op.call: init %Z to %.loc27_3 = call %T.as.DefaultOrUnformed.impl.Op.specific_fn()
// CHECK:STDOUT:     assign %u.var, %T.as.DefaultOrUnformed.impl.Op.call
// CHECK:STDOUT:     %Z.ref.loc27: type = name_ref Z, file.%Z.decl [concrete = constants.%Z]
// CHECK:STDOUT:     %u: ref %Z = wrapper_binding u, %u.var
// CHECK:STDOUT:     %T.ref: %facet_type = name_ref T, %T.loc26_17.2 [symbolic = %T.loc26_17.1 (constants.%T.5b7)]
// CHECK:STDOUT:     %T.as_type.loc28_4.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc28_4.2 (constants.%T.as_type.0af)]
// CHECK:STDOUT:     %.loc28_4.1: type = converted %T.ref, %T.as_type.loc28_4.1 [symbolic = %T.as_type.loc28_4.2 (constants.%T.as_type.0af)]
// CHECK:STDOUT:     %.loc28_4.2: %A.assoc_type.a54 = specific_constant @A.WithSelf.%assoc0.loc6_41.1, @A.WithSelf(constants.%X, constants.%T.5b7) [concrete = constants.%assoc0.987]
// CHECK:STDOUT:     %F.ref: %A.assoc_type.a54 = name_ref F, %.loc28_4.2 [concrete = constants.%assoc0.987]
// CHECK:STDOUT:     %impl.elem0.loc28_4.1: @CallGeneric.%.loc28_4.3 (%.151) = impl_witness_access constants.%A.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc28_4.2 (constants.%impl.elem0.1a3)]
// CHECK:STDOUT:     %Z.ref.loc28: type = name_ref Z, file.%Z.decl [concrete = constants.%Z]
// CHECK:STDOUT:     %u.ref: ref %Z = name_ref u, %u
// CHECK:STDOUT:     %addr: %ptr.415 = addr_of %u.ref
// CHECK:STDOUT:     %specific_impl_fn.loc28_4.1: <specific function> = specific_impl_function %impl.elem0.loc28_4.1, @A.WithSelf.F(constants.%X, constants.%A.facet.318, constants.%Z) [symbolic = %specific_impl_fn.loc28_4.2 (constants.%specific_impl_fn.74d)]
// CHECK:STDOUT:     %.loc28_12.1: ref @CallGeneric.%tuple.type (%tuple.type.c2a) = temporary_storage
// CHECK:STDOUT:     %A.WithSelf.F.call: init @CallGeneric.%tuple.type (%tuple.type.c2a) to %.loc28_12.1 = call %specific_impl_fn.loc28_4.1(%addr)
// CHECK:STDOUT:     %.loc28_12.2: ref @CallGeneric.%tuple.type (%tuple.type.c2a) = temporary %.loc28_12.1, %A.WithSelf.F.call
// CHECK:STDOUT:     %impl.elem0.loc28_12.1: @CallGeneric.%.loc28_12.5 (%.89c) = impl_witness_access constants.%Destroy.lookup_impl_witness.a02, element0 [symbolic = %impl.elem0.loc28_12.2 (constants.%impl.elem0.a1a)]
// CHECK:STDOUT:     %bound_method.loc28_12.1: <bound method> = bound_method %.loc28_12.2, %impl.elem0.loc28_12.1
// CHECK:STDOUT:     %Destroy.facet.loc28_12.1: %Destroy.type = facet_value constants.%tuple.type.c2a, (constants.%Destroy.lookup_impl_witness.a02) [symbolic = %Destroy.facet.loc28_12.3 (constants.%Destroy.facet.2f8)]
// CHECK:STDOUT:     %.loc28_12.3: %Destroy.type = converted constants.%tuple.type.c2a, %Destroy.facet.loc28_12.1 [symbolic = %Destroy.facet.loc28_12.3 (constants.%Destroy.facet.2f8)]
// CHECK:STDOUT:     %Destroy.facet.loc28_12.2: %Destroy.type = facet_value constants.%tuple.type.c2a, (constants.%Destroy.lookup_impl_witness.a02) [symbolic = %Destroy.facet.loc28_12.3 (constants.%Destroy.facet.2f8)]
// CHECK:STDOUT:     %.loc28_12.4: %Destroy.type = converted constants.%tuple.type.c2a, %Destroy.facet.loc28_12.2 [symbolic = %Destroy.facet.loc28_12.3 (constants.%Destroy.facet.2f8)]
// CHECK:STDOUT:     %specific_impl_fn.loc28_12.1: <specific function> = specific_impl_function %impl.elem0.loc28_12.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.2f8) [symbolic = %specific_impl_fn.loc28_12.2 (constants.%specific_impl_fn.65e)]
// CHECK:STDOUT:     %bound_method.loc28_12.2: <bound method> = bound_method %.loc28_12.2, %specific_impl_fn.loc28_12.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc28_12.2(%.loc28_12.2)
// CHECK:STDOUT:     %Destroy.Op.bound: <bound method> = bound_method %u.var, constants.%Destroy.Op.1a2547.6
// CHECK:STDOUT:     %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%u.var)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallIndirect() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CallGeneric.ref: %CallGeneric.type = name_ref CallGeneric, file.%CallGeneric.decl [concrete = constants.%CallGeneric]
// CHECK:STDOUT:   %Y.ref: type = name_ref Y, file.%Y.decl [concrete = constants.%Y]
// CHECK:STDOUT:   %facet_value: %facet_type = facet_value %Y.ref, (constants.%A.impl_witness, constants.%custom_witness.df9cc1.3) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc32: %facet_type = converted %Y.ref, %facet_value [concrete = constants.%facet_value]
// CHECK:STDOUT:   %CallGeneric.specific_fn: <specific function> = specific_function %CallGeneric.ref, @CallGeneric(constants.%facet_value) [concrete = constants.%CallGeneric.specific_fn]
// CHECK:STDOUT:   %CallGeneric.call: init %empty_tuple.type = call %CallGeneric.specific_fn()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%T.67d) {
// CHECK:STDOUT:   %T.patt.loc5_14.2 => constants.%T.patt.d47
// CHECK:STDOUT:   %T.loc5_14.1 => constants.%T.67d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%T.67d, constants.%Self.49a) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf.F(constants.%T.67d, constants.%Self.49a, constants.%U.4e7) {
// CHECK:STDOUT:   %U.patt.loc6_9.2 => constants.%U.patt.b22
// CHECK:STDOUT:   %U.loc6_9.1 => constants.%U.4e7
// CHECK:STDOUT:   %ptr.loc6_22.1 => constants.%ptr.8f6
// CHECK:STDOUT:   %pattern_type.loc6_19 => constants.%pattern_type.986
// CHECK:STDOUT:   %u.param_patt.loc6_19.2 => constants.%u.param_patt.a01
// CHECK:STDOUT:   %u.patt.loc6_19.2 => constants.%u.patt.c78
// CHECK:STDOUT:   %T => constants.%T.67d
// CHECK:STDOUT:   %A.type => constants.%A.type.acd
// CHECK:STDOUT:   %Self => constants.%Self.49a
// CHECK:STDOUT:   %tuple.type.loc6_40.1 => constants.%tuple.type.76f
// CHECK:STDOUT:   %tuple => constants.%tuple.0f4
// CHECK:STDOUT:   %Self.as_type.loc6_40.1 => constants.%Self.as_type.763
// CHECK:STDOUT:   %tuple.type.loc6_40.2 => constants.%tuple.type.640
// CHECK:STDOUT:   %.loc6_40.1 => constants.%.f0d
// CHECK:STDOUT:   %pattern_type.loc6_40 => constants.%pattern_type.0d7
// CHECK:STDOUT:   %return.param_patt.loc6_40.2 => constants.%return.param_patt.9d3
// CHECK:STDOUT:   %return.patt.loc6_25.2 => constants.%return.patt.8aa
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%X) {
// CHECK:STDOUT:   %T.patt.loc5_14.2 => constants.%T.patt.d47
// CHECK:STDOUT:   %T.loc5_14.1 => constants.%X
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.type => constants.%A.type.df0
// CHECK:STDOUT:   %Self.loc5_23.2 => constants.%Self.891
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
// CHECK:STDOUT:   %assoc0.loc6_41.2 => constants.%assoc0.987
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y.as.A.impl.F(constants.%U.67d) {
// CHECK:STDOUT:   %U.patt.loc16_9.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc16_9.1 => constants.%U.67d
// CHECK:STDOUT:   %ptr.loc16_22.1 => constants.%ptr.e8f8f9.1
// CHECK:STDOUT:   %pattern_type.loc16_19 => constants.%pattern_type.4f4b84.1
// CHECK:STDOUT:   %u.param_patt.loc16_19.2 => constants.%u.param_patt.58b
// CHECK:STDOUT:   %u.patt.loc16_19.2 => constants.%u.patt.542
// CHECK:STDOUT:   %tuple => constants.%tuple.f57
// CHECK:STDOUT:   %tuple.type.loc16 => constants.%tuple.type.316
// CHECK:STDOUT:   %.loc16_37.1 => constants.%.3c1
// CHECK:STDOUT:   %pattern_type.loc16_37 => constants.%pattern_type.178
// CHECK:STDOUT:   %return.param_patt.loc16_37.2 => constants.%return.param_patt.b96
// CHECK:STDOUT:   %return.patt.loc16_25.2 => constants.%return.patt.f2f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%X, constants.%A.facet.c65) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.df0
// CHECK:STDOUT:   %Self => constants.%A.facet.c65
// CHECK:STDOUT:   %A.WithSelf.F.type => constants.%A.WithSelf.F.type.22c
// CHECK:STDOUT:   %A.WithSelf.F => constants.%A.WithSelf.F.06a
// CHECK:STDOUT:   %A.assoc_type => constants.%A.assoc_type.a54
// CHECK:STDOUT:   %assoc0.loc6_41.2 => constants.%assoc0.987
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf.F(constants.%X, constants.%A.facet.c65, constants.%U.67d) {
// CHECK:STDOUT:   %U.patt.loc6_9.2 => constants.%U.patt.b22
// CHECK:STDOUT:   %U.loc6_9.1 => constants.%U.67d
// CHECK:STDOUT:   %ptr.loc6_22.1 => constants.%ptr.e8f8f9.1
// CHECK:STDOUT:   %pattern_type.loc6_19 => constants.%pattern_type.4f4b84.1
// CHECK:STDOUT:   %u.param_patt.loc6_19.2 => constants.%u.param_patt.58b
// CHECK:STDOUT:   %u.patt.loc6_19.2 => constants.%u.patt.542
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.df0
// CHECK:STDOUT:   %Self => constants.%A.facet.c65
// CHECK:STDOUT:   %tuple.type.loc6_40.1 => constants.%tuple.type.966
// CHECK:STDOUT:   %tuple => constants.%tuple.7f7
// CHECK:STDOUT:   %Self.as_type.loc6_40.1 => constants.%Y
// CHECK:STDOUT:   %tuple.type.loc6_40.2 => constants.%tuple.type.316
// CHECK:STDOUT:   %.loc6_40.1 => constants.%.3c1
// CHECK:STDOUT:   %pattern_type.loc6_40 => constants.%pattern_type.178
// CHECK:STDOUT:   %return.param_patt.loc6_40.2 => constants.%return.param_patt.b96
// CHECK:STDOUT:   %return.patt.loc6_25.2 => constants.%return.patt.f2f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y.as.A.impl.F(constants.%Z) {
// CHECK:STDOUT:   %U.patt.loc16_9.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc16_9.1 => constants.%Z
// CHECK:STDOUT:   %ptr.loc16_22.1 => constants.%ptr.415
// CHECK:STDOUT:   %pattern_type.loc16_19 => constants.%pattern_type.b4b
// CHECK:STDOUT:   %u.param_patt.loc16_19.2 => constants.%u.param_patt.5c1
// CHECK:STDOUT:   %u.patt.loc16_19.2 => constants.%u.patt.5b6
// CHECK:STDOUT:   %tuple => constants.%tuple.8e4
// CHECK:STDOUT:   %tuple.type.loc16 => constants.%tuple.type.2cd
// CHECK:STDOUT:   %.loc16_37.1 => constants.%.5bc
// CHECK:STDOUT:   %pattern_type.loc16_37 => constants.%pattern_type.b33
// CHECK:STDOUT:   %return.param_patt.loc16_37.2 => constants.%return.param_patt.75d
// CHECK:STDOUT:   %return.patt.loc16_25.2 => constants.%return.patt.8bb
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc16_19 => constants.%complete_type.458
// CHECK:STDOUT:   %require_complete.loc16_25 => constants.%complete_type.dec
// CHECK:STDOUT:   %tuple.type.loc17 => constants.%tuple.type.2c5
// CHECK:STDOUT:   %.loc17_21.3 => constants.%.072
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.impl_witness.d9a
// CHECK:STDOUT:   %Copy.facet.loc17_21.3 => constants.%Copy.facet.302f
// CHECK:STDOUT:   %Copy.WithSelf.Op.type => constants.%Copy.WithSelf.Op.type.b68
// CHECK:STDOUT:   %.loc17_21.4 => constants.%.6df
// CHECK:STDOUT:   %impl.elem0.loc17_21.2 => constants.%ptr.as.Copy.impl.Op.ebe
// CHECK:STDOUT:   %specific_impl_fn.loc17_21.2 => constants.%ptr.as.Copy.impl.Op.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGeneric(constants.%T.5b7) {
// CHECK:STDOUT:   %T.patt.loc26_17.2 => constants.%T.patt.c17
// CHECK:STDOUT:   %T.loc26_17.1 => constants.%T.5b7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%X, constants.%T.5b7) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.df0
// CHECK:STDOUT:   %Self => constants.%T.5b7
// CHECK:STDOUT:   %A.WithSelf.F.type => constants.%A.WithSelf.F.type.91d
// CHECK:STDOUT:   %A.WithSelf.F => constants.%A.WithSelf.F.3f4
// CHECK:STDOUT:   %A.assoc_type => constants.%A.assoc_type.a54
// CHECK:STDOUT:   %assoc0.loc6_41.2 => constants.%assoc0.987
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%X, constants.%A.facet.318) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.df0
// CHECK:STDOUT:   %Self => constants.%A.facet.318
// CHECK:STDOUT:   %A.WithSelf.F.type => constants.%A.WithSelf.F.type.497
// CHECK:STDOUT:   %A.WithSelf.F => constants.%A.WithSelf.F.a4d
// CHECK:STDOUT:   %A.assoc_type => constants.%A.assoc_type.a54
// CHECK:STDOUT:   %assoc0.loc6_41.2 => constants.%assoc0.987
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf.F(constants.%X, constants.%A.facet.318, constants.%Z) {
// CHECK:STDOUT:   %U.patt.loc6_9.2 => constants.%U.patt.b22
// CHECK:STDOUT:   %U.loc6_9.1 => constants.%Z
// CHECK:STDOUT:   %ptr.loc6_22.1 => constants.%ptr.415
// CHECK:STDOUT:   %pattern_type.loc6_19 => constants.%pattern_type.b4b
// CHECK:STDOUT:   %u.param_patt.loc6_19.2 => constants.%u.param_patt.5c1
// CHECK:STDOUT:   %u.patt.loc6_19.2 => constants.%u.patt.5b6
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.df0
// CHECK:STDOUT:   %Self => constants.%A.facet.318
// CHECK:STDOUT:   %tuple.type.loc6_40.1 => constants.%tuple.type.966
// CHECK:STDOUT:   %tuple => constants.%tuple.575
// CHECK:STDOUT:   %Self.as_type.loc6_40.1 => constants.%T.as_type.0af
// CHECK:STDOUT:   %tuple.type.loc6_40.2 => constants.%tuple.type.c2a
// CHECK:STDOUT:   %.loc6_40.1 => constants.%.546
// CHECK:STDOUT:   %pattern_type.loc6_40 => constants.%pattern_type.0e9
// CHECK:STDOUT:   %return.param_patt.loc6_40.2 => constants.%return.param_patt.579
// CHECK:STDOUT:   %return.patt.loc6_25.2 => constants.%return.patt.62e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGeneric(constants.%facet_value) {
// CHECK:STDOUT:   %T.patt.loc26_17.2 => constants.%T.patt.c17
// CHECK:STDOUT:   %T.loc26_17.1 => constants.%facet_value
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc28_4.2 => constants.%Y
// CHECK:STDOUT:   %A.lookup_impl_witness => constants.%A.impl_witness
// CHECK:STDOUT:   %A.facet => constants.%A.facet.c65
// CHECK:STDOUT:   %A.WithSelf.F.type => constants.%A.WithSelf.F.type.22c
// CHECK:STDOUT:   %.loc28_4.3 => constants.%.7d5
// CHECK:STDOUT:   %impl.elem0.loc28_4.2 => constants.%Y.as.A.impl.F
// CHECK:STDOUT:   %specific_impl_fn.loc28_4.2 => constants.%Y.as.A.impl.F.specific_fn
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.2cd
// CHECK:STDOUT:   %require_complete => constants.%complete_type.dec
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.df9cc1.5
// CHECK:STDOUT:   %Destroy.facet.loc28_12.3 => constants.%Destroy.facet.e95
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.964
// CHECK:STDOUT:   %.loc28_12.5 => constants.%.41a
// CHECK:STDOUT:   %impl.elem0.loc28_12.2 => constants.%Destroy.Op.1a2547.5
// CHECK:STDOUT:   %specific_impl_fn.loc28_12.2 => constants.%Destroy.Op.1a2547.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf.F(constants.%X, constants.%A.facet.c65, constants.%Z) {
// CHECK:STDOUT:   %U.patt.loc6_9.2 => constants.%U.patt.b22
// CHECK:STDOUT:   %U.loc6_9.1 => constants.%Z
// CHECK:STDOUT:   %ptr.loc6_22.1 => constants.%ptr.415
// CHECK:STDOUT:   %pattern_type.loc6_19 => constants.%pattern_type.b4b
// CHECK:STDOUT:   %u.param_patt.loc6_19.2 => constants.%u.param_patt.5c1
// CHECK:STDOUT:   %u.patt.loc6_19.2 => constants.%u.patt.5b6
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.df0
// CHECK:STDOUT:   %Self => constants.%A.facet.c65
// CHECK:STDOUT:   %tuple.type.loc6_40.1 => constants.%tuple.type.966
// CHECK:STDOUT:   %tuple => constants.%tuple.1b5
// CHECK:STDOUT:   %Self.as_type.loc6_40.1 => constants.%Y
// CHECK:STDOUT:   %tuple.type.loc6_40.2 => constants.%tuple.type.2cd
// CHECK:STDOUT:   %.loc6_40.1 => constants.%.5bc
// CHECK:STDOUT:   %pattern_type.loc6_40 => constants.%pattern_type.b33
// CHECK:STDOUT:   %return.param_patt.loc6_40.2 => constants.%return.param_patt.75d
// CHECK:STDOUT:   %return.patt.loc6_25.2 => constants.%return.patt.8bb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- generic_method_more_params.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt.d47: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %A.type.55a: type = generic_interface_type @A [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %A.generic: %A.type.55a = struct_value () [concrete]
// CHECK:STDOUT:   %A.type.acd: type = facet_type <@A, @A(%T.67d)> [symbolic]
// CHECK:STDOUT:   %Self.49a: %A.type.acd = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %U.patt.b22: %pattern_type.98f = symbolic_binding_pattern U, 2 [symbolic]
// CHECK:STDOUT:   %U.4e7: type = symbolic_binding U, 2 [symbolic]
// CHECK:STDOUT:   %pattern_type.62e: type = pattern_type %U.4e7 [symbolic]
// CHECK:STDOUT:   %u.param_patt.6ef: %pattern_type.62e = value_param_pattern [symbolic]
// CHECK:STDOUT:   %u.patt.8af: %pattern_type.62e = at_binding_pattern u, %u.param_patt.6ef [symbolic]
// CHECK:STDOUT:   %tuple.type.76f: type = tuple_type (type, %A.type.acd, type) [symbolic]
// CHECK:STDOUT:   %tuple.55a: %tuple.type.76f = tuple_value (%T.67d, %Self.49a, %U.4e7) [symbolic]
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
// CHECK:STDOUT:   %V1.patt: %pattern_type.98f = symbolic_binding_pattern V1, 0 [symbolic]
// CHECK:STDOUT:   %V1: type = symbolic_binding V1, 0 [symbolic]
// CHECK:STDOUT:   %V2.patt: %pattern_type.98f = symbolic_binding_pattern V2, 1 [symbolic]
// CHECK:STDOUT:   %V2: type = symbolic_binding V2, 1 [symbolic]
// CHECK:STDOUT:   %W.patt: %pattern_type.98f = symbolic_binding_pattern W, 2 [symbolic]
// CHECK:STDOUT:   %W: type = symbolic_binding W, 2 [symbolic]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.4b9: %tuple.type.24b = tuple_value (%V1, %V2) [symbolic]
// CHECK:STDOUT:   %tuple.type.a5e: type = tuple_type (%V1, %V2) [symbolic]
// CHECK:STDOUT:   %A.type.3ad: type = facet_type <@A, @A(%W)> [symbolic]
// CHECK:STDOUT:   %A.impl_witness.756: <witness> = impl_witness @tuple.type.as.A.impl.%A.impl_witness_table, @tuple.type.as.A.impl(%V1, %V2, %W) [symbolic]
// CHECK:STDOUT:   %Self.0f8: %A.type.3ad = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %A.WithSelf.F.type.64e: type = fn_type @A.WithSelf.F, @A.WithSelf(%W, %Self.49a) [symbolic]
// CHECK:STDOUT:   %A.WithSelf.F.1fb: %A.WithSelf.F.type.64e = struct_value () [symbolic]
// CHECK:STDOUT:   %A.assoc_type.250: type = assoc_entity_type @A, @A(%W) [symbolic]
// CHECK:STDOUT:   %assoc0.6de: %A.assoc_type.250 = assoc_entity element0, @A.WithSelf.%A.WithSelf.F.decl [symbolic]
// CHECK:STDOUT:   %require_complete.7b2: <witness> = require_complete_type %A.type.3ad [symbolic]
// CHECK:STDOUT:   %U.patt.637: %pattern_type.98f = symbolic_binding_pattern U, 3 [symbolic]
// CHECK:STDOUT:   %U.ce4: type = symbolic_binding U, 3 [symbolic]
// CHECK:STDOUT:   %pattern_type.e4a: type = pattern_type %U.ce4 [symbolic]
// CHECK:STDOUT:   %u.param_patt.afe: %pattern_type.e4a = value_param_pattern [symbolic]
// CHECK:STDOUT:   %u.patt.e63: %pattern_type.e4a = at_binding_pattern u, %u.param_patt.afe [symbolic]
// CHECK:STDOUT:   %tuple.type.11f: type = tuple_type (type, %tuple.type.24b, type) [concrete]
// CHECK:STDOUT:   %tuple.117: %tuple.type.11f = tuple_value (%W, %tuple.4b9, %U.ce4) [symbolic]
// CHECK:STDOUT:   %tuple.type.897: type = tuple_type (%W, %tuple.type.a5e, %U.ce4) [symbolic]
// CHECK:STDOUT:   %.f8e: Core.Form = init_form %tuple.type.897 [symbolic]
// CHECK:STDOUT:   %pattern_type.5cc: type = pattern_type %tuple.type.897 [symbolic]
// CHECK:STDOUT:   %return.param_patt.b1c: %pattern_type.5cc = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.bad9: %pattern_type.5cc = return_slot_pattern %return.param_patt.b1c, %tuple.type.897 [symbolic]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type.44a: type = fn_type @tuple.type.as.A.impl.F, @tuple.type.as.A.impl(%V1, %V2, %W) [symbolic]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.742: %tuple.type.as.A.impl.F.type.44a = struct_value () [symbolic]
// CHECK:STDOUT:   %A.facet.ab0: %A.type.3ad = facet_value %tuple.type.a5e, (%A.impl_witness.756) [symbolic]
// CHECK:STDOUT:   %A.WithSelf.F.type.5f0: type = fn_type @A.WithSelf.F, @A.WithSelf(%W, %A.facet.ab0) [symbolic]
// CHECK:STDOUT:   %A.WithSelf.F.147: %A.WithSelf.F.type.5f0 = struct_value () [symbolic]
// CHECK:STDOUT:   %tuple.type.a90: type = tuple_type (type, %A.type.3ad, type) [symbolic]
// CHECK:STDOUT:   %tuple.059: %tuple.type.a90 = tuple_value (%W, %A.facet.ab0, %U.ce4) [symbolic]
// CHECK:STDOUT:   %require_complete.5a4: <witness> = require_complete_type %U.ce4 [symbolic]
// CHECK:STDOUT:   %require_complete.da0: <witness> = require_complete_type %tuple.type.897 [symbolic]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.specific_fn.37b: <specific function> = specific_function %tuple.type.as.A.impl.F.742, @tuple.type.as.A.impl.F(%V1, %V2, %W, %U.ce4) [symbolic]
// CHECK:STDOUT:   %Call.type: type = fn_type @Call [concrete]
// CHECK:STDOUT:   %Call: %Call.type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.8db: %tuple.type.24b = tuple_value (%Y1, %Y2) [concrete]
// CHECK:STDOUT:   %tuple.type.59f: type = tuple_type (%Y1, %Y2) [concrete]
// CHECK:STDOUT:   %A.type.df0: type = facet_type <@A, @A(%X)> [concrete]
// CHECK:STDOUT:   %A.impl_witness.6b7: <witness> = impl_witness @tuple.type.as.A.impl.%A.impl_witness_table, @tuple.type.as.A.impl(%Y1, %Y2, %X) [concrete]
// CHECK:STDOUT:   %Self.891: %A.type.df0 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %A.WithSelf.F.type.ad4: type = fn_type @A.WithSelf.F, @A.WithSelf(%X, %Self.49a) [symbolic]
// CHECK:STDOUT:   %A.WithSelf.F.1e2: %A.WithSelf.F.type.ad4 = struct_value () [symbolic]
// CHECK:STDOUT:   %A.assoc_type.a54: type = assoc_entity_type @A, @A(%X) [concrete]
// CHECK:STDOUT:   %assoc0.987: %A.assoc_type.a54 = assoc_entity element0, @A.WithSelf.%A.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %complete_type.e33: <witness> = complete_type_witness %A.type.df0 [concrete]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type.745: type = fn_type @tuple.type.as.A.impl.F, @tuple.type.as.A.impl(%Y1, %Y2, %X) [concrete]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.749: %tuple.type.as.A.impl.F.type.745 = struct_value () [concrete]
// CHECK:STDOUT:   %A.facet.682: %A.type.df0 = facet_value %tuple.type.59f, (%A.impl_witness.6b7) [concrete]
// CHECK:STDOUT:   %A.WithSelf.F.type.243: type = fn_type @A.WithSelf.F, @A.WithSelf(%X, %A.facet.682) [concrete]
// CHECK:STDOUT:   %A.WithSelf.F.5b5: %A.WithSelf.F.type.243 = struct_value () [concrete]
// CHECK:STDOUT:   %.d9b: type = fn_type_with_self_type %A.WithSelf.F.type.243, %A.facet.682 [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.34e: type = pattern_type %Z [concrete]
// CHECK:STDOUT:   %u.param_patt.a3e: %pattern_type.34e = value_param_pattern [concrete]
// CHECK:STDOUT:   %u.patt.a51: %pattern_type.34e = at_binding_pattern u, %u.param_patt.a3e [concrete]
// CHECK:STDOUT:   %tuple.1a1: %tuple.type.11f = tuple_value (%X, %tuple.8db, %Z) [concrete]
// CHECK:STDOUT:   %tuple.type.d40: type = tuple_type (%X, %tuple.type.59f, %Z) [concrete]
// CHECK:STDOUT:   %.1a9: Core.Form = init_form %tuple.type.d40 [concrete]
// CHECK:STDOUT:   %pattern_type.6a8: type = pattern_type %tuple.type.d40 [concrete]
// CHECK:STDOUT:   %return.param_patt.fb7: %pattern_type.6a8 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.badc: %pattern_type.6a8 = return_slot_pattern %return.param_patt.fb7, %tuple.type.d40 [concrete]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.specific_fn.981: <specific function> = specific_function %tuple.type.as.A.impl.F.749, @tuple.type.as.A.impl.F(%Y1, %Y2, %X, %Z) [concrete]
// CHECK:STDOUT:   %Z.val: %Z = struct_value () [concrete]
// CHECK:STDOUT:   %.7dc: ref %Z = temporary invalid, %Z.val [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.5: type = fn_type @Destroy.Op.loc24_39.5 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.5: %Destroy.Op.type.1d8f74.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df9cc1.5: <witness> = custom_witness (%Destroy.Op.1a2547.5), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.6: type = fn_type @Destroy.Op.loc24_39.6 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.6: %Destroy.Op.type.1d8f74.6 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.7: type = fn_type @Destroy.Op.loc24_39.7 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.7: %Destroy.Op.type.1d8f74.7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df9cc1.7: <witness> = custom_witness (%Destroy.Op.1a2547.7), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.50b: %Destroy.type = facet_value %tuple.type.d40, (%custom_witness.df9cc1.7) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.43f: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.50b) [concrete]
// CHECK:STDOUT:   %.3d3: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.43f, %Destroy.facet.50b [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.7dc, %Destroy.Op.1a2547.6 [concrete]
// CHECK:STDOUT:   %BitAndWith.type.a06: type = generic_interface_type @BitAndWith [concrete]
// CHECK:STDOUT:   %BitAndWith.generic: %BitAndWith.type.a06 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndWith.type.802: type = facet_type <@BitAndWith, @BitAndWith(type)> [concrete]
// CHECK:STDOUT:   %BitAndWith.impl_witness: <witness> = impl_witness imports.%BitAndWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %BitAndWith.facet: %BitAndWith.type.802 = facet_value type, (%BitAndWith.impl_witness) [concrete]
// CHECK:STDOUT:   %BitAndWith.WithSelf.Op.type.38f: type = fn_type @BitAndWith.WithSelf.Op, @BitAndWith.WithSelf(type, %BitAndWith.facet) [concrete]
// CHECK:STDOUT:   %.d56: type = fn_type_with_self_type %BitAndWith.WithSelf.Op.type.38f, %BitAndWith.facet [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.type: type = fn_type @type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op: %type.as.BitAndWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.bound: <bound method> = bound_method %A.type.df0, %type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@A, @A(%X) & @Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.871: type = pattern_type %facet_type [concrete]
// CHECK:STDOUT:   %T.patt.c17: %pattern_type.871 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.5b7: %facet_type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %CallGeneric.type: type = fn_type @CallGeneric [concrete]
// CHECK:STDOUT:   %CallGeneric: %CallGeneric.type = struct_value () [concrete]
// CHECK:STDOUT:   %A.WithSelf.F.type.91d: type = fn_type @A.WithSelf.F, @A.WithSelf(%X, %T.5b7) [symbolic]
// CHECK:STDOUT:   %A.WithSelf.F.3f4: %A.WithSelf.F.type.91d = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.5b7 [symbolic]
// CHECK:STDOUT:   %A.lookup_impl_witness: <witness> = lookup_impl_witness %T.5b7, @A, @A(%X) [symbolic]
// CHECK:STDOUT:   %A.facet.318: %A.type.df0 = facet_value %T.as_type, (%A.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %A.WithSelf.F.type.497: type = fn_type @A.WithSelf.F, @A.WithSelf(%X, %A.facet.318) [symbolic]
// CHECK:STDOUT:   %A.WithSelf.F.a4d: %A.WithSelf.F.type.497 = struct_value () [symbolic]
// CHECK:STDOUT:   %.151: type = fn_type_with_self_type %A.WithSelf.F.type.497, %A.facet.318 [symbolic]
// CHECK:STDOUT:   %impl.elem0.1a3: %.151 = impl_witness_access %A.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %tuple.type.966: type = tuple_type (type, %A.type.df0, type) [concrete]
// CHECK:STDOUT:   %tuple.b58: %tuple.type.966 = tuple_value (%X, %A.facet.318, %Z) [symbolic]
// CHECK:STDOUT:   %tuple.type.036: type = tuple_type (%X, %T.as_type, %Z) [symbolic]
// CHECK:STDOUT:   %.772: Core.Form = init_form %tuple.type.036 [symbolic]
// CHECK:STDOUT:   %pattern_type.8f9: type = pattern_type %tuple.type.036 [symbolic]
// CHECK:STDOUT:   %return.param_patt.cc2: %pattern_type.8f9 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.eed: %pattern_type.8f9 = return_slot_pattern %return.param_patt.cc2, %tuple.type.036 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.74d: <specific function> = specific_impl_function %impl.elem0.1a3, @A.WithSelf.F(%X, %A.facet.318, %Z) [symbolic]
// CHECK:STDOUT:   %require_complete.4e1: <witness> = require_complete_type %tuple.type.036 [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.8dc: <witness> = lookup_impl_witness %tuple.type.036, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.fef: %Destroy.type = facet_value %tuple.type.036, (%Destroy.lookup_impl_witness.8dc) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.937: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.fef) [symbolic]
// CHECK:STDOUT:   %.93b: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.937, %Destroy.facet.fef [symbolic]
// CHECK:STDOUT:   %impl.elem0.743: %.93b = impl_witness_access %Destroy.lookup_impl_witness.8dc, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.618: <specific function> = specific_impl_function %impl.elem0.743, @Destroy.WithSelf.Op(%Destroy.facet.fef) [symbolic]
// CHECK:STDOUT:   %CallIndirect.type: type = fn_type @CallIndirect [concrete]
// CHECK:STDOUT:   %CallIndirect: %CallIndirect.type = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value: %facet_type = facet_value %tuple.type.59f, (%A.impl_witness.6b7, %custom_witness.df9cc1.5) [concrete]
// CHECK:STDOUT:   %CallGeneric.specific_fn: <specific function> = specific_function %CallGeneric, @CallGeneric(%facet_value) [concrete]
// CHECK:STDOUT:   %complete_type.8f6: <witness> = complete_type_witness %tuple.type.d40 [concrete]
// CHECK:STDOUT:   %tuple.92c: %tuple.type.966 = tuple_value (%X, %A.facet.682, %Z) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .BitAndWith = %Core.BitAndWith
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
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
// CHECK:STDOUT:     %T.patt.loc5_14.1: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc5_14.2 (constants.%T.patt.d47)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5_17.1: type = splice_block %.loc5_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc5_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc5_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc5_14.1 (constants.%T.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.decl: type = class_decl @X [concrete = constants.%X] {} {}
// CHECK:STDOUT:   %Y1.decl: type = class_decl @Y1 [concrete = constants.%Y1] {} {}
// CHECK:STDOUT:   %Y2.decl: type = class_decl @Y2 [concrete = constants.%Y2] {} {}
// CHECK:STDOUT:   %Z.decl: type = class_decl @Z [concrete = constants.%Z] {} {}
// CHECK:STDOUT:   impl_decl @tuple.type.as.A.impl [concrete] {
// CHECK:STDOUT:     %V1.patt.loc14_16.1: %pattern_type.98f = symbolic_binding_pattern V1, 0 [symbolic = %V1.patt.loc14_16.2 (constants.%V1.patt)]
// CHECK:STDOUT:     %V2.patt.loc14_27.1: %pattern_type.98f = symbolic_binding_pattern V2, 1 [symbolic = %V2.patt.loc14_27.2 (constants.%V2.patt)]
// CHECK:STDOUT:     %W.patt.loc14_37.1: %pattern_type.98f = symbolic_binding_pattern W, 2 [symbolic = %W.patt.loc14_37.2 (constants.%W.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %V1.ref: type = name_ref V1, %V1.loc14_16.2 [symbolic = %V1.loc14_16.1 (constants.%V1)]
// CHECK:STDOUT:     %V2.ref: type = name_ref V2, %V2.loc14_27.2 [symbolic = %V2.loc14_27.1 (constants.%V2)]
// CHECK:STDOUT:     %.loc14_53.1: %tuple.type.24b = tuple_literal (%V1.ref, %V2.ref) [symbolic = %tuple (constants.%tuple.4b9)]
// CHECK:STDOUT:     %.loc14_53.2: type = converted %.loc14_53.1, constants.%tuple.type.a5e [symbolic = %tuple.type (constants.%tuple.type.a5e)]
// CHECK:STDOUT:     %A.ref: %A.type.55a = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:     %W.ref: type = name_ref W, %W.loc14_37.2 [symbolic = %W.loc14_37.1 (constants.%W)]
// CHECK:STDOUT:     %A.type.loc14_61.2: type = facet_type <@A, @A(constants.%W)> [symbolic = %A.type.loc14_61.1 (constants.%A.type.3ad)]
// CHECK:STDOUT:     %.loc14_19.1: type = splice_block %.loc14_19.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.loc14_16: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc14_19.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %V1.loc14_16.2: type = symbolic_binding V1, 0 [symbolic = %V1.loc14_16.1 (constants.%V1)]
// CHECK:STDOUT:     %.loc14_30.1: type = splice_block %.loc14_30.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.loc14_27: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc14_30.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %V2.loc14_27.2: type = symbolic_binding V2, 1 [symbolic = %V2.loc14_27.1 (constants.%V2)]
// CHECK:STDOUT:     %.loc14_40.1: type = splice_block %.loc14_40.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.loc14_37: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc14_40.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %W.loc14_37.2: type = symbolic_binding W, 2 [symbolic = %W.loc14_37.1 (constants.%W)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Call.decl: %Call.type = fn_decl @Call [concrete = constants.%Call] {} {}
// CHECK:STDOUT:   %CallGeneric.decl: %CallGeneric.type = fn_decl @CallGeneric [concrete = constants.%CallGeneric] {
// CHECK:STDOUT:     %T.patt.loc27_17.1: %pattern_type.871 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc27_17.2 (constants.%T.patt.c17)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc27_25.1: type = splice_block %.loc27_25.3 [concrete = constants.%facet_type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %A.ref: %A.type.55a = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:       %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:       %A.type: type = facet_type <@A, @A(constants.%X)> [concrete = constants.%A.type.df0]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:       %impl.elem0.loc27: %.d56 = impl_witness_access constants.%BitAndWith.impl_witness, element0 [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:       %bound_method.loc27: <bound method> = bound_method %A.type, %impl.elem0.loc27 [concrete = constants.%type.as.BitAndWith.impl.Op.bound]
// CHECK:STDOUT:       %type.as.BitAndWith.impl.Op.call: init type = call %bound_method.loc27(%A.type, %Destroy.ref) [concrete = constants.%facet_type]
// CHECK:STDOUT:       %.loc27_25.2: type = value_of_initializer %type.as.BitAndWith.impl.Op.call [concrete = constants.%facet_type]
// CHECK:STDOUT:       %.loc27_25.3: type = converted %type.as.BitAndWith.impl.Op.call, %.loc27_25.2 [concrete = constants.%facet_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc27_17.2: %facet_type = symbolic_binding T, 0 [symbolic = %T.loc27_17.1 (constants.%T.5b7)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallIndirect.decl: %CallIndirect.type = fn_decl @CallIndirect [concrete = constants.%CallIndirect] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @A(%T.loc5_14.2: type) {
// CHECK:STDOUT:   %T.patt.loc5_14.2: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc5_14.2 (constants.%T.patt.d47)]
// CHECK:STDOUT:   %T.loc5_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc5_14.1 (constants.%T.67d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.type: type = facet_type <@A, @A(%T.loc5_14.1)> [symbolic = %A.type (constants.%A.type.acd)]
// CHECK:STDOUT:   %Self.loc5_23.2: @A.%A.type (%A.type.acd) = symbolic_binding Self, 1 [symbolic = %Self.loc5_23.2 (constants.%Self.49a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc5_23.1: @A.%A.type (%A.type.acd) = symbolic_binding Self, 1 [symbolic = %Self.loc5_23.2 (constants.%Self.49a)]
// CHECK:STDOUT:     %A.WithSelf.decl = interface_with_self_decl @A [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %A.WithSelf.F.decl: @A.WithSelf.%A.WithSelf.F.type (%A.WithSelf.F.type.08d) = fn_decl @A.WithSelf.F [symbolic = @A.WithSelf.%A.WithSelf.F (constants.%A.WithSelf.F.3fa)] {
// CHECK:STDOUT:       %U.patt.loc6_9.1: %pattern_type.98f = symbolic_binding_pattern U, 2 [symbolic = %U.patt.loc6_9.2 (constants.%U.patt.b22)]
// CHECK:STDOUT:       %u.param_patt.loc6_19.1: @A.WithSelf.F.%pattern_type.loc6_19 (%pattern_type.62e) = value_param_pattern [symbolic = %u.param_patt.loc6_19.2 (constants.%u.param_patt.6ef)]
// CHECK:STDOUT:       %u.patt.loc6_19.1: @A.WithSelf.F.%pattern_type.loc6_19 (%pattern_type.62e) = at_binding_pattern u, %u.param_patt.loc6_19.1 [symbolic = %u.patt.loc6_19.2 (constants.%u.patt.8af)]
// CHECK:STDOUT:       %return.param_patt.loc6_38.1: @A.WithSelf.F.%pattern_type.loc6_38 (%pattern_type.d6c) = out_param_pattern [symbolic = %return.param_patt.loc6_38.2 (constants.%return.param_patt.52a)]
// CHECK:STDOUT:       %return.patt.loc6_24.1: @A.WithSelf.F.%pattern_type.loc6_38 (%pattern_type.d6c) = return_slot_pattern %return.param_patt.loc6_38.1, %.loc6_38.4 [symbolic = %return.patt.loc6_24.2 (constants.%return.patt.808)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @A.%T.loc5_14.2 [symbolic = %T (constants.%T.67d)]
// CHECK:STDOUT:       %.loc6_31: @A.WithSelf.F.%A.type (%A.type.acd) = specific_constant @A.%Self.loc5_23.1, @A(constants.%T.67d) [symbolic = %Self (constants.%Self.49a)]
// CHECK:STDOUT:       %Self.ref: @A.WithSelf.F.%A.type (%A.type.acd) = name_ref Self, %.loc6_31 [symbolic = %Self (constants.%Self.49a)]
// CHECK:STDOUT:       %U.ref.loc6_37: type = name_ref U, %U.loc6_9.2 [symbolic = %U.loc6_9.1 (constants.%U.4e7)]
// CHECK:STDOUT:       %.loc6_38.2: @A.WithSelf.F.%tuple.type.loc6_38.1 (%tuple.type.76f) = tuple_literal (%T.ref, %Self.ref, %U.ref.loc6_37) [symbolic = %tuple (constants.%tuple.55a)]
// CHECK:STDOUT:       %Self.as_type.loc6_38.2: type = facet_access_type constants.%Self.49a [symbolic = %Self.as_type.loc6_38.1 (constants.%Self.as_type.763)]
// CHECK:STDOUT:       %.loc6_38.3: type = converted constants.%Self.49a, %Self.as_type.loc6_38.2 [symbolic = %Self.as_type.loc6_38.1 (constants.%Self.as_type.763)]
// CHECK:STDOUT:       %.loc6_38.4: type = converted %.loc6_38.2, constants.%tuple.type.d73 [symbolic = %tuple.type.loc6_38.2 (constants.%tuple.type.d73)]
// CHECK:STDOUT:       %.loc6_38.5: Core.Form = init_form %.loc6_38.4 [symbolic = %.loc6_38.1 (constants.%.bc3)]
// CHECK:STDOUT:       %.loc6_12.1: type = splice_block %.loc6_12.2 [concrete = type] {
// CHECK:STDOUT:         %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:         %.loc6_12.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %U.loc6_9.2: type = symbolic_binding U, 2 [symbolic = %U.loc6_9.1 (constants.%U.4e7)]
// CHECK:STDOUT:       %u.param: @A.WithSelf.F.%U.loc6_9.1 (%U.4e7) = value_param call_param0
// CHECK:STDOUT:       %U.ref.loc6_21: type = name_ref U, %U.loc6_9.2 [symbolic = %U.loc6_9.1 (constants.%U.4e7)]
// CHECK:STDOUT:       %u: @A.WithSelf.F.%U.loc6_9.1 (%U.4e7) = wrapper_binding u, %u.param
// CHECK:STDOUT:       %return.param: ref @A.WithSelf.F.%tuple.type.loc6_38.2 (%tuple.type.d73) = out_param call_param1
// CHECK:STDOUT:       %return: ref @A.WithSelf.F.%tuple.type.loc6_38.2 (%tuple.type.d73) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc6_39.1: @A.WithSelf.%A.assoc_type (%A.assoc_type.447) = assoc_entity element0, %A.WithSelf.F.decl [symbolic = %assoc0.loc6_39.2 (constants.%assoc0.d4b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc5_23.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .F = @A.WithSelf.%assoc0.loc6_39.1
// CHECK:STDOUT:     .W = <poisoned>
// CHECK:STDOUT:     .V1 = <poisoned>
// CHECK:STDOUT:     .V2 = <poisoned>
// CHECK:STDOUT:     witness = (@A.WithSelf.%A.WithSelf.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @tuple.type.as.A.impl(%V1.loc14_16.2: type, %V2.loc14_27.2: type, %W.loc14_37.2: type) {
// CHECK:STDOUT:   %V1.patt.loc14_16.2: %pattern_type.98f = symbolic_binding_pattern V1, 0 [symbolic = %V1.patt.loc14_16.2 (constants.%V1.patt)]
// CHECK:STDOUT:   %V1.loc14_16.1: type = symbolic_binding V1, 0 [symbolic = %V1.loc14_16.1 (constants.%V1)]
// CHECK:STDOUT:   %V2.patt.loc14_27.2: %pattern_type.98f = symbolic_binding_pattern V2, 1 [symbolic = %V2.patt.loc14_27.2 (constants.%V2.patt)]
// CHECK:STDOUT:   %V2.loc14_27.1: type = symbolic_binding V2, 1 [symbolic = %V2.loc14_27.1 (constants.%V2)]
// CHECK:STDOUT:   %W.patt.loc14_37.2: %pattern_type.98f = symbolic_binding_pattern W, 2 [symbolic = %W.patt.loc14_37.2 (constants.%W.patt)]
// CHECK:STDOUT:   %W.loc14_37.1: type = symbolic_binding W, 2 [symbolic = %W.loc14_37.1 (constants.%W)]
// CHECK:STDOUT:   %tuple: %tuple.type.24b = tuple_value (%V1.loc14_16.1, %V2.loc14_27.1) [symbolic = %tuple (constants.%tuple.4b9)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%V1.loc14_16.1, %V2.loc14_27.1) [symbolic = %tuple.type (constants.%tuple.type.a5e)]
// CHECK:STDOUT:   %A.type.loc14_61.1: type = facet_type <@A, @A(%W.loc14_37.1)> [symbolic = %A.type.loc14_61.1 (constants.%A.type.3ad)]
// CHECK:STDOUT:   %A.impl_witness.loc14_63.2: <witness> = impl_witness %A.impl_witness_table, @tuple.type.as.A.impl(%V1.loc14_16.1, %V2.loc14_27.1, %W.loc14_37.1) [symbolic = %A.impl_witness.loc14_63.2 (constants.%A.impl_witness.756)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %A.type.loc14_61.1 [symbolic = %require_complete (constants.%require_complete.7b2)]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type: type = fn_type @tuple.type.as.A.impl.F, @tuple.type.as.A.impl(%V1.loc14_16.1, %V2.loc14_27.1, %W.loc14_37.1) [symbolic = %tuple.type.as.A.impl.F.type (constants.%tuple.type.as.A.impl.F.type.44a)]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F: @tuple.type.as.A.impl.%tuple.type.as.A.impl.F.type (%tuple.type.as.A.impl.F.type.44a) = struct_value () [symbolic = %tuple.type.as.A.impl.F (constants.%tuple.type.as.A.impl.F.742)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc14_53.2 as %A.type.loc14_61.2 {
// CHECK:STDOUT:     %tuple.type.as.A.impl.F.decl: @tuple.type.as.A.impl.%tuple.type.as.A.impl.F.type (%tuple.type.as.A.impl.F.type.44a) = fn_decl @tuple.type.as.A.impl.F [symbolic = @tuple.type.as.A.impl.%tuple.type.as.A.impl.F (constants.%tuple.type.as.A.impl.F.742)] {
// CHECK:STDOUT:       %U.patt.loc17_9.1: %pattern_type.98f = symbolic_binding_pattern U, 3 [symbolic = %U.patt.loc17_9.2 (constants.%U.patt.637)]
// CHECK:STDOUT:       %u.param_patt.loc17_19.1: @tuple.type.as.A.impl.F.%pattern_type.loc17_19 (%pattern_type.e4a) = value_param_pattern [symbolic = %u.param_patt.loc17_19.2 (constants.%u.param_patt.afe)]
// CHECK:STDOUT:       %u.patt.loc17_19.1: @tuple.type.as.A.impl.F.%pattern_type.loc17_19 (%pattern_type.e4a) = at_binding_pattern u, %u.param_patt.loc17_19.1 [symbolic = %u.patt.loc17_19.2 (constants.%u.patt.e63)]
// CHECK:STDOUT:       %return.param_patt.loc17_42.1: @tuple.type.as.A.impl.F.%pattern_type.loc17_42 (%pattern_type.5cc) = out_param_pattern [symbolic = %return.param_patt.loc17_42.2 (constants.%return.param_patt.b1c)]
// CHECK:STDOUT:       %return.patt.loc17_24.1: @tuple.type.as.A.impl.F.%pattern_type.loc17_42 (%pattern_type.5cc) = return_slot_pattern %return.param_patt.loc17_42.1, %.loc17_42.5 [symbolic = %return.patt.loc17_24.2 (constants.%return.patt.bad9)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %W.ref: type = name_ref W, @tuple.type.as.A.impl.%W.loc14_37.2 [symbolic = %W (constants.%W)]
// CHECK:STDOUT:       %V1.ref: type = name_ref V1, @tuple.type.as.A.impl.%V1.loc14_16.2 [symbolic = %V1 (constants.%V1)]
// CHECK:STDOUT:       %V2.ref: type = name_ref V2, @tuple.type.as.A.impl.%V2.loc14_27.2 [symbolic = %V2 (constants.%V2)]
// CHECK:STDOUT:       %.loc17_38: %tuple.type.24b = tuple_literal (%V1.ref, %V2.ref) [symbolic = %tuple.loc17_38 (constants.%tuple.4b9)]
// CHECK:STDOUT:       %U.ref.loc17_41: type = name_ref U, %U.loc17_9.2 [symbolic = %U.loc17_9.1 (constants.%U.ce4)]
// CHECK:STDOUT:       %.loc17_42.3: %tuple.type.11f = tuple_literal (%W.ref, %.loc17_38, %U.ref.loc17_41) [symbolic = %tuple.loc17_42 (constants.%tuple.117)]
// CHECK:STDOUT:       %.loc17_42.4: type = converted constants.%tuple.4b9, constants.%tuple.type.a5e [symbolic = %tuple.type.loc17_42.1 (constants.%tuple.type.a5e)]
// CHECK:STDOUT:       %.loc17_42.5: type = converted %.loc17_42.3, constants.%tuple.type.897 [symbolic = %tuple.type.loc17_42.2 (constants.%tuple.type.897)]
// CHECK:STDOUT:       %.loc17_42.6: Core.Form = init_form %.loc17_42.5 [symbolic = %.loc17_42.2 (constants.%.f8e)]
// CHECK:STDOUT:       %.loc17_12.1: type = splice_block %.loc17_12.2 [concrete = type] {
// CHECK:STDOUT:         %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:         %.loc17_12.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %U.loc17_9.2: type = symbolic_binding U, 3 [symbolic = %U.loc17_9.1 (constants.%U.ce4)]
// CHECK:STDOUT:       %u.param: @tuple.type.as.A.impl.F.%U.loc17_9.1 (%U.ce4) = value_param call_param0
// CHECK:STDOUT:       %U.ref.loc17_21: type = name_ref U, %U.loc17_9.2 [symbolic = %U.loc17_9.1 (constants.%U.ce4)]
// CHECK:STDOUT:       %u: @tuple.type.as.A.impl.F.%U.loc17_9.1 (%U.ce4) = wrapper_binding u, %u.param
// CHECK:STDOUT:       %return.param: ref @tuple.type.as.A.impl.F.%tuple.type.loc17_42.2 (%tuple.type.897) = out_param call_param1
// CHECK:STDOUT:       %return: ref @tuple.type.as.A.impl.F.%tuple.type.loc17_42.2 (%tuple.type.897) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %A.impl_witness_table = impl_witness_table (%tuple.type.as.A.impl.F.decl), @tuple.type.as.A.impl [concrete]
// CHECK:STDOUT:     %A.impl_witness.loc14_63.1: <witness> = impl_witness %A.impl_witness_table, @tuple.type.as.A.impl(constants.%V1, constants.%V2, constants.%W) [symbolic = %A.impl_witness.loc14_63.2 (constants.%A.impl_witness.756)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .W = <poisoned>
// CHECK:STDOUT:     .V1 = <poisoned>
// CHECK:STDOUT:     .V2 = <poisoned>
// CHECK:STDOUT:     .F = %tuple.type.as.A.impl.F.decl
// CHECK:STDOUT:     extend %A.type.loc14_61.2
// CHECK:STDOUT:     witness = %A.impl_witness.loc14_63.1
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
// CHECK:STDOUT: generic fn @A.WithSelf.F(@A.%T.loc5_14.2: type, @A.%Self.loc5_23.1: @A.%A.type (%A.type.acd), %U.loc6_9.2: type) {
// CHECK:STDOUT:   %U.patt.loc6_9.2: %pattern_type.98f = symbolic_binding_pattern U, 2 [symbolic = %U.patt.loc6_9.2 (constants.%U.patt.b22)]
// CHECK:STDOUT:   %U.loc6_9.1: type = symbolic_binding U, 2 [symbolic = %U.loc6_9.1 (constants.%U.4e7)]
// CHECK:STDOUT:   %pattern_type.loc6_19: type = pattern_type %U.loc6_9.1 [symbolic = %pattern_type.loc6_19 (constants.%pattern_type.62e)]
// CHECK:STDOUT:   %u.param_patt.loc6_19.2: @A.WithSelf.F.%pattern_type.loc6_19 (%pattern_type.62e) = value_param_pattern [symbolic = %u.param_patt.loc6_19.2 (constants.%u.param_patt.6ef)]
// CHECK:STDOUT:   %u.patt.loc6_19.2: @A.WithSelf.F.%pattern_type.loc6_19 (%pattern_type.62e) = at_binding_pattern u, %u.param_patt.loc6_19.2 [symbolic = %u.patt.loc6_19.2 (constants.%u.patt.8af)]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T.67d)]
// CHECK:STDOUT:   %A.type: type = facet_type <@A, @A(%T)> [symbolic = %A.type (constants.%A.type.acd)]
// CHECK:STDOUT:   %Self: @A.WithSelf.F.%A.type (%A.type.acd) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.49a)]
// CHECK:STDOUT:   %tuple.type.loc6_38.1: type = tuple_type (type, %A.type, type) [symbolic = %tuple.type.loc6_38.1 (constants.%tuple.type.76f)]
// CHECK:STDOUT:   %tuple: @A.WithSelf.F.%tuple.type.loc6_38.1 (%tuple.type.76f) = tuple_value (%T, %Self, %U.loc6_9.1) [symbolic = %tuple (constants.%tuple.55a)]
// CHECK:STDOUT:   %Self.as_type.loc6_38.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc6_38.1 (constants.%Self.as_type.763)]
// CHECK:STDOUT:   %tuple.type.loc6_38.2: type = tuple_type (%T, %Self.as_type.loc6_38.1, %U.loc6_9.1) [symbolic = %tuple.type.loc6_38.2 (constants.%tuple.type.d73)]
// CHECK:STDOUT:   %.loc6_38.1: Core.Form = init_form %tuple.type.loc6_38.2 [symbolic = %.loc6_38.1 (constants.%.bc3)]
// CHECK:STDOUT:   %pattern_type.loc6_38: type = pattern_type %tuple.type.loc6_38.2 [symbolic = %pattern_type.loc6_38 (constants.%pattern_type.d6c)]
// CHECK:STDOUT:   %return.param_patt.loc6_38.2: @A.WithSelf.F.%pattern_type.loc6_38 (%pattern_type.d6c) = out_param_pattern [symbolic = %return.param_patt.loc6_38.2 (constants.%return.param_patt.52a)]
// CHECK:STDOUT:   %return.patt.loc6_24.2: @A.WithSelf.F.%pattern_type.loc6_38 (%pattern_type.d6c) = return_slot_pattern %return.param_patt.loc6_38.2, %tuple.type.loc6_38.2 [symbolic = %return.patt.loc6_24.2 (constants.%return.patt.808)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param: @A.WithSelf.F.%U.loc6_9.1 (%U.4e7)) -> out %return.param: @A.WithSelf.F.%tuple.type.loc6_38.2 (%tuple.type.d73);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @tuple.type.as.A.impl.F(@tuple.type.as.A.impl.%V1.loc14_16.2: type, @tuple.type.as.A.impl.%V2.loc14_27.2: type, @tuple.type.as.A.impl.%W.loc14_37.2: type, %U.loc17_9.2: type) {
// CHECK:STDOUT:   %U.patt.loc17_9.2: %pattern_type.98f = symbolic_binding_pattern U, 3 [symbolic = %U.patt.loc17_9.2 (constants.%U.patt.637)]
// CHECK:STDOUT:   %U.loc17_9.1: type = symbolic_binding U, 3 [symbolic = %U.loc17_9.1 (constants.%U.ce4)]
// CHECK:STDOUT:   %pattern_type.loc17_19: type = pattern_type %U.loc17_9.1 [symbolic = %pattern_type.loc17_19 (constants.%pattern_type.e4a)]
// CHECK:STDOUT:   %u.param_patt.loc17_19.2: @tuple.type.as.A.impl.F.%pattern_type.loc17_19 (%pattern_type.e4a) = value_param_pattern [symbolic = %u.param_patt.loc17_19.2 (constants.%u.param_patt.afe)]
// CHECK:STDOUT:   %u.patt.loc17_19.2: @tuple.type.as.A.impl.F.%pattern_type.loc17_19 (%pattern_type.e4a) = at_binding_pattern u, %u.param_patt.loc17_19.2 [symbolic = %u.patt.loc17_19.2 (constants.%u.patt.e63)]
// CHECK:STDOUT:   %W: type = symbolic_binding W, 2 [symbolic = %W (constants.%W)]
// CHECK:STDOUT:   %V1: type = symbolic_binding V1, 0 [symbolic = %V1 (constants.%V1)]
// CHECK:STDOUT:   %V2: type = symbolic_binding V2, 1 [symbolic = %V2 (constants.%V2)]
// CHECK:STDOUT:   %tuple.loc17_38: %tuple.type.24b = tuple_value (%V1, %V2) [symbolic = %tuple.loc17_38 (constants.%tuple.4b9)]
// CHECK:STDOUT:   %tuple.loc17_42: %tuple.type.11f = tuple_value (%W, %tuple.loc17_38, %U.loc17_9.1) [symbolic = %tuple.loc17_42 (constants.%tuple.117)]
// CHECK:STDOUT:   %tuple.type.loc17_42.1: type = tuple_type (%V1, %V2) [symbolic = %tuple.type.loc17_42.1 (constants.%tuple.type.a5e)]
// CHECK:STDOUT:   %tuple.type.loc17_42.2: type = tuple_type (%W, %tuple.type.loc17_42.1, %U.loc17_9.1) [symbolic = %tuple.type.loc17_42.2 (constants.%tuple.type.897)]
// CHECK:STDOUT:   %.loc17_42.2: Core.Form = init_form %tuple.type.loc17_42.2 [symbolic = %.loc17_42.2 (constants.%.f8e)]
// CHECK:STDOUT:   %pattern_type.loc17_42: type = pattern_type %tuple.type.loc17_42.2 [symbolic = %pattern_type.loc17_42 (constants.%pattern_type.5cc)]
// CHECK:STDOUT:   %return.param_patt.loc17_42.2: @tuple.type.as.A.impl.F.%pattern_type.loc17_42 (%pattern_type.5cc) = out_param_pattern [symbolic = %return.param_patt.loc17_42.2 (constants.%return.param_patt.b1c)]
// CHECK:STDOUT:   %return.patt.loc17_24.2: @tuple.type.as.A.impl.F.%pattern_type.loc17_42 (%pattern_type.5cc) = return_slot_pattern %return.param_patt.loc17_42.2, %tuple.type.loc17_42.2 [symbolic = %return.patt.loc17_24.2 (constants.%return.patt.bad9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc17_19: <witness> = require_complete_type %U.loc17_9.1 [symbolic = %require_complete.loc17_19 (constants.%require_complete.5a4)]
// CHECK:STDOUT:   %require_complete.loc17_24: <witness> = require_complete_type %tuple.type.loc17_42.2 [symbolic = %require_complete.loc17_24 (constants.%require_complete.da0)]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type: type = fn_type @tuple.type.as.A.impl.F, @tuple.type.as.A.impl(%V1, %V2, %W) [symbolic = %tuple.type.as.A.impl.F.type (constants.%tuple.type.as.A.impl.F.type.44a)]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F: @tuple.type.as.A.impl.F.%tuple.type.as.A.impl.F.type (%tuple.type.as.A.impl.F.type.44a) = struct_value () [symbolic = %tuple.type.as.A.impl.F (constants.%tuple.type.as.A.impl.F.742)]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.specific_fn.loc18_12.2: <specific function> = specific_function %tuple.type.as.A.impl.F, @tuple.type.as.A.impl.F(%V1, %V2, %W, %U.loc17_9.1) [symbolic = %tuple.type.as.A.impl.F.specific_fn.loc18_12.2 (constants.%tuple.type.as.A.impl.F.specific_fn.37b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param: @tuple.type.as.A.impl.F.%U.loc17_9.1 (%U.ce4)) -> out %return.param: @tuple.type.as.A.impl.F.%tuple.type.loc17_42.2 (%tuple.type.897) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc18: @tuple.type.as.A.impl.F.%tuple.type.as.A.impl.F.type (%tuple.type.as.A.impl.F.type.44a) = specific_constant @tuple.type.as.A.impl.%tuple.type.as.A.impl.F.decl, @tuple.type.as.A.impl(constants.%V1, constants.%V2, constants.%W) [symbolic = %tuple.type.as.A.impl.F (constants.%tuple.type.as.A.impl.F.742)]
// CHECK:STDOUT:     %F.ref: @tuple.type.as.A.impl.F.%tuple.type.as.A.impl.F.type (%tuple.type.as.A.impl.F.type.44a) = name_ref F, %.loc18 [symbolic = %tuple.type.as.A.impl.F (constants.%tuple.type.as.A.impl.F.742)]
// CHECK:STDOUT:     %U.ref.loc18: type = name_ref U, %U.loc17_9.2 [symbolic = %U.loc17_9.1 (constants.%U.ce4)]
// CHECK:STDOUT:     %u.ref: @tuple.type.as.A.impl.F.%U.loc17_9.1 (%U.ce4) = name_ref u, %u
// CHECK:STDOUT:     %tuple.type.as.A.impl.F.specific_fn.loc18_12.1: <specific function> = specific_function %F.ref, @tuple.type.as.A.impl.F(constants.%V1, constants.%V2, constants.%W, constants.%U.ce4) [symbolic = %tuple.type.as.A.impl.F.specific_fn.loc18_12.2 (constants.%tuple.type.as.A.impl.F.specific_fn.37b)]
// CHECK:STDOUT:     %.loc17_42.1: ref @tuple.type.as.A.impl.F.%tuple.type.loc17_42.2 (%tuple.type.897) = splice_block %return.param {}
// CHECK:STDOUT:     %tuple.type.as.A.impl.F.call: init @tuple.type.as.A.impl.F.%tuple.type.loc17_42.2 (%tuple.type.897) to %.loc17_42.1 = call %tuple.type.as.A.impl.F.specific_fn.loc18_12.1(%u.ref)
// CHECK:STDOUT:     return %tuple.type.as.A.impl.F.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Y1.ref: type = name_ref Y1, file.%Y1.decl [concrete = constants.%Y1]
// CHECK:STDOUT:   %Y2.ref: type = name_ref Y2, file.%Y2.decl [concrete = constants.%Y2]
// CHECK:STDOUT:   %.loc24_12: %tuple.type.24b = tuple_literal (%Y1.ref, %Y2.ref) [concrete = constants.%tuple.8db]
// CHECK:STDOUT:   %.loc24_17: type = type_literal type [concrete = type]
// CHECK:STDOUT:   %.loc24_14: type = converted %.loc24_12, constants.%tuple.type.59f [concrete = constants.%tuple.type.59f]
// CHECK:STDOUT:   %A.ref: %A.type.55a = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:   %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %A.type: type = facet_type <@A, @A(constants.%X)> [concrete = constants.%A.type.df0]
// CHECK:STDOUT:   %A.facet: %A.type.df0 = facet_value %.loc24_14, (constants.%A.impl_witness.6b7) [concrete = constants.%A.facet.682]
// CHECK:STDOUT:   %.loc24_23: %A.type.df0 = converted %.loc24_14, %A.facet [concrete = constants.%A.facet.682]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc24_23 [concrete = constants.%tuple.type.59f]
// CHECK:STDOUT:   %.loc24_31.1: type = converted %.loc24_23, %as_type [concrete = constants.%tuple.type.59f]
// CHECK:STDOUT:   %.loc24_31.2: %A.assoc_type.a54 = specific_constant @A.WithSelf.%assoc0.loc6_39.1, @A.WithSelf(constants.%X, constants.%A.facet.682) [concrete = constants.%assoc0.987]
// CHECK:STDOUT:   %F.ref: %A.assoc_type.a54 = name_ref F, %.loc24_31.2 [concrete = constants.%assoc0.987]
// CHECK:STDOUT:   %impl.elem0: %.d9b = impl_witness_access constants.%A.impl_witness.6b7, element0 [concrete = constants.%tuple.type.as.A.impl.F.749]
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
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.loc24_39.2, constants.%Destroy.Op.1a2547.7
// CHECK:STDOUT:   %Destroy.Op.call.loc24_39: init %empty_tuple.type = call %Destroy.Op.bound(%.loc24_39.2)
// CHECK:STDOUT:   %Destroy.Op.call.loc24_38: init %empty_tuple.type = call constants.%Destroy.Op.bound(constants.%.7dc)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc24_39.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc24_39.2(%self.param: ref %X) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc24_39.3(%self.param: ref %Y1) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc24_39.4(%self.param: ref %Y2) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc24_39.5(%self.param: ref %tuple.type.59f) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc24_39.6(%self.param: ref %Z) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc24_39.7(%self.param: ref %tuple.type.d40) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallGeneric(%T.loc27_17.2: %facet_type) {
// CHECK:STDOUT:   %T.patt.loc27_17.2: %pattern_type.871 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc27_17.2 (constants.%T.patt.c17)]
// CHECK:STDOUT:   %T.loc27_17.1: %facet_type = symbolic_binding T, 0 [symbolic = %T.loc27_17.1 (constants.%T.5b7)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc28_4.2: type = facet_access_type %T.loc27_17.1 [symbolic = %T.as_type.loc28_4.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %A.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc27_17.1, @A, @A(constants.%X) [symbolic = %A.lookup_impl_witness (constants.%A.lookup_impl_witness)]
// CHECK:STDOUT:   %A.facet: %A.type.df0 = facet_value %T.as_type.loc28_4.2, (%A.lookup_impl_witness) [symbolic = %A.facet (constants.%A.facet.318)]
// CHECK:STDOUT:   %A.WithSelf.F.type: type = fn_type @A.WithSelf.F, @A.WithSelf(constants.%X, %A.facet) [symbolic = %A.WithSelf.F.type (constants.%A.WithSelf.F.type.497)]
// CHECK:STDOUT:   %.loc28_4.3: type = fn_type_with_self_type %A.WithSelf.F.type, %A.facet [symbolic = %.loc28_4.3 (constants.%.151)]
// CHECK:STDOUT:   %impl.elem0.loc28_4.2: @CallGeneric.%.loc28_4.3 (%.151) = impl_witness_access %A.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc28_4.2 (constants.%impl.elem0.1a3)]
// CHECK:STDOUT:   %specific_impl_fn.loc28_4.2: <specific function> = specific_impl_function %impl.elem0.loc28_4.2, @A.WithSelf.F(constants.%X, %A.facet, constants.%Z) [symbolic = %specific_impl_fn.loc28_4.2 (constants.%specific_impl_fn.74d)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (constants.%X, %T.as_type.loc28_4.2, constants.%Z) [symbolic = %tuple.type (constants.%tuple.type.036)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %tuple.type [symbolic = %require_complete (constants.%require_complete.4e1)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %tuple.type, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.8dc)]
// CHECK:STDOUT:   %Destroy.facet.loc28_12.3: %Destroy.type = facet_value %tuple.type, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet.loc28_12.3 (constants.%Destroy.facet.fef)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc28_12.3) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.937)]
// CHECK:STDOUT:   %.loc28_12.5: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Destroy.facet.loc28_12.3 [symbolic = %.loc28_12.5 (constants.%.93b)]
// CHECK:STDOUT:   %impl.elem0.loc28_12.2: @CallGeneric.%.loc28_12.5 (%.93b) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc28_12.2 (constants.%impl.elem0.743)]
// CHECK:STDOUT:   %specific_impl_fn.loc28_12.2: <specific function> = specific_impl_function %impl.elem0.loc28_12.2, @Destroy.WithSelf.Op(%Destroy.facet.loc28_12.3) [symbolic = %specific_impl_fn.loc28_12.2 (constants.%specific_impl_fn.618)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref: %facet_type = name_ref T, %T.loc27_17.2 [symbolic = %T.loc27_17.1 (constants.%T.5b7)]
// CHECK:STDOUT:     %T.as_type.loc28_4.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc28_4.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc28_4.1: type = converted %T.ref, %T.as_type.loc28_4.1 [symbolic = %T.as_type.loc28_4.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc28_4.2: %A.assoc_type.a54 = specific_constant @A.WithSelf.%assoc0.loc6_39.1, @A.WithSelf(constants.%X, constants.%T.5b7) [concrete = constants.%assoc0.987]
// CHECK:STDOUT:     %F.ref: %A.assoc_type.a54 = name_ref F, %.loc28_4.2 [concrete = constants.%assoc0.987]
// CHECK:STDOUT:     %impl.elem0.loc28_4.1: @CallGeneric.%.loc28_4.3 (%.151) = impl_witness_access constants.%A.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc28_4.2 (constants.%impl.elem0.1a3)]
// CHECK:STDOUT:     %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z]
// CHECK:STDOUT:     %.loc28_11.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %specific_impl_fn.loc28_4.1: <specific function> = specific_impl_function %impl.elem0.loc28_4.1, @A.WithSelf.F(constants.%X, constants.%A.facet.318, constants.%Z) [symbolic = %specific_impl_fn.loc28_4.2 (constants.%specific_impl_fn.74d)]
// CHECK:STDOUT:     %.loc28_12.1: ref @CallGeneric.%tuple.type (%tuple.type.036) = temporary_storage
// CHECK:STDOUT:     %.loc28_11.2: ref %Z = temporary_storage
// CHECK:STDOUT:     %.loc28_11.3: init %Z to %.loc28_11.2 = class_init () [concrete = constants.%Z.val]
// CHECK:STDOUT:     %.loc28_11.4: init %Z = converted %.loc28_11.1, %.loc28_11.3 [concrete = constants.%Z.val]
// CHECK:STDOUT:     %.loc28_11.5: ref %Z = temporary %.loc28_11.2, %.loc28_11.4 [concrete = constants.%.7dc]
// CHECK:STDOUT:     %.loc28_11.6: %Z = acquire_value %.loc28_11.5 [concrete = constants.%Z.val]
// CHECK:STDOUT:     %A.WithSelf.F.call: init @CallGeneric.%tuple.type (%tuple.type.036) to %.loc28_12.1 = call %specific_impl_fn.loc28_4.1(%.loc28_11.6)
// CHECK:STDOUT:     %.loc28_12.2: ref @CallGeneric.%tuple.type (%tuple.type.036) = temporary %.loc28_12.1, %A.WithSelf.F.call
// CHECK:STDOUT:     %impl.elem0.loc28_12.1: @CallGeneric.%.loc28_12.5 (%.93b) = impl_witness_access constants.%Destroy.lookup_impl_witness.8dc, element0 [symbolic = %impl.elem0.loc28_12.2 (constants.%impl.elem0.743)]
// CHECK:STDOUT:     %bound_method.loc28_12.1: <bound method> = bound_method %.loc28_12.2, %impl.elem0.loc28_12.1
// CHECK:STDOUT:     %Destroy.facet.loc28_12.1: %Destroy.type = facet_value constants.%tuple.type.036, (constants.%Destroy.lookup_impl_witness.8dc) [symbolic = %Destroy.facet.loc28_12.3 (constants.%Destroy.facet.fef)]
// CHECK:STDOUT:     %.loc28_12.3: %Destroy.type = converted constants.%tuple.type.036, %Destroy.facet.loc28_12.1 [symbolic = %Destroy.facet.loc28_12.3 (constants.%Destroy.facet.fef)]
// CHECK:STDOUT:     %Destroy.facet.loc28_12.2: %Destroy.type = facet_value constants.%tuple.type.036, (constants.%Destroy.lookup_impl_witness.8dc) [symbolic = %Destroy.facet.loc28_12.3 (constants.%Destroy.facet.fef)]
// CHECK:STDOUT:     %.loc28_12.4: %Destroy.type = converted constants.%tuple.type.036, %Destroy.facet.loc28_12.2 [symbolic = %Destroy.facet.loc28_12.3 (constants.%Destroy.facet.fef)]
// CHECK:STDOUT:     %specific_impl_fn.loc28_12.1: <specific function> = specific_impl_function %impl.elem0.loc28_12.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.fef) [symbolic = %specific_impl_fn.loc28_12.2 (constants.%specific_impl_fn.618)]
// CHECK:STDOUT:     %bound_method.loc28_12.2: <bound method> = bound_method %.loc28_12.2, %specific_impl_fn.loc28_12.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc28_12.2(%.loc28_12.2)
// CHECK:STDOUT:     %Destroy.Op.call: init %empty_tuple.type = call constants.%Destroy.Op.bound(constants.%.7dc)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallIndirect() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CallGeneric.ref: %CallGeneric.type = name_ref CallGeneric, file.%CallGeneric.decl [concrete = constants.%CallGeneric]
// CHECK:STDOUT:   %Y1.ref: type = name_ref Y1, file.%Y1.decl [concrete = constants.%Y1]
// CHECK:STDOUT:   %Y2.ref: type = name_ref Y2, file.%Y2.decl [concrete = constants.%Y2]
// CHECK:STDOUT:   %.loc33_22: %tuple.type.24b = tuple_literal (%Y1.ref, %Y2.ref) [concrete = constants.%tuple.8db]
// CHECK:STDOUT:   %.loc33_27: type = type_literal type [concrete = type]
// CHECK:STDOUT:   %.loc33_24: type = converted %.loc33_22, constants.%tuple.type.59f [concrete = constants.%tuple.type.59f]
// CHECK:STDOUT:   %facet_value: %facet_type = facet_value %.loc33_24, (constants.%A.impl_witness.6b7, constants.%custom_witness.df9cc1.5) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc33_31: %facet_type = converted %.loc33_24, %facet_value [concrete = constants.%facet_value]
// CHECK:STDOUT:   %CallGeneric.specific_fn: <specific function> = specific_function %CallGeneric.ref, @CallGeneric(constants.%facet_value) [concrete = constants.%CallGeneric.specific_fn]
// CHECK:STDOUT:   %CallGeneric.call: init %empty_tuple.type = call %CallGeneric.specific_fn()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%T.67d) {
// CHECK:STDOUT:   %T.patt.loc5_14.2 => constants.%T.patt.d47
// CHECK:STDOUT:   %T.loc5_14.1 => constants.%T.67d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%T.67d, constants.%Self.49a) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf.F(constants.%T.67d, constants.%Self.49a, constants.%U.4e7) {
// CHECK:STDOUT:   %U.patt.loc6_9.2 => constants.%U.patt.b22
// CHECK:STDOUT:   %U.loc6_9.1 => constants.%U.4e7
// CHECK:STDOUT:   %pattern_type.loc6_19 => constants.%pattern_type.62e
// CHECK:STDOUT:   %u.param_patt.loc6_19.2 => constants.%u.param_patt.6ef
// CHECK:STDOUT:   %u.patt.loc6_19.2 => constants.%u.patt.8af
// CHECK:STDOUT:   %T => constants.%T.67d
// CHECK:STDOUT:   %A.type => constants.%A.type.acd
// CHECK:STDOUT:   %Self => constants.%Self.49a
// CHECK:STDOUT:   %tuple.type.loc6_38.1 => constants.%tuple.type.76f
// CHECK:STDOUT:   %tuple => constants.%tuple.55a
// CHECK:STDOUT:   %Self.as_type.loc6_38.1 => constants.%Self.as_type.763
// CHECK:STDOUT:   %tuple.type.loc6_38.2 => constants.%tuple.type.d73
// CHECK:STDOUT:   %.loc6_38.1 => constants.%.bc3
// CHECK:STDOUT:   %pattern_type.loc6_38 => constants.%pattern_type.d6c
// CHECK:STDOUT:   %return.param_patt.loc6_38.2 => constants.%return.param_patt.52a
// CHECK:STDOUT:   %return.patt.loc6_24.2 => constants.%return.patt.808
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%W) {
// CHECK:STDOUT:   %T.patt.loc5_14.2 => constants.%T.patt.d47
// CHECK:STDOUT:   %T.loc5_14.1 => constants.%W
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.type => constants.%A.type.3ad
// CHECK:STDOUT:   %Self.loc5_23.2 => constants.%Self.0f8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @tuple.type.as.A.impl(constants.%V1, constants.%V2, constants.%W) {
// CHECK:STDOUT:   %V1.patt.loc14_16.2 => constants.%V1.patt
// CHECK:STDOUT:   %V1.loc14_16.1 => constants.%V1
// CHECK:STDOUT:   %V2.patt.loc14_27.2 => constants.%V2.patt
// CHECK:STDOUT:   %V2.loc14_27.1 => constants.%V2
// CHECK:STDOUT:   %W.patt.loc14_37.2 => constants.%W.patt
// CHECK:STDOUT:   %W.loc14_37.1 => constants.%W
// CHECK:STDOUT:   %tuple => constants.%tuple.4b9
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.a5e
// CHECK:STDOUT:   %A.type.loc14_61.1 => constants.%A.type.3ad
// CHECK:STDOUT:   %A.impl_witness.loc14_63.2 => constants.%A.impl_witness.756
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
// CHECK:STDOUT:   %assoc0.loc6_39.2 => constants.%assoc0.6de
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @tuple.type.as.A.impl.F(constants.%V1, constants.%V2, constants.%W, constants.%U.ce4) {
// CHECK:STDOUT:   %U.patt.loc17_9.2 => constants.%U.patt.637
// CHECK:STDOUT:   %U.loc17_9.1 => constants.%U.ce4
// CHECK:STDOUT:   %pattern_type.loc17_19 => constants.%pattern_type.e4a
// CHECK:STDOUT:   %u.param_patt.loc17_19.2 => constants.%u.param_patt.afe
// CHECK:STDOUT:   %u.patt.loc17_19.2 => constants.%u.patt.e63
// CHECK:STDOUT:   %W => constants.%W
// CHECK:STDOUT:   %V1 => constants.%V1
// CHECK:STDOUT:   %V2 => constants.%V2
// CHECK:STDOUT:   %tuple.loc17_38 => constants.%tuple.4b9
// CHECK:STDOUT:   %tuple.loc17_42 => constants.%tuple.117
// CHECK:STDOUT:   %tuple.type.loc17_42.1 => constants.%tuple.type.a5e
// CHECK:STDOUT:   %tuple.type.loc17_42.2 => constants.%tuple.type.897
// CHECK:STDOUT:   %.loc17_42.2 => constants.%.f8e
// CHECK:STDOUT:   %pattern_type.loc17_42 => constants.%pattern_type.5cc
// CHECK:STDOUT:   %return.param_patt.loc17_42.2 => constants.%return.param_patt.b1c
// CHECK:STDOUT:   %return.patt.loc17_24.2 => constants.%return.patt.bad9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc17_19 => constants.%require_complete.5a4
// CHECK:STDOUT:   %require_complete.loc17_24 => constants.%require_complete.da0
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type => constants.%tuple.type.as.A.impl.F.type.44a
// CHECK:STDOUT:   %tuple.type.as.A.impl.F => constants.%tuple.type.as.A.impl.F.742
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.specific_fn.loc18_12.2 => constants.%tuple.type.as.A.impl.F.specific_fn.37b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%W, constants.%A.facet.ab0) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%W
// CHECK:STDOUT:   %A.type => constants.%A.type.3ad
// CHECK:STDOUT:   %Self => constants.%A.facet.ab0
// CHECK:STDOUT:   %A.WithSelf.F.type => constants.%A.WithSelf.F.type.5f0
// CHECK:STDOUT:   %A.WithSelf.F => constants.%A.WithSelf.F.147
// CHECK:STDOUT:   %A.assoc_type => constants.%A.assoc_type.250
// CHECK:STDOUT:   %assoc0.loc6_39.2 => constants.%assoc0.6de
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf.F(constants.%W, constants.%A.facet.ab0, constants.%U.ce4) {
// CHECK:STDOUT:   %U.patt.loc6_9.2 => constants.%U.patt.b22
// CHECK:STDOUT:   %U.loc6_9.1 => constants.%U.ce4
// CHECK:STDOUT:   %pattern_type.loc6_19 => constants.%pattern_type.e4a
// CHECK:STDOUT:   %u.param_patt.loc6_19.2 => constants.%u.param_patt.afe
// CHECK:STDOUT:   %u.patt.loc6_19.2 => constants.%u.patt.e63
// CHECK:STDOUT:   %T => constants.%W
// CHECK:STDOUT:   %A.type => constants.%A.type.3ad
// CHECK:STDOUT:   %Self => constants.%A.facet.ab0
// CHECK:STDOUT:   %tuple.type.loc6_38.1 => constants.%tuple.type.a90
// CHECK:STDOUT:   %tuple => constants.%tuple.059
// CHECK:STDOUT:   %Self.as_type.loc6_38.1 => constants.%tuple.type.a5e
// CHECK:STDOUT:   %tuple.type.loc6_38.2 => constants.%tuple.type.897
// CHECK:STDOUT:   %.loc6_38.1 => constants.%.f8e
// CHECK:STDOUT:   %pattern_type.loc6_38 => constants.%pattern_type.5cc
// CHECK:STDOUT:   %return.param_patt.loc6_38.2 => constants.%return.param_patt.b1c
// CHECK:STDOUT:   %return.patt.loc6_24.2 => constants.%return.patt.bad9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%X) {
// CHECK:STDOUT:   %T.patt.loc5_14.2 => constants.%T.patt.d47
// CHECK:STDOUT:   %T.loc5_14.1 => constants.%X
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.type => constants.%A.type.df0
// CHECK:STDOUT:   %Self.loc5_23.2 => constants.%Self.891
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @tuple.type.as.A.impl(constants.%Y1, constants.%Y2, constants.%X) {
// CHECK:STDOUT:   %V1.patt.loc14_16.2 => constants.%V1.patt
// CHECK:STDOUT:   %V1.loc14_16.1 => constants.%Y1
// CHECK:STDOUT:   %V2.patt.loc14_27.2 => constants.%V2.patt
// CHECK:STDOUT:   %V2.loc14_27.1 => constants.%Y2
// CHECK:STDOUT:   %W.patt.loc14_37.2 => constants.%W.patt
// CHECK:STDOUT:   %W.loc14_37.1 => constants.%X
// CHECK:STDOUT:   %tuple => constants.%tuple.8db
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.59f
// CHECK:STDOUT:   %A.type.loc14_61.1 => constants.%A.type.df0
// CHECK:STDOUT:   %A.impl_witness.loc14_63.2 => constants.%A.impl_witness.6b7
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
// CHECK:STDOUT:   %assoc0.loc6_39.2 => constants.%assoc0.987
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%X, constants.%A.facet.682) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.df0
// CHECK:STDOUT:   %Self => constants.%A.facet.682
// CHECK:STDOUT:   %A.WithSelf.F.type => constants.%A.WithSelf.F.type.243
// CHECK:STDOUT:   %A.WithSelf.F => constants.%A.WithSelf.F.5b5
// CHECK:STDOUT:   %A.assoc_type => constants.%A.assoc_type.a54
// CHECK:STDOUT:   %assoc0.loc6_39.2 => constants.%assoc0.987
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @tuple.type.as.A.impl.F(constants.%Y1, constants.%Y2, constants.%X, constants.%Z) {
// CHECK:STDOUT:   %U.patt.loc17_9.2 => constants.%U.patt.637
// CHECK:STDOUT:   %U.loc17_9.1 => constants.%Z
// CHECK:STDOUT:   %pattern_type.loc17_19 => constants.%pattern_type.34e
// CHECK:STDOUT:   %u.param_patt.loc17_19.2 => constants.%u.param_patt.a3e
// CHECK:STDOUT:   %u.patt.loc17_19.2 => constants.%u.patt.a51
// CHECK:STDOUT:   %W => constants.%X
// CHECK:STDOUT:   %V1 => constants.%Y1
// CHECK:STDOUT:   %V2 => constants.%Y2
// CHECK:STDOUT:   %tuple.loc17_38 => constants.%tuple.8db
// CHECK:STDOUT:   %tuple.loc17_42 => constants.%tuple.1a1
// CHECK:STDOUT:   %tuple.type.loc17_42.1 => constants.%tuple.type.59f
// CHECK:STDOUT:   %tuple.type.loc17_42.2 => constants.%tuple.type.d40
// CHECK:STDOUT:   %.loc17_42.2 => constants.%.1a9
// CHECK:STDOUT:   %pattern_type.loc17_42 => constants.%pattern_type.6a8
// CHECK:STDOUT:   %return.param_patt.loc17_42.2 => constants.%return.param_patt.fb7
// CHECK:STDOUT:   %return.patt.loc17_24.2 => constants.%return.patt.badc
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc17_19 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc17_24 => constants.%complete_type.8f6
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type => constants.%tuple.type.as.A.impl.F.type.745
// CHECK:STDOUT:   %tuple.type.as.A.impl.F => constants.%tuple.type.as.A.impl.F.749
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.specific_fn.loc18_12.2 => constants.%tuple.type.as.A.impl.F.specific_fn.981
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGeneric(constants.%T.5b7) {
// CHECK:STDOUT:   %T.patt.loc27_17.2 => constants.%T.patt.c17
// CHECK:STDOUT:   %T.loc27_17.1 => constants.%T.5b7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%X, constants.%T.5b7) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.df0
// CHECK:STDOUT:   %Self => constants.%T.5b7
// CHECK:STDOUT:   %A.WithSelf.F.type => constants.%A.WithSelf.F.type.91d
// CHECK:STDOUT:   %A.WithSelf.F => constants.%A.WithSelf.F.3f4
// CHECK:STDOUT:   %A.assoc_type => constants.%A.assoc_type.a54
// CHECK:STDOUT:   %assoc0.loc6_39.2 => constants.%assoc0.987
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%X, constants.%A.facet.318) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.df0
// CHECK:STDOUT:   %Self => constants.%A.facet.318
// CHECK:STDOUT:   %A.WithSelf.F.type => constants.%A.WithSelf.F.type.497
// CHECK:STDOUT:   %A.WithSelf.F => constants.%A.WithSelf.F.a4d
// CHECK:STDOUT:   %A.assoc_type => constants.%A.assoc_type.a54
// CHECK:STDOUT:   %assoc0.loc6_39.2 => constants.%assoc0.987
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf.F(constants.%X, constants.%A.facet.318, constants.%Z) {
// CHECK:STDOUT:   %U.patt.loc6_9.2 => constants.%U.patt.b22
// CHECK:STDOUT:   %U.loc6_9.1 => constants.%Z
// CHECK:STDOUT:   %pattern_type.loc6_19 => constants.%pattern_type.34e
// CHECK:STDOUT:   %u.param_patt.loc6_19.2 => constants.%u.param_patt.a3e
// CHECK:STDOUT:   %u.patt.loc6_19.2 => constants.%u.patt.a51
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.df0
// CHECK:STDOUT:   %Self => constants.%A.facet.318
// CHECK:STDOUT:   %tuple.type.loc6_38.1 => constants.%tuple.type.966
// CHECK:STDOUT:   %tuple => constants.%tuple.b58
// CHECK:STDOUT:   %Self.as_type.loc6_38.1 => constants.%T.as_type
// CHECK:STDOUT:   %tuple.type.loc6_38.2 => constants.%tuple.type.036
// CHECK:STDOUT:   %.loc6_38.1 => constants.%.772
// CHECK:STDOUT:   %pattern_type.loc6_38 => constants.%pattern_type.8f9
// CHECK:STDOUT:   %return.param_patt.loc6_38.2 => constants.%return.param_patt.cc2
// CHECK:STDOUT:   %return.patt.loc6_24.2 => constants.%return.patt.eed
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGeneric(constants.%facet_value) {
// CHECK:STDOUT:   %T.patt.loc27_17.2 => constants.%T.patt.c17
// CHECK:STDOUT:   %T.loc27_17.1 => constants.%facet_value
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc28_4.2 => constants.%tuple.type.59f
// CHECK:STDOUT:   %A.lookup_impl_witness => constants.%A.impl_witness.6b7
// CHECK:STDOUT:   %A.facet => constants.%A.facet.682
// CHECK:STDOUT:   %A.WithSelf.F.type => constants.%A.WithSelf.F.type.243
// CHECK:STDOUT:   %.loc28_4.3 => constants.%.d9b
// CHECK:STDOUT:   %impl.elem0.loc28_4.2 => constants.%tuple.type.as.A.impl.F.749
// CHECK:STDOUT:   %specific_impl_fn.loc28_4.2 => constants.%tuple.type.as.A.impl.F.specific_fn.981
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.d40
// CHECK:STDOUT:   %require_complete => constants.%complete_type.8f6
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.df9cc1.7
// CHECK:STDOUT:   %Destroy.facet.loc28_12.3 => constants.%Destroy.facet.50b
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.43f
// CHECK:STDOUT:   %.loc28_12.5 => constants.%.3d3
// CHECK:STDOUT:   %impl.elem0.loc28_12.2 => constants.%Destroy.Op.1a2547.7
// CHECK:STDOUT:   %specific_impl_fn.loc28_12.2 => constants.%Destroy.Op.1a2547.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf.F(constants.%X, constants.%A.facet.682, constants.%Z) {
// CHECK:STDOUT:   %U.patt.loc6_9.2 => constants.%U.patt.b22
// CHECK:STDOUT:   %U.loc6_9.1 => constants.%Z
// CHECK:STDOUT:   %pattern_type.loc6_19 => constants.%pattern_type.34e
// CHECK:STDOUT:   %u.param_patt.loc6_19.2 => constants.%u.param_patt.a3e
// CHECK:STDOUT:   %u.patt.loc6_19.2 => constants.%u.patt.a51
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.df0
// CHECK:STDOUT:   %Self => constants.%A.facet.682
// CHECK:STDOUT:   %tuple.type.loc6_38.1 => constants.%tuple.type.966
// CHECK:STDOUT:   %tuple => constants.%tuple.92c
// CHECK:STDOUT:   %Self.as_type.loc6_38.1 => constants.%tuple.type.59f
// CHECK:STDOUT:   %tuple.type.loc6_38.2 => constants.%tuple.type.d40
// CHECK:STDOUT:   %.loc6_38.1 => constants.%.1a9
// CHECK:STDOUT:   %pattern_type.loc6_38 => constants.%pattern_type.6a8
// CHECK:STDOUT:   %return.param_patt.loc6_38.2 => constants.%return.param_patt.fb7
// CHECK:STDOUT:   %return.patt.loc6_24.2 => constants.%return.patt.badc
// CHECK:STDOUT: }
// CHECK:STDOUT:

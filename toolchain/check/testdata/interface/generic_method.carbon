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
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %A.type.9ad: type = generic_interface_type @A [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %A.generic: %A.type.9ad = struct_value () [concrete]
// CHECK:STDOUT:   %A.type.ef5: type = facet_type <@A, @A(%T.67d)> [symbolic]
// CHECK:STDOUT:   %Self.0bb: %A.type.ef5 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %U.4e7: type = symbolic_binding U, 2 [symbolic]
// CHECK:STDOUT:   %ptr.8f6: type = ptr_type %U.4e7 [symbolic]
// CHECK:STDOUT:   %pattern_type.986: type = pattern_type %ptr.8f6 [symbolic]
// CHECK:STDOUT:   %tuple.type.721: type = tuple_type (type, %A.type.ef5, type) [symbolic]
// CHECK:STDOUT:   %tuple.149: %tuple.type.721 = tuple_value (%T.67d, %Self.0bb, %ptr.8f6) [symbolic]
// CHECK:STDOUT:   %Self.as_type.2f2: type = facet_access_type %Self.0bb [symbolic]
// CHECK:STDOUT:   %tuple.type.c48: type = tuple_type (%T.67d, %Self.as_type.2f2, %ptr.8f6) [symbolic]
// CHECK:STDOUT:   %.973: Core.Form = init_form %tuple.type.c48 [symbolic]
// CHECK:STDOUT:   %pattern_type.274: type = pattern_type %tuple.type.c48 [symbolic]
// CHECK:STDOUT:   %A.WithSelf.F.type.8f1: type = fn_type @A.WithSelf.F, @A.WithSelf(%T.67d, %Self.0bb) [symbolic]
// CHECK:STDOUT:   %A.WithSelf.F.d3d: %A.WithSelf.F.type.8f1 = struct_value () [symbolic]
// CHECK:STDOUT:   %A.assoc_type.24d: type = assoc_entity_type @A, @A(%T.67d) [symbolic]
// CHECK:STDOUT:   %assoc0.ecf: %A.assoc_type.24d = assoc_entity element0, @A.WithSelf.%A.WithSelf.F.decl [symbolic]
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Y: type = class_type @Y [concrete]
// CHECK:STDOUT:   %Z: type = class_type @Z [concrete]
// CHECK:STDOUT:   %A.type.1f4: type = facet_type <@A, @A(%X)> [concrete]
// CHECK:STDOUT:   %A.impl_witness: <witness> = impl_witness @Y.as.A.impl.%A.impl_witness_table [concrete]
// CHECK:STDOUT:   %Self.edd: %A.type.1f4 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %A.WithSelf.F.type.82f: type = fn_type @A.WithSelf.F, @A.WithSelf(%X, %Self.0bb) [symbolic]
// CHECK:STDOUT:   %A.WithSelf.F.7f8: %A.WithSelf.F.type.82f = struct_value () [symbolic]
// CHECK:STDOUT:   %A.assoc_type.409: type = assoc_entity_type @A, @A(%X) [concrete]
// CHECK:STDOUT:   %assoc0.4b8: %A.assoc_type.409 = assoc_entity element0, @A.WithSelf.%A.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %U.67d: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %ptr.e8f8f9.1: type = ptr_type %U.67d [symbolic]
// CHECK:STDOUT:   %pattern_type.4f4b84.1: type = pattern_type %ptr.e8f8f9.1 [symbolic]
// CHECK:STDOUT:   %tuple.type.ff9: type = tuple_type (type, type, type) [concrete]
// CHECK:STDOUT:   %tuple.e08: %tuple.type.ff9 = tuple_value (%X, %Y, %ptr.e8f8f9.1) [symbolic]
// CHECK:STDOUT:   %tuple.type.711: type = tuple_type (%X, %Y, %ptr.e8f8f9.1) [symbolic]
// CHECK:STDOUT:   %.f98: Core.Form = init_form %tuple.type.711 [symbolic]
// CHECK:STDOUT:   %pattern_type.832: type = pattern_type %tuple.type.711 [symbolic]
// CHECK:STDOUT:   %Y.as.A.impl.F.type: type = fn_type @Y.as.A.impl.F [concrete]
// CHECK:STDOUT:   %Y.as.A.impl.F: %Y.as.A.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %A.facet.0c5: %A.type.1f4 = facet_value %Y, (%A.impl_witness) [concrete]
// CHECK:STDOUT:   %A.WithSelf.F.type.423: type = fn_type @A.WithSelf.F, @A.WithSelf(%X, %A.facet.0c5) [concrete]
// CHECK:STDOUT:   %A.WithSelf.F.90b: %A.WithSelf.F.type.423 = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.9b2: type = tuple_type (type, %A.type.1f4, type) [concrete]
// CHECK:STDOUT:   %tuple.afd: %tuple.type.9b2 = tuple_value (%X, %A.facet.0c5, %ptr.e8f8f9.1) [symbolic]
// CHECK:STDOUT:   %require_complete.ef162c.1: <witness> = require_complete_type %ptr.e8f8f9.1 [symbolic]
// CHECK:STDOUT:   %require_complete.f4e: <witness> = require_complete_type %tuple.type.711 [symbolic]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.dd2: type = tuple_type (%empty_struct_type, %empty_struct_type, %ptr.e8f8f9.1) [symbolic]
// CHECK:STDOUT:   %X.val: %X = struct_value () [concrete]
// CHECK:STDOUT:   %Y.val: %Y = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.8a8539.1: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.6bf536.1: %ptr.as.Copy.impl.Op.type.8a8539.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.978: <witness> = lookup_impl_witness %ptr.e8f8f9.1, @Copy [symbolic]
// CHECK:STDOUT:   %.748: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%U.67d) [symbolic]
// CHECK:STDOUT:   %Copy.facet.b5f: %Copy.type = facet_value %ptr.e8f8f9.1, (%Copy.lookup_impl_witness.978) [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.172: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.b5f) [symbolic]
// CHECK:STDOUT:   %.60c: type = fn_type_with_self_type %Copy.WithSelf.Op.type.172, %Copy.facet.b5f [symbolic]
// CHECK:STDOUT:   %impl.elem0.adc: %.60c = impl_witness_access %Copy.lookup_impl_witness.978, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.dbc: <specific function> = specific_impl_function %impl.elem0.adc, @Copy.WithSelf.Op(%Copy.facet.b5f) [symbolic]
// CHECK:STDOUT:   %Call.type: type = fn_type @Call [concrete]
// CHECK:STDOUT:   %Call: %Call.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.dd4: type = pattern_type %Z [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.type.42d: type = fn_type @T.as.DefaultOrUnformed.impl.Op, @T.as.DefaultOrUnformed.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.57e: %T.as.DefaultOrUnformed.impl.Op.type.42d = struct_value () [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.ece: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.28c, @T.as.DefaultOrUnformed.impl(%Z) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.type.45a: type = fn_type @T.as.DefaultOrUnformed.impl.Op, @T.as.DefaultOrUnformed.impl(%Z) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.cb6: %T.as.DefaultOrUnformed.impl.Op.type.45a = struct_value () [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value %Z, (%DefaultOrUnformed.impl_witness.ece) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.specific_fn: <specific function> = specific_function %T.as.DefaultOrUnformed.impl.Op.cb6, @T.as.DefaultOrUnformed.impl.Op(%Z) [concrete]
// CHECK:STDOUT:   %.c69: type = fn_type_with_self_type %A.WithSelf.F.type.423, %A.facet.0c5 [concrete]
// CHECK:STDOUT:   %ptr.b15: type = ptr_type %Z [concrete]
// CHECK:STDOUT:   %pattern_type.9d3: type = pattern_type %ptr.b15 [concrete]
// CHECK:STDOUT:   %tuple.469: %tuple.type.ff9 = tuple_value (%X, %Y, %ptr.b15) [concrete]
// CHECK:STDOUT:   %tuple.type.348: type = tuple_type (%X, %Y, %ptr.b15) [concrete]
// CHECK:STDOUT:   %.40c: Core.Form = init_form %tuple.type.348 [concrete]
// CHECK:STDOUT:   %pattern_type.b7b: type = pattern_type %tuple.type.348 [concrete]
// CHECK:STDOUT:   %Y.as.A.impl.F.specific_fn: <specific function> = specific_function %Y.as.A.impl.F, @Y.as.A.impl.F(%Z) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.3f5f78.3: type = fn_type @Destroy.Op.loc23_22.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.f0ba5f.3: %Destroy.Op.type.3f5f78.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.1367bd.3: <witness> = custom_witness (%Destroy.Op.f0ba5f.3), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.3f5f78.5: type = fn_type @Destroy.Op.loc23_22.5 [concrete]
// CHECK:STDOUT:   %Destroy.Op.f0ba5f.5: %Destroy.Op.type.3f5f78.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.1367bd.5: <witness> = custom_witness (%Destroy.Op.f0ba5f.5), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.21e: %Destroy.type = facet_value %tuple.type.348, (%custom_witness.1367bd.5) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.af0: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.21e) [concrete]
// CHECK:STDOUT:   %.0ae: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.af0, %Destroy.facet.21e [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.3f5f78.6: type = fn_type @Destroy.Op.loc22 [concrete]
// CHECK:STDOUT:   %Destroy.Op.f0ba5f.6: %Destroy.Op.type.3f5f78.6 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndWith.type.447: type = generic_interface_type @BitAndWith [concrete]
// CHECK:STDOUT:   %BitAndWith.generic: %BitAndWith.type.447 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndWith.type.d17: type = facet_type <@BitAndWith, @BitAndWith(type)> [concrete]
// CHECK:STDOUT:   %BitAndWith.impl_witness: <witness> = impl_witness imports.%BitAndWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %BitAndWith.facet: %BitAndWith.type.d17 = facet_value type, (%BitAndWith.impl_witness) [concrete]
// CHECK:STDOUT:   %BitAndWith.WithSelf.Op.type.3c5: type = fn_type @BitAndWith.WithSelf.Op, @BitAndWith.WithSelf(type, %BitAndWith.facet) [concrete]
// CHECK:STDOUT:   %.849: type = fn_type_with_self_type %BitAndWith.WithSelf.Op.type.3c5, %BitAndWith.facet [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.type: type = fn_type @type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op: %type.as.BitAndWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.bound: <bound method> = bound_method %A.type.1f4, %type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@A, @A(%X) & @Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.689: type = pattern_type %facet_type [concrete]
// CHECK:STDOUT:   %T.0ab: %facet_type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %CallGeneric.type: type = fn_type @CallGeneric [concrete]
// CHECK:STDOUT:   %CallGeneric: %CallGeneric.type = struct_value () [concrete]
// CHECK:STDOUT:   %A.WithSelf.F.type.cf1: type = fn_type @A.WithSelf.F, @A.WithSelf(%X, %T.0ab) [symbolic]
// CHECK:STDOUT:   %A.WithSelf.F.a39: %A.WithSelf.F.type.cf1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.010: type = facet_access_type %T.0ab [symbolic]
// CHECK:STDOUT:   %A.lookup_impl_witness: <witness> = lookup_impl_witness %T.0ab, @A, @A(%X) [symbolic]
// CHECK:STDOUT:   %A.facet.88a: %A.type.1f4 = facet_value %T.as_type.010, (%A.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %A.WithSelf.F.type.9a6: type = fn_type @A.WithSelf.F, @A.WithSelf(%X, %A.facet.88a) [symbolic]
// CHECK:STDOUT:   %A.WithSelf.F.de8: %A.WithSelf.F.type.9a6 = struct_value () [symbolic]
// CHECK:STDOUT:   %.ae0: type = fn_type_with_self_type %A.WithSelf.F.type.9a6, %A.facet.88a [symbolic]
// CHECK:STDOUT:   %impl.elem0.4fc: %.ae0 = impl_witness_access %A.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %tuple.e42: %tuple.type.9b2 = tuple_value (%X, %A.facet.88a, %ptr.b15) [symbolic]
// CHECK:STDOUT:   %tuple.type.27a: type = tuple_type (%X, %T.as_type.010, %ptr.b15) [symbolic]
// CHECK:STDOUT:   %.52d: Core.Form = init_form %tuple.type.27a [symbolic]
// CHECK:STDOUT:   %pattern_type.0a1: type = pattern_type %tuple.type.27a [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.5bf: <specific function> = specific_impl_function %impl.elem0.4fc, @A.WithSelf.F(%X, %A.facet.88a, %Z) [symbolic]
// CHECK:STDOUT:   %require_complete.703: <witness> = require_complete_type %tuple.type.27a [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.33f: <witness> = lookup_impl_witness %tuple.type.27a, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.7eb: %Destroy.type = facet_value %tuple.type.27a, (%Destroy.lookup_impl_witness.33f) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.0a4: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.7eb) [symbolic]
// CHECK:STDOUT:   %.7ed: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.0a4, %Destroy.facet.7eb [symbolic]
// CHECK:STDOUT:   %impl.elem0.ad1: %.7ed = impl_witness_access %Destroy.lookup_impl_witness.33f, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.f4d: <specific function> = specific_impl_function %impl.elem0.ad1, @Destroy.WithSelf.Op(%Destroy.facet.7eb) [symbolic]
// CHECK:STDOUT:   %CallIndirect.type: type = fn_type @CallIndirect [concrete]
// CHECK:STDOUT:   %CallIndirect: %CallIndirect.type = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value: %facet_type = facet_value %Y, (%A.impl_witness, %custom_witness.1367bd.3) [concrete]
// CHECK:STDOUT:   %CallGeneric.specific_fn: <specific function> = specific_function %CallGeneric, @CallGeneric(%facet_value) [concrete]
// CHECK:STDOUT:   %complete_type.873: <witness> = complete_type_witness %ptr.b15 [concrete]
// CHECK:STDOUT:   %complete_type.79c: <witness> = complete_type_witness %tuple.type.348 [concrete]
// CHECK:STDOUT:   %tuple.type.e15: type = tuple_type (%empty_struct_type, %empty_struct_type, %ptr.b15) [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.9c3: <witness> = impl_witness imports.%Copy.impl_witness_table.3cb, @ptr.as.Copy.impl(%Z) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.371: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%Z) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.7d4: %ptr.as.Copy.impl.Op.type.371 = struct_value () [concrete]
// CHECK:STDOUT:   %.eb0: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%Z) [concrete]
// CHECK:STDOUT:   %Copy.facet.3da: %Copy.type = facet_value %ptr.b15, (%Copy.impl_witness.9c3) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.6ab: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.3da) [concrete]
// CHECK:STDOUT:   %.f67: type = fn_type_with_self_type %Copy.WithSelf.Op.type.6ab, %Copy.facet.3da [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.7d4, @ptr.as.Copy.impl.Op(%Z) [concrete]
// CHECK:STDOUT:   %tuple.b11: %tuple.type.9b2 = tuple_value (%X, %A.facet.0c5, %ptr.b15) [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.3ef: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.8a8539.1) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.6bf536.1)]
// CHECK:STDOUT:   %Copy.impl_witness_table.3cb = impl_witness_table (%Core.import_ref.3ef), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.DefaultOrUnformed: type = import_ref Core//prelude/parts/default, DefaultOrUnformed, loaded [concrete = constants.%DefaultOrUnformed.type]
// CHECK:STDOUT:   %Core.import_ref.51d: @T.as.DefaultOrUnformed.impl.%T.as.DefaultOrUnformed.impl.Op.type (%T.as.DefaultOrUnformed.impl.Op.type.42d) = import_ref Core//prelude/parts/default, loc{{\d+_\d+}}, loaded [symbolic = @T.as.DefaultOrUnformed.impl.%T.as.DefaultOrUnformed.impl.Op (constants.%T.as.DefaultOrUnformed.impl.Op.57e)]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness_table.28c = impl_witness_table (%Core.import_ref.51d), @T.as.DefaultOrUnformed.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.BitAndWith: %BitAndWith.type.447 = import_ref Core//prelude/parts/as, BitAndWith, loaded [concrete = constants.%BitAndWith.generic]
// CHECK:STDOUT:   %Core.import_ref.4c9: %type.as.BitAndWith.impl.Op.type = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, loaded [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:   %BitAndWith.impl_witness_table = impl_witness_table (%Core.import_ref.4c9), @type.as.BitAndWith.impl [concrete]
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
// CHECK:STDOUT:   %A.decl: %A.type.9ad = interface_decl @A [concrete = constants.%A.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
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
// CHECK:STDOUT:     %A.ref: %A.type.9ad = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %A.type: type = facet_type <@A, @A(constants.%X)> [concrete = constants.%A.type.1f4]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Call.decl: %Call.type = fn_decl @Call [concrete = constants.%Call] {} {}
// CHECK:STDOUT:   %CallGeneric.decl: %CallGeneric.type = fn_decl @CallGeneric [concrete = constants.%CallGeneric] {
// CHECK:STDOUT:     %T.patt: %pattern_type.689 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc26_25.1: type = splice_block %.loc26_25.3 [concrete = constants.%facet_type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %A.ref: %A.type.9ad = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:       %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:       %A.type: type = facet_type <@A, @A(constants.%X)> [concrete = constants.%A.type.1f4]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:       %impl.elem0.loc26: %.849 = impl_witness_access constants.%BitAndWith.impl_witness, element0 [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:       %bound_method.loc26: <bound method> = bound_method %A.type, %impl.elem0.loc26 [concrete = constants.%type.as.BitAndWith.impl.Op.bound]
// CHECK:STDOUT:       %type.as.BitAndWith.impl.Op.call: init type = call %bound_method.loc26(%A.type, %Destroy.ref) [concrete = constants.%facet_type]
// CHECK:STDOUT:       %.loc26_25.2: type = value_of_initializer %type.as.BitAndWith.impl.Op.call [concrete = constants.%facet_type]
// CHECK:STDOUT:       %.loc26_25.3: type = converted %type.as.BitAndWith.impl.Op.call, %.loc26_25.2 [concrete = constants.%facet_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc26_17.2: %facet_type = symbolic_binding T, 0 [symbolic = %T.loc26_17.1 (constants.%T.0ab)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallIndirect.decl: %CallIndirect.type = fn_decl @CallIndirect [concrete = constants.%CallIndirect] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @A(%T.loc5_14.2: type) {
// CHECK:STDOUT:   %T.loc5_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc5_14.1 (constants.%T.67d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.type: type = facet_type <@A, @A(%T.loc5_14.1)> [symbolic = %A.type (constants.%A.type.ef5)]
// CHECK:STDOUT:   %Self.loc5_23.2: @A.%A.type (%A.type.ef5) = symbolic_binding Self, 1 [symbolic = %Self.loc5_23.2 (constants.%Self.0bb)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc5_23.1: @A.%A.type (%A.type.ef5) = symbolic_binding Self, 1 [symbolic = %Self.loc5_23.2 (constants.%Self.0bb)]
// CHECK:STDOUT:     %A.WithSelf.decl = interface_with_self_decl @A [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %A.WithSelf.F.decl: @A.WithSelf.%A.WithSelf.F.type (%A.WithSelf.F.type.8f1) = fn_decl @A.WithSelf.F [symbolic = @A.WithSelf.%A.WithSelf.F (constants.%A.WithSelf.F.d3d)] {
// CHECK:STDOUT:       %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 2 [concrete]
// CHECK:STDOUT:       %u.param_patt: @A.WithSelf.F.%pattern_type.loc6_19 (%pattern_type.986) = value_param_pattern [concrete]
// CHECK:STDOUT:       %u.patt: @A.WithSelf.F.%pattern_type.loc6_19 (%pattern_type.986) = at_binding_pattern u, %u.param_patt [concrete]
// CHECK:STDOUT:       %return.param_patt: @A.WithSelf.F.%pattern_type.loc6_40 (%pattern_type.274) = out_param_pattern [concrete]
// CHECK:STDOUT:       %return.patt: @A.WithSelf.F.%pattern_type.loc6_40 (%pattern_type.274) = return_slot_pattern %return.param_patt, %.loc6_40.4 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @A.%T.loc5_14.2 [symbolic = %T (constants.%T.67d)]
// CHECK:STDOUT:       %.loc6_32: @A.WithSelf.F.%A.type (%A.type.ef5) = specific_constant @A.%Self.loc5_23.1, @A(constants.%T.67d) [symbolic = %Self (constants.%Self.0bb)]
// CHECK:STDOUT:       %Self.ref: @A.WithSelf.F.%A.type (%A.type.ef5) = name_ref Self, %.loc6_32 [symbolic = %Self (constants.%Self.0bb)]
// CHECK:STDOUT:       %U.ref.loc6_38: type = name_ref U, %U.loc6_9.2 [symbolic = %U.loc6_9.1 (constants.%U.4e7)]
// CHECK:STDOUT:       %ptr.loc6_39: type = ptr_type %U.ref.loc6_38 [symbolic = %ptr.loc6_22.1 (constants.%ptr.8f6)]
// CHECK:STDOUT:       %.loc6_40.2: @A.WithSelf.F.%tuple.type.loc6_40.1 (%tuple.type.721) = tuple_literal (%T.ref, %Self.ref, %ptr.loc6_39) [symbolic = %tuple (constants.%tuple.149)]
// CHECK:STDOUT:       %Self.as_type.loc6_40.2: type = facet_access_type constants.%Self.0bb [symbolic = %Self.as_type.loc6_40.1 (constants.%Self.as_type.2f2)]
// CHECK:STDOUT:       %.loc6_40.3: type = converted constants.%Self.0bb, %Self.as_type.loc6_40.2 [symbolic = %Self.as_type.loc6_40.1 (constants.%Self.as_type.2f2)]
// CHECK:STDOUT:       %.loc6_40.4: type = converted %.loc6_40.2, constants.%tuple.type.c48 [symbolic = %tuple.type.loc6_40.2 (constants.%tuple.type.c48)]
// CHECK:STDOUT:       %.loc6_40.5: Core.Form = init_form %.loc6_40.4 [symbolic = %.loc6_40.1 (constants.%.973)]
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
// CHECK:STDOUT:       %u: @A.WithSelf.F.%ptr.loc6_22.1 (%ptr.8f6) = value_binding u, %u.param
// CHECK:STDOUT:       %return.param: ref @A.WithSelf.F.%tuple.type.loc6_40.2 (%tuple.type.c48) = out_param call_param1
// CHECK:STDOUT:       %return: ref @A.WithSelf.F.%tuple.type.loc6_40.2 (%tuple.type.c48) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc6_41.1: @A.WithSelf.%A.assoc_type (%A.assoc_type.24d) = assoc_entity element0, %A.WithSelf.F.decl [symbolic = %assoc0.loc6_41.2 (constants.%assoc0.ecf)]
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
// CHECK:STDOUT:     %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 0 [concrete]
// CHECK:STDOUT:     %u.param_patt: @Y.as.A.impl.F.%pattern_type.loc16_19 (%pattern_type.4f4b84.1) = value_param_pattern [concrete]
// CHECK:STDOUT:     %u.patt: @Y.as.A.impl.F.%pattern_type.loc16_19 (%pattern_type.4f4b84.1) = at_binding_pattern u, %u.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: @Y.as.A.impl.F.%pattern_type.loc16_37 (%pattern_type.832) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @Y.as.A.impl.F.%pattern_type.loc16_37 (%pattern_type.832) = return_slot_pattern %return.param_patt, %.loc16_37.3 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %Y.ref: type = name_ref Y, file.%Y.decl [concrete = constants.%Y]
// CHECK:STDOUT:     %U.ref.loc16_35: type = name_ref U, %U.loc16_9.2 [symbolic = %U.loc16_9.1 (constants.%U.67d)]
// CHECK:STDOUT:     %ptr.loc16_36: type = ptr_type %U.ref.loc16_35 [symbolic = %ptr.loc16_22.1 (constants.%ptr.e8f8f9.1)]
// CHECK:STDOUT:     %.loc16_37.2: %tuple.type.ff9 = tuple_literal (%X.ref, %Y.ref, %ptr.loc16_36) [symbolic = %tuple (constants.%tuple.e08)]
// CHECK:STDOUT:     %.loc16_37.3: type = converted %.loc16_37.2, constants.%tuple.type.711 [symbolic = %tuple.type.loc16 (constants.%tuple.type.711)]
// CHECK:STDOUT:     %.loc16_37.4: Core.Form = init_form %.loc16_37.3 [symbolic = %.loc16_37.1 (constants.%.f98)]
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
// CHECK:STDOUT:     %u: @Y.as.A.impl.F.%ptr.loc16_22.1 (%ptr.e8f8f9.1) = value_binding u, %u.param
// CHECK:STDOUT:     %return.param: ref @Y.as.A.impl.F.%tuple.type.loc16 (%tuple.type.711) = out_param call_param1
// CHECK:STDOUT:     %return: ref @Y.as.A.impl.F.%tuple.type.loc16 (%tuple.type.711) = return_slot %return.param
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
// CHECK:STDOUT: generic fn @A.WithSelf.F(@A.%T.loc5_14.2: type, @A.%Self.loc5_23.1: @A.%A.type (%A.type.ef5), %U.loc6_9.2: type) {
// CHECK:STDOUT:   %U.loc6_9.1: type = symbolic_binding U, 2 [symbolic = %U.loc6_9.1 (constants.%U.4e7)]
// CHECK:STDOUT:   %ptr.loc6_22.1: type = ptr_type %U.loc6_9.1 [symbolic = %ptr.loc6_22.1 (constants.%ptr.8f6)]
// CHECK:STDOUT:   %pattern_type.loc6_19: type = pattern_type %ptr.loc6_22.1 [symbolic = %pattern_type.loc6_19 (constants.%pattern_type.986)]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T.67d)]
// CHECK:STDOUT:   %A.type: type = facet_type <@A, @A(%T)> [symbolic = %A.type (constants.%A.type.ef5)]
// CHECK:STDOUT:   %Self: @A.WithSelf.F.%A.type (%A.type.ef5) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.0bb)]
// CHECK:STDOUT:   %tuple.type.loc6_40.1: type = tuple_type (type, %A.type, type) [symbolic = %tuple.type.loc6_40.1 (constants.%tuple.type.721)]
// CHECK:STDOUT:   %tuple: @A.WithSelf.F.%tuple.type.loc6_40.1 (%tuple.type.721) = tuple_value (%T, %Self, %ptr.loc6_22.1) [symbolic = %tuple (constants.%tuple.149)]
// CHECK:STDOUT:   %Self.as_type.loc6_40.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc6_40.1 (constants.%Self.as_type.2f2)]
// CHECK:STDOUT:   %tuple.type.loc6_40.2: type = tuple_type (%T, %Self.as_type.loc6_40.1, %ptr.loc6_22.1) [symbolic = %tuple.type.loc6_40.2 (constants.%tuple.type.c48)]
// CHECK:STDOUT:   %.loc6_40.1: Core.Form = init_form %tuple.type.loc6_40.2 [symbolic = %.loc6_40.1 (constants.%.973)]
// CHECK:STDOUT:   %pattern_type.loc6_40: type = pattern_type %tuple.type.loc6_40.2 [symbolic = %pattern_type.loc6_40 (constants.%pattern_type.274)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param: @A.WithSelf.F.%ptr.loc6_22.1 (%ptr.8f6)) -> out %return.param: @A.WithSelf.F.%tuple.type.loc6_40.2 (%tuple.type.c48);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Y.as.A.impl.F(%U.loc16_9.2: type) {
// CHECK:STDOUT:   %U.loc16_9.1: type = symbolic_binding U, 0 [symbolic = %U.loc16_9.1 (constants.%U.67d)]
// CHECK:STDOUT:   %ptr.loc16_22.1: type = ptr_type %U.loc16_9.1 [symbolic = %ptr.loc16_22.1 (constants.%ptr.e8f8f9.1)]
// CHECK:STDOUT:   %pattern_type.loc16_19: type = pattern_type %ptr.loc16_22.1 [symbolic = %pattern_type.loc16_19 (constants.%pattern_type.4f4b84.1)]
// CHECK:STDOUT:   %tuple: %tuple.type.ff9 = tuple_value (constants.%X, constants.%Y, %ptr.loc16_22.1) [symbolic = %tuple (constants.%tuple.e08)]
// CHECK:STDOUT:   %tuple.type.loc16: type = tuple_type (constants.%X, constants.%Y, %ptr.loc16_22.1) [symbolic = %tuple.type.loc16 (constants.%tuple.type.711)]
// CHECK:STDOUT:   %.loc16_37.1: Core.Form = init_form %tuple.type.loc16 [symbolic = %.loc16_37.1 (constants.%.f98)]
// CHECK:STDOUT:   %pattern_type.loc16_37: type = pattern_type %tuple.type.loc16 [symbolic = %pattern_type.loc16_37 (constants.%pattern_type.832)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc16_19: <witness> = require_complete_type %ptr.loc16_22.1 [symbolic = %require_complete.loc16_19 (constants.%require_complete.ef162c.1)]
// CHECK:STDOUT:   %require_complete.loc16_25: <witness> = require_complete_type %tuple.type.loc16 [symbolic = %require_complete.loc16_25 (constants.%require_complete.f4e)]
// CHECK:STDOUT:   %tuple.type.loc17: type = tuple_type (constants.%empty_struct_type, constants.%empty_struct_type, %ptr.loc16_22.1) [symbolic = %tuple.type.loc17 (constants.%tuple.type.dd2)]
// CHECK:STDOUT:   %.loc17_21.1: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%U.loc16_9.1) [symbolic = %.loc17_21.1 (constants.%.748)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr.loc16_22.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.978)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.loc16_22.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet.b5f)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.172)]
// CHECK:STDOUT:   %.loc17_21.2: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %Copy.facet [symbolic = %.loc17_21.2 (constants.%.60c)]
// CHECK:STDOUT:   %impl.elem0.loc17_21.2: @Y.as.A.impl.F.%.loc17_21.2 (%.60c) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc17_21.2 (constants.%impl.elem0.adc)]
// CHECK:STDOUT:   %specific_impl_fn.loc17_21.2: <specific function> = specific_impl_function %impl.elem0.loc17_21.2, @Copy.WithSelf.Op(%Copy.facet) [symbolic = %specific_impl_fn.loc17_21.2 (constants.%specific_impl_fn.dbc)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param: @Y.as.A.impl.F.%ptr.loc16_22.1 (%ptr.e8f8f9.1)) -> out %return.param: @Y.as.A.impl.F.%tuple.type.loc16 (%tuple.type.711) {
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
// CHECK:STDOUT:     %impl.elem0.loc17_21.1: @Y.as.A.impl.F.%.loc17_21.2 (%.60c) = impl_witness_access constants.%Copy.lookup_impl_witness.978, element0 [symbolic = %impl.elem0.loc17_21.2 (constants.%impl.elem0.adc)]
// CHECK:STDOUT:     %bound_method.loc17_21.1: <bound method> = bound_method %u.ref, %impl.elem0.loc17_21.1
// CHECK:STDOUT:     %specific_impl_fn.loc17_21.1: <specific function> = specific_impl_function %impl.elem0.loc17_21.1, @Copy.WithSelf.Op(constants.%Copy.facet.b5f) [symbolic = %specific_impl_fn.loc17_21.2 (constants.%specific_impl_fn.dbc)]
// CHECK:STDOUT:     %bound_method.loc17_21.2: <bound method> = bound_method %u.ref, %specific_impl_fn.loc17_21.1
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @Y.as.A.impl.F.%ptr.loc16_22.1 (%ptr.e8f8f9.1) = call %bound_method.loc17_21.2(%u.ref)
// CHECK:STDOUT:     %tuple.elem2: ref @Y.as.A.impl.F.%ptr.loc16_22.1 (%ptr.e8f8f9.1) = tuple_access %return.param, element2
// CHECK:STDOUT:     %.loc17_22.4: init @Y.as.A.impl.F.%ptr.loc16_22.1 (%ptr.e8f8f9.1) to %tuple.elem2 = in_place_init %Copy.WithSelf.Op.call
// CHECK:STDOUT:     %.loc17_22.5: init @Y.as.A.impl.F.%tuple.type.loc16 (%tuple.type.711) to %return.param = tuple_init (%.loc17_22.2, %.loc17_22.3, %.loc17_22.4)
// CHECK:STDOUT:     %.loc17_23: init @Y.as.A.impl.F.%tuple.type.loc16 (%tuple.type.711) = converted %.loc17_22.1, %.loc17_22.5
// CHECK:STDOUT:     return %.loc17_23 to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %u.patt: %pattern_type.dd4 = ref_binding_pattern u [concrete]
// CHECK:STDOUT:     %u.var_patt: %pattern_type.dd4 = var_pattern %u.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %u.var: ref %Z = var %u.var_patt
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value constants.%Z, (constants.%DefaultOrUnformed.impl_witness.ece) [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %.loc22_11.1: %DefaultOrUnformed.type = converted constants.%Z, %DefaultOrUnformed.facet [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %as_type.loc22: type = facet_access_type %.loc22_11.1 [concrete = constants.%Z]
// CHECK:STDOUT:   %.loc22_11.2: type = converted %.loc22_11.1, %as_type.loc22 [concrete = constants.%Z]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.specific_fn: <specific function> = specific_function constants.%T.as.DefaultOrUnformed.impl.Op.cb6, @T.as.DefaultOrUnformed.impl.Op(constants.%Z) [concrete = constants.%T.as.DefaultOrUnformed.impl.Op.specific_fn]
// CHECK:STDOUT:   %.loc22_3: ref %Z = splice_block %u.var {}
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.call: init %Z to %.loc22_3 = call %T.as.DefaultOrUnformed.impl.Op.specific_fn()
// CHECK:STDOUT:   assign %u.var, %T.as.DefaultOrUnformed.impl.Op.call
// CHECK:STDOUT:   %Z.ref.loc22: type = name_ref Z, file.%Z.decl [concrete = constants.%Z]
// CHECK:STDOUT:   %u: ref %Z = ref_binding u, %u.var
// CHECK:STDOUT:   %Y.ref: type = name_ref Y, file.%Y.decl [concrete = constants.%Y]
// CHECK:STDOUT:   %A.ref: %A.type.9ad = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:   %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %A.type: type = facet_type <@A, @A(constants.%X)> [concrete = constants.%A.type.1f4]
// CHECK:STDOUT:   %A.facet: %A.type.1f4 = facet_value %Y.ref, (constants.%A.impl_witness) [concrete = constants.%A.facet.0c5]
// CHECK:STDOUT:   %.loc23_6: %A.type.1f4 = converted %Y.ref, %A.facet [concrete = constants.%A.facet.0c5]
// CHECK:STDOUT:   %as_type.loc23: type = facet_access_type %.loc23_6 [concrete = constants.%Y]
// CHECK:STDOUT:   %.loc23_14.1: type = converted %.loc23_6, %as_type.loc23 [concrete = constants.%Y]
// CHECK:STDOUT:   %.loc23_14.2: %A.assoc_type.409 = specific_constant @A.WithSelf.%assoc0.loc6_41.1, @A.WithSelf(constants.%X, constants.%A.facet.0c5) [concrete = constants.%assoc0.4b8]
// CHECK:STDOUT:   %F.ref: %A.assoc_type.409 = name_ref F, %.loc23_14.2 [concrete = constants.%assoc0.4b8]
// CHECK:STDOUT:   %impl.elem0: %.c69 = impl_witness_access constants.%A.impl_witness, element0 [concrete = constants.%Y.as.A.impl.F]
// CHECK:STDOUT:   %Z.ref.loc23: type = name_ref Z, file.%Z.decl [concrete = constants.%Z]
// CHECK:STDOUT:   %u.ref: ref %Z = name_ref u, %u
// CHECK:STDOUT:   %addr: %ptr.b15 = addr_of %u.ref
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Y.as.A.impl.F(constants.%Z) [concrete = constants.%Y.as.A.impl.F.specific_fn]
// CHECK:STDOUT:   %.loc23_22.1: ref %tuple.type.348 = temporary_storage
// CHECK:STDOUT:   %Y.as.A.impl.F.call: init %tuple.type.348 to %.loc23_22.1 = call %specific_fn(%addr)
// CHECK:STDOUT:   %.loc23_22.2: ref %tuple.type.348 = temporary %.loc23_22.1, %Y.as.A.impl.F.call
// CHECK:STDOUT:   %Destroy.Op.bound.loc23: <bound method> = bound_method %.loc23_22.2, constants.%Destroy.Op.f0ba5f.5
// CHECK:STDOUT:   %Destroy.Op.call.loc23: init %empty_tuple.type = call %Destroy.Op.bound.loc23(%.loc23_22.2)
// CHECK:STDOUT:   %Destroy.Op.bound.loc22: <bound method> = bound_method %u.var, constants.%Destroy.Op.f0ba5f.6
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
// CHECK:STDOUT: fn @Destroy.Op.loc23_22.4(%self.param: ref %ptr.b15) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc23_22.5(%self.param: ref %tuple.type.348) {
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
// CHECK:STDOUT:   %T.loc26_17.1: %facet_type = symbolic_binding T, 0 [symbolic = %T.loc26_17.1 (constants.%T.0ab)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc28_4.2: type = facet_access_type %T.loc26_17.1 [symbolic = %T.as_type.loc28_4.2 (constants.%T.as_type.010)]
// CHECK:STDOUT:   %A.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc26_17.1, @A, @A(constants.%X) [symbolic = %A.lookup_impl_witness (constants.%A.lookup_impl_witness)]
// CHECK:STDOUT:   %A.facet: %A.type.1f4 = facet_value %T.as_type.loc28_4.2, (%A.lookup_impl_witness) [symbolic = %A.facet (constants.%A.facet.88a)]
// CHECK:STDOUT:   %A.WithSelf.F.type: type = fn_type @A.WithSelf.F, @A.WithSelf(constants.%X, %A.facet) [symbolic = %A.WithSelf.F.type (constants.%A.WithSelf.F.type.9a6)]
// CHECK:STDOUT:   %.loc28_4.3: type = fn_type_with_self_type %A.WithSelf.F.type, %A.facet [symbolic = %.loc28_4.3 (constants.%.ae0)]
// CHECK:STDOUT:   %impl.elem0.loc28_4.2: @CallGeneric.%.loc28_4.3 (%.ae0) = impl_witness_access %A.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc28_4.2 (constants.%impl.elem0.4fc)]
// CHECK:STDOUT:   %specific_impl_fn.loc28_4.2: <specific function> = specific_impl_function %impl.elem0.loc28_4.2, @A.WithSelf.F(constants.%X, %A.facet, constants.%Z) [symbolic = %specific_impl_fn.loc28_4.2 (constants.%specific_impl_fn.5bf)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (constants.%X, %T.as_type.loc28_4.2, constants.%ptr.b15) [symbolic = %tuple.type (constants.%tuple.type.27a)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %tuple.type [symbolic = %require_complete (constants.%require_complete.703)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %tuple.type, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.33f)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %tuple.type, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet (constants.%Destroy.facet.7eb)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.0a4)]
// CHECK:STDOUT:   %.loc28_12.3: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Destroy.facet [symbolic = %.loc28_12.3 (constants.%.7ed)]
// CHECK:STDOUT:   %impl.elem0.loc28_12.2: @CallGeneric.%.loc28_12.3 (%.7ed) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc28_12.2 (constants.%impl.elem0.ad1)]
// CHECK:STDOUT:   %specific_impl_fn.loc28_12.2: <specific function> = specific_impl_function %impl.elem0.loc28_12.2, @Destroy.WithSelf.Op(%Destroy.facet) [symbolic = %specific_impl_fn.loc28_12.2 (constants.%specific_impl_fn.f4d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %u.patt: %pattern_type.dd4 = ref_binding_pattern u [concrete]
// CHECK:STDOUT:       %u.var_patt: %pattern_type.dd4 = var_pattern %u.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u.var: ref %Z = var %u.var_patt
// CHECK:STDOUT:     %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value constants.%Z, (constants.%DefaultOrUnformed.impl_witness.ece) [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:     %.loc27_11.1: %DefaultOrUnformed.type = converted constants.%Z, %DefaultOrUnformed.facet [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:     %as_type: type = facet_access_type %.loc27_11.1 [concrete = constants.%Z]
// CHECK:STDOUT:     %.loc27_11.2: type = converted %.loc27_11.1, %as_type [concrete = constants.%Z]
// CHECK:STDOUT:     %T.as.DefaultOrUnformed.impl.Op.specific_fn: <specific function> = specific_function constants.%T.as.DefaultOrUnformed.impl.Op.cb6, @T.as.DefaultOrUnformed.impl.Op(constants.%Z) [concrete = constants.%T.as.DefaultOrUnformed.impl.Op.specific_fn]
// CHECK:STDOUT:     %.loc27_3: ref %Z = splice_block %u.var {}
// CHECK:STDOUT:     %T.as.DefaultOrUnformed.impl.Op.call: init %Z to %.loc27_3 = call %T.as.DefaultOrUnformed.impl.Op.specific_fn()
// CHECK:STDOUT:     assign %u.var, %T.as.DefaultOrUnformed.impl.Op.call
// CHECK:STDOUT:     %Z.ref.loc27: type = name_ref Z, file.%Z.decl [concrete = constants.%Z]
// CHECK:STDOUT:     %u: ref %Z = ref_binding u, %u.var
// CHECK:STDOUT:     %T.ref: %facet_type = name_ref T, %T.loc26_17.2 [symbolic = %T.loc26_17.1 (constants.%T.0ab)]
// CHECK:STDOUT:     %T.as_type.loc28_4.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc28_4.2 (constants.%T.as_type.010)]
// CHECK:STDOUT:     %.loc28_4.1: type = converted %T.ref, %T.as_type.loc28_4.1 [symbolic = %T.as_type.loc28_4.2 (constants.%T.as_type.010)]
// CHECK:STDOUT:     %.loc28_4.2: %A.assoc_type.409 = specific_constant @A.WithSelf.%assoc0.loc6_41.1, @A.WithSelf(constants.%X, constants.%T.0ab) [concrete = constants.%assoc0.4b8]
// CHECK:STDOUT:     %F.ref: %A.assoc_type.409 = name_ref F, %.loc28_4.2 [concrete = constants.%assoc0.4b8]
// CHECK:STDOUT:     %impl.elem0.loc28_4.1: @CallGeneric.%.loc28_4.3 (%.ae0) = impl_witness_access constants.%A.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc28_4.2 (constants.%impl.elem0.4fc)]
// CHECK:STDOUT:     %Z.ref.loc28: type = name_ref Z, file.%Z.decl [concrete = constants.%Z]
// CHECK:STDOUT:     %u.ref: ref %Z = name_ref u, %u
// CHECK:STDOUT:     %addr: %ptr.b15 = addr_of %u.ref
// CHECK:STDOUT:     %specific_impl_fn.loc28_4.1: <specific function> = specific_impl_function %impl.elem0.loc28_4.1, @A.WithSelf.F(constants.%X, constants.%A.facet.88a, constants.%Z) [symbolic = %specific_impl_fn.loc28_4.2 (constants.%specific_impl_fn.5bf)]
// CHECK:STDOUT:     %.loc28_12.1: ref @CallGeneric.%tuple.type (%tuple.type.27a) = temporary_storage
// CHECK:STDOUT:     %A.WithSelf.F.call: init @CallGeneric.%tuple.type (%tuple.type.27a) to %.loc28_12.1 = call %specific_impl_fn.loc28_4.1(%addr)
// CHECK:STDOUT:     %.loc28_12.2: ref @CallGeneric.%tuple.type (%tuple.type.27a) = temporary %.loc28_12.1, %A.WithSelf.F.call
// CHECK:STDOUT:     %impl.elem0.loc28_12.1: @CallGeneric.%.loc28_12.3 (%.7ed) = impl_witness_access constants.%Destroy.lookup_impl_witness.33f, element0 [symbolic = %impl.elem0.loc28_12.2 (constants.%impl.elem0.ad1)]
// CHECK:STDOUT:     %bound_method.loc28_12.1: <bound method> = bound_method %.loc28_12.2, %impl.elem0.loc28_12.1
// CHECK:STDOUT:     %specific_impl_fn.loc28_12.1: <specific function> = specific_impl_function %impl.elem0.loc28_12.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.7eb) [symbolic = %specific_impl_fn.loc28_12.2 (constants.%specific_impl_fn.f4d)]
// CHECK:STDOUT:     %bound_method.loc28_12.2: <bound method> = bound_method %.loc28_12.2, %specific_impl_fn.loc28_12.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc28_12.2(%.loc28_12.2)
// CHECK:STDOUT:     %Destroy.Op.bound: <bound method> = bound_method %u.var, constants.%Destroy.Op.f0ba5f.6
// CHECK:STDOUT:     %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%u.var)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallIndirect() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CallGeneric.ref: %CallGeneric.type = name_ref CallGeneric, file.%CallGeneric.decl [concrete = constants.%CallGeneric]
// CHECK:STDOUT:   %Y.ref: type = name_ref Y, file.%Y.decl [concrete = constants.%Y]
// CHECK:STDOUT:   %facet_value: %facet_type = facet_value %Y.ref, (constants.%A.impl_witness, constants.%custom_witness.1367bd.3) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc32: %facet_type = converted %Y.ref, %facet_value [concrete = constants.%facet_value]
// CHECK:STDOUT:   %CallGeneric.specific_fn: <specific function> = specific_function %CallGeneric.ref, @CallGeneric(constants.%facet_value) [concrete = constants.%CallGeneric.specific_fn]
// CHECK:STDOUT:   %CallGeneric.call: init %empty_tuple.type = call %CallGeneric.specific_fn()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%T.67d) {
// CHECK:STDOUT:   %T.loc5_14.1 => constants.%T.67d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%T.67d, constants.%Self.0bb) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf.F(constants.%T.67d, constants.%Self.0bb, constants.%U.4e7) {
// CHECK:STDOUT:   %U.loc6_9.1 => constants.%U.4e7
// CHECK:STDOUT:   %ptr.loc6_22.1 => constants.%ptr.8f6
// CHECK:STDOUT:   %pattern_type.loc6_19 => constants.%pattern_type.986
// CHECK:STDOUT:   %T => constants.%T.67d
// CHECK:STDOUT:   %A.type => constants.%A.type.ef5
// CHECK:STDOUT:   %Self => constants.%Self.0bb
// CHECK:STDOUT:   %tuple.type.loc6_40.1 => constants.%tuple.type.721
// CHECK:STDOUT:   %tuple => constants.%tuple.149
// CHECK:STDOUT:   %Self.as_type.loc6_40.1 => constants.%Self.as_type.2f2
// CHECK:STDOUT:   %tuple.type.loc6_40.2 => constants.%tuple.type.c48
// CHECK:STDOUT:   %.loc6_40.1 => constants.%.973
// CHECK:STDOUT:   %pattern_type.loc6_40 => constants.%pattern_type.274
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%X) {
// CHECK:STDOUT:   %T.loc5_14.1 => constants.%X
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.type => constants.%A.type.1f4
// CHECK:STDOUT:   %Self.loc5_23.2 => constants.%Self.edd
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%X, constants.%Self.0bb) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.1f4
// CHECK:STDOUT:   %Self => constants.%Self.0bb
// CHECK:STDOUT:   %A.WithSelf.F.type => constants.%A.WithSelf.F.type.82f
// CHECK:STDOUT:   %A.WithSelf.F => constants.%A.WithSelf.F.7f8
// CHECK:STDOUT:   %A.assoc_type => constants.%A.assoc_type.409
// CHECK:STDOUT:   %assoc0.loc6_41.2 => constants.%assoc0.4b8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y.as.A.impl.F(constants.%U.67d) {
// CHECK:STDOUT:   %U.loc16_9.1 => constants.%U.67d
// CHECK:STDOUT:   %ptr.loc16_22.1 => constants.%ptr.e8f8f9.1
// CHECK:STDOUT:   %pattern_type.loc16_19 => constants.%pattern_type.4f4b84.1
// CHECK:STDOUT:   %tuple => constants.%tuple.e08
// CHECK:STDOUT:   %tuple.type.loc16 => constants.%tuple.type.711
// CHECK:STDOUT:   %.loc16_37.1 => constants.%.f98
// CHECK:STDOUT:   %pattern_type.loc16_37 => constants.%pattern_type.832
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%X, constants.%A.facet.0c5) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.1f4
// CHECK:STDOUT:   %Self => constants.%A.facet.0c5
// CHECK:STDOUT:   %A.WithSelf.F.type => constants.%A.WithSelf.F.type.423
// CHECK:STDOUT:   %A.WithSelf.F => constants.%A.WithSelf.F.90b
// CHECK:STDOUT:   %A.assoc_type => constants.%A.assoc_type.409
// CHECK:STDOUT:   %assoc0.loc6_41.2 => constants.%assoc0.4b8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf.F(constants.%X, constants.%A.facet.0c5, constants.%U.67d) {
// CHECK:STDOUT:   %U.loc6_9.1 => constants.%U.67d
// CHECK:STDOUT:   %ptr.loc6_22.1 => constants.%ptr.e8f8f9.1
// CHECK:STDOUT:   %pattern_type.loc6_19 => constants.%pattern_type.4f4b84.1
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.1f4
// CHECK:STDOUT:   %Self => constants.%A.facet.0c5
// CHECK:STDOUT:   %tuple.type.loc6_40.1 => constants.%tuple.type.9b2
// CHECK:STDOUT:   %tuple => constants.%tuple.afd
// CHECK:STDOUT:   %Self.as_type.loc6_40.1 => constants.%Y
// CHECK:STDOUT:   %tuple.type.loc6_40.2 => constants.%tuple.type.711
// CHECK:STDOUT:   %.loc6_40.1 => constants.%.f98
// CHECK:STDOUT:   %pattern_type.loc6_40 => constants.%pattern_type.832
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y.as.A.impl.F(constants.%Z) {
// CHECK:STDOUT:   %U.loc16_9.1 => constants.%Z
// CHECK:STDOUT:   %ptr.loc16_22.1 => constants.%ptr.b15
// CHECK:STDOUT:   %pattern_type.loc16_19 => constants.%pattern_type.9d3
// CHECK:STDOUT:   %tuple => constants.%tuple.469
// CHECK:STDOUT:   %tuple.type.loc16 => constants.%tuple.type.348
// CHECK:STDOUT:   %.loc16_37.1 => constants.%.40c
// CHECK:STDOUT:   %pattern_type.loc16_37 => constants.%pattern_type.b7b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc16_19 => constants.%complete_type.873
// CHECK:STDOUT:   %require_complete.loc16_25 => constants.%complete_type.79c
// CHECK:STDOUT:   %tuple.type.loc17 => constants.%tuple.type.e15
// CHECK:STDOUT:   %.loc17_21.1 => constants.%.eb0
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.impl_witness.9c3
// CHECK:STDOUT:   %Copy.facet => constants.%Copy.facet.3da
// CHECK:STDOUT:   %Copy.WithSelf.Op.type => constants.%Copy.WithSelf.Op.type.6ab
// CHECK:STDOUT:   %.loc17_21.2 => constants.%.f67
// CHECK:STDOUT:   %impl.elem0.loc17_21.2 => constants.%ptr.as.Copy.impl.Op.7d4
// CHECK:STDOUT:   %specific_impl_fn.loc17_21.2 => constants.%ptr.as.Copy.impl.Op.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGeneric(constants.%T.0ab) {
// CHECK:STDOUT:   %T.loc26_17.1 => constants.%T.0ab
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%X, constants.%T.0ab) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.1f4
// CHECK:STDOUT:   %Self => constants.%T.0ab
// CHECK:STDOUT:   %A.WithSelf.F.type => constants.%A.WithSelf.F.type.cf1
// CHECK:STDOUT:   %A.WithSelf.F => constants.%A.WithSelf.F.a39
// CHECK:STDOUT:   %A.assoc_type => constants.%A.assoc_type.409
// CHECK:STDOUT:   %assoc0.loc6_41.2 => constants.%assoc0.4b8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%X, constants.%A.facet.88a) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.1f4
// CHECK:STDOUT:   %Self => constants.%A.facet.88a
// CHECK:STDOUT:   %A.WithSelf.F.type => constants.%A.WithSelf.F.type.9a6
// CHECK:STDOUT:   %A.WithSelf.F => constants.%A.WithSelf.F.de8
// CHECK:STDOUT:   %A.assoc_type => constants.%A.assoc_type.409
// CHECK:STDOUT:   %assoc0.loc6_41.2 => constants.%assoc0.4b8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf.F(constants.%X, constants.%A.facet.88a, constants.%Z) {
// CHECK:STDOUT:   %U.loc6_9.1 => constants.%Z
// CHECK:STDOUT:   %ptr.loc6_22.1 => constants.%ptr.b15
// CHECK:STDOUT:   %pattern_type.loc6_19 => constants.%pattern_type.9d3
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.1f4
// CHECK:STDOUT:   %Self => constants.%A.facet.88a
// CHECK:STDOUT:   %tuple.type.loc6_40.1 => constants.%tuple.type.9b2
// CHECK:STDOUT:   %tuple => constants.%tuple.e42
// CHECK:STDOUT:   %Self.as_type.loc6_40.1 => constants.%T.as_type.010
// CHECK:STDOUT:   %tuple.type.loc6_40.2 => constants.%tuple.type.27a
// CHECK:STDOUT:   %.loc6_40.1 => constants.%.52d
// CHECK:STDOUT:   %pattern_type.loc6_40 => constants.%pattern_type.0a1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGeneric(constants.%facet_value) {
// CHECK:STDOUT:   %T.loc26_17.1 => constants.%facet_value
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc28_4.2 => constants.%Y
// CHECK:STDOUT:   %A.lookup_impl_witness => constants.%A.impl_witness
// CHECK:STDOUT:   %A.facet => constants.%A.facet.0c5
// CHECK:STDOUT:   %A.WithSelf.F.type => constants.%A.WithSelf.F.type.423
// CHECK:STDOUT:   %.loc28_4.3 => constants.%.c69
// CHECK:STDOUT:   %impl.elem0.loc28_4.2 => constants.%Y.as.A.impl.F
// CHECK:STDOUT:   %specific_impl_fn.loc28_4.2 => constants.%Y.as.A.impl.F.specific_fn
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.348
// CHECK:STDOUT:   %require_complete => constants.%complete_type.79c
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.1367bd.5
// CHECK:STDOUT:   %Destroy.facet => constants.%Destroy.facet.21e
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.af0
// CHECK:STDOUT:   %.loc28_12.3 => constants.%.0ae
// CHECK:STDOUT:   %impl.elem0.loc28_12.2 => constants.%Destroy.Op.f0ba5f.5
// CHECK:STDOUT:   %specific_impl_fn.loc28_12.2 => constants.%Destroy.Op.f0ba5f.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf.F(constants.%X, constants.%A.facet.0c5, constants.%Z) {
// CHECK:STDOUT:   %U.loc6_9.1 => constants.%Z
// CHECK:STDOUT:   %ptr.loc6_22.1 => constants.%ptr.b15
// CHECK:STDOUT:   %pattern_type.loc6_19 => constants.%pattern_type.9d3
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.1f4
// CHECK:STDOUT:   %Self => constants.%A.facet.0c5
// CHECK:STDOUT:   %tuple.type.loc6_40.1 => constants.%tuple.type.9b2
// CHECK:STDOUT:   %tuple => constants.%tuple.b11
// CHECK:STDOUT:   %Self.as_type.loc6_40.1 => constants.%Y
// CHECK:STDOUT:   %tuple.type.loc6_40.2 => constants.%tuple.type.348
// CHECK:STDOUT:   %.loc6_40.1 => constants.%.40c
// CHECK:STDOUT:   %pattern_type.loc6_40 => constants.%pattern_type.b7b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- generic_method_more_params.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %A.type.9ad: type = generic_interface_type @A [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %A.generic: %A.type.9ad = struct_value () [concrete]
// CHECK:STDOUT:   %A.type.ef5: type = facet_type <@A, @A(%T.67d)> [symbolic]
// CHECK:STDOUT:   %Self.0bb: %A.type.ef5 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %U.4e7: type = symbolic_binding U, 2 [symbolic]
// CHECK:STDOUT:   %pattern_type.62e: type = pattern_type %U.4e7 [symbolic]
// CHECK:STDOUT:   %tuple.type.721: type = tuple_type (type, %A.type.ef5, type) [symbolic]
// CHECK:STDOUT:   %tuple.147: %tuple.type.721 = tuple_value (%T.67d, %Self.0bb, %U.4e7) [symbolic]
// CHECK:STDOUT:   %Self.as_type.2f2: type = facet_access_type %Self.0bb [symbolic]
// CHECK:STDOUT:   %tuple.type.849: type = tuple_type (%T.67d, %Self.as_type.2f2, %U.4e7) [symbolic]
// CHECK:STDOUT:   %.481: Core.Form = init_form %tuple.type.849 [symbolic]
// CHECK:STDOUT:   %pattern_type.e19: type = pattern_type %tuple.type.849 [symbolic]
// CHECK:STDOUT:   %A.WithSelf.F.type.8f1: type = fn_type @A.WithSelf.F, @A.WithSelf(%T.67d, %Self.0bb) [symbolic]
// CHECK:STDOUT:   %A.WithSelf.F.d3d: %A.WithSelf.F.type.8f1 = struct_value () [symbolic]
// CHECK:STDOUT:   %A.assoc_type.24d: type = assoc_entity_type @A, @A(%T.67d) [symbolic]
// CHECK:STDOUT:   %assoc0.ecf: %A.assoc_type.24d = assoc_entity element0, @A.WithSelf.%A.WithSelf.F.decl [symbolic]
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Y1: type = class_type @Y1 [concrete]
// CHECK:STDOUT:   %Y2: type = class_type @Y2 [concrete]
// CHECK:STDOUT:   %Z: type = class_type @Z [concrete]
// CHECK:STDOUT:   %V1: type = symbolic_binding V1, 0 [symbolic]
// CHECK:STDOUT:   %V2: type = symbolic_binding V2, 1 [symbolic]
// CHECK:STDOUT:   %W: type = symbolic_binding W, 2 [symbolic]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.4b9: %tuple.type.24b = tuple_value (%V1, %V2) [symbolic]
// CHECK:STDOUT:   %tuple.type.a5e: type = tuple_type (%V1, %V2) [symbolic]
// CHECK:STDOUT:   %A.type.6c4: type = facet_type <@A, @A(%W)> [symbolic]
// CHECK:STDOUT:   %A.impl_witness.83f: <witness> = impl_witness @tuple.type.as.A.impl.%A.impl_witness_table, @tuple.type.as.A.impl(%V1, %V2, %W) [symbolic]
// CHECK:STDOUT:   %Self.aca: %A.type.6c4 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %A.WithSelf.F.type.bda: type = fn_type @A.WithSelf.F, @A.WithSelf(%W, %Self.0bb) [symbolic]
// CHECK:STDOUT:   %A.WithSelf.F.7b6: %A.WithSelf.F.type.bda = struct_value () [symbolic]
// CHECK:STDOUT:   %A.assoc_type.b99: type = assoc_entity_type @A, @A(%W) [symbolic]
// CHECK:STDOUT:   %assoc0.fa0: %A.assoc_type.b99 = assoc_entity element0, @A.WithSelf.%A.WithSelf.F.decl [symbolic]
// CHECK:STDOUT:   %require_complete.42b: <witness> = require_complete_type %A.type.6c4 [symbolic]
// CHECK:STDOUT:   %U.ce4: type = symbolic_binding U, 3 [symbolic]
// CHECK:STDOUT:   %pattern_type.e4a: type = pattern_type %U.ce4 [symbolic]
// CHECK:STDOUT:   %tuple.type.11f: type = tuple_type (type, %tuple.type.24b, type) [concrete]
// CHECK:STDOUT:   %tuple.117: %tuple.type.11f = tuple_value (%W, %tuple.4b9, %U.ce4) [symbolic]
// CHECK:STDOUT:   %tuple.type.897: type = tuple_type (%W, %tuple.type.a5e, %U.ce4) [symbolic]
// CHECK:STDOUT:   %.f8e: Core.Form = init_form %tuple.type.897 [symbolic]
// CHECK:STDOUT:   %pattern_type.5cc: type = pattern_type %tuple.type.897 [symbolic]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type.61d: type = fn_type @tuple.type.as.A.impl.F, @tuple.type.as.A.impl(%V1, %V2, %W) [symbolic]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.f96: %tuple.type.as.A.impl.F.type.61d = struct_value () [symbolic]
// CHECK:STDOUT:   %A.facet.b23: %A.type.6c4 = facet_value %tuple.type.a5e, (%A.impl_witness.83f) [symbolic]
// CHECK:STDOUT:   %A.WithSelf.F.type.9ed: type = fn_type @A.WithSelf.F, @A.WithSelf(%W, %A.facet.b23) [symbolic]
// CHECK:STDOUT:   %A.WithSelf.F.0ed: %A.WithSelf.F.type.9ed = struct_value () [symbolic]
// CHECK:STDOUT:   %tuple.type.afb: type = tuple_type (type, %A.type.6c4, type) [symbolic]
// CHECK:STDOUT:   %tuple.367: %tuple.type.afb = tuple_value (%W, %A.facet.b23, %U.ce4) [symbolic]
// CHECK:STDOUT:   %require_complete.5a4: <witness> = require_complete_type %U.ce4 [symbolic]
// CHECK:STDOUT:   %require_complete.da0: <witness> = require_complete_type %tuple.type.897 [symbolic]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.specific_fn.fb3: <specific function> = specific_function %tuple.type.as.A.impl.F.f96, @tuple.type.as.A.impl.F(%V1, %V2, %W, %U.ce4) [symbolic]
// CHECK:STDOUT:   %Call.type: type = fn_type @Call [concrete]
// CHECK:STDOUT:   %Call: %Call.type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.78f: %tuple.type.24b = tuple_value (%Y1, %Y2) [concrete]
// CHECK:STDOUT:   %tuple.type.490: type = tuple_type (%Y1, %Y2) [concrete]
// CHECK:STDOUT:   %A.type.1f4: type = facet_type <@A, @A(%X)> [concrete]
// CHECK:STDOUT:   %A.impl_witness.e70: <witness> = impl_witness @tuple.type.as.A.impl.%A.impl_witness_table, @tuple.type.as.A.impl(%Y1, %Y2, %X) [concrete]
// CHECK:STDOUT:   %Self.edd: %A.type.1f4 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %A.WithSelf.F.type.82f: type = fn_type @A.WithSelf.F, @A.WithSelf(%X, %Self.0bb) [symbolic]
// CHECK:STDOUT:   %A.WithSelf.F.7f8: %A.WithSelf.F.type.82f = struct_value () [symbolic]
// CHECK:STDOUT:   %A.assoc_type.409: type = assoc_entity_type @A, @A(%X) [concrete]
// CHECK:STDOUT:   %assoc0.4b8: %A.assoc_type.409 = assoc_entity element0, @A.WithSelf.%A.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %complete_type.722: <witness> = complete_type_witness %A.type.1f4 [concrete]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type.906: type = fn_type @tuple.type.as.A.impl.F, @tuple.type.as.A.impl(%Y1, %Y2, %X) [concrete]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.0e7: %tuple.type.as.A.impl.F.type.906 = struct_value () [concrete]
// CHECK:STDOUT:   %A.facet.56d: %A.type.1f4 = facet_value %tuple.type.490, (%A.impl_witness.e70) [concrete]
// CHECK:STDOUT:   %A.WithSelf.F.type.e10: type = fn_type @A.WithSelf.F, @A.WithSelf(%X, %A.facet.56d) [concrete]
// CHECK:STDOUT:   %A.WithSelf.F.f71: %A.WithSelf.F.type.e10 = struct_value () [concrete]
// CHECK:STDOUT:   %.be3: type = fn_type_with_self_type %A.WithSelf.F.type.e10, %A.facet.56d [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.dd4: type = pattern_type %Z [concrete]
// CHECK:STDOUT:   %tuple.3be: %tuple.type.11f = tuple_value (%X, %tuple.78f, %Z) [concrete]
// CHECK:STDOUT:   %tuple.type.514: type = tuple_type (%X, %tuple.type.490, %Z) [concrete]
// CHECK:STDOUT:   %.845: Core.Form = init_form %tuple.type.514 [concrete]
// CHECK:STDOUT:   %pattern_type.790: type = pattern_type %tuple.type.514 [concrete]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.specific_fn.58e: <specific function> = specific_function %tuple.type.as.A.impl.F.0e7, @tuple.type.as.A.impl.F(%Y1, %Y2, %X, %Z) [concrete]
// CHECK:STDOUT:   %Z.val: %Z = struct_value () [concrete]
// CHECK:STDOUT:   %.bef: ref %Z = temporary invalid, %Z.val [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.3f5f78.5: type = fn_type @Destroy.Op.loc24_39.5 [concrete]
// CHECK:STDOUT:   %Destroy.Op.f0ba5f.5: %Destroy.Op.type.3f5f78.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.1367bd.5: <witness> = custom_witness (%Destroy.Op.f0ba5f.5), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.3f5f78.6: type = fn_type @Destroy.Op.loc24_39.6 [concrete]
// CHECK:STDOUT:   %Destroy.Op.f0ba5f.6: %Destroy.Op.type.3f5f78.6 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.3f5f78.7: type = fn_type @Destroy.Op.loc24_39.7 [concrete]
// CHECK:STDOUT:   %Destroy.Op.f0ba5f.7: %Destroy.Op.type.3f5f78.7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.1367bd.7: <witness> = custom_witness (%Destroy.Op.f0ba5f.7), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.4e6: %Destroy.type = facet_value %tuple.type.514, (%custom_witness.1367bd.7) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.275: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.4e6) [concrete]
// CHECK:STDOUT:   %.f2d: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.275, %Destroy.facet.4e6 [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.bef, %Destroy.Op.f0ba5f.6 [concrete]
// CHECK:STDOUT:   %BitAndWith.type.447: type = generic_interface_type @BitAndWith [concrete]
// CHECK:STDOUT:   %BitAndWith.generic: %BitAndWith.type.447 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndWith.type.d17: type = facet_type <@BitAndWith, @BitAndWith(type)> [concrete]
// CHECK:STDOUT:   %BitAndWith.impl_witness: <witness> = impl_witness imports.%BitAndWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %BitAndWith.facet: %BitAndWith.type.d17 = facet_value type, (%BitAndWith.impl_witness) [concrete]
// CHECK:STDOUT:   %BitAndWith.WithSelf.Op.type.3c5: type = fn_type @BitAndWith.WithSelf.Op, @BitAndWith.WithSelf(type, %BitAndWith.facet) [concrete]
// CHECK:STDOUT:   %.849: type = fn_type_with_self_type %BitAndWith.WithSelf.Op.type.3c5, %BitAndWith.facet [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.type: type = fn_type @type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op: %type.as.BitAndWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.bound: <bound method> = bound_method %A.type.1f4, %type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@A, @A(%X) & @Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.689: type = pattern_type %facet_type [concrete]
// CHECK:STDOUT:   %T.0ab: %facet_type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %CallGeneric.type: type = fn_type @CallGeneric [concrete]
// CHECK:STDOUT:   %CallGeneric: %CallGeneric.type = struct_value () [concrete]
// CHECK:STDOUT:   %A.WithSelf.F.type.cf1: type = fn_type @A.WithSelf.F, @A.WithSelf(%X, %T.0ab) [symbolic]
// CHECK:STDOUT:   %A.WithSelf.F.a39: %A.WithSelf.F.type.cf1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.0ab [symbolic]
// CHECK:STDOUT:   %A.lookup_impl_witness: <witness> = lookup_impl_witness %T.0ab, @A, @A(%X) [symbolic]
// CHECK:STDOUT:   %A.facet.88a: %A.type.1f4 = facet_value %T.as_type, (%A.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %A.WithSelf.F.type.9a6: type = fn_type @A.WithSelf.F, @A.WithSelf(%X, %A.facet.88a) [symbolic]
// CHECK:STDOUT:   %A.WithSelf.F.de8: %A.WithSelf.F.type.9a6 = struct_value () [symbolic]
// CHECK:STDOUT:   %.ae0: type = fn_type_with_self_type %A.WithSelf.F.type.9a6, %A.facet.88a [symbolic]
// CHECK:STDOUT:   %impl.elem0.4fc: %.ae0 = impl_witness_access %A.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %tuple.type.9b2: type = tuple_type (type, %A.type.1f4, type) [concrete]
// CHECK:STDOUT:   %tuple.b0d: %tuple.type.9b2 = tuple_value (%X, %A.facet.88a, %Z) [symbolic]
// CHECK:STDOUT:   %tuple.type.d85: type = tuple_type (%X, %T.as_type, %Z) [symbolic]
// CHECK:STDOUT:   %.aa4: Core.Form = init_form %tuple.type.d85 [symbolic]
// CHECK:STDOUT:   %pattern_type.200: type = pattern_type %tuple.type.d85 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.5bf: <specific function> = specific_impl_function %impl.elem0.4fc, @A.WithSelf.F(%X, %A.facet.88a, %Z) [symbolic]
// CHECK:STDOUT:   %require_complete.227: <witness> = require_complete_type %tuple.type.d85 [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.2b3: <witness> = lookup_impl_witness %tuple.type.d85, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.23f: %Destroy.type = facet_value %tuple.type.d85, (%Destroy.lookup_impl_witness.2b3) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.a81: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.23f) [symbolic]
// CHECK:STDOUT:   %.a85: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.a81, %Destroy.facet.23f [symbolic]
// CHECK:STDOUT:   %impl.elem0.f12: %.a85 = impl_witness_access %Destroy.lookup_impl_witness.2b3, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.99f: <specific function> = specific_impl_function %impl.elem0.f12, @Destroy.WithSelf.Op(%Destroy.facet.23f) [symbolic]
// CHECK:STDOUT:   %CallIndirect.type: type = fn_type @CallIndirect [concrete]
// CHECK:STDOUT:   %CallIndirect: %CallIndirect.type = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value: %facet_type = facet_value %tuple.type.490, (%A.impl_witness.e70, %custom_witness.1367bd.5) [concrete]
// CHECK:STDOUT:   %CallGeneric.specific_fn: <specific function> = specific_function %CallGeneric, @CallGeneric(%facet_value) [concrete]
// CHECK:STDOUT:   %complete_type.465: <witness> = complete_type_witness %tuple.type.514 [concrete]
// CHECK:STDOUT:   %tuple.7d6: %tuple.type.9b2 = tuple_value (%X, %A.facet.56d, %Z) [concrete]
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
// CHECK:STDOUT:   %Core.BitAndWith: %BitAndWith.type.447 = import_ref Core//prelude/parts/as, BitAndWith, loaded [concrete = constants.%BitAndWith.generic]
// CHECK:STDOUT:   %Core.import_ref.4c9: %type.as.BitAndWith.impl.Op.type = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, loaded [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:   %BitAndWith.impl_witness_table = impl_witness_table (%Core.import_ref.4c9), @type.as.BitAndWith.impl [concrete]
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
// CHECK:STDOUT:   %A.decl: %A.type.9ad = interface_decl @A [concrete = constants.%A.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
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
// CHECK:STDOUT:     %V1.patt: %pattern_type.98f = symbolic_binding_pattern V1, 0 [concrete]
// CHECK:STDOUT:     %V2.patt: %pattern_type.98f = symbolic_binding_pattern V2, 1 [concrete]
// CHECK:STDOUT:     %W.patt: %pattern_type.98f = symbolic_binding_pattern W, 2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %V1.ref: type = name_ref V1, %V1.loc14_16.2 [symbolic = %V1.loc14_16.1 (constants.%V1)]
// CHECK:STDOUT:     %V2.ref: type = name_ref V2, %V2.loc14_27.2 [symbolic = %V2.loc14_27.1 (constants.%V2)]
// CHECK:STDOUT:     %.loc14_53.1: %tuple.type.24b = tuple_literal (%V1.ref, %V2.ref) [symbolic = %tuple (constants.%tuple.4b9)]
// CHECK:STDOUT:     %.loc14_53.2: type = converted %.loc14_53.1, constants.%tuple.type.a5e [symbolic = %tuple.type (constants.%tuple.type.a5e)]
// CHECK:STDOUT:     %A.ref: %A.type.9ad = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:     %W.ref: type = name_ref W, %W.loc14_37.2 [symbolic = %W.loc14_37.1 (constants.%W)]
// CHECK:STDOUT:     %A.type.loc14_61.2: type = facet_type <@A, @A(constants.%W)> [symbolic = %A.type.loc14_61.1 (constants.%A.type.6c4)]
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
// CHECK:STDOUT:     %T.patt: %pattern_type.689 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc27_25.1: type = splice_block %.loc27_25.3 [concrete = constants.%facet_type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %A.ref: %A.type.9ad = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:       %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:       %A.type: type = facet_type <@A, @A(constants.%X)> [concrete = constants.%A.type.1f4]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:       %impl.elem0.loc27: %.849 = impl_witness_access constants.%BitAndWith.impl_witness, element0 [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:       %bound_method.loc27: <bound method> = bound_method %A.type, %impl.elem0.loc27 [concrete = constants.%type.as.BitAndWith.impl.Op.bound]
// CHECK:STDOUT:       %type.as.BitAndWith.impl.Op.call: init type = call %bound_method.loc27(%A.type, %Destroy.ref) [concrete = constants.%facet_type]
// CHECK:STDOUT:       %.loc27_25.2: type = value_of_initializer %type.as.BitAndWith.impl.Op.call [concrete = constants.%facet_type]
// CHECK:STDOUT:       %.loc27_25.3: type = converted %type.as.BitAndWith.impl.Op.call, %.loc27_25.2 [concrete = constants.%facet_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc27_17.2: %facet_type = symbolic_binding T, 0 [symbolic = %T.loc27_17.1 (constants.%T.0ab)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallIndirect.decl: %CallIndirect.type = fn_decl @CallIndirect [concrete = constants.%CallIndirect] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @A(%T.loc5_14.2: type) {
// CHECK:STDOUT:   %T.loc5_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc5_14.1 (constants.%T.67d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.type: type = facet_type <@A, @A(%T.loc5_14.1)> [symbolic = %A.type (constants.%A.type.ef5)]
// CHECK:STDOUT:   %Self.loc5_23.2: @A.%A.type (%A.type.ef5) = symbolic_binding Self, 1 [symbolic = %Self.loc5_23.2 (constants.%Self.0bb)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc5_23.1: @A.%A.type (%A.type.ef5) = symbolic_binding Self, 1 [symbolic = %Self.loc5_23.2 (constants.%Self.0bb)]
// CHECK:STDOUT:     %A.WithSelf.decl = interface_with_self_decl @A [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %A.WithSelf.F.decl: @A.WithSelf.%A.WithSelf.F.type (%A.WithSelf.F.type.8f1) = fn_decl @A.WithSelf.F [symbolic = @A.WithSelf.%A.WithSelf.F (constants.%A.WithSelf.F.d3d)] {
// CHECK:STDOUT:       %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 2 [concrete]
// CHECK:STDOUT:       %u.param_patt: @A.WithSelf.F.%pattern_type.loc6_19 (%pattern_type.62e) = value_param_pattern [concrete]
// CHECK:STDOUT:       %u.patt: @A.WithSelf.F.%pattern_type.loc6_19 (%pattern_type.62e) = at_binding_pattern u, %u.param_patt [concrete]
// CHECK:STDOUT:       %return.param_patt: @A.WithSelf.F.%pattern_type.loc6_38 (%pattern_type.e19) = out_param_pattern [concrete]
// CHECK:STDOUT:       %return.patt: @A.WithSelf.F.%pattern_type.loc6_38 (%pattern_type.e19) = return_slot_pattern %return.param_patt, %.loc6_38.4 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @A.%T.loc5_14.2 [symbolic = %T (constants.%T.67d)]
// CHECK:STDOUT:       %.loc6_31: @A.WithSelf.F.%A.type (%A.type.ef5) = specific_constant @A.%Self.loc5_23.1, @A(constants.%T.67d) [symbolic = %Self (constants.%Self.0bb)]
// CHECK:STDOUT:       %Self.ref: @A.WithSelf.F.%A.type (%A.type.ef5) = name_ref Self, %.loc6_31 [symbolic = %Self (constants.%Self.0bb)]
// CHECK:STDOUT:       %U.ref.loc6_37: type = name_ref U, %U.loc6_9.2 [symbolic = %U.loc6_9.1 (constants.%U.4e7)]
// CHECK:STDOUT:       %.loc6_38.2: @A.WithSelf.F.%tuple.type.loc6_38.1 (%tuple.type.721) = tuple_literal (%T.ref, %Self.ref, %U.ref.loc6_37) [symbolic = %tuple (constants.%tuple.147)]
// CHECK:STDOUT:       %Self.as_type.loc6_38.2: type = facet_access_type constants.%Self.0bb [symbolic = %Self.as_type.loc6_38.1 (constants.%Self.as_type.2f2)]
// CHECK:STDOUT:       %.loc6_38.3: type = converted constants.%Self.0bb, %Self.as_type.loc6_38.2 [symbolic = %Self.as_type.loc6_38.1 (constants.%Self.as_type.2f2)]
// CHECK:STDOUT:       %.loc6_38.4: type = converted %.loc6_38.2, constants.%tuple.type.849 [symbolic = %tuple.type.loc6_38.2 (constants.%tuple.type.849)]
// CHECK:STDOUT:       %.loc6_38.5: Core.Form = init_form %.loc6_38.4 [symbolic = %.loc6_38.1 (constants.%.481)]
// CHECK:STDOUT:       %.loc6_12.1: type = splice_block %.loc6_12.2 [concrete = type] {
// CHECK:STDOUT:         %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:         %.loc6_12.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %U.loc6_9.2: type = symbolic_binding U, 2 [symbolic = %U.loc6_9.1 (constants.%U.4e7)]
// CHECK:STDOUT:       %u.param: @A.WithSelf.F.%U.loc6_9.1 (%U.4e7) = value_param call_param0
// CHECK:STDOUT:       %U.ref.loc6_21: type = name_ref U, %U.loc6_9.2 [symbolic = %U.loc6_9.1 (constants.%U.4e7)]
// CHECK:STDOUT:       %u: @A.WithSelf.F.%U.loc6_9.1 (%U.4e7) = value_binding u, %u.param
// CHECK:STDOUT:       %return.param: ref @A.WithSelf.F.%tuple.type.loc6_38.2 (%tuple.type.849) = out_param call_param1
// CHECK:STDOUT:       %return: ref @A.WithSelf.F.%tuple.type.loc6_38.2 (%tuple.type.849) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc6_39.1: @A.WithSelf.%A.assoc_type (%A.assoc_type.24d) = assoc_entity element0, %A.WithSelf.F.decl [symbolic = %assoc0.loc6_39.2 (constants.%assoc0.ecf)]
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
// CHECK:STDOUT:   %V1.loc14_16.1: type = symbolic_binding V1, 0 [symbolic = %V1.loc14_16.1 (constants.%V1)]
// CHECK:STDOUT:   %V2.loc14_27.1: type = symbolic_binding V2, 1 [symbolic = %V2.loc14_27.1 (constants.%V2)]
// CHECK:STDOUT:   %W.loc14_37.1: type = symbolic_binding W, 2 [symbolic = %W.loc14_37.1 (constants.%W)]
// CHECK:STDOUT:   %tuple: %tuple.type.24b = tuple_value (%V1.loc14_16.1, %V2.loc14_27.1) [symbolic = %tuple (constants.%tuple.4b9)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%V1.loc14_16.1, %V2.loc14_27.1) [symbolic = %tuple.type (constants.%tuple.type.a5e)]
// CHECK:STDOUT:   %A.type.loc14_61.1: type = facet_type <@A, @A(%W.loc14_37.1)> [symbolic = %A.type.loc14_61.1 (constants.%A.type.6c4)]
// CHECK:STDOUT:   %A.impl_witness.loc14_63.2: <witness> = impl_witness %A.impl_witness_table, @tuple.type.as.A.impl(%V1.loc14_16.1, %V2.loc14_27.1, %W.loc14_37.1) [symbolic = %A.impl_witness.loc14_63.2 (constants.%A.impl_witness.83f)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %A.type.loc14_61.1 [symbolic = %require_complete (constants.%require_complete.42b)]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type: type = fn_type @tuple.type.as.A.impl.F, @tuple.type.as.A.impl(%V1.loc14_16.1, %V2.loc14_27.1, %W.loc14_37.1) [symbolic = %tuple.type.as.A.impl.F.type (constants.%tuple.type.as.A.impl.F.type.61d)]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F: @tuple.type.as.A.impl.%tuple.type.as.A.impl.F.type (%tuple.type.as.A.impl.F.type.61d) = struct_value () [symbolic = %tuple.type.as.A.impl.F (constants.%tuple.type.as.A.impl.F.f96)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc14_53.2 as %A.type.loc14_61.2 {
// CHECK:STDOUT:     %tuple.type.as.A.impl.F.decl: @tuple.type.as.A.impl.%tuple.type.as.A.impl.F.type (%tuple.type.as.A.impl.F.type.61d) = fn_decl @tuple.type.as.A.impl.F [symbolic = @tuple.type.as.A.impl.%tuple.type.as.A.impl.F (constants.%tuple.type.as.A.impl.F.f96)] {
// CHECK:STDOUT:       %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 3 [concrete]
// CHECK:STDOUT:       %u.param_patt: @tuple.type.as.A.impl.F.%pattern_type.loc17_19 (%pattern_type.e4a) = value_param_pattern [concrete]
// CHECK:STDOUT:       %u.patt: @tuple.type.as.A.impl.F.%pattern_type.loc17_19 (%pattern_type.e4a) = at_binding_pattern u, %u.param_patt [concrete]
// CHECK:STDOUT:       %return.param_patt: @tuple.type.as.A.impl.F.%pattern_type.loc17_42 (%pattern_type.5cc) = out_param_pattern [concrete]
// CHECK:STDOUT:       %return.patt: @tuple.type.as.A.impl.F.%pattern_type.loc17_42 (%pattern_type.5cc) = return_slot_pattern %return.param_patt, %.loc17_42.5 [concrete]
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
// CHECK:STDOUT:       %u: @tuple.type.as.A.impl.F.%U.loc17_9.1 (%U.ce4) = value_binding u, %u.param
// CHECK:STDOUT:       %return.param: ref @tuple.type.as.A.impl.F.%tuple.type.loc17_42.2 (%tuple.type.897) = out_param call_param1
// CHECK:STDOUT:       %return: ref @tuple.type.as.A.impl.F.%tuple.type.loc17_42.2 (%tuple.type.897) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %A.impl_witness_table = impl_witness_table (%tuple.type.as.A.impl.F.decl), @tuple.type.as.A.impl [concrete]
// CHECK:STDOUT:     %A.impl_witness.loc14_63.1: <witness> = impl_witness %A.impl_witness_table, @tuple.type.as.A.impl(constants.%V1, constants.%V2, constants.%W) [symbolic = %A.impl_witness.loc14_63.2 (constants.%A.impl_witness.83f)]
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
// CHECK:STDOUT: generic fn @A.WithSelf.F(@A.%T.loc5_14.2: type, @A.%Self.loc5_23.1: @A.%A.type (%A.type.ef5), %U.loc6_9.2: type) {
// CHECK:STDOUT:   %U.loc6_9.1: type = symbolic_binding U, 2 [symbolic = %U.loc6_9.1 (constants.%U.4e7)]
// CHECK:STDOUT:   %pattern_type.loc6_19: type = pattern_type %U.loc6_9.1 [symbolic = %pattern_type.loc6_19 (constants.%pattern_type.62e)]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T.67d)]
// CHECK:STDOUT:   %A.type: type = facet_type <@A, @A(%T)> [symbolic = %A.type (constants.%A.type.ef5)]
// CHECK:STDOUT:   %Self: @A.WithSelf.F.%A.type (%A.type.ef5) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.0bb)]
// CHECK:STDOUT:   %tuple.type.loc6_38.1: type = tuple_type (type, %A.type, type) [symbolic = %tuple.type.loc6_38.1 (constants.%tuple.type.721)]
// CHECK:STDOUT:   %tuple: @A.WithSelf.F.%tuple.type.loc6_38.1 (%tuple.type.721) = tuple_value (%T, %Self, %U.loc6_9.1) [symbolic = %tuple (constants.%tuple.147)]
// CHECK:STDOUT:   %Self.as_type.loc6_38.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc6_38.1 (constants.%Self.as_type.2f2)]
// CHECK:STDOUT:   %tuple.type.loc6_38.2: type = tuple_type (%T, %Self.as_type.loc6_38.1, %U.loc6_9.1) [symbolic = %tuple.type.loc6_38.2 (constants.%tuple.type.849)]
// CHECK:STDOUT:   %.loc6_38.1: Core.Form = init_form %tuple.type.loc6_38.2 [symbolic = %.loc6_38.1 (constants.%.481)]
// CHECK:STDOUT:   %pattern_type.loc6_38: type = pattern_type %tuple.type.loc6_38.2 [symbolic = %pattern_type.loc6_38 (constants.%pattern_type.e19)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param: @A.WithSelf.F.%U.loc6_9.1 (%U.4e7)) -> out %return.param: @A.WithSelf.F.%tuple.type.loc6_38.2 (%tuple.type.849);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @tuple.type.as.A.impl.F(@tuple.type.as.A.impl.%V1.loc14_16.2: type, @tuple.type.as.A.impl.%V2.loc14_27.2: type, @tuple.type.as.A.impl.%W.loc14_37.2: type, %U.loc17_9.2: type) {
// CHECK:STDOUT:   %U.loc17_9.1: type = symbolic_binding U, 3 [symbolic = %U.loc17_9.1 (constants.%U.ce4)]
// CHECK:STDOUT:   %pattern_type.loc17_19: type = pattern_type %U.loc17_9.1 [symbolic = %pattern_type.loc17_19 (constants.%pattern_type.e4a)]
// CHECK:STDOUT:   %W: type = symbolic_binding W, 2 [symbolic = %W (constants.%W)]
// CHECK:STDOUT:   %V1: type = symbolic_binding V1, 0 [symbolic = %V1 (constants.%V1)]
// CHECK:STDOUT:   %V2: type = symbolic_binding V2, 1 [symbolic = %V2 (constants.%V2)]
// CHECK:STDOUT:   %tuple.loc17_38: %tuple.type.24b = tuple_value (%V1, %V2) [symbolic = %tuple.loc17_38 (constants.%tuple.4b9)]
// CHECK:STDOUT:   %tuple.loc17_42: %tuple.type.11f = tuple_value (%W, %tuple.loc17_38, %U.loc17_9.1) [symbolic = %tuple.loc17_42 (constants.%tuple.117)]
// CHECK:STDOUT:   %tuple.type.loc17_42.1: type = tuple_type (%V1, %V2) [symbolic = %tuple.type.loc17_42.1 (constants.%tuple.type.a5e)]
// CHECK:STDOUT:   %tuple.type.loc17_42.2: type = tuple_type (%W, %tuple.type.loc17_42.1, %U.loc17_9.1) [symbolic = %tuple.type.loc17_42.2 (constants.%tuple.type.897)]
// CHECK:STDOUT:   %.loc17_42.2: Core.Form = init_form %tuple.type.loc17_42.2 [symbolic = %.loc17_42.2 (constants.%.f8e)]
// CHECK:STDOUT:   %pattern_type.loc17_42: type = pattern_type %tuple.type.loc17_42.2 [symbolic = %pattern_type.loc17_42 (constants.%pattern_type.5cc)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc17_19: <witness> = require_complete_type %U.loc17_9.1 [symbolic = %require_complete.loc17_19 (constants.%require_complete.5a4)]
// CHECK:STDOUT:   %require_complete.loc17_24: <witness> = require_complete_type %tuple.type.loc17_42.2 [symbolic = %require_complete.loc17_24 (constants.%require_complete.da0)]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type: type = fn_type @tuple.type.as.A.impl.F, @tuple.type.as.A.impl(%V1, %V2, %W) [symbolic = %tuple.type.as.A.impl.F.type (constants.%tuple.type.as.A.impl.F.type.61d)]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F: @tuple.type.as.A.impl.F.%tuple.type.as.A.impl.F.type (%tuple.type.as.A.impl.F.type.61d) = struct_value () [symbolic = %tuple.type.as.A.impl.F (constants.%tuple.type.as.A.impl.F.f96)]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.specific_fn.loc18_12.2: <specific function> = specific_function %tuple.type.as.A.impl.F, @tuple.type.as.A.impl.F(%V1, %V2, %W, %U.loc17_9.1) [symbolic = %tuple.type.as.A.impl.F.specific_fn.loc18_12.2 (constants.%tuple.type.as.A.impl.F.specific_fn.fb3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param: @tuple.type.as.A.impl.F.%U.loc17_9.1 (%U.ce4)) -> out %return.param: @tuple.type.as.A.impl.F.%tuple.type.loc17_42.2 (%tuple.type.897) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc18: @tuple.type.as.A.impl.F.%tuple.type.as.A.impl.F.type (%tuple.type.as.A.impl.F.type.61d) = specific_constant @tuple.type.as.A.impl.%tuple.type.as.A.impl.F.decl, @tuple.type.as.A.impl(constants.%V1, constants.%V2, constants.%W) [symbolic = %tuple.type.as.A.impl.F (constants.%tuple.type.as.A.impl.F.f96)]
// CHECK:STDOUT:     %F.ref: @tuple.type.as.A.impl.F.%tuple.type.as.A.impl.F.type (%tuple.type.as.A.impl.F.type.61d) = name_ref F, %.loc18 [symbolic = %tuple.type.as.A.impl.F (constants.%tuple.type.as.A.impl.F.f96)]
// CHECK:STDOUT:     %U.ref.loc18: type = name_ref U, %U.loc17_9.2 [symbolic = %U.loc17_9.1 (constants.%U.ce4)]
// CHECK:STDOUT:     %u.ref: @tuple.type.as.A.impl.F.%U.loc17_9.1 (%U.ce4) = name_ref u, %u
// CHECK:STDOUT:     %tuple.type.as.A.impl.F.specific_fn.loc18_12.1: <specific function> = specific_function %F.ref, @tuple.type.as.A.impl.F(constants.%V1, constants.%V2, constants.%W, constants.%U.ce4) [symbolic = %tuple.type.as.A.impl.F.specific_fn.loc18_12.2 (constants.%tuple.type.as.A.impl.F.specific_fn.fb3)]
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
// CHECK:STDOUT:   %.loc24_12: %tuple.type.24b = tuple_literal (%Y1.ref, %Y2.ref) [concrete = constants.%tuple.78f]
// CHECK:STDOUT:   %.loc24_17: type = type_literal type [concrete = type]
// CHECK:STDOUT:   %.loc24_14: type = converted %.loc24_12, constants.%tuple.type.490 [concrete = constants.%tuple.type.490]
// CHECK:STDOUT:   %A.ref: %A.type.9ad = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:   %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %A.type: type = facet_type <@A, @A(constants.%X)> [concrete = constants.%A.type.1f4]
// CHECK:STDOUT:   %A.facet: %A.type.1f4 = facet_value %.loc24_14, (constants.%A.impl_witness.e70) [concrete = constants.%A.facet.56d]
// CHECK:STDOUT:   %.loc24_23: %A.type.1f4 = converted %.loc24_14, %A.facet [concrete = constants.%A.facet.56d]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc24_23 [concrete = constants.%tuple.type.490]
// CHECK:STDOUT:   %.loc24_31.1: type = converted %.loc24_23, %as_type [concrete = constants.%tuple.type.490]
// CHECK:STDOUT:   %.loc24_31.2: %A.assoc_type.409 = specific_constant @A.WithSelf.%assoc0.loc6_39.1, @A.WithSelf(constants.%X, constants.%A.facet.56d) [concrete = constants.%assoc0.4b8]
// CHECK:STDOUT:   %F.ref: %A.assoc_type.409 = name_ref F, %.loc24_31.2 [concrete = constants.%assoc0.4b8]
// CHECK:STDOUT:   %impl.elem0: %.be3 = impl_witness_access constants.%A.impl_witness.e70, element0 [concrete = constants.%tuple.type.as.A.impl.F.0e7]
// CHECK:STDOUT:   %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z]
// CHECK:STDOUT:   %.loc24_38.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @tuple.type.as.A.impl.F(constants.%Y1, constants.%Y2, constants.%X, constants.%Z) [concrete = constants.%tuple.type.as.A.impl.F.specific_fn.58e]
// CHECK:STDOUT:   %.loc24_39.1: ref %tuple.type.514 = temporary_storage
// CHECK:STDOUT:   %.loc24_38.2: ref %Z = temporary_storage
// CHECK:STDOUT:   %.loc24_38.3: init %Z to %.loc24_38.2 = class_init () [concrete = constants.%Z.val]
// CHECK:STDOUT:   %.loc24_38.4: init %Z = converted %.loc24_38.1, %.loc24_38.3 [concrete = constants.%Z.val]
// CHECK:STDOUT:   %.loc24_38.5: ref %Z = temporary %.loc24_38.2, %.loc24_38.4 [concrete = constants.%.bef]
// CHECK:STDOUT:   %.loc24_38.6: %Z = acquire_value %.loc24_38.5 [concrete = constants.%Z.val]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.call: init %tuple.type.514 to %.loc24_39.1 = call %specific_fn(%.loc24_38.6)
// CHECK:STDOUT:   %.loc24_39.2: ref %tuple.type.514 = temporary %.loc24_39.1, %tuple.type.as.A.impl.F.call
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.loc24_39.2, constants.%Destroy.Op.f0ba5f.7
// CHECK:STDOUT:   %Destroy.Op.call.loc24_39: init %empty_tuple.type = call %Destroy.Op.bound(%.loc24_39.2)
// CHECK:STDOUT:   %Destroy.Op.call.loc24_38: init %empty_tuple.type = call constants.%Destroy.Op.bound(constants.%.bef)
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
// CHECK:STDOUT: fn @Destroy.Op.loc24_39.5(%self.param: ref %tuple.type.490) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc24_39.6(%self.param: ref %Z) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc24_39.7(%self.param: ref %tuple.type.514) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallGeneric(%T.loc27_17.2: %facet_type) {
// CHECK:STDOUT:   %T.loc27_17.1: %facet_type = symbolic_binding T, 0 [symbolic = %T.loc27_17.1 (constants.%T.0ab)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc28_4.2: type = facet_access_type %T.loc27_17.1 [symbolic = %T.as_type.loc28_4.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %A.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc27_17.1, @A, @A(constants.%X) [symbolic = %A.lookup_impl_witness (constants.%A.lookup_impl_witness)]
// CHECK:STDOUT:   %A.facet: %A.type.1f4 = facet_value %T.as_type.loc28_4.2, (%A.lookup_impl_witness) [symbolic = %A.facet (constants.%A.facet.88a)]
// CHECK:STDOUT:   %A.WithSelf.F.type: type = fn_type @A.WithSelf.F, @A.WithSelf(constants.%X, %A.facet) [symbolic = %A.WithSelf.F.type (constants.%A.WithSelf.F.type.9a6)]
// CHECK:STDOUT:   %.loc28_4.3: type = fn_type_with_self_type %A.WithSelf.F.type, %A.facet [symbolic = %.loc28_4.3 (constants.%.ae0)]
// CHECK:STDOUT:   %impl.elem0.loc28_4.2: @CallGeneric.%.loc28_4.3 (%.ae0) = impl_witness_access %A.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc28_4.2 (constants.%impl.elem0.4fc)]
// CHECK:STDOUT:   %specific_impl_fn.loc28_4.2: <specific function> = specific_impl_function %impl.elem0.loc28_4.2, @A.WithSelf.F(constants.%X, %A.facet, constants.%Z) [symbolic = %specific_impl_fn.loc28_4.2 (constants.%specific_impl_fn.5bf)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (constants.%X, %T.as_type.loc28_4.2, constants.%Z) [symbolic = %tuple.type (constants.%tuple.type.d85)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %tuple.type [symbolic = %require_complete (constants.%require_complete.227)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %tuple.type, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.2b3)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %tuple.type, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet (constants.%Destroy.facet.23f)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.a81)]
// CHECK:STDOUT:   %.loc28_12.3: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Destroy.facet [symbolic = %.loc28_12.3 (constants.%.a85)]
// CHECK:STDOUT:   %impl.elem0.loc28_12.2: @CallGeneric.%.loc28_12.3 (%.a85) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc28_12.2 (constants.%impl.elem0.f12)]
// CHECK:STDOUT:   %specific_impl_fn.loc28_12.2: <specific function> = specific_impl_function %impl.elem0.loc28_12.2, @Destroy.WithSelf.Op(%Destroy.facet) [symbolic = %specific_impl_fn.loc28_12.2 (constants.%specific_impl_fn.99f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref: %facet_type = name_ref T, %T.loc27_17.2 [symbolic = %T.loc27_17.1 (constants.%T.0ab)]
// CHECK:STDOUT:     %T.as_type.loc28_4.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc28_4.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc28_4.1: type = converted %T.ref, %T.as_type.loc28_4.1 [symbolic = %T.as_type.loc28_4.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc28_4.2: %A.assoc_type.409 = specific_constant @A.WithSelf.%assoc0.loc6_39.1, @A.WithSelf(constants.%X, constants.%T.0ab) [concrete = constants.%assoc0.4b8]
// CHECK:STDOUT:     %F.ref: %A.assoc_type.409 = name_ref F, %.loc28_4.2 [concrete = constants.%assoc0.4b8]
// CHECK:STDOUT:     %impl.elem0.loc28_4.1: @CallGeneric.%.loc28_4.3 (%.ae0) = impl_witness_access constants.%A.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc28_4.2 (constants.%impl.elem0.4fc)]
// CHECK:STDOUT:     %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z]
// CHECK:STDOUT:     %.loc28_11.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %specific_impl_fn.loc28_4.1: <specific function> = specific_impl_function %impl.elem0.loc28_4.1, @A.WithSelf.F(constants.%X, constants.%A.facet.88a, constants.%Z) [symbolic = %specific_impl_fn.loc28_4.2 (constants.%specific_impl_fn.5bf)]
// CHECK:STDOUT:     %.loc28_12.1: ref @CallGeneric.%tuple.type (%tuple.type.d85) = temporary_storage
// CHECK:STDOUT:     %.loc28_11.2: ref %Z = temporary_storage
// CHECK:STDOUT:     %.loc28_11.3: init %Z to %.loc28_11.2 = class_init () [concrete = constants.%Z.val]
// CHECK:STDOUT:     %.loc28_11.4: init %Z = converted %.loc28_11.1, %.loc28_11.3 [concrete = constants.%Z.val]
// CHECK:STDOUT:     %.loc28_11.5: ref %Z = temporary %.loc28_11.2, %.loc28_11.4 [concrete = constants.%.bef]
// CHECK:STDOUT:     %.loc28_11.6: %Z = acquire_value %.loc28_11.5 [concrete = constants.%Z.val]
// CHECK:STDOUT:     %A.WithSelf.F.call: init @CallGeneric.%tuple.type (%tuple.type.d85) to %.loc28_12.1 = call %specific_impl_fn.loc28_4.1(%.loc28_11.6)
// CHECK:STDOUT:     %.loc28_12.2: ref @CallGeneric.%tuple.type (%tuple.type.d85) = temporary %.loc28_12.1, %A.WithSelf.F.call
// CHECK:STDOUT:     %impl.elem0.loc28_12.1: @CallGeneric.%.loc28_12.3 (%.a85) = impl_witness_access constants.%Destroy.lookup_impl_witness.2b3, element0 [symbolic = %impl.elem0.loc28_12.2 (constants.%impl.elem0.f12)]
// CHECK:STDOUT:     %bound_method.loc28_12.1: <bound method> = bound_method %.loc28_12.2, %impl.elem0.loc28_12.1
// CHECK:STDOUT:     %specific_impl_fn.loc28_12.1: <specific function> = specific_impl_function %impl.elem0.loc28_12.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.23f) [symbolic = %specific_impl_fn.loc28_12.2 (constants.%specific_impl_fn.99f)]
// CHECK:STDOUT:     %bound_method.loc28_12.2: <bound method> = bound_method %.loc28_12.2, %specific_impl_fn.loc28_12.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc28_12.2(%.loc28_12.2)
// CHECK:STDOUT:     %Destroy.Op.call: init %empty_tuple.type = call constants.%Destroy.Op.bound(constants.%.bef)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallIndirect() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CallGeneric.ref: %CallGeneric.type = name_ref CallGeneric, file.%CallGeneric.decl [concrete = constants.%CallGeneric]
// CHECK:STDOUT:   %Y1.ref: type = name_ref Y1, file.%Y1.decl [concrete = constants.%Y1]
// CHECK:STDOUT:   %Y2.ref: type = name_ref Y2, file.%Y2.decl [concrete = constants.%Y2]
// CHECK:STDOUT:   %.loc33_22: %tuple.type.24b = tuple_literal (%Y1.ref, %Y2.ref) [concrete = constants.%tuple.78f]
// CHECK:STDOUT:   %.loc33_27: type = type_literal type [concrete = type]
// CHECK:STDOUT:   %.loc33_24: type = converted %.loc33_22, constants.%tuple.type.490 [concrete = constants.%tuple.type.490]
// CHECK:STDOUT:   %facet_value: %facet_type = facet_value %.loc33_24, (constants.%A.impl_witness.e70, constants.%custom_witness.1367bd.5) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc33_31: %facet_type = converted %.loc33_24, %facet_value [concrete = constants.%facet_value]
// CHECK:STDOUT:   %CallGeneric.specific_fn: <specific function> = specific_function %CallGeneric.ref, @CallGeneric(constants.%facet_value) [concrete = constants.%CallGeneric.specific_fn]
// CHECK:STDOUT:   %CallGeneric.call: init %empty_tuple.type = call %CallGeneric.specific_fn()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%T.67d) {
// CHECK:STDOUT:   %T.loc5_14.1 => constants.%T.67d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%T.67d, constants.%Self.0bb) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf.F(constants.%T.67d, constants.%Self.0bb, constants.%U.4e7) {
// CHECK:STDOUT:   %U.loc6_9.1 => constants.%U.4e7
// CHECK:STDOUT:   %pattern_type.loc6_19 => constants.%pattern_type.62e
// CHECK:STDOUT:   %T => constants.%T.67d
// CHECK:STDOUT:   %A.type => constants.%A.type.ef5
// CHECK:STDOUT:   %Self => constants.%Self.0bb
// CHECK:STDOUT:   %tuple.type.loc6_38.1 => constants.%tuple.type.721
// CHECK:STDOUT:   %tuple => constants.%tuple.147
// CHECK:STDOUT:   %Self.as_type.loc6_38.1 => constants.%Self.as_type.2f2
// CHECK:STDOUT:   %tuple.type.loc6_38.2 => constants.%tuple.type.849
// CHECK:STDOUT:   %.loc6_38.1 => constants.%.481
// CHECK:STDOUT:   %pattern_type.loc6_38 => constants.%pattern_type.e19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%W) {
// CHECK:STDOUT:   %T.loc5_14.1 => constants.%W
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.type => constants.%A.type.6c4
// CHECK:STDOUT:   %Self.loc5_23.2 => constants.%Self.aca
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @tuple.type.as.A.impl(constants.%V1, constants.%V2, constants.%W) {
// CHECK:STDOUT:   %V1.loc14_16.1 => constants.%V1
// CHECK:STDOUT:   %V2.loc14_27.1 => constants.%V2
// CHECK:STDOUT:   %W.loc14_37.1 => constants.%W
// CHECK:STDOUT:   %tuple => constants.%tuple.4b9
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.a5e
// CHECK:STDOUT:   %A.type.loc14_61.1 => constants.%A.type.6c4
// CHECK:STDOUT:   %A.impl_witness.loc14_63.2 => constants.%A.impl_witness.83f
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.42b
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type => constants.%tuple.type.as.A.impl.F.type.61d
// CHECK:STDOUT:   %tuple.type.as.A.impl.F => constants.%tuple.type.as.A.impl.F.f96
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%W, constants.%Self.0bb) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%W
// CHECK:STDOUT:   %A.type => constants.%A.type.6c4
// CHECK:STDOUT:   %Self => constants.%Self.0bb
// CHECK:STDOUT:   %A.WithSelf.F.type => constants.%A.WithSelf.F.type.bda
// CHECK:STDOUT:   %A.WithSelf.F => constants.%A.WithSelf.F.7b6
// CHECK:STDOUT:   %A.assoc_type => constants.%A.assoc_type.b99
// CHECK:STDOUT:   %assoc0.loc6_39.2 => constants.%assoc0.fa0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @tuple.type.as.A.impl.F(constants.%V1, constants.%V2, constants.%W, constants.%U.ce4) {
// CHECK:STDOUT:   %U.loc17_9.1 => constants.%U.ce4
// CHECK:STDOUT:   %pattern_type.loc17_19 => constants.%pattern_type.e4a
// CHECK:STDOUT:   %W => constants.%W
// CHECK:STDOUT:   %V1 => constants.%V1
// CHECK:STDOUT:   %V2 => constants.%V2
// CHECK:STDOUT:   %tuple.loc17_38 => constants.%tuple.4b9
// CHECK:STDOUT:   %tuple.loc17_42 => constants.%tuple.117
// CHECK:STDOUT:   %tuple.type.loc17_42.1 => constants.%tuple.type.a5e
// CHECK:STDOUT:   %tuple.type.loc17_42.2 => constants.%tuple.type.897
// CHECK:STDOUT:   %.loc17_42.2 => constants.%.f8e
// CHECK:STDOUT:   %pattern_type.loc17_42 => constants.%pattern_type.5cc
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc17_19 => constants.%require_complete.5a4
// CHECK:STDOUT:   %require_complete.loc17_24 => constants.%require_complete.da0
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type => constants.%tuple.type.as.A.impl.F.type.61d
// CHECK:STDOUT:   %tuple.type.as.A.impl.F => constants.%tuple.type.as.A.impl.F.f96
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.specific_fn.loc18_12.2 => constants.%tuple.type.as.A.impl.F.specific_fn.fb3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%W, constants.%A.facet.b23) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%W
// CHECK:STDOUT:   %A.type => constants.%A.type.6c4
// CHECK:STDOUT:   %Self => constants.%A.facet.b23
// CHECK:STDOUT:   %A.WithSelf.F.type => constants.%A.WithSelf.F.type.9ed
// CHECK:STDOUT:   %A.WithSelf.F => constants.%A.WithSelf.F.0ed
// CHECK:STDOUT:   %A.assoc_type => constants.%A.assoc_type.b99
// CHECK:STDOUT:   %assoc0.loc6_39.2 => constants.%assoc0.fa0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf.F(constants.%W, constants.%A.facet.b23, constants.%U.ce4) {
// CHECK:STDOUT:   %U.loc6_9.1 => constants.%U.ce4
// CHECK:STDOUT:   %pattern_type.loc6_19 => constants.%pattern_type.e4a
// CHECK:STDOUT:   %T => constants.%W
// CHECK:STDOUT:   %A.type => constants.%A.type.6c4
// CHECK:STDOUT:   %Self => constants.%A.facet.b23
// CHECK:STDOUT:   %tuple.type.loc6_38.1 => constants.%tuple.type.afb
// CHECK:STDOUT:   %tuple => constants.%tuple.367
// CHECK:STDOUT:   %Self.as_type.loc6_38.1 => constants.%tuple.type.a5e
// CHECK:STDOUT:   %tuple.type.loc6_38.2 => constants.%tuple.type.897
// CHECK:STDOUT:   %.loc6_38.1 => constants.%.f8e
// CHECK:STDOUT:   %pattern_type.loc6_38 => constants.%pattern_type.5cc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%X) {
// CHECK:STDOUT:   %T.loc5_14.1 => constants.%X
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.type => constants.%A.type.1f4
// CHECK:STDOUT:   %Self.loc5_23.2 => constants.%Self.edd
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @tuple.type.as.A.impl(constants.%Y1, constants.%Y2, constants.%X) {
// CHECK:STDOUT:   %V1.loc14_16.1 => constants.%Y1
// CHECK:STDOUT:   %V2.loc14_27.1 => constants.%Y2
// CHECK:STDOUT:   %W.loc14_37.1 => constants.%X
// CHECK:STDOUT:   %tuple => constants.%tuple.78f
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.490
// CHECK:STDOUT:   %A.type.loc14_61.1 => constants.%A.type.1f4
// CHECK:STDOUT:   %A.impl_witness.loc14_63.2 => constants.%A.impl_witness.e70
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.722
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type => constants.%tuple.type.as.A.impl.F.type.906
// CHECK:STDOUT:   %tuple.type.as.A.impl.F => constants.%tuple.type.as.A.impl.F.0e7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%X, constants.%Self.0bb) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.1f4
// CHECK:STDOUT:   %Self => constants.%Self.0bb
// CHECK:STDOUT:   %A.WithSelf.F.type => constants.%A.WithSelf.F.type.82f
// CHECK:STDOUT:   %A.WithSelf.F => constants.%A.WithSelf.F.7f8
// CHECK:STDOUT:   %A.assoc_type => constants.%A.assoc_type.409
// CHECK:STDOUT:   %assoc0.loc6_39.2 => constants.%assoc0.4b8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%X, constants.%A.facet.56d) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.1f4
// CHECK:STDOUT:   %Self => constants.%A.facet.56d
// CHECK:STDOUT:   %A.WithSelf.F.type => constants.%A.WithSelf.F.type.e10
// CHECK:STDOUT:   %A.WithSelf.F => constants.%A.WithSelf.F.f71
// CHECK:STDOUT:   %A.assoc_type => constants.%A.assoc_type.409
// CHECK:STDOUT:   %assoc0.loc6_39.2 => constants.%assoc0.4b8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @tuple.type.as.A.impl.F(constants.%Y1, constants.%Y2, constants.%X, constants.%Z) {
// CHECK:STDOUT:   %U.loc17_9.1 => constants.%Z
// CHECK:STDOUT:   %pattern_type.loc17_19 => constants.%pattern_type.dd4
// CHECK:STDOUT:   %W => constants.%X
// CHECK:STDOUT:   %V1 => constants.%Y1
// CHECK:STDOUT:   %V2 => constants.%Y2
// CHECK:STDOUT:   %tuple.loc17_38 => constants.%tuple.78f
// CHECK:STDOUT:   %tuple.loc17_42 => constants.%tuple.3be
// CHECK:STDOUT:   %tuple.type.loc17_42.1 => constants.%tuple.type.490
// CHECK:STDOUT:   %tuple.type.loc17_42.2 => constants.%tuple.type.514
// CHECK:STDOUT:   %.loc17_42.2 => constants.%.845
// CHECK:STDOUT:   %pattern_type.loc17_42 => constants.%pattern_type.790
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc17_19 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc17_24 => constants.%complete_type.465
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type => constants.%tuple.type.as.A.impl.F.type.906
// CHECK:STDOUT:   %tuple.type.as.A.impl.F => constants.%tuple.type.as.A.impl.F.0e7
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.specific_fn.loc18_12.2 => constants.%tuple.type.as.A.impl.F.specific_fn.58e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGeneric(constants.%T.0ab) {
// CHECK:STDOUT:   %T.loc27_17.1 => constants.%T.0ab
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%X, constants.%T.0ab) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.1f4
// CHECK:STDOUT:   %Self => constants.%T.0ab
// CHECK:STDOUT:   %A.WithSelf.F.type => constants.%A.WithSelf.F.type.cf1
// CHECK:STDOUT:   %A.WithSelf.F => constants.%A.WithSelf.F.a39
// CHECK:STDOUT:   %A.assoc_type => constants.%A.assoc_type.409
// CHECK:STDOUT:   %assoc0.loc6_39.2 => constants.%assoc0.4b8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%X, constants.%A.facet.88a) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.1f4
// CHECK:STDOUT:   %Self => constants.%A.facet.88a
// CHECK:STDOUT:   %A.WithSelf.F.type => constants.%A.WithSelf.F.type.9a6
// CHECK:STDOUT:   %A.WithSelf.F => constants.%A.WithSelf.F.de8
// CHECK:STDOUT:   %A.assoc_type => constants.%A.assoc_type.409
// CHECK:STDOUT:   %assoc0.loc6_39.2 => constants.%assoc0.4b8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf.F(constants.%X, constants.%A.facet.88a, constants.%Z) {
// CHECK:STDOUT:   %U.loc6_9.1 => constants.%Z
// CHECK:STDOUT:   %pattern_type.loc6_19 => constants.%pattern_type.dd4
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.1f4
// CHECK:STDOUT:   %Self => constants.%A.facet.88a
// CHECK:STDOUT:   %tuple.type.loc6_38.1 => constants.%tuple.type.9b2
// CHECK:STDOUT:   %tuple => constants.%tuple.b0d
// CHECK:STDOUT:   %Self.as_type.loc6_38.1 => constants.%T.as_type
// CHECK:STDOUT:   %tuple.type.loc6_38.2 => constants.%tuple.type.d85
// CHECK:STDOUT:   %.loc6_38.1 => constants.%.aa4
// CHECK:STDOUT:   %pattern_type.loc6_38 => constants.%pattern_type.200
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGeneric(constants.%facet_value) {
// CHECK:STDOUT:   %T.loc27_17.1 => constants.%facet_value
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc28_4.2 => constants.%tuple.type.490
// CHECK:STDOUT:   %A.lookup_impl_witness => constants.%A.impl_witness.e70
// CHECK:STDOUT:   %A.facet => constants.%A.facet.56d
// CHECK:STDOUT:   %A.WithSelf.F.type => constants.%A.WithSelf.F.type.e10
// CHECK:STDOUT:   %.loc28_4.3 => constants.%.be3
// CHECK:STDOUT:   %impl.elem0.loc28_4.2 => constants.%tuple.type.as.A.impl.F.0e7
// CHECK:STDOUT:   %specific_impl_fn.loc28_4.2 => constants.%tuple.type.as.A.impl.F.specific_fn.58e
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.514
// CHECK:STDOUT:   %require_complete => constants.%complete_type.465
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.1367bd.7
// CHECK:STDOUT:   %Destroy.facet => constants.%Destroy.facet.4e6
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.275
// CHECK:STDOUT:   %.loc28_12.3 => constants.%.f2d
// CHECK:STDOUT:   %impl.elem0.loc28_12.2 => constants.%Destroy.Op.f0ba5f.7
// CHECK:STDOUT:   %specific_impl_fn.loc28_12.2 => constants.%Destroy.Op.f0ba5f.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf.F(constants.%X, constants.%A.facet.56d, constants.%Z) {
// CHECK:STDOUT:   %U.loc6_9.1 => constants.%Z
// CHECK:STDOUT:   %pattern_type.loc6_19 => constants.%pattern_type.dd4
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.1f4
// CHECK:STDOUT:   %Self => constants.%A.facet.56d
// CHECK:STDOUT:   %tuple.type.loc6_38.1 => constants.%tuple.type.9b2
// CHECK:STDOUT:   %tuple => constants.%tuple.7d6
// CHECK:STDOUT:   %Self.as_type.loc6_38.1 => constants.%tuple.type.490
// CHECK:STDOUT:   %tuple.type.loc6_38.2 => constants.%tuple.type.514
// CHECK:STDOUT:   %.loc6_38.1 => constants.%.845
// CHECK:STDOUT:   %pattern_type.loc6_38 => constants.%pattern_type.790
// CHECK:STDOUT: }
// CHECK:STDOUT:

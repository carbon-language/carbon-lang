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
// CHECK:STDOUT:   %A.type.0af: type = generic_interface_type @A [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %A.generic: %A.type.0af = struct_value () [concrete]
// CHECK:STDOUT:   %A.type.5ea: type = facet_type <@A, @A(%T.67d)> [symbolic]
// CHECK:STDOUT:   %Self.a33: %A.type.5ea = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %U.4e7: type = symbolic_binding U, 2 [symbolic]
// CHECK:STDOUT:   %ptr.8f6: type = ptr_type %U.4e7 [symbolic]
// CHECK:STDOUT:   %pattern_type.986: type = pattern_type %ptr.8f6 [symbolic]
// CHECK:STDOUT:   %tuple.type.54d: type = tuple_type (type, %A.type.5ea, type) [symbolic]
// CHECK:STDOUT:   %tuple.44b: %tuple.type.54d = tuple_value (%T.67d, %Self.a33, %ptr.8f6) [symbolic]
// CHECK:STDOUT:   %Self.as_type.0a5: type = facet_access_type %Self.a33 [symbolic]
// CHECK:STDOUT:   %tuple.type.67c: type = tuple_type (%T.67d, %Self.as_type.0a5, %ptr.8f6) [symbolic]
// CHECK:STDOUT:   %.f77: Core.Form = init_form %tuple.type.67c [symbolic]
// CHECK:STDOUT:   %pattern_type.050: type = pattern_type %tuple.type.67c [symbolic]
// CHECK:STDOUT:   %A.WithSelf.F.type.f0d: type = fn_type @A.WithSelf.F, @A.WithSelf(%T.67d, %Self.a33) [symbolic]
// CHECK:STDOUT:   %A.WithSelf.F.628: %A.WithSelf.F.type.f0d = struct_value () [symbolic]
// CHECK:STDOUT:   %A.assoc_type.092: type = assoc_entity_type @A, @A(%T.67d) [symbolic]
// CHECK:STDOUT:   %assoc0.7bb: %A.assoc_type.092 = assoc_entity element0, @A.WithSelf.%A.WithSelf.F.decl [symbolic]
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Y: type = class_type @Y [concrete]
// CHECK:STDOUT:   %Z: type = class_type @Z [concrete]
// CHECK:STDOUT:   %A.type.ef5: type = facet_type <@A, @A(%X)> [concrete]
// CHECK:STDOUT:   %A.impl_witness: <witness> = impl_witness @Y.as.A.impl.%A.impl_witness_table [concrete]
// CHECK:STDOUT:   %Self.0c0: %A.type.ef5 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %A.WithSelf.F.type.c12: type = fn_type @A.WithSelf.F, @A.WithSelf(%X, %Self.a33) [symbolic]
// CHECK:STDOUT:   %A.WithSelf.F.612: %A.WithSelf.F.type.c12 = struct_value () [symbolic]
// CHECK:STDOUT:   %A.assoc_type.3ce: type = assoc_entity_type @A, @A(%X) [concrete]
// CHECK:STDOUT:   %assoc0.2c2: %A.assoc_type.3ce = assoc_entity element0, @A.WithSelf.%A.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %U.67d: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %ptr.e8f8f9.1: type = ptr_type %U.67d [symbolic]
// CHECK:STDOUT:   %pattern_type.4f4b84.1: type = pattern_type %ptr.e8f8f9.1 [symbolic]
// CHECK:STDOUT:   %tuple.type.ff9: type = tuple_type (type, type, type) [concrete]
// CHECK:STDOUT:   %tuple.fdd: %tuple.type.ff9 = tuple_value (%X, %Y, %ptr.e8f8f9.1) [symbolic]
// CHECK:STDOUT:   %tuple.type.4b0: type = tuple_type (%X, %Y, %ptr.e8f8f9.1) [symbolic]
// CHECK:STDOUT:   %.de3: Core.Form = init_form %tuple.type.4b0 [symbolic]
// CHECK:STDOUT:   %pattern_type.145: type = pattern_type %tuple.type.4b0 [symbolic]
// CHECK:STDOUT:   %Y.as.A.impl.F.type: type = fn_type @Y.as.A.impl.F [concrete]
// CHECK:STDOUT:   %Y.as.A.impl.F: %Y.as.A.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %A.facet.027: %A.type.ef5 = facet_value %Y, (%A.impl_witness) [concrete]
// CHECK:STDOUT:   %A.WithSelf.F.type.d17: type = fn_type @A.WithSelf.F, @A.WithSelf(%X, %A.facet.027) [concrete]
// CHECK:STDOUT:   %A.WithSelf.F.a05: %A.WithSelf.F.type.d17 = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.d63: type = tuple_type (type, %A.type.ef5, type) [concrete]
// CHECK:STDOUT:   %tuple.6cb: %tuple.type.d63 = tuple_value (%X, %A.facet.027, %ptr.e8f8f9.1) [symbolic]
// CHECK:STDOUT:   %require_complete.ef162c.1: <witness> = require_complete_type %ptr.e8f8f9.1 [symbolic]
// CHECK:STDOUT:   %require_complete.4f8: <witness> = require_complete_type %tuple.type.4b0 [symbolic]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.dd2: type = tuple_type (%empty_struct_type, %empty_struct_type, %ptr.e8f8f9.1) [symbolic]
// CHECK:STDOUT:   %X.val: %X = struct_value () [concrete]
// CHECK:STDOUT:   %Y.val: %Y = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.aa424f.1: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.5a1c66.1: %ptr.as.Copy.impl.Op.type.aa424f.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.2bf: <witness> = lookup_impl_witness %ptr.e8f8f9.1, @Copy [symbolic]
// CHECK:STDOUT:   %.b6a: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%U.67d) [symbolic]
// CHECK:STDOUT:   %Copy.facet.44d: %Copy.type = facet_value %ptr.e8f8f9.1, (%Copy.lookup_impl_witness.2bf) [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.72b: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.44d) [symbolic]
// CHECK:STDOUT:   %.39b: type = fn_type_with_self_type %Copy.WithSelf.Op.type.72b, %Copy.facet.44d [symbolic]
// CHECK:STDOUT:   %impl.elem0.8a3: %.39b = impl_witness_access %Copy.lookup_impl_witness.2bf, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.a13: <specific function> = specific_impl_function %impl.elem0.8a3, @Copy.WithSelf.Op(%Copy.facet.44d) [symbolic]
// CHECK:STDOUT:   %Call.type: type = fn_type @Call [concrete]
// CHECK:STDOUT:   %Call: %Call.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.f0a: type = pattern_type %Z [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.type.bbe: type = fn_type @T.as.DefaultOrUnformed.impl.Op, @T.as.DefaultOrUnformed.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.8f6: %T.as.DefaultOrUnformed.impl.Op.type.bbe = struct_value () [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.868: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.597, @T.as.DefaultOrUnformed.impl(%Z) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.type.121: type = fn_type @T.as.DefaultOrUnformed.impl.Op, @T.as.DefaultOrUnformed.impl(%Z) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.b4b: %T.as.DefaultOrUnformed.impl.Op.type.121 = struct_value () [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value %Z, (%DefaultOrUnformed.impl_witness.868) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.specific_fn: <specific function> = specific_function %T.as.DefaultOrUnformed.impl.Op.b4b, @T.as.DefaultOrUnformed.impl.Op(%Z) [concrete]
// CHECK:STDOUT:   %.7ae: type = fn_type_with_self_type %A.WithSelf.F.type.d17, %A.facet.027 [concrete]
// CHECK:STDOUT:   %ptr.1c4: type = ptr_type %Z [concrete]
// CHECK:STDOUT:   %pattern_type.8b9: type = pattern_type %ptr.1c4 [concrete]
// CHECK:STDOUT:   %tuple.61d: %tuple.type.ff9 = tuple_value (%X, %Y, %ptr.1c4) [concrete]
// CHECK:STDOUT:   %tuple.type.cf4: type = tuple_type (%X, %Y, %ptr.1c4) [concrete]
// CHECK:STDOUT:   %.d9d: Core.Form = init_form %tuple.type.cf4 [concrete]
// CHECK:STDOUT:   %pattern_type.213: type = pattern_type %tuple.type.cf4 [concrete]
// CHECK:STDOUT:   %Y.as.A.impl.F.specific_fn: <specific function> = specific_function %Y.as.A.impl.F, @Y.as.A.impl.F(%Z) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.3: type = fn_type @Destroy.Op.loc23_22.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.3: %Destroy.Op.type.790b3b.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.e20f1f.3: <witness> = custom_witness (%Destroy.Op.2dd605.3), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.5: type = fn_type @Destroy.Op.loc23_22.5 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.5: %Destroy.Op.type.790b3b.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.e20f1f.5: <witness> = custom_witness (%Destroy.Op.2dd605.5), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.062: %Destroy.type = facet_value %tuple.type.cf4, (%custom_witness.e20f1f.5) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.d44: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.062) [concrete]
// CHECK:STDOUT:   %.95e: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.d44, %Destroy.facet.062 [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.6: type = fn_type @Destroy.Op.loc22 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.6: %Destroy.Op.type.790b3b.6 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndWith.type.fd8: type = generic_interface_type @BitAndWith [concrete]
// CHECK:STDOUT:   %BitAndWith.generic: %BitAndWith.type.fd8 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndWith.type.2af: type = facet_type <@BitAndWith, @BitAndWith(type)> [concrete]
// CHECK:STDOUT:   %BitAndWith.impl_witness: <witness> = impl_witness imports.%BitAndWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %BitAndWith.facet: %BitAndWith.type.2af = facet_value type, (%BitAndWith.impl_witness) [concrete]
// CHECK:STDOUT:   %BitAndWith.WithSelf.Op.type.5d1: type = fn_type @BitAndWith.WithSelf.Op, @BitAndWith.WithSelf(type, %BitAndWith.facet) [concrete]
// CHECK:STDOUT:   %.9ee: type = fn_type_with_self_type %BitAndWith.WithSelf.Op.type.5d1, %BitAndWith.facet [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.type: type = fn_type @type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op: %type.as.BitAndWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.bound: <bound method> = bound_method %A.type.ef5, %type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@A, @A(%X) & @Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.e1b: type = pattern_type %facet_type [concrete]
// CHECK:STDOUT:   %T.0c6: %facet_type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %CallGeneric.type: type = fn_type @CallGeneric [concrete]
// CHECK:STDOUT:   %CallGeneric: %CallGeneric.type = struct_value () [concrete]
// CHECK:STDOUT:   %A.WithSelf.F.type.5f2: type = fn_type @A.WithSelf.F, @A.WithSelf(%X, %T.0c6) [symbolic]
// CHECK:STDOUT:   %A.WithSelf.F.923: %A.WithSelf.F.type.5f2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.586: type = facet_access_type %T.0c6 [symbolic]
// CHECK:STDOUT:   %A.lookup_impl_witness: <witness> = lookup_impl_witness %T.0c6, @A, @A(%X) [symbolic]
// CHECK:STDOUT:   %A.facet.210: %A.type.ef5 = facet_value %T.as_type.586, (%A.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %A.WithSelf.F.type.b88: type = fn_type @A.WithSelf.F, @A.WithSelf(%X, %A.facet.210) [symbolic]
// CHECK:STDOUT:   %A.WithSelf.F.702: %A.WithSelf.F.type.b88 = struct_value () [symbolic]
// CHECK:STDOUT:   %.b59: type = fn_type_with_self_type %A.WithSelf.F.type.b88, %A.facet.210 [symbolic]
// CHECK:STDOUT:   %impl.elem0.65b: %.b59 = impl_witness_access %A.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %tuple.e0a: %tuple.type.d63 = tuple_value (%X, %A.facet.210, %ptr.1c4) [symbolic]
// CHECK:STDOUT:   %tuple.type.0d8: type = tuple_type (%X, %T.as_type.586, %ptr.1c4) [symbolic]
// CHECK:STDOUT:   %.cfc: Core.Form = init_form %tuple.type.0d8 [symbolic]
// CHECK:STDOUT:   %pattern_type.b37: type = pattern_type %tuple.type.0d8 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.846: <specific function> = specific_impl_function %impl.elem0.65b, @A.WithSelf.F(%X, %A.facet.210, %Z) [symbolic]
// CHECK:STDOUT:   %require_complete.04f: <witness> = require_complete_type %tuple.type.0d8 [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.57c: <witness> = lookup_impl_witness %tuple.type.0d8, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.096: %Destroy.type = facet_value %tuple.type.0d8, (%Destroy.lookup_impl_witness.57c) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.9ca: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.096) [symbolic]
// CHECK:STDOUT:   %.83e: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.9ca, %Destroy.facet.096 [symbolic]
// CHECK:STDOUT:   %impl.elem0.c24: %.83e = impl_witness_access %Destroy.lookup_impl_witness.57c, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.da7: <specific function> = specific_impl_function %impl.elem0.c24, @Destroy.WithSelf.Op(%Destroy.facet.096) [symbolic]
// CHECK:STDOUT:   %CallIndirect.type: type = fn_type @CallIndirect [concrete]
// CHECK:STDOUT:   %CallIndirect: %CallIndirect.type = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value: %facet_type = facet_value %Y, (%A.impl_witness, %custom_witness.e20f1f.3) [concrete]
// CHECK:STDOUT:   %CallGeneric.specific_fn: <specific function> = specific_function %CallGeneric, @CallGeneric(%facet_value) [concrete]
// CHECK:STDOUT:   %complete_type.6c5: <witness> = complete_type_witness %ptr.1c4 [concrete]
// CHECK:STDOUT:   %complete_type.2ad: <witness> = complete_type_witness %tuple.type.cf4 [concrete]
// CHECK:STDOUT:   %tuple.type.738: type = tuple_type (%empty_struct_type, %empty_struct_type, %ptr.1c4) [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.803: <witness> = impl_witness imports.%Copy.impl_witness_table.6b9, @ptr.as.Copy.impl(%Z) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.6ee: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%Z) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.f43: %ptr.as.Copy.impl.Op.type.6ee = struct_value () [concrete]
// CHECK:STDOUT:   %.d55: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%Z) [concrete]
// CHECK:STDOUT:   %Copy.facet.777: %Copy.type = facet_value %ptr.1c4, (%Copy.impl_witness.803) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.c51: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.777) [concrete]
// CHECK:STDOUT:   %.8ee: type = fn_type_with_self_type %Copy.WithSelf.Op.type.c51, %Copy.facet.777 [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.f43, @ptr.as.Copy.impl.Op(%Z) [concrete]
// CHECK:STDOUT:   %tuple.6ea: %tuple.type.d63 = tuple_value (%X, %A.facet.027, %ptr.1c4) [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.aa4: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.aa424f.1) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.5a1c66.1)]
// CHECK:STDOUT:   %Copy.impl_witness_table.6b9 = impl_witness_table (%Core.import_ref.aa4), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.DefaultOrUnformed: type = import_ref Core//prelude/parts/default, DefaultOrUnformed, loaded [concrete = constants.%DefaultOrUnformed.type]
// CHECK:STDOUT:   %Core.import_ref.1d0: @T.as.DefaultOrUnformed.impl.%T.as.DefaultOrUnformed.impl.Op.type (%T.as.DefaultOrUnformed.impl.Op.type.bbe) = import_ref Core//prelude/parts/default, loc{{\d+_\d+}}, loaded [symbolic = @T.as.DefaultOrUnformed.impl.%T.as.DefaultOrUnformed.impl.Op (constants.%T.as.DefaultOrUnformed.impl.Op.8f6)]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness_table.597 = impl_witness_table (%Core.import_ref.1d0), @T.as.DefaultOrUnformed.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.BitAndWith: %BitAndWith.type.fd8 = import_ref Core//prelude/parts/as, BitAndWith, loaded [concrete = constants.%BitAndWith.generic]
// CHECK:STDOUT:   %Core.import_ref.f63: %type.as.BitAndWith.impl.Op.type = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, loaded [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:   %BitAndWith.impl_witness_table = impl_witness_table (%Core.import_ref.f63), @type.as.BitAndWith.impl [concrete]
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
// CHECK:STDOUT:   %A.decl: %A.type.0af = interface_decl @A [concrete = constants.%A.generic] {
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
// CHECK:STDOUT:     %A.ref: %A.type.0af = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %A.type: type = facet_type <@A, @A(constants.%X)> [concrete = constants.%A.type.ef5]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Call.decl: %Call.type = fn_decl @Call [concrete = constants.%Call] {} {}
// CHECK:STDOUT:   %CallGeneric.decl: %CallGeneric.type = fn_decl @CallGeneric [concrete = constants.%CallGeneric] {
// CHECK:STDOUT:     %T.patt: %pattern_type.e1b = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc26_25.1: type = splice_block %.loc26_25.3 [concrete = constants.%facet_type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %A.ref: %A.type.0af = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:       %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:       %A.type: type = facet_type <@A, @A(constants.%X)> [concrete = constants.%A.type.ef5]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:       %impl.elem0.loc26: %.9ee = impl_witness_access constants.%BitAndWith.impl_witness, element0 [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:       %bound_method.loc26: <bound method> = bound_method %A.type, %impl.elem0.loc26 [concrete = constants.%type.as.BitAndWith.impl.Op.bound]
// CHECK:STDOUT:       %type.as.BitAndWith.impl.Op.call: init type = call %bound_method.loc26(%A.type, %Destroy.ref) [concrete = constants.%facet_type]
// CHECK:STDOUT:       %.loc26_25.2: type = value_of_initializer %type.as.BitAndWith.impl.Op.call [concrete = constants.%facet_type]
// CHECK:STDOUT:       %.loc26_25.3: type = converted %type.as.BitAndWith.impl.Op.call, %.loc26_25.2 [concrete = constants.%facet_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc26_17.2: %facet_type = symbolic_binding T, 0 [symbolic = %T.loc26_17.1 (constants.%T.0c6)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallIndirect.decl: %CallIndirect.type = fn_decl @CallIndirect [concrete = constants.%CallIndirect] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @A(%T.loc5_14.2: type) {
// CHECK:STDOUT:   %T.loc5_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc5_14.1 (constants.%T.67d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.type: type = facet_type <@A, @A(%T.loc5_14.1)> [symbolic = %A.type (constants.%A.type.5ea)]
// CHECK:STDOUT:   %Self.loc5_23.2: @A.%A.type (%A.type.5ea) = symbolic_binding Self, 1 [symbolic = %Self.loc5_23.2 (constants.%Self.a33)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc5_23.1: @A.%A.type (%A.type.5ea) = symbolic_binding Self, 1 [symbolic = %Self.loc5_23.2 (constants.%Self.a33)]
// CHECK:STDOUT:     %A.WithSelf.decl = interface_with_self_decl @A [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %A.WithSelf.F.decl: @A.WithSelf.%A.WithSelf.F.type (%A.WithSelf.F.type.f0d) = fn_decl @A.WithSelf.F [symbolic = @A.WithSelf.%A.WithSelf.F (constants.%A.WithSelf.F.628)] {
// CHECK:STDOUT:       %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 2 [concrete]
// CHECK:STDOUT:       %u.param_patt: @A.WithSelf.F.%pattern_type.loc6_19 (%pattern_type.986) = value_param_pattern [concrete]
// CHECK:STDOUT:       %u.patt: @A.WithSelf.F.%pattern_type.loc6_19 (%pattern_type.986) = at_binding_pattern u, %u.param_patt [concrete]
// CHECK:STDOUT:       %return.param_patt: @A.WithSelf.F.%pattern_type.loc6_40 (%pattern_type.050) = out_param_pattern [concrete]
// CHECK:STDOUT:       %return.patt: @A.WithSelf.F.%pattern_type.loc6_40 (%pattern_type.050) = return_slot_pattern %return.param_patt, %.loc6_40.4 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @A.%T.loc5_14.2 [symbolic = %T (constants.%T.67d)]
// CHECK:STDOUT:       %.loc6_32: @A.WithSelf.F.%A.type (%A.type.5ea) = specific_constant @A.%Self.loc5_23.1, @A(constants.%T.67d) [symbolic = %Self (constants.%Self.a33)]
// CHECK:STDOUT:       %Self.ref: @A.WithSelf.F.%A.type (%A.type.5ea) = name_ref Self, %.loc6_32 [symbolic = %Self (constants.%Self.a33)]
// CHECK:STDOUT:       %U.ref.loc6_38: type = name_ref U, %U.loc6_9.2 [symbolic = %U.loc6_9.1 (constants.%U.4e7)]
// CHECK:STDOUT:       %ptr.loc6_39: type = ptr_type %U.ref.loc6_38 [symbolic = %ptr.loc6_22.1 (constants.%ptr.8f6)]
// CHECK:STDOUT:       %.loc6_40.2: @A.WithSelf.F.%tuple.type.loc6_40.1 (%tuple.type.54d) = tuple_literal (%T.ref, %Self.ref, %ptr.loc6_39) [symbolic = %tuple (constants.%tuple.44b)]
// CHECK:STDOUT:       %Self.as_type.loc6_40.2: type = facet_access_type constants.%Self.a33 [symbolic = %Self.as_type.loc6_40.1 (constants.%Self.as_type.0a5)]
// CHECK:STDOUT:       %.loc6_40.3: type = converted constants.%Self.a33, %Self.as_type.loc6_40.2 [symbolic = %Self.as_type.loc6_40.1 (constants.%Self.as_type.0a5)]
// CHECK:STDOUT:       %.loc6_40.4: type = converted %.loc6_40.2, constants.%tuple.type.67c [symbolic = %tuple.type.loc6_40.2 (constants.%tuple.type.67c)]
// CHECK:STDOUT:       %.loc6_40.5: Core.Form = init_form %.loc6_40.4 [symbolic = %.loc6_40.1 (constants.%.f77)]
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
// CHECK:STDOUT:       %return.param: ref @A.WithSelf.F.%tuple.type.loc6_40.2 (%tuple.type.67c) = out_param call_param1
// CHECK:STDOUT:       %return: ref @A.WithSelf.F.%tuple.type.loc6_40.2 (%tuple.type.67c) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc6_41.1: @A.WithSelf.%A.assoc_type (%A.assoc_type.092) = assoc_entity element0, %A.WithSelf.F.decl [symbolic = %assoc0.loc6_41.2 (constants.%assoc0.7bb)]
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
// CHECK:STDOUT:     %return.param_patt: @Y.as.A.impl.F.%pattern_type.loc16_37 (%pattern_type.145) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @Y.as.A.impl.F.%pattern_type.loc16_37 (%pattern_type.145) = return_slot_pattern %return.param_patt, %.loc16_37.3 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %Y.ref: type = name_ref Y, file.%Y.decl [concrete = constants.%Y]
// CHECK:STDOUT:     %U.ref.loc16_35: type = name_ref U, %U.loc16_9.2 [symbolic = %U.loc16_9.1 (constants.%U.67d)]
// CHECK:STDOUT:     %ptr.loc16_36: type = ptr_type %U.ref.loc16_35 [symbolic = %ptr.loc16_22.1 (constants.%ptr.e8f8f9.1)]
// CHECK:STDOUT:     %.loc16_37.2: %tuple.type.ff9 = tuple_literal (%X.ref, %Y.ref, %ptr.loc16_36) [symbolic = %tuple (constants.%tuple.fdd)]
// CHECK:STDOUT:     %.loc16_37.3: type = converted %.loc16_37.2, constants.%tuple.type.4b0 [symbolic = %tuple.type.loc16 (constants.%tuple.type.4b0)]
// CHECK:STDOUT:     %.loc16_37.4: Core.Form = init_form %.loc16_37.3 [symbolic = %.loc16_37.1 (constants.%.de3)]
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
// CHECK:STDOUT:     %return.param: ref @Y.as.A.impl.F.%tuple.type.loc16 (%tuple.type.4b0) = out_param call_param1
// CHECK:STDOUT:     %return: ref @Y.as.A.impl.F.%tuple.type.loc16 (%tuple.type.4b0) = return_slot %return.param
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
// CHECK:STDOUT: generic fn @A.WithSelf.F(@A.%T.loc5_14.2: type, @A.%Self.loc5_23.1: @A.%A.type (%A.type.5ea), %U.loc6_9.2: type) {
// CHECK:STDOUT:   %U.loc6_9.1: type = symbolic_binding U, 2 [symbolic = %U.loc6_9.1 (constants.%U.4e7)]
// CHECK:STDOUT:   %ptr.loc6_22.1: type = ptr_type %U.loc6_9.1 [symbolic = %ptr.loc6_22.1 (constants.%ptr.8f6)]
// CHECK:STDOUT:   %pattern_type.loc6_19: type = pattern_type %ptr.loc6_22.1 [symbolic = %pattern_type.loc6_19 (constants.%pattern_type.986)]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T.67d)]
// CHECK:STDOUT:   %A.type: type = facet_type <@A, @A(%T)> [symbolic = %A.type (constants.%A.type.5ea)]
// CHECK:STDOUT:   %Self: @A.WithSelf.F.%A.type (%A.type.5ea) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.a33)]
// CHECK:STDOUT:   %tuple.type.loc6_40.1: type = tuple_type (type, %A.type, type) [symbolic = %tuple.type.loc6_40.1 (constants.%tuple.type.54d)]
// CHECK:STDOUT:   %tuple: @A.WithSelf.F.%tuple.type.loc6_40.1 (%tuple.type.54d) = tuple_value (%T, %Self, %ptr.loc6_22.1) [symbolic = %tuple (constants.%tuple.44b)]
// CHECK:STDOUT:   %Self.as_type.loc6_40.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc6_40.1 (constants.%Self.as_type.0a5)]
// CHECK:STDOUT:   %tuple.type.loc6_40.2: type = tuple_type (%T, %Self.as_type.loc6_40.1, %ptr.loc6_22.1) [symbolic = %tuple.type.loc6_40.2 (constants.%tuple.type.67c)]
// CHECK:STDOUT:   %.loc6_40.1: Core.Form = init_form %tuple.type.loc6_40.2 [symbolic = %.loc6_40.1 (constants.%.f77)]
// CHECK:STDOUT:   %pattern_type.loc6_40: type = pattern_type %tuple.type.loc6_40.2 [symbolic = %pattern_type.loc6_40 (constants.%pattern_type.050)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param: @A.WithSelf.F.%ptr.loc6_22.1 (%ptr.8f6)) -> out %return.param: @A.WithSelf.F.%tuple.type.loc6_40.2 (%tuple.type.67c);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Y.as.A.impl.F(%U.loc16_9.2: type) {
// CHECK:STDOUT:   %U.loc16_9.1: type = symbolic_binding U, 0 [symbolic = %U.loc16_9.1 (constants.%U.67d)]
// CHECK:STDOUT:   %ptr.loc16_22.1: type = ptr_type %U.loc16_9.1 [symbolic = %ptr.loc16_22.1 (constants.%ptr.e8f8f9.1)]
// CHECK:STDOUT:   %pattern_type.loc16_19: type = pattern_type %ptr.loc16_22.1 [symbolic = %pattern_type.loc16_19 (constants.%pattern_type.4f4b84.1)]
// CHECK:STDOUT:   %tuple: %tuple.type.ff9 = tuple_value (constants.%X, constants.%Y, %ptr.loc16_22.1) [symbolic = %tuple (constants.%tuple.fdd)]
// CHECK:STDOUT:   %tuple.type.loc16: type = tuple_type (constants.%X, constants.%Y, %ptr.loc16_22.1) [symbolic = %tuple.type.loc16 (constants.%tuple.type.4b0)]
// CHECK:STDOUT:   %.loc16_37.1: Core.Form = init_form %tuple.type.loc16 [symbolic = %.loc16_37.1 (constants.%.de3)]
// CHECK:STDOUT:   %pattern_type.loc16_37: type = pattern_type %tuple.type.loc16 [symbolic = %pattern_type.loc16_37 (constants.%pattern_type.145)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc16_19: <witness> = require_complete_type %ptr.loc16_22.1 [symbolic = %require_complete.loc16_19 (constants.%require_complete.ef162c.1)]
// CHECK:STDOUT:   %require_complete.loc16_25: <witness> = require_complete_type %tuple.type.loc16 [symbolic = %require_complete.loc16_25 (constants.%require_complete.4f8)]
// CHECK:STDOUT:   %tuple.type.loc17: type = tuple_type (constants.%empty_struct_type, constants.%empty_struct_type, %ptr.loc16_22.1) [symbolic = %tuple.type.loc17 (constants.%tuple.type.dd2)]
// CHECK:STDOUT:   %.loc17_21.1: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%U.loc16_9.1) [symbolic = %.loc17_21.1 (constants.%.b6a)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr.loc16_22.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.2bf)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.loc16_22.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet.44d)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.72b)]
// CHECK:STDOUT:   %.loc17_21.2: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %Copy.facet [symbolic = %.loc17_21.2 (constants.%.39b)]
// CHECK:STDOUT:   %impl.elem0.loc17_21.2: @Y.as.A.impl.F.%.loc17_21.2 (%.39b) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc17_21.2 (constants.%impl.elem0.8a3)]
// CHECK:STDOUT:   %specific_impl_fn.loc17_21.2: <specific function> = specific_impl_function %impl.elem0.loc17_21.2, @Copy.WithSelf.Op(%Copy.facet) [symbolic = %specific_impl_fn.loc17_21.2 (constants.%specific_impl_fn.a13)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param: @Y.as.A.impl.F.%ptr.loc16_22.1 (%ptr.e8f8f9.1)) -> out %return.param: @Y.as.A.impl.F.%tuple.type.loc16 (%tuple.type.4b0) {
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
// CHECK:STDOUT:     %impl.elem0.loc17_21.1: @Y.as.A.impl.F.%.loc17_21.2 (%.39b) = impl_witness_access constants.%Copy.lookup_impl_witness.2bf, element0 [symbolic = %impl.elem0.loc17_21.2 (constants.%impl.elem0.8a3)]
// CHECK:STDOUT:     %bound_method.loc17_21.1: <bound method> = bound_method %u.ref, %impl.elem0.loc17_21.1
// CHECK:STDOUT:     %specific_impl_fn.loc17_21.1: <specific function> = specific_impl_function %impl.elem0.loc17_21.1, @Copy.WithSelf.Op(constants.%Copy.facet.44d) [symbolic = %specific_impl_fn.loc17_21.2 (constants.%specific_impl_fn.a13)]
// CHECK:STDOUT:     %bound_method.loc17_21.2: <bound method> = bound_method %u.ref, %specific_impl_fn.loc17_21.1
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @Y.as.A.impl.F.%ptr.loc16_22.1 (%ptr.e8f8f9.1) = call %bound_method.loc17_21.2(%u.ref)
// CHECK:STDOUT:     %tuple.elem2: ref @Y.as.A.impl.F.%ptr.loc16_22.1 (%ptr.e8f8f9.1) = tuple_access %return.param, element2
// CHECK:STDOUT:     %.loc17_22.4: init @Y.as.A.impl.F.%ptr.loc16_22.1 (%ptr.e8f8f9.1) to %tuple.elem2 = in_place_init %Copy.WithSelf.Op.call
// CHECK:STDOUT:     %.loc17_22.5: init @Y.as.A.impl.F.%tuple.type.loc16 (%tuple.type.4b0) to %return.param = tuple_init (%.loc17_22.2, %.loc17_22.3, %.loc17_22.4)
// CHECK:STDOUT:     %.loc17_23: init @Y.as.A.impl.F.%tuple.type.loc16 (%tuple.type.4b0) = converted %.loc17_22.1, %.loc17_22.5
// CHECK:STDOUT:     return %.loc17_23 to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %u.patt: %pattern_type.f0a = ref_binding_pattern u [concrete]
// CHECK:STDOUT:     %u.var_patt: %pattern_type.f0a = var_pattern %u.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %u.var: ref %Z = var %u.var_patt
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value constants.%Z, (constants.%DefaultOrUnformed.impl_witness.868) [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %.loc22_11.1: %DefaultOrUnformed.type = converted constants.%Z, %DefaultOrUnformed.facet [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %as_type.loc22: type = facet_access_type %.loc22_11.1 [concrete = constants.%Z]
// CHECK:STDOUT:   %.loc22_11.2: type = converted %.loc22_11.1, %as_type.loc22 [concrete = constants.%Z]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.specific_fn: <specific function> = specific_function constants.%T.as.DefaultOrUnformed.impl.Op.b4b, @T.as.DefaultOrUnformed.impl.Op(constants.%Z) [concrete = constants.%T.as.DefaultOrUnformed.impl.Op.specific_fn]
// CHECK:STDOUT:   %.loc22_3: ref %Z = splice_block %u.var {}
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.call: init %Z to %.loc22_3 = call %T.as.DefaultOrUnformed.impl.Op.specific_fn()
// CHECK:STDOUT:   assign %u.var, %T.as.DefaultOrUnformed.impl.Op.call
// CHECK:STDOUT:   %Z.ref.loc22: type = name_ref Z, file.%Z.decl [concrete = constants.%Z]
// CHECK:STDOUT:   %u: ref %Z = ref_binding u, %u.var
// CHECK:STDOUT:   %Y.ref: type = name_ref Y, file.%Y.decl [concrete = constants.%Y]
// CHECK:STDOUT:   %A.ref: %A.type.0af = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:   %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %A.type: type = facet_type <@A, @A(constants.%X)> [concrete = constants.%A.type.ef5]
// CHECK:STDOUT:   %A.facet: %A.type.ef5 = facet_value %Y.ref, (constants.%A.impl_witness) [concrete = constants.%A.facet.027]
// CHECK:STDOUT:   %.loc23_6: %A.type.ef5 = converted %Y.ref, %A.facet [concrete = constants.%A.facet.027]
// CHECK:STDOUT:   %as_type.loc23: type = facet_access_type %.loc23_6 [concrete = constants.%Y]
// CHECK:STDOUT:   %.loc23_14.1: type = converted %.loc23_6, %as_type.loc23 [concrete = constants.%Y]
// CHECK:STDOUT:   %.loc23_14.2: %A.assoc_type.3ce = specific_constant @A.WithSelf.%assoc0.loc6_41.1, @A.WithSelf(constants.%X, constants.%A.facet.027) [concrete = constants.%assoc0.2c2]
// CHECK:STDOUT:   %F.ref: %A.assoc_type.3ce = name_ref F, %.loc23_14.2 [concrete = constants.%assoc0.2c2]
// CHECK:STDOUT:   %impl.elem0: %.7ae = impl_witness_access constants.%A.impl_witness, element0 [concrete = constants.%Y.as.A.impl.F]
// CHECK:STDOUT:   %Z.ref.loc23: type = name_ref Z, file.%Z.decl [concrete = constants.%Z]
// CHECK:STDOUT:   %u.ref: ref %Z = name_ref u, %u
// CHECK:STDOUT:   %addr: %ptr.1c4 = addr_of %u.ref
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Y.as.A.impl.F(constants.%Z) [concrete = constants.%Y.as.A.impl.F.specific_fn]
// CHECK:STDOUT:   %.loc23_22.1: ref %tuple.type.cf4 = temporary_storage
// CHECK:STDOUT:   %Y.as.A.impl.F.call: init %tuple.type.cf4 to %.loc23_22.1 = call %specific_fn(%addr)
// CHECK:STDOUT:   %.loc23_22.2: ref %tuple.type.cf4 = temporary %.loc23_22.1, %Y.as.A.impl.F.call
// CHECK:STDOUT:   %Destroy.Op.bound.loc23: <bound method> = bound_method %.loc23_22.2, constants.%Destroy.Op.2dd605.5
// CHECK:STDOUT:   %Destroy.Op.call.loc23: init %empty_tuple.type = call %Destroy.Op.bound.loc23(%.loc23_22.2)
// CHECK:STDOUT:   %Destroy.Op.bound.loc22: <bound method> = bound_method %u.var, constants.%Destroy.Op.2dd605.6
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
// CHECK:STDOUT: fn @Destroy.Op.loc23_22.4(%self.param: ref %ptr.1c4) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc23_22.5(%self.param: ref %tuple.type.cf4) {
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
// CHECK:STDOUT:   %T.loc26_17.1: %facet_type = symbolic_binding T, 0 [symbolic = %T.loc26_17.1 (constants.%T.0c6)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc28_4.2: type = facet_access_type %T.loc26_17.1 [symbolic = %T.as_type.loc28_4.2 (constants.%T.as_type.586)]
// CHECK:STDOUT:   %A.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc26_17.1, @A, @A(constants.%X) [symbolic = %A.lookup_impl_witness (constants.%A.lookup_impl_witness)]
// CHECK:STDOUT:   %A.facet: %A.type.ef5 = facet_value %T.as_type.loc28_4.2, (%A.lookup_impl_witness) [symbolic = %A.facet (constants.%A.facet.210)]
// CHECK:STDOUT:   %A.WithSelf.F.type: type = fn_type @A.WithSelf.F, @A.WithSelf(constants.%X, %A.facet) [symbolic = %A.WithSelf.F.type (constants.%A.WithSelf.F.type.b88)]
// CHECK:STDOUT:   %.loc28_4.3: type = fn_type_with_self_type %A.WithSelf.F.type, %A.facet [symbolic = %.loc28_4.3 (constants.%.b59)]
// CHECK:STDOUT:   %impl.elem0.loc28_4.2: @CallGeneric.%.loc28_4.3 (%.b59) = impl_witness_access %A.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc28_4.2 (constants.%impl.elem0.65b)]
// CHECK:STDOUT:   %specific_impl_fn.loc28_4.2: <specific function> = specific_impl_function %impl.elem0.loc28_4.2, @A.WithSelf.F(constants.%X, %A.facet, constants.%Z) [symbolic = %specific_impl_fn.loc28_4.2 (constants.%specific_impl_fn.846)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (constants.%X, %T.as_type.loc28_4.2, constants.%ptr.1c4) [symbolic = %tuple.type (constants.%tuple.type.0d8)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %tuple.type [symbolic = %require_complete (constants.%require_complete.04f)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %tuple.type, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.57c)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %tuple.type, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet (constants.%Destroy.facet.096)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.9ca)]
// CHECK:STDOUT:   %.loc28_12.3: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Destroy.facet [symbolic = %.loc28_12.3 (constants.%.83e)]
// CHECK:STDOUT:   %impl.elem0.loc28_12.2: @CallGeneric.%.loc28_12.3 (%.83e) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc28_12.2 (constants.%impl.elem0.c24)]
// CHECK:STDOUT:   %specific_impl_fn.loc28_12.2: <specific function> = specific_impl_function %impl.elem0.loc28_12.2, @Destroy.WithSelf.Op(%Destroy.facet) [symbolic = %specific_impl_fn.loc28_12.2 (constants.%specific_impl_fn.da7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %u.patt: %pattern_type.f0a = ref_binding_pattern u [concrete]
// CHECK:STDOUT:       %u.var_patt: %pattern_type.f0a = var_pattern %u.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u.var: ref %Z = var %u.var_patt
// CHECK:STDOUT:     %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value constants.%Z, (constants.%DefaultOrUnformed.impl_witness.868) [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:     %.loc27_11.1: %DefaultOrUnformed.type = converted constants.%Z, %DefaultOrUnformed.facet [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:     %as_type: type = facet_access_type %.loc27_11.1 [concrete = constants.%Z]
// CHECK:STDOUT:     %.loc27_11.2: type = converted %.loc27_11.1, %as_type [concrete = constants.%Z]
// CHECK:STDOUT:     %T.as.DefaultOrUnformed.impl.Op.specific_fn: <specific function> = specific_function constants.%T.as.DefaultOrUnformed.impl.Op.b4b, @T.as.DefaultOrUnformed.impl.Op(constants.%Z) [concrete = constants.%T.as.DefaultOrUnformed.impl.Op.specific_fn]
// CHECK:STDOUT:     %.loc27_3: ref %Z = splice_block %u.var {}
// CHECK:STDOUT:     %T.as.DefaultOrUnformed.impl.Op.call: init %Z to %.loc27_3 = call %T.as.DefaultOrUnformed.impl.Op.specific_fn()
// CHECK:STDOUT:     assign %u.var, %T.as.DefaultOrUnformed.impl.Op.call
// CHECK:STDOUT:     %Z.ref.loc27: type = name_ref Z, file.%Z.decl [concrete = constants.%Z]
// CHECK:STDOUT:     %u: ref %Z = ref_binding u, %u.var
// CHECK:STDOUT:     %T.ref: %facet_type = name_ref T, %T.loc26_17.2 [symbolic = %T.loc26_17.1 (constants.%T.0c6)]
// CHECK:STDOUT:     %T.as_type.loc28_4.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc28_4.2 (constants.%T.as_type.586)]
// CHECK:STDOUT:     %.loc28_4.1: type = converted %T.ref, %T.as_type.loc28_4.1 [symbolic = %T.as_type.loc28_4.2 (constants.%T.as_type.586)]
// CHECK:STDOUT:     %.loc28_4.2: %A.assoc_type.3ce = specific_constant @A.WithSelf.%assoc0.loc6_41.1, @A.WithSelf(constants.%X, constants.%T.0c6) [concrete = constants.%assoc0.2c2]
// CHECK:STDOUT:     %F.ref: %A.assoc_type.3ce = name_ref F, %.loc28_4.2 [concrete = constants.%assoc0.2c2]
// CHECK:STDOUT:     %impl.elem0.loc28_4.1: @CallGeneric.%.loc28_4.3 (%.b59) = impl_witness_access constants.%A.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc28_4.2 (constants.%impl.elem0.65b)]
// CHECK:STDOUT:     %Z.ref.loc28: type = name_ref Z, file.%Z.decl [concrete = constants.%Z]
// CHECK:STDOUT:     %u.ref: ref %Z = name_ref u, %u
// CHECK:STDOUT:     %addr: %ptr.1c4 = addr_of %u.ref
// CHECK:STDOUT:     %specific_impl_fn.loc28_4.1: <specific function> = specific_impl_function %impl.elem0.loc28_4.1, @A.WithSelf.F(constants.%X, constants.%A.facet.210, constants.%Z) [symbolic = %specific_impl_fn.loc28_4.2 (constants.%specific_impl_fn.846)]
// CHECK:STDOUT:     %.loc28_12.1: ref @CallGeneric.%tuple.type (%tuple.type.0d8) = temporary_storage
// CHECK:STDOUT:     %A.WithSelf.F.call: init @CallGeneric.%tuple.type (%tuple.type.0d8) to %.loc28_12.1 = call %specific_impl_fn.loc28_4.1(%addr)
// CHECK:STDOUT:     %.loc28_12.2: ref @CallGeneric.%tuple.type (%tuple.type.0d8) = temporary %.loc28_12.1, %A.WithSelf.F.call
// CHECK:STDOUT:     %impl.elem0.loc28_12.1: @CallGeneric.%.loc28_12.3 (%.83e) = impl_witness_access constants.%Destroy.lookup_impl_witness.57c, element0 [symbolic = %impl.elem0.loc28_12.2 (constants.%impl.elem0.c24)]
// CHECK:STDOUT:     %bound_method.loc28_12.1: <bound method> = bound_method %.loc28_12.2, %impl.elem0.loc28_12.1
// CHECK:STDOUT:     %specific_impl_fn.loc28_12.1: <specific function> = specific_impl_function %impl.elem0.loc28_12.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.096) [symbolic = %specific_impl_fn.loc28_12.2 (constants.%specific_impl_fn.da7)]
// CHECK:STDOUT:     %bound_method.loc28_12.2: <bound method> = bound_method %.loc28_12.2, %specific_impl_fn.loc28_12.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc28_12.2(%.loc28_12.2)
// CHECK:STDOUT:     %Destroy.Op.bound: <bound method> = bound_method %u.var, constants.%Destroy.Op.2dd605.6
// CHECK:STDOUT:     %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%u.var)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallIndirect() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CallGeneric.ref: %CallGeneric.type = name_ref CallGeneric, file.%CallGeneric.decl [concrete = constants.%CallGeneric]
// CHECK:STDOUT:   %Y.ref: type = name_ref Y, file.%Y.decl [concrete = constants.%Y]
// CHECK:STDOUT:   %facet_value: %facet_type = facet_value %Y.ref, (constants.%A.impl_witness, constants.%custom_witness.e20f1f.3) [concrete = constants.%facet_value]
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
// CHECK:STDOUT: specific @A.WithSelf(constants.%T.67d, constants.%Self.a33) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf.F(constants.%T.67d, constants.%Self.a33, constants.%U.4e7) {
// CHECK:STDOUT:   %U.loc6_9.1 => constants.%U.4e7
// CHECK:STDOUT:   %ptr.loc6_22.1 => constants.%ptr.8f6
// CHECK:STDOUT:   %pattern_type.loc6_19 => constants.%pattern_type.986
// CHECK:STDOUT:   %T => constants.%T.67d
// CHECK:STDOUT:   %A.type => constants.%A.type.5ea
// CHECK:STDOUT:   %Self => constants.%Self.a33
// CHECK:STDOUT:   %tuple.type.loc6_40.1 => constants.%tuple.type.54d
// CHECK:STDOUT:   %tuple => constants.%tuple.44b
// CHECK:STDOUT:   %Self.as_type.loc6_40.1 => constants.%Self.as_type.0a5
// CHECK:STDOUT:   %tuple.type.loc6_40.2 => constants.%tuple.type.67c
// CHECK:STDOUT:   %.loc6_40.1 => constants.%.f77
// CHECK:STDOUT:   %pattern_type.loc6_40 => constants.%pattern_type.050
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%X) {
// CHECK:STDOUT:   %T.loc5_14.1 => constants.%X
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.type => constants.%A.type.ef5
// CHECK:STDOUT:   %Self.loc5_23.2 => constants.%Self.0c0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%X, constants.%Self.a33) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.ef5
// CHECK:STDOUT:   %Self => constants.%Self.a33
// CHECK:STDOUT:   %A.WithSelf.F.type => constants.%A.WithSelf.F.type.c12
// CHECK:STDOUT:   %A.WithSelf.F => constants.%A.WithSelf.F.612
// CHECK:STDOUT:   %A.assoc_type => constants.%A.assoc_type.3ce
// CHECK:STDOUT:   %assoc0.loc6_41.2 => constants.%assoc0.2c2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y.as.A.impl.F(constants.%U.67d) {
// CHECK:STDOUT:   %U.loc16_9.1 => constants.%U.67d
// CHECK:STDOUT:   %ptr.loc16_22.1 => constants.%ptr.e8f8f9.1
// CHECK:STDOUT:   %pattern_type.loc16_19 => constants.%pattern_type.4f4b84.1
// CHECK:STDOUT:   %tuple => constants.%tuple.fdd
// CHECK:STDOUT:   %tuple.type.loc16 => constants.%tuple.type.4b0
// CHECK:STDOUT:   %.loc16_37.1 => constants.%.de3
// CHECK:STDOUT:   %pattern_type.loc16_37 => constants.%pattern_type.145
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%X, constants.%A.facet.027) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.ef5
// CHECK:STDOUT:   %Self => constants.%A.facet.027
// CHECK:STDOUT:   %A.WithSelf.F.type => constants.%A.WithSelf.F.type.d17
// CHECK:STDOUT:   %A.WithSelf.F => constants.%A.WithSelf.F.a05
// CHECK:STDOUT:   %A.assoc_type => constants.%A.assoc_type.3ce
// CHECK:STDOUT:   %assoc0.loc6_41.2 => constants.%assoc0.2c2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf.F(constants.%X, constants.%A.facet.027, constants.%U.67d) {
// CHECK:STDOUT:   %U.loc6_9.1 => constants.%U.67d
// CHECK:STDOUT:   %ptr.loc6_22.1 => constants.%ptr.e8f8f9.1
// CHECK:STDOUT:   %pattern_type.loc6_19 => constants.%pattern_type.4f4b84.1
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.ef5
// CHECK:STDOUT:   %Self => constants.%A.facet.027
// CHECK:STDOUT:   %tuple.type.loc6_40.1 => constants.%tuple.type.d63
// CHECK:STDOUT:   %tuple => constants.%tuple.6cb
// CHECK:STDOUT:   %Self.as_type.loc6_40.1 => constants.%Y
// CHECK:STDOUT:   %tuple.type.loc6_40.2 => constants.%tuple.type.4b0
// CHECK:STDOUT:   %.loc6_40.1 => constants.%.de3
// CHECK:STDOUT:   %pattern_type.loc6_40 => constants.%pattern_type.145
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y.as.A.impl.F(constants.%Z) {
// CHECK:STDOUT:   %U.loc16_9.1 => constants.%Z
// CHECK:STDOUT:   %ptr.loc16_22.1 => constants.%ptr.1c4
// CHECK:STDOUT:   %pattern_type.loc16_19 => constants.%pattern_type.8b9
// CHECK:STDOUT:   %tuple => constants.%tuple.61d
// CHECK:STDOUT:   %tuple.type.loc16 => constants.%tuple.type.cf4
// CHECK:STDOUT:   %.loc16_37.1 => constants.%.d9d
// CHECK:STDOUT:   %pattern_type.loc16_37 => constants.%pattern_type.213
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc16_19 => constants.%complete_type.6c5
// CHECK:STDOUT:   %require_complete.loc16_25 => constants.%complete_type.2ad
// CHECK:STDOUT:   %tuple.type.loc17 => constants.%tuple.type.738
// CHECK:STDOUT:   %.loc17_21.1 => constants.%.d55
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.impl_witness.803
// CHECK:STDOUT:   %Copy.facet => constants.%Copy.facet.777
// CHECK:STDOUT:   %Copy.WithSelf.Op.type => constants.%Copy.WithSelf.Op.type.c51
// CHECK:STDOUT:   %.loc17_21.2 => constants.%.8ee
// CHECK:STDOUT:   %impl.elem0.loc17_21.2 => constants.%ptr.as.Copy.impl.Op.f43
// CHECK:STDOUT:   %specific_impl_fn.loc17_21.2 => constants.%ptr.as.Copy.impl.Op.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGeneric(constants.%T.0c6) {
// CHECK:STDOUT:   %T.loc26_17.1 => constants.%T.0c6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%X, constants.%T.0c6) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.ef5
// CHECK:STDOUT:   %Self => constants.%T.0c6
// CHECK:STDOUT:   %A.WithSelf.F.type => constants.%A.WithSelf.F.type.5f2
// CHECK:STDOUT:   %A.WithSelf.F => constants.%A.WithSelf.F.923
// CHECK:STDOUT:   %A.assoc_type => constants.%A.assoc_type.3ce
// CHECK:STDOUT:   %assoc0.loc6_41.2 => constants.%assoc0.2c2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%X, constants.%A.facet.210) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.ef5
// CHECK:STDOUT:   %Self => constants.%A.facet.210
// CHECK:STDOUT:   %A.WithSelf.F.type => constants.%A.WithSelf.F.type.b88
// CHECK:STDOUT:   %A.WithSelf.F => constants.%A.WithSelf.F.702
// CHECK:STDOUT:   %A.assoc_type => constants.%A.assoc_type.3ce
// CHECK:STDOUT:   %assoc0.loc6_41.2 => constants.%assoc0.2c2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf.F(constants.%X, constants.%A.facet.210, constants.%Z) {
// CHECK:STDOUT:   %U.loc6_9.1 => constants.%Z
// CHECK:STDOUT:   %ptr.loc6_22.1 => constants.%ptr.1c4
// CHECK:STDOUT:   %pattern_type.loc6_19 => constants.%pattern_type.8b9
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.ef5
// CHECK:STDOUT:   %Self => constants.%A.facet.210
// CHECK:STDOUT:   %tuple.type.loc6_40.1 => constants.%tuple.type.d63
// CHECK:STDOUT:   %tuple => constants.%tuple.e0a
// CHECK:STDOUT:   %Self.as_type.loc6_40.1 => constants.%T.as_type.586
// CHECK:STDOUT:   %tuple.type.loc6_40.2 => constants.%tuple.type.0d8
// CHECK:STDOUT:   %.loc6_40.1 => constants.%.cfc
// CHECK:STDOUT:   %pattern_type.loc6_40 => constants.%pattern_type.b37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGeneric(constants.%facet_value) {
// CHECK:STDOUT:   %T.loc26_17.1 => constants.%facet_value
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc28_4.2 => constants.%Y
// CHECK:STDOUT:   %A.lookup_impl_witness => constants.%A.impl_witness
// CHECK:STDOUT:   %A.facet => constants.%A.facet.027
// CHECK:STDOUT:   %A.WithSelf.F.type => constants.%A.WithSelf.F.type.d17
// CHECK:STDOUT:   %.loc28_4.3 => constants.%.7ae
// CHECK:STDOUT:   %impl.elem0.loc28_4.2 => constants.%Y.as.A.impl.F
// CHECK:STDOUT:   %specific_impl_fn.loc28_4.2 => constants.%Y.as.A.impl.F.specific_fn
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.cf4
// CHECK:STDOUT:   %require_complete => constants.%complete_type.2ad
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e20f1f.5
// CHECK:STDOUT:   %Destroy.facet => constants.%Destroy.facet.062
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.d44
// CHECK:STDOUT:   %.loc28_12.3 => constants.%.95e
// CHECK:STDOUT:   %impl.elem0.loc28_12.2 => constants.%Destroy.Op.2dd605.5
// CHECK:STDOUT:   %specific_impl_fn.loc28_12.2 => constants.%Destroy.Op.2dd605.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf.F(constants.%X, constants.%A.facet.027, constants.%Z) {
// CHECK:STDOUT:   %U.loc6_9.1 => constants.%Z
// CHECK:STDOUT:   %ptr.loc6_22.1 => constants.%ptr.1c4
// CHECK:STDOUT:   %pattern_type.loc6_19 => constants.%pattern_type.8b9
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.ef5
// CHECK:STDOUT:   %Self => constants.%A.facet.027
// CHECK:STDOUT:   %tuple.type.loc6_40.1 => constants.%tuple.type.d63
// CHECK:STDOUT:   %tuple => constants.%tuple.6ea
// CHECK:STDOUT:   %Self.as_type.loc6_40.1 => constants.%Y
// CHECK:STDOUT:   %tuple.type.loc6_40.2 => constants.%tuple.type.cf4
// CHECK:STDOUT:   %.loc6_40.1 => constants.%.d9d
// CHECK:STDOUT:   %pattern_type.loc6_40 => constants.%pattern_type.213
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- generic_method_more_params.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %A.type.0af: type = generic_interface_type @A [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %A.generic: %A.type.0af = struct_value () [concrete]
// CHECK:STDOUT:   %A.type.5ea: type = facet_type <@A, @A(%T.67d)> [symbolic]
// CHECK:STDOUT:   %Self.a33: %A.type.5ea = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %U.4e7: type = symbolic_binding U, 2 [symbolic]
// CHECK:STDOUT:   %pattern_type.62e: type = pattern_type %U.4e7 [symbolic]
// CHECK:STDOUT:   %tuple.type.54d: type = tuple_type (type, %A.type.5ea, type) [symbolic]
// CHECK:STDOUT:   %tuple.ca1: %tuple.type.54d = tuple_value (%T.67d, %Self.a33, %U.4e7) [symbolic]
// CHECK:STDOUT:   %Self.as_type.0a5: type = facet_access_type %Self.a33 [symbolic]
// CHECK:STDOUT:   %tuple.type.a60: type = tuple_type (%T.67d, %Self.as_type.0a5, %U.4e7) [symbolic]
// CHECK:STDOUT:   %.ddd: Core.Form = init_form %tuple.type.a60 [symbolic]
// CHECK:STDOUT:   %pattern_type.41c: type = pattern_type %tuple.type.a60 [symbolic]
// CHECK:STDOUT:   %A.WithSelf.F.type.f0d: type = fn_type @A.WithSelf.F, @A.WithSelf(%T.67d, %Self.a33) [symbolic]
// CHECK:STDOUT:   %A.WithSelf.F.628: %A.WithSelf.F.type.f0d = struct_value () [symbolic]
// CHECK:STDOUT:   %A.assoc_type.092: type = assoc_entity_type @A, @A(%T.67d) [symbolic]
// CHECK:STDOUT:   %assoc0.7bb: %A.assoc_type.092 = assoc_entity element0, @A.WithSelf.%A.WithSelf.F.decl [symbolic]
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
// CHECK:STDOUT:   %A.type.e49: type = facet_type <@A, @A(%W)> [symbolic]
// CHECK:STDOUT:   %A.impl_witness.6c9: <witness> = impl_witness @tuple.type.as.A.impl.%A.impl_witness_table, @tuple.type.as.A.impl(%V1, %V2, %W) [symbolic]
// CHECK:STDOUT:   %Self.2ba: %A.type.e49 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %A.WithSelf.F.type.487: type = fn_type @A.WithSelf.F, @A.WithSelf(%W, %Self.a33) [symbolic]
// CHECK:STDOUT:   %A.WithSelf.F.e2a: %A.WithSelf.F.type.487 = struct_value () [symbolic]
// CHECK:STDOUT:   %A.assoc_type.edf: type = assoc_entity_type @A, @A(%W) [symbolic]
// CHECK:STDOUT:   %assoc0.5cb: %A.assoc_type.edf = assoc_entity element0, @A.WithSelf.%A.WithSelf.F.decl [symbolic]
// CHECK:STDOUT:   %require_complete.b42: <witness> = require_complete_type %A.type.e49 [symbolic]
// CHECK:STDOUT:   %U.ce4: type = symbolic_binding U, 3 [symbolic]
// CHECK:STDOUT:   %pattern_type.e4a: type = pattern_type %U.ce4 [symbolic]
// CHECK:STDOUT:   %tuple.type.11f: type = tuple_type (type, %tuple.type.24b, type) [concrete]
// CHECK:STDOUT:   %tuple.117: %tuple.type.11f = tuple_value (%W, %tuple.4b9, %U.ce4) [symbolic]
// CHECK:STDOUT:   %tuple.type.897: type = tuple_type (%W, %tuple.type.a5e, %U.ce4) [symbolic]
// CHECK:STDOUT:   %.f8e: Core.Form = init_form %tuple.type.897 [symbolic]
// CHECK:STDOUT:   %pattern_type.5cc: type = pattern_type %tuple.type.897 [symbolic]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type.208: type = fn_type @tuple.type.as.A.impl.F, @tuple.type.as.A.impl(%V1, %V2, %W) [symbolic]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.4ce: %tuple.type.as.A.impl.F.type.208 = struct_value () [symbolic]
// CHECK:STDOUT:   %A.facet.2a9: %A.type.e49 = facet_value %tuple.type.a5e, (%A.impl_witness.6c9) [symbolic]
// CHECK:STDOUT:   %A.WithSelf.F.type.234: type = fn_type @A.WithSelf.F, @A.WithSelf(%W, %A.facet.2a9) [symbolic]
// CHECK:STDOUT:   %A.WithSelf.F.329: %A.WithSelf.F.type.234 = struct_value () [symbolic]
// CHECK:STDOUT:   %tuple.type.ec9: type = tuple_type (type, %A.type.e49, type) [symbolic]
// CHECK:STDOUT:   %tuple.db8: %tuple.type.ec9 = tuple_value (%W, %A.facet.2a9, %U.ce4) [symbolic]
// CHECK:STDOUT:   %require_complete.5a4: <witness> = require_complete_type %U.ce4 [symbolic]
// CHECK:STDOUT:   %require_complete.da0: <witness> = require_complete_type %tuple.type.897 [symbolic]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.specific_fn.e1b: <specific function> = specific_function %tuple.type.as.A.impl.F.4ce, @tuple.type.as.A.impl.F(%V1, %V2, %W, %U.ce4) [symbolic]
// CHECK:STDOUT:   %Call.type: type = fn_type @Call [concrete]
// CHECK:STDOUT:   %Call: %Call.type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.334: %tuple.type.24b = tuple_value (%Y1, %Y2) [concrete]
// CHECK:STDOUT:   %tuple.type.a7c: type = tuple_type (%Y1, %Y2) [concrete]
// CHECK:STDOUT:   %A.type.ef5: type = facet_type <@A, @A(%X)> [concrete]
// CHECK:STDOUT:   %A.impl_witness.755: <witness> = impl_witness @tuple.type.as.A.impl.%A.impl_witness_table, @tuple.type.as.A.impl(%Y1, %Y2, %X) [concrete]
// CHECK:STDOUT:   %Self.0c0: %A.type.ef5 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %A.WithSelf.F.type.c12: type = fn_type @A.WithSelf.F, @A.WithSelf(%X, %Self.a33) [symbolic]
// CHECK:STDOUT:   %A.WithSelf.F.612: %A.WithSelf.F.type.c12 = struct_value () [symbolic]
// CHECK:STDOUT:   %A.assoc_type.3ce: type = assoc_entity_type @A, @A(%X) [concrete]
// CHECK:STDOUT:   %assoc0.2c2: %A.assoc_type.3ce = assoc_entity element0, @A.WithSelf.%A.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %complete_type.740: <witness> = complete_type_witness %A.type.ef5 [concrete]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type.d8e: type = fn_type @tuple.type.as.A.impl.F, @tuple.type.as.A.impl(%Y1, %Y2, %X) [concrete]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.677: %tuple.type.as.A.impl.F.type.d8e = struct_value () [concrete]
// CHECK:STDOUT:   %A.facet.d43: %A.type.ef5 = facet_value %tuple.type.a7c, (%A.impl_witness.755) [concrete]
// CHECK:STDOUT:   %A.WithSelf.F.type.4b7: type = fn_type @A.WithSelf.F, @A.WithSelf(%X, %A.facet.d43) [concrete]
// CHECK:STDOUT:   %A.WithSelf.F.6c3: %A.WithSelf.F.type.4b7 = struct_value () [concrete]
// CHECK:STDOUT:   %.0d1: type = fn_type_with_self_type %A.WithSelf.F.type.4b7, %A.facet.d43 [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.f0a: type = pattern_type %Z [concrete]
// CHECK:STDOUT:   %tuple.b30: %tuple.type.11f = tuple_value (%X, %tuple.334, %Z) [concrete]
// CHECK:STDOUT:   %tuple.type.df6: type = tuple_type (%X, %tuple.type.a7c, %Z) [concrete]
// CHECK:STDOUT:   %.ec0: Core.Form = init_form %tuple.type.df6 [concrete]
// CHECK:STDOUT:   %pattern_type.cc8: type = pattern_type %tuple.type.df6 [concrete]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.specific_fn.b53: <specific function> = specific_function %tuple.type.as.A.impl.F.677, @tuple.type.as.A.impl.F(%Y1, %Y2, %X, %Z) [concrete]
// CHECK:STDOUT:   %Z.val: %Z = struct_value () [concrete]
// CHECK:STDOUT:   %.224: ref %Z = temporary invalid, %Z.val [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.5: type = fn_type @Destroy.Op.loc24_39.5 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.5: %Destroy.Op.type.790b3b.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.e20f1f.5: <witness> = custom_witness (%Destroy.Op.2dd605.5), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.6: type = fn_type @Destroy.Op.loc24_39.6 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.6: %Destroy.Op.type.790b3b.6 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.7: type = fn_type @Destroy.Op.loc24_39.7 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.7: %Destroy.Op.type.790b3b.7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.e20f1f.7: <witness> = custom_witness (%Destroy.Op.2dd605.7), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.195: %Destroy.type = facet_value %tuple.type.df6, (%custom_witness.e20f1f.7) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.acf: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.195) [concrete]
// CHECK:STDOUT:   %.cdc: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.acf, %Destroy.facet.195 [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.224, %Destroy.Op.2dd605.6 [concrete]
// CHECK:STDOUT:   %BitAndWith.type.fd8: type = generic_interface_type @BitAndWith [concrete]
// CHECK:STDOUT:   %BitAndWith.generic: %BitAndWith.type.fd8 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndWith.type.2af: type = facet_type <@BitAndWith, @BitAndWith(type)> [concrete]
// CHECK:STDOUT:   %BitAndWith.impl_witness: <witness> = impl_witness imports.%BitAndWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %BitAndWith.facet: %BitAndWith.type.2af = facet_value type, (%BitAndWith.impl_witness) [concrete]
// CHECK:STDOUT:   %BitAndWith.WithSelf.Op.type.5d1: type = fn_type @BitAndWith.WithSelf.Op, @BitAndWith.WithSelf(type, %BitAndWith.facet) [concrete]
// CHECK:STDOUT:   %.9ee: type = fn_type_with_self_type %BitAndWith.WithSelf.Op.type.5d1, %BitAndWith.facet [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.type: type = fn_type @type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op: %type.as.BitAndWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.bound: <bound method> = bound_method %A.type.ef5, %type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@A, @A(%X) & @Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.e1b: type = pattern_type %facet_type [concrete]
// CHECK:STDOUT:   %T.0c6: %facet_type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %CallGeneric.type: type = fn_type @CallGeneric [concrete]
// CHECK:STDOUT:   %CallGeneric: %CallGeneric.type = struct_value () [concrete]
// CHECK:STDOUT:   %A.WithSelf.F.type.5f2: type = fn_type @A.WithSelf.F, @A.WithSelf(%X, %T.0c6) [symbolic]
// CHECK:STDOUT:   %A.WithSelf.F.923: %A.WithSelf.F.type.5f2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.0c6 [symbolic]
// CHECK:STDOUT:   %A.lookup_impl_witness: <witness> = lookup_impl_witness %T.0c6, @A, @A(%X) [symbolic]
// CHECK:STDOUT:   %A.facet.210: %A.type.ef5 = facet_value %T.as_type, (%A.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %A.WithSelf.F.type.b88: type = fn_type @A.WithSelf.F, @A.WithSelf(%X, %A.facet.210) [symbolic]
// CHECK:STDOUT:   %A.WithSelf.F.702: %A.WithSelf.F.type.b88 = struct_value () [symbolic]
// CHECK:STDOUT:   %.b59: type = fn_type_with_self_type %A.WithSelf.F.type.b88, %A.facet.210 [symbolic]
// CHECK:STDOUT:   %impl.elem0.65b: %.b59 = impl_witness_access %A.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %tuple.type.d63: type = tuple_type (type, %A.type.ef5, type) [concrete]
// CHECK:STDOUT:   %tuple.4d5: %tuple.type.d63 = tuple_value (%X, %A.facet.210, %Z) [symbolic]
// CHECK:STDOUT:   %tuple.type.5f8: type = tuple_type (%X, %T.as_type, %Z) [symbolic]
// CHECK:STDOUT:   %.bce: Core.Form = init_form %tuple.type.5f8 [symbolic]
// CHECK:STDOUT:   %pattern_type.bb6: type = pattern_type %tuple.type.5f8 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.846: <specific function> = specific_impl_function %impl.elem0.65b, @A.WithSelf.F(%X, %A.facet.210, %Z) [symbolic]
// CHECK:STDOUT:   %require_complete.422: <witness> = require_complete_type %tuple.type.5f8 [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.9d8: <witness> = lookup_impl_witness %tuple.type.5f8, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.10f: %Destroy.type = facet_value %tuple.type.5f8, (%Destroy.lookup_impl_witness.9d8) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.2b3: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.10f) [symbolic]
// CHECK:STDOUT:   %.446: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.2b3, %Destroy.facet.10f [symbolic]
// CHECK:STDOUT:   %impl.elem0.14b: %.446 = impl_witness_access %Destroy.lookup_impl_witness.9d8, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.2ca: <specific function> = specific_impl_function %impl.elem0.14b, @Destroy.WithSelf.Op(%Destroy.facet.10f) [symbolic]
// CHECK:STDOUT:   %CallIndirect.type: type = fn_type @CallIndirect [concrete]
// CHECK:STDOUT:   %CallIndirect: %CallIndirect.type = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value: %facet_type = facet_value %tuple.type.a7c, (%A.impl_witness.755, %custom_witness.e20f1f.5) [concrete]
// CHECK:STDOUT:   %CallGeneric.specific_fn: <specific function> = specific_function %CallGeneric, @CallGeneric(%facet_value) [concrete]
// CHECK:STDOUT:   %complete_type.466: <witness> = complete_type_witness %tuple.type.df6 [concrete]
// CHECK:STDOUT:   %tuple.aac: %tuple.type.d63 = tuple_value (%X, %A.facet.d43, %Z) [concrete]
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
// CHECK:STDOUT:   %Core.BitAndWith: %BitAndWith.type.fd8 = import_ref Core//prelude/parts/as, BitAndWith, loaded [concrete = constants.%BitAndWith.generic]
// CHECK:STDOUT:   %Core.import_ref.f63: %type.as.BitAndWith.impl.Op.type = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, loaded [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:   %BitAndWith.impl_witness_table = impl_witness_table (%Core.import_ref.f63), @type.as.BitAndWith.impl [concrete]
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
// CHECK:STDOUT:   %A.decl: %A.type.0af = interface_decl @A [concrete = constants.%A.generic] {
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
// CHECK:STDOUT:     %A.ref: %A.type.0af = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:     %W.ref: type = name_ref W, %W.loc14_37.2 [symbolic = %W.loc14_37.1 (constants.%W)]
// CHECK:STDOUT:     %A.type.loc14_61.2: type = facet_type <@A, @A(constants.%W)> [symbolic = %A.type.loc14_61.1 (constants.%A.type.e49)]
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
// CHECK:STDOUT:     %T.patt: %pattern_type.e1b = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc27_25.1: type = splice_block %.loc27_25.3 [concrete = constants.%facet_type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %A.ref: %A.type.0af = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:       %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:       %A.type: type = facet_type <@A, @A(constants.%X)> [concrete = constants.%A.type.ef5]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:       %impl.elem0.loc27: %.9ee = impl_witness_access constants.%BitAndWith.impl_witness, element0 [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:       %bound_method.loc27: <bound method> = bound_method %A.type, %impl.elem0.loc27 [concrete = constants.%type.as.BitAndWith.impl.Op.bound]
// CHECK:STDOUT:       %type.as.BitAndWith.impl.Op.call: init type = call %bound_method.loc27(%A.type, %Destroy.ref) [concrete = constants.%facet_type]
// CHECK:STDOUT:       %.loc27_25.2: type = value_of_initializer %type.as.BitAndWith.impl.Op.call [concrete = constants.%facet_type]
// CHECK:STDOUT:       %.loc27_25.3: type = converted %type.as.BitAndWith.impl.Op.call, %.loc27_25.2 [concrete = constants.%facet_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc27_17.2: %facet_type = symbolic_binding T, 0 [symbolic = %T.loc27_17.1 (constants.%T.0c6)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallIndirect.decl: %CallIndirect.type = fn_decl @CallIndirect [concrete = constants.%CallIndirect] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @A(%T.loc5_14.2: type) {
// CHECK:STDOUT:   %T.loc5_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc5_14.1 (constants.%T.67d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.type: type = facet_type <@A, @A(%T.loc5_14.1)> [symbolic = %A.type (constants.%A.type.5ea)]
// CHECK:STDOUT:   %Self.loc5_23.2: @A.%A.type (%A.type.5ea) = symbolic_binding Self, 1 [symbolic = %Self.loc5_23.2 (constants.%Self.a33)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc5_23.1: @A.%A.type (%A.type.5ea) = symbolic_binding Self, 1 [symbolic = %Self.loc5_23.2 (constants.%Self.a33)]
// CHECK:STDOUT:     %A.WithSelf.decl = interface_with_self_decl @A [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %A.WithSelf.F.decl: @A.WithSelf.%A.WithSelf.F.type (%A.WithSelf.F.type.f0d) = fn_decl @A.WithSelf.F [symbolic = @A.WithSelf.%A.WithSelf.F (constants.%A.WithSelf.F.628)] {
// CHECK:STDOUT:       %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 2 [concrete]
// CHECK:STDOUT:       %u.param_patt: @A.WithSelf.F.%pattern_type.loc6_19 (%pattern_type.62e) = value_param_pattern [concrete]
// CHECK:STDOUT:       %u.patt: @A.WithSelf.F.%pattern_type.loc6_19 (%pattern_type.62e) = at_binding_pattern u, %u.param_patt [concrete]
// CHECK:STDOUT:       %return.param_patt: @A.WithSelf.F.%pattern_type.loc6_38 (%pattern_type.41c) = out_param_pattern [concrete]
// CHECK:STDOUT:       %return.patt: @A.WithSelf.F.%pattern_type.loc6_38 (%pattern_type.41c) = return_slot_pattern %return.param_patt, %.loc6_38.4 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @A.%T.loc5_14.2 [symbolic = %T (constants.%T.67d)]
// CHECK:STDOUT:       %.loc6_31: @A.WithSelf.F.%A.type (%A.type.5ea) = specific_constant @A.%Self.loc5_23.1, @A(constants.%T.67d) [symbolic = %Self (constants.%Self.a33)]
// CHECK:STDOUT:       %Self.ref: @A.WithSelf.F.%A.type (%A.type.5ea) = name_ref Self, %.loc6_31 [symbolic = %Self (constants.%Self.a33)]
// CHECK:STDOUT:       %U.ref.loc6_37: type = name_ref U, %U.loc6_9.2 [symbolic = %U.loc6_9.1 (constants.%U.4e7)]
// CHECK:STDOUT:       %.loc6_38.2: @A.WithSelf.F.%tuple.type.loc6_38.1 (%tuple.type.54d) = tuple_literal (%T.ref, %Self.ref, %U.ref.loc6_37) [symbolic = %tuple (constants.%tuple.ca1)]
// CHECK:STDOUT:       %Self.as_type.loc6_38.2: type = facet_access_type constants.%Self.a33 [symbolic = %Self.as_type.loc6_38.1 (constants.%Self.as_type.0a5)]
// CHECK:STDOUT:       %.loc6_38.3: type = converted constants.%Self.a33, %Self.as_type.loc6_38.2 [symbolic = %Self.as_type.loc6_38.1 (constants.%Self.as_type.0a5)]
// CHECK:STDOUT:       %.loc6_38.4: type = converted %.loc6_38.2, constants.%tuple.type.a60 [symbolic = %tuple.type.loc6_38.2 (constants.%tuple.type.a60)]
// CHECK:STDOUT:       %.loc6_38.5: Core.Form = init_form %.loc6_38.4 [symbolic = %.loc6_38.1 (constants.%.ddd)]
// CHECK:STDOUT:       %.loc6_12.1: type = splice_block %.loc6_12.2 [concrete = type] {
// CHECK:STDOUT:         %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:         %.loc6_12.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %U.loc6_9.2: type = symbolic_binding U, 2 [symbolic = %U.loc6_9.1 (constants.%U.4e7)]
// CHECK:STDOUT:       %u.param: @A.WithSelf.F.%U.loc6_9.1 (%U.4e7) = value_param call_param0
// CHECK:STDOUT:       %U.ref.loc6_21: type = name_ref U, %U.loc6_9.2 [symbolic = %U.loc6_9.1 (constants.%U.4e7)]
// CHECK:STDOUT:       %u: @A.WithSelf.F.%U.loc6_9.1 (%U.4e7) = value_binding u, %u.param
// CHECK:STDOUT:       %return.param: ref @A.WithSelf.F.%tuple.type.loc6_38.2 (%tuple.type.a60) = out_param call_param1
// CHECK:STDOUT:       %return: ref @A.WithSelf.F.%tuple.type.loc6_38.2 (%tuple.type.a60) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc6_39.1: @A.WithSelf.%A.assoc_type (%A.assoc_type.092) = assoc_entity element0, %A.WithSelf.F.decl [symbolic = %assoc0.loc6_39.2 (constants.%assoc0.7bb)]
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
// CHECK:STDOUT:   %A.type.loc14_61.1: type = facet_type <@A, @A(%W.loc14_37.1)> [symbolic = %A.type.loc14_61.1 (constants.%A.type.e49)]
// CHECK:STDOUT:   %A.impl_witness.loc14_63.2: <witness> = impl_witness %A.impl_witness_table, @tuple.type.as.A.impl(%V1.loc14_16.1, %V2.loc14_27.1, %W.loc14_37.1) [symbolic = %A.impl_witness.loc14_63.2 (constants.%A.impl_witness.6c9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %A.type.loc14_61.1 [symbolic = %require_complete (constants.%require_complete.b42)]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type: type = fn_type @tuple.type.as.A.impl.F, @tuple.type.as.A.impl(%V1.loc14_16.1, %V2.loc14_27.1, %W.loc14_37.1) [symbolic = %tuple.type.as.A.impl.F.type (constants.%tuple.type.as.A.impl.F.type.208)]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F: @tuple.type.as.A.impl.%tuple.type.as.A.impl.F.type (%tuple.type.as.A.impl.F.type.208) = struct_value () [symbolic = %tuple.type.as.A.impl.F (constants.%tuple.type.as.A.impl.F.4ce)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc14_53.2 as %A.type.loc14_61.2 {
// CHECK:STDOUT:     %tuple.type.as.A.impl.F.decl: @tuple.type.as.A.impl.%tuple.type.as.A.impl.F.type (%tuple.type.as.A.impl.F.type.208) = fn_decl @tuple.type.as.A.impl.F [symbolic = @tuple.type.as.A.impl.%tuple.type.as.A.impl.F (constants.%tuple.type.as.A.impl.F.4ce)] {
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
// CHECK:STDOUT:     %A.impl_witness.loc14_63.1: <witness> = impl_witness %A.impl_witness_table, @tuple.type.as.A.impl(constants.%V1, constants.%V2, constants.%W) [symbolic = %A.impl_witness.loc14_63.2 (constants.%A.impl_witness.6c9)]
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
// CHECK:STDOUT: generic fn @A.WithSelf.F(@A.%T.loc5_14.2: type, @A.%Self.loc5_23.1: @A.%A.type (%A.type.5ea), %U.loc6_9.2: type) {
// CHECK:STDOUT:   %U.loc6_9.1: type = symbolic_binding U, 2 [symbolic = %U.loc6_9.1 (constants.%U.4e7)]
// CHECK:STDOUT:   %pattern_type.loc6_19: type = pattern_type %U.loc6_9.1 [symbolic = %pattern_type.loc6_19 (constants.%pattern_type.62e)]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T.67d)]
// CHECK:STDOUT:   %A.type: type = facet_type <@A, @A(%T)> [symbolic = %A.type (constants.%A.type.5ea)]
// CHECK:STDOUT:   %Self: @A.WithSelf.F.%A.type (%A.type.5ea) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.a33)]
// CHECK:STDOUT:   %tuple.type.loc6_38.1: type = tuple_type (type, %A.type, type) [symbolic = %tuple.type.loc6_38.1 (constants.%tuple.type.54d)]
// CHECK:STDOUT:   %tuple: @A.WithSelf.F.%tuple.type.loc6_38.1 (%tuple.type.54d) = tuple_value (%T, %Self, %U.loc6_9.1) [symbolic = %tuple (constants.%tuple.ca1)]
// CHECK:STDOUT:   %Self.as_type.loc6_38.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc6_38.1 (constants.%Self.as_type.0a5)]
// CHECK:STDOUT:   %tuple.type.loc6_38.2: type = tuple_type (%T, %Self.as_type.loc6_38.1, %U.loc6_9.1) [symbolic = %tuple.type.loc6_38.2 (constants.%tuple.type.a60)]
// CHECK:STDOUT:   %.loc6_38.1: Core.Form = init_form %tuple.type.loc6_38.2 [symbolic = %.loc6_38.1 (constants.%.ddd)]
// CHECK:STDOUT:   %pattern_type.loc6_38: type = pattern_type %tuple.type.loc6_38.2 [symbolic = %pattern_type.loc6_38 (constants.%pattern_type.41c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param: @A.WithSelf.F.%U.loc6_9.1 (%U.4e7)) -> out %return.param: @A.WithSelf.F.%tuple.type.loc6_38.2 (%tuple.type.a60);
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
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type: type = fn_type @tuple.type.as.A.impl.F, @tuple.type.as.A.impl(%V1, %V2, %W) [symbolic = %tuple.type.as.A.impl.F.type (constants.%tuple.type.as.A.impl.F.type.208)]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F: @tuple.type.as.A.impl.F.%tuple.type.as.A.impl.F.type (%tuple.type.as.A.impl.F.type.208) = struct_value () [symbolic = %tuple.type.as.A.impl.F (constants.%tuple.type.as.A.impl.F.4ce)]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.specific_fn.loc18_12.2: <specific function> = specific_function %tuple.type.as.A.impl.F, @tuple.type.as.A.impl.F(%V1, %V2, %W, %U.loc17_9.1) [symbolic = %tuple.type.as.A.impl.F.specific_fn.loc18_12.2 (constants.%tuple.type.as.A.impl.F.specific_fn.e1b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param: @tuple.type.as.A.impl.F.%U.loc17_9.1 (%U.ce4)) -> out %return.param: @tuple.type.as.A.impl.F.%tuple.type.loc17_42.2 (%tuple.type.897) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc18: @tuple.type.as.A.impl.F.%tuple.type.as.A.impl.F.type (%tuple.type.as.A.impl.F.type.208) = specific_constant @tuple.type.as.A.impl.%tuple.type.as.A.impl.F.decl, @tuple.type.as.A.impl(constants.%V1, constants.%V2, constants.%W) [symbolic = %tuple.type.as.A.impl.F (constants.%tuple.type.as.A.impl.F.4ce)]
// CHECK:STDOUT:     %F.ref: @tuple.type.as.A.impl.F.%tuple.type.as.A.impl.F.type (%tuple.type.as.A.impl.F.type.208) = name_ref F, %.loc18 [symbolic = %tuple.type.as.A.impl.F (constants.%tuple.type.as.A.impl.F.4ce)]
// CHECK:STDOUT:     %U.ref.loc18: type = name_ref U, %U.loc17_9.2 [symbolic = %U.loc17_9.1 (constants.%U.ce4)]
// CHECK:STDOUT:     %u.ref: @tuple.type.as.A.impl.F.%U.loc17_9.1 (%U.ce4) = name_ref u, %u
// CHECK:STDOUT:     %tuple.type.as.A.impl.F.specific_fn.loc18_12.1: <specific function> = specific_function %F.ref, @tuple.type.as.A.impl.F(constants.%V1, constants.%V2, constants.%W, constants.%U.ce4) [symbolic = %tuple.type.as.A.impl.F.specific_fn.loc18_12.2 (constants.%tuple.type.as.A.impl.F.specific_fn.e1b)]
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
// CHECK:STDOUT:   %.loc24_12: %tuple.type.24b = tuple_literal (%Y1.ref, %Y2.ref) [concrete = constants.%tuple.334]
// CHECK:STDOUT:   %.loc24_17: type = type_literal type [concrete = type]
// CHECK:STDOUT:   %.loc24_14: type = converted %.loc24_12, constants.%tuple.type.a7c [concrete = constants.%tuple.type.a7c]
// CHECK:STDOUT:   %A.ref: %A.type.0af = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:   %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %A.type: type = facet_type <@A, @A(constants.%X)> [concrete = constants.%A.type.ef5]
// CHECK:STDOUT:   %A.facet: %A.type.ef5 = facet_value %.loc24_14, (constants.%A.impl_witness.755) [concrete = constants.%A.facet.d43]
// CHECK:STDOUT:   %.loc24_23: %A.type.ef5 = converted %.loc24_14, %A.facet [concrete = constants.%A.facet.d43]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc24_23 [concrete = constants.%tuple.type.a7c]
// CHECK:STDOUT:   %.loc24_31.1: type = converted %.loc24_23, %as_type [concrete = constants.%tuple.type.a7c]
// CHECK:STDOUT:   %.loc24_31.2: %A.assoc_type.3ce = specific_constant @A.WithSelf.%assoc0.loc6_39.1, @A.WithSelf(constants.%X, constants.%A.facet.d43) [concrete = constants.%assoc0.2c2]
// CHECK:STDOUT:   %F.ref: %A.assoc_type.3ce = name_ref F, %.loc24_31.2 [concrete = constants.%assoc0.2c2]
// CHECK:STDOUT:   %impl.elem0: %.0d1 = impl_witness_access constants.%A.impl_witness.755, element0 [concrete = constants.%tuple.type.as.A.impl.F.677]
// CHECK:STDOUT:   %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z]
// CHECK:STDOUT:   %.loc24_38.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @tuple.type.as.A.impl.F(constants.%Y1, constants.%Y2, constants.%X, constants.%Z) [concrete = constants.%tuple.type.as.A.impl.F.specific_fn.b53]
// CHECK:STDOUT:   %.loc24_39.1: ref %tuple.type.df6 = temporary_storage
// CHECK:STDOUT:   %.loc24_38.2: ref %Z = temporary_storage
// CHECK:STDOUT:   %.loc24_38.3: init %Z to %.loc24_38.2 = class_init () [concrete = constants.%Z.val]
// CHECK:STDOUT:   %.loc24_38.4: init %Z = converted %.loc24_38.1, %.loc24_38.3 [concrete = constants.%Z.val]
// CHECK:STDOUT:   %.loc24_38.5: ref %Z = temporary %.loc24_38.2, %.loc24_38.4 [concrete = constants.%.224]
// CHECK:STDOUT:   %.loc24_38.6: %Z = acquire_value %.loc24_38.5 [concrete = constants.%Z.val]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.call: init %tuple.type.df6 to %.loc24_39.1 = call %specific_fn(%.loc24_38.6)
// CHECK:STDOUT:   %.loc24_39.2: ref %tuple.type.df6 = temporary %.loc24_39.1, %tuple.type.as.A.impl.F.call
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.loc24_39.2, constants.%Destroy.Op.2dd605.7
// CHECK:STDOUT:   %Destroy.Op.call.loc24_39: init %empty_tuple.type = call %Destroy.Op.bound(%.loc24_39.2)
// CHECK:STDOUT:   %Destroy.Op.call.loc24_38: init %empty_tuple.type = call constants.%Destroy.Op.bound(constants.%.224)
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
// CHECK:STDOUT: fn @Destroy.Op.loc24_39.5(%self.param: ref %tuple.type.a7c) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc24_39.6(%self.param: ref %Z) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc24_39.7(%self.param: ref %tuple.type.df6) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallGeneric(%T.loc27_17.2: %facet_type) {
// CHECK:STDOUT:   %T.loc27_17.1: %facet_type = symbolic_binding T, 0 [symbolic = %T.loc27_17.1 (constants.%T.0c6)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc28_4.2: type = facet_access_type %T.loc27_17.1 [symbolic = %T.as_type.loc28_4.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %A.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc27_17.1, @A, @A(constants.%X) [symbolic = %A.lookup_impl_witness (constants.%A.lookup_impl_witness)]
// CHECK:STDOUT:   %A.facet: %A.type.ef5 = facet_value %T.as_type.loc28_4.2, (%A.lookup_impl_witness) [symbolic = %A.facet (constants.%A.facet.210)]
// CHECK:STDOUT:   %A.WithSelf.F.type: type = fn_type @A.WithSelf.F, @A.WithSelf(constants.%X, %A.facet) [symbolic = %A.WithSelf.F.type (constants.%A.WithSelf.F.type.b88)]
// CHECK:STDOUT:   %.loc28_4.3: type = fn_type_with_self_type %A.WithSelf.F.type, %A.facet [symbolic = %.loc28_4.3 (constants.%.b59)]
// CHECK:STDOUT:   %impl.elem0.loc28_4.2: @CallGeneric.%.loc28_4.3 (%.b59) = impl_witness_access %A.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc28_4.2 (constants.%impl.elem0.65b)]
// CHECK:STDOUT:   %specific_impl_fn.loc28_4.2: <specific function> = specific_impl_function %impl.elem0.loc28_4.2, @A.WithSelf.F(constants.%X, %A.facet, constants.%Z) [symbolic = %specific_impl_fn.loc28_4.2 (constants.%specific_impl_fn.846)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (constants.%X, %T.as_type.loc28_4.2, constants.%Z) [symbolic = %tuple.type (constants.%tuple.type.5f8)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %tuple.type [symbolic = %require_complete (constants.%require_complete.422)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %tuple.type, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.9d8)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %tuple.type, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet (constants.%Destroy.facet.10f)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.2b3)]
// CHECK:STDOUT:   %.loc28_12.3: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Destroy.facet [symbolic = %.loc28_12.3 (constants.%.446)]
// CHECK:STDOUT:   %impl.elem0.loc28_12.2: @CallGeneric.%.loc28_12.3 (%.446) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc28_12.2 (constants.%impl.elem0.14b)]
// CHECK:STDOUT:   %specific_impl_fn.loc28_12.2: <specific function> = specific_impl_function %impl.elem0.loc28_12.2, @Destroy.WithSelf.Op(%Destroy.facet) [symbolic = %specific_impl_fn.loc28_12.2 (constants.%specific_impl_fn.2ca)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref: %facet_type = name_ref T, %T.loc27_17.2 [symbolic = %T.loc27_17.1 (constants.%T.0c6)]
// CHECK:STDOUT:     %T.as_type.loc28_4.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc28_4.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc28_4.1: type = converted %T.ref, %T.as_type.loc28_4.1 [symbolic = %T.as_type.loc28_4.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc28_4.2: %A.assoc_type.3ce = specific_constant @A.WithSelf.%assoc0.loc6_39.1, @A.WithSelf(constants.%X, constants.%T.0c6) [concrete = constants.%assoc0.2c2]
// CHECK:STDOUT:     %F.ref: %A.assoc_type.3ce = name_ref F, %.loc28_4.2 [concrete = constants.%assoc0.2c2]
// CHECK:STDOUT:     %impl.elem0.loc28_4.1: @CallGeneric.%.loc28_4.3 (%.b59) = impl_witness_access constants.%A.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc28_4.2 (constants.%impl.elem0.65b)]
// CHECK:STDOUT:     %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z]
// CHECK:STDOUT:     %.loc28_11.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %specific_impl_fn.loc28_4.1: <specific function> = specific_impl_function %impl.elem0.loc28_4.1, @A.WithSelf.F(constants.%X, constants.%A.facet.210, constants.%Z) [symbolic = %specific_impl_fn.loc28_4.2 (constants.%specific_impl_fn.846)]
// CHECK:STDOUT:     %.loc28_12.1: ref @CallGeneric.%tuple.type (%tuple.type.5f8) = temporary_storage
// CHECK:STDOUT:     %.loc28_11.2: ref %Z = temporary_storage
// CHECK:STDOUT:     %.loc28_11.3: init %Z to %.loc28_11.2 = class_init () [concrete = constants.%Z.val]
// CHECK:STDOUT:     %.loc28_11.4: init %Z = converted %.loc28_11.1, %.loc28_11.3 [concrete = constants.%Z.val]
// CHECK:STDOUT:     %.loc28_11.5: ref %Z = temporary %.loc28_11.2, %.loc28_11.4 [concrete = constants.%.224]
// CHECK:STDOUT:     %.loc28_11.6: %Z = acquire_value %.loc28_11.5 [concrete = constants.%Z.val]
// CHECK:STDOUT:     %A.WithSelf.F.call: init @CallGeneric.%tuple.type (%tuple.type.5f8) to %.loc28_12.1 = call %specific_impl_fn.loc28_4.1(%.loc28_11.6)
// CHECK:STDOUT:     %.loc28_12.2: ref @CallGeneric.%tuple.type (%tuple.type.5f8) = temporary %.loc28_12.1, %A.WithSelf.F.call
// CHECK:STDOUT:     %impl.elem0.loc28_12.1: @CallGeneric.%.loc28_12.3 (%.446) = impl_witness_access constants.%Destroy.lookup_impl_witness.9d8, element0 [symbolic = %impl.elem0.loc28_12.2 (constants.%impl.elem0.14b)]
// CHECK:STDOUT:     %bound_method.loc28_12.1: <bound method> = bound_method %.loc28_12.2, %impl.elem0.loc28_12.1
// CHECK:STDOUT:     %specific_impl_fn.loc28_12.1: <specific function> = specific_impl_function %impl.elem0.loc28_12.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.10f) [symbolic = %specific_impl_fn.loc28_12.2 (constants.%specific_impl_fn.2ca)]
// CHECK:STDOUT:     %bound_method.loc28_12.2: <bound method> = bound_method %.loc28_12.2, %specific_impl_fn.loc28_12.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc28_12.2(%.loc28_12.2)
// CHECK:STDOUT:     %Destroy.Op.call: init %empty_tuple.type = call constants.%Destroy.Op.bound(constants.%.224)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallIndirect() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CallGeneric.ref: %CallGeneric.type = name_ref CallGeneric, file.%CallGeneric.decl [concrete = constants.%CallGeneric]
// CHECK:STDOUT:   %Y1.ref: type = name_ref Y1, file.%Y1.decl [concrete = constants.%Y1]
// CHECK:STDOUT:   %Y2.ref: type = name_ref Y2, file.%Y2.decl [concrete = constants.%Y2]
// CHECK:STDOUT:   %.loc33_22: %tuple.type.24b = tuple_literal (%Y1.ref, %Y2.ref) [concrete = constants.%tuple.334]
// CHECK:STDOUT:   %.loc33_27: type = type_literal type [concrete = type]
// CHECK:STDOUT:   %.loc33_24: type = converted %.loc33_22, constants.%tuple.type.a7c [concrete = constants.%tuple.type.a7c]
// CHECK:STDOUT:   %facet_value: %facet_type = facet_value %.loc33_24, (constants.%A.impl_witness.755, constants.%custom_witness.e20f1f.5) [concrete = constants.%facet_value]
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
// CHECK:STDOUT: specific @A.WithSelf(constants.%T.67d, constants.%Self.a33) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf.F(constants.%T.67d, constants.%Self.a33, constants.%U.4e7) {
// CHECK:STDOUT:   %U.loc6_9.1 => constants.%U.4e7
// CHECK:STDOUT:   %pattern_type.loc6_19 => constants.%pattern_type.62e
// CHECK:STDOUT:   %T => constants.%T.67d
// CHECK:STDOUT:   %A.type => constants.%A.type.5ea
// CHECK:STDOUT:   %Self => constants.%Self.a33
// CHECK:STDOUT:   %tuple.type.loc6_38.1 => constants.%tuple.type.54d
// CHECK:STDOUT:   %tuple => constants.%tuple.ca1
// CHECK:STDOUT:   %Self.as_type.loc6_38.1 => constants.%Self.as_type.0a5
// CHECK:STDOUT:   %tuple.type.loc6_38.2 => constants.%tuple.type.a60
// CHECK:STDOUT:   %.loc6_38.1 => constants.%.ddd
// CHECK:STDOUT:   %pattern_type.loc6_38 => constants.%pattern_type.41c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%W) {
// CHECK:STDOUT:   %T.loc5_14.1 => constants.%W
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.type => constants.%A.type.e49
// CHECK:STDOUT:   %Self.loc5_23.2 => constants.%Self.2ba
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @tuple.type.as.A.impl(constants.%V1, constants.%V2, constants.%W) {
// CHECK:STDOUT:   %V1.loc14_16.1 => constants.%V1
// CHECK:STDOUT:   %V2.loc14_27.1 => constants.%V2
// CHECK:STDOUT:   %W.loc14_37.1 => constants.%W
// CHECK:STDOUT:   %tuple => constants.%tuple.4b9
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.a5e
// CHECK:STDOUT:   %A.type.loc14_61.1 => constants.%A.type.e49
// CHECK:STDOUT:   %A.impl_witness.loc14_63.2 => constants.%A.impl_witness.6c9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.b42
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type => constants.%tuple.type.as.A.impl.F.type.208
// CHECK:STDOUT:   %tuple.type.as.A.impl.F => constants.%tuple.type.as.A.impl.F.4ce
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%W, constants.%Self.a33) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%W
// CHECK:STDOUT:   %A.type => constants.%A.type.e49
// CHECK:STDOUT:   %Self => constants.%Self.a33
// CHECK:STDOUT:   %A.WithSelf.F.type => constants.%A.WithSelf.F.type.487
// CHECK:STDOUT:   %A.WithSelf.F => constants.%A.WithSelf.F.e2a
// CHECK:STDOUT:   %A.assoc_type => constants.%A.assoc_type.edf
// CHECK:STDOUT:   %assoc0.loc6_39.2 => constants.%assoc0.5cb
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
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type => constants.%tuple.type.as.A.impl.F.type.208
// CHECK:STDOUT:   %tuple.type.as.A.impl.F => constants.%tuple.type.as.A.impl.F.4ce
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.specific_fn.loc18_12.2 => constants.%tuple.type.as.A.impl.F.specific_fn.e1b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%W, constants.%A.facet.2a9) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%W
// CHECK:STDOUT:   %A.type => constants.%A.type.e49
// CHECK:STDOUT:   %Self => constants.%A.facet.2a9
// CHECK:STDOUT:   %A.WithSelf.F.type => constants.%A.WithSelf.F.type.234
// CHECK:STDOUT:   %A.WithSelf.F => constants.%A.WithSelf.F.329
// CHECK:STDOUT:   %A.assoc_type => constants.%A.assoc_type.edf
// CHECK:STDOUT:   %assoc0.loc6_39.2 => constants.%assoc0.5cb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf.F(constants.%W, constants.%A.facet.2a9, constants.%U.ce4) {
// CHECK:STDOUT:   %U.loc6_9.1 => constants.%U.ce4
// CHECK:STDOUT:   %pattern_type.loc6_19 => constants.%pattern_type.e4a
// CHECK:STDOUT:   %T => constants.%W
// CHECK:STDOUT:   %A.type => constants.%A.type.e49
// CHECK:STDOUT:   %Self => constants.%A.facet.2a9
// CHECK:STDOUT:   %tuple.type.loc6_38.1 => constants.%tuple.type.ec9
// CHECK:STDOUT:   %tuple => constants.%tuple.db8
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
// CHECK:STDOUT:   %A.type => constants.%A.type.ef5
// CHECK:STDOUT:   %Self.loc5_23.2 => constants.%Self.0c0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @tuple.type.as.A.impl(constants.%Y1, constants.%Y2, constants.%X) {
// CHECK:STDOUT:   %V1.loc14_16.1 => constants.%Y1
// CHECK:STDOUT:   %V2.loc14_27.1 => constants.%Y2
// CHECK:STDOUT:   %W.loc14_37.1 => constants.%X
// CHECK:STDOUT:   %tuple => constants.%tuple.334
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.a7c
// CHECK:STDOUT:   %A.type.loc14_61.1 => constants.%A.type.ef5
// CHECK:STDOUT:   %A.impl_witness.loc14_63.2 => constants.%A.impl_witness.755
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.740
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type => constants.%tuple.type.as.A.impl.F.type.d8e
// CHECK:STDOUT:   %tuple.type.as.A.impl.F => constants.%tuple.type.as.A.impl.F.677
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%X, constants.%Self.a33) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.ef5
// CHECK:STDOUT:   %Self => constants.%Self.a33
// CHECK:STDOUT:   %A.WithSelf.F.type => constants.%A.WithSelf.F.type.c12
// CHECK:STDOUT:   %A.WithSelf.F => constants.%A.WithSelf.F.612
// CHECK:STDOUT:   %A.assoc_type => constants.%A.assoc_type.3ce
// CHECK:STDOUT:   %assoc0.loc6_39.2 => constants.%assoc0.2c2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%X, constants.%A.facet.d43) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.ef5
// CHECK:STDOUT:   %Self => constants.%A.facet.d43
// CHECK:STDOUT:   %A.WithSelf.F.type => constants.%A.WithSelf.F.type.4b7
// CHECK:STDOUT:   %A.WithSelf.F => constants.%A.WithSelf.F.6c3
// CHECK:STDOUT:   %A.assoc_type => constants.%A.assoc_type.3ce
// CHECK:STDOUT:   %assoc0.loc6_39.2 => constants.%assoc0.2c2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @tuple.type.as.A.impl.F(constants.%Y1, constants.%Y2, constants.%X, constants.%Z) {
// CHECK:STDOUT:   %U.loc17_9.1 => constants.%Z
// CHECK:STDOUT:   %pattern_type.loc17_19 => constants.%pattern_type.f0a
// CHECK:STDOUT:   %W => constants.%X
// CHECK:STDOUT:   %V1 => constants.%Y1
// CHECK:STDOUT:   %V2 => constants.%Y2
// CHECK:STDOUT:   %tuple.loc17_38 => constants.%tuple.334
// CHECK:STDOUT:   %tuple.loc17_42 => constants.%tuple.b30
// CHECK:STDOUT:   %tuple.type.loc17_42.1 => constants.%tuple.type.a7c
// CHECK:STDOUT:   %tuple.type.loc17_42.2 => constants.%tuple.type.df6
// CHECK:STDOUT:   %.loc17_42.2 => constants.%.ec0
// CHECK:STDOUT:   %pattern_type.loc17_42 => constants.%pattern_type.cc8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc17_19 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc17_24 => constants.%complete_type.466
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type => constants.%tuple.type.as.A.impl.F.type.d8e
// CHECK:STDOUT:   %tuple.type.as.A.impl.F => constants.%tuple.type.as.A.impl.F.677
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.specific_fn.loc18_12.2 => constants.%tuple.type.as.A.impl.F.specific_fn.b53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGeneric(constants.%T.0c6) {
// CHECK:STDOUT:   %T.loc27_17.1 => constants.%T.0c6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%X, constants.%T.0c6) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.ef5
// CHECK:STDOUT:   %Self => constants.%T.0c6
// CHECK:STDOUT:   %A.WithSelf.F.type => constants.%A.WithSelf.F.type.5f2
// CHECK:STDOUT:   %A.WithSelf.F => constants.%A.WithSelf.F.923
// CHECK:STDOUT:   %A.assoc_type => constants.%A.assoc_type.3ce
// CHECK:STDOUT:   %assoc0.loc6_39.2 => constants.%assoc0.2c2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%X, constants.%A.facet.210) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.ef5
// CHECK:STDOUT:   %Self => constants.%A.facet.210
// CHECK:STDOUT:   %A.WithSelf.F.type => constants.%A.WithSelf.F.type.b88
// CHECK:STDOUT:   %A.WithSelf.F => constants.%A.WithSelf.F.702
// CHECK:STDOUT:   %A.assoc_type => constants.%A.assoc_type.3ce
// CHECK:STDOUT:   %assoc0.loc6_39.2 => constants.%assoc0.2c2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf.F(constants.%X, constants.%A.facet.210, constants.%Z) {
// CHECK:STDOUT:   %U.loc6_9.1 => constants.%Z
// CHECK:STDOUT:   %pattern_type.loc6_19 => constants.%pattern_type.f0a
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.ef5
// CHECK:STDOUT:   %Self => constants.%A.facet.210
// CHECK:STDOUT:   %tuple.type.loc6_38.1 => constants.%tuple.type.d63
// CHECK:STDOUT:   %tuple => constants.%tuple.4d5
// CHECK:STDOUT:   %Self.as_type.loc6_38.1 => constants.%T.as_type
// CHECK:STDOUT:   %tuple.type.loc6_38.2 => constants.%tuple.type.5f8
// CHECK:STDOUT:   %.loc6_38.1 => constants.%.bce
// CHECK:STDOUT:   %pattern_type.loc6_38 => constants.%pattern_type.bb6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGeneric(constants.%facet_value) {
// CHECK:STDOUT:   %T.loc27_17.1 => constants.%facet_value
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc28_4.2 => constants.%tuple.type.a7c
// CHECK:STDOUT:   %A.lookup_impl_witness => constants.%A.impl_witness.755
// CHECK:STDOUT:   %A.facet => constants.%A.facet.d43
// CHECK:STDOUT:   %A.WithSelf.F.type => constants.%A.WithSelf.F.type.4b7
// CHECK:STDOUT:   %.loc28_4.3 => constants.%.0d1
// CHECK:STDOUT:   %impl.elem0.loc28_4.2 => constants.%tuple.type.as.A.impl.F.677
// CHECK:STDOUT:   %specific_impl_fn.loc28_4.2 => constants.%tuple.type.as.A.impl.F.specific_fn.b53
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.df6
// CHECK:STDOUT:   %require_complete => constants.%complete_type.466
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e20f1f.7
// CHECK:STDOUT:   %Destroy.facet => constants.%Destroy.facet.195
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.acf
// CHECK:STDOUT:   %.loc28_12.3 => constants.%.cdc
// CHECK:STDOUT:   %impl.elem0.loc28_12.2 => constants.%Destroy.Op.2dd605.7
// CHECK:STDOUT:   %specific_impl_fn.loc28_12.2 => constants.%Destroy.Op.2dd605.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf.F(constants.%X, constants.%A.facet.d43, constants.%Z) {
// CHECK:STDOUT:   %U.loc6_9.1 => constants.%Z
// CHECK:STDOUT:   %pattern_type.loc6_19 => constants.%pattern_type.f0a
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.ef5
// CHECK:STDOUT:   %Self => constants.%A.facet.d43
// CHECK:STDOUT:   %tuple.type.loc6_38.1 => constants.%tuple.type.d63
// CHECK:STDOUT:   %tuple => constants.%tuple.aac
// CHECK:STDOUT:   %Self.as_type.loc6_38.1 => constants.%tuple.type.a7c
// CHECK:STDOUT:   %tuple.type.loc6_38.2 => constants.%tuple.type.df6
// CHECK:STDOUT:   %.loc6_38.1 => constants.%.ec0
// CHECK:STDOUT:   %pattern_type.loc6_38 => constants.%pattern_type.cc8
// CHECK:STDOUT: }
// CHECK:STDOUT:

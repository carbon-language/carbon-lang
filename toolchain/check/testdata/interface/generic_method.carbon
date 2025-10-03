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
interface A(T:! Core.Destroy) {
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
interface A(T:! Core.Destroy) {
  fn F(U:! Core.Destroy, u: U) -> (T, Self, U);
}

class X {}
class Y1 {}
class Y2 {}
class Z {}

impl forall [V1:! Core.Destroy, V2:! Core.Destroy, W:! Core.Destroy] (V1, V2) as A(W) {
  // Here, U has index 3. The specific used for A.F needs to renumber its U from
  // index 2 to index 3.
  fn F(U:! Core.Destroy, u: U) -> (W, (V1, V2), U) {
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
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.afb: %Destroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.3ab: type = pattern_type %Destroy.type [concrete]
// CHECK:STDOUT:   %A.type.495: type = generic_interface_type @A [concrete]
// CHECK:STDOUT:   %A.generic: %A.type.495 = struct_value () [concrete]
// CHECK:STDOUT:   %A.type.e04: type = facet_type <@A, @A(%T.afb)> [symbolic]
// CHECK:STDOUT:   %Self.0c7: %A.type.e04 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %U.63a: type = symbolic_binding U, 2 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %ptr.e49: type = ptr_type %U.63a [symbolic]
// CHECK:STDOUT:   %pattern_type.854: type = pattern_type %ptr.e49 [symbolic]
// CHECK:STDOUT:   %tuple.type.337: type = tuple_type (%Destroy.type, %A.type.e04, type) [symbolic]
// CHECK:STDOUT:   %tuple.3bc: %tuple.type.337 = tuple_value (%T.afb, %Self.0c7, %ptr.e49) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.d87: type = symbolic_binding_type T, 0, %T.afb [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.a47: type = symbolic_binding_type Self, 1, %Self.0c7 [symbolic]
// CHECK:STDOUT:   %tuple.type.132: type = tuple_type (%T.binding.as_type.d87, %Self.binding.as_type.a47, %ptr.e49) [symbolic]
// CHECK:STDOUT:   %pattern_type.dd1: type = pattern_type %tuple.type.132 [symbolic]
// CHECK:STDOUT:   %A.F.type.4bf: type = fn_type @A.F, @A(%T.afb) [symbolic]
// CHECK:STDOUT:   %A.F.186: %A.F.type.4bf = struct_value () [symbolic]
// CHECK:STDOUT:   %A.assoc_type.baa: type = assoc_entity_type @A, @A(%T.afb) [symbolic]
// CHECK:STDOUT:   %assoc0.030: %A.assoc_type.baa = assoc_entity element0, @A.%A.F.decl [symbolic]
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Y: type = class_type @Y [concrete]
// CHECK:STDOUT:   %Z: type = class_type @Z [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %DestroyT: %type_where = symbolic_binding DestroyT, 0 [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.15c: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%DestroyT) [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.fd5: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.15c = struct_value () [symbolic]
// CHECK:STDOUT:   %facet_value.b19: %type_where = facet_value %X, () [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.0eb: <witness> = impl_witness imports.%Destroy.impl_witness_table, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.b19) [concrete]
// CHECK:STDOUT:   %Destroy.facet.08b: %Destroy.type = facet_value %X, (%Destroy.impl_witness.0eb) [concrete]
// CHECK:STDOUT:   %A.type.c3a: type = facet_type <@A, @A(%Destroy.facet.08b)> [concrete]
// CHECK:STDOUT:   %Self.369: %A.type.c3a = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %A.F.type.1cf: type = fn_type @A.F, @A(%Destroy.facet.08b) [concrete]
// CHECK:STDOUT:   %A.F.d66: %A.F.type.1cf = struct_value () [concrete]
// CHECK:STDOUT:   %A.assoc_type.0c3: type = assoc_entity_type @A, @A(%Destroy.facet.08b) [concrete]
// CHECK:STDOUT:   %assoc0.43e: %A.assoc_type.0c3 = assoc_entity element0, @A.%A.F.decl [concrete]
// CHECK:STDOUT:   %A.impl_witness: <witness> = impl_witness file.%A.impl_witness_table [concrete]
// CHECK:STDOUT:   %U.d9f: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %ptr.4f0b5c.1: type = ptr_type %U.d9f [symbolic]
// CHECK:STDOUT:   %pattern_type.a603a0.1: type = pattern_type %ptr.4f0b5c.1 [symbolic]
// CHECK:STDOUT:   %tuple.type.ff9: type = tuple_type (type, type, type) [concrete]
// CHECK:STDOUT:   %tuple.bcf: %tuple.type.ff9 = tuple_value (%X, %Y, %ptr.4f0b5c.1) [symbolic]
// CHECK:STDOUT:   %tuple.type.576: type = tuple_type (%X, %Y, %ptr.4f0b5c.1) [symbolic]
// CHECK:STDOUT:   %pattern_type.e0b: type = pattern_type %tuple.type.576 [symbolic]
// CHECK:STDOUT:   %Y.as.A.impl.F.type: type = fn_type @Y.as.A.impl.F [concrete]
// CHECK:STDOUT:   %Y.as.A.impl.F: %Y.as.A.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %A.facet.8b3: %A.type.c3a = facet_value %Y, (%A.impl_witness) [concrete]
// CHECK:STDOUT:   %tuple.type.cfb: type = tuple_type (%Destroy.type, %A.type.c3a, type) [concrete]
// CHECK:STDOUT:   %tuple.1b3: %tuple.type.cfb = tuple_value (%Destroy.facet.08b, %A.facet.8b3, %ptr.4f0b5c.1) [symbolic]
// CHECK:STDOUT:   %require_complete.dc0: <witness> = require_complete_type %tuple.type.576 [symbolic]
// CHECK:STDOUT:   %require_complete.482d3f.1: <witness> = require_complete_type %ptr.4f0b5c.1 [symbolic]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.f88: type = tuple_type (%empty_struct_type, %empty_struct_type, %ptr.4f0b5c.1) [symbolic]
// CHECK:STDOUT:   %X.val: %X = struct_value () [concrete]
// CHECK:STDOUT:   %Y.val: %Y = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %T.d9f: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.75bcbe.1: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.d9f) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.692ac2.1: %ptr.as.Copy.impl.Op.type.75bcbe.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.dcd: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%U.d9f) [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.c66: <witness> = lookup_impl_witness %ptr.4f0b5c.1, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.facet.747: %Copy.type = facet_value %ptr.4f0b5c.1, (%Copy.lookup_impl_witness.c66) [symbolic]
// CHECK:STDOUT:   %.89d: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.747 [symbolic]
// CHECK:STDOUT:   %impl.elem0.928: %.89d = impl_witness_access %Copy.lookup_impl_witness.c66, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.6a1: <specific function> = specific_impl_function %impl.elem0.928, @Copy.Op(%Copy.facet.747) [symbolic]
// CHECK:STDOUT:   %Call.type: type = fn_type @Call [concrete]
// CHECK:STDOUT:   %Call: %Call.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.8f9: type = pattern_type %Z [concrete]
// CHECK:STDOUT:   %.e73: type = fn_type_with_self_type %A.F.type.1cf, %A.facet.8b3 [concrete]
// CHECK:STDOUT:   %ptr.fb6: type = ptr_type %Z [concrete]
// CHECK:STDOUT:   %pattern_type.f76: type = pattern_type %ptr.fb6 [concrete]
// CHECK:STDOUT:   %tuple.f10: %tuple.type.ff9 = tuple_value (%X, %Y, %ptr.fb6) [concrete]
// CHECK:STDOUT:   %tuple.type.092: type = tuple_type (%X, %Y, %ptr.fb6) [concrete]
// CHECK:STDOUT:   %pattern_type.48d: type = pattern_type %tuple.type.092 [concrete]
// CHECK:STDOUT:   %Y.as.A.impl.F.specific_fn: <specific function> = specific_function %Y.as.A.impl.F, @Y.as.A.impl.F(%Z) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %facet_value.352: %type_where = facet_value %tuple.type.092, () [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.cc1: <witness> = impl_witness imports.%Destroy.impl_witness_table, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.352) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.cee: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.352) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.056: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.cee = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type.05d: <witness> = complete_type_witness %tuple.type.092 [concrete]
// CHECK:STDOUT:   %.d23: require_specific_def_type = require_specific_def @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.352) [concrete]
// CHECK:STDOUT:   %Destroy.facet.272: %Destroy.type = facet_value %tuple.type.092, (%Destroy.impl_witness.cc1) [concrete]
// CHECK:STDOUT:   %.30d: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet.272 [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.7b8: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op.056, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value.352) [concrete]
// CHECK:STDOUT:   %facet_value.4e3: %type_where = facet_value %Z, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.d83: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.4e3) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.2cc: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.d83 = struct_value () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.cb3: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op.2cc, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value.4e3) [concrete]
// CHECK:STDOUT:   %BitAndWith.type.f2e: type = generic_interface_type @BitAndWith [concrete]
// CHECK:STDOUT:   %BitAndWith.generic: %BitAndWith.type.f2e = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndWith.type.8a6: type = facet_type <@BitAndWith, @BitAndWith(type)> [concrete]
// CHECK:STDOUT:   %BitAndWith.Op.type.9a3: type = fn_type @BitAndWith.Op, @BitAndWith(type) [concrete]
// CHECK:STDOUT:   %BitAndWith.impl_witness: <witness> = impl_witness imports.%BitAndWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %BitAndWith.facet: %BitAndWith.type.8a6 = facet_value type, (%BitAndWith.impl_witness) [concrete]
// CHECK:STDOUT:   %.fa7: type = fn_type_with_self_type %BitAndWith.Op.type.9a3, %BitAndWith.facet [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.type: type = fn_type @type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op: %type.as.BitAndWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.bound: <bound method> = bound_method %A.type.c3a, %type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@Destroy & @A, @A(%Destroy.facet.08b)> [concrete]
// CHECK:STDOUT:   %T.c3e: %facet_type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.c82: type = pattern_type %facet_type [concrete]
// CHECK:STDOUT:   %CallGeneric.type: type = fn_type @CallGeneric [concrete]
// CHECK:STDOUT:   %CallGeneric: %CallGeneric.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.binding.as_type.7fc: type = symbolic_binding_type T, 0, %T.c3e [symbolic]
// CHECK:STDOUT:   %A.lookup_impl_witness: <witness> = lookup_impl_witness %T.c3e, @A, @A(%Destroy.facet.08b) [symbolic]
// CHECK:STDOUT:   %A.facet.baa: %A.type.c3a = facet_value %T.binding.as_type.7fc, (%A.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %.dc6: type = fn_type_with_self_type %A.F.type.1cf, %A.facet.baa [symbolic]
// CHECK:STDOUT:   %impl.elem0.899: %.dc6 = impl_witness_access %A.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %tuple.9ef: %tuple.type.cfb = tuple_value (%Destroy.facet.08b, %A.facet.baa, %ptr.fb6) [symbolic]
// CHECK:STDOUT:   %tuple.type.063: type = tuple_type (%X, %T.binding.as_type.7fc, %ptr.fb6) [symbolic]
// CHECK:STDOUT:   %pattern_type.150: type = pattern_type %tuple.type.063 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.bf2: <specific function> = specific_impl_function %impl.elem0.899, @A.F(%Destroy.facet.08b, %A.facet.baa, %Z) [symbolic]
// CHECK:STDOUT:   %require_complete.379: <witness> = require_complete_type %tuple.type.063 [symbolic]
// CHECK:STDOUT:   %facet_value.94e: %type_where = facet_value %tuple.type.063, () [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.ff5: <witness> = impl_witness imports.%Destroy.impl_witness_table, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.94e) [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.99f: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.94e) [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.f7e: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.99f = struct_value () [symbolic]
// CHECK:STDOUT:   %.bd7: require_specific_def_type = require_specific_def @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.94e) [symbolic]
// CHECK:STDOUT:   %Destroy.facet.62b: %Destroy.type = facet_value %tuple.type.063, (%Destroy.impl_witness.ff5) [symbolic]
// CHECK:STDOUT:   %.571: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet.62b [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.189: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op.f7e, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value.94e) [symbolic]
// CHECK:STDOUT:   %CallIndirect.type: type = fn_type @CallIndirect [concrete]
// CHECK:STDOUT:   %CallIndirect: %CallIndirect.type = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.5b7: %type_where = facet_value %Y, () [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.688: <witness> = impl_witness imports.%Destroy.impl_witness_table, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.5b7) [concrete]
// CHECK:STDOUT:   %facet_value.312: %facet_type = facet_value %Y, (%Destroy.impl_witness.688, %A.impl_witness) [concrete]
// CHECK:STDOUT:   %CallGeneric.specific_fn: <specific function> = specific_function %CallGeneric, @CallGeneric(%facet_value.312) [concrete]
// CHECK:STDOUT:   %complete_type.d3e: <witness> = complete_type_witness %ptr.fb6 [concrete]
// CHECK:STDOUT:   %tuple.type.953: type = tuple_type (%empty_struct_type, %empty_struct_type, %ptr.fb6) [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.f1c: <witness> = impl_witness imports.%Copy.impl_witness_table.67d, @ptr.as.Copy.impl(%Z) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.31e: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%Z) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.525: %ptr.as.Copy.impl.Op.type.31e = struct_value () [concrete]
// CHECK:STDOUT:   %.f24: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%Z) [concrete]
// CHECK:STDOUT:   %Copy.facet.347: %Copy.type = facet_value %ptr.fb6, (%Copy.impl_witness.f1c) [concrete]
// CHECK:STDOUT:   %.90e: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.347 [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.525, @ptr.as.Copy.impl.Op(%Z) [concrete]
// CHECK:STDOUT:   %tuple.349: %tuple.type.cfb = tuple_value (%Destroy.facet.08b, %A.facet.8b3, %ptr.fb6) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .BitAndWith = %Core.BitAndWith
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.5c4: @DestroyT.binding.as_type.as.Destroy.impl.%DestroyT.binding.as_type.as.Destroy.impl.Op.type (%DestroyT.binding.as_type.as.Destroy.impl.Op.type.15c) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @DestroyT.binding.as_type.as.Destroy.impl.%DestroyT.binding.as_type.as.Destroy.impl.Op (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.fd5)]
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (%Core.import_ref.5c4), @DestroyT.binding.as_type.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.659: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.75bcbe.1) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.692ac2.1)]
// CHECK:STDOUT:   %Copy.impl_witness_table.67d = impl_witness_table (%Core.import_ref.659), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.BitAndWith: %BitAndWith.type.f2e = import_ref Core//prelude/parts/as, BitAndWith, loaded [concrete = constants.%BitAndWith.generic]
// CHECK:STDOUT:   %Core.import_ref.636: %type.as.BitAndWith.impl.Op.type = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, loaded [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:   %BitAndWith.impl_witness_table = impl_witness_table (%Core.import_ref.636), @type.as.BitAndWith.impl [concrete]
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
// CHECK:STDOUT:   %A.decl: %A.type.495 = interface_decl @A [concrete = constants.%A.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.3ab = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5: type = splice_block %Destroy.ref [concrete = constants.%Destroy.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc5_13.2: %Destroy.type = symbolic_binding T, 0 [symbolic = %T.loc5_13.1 (constants.%T.afb)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.decl: type = class_decl @X [concrete = constants.%X] {} {}
// CHECK:STDOUT:   %Y.decl: type = class_decl @Y [concrete = constants.%Y] {} {}
// CHECK:STDOUT:   %Z.decl: type = class_decl @Z [concrete = constants.%Z] {} {}
// CHECK:STDOUT:   impl_decl @Y.as.A.impl [concrete] {} {
// CHECK:STDOUT:     %Y.ref: type = name_ref Y, file.%Y.decl [concrete = constants.%Y]
// CHECK:STDOUT:     %A.ref: %A.type.495 = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %Destroy.facet: %Destroy.type = facet_value %X.ref, (constants.%Destroy.impl_witness.0eb) [concrete = constants.%Destroy.facet.08b]
// CHECK:STDOUT:     %.loc13: %Destroy.type = converted %X.ref, %Destroy.facet [concrete = constants.%Destroy.facet.08b]
// CHECK:STDOUT:     %A.type: type = facet_type <@A, @A(constants.%Destroy.facet.08b)> [concrete = constants.%A.type.c3a]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.impl_witness_table = impl_witness_table (@Y.as.A.impl.%Y.as.A.impl.F.decl), @Y.as.A.impl [concrete]
// CHECK:STDOUT:   %A.impl_witness: <witness> = impl_witness %A.impl_witness_table [concrete = constants.%A.impl_witness]
// CHECK:STDOUT:   %Call.decl: %Call.type = fn_decl @Call [concrete = constants.%Call] {} {}
// CHECK:STDOUT:   %CallGeneric.decl: %CallGeneric.type = fn_decl @CallGeneric [concrete = constants.%CallGeneric] {
// CHECK:STDOUT:     %T.patt: %pattern_type.c82 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc26_25.1: type = splice_block %.loc26_25.3 [concrete = constants.%facet_type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %A.ref: %A.type.495 = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:       %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:       %Destroy.facet.loc26: %Destroy.type = facet_value %X.ref, (constants.%Destroy.impl_witness.0eb) [concrete = constants.%Destroy.facet.08b]
// CHECK:STDOUT:       %.loc26_23: %Destroy.type = converted %X.ref, %Destroy.facet.loc26 [concrete = constants.%Destroy.facet.08b]
// CHECK:STDOUT:       %A.type: type = facet_type <@A, @A(constants.%Destroy.facet.08b)> [concrete = constants.%A.type.c3a]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:       %impl.elem0.loc26: %.fa7 = impl_witness_access constants.%BitAndWith.impl_witness, element0 [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:       %bound_method.loc26: <bound method> = bound_method %A.type, %impl.elem0.loc26 [concrete = constants.%type.as.BitAndWith.impl.Op.bound]
// CHECK:STDOUT:       %type.as.BitAndWith.impl.Op.call: init type = call %bound_method.loc26(%A.type, %Destroy.ref) [concrete = constants.%facet_type]
// CHECK:STDOUT:       %.loc26_25.2: type = value_of_initializer %type.as.BitAndWith.impl.Op.call [concrete = constants.%facet_type]
// CHECK:STDOUT:       %.loc26_25.3: type = converted %type.as.BitAndWith.impl.Op.call, %.loc26_25.2 [concrete = constants.%facet_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc26_16.2: %facet_type = symbolic_binding T, 0 [symbolic = %T.loc26_16.1 (constants.%T.c3e)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallIndirect.decl: %CallIndirect.type = fn_decl @CallIndirect [concrete = constants.%CallIndirect] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @A(%T.loc5_13.2: %Destroy.type) {
// CHECK:STDOUT:   %T.loc5_13.1: %Destroy.type = symbolic_binding T, 0 [symbolic = %T.loc5_13.1 (constants.%T.afb)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.type: type = facet_type <@A, @A(%T.loc5_13.1)> [symbolic = %A.type (constants.%A.type.e04)]
// CHECK:STDOUT:   %Self.loc5_31.2: @A.%A.type (%A.type.e04) = symbolic_binding Self, 1 [symbolic = %Self.loc5_31.2 (constants.%Self.0c7)]
// CHECK:STDOUT:   %A.F.type: type = fn_type @A.F, @A(%T.loc5_13.1) [symbolic = %A.F.type (constants.%A.F.type.4bf)]
// CHECK:STDOUT:   %A.F: @A.%A.F.type (%A.F.type.4bf) = struct_value () [symbolic = %A.F (constants.%A.F.186)]
// CHECK:STDOUT:   %A.assoc_type: type = assoc_entity_type @A, @A(%T.loc5_13.1) [symbolic = %A.assoc_type (constants.%A.assoc_type.baa)]
// CHECK:STDOUT:   %assoc0.loc6_41.2: @A.%A.assoc_type (%A.assoc_type.baa) = assoc_entity element0, %A.F.decl [symbolic = %assoc0.loc6_41.2 (constants.%assoc0.030)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc5_31.1: @A.%A.type (%A.type.e04) = symbolic_binding Self, 1 [symbolic = %Self.loc5_31.2 (constants.%Self.0c7)]
// CHECK:STDOUT:     %A.F.decl: @A.%A.F.type (%A.F.type.4bf) = fn_decl @A.F [symbolic = @A.%A.F (constants.%A.F.186)] {
// CHECK:STDOUT:       %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 2 [concrete]
// CHECK:STDOUT:       %u.patt: @A.F.%pattern_type.loc6_18 (%pattern_type.854) = value_binding_pattern u [concrete]
// CHECK:STDOUT:       %u.param_patt: @A.F.%pattern_type.loc6_18 (%pattern_type.854) = value_param_pattern %u.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @A.F.%pattern_type.loc6_25 (%pattern_type.dd1) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @A.F.%pattern_type.loc6_25 (%pattern_type.dd1) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: %Destroy.type = name_ref T, @A.%T.loc5_13.2 [symbolic = %T (constants.%T.afb)]
// CHECK:STDOUT:       %.loc6_32: @A.F.%A.type (%A.type.e04) = specific_constant @A.%Self.loc5_31.1, @A(constants.%T.afb) [symbolic = %Self (constants.%Self.0c7)]
// CHECK:STDOUT:       %Self.ref: @A.F.%A.type (%A.type.e04) = name_ref Self, %.loc6_32 [symbolic = %Self (constants.%Self.0c7)]
// CHECK:STDOUT:       %U.ref.loc6_38: type = name_ref U, %U.loc6_8.2 [symbolic = %U.loc6_8.1 (constants.%U.63a)]
// CHECK:STDOUT:       %ptr.loc6_39: type = ptr_type %U.ref.loc6_38 [symbolic = %ptr.loc6_22.1 (constants.%ptr.e49)]
// CHECK:STDOUT:       %.loc6_40.1: @A.F.%tuple.type.loc6_40.1 (%tuple.type.337) = tuple_literal (%T.ref, %Self.ref, %ptr.loc6_39) [symbolic = %tuple (constants.%tuple.3bc)]
// CHECK:STDOUT:       %T.as_type: type = facet_access_type constants.%T.afb [symbolic = %T.binding.as_type (constants.%T.binding.as_type.d87)]
// CHECK:STDOUT:       %.loc6_40.2: type = converted constants.%T.afb, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type.d87)]
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type constants.%Self.0c7 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.a47)]
// CHECK:STDOUT:       %.loc6_40.3: type = converted constants.%Self.0c7, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.a47)]
// CHECK:STDOUT:       %.loc6_40.4: type = converted %.loc6_40.1, constants.%tuple.type.132 [symbolic = %tuple.type.loc6_40.2 (constants.%tuple.type.132)]
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %U.loc6_8.2: type = symbolic_binding U, 2 [symbolic = %U.loc6_8.1 (constants.%U.63a)]
// CHECK:STDOUT:       %u.param: @A.F.%ptr.loc6_22.1 (%ptr.e49) = value_param call_param0
// CHECK:STDOUT:       %.loc6_22: type = splice_block %ptr.loc6_22.2 [symbolic = %ptr.loc6_22.1 (constants.%ptr.e49)] {
// CHECK:STDOUT:         %U.ref.loc6_21: type = name_ref U, %U.loc6_8.2 [symbolic = %U.loc6_8.1 (constants.%U.63a)]
// CHECK:STDOUT:         %ptr.loc6_22.2: type = ptr_type %U.ref.loc6_21 [symbolic = %ptr.loc6_22.1 (constants.%ptr.e49)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %u: @A.F.%ptr.loc6_22.1 (%ptr.e49) = value_binding u, %u.param
// CHECK:STDOUT:       %return.param: ref @A.F.%tuple.type.loc6_40.2 (%tuple.type.132) = out_param call_param1
// CHECK:STDOUT:       %return: ref @A.F.%tuple.type.loc6_40.2 (%tuple.type.132) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc6_41.1: @A.%A.assoc_type (%A.assoc_type.baa) = assoc_entity element0, %A.F.decl [symbolic = %assoc0.loc6_41.2 (constants.%assoc0.030)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc5_31.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .F = %assoc0.loc6_41.1
// CHECK:STDOUT:     witness = (%A.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Y.as.A.impl: %Y.ref as %A.type {
// CHECK:STDOUT:   %Y.as.A.impl.F.decl: %Y.as.A.impl.F.type = fn_decl @Y.as.A.impl.F [concrete = constants.%Y.as.A.impl.F] {
// CHECK:STDOUT:     %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 0 [concrete]
// CHECK:STDOUT:     %u.patt: @Y.as.A.impl.F.%pattern_type.loc16_18 (%pattern_type.a603a0.1) = value_binding_pattern u [concrete]
// CHECK:STDOUT:     %u.param_patt: @Y.as.A.impl.F.%pattern_type.loc16_18 (%pattern_type.a603a0.1) = value_param_pattern %u.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @Y.as.A.impl.F.%pattern_type.loc16_25 (%pattern_type.e0b) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @Y.as.A.impl.F.%pattern_type.loc16_25 (%pattern_type.e0b) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %Y.ref: type = name_ref Y, file.%Y.decl [concrete = constants.%Y]
// CHECK:STDOUT:     %U.ref.loc16_35: type = name_ref U, %U.loc16_8.2 [symbolic = %U.loc16_8.1 (constants.%U.d9f)]
// CHECK:STDOUT:     %ptr.loc16_36: type = ptr_type %U.ref.loc16_35 [symbolic = %ptr.loc16_22.1 (constants.%ptr.4f0b5c.1)]
// CHECK:STDOUT:     %.loc16_37.1: %tuple.type.ff9 = tuple_literal (%X.ref, %Y.ref, %ptr.loc16_36) [symbolic = %tuple (constants.%tuple.bcf)]
// CHECK:STDOUT:     %.loc16_37.2: type = converted %.loc16_37.1, constants.%tuple.type.576 [symbolic = %tuple.type.loc16 (constants.%tuple.type.576)]
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %U.loc16_8.2: type = symbolic_binding U, 0 [symbolic = %U.loc16_8.1 (constants.%U.d9f)]
// CHECK:STDOUT:     %u.param: @Y.as.A.impl.F.%ptr.loc16_22.1 (%ptr.4f0b5c.1) = value_param call_param0
// CHECK:STDOUT:     %.loc16_22: type = splice_block %ptr.loc16_22.2 [symbolic = %ptr.loc16_22.1 (constants.%ptr.4f0b5c.1)] {
// CHECK:STDOUT:       %U.ref.loc16_21: type = name_ref U, %U.loc16_8.2 [symbolic = %U.loc16_8.1 (constants.%U.d9f)]
// CHECK:STDOUT:       %ptr.loc16_22.2: type = ptr_type %U.ref.loc16_21 [symbolic = %ptr.loc16_22.1 (constants.%ptr.4f0b5c.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: @Y.as.A.impl.F.%ptr.loc16_22.1 (%ptr.4f0b5c.1) = value_binding u, %u.param
// CHECK:STDOUT:     %return.param: ref @Y.as.A.impl.F.%tuple.type.loc16 (%tuple.type.576) = out_param call_param1
// CHECK:STDOUT:     %return: ref @Y.as.A.impl.F.%tuple.type.loc16 (%tuple.type.576) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .X = <poisoned>
// CHECK:STDOUT:   .Y = <poisoned>
// CHECK:STDOUT:   .F = %Y.as.A.impl.F.decl
// CHECK:STDOUT:   witness = file.%A.impl_witness
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
// CHECK:STDOUT: generic fn @A.F(@A.%T.loc5_13.2: %Destroy.type, @A.%Self.loc5_31.1: @A.%A.type (%A.type.e04), %U.loc6_8.2: type) {
// CHECK:STDOUT:   %U.loc6_8.1: type = symbolic_binding U, 2 [symbolic = %U.loc6_8.1 (constants.%U.63a)]
// CHECK:STDOUT:   %ptr.loc6_22.1: type = ptr_type %U.loc6_8.1 [symbolic = %ptr.loc6_22.1 (constants.%ptr.e49)]
// CHECK:STDOUT:   %pattern_type.loc6_18: type = pattern_type %ptr.loc6_22.1 [symbolic = %pattern_type.loc6_18 (constants.%pattern_type.854)]
// CHECK:STDOUT:   %T: %Destroy.type = symbolic_binding T, 0 [symbolic = %T (constants.%T.afb)]
// CHECK:STDOUT:   %A.type: type = facet_type <@A, @A(%T)> [symbolic = %A.type (constants.%A.type.e04)]
// CHECK:STDOUT:   %Self: @A.F.%A.type (%A.type.e04) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.0c7)]
// CHECK:STDOUT:   %tuple.type.loc6_40.1: type = tuple_type (constants.%Destroy.type, %A.type, type) [symbolic = %tuple.type.loc6_40.1 (constants.%tuple.type.337)]
// CHECK:STDOUT:   %tuple: @A.F.%tuple.type.loc6_40.1 (%tuple.type.337) = tuple_value (%T, %Self, %ptr.loc6_22.1) [symbolic = %tuple (constants.%tuple.3bc)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T [symbolic = %T.binding.as_type (constants.%T.binding.as_type.d87)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.a47)]
// CHECK:STDOUT:   %tuple.type.loc6_40.2: type = tuple_type (%T.binding.as_type, %Self.binding.as_type, %ptr.loc6_22.1) [symbolic = %tuple.type.loc6_40.2 (constants.%tuple.type.132)]
// CHECK:STDOUT:   %pattern_type.loc6_25: type = pattern_type %tuple.type.loc6_40.2 [symbolic = %pattern_type.loc6_25 (constants.%pattern_type.dd1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param: @A.F.%ptr.loc6_22.1 (%ptr.e49)) -> @A.F.%tuple.type.loc6_40.2 (%tuple.type.132);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Y.as.A.impl.F(%U.loc16_8.2: type) {
// CHECK:STDOUT:   %U.loc16_8.1: type = symbolic_binding U, 0 [symbolic = %U.loc16_8.1 (constants.%U.d9f)]
// CHECK:STDOUT:   %ptr.loc16_22.1: type = ptr_type %U.loc16_8.1 [symbolic = %ptr.loc16_22.1 (constants.%ptr.4f0b5c.1)]
// CHECK:STDOUT:   %pattern_type.loc16_18: type = pattern_type %ptr.loc16_22.1 [symbolic = %pattern_type.loc16_18 (constants.%pattern_type.a603a0.1)]
// CHECK:STDOUT:   %tuple: %tuple.type.ff9 = tuple_value (constants.%X, constants.%Y, %ptr.loc16_22.1) [symbolic = %tuple (constants.%tuple.bcf)]
// CHECK:STDOUT:   %tuple.type.loc16: type = tuple_type (constants.%X, constants.%Y, %ptr.loc16_22.1) [symbolic = %tuple.type.loc16 (constants.%tuple.type.576)]
// CHECK:STDOUT:   %pattern_type.loc16_25: type = pattern_type %tuple.type.loc16 [symbolic = %pattern_type.loc16_25 (constants.%pattern_type.e0b)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc16_25: <witness> = require_complete_type %tuple.type.loc16 [symbolic = %require_complete.loc16_25 (constants.%require_complete.dc0)]
// CHECK:STDOUT:   %require_complete.loc16_19: <witness> = require_complete_type %ptr.loc16_22.1 [symbolic = %require_complete.loc16_19 (constants.%require_complete.482d3f.1)]
// CHECK:STDOUT:   %tuple.type.loc17: type = tuple_type (constants.%empty_struct_type, constants.%empty_struct_type, %ptr.loc16_22.1) [symbolic = %tuple.type.loc17 (constants.%tuple.type.f88)]
// CHECK:STDOUT:   %.loc17_21.2: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%U.loc16_8.1) [symbolic = %.loc17_21.2 (constants.%.dcd)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr.loc16_22.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.c66)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.loc16_22.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet.747)]
// CHECK:STDOUT:   %.loc17_21.3: type = fn_type_with_self_type constants.%Copy.Op.type, %Copy.facet [symbolic = %.loc17_21.3 (constants.%.89d)]
// CHECK:STDOUT:   %impl.elem0.loc17_21.2: @Y.as.A.impl.F.%.loc17_21.3 (%.89d) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc17_21.2 (constants.%impl.elem0.928)]
// CHECK:STDOUT:   %specific_impl_fn.loc17_21.2: <specific function> = specific_impl_function %impl.elem0.loc17_21.2, @Copy.Op(%Copy.facet) [symbolic = %specific_impl_fn.loc17_21.2 (constants.%specific_impl_fn.6a1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param: @Y.as.A.impl.F.%ptr.loc16_22.1 (%ptr.4f0b5c.1)) -> %return.param: @Y.as.A.impl.F.%tuple.type.loc16 (%tuple.type.576) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc17_14.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc17_18.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %u.ref: @Y.as.A.impl.F.%ptr.loc16_22.1 (%ptr.4f0b5c.1) = name_ref u, %u
// CHECK:STDOUT:     %.loc17_22.1: @Y.as.A.impl.F.%tuple.type.loc17 (%tuple.type.f88) = tuple_literal (%.loc17_14.1, %.loc17_18.1, %u.ref)
// CHECK:STDOUT:     %tuple.elem0: ref %X = tuple_access %return, element0
// CHECK:STDOUT:     %.loc17_14.2: init %X = class_init (), %tuple.elem0 [concrete = constants.%X.val]
// CHECK:STDOUT:     %.loc17_22.2: init %X = converted %.loc17_14.1, %.loc17_14.2 [concrete = constants.%X.val]
// CHECK:STDOUT:     %tuple.elem1: ref %Y = tuple_access %return, element1
// CHECK:STDOUT:     %.loc17_18.2: init %Y = class_init (), %tuple.elem1 [concrete = constants.%Y.val]
// CHECK:STDOUT:     %.loc17_22.3: init %Y = converted %.loc17_18.1, %.loc17_18.2 [concrete = constants.%Y.val]
// CHECK:STDOUT:     %impl.elem0.loc17_21.1: @Y.as.A.impl.F.%.loc17_21.3 (%.89d) = impl_witness_access constants.%Copy.lookup_impl_witness.c66, element0 [symbolic = %impl.elem0.loc17_21.2 (constants.%impl.elem0.928)]
// CHECK:STDOUT:     %bound_method.loc17_21.1: <bound method> = bound_method %u.ref, %impl.elem0.loc17_21.1
// CHECK:STDOUT:     %specific_impl_fn.loc17_21.1: <specific function> = specific_impl_function %impl.elem0.loc17_21.1, @Copy.Op(constants.%Copy.facet.747) [symbolic = %specific_impl_fn.loc17_21.2 (constants.%specific_impl_fn.6a1)]
// CHECK:STDOUT:     %bound_method.loc17_21.2: <bound method> = bound_method %u.ref, %specific_impl_fn.loc17_21.1
// CHECK:STDOUT:     %.loc17_21.1: init @Y.as.A.impl.F.%ptr.loc16_22.1 (%ptr.4f0b5c.1) = call %bound_method.loc17_21.2(%u.ref)
// CHECK:STDOUT:     %tuple.elem2: ref @Y.as.A.impl.F.%ptr.loc16_22.1 (%ptr.4f0b5c.1) = tuple_access %return, element2
// CHECK:STDOUT:     %.loc17_22.4: init @Y.as.A.impl.F.%ptr.loc16_22.1 (%ptr.4f0b5c.1) = initialize_from %.loc17_21.1 to %tuple.elem2
// CHECK:STDOUT:     %.loc17_22.5: init @Y.as.A.impl.F.%tuple.type.loc16 (%tuple.type.576) = tuple_init (%.loc17_22.2, %.loc17_22.3, %.loc17_22.4) to %return
// CHECK:STDOUT:     %.loc17_23: init @Y.as.A.impl.F.%tuple.type.loc16 (%tuple.type.576) = converted %.loc17_22.1, %.loc17_22.5
// CHECK:STDOUT:     return %.loc17_23 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %u.patt: %pattern_type.8f9 = ref_binding_pattern u [concrete]
// CHECK:STDOUT:     %u.var_patt: %pattern_type.8f9 = var_pattern %u.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %u.var: ref %Z = var %u.var_patt
// CHECK:STDOUT:   %Z.ref.loc22: type = name_ref Z, file.%Z.decl [concrete = constants.%Z]
// CHECK:STDOUT:   %u: ref %Z = ref_binding u, %u.var
// CHECK:STDOUT:   %Y.ref: type = name_ref Y, file.%Y.decl [concrete = constants.%Y]
// CHECK:STDOUT:   %A.ref: %A.type.495 = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:   %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %X.ref, (constants.%Destroy.impl_witness.0eb) [concrete = constants.%Destroy.facet.08b]
// CHECK:STDOUT:   %.loc23_12: %Destroy.type = converted %X.ref, %Destroy.facet [concrete = constants.%Destroy.facet.08b]
// CHECK:STDOUT:   %A.type: type = facet_type <@A, @A(constants.%Destroy.facet.08b)> [concrete = constants.%A.type.c3a]
// CHECK:STDOUT:   %A.facet: %A.type.c3a = facet_value %Y.ref, (constants.%A.impl_witness) [concrete = constants.%A.facet.8b3]
// CHECK:STDOUT:   %.loc23_6: %A.type.c3a = converted %Y.ref, %A.facet [concrete = constants.%A.facet.8b3]
// CHECK:STDOUT:   %.loc23_14.1: %A.assoc_type.0c3 = specific_constant @A.%assoc0.loc6_41.1, @A(constants.%Destroy.facet.08b) [concrete = constants.%assoc0.43e]
// CHECK:STDOUT:   %F.ref: %A.assoc_type.0c3 = name_ref F, %.loc23_14.1 [concrete = constants.%assoc0.43e]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc23_6 [concrete = constants.%Y]
// CHECK:STDOUT:   %.loc23_14.2: type = converted %.loc23_6, %as_type [concrete = constants.%Y]
// CHECK:STDOUT:   %impl.elem0: %.e73 = impl_witness_access constants.%A.impl_witness, element0 [concrete = constants.%Y.as.A.impl.F]
// CHECK:STDOUT:   %Z.ref.loc23: type = name_ref Z, file.%Z.decl [concrete = constants.%Z]
// CHECK:STDOUT:   %u.ref: ref %Z = name_ref u, %u
// CHECK:STDOUT:   %addr: %ptr.fb6 = addr_of %u.ref
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Y.as.A.impl.F(constants.%Z) [concrete = constants.%Y.as.A.impl.F.specific_fn]
// CHECK:STDOUT:   %.loc23_22.1: ref %tuple.type.092 = temporary_storage
// CHECK:STDOUT:   %Y.as.A.impl.F.call: init %tuple.type.092 = call %specific_fn(%addr) to %.loc23_22.1
// CHECK:STDOUT:   %.loc23_22.2: ref %tuple.type.092 = temporary %.loc23_22.1, %Y.as.A.impl.F.call
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc23: <bound method> = bound_method %.loc23_22.2, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.056
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.056, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.352) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.7b8]
// CHECK:STDOUT:   %bound_method.loc23: <bound method> = bound_method %.loc23_22.2, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc23: init %empty_tuple.type = call %bound_method.loc23(%.loc23_22.2)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc22: <bound method> = bound_method %u.var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.2cc
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.2: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.2cc, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.4e3) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.cb3]
// CHECK:STDOUT:   %bound_method.loc22: <bound method> = bound_method %u.var, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc22: init %empty_tuple.type = call %bound_method.loc22(%u.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallGeneric(%T.loc26_16.2: %facet_type) {
// CHECK:STDOUT:   %T.loc26_16.1: %facet_type = symbolic_binding T, 0 [symbolic = %T.loc26_16.1 (constants.%T.c3e)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc26_16.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.7fc)]
// CHECK:STDOUT:   %A.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc26_16.1, @A, @A(constants.%Destroy.facet.08b) [symbolic = %A.lookup_impl_witness (constants.%A.lookup_impl_witness)]
// CHECK:STDOUT:   %A.facet: %A.type.c3a = facet_value %T.binding.as_type, (%A.lookup_impl_witness) [symbolic = %A.facet (constants.%A.facet.baa)]
// CHECK:STDOUT:   %.loc28_4.3: type = fn_type_with_self_type constants.%A.F.type.1cf, %A.facet [symbolic = %.loc28_4.3 (constants.%.dc6)]
// CHECK:STDOUT:   %impl.elem0.loc28_4.2: @CallGeneric.%.loc28_4.3 (%.dc6) = impl_witness_access %A.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc28_4.2 (constants.%impl.elem0.899)]
// CHECK:STDOUT:   %specific_impl_fn.loc28_4.2: <specific function> = specific_impl_function %impl.elem0.loc28_4.2, @A.F(constants.%Destroy.facet.08b, %A.facet, constants.%Z) [symbolic = %specific_impl_fn.loc28_4.2 (constants.%specific_impl_fn.bf2)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (constants.%X, %T.binding.as_type, constants.%ptr.fb6) [symbolic = %tuple.type (constants.%tuple.type.063)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %tuple.type [symbolic = %require_complete (constants.%require_complete.379)]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %tuple.type, () [symbolic = %facet_value (constants.%facet_value.94e)]
// CHECK:STDOUT:   %.loc28_12.4: require_specific_def_type = require_specific_def @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [symbolic = %.loc28_12.4 (constants.%.bd7)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness imports.%Destroy.impl_witness_table, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness.ff5)]
// CHECK:STDOUT:   %Destroy.facet.loc28: %Destroy.type = facet_value %tuple.type, (%Destroy.impl_witness) [symbolic = %Destroy.facet.loc28 (constants.%Destroy.facet.62b)]
// CHECK:STDOUT:   %.loc28_12.5: type = fn_type_with_self_type constants.%Destroy.Op.type, %Destroy.facet.loc28 [symbolic = %.loc28_12.5 (constants.%.571)]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op.type (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.type.99f)]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op: @CallGeneric.%DestroyT.binding.as_type.as.Destroy.impl.Op.type (%DestroyT.binding.as_type.as.Destroy.impl.Op.type.99f) = struct_value () [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.f7e)]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.loc28: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value) [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.loc28 (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.189)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %u.patt: %pattern_type.8f9 = ref_binding_pattern u [concrete]
// CHECK:STDOUT:       %u.var_patt: %pattern_type.8f9 = var_pattern %u.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u.var: ref %Z = var %u.var_patt
// CHECK:STDOUT:     %Z.ref.loc27: type = name_ref Z, file.%Z.decl [concrete = constants.%Z]
// CHECK:STDOUT:     %u: ref %Z = ref_binding u, %u.var
// CHECK:STDOUT:     %T.ref: %facet_type = name_ref T, %T.loc26_16.2 [symbolic = %T.loc26_16.1 (constants.%T.c3e)]
// CHECK:STDOUT:     %.loc28_4.1: %A.assoc_type.0c3 = specific_constant @A.%assoc0.loc6_41.1, @A(constants.%Destroy.facet.08b) [concrete = constants.%assoc0.43e]
// CHECK:STDOUT:     %F.ref: %A.assoc_type.0c3 = name_ref F, %.loc28_4.1 [concrete = constants.%assoc0.43e]
// CHECK:STDOUT:     %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type.7fc)]
// CHECK:STDOUT:     %.loc28_4.2: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type.7fc)]
// CHECK:STDOUT:     %impl.elem0.loc28_4.1: @CallGeneric.%.loc28_4.3 (%.dc6) = impl_witness_access constants.%A.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc28_4.2 (constants.%impl.elem0.899)]
// CHECK:STDOUT:     %Z.ref.loc28: type = name_ref Z, file.%Z.decl [concrete = constants.%Z]
// CHECK:STDOUT:     %u.ref: ref %Z = name_ref u, %u
// CHECK:STDOUT:     %addr: %ptr.fb6 = addr_of %u.ref
// CHECK:STDOUT:     %specific_impl_fn.loc28_4.1: <specific function> = specific_impl_function %impl.elem0.loc28_4.1, @A.F(constants.%Destroy.facet.08b, constants.%A.facet.baa, constants.%Z) [symbolic = %specific_impl_fn.loc28_4.2 (constants.%specific_impl_fn.bf2)]
// CHECK:STDOUT:     %.loc28_12.1: ref @CallGeneric.%tuple.type (%tuple.type.063) = temporary_storage
// CHECK:STDOUT:     %.loc28_12.2: init @CallGeneric.%tuple.type (%tuple.type.063) = call %specific_impl_fn.loc28_4.1(%addr) to %.loc28_12.1
// CHECK:STDOUT:     %.loc28_12.3: ref @CallGeneric.%tuple.type (%tuple.type.063) = temporary %.loc28_12.1, %.loc28_12.2
// CHECK:STDOUT:     %impl.elem0.loc28_12: @CallGeneric.%.loc28_12.5 (%.571) = impl_witness_access constants.%Destroy.impl_witness.ff5, element0 [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.f7e)]
// CHECK:STDOUT:     %bound_method.loc28_12.1: <bound method> = bound_method %.loc28_12.3, %impl.elem0.loc28_12
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0.loc28_12, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.94e) [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.loc28 (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.189)]
// CHECK:STDOUT:     %bound_method.loc28_12.2: <bound method> = bound_method %.loc28_12.3, %specific_fn
// CHECK:STDOUT:     %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc28: init %empty_tuple.type = call %bound_method.loc28_12.2(%.loc28_12.3)
// CHECK:STDOUT:     %DestroyT.binding.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %u.var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.2cc
// CHECK:STDOUT:     %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.2cc, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.4e3) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.cb3]
// CHECK:STDOUT:     %bound_method.loc27: <bound method> = bound_method %u.var, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:     %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc27: init %empty_tuple.type = call %bound_method.loc27(%u.var)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallIndirect() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CallGeneric.ref: %CallGeneric.type = name_ref CallGeneric, file.%CallGeneric.decl [concrete = constants.%CallGeneric]
// CHECK:STDOUT:   %Y.ref: type = name_ref Y, file.%Y.decl [concrete = constants.%Y]
// CHECK:STDOUT:   %facet_value: %facet_type = facet_value %Y.ref, (constants.%Destroy.impl_witness.688, constants.%A.impl_witness) [concrete = constants.%facet_value.312]
// CHECK:STDOUT:   %.loc32: %facet_type = converted %Y.ref, %facet_value [concrete = constants.%facet_value.312]
// CHECK:STDOUT:   %CallGeneric.specific_fn: <specific function> = specific_function %CallGeneric.ref, @CallGeneric(constants.%facet_value.312) [concrete = constants.%CallGeneric.specific_fn]
// CHECK:STDOUT:   %CallGeneric.call: init %empty_tuple.type = call %CallGeneric.specific_fn()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%T.afb) {
// CHECK:STDOUT:   %T.loc5_13.1 => constants.%T.afb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.F(constants.%T.afb, constants.%Self.0c7, constants.%U.63a) {
// CHECK:STDOUT:   %U.loc6_8.1 => constants.%U.63a
// CHECK:STDOUT:   %ptr.loc6_22.1 => constants.%ptr.e49
// CHECK:STDOUT:   %pattern_type.loc6_18 => constants.%pattern_type.854
// CHECK:STDOUT:   %T => constants.%T.afb
// CHECK:STDOUT:   %A.type => constants.%A.type.e04
// CHECK:STDOUT:   %Self => constants.%Self.0c7
// CHECK:STDOUT:   %tuple.type.loc6_40.1 => constants.%tuple.type.337
// CHECK:STDOUT:   %tuple => constants.%tuple.3bc
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type.d87
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.a47
// CHECK:STDOUT:   %tuple.type.loc6_40.2 => constants.%tuple.type.132
// CHECK:STDOUT:   %pattern_type.loc6_25 => constants.%pattern_type.dd1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%Destroy.facet.08b) {
// CHECK:STDOUT:   %T.loc5_13.1 => constants.%Destroy.facet.08b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.type => constants.%A.type.c3a
// CHECK:STDOUT:   %Self.loc5_31.2 => constants.%Self.369
// CHECK:STDOUT:   %A.F.type => constants.%A.F.type.1cf
// CHECK:STDOUT:   %A.F => constants.%A.F.d66
// CHECK:STDOUT:   %A.assoc_type => constants.%A.assoc_type.0c3
// CHECK:STDOUT:   %assoc0.loc6_41.2 => constants.%assoc0.43e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y.as.A.impl.F(constants.%U.d9f) {
// CHECK:STDOUT:   %U.loc16_8.1 => constants.%U.d9f
// CHECK:STDOUT:   %ptr.loc16_22.1 => constants.%ptr.4f0b5c.1
// CHECK:STDOUT:   %pattern_type.loc16_18 => constants.%pattern_type.a603a0.1
// CHECK:STDOUT:   %tuple => constants.%tuple.bcf
// CHECK:STDOUT:   %tuple.type.loc16 => constants.%tuple.type.576
// CHECK:STDOUT:   %pattern_type.loc16_25 => constants.%pattern_type.e0b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.F(constants.%Destroy.facet.08b, constants.%A.facet.8b3, constants.%U.d9f) {
// CHECK:STDOUT:   %U.loc6_8.1 => constants.%U.d9f
// CHECK:STDOUT:   %ptr.loc6_22.1 => constants.%ptr.4f0b5c.1
// CHECK:STDOUT:   %pattern_type.loc6_18 => constants.%pattern_type.a603a0.1
// CHECK:STDOUT:   %T => constants.%Destroy.facet.08b
// CHECK:STDOUT:   %A.type => constants.%A.type.c3a
// CHECK:STDOUT:   %Self => constants.%A.facet.8b3
// CHECK:STDOUT:   %tuple.type.loc6_40.1 => constants.%tuple.type.cfb
// CHECK:STDOUT:   %tuple => constants.%tuple.1b3
// CHECK:STDOUT:   %T.binding.as_type => constants.%X
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Y
// CHECK:STDOUT:   %tuple.type.loc6_40.2 => constants.%tuple.type.576
// CHECK:STDOUT:   %pattern_type.loc6_25 => constants.%pattern_type.e0b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y.as.A.impl.F(constants.%Z) {
// CHECK:STDOUT:   %U.loc16_8.1 => constants.%Z
// CHECK:STDOUT:   %ptr.loc16_22.1 => constants.%ptr.fb6
// CHECK:STDOUT:   %pattern_type.loc16_18 => constants.%pattern_type.f76
// CHECK:STDOUT:   %tuple => constants.%tuple.f10
// CHECK:STDOUT:   %tuple.type.loc16 => constants.%tuple.type.092
// CHECK:STDOUT:   %pattern_type.loc16_25 => constants.%pattern_type.48d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc16_25 => constants.%complete_type.05d
// CHECK:STDOUT:   %require_complete.loc16_19 => constants.%complete_type.d3e
// CHECK:STDOUT:   %tuple.type.loc17 => constants.%tuple.type.953
// CHECK:STDOUT:   %.loc17_21.2 => constants.%.f24
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.impl_witness.f1c
// CHECK:STDOUT:   %Copy.facet => constants.%Copy.facet.347
// CHECK:STDOUT:   %.loc17_21.3 => constants.%.90e
// CHECK:STDOUT:   %impl.elem0.loc17_21.2 => constants.%ptr.as.Copy.impl.Op.525
// CHECK:STDOUT:   %specific_impl_fn.loc17_21.2 => constants.%ptr.as.Copy.impl.Op.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGeneric(constants.%T.c3e) {
// CHECK:STDOUT:   %T.loc26_16.1 => constants.%T.c3e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.F(constants.%Destroy.facet.08b, constants.%A.facet.baa, constants.%Z) {
// CHECK:STDOUT:   %U.loc6_8.1 => constants.%Z
// CHECK:STDOUT:   %ptr.loc6_22.1 => constants.%ptr.fb6
// CHECK:STDOUT:   %pattern_type.loc6_18 => constants.%pattern_type.f76
// CHECK:STDOUT:   %T => constants.%Destroy.facet.08b
// CHECK:STDOUT:   %A.type => constants.%A.type.c3a
// CHECK:STDOUT:   %Self => constants.%A.facet.baa
// CHECK:STDOUT:   %tuple.type.loc6_40.1 => constants.%tuple.type.cfb
// CHECK:STDOUT:   %tuple => constants.%tuple.9ef
// CHECK:STDOUT:   %T.binding.as_type => constants.%X
// CHECK:STDOUT:   %Self.binding.as_type => constants.%T.binding.as_type.7fc
// CHECK:STDOUT:   %tuple.type.loc6_40.2 => constants.%tuple.type.063
// CHECK:STDOUT:   %pattern_type.loc6_25 => constants.%pattern_type.150
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGeneric(constants.%facet_value.312) {
// CHECK:STDOUT:   %T.loc26_16.1 => constants.%facet_value.312
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.binding.as_type => constants.%Y
// CHECK:STDOUT:   %A.lookup_impl_witness => constants.%A.impl_witness
// CHECK:STDOUT:   %A.facet => constants.%A.facet.8b3
// CHECK:STDOUT:   %.loc28_4.3 => constants.%.e73
// CHECK:STDOUT:   %impl.elem0.loc28_4.2 => constants.%Y.as.A.impl.F
// CHECK:STDOUT:   %specific_impl_fn.loc28_4.2 => constants.%Y.as.A.impl.F.specific_fn
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.092
// CHECK:STDOUT:   %require_complete => constants.%complete_type.05d
// CHECK:STDOUT:   %facet_value => constants.%facet_value.352
// CHECK:STDOUT:   %.loc28_12.4 => constants.%.d23
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.cc1
// CHECK:STDOUT:   %Destroy.facet.loc28 => constants.%Destroy.facet.272
// CHECK:STDOUT:   %.loc28_12.5 => constants.%.30d
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type => constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.type.cee
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op => constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.056
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.loc28 => constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.7b8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.F(constants.%Destroy.facet.08b, constants.%A.facet.8b3, constants.%Z) {
// CHECK:STDOUT:   %U.loc6_8.1 => constants.%Z
// CHECK:STDOUT:   %ptr.loc6_22.1 => constants.%ptr.fb6
// CHECK:STDOUT:   %pattern_type.loc6_18 => constants.%pattern_type.f76
// CHECK:STDOUT:   %T => constants.%Destroy.facet.08b
// CHECK:STDOUT:   %A.type => constants.%A.type.c3a
// CHECK:STDOUT:   %Self => constants.%A.facet.8b3
// CHECK:STDOUT:   %tuple.type.loc6_40.1 => constants.%tuple.type.cfb
// CHECK:STDOUT:   %tuple => constants.%tuple.349
// CHECK:STDOUT:   %T.binding.as_type => constants.%X
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Y
// CHECK:STDOUT:   %tuple.type.loc6_40.2 => constants.%tuple.type.092
// CHECK:STDOUT:   %pattern_type.loc6_25 => constants.%pattern_type.48d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- generic_method_more_params.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.afb: %Destroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.3ab: type = pattern_type %Destroy.type [concrete]
// CHECK:STDOUT:   %A.type.495: type = generic_interface_type @A [concrete]
// CHECK:STDOUT:   %A.generic: %A.type.495 = struct_value () [concrete]
// CHECK:STDOUT:   %A.type.e04: type = facet_type <@A, @A(%T.afb)> [symbolic]
// CHECK:STDOUT:   %Self.0c7: %A.type.e04 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %U.0e3: %Destroy.type = symbolic_binding U, 2 [symbolic]
// CHECK:STDOUT:   %U.binding.as_type.d15: type = symbolic_binding_type U, 2, %U.0e3 [symbolic]
// CHECK:STDOUT:   %pattern_type.a9d: type = pattern_type %U.binding.as_type.d15 [symbolic]
// CHECK:STDOUT:   %tuple.type.d94: type = tuple_type (%Destroy.type, %A.type.e04, %Destroy.type) [symbolic]
// CHECK:STDOUT:   %tuple.13a: %tuple.type.d94 = tuple_value (%T.afb, %Self.0c7, %U.0e3) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.d87: type = symbolic_binding_type T, 0, %T.afb [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.a47: type = symbolic_binding_type Self, 1, %Self.0c7 [symbolic]
// CHECK:STDOUT:   %tuple.type.562: type = tuple_type (%T.binding.as_type.d87, %Self.binding.as_type.a47, %U.binding.as_type.d15) [symbolic]
// CHECK:STDOUT:   %pattern_type.321: type = pattern_type %tuple.type.562 [symbolic]
// CHECK:STDOUT:   %A.F.type.4bf: type = fn_type @A.F, @A(%T.afb) [symbolic]
// CHECK:STDOUT:   %A.F.186: %A.F.type.4bf = struct_value () [symbolic]
// CHECK:STDOUT:   %A.assoc_type.baa: type = assoc_entity_type @A, @A(%T.afb) [symbolic]
// CHECK:STDOUT:   %assoc0.030: %A.assoc_type.baa = assoc_entity element0, @A.%A.F.decl [symbolic]
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Y1: type = class_type @Y1 [concrete]
// CHECK:STDOUT:   %Y2: type = class_type @Y2 [concrete]
// CHECK:STDOUT:   %Z: type = class_type @Z [concrete]
// CHECK:STDOUT:   %V1: %Destroy.type = symbolic_binding V1, 0 [symbolic]
// CHECK:STDOUT:   %V2: %Destroy.type = symbolic_binding V2, 1 [symbolic]
// CHECK:STDOUT:   %W: %Destroy.type = symbolic_binding W, 2 [symbolic]
// CHECK:STDOUT:   %tuple.type.bf5: type = tuple_type (%Destroy.type, %Destroy.type) [concrete]
// CHECK:STDOUT:   %tuple.9a1: %tuple.type.bf5 = tuple_value (%V1, %V2) [symbolic]
// CHECK:STDOUT:   %V1.binding.as_type: type = symbolic_binding_type V1, 0, %V1 [symbolic]
// CHECK:STDOUT:   %V2.binding.as_type: type = symbolic_binding_type V2, 1, %V2 [symbolic]
// CHECK:STDOUT:   %tuple.type.22e: type = tuple_type (%V1.binding.as_type, %V2.binding.as_type) [symbolic]
// CHECK:STDOUT:   %A.type.ced: type = facet_type <@A, @A(%W)> [symbolic]
// CHECK:STDOUT:   %Self.a10: %A.type.ced = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %A.F.type.065: type = fn_type @A.F, @A(%W) [symbolic]
// CHECK:STDOUT:   %A.F.25d: %A.F.type.065 = struct_value () [symbolic]
// CHECK:STDOUT:   %A.assoc_type.2f5: type = assoc_entity_type @A, @A(%W) [symbolic]
// CHECK:STDOUT:   %assoc0.340: %A.assoc_type.2f5 = assoc_entity element0, @A.%A.F.decl [symbolic]
// CHECK:STDOUT:   %require_complete.38a: <witness> = require_complete_type %A.type.ced [symbolic]
// CHECK:STDOUT:   %A.impl_witness.27e: <witness> = impl_witness file.%A.impl_witness_table, @tuple.type.as.A.impl(%V1, %V2, %W) [symbolic]
// CHECK:STDOUT:   %U.49a: %Destroy.type = symbolic_binding U, 3 [symbolic]
// CHECK:STDOUT:   %U.binding.as_type.ab9: type = symbolic_binding_type U, 3, %U.49a [symbolic]
// CHECK:STDOUT:   %pattern_type.499: type = pattern_type %U.binding.as_type.ab9 [symbolic]
// CHECK:STDOUT:   %tuple.type.c44: type = tuple_type (%Destroy.type, %tuple.type.bf5, %Destroy.type) [concrete]
// CHECK:STDOUT:   %tuple.f66: %tuple.type.c44 = tuple_value (%W, %tuple.9a1, %U.49a) [symbolic]
// CHECK:STDOUT:   %W.binding.as_type: type = symbolic_binding_type W, 2, %W [symbolic]
// CHECK:STDOUT:   %tuple.type.c11: type = tuple_type (%W.binding.as_type, %tuple.type.22e, %U.binding.as_type.ab9) [symbolic]
// CHECK:STDOUT:   %pattern_type.b37: type = pattern_type %tuple.type.c11 [symbolic]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type.e6d: type = fn_type @tuple.type.as.A.impl.F, @tuple.type.as.A.impl(%V1, %V2, %W) [symbolic]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.920: %tuple.type.as.A.impl.F.type.e6d = struct_value () [symbolic]
// CHECK:STDOUT:   %A.facet.13e: %A.type.ced = facet_value %tuple.type.22e, (%A.impl_witness.27e) [symbolic]
// CHECK:STDOUT:   %tuple.type.5a5: type = tuple_type (%Destroy.type, %A.type.ced, %Destroy.type) [symbolic]
// CHECK:STDOUT:   %tuple.13b: %tuple.type.5a5 = tuple_value (%W, %A.facet.13e, %U.49a) [symbolic]
// CHECK:STDOUT:   %require_complete.921: <witness> = require_complete_type %tuple.type.c11 [symbolic]
// CHECK:STDOUT:   %require_complete.f93: <witness> = require_complete_type %U.binding.as_type.ab9 [symbolic]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.specific_fn.fb2: <specific function> = specific_function %tuple.type.as.A.impl.F.920, @tuple.type.as.A.impl.F(%V1, %V2, %W, %U.49a) [symbolic]
// CHECK:STDOUT:   %Call.type: type = fn_type @Call [concrete]
// CHECK:STDOUT:   %Call: %Call.type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.821: %tuple.type.24b = tuple_value (%Y1, %Y2) [concrete]
// CHECK:STDOUT:   %tuple.type.a46: type = tuple_type (%Y1, %Y2) [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %DestroyT: %type_where = symbolic_binding DestroyT, 0 [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.15c: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%DestroyT) [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.fd5: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.15c = struct_value () [symbolic]
// CHECK:STDOUT:   %facet_value.b19: %type_where = facet_value %X, () [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.0eb: <witness> = impl_witness imports.%Destroy.impl_witness_table, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.b19) [concrete]
// CHECK:STDOUT:   %Destroy.facet.08b: %Destroy.type = facet_value %X, (%Destroy.impl_witness.0eb) [concrete]
// CHECK:STDOUT:   %A.type.c3a: type = facet_type <@A, @A(%Destroy.facet.08b)> [concrete]
// CHECK:STDOUT:   %facet_value.ff7: %type_where = facet_value %Y1, () [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.0f2: <witness> = impl_witness imports.%Destroy.impl_witness_table, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.ff7) [concrete]
// CHECK:STDOUT:   %Destroy.facet.337: %Destroy.type = facet_value %Y1, (%Destroy.impl_witness.0f2) [concrete]
// CHECK:STDOUT:   %facet_value.5ba: %type_where = facet_value %Y2, () [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.995: <witness> = impl_witness imports.%Destroy.impl_witness_table, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.5ba) [concrete]
// CHECK:STDOUT:   %Destroy.facet.f98: %Destroy.type = facet_value %Y2, (%Destroy.impl_witness.995) [concrete]
// CHECK:STDOUT:   %tuple.bb2: %tuple.type.bf5 = tuple_value (%Destroy.facet.337, %Destroy.facet.f98) [concrete]
// CHECK:STDOUT:   %Self.369: %A.type.c3a = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %A.F.type.1cf: type = fn_type @A.F, @A(%Destroy.facet.08b) [concrete]
// CHECK:STDOUT:   %A.F.d66: %A.F.type.1cf = struct_value () [concrete]
// CHECK:STDOUT:   %A.assoc_type.0c3: type = assoc_entity_type @A, @A(%Destroy.facet.08b) [concrete]
// CHECK:STDOUT:   %assoc0.43e: %A.assoc_type.0c3 = assoc_entity element0, @A.%A.F.decl [concrete]
// CHECK:STDOUT:   %complete_type.309: <witness> = complete_type_witness %A.type.c3a [concrete]
// CHECK:STDOUT:   %A.impl_witness.ccf: <witness> = impl_witness file.%A.impl_witness_table, @tuple.type.as.A.impl(%Destroy.facet.337, %Destroy.facet.f98, %Destroy.facet.08b) [concrete]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type.c24: type = fn_type @tuple.type.as.A.impl.F, @tuple.type.as.A.impl(%Destroy.facet.337, %Destroy.facet.f98, %Destroy.facet.08b) [concrete]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.ae8: %tuple.type.as.A.impl.F.type.c24 = struct_value () [concrete]
// CHECK:STDOUT:   %A.facet.7e6: %A.type.c3a = facet_value %tuple.type.a46, (%A.impl_witness.ccf) [concrete]
// CHECK:STDOUT:   %.dd2: type = fn_type_with_self_type %A.F.type.1cf, %A.facet.7e6 [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.4e3: %type_where = facet_value %Z, () [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.10a: <witness> = impl_witness imports.%Destroy.impl_witness_table, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.4e3) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.d83: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.4e3) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.2cc: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.d83 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.376: %Destroy.type = facet_value %Z, (%Destroy.impl_witness.10a) [concrete]
// CHECK:STDOUT:   %facet_value.7c2: %type_where = facet_value %empty_struct_type, () [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.44e: <witness> = impl_witness imports.%Destroy.impl_witness_table, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.7c2) [concrete]
// CHECK:STDOUT:   %Destroy.facet.c69: %Destroy.type = facet_value %empty_struct_type, (%Destroy.impl_witness.44e) [concrete]
// CHECK:STDOUT:   %pattern_type.8f9: type = pattern_type %Z [concrete]
// CHECK:STDOUT:   %tuple.399: %tuple.type.c44 = tuple_value (%Destroy.facet.08b, %tuple.bb2, %Destroy.facet.376) [concrete]
// CHECK:STDOUT:   %tuple.type.415: type = tuple_type (%X, %tuple.type.a46, %Z) [concrete]
// CHECK:STDOUT:   %pattern_type.0b2: type = pattern_type %tuple.type.415 [concrete]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.specific_fn.66c: <specific function> = specific_function %tuple.type.as.A.impl.F.ae8, @tuple.type.as.A.impl.F(%Destroy.facet.337, %Destroy.facet.f98, %Destroy.facet.08b, %Destroy.facet.376) [concrete]
// CHECK:STDOUT:   %Z.val: %Z = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %facet_value.71a: %type_where = facet_value %tuple.type.415, () [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.c75: <witness> = impl_witness imports.%Destroy.impl_witness_table, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.71a) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.5cd: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.71a) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.472: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.5cd = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type.aa8: <witness> = complete_type_witness %tuple.type.415 [concrete]
// CHECK:STDOUT:   %.a06: require_specific_def_type = require_specific_def @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.71a) [concrete]
// CHECK:STDOUT:   %Destroy.facet.b37: %Destroy.type = facet_value %tuple.type.415, (%Destroy.impl_witness.c75) [concrete]
// CHECK:STDOUT:   %.322: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet.b37 [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.12b: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op.472, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value.71a) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.cb3: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op.2cc, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value.4e3) [concrete]
// CHECK:STDOUT:   %BitAndWith.type.f2e: type = generic_interface_type @BitAndWith [concrete]
// CHECK:STDOUT:   %BitAndWith.generic: %BitAndWith.type.f2e = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndWith.type.8a6: type = facet_type <@BitAndWith, @BitAndWith(type)> [concrete]
// CHECK:STDOUT:   %BitAndWith.Op.type.9a3: type = fn_type @BitAndWith.Op, @BitAndWith(type) [concrete]
// CHECK:STDOUT:   %BitAndWith.impl_witness: <witness> = impl_witness imports.%BitAndWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %BitAndWith.facet: %BitAndWith.type.8a6 = facet_value type, (%BitAndWith.impl_witness) [concrete]
// CHECK:STDOUT:   %.fa7: type = fn_type_with_self_type %BitAndWith.Op.type.9a3, %BitAndWith.facet [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.type: type = fn_type @type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op: %type.as.BitAndWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.bound: <bound method> = bound_method %A.type.c3a, %type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@Destroy & @A, @A(%Destroy.facet.08b)> [concrete]
// CHECK:STDOUT:   %T.c3e: %facet_type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.c82: type = pattern_type %facet_type [concrete]
// CHECK:STDOUT:   %CallGeneric.type: type = fn_type @CallGeneric [concrete]
// CHECK:STDOUT:   %CallGeneric: %CallGeneric.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.binding.as_type.7fc: type = symbolic_binding_type T, 0, %T.c3e [symbolic]
// CHECK:STDOUT:   %A.lookup_impl_witness: <witness> = lookup_impl_witness %T.c3e, @A, @A(%Destroy.facet.08b) [symbolic]
// CHECK:STDOUT:   %A.facet.baa: %A.type.c3a = facet_value %T.binding.as_type.7fc, (%A.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %.dc6: type = fn_type_with_self_type %A.F.type.1cf, %A.facet.baa [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.dc6 = impl_witness_access %A.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %tuple.type.672: type = tuple_type (%Destroy.type, %A.type.c3a, %Destroy.type) [concrete]
// CHECK:STDOUT:   %tuple.91f: %tuple.type.672 = tuple_value (%Destroy.facet.08b, %A.facet.baa, %Destroy.facet.376) [symbolic]
// CHECK:STDOUT:   %tuple.type.4dc: type = tuple_type (%X, %T.binding.as_type.7fc, %Z) [symbolic]
// CHECK:STDOUT:   %pattern_type.295: type = pattern_type %tuple.type.4dc [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @A.F(%Destroy.facet.08b, %A.facet.baa, %Destroy.facet.376) [symbolic]
// CHECK:STDOUT:   %require_complete.5fa: <witness> = require_complete_type %tuple.type.4dc [symbolic]
// CHECK:STDOUT:   %facet_value.4bd: %type_where = facet_value %tuple.type.4dc, () [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.3f4: <witness> = impl_witness imports.%Destroy.impl_witness_table, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.4bd) [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.d95: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.4bd) [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.e3e: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.d95 = struct_value () [symbolic]
// CHECK:STDOUT:   %.c3f: require_specific_def_type = require_specific_def @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.4bd) [symbolic]
// CHECK:STDOUT:   %Destroy.facet.8e1: %Destroy.type = facet_value %tuple.type.4dc, (%Destroy.impl_witness.3f4) [symbolic]
// CHECK:STDOUT:   %.480: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet.8e1 [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.fd9: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op.e3e, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value.4bd) [symbolic]
// CHECK:STDOUT:   %CallIndirect.type: type = fn_type @CallIndirect [concrete]
// CHECK:STDOUT:   %CallIndirect: %CallIndirect.type = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.3b5: %type_where = facet_value %tuple.type.a46, () [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.5a3: <witness> = impl_witness imports.%Destroy.impl_witness_table, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.3b5) [concrete]
// CHECK:STDOUT:   %facet_value.e6f: %facet_type = facet_value %tuple.type.a46, (%Destroy.impl_witness.5a3, %A.impl_witness.ccf) [concrete]
// CHECK:STDOUT:   %CallGeneric.specific_fn: <specific function> = specific_function %CallGeneric, @CallGeneric(%facet_value.e6f) [concrete]
// CHECK:STDOUT:   %tuple.a17: %tuple.type.672 = tuple_value (%Destroy.facet.08b, %A.facet.7e6, %Destroy.facet.376) [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.5c4: @DestroyT.binding.as_type.as.Destroy.impl.%DestroyT.binding.as_type.as.Destroy.impl.Op.type (%DestroyT.binding.as_type.as.Destroy.impl.Op.type.15c) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @DestroyT.binding.as_type.as.Destroy.impl.%DestroyT.binding.as_type.as.Destroy.impl.Op (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.fd5)]
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (%Core.import_ref.5c4), @DestroyT.binding.as_type.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Core.BitAndWith: %BitAndWith.type.f2e = import_ref Core//prelude/parts/as, BitAndWith, loaded [concrete = constants.%BitAndWith.generic]
// CHECK:STDOUT:   %Core.import_ref.636: %type.as.BitAndWith.impl.Op.type = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, loaded [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:   %BitAndWith.impl_witness_table = impl_witness_table (%Core.import_ref.636), @type.as.BitAndWith.impl [concrete]
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
// CHECK:STDOUT:   %A.decl: %A.type.495 = interface_decl @A [concrete = constants.%A.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.3ab = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5: type = splice_block %Destroy.ref [concrete = constants.%Destroy.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc5_13.2: %Destroy.type = symbolic_binding T, 0 [symbolic = %T.loc5_13.1 (constants.%T.afb)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.decl: type = class_decl @X [concrete = constants.%X] {} {}
// CHECK:STDOUT:   %Y1.decl: type = class_decl @Y1 [concrete = constants.%Y1] {} {}
// CHECK:STDOUT:   %Y2.decl: type = class_decl @Y2 [concrete = constants.%Y2] {} {}
// CHECK:STDOUT:   %Z.decl: type = class_decl @Z [concrete = constants.%Z] {} {}
// CHECK:STDOUT:   impl_decl @tuple.type.as.A.impl [concrete] {
// CHECK:STDOUT:     %V1.patt: %pattern_type.3ab = symbolic_binding_pattern V1, 0 [concrete]
// CHECK:STDOUT:     %V2.patt: %pattern_type.3ab = symbolic_binding_pattern V2, 1 [concrete]
// CHECK:STDOUT:     %W.patt: %pattern_type.3ab = symbolic_binding_pattern W, 2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %V1.ref: %Destroy.type = name_ref V1, %V1.loc14_14.2 [symbolic = %V1.loc14_14.1 (constants.%V1)]
// CHECK:STDOUT:     %V2.ref: %Destroy.type = name_ref V2, %V2.loc14_33.2 [symbolic = %V2.loc14_33.1 (constants.%V2)]
// CHECK:STDOUT:     %.loc14_77.1: %tuple.type.bf5 = tuple_literal (%V1.ref, %V2.ref) [symbolic = %tuple (constants.%tuple.9a1)]
// CHECK:STDOUT:     %V1.as_type: type = facet_access_type constants.%V1 [symbolic = %V1.binding.as_type (constants.%V1.binding.as_type)]
// CHECK:STDOUT:     %.loc14_77.2: type = converted constants.%V1, %V1.as_type [symbolic = %V1.binding.as_type (constants.%V1.binding.as_type)]
// CHECK:STDOUT:     %V2.as_type: type = facet_access_type constants.%V2 [symbolic = %V2.binding.as_type (constants.%V2.binding.as_type)]
// CHECK:STDOUT:     %.loc14_77.3: type = converted constants.%V2, %V2.as_type [symbolic = %V2.binding.as_type (constants.%V2.binding.as_type)]
// CHECK:STDOUT:     %.loc14_77.4: type = converted %.loc14_77.1, constants.%tuple.type.22e [symbolic = %tuple.type (constants.%tuple.type.22e)]
// CHECK:STDOUT:     %A.ref: %A.type.495 = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:     %W.ref: %Destroy.type = name_ref W, %W.loc14_52.2 [symbolic = %W.loc14_52.1 (constants.%W)]
// CHECK:STDOUT:     %A.type.loc14_85.2: type = facet_type <@A, @A(constants.%W)> [symbolic = %A.type.loc14_85.1 (constants.%A.type.ced)]
// CHECK:STDOUT:     %.loc14_23: type = splice_block %Destroy.ref.loc14_23 [concrete = constants.%Destroy.type] {
// CHECK:STDOUT:       %.Self.3: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Core.ref.loc14_19: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Destroy.ref.loc14_23: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %V1.loc14_14.2: %Destroy.type = symbolic_binding V1, 0 [symbolic = %V1.loc14_14.1 (constants.%V1)]
// CHECK:STDOUT:     %.loc14_42: type = splice_block %Destroy.ref.loc14_42 [concrete = constants.%Destroy.type] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Core.ref.loc14_38: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Destroy.ref.loc14_42: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %V2.loc14_33.2: %Destroy.type = symbolic_binding V2, 1 [symbolic = %V2.loc14_33.1 (constants.%V2)]
// CHECK:STDOUT:     %.loc14_60: type = splice_block %Destroy.ref.loc14_60 [concrete = constants.%Destroy.type] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Core.ref.loc14_56: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Destroy.ref.loc14_60: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %W.loc14_52.2: %Destroy.type = symbolic_binding W, 2 [symbolic = %W.loc14_52.1 (constants.%W)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.impl_witness_table = impl_witness_table (@tuple.type.as.A.impl.%tuple.type.as.A.impl.F.decl), @tuple.type.as.A.impl [concrete]
// CHECK:STDOUT:   %A.impl_witness: <witness> = impl_witness %A.impl_witness_table, @tuple.type.as.A.impl(constants.%V1, constants.%V2, constants.%W) [symbolic = @tuple.type.as.A.impl.%A.impl_witness (constants.%A.impl_witness.27e)]
// CHECK:STDOUT:   %Call.decl: %Call.type = fn_decl @Call [concrete = constants.%Call] {} {}
// CHECK:STDOUT:   %CallGeneric.decl: %CallGeneric.type = fn_decl @CallGeneric [concrete = constants.%CallGeneric] {
// CHECK:STDOUT:     %T.patt: %pattern_type.c82 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc27_25.1: type = splice_block %.loc27_25.3 [concrete = constants.%facet_type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %A.ref: %A.type.495 = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:       %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:       %Destroy.facet.loc27: %Destroy.type = facet_value %X.ref, (constants.%Destroy.impl_witness.0eb) [concrete = constants.%Destroy.facet.08b]
// CHECK:STDOUT:       %.loc27_23: %Destroy.type = converted %X.ref, %Destroy.facet.loc27 [concrete = constants.%Destroy.facet.08b]
// CHECK:STDOUT:       %A.type: type = facet_type <@A, @A(constants.%Destroy.facet.08b)> [concrete = constants.%A.type.c3a]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:       %impl.elem0.loc27: %.fa7 = impl_witness_access constants.%BitAndWith.impl_witness, element0 [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:       %bound_method.loc27: <bound method> = bound_method %A.type, %impl.elem0.loc27 [concrete = constants.%type.as.BitAndWith.impl.Op.bound]
// CHECK:STDOUT:       %type.as.BitAndWith.impl.Op.call: init type = call %bound_method.loc27(%A.type, %Destroy.ref) [concrete = constants.%facet_type]
// CHECK:STDOUT:       %.loc27_25.2: type = value_of_initializer %type.as.BitAndWith.impl.Op.call [concrete = constants.%facet_type]
// CHECK:STDOUT:       %.loc27_25.3: type = converted %type.as.BitAndWith.impl.Op.call, %.loc27_25.2 [concrete = constants.%facet_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc27_16.2: %facet_type = symbolic_binding T, 0 [symbolic = %T.loc27_16.1 (constants.%T.c3e)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallIndirect.decl: %CallIndirect.type = fn_decl @CallIndirect [concrete = constants.%CallIndirect] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @A(%T.loc5_13.2: %Destroy.type) {
// CHECK:STDOUT:   %T.loc5_13.1: %Destroy.type = symbolic_binding T, 0 [symbolic = %T.loc5_13.1 (constants.%T.afb)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.type: type = facet_type <@A, @A(%T.loc5_13.1)> [symbolic = %A.type (constants.%A.type.e04)]
// CHECK:STDOUT:   %Self.loc5_31.2: @A.%A.type (%A.type.e04) = symbolic_binding Self, 1 [symbolic = %Self.loc5_31.2 (constants.%Self.0c7)]
// CHECK:STDOUT:   %A.F.type: type = fn_type @A.F, @A(%T.loc5_13.1) [symbolic = %A.F.type (constants.%A.F.type.4bf)]
// CHECK:STDOUT:   %A.F: @A.%A.F.type (%A.F.type.4bf) = struct_value () [symbolic = %A.F (constants.%A.F.186)]
// CHECK:STDOUT:   %A.assoc_type: type = assoc_entity_type @A, @A(%T.loc5_13.1) [symbolic = %A.assoc_type (constants.%A.assoc_type.baa)]
// CHECK:STDOUT:   %assoc0.loc6_47.2: @A.%A.assoc_type (%A.assoc_type.baa) = assoc_entity element0, %A.F.decl [symbolic = %assoc0.loc6_47.2 (constants.%assoc0.030)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc5_31.1: @A.%A.type (%A.type.e04) = symbolic_binding Self, 1 [symbolic = %Self.loc5_31.2 (constants.%Self.0c7)]
// CHECK:STDOUT:     %A.F.decl: @A.%A.F.type (%A.F.type.4bf) = fn_decl @A.F [symbolic = @A.%A.F (constants.%A.F.186)] {
// CHECK:STDOUT:       %U.patt: %pattern_type.3ab = symbolic_binding_pattern U, 2 [concrete]
// CHECK:STDOUT:       %u.patt: @A.F.%pattern_type.loc6_26 (%pattern_type.a9d) = value_binding_pattern u [concrete]
// CHECK:STDOUT:       %u.param_patt: @A.F.%pattern_type.loc6_26 (%pattern_type.a9d) = value_param_pattern %u.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @A.F.%pattern_type.loc6_32 (%pattern_type.321) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @A.F.%pattern_type.loc6_32 (%pattern_type.321) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: %Destroy.type = name_ref T, @A.%T.loc5_13.2 [symbolic = %T (constants.%T.afb)]
// CHECK:STDOUT:       %.loc6_39: @A.F.%A.type (%A.type.e04) = specific_constant @A.%Self.loc5_31.1, @A(constants.%T.afb) [symbolic = %Self (constants.%Self.0c7)]
// CHECK:STDOUT:       %Self.ref: @A.F.%A.type (%A.type.e04) = name_ref Self, %.loc6_39 [symbolic = %Self (constants.%Self.0c7)]
// CHECK:STDOUT:       %U.ref.loc6_45: %Destroy.type = name_ref U, %U.loc6_8.2 [symbolic = %U.loc6_8.1 (constants.%U.0e3)]
// CHECK:STDOUT:       %.loc6_46.1: @A.F.%tuple.type.loc6_46.1 (%tuple.type.d94) = tuple_literal (%T.ref, %Self.ref, %U.ref.loc6_45) [symbolic = %tuple (constants.%tuple.13a)]
// CHECK:STDOUT:       %T.as_type: type = facet_access_type constants.%T.afb [symbolic = %T.binding.as_type (constants.%T.binding.as_type.d87)]
// CHECK:STDOUT:       %.loc6_46.2: type = converted constants.%T.afb, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type.d87)]
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type constants.%Self.0c7 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.a47)]
// CHECK:STDOUT:       %.loc6_46.3: type = converted constants.%Self.0c7, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.a47)]
// CHECK:STDOUT:       %U.as_type.loc6_46: type = facet_access_type constants.%U.0e3 [symbolic = %U.binding.as_type (constants.%U.binding.as_type.d15)]
// CHECK:STDOUT:       %.loc6_46.4: type = converted constants.%U.0e3, %U.as_type.loc6_46 [symbolic = %U.binding.as_type (constants.%U.binding.as_type.d15)]
// CHECK:STDOUT:       %.loc6_46.5: type = converted %.loc6_46.1, constants.%tuple.type.562 [symbolic = %tuple.type.loc6_46.2 (constants.%tuple.type.562)]
// CHECK:STDOUT:       %.loc6_16: type = splice_block %Destroy.ref [concrete = constants.%Destroy.type] {
// CHECK:STDOUT:         %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:         %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:         %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %U.loc6_8.2: %Destroy.type = symbolic_binding U, 2 [symbolic = %U.loc6_8.1 (constants.%U.0e3)]
// CHECK:STDOUT:       %u.param: @A.F.%U.binding.as_type (%U.binding.as_type.d15) = value_param call_param0
// CHECK:STDOUT:       %.loc6_29.1: type = splice_block %.loc6_29.2 [symbolic = %U.binding.as_type (constants.%U.binding.as_type.d15)] {
// CHECK:STDOUT:         %U.ref.loc6_29: %Destroy.type = name_ref U, %U.loc6_8.2 [symbolic = %U.loc6_8.1 (constants.%U.0e3)]
// CHECK:STDOUT:         %U.as_type.loc6_29: type = facet_access_type %U.ref.loc6_29 [symbolic = %U.binding.as_type (constants.%U.binding.as_type.d15)]
// CHECK:STDOUT:         %.loc6_29.2: type = converted %U.ref.loc6_29, %U.as_type.loc6_29 [symbolic = %U.binding.as_type (constants.%U.binding.as_type.d15)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %u: @A.F.%U.binding.as_type (%U.binding.as_type.d15) = value_binding u, %u.param
// CHECK:STDOUT:       %return.param: ref @A.F.%tuple.type.loc6_46.2 (%tuple.type.562) = out_param call_param1
// CHECK:STDOUT:       %return: ref @A.F.%tuple.type.loc6_46.2 (%tuple.type.562) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc6_47.1: @A.%A.assoc_type (%A.assoc_type.baa) = assoc_entity element0, %A.F.decl [symbolic = %assoc0.loc6_47.2 (constants.%assoc0.030)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc5_31.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .F = %assoc0.loc6_47.1
// CHECK:STDOUT:     witness = (%A.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @tuple.type.as.A.impl(%V1.loc14_14.2: %Destroy.type, %V2.loc14_33.2: %Destroy.type, %W.loc14_52.2: %Destroy.type) {
// CHECK:STDOUT:   %V1.loc14_14.1: %Destroy.type = symbolic_binding V1, 0 [symbolic = %V1.loc14_14.1 (constants.%V1)]
// CHECK:STDOUT:   %V2.loc14_33.1: %Destroy.type = symbolic_binding V2, 1 [symbolic = %V2.loc14_33.1 (constants.%V2)]
// CHECK:STDOUT:   %W.loc14_52.1: %Destroy.type = symbolic_binding W, 2 [symbolic = %W.loc14_52.1 (constants.%W)]
// CHECK:STDOUT:   %tuple: %tuple.type.bf5 = tuple_value (%V1.loc14_14.1, %V2.loc14_33.1) [symbolic = %tuple (constants.%tuple.9a1)]
// CHECK:STDOUT:   %V1.binding.as_type: type = symbolic_binding_type V1, 0, %V1.loc14_14.1 [symbolic = %V1.binding.as_type (constants.%V1.binding.as_type)]
// CHECK:STDOUT:   %V2.binding.as_type: type = symbolic_binding_type V2, 1, %V2.loc14_33.1 [symbolic = %V2.binding.as_type (constants.%V2.binding.as_type)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%V1.binding.as_type, %V2.binding.as_type) [symbolic = %tuple.type (constants.%tuple.type.22e)]
// CHECK:STDOUT:   %A.type.loc14_85.1: type = facet_type <@A, @A(%W.loc14_52.1)> [symbolic = %A.type.loc14_85.1 (constants.%A.type.ced)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %A.type.loc14_85.1 [symbolic = %require_complete (constants.%require_complete.38a)]
// CHECK:STDOUT:   %A.impl_witness: <witness> = impl_witness file.%A.impl_witness_table, @tuple.type.as.A.impl(%V1.loc14_14.1, %V2.loc14_33.1, %W.loc14_52.1) [symbolic = %A.impl_witness (constants.%A.impl_witness.27e)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type: type = fn_type @tuple.type.as.A.impl.F, @tuple.type.as.A.impl(%V1.loc14_14.1, %V2.loc14_33.1, %W.loc14_52.1) [symbolic = %tuple.type.as.A.impl.F.type (constants.%tuple.type.as.A.impl.F.type.e6d)]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F: @tuple.type.as.A.impl.%tuple.type.as.A.impl.F.type (%tuple.type.as.A.impl.F.type.e6d) = struct_value () [symbolic = %tuple.type.as.A.impl.F (constants.%tuple.type.as.A.impl.F.920)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc14_77.4 as %A.type.loc14_85.2 {
// CHECK:STDOUT:     %tuple.type.as.A.impl.F.decl: @tuple.type.as.A.impl.%tuple.type.as.A.impl.F.type (%tuple.type.as.A.impl.F.type.e6d) = fn_decl @tuple.type.as.A.impl.F [symbolic = @tuple.type.as.A.impl.%tuple.type.as.A.impl.F (constants.%tuple.type.as.A.impl.F.920)] {
// CHECK:STDOUT:       %U.patt: %pattern_type.3ab = symbolic_binding_pattern U, 3 [concrete]
// CHECK:STDOUT:       %u.patt: @tuple.type.as.A.impl.F.%pattern_type.loc17_26 (%pattern_type.499) = value_binding_pattern u [concrete]
// CHECK:STDOUT:       %u.param_patt: @tuple.type.as.A.impl.F.%pattern_type.loc17_26 (%pattern_type.499) = value_param_pattern %u.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @tuple.type.as.A.impl.F.%pattern_type.loc17_32 (%pattern_type.b37) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @tuple.type.as.A.impl.F.%pattern_type.loc17_32 (%pattern_type.b37) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %W.ref: %Destroy.type = name_ref W, @tuple.type.as.A.impl.%W.loc14_52.2 [symbolic = %W (constants.%W)]
// CHECK:STDOUT:       %V1.ref: %Destroy.type = name_ref V1, @tuple.type.as.A.impl.%V1.loc14_14.2 [symbolic = %V1 (constants.%V1)]
// CHECK:STDOUT:       %V2.ref: %Destroy.type = name_ref V2, @tuple.type.as.A.impl.%V2.loc14_33.2 [symbolic = %V2 (constants.%V2)]
// CHECK:STDOUT:       %.loc17_46: %tuple.type.bf5 = tuple_literal (%V1.ref, %V2.ref) [symbolic = %tuple.loc17_46 (constants.%tuple.9a1)]
// CHECK:STDOUT:       %U.ref.loc17_49: %Destroy.type = name_ref U, %U.loc17_8.2 [symbolic = %U.loc17_8.1 (constants.%U.49a)]
// CHECK:STDOUT:       %.loc17_50.1: %tuple.type.c44 = tuple_literal (%W.ref, %.loc17_46, %U.ref.loc17_49) [symbolic = %tuple.loc17_50 (constants.%tuple.f66)]
// CHECK:STDOUT:       %W.as_type: type = facet_access_type constants.%W [symbolic = %W.binding.as_type (constants.%W.binding.as_type)]
// CHECK:STDOUT:       %.loc17_50.2: type = converted constants.%W, %W.as_type [symbolic = %W.binding.as_type (constants.%W.binding.as_type)]
// CHECK:STDOUT:       %V1.as_type: type = facet_access_type constants.%V1 [symbolic = %V1.binding.as_type (constants.%V1.binding.as_type)]
// CHECK:STDOUT:       %.loc17_50.3: type = converted constants.%V1, %V1.as_type [symbolic = %V1.binding.as_type (constants.%V1.binding.as_type)]
// CHECK:STDOUT:       %V2.as_type: type = facet_access_type constants.%V2 [symbolic = %V2.binding.as_type (constants.%V2.binding.as_type)]
// CHECK:STDOUT:       %.loc17_50.4: type = converted constants.%V2, %V2.as_type [symbolic = %V2.binding.as_type (constants.%V2.binding.as_type)]
// CHECK:STDOUT:       %.loc17_50.5: type = converted constants.%tuple.9a1, constants.%tuple.type.22e [symbolic = %tuple.type.loc17_50.1 (constants.%tuple.type.22e)]
// CHECK:STDOUT:       %U.as_type.loc17_50: type = facet_access_type constants.%U.49a [symbolic = %U.binding.as_type (constants.%U.binding.as_type.ab9)]
// CHECK:STDOUT:       %.loc17_50.6: type = converted constants.%U.49a, %U.as_type.loc17_50 [symbolic = %U.binding.as_type (constants.%U.binding.as_type.ab9)]
// CHECK:STDOUT:       %.loc17_50.7: type = converted %.loc17_50.1, constants.%tuple.type.c11 [symbolic = %tuple.type.loc17_50.2 (constants.%tuple.type.c11)]
// CHECK:STDOUT:       %.loc17_16: type = splice_block %Destroy.ref [concrete = constants.%Destroy.type] {
// CHECK:STDOUT:         %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:         %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:         %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %U.loc17_8.2: %Destroy.type = symbolic_binding U, 3 [symbolic = %U.loc17_8.1 (constants.%U.49a)]
// CHECK:STDOUT:       %u.param: @tuple.type.as.A.impl.F.%U.binding.as_type (%U.binding.as_type.ab9) = value_param call_param0
// CHECK:STDOUT:       %.loc17_29.1: type = splice_block %.loc17_29.2 [symbolic = %U.binding.as_type (constants.%U.binding.as_type.ab9)] {
// CHECK:STDOUT:         %U.ref.loc17_29: %Destroy.type = name_ref U, %U.loc17_8.2 [symbolic = %U.loc17_8.1 (constants.%U.49a)]
// CHECK:STDOUT:         %U.as_type.loc17_29: type = facet_access_type %U.ref.loc17_29 [symbolic = %U.binding.as_type (constants.%U.binding.as_type.ab9)]
// CHECK:STDOUT:         %.loc17_29.2: type = converted %U.ref.loc17_29, %U.as_type.loc17_29 [symbolic = %U.binding.as_type (constants.%U.binding.as_type.ab9)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %u: @tuple.type.as.A.impl.F.%U.binding.as_type (%U.binding.as_type.ab9) = value_binding u, %u.param
// CHECK:STDOUT:       %return.param: ref @tuple.type.as.A.impl.F.%tuple.type.loc17_50.2 (%tuple.type.c11) = out_param call_param1
// CHECK:STDOUT:       %return: ref @tuple.type.as.A.impl.F.%tuple.type.loc17_50.2 (%tuple.type.c11) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .W = <poisoned>
// CHECK:STDOUT:     .V1 = <poisoned>
// CHECK:STDOUT:     .V2 = <poisoned>
// CHECK:STDOUT:     .F = %tuple.type.as.A.impl.F.decl
// CHECK:STDOUT:     witness = file.%A.impl_witness
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
// CHECK:STDOUT: generic fn @A.F(@A.%T.loc5_13.2: %Destroy.type, @A.%Self.loc5_31.1: @A.%A.type (%A.type.e04), %U.loc6_8.2: %Destroy.type) {
// CHECK:STDOUT:   %U.loc6_8.1: %Destroy.type = symbolic_binding U, 2 [symbolic = %U.loc6_8.1 (constants.%U.0e3)]
// CHECK:STDOUT:   %U.binding.as_type: type = symbolic_binding_type U, 2, %U.loc6_8.1 [symbolic = %U.binding.as_type (constants.%U.binding.as_type.d15)]
// CHECK:STDOUT:   %pattern_type.loc6_26: type = pattern_type %U.binding.as_type [symbolic = %pattern_type.loc6_26 (constants.%pattern_type.a9d)]
// CHECK:STDOUT:   %T: %Destroy.type = symbolic_binding T, 0 [symbolic = %T (constants.%T.afb)]
// CHECK:STDOUT:   %A.type: type = facet_type <@A, @A(%T)> [symbolic = %A.type (constants.%A.type.e04)]
// CHECK:STDOUT:   %Self: @A.F.%A.type (%A.type.e04) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.0c7)]
// CHECK:STDOUT:   %tuple.type.loc6_46.1: type = tuple_type (constants.%Destroy.type, %A.type, constants.%Destroy.type) [symbolic = %tuple.type.loc6_46.1 (constants.%tuple.type.d94)]
// CHECK:STDOUT:   %tuple: @A.F.%tuple.type.loc6_46.1 (%tuple.type.d94) = tuple_value (%T, %Self, %U.loc6_8.1) [symbolic = %tuple (constants.%tuple.13a)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T [symbolic = %T.binding.as_type (constants.%T.binding.as_type.d87)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.a47)]
// CHECK:STDOUT:   %tuple.type.loc6_46.2: type = tuple_type (%T.binding.as_type, %Self.binding.as_type, %U.binding.as_type) [symbolic = %tuple.type.loc6_46.2 (constants.%tuple.type.562)]
// CHECK:STDOUT:   %pattern_type.loc6_32: type = pattern_type %tuple.type.loc6_46.2 [symbolic = %pattern_type.loc6_32 (constants.%pattern_type.321)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param: @A.F.%U.binding.as_type (%U.binding.as_type.d15)) -> @A.F.%tuple.type.loc6_46.2 (%tuple.type.562);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @tuple.type.as.A.impl.F(@tuple.type.as.A.impl.%V1.loc14_14.2: %Destroy.type, @tuple.type.as.A.impl.%V2.loc14_33.2: %Destroy.type, @tuple.type.as.A.impl.%W.loc14_52.2: %Destroy.type, %U.loc17_8.2: %Destroy.type) {
// CHECK:STDOUT:   %U.loc17_8.1: %Destroy.type = symbolic_binding U, 3 [symbolic = %U.loc17_8.1 (constants.%U.49a)]
// CHECK:STDOUT:   %U.binding.as_type: type = symbolic_binding_type U, 3, %U.loc17_8.1 [symbolic = %U.binding.as_type (constants.%U.binding.as_type.ab9)]
// CHECK:STDOUT:   %pattern_type.loc17_26: type = pattern_type %U.binding.as_type [symbolic = %pattern_type.loc17_26 (constants.%pattern_type.499)]
// CHECK:STDOUT:   %W: %Destroy.type = symbolic_binding W, 2 [symbolic = %W (constants.%W)]
// CHECK:STDOUT:   %V1: %Destroy.type = symbolic_binding V1, 0 [symbolic = %V1 (constants.%V1)]
// CHECK:STDOUT:   %V2: %Destroy.type = symbolic_binding V2, 1 [symbolic = %V2 (constants.%V2)]
// CHECK:STDOUT:   %tuple.loc17_46: %tuple.type.bf5 = tuple_value (%V1, %V2) [symbolic = %tuple.loc17_46 (constants.%tuple.9a1)]
// CHECK:STDOUT:   %tuple.loc17_50: %tuple.type.c44 = tuple_value (%W, %tuple.loc17_46, %U.loc17_8.1) [symbolic = %tuple.loc17_50 (constants.%tuple.f66)]
// CHECK:STDOUT:   %W.binding.as_type: type = symbolic_binding_type W, 2, %W [symbolic = %W.binding.as_type (constants.%W.binding.as_type)]
// CHECK:STDOUT:   %V1.binding.as_type: type = symbolic_binding_type V1, 0, %V1 [symbolic = %V1.binding.as_type (constants.%V1.binding.as_type)]
// CHECK:STDOUT:   %V2.binding.as_type: type = symbolic_binding_type V2, 1, %V2 [symbolic = %V2.binding.as_type (constants.%V2.binding.as_type)]
// CHECK:STDOUT:   %tuple.type.loc17_50.1: type = tuple_type (%V1.binding.as_type, %V2.binding.as_type) [symbolic = %tuple.type.loc17_50.1 (constants.%tuple.type.22e)]
// CHECK:STDOUT:   %tuple.type.loc17_50.2: type = tuple_type (%W.binding.as_type, %tuple.type.loc17_50.1, %U.binding.as_type) [symbolic = %tuple.type.loc17_50.2 (constants.%tuple.type.c11)]
// CHECK:STDOUT:   %pattern_type.loc17_32: type = pattern_type %tuple.type.loc17_50.2 [symbolic = %pattern_type.loc17_32 (constants.%pattern_type.b37)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc17_32: <witness> = require_complete_type %tuple.type.loc17_50.2 [symbolic = %require_complete.loc17_32 (constants.%require_complete.921)]
// CHECK:STDOUT:   %require_complete.loc17_27: <witness> = require_complete_type %U.binding.as_type [symbolic = %require_complete.loc17_27 (constants.%require_complete.f93)]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type: type = fn_type @tuple.type.as.A.impl.F, @tuple.type.as.A.impl(%V1, %V2, %W) [symbolic = %tuple.type.as.A.impl.F.type (constants.%tuple.type.as.A.impl.F.type.e6d)]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F: @tuple.type.as.A.impl.F.%tuple.type.as.A.impl.F.type (%tuple.type.as.A.impl.F.type.e6d) = struct_value () [symbolic = %tuple.type.as.A.impl.F (constants.%tuple.type.as.A.impl.F.920)]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.specific_fn.loc18_12.2: <specific function> = specific_function %tuple.type.as.A.impl.F, @tuple.type.as.A.impl.F(%V1, %V2, %W, %U.loc17_8.1) [symbolic = %tuple.type.as.A.impl.F.specific_fn.loc18_12.2 (constants.%tuple.type.as.A.impl.F.specific_fn.fb2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param: @tuple.type.as.A.impl.F.%U.binding.as_type (%U.binding.as_type.ab9)) -> %return.param: @tuple.type.as.A.impl.F.%tuple.type.loc17_50.2 (%tuple.type.c11) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc18_12: @tuple.type.as.A.impl.F.%tuple.type.as.A.impl.F.type (%tuple.type.as.A.impl.F.type.e6d) = specific_constant @tuple.type.as.A.impl.%tuple.type.as.A.impl.F.decl, @tuple.type.as.A.impl(constants.%V1, constants.%V2, constants.%W) [symbolic = %tuple.type.as.A.impl.F (constants.%tuple.type.as.A.impl.F.920)]
// CHECK:STDOUT:     %F.ref: @tuple.type.as.A.impl.F.%tuple.type.as.A.impl.F.type (%tuple.type.as.A.impl.F.type.e6d) = name_ref F, %.loc18_12 [symbolic = %tuple.type.as.A.impl.F (constants.%tuple.type.as.A.impl.F.920)]
// CHECK:STDOUT:     %U.ref.loc18: %Destroy.type = name_ref U, %U.loc17_8.2 [symbolic = %U.loc17_8.1 (constants.%U.49a)]
// CHECK:STDOUT:     %u.ref: @tuple.type.as.A.impl.F.%U.binding.as_type (%U.binding.as_type.ab9) = name_ref u, %u
// CHECK:STDOUT:     %.loc18_18.1: %Destroy.type = converted constants.%U.binding.as_type.ab9, constants.%U.49a [symbolic = %U.loc17_8.1 (constants.%U.49a)]
// CHECK:STDOUT:     %.loc18_18.2: %Destroy.type = converted constants.%U.binding.as_type.ab9, constants.%U.49a [symbolic = %U.loc17_8.1 (constants.%U.49a)]
// CHECK:STDOUT:     %tuple.type.as.A.impl.F.specific_fn.loc18_12.1: <specific function> = specific_function %F.ref, @tuple.type.as.A.impl.F(constants.%V1, constants.%V2, constants.%W, constants.%U.49a) [symbolic = %tuple.type.as.A.impl.F.specific_fn.loc18_12.2 (constants.%tuple.type.as.A.impl.F.specific_fn.fb2)]
// CHECK:STDOUT:     %.loc17_32: ref @tuple.type.as.A.impl.F.%tuple.type.loc17_50.2 (%tuple.type.c11) = splice_block %return {}
// CHECK:STDOUT:     %tuple.type.as.A.impl.F.call: init @tuple.type.as.A.impl.F.%tuple.type.loc17_50.2 (%tuple.type.c11) = call %tuple.type.as.A.impl.F.specific_fn.loc18_12.1(%u.ref) to %.loc17_32
// CHECK:STDOUT:     return %tuple.type.as.A.impl.F.call to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Y1.ref: type = name_ref Y1, file.%Y1.decl [concrete = constants.%Y1]
// CHECK:STDOUT:   %Y2.ref: type = name_ref Y2, file.%Y2.decl [concrete = constants.%Y2]
// CHECK:STDOUT:   %.loc24_12: %tuple.type.24b = tuple_literal (%Y1.ref, %Y2.ref) [concrete = constants.%tuple.821]
// CHECK:STDOUT:   %.loc24_14: type = converted %.loc24_12, constants.%tuple.type.a46 [concrete = constants.%tuple.type.a46]
// CHECK:STDOUT:   %A.ref: %A.type.495 = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:   %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %Destroy.facet.loc24_29: %Destroy.type = facet_value %X.ref, (constants.%Destroy.impl_witness.0eb) [concrete = constants.%Destroy.facet.08b]
// CHECK:STDOUT:   %.loc24_29: %Destroy.type = converted %X.ref, %Destroy.facet.loc24_29 [concrete = constants.%Destroy.facet.08b]
// CHECK:STDOUT:   %A.type: type = facet_type <@A, @A(constants.%Destroy.facet.08b)> [concrete = constants.%A.type.c3a]
// CHECK:STDOUT:   %A.facet: %A.type.c3a = facet_value %.loc24_14, (constants.%A.impl_witness.ccf) [concrete = constants.%A.facet.7e6]
// CHECK:STDOUT:   %.loc24_23: %A.type.c3a = converted %.loc24_14, %A.facet [concrete = constants.%A.facet.7e6]
// CHECK:STDOUT:   %.loc24_31.1: %A.assoc_type.0c3 = specific_constant @A.%assoc0.loc6_47.1, @A(constants.%Destroy.facet.08b) [concrete = constants.%assoc0.43e]
// CHECK:STDOUT:   %F.ref: %A.assoc_type.0c3 = name_ref F, %.loc24_31.1 [concrete = constants.%assoc0.43e]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc24_23 [concrete = constants.%tuple.type.a46]
// CHECK:STDOUT:   %.loc24_31.2: type = converted %.loc24_23, %as_type [concrete = constants.%tuple.type.a46]
// CHECK:STDOUT:   %impl.elem0: %.dd2 = impl_witness_access constants.%A.impl_witness.ccf, element0 [concrete = constants.%tuple.type.as.A.impl.F.ae8]
// CHECK:STDOUT:   %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z]
// CHECK:STDOUT:   %.loc24_38.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %Destroy.facet.loc24_39.1: %Destroy.type = facet_value %Z.ref, (constants.%Destroy.impl_witness.10a) [concrete = constants.%Destroy.facet.376]
// CHECK:STDOUT:   %.loc24_39.1: %Destroy.type = converted %Z.ref, %Destroy.facet.loc24_39.1 [concrete = constants.%Destroy.facet.376]
// CHECK:STDOUT:   %Destroy.facet.loc24_39.2: %Destroy.type = facet_value constants.%empty_struct_type, (constants.%Destroy.impl_witness.44e) [concrete = constants.%Destroy.facet.c69]
// CHECK:STDOUT:   %.loc24_39.2: %Destroy.type = converted constants.%empty_struct_type, %Destroy.facet.loc24_39.2 [concrete = constants.%Destroy.facet.c69]
// CHECK:STDOUT:   %Destroy.facet.loc24_39.3: %Destroy.type = facet_value constants.%empty_struct_type, (constants.%Destroy.impl_witness.44e) [concrete = constants.%Destroy.facet.c69]
// CHECK:STDOUT:   %.loc24_39.3: %Destroy.type = converted constants.%empty_struct_type, %Destroy.facet.loc24_39.3 [concrete = constants.%Destroy.facet.c69]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @tuple.type.as.A.impl.F(constants.%Destroy.facet.337, constants.%Destroy.facet.f98, constants.%Destroy.facet.08b, constants.%Destroy.facet.376) [concrete = constants.%tuple.type.as.A.impl.F.specific_fn.66c]
// CHECK:STDOUT:   %.loc24_39.4: ref %tuple.type.415 = temporary_storage
// CHECK:STDOUT:   %.loc24_38.2: ref %Z = temporary_storage
// CHECK:STDOUT:   %.loc24_38.3: init %Z = class_init (), %.loc24_38.2 [concrete = constants.%Z.val]
// CHECK:STDOUT:   %.loc24_38.4: ref %Z = temporary %.loc24_38.2, %.loc24_38.3
// CHECK:STDOUT:   %.loc24_38.5: ref %Z = converted %.loc24_38.1, %.loc24_38.4
// CHECK:STDOUT:   %.loc24_38.6: %Z = acquire_value %.loc24_38.5
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.call: init %tuple.type.415 = call %specific_fn(%.loc24_38.6) to %.loc24_39.4
// CHECK:STDOUT:   %.loc24_39.5: ref %tuple.type.415 = temporary %.loc24_39.4, %tuple.type.as.A.impl.F.call
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc24_39: <bound method> = bound_method %.loc24_39.5, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.472
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.472, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.71a) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.12b]
// CHECK:STDOUT:   %bound_method.loc24_39: <bound method> = bound_method %.loc24_39.5, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc24_39: init %empty_tuple.type = call %bound_method.loc24_39(%.loc24_39.5)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc24_38: <bound method> = bound_method %.loc24_38.4, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.2cc
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.2: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.2cc, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.4e3) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.cb3]
// CHECK:STDOUT:   %bound_method.loc24_38: <bound method> = bound_method %.loc24_38.4, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc24_38: init %empty_tuple.type = call %bound_method.loc24_38(%.loc24_38.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallGeneric(%T.loc27_16.2: %facet_type) {
// CHECK:STDOUT:   %T.loc27_16.1: %facet_type = symbolic_binding T, 0 [symbolic = %T.loc27_16.1 (constants.%T.c3e)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc27_16.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.7fc)]
// CHECK:STDOUT:   %A.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc27_16.1, @A, @A(constants.%Destroy.facet.08b) [symbolic = %A.lookup_impl_witness (constants.%A.lookup_impl_witness)]
// CHECK:STDOUT:   %A.facet: %A.type.c3a = facet_value %T.binding.as_type, (%A.lookup_impl_witness) [symbolic = %A.facet (constants.%A.facet.baa)]
// CHECK:STDOUT:   %.loc28_4.3: type = fn_type_with_self_type constants.%A.F.type.1cf, %A.facet [symbolic = %.loc28_4.3 (constants.%.dc6)]
// CHECK:STDOUT:   %impl.elem0.loc28_4.2: @CallGeneric.%.loc28_4.3 (%.dc6) = impl_witness_access %A.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc28_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc28_4.2: <specific function> = specific_impl_function %impl.elem0.loc28_4.2, @A.F(constants.%Destroy.facet.08b, %A.facet, constants.%Destroy.facet.376) [symbolic = %specific_impl_fn.loc28_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (constants.%X, %T.binding.as_type, constants.%Z) [symbolic = %tuple.type (constants.%tuple.type.4dc)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %tuple.type [symbolic = %require_complete (constants.%require_complete.5fa)]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %tuple.type, () [symbolic = %facet_value (constants.%facet_value.4bd)]
// CHECK:STDOUT:   %.loc28_12.7: require_specific_def_type = require_specific_def @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [symbolic = %.loc28_12.7 (constants.%.c3f)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness imports.%Destroy.impl_witness_table, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness.3f4)]
// CHECK:STDOUT:   %Destroy.facet.loc28_12.4: %Destroy.type = facet_value %tuple.type, (%Destroy.impl_witness) [symbolic = %Destroy.facet.loc28_12.4 (constants.%Destroy.facet.8e1)]
// CHECK:STDOUT:   %.loc28_12.8: type = fn_type_with_self_type constants.%Destroy.Op.type, %Destroy.facet.loc28_12.4 [symbolic = %.loc28_12.8 (constants.%.480)]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op.type (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.type.d95)]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op: @CallGeneric.%DestroyT.binding.as_type.as.Destroy.impl.Op.type (%DestroyT.binding.as_type.as.Destroy.impl.Op.type.d95) = struct_value () [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.e3e)]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.loc28: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value) [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.loc28 (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.fd9)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref: %facet_type = name_ref T, %T.loc27_16.2 [symbolic = %T.loc27_16.1 (constants.%T.c3e)]
// CHECK:STDOUT:     %.loc28_4.1: %A.assoc_type.0c3 = specific_constant @A.%assoc0.loc6_47.1, @A(constants.%Destroy.facet.08b) [concrete = constants.%assoc0.43e]
// CHECK:STDOUT:     %F.ref: %A.assoc_type.0c3 = name_ref F, %.loc28_4.1 [concrete = constants.%assoc0.43e]
// CHECK:STDOUT:     %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type.7fc)]
// CHECK:STDOUT:     %.loc28_4.2: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type.7fc)]
// CHECK:STDOUT:     %impl.elem0.loc28_4.1: @CallGeneric.%.loc28_4.3 (%.dc6) = impl_witness_access constants.%A.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc28_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z]
// CHECK:STDOUT:     %.loc28_11.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %Destroy.facet.loc28_12.1: %Destroy.type = facet_value %Z.ref, (constants.%Destroy.impl_witness.10a) [concrete = constants.%Destroy.facet.376]
// CHECK:STDOUT:     %.loc28_12.1: %Destroy.type = converted %Z.ref, %Destroy.facet.loc28_12.1 [concrete = constants.%Destroy.facet.376]
// CHECK:STDOUT:     %Destroy.facet.loc28_12.2: %Destroy.type = facet_value constants.%empty_struct_type, (constants.%Destroy.impl_witness.44e) [concrete = constants.%Destroy.facet.c69]
// CHECK:STDOUT:     %.loc28_12.2: %Destroy.type = converted constants.%empty_struct_type, %Destroy.facet.loc28_12.2 [concrete = constants.%Destroy.facet.c69]
// CHECK:STDOUT:     %Destroy.facet.loc28_12.3: %Destroy.type = facet_value constants.%empty_struct_type, (constants.%Destroy.impl_witness.44e) [concrete = constants.%Destroy.facet.c69]
// CHECK:STDOUT:     %.loc28_12.3: %Destroy.type = converted constants.%empty_struct_type, %Destroy.facet.loc28_12.3 [concrete = constants.%Destroy.facet.c69]
// CHECK:STDOUT:     %specific_impl_fn.loc28_4.1: <specific function> = specific_impl_function %impl.elem0.loc28_4.1, @A.F(constants.%Destroy.facet.08b, constants.%A.facet.baa, constants.%Destroy.facet.376) [symbolic = %specific_impl_fn.loc28_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %.loc28_12.4: ref @CallGeneric.%tuple.type (%tuple.type.4dc) = temporary_storage
// CHECK:STDOUT:     %.loc28_11.2: ref %Z = temporary_storage
// CHECK:STDOUT:     %.loc28_11.3: init %Z = class_init (), %.loc28_11.2 [concrete = constants.%Z.val]
// CHECK:STDOUT:     %.loc28_11.4: ref %Z = temporary %.loc28_11.2, %.loc28_11.3
// CHECK:STDOUT:     %.loc28_11.5: ref %Z = converted %.loc28_11.1, %.loc28_11.4
// CHECK:STDOUT:     %.loc28_11.6: %Z = acquire_value %.loc28_11.5
// CHECK:STDOUT:     %.loc28_12.5: init @CallGeneric.%tuple.type (%tuple.type.4dc) = call %specific_impl_fn.loc28_4.1(%.loc28_11.6) to %.loc28_12.4
// CHECK:STDOUT:     %.loc28_12.6: ref @CallGeneric.%tuple.type (%tuple.type.4dc) = temporary %.loc28_12.4, %.loc28_12.5
// CHECK:STDOUT:     %impl.elem0.loc28_12: @CallGeneric.%.loc28_12.8 (%.480) = impl_witness_access constants.%Destroy.impl_witness.3f4, element0 [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.e3e)]
// CHECK:STDOUT:     %bound_method.loc28_12.1: <bound method> = bound_method %.loc28_12.6, %impl.elem0.loc28_12
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0.loc28_12, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.4bd) [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.loc28 (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.fd9)]
// CHECK:STDOUT:     %bound_method.loc28_12.2: <bound method> = bound_method %.loc28_12.6, %specific_fn
// CHECK:STDOUT:     %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc28_12: init %empty_tuple.type = call %bound_method.loc28_12.2(%.loc28_12.6)
// CHECK:STDOUT:     %DestroyT.binding.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc28_11.4, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.2cc
// CHECK:STDOUT:     %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.2cc, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.4e3) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.cb3]
// CHECK:STDOUT:     %bound_method.loc28_11: <bound method> = bound_method %.loc28_11.4, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:     %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc28_11: init %empty_tuple.type = call %bound_method.loc28_11(%.loc28_11.4)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallIndirect() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CallGeneric.ref: %CallGeneric.type = name_ref CallGeneric, file.%CallGeneric.decl [concrete = constants.%CallGeneric]
// CHECK:STDOUT:   %Y1.ref: type = name_ref Y1, file.%Y1.decl [concrete = constants.%Y1]
// CHECK:STDOUT:   %Y2.ref: type = name_ref Y2, file.%Y2.decl [concrete = constants.%Y2]
// CHECK:STDOUT:   %.loc33_22: %tuple.type.24b = tuple_literal (%Y1.ref, %Y2.ref) [concrete = constants.%tuple.821]
// CHECK:STDOUT:   %.loc33_24: type = converted %.loc33_22, constants.%tuple.type.a46 [concrete = constants.%tuple.type.a46]
// CHECK:STDOUT:   %facet_value: %facet_type = facet_value %.loc33_24, (constants.%Destroy.impl_witness.5a3, constants.%A.impl_witness.ccf) [concrete = constants.%facet_value.e6f]
// CHECK:STDOUT:   %.loc33_31: %facet_type = converted %.loc33_24, %facet_value [concrete = constants.%facet_value.e6f]
// CHECK:STDOUT:   %CallGeneric.specific_fn: <specific function> = specific_function %CallGeneric.ref, @CallGeneric(constants.%facet_value.e6f) [concrete = constants.%CallGeneric.specific_fn]
// CHECK:STDOUT:   %CallGeneric.call: init %empty_tuple.type = call %CallGeneric.specific_fn()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%T.afb) {
// CHECK:STDOUT:   %T.loc5_13.1 => constants.%T.afb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.F(constants.%T.afb, constants.%Self.0c7, constants.%U.0e3) {
// CHECK:STDOUT:   %U.loc6_8.1 => constants.%U.0e3
// CHECK:STDOUT:   %U.binding.as_type => constants.%U.binding.as_type.d15
// CHECK:STDOUT:   %pattern_type.loc6_26 => constants.%pattern_type.a9d
// CHECK:STDOUT:   %T => constants.%T.afb
// CHECK:STDOUT:   %A.type => constants.%A.type.e04
// CHECK:STDOUT:   %Self => constants.%Self.0c7
// CHECK:STDOUT:   %tuple.type.loc6_46.1 => constants.%tuple.type.d94
// CHECK:STDOUT:   %tuple => constants.%tuple.13a
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type.d87
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.a47
// CHECK:STDOUT:   %tuple.type.loc6_46.2 => constants.%tuple.type.562
// CHECK:STDOUT:   %pattern_type.loc6_32 => constants.%pattern_type.321
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%W) {
// CHECK:STDOUT:   %T.loc5_13.1 => constants.%W
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.type => constants.%A.type.ced
// CHECK:STDOUT:   %Self.loc5_31.2 => constants.%Self.a10
// CHECK:STDOUT:   %A.F.type => constants.%A.F.type.065
// CHECK:STDOUT:   %A.F => constants.%A.F.25d
// CHECK:STDOUT:   %A.assoc_type => constants.%A.assoc_type.2f5
// CHECK:STDOUT:   %assoc0.loc6_47.2 => constants.%assoc0.340
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @tuple.type.as.A.impl(constants.%V1, constants.%V2, constants.%W) {
// CHECK:STDOUT:   %V1.loc14_14.1 => constants.%V1
// CHECK:STDOUT:   %V2.loc14_33.1 => constants.%V2
// CHECK:STDOUT:   %W.loc14_52.1 => constants.%W
// CHECK:STDOUT:   %tuple => constants.%tuple.9a1
// CHECK:STDOUT:   %V1.binding.as_type => constants.%V1.binding.as_type
// CHECK:STDOUT:   %V2.binding.as_type => constants.%V2.binding.as_type
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.22e
// CHECK:STDOUT:   %A.type.loc14_85.1 => constants.%A.type.ced
// CHECK:STDOUT:   %require_complete => constants.%require_complete.38a
// CHECK:STDOUT:   %A.impl_witness => constants.%A.impl_witness.27e
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type => constants.%tuple.type.as.A.impl.F.type.e6d
// CHECK:STDOUT:   %tuple.type.as.A.impl.F => constants.%tuple.type.as.A.impl.F.920
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @tuple.type.as.A.impl.F(constants.%V1, constants.%V2, constants.%W, constants.%U.49a) {
// CHECK:STDOUT:   %U.loc17_8.1 => constants.%U.49a
// CHECK:STDOUT:   %U.binding.as_type => constants.%U.binding.as_type.ab9
// CHECK:STDOUT:   %pattern_type.loc17_26 => constants.%pattern_type.499
// CHECK:STDOUT:   %W => constants.%W
// CHECK:STDOUT:   %V1 => constants.%V1
// CHECK:STDOUT:   %V2 => constants.%V2
// CHECK:STDOUT:   %tuple.loc17_46 => constants.%tuple.9a1
// CHECK:STDOUT:   %tuple.loc17_50 => constants.%tuple.f66
// CHECK:STDOUT:   %W.binding.as_type => constants.%W.binding.as_type
// CHECK:STDOUT:   %V1.binding.as_type => constants.%V1.binding.as_type
// CHECK:STDOUT:   %V2.binding.as_type => constants.%V2.binding.as_type
// CHECK:STDOUT:   %tuple.type.loc17_50.1 => constants.%tuple.type.22e
// CHECK:STDOUT:   %tuple.type.loc17_50.2 => constants.%tuple.type.c11
// CHECK:STDOUT:   %pattern_type.loc17_32 => constants.%pattern_type.b37
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc17_32 => constants.%require_complete.921
// CHECK:STDOUT:   %require_complete.loc17_27 => constants.%require_complete.f93
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type => constants.%tuple.type.as.A.impl.F.type.e6d
// CHECK:STDOUT:   %tuple.type.as.A.impl.F => constants.%tuple.type.as.A.impl.F.920
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.specific_fn.loc18_12.2 => constants.%tuple.type.as.A.impl.F.specific_fn.fb2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.F(constants.%W, constants.%A.facet.13e, constants.%U.49a) {
// CHECK:STDOUT:   %U.loc6_8.1 => constants.%U.49a
// CHECK:STDOUT:   %U.binding.as_type => constants.%U.binding.as_type.ab9
// CHECK:STDOUT:   %pattern_type.loc6_26 => constants.%pattern_type.499
// CHECK:STDOUT:   %T => constants.%W
// CHECK:STDOUT:   %A.type => constants.%A.type.ced
// CHECK:STDOUT:   %Self => constants.%A.facet.13e
// CHECK:STDOUT:   %tuple.type.loc6_46.1 => constants.%tuple.type.5a5
// CHECK:STDOUT:   %tuple => constants.%tuple.13b
// CHECK:STDOUT:   %T.binding.as_type => constants.%W.binding.as_type
// CHECK:STDOUT:   %Self.binding.as_type => constants.%tuple.type.22e
// CHECK:STDOUT:   %tuple.type.loc6_46.2 => constants.%tuple.type.c11
// CHECK:STDOUT:   %pattern_type.loc6_32 => constants.%pattern_type.b37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%Destroy.facet.08b) {
// CHECK:STDOUT:   %T.loc5_13.1 => constants.%Destroy.facet.08b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.type => constants.%A.type.c3a
// CHECK:STDOUT:   %Self.loc5_31.2 => constants.%Self.369
// CHECK:STDOUT:   %A.F.type => constants.%A.F.type.1cf
// CHECK:STDOUT:   %A.F => constants.%A.F.d66
// CHECK:STDOUT:   %A.assoc_type => constants.%A.assoc_type.0c3
// CHECK:STDOUT:   %assoc0.loc6_47.2 => constants.%assoc0.43e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @tuple.type.as.A.impl(constants.%Destroy.facet.337, constants.%Destroy.facet.f98, constants.%Destroy.facet.08b) {
// CHECK:STDOUT:   %V1.loc14_14.1 => constants.%Destroy.facet.337
// CHECK:STDOUT:   %V2.loc14_33.1 => constants.%Destroy.facet.f98
// CHECK:STDOUT:   %W.loc14_52.1 => constants.%Destroy.facet.08b
// CHECK:STDOUT:   %tuple => constants.%tuple.bb2
// CHECK:STDOUT:   %V1.binding.as_type => constants.%Y1
// CHECK:STDOUT:   %V2.binding.as_type => constants.%Y2
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.a46
// CHECK:STDOUT:   %A.type.loc14_85.1 => constants.%A.type.c3a
// CHECK:STDOUT:   %require_complete => constants.%complete_type.309
// CHECK:STDOUT:   %A.impl_witness => constants.%A.impl_witness.ccf
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type => constants.%tuple.type.as.A.impl.F.type.c24
// CHECK:STDOUT:   %tuple.type.as.A.impl.F => constants.%tuple.type.as.A.impl.F.ae8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @tuple.type.as.A.impl.F(constants.%Destroy.facet.337, constants.%Destroy.facet.f98, constants.%Destroy.facet.08b, constants.%Destroy.facet.376) {
// CHECK:STDOUT:   %U.loc17_8.1 => constants.%Destroy.facet.376
// CHECK:STDOUT:   %U.binding.as_type => constants.%Z
// CHECK:STDOUT:   %pattern_type.loc17_26 => constants.%pattern_type.8f9
// CHECK:STDOUT:   %W => constants.%Destroy.facet.08b
// CHECK:STDOUT:   %V1 => constants.%Destroy.facet.337
// CHECK:STDOUT:   %V2 => constants.%Destroy.facet.f98
// CHECK:STDOUT:   %tuple.loc17_46 => constants.%tuple.bb2
// CHECK:STDOUT:   %tuple.loc17_50 => constants.%tuple.399
// CHECK:STDOUT:   %W.binding.as_type => constants.%X
// CHECK:STDOUT:   %V1.binding.as_type => constants.%Y1
// CHECK:STDOUT:   %V2.binding.as_type => constants.%Y2
// CHECK:STDOUT:   %tuple.type.loc17_50.1 => constants.%tuple.type.a46
// CHECK:STDOUT:   %tuple.type.loc17_50.2 => constants.%tuple.type.415
// CHECK:STDOUT:   %pattern_type.loc17_32 => constants.%pattern_type.0b2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc17_32 => constants.%complete_type.aa8
// CHECK:STDOUT:   %require_complete.loc17_27 => constants.%complete_type.357
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type => constants.%tuple.type.as.A.impl.F.type.c24
// CHECK:STDOUT:   %tuple.type.as.A.impl.F => constants.%tuple.type.as.A.impl.F.ae8
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.specific_fn.loc18_12.2 => constants.%tuple.type.as.A.impl.F.specific_fn.66c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGeneric(constants.%T.c3e) {
// CHECK:STDOUT:   %T.loc27_16.1 => constants.%T.c3e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.F(constants.%Destroy.facet.08b, constants.%A.facet.baa, constants.%Destroy.facet.376) {
// CHECK:STDOUT:   %U.loc6_8.1 => constants.%Destroy.facet.376
// CHECK:STDOUT:   %U.binding.as_type => constants.%Z
// CHECK:STDOUT:   %pattern_type.loc6_26 => constants.%pattern_type.8f9
// CHECK:STDOUT:   %T => constants.%Destroy.facet.08b
// CHECK:STDOUT:   %A.type => constants.%A.type.c3a
// CHECK:STDOUT:   %Self => constants.%A.facet.baa
// CHECK:STDOUT:   %tuple.type.loc6_46.1 => constants.%tuple.type.672
// CHECK:STDOUT:   %tuple => constants.%tuple.91f
// CHECK:STDOUT:   %T.binding.as_type => constants.%X
// CHECK:STDOUT:   %Self.binding.as_type => constants.%T.binding.as_type.7fc
// CHECK:STDOUT:   %tuple.type.loc6_46.2 => constants.%tuple.type.4dc
// CHECK:STDOUT:   %pattern_type.loc6_32 => constants.%pattern_type.295
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGeneric(constants.%facet_value.e6f) {
// CHECK:STDOUT:   %T.loc27_16.1 => constants.%facet_value.e6f
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.binding.as_type => constants.%tuple.type.a46
// CHECK:STDOUT:   %A.lookup_impl_witness => constants.%A.impl_witness.ccf
// CHECK:STDOUT:   %A.facet => constants.%A.facet.7e6
// CHECK:STDOUT:   %.loc28_4.3 => constants.%.dd2
// CHECK:STDOUT:   %impl.elem0.loc28_4.2 => constants.%tuple.type.as.A.impl.F.ae8
// CHECK:STDOUT:   %specific_impl_fn.loc28_4.2 => constants.%tuple.type.as.A.impl.F.specific_fn.66c
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.415
// CHECK:STDOUT:   %require_complete => constants.%complete_type.aa8
// CHECK:STDOUT:   %facet_value => constants.%facet_value.71a
// CHECK:STDOUT:   %.loc28_12.7 => constants.%.a06
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.c75
// CHECK:STDOUT:   %Destroy.facet.loc28_12.4 => constants.%Destroy.facet.b37
// CHECK:STDOUT:   %.loc28_12.8 => constants.%.322
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type => constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.type.5cd
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op => constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.472
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.loc28 => constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.12b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.F(constants.%Destroy.facet.08b, constants.%A.facet.7e6, constants.%Destroy.facet.376) {
// CHECK:STDOUT:   %U.loc6_8.1 => constants.%Destroy.facet.376
// CHECK:STDOUT:   %U.binding.as_type => constants.%Z
// CHECK:STDOUT:   %pattern_type.loc6_26 => constants.%pattern_type.8f9
// CHECK:STDOUT:   %T => constants.%Destroy.facet.08b
// CHECK:STDOUT:   %A.type => constants.%A.type.c3a
// CHECK:STDOUT:   %Self => constants.%A.facet.7e6
// CHECK:STDOUT:   %tuple.type.loc6_46.1 => constants.%tuple.type.672
// CHECK:STDOUT:   %tuple => constants.%tuple.a17
// CHECK:STDOUT:   %T.binding.as_type => constants.%X
// CHECK:STDOUT:   %Self.binding.as_type => constants.%tuple.type.a46
// CHECK:STDOUT:   %tuple.type.loc6_46.2 => constants.%tuple.type.415
// CHECK:STDOUT:   %pattern_type.loc6_32 => constants.%pattern_type.0b2
// CHECK:STDOUT: }
// CHECK:STDOUT:

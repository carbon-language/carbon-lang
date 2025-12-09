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

fn CallGeneric(T:! A(X)) {
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

fn CallGeneric(T:! A(X)) {
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
// CHECK:STDOUT:   %T.d9f: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %A.type.495: type = generic_interface_type @A [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %A.generic: %A.type.495 = struct_value () [concrete]
// CHECK:STDOUT:   %A.type.c1c: type = facet_type <@A, @A(%T.d9f)> [symbolic]
// CHECK:STDOUT:   %Self.27c: %A.type.c1c = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %U.63a: type = symbolic_binding U, 2 [symbolic]
// CHECK:STDOUT:   %ptr.e49: type = ptr_type %U.63a [symbolic]
// CHECK:STDOUT:   %pattern_type.854: type = pattern_type %ptr.e49 [symbolic]
// CHECK:STDOUT:   %tuple.type.004: type = tuple_type (type, %A.type.c1c, type) [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.410: type = symbolic_binding_type Self, 1, %Self.27c [symbolic]
// CHECK:STDOUT:   %tuple.type.235: type = tuple_type (%T.d9f, %Self.binding.as_type.410, %ptr.e49) [symbolic]
// CHECK:STDOUT:   %pattern_type.85c: type = pattern_type %tuple.type.235 [symbolic]
// CHECK:STDOUT:   %A.F.type.de8: type = fn_type @A.F, @A(%T.d9f) [symbolic]
// CHECK:STDOUT:   %A.F.bd2: %A.F.type.de8 = struct_value () [symbolic]
// CHECK:STDOUT:   %A.assoc_type.23a: type = assoc_entity_type @A, @A(%T.d9f) [symbolic]
// CHECK:STDOUT:   %assoc0.1c7: %A.assoc_type.23a = assoc_entity element0, @A.%A.F.decl [symbolic]
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Y: type = class_type @Y [concrete]
// CHECK:STDOUT:   %Z: type = class_type @Z [concrete]
// CHECK:STDOUT:   %A.type.0a4: type = facet_type <@A, @A(%X)> [concrete]
// CHECK:STDOUT:   %Self.789: %A.type.0a4 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %A.F.type.13d: type = fn_type @A.F, @A(%X) [concrete]
// CHECK:STDOUT:   %A.F.d83: %A.F.type.13d = struct_value () [concrete]
// CHECK:STDOUT:   %A.assoc_type.296: type = assoc_entity_type @A, @A(%X) [concrete]
// CHECK:STDOUT:   %assoc0.6a1: %A.assoc_type.296 = assoc_entity element0, @A.%A.F.decl [concrete]
// CHECK:STDOUT:   %A.impl_witness: <witness> = impl_witness file.%A.impl_witness_table [concrete]
// CHECK:STDOUT:   %U.d9f: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %ptr.4f0b5c.1: type = ptr_type %U.d9f [symbolic]
// CHECK:STDOUT:   %pattern_type.a603a0.1: type = pattern_type %ptr.4f0b5c.1 [symbolic]
// CHECK:STDOUT:   %tuple.type.ff9: type = tuple_type (type, type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.576: type = tuple_type (%X, %Y, %ptr.4f0b5c.1) [symbolic]
// CHECK:STDOUT:   %pattern_type.e0b: type = pattern_type %tuple.type.576 [symbolic]
// CHECK:STDOUT:   %Y.as.A.impl.F.type: type = fn_type @Y.as.A.impl.F [concrete]
// CHECK:STDOUT:   %Y.as.A.impl.F: %Y.as.A.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %A.facet: %A.type.0a4 = facet_value %Y, (%A.impl_witness) [concrete]
// CHECK:STDOUT:   %tuple.type.5a1: type = tuple_type (type, %A.type.0a4, type) [concrete]
// CHECK:STDOUT:   %require_complete.dc0: <witness> = require_complete_type %tuple.type.576 [symbolic]
// CHECK:STDOUT:   %require_complete.482d3f.1: <witness> = require_complete_type %ptr.4f0b5c.1 [symbolic]
// CHECK:STDOUT:   %tuple.type.f88: type = tuple_type (%empty_struct_type, %empty_struct_type, %ptr.4f0b5c.1) [symbolic]
// CHECK:STDOUT:   %X.val: %X = struct_value () [concrete]
// CHECK:STDOUT:   %Y.val: %Y = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.75bcbe.1: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.d9f) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.692ac2.1: %ptr.as.Copy.impl.Op.type.75bcbe.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.c66: <witness> = lookup_impl_witness %ptr.4f0b5c.1, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.facet.747: %Copy.type = facet_value %ptr.4f0b5c.1, (%Copy.lookup_impl_witness.c66) [symbolic]
// CHECK:STDOUT:   %.89d: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.747 [symbolic]
// CHECK:STDOUT:   %impl.elem0.928: %.89d = impl_witness_access %Copy.lookup_impl_witness.c66, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.6a1: <specific function> = specific_impl_function %impl.elem0.928, @Copy.Op(%Copy.facet.747) [symbolic]
// CHECK:STDOUT:   %Call.type: type = fn_type @Call [concrete]
// CHECK:STDOUT:   %Call: %Call.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.8f9: type = pattern_type %Z [concrete]
// CHECK:STDOUT:   %.d35: type = fn_type_with_self_type %A.F.type.13d, %A.facet [concrete]
// CHECK:STDOUT:   %ptr.fb6: type = ptr_type %Z [concrete]
// CHECK:STDOUT:   %pattern_type.f76: type = pattern_type %ptr.fb6 [concrete]
// CHECK:STDOUT:   %tuple.type.092: type = tuple_type (%X, %Y, %ptr.fb6) [concrete]
// CHECK:STDOUT:   %pattern_type.48d: type = pattern_type %tuple.type.092 [concrete]
// CHECK:STDOUT:   %Y.as.A.impl.F.specific_fn: <specific function> = specific_function %Y.as.A.impl.F, @Y.as.A.impl.F(%Z) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %DestroyT: %type_where = symbolic_binding DestroyT, 0 [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.15c: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%DestroyT) [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.fd5: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.15c = struct_value () [symbolic]
// CHECK:STDOUT:   %facet_value.352: %type_where = facet_value %tuple.type.092, () [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.cc1: <witness> = impl_witness imports.%Destroy.impl_witness_table, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.352) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.cee: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.352) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.056: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.cee = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.5df: type = ptr_type %tuple.type.092 [concrete]
// CHECK:STDOUT:   %complete_type.aab: <witness> = complete_type_witness %ptr.5df [concrete]
// CHECK:STDOUT:   %Destroy.facet.272: %Destroy.type = facet_value %tuple.type.092, (%Destroy.impl_witness.cc1) [concrete]
// CHECK:STDOUT:   %.30d: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet.272 [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.7b8: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op.056, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value.352) [concrete]
// CHECK:STDOUT:   %facet_value.4e3: %type_where = facet_value %Z, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.d83: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.4e3) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.2cc: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.d83 = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type.d3e: <witness> = complete_type_witness %ptr.fb6 [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.cb3: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op.2cc, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value.4e3) [concrete]
// CHECK:STDOUT:   %T.fb1: %A.type.0a4 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.809: type = pattern_type %A.type.0a4 [concrete]
// CHECK:STDOUT:   %CallGeneric.type: type = fn_type @CallGeneric [concrete]
// CHECK:STDOUT:   %CallGeneric: %CallGeneric.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.binding.as_type.3a2: type = symbolic_binding_type T, 0, %T.fb1 [symbolic]
// CHECK:STDOUT:   %A.lookup_impl_witness: <witness> = lookup_impl_witness %T.fb1, @A, @A(%X) [symbolic]
// CHECK:STDOUT:   %.6ed: type = fn_type_with_self_type %A.F.type.13d, %T.fb1 [symbolic]
// CHECK:STDOUT:   %impl.elem0.9bb: %.6ed = impl_witness_access %A.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %tuple.type.86a: type = tuple_type (%X, %T.binding.as_type.3a2, %ptr.fb6) [symbolic]
// CHECK:STDOUT:   %pattern_type.bf2: type = pattern_type %tuple.type.86a [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.0b2: <specific function> = specific_impl_function %impl.elem0.9bb, @A.F(%X, %T.fb1, %Z) [symbolic]
// CHECK:STDOUT:   %require_complete.93d: <witness> = require_complete_type %tuple.type.86a [symbolic]
// CHECK:STDOUT:   %facet_value.a42: %type_where = facet_value %tuple.type.86a, () [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.7b0: <witness> = impl_witness imports.%Destroy.impl_witness_table, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.a42) [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.c7f: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.a42) [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.8d9: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.c7f = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.0f5: type = ptr_type %tuple.type.86a [symbolic]
// CHECK:STDOUT:   %require_complete.536: <witness> = require_complete_type %ptr.0f5 [symbolic]
// CHECK:STDOUT:   %Destroy.facet.ac7: %Destroy.type = facet_value %tuple.type.86a, (%Destroy.impl_witness.7b0) [symbolic]
// CHECK:STDOUT:   %.47a: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet.ac7 [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.b2a: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op.8d9, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value.a42) [symbolic]
// CHECK:STDOUT:   %CallIndirect.type: type = fn_type @CallIndirect [concrete]
// CHECK:STDOUT:   %CallIndirect: %CallIndirect.type = struct_value () [concrete]
// CHECK:STDOUT:   %CallGeneric.specific_fn: <specific function> = specific_function %CallGeneric, @CallGeneric(%A.facet) [concrete]
// CHECK:STDOUT:   %complete_type.05d: <witness> = complete_type_witness %tuple.type.092 [concrete]
// CHECK:STDOUT:   %tuple.type.953: type = tuple_type (%empty_struct_type, %empty_struct_type, %ptr.fb6) [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.f1c: <witness> = impl_witness imports.%Copy.impl_witness_table.67d, @ptr.as.Copy.impl(%Z) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.31e: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%Z) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.525: %ptr.as.Copy.impl.Op.type.31e = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.347: %Copy.type = facet_value %ptr.fb6, (%Copy.impl_witness.f1c) [concrete]
// CHECK:STDOUT:   %.90e: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.347 [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.525, @ptr.as.Copy.impl.Op(%Z) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.659: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.75bcbe.1) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.692ac2.1)]
// CHECK:STDOUT:   %Copy.impl_witness_table.67d = impl_witness_table (%Core.import_ref.659), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.5c4: @DestroyT.binding.as_type.as.Destroy.impl.%DestroyT.binding.as_type.as.Destroy.impl.Op.type (%DestroyT.binding.as_type.as.Destroy.impl.Op.type.15c) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @DestroyT.binding.as_type.as.Destroy.impl.%DestroyT.binding.as_type.as.Destroy.impl.Op (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.fd5)]
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (%Core.import_ref.5c4), @DestroyT.binding.as_type.as.Destroy.impl [concrete]
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
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc5_13.2: type = symbolic_binding T, 0 [symbolic = %T.loc5_13.1 (constants.%T.d9f)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.decl: type = class_decl @X [concrete = constants.%X] {} {}
// CHECK:STDOUT:   %Y.decl: type = class_decl @Y [concrete = constants.%Y] {} {}
// CHECK:STDOUT:   %Z.decl: type = class_decl @Z [concrete = constants.%Z] {} {}
// CHECK:STDOUT:   impl_decl @Y.as.A.impl [concrete] {} {
// CHECK:STDOUT:     %Y.ref: type = name_ref Y, file.%Y.decl [concrete = constants.%Y]
// CHECK:STDOUT:     %A.ref: %A.type.495 = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %A.type: type = facet_type <@A, @A(constants.%X)> [concrete = constants.%A.type.0a4]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.impl_witness_table = impl_witness_table (@Y.as.A.impl.%Y.as.A.impl.F.decl), @Y.as.A.impl [concrete]
// CHECK:STDOUT:   %A.impl_witness: <witness> = impl_witness %A.impl_witness_table [concrete = constants.%A.impl_witness]
// CHECK:STDOUT:   %Call.decl: %Call.type = fn_decl @Call [concrete = constants.%Call] {} {}
// CHECK:STDOUT:   %CallGeneric.decl: %CallGeneric.type = fn_decl @CallGeneric [concrete = constants.%CallGeneric] {
// CHECK:STDOUT:     %T.patt: %pattern_type.809 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc26: type = splice_block %A.type [concrete = constants.%A.type.0a4] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %A.ref: %A.type.495 = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:       %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:       %A.type: type = facet_type <@A, @A(constants.%X)> [concrete = constants.%A.type.0a4]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc26_16.2: %A.type.0a4 = symbolic_binding T, 0 [symbolic = %T.loc26_16.1 (constants.%T.fb1)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallIndirect.decl: %CallIndirect.type = fn_decl @CallIndirect [concrete = constants.%CallIndirect] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @A(%T.loc5_13.2: type) {
// CHECK:STDOUT:   %T.loc5_13.1: type = symbolic_binding T, 0 [symbolic = %T.loc5_13.1 (constants.%T.d9f)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.type: type = facet_type <@A, @A(%T.loc5_13.1)> [symbolic = %A.type (constants.%A.type.c1c)]
// CHECK:STDOUT:   %Self.loc5_23.2: @A.%A.type (%A.type.c1c) = symbolic_binding Self, 1 [symbolic = %Self.loc5_23.2 (constants.%Self.27c)]
// CHECK:STDOUT:   %A.F.type: type = fn_type @A.F, @A(%T.loc5_13.1) [symbolic = %A.F.type (constants.%A.F.type.de8)]
// CHECK:STDOUT:   %A.F: @A.%A.F.type (%A.F.type.de8) = struct_value () [symbolic = %A.F (constants.%A.F.bd2)]
// CHECK:STDOUT:   %A.assoc_type: type = assoc_entity_type @A, @A(%T.loc5_13.1) [symbolic = %A.assoc_type (constants.%A.assoc_type.23a)]
// CHECK:STDOUT:   %assoc0.loc6_41.2: @A.%A.assoc_type (%A.assoc_type.23a) = assoc_entity element0, %A.F.decl [symbolic = %assoc0.loc6_41.2 (constants.%assoc0.1c7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc5_23.1: @A.%A.type (%A.type.c1c) = symbolic_binding Self, 1 [symbolic = %Self.loc5_23.2 (constants.%Self.27c)]
// CHECK:STDOUT:     %A.F.decl: @A.%A.F.type (%A.F.type.de8) = fn_decl @A.F [symbolic = @A.%A.F (constants.%A.F.bd2)] {
// CHECK:STDOUT:       %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 2 [concrete]
// CHECK:STDOUT:       %u.patt: @A.F.%pattern_type.loc6_18 (%pattern_type.854) = value_binding_pattern u [concrete]
// CHECK:STDOUT:       %u.param_patt: @A.F.%pattern_type.loc6_18 (%pattern_type.854) = value_param_pattern %u.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @A.F.%pattern_type.loc6_25 (%pattern_type.85c) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @A.F.%pattern_type.loc6_25 (%pattern_type.85c) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @A.%T.loc5_13.2 [symbolic = %T (constants.%T.d9f)]
// CHECK:STDOUT:       %.loc6_32: @A.F.%A.type (%A.type.c1c) = specific_constant @A.%Self.loc5_23.1, @A(constants.%T.d9f) [symbolic = %Self (constants.%Self.27c)]
// CHECK:STDOUT:       %Self.ref: @A.F.%A.type (%A.type.c1c) = name_ref Self, %.loc6_32 [symbolic = %Self (constants.%Self.27c)]
// CHECK:STDOUT:       %U.ref.loc6_38: type = name_ref U, %U.loc6_8.2 [symbolic = %U.loc6_8.1 (constants.%U.63a)]
// CHECK:STDOUT:       %ptr.loc6_39: type = ptr_type %U.ref.loc6_38 [symbolic = %ptr.loc6_22.1 (constants.%ptr.e49)]
// CHECK:STDOUT:       %.loc6_40.1: @A.F.%tuple.type.loc6_40.1 (%tuple.type.004) = tuple_literal (%T.ref, %Self.ref, %ptr.loc6_39)
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.410)]
// CHECK:STDOUT:       %.loc6_40.2: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.410)]
// CHECK:STDOUT:       %.loc6_40.3: type = converted %.loc6_40.1, constants.%tuple.type.235 [symbolic = %tuple.type.loc6_40.2 (constants.%tuple.type.235)]
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %U.loc6_8.2: type = symbolic_binding U, 2 [symbolic = %U.loc6_8.1 (constants.%U.63a)]
// CHECK:STDOUT:       %u.param: @A.F.%ptr.loc6_22.1 (%ptr.e49) = value_param call_param0
// CHECK:STDOUT:       %.loc6_22: type = splice_block %ptr.loc6_22.2 [symbolic = %ptr.loc6_22.1 (constants.%ptr.e49)] {
// CHECK:STDOUT:         %U.ref.loc6_21: type = name_ref U, %U.loc6_8.2 [symbolic = %U.loc6_8.1 (constants.%U.63a)]
// CHECK:STDOUT:         %ptr.loc6_22.2: type = ptr_type %U.ref.loc6_21 [symbolic = %ptr.loc6_22.1 (constants.%ptr.e49)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %u: @A.F.%ptr.loc6_22.1 (%ptr.e49) = value_binding u, %u.param
// CHECK:STDOUT:       %return.param: ref @A.F.%tuple.type.loc6_40.2 (%tuple.type.235) = out_param call_param1
// CHECK:STDOUT:       %return: ref @A.F.%tuple.type.loc6_40.2 (%tuple.type.235) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc6_41.1: @A.%A.assoc_type (%A.assoc_type.23a) = assoc_entity element0, %A.F.decl [symbolic = %assoc0.loc6_41.2 (constants.%assoc0.1c7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc5_23.1
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
// CHECK:STDOUT:     %.loc16_37.1: %tuple.type.ff9 = tuple_literal (%X.ref, %Y.ref, %ptr.loc16_36)
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
// CHECK:STDOUT: generic fn @A.F(@A.%T.loc5_13.2: type, @A.%Self.loc5_23.1: @A.%A.type (%A.type.c1c), %U.loc6_8.2: type) {
// CHECK:STDOUT:   %U.loc6_8.1: type = symbolic_binding U, 2 [symbolic = %U.loc6_8.1 (constants.%U.63a)]
// CHECK:STDOUT:   %ptr.loc6_22.1: type = ptr_type %U.loc6_8.1 [symbolic = %ptr.loc6_22.1 (constants.%ptr.e49)]
// CHECK:STDOUT:   %pattern_type.loc6_18: type = pattern_type %ptr.loc6_22.1 [symbolic = %pattern_type.loc6_18 (constants.%pattern_type.854)]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T.d9f)]
// CHECK:STDOUT:   %A.type: type = facet_type <@A, @A(%T)> [symbolic = %A.type (constants.%A.type.c1c)]
// CHECK:STDOUT:   %Self: @A.F.%A.type (%A.type.c1c) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.27c)]
// CHECK:STDOUT:   %tuple.type.loc6_40.1: type = tuple_type (type, %A.type, type) [symbolic = %tuple.type.loc6_40.1 (constants.%tuple.type.004)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.410)]
// CHECK:STDOUT:   %tuple.type.loc6_40.2: type = tuple_type (%T, %Self.binding.as_type, %ptr.loc6_22.1) [symbolic = %tuple.type.loc6_40.2 (constants.%tuple.type.235)]
// CHECK:STDOUT:   %pattern_type.loc6_25: type = pattern_type %tuple.type.loc6_40.2 [symbolic = %pattern_type.loc6_25 (constants.%pattern_type.85c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param: @A.F.%ptr.loc6_22.1 (%ptr.e49)) -> @A.F.%tuple.type.loc6_40.2 (%tuple.type.235);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Y.as.A.impl.F(%U.loc16_8.2: type) {
// CHECK:STDOUT:   %U.loc16_8.1: type = symbolic_binding U, 0 [symbolic = %U.loc16_8.1 (constants.%U.d9f)]
// CHECK:STDOUT:   %ptr.loc16_22.1: type = ptr_type %U.loc16_8.1 [symbolic = %ptr.loc16_22.1 (constants.%ptr.4f0b5c.1)]
// CHECK:STDOUT:   %pattern_type.loc16_18: type = pattern_type %ptr.loc16_22.1 [symbolic = %pattern_type.loc16_18 (constants.%pattern_type.a603a0.1)]
// CHECK:STDOUT:   %tuple.type.loc16: type = tuple_type (constants.%X, constants.%Y, %ptr.loc16_22.1) [symbolic = %tuple.type.loc16 (constants.%tuple.type.576)]
// CHECK:STDOUT:   %pattern_type.loc16_25: type = pattern_type %tuple.type.loc16 [symbolic = %pattern_type.loc16_25 (constants.%pattern_type.e0b)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc16_25: <witness> = require_complete_type %tuple.type.loc16 [symbolic = %require_complete.loc16_25 (constants.%require_complete.dc0)]
// CHECK:STDOUT:   %require_complete.loc16_19: <witness> = require_complete_type %ptr.loc16_22.1 [symbolic = %require_complete.loc16_19 (constants.%require_complete.482d3f.1)]
// CHECK:STDOUT:   %tuple.type.loc17: type = tuple_type (constants.%empty_struct_type, constants.%empty_struct_type, %ptr.loc16_22.1) [symbolic = %tuple.type.loc17 (constants.%tuple.type.f88)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr.loc16_22.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.c66)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.loc16_22.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet.747)]
// CHECK:STDOUT:   %.loc17_21.2: type = fn_type_with_self_type constants.%Copy.Op.type, %Copy.facet [symbolic = %.loc17_21.2 (constants.%.89d)]
// CHECK:STDOUT:   %impl.elem0.loc17_21.2: @Y.as.A.impl.F.%.loc17_21.2 (%.89d) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc17_21.2 (constants.%impl.elem0.928)]
// CHECK:STDOUT:   %specific_impl_fn.loc17_21.2: <specific function> = specific_impl_function %impl.elem0.loc17_21.2, @Copy.Op(%Copy.facet) [symbolic = %specific_impl_fn.loc17_21.2 (constants.%specific_impl_fn.6a1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param: @Y.as.A.impl.F.%ptr.loc16_22.1 (%ptr.4f0b5c.1)) -> %return.param: @Y.as.A.impl.F.%tuple.type.loc16 (%tuple.type.576) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc17_14.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc17_18.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %u.ref: @Y.as.A.impl.F.%ptr.loc16_22.1 (%ptr.4f0b5c.1) = name_ref u, %u
// CHECK:STDOUT:     %.loc17_22.1: @Y.as.A.impl.F.%tuple.type.loc17 (%tuple.type.f88) = tuple_literal (%.loc17_14.1, %.loc17_18.1, %u.ref)
// CHECK:STDOUT:     %tuple.elem0: ref %X = tuple_access %return, element0
// CHECK:STDOUT:     %.loc17_14.2: init %X = class_init (), %tuple.elem0 [concrete = constants.%X.val]
// CHECK:STDOUT:     %.loc17_22.2: init %X = converted %.loc17_14.1, %.loc17_14.2 [concrete = constants.%X.val]
// CHECK:STDOUT:     %tuple.elem1: ref %Y = tuple_access %return, element1
// CHECK:STDOUT:     %.loc17_18.2: init %Y = class_init (), %tuple.elem1 [concrete = constants.%Y.val]
// CHECK:STDOUT:     %.loc17_22.3: init %Y = converted %.loc17_18.1, %.loc17_18.2 [concrete = constants.%Y.val]
// CHECK:STDOUT:     %impl.elem0.loc17_21.1: @Y.as.A.impl.F.%.loc17_21.2 (%.89d) = impl_witness_access constants.%Copy.lookup_impl_witness.c66, element0 [symbolic = %impl.elem0.loc17_21.2 (constants.%impl.elem0.928)]
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
// CHECK:STDOUT:   %A.type: type = facet_type <@A, @A(constants.%X)> [concrete = constants.%A.type.0a4]
// CHECK:STDOUT:   %A.facet: %A.type.0a4 = facet_value %Y.ref, (constants.%A.impl_witness) [concrete = constants.%A.facet]
// CHECK:STDOUT:   %.loc23_6: %A.type.0a4 = converted %Y.ref, %A.facet [concrete = constants.%A.facet]
// CHECK:STDOUT:   %.loc23_14.1: %A.assoc_type.296 = specific_constant @A.%assoc0.loc6_41.1, @A(constants.%X) [concrete = constants.%assoc0.6a1]
// CHECK:STDOUT:   %F.ref: %A.assoc_type.296 = name_ref F, %.loc23_14.1 [concrete = constants.%assoc0.6a1]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc23_6 [concrete = constants.%Y]
// CHECK:STDOUT:   %.loc23_14.2: type = converted %.loc23_6, %as_type [concrete = constants.%Y]
// CHECK:STDOUT:   %impl.elem0: %.d35 = impl_witness_access constants.%A.impl_witness, element0 [concrete = constants.%Y.as.A.impl.F]
// CHECK:STDOUT:   %Z.ref.loc23: type = name_ref Z, file.%Z.decl [concrete = constants.%Z]
// CHECK:STDOUT:   %u.ref: ref %Z = name_ref u, %u
// CHECK:STDOUT:   %addr.loc23_20: %ptr.fb6 = addr_of %u.ref
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Y.as.A.impl.F(constants.%Z) [concrete = constants.%Y.as.A.impl.F.specific_fn]
// CHECK:STDOUT:   %.loc23_22.1: ref %tuple.type.092 = temporary_storage
// CHECK:STDOUT:   %Y.as.A.impl.F.call: init %tuple.type.092 = call %specific_fn(%addr.loc23_20) to %.loc23_22.1
// CHECK:STDOUT:   %.loc23_22.2: ref %tuple.type.092 = temporary %.loc23_22.1, %Y.as.A.impl.F.call
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc23: <bound method> = bound_method %.loc23_22.2, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.056
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.056, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.352) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.7b8]
// CHECK:STDOUT:   %bound_method.loc23: <bound method> = bound_method %.loc23_22.2, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc23_22: %ptr.5df = addr_of %.loc23_22.2
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc23: init %empty_tuple.type = call %bound_method.loc23(%addr.loc23_22)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc22: <bound method> = bound_method %u.var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.2cc
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.2: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.2cc, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.4e3) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.cb3]
// CHECK:STDOUT:   %bound_method.loc22: <bound method> = bound_method %u.var, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc22: %ptr.fb6 = addr_of %u.var
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc22: init %empty_tuple.type = call %bound_method.loc22(%addr.loc22)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallGeneric(%T.loc26_16.2: %A.type.0a4) {
// CHECK:STDOUT:   %T.loc26_16.1: %A.type.0a4 = symbolic_binding T, 0 [symbolic = %T.loc26_16.1 (constants.%T.fb1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc26_16.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.3a2)]
// CHECK:STDOUT:   %A.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc26_16.1, @A, @A(constants.%X) [symbolic = %A.lookup_impl_witness (constants.%A.lookup_impl_witness)]
// CHECK:STDOUT:   %.loc28_4.3: type = fn_type_with_self_type constants.%A.F.type.13d, %T.loc26_16.1 [symbolic = %.loc28_4.3 (constants.%.6ed)]
// CHECK:STDOUT:   %impl.elem0.loc28_4.2: @CallGeneric.%.loc28_4.3 (%.6ed) = impl_witness_access %A.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc28_4.2 (constants.%impl.elem0.9bb)]
// CHECK:STDOUT:   %specific_impl_fn.loc28_4.2: <specific function> = specific_impl_function %impl.elem0.loc28_4.2, @A.F(constants.%X, %T.loc26_16.1, constants.%Z) [symbolic = %specific_impl_fn.loc28_4.2 (constants.%specific_impl_fn.0b2)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (constants.%X, %T.binding.as_type, constants.%ptr.fb6) [symbolic = %tuple.type (constants.%tuple.type.86a)]
// CHECK:STDOUT:   %require_complete.loc28_12.1: <witness> = require_complete_type %tuple.type [symbolic = %require_complete.loc28_12.1 (constants.%require_complete.93d)]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %tuple.type, () [symbolic = %facet_value (constants.%facet_value.a42)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness imports.%Destroy.impl_witness_table, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness.7b0)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %tuple.type, (%Destroy.impl_witness) [symbolic = %Destroy.facet (constants.%Destroy.facet.ac7)]
// CHECK:STDOUT:   %.loc28_12.4: type = fn_type_with_self_type constants.%Destroy.Op.type, %Destroy.facet [symbolic = %.loc28_12.4 (constants.%.47a)]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op.type (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.type.c7f)]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op: @CallGeneric.%DestroyT.binding.as_type.as.Destroy.impl.Op.type (%DestroyT.binding.as_type.as.Destroy.impl.Op.type.c7f) = struct_value () [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.8d9)]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.loc28: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value) [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.loc28 (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.b2a)]
// CHECK:STDOUT:   %ptr: type = ptr_type %tuple.type [symbolic = %ptr (constants.%ptr.0f5)]
// CHECK:STDOUT:   %require_complete.loc28_12.2: <witness> = require_complete_type %ptr [symbolic = %require_complete.loc28_12.2 (constants.%require_complete.536)]
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
// CHECK:STDOUT:     %T.ref: %A.type.0a4 = name_ref T, %T.loc26_16.2 [symbolic = %T.loc26_16.1 (constants.%T.fb1)]
// CHECK:STDOUT:     %.loc28_4.1: %A.assoc_type.296 = specific_constant @A.%assoc0.loc6_41.1, @A(constants.%X) [concrete = constants.%assoc0.6a1]
// CHECK:STDOUT:     %F.ref: %A.assoc_type.296 = name_ref F, %.loc28_4.1 [concrete = constants.%assoc0.6a1]
// CHECK:STDOUT:     %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type.3a2)]
// CHECK:STDOUT:     %.loc28_4.2: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type.3a2)]
// CHECK:STDOUT:     %impl.elem0.loc28_4.1: @CallGeneric.%.loc28_4.3 (%.6ed) = impl_witness_access constants.%A.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc28_4.2 (constants.%impl.elem0.9bb)]
// CHECK:STDOUT:     %Z.ref.loc28: type = name_ref Z, file.%Z.decl [concrete = constants.%Z]
// CHECK:STDOUT:     %u.ref: ref %Z = name_ref u, %u
// CHECK:STDOUT:     %addr.loc28_10: %ptr.fb6 = addr_of %u.ref
// CHECK:STDOUT:     %specific_impl_fn.loc28_4.1: <specific function> = specific_impl_function %impl.elem0.loc28_4.1, @A.F(constants.%X, constants.%T.fb1, constants.%Z) [symbolic = %specific_impl_fn.loc28_4.2 (constants.%specific_impl_fn.0b2)]
// CHECK:STDOUT:     %.loc28_12.1: ref @CallGeneric.%tuple.type (%tuple.type.86a) = temporary_storage
// CHECK:STDOUT:     %.loc28_12.2: init @CallGeneric.%tuple.type (%tuple.type.86a) = call %specific_impl_fn.loc28_4.1(%addr.loc28_10) to %.loc28_12.1
// CHECK:STDOUT:     %.loc28_12.3: ref @CallGeneric.%tuple.type (%tuple.type.86a) = temporary %.loc28_12.1, %.loc28_12.2
// CHECK:STDOUT:     %impl.elem0.loc28_12: @CallGeneric.%.loc28_12.4 (%.47a) = impl_witness_access constants.%Destroy.impl_witness.7b0, element0 [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.8d9)]
// CHECK:STDOUT:     %bound_method.loc28_12.1: <bound method> = bound_method %.loc28_12.3, %impl.elem0.loc28_12
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0.loc28_12, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.a42) [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.loc28 (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.b2a)]
// CHECK:STDOUT:     %bound_method.loc28_12.2: <bound method> = bound_method %.loc28_12.3, %specific_fn
// CHECK:STDOUT:     %addr.loc28_12: @CallGeneric.%ptr (%ptr.0f5) = addr_of %.loc28_12.3
// CHECK:STDOUT:     %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc28: init %empty_tuple.type = call %bound_method.loc28_12.2(%addr.loc28_12)
// CHECK:STDOUT:     %DestroyT.binding.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %u.var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.2cc
// CHECK:STDOUT:     %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.2cc, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.4e3) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.cb3]
// CHECK:STDOUT:     %bound_method.loc27: <bound method> = bound_method %u.var, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:     %addr.loc27: %ptr.fb6 = addr_of %u.var
// CHECK:STDOUT:     %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc27: init %empty_tuple.type = call %bound_method.loc27(%addr.loc27)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallIndirect() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CallGeneric.ref: %CallGeneric.type = name_ref CallGeneric, file.%CallGeneric.decl [concrete = constants.%CallGeneric]
// CHECK:STDOUT:   %Y.ref: type = name_ref Y, file.%Y.decl [concrete = constants.%Y]
// CHECK:STDOUT:   %A.facet: %A.type.0a4 = facet_value %Y.ref, (constants.%A.impl_witness) [concrete = constants.%A.facet]
// CHECK:STDOUT:   %.loc32: %A.type.0a4 = converted %Y.ref, %A.facet [concrete = constants.%A.facet]
// CHECK:STDOUT:   %CallGeneric.specific_fn: <specific function> = specific_function %CallGeneric.ref, @CallGeneric(constants.%A.facet) [concrete = constants.%CallGeneric.specific_fn]
// CHECK:STDOUT:   %CallGeneric.call: init %empty_tuple.type = call %CallGeneric.specific_fn()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%T.d9f) {
// CHECK:STDOUT:   %T.loc5_13.1 => constants.%T.d9f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.F(constants.%T.d9f, constants.%Self.27c, constants.%U.63a) {
// CHECK:STDOUT:   %U.loc6_8.1 => constants.%U.63a
// CHECK:STDOUT:   %ptr.loc6_22.1 => constants.%ptr.e49
// CHECK:STDOUT:   %pattern_type.loc6_18 => constants.%pattern_type.854
// CHECK:STDOUT:   %T => constants.%T.d9f
// CHECK:STDOUT:   %A.type => constants.%A.type.c1c
// CHECK:STDOUT:   %Self => constants.%Self.27c
// CHECK:STDOUT:   %tuple.type.loc6_40.1 => constants.%tuple.type.004
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.410
// CHECK:STDOUT:   %tuple.type.loc6_40.2 => constants.%tuple.type.235
// CHECK:STDOUT:   %pattern_type.loc6_25 => constants.%pattern_type.85c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%X) {
// CHECK:STDOUT:   %T.loc5_13.1 => constants.%X
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.type => constants.%A.type.0a4
// CHECK:STDOUT:   %Self.loc5_23.2 => constants.%Self.789
// CHECK:STDOUT:   %A.F.type => constants.%A.F.type.13d
// CHECK:STDOUT:   %A.F => constants.%A.F.d83
// CHECK:STDOUT:   %A.assoc_type => constants.%A.assoc_type.296
// CHECK:STDOUT:   %assoc0.loc6_41.2 => constants.%assoc0.6a1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y.as.A.impl.F(constants.%U.d9f) {
// CHECK:STDOUT:   %U.loc16_8.1 => constants.%U.d9f
// CHECK:STDOUT:   %ptr.loc16_22.1 => constants.%ptr.4f0b5c.1
// CHECK:STDOUT:   %pattern_type.loc16_18 => constants.%pattern_type.a603a0.1
// CHECK:STDOUT:   %tuple.type.loc16 => constants.%tuple.type.576
// CHECK:STDOUT:   %pattern_type.loc16_25 => constants.%pattern_type.e0b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.F(constants.%X, constants.%A.facet, constants.%U.d9f) {
// CHECK:STDOUT:   %U.loc6_8.1 => constants.%U.d9f
// CHECK:STDOUT:   %ptr.loc6_22.1 => constants.%ptr.4f0b5c.1
// CHECK:STDOUT:   %pattern_type.loc6_18 => constants.%pattern_type.a603a0.1
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.0a4
// CHECK:STDOUT:   %Self => constants.%A.facet
// CHECK:STDOUT:   %tuple.type.loc6_40.1 => constants.%tuple.type.5a1
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Y
// CHECK:STDOUT:   %tuple.type.loc6_40.2 => constants.%tuple.type.576
// CHECK:STDOUT:   %pattern_type.loc6_25 => constants.%pattern_type.e0b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y.as.A.impl.F(constants.%Z) {
// CHECK:STDOUT:   %U.loc16_8.1 => constants.%Z
// CHECK:STDOUT:   %ptr.loc16_22.1 => constants.%ptr.fb6
// CHECK:STDOUT:   %pattern_type.loc16_18 => constants.%pattern_type.f76
// CHECK:STDOUT:   %tuple.type.loc16 => constants.%tuple.type.092
// CHECK:STDOUT:   %pattern_type.loc16_25 => constants.%pattern_type.48d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc16_25 => constants.%complete_type.05d
// CHECK:STDOUT:   %require_complete.loc16_19 => constants.%complete_type.d3e
// CHECK:STDOUT:   %tuple.type.loc17 => constants.%tuple.type.953
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.impl_witness.f1c
// CHECK:STDOUT:   %Copy.facet => constants.%Copy.facet.347
// CHECK:STDOUT:   %.loc17_21.2 => constants.%.90e
// CHECK:STDOUT:   %impl.elem0.loc17_21.2 => constants.%ptr.as.Copy.impl.Op.525
// CHECK:STDOUT:   %specific_impl_fn.loc17_21.2 => constants.%ptr.as.Copy.impl.Op.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGeneric(constants.%T.fb1) {
// CHECK:STDOUT:   %T.loc26_16.1 => constants.%T.fb1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.F(constants.%X, constants.%T.fb1, constants.%Z) {
// CHECK:STDOUT:   %U.loc6_8.1 => constants.%Z
// CHECK:STDOUT:   %ptr.loc6_22.1 => constants.%ptr.fb6
// CHECK:STDOUT:   %pattern_type.loc6_18 => constants.%pattern_type.f76
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.0a4
// CHECK:STDOUT:   %Self => constants.%T.fb1
// CHECK:STDOUT:   %tuple.type.loc6_40.1 => constants.%tuple.type.5a1
// CHECK:STDOUT:   %Self.binding.as_type => constants.%T.binding.as_type.3a2
// CHECK:STDOUT:   %tuple.type.loc6_40.2 => constants.%tuple.type.86a
// CHECK:STDOUT:   %pattern_type.loc6_25 => constants.%pattern_type.bf2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGeneric(constants.%A.facet) {
// CHECK:STDOUT:   %T.loc26_16.1 => constants.%A.facet
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.binding.as_type => constants.%Y
// CHECK:STDOUT:   %A.lookup_impl_witness => constants.%A.impl_witness
// CHECK:STDOUT:   %.loc28_4.3 => constants.%.d35
// CHECK:STDOUT:   %impl.elem0.loc28_4.2 => constants.%Y.as.A.impl.F
// CHECK:STDOUT:   %specific_impl_fn.loc28_4.2 => constants.%Y.as.A.impl.F.specific_fn
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.092
// CHECK:STDOUT:   %require_complete.loc28_12.1 => constants.%complete_type.05d
// CHECK:STDOUT:   %facet_value => constants.%facet_value.352
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.cc1
// CHECK:STDOUT:   %Destroy.facet => constants.%Destroy.facet.272
// CHECK:STDOUT:   %.loc28_12.4 => constants.%.30d
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type => constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.type.cee
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op => constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.056
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.loc28 => constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.7b8
// CHECK:STDOUT:   %ptr => constants.%ptr.5df
// CHECK:STDOUT:   %require_complete.loc28_12.2 => constants.%complete_type.aab
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.F(constants.%X, constants.%A.facet, constants.%Z) {
// CHECK:STDOUT:   %U.loc6_8.1 => constants.%Z
// CHECK:STDOUT:   %ptr.loc6_22.1 => constants.%ptr.fb6
// CHECK:STDOUT:   %pattern_type.loc6_18 => constants.%pattern_type.f76
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.0a4
// CHECK:STDOUT:   %Self => constants.%A.facet
// CHECK:STDOUT:   %tuple.type.loc6_40.1 => constants.%tuple.type.5a1
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
// CHECK:STDOUT:   %T.d9f: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %A.type.495: type = generic_interface_type @A [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %A.generic: %A.type.495 = struct_value () [concrete]
// CHECK:STDOUT:   %A.type.c1c: type = facet_type <@A, @A(%T.d9f)> [symbolic]
// CHECK:STDOUT:   %Self.27c: %A.type.c1c = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %U.63a: type = symbolic_binding U, 2 [symbolic]
// CHECK:STDOUT:   %pattern_type.533: type = pattern_type %U.63a [symbolic]
// CHECK:STDOUT:   %tuple.type.004: type = tuple_type (type, %A.type.c1c, type) [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.410: type = symbolic_binding_type Self, 1, %Self.27c [symbolic]
// CHECK:STDOUT:   %tuple.type.e5f: type = tuple_type (%T.d9f, %Self.binding.as_type.410, %U.63a) [symbolic]
// CHECK:STDOUT:   %pattern_type.845: type = pattern_type %tuple.type.e5f [symbolic]
// CHECK:STDOUT:   %A.F.type.de8: type = fn_type @A.F, @A(%T.d9f) [symbolic]
// CHECK:STDOUT:   %A.F.bd2: %A.F.type.de8 = struct_value () [symbolic]
// CHECK:STDOUT:   %A.assoc_type.23a: type = assoc_entity_type @A, @A(%T.d9f) [symbolic]
// CHECK:STDOUT:   %assoc0.1c7: %A.assoc_type.23a = assoc_entity element0, @A.%A.F.decl [symbolic]
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
// CHECK:STDOUT:   %tuple.type.d18: type = tuple_type (%V1, %V2) [symbolic]
// CHECK:STDOUT:   %A.type.2ec: type = facet_type <@A, @A(%W)> [symbolic]
// CHECK:STDOUT:   %Self.b6c: %A.type.2ec = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %A.F.type.f65: type = fn_type @A.F, @A(%W) [symbolic]
// CHECK:STDOUT:   %A.F.d51: %A.F.type.f65 = struct_value () [symbolic]
// CHECK:STDOUT:   %A.assoc_type.abb: type = assoc_entity_type @A, @A(%W) [symbolic]
// CHECK:STDOUT:   %assoc0.973: %A.assoc_type.abb = assoc_entity element0, @A.%A.F.decl [symbolic]
// CHECK:STDOUT:   %require_complete.4d8: <witness> = require_complete_type %A.type.2ec [symbolic]
// CHECK:STDOUT:   %A.impl_witness.2bc: <witness> = impl_witness file.%A.impl_witness_table, @tuple.type.as.A.impl(%V1, %V2, %W) [symbolic]
// CHECK:STDOUT:   %U.800: type = symbolic_binding U, 3 [symbolic]
// CHECK:STDOUT:   %pattern_type.332: type = pattern_type %U.800 [symbolic]
// CHECK:STDOUT:   %tuple.type.11f: type = tuple_type (type, %tuple.type.24b, type) [concrete]
// CHECK:STDOUT:   %tuple.type.d16: type = tuple_type (%W, %tuple.type.d18, %U.800) [symbolic]
// CHECK:STDOUT:   %pattern_type.cbb: type = pattern_type %tuple.type.d16 [symbolic]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type.172: type = fn_type @tuple.type.as.A.impl.F, @tuple.type.as.A.impl(%V1, %V2, %W) [symbolic]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.c93: %tuple.type.as.A.impl.F.type.172 = struct_value () [symbolic]
// CHECK:STDOUT:   %A.facet.6f0: %A.type.2ec = facet_value %tuple.type.d18, (%A.impl_witness.2bc) [symbolic]
// CHECK:STDOUT:   %tuple.type.e39: type = tuple_type (type, %A.type.2ec, type) [symbolic]
// CHECK:STDOUT:   %require_complete.923: <witness> = require_complete_type %tuple.type.d16 [symbolic]
// CHECK:STDOUT:   %require_complete.997: <witness> = require_complete_type %U.800 [symbolic]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.specific_fn.5bf: <specific function> = specific_function %tuple.type.as.A.impl.F.c93, @tuple.type.as.A.impl.F(%V1, %V2, %W, %U.800) [symbolic]
// CHECK:STDOUT:   %Call.type: type = fn_type @Call [concrete]
// CHECK:STDOUT:   %Call: %Call.type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.a46: type = tuple_type (%Y1, %Y2) [concrete]
// CHECK:STDOUT:   %A.type.0a4: type = facet_type <@A, @A(%X)> [concrete]
// CHECK:STDOUT:   %Self.789: %A.type.0a4 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %A.F.type.13d: type = fn_type @A.F, @A(%X) [concrete]
// CHECK:STDOUT:   %A.F.d83: %A.F.type.13d = struct_value () [concrete]
// CHECK:STDOUT:   %A.assoc_type.296: type = assoc_entity_type @A, @A(%X) [concrete]
// CHECK:STDOUT:   %assoc0.6a1: %A.assoc_type.296 = assoc_entity element0, @A.%A.F.decl [concrete]
// CHECK:STDOUT:   %complete_type.e2d: <witness> = complete_type_witness %A.type.0a4 [concrete]
// CHECK:STDOUT:   %A.impl_witness.be7: <witness> = impl_witness file.%A.impl_witness_table, @tuple.type.as.A.impl(%Y1, %Y2, %X) [concrete]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type.415: type = fn_type @tuple.type.as.A.impl.F, @tuple.type.as.A.impl(%Y1, %Y2, %X) [concrete]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.735: %tuple.type.as.A.impl.F.type.415 = struct_value () [concrete]
// CHECK:STDOUT:   %A.facet.f26: %A.type.0a4 = facet_value %tuple.type.a46, (%A.impl_witness.be7) [concrete]
// CHECK:STDOUT:   %.a2b: type = fn_type_with_self_type %A.F.type.13d, %A.facet.f26 [concrete]
// CHECK:STDOUT:   %pattern_type.8f9: type = pattern_type %Z [concrete]
// CHECK:STDOUT:   %tuple.type.415: type = tuple_type (%X, %tuple.type.a46, %Z) [concrete]
// CHECK:STDOUT:   %pattern_type.0b2: type = pattern_type %tuple.type.415 [concrete]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.specific_fn.5c5: <specific function> = specific_function %tuple.type.as.A.impl.F.735, @tuple.type.as.A.impl.F(%Y1, %Y2, %X, %Z) [concrete]
// CHECK:STDOUT:   %Z.val: %Z = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %DestroyT: %type_where = symbolic_binding DestroyT, 0 [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.15c: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%DestroyT) [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.fd5: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.15c = struct_value () [symbolic]
// CHECK:STDOUT:   %facet_value.71a: %type_where = facet_value %tuple.type.415, () [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.c75: <witness> = impl_witness imports.%Destroy.impl_witness_table, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.71a) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.5cd: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.71a) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.472: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.5cd = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.ad9: type = ptr_type %tuple.type.415 [concrete]
// CHECK:STDOUT:   %complete_type.e23: <witness> = complete_type_witness %ptr.ad9 [concrete]
// CHECK:STDOUT:   %Destroy.facet.b37: %Destroy.type = facet_value %tuple.type.415, (%Destroy.impl_witness.c75) [concrete]
// CHECK:STDOUT:   %.322: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet.b37 [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.12b: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op.472, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value.71a) [concrete]
// CHECK:STDOUT:   %facet_value.4e3: %type_where = facet_value %Z, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.d83: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.4e3) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.2cc: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.d83 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.fb6: type = ptr_type %Z [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.cb3: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op.2cc, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value.4e3) [concrete]
// CHECK:STDOUT:   %T.fb1: %A.type.0a4 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.809: type = pattern_type %A.type.0a4 [concrete]
// CHECK:STDOUT:   %CallGeneric.type: type = fn_type @CallGeneric [concrete]
// CHECK:STDOUT:   %CallGeneric: %CallGeneric.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.fb1 [symbolic]
// CHECK:STDOUT:   %A.lookup_impl_witness: <witness> = lookup_impl_witness %T.fb1, @A, @A(%X) [symbolic]
// CHECK:STDOUT:   %.6ed: type = fn_type_with_self_type %A.F.type.13d, %T.fb1 [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.6ed = impl_witness_access %A.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %tuple.type.5a1: type = tuple_type (type, %A.type.0a4, type) [concrete]
// CHECK:STDOUT:   %tuple.type.bf3: type = tuple_type (%X, %T.binding.as_type, %Z) [symbolic]
// CHECK:STDOUT:   %pattern_type.e6a: type = pattern_type %tuple.type.bf3 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @A.F(%X, %T.fb1, %Z) [symbolic]
// CHECK:STDOUT:   %require_complete.4e9: <witness> = require_complete_type %tuple.type.bf3 [symbolic]
// CHECK:STDOUT:   %facet_value.185: %type_where = facet_value %tuple.type.bf3, () [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.cd1: <witness> = impl_witness imports.%Destroy.impl_witness_table, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.185) [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.a5d: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.185) [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.a08: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.a5d = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.4ce: type = ptr_type %tuple.type.bf3 [symbolic]
// CHECK:STDOUT:   %require_complete.7d0: <witness> = require_complete_type %ptr.4ce [symbolic]
// CHECK:STDOUT:   %Destroy.facet.1b6: %Destroy.type = facet_value %tuple.type.bf3, (%Destroy.impl_witness.cd1) [symbolic]
// CHECK:STDOUT:   %.cce: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet.1b6 [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.760: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op.a08, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value.185) [symbolic]
// CHECK:STDOUT:   %CallIndirect.type: type = fn_type @CallIndirect [concrete]
// CHECK:STDOUT:   %CallIndirect: %CallIndirect.type = struct_value () [concrete]
// CHECK:STDOUT:   %CallGeneric.specific_fn: <specific function> = specific_function %CallGeneric, @CallGeneric(%A.facet.f26) [concrete]
// CHECK:STDOUT:   %complete_type.aa8: <witness> = complete_type_witness %tuple.type.415 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.5c4: @DestroyT.binding.as_type.as.Destroy.impl.%DestroyT.binding.as_type.as.Destroy.impl.Op.type (%DestroyT.binding.as_type.as.Destroy.impl.Op.type.15c) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @DestroyT.binding.as_type.as.Destroy.impl.%DestroyT.binding.as_type.as.Destroy.impl.Op (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.fd5)]
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (%Core.import_ref.5c4), @DestroyT.binding.as_type.as.Destroy.impl [concrete]
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
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc5_13.2: type = symbolic_binding T, 0 [symbolic = %T.loc5_13.1 (constants.%T.d9f)]
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
// CHECK:STDOUT:     %V1.ref: type = name_ref V1, %V1.loc14_14.2 [symbolic = %V1.loc14_14.1 (constants.%V1)]
// CHECK:STDOUT:     %V2.ref: type = name_ref V2, %V2.loc14_25.2 [symbolic = %V2.loc14_25.1 (constants.%V2)]
// CHECK:STDOUT:     %.loc14_53.1: %tuple.type.24b = tuple_literal (%V1.ref, %V2.ref)
// CHECK:STDOUT:     %.loc14_53.2: type = converted %.loc14_53.1, constants.%tuple.type.d18 [symbolic = %tuple.type (constants.%tuple.type.d18)]
// CHECK:STDOUT:     %A.ref: %A.type.495 = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:     %W.ref: type = name_ref W, %W.loc14_36.2 [symbolic = %W.loc14_36.1 (constants.%W)]
// CHECK:STDOUT:     %A.type.loc14_61.2: type = facet_type <@A, @A(constants.%W)> [symbolic = %A.type.loc14_61.1 (constants.%A.type.2ec)]
// CHECK:STDOUT:     %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %V1.loc14_14.2: type = symbolic_binding V1, 0 [symbolic = %V1.loc14_14.1 (constants.%V1)]
// CHECK:STDOUT:     %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %V2.loc14_25.2: type = symbolic_binding V2, 1 [symbolic = %V2.loc14_25.1 (constants.%V2)]
// CHECK:STDOUT:     %.Self.3: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %W.loc14_36.2: type = symbolic_binding W, 2 [symbolic = %W.loc14_36.1 (constants.%W)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.impl_witness_table = impl_witness_table (@tuple.type.as.A.impl.%tuple.type.as.A.impl.F.decl), @tuple.type.as.A.impl [concrete]
// CHECK:STDOUT:   %A.impl_witness: <witness> = impl_witness %A.impl_witness_table, @tuple.type.as.A.impl(constants.%V1, constants.%V2, constants.%W) [symbolic = @tuple.type.as.A.impl.%A.impl_witness (constants.%A.impl_witness.2bc)]
// CHECK:STDOUT:   %Call.decl: %Call.type = fn_decl @Call [concrete = constants.%Call] {} {}
// CHECK:STDOUT:   %CallGeneric.decl: %CallGeneric.type = fn_decl @CallGeneric [concrete = constants.%CallGeneric] {
// CHECK:STDOUT:     %T.patt: %pattern_type.809 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc27: type = splice_block %A.type [concrete = constants.%A.type.0a4] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %A.ref: %A.type.495 = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:       %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:       %A.type: type = facet_type <@A, @A(constants.%X)> [concrete = constants.%A.type.0a4]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc27_16.2: %A.type.0a4 = symbolic_binding T, 0 [symbolic = %T.loc27_16.1 (constants.%T.fb1)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallIndirect.decl: %CallIndirect.type = fn_decl @CallIndirect [concrete = constants.%CallIndirect] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @A(%T.loc5_13.2: type) {
// CHECK:STDOUT:   %T.loc5_13.1: type = symbolic_binding T, 0 [symbolic = %T.loc5_13.1 (constants.%T.d9f)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.type: type = facet_type <@A, @A(%T.loc5_13.1)> [symbolic = %A.type (constants.%A.type.c1c)]
// CHECK:STDOUT:   %Self.loc5_23.2: @A.%A.type (%A.type.c1c) = symbolic_binding Self, 1 [symbolic = %Self.loc5_23.2 (constants.%Self.27c)]
// CHECK:STDOUT:   %A.F.type: type = fn_type @A.F, @A(%T.loc5_13.1) [symbolic = %A.F.type (constants.%A.F.type.de8)]
// CHECK:STDOUT:   %A.F: @A.%A.F.type (%A.F.type.de8) = struct_value () [symbolic = %A.F (constants.%A.F.bd2)]
// CHECK:STDOUT:   %A.assoc_type: type = assoc_entity_type @A, @A(%T.loc5_13.1) [symbolic = %A.assoc_type (constants.%A.assoc_type.23a)]
// CHECK:STDOUT:   %assoc0.loc6_39.2: @A.%A.assoc_type (%A.assoc_type.23a) = assoc_entity element0, %A.F.decl [symbolic = %assoc0.loc6_39.2 (constants.%assoc0.1c7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc5_23.1: @A.%A.type (%A.type.c1c) = symbolic_binding Self, 1 [symbolic = %Self.loc5_23.2 (constants.%Self.27c)]
// CHECK:STDOUT:     %A.F.decl: @A.%A.F.type (%A.F.type.de8) = fn_decl @A.F [symbolic = @A.%A.F (constants.%A.F.bd2)] {
// CHECK:STDOUT:       %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 2 [concrete]
// CHECK:STDOUT:       %u.patt: @A.F.%pattern_type.loc6_18 (%pattern_type.533) = value_binding_pattern u [concrete]
// CHECK:STDOUT:       %u.param_patt: @A.F.%pattern_type.loc6_18 (%pattern_type.533) = value_param_pattern %u.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @A.F.%pattern_type.loc6_24 (%pattern_type.845) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @A.F.%pattern_type.loc6_24 (%pattern_type.845) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @A.%T.loc5_13.2 [symbolic = %T (constants.%T.d9f)]
// CHECK:STDOUT:       %.loc6_31: @A.F.%A.type (%A.type.c1c) = specific_constant @A.%Self.loc5_23.1, @A(constants.%T.d9f) [symbolic = %Self (constants.%Self.27c)]
// CHECK:STDOUT:       %Self.ref: @A.F.%A.type (%A.type.c1c) = name_ref Self, %.loc6_31 [symbolic = %Self (constants.%Self.27c)]
// CHECK:STDOUT:       %U.ref.loc6_37: type = name_ref U, %U.loc6_8.2 [symbolic = %U.loc6_8.1 (constants.%U.63a)]
// CHECK:STDOUT:       %.loc6_38.1: @A.F.%tuple.type.loc6_38.1 (%tuple.type.004) = tuple_literal (%T.ref, %Self.ref, %U.ref.loc6_37)
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.410)]
// CHECK:STDOUT:       %.loc6_38.2: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.410)]
// CHECK:STDOUT:       %.loc6_38.3: type = converted %.loc6_38.1, constants.%tuple.type.e5f [symbolic = %tuple.type.loc6_38.2 (constants.%tuple.type.e5f)]
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %U.loc6_8.2: type = symbolic_binding U, 2 [symbolic = %U.loc6_8.1 (constants.%U.63a)]
// CHECK:STDOUT:       %u.param: @A.F.%U.loc6_8.1 (%U.63a) = value_param call_param0
// CHECK:STDOUT:       %U.ref.loc6_21: type = name_ref U, %U.loc6_8.2 [symbolic = %U.loc6_8.1 (constants.%U.63a)]
// CHECK:STDOUT:       %u: @A.F.%U.loc6_8.1 (%U.63a) = value_binding u, %u.param
// CHECK:STDOUT:       %return.param: ref @A.F.%tuple.type.loc6_38.2 (%tuple.type.e5f) = out_param call_param1
// CHECK:STDOUT:       %return: ref @A.F.%tuple.type.loc6_38.2 (%tuple.type.e5f) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc6_39.1: @A.%A.assoc_type (%A.assoc_type.23a) = assoc_entity element0, %A.F.decl [symbolic = %assoc0.loc6_39.2 (constants.%assoc0.1c7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc5_23.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .F = %assoc0.loc6_39.1
// CHECK:STDOUT:     witness = (%A.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @tuple.type.as.A.impl(%V1.loc14_14.2: type, %V2.loc14_25.2: type, %W.loc14_36.2: type) {
// CHECK:STDOUT:   %V1.loc14_14.1: type = symbolic_binding V1, 0 [symbolic = %V1.loc14_14.1 (constants.%V1)]
// CHECK:STDOUT:   %V2.loc14_25.1: type = symbolic_binding V2, 1 [symbolic = %V2.loc14_25.1 (constants.%V2)]
// CHECK:STDOUT:   %W.loc14_36.1: type = symbolic_binding W, 2 [symbolic = %W.loc14_36.1 (constants.%W)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%V1.loc14_14.1, %V2.loc14_25.1) [symbolic = %tuple.type (constants.%tuple.type.d18)]
// CHECK:STDOUT:   %A.type.loc14_61.1: type = facet_type <@A, @A(%W.loc14_36.1)> [symbolic = %A.type.loc14_61.1 (constants.%A.type.2ec)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %A.type.loc14_61.1 [symbolic = %require_complete (constants.%require_complete.4d8)]
// CHECK:STDOUT:   %A.impl_witness: <witness> = impl_witness file.%A.impl_witness_table, @tuple.type.as.A.impl(%V1.loc14_14.1, %V2.loc14_25.1, %W.loc14_36.1) [symbolic = %A.impl_witness (constants.%A.impl_witness.2bc)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type: type = fn_type @tuple.type.as.A.impl.F, @tuple.type.as.A.impl(%V1.loc14_14.1, %V2.loc14_25.1, %W.loc14_36.1) [symbolic = %tuple.type.as.A.impl.F.type (constants.%tuple.type.as.A.impl.F.type.172)]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F: @tuple.type.as.A.impl.%tuple.type.as.A.impl.F.type (%tuple.type.as.A.impl.F.type.172) = struct_value () [symbolic = %tuple.type.as.A.impl.F (constants.%tuple.type.as.A.impl.F.c93)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc14_53.2 as %A.type.loc14_61.2 {
// CHECK:STDOUT:     %tuple.type.as.A.impl.F.decl: @tuple.type.as.A.impl.%tuple.type.as.A.impl.F.type (%tuple.type.as.A.impl.F.type.172) = fn_decl @tuple.type.as.A.impl.F [symbolic = @tuple.type.as.A.impl.%tuple.type.as.A.impl.F (constants.%tuple.type.as.A.impl.F.c93)] {
// CHECK:STDOUT:       %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 3 [concrete]
// CHECK:STDOUT:       %u.patt: @tuple.type.as.A.impl.F.%pattern_type.loc17_18 (%pattern_type.332) = value_binding_pattern u [concrete]
// CHECK:STDOUT:       %u.param_patt: @tuple.type.as.A.impl.F.%pattern_type.loc17_18 (%pattern_type.332) = value_param_pattern %u.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @tuple.type.as.A.impl.F.%pattern_type.loc17_24 (%pattern_type.cbb) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @tuple.type.as.A.impl.F.%pattern_type.loc17_24 (%pattern_type.cbb) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %W.ref: type = name_ref W, @tuple.type.as.A.impl.%W.loc14_36.2 [symbolic = %W (constants.%W)]
// CHECK:STDOUT:       %V1.ref: type = name_ref V1, @tuple.type.as.A.impl.%V1.loc14_14.2 [symbolic = %V1 (constants.%V1)]
// CHECK:STDOUT:       %V2.ref: type = name_ref V2, @tuple.type.as.A.impl.%V2.loc14_25.2 [symbolic = %V2 (constants.%V2)]
// CHECK:STDOUT:       %.loc17_38: %tuple.type.24b = tuple_literal (%V1.ref, %V2.ref)
// CHECK:STDOUT:       %U.ref.loc17_41: type = name_ref U, %U.loc17_8.2 [symbolic = %U.loc17_8.1 (constants.%U.800)]
// CHECK:STDOUT:       %.loc17_42.1: %tuple.type.11f = tuple_literal (%W.ref, %.loc17_38, %U.ref.loc17_41)
// CHECK:STDOUT:       %.loc17_42.2: type = converted %.loc17_38, constants.%tuple.type.d18 [symbolic = %tuple.type.loc17_42.1 (constants.%tuple.type.d18)]
// CHECK:STDOUT:       %.loc17_42.3: type = converted %.loc17_42.1, constants.%tuple.type.d16 [symbolic = %tuple.type.loc17_42.2 (constants.%tuple.type.d16)]
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %U.loc17_8.2: type = symbolic_binding U, 3 [symbolic = %U.loc17_8.1 (constants.%U.800)]
// CHECK:STDOUT:       %u.param: @tuple.type.as.A.impl.F.%U.loc17_8.1 (%U.800) = value_param call_param0
// CHECK:STDOUT:       %U.ref.loc17_21: type = name_ref U, %U.loc17_8.2 [symbolic = %U.loc17_8.1 (constants.%U.800)]
// CHECK:STDOUT:       %u: @tuple.type.as.A.impl.F.%U.loc17_8.1 (%U.800) = value_binding u, %u.param
// CHECK:STDOUT:       %return.param: ref @tuple.type.as.A.impl.F.%tuple.type.loc17_42.2 (%tuple.type.d16) = out_param call_param1
// CHECK:STDOUT:       %return: ref @tuple.type.as.A.impl.F.%tuple.type.loc17_42.2 (%tuple.type.d16) = return_slot %return.param
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
// CHECK:STDOUT: generic fn @A.F(@A.%T.loc5_13.2: type, @A.%Self.loc5_23.1: @A.%A.type (%A.type.c1c), %U.loc6_8.2: type) {
// CHECK:STDOUT:   %U.loc6_8.1: type = symbolic_binding U, 2 [symbolic = %U.loc6_8.1 (constants.%U.63a)]
// CHECK:STDOUT:   %pattern_type.loc6_18: type = pattern_type %U.loc6_8.1 [symbolic = %pattern_type.loc6_18 (constants.%pattern_type.533)]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T.d9f)]
// CHECK:STDOUT:   %A.type: type = facet_type <@A, @A(%T)> [symbolic = %A.type (constants.%A.type.c1c)]
// CHECK:STDOUT:   %Self: @A.F.%A.type (%A.type.c1c) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.27c)]
// CHECK:STDOUT:   %tuple.type.loc6_38.1: type = tuple_type (type, %A.type, type) [symbolic = %tuple.type.loc6_38.1 (constants.%tuple.type.004)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.410)]
// CHECK:STDOUT:   %tuple.type.loc6_38.2: type = tuple_type (%T, %Self.binding.as_type, %U.loc6_8.1) [symbolic = %tuple.type.loc6_38.2 (constants.%tuple.type.e5f)]
// CHECK:STDOUT:   %pattern_type.loc6_24: type = pattern_type %tuple.type.loc6_38.2 [symbolic = %pattern_type.loc6_24 (constants.%pattern_type.845)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param: @A.F.%U.loc6_8.1 (%U.63a)) -> @A.F.%tuple.type.loc6_38.2 (%tuple.type.e5f);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @tuple.type.as.A.impl.F(@tuple.type.as.A.impl.%V1.loc14_14.2: type, @tuple.type.as.A.impl.%V2.loc14_25.2: type, @tuple.type.as.A.impl.%W.loc14_36.2: type, %U.loc17_8.2: type) {
// CHECK:STDOUT:   %U.loc17_8.1: type = symbolic_binding U, 3 [symbolic = %U.loc17_8.1 (constants.%U.800)]
// CHECK:STDOUT:   %pattern_type.loc17_18: type = pattern_type %U.loc17_8.1 [symbolic = %pattern_type.loc17_18 (constants.%pattern_type.332)]
// CHECK:STDOUT:   %W: type = symbolic_binding W, 2 [symbolic = %W (constants.%W)]
// CHECK:STDOUT:   %V1: type = symbolic_binding V1, 0 [symbolic = %V1 (constants.%V1)]
// CHECK:STDOUT:   %V2: type = symbolic_binding V2, 1 [symbolic = %V2 (constants.%V2)]
// CHECK:STDOUT:   %tuple.type.loc17_42.1: type = tuple_type (%V1, %V2) [symbolic = %tuple.type.loc17_42.1 (constants.%tuple.type.d18)]
// CHECK:STDOUT:   %tuple.type.loc17_42.2: type = tuple_type (%W, %tuple.type.loc17_42.1, %U.loc17_8.1) [symbolic = %tuple.type.loc17_42.2 (constants.%tuple.type.d16)]
// CHECK:STDOUT:   %pattern_type.loc17_24: type = pattern_type %tuple.type.loc17_42.2 [symbolic = %pattern_type.loc17_24 (constants.%pattern_type.cbb)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc17_24: <witness> = require_complete_type %tuple.type.loc17_42.2 [symbolic = %require_complete.loc17_24 (constants.%require_complete.923)]
// CHECK:STDOUT:   %require_complete.loc17_19: <witness> = require_complete_type %U.loc17_8.1 [symbolic = %require_complete.loc17_19 (constants.%require_complete.997)]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type: type = fn_type @tuple.type.as.A.impl.F, @tuple.type.as.A.impl(%V1, %V2, %W) [symbolic = %tuple.type.as.A.impl.F.type (constants.%tuple.type.as.A.impl.F.type.172)]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F: @tuple.type.as.A.impl.F.%tuple.type.as.A.impl.F.type (%tuple.type.as.A.impl.F.type.172) = struct_value () [symbolic = %tuple.type.as.A.impl.F (constants.%tuple.type.as.A.impl.F.c93)]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.specific_fn.loc18_12.2: <specific function> = specific_function %tuple.type.as.A.impl.F, @tuple.type.as.A.impl.F(%V1, %V2, %W, %U.loc17_8.1) [symbolic = %tuple.type.as.A.impl.F.specific_fn.loc18_12.2 (constants.%tuple.type.as.A.impl.F.specific_fn.5bf)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param: @tuple.type.as.A.impl.F.%U.loc17_8.1 (%U.800)) -> %return.param: @tuple.type.as.A.impl.F.%tuple.type.loc17_42.2 (%tuple.type.d16) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc18: @tuple.type.as.A.impl.F.%tuple.type.as.A.impl.F.type (%tuple.type.as.A.impl.F.type.172) = specific_constant @tuple.type.as.A.impl.%tuple.type.as.A.impl.F.decl, @tuple.type.as.A.impl(constants.%V1, constants.%V2, constants.%W) [symbolic = %tuple.type.as.A.impl.F (constants.%tuple.type.as.A.impl.F.c93)]
// CHECK:STDOUT:     %F.ref: @tuple.type.as.A.impl.F.%tuple.type.as.A.impl.F.type (%tuple.type.as.A.impl.F.type.172) = name_ref F, %.loc18 [symbolic = %tuple.type.as.A.impl.F (constants.%tuple.type.as.A.impl.F.c93)]
// CHECK:STDOUT:     %U.ref.loc18: type = name_ref U, %U.loc17_8.2 [symbolic = %U.loc17_8.1 (constants.%U.800)]
// CHECK:STDOUT:     %u.ref: @tuple.type.as.A.impl.F.%U.loc17_8.1 (%U.800) = name_ref u, %u
// CHECK:STDOUT:     %tuple.type.as.A.impl.F.specific_fn.loc18_12.1: <specific function> = specific_function %F.ref, @tuple.type.as.A.impl.F(constants.%V1, constants.%V2, constants.%W, constants.%U.800) [symbolic = %tuple.type.as.A.impl.F.specific_fn.loc18_12.2 (constants.%tuple.type.as.A.impl.F.specific_fn.5bf)]
// CHECK:STDOUT:     %.loc17_24: ref @tuple.type.as.A.impl.F.%tuple.type.loc17_42.2 (%tuple.type.d16) = splice_block %return {}
// CHECK:STDOUT:     %tuple.type.as.A.impl.F.call: init @tuple.type.as.A.impl.F.%tuple.type.loc17_42.2 (%tuple.type.d16) = call %tuple.type.as.A.impl.F.specific_fn.loc18_12.1(%u.ref) to %.loc17_24
// CHECK:STDOUT:     return %tuple.type.as.A.impl.F.call to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Y1.ref: type = name_ref Y1, file.%Y1.decl [concrete = constants.%Y1]
// CHECK:STDOUT:   %Y2.ref: type = name_ref Y2, file.%Y2.decl [concrete = constants.%Y2]
// CHECK:STDOUT:   %.loc24_12: %tuple.type.24b = tuple_literal (%Y1.ref, %Y2.ref)
// CHECK:STDOUT:   %.loc24_14: type = converted %.loc24_12, constants.%tuple.type.a46 [concrete = constants.%tuple.type.a46]
// CHECK:STDOUT:   %A.ref: %A.type.495 = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:   %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %A.type: type = facet_type <@A, @A(constants.%X)> [concrete = constants.%A.type.0a4]
// CHECK:STDOUT:   %A.facet: %A.type.0a4 = facet_value %.loc24_14, (constants.%A.impl_witness.be7) [concrete = constants.%A.facet.f26]
// CHECK:STDOUT:   %.loc24_23: %A.type.0a4 = converted %.loc24_14, %A.facet [concrete = constants.%A.facet.f26]
// CHECK:STDOUT:   %.loc24_31.1: %A.assoc_type.296 = specific_constant @A.%assoc0.loc6_39.1, @A(constants.%X) [concrete = constants.%assoc0.6a1]
// CHECK:STDOUT:   %F.ref: %A.assoc_type.296 = name_ref F, %.loc24_31.1 [concrete = constants.%assoc0.6a1]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc24_23 [concrete = constants.%tuple.type.a46]
// CHECK:STDOUT:   %.loc24_31.2: type = converted %.loc24_23, %as_type [concrete = constants.%tuple.type.a46]
// CHECK:STDOUT:   %impl.elem0: %.a2b = impl_witness_access constants.%A.impl_witness.be7, element0 [concrete = constants.%tuple.type.as.A.impl.F.735]
// CHECK:STDOUT:   %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z]
// CHECK:STDOUT:   %.loc24_38.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @tuple.type.as.A.impl.F(constants.%Y1, constants.%Y2, constants.%X, constants.%Z) [concrete = constants.%tuple.type.as.A.impl.F.specific_fn.5c5]
// CHECK:STDOUT:   %.loc24_39.1: ref %tuple.type.415 = temporary_storage
// CHECK:STDOUT:   %.loc24_38.2: ref %Z = temporary_storage
// CHECK:STDOUT:   %.loc24_38.3: init %Z = class_init (), %.loc24_38.2 [concrete = constants.%Z.val]
// CHECK:STDOUT:   %.loc24_38.4: ref %Z = temporary %.loc24_38.2, %.loc24_38.3
// CHECK:STDOUT:   %.loc24_38.5: ref %Z = converted %.loc24_38.1, %.loc24_38.4
// CHECK:STDOUT:   %.loc24_38.6: %Z = acquire_value %.loc24_38.5
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.call: init %tuple.type.415 = call %specific_fn(%.loc24_38.6) to %.loc24_39.1
// CHECK:STDOUT:   %.loc24_39.2: ref %tuple.type.415 = temporary %.loc24_39.1, %tuple.type.as.A.impl.F.call
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc24_39: <bound method> = bound_method %.loc24_39.2, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.472
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.472, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.71a) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.12b]
// CHECK:STDOUT:   %bound_method.loc24_39: <bound method> = bound_method %.loc24_39.2, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc24_39: %ptr.ad9 = addr_of %.loc24_39.2
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc24_39: init %empty_tuple.type = call %bound_method.loc24_39(%addr.loc24_39)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc24_38: <bound method> = bound_method %.loc24_38.4, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.2cc
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.2: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.2cc, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.4e3) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.cb3]
// CHECK:STDOUT:   %bound_method.loc24_38: <bound method> = bound_method %.loc24_38.4, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc24_38: %ptr.fb6 = addr_of %.loc24_38.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc24_38: init %empty_tuple.type = call %bound_method.loc24_38(%addr.loc24_38)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallGeneric(%T.loc27_16.2: %A.type.0a4) {
// CHECK:STDOUT:   %T.loc27_16.1: %A.type.0a4 = symbolic_binding T, 0 [symbolic = %T.loc27_16.1 (constants.%T.fb1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc27_16.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %A.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc27_16.1, @A, @A(constants.%X) [symbolic = %A.lookup_impl_witness (constants.%A.lookup_impl_witness)]
// CHECK:STDOUT:   %.loc28_4.3: type = fn_type_with_self_type constants.%A.F.type.13d, %T.loc27_16.1 [symbolic = %.loc28_4.3 (constants.%.6ed)]
// CHECK:STDOUT:   %impl.elem0.loc28_4.2: @CallGeneric.%.loc28_4.3 (%.6ed) = impl_witness_access %A.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc28_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc28_4.2: <specific function> = specific_impl_function %impl.elem0.loc28_4.2, @A.F(constants.%X, %T.loc27_16.1, constants.%Z) [symbolic = %specific_impl_fn.loc28_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (constants.%X, %T.binding.as_type, constants.%Z) [symbolic = %tuple.type (constants.%tuple.type.bf3)]
// CHECK:STDOUT:   %require_complete.loc28_12.1: <witness> = require_complete_type %tuple.type [symbolic = %require_complete.loc28_12.1 (constants.%require_complete.4e9)]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %tuple.type, () [symbolic = %facet_value (constants.%facet_value.185)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness imports.%Destroy.impl_witness_table, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness.cd1)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %tuple.type, (%Destroy.impl_witness) [symbolic = %Destroy.facet (constants.%Destroy.facet.1b6)]
// CHECK:STDOUT:   %.loc28_12.4: type = fn_type_with_self_type constants.%Destroy.Op.type, %Destroy.facet [symbolic = %.loc28_12.4 (constants.%.cce)]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op.type (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.type.a5d)]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op: @CallGeneric.%DestroyT.binding.as_type.as.Destroy.impl.Op.type (%DestroyT.binding.as_type.as.Destroy.impl.Op.type.a5d) = struct_value () [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.a08)]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.loc28: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value) [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.loc28 (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.760)]
// CHECK:STDOUT:   %ptr: type = ptr_type %tuple.type [symbolic = %ptr (constants.%ptr.4ce)]
// CHECK:STDOUT:   %require_complete.loc28_12.2: <witness> = require_complete_type %ptr [symbolic = %require_complete.loc28_12.2 (constants.%require_complete.7d0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref: %A.type.0a4 = name_ref T, %T.loc27_16.2 [symbolic = %T.loc27_16.1 (constants.%T.fb1)]
// CHECK:STDOUT:     %.loc28_4.1: %A.assoc_type.296 = specific_constant @A.%assoc0.loc6_39.1, @A(constants.%X) [concrete = constants.%assoc0.6a1]
// CHECK:STDOUT:     %F.ref: %A.assoc_type.296 = name_ref F, %.loc28_4.1 [concrete = constants.%assoc0.6a1]
// CHECK:STDOUT:     %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc28_4.2: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %impl.elem0.loc28_4.1: @CallGeneric.%.loc28_4.3 (%.6ed) = impl_witness_access constants.%A.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc28_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z]
// CHECK:STDOUT:     %.loc28_11.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %specific_impl_fn.loc28_4.1: <specific function> = specific_impl_function %impl.elem0.loc28_4.1, @A.F(constants.%X, constants.%T.fb1, constants.%Z) [symbolic = %specific_impl_fn.loc28_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %.loc28_12.1: ref @CallGeneric.%tuple.type (%tuple.type.bf3) = temporary_storage
// CHECK:STDOUT:     %.loc28_11.2: ref %Z = temporary_storage
// CHECK:STDOUT:     %.loc28_11.3: init %Z = class_init (), %.loc28_11.2 [concrete = constants.%Z.val]
// CHECK:STDOUT:     %.loc28_11.4: ref %Z = temporary %.loc28_11.2, %.loc28_11.3
// CHECK:STDOUT:     %.loc28_11.5: ref %Z = converted %.loc28_11.1, %.loc28_11.4
// CHECK:STDOUT:     %.loc28_11.6: %Z = acquire_value %.loc28_11.5
// CHECK:STDOUT:     %.loc28_12.2: init @CallGeneric.%tuple.type (%tuple.type.bf3) = call %specific_impl_fn.loc28_4.1(%.loc28_11.6) to %.loc28_12.1
// CHECK:STDOUT:     %.loc28_12.3: ref @CallGeneric.%tuple.type (%tuple.type.bf3) = temporary %.loc28_12.1, %.loc28_12.2
// CHECK:STDOUT:     %impl.elem0.loc28_12: @CallGeneric.%.loc28_12.4 (%.cce) = impl_witness_access constants.%Destroy.impl_witness.cd1, element0 [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.a08)]
// CHECK:STDOUT:     %bound_method.loc28_12.1: <bound method> = bound_method %.loc28_12.3, %impl.elem0.loc28_12
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0.loc28_12, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.185) [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.loc28 (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.760)]
// CHECK:STDOUT:     %bound_method.loc28_12.2: <bound method> = bound_method %.loc28_12.3, %specific_fn
// CHECK:STDOUT:     %addr.loc28_12: @CallGeneric.%ptr (%ptr.4ce) = addr_of %.loc28_12.3
// CHECK:STDOUT:     %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc28_12: init %empty_tuple.type = call %bound_method.loc28_12.2(%addr.loc28_12)
// CHECK:STDOUT:     %DestroyT.binding.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc28_11.4, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.2cc
// CHECK:STDOUT:     %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.2cc, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.4e3) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.cb3]
// CHECK:STDOUT:     %bound_method.loc28_11: <bound method> = bound_method %.loc28_11.4, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:     %addr.loc28_11: %ptr.fb6 = addr_of %.loc28_11.4
// CHECK:STDOUT:     %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc28_11: init %empty_tuple.type = call %bound_method.loc28_11(%addr.loc28_11)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallIndirect() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CallGeneric.ref: %CallGeneric.type = name_ref CallGeneric, file.%CallGeneric.decl [concrete = constants.%CallGeneric]
// CHECK:STDOUT:   %Y1.ref: type = name_ref Y1, file.%Y1.decl [concrete = constants.%Y1]
// CHECK:STDOUT:   %Y2.ref: type = name_ref Y2, file.%Y2.decl [concrete = constants.%Y2]
// CHECK:STDOUT:   %.loc33_22: %tuple.type.24b = tuple_literal (%Y1.ref, %Y2.ref)
// CHECK:STDOUT:   %.loc33_24: type = converted %.loc33_22, constants.%tuple.type.a46 [concrete = constants.%tuple.type.a46]
// CHECK:STDOUT:   %A.facet: %A.type.0a4 = facet_value %.loc33_24, (constants.%A.impl_witness.be7) [concrete = constants.%A.facet.f26]
// CHECK:STDOUT:   %.loc33_31: %A.type.0a4 = converted %.loc33_24, %A.facet [concrete = constants.%A.facet.f26]
// CHECK:STDOUT:   %CallGeneric.specific_fn: <specific function> = specific_function %CallGeneric.ref, @CallGeneric(constants.%A.facet.f26) [concrete = constants.%CallGeneric.specific_fn]
// CHECK:STDOUT:   %CallGeneric.call: init %empty_tuple.type = call %CallGeneric.specific_fn()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%T.d9f) {
// CHECK:STDOUT:   %T.loc5_13.1 => constants.%T.d9f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.F(constants.%T.d9f, constants.%Self.27c, constants.%U.63a) {
// CHECK:STDOUT:   %U.loc6_8.1 => constants.%U.63a
// CHECK:STDOUT:   %pattern_type.loc6_18 => constants.%pattern_type.533
// CHECK:STDOUT:   %T => constants.%T.d9f
// CHECK:STDOUT:   %A.type => constants.%A.type.c1c
// CHECK:STDOUT:   %Self => constants.%Self.27c
// CHECK:STDOUT:   %tuple.type.loc6_38.1 => constants.%tuple.type.004
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.410
// CHECK:STDOUT:   %tuple.type.loc6_38.2 => constants.%tuple.type.e5f
// CHECK:STDOUT:   %pattern_type.loc6_24 => constants.%pattern_type.845
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%W) {
// CHECK:STDOUT:   %T.loc5_13.1 => constants.%W
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.type => constants.%A.type.2ec
// CHECK:STDOUT:   %Self.loc5_23.2 => constants.%Self.b6c
// CHECK:STDOUT:   %A.F.type => constants.%A.F.type.f65
// CHECK:STDOUT:   %A.F => constants.%A.F.d51
// CHECK:STDOUT:   %A.assoc_type => constants.%A.assoc_type.abb
// CHECK:STDOUT:   %assoc0.loc6_39.2 => constants.%assoc0.973
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @tuple.type.as.A.impl(constants.%V1, constants.%V2, constants.%W) {
// CHECK:STDOUT:   %V1.loc14_14.1 => constants.%V1
// CHECK:STDOUT:   %V2.loc14_25.1 => constants.%V2
// CHECK:STDOUT:   %W.loc14_36.1 => constants.%W
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.d18
// CHECK:STDOUT:   %A.type.loc14_61.1 => constants.%A.type.2ec
// CHECK:STDOUT:   %require_complete => constants.%require_complete.4d8
// CHECK:STDOUT:   %A.impl_witness => constants.%A.impl_witness.2bc
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type => constants.%tuple.type.as.A.impl.F.type.172
// CHECK:STDOUT:   %tuple.type.as.A.impl.F => constants.%tuple.type.as.A.impl.F.c93
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @tuple.type.as.A.impl.F(constants.%V1, constants.%V2, constants.%W, constants.%U.800) {
// CHECK:STDOUT:   %U.loc17_8.1 => constants.%U.800
// CHECK:STDOUT:   %pattern_type.loc17_18 => constants.%pattern_type.332
// CHECK:STDOUT:   %W => constants.%W
// CHECK:STDOUT:   %V1 => constants.%V1
// CHECK:STDOUT:   %V2 => constants.%V2
// CHECK:STDOUT:   %tuple.type.loc17_42.1 => constants.%tuple.type.d18
// CHECK:STDOUT:   %tuple.type.loc17_42.2 => constants.%tuple.type.d16
// CHECK:STDOUT:   %pattern_type.loc17_24 => constants.%pattern_type.cbb
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc17_24 => constants.%require_complete.923
// CHECK:STDOUT:   %require_complete.loc17_19 => constants.%require_complete.997
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type => constants.%tuple.type.as.A.impl.F.type.172
// CHECK:STDOUT:   %tuple.type.as.A.impl.F => constants.%tuple.type.as.A.impl.F.c93
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.specific_fn.loc18_12.2 => constants.%tuple.type.as.A.impl.F.specific_fn.5bf
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.F(constants.%W, constants.%A.facet.6f0, constants.%U.800) {
// CHECK:STDOUT:   %U.loc6_8.1 => constants.%U.800
// CHECK:STDOUT:   %pattern_type.loc6_18 => constants.%pattern_type.332
// CHECK:STDOUT:   %T => constants.%W
// CHECK:STDOUT:   %A.type => constants.%A.type.2ec
// CHECK:STDOUT:   %Self => constants.%A.facet.6f0
// CHECK:STDOUT:   %tuple.type.loc6_38.1 => constants.%tuple.type.e39
// CHECK:STDOUT:   %Self.binding.as_type => constants.%tuple.type.d18
// CHECK:STDOUT:   %tuple.type.loc6_38.2 => constants.%tuple.type.d16
// CHECK:STDOUT:   %pattern_type.loc6_24 => constants.%pattern_type.cbb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%X) {
// CHECK:STDOUT:   %T.loc5_13.1 => constants.%X
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.type => constants.%A.type.0a4
// CHECK:STDOUT:   %Self.loc5_23.2 => constants.%Self.789
// CHECK:STDOUT:   %A.F.type => constants.%A.F.type.13d
// CHECK:STDOUT:   %A.F => constants.%A.F.d83
// CHECK:STDOUT:   %A.assoc_type => constants.%A.assoc_type.296
// CHECK:STDOUT:   %assoc0.loc6_39.2 => constants.%assoc0.6a1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @tuple.type.as.A.impl(constants.%Y1, constants.%Y2, constants.%X) {
// CHECK:STDOUT:   %V1.loc14_14.1 => constants.%Y1
// CHECK:STDOUT:   %V2.loc14_25.1 => constants.%Y2
// CHECK:STDOUT:   %W.loc14_36.1 => constants.%X
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.a46
// CHECK:STDOUT:   %A.type.loc14_61.1 => constants.%A.type.0a4
// CHECK:STDOUT:   %require_complete => constants.%complete_type.e2d
// CHECK:STDOUT:   %A.impl_witness => constants.%A.impl_witness.be7
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type => constants.%tuple.type.as.A.impl.F.type.415
// CHECK:STDOUT:   %tuple.type.as.A.impl.F => constants.%tuple.type.as.A.impl.F.735
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @tuple.type.as.A.impl.F(constants.%Y1, constants.%Y2, constants.%X, constants.%Z) {
// CHECK:STDOUT:   %U.loc17_8.1 => constants.%Z
// CHECK:STDOUT:   %pattern_type.loc17_18 => constants.%pattern_type.8f9
// CHECK:STDOUT:   %W => constants.%X
// CHECK:STDOUT:   %V1 => constants.%Y1
// CHECK:STDOUT:   %V2 => constants.%Y2
// CHECK:STDOUT:   %tuple.type.loc17_42.1 => constants.%tuple.type.a46
// CHECK:STDOUT:   %tuple.type.loc17_42.2 => constants.%tuple.type.415
// CHECK:STDOUT:   %pattern_type.loc17_24 => constants.%pattern_type.0b2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc17_24 => constants.%complete_type.aa8
// CHECK:STDOUT:   %require_complete.loc17_19 => constants.%complete_type.357
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type => constants.%tuple.type.as.A.impl.F.type.415
// CHECK:STDOUT:   %tuple.type.as.A.impl.F => constants.%tuple.type.as.A.impl.F.735
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.specific_fn.loc18_12.2 => constants.%tuple.type.as.A.impl.F.specific_fn.5c5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGeneric(constants.%T.fb1) {
// CHECK:STDOUT:   %T.loc27_16.1 => constants.%T.fb1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.F(constants.%X, constants.%T.fb1, constants.%Z) {
// CHECK:STDOUT:   %U.loc6_8.1 => constants.%Z
// CHECK:STDOUT:   %pattern_type.loc6_18 => constants.%pattern_type.8f9
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.0a4
// CHECK:STDOUT:   %Self => constants.%T.fb1
// CHECK:STDOUT:   %tuple.type.loc6_38.1 => constants.%tuple.type.5a1
// CHECK:STDOUT:   %Self.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %tuple.type.loc6_38.2 => constants.%tuple.type.bf3
// CHECK:STDOUT:   %pattern_type.loc6_24 => constants.%pattern_type.e6a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGeneric(constants.%A.facet.f26) {
// CHECK:STDOUT:   %T.loc27_16.1 => constants.%A.facet.f26
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.binding.as_type => constants.%tuple.type.a46
// CHECK:STDOUT:   %A.lookup_impl_witness => constants.%A.impl_witness.be7
// CHECK:STDOUT:   %.loc28_4.3 => constants.%.a2b
// CHECK:STDOUT:   %impl.elem0.loc28_4.2 => constants.%tuple.type.as.A.impl.F.735
// CHECK:STDOUT:   %specific_impl_fn.loc28_4.2 => constants.%tuple.type.as.A.impl.F.specific_fn.5c5
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.415
// CHECK:STDOUT:   %require_complete.loc28_12.1 => constants.%complete_type.aa8
// CHECK:STDOUT:   %facet_value => constants.%facet_value.71a
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.c75
// CHECK:STDOUT:   %Destroy.facet => constants.%Destroy.facet.b37
// CHECK:STDOUT:   %.loc28_12.4 => constants.%.322
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type => constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.type.5cd
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op => constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.472
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.loc28 => constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.12b
// CHECK:STDOUT:   %ptr => constants.%ptr.ad9
// CHECK:STDOUT:   %require_complete.loc28_12.2 => constants.%complete_type.e23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.F(constants.%X, constants.%A.facet.f26, constants.%Z) {
// CHECK:STDOUT:   %U.loc6_8.1 => constants.%Z
// CHECK:STDOUT:   %pattern_type.loc6_18 => constants.%pattern_type.8f9
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.0a4
// CHECK:STDOUT:   %Self => constants.%A.facet.f26
// CHECK:STDOUT:   %tuple.type.loc6_38.1 => constants.%tuple.type.5a1
// CHECK:STDOUT:   %Self.binding.as_type => constants.%tuple.type.a46
// CHECK:STDOUT:   %tuple.type.loc6_38.2 => constants.%tuple.type.415
// CHECK:STDOUT:   %pattern_type.loc6_24 => constants.%pattern_type.0b2
// CHECK:STDOUT: }
// CHECK:STDOUT:

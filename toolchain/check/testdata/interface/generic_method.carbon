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
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %A.type.464: type = generic_interface_type @A [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %A.generic: %A.type.464 = struct_value () [concrete]
// CHECK:STDOUT:   %A.type.314: type = facet_type <@A, @A(%T.67d)> [symbolic]
// CHECK:STDOUT:   %Self.9b1: %A.type.314 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %U.4e7: type = symbolic_binding U, 2 [symbolic]
// CHECK:STDOUT:   %ptr.8f6: type = ptr_type %U.4e7 [symbolic]
// CHECK:STDOUT:   %pattern_type.986: type = pattern_type %ptr.8f6 [symbolic]
// CHECK:STDOUT:   %tuple.type.f42: type = tuple_type (type, %A.type.314, type) [symbolic]
// CHECK:STDOUT:   %tuple.829: %tuple.type.f42 = tuple_value (%T.67d, %Self.9b1, %ptr.8f6) [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.78c: type = symbolic_binding_type Self, 1, %Self.9b1 [symbolic]
// CHECK:STDOUT:   %tuple.type.d56: type = tuple_type (%T.67d, %Self.binding.as_type.78c, %ptr.8f6) [symbolic]
// CHECK:STDOUT:   %pattern_type.a5f: type = pattern_type %tuple.type.d56 [symbolic]
// CHECK:STDOUT:   %A.F.type.71b: type = fn_type @A.F, @A(%T.67d) [symbolic]
// CHECK:STDOUT:   %A.F.155: %A.F.type.71b = struct_value () [symbolic]
// CHECK:STDOUT:   %A.assoc_type.9f4: type = assoc_entity_type @A, @A(%T.67d) [symbolic]
// CHECK:STDOUT:   %assoc0.c2b: %A.assoc_type.9f4 = assoc_entity element0, @A.%A.F.decl [symbolic]
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Y: type = class_type @Y [concrete]
// CHECK:STDOUT:   %Z: type = class_type @Z [concrete]
// CHECK:STDOUT:   %A.type.f82: type = facet_type <@A, @A(%X)> [concrete]
// CHECK:STDOUT:   %A.impl_witness: <witness> = impl_witness file.%A.impl_witness_table [concrete]
// CHECK:STDOUT:   %Self.aab: %A.type.f82 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %A.F.type.49b: type = fn_type @A.F, @A(%X) [concrete]
// CHECK:STDOUT:   %A.F.768: %A.F.type.49b = struct_value () [concrete]
// CHECK:STDOUT:   %A.assoc_type.f05: type = assoc_entity_type @A, @A(%X) [concrete]
// CHECK:STDOUT:   %assoc0.249: %A.assoc_type.f05 = assoc_entity element0, @A.%A.F.decl [concrete]
// CHECK:STDOUT:   %U.67d: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %ptr.e8f8f9.1: type = ptr_type %U.67d [symbolic]
// CHECK:STDOUT:   %pattern_type.4f4b84.1: type = pattern_type %ptr.e8f8f9.1 [symbolic]
// CHECK:STDOUT:   %tuple.type.ff9: type = tuple_type (type, type, type) [concrete]
// CHECK:STDOUT:   %tuple.7ab: %tuple.type.ff9 = tuple_value (%X, %Y, %ptr.e8f8f9.1) [symbolic]
// CHECK:STDOUT:   %tuple.type.244: type = tuple_type (%X, %Y, %ptr.e8f8f9.1) [symbolic]
// CHECK:STDOUT:   %pattern_type.06d: type = pattern_type %tuple.type.244 [symbolic]
// CHECK:STDOUT:   %Y.as.A.impl.F.type: type = fn_type @Y.as.A.impl.F [concrete]
// CHECK:STDOUT:   %Y.as.A.impl.F: %Y.as.A.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %A.facet: %A.type.f82 = facet_value %Y, (%A.impl_witness) [concrete]
// CHECK:STDOUT:   %tuple.type.298: type = tuple_type (type, %A.type.f82, type) [concrete]
// CHECK:STDOUT:   %tuple.74e: %tuple.type.298 = tuple_value (%X, %A.facet, %ptr.e8f8f9.1) [symbolic]
// CHECK:STDOUT:   %require_complete.034: <witness> = require_complete_type %tuple.type.244 [symbolic]
// CHECK:STDOUT:   %require_complete.ef162c.1: <witness> = require_complete_type %ptr.e8f8f9.1 [symbolic]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.dd2: type = tuple_type (%empty_struct_type, %empty_struct_type, %ptr.e8f8f9.1) [symbolic]
// CHECK:STDOUT:   %X.val: %X = struct_value () [concrete]
// CHECK:STDOUT:   %Y.val: %Y = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.b05be2.1: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.8b8eb7.1: %ptr.as.Copy.impl.Op.type.b05be2.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.841: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%U.67d) [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.2e6: <witness> = lookup_impl_witness %ptr.e8f8f9.1, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.facet.29b: %Copy.type = facet_value %ptr.e8f8f9.1, (%Copy.lookup_impl_witness.2e6) [symbolic]
// CHECK:STDOUT:   %.cb5: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.29b [symbolic]
// CHECK:STDOUT:   %impl.elem0.771: %.cb5 = impl_witness_access %Copy.lookup_impl_witness.2e6, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.b85: <specific function> = specific_impl_function %impl.elem0.771, @Copy.Op(%Copy.facet.29b) [symbolic]
// CHECK:STDOUT:   %Call.type: type = fn_type @Call [concrete]
// CHECK:STDOUT:   %Call: %Call.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.bdb: type = pattern_type %Z [concrete]
// CHECK:STDOUT:   %.291: type = fn_type_with_self_type %A.F.type.49b, %A.facet [concrete]
// CHECK:STDOUT:   %ptr.2a0: type = ptr_type %Z [concrete]
// CHECK:STDOUT:   %pattern_type.771: type = pattern_type %ptr.2a0 [concrete]
// CHECK:STDOUT:   %tuple.84f: %tuple.type.ff9 = tuple_value (%X, %Y, %ptr.2a0) [concrete]
// CHECK:STDOUT:   %tuple.type.847: type = tuple_type (%X, %Y, %ptr.2a0) [concrete]
// CHECK:STDOUT:   %pattern_type.79c: type = pattern_type %tuple.type.847 [concrete]
// CHECK:STDOUT:   %Y.as.A.impl.F.specific_fn: <specific function> = specific_function %Y.as.A.impl.F, @Y.as.A.impl.F(%Z) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %DestroyT: %type_where = symbolic_binding DestroyT, 0 [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.0bf: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%DestroyT) [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.97a: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.0bf = struct_value () [symbolic]
// CHECK:STDOUT:   %facet_value.f1e: %type_where = facet_value %tuple.type.847, () [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.a85: <witness> = impl_witness imports.%Destroy.impl_witness_table, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.f1e) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.7fc: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.f1e) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.5c5: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.7fc = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type.28e: <witness> = complete_type_witness %tuple.type.847 [concrete]
// CHECK:STDOUT:   %.daf: require_specific_def_type = require_specific_def @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.f1e) [concrete]
// CHECK:STDOUT:   %Destroy.facet.842: %Destroy.type = facet_value %tuple.type.847, (%Destroy.impl_witness.a85) [concrete]
// CHECK:STDOUT:   %.22c: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet.842 [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.6f7: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op.5c5, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value.f1e) [concrete]
// CHECK:STDOUT:   %facet_value.5de: %type_where = facet_value %Z, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.17b: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.5de) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.72c: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.17b = struct_value () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.8a3: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op.72c, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value.5de) [concrete]
// CHECK:STDOUT:   %T.19a: %A.type.f82 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.e04: type = pattern_type %A.type.f82 [concrete]
// CHECK:STDOUT:   %CallGeneric.type: type = fn_type @CallGeneric [concrete]
// CHECK:STDOUT:   %CallGeneric: %CallGeneric.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.binding.as_type.935: type = symbolic_binding_type T, 0, %T.19a [symbolic]
// CHECK:STDOUT:   %A.lookup_impl_witness: <witness> = lookup_impl_witness %T.19a, @A, @A(%X) [symbolic]
// CHECK:STDOUT:   %.14f: type = fn_type_with_self_type %A.F.type.49b, %T.19a [symbolic]
// CHECK:STDOUT:   %impl.elem0.0bc: %.14f = impl_witness_access %A.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %tuple.8a5: %tuple.type.298 = tuple_value (%X, %T.19a, %ptr.2a0) [symbolic]
// CHECK:STDOUT:   %tuple.type.d09: type = tuple_type (%X, %T.binding.as_type.935, %ptr.2a0) [symbolic]
// CHECK:STDOUT:   %pattern_type.93b: type = pattern_type %tuple.type.d09 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.df5: <specific function> = specific_impl_function %impl.elem0.0bc, @A.F(%X, %T.19a, %Z) [symbolic]
// CHECK:STDOUT:   %require_complete.e14: <witness> = require_complete_type %tuple.type.d09 [symbolic]
// CHECK:STDOUT:   %facet_value.c0a: %type_where = facet_value %tuple.type.d09, () [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.3ff: <witness> = impl_witness imports.%Destroy.impl_witness_table, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.c0a) [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.ce9: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.c0a) [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.11f: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.ce9 = struct_value () [symbolic]
// CHECK:STDOUT:   %.7bb: require_specific_def_type = require_specific_def @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.c0a) [symbolic]
// CHECK:STDOUT:   %Destroy.facet.77a: %Destroy.type = facet_value %tuple.type.d09, (%Destroy.impl_witness.3ff) [symbolic]
// CHECK:STDOUT:   %.5f4: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet.77a [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1b1: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op.11f, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value.c0a) [symbolic]
// CHECK:STDOUT:   %CallIndirect.type: type = fn_type @CallIndirect [concrete]
// CHECK:STDOUT:   %CallIndirect: %CallIndirect.type = struct_value () [concrete]
// CHECK:STDOUT:   %CallGeneric.specific_fn: <specific function> = specific_function %CallGeneric, @CallGeneric(%A.facet) [concrete]
// CHECK:STDOUT:   %complete_type.543: <witness> = complete_type_witness %ptr.2a0 [concrete]
// CHECK:STDOUT:   %tuple.type.0dd: type = tuple_type (%empty_struct_type, %empty_struct_type, %ptr.2a0) [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.dc6: <witness> = impl_witness imports.%Copy.impl_witness_table.4df, @ptr.as.Copy.impl(%Z) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.375: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%Z) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.7d1: %ptr.as.Copy.impl.Op.type.375 = struct_value () [concrete]
// CHECK:STDOUT:   %.3fe: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%Z) [concrete]
// CHECK:STDOUT:   %Copy.facet.f23: %Copy.type = facet_value %ptr.2a0, (%Copy.impl_witness.dc6) [concrete]
// CHECK:STDOUT:   %.da6: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.f23 [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.7d1, @ptr.as.Copy.impl.Op(%Z) [concrete]
// CHECK:STDOUT:   %tuple.f66: %tuple.type.298 = tuple_value (%X, %A.facet, %ptr.2a0) [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.b85: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.b05be2.1) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.8b8eb7.1)]
// CHECK:STDOUT:   %Copy.impl_witness_table.4df = impl_witness_table (%Core.import_ref.b85), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.93a: @DestroyT.binding.as_type.as.Destroy.impl.%DestroyT.binding.as_type.as.Destroy.impl.Op.type (%DestroyT.binding.as_type.as.Destroy.impl.Op.type.0bf) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @DestroyT.binding.as_type.as.Destroy.impl.%DestroyT.binding.as_type.as.Destroy.impl.Op (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.97a)]
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (%Core.import_ref.93a), @DestroyT.binding.as_type.as.Destroy.impl [concrete]
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
// CHECK:STDOUT:   %A.decl: %A.type.464 = interface_decl @A [concrete = constants.%A.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc5_13.2: type = symbolic_binding T, 0 [symbolic = %T.loc5_13.1 (constants.%T.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.decl: type = class_decl @X [concrete = constants.%X] {} {}
// CHECK:STDOUT:   %Y.decl: type = class_decl @Y [concrete = constants.%Y] {} {}
// CHECK:STDOUT:   %Z.decl: type = class_decl @Z [concrete = constants.%Z] {} {}
// CHECK:STDOUT:   impl_decl @Y.as.A.impl [concrete] {} {
// CHECK:STDOUT:     %Y.ref: type = name_ref Y, file.%Y.decl [concrete = constants.%Y]
// CHECK:STDOUT:     %A.ref: %A.type.464 = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %A.type: type = facet_type <@A, @A(constants.%X)> [concrete = constants.%A.type.f82]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.impl_witness_table = impl_witness_table (@Y.as.A.impl.%Y.as.A.impl.F.decl), @Y.as.A.impl [concrete]
// CHECK:STDOUT:   %A.impl_witness: <witness> = impl_witness %A.impl_witness_table [concrete = constants.%A.impl_witness]
// CHECK:STDOUT:   %Call.decl: %Call.type = fn_decl @Call [concrete = constants.%Call] {} {}
// CHECK:STDOUT:   %CallGeneric.decl: %CallGeneric.type = fn_decl @CallGeneric [concrete = constants.%CallGeneric] {
// CHECK:STDOUT:     %T.patt: %pattern_type.e04 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc26: type = splice_block %A.type [concrete = constants.%A.type.f82] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %A.ref: %A.type.464 = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:       %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:       %A.type: type = facet_type <@A, @A(constants.%X)> [concrete = constants.%A.type.f82]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc26_16.2: %A.type.f82 = symbolic_binding T, 0 [symbolic = %T.loc26_16.1 (constants.%T.19a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallIndirect.decl: %CallIndirect.type = fn_decl @CallIndirect [concrete = constants.%CallIndirect] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @A(%T.loc5_13.2: type) {
// CHECK:STDOUT:   %T.loc5_13.1: type = symbolic_binding T, 0 [symbolic = %T.loc5_13.1 (constants.%T.67d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.type: type = facet_type <@A, @A(%T.loc5_13.1)> [symbolic = %A.type (constants.%A.type.314)]
// CHECK:STDOUT:   %Self.loc5_23.2: @A.%A.type (%A.type.314) = symbolic_binding Self, 1 [symbolic = %Self.loc5_23.2 (constants.%Self.9b1)]
// CHECK:STDOUT:   %A.F.type: type = fn_type @A.F, @A(%T.loc5_13.1) [symbolic = %A.F.type (constants.%A.F.type.71b)]
// CHECK:STDOUT:   %A.F: @A.%A.F.type (%A.F.type.71b) = struct_value () [symbolic = %A.F (constants.%A.F.155)]
// CHECK:STDOUT:   %A.assoc_type: type = assoc_entity_type @A, @A(%T.loc5_13.1) [symbolic = %A.assoc_type (constants.%A.assoc_type.9f4)]
// CHECK:STDOUT:   %assoc0.loc6_41.2: @A.%A.assoc_type (%A.assoc_type.9f4) = assoc_entity element0, %A.F.decl [symbolic = %assoc0.loc6_41.2 (constants.%assoc0.c2b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc5_23.1: @A.%A.type (%A.type.314) = symbolic_binding Self, 1 [symbolic = %Self.loc5_23.2 (constants.%Self.9b1)]
// CHECK:STDOUT:     %A.F.decl: @A.%A.F.type (%A.F.type.71b) = fn_decl @A.F [symbolic = @A.%A.F (constants.%A.F.155)] {
// CHECK:STDOUT:       %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 2 [concrete]
// CHECK:STDOUT:       %u.patt: @A.F.%pattern_type.loc6_18 (%pattern_type.986) = value_binding_pattern u [concrete]
// CHECK:STDOUT:       %u.param_patt: @A.F.%pattern_type.loc6_18 (%pattern_type.986) = value_param_pattern %u.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @A.F.%pattern_type.loc6_25 (%pattern_type.a5f) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @A.F.%pattern_type.loc6_25 (%pattern_type.a5f) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @A.%T.loc5_13.2 [symbolic = %T (constants.%T.67d)]
// CHECK:STDOUT:       %.loc6_32: @A.F.%A.type (%A.type.314) = specific_constant @A.%Self.loc5_23.1, @A(constants.%T.67d) [symbolic = %Self (constants.%Self.9b1)]
// CHECK:STDOUT:       %Self.ref: @A.F.%A.type (%A.type.314) = name_ref Self, %.loc6_32 [symbolic = %Self (constants.%Self.9b1)]
// CHECK:STDOUT:       %U.ref.loc6_38: type = name_ref U, %U.loc6_8.2 [symbolic = %U.loc6_8.1 (constants.%U.4e7)]
// CHECK:STDOUT:       %ptr.loc6_39: type = ptr_type %U.ref.loc6_38 [symbolic = %ptr.loc6_22.1 (constants.%ptr.8f6)]
// CHECK:STDOUT:       %.loc6_40.1: @A.F.%tuple.type.loc6_40.1 (%tuple.type.f42) = tuple_literal (%T.ref, %Self.ref, %ptr.loc6_39) [symbolic = %tuple (constants.%tuple.829)]
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type constants.%Self.9b1 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.78c)]
// CHECK:STDOUT:       %.loc6_40.2: type = converted constants.%Self.9b1, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.78c)]
// CHECK:STDOUT:       %.loc6_40.3: type = converted %.loc6_40.1, constants.%tuple.type.d56 [symbolic = %tuple.type.loc6_40.2 (constants.%tuple.type.d56)]
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %U.loc6_8.2: type = symbolic_binding U, 2 [symbolic = %U.loc6_8.1 (constants.%U.4e7)]
// CHECK:STDOUT:       %u.param: @A.F.%ptr.loc6_22.1 (%ptr.8f6) = value_param call_param0
// CHECK:STDOUT:       %.loc6_22: type = splice_block %ptr.loc6_22.2 [symbolic = %ptr.loc6_22.1 (constants.%ptr.8f6)] {
// CHECK:STDOUT:         %U.ref.loc6_21: type = name_ref U, %U.loc6_8.2 [symbolic = %U.loc6_8.1 (constants.%U.4e7)]
// CHECK:STDOUT:         %ptr.loc6_22.2: type = ptr_type %U.ref.loc6_21 [symbolic = %ptr.loc6_22.1 (constants.%ptr.8f6)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %u: @A.F.%ptr.loc6_22.1 (%ptr.8f6) = value_binding u, %u.param
// CHECK:STDOUT:       %return.param: ref @A.F.%tuple.type.loc6_40.2 (%tuple.type.d56) = out_param call_param1
// CHECK:STDOUT:       %return: ref @A.F.%tuple.type.loc6_40.2 (%tuple.type.d56) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc6_41.1: @A.%A.assoc_type (%A.assoc_type.9f4) = assoc_entity element0, %A.F.decl [symbolic = %assoc0.loc6_41.2 (constants.%assoc0.c2b)]
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
// CHECK:STDOUT:     %u.patt: @Y.as.A.impl.F.%pattern_type.loc16_18 (%pattern_type.4f4b84.1) = value_binding_pattern u [concrete]
// CHECK:STDOUT:     %u.param_patt: @Y.as.A.impl.F.%pattern_type.loc16_18 (%pattern_type.4f4b84.1) = value_param_pattern %u.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @Y.as.A.impl.F.%pattern_type.loc16_25 (%pattern_type.06d) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @Y.as.A.impl.F.%pattern_type.loc16_25 (%pattern_type.06d) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %Y.ref: type = name_ref Y, file.%Y.decl [concrete = constants.%Y]
// CHECK:STDOUT:     %U.ref.loc16_35: type = name_ref U, %U.loc16_8.2 [symbolic = %U.loc16_8.1 (constants.%U.67d)]
// CHECK:STDOUT:     %ptr.loc16_36: type = ptr_type %U.ref.loc16_35 [symbolic = %ptr.loc16_22.1 (constants.%ptr.e8f8f9.1)]
// CHECK:STDOUT:     %.loc16_37.1: %tuple.type.ff9 = tuple_literal (%X.ref, %Y.ref, %ptr.loc16_36) [symbolic = %tuple (constants.%tuple.7ab)]
// CHECK:STDOUT:     %.loc16_37.2: type = converted %.loc16_37.1, constants.%tuple.type.244 [symbolic = %tuple.type.loc16 (constants.%tuple.type.244)]
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %U.loc16_8.2: type = symbolic_binding U, 0 [symbolic = %U.loc16_8.1 (constants.%U.67d)]
// CHECK:STDOUT:     %u.param: @Y.as.A.impl.F.%ptr.loc16_22.1 (%ptr.e8f8f9.1) = value_param call_param0
// CHECK:STDOUT:     %.loc16_22: type = splice_block %ptr.loc16_22.2 [symbolic = %ptr.loc16_22.1 (constants.%ptr.e8f8f9.1)] {
// CHECK:STDOUT:       %U.ref.loc16_21: type = name_ref U, %U.loc16_8.2 [symbolic = %U.loc16_8.1 (constants.%U.67d)]
// CHECK:STDOUT:       %ptr.loc16_22.2: type = ptr_type %U.ref.loc16_21 [symbolic = %ptr.loc16_22.1 (constants.%ptr.e8f8f9.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: @Y.as.A.impl.F.%ptr.loc16_22.1 (%ptr.e8f8f9.1) = value_binding u, %u.param
// CHECK:STDOUT:     %return.param: ref @Y.as.A.impl.F.%tuple.type.loc16 (%tuple.type.244) = out_param call_param1
// CHECK:STDOUT:     %return: ref @Y.as.A.impl.F.%tuple.type.loc16 (%tuple.type.244) = return_slot %return.param
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
// CHECK:STDOUT: generic fn @A.F(@A.%T.loc5_13.2: type, @A.%Self.loc5_23.1: @A.%A.type (%A.type.314), %U.loc6_8.2: type) {
// CHECK:STDOUT:   %U.loc6_8.1: type = symbolic_binding U, 2 [symbolic = %U.loc6_8.1 (constants.%U.4e7)]
// CHECK:STDOUT:   %ptr.loc6_22.1: type = ptr_type %U.loc6_8.1 [symbolic = %ptr.loc6_22.1 (constants.%ptr.8f6)]
// CHECK:STDOUT:   %pattern_type.loc6_18: type = pattern_type %ptr.loc6_22.1 [symbolic = %pattern_type.loc6_18 (constants.%pattern_type.986)]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T.67d)]
// CHECK:STDOUT:   %A.type: type = facet_type <@A, @A(%T)> [symbolic = %A.type (constants.%A.type.314)]
// CHECK:STDOUT:   %Self: @A.F.%A.type (%A.type.314) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.9b1)]
// CHECK:STDOUT:   %tuple.type.loc6_40.1: type = tuple_type (type, %A.type, type) [symbolic = %tuple.type.loc6_40.1 (constants.%tuple.type.f42)]
// CHECK:STDOUT:   %tuple: @A.F.%tuple.type.loc6_40.1 (%tuple.type.f42) = tuple_value (%T, %Self, %ptr.loc6_22.1) [symbolic = %tuple (constants.%tuple.829)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.78c)]
// CHECK:STDOUT:   %tuple.type.loc6_40.2: type = tuple_type (%T, %Self.binding.as_type, %ptr.loc6_22.1) [symbolic = %tuple.type.loc6_40.2 (constants.%tuple.type.d56)]
// CHECK:STDOUT:   %pattern_type.loc6_25: type = pattern_type %tuple.type.loc6_40.2 [symbolic = %pattern_type.loc6_25 (constants.%pattern_type.a5f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param: @A.F.%ptr.loc6_22.1 (%ptr.8f6)) -> @A.F.%tuple.type.loc6_40.2 (%tuple.type.d56);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Y.as.A.impl.F(%U.loc16_8.2: type) {
// CHECK:STDOUT:   %U.loc16_8.1: type = symbolic_binding U, 0 [symbolic = %U.loc16_8.1 (constants.%U.67d)]
// CHECK:STDOUT:   %ptr.loc16_22.1: type = ptr_type %U.loc16_8.1 [symbolic = %ptr.loc16_22.1 (constants.%ptr.e8f8f9.1)]
// CHECK:STDOUT:   %pattern_type.loc16_18: type = pattern_type %ptr.loc16_22.1 [symbolic = %pattern_type.loc16_18 (constants.%pattern_type.4f4b84.1)]
// CHECK:STDOUT:   %tuple: %tuple.type.ff9 = tuple_value (constants.%X, constants.%Y, %ptr.loc16_22.1) [symbolic = %tuple (constants.%tuple.7ab)]
// CHECK:STDOUT:   %tuple.type.loc16: type = tuple_type (constants.%X, constants.%Y, %ptr.loc16_22.1) [symbolic = %tuple.type.loc16 (constants.%tuple.type.244)]
// CHECK:STDOUT:   %pattern_type.loc16_25: type = pattern_type %tuple.type.loc16 [symbolic = %pattern_type.loc16_25 (constants.%pattern_type.06d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc16_37: <witness> = require_complete_type %tuple.type.loc16 [symbolic = %require_complete.loc16_37 (constants.%require_complete.034)]
// CHECK:STDOUT:   %require_complete.loc16_19: <witness> = require_complete_type %ptr.loc16_22.1 [symbolic = %require_complete.loc16_19 (constants.%require_complete.ef162c.1)]
// CHECK:STDOUT:   %tuple.type.loc17: type = tuple_type (constants.%empty_struct_type, constants.%empty_struct_type, %ptr.loc16_22.1) [symbolic = %tuple.type.loc17 (constants.%tuple.type.dd2)]
// CHECK:STDOUT:   %.loc17_21.1: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%U.loc16_8.1) [symbolic = %.loc17_21.1 (constants.%.841)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr.loc16_22.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.2e6)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.loc16_22.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet.29b)]
// CHECK:STDOUT:   %.loc17_21.2: type = fn_type_with_self_type constants.%Copy.Op.type, %Copy.facet [symbolic = %.loc17_21.2 (constants.%.cb5)]
// CHECK:STDOUT:   %impl.elem0.loc17_21.2: @Y.as.A.impl.F.%.loc17_21.2 (%.cb5) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc17_21.2 (constants.%impl.elem0.771)]
// CHECK:STDOUT:   %specific_impl_fn.loc17_21.2: <specific function> = specific_impl_function %impl.elem0.loc17_21.2, @Copy.Op(%Copy.facet) [symbolic = %specific_impl_fn.loc17_21.2 (constants.%specific_impl_fn.b85)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param: @Y.as.A.impl.F.%ptr.loc16_22.1 (%ptr.e8f8f9.1)) -> %return.param: @Y.as.A.impl.F.%tuple.type.loc16 (%tuple.type.244) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc17_14.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc17_18.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %u.ref: @Y.as.A.impl.F.%ptr.loc16_22.1 (%ptr.e8f8f9.1) = name_ref u, %u
// CHECK:STDOUT:     %.loc17_22.1: @Y.as.A.impl.F.%tuple.type.loc17 (%tuple.type.dd2) = tuple_literal (%.loc17_14.1, %.loc17_18.1, %u.ref)
// CHECK:STDOUT:     %tuple.elem0: ref %X = tuple_access %return, element0
// CHECK:STDOUT:     %.loc17_14.2: init %X = class_init (), %tuple.elem0 [concrete = constants.%X.val]
// CHECK:STDOUT:     %.loc17_22.2: init %X = converted %.loc17_14.1, %.loc17_14.2 [concrete = constants.%X.val]
// CHECK:STDOUT:     %tuple.elem1: ref %Y = tuple_access %return, element1
// CHECK:STDOUT:     %.loc17_18.2: init %Y = class_init (), %tuple.elem1 [concrete = constants.%Y.val]
// CHECK:STDOUT:     %.loc17_22.3: init %Y = converted %.loc17_18.1, %.loc17_18.2 [concrete = constants.%Y.val]
// CHECK:STDOUT:     %impl.elem0.loc17_21.1: @Y.as.A.impl.F.%.loc17_21.2 (%.cb5) = impl_witness_access constants.%Copy.lookup_impl_witness.2e6, element0 [symbolic = %impl.elem0.loc17_21.2 (constants.%impl.elem0.771)]
// CHECK:STDOUT:     %bound_method.loc17_21.1: <bound method> = bound_method %u.ref, %impl.elem0.loc17_21.1
// CHECK:STDOUT:     %specific_impl_fn.loc17_21.1: <specific function> = specific_impl_function %impl.elem0.loc17_21.1, @Copy.Op(constants.%Copy.facet.29b) [symbolic = %specific_impl_fn.loc17_21.2 (constants.%specific_impl_fn.b85)]
// CHECK:STDOUT:     %bound_method.loc17_21.2: <bound method> = bound_method %u.ref, %specific_impl_fn.loc17_21.1
// CHECK:STDOUT:     %Copy.Op.call: init @Y.as.A.impl.F.%ptr.loc16_22.1 (%ptr.e8f8f9.1) = call %bound_method.loc17_21.2(%u.ref)
// CHECK:STDOUT:     %tuple.elem2: ref @Y.as.A.impl.F.%ptr.loc16_22.1 (%ptr.e8f8f9.1) = tuple_access %return, element2
// CHECK:STDOUT:     %.loc17_22.4: init @Y.as.A.impl.F.%ptr.loc16_22.1 (%ptr.e8f8f9.1) = initialize_from %Copy.Op.call to %tuple.elem2
// CHECK:STDOUT:     %.loc17_22.5: init @Y.as.A.impl.F.%tuple.type.loc16 (%tuple.type.244) = tuple_init (%.loc17_22.2, %.loc17_22.3, %.loc17_22.4) to %return
// CHECK:STDOUT:     %.loc17_23: init @Y.as.A.impl.F.%tuple.type.loc16 (%tuple.type.244) = converted %.loc17_22.1, %.loc17_22.5
// CHECK:STDOUT:     return %.loc17_23 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %u.patt: %pattern_type.bdb = ref_binding_pattern u [concrete]
// CHECK:STDOUT:     %u.var_patt: %pattern_type.bdb = var_pattern %u.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %u.var: ref %Z = var %u.var_patt
// CHECK:STDOUT:   %Z.ref.loc22: type = name_ref Z, file.%Z.decl [concrete = constants.%Z]
// CHECK:STDOUT:   %u: ref %Z = ref_binding u, %u.var
// CHECK:STDOUT:   %Y.ref: type = name_ref Y, file.%Y.decl [concrete = constants.%Y]
// CHECK:STDOUT:   %A.ref: %A.type.464 = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:   %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %A.type: type = facet_type <@A, @A(constants.%X)> [concrete = constants.%A.type.f82]
// CHECK:STDOUT:   %A.facet: %A.type.f82 = facet_value %Y.ref, (constants.%A.impl_witness) [concrete = constants.%A.facet]
// CHECK:STDOUT:   %.loc23_6: %A.type.f82 = converted %Y.ref, %A.facet [concrete = constants.%A.facet]
// CHECK:STDOUT:   %.loc23_14.1: %A.assoc_type.f05 = specific_constant @A.%assoc0.loc6_41.1, @A(constants.%X) [concrete = constants.%assoc0.249]
// CHECK:STDOUT:   %F.ref: %A.assoc_type.f05 = name_ref F, %.loc23_14.1 [concrete = constants.%assoc0.249]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc23_6 [concrete = constants.%Y]
// CHECK:STDOUT:   %.loc23_14.2: type = converted %.loc23_6, %as_type [concrete = constants.%Y]
// CHECK:STDOUT:   %impl.elem0: %.291 = impl_witness_access constants.%A.impl_witness, element0 [concrete = constants.%Y.as.A.impl.F]
// CHECK:STDOUT:   %Z.ref.loc23: type = name_ref Z, file.%Z.decl [concrete = constants.%Z]
// CHECK:STDOUT:   %u.ref: ref %Z = name_ref u, %u
// CHECK:STDOUT:   %addr: %ptr.2a0 = addr_of %u.ref
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Y.as.A.impl.F(constants.%Z) [concrete = constants.%Y.as.A.impl.F.specific_fn]
// CHECK:STDOUT:   %.loc23_22.1: ref %tuple.type.847 = temporary_storage
// CHECK:STDOUT:   %Y.as.A.impl.F.call: init %tuple.type.847 = call %specific_fn(%addr) to %.loc23_22.1
// CHECK:STDOUT:   %.loc23_22.2: ref %tuple.type.847 = temporary %.loc23_22.1, %Y.as.A.impl.F.call
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc23: <bound method> = bound_method %.loc23_22.2, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.5c5
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.5c5, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.f1e) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.6f7]
// CHECK:STDOUT:   %bound_method.loc23: <bound method> = bound_method %.loc23_22.2, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc23: init %empty_tuple.type = call %bound_method.loc23(%.loc23_22.2)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc22: <bound method> = bound_method %u.var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.72c
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.2: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.72c, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.5de) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.8a3]
// CHECK:STDOUT:   %bound_method.loc22: <bound method> = bound_method %u.var, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc22: init %empty_tuple.type = call %bound_method.loc22(%u.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallGeneric(%T.loc26_16.2: %A.type.f82) {
// CHECK:STDOUT:   %T.loc26_16.1: %A.type.f82 = symbolic_binding T, 0 [symbolic = %T.loc26_16.1 (constants.%T.19a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc26_16.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.935)]
// CHECK:STDOUT:   %A.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc26_16.1, @A, @A(constants.%X) [symbolic = %A.lookup_impl_witness (constants.%A.lookup_impl_witness)]
// CHECK:STDOUT:   %.loc28_4.3: type = fn_type_with_self_type constants.%A.F.type.49b, %T.loc26_16.1 [symbolic = %.loc28_4.3 (constants.%.14f)]
// CHECK:STDOUT:   %impl.elem0.loc28_4.2: @CallGeneric.%.loc28_4.3 (%.14f) = impl_witness_access %A.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc28_4.2 (constants.%impl.elem0.0bc)]
// CHECK:STDOUT:   %specific_impl_fn.loc28_4.2: <specific function> = specific_impl_function %impl.elem0.loc28_4.2, @A.F(constants.%X, %T.loc26_16.1, constants.%Z) [symbolic = %specific_impl_fn.loc28_4.2 (constants.%specific_impl_fn.df5)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (constants.%X, %T.binding.as_type, constants.%ptr.2a0) [symbolic = %tuple.type (constants.%tuple.type.d09)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %tuple.type [symbolic = %require_complete (constants.%require_complete.e14)]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %tuple.type, () [symbolic = %facet_value (constants.%facet_value.c0a)]
// CHECK:STDOUT:   %.loc28_12.3: require_specific_def_type = require_specific_def @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [symbolic = %.loc28_12.3 (constants.%.7bb)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness imports.%Destroy.impl_witness_table, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness.3ff)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %tuple.type, (%Destroy.impl_witness) [symbolic = %Destroy.facet (constants.%Destroy.facet.77a)]
// CHECK:STDOUT:   %.loc28_12.4: type = fn_type_with_self_type constants.%Destroy.Op.type, %Destroy.facet [symbolic = %.loc28_12.4 (constants.%.5f4)]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op.type (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.type.ce9)]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op: @CallGeneric.%DestroyT.binding.as_type.as.Destroy.impl.Op.type (%DestroyT.binding.as_type.as.Destroy.impl.Op.type.ce9) = struct_value () [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.11f)]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.loc28: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value) [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.loc28 (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1b1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %u.patt: %pattern_type.bdb = ref_binding_pattern u [concrete]
// CHECK:STDOUT:       %u.var_patt: %pattern_type.bdb = var_pattern %u.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u.var: ref %Z = var %u.var_patt
// CHECK:STDOUT:     %Z.ref.loc27: type = name_ref Z, file.%Z.decl [concrete = constants.%Z]
// CHECK:STDOUT:     %u: ref %Z = ref_binding u, %u.var
// CHECK:STDOUT:     %T.ref: %A.type.f82 = name_ref T, %T.loc26_16.2 [symbolic = %T.loc26_16.1 (constants.%T.19a)]
// CHECK:STDOUT:     %.loc28_4.1: %A.assoc_type.f05 = specific_constant @A.%assoc0.loc6_41.1, @A(constants.%X) [concrete = constants.%assoc0.249]
// CHECK:STDOUT:     %F.ref: %A.assoc_type.f05 = name_ref F, %.loc28_4.1 [concrete = constants.%assoc0.249]
// CHECK:STDOUT:     %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type.935)]
// CHECK:STDOUT:     %.loc28_4.2: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type.935)]
// CHECK:STDOUT:     %impl.elem0.loc28_4.1: @CallGeneric.%.loc28_4.3 (%.14f) = impl_witness_access constants.%A.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc28_4.2 (constants.%impl.elem0.0bc)]
// CHECK:STDOUT:     %Z.ref.loc28: type = name_ref Z, file.%Z.decl [concrete = constants.%Z]
// CHECK:STDOUT:     %u.ref: ref %Z = name_ref u, %u
// CHECK:STDOUT:     %addr: %ptr.2a0 = addr_of %u.ref
// CHECK:STDOUT:     %specific_impl_fn.loc28_4.1: <specific function> = specific_impl_function %impl.elem0.loc28_4.1, @A.F(constants.%X, constants.%T.19a, constants.%Z) [symbolic = %specific_impl_fn.loc28_4.2 (constants.%specific_impl_fn.df5)]
// CHECK:STDOUT:     %.loc28_12.1: ref @CallGeneric.%tuple.type (%tuple.type.d09) = temporary_storage
// CHECK:STDOUT:     %A.F.call: init @CallGeneric.%tuple.type (%tuple.type.d09) = call %specific_impl_fn.loc28_4.1(%addr) to %.loc28_12.1
// CHECK:STDOUT:     %.loc28_12.2: ref @CallGeneric.%tuple.type (%tuple.type.d09) = temporary %.loc28_12.1, %A.F.call
// CHECK:STDOUT:     %impl.elem0.loc28_12: @CallGeneric.%.loc28_12.4 (%.5f4) = impl_witness_access constants.%Destroy.impl_witness.3ff, element0 [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.11f)]
// CHECK:STDOUT:     %bound_method.loc28_12.1: <bound method> = bound_method %.loc28_12.2, %impl.elem0.loc28_12
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0.loc28_12, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.c0a) [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.loc28 (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1b1)]
// CHECK:STDOUT:     %bound_method.loc28_12.2: <bound method> = bound_method %.loc28_12.2, %specific_fn
// CHECK:STDOUT:     %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc28: init %empty_tuple.type = call %bound_method.loc28_12.2(%.loc28_12.2)
// CHECK:STDOUT:     %DestroyT.binding.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %u.var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.72c
// CHECK:STDOUT:     %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.72c, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.5de) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.8a3]
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
// CHECK:STDOUT:   %A.facet: %A.type.f82 = facet_value %Y.ref, (constants.%A.impl_witness) [concrete = constants.%A.facet]
// CHECK:STDOUT:   %.loc32: %A.type.f82 = converted %Y.ref, %A.facet [concrete = constants.%A.facet]
// CHECK:STDOUT:   %CallGeneric.specific_fn: <specific function> = specific_function %CallGeneric.ref, @CallGeneric(constants.%A.facet) [concrete = constants.%CallGeneric.specific_fn]
// CHECK:STDOUT:   %CallGeneric.call: init %empty_tuple.type = call %CallGeneric.specific_fn()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%T.67d) {
// CHECK:STDOUT:   %T.loc5_13.1 => constants.%T.67d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.F(constants.%T.67d, constants.%Self.9b1, constants.%U.4e7) {
// CHECK:STDOUT:   %U.loc6_8.1 => constants.%U.4e7
// CHECK:STDOUT:   %ptr.loc6_22.1 => constants.%ptr.8f6
// CHECK:STDOUT:   %pattern_type.loc6_18 => constants.%pattern_type.986
// CHECK:STDOUT:   %T => constants.%T.67d
// CHECK:STDOUT:   %A.type => constants.%A.type.314
// CHECK:STDOUT:   %Self => constants.%Self.9b1
// CHECK:STDOUT:   %tuple.type.loc6_40.1 => constants.%tuple.type.f42
// CHECK:STDOUT:   %tuple => constants.%tuple.829
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.78c
// CHECK:STDOUT:   %tuple.type.loc6_40.2 => constants.%tuple.type.d56
// CHECK:STDOUT:   %pattern_type.loc6_25 => constants.%pattern_type.a5f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%X) {
// CHECK:STDOUT:   %T.loc5_13.1 => constants.%X
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.type => constants.%A.type.f82
// CHECK:STDOUT:   %Self.loc5_23.2 => constants.%Self.aab
// CHECK:STDOUT:   %A.F.type => constants.%A.F.type.49b
// CHECK:STDOUT:   %A.F => constants.%A.F.768
// CHECK:STDOUT:   %A.assoc_type => constants.%A.assoc_type.f05
// CHECK:STDOUT:   %assoc0.loc6_41.2 => constants.%assoc0.249
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y.as.A.impl.F(constants.%U.67d) {
// CHECK:STDOUT:   %U.loc16_8.1 => constants.%U.67d
// CHECK:STDOUT:   %ptr.loc16_22.1 => constants.%ptr.e8f8f9.1
// CHECK:STDOUT:   %pattern_type.loc16_18 => constants.%pattern_type.4f4b84.1
// CHECK:STDOUT:   %tuple => constants.%tuple.7ab
// CHECK:STDOUT:   %tuple.type.loc16 => constants.%tuple.type.244
// CHECK:STDOUT:   %pattern_type.loc16_25 => constants.%pattern_type.06d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.F(constants.%X, constants.%A.facet, constants.%U.67d) {
// CHECK:STDOUT:   %U.loc6_8.1 => constants.%U.67d
// CHECK:STDOUT:   %ptr.loc6_22.1 => constants.%ptr.e8f8f9.1
// CHECK:STDOUT:   %pattern_type.loc6_18 => constants.%pattern_type.4f4b84.1
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.f82
// CHECK:STDOUT:   %Self => constants.%A.facet
// CHECK:STDOUT:   %tuple.type.loc6_40.1 => constants.%tuple.type.298
// CHECK:STDOUT:   %tuple => constants.%tuple.74e
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Y
// CHECK:STDOUT:   %tuple.type.loc6_40.2 => constants.%tuple.type.244
// CHECK:STDOUT:   %pattern_type.loc6_25 => constants.%pattern_type.06d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y.as.A.impl.F(constants.%Z) {
// CHECK:STDOUT:   %U.loc16_8.1 => constants.%Z
// CHECK:STDOUT:   %ptr.loc16_22.1 => constants.%ptr.2a0
// CHECK:STDOUT:   %pattern_type.loc16_18 => constants.%pattern_type.771
// CHECK:STDOUT:   %tuple => constants.%tuple.84f
// CHECK:STDOUT:   %tuple.type.loc16 => constants.%tuple.type.847
// CHECK:STDOUT:   %pattern_type.loc16_25 => constants.%pattern_type.79c
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc16_37 => constants.%complete_type.28e
// CHECK:STDOUT:   %require_complete.loc16_19 => constants.%complete_type.543
// CHECK:STDOUT:   %tuple.type.loc17 => constants.%tuple.type.0dd
// CHECK:STDOUT:   %.loc17_21.1 => constants.%.3fe
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.impl_witness.dc6
// CHECK:STDOUT:   %Copy.facet => constants.%Copy.facet.f23
// CHECK:STDOUT:   %.loc17_21.2 => constants.%.da6
// CHECK:STDOUT:   %impl.elem0.loc17_21.2 => constants.%ptr.as.Copy.impl.Op.7d1
// CHECK:STDOUT:   %specific_impl_fn.loc17_21.2 => constants.%ptr.as.Copy.impl.Op.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGeneric(constants.%T.19a) {
// CHECK:STDOUT:   %T.loc26_16.1 => constants.%T.19a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.F(constants.%X, constants.%T.19a, constants.%Z) {
// CHECK:STDOUT:   %U.loc6_8.1 => constants.%Z
// CHECK:STDOUT:   %ptr.loc6_22.1 => constants.%ptr.2a0
// CHECK:STDOUT:   %pattern_type.loc6_18 => constants.%pattern_type.771
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.f82
// CHECK:STDOUT:   %Self => constants.%T.19a
// CHECK:STDOUT:   %tuple.type.loc6_40.1 => constants.%tuple.type.298
// CHECK:STDOUT:   %tuple => constants.%tuple.8a5
// CHECK:STDOUT:   %Self.binding.as_type => constants.%T.binding.as_type.935
// CHECK:STDOUT:   %tuple.type.loc6_40.2 => constants.%tuple.type.d09
// CHECK:STDOUT:   %pattern_type.loc6_25 => constants.%pattern_type.93b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGeneric(constants.%A.facet) {
// CHECK:STDOUT:   %T.loc26_16.1 => constants.%A.facet
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.binding.as_type => constants.%Y
// CHECK:STDOUT:   %A.lookup_impl_witness => constants.%A.impl_witness
// CHECK:STDOUT:   %.loc28_4.3 => constants.%.291
// CHECK:STDOUT:   %impl.elem0.loc28_4.2 => constants.%Y.as.A.impl.F
// CHECK:STDOUT:   %specific_impl_fn.loc28_4.2 => constants.%Y.as.A.impl.F.specific_fn
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.847
// CHECK:STDOUT:   %require_complete => constants.%complete_type.28e
// CHECK:STDOUT:   %facet_value => constants.%facet_value.f1e
// CHECK:STDOUT:   %.loc28_12.3 => constants.%.daf
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.a85
// CHECK:STDOUT:   %Destroy.facet => constants.%Destroy.facet.842
// CHECK:STDOUT:   %.loc28_12.4 => constants.%.22c
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type => constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.type.7fc
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op => constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.5c5
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.loc28 => constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.6f7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.F(constants.%X, constants.%A.facet, constants.%Z) {
// CHECK:STDOUT:   %U.loc6_8.1 => constants.%Z
// CHECK:STDOUT:   %ptr.loc6_22.1 => constants.%ptr.2a0
// CHECK:STDOUT:   %pattern_type.loc6_18 => constants.%pattern_type.771
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.f82
// CHECK:STDOUT:   %Self => constants.%A.facet
// CHECK:STDOUT:   %tuple.type.loc6_40.1 => constants.%tuple.type.298
// CHECK:STDOUT:   %tuple => constants.%tuple.f66
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Y
// CHECK:STDOUT:   %tuple.type.loc6_40.2 => constants.%tuple.type.847
// CHECK:STDOUT:   %pattern_type.loc6_25 => constants.%pattern_type.79c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- generic_method_more_params.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %A.type.464: type = generic_interface_type @A [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %A.generic: %A.type.464 = struct_value () [concrete]
// CHECK:STDOUT:   %A.type.314: type = facet_type <@A, @A(%T.67d)> [symbolic]
// CHECK:STDOUT:   %Self.9b1: %A.type.314 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %U.4e7: type = symbolic_binding U, 2 [symbolic]
// CHECK:STDOUT:   %pattern_type.62e: type = pattern_type %U.4e7 [symbolic]
// CHECK:STDOUT:   %tuple.type.f42: type = tuple_type (type, %A.type.314, type) [symbolic]
// CHECK:STDOUT:   %tuple.28e: %tuple.type.f42 = tuple_value (%T.67d, %Self.9b1, %U.4e7) [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.78c: type = symbolic_binding_type Self, 1, %Self.9b1 [symbolic]
// CHECK:STDOUT:   %tuple.type.6a9: type = tuple_type (%T.67d, %Self.binding.as_type.78c, %U.4e7) [symbolic]
// CHECK:STDOUT:   %pattern_type.0dd: type = pattern_type %tuple.type.6a9 [symbolic]
// CHECK:STDOUT:   %A.F.type.71b: type = fn_type @A.F, @A(%T.67d) [symbolic]
// CHECK:STDOUT:   %A.F.155: %A.F.type.71b = struct_value () [symbolic]
// CHECK:STDOUT:   %A.assoc_type.9f4: type = assoc_entity_type @A, @A(%T.67d) [symbolic]
// CHECK:STDOUT:   %assoc0.c2b: %A.assoc_type.9f4 = assoc_entity element0, @A.%A.F.decl [symbolic]
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
// CHECK:STDOUT:   %A.type.62d: type = facet_type <@A, @A(%W)> [symbolic]
// CHECK:STDOUT:   %A.impl_witness.d95: <witness> = impl_witness file.%A.impl_witness_table, @tuple.type.as.A.impl(%V1, %V2, %W) [symbolic]
// CHECK:STDOUT:   %Self.6ee: %A.type.62d = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %A.F.type.0cd: type = fn_type @A.F, @A(%W) [symbolic]
// CHECK:STDOUT:   %A.F.cef: %A.F.type.0cd = struct_value () [symbolic]
// CHECK:STDOUT:   %A.assoc_type.7a9: type = assoc_entity_type @A, @A(%W) [symbolic]
// CHECK:STDOUT:   %assoc0.264: %A.assoc_type.7a9 = assoc_entity element0, @A.%A.F.decl [symbolic]
// CHECK:STDOUT:   %require_complete.4be: <witness> = require_complete_type %A.type.62d [symbolic]
// CHECK:STDOUT:   %U.ce4: type = symbolic_binding U, 3 [symbolic]
// CHECK:STDOUT:   %pattern_type.e4a: type = pattern_type %U.ce4 [symbolic]
// CHECK:STDOUT:   %tuple.type.11f: type = tuple_type (type, %tuple.type.24b, type) [concrete]
// CHECK:STDOUT:   %tuple.117: %tuple.type.11f = tuple_value (%W, %tuple.4b9, %U.ce4) [symbolic]
// CHECK:STDOUT:   %tuple.type.897: type = tuple_type (%W, %tuple.type.a5e, %U.ce4) [symbolic]
// CHECK:STDOUT:   %pattern_type.5cc: type = pattern_type %tuple.type.897 [symbolic]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type.f4a: type = fn_type @tuple.type.as.A.impl.F, @tuple.type.as.A.impl(%V1, %V2, %W) [symbolic]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.3c3: %tuple.type.as.A.impl.F.type.f4a = struct_value () [symbolic]
// CHECK:STDOUT:   %A.facet.e1a: %A.type.62d = facet_value %tuple.type.a5e, (%A.impl_witness.d95) [symbolic]
// CHECK:STDOUT:   %tuple.type.26b: type = tuple_type (type, %A.type.62d, type) [symbolic]
// CHECK:STDOUT:   %tuple.31e: %tuple.type.26b = tuple_value (%W, %A.facet.e1a, %U.ce4) [symbolic]
// CHECK:STDOUT:   %require_complete.da0: <witness> = require_complete_type %tuple.type.897 [symbolic]
// CHECK:STDOUT:   %require_complete.5a4: <witness> = require_complete_type %U.ce4 [symbolic]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.specific_fn.3d7: <specific function> = specific_function %tuple.type.as.A.impl.F.3c3, @tuple.type.as.A.impl.F(%V1, %V2, %W, %U.ce4) [symbolic]
// CHECK:STDOUT:   %Call.type: type = fn_type @Call [concrete]
// CHECK:STDOUT:   %Call: %Call.type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.dd5: %tuple.type.24b = tuple_value (%Y1, %Y2) [concrete]
// CHECK:STDOUT:   %tuple.type.ab2: type = tuple_type (%Y1, %Y2) [concrete]
// CHECK:STDOUT:   %A.type.f82: type = facet_type <@A, @A(%X)> [concrete]
// CHECK:STDOUT:   %A.impl_witness.098: <witness> = impl_witness file.%A.impl_witness_table, @tuple.type.as.A.impl(%Y1, %Y2, %X) [concrete]
// CHECK:STDOUT:   %Self.aab: %A.type.f82 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %A.F.type.49b: type = fn_type @A.F, @A(%X) [concrete]
// CHECK:STDOUT:   %A.F.768: %A.F.type.49b = struct_value () [concrete]
// CHECK:STDOUT:   %A.assoc_type.f05: type = assoc_entity_type @A, @A(%X) [concrete]
// CHECK:STDOUT:   %assoc0.249: %A.assoc_type.f05 = assoc_entity element0, @A.%A.F.decl [concrete]
// CHECK:STDOUT:   %complete_type.df7: <witness> = complete_type_witness %A.type.f82 [concrete]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type.9c3: type = fn_type @tuple.type.as.A.impl.F, @tuple.type.as.A.impl(%Y1, %Y2, %X) [concrete]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.44b: %tuple.type.as.A.impl.F.type.9c3 = struct_value () [concrete]
// CHECK:STDOUT:   %A.facet.e22: %A.type.f82 = facet_value %tuple.type.ab2, (%A.impl_witness.098) [concrete]
// CHECK:STDOUT:   %.cf4: type = fn_type_with_self_type %A.F.type.49b, %A.facet.e22 [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.bdb: type = pattern_type %Z [concrete]
// CHECK:STDOUT:   %tuple.66d: %tuple.type.11f = tuple_value (%X, %tuple.dd5, %Z) [concrete]
// CHECK:STDOUT:   %tuple.type.65a: type = tuple_type (%X, %tuple.type.ab2, %Z) [concrete]
// CHECK:STDOUT:   %pattern_type.30b: type = pattern_type %tuple.type.65a [concrete]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.specific_fn.6e3: <specific function> = specific_function %tuple.type.as.A.impl.F.44b, @tuple.type.as.A.impl.F(%Y1, %Y2, %X, %Z) [concrete]
// CHECK:STDOUT:   %Z.val: %Z = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %DestroyT: %type_where = symbolic_binding DestroyT, 0 [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.0bf: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%DestroyT) [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.97a: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.0bf = struct_value () [symbolic]
// CHECK:STDOUT:   %facet_value.917: %type_where = facet_value %tuple.type.65a, () [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.776: <witness> = impl_witness imports.%Destroy.impl_witness_table, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.917) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.b70: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.917) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.772: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.b70 = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type.cf2: <witness> = complete_type_witness %tuple.type.65a [concrete]
// CHECK:STDOUT:   %.110: require_specific_def_type = require_specific_def @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.917) [concrete]
// CHECK:STDOUT:   %Destroy.facet.15a: %Destroy.type = facet_value %tuple.type.65a, (%Destroy.impl_witness.776) [concrete]
// CHECK:STDOUT:   %.ee4: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet.15a [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.0b9: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op.772, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value.917) [concrete]
// CHECK:STDOUT:   %facet_value.5de: %type_where = facet_value %Z, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.17b: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.5de) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.72c: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.17b = struct_value () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.8a3: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op.72c, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value.5de) [concrete]
// CHECK:STDOUT:   %T.19a: %A.type.f82 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.e04: type = pattern_type %A.type.f82 [concrete]
// CHECK:STDOUT:   %CallGeneric.type: type = fn_type @CallGeneric [concrete]
// CHECK:STDOUT:   %CallGeneric: %CallGeneric.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.19a [symbolic]
// CHECK:STDOUT:   %A.lookup_impl_witness: <witness> = lookup_impl_witness %T.19a, @A, @A(%X) [symbolic]
// CHECK:STDOUT:   %.14f: type = fn_type_with_self_type %A.F.type.49b, %T.19a [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.14f = impl_witness_access %A.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %tuple.type.298: type = tuple_type (type, %A.type.f82, type) [concrete]
// CHECK:STDOUT:   %tuple.c8d: %tuple.type.298 = tuple_value (%X, %T.19a, %Z) [symbolic]
// CHECK:STDOUT:   %tuple.type.50b: type = tuple_type (%X, %T.binding.as_type, %Z) [symbolic]
// CHECK:STDOUT:   %pattern_type.56e: type = pattern_type %tuple.type.50b [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @A.F(%X, %T.19a, %Z) [symbolic]
// CHECK:STDOUT:   %require_complete.b37: <witness> = require_complete_type %tuple.type.50b [symbolic]
// CHECK:STDOUT:   %facet_value.f56: %type_where = facet_value %tuple.type.50b, () [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.cec: <witness> = impl_witness imports.%Destroy.impl_witness_table, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.f56) [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.a72: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.f56) [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.84d: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.a72 = struct_value () [symbolic]
// CHECK:STDOUT:   %.ac1: require_specific_def_type = require_specific_def @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.f56) [symbolic]
// CHECK:STDOUT:   %Destroy.facet.ddd: %Destroy.type = facet_value %tuple.type.50b, (%Destroy.impl_witness.cec) [symbolic]
// CHECK:STDOUT:   %.b7f: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet.ddd [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.da9: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op.84d, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value.f56) [symbolic]
// CHECK:STDOUT:   %CallIndirect.type: type = fn_type @CallIndirect [concrete]
// CHECK:STDOUT:   %CallIndirect: %CallIndirect.type = struct_value () [concrete]
// CHECK:STDOUT:   %CallGeneric.specific_fn: <specific function> = specific_function %CallGeneric, @CallGeneric(%A.facet.e22) [concrete]
// CHECK:STDOUT:   %tuple.f1f: %tuple.type.298 = tuple_value (%X, %A.facet.e22, %Z) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.93a: @DestroyT.binding.as_type.as.Destroy.impl.%DestroyT.binding.as_type.as.Destroy.impl.Op.type (%DestroyT.binding.as_type.as.Destroy.impl.Op.type.0bf) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @DestroyT.binding.as_type.as.Destroy.impl.%DestroyT.binding.as_type.as.Destroy.impl.Op (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.97a)]
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (%Core.import_ref.93a), @DestroyT.binding.as_type.as.Destroy.impl [concrete]
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
// CHECK:STDOUT:   %A.decl: %A.type.464 = interface_decl @A [concrete = constants.%A.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc5_13.2: type = symbolic_binding T, 0 [symbolic = %T.loc5_13.1 (constants.%T.67d)]
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
// CHECK:STDOUT:     %.loc14_53.1: %tuple.type.24b = tuple_literal (%V1.ref, %V2.ref) [symbolic = %tuple (constants.%tuple.4b9)]
// CHECK:STDOUT:     %.loc14_53.2: type = converted %.loc14_53.1, constants.%tuple.type.a5e [symbolic = %tuple.type (constants.%tuple.type.a5e)]
// CHECK:STDOUT:     %A.ref: %A.type.464 = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:     %W.ref: type = name_ref W, %W.loc14_36.2 [symbolic = %W.loc14_36.1 (constants.%W)]
// CHECK:STDOUT:     %A.type.loc14_61.2: type = facet_type <@A, @A(constants.%W)> [symbolic = %A.type.loc14_61.1 (constants.%A.type.62d)]
// CHECK:STDOUT:     %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %V1.loc14_14.2: type = symbolic_binding V1, 0 [symbolic = %V1.loc14_14.1 (constants.%V1)]
// CHECK:STDOUT:     %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %V2.loc14_25.2: type = symbolic_binding V2, 1 [symbolic = %V2.loc14_25.1 (constants.%V2)]
// CHECK:STDOUT:     %.Self.3: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %W.loc14_36.2: type = symbolic_binding W, 2 [symbolic = %W.loc14_36.1 (constants.%W)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.impl_witness_table = impl_witness_table (@tuple.type.as.A.impl.%tuple.type.as.A.impl.F.decl), @tuple.type.as.A.impl [concrete]
// CHECK:STDOUT:   %A.impl_witness: <witness> = impl_witness %A.impl_witness_table, @tuple.type.as.A.impl(constants.%V1, constants.%V2, constants.%W) [symbolic = @tuple.type.as.A.impl.%A.impl_witness (constants.%A.impl_witness.d95)]
// CHECK:STDOUT:   %Call.decl: %Call.type = fn_decl @Call [concrete = constants.%Call] {} {}
// CHECK:STDOUT:   %CallGeneric.decl: %CallGeneric.type = fn_decl @CallGeneric [concrete = constants.%CallGeneric] {
// CHECK:STDOUT:     %T.patt: %pattern_type.e04 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc27: type = splice_block %A.type [concrete = constants.%A.type.f82] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %A.ref: %A.type.464 = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:       %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:       %A.type: type = facet_type <@A, @A(constants.%X)> [concrete = constants.%A.type.f82]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc27_16.2: %A.type.f82 = symbolic_binding T, 0 [symbolic = %T.loc27_16.1 (constants.%T.19a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallIndirect.decl: %CallIndirect.type = fn_decl @CallIndirect [concrete = constants.%CallIndirect] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @A(%T.loc5_13.2: type) {
// CHECK:STDOUT:   %T.loc5_13.1: type = symbolic_binding T, 0 [symbolic = %T.loc5_13.1 (constants.%T.67d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.type: type = facet_type <@A, @A(%T.loc5_13.1)> [symbolic = %A.type (constants.%A.type.314)]
// CHECK:STDOUT:   %Self.loc5_23.2: @A.%A.type (%A.type.314) = symbolic_binding Self, 1 [symbolic = %Self.loc5_23.2 (constants.%Self.9b1)]
// CHECK:STDOUT:   %A.F.type: type = fn_type @A.F, @A(%T.loc5_13.1) [symbolic = %A.F.type (constants.%A.F.type.71b)]
// CHECK:STDOUT:   %A.F: @A.%A.F.type (%A.F.type.71b) = struct_value () [symbolic = %A.F (constants.%A.F.155)]
// CHECK:STDOUT:   %A.assoc_type: type = assoc_entity_type @A, @A(%T.loc5_13.1) [symbolic = %A.assoc_type (constants.%A.assoc_type.9f4)]
// CHECK:STDOUT:   %assoc0.loc6_39.2: @A.%A.assoc_type (%A.assoc_type.9f4) = assoc_entity element0, %A.F.decl [symbolic = %assoc0.loc6_39.2 (constants.%assoc0.c2b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc5_23.1: @A.%A.type (%A.type.314) = symbolic_binding Self, 1 [symbolic = %Self.loc5_23.2 (constants.%Self.9b1)]
// CHECK:STDOUT:     %A.F.decl: @A.%A.F.type (%A.F.type.71b) = fn_decl @A.F [symbolic = @A.%A.F (constants.%A.F.155)] {
// CHECK:STDOUT:       %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 2 [concrete]
// CHECK:STDOUT:       %u.patt: @A.F.%pattern_type.loc6_18 (%pattern_type.62e) = value_binding_pattern u [concrete]
// CHECK:STDOUT:       %u.param_patt: @A.F.%pattern_type.loc6_18 (%pattern_type.62e) = value_param_pattern %u.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @A.F.%pattern_type.loc6_24 (%pattern_type.0dd) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @A.F.%pattern_type.loc6_24 (%pattern_type.0dd) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @A.%T.loc5_13.2 [symbolic = %T (constants.%T.67d)]
// CHECK:STDOUT:       %.loc6_31: @A.F.%A.type (%A.type.314) = specific_constant @A.%Self.loc5_23.1, @A(constants.%T.67d) [symbolic = %Self (constants.%Self.9b1)]
// CHECK:STDOUT:       %Self.ref: @A.F.%A.type (%A.type.314) = name_ref Self, %.loc6_31 [symbolic = %Self (constants.%Self.9b1)]
// CHECK:STDOUT:       %U.ref.loc6_37: type = name_ref U, %U.loc6_8.2 [symbolic = %U.loc6_8.1 (constants.%U.4e7)]
// CHECK:STDOUT:       %.loc6_38.1: @A.F.%tuple.type.loc6_38.1 (%tuple.type.f42) = tuple_literal (%T.ref, %Self.ref, %U.ref.loc6_37) [symbolic = %tuple (constants.%tuple.28e)]
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type constants.%Self.9b1 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.78c)]
// CHECK:STDOUT:       %.loc6_38.2: type = converted constants.%Self.9b1, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.78c)]
// CHECK:STDOUT:       %.loc6_38.3: type = converted %.loc6_38.1, constants.%tuple.type.6a9 [symbolic = %tuple.type.loc6_38.2 (constants.%tuple.type.6a9)]
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %U.loc6_8.2: type = symbolic_binding U, 2 [symbolic = %U.loc6_8.1 (constants.%U.4e7)]
// CHECK:STDOUT:       %u.param: @A.F.%U.loc6_8.1 (%U.4e7) = value_param call_param0
// CHECK:STDOUT:       %U.ref.loc6_21: type = name_ref U, %U.loc6_8.2 [symbolic = %U.loc6_8.1 (constants.%U.4e7)]
// CHECK:STDOUT:       %u: @A.F.%U.loc6_8.1 (%U.4e7) = value_binding u, %u.param
// CHECK:STDOUT:       %return.param: ref @A.F.%tuple.type.loc6_38.2 (%tuple.type.6a9) = out_param call_param1
// CHECK:STDOUT:       %return: ref @A.F.%tuple.type.loc6_38.2 (%tuple.type.6a9) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc6_39.1: @A.%A.assoc_type (%A.assoc_type.9f4) = assoc_entity element0, %A.F.decl [symbolic = %assoc0.loc6_39.2 (constants.%assoc0.c2b)]
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
// CHECK:STDOUT:   %tuple: %tuple.type.24b = tuple_value (%V1.loc14_14.1, %V2.loc14_25.1) [symbolic = %tuple (constants.%tuple.4b9)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%V1.loc14_14.1, %V2.loc14_25.1) [symbolic = %tuple.type (constants.%tuple.type.a5e)]
// CHECK:STDOUT:   %A.type.loc14_61.1: type = facet_type <@A, @A(%W.loc14_36.1)> [symbolic = %A.type.loc14_61.1 (constants.%A.type.62d)]
// CHECK:STDOUT:   %A.impl_witness: <witness> = impl_witness file.%A.impl_witness_table, @tuple.type.as.A.impl(%V1.loc14_14.1, %V2.loc14_25.1, %W.loc14_36.1) [symbolic = %A.impl_witness (constants.%A.impl_witness.d95)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %A.type.loc14_61.1 [symbolic = %require_complete (constants.%require_complete.4be)]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type: type = fn_type @tuple.type.as.A.impl.F, @tuple.type.as.A.impl(%V1.loc14_14.1, %V2.loc14_25.1, %W.loc14_36.1) [symbolic = %tuple.type.as.A.impl.F.type (constants.%tuple.type.as.A.impl.F.type.f4a)]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F: @tuple.type.as.A.impl.%tuple.type.as.A.impl.F.type (%tuple.type.as.A.impl.F.type.f4a) = struct_value () [symbolic = %tuple.type.as.A.impl.F (constants.%tuple.type.as.A.impl.F.3c3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc14_53.2 as %A.type.loc14_61.2 {
// CHECK:STDOUT:     %tuple.type.as.A.impl.F.decl: @tuple.type.as.A.impl.%tuple.type.as.A.impl.F.type (%tuple.type.as.A.impl.F.type.f4a) = fn_decl @tuple.type.as.A.impl.F [symbolic = @tuple.type.as.A.impl.%tuple.type.as.A.impl.F (constants.%tuple.type.as.A.impl.F.3c3)] {
// CHECK:STDOUT:       %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 3 [concrete]
// CHECK:STDOUT:       %u.patt: @tuple.type.as.A.impl.F.%pattern_type.loc17_18 (%pattern_type.e4a) = value_binding_pattern u [concrete]
// CHECK:STDOUT:       %u.param_patt: @tuple.type.as.A.impl.F.%pattern_type.loc17_18 (%pattern_type.e4a) = value_param_pattern %u.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @tuple.type.as.A.impl.F.%pattern_type.loc17_24 (%pattern_type.5cc) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @tuple.type.as.A.impl.F.%pattern_type.loc17_24 (%pattern_type.5cc) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %W.ref: type = name_ref W, @tuple.type.as.A.impl.%W.loc14_36.2 [symbolic = %W (constants.%W)]
// CHECK:STDOUT:       %V1.ref: type = name_ref V1, @tuple.type.as.A.impl.%V1.loc14_14.2 [symbolic = %V1 (constants.%V1)]
// CHECK:STDOUT:       %V2.ref: type = name_ref V2, @tuple.type.as.A.impl.%V2.loc14_25.2 [symbolic = %V2 (constants.%V2)]
// CHECK:STDOUT:       %.loc17_38: %tuple.type.24b = tuple_literal (%V1.ref, %V2.ref) [symbolic = %tuple.loc17_38 (constants.%tuple.4b9)]
// CHECK:STDOUT:       %U.ref.loc17_41: type = name_ref U, %U.loc17_8.2 [symbolic = %U.loc17_8.1 (constants.%U.ce4)]
// CHECK:STDOUT:       %.loc17_42.1: %tuple.type.11f = tuple_literal (%W.ref, %.loc17_38, %U.ref.loc17_41) [symbolic = %tuple.loc17_42 (constants.%tuple.117)]
// CHECK:STDOUT:       %.loc17_42.2: type = converted constants.%tuple.4b9, constants.%tuple.type.a5e [symbolic = %tuple.type.loc17_42.1 (constants.%tuple.type.a5e)]
// CHECK:STDOUT:       %.loc17_42.3: type = converted %.loc17_42.1, constants.%tuple.type.897 [symbolic = %tuple.type.loc17_42.2 (constants.%tuple.type.897)]
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %U.loc17_8.2: type = symbolic_binding U, 3 [symbolic = %U.loc17_8.1 (constants.%U.ce4)]
// CHECK:STDOUT:       %u.param: @tuple.type.as.A.impl.F.%U.loc17_8.1 (%U.ce4) = value_param call_param0
// CHECK:STDOUT:       %U.ref.loc17_21: type = name_ref U, %U.loc17_8.2 [symbolic = %U.loc17_8.1 (constants.%U.ce4)]
// CHECK:STDOUT:       %u: @tuple.type.as.A.impl.F.%U.loc17_8.1 (%U.ce4) = value_binding u, %u.param
// CHECK:STDOUT:       %return.param: ref @tuple.type.as.A.impl.F.%tuple.type.loc17_42.2 (%tuple.type.897) = out_param call_param1
// CHECK:STDOUT:       %return: ref @tuple.type.as.A.impl.F.%tuple.type.loc17_42.2 (%tuple.type.897) = return_slot %return.param
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
// CHECK:STDOUT: generic fn @A.F(@A.%T.loc5_13.2: type, @A.%Self.loc5_23.1: @A.%A.type (%A.type.314), %U.loc6_8.2: type) {
// CHECK:STDOUT:   %U.loc6_8.1: type = symbolic_binding U, 2 [symbolic = %U.loc6_8.1 (constants.%U.4e7)]
// CHECK:STDOUT:   %pattern_type.loc6_18: type = pattern_type %U.loc6_8.1 [symbolic = %pattern_type.loc6_18 (constants.%pattern_type.62e)]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T.67d)]
// CHECK:STDOUT:   %A.type: type = facet_type <@A, @A(%T)> [symbolic = %A.type (constants.%A.type.314)]
// CHECK:STDOUT:   %Self: @A.F.%A.type (%A.type.314) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.9b1)]
// CHECK:STDOUT:   %tuple.type.loc6_38.1: type = tuple_type (type, %A.type, type) [symbolic = %tuple.type.loc6_38.1 (constants.%tuple.type.f42)]
// CHECK:STDOUT:   %tuple: @A.F.%tuple.type.loc6_38.1 (%tuple.type.f42) = tuple_value (%T, %Self, %U.loc6_8.1) [symbolic = %tuple (constants.%tuple.28e)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.78c)]
// CHECK:STDOUT:   %tuple.type.loc6_38.2: type = tuple_type (%T, %Self.binding.as_type, %U.loc6_8.1) [symbolic = %tuple.type.loc6_38.2 (constants.%tuple.type.6a9)]
// CHECK:STDOUT:   %pattern_type.loc6_24: type = pattern_type %tuple.type.loc6_38.2 [symbolic = %pattern_type.loc6_24 (constants.%pattern_type.0dd)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param: @A.F.%U.loc6_8.1 (%U.4e7)) -> @A.F.%tuple.type.loc6_38.2 (%tuple.type.6a9);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @tuple.type.as.A.impl.F(@tuple.type.as.A.impl.%V1.loc14_14.2: type, @tuple.type.as.A.impl.%V2.loc14_25.2: type, @tuple.type.as.A.impl.%W.loc14_36.2: type, %U.loc17_8.2: type) {
// CHECK:STDOUT:   %U.loc17_8.1: type = symbolic_binding U, 3 [symbolic = %U.loc17_8.1 (constants.%U.ce4)]
// CHECK:STDOUT:   %pattern_type.loc17_18: type = pattern_type %U.loc17_8.1 [symbolic = %pattern_type.loc17_18 (constants.%pattern_type.e4a)]
// CHECK:STDOUT:   %W: type = symbolic_binding W, 2 [symbolic = %W (constants.%W)]
// CHECK:STDOUT:   %V1: type = symbolic_binding V1, 0 [symbolic = %V1 (constants.%V1)]
// CHECK:STDOUT:   %V2: type = symbolic_binding V2, 1 [symbolic = %V2 (constants.%V2)]
// CHECK:STDOUT:   %tuple.loc17_38: %tuple.type.24b = tuple_value (%V1, %V2) [symbolic = %tuple.loc17_38 (constants.%tuple.4b9)]
// CHECK:STDOUT:   %tuple.loc17_42: %tuple.type.11f = tuple_value (%W, %tuple.loc17_38, %U.loc17_8.1) [symbolic = %tuple.loc17_42 (constants.%tuple.117)]
// CHECK:STDOUT:   %tuple.type.loc17_42.1: type = tuple_type (%V1, %V2) [symbolic = %tuple.type.loc17_42.1 (constants.%tuple.type.a5e)]
// CHECK:STDOUT:   %tuple.type.loc17_42.2: type = tuple_type (%W, %tuple.type.loc17_42.1, %U.loc17_8.1) [symbolic = %tuple.type.loc17_42.2 (constants.%tuple.type.897)]
// CHECK:STDOUT:   %pattern_type.loc17_24: type = pattern_type %tuple.type.loc17_42.2 [symbolic = %pattern_type.loc17_24 (constants.%pattern_type.5cc)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc17_42: <witness> = require_complete_type %tuple.type.loc17_42.2 [symbolic = %require_complete.loc17_42 (constants.%require_complete.da0)]
// CHECK:STDOUT:   %require_complete.loc17_19: <witness> = require_complete_type %U.loc17_8.1 [symbolic = %require_complete.loc17_19 (constants.%require_complete.5a4)]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type: type = fn_type @tuple.type.as.A.impl.F, @tuple.type.as.A.impl(%V1, %V2, %W) [symbolic = %tuple.type.as.A.impl.F.type (constants.%tuple.type.as.A.impl.F.type.f4a)]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F: @tuple.type.as.A.impl.F.%tuple.type.as.A.impl.F.type (%tuple.type.as.A.impl.F.type.f4a) = struct_value () [symbolic = %tuple.type.as.A.impl.F (constants.%tuple.type.as.A.impl.F.3c3)]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.specific_fn.loc18_12.2: <specific function> = specific_function %tuple.type.as.A.impl.F, @tuple.type.as.A.impl.F(%V1, %V2, %W, %U.loc17_8.1) [symbolic = %tuple.type.as.A.impl.F.specific_fn.loc18_12.2 (constants.%tuple.type.as.A.impl.F.specific_fn.3d7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param: @tuple.type.as.A.impl.F.%U.loc17_8.1 (%U.ce4)) -> %return.param: @tuple.type.as.A.impl.F.%tuple.type.loc17_42.2 (%tuple.type.897) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc18: @tuple.type.as.A.impl.F.%tuple.type.as.A.impl.F.type (%tuple.type.as.A.impl.F.type.f4a) = specific_constant @tuple.type.as.A.impl.%tuple.type.as.A.impl.F.decl, @tuple.type.as.A.impl(constants.%V1, constants.%V2, constants.%W) [symbolic = %tuple.type.as.A.impl.F (constants.%tuple.type.as.A.impl.F.3c3)]
// CHECK:STDOUT:     %F.ref: @tuple.type.as.A.impl.F.%tuple.type.as.A.impl.F.type (%tuple.type.as.A.impl.F.type.f4a) = name_ref F, %.loc18 [symbolic = %tuple.type.as.A.impl.F (constants.%tuple.type.as.A.impl.F.3c3)]
// CHECK:STDOUT:     %U.ref.loc18: type = name_ref U, %U.loc17_8.2 [symbolic = %U.loc17_8.1 (constants.%U.ce4)]
// CHECK:STDOUT:     %u.ref: @tuple.type.as.A.impl.F.%U.loc17_8.1 (%U.ce4) = name_ref u, %u
// CHECK:STDOUT:     %tuple.type.as.A.impl.F.specific_fn.loc18_12.1: <specific function> = specific_function %F.ref, @tuple.type.as.A.impl.F(constants.%V1, constants.%V2, constants.%W, constants.%U.ce4) [symbolic = %tuple.type.as.A.impl.F.specific_fn.loc18_12.2 (constants.%tuple.type.as.A.impl.F.specific_fn.3d7)]
// CHECK:STDOUT:     %.loc17_24: ref @tuple.type.as.A.impl.F.%tuple.type.loc17_42.2 (%tuple.type.897) = splice_block %return {}
// CHECK:STDOUT:     %tuple.type.as.A.impl.F.call: init @tuple.type.as.A.impl.F.%tuple.type.loc17_42.2 (%tuple.type.897) = call %tuple.type.as.A.impl.F.specific_fn.loc18_12.1(%u.ref) to %.loc17_24
// CHECK:STDOUT:     return %tuple.type.as.A.impl.F.call to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Y1.ref: type = name_ref Y1, file.%Y1.decl [concrete = constants.%Y1]
// CHECK:STDOUT:   %Y2.ref: type = name_ref Y2, file.%Y2.decl [concrete = constants.%Y2]
// CHECK:STDOUT:   %.loc24_12: %tuple.type.24b = tuple_literal (%Y1.ref, %Y2.ref) [concrete = constants.%tuple.dd5]
// CHECK:STDOUT:   %.loc24_14: type = converted %.loc24_12, constants.%tuple.type.ab2 [concrete = constants.%tuple.type.ab2]
// CHECK:STDOUT:   %A.ref: %A.type.464 = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:   %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %A.type: type = facet_type <@A, @A(constants.%X)> [concrete = constants.%A.type.f82]
// CHECK:STDOUT:   %A.facet: %A.type.f82 = facet_value %.loc24_14, (constants.%A.impl_witness.098) [concrete = constants.%A.facet.e22]
// CHECK:STDOUT:   %.loc24_23: %A.type.f82 = converted %.loc24_14, %A.facet [concrete = constants.%A.facet.e22]
// CHECK:STDOUT:   %.loc24_31.1: %A.assoc_type.f05 = specific_constant @A.%assoc0.loc6_39.1, @A(constants.%X) [concrete = constants.%assoc0.249]
// CHECK:STDOUT:   %F.ref: %A.assoc_type.f05 = name_ref F, %.loc24_31.1 [concrete = constants.%assoc0.249]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc24_23 [concrete = constants.%tuple.type.ab2]
// CHECK:STDOUT:   %.loc24_31.2: type = converted %.loc24_23, %as_type [concrete = constants.%tuple.type.ab2]
// CHECK:STDOUT:   %impl.elem0: %.cf4 = impl_witness_access constants.%A.impl_witness.098, element0 [concrete = constants.%tuple.type.as.A.impl.F.44b]
// CHECK:STDOUT:   %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z]
// CHECK:STDOUT:   %.loc24_38.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @tuple.type.as.A.impl.F(constants.%Y1, constants.%Y2, constants.%X, constants.%Z) [concrete = constants.%tuple.type.as.A.impl.F.specific_fn.6e3]
// CHECK:STDOUT:   %.loc24_39.1: ref %tuple.type.65a = temporary_storage
// CHECK:STDOUT:   %.loc24_38.2: ref %Z = temporary_storage
// CHECK:STDOUT:   %.loc24_38.3: init %Z = class_init (), %.loc24_38.2 [concrete = constants.%Z.val]
// CHECK:STDOUT:   %.loc24_38.4: ref %Z = temporary %.loc24_38.2, %.loc24_38.3
// CHECK:STDOUT:   %.loc24_38.5: ref %Z = converted %.loc24_38.1, %.loc24_38.4
// CHECK:STDOUT:   %.loc24_38.6: %Z = acquire_value %.loc24_38.5
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.call: init %tuple.type.65a = call %specific_fn(%.loc24_38.6) to %.loc24_39.1
// CHECK:STDOUT:   %.loc24_39.2: ref %tuple.type.65a = temporary %.loc24_39.1, %tuple.type.as.A.impl.F.call
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc24_39: <bound method> = bound_method %.loc24_39.2, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.772
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.772, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.917) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.0b9]
// CHECK:STDOUT:   %bound_method.loc24_39: <bound method> = bound_method %.loc24_39.2, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc24_39: init %empty_tuple.type = call %bound_method.loc24_39(%.loc24_39.2)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc24_38: <bound method> = bound_method %.loc24_38.4, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.72c
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.2: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.72c, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.5de) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.8a3]
// CHECK:STDOUT:   %bound_method.loc24_38: <bound method> = bound_method %.loc24_38.4, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc24_38: init %empty_tuple.type = call %bound_method.loc24_38(%.loc24_38.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallGeneric(%T.loc27_16.2: %A.type.f82) {
// CHECK:STDOUT:   %T.loc27_16.1: %A.type.f82 = symbolic_binding T, 0 [symbolic = %T.loc27_16.1 (constants.%T.19a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc27_16.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %A.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc27_16.1, @A, @A(constants.%X) [symbolic = %A.lookup_impl_witness (constants.%A.lookup_impl_witness)]
// CHECK:STDOUT:   %.loc28_4.3: type = fn_type_with_self_type constants.%A.F.type.49b, %T.loc27_16.1 [symbolic = %.loc28_4.3 (constants.%.14f)]
// CHECK:STDOUT:   %impl.elem0.loc28_4.2: @CallGeneric.%.loc28_4.3 (%.14f) = impl_witness_access %A.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc28_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc28_4.2: <specific function> = specific_impl_function %impl.elem0.loc28_4.2, @A.F(constants.%X, %T.loc27_16.1, constants.%Z) [symbolic = %specific_impl_fn.loc28_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (constants.%X, %T.binding.as_type, constants.%Z) [symbolic = %tuple.type (constants.%tuple.type.50b)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %tuple.type [symbolic = %require_complete (constants.%require_complete.b37)]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %tuple.type, () [symbolic = %facet_value (constants.%facet_value.f56)]
// CHECK:STDOUT:   %.loc28_12.3: require_specific_def_type = require_specific_def @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [symbolic = %.loc28_12.3 (constants.%.ac1)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness imports.%Destroy.impl_witness_table, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness.cec)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %tuple.type, (%Destroy.impl_witness) [symbolic = %Destroy.facet (constants.%Destroy.facet.ddd)]
// CHECK:STDOUT:   %.loc28_12.4: type = fn_type_with_self_type constants.%Destroy.Op.type, %Destroy.facet [symbolic = %.loc28_12.4 (constants.%.b7f)]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op.type (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.type.a72)]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op: @CallGeneric.%DestroyT.binding.as_type.as.Destroy.impl.Op.type (%DestroyT.binding.as_type.as.Destroy.impl.Op.type.a72) = struct_value () [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.84d)]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.loc28: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value) [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.loc28 (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.da9)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref: %A.type.f82 = name_ref T, %T.loc27_16.2 [symbolic = %T.loc27_16.1 (constants.%T.19a)]
// CHECK:STDOUT:     %.loc28_4.1: %A.assoc_type.f05 = specific_constant @A.%assoc0.loc6_39.1, @A(constants.%X) [concrete = constants.%assoc0.249]
// CHECK:STDOUT:     %F.ref: %A.assoc_type.f05 = name_ref F, %.loc28_4.1 [concrete = constants.%assoc0.249]
// CHECK:STDOUT:     %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc28_4.2: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %impl.elem0.loc28_4.1: @CallGeneric.%.loc28_4.3 (%.14f) = impl_witness_access constants.%A.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc28_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z]
// CHECK:STDOUT:     %.loc28_11.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %specific_impl_fn.loc28_4.1: <specific function> = specific_impl_function %impl.elem0.loc28_4.1, @A.F(constants.%X, constants.%T.19a, constants.%Z) [symbolic = %specific_impl_fn.loc28_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %.loc28_12.1: ref @CallGeneric.%tuple.type (%tuple.type.50b) = temporary_storage
// CHECK:STDOUT:     %.loc28_11.2: ref %Z = temporary_storage
// CHECK:STDOUT:     %.loc28_11.3: init %Z = class_init (), %.loc28_11.2 [concrete = constants.%Z.val]
// CHECK:STDOUT:     %.loc28_11.4: ref %Z = temporary %.loc28_11.2, %.loc28_11.3
// CHECK:STDOUT:     %.loc28_11.5: ref %Z = converted %.loc28_11.1, %.loc28_11.4
// CHECK:STDOUT:     %.loc28_11.6: %Z = acquire_value %.loc28_11.5
// CHECK:STDOUT:     %A.F.call: init @CallGeneric.%tuple.type (%tuple.type.50b) = call %specific_impl_fn.loc28_4.1(%.loc28_11.6) to %.loc28_12.1
// CHECK:STDOUT:     %.loc28_12.2: ref @CallGeneric.%tuple.type (%tuple.type.50b) = temporary %.loc28_12.1, %A.F.call
// CHECK:STDOUT:     %impl.elem0.loc28_12: @CallGeneric.%.loc28_12.4 (%.b7f) = impl_witness_access constants.%Destroy.impl_witness.cec, element0 [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.84d)]
// CHECK:STDOUT:     %bound_method.loc28_12.1: <bound method> = bound_method %.loc28_12.2, %impl.elem0.loc28_12
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0.loc28_12, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.f56) [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.loc28 (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.da9)]
// CHECK:STDOUT:     %bound_method.loc28_12.2: <bound method> = bound_method %.loc28_12.2, %specific_fn
// CHECK:STDOUT:     %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc28_12: init %empty_tuple.type = call %bound_method.loc28_12.2(%.loc28_12.2)
// CHECK:STDOUT:     %DestroyT.binding.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc28_11.4, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.72c
// CHECK:STDOUT:     %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.72c, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.5de) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.8a3]
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
// CHECK:STDOUT:   %.loc33_22: %tuple.type.24b = tuple_literal (%Y1.ref, %Y2.ref) [concrete = constants.%tuple.dd5]
// CHECK:STDOUT:   %.loc33_24: type = converted %.loc33_22, constants.%tuple.type.ab2 [concrete = constants.%tuple.type.ab2]
// CHECK:STDOUT:   %A.facet: %A.type.f82 = facet_value %.loc33_24, (constants.%A.impl_witness.098) [concrete = constants.%A.facet.e22]
// CHECK:STDOUT:   %.loc33_31: %A.type.f82 = converted %.loc33_24, %A.facet [concrete = constants.%A.facet.e22]
// CHECK:STDOUT:   %CallGeneric.specific_fn: <specific function> = specific_function %CallGeneric.ref, @CallGeneric(constants.%A.facet.e22) [concrete = constants.%CallGeneric.specific_fn]
// CHECK:STDOUT:   %CallGeneric.call: init %empty_tuple.type = call %CallGeneric.specific_fn()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%T.67d) {
// CHECK:STDOUT:   %T.loc5_13.1 => constants.%T.67d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.F(constants.%T.67d, constants.%Self.9b1, constants.%U.4e7) {
// CHECK:STDOUT:   %U.loc6_8.1 => constants.%U.4e7
// CHECK:STDOUT:   %pattern_type.loc6_18 => constants.%pattern_type.62e
// CHECK:STDOUT:   %T => constants.%T.67d
// CHECK:STDOUT:   %A.type => constants.%A.type.314
// CHECK:STDOUT:   %Self => constants.%Self.9b1
// CHECK:STDOUT:   %tuple.type.loc6_38.1 => constants.%tuple.type.f42
// CHECK:STDOUT:   %tuple => constants.%tuple.28e
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.78c
// CHECK:STDOUT:   %tuple.type.loc6_38.2 => constants.%tuple.type.6a9
// CHECK:STDOUT:   %pattern_type.loc6_24 => constants.%pattern_type.0dd
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%W) {
// CHECK:STDOUT:   %T.loc5_13.1 => constants.%W
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.type => constants.%A.type.62d
// CHECK:STDOUT:   %Self.loc5_23.2 => constants.%Self.6ee
// CHECK:STDOUT:   %A.F.type => constants.%A.F.type.0cd
// CHECK:STDOUT:   %A.F => constants.%A.F.cef
// CHECK:STDOUT:   %A.assoc_type => constants.%A.assoc_type.7a9
// CHECK:STDOUT:   %assoc0.loc6_39.2 => constants.%assoc0.264
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @tuple.type.as.A.impl(constants.%V1, constants.%V2, constants.%W) {
// CHECK:STDOUT:   %V1.loc14_14.1 => constants.%V1
// CHECK:STDOUT:   %V2.loc14_25.1 => constants.%V2
// CHECK:STDOUT:   %W.loc14_36.1 => constants.%W
// CHECK:STDOUT:   %tuple => constants.%tuple.4b9
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.a5e
// CHECK:STDOUT:   %A.type.loc14_61.1 => constants.%A.type.62d
// CHECK:STDOUT:   %A.impl_witness => constants.%A.impl_witness.d95
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.4be
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type => constants.%tuple.type.as.A.impl.F.type.f4a
// CHECK:STDOUT:   %tuple.type.as.A.impl.F => constants.%tuple.type.as.A.impl.F.3c3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @tuple.type.as.A.impl.F(constants.%V1, constants.%V2, constants.%W, constants.%U.ce4) {
// CHECK:STDOUT:   %U.loc17_8.1 => constants.%U.ce4
// CHECK:STDOUT:   %pattern_type.loc17_18 => constants.%pattern_type.e4a
// CHECK:STDOUT:   %W => constants.%W
// CHECK:STDOUT:   %V1 => constants.%V1
// CHECK:STDOUT:   %V2 => constants.%V2
// CHECK:STDOUT:   %tuple.loc17_38 => constants.%tuple.4b9
// CHECK:STDOUT:   %tuple.loc17_42 => constants.%tuple.117
// CHECK:STDOUT:   %tuple.type.loc17_42.1 => constants.%tuple.type.a5e
// CHECK:STDOUT:   %tuple.type.loc17_42.2 => constants.%tuple.type.897
// CHECK:STDOUT:   %pattern_type.loc17_24 => constants.%pattern_type.5cc
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc17_42 => constants.%require_complete.da0
// CHECK:STDOUT:   %require_complete.loc17_19 => constants.%require_complete.5a4
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type => constants.%tuple.type.as.A.impl.F.type.f4a
// CHECK:STDOUT:   %tuple.type.as.A.impl.F => constants.%tuple.type.as.A.impl.F.3c3
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.specific_fn.loc18_12.2 => constants.%tuple.type.as.A.impl.F.specific_fn.3d7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.F(constants.%W, constants.%A.facet.e1a, constants.%U.ce4) {
// CHECK:STDOUT:   %U.loc6_8.1 => constants.%U.ce4
// CHECK:STDOUT:   %pattern_type.loc6_18 => constants.%pattern_type.e4a
// CHECK:STDOUT:   %T => constants.%W
// CHECK:STDOUT:   %A.type => constants.%A.type.62d
// CHECK:STDOUT:   %Self => constants.%A.facet.e1a
// CHECK:STDOUT:   %tuple.type.loc6_38.1 => constants.%tuple.type.26b
// CHECK:STDOUT:   %tuple => constants.%tuple.31e
// CHECK:STDOUT:   %Self.binding.as_type => constants.%tuple.type.a5e
// CHECK:STDOUT:   %tuple.type.loc6_38.2 => constants.%tuple.type.897
// CHECK:STDOUT:   %pattern_type.loc6_24 => constants.%pattern_type.5cc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%X) {
// CHECK:STDOUT:   %T.loc5_13.1 => constants.%X
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.type => constants.%A.type.f82
// CHECK:STDOUT:   %Self.loc5_23.2 => constants.%Self.aab
// CHECK:STDOUT:   %A.F.type => constants.%A.F.type.49b
// CHECK:STDOUT:   %A.F => constants.%A.F.768
// CHECK:STDOUT:   %A.assoc_type => constants.%A.assoc_type.f05
// CHECK:STDOUT:   %assoc0.loc6_39.2 => constants.%assoc0.249
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @tuple.type.as.A.impl(constants.%Y1, constants.%Y2, constants.%X) {
// CHECK:STDOUT:   %V1.loc14_14.1 => constants.%Y1
// CHECK:STDOUT:   %V2.loc14_25.1 => constants.%Y2
// CHECK:STDOUT:   %W.loc14_36.1 => constants.%X
// CHECK:STDOUT:   %tuple => constants.%tuple.dd5
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.ab2
// CHECK:STDOUT:   %A.type.loc14_61.1 => constants.%A.type.f82
// CHECK:STDOUT:   %A.impl_witness => constants.%A.impl_witness.098
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.df7
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type => constants.%tuple.type.as.A.impl.F.type.9c3
// CHECK:STDOUT:   %tuple.type.as.A.impl.F => constants.%tuple.type.as.A.impl.F.44b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @tuple.type.as.A.impl.F(constants.%Y1, constants.%Y2, constants.%X, constants.%Z) {
// CHECK:STDOUT:   %U.loc17_8.1 => constants.%Z
// CHECK:STDOUT:   %pattern_type.loc17_18 => constants.%pattern_type.bdb
// CHECK:STDOUT:   %W => constants.%X
// CHECK:STDOUT:   %V1 => constants.%Y1
// CHECK:STDOUT:   %V2 => constants.%Y2
// CHECK:STDOUT:   %tuple.loc17_38 => constants.%tuple.dd5
// CHECK:STDOUT:   %tuple.loc17_42 => constants.%tuple.66d
// CHECK:STDOUT:   %tuple.type.loc17_42.1 => constants.%tuple.type.ab2
// CHECK:STDOUT:   %tuple.type.loc17_42.2 => constants.%tuple.type.65a
// CHECK:STDOUT:   %pattern_type.loc17_24 => constants.%pattern_type.30b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc17_42 => constants.%complete_type.cf2
// CHECK:STDOUT:   %require_complete.loc17_19 => constants.%complete_type.357
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type => constants.%tuple.type.as.A.impl.F.type.9c3
// CHECK:STDOUT:   %tuple.type.as.A.impl.F => constants.%tuple.type.as.A.impl.F.44b
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.specific_fn.loc18_12.2 => constants.%tuple.type.as.A.impl.F.specific_fn.6e3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGeneric(constants.%T.19a) {
// CHECK:STDOUT:   %T.loc27_16.1 => constants.%T.19a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.F(constants.%X, constants.%T.19a, constants.%Z) {
// CHECK:STDOUT:   %U.loc6_8.1 => constants.%Z
// CHECK:STDOUT:   %pattern_type.loc6_18 => constants.%pattern_type.bdb
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.f82
// CHECK:STDOUT:   %Self => constants.%T.19a
// CHECK:STDOUT:   %tuple.type.loc6_38.1 => constants.%tuple.type.298
// CHECK:STDOUT:   %tuple => constants.%tuple.c8d
// CHECK:STDOUT:   %Self.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %tuple.type.loc6_38.2 => constants.%tuple.type.50b
// CHECK:STDOUT:   %pattern_type.loc6_24 => constants.%pattern_type.56e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGeneric(constants.%A.facet.e22) {
// CHECK:STDOUT:   %T.loc27_16.1 => constants.%A.facet.e22
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.binding.as_type => constants.%tuple.type.ab2
// CHECK:STDOUT:   %A.lookup_impl_witness => constants.%A.impl_witness.098
// CHECK:STDOUT:   %.loc28_4.3 => constants.%.cf4
// CHECK:STDOUT:   %impl.elem0.loc28_4.2 => constants.%tuple.type.as.A.impl.F.44b
// CHECK:STDOUT:   %specific_impl_fn.loc28_4.2 => constants.%tuple.type.as.A.impl.F.specific_fn.6e3
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.65a
// CHECK:STDOUT:   %require_complete => constants.%complete_type.cf2
// CHECK:STDOUT:   %facet_value => constants.%facet_value.917
// CHECK:STDOUT:   %.loc28_12.3 => constants.%.110
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.776
// CHECK:STDOUT:   %Destroy.facet => constants.%Destroy.facet.15a
// CHECK:STDOUT:   %.loc28_12.4 => constants.%.ee4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type => constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.type.b70
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op => constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.772
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.loc28 => constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.0b9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.F(constants.%X, constants.%A.facet.e22, constants.%Z) {
// CHECK:STDOUT:   %U.loc6_8.1 => constants.%Z
// CHECK:STDOUT:   %pattern_type.loc6_18 => constants.%pattern_type.bdb
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.f82
// CHECK:STDOUT:   %Self => constants.%A.facet.e22
// CHECK:STDOUT:   %tuple.type.loc6_38.1 => constants.%tuple.type.298
// CHECK:STDOUT:   %tuple => constants.%tuple.f1f
// CHECK:STDOUT:   %Self.binding.as_type => constants.%tuple.type.ab2
// CHECK:STDOUT:   %tuple.type.loc6_38.2 => constants.%tuple.type.65a
// CHECK:STDOUT:   %pattern_type.loc6_24 => constants.%pattern_type.30b
// CHECK:STDOUT: }
// CHECK:STDOUT:

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
// CHECK:STDOUT:   %.Self: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %A.type.495: type = generic_interface_type @A [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %A.generic: %A.type.495 = struct_value () [concrete]
// CHECK:STDOUT:   %A.type.806: type = facet_type <@A, @A(%T.8b3)> [symbolic]
// CHECK:STDOUT:   %Self.b74: %A.type.806 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %U.2cb: type = bind_symbolic_name U, 2 [symbolic]
// CHECK:STDOUT:   %ptr.204: type = ptr_type %U.2cb [symbolic]
// CHECK:STDOUT:   %pattern_type.eee: type = pattern_type %ptr.204 [symbolic]
// CHECK:STDOUT:   %tuple.type.4c5: type = tuple_type (type, %A.type.806, type) [symbolic]
// CHECK:STDOUT:   %Self.as_type.327: type = facet_access_type %Self.b74 [symbolic]
// CHECK:STDOUT:   %tuple.type.5a8: type = tuple_type (%T.8b3, %Self.as_type.327, %ptr.204) [symbolic]
// CHECK:STDOUT:   %pattern_type.0eb: type = pattern_type %tuple.type.5a8 [symbolic]
// CHECK:STDOUT:   %A.F.type.17a: type = fn_type @A.F, @A(%T.8b3) [symbolic]
// CHECK:STDOUT:   %A.F.0d8: %A.F.type.17a = struct_value () [symbolic]
// CHECK:STDOUT:   %A.assoc_type.ed3: type = assoc_entity_type @A, @A(%T.8b3) [symbolic]
// CHECK:STDOUT:   %assoc0.fcb: %A.assoc_type.ed3 = assoc_entity element0, @A.%A.F.decl [symbolic]
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.286: <witness> = impl_witness @X.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.d17: type = ptr_type %X [concrete]
// CHECK:STDOUT:   %pattern_type.1c6: type = pattern_type %ptr.d17 [concrete]
// CHECK:STDOUT:   %X.as.Destroy.impl.Op.type: type = fn_type @X.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %X.as.Destroy.impl.Op: %X.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Y: type = class_type @Y [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.ad0: <witness> = impl_witness @Y.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.455: type = ptr_type %Y [concrete]
// CHECK:STDOUT:   %pattern_type.0ae: type = pattern_type %ptr.455 [concrete]
// CHECK:STDOUT:   %Y.as.Destroy.impl.Op.type: type = fn_type @Y.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %Y.as.Destroy.impl.Op: %Y.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Z: type = class_type @Z [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.417: <witness> = impl_witness @Z.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.fb6: type = ptr_type %Z [concrete]
// CHECK:STDOUT:   %pattern_type.f76: type = pattern_type %ptr.fb6 [concrete]
// CHECK:STDOUT:   %Z.as.Destroy.impl.Op.type: type = fn_type @Z.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %Z.as.Destroy.impl.Op: %Z.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %A.type.515: type = facet_type <@A, @A(%X)> [concrete]
// CHECK:STDOUT:   %Self.522: %A.type.515 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %A.F.type.13d: type = fn_type @A.F, @A(%X) [concrete]
// CHECK:STDOUT:   %A.F.d83: %A.F.type.13d = struct_value () [concrete]
// CHECK:STDOUT:   %A.assoc_type.296: type = assoc_entity_type @A, @A(%X) [concrete]
// CHECK:STDOUT:   %assoc0.5f6: %A.assoc_type.296 = assoc_entity element0, @A.%A.F.decl [concrete]
// CHECK:STDOUT:   %A.impl_witness: <witness> = impl_witness file.%A.impl_witness_table [concrete]
// CHECK:STDOUT:   %U.8b3: type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %ptr.79f131.1: type = ptr_type %U.8b3 [symbolic]
// CHECK:STDOUT:   %pattern_type.afe393.1: type = pattern_type %ptr.79f131.1 [symbolic]
// CHECK:STDOUT:   %tuple.type.ff9: type = tuple_type (type, type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.c14: type = tuple_type (%X, %Y, %ptr.79f131.1) [symbolic]
// CHECK:STDOUT:   %pattern_type.ac9: type = pattern_type %tuple.type.c14 [symbolic]
// CHECK:STDOUT:   %Y.as.A.impl.F.type: type = fn_type @Y.as.A.impl.F [concrete]
// CHECK:STDOUT:   %Y.as.A.impl.F: %Y.as.A.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %A.facet.f2b: %A.type.515 = facet_value %Y, (%A.impl_witness) [concrete]
// CHECK:STDOUT:   %tuple.type.281: type = tuple_type (type, %A.type.515, type) [concrete]
// CHECK:STDOUT:   %require_complete.86f: <witness> = require_complete_type %tuple.type.c14 [symbolic]
// CHECK:STDOUT:   %require_complete.6e5e64.1: <witness> = require_complete_type %ptr.79f131.1 [symbolic]
// CHECK:STDOUT:   %tuple.type.e44: type = tuple_type (%empty_struct_type, %empty_struct_type, %ptr.79f131.1) [symbolic]
// CHECK:STDOUT:   %X.val: %X = struct_value () [concrete]
// CHECK:STDOUT:   %Y.val: %Y = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.f23dbd.1: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.8b3) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.abf12b.1: %ptr.as.Copy.impl.Op.type.f23dbd.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.cb2: <witness> = lookup_impl_witness %ptr.79f131.1, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.facet.def: %Copy.type = facet_value %ptr.79f131.1, (%Copy.lookup_impl_witness.cb2) [symbolic]
// CHECK:STDOUT:   %.fe5: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.def [symbolic]
// CHECK:STDOUT:   %impl.elem0.429: %.fe5 = impl_witness_access %Copy.lookup_impl_witness.cb2, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.d8a: <specific function> = specific_impl_function %impl.elem0.429, @Copy.Op(%Copy.facet.def) [symbolic]
// CHECK:STDOUT:   %Call.type: type = fn_type @Call [concrete]
// CHECK:STDOUT:   %Call: %Call.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.8f9: type = pattern_type %Z [concrete]
// CHECK:STDOUT:   %.7c0: type = fn_type_with_self_type %A.F.type.13d, %A.facet.f2b [concrete]
// CHECK:STDOUT:   %tuple.type.092: type = tuple_type (%X, %Y, %ptr.fb6) [concrete]
// CHECK:STDOUT:   %pattern_type.48d: type = pattern_type %tuple.type.092 [concrete]
// CHECK:STDOUT:   %Y.as.A.impl.F.specific_fn: <specific function> = specific_function %Y.as.A.impl.F, @Y.as.A.impl.F(%Z) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.305: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%T.8b3) [symbolic]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.e3a: %T.as.Destroy.impl.Op.type.305 = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.d7d: <witness> = impl_witness imports.%Destroy.impl_witness_table, @T.as.Destroy.impl(%tuple.type.092) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.dde: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%tuple.type.092) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.64e: %T.as.Destroy.impl.Op.type.dde = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.5df: type = ptr_type %tuple.type.092 [concrete]
// CHECK:STDOUT:   %complete_type.aab: <witness> = complete_type_witness %ptr.5df [concrete]
// CHECK:STDOUT:   %Destroy.facet.ed1: %Destroy.type = facet_value %tuple.type.092, (%Destroy.impl_witness.d7d) [concrete]
// CHECK:STDOUT:   %.928: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet.ed1 [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %T.as.Destroy.impl.Op.64e, @T.as.Destroy.impl.Op(%tuple.type.092) [concrete]
// CHECK:STDOUT:   %T.3b6: %A.type.515 = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.05f: type = pattern_type %A.type.515 [concrete]
// CHECK:STDOUT:   %CallGeneric.type: type = fn_type @CallGeneric [concrete]
// CHECK:STDOUT:   %CallGeneric: %CallGeneric.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.fc2: type = facet_access_type %T.3b6 [symbolic]
// CHECK:STDOUT:   %A.lookup_impl_witness: <witness> = lookup_impl_witness %T.3b6, @A, @A(%X) [symbolic]
// CHECK:STDOUT:   %A.facet.97b: %A.type.515 = facet_value %T.as_type.fc2, (%A.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %.7c8: type = fn_type_with_self_type %A.F.type.13d, %A.facet.97b [symbolic]
// CHECK:STDOUT:   %impl.elem0.eda: %.7c8 = impl_witness_access %A.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %tuple.type.fd8: type = tuple_type (%X, %T.as_type.fc2, %ptr.fb6) [symbolic]
// CHECK:STDOUT:   %pattern_type.1d1: type = pattern_type %tuple.type.fd8 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.9ce: <specific function> = specific_impl_function %impl.elem0.eda, @A.F(%X, %A.facet.97b, %Z) [symbolic]
// CHECK:STDOUT:   %require_complete.f71: <witness> = require_complete_type %tuple.type.fd8 [symbolic]
// CHECK:STDOUT:   %ptr.1b6: type = ptr_type %tuple.type.fd8 [symbolic]
// CHECK:STDOUT:   %require_complete.cda: <witness> = require_complete_type %ptr.1b6 [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %tuple.type.fd8, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.709: %Destroy.type = facet_value %tuple.type.fd8, (%Destroy.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %.768: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet.709 [symbolic]
// CHECK:STDOUT:   %impl.elem0.c2c: %.768 = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.5e2: <specific function> = specific_impl_function %impl.elem0.c2c, @Destroy.Op(%Destroy.facet.709) [symbolic]
// CHECK:STDOUT:   %CallIndirect.type: type = fn_type @CallIndirect [concrete]
// CHECK:STDOUT:   %CallIndirect: %CallIndirect.type = struct_value () [concrete]
// CHECK:STDOUT:   %CallGeneric.specific_fn: <specific function> = specific_function %CallGeneric, @CallGeneric(%A.facet.f2b) [concrete]
// CHECK:STDOUT:   %complete_type.05d: <witness> = complete_type_witness %tuple.type.092 [concrete]
// CHECK:STDOUT:   %complete_type.d3e: <witness> = complete_type_witness %ptr.fb6 [concrete]
// CHECK:STDOUT:   %tuple.type.953: type = tuple_type (%empty_struct_type, %empty_struct_type, %ptr.fb6) [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.f8c: <witness> = impl_witness imports.%Copy.impl_witness_table.a71, @ptr.as.Copy.impl(%Z) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.0d9: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%Z) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.ac3: %ptr.as.Copy.impl.Op.type.0d9 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.27d: %Copy.type = facet_value %ptr.fb6, (%Copy.impl_witness.f8c) [concrete]
// CHECK:STDOUT:   %.67f: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.27d [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.ac3, @ptr.as.Copy.impl.Op(%Z) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.de9: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.f23dbd.1) = import_ref Core//prelude/parts/copy, loc36_31, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.abf12b.1)]
// CHECK:STDOUT:   %Copy.impl_witness_table.a71 = impl_witness_table (%Core.import_ref.de9), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.efe: @T.as.Destroy.impl.%T.as.Destroy.impl.Op.type (%T.as.Destroy.impl.Op.type.305) = import_ref Core//prelude/parts/destroy, loc8_29, loaded [symbolic = @T.as.Destroy.impl.%T.as.Destroy.impl.Op (constants.%T.as.Destroy.impl.Op.e3a)]
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (%Core.import_ref.efe), @T.as.Destroy.impl [concrete]
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
// CHECK:STDOUT:     %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc5_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc5_13.1 (constants.%T.8b3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.decl: type = class_decl @X [concrete = constants.%X] {} {}
// CHECK:STDOUT:   %Y.decl: type = class_decl @Y [concrete = constants.%Y] {} {}
// CHECK:STDOUT:   %Z.decl: type = class_decl @Z [concrete = constants.%Z] {} {}
// CHECK:STDOUT:   impl_decl @Y.as.A.impl [concrete] {} {
// CHECK:STDOUT:     %Y.ref: type = name_ref Y, file.%Y.decl [concrete = constants.%Y]
// CHECK:STDOUT:     %A.ref: %A.type.495 = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %A.type: type = facet_type <@A, @A(constants.%X)> [concrete = constants.%A.type.515]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.impl_witness_table = impl_witness_table (@Y.as.A.impl.%Y.as.A.impl.F.decl), @Y.as.A.impl [concrete]
// CHECK:STDOUT:   %A.impl_witness: <witness> = impl_witness %A.impl_witness_table [concrete = constants.%A.impl_witness]
// CHECK:STDOUT:   %Call.decl: %Call.type = fn_decl @Call [concrete = constants.%Call] {} {}
// CHECK:STDOUT:   %CallGeneric.decl: %CallGeneric.type = fn_decl @CallGeneric [concrete = constants.%CallGeneric] {
// CHECK:STDOUT:     %T.patt: %pattern_type.05f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc26: type = splice_block %A.type [concrete = constants.%A.type.515] {
// CHECK:STDOUT:       %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %A.ref: %A.type.495 = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:       %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:       %A.type: type = facet_type <@A, @A(constants.%X)> [concrete = constants.%A.type.515]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc26_16.2: %A.type.515 = bind_symbolic_name T, 0 [symbolic = %T.loc26_16.1 (constants.%T.3b6)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallIndirect.decl: %CallIndirect.type = fn_decl @CallIndirect [concrete = constants.%CallIndirect] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @A(%T.loc5_13.2: type) {
// CHECK:STDOUT:   %T.loc5_13.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc5_13.1 (constants.%T.8b3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.type: type = facet_type <@A, @A(%T.loc5_13.1)> [symbolic = %A.type (constants.%A.type.806)]
// CHECK:STDOUT:   %Self.2: @A.%A.type (%A.type.806) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.b74)]
// CHECK:STDOUT:   %A.F.type: type = fn_type @A.F, @A(%T.loc5_13.1) [symbolic = %A.F.type (constants.%A.F.type.17a)]
// CHECK:STDOUT:   %A.F: @A.%A.F.type (%A.F.type.17a) = struct_value () [symbolic = %A.F (constants.%A.F.0d8)]
// CHECK:STDOUT:   %A.assoc_type: type = assoc_entity_type @A, @A(%T.loc5_13.1) [symbolic = %A.assoc_type (constants.%A.assoc_type.ed3)]
// CHECK:STDOUT:   %assoc0.loc6_41.2: @A.%A.assoc_type (%A.assoc_type.ed3) = assoc_entity element0, %A.F.decl [symbolic = %assoc0.loc6_41.2 (constants.%assoc0.fcb)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @A.%A.type (%A.type.806) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.b74)]
// CHECK:STDOUT:     %A.F.decl: @A.%A.F.type (%A.F.type.17a) = fn_decl @A.F [symbolic = @A.%A.F (constants.%A.F.0d8)] {
// CHECK:STDOUT:       %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 2 [concrete]
// CHECK:STDOUT:       %u.patt: @A.F.%pattern_type.loc6_18 (%pattern_type.eee) = binding_pattern u [concrete]
// CHECK:STDOUT:       %u.param_patt: @A.F.%pattern_type.loc6_18 (%pattern_type.eee) = value_param_pattern %u.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @A.F.%pattern_type.loc6_25 (%pattern_type.0eb) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @A.F.%pattern_type.loc6_25 (%pattern_type.0eb) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @A.%T.loc5_13.2 [symbolic = %T (constants.%T.8b3)]
// CHECK:STDOUT:       %.loc6_32: @A.F.%A.type (%A.type.806) = specific_constant @A.%Self.1, @A(constants.%T.8b3) [symbolic = %Self (constants.%Self.b74)]
// CHECK:STDOUT:       %Self.ref: @A.F.%A.type (%A.type.806) = name_ref Self, %.loc6_32 [symbolic = %Self (constants.%Self.b74)]
// CHECK:STDOUT:       %U.ref.loc6_38: type = name_ref U, %U.loc6_8.2 [symbolic = %U.loc6_8.1 (constants.%U.2cb)]
// CHECK:STDOUT:       %ptr.loc6_39: type = ptr_type %U.ref.loc6_38 [symbolic = %ptr.loc6_22.1 (constants.%ptr.204)]
// CHECK:STDOUT:       %.loc6_40.1: @A.F.%tuple.type.loc6_40.1 (%tuple.type.4c5) = tuple_literal (%T.ref, %Self.ref, %ptr.loc6_39)
// CHECK:STDOUT:       %Self.as_type.loc6_40.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc6_40.1 (constants.%Self.as_type.327)]
// CHECK:STDOUT:       %.loc6_40.2: type = converted %Self.ref, %Self.as_type.loc6_40.2 [symbolic = %Self.as_type.loc6_40.1 (constants.%Self.as_type.327)]
// CHECK:STDOUT:       %.loc6_40.3: type = converted %.loc6_40.1, constants.%tuple.type.5a8 [symbolic = %tuple.type.loc6_40.2 (constants.%tuple.type.5a8)]
// CHECK:STDOUT:       %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %U.loc6_8.2: type = bind_symbolic_name U, 2 [symbolic = %U.loc6_8.1 (constants.%U.2cb)]
// CHECK:STDOUT:       %u.param: @A.F.%ptr.loc6_22.1 (%ptr.204) = value_param call_param0
// CHECK:STDOUT:       %.loc6_22: type = splice_block %ptr.loc6_22.2 [symbolic = %ptr.loc6_22.1 (constants.%ptr.204)] {
// CHECK:STDOUT:         %U.ref.loc6_21: type = name_ref U, %U.loc6_8.2 [symbolic = %U.loc6_8.1 (constants.%U.2cb)]
// CHECK:STDOUT:         %ptr.loc6_22.2: type = ptr_type %U.ref.loc6_21 [symbolic = %ptr.loc6_22.1 (constants.%ptr.204)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %u: @A.F.%ptr.loc6_22.1 (%ptr.204) = bind_name u, %u.param
// CHECK:STDOUT:       %return.param: ref @A.F.%tuple.type.loc6_40.2 (%tuple.type.5a8) = out_param call_param1
// CHECK:STDOUT:       %return: ref @A.F.%tuple.type.loc6_40.2 (%tuple.type.5a8) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc6_41.1: @A.%A.assoc_type (%A.assoc_type.ed3) = assoc_entity element0, %A.F.decl [symbolic = %assoc0.loc6_41.2 (constants.%assoc0.fcb)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .F = %assoc0.loc6_41.1
// CHECK:STDOUT:     witness = (%A.F.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @X.as.Destroy.impl: @X.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:   %X.as.Destroy.impl.Op.decl: %X.as.Destroy.impl.Op.type = fn_decl @X.as.Destroy.impl.Op [concrete = constants.%X.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.1c6 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.1c6 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc9: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.d17 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%X [concrete = constants.%X]
// CHECK:STDOUT:     %self: %ptr.d17 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %X.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @X.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Y.as.Destroy.impl: @Y.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:   %Y.as.Destroy.impl.Op.decl: %Y.as.Destroy.impl.Op.type = fn_decl @Y.as.Destroy.impl.Op [concrete = constants.%Y.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.0ae = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.0ae = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc10: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.455 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Y [concrete = constants.%Y]
// CHECK:STDOUT:     %self: %ptr.455 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %Y.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @Y.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Z.as.Destroy.impl: @Z.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:   %Z.as.Destroy.impl.Op.decl: %Z.as.Destroy.impl.Op.type = fn_decl @Z.as.Destroy.impl.Op [concrete = constants.%Z.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.f76 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.f76 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc11: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.fb6 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Z [concrete = constants.%Z]
// CHECK:STDOUT:     %self: %ptr.fb6 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %Z.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @Z.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Y.as.A.impl: %Y.ref as %A.type {
// CHECK:STDOUT:   %Y.as.A.impl.F.decl: %Y.as.A.impl.F.type = fn_decl @Y.as.A.impl.F [concrete = constants.%Y.as.A.impl.F] {
// CHECK:STDOUT:     %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 0 [concrete]
// CHECK:STDOUT:     %u.patt: @Y.as.A.impl.F.%pattern_type.loc16_18 (%pattern_type.afe393.1) = binding_pattern u [concrete]
// CHECK:STDOUT:     %u.param_patt: @Y.as.A.impl.F.%pattern_type.loc16_18 (%pattern_type.afe393.1) = value_param_pattern %u.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @Y.as.A.impl.F.%pattern_type.loc16_25 (%pattern_type.ac9) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @Y.as.A.impl.F.%pattern_type.loc16_25 (%pattern_type.ac9) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %Y.ref: type = name_ref Y, file.%Y.decl [concrete = constants.%Y]
// CHECK:STDOUT:     %U.ref.loc16_35: type = name_ref U, %U.loc16_8.2 [symbolic = %U.loc16_8.1 (constants.%U.8b3)]
// CHECK:STDOUT:     %ptr.loc16_36: type = ptr_type %U.ref.loc16_35 [symbolic = %ptr.loc16_22.1 (constants.%ptr.79f131.1)]
// CHECK:STDOUT:     %.loc16_37.1: %tuple.type.ff9 = tuple_literal (%X.ref, %Y.ref, %ptr.loc16_36)
// CHECK:STDOUT:     %.loc16_37.2: type = converted %.loc16_37.1, constants.%tuple.type.c14 [symbolic = %tuple.type.loc16 (constants.%tuple.type.c14)]
// CHECK:STDOUT:     %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %U.loc16_8.2: type = bind_symbolic_name U, 0 [symbolic = %U.loc16_8.1 (constants.%U.8b3)]
// CHECK:STDOUT:     %u.param: @Y.as.A.impl.F.%ptr.loc16_22.1 (%ptr.79f131.1) = value_param call_param0
// CHECK:STDOUT:     %.loc16_22: type = splice_block %ptr.loc16_22.2 [symbolic = %ptr.loc16_22.1 (constants.%ptr.79f131.1)] {
// CHECK:STDOUT:       %U.ref.loc16_21: type = name_ref U, %U.loc16_8.2 [symbolic = %U.loc16_8.1 (constants.%U.8b3)]
// CHECK:STDOUT:       %ptr.loc16_22.2: type = ptr_type %U.ref.loc16_21 [symbolic = %ptr.loc16_22.1 (constants.%ptr.79f131.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: @Y.as.A.impl.F.%ptr.loc16_22.1 (%ptr.79f131.1) = bind_name u, %u.param
// CHECK:STDOUT:     %return.param: ref @Y.as.A.impl.F.%tuple.type.loc16 (%tuple.type.c14) = out_param call_param1
// CHECK:STDOUT:     %return: ref @Y.as.A.impl.F.%tuple.type.loc16 (%tuple.type.c14) = return_slot %return.param
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
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%X [concrete = constants.%X]
// CHECK:STDOUT:   impl_decl @X.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@X.as.Destroy.impl.%X.as.Destroy.impl.Op.decl), @X.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.286]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Y {
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%Y [concrete = constants.%Y]
// CHECK:STDOUT:   impl_decl @Y.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@Y.as.Destroy.impl.%Y.as.Destroy.impl.Op.decl), @Y.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.ad0]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Y
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Z {
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%Z [concrete = constants.%Z]
// CHECK:STDOUT:   impl_decl @Z.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@Z.as.Destroy.impl.%Z.as.Destroy.impl.Op.decl), @Z.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.417]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Z
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @A.F(@A.%T.loc5_13.2: type, @A.%Self.1: @A.%A.type (%A.type.806), %U.loc6_8.2: type) {
// CHECK:STDOUT:   %U.loc6_8.1: type = bind_symbolic_name U, 2 [symbolic = %U.loc6_8.1 (constants.%U.2cb)]
// CHECK:STDOUT:   %ptr.loc6_22.1: type = ptr_type %U.loc6_8.1 [symbolic = %ptr.loc6_22.1 (constants.%ptr.204)]
// CHECK:STDOUT:   %pattern_type.loc6_18: type = pattern_type %ptr.loc6_22.1 [symbolic = %pattern_type.loc6_18 (constants.%pattern_type.eee)]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T.8b3)]
// CHECK:STDOUT:   %A.type: type = facet_type <@A, @A(%T)> [symbolic = %A.type (constants.%A.type.806)]
// CHECK:STDOUT:   %Self: @A.F.%A.type (%A.type.806) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.b74)]
// CHECK:STDOUT:   %tuple.type.loc6_40.1: type = tuple_type (type, %A.type, type) [symbolic = %tuple.type.loc6_40.1 (constants.%tuple.type.4c5)]
// CHECK:STDOUT:   %Self.as_type.loc6_40.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc6_40.1 (constants.%Self.as_type.327)]
// CHECK:STDOUT:   %tuple.type.loc6_40.2: type = tuple_type (%T, %Self.as_type.loc6_40.1, %ptr.loc6_22.1) [symbolic = %tuple.type.loc6_40.2 (constants.%tuple.type.5a8)]
// CHECK:STDOUT:   %pattern_type.loc6_25: type = pattern_type %tuple.type.loc6_40.2 [symbolic = %pattern_type.loc6_25 (constants.%pattern_type.0eb)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param: @A.F.%ptr.loc6_22.1 (%ptr.204)) -> @A.F.%tuple.type.loc6_40.2 (%tuple.type.5a8);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @X.as.Destroy.impl.Op(%self.param: %ptr.d17) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Y.as.Destroy.impl.Op(%self.param: %ptr.455) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Z.as.Destroy.impl.Op(%self.param: %ptr.fb6) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Y.as.A.impl.F(%U.loc16_8.2: type) {
// CHECK:STDOUT:   %U.loc16_8.1: type = bind_symbolic_name U, 0 [symbolic = %U.loc16_8.1 (constants.%U.8b3)]
// CHECK:STDOUT:   %ptr.loc16_22.1: type = ptr_type %U.loc16_8.1 [symbolic = %ptr.loc16_22.1 (constants.%ptr.79f131.1)]
// CHECK:STDOUT:   %pattern_type.loc16_18: type = pattern_type %ptr.loc16_22.1 [symbolic = %pattern_type.loc16_18 (constants.%pattern_type.afe393.1)]
// CHECK:STDOUT:   %tuple.type.loc16: type = tuple_type (constants.%X, constants.%Y, %ptr.loc16_22.1) [symbolic = %tuple.type.loc16 (constants.%tuple.type.c14)]
// CHECK:STDOUT:   %pattern_type.loc16_25: type = pattern_type %tuple.type.loc16 [symbolic = %pattern_type.loc16_25 (constants.%pattern_type.ac9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc16_25: <witness> = require_complete_type %tuple.type.loc16 [symbolic = %require_complete.loc16_25 (constants.%require_complete.86f)]
// CHECK:STDOUT:   %require_complete.loc16_19: <witness> = require_complete_type %ptr.loc16_22.1 [symbolic = %require_complete.loc16_19 (constants.%require_complete.6e5e64.1)]
// CHECK:STDOUT:   %tuple.type.loc17: type = tuple_type (constants.%empty_struct_type, constants.%empty_struct_type, %ptr.loc16_22.1) [symbolic = %tuple.type.loc17 (constants.%tuple.type.e44)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr.loc16_22.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.cb2)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.loc16_22.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet.def)]
// CHECK:STDOUT:   %.loc17_21.2: type = fn_type_with_self_type constants.%Copy.Op.type, %Copy.facet [symbolic = %.loc17_21.2 (constants.%.fe5)]
// CHECK:STDOUT:   %impl.elem0.loc17_21.2: @Y.as.A.impl.F.%.loc17_21.2 (%.fe5) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc17_21.2 (constants.%impl.elem0.429)]
// CHECK:STDOUT:   %specific_impl_fn.loc17_21.2: <specific function> = specific_impl_function %impl.elem0.loc17_21.2, @Copy.Op(%Copy.facet) [symbolic = %specific_impl_fn.loc17_21.2 (constants.%specific_impl_fn.d8a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param: @Y.as.A.impl.F.%ptr.loc16_22.1 (%ptr.79f131.1)) -> %return.param: @Y.as.A.impl.F.%tuple.type.loc16 (%tuple.type.c14) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc17_14.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc17_18.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %u.ref: @Y.as.A.impl.F.%ptr.loc16_22.1 (%ptr.79f131.1) = name_ref u, %u
// CHECK:STDOUT:     %.loc17_22.1: @Y.as.A.impl.F.%tuple.type.loc17 (%tuple.type.e44) = tuple_literal (%.loc17_14.1, %.loc17_18.1, %u.ref)
// CHECK:STDOUT:     %tuple.elem0: ref %X = tuple_access %return, element0
// CHECK:STDOUT:     %.loc17_14.2: init %X = class_init (), %tuple.elem0 [concrete = constants.%X.val]
// CHECK:STDOUT:     %.loc17_22.2: init %X = converted %.loc17_14.1, %.loc17_14.2 [concrete = constants.%X.val]
// CHECK:STDOUT:     %tuple.elem1: ref %Y = tuple_access %return, element1
// CHECK:STDOUT:     %.loc17_18.2: init %Y = class_init (), %tuple.elem1 [concrete = constants.%Y.val]
// CHECK:STDOUT:     %.loc17_22.3: init %Y = converted %.loc17_18.1, %.loc17_18.2 [concrete = constants.%Y.val]
// CHECK:STDOUT:     %impl.elem0.loc17_21.1: @Y.as.A.impl.F.%.loc17_21.2 (%.fe5) = impl_witness_access constants.%Copy.lookup_impl_witness.cb2, element0 [symbolic = %impl.elem0.loc17_21.2 (constants.%impl.elem0.429)]
// CHECK:STDOUT:     %bound_method.loc17_21.1: <bound method> = bound_method %u.ref, %impl.elem0.loc17_21.1
// CHECK:STDOUT:     %specific_impl_fn.loc17_21.1: <specific function> = specific_impl_function %impl.elem0.loc17_21.1, @Copy.Op(constants.%Copy.facet.def) [symbolic = %specific_impl_fn.loc17_21.2 (constants.%specific_impl_fn.d8a)]
// CHECK:STDOUT:     %bound_method.loc17_21.2: <bound method> = bound_method %u.ref, %specific_impl_fn.loc17_21.1
// CHECK:STDOUT:     %.loc17_21.1: init @Y.as.A.impl.F.%ptr.loc16_22.1 (%ptr.79f131.1) = call %bound_method.loc17_21.2(%u.ref)
// CHECK:STDOUT:     %tuple.elem2: ref @Y.as.A.impl.F.%ptr.loc16_22.1 (%ptr.79f131.1) = tuple_access %return, element2
// CHECK:STDOUT:     %.loc17_22.4: init @Y.as.A.impl.F.%ptr.loc16_22.1 (%ptr.79f131.1) = initialize_from %.loc17_21.1 to %tuple.elem2
// CHECK:STDOUT:     %.loc17_22.5: init @Y.as.A.impl.F.%tuple.type.loc16 (%tuple.type.c14) = tuple_init (%.loc17_22.2, %.loc17_22.3, %.loc17_22.4) to %return
// CHECK:STDOUT:     %.loc17_23: init @Y.as.A.impl.F.%tuple.type.loc16 (%tuple.type.c14) = converted %.loc17_22.1, %.loc17_22.5
// CHECK:STDOUT:     return %.loc17_23 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %u.patt: %pattern_type.8f9 = binding_pattern u [concrete]
// CHECK:STDOUT:     %u.var_patt: %pattern_type.8f9 = var_pattern %u.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %u.var: ref %Z = var %u.var_patt
// CHECK:STDOUT:   %Z.ref.loc22: type = name_ref Z, file.%Z.decl [concrete = constants.%Z]
// CHECK:STDOUT:   %u: ref %Z = bind_name u, %u.var
// CHECK:STDOUT:   %Y.ref: type = name_ref Y, file.%Y.decl [concrete = constants.%Y]
// CHECK:STDOUT:   %A.ref: %A.type.495 = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:   %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %A.type: type = facet_type <@A, @A(constants.%X)> [concrete = constants.%A.type.515]
// CHECK:STDOUT:   %A.facet: %A.type.515 = facet_value constants.%Y, (constants.%A.impl_witness) [concrete = constants.%A.facet.f2b]
// CHECK:STDOUT:   %.loc23_6: %A.type.515 = converted %Y.ref, %A.facet [concrete = constants.%A.facet.f2b]
// CHECK:STDOUT:   %.loc23_14.1: %A.assoc_type.296 = specific_constant @A.%assoc0.loc6_41.1, @A(constants.%X) [concrete = constants.%assoc0.5f6]
// CHECK:STDOUT:   %F.ref: %A.assoc_type.296 = name_ref F, %.loc23_14.1 [concrete = constants.%assoc0.5f6]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc23_6 [concrete = constants.%Y]
// CHECK:STDOUT:   %.loc23_14.2: type = converted %.loc23_6, %as_type [concrete = constants.%Y]
// CHECK:STDOUT:   %impl.elem0: %.7c0 = impl_witness_access constants.%A.impl_witness, element0 [concrete = constants.%Y.as.A.impl.F]
// CHECK:STDOUT:   %Z.ref.loc23: type = name_ref Z, file.%Z.decl [concrete = constants.%Z]
// CHECK:STDOUT:   %u.ref: ref %Z = name_ref u, %u
// CHECK:STDOUT:   %addr.loc23_20: %ptr.fb6 = addr_of %u.ref
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Y.as.A.impl.F(constants.%Z) [concrete = constants.%Y.as.A.impl.F.specific_fn]
// CHECK:STDOUT:   %.loc23_22.1: ref %tuple.type.092 = temporary_storage
// CHECK:STDOUT:   %Y.as.A.impl.F.call: init %tuple.type.092 = call %specific_fn(%addr.loc23_20) to %.loc23_22.1
// CHECK:STDOUT:   %.loc23_22.2: ref %tuple.type.092 = temporary %.loc23_22.1, %Y.as.A.impl.F.call
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc23_22.2, constants.%T.as.Destroy.impl.Op.64e
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function constants.%T.as.Destroy.impl.Op.64e, @T.as.Destroy.impl.Op(constants.%tuple.type.092) [concrete = constants.%T.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc23_22.2, %T.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc23_22: %ptr.5df = addr_of %.loc23_22.2
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr.loc23_22)
// CHECK:STDOUT:   %Z.as.Destroy.impl.Op.bound: <bound method> = bound_method %u.var, constants.%Z.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr.loc22: %ptr.fb6 = addr_of %u.var
// CHECK:STDOUT:   %Z.as.Destroy.impl.Op.call: init %empty_tuple.type = call %Z.as.Destroy.impl.Op.bound(%addr.loc22)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallGeneric(%T.loc26_16.2: %A.type.515) {
// CHECK:STDOUT:   %T.loc26_16.1: %A.type.515 = bind_symbolic_name T, 0 [symbolic = %T.loc26_16.1 (constants.%T.3b6)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc28_4.2: type = facet_access_type %T.loc26_16.1 [symbolic = %T.as_type.loc28_4.2 (constants.%T.as_type.fc2)]
// CHECK:STDOUT:   %A.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc26_16.1, @A, @A(constants.%X) [symbolic = %A.lookup_impl_witness (constants.%A.lookup_impl_witness)]
// CHECK:STDOUT:   %A.facet: %A.type.515 = facet_value %T.as_type.loc28_4.2, (%A.lookup_impl_witness) [symbolic = %A.facet (constants.%A.facet.97b)]
// CHECK:STDOUT:   %.loc28_4.3: type = fn_type_with_self_type constants.%A.F.type.13d, %A.facet [symbolic = %.loc28_4.3 (constants.%.7c8)]
// CHECK:STDOUT:   %impl.elem0.loc28_4.2: @CallGeneric.%.loc28_4.3 (%.7c8) = impl_witness_access %A.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc28_4.2 (constants.%impl.elem0.eda)]
// CHECK:STDOUT:   %specific_impl_fn.loc28_4.2: <specific function> = specific_impl_function %impl.elem0.loc28_4.2, @A.F(constants.%X, %A.facet, constants.%Z) [symbolic = %specific_impl_fn.loc28_4.2 (constants.%specific_impl_fn.9ce)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (constants.%X, %T.as_type.loc28_4.2, constants.%ptr.fb6) [symbolic = %tuple.type (constants.%tuple.type.fd8)]
// CHECK:STDOUT:   %require_complete.loc28_12.1: <witness> = require_complete_type %tuple.type [symbolic = %require_complete.loc28_12.1 (constants.%require_complete.f71)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %tuple.type, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %tuple.type, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet (constants.%Destroy.facet.709)]
// CHECK:STDOUT:   %.loc28_12.5: type = fn_type_with_self_type constants.%Destroy.Op.type, %Destroy.facet [symbolic = %.loc28_12.5 (constants.%.768)]
// CHECK:STDOUT:   %impl.elem0.loc28_12.2: @CallGeneric.%.loc28_12.5 (%.768) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc28_12.2 (constants.%impl.elem0.c2c)]
// CHECK:STDOUT:   %specific_impl_fn.loc28_12.2: <specific function> = specific_impl_function %impl.elem0.loc28_12.2, @Destroy.Op(%Destroy.facet) [symbolic = %specific_impl_fn.loc28_12.2 (constants.%specific_impl_fn.5e2)]
// CHECK:STDOUT:   %ptr: type = ptr_type %tuple.type [symbolic = %ptr (constants.%ptr.1b6)]
// CHECK:STDOUT:   %require_complete.loc28_12.2: <witness> = require_complete_type %ptr [symbolic = %require_complete.loc28_12.2 (constants.%require_complete.cda)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %u.patt: %pattern_type.8f9 = binding_pattern u [concrete]
// CHECK:STDOUT:       %u.var_patt: %pattern_type.8f9 = var_pattern %u.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u.var: ref %Z = var %u.var_patt
// CHECK:STDOUT:     %Z.ref.loc27: type = name_ref Z, file.%Z.decl [concrete = constants.%Z]
// CHECK:STDOUT:     %u: ref %Z = bind_name u, %u.var
// CHECK:STDOUT:     %T.ref: %A.type.515 = name_ref T, %T.loc26_16.2 [symbolic = %T.loc26_16.1 (constants.%T.3b6)]
// CHECK:STDOUT:     %.loc28_4.1: %A.assoc_type.296 = specific_constant @A.%assoc0.loc6_41.1, @A(constants.%X) [concrete = constants.%assoc0.5f6]
// CHECK:STDOUT:     %F.ref: %A.assoc_type.296 = name_ref F, %.loc28_4.1 [concrete = constants.%assoc0.5f6]
// CHECK:STDOUT:     %T.as_type.loc28_4.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc28_4.2 (constants.%T.as_type.fc2)]
// CHECK:STDOUT:     %.loc28_4.2: type = converted %T.ref, %T.as_type.loc28_4.1 [symbolic = %T.as_type.loc28_4.2 (constants.%T.as_type.fc2)]
// CHECK:STDOUT:     %impl.elem0.loc28_4.1: @CallGeneric.%.loc28_4.3 (%.7c8) = impl_witness_access constants.%A.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc28_4.2 (constants.%impl.elem0.eda)]
// CHECK:STDOUT:     %Z.ref.loc28: type = name_ref Z, file.%Z.decl [concrete = constants.%Z]
// CHECK:STDOUT:     %u.ref: ref %Z = name_ref u, %u
// CHECK:STDOUT:     %addr.loc28_10: %ptr.fb6 = addr_of %u.ref
// CHECK:STDOUT:     %specific_impl_fn.loc28_4.1: <specific function> = specific_impl_function %impl.elem0.loc28_4.1, @A.F(constants.%X, constants.%A.facet.97b, constants.%Z) [symbolic = %specific_impl_fn.loc28_4.2 (constants.%specific_impl_fn.9ce)]
// CHECK:STDOUT:     %.loc28_12.1: ref @CallGeneric.%tuple.type (%tuple.type.fd8) = temporary_storage
// CHECK:STDOUT:     %.loc28_12.2: init @CallGeneric.%tuple.type (%tuple.type.fd8) = call %specific_impl_fn.loc28_4.1(%addr.loc28_10) to %.loc28_12.1
// CHECK:STDOUT:     %.loc28_12.3: ref @CallGeneric.%tuple.type (%tuple.type.fd8) = temporary %.loc28_12.1, %.loc28_12.2
// CHECK:STDOUT:     %impl.elem0.loc28_12.1: @CallGeneric.%.loc28_12.5 (%.768) = impl_witness_access constants.%Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc28_12.2 (constants.%impl.elem0.c2c)]
// CHECK:STDOUT:     %bound_method.loc28_12.1: <bound method> = bound_method %.loc28_12.3, %impl.elem0.loc28_12.1
// CHECK:STDOUT:     %specific_impl_fn.loc28_12.1: <specific function> = specific_impl_function %impl.elem0.loc28_12.1, @Destroy.Op(constants.%Destroy.facet.709) [symbolic = %specific_impl_fn.loc28_12.2 (constants.%specific_impl_fn.5e2)]
// CHECK:STDOUT:     %bound_method.loc28_12.2: <bound method> = bound_method %.loc28_12.3, %specific_impl_fn.loc28_12.1
// CHECK:STDOUT:     %addr.loc28_12: @CallGeneric.%ptr (%ptr.1b6) = addr_of %.loc28_12.3
// CHECK:STDOUT:     %.loc28_12.4: init %empty_tuple.type = call %bound_method.loc28_12.2(%addr.loc28_12)
// CHECK:STDOUT:     %Z.as.Destroy.impl.Op.bound: <bound method> = bound_method %u.var, constants.%Z.as.Destroy.impl.Op
// CHECK:STDOUT:     %addr.loc27: %ptr.fb6 = addr_of %u.var
// CHECK:STDOUT:     %Z.as.Destroy.impl.Op.call: init %empty_tuple.type = call %Z.as.Destroy.impl.Op.bound(%addr.loc27)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallIndirect() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CallGeneric.ref: %CallGeneric.type = name_ref CallGeneric, file.%CallGeneric.decl [concrete = constants.%CallGeneric]
// CHECK:STDOUT:   %Y.ref: type = name_ref Y, file.%Y.decl [concrete = constants.%Y]
// CHECK:STDOUT:   %A.facet: %A.type.515 = facet_value constants.%Y, (constants.%A.impl_witness) [concrete = constants.%A.facet.f2b]
// CHECK:STDOUT:   %.loc32: %A.type.515 = converted %Y.ref, %A.facet [concrete = constants.%A.facet.f2b]
// CHECK:STDOUT:   %CallGeneric.specific_fn: <specific function> = specific_function %CallGeneric.ref, @CallGeneric(constants.%A.facet.f2b) [concrete = constants.%CallGeneric.specific_fn]
// CHECK:STDOUT:   %CallGeneric.call: init %empty_tuple.type = call %CallGeneric.specific_fn()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc5_13.1 => constants.%T.8b3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.F(constants.%T.8b3, constants.%Self.b74, constants.%U.2cb) {
// CHECK:STDOUT:   %U.loc6_8.1 => constants.%U.2cb
// CHECK:STDOUT:   %ptr.loc6_22.1 => constants.%ptr.204
// CHECK:STDOUT:   %pattern_type.loc6_18 => constants.%pattern_type.eee
// CHECK:STDOUT:   %T => constants.%T.8b3
// CHECK:STDOUT:   %A.type => constants.%A.type.806
// CHECK:STDOUT:   %Self => constants.%Self.b74
// CHECK:STDOUT:   %tuple.type.loc6_40.1 => constants.%tuple.type.4c5
// CHECK:STDOUT:   %Self.as_type.loc6_40.1 => constants.%Self.as_type.327
// CHECK:STDOUT:   %tuple.type.loc6_40.2 => constants.%tuple.type.5a8
// CHECK:STDOUT:   %pattern_type.loc6_25 => constants.%pattern_type.0eb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%X) {
// CHECK:STDOUT:   %T.loc5_13.1 => constants.%X
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.type => constants.%A.type.515
// CHECK:STDOUT:   %Self.2 => constants.%Self.522
// CHECK:STDOUT:   %A.F.type => constants.%A.F.type.13d
// CHECK:STDOUT:   %A.F => constants.%A.F.d83
// CHECK:STDOUT:   %A.assoc_type => constants.%A.assoc_type.296
// CHECK:STDOUT:   %assoc0.loc6_41.2 => constants.%assoc0.5f6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y.as.A.impl.F(constants.%U.8b3) {
// CHECK:STDOUT:   %U.loc16_8.1 => constants.%U.8b3
// CHECK:STDOUT:   %ptr.loc16_22.1 => constants.%ptr.79f131.1
// CHECK:STDOUT:   %pattern_type.loc16_18 => constants.%pattern_type.afe393.1
// CHECK:STDOUT:   %tuple.type.loc16 => constants.%tuple.type.c14
// CHECK:STDOUT:   %pattern_type.loc16_25 => constants.%pattern_type.ac9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.F(constants.%X, constants.%A.facet.f2b, constants.%U.8b3) {
// CHECK:STDOUT:   %U.loc6_8.1 => constants.%U.8b3
// CHECK:STDOUT:   %ptr.loc6_22.1 => constants.%ptr.79f131.1
// CHECK:STDOUT:   %pattern_type.loc6_18 => constants.%pattern_type.afe393.1
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.515
// CHECK:STDOUT:   %Self => constants.%A.facet.f2b
// CHECK:STDOUT:   %tuple.type.loc6_40.1 => constants.%tuple.type.281
// CHECK:STDOUT:   %Self.as_type.loc6_40.1 => constants.%Y
// CHECK:STDOUT:   %tuple.type.loc6_40.2 => constants.%tuple.type.c14
// CHECK:STDOUT:   %pattern_type.loc6_25 => constants.%pattern_type.ac9
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
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.impl_witness.f8c
// CHECK:STDOUT:   %Copy.facet => constants.%Copy.facet.27d
// CHECK:STDOUT:   %.loc17_21.2 => constants.%.67f
// CHECK:STDOUT:   %impl.elem0.loc17_21.2 => constants.%ptr.as.Copy.impl.Op.ac3
// CHECK:STDOUT:   %specific_impl_fn.loc17_21.2 => constants.%ptr.as.Copy.impl.Op.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGeneric(constants.%T.3b6) {
// CHECK:STDOUT:   %T.loc26_16.1 => constants.%T.3b6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.F(constants.%X, constants.%A.facet.97b, constants.%Z) {
// CHECK:STDOUT:   %U.loc6_8.1 => constants.%Z
// CHECK:STDOUT:   %ptr.loc6_22.1 => constants.%ptr.fb6
// CHECK:STDOUT:   %pattern_type.loc6_18 => constants.%pattern_type.f76
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.515
// CHECK:STDOUT:   %Self => constants.%A.facet.97b
// CHECK:STDOUT:   %tuple.type.loc6_40.1 => constants.%tuple.type.281
// CHECK:STDOUT:   %Self.as_type.loc6_40.1 => constants.%T.as_type.fc2
// CHECK:STDOUT:   %tuple.type.loc6_40.2 => constants.%tuple.type.fd8
// CHECK:STDOUT:   %pattern_type.loc6_25 => constants.%pattern_type.1d1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGeneric(constants.%A.facet.f2b) {
// CHECK:STDOUT:   %T.loc26_16.1 => constants.%A.facet.f2b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc28_4.2 => constants.%Y
// CHECK:STDOUT:   %A.lookup_impl_witness => constants.%A.impl_witness
// CHECK:STDOUT:   %A.facet => constants.%A.facet.f2b
// CHECK:STDOUT:   %.loc28_4.3 => constants.%.7c0
// CHECK:STDOUT:   %impl.elem0.loc28_4.2 => constants.%Y.as.A.impl.F
// CHECK:STDOUT:   %specific_impl_fn.loc28_4.2 => constants.%Y.as.A.impl.F.specific_fn
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.092
// CHECK:STDOUT:   %require_complete.loc28_12.1 => constants.%complete_type.05d
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%Destroy.impl_witness.d7d
// CHECK:STDOUT:   %Destroy.facet => constants.%Destroy.facet.ed1
// CHECK:STDOUT:   %.loc28_12.5 => constants.%.928
// CHECK:STDOUT:   %impl.elem0.loc28_12.2 => constants.%T.as.Destroy.impl.Op.64e
// CHECK:STDOUT:   %specific_impl_fn.loc28_12.2 => constants.%T.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %ptr => constants.%ptr.5df
// CHECK:STDOUT:   %require_complete.loc28_12.2 => constants.%complete_type.aab
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.F(constants.%X, constants.%A.facet.f2b, constants.%Z) {
// CHECK:STDOUT:   %U.loc6_8.1 => constants.%Z
// CHECK:STDOUT:   %ptr.loc6_22.1 => constants.%ptr.fb6
// CHECK:STDOUT:   %pattern_type.loc6_18 => constants.%pattern_type.f76
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.515
// CHECK:STDOUT:   %Self => constants.%A.facet.f2b
// CHECK:STDOUT:   %tuple.type.loc6_40.1 => constants.%tuple.type.281
// CHECK:STDOUT:   %Self.as_type.loc6_40.1 => constants.%Y
// CHECK:STDOUT:   %tuple.type.loc6_40.2 => constants.%tuple.type.092
// CHECK:STDOUT:   %pattern_type.loc6_25 => constants.%pattern_type.48d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- generic_method_more_params.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %A.type.495: type = generic_interface_type @A [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %A.generic: %A.type.495 = struct_value () [concrete]
// CHECK:STDOUT:   %A.type.806: type = facet_type <@A, @A(%T.8b3)> [symbolic]
// CHECK:STDOUT:   %Self.b74: %A.type.806 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %U.2cb: type = bind_symbolic_name U, 2 [symbolic]
// CHECK:STDOUT:   %pattern_type.20f: type = pattern_type %U.2cb [symbolic]
// CHECK:STDOUT:   %tuple.type.4c5: type = tuple_type (type, %A.type.806, type) [symbolic]
// CHECK:STDOUT:   %Self.as_type.327: type = facet_access_type %Self.b74 [symbolic]
// CHECK:STDOUT:   %tuple.type.6ec: type = tuple_type (%T.8b3, %Self.as_type.327, %U.2cb) [symbolic]
// CHECK:STDOUT:   %pattern_type.e2e: type = pattern_type %tuple.type.6ec [symbolic]
// CHECK:STDOUT:   %A.F.type.17a: type = fn_type @A.F, @A(%T.8b3) [symbolic]
// CHECK:STDOUT:   %A.F.0d8: %A.F.type.17a = struct_value () [symbolic]
// CHECK:STDOUT:   %A.assoc_type.ed3: type = assoc_entity_type @A, @A(%T.8b3) [symbolic]
// CHECK:STDOUT:   %assoc0.fcb: %A.assoc_type.ed3 = assoc_entity element0, @A.%A.F.decl [symbolic]
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.286: <witness> = impl_witness @X.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.d17: type = ptr_type %X [concrete]
// CHECK:STDOUT:   %pattern_type.1c6: type = pattern_type %ptr.d17 [concrete]
// CHECK:STDOUT:   %X.as.Destroy.impl.Op.type: type = fn_type @X.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %X.as.Destroy.impl.Op: %X.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Y1: type = class_type @Y1 [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.6b0: <witness> = impl_witness @Y1.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.723: type = ptr_type %Y1 [concrete]
// CHECK:STDOUT:   %pattern_type.273: type = pattern_type %ptr.723 [concrete]
// CHECK:STDOUT:   %Y1.as.Destroy.impl.Op.type: type = fn_type @Y1.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %Y1.as.Destroy.impl.Op: %Y1.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Y2: type = class_type @Y2 [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.dbf: <witness> = impl_witness @Y2.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.23d: type = ptr_type %Y2 [concrete]
// CHECK:STDOUT:   %pattern_type.484: type = pattern_type %ptr.23d [concrete]
// CHECK:STDOUT:   %Y2.as.Destroy.impl.Op.type: type = fn_type @Y2.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %Y2.as.Destroy.impl.Op: %Y2.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Z: type = class_type @Z [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.417: <witness> = impl_witness @Z.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.fb6: type = ptr_type %Z [concrete]
// CHECK:STDOUT:   %pattern_type.f76: type = pattern_type %ptr.fb6 [concrete]
// CHECK:STDOUT:   %Z.as.Destroy.impl.Op.type: type = fn_type @Z.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %Z.as.Destroy.impl.Op: %Z.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %V1: type = bind_symbolic_name V1, 0 [symbolic]
// CHECK:STDOUT:   %V2: type = bind_symbolic_name V2, 1 [symbolic]
// CHECK:STDOUT:   %W: type = bind_symbolic_name W, 2 [symbolic]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.30b: type = tuple_type (%V1, %V2) [symbolic]
// CHECK:STDOUT:   %A.type.ffa: type = facet_type <@A, @A(%W)> [symbolic]
// CHECK:STDOUT:   %Self.0b4: %A.type.ffa = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %A.F.type.904: type = fn_type @A.F, @A(%W) [symbolic]
// CHECK:STDOUT:   %A.F.1ee: %A.F.type.904 = struct_value () [symbolic]
// CHECK:STDOUT:   %A.assoc_type.b05: type = assoc_entity_type @A, @A(%W) [symbolic]
// CHECK:STDOUT:   %assoc0.c82: %A.assoc_type.b05 = assoc_entity element0, @A.%A.F.decl [symbolic]
// CHECK:STDOUT:   %require_complete.116: <witness> = require_complete_type %A.type.ffa [symbolic]
// CHECK:STDOUT:   %A.impl_witness.3ab: <witness> = impl_witness file.%A.impl_witness_table, @tuple.type.as.A.impl(%V1, %V2, %W) [symbolic]
// CHECK:STDOUT:   %U.753: type = bind_symbolic_name U, 3 [symbolic]
// CHECK:STDOUT:   %pattern_type.549: type = pattern_type %U.753 [symbolic]
// CHECK:STDOUT:   %tuple.type.11f: type = tuple_type (type, %tuple.type.24b, type) [concrete]
// CHECK:STDOUT:   %tuple.type.8ae: type = tuple_type (%W, %tuple.type.30b, %U.753) [symbolic]
// CHECK:STDOUT:   %pattern_type.d4a: type = pattern_type %tuple.type.8ae [symbolic]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type.dac: type = fn_type @tuple.type.as.A.impl.F, @tuple.type.as.A.impl(%V1, %V2, %W) [symbolic]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.9d5: %tuple.type.as.A.impl.F.type.dac = struct_value () [symbolic]
// CHECK:STDOUT:   %A.facet.b6a: %A.type.ffa = facet_value %tuple.type.30b, (%A.impl_witness.3ab) [symbolic]
// CHECK:STDOUT:   %tuple.type.6bd: type = tuple_type (type, %A.type.ffa, type) [symbolic]
// CHECK:STDOUT:   %require_complete.d4d: <witness> = require_complete_type %tuple.type.8ae [symbolic]
// CHECK:STDOUT:   %require_complete.ed4: <witness> = require_complete_type %U.753 [symbolic]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.specific_fn.cc7: <specific function> = specific_function %tuple.type.as.A.impl.F.9d5, @tuple.type.as.A.impl.F(%V1, %V2, %W, %U.753) [symbolic]
// CHECK:STDOUT:   %Call.type: type = fn_type @Call [concrete]
// CHECK:STDOUT:   %Call: %Call.type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.a46: type = tuple_type (%Y1, %Y2) [concrete]
// CHECK:STDOUT:   %A.type.515: type = facet_type <@A, @A(%X)> [concrete]
// CHECK:STDOUT:   %Self.522: %A.type.515 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %A.F.type.13d: type = fn_type @A.F, @A(%X) [concrete]
// CHECK:STDOUT:   %A.F.d83: %A.F.type.13d = struct_value () [concrete]
// CHECK:STDOUT:   %A.assoc_type.296: type = assoc_entity_type @A, @A(%X) [concrete]
// CHECK:STDOUT:   %assoc0.5f6: %A.assoc_type.296 = assoc_entity element0, @A.%A.F.decl [concrete]
// CHECK:STDOUT:   %complete_type.1a4: <witness> = complete_type_witness %A.type.515 [concrete]
// CHECK:STDOUT:   %A.impl_witness.0ca: <witness> = impl_witness file.%A.impl_witness_table, @tuple.type.as.A.impl(%Y1, %Y2, %X) [concrete]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type.490: type = fn_type @tuple.type.as.A.impl.F, @tuple.type.as.A.impl(%Y1, %Y2, %X) [concrete]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.d69: %tuple.type.as.A.impl.F.type.490 = struct_value () [concrete]
// CHECK:STDOUT:   %A.facet.2e2: %A.type.515 = facet_value %tuple.type.a46, (%A.impl_witness.0ca) [concrete]
// CHECK:STDOUT:   %.9db: type = fn_type_with_self_type %A.F.type.13d, %A.facet.2e2 [concrete]
// CHECK:STDOUT:   %pattern_type.8f9: type = pattern_type %Z [concrete]
// CHECK:STDOUT:   %tuple.type.415: type = tuple_type (%X, %tuple.type.a46, %Z) [concrete]
// CHECK:STDOUT:   %pattern_type.0b2: type = pattern_type %tuple.type.415 [concrete]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.specific_fn.6bf: <specific function> = specific_function %tuple.type.as.A.impl.F.d69, @tuple.type.as.A.impl.F(%Y1, %Y2, %X, %Z) [concrete]
// CHECK:STDOUT:   %Z.val: %Z = struct_value () [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.305: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%T.8b3) [symbolic]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.e3a: %T.as.Destroy.impl.Op.type.305 = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.c7e: <witness> = impl_witness imports.%Destroy.impl_witness_table, @T.as.Destroy.impl(%tuple.type.415) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.6d0: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%tuple.type.415) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.f0c: %T.as.Destroy.impl.Op.type.6d0 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.ad9: type = ptr_type %tuple.type.415 [concrete]
// CHECK:STDOUT:   %complete_type.e23: <witness> = complete_type_witness %ptr.ad9 [concrete]
// CHECK:STDOUT:   %Destroy.facet.c14: %Destroy.type = facet_value %tuple.type.415, (%Destroy.impl_witness.c7e) [concrete]
// CHECK:STDOUT:   %.397: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet.c14 [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %T.as.Destroy.impl.Op.f0c, @T.as.Destroy.impl.Op(%tuple.type.415) [concrete]
// CHECK:STDOUT:   %T.3b6: %A.type.515 = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.05f: type = pattern_type %A.type.515 [concrete]
// CHECK:STDOUT:   %CallGeneric.type: type = fn_type @CallGeneric [concrete]
// CHECK:STDOUT:   %CallGeneric: %CallGeneric.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.3b6 [symbolic]
// CHECK:STDOUT:   %A.lookup_impl_witness: <witness> = lookup_impl_witness %T.3b6, @A, @A(%X) [symbolic]
// CHECK:STDOUT:   %A.facet.97b: %A.type.515 = facet_value %T.as_type, (%A.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %.7c8: type = fn_type_with_self_type %A.F.type.13d, %A.facet.97b [symbolic]
// CHECK:STDOUT:   %impl.elem0.eda: %.7c8 = impl_witness_access %A.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %tuple.type.281: type = tuple_type (type, %A.type.515, type) [concrete]
// CHECK:STDOUT:   %tuple.type.379: type = tuple_type (%X, %T.as_type, %Z) [symbolic]
// CHECK:STDOUT:   %pattern_type.fcb: type = pattern_type %tuple.type.379 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.9ce: <specific function> = specific_impl_function %impl.elem0.eda, @A.F(%X, %A.facet.97b, %Z) [symbolic]
// CHECK:STDOUT:   %require_complete.d0c: <witness> = require_complete_type %tuple.type.379 [symbolic]
// CHECK:STDOUT:   %ptr.e41: type = ptr_type %tuple.type.379 [symbolic]
// CHECK:STDOUT:   %require_complete.1f9: <witness> = require_complete_type %ptr.e41 [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %tuple.type.379, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.a89: %Destroy.type = facet_value %tuple.type.379, (%Destroy.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %.a0b: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet.a89 [symbolic]
// CHECK:STDOUT:   %impl.elem0.80a: %.a0b = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.515: <specific function> = specific_impl_function %impl.elem0.80a, @Destroy.Op(%Destroy.facet.a89) [symbolic]
// CHECK:STDOUT:   %CallIndirect.type: type = fn_type @CallIndirect [concrete]
// CHECK:STDOUT:   %CallIndirect: %CallIndirect.type = struct_value () [concrete]
// CHECK:STDOUT:   %CallGeneric.specific_fn: <specific function> = specific_function %CallGeneric, @CallGeneric(%A.facet.2e2) [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.efe: @T.as.Destroy.impl.%T.as.Destroy.impl.Op.type (%T.as.Destroy.impl.Op.type.305) = import_ref Core//prelude/parts/destroy, loc8_29, loaded [symbolic = @T.as.Destroy.impl.%T.as.Destroy.impl.Op (constants.%T.as.Destroy.impl.Op.e3a)]
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (%Core.import_ref.efe), @T.as.Destroy.impl [concrete]
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
// CHECK:STDOUT:     %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc5_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc5_13.1 (constants.%T.8b3)]
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
// CHECK:STDOUT:     %.loc14_53.2: type = converted %.loc14_53.1, constants.%tuple.type.30b [symbolic = %tuple.type (constants.%tuple.type.30b)]
// CHECK:STDOUT:     %A.ref: %A.type.495 = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:     %W.ref: type = name_ref W, %W.loc14_36.2 [symbolic = %W.loc14_36.1 (constants.%W)]
// CHECK:STDOUT:     %A.type.loc14_61.2: type = facet_type <@A, @A(constants.%W)> [symbolic = %A.type.loc14_61.1 (constants.%A.type.ffa)]
// CHECK:STDOUT:     %.Self.1: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %V1.loc14_14.2: type = bind_symbolic_name V1, 0 [symbolic = %V1.loc14_14.1 (constants.%V1)]
// CHECK:STDOUT:     %.Self.2: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %V2.loc14_25.2: type = bind_symbolic_name V2, 1 [symbolic = %V2.loc14_25.1 (constants.%V2)]
// CHECK:STDOUT:     %.Self.3: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %W.loc14_36.2: type = bind_symbolic_name W, 2 [symbolic = %W.loc14_36.1 (constants.%W)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.impl_witness_table = impl_witness_table (@tuple.type.as.A.impl.%tuple.type.as.A.impl.F.decl), @tuple.type.as.A.impl [concrete]
// CHECK:STDOUT:   %A.impl_witness: <witness> = impl_witness %A.impl_witness_table, @tuple.type.as.A.impl(constants.%V1, constants.%V2, constants.%W) [symbolic = @tuple.type.as.A.impl.%A.impl_witness (constants.%A.impl_witness.3ab)]
// CHECK:STDOUT:   %Call.decl: %Call.type = fn_decl @Call [concrete = constants.%Call] {} {}
// CHECK:STDOUT:   %CallGeneric.decl: %CallGeneric.type = fn_decl @CallGeneric [concrete = constants.%CallGeneric] {
// CHECK:STDOUT:     %T.patt: %pattern_type.05f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc27: type = splice_block %A.type [concrete = constants.%A.type.515] {
// CHECK:STDOUT:       %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %A.ref: %A.type.495 = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:       %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:       %A.type: type = facet_type <@A, @A(constants.%X)> [concrete = constants.%A.type.515]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc27_16.2: %A.type.515 = bind_symbolic_name T, 0 [symbolic = %T.loc27_16.1 (constants.%T.3b6)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallIndirect.decl: %CallIndirect.type = fn_decl @CallIndirect [concrete = constants.%CallIndirect] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @A(%T.loc5_13.2: type) {
// CHECK:STDOUT:   %T.loc5_13.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc5_13.1 (constants.%T.8b3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.type: type = facet_type <@A, @A(%T.loc5_13.1)> [symbolic = %A.type (constants.%A.type.806)]
// CHECK:STDOUT:   %Self.2: @A.%A.type (%A.type.806) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.b74)]
// CHECK:STDOUT:   %A.F.type: type = fn_type @A.F, @A(%T.loc5_13.1) [symbolic = %A.F.type (constants.%A.F.type.17a)]
// CHECK:STDOUT:   %A.F: @A.%A.F.type (%A.F.type.17a) = struct_value () [symbolic = %A.F (constants.%A.F.0d8)]
// CHECK:STDOUT:   %A.assoc_type: type = assoc_entity_type @A, @A(%T.loc5_13.1) [symbolic = %A.assoc_type (constants.%A.assoc_type.ed3)]
// CHECK:STDOUT:   %assoc0.loc6_39.2: @A.%A.assoc_type (%A.assoc_type.ed3) = assoc_entity element0, %A.F.decl [symbolic = %assoc0.loc6_39.2 (constants.%assoc0.fcb)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @A.%A.type (%A.type.806) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.b74)]
// CHECK:STDOUT:     %A.F.decl: @A.%A.F.type (%A.F.type.17a) = fn_decl @A.F [symbolic = @A.%A.F (constants.%A.F.0d8)] {
// CHECK:STDOUT:       %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 2 [concrete]
// CHECK:STDOUT:       %u.patt: @A.F.%pattern_type.loc6_18 (%pattern_type.20f) = binding_pattern u [concrete]
// CHECK:STDOUT:       %u.param_patt: @A.F.%pattern_type.loc6_18 (%pattern_type.20f) = value_param_pattern %u.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @A.F.%pattern_type.loc6_24 (%pattern_type.e2e) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @A.F.%pattern_type.loc6_24 (%pattern_type.e2e) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @A.%T.loc5_13.2 [symbolic = %T (constants.%T.8b3)]
// CHECK:STDOUT:       %.loc6_31: @A.F.%A.type (%A.type.806) = specific_constant @A.%Self.1, @A(constants.%T.8b3) [symbolic = %Self (constants.%Self.b74)]
// CHECK:STDOUT:       %Self.ref: @A.F.%A.type (%A.type.806) = name_ref Self, %.loc6_31 [symbolic = %Self (constants.%Self.b74)]
// CHECK:STDOUT:       %U.ref.loc6_37: type = name_ref U, %U.loc6_8.2 [symbolic = %U.loc6_8.1 (constants.%U.2cb)]
// CHECK:STDOUT:       %.loc6_38.1: @A.F.%tuple.type.loc6_38.1 (%tuple.type.4c5) = tuple_literal (%T.ref, %Self.ref, %U.ref.loc6_37)
// CHECK:STDOUT:       %Self.as_type.loc6_38.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc6_38.1 (constants.%Self.as_type.327)]
// CHECK:STDOUT:       %.loc6_38.2: type = converted %Self.ref, %Self.as_type.loc6_38.2 [symbolic = %Self.as_type.loc6_38.1 (constants.%Self.as_type.327)]
// CHECK:STDOUT:       %.loc6_38.3: type = converted %.loc6_38.1, constants.%tuple.type.6ec [symbolic = %tuple.type.loc6_38.2 (constants.%tuple.type.6ec)]
// CHECK:STDOUT:       %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %U.loc6_8.2: type = bind_symbolic_name U, 2 [symbolic = %U.loc6_8.1 (constants.%U.2cb)]
// CHECK:STDOUT:       %u.param: @A.F.%U.loc6_8.1 (%U.2cb) = value_param call_param0
// CHECK:STDOUT:       %U.ref.loc6_21: type = name_ref U, %U.loc6_8.2 [symbolic = %U.loc6_8.1 (constants.%U.2cb)]
// CHECK:STDOUT:       %u: @A.F.%U.loc6_8.1 (%U.2cb) = bind_name u, %u.param
// CHECK:STDOUT:       %return.param: ref @A.F.%tuple.type.loc6_38.2 (%tuple.type.6ec) = out_param call_param1
// CHECK:STDOUT:       %return: ref @A.F.%tuple.type.loc6_38.2 (%tuple.type.6ec) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc6_39.1: @A.%A.assoc_type (%A.assoc_type.ed3) = assoc_entity element0, %A.F.decl [symbolic = %assoc0.loc6_39.2 (constants.%assoc0.fcb)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .F = %assoc0.loc6_39.1
// CHECK:STDOUT:     witness = (%A.F.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @X.as.Destroy.impl: @X.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:   %X.as.Destroy.impl.Op.decl: %X.as.Destroy.impl.Op.type = fn_decl @X.as.Destroy.impl.Op [concrete = constants.%X.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.1c6 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.1c6 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc9: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.d17 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%X [concrete = constants.%X]
// CHECK:STDOUT:     %self: %ptr.d17 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %X.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @X.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Y1.as.Destroy.impl: @Y1.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:   %Y1.as.Destroy.impl.Op.decl: %Y1.as.Destroy.impl.Op.type = fn_decl @Y1.as.Destroy.impl.Op [concrete = constants.%Y1.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.273 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.273 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc10: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.723 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Y1 [concrete = constants.%Y1]
// CHECK:STDOUT:     %self: %ptr.723 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %Y1.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @Y1.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Y2.as.Destroy.impl: @Y2.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:   %Y2.as.Destroy.impl.Op.decl: %Y2.as.Destroy.impl.Op.type = fn_decl @Y2.as.Destroy.impl.Op [concrete = constants.%Y2.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.484 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.484 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc11: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.23d = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Y2 [concrete = constants.%Y2]
// CHECK:STDOUT:     %self: %ptr.23d = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %Y2.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @Y2.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Z.as.Destroy.impl: @Z.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:   %Z.as.Destroy.impl.Op.decl: %Z.as.Destroy.impl.Op.type = fn_decl @Z.as.Destroy.impl.Op [concrete = constants.%Z.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.f76 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.f76 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc12: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.fb6 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Z [concrete = constants.%Z]
// CHECK:STDOUT:     %self: %ptr.fb6 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %Z.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @Z.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @tuple.type.as.A.impl(%V1.loc14_14.2: type, %V2.loc14_25.2: type, %W.loc14_36.2: type) {
// CHECK:STDOUT:   %V1.loc14_14.1: type = bind_symbolic_name V1, 0 [symbolic = %V1.loc14_14.1 (constants.%V1)]
// CHECK:STDOUT:   %V2.loc14_25.1: type = bind_symbolic_name V2, 1 [symbolic = %V2.loc14_25.1 (constants.%V2)]
// CHECK:STDOUT:   %W.loc14_36.1: type = bind_symbolic_name W, 2 [symbolic = %W.loc14_36.1 (constants.%W)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%V1.loc14_14.1, %V2.loc14_25.1) [symbolic = %tuple.type (constants.%tuple.type.30b)]
// CHECK:STDOUT:   %A.type.loc14_61.1: type = facet_type <@A, @A(%W.loc14_36.1)> [symbolic = %A.type.loc14_61.1 (constants.%A.type.ffa)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %A.type.loc14_61.1 [symbolic = %require_complete (constants.%require_complete.116)]
// CHECK:STDOUT:   %A.impl_witness: <witness> = impl_witness file.%A.impl_witness_table, @tuple.type.as.A.impl(%V1.loc14_14.1, %V2.loc14_25.1, %W.loc14_36.1) [symbolic = %A.impl_witness (constants.%A.impl_witness.3ab)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type: type = fn_type @tuple.type.as.A.impl.F, @tuple.type.as.A.impl(%V1.loc14_14.1, %V2.loc14_25.1, %W.loc14_36.1) [symbolic = %tuple.type.as.A.impl.F.type (constants.%tuple.type.as.A.impl.F.type.dac)]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F: @tuple.type.as.A.impl.%tuple.type.as.A.impl.F.type (%tuple.type.as.A.impl.F.type.dac) = struct_value () [symbolic = %tuple.type.as.A.impl.F (constants.%tuple.type.as.A.impl.F.9d5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc14_53.2 as %A.type.loc14_61.2 {
// CHECK:STDOUT:     %tuple.type.as.A.impl.F.decl: @tuple.type.as.A.impl.%tuple.type.as.A.impl.F.type (%tuple.type.as.A.impl.F.type.dac) = fn_decl @tuple.type.as.A.impl.F [symbolic = @tuple.type.as.A.impl.%tuple.type.as.A.impl.F (constants.%tuple.type.as.A.impl.F.9d5)] {
// CHECK:STDOUT:       %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 3 [concrete]
// CHECK:STDOUT:       %u.patt: @tuple.type.as.A.impl.F.%pattern_type.loc17_18 (%pattern_type.549) = binding_pattern u [concrete]
// CHECK:STDOUT:       %u.param_patt: @tuple.type.as.A.impl.F.%pattern_type.loc17_18 (%pattern_type.549) = value_param_pattern %u.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @tuple.type.as.A.impl.F.%pattern_type.loc17_24 (%pattern_type.d4a) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @tuple.type.as.A.impl.F.%pattern_type.loc17_24 (%pattern_type.d4a) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %W.ref: type = name_ref W, @tuple.type.as.A.impl.%W.loc14_36.2 [symbolic = %W (constants.%W)]
// CHECK:STDOUT:       %V1.ref: type = name_ref V1, @tuple.type.as.A.impl.%V1.loc14_14.2 [symbolic = %V1 (constants.%V1)]
// CHECK:STDOUT:       %V2.ref: type = name_ref V2, @tuple.type.as.A.impl.%V2.loc14_25.2 [symbolic = %V2 (constants.%V2)]
// CHECK:STDOUT:       %.loc17_38: %tuple.type.24b = tuple_literal (%V1.ref, %V2.ref)
// CHECK:STDOUT:       %U.ref.loc17_41: type = name_ref U, %U.loc17_8.2 [symbolic = %U.loc17_8.1 (constants.%U.753)]
// CHECK:STDOUT:       %.loc17_42.1: %tuple.type.11f = tuple_literal (%W.ref, %.loc17_38, %U.ref.loc17_41)
// CHECK:STDOUT:       %.loc17_42.2: type = converted %.loc17_38, constants.%tuple.type.30b [symbolic = %tuple.type.loc17_42.1 (constants.%tuple.type.30b)]
// CHECK:STDOUT:       %.loc17_42.3: type = converted %.loc17_42.1, constants.%tuple.type.8ae [symbolic = %tuple.type.loc17_42.2 (constants.%tuple.type.8ae)]
// CHECK:STDOUT:       %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %U.loc17_8.2: type = bind_symbolic_name U, 3 [symbolic = %U.loc17_8.1 (constants.%U.753)]
// CHECK:STDOUT:       %u.param: @tuple.type.as.A.impl.F.%U.loc17_8.1 (%U.753) = value_param call_param0
// CHECK:STDOUT:       %U.ref.loc17_21: type = name_ref U, %U.loc17_8.2 [symbolic = %U.loc17_8.1 (constants.%U.753)]
// CHECK:STDOUT:       %u: @tuple.type.as.A.impl.F.%U.loc17_8.1 (%U.753) = bind_name u, %u.param
// CHECK:STDOUT:       %return.param: ref @tuple.type.as.A.impl.F.%tuple.type.loc17_42.2 (%tuple.type.8ae) = out_param call_param1
// CHECK:STDOUT:       %return: ref @tuple.type.as.A.impl.F.%tuple.type.loc17_42.2 (%tuple.type.8ae) = return_slot %return.param
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
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%X [concrete = constants.%X]
// CHECK:STDOUT:   impl_decl @X.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@X.as.Destroy.impl.%X.as.Destroy.impl.Op.decl), @X.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.286]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Y1 {
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%Y1 [concrete = constants.%Y1]
// CHECK:STDOUT:   impl_decl @Y1.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@Y1.as.Destroy.impl.%Y1.as.Destroy.impl.Op.decl), @Y1.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.6b0]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Y1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Y2 {
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%Y2 [concrete = constants.%Y2]
// CHECK:STDOUT:   impl_decl @Y2.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@Y2.as.Destroy.impl.%Y2.as.Destroy.impl.Op.decl), @Y2.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.dbf]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Y2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Z {
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%Z [concrete = constants.%Z]
// CHECK:STDOUT:   impl_decl @Z.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@Z.as.Destroy.impl.%Z.as.Destroy.impl.Op.decl), @Z.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.417]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Z
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @A.F(@A.%T.loc5_13.2: type, @A.%Self.1: @A.%A.type (%A.type.806), %U.loc6_8.2: type) {
// CHECK:STDOUT:   %U.loc6_8.1: type = bind_symbolic_name U, 2 [symbolic = %U.loc6_8.1 (constants.%U.2cb)]
// CHECK:STDOUT:   %pattern_type.loc6_18: type = pattern_type %U.loc6_8.1 [symbolic = %pattern_type.loc6_18 (constants.%pattern_type.20f)]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T.8b3)]
// CHECK:STDOUT:   %A.type: type = facet_type <@A, @A(%T)> [symbolic = %A.type (constants.%A.type.806)]
// CHECK:STDOUT:   %Self: @A.F.%A.type (%A.type.806) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.b74)]
// CHECK:STDOUT:   %tuple.type.loc6_38.1: type = tuple_type (type, %A.type, type) [symbolic = %tuple.type.loc6_38.1 (constants.%tuple.type.4c5)]
// CHECK:STDOUT:   %Self.as_type.loc6_38.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc6_38.1 (constants.%Self.as_type.327)]
// CHECK:STDOUT:   %tuple.type.loc6_38.2: type = tuple_type (%T, %Self.as_type.loc6_38.1, %U.loc6_8.1) [symbolic = %tuple.type.loc6_38.2 (constants.%tuple.type.6ec)]
// CHECK:STDOUT:   %pattern_type.loc6_24: type = pattern_type %tuple.type.loc6_38.2 [symbolic = %pattern_type.loc6_24 (constants.%pattern_type.e2e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param: @A.F.%U.loc6_8.1 (%U.2cb)) -> @A.F.%tuple.type.loc6_38.2 (%tuple.type.6ec);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @X.as.Destroy.impl.Op(%self.param: %ptr.d17) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Y1.as.Destroy.impl.Op(%self.param: %ptr.723) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Y2.as.Destroy.impl.Op(%self.param: %ptr.23d) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Z.as.Destroy.impl.Op(%self.param: %ptr.fb6) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @tuple.type.as.A.impl.F(@tuple.type.as.A.impl.%V1.loc14_14.2: type, @tuple.type.as.A.impl.%V2.loc14_25.2: type, @tuple.type.as.A.impl.%W.loc14_36.2: type, %U.loc17_8.2: type) {
// CHECK:STDOUT:   %U.loc17_8.1: type = bind_symbolic_name U, 3 [symbolic = %U.loc17_8.1 (constants.%U.753)]
// CHECK:STDOUT:   %pattern_type.loc17_18: type = pattern_type %U.loc17_8.1 [symbolic = %pattern_type.loc17_18 (constants.%pattern_type.549)]
// CHECK:STDOUT:   %W: type = bind_symbolic_name W, 2 [symbolic = %W (constants.%W)]
// CHECK:STDOUT:   %V1: type = bind_symbolic_name V1, 0 [symbolic = %V1 (constants.%V1)]
// CHECK:STDOUT:   %V2: type = bind_symbolic_name V2, 1 [symbolic = %V2 (constants.%V2)]
// CHECK:STDOUT:   %tuple.type.loc17_42.1: type = tuple_type (%V1, %V2) [symbolic = %tuple.type.loc17_42.1 (constants.%tuple.type.30b)]
// CHECK:STDOUT:   %tuple.type.loc17_42.2: type = tuple_type (%W, %tuple.type.loc17_42.1, %U.loc17_8.1) [symbolic = %tuple.type.loc17_42.2 (constants.%tuple.type.8ae)]
// CHECK:STDOUT:   %pattern_type.loc17_24: type = pattern_type %tuple.type.loc17_42.2 [symbolic = %pattern_type.loc17_24 (constants.%pattern_type.d4a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc17_24: <witness> = require_complete_type %tuple.type.loc17_42.2 [symbolic = %require_complete.loc17_24 (constants.%require_complete.d4d)]
// CHECK:STDOUT:   %require_complete.loc17_19: <witness> = require_complete_type %U.loc17_8.1 [symbolic = %require_complete.loc17_19 (constants.%require_complete.ed4)]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type: type = fn_type @tuple.type.as.A.impl.F, @tuple.type.as.A.impl(%V1, %V2, %W) [symbolic = %tuple.type.as.A.impl.F.type (constants.%tuple.type.as.A.impl.F.type.dac)]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F: @tuple.type.as.A.impl.F.%tuple.type.as.A.impl.F.type (%tuple.type.as.A.impl.F.type.dac) = struct_value () [symbolic = %tuple.type.as.A.impl.F (constants.%tuple.type.as.A.impl.F.9d5)]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.specific_fn.loc18_12.2: <specific function> = specific_function %tuple.type.as.A.impl.F, @tuple.type.as.A.impl.F(%V1, %V2, %W, %U.loc17_8.1) [symbolic = %tuple.type.as.A.impl.F.specific_fn.loc18_12.2 (constants.%tuple.type.as.A.impl.F.specific_fn.cc7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param: @tuple.type.as.A.impl.F.%U.loc17_8.1 (%U.753)) -> %return.param: @tuple.type.as.A.impl.F.%tuple.type.loc17_42.2 (%tuple.type.8ae) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc18: @tuple.type.as.A.impl.F.%tuple.type.as.A.impl.F.type (%tuple.type.as.A.impl.F.type.dac) = specific_constant @tuple.type.as.A.impl.%tuple.type.as.A.impl.F.decl, @tuple.type.as.A.impl(constants.%V1, constants.%V2, constants.%W) [symbolic = %tuple.type.as.A.impl.F (constants.%tuple.type.as.A.impl.F.9d5)]
// CHECK:STDOUT:     %F.ref: @tuple.type.as.A.impl.F.%tuple.type.as.A.impl.F.type (%tuple.type.as.A.impl.F.type.dac) = name_ref F, %.loc18 [symbolic = %tuple.type.as.A.impl.F (constants.%tuple.type.as.A.impl.F.9d5)]
// CHECK:STDOUT:     %U.ref.loc18: type = name_ref U, %U.loc17_8.2 [symbolic = %U.loc17_8.1 (constants.%U.753)]
// CHECK:STDOUT:     %u.ref: @tuple.type.as.A.impl.F.%U.loc17_8.1 (%U.753) = name_ref u, %u
// CHECK:STDOUT:     %tuple.type.as.A.impl.F.specific_fn.loc18_12.1: <specific function> = specific_function %F.ref, @tuple.type.as.A.impl.F(constants.%V1, constants.%V2, constants.%W, constants.%U.753) [symbolic = %tuple.type.as.A.impl.F.specific_fn.loc18_12.2 (constants.%tuple.type.as.A.impl.F.specific_fn.cc7)]
// CHECK:STDOUT:     %.loc17_24: ref @tuple.type.as.A.impl.F.%tuple.type.loc17_42.2 (%tuple.type.8ae) = splice_block %return {}
// CHECK:STDOUT:     %tuple.type.as.A.impl.F.call: init @tuple.type.as.A.impl.F.%tuple.type.loc17_42.2 (%tuple.type.8ae) = call %tuple.type.as.A.impl.F.specific_fn.loc18_12.1(%u.ref) to %.loc17_24
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
// CHECK:STDOUT:   %A.type: type = facet_type <@A, @A(constants.%X)> [concrete = constants.%A.type.515]
// CHECK:STDOUT:   %A.facet: %A.type.515 = facet_value constants.%tuple.type.a46, (constants.%A.impl_witness.0ca) [concrete = constants.%A.facet.2e2]
// CHECK:STDOUT:   %.loc24_23: %A.type.515 = converted %.loc24_14, %A.facet [concrete = constants.%A.facet.2e2]
// CHECK:STDOUT:   %.loc24_31.1: %A.assoc_type.296 = specific_constant @A.%assoc0.loc6_39.1, @A(constants.%X) [concrete = constants.%assoc0.5f6]
// CHECK:STDOUT:   %F.ref: %A.assoc_type.296 = name_ref F, %.loc24_31.1 [concrete = constants.%assoc0.5f6]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc24_23 [concrete = constants.%tuple.type.a46]
// CHECK:STDOUT:   %.loc24_31.2: type = converted %.loc24_23, %as_type [concrete = constants.%tuple.type.a46]
// CHECK:STDOUT:   %impl.elem0: %.9db = impl_witness_access constants.%A.impl_witness.0ca, element0 [concrete = constants.%tuple.type.as.A.impl.F.d69]
// CHECK:STDOUT:   %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z]
// CHECK:STDOUT:   %.loc24_38.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @tuple.type.as.A.impl.F(constants.%Y1, constants.%Y2, constants.%X, constants.%Z) [concrete = constants.%tuple.type.as.A.impl.F.specific_fn.6bf]
// CHECK:STDOUT:   %.loc24_39.1: ref %tuple.type.415 = temporary_storage
// CHECK:STDOUT:   %.loc24_38.2: ref %Z = temporary_storage
// CHECK:STDOUT:   %.loc24_38.3: init %Z = class_init (), %.loc24_38.2 [concrete = constants.%Z.val]
// CHECK:STDOUT:   %.loc24_38.4: ref %Z = temporary %.loc24_38.2, %.loc24_38.3
// CHECK:STDOUT:   %.loc24_38.5: ref %Z = converted %.loc24_38.1, %.loc24_38.4
// CHECK:STDOUT:   %.loc24_38.6: %Z = bind_value %.loc24_38.5
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.call: init %tuple.type.415 = call %specific_fn(%.loc24_38.6) to %.loc24_39.1
// CHECK:STDOUT:   %.loc24_39.2: ref %tuple.type.415 = temporary %.loc24_39.1, %tuple.type.as.A.impl.F.call
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc24_39.2, constants.%T.as.Destroy.impl.Op.f0c
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function constants.%T.as.Destroy.impl.Op.f0c, @T.as.Destroy.impl.Op(constants.%tuple.type.415) [concrete = constants.%T.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc24_39.2, %T.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc24_39: %ptr.ad9 = addr_of %.loc24_39.2
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr.loc24_39)
// CHECK:STDOUT:   %Z.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc24_38.4, constants.%Z.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr.loc24_38: %ptr.fb6 = addr_of %.loc24_38.4
// CHECK:STDOUT:   %Z.as.Destroy.impl.Op.call: init %empty_tuple.type = call %Z.as.Destroy.impl.Op.bound(%addr.loc24_38)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallGeneric(%T.loc27_16.2: %A.type.515) {
// CHECK:STDOUT:   %T.loc27_16.1: %A.type.515 = bind_symbolic_name T, 0 [symbolic = %T.loc27_16.1 (constants.%T.3b6)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc28_4.2: type = facet_access_type %T.loc27_16.1 [symbolic = %T.as_type.loc28_4.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %A.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc27_16.1, @A, @A(constants.%X) [symbolic = %A.lookup_impl_witness (constants.%A.lookup_impl_witness)]
// CHECK:STDOUT:   %A.facet: %A.type.515 = facet_value %T.as_type.loc28_4.2, (%A.lookup_impl_witness) [symbolic = %A.facet (constants.%A.facet.97b)]
// CHECK:STDOUT:   %.loc28_4.3: type = fn_type_with_self_type constants.%A.F.type.13d, %A.facet [symbolic = %.loc28_4.3 (constants.%.7c8)]
// CHECK:STDOUT:   %impl.elem0.loc28_4.2: @CallGeneric.%.loc28_4.3 (%.7c8) = impl_witness_access %A.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc28_4.2 (constants.%impl.elem0.eda)]
// CHECK:STDOUT:   %specific_impl_fn.loc28_4.2: <specific function> = specific_impl_function %impl.elem0.loc28_4.2, @A.F(constants.%X, %A.facet, constants.%Z) [symbolic = %specific_impl_fn.loc28_4.2 (constants.%specific_impl_fn.9ce)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (constants.%X, %T.as_type.loc28_4.2, constants.%Z) [symbolic = %tuple.type (constants.%tuple.type.379)]
// CHECK:STDOUT:   %require_complete.loc28_12.1: <witness> = require_complete_type %tuple.type [symbolic = %require_complete.loc28_12.1 (constants.%require_complete.d0c)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %tuple.type, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %tuple.type, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet (constants.%Destroy.facet.a89)]
// CHECK:STDOUT:   %.loc28_12.5: type = fn_type_with_self_type constants.%Destroy.Op.type, %Destroy.facet [symbolic = %.loc28_12.5 (constants.%.a0b)]
// CHECK:STDOUT:   %impl.elem0.loc28_12.2: @CallGeneric.%.loc28_12.5 (%.a0b) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc28_12.2 (constants.%impl.elem0.80a)]
// CHECK:STDOUT:   %specific_impl_fn.loc28_12.2: <specific function> = specific_impl_function %impl.elem0.loc28_12.2, @Destroy.Op(%Destroy.facet) [symbolic = %specific_impl_fn.loc28_12.2 (constants.%specific_impl_fn.515)]
// CHECK:STDOUT:   %ptr: type = ptr_type %tuple.type [symbolic = %ptr (constants.%ptr.e41)]
// CHECK:STDOUT:   %require_complete.loc28_12.2: <witness> = require_complete_type %ptr [symbolic = %require_complete.loc28_12.2 (constants.%require_complete.1f9)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref: %A.type.515 = name_ref T, %T.loc27_16.2 [symbolic = %T.loc27_16.1 (constants.%T.3b6)]
// CHECK:STDOUT:     %.loc28_4.1: %A.assoc_type.296 = specific_constant @A.%assoc0.loc6_39.1, @A(constants.%X) [concrete = constants.%assoc0.5f6]
// CHECK:STDOUT:     %F.ref: %A.assoc_type.296 = name_ref F, %.loc28_4.1 [concrete = constants.%assoc0.5f6]
// CHECK:STDOUT:     %T.as_type.loc28_4.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc28_4.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc28_4.2: type = converted %T.ref, %T.as_type.loc28_4.1 [symbolic = %T.as_type.loc28_4.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %impl.elem0.loc28_4.1: @CallGeneric.%.loc28_4.3 (%.7c8) = impl_witness_access constants.%A.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc28_4.2 (constants.%impl.elem0.eda)]
// CHECK:STDOUT:     %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z]
// CHECK:STDOUT:     %.loc28_11.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %specific_impl_fn.loc28_4.1: <specific function> = specific_impl_function %impl.elem0.loc28_4.1, @A.F(constants.%X, constants.%A.facet.97b, constants.%Z) [symbolic = %specific_impl_fn.loc28_4.2 (constants.%specific_impl_fn.9ce)]
// CHECK:STDOUT:     %.loc28_12.1: ref @CallGeneric.%tuple.type (%tuple.type.379) = temporary_storage
// CHECK:STDOUT:     %.loc28_11.2: ref %Z = temporary_storage
// CHECK:STDOUT:     %.loc28_11.3: init %Z = class_init (), %.loc28_11.2 [concrete = constants.%Z.val]
// CHECK:STDOUT:     %.loc28_11.4: ref %Z = temporary %.loc28_11.2, %.loc28_11.3
// CHECK:STDOUT:     %.loc28_11.5: ref %Z = converted %.loc28_11.1, %.loc28_11.4
// CHECK:STDOUT:     %.loc28_11.6: %Z = bind_value %.loc28_11.5
// CHECK:STDOUT:     %.loc28_12.2: init @CallGeneric.%tuple.type (%tuple.type.379) = call %specific_impl_fn.loc28_4.1(%.loc28_11.6) to %.loc28_12.1
// CHECK:STDOUT:     %.loc28_12.3: ref @CallGeneric.%tuple.type (%tuple.type.379) = temporary %.loc28_12.1, %.loc28_12.2
// CHECK:STDOUT:     %impl.elem0.loc28_12.1: @CallGeneric.%.loc28_12.5 (%.a0b) = impl_witness_access constants.%Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc28_12.2 (constants.%impl.elem0.80a)]
// CHECK:STDOUT:     %bound_method.loc28_12.1: <bound method> = bound_method %.loc28_12.3, %impl.elem0.loc28_12.1
// CHECK:STDOUT:     %specific_impl_fn.loc28_12.1: <specific function> = specific_impl_function %impl.elem0.loc28_12.1, @Destroy.Op(constants.%Destroy.facet.a89) [symbolic = %specific_impl_fn.loc28_12.2 (constants.%specific_impl_fn.515)]
// CHECK:STDOUT:     %bound_method.loc28_12.2: <bound method> = bound_method %.loc28_12.3, %specific_impl_fn.loc28_12.1
// CHECK:STDOUT:     %addr.loc28_12: @CallGeneric.%ptr (%ptr.e41) = addr_of %.loc28_12.3
// CHECK:STDOUT:     %.loc28_12.4: init %empty_tuple.type = call %bound_method.loc28_12.2(%addr.loc28_12)
// CHECK:STDOUT:     %Z.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc28_11.4, constants.%Z.as.Destroy.impl.Op
// CHECK:STDOUT:     %addr.loc28_11: %ptr.fb6 = addr_of %.loc28_11.4
// CHECK:STDOUT:     %Z.as.Destroy.impl.Op.call: init %empty_tuple.type = call %Z.as.Destroy.impl.Op.bound(%addr.loc28_11)
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
// CHECK:STDOUT:   %A.facet: %A.type.515 = facet_value constants.%tuple.type.a46, (constants.%A.impl_witness.0ca) [concrete = constants.%A.facet.2e2]
// CHECK:STDOUT:   %.loc33_31: %A.type.515 = converted %.loc33_24, %A.facet [concrete = constants.%A.facet.2e2]
// CHECK:STDOUT:   %CallGeneric.specific_fn: <specific function> = specific_function %CallGeneric.ref, @CallGeneric(constants.%A.facet.2e2) [concrete = constants.%CallGeneric.specific_fn]
// CHECK:STDOUT:   %CallGeneric.call: init %empty_tuple.type = call %CallGeneric.specific_fn()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc5_13.1 => constants.%T.8b3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.F(constants.%T.8b3, constants.%Self.b74, constants.%U.2cb) {
// CHECK:STDOUT:   %U.loc6_8.1 => constants.%U.2cb
// CHECK:STDOUT:   %pattern_type.loc6_18 => constants.%pattern_type.20f
// CHECK:STDOUT:   %T => constants.%T.8b3
// CHECK:STDOUT:   %A.type => constants.%A.type.806
// CHECK:STDOUT:   %Self => constants.%Self.b74
// CHECK:STDOUT:   %tuple.type.loc6_38.1 => constants.%tuple.type.4c5
// CHECK:STDOUT:   %Self.as_type.loc6_38.1 => constants.%Self.as_type.327
// CHECK:STDOUT:   %tuple.type.loc6_38.2 => constants.%tuple.type.6ec
// CHECK:STDOUT:   %pattern_type.loc6_24 => constants.%pattern_type.e2e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%W) {
// CHECK:STDOUT:   %T.loc5_13.1 => constants.%W
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.type => constants.%A.type.ffa
// CHECK:STDOUT:   %Self.2 => constants.%Self.0b4
// CHECK:STDOUT:   %A.F.type => constants.%A.F.type.904
// CHECK:STDOUT:   %A.F => constants.%A.F.1ee
// CHECK:STDOUT:   %A.assoc_type => constants.%A.assoc_type.b05
// CHECK:STDOUT:   %assoc0.loc6_39.2 => constants.%assoc0.c82
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @tuple.type.as.A.impl(constants.%V1, constants.%V2, constants.%W) {
// CHECK:STDOUT:   %V1.loc14_14.1 => constants.%V1
// CHECK:STDOUT:   %V2.loc14_25.1 => constants.%V2
// CHECK:STDOUT:   %W.loc14_36.1 => constants.%W
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.30b
// CHECK:STDOUT:   %A.type.loc14_61.1 => constants.%A.type.ffa
// CHECK:STDOUT:   %require_complete => constants.%require_complete.116
// CHECK:STDOUT:   %A.impl_witness => constants.%A.impl_witness.3ab
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type => constants.%tuple.type.as.A.impl.F.type.dac
// CHECK:STDOUT:   %tuple.type.as.A.impl.F => constants.%tuple.type.as.A.impl.F.9d5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @tuple.type.as.A.impl.F(constants.%V1, constants.%V2, constants.%W, constants.%U.753) {
// CHECK:STDOUT:   %U.loc17_8.1 => constants.%U.753
// CHECK:STDOUT:   %pattern_type.loc17_18 => constants.%pattern_type.549
// CHECK:STDOUT:   %W => constants.%W
// CHECK:STDOUT:   %V1 => constants.%V1
// CHECK:STDOUT:   %V2 => constants.%V2
// CHECK:STDOUT:   %tuple.type.loc17_42.1 => constants.%tuple.type.30b
// CHECK:STDOUT:   %tuple.type.loc17_42.2 => constants.%tuple.type.8ae
// CHECK:STDOUT:   %pattern_type.loc17_24 => constants.%pattern_type.d4a
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc17_24 => constants.%require_complete.d4d
// CHECK:STDOUT:   %require_complete.loc17_19 => constants.%require_complete.ed4
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type => constants.%tuple.type.as.A.impl.F.type.dac
// CHECK:STDOUT:   %tuple.type.as.A.impl.F => constants.%tuple.type.as.A.impl.F.9d5
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.specific_fn.loc18_12.2 => constants.%tuple.type.as.A.impl.F.specific_fn.cc7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.F(constants.%W, constants.%A.facet.b6a, constants.%U.753) {
// CHECK:STDOUT:   %U.loc6_8.1 => constants.%U.753
// CHECK:STDOUT:   %pattern_type.loc6_18 => constants.%pattern_type.549
// CHECK:STDOUT:   %T => constants.%W
// CHECK:STDOUT:   %A.type => constants.%A.type.ffa
// CHECK:STDOUT:   %Self => constants.%A.facet.b6a
// CHECK:STDOUT:   %tuple.type.loc6_38.1 => constants.%tuple.type.6bd
// CHECK:STDOUT:   %Self.as_type.loc6_38.1 => constants.%tuple.type.30b
// CHECK:STDOUT:   %tuple.type.loc6_38.2 => constants.%tuple.type.8ae
// CHECK:STDOUT:   %pattern_type.loc6_24 => constants.%pattern_type.d4a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%X) {
// CHECK:STDOUT:   %T.loc5_13.1 => constants.%X
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.type => constants.%A.type.515
// CHECK:STDOUT:   %Self.2 => constants.%Self.522
// CHECK:STDOUT:   %A.F.type => constants.%A.F.type.13d
// CHECK:STDOUT:   %A.F => constants.%A.F.d83
// CHECK:STDOUT:   %A.assoc_type => constants.%A.assoc_type.296
// CHECK:STDOUT:   %assoc0.loc6_39.2 => constants.%assoc0.5f6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @tuple.type.as.A.impl(constants.%Y1, constants.%Y2, constants.%X) {
// CHECK:STDOUT:   %V1.loc14_14.1 => constants.%Y1
// CHECK:STDOUT:   %V2.loc14_25.1 => constants.%Y2
// CHECK:STDOUT:   %W.loc14_36.1 => constants.%X
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.a46
// CHECK:STDOUT:   %A.type.loc14_61.1 => constants.%A.type.515
// CHECK:STDOUT:   %require_complete => constants.%complete_type.1a4
// CHECK:STDOUT:   %A.impl_witness => constants.%A.impl_witness.0ca
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type => constants.%tuple.type.as.A.impl.F.type.490
// CHECK:STDOUT:   %tuple.type.as.A.impl.F => constants.%tuple.type.as.A.impl.F.d69
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
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type => constants.%tuple.type.as.A.impl.F.type.490
// CHECK:STDOUT:   %tuple.type.as.A.impl.F => constants.%tuple.type.as.A.impl.F.d69
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.specific_fn.loc18_12.2 => constants.%tuple.type.as.A.impl.F.specific_fn.6bf
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGeneric(constants.%T.3b6) {
// CHECK:STDOUT:   %T.loc27_16.1 => constants.%T.3b6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.F(constants.%X, constants.%A.facet.97b, constants.%Z) {
// CHECK:STDOUT:   %U.loc6_8.1 => constants.%Z
// CHECK:STDOUT:   %pattern_type.loc6_18 => constants.%pattern_type.8f9
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.515
// CHECK:STDOUT:   %Self => constants.%A.facet.97b
// CHECK:STDOUT:   %tuple.type.loc6_38.1 => constants.%tuple.type.281
// CHECK:STDOUT:   %Self.as_type.loc6_38.1 => constants.%T.as_type
// CHECK:STDOUT:   %tuple.type.loc6_38.2 => constants.%tuple.type.379
// CHECK:STDOUT:   %pattern_type.loc6_24 => constants.%pattern_type.fcb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGeneric(constants.%A.facet.2e2) {
// CHECK:STDOUT:   %T.loc27_16.1 => constants.%A.facet.2e2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc28_4.2 => constants.%tuple.type.a46
// CHECK:STDOUT:   %A.lookup_impl_witness => constants.%A.impl_witness.0ca
// CHECK:STDOUT:   %A.facet => constants.%A.facet.2e2
// CHECK:STDOUT:   %.loc28_4.3 => constants.%.9db
// CHECK:STDOUT:   %impl.elem0.loc28_4.2 => constants.%tuple.type.as.A.impl.F.d69
// CHECK:STDOUT:   %specific_impl_fn.loc28_4.2 => constants.%tuple.type.as.A.impl.F.specific_fn.6bf
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.415
// CHECK:STDOUT:   %require_complete.loc28_12.1 => constants.%complete_type.aa8
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%Destroy.impl_witness.c7e
// CHECK:STDOUT:   %Destroy.facet => constants.%Destroy.facet.c14
// CHECK:STDOUT:   %.loc28_12.5 => constants.%.397
// CHECK:STDOUT:   %impl.elem0.loc28_12.2 => constants.%T.as.Destroy.impl.Op.f0c
// CHECK:STDOUT:   %specific_impl_fn.loc28_12.2 => constants.%T.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %ptr => constants.%ptr.ad9
// CHECK:STDOUT:   %require_complete.loc28_12.2 => constants.%complete_type.e23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.F(constants.%X, constants.%A.facet.2e2, constants.%Z) {
// CHECK:STDOUT:   %U.loc6_8.1 => constants.%Z
// CHECK:STDOUT:   %pattern_type.loc6_18 => constants.%pattern_type.8f9
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.515
// CHECK:STDOUT:   %Self => constants.%A.facet.2e2
// CHECK:STDOUT:   %tuple.type.loc6_38.1 => constants.%tuple.type.281
// CHECK:STDOUT:   %Self.as_type.loc6_38.1 => constants.%tuple.type.a46
// CHECK:STDOUT:   %tuple.type.loc6_38.2 => constants.%tuple.type.415
// CHECK:STDOUT:   %pattern_type.loc6_24 => constants.%pattern_type.0b2
// CHECK:STDOUT: }
// CHECK:STDOUT:

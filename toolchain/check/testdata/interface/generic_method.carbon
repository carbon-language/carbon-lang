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
// CHECK:STDOUT:   %A.type.d88: type = facet_type <@A, @A(%T.8b3)> [symbolic]
// CHECK:STDOUT:   %Self.c6b: %A.type.d88 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %U.2cb: type = bind_symbolic_name U, 2 [symbolic]
// CHECK:STDOUT:   %ptr.204: type = ptr_type %U.2cb [symbolic]
// CHECK:STDOUT:   %pattern_type.eee: type = pattern_type %ptr.204 [symbolic]
// CHECK:STDOUT:   %tuple.type.098: type = tuple_type (type, %A.type.d88, type) [symbolic]
// CHECK:STDOUT:   %Self.as_type.707: type = facet_access_type %Self.c6b [symbolic]
// CHECK:STDOUT:   %tuple.type.cb0: type = tuple_type (%T.8b3, %Self.as_type.707, %ptr.204) [symbolic]
// CHECK:STDOUT:   %pattern_type.44d: type = pattern_type %tuple.type.cb0 [symbolic]
// CHECK:STDOUT:   %A.F.type.17a: type = fn_type @A.F, @A(%T.8b3) [symbolic]
// CHECK:STDOUT:   %A.F.0d8: %A.F.type.17a = struct_value () [symbolic]
// CHECK:STDOUT:   %A.assoc_type.ed3: type = assoc_entity_type @A, @A(%T.8b3) [symbolic]
// CHECK:STDOUT:   %assoc0.fcb: %A.assoc_type.ed3 = assoc_entity element0, @A.%A.F.decl [symbolic]
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Y: type = class_type @Y [concrete]
// CHECK:STDOUT:   %Z: type = class_type @Z [concrete]
// CHECK:STDOUT:   %A.type.0a4: type = facet_type <@A, @A(%X)> [concrete]
// CHECK:STDOUT:   %Self.08e: %A.type.0a4 = bind_symbolic_name Self, 1 [symbolic]
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
// CHECK:STDOUT:   %A.facet: %A.type.0a4 = facet_value %Y, (%A.impl_witness) [concrete]
// CHECK:STDOUT:   %tuple.type.5a1: type = tuple_type (type, %A.type.0a4, type) [concrete]
// CHECK:STDOUT:   %require_complete.86f: <witness> = require_complete_type %tuple.type.c14 [symbolic]
// CHECK:STDOUT:   %require_complete.6e5e64.1: <witness> = require_complete_type %ptr.79f131.1 [symbolic]
// CHECK:STDOUT:   %tuple.type.e44: type = tuple_type (%empty_struct_type, %empty_struct_type, %ptr.79f131.1) [symbolic]
// CHECK:STDOUT:   %X.val: %X = struct_value () [concrete]
// CHECK:STDOUT:   %Y.val: %Y = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.31ff4e.1: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.8b3) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.8a8b1a.1: %ptr.as.Copy.impl.Op.type.31ff4e.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.cb2: <witness> = lookup_impl_witness %ptr.79f131.1, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.facet.2d1: %Copy.type = facet_value %ptr.79f131.1, (%Copy.lookup_impl_witness.cb2) [symbolic]
// CHECK:STDOUT:   %.1cc: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.2d1 [symbolic]
// CHECK:STDOUT:   %impl.elem0.751: %.1cc = impl_witness_access %Copy.lookup_impl_witness.cb2, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.f44: <specific function> = specific_impl_function %impl.elem0.751, @Copy.Op(%Copy.facet.2d1) [symbolic]
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
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanAggregateDestroy>> [concrete]
// CHECK:STDOUT:   %AggregateT: %type_where = bind_symbolic_name AggregateT, 0 [symbolic]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.type.190: type = fn_type @AggregateT.as_type.as.Destroy.impl.Op, @AggregateT.as_type.as.Destroy.impl(%AggregateT) [symbolic]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.8a0: %AggregateT.as_type.as.Destroy.impl.Op.type.190 = struct_value () [symbolic]
// CHECK:STDOUT:   %facet_value.352: %type_where = facet_value %tuple.type.092, () [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.89a: <witness> = impl_witness imports.%Destroy.impl_witness_table.2d3, @AggregateT.as_type.as.Destroy.impl(%facet_value.352) [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.type.ec5: type = fn_type @AggregateT.as_type.as.Destroy.impl.Op, @AggregateT.as_type.as.Destroy.impl(%facet_value.352) [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.ed3: %AggregateT.as_type.as.Destroy.impl.Op.type.ec5 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.5df: type = ptr_type %tuple.type.092 [concrete]
// CHECK:STDOUT:   %complete_type.aab: <witness> = complete_type_witness %ptr.5df [concrete]
// CHECK:STDOUT:   %Destroy.facet.d0e: %Destroy.type = facet_value %tuple.type.092, (%Destroy.impl_witness.89a) [concrete]
// CHECK:STDOUT:   %.f61: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet.d0e [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.specific_fn.71f: <specific function> = specific_function %AggregateT.as_type.as.Destroy.impl.Op.ed3, @AggregateT.as_type.as.Destroy.impl.Op(%facet_value.352) [concrete]
// CHECK:STDOUT:   %facet_value.4e3: %type_where = facet_value %Z, () [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.type.18a: type = fn_type @AggregateT.as_type.as.Destroy.impl.Op, @AggregateT.as_type.as.Destroy.impl(%facet_value.4e3) [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.dd5: %AggregateT.as_type.as.Destroy.impl.Op.type.18a = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type.d3e: <witness> = complete_type_witness %ptr.fb6 [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.specific_fn.e7d: <specific function> = specific_function %AggregateT.as_type.as.Destroy.impl.Op.dd5, @AggregateT.as_type.as.Destroy.impl.Op(%facet_value.4e3) [concrete]
// CHECK:STDOUT:   %T.7bf: %A.type.0a4 = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.809: type = pattern_type %A.type.0a4 [concrete]
// CHECK:STDOUT:   %CallGeneric.type: type = fn_type @CallGeneric [concrete]
// CHECK:STDOUT:   %CallGeneric: %CallGeneric.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.b3a: type = facet_access_type %T.7bf [symbolic]
// CHECK:STDOUT:   %A.lookup_impl_witness: <witness> = lookup_impl_witness %T.7bf, @A, @A(%X) [symbolic]
// CHECK:STDOUT:   %.0c6: type = fn_type_with_self_type %A.F.type.13d, %T.7bf [symbolic]
// CHECK:STDOUT:   %impl.elem0.dd2: %.0c6 = impl_witness_access %A.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %tuple.type.af6: type = tuple_type (%X, %T.as_type.b3a, %ptr.fb6) [symbolic]
// CHECK:STDOUT:   %pattern_type.512: type = pattern_type %tuple.type.af6 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.304: <specific function> = specific_impl_function %impl.elem0.dd2, @A.F(%X, %T.7bf, %Z) [symbolic]
// CHECK:STDOUT:   %require_complete.6ed: <witness> = require_complete_type %tuple.type.af6 [symbolic]
// CHECK:STDOUT:   %facet_value.caf: %type_where = facet_value %tuple.type.af6, () [symbolic]
// CHECK:STDOUT:   %ptr.fd2: type = ptr_type %tuple.type.af6 [symbolic]
// CHECK:STDOUT:   %require_complete.b2a: <witness> = require_complete_type %ptr.fd2 [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.47b: <witness> = lookup_impl_witness %tuple.type.af6, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.aef: %Destroy.type = facet_value %tuple.type.af6, (%Destroy.lookup_impl_witness.47b) [symbolic]
// CHECK:STDOUT:   %.650: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet.aef [symbolic]
// CHECK:STDOUT:   %impl.elem0.142: %.650 = impl_witness_access %Destroy.lookup_impl_witness.47b, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.fc8: <specific function> = specific_impl_function %impl.elem0.142, @Destroy.Op(%Destroy.facet.aef) [symbolic]
// CHECK:STDOUT:   %CallIndirect.type: type = fn_type @CallIndirect [concrete]
// CHECK:STDOUT:   %CallIndirect: %CallIndirect.type = struct_value () [concrete]
// CHECK:STDOUT:   %CallGeneric.specific_fn: <specific function> = specific_function %CallGeneric, @CallGeneric(%A.facet) [concrete]
// CHECK:STDOUT:   %complete_type.05d: <witness> = complete_type_witness %tuple.type.092 [concrete]
// CHECK:STDOUT:   %tuple.type.953: type = tuple_type (%empty_struct_type, %empty_struct_type, %ptr.fb6) [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.4b2: <witness> = impl_witness imports.%Copy.impl_witness_table.53c, @ptr.as.Copy.impl(%Z) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.8a3: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%Z) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.0c5: %ptr.as.Copy.impl.Op.type.8a3 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.cd8: %Copy.type = facet_value %ptr.fb6, (%Copy.impl_witness.4b2) [concrete]
// CHECK:STDOUT:   %.7f9: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.cd8 [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.0c5, @ptr.as.Copy.impl.Op(%Z) [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.0e4: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.31ff4e.1) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.8a8b1a.1)]
// CHECK:STDOUT:   %Copy.impl_witness_table.53c = impl_witness_table (%Core.import_ref.0e4), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.d51: @AggregateT.as_type.as.Destroy.impl.%AggregateT.as_type.as.Destroy.impl.Op.type (%AggregateT.as_type.as.Destroy.impl.Op.type.190) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @AggregateT.as_type.as.Destroy.impl.%AggregateT.as_type.as.Destroy.impl.Op (constants.%AggregateT.as_type.as.Destroy.impl.Op.8a0)]
// CHECK:STDOUT:   %Destroy.impl_witness_table.2d3 = impl_witness_table (%Core.import_ref.d51), @AggregateT.as_type.as.Destroy.impl [concrete]
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
// CHECK:STDOUT:     %A.type: type = facet_type <@A, @A(constants.%X)> [concrete = constants.%A.type.0a4]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.impl_witness_table = impl_witness_table (@Y.as.A.impl.%Y.as.A.impl.F.decl), @Y.as.A.impl [concrete]
// CHECK:STDOUT:   %A.impl_witness: <witness> = impl_witness %A.impl_witness_table [concrete = constants.%A.impl_witness]
// CHECK:STDOUT:   %Call.decl: %Call.type = fn_decl @Call [concrete = constants.%Call] {} {}
// CHECK:STDOUT:   %CallGeneric.decl: %CallGeneric.type = fn_decl @CallGeneric [concrete = constants.%CallGeneric] {
// CHECK:STDOUT:     %T.patt: %pattern_type.809 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc26: type = splice_block %A.type [concrete = constants.%A.type.0a4] {
// CHECK:STDOUT:       %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %A.ref: %A.type.495 = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:       %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:       %A.type: type = facet_type <@A, @A(constants.%X)> [concrete = constants.%A.type.0a4]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc26_16.2: %A.type.0a4 = bind_symbolic_name T, 0 [symbolic = %T.loc26_16.1 (constants.%T.7bf)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallIndirect.decl: %CallIndirect.type = fn_decl @CallIndirect [concrete = constants.%CallIndirect] {} {}
// CHECK:STDOUT:   %facet_value: %type_where = facet_value constants.%tuple.type.092, () [concrete = constants.%facet_value.352]
// CHECK:STDOUT:   %.loc28: %type_where = converted constants.%tuple.type.092, %facet_value [concrete = constants.%facet_value.352]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @A(%T.loc5_13.2: type) {
// CHECK:STDOUT:   %T.loc5_13.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc5_13.1 (constants.%T.8b3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.type: type = facet_type <@A, @A(%T.loc5_13.1)> [symbolic = %A.type (constants.%A.type.d88)]
// CHECK:STDOUT:   %Self.2: @A.%A.type (%A.type.d88) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.c6b)]
// CHECK:STDOUT:   %A.F.type: type = fn_type @A.F, @A(%T.loc5_13.1) [symbolic = %A.F.type (constants.%A.F.type.17a)]
// CHECK:STDOUT:   %A.F: @A.%A.F.type (%A.F.type.17a) = struct_value () [symbolic = %A.F (constants.%A.F.0d8)]
// CHECK:STDOUT:   %A.assoc_type: type = assoc_entity_type @A, @A(%T.loc5_13.1) [symbolic = %A.assoc_type (constants.%A.assoc_type.ed3)]
// CHECK:STDOUT:   %assoc0.loc6_41.2: @A.%A.assoc_type (%A.assoc_type.ed3) = assoc_entity element0, %A.F.decl [symbolic = %assoc0.loc6_41.2 (constants.%assoc0.fcb)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @A.%A.type (%A.type.d88) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.c6b)]
// CHECK:STDOUT:     %A.F.decl: @A.%A.F.type (%A.F.type.17a) = fn_decl @A.F [symbolic = @A.%A.F (constants.%A.F.0d8)] {
// CHECK:STDOUT:       %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 2 [concrete]
// CHECK:STDOUT:       %u.patt: @A.F.%pattern_type.loc6_18 (%pattern_type.eee) = binding_pattern u [concrete]
// CHECK:STDOUT:       %u.param_patt: @A.F.%pattern_type.loc6_18 (%pattern_type.eee) = value_param_pattern %u.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @A.F.%pattern_type.loc6_25 (%pattern_type.44d) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @A.F.%pattern_type.loc6_25 (%pattern_type.44d) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @A.%T.loc5_13.2 [symbolic = %T (constants.%T.8b3)]
// CHECK:STDOUT:       %.loc6_32: @A.F.%A.type (%A.type.d88) = specific_constant @A.%Self.1, @A(constants.%T.8b3) [symbolic = %Self (constants.%Self.c6b)]
// CHECK:STDOUT:       %Self.ref: @A.F.%A.type (%A.type.d88) = name_ref Self, %.loc6_32 [symbolic = %Self (constants.%Self.c6b)]
// CHECK:STDOUT:       %U.ref.loc6_38: type = name_ref U, %U.loc6_8.2 [symbolic = %U.loc6_8.1 (constants.%U.2cb)]
// CHECK:STDOUT:       %ptr.loc6_39: type = ptr_type %U.ref.loc6_38 [symbolic = %ptr.loc6_22.1 (constants.%ptr.204)]
// CHECK:STDOUT:       %.loc6_40.1: @A.F.%tuple.type.loc6_40.1 (%tuple.type.098) = tuple_literal (%T.ref, %Self.ref, %ptr.loc6_39)
// CHECK:STDOUT:       %Self.as_type.loc6_40.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc6_40.1 (constants.%Self.as_type.707)]
// CHECK:STDOUT:       %.loc6_40.2: type = converted %Self.ref, %Self.as_type.loc6_40.2 [symbolic = %Self.as_type.loc6_40.1 (constants.%Self.as_type.707)]
// CHECK:STDOUT:       %.loc6_40.3: type = converted %.loc6_40.1, constants.%tuple.type.cb0 [symbolic = %tuple.type.loc6_40.2 (constants.%tuple.type.cb0)]
// CHECK:STDOUT:       %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %U.loc6_8.2: type = bind_symbolic_name U, 2 [symbolic = %U.loc6_8.1 (constants.%U.2cb)]
// CHECK:STDOUT:       %u.param: @A.F.%ptr.loc6_22.1 (%ptr.204) = value_param call_param0
// CHECK:STDOUT:       %.loc6_22: type = splice_block %ptr.loc6_22.2 [symbolic = %ptr.loc6_22.1 (constants.%ptr.204)] {
// CHECK:STDOUT:         %U.ref.loc6_21: type = name_ref U, %U.loc6_8.2 [symbolic = %U.loc6_8.1 (constants.%U.2cb)]
// CHECK:STDOUT:         %ptr.loc6_22.2: type = ptr_type %U.ref.loc6_21 [symbolic = %ptr.loc6_22.1 (constants.%ptr.204)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %u: @A.F.%ptr.loc6_22.1 (%ptr.204) = bind_name u, %u.param
// CHECK:STDOUT:       %return.param: ref @A.F.%tuple.type.loc6_40.2 (%tuple.type.cb0) = out_param call_param1
// CHECK:STDOUT:       %return: ref @A.F.%tuple.type.loc6_40.2 (%tuple.type.cb0) = return_slot %return.param
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
// CHECK:STDOUT: generic fn @A.F(@A.%T.loc5_13.2: type, @A.%Self.1: @A.%A.type (%A.type.d88), %U.loc6_8.2: type) {
// CHECK:STDOUT:   %U.loc6_8.1: type = bind_symbolic_name U, 2 [symbolic = %U.loc6_8.1 (constants.%U.2cb)]
// CHECK:STDOUT:   %ptr.loc6_22.1: type = ptr_type %U.loc6_8.1 [symbolic = %ptr.loc6_22.1 (constants.%ptr.204)]
// CHECK:STDOUT:   %pattern_type.loc6_18: type = pattern_type %ptr.loc6_22.1 [symbolic = %pattern_type.loc6_18 (constants.%pattern_type.eee)]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T.8b3)]
// CHECK:STDOUT:   %A.type: type = facet_type <@A, @A(%T)> [symbolic = %A.type (constants.%A.type.d88)]
// CHECK:STDOUT:   %Self: @A.F.%A.type (%A.type.d88) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.c6b)]
// CHECK:STDOUT:   %tuple.type.loc6_40.1: type = tuple_type (type, %A.type, type) [symbolic = %tuple.type.loc6_40.1 (constants.%tuple.type.098)]
// CHECK:STDOUT:   %Self.as_type.loc6_40.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc6_40.1 (constants.%Self.as_type.707)]
// CHECK:STDOUT:   %tuple.type.loc6_40.2: type = tuple_type (%T, %Self.as_type.loc6_40.1, %ptr.loc6_22.1) [symbolic = %tuple.type.loc6_40.2 (constants.%tuple.type.cb0)]
// CHECK:STDOUT:   %pattern_type.loc6_25: type = pattern_type %tuple.type.loc6_40.2 [symbolic = %pattern_type.loc6_25 (constants.%pattern_type.44d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param: @A.F.%ptr.loc6_22.1 (%ptr.204)) -> @A.F.%tuple.type.loc6_40.2 (%tuple.type.cb0);
// CHECK:STDOUT: }
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
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.loc16_22.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet.2d1)]
// CHECK:STDOUT:   %.loc17_21.2: type = fn_type_with_self_type constants.%Copy.Op.type, %Copy.facet [symbolic = %.loc17_21.2 (constants.%.1cc)]
// CHECK:STDOUT:   %impl.elem0.loc17_21.2: @Y.as.A.impl.F.%.loc17_21.2 (%.1cc) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc17_21.2 (constants.%impl.elem0.751)]
// CHECK:STDOUT:   %specific_impl_fn.loc17_21.2: <specific function> = specific_impl_function %impl.elem0.loc17_21.2, @Copy.Op(%Copy.facet) [symbolic = %specific_impl_fn.loc17_21.2 (constants.%specific_impl_fn.f44)]
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
// CHECK:STDOUT:     %impl.elem0.loc17_21.1: @Y.as.A.impl.F.%.loc17_21.2 (%.1cc) = impl_witness_access constants.%Copy.lookup_impl_witness.cb2, element0 [symbolic = %impl.elem0.loc17_21.2 (constants.%impl.elem0.751)]
// CHECK:STDOUT:     %bound_method.loc17_21.1: <bound method> = bound_method %u.ref, %impl.elem0.loc17_21.1
// CHECK:STDOUT:     %specific_impl_fn.loc17_21.1: <specific function> = specific_impl_function %impl.elem0.loc17_21.1, @Copy.Op(constants.%Copy.facet.2d1) [symbolic = %specific_impl_fn.loc17_21.2 (constants.%specific_impl_fn.f44)]
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
// CHECK:STDOUT:   %A.type: type = facet_type <@A, @A(constants.%X)> [concrete = constants.%A.type.0a4]
// CHECK:STDOUT:   %A.facet: %A.type.0a4 = facet_value %Y.ref, (constants.%A.impl_witness) [concrete = constants.%A.facet]
// CHECK:STDOUT:   %.loc23_6: %A.type.0a4 = converted %Y.ref, %A.facet [concrete = constants.%A.facet]
// CHECK:STDOUT:   %.loc23_14.1: %A.assoc_type.296 = specific_constant @A.%assoc0.loc6_41.1, @A(constants.%X) [concrete = constants.%assoc0.5f6]
// CHECK:STDOUT:   %F.ref: %A.assoc_type.296 = name_ref F, %.loc23_14.1 [concrete = constants.%assoc0.5f6]
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
// CHECK:STDOUT:   %facet_value.loc23: %type_where = facet_value constants.%tuple.type.092, () [concrete = constants.%facet_value.352]
// CHECK:STDOUT:   %.loc23_22.3: %type_where = converted constants.%tuple.type.092, %facet_value.loc23 [concrete = constants.%facet_value.352]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.bound.loc23: <bound method> = bound_method %.loc23_22.2, constants.%AggregateT.as_type.as.Destroy.impl.Op.ed3
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.specific_fn.1: <specific function> = specific_function constants.%AggregateT.as_type.as.Destroy.impl.Op.ed3, @AggregateT.as_type.as.Destroy.impl.Op(constants.%facet_value.352) [concrete = constants.%AggregateT.as_type.as.Destroy.impl.Op.specific_fn.71f]
// CHECK:STDOUT:   %bound_method.loc23: <bound method> = bound_method %.loc23_22.2, %AggregateT.as_type.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc23_22: %ptr.5df = addr_of %.loc23_22.2
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.call.loc23: init %empty_tuple.type = call %bound_method.loc23(%addr.loc23_22)
// CHECK:STDOUT:   %facet_value.loc22: %type_where = facet_value constants.%Z, () [concrete = constants.%facet_value.4e3]
// CHECK:STDOUT:   %.loc22: %type_where = converted constants.%Z, %facet_value.loc22 [concrete = constants.%facet_value.4e3]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.bound.loc22: <bound method> = bound_method %u.var, constants.%AggregateT.as_type.as.Destroy.impl.Op.dd5
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.specific_fn.2: <specific function> = specific_function constants.%AggregateT.as_type.as.Destroy.impl.Op.dd5, @AggregateT.as_type.as.Destroy.impl.Op(constants.%facet_value.4e3) [concrete = constants.%AggregateT.as_type.as.Destroy.impl.Op.specific_fn.e7d]
// CHECK:STDOUT:   %bound_method.loc22: <bound method> = bound_method %u.var, %AggregateT.as_type.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc22: %ptr.fb6 = addr_of %u.var
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.call.loc22: init %empty_tuple.type = call %bound_method.loc22(%addr.loc22)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallGeneric(%T.loc26_16.2: %A.type.0a4) {
// CHECK:STDOUT:   %T.loc26_16.1: %A.type.0a4 = bind_symbolic_name T, 0 [symbolic = %T.loc26_16.1 (constants.%T.7bf)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc28_4.2: type = facet_access_type %T.loc26_16.1 [symbolic = %T.as_type.loc28_4.2 (constants.%T.as_type.b3a)]
// CHECK:STDOUT:   %A.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc26_16.1, @A, @A(constants.%X) [symbolic = %A.lookup_impl_witness (constants.%A.lookup_impl_witness)]
// CHECK:STDOUT:   %.loc28_4.3: type = fn_type_with_self_type constants.%A.F.type.13d, %T.loc26_16.1 [symbolic = %.loc28_4.3 (constants.%.0c6)]
// CHECK:STDOUT:   %impl.elem0.loc28_4.2: @CallGeneric.%.loc28_4.3 (%.0c6) = impl_witness_access %A.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc28_4.2 (constants.%impl.elem0.dd2)]
// CHECK:STDOUT:   %specific_impl_fn.loc28_4.2: <specific function> = specific_impl_function %impl.elem0.loc28_4.2, @A.F(constants.%X, %T.loc26_16.1, constants.%Z) [symbolic = %specific_impl_fn.loc28_4.2 (constants.%specific_impl_fn.304)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (constants.%X, %T.as_type.loc28_4.2, constants.%ptr.fb6) [symbolic = %tuple.type (constants.%tuple.type.af6)]
// CHECK:STDOUT:   %require_complete.loc28_12.1: <witness> = require_complete_type %tuple.type [symbolic = %require_complete.loc28_12.1 (constants.%require_complete.6ed)]
// CHECK:STDOUT:   %facet_value.loc28_12.2: %type_where = facet_value %tuple.type, () [symbolic = %facet_value.loc28_12.2 (constants.%facet_value.caf)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %tuple.type, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.47b)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %tuple.type, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet (constants.%Destroy.facet.aef)]
// CHECK:STDOUT:   %.loc28_12.6: type = fn_type_with_self_type constants.%Destroy.Op.type, %Destroy.facet [symbolic = %.loc28_12.6 (constants.%.650)]
// CHECK:STDOUT:   %impl.elem0.loc28_12.2: @CallGeneric.%.loc28_12.6 (%.650) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc28_12.2 (constants.%impl.elem0.142)]
// CHECK:STDOUT:   %specific_impl_fn.loc28_12.2: <specific function> = specific_impl_function %impl.elem0.loc28_12.2, @Destroy.Op(%Destroy.facet) [symbolic = %specific_impl_fn.loc28_12.2 (constants.%specific_impl_fn.fc8)]
// CHECK:STDOUT:   %ptr: type = ptr_type %tuple.type [symbolic = %ptr (constants.%ptr.fd2)]
// CHECK:STDOUT:   %require_complete.loc28_12.2: <witness> = require_complete_type %ptr [symbolic = %require_complete.loc28_12.2 (constants.%require_complete.b2a)]
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
// CHECK:STDOUT:     %T.ref: %A.type.0a4 = name_ref T, %T.loc26_16.2 [symbolic = %T.loc26_16.1 (constants.%T.7bf)]
// CHECK:STDOUT:     %.loc28_4.1: %A.assoc_type.296 = specific_constant @A.%assoc0.loc6_41.1, @A(constants.%X) [concrete = constants.%assoc0.5f6]
// CHECK:STDOUT:     %F.ref: %A.assoc_type.296 = name_ref F, %.loc28_4.1 [concrete = constants.%assoc0.5f6]
// CHECK:STDOUT:     %T.as_type.loc28_4.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc28_4.2 (constants.%T.as_type.b3a)]
// CHECK:STDOUT:     %.loc28_4.2: type = converted %T.ref, %T.as_type.loc28_4.1 [symbolic = %T.as_type.loc28_4.2 (constants.%T.as_type.b3a)]
// CHECK:STDOUT:     %impl.elem0.loc28_4.1: @CallGeneric.%.loc28_4.3 (%.0c6) = impl_witness_access constants.%A.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc28_4.2 (constants.%impl.elem0.dd2)]
// CHECK:STDOUT:     %Z.ref.loc28: type = name_ref Z, file.%Z.decl [concrete = constants.%Z]
// CHECK:STDOUT:     %u.ref: ref %Z = name_ref u, %u
// CHECK:STDOUT:     %addr.loc28_10: %ptr.fb6 = addr_of %u.ref
// CHECK:STDOUT:     %specific_impl_fn.loc28_4.1: <specific function> = specific_impl_function %impl.elem0.loc28_4.1, @A.F(constants.%X, constants.%T.7bf, constants.%Z) [symbolic = %specific_impl_fn.loc28_4.2 (constants.%specific_impl_fn.304)]
// CHECK:STDOUT:     %.loc28_12.1: ref @CallGeneric.%tuple.type (%tuple.type.af6) = temporary_storage
// CHECK:STDOUT:     %.loc28_12.2: init @CallGeneric.%tuple.type (%tuple.type.af6) = call %specific_impl_fn.loc28_4.1(%addr.loc28_10) to %.loc28_12.1
// CHECK:STDOUT:     %.loc28_12.3: ref @CallGeneric.%tuple.type (%tuple.type.af6) = temporary %.loc28_12.1, %.loc28_12.2
// CHECK:STDOUT:     %facet_value.loc28_12.1: %type_where = facet_value constants.%tuple.type.af6, () [symbolic = %facet_value.loc28_12.2 (constants.%facet_value.caf)]
// CHECK:STDOUT:     %.loc28_12.4: %type_where = converted constants.%tuple.type.af6, %facet_value.loc28_12.1 [symbolic = %facet_value.loc28_12.2 (constants.%facet_value.caf)]
// CHECK:STDOUT:     %impl.elem0.loc28_12.1: @CallGeneric.%.loc28_12.6 (%.650) = impl_witness_access constants.%Destroy.lookup_impl_witness.47b, element0 [symbolic = %impl.elem0.loc28_12.2 (constants.%impl.elem0.142)]
// CHECK:STDOUT:     %bound_method.loc28_12.1: <bound method> = bound_method %.loc28_12.3, %impl.elem0.loc28_12.1
// CHECK:STDOUT:     %specific_impl_fn.loc28_12.1: <specific function> = specific_impl_function %impl.elem0.loc28_12.1, @Destroy.Op(constants.%Destroy.facet.aef) [symbolic = %specific_impl_fn.loc28_12.2 (constants.%specific_impl_fn.fc8)]
// CHECK:STDOUT:     %bound_method.loc28_12.2: <bound method> = bound_method %.loc28_12.3, %specific_impl_fn.loc28_12.1
// CHECK:STDOUT:     %addr.loc28_12: @CallGeneric.%ptr (%ptr.fd2) = addr_of %.loc28_12.3
// CHECK:STDOUT:     %.loc28_12.5: init %empty_tuple.type = call %bound_method.loc28_12.2(%addr.loc28_12)
// CHECK:STDOUT:     %facet_value.loc27: %type_where = facet_value constants.%Z, () [concrete = constants.%facet_value.4e3]
// CHECK:STDOUT:     %.loc27: %type_where = converted constants.%Z, %facet_value.loc27 [concrete = constants.%facet_value.4e3]
// CHECK:STDOUT:     %AggregateT.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %u.var, constants.%AggregateT.as_type.as.Destroy.impl.Op.dd5
// CHECK:STDOUT:     %AggregateT.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function constants.%AggregateT.as_type.as.Destroy.impl.Op.dd5, @AggregateT.as_type.as.Destroy.impl.Op(constants.%facet_value.4e3) [concrete = constants.%AggregateT.as_type.as.Destroy.impl.Op.specific_fn.e7d]
// CHECK:STDOUT:     %bound_method.loc27: <bound method> = bound_method %u.var, %AggregateT.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:     %addr.loc27: %ptr.fb6 = addr_of %u.var
// CHECK:STDOUT:     %AggregateT.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc27(%addr.loc27)
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
// CHECK:STDOUT: specific @A(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc5_13.1 => constants.%T.8b3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.F(constants.%T.8b3, constants.%Self.c6b, constants.%U.2cb) {
// CHECK:STDOUT:   %U.loc6_8.1 => constants.%U.2cb
// CHECK:STDOUT:   %ptr.loc6_22.1 => constants.%ptr.204
// CHECK:STDOUT:   %pattern_type.loc6_18 => constants.%pattern_type.eee
// CHECK:STDOUT:   %T => constants.%T.8b3
// CHECK:STDOUT:   %A.type => constants.%A.type.d88
// CHECK:STDOUT:   %Self => constants.%Self.c6b
// CHECK:STDOUT:   %tuple.type.loc6_40.1 => constants.%tuple.type.098
// CHECK:STDOUT:   %Self.as_type.loc6_40.1 => constants.%Self.as_type.707
// CHECK:STDOUT:   %tuple.type.loc6_40.2 => constants.%tuple.type.cb0
// CHECK:STDOUT:   %pattern_type.loc6_25 => constants.%pattern_type.44d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%X) {
// CHECK:STDOUT:   %T.loc5_13.1 => constants.%X
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.type => constants.%A.type.0a4
// CHECK:STDOUT:   %Self.2 => constants.%Self.08e
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
// CHECK:STDOUT: specific @A.F(constants.%X, constants.%A.facet, constants.%U.8b3) {
// CHECK:STDOUT:   %U.loc6_8.1 => constants.%U.8b3
// CHECK:STDOUT:   %ptr.loc6_22.1 => constants.%ptr.79f131.1
// CHECK:STDOUT:   %pattern_type.loc6_18 => constants.%pattern_type.afe393.1
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.0a4
// CHECK:STDOUT:   %Self => constants.%A.facet
// CHECK:STDOUT:   %tuple.type.loc6_40.1 => constants.%tuple.type.5a1
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
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.impl_witness.4b2
// CHECK:STDOUT:   %Copy.facet => constants.%Copy.facet.cd8
// CHECK:STDOUT:   %.loc17_21.2 => constants.%.7f9
// CHECK:STDOUT:   %impl.elem0.loc17_21.2 => constants.%ptr.as.Copy.impl.Op.0c5
// CHECK:STDOUT:   %specific_impl_fn.loc17_21.2 => constants.%ptr.as.Copy.impl.Op.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGeneric(constants.%T.7bf) {
// CHECK:STDOUT:   %T.loc26_16.1 => constants.%T.7bf
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.F(constants.%X, constants.%T.7bf, constants.%Z) {
// CHECK:STDOUT:   %U.loc6_8.1 => constants.%Z
// CHECK:STDOUT:   %ptr.loc6_22.1 => constants.%ptr.fb6
// CHECK:STDOUT:   %pattern_type.loc6_18 => constants.%pattern_type.f76
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.0a4
// CHECK:STDOUT:   %Self => constants.%T.7bf
// CHECK:STDOUT:   %tuple.type.loc6_40.1 => constants.%tuple.type.5a1
// CHECK:STDOUT:   %Self.as_type.loc6_40.1 => constants.%T.as_type.b3a
// CHECK:STDOUT:   %tuple.type.loc6_40.2 => constants.%tuple.type.af6
// CHECK:STDOUT:   %pattern_type.loc6_25 => constants.%pattern_type.512
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGeneric(constants.%A.facet) {
// CHECK:STDOUT:   %T.loc26_16.1 => constants.%A.facet
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc28_4.2 => constants.%Y
// CHECK:STDOUT:   %A.lookup_impl_witness => constants.%A.impl_witness
// CHECK:STDOUT:   %.loc28_4.3 => constants.%.d35
// CHECK:STDOUT:   %impl.elem0.loc28_4.2 => constants.%Y.as.A.impl.F
// CHECK:STDOUT:   %specific_impl_fn.loc28_4.2 => constants.%Y.as.A.impl.F.specific_fn
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.092
// CHECK:STDOUT:   %require_complete.loc28_12.1 => constants.%complete_type.05d
// CHECK:STDOUT:   %facet_value.loc28_12.2 => constants.%facet_value.352
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%Destroy.impl_witness.89a
// CHECK:STDOUT:   %Destroy.facet => constants.%Destroy.facet.d0e
// CHECK:STDOUT:   %.loc28_12.6 => constants.%.f61
// CHECK:STDOUT:   %impl.elem0.loc28_12.2 => constants.%AggregateT.as_type.as.Destroy.impl.Op.ed3
// CHECK:STDOUT:   %specific_impl_fn.loc28_12.2 => constants.%AggregateT.as_type.as.Destroy.impl.Op.specific_fn.71f
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
// CHECK:STDOUT:   %A.type.d88: type = facet_type <@A, @A(%T.8b3)> [symbolic]
// CHECK:STDOUT:   %Self.c6b: %A.type.d88 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %U.2cb: type = bind_symbolic_name U, 2 [symbolic]
// CHECK:STDOUT:   %pattern_type.20f: type = pattern_type %U.2cb [symbolic]
// CHECK:STDOUT:   %tuple.type.098: type = tuple_type (type, %A.type.d88, type) [symbolic]
// CHECK:STDOUT:   %Self.as_type.707: type = facet_access_type %Self.c6b [symbolic]
// CHECK:STDOUT:   %tuple.type.325: type = tuple_type (%T.8b3, %Self.as_type.707, %U.2cb) [symbolic]
// CHECK:STDOUT:   %pattern_type.491: type = pattern_type %tuple.type.325 [symbolic]
// CHECK:STDOUT:   %A.F.type.17a: type = fn_type @A.F, @A(%T.8b3) [symbolic]
// CHECK:STDOUT:   %A.F.0d8: %A.F.type.17a = struct_value () [symbolic]
// CHECK:STDOUT:   %A.assoc_type.ed3: type = assoc_entity_type @A, @A(%T.8b3) [symbolic]
// CHECK:STDOUT:   %assoc0.fcb: %A.assoc_type.ed3 = assoc_entity element0, @A.%A.F.decl [symbolic]
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Y1: type = class_type @Y1 [concrete]
// CHECK:STDOUT:   %Y2: type = class_type @Y2 [concrete]
// CHECK:STDOUT:   %Z: type = class_type @Z [concrete]
// CHECK:STDOUT:   %V1: type = bind_symbolic_name V1, 0 [symbolic]
// CHECK:STDOUT:   %V2: type = bind_symbolic_name V2, 1 [symbolic]
// CHECK:STDOUT:   %W: type = bind_symbolic_name W, 2 [symbolic]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.30b: type = tuple_type (%V1, %V2) [symbolic]
// CHECK:STDOUT:   %A.type.ff4: type = facet_type <@A, @A(%W)> [symbolic]
// CHECK:STDOUT:   %Self.6c5: %A.type.ff4 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %A.F.type.904: type = fn_type @A.F, @A(%W) [symbolic]
// CHECK:STDOUT:   %A.F.1ee: %A.F.type.904 = struct_value () [symbolic]
// CHECK:STDOUT:   %A.assoc_type.b05: type = assoc_entity_type @A, @A(%W) [symbolic]
// CHECK:STDOUT:   %assoc0.c82: %A.assoc_type.b05 = assoc_entity element0, @A.%A.F.decl [symbolic]
// CHECK:STDOUT:   %require_complete.651: <witness> = require_complete_type %A.type.ff4 [symbolic]
// CHECK:STDOUT:   %A.impl_witness.4bf: <witness> = impl_witness file.%A.impl_witness_table, @tuple.type.as.A.impl(%V1, %V2, %W) [symbolic]
// CHECK:STDOUT:   %U.753: type = bind_symbolic_name U, 3 [symbolic]
// CHECK:STDOUT:   %pattern_type.549: type = pattern_type %U.753 [symbolic]
// CHECK:STDOUT:   %tuple.type.11f: type = tuple_type (type, %tuple.type.24b, type) [concrete]
// CHECK:STDOUT:   %tuple.type.8ae: type = tuple_type (%W, %tuple.type.30b, %U.753) [symbolic]
// CHECK:STDOUT:   %pattern_type.d4a: type = pattern_type %tuple.type.8ae [symbolic]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type.ce1: type = fn_type @tuple.type.as.A.impl.F, @tuple.type.as.A.impl(%V1, %V2, %W) [symbolic]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.cb7: %tuple.type.as.A.impl.F.type.ce1 = struct_value () [symbolic]
// CHECK:STDOUT:   %A.facet.452: %A.type.ff4 = facet_value %tuple.type.30b, (%A.impl_witness.4bf) [symbolic]
// CHECK:STDOUT:   %tuple.type.42f: type = tuple_type (type, %A.type.ff4, type) [symbolic]
// CHECK:STDOUT:   %require_complete.d4d: <witness> = require_complete_type %tuple.type.8ae [symbolic]
// CHECK:STDOUT:   %require_complete.ed4: <witness> = require_complete_type %U.753 [symbolic]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.specific_fn.0dd: <specific function> = specific_function %tuple.type.as.A.impl.F.cb7, @tuple.type.as.A.impl.F(%V1, %V2, %W, %U.753) [symbolic]
// CHECK:STDOUT:   %Call.type: type = fn_type @Call [concrete]
// CHECK:STDOUT:   %Call: %Call.type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.a46: type = tuple_type (%Y1, %Y2) [concrete]
// CHECK:STDOUT:   %A.type.0a4: type = facet_type <@A, @A(%X)> [concrete]
// CHECK:STDOUT:   %Self.08e: %A.type.0a4 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %A.F.type.13d: type = fn_type @A.F, @A(%X) [concrete]
// CHECK:STDOUT:   %A.F.d83: %A.F.type.13d = struct_value () [concrete]
// CHECK:STDOUT:   %A.assoc_type.296: type = assoc_entity_type @A, @A(%X) [concrete]
// CHECK:STDOUT:   %assoc0.5f6: %A.assoc_type.296 = assoc_entity element0, @A.%A.F.decl [concrete]
// CHECK:STDOUT:   %complete_type.e2d: <witness> = complete_type_witness %A.type.0a4 [concrete]
// CHECK:STDOUT:   %A.impl_witness.c8b: <witness> = impl_witness file.%A.impl_witness_table, @tuple.type.as.A.impl(%Y1, %Y2, %X) [concrete]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type.bdb: type = fn_type @tuple.type.as.A.impl.F, @tuple.type.as.A.impl(%Y1, %Y2, %X) [concrete]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.e39: %tuple.type.as.A.impl.F.type.bdb = struct_value () [concrete]
// CHECK:STDOUT:   %A.facet.cdd: %A.type.0a4 = facet_value %tuple.type.a46, (%A.impl_witness.c8b) [concrete]
// CHECK:STDOUT:   %.9b3: type = fn_type_with_self_type %A.F.type.13d, %A.facet.cdd [concrete]
// CHECK:STDOUT:   %pattern_type.8f9: type = pattern_type %Z [concrete]
// CHECK:STDOUT:   %tuple.type.415: type = tuple_type (%X, %tuple.type.a46, %Z) [concrete]
// CHECK:STDOUT:   %pattern_type.0b2: type = pattern_type %tuple.type.415 [concrete]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.specific_fn.417: <specific function> = specific_function %tuple.type.as.A.impl.F.e39, @tuple.type.as.A.impl.F(%Y1, %Y2, %X, %Z) [concrete]
// CHECK:STDOUT:   %Z.val: %Z = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanAggregateDestroy>> [concrete]
// CHECK:STDOUT:   %AggregateT: %type_where = bind_symbolic_name AggregateT, 0 [symbolic]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.type.190: type = fn_type @AggregateT.as_type.as.Destroy.impl.Op, @AggregateT.as_type.as.Destroy.impl(%AggregateT) [symbolic]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.8a0: %AggregateT.as_type.as.Destroy.impl.Op.type.190 = struct_value () [symbolic]
// CHECK:STDOUT:   %facet_value.71a: %type_where = facet_value %tuple.type.415, () [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.37a: <witness> = impl_witness imports.%Destroy.impl_witness_table.2d3, @AggregateT.as_type.as.Destroy.impl(%facet_value.71a) [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.type.d92: type = fn_type @AggregateT.as_type.as.Destroy.impl.Op, @AggregateT.as_type.as.Destroy.impl(%facet_value.71a) [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.f1a: %AggregateT.as_type.as.Destroy.impl.Op.type.d92 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.ad9: type = ptr_type %tuple.type.415 [concrete]
// CHECK:STDOUT:   %complete_type.e23: <witness> = complete_type_witness %ptr.ad9 [concrete]
// CHECK:STDOUT:   %Destroy.facet.684: %Destroy.type = facet_value %tuple.type.415, (%Destroy.impl_witness.37a) [concrete]
// CHECK:STDOUT:   %.27f: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet.684 [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.specific_fn.578: <specific function> = specific_function %AggregateT.as_type.as.Destroy.impl.Op.f1a, @AggregateT.as_type.as.Destroy.impl.Op(%facet_value.71a) [concrete]
// CHECK:STDOUT:   %facet_value.4e3: %type_where = facet_value %Z, () [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.type.18a: type = fn_type @AggregateT.as_type.as.Destroy.impl.Op, @AggregateT.as_type.as.Destroy.impl(%facet_value.4e3) [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.dd5: %AggregateT.as_type.as.Destroy.impl.Op.type.18a = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.fb6: type = ptr_type %Z [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.specific_fn.e7d: <specific function> = specific_function %AggregateT.as_type.as.Destroy.impl.Op.dd5, @AggregateT.as_type.as.Destroy.impl.Op(%facet_value.4e3) [concrete]
// CHECK:STDOUT:   %T.7bf: %A.type.0a4 = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.809: type = pattern_type %A.type.0a4 [concrete]
// CHECK:STDOUT:   %CallGeneric.type: type = fn_type @CallGeneric [concrete]
// CHECK:STDOUT:   %CallGeneric: %CallGeneric.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.7bf [symbolic]
// CHECK:STDOUT:   %A.lookup_impl_witness: <witness> = lookup_impl_witness %T.7bf, @A, @A(%X) [symbolic]
// CHECK:STDOUT:   %.0c6: type = fn_type_with_self_type %A.F.type.13d, %T.7bf [symbolic]
// CHECK:STDOUT:   %impl.elem0.dd2: %.0c6 = impl_witness_access %A.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %tuple.type.5a1: type = tuple_type (type, %A.type.0a4, type) [concrete]
// CHECK:STDOUT:   %tuple.type.780: type = tuple_type (%X, %T.as_type, %Z) [symbolic]
// CHECK:STDOUT:   %pattern_type.f87: type = pattern_type %tuple.type.780 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.304: <specific function> = specific_impl_function %impl.elem0.dd2, @A.F(%X, %T.7bf, %Z) [symbolic]
// CHECK:STDOUT:   %require_complete.4ef: <witness> = require_complete_type %tuple.type.780 [symbolic]
// CHECK:STDOUT:   %facet_value.52b: %type_where = facet_value %tuple.type.780, () [symbolic]
// CHECK:STDOUT:   %ptr.563: type = ptr_type %tuple.type.780 [symbolic]
// CHECK:STDOUT:   %require_complete.147: <witness> = require_complete_type %ptr.563 [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.295: <witness> = lookup_impl_witness %tuple.type.780, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.be7: %Destroy.type = facet_value %tuple.type.780, (%Destroy.lookup_impl_witness.295) [symbolic]
// CHECK:STDOUT:   %.6b1: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet.be7 [symbolic]
// CHECK:STDOUT:   %impl.elem0.f91: %.6b1 = impl_witness_access %Destroy.lookup_impl_witness.295, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.bf8: <specific function> = specific_impl_function %impl.elem0.f91, @Destroy.Op(%Destroy.facet.be7) [symbolic]
// CHECK:STDOUT:   %CallIndirect.type: type = fn_type @CallIndirect [concrete]
// CHECK:STDOUT:   %CallIndirect: %CallIndirect.type = struct_value () [concrete]
// CHECK:STDOUT:   %CallGeneric.specific_fn: <specific function> = specific_function %CallGeneric, @CallGeneric(%A.facet.cdd) [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.d51: @AggregateT.as_type.as.Destroy.impl.%AggregateT.as_type.as.Destroy.impl.Op.type (%AggregateT.as_type.as.Destroy.impl.Op.type.190) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @AggregateT.as_type.as.Destroy.impl.%AggregateT.as_type.as.Destroy.impl.Op (constants.%AggregateT.as_type.as.Destroy.impl.Op.8a0)]
// CHECK:STDOUT:   %Destroy.impl_witness_table.2d3 = impl_witness_table (%Core.import_ref.d51), @AggregateT.as_type.as.Destroy.impl [concrete]
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
// CHECK:STDOUT:     %A.type.loc14_61.2: type = facet_type <@A, @A(constants.%W)> [symbolic = %A.type.loc14_61.1 (constants.%A.type.ff4)]
// CHECK:STDOUT:     %.Self.1: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %V1.loc14_14.2: type = bind_symbolic_name V1, 0 [symbolic = %V1.loc14_14.1 (constants.%V1)]
// CHECK:STDOUT:     %.Self.2: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %V2.loc14_25.2: type = bind_symbolic_name V2, 1 [symbolic = %V2.loc14_25.1 (constants.%V2)]
// CHECK:STDOUT:     %.Self.3: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %W.loc14_36.2: type = bind_symbolic_name W, 2 [symbolic = %W.loc14_36.1 (constants.%W)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.impl_witness_table = impl_witness_table (@tuple.type.as.A.impl.%tuple.type.as.A.impl.F.decl), @tuple.type.as.A.impl [concrete]
// CHECK:STDOUT:   %A.impl_witness: <witness> = impl_witness %A.impl_witness_table, @tuple.type.as.A.impl(constants.%V1, constants.%V2, constants.%W) [symbolic = @tuple.type.as.A.impl.%A.impl_witness (constants.%A.impl_witness.4bf)]
// CHECK:STDOUT:   %Call.decl: %Call.type = fn_decl @Call [concrete = constants.%Call] {} {}
// CHECK:STDOUT:   %CallGeneric.decl: %CallGeneric.type = fn_decl @CallGeneric [concrete = constants.%CallGeneric] {
// CHECK:STDOUT:     %T.patt: %pattern_type.809 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc27: type = splice_block %A.type [concrete = constants.%A.type.0a4] {
// CHECK:STDOUT:       %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %A.ref: %A.type.495 = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:       %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:       %A.type: type = facet_type <@A, @A(constants.%X)> [concrete = constants.%A.type.0a4]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc27_16.2: %A.type.0a4 = bind_symbolic_name T, 0 [symbolic = %T.loc27_16.1 (constants.%T.7bf)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallIndirect.decl: %CallIndirect.type = fn_decl @CallIndirect [concrete = constants.%CallIndirect] {} {}
// CHECK:STDOUT:   %facet_value: %type_where = facet_value constants.%tuple.type.415, () [concrete = constants.%facet_value.71a]
// CHECK:STDOUT:   %.loc28: %type_where = converted constants.%tuple.type.415, %facet_value [concrete = constants.%facet_value.71a]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @A(%T.loc5_13.2: type) {
// CHECK:STDOUT:   %T.loc5_13.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc5_13.1 (constants.%T.8b3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.type: type = facet_type <@A, @A(%T.loc5_13.1)> [symbolic = %A.type (constants.%A.type.d88)]
// CHECK:STDOUT:   %Self.2: @A.%A.type (%A.type.d88) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.c6b)]
// CHECK:STDOUT:   %A.F.type: type = fn_type @A.F, @A(%T.loc5_13.1) [symbolic = %A.F.type (constants.%A.F.type.17a)]
// CHECK:STDOUT:   %A.F: @A.%A.F.type (%A.F.type.17a) = struct_value () [symbolic = %A.F (constants.%A.F.0d8)]
// CHECK:STDOUT:   %A.assoc_type: type = assoc_entity_type @A, @A(%T.loc5_13.1) [symbolic = %A.assoc_type (constants.%A.assoc_type.ed3)]
// CHECK:STDOUT:   %assoc0.loc6_39.2: @A.%A.assoc_type (%A.assoc_type.ed3) = assoc_entity element0, %A.F.decl [symbolic = %assoc0.loc6_39.2 (constants.%assoc0.fcb)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @A.%A.type (%A.type.d88) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.c6b)]
// CHECK:STDOUT:     %A.F.decl: @A.%A.F.type (%A.F.type.17a) = fn_decl @A.F [symbolic = @A.%A.F (constants.%A.F.0d8)] {
// CHECK:STDOUT:       %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 2 [concrete]
// CHECK:STDOUT:       %u.patt: @A.F.%pattern_type.loc6_18 (%pattern_type.20f) = binding_pattern u [concrete]
// CHECK:STDOUT:       %u.param_patt: @A.F.%pattern_type.loc6_18 (%pattern_type.20f) = value_param_pattern %u.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @A.F.%pattern_type.loc6_24 (%pattern_type.491) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @A.F.%pattern_type.loc6_24 (%pattern_type.491) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @A.%T.loc5_13.2 [symbolic = %T (constants.%T.8b3)]
// CHECK:STDOUT:       %.loc6_31: @A.F.%A.type (%A.type.d88) = specific_constant @A.%Self.1, @A(constants.%T.8b3) [symbolic = %Self (constants.%Self.c6b)]
// CHECK:STDOUT:       %Self.ref: @A.F.%A.type (%A.type.d88) = name_ref Self, %.loc6_31 [symbolic = %Self (constants.%Self.c6b)]
// CHECK:STDOUT:       %U.ref.loc6_37: type = name_ref U, %U.loc6_8.2 [symbolic = %U.loc6_8.1 (constants.%U.2cb)]
// CHECK:STDOUT:       %.loc6_38.1: @A.F.%tuple.type.loc6_38.1 (%tuple.type.098) = tuple_literal (%T.ref, %Self.ref, %U.ref.loc6_37)
// CHECK:STDOUT:       %Self.as_type.loc6_38.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc6_38.1 (constants.%Self.as_type.707)]
// CHECK:STDOUT:       %.loc6_38.2: type = converted %Self.ref, %Self.as_type.loc6_38.2 [symbolic = %Self.as_type.loc6_38.1 (constants.%Self.as_type.707)]
// CHECK:STDOUT:       %.loc6_38.3: type = converted %.loc6_38.1, constants.%tuple.type.325 [symbolic = %tuple.type.loc6_38.2 (constants.%tuple.type.325)]
// CHECK:STDOUT:       %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %U.loc6_8.2: type = bind_symbolic_name U, 2 [symbolic = %U.loc6_8.1 (constants.%U.2cb)]
// CHECK:STDOUT:       %u.param: @A.F.%U.loc6_8.1 (%U.2cb) = value_param call_param0
// CHECK:STDOUT:       %U.ref.loc6_21: type = name_ref U, %U.loc6_8.2 [symbolic = %U.loc6_8.1 (constants.%U.2cb)]
// CHECK:STDOUT:       %u: @A.F.%U.loc6_8.1 (%U.2cb) = bind_name u, %u.param
// CHECK:STDOUT:       %return.param: ref @A.F.%tuple.type.loc6_38.2 (%tuple.type.325) = out_param call_param1
// CHECK:STDOUT:       %return: ref @A.F.%tuple.type.loc6_38.2 (%tuple.type.325) = return_slot %return.param
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
// CHECK:STDOUT: generic impl @tuple.type.as.A.impl(%V1.loc14_14.2: type, %V2.loc14_25.2: type, %W.loc14_36.2: type) {
// CHECK:STDOUT:   %V1.loc14_14.1: type = bind_symbolic_name V1, 0 [symbolic = %V1.loc14_14.1 (constants.%V1)]
// CHECK:STDOUT:   %V2.loc14_25.1: type = bind_symbolic_name V2, 1 [symbolic = %V2.loc14_25.1 (constants.%V2)]
// CHECK:STDOUT:   %W.loc14_36.1: type = bind_symbolic_name W, 2 [symbolic = %W.loc14_36.1 (constants.%W)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%V1.loc14_14.1, %V2.loc14_25.1) [symbolic = %tuple.type (constants.%tuple.type.30b)]
// CHECK:STDOUT:   %A.type.loc14_61.1: type = facet_type <@A, @A(%W.loc14_36.1)> [symbolic = %A.type.loc14_61.1 (constants.%A.type.ff4)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %A.type.loc14_61.1 [symbolic = %require_complete (constants.%require_complete.651)]
// CHECK:STDOUT:   %A.impl_witness: <witness> = impl_witness file.%A.impl_witness_table, @tuple.type.as.A.impl(%V1.loc14_14.1, %V2.loc14_25.1, %W.loc14_36.1) [symbolic = %A.impl_witness (constants.%A.impl_witness.4bf)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type: type = fn_type @tuple.type.as.A.impl.F, @tuple.type.as.A.impl(%V1.loc14_14.1, %V2.loc14_25.1, %W.loc14_36.1) [symbolic = %tuple.type.as.A.impl.F.type (constants.%tuple.type.as.A.impl.F.type.ce1)]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F: @tuple.type.as.A.impl.%tuple.type.as.A.impl.F.type (%tuple.type.as.A.impl.F.type.ce1) = struct_value () [symbolic = %tuple.type.as.A.impl.F (constants.%tuple.type.as.A.impl.F.cb7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc14_53.2 as %A.type.loc14_61.2 {
// CHECK:STDOUT:     %tuple.type.as.A.impl.F.decl: @tuple.type.as.A.impl.%tuple.type.as.A.impl.F.type (%tuple.type.as.A.impl.F.type.ce1) = fn_decl @tuple.type.as.A.impl.F [symbolic = @tuple.type.as.A.impl.%tuple.type.as.A.impl.F (constants.%tuple.type.as.A.impl.F.cb7)] {
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
// CHECK:STDOUT: generic fn @A.F(@A.%T.loc5_13.2: type, @A.%Self.1: @A.%A.type (%A.type.d88), %U.loc6_8.2: type) {
// CHECK:STDOUT:   %U.loc6_8.1: type = bind_symbolic_name U, 2 [symbolic = %U.loc6_8.1 (constants.%U.2cb)]
// CHECK:STDOUT:   %pattern_type.loc6_18: type = pattern_type %U.loc6_8.1 [symbolic = %pattern_type.loc6_18 (constants.%pattern_type.20f)]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T.8b3)]
// CHECK:STDOUT:   %A.type: type = facet_type <@A, @A(%T)> [symbolic = %A.type (constants.%A.type.d88)]
// CHECK:STDOUT:   %Self: @A.F.%A.type (%A.type.d88) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.c6b)]
// CHECK:STDOUT:   %tuple.type.loc6_38.1: type = tuple_type (type, %A.type, type) [symbolic = %tuple.type.loc6_38.1 (constants.%tuple.type.098)]
// CHECK:STDOUT:   %Self.as_type.loc6_38.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc6_38.1 (constants.%Self.as_type.707)]
// CHECK:STDOUT:   %tuple.type.loc6_38.2: type = tuple_type (%T, %Self.as_type.loc6_38.1, %U.loc6_8.1) [symbolic = %tuple.type.loc6_38.2 (constants.%tuple.type.325)]
// CHECK:STDOUT:   %pattern_type.loc6_24: type = pattern_type %tuple.type.loc6_38.2 [symbolic = %pattern_type.loc6_24 (constants.%pattern_type.491)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param: @A.F.%U.loc6_8.1 (%U.2cb)) -> @A.F.%tuple.type.loc6_38.2 (%tuple.type.325);
// CHECK:STDOUT: }
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
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type: type = fn_type @tuple.type.as.A.impl.F, @tuple.type.as.A.impl(%V1, %V2, %W) [symbolic = %tuple.type.as.A.impl.F.type (constants.%tuple.type.as.A.impl.F.type.ce1)]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F: @tuple.type.as.A.impl.F.%tuple.type.as.A.impl.F.type (%tuple.type.as.A.impl.F.type.ce1) = struct_value () [symbolic = %tuple.type.as.A.impl.F (constants.%tuple.type.as.A.impl.F.cb7)]
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.specific_fn.loc18_12.2: <specific function> = specific_function %tuple.type.as.A.impl.F, @tuple.type.as.A.impl.F(%V1, %V2, %W, %U.loc17_8.1) [symbolic = %tuple.type.as.A.impl.F.specific_fn.loc18_12.2 (constants.%tuple.type.as.A.impl.F.specific_fn.0dd)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param: @tuple.type.as.A.impl.F.%U.loc17_8.1 (%U.753)) -> %return.param: @tuple.type.as.A.impl.F.%tuple.type.loc17_42.2 (%tuple.type.8ae) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc18: @tuple.type.as.A.impl.F.%tuple.type.as.A.impl.F.type (%tuple.type.as.A.impl.F.type.ce1) = specific_constant @tuple.type.as.A.impl.%tuple.type.as.A.impl.F.decl, @tuple.type.as.A.impl(constants.%V1, constants.%V2, constants.%W) [symbolic = %tuple.type.as.A.impl.F (constants.%tuple.type.as.A.impl.F.cb7)]
// CHECK:STDOUT:     %F.ref: @tuple.type.as.A.impl.F.%tuple.type.as.A.impl.F.type (%tuple.type.as.A.impl.F.type.ce1) = name_ref F, %.loc18 [symbolic = %tuple.type.as.A.impl.F (constants.%tuple.type.as.A.impl.F.cb7)]
// CHECK:STDOUT:     %U.ref.loc18: type = name_ref U, %U.loc17_8.2 [symbolic = %U.loc17_8.1 (constants.%U.753)]
// CHECK:STDOUT:     %u.ref: @tuple.type.as.A.impl.F.%U.loc17_8.1 (%U.753) = name_ref u, %u
// CHECK:STDOUT:     %tuple.type.as.A.impl.F.specific_fn.loc18_12.1: <specific function> = specific_function %F.ref, @tuple.type.as.A.impl.F(constants.%V1, constants.%V2, constants.%W, constants.%U.753) [symbolic = %tuple.type.as.A.impl.F.specific_fn.loc18_12.2 (constants.%tuple.type.as.A.impl.F.specific_fn.0dd)]
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
// CHECK:STDOUT:   %A.type: type = facet_type <@A, @A(constants.%X)> [concrete = constants.%A.type.0a4]
// CHECK:STDOUT:   %A.facet: %A.type.0a4 = facet_value %.loc24_14, (constants.%A.impl_witness.c8b) [concrete = constants.%A.facet.cdd]
// CHECK:STDOUT:   %.loc24_23: %A.type.0a4 = converted %.loc24_14, %A.facet [concrete = constants.%A.facet.cdd]
// CHECK:STDOUT:   %.loc24_31.1: %A.assoc_type.296 = specific_constant @A.%assoc0.loc6_39.1, @A(constants.%X) [concrete = constants.%assoc0.5f6]
// CHECK:STDOUT:   %F.ref: %A.assoc_type.296 = name_ref F, %.loc24_31.1 [concrete = constants.%assoc0.5f6]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc24_23 [concrete = constants.%tuple.type.a46]
// CHECK:STDOUT:   %.loc24_31.2: type = converted %.loc24_23, %as_type [concrete = constants.%tuple.type.a46]
// CHECK:STDOUT:   %impl.elem0: %.9b3 = impl_witness_access constants.%A.impl_witness.c8b, element0 [concrete = constants.%tuple.type.as.A.impl.F.e39]
// CHECK:STDOUT:   %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z]
// CHECK:STDOUT:   %.loc24_38.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @tuple.type.as.A.impl.F(constants.%Y1, constants.%Y2, constants.%X, constants.%Z) [concrete = constants.%tuple.type.as.A.impl.F.specific_fn.417]
// CHECK:STDOUT:   %.loc24_39.1: ref %tuple.type.415 = temporary_storage
// CHECK:STDOUT:   %.loc24_38.2: ref %Z = temporary_storage
// CHECK:STDOUT:   %.loc24_38.3: init %Z = class_init (), %.loc24_38.2 [concrete = constants.%Z.val]
// CHECK:STDOUT:   %.loc24_38.4: ref %Z = temporary %.loc24_38.2, %.loc24_38.3
// CHECK:STDOUT:   %.loc24_38.5: ref %Z = converted %.loc24_38.1, %.loc24_38.4
// CHECK:STDOUT:   %.loc24_38.6: %Z = bind_value %.loc24_38.5
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.call: init %tuple.type.415 = call %specific_fn(%.loc24_38.6) to %.loc24_39.1
// CHECK:STDOUT:   %.loc24_39.2: ref %tuple.type.415 = temporary %.loc24_39.1, %tuple.type.as.A.impl.F.call
// CHECK:STDOUT:   %facet_value.loc24_39: %type_where = facet_value constants.%tuple.type.415, () [concrete = constants.%facet_value.71a]
// CHECK:STDOUT:   %.loc24_39.3: %type_where = converted constants.%tuple.type.415, %facet_value.loc24_39 [concrete = constants.%facet_value.71a]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.bound.loc24_39: <bound method> = bound_method %.loc24_39.2, constants.%AggregateT.as_type.as.Destroy.impl.Op.f1a
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.specific_fn.1: <specific function> = specific_function constants.%AggregateT.as_type.as.Destroy.impl.Op.f1a, @AggregateT.as_type.as.Destroy.impl.Op(constants.%facet_value.71a) [concrete = constants.%AggregateT.as_type.as.Destroy.impl.Op.specific_fn.578]
// CHECK:STDOUT:   %bound_method.loc24_39: <bound method> = bound_method %.loc24_39.2, %AggregateT.as_type.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc24_39: %ptr.ad9 = addr_of %.loc24_39.2
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.call.loc24_39: init %empty_tuple.type = call %bound_method.loc24_39(%addr.loc24_39)
// CHECK:STDOUT:   %facet_value.loc24_38: %type_where = facet_value constants.%Z, () [concrete = constants.%facet_value.4e3]
// CHECK:STDOUT:   %.loc24_38.7: %type_where = converted constants.%Z, %facet_value.loc24_38 [concrete = constants.%facet_value.4e3]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.bound.loc24_38: <bound method> = bound_method %.loc24_38.4, constants.%AggregateT.as_type.as.Destroy.impl.Op.dd5
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.specific_fn.2: <specific function> = specific_function constants.%AggregateT.as_type.as.Destroy.impl.Op.dd5, @AggregateT.as_type.as.Destroy.impl.Op(constants.%facet_value.4e3) [concrete = constants.%AggregateT.as_type.as.Destroy.impl.Op.specific_fn.e7d]
// CHECK:STDOUT:   %bound_method.loc24_38: <bound method> = bound_method %.loc24_38.4, %AggregateT.as_type.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc24_38: %ptr.fb6 = addr_of %.loc24_38.4
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.call.loc24_38: init %empty_tuple.type = call %bound_method.loc24_38(%addr.loc24_38)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallGeneric(%T.loc27_16.2: %A.type.0a4) {
// CHECK:STDOUT:   %T.loc27_16.1: %A.type.0a4 = bind_symbolic_name T, 0 [symbolic = %T.loc27_16.1 (constants.%T.7bf)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc28_4.2: type = facet_access_type %T.loc27_16.1 [symbolic = %T.as_type.loc28_4.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %A.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc27_16.1, @A, @A(constants.%X) [symbolic = %A.lookup_impl_witness (constants.%A.lookup_impl_witness)]
// CHECK:STDOUT:   %.loc28_4.3: type = fn_type_with_self_type constants.%A.F.type.13d, %T.loc27_16.1 [symbolic = %.loc28_4.3 (constants.%.0c6)]
// CHECK:STDOUT:   %impl.elem0.loc28_4.2: @CallGeneric.%.loc28_4.3 (%.0c6) = impl_witness_access %A.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc28_4.2 (constants.%impl.elem0.dd2)]
// CHECK:STDOUT:   %specific_impl_fn.loc28_4.2: <specific function> = specific_impl_function %impl.elem0.loc28_4.2, @A.F(constants.%X, %T.loc27_16.1, constants.%Z) [symbolic = %specific_impl_fn.loc28_4.2 (constants.%specific_impl_fn.304)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (constants.%X, %T.as_type.loc28_4.2, constants.%Z) [symbolic = %tuple.type (constants.%tuple.type.780)]
// CHECK:STDOUT:   %require_complete.loc28_12.1: <witness> = require_complete_type %tuple.type [symbolic = %require_complete.loc28_12.1 (constants.%require_complete.4ef)]
// CHECK:STDOUT:   %facet_value.loc28_12.2: %type_where = facet_value %tuple.type, () [symbolic = %facet_value.loc28_12.2 (constants.%facet_value.52b)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %tuple.type, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.295)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %tuple.type, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet (constants.%Destroy.facet.be7)]
// CHECK:STDOUT:   %.loc28_12.6: type = fn_type_with_self_type constants.%Destroy.Op.type, %Destroy.facet [symbolic = %.loc28_12.6 (constants.%.6b1)]
// CHECK:STDOUT:   %impl.elem0.loc28_12.2: @CallGeneric.%.loc28_12.6 (%.6b1) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc28_12.2 (constants.%impl.elem0.f91)]
// CHECK:STDOUT:   %specific_impl_fn.loc28_12.2: <specific function> = specific_impl_function %impl.elem0.loc28_12.2, @Destroy.Op(%Destroy.facet) [symbolic = %specific_impl_fn.loc28_12.2 (constants.%specific_impl_fn.bf8)]
// CHECK:STDOUT:   %ptr: type = ptr_type %tuple.type [symbolic = %ptr (constants.%ptr.563)]
// CHECK:STDOUT:   %require_complete.loc28_12.2: <witness> = require_complete_type %ptr [symbolic = %require_complete.loc28_12.2 (constants.%require_complete.147)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref: %A.type.0a4 = name_ref T, %T.loc27_16.2 [symbolic = %T.loc27_16.1 (constants.%T.7bf)]
// CHECK:STDOUT:     %.loc28_4.1: %A.assoc_type.296 = specific_constant @A.%assoc0.loc6_39.1, @A(constants.%X) [concrete = constants.%assoc0.5f6]
// CHECK:STDOUT:     %F.ref: %A.assoc_type.296 = name_ref F, %.loc28_4.1 [concrete = constants.%assoc0.5f6]
// CHECK:STDOUT:     %T.as_type.loc28_4.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc28_4.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc28_4.2: type = converted %T.ref, %T.as_type.loc28_4.1 [symbolic = %T.as_type.loc28_4.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %impl.elem0.loc28_4.1: @CallGeneric.%.loc28_4.3 (%.0c6) = impl_witness_access constants.%A.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc28_4.2 (constants.%impl.elem0.dd2)]
// CHECK:STDOUT:     %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z]
// CHECK:STDOUT:     %.loc28_11.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %specific_impl_fn.loc28_4.1: <specific function> = specific_impl_function %impl.elem0.loc28_4.1, @A.F(constants.%X, constants.%T.7bf, constants.%Z) [symbolic = %specific_impl_fn.loc28_4.2 (constants.%specific_impl_fn.304)]
// CHECK:STDOUT:     %.loc28_12.1: ref @CallGeneric.%tuple.type (%tuple.type.780) = temporary_storage
// CHECK:STDOUT:     %.loc28_11.2: ref %Z = temporary_storage
// CHECK:STDOUT:     %.loc28_11.3: init %Z = class_init (), %.loc28_11.2 [concrete = constants.%Z.val]
// CHECK:STDOUT:     %.loc28_11.4: ref %Z = temporary %.loc28_11.2, %.loc28_11.3
// CHECK:STDOUT:     %.loc28_11.5: ref %Z = converted %.loc28_11.1, %.loc28_11.4
// CHECK:STDOUT:     %.loc28_11.6: %Z = bind_value %.loc28_11.5
// CHECK:STDOUT:     %.loc28_12.2: init @CallGeneric.%tuple.type (%tuple.type.780) = call %specific_impl_fn.loc28_4.1(%.loc28_11.6) to %.loc28_12.1
// CHECK:STDOUT:     %.loc28_12.3: ref @CallGeneric.%tuple.type (%tuple.type.780) = temporary %.loc28_12.1, %.loc28_12.2
// CHECK:STDOUT:     %facet_value.loc28_12.1: %type_where = facet_value constants.%tuple.type.780, () [symbolic = %facet_value.loc28_12.2 (constants.%facet_value.52b)]
// CHECK:STDOUT:     %.loc28_12.4: %type_where = converted constants.%tuple.type.780, %facet_value.loc28_12.1 [symbolic = %facet_value.loc28_12.2 (constants.%facet_value.52b)]
// CHECK:STDOUT:     %impl.elem0.loc28_12.1: @CallGeneric.%.loc28_12.6 (%.6b1) = impl_witness_access constants.%Destroy.lookup_impl_witness.295, element0 [symbolic = %impl.elem0.loc28_12.2 (constants.%impl.elem0.f91)]
// CHECK:STDOUT:     %bound_method.loc28_12.1: <bound method> = bound_method %.loc28_12.3, %impl.elem0.loc28_12.1
// CHECK:STDOUT:     %specific_impl_fn.loc28_12.1: <specific function> = specific_impl_function %impl.elem0.loc28_12.1, @Destroy.Op(constants.%Destroy.facet.be7) [symbolic = %specific_impl_fn.loc28_12.2 (constants.%specific_impl_fn.bf8)]
// CHECK:STDOUT:     %bound_method.loc28_12.2: <bound method> = bound_method %.loc28_12.3, %specific_impl_fn.loc28_12.1
// CHECK:STDOUT:     %addr.loc28_12: @CallGeneric.%ptr (%ptr.563) = addr_of %.loc28_12.3
// CHECK:STDOUT:     %.loc28_12.5: init %empty_tuple.type = call %bound_method.loc28_12.2(%addr.loc28_12)
// CHECK:STDOUT:     %facet_value.loc28_11: %type_where = facet_value constants.%Z, () [concrete = constants.%facet_value.4e3]
// CHECK:STDOUT:     %.loc28_11.7: %type_where = converted constants.%Z, %facet_value.loc28_11 [concrete = constants.%facet_value.4e3]
// CHECK:STDOUT:     %AggregateT.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc28_11.4, constants.%AggregateT.as_type.as.Destroy.impl.Op.dd5
// CHECK:STDOUT:     %AggregateT.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function constants.%AggregateT.as_type.as.Destroy.impl.Op.dd5, @AggregateT.as_type.as.Destroy.impl.Op(constants.%facet_value.4e3) [concrete = constants.%AggregateT.as_type.as.Destroy.impl.Op.specific_fn.e7d]
// CHECK:STDOUT:     %bound_method.loc28_11: <bound method> = bound_method %.loc28_11.4, %AggregateT.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:     %addr.loc28_11: %ptr.fb6 = addr_of %.loc28_11.4
// CHECK:STDOUT:     %AggregateT.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc28_11(%addr.loc28_11)
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
// CHECK:STDOUT:   %A.facet: %A.type.0a4 = facet_value %.loc33_24, (constants.%A.impl_witness.c8b) [concrete = constants.%A.facet.cdd]
// CHECK:STDOUT:   %.loc33_31: %A.type.0a4 = converted %.loc33_24, %A.facet [concrete = constants.%A.facet.cdd]
// CHECK:STDOUT:   %CallGeneric.specific_fn: <specific function> = specific_function %CallGeneric.ref, @CallGeneric(constants.%A.facet.cdd) [concrete = constants.%CallGeneric.specific_fn]
// CHECK:STDOUT:   %CallGeneric.call: init %empty_tuple.type = call %CallGeneric.specific_fn()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc5_13.1 => constants.%T.8b3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.F(constants.%T.8b3, constants.%Self.c6b, constants.%U.2cb) {
// CHECK:STDOUT:   %U.loc6_8.1 => constants.%U.2cb
// CHECK:STDOUT:   %pattern_type.loc6_18 => constants.%pattern_type.20f
// CHECK:STDOUT:   %T => constants.%T.8b3
// CHECK:STDOUT:   %A.type => constants.%A.type.d88
// CHECK:STDOUT:   %Self => constants.%Self.c6b
// CHECK:STDOUT:   %tuple.type.loc6_38.1 => constants.%tuple.type.098
// CHECK:STDOUT:   %Self.as_type.loc6_38.1 => constants.%Self.as_type.707
// CHECK:STDOUT:   %tuple.type.loc6_38.2 => constants.%tuple.type.325
// CHECK:STDOUT:   %pattern_type.loc6_24 => constants.%pattern_type.491
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%W) {
// CHECK:STDOUT:   %T.loc5_13.1 => constants.%W
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.type => constants.%A.type.ff4
// CHECK:STDOUT:   %Self.2 => constants.%Self.6c5
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
// CHECK:STDOUT:   %A.type.loc14_61.1 => constants.%A.type.ff4
// CHECK:STDOUT:   %require_complete => constants.%require_complete.651
// CHECK:STDOUT:   %A.impl_witness => constants.%A.impl_witness.4bf
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type => constants.%tuple.type.as.A.impl.F.type.ce1
// CHECK:STDOUT:   %tuple.type.as.A.impl.F => constants.%tuple.type.as.A.impl.F.cb7
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
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type => constants.%tuple.type.as.A.impl.F.type.ce1
// CHECK:STDOUT:   %tuple.type.as.A.impl.F => constants.%tuple.type.as.A.impl.F.cb7
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.specific_fn.loc18_12.2 => constants.%tuple.type.as.A.impl.F.specific_fn.0dd
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.F(constants.%W, constants.%A.facet.452, constants.%U.753) {
// CHECK:STDOUT:   %U.loc6_8.1 => constants.%U.753
// CHECK:STDOUT:   %pattern_type.loc6_18 => constants.%pattern_type.549
// CHECK:STDOUT:   %T => constants.%W
// CHECK:STDOUT:   %A.type => constants.%A.type.ff4
// CHECK:STDOUT:   %Self => constants.%A.facet.452
// CHECK:STDOUT:   %tuple.type.loc6_38.1 => constants.%tuple.type.42f
// CHECK:STDOUT:   %Self.as_type.loc6_38.1 => constants.%tuple.type.30b
// CHECK:STDOUT:   %tuple.type.loc6_38.2 => constants.%tuple.type.8ae
// CHECK:STDOUT:   %pattern_type.loc6_24 => constants.%pattern_type.d4a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%X) {
// CHECK:STDOUT:   %T.loc5_13.1 => constants.%X
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.type => constants.%A.type.0a4
// CHECK:STDOUT:   %Self.2 => constants.%Self.08e
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
// CHECK:STDOUT:   %A.type.loc14_61.1 => constants.%A.type.0a4
// CHECK:STDOUT:   %require_complete => constants.%complete_type.e2d
// CHECK:STDOUT:   %A.impl_witness => constants.%A.impl_witness.c8b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type => constants.%tuple.type.as.A.impl.F.type.bdb
// CHECK:STDOUT:   %tuple.type.as.A.impl.F => constants.%tuple.type.as.A.impl.F.e39
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
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.type => constants.%tuple.type.as.A.impl.F.type.bdb
// CHECK:STDOUT:   %tuple.type.as.A.impl.F => constants.%tuple.type.as.A.impl.F.e39
// CHECK:STDOUT:   %tuple.type.as.A.impl.F.specific_fn.loc18_12.2 => constants.%tuple.type.as.A.impl.F.specific_fn.417
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGeneric(constants.%T.7bf) {
// CHECK:STDOUT:   %T.loc27_16.1 => constants.%T.7bf
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.F(constants.%X, constants.%T.7bf, constants.%Z) {
// CHECK:STDOUT:   %U.loc6_8.1 => constants.%Z
// CHECK:STDOUT:   %pattern_type.loc6_18 => constants.%pattern_type.8f9
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.0a4
// CHECK:STDOUT:   %Self => constants.%T.7bf
// CHECK:STDOUT:   %tuple.type.loc6_38.1 => constants.%tuple.type.5a1
// CHECK:STDOUT:   %Self.as_type.loc6_38.1 => constants.%T.as_type
// CHECK:STDOUT:   %tuple.type.loc6_38.2 => constants.%tuple.type.780
// CHECK:STDOUT:   %pattern_type.loc6_24 => constants.%pattern_type.f87
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGeneric(constants.%A.facet.cdd) {
// CHECK:STDOUT:   %T.loc27_16.1 => constants.%A.facet.cdd
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc28_4.2 => constants.%tuple.type.a46
// CHECK:STDOUT:   %A.lookup_impl_witness => constants.%A.impl_witness.c8b
// CHECK:STDOUT:   %.loc28_4.3 => constants.%.9b3
// CHECK:STDOUT:   %impl.elem0.loc28_4.2 => constants.%tuple.type.as.A.impl.F.e39
// CHECK:STDOUT:   %specific_impl_fn.loc28_4.2 => constants.%tuple.type.as.A.impl.F.specific_fn.417
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.415
// CHECK:STDOUT:   %require_complete.loc28_12.1 => constants.%complete_type.aa8
// CHECK:STDOUT:   %facet_value.loc28_12.2 => constants.%facet_value.71a
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%Destroy.impl_witness.37a
// CHECK:STDOUT:   %Destroy.facet => constants.%Destroy.facet.684
// CHECK:STDOUT:   %.loc28_12.6 => constants.%.27f
// CHECK:STDOUT:   %impl.elem0.loc28_12.2 => constants.%AggregateT.as_type.as.Destroy.impl.Op.f1a
// CHECK:STDOUT:   %specific_impl_fn.loc28_12.2 => constants.%AggregateT.as_type.as.Destroy.impl.Op.specific_fn.578
// CHECK:STDOUT:   %ptr => constants.%ptr.ad9
// CHECK:STDOUT:   %require_complete.loc28_12.2 => constants.%complete_type.e23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.F(constants.%X, constants.%A.facet.cdd, constants.%Z) {
// CHECK:STDOUT:   %U.loc6_8.1 => constants.%Z
// CHECK:STDOUT:   %pattern_type.loc6_18 => constants.%pattern_type.8f9
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %A.type => constants.%A.type.0a4
// CHECK:STDOUT:   %Self => constants.%A.facet.cdd
// CHECK:STDOUT:   %tuple.type.loc6_38.1 => constants.%tuple.type.5a1
// CHECK:STDOUT:   %Self.as_type.loc6_38.1 => constants.%tuple.type.a46
// CHECK:STDOUT:   %tuple.type.loc6_38.2 => constants.%tuple.type.415
// CHECK:STDOUT:   %pattern_type.loc6_24 => constants.%pattern_type.0b2
// CHECK:STDOUT: }
// CHECK:STDOUT:

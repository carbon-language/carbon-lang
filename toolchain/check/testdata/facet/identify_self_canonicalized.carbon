// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/identify_self_canonicalized.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/identify_self_canonicalized.carbon

// --- identify_self_canonicalized.carbon
library "[[@TEST_NAME]]";

interface Z {}

constraint W {
  extend require impls Z;
}

interface Y {}

// T is a facet value (a SymbolicBinding), which is converted to `type`. The
// identified facet type finds `(T as type) impls Z` from the top level facet
// type. It also finds `T as Z` through `W`. It's possible for these to disagree
// on the self and thus present as two different required interfaces in the
// identified facet type. Instead we should only get a single interface with the
// canonical `T` facet value as the self value.

impl forall [T:! Y] T as W & Z {}

// --- order_is_independent_of_self.carbon
library "[[@TEST_NAME]]";

interface Z(A:! type, B:! type) {
    let T:! type;
}

// If sorted by specific ids, the order of interfaces in N would be:
// 1. Z((), Self).
// 2. Z({}, Self).
//
// Since these are the first time the specifics are seen, and they are seen in
// that order.
constraint N {
  require impls Z((), Self);
  require impls Z({}, Self);
}

// If sorted by specific ids, the order of interfaces in (C as N) would be:
// 1. Z({}, C).
// 2. Z((), C).
//
// Note that the order differs from that seen in the self-specific of N.
class C {}
//@dump-sem-ir-begin
impl C as Z({}, C) where .T = {} {}
impl C as Z((), C) where .T = () {}
//@dump-sem-ir-end

// If sorted by specific ids, the order of interfaces in (D as N) would be:
// 1. Z((), C).
// 2. Z({}, C).
//
// This order matches that in the self-specific of N.
class D {}
//@dump-sem-ir-begin
impl D as Z((), D) where .T = () {}
impl D as Z({}, D) where .T = {} {}
//@dump-sem-ir-end

fn F() {
  // Here we produce a FacetValue of type N for both C and D. These look like
  // `facet_value %C` and `facet_value %D` in the same. We expect that the order
  // of the witnesses in the facet values are the same for both facet values.
  // This means they are using the order of the interfaces as defined by the
  // self-specific of N, rather than the specific where Self is replaced by C or
  // D respectively.

  //@dump-sem-ir-begin
  C as N;
  D as N;
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- order_is_independent_of_self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Z.type.352: type = generic_interface_type @Z [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Z.generic: %Z.type.352 = struct_value () [concrete]
// CHECK:STDOUT:   %N.type: type = facet_type <@N> [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %Z.type.729: type = facet_type <@Z, @Z(%empty_struct_type, %C)> [concrete]
// CHECK:STDOUT:   %.Self.26f: %Z.type.729 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Z.assoc_type.851: type = assoc_entity_type @Z, @Z(%empty_struct_type, %C) [concrete]
// CHECK:STDOUT:   %assoc0.a7b: %Z.assoc_type.851 = assoc_entity element0, @Z.WithSelf.%T [concrete]
// CHECK:STDOUT:   %.Self.binding.as_type.a32: type = symbolic_binding_type .Self, %.Self.26f [symbolic_self]
// CHECK:STDOUT:   %Z.lookup_impl_witness.35b: <witness> = lookup_impl_witness %.Self.26f, @Z, @Z(%empty_struct_type, %C) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.19f: type = impl_witness_access %Z.lookup_impl_witness.35b, element0 [symbolic_self]
// CHECK:STDOUT:   %Z_where.type.615: type = facet_type <@Z, @Z(%empty_struct_type, %C) where %impl.elem0.19f = %empty_struct_type> [concrete]
// CHECK:STDOUT:   %Z.impl_witness.1ba: <witness> = impl_witness @C.as.Z.impl.a45.%Z.impl_witness_table [concrete]
// CHECK:STDOUT:   %Z.type.dc0: type = facet_type <@Z, @Z(%empty_tuple.type, %C)> [concrete]
// CHECK:STDOUT:   %.Self.bce: %Z.type.dc0 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Z.assoc_type.583: type = assoc_entity_type @Z, @Z(%empty_tuple.type, %C) [concrete]
// CHECK:STDOUT:   %assoc0.911: %Z.assoc_type.583 = assoc_entity element0, @Z.WithSelf.%T [concrete]
// CHECK:STDOUT:   %.Self.binding.as_type.d4f: type = symbolic_binding_type .Self, %.Self.bce [symbolic_self]
// CHECK:STDOUT:   %Z.lookup_impl_witness.9de: <witness> = lookup_impl_witness %.Self.bce, @Z, @Z(%empty_tuple.type, %C) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.5ea: type = impl_witness_access %Z.lookup_impl_witness.9de, element0 [symbolic_self]
// CHECK:STDOUT:   %Z_where.type.018: type = facet_type <@Z, @Z(%empty_tuple.type, %C) where %impl.elem0.5ea = %empty_tuple.type> [concrete]
// CHECK:STDOUT:   %Z.impl_witness.b50: <witness> = impl_witness @C.as.Z.impl.72e.%Z.impl_witness_table [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %Z.type.34c: type = facet_type <@Z, @Z(%empty_tuple.type, %D)> [concrete]
// CHECK:STDOUT:   %.Self.098: %Z.type.34c = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Z.assoc_type.d65: type = assoc_entity_type @Z, @Z(%empty_tuple.type, %D) [concrete]
// CHECK:STDOUT:   %assoc0.71a: %Z.assoc_type.d65 = assoc_entity element0, @Z.WithSelf.%T [concrete]
// CHECK:STDOUT:   %.Self.binding.as_type.e7b: type = symbolic_binding_type .Self, %.Self.098 [symbolic_self]
// CHECK:STDOUT:   %Z.lookup_impl_witness.342: <witness> = lookup_impl_witness %.Self.098, @Z, @Z(%empty_tuple.type, %D) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.624: type = impl_witness_access %Z.lookup_impl_witness.342, element0 [symbolic_self]
// CHECK:STDOUT:   %Z_where.type.7ad: type = facet_type <@Z, @Z(%empty_tuple.type, %D) where %impl.elem0.624 = %empty_tuple.type> [concrete]
// CHECK:STDOUT:   %Z.impl_witness.e25: <witness> = impl_witness @D.as.Z.impl.313.%Z.impl_witness_table [concrete]
// CHECK:STDOUT:   %Z.type.1f7: type = facet_type <@Z, @Z(%empty_struct_type, %D)> [concrete]
// CHECK:STDOUT:   %.Self.acb: %Z.type.1f7 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Z.assoc_type.eb3: type = assoc_entity_type @Z, @Z(%empty_struct_type, %D) [concrete]
// CHECK:STDOUT:   %assoc0.779: %Z.assoc_type.eb3 = assoc_entity element0, @Z.WithSelf.%T [concrete]
// CHECK:STDOUT:   %.Self.binding.as_type.122: type = symbolic_binding_type .Self, %.Self.acb [symbolic_self]
// CHECK:STDOUT:   %Z.lookup_impl_witness.abd: <witness> = lookup_impl_witness %.Self.acb, @Z, @Z(%empty_struct_type, %D) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.bfd: type = impl_witness_access %Z.lookup_impl_witness.abd, element0 [symbolic_self]
// CHECK:STDOUT:   %Z_where.type.fac: type = facet_type <@Z, @Z(%empty_struct_type, %D) where %impl.elem0.bfd = %empty_struct_type> [concrete]
// CHECK:STDOUT:   %Z.impl_witness.524: <witness> = impl_witness @D.as.Z.impl.eee.%Z.impl_witness_table [concrete]
// CHECK:STDOUT:   %N.facet.e38: %N.type = facet_value %C, (%Z.impl_witness.b50, %Z.impl_witness.1ba) [concrete]
// CHECK:STDOUT:   %N.facet.5a5: %N.type = facet_value %D, (%Z.impl_witness.e25, %Z.impl_witness.524) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   impl_decl @C.as.Z.impl.a45 [concrete] {} {
// CHECK:STDOUT:     %C.ref.loc25_6: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %Z.ref: %Z.type.352 = name_ref Z, file.%Z.decl [concrete = constants.%Z.generic]
// CHECK:STDOUT:     %.loc25_14: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %C.ref.loc25_17: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %.loc25_18: type = converted %.loc25_14, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %Z.type: type = facet_type <@Z, @Z(constants.%empty_struct_type, constants.%C)> [concrete = constants.%Z.type.729]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %.Self.ref: %Z.type.729 = name_ref .Self, %.Self [symbolic_self = constants.%.Self.26f]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type.a32]
// CHECK:STDOUT:     %.loc25_26.1: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type.a32]
// CHECK:STDOUT:     %.loc25_26.2: %Z.assoc_type.851 = specific_constant @T.%assoc0, @Z.WithSelf(constants.%empty_struct_type, constants.%C, constants.%.Self.26f) [concrete = constants.%assoc0.a7b]
// CHECK:STDOUT:     %T.ref: %Z.assoc_type.851 = name_ref T, %.loc25_26.2 [concrete = constants.%assoc0.a7b]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%Z.lookup_impl_witness.35b, element0 [symbolic_self = constants.%impl.elem0.19f]
// CHECK:STDOUT:     %.loc25_32.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc25_32.2: type = converted %.loc25_32.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc25_20: type = where_expr %.Self [concrete = constants.%Z_where.type.615] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%Z.type.729
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %.loc25_32.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.Z.impl.72e [concrete] {} {
// CHECK:STDOUT:     %C.ref.loc26_6: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %Z.ref: %Z.type.352 = name_ref Z, file.%Z.decl [concrete = constants.%Z.generic]
// CHECK:STDOUT:     %.loc26_14: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %C.ref.loc26_17: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %.loc26_18: type = converted %.loc26_14, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %Z.type: type = facet_type <@Z, @Z(constants.%empty_tuple.type, constants.%C)> [concrete = constants.%Z.type.dc0]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %.Self.ref: %Z.type.dc0 = name_ref .Self, %.Self [symbolic_self = constants.%.Self.bce]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type.d4f]
// CHECK:STDOUT:     %.loc26_26.1: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type.d4f]
// CHECK:STDOUT:     %.loc26_26.2: %Z.assoc_type.583 = specific_constant @T.%assoc0, @Z.WithSelf(constants.%empty_tuple.type, constants.%C, constants.%.Self.bce) [concrete = constants.%assoc0.911]
// CHECK:STDOUT:     %T.ref: %Z.assoc_type.583 = name_ref T, %.loc26_26.2 [concrete = constants.%assoc0.911]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%Z.lookup_impl_witness.9de, element0 [symbolic_self = constants.%impl.elem0.5ea]
// CHECK:STDOUT:     %.loc26_32.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc26_32.2: type = converted %.loc26_32.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc26_20: type = where_expr %.Self [concrete = constants.%Z_where.type.018] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%Z.type.dc0
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %.loc26_32.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @D.as.Z.impl.313 [concrete] {} {
// CHECK:STDOUT:     %D.ref.loc36_6: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %Z.ref: %Z.type.352 = name_ref Z, file.%Z.decl [concrete = constants.%Z.generic]
// CHECK:STDOUT:     %.loc36_14: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %D.ref.loc36_17: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc36_18: type = converted %.loc36_14, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %Z.type: type = facet_type <@Z, @Z(constants.%empty_tuple.type, constants.%D)> [concrete = constants.%Z.type.34c]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %.Self.ref: %Z.type.34c = name_ref .Self, %.Self [symbolic_self = constants.%.Self.098]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type.e7b]
// CHECK:STDOUT:     %.loc36_26.1: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type.e7b]
// CHECK:STDOUT:     %.loc36_26.2: %Z.assoc_type.d65 = specific_constant @T.%assoc0, @Z.WithSelf(constants.%empty_tuple.type, constants.%D, constants.%.Self.098) [concrete = constants.%assoc0.71a]
// CHECK:STDOUT:     %T.ref: %Z.assoc_type.d65 = name_ref T, %.loc36_26.2 [concrete = constants.%assoc0.71a]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%Z.lookup_impl_witness.342, element0 [symbolic_self = constants.%impl.elem0.624]
// CHECK:STDOUT:     %.loc36_32.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc36_32.2: type = converted %.loc36_32.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc36_20: type = where_expr %.Self [concrete = constants.%Z_where.type.7ad] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%Z.type.34c
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %.loc36_32.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @D.as.Z.impl.eee [concrete] {} {
// CHECK:STDOUT:     %D.ref.loc37_6: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %Z.ref: %Z.type.352 = name_ref Z, file.%Z.decl [concrete = constants.%Z.generic]
// CHECK:STDOUT:     %.loc37_14: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %D.ref.loc37_17: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc37_18: type = converted %.loc37_14, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %Z.type: type = facet_type <@Z, @Z(constants.%empty_struct_type, constants.%D)> [concrete = constants.%Z.type.1f7]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %.Self.ref: %Z.type.1f7 = name_ref .Self, %.Self [symbolic_self = constants.%.Self.acb]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type.122]
// CHECK:STDOUT:     %.loc37_26.1: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type.122]
// CHECK:STDOUT:     %.loc37_26.2: %Z.assoc_type.eb3 = specific_constant @T.%assoc0, @Z.WithSelf(constants.%empty_struct_type, constants.%D, constants.%.Self.acb) [concrete = constants.%assoc0.779]
// CHECK:STDOUT:     %T.ref: %Z.assoc_type.eb3 = name_ref T, %.loc37_26.2 [concrete = constants.%assoc0.779]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%Z.lookup_impl_witness.abd, element0 [symbolic_self = constants.%impl.elem0.bfd]
// CHECK:STDOUT:     %.loc37_32.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc37_32.2: type = converted %.loc37_32.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc37_20: type = where_expr %.Self [concrete = constants.%Z_where.type.fac] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%Z.type.1f7
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %.loc37_32.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.Z.impl.a45: %C.ref.loc25_6 as %.loc25_20 {
// CHECK:STDOUT:   %Z.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @C.as.Z.impl.a45 [concrete]
// CHECK:STDOUT:   %Z.impl_witness: <witness> = impl_witness %Z.impl_witness_table [concrete = constants.%Z.impl_witness.1ba]
// CHECK:STDOUT:   %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = %Z.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.Z.impl.72e: %C.ref.loc26_6 as %.loc26_20 {
// CHECK:STDOUT:   %Z.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @C.as.Z.impl.72e [concrete]
// CHECK:STDOUT:   %Z.impl_witness: <witness> = impl_witness %Z.impl_witness_table [concrete = constants.%Z.impl_witness.b50]
// CHECK:STDOUT:   %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = %Z.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @D.as.Z.impl.313: %D.ref.loc36_6 as %.loc36_20 {
// CHECK:STDOUT:   %Z.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @D.as.Z.impl.313 [concrete]
// CHECK:STDOUT:   %Z.impl_witness: <witness> = impl_witness %Z.impl_witness_table [concrete = constants.%Z.impl_witness.e25]
// CHECK:STDOUT:   %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = %Z.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @D.as.Z.impl.eee: %D.ref.loc37_6 as %.loc37_20 {
// CHECK:STDOUT:   %Z.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @D.as.Z.impl.eee [concrete]
// CHECK:STDOUT:   %Z.impl_witness: <witness> = impl_witness %Z.impl_witness_table [concrete = constants.%Z.impl_witness.524]
// CHECK:STDOUT:   %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = %Z.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %N.ref.loc49: type = name_ref N, file.%N.decl [concrete = constants.%N.type]
// CHECK:STDOUT:   %N.facet.loc49: %N.type = facet_value %C.ref, (constants.%Z.impl_witness.b50, constants.%Z.impl_witness.1ba) [concrete = constants.%N.facet.e38]
// CHECK:STDOUT:   %.loc49: %N.type = converted %C.ref, %N.facet.loc49 [concrete = constants.%N.facet.e38]
// CHECK:STDOUT:   %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:   %N.ref.loc50: type = name_ref N, file.%N.decl [concrete = constants.%N.type]
// CHECK:STDOUT:   %N.facet.loc50: %N.type = facet_value %D.ref, (constants.%Z.impl_witness.e25, constants.%Z.impl_witness.524) [concrete = constants.%N.facet.5a5]
// CHECK:STDOUT:   %.loc50: %N.type = converted %D.ref, %N.facet.loc50 [concrete = constants.%N.facet.5a5]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:

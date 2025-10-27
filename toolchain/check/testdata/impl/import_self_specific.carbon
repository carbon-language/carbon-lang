// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
// EXTRA-ARGS: --dump-sem-ir-ranges=ignore
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/import_self_specific.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/import_self_specific.carbon

// --- impl_def.carbon
library "[[@TEST_NAME]]";

// The I.Assoc will be a facet of type Y.
interface Y {}
impl forall [T:! type] T as Y {}

// C takes a facet of type Z.
interface Z {}
impl forall [U:! type] U as Z {}

class C(V:! Z) {
  adapt ();
}

interface I {
  let Assoc:! Y;

  // This is a generic function with a specific that contains a reference `Self`
  // in it (in the ImplWitnessAccess for `Self.Assoc`). The `Self.Assoc` is
  // converted from `Y` to `Z`, so it holds a FacetValue in the specific with a
  // witness for impl lookup of `Self.Assoc as Z`. This `Self` gets imported
  // when an `impl` implements it, and should not be introduced into that
  // `impl`'s generic eval block.
  fn F(c: C(Assoc));
}

// --- impl_use.carbon
library "[[@TEST_NAME]]";

import library "impl_def";

class E {}

class D(N:! E) {}

// The eval block for this generic `impl` must not include any instructions
// derived from the import of `I.F`.
//
// This impl will import each entry in the interface `I`'s witness table. In
// particular, that includes `F` and the specific containing `Self.Assoc as Z`.
// When it does so, the eval block of `F` forms a new local eval block for the
// imported generic. And when the generic is evaluated against a newly
// constructed local self specific, any instructions created in the process of
// that evaluation (such as inside impl lookup deduction against the U in the
// impl for Z) should not be inappropriately added to this impl's generic eval
// block.
impl forall [N:! E] D(N) as I where .Assoc = () {
  fn F(c: C(())) {}
}

// CHECK:STDOUT: --- impl_def.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y> [concrete]
// CHECK:STDOUT:   %Self.29b: %Y.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Y.impl_witness: <witness> = impl_witness file.%Y.impl_witness_table, @T.as.Y.impl(%T) [symbolic]
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %Self.3b3: %Z.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %U: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %Z.impl_witness.5e4: <witness> = impl_witness file.%Z.impl_witness_table, @U.as.Z.impl(%U) [symbolic]
// CHECK:STDOUT:   %V: %Z.type = symbolic_binding V, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.4a0: type = pattern_type %Z.type [concrete]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.4af: type = class_type @C, @C(%V) [symbolic]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_tuple.type [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.9f2: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.%Assoc [concrete]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %Self.9f2, @I [symbolic]
// CHECK:STDOUT:   %impl.elem0: %Y.type = impl_witness_access %I.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %as_type: type = facet_access_type %impl.elem0 [symbolic]
// CHECK:STDOUT:   %Z.impl_witness.58d: <witness> = impl_witness file.%Z.impl_witness_table, @U.as.Z.impl(%as_type) [symbolic]
// CHECK:STDOUT:   %Z.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem0, @Z [symbolic]
// CHECK:STDOUT:   %Z.facet: %Z.type = facet_value %as_type, (%Z.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %C.a0e: type = class_type @C, @C(%Z.facet) [symbolic]
// CHECK:STDOUT:   %pattern_type.8e3: type = pattern_type %C.a0e [symbolic]
// CHECK:STDOUT:   %I.F.type: type = fn_type @I.F [concrete]
// CHECK:STDOUT:   %I.F: %I.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1: %I.assoc_type = assoc_entity element1, @I.%I.F.decl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Y = %Y.decl
// CHECK:STDOUT:     .Z = %Z.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Y.decl: type = interface_decl @Y [concrete = constants.%Y.type] {} {}
// CHECK:STDOUT:   impl_decl @T.as.Y.impl [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc5_14.1 [symbolic = %T.loc5_14.2 (constants.%T)]
// CHECK:STDOUT:     %Y.ref: type = name_ref Y, file.%Y.decl [concrete = constants.%Y.type]
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc5_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc5_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Y.impl_witness_table = impl_witness_table (), @T.as.Y.impl [concrete]
// CHECK:STDOUT:   %Y.impl_witness: <witness> = impl_witness %Y.impl_witness_table, @T.as.Y.impl(constants.%T) [symbolic = @T.as.Y.impl.%Y.impl_witness (constants.%Y.impl_witness)]
// CHECK:STDOUT:   %Z.decl: type = interface_decl @Z [concrete = constants.%Z.type] {} {}
// CHECK:STDOUT:   impl_decl @U.as.Z.impl [concrete] {
// CHECK:STDOUT:     %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.ref: type = name_ref U, %U.loc9_14.1 [symbolic = %U.loc9_14.2 (constants.%U)]
// CHECK:STDOUT:     %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z.type]
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %U.loc9_14.1: type = symbolic_binding U, 0 [symbolic = %U.loc9_14.2 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Z.impl_witness_table = impl_witness_table (), @U.as.Z.impl [concrete]
// CHECK:STDOUT:   %Z.impl_witness: <witness> = impl_witness %Z.impl_witness_table, @U.as.Z.impl(constants.%U) [symbolic = @U.as.Z.impl.%Z.impl_witness (constants.%Z.impl_witness.5e4)]
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [concrete = constants.%C.generic] {
// CHECK:STDOUT:     %V.patt: %pattern_type.4a0 = symbolic_binding_pattern V, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11: type = splice_block %Z.ref [concrete = constants.%Z.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %V.loc11_9.2: %Z.type = symbolic_binding V, 0 [symbolic = %V.loc11_9.1 (constants.%V)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Y {
// CHECK:STDOUT:   %Self: %Y.type = symbolic_binding Self, 0 [symbolic = constants.%Self.29b]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Z {
// CHECK:STDOUT:   %Self: %Z.type = symbolic_binding Self, 0 [symbolic = constants.%Self.3b3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self.9f2]
// CHECK:STDOUT:   %Assoc: %Y.type = assoc_const_decl @Assoc [concrete] {
// CHECK:STDOUT:     %assoc0: %I.assoc_type = assoc_entity element0, @I.%Assoc [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.F.decl: %I.F.type = fn_decl @I.F [concrete = constants.%I.F] {
// CHECK:STDOUT:     %c.patt: @I.F.%pattern_type (%pattern_type.8e3) = value_binding_pattern c [concrete]
// CHECK:STDOUT:     %c.param_patt: @I.F.%pattern_type (%pattern_type.8e3) = value_param_pattern %c.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.param: @I.F.%C.loc24_18.1 (%C.a0e) = value_param call_param0
// CHECK:STDOUT:     %.loc24_18.1: type = splice_block %C.loc24_18.2 [symbolic = %C.loc24_18.1 (constants.%C.a0e)] {
// CHECK:STDOUT:       %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:       %impl.elem0.loc24_13.2: %Y.type = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc24_13.1 (constants.%impl.elem0)]
// CHECK:STDOUT:       %Assoc.ref: %Y.type = name_ref Assoc, %impl.elem0.loc24_13.2 [symbolic = %impl.elem0.loc24_13.1 (constants.%impl.elem0)]
// CHECK:STDOUT:       %Assoc.as_type: type = facet_access_type %Assoc.ref [symbolic = %as_type (constants.%as_type)]
// CHECK:STDOUT:       %Z.facet.loc24_18.2: %Z.type = facet_value %Assoc.as_type, (constants.%Z.lookup_impl_witness) [symbolic = %Z.facet.loc24_18.1 (constants.%Z.facet)]
// CHECK:STDOUT:       %.loc24_18.2: %Z.type = converted %Assoc.ref, %Z.facet.loc24_18.2 [symbolic = %Z.facet.loc24_18.1 (constants.%Z.facet)]
// CHECK:STDOUT:       %C.loc24_18.2: type = class_type @C, @C(constants.%Z.facet) [symbolic = %C.loc24_18.1 (constants.%C.a0e)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: @I.F.%C.loc24_18.1 (%C.a0e) = value_binding c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %I.assoc_type = assoc_entity element1, %I.F.decl [concrete = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Y = <poisoned>
// CHECK:STDOUT:   .Assoc = @Assoc.%assoc0
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .F = %assoc1
// CHECK:STDOUT:   witness = (%Assoc, %I.F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @Assoc(@I.%Self: %I.type) {
// CHECK:STDOUT:   assoc_const Assoc:! %Y.type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.as.Y.impl(%T.loc5_14.1: type) {
// CHECK:STDOUT:   %T.loc5_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc5_14.2 (constants.%T)]
// CHECK:STDOUT:   %Y.impl_witness: <witness> = impl_witness file.%Y.impl_witness_table, @T.as.Y.impl(%T.loc5_14.2) [symbolic = %Y.impl_witness (constants.%Y.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref as %Y.ref {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%Y.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @U.as.Z.impl(%U.loc9_14.1: type) {
// CHECK:STDOUT:   %U.loc9_14.2: type = symbolic_binding U, 0 [symbolic = %U.loc9_14.2 (constants.%U)]
// CHECK:STDOUT:   %Z.impl_witness: <witness> = impl_witness file.%Z.impl_witness_table, @U.as.Z.impl(%U.loc9_14.2) [symbolic = %Z.impl_witness (constants.%Z.impl_witness.5e4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %U.ref as %Z.ref {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%Z.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%V.loc11_9.2: %Z.type) {
// CHECK:STDOUT:   %V.loc11_9.1: %Z.type = symbolic_binding V, 0 [symbolic = %V.loc11_9.1 (constants.%V)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %.loc12_10: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc12_11: type = converted %.loc12_10, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     adapt_decl %.loc12_11 [concrete]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_tuple.type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.4af
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.F(@I.%Self: %I.type) {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.9f2)]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @I [symbolic = %I.lookup_impl_witness (constants.%I.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc24_13.1: %Y.type = impl_witness_access %I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc24_13.1 (constants.%impl.elem0)]
// CHECK:STDOUT:   %as_type: type = facet_access_type %impl.elem0.loc24_13.1 [symbolic = %as_type (constants.%as_type)]
// CHECK:STDOUT:   %Z.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem0.loc24_13.1, @Z [symbolic = %Z.lookup_impl_witness (constants.%Z.lookup_impl_witness)]
// CHECK:STDOUT:   %Z.facet.loc24_18.1: %Z.type = facet_value %as_type, (%Z.lookup_impl_witness) [symbolic = %Z.facet.loc24_18.1 (constants.%Z.facet)]
// CHECK:STDOUT:   %C.loc24_18.1: type = class_type @C, @C(%Z.facet.loc24_18.1) [symbolic = %C.loc24_18.1 (constants.%C.a0e)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C.loc24_18.1 [symbolic = %pattern_type (constants.%pattern_type.8e3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%c.param: @I.F.%C.loc24_18.1 (%C.a0e));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.Y.impl(constants.%T) {
// CHECK:STDOUT:   %T.loc5_14.2 => constants.%T
// CHECK:STDOUT:   %Y.impl_witness => constants.%Y.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U.as.Z.impl(constants.%U) {
// CHECK:STDOUT:   %U.loc9_14.2 => constants.%U
// CHECK:STDOUT:   %Z.impl_witness => constants.%Z.impl_witness.5e4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%V) {
// CHECK:STDOUT:   %V.loc11_9.1 => constants.%V
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Assoc(constants.%Self.9f2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U.as.Z.impl(constants.%as_type) {
// CHECK:STDOUT:   %U.loc9_14.2 => constants.%as_type
// CHECK:STDOUT:   %Z.impl_witness => constants.%Z.impl_witness.58d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%Z.facet) {
// CHECK:STDOUT:   %V.loc11_9.1 => constants.%Z.facet
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.F(constants.%Self.9f2) {
// CHECK:STDOUT:   %Self => constants.%Self.9f2
// CHECK:STDOUT:   %I.lookup_impl_witness => constants.%I.lookup_impl_witness
// CHECK:STDOUT:   %impl.elem0.loc24_13.1 => constants.%impl.elem0
// CHECK:STDOUT:   %as_type => constants.%as_type
// CHECK:STDOUT:   %Z.lookup_impl_witness => constants.%Z.lookup_impl_witness
// CHECK:STDOUT:   %Z.facet.loc24_18.1 => constants.%Z.facet
// CHECK:STDOUT:   %C.loc24_18.1 => constants.%C.a0e
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.8e3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- impl_use.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %E: type = class_type @E [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.3b7: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %N: %E = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.a4a: type = pattern_type %E [concrete]
// CHECK:STDOUT:   %D.type: type = generic_class_type @D [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %D.generic: %D.type = struct_value () [concrete]
// CHECK:STDOUT:   %D: type = class_type @D, @D(%N) [symbolic]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.9f2: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %.Self.420: %I.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, imports.%Main.import_ref.1d8 [concrete]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self.420 [symbolic_self]
// CHECK:STDOUT:   %I.lookup_impl_witness.fd8: <witness> = lookup_impl_witness %.Self.420, @I [symbolic_self]
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y> [concrete]
// CHECK:STDOUT:   %impl.elem0.c51: %Y.type = impl_witness_access %I.lookup_impl_witness.fd8, element0 [symbolic_self]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Y.impl_witness.278: <witness> = impl_witness imports.%Y.impl_witness_table, @T.as.Y.impl(%T) [symbolic]
// CHECK:STDOUT:   %Y.impl_witness.8d4: <witness> = impl_witness imports.%Y.impl_witness_table, @T.as.Y.impl(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %Y.facet: %Y.type = facet_value %empty_tuple.type, (%Y.impl_witness.8d4) [concrete]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I where %impl.elem0.c51 = %Y.facet> [concrete]
// CHECK:STDOUT:   %I.F.type: type = fn_type @I.F [concrete]
// CHECK:STDOUT:   %I.F: %I.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %I.lookup_impl_witness.9b0: <witness> = lookup_impl_witness %Self.9f2, @I [symbolic]
// CHECK:STDOUT:   %impl.elem0.d3e: %Y.type = impl_witness_access %I.lookup_impl_witness.9b0, element0 [symbolic]
// CHECK:STDOUT:   %Z.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem0.d3e, @Z [symbolic]
// CHECK:STDOUT:   %as_type: type = facet_access_type %impl.elem0.d3e [symbolic]
// CHECK:STDOUT:   %Z.facet.aa8: %Z.type = facet_value %as_type, (%Z.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type.782: <witness> = complete_type_witness %empty_tuple.type [concrete]
// CHECK:STDOUT:   %V: %Z.type = symbolic_binding V, 0 [symbolic]
// CHECK:STDOUT:   %C.a0e: type = class_type @C, @C(%Z.facet.aa8) [symbolic]
// CHECK:STDOUT:   %pattern_type.8e3: type = pattern_type %C.a0e [symbolic]
// CHECK:STDOUT:   %U: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %Z.impl_witness.5e4: <witness> = impl_witness imports.%Z.impl_witness_table, @U.as.Z.impl(%U) [symbolic]
// CHECK:STDOUT:   %Z.impl_witness.58d: <witness> = impl_witness imports.%Z.impl_witness_table, @U.as.Z.impl(%as_type) [symbolic]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness file.%I.impl_witness_table, @D.as.I.impl(%N) [symbolic]
// CHECK:STDOUT:   %Z.impl_witness.b07: <witness> = impl_witness imports.%Z.impl_witness_table, @U.as.Z.impl(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %Z.facet.996: %Z.type = facet_value %empty_tuple.type, (%Z.impl_witness.b07) [concrete]
// CHECK:STDOUT:   %C.3c2: type = class_type @C, @C(%Z.facet.996) [concrete]
// CHECK:STDOUT:   %pattern_type.b50: type = pattern_type %C.3c2 [concrete]
// CHECK:STDOUT:   %D.as.I.impl.F.type: type = fn_type @D.as.I.impl.F, @D.as.I.impl(%N) [symbolic]
// CHECK:STDOUT:   %D.as.I.impl.F: %D.as.I.impl.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %D, (%I.impl_witness) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Y = import_ref Main//impl_def, Y, unloaded
// CHECK:STDOUT:   %Main.Z = import_ref Main//impl_def, Z, unloaded
// CHECK:STDOUT:   %Main.C: %C.type = import_ref Main//impl_def, C, loaded [concrete = constants.%C.generic]
// CHECK:STDOUT:   %Main.I: type = import_ref Main//impl_def, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.import_ref.8df = import_ref Main//impl_def, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.9ff: %I.assoc_type = import_ref Main//impl_def, loc16_12, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %Main.import_ref.abf = import_ref Main//impl_def, loc24_20, unloaded
// CHECK:STDOUT:   %Main.Assoc: %Y.type = import_ref Main//impl_def, Assoc, loaded [concrete = %Assoc]
// CHECK:STDOUT:   %Main.F: %I.F.type = import_ref Main//impl_def, F, loaded [concrete = constants.%I.F]
// CHECK:STDOUT:   %Main.import_ref.1d8: %Y.type = import_ref Main//impl_def, loc16_12, loaded [concrete = %Assoc]
// CHECK:STDOUT:   %Main.import_ref.581 = import_ref Main//impl_def, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Assoc: %Y.type = assoc_const_decl @Assoc [concrete] {}
// CHECK:STDOUT:   %Main.import_ref.e339be.1: %I.type = import_ref Main//impl_def, inst{{[0-9A-F]+}} [no loc], loaded [symbolic = constants.%Self.9f2]
// CHECK:STDOUT:   %Main.import_ref.338: <witness> = import_ref Main//impl_def, loc5_31, loaded [symbolic = @T.as.Y.impl.%Y.impl_witness (constants.%Y.impl_witness.278)]
// CHECK:STDOUT:   %Main.import_ref.efcd44.1: type = import_ref Main//impl_def, loc5_14, loaded [symbolic = @T.as.Y.impl.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.cb5: type = import_ref Main//impl_def, loc5_24, loaded [symbolic = @T.as.Y.impl.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.0a1: type = import_ref Main//impl_def, loc5_29, loaded [concrete = constants.%Y.type]
// CHECK:STDOUT:   %Y.impl_witness_table = impl_witness_table (), @T.as.Y.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.462 = import_ref Main//impl_def, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.62f: %Z.type = import_ref Main//impl_def, loc11_9, loaded [symbolic = @C.%V (constants.%V)]
// CHECK:STDOUT:   %Main.import_ref.3fc: <witness> = import_ref Main//impl_def, loc13_1, loaded [concrete = constants.%complete_type.782]
// CHECK:STDOUT:   %Main.import_ref.f7e = import_ref Main//impl_def, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.e339be.2: %I.type = import_ref Main//impl_def, inst{{[0-9A-F]+}} [no loc], loaded [symbolic = constants.%Self.9f2]
// CHECK:STDOUT:   %Main.import_ref.cfb: <witness> = import_ref Main//impl_def, loc9_31, loaded [symbolic = @U.as.Z.impl.%Z.impl_witness (constants.%Z.impl_witness.5e4)]
// CHECK:STDOUT:   %Main.import_ref.efcd44.2: type = import_ref Main//impl_def, loc9_14, loaded [symbolic = @U.as.Z.impl.%U (constants.%U)]
// CHECK:STDOUT:   %Main.import_ref.ff3: type = import_ref Main//impl_def, loc9_24, loaded [symbolic = @U.as.Z.impl.%U (constants.%U)]
// CHECK:STDOUT:   %Main.import_ref.df1: type = import_ref Main//impl_def, loc9_29, loaded [concrete = constants.%Z.type]
// CHECK:STDOUT:   %Z.impl_witness_table = impl_witness_table (), @U.as.Z.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Y = imports.%Main.Y
// CHECK:STDOUT:     .Z = imports.%Main.Z
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .I = imports.%Main.I
// CHECK:STDOUT:     .E = %E.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %E.decl: type = class_decl @E [concrete = constants.%E] {} {}
// CHECK:STDOUT:   %D.decl: %D.type = class_decl @D [concrete = constants.%D.generic] {
// CHECK:STDOUT:     %N.patt: %pattern_type.a4a = symbolic_binding_pattern N, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc7: type = splice_block %E.ref [concrete = constants.%E] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.3b7]
// CHECK:STDOUT:       %E.ref: type = name_ref E, file.%E.decl [concrete = constants.%E]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc7_9.2: %E = symbolic_binding N, 0 [symbolic = %N.loc7_9.1 (constants.%N)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @D.as.I.impl [concrete] {
// CHECK:STDOUT:     %N.patt: %pattern_type.a4a = symbolic_binding_pattern N, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: %D.type = name_ref D, file.%D.decl [concrete = constants.%D.generic]
// CHECK:STDOUT:     %N.ref: %E = name_ref N, %N.loc20_14.2 [symbolic = %N.loc20_14.1 (constants.%N)]
// CHECK:STDOUT:     %D.loc20_24.2: type = class_type @D, @D(constants.%N) [symbolic = %D.loc20_24.1 (constants.%D)]
// CHECK:STDOUT:     %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self.1: %I.type = symbolic_binding .Self [symbolic_self = constants.%.Self.420]
// CHECK:STDOUT:     %.Self.ref: %I.type = name_ref .Self, %.Self.1 [symbolic_self = constants.%.Self.420]
// CHECK:STDOUT:     %Assoc.ref: %I.assoc_type = name_ref Assoc, imports.%Main.import_ref.9ff [concrete = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc20_37: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %impl.elem0: %Y.type = impl_witness_access constants.%I.lookup_impl_witness.fd8, element0 [symbolic_self = constants.%impl.elem0.c51]
// CHECK:STDOUT:     %.loc20_47.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %Y.facet: %Y.type = facet_value constants.%empty_tuple.type, (constants.%Y.impl_witness.8d4) [concrete = constants.%Y.facet]
// CHECK:STDOUT:     %.loc20_47.2: %Y.type = converted %.loc20_47.1, %Y.facet [concrete = constants.%Y.facet]
// CHECK:STDOUT:     %.loc20_31: type = where_expr %.Self.1 [concrete = constants.%I_where.type] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%I.type
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %.loc20_47.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc20_18: type = splice_block %E.ref [concrete = constants.%E] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.3b7]
// CHECK:STDOUT:       %E.ref: type = name_ref E, file.%E.decl [concrete = constants.%E]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc20_14.2: %E = symbolic_binding N, 0 [symbolic = %N.loc20_14.1 (constants.%N)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant, @D.as.I.impl.%D.as.I.impl.F.decl), @D.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table, @D.as.I.impl(constants.%N) [symbolic = @D.as.I.impl.%I.impl_witness (constants.%I.impl_witness)]
// CHECK:STDOUT:   %impl_witness_assoc_constant: %Y.type = impl_witness_assoc_constant constants.%Y.facet [concrete = constants.%Y.facet]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I [from "impl_def.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.8df
// CHECK:STDOUT:   .Assoc = imports.%Main.import_ref.9ff
// CHECK:STDOUT:   .F = imports.%Main.import_ref.abf
// CHECK:STDOUT:   witness = (imports.%Main.Assoc, imports.%Main.F)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Y [from "impl_def.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.581
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Z [from "impl_def.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.462
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @Assoc(imports.%Main.import_ref.e339be.1: %I.type) [from "impl_def.carbon"] {
// CHECK:STDOUT:   assoc_const Assoc:! %Y.type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.as.Y.impl(imports.%Main.import_ref.efcd44.1: type) [from "impl_def.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Y.impl_witness: <witness> = impl_witness imports.%Y.impl_witness_table, @T.as.Y.impl(%T) [symbolic = %Y.impl_witness (constants.%Y.impl_witness.278)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Main.import_ref.cb5 as imports.%Main.import_ref.0a1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%Main.import_ref.338
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @U.as.Z.impl(imports.%Main.import_ref.efcd44.2: type) [from "impl_def.carbon"] {
// CHECK:STDOUT:   %U: type = symbolic_binding U, 0 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:   %Z.impl_witness: <witness> = impl_witness imports.%Z.impl_witness_table, @U.as.Z.impl(%U) [symbolic = %Z.impl_witness (constants.%Z.impl_witness.5e4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Main.import_ref.ff3 as imports.%Main.import_ref.df1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%Main.import_ref.cfb
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @D.as.I.impl(%N.loc20_14.2: %E) {
// CHECK:STDOUT:   %N.loc20_14.1: %E = symbolic_binding N, 0 [symbolic = %N.loc20_14.1 (constants.%N)]
// CHECK:STDOUT:   %D.loc20_24.1: type = class_type @D, @D(%N.loc20_14.1) [symbolic = %D.loc20_24.1 (constants.%D)]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness file.%I.impl_witness_table, @D.as.I.impl(%N.loc20_14.1) [symbolic = %I.impl_witness (constants.%I.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %D.as.I.impl.F.type: type = fn_type @D.as.I.impl.F, @D.as.I.impl(%N.loc20_14.1) [symbolic = %D.as.I.impl.F.type (constants.%D.as.I.impl.F.type)]
// CHECK:STDOUT:   %D.as.I.impl.F: @D.as.I.impl.%D.as.I.impl.F.type (%D.as.I.impl.F.type) = struct_value () [symbolic = %D.as.I.impl.F (constants.%D.as.I.impl.F)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %D.loc20_24.2 as %.loc20_31 {
// CHECK:STDOUT:     %D.as.I.impl.F.decl: @D.as.I.impl.%D.as.I.impl.F.type (%D.as.I.impl.F.type) = fn_decl @D.as.I.impl.F [symbolic = @D.as.I.impl.%D.as.I.impl.F (constants.%D.as.I.impl.F)] {
// CHECK:STDOUT:       %c.patt: %pattern_type.b50 = value_binding_pattern c [concrete]
// CHECK:STDOUT:       %c.param_patt: %pattern_type.b50 = value_param_pattern %c.patt, call_param0 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %c.param: %C.3c2 = value_param call_param0
// CHECK:STDOUT:       %.loc21_15.1: type = splice_block %C [concrete = constants.%C.3c2] {
// CHECK:STDOUT:         %C.ref: %C.type = name_ref C, imports.%Main.C [concrete = constants.%C.generic]
// CHECK:STDOUT:         %.loc21_14: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:         %Z.facet: %Z.type = facet_value constants.%empty_tuple.type, (constants.%Z.impl_witness.b07) [concrete = constants.%Z.facet.996]
// CHECK:STDOUT:         %.loc21_15.2: %Z.type = converted %.loc21_14, %Z.facet [concrete = constants.%Z.facet.996]
// CHECK:STDOUT:         %C: type = class_type @C, @C(constants.%Z.facet.996) [concrete = constants.%C.3c2]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %c: %C.3c2 = value_binding c, %c.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .C = <poisoned>
// CHECK:STDOUT:     .F = %D.as.I.impl.F.decl
// CHECK:STDOUT:     witness = file.%I.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @E {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%E
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @D(%N.loc7_9.2: %E) {
// CHECK:STDOUT:   %N.loc7_9.1: %E = symbolic_binding N, 0 [symbolic = %N.loc7_9.1 (constants.%N)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%D
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(imports.%Main.import_ref.62f: %Z.type) [from "impl_def.carbon"] {
// CHECK:STDOUT:   %V: %Z.type = symbolic_binding V, 0 [symbolic = %V (constants.%V)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%Main.import_ref.3fc
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.f7e
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.F(imports.%Main.import_ref.e339be.2: %I.type) [from "impl_def.carbon"] {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.9f2)]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @I [symbolic = %I.lookup_impl_witness (constants.%I.lookup_impl_witness.9b0)]
// CHECK:STDOUT:   %impl.elem0: %Y.type = impl_witness_access %I.lookup_impl_witness, element0 [symbolic = %impl.elem0 (constants.%impl.elem0.d3e)]
// CHECK:STDOUT:   %as_type: type = facet_access_type %impl.elem0 [symbolic = %as_type (constants.%as_type)]
// CHECK:STDOUT:   %Z.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem0, @Z [symbolic = %Z.lookup_impl_witness (constants.%Z.lookup_impl_witness)]
// CHECK:STDOUT:   %Z.facet: %Z.type = facet_value %as_type, (%Z.lookup_impl_witness) [symbolic = %Z.facet (constants.%Z.facet.aa8)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%Z.facet) [symbolic = %C (constants.%C.a0e)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C [symbolic = %pattern_type (constants.%pattern_type.8e3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @D.as.I.impl.F(@D.as.I.impl.%N.loc20_14.2: %E) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%c.param: %C.3c2) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D(constants.%N) {
// CHECK:STDOUT:   %N.loc7_9.1 => constants.%N
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Assoc(constants.%Self.9f2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Assoc(constants.%.Self.420) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.Y.impl(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Y.impl_witness => constants.%Y.impl_witness.278
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.Y.impl(constants.%empty_tuple.type) {
// CHECK:STDOUT:   %T => constants.%empty_tuple.type
// CHECK:STDOUT:   %Y.impl_witness => constants.%Y.impl_witness.8d4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.as.I.impl(constants.%N) {
// CHECK:STDOUT:   %N.loc20_14.1 => constants.%N
// CHECK:STDOUT:   %D.loc20_24.1 => constants.%D
// CHECK:STDOUT:   %I.impl_witness => constants.%I.impl_witness
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %D.as.I.impl.F.type => constants.%D.as.I.impl.F.type
// CHECK:STDOUT:   %D.as.I.impl.F => constants.%D.as.I.impl.F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%V) {
// CHECK:STDOUT:   %V => constants.%V
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%Z.facet.aa8) {
// CHECK:STDOUT:   %V => constants.%Z.facet.aa8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.F(constants.%Self.9f2) {
// CHECK:STDOUT:   %Self => constants.%Self.9f2
// CHECK:STDOUT:   %I.lookup_impl_witness => constants.%I.lookup_impl_witness.9b0
// CHECK:STDOUT:   %impl.elem0 => constants.%impl.elem0.d3e
// CHECK:STDOUT:   %as_type => constants.%as_type
// CHECK:STDOUT:   %Z.lookup_impl_witness => constants.%Z.lookup_impl_witness
// CHECK:STDOUT:   %Z.facet => constants.%Z.facet.aa8
// CHECK:STDOUT:   %C => constants.%C.a0e
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.8e3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U.as.Z.impl(constants.%U) {
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT:   %Z.impl_witness => constants.%Z.impl_witness.5e4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U.as.Z.impl(constants.%as_type) {
// CHECK:STDOUT:   %U => constants.%as_type
// CHECK:STDOUT:   %Z.impl_witness => constants.%Z.impl_witness.58d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U.as.Z.impl(constants.%empty_tuple.type) {
// CHECK:STDOUT:   %U => constants.%empty_tuple.type
// CHECK:STDOUT:   %Z.impl_witness => constants.%Z.impl_witness.b07
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%Z.facet.996) {
// CHECK:STDOUT:   %V => constants.%Z.facet.996
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.as.I.impl.F(constants.%N) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.F(constants.%I.facet) {
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %I.lookup_impl_witness => constants.%I.impl_witness
// CHECK:STDOUT:   %impl.elem0 => constants.%Y.facet
// CHECK:STDOUT:   %as_type => constants.%empty_tuple.type
// CHECK:STDOUT:   %Z.lookup_impl_witness => constants.%Z.impl_witness.b07
// CHECK:STDOUT:   %Z.facet => constants.%Z.facet.996
// CHECK:STDOUT:   %C => constants.%C.3c2
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.b50
// CHECK:STDOUT: }
// CHECK:STDOUT:

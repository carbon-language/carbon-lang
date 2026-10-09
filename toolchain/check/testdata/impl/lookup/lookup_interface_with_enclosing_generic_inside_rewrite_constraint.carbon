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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/lookup/lookup_interface_with_enclosing_generic_inside_rewrite_constraint.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/lookup/lookup_interface_with_enclosing_generic_inside_rewrite_constraint.carbon

// --- lookup_interface_with_encenclosing_specific_is_concrete_type.carbon
library "[[@TEST_NAME]]";

// 1. The function `Outer.G` is written inside the generic `Outer`. The facet
//    value `H` has a specific with a `SymbolicBinding` for the generic
//    parameter `OuterParam`.
//
// 2. The use of `.T` in the rewrite constraint does a lookup for `.Self` which
//    finds a `FacetType` that includes the interface `Y` which is _also inside_
//    the generic `Outer` class. This results in a `LookupImplWitness`
//    instruction where the query `SpecificInterface` has a specific argument
//    that is a `SymbolicBinding` for the enclosing generic's parameter
//    `OuterParam`.
//
// 3. The call to `G` from a concrete function `F` is against `Outer.G` for a
//    specific `Outer` with a concrete value `()` for `OuterParam`. At the end
//    of deduction for the parameter `H`, we `Subst` the concrete `()` into the
//    type of the function's generic parameters, which is the `FacetType` of
//    `H`.
//
// 4. This substitution includes the rewrite constraints in the `FacetType`,
//    where there is a `LookupImplWitness` instruction (from step 2) with a
//    query `SpecificInterface` that has a `SymbolicBinding`. When
//    substituting, the `SymbolicBinding` is changed to the concrete `()` from
//    the caller's specific. Then the `LookupImplWitness` instruction is
//    evaluated again with its new values.
//
// What used to go wrong here:
//
// 5. The `LookupImplWitness` with the concrete `()` in the specific for the
//    query interface attempts to find a witness from the facet type of `H`. The
//    facet type has a `SpecificInterface` with `SymbolicBinding` in the same
//    position, so it doesn't match and the witness is not used.
//
// 6. The `LookupImplWitness` evaluation fails (this is supposed to be
//    impossible when re-evaluating `LookupImplWitness` so we have violated our
//    assumptions and everything gets in a bad state. The return value is a
//    "runtime" constant value, which Subst turned into a `None` instruction id
//    in the `ImplWitnessAccess` instruction for `.T`. Then we crash later when
//    we try to turn `None` into an `ImplWitness` in eval.
//
// What we need to happen instead:
//
// 7. The lookup in the `FacetType` of `H` needs to succeed. The facet type,
//    which is the self type in the `LookupImplWitness`, must also have the same
//    substitution happen to it as happened to the interface in the query. Then
//    the specific ids in the query interface and the facet type will continue
//    to match after substitution.

class Outer(OuterParam: type) {
  interface Y {
    let T: type;
  }

  fn G(unused generic H: Y where .T = ()) {}
}

class C;
impl C as Outer(()).Y where .T = () {}

fn F() {
  Outer(()).G(C);
}

// --- enclosing_specific_is_facet_value.carbon
library "[[@TEST_NAME]]";

interface Z1 {}
interface Z2 {}

class Outer(OuterParam: Z1) {
  interface Y {
    let T: type;
  }

  fn G(unused generic H: Y where .T = ()) {}

  class C;
  impl C as Y where .T = () {}
}

impl () as Z1 {}
impl () as Z2 {}

fn F() {
  Outer(() as (Z1 & Z2)).G(Outer(() as (Z1 & Z2)).C);
}

// CHECK:STDOUT: --- lookup_interface_with_encenclosing_specific_is_concrete_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen.e58: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type type [concrete]
// CHECK:STDOUT:   %OuterParam.patt: %pattern_type.9a5 = symbolic_binding_pattern OuterParam, 0 [symbolic]
// CHECK:STDOUT:   %OuterParam: type = symbolic_binding OuterParam, 0 [symbolic]
// CHECK:STDOUT:   %Outer.type: type = generic_class_type @Outer [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Outer.generic: %Outer.type = struct_value () [concrete]
// CHECK:STDOUT:   %Outer.0a6: type = class_type @Outer, @Outer(%OuterParam) [symbolic]
// CHECK:STDOUT:   %Y.type.eff: type = facet_type <@Y, @Y(%OuterParam)> [symbolic]
// CHECK:STDOUT:   %Self.f65: %Y.type.eff = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Y.assoc_type.edf: type = assoc_entity_type @Y, @Y(%OuterParam) [symbolic]
// CHECK:STDOUT:   %assoc0.05f: %Y.assoc_type.edf = assoc_entity element0, @Y.WithSelf.%T [symbolic]
// CHECK:STDOUT:   %.Self.frozen.b95: %Y.type.eff = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Y.type.eff [symbolic]
// CHECK:STDOUT:   %.Self.frozen.as_type.55f: type = facet_access_type %.Self.frozen.b95 [symbolic]
// CHECK:STDOUT:   %Y.lookup_impl_witness.eba: <witness> = lookup_impl_witness %.Self.frozen.b95, @Y, @Y(%OuterParam) [symbolic]
// CHECK:STDOUT:   %impl.elem0.29d: type = impl_witness_access %Y.lookup_impl_witness.eba, element0 [symbolic]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %.Self.273: %Y.type.eff = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %Y.lookup_impl_witness.f46: <witness> = lookup_impl_witness %.Self.273, @Y, @Y(%OuterParam) [symbolic]
// CHECK:STDOUT:   %impl.elem0.d7a: type = impl_witness_access %Y.lookup_impl_witness.f46, element0 [symbolic]
// CHECK:STDOUT:   %Y_where.type.fa0: type = facet_type <@Y, @Y(%OuterParam) where %impl.elem0.d7a = %empty_tuple.type> [symbolic]
// CHECK:STDOUT:   %pattern_type.b98: type = pattern_type %Y_where.type.fa0 [symbolic]
// CHECK:STDOUT:   %H.patt.314: %pattern_type.b98 = symbolic_binding_pattern H, 1 [symbolic]
// CHECK:STDOUT:   %H: %Y_where.type.fa0 = symbolic_binding H, 1 [symbolic]
// CHECK:STDOUT:   %Outer.G.type.524: type = fn_type @Outer.G, @Outer(%OuterParam) [symbolic]
// CHECK:STDOUT:   %Outer.G.42e: %Outer.G.type.524 = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %Outer.54c: type = class_type @Outer, @Outer(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %Y.type.d7b: type = facet_type <@Y, @Y(%empty_tuple.type)> [concrete]
// CHECK:STDOUT:   %Outer.G.type.08d: type = fn_type @Outer.G, @Outer(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %Outer.G.e87: %Outer.G.type.08d = struct_value () [concrete]
// CHECK:STDOUT:   %.Self.frozen.62f: %Y.type.d7b = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Self.caf: %Y.type.d7b = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Y.assoc_type.7d9: type = assoc_entity_type @Y, @Y(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %assoc0.559: %Y.assoc_type.7d9 = assoc_entity element0, @Y.WithSelf.%T [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.a06: type = facet_access_type %.Self.frozen.62f [symbolic_self]
// CHECK:STDOUT:   %.e7b: <witness> = impl_self_witness %.Self.frozen.62f, @Y, @Y(%empty_tuple.type) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.654: type = impl_witness_access %.e7b, element0 [symbolic_self]
// CHECK:STDOUT:   %.Self.37b: %Y.type.d7b = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.8a7: <witness> = impl_self_witness %.Self.37b, @Y, @Y(%empty_tuple.type) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.e5f: type = impl_witness_access %.8a7, element0 [symbolic_self]
// CHECK:STDOUT:   %Y_where.type.852: type = facet_type <@Y, @Y(%empty_tuple.type) where %impl.elem0.e5f = %empty_tuple.type> [concrete]
// CHECK:STDOUT:   %.a8e: <witness> = impl_self_witness %C, @Y, @Y(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %Y.impl_witness: <witness> = impl_witness @C.as.Y.impl.%Y.impl_witness_table [concrete]
// CHECK:STDOUT:   %Y.facet.8dd: %Y.type.d7b = facet_value %C, (%Y.impl_witness) [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Y.lookup_impl_witness.b19: <witness> = lookup_impl_witness %.Self.37b, @Y, @Y(%empty_tuple.type) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.b95: type = impl_witness_access %Y.lookup_impl_witness.b19, element0 [symbolic_self]
// CHECK:STDOUT:   %Y_where.type.6ce: type = facet_type <@Y, @Y(%empty_tuple.type) where %impl.elem0.b95 = %empty_tuple.type> [concrete]
// CHECK:STDOUT:   %facet_value: %Y_where.type.6ce = facet_value %C, (%Y.impl_witness) [concrete]
// CHECK:STDOUT:   %complete_type.2cf: <witness> = complete_type_witness %Y.type.d7b [concrete]
// CHECK:STDOUT:   %Y.lookup_impl_witness.e35: <witness> = lookup_impl_witness %.Self.frozen.62f, @Y, @Y(%empty_tuple.type) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.798: type = impl_witness_access %Y.lookup_impl_witness.e35, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.f97: type = pattern_type %Y_where.type.6ce [concrete]
// CHECK:STDOUT:   %H.patt.7ee: %pattern_type.f97 = symbolic_binding_pattern H, 1 [symbolic]
// CHECK:STDOUT:   %Outer.G.specific_fn: <specific function> = specific_function %Outer.G.e87, @Outer.G(%empty_tuple.type, %facet_value) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Outer = %Outer.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Outer.decl: %Outer.type = class_decl @Outer [concrete = constants.%Outer.generic] {
// CHECK:STDOUT:     %OuterParam.patt.loc49_23.1: %pattern_type.9a5 = symbolic_binding_pattern OuterParam, 0 [symbolic = %OuterParam.patt.loc49_23.2 (constants.%OuterParam.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc49_25.1: type = splice_block %.loc49_25.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.e58]
// CHECK:STDOUT:       %.loc49_25.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %OuterParam.loc49_23.2: type = symbolic_binding OuterParam, 0 [symbolic = %OuterParam.loc49_23.1 (constants.%OuterParam)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @C.as.Y.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %Outer.ref: %Outer.type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:     %.loc58_18: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc58_19: type = converted %.loc58_18, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %Outer: type = class_type @Outer, @Outer(constants.%empty_tuple.type) [concrete = constants.%Outer.54c]
// CHECK:STDOUT:     %.loc58_20: type = specific_constant @Outer.%Y.decl, @Outer(constants.%empty_tuple.type) [concrete = constants.%Y.type.d7b]
// CHECK:STDOUT:     %Y.ref: type = name_ref Y, %.loc58_20 [concrete = constants.%Y.type.d7b]
// CHECK:STDOUT:     %.Self.frozen: %Y.type.d7b = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.62f]
// CHECK:STDOUT:     %base_facet_type = requirement_base_facet_type %Y.ref [concrete]
// CHECK:STDOUT:     %.Self.ref: %Y.type.d7b = name_ref .Self, %.Self.frozen [symbolic_self = constants.%.Self.frozen.62f]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.frozen.as_type.a06]
// CHECK:STDOUT:     %.loc58_29.1: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.frozen.as_type.a06]
// CHECK:STDOUT:     %.loc58_29.2: %Y.assoc_type.7d9 = specific_constant @T.%assoc0, @Y.WithSelf(constants.%empty_tuple.type, constants.%.Self.frozen.62f) [concrete = constants.%assoc0.559]
// CHECK:STDOUT:     %T.ref: %Y.assoc_type.7d9 = name_ref T, %.loc58_29.2 [concrete = constants.%assoc0.559]
// CHECK:STDOUT:     %impl.elem0.loc58_29.1: type = impl_witness_access constants.%.e7b, element0 [symbolic_self = constants.%impl.elem0.654]
// CHECK:STDOUT:     %.loc58_35.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc58_35.2: type = converted %.loc58_35.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %rewrite.loc58_32.1 = requirement_rewrite %impl.elem0.loc58_29.1, %.loc58_35.2 [concrete]
// CHECK:STDOUT:     %impl.elem0.loc58_29.2: type = impl_witness_access constants.%.8a7, element0 [symbolic_self = constants.%impl.elem0.e5f]
// CHECK:STDOUT:     %rewrite.loc58_32.2 = requirement_rewrite %impl.elem0.loc58_29.2, %.loc58_35.2 [concrete]
// CHECK:STDOUT:     %.loc58_23: type = where_expr [concrete = constants.%Y_where.type.852] {
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %Y.ref [concrete]
// CHECK:STDOUT:       %rewrite.loc58_32.2 = requirement_rewrite %impl.elem0.loc58_29.2, %.loc58_35.2 [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc58: <witness> = impl_self_witness @C.as.Y.impl.%C.ref, @Y, @Y(constants.%empty_tuple.type) [concrete = constants.%.a8e]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Y(@Outer.%OuterParam.loc49_23.2: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %OuterParam: type = symbolic_binding OuterParam, 0 [symbolic = %OuterParam (constants.%OuterParam)]
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y, @Y(%OuterParam)> [symbolic = %Y.type (constants.%Y.type.eff)]
// CHECK:STDOUT:   %Self.loc50_15.2: @Y.%Y.type (%Y.type.eff) = symbolic_binding Self, 1 [symbolic = %Self.loc50_15.2 (constants.%Self.f65)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc50_15.1: @Y.%Y.type (%Y.type.eff) = symbolic_binding Self, 1 [symbolic = %Self.loc50_15.2 (constants.%Self.f65)]
// CHECK:STDOUT:     %Y.WithSelf.decl = interface_with_self_decl @Y [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %T: type = assoc_const_decl @T [concrete] {
// CHECK:STDOUT:       %assoc0: @Y.WithSelf.%Y.assoc_type (%Y.assoc_type.edf) = assoc_entity element0, @Y.WithSelf.%T [symbolic = @Y.WithSelf.%assoc0 (constants.%assoc0.05f)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc50_15.1
// CHECK:STDOUT:     .T = @T.%assoc0
// CHECK:STDOUT:     witness = (@Y.WithSelf.%T)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.Y.impl: %C.ref as %Y.ref {
// CHECK:STDOUT:   %Y.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @C.as.Y.impl [concrete]
// CHECK:STDOUT:   %Y.impl_witness: <witness> = impl_witness %Y.impl_witness_table [concrete = constants.%Y.impl_witness]
// CHECK:STDOUT:   %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend %Y.ref
// CHECK:STDOUT:   witness = %Y.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Outer(%OuterParam.loc49_23.2: type) {
// CHECK:STDOUT:   %OuterParam.patt.loc49_23.2: %pattern_type.9a5 = symbolic_binding_pattern OuterParam, 0 [symbolic = %OuterParam.patt.loc49_23.2 (constants.%OuterParam.patt)]
// CHECK:STDOUT:   %OuterParam.loc49_23.1: type = symbolic_binding OuterParam, 0 [symbolic = %OuterParam.loc49_23.1 (constants.%OuterParam)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y, @Y(%OuterParam.loc49_23.1)> [symbolic = %Y.type (constants.%Y.type.eff)]
// CHECK:STDOUT:   %Outer.G.type: type = fn_type @Outer.G, @Outer(%OuterParam.loc49_23.1) [symbolic = %Outer.G.type (constants.%Outer.G.type.524)]
// CHECK:STDOUT:   %Outer.G: @Outer.%Outer.G.type (%Outer.G.type.524) = struct_value () [symbolic = %Outer.G (constants.%Outer.G.42e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Y.decl: type = interface_decl @Y [symbolic = @Outer.%Y.type (constants.%Y.type.eff)] {} {}
// CHECK:STDOUT:     %Outer.G.decl: @Outer.%Outer.G.type (%Outer.G.type.524) = fn_decl @Outer.G [symbolic = @Outer.%Outer.G (constants.%Outer.G.42e)] {
// CHECK:STDOUT:       %H.patt.loc54_24.1: @Outer.G.%pattern_type (%pattern_type.b98) = symbolic_binding_pattern H, 1 [symbolic = %H.patt.loc54_24.2 (constants.%H.patt.314)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc54_28.1: type = splice_block %.loc54_28.2 [symbolic = %Y_where.type (constants.%Y_where.type.fa0)] {
// CHECK:STDOUT:         %.Self.frozen.loc54_24: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.e58]
// CHECK:STDOUT:         %.loc54_26: type = specific_constant @Outer.%Y.decl, @Outer(constants.%OuterParam) [symbolic = %Y.type (constants.%Y.type.eff)]
// CHECK:STDOUT:         %Y.ref: type = name_ref Y, %.loc54_26 [symbolic = %Y.type (constants.%Y.type.eff)]
// CHECK:STDOUT:         %.Self.frozen.loc54_28.2: @Outer.G.%Y.type (%Y.type.eff) = symbolic_binding .Self [symbolic = %.Self.frozen.loc54_28.1 (constants.%.Self.frozen.b95)]
// CHECK:STDOUT:         %base_facet_type = requirement_base_facet_type %Y.ref [concrete]
// CHECK:STDOUT:         %.Self.ref: @Outer.G.%Y.type (%Y.type.eff) = name_ref .Self, %.Self.frozen.loc54_28.2 [symbolic = %.Self.frozen.loc54_28.1 (constants.%.Self.frozen.b95)]
// CHECK:STDOUT:         %.Self.as_type: type = facet_access_type %.Self.ref [symbolic = %.Self.frozen.as_type (constants.%.Self.frozen.as_type.55f)]
// CHECK:STDOUT:         %.loc54_34.1: type = converted %.Self.ref, %.Self.as_type [symbolic = %.Self.frozen.as_type (constants.%.Self.frozen.as_type.55f)]
// CHECK:STDOUT:         %.loc54_34.2: @Outer.G.%Y.assoc_type (%Y.assoc_type.edf) = specific_constant @T.%assoc0, @Y.WithSelf(constants.%OuterParam, constants.%.Self.frozen.b95) [symbolic = %assoc0 (constants.%assoc0.05f)]
// CHECK:STDOUT:         %T.ref: @Outer.G.%Y.assoc_type (%Y.assoc_type.edf) = name_ref T, %.loc54_34.2 [symbolic = %assoc0 (constants.%assoc0.05f)]
// CHECK:STDOUT:         %impl.elem0.loc54_34.3: type = impl_witness_access constants.%Y.lookup_impl_witness.eba, element0 [symbolic = %impl.elem0.loc54_34.1 (constants.%impl.elem0.29d)]
// CHECK:STDOUT:         %.loc54_40.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:         %.loc54_40.2: type = converted %.loc54_40.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:         %rewrite.loc54_37.1 = requirement_rewrite %impl.elem0.loc54_34.3, %.loc54_40.2 [concrete]
// CHECK:STDOUT:         %impl.elem0.loc54_34.4: type = impl_witness_access constants.%Y.lookup_impl_witness.f46, element0 [symbolic = %impl.elem0.loc54_34.2 (constants.%impl.elem0.d7a)]
// CHECK:STDOUT:         %rewrite.loc54_37.2 = requirement_rewrite %impl.elem0.loc54_34.4, %.loc54_40.2 [concrete]
// CHECK:STDOUT:         %.loc54_28.2: type = where_expr [symbolic = %Y_where.type (constants.%Y_where.type.fa0)] {
// CHECK:STDOUT:           %base_facet_type = requirement_base_facet_type %Y.ref [concrete]
// CHECK:STDOUT:           %rewrite.loc54_37.2 = requirement_rewrite %impl.elem0.loc54_34.4, %.loc54_40.2 [concrete]
// CHECK:STDOUT:         }
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %H.loc54_24.2: @Outer.G.%Y_where.type (%Y_where.type.fa0) = symbolic_binding H, 1 [symbolic = %H.loc54_24.1 (constants.%H)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Outer.0a6
// CHECK:STDOUT:     .Y = %Y.decl
// CHECK:STDOUT:     .G = %Outer.G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C;
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Outer.G(@Outer.%OuterParam.loc49_23.2: type, %H.loc54_24.2: @Outer.G.%Y_where.type (%Y_where.type.fa0)) {
// CHECK:STDOUT:   %OuterParam: type = symbolic_binding OuterParam, 0 [symbolic = %OuterParam (constants.%OuterParam)]
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y, @Y(%OuterParam)> [symbolic = %Y.type (constants.%Y.type.eff)]
// CHECK:STDOUT:   %.Self.frozen.loc54_28.1: @Outer.G.%Y.type (%Y.type.eff) = symbolic_binding .Self [symbolic = %.Self.frozen.loc54_28.1 (constants.%.Self.frozen.b95)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Y.type [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %.Self.frozen.as_type: type = facet_access_type %.Self.frozen.loc54_28.1 [symbolic = %.Self.frozen.as_type (constants.%.Self.frozen.as_type.55f)]
// CHECK:STDOUT:   %Y.assoc_type: type = assoc_entity_type @Y, @Y(%OuterParam) [symbolic = %Y.assoc_type (constants.%Y.assoc_type.edf)]
// CHECK:STDOUT:   %assoc0: @Outer.G.%Y.assoc_type (%Y.assoc_type.edf) = assoc_entity element0, @Y.WithSelf.%T [symbolic = %assoc0 (constants.%assoc0.05f)]
// CHECK:STDOUT:   %Y.lookup_impl_witness.loc54_34.1: <witness> = lookup_impl_witness %.Self.frozen.loc54_28.1, @Y, @Y(%OuterParam) [symbolic = %Y.lookup_impl_witness.loc54_34.1 (constants.%Y.lookup_impl_witness.eba)]
// CHECK:STDOUT:   %impl.elem0.loc54_34.1: type = impl_witness_access %Y.lookup_impl_witness.loc54_34.1, element0 [symbolic = %impl.elem0.loc54_34.1 (constants.%impl.elem0.29d)]
// CHECK:STDOUT:   %.Self: @Outer.G.%Y.type (%Y.type.eff) = symbolic_binding .Self [symbolic = %.Self (constants.%.Self.273)]
// CHECK:STDOUT:   %Y.lookup_impl_witness.loc54_34.2: <witness> = lookup_impl_witness %.Self, @Y, @Y(%OuterParam) [symbolic = %Y.lookup_impl_witness.loc54_34.2 (constants.%Y.lookup_impl_witness.f46)]
// CHECK:STDOUT:   %impl.elem0.loc54_34.2: type = impl_witness_access %Y.lookup_impl_witness.loc54_34.2, element0 [symbolic = %impl.elem0.loc54_34.2 (constants.%impl.elem0.d7a)]
// CHECK:STDOUT:   %Y_where.type: type = facet_type <@Y, @Y(%OuterParam) where %impl.elem0.loc54_34.2 = constants.%empty_tuple.type> [symbolic = %Y_where.type (constants.%Y_where.type.fa0)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Y_where.type [symbolic = %pattern_type (constants.%pattern_type.b98)]
// CHECK:STDOUT:   %H.patt.loc54_24.2: @Outer.G.%pattern_type (%pattern_type.b98) = symbolic_binding_pattern H, 1 [symbolic = %H.patt.loc54_24.2 (constants.%H.patt.314)]
// CHECK:STDOUT:   %H.loc54_24.1: @Outer.G.%Y_where.type (%Y_where.type.fa0) = symbolic_binding H, 1 [symbolic = %H.loc54_24.1 (constants.%H)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Outer.ref: %Outer.type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:   %.loc61_10: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc61_11: type = converted %.loc61_10, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %Outer: type = class_type @Outer, @Outer(constants.%empty_tuple.type) [concrete = constants.%Outer.54c]
// CHECK:STDOUT:   %.loc61_12: %Outer.G.type.08d = specific_constant @Outer.%Outer.G.decl, @Outer(constants.%empty_tuple.type) [concrete = constants.%Outer.G.e87]
// CHECK:STDOUT:   %G.ref: %Outer.G.type.08d = name_ref G, %.loc61_12 [concrete = constants.%Outer.G.e87]
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %facet_value: %Y_where.type.6ce = facet_value constants.%C, (constants.%Y.impl_witness) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc61_16: %Y_where.type.6ce = converted constants.%C, %facet_value [concrete = constants.%facet_value]
// CHECK:STDOUT:   %Outer.G.specific_fn: <specific function> = specific_function %G.ref, @Outer.G(constants.%empty_tuple.type, constants.%facet_value) [concrete = constants.%Outer.G.specific_fn]
// CHECK:STDOUT:   %Outer.G.call: init %empty_tuple.type = call %Outer.G.specific_fn()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%OuterParam) {
// CHECK:STDOUT:   %OuterParam.patt.loc49_23.2 => constants.%OuterParam.patt
// CHECK:STDOUT:   %OuterParam.loc49_23.1 => constants.%OuterParam
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Y.type => constants.%Y.type.eff
// CHECK:STDOUT:   %Outer.G.type => constants.%Outer.G.type.524
// CHECK:STDOUT:   %Outer.G => constants.%Outer.G.42e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y(constants.%OuterParam) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %OuterParam => constants.%OuterParam
// CHECK:STDOUT:   %Y.type => constants.%Y.type.eff
// CHECK:STDOUT:   %Self.loc50_15.2 => constants.%Self.f65
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y.WithSelf(constants.%OuterParam, constants.%Self.f65) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %OuterParam => constants.%OuterParam
// CHECK:STDOUT:   %Y.assoc_type => constants.%Y.assoc_type.edf
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.05f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y.WithSelf(constants.%OuterParam, constants.%.Self.frozen.b95) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %OuterParam => constants.%OuterParam
// CHECK:STDOUT:   %Y.assoc_type => constants.%Y.assoc_type.edf
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.05f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer.G(constants.%OuterParam, constants.%H) {
// CHECK:STDOUT:   %OuterParam => constants.%OuterParam
// CHECK:STDOUT:   %Y.type => constants.%Y.type.eff
// CHECK:STDOUT:   %.Self.frozen.loc54_28.1 => constants.%.Self.frozen.b95
// CHECK:STDOUT:   %require_complete => constants.%require_complete
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.55f
// CHECK:STDOUT:   %Y.assoc_type => constants.%Y.assoc_type.edf
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.05f
// CHECK:STDOUT:   %Y.lookup_impl_witness.loc54_34.1 => constants.%Y.lookup_impl_witness.eba
// CHECK:STDOUT:   %impl.elem0.loc54_34.1 => constants.%impl.elem0.29d
// CHECK:STDOUT:   %.Self => constants.%.Self.273
// CHECK:STDOUT:   %Y.lookup_impl_witness.loc54_34.2 => constants.%Y.lookup_impl_witness.f46
// CHECK:STDOUT:   %impl.elem0.loc54_34.2 => constants.%impl.elem0.d7a
// CHECK:STDOUT:   %Y_where.type => constants.%Y_where.type.fa0
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.b98
// CHECK:STDOUT:   %H.patt.loc54_24.2 => constants.%H.patt.314
// CHECK:STDOUT:   %H.loc54_24.1 => constants.%H
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%empty_tuple.type) {
// CHECK:STDOUT:   %OuterParam.patt.loc49_23.2 => constants.%OuterParam.patt
// CHECK:STDOUT:   %OuterParam.loc49_23.1 => constants.%empty_tuple.type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Y.type => constants.%Y.type.d7b
// CHECK:STDOUT:   %Outer.G.type => constants.%Outer.G.type.08d
// CHECK:STDOUT:   %Outer.G => constants.%Outer.G.e87
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y(constants.%empty_tuple.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %OuterParam => constants.%empty_tuple.type
// CHECK:STDOUT:   %Y.type => constants.%Y.type.d7b
// CHECK:STDOUT:   %Self.loc50_15.2 => constants.%Self.caf
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y.WithSelf(constants.%empty_tuple.type, constants.%Self.f65) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %OuterParam => constants.%empty_tuple.type
// CHECK:STDOUT:   %Y.assoc_type => constants.%Y.assoc_type.7d9
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.559
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y.WithSelf(constants.%empty_tuple.type, constants.%.Self.frozen.62f) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %OuterParam => constants.%empty_tuple.type
// CHECK:STDOUT:   %Y.assoc_type => constants.%Y.assoc_type.7d9
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.559
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y.WithSelf(constants.%empty_tuple.type, constants.%Y.facet.8dd) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %OuterParam => constants.%empty_tuple.type
// CHECK:STDOUT:   %Y.assoc_type => constants.%Y.assoc_type.7d9
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.559
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer.G(constants.%empty_tuple.type, constants.%facet_value) {
// CHECK:STDOUT:   %OuterParam => constants.%empty_tuple.type
// CHECK:STDOUT:   %Y.type => constants.%Y.type.d7b
// CHECK:STDOUT:   %.Self.frozen.loc54_28.1 => constants.%.Self.frozen.62f
// CHECK:STDOUT:   %require_complete => constants.%complete_type.2cf
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.a06
// CHECK:STDOUT:   %Y.assoc_type => constants.%Y.assoc_type.7d9
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.559
// CHECK:STDOUT:   %Y.lookup_impl_witness.loc54_34.1 => constants.%Y.lookup_impl_witness.e35
// CHECK:STDOUT:   %impl.elem0.loc54_34.1 => constants.%impl.elem0.798
// CHECK:STDOUT:   %.Self => constants.%.Self.37b
// CHECK:STDOUT:   %Y.lookup_impl_witness.loc54_34.2 => constants.%Y.lookup_impl_witness.b19
// CHECK:STDOUT:   %impl.elem0.loc54_34.2 => constants.%impl.elem0.b95
// CHECK:STDOUT:   %Y_where.type => constants.%Y_where.type.6ce
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.f97
// CHECK:STDOUT:   %H.patt.loc54_24.2 => constants.%H.patt.7ee
// CHECK:STDOUT:   %H.loc54_24.1 => constants.%facet_value
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- enclosing_specific_is_facet_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Z1.type: type = facet_type <@Z1> [concrete]
// CHECK:STDOUT:   %Self.bc4: %Z1.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Z2.type: type = facet_type <@Z2> [concrete]
// CHECK:STDOUT:   %Self.5ad: %Z2.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %.Self.frozen.e58: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.f4f: type = pattern_type %Z1.type [concrete]
// CHECK:STDOUT:   %OuterParam.patt: %pattern_type.f4f = symbolic_binding_pattern OuterParam, 0 [symbolic]
// CHECK:STDOUT:   %OuterParam: %Z1.type = symbolic_binding OuterParam, 0 [symbolic]
// CHECK:STDOUT:   %Outer.type: type = generic_class_type @Outer [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Outer.generic: %Outer.type = struct_value () [concrete]
// CHECK:STDOUT:   %Outer.bbd: type = class_type @Outer, @Outer(%OuterParam) [symbolic]
// CHECK:STDOUT:   %Y.type.9a0: type = facet_type <@Y, @Y(%OuterParam)> [symbolic]
// CHECK:STDOUT:   %Self.d06: %Y.type.9a0 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Y.assoc_type.5d5: type = assoc_entity_type @Y, @Y(%OuterParam) [symbolic]
// CHECK:STDOUT:   %assoc0.b45: %Y.assoc_type.5d5 = assoc_entity element0, @Y.WithSelf.%T [symbolic]
// CHECK:STDOUT:   %.Self.frozen.39e: %Y.type.9a0 = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Y.type.9a0 [symbolic]
// CHECK:STDOUT:   %.Self.frozen.as_type.618: type = facet_access_type %.Self.frozen.39e [symbolic]
// CHECK:STDOUT:   %Y.lookup_impl_witness.580: <witness> = lookup_impl_witness %.Self.frozen.39e, @Y, @Y(%OuterParam) [symbolic]
// CHECK:STDOUT:   %impl.elem0.be8: type = impl_witness_access %Y.lookup_impl_witness.580, element0 [symbolic]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %.Self.2b6: %Y.type.9a0 = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %Y.lookup_impl_witness.00e: <witness> = lookup_impl_witness %.Self.2b6, @Y, @Y(%OuterParam) [symbolic]
// CHECK:STDOUT:   %impl.elem0.43a: type = impl_witness_access %Y.lookup_impl_witness.00e, element0 [symbolic]
// CHECK:STDOUT:   %Y_where.type.352: type = facet_type <@Y, @Y(%OuterParam) where %impl.elem0.43a = %empty_tuple.type> [symbolic]
// CHECK:STDOUT:   %pattern_type.8be: type = pattern_type %Y_where.type.352 [symbolic]
// CHECK:STDOUT:   %H.patt.57a: %pattern_type.8be = symbolic_binding_pattern H, 1 [symbolic]
// CHECK:STDOUT:   %H: %Y_where.type.352 = symbolic_binding H, 1 [symbolic]
// CHECK:STDOUT:   %Outer.G.type.b7b: type = fn_type @Outer.G, @Outer(%OuterParam) [symbolic]
// CHECK:STDOUT:   %Outer.G.27c: %Outer.G.type.b7b = struct_value () [symbolic]
// CHECK:STDOUT:   %C.5f2: type = class_type @C, @C(%OuterParam) [symbolic]
// CHECK:STDOUT:   %.eeb: <witness> = impl_self_witness %.Self.frozen.39e, @Y, @Y(%OuterParam) [symbolic]
// CHECK:STDOUT:   %impl.elem0.756: type = impl_witness_access %.eeb, element0 [symbolic]
// CHECK:STDOUT:   %.66e: <witness> = impl_self_witness %.Self.2b6, @Y, @Y(%OuterParam) [symbolic]
// CHECK:STDOUT:   %impl.elem0.295: type = impl_witness_access %.66e, element0 [symbolic]
// CHECK:STDOUT:   %Y_where.type.aa4: type = facet_type <@Y, @Y(%OuterParam) where %impl.elem0.295 = %empty_tuple.type> [symbolic]
// CHECK:STDOUT:   %.6b3: <witness> = impl_self_witness %C.5f2, @Y, @Y(%OuterParam) [symbolic]
// CHECK:STDOUT:   %Y.impl_witness.d75: <witness> = impl_witness @C.as.Y.impl.%Y.impl_witness_table, @C.as.Y.impl(%OuterParam) [symbolic]
// CHECK:STDOUT:   %Y.facet.44a: %Y.type.9a0 = facet_value %C.5f2, (%Y.impl_witness.d75) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %.9a5: <witness> = impl_self_witness %empty_tuple.type, @Z1 [concrete]
// CHECK:STDOUT:   %Z1.impl_witness: <witness> = impl_witness @empty_tuple.type.as.Z1.impl.%Z1.impl_witness_table [concrete]
// CHECK:STDOUT:   %Z1.facet: %Z1.type = facet_value %empty_tuple.type, (%Z1.impl_witness) [concrete]
// CHECK:STDOUT:   %.6bb: <witness> = impl_self_witness %empty_tuple.type, @Z2 [concrete]
// CHECK:STDOUT:   %Z2.impl_witness: <witness> = impl_witness @empty_tuple.type.as.Z2.impl.%Z2.impl_witness_table [concrete]
// CHECK:STDOUT:   %Z2.facet: %Z2.type = facet_value %empty_tuple.type, (%Z2.impl_witness) [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndWith.type.a06: type = generic_interface_type @BitAndWith [concrete]
// CHECK:STDOUT:   %BitAndWith.generic: %BitAndWith.type.a06 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndWith.type.7f7: type = facet_type <@BitAndWith, @BitAndWith(type)> [concrete]
// CHECK:STDOUT:   %BitAndWith.impl_witness: <witness> = impl_witness imports.%BitAndWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %BitAndWith.facet: %BitAndWith.type.7f7 = facet_value type, (%BitAndWith.impl_witness) [concrete]
// CHECK:STDOUT:   %BitAndWith.WithSelf.Op.type.1a6: type = fn_type @BitAndWith.WithSelf.Op, @BitAndWith.WithSelf(type, %BitAndWith.facet) [concrete]
// CHECK:STDOUT:   %.b4e: type = fn_type_with_self_type %BitAndWith.WithSelf.Op.type.1a6, %BitAndWith.facet [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.type: type = fn_type @type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op: %type.as.BitAndWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.bound: <bound method> = bound_method %Z1.type, %type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@Z1 & @Z2> [concrete]
// CHECK:STDOUT:   %facet_value.794: %facet_type = facet_value %empty_tuple.type, (%Z1.impl_witness, %Z2.impl_witness) [concrete]
// CHECK:STDOUT:   %Outer.01a: type = class_type @Outer, @Outer(%Z1.facet) [concrete]
// CHECK:STDOUT:   %Y.type.30f: type = facet_type <@Y, @Y(%Z1.facet)> [concrete]
// CHECK:STDOUT:   %Outer.G.type.e1d: type = fn_type @Outer.G, @Outer(%Z1.facet) [concrete]
// CHECK:STDOUT:   %Outer.G.163: %Outer.G.type.e1d = struct_value () [concrete]
// CHECK:STDOUT:   %C.d9e: type = class_type @C, @C(%Z1.facet) [concrete]
// CHECK:STDOUT:   %.Self.1da: %Y.type.30f = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Y.lookup_impl_witness.0bd: <witness> = lookup_impl_witness %.Self.1da, @Y, @Y(%Z1.facet) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.186: type = impl_witness_access %Y.lookup_impl_witness.0bd, element0 [symbolic_self]
// CHECK:STDOUT:   %Y_where.type.677: type = facet_type <@Y, @Y(%Z1.facet) where %impl.elem0.186 = %empty_tuple.type> [concrete]
// CHECK:STDOUT:   %.Self.frozen.126: %Y.type.30f = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Self.1f8: %Y.type.30f = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Y.assoc_type.104: type = assoc_entity_type @Y, @Y(%Z1.facet) [concrete]
// CHECK:STDOUT:   %assoc0.2bd: %Y.assoc_type.104 = assoc_entity element0, @Y.WithSelf.%T [concrete]
// CHECK:STDOUT:   %complete_type.d0d: <witness> = complete_type_witness %Y.type.30f [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.746: type = facet_access_type %.Self.frozen.126 [symbolic_self]
// CHECK:STDOUT:   %.03e: <witness> = impl_self_witness %.Self.frozen.126, @Y, @Y(%Z1.facet) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.ad3: type = impl_witness_access %.03e, element0 [symbolic_self]
// CHECK:STDOUT:   %.4bb: <witness> = impl_self_witness %.Self.1da, @Y, @Y(%Z1.facet) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.eb1: type = impl_witness_access %.4bb, element0 [symbolic_self]
// CHECK:STDOUT:   %Y_where.type.d3d: type = facet_type <@Y, @Y(%Z1.facet) where %impl.elem0.eb1 = %empty_tuple.type> [concrete]
// CHECK:STDOUT:   %.9bc: <witness> = impl_self_witness %C.d9e, @Y, @Y(%Z1.facet) [concrete]
// CHECK:STDOUT:   %Y.impl_witness.1b2: <witness> = impl_witness @C.as.Y.impl.%Y.impl_witness_table, @C.as.Y.impl(%Z1.facet) [concrete]
// CHECK:STDOUT:   %facet_value.c7c: %Y_where.type.677 = facet_value %C.d9e, (%Y.impl_witness.1b2) [concrete]
// CHECK:STDOUT:   %Y.lookup_impl_witness.a66: <witness> = lookup_impl_witness %.Self.frozen.126, @Y, @Y(%Z1.facet) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.e0b: type = impl_witness_access %Y.lookup_impl_witness.a66, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.285: type = pattern_type %Y_where.type.677 [concrete]
// CHECK:STDOUT:   %H.patt.305: %pattern_type.285 = symbolic_binding_pattern H, 1 [symbolic]
// CHECK:STDOUT:   %Outer.G.specific_fn: <specific function> = specific_function %Outer.G.163, @Outer.G(%Z1.facet, %facet_value.c7c) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .BitAndWith = %Core.BitAndWith
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.BitAndWith: %BitAndWith.type.a06 = import_ref Core//prelude/parts/as, BitAndWith, loaded [concrete = constants.%BitAndWith.generic]
// CHECK:STDOUT:   %Core.import_ref.9c3: %type.as.BitAndWith.impl.Op.type = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, loaded [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:   %BitAndWith.impl_witness_table = impl_witness_table (%Core.import_ref.9c3), @type.as.BitAndWith.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Z1 = %Z1.decl
// CHECK:STDOUT:     .Z2 = %Z2.decl
// CHECK:STDOUT:     .Outer = %Outer.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Z1.decl: type = interface_decl @Z1 [concrete = constants.%Z1.type] {} {}
// CHECK:STDOUT:   %Z2.decl: type = interface_decl @Z2 [concrete = constants.%Z2.type] {} {}
// CHECK:STDOUT:   %Outer.decl: %Outer.type = class_decl @Outer [concrete = constants.%Outer.generic] {
// CHECK:STDOUT:     %OuterParam.patt.loc6_23.1: %pattern_type.f4f = symbolic_binding_pattern OuterParam, 0 [symbolic = %OuterParam.patt.loc6_23.2 (constants.%OuterParam.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6: type = splice_block %Z1.ref [concrete = constants.%Z1.type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.e58]
// CHECK:STDOUT:       %Z1.ref: type = name_ref Z1, file.%Z1.decl [concrete = constants.%Z1.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %OuterParam.loc6_23.2: %Z1.type = symbolic_binding OuterParam, 0 [symbolic = %OuterParam.loc6_23.1 (constants.%OuterParam)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @empty_tuple.type.as.Z1.impl [concrete] {} {
// CHECK:STDOUT:     %.loc17_7.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc17_7.2: type = converted %.loc17_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %Z1.ref: type = name_ref Z1, file.%Z1.decl [concrete = constants.%Z1.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc17: <witness> = impl_self_witness @empty_tuple.type.as.Z1.impl.%.loc17_7.2, @Z1 [concrete = constants.%.9a5]
// CHECK:STDOUT:   impl_decl @empty_tuple.type.as.Z2.impl [concrete] {} {
// CHECK:STDOUT:     %.loc18_7.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc18_7.2: type = converted %.loc18_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %Z2.ref: type = name_ref Z2, file.%Z2.decl [concrete = constants.%Z2.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc18: <witness> = impl_self_witness @empty_tuple.type.as.Z2.impl.%.loc18_7.2, @Z2 [concrete = constants.%.6bb]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Z1 {
// CHECK:STDOUT:   %Self: %Z1.type = symbolic_binding Self, 0 [symbolic = constants.%Self.bc4]
// CHECK:STDOUT:   %Z1.WithSelf.decl = interface_with_self_decl @Z1 [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Z2 {
// CHECK:STDOUT:   %Self: %Z2.type = symbolic_binding Self, 0 [symbolic = constants.%Self.5ad]
// CHECK:STDOUT:   %Z2.WithSelf.decl = interface_with_self_decl @Z2 [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Y(@Outer.%OuterParam.loc6_23.2: %Z1.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %OuterParam: %Z1.type = symbolic_binding OuterParam, 0 [symbolic = %OuterParam (constants.%OuterParam)]
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y, @Y(%OuterParam)> [symbolic = %Y.type (constants.%Y.type.9a0)]
// CHECK:STDOUT:   %Self.loc7_15.2: @Y.%Y.type (%Y.type.9a0) = symbolic_binding Self, 1 [symbolic = %Self.loc7_15.2 (constants.%Self.d06)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc7_15.1: @Y.%Y.type (%Y.type.9a0) = symbolic_binding Self, 1 [symbolic = %Self.loc7_15.2 (constants.%Self.d06)]
// CHECK:STDOUT:     %Y.WithSelf.decl = interface_with_self_decl @Y [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %T: type = assoc_const_decl @T [concrete] {
// CHECK:STDOUT:       %assoc0: @Y.WithSelf.%Y.assoc_type (%Y.assoc_type.5d5) = assoc_entity element0, @Y.WithSelf.%T [symbolic = @Y.WithSelf.%assoc0 (constants.%assoc0.b45)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc7_15.1
// CHECK:STDOUT:     .T = @T.%assoc0
// CHECK:STDOUT:     witness = (@Y.WithSelf.%T)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.Y.impl(@Outer.%OuterParam.loc6_23.2: %Z1.type) {
// CHECK:STDOUT:   %OuterParam: %Z1.type = symbolic_binding OuterParam, 0 [symbolic = %OuterParam (constants.%OuterParam)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%OuterParam) [symbolic = %C (constants.%C.5f2)]
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y, @Y(%OuterParam)> [symbolic = %Y.type (constants.%Y.type.9a0)]
// CHECK:STDOUT:   %.Self.frozen.loc14_15.2: @C.as.Y.impl.%Y.type (%Y.type.9a0) = symbolic_binding .Self [symbolic = %.Self.frozen.loc14_15.2 (constants.%.Self.frozen.39e)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Y.type [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %.Self.frozen.as_type: type = facet_access_type %.Self.frozen.loc14_15.2 [symbolic = %.Self.frozen.as_type (constants.%.Self.frozen.as_type.618)]
// CHECK:STDOUT:   %Y.assoc_type: type = assoc_entity_type @Y, @Y(%OuterParam) [symbolic = %Y.assoc_type (constants.%Y.assoc_type.5d5)]
// CHECK:STDOUT:   %assoc0: @C.as.Y.impl.%Y.assoc_type (%Y.assoc_type.5d5) = assoc_entity element0, @Y.WithSelf.%T [symbolic = %assoc0 (constants.%assoc0.b45)]
// CHECK:STDOUT:   %.loc14_21.3: <witness> = impl_self_witness %.Self.frozen.loc14_15.2, @Y, @Y(%OuterParam) [symbolic = %.loc14_21.3 (constants.%.eeb)]
// CHECK:STDOUT:   %impl.elem0.loc14_21.3: type = impl_witness_access %.loc14_21.3, element0 [symbolic = %impl.elem0.loc14_21.3 (constants.%impl.elem0.756)]
// CHECK:STDOUT:   %.Self: @C.as.Y.impl.%Y.type (%Y.type.9a0) = symbolic_binding .Self [symbolic = %.Self (constants.%.Self.2b6)]
// CHECK:STDOUT:   %.loc14_21.4: <witness> = impl_self_witness %.Self, @Y, @Y(%OuterParam) [symbolic = %.loc14_21.4 (constants.%.66e)]
// CHECK:STDOUT:   %impl.elem0.loc14_21.4: type = impl_witness_access %.loc14_21.4, element0 [symbolic = %impl.elem0.loc14_21.4 (constants.%impl.elem0.295)]
// CHECK:STDOUT:   %Y_where.type: type = facet_type <@Y, @Y(%OuterParam) where %impl.elem0.loc14_21.4 = constants.%empty_tuple.type> [symbolic = %Y_where.type (constants.%Y_where.type.aa4)]
// CHECK:STDOUT:   %.loc14_29: <witness> = impl_self_witness %C, @Y, @Y(%OuterParam) [symbolic = %.loc14_29 (constants.%.6b3)]
// CHECK:STDOUT:   %Y.impl_witness.loc14_29.2: <witness> = impl_witness %Y.impl_witness_table, @C.as.Y.impl(%OuterParam) [symbolic = %Y.impl_witness.loc14_29.2 (constants.%Y.impl_witness.d75)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref as %Y.ref {
// CHECK:STDOUT:     %Y.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @C.as.Y.impl [concrete]
// CHECK:STDOUT:     %Y.impl_witness.loc14_29.1: <witness> = impl_witness %Y.impl_witness_table, @C.as.Y.impl(constants.%OuterParam) [symbolic = %Y.impl_witness.loc14_29.2 (constants.%Y.impl_witness.d75)]
// CHECK:STDOUT:     %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     extend %Y.ref
// CHECK:STDOUT:     witness = %Y.impl_witness.loc14_29.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @empty_tuple.type.as.Z1.impl: %.loc17_7.2 as %Z1.ref {
// CHECK:STDOUT:   %Z1.impl_witness_table = impl_witness_table (), @empty_tuple.type.as.Z1.impl [concrete]
// CHECK:STDOUT:   %Z1.impl_witness: <witness> = impl_witness %Z1.impl_witness_table [concrete = constants.%Z1.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend %Z1.ref
// CHECK:STDOUT:   witness = %Z1.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @empty_tuple.type.as.Z2.impl: %.loc18_7.2 as %Z2.ref {
// CHECK:STDOUT:   %Z2.impl_witness_table = impl_witness_table (), @empty_tuple.type.as.Z2.impl [concrete]
// CHECK:STDOUT:   %Z2.impl_witness: <witness> = impl_witness %Z2.impl_witness_table [concrete = constants.%Z2.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend %Z2.ref
// CHECK:STDOUT:   witness = %Z2.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Outer(%OuterParam.loc6_23.2: %Z1.type) {
// CHECK:STDOUT:   %OuterParam.patt.loc6_23.2: %pattern_type.f4f = symbolic_binding_pattern OuterParam, 0 [symbolic = %OuterParam.patt.loc6_23.2 (constants.%OuterParam.patt)]
// CHECK:STDOUT:   %OuterParam.loc6_23.1: %Z1.type = symbolic_binding OuterParam, 0 [symbolic = %OuterParam.loc6_23.1 (constants.%OuterParam)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y, @Y(%OuterParam.loc6_23.1)> [symbolic = %Y.type (constants.%Y.type.9a0)]
// CHECK:STDOUT:   %Outer.G.type: type = fn_type @Outer.G, @Outer(%OuterParam.loc6_23.1) [symbolic = %Outer.G.type (constants.%Outer.G.type.b7b)]
// CHECK:STDOUT:   %Outer.G: @Outer.%Outer.G.type (%Outer.G.type.b7b) = struct_value () [symbolic = %Outer.G (constants.%Outer.G.27c)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%OuterParam.loc6_23.1) [symbolic = %C (constants.%C.5f2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Y.decl: type = interface_decl @Y [symbolic = @Outer.%Y.type (constants.%Y.type.9a0)] {} {}
// CHECK:STDOUT:     %Outer.G.decl: @Outer.%Outer.G.type (%Outer.G.type.b7b) = fn_decl @Outer.G [symbolic = @Outer.%Outer.G (constants.%Outer.G.27c)] {
// CHECK:STDOUT:       %H.patt.loc11_24.1: @Outer.G.%pattern_type (%pattern_type.8be) = symbolic_binding_pattern H, 1 [symbolic = %H.patt.loc11_24.2 (constants.%H.patt.57a)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc11_28.1: type = splice_block %.loc11_28.2 [symbolic = %Y_where.type (constants.%Y_where.type.352)] {
// CHECK:STDOUT:         %.Self.frozen.loc11_24: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.e58]
// CHECK:STDOUT:         %.loc11_26: type = specific_constant @Outer.%Y.decl, @Outer(constants.%OuterParam) [symbolic = %Y.type (constants.%Y.type.9a0)]
// CHECK:STDOUT:         %Y.ref: type = name_ref Y, %.loc11_26 [symbolic = %Y.type (constants.%Y.type.9a0)]
// CHECK:STDOUT:         %.Self.frozen.loc11_28.2: @Outer.G.%Y.type (%Y.type.9a0) = symbolic_binding .Self [symbolic = %.Self.frozen.loc11_28.1 (constants.%.Self.frozen.39e)]
// CHECK:STDOUT:         %base_facet_type = requirement_base_facet_type %Y.ref [concrete]
// CHECK:STDOUT:         %.Self.ref: @Outer.G.%Y.type (%Y.type.9a0) = name_ref .Self, %.Self.frozen.loc11_28.2 [symbolic = %.Self.frozen.loc11_28.1 (constants.%.Self.frozen.39e)]
// CHECK:STDOUT:         %.Self.as_type: type = facet_access_type %.Self.ref [symbolic = %.Self.frozen.as_type (constants.%.Self.frozen.as_type.618)]
// CHECK:STDOUT:         %.loc11_34.1: type = converted %.Self.ref, %.Self.as_type [symbolic = %.Self.frozen.as_type (constants.%.Self.frozen.as_type.618)]
// CHECK:STDOUT:         %.loc11_34.2: @Outer.G.%Y.assoc_type (%Y.assoc_type.5d5) = specific_constant @T.%assoc0, @Y.WithSelf(constants.%OuterParam, constants.%.Self.frozen.39e) [symbolic = %assoc0 (constants.%assoc0.b45)]
// CHECK:STDOUT:         %T.ref: @Outer.G.%Y.assoc_type (%Y.assoc_type.5d5) = name_ref T, %.loc11_34.2 [symbolic = %assoc0 (constants.%assoc0.b45)]
// CHECK:STDOUT:         %impl.elem0.loc11_34.3: type = impl_witness_access constants.%Y.lookup_impl_witness.580, element0 [symbolic = %impl.elem0.loc11_34.1 (constants.%impl.elem0.be8)]
// CHECK:STDOUT:         %.loc11_40.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:         %.loc11_40.2: type = converted %.loc11_40.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:         %rewrite.loc11_37.1 = requirement_rewrite %impl.elem0.loc11_34.3, %.loc11_40.2 [concrete]
// CHECK:STDOUT:         %impl.elem0.loc11_34.4: type = impl_witness_access constants.%Y.lookup_impl_witness.00e, element0 [symbolic = %impl.elem0.loc11_34.2 (constants.%impl.elem0.43a)]
// CHECK:STDOUT:         %rewrite.loc11_37.2 = requirement_rewrite %impl.elem0.loc11_34.4, %.loc11_40.2 [concrete]
// CHECK:STDOUT:         %.loc11_28.2: type = where_expr [symbolic = %Y_where.type (constants.%Y_where.type.352)] {
// CHECK:STDOUT:           %base_facet_type = requirement_base_facet_type %Y.ref [concrete]
// CHECK:STDOUT:           %rewrite.loc11_37.2 = requirement_rewrite %impl.elem0.loc11_34.4, %.loc11_40.2 [concrete]
// CHECK:STDOUT:         }
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %H.loc11_24.2: @Outer.G.%Y_where.type (%Y_where.type.352) = symbolic_binding H, 1 [symbolic = %H.loc11_24.1 (constants.%H)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %C.decl: type = class_decl @C [symbolic = @Outer.%C (constants.%C.5f2)] {} {}
// CHECK:STDOUT:     impl_decl @C.as.Y.impl [concrete] {} {
// CHECK:STDOUT:       %.loc14_8: type = specific_constant @Outer.%C.decl, @Outer(constants.%OuterParam) [symbolic = %C (constants.%C.5f2)]
// CHECK:STDOUT:       %C.ref: type = name_ref C, %.loc14_8 [symbolic = %C (constants.%C.5f2)]
// CHECK:STDOUT:       %.loc14_13: type = specific_constant @Outer.%Y.decl, @Outer(constants.%OuterParam) [symbolic = %Y.type (constants.%Y.type.9a0)]
// CHECK:STDOUT:       %Y.ref: type = name_ref Y, %.loc14_13 [symbolic = %Y.type (constants.%Y.type.9a0)]
// CHECK:STDOUT:       %.Self.frozen.loc14_15.1: @C.as.Y.impl.%Y.type (%Y.type.9a0) = symbolic_binding .Self [symbolic = %.Self.frozen.loc14_15.2 (constants.%.Self.frozen.39e)]
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %Y.ref [concrete]
// CHECK:STDOUT:       %.Self.ref: @C.as.Y.impl.%Y.type (%Y.type.9a0) = name_ref .Self, %.Self.frozen.loc14_15.1 [symbolic = %.Self.frozen.loc14_15.2 (constants.%.Self.frozen.39e)]
// CHECK:STDOUT:       %.Self.as_type: type = facet_access_type %.Self.ref [symbolic = %.Self.frozen.as_type (constants.%.Self.frozen.as_type.618)]
// CHECK:STDOUT:       %.loc14_21.1: type = converted %.Self.ref, %.Self.as_type [symbolic = %.Self.frozen.as_type (constants.%.Self.frozen.as_type.618)]
// CHECK:STDOUT:       %.loc14_21.2: @C.as.Y.impl.%Y.assoc_type (%Y.assoc_type.5d5) = specific_constant @T.%assoc0, @Y.WithSelf(constants.%OuterParam, constants.%.Self.frozen.39e) [symbolic = %assoc0 (constants.%assoc0.b45)]
// CHECK:STDOUT:       %T.ref: @C.as.Y.impl.%Y.assoc_type (%Y.assoc_type.5d5) = name_ref T, %.loc14_21.2 [symbolic = %assoc0 (constants.%assoc0.b45)]
// CHECK:STDOUT:       %impl.elem0.loc14_21.1: type = impl_witness_access constants.%.eeb, element0 [symbolic = %impl.elem0.loc14_21.3 (constants.%impl.elem0.756)]
// CHECK:STDOUT:       %.loc14_27.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc14_27.2: type = converted %.loc14_27.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %rewrite.loc14_24.1 = requirement_rewrite %impl.elem0.loc14_21.1, %.loc14_27.2 [concrete]
// CHECK:STDOUT:       %impl.elem0.loc14_21.2: type = impl_witness_access constants.%.66e, element0 [symbolic = %impl.elem0.loc14_21.4 (constants.%impl.elem0.295)]
// CHECK:STDOUT:       %rewrite.loc14_24.2 = requirement_rewrite %impl.elem0.loc14_21.2, %.loc14_27.2 [concrete]
// CHECK:STDOUT:       %.loc14_15: type = where_expr [symbolic = %Y_where.type (constants.%Y_where.type.aa4)] {
// CHECK:STDOUT:         %base_facet_type = requirement_base_facet_type %Y.ref [concrete]
// CHECK:STDOUT:         %rewrite.loc14_24.2 = requirement_rewrite %impl.elem0.loc14_21.2, %.loc14_27.2 [concrete]
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc14: <witness> = impl_self_witness @C.as.Y.impl.%C.ref, @Y, @Y(constants.%OuterParam) [symbolic = @C.as.Y.impl.%.loc14_29 (constants.%.6b3)]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Outer.bbd
// CHECK:STDOUT:     .Y = %Y.decl
// CHECK:STDOUT:     .G = %Outer.G.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(@Outer.%OuterParam.loc6_23.2: %Z1.type) {
// CHECK:STDOUT:   class;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Outer.G(@Outer.%OuterParam.loc6_23.2: %Z1.type, %H.loc11_24.2: @Outer.G.%Y_where.type (%Y_where.type.352)) {
// CHECK:STDOUT:   %OuterParam: %Z1.type = symbolic_binding OuterParam, 0 [symbolic = %OuterParam (constants.%OuterParam)]
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y, @Y(%OuterParam)> [symbolic = %Y.type (constants.%Y.type.9a0)]
// CHECK:STDOUT:   %.Self.frozen.loc11_28.1: @Outer.G.%Y.type (%Y.type.9a0) = symbolic_binding .Self [symbolic = %.Self.frozen.loc11_28.1 (constants.%.Self.frozen.39e)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Y.type [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %.Self.frozen.as_type: type = facet_access_type %.Self.frozen.loc11_28.1 [symbolic = %.Self.frozen.as_type (constants.%.Self.frozen.as_type.618)]
// CHECK:STDOUT:   %Y.assoc_type: type = assoc_entity_type @Y, @Y(%OuterParam) [symbolic = %Y.assoc_type (constants.%Y.assoc_type.5d5)]
// CHECK:STDOUT:   %assoc0: @Outer.G.%Y.assoc_type (%Y.assoc_type.5d5) = assoc_entity element0, @Y.WithSelf.%T [symbolic = %assoc0 (constants.%assoc0.b45)]
// CHECK:STDOUT:   %Y.lookup_impl_witness.loc11_34.1: <witness> = lookup_impl_witness %.Self.frozen.loc11_28.1, @Y, @Y(%OuterParam) [symbolic = %Y.lookup_impl_witness.loc11_34.1 (constants.%Y.lookup_impl_witness.580)]
// CHECK:STDOUT:   %impl.elem0.loc11_34.1: type = impl_witness_access %Y.lookup_impl_witness.loc11_34.1, element0 [symbolic = %impl.elem0.loc11_34.1 (constants.%impl.elem0.be8)]
// CHECK:STDOUT:   %.Self: @Outer.G.%Y.type (%Y.type.9a0) = symbolic_binding .Self [symbolic = %.Self (constants.%.Self.2b6)]
// CHECK:STDOUT:   %Y.lookup_impl_witness.loc11_34.2: <witness> = lookup_impl_witness %.Self, @Y, @Y(%OuterParam) [symbolic = %Y.lookup_impl_witness.loc11_34.2 (constants.%Y.lookup_impl_witness.00e)]
// CHECK:STDOUT:   %impl.elem0.loc11_34.2: type = impl_witness_access %Y.lookup_impl_witness.loc11_34.2, element0 [symbolic = %impl.elem0.loc11_34.2 (constants.%impl.elem0.43a)]
// CHECK:STDOUT:   %Y_where.type: type = facet_type <@Y, @Y(%OuterParam) where %impl.elem0.loc11_34.2 = constants.%empty_tuple.type> [symbolic = %Y_where.type (constants.%Y_where.type.352)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Y_where.type [symbolic = %pattern_type (constants.%pattern_type.8be)]
// CHECK:STDOUT:   %H.patt.loc11_24.2: @Outer.G.%pattern_type (%pattern_type.8be) = symbolic_binding_pattern H, 1 [symbolic = %H.patt.loc11_24.2 (constants.%H.patt.57a)]
// CHECK:STDOUT:   %H.loc11_24.1: @Outer.G.%Y_where.type (%Y_where.type.352) = symbolic_binding H, 1 [symbolic = %H.loc11_24.1 (constants.%H)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Outer.ref.loc21_3: %Outer.type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:   %.loc21_10: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %Z1.ref.loc21_16: type = name_ref Z1, file.%Z1.decl [concrete = constants.%Z1.type]
// CHECK:STDOUT:   %Z2.ref.loc21_21: type = name_ref Z2, file.%Z2.decl [concrete = constants.%Z2.type]
// CHECK:STDOUT:   %impl.elem0.loc21_19: %.b4e = impl_witness_access constants.%BitAndWith.impl_witness, element0 [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:   %bound_method.loc21_19: <bound method> = bound_method %Z1.ref.loc21_16, %impl.elem0.loc21_19 [concrete = constants.%type.as.BitAndWith.impl.Op.bound]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.call.loc21_19: init type = call %bound_method.loc21_19(%Z1.ref.loc21_16, %Z2.ref.loc21_21) [concrete = constants.%facet_type]
// CHECK:STDOUT:   %.loc21_23.1: type = value_of_initializer %type.as.BitAndWith.impl.Op.call.loc21_19 [concrete = constants.%facet_type]
// CHECK:STDOUT:   %.loc21_23.2: type = converted %type.as.BitAndWith.impl.Op.call.loc21_19, %.loc21_23.1 [concrete = constants.%facet_type]
// CHECK:STDOUT:   %facet_value.loc21_12: %facet_type = facet_value constants.%empty_tuple.type, (constants.%Z1.impl_witness, constants.%Z2.impl_witness) [concrete = constants.%facet_value.794]
// CHECK:STDOUT:   %.loc21_12: %facet_type = converted %.loc21_10, %facet_value.loc21_12 [concrete = constants.%facet_value.794]
// CHECK:STDOUT:   %as_type.loc21_24: type = facet_access_type %.loc21_12 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %Z1.facet.loc21_24: %Z1.type = facet_value %as_type.loc21_24, (constants.%Z1.impl_witness) [concrete = constants.%Z1.facet]
// CHECK:STDOUT:   %.loc21_24: %Z1.type = converted %.loc21_12, %Z1.facet.loc21_24 [concrete = constants.%Z1.facet]
// CHECK:STDOUT:   %Outer.loc21_24: type = class_type @Outer, @Outer(constants.%Z1.facet) [concrete = constants.%Outer.01a]
// CHECK:STDOUT:   %.loc21_25: %Outer.G.type.e1d = specific_constant @Outer.%Outer.G.decl, @Outer(constants.%Z1.facet) [concrete = constants.%Outer.G.163]
// CHECK:STDOUT:   %G.ref: %Outer.G.type.e1d = name_ref G, %.loc21_25 [concrete = constants.%Outer.G.163]
// CHECK:STDOUT:   %Outer.ref.loc21_28: %Outer.type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:   %.loc21_35: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %Z1.ref.loc21_41: type = name_ref Z1, file.%Z1.decl [concrete = constants.%Z1.type]
// CHECK:STDOUT:   %Z2.ref.loc21_46: type = name_ref Z2, file.%Z2.decl [concrete = constants.%Z2.type]
// CHECK:STDOUT:   %impl.elem0.loc21_44: %.b4e = impl_witness_access constants.%BitAndWith.impl_witness, element0 [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:   %bound_method.loc21_44: <bound method> = bound_method %Z1.ref.loc21_41, %impl.elem0.loc21_44 [concrete = constants.%type.as.BitAndWith.impl.Op.bound]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.call.loc21_44: init type = call %bound_method.loc21_44(%Z1.ref.loc21_41, %Z2.ref.loc21_46) [concrete = constants.%facet_type]
// CHECK:STDOUT:   %.loc21_48.1: type = value_of_initializer %type.as.BitAndWith.impl.Op.call.loc21_44 [concrete = constants.%facet_type]
// CHECK:STDOUT:   %.loc21_48.2: type = converted %type.as.BitAndWith.impl.Op.call.loc21_44, %.loc21_48.1 [concrete = constants.%facet_type]
// CHECK:STDOUT:   %facet_value.loc21_37: %facet_type = facet_value constants.%empty_tuple.type, (constants.%Z1.impl_witness, constants.%Z2.impl_witness) [concrete = constants.%facet_value.794]
// CHECK:STDOUT:   %.loc21_37: %facet_type = converted %.loc21_35, %facet_value.loc21_37 [concrete = constants.%facet_value.794]
// CHECK:STDOUT:   %as_type.loc21_49: type = facet_access_type %.loc21_37 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %Z1.facet.loc21_49: %Z1.type = facet_value %as_type.loc21_49, (constants.%Z1.impl_witness) [concrete = constants.%Z1.facet]
// CHECK:STDOUT:   %.loc21_49: %Z1.type = converted %.loc21_37, %Z1.facet.loc21_49 [concrete = constants.%Z1.facet]
// CHECK:STDOUT:   %Outer.loc21_49: type = class_type @Outer, @Outer(constants.%Z1.facet) [concrete = constants.%Outer.01a]
// CHECK:STDOUT:   %.loc21_50: type = specific_constant @Outer.%C.decl, @Outer(constants.%Z1.facet) [concrete = constants.%C.d9e]
// CHECK:STDOUT:   %C.ref: type = name_ref C, %.loc21_50 [concrete = constants.%C.d9e]
// CHECK:STDOUT:   %facet_value.loc21_52: %Y_where.type.677 = facet_value constants.%C.d9e, (constants.%Y.impl_witness.1b2) [concrete = constants.%facet_value.c7c]
// CHECK:STDOUT:   %.loc21_52: %Y_where.type.677 = converted constants.%C.d9e, %facet_value.loc21_52 [concrete = constants.%facet_value.c7c]
// CHECK:STDOUT:   %Outer.G.specific_fn: <specific function> = specific_function %G.ref, @Outer.G(constants.%Z1.facet, constants.%facet_value.c7c) [concrete = constants.%Outer.G.specific_fn]
// CHECK:STDOUT:   %Outer.G.call: init %empty_tuple.type = call %Outer.G.specific_fn()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z1.WithSelf(constants.%Self.bc4) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z2.WithSelf(constants.%Self.5ad) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%OuterParam) {
// CHECK:STDOUT:   %OuterParam.patt.loc6_23.2 => constants.%OuterParam.patt
// CHECK:STDOUT:   %OuterParam.loc6_23.1 => constants.%OuterParam
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Y.type => constants.%Y.type.9a0
// CHECK:STDOUT:   %Outer.G.type => constants.%Outer.G.type.b7b
// CHECK:STDOUT:   %Outer.G => constants.%Outer.G.27c
// CHECK:STDOUT:   %C => constants.%C.5f2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y(constants.%OuterParam) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %OuterParam => constants.%OuterParam
// CHECK:STDOUT:   %Y.type => constants.%Y.type.9a0
// CHECK:STDOUT:   %Self.loc7_15.2 => constants.%Self.d06
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y.WithSelf(constants.%OuterParam, constants.%Self.d06) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %OuterParam => constants.%OuterParam
// CHECK:STDOUT:   %Y.assoc_type => constants.%Y.assoc_type.5d5
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.b45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y.WithSelf(constants.%OuterParam, constants.%.Self.frozen.39e) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %OuterParam => constants.%OuterParam
// CHECK:STDOUT:   %Y.assoc_type => constants.%Y.assoc_type.5d5
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.b45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer.G(constants.%OuterParam, constants.%H) {
// CHECK:STDOUT:   %OuterParam => constants.%OuterParam
// CHECK:STDOUT:   %Y.type => constants.%Y.type.9a0
// CHECK:STDOUT:   %.Self.frozen.loc11_28.1 => constants.%.Self.frozen.39e
// CHECK:STDOUT:   %require_complete => constants.%require_complete
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.618
// CHECK:STDOUT:   %Y.assoc_type => constants.%Y.assoc_type.5d5
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.b45
// CHECK:STDOUT:   %Y.lookup_impl_witness.loc11_34.1 => constants.%Y.lookup_impl_witness.580
// CHECK:STDOUT:   %impl.elem0.loc11_34.1 => constants.%impl.elem0.be8
// CHECK:STDOUT:   %.Self => constants.%.Self.2b6
// CHECK:STDOUT:   %Y.lookup_impl_witness.loc11_34.2 => constants.%Y.lookup_impl_witness.00e
// CHECK:STDOUT:   %impl.elem0.loc11_34.2 => constants.%impl.elem0.43a
// CHECK:STDOUT:   %Y_where.type => constants.%Y_where.type.352
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.8be
// CHECK:STDOUT:   %H.patt.loc11_24.2 => constants.%H.patt.57a
// CHECK:STDOUT:   %H.loc11_24.1 => constants.%H
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%OuterParam) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Y.impl(constants.%OuterParam) {
// CHECK:STDOUT:   %OuterParam => constants.%OuterParam
// CHECK:STDOUT:   %C => constants.%C.5f2
// CHECK:STDOUT:   %Y.type => constants.%Y.type.9a0
// CHECK:STDOUT:   %.Self.frozen.loc14_15.2 => constants.%.Self.frozen.39e
// CHECK:STDOUT:   %require_complete => constants.%require_complete
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.618
// CHECK:STDOUT:   %Y.assoc_type => constants.%Y.assoc_type.5d5
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.b45
// CHECK:STDOUT:   %.loc14_21.3 => constants.%.eeb
// CHECK:STDOUT:   %impl.elem0.loc14_21.3 => constants.%impl.elem0.756
// CHECK:STDOUT:   %.Self => constants.%.Self.2b6
// CHECK:STDOUT:   %.loc14_21.4 => constants.%.66e
// CHECK:STDOUT:   %impl.elem0.loc14_21.4 => constants.%impl.elem0.295
// CHECK:STDOUT:   %Y_where.type => constants.%Y_where.type.aa4
// CHECK:STDOUT:   %.loc14_29 => constants.%.6b3
// CHECK:STDOUT:   %Y.impl_witness.loc14_29.2 => constants.%Y.impl_witness.d75
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y.WithSelf(constants.%OuterParam, constants.%Y.facet.44a) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %OuterParam => constants.%OuterParam
// CHECK:STDOUT:   %Y.assoc_type => constants.%Y.assoc_type.5d5
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.b45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z1.WithSelf(constants.%Z1.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z2.WithSelf(constants.%Z2.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%Z1.facet) {
// CHECK:STDOUT:   %OuterParam.patt.loc6_23.2 => constants.%OuterParam.patt
// CHECK:STDOUT:   %OuterParam.loc6_23.1 => constants.%Z1.facet
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Y.type => constants.%Y.type.30f
// CHECK:STDOUT:   %Outer.G.type => constants.%Outer.G.type.e1d
// CHECK:STDOUT:   %Outer.G => constants.%Outer.G.163
// CHECK:STDOUT:   %C => constants.%C.d9e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y(constants.%Z1.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %OuterParam => constants.%Z1.facet
// CHECK:STDOUT:   %Y.type => constants.%Y.type.30f
// CHECK:STDOUT:   %Self.loc7_15.2 => constants.%Self.1f8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%Z1.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Y.impl(constants.%Z1.facet) {
// CHECK:STDOUT:   %OuterParam => constants.%Z1.facet
// CHECK:STDOUT:   %C => constants.%C.d9e
// CHECK:STDOUT:   %Y.type => constants.%Y.type.30f
// CHECK:STDOUT:   %.Self.frozen.loc14_15.2 => constants.%.Self.frozen.126
// CHECK:STDOUT:   %require_complete => constants.%complete_type.d0d
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.746
// CHECK:STDOUT:   %Y.assoc_type => constants.%Y.assoc_type.104
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.2bd
// CHECK:STDOUT:   %.loc14_21.3 => constants.%.03e
// CHECK:STDOUT:   %impl.elem0.loc14_21.3 => constants.%impl.elem0.ad3
// CHECK:STDOUT:   %.Self => constants.%.Self.1da
// CHECK:STDOUT:   %.loc14_21.4 => constants.%.4bb
// CHECK:STDOUT:   %impl.elem0.loc14_21.4 => constants.%impl.elem0.eb1
// CHECK:STDOUT:   %Y_where.type => constants.%Y_where.type.d3d
// CHECK:STDOUT:   %.loc14_29 => constants.%.9bc
// CHECK:STDOUT:   %Y.impl_witness.loc14_29.2 => constants.%Y.impl_witness.1b2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y.WithSelf(constants.%Z1.facet, constants.%Self.d06) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %OuterParam => constants.%Z1.facet
// CHECK:STDOUT:   %Y.assoc_type => constants.%Y.assoc_type.104
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.2bd
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer.G(constants.%Z1.facet, constants.%facet_value.c7c) {
// CHECK:STDOUT:   %OuterParam => constants.%Z1.facet
// CHECK:STDOUT:   %Y.type => constants.%Y.type.30f
// CHECK:STDOUT:   %.Self.frozen.loc11_28.1 => constants.%.Self.frozen.126
// CHECK:STDOUT:   %require_complete => constants.%complete_type.d0d
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.746
// CHECK:STDOUT:   %Y.assoc_type => constants.%Y.assoc_type.104
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.2bd
// CHECK:STDOUT:   %Y.lookup_impl_witness.loc11_34.1 => constants.%Y.lookup_impl_witness.a66
// CHECK:STDOUT:   %impl.elem0.loc11_34.1 => constants.%impl.elem0.e0b
// CHECK:STDOUT:   %.Self => constants.%.Self.1da
// CHECK:STDOUT:   %Y.lookup_impl_witness.loc11_34.2 => constants.%Y.lookup_impl_witness.0bd
// CHECK:STDOUT:   %impl.elem0.loc11_34.2 => constants.%impl.elem0.186
// CHECK:STDOUT:   %Y_where.type => constants.%Y_where.type.677
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.285
// CHECK:STDOUT:   %H.patt.loc11_24.2 => constants.%H.patt.305
// CHECK:STDOUT:   %H.loc11_24.1 => constants.%facet_value.c7c
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:

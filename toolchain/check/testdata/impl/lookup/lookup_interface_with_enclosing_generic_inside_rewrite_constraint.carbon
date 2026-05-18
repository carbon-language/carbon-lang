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

class Outer(OuterParam:! type) {
  interface Y {
    let T:! type;
  }

  fn G(unused H:! Y where .T = ()) {}
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

class Outer(OuterParam:! Z1) {
  interface Y {
    let T:! type;
  }

  fn G(unused H:! Y where .T = ()) {}

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
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.c39: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %OuterParam: type = symbolic_binding OuterParam, 0 [symbolic]
// CHECK:STDOUT:   %Outer.type: type = generic_class_type @Outer [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Outer.generic: %Outer.type = struct_value () [concrete]
// CHECK:STDOUT:   %Outer.6e1: type = class_type @Outer, @Outer(%OuterParam) [symbolic]
// CHECK:STDOUT:   %Y.type.fb1: type = facet_type <@Y, @Y(%OuterParam)> [symbolic]
// CHECK:STDOUT:   %Self.1c1: %Y.type.fb1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Y.assoc_type.062: type = assoc_entity_type @Y, @Y(%OuterParam) [symbolic]
// CHECK:STDOUT:   %assoc0.6ee: %Y.assoc_type.062 = assoc_entity element0, @Y.WithSelf.%T [symbolic]
// CHECK:STDOUT:   %.Self.dce: %Y.type.fb1 = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Y.type.fb1 [symbolic]
// CHECK:STDOUT:   %.Self.as_type.1a0: type = facet_access_type %.Self.dce [symbolic]
// CHECK:STDOUT:   %Y.lookup_impl_witness.906: <witness> = lookup_impl_witness %.Self.dce, @Y, @Y(%OuterParam) [symbolic]
// CHECK:STDOUT:   %impl.elem0.29e: type = impl_witness_access %Y.lookup_impl_witness.906, element0 [symbolic]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %Y_where.type.5e9: type = facet_type <@Y, @Y(%OuterParam) where %impl.elem0.29e = %empty_tuple.type> [symbolic]
// CHECK:STDOUT:   %pattern_type.b4b: type = pattern_type %Y_where.type.5e9 [symbolic]
// CHECK:STDOUT:   %H: %Y_where.type.5e9 = symbolic_binding H, 1 [symbolic]
// CHECK:STDOUT:   %Outer.G.type.138: type = fn_type @Outer.G, @Outer(%OuterParam) [symbolic]
// CHECK:STDOUT:   %Outer.G.3b7: %Outer.G.type.138 = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %Outer.026: type = class_type @Outer, @Outer(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %Y.type.d76: type = facet_type <@Y, @Y(%empty_tuple.type)> [concrete]
// CHECK:STDOUT:   %Outer.G.type.c29: type = fn_type @Outer.G, @Outer(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %Outer.G.887: %Outer.G.type.c29 = struct_value () [concrete]
// CHECK:STDOUT:   %.Self.6e4: %Y.type.d76 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Self.837: %Y.type.d76 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Y.assoc_type.499: type = assoc_entity_type @Y, @Y(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %assoc0.28f: %Y.assoc_type.499 = assoc_entity element0, @Y.WithSelf.%T [concrete]
// CHECK:STDOUT:   %.Self.as_type.18e: type = facet_access_type %.Self.6e4 [symbolic_self]
// CHECK:STDOUT:   %Y.lookup_impl_witness.849: <witness> = lookup_impl_witness %.Self.6e4, @Y, @Y(%empty_tuple.type) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.399: type = impl_witness_access %Y.lookup_impl_witness.849, element0 [symbolic_self]
// CHECK:STDOUT:   %Y_where.type.ce0: type = facet_type <@Y, @Y(%empty_tuple.type) where %impl.elem0.399 = %empty_tuple.type> [concrete]
// CHECK:STDOUT:   %Y.impl_witness: <witness> = impl_witness @C.as.Y.impl.%Y.impl_witness_table [concrete]
// CHECK:STDOUT:   %Y.facet: %Y.type.d76 = facet_value %C, (%Y.impl_witness) [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value: %Y_where.type.ce0 = facet_value %C, (%Y.impl_witness) [concrete]
// CHECK:STDOUT:   %complete_type.2a1: <witness> = complete_type_witness %Y.type.d76 [concrete]
// CHECK:STDOUT:   %pattern_type.405: type = pattern_type %Y_where.type.ce0 [concrete]
// CHECK:STDOUT:   %Outer.G.specific_fn: <specific function> = specific_function %Outer.G.887, @Outer.G(%empty_tuple.type, %facet_value) [concrete]
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
// CHECK:STDOUT:     %OuterParam.patt: %pattern_type.98f = symbolic_binding_pattern OuterParam, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc49_26.1: type = splice_block %.loc49_26.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %.loc49_26.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %OuterParam.loc49_23.2: type = symbolic_binding OuterParam, 0 [symbolic = %OuterParam.loc49_23.1 (constants.%OuterParam)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @C.as.Y.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %Outer.ref: %Outer.type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:     %.loc58_18: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc58_19: type = converted %.loc58_18, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %Outer: type = class_type @Outer, @Outer(constants.%empty_tuple.type) [concrete = constants.%Outer.026]
// CHECK:STDOUT:     %.loc58_20: type = specific_constant @Outer.%Y.decl, @Outer(constants.%empty_tuple.type) [concrete = constants.%Y.type.d76]
// CHECK:STDOUT:     %Y.ref: type = name_ref Y, %.loc58_20 [concrete = constants.%Y.type.d76]
// CHECK:STDOUT:     %.Self: %Y.type.d76 = symbolic_binding .Self [symbolic_self = constants.%.Self.6e4]
// CHECK:STDOUT:     %.Self.ref: %Y.type.d76 = name_ref .Self, %.Self [symbolic_self = constants.%.Self.6e4]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type.18e]
// CHECK:STDOUT:     %.loc58_29.1: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type.18e]
// CHECK:STDOUT:     %.loc58_29.2: %Y.assoc_type.499 = specific_constant @T.%assoc0, @Y.WithSelf(constants.%empty_tuple.type, constants.%.Self.6e4) [concrete = constants.%assoc0.28f]
// CHECK:STDOUT:     %T.ref: %Y.assoc_type.499 = name_ref T, %.loc58_29.2 [concrete = constants.%assoc0.28f]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%Y.lookup_impl_witness.849, element0 [symbolic_self = constants.%impl.elem0.399]
// CHECK:STDOUT:     %.loc58_35.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc58_35.2: type = converted %.loc58_35.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc58_23: type = where_expr [concrete = constants.%Y_where.type.ce0] {
// CHECK:STDOUT:       requirement_base_facet_type %Y.ref
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %.loc58_35.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Y(@Outer.%OuterParam.loc49_23.2: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %OuterParam: type = symbolic_binding OuterParam, 0 [symbolic = %OuterParam (constants.%OuterParam)]
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y, @Y(%OuterParam)> [symbolic = %Y.type (constants.%Y.type.fb1)]
// CHECK:STDOUT:   %Self.loc50_15.2: @Y.%Y.type (%Y.type.fb1) = symbolic_binding Self, 1 [symbolic = %Self.loc50_15.2 (constants.%Self.1c1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc50_15.1: @Y.%Y.type (%Y.type.fb1) = symbolic_binding Self, 1 [symbolic = %Self.loc50_15.2 (constants.%Self.1c1)]
// CHECK:STDOUT:     %Y.WithSelf.decl = interface_with_self_decl @Y [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %T: type = assoc_const_decl @T [concrete] {
// CHECK:STDOUT:       %assoc0: @Y.WithSelf.%Y.assoc_type (%Y.assoc_type.062) = assoc_entity element0, @Y.WithSelf.%T [symbolic = @Y.WithSelf.%assoc0 (constants.%assoc0.6ee)]
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
// CHECK:STDOUT: impl @C.as.Y.impl: %C.ref as %.loc58_23 {
// CHECK:STDOUT:   %Y.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @C.as.Y.impl [concrete]
// CHECK:STDOUT:   %Y.impl_witness: <witness> = impl_witness %Y.impl_witness_table [concrete = constants.%Y.impl_witness]
// CHECK:STDOUT:   %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend %.loc58_23
// CHECK:STDOUT:   witness = %Y.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Outer(%OuterParam.loc49_23.2: type) {
// CHECK:STDOUT:   %OuterParam.loc49_23.1: type = symbolic_binding OuterParam, 0 [symbolic = %OuterParam.loc49_23.1 (constants.%OuterParam)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y, @Y(%OuterParam.loc49_23.1)> [symbolic = %Y.type (constants.%Y.type.fb1)]
// CHECK:STDOUT:   %Outer.G.type: type = fn_type @Outer.G, @Outer(%OuterParam.loc49_23.1) [symbolic = %Outer.G.type (constants.%Outer.G.type.138)]
// CHECK:STDOUT:   %Outer.G: @Outer.%Outer.G.type (%Outer.G.type.138) = struct_value () [symbolic = %Outer.G (constants.%Outer.G.3b7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Y.decl: type = interface_decl @Y [symbolic = @Outer.%Y.type (constants.%Y.type.fb1)] {} {}
// CHECK:STDOUT:     %Outer.G.decl: @Outer.%Outer.G.type (%Outer.G.type.138) = fn_decl @Outer.G [symbolic = @Outer.%Outer.G (constants.%Outer.G.3b7)] {
// CHECK:STDOUT:       %H.patt: @Outer.G.%pattern_type (%pattern_type.b4b) = symbolic_binding_pattern H, 1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc54_21.1: type = splice_block %.loc54_21.2 [symbolic = %Y_where.type (constants.%Y_where.type.5e9)] {
// CHECK:STDOUT:         %.Self.loc54_16: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:         %.loc54_19: type = specific_constant @Outer.%Y.decl, @Outer(constants.%OuterParam) [symbolic = %Y.type (constants.%Y.type.fb1)]
// CHECK:STDOUT:         %Y.ref: type = name_ref Y, %.loc54_19 [symbolic = %Y.type (constants.%Y.type.fb1)]
// CHECK:STDOUT:         %.Self.loc54_21.2: @Outer.G.%Y.type (%Y.type.fb1) = symbolic_binding .Self [symbolic = %.Self.loc54_21.1 (constants.%.Self.dce)]
// CHECK:STDOUT:         %.Self.ref: @Outer.G.%Y.type (%Y.type.fb1) = name_ref .Self, %.Self.loc54_21.2 [symbolic = %.Self.loc54_21.1 (constants.%.Self.dce)]
// CHECK:STDOUT:         %.Self.as_type.loc54_27.2: type = facet_access_type %.Self.ref [symbolic = %.Self.as_type.loc54_27.1 (constants.%.Self.as_type.1a0)]
// CHECK:STDOUT:         %.loc54_27.1: type = converted %.Self.ref, %.Self.as_type.loc54_27.2 [symbolic = %.Self.as_type.loc54_27.1 (constants.%.Self.as_type.1a0)]
// CHECK:STDOUT:         %.loc54_27.2: @Outer.G.%Y.assoc_type (%Y.assoc_type.062) = specific_constant @T.%assoc0, @Y.WithSelf(constants.%OuterParam, constants.%.Self.dce) [symbolic = %assoc0 (constants.%assoc0.6ee)]
// CHECK:STDOUT:         %T.ref: @Outer.G.%Y.assoc_type (%Y.assoc_type.062) = name_ref T, %.loc54_27.2 [symbolic = %assoc0 (constants.%assoc0.6ee)]
// CHECK:STDOUT:         %impl.elem0.loc54_27.2: type = impl_witness_access constants.%Y.lookup_impl_witness.906, element0 [symbolic = %impl.elem0.loc54_27.1 (constants.%impl.elem0.29e)]
// CHECK:STDOUT:         %.loc54_33.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:         %.loc54_33.2: type = converted %.loc54_33.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:         %.loc54_21.2: type = where_expr [symbolic = %Y_where.type (constants.%Y_where.type.5e9)] {
// CHECK:STDOUT:           requirement_base_facet_type %Y.ref
// CHECK:STDOUT:           requirement_rewrite %impl.elem0.loc54_27.2, %.loc54_33.2
// CHECK:STDOUT:         }
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %H.loc54_16.2: @Outer.G.%Y_where.type (%Y_where.type.5e9) = symbolic_binding H, 1 [symbolic = %H.loc54_16.1 (constants.%H)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Outer.6e1
// CHECK:STDOUT:     .Y = %Y.decl
// CHECK:STDOUT:     .G = %Outer.G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C;
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Outer.G(@Outer.%OuterParam.loc49_23.2: type, %H.loc54_16.2: @Outer.G.%Y_where.type (%Y_where.type.5e9)) {
// CHECK:STDOUT:   %OuterParam: type = symbolic_binding OuterParam, 0 [symbolic = %OuterParam (constants.%OuterParam)]
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y, @Y(%OuterParam)> [symbolic = %Y.type (constants.%Y.type.fb1)]
// CHECK:STDOUT:   %.Self.loc54_21.1: @Outer.G.%Y.type (%Y.type.fb1) = symbolic_binding .Self [symbolic = %.Self.loc54_21.1 (constants.%.Self.dce)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Y.type [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %.Self.as_type.loc54_27.1: type = facet_access_type %.Self.loc54_21.1 [symbolic = %.Self.as_type.loc54_27.1 (constants.%.Self.as_type.1a0)]
// CHECK:STDOUT:   %Y.assoc_type: type = assoc_entity_type @Y, @Y(%OuterParam) [symbolic = %Y.assoc_type (constants.%Y.assoc_type.062)]
// CHECK:STDOUT:   %assoc0: @Outer.G.%Y.assoc_type (%Y.assoc_type.062) = assoc_entity element0, @Y.WithSelf.%T [symbolic = %assoc0 (constants.%assoc0.6ee)]
// CHECK:STDOUT:   %Y.lookup_impl_witness: <witness> = lookup_impl_witness %.Self.loc54_21.1, @Y, @Y(%OuterParam) [symbolic = %Y.lookup_impl_witness (constants.%Y.lookup_impl_witness.906)]
// CHECK:STDOUT:   %impl.elem0.loc54_27.1: type = impl_witness_access %Y.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc54_27.1 (constants.%impl.elem0.29e)]
// CHECK:STDOUT:   %Y_where.type: type = facet_type <@Y, @Y(%OuterParam) where %impl.elem0.loc54_27.1 = constants.%empty_tuple.type> [symbolic = %Y_where.type (constants.%Y_where.type.5e9)]
// CHECK:STDOUT:   %H.loc54_16.1: @Outer.G.%Y_where.type (%Y_where.type.5e9) = symbolic_binding H, 1 [symbolic = %H.loc54_16.1 (constants.%H)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Y_where.type [symbolic = %pattern_type (constants.%pattern_type.b4b)]
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
// CHECK:STDOUT:   %Outer: type = class_type @Outer, @Outer(constants.%empty_tuple.type) [concrete = constants.%Outer.026]
// CHECK:STDOUT:   %.loc61_12: %Outer.G.type.c29 = specific_constant @Outer.%Outer.G.decl, @Outer(constants.%empty_tuple.type) [concrete = constants.%Outer.G.887]
// CHECK:STDOUT:   %G.ref: %Outer.G.type.c29 = name_ref G, %.loc61_12 [concrete = constants.%Outer.G.887]
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %facet_value: %Y_where.type.ce0 = facet_value constants.%C, (constants.%Y.impl_witness) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc61_16: %Y_where.type.ce0 = converted constants.%C, %facet_value [concrete = constants.%facet_value]
// CHECK:STDOUT:   %Outer.G.specific_fn: <specific function> = specific_function %G.ref, @Outer.G(constants.%empty_tuple.type, constants.%facet_value) [concrete = constants.%Outer.G.specific_fn]
// CHECK:STDOUT:   %Outer.G.call: init %empty_tuple.type = call %Outer.G.specific_fn()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%OuterParam) {
// CHECK:STDOUT:   %OuterParam.loc49_23.1 => constants.%OuterParam
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Y.type => constants.%Y.type.fb1
// CHECK:STDOUT:   %Outer.G.type => constants.%Outer.G.type.138
// CHECK:STDOUT:   %Outer.G => constants.%Outer.G.3b7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y(constants.%OuterParam) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %OuterParam => constants.%OuterParam
// CHECK:STDOUT:   %Y.type => constants.%Y.type.fb1
// CHECK:STDOUT:   %Self.loc50_15.2 => constants.%Self.1c1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y.WithSelf(constants.%OuterParam, constants.%Self.1c1) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %OuterParam => constants.%OuterParam
// CHECK:STDOUT:   %Y.assoc_type => constants.%Y.assoc_type.062
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.6ee
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y.WithSelf(constants.%OuterParam, constants.%.Self.dce) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %OuterParam => constants.%OuterParam
// CHECK:STDOUT:   %Y.assoc_type => constants.%Y.assoc_type.062
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.6ee
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer.G(constants.%OuterParam, constants.%H) {
// CHECK:STDOUT:   %OuterParam => constants.%OuterParam
// CHECK:STDOUT:   %Y.type => constants.%Y.type.fb1
// CHECK:STDOUT:   %.Self.loc54_21.1 => constants.%.Self.dce
// CHECK:STDOUT:   %require_complete => constants.%require_complete
// CHECK:STDOUT:   %.Self.as_type.loc54_27.1 => constants.%.Self.as_type.1a0
// CHECK:STDOUT:   %Y.assoc_type => constants.%Y.assoc_type.062
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.6ee
// CHECK:STDOUT:   %Y.lookup_impl_witness => constants.%Y.lookup_impl_witness.906
// CHECK:STDOUT:   %impl.elem0.loc54_27.1 => constants.%impl.elem0.29e
// CHECK:STDOUT:   %Y_where.type => constants.%Y_where.type.5e9
// CHECK:STDOUT:   %H.loc54_16.1 => constants.%H
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.b4b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%empty_tuple.type) {
// CHECK:STDOUT:   %OuterParam.loc49_23.1 => constants.%empty_tuple.type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Y.type => constants.%Y.type.d76
// CHECK:STDOUT:   %Outer.G.type => constants.%Outer.G.type.c29
// CHECK:STDOUT:   %Outer.G => constants.%Outer.G.887
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y(constants.%empty_tuple.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %OuterParam => constants.%empty_tuple.type
// CHECK:STDOUT:   %Y.type => constants.%Y.type.d76
// CHECK:STDOUT:   %Self.loc50_15.2 => constants.%Self.837
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y.WithSelf(constants.%empty_tuple.type, constants.%Self.1c1) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %OuterParam => constants.%empty_tuple.type
// CHECK:STDOUT:   %Y.assoc_type => constants.%Y.assoc_type.499
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.28f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y.WithSelf(constants.%empty_tuple.type, constants.%.Self.6e4) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %OuterParam => constants.%empty_tuple.type
// CHECK:STDOUT:   %Y.assoc_type => constants.%Y.assoc_type.499
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.28f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y.WithSelf(constants.%empty_tuple.type, constants.%Y.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %OuterParam => constants.%empty_tuple.type
// CHECK:STDOUT:   %Y.assoc_type => constants.%Y.assoc_type.499
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.28f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer.G(constants.%empty_tuple.type, constants.%facet_value) {
// CHECK:STDOUT:   %OuterParam => constants.%empty_tuple.type
// CHECK:STDOUT:   %Y.type => constants.%Y.type.d76
// CHECK:STDOUT:   %.Self.loc54_21.1 => constants.%.Self.6e4
// CHECK:STDOUT:   %require_complete => constants.%complete_type.2a1
// CHECK:STDOUT:   %.Self.as_type.loc54_27.1 => constants.%.Self.as_type.18e
// CHECK:STDOUT:   %Y.assoc_type => constants.%Y.assoc_type.499
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.28f
// CHECK:STDOUT:   %Y.lookup_impl_witness => constants.%Y.lookup_impl_witness.849
// CHECK:STDOUT:   %impl.elem0.loc54_27.1 => constants.%impl.elem0.399
// CHECK:STDOUT:   %Y_where.type => constants.%Y_where.type.ce0
// CHECK:STDOUT:   %H.loc54_16.1 => constants.%facet_value
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.405
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- enclosing_specific_is_facet_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Z1.type: type = facet_type <@Z1> [concrete]
// CHECK:STDOUT:   %Self.641: %Z1.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Z2.type: type = facet_type <@Z2> [concrete]
// CHECK:STDOUT:   %Self.b23: %Z2.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.c39: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.ff3: type = pattern_type %Z1.type [concrete]
// CHECK:STDOUT:   %OuterParam: %Z1.type = symbolic_binding OuterParam, 0 [symbolic]
// CHECK:STDOUT:   %Outer.type: type = generic_class_type @Outer [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Outer.generic: %Outer.type = struct_value () [concrete]
// CHECK:STDOUT:   %Outer.46c: type = class_type @Outer, @Outer(%OuterParam) [symbolic]
// CHECK:STDOUT:   %Y.type.69f: type = facet_type <@Y, @Y(%OuterParam)> [symbolic]
// CHECK:STDOUT:   %Self.3c2: %Y.type.69f = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Y.assoc_type.10f: type = assoc_entity_type @Y, @Y(%OuterParam) [symbolic]
// CHECK:STDOUT:   %assoc0.5a4: %Y.assoc_type.10f = assoc_entity element0, @Y.WithSelf.%T [symbolic]
// CHECK:STDOUT:   %.Self.766: %Y.type.69f = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %require_complete.b56: <witness> = require_complete_type %Y.type.69f [symbolic]
// CHECK:STDOUT:   %.Self.as_type.9c1: type = facet_access_type %.Self.766 [symbolic]
// CHECK:STDOUT:   %Y.lookup_impl_witness.838: <witness> = lookup_impl_witness %.Self.766, @Y, @Y(%OuterParam) [symbolic]
// CHECK:STDOUT:   %impl.elem0.96e: type = impl_witness_access %Y.lookup_impl_witness.838, element0 [symbolic]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %Y_where.type.228: type = facet_type <@Y, @Y(%OuterParam) where %impl.elem0.96e = %empty_tuple.type> [symbolic]
// CHECK:STDOUT:   %pattern_type.ab5: type = pattern_type %Y_where.type.228 [symbolic]
// CHECK:STDOUT:   %H: %Y_where.type.228 = symbolic_binding H, 1 [symbolic]
// CHECK:STDOUT:   %Outer.G.type.cd7: type = fn_type @Outer.G, @Outer(%OuterParam) [symbolic]
// CHECK:STDOUT:   %Outer.G.639: %Outer.G.type.cd7 = struct_value () [symbolic]
// CHECK:STDOUT:   %C.998: type = class_type @C, @C(%OuterParam) [symbolic]
// CHECK:STDOUT:   %Y.impl_witness.79e: <witness> = impl_witness @C.as.Y.impl.%Y.impl_witness_table, @C.as.Y.impl(%OuterParam) [symbolic]
// CHECK:STDOUT:   %require_complete.cbb: <witness> = require_complete_type %Y_where.type.228 [symbolic]
// CHECK:STDOUT:   %Y.facet.8a6: %Y.type.69f = facet_value %C.998, (%Y.impl_witness.79e) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Z1.impl_witness: <witness> = impl_witness @empty_tuple.type.as.Z1.impl.%Z1.impl_witness_table [concrete]
// CHECK:STDOUT:   %Z1.facet: %Z1.type = facet_value %empty_tuple.type, (%Z1.impl_witness) [concrete]
// CHECK:STDOUT:   %Z2.impl_witness: <witness> = impl_witness @empty_tuple.type.as.Z2.impl.%Z2.impl_witness_table [concrete]
// CHECK:STDOUT:   %Z2.facet: %Z2.type = facet_value %empty_tuple.type, (%Z2.impl_witness) [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndWith.type.447: type = generic_interface_type @BitAndWith [concrete]
// CHECK:STDOUT:   %BitAndWith.generic: %BitAndWith.type.447 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndWith.type.d17: type = facet_type <@BitAndWith, @BitAndWith(type)> [concrete]
// CHECK:STDOUT:   %BitAndWith.impl_witness: <witness> = impl_witness imports.%BitAndWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %BitAndWith.facet: %BitAndWith.type.d17 = facet_value type, (%BitAndWith.impl_witness) [concrete]
// CHECK:STDOUT:   %BitAndWith.WithSelf.Op.type.3c5: type = fn_type @BitAndWith.WithSelf.Op, @BitAndWith.WithSelf(type, %BitAndWith.facet) [concrete]
// CHECK:STDOUT:   %.849: type = fn_type_with_self_type %BitAndWith.WithSelf.Op.type.3c5, %BitAndWith.facet [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.type: type = fn_type @type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op: %type.as.BitAndWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.bound: <bound method> = bound_method %Z1.type, %type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@Z1 & @Z2> [concrete]
// CHECK:STDOUT:   %facet_value.f54: %facet_type = facet_value %empty_tuple.type, (%Z1.impl_witness, %Z2.impl_witness) [concrete]
// CHECK:STDOUT:   %Outer.2a6: type = class_type @Outer, @Outer(%Z1.facet) [concrete]
// CHECK:STDOUT:   %Y.type.f5b: type = facet_type <@Y, @Y(%Z1.facet)> [concrete]
// CHECK:STDOUT:   %Outer.G.type.9f4: type = fn_type @Outer.G, @Outer(%Z1.facet) [concrete]
// CHECK:STDOUT:   %Outer.G.268: %Outer.G.type.9f4 = struct_value () [concrete]
// CHECK:STDOUT:   %C.c35: type = class_type @C, @C(%Z1.facet) [concrete]
// CHECK:STDOUT:   %.Self.d74: %Y.type.f5b = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Y.lookup_impl_witness.25a: <witness> = lookup_impl_witness %.Self.d74, @Y, @Y(%Z1.facet) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.581: type = impl_witness_access %Y.lookup_impl_witness.25a, element0 [symbolic_self]
// CHECK:STDOUT:   %Y_where.type.125: type = facet_type <@Y, @Y(%Z1.facet) where %impl.elem0.581 = %empty_tuple.type> [concrete]
// CHECK:STDOUT:   %Self.2d8: %Y.type.f5b = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Y.assoc_type.845: type = assoc_entity_type @Y, @Y(%Z1.facet) [concrete]
// CHECK:STDOUT:   %assoc0.933: %Y.assoc_type.845 = assoc_entity element0, @Y.WithSelf.%T [concrete]
// CHECK:STDOUT:   %complete_type.e54: <witness> = complete_type_witness %Y.type.f5b [concrete]
// CHECK:STDOUT:   %.Self.as_type.194: type = facet_access_type %.Self.d74 [symbolic_self]
// CHECK:STDOUT:   %Y.impl_witness.f15: <witness> = impl_witness @C.as.Y.impl.%Y.impl_witness_table, @C.as.Y.impl(%Z1.facet) [concrete]
// CHECK:STDOUT:   %complete_type.f38: <witness> = complete_type_witness %Y_where.type.125 [concrete]
// CHECK:STDOUT:   %facet_value.4ab: %Y_where.type.125 = facet_value %C.c35, (%Y.impl_witness.f15) [concrete]
// CHECK:STDOUT:   %pattern_type.1f9: type = pattern_type %Y_where.type.125 [concrete]
// CHECK:STDOUT:   %Outer.G.specific_fn: <specific function> = specific_function %Outer.G.268, @Outer.G(%Z1.facet, %facet_value.4ab) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .BitAndWith = %Core.BitAndWith
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.BitAndWith: %BitAndWith.type.447 = import_ref Core//prelude/parts/as, BitAndWith, loaded [concrete = constants.%BitAndWith.generic]
// CHECK:STDOUT:   %Core.import_ref.4c9: %type.as.BitAndWith.impl.Op.type = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, loaded [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:   %BitAndWith.impl_witness_table = impl_witness_table (%Core.import_ref.4c9), @type.as.BitAndWith.impl [concrete]
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
// CHECK:STDOUT:     %OuterParam.patt: %pattern_type.ff3 = symbolic_binding_pattern OuterParam, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6: type = splice_block %Z1.ref [concrete = constants.%Z1.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %Z1.ref: type = name_ref Z1, file.%Z1.decl [concrete = constants.%Z1.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %OuterParam.loc6_23.2: %Z1.type = symbolic_binding OuterParam, 0 [symbolic = %OuterParam.loc6_23.1 (constants.%OuterParam)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @empty_tuple.type.as.Z1.impl [concrete] {} {
// CHECK:STDOUT:     %.loc17_7.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc17_7.2: type = converted %.loc17_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %Z1.ref: type = name_ref Z1, file.%Z1.decl [concrete = constants.%Z1.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @empty_tuple.type.as.Z2.impl [concrete] {} {
// CHECK:STDOUT:     %.loc18_7.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc18_7.2: type = converted %.loc18_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %Z2.ref: type = name_ref Z2, file.%Z2.decl [concrete = constants.%Z2.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Z1 {
// CHECK:STDOUT:   %Self: %Z1.type = symbolic_binding Self, 0 [symbolic = constants.%Self.641]
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
// CHECK:STDOUT:   %Self: %Z2.type = symbolic_binding Self, 0 [symbolic = constants.%Self.b23]
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
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y, @Y(%OuterParam)> [symbolic = %Y.type (constants.%Y.type.69f)]
// CHECK:STDOUT:   %Self.loc7_15.2: @Y.%Y.type (%Y.type.69f) = symbolic_binding Self, 1 [symbolic = %Self.loc7_15.2 (constants.%Self.3c2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc7_15.1: @Y.%Y.type (%Y.type.69f) = symbolic_binding Self, 1 [symbolic = %Self.loc7_15.2 (constants.%Self.3c2)]
// CHECK:STDOUT:     %Y.WithSelf.decl = interface_with_self_decl @Y [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %T: type = assoc_const_decl @T [concrete] {
// CHECK:STDOUT:       %assoc0: @Y.WithSelf.%Y.assoc_type (%Y.assoc_type.10f) = assoc_entity element0, @Y.WithSelf.%T [symbolic = @Y.WithSelf.%assoc0 (constants.%assoc0.5a4)]
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
// CHECK:STDOUT:   %C: type = class_type @C, @C(%OuterParam) [symbolic = %C (constants.%C.998)]
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y, @Y(%OuterParam)> [symbolic = %Y.type (constants.%Y.type.69f)]
// CHECK:STDOUT:   %.Self.loc14_15.2: @C.as.Y.impl.%Y.type (%Y.type.69f) = symbolic_binding .Self [symbolic = %.Self.loc14_15.2 (constants.%.Self.766)]
// CHECK:STDOUT:   %require_complete.loc14_21: <witness> = require_complete_type %Y.type [symbolic = %require_complete.loc14_21 (constants.%require_complete.b56)]
// CHECK:STDOUT:   %.Self.as_type.loc14_21.2: type = facet_access_type %.Self.loc14_15.2 [symbolic = %.Self.as_type.loc14_21.2 (constants.%.Self.as_type.9c1)]
// CHECK:STDOUT:   %Y.assoc_type: type = assoc_entity_type @Y, @Y(%OuterParam) [symbolic = %Y.assoc_type (constants.%Y.assoc_type.10f)]
// CHECK:STDOUT:   %assoc0: @C.as.Y.impl.%Y.assoc_type (%Y.assoc_type.10f) = assoc_entity element0, @Y.WithSelf.%T [symbolic = %assoc0 (constants.%assoc0.5a4)]
// CHECK:STDOUT:   %Y.lookup_impl_witness: <witness> = lookup_impl_witness %.Self.loc14_15.2, @Y, @Y(%OuterParam) [symbolic = %Y.lookup_impl_witness (constants.%Y.lookup_impl_witness.838)]
// CHECK:STDOUT:   %impl.elem0.loc14_21.2: type = impl_witness_access %Y.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc14_21.2 (constants.%impl.elem0.96e)]
// CHECK:STDOUT:   %Y_where.type: type = facet_type <@Y, @Y(%OuterParam) where %impl.elem0.loc14_21.2 = constants.%empty_tuple.type> [symbolic = %Y_where.type (constants.%Y_where.type.228)]
// CHECK:STDOUT:   %Y.impl_witness.loc14_29.2: <witness> = impl_witness %Y.impl_witness_table, @C.as.Y.impl(%OuterParam) [symbolic = %Y.impl_witness.loc14_29.2 (constants.%Y.impl_witness.79e)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc14_15: <witness> = require_complete_type %Y_where.type [symbolic = %require_complete.loc14_15 (constants.%require_complete.cbb)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref as %.loc14_15 {
// CHECK:STDOUT:     %Y.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @C.as.Y.impl [concrete]
// CHECK:STDOUT:     %Y.impl_witness.loc14_29.1: <witness> = impl_witness %Y.impl_witness_table, @C.as.Y.impl(constants.%OuterParam) [symbolic = %Y.impl_witness.loc14_29.2 (constants.%Y.impl_witness.79e)]
// CHECK:STDOUT:     %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     extend %.loc14_15
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
// CHECK:STDOUT:   %OuterParam.loc6_23.1: %Z1.type = symbolic_binding OuterParam, 0 [symbolic = %OuterParam.loc6_23.1 (constants.%OuterParam)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y, @Y(%OuterParam.loc6_23.1)> [symbolic = %Y.type (constants.%Y.type.69f)]
// CHECK:STDOUT:   %Outer.G.type: type = fn_type @Outer.G, @Outer(%OuterParam.loc6_23.1) [symbolic = %Outer.G.type (constants.%Outer.G.type.cd7)]
// CHECK:STDOUT:   %Outer.G: @Outer.%Outer.G.type (%Outer.G.type.cd7) = struct_value () [symbolic = %Outer.G (constants.%Outer.G.639)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%OuterParam.loc6_23.1) [symbolic = %C (constants.%C.998)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Y.decl: type = interface_decl @Y [symbolic = @Outer.%Y.type (constants.%Y.type.69f)] {} {}
// CHECK:STDOUT:     %Outer.G.decl: @Outer.%Outer.G.type (%Outer.G.type.cd7) = fn_decl @Outer.G [symbolic = @Outer.%Outer.G (constants.%Outer.G.639)] {
// CHECK:STDOUT:       %H.patt: @Outer.G.%pattern_type (%pattern_type.ab5) = symbolic_binding_pattern H, 1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc11_21.1: type = splice_block %.loc11_21.2 [symbolic = %Y_where.type (constants.%Y_where.type.228)] {
// CHECK:STDOUT:         %.Self.loc11_16: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:         %.loc11_19: type = specific_constant @Outer.%Y.decl, @Outer(constants.%OuterParam) [symbolic = %Y.type (constants.%Y.type.69f)]
// CHECK:STDOUT:         %Y.ref: type = name_ref Y, %.loc11_19 [symbolic = %Y.type (constants.%Y.type.69f)]
// CHECK:STDOUT:         %.Self.loc11_21.2: @Outer.G.%Y.type (%Y.type.69f) = symbolic_binding .Self [symbolic = %.Self.loc11_21.1 (constants.%.Self.766)]
// CHECK:STDOUT:         %.Self.ref: @Outer.G.%Y.type (%Y.type.69f) = name_ref .Self, %.Self.loc11_21.2 [symbolic = %.Self.loc11_21.1 (constants.%.Self.766)]
// CHECK:STDOUT:         %.Self.as_type.loc11_27.2: type = facet_access_type %.Self.ref [symbolic = %.Self.as_type.loc11_27.1 (constants.%.Self.as_type.9c1)]
// CHECK:STDOUT:         %.loc11_27.1: type = converted %.Self.ref, %.Self.as_type.loc11_27.2 [symbolic = %.Self.as_type.loc11_27.1 (constants.%.Self.as_type.9c1)]
// CHECK:STDOUT:         %.loc11_27.2: @Outer.G.%Y.assoc_type (%Y.assoc_type.10f) = specific_constant @T.%assoc0, @Y.WithSelf(constants.%OuterParam, constants.%.Self.766) [symbolic = %assoc0 (constants.%assoc0.5a4)]
// CHECK:STDOUT:         %T.ref: @Outer.G.%Y.assoc_type (%Y.assoc_type.10f) = name_ref T, %.loc11_27.2 [symbolic = %assoc0 (constants.%assoc0.5a4)]
// CHECK:STDOUT:         %impl.elem0.loc11_27.2: type = impl_witness_access constants.%Y.lookup_impl_witness.838, element0 [symbolic = %impl.elem0.loc11_27.1 (constants.%impl.elem0.96e)]
// CHECK:STDOUT:         %.loc11_33.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:         %.loc11_33.2: type = converted %.loc11_33.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:         %.loc11_21.2: type = where_expr [symbolic = %Y_where.type (constants.%Y_where.type.228)] {
// CHECK:STDOUT:           requirement_base_facet_type %Y.ref
// CHECK:STDOUT:           requirement_rewrite %impl.elem0.loc11_27.2, %.loc11_33.2
// CHECK:STDOUT:         }
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %H.loc11_16.2: @Outer.G.%Y_where.type (%Y_where.type.228) = symbolic_binding H, 1 [symbolic = %H.loc11_16.1 (constants.%H)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %C.decl: type = class_decl @C [symbolic = @Outer.%C (constants.%C.998)] {} {}
// CHECK:STDOUT:     impl_decl @C.as.Y.impl [concrete] {} {
// CHECK:STDOUT:       %.loc14_8: type = specific_constant @Outer.%C.decl, @Outer(constants.%OuterParam) [symbolic = %C (constants.%C.998)]
// CHECK:STDOUT:       %C.ref: type = name_ref C, %.loc14_8 [symbolic = %C (constants.%C.998)]
// CHECK:STDOUT:       %.loc14_13: type = specific_constant @Outer.%Y.decl, @Outer(constants.%OuterParam) [symbolic = %Y.type (constants.%Y.type.69f)]
// CHECK:STDOUT:       %Y.ref: type = name_ref Y, %.loc14_13 [symbolic = %Y.type (constants.%Y.type.69f)]
// CHECK:STDOUT:       %.Self.loc14_15.1: @C.as.Y.impl.%Y.type (%Y.type.69f) = symbolic_binding .Self [symbolic = %.Self.loc14_15.2 (constants.%.Self.766)]
// CHECK:STDOUT:       %.Self.ref: @C.as.Y.impl.%Y.type (%Y.type.69f) = name_ref .Self, %.Self.loc14_15.1 [symbolic = %.Self.loc14_15.2 (constants.%.Self.766)]
// CHECK:STDOUT:       %.Self.as_type.loc14_21.1: type = facet_access_type %.Self.ref [symbolic = %.Self.as_type.loc14_21.2 (constants.%.Self.as_type.9c1)]
// CHECK:STDOUT:       %.loc14_21.1: type = converted %.Self.ref, %.Self.as_type.loc14_21.1 [symbolic = %.Self.as_type.loc14_21.2 (constants.%.Self.as_type.9c1)]
// CHECK:STDOUT:       %.loc14_21.2: @C.as.Y.impl.%Y.assoc_type (%Y.assoc_type.10f) = specific_constant @T.%assoc0, @Y.WithSelf(constants.%OuterParam, constants.%.Self.766) [symbolic = %assoc0 (constants.%assoc0.5a4)]
// CHECK:STDOUT:       %T.ref: @C.as.Y.impl.%Y.assoc_type (%Y.assoc_type.10f) = name_ref T, %.loc14_21.2 [symbolic = %assoc0 (constants.%assoc0.5a4)]
// CHECK:STDOUT:       %impl.elem0.loc14_21.1: type = impl_witness_access constants.%Y.lookup_impl_witness.838, element0 [symbolic = %impl.elem0.loc14_21.2 (constants.%impl.elem0.96e)]
// CHECK:STDOUT:       %.loc14_27.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc14_27.2: type = converted %.loc14_27.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %.loc14_15: type = where_expr [symbolic = %Y_where.type (constants.%Y_where.type.228)] {
// CHECK:STDOUT:         requirement_base_facet_type %Y.ref
// CHECK:STDOUT:         requirement_rewrite %impl.elem0.loc14_21.1, %.loc14_27.2
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Outer.46c
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
// CHECK:STDOUT: generic fn @Outer.G(@Outer.%OuterParam.loc6_23.2: %Z1.type, %H.loc11_16.2: @Outer.G.%Y_where.type (%Y_where.type.228)) {
// CHECK:STDOUT:   %OuterParam: %Z1.type = symbolic_binding OuterParam, 0 [symbolic = %OuterParam (constants.%OuterParam)]
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y, @Y(%OuterParam)> [symbolic = %Y.type (constants.%Y.type.69f)]
// CHECK:STDOUT:   %.Self.loc11_21.1: @Outer.G.%Y.type (%Y.type.69f) = symbolic_binding .Self [symbolic = %.Self.loc11_21.1 (constants.%.Self.766)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Y.type [symbolic = %require_complete (constants.%require_complete.b56)]
// CHECK:STDOUT:   %.Self.as_type.loc11_27.1: type = facet_access_type %.Self.loc11_21.1 [symbolic = %.Self.as_type.loc11_27.1 (constants.%.Self.as_type.9c1)]
// CHECK:STDOUT:   %Y.assoc_type: type = assoc_entity_type @Y, @Y(%OuterParam) [symbolic = %Y.assoc_type (constants.%Y.assoc_type.10f)]
// CHECK:STDOUT:   %assoc0: @Outer.G.%Y.assoc_type (%Y.assoc_type.10f) = assoc_entity element0, @Y.WithSelf.%T [symbolic = %assoc0 (constants.%assoc0.5a4)]
// CHECK:STDOUT:   %Y.lookup_impl_witness: <witness> = lookup_impl_witness %.Self.loc11_21.1, @Y, @Y(%OuterParam) [symbolic = %Y.lookup_impl_witness (constants.%Y.lookup_impl_witness.838)]
// CHECK:STDOUT:   %impl.elem0.loc11_27.1: type = impl_witness_access %Y.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc11_27.1 (constants.%impl.elem0.96e)]
// CHECK:STDOUT:   %Y_where.type: type = facet_type <@Y, @Y(%OuterParam) where %impl.elem0.loc11_27.1 = constants.%empty_tuple.type> [symbolic = %Y_where.type (constants.%Y_where.type.228)]
// CHECK:STDOUT:   %H.loc11_16.1: @Outer.G.%Y_where.type (%Y_where.type.228) = symbolic_binding H, 1 [symbolic = %H.loc11_16.1 (constants.%H)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Y_where.type [symbolic = %pattern_type (constants.%pattern_type.ab5)]
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
// CHECK:STDOUT:   %impl.elem0.loc21_19: %.849 = impl_witness_access constants.%BitAndWith.impl_witness, element0 [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:   %bound_method.loc21_19: <bound method> = bound_method %Z1.ref.loc21_16, %impl.elem0.loc21_19 [concrete = constants.%type.as.BitAndWith.impl.Op.bound]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.call.loc21_19: init type = call %bound_method.loc21_19(%Z1.ref.loc21_16, %Z2.ref.loc21_21) [concrete = constants.%facet_type]
// CHECK:STDOUT:   %.loc21_23.1: type = value_of_initializer %type.as.BitAndWith.impl.Op.call.loc21_19 [concrete = constants.%facet_type]
// CHECK:STDOUT:   %.loc21_23.2: type = converted %type.as.BitAndWith.impl.Op.call.loc21_19, %.loc21_23.1 [concrete = constants.%facet_type]
// CHECK:STDOUT:   %facet_value.loc21_12: %facet_type = facet_value constants.%empty_tuple.type, (constants.%Z1.impl_witness, constants.%Z2.impl_witness) [concrete = constants.%facet_value.f54]
// CHECK:STDOUT:   %.loc21_12: %facet_type = converted %.loc21_10, %facet_value.loc21_12 [concrete = constants.%facet_value.f54]
// CHECK:STDOUT:   %as_type.loc21_24: type = facet_access_type %.loc21_12 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %Z1.facet.loc21_24: %Z1.type = facet_value %as_type.loc21_24, (constants.%Z1.impl_witness) [concrete = constants.%Z1.facet]
// CHECK:STDOUT:   %.loc21_24: %Z1.type = converted %.loc21_12, %Z1.facet.loc21_24 [concrete = constants.%Z1.facet]
// CHECK:STDOUT:   %Outer.loc21_24: type = class_type @Outer, @Outer(constants.%Z1.facet) [concrete = constants.%Outer.2a6]
// CHECK:STDOUT:   %.loc21_25: %Outer.G.type.9f4 = specific_constant @Outer.%Outer.G.decl, @Outer(constants.%Z1.facet) [concrete = constants.%Outer.G.268]
// CHECK:STDOUT:   %G.ref: %Outer.G.type.9f4 = name_ref G, %.loc21_25 [concrete = constants.%Outer.G.268]
// CHECK:STDOUT:   %Outer.ref.loc21_28: %Outer.type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:   %.loc21_35: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %Z1.ref.loc21_41: type = name_ref Z1, file.%Z1.decl [concrete = constants.%Z1.type]
// CHECK:STDOUT:   %Z2.ref.loc21_46: type = name_ref Z2, file.%Z2.decl [concrete = constants.%Z2.type]
// CHECK:STDOUT:   %impl.elem0.loc21_44: %.849 = impl_witness_access constants.%BitAndWith.impl_witness, element0 [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:   %bound_method.loc21_44: <bound method> = bound_method %Z1.ref.loc21_41, %impl.elem0.loc21_44 [concrete = constants.%type.as.BitAndWith.impl.Op.bound]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.call.loc21_44: init type = call %bound_method.loc21_44(%Z1.ref.loc21_41, %Z2.ref.loc21_46) [concrete = constants.%facet_type]
// CHECK:STDOUT:   %.loc21_48.1: type = value_of_initializer %type.as.BitAndWith.impl.Op.call.loc21_44 [concrete = constants.%facet_type]
// CHECK:STDOUT:   %.loc21_48.2: type = converted %type.as.BitAndWith.impl.Op.call.loc21_44, %.loc21_48.1 [concrete = constants.%facet_type]
// CHECK:STDOUT:   %facet_value.loc21_37: %facet_type = facet_value constants.%empty_tuple.type, (constants.%Z1.impl_witness, constants.%Z2.impl_witness) [concrete = constants.%facet_value.f54]
// CHECK:STDOUT:   %.loc21_37: %facet_type = converted %.loc21_35, %facet_value.loc21_37 [concrete = constants.%facet_value.f54]
// CHECK:STDOUT:   %as_type.loc21_49: type = facet_access_type %.loc21_37 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %Z1.facet.loc21_49: %Z1.type = facet_value %as_type.loc21_49, (constants.%Z1.impl_witness) [concrete = constants.%Z1.facet]
// CHECK:STDOUT:   %.loc21_49: %Z1.type = converted %.loc21_37, %Z1.facet.loc21_49 [concrete = constants.%Z1.facet]
// CHECK:STDOUT:   %Outer.loc21_49: type = class_type @Outer, @Outer(constants.%Z1.facet) [concrete = constants.%Outer.2a6]
// CHECK:STDOUT:   %.loc21_50: type = specific_constant @Outer.%C.decl, @Outer(constants.%Z1.facet) [concrete = constants.%C.c35]
// CHECK:STDOUT:   %C.ref: type = name_ref C, %.loc21_50 [concrete = constants.%C.c35]
// CHECK:STDOUT:   %facet_value.loc21_52: %Y_where.type.125 = facet_value constants.%C.c35, (constants.%Y.impl_witness.f15) [concrete = constants.%facet_value.4ab]
// CHECK:STDOUT:   %.loc21_52: %Y_where.type.125 = converted constants.%C.c35, %facet_value.loc21_52 [concrete = constants.%facet_value.4ab]
// CHECK:STDOUT:   %Outer.G.specific_fn: <specific function> = specific_function %G.ref, @Outer.G(constants.%Z1.facet, constants.%facet_value.4ab) [concrete = constants.%Outer.G.specific_fn]
// CHECK:STDOUT:   %Outer.G.call: init %empty_tuple.type = call %Outer.G.specific_fn()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z1.WithSelf(constants.%Self.641) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z2.WithSelf(constants.%Self.b23) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%OuterParam) {
// CHECK:STDOUT:   %OuterParam.loc6_23.1 => constants.%OuterParam
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Y.type => constants.%Y.type.69f
// CHECK:STDOUT:   %Outer.G.type => constants.%Outer.G.type.cd7
// CHECK:STDOUT:   %Outer.G => constants.%Outer.G.639
// CHECK:STDOUT:   %C => constants.%C.998
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y(constants.%OuterParam) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %OuterParam => constants.%OuterParam
// CHECK:STDOUT:   %Y.type => constants.%Y.type.69f
// CHECK:STDOUT:   %Self.loc7_15.2 => constants.%Self.3c2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y.WithSelf(constants.%OuterParam, constants.%Self.3c2) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %OuterParam => constants.%OuterParam
// CHECK:STDOUT:   %Y.assoc_type => constants.%Y.assoc_type.10f
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.5a4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y.WithSelf(constants.%OuterParam, constants.%.Self.766) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %OuterParam => constants.%OuterParam
// CHECK:STDOUT:   %Y.assoc_type => constants.%Y.assoc_type.10f
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.5a4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer.G(constants.%OuterParam, constants.%H) {
// CHECK:STDOUT:   %OuterParam => constants.%OuterParam
// CHECK:STDOUT:   %Y.type => constants.%Y.type.69f
// CHECK:STDOUT:   %.Self.loc11_21.1 => constants.%.Self.766
// CHECK:STDOUT:   %require_complete => constants.%require_complete.b56
// CHECK:STDOUT:   %.Self.as_type.loc11_27.1 => constants.%.Self.as_type.9c1
// CHECK:STDOUT:   %Y.assoc_type => constants.%Y.assoc_type.10f
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.5a4
// CHECK:STDOUT:   %Y.lookup_impl_witness => constants.%Y.lookup_impl_witness.838
// CHECK:STDOUT:   %impl.elem0.loc11_27.1 => constants.%impl.elem0.96e
// CHECK:STDOUT:   %Y_where.type => constants.%Y_where.type.228
// CHECK:STDOUT:   %H.loc11_16.1 => constants.%H
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.ab5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%OuterParam) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Y.impl(constants.%OuterParam) {
// CHECK:STDOUT:   %OuterParam => constants.%OuterParam
// CHECK:STDOUT:   %C => constants.%C.998
// CHECK:STDOUT:   %Y.type => constants.%Y.type.69f
// CHECK:STDOUT:   %.Self.loc14_15.2 => constants.%.Self.766
// CHECK:STDOUT:   %require_complete.loc14_21 => constants.%require_complete.b56
// CHECK:STDOUT:   %.Self.as_type.loc14_21.2 => constants.%.Self.as_type.9c1
// CHECK:STDOUT:   %Y.assoc_type => constants.%Y.assoc_type.10f
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.5a4
// CHECK:STDOUT:   %Y.lookup_impl_witness => constants.%Y.lookup_impl_witness.838
// CHECK:STDOUT:   %impl.elem0.loc14_21.2 => constants.%impl.elem0.96e
// CHECK:STDOUT:   %Y_where.type => constants.%Y_where.type.228
// CHECK:STDOUT:   %Y.impl_witness.loc14_29.2 => constants.%Y.impl_witness.79e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y.WithSelf(constants.%OuterParam, constants.%Y.facet.8a6) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %OuterParam => constants.%OuterParam
// CHECK:STDOUT:   %Y.assoc_type => constants.%Y.assoc_type.10f
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.5a4
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
// CHECK:STDOUT:   %OuterParam.loc6_23.1 => constants.%Z1.facet
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Y.type => constants.%Y.type.f5b
// CHECK:STDOUT:   %Outer.G.type => constants.%Outer.G.type.9f4
// CHECK:STDOUT:   %Outer.G => constants.%Outer.G.268
// CHECK:STDOUT:   %C => constants.%C.c35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y(constants.%Z1.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %OuterParam => constants.%Z1.facet
// CHECK:STDOUT:   %Y.type => constants.%Y.type.f5b
// CHECK:STDOUT:   %Self.loc7_15.2 => constants.%Self.2d8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%Z1.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Y.impl(constants.%Z1.facet) {
// CHECK:STDOUT:   %OuterParam => constants.%Z1.facet
// CHECK:STDOUT:   %C => constants.%C.c35
// CHECK:STDOUT:   %Y.type => constants.%Y.type.f5b
// CHECK:STDOUT:   %.Self.loc14_15.2 => constants.%.Self.d74
// CHECK:STDOUT:   %require_complete.loc14_21 => constants.%complete_type.e54
// CHECK:STDOUT:   %.Self.as_type.loc14_21.2 => constants.%.Self.as_type.194
// CHECK:STDOUT:   %Y.assoc_type => constants.%Y.assoc_type.845
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.933
// CHECK:STDOUT:   %Y.lookup_impl_witness => constants.%Y.lookup_impl_witness.25a
// CHECK:STDOUT:   %impl.elem0.loc14_21.2 => constants.%impl.elem0.581
// CHECK:STDOUT:   %Y_where.type => constants.%Y_where.type.125
// CHECK:STDOUT:   %Y.impl_witness.loc14_29.2 => constants.%Y.impl_witness.f15
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc14_15 => constants.%complete_type.f38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y.WithSelf(constants.%Z1.facet, constants.%Self.3c2) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %OuterParam => constants.%Z1.facet
// CHECK:STDOUT:   %Y.assoc_type => constants.%Y.assoc_type.845
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.933
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer.G(constants.%Z1.facet, constants.%facet_value.4ab) {
// CHECK:STDOUT:   %OuterParam => constants.%Z1.facet
// CHECK:STDOUT:   %Y.type => constants.%Y.type.f5b
// CHECK:STDOUT:   %.Self.loc11_21.1 => constants.%.Self.d74
// CHECK:STDOUT:   %require_complete => constants.%complete_type.e54
// CHECK:STDOUT:   %.Self.as_type.loc11_27.1 => constants.%.Self.as_type.194
// CHECK:STDOUT:   %Y.assoc_type => constants.%Y.assoc_type.845
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.933
// CHECK:STDOUT:   %Y.lookup_impl_witness => constants.%Y.lookup_impl_witness.25a
// CHECK:STDOUT:   %impl.elem0.loc11_27.1 => constants.%impl.elem0.581
// CHECK:STDOUT:   %Y_where.type => constants.%Y_where.type.125
// CHECK:STDOUT:   %H.loc11_16.1 => constants.%facet_value.4ab
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.1f9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:

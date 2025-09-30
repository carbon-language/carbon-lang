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
//    value `H` has a specific with a `BindSymbolicName` for the generic
//    parameter `OuterParam`.
//
// 2. The use of `.T` in the rewrite constraint does a lookup for `.Self` which
//    finds a `FacetType` that includes the interface `Y` which is _also inside_
//    the generic `Outer` class. This results in a `LookupImplWitness`
//    instruction where the query `SpecificInterface` has a specific argument
//    that is a `BindSymbolicName` for the enclosing generic's parameter
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
//    query `SpecificInterface` that has a `BindSymbolicName`. When
//    substituting, the `BindSymbolicName` is changed to the concrete `()` from
//    the caller's specific. Then the `LookupImplWitness` instruction is
//    evaluated again with its new values.
//
// What used to go wrong here:
//
// 5. The `LookupImplWitness` with the concrete `()` in the specific for the
//    query interface attempts to find a witness from the facet type of `H`. The
//    facet type has a `SpecificInterface` with `BindSymbolicName` in the same
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

  fn G(H:! Y where .T = ()) {}
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

  fn G(H:! Y where .T = ()) {}

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
// CHECK:STDOUT:   %.Self.659: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %OuterParam: type = bind_symbolic_name OuterParam, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Outer.type: type = generic_class_type @Outer [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Outer.generic: %Outer.type = struct_value () [concrete]
// CHECK:STDOUT:   %Outer.9d6: type = class_type @Outer, @Outer(%OuterParam) [symbolic]
// CHECK:STDOUT:   %Y.type.420: type = facet_type <@Y, @Y(%OuterParam)> [symbolic]
// CHECK:STDOUT:   %Self.42a: %Y.type.420 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Y.assoc_type.56e: type = assoc_entity_type @Y, @Y(%OuterParam) [symbolic]
// CHECK:STDOUT:   %assoc0.044: %Y.assoc_type.56e = assoc_entity element0, @Y.%T [symbolic]
// CHECK:STDOUT:   %.Self.5c8: %Y.type.420 = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Y.type.420 [symbolic]
// CHECK:STDOUT:   %.Self.as_type.667: type = facet_access_type %.Self.5c8 [symbolic]
// CHECK:STDOUT:   %Y.lookup_impl_witness.a57: <witness> = lookup_impl_witness %.Self.5c8, @Y, @Y(%OuterParam) [symbolic]
// CHECK:STDOUT:   %impl.elem0.a37: type = impl_witness_access %Y.lookup_impl_witness.a57, element0 [symbolic]
// CHECK:STDOUT:   %Y_where.type.197: type = facet_type <@Y, @Y(%OuterParam) where %impl.elem0.a37 = %empty_tuple.type> [symbolic]
// CHECK:STDOUT:   %H: %Y_where.type.197 = bind_symbolic_name H, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.ce6: type = pattern_type %Y_where.type.197 [symbolic]
// CHECK:STDOUT:   %Outer.G.type.d49: type = fn_type @Outer.G, @Outer(%OuterParam) [symbolic]
// CHECK:STDOUT:   %Outer.G.f3c: %Outer.G.type.d49 = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %Outer.8f5: type = class_type @Outer, @Outer(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %Y.type.c91: type = facet_type <@Y, @Y(%empty_tuple.type)> [concrete]
// CHECK:STDOUT:   %Outer.G.type.96c: type = fn_type @Outer.G, @Outer(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %Outer.G.510: %Outer.G.type.96c = struct_value () [concrete]
// CHECK:STDOUT:   %.Self.0fc: %Y.type.c91 = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %Self.177: %Y.type.c91 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Y.assoc_type.06e: type = assoc_entity_type @Y, @Y(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %assoc0.0a6: %Y.assoc_type.06e = assoc_entity element0, @Y.%T [concrete]
// CHECK:STDOUT:   %.Self.as_type.2b9: type = facet_access_type %.Self.0fc [symbolic_self]
// CHECK:STDOUT:   %Y.lookup_impl_witness.1fc: <witness> = lookup_impl_witness %.Self.0fc, @Y, @Y(%empty_tuple.type) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.f73: type = impl_witness_access %Y.lookup_impl_witness.1fc, element0 [symbolic_self]
// CHECK:STDOUT:   %Y_where.type.145: type = facet_type <@Y, @Y(%empty_tuple.type) where %impl.elem0.f73 = %empty_tuple.type> [concrete]
// CHECK:STDOUT:   %Y.impl_witness: <witness> = impl_witness file.%Y.impl_witness_table [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value: %Y_where.type.145 = facet_value %C, (%Y.impl_witness) [concrete]
// CHECK:STDOUT:   %complete_type.315: <witness> = complete_type_witness %Y.type.c91 [concrete]
// CHECK:STDOUT:   %pattern_type.123: type = pattern_type %Y_where.type.145 [concrete]
// CHECK:STDOUT:   %Outer.G.specific_fn: <specific function> = specific_function %Outer.G.510, @Outer.G(%empty_tuple.type, %facet_value) [concrete]
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
// CHECK:STDOUT:     %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self.659]
// CHECK:STDOUT:     %OuterParam.loc49_13.2: type = bind_symbolic_name OuterParam, 0 [symbolic = %OuterParam.loc49_13.1 (constants.%OuterParam)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @C.as.Y.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %Outer.ref: %Outer.type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:     %.loc58_18: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc58_19: type = converted %.loc58_18, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %Outer: type = class_type @Outer, @Outer(constants.%empty_tuple.type) [concrete = constants.%Outer.8f5]
// CHECK:STDOUT:     %.loc58_20: type = specific_constant @Outer.%Y.decl, @Outer(constants.%empty_tuple.type) [concrete = constants.%Y.type.c91]
// CHECK:STDOUT:     %Y.ref: type = name_ref Y, %.loc58_20 [concrete = constants.%Y.type.c91]
// CHECK:STDOUT:     %.Self: %Y.type.c91 = bind_symbolic_name .Self [symbolic_self = constants.%.Self.0fc]
// CHECK:STDOUT:     %.Self.ref: %Y.type.c91 = name_ref .Self, %.Self [symbolic_self = constants.%.Self.0fc]
// CHECK:STDOUT:     %.loc58_29.1: %Y.assoc_type.06e = specific_constant @T.%assoc0, @Y(constants.%empty_tuple.type) [concrete = constants.%assoc0.0a6]
// CHECK:STDOUT:     %T.ref: %Y.assoc_type.06e = name_ref T, %.loc58_29.1 [concrete = constants.%assoc0.0a6]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type.2b9]
// CHECK:STDOUT:     %.loc58_29.2: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type.2b9]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%Y.lookup_impl_witness.1fc, element0 [symbolic_self = constants.%impl.elem0.f73]
// CHECK:STDOUT:     %.loc58_35.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc58_35.2: type = converted %.loc58_35.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc58_23: type = where_expr %.Self [concrete = constants.%Y_where.type.145] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%Y.type.c91
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %.loc58_35.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Y.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @C.as.Y.impl [concrete]
// CHECK:STDOUT:   %Y.impl_witness: <witness> = impl_witness %Y.impl_witness_table [concrete = constants.%Y.impl_witness]
// CHECK:STDOUT:   %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Y(@Outer.%OuterParam.loc49_13.2: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %OuterParam: type = bind_symbolic_name OuterParam, 0 [symbolic = %OuterParam (constants.%OuterParam)]
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y, @Y(%OuterParam)> [symbolic = %Y.type (constants.%Y.type.420)]
// CHECK:STDOUT:   %Self.2: @Y.%Y.type (%Y.type.420) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.42a)]
// CHECK:STDOUT:   %Y.assoc_type: type = assoc_entity_type @Y, @Y(%OuterParam) [symbolic = %Y.assoc_type (constants.%Y.assoc_type.56e)]
// CHECK:STDOUT:   %assoc0: @Y.%Y.assoc_type (%Y.assoc_type.56e) = assoc_entity element0, %T [symbolic = %assoc0 (constants.%assoc0.044)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Y.%Y.type (%Y.type.420) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.42a)]
// CHECK:STDOUT:     %T: type = assoc_const_decl @T [concrete] {
// CHECK:STDOUT:       %assoc0: @Y.%Y.assoc_type (%Y.assoc_type.56e) = assoc_entity element0, @Y.%T [symbolic = @Y.%assoc0 (constants.%assoc0.044)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .T = @T.%assoc0
// CHECK:STDOUT:     witness = (%T)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @T(@Outer.%OuterParam.loc49_13.2: type, @Y.%Self.1: @Y.%Y.type (%Y.type.420)) {
// CHECK:STDOUT:   assoc_const T:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.Y.impl: %C.ref as %.loc58_23 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%Y.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Outer(%OuterParam.loc49_13.2: type) {
// CHECK:STDOUT:   %OuterParam.loc49_13.1: type = bind_symbolic_name OuterParam, 0 [symbolic = %OuterParam.loc49_13.1 (constants.%OuterParam)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y, @Y(%OuterParam.loc49_13.1)> [symbolic = %Y.type (constants.%Y.type.420)]
// CHECK:STDOUT:   %Outer.G.type: type = fn_type @Outer.G, @Outer(%OuterParam.loc49_13.1) [symbolic = %Outer.G.type (constants.%Outer.G.type.d49)]
// CHECK:STDOUT:   %Outer.G: @Outer.%Outer.G.type (%Outer.G.type.d49) = struct_value () [symbolic = %Outer.G (constants.%Outer.G.f3c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Y.decl: type = interface_decl @Y [symbolic = @Outer.%Y.type (constants.%Y.type.420)] {} {}
// CHECK:STDOUT:     %Outer.G.decl: @Outer.%Outer.G.type (%Outer.G.type.d49) = fn_decl @Outer.G [symbolic = @Outer.%Outer.G (constants.%Outer.G.f3c)] {
// CHECK:STDOUT:       %H.patt: @Outer.G.%pattern_type (%pattern_type.ce6) = symbolic_binding_pattern H, 1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc54_14.1: type = splice_block %.loc54_14.2 [symbolic = %Y_where.type (constants.%Y_where.type.197)] {
// CHECK:STDOUT:         %.Self.2: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self.659]
// CHECK:STDOUT:         %.loc54_12: type = specific_constant @Outer.%Y.decl, @Outer(constants.%OuterParam) [symbolic = %Y.type (constants.%Y.type.420)]
// CHECK:STDOUT:         %Y.ref: type = name_ref Y, %.loc54_12 [symbolic = %Y.type (constants.%Y.type.420)]
// CHECK:STDOUT:         %.Self.3: @Outer.G.%Y.type (%Y.type.420) = bind_symbolic_name .Self [symbolic = %.Self.1 (constants.%.Self.5c8)]
// CHECK:STDOUT:         %.Self.ref: @Outer.G.%Y.type (%Y.type.420) = name_ref .Self, %.Self.3 [symbolic = %.Self.1 (constants.%.Self.5c8)]
// CHECK:STDOUT:         %.loc54_20.1: @Outer.G.%Y.assoc_type (%Y.assoc_type.56e) = specific_constant @T.%assoc0, @Y(constants.%OuterParam) [symbolic = %assoc0 (constants.%assoc0.044)]
// CHECK:STDOUT:         %T.ref: @Outer.G.%Y.assoc_type (%Y.assoc_type.56e) = name_ref T, %.loc54_20.1 [symbolic = %assoc0 (constants.%assoc0.044)]
// CHECK:STDOUT:         %.Self.as_type.loc54_20.2: type = facet_access_type %.Self.ref [symbolic = %.Self.as_type.loc54_20.1 (constants.%.Self.as_type.667)]
// CHECK:STDOUT:         %.loc54_20.2: type = converted %.Self.ref, %.Self.as_type.loc54_20.2 [symbolic = %.Self.as_type.loc54_20.1 (constants.%.Self.as_type.667)]
// CHECK:STDOUT:         %impl.elem0.loc54_20.2: type = impl_witness_access constants.%Y.lookup_impl_witness.a57, element0 [symbolic = %impl.elem0.loc54_20.1 (constants.%impl.elem0.a37)]
// CHECK:STDOUT:         %.loc54_26.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:         %.loc54_26.2: type = converted %.loc54_26.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:         %.loc54_14.2: type = where_expr %.Self.3 [symbolic = %Y_where.type (constants.%Y_where.type.197)] {
// CHECK:STDOUT:           requirement_base_facet_type constants.%Y.type.420
// CHECK:STDOUT:           requirement_rewrite %impl.elem0.loc54_20.2, %.loc54_26.2
// CHECK:STDOUT:         }
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %H.loc54_8.2: @Outer.G.%Y_where.type (%Y_where.type.197) = bind_symbolic_name H, 1 [symbolic = %H.loc54_8.1 (constants.%H)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Outer.9d6
// CHECK:STDOUT:     .Y = %Y.decl
// CHECK:STDOUT:     .G = %Outer.G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C;
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Outer.G(@Outer.%OuterParam.loc49_13.2: type, %H.loc54_8.2: @Outer.G.%Y_where.type (%Y_where.type.197)) {
// CHECK:STDOUT:   %OuterParam: type = bind_symbolic_name OuterParam, 0 [symbolic = %OuterParam (constants.%OuterParam)]
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y, @Y(%OuterParam)> [symbolic = %Y.type (constants.%Y.type.420)]
// CHECK:STDOUT:   %.Self.1: @Outer.G.%Y.type (%Y.type.420) = bind_symbolic_name .Self [symbolic = %.Self.1 (constants.%.Self.5c8)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Y.type [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %Y.assoc_type: type = assoc_entity_type @Y, @Y(%OuterParam) [symbolic = %Y.assoc_type (constants.%Y.assoc_type.56e)]
// CHECK:STDOUT:   %assoc0: @Outer.G.%Y.assoc_type (%Y.assoc_type.56e) = assoc_entity element0, @Y.%T [symbolic = %assoc0 (constants.%assoc0.044)]
// CHECK:STDOUT:   %.Self.as_type.loc54_20.1: type = facet_access_type %.Self.1 [symbolic = %.Self.as_type.loc54_20.1 (constants.%.Self.as_type.667)]
// CHECK:STDOUT:   %Y.lookup_impl_witness: <witness> = lookup_impl_witness %.Self.1, @Y, @Y(%OuterParam) [symbolic = %Y.lookup_impl_witness (constants.%Y.lookup_impl_witness.a57)]
// CHECK:STDOUT:   %impl.elem0.loc54_20.1: type = impl_witness_access %Y.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc54_20.1 (constants.%impl.elem0.a37)]
// CHECK:STDOUT:   %Y_where.type: type = facet_type <@Y, @Y(%OuterParam) where %impl.elem0.loc54_20.1 = constants.%empty_tuple.type> [symbolic = %Y_where.type (constants.%Y_where.type.197)]
// CHECK:STDOUT:   %H.loc54_8.1: @Outer.G.%Y_where.type (%Y_where.type.197) = bind_symbolic_name H, 1 [symbolic = %H.loc54_8.1 (constants.%H)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Y_where.type [symbolic = %pattern_type (constants.%pattern_type.ce6)]
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
// CHECK:STDOUT:   %.loc61_10: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc61_11: type = converted %.loc61_10, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %Outer: type = class_type @Outer, @Outer(constants.%empty_tuple.type) [concrete = constants.%Outer.8f5]
// CHECK:STDOUT:   %.loc61_12: %Outer.G.type.96c = specific_constant @Outer.%Outer.G.decl, @Outer(constants.%empty_tuple.type) [concrete = constants.%Outer.G.510]
// CHECK:STDOUT:   %G.ref: %Outer.G.type.96c = name_ref G, %.loc61_12 [concrete = constants.%Outer.G.510]
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %facet_value: %Y_where.type.145 = facet_value constants.%C, (constants.%Y.impl_witness) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc61_16: %Y_where.type.145 = converted constants.%C, %facet_value [concrete = constants.%facet_value]
// CHECK:STDOUT:   %Outer.G.specific_fn: <specific function> = specific_function %G.ref, @Outer.G(constants.%empty_tuple.type, constants.%facet_value) [concrete = constants.%Outer.G.specific_fn]
// CHECK:STDOUT:   %Outer.G.call: init %empty_tuple.type = call %Outer.G.specific_fn()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%OuterParam) {
// CHECK:STDOUT:   %OuterParam.loc49_13.1 => constants.%OuterParam
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Y.type => constants.%Y.type.420
// CHECK:STDOUT:   %Outer.G.type => constants.%Outer.G.type.d49
// CHECK:STDOUT:   %Outer.G => constants.%Outer.G.f3c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y(constants.%OuterParam) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %OuterParam => constants.%OuterParam
// CHECK:STDOUT:   %Y.type => constants.%Y.type.420
// CHECK:STDOUT:   %Self.2 => constants.%Self.42a
// CHECK:STDOUT:   %Y.assoc_type => constants.%Y.assoc_type.56e
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.044
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T(constants.%OuterParam, constants.%Self.42a) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T(constants.%OuterParam, constants.%.Self.5c8) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer.G(constants.%OuterParam, constants.%H) {
// CHECK:STDOUT:   %OuterParam => constants.%OuterParam
// CHECK:STDOUT:   %Y.type => constants.%Y.type.420
// CHECK:STDOUT:   %.Self.1 => constants.%.Self.5c8
// CHECK:STDOUT:   %require_complete => constants.%require_complete
// CHECK:STDOUT:   %Y.assoc_type => constants.%Y.assoc_type.56e
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.044
// CHECK:STDOUT:   %.Self.as_type.loc54_20.1 => constants.%.Self.as_type.667
// CHECK:STDOUT:   %Y.lookup_impl_witness => constants.%Y.lookup_impl_witness.a57
// CHECK:STDOUT:   %impl.elem0.loc54_20.1 => constants.%impl.elem0.a37
// CHECK:STDOUT:   %Y_where.type => constants.%Y_where.type.197
// CHECK:STDOUT:   %H.loc54_8.1 => constants.%H
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.ce6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%empty_tuple.type) {
// CHECK:STDOUT:   %OuterParam.loc49_13.1 => constants.%empty_tuple.type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Y.type => constants.%Y.type.c91
// CHECK:STDOUT:   %Outer.G.type => constants.%Outer.G.type.96c
// CHECK:STDOUT:   %Outer.G => constants.%Outer.G.510
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y(constants.%empty_tuple.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %OuterParam => constants.%empty_tuple.type
// CHECK:STDOUT:   %Y.type => constants.%Y.type.c91
// CHECK:STDOUT:   %Self.2 => constants.%Self.177
// CHECK:STDOUT:   %Y.assoc_type => constants.%Y.assoc_type.06e
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.0a6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T(constants.%empty_tuple.type, constants.%.Self.0fc) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer.G(constants.%empty_tuple.type, constants.%facet_value) {
// CHECK:STDOUT:   %OuterParam => constants.%empty_tuple.type
// CHECK:STDOUT:   %Y.type => constants.%Y.type.c91
// CHECK:STDOUT:   %.Self.1 => constants.%.Self.0fc
// CHECK:STDOUT:   %require_complete => constants.%complete_type.315
// CHECK:STDOUT:   %Y.assoc_type => constants.%Y.assoc_type.06e
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.0a6
// CHECK:STDOUT:   %.Self.as_type.loc54_20.1 => constants.%.Self.as_type.2b9
// CHECK:STDOUT:   %Y.lookup_impl_witness => constants.%Y.lookup_impl_witness.1fc
// CHECK:STDOUT:   %impl.elem0.loc54_20.1 => constants.%impl.elem0.f73
// CHECK:STDOUT:   %Y_where.type => constants.%Y_where.type.145
// CHECK:STDOUT:   %H.loc54_8.1 => constants.%facet_value
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.123
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- enclosing_specific_is_facet_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Z1.type: type = facet_type <@Z1> [concrete]
// CHECK:STDOUT:   %Self.588: %Z1.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Z2.type: type = facet_type <@Z2> [concrete]
// CHECK:STDOUT:   %Self.24a: %Z2.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.659: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %OuterParam: %Z1.type = bind_symbolic_name OuterParam, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.361: type = pattern_type %Z1.type [concrete]
// CHECK:STDOUT:   %Outer.type: type = generic_class_type @Outer [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Outer.generic: %Outer.type = struct_value () [concrete]
// CHECK:STDOUT:   %Outer.bbd: type = class_type @Outer, @Outer(%OuterParam) [symbolic]
// CHECK:STDOUT:   %Y.type.699: type = facet_type <@Y, @Y(%OuterParam)> [symbolic]
// CHECK:STDOUT:   %Self.763: %Y.type.699 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Y.assoc_type.c54: type = assoc_entity_type @Y, @Y(%OuterParam) [symbolic]
// CHECK:STDOUT:   %assoc0.fca: %Y.assoc_type.c54 = assoc_entity element0, @Y.%T [symbolic]
// CHECK:STDOUT:   %.Self.707: %Y.type.699 = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %require_complete.83f: <witness> = require_complete_type %Y.type.699 [symbolic]
// CHECK:STDOUT:   %.Self.as_type.29f: type = facet_access_type %.Self.707 [symbolic]
// CHECK:STDOUT:   %Y.lookup_impl_witness.5a6: <witness> = lookup_impl_witness %.Self.707, @Y, @Y(%OuterParam) [symbolic]
// CHECK:STDOUT:   %impl.elem0.31f: type = impl_witness_access %Y.lookup_impl_witness.5a6, element0 [symbolic]
// CHECK:STDOUT:   %Y_where.type.6e4: type = facet_type <@Y, @Y(%OuterParam) where %impl.elem0.31f = %empty_tuple.type> [symbolic]
// CHECK:STDOUT:   %H: %Y_where.type.6e4 = bind_symbolic_name H, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.576: type = pattern_type %Y_where.type.6e4 [symbolic]
// CHECK:STDOUT:   %Outer.G.type.6f1: type = fn_type @Outer.G, @Outer(%OuterParam) [symbolic]
// CHECK:STDOUT:   %Outer.G.551: %Outer.G.type.6f1 = struct_value () [symbolic]
// CHECK:STDOUT:   %C.47f: type = class_type @C, @C(%OuterParam) [symbolic]
// CHECK:STDOUT:   %require_complete.e67: <witness> = require_complete_type %Y_where.type.6e4 [symbolic]
// CHECK:STDOUT:   %Y.impl_witness.d06: <witness> = impl_witness @Outer.%Y.impl_witness_table, @C.as.Y.impl(%OuterParam) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Z1.impl_witness: <witness> = impl_witness file.%Z1.impl_witness_table [concrete]
// CHECK:STDOUT:   %Z2.impl_witness: <witness> = impl_witness file.%Z2.impl_witness_table [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndWith.type.f2e: type = generic_interface_type @BitAndWith [concrete]
// CHECK:STDOUT:   %BitAndWith.generic: %BitAndWith.type.f2e = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndWith.type.8a6: type = facet_type <@BitAndWith, @BitAndWith(type)> [concrete]
// CHECK:STDOUT:   %BitAndWith.Op.type.9a3: type = fn_type @BitAndWith.Op, @BitAndWith(type) [concrete]
// CHECK:STDOUT:   %BitAndWith.impl_witness: <witness> = impl_witness imports.%BitAndWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %BitAndWith.facet: %BitAndWith.type.8a6 = facet_value type, (%BitAndWith.impl_witness) [concrete]
// CHECK:STDOUT:   %.fa7: type = fn_type_with_self_type %BitAndWith.Op.type.9a3, %BitAndWith.facet [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.type: type = fn_type @type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op: %type.as.BitAndWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.bound: <bound method> = bound_method %Z1.type, %type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@Z1 & @Z2> [concrete]
// CHECK:STDOUT:   %facet_value.c33: %facet_type = facet_value %empty_tuple.type, (%Z1.impl_witness, %Z2.impl_witness) [concrete]
// CHECK:STDOUT:   %Z1.facet: %Z1.type = facet_value %empty_tuple.type, (%Z1.impl_witness) [concrete]
// CHECK:STDOUT:   %Outer.63c: type = class_type @Outer, @Outer(%Z1.facet) [concrete]
// CHECK:STDOUT:   %Y.type.048: type = facet_type <@Y, @Y(%Z1.facet)> [concrete]
// CHECK:STDOUT:   %Outer.G.type.2b7: type = fn_type @Outer.G, @Outer(%Z1.facet) [concrete]
// CHECK:STDOUT:   %Outer.G.2d7: %Outer.G.type.2b7 = struct_value () [concrete]
// CHECK:STDOUT:   %C.b33: type = class_type @C, @C(%Z1.facet) [concrete]
// CHECK:STDOUT:   %.Self.f9f: %Y.type.048 = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %Y.lookup_impl_witness.9a6: <witness> = lookup_impl_witness %.Self.f9f, @Y, @Y(%Z1.facet) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.735: type = impl_witness_access %Y.lookup_impl_witness.9a6, element0 [symbolic_self]
// CHECK:STDOUT:   %Y_where.type.15e: type = facet_type <@Y, @Y(%Z1.facet) where %impl.elem0.735 = %empty_tuple.type> [concrete]
// CHECK:STDOUT:   %Self.2c0: %Y.type.048 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Y.assoc_type.38c: type = assoc_entity_type @Y, @Y(%Z1.facet) [concrete]
// CHECK:STDOUT:   %assoc0.84b: %Y.assoc_type.38c = assoc_entity element0, @Y.%T [concrete]
// CHECK:STDOUT:   %complete_type.7f3: <witness> = complete_type_witness %Y.type.048 [concrete]
// CHECK:STDOUT:   %.Self.as_type.b3e: type = facet_access_type %.Self.f9f [symbolic_self]
// CHECK:STDOUT:   %complete_type.01a: <witness> = complete_type_witness %Y_where.type.15e [concrete]
// CHECK:STDOUT:   %Y.impl_witness.917: <witness> = impl_witness @Outer.%Y.impl_witness_table, @C.as.Y.impl(%Z1.facet) [concrete]
// CHECK:STDOUT:   %facet_value.0c6: %Y_where.type.15e = facet_value %C.b33, (%Y.impl_witness.917) [concrete]
// CHECK:STDOUT:   %pattern_type.328: type = pattern_type %Y_where.type.15e [concrete]
// CHECK:STDOUT:   %Outer.G.specific_fn: <specific function> = specific_function %Outer.G.2d7, @Outer.G(%Z1.facet, %facet_value.0c6) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .BitAndWith = %Core.BitAndWith
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.BitAndWith: %BitAndWith.type.f2e = import_ref Core//prelude/parts/as, BitAndWith, loaded [concrete = constants.%BitAndWith.generic]
// CHECK:STDOUT:   %Core.import_ref.636: %type.as.BitAndWith.impl.Op.type = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, loaded [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:   %BitAndWith.impl_witness_table = impl_witness_table (%Core.import_ref.636), @type.as.BitAndWith.impl [concrete]
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
// CHECK:STDOUT:     %OuterParam.patt: %pattern_type.361 = symbolic_binding_pattern OuterParam, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6: type = splice_block %Z1.ref [concrete = constants.%Z1.type] {
// CHECK:STDOUT:       %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self.659]
// CHECK:STDOUT:       %Z1.ref: type = name_ref Z1, file.%Z1.decl [concrete = constants.%Z1.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %OuterParam.loc6_13.2: %Z1.type = bind_symbolic_name OuterParam, 0 [symbolic = %OuterParam.loc6_13.1 (constants.%OuterParam)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @empty_tuple.type.as.Z1.impl [concrete] {} {
// CHECK:STDOUT:     %.loc17_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc17_7.2: type = converted %.loc17_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %Z1.ref: type = name_ref Z1, file.%Z1.decl [concrete = constants.%Z1.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Z1.impl_witness_table = impl_witness_table (), @empty_tuple.type.as.Z1.impl [concrete]
// CHECK:STDOUT:   %Z1.impl_witness: <witness> = impl_witness %Z1.impl_witness_table [concrete = constants.%Z1.impl_witness]
// CHECK:STDOUT:   impl_decl @empty_tuple.type.as.Z2.impl [concrete] {} {
// CHECK:STDOUT:     %.loc18_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc18_7.2: type = converted %.loc18_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %Z2.ref: type = name_ref Z2, file.%Z2.decl [concrete = constants.%Z2.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Z2.impl_witness_table = impl_witness_table (), @empty_tuple.type.as.Z2.impl [concrete]
// CHECK:STDOUT:   %Z2.impl_witness: <witness> = impl_witness %Z2.impl_witness_table [concrete = constants.%Z2.impl_witness]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Z1 {
// CHECK:STDOUT:   %Self: %Z1.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.588]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Z2 {
// CHECK:STDOUT:   %Self: %Z2.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.24a]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Y(@Outer.%OuterParam.loc6_13.2: %Z1.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %OuterParam: %Z1.type = bind_symbolic_name OuterParam, 0 [symbolic = %OuterParam (constants.%OuterParam)]
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y, @Y(%OuterParam)> [symbolic = %Y.type (constants.%Y.type.699)]
// CHECK:STDOUT:   %Self.2: @Y.%Y.type (%Y.type.699) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.763)]
// CHECK:STDOUT:   %Y.assoc_type: type = assoc_entity_type @Y, @Y(%OuterParam) [symbolic = %Y.assoc_type (constants.%Y.assoc_type.c54)]
// CHECK:STDOUT:   %assoc0: @Y.%Y.assoc_type (%Y.assoc_type.c54) = assoc_entity element0, %T [symbolic = %assoc0 (constants.%assoc0.fca)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Y.%Y.type (%Y.type.699) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.763)]
// CHECK:STDOUT:     %T: type = assoc_const_decl @T [concrete] {
// CHECK:STDOUT:       %assoc0: @Y.%Y.assoc_type (%Y.assoc_type.c54) = assoc_entity element0, @Y.%T [symbolic = @Y.%assoc0 (constants.%assoc0.fca)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .T = @T.%assoc0
// CHECK:STDOUT:     witness = (%T)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @T(@Outer.%OuterParam.loc6_13.2: %Z1.type, @Y.%Self.1: @Y.%Y.type (%Y.type.699)) {
// CHECK:STDOUT:   assoc_const T:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.Y.impl(@Outer.%OuterParam.loc6_13.2: %Z1.type) {
// CHECK:STDOUT:   %OuterParam: %Z1.type = bind_symbolic_name OuterParam, 0 [symbolic = %OuterParam (constants.%OuterParam)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%OuterParam) [symbolic = %C (constants.%C.47f)]
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y, @Y(%OuterParam)> [symbolic = %Y.type (constants.%Y.type.699)]
// CHECK:STDOUT:   %.Self.2: @C.as.Y.impl.%Y.type (%Y.type.699) = bind_symbolic_name .Self [symbolic = %.Self.2 (constants.%.Self.707)]
// CHECK:STDOUT:   %require_complete.loc14_21: <witness> = require_complete_type %Y.type [symbolic = %require_complete.loc14_21 (constants.%require_complete.83f)]
// CHECK:STDOUT:   %Y.assoc_type: type = assoc_entity_type @Y, @Y(%OuterParam) [symbolic = %Y.assoc_type (constants.%Y.assoc_type.c54)]
// CHECK:STDOUT:   %assoc0: @C.as.Y.impl.%Y.assoc_type (%Y.assoc_type.c54) = assoc_entity element0, @Y.%T [symbolic = %assoc0 (constants.%assoc0.fca)]
// CHECK:STDOUT:   %.Self.as_type.loc14_21.2: type = facet_access_type %.Self.2 [symbolic = %.Self.as_type.loc14_21.2 (constants.%.Self.as_type.29f)]
// CHECK:STDOUT:   %Y.lookup_impl_witness: <witness> = lookup_impl_witness %.Self.2, @Y, @Y(%OuterParam) [symbolic = %Y.lookup_impl_witness (constants.%Y.lookup_impl_witness.5a6)]
// CHECK:STDOUT:   %impl.elem0.loc14_21.2: type = impl_witness_access %Y.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc14_21.2 (constants.%impl.elem0.31f)]
// CHECK:STDOUT:   %Y_where.type: type = facet_type <@Y, @Y(%OuterParam) where %impl.elem0.loc14_21.2 = constants.%empty_tuple.type> [symbolic = %Y_where.type (constants.%Y_where.type.6e4)]
// CHECK:STDOUT:   %require_complete.loc14_15: <witness> = require_complete_type %Y_where.type [symbolic = %require_complete.loc14_15 (constants.%require_complete.e67)]
// CHECK:STDOUT:   %Y.impl_witness: <witness> = impl_witness @Outer.%Y.impl_witness_table, @C.as.Y.impl(%OuterParam) [symbolic = %Y.impl_witness (constants.%Y.impl_witness.d06)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref as %.loc14_15 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = @Outer.%Y.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @empty_tuple.type.as.Z1.impl: %.loc17_7.2 as %Z1.ref {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%Z1.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @empty_tuple.type.as.Z2.impl: %.loc18_7.2 as %Z2.ref {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%Z2.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Outer(%OuterParam.loc6_13.2: %Z1.type) {
// CHECK:STDOUT:   %OuterParam.loc6_13.1: %Z1.type = bind_symbolic_name OuterParam, 0 [symbolic = %OuterParam.loc6_13.1 (constants.%OuterParam)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y, @Y(%OuterParam.loc6_13.1)> [symbolic = %Y.type (constants.%Y.type.699)]
// CHECK:STDOUT:   %Outer.G.type: type = fn_type @Outer.G, @Outer(%OuterParam.loc6_13.1) [symbolic = %Outer.G.type (constants.%Outer.G.type.6f1)]
// CHECK:STDOUT:   %Outer.G: @Outer.%Outer.G.type (%Outer.G.type.6f1) = struct_value () [symbolic = %Outer.G (constants.%Outer.G.551)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%OuterParam.loc6_13.1) [symbolic = %C (constants.%C.47f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Y.decl: type = interface_decl @Y [symbolic = @Outer.%Y.type (constants.%Y.type.699)] {} {}
// CHECK:STDOUT:     %Outer.G.decl: @Outer.%Outer.G.type (%Outer.G.type.6f1) = fn_decl @Outer.G [symbolic = @Outer.%Outer.G (constants.%Outer.G.551)] {
// CHECK:STDOUT:       %H.patt: @Outer.G.%pattern_type (%pattern_type.576) = symbolic_binding_pattern H, 1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc11_14.1: type = splice_block %.loc11_14.2 [symbolic = %Y_where.type (constants.%Y_where.type.6e4)] {
// CHECK:STDOUT:         %.Self.2: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self.659]
// CHECK:STDOUT:         %.loc11_12: type = specific_constant @Outer.%Y.decl, @Outer(constants.%OuterParam) [symbolic = %Y.type (constants.%Y.type.699)]
// CHECK:STDOUT:         %Y.ref: type = name_ref Y, %.loc11_12 [symbolic = %Y.type (constants.%Y.type.699)]
// CHECK:STDOUT:         %.Self.3: @Outer.G.%Y.type (%Y.type.699) = bind_symbolic_name .Self [symbolic = %.Self.1 (constants.%.Self.707)]
// CHECK:STDOUT:         %.Self.ref: @Outer.G.%Y.type (%Y.type.699) = name_ref .Self, %.Self.3 [symbolic = %.Self.1 (constants.%.Self.707)]
// CHECK:STDOUT:         %.loc11_20.1: @Outer.G.%Y.assoc_type (%Y.assoc_type.c54) = specific_constant @T.%assoc0, @Y(constants.%OuterParam) [symbolic = %assoc0 (constants.%assoc0.fca)]
// CHECK:STDOUT:         %T.ref: @Outer.G.%Y.assoc_type (%Y.assoc_type.c54) = name_ref T, %.loc11_20.1 [symbolic = %assoc0 (constants.%assoc0.fca)]
// CHECK:STDOUT:         %.Self.as_type.loc11_20.2: type = facet_access_type %.Self.ref [symbolic = %.Self.as_type.loc11_20.1 (constants.%.Self.as_type.29f)]
// CHECK:STDOUT:         %.loc11_20.2: type = converted %.Self.ref, %.Self.as_type.loc11_20.2 [symbolic = %.Self.as_type.loc11_20.1 (constants.%.Self.as_type.29f)]
// CHECK:STDOUT:         %impl.elem0.loc11_20.2: type = impl_witness_access constants.%Y.lookup_impl_witness.5a6, element0 [symbolic = %impl.elem0.loc11_20.1 (constants.%impl.elem0.31f)]
// CHECK:STDOUT:         %.loc11_26.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:         %.loc11_26.2: type = converted %.loc11_26.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:         %.loc11_14.2: type = where_expr %.Self.3 [symbolic = %Y_where.type (constants.%Y_where.type.6e4)] {
// CHECK:STDOUT:           requirement_base_facet_type constants.%Y.type.699
// CHECK:STDOUT:           requirement_rewrite %impl.elem0.loc11_20.2, %.loc11_26.2
// CHECK:STDOUT:         }
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %H.loc11_8.2: @Outer.G.%Y_where.type (%Y_where.type.6e4) = bind_symbolic_name H, 1 [symbolic = %H.loc11_8.1 (constants.%H)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %C.decl: type = class_decl @C [symbolic = @Outer.%C (constants.%C.47f)] {} {}
// CHECK:STDOUT:     impl_decl @C.as.Y.impl [concrete] {} {
// CHECK:STDOUT:       %.loc14_8: type = specific_constant @Outer.%C.decl, @Outer(constants.%OuterParam) [symbolic = %C (constants.%C.47f)]
// CHECK:STDOUT:       %C.ref: type = name_ref C, %.loc14_8 [symbolic = %C (constants.%C.47f)]
// CHECK:STDOUT:       %.loc14_13: type = specific_constant @Outer.%Y.decl, @Outer(constants.%OuterParam) [symbolic = %Y.type (constants.%Y.type.699)]
// CHECK:STDOUT:       %Y.ref: type = name_ref Y, %.loc14_13 [symbolic = %Y.type (constants.%Y.type.699)]
// CHECK:STDOUT:       %.Self.1: @C.as.Y.impl.%Y.type (%Y.type.699) = bind_symbolic_name .Self [symbolic = %.Self.2 (constants.%.Self.707)]
// CHECK:STDOUT:       %.Self.ref: @C.as.Y.impl.%Y.type (%Y.type.699) = name_ref .Self, %.Self.1 [symbolic = %.Self.2 (constants.%.Self.707)]
// CHECK:STDOUT:       %.loc14_21.1: @C.as.Y.impl.%Y.assoc_type (%Y.assoc_type.c54) = specific_constant @T.%assoc0, @Y(constants.%OuterParam) [symbolic = %assoc0 (constants.%assoc0.fca)]
// CHECK:STDOUT:       %T.ref: @C.as.Y.impl.%Y.assoc_type (%Y.assoc_type.c54) = name_ref T, %.loc14_21.1 [symbolic = %assoc0 (constants.%assoc0.fca)]
// CHECK:STDOUT:       %.Self.as_type.loc14_21.1: type = facet_access_type %.Self.ref [symbolic = %.Self.as_type.loc14_21.2 (constants.%.Self.as_type.29f)]
// CHECK:STDOUT:       %.loc14_21.2: type = converted %.Self.ref, %.Self.as_type.loc14_21.1 [symbolic = %.Self.as_type.loc14_21.2 (constants.%.Self.as_type.29f)]
// CHECK:STDOUT:       %impl.elem0.loc14_21.1: type = impl_witness_access constants.%Y.lookup_impl_witness.5a6, element0 [symbolic = %impl.elem0.loc14_21.2 (constants.%impl.elem0.31f)]
// CHECK:STDOUT:       %.loc14_27.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc14_27.2: type = converted %.loc14_27.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %.loc14_15: type = where_expr %.Self.1 [symbolic = %Y_where.type (constants.%Y_where.type.6e4)] {
// CHECK:STDOUT:         requirement_base_facet_type constants.%Y.type.699
// CHECK:STDOUT:         requirement_rewrite %impl.elem0.loc14_21.1, %.loc14_27.2
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Y.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @C.as.Y.impl [concrete]
// CHECK:STDOUT:     %Y.impl_witness: <witness> = impl_witness %Y.impl_witness_table, @C.as.Y.impl(constants.%OuterParam) [symbolic = @C.as.Y.impl.%Y.impl_witness (constants.%Y.impl_witness.d06)]
// CHECK:STDOUT:     %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
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
// CHECK:STDOUT: generic class @C(@Outer.%OuterParam.loc6_13.2: %Z1.type) {
// CHECK:STDOUT:   class;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Outer.G(@Outer.%OuterParam.loc6_13.2: %Z1.type, %H.loc11_8.2: @Outer.G.%Y_where.type (%Y_where.type.6e4)) {
// CHECK:STDOUT:   %OuterParam: %Z1.type = bind_symbolic_name OuterParam, 0 [symbolic = %OuterParam (constants.%OuterParam)]
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y, @Y(%OuterParam)> [symbolic = %Y.type (constants.%Y.type.699)]
// CHECK:STDOUT:   %.Self.1: @Outer.G.%Y.type (%Y.type.699) = bind_symbolic_name .Self [symbolic = %.Self.1 (constants.%.Self.707)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Y.type [symbolic = %require_complete (constants.%require_complete.83f)]
// CHECK:STDOUT:   %Y.assoc_type: type = assoc_entity_type @Y, @Y(%OuterParam) [symbolic = %Y.assoc_type (constants.%Y.assoc_type.c54)]
// CHECK:STDOUT:   %assoc0: @Outer.G.%Y.assoc_type (%Y.assoc_type.c54) = assoc_entity element0, @Y.%T [symbolic = %assoc0 (constants.%assoc0.fca)]
// CHECK:STDOUT:   %.Self.as_type.loc11_20.1: type = facet_access_type %.Self.1 [symbolic = %.Self.as_type.loc11_20.1 (constants.%.Self.as_type.29f)]
// CHECK:STDOUT:   %Y.lookup_impl_witness: <witness> = lookup_impl_witness %.Self.1, @Y, @Y(%OuterParam) [symbolic = %Y.lookup_impl_witness (constants.%Y.lookup_impl_witness.5a6)]
// CHECK:STDOUT:   %impl.elem0.loc11_20.1: type = impl_witness_access %Y.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc11_20.1 (constants.%impl.elem0.31f)]
// CHECK:STDOUT:   %Y_where.type: type = facet_type <@Y, @Y(%OuterParam) where %impl.elem0.loc11_20.1 = constants.%empty_tuple.type> [symbolic = %Y_where.type (constants.%Y_where.type.6e4)]
// CHECK:STDOUT:   %H.loc11_8.1: @Outer.G.%Y_where.type (%Y_where.type.6e4) = bind_symbolic_name H, 1 [symbolic = %H.loc11_8.1 (constants.%H)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Y_where.type [symbolic = %pattern_type (constants.%pattern_type.576)]
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
// CHECK:STDOUT:   %.loc21_10: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %Z1.ref.loc21_16: type = name_ref Z1, file.%Z1.decl [concrete = constants.%Z1.type]
// CHECK:STDOUT:   %Z2.ref.loc21_21: type = name_ref Z2, file.%Z2.decl [concrete = constants.%Z2.type]
// CHECK:STDOUT:   %impl.elem0.loc21_19: %.fa7 = impl_witness_access constants.%BitAndWith.impl_witness, element0 [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:   %bound_method.loc21_19: <bound method> = bound_method %Z1.ref.loc21_16, %impl.elem0.loc21_19 [concrete = constants.%type.as.BitAndWith.impl.Op.bound]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.call.loc21_19: init type = call %bound_method.loc21_19(%Z1.ref.loc21_16, %Z2.ref.loc21_21) [concrete = constants.%facet_type]
// CHECK:STDOUT:   %.loc21_23.1: type = value_of_initializer %type.as.BitAndWith.impl.Op.call.loc21_19 [concrete = constants.%facet_type]
// CHECK:STDOUT:   %.loc21_23.2: type = converted %type.as.BitAndWith.impl.Op.call.loc21_19, %.loc21_23.1 [concrete = constants.%facet_type]
// CHECK:STDOUT:   %facet_value.loc21_12: %facet_type = facet_value constants.%empty_tuple.type, (constants.%Z1.impl_witness, constants.%Z2.impl_witness) [concrete = constants.%facet_value.c33]
// CHECK:STDOUT:   %.loc21_12: %facet_type = converted %.loc21_10, %facet_value.loc21_12 [concrete = constants.%facet_value.c33]
// CHECK:STDOUT:   %as_type.loc21_24: type = facet_access_type %.loc21_12 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %Z1.facet.loc21_24: %Z1.type = facet_value %as_type.loc21_24, (constants.%Z1.impl_witness) [concrete = constants.%Z1.facet]
// CHECK:STDOUT:   %.loc21_24: %Z1.type = converted %.loc21_12, %Z1.facet.loc21_24 [concrete = constants.%Z1.facet]
// CHECK:STDOUT:   %Outer.loc21_24: type = class_type @Outer, @Outer(constants.%Z1.facet) [concrete = constants.%Outer.63c]
// CHECK:STDOUT:   %.loc21_25: %Outer.G.type.2b7 = specific_constant @Outer.%Outer.G.decl, @Outer(constants.%Z1.facet) [concrete = constants.%Outer.G.2d7]
// CHECK:STDOUT:   %G.ref: %Outer.G.type.2b7 = name_ref G, %.loc21_25 [concrete = constants.%Outer.G.2d7]
// CHECK:STDOUT:   %Outer.ref.loc21_28: %Outer.type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:   %.loc21_35: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %Z1.ref.loc21_41: type = name_ref Z1, file.%Z1.decl [concrete = constants.%Z1.type]
// CHECK:STDOUT:   %Z2.ref.loc21_46: type = name_ref Z2, file.%Z2.decl [concrete = constants.%Z2.type]
// CHECK:STDOUT:   %impl.elem0.loc21_44: %.fa7 = impl_witness_access constants.%BitAndWith.impl_witness, element0 [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:   %bound_method.loc21_44: <bound method> = bound_method %Z1.ref.loc21_41, %impl.elem0.loc21_44 [concrete = constants.%type.as.BitAndWith.impl.Op.bound]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.call.loc21_44: init type = call %bound_method.loc21_44(%Z1.ref.loc21_41, %Z2.ref.loc21_46) [concrete = constants.%facet_type]
// CHECK:STDOUT:   %.loc21_48.1: type = value_of_initializer %type.as.BitAndWith.impl.Op.call.loc21_44 [concrete = constants.%facet_type]
// CHECK:STDOUT:   %.loc21_48.2: type = converted %type.as.BitAndWith.impl.Op.call.loc21_44, %.loc21_48.1 [concrete = constants.%facet_type]
// CHECK:STDOUT:   %facet_value.loc21_37: %facet_type = facet_value constants.%empty_tuple.type, (constants.%Z1.impl_witness, constants.%Z2.impl_witness) [concrete = constants.%facet_value.c33]
// CHECK:STDOUT:   %.loc21_37: %facet_type = converted %.loc21_35, %facet_value.loc21_37 [concrete = constants.%facet_value.c33]
// CHECK:STDOUT:   %as_type.loc21_49: type = facet_access_type %.loc21_37 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %Z1.facet.loc21_49: %Z1.type = facet_value %as_type.loc21_49, (constants.%Z1.impl_witness) [concrete = constants.%Z1.facet]
// CHECK:STDOUT:   %.loc21_49: %Z1.type = converted %.loc21_37, %Z1.facet.loc21_49 [concrete = constants.%Z1.facet]
// CHECK:STDOUT:   %Outer.loc21_49: type = class_type @Outer, @Outer(constants.%Z1.facet) [concrete = constants.%Outer.63c]
// CHECK:STDOUT:   %.loc21_50: type = specific_constant @Outer.%C.decl, @Outer(constants.%Z1.facet) [concrete = constants.%C.b33]
// CHECK:STDOUT:   %C.ref: type = name_ref C, %.loc21_50 [concrete = constants.%C.b33]
// CHECK:STDOUT:   %facet_value.loc21_52: %Y_where.type.15e = facet_value constants.%C.b33, (constants.%Y.impl_witness.917) [concrete = constants.%facet_value.0c6]
// CHECK:STDOUT:   %.loc21_52: %Y_where.type.15e = converted constants.%C.b33, %facet_value.loc21_52 [concrete = constants.%facet_value.0c6]
// CHECK:STDOUT:   %Outer.G.specific_fn: <specific function> = specific_function %G.ref, @Outer.G(constants.%Z1.facet, constants.%facet_value.0c6) [concrete = constants.%Outer.G.specific_fn]
// CHECK:STDOUT:   %Outer.G.call: init %empty_tuple.type = call %Outer.G.specific_fn()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%OuterParam) {
// CHECK:STDOUT:   %OuterParam.loc6_13.1 => constants.%OuterParam
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Y.type => constants.%Y.type.699
// CHECK:STDOUT:   %Outer.G.type => constants.%Outer.G.type.6f1
// CHECK:STDOUT:   %Outer.G => constants.%Outer.G.551
// CHECK:STDOUT:   %C => constants.%C.47f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y(constants.%OuterParam) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %OuterParam => constants.%OuterParam
// CHECK:STDOUT:   %Y.type => constants.%Y.type.699
// CHECK:STDOUT:   %Self.2 => constants.%Self.763
// CHECK:STDOUT:   %Y.assoc_type => constants.%Y.assoc_type.c54
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.fca
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T(constants.%OuterParam, constants.%Self.763) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T(constants.%OuterParam, constants.%.Self.707) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer.G(constants.%OuterParam, constants.%H) {
// CHECK:STDOUT:   %OuterParam => constants.%OuterParam
// CHECK:STDOUT:   %Y.type => constants.%Y.type.699
// CHECK:STDOUT:   %.Self.1 => constants.%.Self.707
// CHECK:STDOUT:   %require_complete => constants.%require_complete.83f
// CHECK:STDOUT:   %Y.assoc_type => constants.%Y.assoc_type.c54
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.fca
// CHECK:STDOUT:   %.Self.as_type.loc11_20.1 => constants.%.Self.as_type.29f
// CHECK:STDOUT:   %Y.lookup_impl_witness => constants.%Y.lookup_impl_witness.5a6
// CHECK:STDOUT:   %impl.elem0.loc11_20.1 => constants.%impl.elem0.31f
// CHECK:STDOUT:   %Y_where.type => constants.%Y_where.type.6e4
// CHECK:STDOUT:   %H.loc11_8.1 => constants.%H
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.576
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%OuterParam) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Y.impl(constants.%OuterParam) {
// CHECK:STDOUT:   %OuterParam => constants.%OuterParam
// CHECK:STDOUT:   %C => constants.%C.47f
// CHECK:STDOUT:   %Y.type => constants.%Y.type.699
// CHECK:STDOUT:   %.Self.2 => constants.%.Self.707
// CHECK:STDOUT:   %require_complete.loc14_21 => constants.%require_complete.83f
// CHECK:STDOUT:   %Y.assoc_type => constants.%Y.assoc_type.c54
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.fca
// CHECK:STDOUT:   %.Self.as_type.loc14_21.2 => constants.%.Self.as_type.29f
// CHECK:STDOUT:   %Y.lookup_impl_witness => constants.%Y.lookup_impl_witness.5a6
// CHECK:STDOUT:   %impl.elem0.loc14_21.2 => constants.%impl.elem0.31f
// CHECK:STDOUT:   %Y_where.type => constants.%Y_where.type.6e4
// CHECK:STDOUT:   %require_complete.loc14_15 => constants.%require_complete.e67
// CHECK:STDOUT:   %Y.impl_witness => constants.%Y.impl_witness.d06
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%Z1.facet) {
// CHECK:STDOUT:   %OuterParam.loc6_13.1 => constants.%Z1.facet
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Y.type => constants.%Y.type.048
// CHECK:STDOUT:   %Outer.G.type => constants.%Outer.G.type.2b7
// CHECK:STDOUT:   %Outer.G => constants.%Outer.G.2d7
// CHECK:STDOUT:   %C => constants.%C.b33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y(constants.%Z1.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %OuterParam => constants.%Z1.facet
// CHECK:STDOUT:   %Y.type => constants.%Y.type.048
// CHECK:STDOUT:   %Self.2 => constants.%Self.2c0
// CHECK:STDOUT:   %Y.assoc_type => constants.%Y.assoc_type.38c
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.84b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%Z1.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Y.impl(constants.%Z1.facet) {
// CHECK:STDOUT:   %OuterParam => constants.%Z1.facet
// CHECK:STDOUT:   %C => constants.%C.b33
// CHECK:STDOUT:   %Y.type => constants.%Y.type.048
// CHECK:STDOUT:   %.Self.2 => constants.%.Self.f9f
// CHECK:STDOUT:   %require_complete.loc14_21 => constants.%complete_type.7f3
// CHECK:STDOUT:   %Y.assoc_type => constants.%Y.assoc_type.38c
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.84b
// CHECK:STDOUT:   %.Self.as_type.loc14_21.2 => constants.%.Self.as_type.b3e
// CHECK:STDOUT:   %Y.lookup_impl_witness => constants.%Y.lookup_impl_witness.9a6
// CHECK:STDOUT:   %impl.elem0.loc14_21.2 => constants.%impl.elem0.735
// CHECK:STDOUT:   %Y_where.type => constants.%Y_where.type.15e
// CHECK:STDOUT:   %require_complete.loc14_15 => constants.%complete_type.01a
// CHECK:STDOUT:   %Y.impl_witness => constants.%Y.impl_witness.917
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer.G(constants.%Z1.facet, constants.%facet_value.0c6) {
// CHECK:STDOUT:   %OuterParam => constants.%Z1.facet
// CHECK:STDOUT:   %Y.type => constants.%Y.type.048
// CHECK:STDOUT:   %.Self.1 => constants.%.Self.f9f
// CHECK:STDOUT:   %require_complete => constants.%complete_type.7f3
// CHECK:STDOUT:   %Y.assoc_type => constants.%Y.assoc_type.38c
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.84b
// CHECK:STDOUT:   %.Self.as_type.loc11_20.1 => constants.%.Self.as_type.b3e
// CHECK:STDOUT:   %Y.lookup_impl_witness => constants.%Y.lookup_impl_witness.9a6
// CHECK:STDOUT:   %impl.elem0.loc11_20.1 => constants.%impl.elem0.735
// CHECK:STDOUT:   %Y_where.type => constants.%Y_where.type.15e
// CHECK:STDOUT:   %H.loc11_8.1 => constants.%facet_value.0c6
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.328
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:

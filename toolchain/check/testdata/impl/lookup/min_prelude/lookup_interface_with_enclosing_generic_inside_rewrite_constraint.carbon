// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/min_prelude/facet_types.carbon
// EXTRA-ARGS: --custom-core
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/lookup/min_prelude/lookup_interface_with_enclosing_generic_inside_rewrite_constraint.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/lookup/min_prelude/lookup_interface_with_enclosing_generic_inside_rewrite_constraint.carbon

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
// 7. The lookup in the `FacetType` of `H` needs to succeed. Having a more
//    specific value in the query interface's specific arguments is fine.
//    Instead of comparing for equality, look to see that the specific of the
//    query interface is *compatible* with the specific of the facet type's
//    `SpecificInterface`.

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
// CHECK:STDOUT:   %OuterParam: type = bind_symbolic_name OuterParam, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Outer.type: type = generic_class_type @Outer [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Outer.generic: %Outer.type = struct_value () [concrete]
// CHECK:STDOUT:   %Outer.9d6: type = class_type @Outer, @Outer(%OuterParam) [symbolic]
// CHECK:STDOUT:   %Y.type.49b: type = facet_type <@Y, @Y(%OuterParam)> [symbolic]
// CHECK:STDOUT:   %Self.fa7: %Y.type.49b = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Y.assoc_type.56e: type = assoc_entity_type @Y, @Y(%OuterParam) [symbolic]
// CHECK:STDOUT:   %assoc0.044: %Y.assoc_type.56e = assoc_entity element0, @Y.%T [symbolic]
// CHECK:STDOUT:   %.Self.c19: %Y.type.49b = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Y.type.49b [symbolic]
// CHECK:STDOUT:   %.Self.as_type.4b2: type = facet_access_type %.Self.c19 [symbolic]
// CHECK:STDOUT:   %Y.lookup_impl_witness.369: <witness> = lookup_impl_witness %.Self.c19, @Y, @Y(%OuterParam) [symbolic]
// CHECK:STDOUT:   %Y.facet.1b9: %Y.type.49b = facet_value %.Self.as_type.4b2, (%Y.lookup_impl_witness.369) [symbolic]
// CHECK:STDOUT:   %impl.elem0.409: type = impl_witness_access %Y.lookup_impl_witness.369, element0 [symbolic]
// CHECK:STDOUT:   %Y_where.type.324: type = facet_type <@Y, @Y(%OuterParam) where %impl.elem0.409 = %empty_tuple.type> [symbolic]
// CHECK:STDOUT:   %H: %Y_where.type.324 = bind_symbolic_name H, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.f71: type = pattern_type %Y_where.type.324 [symbolic]
// CHECK:STDOUT:   %G.type.d49: type = fn_type @G, @Outer(%OuterParam) [symbolic]
// CHECK:STDOUT:   %G.f3c: %G.type.d49 = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %Outer.8f5: type = class_type @Outer, @Outer(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %Y.type.080: type = facet_type <@Y, @Y(%empty_tuple.type)> [concrete]
// CHECK:STDOUT:   %G.type.96c: type = fn_type @G, @Outer(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %G.510: %G.type.96c = struct_value () [concrete]
// CHECK:STDOUT:   %.Self.37b: %Y.type.080 = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %Self.3a4: %Y.type.080 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Y.assoc_type.06e: type = assoc_entity_type @Y, @Y(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %assoc0.0a6: %Y.assoc_type.06e = assoc_entity element0, @Y.%T [concrete]
// CHECK:STDOUT:   %.Self.as_type.03a: type = facet_access_type %.Self.37b [symbolic_self]
// CHECK:STDOUT:   %Y.lookup_impl_witness.218: <witness> = lookup_impl_witness %.Self.37b, @Y, @Y(%empty_tuple.type) [symbolic_self]
// CHECK:STDOUT:   %Y.facet.1bc: %Y.type.080 = facet_value %.Self.as_type.03a, (%Y.lookup_impl_witness.218) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.a8c: type = impl_witness_access %Y.lookup_impl_witness.218, element0 [symbolic_self]
// CHECK:STDOUT:   %Y_where.type.8eb: type = facet_type <@Y, @Y(%empty_tuple.type) where %impl.elem0.a8c = %empty_tuple.type> [concrete]
// CHECK:STDOUT:   %Y.impl_witness: <witness> = impl_witness file.%Y.impl_witness_table [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Y.lookup_impl_witness.1ff: <witness> = lookup_impl_witness %.Self.c19, @Y, @Y(%empty_tuple.type) [symbolic]
// CHECK:STDOUT:   %impl.elem0.fd0: type = impl_witness_access %Y.lookup_impl_witness.1ff, element0 [symbolic]
// CHECK:STDOUT:   %Y_where.type.f47: type = facet_type <@Y, @Y(%empty_tuple.type) where %impl.elem0.fd0 = %empty_tuple.type> [symbolic]
// CHECK:STDOUT:   %facet_value: %Y_where.type.f47 = facet_value %C, (%Y.impl_witness) [symbolic]
// CHECK:STDOUT:   %complete_type.12f: <witness> = complete_type_witness %Y.type.080 [concrete]
// CHECK:STDOUT:   %pattern_type.ce4: type = pattern_type %Y_where.type.8eb [concrete]
// CHECK:STDOUT:   %G.specific_fn: <specific function> = specific_function %G.510, @G(%empty_tuple.type, %facet_value) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
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
// CHECK:STDOUT:     %OuterParam.loc49_13.1: type = bind_symbolic_name OuterParam, 0 [symbolic = %OuterParam.loc49_13.2 (constants.%OuterParam)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %Outer.ref: %Outer.type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:     %.loc58_18: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc58_19: type = converted %.loc58_18, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %Outer: type = class_type @Outer, @Outer(constants.%empty_tuple.type) [concrete = constants.%Outer.8f5]
// CHECK:STDOUT:     %.loc58_20: type = specific_constant @Outer.%Y.decl, @Outer(constants.%empty_tuple.type) [concrete = constants.%Y.type.080]
// CHECK:STDOUT:     %Y.ref: type = name_ref Y, %.loc58_20 [concrete = constants.%Y.type.080]
// CHECK:STDOUT:     %.Self: %Y.type.080 = bind_symbolic_name .Self [symbolic_self = constants.%.Self.37b]
// CHECK:STDOUT:     %.Self.ref: %Y.type.080 = name_ref .Self, %.Self [symbolic_self = constants.%.Self.37b]
// CHECK:STDOUT:     %.loc58_29.1: %Y.assoc_type.06e = specific_constant @T.%assoc0, @Y(constants.%empty_tuple.type) [concrete = constants.%assoc0.0a6]
// CHECK:STDOUT:     %T.ref: %Y.assoc_type.06e = name_ref T, %.loc58_29.1 [concrete = constants.%assoc0.0a6]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type.03a]
// CHECK:STDOUT:     %.loc58_29.2: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type.03a]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%Y.lookup_impl_witness.218, element0 [symbolic_self = constants.%impl.elem0.a8c]
// CHECK:STDOUT:     %.loc58_35.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc58_35.2: type = converted %.loc58_35.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc58_23: type = where_expr %.Self [concrete = constants.%Y_where.type.8eb] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %.loc58_35.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Y.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @impl [concrete]
// CHECK:STDOUT:   %Y.impl_witness: <witness> = impl_witness %Y.impl_witness_table [concrete = constants.%Y.impl_witness]
// CHECK:STDOUT:   %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Y(@Outer.%OuterParam.loc49_13.1: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %OuterParam: type = bind_symbolic_name OuterParam, 0 [symbolic = %OuterParam (constants.%OuterParam)]
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y, @Y(%OuterParam)> [symbolic = %Y.type (constants.%Y.type.49b)]
// CHECK:STDOUT:   %Self.2: @Y.%Y.type (%Y.type.49b) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.fa7)]
// CHECK:STDOUT:   %Y.assoc_type: type = assoc_entity_type @Y, @Y(%OuterParam) [symbolic = %Y.assoc_type (constants.%Y.assoc_type.56e)]
// CHECK:STDOUT:   %assoc0: @Y.%Y.assoc_type (%Y.assoc_type.56e) = assoc_entity element0, %T [symbolic = %assoc0 (constants.%assoc0.044)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Y.%Y.type (%Y.type.49b) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.fa7)]
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
// CHECK:STDOUT: generic assoc_const @T(@Outer.%OuterParam.loc49_13.1: type, @Y.%Self.1: @Y.%Y.type (%Y.type.49b)) {
// CHECK:STDOUT:   assoc_const T:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %C.ref as %.loc58_23 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%Y.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Outer(%OuterParam.loc49_13.1: type) {
// CHECK:STDOUT:   %OuterParam.loc49_13.2: type = bind_symbolic_name OuterParam, 0 [symbolic = %OuterParam.loc49_13.2 (constants.%OuterParam)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y, @Y(%OuterParam.loc49_13.2)> [symbolic = %Y.type (constants.%Y.type.49b)]
// CHECK:STDOUT:   %G.type: type = fn_type @G, @Outer(%OuterParam.loc49_13.2) [symbolic = %G.type (constants.%G.type.d49)]
// CHECK:STDOUT:   %G: @Outer.%G.type (%G.type.d49) = struct_value () [symbolic = %G (constants.%G.f3c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Y.decl: type = interface_decl @Y [symbolic = @Outer.%Y.type (constants.%Y.type.49b)] {} {}
// CHECK:STDOUT:     %G.decl: @Outer.%G.type (%G.type.d49) = fn_decl @G [symbolic = @Outer.%G (constants.%G.f3c)] {
// CHECK:STDOUT:       %H.patt: @G.%pattern_type (%pattern_type.f71) = symbolic_binding_pattern H, 1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc54_14.1: type = splice_block %.loc54_14.2 [symbolic = %Y_where.type (constants.%Y_where.type.324)] {
// CHECK:STDOUT:         %.loc54_12: type = specific_constant @Outer.%Y.decl, @Outer(constants.%OuterParam) [symbolic = %Y.type (constants.%Y.type.49b)]
// CHECK:STDOUT:         %Y.ref: type = name_ref Y, %.loc54_12 [symbolic = %Y.type (constants.%Y.type.49b)]
// CHECK:STDOUT:         %.Self.2: @G.%Y.type (%Y.type.49b) = bind_symbolic_name .Self [symbolic = %.Self.1 (constants.%.Self.c19)]
// CHECK:STDOUT:         %.Self.ref: @G.%Y.type (%Y.type.49b) = name_ref .Self, %.Self.2 [symbolic = %.Self.1 (constants.%.Self.c19)]
// CHECK:STDOUT:         %.loc54_20.1: @G.%Y.assoc_type (%Y.assoc_type.56e) = specific_constant @T.%assoc0, @Y(constants.%OuterParam) [symbolic = %assoc0 (constants.%assoc0.044)]
// CHECK:STDOUT:         %T.ref: @G.%Y.assoc_type (%Y.assoc_type.56e) = name_ref T, %.loc54_20.1 [symbolic = %assoc0 (constants.%assoc0.044)]
// CHECK:STDOUT:         %.Self.as_type.loc54_20.2: type = facet_access_type %.Self.ref [symbolic = %.Self.as_type.loc54_20.1 (constants.%.Self.as_type.4b2)]
// CHECK:STDOUT:         %.loc54_20.2: type = converted %.Self.ref, %.Self.as_type.loc54_20.2 [symbolic = %.Self.as_type.loc54_20.1 (constants.%.Self.as_type.4b2)]
// CHECK:STDOUT:         %impl.elem0.loc54_20.2: type = impl_witness_access constants.%Y.lookup_impl_witness.369, element0 [symbolic = %impl.elem0.loc54_20.1 (constants.%impl.elem0.409)]
// CHECK:STDOUT:         %.loc54_26.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:         %.loc54_26.2: type = converted %.loc54_26.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:         %.loc54_14.2: type = where_expr %.Self.2 [symbolic = %Y_where.type (constants.%Y_where.type.324)] {
// CHECK:STDOUT:           requirement_rewrite %impl.elem0.loc54_20.2, %.loc54_26.2
// CHECK:STDOUT:         }
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %H.loc54_8.2: @G.%Y_where.type (%Y_where.type.324) = bind_symbolic_name H, 1 [symbolic = %H.loc54_8.1 (constants.%H)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Outer.9d6
// CHECK:STDOUT:     .Y = %Y.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C;
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(@Outer.%OuterParam.loc49_13.1: type, %H.loc54_8.2: @G.%Y_where.type (%Y_where.type.324)) {
// CHECK:STDOUT:   %OuterParam: type = bind_symbolic_name OuterParam, 0 [symbolic = %OuterParam (constants.%OuterParam)]
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y, @Y(%OuterParam)> [symbolic = %Y.type (constants.%Y.type.49b)]
// CHECK:STDOUT:   %.Self.1: @G.%Y.type (%Y.type.49b) = bind_symbolic_name .Self [symbolic = %.Self.1 (constants.%.Self.c19)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Y.type [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %Y.assoc_type: type = assoc_entity_type @Y, @Y(%OuterParam) [symbolic = %Y.assoc_type (constants.%Y.assoc_type.56e)]
// CHECK:STDOUT:   %assoc0: @G.%Y.assoc_type (%Y.assoc_type.56e) = assoc_entity element0, @Y.%T [symbolic = %assoc0 (constants.%assoc0.044)]
// CHECK:STDOUT:   %.Self.as_type.loc54_20.1: type = facet_access_type %.Self.1 [symbolic = %.Self.as_type.loc54_20.1 (constants.%.Self.as_type.4b2)]
// CHECK:STDOUT:   %Y.lookup_impl_witness: <witness> = lookup_impl_witness %.Self.1, @Y, @Y(%OuterParam) [symbolic = %Y.lookup_impl_witness (constants.%Y.lookup_impl_witness.369)]
// CHECK:STDOUT:   %impl.elem0.loc54_20.1: type = impl_witness_access %Y.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc54_20.1 (constants.%impl.elem0.409)]
// CHECK:STDOUT:   %Y_where.type: type = facet_type <@Y, @Y(%OuterParam) where %impl.elem0.loc54_20.1 = constants.%empty_tuple.type> [symbolic = %Y_where.type (constants.%Y_where.type.324)]
// CHECK:STDOUT:   %H.loc54_8.1: @G.%Y_where.type (%Y_where.type.324) = bind_symbolic_name H, 1 [symbolic = %H.loc54_8.1 (constants.%H)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Y_where.type [symbolic = %pattern_type (constants.%pattern_type.f71)]
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
// CHECK:STDOUT:   %.loc61_12: %G.type.96c = specific_constant @Outer.%G.decl, @Outer(constants.%empty_tuple.type) [concrete = constants.%G.510]
// CHECK:STDOUT:   %G.ref: %G.type.96c = name_ref G, %.loc61_12 [concrete = constants.%G.510]
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %facet_value: %Y_where.type.f47 = facet_value constants.%C, (constants.%Y.impl_witness) [symbolic = constants.%facet_value]
// CHECK:STDOUT:   %.loc61_16: %Y_where.type.f47 = converted constants.%C, %facet_value [symbolic = constants.%facet_value]
// CHECK:STDOUT:   %G.specific_fn: <specific function> = specific_function %G.ref, @G(constants.%empty_tuple.type, constants.%facet_value) [symbolic = constants.%G.specific_fn]
// CHECK:STDOUT:   %G.call: init %empty_tuple.type = call %G.specific_fn()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%OuterParam) {
// CHECK:STDOUT:   %OuterParam.loc49_13.2 => constants.%OuterParam
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Y.type => constants.%Y.type.49b
// CHECK:STDOUT:   %G.type => constants.%G.type.d49
// CHECK:STDOUT:   %G => constants.%G.f3c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y(constants.%OuterParam) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %OuterParam => constants.%OuterParam
// CHECK:STDOUT:   %Y.type => constants.%Y.type.49b
// CHECK:STDOUT:   %Self.2 => constants.%Self.fa7
// CHECK:STDOUT:   %Y.assoc_type => constants.%Y.assoc_type.56e
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.044
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T(constants.%OuterParam, constants.%Self.fa7) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T(constants.%OuterParam, constants.%Y.facet.1b9) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%OuterParam, constants.%H) {
// CHECK:STDOUT:   %OuterParam => constants.%OuterParam
// CHECK:STDOUT:   %Y.type => constants.%Y.type.49b
// CHECK:STDOUT:   %.Self.1 => constants.%.Self.c19
// CHECK:STDOUT:   %require_complete => constants.%require_complete
// CHECK:STDOUT:   %Y.assoc_type => constants.%Y.assoc_type.56e
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.044
// CHECK:STDOUT:   %.Self.as_type.loc54_20.1 => constants.%.Self.as_type.4b2
// CHECK:STDOUT:   %Y.lookup_impl_witness => constants.%Y.lookup_impl_witness.369
// CHECK:STDOUT:   %impl.elem0.loc54_20.1 => constants.%impl.elem0.409
// CHECK:STDOUT:   %Y_where.type => constants.%Y_where.type.324
// CHECK:STDOUT:   %H.loc54_8.1 => constants.%H
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.f71
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%empty_tuple.type) {
// CHECK:STDOUT:   %OuterParam.loc49_13.2 => constants.%empty_tuple.type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Y.type => constants.%Y.type.080
// CHECK:STDOUT:   %G.type => constants.%G.type.96c
// CHECK:STDOUT:   %G => constants.%G.510
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y(constants.%empty_tuple.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %OuterParam => constants.%empty_tuple.type
// CHECK:STDOUT:   %Y.type => constants.%Y.type.080
// CHECK:STDOUT:   %Self.2 => constants.%Self.3a4
// CHECK:STDOUT:   %Y.assoc_type => constants.%Y.assoc_type.06e
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.0a6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T(constants.%empty_tuple.type, constants.%Y.facet.1bc) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%empty_tuple.type, constants.%facet_value) {
// CHECK:STDOUT:   %OuterParam => constants.%empty_tuple.type
// CHECK:STDOUT:   %Y.type => constants.%Y.type.080
// CHECK:STDOUT:   %.Self.1 => constants.%.Self.37b
// CHECK:STDOUT:   %require_complete => constants.%complete_type.12f
// CHECK:STDOUT:   %Y.assoc_type => constants.%Y.assoc_type.06e
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.0a6
// CHECK:STDOUT:   %.Self.as_type.loc54_20.1 => constants.%.Self.as_type.03a
// CHECK:STDOUT:   %Y.lookup_impl_witness => constants.%Y.lookup_impl_witness.218
// CHECK:STDOUT:   %impl.elem0.loc54_20.1 => constants.%impl.elem0.a8c
// CHECK:STDOUT:   %Y_where.type => constants.%Y_where.type.8eb
// CHECK:STDOUT:   %H.loc54_8.1 => constants.%facet_value
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.ce4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- enclosing_specific_is_facet_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Z1.type: type = facet_type <@Z1> [concrete]
// CHECK:STDOUT:   %Self.875: %Z1.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Z2.type: type = facet_type <@Z2> [concrete]
// CHECK:STDOUT:   %Self.14e: %Z2.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %OuterParam: %Z1.type = bind_symbolic_name OuterParam, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.79b: type = pattern_type %Z1.type [concrete]
// CHECK:STDOUT:   %Outer.type: type = generic_class_type @Outer [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Outer.generic: %Outer.type = struct_value () [concrete]
// CHECK:STDOUT:   %Outer.fea: type = class_type @Outer, @Outer(%OuterParam) [symbolic]
// CHECK:STDOUT:   %Y.type.5e4: type = facet_type <@Y, @Y(%OuterParam)> [symbolic]
// CHECK:STDOUT:   %Self.048: %Y.type.5e4 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Y.assoc_type.2a6: type = assoc_entity_type @Y, @Y(%OuterParam) [symbolic]
// CHECK:STDOUT:   %assoc0.603: %Y.assoc_type.2a6 = assoc_entity element0, @Y.%T [symbolic]
// CHECK:STDOUT:   %.Self.99e: %Y.type.5e4 = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %require_complete.d37: <witness> = require_complete_type %Y.type.5e4 [symbolic]
// CHECK:STDOUT:   %.Self.as_type.3b9: type = facet_access_type %.Self.99e [symbolic]
// CHECK:STDOUT:   %Y.lookup_impl_witness.c65: <witness> = lookup_impl_witness %.Self.99e, @Y, @Y(%OuterParam) [symbolic]
// CHECK:STDOUT:   %Y.facet: %Y.type.5e4 = facet_value %.Self.as_type.3b9, (%Y.lookup_impl_witness.c65) [symbolic]
// CHECK:STDOUT:   %impl.elem0.23d: type = impl_witness_access %Y.lookup_impl_witness.c65, element0 [symbolic]
// CHECK:STDOUT:   %Y_where.type.fee: type = facet_type <@Y, @Y(%OuterParam) where %impl.elem0.23d = %empty_tuple.type> [symbolic]
// CHECK:STDOUT:   %H: %Y_where.type.fee = bind_symbolic_name H, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.b86: type = pattern_type %Y_where.type.fee [symbolic]
// CHECK:STDOUT:   %G.type.854: type = fn_type @G, @Outer(%OuterParam) [symbolic]
// CHECK:STDOUT:   %G.e2b: %G.type.854 = struct_value () [symbolic]
// CHECK:STDOUT:   %C.4b0: type = class_type @C, @C(%OuterParam) [symbolic]
// CHECK:STDOUT:   %require_complete.1fd: <witness> = require_complete_type %Y_where.type.fee [symbolic]
// CHECK:STDOUT:   %Y.impl_witness.f45: <witness> = impl_witness @Outer.%Y.impl_witness_table, @impl.cc8(%OuterParam) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Z1.impl_witness: <witness> = impl_witness file.%Z1.impl_witness_table [concrete]
// CHECK:STDOUT:   %Z2.impl_witness: <witness> = impl_witness file.%Z2.impl_witness_table [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %BitAnd.type: type = facet_type <@BitAnd> [concrete]
// CHECK:STDOUT:   %Self.25f: %BitAnd.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %BitAnd.assoc_type: type = assoc_entity_type @BitAnd [concrete]
// CHECK:STDOUT:   %assoc0.d45: %BitAnd.assoc_type = assoc_entity element0, imports.%Core.import_ref.a93 [concrete]
// CHECK:STDOUT:   %Op.type.27a: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %Op.ab9: %Op.type.27a = struct_value () [concrete]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.25f [symbolic]
// CHECK:STDOUT:   %pattern_type.67d: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %BitAnd.impl_witness.b7b: <witness> = impl_witness imports.%BitAnd.impl_witness_table, @impl.f92(%T) [symbolic]
// CHECK:STDOUT:   %Op.type.f99: type = fn_type @Op.2, @impl.f92(%T) [symbolic]
// CHECK:STDOUT:   %Op.05a: %Op.type.f99 = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %BitAnd.impl_witness.0e5: <witness> = impl_witness imports.%BitAnd.impl_witness_table, @impl.f92(type) [concrete]
// CHECK:STDOUT:   %Op.type.eb8: type = fn_type @Op.2, @impl.f92(type) [concrete]
// CHECK:STDOUT:   %Op.444: %Op.type.eb8 = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type.473: <witness> = complete_type_witness type [concrete]
// CHECK:STDOUT:   %BitAnd.facet: %BitAnd.type = facet_value type, (%BitAnd.impl_witness.0e5) [concrete]
// CHECK:STDOUT:   %.518: type = fn_type_with_self_type %Op.type.27a, %BitAnd.facet [concrete]
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %Z1.type, %Op.444 [concrete]
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function %Op.444, @Op.2(type) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %Z1.type, %Op.specific_fn [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@Z1 & @Z2> [concrete]
// CHECK:STDOUT:   %facet_value.74b: %facet_type = facet_value %empty_tuple.type, (%Z1.impl_witness, %Z2.impl_witness) [concrete]
// CHECK:STDOUT:   %Z1.facet: %Z1.type = facet_value %empty_tuple.type, (%Z1.impl_witness) [concrete]
// CHECK:STDOUT:   %Outer.aff: type = class_type @Outer, @Outer(%Z1.facet) [concrete]
// CHECK:STDOUT:   %Y.type.927: type = facet_type <@Y, @Y(%Z1.facet)> [concrete]
// CHECK:STDOUT:   %G.type.663: type = fn_type @G, @Outer(%Z1.facet) [concrete]
// CHECK:STDOUT:   %G.ba4: %G.type.663 = struct_value () [concrete]
// CHECK:STDOUT:   %C.e73: type = class_type @C, @C(%Z1.facet) [concrete]
// CHECK:STDOUT:   %Y.lookup_impl_witness.20c: <witness> = lookup_impl_witness %.Self.99e, @Y, @Y(%Z1.facet) [symbolic]
// CHECK:STDOUT:   %impl.elem0.aad: type = impl_witness_access %Y.lookup_impl_witness.20c, element0 [symbolic]
// CHECK:STDOUT:   %Y_where.type.d44: type = facet_type <@Y, @Y(%Z1.facet) where %impl.elem0.aad = %empty_tuple.type> [symbolic]
// CHECK:STDOUT:   %.Self.3de: %Y.type.927 = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %Self.bdf: %Y.type.927 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Y.assoc_type.93b: type = assoc_entity_type @Y, @Y(%Z1.facet) [concrete]
// CHECK:STDOUT:   %assoc0.aa8: %Y.assoc_type.93b = assoc_entity element0, @Y.%T [concrete]
// CHECK:STDOUT:   %complete_type.066: <witness> = complete_type_witness %Y.type.927 [concrete]
// CHECK:STDOUT:   %.Self.as_type.7ce: type = facet_access_type %.Self.3de [symbolic_self]
// CHECK:STDOUT:   %Y.lookup_impl_witness.a49: <witness> = lookup_impl_witness %.Self.3de, @Y, @Y(%Z1.facet) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.8ce: type = impl_witness_access %Y.lookup_impl_witness.a49, element0 [symbolic_self]
// CHECK:STDOUT:   %Y_where.type.6af: type = facet_type <@Y, @Y(%Z1.facet) where %impl.elem0.8ce = %empty_tuple.type> [concrete]
// CHECK:STDOUT:   %complete_type.997: <witness> = complete_type_witness %Y_where.type.6af [concrete]
// CHECK:STDOUT:   %Y.impl_witness.997: <witness> = impl_witness @Outer.%Y.impl_witness_table, @impl.cc8(%Z1.facet) [concrete]
// CHECK:STDOUT:   %facet_value.ac2: %Y_where.type.d44 = facet_value %C.e73, (%Y.impl_witness.997) [symbolic]
// CHECK:STDOUT:   %pattern_type.b1a: type = pattern_type %Y_where.type.6af [concrete]
// CHECK:STDOUT:   %G.specific_fn: <specific function> = specific_function %G.ba4, @G(%Z1.facet, %facet_value.ac2) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .BitAnd = %Core.BitAnd
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.BitAnd: type = import_ref Core//prelude, BitAnd, loaded [concrete = constants.%BitAnd.type]
// CHECK:STDOUT:   %Core.import_ref.ad0 = import_ref Core//prelude, inst107 [no loc], unloaded
// CHECK:STDOUT:   %Core.import_ref.a46: %BitAnd.assoc_type = import_ref Core//prelude, loc18_41, loaded [concrete = constants.%assoc0.d45]
// CHECK:STDOUT:   %Core.Op = import_ref Core//prelude, Op, unloaded
// CHECK:STDOUT:   %Core.import_ref.a93: %Op.type.27a = import_ref Core//prelude, loc18_41, loaded [concrete = constants.%Op.ab9]
// CHECK:STDOUT:   %Core.import_ref.040: %BitAnd.type = import_ref Core//prelude, inst107 [no loc], loaded [symbolic = constants.%Self.25f]
// CHECK:STDOUT:   %Core.import_ref.140: <witness> = import_ref Core//prelude, loc21_36, loaded [symbolic = @impl.f92.%BitAnd.impl_witness (constants.%BitAnd.impl_witness.b7b)]
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.1: type = import_ref Core//prelude, loc21_14, loaded [symbolic = @impl.f92.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.583: type = import_ref Core//prelude, loc21_24, loaded [symbolic = @impl.f92.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.9c1: type = import_ref Core//prelude, loc21_29, loaded [concrete = constants.%BitAnd.type]
// CHECK:STDOUT:   %Core.import_ref.1e6: @impl.f92.%Op.type (%Op.type.f99) = import_ref Core//prelude, loc22_42, loaded [symbolic = @impl.f92.%Op (constants.%Op.05a)]
// CHECK:STDOUT:   %BitAnd.impl_witness_table = impl_witness_table (%Core.import_ref.1e6), @impl.f92 [concrete]
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.2: type = import_ref Core//prelude, loc21_14, loaded [symbolic = @impl.f92.%T (constants.%T)]
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
// CHECK:STDOUT:     %OuterParam.patt: %pattern_type.79b = symbolic_binding_pattern OuterParam, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Z1.ref: type = name_ref Z1, file.%Z1.decl [concrete = constants.%Z1.type]
// CHECK:STDOUT:     %OuterParam.loc6_13.1: %Z1.type = bind_symbolic_name OuterParam, 0 [symbolic = %OuterParam.loc6_13.2 (constants.%OuterParam)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.ff8 [concrete] {} {
// CHECK:STDOUT:     %.loc17_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc17_7.2: type = converted %.loc17_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %Z1.ref: type = name_ref Z1, file.%Z1.decl [concrete = constants.%Z1.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Z1.impl_witness_table = impl_witness_table (), @impl.ff8 [concrete]
// CHECK:STDOUT:   %Z1.impl_witness: <witness> = impl_witness %Z1.impl_witness_table [concrete = constants.%Z1.impl_witness]
// CHECK:STDOUT:   impl_decl @impl.272 [concrete] {} {
// CHECK:STDOUT:     %.loc18_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc18_7.2: type = converted %.loc18_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %Z2.ref: type = name_ref Z2, file.%Z2.decl [concrete = constants.%Z2.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Z2.impl_witness_table = impl_witness_table (), @impl.272 [concrete]
// CHECK:STDOUT:   %Z2.impl_witness: <witness> = impl_witness %Z2.impl_witness_table [concrete = constants.%Z2.impl_witness]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Z1 {
// CHECK:STDOUT:   %Self: %Z1.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.875]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Z2 {
// CHECK:STDOUT:   %Self: %Z2.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.14e]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Y(@Outer.%OuterParam.loc6_13.1: %Z1.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %OuterParam: %Z1.type = bind_symbolic_name OuterParam, 0 [symbolic = %OuterParam (constants.%OuterParam)]
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y, @Y(%OuterParam)> [symbolic = %Y.type (constants.%Y.type.5e4)]
// CHECK:STDOUT:   %Self.2: @Y.%Y.type (%Y.type.5e4) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.048)]
// CHECK:STDOUT:   %Y.assoc_type: type = assoc_entity_type @Y, @Y(%OuterParam) [symbolic = %Y.assoc_type (constants.%Y.assoc_type.2a6)]
// CHECK:STDOUT:   %assoc0: @Y.%Y.assoc_type (%Y.assoc_type.2a6) = assoc_entity element0, %T [symbolic = %assoc0 (constants.%assoc0.603)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Y.%Y.type (%Y.type.5e4) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.048)]
// CHECK:STDOUT:     %T: type = assoc_const_decl @T [concrete] {
// CHECK:STDOUT:       %assoc0: @Y.%Y.assoc_type (%Y.assoc_type.2a6) = assoc_entity element0, @Y.%T [symbolic = @Y.%assoc0 (constants.%assoc0.603)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .T = @T.%assoc0
// CHECK:STDOUT:     witness = (%T)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @BitAnd [from "include_files/facet_types.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Core.import_ref.ad0
// CHECK:STDOUT:   .Op = imports.%Core.import_ref.a46
// CHECK:STDOUT:   witness = (imports.%Core.Op)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @T(@Outer.%OuterParam.loc6_13.1: %Z1.type, @Y.%Self.1: @Y.%Y.type (%Y.type.5e4)) {
// CHECK:STDOUT:   assoc_const T:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.cc8(@Outer.%OuterParam.loc6_13.1: %Z1.type) {
// CHECK:STDOUT:   %OuterParam: %Z1.type = bind_symbolic_name OuterParam, 0 [symbolic = %OuterParam (constants.%OuterParam)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%OuterParam) [symbolic = %C (constants.%C.4b0)]
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y, @Y(%OuterParam)> [symbolic = %Y.type (constants.%Y.type.5e4)]
// CHECK:STDOUT:   %.Self.2: @impl.cc8.%Y.type (%Y.type.5e4) = bind_symbolic_name .Self [symbolic = %.Self.2 (constants.%.Self.99e)]
// CHECK:STDOUT:   %require_complete.loc14_21: <witness> = require_complete_type %Y.type [symbolic = %require_complete.loc14_21 (constants.%require_complete.d37)]
// CHECK:STDOUT:   %Y.assoc_type: type = assoc_entity_type @Y, @Y(%OuterParam) [symbolic = %Y.assoc_type (constants.%Y.assoc_type.2a6)]
// CHECK:STDOUT:   %assoc0: @impl.cc8.%Y.assoc_type (%Y.assoc_type.2a6) = assoc_entity element0, @Y.%T [symbolic = %assoc0 (constants.%assoc0.603)]
// CHECK:STDOUT:   %.Self.as_type.loc14_21.2: type = facet_access_type %.Self.2 [symbolic = %.Self.as_type.loc14_21.2 (constants.%.Self.as_type.3b9)]
// CHECK:STDOUT:   %Y.lookup_impl_witness: <witness> = lookup_impl_witness %.Self.2, @Y, @Y(%OuterParam) [symbolic = %Y.lookup_impl_witness (constants.%Y.lookup_impl_witness.c65)]
// CHECK:STDOUT:   %impl.elem0.loc14_21.2: type = impl_witness_access %Y.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc14_21.2 (constants.%impl.elem0.23d)]
// CHECK:STDOUT:   %Y_where.type: type = facet_type <@Y, @Y(%OuterParam) where %impl.elem0.loc14_21.2 = constants.%empty_tuple.type> [symbolic = %Y_where.type (constants.%Y_where.type.fee)]
// CHECK:STDOUT:   %require_complete.loc14_15: <witness> = require_complete_type %Y_where.type [symbolic = %require_complete.loc14_15 (constants.%require_complete.1fd)]
// CHECK:STDOUT:   %Y.impl_witness: <witness> = impl_witness @Outer.%Y.impl_witness_table, @impl.cc8(%OuterParam) [symbolic = %Y.impl_witness (constants.%Y.impl_witness.f45)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref as %.loc14_15 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = @Outer.%Y.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.ff8: %.loc17_7.2 as %Z1.ref {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%Z1.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.272: %.loc18_7.2 as %Z2.ref {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%Z2.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.f92(imports.%Core.import_ref.5ab3ec.1: type) [from "include_files/facet_types.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %BitAnd.impl_witness: <witness> = impl_witness imports.%BitAnd.impl_witness_table, @impl.f92(%T) [symbolic = %BitAnd.impl_witness (constants.%BitAnd.impl_witness.b7b)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type: type = fn_type @Op.2, @impl.f92(%T) [symbolic = %Op.type (constants.%Op.type.f99)]
// CHECK:STDOUT:   %Op: @impl.f92.%Op.type (%Op.type.f99) = struct_value () [symbolic = %Op (constants.%Op.05a)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Core.import_ref.583 as imports.%Core.import_ref.9c1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%Core.import_ref.140
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Outer(%OuterParam.loc6_13.1: %Z1.type) {
// CHECK:STDOUT:   %OuterParam.loc6_13.2: %Z1.type = bind_symbolic_name OuterParam, 0 [symbolic = %OuterParam.loc6_13.2 (constants.%OuterParam)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y, @Y(%OuterParam.loc6_13.2)> [symbolic = %Y.type (constants.%Y.type.5e4)]
// CHECK:STDOUT:   %G.type: type = fn_type @G, @Outer(%OuterParam.loc6_13.2) [symbolic = %G.type (constants.%G.type.854)]
// CHECK:STDOUT:   %G: @Outer.%G.type (%G.type.854) = struct_value () [symbolic = %G (constants.%G.e2b)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%OuterParam.loc6_13.2) [symbolic = %C (constants.%C.4b0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Y.decl: type = interface_decl @Y [symbolic = @Outer.%Y.type (constants.%Y.type.5e4)] {} {}
// CHECK:STDOUT:     %G.decl: @Outer.%G.type (%G.type.854) = fn_decl @G [symbolic = @Outer.%G (constants.%G.e2b)] {
// CHECK:STDOUT:       %H.patt: @G.%pattern_type (%pattern_type.b86) = symbolic_binding_pattern H, 1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc11_14.1: type = splice_block %.loc11_14.2 [symbolic = %Y_where.type (constants.%Y_where.type.fee)] {
// CHECK:STDOUT:         %.loc11_12: type = specific_constant @Outer.%Y.decl, @Outer(constants.%OuterParam) [symbolic = %Y.type (constants.%Y.type.5e4)]
// CHECK:STDOUT:         %Y.ref: type = name_ref Y, %.loc11_12 [symbolic = %Y.type (constants.%Y.type.5e4)]
// CHECK:STDOUT:         %.Self.2: @G.%Y.type (%Y.type.5e4) = bind_symbolic_name .Self [symbolic = %.Self.1 (constants.%.Self.99e)]
// CHECK:STDOUT:         %.Self.ref: @G.%Y.type (%Y.type.5e4) = name_ref .Self, %.Self.2 [symbolic = %.Self.1 (constants.%.Self.99e)]
// CHECK:STDOUT:         %.loc11_20.1: @G.%Y.assoc_type (%Y.assoc_type.2a6) = specific_constant @T.%assoc0, @Y(constants.%OuterParam) [symbolic = %assoc0 (constants.%assoc0.603)]
// CHECK:STDOUT:         %T.ref: @G.%Y.assoc_type (%Y.assoc_type.2a6) = name_ref T, %.loc11_20.1 [symbolic = %assoc0 (constants.%assoc0.603)]
// CHECK:STDOUT:         %.Self.as_type.loc11_20.2: type = facet_access_type %.Self.ref [symbolic = %.Self.as_type.loc11_20.1 (constants.%.Self.as_type.3b9)]
// CHECK:STDOUT:         %.loc11_20.2: type = converted %.Self.ref, %.Self.as_type.loc11_20.2 [symbolic = %.Self.as_type.loc11_20.1 (constants.%.Self.as_type.3b9)]
// CHECK:STDOUT:         %impl.elem0.loc11_20.2: type = impl_witness_access constants.%Y.lookup_impl_witness.c65, element0 [symbolic = %impl.elem0.loc11_20.1 (constants.%impl.elem0.23d)]
// CHECK:STDOUT:         %.loc11_26.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:         %.loc11_26.2: type = converted %.loc11_26.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:         %.loc11_14.2: type = where_expr %.Self.2 [symbolic = %Y_where.type (constants.%Y_where.type.fee)] {
// CHECK:STDOUT:           requirement_rewrite %impl.elem0.loc11_20.2, %.loc11_26.2
// CHECK:STDOUT:         }
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %H.loc11_8.2: @G.%Y_where.type (%Y_where.type.fee) = bind_symbolic_name H, 1 [symbolic = %H.loc11_8.1 (constants.%H)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %C.decl: type = class_decl @C [symbolic = @Outer.%C (constants.%C.4b0)] {} {}
// CHECK:STDOUT:     impl_decl @impl.cc8 [concrete] {} {
// CHECK:STDOUT:       %.loc14_8: type = specific_constant @Outer.%C.decl, @Outer(constants.%OuterParam) [symbolic = %C (constants.%C.4b0)]
// CHECK:STDOUT:       %C.ref: type = name_ref C, %.loc14_8 [symbolic = %C (constants.%C.4b0)]
// CHECK:STDOUT:       %.loc14_13: type = specific_constant @Outer.%Y.decl, @Outer(constants.%OuterParam) [symbolic = %Y.type (constants.%Y.type.5e4)]
// CHECK:STDOUT:       %Y.ref: type = name_ref Y, %.loc14_13 [symbolic = %Y.type (constants.%Y.type.5e4)]
// CHECK:STDOUT:       %.Self.1: @impl.cc8.%Y.type (%Y.type.5e4) = bind_symbolic_name .Self [symbolic = %.Self.2 (constants.%.Self.99e)]
// CHECK:STDOUT:       %.Self.ref: @impl.cc8.%Y.type (%Y.type.5e4) = name_ref .Self, %.Self.1 [symbolic = %.Self.2 (constants.%.Self.99e)]
// CHECK:STDOUT:       %.loc14_21.1: @impl.cc8.%Y.assoc_type (%Y.assoc_type.2a6) = specific_constant @T.%assoc0, @Y(constants.%OuterParam) [symbolic = %assoc0 (constants.%assoc0.603)]
// CHECK:STDOUT:       %T.ref: @impl.cc8.%Y.assoc_type (%Y.assoc_type.2a6) = name_ref T, %.loc14_21.1 [symbolic = %assoc0 (constants.%assoc0.603)]
// CHECK:STDOUT:       %.Self.as_type.loc14_21.1: type = facet_access_type %.Self.ref [symbolic = %.Self.as_type.loc14_21.2 (constants.%.Self.as_type.3b9)]
// CHECK:STDOUT:       %.loc14_21.2: type = converted %.Self.ref, %.Self.as_type.loc14_21.1 [symbolic = %.Self.as_type.loc14_21.2 (constants.%.Self.as_type.3b9)]
// CHECK:STDOUT:       %impl.elem0.loc14_21.1: type = impl_witness_access constants.%Y.lookup_impl_witness.c65, element0 [symbolic = %impl.elem0.loc14_21.2 (constants.%impl.elem0.23d)]
// CHECK:STDOUT:       %.loc14_27.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc14_27.2: type = converted %.loc14_27.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %.loc14_15: type = where_expr %.Self.1 [symbolic = %Y_where.type (constants.%Y_where.type.fee)] {
// CHECK:STDOUT:         requirement_rewrite %impl.elem0.loc14_21.1, %.loc14_27.2
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Y.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @impl.cc8 [concrete]
// CHECK:STDOUT:     %Y.impl_witness: <witness> = impl_witness %Y.impl_witness_table, @impl.cc8(constants.%OuterParam) [symbolic = @impl.cc8.%Y.impl_witness (constants.%Y.impl_witness.f45)]
// CHECK:STDOUT:     %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Outer.fea
// CHECK:STDOUT:     .Y = %Y.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(@Outer.%OuterParam.loc6_13.1: %Z1.type) {
// CHECK:STDOUT:   class;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(@Outer.%OuterParam.loc6_13.1: %Z1.type, %H.loc11_8.2: @G.%Y_where.type (%Y_where.type.fee)) {
// CHECK:STDOUT:   %OuterParam: %Z1.type = bind_symbolic_name OuterParam, 0 [symbolic = %OuterParam (constants.%OuterParam)]
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y, @Y(%OuterParam)> [symbolic = %Y.type (constants.%Y.type.5e4)]
// CHECK:STDOUT:   %.Self.1: @G.%Y.type (%Y.type.5e4) = bind_symbolic_name .Self [symbolic = %.Self.1 (constants.%.Self.99e)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Y.type [symbolic = %require_complete (constants.%require_complete.d37)]
// CHECK:STDOUT:   %Y.assoc_type: type = assoc_entity_type @Y, @Y(%OuterParam) [symbolic = %Y.assoc_type (constants.%Y.assoc_type.2a6)]
// CHECK:STDOUT:   %assoc0: @G.%Y.assoc_type (%Y.assoc_type.2a6) = assoc_entity element0, @Y.%T [symbolic = %assoc0 (constants.%assoc0.603)]
// CHECK:STDOUT:   %.Self.as_type.loc11_20.1: type = facet_access_type %.Self.1 [symbolic = %.Self.as_type.loc11_20.1 (constants.%.Self.as_type.3b9)]
// CHECK:STDOUT:   %Y.lookup_impl_witness: <witness> = lookup_impl_witness %.Self.1, @Y, @Y(%OuterParam) [symbolic = %Y.lookup_impl_witness (constants.%Y.lookup_impl_witness.c65)]
// CHECK:STDOUT:   %impl.elem0.loc11_20.1: type = impl_witness_access %Y.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc11_20.1 (constants.%impl.elem0.23d)]
// CHECK:STDOUT:   %Y_where.type: type = facet_type <@Y, @Y(%OuterParam) where %impl.elem0.loc11_20.1 = constants.%empty_tuple.type> [symbolic = %Y_where.type (constants.%Y_where.type.fee)]
// CHECK:STDOUT:   %H.loc11_8.1: @G.%Y_where.type (%Y_where.type.fee) = bind_symbolic_name H, 1 [symbolic = %H.loc11_8.1 (constants.%H)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Y_where.type [symbolic = %pattern_type (constants.%pattern_type.b86)]
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
// CHECK:STDOUT:   %impl.elem0.loc21_19: %.518 = impl_witness_access constants.%BitAnd.impl_witness.0e5, element0 [concrete = constants.%Op.444]
// CHECK:STDOUT:   %bound_method.loc21_19.1: <bound method> = bound_method %Z1.ref.loc21_16, %impl.elem0.loc21_19 [concrete = constants.%Op.bound]
// CHECK:STDOUT:   %specific_fn.loc21_19: <specific function> = specific_function %impl.elem0.loc21_19, @Op.2(type) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc21_19.2: <bound method> = bound_method %Z1.ref.loc21_16, %specific_fn.loc21_19 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %type.and.loc21_19: init type = call %bound_method.loc21_19.2(%Z1.ref.loc21_16, %Z2.ref.loc21_21) [concrete = constants.%facet_type]
// CHECK:STDOUT:   %.loc21_23.1: type = value_of_initializer %type.and.loc21_19 [concrete = constants.%facet_type]
// CHECK:STDOUT:   %.loc21_23.2: type = converted %type.and.loc21_19, %.loc21_23.1 [concrete = constants.%facet_type]
// CHECK:STDOUT:   %facet_value.loc21_12: %facet_type = facet_value constants.%empty_tuple.type, (constants.%Z1.impl_witness, constants.%Z2.impl_witness) [concrete = constants.%facet_value.74b]
// CHECK:STDOUT:   %.loc21_12: %facet_type = converted %.loc21_10, %facet_value.loc21_12 [concrete = constants.%facet_value.74b]
// CHECK:STDOUT:   %as_type.loc21_24: type = facet_access_type constants.%facet_value.74b [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %Z1.facet.loc21_24: %Z1.type = facet_value %as_type.loc21_24, (constants.%Z1.impl_witness) [concrete = constants.%Z1.facet]
// CHECK:STDOUT:   %.loc21_24: %Z1.type = converted %.loc21_12, %Z1.facet.loc21_24 [concrete = constants.%Z1.facet]
// CHECK:STDOUT:   %Outer.loc21_24: type = class_type @Outer, @Outer(constants.%Z1.facet) [concrete = constants.%Outer.aff]
// CHECK:STDOUT:   %.loc21_25: %G.type.663 = specific_constant @Outer.%G.decl, @Outer(constants.%Z1.facet) [concrete = constants.%G.ba4]
// CHECK:STDOUT:   %G.ref: %G.type.663 = name_ref G, %.loc21_25 [concrete = constants.%G.ba4]
// CHECK:STDOUT:   %Outer.ref.loc21_28: %Outer.type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:   %.loc21_35: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %Z1.ref.loc21_41: type = name_ref Z1, file.%Z1.decl [concrete = constants.%Z1.type]
// CHECK:STDOUT:   %Z2.ref.loc21_46: type = name_ref Z2, file.%Z2.decl [concrete = constants.%Z2.type]
// CHECK:STDOUT:   %impl.elem0.loc21_44: %.518 = impl_witness_access constants.%BitAnd.impl_witness.0e5, element0 [concrete = constants.%Op.444]
// CHECK:STDOUT:   %bound_method.loc21_44.1: <bound method> = bound_method %Z1.ref.loc21_41, %impl.elem0.loc21_44 [concrete = constants.%Op.bound]
// CHECK:STDOUT:   %specific_fn.loc21_44: <specific function> = specific_function %impl.elem0.loc21_44, @Op.2(type) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc21_44.2: <bound method> = bound_method %Z1.ref.loc21_41, %specific_fn.loc21_44 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %type.and.loc21_44: init type = call %bound_method.loc21_44.2(%Z1.ref.loc21_41, %Z2.ref.loc21_46) [concrete = constants.%facet_type]
// CHECK:STDOUT:   %.loc21_48.1: type = value_of_initializer %type.and.loc21_44 [concrete = constants.%facet_type]
// CHECK:STDOUT:   %.loc21_48.2: type = converted %type.and.loc21_44, %.loc21_48.1 [concrete = constants.%facet_type]
// CHECK:STDOUT:   %facet_value.loc21_37: %facet_type = facet_value constants.%empty_tuple.type, (constants.%Z1.impl_witness, constants.%Z2.impl_witness) [concrete = constants.%facet_value.74b]
// CHECK:STDOUT:   %.loc21_37: %facet_type = converted %.loc21_35, %facet_value.loc21_37 [concrete = constants.%facet_value.74b]
// CHECK:STDOUT:   %as_type.loc21_49: type = facet_access_type constants.%facet_value.74b [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %Z1.facet.loc21_49: %Z1.type = facet_value %as_type.loc21_49, (constants.%Z1.impl_witness) [concrete = constants.%Z1.facet]
// CHECK:STDOUT:   %.loc21_49: %Z1.type = converted %.loc21_37, %Z1.facet.loc21_49 [concrete = constants.%Z1.facet]
// CHECK:STDOUT:   %Outer.loc21_49: type = class_type @Outer, @Outer(constants.%Z1.facet) [concrete = constants.%Outer.aff]
// CHECK:STDOUT:   %.loc21_50: type = specific_constant @Outer.%C.decl, @Outer(constants.%Z1.facet) [concrete = constants.%C.e73]
// CHECK:STDOUT:   %C.ref: type = name_ref C, %.loc21_50 [concrete = constants.%C.e73]
// CHECK:STDOUT:   %facet_value.loc21_52: %Y_where.type.d44 = facet_value constants.%C.e73, (constants.%Y.impl_witness.997) [symbolic = constants.%facet_value.ac2]
// CHECK:STDOUT:   %.loc21_52: %Y_where.type.d44 = converted constants.%C.e73, %facet_value.loc21_52 [symbolic = constants.%facet_value.ac2]
// CHECK:STDOUT:   %G.specific_fn: <specific function> = specific_function %G.ref, @G(constants.%Z1.facet, constants.%facet_value.ac2) [symbolic = constants.%G.specific_fn]
// CHECK:STDOUT:   %G.call: init %empty_tuple.type = call %G.specific_fn()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.1(imports.%Core.import_ref.040: %BitAnd.type) [from "include_files/facet_types.carbon"] {
// CHECK:STDOUT:   %Self: %BitAnd.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.25f)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type [symbolic = %pattern_type (constants.%pattern_type.67d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.2(imports.%Core.import_ref.5ab3ec.2: type) [from "include_files/facet_types.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T [symbolic = %pattern_type (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn = "type.and";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%OuterParam) {
// CHECK:STDOUT:   %OuterParam.loc6_13.2 => constants.%OuterParam
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Y.type => constants.%Y.type.5e4
// CHECK:STDOUT:   %G.type => constants.%G.type.854
// CHECK:STDOUT:   %G => constants.%G.e2b
// CHECK:STDOUT:   %C => constants.%C.4b0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y(constants.%OuterParam) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %OuterParam => constants.%OuterParam
// CHECK:STDOUT:   %Y.type => constants.%Y.type.5e4
// CHECK:STDOUT:   %Self.2 => constants.%Self.048
// CHECK:STDOUT:   %Y.assoc_type => constants.%Y.assoc_type.2a6
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.603
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T(constants.%OuterParam, constants.%Self.048) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T(constants.%OuterParam, constants.%Y.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%OuterParam, constants.%H) {
// CHECK:STDOUT:   %OuterParam => constants.%OuterParam
// CHECK:STDOUT:   %Y.type => constants.%Y.type.5e4
// CHECK:STDOUT:   %.Self.1 => constants.%.Self.99e
// CHECK:STDOUT:   %require_complete => constants.%require_complete.d37
// CHECK:STDOUT:   %Y.assoc_type => constants.%Y.assoc_type.2a6
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.603
// CHECK:STDOUT:   %.Self.as_type.loc11_20.1 => constants.%.Self.as_type.3b9
// CHECK:STDOUT:   %Y.lookup_impl_witness => constants.%Y.lookup_impl_witness.c65
// CHECK:STDOUT:   %impl.elem0.loc11_20.1 => constants.%impl.elem0.23d
// CHECK:STDOUT:   %Y_where.type => constants.%Y_where.type.fee
// CHECK:STDOUT:   %H.loc11_8.1 => constants.%H
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.b86
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%OuterParam) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.cc8(constants.%OuterParam) {
// CHECK:STDOUT:   %OuterParam => constants.%OuterParam
// CHECK:STDOUT:   %C => constants.%C.4b0
// CHECK:STDOUT:   %Y.type => constants.%Y.type.5e4
// CHECK:STDOUT:   %.Self.2 => constants.%.Self.99e
// CHECK:STDOUT:   %require_complete.loc14_21 => constants.%require_complete.d37
// CHECK:STDOUT:   %Y.assoc_type => constants.%Y.assoc_type.2a6
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.603
// CHECK:STDOUT:   %.Self.as_type.loc14_21.2 => constants.%.Self.as_type.3b9
// CHECK:STDOUT:   %Y.lookup_impl_witness => constants.%Y.lookup_impl_witness.c65
// CHECK:STDOUT:   %impl.elem0.loc14_21.2 => constants.%impl.elem0.23d
// CHECK:STDOUT:   %Y_where.type => constants.%Y_where.type.fee
// CHECK:STDOUT:   %require_complete.loc14_15 => constants.%require_complete.1fd
// CHECK:STDOUT:   %Y.impl_witness => constants.%Y.impl_witness.f45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%Self.25f) {
// CHECK:STDOUT:   %Self => constants.%Self.25f
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.67d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.f92(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %BitAnd.impl_witness => constants.%BitAnd.impl_witness.b7b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.2(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.f92(type) {
// CHECK:STDOUT:   %T => type
// CHECK:STDOUT:   %BitAnd.impl_witness => constants.%BitAnd.impl_witness.0e5
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type => constants.%Op.type.eb8
// CHECK:STDOUT:   %Op => constants.%Op.444
// CHECK:STDOUT:   %require_complete => constants.%complete_type.473
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.2(type) {
// CHECK:STDOUT:   %T => type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.98f
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%Z1.facet) {
// CHECK:STDOUT:   %OuterParam.loc6_13.2 => constants.%Z1.facet
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Y.type => constants.%Y.type.927
// CHECK:STDOUT:   %G.type => constants.%G.type.663
// CHECK:STDOUT:   %G => constants.%G.ba4
// CHECK:STDOUT:   %C => constants.%C.e73
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y(constants.%Z1.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %OuterParam => constants.%Z1.facet
// CHECK:STDOUT:   %Y.type => constants.%Y.type.927
// CHECK:STDOUT:   %Self.2 => constants.%Self.bdf
// CHECK:STDOUT:   %Y.assoc_type => constants.%Y.assoc_type.93b
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.aa8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%Z1.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.cc8(constants.%Z1.facet) {
// CHECK:STDOUT:   %OuterParam => constants.%Z1.facet
// CHECK:STDOUT:   %C => constants.%C.e73
// CHECK:STDOUT:   %Y.type => constants.%Y.type.927
// CHECK:STDOUT:   %.Self.2 => constants.%.Self.3de
// CHECK:STDOUT:   %require_complete.loc14_21 => constants.%complete_type.066
// CHECK:STDOUT:   %Y.assoc_type => constants.%Y.assoc_type.93b
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.aa8
// CHECK:STDOUT:   %.Self.as_type.loc14_21.2 => constants.%.Self.as_type.7ce
// CHECK:STDOUT:   %Y.lookup_impl_witness => constants.%Y.lookup_impl_witness.a49
// CHECK:STDOUT:   %impl.elem0.loc14_21.2 => constants.%impl.elem0.8ce
// CHECK:STDOUT:   %Y_where.type => constants.%Y_where.type.6af
// CHECK:STDOUT:   %require_complete.loc14_15 => constants.%complete_type.997
// CHECK:STDOUT:   %Y.impl_witness => constants.%Y.impl_witness.997
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%Z1.facet, constants.%facet_value.ac2) {
// CHECK:STDOUT:   %OuterParam => constants.%Z1.facet
// CHECK:STDOUT:   %Y.type => constants.%Y.type.927
// CHECK:STDOUT:   %.Self.1 => constants.%.Self.3de
// CHECK:STDOUT:   %require_complete => constants.%complete_type.066
// CHECK:STDOUT:   %Y.assoc_type => constants.%Y.assoc_type.93b
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.aa8
// CHECK:STDOUT:   %.Self.as_type.loc11_20.1 => constants.%.Self.as_type.7ce
// CHECK:STDOUT:   %Y.lookup_impl_witness => constants.%Y.lookup_impl_witness.a49
// CHECK:STDOUT:   %impl.elem0.loc11_20.1 => constants.%impl.elem0.8ce
// CHECK:STDOUT:   %Y_where.type => constants.%Y_where.type.6af
// CHECK:STDOUT:   %H.loc11_8.1 => constants.%facet_value.ac2
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.b1a
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- include_files/facet_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %As.type.b51: type = generic_interface_type @As [concrete]
// CHECK:STDOUT:   %As.generic: %As.type.b51 = struct_value () [concrete]
// CHECK:STDOUT:   %As.type.8ba: type = facet_type <@As, @As(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.b4e: %As.type.8ba = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type.7f0: type = facet_access_type %Self.b4e [symbolic]
// CHECK:STDOUT:   %pattern_type.947: type = pattern_type %Self.as_type.7f0 [symbolic]
// CHECK:STDOUT:   %pattern_type.7dcd0a.1: type = pattern_type %Dest [symbolic]
// CHECK:STDOUT:   %Convert.type.ad1: type = fn_type @Convert.1, @As(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.0ed: %Convert.type.ad1 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.assoc_type: type = assoc_entity_type @As, @As(%Dest) [symbolic]
// CHECK:STDOUT:   %assoc0.1d5: %As.assoc_type = assoc_entity element0, @As.%Convert.decl [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.96f: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.96f = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.07f: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.0f3: %ImplicitAs.type.07f = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type.419: type = facet_access_type %Self.0f3 [symbolic]
// CHECK:STDOUT:   %pattern_type.a93: type = pattern_type %Self.as_type.419 [symbolic]
// CHECK:STDOUT:   %Convert.type.4cf: type = fn_type @Convert.2, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.147: %Convert.type.4cf = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %assoc0.8f8: %ImplicitAs.assoc_type = assoc_entity element0, @ImplicitAs.%Convert.decl [symbolic]
// CHECK:STDOUT:   %BitAnd.type: type = facet_type <@BitAnd> [concrete]
// CHECK:STDOUT:   %Self.e44: %BitAnd.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type.560: type = facet_access_type %Self.e44 [symbolic]
// CHECK:STDOUT:   %pattern_type.a80: type = pattern_type %Self.as_type.560 [symbolic]
// CHECK:STDOUT:   %Op.type.613: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %Op.d98: %Op.type.613 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAnd.assoc_type: type = assoc_entity_type @BitAnd [concrete]
// CHECK:STDOUT:   %assoc0.220: %BitAnd.assoc_type = assoc_entity element0, @BitAnd.%Op.decl [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %BitAnd.impl_witness: <witness> = impl_witness file.%BitAnd.impl_witness_table, @impl(%T) [symbolic]
// CHECK:STDOUT:   %pattern_type.7dcd0a.2: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %Op.type.28d: type = fn_type @Op.2, @impl(%T) [symbolic]
// CHECK:STDOUT:   %Op.902: %Op.type.28d = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %BitAnd.facet: %BitAnd.type = facet_value %T, (%BitAnd.impl_witness) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .As = %As.decl
// CHECK:STDOUT:     .ImplicitAs = %ImplicitAs.decl
// CHECK:STDOUT:     .BitAnd = %BitAnd.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %As.decl: %As.type.b51 = interface_decl @As [concrete = constants.%As.generic] {
// CHECK:STDOUT:     %Dest.patt: %pattern_type.98f = symbolic_binding_pattern Dest, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Dest.loc9_14.1: type = bind_symbolic_name Dest, 0 [symbolic = %Dest.loc9_14.2 (constants.%Dest)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.decl: %ImplicitAs.type.96f = interface_decl @ImplicitAs [concrete = constants.%ImplicitAs.generic] {
// CHECK:STDOUT:     %Dest.patt: %pattern_type.98f = symbolic_binding_pattern Dest, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Dest.loc13_22.1: type = bind_symbolic_name Dest, 0 [symbolic = %Dest.loc13_22.2 (constants.%Dest)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %BitAnd.decl: type = interface_decl @BitAnd [concrete = constants.%BitAnd.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc21_14.1 [symbolic = %T.loc21_14.2 (constants.%T)]
// CHECK:STDOUT:     %BitAnd.ref: type = name_ref BitAnd, file.%BitAnd.decl [concrete = constants.%BitAnd.type]
// CHECK:STDOUT:     %T.loc21_14.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc21_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %BitAnd.impl_witness_table = impl_witness_table (@impl.%Op.decl), @impl [concrete]
// CHECK:STDOUT:   %BitAnd.impl_witness: <witness> = impl_witness %BitAnd.impl_witness_table, @impl(constants.%T) [symbolic = @impl.%BitAnd.impl_witness (constants.%BitAnd.impl_witness)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @As(%Dest.loc9_14.1: type) {
// CHECK:STDOUT:   %Dest.loc9_14.2: type = bind_symbolic_name Dest, 0 [symbolic = %Dest.loc9_14.2 (constants.%Dest)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%Dest.loc9_14.2)> [symbolic = %As.type (constants.%As.type.8ba)]
// CHECK:STDOUT:   %Self.2: @As.%As.type (%As.type.8ba) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.b4e)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.1, @As(%Dest.loc9_14.2) [symbolic = %Convert.type (constants.%Convert.type.ad1)]
// CHECK:STDOUT:   %Convert: @As.%Convert.type (%Convert.type.ad1) = struct_value () [symbolic = %Convert (constants.%Convert.0ed)]
// CHECK:STDOUT:   %As.assoc_type: type = assoc_entity_type @As, @As(%Dest.loc9_14.2) [symbolic = %As.assoc_type (constants.%As.assoc_type)]
// CHECK:STDOUT:   %assoc0.loc10_35.2: @As.%As.assoc_type (%As.assoc_type) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc10_35.2 (constants.%assoc0.1d5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @As.%As.type (%As.type.8ba) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.b4e)]
// CHECK:STDOUT:     %Convert.decl: @As.%Convert.type (%Convert.type.ad1) = fn_decl @Convert.1 [symbolic = @As.%Convert (constants.%Convert.0ed)] {
// CHECK:STDOUT:       %self.patt: @Convert.1.%pattern_type.loc10_14 (%pattern_type.947) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @Convert.1.%pattern_type.loc10_14 (%pattern_type.947) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @Convert.1.%pattern_type.loc10_28 (%pattern_type.7dcd0a.1) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @Convert.1.%pattern_type.loc10_28 (%pattern_type.7dcd0a.1) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Dest.ref: type = name_ref Dest, @As.%Dest.loc9_14.1 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:       %self.param: @Convert.1.%Self.as_type.loc10_20.1 (%Self.as_type.7f0) = value_param call_param0
// CHECK:STDOUT:       %.loc10_20.1: type = splice_block %.loc10_20.3 [symbolic = %Self.as_type.loc10_20.1 (constants.%Self.as_type.7f0)] {
// CHECK:STDOUT:         %.loc10_20.2: @Convert.1.%As.type (%As.type.8ba) = specific_constant @As.%Self.1, @As(constants.%Dest) [symbolic = %Self (constants.%Self.b4e)]
// CHECK:STDOUT:         %Self.ref: @Convert.1.%As.type (%As.type.8ba) = name_ref Self, %.loc10_20.2 [symbolic = %Self (constants.%Self.b4e)]
// CHECK:STDOUT:         %Self.as_type.loc10_20.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc10_20.1 (constants.%Self.as_type.7f0)]
// CHECK:STDOUT:         %.loc10_20.3: type = converted %Self.ref, %Self.as_type.loc10_20.2 [symbolic = %Self.as_type.loc10_20.1 (constants.%Self.as_type.7f0)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Convert.1.%Self.as_type.loc10_20.1 (%Self.as_type.7f0) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @Convert.1.%Dest (%Dest) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Convert.1.%Dest (%Dest) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc10_35.1: @As.%As.assoc_type (%As.assoc_type) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc10_35.2 (constants.%assoc0.1d5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .Dest = <poisoned>
// CHECK:STDOUT:     .Convert = %assoc0.loc10_35.1
// CHECK:STDOUT:     witness = (%Convert.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(%Dest.loc13_22.1: type) {
// CHECK:STDOUT:   %Dest.loc13_22.2: type = bind_symbolic_name Dest, 0 [symbolic = %Dest.loc13_22.2 (constants.%Dest)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest.loc13_22.2)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.07f)]
// CHECK:STDOUT:   %Self.2: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.0f3)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.2, @ImplicitAs(%Dest.loc13_22.2) [symbolic = %Convert.type (constants.%Convert.type.4cf)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.4cf) = struct_value () [symbolic = %Convert (constants.%Convert.147)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%Dest.loc13_22.2) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type)]
// CHECK:STDOUT:   %assoc0.loc14_35.2: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc14_35.2 (constants.%assoc0.8f8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.0f3)]
// CHECK:STDOUT:     %Convert.decl: @ImplicitAs.%Convert.type (%Convert.type.4cf) = fn_decl @Convert.2 [symbolic = @ImplicitAs.%Convert (constants.%Convert.147)] {
// CHECK:STDOUT:       %self.patt: @Convert.2.%pattern_type.loc14_14 (%pattern_type.a93) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @Convert.2.%pattern_type.loc14_14 (%pattern_type.a93) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @Convert.2.%pattern_type.loc14_28 (%pattern_type.7dcd0a.1) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @Convert.2.%pattern_type.loc14_28 (%pattern_type.7dcd0a.1) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Dest.ref: type = name_ref Dest, @ImplicitAs.%Dest.loc13_22.1 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:       %self.param: @Convert.2.%Self.as_type.loc14_20.1 (%Self.as_type.419) = value_param call_param0
// CHECK:STDOUT:       %.loc14_20.1: type = splice_block %.loc14_20.3 [symbolic = %Self.as_type.loc14_20.1 (constants.%Self.as_type.419)] {
// CHECK:STDOUT:         %.loc14_20.2: @Convert.2.%ImplicitAs.type (%ImplicitAs.type.07f) = specific_constant @ImplicitAs.%Self.1, @ImplicitAs(constants.%Dest) [symbolic = %Self (constants.%Self.0f3)]
// CHECK:STDOUT:         %Self.ref: @Convert.2.%ImplicitAs.type (%ImplicitAs.type.07f) = name_ref Self, %.loc14_20.2 [symbolic = %Self (constants.%Self.0f3)]
// CHECK:STDOUT:         %Self.as_type.loc14_20.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc14_20.1 (constants.%Self.as_type.419)]
// CHECK:STDOUT:         %.loc14_20.3: type = converted %Self.ref, %Self.as_type.loc14_20.2 [symbolic = %Self.as_type.loc14_20.1 (constants.%Self.as_type.419)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Convert.2.%Self.as_type.loc14_20.1 (%Self.as_type.419) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @Convert.2.%Dest (%Dest) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Convert.2.%Dest (%Dest) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc14_35.1: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc14_35.2 (constants.%assoc0.8f8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .Dest = <poisoned>
// CHECK:STDOUT:     .Convert = %assoc0.loc14_35.1
// CHECK:STDOUT:     witness = (%Convert.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @BitAnd {
// CHECK:STDOUT:   %Self: %BitAnd.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.e44]
// CHECK:STDOUT:   %Op.decl: %Op.type.613 = fn_decl @Op.1 [concrete = constants.%Op.d98] {
// CHECK:STDOUT:     %self.patt: @Op.1.%pattern_type (%pattern_type.a80) = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @Op.1.%pattern_type (%pattern_type.a80) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %other.patt: @Op.1.%pattern_type (%pattern_type.a80) = binding_pattern other [concrete]
// CHECK:STDOUT:     %other.param_patt: @Op.1.%pattern_type (%pattern_type.a80) = value_param_pattern %other.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: @Op.1.%pattern_type (%pattern_type.a80) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @Op.1.%pattern_type (%pattern_type.a80) = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc18_37: %BitAnd.type = name_ref Self, @BitAnd.%Self [symbolic = %Self (constants.%Self.e44)]
// CHECK:STDOUT:     %Self.as_type.loc18_37: type = facet_access_type %Self.ref.loc18_37 [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.560)]
// CHECK:STDOUT:     %.loc18_37: type = converted %Self.ref.loc18_37, %Self.as_type.loc18_37 [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.560)]
// CHECK:STDOUT:     %self.param: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = value_param call_param0
// CHECK:STDOUT:     %.loc18_15.1: type = splice_block %.loc18_15.2 [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.560)] {
// CHECK:STDOUT:       %Self.ref.loc18_15: %BitAnd.type = name_ref Self, @BitAnd.%Self [symbolic = %Self (constants.%Self.e44)]
// CHECK:STDOUT:       %Self.as_type.loc18_15.2: type = facet_access_type %Self.ref.loc18_15 [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.560)]
// CHECK:STDOUT:       %.loc18_15.2: type = converted %Self.ref.loc18_15, %Self.as_type.loc18_15.2 [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.560)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = value_param call_param1
// CHECK:STDOUT:     %.loc18_28.1: type = splice_block %.loc18_28.2 [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.560)] {
// CHECK:STDOUT:       %Self.ref.loc18_28: %BitAnd.type = name_ref Self, @BitAnd.%Self [symbolic = %Self (constants.%Self.e44)]
// CHECK:STDOUT:       %Self.as_type.loc18_28: type = facet_access_type %Self.ref.loc18_28 [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.560)]
// CHECK:STDOUT:       %.loc18_28.2: type = converted %Self.ref.loc18_28, %Self.as_type.loc18_28 [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.560)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %other: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = bind_name other, %other.param
// CHECK:STDOUT:     %return.param: ref @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = out_param call_param2
// CHECK:STDOUT:     %return: ref @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %BitAnd.assoc_type = assoc_entity element0, %Op.decl [concrete = constants.%assoc0.220]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Op = %assoc0
// CHECK:STDOUT:   witness = (%Op.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl(%T.loc21_14.1: type) {
// CHECK:STDOUT:   %T.loc21_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc21_14.2 (constants.%T)]
// CHECK:STDOUT:   %BitAnd.impl_witness: <witness> = impl_witness file.%BitAnd.impl_witness_table, @impl(%T.loc21_14.2) [symbolic = %BitAnd.impl_witness (constants.%BitAnd.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type: type = fn_type @Op.2, @impl(%T.loc21_14.2) [symbolic = %Op.type (constants.%Op.type.28d)]
// CHECK:STDOUT:   %Op: @impl.%Op.type (%Op.type.28d) = struct_value () [symbolic = %Op (constants.%Op.902)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc21_14.2 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref as %BitAnd.ref {
// CHECK:STDOUT:     %Op.decl: @impl.%Op.type (%Op.type.28d) = fn_decl @Op.2 [symbolic = @impl.%Op (constants.%Op.902)] {
// CHECK:STDOUT:       %self.patt: @Op.2.%pattern_type (%pattern_type.7dcd0a.2) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @Op.2.%pattern_type (%pattern_type.7dcd0a.2) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %other.patt: @Op.2.%pattern_type (%pattern_type.7dcd0a.2) = binding_pattern other [concrete]
// CHECK:STDOUT:       %other.param_patt: @Op.2.%pattern_type (%pattern_type.7dcd0a.2) = value_param_pattern %other.patt, call_param1 [concrete]
// CHECK:STDOUT:       %return.patt: @Op.2.%pattern_type (%pattern_type.7dcd0a.2) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @Op.2.%pattern_type (%pattern_type.7dcd0a.2) = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Self.ref.loc22_37: type = name_ref Self, @impl.%T.ref [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self.param: @Op.2.%T (%T) = value_param call_param0
// CHECK:STDOUT:       %Self.ref.loc22_15: type = name_ref Self, @impl.%T.ref [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self: @Op.2.%T (%T) = bind_name self, %self.param
// CHECK:STDOUT:       %other.param: @Op.2.%T (%T) = value_param call_param1
// CHECK:STDOUT:       %Self.ref.loc22_28: type = name_ref Self, @impl.%T.ref [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %other: @Op.2.%T (%T) = bind_name other, %other.param
// CHECK:STDOUT:       %return.param: ref @Op.2.%T (%T) = out_param call_param2
// CHECK:STDOUT:       %return: ref @Op.2.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Op = %Op.decl
// CHECK:STDOUT:     witness = file.%BitAnd.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.1(@As.%Dest.loc9_14.1: type, @As.%Self.1: @As.%As.type (%As.type.8ba)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%Dest)> [symbolic = %As.type (constants.%As.type.8ba)]
// CHECK:STDOUT:   %Self: @Convert.1.%As.type (%As.type.8ba) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.b4e)]
// CHECK:STDOUT:   %Self.as_type.loc10_20.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc10_20.1 (constants.%Self.as_type.7f0)]
// CHECK:STDOUT:   %pattern_type.loc10_14: type = pattern_type %Self.as_type.loc10_20.1 [symbolic = %pattern_type.loc10_14 (constants.%pattern_type.947)]
// CHECK:STDOUT:   %pattern_type.loc10_28: type = pattern_type %Dest [symbolic = %pattern_type.loc10_28 (constants.%pattern_type.7dcd0a.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Convert.1.%Self.as_type.loc10_20.1 (%Self.as_type.7f0)) -> @Convert.1.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.2(@ImplicitAs.%Dest.loc13_22.1: type, @ImplicitAs.%Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.07f)]
// CHECK:STDOUT:   %Self: @Convert.2.%ImplicitAs.type (%ImplicitAs.type.07f) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.0f3)]
// CHECK:STDOUT:   %Self.as_type.loc14_20.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc14_20.1 (constants.%Self.as_type.419)]
// CHECK:STDOUT:   %pattern_type.loc14_14: type = pattern_type %Self.as_type.loc14_20.1 [symbolic = %pattern_type.loc14_14 (constants.%pattern_type.a93)]
// CHECK:STDOUT:   %pattern_type.loc14_28: type = pattern_type %Dest [symbolic = %pattern_type.loc14_28 (constants.%pattern_type.7dcd0a.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Convert.2.%Self.as_type.loc14_20.1 (%Self.as_type.419)) -> @Convert.2.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.1(@BitAnd.%Self: %BitAnd.type) {
// CHECK:STDOUT:   %Self: %BitAnd.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.e44)]
// CHECK:STDOUT:   %Self.as_type.loc18_15.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.560)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc18_15.1 [symbolic = %pattern_type (constants.%pattern_type.a80)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560), %other.param: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560)) -> @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.2(@impl.%T.loc21_14.1: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T [symbolic = %pattern_type (constants.%pattern_type.7dcd0a.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Op.2.%T (%T), %other.param: @Op.2.%T (%T)) -> @Op.2.%T (%T) = "type.and";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%Dest) {
// CHECK:STDOUT:   %Dest.loc9_14.2 => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%Dest, constants.%Self.b4e) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %As.type => constants.%As.type.8ba
// CHECK:STDOUT:   %Self => constants.%Self.b4e
// CHECK:STDOUT:   %Self.as_type.loc10_20.1 => constants.%Self.as_type.7f0
// CHECK:STDOUT:   %pattern_type.loc10_14 => constants.%pattern_type.947
// CHECK:STDOUT:   %pattern_type.loc10_28 => constants.%pattern_type.7dcd0a.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest.loc13_22.2 => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.2(constants.%Dest, constants.%Self.0f3) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.07f
// CHECK:STDOUT:   %Self => constants.%Self.0f3
// CHECK:STDOUT:   %Self.as_type.loc14_20.1 => constants.%Self.as_type.419
// CHECK:STDOUT:   %pattern_type.loc14_14 => constants.%pattern_type.a93
// CHECK:STDOUT:   %pattern_type.loc14_28 => constants.%pattern_type.7dcd0a.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%Self.e44) {
// CHECK:STDOUT:   %Self => constants.%Self.e44
// CHECK:STDOUT:   %Self.as_type.loc18_15.1 => constants.%Self.as_type.560
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a80
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%T) {
// CHECK:STDOUT:   %T.loc21_14.2 => constants.%T
// CHECK:STDOUT:   %BitAnd.impl_witness => constants.%BitAnd.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.2(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dcd0a.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%BitAnd.facet) {
// CHECK:STDOUT:   %Self => constants.%BitAnd.facet
// CHECK:STDOUT:   %Self.as_type.loc18_15.1 => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dcd0a.2
// CHECK:STDOUT: }
// CHECK:STDOUT:

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/incomplete.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/incomplete.carbon

// --- fail_empty_struct.carbon
library "[[@TEST_NAME]]";

interface I;

// Requires I identified.
impl {} as I;

// CHECK:STDERR: fail_empty_struct.carbon:[[@LINE+3]]:1-14: error: definition of impl as incomplete facet type `I` [ImplAsIncompleteFacetTypeDefinition]
// CHECK:STDERR: fail_empty_struct.carbon:[[@LINE-6]]:1-12: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR:
impl {} as I {}

// --- fail_class.carbon
library "[[@TEST_NAME]]";

interface J;
class C {}

// Requires I identified.
// CHECK:STDERR: fail_class.carbon:[[@LINE+2]]:11: error: name `I` not found [NameNotFound]
// CHECK:STDERR:
impl C as I;

// CHECK:STDERR: fail_class.carbon:[[@LINE+3]]:1-13: error: definition of impl as incomplete facet type `J` [ImplAsIncompleteFacetTypeDefinition]
// CHECK:STDERR: fail_class.carbon:[[@LINE-9]]:1-12: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR:
impl C as J {}

// --- fail_class_no_forward_decl.carbon
library "[[@TEST_NAME]]";

interface J;
class C {}

// CHECK:STDERR: fail_class_no_forward_decl.carbon:[[@LINE+3]]:1-13: error: definition of impl as incomplete facet type `J` [ImplAsIncompleteFacetTypeDefinition]
// CHECK:STDERR: fail_class_no_forward_decl.carbon:[[@LINE-4]]:1-12: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR:
impl C as J {}

// --- fail_class_with_rewrite.carbon
library "[[@TEST_NAME]]";

interface J;
class C {}

// CHECK:STDERR: fail_class_with_rewrite.carbon:[[@LINE+3]]:19-20: error: member access into facet of incomplete type `J` [IncompleteTypeInMemberAccessOfFacet]
// CHECK:STDERR: fail_class_with_rewrite.carbon:[[@LINE-4]]:1-12: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR:
impl C as J where .X = ();

// CHECK:STDERR: fail_class_with_rewrite.carbon:[[@LINE+3]]:19-20: error: member access into facet of incomplete type `J` [IncompleteTypeInMemberAccessOfFacet]
// CHECK:STDERR: fail_class_with_rewrite.carbon:[[@LINE-9]]:1-12: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR:
impl C as J where .X = () {}

// --- incomplete_where.carbon
library "[[@TEST_NAME]]";

class C {}

interface I {}
interface Incomplete;

impl C as I where .Self impls Incomplete;

interface Incomplete {}
impl C as Incomplete {}

impl C as I where .Self impls Incomplete {}

// --- incomplete_where_rewrite.carbon
library "[[@TEST_NAME]]";

class C {}

interface J { let T: type; }
interface Incomplete;

impl C as J where .Self impls Incomplete and .T = ();

interface Incomplete {}
impl C as Incomplete {}

impl C as J where .Self impls Incomplete and .T = () {}

// --- fail_incomplete_where_definition.carbon
library "[[@TEST_NAME]]";

class C {}

interface I {}
interface Incomplete;

// CHECK:STDERR: fail_incomplete_where_definition.carbon:[[@LINE+2]]:1-42: error: constraint `I where .Self impls Incomplete` being implemented requires that `C` implements `Incomplete` [IdentifiedRequireImplsNotImplemented]
// CHECK:STDERR:
impl C as I where .Self impls Incomplete {}

// --- fail_declaration_lookup_into_incomplete.carbon
library "[[@TEST_NAME]]";

class C {}

interface I {}
interface Incomplete;

// CHECK:STDERR: fail_declaration_lookup_into_incomplete.carbon:[[@LINE+2]]:1-42: error: constraint `I where .Self impls Incomplete` being implemented requires that `C` implements `Incomplete` [IdentifiedRequireImplsNotImplemented]
// CHECK:STDERR:
impl C as I where .Self impls Incomplete {
  // CHECK:STDERR: fail_declaration_lookup_into_incomplete.carbon:[[@LINE+3]]:19-30: error: member access into incomplete facet type `Incomplete` [QualifiedExprInIncompleteFacetTypeScope]
  // CHECK:STDERR: fail_declaration_lookup_into_incomplete.carbon:[[@LINE-6]]:1-21: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR:
  fn F() -> Self.(Incomplete.T);
};

// --- fail_unidentified_constraint.carbon
library "[[@TEST_NAME]]";

constraint X;
class C {}

// Requires X identified.
// CHECK:STDERR: fail_unidentified_constraint.carbon:[[@LINE+3]]:1-12: error: facet type `X` cannot be identified in `impl as` [ImplOfUnidentifiedFacetType]
// CHECK:STDERR: fail_unidentified_constraint.carbon:[[@LINE-5]]:1-13: note: constraint was forward declared here [NamedConstraintForwardDeclaredHere]
// CHECK:STDERR:
impl C as X;

// --- nested_require_incomplete_interface.carbon
library "[[@TEST_NAME]]";

interface Z;
constraint Y {
  // Not extend, so only required to be identified.
  require impls Z;
}
interface X {
  let X1: type;

  // Not extend, so only required to be identified.
  require impls Y;
}

// Requires `X` to be complete, doesn't require Z to be complete.
fn F(unused generic T: X where .X1 = {}) {}

// --- fail_incomplete_constraint.carbon
library "[[@TEST_NAME]]";

constraint A;

interface B {
  // CHECK:STDERR: fail_incomplete_constraint.carbon:[[@LINE+3]]:17: error: facet type `A` cannot be identified in `require` declaration [RequireImplsUnidentifiedFacetType]
  // CHECK:STDERR: fail_incomplete_constraint.carbon:[[@LINE-4]]:1-13: note: constraint was forward declared here [NamedConstraintForwardDeclaredHere]
  // CHECK:STDERR:
  require impls A;
}

// CHECK:STDOUT: --- fail_empty_struct.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %.aaf: <witness> = impl_self_witness %empty_struct_type, @I [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @empty_struct_type.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   impl_decl @empty_struct_type.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %.loc6_7.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc6_7.2: type = converted %.loc6_7.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %I.ref.loc6: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc6: <witness> = impl_self_witness @empty_struct_type.as.I.impl.%.loc6_7.2, @I [concrete = constants.%.aaf]
// CHECK:STDOUT:   impl_decl @empty_struct_type.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %.loc11_7.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc11_7.2: type = converted %.loc11_7.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %I.ref.loc11: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc11: <witness> = impl_self_witness @empty_struct_type.as.I.impl.%.loc11_7.2, @I [concrete = constants.%.aaf]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I;
// CHECK:STDOUT:
// CHECK:STDOUT: impl @empty_struct_type.as.I.impl: %.loc6_7.2 as %I.ref.loc6 {
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (), @empty_struct_type.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend %I.ref.loc6
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %.ffd: <witness> = impl_self_witness %C, @J [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness @C.as.J.impl.%J.impl_witness_table [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .I = <poisoned>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [concrete = constants.%J.type] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @C.as.<error>.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: <error> = name_ref I, <error> [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.J.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc14: <witness> = impl_self_witness @C.as.J.impl.%C.ref, @J [concrete = constants.%.ffd]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J;
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.<error>.impl: %C.ref as <error>;
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.J.impl: %C.ref as %J.ref {
// CHECK:STDOUT:   %J.impl_witness_table = impl_witness_table (), @C.as.J.impl [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend %J.ref
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_class_no_forward_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %.ffd: <witness> = impl_self_witness %C, @J [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness @C.as.J.impl.%J.impl_witness_table [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [concrete = constants.%J.type] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @C.as.J.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9: <witness> = impl_self_witness @C.as.J.impl.%C.ref, @J [concrete = constants.%.ffd]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J;
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.J.impl: %C.ref as %J.ref {
// CHECK:STDOUT:   %J.impl_witness_table = impl_witness_table (), @C.as.J.impl [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend %J.ref
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_class_with_rewrite.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %.Self.frozen: %J.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.frozen.as_type: type = facet_access_type %.Self.frozen [symbolic_self]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [concrete = constants.%J.type] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @C.as.<error>.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref.loc9: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %J.ref.loc9: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %.Self.frozen.loc9: %J.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %base_facet_type.loc9 = requirement_base_facet_type %J.ref.loc9 [concrete]
// CHECK:STDOUT:     %.Self.ref.loc9: %J.type = name_ref .Self, %.Self.frozen.loc9 [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %.Self.as_type.loc9: type = facet_access_type %.Self.ref.loc9 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %.loc9_19: type = converted %.Self.ref.loc9, %.Self.as_type.loc9 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %X.ref.loc9: <error> = name_ref X, <error> [concrete = <error>]
// CHECK:STDOUT:     %.loc9_25: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %rewrite.loc9 = requirement_rewrite %X.ref.loc9, <error> [concrete]
// CHECK:STDOUT:     %.loc9_13: type = where_expr [concrete = <error>] {
// CHECK:STDOUT:       %base_facet_type.loc9 = requirement_base_facet_type %J.ref.loc9 [concrete]
// CHECK:STDOUT:       %rewrite.loc9 = requirement_rewrite %X.ref.loc9, <error> [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.<error>.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref.loc14: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %J.ref.loc14: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %.Self.frozen.loc14: %J.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %base_facet_type.loc14 = requirement_base_facet_type %J.ref.loc14 [concrete]
// CHECK:STDOUT:     %.Self.ref.loc14: %J.type = name_ref .Self, %.Self.frozen.loc14 [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %.Self.as_type.loc14: type = facet_access_type %.Self.ref.loc14 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %.loc14_19: type = converted %.Self.ref.loc14, %.Self.as_type.loc14 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %X.ref.loc14: <error> = name_ref X, <error> [concrete = <error>]
// CHECK:STDOUT:     %.loc14_25: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %rewrite.loc14 = requirement_rewrite %X.ref.loc14, <error> [concrete]
// CHECK:STDOUT:     %.loc14_13: type = where_expr [concrete = <error>] {
// CHECK:STDOUT:       %base_facet_type.loc14 = requirement_base_facet_type %J.ref.loc14 [concrete]
// CHECK:STDOUT:       %rewrite.loc14 = requirement_rewrite %X.ref.loc14, <error> [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J;
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.<error>.impl: %C.ref.loc9 as <error> {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend <error>
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- incomplete_where.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.37e: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Incomplete.type: type = facet_type <@Incomplete> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %I.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.frozen.as_type: type = facet_access_type %.Self.frozen [symbolic_self]
// CHECK:STDOUT:   %.Self: %I.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I where .Self impls @Incomplete> [concrete]
// CHECK:STDOUT:   %.b7f: <witness> = impl_self_witness %C, @I [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @C.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %Self.d08: %Incomplete.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %.f0c: <witness> = impl_self_witness %C, @Incomplete [concrete]
// CHECK:STDOUT:   %Incomplete.impl_witness: <witness> = impl_witness @C.as.Incomplete.impl.%Incomplete.impl_witness_table [concrete]
// CHECK:STDOUT:   %Incomplete.facet: %Incomplete.type = facet_value %C, (%Incomplete.impl_witness) [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C, (%I.impl_witness) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .Incomplete = %Incomplete.decl.loc6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %Incomplete.decl.loc6: type = interface_decl @Incomplete [concrete = constants.%Incomplete.type] {} {}
// CHECK:STDOUT:   impl_decl @C.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref.loc8: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref.loc8: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self.frozen.loc8: %I.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %base_facet_type.loc8 = requirement_base_facet_type %I.ref.loc8 [concrete]
// CHECK:STDOUT:     %.Self.ref.loc8_19.1: %I.type = name_ref .Self, %.Self.frozen.loc8 [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %Incomplete.ref.loc8: type = name_ref Incomplete, file.%Incomplete.decl.loc6 [concrete = constants.%Incomplete.type]
// CHECK:STDOUT:     %.Self.as_type.loc8_19.1: type = facet_access_type %.Self.ref.loc8_19.1 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %.loc8_19.1: type = converted %.Self.ref.loc8_19.1, %.Self.as_type.loc8_19.1 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %impls.loc8_25.1 = requirement_impls %.loc8_19.1, %Incomplete.ref.loc8 [concrete]
// CHECK:STDOUT:     %.Self.loc8: %I.type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc8_19.2: %I.type = name_ref .Self, %.Self.loc8 [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.as_type.loc8_19.2: type = facet_access_type %.Self.ref.loc8_19.2 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc8_19.2: type = converted %.Self.ref.loc8_19.1, %.Self.as_type.loc8_19.2 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %impls.loc8_25.2 = requirement_impls %.loc8_19.2, %Incomplete.ref.loc8 [concrete]
// CHECK:STDOUT:     %.loc8_13: type = where_expr [concrete = constants.%I_where.type] {
// CHECK:STDOUT:       %base_facet_type.loc8 = requirement_base_facet_type %I.ref.loc8 [concrete]
// CHECK:STDOUT:       %impls.loc8_25.2 = requirement_impls %.loc8_19.2, %Incomplete.ref.loc8 [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8: <witness> = impl_self_witness @C.as.I.impl.%C.ref.loc8, @I [concrete = constants.%.b7f]
// CHECK:STDOUT:   %Incomplete.decl.loc10: type = interface_decl @Incomplete [concrete = constants.%Incomplete.type] {} {}
// CHECK:STDOUT:   impl_decl @C.as.Incomplete.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %Incomplete.ref: type = name_ref Incomplete, file.%Incomplete.decl.loc6 [concrete = constants.%Incomplete.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc11: <witness> = impl_self_witness @C.as.Incomplete.impl.%C.ref, @Incomplete [concrete = constants.%.f0c]
// CHECK:STDOUT:   impl_decl @C.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref.loc13: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref.loc13: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self.frozen.loc13: %I.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %base_facet_type.loc13 = requirement_base_facet_type %I.ref.loc13 [concrete]
// CHECK:STDOUT:     %.Self.ref.loc13_19.1: %I.type = name_ref .Self, %.Self.frozen.loc13 [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %Incomplete.ref.loc13: type = name_ref Incomplete, file.%Incomplete.decl.loc6 [concrete = constants.%Incomplete.type]
// CHECK:STDOUT:     %.Self.as_type.loc13_19.1: type = facet_access_type %.Self.ref.loc13_19.1 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %.loc13_19.1: type = converted %.Self.ref.loc13_19.1, %.Self.as_type.loc13_19.1 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %impls.loc13_25.1 = requirement_impls %.loc13_19.1, %Incomplete.ref.loc13 [concrete]
// CHECK:STDOUT:     %.Self.loc13: %I.type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc13_19.2: %I.type = name_ref .Self, %.Self.loc13 [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.as_type.loc13_19.2: type = facet_access_type %.Self.ref.loc13_19.2 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc13_19.2: type = converted %.Self.ref.loc13_19.1, %.Self.as_type.loc13_19.2 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %impls.loc13_25.2 = requirement_impls %.loc13_19.2, %Incomplete.ref.loc13 [concrete]
// CHECK:STDOUT:     %.loc13_13: type = where_expr [concrete = constants.%I_where.type] {
// CHECK:STDOUT:       %base_facet_type.loc13 = requirement_base_facet_type %I.ref.loc13 [concrete]
// CHECK:STDOUT:       %impls.loc13_25.2 = requirement_impls %.loc13_19.2, %Incomplete.ref.loc13 [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc13: <witness> = impl_self_witness @C.as.I.impl.%C.ref.loc13, @I [concrete = constants.%.b7f]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self.37e]
// CHECK:STDOUT:   %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Incomplete {
// CHECK:STDOUT:   %Self: %Incomplete.type = symbolic_binding Self, 0 [symbolic = constants.%Self.d08]
// CHECK:STDOUT:   %Incomplete.WithSelf.decl = interface_with_self_decl @Incomplete [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.I.impl: %C.ref.loc8 as %I.ref.loc8 {
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (), @C.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend %I.ref.loc8
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.Incomplete.impl: %C.ref as %Incomplete.ref {
// CHECK:STDOUT:   %Incomplete.impl_witness_table = impl_witness_table (), @C.as.Incomplete.impl [concrete]
// CHECK:STDOUT:   %Incomplete.impl_witness: <witness> = impl_witness %Incomplete.impl_witness_table [concrete = constants.%Incomplete.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend %Incomplete.ref
// CHECK:STDOUT:   witness = %Incomplete.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self.37e) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Incomplete.WithSelf(constants.%Self.d08) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Incomplete.WithSelf(constants.%Incomplete.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- incomplete_where_rewrite.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self.653: %J.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type @J [concrete]
// CHECK:STDOUT:   %assoc0: %J.assoc_type = assoc_entity element0, @J.WithSelf.%T [concrete]
// CHECK:STDOUT:   %Incomplete.type: type = facet_type <@Incomplete> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %J.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.frozen.as_type: type = facet_access_type %.Self.frozen [symbolic_self]
// CHECK:STDOUT:   %.aa3: <witness> = impl_self_witness %.Self.frozen, @J [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.643: type = impl_witness_access %.aa3, element0 [symbolic_self]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %.Self: %J.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %.b2e: <witness> = impl_self_witness %.Self, @J [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.4f1: type = impl_witness_access %.b2e, element0 [symbolic_self]
// CHECK:STDOUT:   %J_where.type: type = facet_type <@J where .Self impls @Incomplete and %impl.elem0.4f1 = %empty_tuple.type> [concrete]
// CHECK:STDOUT:   %.ffd: <witness> = impl_self_witness %C, @J [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness @C.as.J.impl.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %Self.d08: %Incomplete.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %.f0c: <witness> = impl_self_witness %C, @Incomplete [concrete]
// CHECK:STDOUT:   %Incomplete.impl_witness: <witness> = impl_witness @C.as.Incomplete.impl.%Incomplete.impl_witness_table [concrete]
// CHECK:STDOUT:   %Incomplete.facet: %Incomplete.type = facet_value %C, (%Incomplete.impl_witness) [concrete]
// CHECK:STDOUT:   %J.facet: %J.type = facet_value %C, (%J.impl_witness) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .Incomplete = %Incomplete.decl.loc6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [concrete = constants.%J.type] {} {}
// CHECK:STDOUT:   %Incomplete.decl.loc6: type = interface_decl @Incomplete [concrete = constants.%Incomplete.type] {} {}
// CHECK:STDOUT:   impl_decl @C.as.J.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref.loc8: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %J.ref.loc8: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %.Self.frozen.loc8: %J.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %base_facet_type.loc8 = requirement_base_facet_type %J.ref.loc8 [concrete]
// CHECK:STDOUT:     %.Self.ref.loc8_19.1: %J.type = name_ref .Self, %.Self.frozen.loc8 [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %Incomplete.ref.loc8: type = name_ref Incomplete, file.%Incomplete.decl.loc6 [concrete = constants.%Incomplete.type]
// CHECK:STDOUT:     %.Self.as_type.loc8_19.1: type = facet_access_type %.Self.ref.loc8_19.1 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %.loc8_19.1: type = converted %.Self.ref.loc8_19.1, %.Self.as_type.loc8_19.1 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %impls.loc8_25.1 = requirement_impls %.loc8_19.1, %Incomplete.ref.loc8 [concrete]
// CHECK:STDOUT:     %.Self.ref.loc8_46: %J.type = name_ref .Self, %.Self.frozen.loc8 [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %.Self.as_type.loc8_46: type = facet_access_type %.Self.ref.loc8_46 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %.loc8_46: type = converted %.Self.ref.loc8_46, %.Self.as_type.loc8_46 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %T.ref.loc8: %J.assoc_type = name_ref T, @T.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc8_46.1: type = impl_witness_access constants.%.aa3, element0 [symbolic_self = constants.%impl.elem0.643]
// CHECK:STDOUT:     %.loc8_52.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc8_52.2: type = converted %.loc8_52.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %rewrite.loc8_49.1 = requirement_rewrite %impl.elem0.loc8_46.1, %.loc8_52.2 [concrete]
// CHECK:STDOUT:     %.Self.loc8: %J.type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc8_19.2: %J.type = name_ref .Self, %.Self.loc8 [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.as_type.loc8_19.2: type = facet_access_type %.Self.ref.loc8_19.2 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc8_19.2: type = converted %.Self.ref.loc8_19.1, %.Self.as_type.loc8_19.2 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %impls.loc8_25.2 = requirement_impls %.loc8_19.2, %Incomplete.ref.loc8 [concrete]
// CHECK:STDOUT:     %impl.elem0.loc8_46.2: type = impl_witness_access constants.%.b2e, element0 [symbolic_self = constants.%impl.elem0.4f1]
// CHECK:STDOUT:     %rewrite.loc8_49.2 = requirement_rewrite %impl.elem0.loc8_46.2, %.loc8_52.2 [concrete]
// CHECK:STDOUT:     %.loc8_13: type = where_expr [concrete = constants.%J_where.type] {
// CHECK:STDOUT:       %base_facet_type.loc8 = requirement_base_facet_type %J.ref.loc8 [concrete]
// CHECK:STDOUT:       %impls.loc8_25.2 = requirement_impls %.loc8_19.2, %Incomplete.ref.loc8 [concrete]
// CHECK:STDOUT:       %rewrite.loc8_49.2 = requirement_rewrite %impl.elem0.loc8_46.2, %.loc8_52.2 [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8: <witness> = impl_self_witness @C.as.J.impl.%C.ref.loc8, @J [concrete = constants.%.ffd]
// CHECK:STDOUT:   %Incomplete.decl.loc10: type = interface_decl @Incomplete [concrete = constants.%Incomplete.type] {} {}
// CHECK:STDOUT:   impl_decl @C.as.Incomplete.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %Incomplete.ref: type = name_ref Incomplete, file.%Incomplete.decl.loc6 [concrete = constants.%Incomplete.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc11: <witness> = impl_self_witness @C.as.Incomplete.impl.%C.ref, @Incomplete [concrete = constants.%.f0c]
// CHECK:STDOUT:   impl_decl @C.as.J.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref.loc13: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %J.ref.loc13: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %.Self.frozen.loc13: %J.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %base_facet_type.loc13 = requirement_base_facet_type %J.ref.loc13 [concrete]
// CHECK:STDOUT:     %.Self.ref.loc13_19.1: %J.type = name_ref .Self, %.Self.frozen.loc13 [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %Incomplete.ref.loc13: type = name_ref Incomplete, file.%Incomplete.decl.loc6 [concrete = constants.%Incomplete.type]
// CHECK:STDOUT:     %.Self.as_type.loc13_19.1: type = facet_access_type %.Self.ref.loc13_19.1 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %.loc13_19.1: type = converted %.Self.ref.loc13_19.1, %.Self.as_type.loc13_19.1 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %impls.loc13_25.1 = requirement_impls %.loc13_19.1, %Incomplete.ref.loc13 [concrete]
// CHECK:STDOUT:     %.Self.ref.loc13_46: %J.type = name_ref .Self, %.Self.frozen.loc13 [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %.Self.as_type.loc13_46: type = facet_access_type %.Self.ref.loc13_46 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %.loc13_46: type = converted %.Self.ref.loc13_46, %.Self.as_type.loc13_46 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %T.ref.loc13: %J.assoc_type = name_ref T, @T.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc13_46.1: type = impl_witness_access constants.%.aa3, element0 [symbolic_self = constants.%impl.elem0.643]
// CHECK:STDOUT:     %.loc13_52.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc13_52.2: type = converted %.loc13_52.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %rewrite.loc13_49.1 = requirement_rewrite %impl.elem0.loc13_46.1, %.loc13_52.2 [concrete]
// CHECK:STDOUT:     %.Self.loc13: %J.type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc13_19.2: %J.type = name_ref .Self, %.Self.loc13 [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.as_type.loc13_19.2: type = facet_access_type %.Self.ref.loc13_19.2 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc13_19.2: type = converted %.Self.ref.loc13_19.1, %.Self.as_type.loc13_19.2 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %impls.loc13_25.2 = requirement_impls %.loc13_19.2, %Incomplete.ref.loc13 [concrete]
// CHECK:STDOUT:     %impl.elem0.loc13_46.2: type = impl_witness_access constants.%.b2e, element0 [symbolic_self = constants.%impl.elem0.4f1]
// CHECK:STDOUT:     %rewrite.loc13_49.2 = requirement_rewrite %impl.elem0.loc13_46.2, %.loc13_52.2 [concrete]
// CHECK:STDOUT:     %.loc13_13: type = where_expr [concrete = constants.%J_where.type] {
// CHECK:STDOUT:       %base_facet_type.loc13 = requirement_base_facet_type %J.ref.loc13 [concrete]
// CHECK:STDOUT:       %impls.loc13_25.2 = requirement_impls %.loc13_19.2, %Incomplete.ref.loc13 [concrete]
// CHECK:STDOUT:       %rewrite.loc13_49.2 = requirement_rewrite %impl.elem0.loc13_46.2, %.loc13_52.2 [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc13: <witness> = impl_self_witness @C.as.J.impl.%C.ref.loc13, @J [concrete = constants.%.ffd]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = symbolic_binding Self, 0 [symbolic = constants.%Self.653]
// CHECK:STDOUT:   %J.WithSelf.decl = interface_with_self_decl @J [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %T: type = assoc_const_decl @T [concrete] {
// CHECK:STDOUT:     %assoc0: %J.assoc_type = assoc_entity element0, @J.WithSelf.%T [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .T = @T.%assoc0
// CHECK:STDOUT:   witness = (@J.WithSelf.%T)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Incomplete {
// CHECK:STDOUT:   %Self: %Incomplete.type = symbolic_binding Self, 0 [symbolic = constants.%Self.d08]
// CHECK:STDOUT:   %Incomplete.WithSelf.decl = interface_with_self_decl @Incomplete [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.J.impl: %C.ref.loc8 as %J.ref.loc8 {
// CHECK:STDOUT:   %J.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @C.as.J.impl [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness]
// CHECK:STDOUT:   %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend %J.ref.loc8
// CHECK:STDOUT:   witness = %J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.Incomplete.impl: %C.ref as %Incomplete.ref {
// CHECK:STDOUT:   %Incomplete.impl_witness_table = impl_witness_table (), @C.as.Incomplete.impl [concrete]
// CHECK:STDOUT:   %Incomplete.impl_witness: <witness> = impl_witness %Incomplete.impl_witness_table [concrete = constants.%Incomplete.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend %Incomplete.ref
// CHECK:STDOUT:   witness = %Incomplete.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%Self.653) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%.Self.frozen) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Incomplete.WithSelf(constants.%Self.d08) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Incomplete.WithSelf(constants.%Incomplete.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%J.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_incomplete_where_definition.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Incomplete.type: type = facet_type <@Incomplete> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %I.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.frozen.as_type: type = facet_access_type %.Self.frozen [symbolic_self]
// CHECK:STDOUT:   %.Self: %I.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I where .Self impls @Incomplete> [concrete]
// CHECK:STDOUT:   %.b7f: <witness> = impl_self_witness %C, @I [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @C.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .Incomplete = %Incomplete.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %Incomplete.decl: type = interface_decl @Incomplete [concrete = constants.%Incomplete.type] {} {}
// CHECK:STDOUT:   impl_decl @C.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self.frozen: %I.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %base_facet_type = requirement_base_facet_type %I.ref [concrete]
// CHECK:STDOUT:     %.Self.ref.loc10_19.1: %I.type = name_ref .Self, %.Self.frozen [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %Incomplete.ref: type = name_ref Incomplete, file.%Incomplete.decl [concrete = constants.%Incomplete.type]
// CHECK:STDOUT:     %.Self.as_type.loc10_19.1: type = facet_access_type %.Self.ref.loc10_19.1 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %.loc10_19.1: type = converted %.Self.ref.loc10_19.1, %.Self.as_type.loc10_19.1 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %impls.loc10_25.1 = requirement_impls %.loc10_19.1, %Incomplete.ref [concrete]
// CHECK:STDOUT:     %.Self: %I.type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc10_19.2: %I.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.as_type.loc10_19.2: type = facet_access_type %.Self.ref.loc10_19.2 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc10_19.2: type = converted %.Self.ref.loc10_19.1, %.Self.as_type.loc10_19.2 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %impls.loc10_25.2 = requirement_impls %.loc10_19.2, %Incomplete.ref [concrete]
// CHECK:STDOUT:     %.loc10_13: type = where_expr [concrete = constants.%I_where.type] {
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %I.ref [concrete]
// CHECK:STDOUT:       %impls.loc10_25.2 = requirement_impls %.loc10_19.2, %Incomplete.ref [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10: <witness> = impl_self_witness @C.as.I.impl.%C.ref, @I [concrete = constants.%.b7f]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Incomplete;
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.I.impl: %C.ref as %I.ref {
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (), @C.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend %I.ref
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_declaration_lookup_into_incomplete.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Incomplete.type: type = facet_type <@Incomplete> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %I.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.frozen.as_type: type = facet_access_type %.Self.frozen [symbolic_self]
// CHECK:STDOUT:   %.Self: %I.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I where .Self impls @Incomplete> [concrete]
// CHECK:STDOUT:   %.b7f: <witness> = impl_self_witness %C, @I [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @C.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %C.as.I.impl.F.type: type = fn_type @C.as.I.impl.F [concrete]
// CHECK:STDOUT:   %C.as.I.impl.F: %C.as.I.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .Incomplete = %Incomplete.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %Incomplete.decl: type = interface_decl @Incomplete [concrete = constants.%Incomplete.type] {} {}
// CHECK:STDOUT:   impl_decl @C.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self.frozen: %I.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %base_facet_type = requirement_base_facet_type %I.ref [concrete]
// CHECK:STDOUT:     %.Self.ref.loc10_19.1: %I.type = name_ref .Self, %.Self.frozen [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %Incomplete.ref: type = name_ref Incomplete, file.%Incomplete.decl [concrete = constants.%Incomplete.type]
// CHECK:STDOUT:     %.Self.as_type.loc10_19.1: type = facet_access_type %.Self.ref.loc10_19.1 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %.loc10_19.1: type = converted %.Self.ref.loc10_19.1, %.Self.as_type.loc10_19.1 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %impls.loc10_25.1 = requirement_impls %.loc10_19.1, %Incomplete.ref [concrete]
// CHECK:STDOUT:     %.Self: %I.type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc10_19.2: %I.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.as_type.loc10_19.2: type = facet_access_type %.Self.ref.loc10_19.2 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc10_19.2: type = converted %.Self.ref.loc10_19.1, %.Self.as_type.loc10_19.2 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %impls.loc10_25.2 = requirement_impls %.loc10_19.2, %Incomplete.ref [concrete]
// CHECK:STDOUT:     %.loc10_13: type = where_expr [concrete = constants.%I_where.type] {
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %I.ref [concrete]
// CHECK:STDOUT:       %impls.loc10_25.2 = requirement_impls %.loc10_19.2, %Incomplete.ref [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10: <witness> = impl_self_witness @C.as.I.impl.%C.ref, @I [concrete = constants.%.b7f]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Incomplete = <poisoned>
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Incomplete;
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.I.impl: %C.ref as %I.ref {
// CHECK:STDOUT:   %C.as.I.impl.F.decl: %C.as.I.impl.F.type = fn_decl @C.as.I.impl.F [concrete = constants.%C.as.I.impl.F] {
// CHECK:STDOUT:     %return.patt: <error> = return_slot_pattern <error>, <error> [concrete = <error>]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.I.impl.%C.ref [concrete = constants.%C]
// CHECK:STDOUT:     %Incomplete.ref: type = name_ref Incomplete, file.%Incomplete.decl [concrete = constants.%Incomplete.type]
// CHECK:STDOUT:     %T.ref: <error> = name_ref T, <error> [concrete = <error>]
// CHECK:STDOUT:     %return: <error> = return_slot <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (), @C.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Incomplete = <poisoned>
// CHECK:STDOUT:   .F = %C.as.I.impl.F.decl
// CHECK:STDOUT:   extend %I.ref
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.I.impl.F() -> <error>;
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_unidentified_constraint.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %X.type: type = facet_type <@X> [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .X = %X.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.decl: type = constraint_decl @X [concrete = constants.%X.type] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @C.as.<error>.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: constraint @X;
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.<error>.impl: %C.ref as <error>;
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- nested_require_incomplete_interface.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y> [concrete]
// CHECK:STDOUT:   %Self.d87: %Y.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type.138: type = facet_access_type %Self.d87 [symbolic]
// CHECK:STDOUT:   %X.type: type = facet_type <@X> [concrete]
// CHECK:STDOUT:   %Self.b43: %X.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %X.assoc_type: type = assoc_entity_type @X [concrete]
// CHECK:STDOUT:   %assoc0: %X.assoc_type = assoc_entity element0, @X.WithSelf.%X1 [concrete]
// CHECK:STDOUT:   %Self.as_type.ace: type = facet_access_type %Self.b43 [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen.197: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.frozen.5e5: %X.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.frozen.as_type: type = facet_access_type %.Self.frozen.5e5 [symbolic_self]
// CHECK:STDOUT:   %X.lookup_impl_witness.de0: <witness> = lookup_impl_witness %.Self.frozen.5e5, @X [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.2f1: type = impl_witness_access %X.lookup_impl_witness.de0, element0 [symbolic_self]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %.Self: %X.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %X.lookup_impl_witness.f28: <witness> = lookup_impl_witness %.Self, @X [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.061: type = impl_witness_access %X.lookup_impl_witness.f28, element0 [symbolic_self]
// CHECK:STDOUT:   %X_where.type: type = facet_type <@X where %impl.elem0.061 = %empty_struct_type> [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %X_where.type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: %X_where.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Z = %Z.decl
// CHECK:STDOUT:     .Y = %Y.decl
// CHECK:STDOUT:     .X = %X.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Z.decl: type = interface_decl @Z [concrete = constants.%Z.type] {} {}
// CHECK:STDOUT:   %Y.decl: type = constraint_decl @Y [concrete = constants.%Y.type] {} {}
// CHECK:STDOUT:   %X.decl: type = interface_decl @X [concrete = constants.%X.type] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc16_22.1: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc16_22.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc16_26.1: type = splice_block %.loc16_26.2 [concrete = constants.%X_where.type] {
// CHECK:STDOUT:       %.Self.frozen.loc16_22: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.197]
// CHECK:STDOUT:       %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X.type]
// CHECK:STDOUT:       %.Self.frozen.loc16_26: %X.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.5e5]
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %X.ref [concrete]
// CHECK:STDOUT:       %.Self.ref: %X.type = name_ref .Self, %.Self.frozen.loc16_26 [symbolic_self = constants.%.Self.frozen.5e5]
// CHECK:STDOUT:       %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:       %.loc16_32: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:       %X1.ref: %X.assoc_type = name_ref X1, @X1.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:       %impl.elem0.loc16_32.1: type = impl_witness_access constants.%X.lookup_impl_witness.de0, element0 [symbolic_self = constants.%impl.elem0.2f1]
// CHECK:STDOUT:       %.loc16_39.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:       %.loc16_39.2: type = converted %.loc16_39.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:       %rewrite.loc16_36.1 = requirement_rewrite %impl.elem0.loc16_32.1, %.loc16_39.2 [concrete]
// CHECK:STDOUT:       %impl.elem0.loc16_32.2: type = impl_witness_access constants.%X.lookup_impl_witness.f28, element0 [symbolic_self = constants.%impl.elem0.061]
// CHECK:STDOUT:       %rewrite.loc16_36.2 = requirement_rewrite %impl.elem0.loc16_32.2, %.loc16_39.2 [concrete]
// CHECK:STDOUT:       %.loc16_26.2: type = where_expr [concrete = constants.%X_where.type] {
// CHECK:STDOUT:         %base_facet_type = requirement_base_facet_type %X.ref [concrete]
// CHECK:STDOUT:         %rewrite.loc16_36.2 = requirement_rewrite %impl.elem0.loc16_32.2, %.loc16_39.2 [concrete]
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc16_22.2: %X_where.type = symbolic_binding T, 0 [symbolic = %T.loc16_22.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Z;
// CHECK:STDOUT:
// CHECK:STDOUT: interface @X {
// CHECK:STDOUT:   %Self: %X.type = symbolic_binding Self, 0 [symbolic = constants.%Self.b43]
// CHECK:STDOUT:   %X.WithSelf.decl = interface_with_self_decl @X [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %X1: type = assoc_const_decl @X1 [concrete] {
// CHECK:STDOUT:     %assoc0: %X.assoc_type = assoc_entity element0, @X.WithSelf.%X1 [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.WithSelf.Self.as_type.impls.Y.type.require.decl = require_decl @X.WithSelf.Self.as_type.impls.Y.type.require [concrete] {
// CHECK:STDOUT:     require %Self.as_type.loc12_11.1 impls %Y.ref
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.as_type.loc12_11.1: type = facet_access_type @X.%Self [symbolic = %Self.as_type.loc12_11.2 (constants.%Self.as_type.ace)]
// CHECK:STDOUT:     %Y.ref: type = name_ref Y, file.%Y.decl [concrete = constants.%Y.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Y = <poisoned>
// CHECK:STDOUT:   .X1 = @X1.%assoc0
// CHECK:STDOUT:   .Y = <poisoned>
// CHECK:STDOUT:   witness = (@X.WithSelf.%X1)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:   @X.WithSelf.Self.as_type.impls.Y.type.require {
// CHECK:STDOUT:     require @X.WithSelf.Self.as_type.impls.Y.type.require.%Self.as_type.loc12_11.1 impls @X.WithSelf.Self.as_type.impls.Y.type.require.%Y.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: constraint @Y {
// CHECK:STDOUT:   %Self: %Y.type = symbolic_binding Self, 0 [symbolic = constants.%Self.d87]
// CHECK:STDOUT:   %Y.WithSelf.decl = constraint_with_self_decl @Y [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %Y.WithSelf.Self.as_type.impls.Z.type.require.decl = require_decl @Y.WithSelf.Self.as_type.impls.Z.type.require [concrete] {
// CHECK:STDOUT:     require %Self.as_type.loc6_11.1 impls %Z.ref
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.as_type.loc6_11.1: type = facet_access_type @Y.%Self [symbolic = %Self.as_type.loc6_11.2 (constants.%Self.as_type.138)]
// CHECK:STDOUT:     %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Z = <poisoned>
// CHECK:STDOUT:   .Z = <poisoned>
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:   @Y.WithSelf.Self.as_type.impls.Z.type.require {
// CHECK:STDOUT:     require @Y.WithSelf.Self.as_type.impls.Z.type.require.%Self.as_type.loc6_11.1 impls @Y.WithSelf.Self.as_type.impls.Z.type.require.%Z.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @Y.WithSelf.Self.as_type.impls.Z.type.require(@Y.%Self: %Y.type) {
// CHECK:STDOUT:   %Self: %Y.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.d87)]
// CHECK:STDOUT:   %Self.as_type.loc6_11.2: type = facet_access_type %Self [symbolic = %Self.as_type.loc6_11.2 (constants.%Self.as_type.138)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @X.WithSelf.Self.as_type.impls.Y.type.require(@X.%Self: %X.type) {
// CHECK:STDOUT:   %Self: %X.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.b43)]
// CHECK:STDOUT:   %Self.as_type.loc12_11.2: type = facet_access_type %Self [symbolic = %Self.as_type.loc12_11.2 (constants.%Self.as_type.ace)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc16_22.2: %X_where.type) {
// CHECK:STDOUT:   %T.patt.loc16_22.2: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc16_22.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc16_22.1: %X_where.type = symbolic_binding T, 0 [symbolic = %T.loc16_22.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y.WithSelf(constants.%Self.d87) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y.WithSelf.Self.as_type.impls.Z.type.require(constants.%Self.d87) {
// CHECK:STDOUT:   %Self => constants.%Self.d87
// CHECK:STDOUT:   %Self.as_type.loc6_11.2 => constants.%Self.as_type.138
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X.WithSelf(constants.%Self.b43) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y.WithSelf(constants.%Self.b43) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y.WithSelf.Self.as_type.impls.Z.type.require(constants.%Self.b43) {
// CHECK:STDOUT:   %Self => constants.%Self.b43
// CHECK:STDOUT:   %Self.as_type.loc6_11.2 => constants.%Self.as_type.ace
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X.WithSelf.Self.as_type.impls.Y.type.require(constants.%Self.b43) {
// CHECK:STDOUT:   %Self => constants.%Self.b43
// CHECK:STDOUT:   %Self.as_type.loc12_11.2 => constants.%Self.as_type.ace
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X.WithSelf(constants.%.Self.frozen.5e5) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc16_22.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc16_22.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_incomplete_constraint.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = facet_type <@A> [concrete]
// CHECK:STDOUT:   %B.type: type = facet_type <@B> [concrete]
// CHECK:STDOUT:   %Self: %B.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: type = constraint_decl @A [concrete = constants.%A.type] {} {}
// CHECK:STDOUT:   %B.decl: type = interface_decl @B [concrete = constants.%B.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @B {
// CHECK:STDOUT:   %Self: %B.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %B.WithSelf.decl = interface_with_self_decl @B [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: constraint @A;
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B.WithSelf(constants.%Self) {}
// CHECK:STDOUT:

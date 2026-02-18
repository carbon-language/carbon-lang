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

// CHECK:STDERR: fail_empty_struct.carbon:[[@LINE+7]]:1: error: definition of impl as incomplete facet type `I` [ImplAsIncompleteFacetTypeDefinition]
// CHECK:STDERR: impl {} as I {}
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR: fail_empty_struct.carbon:[[@LINE-8]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface I;
// CHECK:STDERR: ^~~~~~~~~~~~
// CHECK:STDERR:
impl {} as I {}

// --- fail_class.carbon
library "[[@TEST_NAME]]";

interface J;
class C {}

// Requires I identified.
// CHECK:STDERR: fail_class.carbon:[[@LINE+4]]:11: error: name `I` not found [NameNotFound]
// CHECK:STDERR: impl C as I;
// CHECK:STDERR:           ^
// CHECK:STDERR:
impl C as I;

// CHECK:STDERR: fail_class.carbon:[[@LINE+7]]:1: error: definition of impl as incomplete facet type `J` [ImplAsIncompleteFacetTypeDefinition]
// CHECK:STDERR: impl C as J {}
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR: fail_class.carbon:[[@LINE-13]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface J;
// CHECK:STDERR: ^~~~~~~~~~~~
// CHECK:STDERR:
impl C as J {}

// --- fail_class_no_forward_decl.carbon
library "[[@TEST_NAME]]";

interface J;
class C {}

// CHECK:STDERR: fail_class_no_forward_decl.carbon:[[@LINE+7]]:1: error: definition of impl as incomplete facet type `J` [ImplAsIncompleteFacetTypeDefinition]
// CHECK:STDERR: impl C as J {}
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR: fail_class_no_forward_decl.carbon:[[@LINE-6]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface J;
// CHECK:STDERR: ^~~~~~~~~~~~
// CHECK:STDERR:
impl C as J {}

// --- fail_class_with_rewrite.carbon
library "[[@TEST_NAME]]";

interface J;
class C {}

// CHECK:STDERR: fail_class_with_rewrite.carbon:[[@LINE+7]]:19: error: member access into facet of incomplete type `J` [IncompleteTypeInMemberAccessOfFacet]
// CHECK:STDERR: impl C as J where .X = ();
// CHECK:STDERR:                   ^~
// CHECK:STDERR: fail_class_with_rewrite.carbon:[[@LINE-6]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface J;
// CHECK:STDERR: ^~~~~~~~~~~~
// CHECK:STDERR:
impl C as J where .X = ();

// CHECK:STDERR: fail_class_with_rewrite.carbon:[[@LINE+7]]:19: error: member access into facet of incomplete type `J` [IncompleteTypeInMemberAccessOfFacet]
// CHECK:STDERR: impl C as J where .X = () {}
// CHECK:STDERR:                   ^~
// CHECK:STDERR: fail_class_with_rewrite.carbon:[[@LINE-15]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface J;
// CHECK:STDERR: ^~~~~~~~~~~~
// CHECK:STDERR:
impl C as J where .X = () {}

// --- fail_todo_class_with_qualified_rewrite.carbon
library "[[@TEST_NAME]]";

interface J;
interface K { let X:! type; }
class C {}

// TODO: There should be no error here.
//
// CHECK:STDERR: fail_todo_class_with_qualified_rewrite.carbon:[[@LINE+8]]:38: error: expected identifier or `Self` after `.` [ExpectedIdentifierOrSelfAfterPeriod]
// CHECK:STDERR: impl C as J where .Self impls K and .(K.X) = ();
// CHECK:STDERR:                                      ^
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_class_with_qualified_rewrite.carbon:[[@LINE+4]]:38: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
// CHECK:STDERR: impl C as J where .Self impls K and .(K.X) = ();
// CHECK:STDERR:                                      ^
// CHECK:STDERR:
impl C as J where .Self impls K and .(K.X) = ();

// TODO: The failure here should be that J is incomplete, once the rewrite of
// `.(K.X)` works. Since any rewrite in the decl requires us to know the size of
// the witness table which requires all interfaces to be complete.
//
// CHECK:STDERR: fail_todo_class_with_qualified_rewrite.carbon:[[@LINE+4]]:38: error: expected identifier or `Self` after `.` [ExpectedIdentifierOrSelfAfterPeriod]
// CHECK:STDERR: impl C as J where .Self impls K and .(K.X) = () {}
// CHECK:STDERR:                                      ^
// CHECK:STDERR:
impl C as J where .Self impls K and .(K.X) = () {}

// --- incomplete_where.carbon
library "[[@TEST_NAME]]";

class C {}

interface I {}
interface Incomplete;

impl C as I where .Self impls Incomplete {}

// --- declaration_incomplete_where_rewrite.carbon
library "[[@TEST_NAME]]";

class C {}

interface J { let T:! type; }
interface Incomplete;

impl C as J where .Self impls Incomplete and .T = ();

impl C as J where .Self impls Incomplete and .T = () {}

// --- fail_declaration_lookup_into_incomplete.carbon
library "[[@TEST_NAME]]";

class C {}

interface I {}
interface Incomplete;

impl C as I where .Self impls Incomplete {
  // CHECK:STDERR: fail_declaration_lookup_into_incomplete.carbon:[[@LINE+7]]:19: error: member access into incomplete facet type `Incomplete` [QualifiedExprInIncompleteFacetTypeScope]
  // CHECK:STDERR:   fn F() -> Self.(Incomplete.T);
  // CHECK:STDERR:                   ^~~~~~~~~~~~
  // CHECK:STDERR: fail_declaration_lookup_into_incomplete.carbon:[[@LINE-6]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Incomplete;
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  fn F() -> Self.(Incomplete.T);
};

// --- fail_unidentified_constraint.carbon
library "[[@TEST_NAME]]";

constraint X;
class C {}

// Requires X identified.
// CHECK:STDERR: fail_unidentified_constraint.carbon:[[@LINE+7]]:1: error: facet type `X` cannot be identified in `impl as` [ImplOfUnidentifiedFacetType]
// CHECK:STDERR: impl C as X;
// CHECK:STDERR: ^~~~~~~~~~~~
// CHECK:STDERR: fail_unidentified_constraint.carbon:[[@LINE-7]]:1: note: constraint was forward declared here [NamedConstraintForwardDeclaredHere]
// CHECK:STDERR: constraint X;
// CHECK:STDERR: ^~~~~~~~~~~~~
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
  let X1:! type;

  // Not extend, so only required to be identified.
  require impls Y;
}

// Requires `X` to be complete, doesn't require Z to be complete.
fn F(T:! X where .X1 = {}) {}

// --- fail_incomplete_constraint.carbon
library "[[@TEST_NAME]]";

constraint A;

interface B {
  // CHECK:STDERR: fail_incomplete_constraint.carbon:[[@LINE+7]]:17: error: facet type `A` cannot be identified in `require` declaration [RequireImplsUnidentifiedFacetType]
  // CHECK:STDERR:   require impls A;
  // CHECK:STDERR:                 ^
  // CHECK:STDERR: fail_incomplete_constraint.carbon:[[@LINE-6]]:1: note: constraint was forward declared here [NamedConstraintForwardDeclaredHere]
  // CHECK:STDERR: constraint A;
  // CHECK:STDERR: ^~~~~~~~~~~~~
  // CHECK:STDERR:
  require impls A;
}

// CHECK:STDOUT: --- fail_empty_struct.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
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
// CHECK:STDOUT:   impl_decl @empty_struct_type.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %.loc15_7.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc15_7.2: type = converted %.loc15_7.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %I.ref.loc15: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I;
// CHECK:STDOUT:
// CHECK:STDOUT: impl @empty_struct_type.as.I.impl: %.loc6_7.2 as %I.ref.loc6 {
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (), @empty_struct_type.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
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
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J;
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.J.impl: %C.ref as %J.ref {
// CHECK:STDOUT:   %J.impl_witness_table = impl_witness_table (), @C.as.J.impl [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
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
// CHECK:STDOUT:   %.Self: %J.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self [symbolic_self]
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
// CHECK:STDOUT:     %C.ref.loc13: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %J.ref.loc13: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %.Self.2: %J.type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc13: %J.type = name_ref .Self, %.Self.2 [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.as_type.loc13: type = facet_access_type %.Self.ref.loc13 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc13_19: type = converted %.Self.ref.loc13, %.Self.as_type.loc13 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %X.ref.loc13: <error> = name_ref X, <error> [concrete = <error>]
// CHECK:STDOUT:     %.loc13_25: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc13_13: type = where_expr %.Self.2 [concrete = <error>] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%J.type
// CHECK:STDOUT:       requirement_rewrite %X.ref.loc13, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.<error>.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref.loc22: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %J.ref.loc22: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %.Self.1: %J.type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc22: %J.type = name_ref .Self, %.Self.1 [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.as_type.loc22: type = facet_access_type %.Self.ref.loc22 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc22_19: type = converted %.Self.ref.loc22, %.Self.as_type.loc22 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %X.ref.loc22: <error> = name_ref X, <error> [concrete = <error>]
// CHECK:STDOUT:     %.loc22_25: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc22_13: type = where_expr %.Self.1 [concrete = <error>] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%J.type
// CHECK:STDOUT:       requirement_rewrite %X.ref.loc22, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J;
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.<error>.impl: %C.ref.loc13 as %.loc13_13 {
// CHECK:STDOUT: !members:
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
// CHECK:STDOUT: --- fail_todo_class_with_qualified_rewrite.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %K.type: type = facet_type <@K> [concrete]
// CHECK:STDOUT:   %Self: %K.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %K.assoc_type: type = assoc_entity_type @K [concrete]
// CHECK:STDOUT:   %assoc0: %K.assoc_type = assoc_entity element0, @K.%X [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J;
// CHECK:STDOUT:
// CHECK:STDOUT: interface @K {
// CHECK:STDOUT:   %Self: %K.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   interface_with_self_decl @K [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %X: type = assoc_const_decl @X [concrete] {
// CHECK:STDOUT:     %assoc0: %K.assoc_type = assoc_entity element0, @K.%X [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .X = @X.%assoc0
// CHECK:STDOUT:   witness = (%X)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
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
// CHECK:STDOUT: specific @K(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- incomplete_where.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Incomplete.type: type = facet_type <@Incomplete> [concrete]
// CHECK:STDOUT:   %.Self: %I.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self [symbolic_self]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I where .Self impls @Incomplete> [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @C.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C, (%I.impl_witness) [concrete]
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
// CHECK:STDOUT:     %.Self: %I.type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %I.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %Incomplete.ref: type = name_ref Incomplete, file.%Incomplete.decl [concrete = constants.%Incomplete.type]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc8_19: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc8_13: type = where_expr %.Self [concrete = constants.%I_where.type] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%I.type
// CHECK:STDOUT:       requirement_impls %.loc8_19, %Incomplete.ref
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   interface_with_self_decl @I [concrete]
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
// CHECK:STDOUT: impl @C.as.I.impl: %C.ref as %.loc8_13 {
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (), @C.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = %I.impl_witness
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
// CHECK:STDOUT: specific @I(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%I.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- declaration_incomplete_where_rewrite.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self: %J.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type @J [concrete]
// CHECK:STDOUT:   %assoc0: %J.assoc_type = assoc_entity element0, @J.%T [concrete]
// CHECK:STDOUT:   %Incomplete.type: type = facet_type <@Incomplete> [concrete]
// CHECK:STDOUT:   %.Self: %J.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self [symbolic_self]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %J.lookup_impl_witness: <witness> = lookup_impl_witness %.Self, @J [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %J.lookup_impl_witness, element0 [symbolic_self]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %J_where.type: type = facet_type <@J where .Self impls @Incomplete and %impl.elem0 = %empty_tuple.type> [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness @C.as.J.impl.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %J.facet: %J.type = facet_value %C, (%J.impl_witness) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .Incomplete = %Incomplete.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [concrete = constants.%J.type] {} {}
// CHECK:STDOUT:   %Incomplete.decl: type = interface_decl @Incomplete [concrete = constants.%Incomplete.type] {} {}
// CHECK:STDOUT:   impl_decl @C.as.J.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref.loc8: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %J.ref.loc8: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %.Self.2: %J.type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc8_19: %J.type = name_ref .Self, %.Self.2 [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %Incomplete.ref.loc8: type = name_ref Incomplete, file.%Incomplete.decl [concrete = constants.%Incomplete.type]
// CHECK:STDOUT:     %.Self.as_type.loc8_19: type = facet_access_type %.Self.ref.loc8_19 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc8_19: type = converted %.Self.ref.loc8_19, %.Self.as_type.loc8_19 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.Self.ref.loc8_46: %J.type = name_ref .Self, %.Self.2 [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.as_type.loc8_46: type = facet_access_type %.Self.ref.loc8_46 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc8_46: type = converted %.Self.ref.loc8_46, %.Self.as_type.loc8_46 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %T.ref.loc8: %J.assoc_type = name_ref T, @T.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc8: type = impl_witness_access constants.%J.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc8_52.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc8_52.2: type = converted %.loc8_52.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc8_13: type = where_expr %.Self.2 [concrete = constants.%J_where.type] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%J.type
// CHECK:STDOUT:       requirement_impls %.loc8_19, %Incomplete.ref.loc8
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc8, %.loc8_52.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.J.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref.loc10: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %J.ref.loc10: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %.Self.1: %J.type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc10_19: %J.type = name_ref .Self, %.Self.1 [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %Incomplete.ref.loc10: type = name_ref Incomplete, file.%Incomplete.decl [concrete = constants.%Incomplete.type]
// CHECK:STDOUT:     %.Self.as_type.loc10_19: type = facet_access_type %.Self.ref.loc10_19 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc10_19: type = converted %.Self.ref.loc10_19, %.Self.as_type.loc10_19 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.Self.ref.loc10_46: %J.type = name_ref .Self, %.Self.1 [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.as_type.loc10_46: type = facet_access_type %.Self.ref.loc10_46 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc10_46: type = converted %.Self.ref.loc10_46, %.Self.as_type.loc10_46 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %T.ref.loc10: %J.assoc_type = name_ref T, @T.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc10: type = impl_witness_access constants.%J.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc10_52.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc10_52.2: type = converted %.loc10_52.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc10_13: type = where_expr %.Self.1 [concrete = constants.%J_where.type] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%J.type
// CHECK:STDOUT:       requirement_impls %.loc10_19, %Incomplete.ref.loc10
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc10, %.loc10_52.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   interface_with_self_decl @J [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %T: type = assoc_const_decl @T [concrete] {
// CHECK:STDOUT:     %assoc0: %J.assoc_type = assoc_entity element0, @J.%T [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .T = @T.%assoc0
// CHECK:STDOUT:   witness = (%T)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Incomplete;
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.J.impl: %C.ref.loc8 as %.loc8_13 {
// CHECK:STDOUT:   %J.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @C.as.J.impl [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness]
// CHECK:STDOUT:   %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = %J.impl_witness
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
// CHECK:STDOUT: specific @J(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%.Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%J.facet) {
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
// CHECK:STDOUT:   %.Self: %I.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self [symbolic_self]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I where .Self impls @Incomplete> [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @C.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %C.as.I.impl.F.type: type = fn_type @C.as.I.impl.F [concrete]
// CHECK:STDOUT:   %C.as.I.impl.F: %C.as.I.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C, (%I.impl_witness) [concrete]
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
// CHECK:STDOUT:     %.Self: %I.type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %I.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %Incomplete.ref: type = name_ref Incomplete, file.%Incomplete.decl [concrete = constants.%Incomplete.type]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc8_19: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc8_13: type = where_expr %.Self [concrete = constants.%I_where.type] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%I.type
// CHECK:STDOUT:       requirement_impls %.loc8_19, %Incomplete.ref
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   interface_with_self_decl @I [concrete]
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
// CHECK:STDOUT: impl @C.as.I.impl: %C.ref as %.loc8_13 {
// CHECK:STDOUT:   %C.as.I.impl.F.decl: %C.as.I.impl.F.type = fn_decl @C.as.I.impl.F [concrete = constants.%C.as.I.impl.F] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.I.impl.%C.ref [concrete = constants.%C]
// CHECK:STDOUT:     %Incomplete.ref: type = name_ref Incomplete, file.%Incomplete.decl [concrete = constants.%Incomplete.type]
// CHECK:STDOUT:     %T.ref: <error> = name_ref T, <error> [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (), @C.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Incomplete = <poisoned>
// CHECK:STDOUT:   .F = %C.as.I.impl.F.decl
// CHECK:STDOUT:   witness = %I.impl_witness
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
// CHECK:STDOUT: specific @I(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%I.facet) {
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
// CHECK:STDOUT: impl @C.as.<error>.impl: %C.ref as %X.ref;
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
// CHECK:STDOUT:   %Self.550: %Y.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.5cb: type = symbolic_binding_type Self, 0, %Self.550 [symbolic]
// CHECK:STDOUT:   %X.type: type = facet_type <@X> [concrete]
// CHECK:STDOUT:   %Self.e52: %X.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %X.assoc_type: type = assoc_entity_type @X [concrete]
// CHECK:STDOUT:   %assoc0: %X.assoc_type = assoc_entity element0, @X.%X1 [concrete]
// CHECK:STDOUT:   %Self.binding.as_type.806: type = symbolic_binding_type Self, 0, %Self.e52 [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.c39: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.637: %X.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self.637 [symbolic_self]
// CHECK:STDOUT:   %X.lookup_impl_witness: <witness> = lookup_impl_witness %.Self.637, @X [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %X.lookup_impl_witness, element0 [symbolic_self]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %X_where.type: type = facet_type <@X where %impl.elem0 = %empty_struct_type> [concrete]
// CHECK:STDOUT:   %T: %X_where.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %X_where.type [concrete]
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
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc16_12.1: type = splice_block %.loc16_12.2 [concrete = constants.%X_where.type] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X.type]
// CHECK:STDOUT:       %.Self.2: %X.type = symbolic_binding .Self [symbolic_self = constants.%.Self.637]
// CHECK:STDOUT:       %.Self.ref: %X.type = name_ref .Self, %.Self.2 [symbolic_self = constants.%.Self.637]
// CHECK:STDOUT:       %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:       %.loc16_18: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:       %X1.ref: %X.assoc_type = name_ref X1, @X1.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:       %impl.elem0: type = impl_witness_access constants.%X.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:       %.loc16_25.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:       %.loc16_25.2: type = converted %.loc16_25.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:       %.loc16_12.2: type = where_expr %.Self.2 [concrete = constants.%X_where.type] {
// CHECK:STDOUT:         requirement_base_facet_type constants.%X.type
// CHECK:STDOUT:         requirement_rewrite %impl.elem0, %.loc16_25.2
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc16_6.2: %X_where.type = symbolic_binding T, 0 [symbolic = %T.loc16_6.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Z;
// CHECK:STDOUT:
// CHECK:STDOUT: interface @X {
// CHECK:STDOUT:   %Self: %X.type = symbolic_binding Self, 0 [symbolic = constants.%Self.e52]
// CHECK:STDOUT:   interface_with_self_decl @X [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %X1: type = assoc_const_decl @X1 [concrete] {
// CHECK:STDOUT:     %assoc0: %X.assoc_type = assoc_entity element0, @X.%X1 [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.WithSelf.Self.binding.as_type.impls.Y.type.require.decl = require_decl @X.WithSelf.Self.binding.as_type.impls.Y.type.require [concrete] {
// CHECK:STDOUT:     require %Self.as_type impls %Y.ref
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.as_type: type = facet_access_type @X.%Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.806)]
// CHECK:STDOUT:     %Y.ref: type = name_ref Y, file.%Y.decl [concrete = constants.%Y.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Y = <poisoned>
// CHECK:STDOUT:   .X1 = @X1.%assoc0
// CHECK:STDOUT:   .Y = <poisoned>
// CHECK:STDOUT:   witness = (%X1)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:   @X.WithSelf.Self.binding.as_type.impls.Y.type.require {
// CHECK:STDOUT:     require @X.WithSelf.Self.binding.as_type.impls.Y.type.require.%Self.as_type impls @X.WithSelf.Self.binding.as_type.impls.Y.type.require.%Y.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: constraint @Y {
// CHECK:STDOUT:   %Self: %Y.type = symbolic_binding Self, 0 [symbolic = constants.%Self.550]
// CHECK:STDOUT:   constraint_with_self_decl @Y [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %Y.WithSelf.Self.binding.as_type.impls.Z.type.require.decl = require_decl @Y.WithSelf.Self.binding.as_type.impls.Z.type.require [concrete] {
// CHECK:STDOUT:     require %Self.as_type impls %Z.ref
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.as_type: type = facet_access_type @Y.%Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.5cb)]
// CHECK:STDOUT:     %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Z = <poisoned>
// CHECK:STDOUT:   .Z = <poisoned>
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:   @Y.WithSelf.Self.binding.as_type.impls.Z.type.require {
// CHECK:STDOUT:     require @Y.WithSelf.Self.binding.as_type.impls.Z.type.require.%Self.as_type impls @Y.WithSelf.Self.binding.as_type.impls.Z.type.require.%Z.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @Y.WithSelf.Self.binding.as_type.impls.Z.type.require(@Y.%Self: %Y.type) {
// CHECK:STDOUT:   %Self: %Y.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.550)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.5cb)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @X.WithSelf.Self.binding.as_type.impls.Y.type.require(@X.%Self: %X.type) {
// CHECK:STDOUT:   %Self: %X.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.e52)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.806)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc16_6.2: %X_where.type) {
// CHECK:STDOUT:   %T.loc16_6.1: %X_where.type = symbolic_binding T, 0 [symbolic = %T.loc16_6.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y(constants.%Self.550) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y.WithSelf.Self.binding.as_type.impls.Z.type.require(constants.%Self.550) {
// CHECK:STDOUT:   %Self => constants.%Self.550
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.5cb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%Self.e52) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y(constants.%Self.e52) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y.WithSelf.Self.binding.as_type.impls.Z.type.require(constants.%Self.e52) {
// CHECK:STDOUT:   %Self => constants.%Self.e52
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.806
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X.WithSelf.Self.binding.as_type.impls.Y.type.require(constants.%Self.e52) {
// CHECK:STDOUT:   %Self => constants.%Self.e52
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.806
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%.Self.637) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc16_6.1 => constants.%T
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
// CHECK:STDOUT:   interface_with_self_decl @B [concrete]
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
// CHECK:STDOUT: specific @B(constants.%Self) {}
// CHECK:STDOUT:

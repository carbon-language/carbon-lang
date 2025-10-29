// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/where_expr/constraints.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/where_expr/constraints.carbon

// --- self_impls_type.carbon

library "[[@TEST_NAME]]";

interface I {}

//@dump-sem-ir-begin
fn F(T:! I where .Self impls type);
//@dump-sem-ir-end

alias Type = type;

//@dump-sem-ir-begin
fn G(T:! I where .Self impls Type);
//@dump-sem-ir-end

// --- fail_self_impls_error.carbon

library "[[@TEST_NAME]]";

interface I {}

//@dump-sem-ir-begin
// Because there's an error inside the `where`, the constant value of the
// `where` should be an error also.
//
// CHECK:STDERR: fail_self_impls_error.carbon:[[@LINE+4]]:30: error: name `J` not found [NameNotFound]
// CHECK:STDERR: fn F(T:! I where .Self impls J);
// CHECK:STDERR:                              ^
// CHECK:STDERR:
fn F(T:! I where .Self impls J);
//@dump-sem-ir-end

// --- state_constraints.carbon

library "[[@TEST_NAME]]";

interface J {}

interface I {
  let Member:! type;
  let Second:! J;
}

//@dump-sem-ir-begin
fn EqualEqual(U:! I where .Self == ());

fn Impls(V:! J where .Self impls I);

fn And(W:! I where .Self impls J and .Member == ());
//@dump-sem-ir-end

// --- associated_type_impls.carbon

library "[[@TEST_NAME]]";

interface L {}
interface M {}

interface K {
  let Associated:! L;
}

//@dump-sem-ir-begin
fn AssociatedTypeImpls(W:! K where .Associated impls M);
//@dump-sem-ir-end

// --- fail_left_of_impls_non_type.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_left_of_impls_non_type.carbon:[[@LINE+7]]:32: error: cannot implicitly convert non-type value of type `Core.IntLiteral` to `type` [ConversionFailureNonTypeToFacet]
// CHECK:STDERR: fn NonTypeImpls(U:! type where 7 impls type);
// CHECK:STDERR:                                ^
// CHECK:STDERR: fail_left_of_impls_non_type.carbon:[[@LINE+4]]:32: note: type `Core.IntLiteral` does not implement interface `Core.ImplicitAs(type)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: fn NonTypeImpls(U:! type where 7 impls type);
// CHECK:STDERR:                                ^
// CHECK:STDERR:
fn NonTypeImpls(U:! type where 7 impls type);

// --- fail_right_of_impls_non_type.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_right_of_impls_non_type.carbon:[[@LINE+7]]:44: error: cannot implicitly convert non-type value of type `Core.IntLiteral` to `type` [ConversionFailureNonTypeToFacet]
// CHECK:STDERR: fn ImplsNonType(U:! type where .Self impls 7);
// CHECK:STDERR:                                            ^
// CHECK:STDERR: fail_right_of_impls_non_type.carbon:[[@LINE+4]]:44: note: type `Core.IntLiteral` does not implement interface `Core.ImplicitAs(type)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: fn ImplsNonType(U:! type where .Self impls 7);
// CHECK:STDERR:                                            ^
// CHECK:STDERR:
fn ImplsNonType(U:! type where .Self impls 7);

// --- fail_right_of_impls_non_facet_type.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_right_of_impls_non_facet_type.carbon:[[@LINE+4]]:51: error: right argument of `impls` requirement must be a facet type [ImplsOnNonFacetType]
// CHECK:STDERR: fn ImplsOfNonFacetType(U:! type where .Self impls i32);
// CHECK:STDERR:                                                   ^~~
// CHECK:STDERR:
fn ImplsOfNonFacetType(U:! type where .Self impls i32);

// --- fail_enforce_impls_constraint.carbon

library "[[@TEST_NAME]]";

import library "state_constraints";

// C implements J but not I.
class C {}
impl C as J {}

fn DoesNotImplI() {
  // CHECK:STDERR: fail_enforce_impls_constraint.carbon:[[@LINE+8]]:3: error: cannot convert type `C` into type implementing `J where .Self impls I` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   Impls(C);
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR: fail_enforce_impls_constraint.carbon:[[@LINE-10]]:1: in import [InImport]
  // CHECK:STDERR: state_constraints.carbon:14:10: note: initializing generic parameter `V` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn Impls(V:! J where .Self impls I);
  // CHECK:STDERR:          ^
  // CHECK:STDERR:
  Impls(C);
}

// --- fail_todo_enforce_equality_constraint.carbon

library "[[@TEST_NAME]]";

import library "state_constraints";

class C {}
impl C as J {}

fn EmptyStruct(Y:! J where .Self == {});

// TODO: Should report that `C` does not meeet the `==` constraint. Right
// now there is no support for `==` constraints so it rejects it as an
// "other currently unsupported constraint."
fn NotEmptyStruct() {
  // CHECK:STDERR: fail_todo_enforce_equality_constraint.carbon:[[@LINE+7]]:3: error: cannot convert type `C` into type implementing `J where...` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   EmptyStruct(C);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_enforce_equality_constraint.carbon:[[@LINE-9]]:16: note: initializing generic parameter `Y` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn EmptyStruct(Y:! J where .Self == {});
  // CHECK:STDERR:                ^
  // CHECK:STDERR:
  EmptyStruct(C);
}

// --- fail_error_in_constraint.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
// CHECK:STDERR: fail_error_in_constraint.carbon:[[@LINE+4]]:41: error: name `I` not found [NameNotFound]
// CHECK:STDERR: fn WithError(U:! type where .Self impls I);
// CHECK:STDERR:                                         ^
// CHECK:STDERR:
fn WithError(U:! type where .Self impls I);
//@dump-sem-ir-end

// --- import_error_in_constraint.carbon

library "[[@TEST_NAME]]";

import library "error_in_constraint";

fn F() {
//@dump-sem-ir-begin
  let x: () = WithError(());
//@dump-sem-ir-end
}

// CHECK:STDOUT: --- self_impls_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %.Self.420: %I.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self.420 [symbolic_self]
// CHECK:STDOUT:   %T: %I.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %I.type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc7_12.1: type = splice_block %.loc7_12.2 [concrete = constants.%I.type] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %.Self.ref: %I.type = name_ref .Self, %.Self.2 [symbolic_self = constants.%.Self.420]
// CHECK:STDOUT:       %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:       %.loc7_18: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:       %.loc7_12.2: type = where_expr %.Self.2 [concrete = constants.%I.type] {
// CHECK:STDOUT:         requirement_base_facet_type constants.%I.type
// CHECK:STDOUT:         requirement_impls %.loc7_18, type
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc7_6.2: %I.type = symbolic_binding T, 0 [symbolic = %T.loc7_6.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc13_12.1: type = splice_block %.loc13_12.2 [concrete = constants.%I.type] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %.Self.ref: %I.type = name_ref .Self, %.Self.2 [symbolic_self = constants.%.Self.420]
// CHECK:STDOUT:       %Type.ref: type = name_ref Type, file.%Type [concrete = type]
// CHECK:STDOUT:       %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:       %.loc13_18: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:       %.loc13_12.2: type = where_expr %.Self.2 [concrete = constants.%I.type] {
// CHECK:STDOUT:         requirement_base_facet_type constants.%I.type
// CHECK:STDOUT:         requirement_impls %.loc13_18, %Type.ref
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc13_6.2: %I.type = symbolic_binding T, 0 [symbolic = %T.loc13_6.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc7_6.2: %I.type) {
// CHECK:STDOUT:   %T.loc7_6.1: %I.type = symbolic_binding T, 0 [symbolic = %T.loc7_6.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%T.loc13_6.2: %I.type) {
// CHECK:STDOUT:   %T.loc13_6.1: %I.type = symbolic_binding T, 0 [symbolic = %T.loc13_6.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc7_6.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%T) {
// CHECK:STDOUT:   %T.loc13_6.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_self_impls_error.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %.Self.420: %I.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self.420 [symbolic_self]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: <error> = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc14_12.1: type = splice_block %.loc14_12.2 [concrete = <error>] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %.Self.ref: %I.type = name_ref .Self, %.Self.2 [symbolic_self = constants.%.Self.420]
// CHECK:STDOUT:       %J.ref: <error> = name_ref J, <error> [concrete = <error>]
// CHECK:STDOUT:       %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:       %.loc14_18: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:       %.loc14_12.2: type = where_expr %.Self.2 [concrete = <error>] {
// CHECK:STDOUT:         requirement_base_facet_type constants.%I.type
// CHECK:STDOUT:         requirement_impls %.loc14_18, <error>
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T: <error> = symbolic_binding T, 0 [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T: <error>) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(<error>) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- state_constraints.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.%Member [concrete]
// CHECK:STDOUT:   %.Self.420: %I.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %I_where.type.a09: type = facet_type <@I where TODO> [concrete]
// CHECK:STDOUT:   %U: %I_where.type.a09 = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.917: type = pattern_type %I_where.type.a09 [concrete]
// CHECK:STDOUT:   %EqualEqual.type: type = fn_type @EqualEqual [concrete]
// CHECK:STDOUT:   %EqualEqual: %EqualEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %.Self.c58: %J.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.binding.as_type.338: type = symbolic_binding_type .Self, %.Self.c58 [symbolic_self]
// CHECK:STDOUT:   %J_where.type: type = facet_type <@J where .Self impls @I> [concrete]
// CHECK:STDOUT:   %V: %J_where.type = symbolic_binding V, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.74f: type = pattern_type %J_where.type [concrete]
// CHECK:STDOUT:   %Impls.type: type = fn_type @Impls [concrete]
// CHECK:STDOUT:   %Impls: %Impls.type = struct_value () [concrete]
// CHECK:STDOUT:   %.Self.binding.as_type.765: type = symbolic_binding_type .Self, %.Self.420 [symbolic_self]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %.Self.420, @I [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %I.lookup_impl_witness, element0 [symbolic_self]
// CHECK:STDOUT:   %I_where.type.e04: type = facet_type <@I where .Self impls @J and TODO> [concrete]
// CHECK:STDOUT:   %W: %I_where.type.e04 = symbolic_binding W, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.194: type = pattern_type %I_where.type.e04 [concrete]
// CHECK:STDOUT:   %And.type: type = fn_type @And [concrete]
// CHECK:STDOUT:   %And: %And.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %EqualEqual.decl: %EqualEqual.type = fn_decl @EqualEqual [concrete = constants.%EqualEqual] {
// CHECK:STDOUT:     %U.patt: %pattern_type.917 = symbolic_binding_pattern U, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc12_21.1: type = splice_block %.loc12_21.2 [concrete = constants.%I_where.type.a09] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %.Self.ref: %I.type = name_ref .Self, %.Self.2 [symbolic_self = constants.%.Self.420]
// CHECK:STDOUT:       %.loc12_37: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc12_21.2: type = where_expr %.Self.2 [concrete = constants.%I_where.type.a09] {
// CHECK:STDOUT:         requirement_base_facet_type constants.%I.type
// CHECK:STDOUT:         requirement_equivalent %.Self.ref, %.loc12_37
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc12_15.2: %I_where.type.a09 = symbolic_binding U, 0 [symbolic = %U.loc12_15.1 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Impls.decl: %Impls.type = fn_decl @Impls [concrete = constants.%Impls] {
// CHECK:STDOUT:     %V.patt: %pattern_type.74f = symbolic_binding_pattern V, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc14_16.1: type = splice_block %.loc14_16.2 [concrete = constants.%J_where.type] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %.Self.ref: %J.type = name_ref .Self, %.Self.2 [symbolic_self = constants.%.Self.c58]
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:       %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type.338]
// CHECK:STDOUT:       %.loc14_22: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type.338]
// CHECK:STDOUT:       %.loc14_16.2: type = where_expr %.Self.2 [concrete = constants.%J_where.type] {
// CHECK:STDOUT:         requirement_base_facet_type constants.%J.type
// CHECK:STDOUT:         requirement_impls %.loc14_22, %I.ref
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %V.loc14_10.2: %J_where.type = symbolic_binding V, 0 [symbolic = %V.loc14_10.1 (constants.%V)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %And.decl: %And.type = fn_decl @And [concrete = constants.%And] {
// CHECK:STDOUT:     %W.patt: %pattern_type.194 = symbolic_binding_pattern W, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc16_14.1: type = splice_block %.loc16_14.2 [concrete = constants.%I_where.type.e04] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %.Self.ref.loc16_20: %I.type = name_ref .Self, %.Self.2 [symbolic_self = constants.%.Self.420]
// CHECK:STDOUT:       %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:       %.Self.as_type.loc16_20: type = facet_access_type %.Self.ref.loc16_20 [symbolic_self = constants.%.Self.binding.as_type.765]
// CHECK:STDOUT:       %.loc16_20: type = converted %.Self.ref.loc16_20, %.Self.as_type.loc16_20 [symbolic_self = constants.%.Self.binding.as_type.765]
// CHECK:STDOUT:       %.Self.ref.loc16_38: %I.type = name_ref .Self, %.Self.2 [symbolic_self = constants.%.Self.420]
// CHECK:STDOUT:       %Member.ref: %I.assoc_type = name_ref Member, @Member.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:       %.Self.as_type.loc16_38: type = facet_access_type %.Self.ref.loc16_38 [symbolic_self = constants.%.Self.binding.as_type.765]
// CHECK:STDOUT:       %.loc16_38: type = converted %.Self.ref.loc16_38, %.Self.as_type.loc16_38 [symbolic_self = constants.%.Self.binding.as_type.765]
// CHECK:STDOUT:       %impl.elem0: type = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:       %.loc16_50: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc16_14.2: type = where_expr %.Self.2 [concrete = constants.%I_where.type.e04] {
// CHECK:STDOUT:         requirement_base_facet_type constants.%I.type
// CHECK:STDOUT:         requirement_impls %.loc16_20, %J.ref
// CHECK:STDOUT:         requirement_equivalent %impl.elem0, %.loc16_50
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %W.loc16_8.2: %I_where.type.e04 = symbolic_binding W, 0 [symbolic = %W.loc16_8.1 (constants.%W)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @EqualEqual(%U.loc12_15.2: %I_where.type.a09) {
// CHECK:STDOUT:   %U.loc12_15.1: %I_where.type.a09 = symbolic_binding U, 0 [symbolic = %U.loc12_15.1 (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Impls(%V.loc14_10.2: %J_where.type) {
// CHECK:STDOUT:   %V.loc14_10.1: %J_where.type = symbolic_binding V, 0 [symbolic = %V.loc14_10.1 (constants.%V)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @And(%W.loc16_8.2: %I_where.type.e04) {
// CHECK:STDOUT:   %W.loc16_8.1: %I_where.type.e04 = symbolic_binding W, 0 [symbolic = %W.loc16_8.1 (constants.%W)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @EqualEqual(constants.%U) {
// CHECK:STDOUT:   %U.loc12_15.1 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Impls(constants.%V) {
// CHECK:STDOUT:   %V.loc14_10.1 => constants.%V
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @And(constants.%W) {
// CHECK:STDOUT:   %W.loc16_8.1 => constants.%W
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- associated_type_impls.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %L.type: type = facet_type <@L> [concrete]
// CHECK:STDOUT:   %M.type: type = facet_type <@M> [concrete]
// CHECK:STDOUT:   %K.type: type = facet_type <@K> [concrete]
// CHECK:STDOUT:   %K.assoc_type: type = assoc_entity_type @K [concrete]
// CHECK:STDOUT:   %assoc0: %K.assoc_type = assoc_entity element0, @K.%Associated [concrete]
// CHECK:STDOUT:   %.Self.a80: %K.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self.a80 [symbolic_self]
// CHECK:STDOUT:   %K.lookup_impl_witness: <witness> = lookup_impl_witness %.Self.a80, @K [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: %L.type = impl_witness_access %K.lookup_impl_witness, element0 [symbolic_self]
// CHECK:STDOUT:   %as_type: type = facet_access_type %impl.elem0 [symbolic_self]
// CHECK:STDOUT:   %K_where.type: type = facet_type <@K where TODO> [concrete]
// CHECK:STDOUT:   %W: %K_where.type = symbolic_binding W, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %K_where.type [concrete]
// CHECK:STDOUT:   %AssociatedTypeImpls.type: type = fn_type @AssociatedTypeImpls [concrete]
// CHECK:STDOUT:   %AssociatedTypeImpls: %AssociatedTypeImpls.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %AssociatedTypeImpls.decl: %AssociatedTypeImpls.type = fn_decl @AssociatedTypeImpls [concrete = constants.%AssociatedTypeImpls] {
// CHECK:STDOUT:     %W.patt: %pattern_type = symbolic_binding_pattern W, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc12_30.1: type = splice_block %.loc12_30.2 [concrete = constants.%K_where.type] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %K.ref: type = name_ref K, file.%K.decl [concrete = constants.%K.type]
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %.Self.ref: %K.type = name_ref .Self, %.Self.2 [symbolic_self = constants.%.Self.a80]
// CHECK:STDOUT:       %Associated.ref: %K.assoc_type = name_ref Associated, @Associated.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:       %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:       %.loc12_36.1: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:       %impl.elem0: %L.type = impl_witness_access constants.%K.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:       %M.ref: type = name_ref M, file.%M.decl [concrete = constants.%M.type]
// CHECK:STDOUT:       %as_type: type = facet_access_type %impl.elem0 [symbolic_self = constants.%as_type]
// CHECK:STDOUT:       %.loc12_36.2: type = converted %impl.elem0, %as_type [symbolic_self = constants.%as_type]
// CHECK:STDOUT:       %.loc12_30.2: type = where_expr %.Self.2 [concrete = constants.%K_where.type] {
// CHECK:STDOUT:         requirement_base_facet_type constants.%K.type
// CHECK:STDOUT:         requirement_impls %.loc12_36.2, %M.ref
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %W.loc12_24.2: %K_where.type = symbolic_binding W, 0 [symbolic = %W.loc12_24.1 (constants.%W)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AssociatedTypeImpls(%W.loc12_24.2: %K_where.type) {
// CHECK:STDOUT:   %W.loc12_24.1: %K_where.type = symbolic_binding W, 0 [symbolic = %W.loc12_24.1 (constants.%W)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AssociatedTypeImpls(constants.%W) {
// CHECK:STDOUT:   %W.loc12_24.1 => constants.%W
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_error_in_constraint.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.Self.bcc: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %WithError.type: type = fn_type @WithError [concrete]
// CHECK:STDOUT:   %WithError: %WithError.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %WithError.decl: %WithError.type = fn_decl @WithError [concrete = constants.%WithError] {
// CHECK:STDOUT:     %U.patt: <error> = symbolic_binding_pattern U, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc9_23.1: type = splice_block %.loc9_23.2 [concrete = <error>] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %.Self.ref: type = name_ref .Self, %.Self.2 [symbolic_self = constants.%.Self.bcc]
// CHECK:STDOUT:       %I.ref: <error> = name_ref I, <error> [concrete = <error>]
// CHECK:STDOUT:       %.loc9_23.2: type = where_expr %.Self.2 [concrete = <error>] {
// CHECK:STDOUT:         requirement_base_facet_type type
// CHECK:STDOUT:         requirement_impls %.Self.ref, <error>
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U: <error> = symbolic_binding U, 0 [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @WithError(%U: <error>) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WithError(<error>) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_error_in_constraint.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %WithError.type: type = fn_type @WithError [concrete]
// CHECK:STDOUT:   %WithError: %WithError.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.WithError: %WithError.type = import_ref Main//error_in_constraint, WithError, loaded [concrete = constants.%WithError]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type = value_binding_pattern x [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %WithError.ref: %WithError.type = name_ref WithError, imports.%Main.WithError [concrete = constants.%WithError]
// CHECK:STDOUT:   %.loc8_26: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc8_11.1: type = splice_block %.loc8_11.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc8_11.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc8_11.3: type = converted %.loc8_11.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: %empty_tuple.type = value_binding x, <error> [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:

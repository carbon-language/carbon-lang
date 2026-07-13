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
fn F(generic T: I where .Self impls type);
//@dump-sem-ir-end

alias Type = type;

//@dump-sem-ir-begin
fn G(generic T: I where .Self impls Type);
//@dump-sem-ir-end

// --- fail_self_impls_error.carbon

library "[[@TEST_NAME]]";

interface I {}

//@dump-sem-ir-begin
// Because there's an error inside the `where`, the constant value of the
// `where` should be an error also.
//
// CHECK:STDERR: fail_self_impls_error.carbon:[[@LINE+4]]:37: error: name `J` not found [NameNotFound]
// CHECK:STDERR: fn F(generic T: I where .Self impls J);
// CHECK:STDERR:                                     ^
// CHECK:STDERR:
fn F(generic T: I where .Self impls J);
//@dump-sem-ir-end

// --- state_constraints.carbon

library "[[@TEST_NAME]]";

interface J {}

interface I {
  let Member: type;
  let Second: J;
}

//@dump-sem-ir-begin
fn EqualEqual(generic U: I where .Self == ());

fn Impls(generic V: J where .Self impls I);

fn And(generic W: I where .Self impls J and .Member == ());
//@dump-sem-ir-end

// --- associated_type_impls.carbon

library "[[@TEST_NAME]]";

interface L {}
interface M {}

interface K {
  let Associated: L;
}

//@dump-sem-ir-begin
fn AssociatedTypeImpls(generic W: K where .Associated impls M);
//@dump-sem-ir-end

// --- fail_left_of_impls_non_type.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_left_of_impls_non_type.carbon:[[@LINE+7]]:39: error: cannot implicitly convert non-type value of type `Core.IntLiteral` to `type` [ConversionFailureNonTypeToFacet]
// CHECK:STDERR: fn NonTypeImpls(generic U: type where 7 impls type);
// CHECK:STDERR:                                       ^
// CHECK:STDERR: fail_left_of_impls_non_type.carbon:[[@LINE+4]]:39: note: type `Core.IntLiteral` does not implement interface `Core.ImplicitAs(type)` [MissingImplInMemberAccessInContext]
// CHECK:STDERR: fn NonTypeImpls(generic U: type where 7 impls type);
// CHECK:STDERR:                                       ^
// CHECK:STDERR:
fn NonTypeImpls(generic U: type where 7 impls type);

// --- fail_right_of_impls_non_type.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_right_of_impls_non_type.carbon:[[@LINE+7]]:51: error: cannot implicitly convert non-type value of type `Core.IntLiteral` to `type` [ConversionFailureNonTypeToFacet]
// CHECK:STDERR: fn ImplsNonType(generic U: type where .Self impls 7);
// CHECK:STDERR:                                                   ^
// CHECK:STDERR: fail_right_of_impls_non_type.carbon:[[@LINE+4]]:51: note: type `Core.IntLiteral` does not implement interface `Core.ImplicitAs(type)` [MissingImplInMemberAccessInContext]
// CHECK:STDERR: fn ImplsNonType(generic U: type where .Self impls 7);
// CHECK:STDERR:                                                   ^
// CHECK:STDERR:
fn ImplsNonType(generic U: type where .Self impls 7);

// --- fail_right_of_impls_non_facet_type.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_right_of_impls_non_facet_type.carbon:[[@LINE+4]]:58: error: right argument of `impls` requirement must be a facet type [ImplsOnNonFacetType]
// CHECK:STDERR: fn ImplsOfNonFacetType(generic U: type where .Self impls i32);
// CHECK:STDERR:                                                          ^~~
// CHECK:STDERR:
fn ImplsOfNonFacetType(generic U: type where .Self impls i32);

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
  // CHECK:STDERR: state_constraints.carbon:14:18: note: initializing generic parameter `V` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn Impls(generic V: J where .Self impls I);
  // CHECK:STDERR:                  ^~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Impls(C);
}

// --- fail_todo_enforce_equality_constraint.carbon

library "[[@TEST_NAME]]";

import library "state_constraints";

class C {}
impl C as J {}

fn EmptyStruct(generic Y: J where .Self == {});

// TODO: Should report that `C` does not meeet the `==` constraint. Right
// now there is no support for `==` constraints so it rejects it as an
// "other currently unsupported constraint."
fn NotEmptyStruct() {
  // CHECK:STDERR: fail_todo_enforce_equality_constraint.carbon:[[@LINE+7]]:3: error: cannot convert type `C` into type implementing `J where...` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   EmptyStruct(C);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_enforce_equality_constraint.carbon:[[@LINE-9]]:24: note: initializing generic parameter `Y` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn EmptyStruct(generic Y: J where .Self == {});
  // CHECK:STDERR:                        ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  EmptyStruct(C);
}

// --- fail_error_in_constraint.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
// CHECK:STDERR: fail_error_in_constraint.carbon:[[@LINE+4]]:48: error: name `I` not found [NameNotFound]
// CHECK:STDERR: fn WithError(generic U: type where .Self impls I);
// CHECK:STDERR:                                                ^
// CHECK:STDERR:
fn WithError(generic U: type where .Self impls I);
//@dump-sem-ir-end

// --- import_error_in_constraint.carbon

library "[[@TEST_NAME]]";

import library "error_in_constraint";

fn F() {
//@dump-sem-ir-begin
  let unused x: () = WithError(());
//@dump-sem-ir-end
}

// CHECK:STDOUT: --- self_impls_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen.197: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.frozen.3a0: %I.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.frozen.as_type: type = facet_access_type %.Self.frozen.3a0 [symbolic_self]
// CHECK:STDOUT:   %.Self: %I.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %I.type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: %I.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc7_15.1: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc7_19.1: type = splice_block %.loc7_19.2 [concrete = constants.%I.type] {
// CHECK:STDOUT:       %.Self.frozen.loc7_15: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.197]
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:       %.Self.frozen.loc7_19: %I.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.3a0]
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %I.ref [concrete]
// CHECK:STDOUT:       %.Self.ref.loc7_25.1: %I.type = name_ref .Self, %.Self.frozen.loc7_19 [symbolic_self = constants.%.Self.frozen.3a0]
// CHECK:STDOUT:       %.loc7_37: type = type_literal type [concrete = type]
// CHECK:STDOUT:       %.Self.as_type.loc7_25.1: type = facet_access_type %.Self.ref.loc7_25.1 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:       %.loc7_25.1: type = converted %.Self.ref.loc7_25.1, %.Self.as_type.loc7_25.1 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:       %impls.loc7_31.1 = requirement_impls %.loc7_25.1, %.loc7_37 [concrete]
// CHECK:STDOUT:       %.Self: %I.type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.Self.ref.loc7_25.2: %I.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.Self.as_type.loc7_25.2: type = facet_access_type %.Self.ref.loc7_25.2 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc7_25.2: type = converted %.Self.ref.loc7_25.1, %.Self.as_type.loc7_25.2 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %impls.loc7_31.2 = requirement_impls %.loc7_25.2, %.loc7_37 [concrete]
// CHECK:STDOUT:       %.loc7_19.2: type = where_expr [concrete = constants.%I.type] {
// CHECK:STDOUT:         %base_facet_type = requirement_base_facet_type %I.ref [concrete]
// CHECK:STDOUT:         %impls.loc7_31.2 = requirement_impls %.loc7_25.2, %.loc7_37 [concrete]
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc7_15.2: %I.type = symbolic_binding T, 0 [symbolic = %T.loc7_15.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %T.patt.loc13_15.1: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc13_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc13_19.1: type = splice_block %.loc13_19.2 [concrete = constants.%I.type] {
// CHECK:STDOUT:       %.Self.frozen.loc13_15: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.197]
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:       %.Self.frozen.loc13_19: %I.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.3a0]
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %I.ref [concrete]
// CHECK:STDOUT:       %.Self.ref.loc13_25.1: %I.type = name_ref .Self, %.Self.frozen.loc13_19 [symbolic_self = constants.%.Self.frozen.3a0]
// CHECK:STDOUT:       %Type.ref: type = name_ref Type, file.%Type [concrete = type]
// CHECK:STDOUT:       %.Self.as_type.loc13_25.1: type = facet_access_type %.Self.ref.loc13_25.1 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:       %.loc13_25.1: type = converted %.Self.ref.loc13_25.1, %.Self.as_type.loc13_25.1 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:       %impls.loc13_31.1 = requirement_impls %.loc13_25.1, %Type.ref [concrete]
// CHECK:STDOUT:       %.Self: %I.type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.Self.ref.loc13_25.2: %I.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.Self.as_type.loc13_25.2: type = facet_access_type %.Self.ref.loc13_25.2 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc13_25.2: type = converted %.Self.ref.loc13_25.1, %.Self.as_type.loc13_25.2 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %impls.loc13_31.2 = requirement_impls %.loc13_25.2, %Type.ref [concrete]
// CHECK:STDOUT:       %.loc13_19.2: type = where_expr [concrete = constants.%I.type] {
// CHECK:STDOUT:         %base_facet_type = requirement_base_facet_type %I.ref [concrete]
// CHECK:STDOUT:         %impls.loc13_31.2 = requirement_impls %.loc13_25.2, %Type.ref [concrete]
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc13_15.2: %I.type = symbolic_binding T, 0 [symbolic = %T.loc13_15.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc7_15.2: %I.type) {
// CHECK:STDOUT:   %T.patt.loc7_15.2: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc7_15.1: %I.type = symbolic_binding T, 0 [symbolic = %T.loc7_15.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%T.loc13_15.2: %I.type) {
// CHECK:STDOUT:   %T.patt.loc13_15.2: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc13_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc13_15.1: %I.type = symbolic_binding T, 0 [symbolic = %T.loc13_15.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc7_15.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc7_15.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc13_15.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc13_15.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_self_impls_error.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen.197: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.frozen.3a0: %I.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.frozen.as_type: type = facet_access_type %.Self.frozen.3a0 [symbolic_self]
// CHECK:STDOUT:   %.Self: %I.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: <error> = symbolic_binding_pattern T, 0 [concrete = <error>]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc14_19.1: type = splice_block %.loc14_19.2 [concrete = <error>] {
// CHECK:STDOUT:       %.Self.frozen.loc14_15: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.197]
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:       %.Self.frozen.loc14_19: %I.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.3a0]
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %I.ref [concrete]
// CHECK:STDOUT:       %.Self.ref.loc14_25.1: %I.type = name_ref .Self, %.Self.frozen.loc14_19 [symbolic_self = constants.%.Self.frozen.3a0]
// CHECK:STDOUT:       %J.ref: <error> = name_ref J, <error> [concrete = <error>]
// CHECK:STDOUT:       %.Self.as_type.loc14_25.1: type = facet_access_type %.Self.ref.loc14_25.1 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:       %.loc14_25.1: type = converted %.Self.ref.loc14_25.1, %.Self.as_type.loc14_25.1 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:       %impls.loc14_31.1 = requirement_impls %.loc14_25.1, <error> [concrete]
// CHECK:STDOUT:       %.Self: %I.type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.Self.ref.loc14_25.2: %I.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.Self.as_type.loc14_25.2: type = facet_access_type %.Self.ref.loc14_25.2 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc14_25.2: type = converted %.Self.ref.loc14_25.1, %.Self.as_type.loc14_25.2 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %impls.loc14_31.2 = requirement_impls %.loc14_25.2, <error> [concrete]
// CHECK:STDOUT:       %.loc14_19.2: type = where_expr [concrete = <error>] {
// CHECK:STDOUT:         %base_facet_type = requirement_base_facet_type %I.ref [concrete]
// CHECK:STDOUT:         %impls.loc14_31.2 = requirement_impls %.loc14_25.2, <error> [concrete]
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
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.WithSelf.%Member [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen.197: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.frozen.3a0: %I.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %.Self.f73: %I.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %I_where.type.267: type = facet_type <@I where TODO> [concrete]
// CHECK:STDOUT:   %pattern_type.603: type = pattern_type %I_where.type.267 [concrete]
// CHECK:STDOUT:   %U.patt: %pattern_type.603 = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %U: %I_where.type.267 = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %EqualEqual.type: type = fn_type @EqualEqual [concrete]
// CHECK:STDOUT:   %EqualEqual: %EqualEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %.Self.frozen.99c: %J.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.frozen.as_type.8c8: type = facet_access_type %.Self.frozen.99c [symbolic_self]
// CHECK:STDOUT:   %.Self.6ef: %J.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type.8c4: type = facet_access_type %.Self.6ef [symbolic_self]
// CHECK:STDOUT:   %J_where.type: type = facet_type <@J where .Self impls @I> [concrete]
// CHECK:STDOUT:   %pattern_type.17c: type = pattern_type %J_where.type [concrete]
// CHECK:STDOUT:   %V.patt: %pattern_type.17c = symbolic_binding_pattern V, 0 [symbolic]
// CHECK:STDOUT:   %V: %J_where.type = symbolic_binding V, 0 [symbolic]
// CHECK:STDOUT:   %Impls.type: type = fn_type @Impls [concrete]
// CHECK:STDOUT:   %Impls: %Impls.type = struct_value () [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.e90: type = facet_access_type %.Self.frozen.3a0 [symbolic_self]
// CHECK:STDOUT:   %I.lookup_impl_witness.ced: <witness> = lookup_impl_witness %.Self.frozen.3a0, @I [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.125: type = impl_witness_access %I.lookup_impl_witness.ced, element0 [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type.f7e: type = facet_access_type %.Self.f73 [symbolic_self]
// CHECK:STDOUT:   %I.lookup_impl_witness.78b: <witness> = lookup_impl_witness %.Self.f73, @I [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.e55: type = impl_witness_access %I.lookup_impl_witness.78b, element0 [symbolic_self]
// CHECK:STDOUT:   %I_where.type.a1d: type = facet_type <@I where .Self impls @J and TODO> [concrete]
// CHECK:STDOUT:   %pattern_type.e37: type = pattern_type %I_where.type.a1d [concrete]
// CHECK:STDOUT:   %W.patt: %pattern_type.e37 = symbolic_binding_pattern W, 0 [symbolic]
// CHECK:STDOUT:   %W: %I_where.type.a1d = symbolic_binding W, 0 [symbolic]
// CHECK:STDOUT:   %And.type: type = fn_type @And [concrete]
// CHECK:STDOUT:   %And: %And.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %EqualEqual.decl: %EqualEqual.type = fn_decl @EqualEqual [concrete = constants.%EqualEqual] {
// CHECK:STDOUT:     %U.patt.loc12_24.1: %pattern_type.603 = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc12_24.2 (constants.%U.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc12_28.1: type = splice_block %.loc12_28.2 [concrete = constants.%I_where.type.267] {
// CHECK:STDOUT:       %.Self.frozen.loc12_24: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.197]
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:       %.Self.frozen.loc12_28: %I.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.3a0]
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %I.ref [concrete]
// CHECK:STDOUT:       %.Self.ref.loc12_34.1: %I.type = name_ref .Self, %.Self.frozen.loc12_28 [symbolic_self = constants.%.Self.frozen.3a0]
// CHECK:STDOUT:       %.loc12_44: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %equiv.loc12_40.1 = requirement_equivalent %.Self.ref.loc12_34.1, %.loc12_44 [concrete]
// CHECK:STDOUT:       %.Self: %I.type = symbolic_binding .Self [symbolic_self = constants.%.Self.f73]
// CHECK:STDOUT:       %.Self.ref.loc12_34.2: %I.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self.f73]
// CHECK:STDOUT:       %equiv.loc12_40.2 = requirement_equivalent %.Self.ref.loc12_34.2, %.loc12_44 [concrete]
// CHECK:STDOUT:       %.loc12_28.2: type = where_expr [concrete = constants.%I_where.type.267] {
// CHECK:STDOUT:         %base_facet_type = requirement_base_facet_type %I.ref [concrete]
// CHECK:STDOUT:         %equiv.loc12_40.2 = requirement_equivalent %.Self.ref.loc12_34.2, %.loc12_44 [concrete]
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc12_24.2: %I_where.type.267 = symbolic_binding U, 0 [symbolic = %U.loc12_24.1 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Impls.decl: %Impls.type = fn_decl @Impls [concrete = constants.%Impls] {
// CHECK:STDOUT:     %V.patt.loc14_19.1: %pattern_type.17c = symbolic_binding_pattern V, 0 [symbolic = %V.patt.loc14_19.2 (constants.%V.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc14_23.1: type = splice_block %.loc14_23.2 [concrete = constants.%J_where.type] {
// CHECK:STDOUT:       %.Self.frozen.loc14_19: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.197]
// CHECK:STDOUT:       %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:       %.Self.frozen.loc14_23: %J.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.99c]
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %J.ref [concrete]
// CHECK:STDOUT:       %.Self.ref.loc14_29.1: %J.type = name_ref .Self, %.Self.frozen.loc14_23 [symbolic_self = constants.%.Self.frozen.99c]
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:       %.Self.as_type.loc14_29.1: type = facet_access_type %.Self.ref.loc14_29.1 [symbolic_self = constants.%.Self.frozen.as_type.8c8]
// CHECK:STDOUT:       %.loc14_29.1: type = converted %.Self.ref.loc14_29.1, %.Self.as_type.loc14_29.1 [symbolic_self = constants.%.Self.frozen.as_type.8c8]
// CHECK:STDOUT:       %impls.loc14_35.1 = requirement_impls %.loc14_29.1, %I.ref [concrete]
// CHECK:STDOUT:       %.Self: %J.type = symbolic_binding .Self [symbolic_self = constants.%.Self.6ef]
// CHECK:STDOUT:       %.Self.ref.loc14_29.2: %J.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self.6ef]
// CHECK:STDOUT:       %.Self.as_type.loc14_29.2: type = facet_access_type %.Self.ref.loc14_29.2 [symbolic_self = constants.%.Self.as_type.8c4]
// CHECK:STDOUT:       %.loc14_29.2: type = converted %.Self.ref.loc14_29.1, %.Self.as_type.loc14_29.2 [symbolic_self = constants.%.Self.as_type.8c4]
// CHECK:STDOUT:       %impls.loc14_35.2 = requirement_impls %.loc14_29.2, %I.ref [concrete]
// CHECK:STDOUT:       %.loc14_23.2: type = where_expr [concrete = constants.%J_where.type] {
// CHECK:STDOUT:         %base_facet_type = requirement_base_facet_type %J.ref [concrete]
// CHECK:STDOUT:         %impls.loc14_35.2 = requirement_impls %.loc14_29.2, %I.ref [concrete]
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %V.loc14_19.2: %J_where.type = symbolic_binding V, 0 [symbolic = %V.loc14_19.1 (constants.%V)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %And.decl: %And.type = fn_decl @And [concrete = constants.%And] {
// CHECK:STDOUT:     %W.patt.loc16_17.1: %pattern_type.e37 = symbolic_binding_pattern W, 0 [symbolic = %W.patt.loc16_17.2 (constants.%W.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc16_21.1: type = splice_block %.loc16_21.2 [concrete = constants.%I_where.type.a1d] {
// CHECK:STDOUT:       %.Self.frozen.loc16_17: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.197]
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:       %.Self.frozen.loc16_21: %I.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.3a0]
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %I.ref [concrete]
// CHECK:STDOUT:       %.Self.ref.loc16_27.1: %I.type = name_ref .Self, %.Self.frozen.loc16_21 [symbolic_self = constants.%.Self.frozen.3a0]
// CHECK:STDOUT:       %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:       %.Self.as_type.loc16_27.1: type = facet_access_type %.Self.ref.loc16_27.1 [symbolic_self = constants.%.Self.frozen.as_type.e90]
// CHECK:STDOUT:       %.loc16_27.1: type = converted %.Self.ref.loc16_27.1, %.Self.as_type.loc16_27.1 [symbolic_self = constants.%.Self.frozen.as_type.e90]
// CHECK:STDOUT:       %impls.loc16_33.1 = requirement_impls %.loc16_27.1, %J.ref [concrete]
// CHECK:STDOUT:       %.Self.ref.loc16_45: %I.type = name_ref .Self, %.Self.frozen.loc16_21 [symbolic_self = constants.%.Self.frozen.3a0]
// CHECK:STDOUT:       %.Self.as_type.loc16_45: type = facet_access_type %.Self.ref.loc16_45 [symbolic_self = constants.%.Self.frozen.as_type.e90]
// CHECK:STDOUT:       %.loc16_45: type = converted %.Self.ref.loc16_45, %.Self.as_type.loc16_45 [symbolic_self = constants.%.Self.frozen.as_type.e90]
// CHECK:STDOUT:       %Member.ref: %I.assoc_type = name_ref Member, @Member.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:       %impl.elem0.loc16_45.1: type = impl_witness_access constants.%I.lookup_impl_witness.ced, element0 [symbolic_self = constants.%impl.elem0.125]
// CHECK:STDOUT:       %.loc16_57: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %equiv.loc16_53.1 = requirement_equivalent %impl.elem0.loc16_45.1, %.loc16_57 [concrete]
// CHECK:STDOUT:       %.Self: %I.type = symbolic_binding .Self [symbolic_self = constants.%.Self.f73]
// CHECK:STDOUT:       %.Self.ref.loc16_27.2: %I.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self.f73]
// CHECK:STDOUT:       %.Self.as_type.loc16_27.2: type = facet_access_type %.Self.ref.loc16_27.2 [symbolic_self = constants.%.Self.as_type.f7e]
// CHECK:STDOUT:       %.loc16_27.2: type = converted %.Self.ref.loc16_27.1, %.Self.as_type.loc16_27.2 [symbolic_self = constants.%.Self.as_type.f7e]
// CHECK:STDOUT:       %impls.loc16_33.2 = requirement_impls %.loc16_27.2, %J.ref [concrete]
// CHECK:STDOUT:       %impl.elem0.loc16_45.2: type = impl_witness_access constants.%I.lookup_impl_witness.78b, element0 [symbolic_self = constants.%impl.elem0.e55]
// CHECK:STDOUT:       %equiv.loc16_53.2 = requirement_equivalent %impl.elem0.loc16_45.2, %.loc16_57 [concrete]
// CHECK:STDOUT:       %.loc16_21.2: type = where_expr [concrete = constants.%I_where.type.a1d] {
// CHECK:STDOUT:         %base_facet_type = requirement_base_facet_type %I.ref [concrete]
// CHECK:STDOUT:         %impls.loc16_33.2 = requirement_impls %.loc16_27.2, %J.ref [concrete]
// CHECK:STDOUT:         %equiv.loc16_53.2 = requirement_equivalent %impl.elem0.loc16_45.2, %.loc16_57 [concrete]
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %W.loc16_17.2: %I_where.type.a1d = symbolic_binding W, 0 [symbolic = %W.loc16_17.1 (constants.%W)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @EqualEqual(%U.loc12_24.2: %I_where.type.267) {
// CHECK:STDOUT:   %U.patt.loc12_24.2: %pattern_type.603 = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc12_24.2 (constants.%U.patt)]
// CHECK:STDOUT:   %U.loc12_24.1: %I_where.type.267 = symbolic_binding U, 0 [symbolic = %U.loc12_24.1 (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Impls(%V.loc14_19.2: %J_where.type) {
// CHECK:STDOUT:   %V.patt.loc14_19.2: %pattern_type.17c = symbolic_binding_pattern V, 0 [symbolic = %V.patt.loc14_19.2 (constants.%V.patt)]
// CHECK:STDOUT:   %V.loc14_19.1: %J_where.type = symbolic_binding V, 0 [symbolic = %V.loc14_19.1 (constants.%V)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @And(%W.loc16_17.2: %I_where.type.a1d) {
// CHECK:STDOUT:   %W.patt.loc16_17.2: %pattern_type.e37 = symbolic_binding_pattern W, 0 [symbolic = %W.patt.loc16_17.2 (constants.%W.patt)]
// CHECK:STDOUT:   %W.loc16_17.1: %I_where.type.a1d = symbolic_binding W, 0 [symbolic = %W.loc16_17.1 (constants.%W)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @EqualEqual(constants.%U) {
// CHECK:STDOUT:   %U.patt.loc12_24.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc12_24.1 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Impls(constants.%V) {
// CHECK:STDOUT:   %V.patt.loc14_19.2 => constants.%V.patt
// CHECK:STDOUT:   %V.loc14_19.1 => constants.%V
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @And(constants.%W) {
// CHECK:STDOUT:   %W.patt.loc16_17.2 => constants.%W.patt
// CHECK:STDOUT:   %W.loc16_17.1 => constants.%W
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- associated_type_impls.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %L.type: type = facet_type <@L> [concrete]
// CHECK:STDOUT:   %M.type: type = facet_type <@M> [concrete]
// CHECK:STDOUT:   %K.type: type = facet_type <@K> [concrete]
// CHECK:STDOUT:   %K.assoc_type: type = assoc_entity_type @K [concrete]
// CHECK:STDOUT:   %assoc0: %K.assoc_type = assoc_entity element0, @K.WithSelf.%Associated [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen.197: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.frozen.b1d: %K.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.frozen.as_type: type = facet_access_type %.Self.frozen.b1d [symbolic_self]
// CHECK:STDOUT:   %K.lookup_impl_witness.46f: <witness> = lookup_impl_witness %.Self.frozen.b1d, @K [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.c97: %L.type = impl_witness_access %K.lookup_impl_witness.46f, element0 [symbolic_self]
// CHECK:STDOUT:   %as_type.3da: type = facet_access_type %impl.elem0.c97 [symbolic_self]
// CHECK:STDOUT:   %.Self: %K.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %K.lookup_impl_witness.4f0: <witness> = lookup_impl_witness %.Self, @K [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.819: %L.type = impl_witness_access %K.lookup_impl_witness.4f0, element0 [symbolic_self]
// CHECK:STDOUT:   %as_type.75b: type = facet_access_type %impl.elem0.819 [symbolic_self]
// CHECK:STDOUT:   %K_where.type: type = facet_type <@K where %impl.elem0.819 impls @M> [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %K_where.type [concrete]
// CHECK:STDOUT:   %W.patt: %pattern_type = symbolic_binding_pattern W, 0 [symbolic]
// CHECK:STDOUT:   %W: %K_where.type = symbolic_binding W, 0 [symbolic]
// CHECK:STDOUT:   %AssociatedTypeImpls.type: type = fn_type @AssociatedTypeImpls [concrete]
// CHECK:STDOUT:   %AssociatedTypeImpls: %AssociatedTypeImpls.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %AssociatedTypeImpls.decl: %AssociatedTypeImpls.type = fn_decl @AssociatedTypeImpls [concrete = constants.%AssociatedTypeImpls] {
// CHECK:STDOUT:     %W.patt.loc12_33.1: %pattern_type = symbolic_binding_pattern W, 0 [symbolic = %W.patt.loc12_33.2 (constants.%W.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc12_37.1: type = splice_block %.loc12_37.2 [concrete = constants.%K_where.type] {
// CHECK:STDOUT:       %.Self.frozen.loc12_33: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.197]
// CHECK:STDOUT:       %K.ref: type = name_ref K, file.%K.decl [concrete = constants.%K.type]
// CHECK:STDOUT:       %.Self.frozen.loc12_37: %K.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.b1d]
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %K.ref [concrete]
// CHECK:STDOUT:       %.Self.ref: %K.type = name_ref .Self, %.Self.frozen.loc12_37 [symbolic_self = constants.%.Self.frozen.b1d]
// CHECK:STDOUT:       %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:       %.loc12_43.1: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:       %Associated.ref: %K.assoc_type = name_ref Associated, @Associated.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:       %impl.elem0.loc12_43.1: %L.type = impl_witness_access constants.%K.lookup_impl_witness.46f, element0 [symbolic_self = constants.%impl.elem0.c97]
// CHECK:STDOUT:       %M.ref: type = name_ref M, file.%M.decl [concrete = constants.%M.type]
// CHECK:STDOUT:       %as_type.loc12_43.1: type = facet_access_type %impl.elem0.loc12_43.1 [symbolic_self = constants.%as_type.3da]
// CHECK:STDOUT:       %.loc12_43.2: type = converted %impl.elem0.loc12_43.1, %as_type.loc12_43.1 [symbolic_self = constants.%as_type.3da]
// CHECK:STDOUT:       %impls.loc12_55.1 = requirement_impls %.loc12_43.2, %M.ref [concrete]
// CHECK:STDOUT:       %impl.elem0.loc12_43.2: %L.type = impl_witness_access constants.%K.lookup_impl_witness.4f0, element0 [symbolic_self = constants.%impl.elem0.819]
// CHECK:STDOUT:       %as_type.loc12_43.2: type = facet_access_type %impl.elem0.loc12_43.2 [symbolic_self = constants.%as_type.75b]
// CHECK:STDOUT:       %.loc12_43.3: type = converted %impl.elem0.loc12_43.1, %as_type.loc12_43.2 [symbolic_self = constants.%as_type.75b]
// CHECK:STDOUT:       %impls.loc12_55.2 = requirement_impls %.loc12_43.3, %M.ref [concrete]
// CHECK:STDOUT:       %.loc12_37.2: type = where_expr [concrete = constants.%K_where.type] {
// CHECK:STDOUT:         %base_facet_type = requirement_base_facet_type %K.ref [concrete]
// CHECK:STDOUT:         %impls.loc12_55.2 = requirement_impls %.loc12_43.3, %M.ref [concrete]
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %W.loc12_33.2: %K_where.type = symbolic_binding W, 0 [symbolic = %W.loc12_33.1 (constants.%W)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AssociatedTypeImpls(%W.loc12_33.2: %K_where.type) {
// CHECK:STDOUT:   %W.patt.loc12_33.2: %pattern_type = symbolic_binding_pattern W, 0 [symbolic = %W.patt.loc12_33.2 (constants.%W.patt)]
// CHECK:STDOUT:   %W.loc12_33.1: %K_where.type = symbolic_binding W, 0 [symbolic = %W.loc12_33.1 (constants.%W)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AssociatedTypeImpls(constants.%W) {
// CHECK:STDOUT:   %W.patt.loc12_33.2 => constants.%W.patt
// CHECK:STDOUT:   %W.loc12_33.1 => constants.%W
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_error_in_constraint.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.frozen.as_type: type = facet_access_type %.Self.frozen [symbolic_self]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %WithError.type: type = fn_type @WithError [concrete]
// CHECK:STDOUT:   %WithError: %WithError.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %WithError.decl: %WithError.type = fn_decl @WithError [concrete = constants.%WithError] {
// CHECK:STDOUT:     %U.patt: <error> = symbolic_binding_pattern U, 0 [concrete = <error>]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc9_30.1: type = splice_block %.loc9_30.2 [concrete = <error>] {
// CHECK:STDOUT:       %.Self.frozen.loc9_23: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc9_25: type = type_literal type [concrete = type]
// CHECK:STDOUT:       %.Self.frozen.loc9_30: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %.loc9_25 [concrete]
// CHECK:STDOUT:       %.Self.ref.loc9_36.1: %type = name_ref .Self, %.Self.frozen.loc9_30 [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %I.ref: <error> = name_ref I, <error> [concrete = <error>]
// CHECK:STDOUT:       %.Self.as_type.loc9_36.1: type = facet_access_type %.Self.ref.loc9_36.1 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:       %.loc9_36.1: type = converted %.Self.ref.loc9_36.1, %.Self.as_type.loc9_36.1 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:       %impls.loc9_42.1 = requirement_impls %.loc9_36.1, <error> [concrete]
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.Self.ref.loc9_36.2: %type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.Self.as_type.loc9_36.2: type = facet_access_type %.Self.ref.loc9_36.2 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc9_36.2: type = converted %.Self.ref.loc9_36.1, %.Self.as_type.loc9_36.2 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %impls.loc9_42.2 = requirement_impls %.loc9_36.2, <error> [concrete]
// CHECK:STDOUT:       %.loc9_30.2: type = where_expr [concrete = <error>] {
// CHECK:STDOUT:         %base_facet_type = requirement_base_facet_type %.loc9_25 [concrete]
// CHECK:STDOUT:         %impls.loc9_42.2 = requirement_impls %.loc9_36.2, <error> [concrete]
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
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type = value_binding_pattern x [concrete]
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
// CHECK:STDOUT:   %WithError.ref: %WithError.type = name_ref WithError, imports.%Main.WithError [concrete = constants.%WithError]
// CHECK:STDOUT:   %.loc8_33: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc8_18.1: type = splice_block %.loc8_18.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc8_18.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc8_18.3: type = converted %.loc8_18.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: %empty_tuple.type = wrapper_binding x, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type = value_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:

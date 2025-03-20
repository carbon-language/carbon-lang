// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/where_expr/constraints.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/where_expr/constraints.carbon

// --- state_constraints.carbon

library "[[@TEST_NAME]]";

interface J {}

interface I {
  let Member:! type;
  let Second:! J;
}

fn EqualEqual(U:! I where .Self == ());

fn Impls(V:! J where .Self impls I);

fn And(W:! I where .Self impls J and .Member == ());

// --- associated_type_impls.carbon

library "[[@TEST_NAME]]";

interface L {}
interface M {}

interface K {
  let Associated:! L;
}

fn AssociatedTypeImpls(W:! K where .Associated impls M);

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

// --- fail_todo_enforce_constraint.carbon

library "[[@TEST_NAME]]";

import library "state_constraints";

// C implements J but not I.
class C {}
impl C as J {}

// TODO: Should report that `C` does not meet the constraint.
fn DoesNotImplI() {
  // CHECK:STDERR: fail_todo_enforce_constraint.carbon:[[@LINE+11]]:3: error: cannot implicitly convert type `C` into type implementing `J where...` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   Impls(C);
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR: fail_todo_enforce_constraint.carbon:[[@LINE+8]]:3: note: type `type` does not implement interface `Core.ImplicitAs(J where...)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   Impls(C);
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR: fail_todo_enforce_constraint.carbon:[[@LINE-14]]:1: in import [InImport]
  // CHECK:STDERR: state_constraints.carbon:13:10: note: initializing generic parameter `V` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn Impls(V:! J where .Self impls I);
  // CHECK:STDERR:          ^
  // CHECK:STDERR:
  Impls(C);
}

fn EmptyStruct(Y:! J where .Self == {});

// TODO: Should report that `C` does not meeet the constraint.
fn NotEmptyStruct() {
  // CHECK:STDERR: fail_todo_enforce_constraint.carbon:[[@LINE+10]]:3: error: cannot implicitly convert type `C` into type implementing `J where...` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   EmptyStruct(C);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_enforce_constraint.carbon:[[@LINE+7]]:3: note: type `type` does not implement interface `Core.ImplicitAs(J where...)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   EmptyStruct(C);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_enforce_constraint.carbon:[[@LINE-10]]:16: note: initializing generic parameter `Y` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn EmptyStruct(Y:! J where .Self == {});
  // CHECK:STDERR:                ^
  // CHECK:STDERR:
  EmptyStruct(C);
}

// CHECK:STDOUT: --- state_constraints.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self.ccd: %J.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.826: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type %I.type [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.%Member [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %assoc1: %I.assoc_type = assoc_entity element1, @I.%Second [concrete]
// CHECK:STDOUT:   %.Self.258: %I.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I where TODO> [concrete]
// CHECK:STDOUT:   %U: %I_where.type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %U.patt: %I_where.type = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %EqualEqual.type: type = fn_type @EqualEqual [concrete]
// CHECK:STDOUT:   %EqualEqual: %EqualEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %.Self.968: %J.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type.78d: type = facet_access_type %.Self.968 [symbolic_self]
// CHECK:STDOUT:   %J_where.type: type = facet_type <@J where TODO> [concrete]
// CHECK:STDOUT:   %V: %J_where.type = bind_symbolic_name V, 0 [symbolic]
// CHECK:STDOUT:   %V.patt: %J_where.type = symbolic_binding_pattern V, 0 [symbolic]
// CHECK:STDOUT:   %Impls.type: type = fn_type @Impls [concrete]
// CHECK:STDOUT:   %Impls: %Impls.type = struct_value () [concrete]
// CHECK:STDOUT:   %.Self.as_type.541: type = facet_access_type %.Self.258 [symbolic_self]
// CHECK:STDOUT:   %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self.258, element0 [symbolic_self]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %.Self.as_type.541, (%.Self.as_wit.iface0) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %.Self.as_wit.iface0, element0 [symbolic_self]
// CHECK:STDOUT:   %W: %I_where.type = bind_symbolic_name W, 0 [symbolic]
// CHECK:STDOUT:   %W.patt: %I_where.type = symbolic_binding_pattern W, 0 [symbolic]
// CHECK:STDOUT:   %And.type: type = fn_type @And [concrete]
// CHECK:STDOUT:   %And: %And.type = struct_value () [concrete]
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
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .EqualEqual = %EqualEqual.decl
// CHECK:STDOUT:     .Impls = %Impls.decl
// CHECK:STDOUT:     .And = %And.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [concrete = constants.%J.type] {} {}
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %EqualEqual.decl: %EqualEqual.type = fn_decl @EqualEqual [concrete = constants.%EqualEqual] {
// CHECK:STDOUT:     %U.patt.loc11_15.1: %I_where.type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc11_15.2 (constants.%U.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11_21.1: type = splice_block %.loc11_21.2 [concrete = constants.%I_where.type] {
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:       %.Self: %I.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self.258]
// CHECK:STDOUT:       %.Self.ref: %I.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self.258]
// CHECK:STDOUT:       %.loc11_37: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc11_21.2: type = where_expr %.Self [concrete = constants.%I_where.type] {
// CHECK:STDOUT:         requirement_equivalent %.Self.ref, %.loc11_37
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc11_15.1: %I_where.type = bind_symbolic_name U, 0 [symbolic = %U.loc11_15.2 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Impls.decl: %Impls.type = fn_decl @Impls [concrete = constants.%Impls] {
// CHECK:STDOUT:     %V.patt.loc13_10.1: %J_where.type = symbolic_binding_pattern V, 0 [symbolic = %V.patt.loc13_10.2 (constants.%V.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc13_16.1: type = splice_block %.loc13_16.2 [concrete = constants.%J_where.type] {
// CHECK:STDOUT:       %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:       %.Self: %J.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self.968]
// CHECK:STDOUT:       %.Self.ref: %J.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self.968]
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:       %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type.78d]
// CHECK:STDOUT:       %.loc13_22: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type.78d]
// CHECK:STDOUT:       %.loc13_16.2: type = where_expr %.Self [concrete = constants.%J_where.type] {
// CHECK:STDOUT:         requirement_impls %.loc13_22, %I.ref
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %V.loc13_10.1: %J_where.type = bind_symbolic_name V, 0 [symbolic = %V.loc13_10.2 (constants.%V)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %And.decl: %And.type = fn_decl @And [concrete = constants.%And] {
// CHECK:STDOUT:     %W.patt.loc15_8.1: %I_where.type = symbolic_binding_pattern W, 0 [symbolic = %W.patt.loc15_8.2 (constants.%W.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc15_14.1: type = splice_block %.loc15_14.2 [concrete = constants.%I_where.type] {
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:       %.Self: %I.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self.258]
// CHECK:STDOUT:       %.Self.ref.loc15_20: %I.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self.258]
// CHECK:STDOUT:       %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:       %.Self.as_type.loc15_20: type = facet_access_type %.Self.ref.loc15_20 [symbolic_self = constants.%.Self.as_type.541]
// CHECK:STDOUT:       %.loc15_20: type = converted %.Self.ref.loc15_20, %.Self.as_type.loc15_20 [symbolic_self = constants.%.Self.as_type.541]
// CHECK:STDOUT:       %.Self.ref.loc15_38: %I.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self.258]
// CHECK:STDOUT:       %Member.ref: %I.assoc_type = name_ref Member, @Member.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:       %.Self.as_type.loc15_38: type = facet_access_type %.Self.ref.loc15_38 [symbolic_self = constants.%.Self.as_type.541]
// CHECK:STDOUT:       %.loc15_38: type = converted %.Self.ref.loc15_38, %.Self.as_type.loc15_38 [symbolic_self = constants.%.Self.as_type.541]
// CHECK:STDOUT:       %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self.ref.loc15_38, element0 [symbolic_self = constants.%.Self.as_wit.iface0]
// CHECK:STDOUT:       %impl.elem0: type = impl_witness_access %.Self.as_wit.iface0, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:       %.loc15_50: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc15_14.2: type = where_expr %.Self [concrete = constants.%I_where.type] {
// CHECK:STDOUT:         requirement_impls %.loc15_20, %J.ref
// CHECK:STDOUT:         requirement_equivalent %impl.elem0, %.loc15_50
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %W.loc15_8.1: %I_where.type = bind_symbolic_name W, 0 [symbolic = %W.loc15_8.2 (constants.%W)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.ccd]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.826]
// CHECK:STDOUT:   %Member: type = assoc_const_decl @Member [concrete] {
// CHECK:STDOUT:     %assoc0: %I.assoc_type = assoc_entity element0, @I.%Member [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Second: %J.type = assoc_const_decl @Second [concrete] {
// CHECK:STDOUT:     %assoc1: %I.assoc_type = assoc_entity element1, @I.%Second [concrete = constants.%assoc1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Member = @Member.%assoc0
// CHECK:STDOUT:   .J = <poisoned>
// CHECK:STDOUT:   .Second = @Second.%assoc1
// CHECK:STDOUT:   witness = (%Member, %Second)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @Member(@I.%Self: %I.type) {
// CHECK:STDOUT:   assoc_const Member:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @Second(@I.%Self: %I.type) {
// CHECK:STDOUT:   assoc_const Second:! %J.type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @EqualEqual(%U.loc11_15.1: %I_where.type) {
// CHECK:STDOUT:   %U.loc11_15.2: %I_where.type = bind_symbolic_name U, 0 [symbolic = %U.loc11_15.2 (constants.%U)]
// CHECK:STDOUT:   %U.patt.loc11_15.2: %I_where.type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc11_15.2 (constants.%U.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%U.patt.loc11_15.1: %I_where.type);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Impls(%V.loc13_10.1: %J_where.type) {
// CHECK:STDOUT:   %V.loc13_10.2: %J_where.type = bind_symbolic_name V, 0 [symbolic = %V.loc13_10.2 (constants.%V)]
// CHECK:STDOUT:   %V.patt.loc13_10.2: %J_where.type = symbolic_binding_pattern V, 0 [symbolic = %V.patt.loc13_10.2 (constants.%V.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%V.patt.loc13_10.1: %J_where.type);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @And(%W.loc15_8.1: %I_where.type) {
// CHECK:STDOUT:   %W.loc15_8.2: %I_where.type = bind_symbolic_name W, 0 [symbolic = %W.loc15_8.2 (constants.%W)]
// CHECK:STDOUT:   %W.patt.loc15_8.2: %I_where.type = symbolic_binding_pattern W, 0 [symbolic = %W.patt.loc15_8.2 (constants.%W.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%W.patt.loc15_8.1: %I_where.type);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Member(constants.%Self.826) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Second(constants.%Self.826) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @EqualEqual(constants.%U) {
// CHECK:STDOUT:   %U.loc11_15.2 => constants.%U
// CHECK:STDOUT:   %U.patt.loc11_15.2 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Impls(constants.%V) {
// CHECK:STDOUT:   %V.loc13_10.2 => constants.%V
// CHECK:STDOUT:   %V.patt.loc13_10.2 => constants.%V
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Member(constants.%I.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @And(constants.%W) {
// CHECK:STDOUT:   %W.loc15_8.2 => constants.%W
// CHECK:STDOUT:   %W.patt.loc15_8.2 => constants.%W
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- associated_type_impls.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %L.type: type = facet_type <@L> [concrete]
// CHECK:STDOUT:   %Self.1d2: %L.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %M.type: type = facet_type <@M> [concrete]
// CHECK:STDOUT:   %Self.bcc: %M.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %K.type: type = facet_type <@K> [concrete]
// CHECK:STDOUT:   %Self.09f: %K.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %K.assoc_type: type = assoc_entity_type %K.type [concrete]
// CHECK:STDOUT:   %assoc0: %K.assoc_type = assoc_entity element0, @K.%Associated [concrete]
// CHECK:STDOUT:   %.Self: %K.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self, element0 [symbolic_self]
// CHECK:STDOUT:   %K.facet: %K.type = facet_value %.Self.as_type, (%.Self.as_wit.iface0) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: %L.type = impl_witness_access %.Self.as_wit.iface0, element0 [symbolic_self]
// CHECK:STDOUT:   %as_type: type = facet_access_type %impl.elem0 [symbolic_self]
// CHECK:STDOUT:   %K_where.type: type = facet_type <@K where TODO> [concrete]
// CHECK:STDOUT:   %W: %K_where.type = bind_symbolic_name W, 0 [symbolic]
// CHECK:STDOUT:   %W.patt: %K_where.type = symbolic_binding_pattern W, 0 [symbolic]
// CHECK:STDOUT:   %AssociatedTypeImpls.type: type = fn_type @AssociatedTypeImpls [concrete]
// CHECK:STDOUT:   %AssociatedTypeImpls: %AssociatedTypeImpls.type = struct_value () [concrete]
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
// CHECK:STDOUT:     .L = %L.decl
// CHECK:STDOUT:     .M = %M.decl
// CHECK:STDOUT:     .K = %K.decl
// CHECK:STDOUT:     .AssociatedTypeImpls = %AssociatedTypeImpls.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %L.decl: type = interface_decl @L [concrete = constants.%L.type] {} {}
// CHECK:STDOUT:   %M.decl: type = interface_decl @M [concrete = constants.%M.type] {} {}
// CHECK:STDOUT:   %K.decl: type = interface_decl @K [concrete = constants.%K.type] {} {}
// CHECK:STDOUT:   %AssociatedTypeImpls.decl: %AssociatedTypeImpls.type = fn_decl @AssociatedTypeImpls [concrete = constants.%AssociatedTypeImpls] {
// CHECK:STDOUT:     %W.patt.loc11_24.1: %K_where.type = symbolic_binding_pattern W, 0 [symbolic = %W.patt.loc11_24.2 (constants.%W.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11_30.1: type = splice_block %.loc11_30.2 [concrete = constants.%K_where.type] {
// CHECK:STDOUT:       %K.ref: type = name_ref K, file.%K.decl [concrete = constants.%K.type]
// CHECK:STDOUT:       %.Self: %K.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.Self.ref: %K.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Associated.ref: %K.assoc_type = name_ref Associated, @Associated.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:       %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc11_36.1: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self.ref, element0 [symbolic_self = constants.%.Self.as_wit.iface0]
// CHECK:STDOUT:       %impl.elem0: %L.type = impl_witness_access %.Self.as_wit.iface0, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:       %M.ref: type = name_ref M, file.%M.decl [concrete = constants.%M.type]
// CHECK:STDOUT:       %as_type: type = facet_access_type %impl.elem0 [symbolic_self = constants.%as_type]
// CHECK:STDOUT:       %.loc11_36.2: type = converted %impl.elem0, %as_type [symbolic_self = constants.%as_type]
// CHECK:STDOUT:       %.loc11_30.2: type = where_expr %.Self [concrete = constants.%K_where.type] {
// CHECK:STDOUT:         requirement_impls %.loc11_36.2, %M.ref
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %W.loc11_24.1: %K_where.type = bind_symbolic_name W, 0 [symbolic = %W.loc11_24.2 (constants.%W)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @L {
// CHECK:STDOUT:   %Self: %L.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.1d2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @M {
// CHECK:STDOUT:   %Self: %M.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.bcc]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @K {
// CHECK:STDOUT:   %Self: %K.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.09f]
// CHECK:STDOUT:   %Associated: %L.type = assoc_const_decl @Associated [concrete] {
// CHECK:STDOUT:     %assoc0: %K.assoc_type = assoc_entity element0, @K.%Associated [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .L = <poisoned>
// CHECK:STDOUT:   .Associated = @Associated.%assoc0
// CHECK:STDOUT:   witness = (%Associated)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @Associated(@K.%Self: %K.type) {
// CHECK:STDOUT:   assoc_const Associated:! %L.type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AssociatedTypeImpls(%W.loc11_24.1: %K_where.type) {
// CHECK:STDOUT:   %W.loc11_24.2: %K_where.type = bind_symbolic_name W, 0 [symbolic = %W.loc11_24.2 (constants.%W)]
// CHECK:STDOUT:   %W.patt.loc11_24.2: %K_where.type = symbolic_binding_pattern W, 0 [symbolic = %W.patt.loc11_24.2 (constants.%W.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%W.patt.loc11_24.1: %K_where.type);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Associated(constants.%Self.09f) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Associated(constants.%K.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AssociatedTypeImpls(constants.%W) {
// CHECK:STDOUT:   %W.loc11_24.2 => constants.%W
// CHECK:STDOUT:   %W.patt.loc11_24.2 => constants.%W
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_left_of_impls_non_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.Self: type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %int_7: Core.IntLiteral = int_value 7 [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where TODO> [concrete]
// CHECK:STDOUT:   %U: %type_where = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %U.patt: %type_where = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %NonTypeImpls.type: type = fn_type @NonTypeImpls [concrete]
// CHECK:STDOUT:   %NonTypeImpls: %NonTypeImpls.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .NonTypeImpls = %NonTypeImpls.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %NonTypeImpls.decl: %NonTypeImpls.type = fn_decl @NonTypeImpls [concrete = constants.%NonTypeImpls] {
// CHECK:STDOUT:     %U.patt.loc11_17.1: %type_where = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc11_17.2 (constants.%U.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11_26.1: type = splice_block %.loc11_26.2 [concrete = constants.%type_where] {
// CHECK:STDOUT:       %.Self: type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %int_7: Core.IntLiteral = int_value 7 [concrete = constants.%int_7]
// CHECK:STDOUT:       %.loc11_32: type = converted %int_7, <error> [concrete = <error>]
// CHECK:STDOUT:       %.loc11_26.2: type = where_expr %.Self [concrete = constants.%type_where] {
// CHECK:STDOUT:         requirement_impls <error>, type
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc11_17.1: %type_where = bind_symbolic_name U, 0 [symbolic = %U.loc11_17.2 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @NonTypeImpls(%U.loc11_17.1: %type_where) {
// CHECK:STDOUT:   %U.loc11_17.2: %type_where = bind_symbolic_name U, 0 [symbolic = %U.loc11_17.2 (constants.%U)]
// CHECK:STDOUT:   %U.patt.loc11_17.2: %type_where = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc11_17.2 (constants.%U.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%U.patt.loc11_17.1: %type_where);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @NonTypeImpls(constants.%U) {
// CHECK:STDOUT:   %U.loc11_17.2 => constants.%U
// CHECK:STDOUT:   %U.patt.loc11_17.2 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_right_of_impls_non_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.Self: type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %int_7: Core.IntLiteral = int_value 7 [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where TODO> [concrete]
// CHECK:STDOUT:   %U: %type_where = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %U.patt: %type_where = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %ImplsNonType.type: type = fn_type @ImplsNonType [concrete]
// CHECK:STDOUT:   %ImplsNonType: %ImplsNonType.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .ImplsNonType = %ImplsNonType.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %ImplsNonType.decl: %ImplsNonType.type = fn_decl @ImplsNonType [concrete = constants.%ImplsNonType] {
// CHECK:STDOUT:     %U.patt.loc11_17.1: %type_where = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc11_17.2 (constants.%U.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11_26.1: type = splice_block %.loc11_26.2 [concrete = constants.%type_where] {
// CHECK:STDOUT:       %.Self: type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.Self.ref: type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %int_7: Core.IntLiteral = int_value 7 [concrete = constants.%int_7]
// CHECK:STDOUT:       %.loc11_44: type = converted %int_7, <error> [concrete = <error>]
// CHECK:STDOUT:       %.loc11_26.2: type = where_expr %.Self [concrete = constants.%type_where] {
// CHECK:STDOUT:         requirement_impls %.Self.ref, <error>
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc11_17.1: %type_where = bind_symbolic_name U, 0 [symbolic = %U.loc11_17.2 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ImplsNonType(%U.loc11_17.1: %type_where) {
// CHECK:STDOUT:   %U.loc11_17.2: %type_where = bind_symbolic_name U, 0 [symbolic = %U.loc11_17.2 (constants.%U)]
// CHECK:STDOUT:   %U.patt.loc11_17.2: %type_where = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc11_17.2 (constants.%U.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%U.patt.loc11_17.1: %type_where);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplsNonType(constants.%U) {
// CHECK:STDOUT:   %U.loc11_17.2 => constants.%U
// CHECK:STDOUT:   %U.patt.loc11_17.2 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_right_of_impls_non_facet_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.Self: type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where TODO> [concrete]
// CHECK:STDOUT:   %U: %type_where = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %U.patt: %type_where = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %ImplsOfNonFacetType.type: type = fn_type @ImplsOfNonFacetType [concrete]
// CHECK:STDOUT:   %ImplsOfNonFacetType: %ImplsOfNonFacetType.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .ImplsOfNonFacetType = %ImplsOfNonFacetType.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %ImplsOfNonFacetType.decl: %ImplsOfNonFacetType.type = fn_decl @ImplsOfNonFacetType [concrete = constants.%ImplsOfNonFacetType] {
// CHECK:STDOUT:     %U.patt.loc8_24.1: %type_where = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc8_24.2 (constants.%U.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8_33.1: type = splice_block %.loc8_33.2 [concrete = constants.%type_where] {
// CHECK:STDOUT:       %.Self: type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.Self.ref: type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %.loc8_33.2: type = where_expr %.Self [concrete = constants.%type_where] {
// CHECK:STDOUT:         requirement_impls %.Self.ref, <error>
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc8_24.1: %type_where = bind_symbolic_name U, 0 [symbolic = %U.loc8_24.2 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ImplsOfNonFacetType(%U.loc8_24.1: %type_where) {
// CHECK:STDOUT:   %U.loc8_24.2: %type_where = bind_symbolic_name U, 0 [symbolic = %U.loc8_24.2 (constants.%U)]
// CHECK:STDOUT:   %U.patt.loc8_24.2: %type_where = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc8_24.2 (constants.%U.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%U.patt.loc8_24.1: %type_where);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplsOfNonFacetType(constants.%U) {
// CHECK:STDOUT:   %U.loc8_24.2 => constants.%U
// CHECK:STDOUT:   %U.patt.loc8_24.2 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_enforce_constraint.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness () [concrete]
// CHECK:STDOUT:   %DoesNotImplI.type: type = fn_type @DoesNotImplI [concrete]
// CHECK:STDOUT:   %DoesNotImplI: %DoesNotImplI.type = struct_value () [concrete]
// CHECK:STDOUT:   %Impls.type: type = fn_type @Impls [concrete]
// CHECK:STDOUT:   %Impls: %Impls.type = struct_value () [concrete]
// CHECK:STDOUT:   %J_where.type: type = facet_type <@J where TODO> [concrete]
// CHECK:STDOUT:   %V.patt: %J_where.type = symbolic_binding_pattern V, 0 [symbolic]
// CHECK:STDOUT:   %V: %J_where.type = bind_symbolic_name V, 0 [symbolic]
// CHECK:STDOUT:   %.Self: %J.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %Y: %J_where.type = bind_symbolic_name Y, 0 [symbolic]
// CHECK:STDOUT:   %Y.patt: %J_where.type = symbolic_binding_pattern Y, 0 [symbolic]
// CHECK:STDOUT:   %EmptyStruct.type: type = fn_type @EmptyStruct [concrete]
// CHECK:STDOUT:   %EmptyStruct: %EmptyStruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %NotEmptyStruct.type: type = fn_type @NotEmptyStruct [concrete]
// CHECK:STDOUT:   %NotEmptyStruct: %NotEmptyStruct.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.J: type = import_ref Main//state_constraints, J, loaded [concrete = constants.%J.type]
// CHECK:STDOUT:   %Main.I = import_ref Main//state_constraints, I, unloaded
// CHECK:STDOUT:   %Main.EqualEqual = import_ref Main//state_constraints, EqualEqual, unloaded
// CHECK:STDOUT:   %Main.Impls: %Impls.type = import_ref Main//state_constraints, Impls, loaded [concrete = constants.%Impls]
// CHECK:STDOUT:   %Main.And = import_ref Main//state_constraints, And, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.8fd = import_ref Main//state_constraints, inst18 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.9d4: %J_where.type = import_ref Main//state_constraints, loc13_10, loaded [symbolic = @Impls.%V (constants.%V)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .J = imports.%Main.J
// CHECK:STDOUT:     .I = imports.%Main.I
// CHECK:STDOUT:     .EqualEqual = imports.%Main.EqualEqual
// CHECK:STDOUT:     .Impls = imports.%Main.Impls
// CHECK:STDOUT:     .And = imports.%Main.And
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .DoesNotImplI = %DoesNotImplI.decl
// CHECK:STDOUT:     .EmptyStruct = %EmptyStruct.decl
// CHECK:STDOUT:     .NotEmptyStruct = %NotEmptyStruct.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %J.ref: type = name_ref J, imports.%Main.J [concrete = constants.%J.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness () [concrete = constants.%impl_witness]
// CHECK:STDOUT:   %DoesNotImplI.decl: %DoesNotImplI.type = fn_decl @DoesNotImplI [concrete = constants.%DoesNotImplI] {} {}
// CHECK:STDOUT:   %EmptyStruct.decl: %EmptyStruct.type = fn_decl @EmptyStruct [concrete = constants.%EmptyStruct] {
// CHECK:STDOUT:     %Y.patt.loc26_16.1: %J_where.type = symbolic_binding_pattern Y, 0 [symbolic = %Y.patt.loc26_16.2 (constants.%Y.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc26_22.1: type = splice_block %.loc26_22.2 [concrete = constants.%J_where.type] {
// CHECK:STDOUT:       %J.ref: type = name_ref J, imports.%Main.J [concrete = constants.%J.type]
// CHECK:STDOUT:       %.Self: %J.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.Self.ref: %J.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc26_38: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:       %.loc26_22.2: type = where_expr %.Self [concrete = constants.%J_where.type] {
// CHECK:STDOUT:         requirement_equivalent %.Self.ref, %.loc26_38
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Y.loc26_16.1: %J_where.type = bind_symbolic_name Y, 0 [symbolic = %Y.loc26_16.2 (constants.%Y)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NotEmptyStruct.decl: %NotEmptyStruct.type = fn_decl @NotEmptyStruct [concrete = constants.%NotEmptyStruct] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J [from "state_constraints.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.8fd
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %C.ref as %J.ref {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DoesNotImplI() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Impls.ref: %Impls.type = name_ref Impls, imports.%Main.Impls [concrete = constants.%Impls]
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc23: %J_where.type = converted %C.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Impls(imports.%Main.import_ref.9d4: %J_where.type) [from "state_constraints.carbon"] {
// CHECK:STDOUT:   %V: %J_where.type = bind_symbolic_name V, 0 [symbolic = %V (constants.%V)]
// CHECK:STDOUT:   %V.patt.2: %J_where.type = symbolic_binding_pattern V, 0 [symbolic = %V.patt.2 (constants.%V.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%V.patt.1: %J_where.type);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @EmptyStruct(%Y.loc26_16.1: %J_where.type) {
// CHECK:STDOUT:   %Y.loc26_16.2: %J_where.type = bind_symbolic_name Y, 0 [symbolic = %Y.loc26_16.2 (constants.%Y)]
// CHECK:STDOUT:   %Y.patt.loc26_16.2: %J_where.type = symbolic_binding_pattern Y, 0 [symbolic = %Y.patt.loc26_16.2 (constants.%Y.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%Y.patt.loc26_16.1: %J_where.type);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NotEmptyStruct() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %EmptyStruct.ref: %EmptyStruct.type = name_ref EmptyStruct, file.%EmptyStruct.decl [concrete = constants.%EmptyStruct]
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc40: %J_where.type = converted %C.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Impls(constants.%V) {
// CHECK:STDOUT:   %V => constants.%V
// CHECK:STDOUT:   %V.patt.2 => constants.%V
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @EmptyStruct(constants.%Y) {
// CHECK:STDOUT:   %Y.loc26_16.2 => constants.%Y
// CHECK:STDOUT:   %Y.patt.loc26_16.2 => constants.%Y
// CHECK:STDOUT: }
// CHECK:STDOUT:

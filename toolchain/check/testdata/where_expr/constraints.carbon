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

// CHECK:STDERR: fail_left_of_impls_non_type.carbon:[[@LINE+7]]:32: error: cannot implicitly convert from `Core.IntLiteral` to `type` [ImplicitAsConversionFailure]
// CHECK:STDERR: fn NonTypeImpls(U:! type where 7 impls type);
// CHECK:STDERR:                                ^
// CHECK:STDERR: fail_left_of_impls_non_type.carbon:[[@LINE+4]]:32: note: type `Core.IntLiteral` does not implement interface `ImplicitAs(type)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: fn NonTypeImpls(U:! type where 7 impls type);
// CHECK:STDERR:                                ^
// CHECK:STDERR:
fn NonTypeImpls(U:! type where 7 impls type);

// --- fail_right_of_impls_non_type.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_right_of_impls_non_type.carbon:[[@LINE+7]]:44: error: cannot implicitly convert from `Core.IntLiteral` to `type` [ImplicitAsConversionFailure]
// CHECK:STDERR: fn ImplsNonType(U:! type where .Self impls 7);
// CHECK:STDERR:                                            ^
// CHECK:STDERR: fail_right_of_impls_non_type.carbon:[[@LINE+4]]:44: note: type `Core.IntLiteral` does not implement interface `ImplicitAs(type)` [MissingImplInMemberAccessNote]
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
  // CHECK:STDERR: fail_todo_enforce_constraint.carbon:[[@LINE+11]]:3: error: cannot implicitly convert from `type` to `J where...` [ImplicitAsConversionFailure]
  // CHECK:STDERR:   Impls(C);
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR: fail_todo_enforce_constraint.carbon:[[@LINE+8]]:3: note: type `type` does not implement interface `ImplicitAs(J where...)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   Impls(C);
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR: fail_todo_enforce_constraint.carbon:[[@LINE-14]]:1: in import [InImport]
  // CHECK:STDERR: state_constraints.carbon:13:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Impls(V:! J where .Self impls I);
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Impls(C);
}

fn EmptyStruct(Y:! J where .Self == {});

// TODO: Should report that `C` does not meeet the constraint.
fn NotEmptyStruct() {
  // CHECK:STDERR: fail_todo_enforce_constraint.carbon:[[@LINE+9]]:3: error: cannot implicitly convert from `type` to `J where...` [ImplicitAsConversionFailure]
  // CHECK:STDERR:   EmptyStruct(C);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_enforce_constraint.carbon:[[@LINE+6]]:3: note: type `type` does not implement interface `ImplicitAs(J where...)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   EmptyStruct(C);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_enforce_constraint.carbon:[[@LINE-10]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn EmptyStruct(Y:! J where .Self == {});
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  EmptyStruct(C);
}

// CHECK:STDOUT: --- state_constraints.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [template]
// CHECK:STDOUT:   %Self.1: %J.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [template]
// CHECK:STDOUT:   %Self.2: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %assoc_type.1: type = assoc_entity_type %I.type, type [template]
// CHECK:STDOUT:   %assoc0: %assoc_type.1 = assoc_entity element0, @I.%Member [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %assoc_type.2: type = assoc_entity_type %I.type, %J.type [template]
// CHECK:STDOUT:   %assoc1: %assoc_type.2 = assoc_entity element1, @I.%Second [template]
// CHECK:STDOUT:   %.Self.1: %I.type = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I where TODO> [template]
// CHECK:STDOUT:   %U: %I_where.type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %U.patt: %I_where.type = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %EqualEqual.type: type = fn_type @EqualEqual [template]
// CHECK:STDOUT:   %EqualEqual: %EqualEqual.type = struct_value () [template]
// CHECK:STDOUT:   %.Self.2: %J.type = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %.Self.as_type.1: type = facet_access_type %.Self.2 [symbolic]
// CHECK:STDOUT:   %J_where.type: type = facet_type <@J where TODO> [template]
// CHECK:STDOUT:   %V: %J_where.type = bind_symbolic_name V, 0 [symbolic]
// CHECK:STDOUT:   %V.patt: %J_where.type = symbolic_binding_pattern V, 0 [symbolic]
// CHECK:STDOUT:   %Impls.type: type = fn_type @Impls [template]
// CHECK:STDOUT:   %Impls: %Impls.type = struct_value () [template]
// CHECK:STDOUT:   %.Self.as_type.2: type = facet_access_type %.Self.1 [symbolic]
// CHECK:STDOUT:   %.Self.as_wit: <witness> = facet_access_witness %.Self.1 [symbolic]
// CHECK:STDOUT:   %impl.elem0: type = interface_witness_access %.Self.as_wit, element0 [symbolic]
// CHECK:STDOUT:   %W: %I_where.type = bind_symbolic_name W, 0 [symbolic]
// CHECK:STDOUT:   %W.patt: %I_where.type = symbolic_binding_pattern W, 0 [symbolic]
// CHECK:STDOUT:   %And.type: type = fn_type @And [template]
// CHECK:STDOUT:   %And: %And.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .EqualEqual = %EqualEqual.decl
// CHECK:STDOUT:     .Impls = %Impls.decl
// CHECK:STDOUT:     .And = %And.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [template = constants.%J.type] {} {}
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [template = constants.%I.type] {} {}
// CHECK:STDOUT:   %EqualEqual.decl: %EqualEqual.type = fn_decl @EqualEqual [template = constants.%EqualEqual] {
// CHECK:STDOUT:     %U.patt.loc11_15.1: %I_where.type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc11_15.2 (constants.%U.patt)]
// CHECK:STDOUT:     %U.param_patt: %I_where.type = value_param_pattern %U.patt.loc11_15.1, runtime_param<invalid> [symbolic = %U.patt.loc11_15.2 (constants.%U.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [template = constants.%I.type]
// CHECK:STDOUT:     %.Self: %I.type = bind_symbolic_name .Self [symbolic = constants.%.Self.1]
// CHECK:STDOUT:     %.Self.ref: %I.type = name_ref .Self, %.Self [symbolic = constants.%.Self.1]
// CHECK:STDOUT:     %.loc11_37: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc11_21: type = where_expr %.Self [template = constants.%I_where.type] {
// CHECK:STDOUT:       requirement_equivalent %.Self.ref, %.loc11_37
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.param: %I_where.type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %U.loc11_15.1: %I_where.type = bind_symbolic_name U, 0, %U.param [symbolic = %U.loc11_15.2 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Impls.decl: %Impls.type = fn_decl @Impls [template = constants.%Impls] {
// CHECK:STDOUT:     %V.patt.loc13_10.1: %J_where.type = symbolic_binding_pattern V, 0 [symbolic = %V.patt.loc13_10.2 (constants.%V.patt)]
// CHECK:STDOUT:     %V.param_patt: %J_where.type = value_param_pattern %V.patt.loc13_10.1, runtime_param<invalid> [symbolic = %V.patt.loc13_10.2 (constants.%V.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [template = constants.%J.type]
// CHECK:STDOUT:     %.Self: %J.type = bind_symbolic_name .Self [symbolic = constants.%.Self.2]
// CHECK:STDOUT:     %.Self.ref: %J.type = name_ref .Self, %.Self [symbolic = constants.%.Self.2]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [template = constants.%I.type]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic = constants.%.Self.as_type.1]
// CHECK:STDOUT:     %.loc13_22: type = converted %.Self.ref, %.Self.as_type [symbolic = constants.%.Self.as_type.1]
// CHECK:STDOUT:     %.loc13_16: type = where_expr %.Self [template = constants.%J_where.type] {
// CHECK:STDOUT:       requirement_impls %.loc13_22, %I.ref
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %V.param: %J_where.type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %V.loc13_10.1: %J_where.type = bind_symbolic_name V, 0, %V.param [symbolic = %V.loc13_10.2 (constants.%V)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %And.decl: %And.type = fn_decl @And [template = constants.%And] {
// CHECK:STDOUT:     %W.patt.loc15_8.1: %I_where.type = symbolic_binding_pattern W, 0 [symbolic = %W.patt.loc15_8.2 (constants.%W.patt)]
// CHECK:STDOUT:     %W.param_patt: %I_where.type = value_param_pattern %W.patt.loc15_8.1, runtime_param<invalid> [symbolic = %W.patt.loc15_8.2 (constants.%W.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [template = constants.%I.type]
// CHECK:STDOUT:     %.Self: %I.type = bind_symbolic_name .Self [symbolic = constants.%.Self.1]
// CHECK:STDOUT:     %.Self.ref.loc15_20: %I.type = name_ref .Self, %.Self [symbolic = constants.%.Self.1]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [template = constants.%J.type]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref.loc15_20 [symbolic = constants.%.Self.as_type.2]
// CHECK:STDOUT:     %.loc15_20: type = converted %.Self.ref.loc15_20, %.Self.as_type [symbolic = constants.%.Self.as_type.2]
// CHECK:STDOUT:     %.Self.ref.loc15_38: %I.type = name_ref .Self, %.Self [symbolic = constants.%.Self.1]
// CHECK:STDOUT:     %Member.ref: %assoc_type.1 = name_ref Member, @I.%assoc0 [template = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_wit: <witness> = facet_access_witness %.Self.ref.loc15_38 [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem0: type = interface_witness_access %.Self.as_wit, element0 [symbolic = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc15_50: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc15_14: type = where_expr %.Self [template = constants.%I_where.type] {
// CHECK:STDOUT:       requirement_impls %.loc15_20, %J.ref
// CHECK:STDOUT:       requirement_equivalent %impl.elem0, %.loc15_50
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %W.param: %I_where.type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %W.loc15_8.1: %I_where.type = bind_symbolic_name W, 0, %W.param [symbolic = %W.loc15_8.2 (constants.%W)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.2]
// CHECK:STDOUT:   %Member: type = assoc_const_decl Member [template]
// CHECK:STDOUT:   %assoc0: %assoc_type.1 = assoc_entity element0, %Member [template = constants.%assoc0]
// CHECK:STDOUT:   %J.ref: type = name_ref J, file.%J.decl [template = constants.%J.type]
// CHECK:STDOUT:   %Second: %J.type = assoc_const_decl Second [template]
// CHECK:STDOUT:   %assoc1: %assoc_type.2 = assoc_entity element1, %Second [template = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Member = %assoc0
// CHECK:STDOUT:   .Second = %assoc1
// CHECK:STDOUT:   witness = (%Member, %Second)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @EqualEqual(%U.loc11_15.1: %I_where.type) {
// CHECK:STDOUT:   %U.loc11_15.2: %I_where.type = bind_symbolic_name U, 0 [symbolic = %U.loc11_15.2 (constants.%U)]
// CHECK:STDOUT:   %U.patt.loc11_15.2: %I_where.type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc11_15.2 (constants.%U.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%U.param_patt: %I_where.type);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Impls(%V.loc13_10.1: %J_where.type) {
// CHECK:STDOUT:   %V.loc13_10.2: %J_where.type = bind_symbolic_name V, 0 [symbolic = %V.loc13_10.2 (constants.%V)]
// CHECK:STDOUT:   %V.patt.loc13_10.2: %J_where.type = symbolic_binding_pattern V, 0 [symbolic = %V.patt.loc13_10.2 (constants.%V.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%V.param_patt: %J_where.type);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @And(%W.loc15_8.1: %I_where.type) {
// CHECK:STDOUT:   %W.loc15_8.2: %I_where.type = bind_symbolic_name W, 0 [symbolic = %W.loc15_8.2 (constants.%W)]
// CHECK:STDOUT:   %W.patt.loc15_8.2: %I_where.type = symbolic_binding_pattern W, 0 [symbolic = %W.patt.loc15_8.2 (constants.%W.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%W.param_patt: %I_where.type);
// CHECK:STDOUT: }
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
// CHECK:STDOUT: specific @And(constants.%W) {
// CHECK:STDOUT:   %W.loc15_8.2 => constants.%W
// CHECK:STDOUT:   %W.patt.loc15_8.2 => constants.%W
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- associated_type_impls.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %L.type: type = facet_type <@L> [template]
// CHECK:STDOUT:   %Self.1: %L.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %M.type: type = facet_type <@M> [template]
// CHECK:STDOUT:   %Self.2: %M.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %K.type: type = facet_type <@K> [template]
// CHECK:STDOUT:   %Self.3: %K.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %assoc_type: type = assoc_entity_type %K.type, %L.type [template]
// CHECK:STDOUT:   %assoc0: %assoc_type = assoc_entity element0, @K.%Associated [template]
// CHECK:STDOUT:   %.Self: %K.type = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %.Self.as_wit: <witness> = facet_access_witness %.Self [symbolic]
// CHECK:STDOUT:   %impl.elem0: %L.type = interface_witness_access %.Self.as_wit, element0 [symbolic]
// CHECK:STDOUT:   %as_type: type = facet_access_type %impl.elem0 [symbolic]
// CHECK:STDOUT:   %K_where.type: type = facet_type <@K where TODO> [template]
// CHECK:STDOUT:   %W: %K_where.type = bind_symbolic_name W, 0 [symbolic]
// CHECK:STDOUT:   %W.patt: %K_where.type = symbolic_binding_pattern W, 0 [symbolic]
// CHECK:STDOUT:   %AssociatedTypeImpls.type: type = fn_type @AssociatedTypeImpls [template]
// CHECK:STDOUT:   %AssociatedTypeImpls: %AssociatedTypeImpls.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .L = %L.decl
// CHECK:STDOUT:     .M = %M.decl
// CHECK:STDOUT:     .K = %K.decl
// CHECK:STDOUT:     .AssociatedTypeImpls = %AssociatedTypeImpls.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %L.decl: type = interface_decl @L [template = constants.%L.type] {} {}
// CHECK:STDOUT:   %M.decl: type = interface_decl @M [template = constants.%M.type] {} {}
// CHECK:STDOUT:   %K.decl: type = interface_decl @K [template = constants.%K.type] {} {}
// CHECK:STDOUT:   %AssociatedTypeImpls.decl: %AssociatedTypeImpls.type = fn_decl @AssociatedTypeImpls [template = constants.%AssociatedTypeImpls] {
// CHECK:STDOUT:     %W.patt.loc11_24.1: %K_where.type = symbolic_binding_pattern W, 0 [symbolic = %W.patt.loc11_24.2 (constants.%W.patt)]
// CHECK:STDOUT:     %W.param_patt: %K_where.type = value_param_pattern %W.patt.loc11_24.1, runtime_param<invalid> [symbolic = %W.patt.loc11_24.2 (constants.%W.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %K.ref: type = name_ref K, file.%K.decl [template = constants.%K.type]
// CHECK:STDOUT:     %.Self: %K.type = bind_symbolic_name .Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %K.type = name_ref .Self, %.Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %Associated.ref: %assoc_type = name_ref Associated, @K.%assoc0 [template = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_wit: <witness> = facet_access_witness %.Self.ref [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem0: %L.type = interface_witness_access %.Self.as_wit, element0 [symbolic = constants.%impl.elem0]
// CHECK:STDOUT:     %M.ref: type = name_ref M, file.%M.decl [template = constants.%M.type]
// CHECK:STDOUT:     %as_type: type = facet_access_type %impl.elem0 [symbolic = constants.%as_type]
// CHECK:STDOUT:     %.loc11_36: type = converted %impl.elem0, %as_type [symbolic = constants.%as_type]
// CHECK:STDOUT:     %.loc11_30: type = where_expr %.Self [template = constants.%K_where.type] {
// CHECK:STDOUT:       requirement_impls %.loc11_36, %M.ref
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %W.param: %K_where.type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %W.loc11_24.1: %K_where.type = bind_symbolic_name W, 0, %W.param [symbolic = %W.loc11_24.2 (constants.%W)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @L {
// CHECK:STDOUT:   %Self: %L.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @M {
// CHECK:STDOUT:   %Self: %M.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @K {
// CHECK:STDOUT:   %Self: %K.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.3]
// CHECK:STDOUT:   %L.ref: type = name_ref L, file.%L.decl [template = constants.%L.type]
// CHECK:STDOUT:   %Associated: %L.type = assoc_const_decl Associated [template]
// CHECK:STDOUT:   %assoc0: %assoc_type = assoc_entity element0, %Associated [template = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Associated = %assoc0
// CHECK:STDOUT:   witness = (%Associated)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AssociatedTypeImpls(%W.loc11_24.1: %K_where.type) {
// CHECK:STDOUT:   %W.loc11_24.2: %K_where.type = bind_symbolic_name W, 0 [symbolic = %W.loc11_24.2 (constants.%W)]
// CHECK:STDOUT:   %W.patt.loc11_24.2: %K_where.type = symbolic_binding_pattern W, 0 [symbolic = %W.patt.loc11_24.2 (constants.%W.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%W.param_patt: %K_where.type);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AssociatedTypeImpls(constants.%W) {
// CHECK:STDOUT:   %W.loc11_24.2 => constants.%W
// CHECK:STDOUT:   %W.patt.loc11_24.2 => constants.%W
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_left_of_impls_non_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.Self: type = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %int_7: Core.IntLiteral = int_value 7 [template]
// CHECK:STDOUT:   %type_where: type = facet_type <type where TODO> [template]
// CHECK:STDOUT:   %U: %type_where = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %U.patt: %type_where = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %NonTypeImpls.type: type = fn_type @NonTypeImpls [template]
// CHECK:STDOUT:   %NonTypeImpls: %NonTypeImpls.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .ImplicitAs = %import_ref.1
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .NonTypeImpls = %NonTypeImpls.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %NonTypeImpls.decl: %NonTypeImpls.type = fn_decl @NonTypeImpls [template = constants.%NonTypeImpls] {
// CHECK:STDOUT:     %U.patt.loc11_17.1: %type_where = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc11_17.2 (constants.%U.patt)]
// CHECK:STDOUT:     %U.param_patt: %type_where = value_param_pattern %U.patt.loc11_17.1, runtime_param<invalid> [symbolic = %U.patt.loc11_17.2 (constants.%U.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: type = bind_symbolic_name .Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %int_7: Core.IntLiteral = int_value 7 [template = constants.%int_7]
// CHECK:STDOUT:     %.loc11_32: type = converted %int_7, <error> [template = <error>]
// CHECK:STDOUT:     %.loc11_26: type = where_expr %.Self [template = constants.%type_where] {
// CHECK:STDOUT:       requirement_impls <error>, type
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.param: %type_where = value_param runtime_param<invalid>
// CHECK:STDOUT:     %U.loc11_17.1: %type_where = bind_symbolic_name U, 0, %U.param [symbolic = %U.loc11_17.2 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @NonTypeImpls(%U.loc11_17.1: %type_where) {
// CHECK:STDOUT:   %U.loc11_17.2: %type_where = bind_symbolic_name U, 0 [symbolic = %U.loc11_17.2 (constants.%U)]
// CHECK:STDOUT:   %U.patt.loc11_17.2: %type_where = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc11_17.2 (constants.%U.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%U.param_patt: %type_where);
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
// CHECK:STDOUT:   %.Self: type = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %int_7: Core.IntLiteral = int_value 7 [template]
// CHECK:STDOUT:   %type_where: type = facet_type <type where TODO> [template]
// CHECK:STDOUT:   %U: %type_where = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %U.patt: %type_where = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %ImplsNonType.type: type = fn_type @ImplsNonType [template]
// CHECK:STDOUT:   %ImplsNonType: %ImplsNonType.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .ImplicitAs = %import_ref.1
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .ImplsNonType = %ImplsNonType.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %ImplsNonType.decl: %ImplsNonType.type = fn_decl @ImplsNonType [template = constants.%ImplsNonType] {
// CHECK:STDOUT:     %U.patt.loc11_17.1: %type_where = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc11_17.2 (constants.%U.patt)]
// CHECK:STDOUT:     %U.param_patt: %type_where = value_param_pattern %U.patt.loc11_17.1, runtime_param<invalid> [symbolic = %U.patt.loc11_17.2 (constants.%U.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: type = bind_symbolic_name .Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: type = name_ref .Self, %.Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %int_7: Core.IntLiteral = int_value 7 [template = constants.%int_7]
// CHECK:STDOUT:     %.loc11_44: type = converted %int_7, <error> [template = <error>]
// CHECK:STDOUT:     %.loc11_26: type = where_expr %.Self [template = constants.%type_where] {
// CHECK:STDOUT:       requirement_impls %.Self.ref, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.param: %type_where = value_param runtime_param<invalid>
// CHECK:STDOUT:     %U.loc11_17.1: %type_where = bind_symbolic_name U, 0, %U.param [symbolic = %U.loc11_17.2 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ImplsNonType(%U.loc11_17.1: %type_where) {
// CHECK:STDOUT:   %U.loc11_17.2: %type_where = bind_symbolic_name U, 0 [symbolic = %U.loc11_17.2 (constants.%U)]
// CHECK:STDOUT:   %U.patt.loc11_17.2: %type_where = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc11_17.2 (constants.%U.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%U.param_patt: %type_where);
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
// CHECK:STDOUT:   %.Self: type = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %type_where: type = facet_type <type where TODO> [template]
// CHECK:STDOUT:   %U: %type_where = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %U.patt: %type_where = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %ImplsOfNonFacetType.type: type = fn_type @ImplsOfNonFacetType [template]
// CHECK:STDOUT:   %ImplsOfNonFacetType: %ImplsOfNonFacetType.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .ImplsOfNonFacetType = %ImplsOfNonFacetType.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %ImplsOfNonFacetType.decl: %ImplsOfNonFacetType.type = fn_decl @ImplsOfNonFacetType [template = constants.%ImplsOfNonFacetType] {
// CHECK:STDOUT:     %U.patt.loc8_24.1: %type_where = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc8_24.2 (constants.%U.patt)]
// CHECK:STDOUT:     %U.param_patt: %type_where = value_param_pattern %U.patt.loc8_24.1, runtime_param<invalid> [symbolic = %U.patt.loc8_24.2 (constants.%U.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: type = bind_symbolic_name .Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: type = name_ref .Self, %.Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc8_51.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc8_51.2: type = converted %int.make_type_signed, %.loc8_51.1 [template = constants.%i32]
// CHECK:STDOUT:     %.loc8_33: type = where_expr %.Self [template = constants.%type_where] {
// CHECK:STDOUT:       requirement_impls %.Self.ref, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.param: %type_where = value_param runtime_param<invalid>
// CHECK:STDOUT:     %U.loc8_24.1: %type_where = bind_symbolic_name U, 0, %U.param [symbolic = %U.loc8_24.2 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ImplsOfNonFacetType(%U.loc8_24.1: %type_where) {
// CHECK:STDOUT:   %U.loc8_24.2: %type_where = bind_symbolic_name U, 0 [symbolic = %U.loc8_24.2 (constants.%U)]
// CHECK:STDOUT:   %U.patt.loc8_24.2: %type_where = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc8_24.2 (constants.%U.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%U.param_patt: %type_where);
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
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [template]
// CHECK:STDOUT:   %interface.1: <witness> = interface_witness () [template]
// CHECK:STDOUT:   %DoesNotImplI.type: type = fn_type @DoesNotImplI [template]
// CHECK:STDOUT:   %DoesNotImplI: %DoesNotImplI.type = struct_value () [template]
// CHECK:STDOUT:   %Impls.type: type = fn_type @Impls [template]
// CHECK:STDOUT:   %Impls: %Impls.type = struct_value () [template]
// CHECK:STDOUT:   %J_where.type: type = facet_type <@J where TODO> [template]
// CHECK:STDOUT:   %V: %J_where.type = bind_symbolic_name V, 0 [symbolic]
// CHECK:STDOUT:   %V.patt: %J_where.type = symbolic_binding_pattern V, 0 [symbolic]
// CHECK:STDOUT:   %.Self: %J.type = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %Y: %J_where.type = bind_symbolic_name Y, 0 [symbolic]
// CHECK:STDOUT:   %Y.patt: %J_where.type = symbolic_binding_pattern Y, 0 [symbolic]
// CHECK:STDOUT:   %EmptyStruct.type: type = fn_type @EmptyStruct [template]
// CHECK:STDOUT:   %EmptyStruct: %EmptyStruct.type = struct_value () [template]
// CHECK:STDOUT:   %NotEmptyStruct.type: type = fn_type @NotEmptyStruct [template]
// CHECK:STDOUT:   %NotEmptyStruct: %NotEmptyStruct.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: type = import_ref Main//state_constraints, inst+3, loaded [template = constants.%J.type]
// CHECK:STDOUT:   %import_ref.2 = import_ref Main//state_constraints, inst+7, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref Main//state_constraints, inst+35, unloaded
// CHECK:STDOUT:   %import_ref.4: %Impls.type = import_ref Main//state_constraints, inst+57, loaded [template = constants.%Impls]
// CHECK:STDOUT:   %import_ref.5 = import_ref Main//state_constraints, inst+85, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .ImplicitAs = %import_ref.7
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.6 = import_ref Main//state_constraints, inst+5, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .J = imports.%import_ref.1
// CHECK:STDOUT:     .I = imports.%import_ref.2
// CHECK:STDOUT:     .EqualEqual = imports.%import_ref.3
// CHECK:STDOUT:     .Impls = imports.%import_ref.4
// CHECK:STDOUT:     .And = imports.%import_ref.5
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .DoesNotImplI = %DoesNotImplI.decl
// CHECK:STDOUT:     .EmptyStruct = %EmptyStruct.decl
// CHECK:STDOUT:     .NotEmptyStruct = %NotEmptyStruct.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @impl.1 [template] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %J.ref: type = name_ref J, imports.%import_ref.1 [template = constants.%J.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %DoesNotImplI.decl: %DoesNotImplI.type = fn_decl @DoesNotImplI [template = constants.%DoesNotImplI] {} {}
// CHECK:STDOUT:   %EmptyStruct.decl: %EmptyStruct.type = fn_decl @EmptyStruct [template = constants.%EmptyStruct] {
// CHECK:STDOUT:     %Y.patt.loc26_16.1: %J_where.type = symbolic_binding_pattern Y, 0 [symbolic = %Y.patt.loc26_16.2 (constants.%Y.patt)]
// CHECK:STDOUT:     %Y.param_patt: %J_where.type = value_param_pattern %Y.patt.loc26_16.1, runtime_param<invalid> [symbolic = %Y.patt.loc26_16.2 (constants.%Y.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %J.ref: type = name_ref J, imports.%import_ref.1 [template = constants.%J.type]
// CHECK:STDOUT:     %.Self: %J.type = bind_symbolic_name .Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %J.type = name_ref .Self, %.Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %.loc26_38: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc26_22: type = where_expr %.Self [template = constants.%J_where.type] {
// CHECK:STDOUT:       requirement_equivalent %.Self.ref, %.loc26_38
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Y.param: %J_where.type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %Y.loc26_16.1: %J_where.type = bind_symbolic_name Y, 0, %Y.param [symbolic = %Y.loc26_16.2 (constants.%Y)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NotEmptyStruct.decl: %NotEmptyStruct.type = fn_decl @NotEmptyStruct [template = constants.%NotEmptyStruct] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.6
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: %C.ref as %J.ref {
// CHECK:STDOUT:   %interface: <witness> = interface_witness () [template = constants.%interface.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = %interface
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DoesNotImplI() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Impls.ref: %Impls.type = name_ref Impls, imports.%import_ref.4 [template = constants.%Impls]
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %.loc23: %J_where.type = converted %C.ref, <error> [template = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Impls(constants.%V: %J_where.type) {
// CHECK:STDOUT:   %V: %J_where.type = bind_symbolic_name V, 0 [symbolic = %V (constants.%V)]
// CHECK:STDOUT:   %V.patt: %J_where.type = symbolic_binding_pattern V, 0 [symbolic = %V.patt (constants.%V.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%V.param_patt: %J_where.type);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @EmptyStruct(%Y.loc26_16.1: %J_where.type) {
// CHECK:STDOUT:   %Y.loc26_16.2: %J_where.type = bind_symbolic_name Y, 0 [symbolic = %Y.loc26_16.2 (constants.%Y)]
// CHECK:STDOUT:   %Y.patt.loc26_16.2: %J_where.type = symbolic_binding_pattern Y, 0 [symbolic = %Y.patt.loc26_16.2 (constants.%Y.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%Y.param_patt: %J_where.type);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NotEmptyStruct() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %EmptyStruct.ref: %EmptyStruct.type = name_ref EmptyStruct, file.%EmptyStruct.decl [template = constants.%EmptyStruct]
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %.loc39: %J_where.type = converted %C.ref, <error> [template = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Impls(constants.%V) {
// CHECK:STDOUT:   %V => constants.%V
// CHECK:STDOUT:   %V.patt => constants.%V
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @EmptyStruct(constants.%Y) {
// CHECK:STDOUT:   %Y.loc26_16.2 => constants.%Y
// CHECK:STDOUT:   %Y.patt.loc26_16.2 => constants.%Y
// CHECK:STDOUT: }
// CHECK:STDOUT:

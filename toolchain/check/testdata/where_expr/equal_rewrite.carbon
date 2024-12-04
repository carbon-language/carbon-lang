// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/where_expr/equal_rewrite.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/where_expr/equal_rewrite.carbon

// --- equal_constraint.carbon

library "[[@TEST_NAME]]";

interface N {
  let P:! type;
}

fn Equal(T:! N where .P = {});

// --- nested_rewrites.carbon

library "[[@TEST_NAME]]";

interface A {
  let B:! type;
  let C:! type;
}

fn NestedRewrite(D:! (A where .B = bool) where .C = ());

// --- repeated_rewrite.carbon

library "[[@TEST_NAME]]";

interface E {
  let F:! type;
}

fn OneRewrite(G:! E where .F = i32) {}

fn RepeatedRewrite(H:! E where .F = i32 and .F = i32) {
  OneRewrite(H);
}

fn OneRewriteAgain(I:! E where .F = i32) {
  RepeatedRewrite(I);
}

// --- rewrites_reordered.carbon

library "[[@TEST_NAME]]";

interface J {
  let K:! type;
  let L:! type;
}

fn Alphabetical(M:! J where .K = () and .L = bool) {}

fn Reversed(N:! J where .L = bool and .K = ()) {
  Alphabetical(N);
}

// --- fail_rewrites_mismatch_right.carbon

library "[[@TEST_NAME]]";

interface O {
  let P:! type;
}

fn WithInteger(Q:! O where .P = i32) {}

fn WithBool(R:! O where .P = bool) {
  // CHECK:STDERR: fail_rewrites_mismatch_right.carbon:[[@LINE+10]]:3: error: cannot implicitly convert from `O where .(O.P) = bool` to `O where .(O.P) = i32` [ImplicitAsConversionFailure]
  // CHECK:STDERR:   WithInteger(R);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR: fail_rewrites_mismatch_right.carbon:[[@LINE+7]]:3: note: type `O where .(O.P) = bool` does not implement interface `ImplicitAs(O where .(O.P) = i32)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   WithInteger(R);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR: fail_rewrites_mismatch_right.carbon:[[@LINE-9]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn WithInteger(Q:! O where .P = i32) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  WithInteger(R);
}

// --- fail_rewrites_mismatch_left.carbon

library "[[@TEST_NAME]]";

interface S {
  let T:! type;
  let U:! type;
}

fn WithT(V:! S where .T = ()) {}

fn WithU(W:! S where .U = ()) {
  // CHECK:STDERR: fail_rewrites_mismatch_left.carbon:[[@LINE+10]]:3: error: cannot implicitly convert from `S where .(S.U) = ()` to `S where .(S.T) = ()` [ImplicitAsConversionFailure]
  // CHECK:STDERR:   WithT(W);
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR: fail_rewrites_mismatch_left.carbon:[[@LINE+7]]:3: note: type `S where .(S.U) = ()` does not implement interface `ImplicitAs(S where .(S.T) = ())` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   WithT(W);
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR: fail_rewrites_mismatch_left.carbon:[[@LINE-9]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn WithT(V:! S where .T = ()) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  WithT(W);
}

// --- fail_import_rewrites.carbon

library "[[@TEST_NAME]]";

import library "equal_constraint";
import library "nested_rewrites";

fn Calls() {
  // CHECK:STDERR: fail_import_rewrites.carbon:[[@LINE+11]]:3: error: cannot implicitly convert from `type` to `N where .(N.P) = {}` [ImplicitAsConversionFailure]
  // CHECK:STDERR:   Equal(bool);
  // CHECK:STDERR:   ^~~~~~~~~~~
  // CHECK:STDERR: fail_import_rewrites.carbon:[[@LINE+8]]:3: note: type `type` does not implement interface `ImplicitAs(N where .(N.P) = {})` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   Equal(bool);
  // CHECK:STDERR:   ^~~~~~~~~~~
  // CHECK:STDERR: fail_import_rewrites.carbon:[[@LINE-10]]:1: in import [InImport]
  // CHECK:STDERR: equal_constraint.carbon:8:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Equal(T:! N where .P = {});
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Equal(bool);

  // CHECK:STDERR: fail_import_rewrites.carbon:[[@LINE+11]]:3: error: cannot implicitly convert from `type` to `A where .(A.C) = () and .(A.B) = bool` [ImplicitAsConversionFailure]
  // CHECK:STDERR:   NestedRewrite(i32);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_rewrites.carbon:[[@LINE+8]]:3: note: type `type` does not implement interface `ImplicitAs(A where .(A.C) = () and .(A.B) = bool)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   NestedRewrite(i32);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_rewrites.carbon:[[@LINE-23]]:1: in import [InImport]
  // CHECK:STDERR: nested_rewrites.carbon:9:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn NestedRewrite(D:! (A where .B = bool) where .C = ());
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  NestedRewrite(i32);
}

// --- fail_check_rewrite_constraints.carbon

library "[[@TEST_NAME]]";

interface I {
  let Member:! type;
}

// `2` can't be converted to the type of `I.Member`
// CHECK:STDERR: fail_check_rewrite_constraints.carbon:[[@LINE+7]]:46: error: cannot implicitly convert from `Core.IntLiteral` to `type` [ImplicitAsConversionFailure]
// CHECK:STDERR: fn RewriteTypeMismatch(X:! I where .Member = 2);
// CHECK:STDERR:                                              ^
// CHECK:STDERR: fail_check_rewrite_constraints.carbon:[[@LINE+4]]:46: note: type `Core.IntLiteral` does not implement interface `ImplicitAs(type)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: fn RewriteTypeMismatch(X:! I where .Member = 2);
// CHECK:STDERR:                                              ^
// CHECK:STDERR:
fn RewriteTypeMismatch(X:! I where .Member = 2);

// --- fail_todo_let.carbon

library "[[@TEST_NAME]]";

interface A {}
class D {}
impl D as A {}
// TODO: This should be a compile-time binding, once that is supported.
// CHECK:STDERR: fail_todo_let.carbon:[[@LINE+7]]:1: error: cannot implicitly convert from `type` to `type where...` [ImplicitAsConversionFailure]
// CHECK:STDERR: let B: type where .Self impls A = D;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_let.carbon:[[@LINE+4]]:1: note: type `type` does not implement interface `ImplicitAs(type where...)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: let B: type where .Self impls A = D;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let B: type where .Self impls A = D;

// --- fail_type_does_not_implement_where.carbon

library "[[@TEST_NAME]]";

interface E {
  let F:! type;
  let G:! type;
}
// Testing how these types get stringified in error messages.

// TODO: This should be a compile-time binding, once that is supported.
// CHECK:STDERR: fail_type_does_not_implement_where.carbon:[[@LINE+7]]:1: error: cannot implicitly convert from `type` to `E where .(E.G) = () and .(E.F) = bool` [ImplicitAsConversionFailure]
// CHECK:STDERR: let H: (E where .F = bool and .G = ()) = f64;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_type_does_not_implement_where.carbon:[[@LINE+4]]:1: note: type `type` does not implement interface `ImplicitAs(E where .(E.G) = () and .(E.F) = bool)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: let H: (E where .F = bool and .G = ()) = f64;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let H: (E where .F = bool and .G = ()) = f64;

// CHECK:STDERR: fail_type_does_not_implement_where.carbon:[[@LINE+7]]:1: error: cannot implicitly convert from `type` to `E where .(E.G) = i32 and .(E.F) = {}` [ImplicitAsConversionFailure]
// CHECK:STDERR: let J: ((E where .F = {}) where .G = i32) = bool;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_type_does_not_implement_where.carbon:[[@LINE+4]]:1: note: type `type` does not implement interface `ImplicitAs(E where .(E.G) = i32 and .(E.F) = {})` [MissingImplInMemberAccessNote]
// CHECK:STDERR: let J: ((E where .F = {}) where .G = i32) = bool;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let J: ((E where .F = {}) where .G = i32) = bool;

// CHECK:STDERR: fail_type_does_not_implement_where.carbon:[[@LINE+6]]:1: error: cannot implicitly convert from `type` to `E where .(E.F) = .(E.G)` [ImplicitAsConversionFailure]
// CHECK:STDERR: let K: (E where .F = .Self.G) = bool;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_type_does_not_implement_where.carbon:[[@LINE+3]]:1: note: type `type` does not implement interface `ImplicitAs(E where .(E.F) = .(E.G))` [MissingImplInMemberAccessNote]
// CHECK:STDERR: let K: (E where .F = .Self.G) = bool;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
let K: (E where .F = .Self.G) = bool;

// CHECK:STDOUT: --- equal_constraint.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %N.type: type = facet_type <@N> [template]
// CHECK:STDOUT:   %Self: %N.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %assoc_type: type = assoc_entity_type %N.type, type [template]
// CHECK:STDOUT:   %assoc0: %assoc_type = assoc_entity element0, @N.%P [template]
// CHECK:STDOUT:   %.Self: %N.type = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %.Self.as_wit: <witness> = facet_access_witness %.Self [symbolic]
// CHECK:STDOUT:   %impl.elem0: type = interface_witness_access %.Self.as_wit, element0 [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %N_where.type: type = facet_type <@N where %impl.elem0 = %empty_struct_type> [template]
// CHECK:STDOUT:   %T: %N_where.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: %N_where.type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Equal.type: type = fn_type @Equal [template]
// CHECK:STDOUT:   %Equal: %Equal.type = struct_value () [template]
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
// CHECK:STDOUT:     .N = %N.decl
// CHECK:STDOUT:     .Equal = %Equal.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %N.decl: type = interface_decl @N [template = constants.%N.type] {} {}
// CHECK:STDOUT:   %Equal.decl: %Equal.type = fn_decl @Equal [template = constants.%Equal] {
// CHECK:STDOUT:     %T.patt.loc8_10.1: %N_where.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_10.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: %N_where.type = value_param_pattern %T.patt.loc8_10.1, runtime_param<invalid> [symbolic = %T.patt.loc8_10.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %N.ref: type = name_ref N, file.%N.decl [template = constants.%N.type]
// CHECK:STDOUT:     %.Self: %N.type = bind_symbolic_name .Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %N.type = name_ref .Self, %.Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %P.ref: %assoc_type = name_ref P, @N.%assoc0 [template = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_wit: <witness> = facet_access_witness %.Self.ref [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem0: type = interface_witness_access %.Self.as_wit, element0 [symbolic = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc8_28.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc8_28.2: type = converted %.loc8_28.1, constants.%empty_struct_type [template = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc8_16: type = where_expr %.Self [template = constants.%N_where.type] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %.loc8_28.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.param: %N_where.type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc8_10.1: %N_where.type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc8_10.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @N {
// CHECK:STDOUT:   %Self: %N.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %P: type = assoc_const_decl P [template]
// CHECK:STDOUT:   %assoc0: %assoc_type = assoc_entity element0, %P [template = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .P = %assoc0
// CHECK:STDOUT:   witness = (%P)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Equal(%T.loc8_10.1: %N_where.type) {
// CHECK:STDOUT:   %T.loc8_10.2: %N_where.type = bind_symbolic_name T, 0 [symbolic = %T.loc8_10.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc8_10.2: %N_where.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_10.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.param_patt: %N_where.type);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Equal(constants.%T) {
// CHECK:STDOUT:   %T.loc8_10.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc8_10.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- nested_rewrites.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = facet_type <@A> [template]
// CHECK:STDOUT:   %Self: %A.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %assoc_type: type = assoc_entity_type %A.type, type [template]
// CHECK:STDOUT:   %assoc0: %assoc_type = assoc_entity element0, @A.%B [template]
// CHECK:STDOUT:   %assoc1: %assoc_type = assoc_entity element1, @A.%C [template]
// CHECK:STDOUT:   %.Self.1: %A.type = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %.Self.as_wit.1: <witness> = facet_access_witness %.Self.1 [symbolic]
// CHECK:STDOUT:   %impl.elem0: type = interface_witness_access %.Self.as_wit.1, element0 [symbolic]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [template]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [template]
// CHECK:STDOUT:   %A_where.type.1: type = facet_type <@A where %impl.elem0 = bool> [template]
// CHECK:STDOUT:   %.Self.2: %A_where.type.1 = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %.Self.as_wit.2: <witness> = facet_access_witness %.Self.2 [symbolic]
// CHECK:STDOUT:   %impl.elem1: type = interface_witness_access %.Self.as_wit.2, element1 [symbolic]
// CHECK:STDOUT:   %A_where.type.2: type = facet_type <@A where %impl.elem1 = %empty_tuple.type and %impl.elem0 = bool> [template]
// CHECK:STDOUT:   %D: %A_where.type.2 = bind_symbolic_name D, 0 [symbolic]
// CHECK:STDOUT:   %D.patt: %A_where.type.2 = symbolic_binding_pattern D, 0 [symbolic]
// CHECK:STDOUT:   %NestedRewrite.type: type = fn_type @NestedRewrite [template]
// CHECK:STDOUT:   %NestedRewrite: %NestedRewrite.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Bool = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .NestedRewrite = %NestedRewrite.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = interface_decl @A [template = constants.%A.type] {} {}
// CHECK:STDOUT:   %NestedRewrite.decl: %NestedRewrite.type = fn_decl @NestedRewrite [template = constants.%NestedRewrite] {
// CHECK:STDOUT:     %D.patt.loc9_18.1: %A_where.type.2 = symbolic_binding_pattern D, 0 [symbolic = %D.patt.loc9_18.2 (constants.%D.patt)]
// CHECK:STDOUT:     %D.param_patt: %A_where.type.2 = value_param_pattern %D.patt.loc9_18.1, runtime_param<invalid> [symbolic = %D.patt.loc9_18.2 (constants.%D.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [template = constants.%A.type]
// CHECK:STDOUT:     %.Self.1: %A.type = bind_symbolic_name .Self [symbolic = constants.%.Self.1]
// CHECK:STDOUT:     %.Self.ref.loc9_31: %A.type = name_ref .Self, %.Self.1 [symbolic = constants.%.Self.1]
// CHECK:STDOUT:     %B.ref: %assoc_type = name_ref B, @A.%assoc0 [template = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_wit.loc9_31: <witness> = facet_access_witness %.Self.ref.loc9_31 [symbolic = constants.%.Self.as_wit.1]
// CHECK:STDOUT:     %impl.elem0: type = interface_witness_access %.Self.as_wit.loc9_31, element0 [symbolic = constants.%impl.elem0]
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc9_36.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc9_36.2: type = converted %bool.make_type, %.loc9_36.1 [template = bool]
// CHECK:STDOUT:     %.loc9_25: type = where_expr %.Self.1 [template = constants.%A_where.type.1] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %.loc9_36.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.Self.2: %A_where.type.1 = bind_symbolic_name .Self [symbolic = constants.%.Self.2]
// CHECK:STDOUT:     %.Self.ref.loc9_48: %A_where.type.1 = name_ref .Self, %.Self.2 [symbolic = constants.%.Self.2]
// CHECK:STDOUT:     %C.ref: %assoc_type = name_ref C, @A.%assoc1 [template = constants.%assoc1]
// CHECK:STDOUT:     %.Self.as_wit.loc9_48: <witness> = facet_access_witness %.Self.ref.loc9_48 [symbolic = constants.%.Self.as_wit.2]
// CHECK:STDOUT:     %impl.elem1: type = interface_witness_access %.Self.as_wit.loc9_48, element1 [symbolic = constants.%impl.elem1]
// CHECK:STDOUT:     %.loc9_54.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc9_54.2: type = converted %.loc9_54.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc9_42: type = where_expr %.Self.2 [template = constants.%A_where.type.2] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem1, %.loc9_54.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %D.param: %A_where.type.2 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %D.loc9_18.1: %A_where.type.2 = bind_symbolic_name D, 0, %D.param [symbolic = %D.loc9_18.2 (constants.%D)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A {
// CHECK:STDOUT:   %Self: %A.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %B: type = assoc_const_decl B [template]
// CHECK:STDOUT:   %assoc0: %assoc_type = assoc_entity element0, %B [template = constants.%assoc0]
// CHECK:STDOUT:   %C: type = assoc_const_decl C [template]
// CHECK:STDOUT:   %assoc1: %assoc_type = assoc_entity element1, %C [template = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .B = %assoc0
// CHECK:STDOUT:   .C = %assoc1
// CHECK:STDOUT:   witness = (%B, %C)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @NestedRewrite(%D.loc9_18.1: %A_where.type.2) {
// CHECK:STDOUT:   %D.loc9_18.2: %A_where.type.2 = bind_symbolic_name D, 0 [symbolic = %D.loc9_18.2 (constants.%D)]
// CHECK:STDOUT:   %D.patt.loc9_18.2: %A_where.type.2 = symbolic_binding_pattern D, 0 [symbolic = %D.patt.loc9_18.2 (constants.%D.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%D.param_patt: %A_where.type.2);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @NestedRewrite(constants.%D) {
// CHECK:STDOUT:   %D.loc9_18.2 => constants.%D
// CHECK:STDOUT:   %D.patt.loc9_18.2 => constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- repeated_rewrite.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %E.type: type = facet_type <@E> [template]
// CHECK:STDOUT:   %Self: %E.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %assoc_type: type = assoc_entity_type %E.type, type [template]
// CHECK:STDOUT:   %assoc0: %assoc_type = assoc_entity element0, @E.%F [template]
// CHECK:STDOUT:   %.Self: %E.type = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %.Self.as_wit: <witness> = facet_access_witness %.Self [symbolic]
// CHECK:STDOUT:   %impl.elem0: type = interface_witness_access %.Self.as_wit, element0 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %E_where.type: type = facet_type <@E where %impl.elem0 = %i32> [template]
// CHECK:STDOUT:   %G: %E_where.type = bind_symbolic_name G, 0 [symbolic]
// CHECK:STDOUT:   %G.patt: %E_where.type = symbolic_binding_pattern G, 0 [symbolic]
// CHECK:STDOUT:   %OneRewrite.type: type = fn_type @OneRewrite [template]
// CHECK:STDOUT:   %OneRewrite: %OneRewrite.type = struct_value () [template]
// CHECK:STDOUT:   %H: %E_where.type = bind_symbolic_name H, 0 [symbolic]
// CHECK:STDOUT:   %H.patt: %E_where.type = symbolic_binding_pattern H, 0 [symbolic]
// CHECK:STDOUT:   %RepeatedRewrite.type: type = fn_type @RepeatedRewrite [template]
// CHECK:STDOUT:   %RepeatedRewrite: %RepeatedRewrite.type = struct_value () [template]
// CHECK:STDOUT:   %OneRewrite.specific_fn.1: <specific function> = specific_function %OneRewrite, @OneRewrite(%H) [symbolic]
// CHECK:STDOUT:   %I: %E_where.type = bind_symbolic_name I, 0 [symbolic]
// CHECK:STDOUT:   %I.patt: %E_where.type = symbolic_binding_pattern I, 0 [symbolic]
// CHECK:STDOUT:   %OneRewriteAgain.type: type = fn_type @OneRewriteAgain [template]
// CHECK:STDOUT:   %OneRewriteAgain: %OneRewriteAgain.type = struct_value () [template]
// CHECK:STDOUT:   %RepeatedRewrite.specific_fn: <specific function> = specific_function %RepeatedRewrite, @RepeatedRewrite(%I) [symbolic]
// CHECK:STDOUT:   %OneRewrite.specific_fn.2: <specific function> = specific_function %OneRewrite, @OneRewrite(%I) [symbolic]
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
// CHECK:STDOUT:     .E = %E.decl
// CHECK:STDOUT:     .OneRewrite = %OneRewrite.decl
// CHECK:STDOUT:     .RepeatedRewrite = %RepeatedRewrite.decl
// CHECK:STDOUT:     .OneRewriteAgain = %OneRewriteAgain.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %E.decl: type = interface_decl @E [template = constants.%E.type] {} {}
// CHECK:STDOUT:   %OneRewrite.decl: %OneRewrite.type = fn_decl @OneRewrite [template = constants.%OneRewrite] {
// CHECK:STDOUT:     %G.patt.loc8_15.1: %E_where.type = symbolic_binding_pattern G, 0 [symbolic = %G.patt.loc8_15.2 (constants.%G.patt)]
// CHECK:STDOUT:     %G.param_patt: %E_where.type = value_param_pattern %G.patt.loc8_15.1, runtime_param<invalid> [symbolic = %G.patt.loc8_15.2 (constants.%G.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %E.ref: type = name_ref E, file.%E.decl [template = constants.%E.type]
// CHECK:STDOUT:     %.Self: %E.type = bind_symbolic_name .Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %E.type = name_ref .Self, %.Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %F.ref: %assoc_type = name_ref F, @E.%assoc0 [template = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_wit: <witness> = facet_access_witness %.Self.ref [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem0: type = interface_witness_access %.Self.as_wit, element0 [symbolic = constants.%impl.elem0]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc8_32.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc8_32.2: type = converted %int.make_type_signed, %.loc8_32.1 [template = constants.%i32]
// CHECK:STDOUT:     %.loc8_21: type = where_expr %.Self [template = constants.%E_where.type] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %.loc8_32.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %G.param: %E_where.type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %G.loc8_15.1: %E_where.type = bind_symbolic_name G, 0, %G.param [symbolic = %G.loc8_15.2 (constants.%G)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RepeatedRewrite.decl: %RepeatedRewrite.type = fn_decl @RepeatedRewrite [template = constants.%RepeatedRewrite] {
// CHECK:STDOUT:     %H.patt.loc10_20.1: %E_where.type = symbolic_binding_pattern H, 0 [symbolic = %H.patt.loc10_20.2 (constants.%H.patt)]
// CHECK:STDOUT:     %H.param_patt: %E_where.type = value_param_pattern %H.patt.loc10_20.1, runtime_param<invalid> [symbolic = %H.patt.loc10_20.2 (constants.%H.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %E.ref: type = name_ref E, file.%E.decl [template = constants.%E.type]
// CHECK:STDOUT:     %.Self: %E.type = bind_symbolic_name .Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc10_32: %E.type = name_ref .Self, %.Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %F.ref.loc10_32: %assoc_type = name_ref F, @E.%assoc0 [template = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_wit.loc10_32: <witness> = facet_access_witness %.Self.ref.loc10_32 [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem0.loc10_32: type = interface_witness_access %.Self.as_wit.loc10_32, element0 [symbolic = constants.%impl.elem0]
// CHECK:STDOUT:     %int_32.loc10_37: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc10_37: init type = call constants.%Int(%int_32.loc10_37) [template = constants.%i32]
// CHECK:STDOUT:     %.loc10_37.1: type = value_of_initializer %int.make_type_signed.loc10_37 [template = constants.%i32]
// CHECK:STDOUT:     %.loc10_37.2: type = converted %int.make_type_signed.loc10_37, %.loc10_37.1 [template = constants.%i32]
// CHECK:STDOUT:     %.Self.ref.loc10_45: %E.type = name_ref .Self, %.Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %F.ref.loc10_45: %assoc_type = name_ref F, @E.%assoc0 [template = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_wit.loc10_45: <witness> = facet_access_witness %.Self.ref.loc10_45 [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem0.loc10_45: type = interface_witness_access %.Self.as_wit.loc10_45, element0 [symbolic = constants.%impl.elem0]
// CHECK:STDOUT:     %int_32.loc10_50: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc10_50: init type = call constants.%Int(%int_32.loc10_50) [template = constants.%i32]
// CHECK:STDOUT:     %.loc10_50.1: type = value_of_initializer %int.make_type_signed.loc10_50 [template = constants.%i32]
// CHECK:STDOUT:     %.loc10_50.2: type = converted %int.make_type_signed.loc10_50, %.loc10_50.1 [template = constants.%i32]
// CHECK:STDOUT:     %.loc10_26: type = where_expr %.Self [template = constants.%E_where.type] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc10_32, %.loc10_37.2
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc10_45, %.loc10_50.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %H.param: %E_where.type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %H.loc10_20.1: %E_where.type = bind_symbolic_name H, 0, %H.param [symbolic = %H.loc10_20.2 (constants.%H)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %OneRewriteAgain.decl: %OneRewriteAgain.type = fn_decl @OneRewriteAgain [template = constants.%OneRewriteAgain] {
// CHECK:STDOUT:     %I.patt.loc14_20.1: %E_where.type = symbolic_binding_pattern I, 0 [symbolic = %I.patt.loc14_20.2 (constants.%I.patt)]
// CHECK:STDOUT:     %I.param_patt: %E_where.type = value_param_pattern %I.patt.loc14_20.1, runtime_param<invalid> [symbolic = %I.patt.loc14_20.2 (constants.%I.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %E.ref: type = name_ref E, file.%E.decl [template = constants.%E.type]
// CHECK:STDOUT:     %.Self: %E.type = bind_symbolic_name .Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %E.type = name_ref .Self, %.Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %F.ref: %assoc_type = name_ref F, @E.%assoc0 [template = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_wit: <witness> = facet_access_witness %.Self.ref [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem0: type = interface_witness_access %.Self.as_wit, element0 [symbolic = constants.%impl.elem0]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc14_37.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc14_37.2: type = converted %int.make_type_signed, %.loc14_37.1 [template = constants.%i32]
// CHECK:STDOUT:     %.loc14_26: type = where_expr %.Self [template = constants.%E_where.type] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %.loc14_37.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %I.param: %E_where.type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %I.loc14_20.1: %E_where.type = bind_symbolic_name I, 0, %I.param [symbolic = %I.loc14_20.2 (constants.%I)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @E {
// CHECK:STDOUT:   %Self: %E.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %F: type = assoc_const_decl F [template]
// CHECK:STDOUT:   %assoc0: %assoc_type = assoc_entity element0, %F [template = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%F)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @OneRewrite(%G.loc8_15.1: %E_where.type) {
// CHECK:STDOUT:   %G.loc8_15.2: %E_where.type = bind_symbolic_name G, 0 [symbolic = %G.loc8_15.2 (constants.%G)]
// CHECK:STDOUT:   %G.patt.loc8_15.2: %E_where.type = symbolic_binding_pattern G, 0 [symbolic = %G.patt.loc8_15.2 (constants.%G.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%G.param_patt: %E_where.type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @RepeatedRewrite(%H.loc10_20.1: %E_where.type) {
// CHECK:STDOUT:   %H.loc10_20.2: %E_where.type = bind_symbolic_name H, 0 [symbolic = %H.loc10_20.2 (constants.%H)]
// CHECK:STDOUT:   %H.patt.loc10_20.2: %E_where.type = symbolic_binding_pattern H, 0 [symbolic = %H.patt.loc10_20.2 (constants.%H.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %OneRewrite.specific_fn.loc11_3.2: <specific function> = specific_function constants.%OneRewrite, @OneRewrite(%H.loc10_20.2) [symbolic = %OneRewrite.specific_fn.loc11_3.2 (constants.%OneRewrite.specific_fn.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%H.param_patt: %E_where.type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %OneRewrite.ref: %OneRewrite.type = name_ref OneRewrite, file.%OneRewrite.decl [template = constants.%OneRewrite]
// CHECK:STDOUT:     %H.ref: %E_where.type = name_ref H, %H.loc10_20.1 [symbolic = %H.loc10_20.2 (constants.%H)]
// CHECK:STDOUT:     %OneRewrite.specific_fn.loc11_3.1: <specific function> = specific_function %OneRewrite.ref, @OneRewrite(constants.%H) [symbolic = %OneRewrite.specific_fn.loc11_3.2 (constants.%OneRewrite.specific_fn.1)]
// CHECK:STDOUT:     %OneRewrite.call: init %empty_tuple.type = call %OneRewrite.specific_fn.loc11_3.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @OneRewriteAgain(%I.loc14_20.1: %E_where.type) {
// CHECK:STDOUT:   %I.loc14_20.2: %E_where.type = bind_symbolic_name I, 0 [symbolic = %I.loc14_20.2 (constants.%I)]
// CHECK:STDOUT:   %I.patt.loc14_20.2: %E_where.type = symbolic_binding_pattern I, 0 [symbolic = %I.patt.loc14_20.2 (constants.%I.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %RepeatedRewrite.specific_fn.loc15_3.2: <specific function> = specific_function constants.%RepeatedRewrite, @RepeatedRewrite(%I.loc14_20.2) [symbolic = %RepeatedRewrite.specific_fn.loc15_3.2 (constants.%RepeatedRewrite.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%I.param_patt: %E_where.type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %RepeatedRewrite.ref: %RepeatedRewrite.type = name_ref RepeatedRewrite, file.%RepeatedRewrite.decl [template = constants.%RepeatedRewrite]
// CHECK:STDOUT:     %I.ref: %E_where.type = name_ref I, %I.loc14_20.1 [symbolic = %I.loc14_20.2 (constants.%I)]
// CHECK:STDOUT:     %RepeatedRewrite.specific_fn.loc15_3.1: <specific function> = specific_function %RepeatedRewrite.ref, @RepeatedRewrite(constants.%I) [symbolic = %RepeatedRewrite.specific_fn.loc15_3.2 (constants.%RepeatedRewrite.specific_fn)]
// CHECK:STDOUT:     %RepeatedRewrite.call: init %empty_tuple.type = call %RepeatedRewrite.specific_fn.loc15_3.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OneRewrite(constants.%G) {
// CHECK:STDOUT:   %G.loc8_15.2 => constants.%G
// CHECK:STDOUT:   %G.patt.loc8_15.2 => constants.%G
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @RepeatedRewrite(constants.%H) {
// CHECK:STDOUT:   %H.loc10_20.2 => constants.%H
// CHECK:STDOUT:   %H.patt.loc10_20.2 => constants.%H
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OneRewrite(constants.%H) {
// CHECK:STDOUT:   %G.loc8_15.2 => constants.%H
// CHECK:STDOUT:   %G.patt.loc8_15.2 => constants.%H
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OneRewrite(@RepeatedRewrite.%H.loc10_20.2) {
// CHECK:STDOUT:   %G.loc8_15.2 => constants.%H
// CHECK:STDOUT:   %G.patt.loc8_15.2 => constants.%H
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OneRewriteAgain(constants.%I) {
// CHECK:STDOUT:   %I.loc14_20.2 => constants.%I
// CHECK:STDOUT:   %I.patt.loc14_20.2 => constants.%I
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @RepeatedRewrite(constants.%I) {
// CHECK:STDOUT:   %H.loc10_20.2 => constants.%I
// CHECK:STDOUT:   %H.patt.loc10_20.2 => constants.%I
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %OneRewrite.specific_fn.loc11_3.2 => constants.%OneRewrite.specific_fn.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @RepeatedRewrite(@OneRewriteAgain.%I.loc14_20.2) {
// CHECK:STDOUT:   %H.loc10_20.2 => constants.%I
// CHECK:STDOUT:   %H.patt.loc10_20.2 => constants.%I
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OneRewrite(constants.%I) {
// CHECK:STDOUT:   %G.loc8_15.2 => constants.%I
// CHECK:STDOUT:   %G.patt.loc8_15.2 => constants.%I
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- rewrites_reordered.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [template]
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %assoc_type: type = assoc_entity_type %J.type, type [template]
// CHECK:STDOUT:   %assoc0: %assoc_type = assoc_entity element0, @J.%K [template]
// CHECK:STDOUT:   %assoc1: %assoc_type = assoc_entity element1, @J.%L [template]
// CHECK:STDOUT:   %.Self: %J.type = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %.Self.as_wit: <witness> = facet_access_witness %.Self [symbolic]
// CHECK:STDOUT:   %impl.elem0: type = interface_witness_access %.Self.as_wit, element0 [symbolic]
// CHECK:STDOUT:   %impl.elem1: type = interface_witness_access %.Self.as_wit, element1 [symbolic]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [template]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [template]
// CHECK:STDOUT:   %J_where.type: type = facet_type <@J where %impl.elem1 = bool and %impl.elem0 = %empty_tuple.type> [template]
// CHECK:STDOUT:   %M: %J_where.type = bind_symbolic_name M, 0 [symbolic]
// CHECK:STDOUT:   %M.patt: %J_where.type = symbolic_binding_pattern M, 0 [symbolic]
// CHECK:STDOUT:   %Alphabetical.type: type = fn_type @Alphabetical [template]
// CHECK:STDOUT:   %Alphabetical: %Alphabetical.type = struct_value () [template]
// CHECK:STDOUT:   %N: %J_where.type = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt: %J_where.type = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %Reversed.type: type = fn_type @Reversed [template]
// CHECK:STDOUT:   %Reversed: %Reversed.type = struct_value () [template]
// CHECK:STDOUT:   %Alphabetical.specific_fn: <specific function> = specific_function %Alphabetical, @Alphabetical(%N) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Bool = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .Alphabetical = %Alphabetical.decl
// CHECK:STDOUT:     .Reversed = %Reversed.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [template = constants.%J.type] {} {}
// CHECK:STDOUT:   %Alphabetical.decl: %Alphabetical.type = fn_decl @Alphabetical [template = constants.%Alphabetical] {
// CHECK:STDOUT:     %M.patt.loc9_17.1: %J_where.type = symbolic_binding_pattern M, 0 [symbolic = %M.patt.loc9_17.2 (constants.%M.patt)]
// CHECK:STDOUT:     %M.param_patt: %J_where.type = value_param_pattern %M.patt.loc9_17.1, runtime_param<invalid> [symbolic = %M.patt.loc9_17.2 (constants.%M.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [template = constants.%J.type]
// CHECK:STDOUT:     %.Self: %J.type = bind_symbolic_name .Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc9_29: %J.type = name_ref .Self, %.Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %K.ref: %assoc_type = name_ref K, @J.%assoc0 [template = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_wit.loc9_29: <witness> = facet_access_witness %.Self.ref.loc9_29 [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem0: type = interface_witness_access %.Self.as_wit.loc9_29, element0 [symbolic = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc9_35.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc9_35.2: type = converted %.loc9_35.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.Self.ref.loc9_41: %J.type = name_ref .Self, %.Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %L.ref: %assoc_type = name_ref L, @J.%assoc1 [template = constants.%assoc1]
// CHECK:STDOUT:     %.Self.as_wit.loc9_41: <witness> = facet_access_witness %.Self.ref.loc9_41 [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem1: type = interface_witness_access %.Self.as_wit.loc9_41, element1 [symbolic = constants.%impl.elem1]
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc9_46.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc9_46.2: type = converted %bool.make_type, %.loc9_46.1 [template = bool]
// CHECK:STDOUT:     %.loc9_23: type = where_expr %.Self [template = constants.%J_where.type] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %.loc9_35.2
// CHECK:STDOUT:       requirement_rewrite %impl.elem1, %.loc9_46.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %M.param: %J_where.type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %M.loc9_17.1: %J_where.type = bind_symbolic_name M, 0, %M.param [symbolic = %M.loc9_17.2 (constants.%M)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Reversed.decl: %Reversed.type = fn_decl @Reversed [template = constants.%Reversed] {
// CHECK:STDOUT:     %N.patt.loc11_13.1: %J_where.type = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc11_13.2 (constants.%N.patt)]
// CHECK:STDOUT:     %N.param_patt: %J_where.type = value_param_pattern %N.patt.loc11_13.1, runtime_param<invalid> [symbolic = %N.patt.loc11_13.2 (constants.%N.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [template = constants.%J.type]
// CHECK:STDOUT:     %.Self: %J.type = bind_symbolic_name .Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc11_25: %J.type = name_ref .Self, %.Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %L.ref: %assoc_type = name_ref L, @J.%assoc1 [template = constants.%assoc1]
// CHECK:STDOUT:     %.Self.as_wit.loc11_25: <witness> = facet_access_witness %.Self.ref.loc11_25 [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem1: type = interface_witness_access %.Self.as_wit.loc11_25, element1 [symbolic = constants.%impl.elem1]
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc11_30.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc11_30.2: type = converted %bool.make_type, %.loc11_30.1 [template = bool]
// CHECK:STDOUT:     %.Self.ref.loc11_39: %J.type = name_ref .Self, %.Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %K.ref: %assoc_type = name_ref K, @J.%assoc0 [template = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_wit.loc11_39: <witness> = facet_access_witness %.Self.ref.loc11_39 [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem0: type = interface_witness_access %.Self.as_wit.loc11_39, element0 [symbolic = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc11_45.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc11_45.2: type = converted %.loc11_45.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc11_19: type = where_expr %.Self [template = constants.%J_where.type] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem1, %.loc11_30.2
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %.loc11_45.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.param: %J_where.type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %N.loc11_13.1: %J_where.type = bind_symbolic_name N, 0, %N.param [symbolic = %N.loc11_13.2 (constants.%N)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %K: type = assoc_const_decl K [template]
// CHECK:STDOUT:   %assoc0: %assoc_type = assoc_entity element0, %K [template = constants.%assoc0]
// CHECK:STDOUT:   %L: type = assoc_const_decl L [template]
// CHECK:STDOUT:   %assoc1: %assoc_type = assoc_entity element1, %L [template = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .K = %assoc0
// CHECK:STDOUT:   .L = %assoc1
// CHECK:STDOUT:   witness = (%K, %L)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Alphabetical(%M.loc9_17.1: %J_where.type) {
// CHECK:STDOUT:   %M.loc9_17.2: %J_where.type = bind_symbolic_name M, 0 [symbolic = %M.loc9_17.2 (constants.%M)]
// CHECK:STDOUT:   %M.patt.loc9_17.2: %J_where.type = symbolic_binding_pattern M, 0 [symbolic = %M.patt.loc9_17.2 (constants.%M.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%M.param_patt: %J_where.type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Reversed(%N.loc11_13.1: %J_where.type) {
// CHECK:STDOUT:   %N.loc11_13.2: %J_where.type = bind_symbolic_name N, 0 [symbolic = %N.loc11_13.2 (constants.%N)]
// CHECK:STDOUT:   %N.patt.loc11_13.2: %J_where.type = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc11_13.2 (constants.%N.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Alphabetical.specific_fn.loc12_3.2: <specific function> = specific_function constants.%Alphabetical, @Alphabetical(%N.loc11_13.2) [symbolic = %Alphabetical.specific_fn.loc12_3.2 (constants.%Alphabetical.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%N.param_patt: %J_where.type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Alphabetical.ref: %Alphabetical.type = name_ref Alphabetical, file.%Alphabetical.decl [template = constants.%Alphabetical]
// CHECK:STDOUT:     %N.ref: %J_where.type = name_ref N, %N.loc11_13.1 [symbolic = %N.loc11_13.2 (constants.%N)]
// CHECK:STDOUT:     %Alphabetical.specific_fn.loc12_3.1: <specific function> = specific_function %Alphabetical.ref, @Alphabetical(constants.%N) [symbolic = %Alphabetical.specific_fn.loc12_3.2 (constants.%Alphabetical.specific_fn)]
// CHECK:STDOUT:     %Alphabetical.call: init %empty_tuple.type = call %Alphabetical.specific_fn.loc12_3.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Alphabetical(constants.%M) {
// CHECK:STDOUT:   %M.loc9_17.2 => constants.%M
// CHECK:STDOUT:   %M.patt.loc9_17.2 => constants.%M
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Reversed(constants.%N) {
// CHECK:STDOUT:   %N.loc11_13.2 => constants.%N
// CHECK:STDOUT:   %N.patt.loc11_13.2 => constants.%N
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Alphabetical(constants.%N) {
// CHECK:STDOUT:   %M.loc9_17.2 => constants.%N
// CHECK:STDOUT:   %M.patt.loc9_17.2 => constants.%N
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Alphabetical(@Reversed.%N.loc11_13.2) {
// CHECK:STDOUT:   %M.loc9_17.2 => constants.%N
// CHECK:STDOUT:   %M.patt.loc9_17.2 => constants.%N
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_rewrites_mismatch_right.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %O.type: type = facet_type <@O> [template]
// CHECK:STDOUT:   %Self.1: %O.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %assoc_type: type = assoc_entity_type %O.type, type [template]
// CHECK:STDOUT:   %assoc0.1: %assoc_type = assoc_entity element0, @O.%P [template]
// CHECK:STDOUT:   %.Self: %O.type = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %.Self.as_wit: <witness> = facet_access_witness %.Self [symbolic]
// CHECK:STDOUT:   %impl.elem0: type = interface_witness_access %.Self.as_wit, element0 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %O_where.type.1: type = facet_type <@O where %impl.elem0 = %i32> [template]
// CHECK:STDOUT:   %Q: %O_where.type.1 = bind_symbolic_name Q, 0 [symbolic]
// CHECK:STDOUT:   %Q.patt: %O_where.type.1 = symbolic_binding_pattern Q, 0 [symbolic]
// CHECK:STDOUT:   %WithInteger.type: type = fn_type @WithInteger [template]
// CHECK:STDOUT:   %WithInteger: %WithInteger.type = struct_value () [template]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [template]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [template]
// CHECK:STDOUT:   %O_where.type.2: type = facet_type <@O where %impl.elem0 = bool> [template]
// CHECK:STDOUT:   %R: %O_where.type.2 = bind_symbolic_name R, 0 [symbolic]
// CHECK:STDOUT:   %R.patt: %O_where.type.2 = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %WithBool.type: type = fn_type @WithBool [template]
// CHECK:STDOUT:   %WithBool: %WithBool.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .Bool = %import_ref.2
// CHECK:STDOUT:     .ImplicitAs = %import_ref.3
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .O = %O.decl
// CHECK:STDOUT:     .WithInteger = %WithInteger.decl
// CHECK:STDOUT:     .WithBool = %WithBool.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %O.decl: type = interface_decl @O [template = constants.%O.type] {} {}
// CHECK:STDOUT:   %WithInteger.decl: %WithInteger.type = fn_decl @WithInteger [template = constants.%WithInteger] {
// CHECK:STDOUT:     %Q.patt.loc8_16.1: %O_where.type.1 = symbolic_binding_pattern Q, 0 [symbolic = %Q.patt.loc8_16.2 (constants.%Q.patt)]
// CHECK:STDOUT:     %Q.param_patt: %O_where.type.1 = value_param_pattern %Q.patt.loc8_16.1, runtime_param<invalid> [symbolic = %Q.patt.loc8_16.2 (constants.%Q.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %O.ref: type = name_ref O, file.%O.decl [template = constants.%O.type]
// CHECK:STDOUT:     %.Self: %O.type = bind_symbolic_name .Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %O.type = name_ref .Self, %.Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %P.ref: %assoc_type = name_ref P, @O.%assoc0 [template = constants.%assoc0.1]
// CHECK:STDOUT:     %.Self.as_wit: <witness> = facet_access_witness %.Self.ref [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem0: type = interface_witness_access %.Self.as_wit, element0 [symbolic = constants.%impl.elem0]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc8_33.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc8_33.2: type = converted %int.make_type_signed, %.loc8_33.1 [template = constants.%i32]
// CHECK:STDOUT:     %.loc8_22: type = where_expr %.Self [template = constants.%O_where.type.1] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %.loc8_33.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Q.param: %O_where.type.1 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %Q.loc8_16.1: %O_where.type.1 = bind_symbolic_name Q, 0, %Q.param [symbolic = %Q.loc8_16.2 (constants.%Q)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %WithBool.decl: %WithBool.type = fn_decl @WithBool [template = constants.%WithBool] {
// CHECK:STDOUT:     %R.patt.loc10_13.1: %O_where.type.2 = symbolic_binding_pattern R, 0 [symbolic = %R.patt.loc10_13.2 (constants.%R.patt)]
// CHECK:STDOUT:     %R.param_patt: %O_where.type.2 = value_param_pattern %R.patt.loc10_13.1, runtime_param<invalid> [symbolic = %R.patt.loc10_13.2 (constants.%R.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %O.ref: type = name_ref O, file.%O.decl [template = constants.%O.type]
// CHECK:STDOUT:     %.Self: %O.type = bind_symbolic_name .Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %O.type = name_ref .Self, %.Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %P.ref: %assoc_type = name_ref P, @O.%assoc0 [template = constants.%assoc0.1]
// CHECK:STDOUT:     %.Self.as_wit: <witness> = facet_access_witness %.Self.ref [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem0: type = interface_witness_access %.Self.as_wit, element0 [symbolic = constants.%impl.elem0]
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc10_30.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc10_30.2: type = converted %bool.make_type, %.loc10_30.1 [template = bool]
// CHECK:STDOUT:     %.loc10_19: type = where_expr %.Self [template = constants.%O_where.type.2] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %.loc10_30.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %R.param: %O_where.type.2 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %R.loc10_13.1: %O_where.type.2 = bind_symbolic_name R, 0, %R.param [symbolic = %R.loc10_13.2 (constants.%R)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @O {
// CHECK:STDOUT:   %Self: %O.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.1]
// CHECK:STDOUT:   %P: type = assoc_const_decl P [template]
// CHECK:STDOUT:   %assoc0: %assoc_type = assoc_entity element0, %P [template = constants.%assoc0.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .P = %assoc0
// CHECK:STDOUT:   witness = (%P)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @WithInteger(%Q.loc8_16.1: %O_where.type.1) {
// CHECK:STDOUT:   %Q.loc8_16.2: %O_where.type.1 = bind_symbolic_name Q, 0 [symbolic = %Q.loc8_16.2 (constants.%Q)]
// CHECK:STDOUT:   %Q.patt.loc8_16.2: %O_where.type.1 = symbolic_binding_pattern Q, 0 [symbolic = %Q.patt.loc8_16.2 (constants.%Q.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%Q.param_patt: %O_where.type.1) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @WithBool(%R.loc10_13.1: %O_where.type.2) {
// CHECK:STDOUT:   %R.loc10_13.2: %O_where.type.2 = bind_symbolic_name R, 0 [symbolic = %R.loc10_13.2 (constants.%R)]
// CHECK:STDOUT:   %R.patt.loc10_13.2: %O_where.type.2 = symbolic_binding_pattern R, 0 [symbolic = %R.patt.loc10_13.2 (constants.%R.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%R.param_patt: %O_where.type.2) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %WithInteger.ref: %WithInteger.type = name_ref WithInteger, file.%WithInteger.decl [template = constants.%WithInteger]
// CHECK:STDOUT:     %R.ref: %O_where.type.2 = name_ref R, %R.loc10_13.1 [symbolic = %R.loc10_13.2 (constants.%R)]
// CHECK:STDOUT:     %.loc21: %O_where.type.1 = converted %R.ref, <error> [template = <error>]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WithInteger(constants.%Q) {
// CHECK:STDOUT:   %Q.loc8_16.2 => constants.%Q
// CHECK:STDOUT:   %Q.patt.loc8_16.2 => constants.%Q
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WithBool(constants.%R) {
// CHECK:STDOUT:   %R.loc10_13.2 => constants.%R
// CHECK:STDOUT:   %R.patt.loc10_13.2 => constants.%R
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_rewrites_mismatch_left.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %S.type: type = facet_type <@S> [template]
// CHECK:STDOUT:   %Self.1: %S.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %assoc_type: type = assoc_entity_type %S.type, type [template]
// CHECK:STDOUT:   %assoc0.1: %assoc_type = assoc_entity element0, @S.%T [template]
// CHECK:STDOUT:   %assoc1: %assoc_type = assoc_entity element1, @S.%U [template]
// CHECK:STDOUT:   %.Self: %S.type = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %.Self.as_wit: <witness> = facet_access_witness %.Self [symbolic]
// CHECK:STDOUT:   %impl.elem0: type = interface_witness_access %.Self.as_wit, element0 [symbolic]
// CHECK:STDOUT:   %S_where.type.1: type = facet_type <@S where %impl.elem0 = %empty_tuple.type> [template]
// CHECK:STDOUT:   %V: %S_where.type.1 = bind_symbolic_name V, 0 [symbolic]
// CHECK:STDOUT:   %V.patt: %S_where.type.1 = symbolic_binding_pattern V, 0 [symbolic]
// CHECK:STDOUT:   %WithT.type: type = fn_type @WithT [template]
// CHECK:STDOUT:   %WithT: %WithT.type = struct_value () [template]
// CHECK:STDOUT:   %impl.elem1: type = interface_witness_access %.Self.as_wit, element1 [symbolic]
// CHECK:STDOUT:   %S_where.type.2: type = facet_type <@S where %impl.elem1 = %empty_tuple.type> [template]
// CHECK:STDOUT:   %W: %S_where.type.2 = bind_symbolic_name W, 0 [symbolic]
// CHECK:STDOUT:   %W.patt: %S_where.type.2 = symbolic_binding_pattern W, 0 [symbolic]
// CHECK:STDOUT:   %WithU.type: type = fn_type @WithU [template]
// CHECK:STDOUT:   %WithU: %WithU.type = struct_value () [template]
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
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     .WithT = %WithT.decl
// CHECK:STDOUT:     .WithU = %WithU.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %S.decl: type = interface_decl @S [template = constants.%S.type] {} {}
// CHECK:STDOUT:   %WithT.decl: %WithT.type = fn_decl @WithT [template = constants.%WithT] {
// CHECK:STDOUT:     %V.patt.loc9_10.1: %S_where.type.1 = symbolic_binding_pattern V, 0 [symbolic = %V.patt.loc9_10.2 (constants.%V.patt)]
// CHECK:STDOUT:     %V.param_patt: %S_where.type.1 = value_param_pattern %V.patt.loc9_10.1, runtime_param<invalid> [symbolic = %V.patt.loc9_10.2 (constants.%V.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %S.ref: type = name_ref S, file.%S.decl [template = constants.%S.type]
// CHECK:STDOUT:     %.Self: %S.type = bind_symbolic_name .Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %S.type = name_ref .Self, %.Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %T.ref: %assoc_type = name_ref T, @S.%assoc0 [template = constants.%assoc0.1]
// CHECK:STDOUT:     %.Self.as_wit: <witness> = facet_access_witness %.Self.ref [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem0: type = interface_witness_access %.Self.as_wit, element0 [symbolic = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc9_28.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc9_28.2: type = converted %.loc9_28.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc9_16: type = where_expr %.Self [template = constants.%S_where.type.1] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %.loc9_28.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %V.param: %S_where.type.1 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %V.loc9_10.1: %S_where.type.1 = bind_symbolic_name V, 0, %V.param [symbolic = %V.loc9_10.2 (constants.%V)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %WithU.decl: %WithU.type = fn_decl @WithU [template = constants.%WithU] {
// CHECK:STDOUT:     %W.patt.loc11_10.1: %S_where.type.2 = symbolic_binding_pattern W, 0 [symbolic = %W.patt.loc11_10.2 (constants.%W.patt)]
// CHECK:STDOUT:     %W.param_patt: %S_where.type.2 = value_param_pattern %W.patt.loc11_10.1, runtime_param<invalid> [symbolic = %W.patt.loc11_10.2 (constants.%W.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %S.ref: type = name_ref S, file.%S.decl [template = constants.%S.type]
// CHECK:STDOUT:     %.Self: %S.type = bind_symbolic_name .Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %S.type = name_ref .Self, %.Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %U.ref: %assoc_type = name_ref U, @S.%assoc1 [template = constants.%assoc1]
// CHECK:STDOUT:     %.Self.as_wit: <witness> = facet_access_witness %.Self.ref [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem1: type = interface_witness_access %.Self.as_wit, element1 [symbolic = constants.%impl.elem1]
// CHECK:STDOUT:     %.loc11_28.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc11_28.2: type = converted %.loc11_28.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc11_16: type = where_expr %.Self [template = constants.%S_where.type.2] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem1, %.loc11_28.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %W.param: %S_where.type.2 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %W.loc11_10.1: %S_where.type.2 = bind_symbolic_name W, 0, %W.param [symbolic = %W.loc11_10.2 (constants.%W)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @S {
// CHECK:STDOUT:   %Self: %S.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.1]
// CHECK:STDOUT:   %T: type = assoc_const_decl T [template]
// CHECK:STDOUT:   %assoc0: %assoc_type = assoc_entity element0, %T [template = constants.%assoc0.1]
// CHECK:STDOUT:   %U: type = assoc_const_decl U [template]
// CHECK:STDOUT:   %assoc1: %assoc_type = assoc_entity element1, %U [template = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .T = %assoc0
// CHECK:STDOUT:   .U = %assoc1
// CHECK:STDOUT:   witness = (%T, %U)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @WithT(%V.loc9_10.1: %S_where.type.1) {
// CHECK:STDOUT:   %V.loc9_10.2: %S_where.type.1 = bind_symbolic_name V, 0 [symbolic = %V.loc9_10.2 (constants.%V)]
// CHECK:STDOUT:   %V.patt.loc9_10.2: %S_where.type.1 = symbolic_binding_pattern V, 0 [symbolic = %V.patt.loc9_10.2 (constants.%V.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%V.param_patt: %S_where.type.1) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @WithU(%W.loc11_10.1: %S_where.type.2) {
// CHECK:STDOUT:   %W.loc11_10.2: %S_where.type.2 = bind_symbolic_name W, 0 [symbolic = %W.loc11_10.2 (constants.%W)]
// CHECK:STDOUT:   %W.patt.loc11_10.2: %S_where.type.2 = symbolic_binding_pattern W, 0 [symbolic = %W.patt.loc11_10.2 (constants.%W.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%W.param_patt: %S_where.type.2) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %WithT.ref: %WithT.type = name_ref WithT, file.%WithT.decl [template = constants.%WithT]
// CHECK:STDOUT:     %W.ref: %S_where.type.2 = name_ref W, %W.loc11_10.1 [symbolic = %W.loc11_10.2 (constants.%W)]
// CHECK:STDOUT:     %.loc22: %S_where.type.1 = converted %W.ref, <error> [template = <error>]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WithT(constants.%V) {
// CHECK:STDOUT:   %V.loc9_10.2 => constants.%V
// CHECK:STDOUT:   %V.patt.loc9_10.2 => constants.%V
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WithU(constants.%W) {
// CHECK:STDOUT:   %W.loc11_10.2 => constants.%W
// CHECK:STDOUT:   %W.patt.loc11_10.2 => constants.%W
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_rewrites.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Calls.type: type = fn_type @Calls [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Calls: %Calls.type = struct_value () [template]
// CHECK:STDOUT:   %Equal.type: type = fn_type @Equal [template]
// CHECK:STDOUT:   %Equal: %Equal.type = struct_value () [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %N.type: type = facet_type <@N> [template]
// CHECK:STDOUT:   %.Self.1: %N.type = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %.Self.as_wit.1: <witness> = facet_access_witness %.Self.1 [symbolic]
// CHECK:STDOUT:   %impl.elem0.1: type = interface_witness_access %.Self.as_wit.1, element0 [symbolic]
// CHECK:STDOUT:   %N_where.type: type = facet_type <@N where %impl.elem0.1 = %empty_struct_type> [template]
// CHECK:STDOUT:   %T: %N_where.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: %N_where.type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [template]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [template]
// CHECK:STDOUT:   %NestedRewrite.type: type = fn_type @NestedRewrite [template]
// CHECK:STDOUT:   %NestedRewrite: %NestedRewrite.type = struct_value () [template]
// CHECK:STDOUT:   %A.type: type = facet_type <@A> [template]
// CHECK:STDOUT:   %.Self.2: %A.type = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %.Self.as_wit.2: <witness> = facet_access_witness %.Self.2 [symbolic]
// CHECK:STDOUT:   %impl.elem0.2: type = interface_witness_access %.Self.as_wit.2, element0 [symbolic]
// CHECK:STDOUT:   %A_where.type.1: type = facet_type <@A where %impl.elem0.2 = bool> [template]
// CHECK:STDOUT:   %.Self.3: %A_where.type.1 = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %.Self.as_wit.3: <witness> = facet_access_witness %.Self.3 [symbolic]
// CHECK:STDOUT:   %impl.elem1: type = interface_witness_access %.Self.as_wit.3, element1 [symbolic]
// CHECK:STDOUT:   %A_where.type.2: type = facet_type <@A where %impl.elem1 = %empty_tuple.type and %impl.elem0.2 = bool> [template]
// CHECK:STDOUT:   %D: %A_where.type.2 = bind_symbolic_name D, 0 [symbolic]
// CHECK:STDOUT:   %D.patt: %A_where.type.2 = symbolic_binding_pattern D, 0 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1 = import_ref Main//equal_constraint, inst+3, unloaded
// CHECK:STDOUT:   %import_ref.2: %Equal.type = import_ref Main//equal_constraint, inst+33, loaded [template = constants.%Equal]
// CHECK:STDOUT:   %import_ref.3 = import_ref Main//nested_rewrites, inst+3, unloaded
// CHECK:STDOUT:   %import_ref.4: %NestedRewrite.type = import_ref Main//nested_rewrites, inst+55, loaded [template = constants.%NestedRewrite]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Bool = %import_ref.8
// CHECK:STDOUT:     .ImplicitAs = %import_ref.9
// CHECK:STDOUT:     .Int = %import_ref.50
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.5 = import_ref Main//equal_constraint, inst+5, unloaded
// CHECK:STDOUT:   %import_ref.6 = import_ref Main//equal_constraint, inst+9, unloaded
// CHECK:STDOUT:   %import_ref.7 = import_ref Main//equal_constraint, inst+7, unloaded
// CHECK:STDOUT:   %import_ref.45 = import_ref Main//nested_rewrites, inst+5, unloaded
// CHECK:STDOUT:   %import_ref.46 = import_ref Main//nested_rewrites, inst+9, unloaded
// CHECK:STDOUT:   %import_ref.47 = import_ref Main//nested_rewrites, inst+12, unloaded
// CHECK:STDOUT:   %import_ref.48 = import_ref Main//nested_rewrites, inst+7, unloaded
// CHECK:STDOUT:   %import_ref.49 = import_ref Main//nested_rewrites, inst+11, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .N = imports.%import_ref.1
// CHECK:STDOUT:     .Equal = imports.%import_ref.2
// CHECK:STDOUT:     .A = imports.%import_ref.3
// CHECK:STDOUT:     .NestedRewrite = imports.%import_ref.4
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Calls = %Calls.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %Calls.decl: %Calls.type = fn_decl @Calls [template = constants.%Calls] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @N {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.5
// CHECK:STDOUT:   .P = imports.%import_ref.6
// CHECK:STDOUT:   witness = (imports.%import_ref.7)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.45
// CHECK:STDOUT:   .B = imports.%import_ref.46
// CHECK:STDOUT:   .C = imports.%import_ref.47
// CHECK:STDOUT:   witness = (imports.%import_ref.48, imports.%import_ref.49)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Calls() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Equal.ref: %Equal.type = name_ref Equal, imports.%import_ref.2 [template = constants.%Equal]
// CHECK:STDOUT:   %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:   %.loc19: %N_where.type = converted %bool.make_type, <error> [template = <error>]
// CHECK:STDOUT:   %NestedRewrite.ref: %NestedRewrite.type = name_ref NestedRewrite, imports.%import_ref.4 [template = constants.%NestedRewrite]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc32: %A_where.type.2 = converted %int.make_type_signed, <error> [template = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Equal(constants.%T: %N_where.type) {
// CHECK:STDOUT:   %T: %N_where.type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: %N_where.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.param_patt: %N_where.type);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @NestedRewrite(constants.%D: %A_where.type.2) {
// CHECK:STDOUT:   %D: %A_where.type.2 = bind_symbolic_name D, 0 [symbolic = %D (constants.%D)]
// CHECK:STDOUT:   %D.patt: %A_where.type.2 = symbolic_binding_pattern D, 0 [symbolic = %D.patt (constants.%D.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%D.param_patt: %A_where.type.2);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Equal(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @NestedRewrite(constants.%D) {
// CHECK:STDOUT:   %D => constants.%D
// CHECK:STDOUT:   %D.patt => constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_check_rewrite_constraints.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [template]
// CHECK:STDOUT:   %Self.1: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %assoc_type: type = assoc_entity_type %I.type, type [template]
// CHECK:STDOUT:   %assoc0.1: %assoc_type = assoc_entity element0, @I.%Member [template]
// CHECK:STDOUT:   %.Self: %I.type = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %.Self.as_wit: <witness> = facet_access_witness %.Self [symbolic]
// CHECK:STDOUT:   %impl.elem0: type = interface_witness_access %.Self.as_wit, element0 [symbolic]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %X.patt: <error> = symbolic_binding_pattern X, 0 [symbolic]
// CHECK:STDOUT:   %RewriteTypeMismatch.type: type = fn_type @RewriteTypeMismatch [template]
// CHECK:STDOUT:   %RewriteTypeMismatch: %RewriteTypeMismatch.type = struct_value () [template]
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
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .RewriteTypeMismatch = %RewriteTypeMismatch.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [template = constants.%I.type] {} {}
// CHECK:STDOUT:   %RewriteTypeMismatch.decl: %RewriteTypeMismatch.type = fn_decl @RewriteTypeMismatch [template = constants.%RewriteTypeMismatch] {
// CHECK:STDOUT:     %X.patt.loc16_24.1: <error> = symbolic_binding_pattern X, 0 [symbolic = %X.patt.loc16_24.2 (constants.%X.patt)]
// CHECK:STDOUT:     %X.param_patt: <error> = value_param_pattern %X.patt.loc16_24.1, runtime_param<invalid> [symbolic = %X.patt.loc16_24.2 (constants.%X.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [template = constants.%I.type]
// CHECK:STDOUT:     %.Self: %I.type = bind_symbolic_name .Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %I.type = name_ref .Self, %.Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %Member.ref: %assoc_type = name_ref Member, @I.%assoc0 [template = constants.%assoc0.1]
// CHECK:STDOUT:     %.Self.as_wit: <witness> = facet_access_witness %.Self.ref [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem0: type = interface_witness_access %.Self.as_wit, element0 [symbolic = constants.%impl.elem0]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2]
// CHECK:STDOUT:     %.loc16_46: type = converted %int_2, <error> [template = <error>]
// CHECK:STDOUT:     %.loc16_30: type = where_expr %.Self [template = <error>] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %X.param: <error> = value_param runtime_param<invalid>
// CHECK:STDOUT:     %X: <error> = bind_symbolic_name X, 0, %X.param [template = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.1]
// CHECK:STDOUT:   %Member: type = assoc_const_decl Member [template]
// CHECK:STDOUT:   %assoc0: %assoc_type = assoc_entity element0, %Member [template = constants.%assoc0.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Member = %assoc0
// CHECK:STDOUT:   witness = (%Member)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @RewriteTypeMismatch(%X: <error>) {
// CHECK:STDOUT:   %X.patt.loc16_24.2: <error> = symbolic_binding_pattern X, 0 [symbolic = %X.patt.loc16_24.2 (constants.%X.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%X.param_patt: <error>);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @RewriteTypeMismatch(<error>) {
// CHECK:STDOUT:   %X.patt.loc16_24.2 => <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_let.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = facet_type <@A> [template]
// CHECK:STDOUT:   %Self.1: %A.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %interface.1: <witness> = interface_witness () [template]
// CHECK:STDOUT:   %.Self: type = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %type_where: type = facet_type <type where TODO> [template]
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
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .B = @__global_init.%B
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = interface_decl @A [template = constants.%A.type] {} {}
// CHECK:STDOUT:   %D.decl: type = class_decl @D [template = constants.%D] {} {}
// CHECK:STDOUT:   impl_decl @impl.1 [template] {} {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [template = constants.%A.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.Self: type = bind_symbolic_name .Self [symbolic = constants.%.Self]
// CHECK:STDOUT:   %.Self.ref: type = name_ref .Self, %.Self [symbolic = constants.%.Self]
// CHECK:STDOUT:   %A.ref: type = name_ref A, %A.decl [template = constants.%A.type]
// CHECK:STDOUT:   %.loc15: type = where_expr %.Self [template = constants.%type_where] {
// CHECK:STDOUT:     requirement_impls %.Self.ref, %A.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A {
// CHECK:STDOUT:   %Self: %A.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: %D.ref as %A.ref {
// CHECK:STDOUT:   %interface: <witness> = interface_witness () [template = constants.%interface.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = %interface
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:   %.loc15: %type_where = converted %D.ref, <error> [template = <error>]
// CHECK:STDOUT:   %B: %type_where = bind_name B, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_type_does_not_implement_where.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %E.type: type = facet_type <@E> [template]
// CHECK:STDOUT:   %Self.1: %E.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %assoc_type: type = assoc_entity_type %E.type, type [template]
// CHECK:STDOUT:   %assoc0.1: %assoc_type = assoc_entity element0, @E.%F [template]
// CHECK:STDOUT:   %assoc1: %assoc_type = assoc_entity element1, @E.%G [template]
// CHECK:STDOUT:   %.Self.1: %E.type = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %.Self.as_wit.1: <witness> = facet_access_witness %.Self.1 [symbolic]
// CHECK:STDOUT:   %impl.elem0: type = interface_witness_access %.Self.as_wit.1, element0 [symbolic]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [template]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [template]
// CHECK:STDOUT:   %impl.elem1.1: type = interface_witness_access %.Self.as_wit.1, element1 [symbolic]
// CHECK:STDOUT:   %E_where.type.1: type = facet_type <@E where %impl.elem1.1 = %empty_tuple.type and %impl.elem0 = bool> [template]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [template]
// CHECK:STDOUT:   %Float.type: type = fn_type @Float [template]
// CHECK:STDOUT:   %Float: %Float.type = struct_value () [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %E_where.type.2: type = facet_type <@E where %impl.elem0 = %empty_struct_type> [template]
// CHECK:STDOUT:   %.Self.2: %E_where.type.2 = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %.Self.as_wit.2: <witness> = facet_access_witness %.Self.2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.2: type = interface_witness_access %.Self.as_wit.2, element1 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %E_where.type.3: type = facet_type <@E where %impl.elem1.2 = %i32 and %impl.elem0 = %empty_struct_type> [template]
// CHECK:STDOUT:   %E_where.type.4: type = facet_type <@E where %impl.elem0 = %impl.elem1.1> [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Bool = %import_ref.1
// CHECK:STDOUT:     .Float = %import_ref.2
// CHECK:STDOUT:     .ImplicitAs = %import_ref.3
// CHECK:STDOUT:     .Int = %import_ref.39
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .E = %E.decl
// CHECK:STDOUT:     .H = @__global_init.%H
// CHECK:STDOUT:     .J = @__global_init.%J
// CHECK:STDOUT:     .K = @__global_init.%K
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %E.decl: type = interface_decl @E [template = constants.%E.type] {} {}
// CHECK:STDOUT:   %E.ref.loc18: type = name_ref E, %E.decl [template = constants.%E.type]
// CHECK:STDOUT:   %.Self.1: %E.type = bind_symbolic_name .Self [symbolic = constants.%.Self.1]
// CHECK:STDOUT:   %.Self.ref.loc18_17: %E.type = name_ref .Self, %.Self.1 [symbolic = constants.%.Self.1]
// CHECK:STDOUT:   %F.ref.loc18: %assoc_type = name_ref F, @E.%assoc0 [template = constants.%assoc0.1]
// CHECK:STDOUT:   %.Self.as_wit.loc18_17: <witness> = facet_access_witness %.Self.ref.loc18_17 [symbolic = constants.%.Self.as_wit.1]
// CHECK:STDOUT:   %impl.elem0.loc18: type = interface_witness_access %.Self.as_wit.loc18_17, element0 [symbolic = constants.%impl.elem0]
// CHECK:STDOUT:   %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:   %.loc18_22.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:   %.loc18_22.2: type = converted %bool.make_type, %.loc18_22.1 [template = bool]
// CHECK:STDOUT:   %.Self.ref.loc18_31: %E.type = name_ref .Self, %.Self.1 [symbolic = constants.%.Self.1]
// CHECK:STDOUT:   %G.ref.loc18: %assoc_type = name_ref G, @E.%assoc1 [template = constants.%assoc1]
// CHECK:STDOUT:   %.Self.as_wit.loc18_31: <witness> = facet_access_witness %.Self.ref.loc18_31 [symbolic = constants.%.Self.as_wit.1]
// CHECK:STDOUT:   %impl.elem1.loc18: type = interface_witness_access %.Self.as_wit.loc18_31, element1 [symbolic = constants.%impl.elem1.1]
// CHECK:STDOUT:   %.loc18_37.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc18_37.2: type = converted %.loc18_37.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   %.loc18_11: type = where_expr %.Self.1 [template = constants.%E_where.type.1] {
// CHECK:STDOUT:     requirement_rewrite %impl.elem0.loc18, %.loc18_22.2
// CHECK:STDOUT:     requirement_rewrite %impl.elem1.loc18, %.loc18_37.2
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %E.ref.loc27: type = name_ref E, %E.decl [template = constants.%E.type]
// CHECK:STDOUT:   %.Self.2: %E.type = bind_symbolic_name .Self [symbolic = constants.%.Self.1]
// CHECK:STDOUT:   %.Self.ref.loc27_18: %E.type = name_ref .Self, %.Self.2 [symbolic = constants.%.Self.1]
// CHECK:STDOUT:   %F.ref.loc27: %assoc_type = name_ref F, @E.%assoc0 [template = constants.%assoc0.1]
// CHECK:STDOUT:   %.Self.as_wit.loc27_18: <witness> = facet_access_witness %.Self.ref.loc27_18 [symbolic = constants.%.Self.as_wit.1]
// CHECK:STDOUT:   %impl.elem0.loc27: type = interface_witness_access %.Self.as_wit.loc27_18, element0 [symbolic = constants.%impl.elem0]
// CHECK:STDOUT:   %.loc27_24.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc27_24.2: type = converted %.loc27_24.1, constants.%empty_struct_type [template = constants.%empty_struct_type]
// CHECK:STDOUT:   %.loc27_12: type = where_expr %.Self.2 [template = constants.%E_where.type.2] {
// CHECK:STDOUT:     requirement_rewrite %impl.elem0.loc27, %.loc27_24.2
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.Self.3: %E_where.type.2 = bind_symbolic_name .Self [symbolic = constants.%.Self.2]
// CHECK:STDOUT:   %.Self.ref.loc27_33: %E_where.type.2 = name_ref .Self, %.Self.3 [symbolic = constants.%.Self.2]
// CHECK:STDOUT:   %G.ref.loc27: %assoc_type = name_ref G, @E.%assoc1 [template = constants.%assoc1]
// CHECK:STDOUT:   %.Self.as_wit.loc27_33: <witness> = facet_access_witness %.Self.ref.loc27_33 [symbolic = constants.%.Self.as_wit.2]
// CHECK:STDOUT:   %impl.elem1.loc27: type = interface_witness_access %.Self.as_wit.loc27_33, element1 [symbolic = constants.%impl.elem1.2]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc27_38.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc27_38.2: type = converted %int.make_type_signed, %.loc27_38.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc27_27: type = where_expr %.Self.3 [template = constants.%E_where.type.3] {
// CHECK:STDOUT:     requirement_rewrite %impl.elem1.loc27, %.loc27_38.2
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %E.ref.loc35: type = name_ref E, %E.decl [template = constants.%E.type]
// CHECK:STDOUT:   %.Self.4: %E.type = bind_symbolic_name .Self [symbolic = constants.%.Self.1]
// CHECK:STDOUT:   %.Self.ref.loc35_17: %E.type = name_ref .Self, %.Self.4 [symbolic = constants.%.Self.1]
// CHECK:STDOUT:   %F.ref.loc35: %assoc_type = name_ref F, @E.%assoc0 [template = constants.%assoc0.1]
// CHECK:STDOUT:   %.Self.as_wit.loc35_17: <witness> = facet_access_witness %.Self.ref.loc35_17 [symbolic = constants.%.Self.as_wit.1]
// CHECK:STDOUT:   %impl.elem0.loc35: type = interface_witness_access %.Self.as_wit.loc35_17, element0 [symbolic = constants.%impl.elem0]
// CHECK:STDOUT:   %.Self.ref.loc35_22: %E.type = name_ref .Self, %.Self.4 [symbolic = constants.%.Self.1]
// CHECK:STDOUT:   %G.ref.loc35: %assoc_type = name_ref G, @E.%assoc1 [template = constants.%assoc1]
// CHECK:STDOUT:   %.Self.as_wit.loc35_27: <witness> = facet_access_witness %.Self.ref.loc35_22 [symbolic = constants.%.Self.as_wit.1]
// CHECK:STDOUT:   %impl.elem1.loc35: type = interface_witness_access %.Self.as_wit.loc35_27, element1 [symbolic = constants.%impl.elem1.1]
// CHECK:STDOUT:   %.loc35: type = where_expr %.Self.4 [template = constants.%E_where.type.4] {
// CHECK:STDOUT:     requirement_rewrite %impl.elem0.loc35, %impl.elem1.loc35
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @E {
// CHECK:STDOUT:   %Self: %E.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.1]
// CHECK:STDOUT:   %F: type = assoc_const_decl F [template]
// CHECK:STDOUT:   %assoc0: %assoc_type = assoc_entity element0, %F [template = constants.%assoc0.1]
// CHECK:STDOUT:   %G: type = assoc_const_decl G [template]
// CHECK:STDOUT:   %assoc1: %assoc_type = assoc_entity element1, %G [template = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   .G = %assoc1
// CHECK:STDOUT:   witness = (%F, %G)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [template = constants.%int_64]
// CHECK:STDOUT:   %float.make_type: init type = call constants.%Float(%int_64) [template = f64]
// CHECK:STDOUT:   %.loc18: %E_where.type.1 = converted %float.make_type, <error> [template = <error>]
// CHECK:STDOUT:   %H: %E_where.type.1 = bind_name H, <error>
// CHECK:STDOUT:   %bool.make_type.loc27: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:   %.loc27: %E_where.type.3 = converted %bool.make_type.loc27, <error> [template = <error>]
// CHECK:STDOUT:   %J: %E_where.type.3 = bind_name J, <error>
// CHECK:STDOUT:   %bool.make_type.loc35: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:   %.loc35: %E_where.type.4 = converted %bool.make_type.loc35, <error> [template = <error>]
// CHECK:STDOUT:   %K: %E_where.type.4 = bind_name K, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

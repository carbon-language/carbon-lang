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
  // CHECK:STDERR: fail_rewrites_mismatch_right.carbon:[[@LINE+7]]:3: note: type `O where .(O.P) = bool` does not implement interface `Core.ImplicitAs(O where .(O.P) = i32)` [MissingImplInMemberAccessNote]
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
  // CHECK:STDERR: fail_rewrites_mismatch_left.carbon:[[@LINE+7]]:3: note: type `S where .(S.U) = ()` does not implement interface `Core.ImplicitAs(S where .(S.T) = ())` [MissingImplInMemberAccessNote]
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
  // CHECK:STDERR: fail_import_rewrites.carbon:[[@LINE+8]]:3: note: type `type` does not implement interface `Core.ImplicitAs(N where .(N.P) = {})` [MissingImplInMemberAccessNote]
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
  // CHECK:STDERR: fail_import_rewrites.carbon:[[@LINE+8]]:3: note: type `type` does not implement interface `Core.ImplicitAs(A where .(A.C) = () and .(A.B) = bool)` [MissingImplInMemberAccessNote]
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
// CHECK:STDERR: fail_check_rewrite_constraints.carbon:[[@LINE+4]]:46: note: type `Core.IntLiteral` does not implement interface `Core.ImplicitAs(type)` [MissingImplInMemberAccessNote]
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
// CHECK:STDERR: fail_todo_let.carbon:[[@LINE+11]]:35: error: semantics TODO: `impl lookup for a FacetType with no interface (using `where .Self impls ...` instead?)` [SemanticsTodo]
// CHECK:STDERR: let B: type where .Self impls A = D;
// CHECK:STDERR:                                   ^
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_let.carbon:[[@LINE+7]]:35: error: cannot implicitly convert from `type` to `type where...` [ImplicitAsConversionFailure]
// CHECK:STDERR: let B: type where .Self impls A = D;
// CHECK:STDERR:                                   ^
// CHECK:STDERR: fail_todo_let.carbon:[[@LINE+4]]:35: note: type `type` does not implement interface `Core.ImplicitAs(type where...)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: let B: type where .Self impls A = D;
// CHECK:STDERR:                                   ^
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
// CHECK:STDERR: fail_type_does_not_implement_where.carbon:[[@LINE+7]]:42: error: cannot implicitly convert from `type` to `E where .(E.G) = () and .(E.F) = bool` [ImplicitAsConversionFailure]
// CHECK:STDERR: let H: (E where .F = bool and .G = ()) = f64;
// CHECK:STDERR:                                          ^~~
// CHECK:STDERR: fail_type_does_not_implement_where.carbon:[[@LINE+4]]:42: note: type `type` does not implement interface `Core.ImplicitAs(E where .(E.G) = () and .(E.F) = bool)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: let H: (E where .F = bool and .G = ()) = f64;
// CHECK:STDERR:                                          ^~~
// CHECK:STDERR:
let H: (E where .F = bool and .G = ()) = f64;

// CHECK:STDERR: fail_type_does_not_implement_where.carbon:[[@LINE+7]]:45: error: cannot implicitly convert from `type` to `E where .(E.G) = i32 and .(E.F) = {}` [ImplicitAsConversionFailure]
// CHECK:STDERR: let J: ((E where .F = {}) where .G = i32) = bool;
// CHECK:STDERR:                                             ^~~~
// CHECK:STDERR: fail_type_does_not_implement_where.carbon:[[@LINE+4]]:45: note: type `type` does not implement interface `Core.ImplicitAs(E where .(E.G) = i32 and .(E.F) = {})` [MissingImplInMemberAccessNote]
// CHECK:STDERR: let J: ((E where .F = {}) where .G = i32) = bool;
// CHECK:STDERR:                                             ^~~~
// CHECK:STDERR:
let J: ((E where .F = {}) where .G = i32) = bool;

// CHECK:STDERR: fail_type_does_not_implement_where.carbon:[[@LINE+7]]:33: error: cannot implicitly convert from `type` to `E where .(E.F) = .(E.G)` [ImplicitAsConversionFailure]
// CHECK:STDERR: let K: (E where .F = .Self.G) = bool;
// CHECK:STDERR:                                 ^~~~
// CHECK:STDERR: fail_type_does_not_implement_where.carbon:[[@LINE+4]]:33: note: type `type` does not implement interface `Core.ImplicitAs(E where .(E.F) = .(E.G))` [MissingImplInMemberAccessNote]
// CHECK:STDERR: let K: (E where .F = .Self.G) = bool;
// CHECK:STDERR:                                 ^~~~
// CHECK:STDERR:
let K: (E where .F = .Self.G) = bool;

// CHECK:STDOUT: --- equal_constraint.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %N.type: type = facet_type <@N> [concrete]
// CHECK:STDOUT:   %Self: %N.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %N.assoc_type: type = assoc_entity_type %N.type [concrete]
// CHECK:STDOUT:   %assoc0: %N.assoc_type = assoc_entity element0, @N.%P [concrete]
// CHECK:STDOUT:   %.Self: %N.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_wit: <witness> = facet_access_witness %.Self [symbolic_self]
// CHECK:STDOUT:   %N.facet: %N.type = facet_value %.Self.as_type, %.Self.as_wit [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %.Self.as_wit, element0 [symbolic_self]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %N_where.type: type = facet_type <@N where %impl.elem0 = %empty_struct_type> [concrete]
// CHECK:STDOUT:   %T: %N_where.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: %N_where.type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Equal.type: type = fn_type @Equal [concrete]
// CHECK:STDOUT:   %Equal: %Equal.type = struct_value () [concrete]
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
// CHECK:STDOUT:     .N = %N.decl
// CHECK:STDOUT:     .Equal = %Equal.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %N.decl: type = interface_decl @N [concrete = constants.%N.type] {} {}
// CHECK:STDOUT:   %Equal.decl: %Equal.type = fn_decl @Equal [concrete = constants.%Equal] {
// CHECK:STDOUT:     %T.patt.loc8_10.1: %N_where.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_10.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: %N_where.type = value_param_pattern %T.patt.loc8_10.1, runtime_param<none> [symbolic = %T.patt.loc8_10.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: %N_where.type = value_param runtime_param<none>
// CHECK:STDOUT:     %.loc8_16.1: type = splice_block %.loc8_16.2 [concrete = constants.%N_where.type] {
// CHECK:STDOUT:       %N.ref: type = name_ref N, file.%N.decl [concrete = constants.%N.type]
// CHECK:STDOUT:       %.Self: %N.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.Self.ref: %N.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %P.ref: %N.assoc_type = name_ref P, @P.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:       %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc8_22: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.Self.as_wit: <witness> = facet_access_witness %.Self.ref [symbolic_self = constants.%.Self.as_wit]
// CHECK:STDOUT:       %impl.elem0: type = impl_witness_access %.Self.as_wit, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:       %.loc8_28.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:       %.loc8_28.2: type = converted %.loc8_28.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:       %.loc8_16.2: type = where_expr %.Self [concrete = constants.%N_where.type] {
// CHECK:STDOUT:         requirement_rewrite %impl.elem0, %.loc8_28.2
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_10.1: %N_where.type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc8_10.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @N {
// CHECK:STDOUT:   %Self: %N.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %P: type = assoc_const_decl @P [concrete] {
// CHECK:STDOUT:     %assoc0: %N.assoc_type = assoc_entity element0, @N.%P [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .P = @P.%assoc0
// CHECK:STDOUT:   witness = (%P)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @P(@N.%Self: %N.type) {
// CHECK:STDOUT:   assoc_const P:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Equal(%T.loc8_10.1: %N_where.type) {
// CHECK:STDOUT:   %T.loc8_10.2: %N_where.type = bind_symbolic_name T, 0 [symbolic = %T.loc8_10.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc8_10.2: %N_where.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_10.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.param_patt: %N_where.type);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @P(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @P(constants.%N.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Equal(constants.%T) {
// CHECK:STDOUT:   %T.loc8_10.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc8_10.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- nested_rewrites.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = facet_type <@A> [concrete]
// CHECK:STDOUT:   %Self: %A.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %A.assoc_type: type = assoc_entity_type %A.type [concrete]
// CHECK:STDOUT:   %assoc0: %A.assoc_type = assoc_entity element0, @A.%B [concrete]
// CHECK:STDOUT:   %assoc1: %A.assoc_type = assoc_entity element1, @A.%C [concrete]
// CHECK:STDOUT:   %.Self.3ca: %A.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.Self.as_type.adb: type = facet_access_type %.Self.3ca [symbolic_self]
// CHECK:STDOUT:   %.Self.as_wit.794: <witness> = facet_access_witness %.Self.3ca [symbolic_self]
// CHECK:STDOUT:   %A.facet.213: %A.type = facet_value %.Self.as_type.adb, %.Self.as_wit.794 [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %.Self.as_wit.794, element0 [symbolic_self]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %A_where.type.ef9: type = facet_type <@A where %impl.elem0 = bool> [concrete]
// CHECK:STDOUT:   %.Self.e46: %A_where.type.ef9 = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type.6f3: type = facet_access_type %.Self.e46 [symbolic_self]
// CHECK:STDOUT:   %.Self.as_wit.a35: <witness> = facet_access_witness %.Self.e46 [symbolic_self]
// CHECK:STDOUT:   %A.facet.e67: %A.type = facet_value %.Self.as_type.6f3, %.Self.as_wit.a35 [symbolic_self]
// CHECK:STDOUT:   %impl.elem1: type = impl_witness_access %.Self.as_wit.a35, element1 [symbolic_self]
// CHECK:STDOUT:   %A_where.type.791: type = facet_type <@A where %impl.elem1 = %empty_tuple.type and %impl.elem0 = bool> [concrete]
// CHECK:STDOUT:   %D: %A_where.type.791 = bind_symbolic_name D, 0 [symbolic]
// CHECK:STDOUT:   %D.patt: %A_where.type.791 = symbolic_binding_pattern D, 0 [symbolic]
// CHECK:STDOUT:   %NestedRewrite.type: type = fn_type @NestedRewrite [concrete]
// CHECK:STDOUT:   %NestedRewrite: %NestedRewrite.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Bool = %Core.Bool
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .NestedRewrite = %NestedRewrite.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = interface_decl @A [concrete = constants.%A.type] {} {}
// CHECK:STDOUT:   %NestedRewrite.decl: %NestedRewrite.type = fn_decl @NestedRewrite [concrete = constants.%NestedRewrite] {
// CHECK:STDOUT:     %D.patt.loc9_18.1: %A_where.type.791 = symbolic_binding_pattern D, 0 [symbolic = %D.patt.loc9_18.2 (constants.%D.patt)]
// CHECK:STDOUT:     %D.param_patt: %A_where.type.791 = value_param_pattern %D.patt.loc9_18.1, runtime_param<none> [symbolic = %D.patt.loc9_18.2 (constants.%D.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.param: %A_where.type.791 = value_param runtime_param<none>
// CHECK:STDOUT:     %.loc9_42.1: type = splice_block %.loc9_42.2 [concrete = constants.%A_where.type.791] {
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:       %.Self.1: %A.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self.3ca]
// CHECK:STDOUT:       %.Self.ref.loc9_31: %A.type = name_ref .Self, %.Self.1 [symbolic_self = constants.%.Self.3ca]
// CHECK:STDOUT:       %B.ref: %A.assoc_type = name_ref B, @B.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:       %.Self.as_type.loc9_31: type = facet_access_type %.Self.ref.loc9_31 [symbolic_self = constants.%.Self.as_type.adb]
// CHECK:STDOUT:       %.loc9_31: type = converted %.Self.ref.loc9_31, %.Self.as_type.loc9_31 [symbolic_self = constants.%.Self.as_type.adb]
// CHECK:STDOUT:       %.Self.as_wit.loc9_31: <witness> = facet_access_witness %.Self.ref.loc9_31 [symbolic_self = constants.%.Self.as_wit.794]
// CHECK:STDOUT:       %impl.elem0: type = impl_witness_access %.Self.as_wit.loc9_31, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:       %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %.loc9_36.1: type = value_of_initializer %bool.make_type [concrete = bool]
// CHECK:STDOUT:       %.loc9_36.2: type = converted %bool.make_type, %.loc9_36.1 [concrete = bool]
// CHECK:STDOUT:       %.loc9_25: type = where_expr %.Self.1 [concrete = constants.%A_where.type.ef9] {
// CHECK:STDOUT:         requirement_rewrite %impl.elem0, %.loc9_36.2
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %.Self.2: %A_where.type.ef9 = bind_symbolic_name .Self [symbolic_self = constants.%.Self.e46]
// CHECK:STDOUT:       %.Self.ref.loc9_48: %A_where.type.ef9 = name_ref .Self, %.Self.2 [symbolic_self = constants.%.Self.e46]
// CHECK:STDOUT:       %C.ref: %A.assoc_type = name_ref C, @C.%assoc1 [concrete = constants.%assoc1]
// CHECK:STDOUT:       %.Self.as_type.loc9_48: type = facet_access_type %.Self.ref.loc9_48 [symbolic_self = constants.%.Self.as_type.6f3]
// CHECK:STDOUT:       %.loc9_48: type = converted %.Self.ref.loc9_48, %.Self.as_type.loc9_48 [symbolic_self = constants.%.Self.as_type.6f3]
// CHECK:STDOUT:       %.Self.as_wit.loc9_48: <witness> = facet_access_witness %.Self.ref.loc9_48 [symbolic_self = constants.%.Self.as_wit.a35]
// CHECK:STDOUT:       %impl.elem1: type = impl_witness_access %.Self.as_wit.loc9_48, element1 [symbolic_self = constants.%impl.elem1]
// CHECK:STDOUT:       %.loc9_54.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc9_54.2: type = converted %.loc9_54.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %.loc9_42.2: type = where_expr %.Self.2 [concrete = constants.%A_where.type.791] {
// CHECK:STDOUT:         requirement_rewrite %impl.elem1, %.loc9_54.2
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %D.loc9_18.1: %A_where.type.791 = bind_symbolic_name D, 0, %D.param [symbolic = %D.loc9_18.2 (constants.%D)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A {
// CHECK:STDOUT:   %Self: %A.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %B: type = assoc_const_decl @B [concrete] {
// CHECK:STDOUT:     %assoc0: %A.assoc_type = assoc_entity element0, @A.%B [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C: type = assoc_const_decl @C [concrete] {
// CHECK:STDOUT:     %assoc1: %A.assoc_type = assoc_entity element1, @A.%C [concrete = constants.%assoc1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .B = @B.%assoc0
// CHECK:STDOUT:   .C = @C.%assoc1
// CHECK:STDOUT:   witness = (%B, %C)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @B(@A.%Self: %A.type) {
// CHECK:STDOUT:   assoc_const B:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @C(@A.%Self: %A.type) {
// CHECK:STDOUT:   assoc_const C:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @NestedRewrite(%D.loc9_18.1: %A_where.type.791) {
// CHECK:STDOUT:   %D.loc9_18.2: %A_where.type.791 = bind_symbolic_name D, 0 [symbolic = %D.loc9_18.2 (constants.%D)]
// CHECK:STDOUT:   %D.patt.loc9_18.2: %A_where.type.791 = symbolic_binding_pattern D, 0 [symbolic = %D.patt.loc9_18.2 (constants.%D.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%D.param_patt: %A_where.type.791);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B(constants.%A.facet.213) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%A.facet.e67) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @NestedRewrite(constants.%D) {
// CHECK:STDOUT:   %D.loc9_18.2 => constants.%D
// CHECK:STDOUT:   %D.patt.loc9_18.2 => constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- repeated_rewrite.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %E.type: type = facet_type <@E> [concrete]
// CHECK:STDOUT:   %Self: %E.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %E.assoc_type: type = assoc_entity_type %E.type [concrete]
// CHECK:STDOUT:   %assoc0: %E.assoc_type = assoc_entity element0, @E.%F [concrete]
// CHECK:STDOUT:   %.Self: %E.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_wit: <witness> = facet_access_witness %.Self [symbolic_self]
// CHECK:STDOUT:   %E.facet: %E.type = facet_value %.Self.as_type, %.Self.as_wit [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %.Self.as_wit, element0 [symbolic_self]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %E_where.type: type = facet_type <@E where %impl.elem0 = %i32> [concrete]
// CHECK:STDOUT:   %G: %E_where.type = bind_symbolic_name G, 0 [symbolic]
// CHECK:STDOUT:   %G.patt: %E_where.type = symbolic_binding_pattern G, 0 [symbolic]
// CHECK:STDOUT:   %OneRewrite.type: type = fn_type @OneRewrite [concrete]
// CHECK:STDOUT:   %OneRewrite: %OneRewrite.type = struct_value () [concrete]
// CHECK:STDOUT:   %H: %E_where.type = bind_symbolic_name H, 0 [symbolic]
// CHECK:STDOUT:   %H.patt: %E_where.type = symbolic_binding_pattern H, 0 [symbolic]
// CHECK:STDOUT:   %RepeatedRewrite.type: type = fn_type @RepeatedRewrite [concrete]
// CHECK:STDOUT:   %RepeatedRewrite: %RepeatedRewrite.type = struct_value () [concrete]
// CHECK:STDOUT:   %OneRewrite.specific_fn.381461.1: <specific function> = specific_function %OneRewrite, @OneRewrite(%H) [symbolic]
// CHECK:STDOUT:   %I: %E_where.type = bind_symbolic_name I, 0 [symbolic]
// CHECK:STDOUT:   %I.patt: %E_where.type = symbolic_binding_pattern I, 0 [symbolic]
// CHECK:STDOUT:   %OneRewriteAgain.type: type = fn_type @OneRewriteAgain [concrete]
// CHECK:STDOUT:   %OneRewriteAgain: %OneRewriteAgain.type = struct_value () [concrete]
// CHECK:STDOUT:   %RepeatedRewrite.specific_fn: <specific function> = specific_function %RepeatedRewrite, @RepeatedRewrite(%I) [symbolic]
// CHECK:STDOUT:   %OneRewrite.specific_fn.381461.2: <specific function> = specific_function %OneRewrite, @OneRewrite(%I) [symbolic]
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
// CHECK:STDOUT:     .E = %E.decl
// CHECK:STDOUT:     .OneRewrite = %OneRewrite.decl
// CHECK:STDOUT:     .RepeatedRewrite = %RepeatedRewrite.decl
// CHECK:STDOUT:     .OneRewriteAgain = %OneRewriteAgain.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %E.decl: type = interface_decl @E [concrete = constants.%E.type] {} {}
// CHECK:STDOUT:   %OneRewrite.decl: %OneRewrite.type = fn_decl @OneRewrite [concrete = constants.%OneRewrite] {
// CHECK:STDOUT:     %G.patt.loc8_15.1: %E_where.type = symbolic_binding_pattern G, 0 [symbolic = %G.patt.loc8_15.2 (constants.%G.patt)]
// CHECK:STDOUT:     %G.param_patt: %E_where.type = value_param_pattern %G.patt.loc8_15.1, runtime_param<none> [symbolic = %G.patt.loc8_15.2 (constants.%G.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %G.param: %E_where.type = value_param runtime_param<none>
// CHECK:STDOUT:     %.loc8_21.1: type = splice_block %.loc8_21.2 [concrete = constants.%E_where.type] {
// CHECK:STDOUT:       %E.ref: type = name_ref E, file.%E.decl [concrete = constants.%E.type]
// CHECK:STDOUT:       %.Self: %E.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.Self.ref: %E.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %F.ref: %E.assoc_type = name_ref F, @F.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:       %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc8_27: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.Self.as_wit: <witness> = facet_access_witness %.Self.ref [symbolic_self = constants.%.Self.as_wit]
// CHECK:STDOUT:       %impl.elem0: type = impl_witness_access %.Self.as_wit, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %.loc8_21.2: type = where_expr %.Self [concrete = constants.%E_where.type] {
// CHECK:STDOUT:         requirement_rewrite %impl.elem0, %i32
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %G.loc8_15.1: %E_where.type = bind_symbolic_name G, 0, %G.param [symbolic = %G.loc8_15.2 (constants.%G)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RepeatedRewrite.decl: %RepeatedRewrite.type = fn_decl @RepeatedRewrite [concrete = constants.%RepeatedRewrite] {
// CHECK:STDOUT:     %H.patt.loc10_20.1: %E_where.type = symbolic_binding_pattern H, 0 [symbolic = %H.patt.loc10_20.2 (constants.%H.patt)]
// CHECK:STDOUT:     %H.param_patt: %E_where.type = value_param_pattern %H.patt.loc10_20.1, runtime_param<none> [symbolic = %H.patt.loc10_20.2 (constants.%H.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %H.param: %E_where.type = value_param runtime_param<none>
// CHECK:STDOUT:     %.loc10_26.1: type = splice_block %.loc10_26.2 [concrete = constants.%E_where.type] {
// CHECK:STDOUT:       %E.ref: type = name_ref E, file.%E.decl [concrete = constants.%E.type]
// CHECK:STDOUT:       %.Self: %E.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.Self.ref.loc10_32: %E.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %F.ref.loc10_32: %E.assoc_type = name_ref F, @F.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:       %.Self.as_type.loc10_32: type = facet_access_type %.Self.ref.loc10_32 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc10_32: type = converted %.Self.ref.loc10_32, %.Self.as_type.loc10_32 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.Self.as_wit.loc10_32: <witness> = facet_access_witness %.Self.ref.loc10_32 [symbolic_self = constants.%.Self.as_wit]
// CHECK:STDOUT:       %impl.elem0.loc10_32: type = impl_witness_access %.Self.as_wit.loc10_32, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:       %int_32.loc10_37: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc10_37: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %.Self.ref.loc10_45: %E.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %F.ref.loc10_45: %E.assoc_type = name_ref F, @F.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:       %.Self.as_type.loc10_45: type = facet_access_type %.Self.ref.loc10_45 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc10_45: type = converted %.Self.ref.loc10_45, %.Self.as_type.loc10_45 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.Self.as_wit.loc10_45: <witness> = facet_access_witness %.Self.ref.loc10_45 [symbolic_self = constants.%.Self.as_wit]
// CHECK:STDOUT:       %impl.elem0.loc10_45: type = impl_witness_access %.Self.as_wit.loc10_45, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:       %int_32.loc10_50: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc10_50: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %.loc10_26.2: type = where_expr %.Self [concrete = constants.%E_where.type] {
// CHECK:STDOUT:         requirement_rewrite %impl.elem0.loc10_32, %i32.loc10_37
// CHECK:STDOUT:         requirement_rewrite %impl.elem0.loc10_45, %i32.loc10_50
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %H.loc10_20.1: %E_where.type = bind_symbolic_name H, 0, %H.param [symbolic = %H.loc10_20.2 (constants.%H)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %OneRewriteAgain.decl: %OneRewriteAgain.type = fn_decl @OneRewriteAgain [concrete = constants.%OneRewriteAgain] {
// CHECK:STDOUT:     %I.patt.loc14_20.1: %E_where.type = symbolic_binding_pattern I, 0 [symbolic = %I.patt.loc14_20.2 (constants.%I.patt)]
// CHECK:STDOUT:     %I.param_patt: %E_where.type = value_param_pattern %I.patt.loc14_20.1, runtime_param<none> [symbolic = %I.patt.loc14_20.2 (constants.%I.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %I.param: %E_where.type = value_param runtime_param<none>
// CHECK:STDOUT:     %.loc14_26.1: type = splice_block %.loc14_26.2 [concrete = constants.%E_where.type] {
// CHECK:STDOUT:       %E.ref: type = name_ref E, file.%E.decl [concrete = constants.%E.type]
// CHECK:STDOUT:       %.Self: %E.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.Self.ref: %E.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %F.ref: %E.assoc_type = name_ref F, @F.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:       %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc14_32: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.Self.as_wit: <witness> = facet_access_witness %.Self.ref [symbolic_self = constants.%.Self.as_wit]
// CHECK:STDOUT:       %impl.elem0: type = impl_witness_access %.Self.as_wit, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %.loc14_26.2: type = where_expr %.Self [concrete = constants.%E_where.type] {
// CHECK:STDOUT:         requirement_rewrite %impl.elem0, %i32
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %I.loc14_20.1: %E_where.type = bind_symbolic_name I, 0, %I.param [symbolic = %I.loc14_20.2 (constants.%I)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @E {
// CHECK:STDOUT:   %Self: %E.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %F: type = assoc_const_decl @F [concrete] {
// CHECK:STDOUT:     %assoc0: %E.assoc_type = assoc_entity element0, @E.%F [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = @F.%assoc0
// CHECK:STDOUT:   witness = (%F)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @F(@E.%Self: %E.type) {
// CHECK:STDOUT:   assoc_const F:! type;
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
// CHECK:STDOUT:   %OneRewrite.specific_fn.loc11_3.2: <specific function> = specific_function constants.%OneRewrite, @OneRewrite(%H.loc10_20.2) [symbolic = %OneRewrite.specific_fn.loc11_3.2 (constants.%OneRewrite.specific_fn.381461.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%H.param_patt: %E_where.type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %OneRewrite.ref: %OneRewrite.type = name_ref OneRewrite, file.%OneRewrite.decl [concrete = constants.%OneRewrite]
// CHECK:STDOUT:     %H.ref: %E_where.type = name_ref H, %H.loc10_20.1 [symbolic = %H.loc10_20.2 (constants.%H)]
// CHECK:STDOUT:     %OneRewrite.specific_fn.loc11_3.1: <specific function> = specific_function %OneRewrite.ref, @OneRewrite(constants.%H) [symbolic = %OneRewrite.specific_fn.loc11_3.2 (constants.%OneRewrite.specific_fn.381461.1)]
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
// CHECK:STDOUT:     %RepeatedRewrite.ref: %RepeatedRewrite.type = name_ref RepeatedRewrite, file.%RepeatedRewrite.decl [concrete = constants.%RepeatedRewrite]
// CHECK:STDOUT:     %I.ref: %E_where.type = name_ref I, %I.loc14_20.1 [symbolic = %I.loc14_20.2 (constants.%I)]
// CHECK:STDOUT:     %RepeatedRewrite.specific_fn.loc15_3.1: <specific function> = specific_function %RepeatedRewrite.ref, @RepeatedRewrite(constants.%I) [symbolic = %RepeatedRewrite.specific_fn.loc15_3.2 (constants.%RepeatedRewrite.specific_fn)]
// CHECK:STDOUT:     %RepeatedRewrite.call: init %empty_tuple.type = call %RepeatedRewrite.specific_fn.loc15_3.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%E.facet) {}
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
// CHECK:STDOUT: specific @OneRewrite(@RepeatedRewrite.%H.loc10_20.2) {}
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
// CHECK:STDOUT:   %OneRewrite.specific_fn.loc11_3.2 => constants.%OneRewrite.specific_fn.381461.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @RepeatedRewrite(@OneRewriteAgain.%I.loc14_20.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OneRewrite(constants.%I) {
// CHECK:STDOUT:   %G.loc8_15.2 => constants.%I
// CHECK:STDOUT:   %G.patt.loc8_15.2 => constants.%I
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- rewrites_reordered.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type %J.type [concrete]
// CHECK:STDOUT:   %assoc0: %J.assoc_type = assoc_entity element0, @J.%K [concrete]
// CHECK:STDOUT:   %assoc1: %J.assoc_type = assoc_entity element1, @J.%L [concrete]
// CHECK:STDOUT:   %.Self: %J.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_wit: <witness> = facet_access_witness %.Self [symbolic_self]
// CHECK:STDOUT:   %J.facet: %J.type = facet_value %.Self.as_type, %.Self.as_wit [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %.Self.as_wit, element0 [symbolic_self]
// CHECK:STDOUT:   %impl.elem1: type = impl_witness_access %.Self.as_wit, element1 [symbolic_self]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %J_where.type: type = facet_type <@J where %impl.elem1 = bool and %impl.elem0 = %empty_tuple.type> [concrete]
// CHECK:STDOUT:   %M: %J_where.type = bind_symbolic_name M, 0 [symbolic]
// CHECK:STDOUT:   %M.patt: %J_where.type = symbolic_binding_pattern M, 0 [symbolic]
// CHECK:STDOUT:   %Alphabetical.type: type = fn_type @Alphabetical [concrete]
// CHECK:STDOUT:   %Alphabetical: %Alphabetical.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: %J_where.type = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt: %J_where.type = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %Reversed.type: type = fn_type @Reversed [concrete]
// CHECK:STDOUT:   %Reversed: %Reversed.type = struct_value () [concrete]
// CHECK:STDOUT:   %Alphabetical.specific_fn: <specific function> = specific_function %Alphabetical, @Alphabetical(%N) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Bool = %Core.Bool
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .Alphabetical = %Alphabetical.decl
// CHECK:STDOUT:     .Reversed = %Reversed.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [concrete = constants.%J.type] {} {}
// CHECK:STDOUT:   %Alphabetical.decl: %Alphabetical.type = fn_decl @Alphabetical [concrete = constants.%Alphabetical] {
// CHECK:STDOUT:     %M.patt.loc9_17.1: %J_where.type = symbolic_binding_pattern M, 0 [symbolic = %M.patt.loc9_17.2 (constants.%M.patt)]
// CHECK:STDOUT:     %M.param_patt: %J_where.type = value_param_pattern %M.patt.loc9_17.1, runtime_param<none> [symbolic = %M.patt.loc9_17.2 (constants.%M.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %M.param: %J_where.type = value_param runtime_param<none>
// CHECK:STDOUT:     %.loc9_23.1: type = splice_block %.loc9_23.2 [concrete = constants.%J_where.type] {
// CHECK:STDOUT:       %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:       %.Self: %J.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.Self.ref.loc9_29: %J.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %K.ref: %J.assoc_type = name_ref K, @K.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:       %.Self.as_type.loc9_29: type = facet_access_type %.Self.ref.loc9_29 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc9_29: type = converted %.Self.ref.loc9_29, %.Self.as_type.loc9_29 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.Self.as_wit.loc9_29: <witness> = facet_access_witness %.Self.ref.loc9_29 [symbolic_self = constants.%.Self.as_wit]
// CHECK:STDOUT:       %impl.elem0: type = impl_witness_access %.Self.as_wit.loc9_29, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:       %.loc9_35.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc9_35.2: type = converted %.loc9_35.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %.Self.ref.loc9_41: %J.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %L.ref: %J.assoc_type = name_ref L, @L.%assoc1 [concrete = constants.%assoc1]
// CHECK:STDOUT:       %.Self.as_type.loc9_41: type = facet_access_type %.Self.ref.loc9_41 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc9_41: type = converted %.Self.ref.loc9_41, %.Self.as_type.loc9_41 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.Self.as_wit.loc9_41: <witness> = facet_access_witness %.Self.ref.loc9_41 [symbolic_self = constants.%.Self.as_wit]
// CHECK:STDOUT:       %impl.elem1: type = impl_witness_access %.Self.as_wit.loc9_41, element1 [symbolic_self = constants.%impl.elem1]
// CHECK:STDOUT:       %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %.loc9_46.1: type = value_of_initializer %bool.make_type [concrete = bool]
// CHECK:STDOUT:       %.loc9_46.2: type = converted %bool.make_type, %.loc9_46.1 [concrete = bool]
// CHECK:STDOUT:       %.loc9_23.2: type = where_expr %.Self [concrete = constants.%J_where.type] {
// CHECK:STDOUT:         requirement_rewrite %impl.elem0, %.loc9_35.2
// CHECK:STDOUT:         requirement_rewrite %impl.elem1, %.loc9_46.2
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %M.loc9_17.1: %J_where.type = bind_symbolic_name M, 0, %M.param [symbolic = %M.loc9_17.2 (constants.%M)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Reversed.decl: %Reversed.type = fn_decl @Reversed [concrete = constants.%Reversed] {
// CHECK:STDOUT:     %N.patt.loc11_13.1: %J_where.type = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc11_13.2 (constants.%N.patt)]
// CHECK:STDOUT:     %N.param_patt: %J_where.type = value_param_pattern %N.patt.loc11_13.1, runtime_param<none> [symbolic = %N.patt.loc11_13.2 (constants.%N.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %N.param: %J_where.type = value_param runtime_param<none>
// CHECK:STDOUT:     %.loc11_19.1: type = splice_block %.loc11_19.2 [concrete = constants.%J_where.type] {
// CHECK:STDOUT:       %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:       %.Self: %J.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.Self.ref.loc11_25: %J.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %L.ref: %J.assoc_type = name_ref L, @L.%assoc1 [concrete = constants.%assoc1]
// CHECK:STDOUT:       %.Self.as_type.loc11_25: type = facet_access_type %.Self.ref.loc11_25 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc11_25: type = converted %.Self.ref.loc11_25, %.Self.as_type.loc11_25 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.Self.as_wit.loc11_25: <witness> = facet_access_witness %.Self.ref.loc11_25 [symbolic_self = constants.%.Self.as_wit]
// CHECK:STDOUT:       %impl.elem1: type = impl_witness_access %.Self.as_wit.loc11_25, element1 [symbolic_self = constants.%impl.elem1]
// CHECK:STDOUT:       %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %.loc11_30.1: type = value_of_initializer %bool.make_type [concrete = bool]
// CHECK:STDOUT:       %.loc11_30.2: type = converted %bool.make_type, %.loc11_30.1 [concrete = bool]
// CHECK:STDOUT:       %.Self.ref.loc11_39: %J.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %K.ref: %J.assoc_type = name_ref K, @K.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:       %.Self.as_type.loc11_39: type = facet_access_type %.Self.ref.loc11_39 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc11_39: type = converted %.Self.ref.loc11_39, %.Self.as_type.loc11_39 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.Self.as_wit.loc11_39: <witness> = facet_access_witness %.Self.ref.loc11_39 [symbolic_self = constants.%.Self.as_wit]
// CHECK:STDOUT:       %impl.elem0: type = impl_witness_access %.Self.as_wit.loc11_39, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:       %.loc11_45.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc11_45.2: type = converted %.loc11_45.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %.loc11_19.2: type = where_expr %.Self [concrete = constants.%J_where.type] {
// CHECK:STDOUT:         requirement_rewrite %impl.elem1, %.loc11_30.2
// CHECK:STDOUT:         requirement_rewrite %impl.elem0, %.loc11_45.2
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc11_13.1: %J_where.type = bind_symbolic_name N, 0, %N.param [symbolic = %N.loc11_13.2 (constants.%N)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %K: type = assoc_const_decl @K [concrete] {
// CHECK:STDOUT:     %assoc0: %J.assoc_type = assoc_entity element0, @J.%K [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %L: type = assoc_const_decl @L [concrete] {
// CHECK:STDOUT:     %assoc1: %J.assoc_type = assoc_entity element1, @J.%L [concrete = constants.%assoc1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .K = @K.%assoc0
// CHECK:STDOUT:   .L = @L.%assoc1
// CHECK:STDOUT:   witness = (%K, %L)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @K(@J.%Self: %J.type) {
// CHECK:STDOUT:   assoc_const K:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @L(@J.%Self: %J.type) {
// CHECK:STDOUT:   assoc_const L:! type;
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
// CHECK:STDOUT:     %Alphabetical.ref: %Alphabetical.type = name_ref Alphabetical, file.%Alphabetical.decl [concrete = constants.%Alphabetical]
// CHECK:STDOUT:     %N.ref: %J_where.type = name_ref N, %N.loc11_13.1 [symbolic = %N.loc11_13.2 (constants.%N)]
// CHECK:STDOUT:     %Alphabetical.specific_fn.loc12_3.1: <specific function> = specific_function %Alphabetical.ref, @Alphabetical(constants.%N) [symbolic = %Alphabetical.specific_fn.loc12_3.2 (constants.%Alphabetical.specific_fn)]
// CHECK:STDOUT:     %Alphabetical.call: init %empty_tuple.type = call %Alphabetical.specific_fn.loc12_3.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @K(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @L(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @K(constants.%J.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @L(constants.%J.facet) {}
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
// CHECK:STDOUT: specific @Alphabetical(@Reversed.%N.loc11_13.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_rewrites_mismatch_right.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %O.type: type = facet_type <@O> [concrete]
// CHECK:STDOUT:   %Self.040: %O.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %O.assoc_type: type = assoc_entity_type %O.type [concrete]
// CHECK:STDOUT:   %assoc0.0e4: %O.assoc_type = assoc_entity element0, @O.%P [concrete]
// CHECK:STDOUT:   %.Self: %O.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_wit: <witness> = facet_access_witness %.Self [symbolic_self]
// CHECK:STDOUT:   %O.facet: %O.type = facet_value %.Self.as_type, %.Self.as_wit [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %.Self.as_wit, element0 [symbolic_self]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %O_where.type.9eb: type = facet_type <@O where %impl.elem0 = %i32> [concrete]
// CHECK:STDOUT:   %Q: %O_where.type.9eb = bind_symbolic_name Q, 0 [symbolic]
// CHECK:STDOUT:   %Q.patt: %O_where.type.9eb = symbolic_binding_pattern Q, 0 [symbolic]
// CHECK:STDOUT:   %WithInteger.type: type = fn_type @WithInteger [concrete]
// CHECK:STDOUT:   %WithInteger: %WithInteger.type = struct_value () [concrete]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %O_where.type.0f2: type = facet_type <@O where %impl.elem0 = bool> [concrete]
// CHECK:STDOUT:   %R: %O_where.type.0f2 = bind_symbolic_name R, 0 [symbolic]
// CHECK:STDOUT:   %R.patt: %O_where.type.0f2 = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %WithBool.type: type = fn_type @WithBool [concrete]
// CHECK:STDOUT:   %WithBool: %WithBool.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Bool = %Core.Bool
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .O = %O.decl
// CHECK:STDOUT:     .WithInteger = %WithInteger.decl
// CHECK:STDOUT:     .WithBool = %WithBool.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %O.decl: type = interface_decl @O [concrete = constants.%O.type] {} {}
// CHECK:STDOUT:   %WithInteger.decl: %WithInteger.type = fn_decl @WithInteger [concrete = constants.%WithInteger] {
// CHECK:STDOUT:     %Q.patt.loc8_16.1: %O_where.type.9eb = symbolic_binding_pattern Q, 0 [symbolic = %Q.patt.loc8_16.2 (constants.%Q.patt)]
// CHECK:STDOUT:     %Q.param_patt: %O_where.type.9eb = value_param_pattern %Q.patt.loc8_16.1, runtime_param<none> [symbolic = %Q.patt.loc8_16.2 (constants.%Q.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Q.param: %O_where.type.9eb = value_param runtime_param<none>
// CHECK:STDOUT:     %.loc8_22.1: type = splice_block %.loc8_22.2 [concrete = constants.%O_where.type.9eb] {
// CHECK:STDOUT:       %O.ref: type = name_ref O, file.%O.decl [concrete = constants.%O.type]
// CHECK:STDOUT:       %.Self: %O.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.Self.ref: %O.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %P.ref: %O.assoc_type = name_ref P, @P.%assoc0 [concrete = constants.%assoc0.0e4]
// CHECK:STDOUT:       %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc8_28: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.Self.as_wit: <witness> = facet_access_witness %.Self.ref [symbolic_self = constants.%.Self.as_wit]
// CHECK:STDOUT:       %impl.elem0: type = impl_witness_access %.Self.as_wit, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %.loc8_22.2: type = where_expr %.Self [concrete = constants.%O_where.type.9eb] {
// CHECK:STDOUT:         requirement_rewrite %impl.elem0, %i32
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Q.loc8_16.1: %O_where.type.9eb = bind_symbolic_name Q, 0, %Q.param [symbolic = %Q.loc8_16.2 (constants.%Q)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %WithBool.decl: %WithBool.type = fn_decl @WithBool [concrete = constants.%WithBool] {
// CHECK:STDOUT:     %R.patt.loc10_13.1: %O_where.type.0f2 = symbolic_binding_pattern R, 0 [symbolic = %R.patt.loc10_13.2 (constants.%R.patt)]
// CHECK:STDOUT:     %R.param_patt: %O_where.type.0f2 = value_param_pattern %R.patt.loc10_13.1, runtime_param<none> [symbolic = %R.patt.loc10_13.2 (constants.%R.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %R.param: %O_where.type.0f2 = value_param runtime_param<none>
// CHECK:STDOUT:     %.loc10_19.1: type = splice_block %.loc10_19.2 [concrete = constants.%O_where.type.0f2] {
// CHECK:STDOUT:       %O.ref: type = name_ref O, file.%O.decl [concrete = constants.%O.type]
// CHECK:STDOUT:       %.Self: %O.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.Self.ref: %O.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %P.ref: %O.assoc_type = name_ref P, @P.%assoc0 [concrete = constants.%assoc0.0e4]
// CHECK:STDOUT:       %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc10_25: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.Self.as_wit: <witness> = facet_access_witness %.Self.ref [symbolic_self = constants.%.Self.as_wit]
// CHECK:STDOUT:       %impl.elem0: type = impl_witness_access %.Self.as_wit, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:       %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %.loc10_30.1: type = value_of_initializer %bool.make_type [concrete = bool]
// CHECK:STDOUT:       %.loc10_30.2: type = converted %bool.make_type, %.loc10_30.1 [concrete = bool]
// CHECK:STDOUT:       %.loc10_19.2: type = where_expr %.Self [concrete = constants.%O_where.type.0f2] {
// CHECK:STDOUT:         requirement_rewrite %impl.elem0, %.loc10_30.2
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %R.loc10_13.1: %O_where.type.0f2 = bind_symbolic_name R, 0, %R.param [symbolic = %R.loc10_13.2 (constants.%R)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @O {
// CHECK:STDOUT:   %Self: %O.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.040]
// CHECK:STDOUT:   %P: type = assoc_const_decl @P [concrete] {
// CHECK:STDOUT:     %assoc0: %O.assoc_type = assoc_entity element0, @O.%P [concrete = constants.%assoc0.0e4]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .P = @P.%assoc0
// CHECK:STDOUT:   witness = (%P)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @P(@O.%Self: %O.type) {
// CHECK:STDOUT:   assoc_const P:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @WithInteger(%Q.loc8_16.1: %O_where.type.9eb) {
// CHECK:STDOUT:   %Q.loc8_16.2: %O_where.type.9eb = bind_symbolic_name Q, 0 [symbolic = %Q.loc8_16.2 (constants.%Q)]
// CHECK:STDOUT:   %Q.patt.loc8_16.2: %O_where.type.9eb = symbolic_binding_pattern Q, 0 [symbolic = %Q.patt.loc8_16.2 (constants.%Q.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%Q.param_patt: %O_where.type.9eb) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @WithBool(%R.loc10_13.1: %O_where.type.0f2) {
// CHECK:STDOUT:   %R.loc10_13.2: %O_where.type.0f2 = bind_symbolic_name R, 0 [symbolic = %R.loc10_13.2 (constants.%R)]
// CHECK:STDOUT:   %R.patt.loc10_13.2: %O_where.type.0f2 = symbolic_binding_pattern R, 0 [symbolic = %R.patt.loc10_13.2 (constants.%R.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%R.param_patt: %O_where.type.0f2) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %WithInteger.ref: %WithInteger.type = name_ref WithInteger, file.%WithInteger.decl [concrete = constants.%WithInteger]
// CHECK:STDOUT:     %R.ref: %O_where.type.0f2 = name_ref R, %R.loc10_13.1 [symbolic = %R.loc10_13.2 (constants.%R)]
// CHECK:STDOUT:     %.loc21: %O_where.type.9eb = converted %R.ref, <error> [concrete = <error>]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @P(constants.%Self.040) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @P(constants.%O.facet) {}
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
// CHECK:STDOUT:   %S.type: type = facet_type <@S> [concrete]
// CHECK:STDOUT:   %Self.b33: %S.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %S.assoc_type: type = assoc_entity_type %S.type [concrete]
// CHECK:STDOUT:   %assoc0.720: %S.assoc_type = assoc_entity element0, @S.%T [concrete]
// CHECK:STDOUT:   %assoc1: %S.assoc_type = assoc_entity element1, @S.%U [concrete]
// CHECK:STDOUT:   %.Self: %S.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_wit: <witness> = facet_access_witness %.Self [symbolic_self]
// CHECK:STDOUT:   %S.facet: %S.type = facet_value %.Self.as_type, %.Self.as_wit [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %.Self.as_wit, element0 [symbolic_self]
// CHECK:STDOUT:   %S_where.type.e40: type = facet_type <@S where %impl.elem0 = %empty_tuple.type> [concrete]
// CHECK:STDOUT:   %V: %S_where.type.e40 = bind_symbolic_name V, 0 [symbolic]
// CHECK:STDOUT:   %V.patt: %S_where.type.e40 = symbolic_binding_pattern V, 0 [symbolic]
// CHECK:STDOUT:   %WithT.type: type = fn_type @WithT [concrete]
// CHECK:STDOUT:   %WithT: %WithT.type = struct_value () [concrete]
// CHECK:STDOUT:   %impl.elem1: type = impl_witness_access %.Self.as_wit, element1 [symbolic_self]
// CHECK:STDOUT:   %S_where.type.357: type = facet_type <@S where %impl.elem1 = %empty_tuple.type> [concrete]
// CHECK:STDOUT:   %W: %S_where.type.357 = bind_symbolic_name W, 0 [symbolic]
// CHECK:STDOUT:   %W.patt: %S_where.type.357 = symbolic_binding_pattern W, 0 [symbolic]
// CHECK:STDOUT:   %WithU.type: type = fn_type @WithU [concrete]
// CHECK:STDOUT:   %WithU: %WithU.type = struct_value () [concrete]
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
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     .WithT = %WithT.decl
// CHECK:STDOUT:     .WithU = %WithU.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %S.decl: type = interface_decl @S [concrete = constants.%S.type] {} {}
// CHECK:STDOUT:   %WithT.decl: %WithT.type = fn_decl @WithT [concrete = constants.%WithT] {
// CHECK:STDOUT:     %V.patt.loc9_10.1: %S_where.type.e40 = symbolic_binding_pattern V, 0 [symbolic = %V.patt.loc9_10.2 (constants.%V.patt)]
// CHECK:STDOUT:     %V.param_patt: %S_where.type.e40 = value_param_pattern %V.patt.loc9_10.1, runtime_param<none> [symbolic = %V.patt.loc9_10.2 (constants.%V.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %V.param: %S_where.type.e40 = value_param runtime_param<none>
// CHECK:STDOUT:     %.loc9_16.1: type = splice_block %.loc9_16.2 [concrete = constants.%S_where.type.e40] {
// CHECK:STDOUT:       %S.ref: type = name_ref S, file.%S.decl [concrete = constants.%S.type]
// CHECK:STDOUT:       %.Self: %S.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.Self.ref: %S.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %T.ref: %S.assoc_type = name_ref T, @T.%assoc0 [concrete = constants.%assoc0.720]
// CHECK:STDOUT:       %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc9_22: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.Self.as_wit: <witness> = facet_access_witness %.Self.ref [symbolic_self = constants.%.Self.as_wit]
// CHECK:STDOUT:       %impl.elem0: type = impl_witness_access %.Self.as_wit, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:       %.loc9_28.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc9_28.2: type = converted %.loc9_28.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %.loc9_16.2: type = where_expr %.Self [concrete = constants.%S_where.type.e40] {
// CHECK:STDOUT:         requirement_rewrite %impl.elem0, %.loc9_28.2
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %V.loc9_10.1: %S_where.type.e40 = bind_symbolic_name V, 0, %V.param [symbolic = %V.loc9_10.2 (constants.%V)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %WithU.decl: %WithU.type = fn_decl @WithU [concrete = constants.%WithU] {
// CHECK:STDOUT:     %W.patt.loc11_10.1: %S_where.type.357 = symbolic_binding_pattern W, 0 [symbolic = %W.patt.loc11_10.2 (constants.%W.patt)]
// CHECK:STDOUT:     %W.param_patt: %S_where.type.357 = value_param_pattern %W.patt.loc11_10.1, runtime_param<none> [symbolic = %W.patt.loc11_10.2 (constants.%W.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %W.param: %S_where.type.357 = value_param runtime_param<none>
// CHECK:STDOUT:     %.loc11_16.1: type = splice_block %.loc11_16.2 [concrete = constants.%S_where.type.357] {
// CHECK:STDOUT:       %S.ref: type = name_ref S, file.%S.decl [concrete = constants.%S.type]
// CHECK:STDOUT:       %.Self: %S.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.Self.ref: %S.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %U.ref: %S.assoc_type = name_ref U, @U.%assoc1 [concrete = constants.%assoc1]
// CHECK:STDOUT:       %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc11_22: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.Self.as_wit: <witness> = facet_access_witness %.Self.ref [symbolic_self = constants.%.Self.as_wit]
// CHECK:STDOUT:       %impl.elem1: type = impl_witness_access %.Self.as_wit, element1 [symbolic_self = constants.%impl.elem1]
// CHECK:STDOUT:       %.loc11_28.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc11_28.2: type = converted %.loc11_28.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %.loc11_16.2: type = where_expr %.Self [concrete = constants.%S_where.type.357] {
// CHECK:STDOUT:         requirement_rewrite %impl.elem1, %.loc11_28.2
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %W.loc11_10.1: %S_where.type.357 = bind_symbolic_name W, 0, %W.param [symbolic = %W.loc11_10.2 (constants.%W)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @S {
// CHECK:STDOUT:   %Self: %S.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.b33]
// CHECK:STDOUT:   %T: type = assoc_const_decl @T [concrete] {
// CHECK:STDOUT:     %assoc0: %S.assoc_type = assoc_entity element0, @S.%T [concrete = constants.%assoc0.720]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %U: type = assoc_const_decl @U [concrete] {
// CHECK:STDOUT:     %assoc1: %S.assoc_type = assoc_entity element1, @S.%U [concrete = constants.%assoc1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .T = @T.%assoc0
// CHECK:STDOUT:   .U = @U.%assoc1
// CHECK:STDOUT:   witness = (%T, %U)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @T(@S.%Self: %S.type) {
// CHECK:STDOUT:   assoc_const T:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @U(@S.%Self: %S.type) {
// CHECK:STDOUT:   assoc_const U:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @WithT(%V.loc9_10.1: %S_where.type.e40) {
// CHECK:STDOUT:   %V.loc9_10.2: %S_where.type.e40 = bind_symbolic_name V, 0 [symbolic = %V.loc9_10.2 (constants.%V)]
// CHECK:STDOUT:   %V.patt.loc9_10.2: %S_where.type.e40 = symbolic_binding_pattern V, 0 [symbolic = %V.patt.loc9_10.2 (constants.%V.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%V.param_patt: %S_where.type.e40) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @WithU(%W.loc11_10.1: %S_where.type.357) {
// CHECK:STDOUT:   %W.loc11_10.2: %S_where.type.357 = bind_symbolic_name W, 0 [symbolic = %W.loc11_10.2 (constants.%W)]
// CHECK:STDOUT:   %W.patt.loc11_10.2: %S_where.type.357 = symbolic_binding_pattern W, 0 [symbolic = %W.patt.loc11_10.2 (constants.%W.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%W.param_patt: %S_where.type.357) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %WithT.ref: %WithT.type = name_ref WithT, file.%WithT.decl [concrete = constants.%WithT]
// CHECK:STDOUT:     %W.ref: %S_where.type.357 = name_ref W, %W.loc11_10.1 [symbolic = %W.loc11_10.2 (constants.%W)]
// CHECK:STDOUT:     %.loc22: %S_where.type.e40 = converted %W.ref, <error> [concrete = <error>]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T(constants.%Self.b33) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%Self.b33) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T(constants.%S.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WithT(constants.%V) {
// CHECK:STDOUT:   %V.loc9_10.2 => constants.%V
// CHECK:STDOUT:   %V.patt.loc9_10.2 => constants.%V
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%S.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WithU(constants.%W) {
// CHECK:STDOUT:   %W.loc11_10.2 => constants.%W
// CHECK:STDOUT:   %W.patt.loc11_10.2 => constants.%W
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_rewrites.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Calls.type: type = fn_type @Calls [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Calls: %Calls.type = struct_value () [concrete]
// CHECK:STDOUT:   %Equal.type.d73: type = fn_type @Equal.1 [concrete]
// CHECK:STDOUT:   %Equal.517: %Equal.type.d73 = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %N.type: type = facet_type <@N> [concrete]
// CHECK:STDOUT:   %.Self.9aa: %N.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_wit.cef: <witness> = facet_access_witness %.Self.9aa [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.51b: type = impl_witness_access %.Self.as_wit.cef, element0 [symbolic_self]
// CHECK:STDOUT:   %N_where.type: type = facet_type <@N where %impl.elem0.51b = %empty_struct_type> [concrete]
// CHECK:STDOUT:   %T.patt: %N_where.type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: %N_where.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %NestedRewrite.type: type = fn_type @NestedRewrite [concrete]
// CHECK:STDOUT:   %NestedRewrite: %NestedRewrite.type = struct_value () [concrete]
// CHECK:STDOUT:   %A.type: type = facet_type <@A> [concrete]
// CHECK:STDOUT:   %.Self.3ca: %A.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_wit.794: <witness> = facet_access_witness %.Self.3ca [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.1d3: type = impl_witness_access %.Self.as_wit.794, element0 [symbolic_self]
// CHECK:STDOUT:   %A_where.type.ef9: type = facet_type <@A where %impl.elem0.1d3 = bool> [concrete]
// CHECK:STDOUT:   %.Self.e46: %A_where.type.ef9 = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_wit.a35: <witness> = facet_access_witness %.Self.e46 [symbolic_self]
// CHECK:STDOUT:   %impl.elem1: type = impl_witness_access %.Self.as_wit.a35, element1 [symbolic_self]
// CHECK:STDOUT:   %A_where.type.791: type = facet_type <@A where %impl.elem1 = %empty_tuple.type and %impl.elem0.1d3 = bool> [concrete]
// CHECK:STDOUT:   %D.patt: %A_where.type.791 = symbolic_binding_pattern D, 0 [symbolic]
// CHECK:STDOUT:   %D: %A_where.type.791 = bind_symbolic_name D, 0 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.N = import_ref Main//equal_constraint, N, unloaded
// CHECK:STDOUT:   %Main.Equal: %Equal.type.d73 = import_ref Main//equal_constraint, Equal, loaded [concrete = constants.%Equal.517]
// CHECK:STDOUT:   %Main.A = import_ref Main//nested_rewrites, A, unloaded
// CHECK:STDOUT:   %Main.NestedRewrite: %NestedRewrite.type = import_ref Main//nested_rewrites, NestedRewrite, loaded [concrete = constants.%NestedRewrite]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Bool = %Core.Bool
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.169 = import_ref Main//equal_constraint, inst17 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.020 = import_ref Main//equal_constraint, loc5_8, unloaded
// CHECK:STDOUT:   %Main.P = import_ref Main//equal_constraint, P, unloaded
// CHECK:STDOUT:   %Main.import_ref.bdf: %N_where.type = import_ref Main//equal_constraint, loc8_10, loaded [symbolic = @Equal.1.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.b61 = import_ref Main//nested_rewrites, inst17 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.91a = import_ref Main//nested_rewrites, loc5_8, unloaded
// CHECK:STDOUT:   %Main.import_ref.55d = import_ref Main//nested_rewrites, loc6_8, unloaded
// CHECK:STDOUT:   %Main.B = import_ref Main//nested_rewrites, B, unloaded
// CHECK:STDOUT:   %Main.C = import_ref Main//nested_rewrites, C, unloaded
// CHECK:STDOUT:   %Main.import_ref.e0c: %A_where.type.791 = import_ref Main//nested_rewrites, loc9_18, loaded [symbolic = @NestedRewrite.%D (constants.%D)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .N = imports.%Main.N
// CHECK:STDOUT:     .Equal = imports.%Main.Equal
// CHECK:STDOUT:     .A = imports.%Main.A
// CHECK:STDOUT:     .NestedRewrite = imports.%Main.NestedRewrite
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Calls = %Calls.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %Calls.decl: %Calls.type = fn_decl @Calls [concrete = constants.%Calls] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @N [from "equal_constraint.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.169
// CHECK:STDOUT:   .P = imports.%Main.import_ref.020
// CHECK:STDOUT:   witness = (imports.%Main.P)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A [from "nested_rewrites.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.b61
// CHECK:STDOUT:   .B = imports.%Main.import_ref.91a
// CHECK:STDOUT:   .C = imports.%Main.import_ref.55d
// CHECK:STDOUT:   witness = (imports.%Main.B, imports.%Main.C)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Calls() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Equal.ref: %Equal.type.d73 = name_ref Equal, imports.%Main.Equal [concrete = constants.%Equal.517]
// CHECK:STDOUT:   %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:   %.loc19: %N_where.type = converted %bool.make_type, <error> [concrete = <error>]
// CHECK:STDOUT:   %NestedRewrite.ref: %NestedRewrite.type = name_ref NestedRewrite, imports.%Main.NestedRewrite [concrete = constants.%NestedRewrite]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc32: %A_where.type.791 = converted %i32, <error> [concrete = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Equal.1(imports.%Main.import_ref.bdf: %N_where.type) [from "equal_constraint.carbon"] {
// CHECK:STDOUT:   %T: %N_where.type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: %N_where.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.param_patt: %N_where.type);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @NestedRewrite(imports.%Main.import_ref.e0c: %A_where.type.791) [from "nested_rewrites.carbon"] {
// CHECK:STDOUT:   %D: %A_where.type.791 = bind_symbolic_name D, 0 [symbolic = %D (constants.%D)]
// CHECK:STDOUT:   %D.patt: %A_where.type.791 = symbolic_binding_pattern D, 0 [symbolic = %D.patt (constants.%D.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%D.param_patt: %A_where.type.791);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Equal.1(constants.%T) {
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
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.826: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type %I.type [concrete]
// CHECK:STDOUT:   %assoc0.05b: %I.assoc_type = assoc_entity element0, @I.%Member [concrete]
// CHECK:STDOUT:   %.Self: %I.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_wit: <witness> = facet_access_witness %.Self [symbolic_self]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %.Self.as_type, %.Self.as_wit [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %.Self.as_wit, element0 [symbolic_self]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %X.patt: <error> = symbolic_binding_pattern X, 0 [symbolic]
// CHECK:STDOUT:   %RewriteTypeMismatch.type: type = fn_type @RewriteTypeMismatch [concrete]
// CHECK:STDOUT:   %RewriteTypeMismatch: %RewriteTypeMismatch.type = struct_value () [concrete]
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
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .RewriteTypeMismatch = %RewriteTypeMismatch.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %RewriteTypeMismatch.decl: %RewriteTypeMismatch.type = fn_decl @RewriteTypeMismatch [concrete = constants.%RewriteTypeMismatch] {
// CHECK:STDOUT:     %X.patt.loc16_24.1: <error> = symbolic_binding_pattern X, 0 [symbolic = %X.patt.loc16_24.2 (constants.%X.patt)]
// CHECK:STDOUT:     %X.param_patt: <error> = value_param_pattern %X.patt.loc16_24.1, runtime_param<none> [symbolic = %X.patt.loc16_24.2 (constants.%X.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %X.param: <error> = value_param runtime_param<none>
// CHECK:STDOUT:     %.loc16_30.1: type = splice_block %.loc16_30.2 [concrete = <error>] {
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:       %.Self: %I.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.Self.ref: %I.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Member.ref: %I.assoc_type = name_ref Member, @Member.%assoc0 [concrete = constants.%assoc0.05b]
// CHECK:STDOUT:       %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc16_36: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.Self.as_wit: <witness> = facet_access_witness %.Self.ref [symbolic_self = constants.%.Self.as_wit]
// CHECK:STDOUT:       %impl.elem0: type = impl_witness_access %.Self.as_wit, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:       %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:       %.loc16_46: type = converted %int_2, <error> [concrete = <error>]
// CHECK:STDOUT:       %.loc16_30.2: type = where_expr %.Self [concrete = <error>] {
// CHECK:STDOUT:         requirement_rewrite %impl.elem0, <error>
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %X: <error> = bind_symbolic_name X, 0, %X.param [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.826]
// CHECK:STDOUT:   %Member: type = assoc_const_decl @Member [concrete] {
// CHECK:STDOUT:     %assoc0: %I.assoc_type = assoc_entity element0, @I.%Member [concrete = constants.%assoc0.05b]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Member = @Member.%assoc0
// CHECK:STDOUT:   witness = (%Member)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @Member(@I.%Self: %I.type) {
// CHECK:STDOUT:   assoc_const Member:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @RewriteTypeMismatch(%X: <error>) {
// CHECK:STDOUT:   %X.patt.loc16_24.2: <error> = symbolic_binding_pattern X, 0 [symbolic = %X.patt.loc16_24.2 (constants.%X.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%X.param_patt: <error>);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Member(constants.%Self.826) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Member(constants.%I.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @RewriteTypeMismatch(<error>) {
// CHECK:STDOUT:   %X.patt.loc16_24.2 => <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_let.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = facet_type <@A> [concrete]
// CHECK:STDOUT:   %Self.31d: %A.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness () [concrete]
// CHECK:STDOUT:   %.Self: type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %type_where: type = facet_type <type where TODO> [concrete]
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
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .B = %B
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = interface_decl @A [concrete = constants.%A.type] {} {}
// CHECK:STDOUT:   %D.decl: type = class_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness () [concrete = constants.%impl_witness]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %B.patt: %type_where = binding_pattern B
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc19_13.1: type = splice_block %.loc19_13.2 [concrete = constants.%type_where] {
// CHECK:STDOUT:     %.Self: type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %A.ref: type = name_ref A, %A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:     %.loc19_13.2: type = where_expr %.Self [concrete = constants.%type_where] {
// CHECK:STDOUT:       requirement_impls %.Self.ref, %A.ref
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc19_35: %type_where = converted @__global_init.%D.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %B: %type_where = bind_name B, <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A {
// CHECK:STDOUT:   %Self: %A.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.31d]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %D.ref as %A.ref {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_type_does_not_implement_where.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %E.type: type = facet_type <@E> [concrete]
// CHECK:STDOUT:   %Self.19a: %E.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %E.assoc_type: type = assoc_entity_type %E.type [concrete]
// CHECK:STDOUT:   %assoc0.9c3: %E.assoc_type = assoc_entity element0, @E.%F [concrete]
// CHECK:STDOUT:   %assoc1: %E.assoc_type = assoc_entity element1, @E.%G [concrete]
// CHECK:STDOUT:   %.Self.da5: %E.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.Self.as_type.34e: type = facet_access_type %.Self.da5 [symbolic_self]
// CHECK:STDOUT:   %.Self.as_wit.f42: <witness> = facet_access_witness %.Self.da5 [symbolic_self]
// CHECK:STDOUT:   %E.facet.74f: %E.type = facet_value %.Self.as_type.34e, %.Self.as_wit.f42 [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %.Self.as_wit.f42, element0 [symbolic_self]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %impl.elem1.91f: type = impl_witness_access %.Self.as_wit.f42, element1 [symbolic_self]
// CHECK:STDOUT:   %E_where.type.366: type = facet_type <@E where %impl.elem1.91f = %empty_tuple.type and %impl.elem0 = bool> [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %Float.type: type = fn_type @Float [concrete]
// CHECK:STDOUT:   %Float: %Float.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %E_where.type.324: type = facet_type <@E where %impl.elem0 = %empty_struct_type> [concrete]
// CHECK:STDOUT:   %.Self.665: %E_where.type.324 = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type.c08: type = facet_access_type %.Self.665 [symbolic_self]
// CHECK:STDOUT:   %.Self.as_wit.bb3: <witness> = facet_access_witness %.Self.665 [symbolic_self]
// CHECK:STDOUT:   %E.facet.f08: %E.type = facet_value %.Self.as_type.c08, %.Self.as_wit.bb3 [symbolic_self]
// CHECK:STDOUT:   %impl.elem1.244: type = impl_witness_access %.Self.as_wit.bb3, element1 [symbolic_self]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %E_where.type.bef: type = facet_type <@E where %impl.elem1.244 = %i32 and %impl.elem0 = %empty_struct_type> [concrete]
// CHECK:STDOUT:   %E_where.type.503: type = facet_type <@E where %impl.elem0 = %impl.elem1.91f> [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Bool = %Core.Bool
// CHECK:STDOUT:     .Float = %Core.Float
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .E = %E.decl
// CHECK:STDOUT:     .H = %H
// CHECK:STDOUT:     .J = %J
// CHECK:STDOUT:     .K = %K
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %E.decl: type = interface_decl @E [concrete = constants.%E.type] {} {}
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %H.patt: %E_where.type.366 = binding_pattern H
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc18_11.1: type = splice_block %.loc18_11.2 [concrete = constants.%E_where.type.366] {
// CHECK:STDOUT:     %E.ref.loc18: type = name_ref E, %E.decl [concrete = constants.%E.type]
// CHECK:STDOUT:     %.Self.1: %E.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self.da5]
// CHECK:STDOUT:     %.Self.ref.loc18_17: %E.type = name_ref .Self, %.Self.1 [symbolic_self = constants.%.Self.da5]
// CHECK:STDOUT:     %F.ref.loc18: %E.assoc_type = name_ref F, @F.%assoc0 [concrete = constants.%assoc0.9c3]
// CHECK:STDOUT:     %.Self.as_type.loc18_17: type = facet_access_type %.Self.ref.loc18_17 [symbolic_self = constants.%.Self.as_type.34e]
// CHECK:STDOUT:     %.loc18_17: type = converted %.Self.ref.loc18_17, %.Self.as_type.loc18_17 [symbolic_self = constants.%.Self.as_type.34e]
// CHECK:STDOUT:     %.Self.as_wit.loc18_17: <witness> = facet_access_witness %.Self.ref.loc18_17 [symbolic_self = constants.%.Self.as_wit.f42]
// CHECK:STDOUT:     %impl.elem0.loc18: type = impl_witness_access %.Self.as_wit.loc18_17, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc18_22.1: type = value_of_initializer %bool.make_type [concrete = bool]
// CHECK:STDOUT:     %.loc18_22.2: type = converted %bool.make_type, %.loc18_22.1 [concrete = bool]
// CHECK:STDOUT:     %.Self.ref.loc18_31: %E.type = name_ref .Self, %.Self.1 [symbolic_self = constants.%.Self.da5]
// CHECK:STDOUT:     %G.ref.loc18: %E.assoc_type = name_ref G, @G.%assoc1 [concrete = constants.%assoc1]
// CHECK:STDOUT:     %.Self.as_type.loc18_31: type = facet_access_type %.Self.ref.loc18_31 [symbolic_self = constants.%.Self.as_type.34e]
// CHECK:STDOUT:     %.loc18_31: type = converted %.Self.ref.loc18_31, %.Self.as_type.loc18_31 [symbolic_self = constants.%.Self.as_type.34e]
// CHECK:STDOUT:     %.Self.as_wit.loc18_31: <witness> = facet_access_witness %.Self.ref.loc18_31 [symbolic_self = constants.%.Self.as_wit.f42]
// CHECK:STDOUT:     %impl.elem1.loc18: type = impl_witness_access %.Self.as_wit.loc18_31, element1 [symbolic_self = constants.%impl.elem1.91f]
// CHECK:STDOUT:     %.loc18_37.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc18_37.2: type = converted %.loc18_37.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc18_11.2: type = where_expr %.Self.1 [concrete = constants.%E_where.type.366] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc18, %.loc18_22.2
// CHECK:STDOUT:       requirement_rewrite %impl.elem1.loc18, %.loc18_37.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc18_42: %E_where.type.366 = converted @__global_init.%float.make_type, <error> [concrete = <error>]
// CHECK:STDOUT:   %H: %E_where.type.366 = bind_name H, <error>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %J.patt: %E_where.type.bef = binding_pattern J
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc27_27.1: type = splice_block %.loc27_27.2 [concrete = constants.%E_where.type.bef] {
// CHECK:STDOUT:     %E.ref.loc27: type = name_ref E, %E.decl [concrete = constants.%E.type]
// CHECK:STDOUT:     %.Self.2: %E.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self.da5]
// CHECK:STDOUT:     %.Self.ref.loc27_18: %E.type = name_ref .Self, %.Self.2 [symbolic_self = constants.%.Self.da5]
// CHECK:STDOUT:     %F.ref.loc27: %E.assoc_type = name_ref F, @F.%assoc0 [concrete = constants.%assoc0.9c3]
// CHECK:STDOUT:     %.Self.as_type.loc27_18: type = facet_access_type %.Self.ref.loc27_18 [symbolic_self = constants.%.Self.as_type.34e]
// CHECK:STDOUT:     %.loc27_18: type = converted %.Self.ref.loc27_18, %.Self.as_type.loc27_18 [symbolic_self = constants.%.Self.as_type.34e]
// CHECK:STDOUT:     %.Self.as_wit.loc27_18: <witness> = facet_access_witness %.Self.ref.loc27_18 [symbolic_self = constants.%.Self.as_wit.f42]
// CHECK:STDOUT:     %impl.elem0.loc27: type = impl_witness_access %.Self.as_wit.loc27_18, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc27_24.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc27_24.2: type = converted %.loc27_24.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc27_12: type = where_expr %.Self.2 [concrete = constants.%E_where.type.324] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc27, %.loc27_24.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.Self.3: %E_where.type.324 = bind_symbolic_name .Self [symbolic_self = constants.%.Self.665]
// CHECK:STDOUT:     %.Self.ref.loc27_33: %E_where.type.324 = name_ref .Self, %.Self.3 [symbolic_self = constants.%.Self.665]
// CHECK:STDOUT:     %G.ref.loc27: %E.assoc_type = name_ref G, @G.%assoc1 [concrete = constants.%assoc1]
// CHECK:STDOUT:     %.Self.as_type.loc27_33: type = facet_access_type %.Self.ref.loc27_33 [symbolic_self = constants.%.Self.as_type.c08]
// CHECK:STDOUT:     %.loc27_33: type = converted %.Self.ref.loc27_33, %.Self.as_type.loc27_33 [symbolic_self = constants.%.Self.as_type.c08]
// CHECK:STDOUT:     %.Self.as_wit.loc27_33: <witness> = facet_access_witness %.Self.ref.loc27_33 [symbolic_self = constants.%.Self.as_wit.bb3]
// CHECK:STDOUT:     %impl.elem1.loc27: type = impl_witness_access %.Self.as_wit.loc27_33, element1 [symbolic_self = constants.%impl.elem1.244]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc27_27.2: type = where_expr %.Self.3 [concrete = constants.%E_where.type.bef] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem1.loc27, %i32
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc27_45: %E_where.type.bef = converted @__global_init.%bool.make_type.loc27, <error> [concrete = <error>]
// CHECK:STDOUT:   %J: %E_where.type.bef = bind_name J, <error>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %K.patt: %E_where.type.503 = binding_pattern K
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc36_11.1: type = splice_block %.loc36_11.2 [concrete = constants.%E_where.type.503] {
// CHECK:STDOUT:     %E.ref.loc36: type = name_ref E, %E.decl [concrete = constants.%E.type]
// CHECK:STDOUT:     %.Self.4: %E.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self.da5]
// CHECK:STDOUT:     %.Self.ref.loc36_17: %E.type = name_ref .Self, %.Self.4 [symbolic_self = constants.%.Self.da5]
// CHECK:STDOUT:     %F.ref.loc36: %E.assoc_type = name_ref F, @F.%assoc0 [concrete = constants.%assoc0.9c3]
// CHECK:STDOUT:     %.Self.as_type.loc36_17: type = facet_access_type %.Self.ref.loc36_17 [symbolic_self = constants.%.Self.as_type.34e]
// CHECK:STDOUT:     %.loc36_17: type = converted %.Self.ref.loc36_17, %.Self.as_type.loc36_17 [symbolic_self = constants.%.Self.as_type.34e]
// CHECK:STDOUT:     %.Self.as_wit.loc36_17: <witness> = facet_access_witness %.Self.ref.loc36_17 [symbolic_self = constants.%.Self.as_wit.f42]
// CHECK:STDOUT:     %impl.elem0.loc36: type = impl_witness_access %.Self.as_wit.loc36_17, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %.Self.ref.loc36_22: %E.type = name_ref .Self, %.Self.4 [symbolic_self = constants.%.Self.da5]
// CHECK:STDOUT:     %G.ref.loc36: %E.assoc_type = name_ref G, @G.%assoc1 [concrete = constants.%assoc1]
// CHECK:STDOUT:     %.Self.as_type.loc36_27: type = facet_access_type %.Self.ref.loc36_22 [symbolic_self = constants.%.Self.as_type.34e]
// CHECK:STDOUT:     %.loc36_27: type = converted %.Self.ref.loc36_22, %.Self.as_type.loc36_27 [symbolic_self = constants.%.Self.as_type.34e]
// CHECK:STDOUT:     %.Self.as_wit.loc36_27: <witness> = facet_access_witness %.Self.ref.loc36_22 [symbolic_self = constants.%.Self.as_wit.f42]
// CHECK:STDOUT:     %impl.elem1.loc36: type = impl_witness_access %.Self.as_wit.loc36_27, element1 [symbolic_self = constants.%impl.elem1.91f]
// CHECK:STDOUT:     %.loc36_11.2: type = where_expr %.Self.4 [concrete = constants.%E_where.type.503] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc36, %impl.elem1.loc36
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc36_33: %E_where.type.503 = converted @__global_init.%bool.make_type.loc36, <error> [concrete = <error>]
// CHECK:STDOUT:   %K: %E_where.type.503 = bind_name K, <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @E {
// CHECK:STDOUT:   %Self: %E.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.19a]
// CHECK:STDOUT:   %F: type = assoc_const_decl @F [concrete] {
// CHECK:STDOUT:     %assoc0: %E.assoc_type = assoc_entity element0, @E.%F [concrete = constants.%assoc0.9c3]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G: type = assoc_const_decl @G [concrete] {
// CHECK:STDOUT:     %assoc1: %E.assoc_type = assoc_entity element1, @E.%G [concrete = constants.%assoc1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = @F.%assoc0
// CHECK:STDOUT:   .G = @G.%assoc1
// CHECK:STDOUT:   witness = (%F, %G)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @F(@E.%Self: %E.type) {
// CHECK:STDOUT:   assoc_const F:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @G(@E.%Self: %E.type) {
// CHECK:STDOUT:   assoc_const G:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:   %float.make_type: init type = call constants.%Float(%int_64) [concrete = f64]
// CHECK:STDOUT:   %bool.make_type.loc27: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:   %bool.make_type.loc36: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Self.19a) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%Self.19a) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%E.facet.74f) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%E.facet.74f) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%E.facet.f08) {}
// CHECK:STDOUT:

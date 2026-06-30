// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/where_expr/equal_rewrite.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/where_expr/equal_rewrite.carbon

// --- equal_constraint.carbon

library "[[@TEST_NAME]]";

interface N {
  let P: type;
}

//@dump-sem-ir-begin
fn Equal(generic T: N where .P = {});
//@dump-sem-ir-end

// --- nested_rewrites.carbon

library "[[@TEST_NAME]]";

interface A {
  let B: type;
  let C: type;
}

//@dump-sem-ir-begin
fn NestedRewrite(generic D: (A where .B = bool) where .C = ());
//@dump-sem-ir-end

// --- repeated_rewrite.carbon

library "[[@TEST_NAME]]";

interface E {
  let F: type;
}

//@dump-sem-ir-begin
fn OneRewrite(unused generic G: E where .F = i32) {}

fn RepeatedRewrite(generic H: E where .F = i32 and .F = i32) {
  //@dump-sem-ir-end
  OneRewrite(H);
}

fn OneRewriteAgain(generic I: E where .F = i32) {
  RepeatedRewrite(I);
}

// --- rewrites_reordered.carbon

library "[[@TEST_NAME]]";

interface J {
  let K: type;
  let L: type;
}

//@dump-sem-ir-begin
fn Alphabetical(unused generic M: J where .K = () and .L = bool) {}

fn Reversed(generic N: J where .L = bool and .K = ()) {
  //@dump-sem-ir-end
  Alphabetical(N);
}

// --- fail_rewrites_mismatch_right.carbon

library "[[@TEST_NAME]]";

interface O {
  let P: type;
}

fn WithInteger(unused generic Q: O where .P = i32) {}

fn WithBool(generic R: O where .P = bool) {
  // CHECK:STDERR: fail_rewrites_mismatch_right.carbon:[[@LINE+7]]:3: error: cannot convert type `R` that implements `O where .(O.P) = bool` into type implementing `O where .(O.P) = i32` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   WithInteger(R);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR: fail_rewrites_mismatch_right.carbon:[[@LINE-6]]:23: note: initializing generic parameter `Q` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn WithInteger(unused Q:! O where .P = i32) {}
  // CHECK:STDERR:                       ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  WithInteger(R);
}

// --- fail_rewrites_mismatch_left.carbon

library "[[@TEST_NAME]]";

interface S {
  let T: type;
  let U: type;
}

fn WithT(unused generic V: S where .T = ()) {}

fn WithU(generic W: S where .U = ()) {
  // CHECK:STDERR: fail_rewrites_mismatch_left.carbon:[[@LINE+7]]:3: error: cannot convert type `W` that implements `S where .(S.U) = ()` into type implementing `S where .(S.T) = ()` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   WithT(W);
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR: fail_rewrites_mismatch_left.carbon:[[@LINE-6]]:17: note: initializing generic parameter `V` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn WithT(unused V:! S where .T = ()) {}
  // CHECK:STDERR:                 ^~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  WithT(W);
}

// --- fail_import_rewrites.carbon

library "[[@TEST_NAME]]";

import library "equal_constraint";
import library "nested_rewrites";

fn Calls() {
  // CHECK:STDERR: fail_import_rewrites.carbon:[[@LINE+8]]:3: error: cannot convert type `bool` into type implementing `N where .(N.P) = {}` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   Equal(bool);
  // CHECK:STDERR:   ^~~~~~~~~~~
  // CHECK:STDERR: fail_import_rewrites.carbon:[[@LINE-7]]:1: in import [InImport]
  // CHECK:STDERR: equal_constraint.carbon:9:10: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn Equal(T:! N where .P = {});
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Equal(bool);

  // CHECK:STDERR: fail_import_rewrites.carbon:[[@LINE+8]]:3: error: cannot convert type `i32` into type implementing `A where .(A.C) = () and .(A.B) = bool` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   NestedRewrite(i32);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_rewrites.carbon:[[@LINE-17]]:1: in import [InImport]
  // CHECK:STDERR: nested_rewrites.carbon:10:18: note: initializing generic parameter `D` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn NestedRewrite(D:! (A where .B = bool) where .C = ());
  // CHECK:STDERR:                  ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  NestedRewrite(i32);
}

// --- fail_check_rewrite_constraints.carbon

library "[[@TEST_NAME]]";

interface I {
  let Member: type;
}

// `2` can't be converted to the type of `I.Member`
// CHECK:STDERR: fail_check_rewrite_constraints.carbon:[[@LINE+7]]:46: error: cannot implicitly convert non-type value of type `Core.IntLiteral` to `type` [ConversionFailureNonTypeToFacet]
// CHECK:STDERR: fn RewriteTypeMismatch(X:! I where .Member = 2);
// CHECK:STDERR:                                              ^
// CHECK:STDERR: fail_check_rewrite_constraints.carbon:[[@LINE+4]]:46: note: type `Core.IntLiteral` does not implement interface `Core.ImplicitAs(type)` [MissingImplInMemberAccessInContext]
// CHECK:STDERR: fn RewriteTypeMismatch(X:! I where .Member = 2);
// CHECK:STDERR:                                              ^
// CHECK:STDERR:
fn RewriteTypeMismatch(generic X: I where .Member = 2);

// --- fail_todo_let.carbon

library "[[@TEST_NAME]]";

interface A {}
class D {}
impl D as A {}

fn F() {
  // CHECK:STDERR: fail_todo_let.carbon:[[@LINE+4]]:7: error: semantics TODO: `local `let :!` bindings are currently unsupported` [SemanticsTodo]
  // CHECK:STDERR:   let E:! type where .Self impls A = D;
  // CHECK:STDERR:       ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let generic E: type where .Self impls A = D;
}

// --- fail_todo_let_compile_time.carbon

library "[[@TEST_NAME]]";

interface A {}
class D {}
impl D as A {}
// TODO: Compile-time binding are not yet supported at file scope.
// CHECK:STDERR: fail_todo_let_compile_time.carbon:[[@LINE+4]]:5: error: semantics TODO: ``let` compile time binding outside function or interface` [SemanticsTodo]
// CHECK:STDERR: let B:! type where .Self impls A = D;
// CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let generic B: type where .Self impls A = D;

// --- fail_type_does_not_implement_where.carbon

library "[[@TEST_NAME]]";

interface E {
  let F: type;
  let G: type;
}
// Testing how these types get stringified in error messages.

// TODO: This should be a compile-time binding, once that is supported.
// CHECK:STDERR: fail_type_does_not_implement_where.carbon:[[@LINE+4]]:42: error: cannot convert type `f64` into type implementing `E where .(E.F) = bool and .(E.G) = ()` [ConversionFailureTypeToFacet]
// CHECK:STDERR: let H: (E where .F = bool and .G = ()) = f64;
// CHECK:STDERR:                                          ^~~
// CHECK:STDERR:
let H: (E where .F = bool and .G = ()) = f64;

// CHECK:STDERR: fail_type_does_not_implement_where.carbon:[[@LINE+4]]:45: error: cannot convert type `bool` into type implementing `E where .(E.F) = {} and .(E.G) = i32` [ConversionFailureTypeToFacet]
// CHECK:STDERR: let J: ((E where .F = {}) where .G = i32) = bool;
// CHECK:STDERR:                                             ^~~~
// CHECK:STDERR:
let J: ((E where .F = {}) where .G = i32) = bool;

// CHECK:STDERR: fail_type_does_not_implement_where.carbon:[[@LINE+4]]:33: error: cannot convert type `bool` into type implementing `E where .(E.F) = .(E.G)` [ConversionFailureTypeToFacet]
// CHECK:STDERR: let K: (E where .F = .Self.G) = bool;
// CHECK:STDERR:                                 ^~~~
// CHECK:STDERR:
let K: (E where .F = .Self.G) = bool;

// CHECK:STDOUT: --- equal_constraint.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %N.type: type = facet_type <@N> [concrete]
// CHECK:STDOUT:   %N.assoc_type: type = assoc_entity_type @N [concrete]
// CHECK:STDOUT:   %assoc0: %N.assoc_type = assoc_entity element0, @N.WithSelf.%P [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen.197: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.frozen.098: %N.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.frozen.as_type: type = facet_access_type %.Self.frozen.098 [symbolic_self]
// CHECK:STDOUT:   %N.lookup_impl_witness.e97: <witness> = lookup_impl_witness %.Self.frozen.098, @N [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.892: type = impl_witness_access %N.lookup_impl_witness.e97, element0 [symbolic_self]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %.Self: %N.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %N.lookup_impl_witness.98e: <witness> = lookup_impl_witness %.Self, @N [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.48c: type = impl_witness_access %N.lookup_impl_witness.98e, element0 [symbolic_self]
// CHECK:STDOUT:   %N_where.type: type = facet_type <@N where %impl.elem0.48c = %empty_struct_type> [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %N_where.type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: %N_where.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Equal.type: type = fn_type @Equal [concrete]
// CHECK:STDOUT:   %Equal: %Equal.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Equal.decl: %Equal.type = fn_decl @Equal [concrete = constants.%Equal] {
// CHECK:STDOUT:     %T.patt.loc9_11.1: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc9_11.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc9_16.1: type = splice_block %.loc9_16.2 [concrete = constants.%N_where.type] {
// CHECK:STDOUT:       %.Self.frozen.loc9_11: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.197]
// CHECK:STDOUT:       %N.ref: type = name_ref N, file.%N.decl [concrete = constants.%N.type]
// CHECK:STDOUT:       %.Self.frozen.loc9_16: %N.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.098]
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %N.ref [concrete]
// CHECK:STDOUT:       %.Self.ref: %N.type = name_ref .Self, %.Self.frozen.loc9_16 [symbolic_self = constants.%.Self.frozen.098]
// CHECK:STDOUT:       %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:       %.loc9_22: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:       %P.ref: %N.assoc_type = name_ref P, @P.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:       %impl.elem0.loc9_22.1: type = impl_witness_access constants.%N.lookup_impl_witness.e97, element0 [symbolic_self = constants.%impl.elem0.892]
// CHECK:STDOUT:       %.loc9_28.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:       %.loc9_28.2: type = converted %.loc9_28.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:       %rewrite.loc9_25.1 = requirement_rewrite %impl.elem0.loc9_22.1, %.loc9_28.2 [concrete]
// CHECK:STDOUT:       %impl.elem0.loc9_22.2: type = impl_witness_access constants.%N.lookup_impl_witness.98e, element0 [symbolic_self = constants.%impl.elem0.48c]
// CHECK:STDOUT:       %rewrite.loc9_25.2 = requirement_rewrite %impl.elem0.loc9_22.2, %.loc9_28.2 [concrete]
// CHECK:STDOUT:       %.loc9_16.2: type = where_expr [concrete = constants.%N_where.type] {
// CHECK:STDOUT:         %base_facet_type = requirement_base_facet_type %N.ref [concrete]
// CHECK:STDOUT:         %rewrite.loc9_25.2 = requirement_rewrite %impl.elem0.loc9_22.2, %.loc9_28.2 [concrete]
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9_11.2: %N_where.type = symbolic_binding T, 0 [symbolic = %T.loc9_11.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Equal(%T.loc9_11.2: %N_where.type) {
// CHECK:STDOUT:   %T.patt.loc9_11.2: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc9_11.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc9_11.1: %N_where.type = symbolic_binding T, 0 [symbolic = %T.loc9_11.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Equal(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc9_11.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc9_11.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- nested_rewrites.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = facet_type <@A> [concrete]
// CHECK:STDOUT:   %A.assoc_type: type = assoc_entity_type @A [concrete]
// CHECK:STDOUT:   %assoc0: %A.assoc_type = assoc_entity element0, @A.WithSelf.%B [concrete]
// CHECK:STDOUT:   %assoc1: %A.assoc_type = assoc_entity element1, @A.WithSelf.%C [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen.197: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.frozen.d4e: %A.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.frozen.as_type: type = facet_access_type %.Self.frozen.d4e [symbolic_self]
// CHECK:STDOUT:   %A.lookup_impl_witness.5db: <witness> = lookup_impl_witness %.Self.frozen.d4e, @A [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.6ea: type = impl_witness_access %A.lookup_impl_witness.5db, element0 [symbolic_self]
// CHECK:STDOUT:   %.Self: %A.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %A.lookup_impl_witness.937: <witness> = lookup_impl_witness %.Self, @A [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.0e0: type = impl_witness_access %A.lookup_impl_witness.937, element0 [symbolic_self]
// CHECK:STDOUT:   %A_where.type.beb: type = facet_type <@A where %impl.elem0.0e0 = bool> [concrete]
// CHECK:STDOUT:   %impl.elem1.06a: type = impl_witness_access %A.lookup_impl_witness.5db, element1 [symbolic_self]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %impl.elem1.844: type = impl_witness_access %A.lookup_impl_witness.937, element1 [symbolic_self]
// CHECK:STDOUT:   %A_where.type.0ba: type = facet_type <@A where %impl.elem0.0e0 = bool and %impl.elem1.844 = %empty_tuple.type> [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %A_where.type.0ba [concrete]
// CHECK:STDOUT:   %D.patt: %pattern_type = symbolic_binding_pattern D, 0 [symbolic]
// CHECK:STDOUT:   %D: %A_where.type.0ba = symbolic_binding D, 0 [symbolic]
// CHECK:STDOUT:   %NestedRewrite.type: type = fn_type @NestedRewrite [concrete]
// CHECK:STDOUT:   %NestedRewrite: %NestedRewrite.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %NestedRewrite.decl: %NestedRewrite.type = fn_decl @NestedRewrite [concrete = constants.%NestedRewrite] {
// CHECK:STDOUT:     %D.patt.loc10_19.1: %pattern_type = symbolic_binding_pattern D, 0 [symbolic = %D.patt.loc10_19.2 (constants.%D.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc10_42.1: type = splice_block %.loc10_42.2 [concrete = constants.%A_where.type.0ba] {
// CHECK:STDOUT:       %.Self.frozen.loc10_19: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.197]
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:       %.Self.frozen.loc10_25: %A.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.d4e]
// CHECK:STDOUT:       %base_facet_type.loc10_25 = requirement_base_facet_type %A.ref [concrete]
// CHECK:STDOUT:       %.Self.ref.loc10_31: %A.type = name_ref .Self, %.Self.frozen.loc10_25 [symbolic_self = constants.%.Self.frozen.d4e]
// CHECK:STDOUT:       %.Self.as_type.loc10_31: type = facet_access_type %.Self.ref.loc10_31 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:       %.loc10_31: type = converted %.Self.ref.loc10_31, %.Self.as_type.loc10_31 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:       %B.ref: %A.assoc_type = name_ref B, @B.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:       %impl.elem0.loc10_31.1: type = impl_witness_access constants.%A.lookup_impl_witness.5db, element0 [symbolic_self = constants.%impl.elem0.6ea]
// CHECK:STDOUT:       %.loc10_36: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:       %rewrite.loc10_34.1 = requirement_rewrite %impl.elem0.loc10_31.1, %.loc10_36 [concrete]
// CHECK:STDOUT:       %impl.elem0.loc10_31.2: type = impl_witness_access constants.%A.lookup_impl_witness.937, element0 [symbolic_self = constants.%impl.elem0.0e0]
// CHECK:STDOUT:       %rewrite.loc10_34.2 = requirement_rewrite %impl.elem0.loc10_31.2, %.loc10_36 [concrete]
// CHECK:STDOUT:       %.loc10_25: type = where_expr [concrete = constants.%A_where.type.beb] {
// CHECK:STDOUT:         %base_facet_type.loc10_25 = requirement_base_facet_type %A.ref [concrete]
// CHECK:STDOUT:         %rewrite.loc10_34.2 = requirement_rewrite %impl.elem0.loc10_31.2, %.loc10_36 [concrete]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %.Self.frozen.loc10_42: %A.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.d4e]
// CHECK:STDOUT:       %base_facet_type.loc10_42 = requirement_base_facet_type %.loc10_25 [concrete]
// CHECK:STDOUT:       %.Self.ref.loc10_48: %A.type = name_ref .Self, %.Self.frozen.loc10_42 [symbolic_self = constants.%.Self.frozen.d4e]
// CHECK:STDOUT:       %.Self.as_type.loc10_48: type = facet_access_type %.Self.ref.loc10_48 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:       %.loc10_48: type = converted %.Self.ref.loc10_48, %.Self.as_type.loc10_48 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:       %C.ref: %A.assoc_type = name_ref C, @C.%assoc1 [concrete = constants.%assoc1]
// CHECK:STDOUT:       %impl.elem1.loc10_48.1: type = impl_witness_access constants.%A.lookup_impl_witness.5db, element1 [symbolic_self = constants.%impl.elem1.06a]
// CHECK:STDOUT:       %.loc10_54.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc10_54.2: type = converted %.loc10_54.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %rewrite.loc10_51.1 = requirement_rewrite %impl.elem1.loc10_48.1, %.loc10_54.2 [concrete]
// CHECK:STDOUT:       %impl.elem1.loc10_48.2: type = impl_witness_access constants.%A.lookup_impl_witness.937, element1 [symbolic_self = constants.%impl.elem1.844]
// CHECK:STDOUT:       %rewrite.loc10_51.2 = requirement_rewrite %impl.elem1.loc10_48.2, %.loc10_54.2 [concrete]
// CHECK:STDOUT:       %.loc10_42.2: type = where_expr [concrete = constants.%A_where.type.0ba] {
// CHECK:STDOUT:         %base_facet_type.loc10_42 = requirement_base_facet_type %.loc10_25 [concrete]
// CHECK:STDOUT:         %rewrite.loc10_51.2 = requirement_rewrite %impl.elem1.loc10_48.2, %.loc10_54.2 [concrete]
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %D.loc10_19.2: %A_where.type.0ba = symbolic_binding D, 0 [symbolic = %D.loc10_19.1 (constants.%D)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @NestedRewrite(%D.loc10_19.2: %A_where.type.0ba) {
// CHECK:STDOUT:   %D.patt.loc10_19.2: %pattern_type = symbolic_binding_pattern D, 0 [symbolic = %D.patt.loc10_19.2 (constants.%D.patt)]
// CHECK:STDOUT:   %D.loc10_19.1: %A_where.type.0ba = symbolic_binding D, 0 [symbolic = %D.loc10_19.1 (constants.%D)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @NestedRewrite(constants.%D) {
// CHECK:STDOUT:   %D.patt.loc10_19.2 => constants.%D.patt
// CHECK:STDOUT:   %D.loc10_19.1 => constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- repeated_rewrite.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %E.type: type = facet_type <@E> [concrete]
// CHECK:STDOUT:   %E.assoc_type: type = assoc_entity_type @E [concrete]
// CHECK:STDOUT:   %assoc0: %E.assoc_type = assoc_entity element0, @E.WithSelf.%F [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen.197: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.frozen.64b: %E.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.frozen.as_type: type = facet_access_type %.Self.frozen.64b [symbolic_self]
// CHECK:STDOUT:   %E.lookup_impl_witness.b1b: <witness> = lookup_impl_witness %.Self.frozen.64b, @E [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.efe: type = impl_witness_access %E.lookup_impl_witness.b1b, element0 [symbolic_self]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %.Self: %E.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %E.lookup_impl_witness.8bd: <witness> = lookup_impl_witness %.Self, @E [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.554: type = impl_witness_access %E.lookup_impl_witness.8bd, element0 [symbolic_self]
// CHECK:STDOUT:   %E_where.type: type = facet_type <@E where %impl.elem0.554 = %i32> [concrete]
// CHECK:STDOUT:   %pattern_type.b73: type = pattern_type %E_where.type [concrete]
// CHECK:STDOUT:   %G.patt: %pattern_type.b73 = symbolic_binding_pattern G, 0 [symbolic]
// CHECK:STDOUT:   %G: %E_where.type = symbolic_binding G, 0 [symbolic]
// CHECK:STDOUT:   %OneRewrite.type: type = fn_type @OneRewrite [concrete]
// CHECK:STDOUT:   %OneRewrite: %OneRewrite.type = struct_value () [concrete]
// CHECK:STDOUT:   %H.patt: %pattern_type.b73 = symbolic_binding_pattern H, 0 [symbolic]
// CHECK:STDOUT:   %H: %E_where.type = symbolic_binding H, 0 [symbolic]
// CHECK:STDOUT:   %RepeatedRewrite.type: type = fn_type @RepeatedRewrite [concrete]
// CHECK:STDOUT:   %RepeatedRewrite: %RepeatedRewrite.type = struct_value () [concrete]
// CHECK:STDOUT:   %I: %E_where.type = symbolic_binding I, 0 [symbolic]
// CHECK:STDOUT:   %OneRewrite.specific_fn.ba0394.2: <specific function> = specific_function %OneRewrite, @OneRewrite(%I) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %OneRewrite.decl: %OneRewrite.type = fn_decl @OneRewrite [concrete = constants.%OneRewrite] {
// CHECK:STDOUT:     %G.patt.loc9_23.1: %pattern_type.b73 = symbolic_binding_pattern G, 0 [symbolic = %G.patt.loc9_23.2 (constants.%G.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc9_28.1: type = splice_block %.loc9_28.2 [concrete = constants.%E_where.type] {
// CHECK:STDOUT:       %.Self.frozen.loc9_23: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.197]
// CHECK:STDOUT:       %E.ref: type = name_ref E, file.%E.decl [concrete = constants.%E.type]
// CHECK:STDOUT:       %.Self.frozen.loc9_28: %E.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.64b]
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %E.ref [concrete]
// CHECK:STDOUT:       %.Self.ref: %E.type = name_ref .Self, %.Self.frozen.loc9_28 [symbolic_self = constants.%.Self.frozen.64b]
// CHECK:STDOUT:       %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:       %.loc9_34: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:       %F.ref: %E.assoc_type = name_ref F, @F.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:       %impl.elem0.loc9_34.1: type = impl_witness_access constants.%E.lookup_impl_witness.b1b, element0 [symbolic_self = constants.%impl.elem0.efe]
// CHECK:STDOUT:       %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %rewrite.loc9_37.1 = requirement_rewrite %impl.elem0.loc9_34.1, %i32 [concrete]
// CHECK:STDOUT:       %impl.elem0.loc9_34.2: type = impl_witness_access constants.%E.lookup_impl_witness.8bd, element0 [symbolic_self = constants.%impl.elem0.554]
// CHECK:STDOUT:       %rewrite.loc9_37.2 = requirement_rewrite %impl.elem0.loc9_34.2, %i32 [concrete]
// CHECK:STDOUT:       %.loc9_28.2: type = where_expr [concrete = constants.%E_where.type] {
// CHECK:STDOUT:         %base_facet_type = requirement_base_facet_type %E.ref [concrete]
// CHECK:STDOUT:         %rewrite.loc9_37.2 = requirement_rewrite %impl.elem0.loc9_34.2, %i32 [concrete]
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %G.loc9_23.2: %E_where.type = symbolic_binding G, 0 [symbolic = %G.loc9_23.1 (constants.%G)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RepeatedRewrite.decl: %RepeatedRewrite.type = fn_decl @RepeatedRewrite [concrete = constants.%RepeatedRewrite] {
// CHECK:STDOUT:     %H.patt.loc11_21.1: %pattern_type.b73 = symbolic_binding_pattern H, 0 [symbolic = %H.patt.loc11_21.2 (constants.%H.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11_26.1: type = splice_block %.loc11_26.2 [concrete = constants.%E_where.type] {
// CHECK:STDOUT:       %.Self.frozen.loc11_21: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.197]
// CHECK:STDOUT:       %E.ref: type = name_ref E, file.%E.decl [concrete = constants.%E.type]
// CHECK:STDOUT:       %.Self.frozen.loc11_26: %E.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.64b]
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %E.ref [concrete]
// CHECK:STDOUT:       %.Self.ref.loc11_32: %E.type = name_ref .Self, %.Self.frozen.loc11_26 [symbolic_self = constants.%.Self.frozen.64b]
// CHECK:STDOUT:       %.Self.as_type.loc11_32: type = facet_access_type %.Self.ref.loc11_32 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:       %.loc11_32: type = converted %.Self.ref.loc11_32, %.Self.as_type.loc11_32 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:       %F.ref.loc11_32: %E.assoc_type = name_ref F, @F.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:       %impl.elem0.loc11_32.1: type = impl_witness_access constants.%E.lookup_impl_witness.b1b, element0 [symbolic_self = constants.%impl.elem0.efe]
// CHECK:STDOUT:       %i32.loc11_37: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %rewrite.loc11_35.1 = requirement_rewrite %impl.elem0.loc11_32.1, %i32.loc11_37 [concrete]
// CHECK:STDOUT:       %.Self.ref.loc11_45: %E.type = name_ref .Self, %.Self.frozen.loc11_26 [symbolic_self = constants.%.Self.frozen.64b]
// CHECK:STDOUT:       %.Self.as_type.loc11_45: type = facet_access_type %.Self.ref.loc11_45 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:       %.loc11_45: type = converted %.Self.ref.loc11_45, %.Self.as_type.loc11_45 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:       %F.ref.loc11_45: %E.assoc_type = name_ref F, @F.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:       %impl.elem0.loc11_45.1: type = impl_witness_access constants.%E.lookup_impl_witness.b1b, element0 [symbolic_self = constants.%impl.elem0.efe]
// CHECK:STDOUT:       %impl.elem0.subst.loc11_45.1: type = impl_witness_access_substituted %impl.elem0.loc11_45.1, %i32.loc11_37 [concrete = constants.%i32]
// CHECK:STDOUT:       %i32.loc11_50: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %rewrite.loc11_48.1 = requirement_rewrite %impl.elem0.subst.loc11_45.1, %i32.loc11_50 [concrete]
// CHECK:STDOUT:       %impl.elem0.loc11_32.2: type = impl_witness_access constants.%E.lookup_impl_witness.8bd, element0 [symbolic_self = constants.%impl.elem0.554]
// CHECK:STDOUT:       %rewrite.loc11_35.2 = requirement_rewrite %impl.elem0.loc11_32.2, %i32.loc11_37 [concrete]
// CHECK:STDOUT:       %impl.elem0.loc11_45.2: type = impl_witness_access constants.%E.lookup_impl_witness.8bd, element0 [symbolic_self = constants.%impl.elem0.554]
// CHECK:STDOUT:       %impl.elem0.subst.loc11_45.2: type = impl_witness_access_substituted %impl.elem0.loc11_45.2, %i32.loc11_37 [concrete = constants.%i32]
// CHECK:STDOUT:       %rewrite.loc11_48.2 = requirement_rewrite %impl.elem0.subst.loc11_45.2, %i32.loc11_50 [concrete]
// CHECK:STDOUT:       %.loc11_26.2: type = where_expr [concrete = constants.%E_where.type] {
// CHECK:STDOUT:         %base_facet_type = requirement_base_facet_type %E.ref [concrete]
// CHECK:STDOUT:         %rewrite.loc11_35.2 = requirement_rewrite %impl.elem0.loc11_32.2, %i32.loc11_37 [concrete]
// CHECK:STDOUT:         %rewrite.loc11_48.2 = requirement_rewrite %impl.elem0.subst.loc11_45.2, %i32.loc11_50 [concrete]
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %H.loc11_21.2: %E_where.type = symbolic_binding H, 0 [symbolic = %H.loc11_21.1 (constants.%H)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @OneRewrite(%G.loc9_23.2: %E_where.type) {
// CHECK:STDOUT:   %G.patt.loc9_23.2: %pattern_type.b73 = symbolic_binding_pattern G, 0 [symbolic = %G.patt.loc9_23.2 (constants.%G.patt)]
// CHECK:STDOUT:   %G.loc9_23.1: %E_where.type = symbolic_binding G, 0 [symbolic = %G.loc9_23.1 (constants.%G)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:
// CHECK:STDOUT:   !observes:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @RepeatedRewrite(%H.loc11_21.2: %E_where.type) {
// CHECK:STDOUT:   %H.patt.loc11_21.2: %pattern_type.b73 = symbolic_binding_pattern H, 0 [symbolic = %H.patt.loc11_21.2 (constants.%H.patt)]
// CHECK:STDOUT:   %H.loc11_21.1: %E_where.type = symbolic_binding H, 0 [symbolic = %H.loc11_21.1 (constants.%H)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   !observes:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OneRewrite(constants.%G) {
// CHECK:STDOUT:   %G.patt.loc9_23.2 => constants.%G.patt
// CHECK:STDOUT:   %G.loc9_23.1 => constants.%G
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @RepeatedRewrite(constants.%H) {
// CHECK:STDOUT:   %H.patt.loc11_21.2 => constants.%H.patt
// CHECK:STDOUT:   %H.loc11_21.1 => constants.%H
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OneRewrite(constants.%H) {
// CHECK:STDOUT:   %G.patt.loc9_23.2 => constants.%G.patt
// CHECK:STDOUT:   %G.loc9_23.1 => constants.%H
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @RepeatedRewrite(constants.%I) {
// CHECK:STDOUT:   %H.patt.loc11_21.2 => constants.%H.patt
// CHECK:STDOUT:   %H.loc11_21.1 => constants.%I
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %OneRewrite.specific_fn.loc13_3.2 => constants.%OneRewrite.specific_fn.ba0394.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OneRewrite(constants.%I) {
// CHECK:STDOUT:   %G.patt.loc9_23.2 => constants.%G.patt
// CHECK:STDOUT:   %G.loc9_23.1 => constants.%I
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- rewrites_reordered.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type @J [concrete]
// CHECK:STDOUT:   %assoc0: %J.assoc_type = assoc_entity element0, @J.WithSelf.%K [concrete]
// CHECK:STDOUT:   %assoc1: %J.assoc_type = assoc_entity element1, @J.WithSelf.%L [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen.197: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.frozen.99c: %J.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.frozen.as_type: type = facet_access_type %.Self.frozen.99c [symbolic_self]
// CHECK:STDOUT:   %J.lookup_impl_witness.cb1: <witness> = lookup_impl_witness %.Self.frozen.99c, @J [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.221: type = impl_witness_access %J.lookup_impl_witness.cb1, element0 [symbolic_self]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %impl.elem1.eed: type = impl_witness_access %J.lookup_impl_witness.cb1, element1 [symbolic_self]
// CHECK:STDOUT:   %.Self: %J.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %J.lookup_impl_witness.df8: <witness> = lookup_impl_witness %.Self, @J [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.bbd: type = impl_witness_access %J.lookup_impl_witness.df8, element0 [symbolic_self]
// CHECK:STDOUT:   %impl.elem1.fb5: type = impl_witness_access %J.lookup_impl_witness.df8, element1 [symbolic_self]
// CHECK:STDOUT:   %J_where.type: type = facet_type <@J where %impl.elem0.bbd = %empty_tuple.type and %impl.elem1.fb5 = bool> [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %J_where.type [concrete]
// CHECK:STDOUT:   %M.patt: %pattern_type = symbolic_binding_pattern M, 0 [symbolic]
// CHECK:STDOUT:   %M: %J_where.type = symbolic_binding M, 0 [symbolic]
// CHECK:STDOUT:   %Alphabetical.type: type = fn_type @Alphabetical [concrete]
// CHECK:STDOUT:   %Alphabetical: %Alphabetical.type = struct_value () [concrete]
// CHECK:STDOUT:   %N.patt: %pattern_type = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %N: %J_where.type = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %Reversed.type: type = fn_type @Reversed [concrete]
// CHECK:STDOUT:   %Reversed: %Reversed.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Alphabetical.decl: %Alphabetical.type = fn_decl @Alphabetical [concrete = constants.%Alphabetical] {
// CHECK:STDOUT:     %M.patt.loc10_25.1: %pattern_type = symbolic_binding_pattern M, 0 [symbolic = %M.patt.loc10_25.2 (constants.%M.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc10_30.1: type = splice_block %.loc10_30.2 [concrete = constants.%J_where.type] {
// CHECK:STDOUT:       %.Self.frozen.loc10_25: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.197]
// CHECK:STDOUT:       %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:       %.Self.frozen.loc10_30: %J.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.99c]
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %J.ref [concrete]
// CHECK:STDOUT:       %.Self.ref.loc10_36: %J.type = name_ref .Self, %.Self.frozen.loc10_30 [symbolic_self = constants.%.Self.frozen.99c]
// CHECK:STDOUT:       %.Self.as_type.loc10_36: type = facet_access_type %.Self.ref.loc10_36 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:       %.loc10_36: type = converted %.Self.ref.loc10_36, %.Self.as_type.loc10_36 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:       %K.ref: %J.assoc_type = name_ref K, @K.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:       %impl.elem0.loc10_36.1: type = impl_witness_access constants.%J.lookup_impl_witness.cb1, element0 [symbolic_self = constants.%impl.elem0.221]
// CHECK:STDOUT:       %.loc10_42.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc10_42.2: type = converted %.loc10_42.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %rewrite.loc10_39.1 = requirement_rewrite %impl.elem0.loc10_36.1, %.loc10_42.2 [concrete]
// CHECK:STDOUT:       %.Self.ref.loc10_48: %J.type = name_ref .Self, %.Self.frozen.loc10_30 [symbolic_self = constants.%.Self.frozen.99c]
// CHECK:STDOUT:       %.Self.as_type.loc10_48: type = facet_access_type %.Self.ref.loc10_48 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:       %.loc10_48: type = converted %.Self.ref.loc10_48, %.Self.as_type.loc10_48 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:       %L.ref: %J.assoc_type = name_ref L, @L.%assoc1 [concrete = constants.%assoc1]
// CHECK:STDOUT:       %impl.elem1.loc10_48.1: type = impl_witness_access constants.%J.lookup_impl_witness.cb1, element1 [symbolic_self = constants.%impl.elem1.eed]
// CHECK:STDOUT:       %.loc10_53: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:       %rewrite.loc10_51.1 = requirement_rewrite %impl.elem1.loc10_48.1, %.loc10_53 [concrete]
// CHECK:STDOUT:       %impl.elem0.loc10_36.2: type = impl_witness_access constants.%J.lookup_impl_witness.df8, element0 [symbolic_self = constants.%impl.elem0.bbd]
// CHECK:STDOUT:       %rewrite.loc10_39.2 = requirement_rewrite %impl.elem0.loc10_36.2, %.loc10_42.2 [concrete]
// CHECK:STDOUT:       %impl.elem1.loc10_48.2: type = impl_witness_access constants.%J.lookup_impl_witness.df8, element1 [symbolic_self = constants.%impl.elem1.fb5]
// CHECK:STDOUT:       %rewrite.loc10_51.2 = requirement_rewrite %impl.elem1.loc10_48.2, %.loc10_53 [concrete]
// CHECK:STDOUT:       %.loc10_30.2: type = where_expr [concrete = constants.%J_where.type] {
// CHECK:STDOUT:         %base_facet_type = requirement_base_facet_type %J.ref [concrete]
// CHECK:STDOUT:         %rewrite.loc10_39.2 = requirement_rewrite %impl.elem0.loc10_36.2, %.loc10_42.2 [concrete]
// CHECK:STDOUT:         %rewrite.loc10_51.2 = requirement_rewrite %impl.elem1.loc10_48.2, %.loc10_53 [concrete]
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %M.loc10_25.2: %J_where.type = symbolic_binding M, 0 [symbolic = %M.loc10_25.1 (constants.%M)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Reversed.decl: %Reversed.type = fn_decl @Reversed [concrete = constants.%Reversed] {
// CHECK:STDOUT:     %N.patt.loc12_14.1: %pattern_type = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc12_14.2 (constants.%N.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc12_19.1: type = splice_block %.loc12_19.2 [concrete = constants.%J_where.type] {
// CHECK:STDOUT:       %.Self.frozen.loc12_14: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.197]
// CHECK:STDOUT:       %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:       %.Self.frozen.loc12_19: %J.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.99c]
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %J.ref [concrete]
// CHECK:STDOUT:       %.Self.ref.loc12_25: %J.type = name_ref .Self, %.Self.frozen.loc12_19 [symbolic_self = constants.%.Self.frozen.99c]
// CHECK:STDOUT:       %.Self.as_type.loc12_25: type = facet_access_type %.Self.ref.loc12_25 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:       %.loc12_25: type = converted %.Self.ref.loc12_25, %.Self.as_type.loc12_25 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:       %L.ref: %J.assoc_type = name_ref L, @L.%assoc1 [concrete = constants.%assoc1]
// CHECK:STDOUT:       %impl.elem1.loc12_25.1: type = impl_witness_access constants.%J.lookup_impl_witness.cb1, element1 [symbolic_self = constants.%impl.elem1.eed]
// CHECK:STDOUT:       %.loc12_30: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:       %rewrite.loc12_28.1 = requirement_rewrite %impl.elem1.loc12_25.1, %.loc12_30 [concrete]
// CHECK:STDOUT:       %.Self.ref.loc12_39: %J.type = name_ref .Self, %.Self.frozen.loc12_19 [symbolic_self = constants.%.Self.frozen.99c]
// CHECK:STDOUT:       %.Self.as_type.loc12_39: type = facet_access_type %.Self.ref.loc12_39 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:       %.loc12_39: type = converted %.Self.ref.loc12_39, %.Self.as_type.loc12_39 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:       %K.ref: %J.assoc_type = name_ref K, @K.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:       %impl.elem0.loc12_39.1: type = impl_witness_access constants.%J.lookup_impl_witness.cb1, element0 [symbolic_self = constants.%impl.elem0.221]
// CHECK:STDOUT:       %.loc12_45.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc12_45.2: type = converted %.loc12_45.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %rewrite.loc12_42.1 = requirement_rewrite %impl.elem0.loc12_39.1, %.loc12_45.2 [concrete]
// CHECK:STDOUT:       %impl.elem1.loc12_25.2: type = impl_witness_access constants.%J.lookup_impl_witness.df8, element1 [symbolic_self = constants.%impl.elem1.fb5]
// CHECK:STDOUT:       %rewrite.loc12_28.2 = requirement_rewrite %impl.elem1.loc12_25.2, %.loc12_30 [concrete]
// CHECK:STDOUT:       %impl.elem0.loc12_39.2: type = impl_witness_access constants.%J.lookup_impl_witness.df8, element0 [symbolic_self = constants.%impl.elem0.bbd]
// CHECK:STDOUT:       %rewrite.loc12_42.2 = requirement_rewrite %impl.elem0.loc12_39.2, %.loc12_45.2 [concrete]
// CHECK:STDOUT:       %.loc12_19.2: type = where_expr [concrete = constants.%J_where.type] {
// CHECK:STDOUT:         %base_facet_type = requirement_base_facet_type %J.ref [concrete]
// CHECK:STDOUT:         %rewrite.loc12_28.2 = requirement_rewrite %impl.elem1.loc12_25.2, %.loc12_30 [concrete]
// CHECK:STDOUT:         %rewrite.loc12_42.2 = requirement_rewrite %impl.elem0.loc12_39.2, %.loc12_45.2 [concrete]
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc12_14.2: %J_where.type = symbolic_binding N, 0 [symbolic = %N.loc12_14.1 (constants.%N)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Alphabetical(%M.loc10_25.2: %J_where.type) {
// CHECK:STDOUT:   %M.patt.loc10_25.2: %pattern_type = symbolic_binding_pattern M, 0 [symbolic = %M.patt.loc10_25.2 (constants.%M.patt)]
// CHECK:STDOUT:   %M.loc10_25.1: %J_where.type = symbolic_binding M, 0 [symbolic = %M.loc10_25.1 (constants.%M)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:
// CHECK:STDOUT:   !observes:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Reversed(%N.loc12_14.2: %J_where.type) {
// CHECK:STDOUT:   %N.patt.loc12_14.2: %pattern_type = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc12_14.2 (constants.%N.patt)]
// CHECK:STDOUT:   %N.loc12_14.1: %J_where.type = symbolic_binding N, 0 [symbolic = %N.loc12_14.1 (constants.%N)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   !observes:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Alphabetical(constants.%M) {
// CHECK:STDOUT:   %M.patt.loc10_25.2 => constants.%M.patt
// CHECK:STDOUT:   %M.loc10_25.1 => constants.%M
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Reversed(constants.%N) {
// CHECK:STDOUT:   %N.patt.loc12_14.2 => constants.%N.patt
// CHECK:STDOUT:   %N.loc12_14.1 => constants.%N
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Alphabetical(constants.%N) {
// CHECK:STDOUT:   %M.patt.loc10_25.2 => constants.%M.patt
// CHECK:STDOUT:   %M.loc10_25.1 => constants.%N
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:

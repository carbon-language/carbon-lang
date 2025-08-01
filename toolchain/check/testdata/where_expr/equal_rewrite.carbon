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
  let P:! type;
}

//@dump-sem-ir-begin
fn Equal(T:! N where .P = {});
//@dump-sem-ir-end

// --- nested_rewrites.carbon

library "[[@TEST_NAME]]";

interface A {
  let B:! type;
  let C:! type;
}

//@dump-sem-ir-begin
fn NestedRewrite(D:! (A where .B = bool) where .C = ());
//@dump-sem-ir-end

// --- repeated_rewrite.carbon

library "[[@TEST_NAME]]";

interface E {
  let F:! type;
}

//@dump-sem-ir-begin
fn OneRewrite(G:! E where .F = i32) {}

fn RepeatedRewrite(H:! E where .F = i32 and .F = i32) {
  //@dump-sem-ir-end
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

//@dump-sem-ir-begin
fn Alphabetical(M:! J where .K = () and .L = bool) {}

fn Reversed(N:! J where .L = bool and .K = ()) {
  //@dump-sem-ir-end
  Alphabetical(N);
}

// --- todo_fail_rewrites_mismatch_right.carbon

library "[[@TEST_NAME]]";

interface O {
  let P:! type;
}

fn WithInteger(Q:! O where .P = i32) {}

fn WithBool(R:! O where .P = bool) {
  // TODO: This should fail.
  WithInteger(R);
}

// --- todo_fail_rewrites_mismatch_left.carbon

library "[[@TEST_NAME]]";

interface S {
  let T:! type;
  let U:! type;
}

fn WithT(V:! S where .T = ()) {}

fn WithU(W:! S where .U = ()) {
  // TODO: This should fail.
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
  // CHECK:STDERR:          ^
  // CHECK:STDERR:
  Equal(bool);

  // CHECK:STDERR: fail_import_rewrites.carbon:[[@LINE+8]]:3: error: cannot convert type `i32` into type implementing `A where .(A.B) = bool and .(A.C) = ()` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   NestedRewrite(i32);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_rewrites.carbon:[[@LINE-17]]:1: in import [InImport]
  // CHECK:STDERR: nested_rewrites.carbon:10:18: note: initializing generic parameter `D` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn NestedRewrite(D:! (A where .B = bool) where .C = ());
  // CHECK:STDERR:                  ^
  // CHECK:STDERR:
  NestedRewrite(i32);
}

// --- fail_check_rewrite_constraints.carbon

library "[[@TEST_NAME]]";

interface I {
  let Member:! type;
}

// `2` can't be converted to the type of `I.Member`
// CHECK:STDERR: fail_check_rewrite_constraints.carbon:[[@LINE+7]]:46: error: cannot implicitly convert non-type value of type `Core.IntLiteral` to `type` [ConversionFailureNonTypeToFacet]
// CHECK:STDERR: fn RewriteTypeMismatch(X:! I where .Member = 2);
// CHECK:STDERR:                                              ^
// CHECK:STDERR: fail_check_rewrite_constraints.carbon:[[@LINE+4]]:46: note: type `Core.IntLiteral` does not implement interface `Core.ImplicitAs(type)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: fn RewriteTypeMismatch(X:! I where .Member = 2);
// CHECK:STDERR:                                              ^
// CHECK:STDERR:
fn RewriteTypeMismatch(X:! I where .Member = 2);

// --- todo_let.carbon

library "[[@TEST_NAME]]";

interface A {}
class D {}
impl D as A {}
// TODO: This should be a compile-time binding, once that is supported at file scope.
let B: type where .Self impls A = D;

fn F() {
  let E:! type where .Self impls A = D;
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
let B:! type where .Self impls A = D;

// --- fail_type_does_not_implement_where.carbon

library "[[@TEST_NAME]]";

interface E {
  let F:! type;
  let G:! type;
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
// CHECK:STDOUT:   %assoc0: %N.assoc_type = assoc_entity element0, @N.%P [concrete]
// CHECK:STDOUT:   %.Self: %N.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %N.lookup_impl_witness: <witness> = lookup_impl_witness %.Self, @N [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %N.lookup_impl_witness, element0 [symbolic_self]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %N_where.type: type = facet_type <@N where %impl.elem0 = %empty_struct_type> [concrete]
// CHECK:STDOUT:   %T: %N_where.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %N_where.type [concrete]
// CHECK:STDOUT:   %Equal.type: type = fn_type @Equal [concrete]
// CHECK:STDOUT:   %Equal: %Equal.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Equal.decl: %Equal.type = fn_decl @Equal [concrete = constants.%Equal] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc9_16.1: type = splice_block %.loc9_16.2 [concrete = constants.%N_where.type] {
// CHECK:STDOUT:       %N.ref: type = name_ref N, file.%N.decl [concrete = constants.%N.type]
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %.Self.ref: %N.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %P.ref: %N.assoc_type = name_ref P, @P.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:       %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc9_22: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %impl.elem0: type = impl_witness_access constants.%N.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:       %.loc9_28.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:       %.loc9_28.2: type = converted %.loc9_28.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:       %.loc9_16.2: type = where_expr %.Self [concrete = constants.%N_where.type] {
// CHECK:STDOUT:         requirement_rewrite %impl.elem0, %.loc9_28.2
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9_10.2: %N_where.type = bind_symbolic_name T, 0 [symbolic = %T.loc9_10.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Equal(%T.loc9_10.2: %N_where.type) {
// CHECK:STDOUT:   %T.loc9_10.1: %N_where.type = bind_symbolic_name T, 0 [symbolic = %T.loc9_10.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Equal(constants.%T) {
// CHECK:STDOUT:   %T.loc9_10.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- nested_rewrites.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = facet_type <@A> [concrete]
// CHECK:STDOUT:   %A.assoc_type: type = assoc_entity_type @A [concrete]
// CHECK:STDOUT:   %assoc0: %A.assoc_type = assoc_entity element0, @A.%B [concrete]
// CHECK:STDOUT:   %assoc1: %A.assoc_type = assoc_entity element1, @A.%C [concrete]
// CHECK:STDOUT:   %.Self.3ca: %A.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type.adb: type = facet_access_type %.Self.3ca [symbolic_self]
// CHECK:STDOUT:   %A.lookup_impl_witness.5ad: <witness> = lookup_impl_witness %.Self.3ca, @A [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %A.lookup_impl_witness.5ad, element0 [symbolic_self]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %A_where.type.ef7: type = facet_type <@A where %impl.elem0 = bool> [concrete]
// CHECK:STDOUT:   %.Self.60a: %A_where.type.ef7 = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type.e64: type = facet_access_type %.Self.60a [symbolic_self]
// CHECK:STDOUT:   %A.lookup_impl_witness.138: <witness> = lookup_impl_witness %.Self.60a, @A [symbolic_self]
// CHECK:STDOUT:   %impl.elem1: type = impl_witness_access %A.lookup_impl_witness.138, element1 [symbolic_self]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %A_where.type.2d0: type = facet_type <@A where %impl.elem0 = bool and %impl.elem1 = %empty_tuple.type> [concrete]
// CHECK:STDOUT:   %D: %A_where.type.2d0 = bind_symbolic_name D, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.24c: type = pattern_type %A_where.type.2d0 [concrete]
// CHECK:STDOUT:   %NestedRewrite.type: type = fn_type @NestedRewrite [concrete]
// CHECK:STDOUT:   %NestedRewrite: %NestedRewrite.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %NestedRewrite.decl: %NestedRewrite.type = fn_decl @NestedRewrite [concrete = constants.%NestedRewrite] {
// CHECK:STDOUT:     %D.patt: %pattern_type.24c = symbolic_binding_pattern D, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc10_42.1: type = splice_block %.loc10_42.2 [concrete = constants.%A_where.type.2d0] {
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %.Self.ref.loc10_31: %A.type = name_ref .Self, %.Self.1 [symbolic_self = constants.%.Self.3ca]
// CHECK:STDOUT:       %B.ref: %A.assoc_type = name_ref B, @B.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:       %.Self.as_type.loc10_31: type = facet_access_type %.Self.ref.loc10_31 [symbolic_self = constants.%.Self.as_type.adb]
// CHECK:STDOUT:       %.loc10_31: type = converted %.Self.ref.loc10_31, %.Self.as_type.loc10_31 [symbolic_self = constants.%.Self.as_type.adb]
// CHECK:STDOUT:       %impl.elem0: type = impl_witness_access constants.%A.lookup_impl_witness.5ad, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:       %Bool.call: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %.loc10_36.1: type = value_of_initializer %Bool.call [concrete = bool]
// CHECK:STDOUT:       %.loc10_36.2: type = converted %Bool.call, %.loc10_36.1 [concrete = bool]
// CHECK:STDOUT:       %.loc10_25: type = where_expr %.Self.1 [concrete = constants.%A_where.type.ef7] {
// CHECK:STDOUT:         requirement_rewrite %impl.elem0, %.loc10_36.2
// CHECK:STDOUT:       }
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %.Self.ref.loc10_48: %A_where.type.ef7 = name_ref .Self, %.Self.2 [symbolic_self = constants.%.Self.60a]
// CHECK:STDOUT:       %C.ref: %A.assoc_type = name_ref C, @C.%assoc1 [concrete = constants.%assoc1]
// CHECK:STDOUT:       %.Self.as_type.loc10_48: type = facet_access_type %.Self.ref.loc10_48 [symbolic_self = constants.%.Self.as_type.e64]
// CHECK:STDOUT:       %.loc10_48: type = converted %.Self.ref.loc10_48, %.Self.as_type.loc10_48 [symbolic_self = constants.%.Self.as_type.e64]
// CHECK:STDOUT:       %impl.elem1: type = impl_witness_access constants.%A.lookup_impl_witness.138, element1 [symbolic_self = constants.%impl.elem1]
// CHECK:STDOUT:       %.loc10_54.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc10_54.2: type = converted %.loc10_54.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %.loc10_42.2: type = where_expr %.Self.2 [concrete = constants.%A_where.type.2d0] {
// CHECK:STDOUT:         requirement_rewrite %impl.elem1, %.loc10_54.2
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %D.loc10_18.2: %A_where.type.2d0 = bind_symbolic_name D, 0 [symbolic = %D.loc10_18.1 (constants.%D)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @NestedRewrite(%D.loc10_18.2: %A_where.type.2d0) {
// CHECK:STDOUT:   %D.loc10_18.1: %A_where.type.2d0 = bind_symbolic_name D, 0 [symbolic = %D.loc10_18.1 (constants.%D)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @NestedRewrite(constants.%D) {
// CHECK:STDOUT:   %D.loc10_18.1 => constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- repeated_rewrite.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %E.type: type = facet_type <@E> [concrete]
// CHECK:STDOUT:   %E.assoc_type: type = assoc_entity_type @E [concrete]
// CHECK:STDOUT:   %assoc0: %E.assoc_type = assoc_entity element0, @E.%F [concrete]
// CHECK:STDOUT:   %.Self: %E.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %E.lookup_impl_witness: <witness> = lookup_impl_witness %.Self, @E [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %E.lookup_impl_witness, element0 [symbolic_self]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %E_where.type: type = facet_type <@E where %impl.elem0 = %i32> [concrete]
// CHECK:STDOUT:   %G: %E_where.type = bind_symbolic_name G, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.c64: type = pattern_type %E_where.type [concrete]
// CHECK:STDOUT:   %OneRewrite.type: type = fn_type @OneRewrite [concrete]
// CHECK:STDOUT:   %OneRewrite: %OneRewrite.type = struct_value () [concrete]
// CHECK:STDOUT:   %H: %E_where.type = bind_symbolic_name H, 0 [symbolic]
// CHECK:STDOUT:   %RepeatedRewrite.type: type = fn_type @RepeatedRewrite [concrete]
// CHECK:STDOUT:   %RepeatedRewrite: %RepeatedRewrite.type = struct_value () [concrete]
// CHECK:STDOUT:   %I: %E_where.type = bind_symbolic_name I, 0 [symbolic]
// CHECK:STDOUT:   %OneRewrite.specific_fn.7fa5d5.2: <specific function> = specific_function %OneRewrite, @OneRewrite(%I) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %OneRewrite.decl: %OneRewrite.type = fn_decl @OneRewrite [concrete = constants.%OneRewrite] {
// CHECK:STDOUT:     %G.patt: %pattern_type.c64 = symbolic_binding_pattern G, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc9_21.1: type = splice_block %.loc9_21.2 [concrete = constants.%E_where.type] {
// CHECK:STDOUT:       %E.ref: type = name_ref E, file.%E.decl [concrete = constants.%E.type]
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %.Self.ref: %E.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %F.ref: %E.assoc_type = name_ref F, @F.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:       %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc9_27: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %impl.elem0: type = impl_witness_access constants.%E.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %.loc9_21.2: type = where_expr %.Self [concrete = constants.%E_where.type] {
// CHECK:STDOUT:         requirement_rewrite %impl.elem0, %i32
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %G.loc9_15.2: %E_where.type = bind_symbolic_name G, 0 [symbolic = %G.loc9_15.1 (constants.%G)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RepeatedRewrite.decl: %RepeatedRewrite.type = fn_decl @RepeatedRewrite [concrete = constants.%RepeatedRewrite] {
// CHECK:STDOUT:     %H.patt: %pattern_type.c64 = symbolic_binding_pattern H, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11_26.1: type = splice_block %.loc11_26.2 [concrete = constants.%E_where.type] {
// CHECK:STDOUT:       %E.ref: type = name_ref E, file.%E.decl [concrete = constants.%E.type]
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %.Self.ref.loc11_32: %E.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %F.ref.loc11_32: %E.assoc_type = name_ref F, @F.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:       %.Self.as_type.loc11_32: type = facet_access_type %.Self.ref.loc11_32 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc11_32: type = converted %.Self.ref.loc11_32, %.Self.as_type.loc11_32 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %impl.elem0.loc11_32: type = impl_witness_access constants.%E.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:       %int_32.loc11_37: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc11_37: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %.Self.ref.loc11_45: %E.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %F.ref.loc11_45: %E.assoc_type = name_ref F, @F.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:       %.Self.as_type.loc11_45: type = facet_access_type %.Self.ref.loc11_45 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc11_45: type = converted %.Self.ref.loc11_45, %.Self.as_type.loc11_45 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %impl.elem0.loc11_45: type = impl_witness_access constants.%E.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:       %int_32.loc11_50: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc11_50: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %.loc11_26.2: type = where_expr %.Self [concrete = constants.%E_where.type] {
// CHECK:STDOUT:         requirement_rewrite %impl.elem0.loc11_32, %i32.loc11_37
// CHECK:STDOUT:         requirement_rewrite %impl.elem0.loc11_45, %i32.loc11_50
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %H.loc11_20.2: %E_where.type = bind_symbolic_name H, 0 [symbolic = %H.loc11_20.1 (constants.%H)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @OneRewrite(%G.loc9_15.2: %E_where.type) {
// CHECK:STDOUT:   %G.loc9_15.1: %E_where.type = bind_symbolic_name G, 0 [symbolic = %G.loc9_15.1 (constants.%G)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @RepeatedRewrite(%H.loc11_20.2: %E_where.type) {
// CHECK:STDOUT:   %H.loc11_20.1: %E_where.type = bind_symbolic_name H, 0 [symbolic = %H.loc11_20.1 (constants.%H)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OneRewrite(constants.%G) {
// CHECK:STDOUT:   %G.loc9_15.1 => constants.%G
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @RepeatedRewrite(constants.%H) {
// CHECK:STDOUT:   %H.loc11_20.1 => constants.%H
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OneRewrite(constants.%H) {
// CHECK:STDOUT:   %G.loc9_15.1 => constants.%H
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @RepeatedRewrite(constants.%I) {
// CHECK:STDOUT:   %H.loc11_20.1 => constants.%I
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %OneRewrite.specific_fn.loc13_3.2 => constants.%OneRewrite.specific_fn.7fa5d5.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OneRewrite(constants.%I) {
// CHECK:STDOUT:   %G.loc9_15.1 => constants.%I
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- rewrites_reordered.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type @J [concrete]
// CHECK:STDOUT:   %assoc0: %J.assoc_type = assoc_entity element0, @J.%K [concrete]
// CHECK:STDOUT:   %assoc1: %J.assoc_type = assoc_entity element1, @J.%L [concrete]
// CHECK:STDOUT:   %.Self: %J.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %J.lookup_impl_witness: <witness> = lookup_impl_witness %.Self, @J [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %J.lookup_impl_witness, element0 [symbolic_self]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %impl.elem1: type = impl_witness_access %J.lookup_impl_witness, element1 [symbolic_self]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %J_where.type: type = facet_type <@J where %impl.elem0 = %empty_tuple.type and %impl.elem1 = bool> [concrete]
// CHECK:STDOUT:   %M: %J_where.type = bind_symbolic_name M, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.49f: type = pattern_type %J_where.type [concrete]
// CHECK:STDOUT:   %Alphabetical.type: type = fn_type @Alphabetical [concrete]
// CHECK:STDOUT:   %Alphabetical: %Alphabetical.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: %J_where.type = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %Reversed.type: type = fn_type @Reversed [concrete]
// CHECK:STDOUT:   %Reversed: %Reversed.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Alphabetical.decl: %Alphabetical.type = fn_decl @Alphabetical [concrete = constants.%Alphabetical] {
// CHECK:STDOUT:     %M.patt: %pattern_type.49f = symbolic_binding_pattern M, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc10_23.1: type = splice_block %.loc10_23.2 [concrete = constants.%J_where.type] {
// CHECK:STDOUT:       %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %.Self.ref.loc10_29: %J.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %K.ref: %J.assoc_type = name_ref K, @K.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:       %.Self.as_type.loc10_29: type = facet_access_type %.Self.ref.loc10_29 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc10_29: type = converted %.Self.ref.loc10_29, %.Self.as_type.loc10_29 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %impl.elem0: type = impl_witness_access constants.%J.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:       %.loc10_35.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc10_35.2: type = converted %.loc10_35.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %.Self.ref.loc10_41: %J.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %L.ref: %J.assoc_type = name_ref L, @L.%assoc1 [concrete = constants.%assoc1]
// CHECK:STDOUT:       %.Self.as_type.loc10_41: type = facet_access_type %.Self.ref.loc10_41 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc10_41: type = converted %.Self.ref.loc10_41, %.Self.as_type.loc10_41 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %impl.elem1: type = impl_witness_access constants.%J.lookup_impl_witness, element1 [symbolic_self = constants.%impl.elem1]
// CHECK:STDOUT:       %Bool.call: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %.loc10_46.1: type = value_of_initializer %Bool.call [concrete = bool]
// CHECK:STDOUT:       %.loc10_46.2: type = converted %Bool.call, %.loc10_46.1 [concrete = bool]
// CHECK:STDOUT:       %.loc10_23.2: type = where_expr %.Self [concrete = constants.%J_where.type] {
// CHECK:STDOUT:         requirement_rewrite %impl.elem0, %.loc10_35.2
// CHECK:STDOUT:         requirement_rewrite %impl.elem1, %.loc10_46.2
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %M.loc10_17.2: %J_where.type = bind_symbolic_name M, 0 [symbolic = %M.loc10_17.1 (constants.%M)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Reversed.decl: %Reversed.type = fn_decl @Reversed [concrete = constants.%Reversed] {
// CHECK:STDOUT:     %N.patt: %pattern_type.49f = symbolic_binding_pattern N, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc12_19.1: type = splice_block %.loc12_19.2 [concrete = constants.%J_where.type] {
// CHECK:STDOUT:       %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %.Self.ref.loc12_25: %J.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %L.ref: %J.assoc_type = name_ref L, @L.%assoc1 [concrete = constants.%assoc1]
// CHECK:STDOUT:       %.Self.as_type.loc12_25: type = facet_access_type %.Self.ref.loc12_25 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc12_25: type = converted %.Self.ref.loc12_25, %.Self.as_type.loc12_25 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %impl.elem1: type = impl_witness_access constants.%J.lookup_impl_witness, element1 [symbolic_self = constants.%impl.elem1]
// CHECK:STDOUT:       %Bool.call: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %.loc12_30.1: type = value_of_initializer %Bool.call [concrete = bool]
// CHECK:STDOUT:       %.loc12_30.2: type = converted %Bool.call, %.loc12_30.1 [concrete = bool]
// CHECK:STDOUT:       %.Self.ref.loc12_39: %J.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %K.ref: %J.assoc_type = name_ref K, @K.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:       %.Self.as_type.loc12_39: type = facet_access_type %.Self.ref.loc12_39 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc12_39: type = converted %.Self.ref.loc12_39, %.Self.as_type.loc12_39 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %impl.elem0: type = impl_witness_access constants.%J.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:       %.loc12_45.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc12_45.2: type = converted %.loc12_45.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %.loc12_19.2: type = where_expr %.Self [concrete = constants.%J_where.type] {
// CHECK:STDOUT:         requirement_rewrite %impl.elem1, %.loc12_30.2
// CHECK:STDOUT:         requirement_rewrite %impl.elem0, %.loc12_45.2
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc12_13.2: %J_where.type = bind_symbolic_name N, 0 [symbolic = %N.loc12_13.1 (constants.%N)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Alphabetical(%M.loc10_17.2: %J_where.type) {
// CHECK:STDOUT:   %M.loc10_17.1: %J_where.type = bind_symbolic_name M, 0 [symbolic = %M.loc10_17.1 (constants.%M)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Reversed(%N.loc12_13.2: %J_where.type) {
// CHECK:STDOUT:   %N.loc12_13.1: %J_where.type = bind_symbolic_name N, 0 [symbolic = %N.loc12_13.1 (constants.%N)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Alphabetical(constants.%M) {
// CHECK:STDOUT:   %M.loc10_17.1 => constants.%M
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Reversed(constants.%N) {
// CHECK:STDOUT:   %N.loc12_13.1 => constants.%N
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Alphabetical(constants.%N) {
// CHECK:STDOUT:   %M.loc10_17.1 => constants.%N
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:

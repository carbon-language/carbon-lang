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

// --- fail_todo_equal_constraint.carbon

library "[[@TEST_NAME]]";

interface N {
  let P:! type;
}

// CHECK:STDERR: fail_todo_equal_constraint.carbon:[[@LINE+7]]:27: error: cannot implicitly convert from `{}` to `<associated type in N>`
// CHECK:STDERR: fn Equal(T:! N where .P = {});
// CHECK:STDERR:                           ^~
// CHECK:STDERR: fail_todo_equal_constraint.carbon:[[@LINE+4]]:27: note: type `{}` does not implement interface `ImplicitAs`
// CHECK:STDERR: fn Equal(T:! N where .P = {});
// CHECK:STDERR:                           ^~
// CHECK:STDERR:
fn Equal(T:! N where .P = {});

// --- fail_todo_associated_type_impls.carbon

library "[[@TEST_NAME]]";

interface L {}
interface M {}

interface K {
  let Associated:! L;
}

// CHECK:STDERR: fail_todo_associated_type_impls.carbon:[[@LINE+7]]:36: error: cannot implicitly convert from `<associated L in K>` to `type`
// CHECK:STDERR: fn AssociatedTypeImpls(W:! K where .Associated impls M);
// CHECK:STDERR:                                    ^~~~~~~~~~~
// CHECK:STDERR: fail_todo_associated_type_impls.carbon:[[@LINE+4]]:36: note: type `<associated L in K>` does not implement interface `ImplicitAs`
// CHECK:STDERR: fn AssociatedTypeImpls(W:! K where .Associated impls M);
// CHECK:STDERR:                                    ^~~~~~~~~~~
// CHECK:STDERR:
fn AssociatedTypeImpls(W:! K where .Associated impls M);

// --- fail_todo_check_rewrite_constraints.carbon

library "[[@TEST_NAME]]";

import library "state_constraints";

// TODO: Should fail since `2` can't be converted to the type of `I.Member`,
// but constraints are not yet type checked.
// CHECK:STDERR: fail_todo_check_rewrite_constraints.carbon:[[@LINE+7]]:46: error: cannot implicitly convert from `i32` to `<associated type in I>`
// CHECK:STDERR: fn RewriteTypeMismatch(X:! I where .Member = 2);
// CHECK:STDERR:                                              ^
// CHECK:STDERR: fail_todo_check_rewrite_constraints.carbon:[[@LINE+4]]:46: note: type `i32` does not implement interface `ImplicitAs`
// CHECK:STDERR: fn RewriteTypeMismatch(X:! I where .Member = 2);
// CHECK:STDERR:                                              ^
// CHECK:STDERR:
fn RewriteTypeMismatch(X:! I where .Member = 2);

// --- fail_left_of_impls_non_type.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_left_of_impls_non_type.carbon:[[@LINE+7]]:32: error: cannot implicitly convert from `i32` to `type`
// CHECK:STDERR: fn NonTypeImpls(U:! type where 7 impls type);
// CHECK:STDERR:                                ^
// CHECK:STDERR: fail_left_of_impls_non_type.carbon:[[@LINE+4]]:32: note: type `i32` does not implement interface `ImplicitAs`
// CHECK:STDERR: fn NonTypeImpls(U:! type where 7 impls type);
// CHECK:STDERR:                                ^
// CHECK:STDERR:
fn NonTypeImpls(U:! type where 7 impls type);

// --- fail_right_of_impls_non_type.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_right_of_impls_non_type.carbon:[[@LINE+7]]:44: error: cannot implicitly convert from `i32` to `type`
// CHECK:STDERR: fn ImplsNonType(U:! type where .Self impls 7);
// CHECK:STDERR:                                            ^
// CHECK:STDERR: fail_right_of_impls_non_type.carbon:[[@LINE+4]]:44: note: type `i32` does not implement interface `ImplicitAs`
// CHECK:STDERR: fn ImplsNonType(U:! type where .Self impls 7);
// CHECK:STDERR:                                            ^
// CHECK:STDERR:
fn ImplsNonType(U:! type where .Self impls 7);

// --- fail_right_of_impls_non_facet_type.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_right_of_impls_non_facet_type.carbon:[[@LINE+4]]:51: error: right argument of `impls` requirement must be a facet type
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
  // CHECK:STDERR: fail_todo_enforce_constraint.carbon:[[@LINE+11]]:3: error: cannot implicitly convert from `type` to `J`
  // CHECK:STDERR:   Impls(C);
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR: fail_todo_enforce_constraint.carbon:[[@LINE+8]]:3: note: type `type` does not implement interface `ImplicitAs`
  // CHECK:STDERR:   Impls(C);
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR: fail_todo_enforce_constraint.carbon:[[@LINE-14]]:1: in import
  // CHECK:STDERR: state_constraints.carbon:13:10: note: initializing generic parameter `V` declared here
  // CHECK:STDERR: fn Impls(V:! J where .Self impls I);
  // CHECK:STDERR:          ^
  // CHECK:STDERR:
  Impls(C);
}

fn EmptyStruct(Y:! J where .Self == {});

// TODO: Should report that `C` does not meeet the constraint.
fn NotEmptyStruct() {
  // CHECK:STDERR: fail_todo_enforce_constraint.carbon:[[@LINE+9]]:3: error: cannot implicitly convert from `type` to `J`
  // CHECK:STDERR:   EmptyStruct(C);
  // CHECK:STDERR:   ^~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_enforce_constraint.carbon:[[@LINE+6]]:3: note: type `type` does not implement interface `ImplicitAs`
  // CHECK:STDERR:   EmptyStruct(C);
  // CHECK:STDERR:   ^~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_enforce_constraint.carbon:[[@LINE-10]]:16: note: initializing generic parameter `Y` declared here
  // CHECK:STDERR: fn EmptyStruct(Y:! J where .Self == {});
  // CHECK:STDERR:                ^
  EmptyStruct(C);
}

// CHECK:STDOUT: --- state_constraints.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @J [template]
// CHECK:STDOUT:   %Self.1: %.1 = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %.2: type = interface_type @I [template]
// CHECK:STDOUT:   %Self.2: %.2 = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %.3: type = assoc_entity_type %.2, type [template]
// CHECK:STDOUT:   %.4: %.3 = assoc_entity element0, @I.%Member [template]
// CHECK:STDOUT:   %.5: type = tuple_type () [template]
// CHECK:STDOUT:   %.6: type = assoc_entity_type %.2, %.1 [template]
// CHECK:STDOUT:   %.7: %.6 = assoc_entity element1, @I.%Second [template]
// CHECK:STDOUT:   %.Self.1: %.2 = bind_symbolic_name .Self, 0 [symbolic]
// CHECK:STDOUT:   %U: %.2 = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %EqualEqual.type: type = fn_type @EqualEqual [template]
// CHECK:STDOUT:   %EqualEqual: %EqualEqual.type = struct_value () [template]
// CHECK:STDOUT:   %.Self.2: %.1 = bind_symbolic_name .Self, 0 [symbolic]
// CHECK:STDOUT:   %V: %.1 = bind_symbolic_name V, 0 [symbolic]
// CHECK:STDOUT:   %Impls.type: type = fn_type @Impls [template]
// CHECK:STDOUT:   %Impls: %Impls.type = struct_value () [template]
// CHECK:STDOUT:   %W: %.2 = bind_symbolic_name W, 0 [symbolic]
// CHECK:STDOUT:   %And.type: type = fn_type @And [template]
// CHECK:STDOUT:   %And: %And.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
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
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [template = constants.%.1] {} {}
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [template = constants.%.2] {} {}
// CHECK:STDOUT:   %EqualEqual.decl: %EqualEqual.type = fn_decl @EqualEqual [template = constants.%EqualEqual] {
// CHECK:STDOUT:     %U.patt: %.2 = symbolic_binding_pattern U, 0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [template = constants.%.2]
// CHECK:STDOUT:     %.Self: %.2 = bind_symbolic_name .Self, 0 [symbolic = constants.%.Self.1]
// CHECK:STDOUT:     %.Self.ref: %.2 = name_ref .Self, %.Self [symbolic = constants.%.Self.1]
// CHECK:STDOUT:     %.loc11_37: %.5 = tuple_literal ()
// CHECK:STDOUT:     %.loc11_21: type = where_expr %.Self [template = constants.%.2] {
// CHECK:STDOUT:       requirement_equivalent %.Self.ref, %.loc11_37
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.param: %.2 = param U, runtime_param<invalid>
// CHECK:STDOUT:     %U.loc11: %.2 = bind_symbolic_name U, 0, %U.param [symbolic = %U.1 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Impls.decl: %Impls.type = fn_decl @Impls [template = constants.%Impls] {
// CHECK:STDOUT:     %V.patt: %.1 = symbolic_binding_pattern V, 0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [template = constants.%.1]
// CHECK:STDOUT:     %.Self: %.1 = bind_symbolic_name .Self, 0 [symbolic = constants.%.Self.2]
// CHECK:STDOUT:     %.Self.ref: %.1 = name_ref .Self, %.Self [symbolic = constants.%.Self.2]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [template = constants.%.2]
// CHECK:STDOUT:     %.loc13_22.1: type = facet_type_access %.Self.ref [symbolic = constants.%.Self.2]
// CHECK:STDOUT:     %.loc13_22.2: type = converted %.Self.ref, %.loc13_22.1 [symbolic = constants.%.Self.2]
// CHECK:STDOUT:     %.loc13_16: type = where_expr %.Self [template = constants.%.1] {
// CHECK:STDOUT:       requirement_impls %.loc13_22.2, %I.ref
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %V.param: %.1 = param V, runtime_param<invalid>
// CHECK:STDOUT:     %V.loc13: %.1 = bind_symbolic_name V, 0, %V.param [symbolic = %V.1 (constants.%V)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %And.decl: %And.type = fn_decl @And [template = constants.%And] {
// CHECK:STDOUT:     %W.patt: %.2 = symbolic_binding_pattern W, 0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [template = constants.%.2]
// CHECK:STDOUT:     %.Self: %.2 = bind_symbolic_name .Self, 0 [symbolic = constants.%.Self.1]
// CHECK:STDOUT:     %.Self.ref.loc15_20: %.2 = name_ref .Self, %.Self [symbolic = constants.%.Self.1]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [template = constants.%.1]
// CHECK:STDOUT:     %.loc15_20.1: type = facet_type_access %.Self.ref.loc15_20 [symbolic = constants.%.Self.1]
// CHECK:STDOUT:     %.loc15_20.2: type = converted %.Self.ref.loc15_20, %.loc15_20.1 [symbolic = constants.%.Self.1]
// CHECK:STDOUT:     %.Self.ref.loc15_38: %.2 = name_ref .Self, %.Self [symbolic = constants.%.Self.1]
// CHECK:STDOUT:     %Member.ref: %.3 = name_ref Member, @I.%.loc7 [template = constants.%.4]
// CHECK:STDOUT:     %.loc15_50: %.5 = tuple_literal ()
// CHECK:STDOUT:     %.loc15_14: type = where_expr %.Self [template = constants.%.2] {
// CHECK:STDOUT:       requirement_impls %.loc15_20.2, %J.ref
// CHECK:STDOUT:       requirement_equivalent %Member.ref, %.loc15_50
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %W.param: %.2 = param W, runtime_param<invalid>
// CHECK:STDOUT:     %W.loc15: %.2 = bind_symbolic_name W, 0, %W.param [symbolic = %W.1 (constants.%W)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %.1 = bind_symbolic_name Self, 0 [symbolic = constants.%Self.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %.2 = bind_symbolic_name Self, 0 [symbolic = constants.%Self.2]
// CHECK:STDOUT:   %Member: type = assoc_const_decl Member [template]
// CHECK:STDOUT:   %.loc7: %.3 = assoc_entity element0, %Member [template = constants.%.4]
// CHECK:STDOUT:   %J.ref: type = name_ref J, file.%J.decl [template = constants.%.1]
// CHECK:STDOUT:   %Second: %.1 = assoc_const_decl Second [template]
// CHECK:STDOUT:   %.loc8: %.6 = assoc_entity element1, %Second [template = constants.%.7]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Member = %.loc7
// CHECK:STDOUT:   .Second = %.loc8
// CHECK:STDOUT:   witness = (%Member, %Second)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @EqualEqual(%U.loc11: %.2) {
// CHECK:STDOUT:   %U.1: %.2 = bind_symbolic_name U, 0 [symbolic = %U.1 (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%U.loc11: %.2);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Impls(%V.loc13: %.1) {
// CHECK:STDOUT:   %V.1: %.1 = bind_symbolic_name V, 0 [symbolic = %V.1 (constants.%V)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%V.loc13: %.1);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @And(%W.loc15: %.2) {
// CHECK:STDOUT:   %W.1: %.2 = bind_symbolic_name W, 0 [symbolic = %W.1 (constants.%W)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%W.loc15: %.2);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @EqualEqual(constants.%U) {
// CHECK:STDOUT:   %U.1 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Impls(constants.%V) {
// CHECK:STDOUT:   %V.1 => constants.%V
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @And(constants.%W) {
// CHECK:STDOUT:   %W.1 => constants.%W
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_equal_constraint.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @N [template]
// CHECK:STDOUT:   %Self.1: %.1 = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %.2: type = assoc_entity_type %.1, type [template]
// CHECK:STDOUT:   %.3: %.2 = assoc_entity element0, @N.%P [template]
// CHECK:STDOUT:   %.Self: %.1 = bind_symbolic_name .Self, 0 [symbolic]
// CHECK:STDOUT:   %.4: type = tuple_type () [template]
// CHECK:STDOUT:   %.5: type = struct_type {} [template]
// CHECK:STDOUT:   %ImplicitAs.type: type = generic_interface_type @ImplicitAs [template]
// CHECK:STDOUT:   %ImplicitAs: %ImplicitAs.type = struct_value () [template]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %.6: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Self.2: @ImplicitAs.%.1 (%.6) = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.3: %.6 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.1: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.1: %Convert.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.7: type = assoc_entity_type %.6, %Convert.type.1 [symbolic]
// CHECK:STDOUT:   %.8: %.7 = assoc_entity element0, imports.%import_ref.5 [symbolic]
// CHECK:STDOUT:   %.9: type = interface_type @ImplicitAs, @ImplicitAs(%.2) [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert, @ImplicitAs(%.2) [template]
// CHECK:STDOUT:   %Convert.2: %Convert.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.10: type = assoc_entity_type %.9, %Convert.type.2 [template]
// CHECK:STDOUT:   %.11: %.10 = assoc_entity element0, imports.%import_ref.5 [template]
// CHECK:STDOUT:   %.12: %.7 = assoc_entity element0, imports.%import_ref.6 [symbolic]
// CHECK:STDOUT:   %struct: %.5 = struct_value () [template]
// CHECK:STDOUT:   %T: %.1 = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Equal.type: type = fn_type @Equal [template]
// CHECK:STDOUT:   %Equal: %Equal.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .ImplicitAs = %import_ref.1
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %ImplicitAs.type = import_ref Core//prelude/operators/as, inst+40, loaded [template = constants.%ImplicitAs]
// CHECK:STDOUT:   %import_ref.2 = import_ref Core//prelude/operators/as, inst+45, unloaded
// CHECK:STDOUT:   %import_ref.3: @ImplicitAs.%.2 (%.7) = import_ref Core//prelude/operators/as, inst+63, loaded [symbolic = @ImplicitAs.%.3 (constants.%.12)]
// CHECK:STDOUT:   %import_ref.4 = import_ref Core//prelude/operators/as, inst+56, unloaded
// CHECK:STDOUT:   %import_ref.5 = import_ref Core//prelude/operators/as, inst+56, unloaded
// CHECK:STDOUT:   %import_ref.6 = import_ref Core//prelude/operators/as, inst+56, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .N = %N.decl
// CHECK:STDOUT:     .Equal = %Equal.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %N.decl: type = interface_decl @N [template = constants.%.1] {} {}
// CHECK:STDOUT:   %Equal.decl: %Equal.type = fn_decl @Equal [template = constants.%Equal] {
// CHECK:STDOUT:     %T.patt: %.1 = symbolic_binding_pattern T, 0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %N.ref: type = name_ref N, file.%N.decl [template = constants.%.1]
// CHECK:STDOUT:     %.Self: %.1 = bind_symbolic_name .Self, 0 [symbolic = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %.1 = name_ref .Self, %.Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %P.ref: %.2 = name_ref P, @N.%.loc5 [template = constants.%.3]
// CHECK:STDOUT:     %.loc15_28.1: %.5 = struct_literal ()
// CHECK:STDOUT:     %.loc15_28.2: type = interface_type @ImplicitAs, @ImplicitAs(constants.%.2) [template = constants.%.9]
// CHECK:STDOUT:     %.loc15_28.3: %.10 = specific_constant imports.%import_ref.3, @ImplicitAs(constants.%.2) [template = constants.%.11]
// CHECK:STDOUT:     %Convert.ref: %.10 = name_ref Convert, %.loc15_28.3 [template = constants.%.11]
// CHECK:STDOUT:     %struct: %.5 = struct_value () [template = constants.%struct]
// CHECK:STDOUT:     %.loc15_28.4: %.5 = converted %.loc15_28.1, %struct [template = constants.%struct]
// CHECK:STDOUT:     %.loc15_28.5: %.2 = converted %.loc15_28.1, <error> [template = <error>]
// CHECK:STDOUT:     %.loc15_16: type = where_expr %.Self [template = constants.%.1] {
// CHECK:STDOUT:       requirement_rewrite %P.ref, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.param: %.1 = param T, runtime_param<invalid>
// CHECK:STDOUT:     %T.loc15: %.1 = bind_symbolic_name T, 0, %T.param [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @N {
// CHECK:STDOUT:   %Self: %.1 = bind_symbolic_name Self, 0 [symbolic = constants.%Self.1]
// CHECK:STDOUT:   %P: type = assoc_const_decl P [template]
// CHECK:STDOUT:   %.loc5: %.2 = assoc_entity element0, %P [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .P = %.loc5
// CHECK:STDOUT:   witness = (%P)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(constants.%Dest: type) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic = %.1 (constants.%.6)]
// CHECK:STDOUT:   %Self: %.6 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.3)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic = %Convert.type (constants.%Convert.type.1)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.1) = struct_value () [symbolic = %Convert (constants.%Convert.1)]
// CHECK:STDOUT:   %.2: type = assoc_entity_type @ImplicitAs.%.1 (%.6), @ImplicitAs.%Convert.type (%Convert.type.1) [symbolic = %.2 (constants.%.7)]
// CHECK:STDOUT:   %.3: @ImplicitAs.%.2 (%.7) = assoc_entity element0, imports.%import_ref.5 [symbolic = %.3 (constants.%.8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.2
// CHECK:STDOUT:     .Convert = imports.%import_ref.3
// CHECK:STDOUT:     witness = (imports.%import_ref.4)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert(constants.%Dest: type, constants.%Self.2: @ImplicitAs.%.1 (%.6)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %.1: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic = %.1 (constants.%.6)]
// CHECK:STDOUT:   %Self: %.6 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self: @Convert.%Self (%Self.3)]() -> @Convert.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Equal(%T.loc15: %.1) {
// CHECK:STDOUT:   %T.1: %.1 = bind_symbolic_name T, 0 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.loc15: %.1);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@ImplicitAs.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert(constants.%Dest, constants.%Self.2) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %.1 => constants.%.6
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%.2) {
// CHECK:STDOUT:   %Dest => constants.%.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1 => constants.%.9
// CHECK:STDOUT:   %Self => constants.%Self.3
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.2
// CHECK:STDOUT:   %Convert => constants.%Convert.2
// CHECK:STDOUT:   %.2 => constants.%.10
// CHECK:STDOUT:   %.3 => constants.%.11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Equal(constants.%T) {
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_associated_type_impls.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @L [template]
// CHECK:STDOUT:   %Self.1: %.1 = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %.2: type = interface_type @M [template]
// CHECK:STDOUT:   %Self.2: %.2 = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %.3: type = interface_type @K [template]
// CHECK:STDOUT:   %Self.3: %.3 = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %.4: type = tuple_type () [template]
// CHECK:STDOUT:   %.5: type = assoc_entity_type %.3, %.1 [template]
// CHECK:STDOUT:   %.6: %.5 = assoc_entity element0, @K.%Associated [template]
// CHECK:STDOUT:   %.Self: %.3 = bind_symbolic_name .Self, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type: type = generic_interface_type @ImplicitAs [template]
// CHECK:STDOUT:   %ImplicitAs: %ImplicitAs.type = struct_value () [template]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %.7: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Self.4: @ImplicitAs.%.1 (%.7) = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.5: %.7 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.1: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.1: %Convert.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.8: type = assoc_entity_type %.7, %Convert.type.1 [symbolic]
// CHECK:STDOUT:   %.9: %.8 = assoc_entity element0, imports.%import_ref.5 [symbolic]
// CHECK:STDOUT:   %.10: type = interface_type @ImplicitAs, @ImplicitAs(type) [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert, @ImplicitAs(type) [template]
// CHECK:STDOUT:   %Convert.2: %Convert.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.11: type = assoc_entity_type %.10, %Convert.type.2 [template]
// CHECK:STDOUT:   %.12: %.11 = assoc_entity element0, imports.%import_ref.5 [template]
// CHECK:STDOUT:   %.13: %.8 = assoc_entity element0, imports.%import_ref.6 [symbolic]
// CHECK:STDOUT:   %W: %.3 = bind_symbolic_name W, 0 [symbolic]
// CHECK:STDOUT:   %AssociatedTypeImpls.type: type = fn_type @AssociatedTypeImpls [template]
// CHECK:STDOUT:   %AssociatedTypeImpls: %AssociatedTypeImpls.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .ImplicitAs = %import_ref.1
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %ImplicitAs.type = import_ref Core//prelude/operators/as, inst+40, loaded [template = constants.%ImplicitAs]
// CHECK:STDOUT:   %import_ref.2 = import_ref Core//prelude/operators/as, inst+45, unloaded
// CHECK:STDOUT:   %import_ref.3: @ImplicitAs.%.2 (%.8) = import_ref Core//prelude/operators/as, inst+63, loaded [symbolic = @ImplicitAs.%.3 (constants.%.13)]
// CHECK:STDOUT:   %import_ref.4 = import_ref Core//prelude/operators/as, inst+56, unloaded
// CHECK:STDOUT:   %import_ref.5 = import_ref Core//prelude/operators/as, inst+56, unloaded
// CHECK:STDOUT:   %import_ref.6 = import_ref Core//prelude/operators/as, inst+56, unloaded
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
// CHECK:STDOUT:   %L.decl: type = interface_decl @L [template = constants.%.1] {} {}
// CHECK:STDOUT:   %M.decl: type = interface_decl @M [template = constants.%.2] {} {}
// CHECK:STDOUT:   %K.decl: type = interface_decl @K [template = constants.%.3] {} {}
// CHECK:STDOUT:   %AssociatedTypeImpls.decl: %AssociatedTypeImpls.type = fn_decl @AssociatedTypeImpls [template = constants.%AssociatedTypeImpls] {
// CHECK:STDOUT:     %W.patt: %.3 = symbolic_binding_pattern W, 0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %K.ref: type = name_ref K, file.%K.decl [template = constants.%.3]
// CHECK:STDOUT:     %.Self: %.3 = bind_symbolic_name .Self, 0 [symbolic = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %.3 = name_ref .Self, %.Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %Associated.ref: %.5 = name_ref Associated, @K.%.loc8 [template = constants.%.6]
// CHECK:STDOUT:     %M.ref: type = name_ref M, file.%M.decl [template = constants.%.2]
// CHECK:STDOUT:     %.loc18_36.1: type = interface_type @ImplicitAs, @ImplicitAs(type) [template = constants.%.10]
// CHECK:STDOUT:     %.loc18_36.2: %.11 = specific_constant imports.%import_ref.3, @ImplicitAs(type) [template = constants.%.12]
// CHECK:STDOUT:     %Convert.ref: %.11 = name_ref Convert, %.loc18_36.2 [template = constants.%.12]
// CHECK:STDOUT:     %.loc18_36.3: type = converted %Associated.ref, <error> [template = <error>]
// CHECK:STDOUT:     %.loc18_30: type = where_expr %.Self [template = constants.%.3] {
// CHECK:STDOUT:       requirement_impls <error>, %M.ref
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %W.param: %.3 = param W, runtime_param<invalid>
// CHECK:STDOUT:     %W.loc18: %.3 = bind_symbolic_name W, 0, %W.param [symbolic = %W.1 (constants.%W)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @L {
// CHECK:STDOUT:   %Self: %.1 = bind_symbolic_name Self, 0 [symbolic = constants.%Self.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @M {
// CHECK:STDOUT:   %Self: %.2 = bind_symbolic_name Self, 0 [symbolic = constants.%Self.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @K {
// CHECK:STDOUT:   %Self: %.3 = bind_symbolic_name Self, 0 [symbolic = constants.%Self.3]
// CHECK:STDOUT:   %L.ref: type = name_ref L, file.%L.decl [template = constants.%.1]
// CHECK:STDOUT:   %Associated: %.1 = assoc_const_decl Associated [template]
// CHECK:STDOUT:   %.loc8: %.5 = assoc_entity element0, %Associated [template = constants.%.6]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Associated = %.loc8
// CHECK:STDOUT:   witness = (%Associated)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(constants.%Dest: type) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic = %.1 (constants.%.7)]
// CHECK:STDOUT:   %Self: %.7 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.5)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic = %Convert.type (constants.%Convert.type.1)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.1) = struct_value () [symbolic = %Convert (constants.%Convert.1)]
// CHECK:STDOUT:   %.2: type = assoc_entity_type @ImplicitAs.%.1 (%.7), @ImplicitAs.%Convert.type (%Convert.type.1) [symbolic = %.2 (constants.%.8)]
// CHECK:STDOUT:   %.3: @ImplicitAs.%.2 (%.8) = assoc_entity element0, imports.%import_ref.5 [symbolic = %.3 (constants.%.9)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.2
// CHECK:STDOUT:     .Convert = imports.%import_ref.3
// CHECK:STDOUT:     witness = (imports.%import_ref.4)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert(constants.%Dest: type, constants.%Self.4: @ImplicitAs.%.1 (%.7)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %.1: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic = %.1 (constants.%.7)]
// CHECK:STDOUT:   %Self: %.7 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self: @Convert.%Self (%Self.5)]() -> @Convert.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AssociatedTypeImpls(%W.loc18: %.3) {
// CHECK:STDOUT:   %W.1: %.3 = bind_symbolic_name W, 0 [symbolic = %W.1 (constants.%W)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%W.loc18: %.3);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@ImplicitAs.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert(constants.%Dest, constants.%Self.4) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %.1 => constants.%.7
// CHECK:STDOUT:   %Self => constants.%Self.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(type) {
// CHECK:STDOUT:   %Dest => type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1 => constants.%.10
// CHECK:STDOUT:   %Self => constants.%Self.5
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.2
// CHECK:STDOUT:   %Convert => constants.%Convert.2
// CHECK:STDOUT:   %.2 => constants.%.11
// CHECK:STDOUT:   %.3 => constants.%.12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AssociatedTypeImpls(constants.%W) {
// CHECK:STDOUT:   %W.1 => constants.%W
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_check_rewrite_constraints.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @I [template]
// CHECK:STDOUT:   %Self.1: %.1 = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %.Self: %.1 = bind_symbolic_name .Self, 0 [symbolic]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %.3: type = assoc_entity_type %.1, type [template]
// CHECK:STDOUT:   %.4: %.3 = assoc_entity element0, imports.%import_ref.11 [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %ImplicitAs.type: type = generic_interface_type @ImplicitAs [template]
// CHECK:STDOUT:   %ImplicitAs: %ImplicitAs.type = struct_value () [template]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %.6: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Self.2: @ImplicitAs.%.1 (%.6) = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.3: %.6 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.1: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.1: %Convert.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.7: type = assoc_entity_type %.6, %Convert.type.1 [symbolic]
// CHECK:STDOUT:   %.8: %.7 = assoc_entity element0, imports.%import_ref.16 [symbolic]
// CHECK:STDOUT:   %.9: type = interface_type @ImplicitAs, @ImplicitAs(%.3) [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert, @ImplicitAs(%.3) [template]
// CHECK:STDOUT:   %Convert.2: %Convert.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.10: type = assoc_entity_type %.9, %Convert.type.2 [template]
// CHECK:STDOUT:   %.11: %.10 = assoc_entity element0, imports.%import_ref.16 [template]
// CHECK:STDOUT:   %.12: %.7 = assoc_entity element0, imports.%import_ref.17 [symbolic]
// CHECK:STDOUT:   %X: %.1 = bind_symbolic_name X, 0 [symbolic]
// CHECK:STDOUT:   %RewriteTypeMismatch.type: type = fn_type @RewriteTypeMismatch [template]
// CHECK:STDOUT:   %RewriteTypeMismatch: %RewriteTypeMismatch.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1 = import_ref Main//state_constraints, inst+3, unloaded
// CHECK:STDOUT:   %import_ref.2: type = import_ref Main//state_constraints, inst+7, loaded [template = constants.%.1]
// CHECK:STDOUT:   %import_ref.3 = import_ref Main//state_constraints, inst+32, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref Main//state_constraints, inst+49, unloaded
// CHECK:STDOUT:   %import_ref.5 = import_ref Main//state_constraints, inst+69, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .ImplicitAs = %import_ref.12
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.6 = import_ref Main//state_constraints, inst+9, unloaded
// CHECK:STDOUT:   %import_ref.7: %.3 = import_ref Main//state_constraints, inst+13, loaded [template = constants.%.4]
// CHECK:STDOUT:   %import_ref.8 = import_ref Main//state_constraints, inst+19, unloaded
// CHECK:STDOUT:   %import_ref.9 = import_ref Main//state_constraints, inst+11, unloaded
// CHECK:STDOUT:   %import_ref.10 = import_ref Main//state_constraints, inst+17, unloaded
// CHECK:STDOUT:   %import_ref.11 = import_ref Main//state_constraints, inst+11, unloaded
// CHECK:STDOUT:   %import_ref.12: %ImplicitAs.type = import_ref Core//prelude/operators/as, inst+40, loaded [template = constants.%ImplicitAs]
// CHECK:STDOUT:   %import_ref.13 = import_ref Core//prelude/operators/as, inst+45, unloaded
// CHECK:STDOUT:   %import_ref.14: @ImplicitAs.%.2 (%.7) = import_ref Core//prelude/operators/as, inst+63, loaded [symbolic = @ImplicitAs.%.3 (constants.%.12)]
// CHECK:STDOUT:   %import_ref.15 = import_ref Core//prelude/operators/as, inst+56, unloaded
// CHECK:STDOUT:   %import_ref.16 = import_ref Core//prelude/operators/as, inst+56, unloaded
// CHECK:STDOUT:   %import_ref.17 = import_ref Core//prelude/operators/as, inst+56, unloaded
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
// CHECK:STDOUT:     .RewriteTypeMismatch = %RewriteTypeMismatch.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %RewriteTypeMismatch.decl: %RewriteTypeMismatch.type = fn_decl @RewriteTypeMismatch [template = constants.%RewriteTypeMismatch] {
// CHECK:STDOUT:     %X.patt: %.1 = symbolic_binding_pattern X, 0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %I.ref: type = name_ref I, imports.%import_ref.2 [template = constants.%.1]
// CHECK:STDOUT:     %.Self: %.1 = bind_symbolic_name .Self, 0 [symbolic = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %.1 = name_ref .Self, %.Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %Member.ref: %.3 = name_ref Member, imports.%import_ref.7 [template = constants.%.4]
// CHECK:STDOUT:     %.loc15_46.1: i32 = int_literal 2 [template = constants.%.5]
// CHECK:STDOUT:     %.loc15_46.2: type = interface_type @ImplicitAs, @ImplicitAs(constants.%.3) [template = constants.%.9]
// CHECK:STDOUT:     %.loc15_46.3: %.10 = specific_constant imports.%import_ref.14, @ImplicitAs(constants.%.3) [template = constants.%.11]
// CHECK:STDOUT:     %Convert.ref: %.10 = name_ref Convert, %.loc15_46.3 [template = constants.%.11]
// CHECK:STDOUT:     %.loc15_46.4: %.3 = converted %.loc15_46.1, <error> [template = <error>]
// CHECK:STDOUT:     %.loc15_30: type = where_expr %.Self [template = constants.%.1] {
// CHECK:STDOUT:       requirement_rewrite %Member.ref, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %X.param: %.1 = param X, runtime_param<invalid>
// CHECK:STDOUT:     %X.loc15: %.1 = bind_symbolic_name X, 0, %X.param [symbolic = %X.1 (constants.%X)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.6
// CHECK:STDOUT:   .Member = imports.%import_ref.7
// CHECK:STDOUT:   .Second = imports.%import_ref.8
// CHECK:STDOUT:   witness = (imports.%import_ref.9, imports.%import_ref.10)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(constants.%Dest: type) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic = %.1 (constants.%.6)]
// CHECK:STDOUT:   %Self: %.6 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.3)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic = %Convert.type (constants.%Convert.type.1)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.1) = struct_value () [symbolic = %Convert (constants.%Convert.1)]
// CHECK:STDOUT:   %.2: type = assoc_entity_type @ImplicitAs.%.1 (%.6), @ImplicitAs.%Convert.type (%Convert.type.1) [symbolic = %.2 (constants.%.7)]
// CHECK:STDOUT:   %.3: @ImplicitAs.%.2 (%.7) = assoc_entity element0, imports.%import_ref.16 [symbolic = %.3 (constants.%.8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.13
// CHECK:STDOUT:     .Convert = imports.%import_ref.14
// CHECK:STDOUT:     witness = (imports.%import_ref.15)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert(constants.%Dest: type, constants.%Self.2: @ImplicitAs.%.1 (%.6)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %.1: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic = %.1 (constants.%.6)]
// CHECK:STDOUT:   %Self: %.6 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self: @Convert.%Self (%Self.3)]() -> @Convert.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @RewriteTypeMismatch(%X.loc15: %.1) {
// CHECK:STDOUT:   %X.1: %.1 = bind_symbolic_name X, 0 [symbolic = %X.1 (constants.%X)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%X.loc15: %.1);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@ImplicitAs.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert(constants.%Dest, constants.%Self.2) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %.1 => constants.%.6
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%.3) {
// CHECK:STDOUT:   %Dest => constants.%.3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1 => constants.%.9
// CHECK:STDOUT:   %Self => constants.%Self.3
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.2
// CHECK:STDOUT:   %Convert => constants.%Convert.2
// CHECK:STDOUT:   %.2 => constants.%.10
// CHECK:STDOUT:   %.3 => constants.%.11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @RewriteTypeMismatch(constants.%X) {
// CHECK:STDOUT:   %X.1 => constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_left_of_impls_non_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.Self: type = bind_symbolic_name .Self, 0 [symbolic]
// CHECK:STDOUT:   %.1: i32 = int_literal 7 [template]
// CHECK:STDOUT:   %ImplicitAs.type: type = generic_interface_type @ImplicitAs [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %ImplicitAs: %ImplicitAs.type = struct_value () [template]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %.3: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Self.1: @ImplicitAs.%.1 (%.3) = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.2: %.3 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.1: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.1: %Convert.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.4: type = assoc_entity_type %.3, %Convert.type.1 [symbolic]
// CHECK:STDOUT:   %.5: %.4 = assoc_entity element0, imports.%import_ref.5 [symbolic]
// CHECK:STDOUT:   %.6: type = interface_type @ImplicitAs, @ImplicitAs(type) [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert, @ImplicitAs(type) [template]
// CHECK:STDOUT:   %Convert.2: %Convert.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.7: type = assoc_entity_type %.6, %Convert.type.2 [template]
// CHECK:STDOUT:   %.8: %.7 = assoc_entity element0, imports.%import_ref.5 [template]
// CHECK:STDOUT:   %.9: %.4 = assoc_entity element0, imports.%import_ref.6 [symbolic]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %NonTypeImpls.type: type = fn_type @NonTypeImpls [template]
// CHECK:STDOUT:   %NonTypeImpls: %NonTypeImpls.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .ImplicitAs = %import_ref.1
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %ImplicitAs.type = import_ref Core//prelude/operators/as, inst+40, loaded [template = constants.%ImplicitAs]
// CHECK:STDOUT:   %import_ref.2 = import_ref Core//prelude/operators/as, inst+45, unloaded
// CHECK:STDOUT:   %import_ref.3: @ImplicitAs.%.2 (%.4) = import_ref Core//prelude/operators/as, inst+63, loaded [symbolic = @ImplicitAs.%.3 (constants.%.9)]
// CHECK:STDOUT:   %import_ref.4 = import_ref Core//prelude/operators/as, inst+56, unloaded
// CHECK:STDOUT:   %import_ref.5 = import_ref Core//prelude/operators/as, inst+56, unloaded
// CHECK:STDOUT:   %import_ref.6 = import_ref Core//prelude/operators/as, inst+56, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .NonTypeImpls = %NonTypeImpls.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %NonTypeImpls.decl: %NonTypeImpls.type = fn_decl @NonTypeImpls [template = constants.%NonTypeImpls] {
// CHECK:STDOUT:     %U.patt: type = symbolic_binding_pattern U, 0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: type = bind_symbolic_name .Self, 0 [symbolic = constants.%.Self]
// CHECK:STDOUT:     %.loc11_32.1: i32 = int_literal 7 [template = constants.%.1]
// CHECK:STDOUT:     %.loc11_32.2: type = interface_type @ImplicitAs, @ImplicitAs(type) [template = constants.%.6]
// CHECK:STDOUT:     %.loc11_32.3: %.7 = specific_constant imports.%import_ref.3, @ImplicitAs(type) [template = constants.%.8]
// CHECK:STDOUT:     %Convert.ref: %.7 = name_ref Convert, %.loc11_32.3 [template = constants.%.8]
// CHECK:STDOUT:     %.loc11_32.4: type = converted %.loc11_32.1, <error> [template = <error>]
// CHECK:STDOUT:     %.loc11_26: type = where_expr %.Self [template = type] {
// CHECK:STDOUT:       requirement_impls <error>, type
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.param: type = param U, runtime_param<invalid>
// CHECK:STDOUT:     %U.loc11: type = bind_symbolic_name U, 0, %U.param [symbolic = %U.1 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(constants.%Dest: type) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic = %.1 (constants.%.3)]
// CHECK:STDOUT:   %Self: %.3 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic = %Convert.type (constants.%Convert.type.1)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.1) = struct_value () [symbolic = %Convert (constants.%Convert.1)]
// CHECK:STDOUT:   %.2: type = assoc_entity_type @ImplicitAs.%.1 (%.3), @ImplicitAs.%Convert.type (%Convert.type.1) [symbolic = %.2 (constants.%.4)]
// CHECK:STDOUT:   %.3: @ImplicitAs.%.2 (%.4) = assoc_entity element0, imports.%import_ref.5 [symbolic = %.3 (constants.%.5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.2
// CHECK:STDOUT:     .Convert = imports.%import_ref.3
// CHECK:STDOUT:     witness = (imports.%import_ref.4)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert(constants.%Dest: type, constants.%Self.1: @ImplicitAs.%.1 (%.3)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %.1: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic = %.1 (constants.%.3)]
// CHECK:STDOUT:   %Self: %.3 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self: @Convert.%Self (%Self.2)]() -> @Convert.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @NonTypeImpls(%U.loc11: type) {
// CHECK:STDOUT:   %U.1: type = bind_symbolic_name U, 0 [symbolic = %U.1 (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%U.loc11: type);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@ImplicitAs.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert(constants.%Dest, constants.%Self.1) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %.1 => constants.%.3
// CHECK:STDOUT:   %Self => constants.%Self.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(type) {
// CHECK:STDOUT:   %Dest => type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1 => constants.%.6
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.2
// CHECK:STDOUT:   %Convert => constants.%Convert.2
// CHECK:STDOUT:   %.2 => constants.%.7
// CHECK:STDOUT:   %.3 => constants.%.8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @NonTypeImpls(constants.%U) {
// CHECK:STDOUT:   %U.1 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_right_of_impls_non_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.Self: type = bind_symbolic_name .Self, 0 [symbolic]
// CHECK:STDOUT:   %.1: i32 = int_literal 7 [template]
// CHECK:STDOUT:   %ImplicitAs.type: type = generic_interface_type @ImplicitAs [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %ImplicitAs: %ImplicitAs.type = struct_value () [template]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %.3: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Self.1: @ImplicitAs.%.1 (%.3) = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.2: %.3 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.1: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.1: %Convert.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.4: type = assoc_entity_type %.3, %Convert.type.1 [symbolic]
// CHECK:STDOUT:   %.5: %.4 = assoc_entity element0, imports.%import_ref.5 [symbolic]
// CHECK:STDOUT:   %.6: type = interface_type @ImplicitAs, @ImplicitAs(type) [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert, @ImplicitAs(type) [template]
// CHECK:STDOUT:   %Convert.2: %Convert.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.7: type = assoc_entity_type %.6, %Convert.type.2 [template]
// CHECK:STDOUT:   %.8: %.7 = assoc_entity element0, imports.%import_ref.5 [template]
// CHECK:STDOUT:   %.9: %.4 = assoc_entity element0, imports.%import_ref.6 [symbolic]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %ImplsNonType.type: type = fn_type @ImplsNonType [template]
// CHECK:STDOUT:   %ImplsNonType: %ImplsNonType.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .ImplicitAs = %import_ref.1
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %ImplicitAs.type = import_ref Core//prelude/operators/as, inst+40, loaded [template = constants.%ImplicitAs]
// CHECK:STDOUT:   %import_ref.2 = import_ref Core//prelude/operators/as, inst+45, unloaded
// CHECK:STDOUT:   %import_ref.3: @ImplicitAs.%.2 (%.4) = import_ref Core//prelude/operators/as, inst+63, loaded [symbolic = @ImplicitAs.%.3 (constants.%.9)]
// CHECK:STDOUT:   %import_ref.4 = import_ref Core//prelude/operators/as, inst+56, unloaded
// CHECK:STDOUT:   %import_ref.5 = import_ref Core//prelude/operators/as, inst+56, unloaded
// CHECK:STDOUT:   %import_ref.6 = import_ref Core//prelude/operators/as, inst+56, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .ImplsNonType = %ImplsNonType.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %ImplsNonType.decl: %ImplsNonType.type = fn_decl @ImplsNonType [template = constants.%ImplsNonType] {
// CHECK:STDOUT:     %U.patt: type = symbolic_binding_pattern U, 0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: type = bind_symbolic_name .Self, 0 [symbolic = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: type = name_ref .Self, %.Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %.loc11_44.1: i32 = int_literal 7 [template = constants.%.1]
// CHECK:STDOUT:     %.loc11_44.2: type = interface_type @ImplicitAs, @ImplicitAs(type) [template = constants.%.6]
// CHECK:STDOUT:     %.loc11_44.3: %.7 = specific_constant imports.%import_ref.3, @ImplicitAs(type) [template = constants.%.8]
// CHECK:STDOUT:     %Convert.ref: %.7 = name_ref Convert, %.loc11_44.3 [template = constants.%.8]
// CHECK:STDOUT:     %.loc11_44.4: type = converted %.loc11_44.1, <error> [template = <error>]
// CHECK:STDOUT:     %.loc11_26: type = where_expr %.Self [template = type] {
// CHECK:STDOUT:       requirement_impls %.Self.ref, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.param: type = param U, runtime_param<invalid>
// CHECK:STDOUT:     %U.loc11: type = bind_symbolic_name U, 0, %U.param [symbolic = %U.1 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(constants.%Dest: type) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic = %.1 (constants.%.3)]
// CHECK:STDOUT:   %Self: %.3 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic = %Convert.type (constants.%Convert.type.1)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.1) = struct_value () [symbolic = %Convert (constants.%Convert.1)]
// CHECK:STDOUT:   %.2: type = assoc_entity_type @ImplicitAs.%.1 (%.3), @ImplicitAs.%Convert.type (%Convert.type.1) [symbolic = %.2 (constants.%.4)]
// CHECK:STDOUT:   %.3: @ImplicitAs.%.2 (%.4) = assoc_entity element0, imports.%import_ref.5 [symbolic = %.3 (constants.%.5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.2
// CHECK:STDOUT:     .Convert = imports.%import_ref.3
// CHECK:STDOUT:     witness = (imports.%import_ref.4)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert(constants.%Dest: type, constants.%Self.1: @ImplicitAs.%.1 (%.3)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %.1: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic = %.1 (constants.%.3)]
// CHECK:STDOUT:   %Self: %.3 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self: @Convert.%Self (%Self.2)]() -> @Convert.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ImplsNonType(%U.loc11: type) {
// CHECK:STDOUT:   %U.1: type = bind_symbolic_name U, 0 [symbolic = %U.1 (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%U.loc11: type);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@ImplicitAs.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert(constants.%Dest, constants.%Self.1) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %.1 => constants.%.3
// CHECK:STDOUT:   %Self => constants.%Self.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(type) {
// CHECK:STDOUT:   %Dest => type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1 => constants.%.6
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.2
// CHECK:STDOUT:   %Convert => constants.%Convert.2
// CHECK:STDOUT:   %.2 => constants.%.7
// CHECK:STDOUT:   %.3 => constants.%.8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplsNonType(constants.%U) {
// CHECK:STDOUT:   %U.1 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_right_of_impls_non_facet_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.Self: type = bind_symbolic_name .Self, 0 [symbolic]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %ImplsOfNonFacetType.type: type = fn_type @ImplsOfNonFacetType [template]
// CHECK:STDOUT:   %ImplsOfNonFacetType: %ImplsOfNonFacetType.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .ImplsOfNonFacetType = %ImplsOfNonFacetType.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %ImplsOfNonFacetType.decl: %ImplsOfNonFacetType.type = fn_decl @ImplsOfNonFacetType [template = constants.%ImplsOfNonFacetType] {
// CHECK:STDOUT:     %U.patt: type = symbolic_binding_pattern U, 0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: type = bind_symbolic_name .Self, 0 [symbolic = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: type = name_ref .Self, %.Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc8_51.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc8_51.2: type = converted %int.make_type_32, %.loc8_51.1 [template = i32]
// CHECK:STDOUT:     %.loc8_33: type = where_expr %.Self [template = type] {
// CHECK:STDOUT:       requirement_impls %.Self.ref, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.param: type = param U, runtime_param<invalid>
// CHECK:STDOUT:     %U.loc8: type = bind_symbolic_name U, 0, %U.param [symbolic = %U.1 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ImplsOfNonFacetType(%U.loc8: type) {
// CHECK:STDOUT:   %U.1: type = bind_symbolic_name U, 0 [symbolic = %U.1 (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%U.loc8: type);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplsOfNonFacetType(constants.%U) {
// CHECK:STDOUT:   %U.1 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_enforce_constraint.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %.3: type = interface_type @J [template]
// CHECK:STDOUT:   %Self.1: %.3 = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %.4: type = tuple_type () [template]
// CHECK:STDOUT:   %.5: <witness> = interface_witness () [template]
// CHECK:STDOUT:   %DoesNotImplI.type: type = fn_type @DoesNotImplI [template]
// CHECK:STDOUT:   %DoesNotImplI: %DoesNotImplI.type = struct_value () [template]
// CHECK:STDOUT:   %Impls.type: type = fn_type @Impls [template]
// CHECK:STDOUT:   %Impls: %Impls.type = struct_value () [template]
// CHECK:STDOUT:   %V: %.3 = bind_symbolic_name V, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type: type = generic_interface_type @ImplicitAs [template]
// CHECK:STDOUT:   %ImplicitAs: %ImplicitAs.type = struct_value () [template]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %.6: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Self.2: @ImplicitAs.%.1 (%.6) = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.3: %.6 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.1: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.1: %Convert.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.7: type = assoc_entity_type %.6, %Convert.type.1 [symbolic]
// CHECK:STDOUT:   %.8: %.7 = assoc_entity element0, imports.%import_ref.11 [symbolic]
// CHECK:STDOUT:   %.9: type = interface_type @ImplicitAs, @ImplicitAs(%.3) [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert, @ImplicitAs(%.3) [template]
// CHECK:STDOUT:   %Convert.2: %Convert.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.10: type = assoc_entity_type %.9, %Convert.type.2 [template]
// CHECK:STDOUT:   %.11: %.10 = assoc_entity element0, imports.%import_ref.11 [template]
// CHECK:STDOUT:   %.12: %.7 = assoc_entity element0, imports.%import_ref.12 [symbolic]
// CHECK:STDOUT:   %.Self: %.3 = bind_symbolic_name .Self, 0 [symbolic]
// CHECK:STDOUT:   %Y: %.3 = bind_symbolic_name Y, 0 [symbolic]
// CHECK:STDOUT:   %EmptyStruct.type: type = fn_type @EmptyStruct [template]
// CHECK:STDOUT:   %EmptyStruct: %EmptyStruct.type = struct_value () [template]
// CHECK:STDOUT:   %NotEmptyStruct.type: type = fn_type @NotEmptyStruct [template]
// CHECK:STDOUT:   %NotEmptyStruct: %NotEmptyStruct.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: type = import_ref Main//state_constraints, inst+3, loaded [template = constants.%.3]
// CHECK:STDOUT:   %import_ref.2 = import_ref Main//state_constraints, inst+7, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref Main//state_constraints, inst+32, unloaded
// CHECK:STDOUT:   %import_ref.4: %Impls.type = import_ref Main//state_constraints, inst+49, loaded [template = constants.%Impls]
// CHECK:STDOUT:   %import_ref.5 = import_ref Main//state_constraints, inst+69, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .ImplicitAs = %import_ref.7
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.6 = import_ref Main//state_constraints, inst+5, unloaded
// CHECK:STDOUT:   %import_ref.7: %ImplicitAs.type = import_ref Core//prelude/operators/as, inst+40, loaded [template = constants.%ImplicitAs]
// CHECK:STDOUT:   %import_ref.8 = import_ref Core//prelude/operators/as, inst+45, unloaded
// CHECK:STDOUT:   %import_ref.9: @ImplicitAs.%.2 (%.7) = import_ref Core//prelude/operators/as, inst+63, loaded [symbolic = @ImplicitAs.%.3 (constants.%.12)]
// CHECK:STDOUT:   %import_ref.10 = import_ref Core//prelude/operators/as, inst+56, unloaded
// CHECK:STDOUT:   %import_ref.11 = import_ref Core//prelude/operators/as, inst+56, unloaded
// CHECK:STDOUT:   %import_ref.12 = import_ref Core//prelude/operators/as, inst+56, unloaded
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
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %J.ref: type = name_ref J, imports.%import_ref.1 [template = constants.%.3]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %DoesNotImplI.decl: %DoesNotImplI.type = fn_decl @DoesNotImplI [template = constants.%DoesNotImplI] {} {}
// CHECK:STDOUT:   %EmptyStruct.decl: %EmptyStruct.type = fn_decl @EmptyStruct [template = constants.%EmptyStruct] {
// CHECK:STDOUT:     %Y.patt: %.3 = symbolic_binding_pattern Y, 0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %J.ref: type = name_ref J, imports.%import_ref.1 [template = constants.%.3]
// CHECK:STDOUT:     %.Self: %.3 = bind_symbolic_name .Self, 0 [symbolic = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %.3 = name_ref .Self, %.Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %.loc26_38: %.1 = struct_literal ()
// CHECK:STDOUT:     %.loc26_22: type = where_expr %.Self [template = constants.%.3] {
// CHECK:STDOUT:       requirement_equivalent %.Self.ref, %.loc26_38
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Y.param: %.3 = param Y, runtime_param<invalid>
// CHECK:STDOUT:     %Y.loc26: %.3 = bind_symbolic_name Y, 0, %Y.param [symbolic = %Y.1 (constants.%Y)]
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
// CHECK:STDOUT: generic interface @ImplicitAs(constants.%Dest: type) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic = %.1 (constants.%.6)]
// CHECK:STDOUT:   %Self: %.6 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.3)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic = %Convert.type (constants.%Convert.type.1)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.1) = struct_value () [symbolic = %Convert (constants.%Convert.1)]
// CHECK:STDOUT:   %.2: type = assoc_entity_type @ImplicitAs.%.1 (%.6), @ImplicitAs.%Convert.type (%Convert.type.1) [symbolic = %.2 (constants.%.7)]
// CHECK:STDOUT:   %.3: @ImplicitAs.%.2 (%.7) = assoc_entity element0, imports.%import_ref.11 [symbolic = %.3 (constants.%.8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.8
// CHECK:STDOUT:     .Convert = imports.%import_ref.9
// CHECK:STDOUT:     witness = (imports.%import_ref.10)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %C as %.3 {
// CHECK:STDOUT:   %.loc8: <witness> = interface_witness () [template = constants.%.5]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = %.loc8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc7: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DoesNotImplI() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Impls.ref: %Impls.type = name_ref Impls, imports.%import_ref.4 [template = constants.%Impls]
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %.loc23_8.1: type = interface_type @ImplicitAs, @ImplicitAs(constants.%.3) [template = constants.%.9]
// CHECK:STDOUT:   %.loc23_8.2: %.10 = specific_constant imports.%import_ref.9, @ImplicitAs(constants.%.3) [template = constants.%.11]
// CHECK:STDOUT:   %Convert.ref: %.10 = name_ref Convert, %.loc23_8.2 [template = constants.%.11]
// CHECK:STDOUT:   %.loc23_8.3: %.3 = converted %C.ref, <error> [template = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Impls(constants.%V: %.3) {
// CHECK:STDOUT:   %V.2: %.3 = bind_symbolic_name V, 0 [symbolic = %V.2 (constants.%V)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%V.1: %.3);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert(constants.%Dest: type, constants.%Self.2: @ImplicitAs.%.1 (%.6)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %.1: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic = %.1 (constants.%.6)]
// CHECK:STDOUT:   %Self: %.6 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self: @Convert.%Self (%Self.3)]() -> @Convert.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @EmptyStruct(%Y.loc26: %.3) {
// CHECK:STDOUT:   %Y.1: %.3 = bind_symbolic_name Y, 0 [symbolic = %Y.1 (constants.%Y)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%Y.loc26: %.3);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NotEmptyStruct() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %EmptyStruct.ref: %EmptyStruct.type = name_ref EmptyStruct, file.%EmptyStruct.decl [template = constants.%EmptyStruct]
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %.loc39_14.1: type = interface_type @ImplicitAs, @ImplicitAs(constants.%.3) [template = constants.%.9]
// CHECK:STDOUT:   %.loc39_14.2: %.10 = specific_constant imports.%import_ref.9, @ImplicitAs(constants.%.3) [template = constants.%.11]
// CHECK:STDOUT:   %Convert.ref: %.10 = name_ref Convert, %.loc39_14.2 [template = constants.%.11]
// CHECK:STDOUT:   %.loc39_14.3: %.3 = converted %C.ref, <error> [template = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Impls(constants.%V) {
// CHECK:STDOUT:   %V.2 => constants.%V
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@ImplicitAs.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert(constants.%Dest, constants.%Self.2) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %.1 => constants.%.6
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%.3) {
// CHECK:STDOUT:   %Dest => constants.%.3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1 => constants.%.9
// CHECK:STDOUT:   %Self => constants.%Self.3
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.2
// CHECK:STDOUT:   %Convert => constants.%Convert.2
// CHECK:STDOUT:   %.2 => constants.%.10
// CHECK:STDOUT:   %.3 => constants.%.11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @EmptyStruct(constants.%Y) {
// CHECK:STDOUT:   %Y.1 => constants.%Y
// CHECK:STDOUT: }
// CHECK:STDOUT:

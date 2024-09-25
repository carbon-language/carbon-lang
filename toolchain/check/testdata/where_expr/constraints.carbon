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

fn Equal(T:! I where .Member = {});

fn EqualEqual(U:! I where .Self == ());

fn Impls(V:! J where .Self impls I);

fn And(W:! I where .Second impls I and .Member = .Second);

// --- todo_check_constraints.carbon

library "[[@TEST_NAME]]";

import library "state_constraints";

// TODO: Should fail since `2` can't be converted to the type of `I.Member`,
// but constraints are not yet type checked.
fn TypeMismatch(X:! I where .Member = 2);

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
  // CHECK:STDERR: state_constraints.carbon:15:10: note: initializing generic parameter `V` declared here
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
// CHECK:STDOUT:   %Self.1: %.1 = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %.2: type = interface_type @I [template]
// CHECK:STDOUT:   %Self.2: %.2 = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %.3: type = assoc_entity_type %.2, type [template]
// CHECK:STDOUT:   %.4: %.3 = assoc_entity element0, @I.%Member [template]
// CHECK:STDOUT:   %.5: type = tuple_type () [template]
// CHECK:STDOUT:   %.6: type = assoc_entity_type %.2, %.1 [template]
// CHECK:STDOUT:   %.7: %.6 = assoc_entity element1, @I.%Second [template]
// CHECK:STDOUT:   %.Self.1: %.2 = bind_symbolic_name .Self 0 [symbolic]
// CHECK:STDOUT:   %.8: type = struct_type {} [template]
// CHECK:STDOUT:   %T: %.2 = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %Equal.type: type = fn_type @Equal [template]
// CHECK:STDOUT:   %Equal: %Equal.type = struct_value () [template]
// CHECK:STDOUT:   %U: %.2 = bind_symbolic_name U 0 [symbolic]
// CHECK:STDOUT:   %EqualEqual.type: type = fn_type @EqualEqual [template]
// CHECK:STDOUT:   %EqualEqual: %EqualEqual.type = struct_value () [template]
// CHECK:STDOUT:   %.Self.2: %.1 = bind_symbolic_name .Self 0 [symbolic]
// CHECK:STDOUT:   %V: %.1 = bind_symbolic_name V 0 [symbolic]
// CHECK:STDOUT:   %Impls.type: type = fn_type @Impls [template]
// CHECK:STDOUT:   %Impls: %Impls.type = struct_value () [template]
// CHECK:STDOUT:   %W: %.2 = bind_symbolic_name W 0 [symbolic]
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
// CHECK:STDOUT:     .Equal = %Equal.decl
// CHECK:STDOUT:     .EqualEqual = %EqualEqual.decl
// CHECK:STDOUT:     .Impls = %Impls.decl
// CHECK:STDOUT:     .And = %And.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [template = constants.%.1] {} {}
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [template = constants.%.2] {} {}
// CHECK:STDOUT:   %Equal.decl: %Equal.type = fn_decl @Equal [template = constants.%Equal] {
// CHECK:STDOUT:     %T.patt: %.2 = symbolic_binding_pattern T 0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [template = constants.%.2]
// CHECK:STDOUT:     %.Self.1: %.2 = bind_symbolic_name .Self 0 [symbolic = %.Self.2 (constants.%.Self.1)]
// CHECK:STDOUT:     %.Self.ref: %.2 = name_ref .Self, %.Self.1 [symbolic = %.Self.2 (constants.%.Self.1)]
// CHECK:STDOUT:     %Member.ref: %.3 = name_ref Member, @I.%.loc7 [template = constants.%.4]
// CHECK:STDOUT:     %.loc11: %.8 = struct_literal ()
// CHECK:STDOUT:     %T.param: %.2 = param T, runtime_param<invalid>
// CHECK:STDOUT:     %T.loc11: %.2 = bind_symbolic_name T 0, %T.param [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %EqualEqual.decl: %EqualEqual.type = fn_decl @EqualEqual [template = constants.%EqualEqual] {
// CHECK:STDOUT:     %U.patt: %.2 = symbolic_binding_pattern U 0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [template = constants.%.2]
// CHECK:STDOUT:     %.Self.1: %.2 = bind_symbolic_name .Self 0 [symbolic = %.Self.2 (constants.%.Self.1)]
// CHECK:STDOUT:     %.Self.ref: %.2 = name_ref .Self, %.Self.1 [symbolic = %.Self.2 (constants.%.Self.1)]
// CHECK:STDOUT:     %.loc13: %.5 = tuple_literal ()
// CHECK:STDOUT:     %U.param: %.2 = param U, runtime_param<invalid>
// CHECK:STDOUT:     %U.loc13: %.2 = bind_symbolic_name U 0, %U.param [symbolic = %U.1 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Impls.decl: %Impls.type = fn_decl @Impls [template = constants.%Impls] {
// CHECK:STDOUT:     %V.patt: %.1 = symbolic_binding_pattern V 0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [template = constants.%.1]
// CHECK:STDOUT:     %.Self.1: %.1 = bind_symbolic_name .Self 0 [symbolic = %.Self.2 (constants.%.Self.2)]
// CHECK:STDOUT:     %.Self.ref: %.1 = name_ref .Self, %.Self.1 [symbolic = %.Self.2 (constants.%.Self.2)]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [template = constants.%.2]
// CHECK:STDOUT:     %V.param: %.1 = param V, runtime_param<invalid>
// CHECK:STDOUT:     %V.loc15: %.1 = bind_symbolic_name V 0, %V.param [symbolic = %V.1 (constants.%V)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %And.decl: %And.type = fn_decl @And [template = constants.%And] {
// CHECK:STDOUT:     %W.patt: %.2 = symbolic_binding_pattern W 0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %I.ref.loc17_12: type = name_ref I, file.%I.decl [template = constants.%.2]
// CHECK:STDOUT:     %.Self.1: %.2 = bind_symbolic_name .Self 0 [symbolic = %.Self.2 (constants.%.Self.1)]
// CHECK:STDOUT:     %.Self.ref.loc17_20: %.2 = name_ref .Self, %.Self.1 [symbolic = %.Self.2 (constants.%.Self.1)]
// CHECK:STDOUT:     %Second.ref.loc17_20: %.6 = name_ref Second, @I.%.loc8 [template = constants.%.7]
// CHECK:STDOUT:     %I.ref.loc17_34: type = name_ref I, file.%I.decl [template = constants.%.2]
// CHECK:STDOUT:     %.Self.ref.loc17_40: %.2 = name_ref .Self, %.Self.1 [symbolic = %.Self.2 (constants.%.Self.1)]
// CHECK:STDOUT:     %Member.ref: %.3 = name_ref Member, @I.%.loc7 [template = constants.%.4]
// CHECK:STDOUT:     %.Self.ref.loc17_50: %.2 = name_ref .Self, %.Self.1 [symbolic = %.Self.2 (constants.%.Self.1)]
// CHECK:STDOUT:     %Second.ref.loc17_50: %.6 = name_ref Second, @I.%.loc8 [template = constants.%.7]
// CHECK:STDOUT:     %W.param: %.2 = param W, runtime_param<invalid>
// CHECK:STDOUT:     %W.loc17: %.2 = bind_symbolic_name W 0, %W.param [symbolic = %W.1 (constants.%W)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %.1 = bind_symbolic_name Self 0 [symbolic = constants.%Self.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %.2 = bind_symbolic_name Self 0 [symbolic = constants.%Self.2]
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
// CHECK:STDOUT: generic fn @Equal(%T.loc11: %.2) {
// CHECK:STDOUT:   %.Self.2: %.2 = bind_symbolic_name .Self 0 [symbolic = %.Self.2 (constants.%.Self.1)]
// CHECK:STDOUT:   %T.1: %.2 = bind_symbolic_name T 0 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.loc11: %.2);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @EqualEqual(%U.loc13: %.2) {
// CHECK:STDOUT:   %.Self.2: %.2 = bind_symbolic_name .Self 0 [symbolic = %.Self.2 (constants.%.Self.1)]
// CHECK:STDOUT:   %U.1: %.2 = bind_symbolic_name U 0 [symbolic = %U.1 (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%U.loc13: %.2);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Impls(%V.loc15: %.1) {
// CHECK:STDOUT:   %.Self.2: %.1 = bind_symbolic_name .Self 0 [symbolic = %.Self.2 (constants.%.Self.2)]
// CHECK:STDOUT:   %V.1: %.1 = bind_symbolic_name V 0 [symbolic = %V.1 (constants.%V)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%V.loc15: %.1);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @And(%W.loc17: %.2) {
// CHECK:STDOUT:   %.Self.2: %.2 = bind_symbolic_name .Self 0 [symbolic = %.Self.2 (constants.%.Self.1)]
// CHECK:STDOUT:   %W.1: %.2 = bind_symbolic_name W 0 [symbolic = %W.1 (constants.%W)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%W.loc17: %.2);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Equal(constants.%T) {
// CHECK:STDOUT:   %.Self.2 => constants.%T
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @EqualEqual(constants.%U) {
// CHECK:STDOUT:   %.Self.2 => constants.%U
// CHECK:STDOUT:   %U.1 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Impls(constants.%V) {
// CHECK:STDOUT:   %.Self.2 => constants.%V
// CHECK:STDOUT:   %V.1 => constants.%V
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @And(constants.%W) {
// CHECK:STDOUT:   %.Self.2 => constants.%W
// CHECK:STDOUT:   %W.1 => constants.%W
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_check_constraints.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @I [template]
// CHECK:STDOUT:   %Self: %.1 = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %.Self: %.1 = bind_symbolic_name .Self 0 [symbolic]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %.3: type = assoc_entity_type %.1, type [template]
// CHECK:STDOUT:   %.4: %.3 = assoc_entity element0, imports.%import_ref.12 [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %X: %.1 = bind_symbolic_name X 0 [symbolic]
// CHECK:STDOUT:   %TypeMismatch.type: type = fn_type @TypeMismatch [template]
// CHECK:STDOUT:   %TypeMismatch: %TypeMismatch.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1 = import_ref Main//state_constraints, inst+3, unloaded
// CHECK:STDOUT:   %import_ref.2: type = import_ref Main//state_constraints, inst+7, loaded [template = constants.%.1]
// CHECK:STDOUT:   %import_ref.3 = import_ref Main//state_constraints, inst+32, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref Main//state_constraints, inst+45, unloaded
// CHECK:STDOUT:   %import_ref.5 = import_ref Main//state_constraints, inst+59, unloaded
// CHECK:STDOUT:   %import_ref.6 = import_ref Main//state_constraints, inst+77, unloaded
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
// CHECK:STDOUT:   %import_ref.7 = import_ref Main//state_constraints, inst+9, unloaded
// CHECK:STDOUT:   %import_ref.8: %.3 = import_ref Main//state_constraints, inst+13, loaded [template = constants.%.4]
// CHECK:STDOUT:   %import_ref.9 = import_ref Main//state_constraints, inst+19, unloaded
// CHECK:STDOUT:   %import_ref.10 = import_ref Main//state_constraints, inst+11, unloaded
// CHECK:STDOUT:   %import_ref.11 = import_ref Main//state_constraints, inst+17, unloaded
// CHECK:STDOUT:   %import_ref.12 = import_ref Main//state_constraints, inst+11, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .J = imports.%import_ref.1
// CHECK:STDOUT:     .I = imports.%import_ref.2
// CHECK:STDOUT:     .Equal = imports.%import_ref.3
// CHECK:STDOUT:     .EqualEqual = imports.%import_ref.4
// CHECK:STDOUT:     .Impls = imports.%import_ref.5
// CHECK:STDOUT:     .And = imports.%import_ref.6
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .TypeMismatch = %TypeMismatch.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %TypeMismatch.decl: %TypeMismatch.type = fn_decl @TypeMismatch [template = constants.%TypeMismatch] {
// CHECK:STDOUT:     %X.patt: %.1 = symbolic_binding_pattern X 0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %I.ref: type = name_ref I, imports.%import_ref.2 [template = constants.%.1]
// CHECK:STDOUT:     %.Self.1: %.1 = bind_symbolic_name .Self 0 [symbolic = %.Self.2 (constants.%.Self)]
// CHECK:STDOUT:     %.Self.ref: %.1 = name_ref .Self, %.Self.1 [symbolic = %.Self.2 (constants.%.Self)]
// CHECK:STDOUT:     %Member.ref: %.3 = name_ref Member, imports.%import_ref.8 [template = constants.%.4]
// CHECK:STDOUT:     %.loc8: i32 = int_literal 2 [template = constants.%.5]
// CHECK:STDOUT:     %X.param: %.1 = param X, runtime_param<invalid>
// CHECK:STDOUT:     %X.loc8: %.1 = bind_symbolic_name X 0, %X.param [symbolic = %X.1 (constants.%X)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.7
// CHECK:STDOUT:   .Member = imports.%import_ref.8
// CHECK:STDOUT:   .Second = imports.%import_ref.9
// CHECK:STDOUT:   witness = (imports.%import_ref.10, imports.%import_ref.11)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TypeMismatch(%X.loc8: %.1) {
// CHECK:STDOUT:   %.Self.2: %.1 = bind_symbolic_name .Self 0 [symbolic = %.Self.2 (constants.%.Self)]
// CHECK:STDOUT:   %X.1: %.1 = bind_symbolic_name X 0 [symbolic = %X.1 (constants.%X)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%X.loc8: %.1);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TypeMismatch(constants.%X) {
// CHECK:STDOUT:   %.Self.2 => constants.%X
// CHECK:STDOUT:   %X.1 => constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_enforce_constraint.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %.3: type = interface_type @J [template]
// CHECK:STDOUT:   %Self.1: %.3 = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %.4: type = tuple_type () [template]
// CHECK:STDOUT:   %.5: <witness> = interface_witness () [template]
// CHECK:STDOUT:   %DoesNotImplI.type: type = fn_type @DoesNotImplI [template]
// CHECK:STDOUT:   %DoesNotImplI: %DoesNotImplI.type = struct_value () [template]
// CHECK:STDOUT:   %Impls.type: type = fn_type @Impls [template]
// CHECK:STDOUT:   %Impls: %Impls.type = struct_value () [template]
// CHECK:STDOUT:   %V: %.3 = bind_symbolic_name V 0 [symbolic]
// CHECK:STDOUT:   %.Self.1: %.3 = bind_symbolic_name .Self 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type: type = generic_interface_type @ImplicitAs [template]
// CHECK:STDOUT:   %ImplicitAs: %ImplicitAs.type = struct_value () [template]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest 0 [symbolic]
// CHECK:STDOUT:   %.6: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Self.2: @ImplicitAs.%.1 (%.6) = bind_symbolic_name Self 1 [symbolic]
// CHECK:STDOUT:   %Self.3: %.6 = bind_symbolic_name Self 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.1: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.1: %Convert.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.7: type = assoc_entity_type %.6, %Convert.type.1 [symbolic]
// CHECK:STDOUT:   %.8: %.7 = assoc_entity element0, imports.%import_ref.12 [symbolic]
// CHECK:STDOUT:   %.9: type = interface_type @ImplicitAs, @ImplicitAs(%.3) [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert, @ImplicitAs(%.3) [template]
// CHECK:STDOUT:   %Convert.2: %Convert.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.10: type = assoc_entity_type %.9, %Convert.type.2 [template]
// CHECK:STDOUT:   %.11: %.10 = assoc_entity element0, imports.%import_ref.12 [template]
// CHECK:STDOUT:   %.12: %.7 = assoc_entity element0, imports.%import_ref.13 [symbolic]
// CHECK:STDOUT:   %.Self.2: %.3 = bind_symbolic_name .Self 0 [symbolic]
// CHECK:STDOUT:   %Y: %.3 = bind_symbolic_name Y 0 [symbolic]
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
// CHECK:STDOUT:   %import_ref.4 = import_ref Main//state_constraints, inst+45, unloaded
// CHECK:STDOUT:   %import_ref.5: %Impls.type = import_ref Main//state_constraints, inst+59, loaded [template = constants.%Impls]
// CHECK:STDOUT:   %import_ref.6 = import_ref Main//state_constraints, inst+77, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .ImplicitAs = %import_ref.8
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.7 = import_ref Main//state_constraints, inst+5, unloaded
// CHECK:STDOUT:   %import_ref.8: %ImplicitAs.type = import_ref Core//prelude/operators/as, inst+40, loaded [template = constants.%ImplicitAs]
// CHECK:STDOUT:   %import_ref.9 = import_ref Core//prelude/operators/as, inst+45, unloaded
// CHECK:STDOUT:   %import_ref.10: @ImplicitAs.%.2 (%.7) = import_ref Core//prelude/operators/as, inst+63, loaded [symbolic = @ImplicitAs.%.3 (constants.%.12)]
// CHECK:STDOUT:   %import_ref.11 = import_ref Core//prelude/operators/as, inst+56, unloaded
// CHECK:STDOUT:   %import_ref.12 = import_ref Core//prelude/operators/as, inst+56, unloaded
// CHECK:STDOUT:   %import_ref.13 = import_ref Core//prelude/operators/as, inst+56, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .J = imports.%import_ref.1
// CHECK:STDOUT:     .I = imports.%import_ref.2
// CHECK:STDOUT:     .Equal = imports.%import_ref.3
// CHECK:STDOUT:     .EqualEqual = imports.%import_ref.4
// CHECK:STDOUT:     .Impls = imports.%import_ref.5
// CHECK:STDOUT:     .And = imports.%import_ref.6
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .DoesNotImplI = %DoesNotImplI.decl
// CHECK:STDOUT:     .EmptyStruct = %EmptyStruct.decl
// CHECK:STDOUT:     .NotEmptyStruct = %NotEmptyStruct.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @impl {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %J.ref: type = name_ref J, imports.%import_ref.1 [template = constants.%.3]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %DoesNotImplI.decl: %DoesNotImplI.type = fn_decl @DoesNotImplI [template = constants.%DoesNotImplI] {} {}
// CHECK:STDOUT:   %EmptyStruct.decl: %EmptyStruct.type = fn_decl @EmptyStruct [template = constants.%EmptyStruct] {
// CHECK:STDOUT:     %Y.patt: %.3 = symbolic_binding_pattern Y 0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %J.ref: type = name_ref J, imports.%import_ref.1 [template = constants.%.3]
// CHECK:STDOUT:     %.Self.1: %.3 = bind_symbolic_name .Self 0 [symbolic = %.Self.2 (constants.%.Self.2)]
// CHECK:STDOUT:     %.Self.ref: %.3 = name_ref .Self, %.Self.1 [symbolic = %.Self.2 (constants.%.Self.2)]
// CHECK:STDOUT:     %.loc26: %.1 = struct_literal ()
// CHECK:STDOUT:     %Y.param: %.3 = param Y, runtime_param<invalid>
// CHECK:STDOUT:     %Y.loc26: %.3 = bind_symbolic_name Y 0, %Y.param [symbolic = %Y.1 (constants.%Y)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NotEmptyStruct.decl: %NotEmptyStruct.type = fn_decl @NotEmptyStruct [template = constants.%NotEmptyStruct] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.7
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(constants.%Dest: type) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic = %.1 (constants.%.6)]
// CHECK:STDOUT:   %Self: %.6 = bind_symbolic_name Self 1 [symbolic = %Self (constants.%Self.3)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic = %Convert.type (constants.%Convert.type.1)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.1) = struct_value () [symbolic = %Convert (constants.%Convert.1)]
// CHECK:STDOUT:   %.2: type = assoc_entity_type @ImplicitAs.%.1 (%.6), @ImplicitAs.%Convert.type (%Convert.type.1) [symbolic = %.2 (constants.%.7)]
// CHECK:STDOUT:   %.3: @ImplicitAs.%.2 (%.7) = assoc_entity element0, imports.%import_ref.12 [symbolic = %.3 (constants.%.8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.9
// CHECK:STDOUT:     .Convert = imports.%import_ref.10
// CHECK:STDOUT:     witness = (imports.%import_ref.11)
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
// CHECK:STDOUT:   %Impls.ref: %Impls.type = name_ref Impls, imports.%import_ref.5 [template = constants.%Impls]
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %.loc23_8.1: type = interface_type @ImplicitAs, @ImplicitAs(constants.%.3) [template = constants.%.9]
// CHECK:STDOUT:   %.loc23_8.2: %.10 = specific_constant imports.%import_ref.10, @ImplicitAs(constants.%.3) [template = constants.%.11]
// CHECK:STDOUT:   %Convert.ref: %.10 = name_ref Convert, %.loc23_8.2 [template = constants.%.11]
// CHECK:STDOUT:   %.loc23_8.3: %.3 = converted %C.ref, <error> [template = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Impls(constants.%V: %.3) {
// CHECK:STDOUT:   %.Self: %.3 = bind_symbolic_name .Self 0 [symbolic = %.Self (constants.%.Self.1)]
// CHECK:STDOUT:   %V.2: %.3 = bind_symbolic_name V 0 [symbolic = %V.2 (constants.%V)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%V.1: %.3);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert(constants.%Dest: type, constants.%Self.2: @ImplicitAs.%.1 (%.6)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %.1: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic = %.1 (constants.%.6)]
// CHECK:STDOUT:   %Self: %.6 = bind_symbolic_name Self 1 [symbolic = %Self (constants.%Self.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self: @Convert.%Self (%Self.3)]() -> @Convert.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @EmptyStruct(%Y.loc26: %.3) {
// CHECK:STDOUT:   %.Self.2: %.3 = bind_symbolic_name .Self 0 [symbolic = %.Self.2 (constants.%.Self.2)]
// CHECK:STDOUT:   %Y.1: %.3 = bind_symbolic_name Y 0 [symbolic = %Y.1 (constants.%Y)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%Y.loc26: %.3);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NotEmptyStruct() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %EmptyStruct.ref: %EmptyStruct.type = name_ref EmptyStruct, file.%EmptyStruct.decl [template = constants.%EmptyStruct]
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %.loc39_14.1: type = interface_type @ImplicitAs, @ImplicitAs(constants.%.3) [template = constants.%.9]
// CHECK:STDOUT:   %.loc39_14.2: %.10 = specific_constant imports.%import_ref.10, @ImplicitAs(constants.%.3) [template = constants.%.11]
// CHECK:STDOUT:   %Convert.ref: %.10 = name_ref Convert, %.loc39_14.2 [template = constants.%.11]
// CHECK:STDOUT:   %.loc39_14.3: %.3 = converted %C.ref, <error> [template = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Impls(constants.%V) {
// CHECK:STDOUT:   %.Self => constants.%V
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
// CHECK:STDOUT:   %.Self.2 => constants.%Y
// CHECK:STDOUT:   %Y.1 => constants.%Y
// CHECK:STDOUT: }
// CHECK:STDOUT:

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/where_expr/designator.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/where_expr/designator.carbon

// --- success.carbon

library "[[@TEST_NAME]]";

interface I {
  let Member:! type;
}

fn PeriodSelf(T:! I where .Self == ());

fn PeriodMember(U:! I where .Member == ());

fn TypeSelfImpls(V:! type where .Self impls I);

// --- fail_wrong_member.carbon

library "[[@TEST_NAME]]";

interface J {
  let Member:! type;
}

// CHECK:STDERR: fail_wrong_member.carbon:[[@LINE+4]]:31: error: name `Mismatch` not found
// CHECK:STDERR: fn PeriodMismatch(W:! J where .Mismatch = {});
// CHECK:STDERR:                               ^~~~~~~~~
// CHECK:STDERR:
fn PeriodMismatch(W:! J where .Mismatch = {});

// --- fail_designator_matches_var.carbon

library "[[@TEST_NAME]]";

fn Foo() -> () {
  var x: ();
  // CHECK:STDERR: fail_designator_matches_var.carbon:[[@LINE+5]]:10: error: name `.Self` not found
  // CHECK:STDERR:   return .x;
  // CHECK:STDERR:          ^~
  // CHECK:STDERR: fail_designator_matches_var.carbon: note: designator may only be used when `.Self` is in scope
  // CHECK:STDERR:
  return .x;
}

// --- fail_unknown_designator.carbon

library "[[@TEST_NAME]]";

fn Bar() -> () {
  // CHECK:STDERR: fail_unknown_designator.carbon:[[@LINE+5]]:10: error: name `.Self` not found
  // CHECK:STDERR:   return .undef;
  // CHECK:STDERR:          ^~~~~~
  // CHECK:STDERR: fail_unknown_designator.carbon: note: designator may only be used when `.Self` is in scope
  // CHECK:STDERR:
  return .undef;
}

// --- fail_dot_self_method_return_value.carbon

library "[[@TEST_NAME]]";

class C {
  // CHECK:STDERR: fail_dot_self_method_return_value.carbon:[[@LINE+4]]:27: error: name `.Self` not found
  // CHECK:STDERR:   fn F() -> Self { return .Self; }
  // CHECK:STDERR:                           ^~~~~
  // CHECK:STDERR:
  fn F() -> Self { return .Self; }
}

// --- fail_dot_self_method_return_type.carbon

library "[[@TEST_NAME]]";

class D {
  // CHECK:STDERR: fail_dot_self_method_return_type.carbon:[[@LINE+3]]:13: error: name `.Self` not found
  // CHECK:STDERR:   fn G() -> .Self { return Self; }
  // CHECK:STDERR:             ^~~~~
  fn G() -> .Self { return Self; }
}

// CHECK:STDOUT: --- success.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @I [template]
// CHECK:STDOUT:   %Self: %.1 = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %.2: type = assoc_entity_type %.1, type [template]
// CHECK:STDOUT:   %.3: %.2 = assoc_entity element0, @I.%Member [template]
// CHECK:STDOUT:   %.Self.1: %.1 = bind_symbolic_name .Self, 0 [symbolic]
// CHECK:STDOUT:   %.4: type = tuple_type () [template]
// CHECK:STDOUT:   %T: %.1 = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %PeriodSelf.type: type = fn_type @PeriodSelf [template]
// CHECK:STDOUT:   %PeriodSelf: %PeriodSelf.type = struct_value () [template]
// CHECK:STDOUT:   %U: %.1 = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %PeriodMember.type: type = fn_type @PeriodMember [template]
// CHECK:STDOUT:   %PeriodMember: %PeriodMember.type = struct_value () [template]
// CHECK:STDOUT:   %.Self.2: type = bind_symbolic_name .Self, 0 [symbolic]
// CHECK:STDOUT:   %V: type = bind_symbolic_name V, 0 [symbolic]
// CHECK:STDOUT:   %TypeSelfImpls.type: type = fn_type @TypeSelfImpls [template]
// CHECK:STDOUT:   %TypeSelfImpls: %TypeSelfImpls.type = struct_value () [template]
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
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .PeriodSelf = %PeriodSelf.decl
// CHECK:STDOUT:     .PeriodMember = %PeriodMember.decl
// CHECK:STDOUT:     .TypeSelfImpls = %TypeSelfImpls.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [template = constants.%.1] {} {}
// CHECK:STDOUT:   %PeriodSelf.decl: %PeriodSelf.type = fn_decl @PeriodSelf [template = constants.%PeriodSelf] {
// CHECK:STDOUT:     %T.patt: %.1 = symbolic_binding_pattern T, 0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [template = constants.%.1]
// CHECK:STDOUT:     %.Self: %.1 = bind_symbolic_name .Self, 0 [symbolic = constants.%.Self.1]
// CHECK:STDOUT:     %.Self.ref: %.1 = name_ref .Self, %.Self [symbolic = constants.%.Self.1]
// CHECK:STDOUT:     %.loc8_37: %.4 = tuple_literal ()
// CHECK:STDOUT:     %.loc8_21: type = where_expr %.Self [template = constants.%.1] {
// CHECK:STDOUT:       requirement_equivalent %.Self.ref, %.loc8_37
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.param: %.1 = param T, runtime_param<invalid>
// CHECK:STDOUT:     %T.loc8: %.1 = bind_symbolic_name T, 0, %T.param [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PeriodMember.decl: %PeriodMember.type = fn_decl @PeriodMember [template = constants.%PeriodMember] {
// CHECK:STDOUT:     %U.patt: %.1 = symbolic_binding_pattern U, 0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [template = constants.%.1]
// CHECK:STDOUT:     %.Self: %.1 = bind_symbolic_name .Self, 0 [symbolic = constants.%.Self.1]
// CHECK:STDOUT:     %.Self.ref: %.1 = name_ref .Self, %.Self [symbolic = constants.%.Self.1]
// CHECK:STDOUT:     %Member.ref: %.2 = name_ref Member, @I.%.loc5 [template = constants.%.3]
// CHECK:STDOUT:     %.loc10_41: %.4 = tuple_literal ()
// CHECK:STDOUT:     %.loc10_23: type = where_expr %.Self [template = constants.%.1] {
// CHECK:STDOUT:       requirement_equivalent %Member.ref, %.loc10_41
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.param: %.1 = param U, runtime_param<invalid>
// CHECK:STDOUT:     %U.loc10: %.1 = bind_symbolic_name U, 0, %U.param [symbolic = %U.1 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TypeSelfImpls.decl: %TypeSelfImpls.type = fn_decl @TypeSelfImpls [template = constants.%TypeSelfImpls] {
// CHECK:STDOUT:     %V.patt: type = symbolic_binding_pattern V, 0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: type = bind_symbolic_name .Self, 0 [symbolic = constants.%.Self.2]
// CHECK:STDOUT:     %.Self.ref: type = name_ref .Self, %.Self [symbolic = constants.%.Self.2]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [template = constants.%.1]
// CHECK:STDOUT:     %.loc12: type = where_expr %.Self [template = type] {
// CHECK:STDOUT:       requirement_impls %.Self.ref, %I.ref
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %V.param: type = param V, runtime_param<invalid>
// CHECK:STDOUT:     %V.loc12: type = bind_symbolic_name V, 0, %V.param [symbolic = %V.1 (constants.%V)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %.1 = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Member: type = assoc_const_decl Member [template]
// CHECK:STDOUT:   %.loc5: %.2 = assoc_entity element0, %Member [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Member = %.loc5
// CHECK:STDOUT:   witness = (%Member)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @PeriodSelf(%T.loc8: %.1) {
// CHECK:STDOUT:   %T.1: %.1 = bind_symbolic_name T, 0 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.loc8: %.1);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @PeriodMember(%U.loc10: %.1) {
// CHECK:STDOUT:   %U.1: %.1 = bind_symbolic_name U, 0 [symbolic = %U.1 (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%U.loc10: %.1);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TypeSelfImpls(%V.loc12: type) {
// CHECK:STDOUT:   %V.1: type = bind_symbolic_name V, 0 [symbolic = %V.1 (constants.%V)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%V.loc12: type);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @PeriodSelf(constants.%T) {
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @PeriodMember(constants.%U) {
// CHECK:STDOUT:   %U.1 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TypeSelfImpls(constants.%V) {
// CHECK:STDOUT:   %V.1 => constants.%V
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_wrong_member.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @J [template]
// CHECK:STDOUT:   %Self: %.1 = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %.2: type = assoc_entity_type %.1, type [template]
// CHECK:STDOUT:   %.3: %.2 = assoc_entity element0, @J.%Member [template]
// CHECK:STDOUT:   %.Self: %.1 = bind_symbolic_name .Self, 0 [symbolic]
// CHECK:STDOUT:   %.4: type = tuple_type () [template]
// CHECK:STDOUT:   %.5: type = struct_type {} [template]
// CHECK:STDOUT:   %W: %.1 = bind_symbolic_name W, 0 [symbolic]
// CHECK:STDOUT:   %PeriodMismatch.type: type = fn_type @PeriodMismatch [template]
// CHECK:STDOUT:   %PeriodMismatch: %PeriodMismatch.type = struct_value () [template]
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
// CHECK:STDOUT:     .PeriodMismatch = %PeriodMismatch.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [template = constants.%.1] {} {}
// CHECK:STDOUT:   %PeriodMismatch.decl: %PeriodMismatch.type = fn_decl @PeriodMismatch [template = constants.%PeriodMismatch] {
// CHECK:STDOUT:     %W.patt: %.1 = symbolic_binding_pattern W, 0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [template = constants.%.1]
// CHECK:STDOUT:     %.Self: %.1 = bind_symbolic_name .Self, 0 [symbolic = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %.1 = name_ref .Self, %.Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %Mismatch.ref: <error> = name_ref Mismatch, <error> [template = <error>]
// CHECK:STDOUT:     %.loc12_44: %.5 = struct_literal ()
// CHECK:STDOUT:     %.loc12_25: type = where_expr %.Self [template = constants.%.1] {
// CHECK:STDOUT:       requirement_rewrite %Mismatch.ref, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %W.param: %.1 = param W, runtime_param<invalid>
// CHECK:STDOUT:     %W.loc12: %.1 = bind_symbolic_name W, 0, %W.param [symbolic = %W.1 (constants.%W)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %.1 = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Member: type = assoc_const_decl Member [template]
// CHECK:STDOUT:   %.loc5: %.2 = assoc_entity element0, %Member [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Member = %.loc5
// CHECK:STDOUT:   witness = (%Member)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @PeriodMismatch(%W.loc12: %.1) {
// CHECK:STDOUT:   %W.1: %.1 = bind_symbolic_name W, 0 [symbolic = %W.1 (constants.%W)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%W.loc12: %.1);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @PeriodMismatch(constants.%W) {
// CHECK:STDOUT:   %W.1 => constants.%W
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_designator_matches_var.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Foo.type: type = fn_type @Foo [template]
// CHECK:STDOUT:   %Foo: %Foo.type = struct_value () [template]
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
// CHECK:STDOUT:     .Foo = %Foo.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Foo.decl: %Foo.type = fn_decl @Foo [template = constants.%Foo] {} {
// CHECK:STDOUT:     %.loc4_14.1: %.1 = tuple_literal ()
// CHECK:STDOUT:     %.loc4_14.2: type = converted %.loc4_14.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %return: ref %.1 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Foo() -> %.1 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc5_11.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc5_11.2: type = converted %.loc5_11.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %x.var: ref %.1 = var x
// CHECK:STDOUT:   %x: ref %.1 = bind_name x, %x.var
// CHECK:STDOUT:   %.Self.ref: <error> = name_ref .Self, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_unknown_designator.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Bar.type: type = fn_type @Bar [template]
// CHECK:STDOUT:   %Bar: %Bar.type = struct_value () [template]
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
// CHECK:STDOUT:     .Bar = %Bar.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Bar.decl: %Bar.type = fn_decl @Bar [template = constants.%Bar] {} {
// CHECK:STDOUT:     %.loc4_14.1: %.1 = tuple_literal ()
// CHECK:STDOUT:     %.loc4_14.2: type = converted %.loc4_14.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %return: ref %.1 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Bar() -> %.1 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.Self.ref: <error> = name_ref .Self, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_dot_self_method_return_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %.3: <witness> = complete_type_witness %.2 [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.2 [template]
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
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [template = constants.%C]
// CHECK:STDOUT:     %return: ref %C = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10: <witness> = complete_type_witness %.2 [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %return: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.Self.ref: <error> = name_ref .Self, <error> [template = <error>]
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_dot_self_method_return_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %.3: <witness> = complete_type_witness %.2 [template]
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
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %D.decl: type = class_decl @D [template = constants.%D] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {} {
// CHECK:STDOUT:     %.Self.ref: <error> = name_ref .Self, <error> [template = <error>]
// CHECK:STDOUT:     %return: ref <error> = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9: <witness> = complete_type_witness %.2 [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> <error> {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%D [template = constants.%D]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:

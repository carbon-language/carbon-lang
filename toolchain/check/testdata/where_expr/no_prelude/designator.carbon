// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/where_expr/no_prelude/designator.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/where_expr/no_prelude/designator.carbon

// --- success.carbon

library "[[@TEST_NAME]]";

interface I {
  let Member:! type;
}

fn PeriodSelf(T:! I where .Self == ());

fn PeriodMember(U:! I where .Member = {});

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
// CHECK:STDOUT:   %Self: %.1 = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %.2: type = assoc_entity_type %.1, type [template]
// CHECK:STDOUT:   %.3: %.2 = assoc_entity element0, @I.%Member [template]
// CHECK:STDOUT:   %.Self.1: %.1 = bind_symbolic_name .Self 0 [symbolic]
// CHECK:STDOUT:   %.4: type = tuple_type () [template]
// CHECK:STDOUT:   %.5: type = where_expr %.1, (<unexpected>.inst+15.loc8_33) [template]
// CHECK:STDOUT:   %T: %.5 = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %PeriodSelf.type: type = fn_type @PeriodSelf [template]
// CHECK:STDOUT:   %PeriodSelf: %PeriodSelf.type = struct_value () [template]
// CHECK:STDOUT:   %.6: type = struct_type {} [template]
// CHECK:STDOUT:   %.7: type = where_expr %.1, (<unexpected>.inst+32.loc10_37) [template]
// CHECK:STDOUT:   %U: %.7 = bind_symbolic_name U 0 [symbolic]
// CHECK:STDOUT:   %PeriodMember.type: type = fn_type @PeriodMember [template]
// CHECK:STDOUT:   %PeriodMember: %PeriodMember.type = struct_value () [template]
// CHECK:STDOUT:   %.Self.2: type = bind_symbolic_name .Self 0 [symbolic]
// CHECK:STDOUT:   %.8: type = where_expr type, (<unexpected>.inst+47.loc12_39) [template]
// CHECK:STDOUT:   %V: %.8 = bind_symbolic_name V 0 [symbolic]
// CHECK:STDOUT:   %TypeSelfImpls.type: type = fn_type @TypeSelfImpls [template]
// CHECK:STDOUT:   %TypeSelfImpls: %TypeSelfImpls.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .PeriodSelf = %PeriodSelf.decl
// CHECK:STDOUT:     .PeriodMember = %PeriodMember.decl
// CHECK:STDOUT:     .TypeSelfImpls = %TypeSelfImpls.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [template = constants.%.1] {}
// CHECK:STDOUT:   %PeriodSelf.decl: %PeriodSelf.type = fn_decl @PeriodSelf [template = constants.%PeriodSelf] {
// CHECK:STDOUT:     %I.ref.loc8: type = name_ref I, %I.decl [template = constants.%.1]
// CHECK:STDOUT:     %.Self.1: %.1 = bind_symbolic_name .Self 0 [symbolic = @PeriodSelf.%.Self (constants.%.Self.1)]
// CHECK:STDOUT:     %.Self.ref.loc8: %.1 = name_ref .Self, %.Self.1 [symbolic = @PeriodSelf.%.Self (constants.%.Self.1)]
// CHECK:STDOUT:     %.loc8_37: %.4 = tuple_literal ()
// CHECK:STDOUT:     %.loc8_21: type = where_expr %.1, (<unexpected>.inst+15.loc8_33) [template = constants.%.5]
// CHECK:STDOUT:     %T.loc8_15.1: %.5 = param T, runtime_param<invalid>
// CHECK:STDOUT:     @PeriodSelf.%T.loc8: %.5 = bind_symbolic_name T 0, %T.loc8_15.1 [symbolic = @PeriodSelf.%T.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PeriodMember.decl: %PeriodMember.type = fn_decl @PeriodMember [template = constants.%PeriodMember] {
// CHECK:STDOUT:     %I.ref.loc10: type = name_ref I, %I.decl [template = constants.%.1]
// CHECK:STDOUT:     %.Self.2: %.1 = bind_symbolic_name .Self 0 [symbolic = @PeriodMember.%.Self (constants.%.Self.1)]
// CHECK:STDOUT:     %.Self.ref.loc10: %.1 = name_ref .Self, %.Self.2 [symbolic = @PeriodMember.%.Self (constants.%.Self.1)]
// CHECK:STDOUT:     %Member.ref: %.2 = name_ref Member, @I.%.loc5 [template = constants.%.3]
// CHECK:STDOUT:     %.loc10_40: %.6 = struct_literal ()
// CHECK:STDOUT:     %.loc10_23: type = where_expr %.1, (<unexpected>.inst+32.loc10_37) [template = constants.%.7]
// CHECK:STDOUT:     %U.loc10_17.1: %.7 = param U, runtime_param<invalid>
// CHECK:STDOUT:     @PeriodMember.%U.loc10: %.7 = bind_symbolic_name U 0, %U.loc10_17.1 [symbolic = @PeriodMember.%U.1 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TypeSelfImpls.decl: %TypeSelfImpls.type = fn_decl @TypeSelfImpls [template = constants.%TypeSelfImpls] {
// CHECK:STDOUT:     %.Self.3: type = bind_symbolic_name .Self 0 [symbolic = @TypeSelfImpls.%.Self (constants.%.Self.2)]
// CHECK:STDOUT:     %.Self.ref.loc12: type = name_ref .Self, %.Self.3 [symbolic = @TypeSelfImpls.%.Self (constants.%.Self.2)]
// CHECK:STDOUT:     %I.ref.loc12: type = name_ref I, %I.decl [template = constants.%.1]
// CHECK:STDOUT:     %.loc12: type = where_expr type, (<unexpected>.inst+47.loc12_39) [template = constants.%.8]
// CHECK:STDOUT:     %V.loc12_18.1: %.8 = param V, runtime_param<invalid>
// CHECK:STDOUT:     @TypeSelfImpls.%V.loc12: %.8 = bind_symbolic_name V 0, %V.loc12_18.1 [symbolic = @TypeSelfImpls.%V.1 (constants.%V)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %.1 = bind_symbolic_name Self 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Member: type = assoc_const_decl Member [template]
// CHECK:STDOUT:   %.loc5: %.2 = assoc_entity element0, %Member [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Member = %.loc5
// CHECK:STDOUT:   witness = (%Member)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @PeriodSelf(%T.loc8: %.5) {
// CHECK:STDOUT:   %.Self: %.1 = bind_symbolic_name .Self 0 [symbolic = %.Self (constants.%.Self.1)]
// CHECK:STDOUT:   %T.1: %.5 = bind_symbolic_name T 0 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.loc8: %.5);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @PeriodMember(%U.loc10: %.7) {
// CHECK:STDOUT:   %.Self: %.1 = bind_symbolic_name .Self 0 [symbolic = %.Self (constants.%.Self.1)]
// CHECK:STDOUT:   %U.1: %.7 = bind_symbolic_name U 0 [symbolic = %U.1 (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%U.loc10: %.7);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TypeSelfImpls(%V.loc12: %.8) {
// CHECK:STDOUT:   %.Self: type = bind_symbolic_name .Self 0 [symbolic = %.Self (constants.%.Self.2)]
// CHECK:STDOUT:   %V.1: %.8 = bind_symbolic_name V 0 [symbolic = %V.1 (constants.%V)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%V.loc12: %.8);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @PeriodSelf(constants.%T) {
// CHECK:STDOUT:   %.Self => constants.%T
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @PeriodMember(constants.%U) {
// CHECK:STDOUT:   %.Self => constants.%U
// CHECK:STDOUT:   %U.1 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TypeSelfImpls(constants.%V) {
// CHECK:STDOUT:   %.Self => constants.%V
// CHECK:STDOUT:   %V.1 => constants.%V
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_wrong_member.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @J [template]
// CHECK:STDOUT:   %Self: %.1 = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %.2: type = assoc_entity_type %.1, type [template]
// CHECK:STDOUT:   %.3: %.2 = assoc_entity element0, @J.%Member [template]
// CHECK:STDOUT:   %.Self: %.1 = bind_symbolic_name .Self 0 [symbolic]
// CHECK:STDOUT:   %.4: type = tuple_type () [template]
// CHECK:STDOUT:   %.5: type = struct_type {} [template]
// CHECK:STDOUT:   %.6: type = where_expr %.1, (<unexpected>.inst+17.loc12_41) [template]
// CHECK:STDOUT:   %W: %.6 = bind_symbolic_name W 0 [symbolic]
// CHECK:STDOUT:   %PeriodMismatch.type: type = fn_type @PeriodMismatch [template]
// CHECK:STDOUT:   %PeriodMismatch: %PeriodMismatch.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .PeriodMismatch = %PeriodMismatch.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [template = constants.%.1] {}
// CHECK:STDOUT:   %PeriodMismatch.decl: %PeriodMismatch.type = fn_decl @PeriodMismatch [template = constants.%PeriodMismatch] {
// CHECK:STDOUT:     %J.ref: type = name_ref J, %J.decl [template = constants.%.1]
// CHECK:STDOUT:     %.Self: %.1 = bind_symbolic_name .Self 0 [symbolic = @PeriodMismatch.%.Self (constants.%.Self)]
// CHECK:STDOUT:     %.Self.ref: %.1 = name_ref .Self, %.Self [symbolic = @PeriodMismatch.%.Self (constants.%.Self)]
// CHECK:STDOUT:     %Mismatch.ref: <error> = name_ref Mismatch, <error> [template = <error>]
// CHECK:STDOUT:     %.loc12_44: %.5 = struct_literal ()
// CHECK:STDOUT:     %.loc12_25: type = where_expr %.1, (<unexpected>.inst+17.loc12_41) [template = constants.%.6]
// CHECK:STDOUT:     %W.loc12_19.1: %.6 = param W, runtime_param<invalid>
// CHECK:STDOUT:     @PeriodMismatch.%W.loc12: %.6 = bind_symbolic_name W 0, %W.loc12_19.1 [symbolic = @PeriodMismatch.%W.1 (constants.%W)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %.1 = bind_symbolic_name Self 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Member: type = assoc_const_decl Member [template]
// CHECK:STDOUT:   %.loc5: %.2 = assoc_entity element0, %Member [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Member = %.loc5
// CHECK:STDOUT:   witness = (%Member)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @PeriodMismatch(%W.loc12: %.6) {
// CHECK:STDOUT:   %.Self: %.1 = bind_symbolic_name .Self 0 [symbolic = %.Self (constants.%.Self)]
// CHECK:STDOUT:   %W.1: %.6 = bind_symbolic_name W 0 [symbolic = %W.1 (constants.%W)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%W.loc12: %.6);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @PeriodMismatch(constants.%W) {
// CHECK:STDOUT:   %.Self => constants.%W
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
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Foo = %Foo.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl: %Foo.type = fn_decl @Foo [template = constants.%Foo] {
// CHECK:STDOUT:     %.loc4_14.1: %.1 = tuple_literal ()
// CHECK:STDOUT:     %.loc4_14.2: type = converted %.loc4_14.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     @Foo.%return: ref %.1 = var <return slot>
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
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Bar = %Bar.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Bar.decl: %Bar.type = fn_decl @Bar [template = constants.%Bar] {
// CHECK:STDOUT:     %.loc4_14.1: %.1 = tuple_literal ()
// CHECK:STDOUT:     %.loc4_14.2: type = converted %.loc4_14.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     @Bar.%return: ref %.1 = var <return slot>
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
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [template = constants.%C]
// CHECK:STDOUT:     %return.var: ref %C = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10: <witness> = complete_type_witness %.2 [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> @C.%return.var: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.Self.ref: <error> = name_ref .Self, <error> [template = <error>]
// CHECK:STDOUT:   return <error> to @C.%return.var
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
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl: type = class_decl @D [template = constants.%D] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %.Self.ref: <error> = name_ref .Self, <error> [template = <error>]
// CHECK:STDOUT:     %return.var: ref <error> = var <return slot>
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

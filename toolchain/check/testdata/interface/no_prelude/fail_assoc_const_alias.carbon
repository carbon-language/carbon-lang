// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/no_prelude/fail_assoc_const_alias.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/no_prelude/fail_assoc_const_alias.carbon

// --- core.carbon
package Core;

interface ImplicitAs(Dest:! type) {
  fn Convert[self: Self]() -> Dest;
}

// --- fail_alias_to_different_interface.carbon
library "[[@TEST_NAME]]";

import Core;

interface I {
  let T:! type;
}

interface J {
  // CHECK:STDERR: fail_alias_to_different_interface.carbon:[[@LINE+7]]:13: error: member access into undefined facet type `I` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   alias U = I.T;
  // CHECK:STDERR:             ^~~
  // CHECK:STDERR: fail_alias_to_different_interface.carbon:[[@LINE-8]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface I {
  // CHECK:STDERR: ^~~~~~~~~~~~~
  // CHECK:STDERR:
  alias U = I.T;
  fn F() -> U;
}

// --- alias_to_different_interface_with_requires.carbon
library "[[@TEST_NAME]]";

import Core;

interface I2 {
  let T2:! type;
}

interface J2;

// CHECK:STDERR: alias_to_different_interface_with_requires.carbon:[[@LINE+7]]:36: error: member access into undefined facet type `I2` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl forall [V:! J2] V as I2 where .T2 = () {}
// CHECK:STDERR:                                    ^~~
// CHECK:STDERR: alias_to_different_interface_with_requires.carbon:[[@LINE-9]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface I2 {
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [V:! J2] V as I2 where .T2 = () {}

interface J2 {
  // CHECK:STDERR: alias_to_different_interface_with_requires.carbon:[[@LINE+7]]:14: error: member access into undefined facet type `I2` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   alias U2 = I2.T2;
  // CHECK:STDERR:              ^~~~~
  // CHECK:STDERR: alias_to_different_interface_with_requires.carbon:[[@LINE-19]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface I2 {
  // CHECK:STDERR: ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  alias U2 = I2.T2;
  fn F2() -> U2;
}

// --- fail_call_method_alias.carbon
library "[[@TEST_NAME]]";

import Core;

interface A {
  fn F[self: Self]() -> type;
}
interface B {
  // CHECK:STDERR: fail_call_method_alias.carbon:[[@LINE+7]]:13: error: member access into undefined facet type `A` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   alias F = A.F;
  // CHECK:STDERR:             ^~~
  // CHECK:STDERR: fail_call_method_alias.carbon:[[@LINE-7]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface A {
  // CHECK:STDERR: ^~~~~~~~~~~~~
  // CHECK:STDERR:
  alias F = A.F;
  fn G() -> F();
}

// --- call_with_compound_member_access.carbon
library "[[@TEST_NAME]]";

import Core;

interface C {
  fn F[self: Self]();
  // TODO: Add `default` once supported.
  fn G[self: Self]() {
    // CHECK:STDERR: call_with_compound_member_access.carbon:[[@LINE+7]]:5: error: member access into undefined facet type `C` [QualifiedExprInUndefinedFacetTypeScope]
    // CHECK:STDERR:     self.F();
    // CHECK:STDERR:     ^~~~~~
    // CHECK:STDERR: call_with_compound_member_access.carbon:[[@LINE-7]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
    // CHECK:STDERR: interface C {
    // CHECK:STDERR: ^~~~~~~~~~~~~
    // CHECK:STDERR:
    self.F();
    self.(F)();
  }
}

// CHECK:STDOUT: --- core.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.96f: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.96f = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.07f: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.07f = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert: %Convert.type = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type %ImplicitAs.type.07f [symbolic]
// CHECK:STDOUT:   %assoc0: %ImplicitAs.assoc_type = assoc_entity element0, @ImplicitAs.%Convert.decl [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %ImplicitAs.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.decl: %ImplicitAs.type.96f = interface_decl @ImplicitAs [concrete = constants.%ImplicitAs.generic] {
// CHECK:STDOUT:     %Dest.patt.loc3_22.1: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt.loc3_22.2 (constants.%Dest.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Dest.loc3_22.1: type = bind_symbolic_name Dest, 0 [symbolic = %Dest.loc3_22.2 (constants.%Dest)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(%Dest.loc3_22.1: type) {
// CHECK:STDOUT:   %Dest.loc3_22.2: type = bind_symbolic_name Dest, 0 [symbolic = %Dest.loc3_22.2 (constants.%Dest)]
// CHECK:STDOUT:   %Dest.patt.loc3_22.2: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt.loc3_22.2 (constants.%Dest.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest.loc3_22.2)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.07f)]
// CHECK:STDOUT:   %Self.2: %ImplicitAs.type.07f = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert, @ImplicitAs(%Dest.loc3_22.2) [symbolic = %Convert.type (constants.%Convert.type)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type) = struct_value () [symbolic = %Convert (constants.%Convert)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type)]
// CHECK:STDOUT:   %assoc0.loc4_35.2: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc4_35.2 (constants.%assoc0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:     %Convert.decl: @ImplicitAs.%Convert.type (%Convert.type) = fn_decl @Convert [symbolic = @ImplicitAs.%Convert (constants.%Convert)] {
// CHECK:STDOUT:       %self.patt: @Convert.%Self.as_type.loc4_20.1 (%Self.as_type) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @Convert.%Self.as_type.loc4_20.1 (%Self.as_type) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:       %return.patt: @Convert.%Dest (%Dest) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @Convert.%Dest (%Dest) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Dest.ref: type = name_ref Dest, @ImplicitAs.%Dest.loc3_22.1 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:       %self.param: @Convert.%Self.as_type.loc4_20.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:       %.loc4_20.1: type = splice_block %.loc4_20.3 [symbolic = %Self.as_type.loc4_20.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:         %.loc4_20.2: @Convert.%ImplicitAs.type (%ImplicitAs.type.07f) = specific_constant @ImplicitAs.%Self.1, @ImplicitAs(constants.%Dest) [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:         %Self.ref: @Convert.%ImplicitAs.type (%ImplicitAs.type.07f) = name_ref Self, %.loc4_20.2 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:         %Self.as_type.loc4_20.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc4_20.1 (constants.%Self.as_type)]
// CHECK:STDOUT:         %.loc4_20.3: type = converted %Self.ref, %Self.as_type.loc4_20.2 [symbolic = %Self.as_type.loc4_20.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Convert.%Self.as_type.loc4_20.1 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @Convert.%Dest (%Dest) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Convert.%Dest (%Dest) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc4_35.1: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc4_35.2 (constants.%assoc0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .Dest = <poisoned>
// CHECK:STDOUT:     .Convert = %assoc0.loc4_35.1
// CHECK:STDOUT:     witness = (%Convert.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert(@ImplicitAs.%Dest.loc3_22.1: type, @ImplicitAs.%Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.07f)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.07f = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc4_20.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc4_20.1 (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.%Self.as_type.loc4_20.1 (%Self.as_type)]() -> @Convert.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest.loc3_22.2 => constants.%Dest
// CHECK:STDOUT:   %Dest.patt.loc3_22.2 => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert(constants.%Dest, constants.%Self) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.07f
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc4_20.1 => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.%Dest) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(%Dest.loc3_22.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_alias_to_different_interface.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.826: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type %I.type [concrete]
// CHECK:STDOUT:   %assoc0.c7e: %I.assoc_type = assoc_entity element0, @I.%T [concrete]
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self.ccd: %J.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type %J.type [concrete]
// CHECK:STDOUT:   %assoc0.ebc: %J.assoc_type = assoc_entity element0, @J.%F.decl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//default
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [concrete = constants.%J.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.826]
// CHECK:STDOUT:   %T: type = assoc_const_decl @T [concrete] {
// CHECK:STDOUT:     %assoc0: %I.assoc_type = assoc_entity element0, @I.%T [concrete = constants.%assoc0.c7e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .T = @T.%assoc0
// CHECK:STDOUT:   witness = (%T)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.ccd]
// CHECK:STDOUT:   %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   %T.ref: <error> = name_ref T, <error> [concrete = <error>]
// CHECK:STDOUT:   %U: <error> = bind_alias U, <error> [concrete = <error>]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %return.patt: <error> = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: <error> = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.ref: <error> = name_ref U, @J.%U [concrete = <error>]
// CHECK:STDOUT:     %return.param: ref <error> = out_param call_param0
// CHECK:STDOUT:     %return: ref <error> = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %J.assoc_type = assoc_entity element0, %F.decl [concrete = constants.%assoc0.ebc]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .I = <poisoned>
// CHECK:STDOUT:   .U = %U
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @T(@I.%Self: %I.type) {
// CHECK:STDOUT:   assoc_const T:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@J.%Self: %J.type) {
// CHECK:STDOUT:   fn() -> <error>;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T(constants.%Self.826) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Self.ccd) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- alias_to_different_interface_with_requires.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I2.type: type = facet_type <@I2> [concrete]
// CHECK:STDOUT:   %Self.c7b: %I2.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %I2.assoc_type: type = assoc_entity_type %I2.type [concrete]
// CHECK:STDOUT:   %assoc0.166: %I2.assoc_type = assoc_entity element0, @I2.%T2 [concrete]
// CHECK:STDOUT:   %J2.type: type = facet_type <@J2> [concrete]
// CHECK:STDOUT:   %V: %J2.type = bind_symbolic_name V, 0 [symbolic]
// CHECK:STDOUT:   %V.patt: %J2.type = symbolic_binding_pattern V, 0 [symbolic]
// CHECK:STDOUT:   %V.as_type: type = facet_access_type %V [symbolic]
// CHECK:STDOUT:   %.Self: %I2.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Self.541: %J2.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F2.type: type = fn_type @F2 [concrete]
// CHECK:STDOUT:   %F2: %F2.type = struct_value () [concrete]
// CHECK:STDOUT:   %J2.assoc_type: type = assoc_entity_type %J2.type [concrete]
// CHECK:STDOUT:   %assoc0.5b2: %J2.assoc_type = assoc_entity element0, @J2.%F2.decl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//default
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I2 = %I2.decl
// CHECK:STDOUT:     .J2 = %J2.decl.loc9
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I2.decl: type = interface_decl @I2 [concrete = constants.%I2.type] {} {}
// CHECK:STDOUT:   %J2.decl.loc9: type = interface_decl @J2 [concrete = constants.%J2.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {
// CHECK:STDOUT:     %V.patt.loc18_14.1: %J2.type = symbolic_binding_pattern V, 0 [symbolic = %V.patt.loc18_14.2 (constants.%V.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %V.ref: %J2.type = name_ref V, %V.loc18_14.1 [symbolic = %V.loc18_14.2 (constants.%V)]
// CHECK:STDOUT:     %V.as_type.loc18_22.1: type = facet_access_type %V.ref [symbolic = %V.as_type.loc18_22.2 (constants.%V.as_type)]
// CHECK:STDOUT:     %.loc18_22: type = converted %V.ref, %V.as_type.loc18_22.1 [symbolic = %V.as_type.loc18_22.2 (constants.%V.as_type)]
// CHECK:STDOUT:     %I2.ref: type = name_ref I2, file.%I2.decl [concrete = constants.%I2.type]
// CHECK:STDOUT:     %.Self: %I2.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %I2.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T2.ref: <error> = name_ref T2, <error> [concrete = <error>]
// CHECK:STDOUT:     %.loc18_43: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc18_30: type = where_expr %.Self [concrete = <error>] {
// CHECK:STDOUT:       requirement_rewrite %T2.ref, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %J2.ref: type = name_ref J2, file.%J2.decl.loc9 [concrete = constants.%J2.type]
// CHECK:STDOUT:     %V.loc18_14.1: %J2.type = bind_symbolic_name V, 0 [symbolic = %V.loc18_14.2 (constants.%V)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J2.decl.loc20: type = interface_decl @J2 [concrete = constants.%J2.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I2 {
// CHECK:STDOUT:   %Self: %I2.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.c7b]
// CHECK:STDOUT:   %T2: type = assoc_const_decl @T2 [concrete] {
// CHECK:STDOUT:     %assoc0: %I2.assoc_type = assoc_entity element0, @I2.%T2 [concrete = constants.%assoc0.166]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .T2 = @T2.%assoc0
// CHECK:STDOUT:   witness = (%T2)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J2 {
// CHECK:STDOUT:   %Self: %J2.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.541]
// CHECK:STDOUT:   %I2.ref: type = name_ref I2, file.%I2.decl [concrete = constants.%I2.type]
// CHECK:STDOUT:   %T2.ref: <error> = name_ref T2, <error> [concrete = <error>]
// CHECK:STDOUT:   %U2: <error> = bind_alias U2, <error> [concrete = <error>]
// CHECK:STDOUT:   %F2.decl: %F2.type = fn_decl @F2 [concrete = constants.%F2] {
// CHECK:STDOUT:     %return.patt: <error> = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: <error> = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U2.ref: <error> = name_ref U2, @J2.%U2 [concrete = <error>]
// CHECK:STDOUT:     %return.param: ref <error> = out_param call_param0
// CHECK:STDOUT:     %return: ref <error> = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %J2.assoc_type = assoc_entity element0, %F2.decl [concrete = constants.%assoc0.5b2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .I2 = <poisoned>
// CHECK:STDOUT:   .U2 = %U2
// CHECK:STDOUT:   .F2 = %assoc0
// CHECK:STDOUT:   witness = (%F2.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @T2(@I2.%Self: %I2.type) {
// CHECK:STDOUT:   assoc_const T2:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl(%V.loc18_14.1: %J2.type) {
// CHECK:STDOUT:   %V.loc18_14.2: %J2.type = bind_symbolic_name V, 0 [symbolic = %V.loc18_14.2 (constants.%V)]
// CHECK:STDOUT:   %V.patt.loc18_14.2: %J2.type = symbolic_binding_pattern V, 0 [symbolic = %V.patt.loc18_14.2 (constants.%V.patt)]
// CHECK:STDOUT:   %V.as_type.loc18_22.2: type = facet_access_type %V.loc18_14.2 [symbolic = %V.as_type.loc18_22.2 (constants.%V.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc18_22 as %.loc18_30 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F2(@J2.%Self: %J2.type) {
// CHECK:STDOUT:   fn() -> <error>;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T2(constants.%Self.c7b) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%V) {
// CHECK:STDOUT:   %V.loc18_14.2 => constants.%V
// CHECK:STDOUT:   %V.patt.loc18_14.2 => constants.%V
// CHECK:STDOUT:   %V.as_type.loc18_22.2 => constants.%V.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F2(constants.%Self.541) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_call_method_alias.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = facet_type <@A> [concrete]
// CHECK:STDOUT:   %Self.31d: %A.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.31d [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %A.assoc_type: type = assoc_entity_type %A.type [concrete]
// CHECK:STDOUT:   %assoc0.70d: %A.assoc_type = assoc_entity element0, @A.%F.decl [concrete]
// CHECK:STDOUT:   %B.type: type = facet_type <@B> [concrete]
// CHECK:STDOUT:   %Self.783: %B.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %B.assoc_type: type = assoc_entity_type %B.type [concrete]
// CHECK:STDOUT:   %assoc0.c1f: %B.assoc_type = assoc_entity element0, @B.%G.decl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//default
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = interface_decl @A [concrete = constants.%A.type] {} {}
// CHECK:STDOUT:   %B.decl: type = interface_decl @B [concrete = constants.%B.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A {
// CHECK:STDOUT:   %Self: %A.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.31d]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %self.patt: @F.%Self.as_type.loc6_14.1 (%Self.as_type) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @F.%Self.as_type.loc6_14.1 (%Self.as_type) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %return.patt: type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: type = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @F.%Self.as_type.loc6_14.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc6_14.1: type = splice_block %.loc6_14.2 [symbolic = %Self.as_type.loc6_14.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:       %Self.ref: %A.type = name_ref Self, @A.%Self [symbolic = %Self (constants.%Self.31d)]
// CHECK:STDOUT:       %Self.as_type.loc6_14.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc6_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc6_14.2: type = converted %Self.ref, %Self.as_type.loc6_14.2 [symbolic = %Self.as_type.loc6_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @F.%Self.as_type.loc6_14.1 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref type = out_param call_param1
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %A.assoc_type = assoc_entity element0, %F.decl [concrete = constants.%assoc0.70d]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @B {
// CHECK:STDOUT:   %Self: %B.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.783]
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   %F.ref: <error> = name_ref F, <error> [concrete = <error>]
// CHECK:STDOUT:   %F: <error> = bind_alias F, <error> [concrete = <error>]
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %return.patt: <error> = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: <error> = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %F.ref: <error> = name_ref F, @B.%F [concrete = <error>]
// CHECK:STDOUT:     %return.param: ref <error> = out_param call_param0
// CHECK:STDOUT:     %return: ref <error> = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %B.assoc_type = assoc_entity element0, %G.decl [concrete = constants.%assoc0.c1f]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT:   .G = %assoc0
// CHECK:STDOUT:   witness = (%G.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@A.%Self: %A.type) {
// CHECK:STDOUT:   %Self: %A.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.31d)]
// CHECK:STDOUT:   %Self.as_type.loc6_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc6_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @F.%Self.as_type.loc6_14.1 (%Self.as_type)]() -> type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(@B.%Self: %B.type) {
// CHECK:STDOUT:   fn() -> <error>;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Self.31d) {
// CHECK:STDOUT:   %Self => constants.%Self.31d
// CHECK:STDOUT:   %Self.as_type.loc6_14.1 => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%Self.783) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- call_with_compound_member_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C.type: type = facet_type <@C> [concrete]
// CHECK:STDOUT:   %Self: %C.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.assoc_type: type = assoc_entity_type %C.type [concrete]
// CHECK:STDOUT:   %assoc0: %C.assoc_type = assoc_entity element0, @C.%F.decl [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1: %C.assoc_type = assoc_entity element1, @C.%G.decl [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %Self.as_wit.iface0: <witness> = facet_access_witness %Self, element0 [symbolic]
// CHECK:STDOUT:   %C.facet: %C.type = facet_value %Self.as_type, (%Self.as_wit.iface0) [symbolic]
// CHECK:STDOUT:   %.70a: type = fn_type_with_self_type %F.type, %C.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.70a = impl_witness_access %Self.as_wit.iface0, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @F(%C.facet) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//default
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = interface_decl @C [concrete = constants.%C.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @C {
// CHECK:STDOUT:   %Self: %C.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %self.patt: @F.%Self.as_type.loc6_14.1 (%Self.as_type) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @F.%Self.as_type.loc6_14.1 (%Self.as_type) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @F.%Self.as_type.loc6_14.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc6_14.1: type = splice_block %.loc6_14.2 [symbolic = %Self.as_type.loc6_14.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:       %Self.ref: %C.type = name_ref Self, @C.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc6_14.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc6_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc6_14.2: type = converted %Self.ref, %Self.as_type.loc6_14.2 [symbolic = %Self.as_type.loc6_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @F.%Self.as_type.loc6_14.1 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %C.assoc_type = assoc_entity element0, %F.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %self.patt: @G.%Self.as_type.loc8_14.1 (%Self.as_type) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @G.%Self.as_type.loc8_14.1 (%Self.as_type) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @G.%Self.as_type.loc8_14.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc8_14.1: type = splice_block %.loc8_14.2 [symbolic = %Self.as_type.loc8_14.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:       %Self.ref: %C.type = name_ref Self, @C.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc8_14.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc8_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc8_14.2: type = converted %Self.ref, %Self.as_type.loc8_14.2 [symbolic = %Self.as_type.loc8_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @G.%Self.as_type.loc8_14.1 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %C.assoc_type = assoc_entity element1, %G.decl [concrete = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   .G = %assoc1
// CHECK:STDOUT:   witness = (%F.decl, %G.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@C.%Self: %C.type) {
// CHECK:STDOUT:   %Self: %C.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc6_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc6_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @F.%Self.as_type.loc6_14.1 (%Self.as_type)]();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(@C.%Self: %C.type) {
// CHECK:STDOUT:   %Self: %C.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc8_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc8_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @G.%Self.as_type.loc8_14.1 (%Self.as_type) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %Self.as_wit.iface0.loc17_11.2: <witness> = facet_access_witness %Self, element0 [symbolic = %Self.as_wit.iface0.loc17_11.2 (constants.%Self.as_wit.iface0)]
// CHECK:STDOUT:   %C.facet: %C.type = facet_value %Self.as_type.loc8_14.1, (%Self.as_wit.iface0.loc17_11.2) [symbolic = %C.facet (constants.%C.facet)]
// CHECK:STDOUT:   %.loc17_11: type = fn_type_with_self_type constants.%F.type, %C.facet [symbolic = %.loc17_11 (constants.%.70a)]
// CHECK:STDOUT:   %impl.elem0.loc17_11.2: @G.%.loc17_11 (%.70a) = impl_witness_access %Self.as_wit.iface0.loc17_11.2, element0 [symbolic = %impl.elem0.loc17_11.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc17_11.2: <specific function> = specific_impl_function %impl.elem0.loc17_11.2, @F(%C.facet) [symbolic = %specific_impl_fn.loc17_11.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @G.%Self.as_type.loc8_14.1 (%Self.as_type)]() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref.loc16: @G.%Self.as_type.loc8_14.1 (%Self.as_type) = name_ref self, %self
// CHECK:STDOUT:     %F.ref.loc16: <error> = name_ref F, <error> [concrete = <error>]
// CHECK:STDOUT:     %self.ref.loc17: @G.%Self.as_type.loc8_14.1 (%Self.as_type) = name_ref self, %self
// CHECK:STDOUT:     %Self.as_wit.iface0.loc17_11.1: <witness> = facet_access_witness @C.%Self, element0 [symbolic = %Self.as_wit.iface0.loc17_11.2 (constants.%Self.as_wit.iface0)]
// CHECK:STDOUT:     %impl.elem0.loc17_11.1: @G.%.loc17_11 (%.70a) = impl_witness_access %Self.as_wit.iface0.loc17_11.1, element0 [symbolic = %impl.elem0.loc17_11.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %F.ref.loc17: @G.%.loc17_11 (%.70a) = name_ref F, %impl.elem0.loc17_11.1 [symbolic = %impl.elem0.loc17_11.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc17_9: <bound method> = bound_method %self.ref.loc17, %F.ref.loc17
// CHECK:STDOUT:     %specific_impl_fn.loc17_11.1: <specific function> = specific_impl_function %F.ref.loc17, @F(constants.%C.facet) [symbolic = %specific_impl_fn.loc17_11.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %bound_method.loc17_14: <bound method> = bound_method %self.ref.loc17, %specific_impl_fn.loc17_11.1
// CHECK:STDOUT:     %.loc17_14: init %empty_tuple.type = call %bound_method.loc17_14(%self.ref.loc17)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc6_14.1 => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc8_14.1 => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%C.facet) {
// CHECK:STDOUT:   %Self => constants.%C.facet
// CHECK:STDOUT:   %Self.as_type.loc6_14.1 => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(@G.%C.facet) {}
// CHECK:STDOUT:

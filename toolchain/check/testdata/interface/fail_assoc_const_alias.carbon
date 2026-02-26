// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/fail_assoc_const_alias.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/fail_assoc_const_alias.carbon

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
  alias U = I.T;
  // CHECK:STDERR: fail_alias_to_different_interface.carbon:[[@LINE+4]]:13: error: cannot convert type `Self` that implements `J` into type implementing `I` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   fn F() -> U;
  // CHECK:STDERR:             ^
  // CHECK:STDERR:
  fn F() -> U;
}

// --- alias_to_different_interface_with_requires.carbon
library "[[@TEST_NAME]]";

import Core;

interface I2 {
  let T2:! type;
}

interface J2;

impl forall [V:! J2] V as I2 where .T2 = () {}

interface J2 {
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
  alias F = A.F;
  // CHECK:STDERR: fail_call_method_alias.carbon:[[@LINE+4]]:13: error: cannot convert type `Self` that implements `B` into type implementing `A` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   fn G() -> F();
  // CHECK:STDERR:             ^
  // CHECK:STDERR:
  fn G() -> F();
}

// --- call_with_compound_member_access.carbon
library "[[@TEST_NAME]]";

import Core;

interface C {
  fn F[self: Self]();
  // TODO: Add `default` once supported.
  fn G[self: Self]() {
    self.F();
    self.(F)();
  }
}

// CHECK:STDOUT: --- core.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Dest: type = symbolic_binding Dest, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.595: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.595 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.9fe: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.9fe = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self [symbolic]
// CHECK:STDOUT:   %pattern_type.8de: type = pattern_type %Self.binding.as_type [symbolic]
// CHECK:STDOUT:   %.3cf: Core.Form = init_form %Dest, call_param1 [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %Dest [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Dest, %Self) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert: %ImplicitAs.WithSelf.Convert.type = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %assoc0: %ImplicitAs.assoc_type = assoc_entity element0, @ImplicitAs.WithSelf.%ImplicitAs.WithSelf.Convert.decl [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %ImplicitAs.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.decl: %ImplicitAs.type.595 = interface_decl @ImplicitAs [concrete = constants.%ImplicitAs.generic] {
// CHECK:STDOUT:     %Dest.patt: %pattern_type.98f = symbolic_binding_pattern Dest, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc3_29.1: type = splice_block %.loc3_29.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc3_29.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Dest.loc3_22.2: type = symbolic_binding Dest, 0 [symbolic = %Dest.loc3_22.1 (constants.%Dest)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(%Dest.loc3_22.2: type) {
// CHECK:STDOUT:   %Dest.loc3_22.1: type = symbolic_binding Dest, 0 [symbolic = %Dest.loc3_22.1 (constants.%Dest)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest.loc3_22.1)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.9fe)]
// CHECK:STDOUT:   %Self.loc3_35.2: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.9fe) = symbolic_binding Self, 1 [symbolic = %Self.loc3_35.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc3_35.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.9fe) = symbolic_binding Self, 1 [symbolic = %Self.loc3_35.2 (constants.%Self)]
// CHECK:STDOUT:     %ImplicitAs.WithSelf.decl = interface_with_self_decl @ImplicitAs [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %ImplicitAs.WithSelf.Convert.decl: @ImplicitAs.WithSelf.%ImplicitAs.WithSelf.Convert.type (%ImplicitAs.WithSelf.Convert.type) = fn_decl @ImplicitAs.WithSelf.Convert [symbolic = @ImplicitAs.WithSelf.%ImplicitAs.WithSelf.Convert (constants.%ImplicitAs.WithSelf.Convert)] {
// CHECK:STDOUT:       %self.patt: @ImplicitAs.WithSelf.Convert.%pattern_type.loc4_14 (%pattern_type.8de) = value_binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @ImplicitAs.WithSelf.Convert.%pattern_type.loc4_14 (%pattern_type.8de) = value_param_pattern %self.patt [concrete]
// CHECK:STDOUT:       %return.patt: @ImplicitAs.WithSelf.Convert.%pattern_type.loc4_28 (%pattern_type.51d) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @ImplicitAs.WithSelf.Convert.%pattern_type.loc4_28 (%pattern_type.51d) = out_param_pattern %return.patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Dest.ref: type = name_ref Dest, @ImplicitAs.%Dest.loc3_22.2 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:       %.loc4_31.2: Core.Form = init_form %Dest.ref, call_param1 [symbolic = %.loc4_31.1 (constants.%.3cf)]
// CHECK:STDOUT:       %self.param: @ImplicitAs.WithSelf.Convert.%Self.binding.as_type (%Self.binding.as_type) = value_param call_param0
// CHECK:STDOUT:       %.loc4_20.1: type = splice_block %.loc4_20.3 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)] {
// CHECK:STDOUT:         %.loc4_20.2: @ImplicitAs.WithSelf.Convert.%ImplicitAs.type (%ImplicitAs.type.9fe) = specific_constant @ImplicitAs.%Self.loc3_35.1, @ImplicitAs(constants.%Dest) [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:         %Self.ref: @ImplicitAs.WithSelf.Convert.%ImplicitAs.type (%ImplicitAs.type.9fe) = name_ref Self, %.loc4_20.2 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:         %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:         %.loc4_20.3: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @ImplicitAs.WithSelf.Convert.%Self.binding.as_type (%Self.binding.as_type) = value_binding self, %self.param
// CHECK:STDOUT:       %return.param: ref @ImplicitAs.WithSelf.Convert.%Dest (%Dest) = out_param call_param1
// CHECK:STDOUT:       %return: ref @ImplicitAs.WithSelf.Convert.%Dest (%Dest) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc4_35.1: @ImplicitAs.WithSelf.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type) = assoc_entity element0, %ImplicitAs.WithSelf.Convert.decl [symbolic = %assoc0.loc4_35.2 (constants.%assoc0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc3_35.1
// CHECK:STDOUT:     .Dest = <poisoned>
// CHECK:STDOUT:     .Dest = <poisoned>
// CHECK:STDOUT:     .Convert = @ImplicitAs.WithSelf.%assoc0.loc4_35.1
// CHECK:STDOUT:     witness = (@ImplicitAs.WithSelf.%ImplicitAs.WithSelf.Convert.decl)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ImplicitAs.WithSelf.Convert(@ImplicitAs.%Dest.loc3_22.2: type, @ImplicitAs.%Self.loc3_35.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.9fe)) {
// CHECK:STDOUT:   %Dest: type = symbolic_binding Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.9fe)]
// CHECK:STDOUT:   %Self: @ImplicitAs.WithSelf.Convert.%ImplicitAs.type (%ImplicitAs.type.9fe) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %pattern_type.loc4_14: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type.loc4_14 (constants.%pattern_type.8de)]
// CHECK:STDOUT:   %.loc4_31.1: Core.Form = init_form %Dest, call_param1 [symbolic = %.loc4_31.1 (constants.%.3cf)]
// CHECK:STDOUT:   %pattern_type.loc4_28: type = pattern_type %Dest [symbolic = %pattern_type.loc4_28 (constants.%pattern_type.51d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @ImplicitAs.WithSelf.Convert.%Self.binding.as_type (%Self.binding.as_type)) -> out %return.param: @ImplicitAs.WithSelf.Convert.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest.loc3_22.1 => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.WithSelf(constants.%Dest, constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.WithSelf.Convert(constants.%Dest, constants.%Self) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.9fe
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %pattern_type.loc4_14 => constants.%pattern_type.8de
// CHECK:STDOUT:   %.loc4_31.1 => constants.%.3cf
// CHECK:STDOUT:   %pattern_type.loc4_28 => constants.%pattern_type.51d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_alias_to_different_interface.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.ab9: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.2e7: %I.assoc_type = assoc_entity element0, @I.WithSelf.%T [concrete]
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self.8a1: %J.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self.8a1 [symbolic]
// CHECK:STDOUT:   %J.WithSelf.F.type: type = fn_type @J.WithSelf.F, @J.WithSelf(%Self.8a1) [symbolic]
// CHECK:STDOUT:   %J.WithSelf.F: %J.WithSelf.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type @J [concrete]
// CHECK:STDOUT:   %assoc0.481: %J.assoc_type = assoc_entity element0, @J.WithSelf.%J.WithSelf.F.decl [concrete]
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
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self.ab9]
// CHECK:STDOUT:   %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %T: type = assoc_const_decl @T [concrete] {
// CHECK:STDOUT:     %assoc0: %I.assoc_type = assoc_entity element0, @I.WithSelf.%T [concrete = constants.%assoc0.2e7]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .T = @T.%assoc0
// CHECK:STDOUT:   witness = (@I.WithSelf.%T)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = symbolic_binding Self, 0 [symbolic = constants.%Self.8a1]
// CHECK:STDOUT:   %J.WithSelf.decl = interface_with_self_decl @J [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   %T.ref: %I.assoc_type = name_ref T, @T.%assoc0 [concrete = constants.%assoc0.2e7]
// CHECK:STDOUT:   %U: %I.assoc_type = alias_binding U, @T.%assoc0 [concrete = constants.%assoc0.2e7]
// CHECK:STDOUT:   %J.WithSelf.F.decl: @J.WithSelf.%J.WithSelf.F.type (%J.WithSelf.F.type) = fn_decl @J.WithSelf.F [symbolic = @J.WithSelf.%J.WithSelf.F (constants.%J.WithSelf.F)] {} {
// CHECK:STDOUT:     %Self.as_type: type = facet_access_type @J.%Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:     %U.ref: <error> = name_ref U, <error> [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %J.assoc_type = assoc_entity element0, %J.WithSelf.F.decl [concrete = constants.%assoc0.481]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .I = <poisoned>
// CHECK:STDOUT:   .I = <poisoned>
// CHECK:STDOUT:   .U = @J.WithSelf.%U
// CHECK:STDOUT:   .F = @J.WithSelf.%assoc0
// CHECK:STDOUT:   witness = (@J.WithSelf.%J.WithSelf.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @J.WithSelf.F(@J.%Self: %J.type) {
// CHECK:STDOUT:   %Self: %J.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.8a1)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> <error>;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self.ab9) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%Self.8a1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.F(constants.%Self.8a1) {
// CHECK:STDOUT:   %Self => constants.%Self.8a1
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- alias_to_different_interface_with_requires.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I2.type: type = facet_type <@I2> [concrete]
// CHECK:STDOUT:   %Self.36e: %I2.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %I2.assoc_type: type = assoc_entity_type @I2 [concrete]
// CHECK:STDOUT:   %assoc0.7c1: %I2.assoc_type = assoc_entity element0, @I2.WithSelf.%T2 [concrete]
// CHECK:STDOUT:   %J2.type: type = facet_type <@J2> [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.c39: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %V: %J2.type = symbolic_binding V, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.b4f: type = pattern_type %J2.type [concrete]
// CHECK:STDOUT:   %V.binding.as_type: type = symbolic_binding_type V, 0, %V [symbolic]
// CHECK:STDOUT:   %.Self.feb: %I2.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self.feb [symbolic_self]
// CHECK:STDOUT:   %I2.lookup_impl_witness.c47: <witness> = lookup_impl_witness %.Self.feb, @I2 [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.f1d: type = impl_witness_access %I2.lookup_impl_witness.c47, element0 [symbolic_self]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %I2_where.type: type = facet_type <@I2 where %impl.elem0.f1d = %empty_tuple.type> [concrete]
// CHECK:STDOUT:   %I2.impl_witness.364: <witness> = impl_witness @V.binding.as_type.as.I2.impl.%I2.impl_witness_table, @V.binding.as_type.as.I2.impl(%V) [symbolic]
// CHECK:STDOUT:   %I2.facet.beb: %I2.type = facet_value %V.binding.as_type, (%I2.impl_witness.364) [symbolic]
// CHECK:STDOUT:   %Self.4a7: %J2.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self.4a7 [symbolic]
// CHECK:STDOUT:   %I2.impl_witness.ddb: <witness> = impl_witness @V.binding.as_type.as.I2.impl.%I2.impl_witness_table, @V.binding.as_type.as.I2.impl(%Self.4a7) [symbolic]
// CHECK:STDOUT:   %.ef7: require_specific_def_type = require_specific_def @V.binding.as_type.as.I2.impl(%Self.4a7) [symbolic]
// CHECK:STDOUT:   %I2.lookup_impl_witness.d97: <witness> = lookup_impl_witness %Self.4a7, @I2 [symbolic]
// CHECK:STDOUT:   %I2.facet.7a5: %I2.type = facet_value %Self.binding.as_type, (%I2.lookup_impl_witness.d97) [symbolic]
// CHECK:STDOUT:   %impl.elem0.006: type = impl_witness_access %I2.lookup_impl_witness.d97, element0 [symbolic]
// CHECK:STDOUT:   %.d9a: Core.Form = init_form %impl.elem0.006, call_param0 [symbolic]
// CHECK:STDOUT:   %pattern_type.a14: type = pattern_type %impl.elem0.006 [symbolic]
// CHECK:STDOUT:   %J2.WithSelf.F2.type: type = fn_type @J2.WithSelf.F2, @J2.WithSelf(%Self.4a7) [symbolic]
// CHECK:STDOUT:   %J2.WithSelf.F2: %J2.WithSelf.F2.type = struct_value () [symbolic]
// CHECK:STDOUT:   %J2.assoc_type: type = assoc_entity_type @J2 [concrete]
// CHECK:STDOUT:   %assoc0.41c: %J2.assoc_type = assoc_entity element0, @J2.WithSelf.%J2.WithSelf.F2.decl [concrete]
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
// CHECK:STDOUT:   impl_decl @V.binding.as_type.as.I2.impl [concrete] {
// CHECK:STDOUT:     %V.patt: %pattern_type.b4f = symbolic_binding_pattern V, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %V.ref: %J2.type = name_ref V, %V.loc11_14.1 [symbolic = %V.loc11_14.2 (constants.%V)]
// CHECK:STDOUT:     %V.as_type: type = facet_access_type %V.ref [symbolic = %V.binding.as_type (constants.%V.binding.as_type)]
// CHECK:STDOUT:     %.loc11_22: type = converted %V.ref, %V.as_type [symbolic = %V.binding.as_type (constants.%V.binding.as_type)]
// CHECK:STDOUT:     %I2.ref: type = name_ref I2, file.%I2.decl [concrete = constants.%I2.type]
// CHECK:STDOUT:     %.Self.1: %I2.type = symbolic_binding .Self [symbolic_self = constants.%.Self.feb]
// CHECK:STDOUT:     %.Self.ref: %I2.type = name_ref .Self, %.Self.1 [symbolic_self = constants.%.Self.feb]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc11_36: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %T2.ref: %I2.assoc_type = name_ref T2, @T2.%assoc0 [concrete = constants.%assoc0.7c1]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%I2.lookup_impl_witness.c47, element0 [symbolic_self = constants.%impl.elem0.f1d]
// CHECK:STDOUT:     %.loc11_43.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc11_43.2: type = converted %.loc11_43.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc11_30: type = where_expr %.Self.1 [concrete = constants.%I2_where.type] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%I2.type
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %.loc11_43.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc11_18: type = splice_block %J2.ref [concrete = constants.%J2.type] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %J2.ref: type = name_ref J2, file.%J2.decl.loc9 [concrete = constants.%J2.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %V.loc11_14.1: %J2.type = symbolic_binding V, 0 [symbolic = %V.loc11_14.2 (constants.%V)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J2.decl.loc13: type = interface_decl @J2 [concrete = constants.%J2.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I2 {
// CHECK:STDOUT:   %Self: %I2.type = symbolic_binding Self, 0 [symbolic = constants.%Self.36e]
// CHECK:STDOUT:   %I2.WithSelf.decl = interface_with_self_decl @I2 [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %T2: type = assoc_const_decl @T2 [concrete] {
// CHECK:STDOUT:     %assoc0: %I2.assoc_type = assoc_entity element0, @I2.WithSelf.%T2 [concrete = constants.%assoc0.7c1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .T2 = @T2.%assoc0
// CHECK:STDOUT:   witness = (@I2.WithSelf.%T2)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J2 {
// CHECK:STDOUT:   %Self: %J2.type = symbolic_binding Self, 0 [symbolic = constants.%Self.4a7]
// CHECK:STDOUT:   %J2.WithSelf.decl = interface_with_self_decl @J2 [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %I2.ref: type = name_ref I2, file.%I2.decl [concrete = constants.%I2.type]
// CHECK:STDOUT:   %T2.ref: %I2.assoc_type = name_ref T2, @T2.%assoc0 [concrete = constants.%assoc0.7c1]
// CHECK:STDOUT:   %U2: %I2.assoc_type = alias_binding U2, @T2.%assoc0 [concrete = constants.%assoc0.7c1]
// CHECK:STDOUT:   %J2.WithSelf.F2.decl: @J2.WithSelf.%J2.WithSelf.F2.type (%J2.WithSelf.F2.type) = fn_decl @J2.WithSelf.F2 [symbolic = @J2.WithSelf.%J2.WithSelf.F2 (constants.%J2.WithSelf.F2)] {
// CHECK:STDOUT:     %return.patt: @J2.WithSelf.F2.%pattern_type (%pattern_type.a14) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @J2.WithSelf.F2.%pattern_type (%pattern_type.a14) = out_param_pattern %return.patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.as_type: type = facet_access_type @J2.%Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:     %I2.facet.loc15_14.2: %I2.type = facet_value %Self.as_type, (constants.%I2.lookup_impl_witness.d97) [symbolic = %I2.facet.loc15_14.1 (constants.%I2.facet.7a5)]
// CHECK:STDOUT:     %.loc15_14.3: %I2.type = converted @J2.%Self, %I2.facet.loc15_14.2 [symbolic = %I2.facet.loc15_14.1 (constants.%I2.facet.7a5)]
// CHECK:STDOUT:     %impl.elem0.loc15_14.2: type = impl_witness_access constants.%I2.lookup_impl_witness.d97, element0 [symbolic = %impl.elem0.loc15_14.1 (constants.%impl.elem0.006)]
// CHECK:STDOUT:     %U2.ref: type = name_ref U2, %impl.elem0.loc15_14.2 [symbolic = %impl.elem0.loc15_14.1 (constants.%impl.elem0.006)]
// CHECK:STDOUT:     %.loc15_14.4: Core.Form = init_form %U2.ref, call_param0 [symbolic = %.loc15_14.2 (constants.%.d9a)]
// CHECK:STDOUT:     %return.param: ref @J2.WithSelf.F2.%impl.elem0.loc15_14.1 (%impl.elem0.006) = out_param call_param0
// CHECK:STDOUT:     %return: ref @J2.WithSelf.F2.%impl.elem0.loc15_14.1 (%impl.elem0.006) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %J2.assoc_type = assoc_entity element0, %J2.WithSelf.F2.decl [concrete = constants.%assoc0.41c]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .I2 = <poisoned>
// CHECK:STDOUT:   .I2 = <poisoned>
// CHECK:STDOUT:   .U2 = @J2.WithSelf.%U2
// CHECK:STDOUT:   .F2 = @J2.WithSelf.%assoc0
// CHECK:STDOUT:   witness = (@J2.WithSelf.%J2.WithSelf.F2.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @V.binding.as_type.as.I2.impl(%V.loc11_14.1: %J2.type) {
// CHECK:STDOUT:   %V.loc11_14.2: %J2.type = symbolic_binding V, 0 [symbolic = %V.loc11_14.2 (constants.%V)]
// CHECK:STDOUT:   %V.binding.as_type: type = symbolic_binding_type V, 0, %V.loc11_14.2 [symbolic = %V.binding.as_type (constants.%V.binding.as_type)]
// CHECK:STDOUT:   %I2.impl_witness.loc11_45.2: <witness> = impl_witness %I2.impl_witness_table, @V.binding.as_type.as.I2.impl(%V.loc11_14.2) [symbolic = %I2.impl_witness.loc11_45.2 (constants.%I2.impl_witness.364)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc11_22 as %.loc11_30 {
// CHECK:STDOUT:     %I2.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @V.binding.as_type.as.I2.impl [concrete]
// CHECK:STDOUT:     %I2.impl_witness.loc11_45.1: <witness> = impl_witness %I2.impl_witness_table, @V.binding.as_type.as.I2.impl(constants.%V) [symbolic = %I2.impl_witness.loc11_45.2 (constants.%I2.impl_witness.364)]
// CHECK:STDOUT:     %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = %I2.impl_witness.loc11_45.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @J2.WithSelf.F2(@J2.%Self: %J2.type) {
// CHECK:STDOUT:   %Self: %J2.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.4a7)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %.loc15_14.1: require_specific_def_type = require_specific_def @V.binding.as_type.as.I2.impl(%Self) [symbolic = %.loc15_14.1 (constants.%.ef7)]
// CHECK:STDOUT:   %I2.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @I2 [symbolic = %I2.lookup_impl_witness (constants.%I2.lookup_impl_witness.d97)]
// CHECK:STDOUT:   %I2.facet.loc15_14.1: %I2.type = facet_value %Self.binding.as_type, (%I2.lookup_impl_witness) [symbolic = %I2.facet.loc15_14.1 (constants.%I2.facet.7a5)]
// CHECK:STDOUT:   %impl.elem0.loc15_14.1: type = impl_witness_access %I2.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc15_14.1 (constants.%impl.elem0.006)]
// CHECK:STDOUT:   %.loc15_14.2: Core.Form = init_form %impl.elem0.loc15_14.1, call_param0 [symbolic = %.loc15_14.2 (constants.%.d9a)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %impl.elem0.loc15_14.1 [symbolic = %pattern_type (constants.%pattern_type.a14)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: @J2.WithSelf.F2.%impl.elem0.loc15_14.1 (%impl.elem0.006);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I2.WithSelf(constants.%Self.36e) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I2.WithSelf(constants.%.Self.feb) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @V.binding.as_type.as.I2.impl(constants.%V) {
// CHECK:STDOUT:   %V.loc11_14.2 => constants.%V
// CHECK:STDOUT:   %V.binding.as_type => constants.%V.binding.as_type
// CHECK:STDOUT:   %I2.impl_witness.loc11_45.2 => constants.%I2.impl_witness.364
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I2.WithSelf(constants.%I2.facet.beb) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J2.WithSelf(constants.%Self.4a7) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @V.binding.as_type.as.I2.impl(constants.%Self.4a7) {
// CHECK:STDOUT:   %V.loc11_14.2 => constants.%Self.4a7
// CHECK:STDOUT:   %V.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %I2.impl_witness.loc11_45.2 => constants.%I2.impl_witness.ddb
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I2.WithSelf(constants.%I2.facet.7a5) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J2.WithSelf.F2(constants.%Self.4a7) {
// CHECK:STDOUT:   %Self => constants.%Self.4a7
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %.loc15_14.1 => constants.%.ef7
// CHECK:STDOUT:   %I2.lookup_impl_witness => constants.%I2.lookup_impl_witness.d97
// CHECK:STDOUT:   %I2.facet.loc15_14.1 => constants.%I2.facet.7a5
// CHECK:STDOUT:   %impl.elem0.loc15_14.1 => constants.%impl.elem0.006
// CHECK:STDOUT:   %.loc15_14.2 => constants.%.d9a
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_call_method_alias.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = facet_type <@A> [concrete]
// CHECK:STDOUT:   %Self.c51: %A.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.55a: type = symbolic_binding_type Self, 0, %Self.c51 [symbolic]
// CHECK:STDOUT:   %pattern_type.df1: type = pattern_type %Self.binding.as_type.55a [symbolic]
// CHECK:STDOUT:   %.b38: Core.Form = init_form type, call_param1 [concrete]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %A.WithSelf.F.type: type = fn_type @A.WithSelf.F, @A.WithSelf(%Self.c51) [symbolic]
// CHECK:STDOUT:   %A.WithSelf.F: %A.WithSelf.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %A.assoc_type: type = assoc_entity_type @A [concrete]
// CHECK:STDOUT:   %assoc0.24a: %A.assoc_type = assoc_entity element0, @A.WithSelf.%A.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %B.type: type = facet_type <@B> [concrete]
// CHECK:STDOUT:   %Self.d0b: %B.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.8ee: type = symbolic_binding_type Self, 0, %Self.d0b [symbolic]
// CHECK:STDOUT:   %B.WithSelf.G.type: type = fn_type @B.WithSelf.G, @B.WithSelf(%Self.d0b) [symbolic]
// CHECK:STDOUT:   %B.WithSelf.G: %B.WithSelf.G.type = struct_value () [symbolic]
// CHECK:STDOUT:   %B.assoc_type: type = assoc_entity_type @B [concrete]
// CHECK:STDOUT:   %assoc0.864: %B.assoc_type = assoc_entity element0, @B.WithSelf.%B.WithSelf.G.decl [concrete]
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
// CHECK:STDOUT:   %Self: %A.type = symbolic_binding Self, 0 [symbolic = constants.%Self.c51]
// CHECK:STDOUT:   %A.WithSelf.decl = interface_with_self_decl @A [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %A.WithSelf.F.decl: @A.WithSelf.%A.WithSelf.F.type (%A.WithSelf.F.type) = fn_decl @A.WithSelf.F [symbolic = @A.WithSelf.%A.WithSelf.F (constants.%A.WithSelf.F)] {
// CHECK:STDOUT:     %self.patt: @A.WithSelf.F.%pattern_type (%pattern_type.df1) = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @A.WithSelf.F.%pattern_type (%pattern_type.df1) = value_param_pattern %self.patt [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.98f = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.98f = out_param_pattern %return.patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6_25.1: type = type_literal type [concrete = type]
// CHECK:STDOUT:     %.loc6_25.2: Core.Form = init_form %.loc6_25.1, call_param1 [concrete = constants.%.b38]
// CHECK:STDOUT:     %self.param: @A.WithSelf.F.%Self.binding.as_type (%Self.binding.as_type.55a) = value_param call_param0
// CHECK:STDOUT:     %.loc6_14.1: type = splice_block %.loc6_14.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.55a)] {
// CHECK:STDOUT:       %Self.ref: %A.type = name_ref Self, @A.%Self [symbolic = %Self (constants.%Self.c51)]
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.55a)]
// CHECK:STDOUT:       %.loc6_14.2: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.55a)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @A.WithSelf.F.%Self.binding.as_type (%Self.binding.as_type.55a) = value_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref type = out_param call_param1
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %A.assoc_type = assoc_entity element0, %A.WithSelf.F.decl [concrete = constants.%assoc0.24a]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = @A.WithSelf.%assoc0
// CHECK:STDOUT:   witness = (@A.WithSelf.%A.WithSelf.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @B {
// CHECK:STDOUT:   %Self: %B.type = symbolic_binding Self, 0 [symbolic = constants.%Self.d0b]
// CHECK:STDOUT:   %B.WithSelf.decl = interface_with_self_decl @B [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   %F.ref: %A.assoc_type = name_ref F, @A.WithSelf.%assoc0 [concrete = constants.%assoc0.24a]
// CHECK:STDOUT:   %F: %A.assoc_type = alias_binding F, @A.WithSelf.%assoc0 [concrete = constants.%assoc0.24a]
// CHECK:STDOUT:   %B.WithSelf.G.decl: @B.WithSelf.%B.WithSelf.G.type (%B.WithSelf.G.type) = fn_decl @B.WithSelf.G [symbolic = @B.WithSelf.%B.WithSelf.G (constants.%B.WithSelf.G)] {} {
// CHECK:STDOUT:     %Self.as_type: type = facet_access_type @B.%Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.8ee)]
// CHECK:STDOUT:     %F.ref: <error> = name_ref F, <error> [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %B.assoc_type = assoc_entity element0, %B.WithSelf.G.decl [concrete = constants.%assoc0.864]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT:   .F = @B.WithSelf.%F
// CHECK:STDOUT:   .G = @B.WithSelf.%assoc0
// CHECK:STDOUT:   witness = (@B.WithSelf.%B.WithSelf.G.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @A.WithSelf.F(@A.%Self: %A.type) {
// CHECK:STDOUT:   %Self: %A.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.c51)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.55a)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.df1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @A.WithSelf.F.%Self.binding.as_type (%Self.binding.as_type.55a)) -> out %return.param: type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @B.WithSelf.G(@B.%Self: %B.type) {
// CHECK:STDOUT:   %Self: %B.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.d0b)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.8ee)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> <error>;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%Self.c51) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.c51
// CHECK:STDOUT:   %A.WithSelf.F.type => constants.%A.WithSelf.F.type
// CHECK:STDOUT:   %A.WithSelf.F => constants.%A.WithSelf.F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf.F(constants.%Self.c51) {
// CHECK:STDOUT:   %Self => constants.%Self.c51
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.55a
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.df1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B.WithSelf(constants.%Self.d0b) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B.WithSelf.G(constants.%Self.d0b) {
// CHECK:STDOUT:   %Self => constants.%Self.d0b
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.8ee
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- call_with_compound_member_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C.type: type = facet_type <@C> [concrete]
// CHECK:STDOUT:   %Self: %C.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic]
// CHECK:STDOUT:   %C.WithSelf.F.type: type = fn_type @C.WithSelf.F, @C.WithSelf(%Self) [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C.WithSelf.F: %C.WithSelf.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %C.assoc_type: type = assoc_entity_type @C [concrete]
// CHECK:STDOUT:   %assoc0: %C.assoc_type = assoc_entity element0, @C.WithSelf.%C.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %C.WithSelf.G.type: type = fn_type @C.WithSelf.G, @C.WithSelf(%Self) [symbolic]
// CHECK:STDOUT:   %C.WithSelf.G: %C.WithSelf.G.type = struct_value () [symbolic]
// CHECK:STDOUT:   %assoc1: %C.assoc_type = assoc_entity element1, @C.WithSelf.%C.WithSelf.G.decl [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Self.binding.as_type [symbolic]
// CHECK:STDOUT:   %C.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @C [symbolic]
// CHECK:STDOUT:   %.018: type = fn_type_with_self_type %C.WithSelf.F.type, %Self [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.018 = impl_witness_access %C.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @C.WithSelf.F(%Self) [symbolic]
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
// CHECK:STDOUT:   %Self: %C.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %C.WithSelf.decl = interface_with_self_decl @C [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %C.WithSelf.F.decl: @C.WithSelf.%C.WithSelf.F.type (%C.WithSelf.F.type) = fn_decl @C.WithSelf.F [symbolic = @C.WithSelf.%C.WithSelf.F (constants.%C.WithSelf.F)] {
// CHECK:STDOUT:     %self.patt: @C.WithSelf.F.%pattern_type (%pattern_type) = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @C.WithSelf.F.%pattern_type (%pattern_type) = value_param_pattern %self.patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @C.WithSelf.F.%Self.binding.as_type (%Self.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc6_14.1: type = splice_block %.loc6_14.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)] {
// CHECK:STDOUT:       %Self.ref: %C.type = name_ref Self, @C.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:       %.loc6_14.2: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @C.WithSelf.F.%Self.binding.as_type (%Self.binding.as_type) = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %C.assoc_type = assoc_entity element0, %C.WithSelf.F.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:   %C.WithSelf.G.decl: @C.WithSelf.%C.WithSelf.G.type (%C.WithSelf.G.type) = fn_decl @C.WithSelf.G [symbolic = @C.WithSelf.%C.WithSelf.G (constants.%C.WithSelf.G)] {
// CHECK:STDOUT:     %self.patt: @C.WithSelf.G.%pattern_type (%pattern_type) = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @C.WithSelf.G.%pattern_type (%pattern_type) = value_param_pattern %self.patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @C.WithSelf.G.%Self.binding.as_type (%Self.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc8_14.1: type = splice_block %.loc8_14.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)] {
// CHECK:STDOUT:       %Self.ref: %C.type = name_ref Self, @C.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:       %.loc8_14.2: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @C.WithSelf.G.%Self.binding.as_type (%Self.binding.as_type) = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %C.assoc_type = assoc_entity element1, %C.WithSelf.G.decl [concrete = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = @C.WithSelf.%assoc0
// CHECK:STDOUT:   .G = @C.WithSelf.%assoc1
// CHECK:STDOUT:   witness = (@C.WithSelf.%C.WithSelf.F.decl, @C.WithSelf.%C.WithSelf.G.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C.WithSelf.F(@C.%Self: %C.type) {
// CHECK:STDOUT:   %Self: %C.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @C.WithSelf.F.%Self.binding.as_type (%Self.binding.as_type));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C.WithSelf.G(@C.%Self: %C.type) {
// CHECK:STDOUT:   %Self: %C.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Self.binding.as_type [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %C.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @C [symbolic = %C.lookup_impl_witness (constants.%C.lookup_impl_witness)]
// CHECK:STDOUT:   %C.WithSelf.F.type: type = fn_type @C.WithSelf.F, @C.WithSelf(%Self) [symbolic = %C.WithSelf.F.type (constants.%C.WithSelf.F.type)]
// CHECK:STDOUT:   %.loc9: type = fn_type_with_self_type %C.WithSelf.F.type, %Self [symbolic = %.loc9 (constants.%.018)]
// CHECK:STDOUT:   %impl.elem0.loc9_9.2: @C.WithSelf.G.%.loc9 (%.018) = impl_witness_access %C.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9_9.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc9_9.2: <specific function> = specific_impl_function %impl.elem0.loc9_9.2, @C.WithSelf.F(%Self) [symbolic = %specific_impl_fn.loc9_9.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @C.WithSelf.G.%Self.binding.as_type (%Self.binding.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref.loc9: @C.WithSelf.G.%Self.binding.as_type (%Self.binding.as_type) = name_ref self, %self
// CHECK:STDOUT:     %F.ref.loc9: %C.assoc_type = name_ref F, @C.WithSelf.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc9_9.1: @C.WithSelf.G.%.loc9 (%.018) = impl_witness_access constants.%C.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9_9.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc9_9: <bound method> = bound_method %self.ref.loc9, %impl.elem0.loc9_9.1
// CHECK:STDOUT:     %specific_impl_fn.loc9_9.1: <specific function> = specific_impl_function %impl.elem0.loc9_9.1, @C.WithSelf.F(constants.%Self) [symbolic = %specific_impl_fn.loc9_9.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %bound_method.loc9_12: <bound method> = bound_method %self.ref.loc9, %specific_impl_fn.loc9_9.1
// CHECK:STDOUT:     %C.WithSelf.F.call.loc9: init %empty_tuple.type = call %bound_method.loc9_12(%self.ref.loc9)
// CHECK:STDOUT:     %self.ref.loc10: @C.WithSelf.G.%Self.binding.as_type (%Self.binding.as_type) = name_ref self, %self
// CHECK:STDOUT:     %impl.elem0.loc10: @C.WithSelf.G.%.loc9 (%.018) = impl_witness_access constants.%C.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9_9.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %F.ref.loc10: @C.WithSelf.G.%.loc9 (%.018) = name_ref F, %impl.elem0.loc10 [symbolic = %impl.elem0.loc9_9.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc10_9: <bound method> = bound_method %self.ref.loc10, %F.ref.loc10
// CHECK:STDOUT:     %specific_impl_fn.loc10: <specific function> = specific_impl_function %F.ref.loc10, @C.WithSelf.F(constants.%Self) [symbolic = %specific_impl_fn.loc9_9.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %bound_method.loc10_14: <bound method> = bound_method %self.ref.loc10, %specific_impl_fn.loc10
// CHECK:STDOUT:     %C.WithSelf.F.call.loc10: init %empty_tuple.type = call %bound_method.loc10_14(%self.ref.loc10)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %C.WithSelf.F.type => constants.%C.WithSelf.F.type
// CHECK:STDOUT:   %C.WithSelf.F => constants.%C.WithSelf.F
// CHECK:STDOUT:   %C.WithSelf.G.type => constants.%C.WithSelf.G.type
// CHECK:STDOUT:   %C.WithSelf.G => constants.%C.WithSelf.G
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.WithSelf.F(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.WithSelf.G(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type
// CHECK:STDOUT: }
// CHECK:STDOUT:

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
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %Dest [symbolic]
// CHECK:STDOUT:   %ImplicitAs.Convert.type: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.Convert: %ImplicitAs.Convert.type = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %assoc0: %ImplicitAs.assoc_type = assoc_entity element0, @ImplicitAs.%ImplicitAs.Convert.decl [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %ImplicitAs.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.decl: %ImplicitAs.type.595 = interface_decl @ImplicitAs [concrete = constants.%ImplicitAs.generic] {
// CHECK:STDOUT:     %Dest.patt: %pattern_type.98f = symbolic_binding_pattern Dest, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
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
// CHECK:STDOUT:   %ImplicitAs.Convert.type: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%Dest.loc3_22.1) [symbolic = %ImplicitAs.Convert.type (constants.%ImplicitAs.Convert.type)]
// CHECK:STDOUT:   %ImplicitAs.Convert: @ImplicitAs.%ImplicitAs.Convert.type (%ImplicitAs.Convert.type) = struct_value () [symbolic = %ImplicitAs.Convert (constants.%ImplicitAs.Convert)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%Dest.loc3_22.1) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type)]
// CHECK:STDOUT:   %assoc0.loc4_35.2: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type) = assoc_entity element0, %ImplicitAs.Convert.decl [symbolic = %assoc0.loc4_35.2 (constants.%assoc0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc3_35.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.9fe) = symbolic_binding Self, 1 [symbolic = %Self.loc3_35.2 (constants.%Self)]
// CHECK:STDOUT:     %ImplicitAs.Convert.decl: @ImplicitAs.%ImplicitAs.Convert.type (%ImplicitAs.Convert.type) = fn_decl @ImplicitAs.Convert [symbolic = @ImplicitAs.%ImplicitAs.Convert (constants.%ImplicitAs.Convert)] {
// CHECK:STDOUT:       %self.patt: @ImplicitAs.Convert.%pattern_type.loc4_14 (%pattern_type.8de) = value_binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @ImplicitAs.Convert.%pattern_type.loc4_14 (%pattern_type.8de) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @ImplicitAs.Convert.%pattern_type.loc4_28 (%pattern_type.51d) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @ImplicitAs.Convert.%pattern_type.loc4_28 (%pattern_type.51d) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Dest.ref: type = name_ref Dest, @ImplicitAs.%Dest.loc3_22.2 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:       %self.param: @ImplicitAs.Convert.%Self.binding.as_type (%Self.binding.as_type) = value_param call_param0
// CHECK:STDOUT:       %.loc4_20.1: type = splice_block %.loc4_20.3 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)] {
// CHECK:STDOUT:         %.loc4_20.2: @ImplicitAs.Convert.%ImplicitAs.type (%ImplicitAs.type.9fe) = specific_constant @ImplicitAs.%Self.loc3_35.1, @ImplicitAs(constants.%Dest) [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:         %Self.ref: @ImplicitAs.Convert.%ImplicitAs.type (%ImplicitAs.type.9fe) = name_ref Self, %.loc4_20.2 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:         %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:         %.loc4_20.3: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @ImplicitAs.Convert.%Self.binding.as_type (%Self.binding.as_type) = value_binding self, %self.param
// CHECK:STDOUT:       %return.param: ref @ImplicitAs.Convert.%Dest (%Dest) = out_param call_param1
// CHECK:STDOUT:       %return: ref @ImplicitAs.Convert.%Dest (%Dest) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc4_35.1: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type) = assoc_entity element0, %ImplicitAs.Convert.decl [symbolic = %assoc0.loc4_35.2 (constants.%assoc0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc3_35.1
// CHECK:STDOUT:     .Dest = <poisoned>
// CHECK:STDOUT:     .Convert = %assoc0.loc4_35.1
// CHECK:STDOUT:     witness = (%ImplicitAs.Convert.decl)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ImplicitAs.Convert(@ImplicitAs.%Dest.loc3_22.2: type, @ImplicitAs.%Self.loc3_35.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.9fe)) {
// CHECK:STDOUT:   %Dest: type = symbolic_binding Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.9fe)]
// CHECK:STDOUT:   %Self: @ImplicitAs.Convert.%ImplicitAs.type (%ImplicitAs.type.9fe) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %pattern_type.loc4_14: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type.loc4_14 (constants.%pattern_type.8de)]
// CHECK:STDOUT:   %pattern_type.loc4_28: type = pattern_type %Dest [symbolic = %pattern_type.loc4_28 (constants.%pattern_type.51d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @ImplicitAs.Convert.%Self.binding.as_type (%Self.binding.as_type)) -> @ImplicitAs.Convert.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest.loc3_22.1 => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.Convert(constants.%Dest, constants.%Self) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.9fe
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %pattern_type.loc4_14 => constants.%pattern_type.8de
// CHECK:STDOUT:   %pattern_type.loc4_28 => constants.%pattern_type.51d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_alias_to_different_interface.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.ab9: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.e06: %I.assoc_type = assoc_entity element0, @I.%T [concrete]
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self.8a1: %J.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self.8a1 [symbolic]
// CHECK:STDOUT:   %J.F.type: type = fn_type @J.F [concrete]
// CHECK:STDOUT:   %J.F: %J.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type @J [concrete]
// CHECK:STDOUT:   %assoc0.5fd: %J.assoc_type = assoc_entity element0, @J.%J.F.decl [concrete]
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
// CHECK:STDOUT:   %T: type = assoc_const_decl @T [concrete] {
// CHECK:STDOUT:     %assoc0: %I.assoc_type = assoc_entity element0, @I.%T [concrete = constants.%assoc0.e06]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .T = @T.%assoc0
// CHECK:STDOUT:   witness = (%T)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = symbolic_binding Self, 0 [symbolic = constants.%Self.8a1]
// CHECK:STDOUT:   %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   %T.ref: %I.assoc_type = name_ref T, @T.%assoc0 [concrete = constants.%assoc0.e06]
// CHECK:STDOUT:   %U: %I.assoc_type = alias_binding U, @T.%assoc0 [concrete = constants.%assoc0.e06]
// CHECK:STDOUT:   %J.F.decl: %J.F.type = fn_decl @J.F [concrete = constants.%J.F] {
// CHECK:STDOUT:     %return.patt: <error> = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: <error> = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.as_type: type = facet_access_type @J.%Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:     %U.ref: <error> = name_ref U, <error> [concrete = <error>]
// CHECK:STDOUT:     %return.param: ref <error> = out_param call_param0
// CHECK:STDOUT:     %return: ref <error> = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %J.assoc_type = assoc_entity element0, %J.F.decl [concrete = constants.%assoc0.5fd]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .I = <poisoned>
// CHECK:STDOUT:   .U = %U
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%J.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @T(@I.%Self: %I.type) {
// CHECK:STDOUT:   assoc_const T:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @J.F(@J.%Self: %J.type) {
// CHECK:STDOUT:   %Self: %J.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.8a1)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> <error>;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T(constants.%Self.ab9) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.F(constants.%Self.8a1) {
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
// CHECK:STDOUT:   %assoc0.e09: %I2.assoc_type = assoc_entity element0, @I2.%T2 [concrete]
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
// CHECK:STDOUT:   %Self.4a7: %J2.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self.4a7 [symbolic]
// CHECK:STDOUT:   %I2.impl_witness.ddb: <witness> = impl_witness @V.binding.as_type.as.I2.impl.%I2.impl_witness_table, @V.binding.as_type.as.I2.impl(%Self.4a7) [symbolic]
// CHECK:STDOUT:   %.ef7: require_specific_def_type = require_specific_def @V.binding.as_type.as.I2.impl(%Self.4a7) [symbolic]
// CHECK:STDOUT:   %I2.lookup_impl_witness.d97: <witness> = lookup_impl_witness %Self.4a7, @I2 [symbolic]
// CHECK:STDOUT:   %I2.facet: %I2.type = facet_value %Self.binding.as_type, (%I2.lookup_impl_witness.d97) [symbolic]
// CHECK:STDOUT:   %impl.elem0.006: type = impl_witness_access %I2.lookup_impl_witness.d97, element0 [symbolic]
// CHECK:STDOUT:   %pattern_type.a14: type = pattern_type %impl.elem0.006 [symbolic]
// CHECK:STDOUT:   %J2.F2.type: type = fn_type @J2.F2 [concrete]
// CHECK:STDOUT:   %J2.F2: %J2.F2.type = struct_value () [concrete]
// CHECK:STDOUT:   %J2.assoc_type: type = assoc_entity_type @J2 [concrete]
// CHECK:STDOUT:   %assoc0.2a0: %J2.assoc_type = assoc_entity element0, @J2.%J2.F2.decl [concrete]
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
// CHECK:STDOUT:     %T2.ref: %I2.assoc_type = name_ref T2, @T2.%assoc0 [concrete = constants.%assoc0.e09]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc11_36: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type]
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
// CHECK:STDOUT:   %T2: type = assoc_const_decl @T2 [concrete] {
// CHECK:STDOUT:     %assoc0: %I2.assoc_type = assoc_entity element0, @I2.%T2 [concrete = constants.%assoc0.e09]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .T2 = @T2.%assoc0
// CHECK:STDOUT:   witness = (%T2)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J2 {
// CHECK:STDOUT:   %Self: %J2.type = symbolic_binding Self, 0 [symbolic = constants.%Self.4a7]
// CHECK:STDOUT:   %I2.ref: type = name_ref I2, file.%I2.decl [concrete = constants.%I2.type]
// CHECK:STDOUT:   %T2.ref: %I2.assoc_type = name_ref T2, @T2.%assoc0 [concrete = constants.%assoc0.e09]
// CHECK:STDOUT:   %U2: %I2.assoc_type = alias_binding U2, @T2.%assoc0 [concrete = constants.%assoc0.e09]
// CHECK:STDOUT:   %J2.F2.decl: %J2.F2.type = fn_decl @J2.F2 [concrete = constants.%J2.F2] {
// CHECK:STDOUT:     %return.patt: @J2.F2.%pattern_type (%pattern_type.a14) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @J2.F2.%pattern_type (%pattern_type.a14) = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.as_type: type = facet_access_type @J2.%Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:     %I2.facet.loc15_14.2: %I2.type = facet_value %Self.as_type, (constants.%I2.lookup_impl_witness.d97) [symbolic = %I2.facet.loc15_14.1 (constants.%I2.facet)]
// CHECK:STDOUT:     %.loc15_14.2: %I2.type = converted @J2.%Self, %I2.facet.loc15_14.2 [symbolic = %I2.facet.loc15_14.1 (constants.%I2.facet)]
// CHECK:STDOUT:     %impl.elem0.loc15_14.2: type = impl_witness_access constants.%I2.lookup_impl_witness.d97, element0 [symbolic = %impl.elem0.loc15_14.1 (constants.%impl.elem0.006)]
// CHECK:STDOUT:     %U2.ref: type = name_ref U2, %impl.elem0.loc15_14.2 [symbolic = %impl.elem0.loc15_14.1 (constants.%impl.elem0.006)]
// CHECK:STDOUT:     %return.param: ref @J2.F2.%impl.elem0.loc15_14.1 (%impl.elem0.006) = out_param call_param0
// CHECK:STDOUT:     %return: ref @J2.F2.%impl.elem0.loc15_14.1 (%impl.elem0.006) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %J2.assoc_type = assoc_entity element0, %J2.F2.decl [concrete = constants.%assoc0.2a0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .I2 = <poisoned>
// CHECK:STDOUT:   .U2 = %U2
// CHECK:STDOUT:   .F2 = %assoc0
// CHECK:STDOUT:   witness = (%J2.F2.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @T2(@I2.%Self: %I2.type) {
// CHECK:STDOUT:   assoc_const T2:! type;
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
// CHECK:STDOUT: generic fn @J2.F2(@J2.%Self: %J2.type) {
// CHECK:STDOUT:   %Self: %J2.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.4a7)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %.loc15_14.1: require_specific_def_type = require_specific_def @V.binding.as_type.as.I2.impl(%Self) [symbolic = %.loc15_14.1 (constants.%.ef7)]
// CHECK:STDOUT:   %I2.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @I2 [symbolic = %I2.lookup_impl_witness (constants.%I2.lookup_impl_witness.d97)]
// CHECK:STDOUT:   %I2.facet.loc15_14.1: %I2.type = facet_value %Self.binding.as_type, (%I2.lookup_impl_witness) [symbolic = %I2.facet.loc15_14.1 (constants.%I2.facet)]
// CHECK:STDOUT:   %impl.elem0.loc15_14.1: type = impl_witness_access %I2.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc15_14.1 (constants.%impl.elem0.006)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %impl.elem0.loc15_14.1 [symbolic = %pattern_type (constants.%pattern_type.a14)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> @J2.F2.%impl.elem0.loc15_14.1 (%impl.elem0.006);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T2(constants.%Self.36e) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T2(constants.%.Self.feb) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @V.binding.as_type.as.I2.impl(constants.%V) {
// CHECK:STDOUT:   %V.loc11_14.2 => constants.%V
// CHECK:STDOUT:   %V.binding.as_type => constants.%V.binding.as_type
// CHECK:STDOUT:   %I2.impl_witness.loc11_45.2 => constants.%I2.impl_witness.364
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @V.binding.as_type.as.I2.impl(constants.%Self.4a7) {
// CHECK:STDOUT:   %V.loc11_14.2 => constants.%Self.4a7
// CHECK:STDOUT:   %V.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %I2.impl_witness.loc11_45.2 => constants.%I2.impl_witness.ddb
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T2(constants.%I2.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J2.F2(constants.%Self.4a7) {
// CHECK:STDOUT:   %Self => constants.%Self.4a7
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %.loc15_14.1 => constants.%.ef7
// CHECK:STDOUT:   %I2.lookup_impl_witness => constants.%I2.lookup_impl_witness.d97
// CHECK:STDOUT:   %I2.facet.loc15_14.1 => constants.%I2.facet
// CHECK:STDOUT:   %impl.elem0.loc15_14.1 => constants.%impl.elem0.006
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
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %A.F.type: type = fn_type @A.F [concrete]
// CHECK:STDOUT:   %A.F: %A.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %A.assoc_type: type = assoc_entity_type @A [concrete]
// CHECK:STDOUT:   %assoc0.e44: %A.assoc_type = assoc_entity element0, @A.%A.F.decl [concrete]
// CHECK:STDOUT:   %B.type: type = facet_type <@B> [concrete]
// CHECK:STDOUT:   %Self.d0b: %B.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.8ee: type = symbolic_binding_type Self, 0, %Self.d0b [symbolic]
// CHECK:STDOUT:   %B.G.type: type = fn_type @B.G [concrete]
// CHECK:STDOUT:   %B.G: %B.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %B.assoc_type: type = assoc_entity_type @B [concrete]
// CHECK:STDOUT:   %assoc0.606: %B.assoc_type = assoc_entity element0, @B.%B.G.decl [concrete]
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
// CHECK:STDOUT:   %A.F.decl: %A.F.type = fn_decl @A.F [concrete = constants.%A.F] {
// CHECK:STDOUT:     %self.patt: @A.F.%pattern_type (%pattern_type.df1) = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @A.F.%pattern_type (%pattern_type.df1) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.98f = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.98f = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @A.F.%Self.binding.as_type (%Self.binding.as_type.55a) = value_param call_param0
// CHECK:STDOUT:     %.loc6_14.1: type = splice_block %.loc6_14.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.55a)] {
// CHECK:STDOUT:       %Self.ref: %A.type = name_ref Self, @A.%Self [symbolic = %Self (constants.%Self.c51)]
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.55a)]
// CHECK:STDOUT:       %.loc6_14.2: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.55a)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @A.F.%Self.binding.as_type (%Self.binding.as_type.55a) = value_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref type = out_param call_param1
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %A.assoc_type = assoc_entity element0, %A.F.decl [concrete = constants.%assoc0.e44]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%A.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @B {
// CHECK:STDOUT:   %Self: %B.type = symbolic_binding Self, 0 [symbolic = constants.%Self.d0b]
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   %F.ref: %A.assoc_type = name_ref F, @A.%assoc0 [concrete = constants.%assoc0.e44]
// CHECK:STDOUT:   %F: %A.assoc_type = alias_binding F, @A.%assoc0 [concrete = constants.%assoc0.e44]
// CHECK:STDOUT:   %B.G.decl: %B.G.type = fn_decl @B.G [concrete = constants.%B.G] {
// CHECK:STDOUT:     %return.patt: <error> = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: <error> = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.as_type: type = facet_access_type @B.%Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.8ee)]
// CHECK:STDOUT:     %F.ref: <error> = name_ref F, <error> [concrete = <error>]
// CHECK:STDOUT:     %return.param: ref <error> = out_param call_param0
// CHECK:STDOUT:     %return: ref <error> = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %B.assoc_type = assoc_entity element0, %B.G.decl [concrete = constants.%assoc0.606]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT:   .G = %assoc0
// CHECK:STDOUT:   witness = (%B.G.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @A.F(@A.%Self: %A.type) {
// CHECK:STDOUT:   %Self: %A.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.c51)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.55a)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.df1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @A.F.%Self.binding.as_type (%Self.binding.as_type.55a)) -> type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @B.G(@B.%Self: %B.type) {
// CHECK:STDOUT:   %Self: %B.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.d0b)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.8ee)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> <error>;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.F(constants.%Self.c51) {
// CHECK:STDOUT:   %Self => constants.%Self.c51
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.55a
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.df1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B.G(constants.%Self.d0b) {
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
// CHECK:STDOUT:   %C.F.type: type = fn_type @C.F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C.F: %C.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.assoc_type: type = assoc_entity_type @C [concrete]
// CHECK:STDOUT:   %assoc0: %C.assoc_type = assoc_entity element0, @C.%C.F.decl [concrete]
// CHECK:STDOUT:   %C.G.type: type = fn_type @C.G [concrete]
// CHECK:STDOUT:   %C.G: %C.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1: %C.assoc_type = assoc_entity element1, @C.%C.G.decl [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Self.binding.as_type [symbolic]
// CHECK:STDOUT:   %C.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @C [symbolic]
// CHECK:STDOUT:   %.52c: type = fn_type_with_self_type %C.F.type, %Self [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.52c = impl_witness_access %C.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @C.F(%Self) [symbolic]
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
// CHECK:STDOUT:   %C.F.decl: %C.F.type = fn_decl @C.F [concrete = constants.%C.F] {
// CHECK:STDOUT:     %self.patt: @C.F.%pattern_type (%pattern_type) = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @C.F.%pattern_type (%pattern_type) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @C.F.%Self.binding.as_type (%Self.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc6_14.1: type = splice_block %.loc6_14.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)] {
// CHECK:STDOUT:       %Self.ref: %C.type = name_ref Self, @C.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:       %.loc6_14.2: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @C.F.%Self.binding.as_type (%Self.binding.as_type) = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %C.assoc_type = assoc_entity element0, %C.F.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:   %C.G.decl: %C.G.type = fn_decl @C.G [concrete = constants.%C.G] {
// CHECK:STDOUT:     %self.patt: @C.G.%pattern_type (%pattern_type) = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @C.G.%pattern_type (%pattern_type) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @C.G.%Self.binding.as_type (%Self.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc8_14.1: type = splice_block %.loc8_14.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)] {
// CHECK:STDOUT:       %Self.ref: %C.type = name_ref Self, @C.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:       %.loc8_14.2: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @C.G.%Self.binding.as_type (%Self.binding.as_type) = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %C.assoc_type = assoc_entity element1, %C.G.decl [concrete = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   .G = %assoc1
// CHECK:STDOUT:   witness = (%C.F.decl, %C.G.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C.F(@C.%Self: %C.type) {
// CHECK:STDOUT:   %Self: %C.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @C.F.%Self.binding.as_type (%Self.binding.as_type));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C.G(@C.%Self: %C.type) {
// CHECK:STDOUT:   %Self: %C.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Self.binding.as_type [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %C.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @C [symbolic = %C.lookup_impl_witness (constants.%C.lookup_impl_witness)]
// CHECK:STDOUT:   %.loc9: type = fn_type_with_self_type constants.%C.F.type, %Self [symbolic = %.loc9 (constants.%.52c)]
// CHECK:STDOUT:   %impl.elem0.loc9_9.2: @C.G.%.loc9 (%.52c) = impl_witness_access %C.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9_9.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc9_9.2: <specific function> = specific_impl_function %impl.elem0.loc9_9.2, @C.F(%Self) [symbolic = %specific_impl_fn.loc9_9.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @C.G.%Self.binding.as_type (%Self.binding.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref.loc9: @C.G.%Self.binding.as_type (%Self.binding.as_type) = name_ref self, %self
// CHECK:STDOUT:     %F.ref.loc9: %C.assoc_type = name_ref F, @C.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc9_9.1: @C.G.%.loc9 (%.52c) = impl_witness_access constants.%C.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9_9.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc9_9: <bound method> = bound_method %self.ref.loc9, %impl.elem0.loc9_9.1
// CHECK:STDOUT:     %specific_impl_fn.loc9_9.1: <specific function> = specific_impl_function %impl.elem0.loc9_9.1, @C.F(constants.%Self) [symbolic = %specific_impl_fn.loc9_9.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %bound_method.loc9_12: <bound method> = bound_method %self.ref.loc9, %specific_impl_fn.loc9_9.1
// CHECK:STDOUT:     %C.F.call.loc9: init %empty_tuple.type = call %bound_method.loc9_12(%self.ref.loc9)
// CHECK:STDOUT:     %self.ref.loc10: @C.G.%Self.binding.as_type (%Self.binding.as_type) = name_ref self, %self
// CHECK:STDOUT:     %impl.elem0.loc10: @C.G.%.loc9 (%.52c) = impl_witness_access constants.%C.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9_9.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %F.ref.loc10: @C.G.%.loc9 (%.52c) = name_ref F, %impl.elem0.loc10 [symbolic = %impl.elem0.loc9_9.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc10_9: <bound method> = bound_method %self.ref.loc10, %F.ref.loc10
// CHECK:STDOUT:     %specific_impl_fn.loc10: <specific function> = specific_impl_function %F.ref.loc10, @C.F(constants.%Self) [symbolic = %specific_impl_fn.loc9_9.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %bound_method.loc10_14: <bound method> = bound_method %self.ref.loc10, %specific_impl_fn.loc10
// CHECK:STDOUT:     %C.F.call.loc10: init %empty_tuple.type = call %bound_method.loc10_14(%self.ref.loc10)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.F(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.G(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type
// CHECK:STDOUT: }
// CHECK:STDOUT:

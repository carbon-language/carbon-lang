// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
// INCLUDE-FILE: toolchain/testing/min_prelude/destroy.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/return/min_prelude/import_convert_function.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/return/min_prelude/import_convert_function.carbon

// --- core.carbon

// A minimized prelude intended to contain as few functions as possible.
package Core;

fn IntLiteral() -> type = "int_literal.make_type";
fn Int(N: IntLiteral()) -> type = "int.make_type_signed";

interface ImplicitAs(T:! type) {
  fn Convert[self: Self]() -> T;
}

impl IntLiteral() as ImplicitAs(i32) {
  fn Convert[self: Self]() -> i32 = "int.convert_checked";
}

// --- library.carbon

package P library "[[@TEST_NAME]]";

import Core;

class C(N:! i32) {}
class D { var n: i32; var m: i32; }
fn Make() -> D { return {.n = 0, .m = 0}; }

impl C(0) as Core.ImplicitAs(D) { fn Convert[self: Self]() -> D { return Make(); } }
impl C(1) as Core.ImplicitAs(D) { fn Convert[self: Self]() -> D { return Make(); } }
impl C(2) as Core.ImplicitAs(D) { fn Convert[self: Self]() -> D { return Make(); } }
impl C(3) as Core.ImplicitAs(D) { fn Convert[self: Self]() -> D { return Make(); } }
impl C(4) as Core.ImplicitAs(D) { fn Convert[self: Self]() -> D { return Make(); } }
impl C(5) as Core.ImplicitAs(D) { fn Convert[self: Self]() -> D { return Make(); } }
impl C(6) as Core.ImplicitAs(D) { fn Convert[self: Self]() -> D { return Make(); } }
impl C(7) as Core.ImplicitAs(D) { fn Convert[self: Self]() -> D { return Make(); } }

// --- user.carbon

import P library "library";
import Core;

// Check that we don't crash if we import functions and reallocate the function
// value store while handling a `return` statement.
fn F0(n: i32) -> P.D {
  if (false) { return {} as P.C(0); }
  if (false) { return {} as P.C(1); }
  if (false) { return {} as P.C(2); }
  if (false) { return {} as P.C(3); }
  if (false) { return {} as P.C(4); }
  if (false) { return {} as P.C(5); }
  if (false) { return {} as P.C(6); }
  if (false) { return {} as P.C(7); }
  return P.Make();
}

// CHECK:STDOUT: --- core.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [concrete]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.dc0: type = pattern_type Core.IntLiteral [concrete]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [concrete]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.96f: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.96f = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.07f: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic]
// CHECK:STDOUT:   %Self.0f3: %ImplicitAs.type.07f = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.0f3 [symbolic]
// CHECK:STDOUT:   %pattern_type.a93: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %Convert.type.4cf: type = fn_type @Convert.1, @ImplicitAs(%T) [symbolic]
// CHECK:STDOUT:   %Convert.147: %Convert.type.4cf = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.095: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.8f8: %ImplicitAs.assoc_type.095 = assoc_entity element0, @ImplicitAs.%Convert.decl [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.11a: type = facet_type <@ImplicitAs, @ImplicitAs(%i32.builtin)> [concrete]
// CHECK:STDOUT:   %Self.e0a: %ImplicitAs.type.11a = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.752: type = fn_type @Convert.1, @ImplicitAs(%i32.builtin) [concrete]
// CHECK:STDOUT:   %Convert.fcc: %Convert.type.752 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.1cf: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%i32.builtin) [concrete]
// CHECK:STDOUT:   %assoc0.3ce: %ImplicitAs.assoc_type.1cf = assoc_entity element0, @ImplicitAs.%Convert.decl [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness file.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.956: type = pattern_type %i32.builtin [concrete]
// CHECK:STDOUT:   %Convert.type.c2a: type = fn_type @Convert.2 [concrete]
// CHECK:STDOUT:   %Convert.40d: %Convert.type.c2a = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.11a = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .IntLiteral = %IntLiteral.decl
// CHECK:STDOUT:     .Int = %Int.decl
// CHECK:STDOUT:     .ImplicitAs = %ImplicitAs.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %IntLiteral.decl: %IntLiteral.type = fn_decl @IntLiteral [concrete = constants.%IntLiteral] {
// CHECK:STDOUT:     %return.patt: %pattern_type.98f = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.98f = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %return.param: ref type = out_param call_param0
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Int.decl: %Int.type = fn_decl @Int [concrete = constants.%Int] {
// CHECK:STDOUT:     %N.patt: %pattern_type.dc0 = binding_pattern N [concrete]
// CHECK:STDOUT:     %N.param_patt: %pattern_type.dc0 = value_param_pattern %N.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.98f = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.98f = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %N.param: Core.IntLiteral = value_param call_param0
// CHECK:STDOUT:     %.loc6_22.1: type = splice_block %.loc6_22.3 [concrete = Core.IntLiteral] {
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [concrete = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc6_22.2: type = value_of_initializer %int_literal.make_type [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc6_22.3: type = converted %int_literal.make_type, %.loc6_22.2 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N: Core.IntLiteral = bind_name N, %N.param
// CHECK:STDOUT:     %return.param: ref type = out_param call_param1
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.decl: %ImplicitAs.type.96f = interface_decl @ImplicitAs [concrete = constants.%ImplicitAs.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc8_22.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_22.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [concrete = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc12_17.1: type = value_of_initializer %int_literal.make_type [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc12_17.2: type = converted %int_literal.make_type, %.loc12_17.1 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.96f = name_ref ImplicitAs, file.%ImplicitAs.decl [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc12_36.1: type = value_of_initializer %int.make_type_signed [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc12_36.2: type = converted %int.make_type_signed, %.loc12_36.1 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%i32.builtin)> [concrete = constants.%ImplicitAs.type.11a]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (@impl.%Convert.decl), @impl [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness %ImplicitAs.impl_witness_table [concrete = constants.%ImplicitAs.impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(%T.loc8_22.1: type) {
// CHECK:STDOUT:   %T.loc8_22.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_22.2 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T.loc8_22.2)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.07f)]
// CHECK:STDOUT:   %Self.2: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.0f3)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.1, @ImplicitAs(%T.loc8_22.2) [symbolic = %Convert.type (constants.%Convert.type.4cf)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.4cf) = struct_value () [symbolic = %Convert (constants.%Convert.147)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%T.loc8_22.2) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type.095)]
// CHECK:STDOUT:   %assoc0.loc9_32.2: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.095) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc9_32.2 (constants.%assoc0.8f8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.0f3)]
// CHECK:STDOUT:     %Convert.decl: @ImplicitAs.%Convert.type (%Convert.type.4cf) = fn_decl @Convert.1 [symbolic = @ImplicitAs.%Convert (constants.%Convert.147)] {
// CHECK:STDOUT:       %self.patt: @Convert.1.%pattern_type.loc9_14 (%pattern_type.a93) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @Convert.1.%pattern_type.loc9_14 (%pattern_type.a93) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @Convert.1.%pattern_type.loc9_28 (%pattern_type.7dc) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @Convert.1.%pattern_type.loc9_28 (%pattern_type.7dc) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @ImplicitAs.%T.loc8_22.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self.param: @Convert.1.%Self.as_type.loc9_20.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:       %.loc9_20.1: type = splice_block %.loc9_20.3 [symbolic = %Self.as_type.loc9_20.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:         %.loc9_20.2: @Convert.1.%ImplicitAs.type (%ImplicitAs.type.07f) = specific_constant @ImplicitAs.%Self.1, @ImplicitAs(constants.%T) [symbolic = %Self (constants.%Self.0f3)]
// CHECK:STDOUT:         %Self.ref: @Convert.1.%ImplicitAs.type (%ImplicitAs.type.07f) = name_ref Self, %.loc9_20.2 [symbolic = %Self (constants.%Self.0f3)]
// CHECK:STDOUT:         %Self.as_type.loc9_20.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc9_20.1 (constants.%Self.as_type)]
// CHECK:STDOUT:         %.loc9_20.3: type = converted %Self.ref, %Self.as_type.loc9_20.2 [symbolic = %Self.as_type.loc9_20.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Convert.1.%Self.as_type.loc9_20.1 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @Convert.1.%T (%T) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Convert.1.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc9_32.1: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.095) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc9_32.2 (constants.%assoc0.8f8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .Convert = %assoc0.loc9_32.1
// CHECK:STDOUT:     witness = (%Convert.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %.loc12_17.2 as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.c2a = fn_decl @Convert.2 [concrete = constants.%Convert.40d] {
// CHECK:STDOUT:     %self.patt: %pattern_type.dc0 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.dc0 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.956 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.956 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc13_31.1: type = value_of_initializer %int.make_type_signed [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc13_31.2: type = converted %int.make_type_signed, %.loc13_31.1 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %self.param: Core.IntLiteral = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.%.loc12_17.2 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %self: Core.IntLiteral = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %i32.builtin = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32.builtin = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   witness = file.%ImplicitAs.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @IntLiteral() -> type = "int_literal.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int(%N.param: Core.IntLiteral) -> type = "int.make_type_signed";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.1(@ImplicitAs.%T.loc8_22.1: type, @ImplicitAs.%Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.07f)]
// CHECK:STDOUT:   %Self: @Convert.1.%ImplicitAs.type (%ImplicitAs.type.07f) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.0f3)]
// CHECK:STDOUT:   %Self.as_type.loc9_20.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc9_20.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type.loc9_14: type = pattern_type %Self.as_type.loc9_20.1 [symbolic = %pattern_type.loc9_14 (constants.%pattern_type.a93)]
// CHECK:STDOUT:   %pattern_type.loc9_28: type = pattern_type %T [symbolic = %pattern_type.loc9_28 (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Convert.1.%Self.as_type.loc9_20.1 (%Self.as_type)) -> @Convert.1.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.2(%self.param: Core.IntLiteral) -> %i32.builtin = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%T) {
// CHECK:STDOUT:   %T.loc8_22.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%T, constants.%Self.0f3) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.07f
// CHECK:STDOUT:   %Self => constants.%Self.0f3
// CHECK:STDOUT:   %Self.as_type.loc9_20.1 => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type.loc9_14 => constants.%pattern_type.a93
// CHECK:STDOUT:   %pattern_type.loc9_28 => constants.%pattern_type.7dc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%i32.builtin) {
// CHECK:STDOUT:   %T.loc8_22.2 => constants.%i32.builtin
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.11a
// CHECK:STDOUT:   %Self.2 => constants.%Self.e0a
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.752
// CHECK:STDOUT:   %Convert => constants.%Convert.fcc
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.1cf
// CHECK:STDOUT:   %assoc0.loc9_32.2 => constants.%assoc0.3ce
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%i32.builtin, constants.%ImplicitAs.facet) {
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.11a
// CHECK:STDOUT:   %Self => constants.%ImplicitAs.facet
// CHECK:STDOUT:   %Self.as_type.loc9_20.1 => Core.IntLiteral
// CHECK:STDOUT:   %pattern_type.loc9_14 => constants.%pattern_type.dc0
// CHECK:STDOUT:   %pattern_type.loc9_28 => constants.%pattern_type.956
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- library.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [concrete]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %N: %i32.builtin = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.956: type = pattern_type %i32.builtin [concrete]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.116: type = class_type @C, @C(%N) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %D.elem: type = unbound_element_type %D, %i32.builtin [concrete]
// CHECK:STDOUT:   %struct_type.n.m.566: type = struct_type {.n: %i32.builtin, .m: %i32.builtin} [concrete]
// CHECK:STDOUT:   %complete_type.682: <witness> = complete_type_witness %struct_type.n.m.566 [concrete]
// CHECK:STDOUT:   %pattern_type.510: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %struct_type.n.m.819: type = struct_type {.n: Core.IntLiteral, .m: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.d62: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic]
// CHECK:STDOUT:   %Self.519: %ImplicitAs.type.d62 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.275: type = fn_type @Convert.1, @ImplicitAs(%T) [symbolic]
// CHECK:STDOUT:   %Convert.42e: %Convert.type.275 = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.519 [symbolic]
// CHECK:STDOUT:   %pattern_type.f3e: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.ca0: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.9f5: %ImplicitAs.assoc_type.ca0 = assoc_entity element0, imports.%Core.import_ref.1c7 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.61e: type = facet_type <@ImplicitAs, @ImplicitAs(%i32.builtin)> [concrete]
// CHECK:STDOUT:   %Self.d28: %ImplicitAs.type.61e = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.059: type = fn_type @Convert.1, @ImplicitAs(%i32.builtin) [concrete]
// CHECK:STDOUT:   %Convert.4d7: %Convert.type.059 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.398: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%i32.builtin) [concrete]
// CHECK:STDOUT:   %assoc0.6fd: %ImplicitAs.assoc_type.398 = assoc_entity element0, imports.%Core.import_ref.1c7 [concrete]
// CHECK:STDOUT:   %assoc0.dc0: %ImplicitAs.assoc_type.ca0 = assoc_entity element0, imports.%Core.import_ref.207 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.07a: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.a3d: %ImplicitAs.type.61e = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.07a) [concrete]
// CHECK:STDOUT:   %.ec1: type = fn_type_with_self_type %Convert.type.059, %ImplicitAs.facet.a3d [concrete]
// CHECK:STDOUT:   %Convert.type.49f: type = fn_type @Convert.2 [concrete]
// CHECK:STDOUT:   %Convert.cb5: %Convert.type.49f = struct_value () [concrete]
// CHECK:STDOUT:   %Convert.bound.04a: <bound method> = bound_method %int_0.5c6, %Convert.cb5 [concrete]
// CHECK:STDOUT:   %int_0.a54: %i32.builtin = int_value 0 [concrete]
// CHECK:STDOUT:   %D.val: %D = struct_value (%int_0.a54, %int_0.a54) [concrete]
// CHECK:STDOUT:   %C.808: type = class_type @C, @C(%int_0.a54) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.94e: type = facet_type <@ImplicitAs, @ImplicitAs(%D)> [concrete]
// CHECK:STDOUT:   %Self.371: %ImplicitAs.type.94e = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.010: type = fn_type @Convert.1, @ImplicitAs(%D) [concrete]
// CHECK:STDOUT:   %Convert.d38: %Convert.type.010 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.75e: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%D) [concrete]
// CHECK:STDOUT:   %assoc0.579: %ImplicitAs.assoc_type.75e = assoc_entity element0, imports.%Core.import_ref.1c7 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.4af: <witness> = impl_witness file.%ImplicitAs.impl_witness_table.loc10 [concrete]
// CHECK:STDOUT:   %pattern_type.658: type = pattern_type %C.808 [concrete]
// CHECK:STDOUT:   %Convert.type.fff: type = fn_type @Convert.3 [concrete]
// CHECK:STDOUT:   %Convert.606: %Convert.type.fff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.a19: %ImplicitAs.type.94e = facet_value %C.808, (%ImplicitAs.impl_witness.4af) [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Convert.bound.92d: <bound method> = bound_method %int_1.5b8, %Convert.cb5 [concrete]
// CHECK:STDOUT:   %int_1.f38: %i32.builtin = int_value 1 [concrete]
// CHECK:STDOUT:   %C.8be: type = class_type @C, @C(%int_1.f38) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.fcc: <witness> = impl_witness file.%ImplicitAs.impl_witness_table.loc11 [concrete]
// CHECK:STDOUT:   %pattern_type.a77: type = pattern_type %C.8be [concrete]
// CHECK:STDOUT:   %Convert.type.89f: type = fn_type @Convert.4 [concrete]
// CHECK:STDOUT:   %Convert.689: %Convert.type.89f = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.340: %ImplicitAs.type.94e = facet_value %C.8be, (%ImplicitAs.impl_witness.fcc) [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %Convert.bound.1b9: <bound method> = bound_method %int_2.ecc, %Convert.cb5 [concrete]
// CHECK:STDOUT:   %int_2.5a1: %i32.builtin = int_value 2 [concrete]
// CHECK:STDOUT:   %C.c17: type = class_type @C, @C(%int_2.5a1) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.03e: <witness> = impl_witness file.%ImplicitAs.impl_witness_table.loc12 [concrete]
// CHECK:STDOUT:   %pattern_type.58e: type = pattern_type %C.c17 [concrete]
// CHECK:STDOUT:   %Convert.type.e90: type = fn_type @Convert.5 [concrete]
// CHECK:STDOUT:   %Convert.ec9: %Convert.type.e90 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.ec0: %ImplicitAs.type.94e = facet_value %C.c17, (%ImplicitAs.impl_witness.03e) [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %Convert.bound.b6b: <bound method> = bound_method %int_3.1ba, %Convert.cb5 [concrete]
// CHECK:STDOUT:   %int_3.a0f: %i32.builtin = int_value 3 [concrete]
// CHECK:STDOUT:   %C.414: type = class_type @C, @C(%int_3.a0f) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.dec: <witness> = impl_witness file.%ImplicitAs.impl_witness_table.loc13 [concrete]
// CHECK:STDOUT:   %pattern_type.861: type = pattern_type %C.414 [concrete]
// CHECK:STDOUT:   %Convert.type.5db: type = fn_type @Convert.6 [concrete]
// CHECK:STDOUT:   %Convert.193: %Convert.type.5db = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.999: %ImplicitAs.type.94e = facet_value %C.414, (%ImplicitAs.impl_witness.dec) [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %Convert.bound.626: <bound method> = bound_method %int_4.0c1, %Convert.cb5 [concrete]
// CHECK:STDOUT:   %int_4.4f1: %i32.builtin = int_value 4 [concrete]
// CHECK:STDOUT:   %C.488: type = class_type @C, @C(%int_4.4f1) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.842: <witness> = impl_witness file.%ImplicitAs.impl_witness_table.loc14 [concrete]
// CHECK:STDOUT:   %pattern_type.d35: type = pattern_type %C.488 [concrete]
// CHECK:STDOUT:   %Convert.type.4e7: type = fn_type @Convert.7 [concrete]
// CHECK:STDOUT:   %Convert.52a: %Convert.type.4e7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.527: %ImplicitAs.type.94e = facet_value %C.488, (%ImplicitAs.impl_witness.842) [concrete]
// CHECK:STDOUT:   %int_5.64b: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %Convert.bound.910: <bound method> = bound_method %int_5.64b, %Convert.cb5 [concrete]
// CHECK:STDOUT:   %int_5.967: %i32.builtin = int_value 5 [concrete]
// CHECK:STDOUT:   %C.3e2: type = class_type @C, @C(%int_5.967) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.eb5: <witness> = impl_witness file.%ImplicitAs.impl_witness_table.loc15 [concrete]
// CHECK:STDOUT:   %pattern_type.c88: type = pattern_type %C.3e2 [concrete]
// CHECK:STDOUT:   %Convert.type.658: type = fn_type @Convert.8 [concrete]
// CHECK:STDOUT:   %Convert.9b6: %Convert.type.658 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.79a: %ImplicitAs.type.94e = facet_value %C.3e2, (%ImplicitAs.impl_witness.eb5) [concrete]
// CHECK:STDOUT:   %int_6.462: Core.IntLiteral = int_value 6 [concrete]
// CHECK:STDOUT:   %Convert.bound.e3a: <bound method> = bound_method %int_6.462, %Convert.cb5 [concrete]
// CHECK:STDOUT:   %int_6.ec5: %i32.builtin = int_value 6 [concrete]
// CHECK:STDOUT:   %C.78c: type = class_type @C, @C(%int_6.ec5) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.377: <witness> = impl_witness file.%ImplicitAs.impl_witness_table.loc16 [concrete]
// CHECK:STDOUT:   %pattern_type.af2: type = pattern_type %C.78c [concrete]
// CHECK:STDOUT:   %Convert.type.623: type = fn_type @Convert.9 [concrete]
// CHECK:STDOUT:   %Convert.8c9: %Convert.type.623 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.5ba: %ImplicitAs.type.94e = facet_value %C.78c, (%ImplicitAs.impl_witness.377) [concrete]
// CHECK:STDOUT:   %int_7.29f: Core.IntLiteral = int_value 7 [concrete]
// CHECK:STDOUT:   %Convert.bound.06a: <bound method> = bound_method %int_7.29f, %Convert.cb5 [concrete]
// CHECK:STDOUT:   %int_7.6ae: %i32.builtin = int_value 7 [concrete]
// CHECK:STDOUT:   %C.6aa: type = class_type @C, @C(%int_7.6ae) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.308: <witness> = impl_witness file.%ImplicitAs.impl_witness_table.loc17 [concrete]
// CHECK:STDOUT:   %pattern_type.045: type = pattern_type %C.6aa [concrete]
// CHECK:STDOUT:   %Convert.type.3f8: type = fn_type @Convert.10 [concrete]
// CHECK:STDOUT:   %Convert.71a: %Convert.type.3f8 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.b5c: %ImplicitAs.type.94e = facet_value %C.6aa, (%ImplicitAs.impl_witness.308) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//default, Int, loaded [concrete = constants.%Int]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//default, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.1: type = import_ref Core//default, loc8_22, loaded [symbolic = @ImplicitAs.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.ff5 = import_ref Core//default, inst49 [no loc], unloaded
// CHECK:STDOUT:   %Core.import_ref.492: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.ca0) = import_ref Core//default, loc9_32, loaded [symbolic = @ImplicitAs.%assoc0 (constants.%assoc0.dc0)]
// CHECK:STDOUT:   %Core.Convert: @ImplicitAs.%Convert.type (%Convert.type.275) = import_ref Core//default, Convert, loaded [symbolic = @ImplicitAs.%Convert (constants.%Convert.42e)]
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.2: type = import_ref Core//default, loc8_22, loaded [symbolic = @ImplicitAs.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.ce1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62) = import_ref Core//default, inst49 [no loc], loaded [symbolic = @ImplicitAs.%Self (constants.%Self.519)]
// CHECK:STDOUT:   %Core.import_ref.1c7: @ImplicitAs.%Convert.type (%Convert.type.275) = import_ref Core//default, loc9_32, loaded [symbolic = @ImplicitAs.%Convert (constants.%Convert.42e)]
// CHECK:STDOUT:   %Core.import_ref.207 = import_ref Core//default, loc9_32, unloaded
// CHECK:STDOUT:   %Core.import_ref.c62: <witness> = import_ref Core//default, loc12_38, loaded [concrete = constants.%ImplicitAs.impl_witness.07a]
// CHECK:STDOUT:   %Core.import_ref.872: type = import_ref Core//default, loc12_17, loaded [concrete = Core.IntLiteral]
// CHECK:STDOUT:   %Core.import_ref.4d9: type = import_ref Core//default, loc12_36, loaded [concrete = constants.%ImplicitAs.type.61e]
// CHECK:STDOUT:   %Core.import_ref.f35: %Convert.type.49f = import_ref Core//default, loc13_35, loaded [concrete = constants.%Convert.cb5]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%Core.import_ref.f35), @impl.68b [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .Make = %Make.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [concrete = constants.%C.generic] {
// CHECK:STDOUT:     %N.patt: %pattern_type.956 = symbolic_binding_pattern N, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6_13.1: type = splice_block %.loc6_13.3 [concrete = constants.%i32.builtin] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %int.make_type_signed: init type = call constants.%Int(%int_32) [concrete = constants.%i32.builtin]
// CHECK:STDOUT:       %.loc6_13.2: type = value_of_initializer %int.make_type_signed [concrete = constants.%i32.builtin]
// CHECK:STDOUT:       %.loc6_13.3: type = converted %int.make_type_signed, %.loc6_13.2 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc6_9.1: %i32.builtin = bind_symbolic_name N, 0 [symbolic = %N.loc6_9.2 (constants.%N)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl: type = class_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   %Make.decl: %Make.type = fn_decl @Make [concrete = constants.%Make] {
// CHECK:STDOUT:     %return.patt: %pattern_type.510 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.510 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param0
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.837 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:     %impl.elem0: %.ec1 = impl_witness_access constants.%ImplicitAs.impl_witness.07a, element0 [concrete = constants.%Convert.cb5]
// CHECK:STDOUT:     %bound_method: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Convert.bound.04a]
// CHECK:STDOUT:     %int.convert_checked: init %i32.builtin = call %bound_method(%int_0) [concrete = constants.%int_0.a54]
// CHECK:STDOUT:     %.loc10_9.1: %i32.builtin = value_of_initializer %int.convert_checked [concrete = constants.%int_0.a54]
// CHECK:STDOUT:     %.loc10_9.2: %i32.builtin = converted %int_0, %.loc10_9.1 [concrete = constants.%int_0.a54]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_0.a54) [concrete = constants.%C.808]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.94e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.loc10 = impl_witness_table (@impl.837.%Convert.decl), @impl.837 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.loc10: <witness> = impl_witness %ImplicitAs.impl_witness_table.loc10 [concrete = constants.%ImplicitAs.impl_witness.4af]
// CHECK:STDOUT:   impl_decl @impl.511 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %impl.elem0: %.ec1 = impl_witness_access constants.%ImplicitAs.impl_witness.07a, element0 [concrete = constants.%Convert.cb5]
// CHECK:STDOUT:     %bound_method: <bound method> = bound_method %int_1, %impl.elem0 [concrete = constants.%Convert.bound.92d]
// CHECK:STDOUT:     %int.convert_checked: init %i32.builtin = call %bound_method(%int_1) [concrete = constants.%int_1.f38]
// CHECK:STDOUT:     %.loc11_9.1: %i32.builtin = value_of_initializer %int.convert_checked [concrete = constants.%int_1.f38]
// CHECK:STDOUT:     %.loc11_9.2: %i32.builtin = converted %int_1, %.loc11_9.1 [concrete = constants.%int_1.f38]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_1.f38) [concrete = constants.%C.8be]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.94e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.loc11 = impl_witness_table (@impl.511.%Convert.decl), @impl.511 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.loc11: <witness> = impl_witness %ImplicitAs.impl_witness_table.loc11 [concrete = constants.%ImplicitAs.impl_witness.fcc]
// CHECK:STDOUT:   impl_decl @impl.141 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %impl.elem0: %.ec1 = impl_witness_access constants.%ImplicitAs.impl_witness.07a, element0 [concrete = constants.%Convert.cb5]
// CHECK:STDOUT:     %bound_method: <bound method> = bound_method %int_2, %impl.elem0 [concrete = constants.%Convert.bound.1b9]
// CHECK:STDOUT:     %int.convert_checked: init %i32.builtin = call %bound_method(%int_2) [concrete = constants.%int_2.5a1]
// CHECK:STDOUT:     %.loc12_9.1: %i32.builtin = value_of_initializer %int.convert_checked [concrete = constants.%int_2.5a1]
// CHECK:STDOUT:     %.loc12_9.2: %i32.builtin = converted %int_2, %.loc12_9.1 [concrete = constants.%int_2.5a1]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_2.5a1) [concrete = constants.%C.c17]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.94e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.loc12 = impl_witness_table (@impl.141.%Convert.decl), @impl.141 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.loc12: <witness> = impl_witness %ImplicitAs.impl_witness_table.loc12 [concrete = constants.%ImplicitAs.impl_witness.03e]
// CHECK:STDOUT:   impl_decl @impl.e6b [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %impl.elem0: %.ec1 = impl_witness_access constants.%ImplicitAs.impl_witness.07a, element0 [concrete = constants.%Convert.cb5]
// CHECK:STDOUT:     %bound_method: <bound method> = bound_method %int_3, %impl.elem0 [concrete = constants.%Convert.bound.b6b]
// CHECK:STDOUT:     %int.convert_checked: init %i32.builtin = call %bound_method(%int_3) [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:     %.loc13_9.1: %i32.builtin = value_of_initializer %int.convert_checked [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:     %.loc13_9.2: %i32.builtin = converted %int_3, %.loc13_9.1 [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_3.a0f) [concrete = constants.%C.414]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.94e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.loc13 = impl_witness_table (@impl.e6b.%Convert.decl), @impl.e6b [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.loc13: <witness> = impl_witness %ImplicitAs.impl_witness_table.loc13 [concrete = constants.%ImplicitAs.impl_witness.dec]
// CHECK:STDOUT:   impl_decl @impl.056 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:     %impl.elem0: %.ec1 = impl_witness_access constants.%ImplicitAs.impl_witness.07a, element0 [concrete = constants.%Convert.cb5]
// CHECK:STDOUT:     %bound_method: <bound method> = bound_method %int_4, %impl.elem0 [concrete = constants.%Convert.bound.626]
// CHECK:STDOUT:     %int.convert_checked: init %i32.builtin = call %bound_method(%int_4) [concrete = constants.%int_4.4f1]
// CHECK:STDOUT:     %.loc14_9.1: %i32.builtin = value_of_initializer %int.convert_checked [concrete = constants.%int_4.4f1]
// CHECK:STDOUT:     %.loc14_9.2: %i32.builtin = converted %int_4, %.loc14_9.1 [concrete = constants.%int_4.4f1]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_4.4f1) [concrete = constants.%C.488]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.94e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.loc14 = impl_witness_table (@impl.056.%Convert.decl), @impl.056 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.loc14: <witness> = impl_witness %ImplicitAs.impl_witness_table.loc14 [concrete = constants.%ImplicitAs.impl_witness.842]
// CHECK:STDOUT:   impl_decl @impl.154 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5.64b]
// CHECK:STDOUT:     %impl.elem0: %.ec1 = impl_witness_access constants.%ImplicitAs.impl_witness.07a, element0 [concrete = constants.%Convert.cb5]
// CHECK:STDOUT:     %bound_method: <bound method> = bound_method %int_5, %impl.elem0 [concrete = constants.%Convert.bound.910]
// CHECK:STDOUT:     %int.convert_checked: init %i32.builtin = call %bound_method(%int_5) [concrete = constants.%int_5.967]
// CHECK:STDOUT:     %.loc15_9.1: %i32.builtin = value_of_initializer %int.convert_checked [concrete = constants.%int_5.967]
// CHECK:STDOUT:     %.loc15_9.2: %i32.builtin = converted %int_5, %.loc15_9.1 [concrete = constants.%int_5.967]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_5.967) [concrete = constants.%C.3e2]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.94e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.loc15 = impl_witness_table (@impl.154.%Convert.decl), @impl.154 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.loc15: <witness> = impl_witness %ImplicitAs.impl_witness_table.loc15 [concrete = constants.%ImplicitAs.impl_witness.eb5]
// CHECK:STDOUT:   impl_decl @impl.a6d [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_6: Core.IntLiteral = int_value 6 [concrete = constants.%int_6.462]
// CHECK:STDOUT:     %impl.elem0: %.ec1 = impl_witness_access constants.%ImplicitAs.impl_witness.07a, element0 [concrete = constants.%Convert.cb5]
// CHECK:STDOUT:     %bound_method: <bound method> = bound_method %int_6, %impl.elem0 [concrete = constants.%Convert.bound.e3a]
// CHECK:STDOUT:     %int.convert_checked: init %i32.builtin = call %bound_method(%int_6) [concrete = constants.%int_6.ec5]
// CHECK:STDOUT:     %.loc16_9.1: %i32.builtin = value_of_initializer %int.convert_checked [concrete = constants.%int_6.ec5]
// CHECK:STDOUT:     %.loc16_9.2: %i32.builtin = converted %int_6, %.loc16_9.1 [concrete = constants.%int_6.ec5]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_6.ec5) [concrete = constants.%C.78c]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.94e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.loc16 = impl_witness_table (@impl.a6d.%Convert.decl), @impl.a6d [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.loc16: <witness> = impl_witness %ImplicitAs.impl_witness_table.loc16 [concrete = constants.%ImplicitAs.impl_witness.377]
// CHECK:STDOUT:   impl_decl @impl.ebc [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_7: Core.IntLiteral = int_value 7 [concrete = constants.%int_7.29f]
// CHECK:STDOUT:     %impl.elem0: %.ec1 = impl_witness_access constants.%ImplicitAs.impl_witness.07a, element0 [concrete = constants.%Convert.cb5]
// CHECK:STDOUT:     %bound_method: <bound method> = bound_method %int_7, %impl.elem0 [concrete = constants.%Convert.bound.06a]
// CHECK:STDOUT:     %int.convert_checked: init %i32.builtin = call %bound_method(%int_7) [concrete = constants.%int_7.6ae]
// CHECK:STDOUT:     %.loc17_9.1: %i32.builtin = value_of_initializer %int.convert_checked [concrete = constants.%int_7.6ae]
// CHECK:STDOUT:     %.loc17_9.2: %i32.builtin = converted %int_7, %.loc17_9.1 [concrete = constants.%int_7.6ae]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_7.6ae) [concrete = constants.%C.6aa]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.94e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.loc17 = impl_witness_table (@impl.ebc.%Convert.decl), @impl.ebc [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.loc17: <witness> = impl_witness %ImplicitAs.impl_witness_table.loc17 [concrete = constants.%ImplicitAs.impl_witness.308]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(imports.%Core.import_ref.5ab3ec.1: type) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.d62)]
// CHECK:STDOUT:   %Self: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.519)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.1, @ImplicitAs(%T) [symbolic = %Convert.type (constants.%Convert.type.275)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.275) = struct_value () [symbolic = %Convert (constants.%Convert.42e)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%T) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type.ca0)]
// CHECK:STDOUT:   %assoc0: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.ca0) = assoc_entity element0, imports.%Core.import_ref.1c7 [symbolic = %assoc0 (constants.%assoc0.9f5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Core.import_ref.ff5
// CHECK:STDOUT:     .Convert = imports.%Core.import_ref.492
// CHECK:STDOUT:     witness = (imports.%Core.Convert)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.68b: imports.%Core.import_ref.872 as imports.%Core.import_ref.4d9 [from "core.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Core.import_ref.c62
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.837: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.fff = fn_decl @Convert.3 [concrete = constants.%Convert.606] {
// CHECK:STDOUT:     %self.patt: %pattern_type.658 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.658 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.510 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.510 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %self.param: %C.808 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.837.%C [concrete = constants.%C.808]
// CHECK:STDOUT:     %self: %C.808 = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   witness = file.%ImplicitAs.impl_witness.loc10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.511: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.89f = fn_decl @Convert.4 [concrete = constants.%Convert.689] {
// CHECK:STDOUT:     %self.patt: %pattern_type.a77 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a77 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.510 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.510 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %self.param: %C.8be = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.511.%C [concrete = constants.%C.8be]
// CHECK:STDOUT:     %self: %C.8be = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   witness = file.%ImplicitAs.impl_witness.loc11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.141: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.e90 = fn_decl @Convert.5 [concrete = constants.%Convert.ec9] {
// CHECK:STDOUT:     %self.patt: %pattern_type.58e = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.58e = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.510 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.510 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %self.param: %C.c17 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.141.%C [concrete = constants.%C.c17]
// CHECK:STDOUT:     %self: %C.c17 = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   witness = file.%ImplicitAs.impl_witness.loc12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.e6b: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.5db = fn_decl @Convert.6 [concrete = constants.%Convert.193] {
// CHECK:STDOUT:     %self.patt: %pattern_type.861 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.861 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.510 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.510 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %self.param: %C.414 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.e6b.%C [concrete = constants.%C.414]
// CHECK:STDOUT:     %self: %C.414 = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   witness = file.%ImplicitAs.impl_witness.loc13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.056: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.4e7 = fn_decl @Convert.7 [concrete = constants.%Convert.52a] {
// CHECK:STDOUT:     %self.patt: %pattern_type.d35 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.d35 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.510 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.510 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %self.param: %C.488 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.056.%C [concrete = constants.%C.488]
// CHECK:STDOUT:     %self: %C.488 = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   witness = file.%ImplicitAs.impl_witness.loc14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.154: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.658 = fn_decl @Convert.8 [concrete = constants.%Convert.9b6] {
// CHECK:STDOUT:     %self.patt: %pattern_type.c88 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.c88 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.510 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.510 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %self.param: %C.3e2 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.154.%C [concrete = constants.%C.3e2]
// CHECK:STDOUT:     %self: %C.3e2 = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   witness = file.%ImplicitAs.impl_witness.loc15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.a6d: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.623 = fn_decl @Convert.9 [concrete = constants.%Convert.8c9] {
// CHECK:STDOUT:     %self.patt: %pattern_type.af2 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.af2 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.510 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.510 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %self.param: %C.78c = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.a6d.%C [concrete = constants.%C.78c]
// CHECK:STDOUT:     %self: %C.78c = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   witness = file.%ImplicitAs.impl_witness.loc16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.ebc: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.3f8 = fn_decl @Convert.10 [concrete = constants.%Convert.71a] {
// CHECK:STDOUT:     %self.patt: %pattern_type.045 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.045 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.510 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.510 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %self.param: %C.6aa = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.ebc.%C [concrete = constants.%C.6aa]
// CHECK:STDOUT:     %self: %C.6aa = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   witness = file.%ImplicitAs.impl_witness.loc17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%N.loc6_9.1: %i32.builtin) {
// CHECK:STDOUT:   %N.loc6_9.2: %i32.builtin = bind_symbolic_name N, 0 [symbolic = %N.loc6_9.2 (constants.%N)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.116
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %int_32.loc7_18: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc7_18: init type = call constants.%Int(%int_32.loc7_18) [concrete = constants.%i32.builtin]
// CHECK:STDOUT:   %.loc7_18.1: type = value_of_initializer %int.make_type_signed.loc7_18 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:   %.loc7_18.2: type = converted %int.make_type_signed.loc7_18, %.loc7_18.1 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:   %.loc7_16: %D.elem = field_decl n, element0 [concrete]
// CHECK:STDOUT:   %int_32.loc7_30: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc7_30: init type = call constants.%Int(%int_32.loc7_30) [concrete = constants.%i32.builtin]
// CHECK:STDOUT:   %.loc7_30.1: type = value_of_initializer %int.make_type_signed.loc7_30 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:   %.loc7_30.2: type = converted %int.make_type_signed.loc7_30, %.loc7_30.1 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:   %.loc7_28: %D.elem = field_decl m, element1 [concrete]
// CHECK:STDOUT:   %struct_type.n.m: type = struct_type {.n: %i32.builtin, .m: %i32.builtin} [concrete = constants.%struct_type.n.m.566]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.n.m [concrete = constants.%complete_type.682]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT:   .n = %.loc7_16
// CHECK:STDOUT:   .m = %.loc7_28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int = "int.make_type_signed" [from "core.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make() -> %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_0.loc8_31: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %int_0.loc8_39: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc8_40.1: %struct_type.n.m.819 = struct_literal (%int_0.loc8_31, %int_0.loc8_39)
// CHECK:STDOUT:   %impl.elem0.loc8_40.1: %.ec1 = impl_witness_access constants.%ImplicitAs.impl_witness.07a, element0 [concrete = constants.%Convert.cb5]
// CHECK:STDOUT:   %bound_method.loc8_40.1: <bound method> = bound_method %int_0.loc8_31, %impl.elem0.loc8_40.1 [concrete = constants.%Convert.bound.04a]
// CHECK:STDOUT:   %int.convert_checked.loc8_40.1: init %i32.builtin = call %bound_method.loc8_40.1(%int_0.loc8_31) [concrete = constants.%int_0.a54]
// CHECK:STDOUT:   %.loc8_40.2: init %i32.builtin = converted %int_0.loc8_31, %int.convert_checked.loc8_40.1 [concrete = constants.%int_0.a54]
// CHECK:STDOUT:   %.loc8_40.3: ref %i32.builtin = class_element_access %return, element0
// CHECK:STDOUT:   %.loc8_40.4: init %i32.builtin = initialize_from %.loc8_40.2 to %.loc8_40.3 [concrete = constants.%int_0.a54]
// CHECK:STDOUT:   %impl.elem0.loc8_40.2: %.ec1 = impl_witness_access constants.%ImplicitAs.impl_witness.07a, element0 [concrete = constants.%Convert.cb5]
// CHECK:STDOUT:   %bound_method.loc8_40.2: <bound method> = bound_method %int_0.loc8_39, %impl.elem0.loc8_40.2 [concrete = constants.%Convert.bound.04a]
// CHECK:STDOUT:   %int.convert_checked.loc8_40.2: init %i32.builtin = call %bound_method.loc8_40.2(%int_0.loc8_39) [concrete = constants.%int_0.a54]
// CHECK:STDOUT:   %.loc8_40.5: init %i32.builtin = converted %int_0.loc8_39, %int.convert_checked.loc8_40.2 [concrete = constants.%int_0.a54]
// CHECK:STDOUT:   %.loc8_40.6: ref %i32.builtin = class_element_access %return, element1
// CHECK:STDOUT:   %.loc8_40.7: init %i32.builtin = initialize_from %.loc8_40.5 to %.loc8_40.6 [concrete = constants.%int_0.a54]
// CHECK:STDOUT:   %.loc8_40.8: init %D = class_init (%.loc8_40.4, %.loc8_40.7), %return [concrete = constants.%D.val]
// CHECK:STDOUT:   %.loc8_41: init %D = converted %.loc8_40.1, %.loc8_40.8 [concrete = constants.%D.val]
// CHECK:STDOUT:   return %.loc8_41 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.1(imports.%Core.import_ref.5ab3ec.2: type, imports.%Core.import_ref.ce1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62)) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.d62)]
// CHECK:STDOUT:   %Self: @Convert.1.%ImplicitAs.type (%ImplicitAs.type.d62) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.519)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type.1: type = pattern_type %Self.as_type [symbolic = %pattern_type.1 (constants.%pattern_type.f3e)]
// CHECK:STDOUT:   %pattern_type.2: type = pattern_type %T [symbolic = %pattern_type.2 (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.2 = "int.convert_checked" [from "core.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.3(%self.param: %C.808) -> %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc10: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc10
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.4(%self.param: %C.8be) -> %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc11: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc11
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.5(%self.param: %C.c17) -> %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc12: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc12
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.6(%self.param: %C.414) -> %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc13: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc13
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.7(%self.param: %C.488) -> %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc14: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc14
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.8(%self.param: %C.3e2) -> %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc15: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc15
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.9(%self.param: %C.78c) -> %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc16: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc16
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.10(%self.param: %C.6aa) -> %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc17: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc17
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%N) {
// CHECK:STDOUT:   %N.loc6_9.2 => constants.%N
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%T, constants.%Self.519) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.d62
// CHECK:STDOUT:   %Self => constants.%Self.519
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.f3e
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.7dc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%i32.builtin) {
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.61e
// CHECK:STDOUT:   %Self => constants.%Self.d28
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.059
// CHECK:STDOUT:   %Convert => constants.%Convert.4d7
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.398
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.6fd
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_0.a54) {
// CHECK:STDOUT:   %N.loc6_9.2 => constants.%int_0.a54
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%D) {
// CHECK:STDOUT:   %T => constants.%D
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.94e
// CHECK:STDOUT:   %Self => constants.%Self.371
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.010
// CHECK:STDOUT:   %Convert => constants.%Convert.d38
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.75e
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.579
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%D, constants.%ImplicitAs.facet.a19) {
// CHECK:STDOUT:   %T => constants.%D
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.94e
// CHECK:STDOUT:   %Self => constants.%ImplicitAs.facet.a19
// CHECK:STDOUT:   %Self.as_type => constants.%C.808
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.658
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.510
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_1.f38) {
// CHECK:STDOUT:   %N.loc6_9.2 => constants.%int_1.f38
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%D, constants.%ImplicitAs.facet.340) {
// CHECK:STDOUT:   %T => constants.%D
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.94e
// CHECK:STDOUT:   %Self => constants.%ImplicitAs.facet.340
// CHECK:STDOUT:   %Self.as_type => constants.%C.8be
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.a77
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.510
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_2.5a1) {
// CHECK:STDOUT:   %N.loc6_9.2 => constants.%int_2.5a1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%D, constants.%ImplicitAs.facet.ec0) {
// CHECK:STDOUT:   %T => constants.%D
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.94e
// CHECK:STDOUT:   %Self => constants.%ImplicitAs.facet.ec0
// CHECK:STDOUT:   %Self.as_type => constants.%C.c17
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.58e
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.510
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_3.a0f) {
// CHECK:STDOUT:   %N.loc6_9.2 => constants.%int_3.a0f
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%D, constants.%ImplicitAs.facet.999) {
// CHECK:STDOUT:   %T => constants.%D
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.94e
// CHECK:STDOUT:   %Self => constants.%ImplicitAs.facet.999
// CHECK:STDOUT:   %Self.as_type => constants.%C.414
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.861
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.510
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_4.4f1) {
// CHECK:STDOUT:   %N.loc6_9.2 => constants.%int_4.4f1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%D, constants.%ImplicitAs.facet.527) {
// CHECK:STDOUT:   %T => constants.%D
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.94e
// CHECK:STDOUT:   %Self => constants.%ImplicitAs.facet.527
// CHECK:STDOUT:   %Self.as_type => constants.%C.488
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.d35
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.510
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_5.967) {
// CHECK:STDOUT:   %N.loc6_9.2 => constants.%int_5.967
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%D, constants.%ImplicitAs.facet.79a) {
// CHECK:STDOUT:   %T => constants.%D
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.94e
// CHECK:STDOUT:   %Self => constants.%ImplicitAs.facet.79a
// CHECK:STDOUT:   %Self.as_type => constants.%C.3e2
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.c88
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.510
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_6.ec5) {
// CHECK:STDOUT:   %N.loc6_9.2 => constants.%int_6.ec5
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%D, constants.%ImplicitAs.facet.5ba) {
// CHECK:STDOUT:   %T => constants.%D
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.94e
// CHECK:STDOUT:   %Self => constants.%ImplicitAs.facet.5ba
// CHECK:STDOUT:   %Self.as_type => constants.%C.78c
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.af2
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.510
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_7.6ae) {
// CHECK:STDOUT:   %N.loc6_9.2 => constants.%int_7.6ae
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%D, constants.%ImplicitAs.facet.b5c) {
// CHECK:STDOUT:   %T => constants.%D
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.94e
// CHECK:STDOUT:   %Self => constants.%ImplicitAs.facet.b5c
// CHECK:STDOUT:   %Self.as_type => constants.%C.6aa
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.045
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.510
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- user.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [concrete]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %pattern_type.956: type = pattern_type %i32.builtin [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %struct_type.n.m: type = struct_type {.n: %i32.builtin, .m: %i32.builtin} [concrete]
// CHECK:STDOUT:   %complete_type.682: <witness> = complete_type_witness %struct_type.n.m [concrete]
// CHECK:STDOUT:   %pattern_type.f8a: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %F0.type: type = fn_type @F0 [concrete]
// CHECK:STDOUT:   %F0: %F0.type = struct_value () [concrete]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %N: %i32.builtin = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.d62: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic]
// CHECK:STDOUT:   %Self.519: %ImplicitAs.type.d62 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.275: type = fn_type @Convert.1, @ImplicitAs(%T) [symbolic]
// CHECK:STDOUT:   %Convert.42e: %Convert.type.275 = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.519 [symbolic]
// CHECK:STDOUT:   %pattern_type.f3e: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.ca0: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.9f5: %ImplicitAs.assoc_type.ca0 = assoc_entity element0, imports.%Core.import_ref.1c7 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.61e: type = facet_type <@ImplicitAs, @ImplicitAs(%i32.builtin)> [concrete]
// CHECK:STDOUT:   %Self.d28: %ImplicitAs.type.61e = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.059: type = fn_type @Convert.1, @ImplicitAs(%i32.builtin) [concrete]
// CHECK:STDOUT:   %Convert.4d7: %Convert.type.059 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.398: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%i32.builtin) [concrete]
// CHECK:STDOUT:   %assoc0.6fd: %ImplicitAs.assoc_type.398 = assoc_entity element0, imports.%Core.import_ref.1c7 [concrete]
// CHECK:STDOUT:   %assoc0.dc0: %ImplicitAs.assoc_type.ca0 = assoc_entity element0, imports.%Core.import_ref.207 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.07a: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.78e [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.a3d: %ImplicitAs.type.61e = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.07a) [concrete]
// CHECK:STDOUT:   %.ec1e: type = fn_type_with_self_type %Convert.type.059, %ImplicitAs.facet.a3d [concrete]
// CHECK:STDOUT:   %Convert.type.49f: type = fn_type @Convert.2 [concrete]
// CHECK:STDOUT:   %Convert.cb5: %Convert.type.49f = struct_value () [concrete]
// CHECK:STDOUT:   %Convert.bound.04a: <bound method> = bound_method %int_0.5c6, %Convert.cb5 [concrete]
// CHECK:STDOUT:   %int_0.a54: %i32.builtin = int_value 0 [concrete]
// CHECK:STDOUT:   %C.76d: type = class_type @C, @C(%int_0.a54) [concrete]
// CHECK:STDOUT:   %C.val.3c3: %C.76d = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.5f9: type = facet_type <@ImplicitAs, @ImplicitAs(%D)> [concrete]
// CHECK:STDOUT:   %Self.656: %ImplicitAs.type.5f9 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.334: type = fn_type @Convert.1, @ImplicitAs(%D) [concrete]
// CHECK:STDOUT:   %Convert.87c: %Convert.type.334 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.953: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%D) [concrete]
// CHECK:STDOUT:   %assoc0.8a3: %ImplicitAs.assoc_type.953 = assoc_entity element0, imports.%Core.import_ref.1c7 [concrete]
// CHECK:STDOUT:   %int_1.f38: %i32.builtin = int_value 1 [concrete]
// CHECK:STDOUT:   %C.012: type = class_type @C, @C(%int_1.f38) [concrete]
// CHECK:STDOUT:   %int_2.5a1: %i32.builtin = int_value 2 [concrete]
// CHECK:STDOUT:   %C.3e6: type = class_type @C, @C(%int_2.5a1) [concrete]
// CHECK:STDOUT:   %int_3.a0f: %i32.builtin = int_value 3 [concrete]
// CHECK:STDOUT:   %C.4d9: type = class_type @C, @C(%int_3.a0f) [concrete]
// CHECK:STDOUT:   %int_4.4f1: %i32.builtin = int_value 4 [concrete]
// CHECK:STDOUT:   %C.a67: type = class_type @C, @C(%int_4.4f1) [concrete]
// CHECK:STDOUT:   %int_5.967: %i32.builtin = int_value 5 [concrete]
// CHECK:STDOUT:   %C.65c: type = class_type @C, @C(%int_5.967) [concrete]
// CHECK:STDOUT:   %int_6.ec5: %i32.builtin = int_value 6 [concrete]
// CHECK:STDOUT:   %C.898: type = class_type @C, @C(%int_6.ec5) [concrete]
// CHECK:STDOUT:   %int_7.6ae: %i32.builtin = int_value 7 [concrete]
// CHECK:STDOUT:   %C.f0a: type = class_type @C, @C(%int_7.6ae) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.6e5: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.497 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.5c5: %ImplicitAs.type.5f9 = facet_value %C.76d, (%ImplicitAs.impl_witness.6e5) [concrete]
// CHECK:STDOUT:   %.18d: type = fn_type_with_self_type %Convert.type.334, %ImplicitAs.facet.5c5 [concrete]
// CHECK:STDOUT:   %Convert.type.c72: type = fn_type @Convert.3 [concrete]
// CHECK:STDOUT:   %Convert.4f5: %Convert.type.c72 = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Convert.bound.92d: <bound method> = bound_method %int_1.5b8, %Convert.cb5 [concrete]
// CHECK:STDOUT:   %C.val.172: %C.012 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.70d: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b5d [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.930: %ImplicitAs.type.5f9 = facet_value %C.012, (%ImplicitAs.impl_witness.70d) [concrete]
// CHECK:STDOUT:   %.d7a: type = fn_type_with_self_type %Convert.type.334, %ImplicitAs.facet.930 [concrete]
// CHECK:STDOUT:   %Convert.type.d88: type = fn_type @Convert.4 [concrete]
// CHECK:STDOUT:   %Convert.ecc: %Convert.type.d88 = struct_value () [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %Convert.bound.1b9: <bound method> = bound_method %int_2.ecc, %Convert.cb5 [concrete]
// CHECK:STDOUT:   %C.val.418: %C.3e6 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.f8e: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.ec0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.05e: %ImplicitAs.type.5f9 = facet_value %C.3e6, (%ImplicitAs.impl_witness.f8e) [concrete]
// CHECK:STDOUT:   %.e6a: type = fn_type_with_self_type %Convert.type.334, %ImplicitAs.facet.05e [concrete]
// CHECK:STDOUT:   %Convert.type.c17: type = fn_type @Convert.5 [concrete]
// CHECK:STDOUT:   %Convert.c2b: %Convert.type.c17 = struct_value () [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %Convert.bound.b6b: <bound method> = bound_method %int_3.1ba, %Convert.cb5 [concrete]
// CHECK:STDOUT:   %C.val.bba: %C.4d9 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.bb8: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.129 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.eb1: %ImplicitAs.type.5f9 = facet_value %C.4d9, (%ImplicitAs.impl_witness.bb8) [concrete]
// CHECK:STDOUT:   %.40b: type = fn_type_with_self_type %Convert.type.334, %ImplicitAs.facet.eb1 [concrete]
// CHECK:STDOUT:   %Convert.type.70f: type = fn_type @Convert.6 [concrete]
// CHECK:STDOUT:   %Convert.2b5: %Convert.type.70f = struct_value () [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %Convert.bound.626: <bound method> = bound_method %int_4.0c1, %Convert.cb5 [concrete]
// CHECK:STDOUT:   %C.val.4b6: %C.a67 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.087: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.fb0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.53f: %ImplicitAs.type.5f9 = facet_value %C.a67, (%ImplicitAs.impl_witness.087) [concrete]
// CHECK:STDOUT:   %.ec10: type = fn_type_with_self_type %Convert.type.334, %ImplicitAs.facet.53f [concrete]
// CHECK:STDOUT:   %Convert.type.782: type = fn_type @Convert.7 [concrete]
// CHECK:STDOUT:   %Convert.625: %Convert.type.782 = struct_value () [concrete]
// CHECK:STDOUT:   %int_5.64b: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %Convert.bound.910: <bound method> = bound_method %int_5.64b, %Convert.cb5 [concrete]
// CHECK:STDOUT:   %C.val.75e: %C.65c = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.f50: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.fd6 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.3ed: %ImplicitAs.type.5f9 = facet_value %C.65c, (%ImplicitAs.impl_witness.f50) [concrete]
// CHECK:STDOUT:   %.c25: type = fn_type_with_self_type %Convert.type.334, %ImplicitAs.facet.3ed [concrete]
// CHECK:STDOUT:   %Convert.type.0e6: type = fn_type @Convert.8 [concrete]
// CHECK:STDOUT:   %Convert.73d: %Convert.type.0e6 = struct_value () [concrete]
// CHECK:STDOUT:   %int_6.462: Core.IntLiteral = int_value 6 [concrete]
// CHECK:STDOUT:   %Convert.bound.e3a: <bound method> = bound_method %int_6.462, %Convert.cb5 [concrete]
// CHECK:STDOUT:   %C.val.02a: %C.898 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.eaa: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.732 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.67f: %ImplicitAs.type.5f9 = facet_value %C.898, (%ImplicitAs.impl_witness.eaa) [concrete]
// CHECK:STDOUT:   %.c5c: type = fn_type_with_self_type %Convert.type.334, %ImplicitAs.facet.67f [concrete]
// CHECK:STDOUT:   %Convert.type.9e9: type = fn_type @Convert.9 [concrete]
// CHECK:STDOUT:   %Convert.e8e: %Convert.type.9e9 = struct_value () [concrete]
// CHECK:STDOUT:   %int_7.29f: Core.IntLiteral = int_value 7 [concrete]
// CHECK:STDOUT:   %Convert.bound.06a: <bound method> = bound_method %int_7.29f, %Convert.cb5 [concrete]
// CHECK:STDOUT:   %C.val.654: %C.f0a = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.e1e: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.c20 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.ce9: %ImplicitAs.type.5f9 = facet_value %C.f0a, (%ImplicitAs.impl_witness.e1e) [concrete]
// CHECK:STDOUT:   %.909: type = fn_type_with_self_type %Convert.type.334, %ImplicitAs.facet.ce9 [concrete]
// CHECK:STDOUT:   %Convert.type.fc1: type = fn_type @Convert.10 [concrete]
// CHECK:STDOUT:   %Convert.430: %Convert.type.fc1 = struct_value () [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %P: <namespace> = namespace file.%P.import, [concrete] {
// CHECK:STDOUT:     .D = %P.D
// CHECK:STDOUT:     .C = %P.C
// CHECK:STDOUT:     .Make = %P.Make
// CHECK:STDOUT:     import P//library
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//default, Int, loaded [concrete = constants.%Int]
// CHECK:STDOUT:   %P.D: type = import_ref P//library, D, loaded [concrete = constants.%D]
// CHECK:STDOUT:   %P.import_ref.7e5: <witness> = import_ref P//library, loc7_35, loaded [concrete = constants.%complete_type.682]
// CHECK:STDOUT:   %P.import_ref.cab = import_ref P//library, inst49 [no loc], unloaded
// CHECK:STDOUT:   %P.import_ref.a99 = import_ref P//library, loc7_16, unloaded
// CHECK:STDOUT:   %P.import_ref.9d2 = import_ref P//library, loc7_28, unloaded
// CHECK:STDOUT:   %P.C: %C.type = import_ref P//library, C, loaded [concrete = constants.%C.generic]
// CHECK:STDOUT:   %P.import_ref.85e: %i32.builtin = import_ref P//library, loc6_9, loaded [symbolic = @C.%N (constants.%N)]
// CHECK:STDOUT:   %P.import_ref.8f2: <witness> = import_ref P//library, loc6_19, loaded [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   %P.import_ref.d9b = import_ref P//library, inst43 [no loc], unloaded
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//default, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.1: type = import_ref Core//default, loc8_22, loaded [symbolic = @ImplicitAs.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.ff5 = import_ref Core//default, inst49 [no loc], unloaded
// CHECK:STDOUT:   %Core.import_ref.492: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.ca0) = import_ref Core//default, loc9_32, loaded [symbolic = @ImplicitAs.%assoc0 (constants.%assoc0.dc0)]
// CHECK:STDOUT:   %Core.Convert = import_ref Core//default, Convert, unloaded
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.2: type = import_ref Core//default, loc8_22, loaded [symbolic = @ImplicitAs.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.ce1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62) = import_ref Core//default, inst49 [no loc], loaded [symbolic = @ImplicitAs.%Self (constants.%Self.519)]
// CHECK:STDOUT:   %Core.import_ref.1c7: @ImplicitAs.%Convert.type (%Convert.type.275) = import_ref Core//default, loc9_32, loaded [symbolic = @ImplicitAs.%Convert (constants.%Convert.42e)]
// CHECK:STDOUT:   %Core.import_ref.207 = import_ref Core//default, loc9_32, unloaded
// CHECK:STDOUT:   %Core.import_ref.c62: <witness> = import_ref Core//default, loc12_38, loaded [concrete = constants.%ImplicitAs.impl_witness.07a]
// CHECK:STDOUT:   %Core.import_ref.872: type = import_ref Core//default, loc12_17, loaded [concrete = Core.IntLiteral]
// CHECK:STDOUT:   %Core.import_ref.4d9: type = import_ref Core//default, loc12_36, loaded [concrete = constants.%ImplicitAs.type.61e]
// CHECK:STDOUT:   %Core.import_ref.f35: %Convert.type.49f = import_ref Core//default, loc13_35, loaded [concrete = constants.%Convert.cb5]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.78e = impl_witness_table (%Core.import_ref.f35), @impl.68b [concrete]
// CHECK:STDOUT:   %P.import_ref.503: <witness> = import_ref P//library, loc10_33, loaded [concrete = constants.%ImplicitAs.impl_witness.6e5]
// CHECK:STDOUT:   %P.import_ref.624: type = import_ref P//library, loc10_9, loaded [concrete = constants.%C.76d]
// CHECK:STDOUT:   %P.import_ref.b769fa.1: type = import_ref P//library, loc10_31, loaded [concrete = constants.%ImplicitAs.type.5f9]
// CHECK:STDOUT:   %P.import_ref.7ac: <witness> = import_ref P//library, loc11_33, loaded [concrete = constants.%ImplicitAs.impl_witness.70d]
// CHECK:STDOUT:   %P.import_ref.8ff: type = import_ref P//library, loc11_9, loaded [concrete = constants.%C.012]
// CHECK:STDOUT:   %P.import_ref.b769fa.2: type = import_ref P//library, loc11_31, loaded [concrete = constants.%ImplicitAs.type.5f9]
// CHECK:STDOUT:   %P.import_ref.255: <witness> = import_ref P//library, loc12_33, loaded [concrete = constants.%ImplicitAs.impl_witness.f8e]
// CHECK:STDOUT:   %P.import_ref.f4d: type = import_ref P//library, loc12_9, loaded [concrete = constants.%C.3e6]
// CHECK:STDOUT:   %P.import_ref.b769fa.3: type = import_ref P//library, loc12_31, loaded [concrete = constants.%ImplicitAs.type.5f9]
// CHECK:STDOUT:   %P.import_ref.51a: <witness> = import_ref P//library, loc13_33, loaded [concrete = constants.%ImplicitAs.impl_witness.bb8]
// CHECK:STDOUT:   %P.import_ref.dd2: type = import_ref P//library, loc13_9, loaded [concrete = constants.%C.4d9]
// CHECK:STDOUT:   %P.import_ref.b769fa.4: type = import_ref P//library, loc13_31, loaded [concrete = constants.%ImplicitAs.type.5f9]
// CHECK:STDOUT:   %P.import_ref.7c4: <witness> = import_ref P//library, loc14_33, loaded [concrete = constants.%ImplicitAs.impl_witness.087]
// CHECK:STDOUT:   %P.import_ref.2c7: type = import_ref P//library, loc14_9, loaded [concrete = constants.%C.a67]
// CHECK:STDOUT:   %P.import_ref.b769fa.5: type = import_ref P//library, loc14_31, loaded [concrete = constants.%ImplicitAs.type.5f9]
// CHECK:STDOUT:   %P.import_ref.30f: <witness> = import_ref P//library, loc15_33, loaded [concrete = constants.%ImplicitAs.impl_witness.f50]
// CHECK:STDOUT:   %P.import_ref.759: type = import_ref P//library, loc15_9, loaded [concrete = constants.%C.65c]
// CHECK:STDOUT:   %P.import_ref.b769fa.6: type = import_ref P//library, loc15_31, loaded [concrete = constants.%ImplicitAs.type.5f9]
// CHECK:STDOUT:   %P.import_ref.664: <witness> = import_ref P//library, loc16_33, loaded [concrete = constants.%ImplicitAs.impl_witness.eaa]
// CHECK:STDOUT:   %P.import_ref.56e: type = import_ref P//library, loc16_9, loaded [concrete = constants.%C.898]
// CHECK:STDOUT:   %P.import_ref.b769fa.7: type = import_ref P//library, loc16_31, loaded [concrete = constants.%ImplicitAs.type.5f9]
// CHECK:STDOUT:   %P.import_ref.322: <witness> = import_ref P//library, loc17_33, loaded [concrete = constants.%ImplicitAs.impl_witness.e1e]
// CHECK:STDOUT:   %P.import_ref.8ba: type = import_ref P//library, loc17_9, loaded [concrete = constants.%C.f0a]
// CHECK:STDOUT:   %P.import_ref.b769fa.8: type = import_ref P//library, loc17_31, loaded [concrete = constants.%ImplicitAs.type.5f9]
// CHECK:STDOUT:   %P.import_ref.5f3: %Convert.type.c72 = import_ref P//library, loc10_65, loaded [concrete = constants.%Convert.4f5]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.497 = impl_witness_table (%P.import_ref.5f3), @impl.b24 [concrete]
// CHECK:STDOUT:   %P.import_ref.4da: %Convert.type.d88 = import_ref P//library, loc11_65, loaded [concrete = constants.%Convert.ecc]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b5d = impl_witness_table (%P.import_ref.4da), @impl.277 [concrete]
// CHECK:STDOUT:   %P.import_ref.ebf: %Convert.type.c17 = import_ref P//library, loc12_65, loaded [concrete = constants.%Convert.c2b]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.ec0 = impl_witness_table (%P.import_ref.ebf), @impl.b5f [concrete]
// CHECK:STDOUT:   %P.import_ref.f13: %Convert.type.70f = import_ref P//library, loc13_65, loaded [concrete = constants.%Convert.2b5]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.129 = impl_witness_table (%P.import_ref.f13), @impl.90b [concrete]
// CHECK:STDOUT:   %P.import_ref.97a: %Convert.type.782 = import_ref P//library, loc14_65, loaded [concrete = constants.%Convert.625]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.fb0 = impl_witness_table (%P.import_ref.97a), @impl.161 [concrete]
// CHECK:STDOUT:   %P.import_ref.e04: %Convert.type.0e6 = import_ref P//library, loc15_65, loaded [concrete = constants.%Convert.73d]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.fd6 = impl_witness_table (%P.import_ref.e04), @impl.54a [concrete]
// CHECK:STDOUT:   %P.import_ref.6aa: %Convert.type.9e9 = import_ref P//library, loc16_65, loaded [concrete = constants.%Convert.e8e]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.732 = impl_witness_table (%P.import_ref.6aa), @impl.d71 [concrete]
// CHECK:STDOUT:   %P.import_ref.243: %Convert.type.fc1 = import_ref P//library, loc17_65, loaded [concrete = constants.%Convert.430]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.c20 = impl_witness_table (%P.import_ref.243), @impl.16b [concrete]
// CHECK:STDOUT:   %P.Make: %Make.type = import_ref P//library, Make, loaded [concrete = constants.%Make]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .P = imports.%P
// CHECK:STDOUT:     .F0 = %F0.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %P.import = import P
// CHECK:STDOUT:   %F0.decl: %F0.type = fn_decl @F0 [concrete = constants.%F0] {
// CHECK:STDOUT:     %n.patt: %pattern_type.956 = binding_pattern n [concrete]
// CHECK:STDOUT:     %n.param_patt: %pattern_type.956 = value_param_pattern %n.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.f8a = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.f8a = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %P.ref.loc7: <namespace> = name_ref P, imports.%P [concrete = imports.%P]
// CHECK:STDOUT:     %D.ref: type = name_ref D, imports.%P.D [concrete = constants.%D]
// CHECK:STDOUT:     %n.param: %i32.builtin = value_param call_param0
// CHECK:STDOUT:     %.loc7_10.1: type = splice_block %.loc7_10.3 [concrete = constants.%i32.builtin] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %int.make_type_signed: init type = call constants.%Int(%int_32) [concrete = constants.%i32.builtin]
// CHECK:STDOUT:       %.loc7_10.2: type = value_of_initializer %int.make_type_signed [concrete = constants.%i32.builtin]
// CHECK:STDOUT:       %.loc7_10.3: type = converted %int.make_type_signed, %.loc7_10.2 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n: %i32.builtin = bind_name n, %n.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(imports.%Core.import_ref.5ab3ec.1: type) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.d62)]
// CHECK:STDOUT:   %Self: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.519)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.1, @ImplicitAs(%T) [symbolic = %Convert.type (constants.%Convert.type.275)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.275) = struct_value () [symbolic = %Convert (constants.%Convert.42e)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%T) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type.ca0)]
// CHECK:STDOUT:   %assoc0: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.ca0) = assoc_entity element0, imports.%Core.import_ref.1c7 [symbolic = %assoc0 (constants.%assoc0.9f5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Core.import_ref.ff5
// CHECK:STDOUT:     .Convert = imports.%Core.import_ref.492
// CHECK:STDOUT:     witness = (imports.%Core.Convert)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.68b: imports.%Core.import_ref.872 as imports.%Core.import_ref.4d9 [from "core.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Core.import_ref.c62
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.b24: imports.%P.import_ref.624 as imports.%P.import_ref.b769fa.1 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.503
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.277: imports.%P.import_ref.8ff as imports.%P.import_ref.b769fa.2 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.7ac
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.b5f: imports.%P.import_ref.f4d as imports.%P.import_ref.b769fa.3 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.255
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.90b: imports.%P.import_ref.dd2 as imports.%P.import_ref.b769fa.4 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.51a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.161: imports.%P.import_ref.2c7 as imports.%P.import_ref.b769fa.5 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.7c4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.54a: imports.%P.import_ref.759 as imports.%P.import_ref.b769fa.6 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.30f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.d71: imports.%P.import_ref.56e as imports.%P.import_ref.b769fa.7 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.664
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.16b: imports.%P.import_ref.8ba as imports.%P.import_ref.b769fa.8 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.322
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D [from "library.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%P.import_ref.7e5
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%P.import_ref.cab
// CHECK:STDOUT:   .n = imports.%P.import_ref.a99
// CHECK:STDOUT:   .m = imports.%P.import_ref.9d2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(imports.%P.import_ref.85e: %i32.builtin) [from "library.carbon"] {
// CHECK:STDOUT:   %N: %i32.builtin = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%P.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%P.import_ref.d9b
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int = "int.make_type_signed" [from "core.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F0(%n.param: %i32.builtin) -> %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %false.loc8: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %false.loc8 br !if.then.loc8 else br !if.else.loc8
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc8:
// CHECK:STDOUT:   %.loc8_24.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %P.ref.loc8: <namespace> = name_ref P, imports.%P [concrete = imports.%P]
// CHECK:STDOUT:   %C.ref.loc8: %C.type = name_ref C, imports.%P.C [concrete = constants.%C.generic]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0.loc8_34: %.ec1e = impl_witness_access constants.%ImplicitAs.impl_witness.07a, element0 [concrete = constants.%Convert.cb5]
// CHECK:STDOUT:   %bound_method.loc8_34: <bound method> = bound_method %int_0, %impl.elem0.loc8_34 [concrete = constants.%Convert.bound.04a]
// CHECK:STDOUT:   %int.convert_checked.loc8: init %i32.builtin = call %bound_method.loc8_34(%int_0) [concrete = constants.%int_0.a54]
// CHECK:STDOUT:   %.loc8_34.1: %i32.builtin = value_of_initializer %int.convert_checked.loc8 [concrete = constants.%int_0.a54]
// CHECK:STDOUT:   %.loc8_34.2: %i32.builtin = converted %int_0, %.loc8_34.1 [concrete = constants.%int_0.a54]
// CHECK:STDOUT:   %C.loc8: type = class_type @C, @C(constants.%int_0.a54) [concrete = constants.%C.76d]
// CHECK:STDOUT:   %.loc8_24.2: ref %C.76d = temporary_storage
// CHECK:STDOUT:   %.loc8_24.3: init %C.76d = class_init (), %.loc8_24.2 [concrete = constants.%C.val.3c3]
// CHECK:STDOUT:   %.loc8_24.4: ref %C.76d = temporary %.loc8_24.2, %.loc8_24.3
// CHECK:STDOUT:   %.loc8_26.1: ref %C.76d = converted %.loc8_24.1, %.loc8_24.4
// CHECK:STDOUT:   %impl.elem0.loc8_35: %.18d = impl_witness_access constants.%ImplicitAs.impl_witness.6e5, element0 [concrete = constants.%Convert.4f5]
// CHECK:STDOUT:   %bound_method.loc8_35: <bound method> = bound_method %.loc8_26.1, %impl.elem0.loc8_35
// CHECK:STDOUT:   %.loc8_35.1: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc8_26.2: %C.76d = bind_value %.loc8_26.1
// CHECK:STDOUT:   %Convert.call.loc8: init %D = call %bound_method.loc8_35(%.loc8_26.2) to %.loc8_35.1
// CHECK:STDOUT:   %.loc8_35.2: init %D = converted %.loc8_26.1, %Convert.call.loc8
// CHECK:STDOUT:   return %.loc8_35.2 to %return
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc8:
// CHECK:STDOUT:   %false.loc9: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %false.loc9 br !if.then.loc9 else br !if.else.loc9
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc9:
// CHECK:STDOUT:   %.loc9_24.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %P.ref.loc9: <namespace> = name_ref P, imports.%P [concrete = imports.%P]
// CHECK:STDOUT:   %C.ref.loc9: %C.type = name_ref C, imports.%P.C [concrete = constants.%C.generic]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc9_34: %.ec1e = impl_witness_access constants.%ImplicitAs.impl_witness.07a, element0 [concrete = constants.%Convert.cb5]
// CHECK:STDOUT:   %bound_method.loc9_34: <bound method> = bound_method %int_1, %impl.elem0.loc9_34 [concrete = constants.%Convert.bound.92d]
// CHECK:STDOUT:   %int.convert_checked.loc9: init %i32.builtin = call %bound_method.loc9_34(%int_1) [concrete = constants.%int_1.f38]
// CHECK:STDOUT:   %.loc9_34.1: %i32.builtin = value_of_initializer %int.convert_checked.loc9 [concrete = constants.%int_1.f38]
// CHECK:STDOUT:   %.loc9_34.2: %i32.builtin = converted %int_1, %.loc9_34.1 [concrete = constants.%int_1.f38]
// CHECK:STDOUT:   %C.loc9: type = class_type @C, @C(constants.%int_1.f38) [concrete = constants.%C.012]
// CHECK:STDOUT:   %.loc9_24.2: ref %C.012 = temporary_storage
// CHECK:STDOUT:   %.loc9_24.3: init %C.012 = class_init (), %.loc9_24.2 [concrete = constants.%C.val.172]
// CHECK:STDOUT:   %.loc9_24.4: ref %C.012 = temporary %.loc9_24.2, %.loc9_24.3
// CHECK:STDOUT:   %.loc9_26.1: ref %C.012 = converted %.loc9_24.1, %.loc9_24.4
// CHECK:STDOUT:   %impl.elem0.loc9_35: %.d7a = impl_witness_access constants.%ImplicitAs.impl_witness.70d, element0 [concrete = constants.%Convert.ecc]
// CHECK:STDOUT:   %bound_method.loc9_35: <bound method> = bound_method %.loc9_26.1, %impl.elem0.loc9_35
// CHECK:STDOUT:   %.loc9_35.1: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc9_26.2: %C.012 = bind_value %.loc9_26.1
// CHECK:STDOUT:   %Convert.call.loc9: init %D = call %bound_method.loc9_35(%.loc9_26.2) to %.loc9_35.1
// CHECK:STDOUT:   %.loc9_35.2: init %D = converted %.loc9_26.1, %Convert.call.loc9
// CHECK:STDOUT:   return %.loc9_35.2 to %return
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc9:
// CHECK:STDOUT:   %false.loc10: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %false.loc10 br !if.then.loc10 else br !if.else.loc10
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc10:
// CHECK:STDOUT:   %.loc10_24.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %P.ref.loc10: <namespace> = name_ref P, imports.%P [concrete = imports.%P]
// CHECK:STDOUT:   %C.ref.loc10: %C.type = name_ref C, imports.%P.C [concrete = constants.%C.generic]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %impl.elem0.loc10_34: %.ec1e = impl_witness_access constants.%ImplicitAs.impl_witness.07a, element0 [concrete = constants.%Convert.cb5]
// CHECK:STDOUT:   %bound_method.loc10_34: <bound method> = bound_method %int_2, %impl.elem0.loc10_34 [concrete = constants.%Convert.bound.1b9]
// CHECK:STDOUT:   %int.convert_checked.loc10: init %i32.builtin = call %bound_method.loc10_34(%int_2) [concrete = constants.%int_2.5a1]
// CHECK:STDOUT:   %.loc10_34.1: %i32.builtin = value_of_initializer %int.convert_checked.loc10 [concrete = constants.%int_2.5a1]
// CHECK:STDOUT:   %.loc10_34.2: %i32.builtin = converted %int_2, %.loc10_34.1 [concrete = constants.%int_2.5a1]
// CHECK:STDOUT:   %C.loc10: type = class_type @C, @C(constants.%int_2.5a1) [concrete = constants.%C.3e6]
// CHECK:STDOUT:   %.loc10_24.2: ref %C.3e6 = temporary_storage
// CHECK:STDOUT:   %.loc10_24.3: init %C.3e6 = class_init (), %.loc10_24.2 [concrete = constants.%C.val.418]
// CHECK:STDOUT:   %.loc10_24.4: ref %C.3e6 = temporary %.loc10_24.2, %.loc10_24.3
// CHECK:STDOUT:   %.loc10_26.1: ref %C.3e6 = converted %.loc10_24.1, %.loc10_24.4
// CHECK:STDOUT:   %impl.elem0.loc10_35: %.e6a = impl_witness_access constants.%ImplicitAs.impl_witness.f8e, element0 [concrete = constants.%Convert.c2b]
// CHECK:STDOUT:   %bound_method.loc10_35: <bound method> = bound_method %.loc10_26.1, %impl.elem0.loc10_35
// CHECK:STDOUT:   %.loc10_35.1: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc10_26.2: %C.3e6 = bind_value %.loc10_26.1
// CHECK:STDOUT:   %Convert.call.loc10: init %D = call %bound_method.loc10_35(%.loc10_26.2) to %.loc10_35.1
// CHECK:STDOUT:   %.loc10_35.2: init %D = converted %.loc10_26.1, %Convert.call.loc10
// CHECK:STDOUT:   return %.loc10_35.2 to %return
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc10:
// CHECK:STDOUT:   %false.loc11: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %false.loc11 br !if.then.loc11 else br !if.else.loc11
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc11:
// CHECK:STDOUT:   %.loc11_24.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %P.ref.loc11: <namespace> = name_ref P, imports.%P [concrete = imports.%P]
// CHECK:STDOUT:   %C.ref.loc11: %C.type = name_ref C, imports.%P.C [concrete = constants.%C.generic]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %impl.elem0.loc11_34: %.ec1e = impl_witness_access constants.%ImplicitAs.impl_witness.07a, element0 [concrete = constants.%Convert.cb5]
// CHECK:STDOUT:   %bound_method.loc11_34: <bound method> = bound_method %int_3, %impl.elem0.loc11_34 [concrete = constants.%Convert.bound.b6b]
// CHECK:STDOUT:   %int.convert_checked.loc11: init %i32.builtin = call %bound_method.loc11_34(%int_3) [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:   %.loc11_34.1: %i32.builtin = value_of_initializer %int.convert_checked.loc11 [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:   %.loc11_34.2: %i32.builtin = converted %int_3, %.loc11_34.1 [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:   %C.loc11: type = class_type @C, @C(constants.%int_3.a0f) [concrete = constants.%C.4d9]
// CHECK:STDOUT:   %.loc11_24.2: ref %C.4d9 = temporary_storage
// CHECK:STDOUT:   %.loc11_24.3: init %C.4d9 = class_init (), %.loc11_24.2 [concrete = constants.%C.val.bba]
// CHECK:STDOUT:   %.loc11_24.4: ref %C.4d9 = temporary %.loc11_24.2, %.loc11_24.3
// CHECK:STDOUT:   %.loc11_26.1: ref %C.4d9 = converted %.loc11_24.1, %.loc11_24.4
// CHECK:STDOUT:   %impl.elem0.loc11_35: %.40b = impl_witness_access constants.%ImplicitAs.impl_witness.bb8, element0 [concrete = constants.%Convert.2b5]
// CHECK:STDOUT:   %bound_method.loc11_35: <bound method> = bound_method %.loc11_26.1, %impl.elem0.loc11_35
// CHECK:STDOUT:   %.loc11_35.1: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc11_26.2: %C.4d9 = bind_value %.loc11_26.1
// CHECK:STDOUT:   %Convert.call.loc11: init %D = call %bound_method.loc11_35(%.loc11_26.2) to %.loc11_35.1
// CHECK:STDOUT:   %.loc11_35.2: init %D = converted %.loc11_26.1, %Convert.call.loc11
// CHECK:STDOUT:   return %.loc11_35.2 to %return
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc11:
// CHECK:STDOUT:   %false.loc12: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %false.loc12 br !if.then.loc12 else br !if.else.loc12
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc12:
// CHECK:STDOUT:   %.loc12_24.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %P.ref.loc12: <namespace> = name_ref P, imports.%P [concrete = imports.%P]
// CHECK:STDOUT:   %C.ref.loc12: %C.type = name_ref C, imports.%P.C [concrete = constants.%C.generic]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:   %impl.elem0.loc12_34: %.ec1e = impl_witness_access constants.%ImplicitAs.impl_witness.07a, element0 [concrete = constants.%Convert.cb5]
// CHECK:STDOUT:   %bound_method.loc12_34: <bound method> = bound_method %int_4, %impl.elem0.loc12_34 [concrete = constants.%Convert.bound.626]
// CHECK:STDOUT:   %int.convert_checked.loc12: init %i32.builtin = call %bound_method.loc12_34(%int_4) [concrete = constants.%int_4.4f1]
// CHECK:STDOUT:   %.loc12_34.1: %i32.builtin = value_of_initializer %int.convert_checked.loc12 [concrete = constants.%int_4.4f1]
// CHECK:STDOUT:   %.loc12_34.2: %i32.builtin = converted %int_4, %.loc12_34.1 [concrete = constants.%int_4.4f1]
// CHECK:STDOUT:   %C.loc12: type = class_type @C, @C(constants.%int_4.4f1) [concrete = constants.%C.a67]
// CHECK:STDOUT:   %.loc12_24.2: ref %C.a67 = temporary_storage
// CHECK:STDOUT:   %.loc12_24.3: init %C.a67 = class_init (), %.loc12_24.2 [concrete = constants.%C.val.4b6]
// CHECK:STDOUT:   %.loc12_24.4: ref %C.a67 = temporary %.loc12_24.2, %.loc12_24.3
// CHECK:STDOUT:   %.loc12_26.1: ref %C.a67 = converted %.loc12_24.1, %.loc12_24.4
// CHECK:STDOUT:   %impl.elem0.loc12_35: %.ec10 = impl_witness_access constants.%ImplicitAs.impl_witness.087, element0 [concrete = constants.%Convert.625]
// CHECK:STDOUT:   %bound_method.loc12_35: <bound method> = bound_method %.loc12_26.1, %impl.elem0.loc12_35
// CHECK:STDOUT:   %.loc12_35.1: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc12_26.2: %C.a67 = bind_value %.loc12_26.1
// CHECK:STDOUT:   %Convert.call.loc12: init %D = call %bound_method.loc12_35(%.loc12_26.2) to %.loc12_35.1
// CHECK:STDOUT:   %.loc12_35.2: init %D = converted %.loc12_26.1, %Convert.call.loc12
// CHECK:STDOUT:   return %.loc12_35.2 to %return
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc12:
// CHECK:STDOUT:   %false.loc13: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %false.loc13 br !if.then.loc13 else br !if.else.loc13
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc13:
// CHECK:STDOUT:   %.loc13_24.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %P.ref.loc13: <namespace> = name_ref P, imports.%P [concrete = imports.%P]
// CHECK:STDOUT:   %C.ref.loc13: %C.type = name_ref C, imports.%P.C [concrete = constants.%C.generic]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5.64b]
// CHECK:STDOUT:   %impl.elem0.loc13_34: %.ec1e = impl_witness_access constants.%ImplicitAs.impl_witness.07a, element0 [concrete = constants.%Convert.cb5]
// CHECK:STDOUT:   %bound_method.loc13_34: <bound method> = bound_method %int_5, %impl.elem0.loc13_34 [concrete = constants.%Convert.bound.910]
// CHECK:STDOUT:   %int.convert_checked.loc13: init %i32.builtin = call %bound_method.loc13_34(%int_5) [concrete = constants.%int_5.967]
// CHECK:STDOUT:   %.loc13_34.1: %i32.builtin = value_of_initializer %int.convert_checked.loc13 [concrete = constants.%int_5.967]
// CHECK:STDOUT:   %.loc13_34.2: %i32.builtin = converted %int_5, %.loc13_34.1 [concrete = constants.%int_5.967]
// CHECK:STDOUT:   %C.loc13: type = class_type @C, @C(constants.%int_5.967) [concrete = constants.%C.65c]
// CHECK:STDOUT:   %.loc13_24.2: ref %C.65c = temporary_storage
// CHECK:STDOUT:   %.loc13_24.3: init %C.65c = class_init (), %.loc13_24.2 [concrete = constants.%C.val.75e]
// CHECK:STDOUT:   %.loc13_24.4: ref %C.65c = temporary %.loc13_24.2, %.loc13_24.3
// CHECK:STDOUT:   %.loc13_26.1: ref %C.65c = converted %.loc13_24.1, %.loc13_24.4
// CHECK:STDOUT:   %impl.elem0.loc13_35: %.c25 = impl_witness_access constants.%ImplicitAs.impl_witness.f50, element0 [concrete = constants.%Convert.73d]
// CHECK:STDOUT:   %bound_method.loc13_35: <bound method> = bound_method %.loc13_26.1, %impl.elem0.loc13_35
// CHECK:STDOUT:   %.loc13_35.1: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc13_26.2: %C.65c = bind_value %.loc13_26.1
// CHECK:STDOUT:   %Convert.call.loc13: init %D = call %bound_method.loc13_35(%.loc13_26.2) to %.loc13_35.1
// CHECK:STDOUT:   %.loc13_35.2: init %D = converted %.loc13_26.1, %Convert.call.loc13
// CHECK:STDOUT:   return %.loc13_35.2 to %return
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc13:
// CHECK:STDOUT:   %false.loc14: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %false.loc14 br !if.then.loc14 else br !if.else.loc14
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc14:
// CHECK:STDOUT:   %.loc14_24.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %P.ref.loc14: <namespace> = name_ref P, imports.%P [concrete = imports.%P]
// CHECK:STDOUT:   %C.ref.loc14: %C.type = name_ref C, imports.%P.C [concrete = constants.%C.generic]
// CHECK:STDOUT:   %int_6: Core.IntLiteral = int_value 6 [concrete = constants.%int_6.462]
// CHECK:STDOUT:   %impl.elem0.loc14_34: %.ec1e = impl_witness_access constants.%ImplicitAs.impl_witness.07a, element0 [concrete = constants.%Convert.cb5]
// CHECK:STDOUT:   %bound_method.loc14_34: <bound method> = bound_method %int_6, %impl.elem0.loc14_34 [concrete = constants.%Convert.bound.e3a]
// CHECK:STDOUT:   %int.convert_checked.loc14: init %i32.builtin = call %bound_method.loc14_34(%int_6) [concrete = constants.%int_6.ec5]
// CHECK:STDOUT:   %.loc14_34.1: %i32.builtin = value_of_initializer %int.convert_checked.loc14 [concrete = constants.%int_6.ec5]
// CHECK:STDOUT:   %.loc14_34.2: %i32.builtin = converted %int_6, %.loc14_34.1 [concrete = constants.%int_6.ec5]
// CHECK:STDOUT:   %C.loc14: type = class_type @C, @C(constants.%int_6.ec5) [concrete = constants.%C.898]
// CHECK:STDOUT:   %.loc14_24.2: ref %C.898 = temporary_storage
// CHECK:STDOUT:   %.loc14_24.3: init %C.898 = class_init (), %.loc14_24.2 [concrete = constants.%C.val.02a]
// CHECK:STDOUT:   %.loc14_24.4: ref %C.898 = temporary %.loc14_24.2, %.loc14_24.3
// CHECK:STDOUT:   %.loc14_26.1: ref %C.898 = converted %.loc14_24.1, %.loc14_24.4
// CHECK:STDOUT:   %impl.elem0.loc14_35: %.c5c = impl_witness_access constants.%ImplicitAs.impl_witness.eaa, element0 [concrete = constants.%Convert.e8e]
// CHECK:STDOUT:   %bound_method.loc14_35: <bound method> = bound_method %.loc14_26.1, %impl.elem0.loc14_35
// CHECK:STDOUT:   %.loc14_35.1: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc14_26.2: %C.898 = bind_value %.loc14_26.1
// CHECK:STDOUT:   %Convert.call.loc14: init %D = call %bound_method.loc14_35(%.loc14_26.2) to %.loc14_35.1
// CHECK:STDOUT:   %.loc14_35.2: init %D = converted %.loc14_26.1, %Convert.call.loc14
// CHECK:STDOUT:   return %.loc14_35.2 to %return
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc14:
// CHECK:STDOUT:   %false.loc15: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %false.loc15 br !if.then.loc15 else br !if.else.loc15
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc15:
// CHECK:STDOUT:   %.loc15_24.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %P.ref.loc15: <namespace> = name_ref P, imports.%P [concrete = imports.%P]
// CHECK:STDOUT:   %C.ref.loc15: %C.type = name_ref C, imports.%P.C [concrete = constants.%C.generic]
// CHECK:STDOUT:   %int_7: Core.IntLiteral = int_value 7 [concrete = constants.%int_7.29f]
// CHECK:STDOUT:   %impl.elem0.loc15_34: %.ec1e = impl_witness_access constants.%ImplicitAs.impl_witness.07a, element0 [concrete = constants.%Convert.cb5]
// CHECK:STDOUT:   %bound_method.loc15_34: <bound method> = bound_method %int_7, %impl.elem0.loc15_34 [concrete = constants.%Convert.bound.06a]
// CHECK:STDOUT:   %int.convert_checked.loc15: init %i32.builtin = call %bound_method.loc15_34(%int_7) [concrete = constants.%int_7.6ae]
// CHECK:STDOUT:   %.loc15_34.1: %i32.builtin = value_of_initializer %int.convert_checked.loc15 [concrete = constants.%int_7.6ae]
// CHECK:STDOUT:   %.loc15_34.2: %i32.builtin = converted %int_7, %.loc15_34.1 [concrete = constants.%int_7.6ae]
// CHECK:STDOUT:   %C.loc15: type = class_type @C, @C(constants.%int_7.6ae) [concrete = constants.%C.f0a]
// CHECK:STDOUT:   %.loc15_24.2: ref %C.f0a = temporary_storage
// CHECK:STDOUT:   %.loc15_24.3: init %C.f0a = class_init (), %.loc15_24.2 [concrete = constants.%C.val.654]
// CHECK:STDOUT:   %.loc15_24.4: ref %C.f0a = temporary %.loc15_24.2, %.loc15_24.3
// CHECK:STDOUT:   %.loc15_26.1: ref %C.f0a = converted %.loc15_24.1, %.loc15_24.4
// CHECK:STDOUT:   %impl.elem0.loc15_35: %.909 = impl_witness_access constants.%ImplicitAs.impl_witness.e1e, element0 [concrete = constants.%Convert.430]
// CHECK:STDOUT:   %bound_method.loc15_35: <bound method> = bound_method %.loc15_26.1, %impl.elem0.loc15_35
// CHECK:STDOUT:   %.loc15_35.1: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc15_26.2: %C.f0a = bind_value %.loc15_26.1
// CHECK:STDOUT:   %Convert.call.loc15: init %D = call %bound_method.loc15_35(%.loc15_26.2) to %.loc15_35.1
// CHECK:STDOUT:   %.loc15_35.2: init %D = converted %.loc15_26.1, %Convert.call.loc15
// CHECK:STDOUT:   return %.loc15_35.2 to %return
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc15:
// CHECK:STDOUT:   %P.ref.loc16: <namespace> = name_ref P, imports.%P [concrete = imports.%P]
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, imports.%P.Make [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc7_15: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc7_15
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.1(imports.%Core.import_ref.5ab3ec.2: type, imports.%Core.import_ref.ce1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62)) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.d62)]
// CHECK:STDOUT:   %Self: @Convert.1.%ImplicitAs.type (%ImplicitAs.type.d62) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.519)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type.1: type = pattern_type %Self.as_type [symbolic = %pattern_type.1 (constants.%pattern_type.f3e)]
// CHECK:STDOUT:   %pattern_type.2: type = pattern_type %T [symbolic = %pattern_type.2 (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.2 = "int.convert_checked" [from "core.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.3 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.4 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.5 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.6 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.7 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.8 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.9 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.10 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%T, constants.%Self.519) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.d62
// CHECK:STDOUT:   %Self => constants.%Self.519
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.f3e
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.7dc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%i32.builtin) {
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.61e
// CHECK:STDOUT:   %Self => constants.%Self.d28
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.059
// CHECK:STDOUT:   %Convert => constants.%Convert.4d7
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.398
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.6fd
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_0.a54) {
// CHECK:STDOUT:   %N => constants.%int_0.a54
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%D) {
// CHECK:STDOUT:   %T => constants.%D
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.5f9
// CHECK:STDOUT:   %Self => constants.%Self.656
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.334
// CHECK:STDOUT:   %Convert => constants.%Convert.87c
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.953
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.8a3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_1.f38) {
// CHECK:STDOUT:   %N => constants.%int_1.f38
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_2.5a1) {
// CHECK:STDOUT:   %N => constants.%int_2.5a1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_3.a0f) {
// CHECK:STDOUT:   %N => constants.%int_3.a0f
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_4.4f1) {
// CHECK:STDOUT:   %N => constants.%int_4.4f1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_5.967) {
// CHECK:STDOUT:   %N => constants.%int_5.967
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_6.ec5) {
// CHECK:STDOUT:   %N => constants.%int_6.ec5
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_7.6ae) {
// CHECK:STDOUT:   %N => constants.%int_7.6ae
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:

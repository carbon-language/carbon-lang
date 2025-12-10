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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/builtin/call_from_operator.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/builtin/call_from_operator.carbon

// --- core.carbon

package Core;

fn IntLiteral() -> type = "int_literal.make_type";
fn Int(N: IntLiteral()) -> type = "int.make_type_signed";

interface AddWith(T:! type) {
  fn Op[self: Self](other: Self) -> Self;
}

interface As(T:! type) {
  fn Convert[self: Self]() -> T;
}

interface ImplicitAs(T:! type) {
  fn Convert[self: Self]() -> T;
}

impl i32 as AddWith(i32) {
  fn Op[self: Self](other: Self) -> Self = "int.sadd";
}

impl IntLiteral() as As(i32) {
  fn Convert[self: Self]() -> i32 = "int.convert_checked";
}

impl IntLiteral() as ImplicitAs(i32) {
  fn Convert[self: Self]() -> i32 = "int.convert_checked";
}

impl i32 as ImplicitAs(IntLiteral()) {
  fn Convert[self: Self]() -> IntLiteral() = "int.convert_checked";
}

// --- user.carbon

import Core;

var arr: array(i32, (1 as i32) + (2 as i32)) = (3, 4, (3 as i32) + (4 as i32));

// CHECK:STDOUT: --- core.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [concrete]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.dc0: type = pattern_type Core.IntLiteral [concrete]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [concrete]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %AddWith.type.b56: type = generic_interface_type @AddWith [concrete]
// CHECK:STDOUT:   %AddWith.generic: %AddWith.type.b56 = struct_value () [concrete]
// CHECK:STDOUT:   %AddWith.type.06e: type = facet_type <@AddWith, @AddWith(%T)> [symbolic]
// CHECK:STDOUT:   %Self.dbc: %AddWith.type.06e = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.588: type = symbolic_binding_type Self, 1, %Self.dbc [symbolic]
// CHECK:STDOUT:   %pattern_type.7c8: type = pattern_type %Self.binding.as_type.588 [symbolic]
// CHECK:STDOUT:   %AddWith.Op.type.421: type = fn_type @AddWith.Op, @AddWith(%T) [symbolic]
// CHECK:STDOUT:   %AddWith.Op.59b: %AddWith.Op.type.421 = struct_value () [symbolic]
// CHECK:STDOUT:   %AddWith.assoc_type.5ad: type = assoc_entity_type @AddWith, @AddWith(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.438: %AddWith.assoc_type.5ad = assoc_entity element0, @AddWith.%AddWith.Op.decl [symbolic]
// CHECK:STDOUT:   %As.type.3c9: type = generic_interface_type @As [concrete]
// CHECK:STDOUT:   %As.generic: %As.type.3c9 = struct_value () [concrete]
// CHECK:STDOUT:   %As.type.888: type = facet_type <@As, @As(%T)> [symbolic]
// CHECK:STDOUT:   %Self.915: %As.type.888 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.f0b: type = symbolic_binding_type Self, 1, %Self.915 [symbolic]
// CHECK:STDOUT:   %pattern_type.69e: type = pattern_type %Self.binding.as_type.f0b [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %As.Convert.type.d47: type = fn_type @As.Convert, @As(%T) [symbolic]
// CHECK:STDOUT:   %As.Convert.4a7: %As.Convert.type.d47 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.assoc_type.520: type = assoc_entity_type @As, @As(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.84f: %As.assoc_type.520 = assoc_entity element0, @As.%As.Convert.decl [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.595: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.595 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.c63: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic]
// CHECK:STDOUT:   %Self.0b8: %ImplicitAs.type.c63 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.c57: type = symbolic_binding_type Self, 1, %Self.0b8 [symbolic]
// CHECK:STDOUT:   %pattern_type.5d1: type = pattern_type %Self.binding.as_type.c57 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.103: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%T) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.Convert.2cc: %ImplicitAs.Convert.type.103 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.b04: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.cd9: %ImplicitAs.assoc_type.b04 = assoc_entity element0, @ImplicitAs.%ImplicitAs.Convert.decl [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %AddWith.type.943: type = facet_type <@AddWith, @AddWith(%i32.builtin)> [concrete]
// CHECK:STDOUT:   %AddWith.impl_witness: <witness> = impl_witness @i32.builtin.as.AddWith.impl.%AddWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %Self.53c: %AddWith.type.943 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %AddWith.Op.type.bae: type = fn_type @AddWith.Op, @AddWith(%i32.builtin) [concrete]
// CHECK:STDOUT:   %AddWith.Op.a6f: %AddWith.Op.type.bae = struct_value () [concrete]
// CHECK:STDOUT:   %AddWith.assoc_type.97c: type = assoc_entity_type @AddWith, @AddWith(%i32.builtin) [concrete]
// CHECK:STDOUT:   %assoc0.61e: %AddWith.assoc_type.97c = assoc_entity element0, @AddWith.%AddWith.Op.decl [concrete]
// CHECK:STDOUT:   %pattern_type.956: type = pattern_type %i32.builtin [concrete]
// CHECK:STDOUT:   %i32.builtin.as.AddWith.impl.Op.type: type = fn_type @i32.builtin.as.AddWith.impl.Op [concrete]
// CHECK:STDOUT:   %i32.builtin.as.AddWith.impl.Op: %i32.builtin.as.AddWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %AddWith.facet: %AddWith.type.943 = facet_value %i32.builtin, (%AddWith.impl_witness) [concrete]
// CHECK:STDOUT:   %As.type.0c6: type = facet_type <@As, @As(%i32.builtin)> [concrete]
// CHECK:STDOUT:   %As.impl_witness: <witness> = impl_witness @Core.IntLiteral.as.As.impl.%As.impl_witness_table [concrete]
// CHECK:STDOUT:   %Self.013: %As.type.0c6 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %As.Convert.type.063: type = fn_type @As.Convert, @As(%i32.builtin) [concrete]
// CHECK:STDOUT:   %As.Convert.701: %As.Convert.type.063 = struct_value () [concrete]
// CHECK:STDOUT:   %As.assoc_type.c44: type = assoc_entity_type @As, @As(%i32.builtin) [concrete]
// CHECK:STDOUT:   %assoc0.d89: %As.assoc_type.c44 = assoc_entity element0, @As.%As.Convert.decl [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type: type = fn_type @Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert: %Core.IntLiteral.as.As.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.0c6 = facet_value Core.IntLiteral, (%As.impl_witness) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.7cd: type = facet_type <@ImplicitAs, @ImplicitAs(%i32.builtin)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.d89: <witness> = impl_witness @Core.IntLiteral.as.ImplicitAs.impl.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %Self.ece: %ImplicitAs.type.7cd = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.424: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32.builtin) [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.a1f: %ImplicitAs.Convert.type.424 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.c3f: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%i32.builtin) [concrete]
// CHECK:STDOUT:   %assoc0.5e9: %ImplicitAs.assoc_type.c3f = assoc_entity element0, @ImplicitAs.%ImplicitAs.Convert.decl [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.b3c: %ImplicitAs.type.7cd = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.d89) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.5c7: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.285: <witness> = impl_witness @i32.builtin.as.ImplicitAs.impl.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %Self.12f: %ImplicitAs.type.5c7 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.785: type = fn_type @ImplicitAs.Convert, @ImplicitAs(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.aec: %ImplicitAs.Convert.type.785 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.793: type = assoc_entity_type @ImplicitAs, @ImplicitAs(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %assoc0.49f: %ImplicitAs.assoc_type.793 = assoc_entity element0, @ImplicitAs.%ImplicitAs.Convert.decl [concrete]
// CHECK:STDOUT:   %i32.builtin.as.ImplicitAs.impl.Convert.type: type = fn_type @i32.builtin.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %i32.builtin.as.ImplicitAs.impl.Convert: %i32.builtin.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.cbd: %ImplicitAs.type.5c7 = facet_value %i32.builtin, (%ImplicitAs.impl_witness.285) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .IntLiteral = %IntLiteral.decl
// CHECK:STDOUT:     .Int = %Int.decl
// CHECK:STDOUT:     .AddWith = %AddWith.decl
// CHECK:STDOUT:     .As = %As.decl
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
// CHECK:STDOUT:     %N.patt: %pattern_type.dc0 = value_binding_pattern N [concrete]
// CHECK:STDOUT:     %N.param_patt: %pattern_type.dc0 = value_param_pattern %N.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.98f = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.98f = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %N.param: Core.IntLiteral = value_param call_param0
// CHECK:STDOUT:     %.loc5_22.1: type = splice_block %.loc5_22.3 [concrete = Core.IntLiteral] {
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [concrete = constants.%IntLiteral]
// CHECK:STDOUT:       %IntLiteral.call: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc5_22.2: type = value_of_initializer %IntLiteral.call [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc5_22.3: type = converted %IntLiteral.call, %.loc5_22.2 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N: Core.IntLiteral = value_binding N, %N.param
// CHECK:STDOUT:     %return.param: ref type = out_param call_param1
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AddWith.decl: %AddWith.type.b56 = interface_decl @AddWith [concrete = constants.%AddWith.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc7_19.2: type = symbolic_binding T, 0 [symbolic = %T.loc7_19.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %As.decl: %As.type.3c9 = interface_decl @As [concrete = constants.%As.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc11_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc11_14.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.decl: %ImplicitAs.type.595 = interface_decl @ImplicitAs [concrete = constants.%ImplicitAs.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc15_22.2: type = symbolic_binding T, 0 [symbolic = %T.loc15_22.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @i32.builtin.as.AddWith.impl [concrete] {} {
// CHECK:STDOUT:     %int_32.loc19_6: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %Int.call.loc19_6: init type = call constants.%Int(%int_32.loc19_6) [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc19_6.1: type = value_of_initializer %Int.call.loc19_6 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc19_6.2: type = converted %Int.call.loc19_6, %.loc19_6.1 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %AddWith.ref: %AddWith.type.b56 = name_ref AddWith, file.%AddWith.decl [concrete = constants.%AddWith.generic]
// CHECK:STDOUT:     %int_32.loc19_21: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %Int.call.loc19_21: init type = call constants.%Int(%int_32.loc19_21) [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc19_24.1: type = value_of_initializer %Int.call.loc19_21 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc19_24.2: type = converted %Int.call.loc19_21, %.loc19_24.1 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %AddWith.type: type = facet_type <@AddWith, @AddWith(constants.%i32.builtin)> [concrete = constants.%AddWith.type.943]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @Core.IntLiteral.as.As.impl [concrete] {} {
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [concrete = constants.%IntLiteral]
// CHECK:STDOUT:     %IntLiteral.call: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc23_17.1: type = value_of_initializer %IntLiteral.call [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc23_17.2: type = converted %IntLiteral.call, %.loc23_17.1 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %As.ref: %As.type.3c9 = name_ref As, file.%As.decl [concrete = constants.%As.generic]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %Int.call: init type = call constants.%Int(%int_32) [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc23_28.1: type = value_of_initializer %Int.call [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc23_28.2: type = converted %Int.call, %.loc23_28.1 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %As.type: type = facet_type <@As, @As(constants.%i32.builtin)> [concrete = constants.%As.type.0c6]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @Core.IntLiteral.as.ImplicitAs.impl [concrete] {} {
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [concrete = constants.%IntLiteral]
// CHECK:STDOUT:     %IntLiteral.call: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc27_17.1: type = value_of_initializer %IntLiteral.call [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc27_17.2: type = converted %IntLiteral.call, %.loc27_17.1 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.595 = name_ref ImplicitAs, file.%ImplicitAs.decl [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %Int.call: init type = call constants.%Int(%int_32) [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc27_36.1: type = value_of_initializer %Int.call [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc27_36.2: type = converted %Int.call, %.loc27_36.1 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%i32.builtin)> [concrete = constants.%ImplicitAs.type.7cd]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @i32.builtin.as.ImplicitAs.impl [concrete] {} {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %Int.call: init type = call constants.%Int(%int_32) [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc31_6.1: type = value_of_initializer %Int.call [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc31_6.2: type = converted %Int.call, %.loc31_6.1 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.595 = name_ref ImplicitAs, file.%ImplicitAs.decl [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [concrete = constants.%IntLiteral]
// CHECK:STDOUT:     %IntLiteral.call: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc31_36.1: type = value_of_initializer %IntLiteral.call [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc31_36.2: type = converted %IntLiteral.call, %.loc31_36.1 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete = constants.%ImplicitAs.type.5c7]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @AddWith(%T.loc7_19.2: type) {
// CHECK:STDOUT:   %T.loc7_19.1: type = symbolic_binding T, 0 [symbolic = %T.loc7_19.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %AddWith.type: type = facet_type <@AddWith, @AddWith(%T.loc7_19.1)> [symbolic = %AddWith.type (constants.%AddWith.type.06e)]
// CHECK:STDOUT:   %Self.loc7_29.2: @AddWith.%AddWith.type (%AddWith.type.06e) = symbolic_binding Self, 1 [symbolic = %Self.loc7_29.2 (constants.%Self.dbc)]
// CHECK:STDOUT:   %AddWith.Op.type: type = fn_type @AddWith.Op, @AddWith(%T.loc7_19.1) [symbolic = %AddWith.Op.type (constants.%AddWith.Op.type.421)]
// CHECK:STDOUT:   %AddWith.Op: @AddWith.%AddWith.Op.type (%AddWith.Op.type.421) = struct_value () [symbolic = %AddWith.Op (constants.%AddWith.Op.59b)]
// CHECK:STDOUT:   %AddWith.assoc_type: type = assoc_entity_type @AddWith, @AddWith(%T.loc7_19.1) [symbolic = %AddWith.assoc_type (constants.%AddWith.assoc_type.5ad)]
// CHECK:STDOUT:   %assoc0.loc8_41.2: @AddWith.%AddWith.assoc_type (%AddWith.assoc_type.5ad) = assoc_entity element0, %AddWith.Op.decl [symbolic = %assoc0.loc8_41.2 (constants.%assoc0.438)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc7_29.1: @AddWith.%AddWith.type (%AddWith.type.06e) = symbolic_binding Self, 1 [symbolic = %Self.loc7_29.2 (constants.%Self.dbc)]
// CHECK:STDOUT:     %AddWith.Op.decl: @AddWith.%AddWith.Op.type (%AddWith.Op.type.421) = fn_decl @AddWith.Op [symbolic = @AddWith.%AddWith.Op (constants.%AddWith.Op.59b)] {
// CHECK:STDOUT:       %self.patt: @AddWith.Op.%pattern_type (%pattern_type.7c8) = value_binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @AddWith.Op.%pattern_type (%pattern_type.7c8) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %other.patt: @AddWith.Op.%pattern_type (%pattern_type.7c8) = value_binding_pattern other [concrete]
// CHECK:STDOUT:       %other.param_patt: @AddWith.Op.%pattern_type (%pattern_type.7c8) = value_param_pattern %other.patt, call_param1 [concrete]
// CHECK:STDOUT:       %return.patt: @AddWith.Op.%pattern_type (%pattern_type.7c8) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @AddWith.Op.%pattern_type (%pattern_type.7c8) = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc8_37.1: @AddWith.Op.%AddWith.type (%AddWith.type.06e) = specific_constant @AddWith.%Self.loc7_29.1, @AddWith(constants.%T) [symbolic = %Self (constants.%Self.dbc)]
// CHECK:STDOUT:       %Self.ref.loc8_37: @AddWith.Op.%AddWith.type (%AddWith.type.06e) = name_ref Self, %.loc8_37.1 [symbolic = %Self (constants.%Self.dbc)]
// CHECK:STDOUT:       %Self.as_type.loc8_37: type = facet_access_type %Self.ref.loc8_37 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.588)]
// CHECK:STDOUT:       %.loc8_37.2: type = converted %Self.ref.loc8_37, %Self.as_type.loc8_37 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.588)]
// CHECK:STDOUT:       %self.param: @AddWith.Op.%Self.binding.as_type (%Self.binding.as_type.588) = value_param call_param0
// CHECK:STDOUT:       %.loc8_15.1: type = splice_block %.loc8_15.3 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.588)] {
// CHECK:STDOUT:         %.loc8_15.2: @AddWith.Op.%AddWith.type (%AddWith.type.06e) = specific_constant @AddWith.%Self.loc7_29.1, @AddWith(constants.%T) [symbolic = %Self (constants.%Self.dbc)]
// CHECK:STDOUT:         %Self.ref.loc8_15: @AddWith.Op.%AddWith.type (%AddWith.type.06e) = name_ref Self, %.loc8_15.2 [symbolic = %Self (constants.%Self.dbc)]
// CHECK:STDOUT:         %Self.as_type.loc8_15: type = facet_access_type %Self.ref.loc8_15 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.588)]
// CHECK:STDOUT:         %.loc8_15.3: type = converted %Self.ref.loc8_15, %Self.as_type.loc8_15 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.588)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @AddWith.Op.%Self.binding.as_type (%Self.binding.as_type.588) = value_binding self, %self.param
// CHECK:STDOUT:       %other.param: @AddWith.Op.%Self.binding.as_type (%Self.binding.as_type.588) = value_param call_param1
// CHECK:STDOUT:       %.loc8_28.1: type = splice_block %.loc8_28.3 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.588)] {
// CHECK:STDOUT:         %.loc8_28.2: @AddWith.Op.%AddWith.type (%AddWith.type.06e) = specific_constant @AddWith.%Self.loc7_29.1, @AddWith(constants.%T) [symbolic = %Self (constants.%Self.dbc)]
// CHECK:STDOUT:         %Self.ref.loc8_28: @AddWith.Op.%AddWith.type (%AddWith.type.06e) = name_ref Self, %.loc8_28.2 [symbolic = %Self (constants.%Self.dbc)]
// CHECK:STDOUT:         %Self.as_type.loc8_28: type = facet_access_type %Self.ref.loc8_28 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.588)]
// CHECK:STDOUT:         %.loc8_28.3: type = converted %Self.ref.loc8_28, %Self.as_type.loc8_28 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.588)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %other: @AddWith.Op.%Self.binding.as_type (%Self.binding.as_type.588) = value_binding other, %other.param
// CHECK:STDOUT:       %return.param: ref @AddWith.Op.%Self.binding.as_type (%Self.binding.as_type.588) = out_param call_param2
// CHECK:STDOUT:       %return: ref @AddWith.Op.%Self.binding.as_type (%Self.binding.as_type.588) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc8_41.1: @AddWith.%AddWith.assoc_type (%AddWith.assoc_type.5ad) = assoc_entity element0, %AddWith.Op.decl [symbolic = %assoc0.loc8_41.2 (constants.%assoc0.438)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc7_29.1
// CHECK:STDOUT:     .Op = %assoc0.loc8_41.1
// CHECK:STDOUT:     witness = (%AddWith.Op.decl)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @As(%T.loc11_14.2: type) {
// CHECK:STDOUT:   %T.loc11_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc11_14.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%T.loc11_14.1)> [symbolic = %As.type (constants.%As.type.888)]
// CHECK:STDOUT:   %Self.loc11_24.2: @As.%As.type (%As.type.888) = symbolic_binding Self, 1 [symbolic = %Self.loc11_24.2 (constants.%Self.915)]
// CHECK:STDOUT:   %As.Convert.type: type = fn_type @As.Convert, @As(%T.loc11_14.1) [symbolic = %As.Convert.type (constants.%As.Convert.type.d47)]
// CHECK:STDOUT:   %As.Convert: @As.%As.Convert.type (%As.Convert.type.d47) = struct_value () [symbolic = %As.Convert (constants.%As.Convert.4a7)]
// CHECK:STDOUT:   %As.assoc_type: type = assoc_entity_type @As, @As(%T.loc11_14.1) [symbolic = %As.assoc_type (constants.%As.assoc_type.520)]
// CHECK:STDOUT:   %assoc0.loc12_32.2: @As.%As.assoc_type (%As.assoc_type.520) = assoc_entity element0, %As.Convert.decl [symbolic = %assoc0.loc12_32.2 (constants.%assoc0.84f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc11_24.1: @As.%As.type (%As.type.888) = symbolic_binding Self, 1 [symbolic = %Self.loc11_24.2 (constants.%Self.915)]
// CHECK:STDOUT:     %As.Convert.decl: @As.%As.Convert.type (%As.Convert.type.d47) = fn_decl @As.Convert [symbolic = @As.%As.Convert (constants.%As.Convert.4a7)] {
// CHECK:STDOUT:       %self.patt: @As.Convert.%pattern_type.loc12_14 (%pattern_type.69e) = value_binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @As.Convert.%pattern_type.loc12_14 (%pattern_type.69e) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @As.Convert.%pattern_type.loc12_28 (%pattern_type.51d) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @As.Convert.%pattern_type.loc12_28 (%pattern_type.51d) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @As.%T.loc11_14.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self.param: @As.Convert.%Self.binding.as_type (%Self.binding.as_type.f0b) = value_param call_param0
// CHECK:STDOUT:       %.loc12_20.1: type = splice_block %.loc12_20.3 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.f0b)] {
// CHECK:STDOUT:         %.loc12_20.2: @As.Convert.%As.type (%As.type.888) = specific_constant @As.%Self.loc11_24.1, @As(constants.%T) [symbolic = %Self (constants.%Self.915)]
// CHECK:STDOUT:         %Self.ref: @As.Convert.%As.type (%As.type.888) = name_ref Self, %.loc12_20.2 [symbolic = %Self (constants.%Self.915)]
// CHECK:STDOUT:         %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.f0b)]
// CHECK:STDOUT:         %.loc12_20.3: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.f0b)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @As.Convert.%Self.binding.as_type (%Self.binding.as_type.f0b) = value_binding self, %self.param
// CHECK:STDOUT:       %return.param: ref @As.Convert.%T (%T) = out_param call_param1
// CHECK:STDOUT:       %return: ref @As.Convert.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc12_32.1: @As.%As.assoc_type (%As.assoc_type.520) = assoc_entity element0, %As.Convert.decl [symbolic = %assoc0.loc12_32.2 (constants.%assoc0.84f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc11_24.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .Convert = %assoc0.loc12_32.1
// CHECK:STDOUT:     witness = (%As.Convert.decl)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(%T.loc15_22.2: type) {
// CHECK:STDOUT:   %T.loc15_22.1: type = symbolic_binding T, 0 [symbolic = %T.loc15_22.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T.loc15_22.1)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.c63)]
// CHECK:STDOUT:   %Self.loc15_32.2: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.c63) = symbolic_binding Self, 1 [symbolic = %Self.loc15_32.2 (constants.%Self.0b8)]
// CHECK:STDOUT:   %ImplicitAs.Convert.type: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%T.loc15_22.1) [symbolic = %ImplicitAs.Convert.type (constants.%ImplicitAs.Convert.type.103)]
// CHECK:STDOUT:   %ImplicitAs.Convert: @ImplicitAs.%ImplicitAs.Convert.type (%ImplicitAs.Convert.type.103) = struct_value () [symbolic = %ImplicitAs.Convert (constants.%ImplicitAs.Convert.2cc)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%T.loc15_22.1) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type.b04)]
// CHECK:STDOUT:   %assoc0.loc16_32.2: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.b04) = assoc_entity element0, %ImplicitAs.Convert.decl [symbolic = %assoc0.loc16_32.2 (constants.%assoc0.cd9)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc15_32.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.c63) = symbolic_binding Self, 1 [symbolic = %Self.loc15_32.2 (constants.%Self.0b8)]
// CHECK:STDOUT:     %ImplicitAs.Convert.decl: @ImplicitAs.%ImplicitAs.Convert.type (%ImplicitAs.Convert.type.103) = fn_decl @ImplicitAs.Convert [symbolic = @ImplicitAs.%ImplicitAs.Convert (constants.%ImplicitAs.Convert.2cc)] {
// CHECK:STDOUT:       %self.patt: @ImplicitAs.Convert.%pattern_type.loc16_14 (%pattern_type.5d1) = value_binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @ImplicitAs.Convert.%pattern_type.loc16_14 (%pattern_type.5d1) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @ImplicitAs.Convert.%pattern_type.loc16_28 (%pattern_type.51d) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @ImplicitAs.Convert.%pattern_type.loc16_28 (%pattern_type.51d) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @ImplicitAs.%T.loc15_22.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self.param: @ImplicitAs.Convert.%Self.binding.as_type (%Self.binding.as_type.c57) = value_param call_param0
// CHECK:STDOUT:       %.loc16_20.1: type = splice_block %.loc16_20.3 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.c57)] {
// CHECK:STDOUT:         %.loc16_20.2: @ImplicitAs.Convert.%ImplicitAs.type (%ImplicitAs.type.c63) = specific_constant @ImplicitAs.%Self.loc15_32.1, @ImplicitAs(constants.%T) [symbolic = %Self (constants.%Self.0b8)]
// CHECK:STDOUT:         %Self.ref: @ImplicitAs.Convert.%ImplicitAs.type (%ImplicitAs.type.c63) = name_ref Self, %.loc16_20.2 [symbolic = %Self (constants.%Self.0b8)]
// CHECK:STDOUT:         %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.c57)]
// CHECK:STDOUT:         %.loc16_20.3: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.c57)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @ImplicitAs.Convert.%Self.binding.as_type (%Self.binding.as_type.c57) = value_binding self, %self.param
// CHECK:STDOUT:       %return.param: ref @ImplicitAs.Convert.%T (%T) = out_param call_param1
// CHECK:STDOUT:       %return: ref @ImplicitAs.Convert.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc16_32.1: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.b04) = assoc_entity element0, %ImplicitAs.Convert.decl [symbolic = %assoc0.loc16_32.2 (constants.%assoc0.cd9)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc15_32.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .Convert = %assoc0.loc16_32.1
// CHECK:STDOUT:     witness = (%ImplicitAs.Convert.decl)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @i32.builtin.as.AddWith.impl: %.loc19_6.2 as %AddWith.type {
// CHECK:STDOUT:   %i32.builtin.as.AddWith.impl.Op.decl: %i32.builtin.as.AddWith.impl.Op.type = fn_decl @i32.builtin.as.AddWith.impl.Op [concrete = constants.%i32.builtin.as.AddWith.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.956 = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.956 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %other.patt: %pattern_type.956 = value_binding_pattern other [concrete]
// CHECK:STDOUT:     %other.param_patt: %pattern_type.956 = value_param_pattern %other.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.956 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.956 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc20_37: type = name_ref Self, @i32.builtin.as.AddWith.impl.%.loc19_6.2 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %self.param: %i32.builtin = value_param call_param0
// CHECK:STDOUT:     %Self.ref.loc20_15: type = name_ref Self, @i32.builtin.as.AddWith.impl.%.loc19_6.2 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %self: %i32.builtin = value_binding self, %self.param
// CHECK:STDOUT:     %other.param: %i32.builtin = value_param call_param1
// CHECK:STDOUT:     %Self.ref.loc20_28: type = name_ref Self, @i32.builtin.as.AddWith.impl.%.loc19_6.2 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %other: %i32.builtin = value_binding other, %other.param
// CHECK:STDOUT:     %return.param: ref %i32.builtin = out_param call_param2
// CHECK:STDOUT:     %return: ref %i32.builtin = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AddWith.impl_witness_table = impl_witness_table (%i32.builtin.as.AddWith.impl.Op.decl), @i32.builtin.as.AddWith.impl [concrete]
// CHECK:STDOUT:   %AddWith.impl_witness: <witness> = impl_witness %AddWith.impl_witness_table [concrete = constants.%AddWith.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %i32.builtin.as.AddWith.impl.Op.decl
// CHECK:STDOUT:   witness = %AddWith.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Core.IntLiteral.as.As.impl: %.loc23_17.2 as %As.type {
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.decl: %Core.IntLiteral.as.As.impl.Convert.type = fn_decl @Core.IntLiteral.as.As.impl.Convert [concrete = constants.%Core.IntLiteral.as.As.impl.Convert] {
// CHECK:STDOUT:     %self.patt: %pattern_type.dc0 = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.dc0 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.956 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.956 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %Int.call: init type = call constants.%Int(%int_32) [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc24_31.1: type = value_of_initializer %Int.call [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc24_31.2: type = converted %Int.call, %.loc24_31.1 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %self.param: Core.IntLiteral = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @Core.IntLiteral.as.As.impl.%.loc23_17.2 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %self: Core.IntLiteral = value_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %i32.builtin = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32.builtin = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %As.impl_witness_table = impl_witness_table (%Core.IntLiteral.as.As.impl.Convert.decl), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT:   %As.impl_witness: <witness> = impl_witness %As.impl_witness_table [concrete = constants.%As.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Convert = %Core.IntLiteral.as.As.impl.Convert.decl
// CHECK:STDOUT:   witness = %As.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Core.IntLiteral.as.ImplicitAs.impl: %.loc27_17.2 as %ImplicitAs.type {
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.decl: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = fn_decl @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert] {
// CHECK:STDOUT:     %self.patt: %pattern_type.dc0 = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.dc0 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.956 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.956 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %Int.call: init type = call constants.%Int(%int_32) [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc28_31.1: type = value_of_initializer %Int.call [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc28_31.2: type = converted %Int.call, %.loc28_31.1 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %self.param: Core.IntLiteral = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @Core.IntLiteral.as.ImplicitAs.impl.%.loc27_17.2 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %self: Core.IntLiteral = value_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %i32.builtin = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32.builtin = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%Core.IntLiteral.as.ImplicitAs.impl.Convert.decl), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness %ImplicitAs.impl_witness_table [concrete = constants.%ImplicitAs.impl_witness.d89]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Convert = %Core.IntLiteral.as.ImplicitAs.impl.Convert.decl
// CHECK:STDOUT:   witness = %ImplicitAs.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @i32.builtin.as.ImplicitAs.impl: %.loc31_6.2 as %ImplicitAs.type {
// CHECK:STDOUT:   %i32.builtin.as.ImplicitAs.impl.Convert.decl: %i32.builtin.as.ImplicitAs.impl.Convert.type = fn_decl @i32.builtin.as.ImplicitAs.impl.Convert [concrete = constants.%i32.builtin.as.ImplicitAs.impl.Convert] {
// CHECK:STDOUT:     %self.patt: %pattern_type.956 = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.956 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.dc0 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.dc0 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [concrete = constants.%IntLiteral]
// CHECK:STDOUT:     %IntLiteral.call: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc32_42.1: type = value_of_initializer %IntLiteral.call [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc32_42.2: type = converted %IntLiteral.call, %.loc32_42.1 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %self.param: %i32.builtin = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @i32.builtin.as.ImplicitAs.impl.%.loc31_6.2 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %self: %i32.builtin = value_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref Core.IntLiteral = out_param call_param1
// CHECK:STDOUT:     %return: ref Core.IntLiteral = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%i32.builtin.as.ImplicitAs.impl.Convert.decl), @i32.builtin.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness %ImplicitAs.impl_witness_table [concrete = constants.%ImplicitAs.impl_witness.285]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .IntLiteral = <poisoned>
// CHECK:STDOUT:   .Convert = %i32.builtin.as.ImplicitAs.impl.Convert.decl
// CHECK:STDOUT:   witness = %ImplicitAs.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @IntLiteral() -> type = "int_literal.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int(%N.param: Core.IntLiteral) -> type = "int.make_type_signed";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AddWith.Op(@AddWith.%T.loc7_19.2: type, @AddWith.%Self.loc7_29.1: @AddWith.%AddWith.type (%AddWith.type.06e)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %AddWith.type: type = facet_type <@AddWith, @AddWith(%T)> [symbolic = %AddWith.type (constants.%AddWith.type.06e)]
// CHECK:STDOUT:   %Self: @AddWith.Op.%AddWith.type (%AddWith.type.06e) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.dbc)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.588)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.7c8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @AddWith.Op.%Self.binding.as_type (%Self.binding.as_type.588), %other.param: @AddWith.Op.%Self.binding.as_type (%Self.binding.as_type.588)) -> @AddWith.Op.%Self.binding.as_type (%Self.binding.as_type.588);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @As.Convert(@As.%T.loc11_14.2: type, @As.%Self.loc11_24.1: @As.%As.type (%As.type.888)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%T)> [symbolic = %As.type (constants.%As.type.888)]
// CHECK:STDOUT:   %Self: @As.Convert.%As.type (%As.type.888) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.915)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.f0b)]
// CHECK:STDOUT:   %pattern_type.loc12_14: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type.loc12_14 (constants.%pattern_type.69e)]
// CHECK:STDOUT:   %pattern_type.loc12_28: type = pattern_type %T [symbolic = %pattern_type.loc12_28 (constants.%pattern_type.51d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @As.Convert.%Self.binding.as_type (%Self.binding.as_type.f0b)) -> @As.Convert.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ImplicitAs.Convert(@ImplicitAs.%T.loc15_22.2: type, @ImplicitAs.%Self.loc15_32.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.c63)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.c63)]
// CHECK:STDOUT:   %Self: @ImplicitAs.Convert.%ImplicitAs.type (%ImplicitAs.type.c63) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.0b8)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.c57)]
// CHECK:STDOUT:   %pattern_type.loc16_14: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type.loc16_14 (constants.%pattern_type.5d1)]
// CHECK:STDOUT:   %pattern_type.loc16_28: type = pattern_type %T [symbolic = %pattern_type.loc16_28 (constants.%pattern_type.51d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @ImplicitAs.Convert.%Self.binding.as_type (%Self.binding.as_type.c57)) -> @ImplicitAs.Convert.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @i32.builtin.as.AddWith.impl.Op(%self.param: %i32.builtin, %other.param: %i32.builtin) -> %i32.builtin = "int.sadd";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Core.IntLiteral.as.As.impl.Convert(%self.param: Core.IntLiteral) -> %i32.builtin = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Core.IntLiteral.as.ImplicitAs.impl.Convert(%self.param: Core.IntLiteral) -> %i32.builtin = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @i32.builtin.as.ImplicitAs.impl.Convert(%self.param: %i32.builtin) -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith(constants.%T) {
// CHECK:STDOUT:   %T.loc7_19.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.Op(constants.%T, constants.%Self.dbc) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %AddWith.type => constants.%AddWith.type.06e
// CHECK:STDOUT:   %Self => constants.%Self.dbc
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.588
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7c8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%T) {
// CHECK:STDOUT:   %T.loc11_14.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As.Convert(constants.%T, constants.%Self.915) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %As.type => constants.%As.type.888
// CHECK:STDOUT:   %Self => constants.%Self.915
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.f0b
// CHECK:STDOUT:   %pattern_type.loc12_14 => constants.%pattern_type.69e
// CHECK:STDOUT:   %pattern_type.loc12_28 => constants.%pattern_type.51d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%T) {
// CHECK:STDOUT:   %T.loc15_22.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.Convert(constants.%T, constants.%Self.0b8) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.c63
// CHECK:STDOUT:   %Self => constants.%Self.0b8
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.c57
// CHECK:STDOUT:   %pattern_type.loc16_14 => constants.%pattern_type.5d1
// CHECK:STDOUT:   %pattern_type.loc16_28 => constants.%pattern_type.51d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith(constants.%i32.builtin) {
// CHECK:STDOUT:   %T.loc7_19.1 => constants.%i32.builtin
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %AddWith.type => constants.%AddWith.type.943
// CHECK:STDOUT:   %Self.loc7_29.2 => constants.%Self.53c
// CHECK:STDOUT:   %AddWith.Op.type => constants.%AddWith.Op.type.bae
// CHECK:STDOUT:   %AddWith.Op => constants.%AddWith.Op.a6f
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.97c
// CHECK:STDOUT:   %assoc0.loc8_41.2 => constants.%assoc0.61e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.Op(constants.%i32.builtin, constants.%AddWith.facet) {
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %AddWith.type => constants.%AddWith.type.943
// CHECK:STDOUT:   %Self => constants.%AddWith.facet
// CHECK:STDOUT:   %Self.binding.as_type => constants.%i32.builtin
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.956
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%i32.builtin) {
// CHECK:STDOUT:   %T.loc11_14.1 => constants.%i32.builtin
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type => constants.%As.type.0c6
// CHECK:STDOUT:   %Self.loc11_24.2 => constants.%Self.013
// CHECK:STDOUT:   %As.Convert.type => constants.%As.Convert.type.063
// CHECK:STDOUT:   %As.Convert => constants.%As.Convert.701
// CHECK:STDOUT:   %As.assoc_type => constants.%As.assoc_type.c44
// CHECK:STDOUT:   %assoc0.loc12_32.2 => constants.%assoc0.d89
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As.Convert(constants.%i32.builtin, constants.%As.facet) {
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %As.type => constants.%As.type.0c6
// CHECK:STDOUT:   %Self => constants.%As.facet
// CHECK:STDOUT:   %Self.binding.as_type => Core.IntLiteral
// CHECK:STDOUT:   %pattern_type.loc12_14 => constants.%pattern_type.dc0
// CHECK:STDOUT:   %pattern_type.loc12_28 => constants.%pattern_type.956
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%i32.builtin) {
// CHECK:STDOUT:   %T.loc15_22.1 => constants.%i32.builtin
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.7cd
// CHECK:STDOUT:   %Self.loc15_32.2 => constants.%Self.ece
// CHECK:STDOUT:   %ImplicitAs.Convert.type => constants.%ImplicitAs.Convert.type.424
// CHECK:STDOUT:   %ImplicitAs.Convert => constants.%ImplicitAs.Convert.a1f
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.c3f
// CHECK:STDOUT:   %assoc0.loc16_32.2 => constants.%assoc0.5e9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.Convert(constants.%i32.builtin, constants.%ImplicitAs.facet.b3c) {
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.7cd
// CHECK:STDOUT:   %Self => constants.%ImplicitAs.facet.b3c
// CHECK:STDOUT:   %Self.binding.as_type => Core.IntLiteral
// CHECK:STDOUT:   %pattern_type.loc16_14 => constants.%pattern_type.dc0
// CHECK:STDOUT:   %pattern_type.loc16_28 => constants.%pattern_type.956
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(Core.IntLiteral) {
// CHECK:STDOUT:   %T.loc15_22.1 => Core.IntLiteral
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.5c7
// CHECK:STDOUT:   %Self.loc15_32.2 => constants.%Self.12f
// CHECK:STDOUT:   %ImplicitAs.Convert.type => constants.%ImplicitAs.Convert.type.785
// CHECK:STDOUT:   %ImplicitAs.Convert => constants.%ImplicitAs.Convert.aec
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.793
// CHECK:STDOUT:   %assoc0.loc16_32.2 => constants.%assoc0.49f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.Convert(Core.IntLiteral, constants.%ImplicitAs.facet.cbd) {
// CHECK:STDOUT:   %T => Core.IntLiteral
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.5c7
// CHECK:STDOUT:   %Self => constants.%ImplicitAs.facet.cbd
// CHECK:STDOUT:   %Self.binding.as_type => constants.%i32.builtin
// CHECK:STDOUT:   %pattern_type.loc16_14 => constants.%pattern_type.956
// CHECK:STDOUT:   %pattern_type.loc16_28 => constants.%pattern_type.dc0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- user.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [concrete]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %As.type.90f: type = generic_interface_type @As [concrete]
// CHECK:STDOUT:   %As.generic: %As.type.90f = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %As.type.84f: type = facet_type <@As, @As(%T)> [symbolic]
// CHECK:STDOUT:   %Self.76d: %As.type.84f = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %As.assoc_type.752: type = assoc_entity_type @As, @As(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.cc1: %As.assoc_type.752 = assoc_entity element0, imports.%Core.import_ref.77d [symbolic]
// CHECK:STDOUT:   %As.Convert.type.47d: type = fn_type @As.Convert, @As(%T) [symbolic]
// CHECK:STDOUT:   %As.Convert.099: %As.Convert.type.47d = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.6d0: type = symbolic_binding_type Self, 1, %Self.76d [symbolic]
// CHECK:STDOUT:   %pattern_type.7b2: type = pattern_type %Self.binding.as_type.6d0 [symbolic]
// CHECK:STDOUT:   %As.type.a29: type = facet_type <@As, @As(%i32.builtin)> [concrete]
// CHECK:STDOUT:   %Self.e41: %As.type.a29 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %As.Convert.type.378: type = fn_type @As.Convert, @As(%i32.builtin) [concrete]
// CHECK:STDOUT:   %As.Convert.e51: %As.Convert.type.378 = struct_value () [concrete]
// CHECK:STDOUT:   %As.assoc_type.bc2: type = assoc_entity_type @As, @As(%i32.builtin) [concrete]
// CHECK:STDOUT:   %assoc0.09a: %As.assoc_type.bc2 = assoc_entity element0, imports.%Core.import_ref.77d [concrete]
// CHECK:STDOUT:   %assoc0.5c9: %As.assoc_type.752 = assoc_entity element0, imports.%Core.import_ref.1ae [symbolic]
// CHECK:STDOUT:   %As.impl_witness: <witness> = impl_witness imports.%As.impl_witness_table [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.a29 = facet_value Core.IntLiteral, (%As.impl_witness) [concrete]
// CHECK:STDOUT:   %.c5a: type = fn_type_with_self_type %As.Convert.type.378, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type: type = fn_type @Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert: %Core.IntLiteral.as.As.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound.e07: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.f38: %i32.builtin = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound.5db: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %int_2.5a1: %i32.builtin = int_value 2 [concrete]
// CHECK:STDOUT:   %AddWith.type.e05: type = generic_interface_type @AddWith [concrete]
// CHECK:STDOUT:   %AddWith.generic: %AddWith.type.e05 = struct_value () [concrete]
// CHECK:STDOUT:   %AddWith.type.552: type = facet_type <@AddWith, @AddWith(%T)> [symbolic]
// CHECK:STDOUT:   %Self.3fe: %AddWith.type.552 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %AddWith.assoc_type.b6a: type = assoc_entity_type @AddWith, @AddWith(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.5a5: %AddWith.assoc_type.b6a = assoc_entity element0, imports.%Core.import_ref.7c35 [symbolic]
// CHECK:STDOUT:   %AddWith.Op.type.216: type = fn_type @AddWith.Op, @AddWith(%T) [symbolic]
// CHECK:STDOUT:   %AddWith.Op.b1d: %AddWith.Op.type.216 = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.964: type = symbolic_binding_type Self, 1, %Self.3fe [symbolic]
// CHECK:STDOUT:   %pattern_type.4a1: type = pattern_type %Self.binding.as_type.964 [symbolic]
// CHECK:STDOUT:   %AddWith.type.2a3: type = facet_type <@AddWith, @AddWith(%i32.builtin)> [concrete]
// CHECK:STDOUT:   %Self.2d9: %AddWith.type.2a3 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %AddWith.Op.type.0b7: type = fn_type @AddWith.Op, @AddWith(%i32.builtin) [concrete]
// CHECK:STDOUT:   %AddWith.Op.9d6: %AddWith.Op.type.0b7 = struct_value () [concrete]
// CHECK:STDOUT:   %AddWith.assoc_type.dff: type = assoc_entity_type @AddWith, @AddWith(%i32.builtin) [concrete]
// CHECK:STDOUT:   %assoc0.1cc: %AddWith.assoc_type.dff = assoc_entity element0, imports.%Core.import_ref.7c35 [concrete]
// CHECK:STDOUT:   %assoc0.7a7: %AddWith.assoc_type.b6a = assoc_entity element0, imports.%Core.import_ref.046 [symbolic]
// CHECK:STDOUT:   %AddWith.impl_witness: <witness> = impl_witness imports.%AddWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %AddWith.facet: %AddWith.type.2a3 = facet_value %i32.builtin, (%AddWith.impl_witness) [concrete]
// CHECK:STDOUT:   %.36f: type = fn_type_with_self_type %AddWith.Op.type.0b7, %AddWith.facet [concrete]
// CHECK:STDOUT:   %i32.builtin.as.AddWith.impl.Op.type: type = fn_type @i32.builtin.as.AddWith.impl.Op [concrete]
// CHECK:STDOUT:   %i32.builtin.as.AddWith.impl.Op: %i32.builtin.as.AddWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin.as.AddWith.impl.Op.bound.9a7: <bound method> = bound_method %int_1.f38, %i32.builtin.as.AddWith.impl.Op [concrete]
// CHECK:STDOUT:   %int_3.a0f: %i32.builtin = int_value 3 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.985: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic]
// CHECK:STDOUT:   %Self.02c: %ImplicitAs.type.985 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.ff3: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.ad7: %ImplicitAs.assoc_type.ff3 = assoc_entity element0, imports.%Core.import_ref.218943.1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.6d3: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%T) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.Convert.50f: %ImplicitAs.Convert.type.6d3 = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.b91: type = symbolic_binding_type Self, 1, %Self.02c [symbolic]
// CHECK:STDOUT:   %pattern_type.7b6: type = pattern_type %Self.binding.as_type.b91 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.7a9: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %Self.222: %ImplicitAs.type.7a9 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.71e: type = fn_type @ImplicitAs.Convert, @ImplicitAs(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.0e2: %ImplicitAs.Convert.type.71e = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.959: type = assoc_entity_type @ImplicitAs, @ImplicitAs(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %assoc0.fc3: %ImplicitAs.assoc_type.959 = assoc_entity element0, imports.%Core.import_ref.218943.1 [concrete]
// CHECK:STDOUT:   %assoc0.0bc: %ImplicitAs.assoc_type.ff3 = assoc_entity element0, imports.%Core.import_ref.fa2 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.873: type = facet_type <@ImplicitAs, @ImplicitAs(%i32.builtin)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.398: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%i32.builtin) [concrete]
// CHECK:STDOUT:   %assoc0.58b: %ImplicitAs.assoc_type.398 = assoc_entity element0, imports.%Core.import_ref.218943.2 [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.059: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32.builtin) [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.4d7: %ImplicitAs.Convert.type.059 = struct_value () [concrete]
// CHECK:STDOUT:   %Self.fca: %ImplicitAs.type.873 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a66: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.36c [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.728: %ImplicitAs.type.7a9 = facet_value %i32.builtin, (%ImplicitAs.impl_witness.a66) [concrete]
// CHECK:STDOUT:   %.39c: type = fn_type_with_self_type %ImplicitAs.Convert.type.71e, %ImplicitAs.facet.728 [concrete]
// CHECK:STDOUT:   %i32.builtin.as.ImplicitAs.impl.Convert.type: type = fn_type @i32.builtin.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %i32.builtin.as.ImplicitAs.impl.Convert: %i32.builtin.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_3.a0f, %i32.builtin.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_3.1ba, %i32.builtin [concrete]
// CHECK:STDOUT:   %pattern_type.9e2: type = pattern_type %array_type [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound.55c: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound.eb9: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %int_4.4f1: %i32.builtin = int_value 4 [concrete]
// CHECK:STDOUT:   %i32.builtin.as.AddWith.impl.Op.bound.81b: <bound method> = bound_method %int_3.a0f, %i32.builtin.as.AddWith.impl.Op [concrete]
// CHECK:STDOUT:   %int_7: %i32.builtin = int_value 7 [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.IntLiteral, Core.IntLiteral, %i32.builtin) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%int_3.1ba, %int_4.0c1, %int_7) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.8e7: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.2b1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.ef1: %ImplicitAs.type.873 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.8e7) [concrete]
// CHECK:STDOUT:   %.f79: type = fn_type_with_self_type %ImplicitAs.Convert.type.059, %ImplicitAs.facet.ef1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.c88: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.6b7: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%int_3.a0f, %int_4.4f1, %int_7) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .As = %Core.As
// CHECK:STDOUT:     .AddWith = %Core.AddWith
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//default, Int, loaded [concrete = constants.%Int]
// CHECK:STDOUT:   %Core.As: %As.type.90f = import_ref Core//default, As, loaded [concrete = constants.%As.generic]
// CHECK:STDOUT:   %Core.import_ref.8d7 = import_ref Core//default, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.e0b: @As.%As.assoc_type (%As.assoc_type.752) = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @As.%assoc0 (constants.%assoc0.5c9)]
// CHECK:STDOUT:   %Core.Convert.1f8 = import_ref Core//default, Convert, unloaded
// CHECK:STDOUT:   %Core.import_ref.b3bc94.1: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @As.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.77d: @As.%As.Convert.type (%As.Convert.type.47d) = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @As.%As.Convert (constants.%As.Convert.099)]
// CHECK:STDOUT:   %Core.import_ref.b3bc94.2: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @As.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.ce9: @As.%As.type (%As.type.84f) = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @As.%Self (constants.%Self.76d)]
// CHECK:STDOUT:   %Core.import_ref.1ae = import_ref Core//default, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.059: <witness> = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = constants.%As.impl_witness]
// CHECK:STDOUT:   %Core.import_ref.a86459.1: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = Core.IntLiteral]
// CHECK:STDOUT:   %Core.import_ref.b6d: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = constants.%As.type.a29]
// CHECK:STDOUT:   %Core.import_ref.43a: %Core.IntLiteral.as.As.impl.Convert.type = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:   %As.impl_witness_table = impl_witness_table (%Core.import_ref.43a), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT:   %Core.AddWith: %AddWith.type.e05 = import_ref Core//default, AddWith, loaded [concrete = constants.%AddWith.generic]
// CHECK:STDOUT:   %Core.import_ref.66d = import_ref Core//default, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.91c: @AddWith.%AddWith.assoc_type (%AddWith.assoc_type.b6a) = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @AddWith.%assoc0 (constants.%assoc0.7a7)]
// CHECK:STDOUT:   %Core.Op = import_ref Core//default, Op, unloaded
// CHECK:STDOUT:   %Core.import_ref.b3bc94.3: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @AddWith.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.7c35: @AddWith.%AddWith.Op.type (%AddWith.Op.type.216) = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @AddWith.%AddWith.Op (constants.%AddWith.Op.b1d)]
// CHECK:STDOUT:   %Core.import_ref.b3bc94.4: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @AddWith.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.cdf: @AddWith.%AddWith.type (%AddWith.type.552) = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @AddWith.%Self (constants.%Self.3fe)]
// CHECK:STDOUT:   %Core.import_ref.046 = import_ref Core//default, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.095: <witness> = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = constants.%AddWith.impl_witness]
// CHECK:STDOUT:   %Core.import_ref.748f05.2: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = constants.%i32.builtin]
// CHECK:STDOUT:   %Core.import_ref.d22: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = constants.%AddWith.type.2a3]
// CHECK:STDOUT:   %Core.import_ref.877: %i32.builtin.as.AddWith.impl.Op.type = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = constants.%i32.builtin.as.AddWith.impl.Op]
// CHECK:STDOUT:   %AddWith.impl_witness_table = impl_witness_table (%Core.import_ref.877), @i32.builtin.as.AddWith.impl [concrete]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//default, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.bd2 = import_ref Core//default, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.a80: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.ff3) = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.%assoc0 (constants.%assoc0.0bc)]
// CHECK:STDOUT:   %Core.Convert.788 = import_ref Core//default, Convert, unloaded
// CHECK:STDOUT:   %Core.import_ref.b3bc94.5: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.218943.1: @ImplicitAs.%ImplicitAs.Convert.type (%ImplicitAs.Convert.type.6d3) = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.%ImplicitAs.Convert (constants.%ImplicitAs.Convert.50f)]
// CHECK:STDOUT:   %Core.import_ref.b3bc94.6: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.b95: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.985) = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.%Self (constants.%Self.02c)]
// CHECK:STDOUT:   %Core.import_ref.fa2 = import_ref Core//default, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.4bf: <witness> = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = constants.%ImplicitAs.impl_witness.8e7]
// CHECK:STDOUT:   %Core.import_ref.218943.2: @ImplicitAs.%ImplicitAs.Convert.type (%ImplicitAs.Convert.type.6d3) = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.%ImplicitAs.Convert (constants.%ImplicitAs.Convert.50f)]
// CHECK:STDOUT:   %Core.import_ref.a86459.2: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = Core.IntLiteral]
// CHECK:STDOUT:   %Core.import_ref.328: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = constants.%ImplicitAs.type.873]
// CHECK:STDOUT:   %Core.import_ref.606: <witness> = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = constants.%ImplicitAs.impl_witness.a66]
// CHECK:STDOUT:   %Core.import_ref.748f05.3: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = constants.%i32.builtin]
// CHECK:STDOUT:   %Core.import_ref.7c36: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = constants.%ImplicitAs.type.7a9]
// CHECK:STDOUT:   %Core.import_ref.3c3: %i32.builtin.as.ImplicitAs.impl.Convert.type = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = constants.%i32.builtin.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.36c = impl_witness_table (%Core.import_ref.3c3), @i32.builtin.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.413: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.2b1 = impl_witness_table (%Core.import_ref.413), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .arr = %arr
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %arr.patt: %pattern_type.9e2 = ref_binding_pattern arr [concrete]
// CHECK:STDOUT:     %arr.var_patt: %pattern_type.9e2 = var_pattern %arr.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %arr.var: ref %array_type = var %arr.var_patt [concrete]
// CHECK:STDOUT:   %.loc4_44: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %int_32.loc4_16: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %Int.call.loc4_16: init type = call constants.%Int(%int_32.loc4_16) [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %int_32.loc4_27: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %Int.call.loc4_27: init type = call constants.%Int(%int_32.loc4_27) [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc4_27.1: type = value_of_initializer %Int.call.loc4_27 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc4_27.2: type = converted %Int.call.loc4_27, %.loc4_27.1 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %impl.elem0.loc4_24: %.c5a = impl_witness_access constants.%As.impl_witness, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc4_24: <bound method> = bound_method %int_1, %impl.elem0.loc4_24 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound.e07]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call.loc4_24: init %i32.builtin = call %bound_method.loc4_24(%int_1) [concrete = constants.%int_1.f38]
// CHECK:STDOUT:     %.loc4_24.1: %i32.builtin = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc4_24 [concrete = constants.%int_1.f38]
// CHECK:STDOUT:     %.loc4_24.2: %i32.builtin = converted %int_1, %.loc4_24.1 [concrete = constants.%int_1.f38]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %int_32.loc4_40: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %Int.call.loc4_40: init type = call constants.%Int(%int_32.loc4_40) [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc4_40.1: type = value_of_initializer %Int.call.loc4_40 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc4_40.2: type = converted %Int.call.loc4_40, %.loc4_40.1 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %impl.elem0.loc4_37: %.c5a = impl_witness_access constants.%As.impl_witness, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc4_37: <bound method> = bound_method %int_2, %impl.elem0.loc4_37 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound.5db]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call.loc4_37: init %i32.builtin = call %bound_method.loc4_37(%int_2) [concrete = constants.%int_2.5a1]
// CHECK:STDOUT:     %.loc4_37.1: %i32.builtin = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc4_37 [concrete = constants.%int_2.5a1]
// CHECK:STDOUT:     %.loc4_37.2: %i32.builtin = converted %int_2, %.loc4_37.1 [concrete = constants.%int_2.5a1]
// CHECK:STDOUT:     %impl.elem0.loc4_32.1: %.36f = impl_witness_access constants.%AddWith.impl_witness, element0 [concrete = constants.%i32.builtin.as.AddWith.impl.Op]
// CHECK:STDOUT:     %bound_method.loc4_32.1: <bound method> = bound_method %.loc4_24.2, %impl.elem0.loc4_32.1 [concrete = constants.%i32.builtin.as.AddWith.impl.Op.bound.9a7]
// CHECK:STDOUT:     %i32.builtin.as.AddWith.impl.Op.call: init %i32.builtin = call %bound_method.loc4_32.1(%.loc4_24.2, %.loc4_37.2) [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:     %.loc4_16.1: type = value_of_initializer %Int.call.loc4_16 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc4_16.2: type = converted %Int.call.loc4_16, %.loc4_16.1 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %impl.elem0.loc4_32.2: %.39c = impl_witness_access constants.%ImplicitAs.impl_witness.a66, element0 [concrete = constants.%i32.builtin.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc4_32.2: <bound method> = bound_method %i32.builtin.as.AddWith.impl.Op.call, %impl.elem0.loc4_32.2 [concrete = constants.%i32.builtin.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %.loc4_32.1: %i32.builtin = value_of_initializer %i32.builtin.as.AddWith.impl.Op.call [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:     %.loc4_32.2: %i32.builtin = converted %i32.builtin.as.AddWith.impl.Op.call, %.loc4_32.1 [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:     %i32.builtin.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.loc4_32.2(%.loc4_32.2) [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %.loc4_32.3: Core.IntLiteral = value_of_initializer %i32.builtin.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %.loc4_32.4: Core.IntLiteral = converted %i32.builtin.as.AddWith.impl.Op.call, %.loc4_32.3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %array_type: type = array_type %.loc4_32.4, %.loc4_16.2 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %arr: ref %array_type = ref_binding arr, %arr.var [concrete = %arr.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @As(imports.%Core.import_ref.b3bc94.1: type) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%T)> [symbolic = %As.type (constants.%As.type.84f)]
// CHECK:STDOUT:   %Self: @As.%As.type (%As.type.84f) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.76d)]
// CHECK:STDOUT:   %As.Convert.type: type = fn_type @As.Convert, @As(%T) [symbolic = %As.Convert.type (constants.%As.Convert.type.47d)]
// CHECK:STDOUT:   %As.Convert: @As.%As.Convert.type (%As.Convert.type.47d) = struct_value () [symbolic = %As.Convert (constants.%As.Convert.099)]
// CHECK:STDOUT:   %As.assoc_type: type = assoc_entity_type @As, @As(%T) [symbolic = %As.assoc_type (constants.%As.assoc_type.752)]
// CHECK:STDOUT:   %assoc0: @As.%As.assoc_type (%As.assoc_type.752) = assoc_entity element0, imports.%Core.import_ref.77d [symbolic = %assoc0 (constants.%assoc0.cc1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Core.import_ref.8d7
// CHECK:STDOUT:     .Convert = imports.%Core.import_ref.e0b
// CHECK:STDOUT:     witness = (imports.%Core.Convert.1f8)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @AddWith(imports.%Core.import_ref.b3bc94.3: type) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %AddWith.type: type = facet_type <@AddWith, @AddWith(%T)> [symbolic = %AddWith.type (constants.%AddWith.type.552)]
// CHECK:STDOUT:   %Self: @AddWith.%AddWith.type (%AddWith.type.552) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.3fe)]
// CHECK:STDOUT:   %AddWith.Op.type: type = fn_type @AddWith.Op, @AddWith(%T) [symbolic = %AddWith.Op.type (constants.%AddWith.Op.type.216)]
// CHECK:STDOUT:   %AddWith.Op: @AddWith.%AddWith.Op.type (%AddWith.Op.type.216) = struct_value () [symbolic = %AddWith.Op (constants.%AddWith.Op.b1d)]
// CHECK:STDOUT:   %AddWith.assoc_type: type = assoc_entity_type @AddWith, @AddWith(%T) [symbolic = %AddWith.assoc_type (constants.%AddWith.assoc_type.b6a)]
// CHECK:STDOUT:   %assoc0: @AddWith.%AddWith.assoc_type (%AddWith.assoc_type.b6a) = assoc_entity element0, imports.%Core.import_ref.7c35 [symbolic = %assoc0 (constants.%assoc0.5a5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Core.import_ref.66d
// CHECK:STDOUT:     .Op = imports.%Core.import_ref.91c
// CHECK:STDOUT:     witness = (imports.%Core.Op)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(imports.%Core.import_ref.b3bc94.5: type) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.985)]
// CHECK:STDOUT:   %Self: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.985) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.02c)]
// CHECK:STDOUT:   %ImplicitAs.Convert.type: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%T) [symbolic = %ImplicitAs.Convert.type (constants.%ImplicitAs.Convert.type.6d3)]
// CHECK:STDOUT:   %ImplicitAs.Convert: @ImplicitAs.%ImplicitAs.Convert.type (%ImplicitAs.Convert.type.6d3) = struct_value () [symbolic = %ImplicitAs.Convert (constants.%ImplicitAs.Convert.50f)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%T) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type.ff3)]
// CHECK:STDOUT:   %assoc0: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.ff3) = assoc_entity element0, imports.%Core.import_ref.218943.1 [symbolic = %assoc0 (constants.%assoc0.ad7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Core.import_ref.bd2
// CHECK:STDOUT:     .Convert = imports.%Core.import_ref.a80
// CHECK:STDOUT:     witness = (imports.%Core.Convert.788)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Core.IntLiteral.as.As.impl: imports.%Core.import_ref.a86459.1 as imports.%Core.import_ref.b6d [from "core.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Core.import_ref.059
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @i32.builtin.as.AddWith.impl: imports.%Core.import_ref.748f05.2 as imports.%Core.import_ref.d22 [from "core.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Core.import_ref.095
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Core.IntLiteral.as.ImplicitAs.impl: imports.%Core.import_ref.a86459.2 as imports.%Core.import_ref.328 [from "core.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Core.import_ref.4bf
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @i32.builtin.as.ImplicitAs.impl: imports.%Core.import_ref.748f05.3 as imports.%Core.import_ref.7c36 [from "core.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Core.import_ref.606
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int = "int.make_type_signed" [from "core.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @As.Convert(imports.%Core.import_ref.b3bc94.2: type, imports.%Core.import_ref.ce9: @As.%As.type (%As.type.84f)) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%T)> [symbolic = %As.type (constants.%As.type.84f)]
// CHECK:STDOUT:   %Self: @As.Convert.%As.type (%As.type.84f) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.76d)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.6d0)]
// CHECK:STDOUT:   %pattern_type.1: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type.1 (constants.%pattern_type.7b2)]
// CHECK:STDOUT:   %pattern_type.2: type = pattern_type %T [symbolic = %pattern_type.2 (constants.%pattern_type.51d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Core.IntLiteral.as.As.impl.Convert = "int.convert_checked" [from "core.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AddWith.Op(imports.%Core.import_ref.b3bc94.4: type, imports.%Core.import_ref.cdf: @AddWith.%AddWith.type (%AddWith.type.552)) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %AddWith.type: type = facet_type <@AddWith, @AddWith(%T)> [symbolic = %AddWith.type (constants.%AddWith.type.552)]
// CHECK:STDOUT:   %Self: @AddWith.Op.%AddWith.type (%AddWith.type.552) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.3fe)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.964)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.4a1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @i32.builtin.as.AddWith.impl.Op = "int.sadd" [from "core.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ImplicitAs.Convert(imports.%Core.import_ref.b3bc94.6: type, imports.%Core.import_ref.b95: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.985)) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.985)]
// CHECK:STDOUT:   %Self: @ImplicitAs.Convert.%ImplicitAs.type (%ImplicitAs.type.985) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.02c)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.b91)]
// CHECK:STDOUT:   %pattern_type.1: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type.1 (constants.%pattern_type.7b6)]
// CHECK:STDOUT:   %pattern_type.2: type = pattern_type %T [symbolic = %pattern_type.2 (constants.%pattern_type.51d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @i32.builtin.as.ImplicitAs.impl.Convert = "int.convert_checked" [from "core.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Core.IntLiteral.as.ImplicitAs.impl.Convert = "int.convert_checked" [from "core.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_3.loc4_49: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %int_4.loc4_52: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:   %int_3.loc4_56: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %int_32.loc4_61: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %Int.call.loc4_61: init type = call constants.%Int(%int_32.loc4_61) [concrete = constants.%i32.builtin]
// CHECK:STDOUT:   %.loc4_61.1: type = value_of_initializer %Int.call.loc4_61 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:   %.loc4_61.2: type = converted %Int.call.loc4_61, %.loc4_61.1 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:   %impl.elem0.loc4_58: %.c5a = impl_witness_access constants.%As.impl_witness, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc4_58: <bound method> = bound_method %int_3.loc4_56, %impl.elem0.loc4_58 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound.55c]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call.loc4_58: init %i32.builtin = call %bound_method.loc4_58(%int_3.loc4_56) [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:   %.loc4_58.1: %i32.builtin = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc4_58 [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:   %.loc4_58.2: %i32.builtin = converted %int_3.loc4_56, %.loc4_58.1 [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:   %int_4.loc4_69: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:   %int_32.loc4_74: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %Int.call.loc4_74: init type = call constants.%Int(%int_32.loc4_74) [concrete = constants.%i32.builtin]
// CHECK:STDOUT:   %.loc4_74.1: type = value_of_initializer %Int.call.loc4_74 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:   %.loc4_74.2: type = converted %Int.call.loc4_74, %.loc4_74.1 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:   %impl.elem0.loc4_71: %.c5a = impl_witness_access constants.%As.impl_witness, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc4_71: <bound method> = bound_method %int_4.loc4_69, %impl.elem0.loc4_71 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound.eb9]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call.loc4_71: init %i32.builtin = call %bound_method.loc4_71(%int_4.loc4_69) [concrete = constants.%int_4.4f1]
// CHECK:STDOUT:   %.loc4_71.1: %i32.builtin = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc4_71 [concrete = constants.%int_4.4f1]
// CHECK:STDOUT:   %.loc4_71.2: %i32.builtin = converted %int_4.loc4_69, %.loc4_71.1 [concrete = constants.%int_4.4f1]
// CHECK:STDOUT:   %impl.elem0.loc4_66: %.36f = impl_witness_access constants.%AddWith.impl_witness, element0 [concrete = constants.%i32.builtin.as.AddWith.impl.Op]
// CHECK:STDOUT:   %bound_method.loc4_66: <bound method> = bound_method %.loc4_58.2, %impl.elem0.loc4_66 [concrete = constants.%i32.builtin.as.AddWith.impl.Op.bound.81b]
// CHECK:STDOUT:   %i32.builtin.as.AddWith.impl.Op.call: init %i32.builtin = call %bound_method.loc4_66(%.loc4_58.2, %.loc4_71.2) [concrete = constants.%int_7]
// CHECK:STDOUT:   %.loc4_78.1: %tuple.type = tuple_literal (%int_3.loc4_49, %int_4.loc4_52, %i32.builtin.as.AddWith.impl.Op.call) [concrete = constants.%tuple]
// CHECK:STDOUT:   %impl.elem0.loc4_78.1: %.f79 = impl_witness_access constants.%ImplicitAs.impl_witness.8e7, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc4_78.1: <bound method> = bound_method %int_3.loc4_49, %impl.elem0.loc4_78.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.c88]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc4_78.1: init %i32.builtin = call %bound_method.loc4_78.1(%int_3.loc4_49) [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:   %.loc4_78.2: init %i32.builtin = converted %int_3.loc4_49, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc4_78.1 [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc4_78.3: ref %i32.builtin = array_index file.%arr.var, %int_0
// CHECK:STDOUT:   %.loc4_78.4: init %i32.builtin = initialize_from %.loc4_78.2 to %.loc4_78.3 [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:   %impl.elem0.loc4_78.2: %.f79 = impl_witness_access constants.%ImplicitAs.impl_witness.8e7, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc4_78.2: <bound method> = bound_method %int_4.loc4_52, %impl.elem0.loc4_78.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.6b7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc4_78.2: init %i32.builtin = call %bound_method.loc4_78.2(%int_4.loc4_52) [concrete = constants.%int_4.4f1]
// CHECK:STDOUT:   %.loc4_78.5: init %i32.builtin = converted %int_4.loc4_52, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc4_78.2 [concrete = constants.%int_4.4f1]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc4_78.6: ref %i32.builtin = array_index file.%arr.var, %int_1
// CHECK:STDOUT:   %.loc4_78.7: init %i32.builtin = initialize_from %.loc4_78.5 to %.loc4_78.6 [concrete = constants.%int_4.4f1]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc4_78.8: ref %i32.builtin = array_index file.%arr.var, %int_2
// CHECK:STDOUT:   %.loc4_78.9: init %i32.builtin = initialize_from %i32.builtin.as.AddWith.impl.Op.call to %.loc4_78.8 [concrete = constants.%int_7]
// CHECK:STDOUT:   %.loc4_78.10: init %array_type = array_init (%.loc4_78.4, %.loc4_78.7, %.loc4_78.9) to file.%arr.var [concrete = constants.%array]
// CHECK:STDOUT:   %.loc4_1: init %array_type = converted %.loc4_78.1, %.loc4_78.10 [concrete = constants.%array]
// CHECK:STDOUT:   assign file.%arr.var, %.loc4_1
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As.Convert(constants.%T, constants.%Self.76d) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %As.type => constants.%As.type.84f
// CHECK:STDOUT:   %Self => constants.%Self.76d
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.6d0
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.7b2
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.51d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%i32.builtin) {
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type => constants.%As.type.a29
// CHECK:STDOUT:   %Self => constants.%Self.e41
// CHECK:STDOUT:   %As.Convert.type => constants.%As.Convert.type.378
// CHECK:STDOUT:   %As.Convert => constants.%As.Convert.e51
// CHECK:STDOUT:   %As.assoc_type => constants.%As.assoc_type.bc2
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.09a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.Op(constants.%T, constants.%Self.3fe) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %AddWith.type => constants.%AddWith.type.552
// CHECK:STDOUT:   %Self => constants.%Self.3fe
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.964
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.4a1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith(constants.%i32.builtin) {
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %AddWith.type => constants.%AddWith.type.2a3
// CHECK:STDOUT:   %Self => constants.%Self.2d9
// CHECK:STDOUT:   %AddWith.Op.type => constants.%AddWith.Op.type.0b7
// CHECK:STDOUT:   %AddWith.Op => constants.%AddWith.Op.9d6
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.dff
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.1cc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.Convert(constants.%T, constants.%Self.02c) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.985
// CHECK:STDOUT:   %Self => constants.%Self.02c
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.b91
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.7b6
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.51d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(Core.IntLiteral) {
// CHECK:STDOUT:   %T => Core.IntLiteral
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.7a9
// CHECK:STDOUT:   %Self => constants.%Self.222
// CHECK:STDOUT:   %ImplicitAs.Convert.type => constants.%ImplicitAs.Convert.type.71e
// CHECK:STDOUT:   %ImplicitAs.Convert => constants.%ImplicitAs.Convert.0e2
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.959
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.fc3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%i32.builtin) {
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.873
// CHECK:STDOUT:   %Self => constants.%Self.fca
// CHECK:STDOUT:   %ImplicitAs.Convert.type => constants.%ImplicitAs.Convert.type.059
// CHECK:STDOUT:   %ImplicitAs.Convert => constants.%ImplicitAs.Convert.4d7
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.398
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.58b
// CHECK:STDOUT: }
// CHECK:STDOUT:
